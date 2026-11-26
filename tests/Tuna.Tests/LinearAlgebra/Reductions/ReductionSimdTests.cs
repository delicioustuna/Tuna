using System.Numerics;
using System.Runtime.CompilerServices;
using Tuna.LinearAlgebra;
using static Tuna.XMath;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>
/// 縮約の SIMD 版が、スカラーの定義（決定点 D4 の和の木と、要素ごとの演算を順に畳み込む最大・最小）とビット単位で一致すること。
/// 定義はこのファイルに書き直したもの（<see cref="ReferenceSum{T}"/> など）で、ライブラリの実装には頼らない。
/// float・double の連続なベクトルと、列が連続した行列（全体・部分行列）が SIMD を通り、ストライド付きのベクトルと転置ビューの行はスカラーの経路を通る。
/// 入力の NaN は1つまでにする（2つ以上のとき、どの NaN が残るかは保証しない）。
/// シグナリング NaN が最大・最小を通ると、JIT の段（Tier0 か最適化済みか）と経路しだいで静かな NaN になることがある（スカラーの T.Max だけでも起きる）。
/// そのため NaN どうしは静かな NaN のビットを立てたうえで比べ、符号と残りのペイロードは一致を求める。
/// </summary>
[Trait("LinearAlgebra", "XMath")]
public class ReductionSimdTests
{
    private const int Block = 64;

    // ブロックの境界（64・512・1024）とその前後、SIMD でまとめて足すブロックの数の境界を含む長さ
    private static readonly int[] Lengths =
    [
        .. Enumerable.Range(0, 140), 191, 192, 193, 255, 256, 257, 511, 512, 513, 767, 1023, 1024, 1025, 1088, 2047, 2048, 2049, 4096 + 5 * Block + 3, 9000, 16384 + 17,
    ];

    private static long Bits<T>(T value) where T : unmanaged =>
        Unsafe.SizeOf<T>() == sizeof(double) ? Unsafe.BitCast<T, long>(value) : Unsafe.BitCast<T, int>(value);

    private static T FromBits<T>(long bits) where T : unmanaged =>
        Unsafe.SizeOf<T>() == sizeof(double) ? Unsafe.BitCast<long, T>(bits) : Unsafe.BitCast<int, T>((int)bits);

    private static void AssertBits<T>(string name, T expected, T actual) where T : unmanaged, IFloatingPointIeee754<T>
    {
        long quiet = Unsafe.SizeOf<T>() == sizeof(double) ? 1L << 51 : 1L << 22;
        long expectedBits = T.IsNaN(expected) ? Bits(expected) | quiet : Bits(expected);
        long actualBits = T.IsNaN(actual) ? Bits(actual) | quiet : Bits(actual);
        Assert.True(expectedBits == actualBits, $"{name}: expected {expected} ({Bits(expected):X}), actual {actual} ({Bits(actual):X})");
    }

    /// <summary>符号と大きさ（2⁻²⁰〜2²⁰）を混ぜた値。足す順序が変われば末尾の桁が変わる</summary>
    private static T[] Values<T>(int length, int seed) where T : IFloatingPointIeee754<T>
    {
        var random = new Random(seed);
        var values = new T[length];
        for (int i = 0; i < length; i++)
        {
            values[i] = T.CreateTruncating((random.NextDouble() * 2 - 1) * Math.ScaleB(1, random.Next(-20, 21)));
        }

        return values;
    }

    /// <summary>1つの位置に特別な値（NaN・シグナリング NaN・±∞・-0・非正規化数・最大値）を入れた値</summary>
    private static IEnumerable<(string Name, T[] Values)> SpecialCases<T>(int length, int seed) where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
    {
        bool isDouble = Unsafe.SizeOf<T>() == sizeof(double);
        T[] specials =
        [
            FromBits<T>(isDouble ? 0x7FF8_0000_0000_0042 : 0x7FC0_0042),
            FromBits<T>(isDouble ? unchecked((long)0xFFF8_0000_0000_1234) : unchecked((int)0xFFC0_1234)),
            FromBits<T>(isDouble ? 0x7FF0_0000_0000_0001 : 0x7F80_0001),
            T.PositiveInfinity, T.NegativeInfinity, T.NegativeZero, T.Epsilon, T.MaxValue,
        ];
        var random = new Random(seed);
        foreach (T special in specials)
        {
            var values = Values<T>(length, seed);
            if (length > 0)
            {
                values[random.Next(length)] = special;
            }

            yield return ($"{special}", values);
        }
    }

    // ---------- 定義（このテストで書き直したもの） ----------

    /// <summary>決定点 D4 の和: 64 個のブロックの中は 0 から順に足し、ブロックの和は繰り上がりの順に結合する</summary>
    private static T ReferenceSum<T>(int length, Func<int, T> term) where T : IFloatingPointIeee754<T>
    {
        var partial = new T[32];
        int depth = 0;
        for (int start = 0, block = 0; start < length; block++)
        {
            int end = Math.Min(length, start + Block);
            T sum = T.Zero;
            for (int i = start; i < end; i++)
            {
                sum += term(i);
            }

            for (int k = block; (k & 1) != 0; k >>= 1)
            {
                sum = partial[--depth] + sum;
            }

            partial[depth++] = sum;
            start = end;
        }

        if (depth == 0)
        {
            return T.Zero;
        }

        T result = partial[--depth];
        while (depth > 0)
        {
            result = partial[--depth] + result;
        }

        return result;
    }

    private static T ReferenceMean<T>(T[] x) where T : IFloatingPointIeee754<T>
    {
        T count = T.CreateTruncating(x.Length);
        T mean = ReferenceSum(x.Length, i => x[i]) / count;
        if (T.IsFinite(mean) || x.Length == 0)
        {
            return mean;
        }

        int k = 65 - BitOperations.LeadingZeroCount((ulong)x.Length);
        T down = T.ScaleB(T.One, -k);
        return ReferenceSum(x.Length, i => x[i] * down) / count * T.ScaleB(T.One, k);
    }

    // 補正つき2パス法（入力は偏差の平方和が溢れない範囲に限る）
    private static T ReferenceVariance<T>(T[] x, int correction) where T : IFloatingPointIeee754<T>
    {
        if (x.Length <= correction)
        {
            return T.NaN;
        }

        T count = T.CreateTruncating(x.Length);
        T mean = ReferenceMean(x);
        T squares = ReferenceSum(x.Length, i => (x[i] - mean) * (x[i] - mean));
        T deviations = ReferenceSum(x.Length, i => x[i] - mean);
        T adjustment = deviations * deviations / count;
        if (T.IsPositiveInfinity(adjustment))
        {
            adjustment = deviations / count * deviations;
        }

        T sum = squares - adjustment;
        return (sum < T.Zero ? T.Zero : sum) / T.CreateTruncating(x.Length - correction);
    }

    private static T Fold<T>(T[] x, T initial, Func<T, T, T> step)
    {
        T result = initial;
        foreach (T value in x)
        {
            result = step(result, value);
        }

        return result;
    }

    private static T ReferenceMax<T>(T[] x) where T : IFloatingPointIeee754<T> => Fold(x[1..], x[0], T.Max);

    private static T ReferenceMin<T>(T[] x) where T : IFloatingPointIeee754<T> => Fold(x[1..], x[0], T.Min);

    // 最大絶対値の指数で 2 の冪に縮めてから平方和をとる（要素ごとに ScaleB）
    private static T ReferenceNorm<T>(T[] x) where T : IFloatingPointIeee754<T>
    {
        T largest = Fold(x, T.Zero, (a, v) => T.IsNaN(T.Abs(v)) || T.Abs(v) > a ? T.Abs(v) : a);
        if (!T.IsFinite(largest) || largest == T.Zero)
        {
            return largest;
        }

        int exponent = T.ILogB(largest);
        T sum = ReferenceSum(x.Length, i =>
        {
            T s = T.ScaleB(x[i], -exponent);
            return s * s;
        });
        return T.ScaleB(T.Sqrt(sum), exponent);
    }

    /// <summary>
    /// 指数の項 <c>exp(x[i] - shift)</c>。縮約の項は要素ごとの Exp（float・double は SIMD 版の指数関数で、Math.Exp と末尾の桁が違いうる）と同じ値なので、
    /// 引き算だけをここで行い、指数は要素ごとの <see cref="Exp{T}(ReadOnlyVectorSpan{T})"/> で求める（その精度と配置によらないことは ElementwiseSimdTests が確かめる）
    /// </summary>
    private static T[] ShiftedExp<T>(T[] x, T shift) where T : unmanaged, IFloatingPointIeee754<T> =>
        Exp(new Tuna.LinearAlgebra.Vector<T>(x.Select(v => v - shift).ToArray())).ToArray();

    // ln Σ exp(x - max) = ln(1 + ((c - 1) + s))。c は最大値に等しい要素の個数、s はそれ以外の項の和（最大値の項は 0 として足す）。
    // ln(1 + ·) は double の log1p（XMath.Log1p）で求める
    private static T ReferenceLogSumOfShiftedExp<T>(T[] x, T max) where T : unmanaged, IFloatingPointIeee754<T>
    {
        int count = x.Count(v => v == max);
        var e = ShiftedExp(x, max);
        T rest = ReferenceSum(x.Length, i => x[i] == max ? T.Zero : e[i]);
        return T.CreateTruncating(Log1p((count - 1) + double.CreateTruncating(rest)));
    }

    private static T ReferenceLogSumExp<T>(T[] x) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (x.Length == 0)
        {
            return T.NegativeInfinity;
        }

        T max = ReferenceMax(x);
        return T.IsFinite(max) ? max + ReferenceLogSumOfShiftedExp(x, max) : max;
    }

    // 対数ソフトマックス (x - max) - ln Σ exp(x - max)。最大値が ±∞ か NaN なら x - max
    private static T[] ReferenceLogSoftmax<T>(T[] x) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (x.Length == 0)
        {
            return [];
        }

        T max = ReferenceMax(x);
        T logSum = T.IsFinite(max) ? ReferenceLogSumOfShiftedExp(x, max) : T.Zero;
        return x.Select(v => (v - max) - logSum).ToArray();
    }

    private static T[] ReferenceSoftmax<T>(T[] x) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (x.Length == 0)
        {
            return [];
        }

        T max = ReferenceMax(x);
        var e = ShiftedExp(x, max);
        T sum = ReferenceSum(e.Length, i => e[i]);
        return e.Select(v => v / sum).ToArray();
    }

    // ---------- ベクトル ----------

    /// <summary>連続なベクトル（SIMD）と、行列の行として取り出したストライド付きのベクトル（スカラーの経路）</summary>
    private static void ForEachLayout<T>(T[] values, Action<string, ReadOnlyVectorSpan<T>> check) where T : unmanaged, IFloatingPointIeee754<T>
    {
        check("contiguous", new Tuna.LinearAlgebra.Vector<T>(values));
        var strided = new Matrix<T>(3, values.Length);
        for (int i = 0; i < values.Length; i++)
        {
            strided[1, i] = values[i];
        }

        check("strided", strided.Row(1));
    }

    private static void CheckVectorReductions<T>() where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
    {
        foreach (int n in Lengths)
        {
            var cases = new List<(string Name, T[] Values)> { ("values", Values<T>(n, n)) };
            cases.AddRange(SpecialCases<T>(n, n + 1));
            var y = Values<T>(n, n + 2);
            foreach (var (name, x) in cases)
            {
                ForEachLayout(x, (layout, v) =>
                {
                    string label = $"{typeof(T).Name} n={n} {name} {layout}";
                    AssertBits($"Sum {label}", ReferenceSum(n, i => x[i]), Sum(v));
                    AssertBits($"Mean {label}", ReferenceMean(x), Mean(v));
                    AssertBits($"Dot {label}", ReferenceSum(n, i => x[i] * y[i]), Dot(v, new Tuna.LinearAlgebra.Vector<T>(y)));
                    AssertBits($"Norm1 {label}", ReferenceSum(n, i => T.Abs(x[i])), Norm1(v));
                    AssertBits($"NormInfinity {label}", Fold(x, T.Zero, (a, e) => T.Max(a, T.Abs(e))), NormInfinity(v));
                    AssertBits($"Norm {label}", ReferenceNorm(x), Norm(v));
                    AssertBits($"LogSumExp {label}", ReferenceLogSumExp(x), LogSumExp(v));
                    if (n > 0)
                    {
                        AssertBits($"Max {label}", ReferenceMax(x), Max(v));
                        AssertBits($"Min {label}", ReferenceMin(x), Min(v));
                    }

                    if (name == "values")
                    {
                        AssertBits($"Variance {label}", ReferenceVariance(x, 1), Variance(v, 1));
                    }
                });
            }
        }
    }

    [Fact]
    public void VectorReductionsMatchTheDefinitionBitwiseForDouble() => CheckVectorReductions<double>();

    [Fact]
    public void VectorReductionsMatchTheDefinitionBitwiseForSingle() => CheckVectorReductions<float>();

    /// <summary>平均の和が溢れる場合（縮尺して足し直す）、ノルムの縮尺の 2 の冪が正規化数でない場合（ScaleB のまま）</summary>
    [Fact]
    public void ScaledPathsMatchTheDefinitionBitwise()
    {
        foreach (int n in new[] { 3, 70, 600, 1500 })
        {
            var huge = Values<double>(n, n).Select(v => v * 1e300).ToArray();
            var tiny = Values<double>(n, n + 1).Select(v => v * 1e-310).ToArray();
            var top = Values<double>(n, n + 2).Select(v => Math.CopySign(double.MaxValue, v) * Math.Abs(v) / 1_048_576).ToArray();
            top[0] = double.MaxValue;
            foreach (var (name, x) in new[] { ("huge", huge), ("tiny", tiny), ("top", top) })
            {
                ForEachLayout(x, (layout, v) =>
                {
                    AssertBits($"Mean {name} n={n} {layout}", ReferenceMean(x), Mean(v));
                    AssertBits($"Norm {name} n={n} {layout}", ReferenceNorm(x), Norm(v));
                });
            }
        }
    }

    // ---------- 行列 ----------

    private static readonly (int Rows, int Columns)[] Shapes =
    [
        (1, 1), (1, 200), (200, 1), (7, 130), (65, 70), (130, 7), (300, 150), (8, 1000), (1000, 8), (513, 3), (3, 513), (600, 600),
    ];

    /// <summary>全体（列が連続）、部分行列（列が連続でストライドあり）、転置ビュー（行が連続。Row* はスカラーの経路）</summary>
    private static IEnumerable<(string Name, Func<MatrixSpan<T>> View)> MatrixLayouts<T>(int rows, int columns, int seed)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var values = Values<T>(rows * columns, seed);
        var dense = Matrix<T>.FromColumnMajor(rows, columns, values);
        yield return ("dense", () => dense.AsMatrixSpan());

        var big = Matrix<T>.FromColumnMajor(rows + 3, columns + 2, Values<T>((rows + 3) * (columns + 2), seed + 1));
        yield return ("submatrix", () => big[1..(rows + 1), 2..(columns + 2)]);

        var stored = Matrix<T>.FromColumnMajor(columns, rows, Values<T>(rows * columns, seed + 2));
        yield return ("transposed", () => stored.T);
    }

    private static T[] ColumnOf<T>(ReadOnlyMatrixSpan<T> a, int j) where T : unmanaged, INumberBase<T>
    {
        var column = new T[a.Rows];
        for (int i = 0; i < column.Length; i++)
        {
            column[i] = a[i, j];
        }

        return column;
    }

    private static T[] RowOf<T>(ReadOnlyMatrixSpan<T> a, int i) where T : unmanaged, INumberBase<T>
    {
        var row = new T[a.Columns];
        for (int j = 0; j < row.Length; j++)
        {
            row[j] = a[i, j];
        }

        return row;
    }

    private static void CheckMatrixReductions<T>() where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
    {
        foreach (var (rows, columns) in Shapes)
        {
            foreach (var (layout, getView) in MatrixLayouts<T>(rows, columns, rows * 1000 + columns))
            {
                ReadOnlyMatrixSpan<T> a = getView();
                string label = $"{typeof(T).Name} {rows}x{columns} {layout}";
                var columnSums = ColumnSums(a);
                var columnMeans = ColumnMeans(a);
                var columnVariances = ColumnVariances(a, 1);
                var columnNorms = ColumnNorms(a);
                var columnMax = ColumnMax(a);
                var columnMin = ColumnMin(a);
                var columnLogSumExp = ColumnLogSumExp(a);
                var columnSoftmax = ColumnSoftmax(a);
                var columnLogSoftmax = ColumnLogSoftmax(a);
                for (int j = 0; j < columns; j++)
                {
                    var c = ColumnOf(a, j);
                    AssertBits($"ColumnSums {label} [{j}]", ReferenceSum(c.Length, i => c[i]), columnSums[j]);
                    AssertBits($"ColumnMeans {label} [{j}]", ReferenceMean(c), columnMeans[j]);
                    AssertBits($"ColumnVariances {label} [{j}]", ReferenceVariance(c, 1), columnVariances[j]);
                    AssertBits($"ColumnNorms {label} [{j}]", ReferenceNorm(c), columnNorms[j]);
                    AssertBits($"ColumnMax {label} [{j}]", ReferenceMax(c), columnMax[j]);
                    AssertBits($"ColumnMin {label} [{j}]", ReferenceMin(c), columnMin[j]);
                    AssertBits($"ColumnLogSumExp {label} [{j}]", ReferenceLogSumExp(c), columnLogSumExp[j]);
                    var softmax = ReferenceSoftmax(c);
                    var logSoftmax = ReferenceLogSoftmax(c);
                    for (int i = 0; i < rows; i++)
                    {
                        AssertBits($"ColumnSoftmax {label} [{i}, {j}]", softmax[i], columnSoftmax[i, j]);
                        AssertBits($"ColumnLogSoftmax {label} [{i}, {j}]", logSoftmax[i], columnLogSoftmax[i, j]);
                    }
                }

                var rowSums = RowSums(a);
                var rowMeans = RowMeans(a);
                var rowVariances = RowVariances(a, 1);
                var rowNorms = RowNorms(a);
                var rowMax = RowMax(a);
                var rowMin = RowMin(a);
                var rowLogSumExp = RowLogSumExp(a);
                var rowSoftmax = RowSoftmax(a);
                var rowLogSoftmax = RowLogSoftmax(a);
                for (int i = 0; i < rows; i++)
                {
                    var r = RowOf(a, i);
                    AssertBits($"RowSums {label} [{i}]", ReferenceSum(r.Length, j => r[j]), rowSums[i]);
                    AssertBits($"RowMeans {label} [{i}]", ReferenceMean(r), rowMeans[i]);
                    AssertBits($"RowVariances {label} [{i}]", ReferenceVariance(r, 1), rowVariances[i]);
                    AssertBits($"RowNorms {label} [{i}]", ReferenceNorm(r), rowNorms[i]);
                    AssertBits($"RowMax {label} [{i}]", ReferenceMax(r), rowMax[i]);
                    AssertBits($"RowMin {label} [{i}]", ReferenceMin(r), rowMin[i]);
                    AssertBits($"RowLogSumExp {label} [{i}]", ReferenceLogSumExp(r), rowLogSumExp[i]);
                    var softmax = ReferenceSoftmax(r);
                    var logSoftmax = ReferenceLogSoftmax(r);
                    for (int j = 0; j < columns; j++)
                    {
                        AssertBits($"RowSoftmax {label} [{i}, {j}]", softmax[j], rowSoftmax[i, j]);
                        AssertBits($"RowLogSoftmax {label} [{i}, {j}]", logSoftmax[j], rowLogSoftmax[i, j]);
                    }
                }
            }
        }
    }

    [Fact]
    public void MatrixReductionsMatchTheDefinitionBitwiseForDouble() => CheckMatrixReductions<double>();

    [Fact]
    public void MatrixReductionsMatchTheDefinitionBitwiseForSingle() => CheckMatrixReductions<float>();

    /// <summary>その場更新のソフトマックス（入力と出力が同じ領域）も同じビットになる</summary>
    [Fact]
    public void InPlaceSoftmaxMatchesTheDefinitionBitwise()
    {
        foreach (var (rows, columns) in Shapes)
        {
            foreach (var (layout, getView) in MatrixLayouts<double>(rows, columns, rows + columns))
            {
                MatrixSpan<double> view = getView();
                ReadOnlyMatrixSpan<double> a = view;
                var expectedColumns = ColumnSoftmax(a);
                var expectedRows = RowSoftmax(a);
                var expectedLogColumns = ColumnLogSoftmax(a);
                var expectedLogRows = RowLogSoftmax(a);
                var original = a.ToMatrix();
                foreach (var (name, expected, apply) in new (string, Matrix<double>, Action<MatrixSpan<double>>)[]
                {
                ("ColumnSoftmax", expectedColumns, x => InPlace.ColumnSoftmax(x)),
                ("RowSoftmax", expectedRows, x => InPlace.RowSoftmax(x)),
                ("ColumnLogSoftmax", expectedLogColumns, x => InPlace.ColumnLogSoftmax(x)),
                ("RowLogSoftmax", expectedLogRows, x => InPlace.RowLogSoftmax(x)),
                })
                {
                    original.AsMatrixSpan().CopyTo(view);
                    apply(view);
                    for (int j = 0; j < columns; j++)
                    {
                        for (int i = 0; i < rows; i++)
                        {
                            AssertBits($"{name} {rows}x{columns} {layout} [{i}, {j}]", expected[i, j], view[i, j]);
                        }
                    }
                }
            }
        }
    }

    // ---------- 最大値と等しい要素が多い入力 ----------

    /// <summary>
    /// 最大値と等しい要素が多い値。kind 0 は -2〜2 の整数、1 は最大値が +0 と -0 の混在、2 は −∞ を含む（全部 −∞ の行・列もある）。
    /// LogSumExp・LogSoftmax の和は最大値に等しい要素の個数を別に数える。連続な float・double は指数の項を作る走査の中で数え、
    /// それ以外は要素を比べて数えるので、同点が多い入力で両方の経路が定義と一致することを確かめる
    /// </summary>
    private static T[] TiedValues<T>(int length, int seed, int kind) where T : IFloatingPointIeee754<T>
    {
        var random = new Random(seed);
        var values = new T[length];
        for (int i = 0; i < length; i++)
        {
            values[i] = kind switch
            {
                0 => T.CreateTruncating(random.Next(-2, 3)),
                1 => random.Next(3) switch { 0 => T.Zero, 1 => T.NegativeZero, _ => T.CreateTruncating(-random.Next(1, 4)) },
                _ => random.Next(4) == 0 ? T.NegativeInfinity : T.CreateTruncating(random.Next(-3, 2) * 0.5),
            };
        }

        return values;
    }

    private static void CheckTiedLogSumExp<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        for (int kind = 0; kind < 3; kind++)
        {
            foreach (int n in Lengths)
            {
                var x = TiedValues<T>(n, n * 3 + kind, kind);
                ForEachLayout(x, (layout, v) =>
                {
                    string label = $"{typeof(T).Name} n={n} kind={kind} {layout}";
                    AssertBits($"LogSumExp {label}", ReferenceLogSumExp(x), LogSumExp(v));
                    var expected = ReferenceLogSoftmax(x);
                    var actual = LogSoftmax(v);
                    for (int i = 0; i < n; i++)
                    {
                        AssertBits($"LogSoftmax {label} [{i}]", expected[i], actual[i]);
                    }
                });
            }

            foreach (var (rows, columns) in Shapes)
            {
                int seed = rows * 1000 + columns + kind;
                var dense = Matrix<T>.FromColumnMajor(rows, columns, TiedValues<T>(rows * columns, seed, kind));
                var big = Matrix<T>.FromColumnMajor(rows + 3, columns + 2, TiedValues<T>((rows + 3) * (columns + 2), seed + 1, kind));
                var stored = Matrix<T>.FromColumnMajor(columns, rows, TiedValues<T>(rows * columns, seed + 2, kind));
                for (int k = 0; k < 3; k++)
                {
                    string layout = k == 0 ? "dense" : k == 1 ? "submatrix" : "transposed";
                    ReadOnlyMatrixSpan<T> a = k == 0 ? dense : k == 1 ? big[1..(rows + 1), 2..(columns + 2)] : stored.T;
                    string label = $"{typeof(T).Name} {rows}x{columns} kind={kind} {layout}";
                    var columnLogSumExp = ColumnLogSumExp(a);
                    var columnLogSoftmax = ColumnLogSoftmax(a);
                    for (int j = 0; j < columns; j++)
                    {
                        var c = ColumnOf(a, j);
                        AssertBits($"ColumnLogSumExp {label} [{j}]", ReferenceLogSumExp(c), columnLogSumExp[j]);
                        var logSoftmax = ReferenceLogSoftmax(c);
                        for (int i = 0; i < rows; i++)
                        {
                            AssertBits($"ColumnLogSoftmax {label} [{i}, {j}]", logSoftmax[i], columnLogSoftmax[i, j]);
                        }
                    }

                    var rowLogSumExp = RowLogSumExp(a);
                    var rowLogSoftmax = RowLogSoftmax(a);
                    for (int i = 0; i < rows; i++)
                    {
                        var r = RowOf(a, i);
                        AssertBits($"RowLogSumExp {label} [{i}]", ReferenceLogSumExp(r), rowLogSumExp[i]);
                        var logSoftmax = ReferenceLogSoftmax(r);
                        for (int j = 0; j < columns; j++)
                        {
                            AssertBits($"RowLogSoftmax {label} [{i}, {j}]", logSoftmax[j], rowLogSoftmax[i, j]);
                        }
                    }
                }
            }
        }
    }

    [Fact]
    public void TiedMaximaMatchTheDefinitionBitwiseForDouble() => CheckTiedLogSumExp<double>();

    [Fact]
    public void TiedMaximaMatchTheDefinitionBitwiseForSingle() => CheckTiedLogSumExp<float>();

    // ---------- 平方和・標準偏差・最大と最小の位置 ----------

    // 値が大きいほうを真とする順序（+0 は -0 より大きい。NaN は呼び出し側で扱う）
    private static bool Greater<T>(T x, T y) where T : INumber<T> =>
        x > y || (x == y && T.IsZero(x) && !T.IsNegative(x) && T.IsNegative(y));

    /// <summary>最大（最小）の最初の位置。NaN があれば最初の NaN の位置（1回の走査で書いた定義）</summary>
    private static int ReferenceIndexOf<T>(T[] x, bool maximum) where T : INumber<T>
    {
        int best = 0;
        if (T.IsNaN(x[0]))
        {
            return 0;
        }

        for (int i = 1; i < x.Length; i++)
        {
            if (T.IsNaN(x[i]))
            {
                return i;
            }

            if (maximum ? Greater(x[i], x[best]) : Greater(x[best], x[i]))
            {
                best = i;
            }
        }

        return best;
    }

    /// <summary>同じ値（±0 を含む）が何度も現れる値</summary>
    private static T[] Ties<T>(int length, int seed) where T : IFloatingPointIeee754<T>
    {
        var random = new Random(seed);
        T[] pool = [-T.One, T.NegativeZero, T.Zero, T.One, T.CreateTruncating(0.5)];
        var values = new T[length];
        for (int i = 0; i < length; i++)
        {
            values[i] = pool[random.Next(pool.Length)];
        }

        return values;
    }

    private static void CheckSummaryStatistics<T>() where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
    {
        foreach (int n in Lengths)
        {
            var cases = new List<(string Name, T[] Values)> { ("values", Values<T>(n, n)), ("ties", Ties<T>(n, n + 3)) };
            cases.AddRange(SpecialCases<T>(n, n + 1));
            foreach (var (name, x) in cases)
            {
                ForEachLayout(x, (layout, v) =>
                {
                    string label = $"{typeof(T).Name} n={n} {name} {layout}";
                    AssertBits($"SumOfSquares {label}", ReferenceSum(n, i => x[i] * x[i]), SumOfSquares(v));
                    if (name is "values" or "ties")
                    {
                        AssertBits($"StandardDeviation {label}", T.Sqrt(ReferenceVariance(x, 1)), StandardDeviation(v, 1));
                    }

                    if (n > 0)
                    {
                        Assert.True(ReferenceIndexOf(x, maximum: true) == IndexOfMax(v), $"IndexOfMax {label}");
                        Assert.True(ReferenceIndexOf(x, maximum: false) == IndexOfMin(v), $"IndexOfMin {label}");
                    }
                });
            }
        }
    }

    [Fact]
    public void SummaryStatisticsMatchTheDefinitionForDouble() => CheckSummaryStatistics<double>();

    [Fact]
    public void SummaryStatisticsMatchTheDefinitionForSingle() => CheckSummaryStatistics<float>();

    private static void CheckMatrixSummaryStatistics<T>() where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
    {
        foreach (var (rows, columns) in Shapes)
        {
            foreach (var (layout, getView) in MatrixLayouts<T>(rows, columns, rows * 1000 + columns))
            {
                foreach (bool ties in new[] { false, true })
                {
                    MatrixSpan<T> view = getView();
                    if (ties)
                    {
                        var values = Ties<T>(rows * columns, rows + columns);
                        for (int j = 0; j < columns; j++)
                        {
                            for (int i = 0; i < rows; i++)
                            {
                                view[i, j] = values[j * rows + i];
                            }
                        }

                        view[rows / 2, columns / 2] = T.NaN;
                    }

                    ReadOnlyMatrixSpan<T> a = view;
                    string label = $"{typeof(T).Name} {rows}x{columns} {layout}{(ties ? " ties" : "")}";
                    var columnSquares = ColumnSumsOfSquares(a);
                    var columnDeviations = ColumnStandardDeviations(a, 1);
                    var columnMax = ColumnIndexOfMax(a);
                    var columnMin = ColumnIndexOfMin(a);
                    for (int j = 0; j < columns; j++)
                    {
                        var c = ColumnOf(a, j);
                        AssertBits($"ColumnSumsOfSquares {label} [{j}]", ReferenceSum(c.Length, i => c[i] * c[i]), columnSquares[j]);
                        AssertBits($"ColumnStandardDeviations {label} [{j}]", T.Sqrt(ReferenceVariance(c, 1)), columnDeviations[j]);
                        Assert.True(ReferenceIndexOf(c, maximum: true) == columnMax[j], $"ColumnIndexOfMax {label} [{j}]");
                        Assert.True(ReferenceIndexOf(c, maximum: false) == columnMin[j], $"ColumnIndexOfMin {label} [{j}]");
                    }

                    var rowSquares = RowSumsOfSquares(a);
                    var rowDeviations = RowStandardDeviations(a, 1);
                    var rowMax = RowIndexOfMax(a);
                    var rowMin = RowIndexOfMin(a);
                    for (int i = 0; i < rows; i++)
                    {
                        var r = RowOf(a, i);
                        AssertBits($"RowSumsOfSquares {label} [{i}]", ReferenceSum(r.Length, j => r[j] * r[j]), rowSquares[i]);
                        AssertBits($"RowStandardDeviations {label} [{i}]", T.Sqrt(ReferenceVariance(r, 1)), rowDeviations[i]);
                        Assert.True(ReferenceIndexOf(r, maximum: true) == rowMax[i], $"RowIndexOfMax {label} [{i}]");
                        Assert.True(ReferenceIndexOf(r, maximum: false) == rowMin[i], $"RowIndexOfMin {label} [{i}]");
                    }
                }
            }
        }
    }

    [Fact]
    public void MatrixSummaryStatisticsMatchTheDefinitionForDouble() => CheckMatrixSummaryStatistics<double>();

    [Fact]
    public void MatrixSummaryStatisticsMatchTheDefinitionForSingle() => CheckMatrixSummaryStatistics<float>();

    /// <summary>整数（スカラーの経路）、コンテナーを渡す版、空の入力</summary>
    [Fact]
    public void IndexOfExtremaHandleIntegersContainersAndEmptyInputs()
    {
        var random = new Random(5);
        var values = Enumerable.Range(0, 300).Select(_ => random.Next(-5, 6)).ToArray();
        var vector = new Tuna.LinearAlgebra.Vector<int>(values);
        Assert.Equal(ReferenceIndexOf(values, maximum: true), IndexOfMax(vector));
        Assert.Equal(ReferenceIndexOf(values, maximum: false), IndexOfMin(vector));
        Assert.Equal(values.Sum(v => v * v), SumOfSquares(vector));

        var matrix = Matrix<int>.FromColumnMajor(20, 15, values[..300]);
        var columnMax = ColumnIndexOfMax(matrix);
        var rowMin = new int[20];
        RowIndexOfMin(matrix, rowMin);
        for (int j = 0; j < 15; j++)
        {
            Assert.Equal(ReferenceIndexOf(values[(j * 20)..((j + 1) * 20)], maximum: true), columnMax[j]);
        }

        for (int i = 0; i < 20; i++)
        {
            Assert.Equal(ReferenceIndexOf(Enumerable.Range(0, 15).Select(j => values[j * 20 + i]).ToArray(), maximum: false), rowMin[i]);
        }

        Assert.Throws<ArgumentException>(() => IndexOfMax(new Tuna.LinearAlgebra.Vector<double>(0)));
        Assert.Throws<ArgumentException>(() => IndexOfMax(new Matrix<double>(0, 3)));
        Assert.Throws<ArgumentException>(() => IndexOfMin(new Matrix<double>(3, 0)));
        Assert.Throws<ArgumentException>(() => ColumnIndexOfMax(new Matrix<double>(0, 3)));
        Assert.Throws<ArgumentException>(() => RowIndexOfMin(new Matrix<double>(3, 0)));
        Assert.Empty(RowIndexOfMax(new Matrix<double>(0, 3)));
        Assert.Empty(ColumnIndexOfMin(new Matrix<double>(3, 0)));
        Assert.Throws<ArgumentException>(() => ColumnIndexOfMax(new Matrix<double>(2, 3), new int[2]));
    }

    [Fact]
    public void MatrixIndexOfExtremaReturnsCoordinatesInTheInputView()
    {
        var matrix = Matrix<int>.FromRowMajor(3, 3, [1, 9, -7, 9, 3, 4, 6, -7, 2]);
        var max = IndexOfMax(matrix);
        Assert.Equal(1, max.Row);
        Assert.Equal(0, max.Column);
        Assert.Equal((2, 1), IndexOfMin(matrix));

        var transposed = matrix.T;
        Assert.Equal((1, 0), IndexOfMax(transposed));
        Assert.Equal((2, 0), IndexOfMin(transposed));

        var slice = matrix[1.., 1..].AsReadOnly();
        Assert.Equal((0, 1), IndexOfMax(slice));
        Assert.Equal((1, 0), IndexOfMin(slice));
    }

    [Fact]
    public void MatrixIndexOfExtremaHandlesNaNAndSignedZeros()
    {
        var nan = Matrix<double>.FromRowMajor(2, 3, [1, 2, double.NaN, 3, double.NaN, 4]);
        Assert.Equal((1, 1), IndexOfMax(nan));
        Assert.Equal((1, 1), IndexOfMin(nan));
        Assert.Equal((2, 0), IndexOfMax(nan.T));
        Assert.Equal((2, 0), IndexOfMin(nan.T));

        var zeros = Matrix<double>.FromRowMajor(2, 2, [0.0, 0.0, -0.0, -0.0]);
        Assert.Equal((0, 0), IndexOfMax(zeros.T));
        Assert.Equal((0, 1), IndexOfMin(zeros.T));
    }

    [Fact]
    public void SignedZerosAreOrderedForDecimalToo()
    {
        // decimal の Max は等しい2つのうち先、Min は後の方を返すが、位置と値は double と同じく -0 < +0 で決める
        decimal negative = new(0, 0, 0, true, 0), positive = 0m;
        Assert.True(decimal.IsNegative(negative));
        foreach (bool negativeFirst in new[] { true, false })
        {
            decimal[] values = negativeFirst ? [negative, positive] : [positive, negative];
            int maximum = negativeFirst ? 1 : 0, minimum = 1 - maximum;
            string label = negativeFirst ? "[-0, +0]" : "[+0, -0]";
            var vector = new Tuna.LinearAlgebra.Vector<decimal>(values);
            var column = Matrix<decimal>.FromColumnMajor(2, 1, values);
            var row = Matrix<decimal>.FromColumnMajor(1, 2, values);
            Assert.True(maximum == IndexOfMax(vector), $"IndexOfMax {label}");
            Assert.True(minimum == IndexOfMin(vector), $"IndexOfMin {label}");
            Assert.True(maximum == ColumnIndexOfMax(column)[0], $"ColumnIndexOfMax {label}");
            Assert.True(minimum == ColumnIndexOfMin(column)[0], $"ColumnIndexOfMin {label}");
            Assert.True(maximum == RowIndexOfMax(row)[0], $"RowIndexOfMax {label}");
            Assert.True(minimum == RowIndexOfMin(row)[0], $"RowIndexOfMin {label}");
            foreach (bool ignoreNaN in new[] { false, true })
            {
                Assert.False(decimal.IsNegative(Max(vector, ignoreNaN)), $"Max {label} ignoreNaN={ignoreNaN}");
                Assert.True(decimal.IsNegative(Min(vector, ignoreNaN)), $"Min {label} ignoreNaN={ignoreNaN}");
                Assert.False(decimal.IsNegative(ColumnMax(column, ignoreNaN)[0]), $"ColumnMax {label} ignoreNaN={ignoreNaN}");
                Assert.True(decimal.IsNegative(RowMin(row, ignoreNaN)[0]), $"RowMin {label} ignoreNaN={ignoreNaN}");
            }
        }

        // 0 でない同じ値や整数は従来どおり最初の位置
        Assert.Equal(0, IndexOfMax(new Tuna.LinearAlgebra.Vector<decimal>([2.5m, 2.50m, 1m])));
        Assert.Equal(1, IndexOfMin(new Tuna.LinearAlgebra.Vector<int>([3, 0, 0])));
    }

    // ---------- NaN を除く和・平均・最大・最小 ----------

    /// <summary>およそ4つに1つを NaN（符号・内容・シグナリングの違うもの）にした値。<paramref name="all"/> なら全部 NaN</summary>
    private static T[] WithNaNs<T>(int length, int seed, bool all = false) where T : unmanaged, IFloatingPointIeee754<T>
    {
        bool isDouble = Unsafe.SizeOf<T>() == sizeof(double);
        T[] nans =
        [
            T.NaN,
            FromBits<T>(isDouble ? 0x7FF8_0000_0000_0042 : 0x7FC0_0042),
            FromBits<T>(isDouble ? 0x7FF0_0000_0000_0001 : 0x7F80_0001),
        ];
        var random = new Random(seed);
        var values = Values<T>(length, seed);
        for (int i = 0; i < length; i++)
        {
            if (all || random.Next(4) == 0)
            {
                values[i] = nans[random.Next(nans.Length)];
            }
        }

        return values;
    }

    private static T ReferenceSumIgnoringNaN<T>(T[] x) where T : IFloatingPointIeee754<T> => ReferenceSum(x.Length, i => T.IsNaN(x[i]) ? T.Zero : x[i]);

    private static T ReferenceMeanIgnoringNaN<T>(T[] x) where T : IFloatingPointIeee754<T>
    {
        int n = x.Count(v => !T.IsNaN(v));
        T count = T.CreateTruncating(n);
        T mean = ReferenceSumIgnoringNaN(x) / count;
        if (T.IsFinite(mean) || n == 0)
        {
            return mean;
        }

        int k = 65 - BitOperations.LeadingZeroCount((ulong)n);
        T down = T.ScaleB(T.One, -k);
        return ReferenceSum(x.Length, i => T.IsNaN(x[i]) ? T.Zero : x[i] * down) / count * T.ScaleB(T.One, k);
    }

    // NaN でない要素の最大（最小）。すべて NaN なら NaN
    private static T ReferenceExtremumIgnoringNaN<T>(T[] x, bool maximum) where T : IFloatingPointIeee754<T>
    {
        var numbers = x.Where(v => !T.IsNaN(v)).ToArray();
        return numbers.Length == 0 ? T.NaN : maximum ? ReferenceMax(numbers) : ReferenceMin(numbers);
    }

    // 期待値が NaN（すべて NaN）なら、どの NaN が残るかは保証しないので NaN であることだけを確かめる
    private static void AssertExtremumIgnoringNaN<T>(string name, T expected, T actual) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (T.IsNaN(expected))
        {
            Assert.True(T.IsNaN(actual), $"{name}: expected NaN, actual {actual}");
        }
        else
        {
            AssertBits(name, expected, actual);
        }
    }

    private static void CheckIgnoringNaN<T>() where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
    {
        foreach (int n in Lengths)
        {
            var cases = new List<(string Name, T[] Values)> { ("nans", WithNaNs<T>(n, n)), ("all NaN", WithNaNs<T>(n, n + 1, all: true)), ("ties", Ties<T>(n, n + 2)) };
            cases.AddRange(SpecialCases<T>(n, n + 3));
            foreach (var (name, x) in cases)
            {
                ForEachLayout(x, (layout, v) =>
                {
                    string label = $"{typeof(T).Name} n={n} {name} {layout}";
                    AssertBits($"Sum ignoreNaN {label}", ReferenceSumIgnoringNaN(x), Sum(v, ignoreNaN: true));
                    AssertBits($"Sum ignoreNaN: false {label}", Sum(v), Sum(v, ignoreNaN: false));
                    AssertBits($"Mean ignoreNaN: false {label}", Mean(v), Mean(v, ignoreNaN: false));
                    AssertBits($"Mean ignoreNaN {label}", ReferenceMeanIgnoringNaN(x), Mean(v, ignoreNaN: true));
                    if (n > 0)
                    {
                        AssertExtremumIgnoringNaN($"Max ignoreNaN {label}", ReferenceExtremumIgnoringNaN(x, maximum: true), Max(v, ignoreNaN: true));
                        AssertBits($"Max ignoreNaN: false {label}", Max(v), Max(v, ignoreNaN: false));
                        AssertBits($"Min ignoreNaN: false {label}", Min(v), Min(v, ignoreNaN: false));
                        AssertExtremumIgnoringNaN($"Min ignoreNaN {label}", ReferenceExtremumIgnoringNaN(x, maximum: false), Min(v, ignoreNaN: true));
                    }

                    // 1ブロックに収まる長さでは、NaN を飛ばして先頭から順に足すのと同じ値になる
                    if (n <= Block)
                    {
                        T sequential = T.Zero;
                        foreach (T value in x.Where(e => !T.IsNaN(e)))
                        {
                            sequential += value;
                        }

                        AssertBits($"Sum ignoreNaN sequential {label}", sequential, Sum(v, ignoreNaN: true));
                    }
                });
            }
        }

        // 途中の和だけが溢れる平均
        foreach (int n in new[] { 3, 70, 600, 1500 })
        {
            var huge = WithNaNs<T>(n, n).Select(e => e * T.MaxValue).ToArray();
            ForEachLayout(huge, (layout, v) => AssertBits($"Mean ignoreNaN huge n={n} {layout}", ReferenceMeanIgnoringNaN(huge), Mean(v, ignoreNaN: true)));
        }
    }

    [Fact]
    public void IgnoringNaNMatchesTheDefinitionForDouble() => CheckIgnoringNaN<double>();

    [Fact]
    public void IgnoringNaNMatchesTheDefinitionForSingle() => CheckIgnoringNaN<float>();

    private static void CheckMatrixIgnoringNaN<T>() where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
    {
        foreach (var (rows, columns) in Shapes)
        {
            foreach (var (layout, getView) in MatrixLayouts<T>(rows, columns, rows * 1000 + columns))
            {
                MatrixSpan<T> view = getView();
                var values = WithNaNs<T>(rows * columns, rows + 7 * columns);
                for (int j = 0; j < columns; j++)
                {
                    for (int i = 0; i < rows; i++)
                    {
                        view[i, j] = values[j * rows + i];
                    }
                }
                // 全部 NaN の行と列
                for (int j = 0; j < columns; j++)
                {
                    view[0, j] = T.NaN;
                }

                for (int i = 0; i < rows; i++)
                {
                    view[i, columns - 1] = T.NaN;
                }

                ReadOnlyMatrixSpan<T> a = view;
                string label = $"{typeof(T).Name} {rows}x{columns} {layout}";
                var columnSums = ColumnSums(a, ignoreNaN: true);
                var columnMeans = ColumnMeans(a, ignoreNaN: true);
                var columnMax = ColumnMax(a, ignoreNaN: true);
                var columnMin = ColumnMin(a, ignoreNaN: true);
                for (int j = 0; j < columns; j++)
                {
                    var c = ColumnOf(a, j);
                    AssertBits($"ColumnSums ignoreNaN {label} [{j}]", ReferenceSumIgnoringNaN(c), columnSums[j]);
                    AssertBits($"ColumnMeans ignoreNaN {label} [{j}]", ReferenceMeanIgnoringNaN(c), columnMeans[j]);
                    AssertExtremumIgnoringNaN($"ColumnMax ignoreNaN {label} [{j}]", ReferenceExtremumIgnoringNaN(c, maximum: true), columnMax[j]);
                    AssertExtremumIgnoringNaN($"ColumnMin ignoreNaN {label} [{j}]", ReferenceExtremumIgnoringNaN(c, maximum: false), columnMin[j]);
                }

                var rowSums = RowSums(a, ignoreNaN: true);
                var rowMeans = RowMeans(a, ignoreNaN: true);
                var rowMax = RowMax(a, ignoreNaN: true);
                var rowMin = RowMin(a, ignoreNaN: true);
                for (int i = 0; i < rows; i++)
                {
                    var r = RowOf(a, i);
                    AssertBits($"RowSums ignoreNaN {label} [{i}]", ReferenceSumIgnoringNaN(r), rowSums[i]);
                    AssertBits($"RowMeans ignoreNaN {label} [{i}]", ReferenceMeanIgnoringNaN(r), rowMeans[i]);
                    AssertExtremumIgnoringNaN($"RowMax ignoreNaN {label} [{i}]", ReferenceExtremumIgnoringNaN(r, maximum: true), rowMax[i]);
                    AssertExtremumIgnoringNaN($"RowMin ignoreNaN {label} [{i}]", ReferenceExtremumIgnoringNaN(r, maximum: false), rowMin[i]);
                }
            }
        }
    }

    [Fact]
    public void MatrixIgnoringNaNMatchesTheDefinitionForDouble() => CheckMatrixIgnoringNaN<double>();

    [Fact]
    public void MatrixIgnoringNaNMatchesTheDefinitionForSingle() => CheckMatrixIgnoringNaN<float>();

    /// <summary>引数を省いた版は NaN を伝播し、コンテナーを渡す版でも ignoreNaN を指定できる</summary>
    [Fact]
    public void IgnoreNaNDefaultsToPropagatingAndAcceptsContainers()
    {
        var x = new Tuna.LinearAlgebra.Vector<double>([1.0, double.NaN, 3.0]);
        Assert.True(double.IsNaN(Sum(x)));
        Assert.True(double.IsNaN(Mean(x)));
        Assert.True(double.IsNaN(Max(x)));
        Assert.True(double.IsNaN(Min(x)));
        Assert.Equal(4.0, Sum(x, ignoreNaN: true));
        Assert.Equal(2.0, Mean(x, ignoreNaN: true));
        Assert.Equal(3.0, Max(x, ignoreNaN: true));
        Assert.Equal(1.0, Min(x, ignoreNaN: true));

        var a = Matrix<double>.FromColumnMajor(2, 2, [1.0, double.NaN, double.NaN, 4.0]);
        Assert.Equal([1.0, 4.0], ColumnSums(a, ignoreNaN: true).ToArray());
        Assert.Equal([1.0, 4.0], RowMeans(a, ignoreNaN: true).ToArray());
        var destination = new Tuna.LinearAlgebra.Vector<double>(2);
        ColumnMax(a, true, destination);
        Assert.Equal([1.0, 4.0], destination.ToArray());
        Assert.True(RowMin(a, ignoreNaN: false).ToArray().All(double.IsNaN));
    }
}
