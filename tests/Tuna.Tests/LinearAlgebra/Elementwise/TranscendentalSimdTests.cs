using System.Numerics;
using System.Runtime.CompilerServices;
using static Tuna.XMath;

namespace Tuna.Tests.LinearAlgebra;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

/// <summary>
/// 指数・対数を含む要素演算の SIMD 版（float・double）。
/// SIMD 版は Math.Exp・Math.Log を自前の指数・対数（誤差 0.51 ulp 未満）に置き換えたもので、スカラーの定義（XMath の同名の関数）とは
/// 末尾の桁が違いうる。そのため次の4つを確かめる。
/// <list type="bullet">
/// <item><description>配置によらない: 同じ入力は、連続領域の本体・端数・部分行列・転置ビュー・ストライド付きのビュー・その場更新のどれでも同じビット</description></item>
/// <item><description>精度: 指数・対数は正しく丸めた値（mpmath で求めた参照値）との差が 0.51 ulp 未満</description></item>
/// <item><description>変更前との近さ: double は変更前（スカラーの定義）と 4 ulp 以内で、ほとんどの入力で同じビット。
/// 特別な値のうち ±0・±∞ は同じビット、NaN は NaN であること。float の合成した演算も NaN 以外は同じビット（2^32 通りで確かめた。ここでは標本）</description></item>
/// <item><description>命令セットによらない: 決まった入力の結果のビット列のハッシュが、AVX-512・AVX2・SSE・SIMD なしで同じ</description></item>
/// </list>
/// 近似版（approximate: true）も同じ表で配置と命令セットによらないことを確かめる。double は正確な版と同じハッシュになる。
/// float の近似版の誤差と特別な入力は ApproximateElementwiseTests で確かめる。
/// </summary>
[Trait("LinearAlgebra", "XMath")]
public class TranscendentalSimdTests
{
    private delegate void UnaryInto<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination) where T : unmanaged, INumberBase<T>;

    private delegate void UnaryInPlace<T>(MatrixSpan<T> x) where T : unmanaged, INumberBase<T>;

    // 各オーバーロード（ベクトル新規割り当て・行列 destination・インプレース）、スカラー定義、代表的な入力範囲
    private sealed record Unary<T>(string Name, Func<T, T> Scalar, Func<Vector<T>, Vector<T>> Allocating, UnaryInto<T> Into, UnaryInPlace<T> InPlace,
        double Low, double High)
        where T : unmanaged, INumberBase<T>;

    private static IEnumerable<Unary<T>> Operations<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        yield return new("Exp", v => T.Exp(v), x => Exp(x), (x, d) => Exp(x, d), x => InPlace.Exp(x), -30, 30);
        yield return new("Log", v => Log(v), x => Log(x), (x, d) => Log(x, d), x => InPlace.Log(x), 1e-6, 1e6);
        yield return new("Log1p", v => Log1p(v), x => Log1p(x), (x, d) => Log1p(x, d), x => InPlace.Log1p(x), -0.99, 10);
        yield return new("Expm1", v => Expm1(v), x => Expm1(x), (x, d) => Expm1(x, d), x => InPlace.Expm1(x), -5, 5);
        yield return new("Tanh", v => Tanh(v), x => Tanh(x), (x, d) => Tanh(x, d), x => InPlace.Tanh(x), -10, 10);
        yield return new("Logistic", v => Logistic(v), x => Logistic(x), (x, d) => Logistic(x, d), x => InPlace.Logistic(x), -40, 40);
        yield return new("Logit", v => Logit(v), x => Logit(x), (x, d) => Logit(x, d), x => InPlace.Logit(x), 0, 1);
        yield return new("LogLogistic", v => LogLogistic(v), x => LogLogistic(x), (x, d) => LogLogistic(x, d), x => InPlace.LogLogistic(x), -40, 40);
        yield return new("Softplus", v => Softplus(v), x => Softplus(x), (x, d) => Softplus(x, d), x => InPlace.Softplus(x), -40, 40);
        yield return new("Silu", v => Silu(v), x => Silu(x), (x, d) => Silu(x, d), x => InPlace.Silu(x), -40, 40);
        yield return new("Gelu approximate", v => Gelu(v, true), x => Gelu(x, true), (x, d) => Gelu(x, true, d), x => InPlace.Gelu(x, true), -10, 10);
        yield return new("Selu", v => Selu(v), x => Selu(x), (x, d) => Selu(x, d), x => InPlace.Selu(x), -10, 10);
        T alpha = T.CreateTruncating(0.7);
        yield return new("Elu", v => Elu(v, alpha), x => Elu(x, alpha), (x, d) => Elu(x, alpha, d), x => InPlace.Elu(x, alpha), -10, 10);

        // 近似版（approximate: true）。double は正確な版と同じ値、float は近似の式（ApproximateElementwiseTests で誤差と特別な入力を確かめる）
        yield return new("Exp approximate", v => T.Exp(v), x => Exp(x, true), (x, d) => Exp(x, true, d), x => InPlace.Exp(x, true), -30, 30);
        yield return new("Tanh approximate", v => Tanh(v, true), x => Tanh(x, true), (x, d) => Tanh(x, true, d), x => InPlace.Tanh(x, true), -10, 10);
        yield return new("Logistic approximate", v => Logistic(v, true), x => Logistic(x, true), (x, d) => Logistic(x, true, d), x => InPlace.Logistic(x, true), -40, 40);
        yield return new("Silu approximate", v => Silu(v, true), x => Silu(x, true), (x, d) => Silu(x, true, d), x => InPlace.Silu(x, true), -40, 40);
        yield return new("Softplus approximate", v => Softplus(v, true), x => Softplus(x, true), (x, d) => Softplus(x, true, d), x => InPlace.Softplus(x, true), -40, 40);
    }

    private static long Bits<T>(T value) where T : unmanaged =>
        Unsafe.SizeOf<T>() == sizeof(double) ? Unsafe.BitCast<T, long>(value) : Unsafe.BitCast<T, int>(value);

    private static T FromBits<T>(long bits) where T : unmanaged =>
        Unsafe.SizeOf<T>() == sizeof(double) ? Unsafe.BitCast<long, T>(bits) : Unsafe.BitCast<int, T>((int)bits);

    /// <summary>有限の値を整数の順序に並べたもの（隣り合う値の差が 1）</summary>
    private static long Ordinal<T>(T value) where T : unmanaged
    {
        long bits = Bits(value);
        long sign = Unsafe.SizeOf<T>() == sizeof(double) ? long.MinValue : int.MinValue;
        return bits < 0 ? sign - bits : bits;
    }

    /// <summary>特別な値（±0・±∞・NaN・非正規化数・最大値・定義域の境目・指数の下位桁あふれと溢れの境目）</summary>
    private static T[] SpecialValues<T>() where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
    {
        bool isDouble = Unsafe.SizeOf<T>() == sizeof(double);
        return
        [
            T.Zero, T.NegativeZero, T.One, -T.One, T.CreateTruncating(0.5), T.CreateTruncating(0.25), T.CreateTruncating(0.75), T.CreateTruncating(2),
            T.Epsilon, -T.Epsilon, T.MaxValue, T.MinValue, T.PositiveInfinity, T.NegativeInfinity, T.NaN,
            FromBits<T>(isDouble ? unchecked((long)0xFFF8_0000_0000_1234) : unchecked((int)0xFFC0_1234)),
            FromBits<T>(isDouble ? 0x7FF8_0000_0000_0042 : 0x7FC0_0042),
            T.CreateTruncating(1e-6), T.CreateTruncating(-1e-6), T.CreateTruncating(1e-20), T.CreateTruncating(-0.6931471805599453),
            T.CreateTruncating(-708.5), T.CreateTruncating(-745.2), T.CreateTruncating(-746), T.CreateTruncating(709.9), T.CreateTruncating(710),
            T.CreateTruncating(-1000), T.CreateTruncating(1000), T.CreateTruncating(-88.5), T.CreateTruncating(89), T.CreateTruncating(-104),
            T.CreateTruncating(2.2250738585072014e-308), T.CreateTruncating(4.9e-322), T.CreateTruncating(1e-310),
        ];
    }

    /// <summary>特別な値と、全域・代表的な範囲の乱数を混ぜた入力</summary>
    private static T[] Pool<T>(Unary<T> op, int seed) where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
    {
        var random = new Random(seed);
        var pool = new List<T>(SpecialValues<T>());
        for (int i = 0; i < 40; i++)
        {
            pool.Add(T.CreateTruncating(op.Low + (op.High - op.Low) * random.NextDouble()));
        }

        for (int i = 0; i < 20; i++)
        {
            pool.Add(T.CreateTruncating((random.NextDouble() - 0.5) * 1600));
        }

        return pool.ToArray();
    }

    /// <summary>
    /// 1要素だけのベクトルで求めた値（端数の経路）を基準に、ビット単位で一致すること。NaN どうしは静かな NaN のビットを立てて比べる
    /// （配置間の比較。Math.Exp・Math.Log との NaN のビット一致は求めない）
    /// </summary>
    private static void AssertSame<T>(string context, T expected, T actual) where T : unmanaged, IFloatingPointIeee754<T>
    {
        long quiet = Unsafe.SizeOf<T>() == sizeof(double) ? 1L << 51 : 1L << 22;
        long e = T.IsNaN(expected) ? Bits(expected) | quiet : Bits(expected);
        long a = T.IsNaN(actual) ? Bits(actual) | quiet : Bits(actual);
        Assert.True(e == a, $"{context}: expected {expected} ({Bits(expected):X}), actual {actual} ({Bits(actual):X})");
    }

    private static void AssertLayoutIndependent<T>() where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
    {
        foreach (var op in Operations<T>())
        {
            var pool = Pool(op, 71);
            T Single(T v) => op.Allocating(new Vector<T>([v]))[0];
            var single = pool.DistinctBy(v => Bits(v)).ToDictionary(v => Bits(v), Single);
            T Expected(T v) => single[Bits(v)];

            // 長さを 0〜70 と変えて SIMD の本体と端数の両方を通し、特別な値がどちらにも入るよう並びをずらす
            for (int n = 0; n <= 70; n++)
            {
                var x = Enumerable.Range(0, n).Select(i => pool[(7 * i + n) % pool.Length]).ToArray();
                var allocated = op.Allocating(new Vector<T>(x));
                var matrix = Matrix<T>.FromColumnMajor(n, 1, x);
                op.InPlace(matrix);
                for (int i = 0; i < n; i++)
                {
                    AssertSame($"{op.Name}({x[i]}) length {n} [{i}]", Expected(x[i]), allocated[i]);
                    AssertSame($"{op.Name}({x[i]}) in place length {n} [{i}]", Expected(x[i]), matrix[i, 0]);
                }
            }

            var original = Matrix<T>.FromColumnMajor(13, 11, Enumerable.Range(0, 13 * 11).Select(i => pool[(5 * i + 1) % pool.Length]).ToArray());

            void AssertBlock(Matrix<T> actual, int rowStart, int rowEnd, int columnStart, int columnEnd, string layout)
            {
                for (int j = 0; j < 11; j++)
                {
                    for (int i = 0; i < 13; i++)
                    {
                        bool inside = i >= rowStart && i < rowEnd && j >= columnStart && j < columnEnd;
                        AssertSame($"{op.Name} {layout} [{i}, {j}]", inside ? Expected(original[i, j]) : original[i, j], actual[i, j]);
                    }
                }
            }

            // 全体が連続した行列
            var target = new Matrix<T>(13, 11);
            op.Into(original, target);
            AssertBlock(target, 0, 13, 0, 11, "dense");

            // 列が連続した部分行列（列ごとに SIMD）
            target = original.Clone();
            op.Into(original[1..12, 2..9], target[1..12, 2..9]);
            AssertBlock(target, 1, 12, 2, 9, "submatrix");

            // 行が連続した転置ビュー（行ごとに SIMD）のその場更新
            target = original.Clone();
            op.InPlace(target.T[2..9, 1..12]);
            AssertBlock(target, 1, 12, 2, 9, "transposed submatrix");

            // 入力と出力の向きが違う・1行だけ（作業領域に集めて SIMD）
            var transposed = new Matrix<T>(11, 13);
            op.Into(original, transposed.T);
            AssertBlock(transposed.T.ToMatrix(), 0, 13, 0, 11, "mixed orientation");
            target = original.Clone();
            op.InPlace(target[3..4, 1..10]);
            AssertBlock(target, 3, 4, 1, 10, "single row");
        }
    }

    [Fact]
    public void ResultsDoNotDependOnLayoutOrPosition()
    {
        AssertLayoutIndependent<double>();
        AssertLayoutIndependent<float>();
    }

    [Theory]
    [InlineData(137, 1)]
    [InlineData(1, 137)]
    [InlineData(13, 11)]
    public void ExpPreservesGapsInStridedViews(int rows, int columns)
    {
        const double sentinel = -123;
        int columnStride = 2 * rows + 1;
        var pool = SpecialValues<double>();
        double[] values = Enumerable.Range(0, rows * columns).Select(i => pool[i % pool.Length]).ToArray();
        var expected = Exp(new ReadOnlyVectorSpan<double>(values));
        double[] inputStorage = Enumerable.Repeat(sentinel, columnStride * columns).ToArray();
        var outputStorage = (double[])inputStorage.Clone();
        var expectedStorage = (double[])inputStorage.Clone();
        for (int j = 0; j < columns; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                int index = 2 * i + columnStride * j;
                inputStorage[index] = values[i + rows * j];
                expectedStorage[index] = expected[i + rows * j];
            }
        }

        var input = new MatrixSpan<double>(inputStorage, [rows, columns], [2, columnStride]);
        var output = new MatrixSpan<double>(outputStorage, [rows, columns], [2, columnStride]);
        Exp(input.AsReadOnly(), output);
        InPlace.Exp(input);
        for (int i = 0; i < expectedStorage.Length; i++)
        {
            AssertSame($"destination [{i}]", expectedStorage[i], outputStorage[i]);
            AssertSame($"in place [{i}]", expectedStorage[i], inputStorage[i]);
        }
    }

    [Fact]
    public void BufferedExpAcceptsBroadcastInput()
    {
        const int rows = 13, columns = 11;
        double[] values = SpecialValues<double>()[10..(10 + rows)];
        var expected = Exp(new ReadOnlyVectorSpan<double>(values));
        var input = new ReadOnlyMatrixSpan<double>(values, [rows, columns], [1, 0]);
        var output = new Matrix<double>(columns, rows).T;
        Exp(input, output);
        for (int j = 0; j < columns; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                AssertSame($"broadcast [{i}, {j}]", expected[i], output[i, j]);
            }
        }
    }

    [Fact]
    public void DoubleResultsStayWithinFourUlpOfThePreviousScalarDefinitionAndKeepSpecialCases()
    {
        foreach (var op in Operations<double>())
        {
            var random = new Random(3);
            var x = SpecialValues<double>().Concat(Enumerable.Range(0, 20_000).Select(_ => op.Low + (op.High - op.Low) * random.NextDouble())).ToArray();
            var result = op.Allocating(new Vector<double>(x));
            int same = 0, comparable = 0;
            for (int i = 0; i < x.Length; i++)
            {
                double before = op.Scalar(x[i]);
                if (!double.IsNaN(before))
                {
                    comparable++;
                    if (Bits(before) == Bits(result[i]))
                    {
                        same++;
                    }
                }

                if (double.IsNaN(before))
                {
                    // CRT による NaN の符号・内容の違いは契約に含めない。合成した演算も同じ。
                    Assert.True(double.IsNaN(result[i]), $"{op.Name}({x[i]:R}): expected NaN, actual {result[i]:R}");
                }
                else if (double.IsInfinity(before) || before == 0)
                {
                    // ±∞・±0 は変更前と同じビット（0 の符号も）。
                    AssertSame($"{op.Name}({x[i]:R})", before, result[i]);
                }
                else
                {
                    // それ以外はほとんど同じビットで、違っても 4 ulp 以内（参照値との誤差の最大は変わらない。docs の第13.5節）
                    Assert.True(Math.Abs(Ordinal(before) - Ordinal(result[i])) <= 4, $"{op.Name}({x[i]:R}): before {before:R}, after {result[i]:R}");
                }
            }

            Assert.True(same >= 0.99 * comparable, $"{op.Name}: only {same} of {comparable} non-NaN results unchanged");
        }
    }

    [Fact]
    public void SingleResultsOfCompositeOperationsAreUnchangedAndExpLogStayWithinOneUlp()
    {
        // float の近似実装（approximate: true）の精度は ApproximateElementwiseTests で検証する
        foreach (var op in Operations<float>().Where(op => !op.Name.EndsWith(" approximate", StringComparison.Ordinal)))
        {
            var random = new Random(5);
            var x = SpecialValues<float>()
                .Concat(Enumerable.Range(0, 20_000).Select(_ => (float)(op.Low + (op.High - op.Low) * random.NextDouble())))
                .Concat(Enumerable.Range(0, 20_000).Select(_ => BitConverter.Int32BitsToSingle(random.Next())))
                .ToArray();
            var result = op.Allocating(new Vector<float>(x));
            for (int i = 0; i < x.Length; i++)
            {
                float before = op.Scalar(x[i]);
                if (float.IsNaN(before))
                {
                    // float の合成演算も内部で double の Exp・Log を通るので、NaN のビットは CRT と一致しなくてよい。
                    Assert.True(float.IsNaN(result[i]), $"{op.Name}({x[i]:R}): expected NaN, actual {result[i]:R}");
                }
                else if (op.Name is "Exp" or "Log")
                {
                    // 変更前は MathF.Exp・MathF.Log（float で計算）。変更後は double で計算して丸めるので、正しく丸めた値に近い
                    Assert.True(Math.Abs(Ordinal(before) - Ordinal(result[i])) <= 1,
                        $"{op.Name}({x[i]:R}): before {before:R}, after {result[i]:R}");
                }
                else
                {
                    // NaN 以外の合成した演算は変更前も double で計算して丸めていたので、同じビット
                    AssertSame($"{op.Name}({x[i]:R})", before, result[i]);
                }
            }
        }
    }

    /// <summary>
    /// NaN（符号・内容・シグナリングの違うもの）の指数・対数と負の数・-∞ の対数は NaN。
    /// ビット列は Math ではなく、同じ API の1要素の結果と比べる。本体と端数のすべての位置で、静かにするビットも含め一致すること。
    /// </summary>
    [Fact]
    public void NaNAndNegativeInputsKeepNaNAndPositionIndependentBits()
    {
        double[] special =
        [
            double.NaN, BitConverter.UInt64BitsToDouble(0x7FF8_0000_0000_0042), BitConverter.UInt64BitsToDouble(0xFFF8_0000_0000_1234),
            BitConverter.UInt64BitsToDouble(0x7FF0_0000_0000_0001), BitConverter.UInt64BitsToDouble(0xFFF4_0000_0000_0000), -1, -1e-300, -double.Epsilon, double.NegativeInfinity,
        ];
        foreach (double value in special)
        {
            double expectedExp = Exp(new Vector<double>([value]))[0];
            double expectedLog = Log(new Vector<double>([value]))[0];
            if (double.IsNaN(value))
            {
                Assert.True(double.IsNaN(expectedExp));
            }
            else
            {
                Assert.Equal(Bits(Math.Exp(value)), Bits(expectedExp));
            }

            Assert.True(double.IsNaN(expectedLog));

            // 各特殊値を必ずすべての位置に置く。周囲は通常の入力にして、特殊レーンだけの処理も通す。
            for (int n = 1; n <= 37; n++)
            {
                for (int position = 0; position < n; position++)
                {
                    var x = Enumerable.Repeat(1.0, n).ToArray();
                    x[position] = value;
                    double exp = Exp(new Vector<double>(x))[position];
                    double log = Log(new Vector<double>(x))[position];
                    if (double.IsNaN(value))
                    {
                        Assert.True(double.IsNaN(exp));
                    }

                    Assert.True(double.IsNaN(log));
                    Assert.True(Bits(expectedExp) == Bits(exp), $"Exp({Bits(value):X16}) length {n} [{position}]: expected {Bits(expectedExp):X16}, actual {Bits(exp):X16}");
                    Assert.True(Bits(expectedLog) == Bits(log), $"Log({Bits(value):X16}) length {n} [{position}]: expected {Bits(expectedLog):X16}, actual {Bits(log):X16}");
                }
            }
        }
    }

    // mpmath（300 ビット）で求めた e^x・ln x を binary64 の組 (hi, lo) にしたもの（hi は最近接丸め）。
    // 入力は 0・1 の近く、表の区間の境目、溢れ・下位桁あふれの近く、全域の乱数
    private static readonly (ulong X, ulong Hi, ulong Lo)[] ExpReferences =
    [
        (0x01A56E1FC2F8F359, 0x3FF0000000000000, 0x0000000000000000), (0x81A56E1FC2F8F359, 0x3FF0000000000000, 0x0000000000000000),
        (0x3DDB7CDFD9D7BDBB, 0x3FF000000006DF38, 0xBC63112D8E5E6D4C), (0xBDDB7CDFD9D7BDBB, 0x3FEFFFFFFFF24190, 0x3C631714B6A191DC),
        (0x3E10000000000000, 0x3FF0000000400000, 0x3C20000000155555), (0x3FF0000000000000, 0x4005BF0A8B145769, 0x3CA4D57EE2B1013A),
        (0xBFF0000000000000, 0x3FD78B56362CEF38, 0xBC6CA8A4270FADF5), (0x3FE0000000000000, 0x3FFA61298E1E069C, 0xBC8B4690082A4906),
        (0x3F662E42FEFA39EF, 0x3FF00B1AFA5ABCBF, 0xBC85041A3DDF0E81), (0xBF662E42FEFA39EF, 0x3FEFE9D96B2A23D9, 0x3C64DB57ACE73191),
        (0x3F562E42FEFA39EF, 0x3FF0058C86DA1C0A, 0xBC75ED7154D6E964), (0x3FE62E42FEFA39EF, 0x4000000000000000, 0xBC8ABC9E3B39803F),
        (0x4024000000000000, 0x40D5829DCF950560, 0xBD783E055CFEA4BB), (0xC024000000000000, 0x3F07CD79B5647C9B, 0xBBA8E936E2ABD9DE),
        (0x4059200000000000, 0x48FFCC37A76F9E76, 0xC57F96A7BE641F4A), (0xC059200000000000, 0x36E01A0E5643644F, 0x3382A275EE16C313),
        (0x4085E00000000000, 0x7F0D945DF4F8EC8E, 0x7B9183392684A46E), (0xC085E00000000000, 0x00D14F2B0FB9307F, 0x00000000000000AC),
        (0x4086200000000000, 0x7FC586F6BF260CF1, 0xFC626EDD2F5049F4), (0xC086200000000000, 0x0017C8AB2288C9AB, 0x0000000000000000),
        (0x40861FFFFFFFFFFF, 0x7FC586F6BF260A40, 0xFC5449ACBF5A1855), (0x3F562D1BF77AB835, 0x3FF0058C3CFEA870, 0xBC92A2400195E6FE),
        (0x3FF3C0CA2A5B1D5D, 0x400B7EC1A3962BE9, 0xBCA1D88735C0A9BA), (0x40562E42FEBA39EF, 0x47EFFFFFDFFFFFF5, 0x4480D8768B722976),
        (0xC055D589F2FE5107, 0x381000000000001E, 0x34A46AF093269DC3), (0x406FD0CC53CD3108, 0x56E2672D7FBBE178, 0xD382F3BFE13DE7A0),
        (0x405C11465F855168, 0x4A0F5E67DD64472F, 0xC6ADC7A87DC8AA36), (0x407F5192BFA8A884, 0x6D1E8739C2DBF169, 0xE99F345E9DE5CFCA),
        (0xC083FA69C3D98008, 0x0649AE9C7EC0F93D, 0x82CEB67077118E2E), (0x407CA3B5582C2840, 0x694103C169BFC419, 0xE5CCACB2996B45F3),
        (0xC077DAFCE007BA05, 0x1D84494E11610D39, 0x19DB9E7B7556E2AD), (0xC085FEDB5F057434, 0x00776809997112D2, 0x8000000000000013),
        (0x407CCE84891C8340, 0x697EE225255D90F3, 0xE61D773289073FF9), (0xC0804C71167C53C1, 0x10E7812ECD78D4D6, 0x0D5A399EB28050D5),
        (0xC07AA6CFEC8C48D2, 0x197BD006B8F15AA4, 0x95E340D296F6C086), (0xC070F1D9495DFAC9, 0x277D199394995335, 0x241F2D407C2FC834),
        (0x4071E1AEF5ADEBD0, 0x59BB24ED52075492, 0xD651BAF89093E5DB), (0x406BE7FD1C28BD4C, 0x5410ED0674FF5A97, 0x50B645A90B0AC834),
        (0xC081687264EFE8A0, 0x0DB43A5AED1ABA0A, 0x8A345510DAB4A567), (0xC07B21896651AB2B, 0x18CA9236F84B25BB, 0x956C799D21DFD70A),
        (0xC07EF30DF6B11012, 0x134817C2D4ACD6C9, 0x8FD40026745C1EB9), (0xC080FFFE769D6D93, 0x0EE2102CC4A20DDC, 0x8B72B79BE54AA375),
        (0x405FF1ECA534C798, 0x4B745C2E49944E0C, 0x4815DFFBA9EC7394), (0xC085E64B4B16187A, 0x00BF8652459DC409, 0x80000000000000BF),
        (0xC0810A3B7EA78949, 0x0EC417CC6C94014E, 0x8B563D9676E3E53A), (0xC07FC70CA407C141, 0x12163E3DAD27A9E2, 0x0EB2C3789A2C8BC5),
        (0xC06AF4D0DEDDC116, 0x2C7D7D2526017A33, 0xA900D136FAC9B56E), (0xC08399B9AD90BF7F, 0x076160685F7A688D, 0x04087EA9074FA971),
        (0x407D88C808A7FF7C, 0x6A8ACB61B72AE4D3, 0xE6FFA94E84C2496C), (0xC07F1F4A7A9FDD14, 0x13084825CA0FA467, 0x8FAC00EFEB3D0DCA),
        (0xBFB5CCC23BD2D060, 0x3FED634821F476B2, 0x3C8D2826CE8D8CF4), (0x3FEA9719E8DDA8E6, 0x40025D283A8520F8, 0x3C90AA8669B67AD6),
        (0x3FC01A59472B3FA8, 0x3FF2251C14EBB326, 0xBC95E3DFD9E89458), (0x3F9AB2AFEB8C7200, 0x3FF06C3242040D60, 0xBC9D72D10A29A347),
        (0x3F9BF35C5574C900, 0x3FF07157A3D35DE9, 0x3C9F377CC481CE87), (0xBFE9EC2D89431414, 0x3FDC77F82148B36C, 0xBC7281C1FBAE7A78),
        (0xBFE4A0C72C818DE6, 0x3FE0CBA4710B64C3, 0x3C899E094AAFFA10), (0x3FCD82F978508158, 0x3FF4261C655EAEA6, 0x3C47551055AC35BB),
        (0xBFC5665C01675F00, 0x3FEB12C6B2A4EC4B, 0x3C889D962D7A3895), (0xBFEA4F67D4970A5E, 0x3FDC20396E63D3E0, 0xBC29D9FD9FA9596C),
        (0xBFB615EDD280D000, 0x3FED5AE30499CBEE, 0x3C85208DD31F306E), (0x3FE0FAB2FE3E7BDC, 0x3FFB33060FBC33E3, 0xBC8EEC8C9C571865),
        (0xBFED116C66FD879C, 0x3FD9CDAB81BA2E3F, 0xBC38E8C030B130DA), (0x3FE6A20C03167D70, 0x40003A4DBD422974, 0x3C9C303B23320831),
    ];

    private static readonly (ulong X, ulong Hi, ulong Lo)[] LogReferences =
    [
        (0x3FF0000000000001, 0x3CAFFFFFFFFFFFFF, 0x3615555555555554), (0x3FEFFFFFFFFFFFFF, 0xBCA0000000000000, 0xB940000000000000),
        (0x3FF0000000400000, 0x3E0FFFFFFFC00000, 0x3A35555555155555), (0x3FEFFFFFFF800000, 0xBE10000000200000, 0xBA35555555955555),
        (0x3FF0000A7C5AC472, 0x3EE4F8AEA9AF0A25, 0x3B8E3BC61590413B), (0x3FEFFFEB074A771D, 0xBEE4F8BC681DF714, 0xBB8A667E01D4CD6A),
        (0x3FF0200000000000, 0x3F7FE02A6B106789, 0xBBCE44B7E3711EBF), (0x3FEFE00000000000, 0xBF70080559588B35, 0xBC1F96638CF63677),
        (0x3FF01FFFFFFFFFFF, 0x3F7FE02A6B10668B, 0xBBDF029B72B6B2D9), (0x3FE6000000000000, 0xBFD7FAFA3BD8151C, 0x3C5219024ACD3B77),
        (0x3FF5FFFFFFFFFFFF, 0x3FD4618BC21C5EC0, 0xBC7AEBACD4FBEB16), (0x4000000000000000, 0x3FE62E42FEFA39EF, 0x3C7ABC9E3B39803F),
        (0x3FE0000000000000, 0xBFE62E42FEFA39EF, 0xBC7ABC9E3B39803F), (0x4024000000000000, 0x40026BB1BBB55516, 0xBCAF48AD494EA3E9),
        (0x3FB999999999999A, 0xC0026BB1BBB55515, 0xBCA8B752B6B15C17), (0x7E37E43C8800759C, 0x4085963447F87FB5, 0x3D1ABCCC0710FCD4),
        (0x01A56E1FC2F8F359, 0xC085963447F87FB5, 0xBD1AA670D35324E6), (0x0010000000000000, 0xC086232BDD7ABCD2, 0xBD1EEF3FEC1BE37F),
        (0x7FEFFFFFFFFFFFFF, 0x40862E42FEFA39EF, 0x3D1A9C9E3B39803F), (0x4005BF0A8B145769, 0x3FF0000000000000, 0xBC8EA8556644E4CD),
        (0x3FF6A09E667F3BCD, 0x3FD62E42FEFA39F0, 0x3C7C2E0E1B1548C2), (0x7572B58D19943A36, 0x40828BEC5D53165C, 0xBC92BDBE99438C81),
        (0x1B127E40FE7EECE7, 0xC0798CFE0C1F18E8, 0xBC89B1CCB988F175), (0x081F852B701690B0, 0xC08357F6DD878252, 0x3C7D282882A936B2),
        (0x09EEBA75666283B0, 0xC082B75B9310C8D0, 0x3C6FB9ECDC07268A), (0x0CEC1CCEBBD8BF31, 0xC081ADE6A53F753D, 0x3C8AB3D492B0697E),
        (0x40861D3896355255, 0x401A3F70B2233F80, 0xBC921AC85F580F68), (0x018B08F30F094B8E, 0xC0859F6F9922F07B, 0xBC8A97E770FB586A),
        (0x0A4DA9225C34949B, 0xC082965E9B3E7671, 0x3C76AF44E3A24711), (0x1C951C7AF938996B, 0xC07880B494564F14, 0x3C74F2717A456F09),
        (0x7E4764460C1EE16C, 0x40859B948B491C32, 0x3C8004FBE3AF1DFF), (0x3D58D680AF82F35E, 0xC03CAC22607B5900, 0xBC8B908518B868B1),
        (0x0A359DA8693725D5, 0xC0829E72264D0A29, 0x3C4D5F12E4607003), (0x4418D712968FCA97, 0x40471802CDE566B0, 0xBC7607BD4FAF9AA4),
        (0x60A036A94EE37AD1, 0x4076A877BC3720BC, 0xBC9D73CC43759FF1), (0x2EF881C8E9C94C5A, 0xC06783825E78D5F8, 0xBC949FDED4CF026A),
        (0x622DB6E241A5C606, 0x4077BC542CBDB718, 0xBC8E1574871B1642), (0x4BB0FB86B79F7E61, 0x40604BE1695C5ECC, 0x3C9531CD547B225E),
        (0x7DEB4C9095470D32, 0x40857B8B82A22942, 0xBC872B813734DF3C), (0x017D9AE827058C92, 0xC085A4412C242B53, 0x3C7E7DB86F25E03D),
        (0x3D71B68176C7FC80, 0xC03B9FCA64A87320, 0xBC7A9B19A9D41ED9), (0x1ED90BAD53A480CA, 0xC076EEB7E4921792, 0xBC7D514C926BEC45),
        (0x046A03695D78A930, 0xC084A0AA7B37BDE6, 0xBC934E38C32E04FB), (0x5A55DB5DBE3BEAB6, 0x40724D1EE0446712, 0x3C8609EF719FD94E),
        (0x27424C51867A1ACA, 0xC0711A8AF6CDD920, 0x3C5D0EB3DBB15710), (0x42058B550914E2C2, 0x40372BDF6CEA9680, 0xBC974FC94B36F80C),
        (0x3FF9B7D1ED8D17C2, 0x3FDE5FEA775FECB0, 0x3C6282C39A45F9E8), (0x3FF7F845321BF120, 0x3FD9DE83CA92769D, 0x3C5D29C35EA95DB4),
        (0x3FF98B7FF820854B, 0x3FDDF1404DBB9B7A, 0xBC6EFEAD90FA55AE), (0x3FFC92AF49ECA898, 0x3FE28E51EAD146E8, 0xBC579653870CD0EF),
        (0x3FF33C7E6D0B3D94, 0x3FC79436996EB7A5, 0xBC6AD365966D7445), (0x3FFC442DB8E3BEC0, 0x3FE235EC36AD03E3, 0xBC666678938188FC),
        (0x3FE5C2F3F22FFEFE, 0xBFD8AD89DAE42CB2, 0x3C7FEF3B872455B6), (0x3FFDA6AEE6E0CA61, 0x3FE3BDBB89BCD384, 0xBC84909281AC38D1),
        (0x3FF5F552C6C540D4, 0x3FD44275042DEB0F, 0xBC55CBC31D028D76), (0x3FE26C89377557D0, 0xBFE1AAAAE1BD568E, 0xBC738C9F399AB2E7),
        (0x3FE2270AD06173CC, 0xBFE224445C44C5F8, 0x3C3C91ECA1DA9DFF), (0x3FF41DFEF7530579, 0x3FCD4F61C2EA9C67, 0x3C6BA0EC6B181A70),
        (0x3FE0ABA640514094, 0xBFE4DDF614F07FBE, 0x3C83DD405FE96978), (0x3FF4559341D7AE19, 0x3FCEAF1F6BC871DA, 0xBC55781EC67DCE2B),
        (0x3FF25C783C0F5C9C, 0x3FC19E90DA0B9233, 0x3C67557BE425C23E), (0x3FF30AFA2B30165F, 0x3FC6490EC5AF7DD0, 0x3C687E47C93E8E79),
        (0x3FF0251396FFAABF, 0x3F827471C8A51876, 0x3C18373DC489E90F), (0x3FE9A6F52220A5A8, 0xBFCC4D3F8384F33A, 0x3C50FBAD488611CA),
    ];
    private static void AssertAccuracy((ulong X, ulong Hi, ulong Lo)[] references, Func<Vector<double>, Vector<double>> function, string name)
    {
        var x = references.Select(r => BitConverter.UInt64BitsToDouble(r.X)).ToArray();
        var result = function(new Vector<double>(x));
        for (int i = 0; i < x.Length; i++)
        {
            double hi = BitConverter.UInt64BitsToDouble(references[i].Hi), lo = BitConverter.UInt64BitsToDouble(references[i].Lo);
            double ulp = Math.ScaleB(1.0, Math.ILogB(hi) - 52);
            double error = Math.Abs((result[i] - hi - lo) / ulp);
            Assert.True(error < 0.51, $"{name}({x[i]:R}) = {result[i]:R}, reference {hi:R} + {lo:R}, error {error:F4} ulp");
        }
    }

    [Fact]
    public void ExpAndLogAreWithinHalfAnUlpPlusOnePercentOfTheCorrectlyRoundedValue()
    {
        AssertAccuracy(ExpReferences, x => Exp(x), "Exp");
        AssertAccuracy(LogReferences, x => Log(x), "Log");
    }

    [Theory]
    // mpmath 100 桁で算出し、最も近い double に丸めた値。
    [InlineData(1e-5, 9.999999999666668e-6)]
    [InlineData(0.125, 0.12435300177159621)]
    [InlineData(0.5, 0.46211715726000974)]
    [InlineData(0.7499999999999999, 0.6351489523872873)]
    [InlineData(0.75, 0.6351489523872873)]
    [InlineData(0.7500000000000001, 0.6351489523872874)]
    [InlineData(1.2, 0.8336546070121552)]
    [InlineData(2.0, 0.9640275800758169)]
    [InlineData(10.0, 0.9999999958776927)]
    public void TanhMatchesHighPrecisionReferences(double x, double expected)
    {
        Assert.InRange(Math.Abs(Ordinal(expected) - Ordinal(Tanh(x))), 0, 1);
        Assert.InRange(Math.Abs(Ordinal(-expected) - Ordinal(Tanh(-x))), 0, 1);
    }

    [Fact]
    public void TanhPreservesTinyInputsAndRoundsAtTheSaturationBoundary()
    {
        // atanh(1 - 2^(-p-1)) を超える最小の表現可能な入力（p は仮数部の精度）。
        Check(19.061547465398498);
        Check(9.010913848876953f);

        static void Check<T>(T saturation) where T : unmanaged, IFloatingPointIeee754<T>
        {
            T below = T.BitDecrement(saturation);
            T almostOne = T.BitDecrement(T.One);
            T[] x = [T.NegativeZero, T.Zero, -T.Epsilon, T.Epsilon, -below, below, -saturation, saturation, T.NegativeInfinity, T.PositiveInfinity];
            T[] expected = [T.NegativeZero, T.Zero, -T.Epsilon, T.Epsilon, -almostOne, almostOne, -T.One, T.One, -T.One, T.One];
            var actual = Tanh(new Vector<T>(x));
            for (int i = 0; i < x.Length; i++)
            {
                AssertSame($"Tanh({x[i]})", expected[i], actual[i]);
            }
        }
    }

    /// <summary>
    /// 決まった入力（指数・対数の表引きの式を通る範囲。Math.Exp・Math.Log に任せる特別な入力は除く）の結果のビット列の FNV-1a ハッシュ。
    /// 各レーンの演算は IEEE 754 で結果が一意に決まる演算（FMA を含む）だけなので、命令セット（AVX-512・AVX2・SSE・Arm・SIMD なし）によらず同じ値になる。
    /// 式を変えたら、変えた理由とともに値を更新する
    /// </summary>
    private static ulong Hash<T>(Unary<T> op) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var random = new Random(11);
        var x = Enumerable.Range(0, 4099).Select(_ => T.CreateTruncating(op.Low + (op.High - op.Low) * random.NextDouble())).ToArray();
        ulong hash = 14695981039346656037;
        foreach (var y in op.Allocating(new Vector<T>(x)).ToArray())
        {
            hash = (hash ^ (ulong)Bits(y)) * 1099511628211;
        }

        return hash;
    }

    [Fact]
    public void ResultsAreTheSameOnEveryInstructionSet()
    {
        var expected = new Dictionary<string, ulong>
        {
            ["Exp double"] = 0x7B49D2CCA9072870,
            ["Log double"] = 0xCD790C1FC0665FF7,
            ["Log1p double"] = 0xBBCD4220FD39E689,
            ["Expm1 double"] = 0x1340D7F2DE53C014,
            ["Tanh double"] = 0xBF7AA6177032982A,
            ["Logistic double"] = 0xD188B0FCC36385E4,
            ["Logit double"] = 0x0BF705E22FA3CB17,
            ["LogLogistic double"] = 0x0401C931A3026407,
            ["Softplus double"] = 0x1DA4B008CAF1273E,
            ["Silu double"] = 0x84F53C7725BC5D8D,
            ["Gelu approximate double"] = 0x6CE0F50EDBB6D9D8,
            ["Selu double"] = 0x0A8EC86EC2887A22,
            ["Elu double"] = 0x26A9CD36ADF48F1C,
            ["Exp float"] = 0x28A4EBBC3EDC26B0,
            ["Log float"] = 0xAB9CFE386F00E23E,
            ["Log1p float"] = 0x4AC207DF6226D624,
            ["Expm1 float"] = 0xB862E73A92E36117,
            ["Tanh float"] = 0x7A119DE15EA80F97,
            ["Logistic float"] = 0x615C6FEA484B8035,
            ["Logit float"] = 0x3796438D13184230,
            ["LogLogistic float"] = 0x9B17224EEBA79827,
            ["Softplus float"] = 0x866A9A15C9E6675F,
            ["Silu float"] = 0x3E6B3FE2EC487497,
            ["Selu float"] = 0xC4A14B737CA849FD,
            ["Elu float"] = 0x2B85F3A612C825D7,

            // 近似版。double は正確な版と同じ値なので同じハッシュ
            ["Exp approximate double"] = 0x7B49D2CCA9072870,
            ["Tanh approximate double"] = 0xBF7AA6177032982A,
            ["Logistic approximate double"] = 0xD188B0FCC36385E4,
            ["Silu approximate double"] = 0x84F53C7725BC5D8D,
            ["Softplus approximate double"] = 0x1DA4B008CAF1273E,
            ["Exp approximate float"] = 0x824269574EAA2EE8,
            ["Tanh approximate float"] = 0xC25DD55720375ECD,
            ["Logistic approximate float"] = 0x9D557D9EE3EC2F83,
            ["Silu approximate float"] = 0xBAA83695379F835E,
            ["Softplus approximate float"] = 0x31292F345CA4B490,
            // float は近似ロジスティック関数ベースで計算する
            ["Gelu approximate float"] = 0x8070FA2FE39982EE,
        };
        var actual = Operations<double>().Select(op => (Name: $"{op.Name} double", Hash: Hash(op)))
            .Concat(Operations<float>().Select(op => (Name: $"{op.Name} float", Hash: Hash(op))));
        var mismatches = actual.Where(a => !(expected.TryGetValue(a.Name, out ulong value) && value == a.Hash)).Select(a => $"[\"{a.Name}\"] = 0x{a.Hash:X16},").ToList();
        Assert.True(mismatches.Count == 0, string.Join(Environment.NewLine, mismatches));
    }
}
