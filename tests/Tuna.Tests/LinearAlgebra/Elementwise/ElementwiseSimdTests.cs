using System.Runtime.CompilerServices;
using Tuna.LinearAlgebra;
using Complex = System.Numerics.Complex;
using static Tuna.XMath;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>
/// 結果が一意に決まる要素演算（四則演算・平方根・符号の操作・比較による選択）の SIMD 版が、スカラーの定義とビット単位で一致すること。
/// float・double の連続領域と、列（行）が連続したビューで SIMD を通り、それ以外のビューは要素ごとの経路を通る。どの経路でも同じビットになる。
/// ただし2つ以上の入力が NaN のときは、結果が NaN であることだけを確かめる。どの NaN の符号・内容が残るかは IEEE 754 が定めず、
/// x86 では命令の第1オペランドの NaN が残るので、JIT が可換な演算の順序を入れ替えるとスカラーの経路でも変わりうる（SSE だけの環境で実際に変わる）。
/// また、シグナリング NaN が最大・最小・Clamp を通ると、JIT の段（Tier0 か最適化済みか）と経路しだいで静かな NaN になることがある（スカラーの T.Max だけでも起きる）。
/// そのため NaN どうしは静かな NaN のビットを立てたうえで比べ、符号と残りのペイロードは一致を求める。
/// </summary>
[Trait("LinearAlgebra", "XMath")]
public class ElementwiseSimdTests
{
    private delegate void UnaryInto<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination) where T : unmanaged, System.Numerics.INumberBase<T>;

    private delegate void UnaryInPlace<T>(MatrixSpan<T> x) where T : unmanaged, System.Numerics.INumberBase<T>;

    private delegate void BinaryInto<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, MatrixSpan<T> destination) where T : unmanaged, System.Numerics.INumberBase<T>;

    // 1 引数演算の各オーバーロード（ベクトル新規割り当て・行列 destination・インプレース）とスカラー定義
    private sealed record Unary<T>(string Name, Func<T, T> Scalar, Func<Vector<T>, Vector<T>> Allocating, UnaryInto<T> Into, UnaryInPlace<T> InPlace,
        int ParameterNaNs = 0)
        where T : unmanaged, System.Numerics.INumberBase<T>;

    // 2 引数演算の各オーバーロードとスカラー定義
    private sealed record Binary<T>(string Name, Func<T, T, T> Scalar, Func<Vector<T>, Vector<T>, Vector<T>> Allocating, BinaryInto<T> Into)
        where T : unmanaged, System.Numerics.INumberBase<T>;

    private static long Bits<T>(T value) where T : unmanaged =>
        Unsafe.SizeOf<T>() == sizeof(double) ? Unsafe.BitCast<T, long>(value) : Unsafe.BitCast<T, int>(value);

    private static T FromBits<T>(long bits) where T : unmanaged =>
        Unsafe.SizeOf<T>() == sizeof(double) ? Unsafe.BitCast<long, T>(bits) : Unsafe.BitCast<int, T>((int)bits);

    /// <summary>NaN（符号と内容の違うもの）・±0・非正規化数・最大値・無限大と乱数</summary>
    private static T[] Pool<T>(int seed) where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>, System.Numerics.IMinMaxValue<T>
    {
        bool isDouble = Unsafe.SizeOf<T>() == sizeof(double);
        var pool = new List<T>
        {
            T.Zero, T.NegativeZero, T.One, -T.One, T.CreateTruncating(0.5), T.CreateTruncating(-2.5), T.CreateTruncating(3),
            T.Epsilon, -T.Epsilon, T.MaxValue, T.MinValue, T.PositiveInfinity, T.NegativeInfinity, T.NaN,
            FromBits<T>(isDouble ? unchecked((long)0xFFF8_0000_0000_1234) : unchecked((int)0xFFC0_1234)),
            FromBits<T>(isDouble ? 0x7FF8_0000_0000_0042 : 0x7FC0_0042),
            FromBits<T>(isDouble ? 0x7FF0_0000_0000_0001 : 0x7F80_0001),
        };
        var random = new Random(seed);
        for (int i = 0; i < 16; i++)
        {
            pool.Add(T.CreateTruncating(8 * random.NextDouble() - 4));
        }

        return pool.ToArray();
    }

    /// <summary>ビット単位で一致すること。<paramref name="manyNaNs"/> が真の位置（2つ以上の入力が NaN）は、どちらも NaN であることだけを確かめる</summary>
    private static void AssertBits<T>(string name, ReadOnlySpan<T> expected, ReadOnlySpan<T> actual, Func<int, bool> manyNaNs)
        where T : unmanaged, System.Numerics.INumberBase<T>
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            bool same = manyNaNs(i) ? T.IsNaN(expected[i]) && T.IsNaN(actual[i]) : QuietedBits(expected[i]) == QuietedBits(actual[i]);
            Assert.True(same, $"{name} [{i}] of {expected.Length}: expected {expected[i]} ({Bits(expected[i]):X}), actual {actual[i]} ({Bits(actual[i]):X})");
        }
    }

    /// <summary>ビット列。NaN なら静かな NaN のビットを立てる</summary>
    private static long QuietedBits<T>(T value) where T : unmanaged, System.Numerics.INumberBase<T>
    {
        long quiet = Unsafe.SizeOf<T>() == sizeof(double) ? 1L << 51 : 1L << 22;
        return T.IsNaN(value) ? Bits(value) | quiet : Bits(value);
    }

    private static int NaNs<T>(params T[] values) where T : System.Numerics.INumberBase<T> => values.Count(T.IsNaN);

    private static IEnumerable<Unary<T>> UnaryOperations<T>(T[] parameters) where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>
    {
        yield return new("Sqrt", v => Sqrt(v), x => Sqrt(x), (x, d) => Sqrt(x, d), x => InPlace.Sqrt(x));
        yield return new("Abs", v => Abs(v), x => Abs(x), (x, d) => Abs(x, d), x => InPlace.Abs(x));
        yield return new("Square", v => Square(v), x => Square(x), (x, d) => Square(x, d), x => InPlace.Square(x));
        yield return new("Reciprocal", v => Reciprocal(v), x => Reciprocal(x), (x, d) => Reciprocal(x, d), x => InPlace.Reciprocal(x));
        yield return new("Relu", v => Relu(v), x => Relu(x), (x, d) => Relu(x, d), x => InPlace.Relu(x));
        yield return new("Negate", v => -v, x => Negate(x), (x, d) => Negate(x, d), x => Negate(x, x));
        foreach (T s in parameters)
        {
            yield return new($"Add {s}", v => v + s, x => Add(x, s), (x, d) => Add(x, s, d), x => InPlace.Add(x, s), NaNs(s));
            yield return new($"Subtract {s}", v => v - s, x => Subtract(x, s), (x, d) => Subtract(x, s, d), x => InPlace.Subtract(x, s), NaNs(s));
            yield return new($"Multiply {s}", v => v * s, x => Multiply(x, s), (x, d) => Multiply(x, s, d), x => InPlace.Multiply(x, s), NaNs(s));
            yield return new($"Divide {s}", v => Divide(v, s), x => Divide(x, s), (x, d) => Divide(x, s, d), x => InPlace.Divide(x, s), NaNs(s));
            yield return new($"Max {s}", v => Max(v, s), x => Max(x, s), (x, d) => Max(x, s, d), x => InPlace.Max(x, s), NaNs(s));
            yield return new($"Min {s}", v => Min(v, s), x => Min(x, s), (x, d) => Min(x, s, d), x => InPlace.Min(x, s), NaNs(s));
            yield return new($"LeakyRelu {s}", v => LeakyRelu(v, s), x => LeakyRelu(x, s), (x, d) => LeakyRelu(x, s, d), x => InPlace.LeakyRelu(x, s), NaNs(s));
            foreach (T t in parameters)
            {
                // min > max は例外になる（別のテスト）
                if (s > t)
                {
                    continue;
                }

                yield return new($"Clamp {s} {t}", v => Clamp(v, s, t), x => Clamp(x, s, t), (x, d) => Clamp(x, s, t, d), x => InPlace.Clamp(x, s, t), NaNs(s, t));
            }
        }
    }

    private static IEnumerable<Binary<T>> BinaryOperations<T>() where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>
    {
        yield return new("Add", (v, w) => v + w, (x, y) => Add(x, y), (x, y, d) => Add(x, y, d));
        yield return new("Subtract", (v, w) => v - w, (x, y) => Subtract(x, y), (x, y, d) => Subtract(x, y, d));
        yield return new("Hadamard", (v, w) => v * w, (x, y) => Hadamard(x, y), (x, y, d) => Hadamard(x, y, d));
        yield return new("Divide", (v, w) => Divide(v, w), (x, y) => Divide(x, y), (x, y, d) => Divide(x, y, d));
        yield return new("Max", (v, w) => Max(v, w), (x, y) => Max(x, y), (x, y, d) => Max(x, y, d));
        yield return new("Min", (v, w) => Min(v, w), (x, y) => Min(x, y), (x, y, d) => Min(x, y, d));
    }

    private static T[] Sample<T>(T[] pool, int n, int step, int shift) => Enumerable.Range(0, n).Select(i => pool[(step * i + shift + n) % pool.Length]).ToArray();

    private static void AssertUnaryAcrossLayouts<T>() where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>, System.Numerics.IMinMaxValue<T>
    {
        var pool = Pool<T>(53);
        var parameters = new[]
        {
            T.Zero, T.NegativeZero, T.CreateTruncating(0.75), T.CreateTruncating(-2), T.Epsilon, T.PositiveInfinity, T.NegativeInfinity,
            T.NaN, FromBits<T>(Unsafe.SizeOf<T>() == sizeof(double) ? unchecked((long)0xFFF8_0000_0000_1234) : unchecked((int)0xFFC0_1234)),
        };
        var original = Matrix<T>.FromColumnMajor(13, 11, Sample(pool, 13 * 11, 5, 1));
        foreach (var op in UnaryOperations<T>(parameters))
        {
            // 長さを 0〜70 と変えて SIMD の本体と端の両方を通し、特別な値がどちらにも入るよう並びをずらす
            for (int n = 0; n <= 70; n++)
            {
                var x = Sample(pool, n, 7, 0);
                var expected = x.Select(op.Scalar).ToArray();
                Func<int, bool> manyNaNs = i => NaNs(x[i]) + op.ParameterNaNs >= 2;
                AssertBits(op.Name, expected, op.Allocating(new Vector<T>(x)).AsSpan(), manyNaNs);
                var matrix = Matrix<T>.FromColumnMajor(n, 1, x);
                op.InPlace(matrix);
                AssertBits(op.Name + " in place", expected, matrix.AsSpan(), manyNaNs);
            }

            void AssertBlock(Matrix<T> actual, int rowStart, int rowEnd, int columnStart, int columnEnd, string layout)
            {
                var e = new T[13 * 11];
                var a = new T[13 * 11];
                var many = new bool[13 * 11];
                for (int j = 0; j < 11; j++)
                {
                    for (int i = 0; i < 13; i++)
                    {
                        bool inside = i >= rowStart && i < rowEnd && j >= columnStart && j < columnEnd;
                        e[i + 13 * j] = inside ? op.Scalar(original[i, j]) : original[i, j];
                        a[i + 13 * j] = actual[i, j];
                        many[i + 13 * j] = inside && NaNs(original[i, j]) + op.ParameterNaNs >= 2;
                    }
                }

                AssertBits($"{op.Name} {layout}", e, a, k => many[k]);
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

            // 入力と出力の向きが違う、1行だけ（要素ごとの経路）
            var transposed = new Matrix<T>(11, 13);
            op.Into(original, transposed.T);
            AssertBlock(transposed.T.ToMatrix(), 0, 13, 0, 11, "mixed orientation");
            target = original.Clone();
            op.InPlace(target[3..4, 1..10]);
            AssertBlock(target, 3, 4, 1, 10, "single row");
        }
    }

    private static void AssertBinaryAcrossLayouts<T>() where T : unmanaged, System.Numerics.IFloatingPointIeee754<T>, System.Numerics.IMinMaxValue<T>
    {
        var pool = Pool<T>(59);
        var a = Matrix<T>.FromColumnMajor(13, 11, Sample(pool, 13 * 11, 5, 1));
        var b = Matrix<T>.FromColumnMajor(13, 11, Sample(pool, 13 * 11, 3, 4));
        foreach (var op in BinaryOperations<T>())
        {
            // 2つの入力の組がどの位置にも入るよう、並びを別の間隔でずらす
            for (int n = 0; n <= 70; n++)
            {
                var x = Sample(pool, n, 7, 0);
                var y = Sample(pool, n, 11, 3);
                var expected = x.Zip(y, op.Scalar).ToArray();
                Func<int, bool> manyNaNs = i => NaNs(x[i], y[i]) >= 2;
                AssertBits(op.Name, expected, op.Allocating(new Vector<T>(x), new Vector<T>(y)).AsSpan(), manyNaNs);
                var matrix = Matrix<T>.FromColumnMajor(n, 1, x);
                op.Into(matrix, Matrix<T>.FromColumnMajor(n, 1, y), matrix);
                AssertBits(op.Name + " in place", expected, matrix.AsSpan(), manyNaNs);
            }

            void AssertBlock(Matrix<T> actual, Matrix<T> before, int rowStart, int rowEnd, int columnStart, int columnEnd, string layout)
            {
                var e = new T[13 * 11];
                var r = new T[13 * 11];
                var many = new bool[13 * 11];
                for (int j = 0; j < 11; j++)
                {
                    for (int i = 0; i < 13; i++)
                    {
                        bool inside = i >= rowStart && i < rowEnd && j >= columnStart && j < columnEnd;
                        e[i + 13 * j] = inside ? op.Scalar(a[i, j], b[i, j]) : before[i, j];
                        r[i + 13 * j] = actual[i, j];
                        many[i + 13 * j] = inside && NaNs(a[i, j], b[i, j]) >= 2;
                    }
                }

                AssertBits($"{op.Name} {layout}", e, r, k => many[k]);
            }

            var target = new Matrix<T>(13, 11);
            op.Into(a, b, target);
            AssertBlock(target, target, 0, 13, 0, 11, "dense");

            target = b.Clone();
            op.Into(a[1..12, 2..9], b[1..12, 2..9], target[1..12, 2..9]);
            AssertBlock(target, b, 1, 12, 2, 9, "submatrix");

            target = a.Clone();
            op.Into(a.T[2..9, 1..12], b.T[2..9, 1..12], target.T[2..9, 1..12]);
            AssertBlock(target, a, 1, 12, 2, 9, "transposed submatrix");

            // 片方の入力だけ転置（要素ごとの経路）
            var bt = b.T.ToMatrix();
            target = new Matrix<T>(13, 11);
            op.Into(a, bt.T, target);
            AssertBlock(target, target, 0, 13, 0, 11, "mixed orientation");
        }
    }

    [Fact]
    public void UnaryOperationsMatchTheScalarDefinitionBitwiseOnEveryLayout()
    {
        AssertUnaryAcrossLayouts<double>();
        AssertUnaryAcrossLayouts<float>();
    }

    [Fact]
    public void BinaryOperationsMatchTheScalarDefinitionBitwiseOnEveryLayout()
    {
        AssertBinaryAcrossLayouts<double>();
        AssertBinaryAcrossLayouts<float>();
    }

    [Fact]
    public void ClampWithMinAboveMaxThrowsOnlyWhenThereAreElements()
    {
        var x = Matrix<double>.FromColumnMajor(13, 11, Enumerable.Range(0, 143).Select(i => i * 0.5).ToArray());
        Assert.Throws<ArgumentException>(() => { Clamp(x, 1.0, 0.0); });
        Assert.Throws<ArgumentException>(() => { InPlace.Clamp(x[1..12, 2..9], 1.0, 0.0); });
        Assert.Throws<ArgumentException>(() => { InPlace.Clamp(x.T[2..9, 1..12], 1.0, 0.0); });
        Assert.Throws<ArgumentException>(() => { Clamp(new[] { 1f, 2f }.ToVector(), 1f, 0f); });
        Assert.Equal(0, Clamp(new Vector<double>(0), 1.0, 0.0).Length);
        Assert.Equal(0, Clamp(x[1..12, 2..2], 1.0, 0.0).Columns);
        // 例外のときは何も書き換えない
        Assert.Equal(Enumerable.Range(0, 143).Select(i => i * 0.5).ToArray(), x.AsSpan().ToArray());
    }

    [Fact]
    public void TypesWithoutSimdKernelsKeepThePerElementPath()
    {
        // 整数・decimal・Half・Complex は要素ごとの経路（スカラーの定義をそのまま使う）
        Assert.Equal(new[] { 3, 2, 0 }, Abs(new[] { -3, 2, 0 }.ToVector()).ToArray());
        Assert.Throws<OverflowException>(() => { Abs(new[] { int.MinValue }.ToVector()); });
        Assert.Equal(new[] { 1.5m, 0, 2 }, Max(new[] { 1.5m, -1, 2 }.ToVector(), 0m).ToArray());
        Assert.Equal(new[] { (Half)2, (Half)0.5 }, Sqrt(new[] { (Half)4, (Half)0.25 }.ToVector()).ToArray());
        Assert.Equal(new[] { new Complex(0, 2) }, Square(new[] { new Complex(1, 1) }.ToVector()).ToArray());
    }
}
