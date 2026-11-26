using System.Numerics;

namespace Tuna.Tests.LinearAlgebra;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

/// <summary>
/// 複素数の要素演算の SIMD 版（DenseElementwise.Complex.cs）が、System.Numerics.Complex の演算子とビット単位で一致すること。
/// 対象は XMath から Complex で使える要素演算のうち SIMD 版のあるもの（符号反転・絶対値・2乗・逆数・スカラーとの加減乗除・要素ごとの加減乗除）。
/// NaN の成分は、NaN であることだけを確かめる（符号・内容は比べない）。スカラーの Complex の演算子でも NaN の符号・内容は JIT しだいで変わるため:
/// 可換な演算の順序の入れ替え（途中で NaN が2つ出会うとき。0·∞ は新たな NaN を生む）に加え、除算の <c>-xr + xi·t</c> を JIT が <c>xi·t - xr</c> に
/// 書き換えると NaN の符号の反転が消える（ReadyToRun の事前コンパイルと JIT の段で値が違うことを確かめた）。0 の符号・∞・有限の値はビット単位で比べる。
/// </summary>
[Trait("LinearAlgebra", "XMath")]
public class ComplexSimdTests
{
    private sealed record Unary(string Name, Func<Complex, Complex> Scalar, Func<Vector<Complex>, Vector<Complex>> Allocating,
        Action<ReadOnlyMatrixSpan<Complex>, MatrixSpan<Complex>> Into);

    private sealed record Binary(string Name, Func<Complex, Complex, Complex> Scalar, Func<Vector<Complex>, Vector<Complex>, Vector<Complex>> Allocating,
        Action<ReadOnlyMatrixSpan<Complex>, ReadOnlyMatrixSpan<Complex>, MatrixSpan<Complex>> Into);

    private static readonly double QuietNaN = BitConverter.Int64BitsToDouble(0x7FF8_0000_0000_0042);
    private static readonly double NegativeNaN = BitConverter.Int64BitsToDouble(unchecked((long)0xFFF8_0000_0000_1234));

    /// <summary>成分の特別な値（±0・±∞・NaN・非正規化数・溢れの近く）と乱数を組み合わせた値。|re| = |im| の組（Smith の方法の分岐の境目）も入れる</summary>
    private static Complex[] Pool()
    {
        double[] parts = [0, -0.0, 1, -2.5, 3, double.Epsilon, -1e-310, 1e300, -1e308, double.MaxValue, double.PositiveInfinity, double.NegativeInfinity, QuietNaN, NegativeNaN];
        var random = new Random(23);
        var pool = new List<Complex>();
        foreach (double re in parts)
        {
            foreach (double im in parts)
            {
                pool.Add(new Complex(re, im));
            }
        }

        for (int i = 0; i < 60; i++)
        {
            double re = 8 * random.NextDouble() - 4;
            pool.Add(new Complex(re, i % 5 == 0 ? -re : 8 * random.NextDouble() - 4));
        }

        return pool.ToArray();
    }

    private static void AssertBits(string context, Complex expected, Complex actual)
    {
        bool Same(double e, double a) => double.IsNaN(e) ? double.IsNaN(a) : BitConverter.DoubleToInt64Bits(e) == BitConverter.DoubleToInt64Bits(a);
        Assert.True(Same(expected.Real, actual.Real) && Same(expected.Imaginary, actual.Imaginary),
            $"{context}: expected ({expected.Real:R}, {expected.Imaginary:R}), actual ({actual.Real:R}, {actual.Imaginary:R})");
    }

    private static IEnumerable<Unary> UnaryOperations()
    {
        yield return new("Negate", z => -z, x => XMath.Negate(x), (x, d) => XMath.Negate(x, d));
        yield return new("Abs", z => Complex.Abs(z), x => XMath.Abs(x), (x, d) => XMath.Abs(x, d));
        yield return new("Square", z => z * z, x => XMath.Square(x), (x, d) => XMath.Square(x, d));
        yield return new("Reciprocal", z => Complex.One / z, x => XMath.Reciprocal(x), (x, d) => XMath.Reciprocal(x, d));
        foreach (var s in new[] { new Complex(0.75, -2), new Complex(0, 0), new Complex(-0.0, 3), new Complex(1e-310, 1e300), new Complex(double.PositiveInfinity, 1), new Complex(QuietNaN, 0.5) })
        {
            yield return new($"Add {s}", z => z + s, x => XMath.Add(x, s), (x, d) => XMath.Add(x, s, d));
            yield return new($"Subtract {s}", z => z - s, x => XMath.Subtract(x, s), (x, d) => XMath.Subtract(x, s, d));
            yield return new($"Multiply {s}", z => z * s, x => XMath.Multiply(x, s), (x, d) => XMath.Multiply(x, s, d));
            yield return new($"Divide {s}", z => z / s, x => XMath.Divide(x, s), (x, d) => XMath.Divide(x, s, d));
        }
    }

    private static IEnumerable<Binary> BinaryOperations()
    {
        yield return new("Add", (a, b) => a + b, (x, y) => XMath.Add(x, y), (x, y, d) => XMath.Add(x, y, d));
        yield return new("Subtract", (a, b) => a - b, (x, y) => XMath.Subtract(x, y), (x, y, d) => XMath.Subtract(x, y, d));
        yield return new("Hadamard", (a, b) => a * b, (x, y) => XMath.Hadamard(x, y), (x, y, d) => XMath.Hadamard(x, y, d));
        yield return new("Divide", (a, b) => a / b, (x, y) => XMath.Divide(x, y), (x, y, d) => XMath.Divide(x, y, d));
    }

    private static Complex[] Sample(Complex[] pool, int n, int step, int shift) => Enumerable.Range(0, n).Select(i => pool[(step * i + shift) % pool.Length]).ToArray();

    [Fact]
    public void UnaryOperationsMatchComplexOperatorsBitwiseOnEveryLayout()
    {
        var pool = Pool();
        var original = Matrix<Complex>.FromColumnMajor(13, 11, Sample(pool, 13 * 11, 5, 1));
        foreach (var op in UnaryOperations())
        {
            // SIMD の本体と端数の両方を通す。全体はすべての値を通す長さにする
            foreach (int n in Enumerable.Range(0, 41).Append(pool.Length))
            {
                var x = Sample(pool, n, 7, n);
                var result = op.Allocating(new Vector<Complex>(x));
                for (int i = 0; i < n; i++)
                {
                    AssertBits($"{op.Name}({x[i]}) length {n}", op.Scalar(x[i]), result[i]);
                }
            }

            // 全体が連続・列が連続した部分行列・行が連続した転置ビュー・向きの違う入出力（作業領域に集める経路）
            void Check(string layout, ReadOnlyMatrixSpan<Complex> view, MatrixSpan<Complex> target)
            {
                op.Into(view, target);
                for (int j = 0; j < view.Columns; j++)
                {
                    for (int i = 0; i < view.Rows; i++)
                    {
                        AssertBits($"{op.Name} {layout} [{i}, {j}] = {view[i, j]}", op.Scalar(view[i, j]), target[i, j]);
                    }
                }
            }

            Check("dense", original, new Matrix<Complex>(13, 11));
            Check("submatrix", original[1..12, 2..9], new Matrix<Complex>(13, 11)[1..12, 2..9]);
            Check("transposed", original.T[2..9, 1..12], new Matrix<Complex>(13, 11).T[2..9, 1..12]);
            Check("mixed orientation", original, new Matrix<Complex>(11, 13).T);
        }
    }

    [Fact]
    public void BinaryOperationsMatchComplexOperatorsBitwiseOnEveryLayout()
    {
        var pool = Pool();
        var a = Matrix<Complex>.FromColumnMajor(13, 11, Sample(pool, 13 * 11, 5, 1));
        var b = Matrix<Complex>.FromColumnMajor(13, 11, Sample(pool, 13 * 11, 3, 4));
        foreach (var op in BinaryOperations())
        {
            // 2つの入力の組がどの位置にも入るよう、並びを別の間隔でずらす。すべての組を通す長さも試す
            foreach (int n in Enumerable.Range(0, 41).Append(pool.Length * 3))
            {
                var x = Sample(pool, n, 7, 0);
                var y = Sample(pool, n, 11, 3);
                var result = op.Allocating(new Vector<Complex>(x), new Vector<Complex>(y));
                for (int i = 0; i < n; i++)
                {
                    AssertBits($"{op.Name}({x[i]}, {y[i]}) length {n}", op.Scalar(x[i], y[i]), result[i]);
                }
            }

            void Check(string layout, ReadOnlyMatrixSpan<Complex> left, ReadOnlyMatrixSpan<Complex> right, MatrixSpan<Complex> target)
            {
                op.Into(left, right, target);
                for (int j = 0; j < left.Columns; j++)
                {
                    for (int i = 0; i < left.Rows; i++)
                    {
                        AssertBits($"{op.Name} {layout} [{i}, {j}]", op.Scalar(left[i, j], right[i, j]), target[i, j]);
                    }
                }
            }

            Check("dense", a, b, new Matrix<Complex>(13, 11));
            Check("submatrix", a[1..12, 2..9], b[1..12, 2..9], new Matrix<Complex>(13, 11)[1..12, 2..9]);
            Check("transposed", a.T[2..9, 1..12], b.T[2..9, 1..12], new Matrix<Complex>(13, 11).T[2..9, 1..12]);
            Check("mixed orientation", a, b.T.ToMatrix().T, new Matrix<Complex>(13, 11));
        }
    }

    [Fact]
    public void InPlaceScalingMatchesComplexMultiplication()
    {
        var x = Sample(Pool(), 37, 3, 2);
        var s = new Complex(-0.5, 1.25);
        var v = new Vector<Complex>(x);
        v *= s;
        for (int i = 0; i < x.Length; i++)
        {
            AssertBits($"{x[i]} * {s}", x[i] * s, v[i]);
        }
    }
}
