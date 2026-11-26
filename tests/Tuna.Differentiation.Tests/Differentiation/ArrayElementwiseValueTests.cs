using System.Numerics;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using static Tuna.Tests.Differentiation.ElementwiseOperations;

namespace Tuna.Tests.Differentiation;

/// <summary>要素演算の値が、XMath の配列版に同じビュー（部分・転置を含む）を渡した結果と同じビットになること</summary>
public class ArrayElementwiseValueTests
{
    public static TheoryData<string> UnaryCases() => UnaryNames();

    public static TheoryData<string> BinaryCases() => BinaryNames();

    public static TheoryData<string> ScalarCases() => ScalarNames();

    [Theory]
    [MemberData(nameof(UnaryCases))]
    public void UnaryValuesMatchXMathBitForBit(string name)
    {
        UnaryValues<double>(name);
        UnaryValues<float>(name);
    }

    [Theory]
    [MemberData(nameof(BinaryCases))]
    public void BinaryValuesMatchXMathBitForBit(string name)
    {
        BinaryValues<double>(name);
        BinaryValues<float>(name);
    }

    [Theory]
    [MemberData(nameof(ScalarCases))]
    public void ScalarOperandValuesMatchXMathBitForBit(string name)
    {
        // float の配列では、スカラーを float に丸めて XMath に渡した結果と同じビット（float にない桁を持つ値と範囲外の値を含める）。
        foreach (double s in new[] { 0.3, -1.7, 2.0, 0.0, -0.0, 1e-50, 1e300, -1e300, double.PositiveInfinity, double.NaN, 0.1 + 1e-12 })
        {
            ScalarValues<double>(name, s);
            ScalarValues<float>(name, s);
        }
    }

    [Fact]
    public void ScalarSubtractIsTheIeeeDifference()
    {
        // s − x は −x + s で求める。IEEE 754 は NaN の符号を定めないので、NaN 以外を符号付きのゼロまで比べる。
        double[] values = Inputs(10, 20, 3);
        var tape = new Tape();
        var x = tape.Leaf(Vector(values));
        var xf = tape.Leaf(Vector(values).ConvertChecked<float>());
        foreach (double s in values)
        {
            var difference = (s - x).Value;
            var single = ((ScalarNode)s - xf).Value;
            for (int i = 0; i < values.Length; i++)
            {
                double expected = s - values[i];
                float expectedSingle = (float)s - (float)values[i];
                Assert.True(double.IsNaN(expected) ? double.IsNaN(difference[i]) : expected.Equals(difference[i]) && double.IsNegative(expected) == double.IsNegative(difference[i]),
                    $"{s:R} - {values[i]:R}: expected {expected:R}, actual {difference[i]:R}");
                Assert.True(float.IsNaN(expectedSingle) ? float.IsNaN(single[i]) : expectedSingle.Equals(single[i]) && float.IsNegative(expectedSingle) == float.IsNegative(single[i]),
                    $"{s:R} - {values[i]:R} (float): expected {expectedSingle:R}, actual {single[i]:R}");
            }
        }
    }

    [Fact]
    public void ValuesDoNotDependOnWhetherTheOperationIsRecorded()
    {
        var x = Vector(Inputs(3, 40, 4));
        var tape = new Tape();
        var leaf = tape.Leaf(x);
        var constant = tape.Constant(x);
        foreach (string name in Unary)
        {
            var recorded = Apply(name, leaf);
            var value = Apply(name, constant);
            Assert.True(recorded.RequiresGradient);
            Assert.False(value.RequiresGradient);
            AssertSameBits(value.Value.ToArray(), recorded.Value.ToArray(), name);
        }

        var tracked = tape.Leaf(0.5);
        foreach (string name in WithScalar)
        {
            // 定数の配列と記録つきのスカラーなら記録する。どちらも定数なら記録しない。
            Assert.True(Apply(name, constant, tracked).RequiresGradient, name);
            Assert.False(Apply(name, constant, 0.5).RequiresGradient, name);
            AssertSameBits(Apply(name, constant, 0.5).Value.ToArray(), Apply(name, leaf, tracked).Value.ToArray(), name);
        }
    }

    [Fact]
    public void OperatorsMatchTheNamedOperations()
    {
        var random = new Random(8);
        var x = ArrayProductTests.RandomVector<double>(random, 23);
        var y = ArrayProductTests.RandomVector<double>(random, 23);
        var m = ArrayProductTests.RandomMatrix<float>(random, 4, 6);
        var n = ArrayProductTests.RandomMatrix<float>(random, 6, 4);
        var tape = new Tape();
        var xn = tape.Leaf(x);
        var yn = tape.Constant(y);
        var s = tape.Leaf(-0.75);
        AssertSameBits(XMath.Add(x, y).ToArray(), (xn + yn).Value.ToArray(), "x + y");
        AssertSameBits(XMath.Subtract(x, y).ToArray(), (xn - yn).Value.ToArray(), "x - y");
        AssertSameBits(XMath.Negate(x).ToArray(), (-xn).Value.ToArray(), "-x");
        AssertSameBits(XMath.Add(x, -0.75).ToArray(), (xn + s).Value.ToArray(), "x + s");
        AssertSameBits(XMath.Add(x, -0.75).ToArray(), (s + xn).Value.ToArray(), "s + x");
        AssertSameBits(XMath.Subtract(x, -0.75).ToArray(), (xn - s).Value.ToArray(), "x - s");
        AssertSameBits(XMath.Multiply(x, -0.75).ToArray(), (xn * s).Value.ToArray(), "x * s");
        AssertSameBits(XMath.Multiply(x, -0.75).ToArray(), (s * xn).Value.ToArray(), "s * x");
        AssertSameBits(XMath.Divide(x, -0.75).ToArray(), (xn / s).Value.ToArray(), "x / s");
        var mn = tape.Leaf(m);
        var nn = tape.Constant(n);
        AssertSameBits(XMath.Add(m, n.T).AsSpan(), (mn + nn.T).Value.ToMatrix().AsSpan(), "M + Nᵀ");
        AssertSameBits(XMath.Subtract(m.T, n).AsSpan(), (mn.T - nn).Value.ToMatrix().AsSpan(), "Mᵀ - N");
        AssertSameBits(XMath.Multiply(m.T, 3f).AsSpan(), (3 * mn.T).Value.ToMatrix().AsSpan(), "3Mᵀ");
        AssertSameBits(XMath.Divide(m, 3f).AsSpan(), (mn / 3).Value.ToMatrix().AsSpan(), "M/3");
    }

    private static void UnaryValues<T>(string name) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var x = Vector(Inputs(1, 45, 6)).ConvertChecked<T>();
        var tape = new Tape();
        var leaf = tape.Leaf(x);
        AssertSameBits(Expected<T>(name, x).ToArray(), Apply(name, leaf).Value.ToArray(), $"{name} {typeof(T).Name} vector");
        AssertSameBits(Expected<T>(name, x[3..^2]).ToArray(), Apply(name, leaf[3..^2]).Value.ToArray(), $"{name} {typeof(T).Name} subvector");
        // 列の間隔が行数と違う部分行列と、その転置。
        var storage = Matrix(Inputs(2, 70, 6), 12, 10).ConvertChecked<T>();
        var constant = tape.Constant(storage);
        var matrixLeaf = tape.Leaf(storage);
        foreach (var (rows, columns) in new[] { (.., ..), (1..9, 2..7), (3..4, ..), (.., 5..6) })
        {
            ReadOnlyMatrixSpan<T> view = storage[rows, columns];
            AssertSameBits(Expected<T>(name, view).AsSpan(), Apply(name, constant[rows, columns]).Value.ToMatrix().AsSpan(), $"{name} {typeof(T).Name} {rows}, {columns}");
            AssertSameBits(Expected<T>(name, view.T).AsSpan(), Apply(name, constant[rows, columns].T).Value.ToMatrix().AsSpan(), $"{name} {typeof(T).Name} {rows}, {columns} T");
            AssertSameBits(Expected<T>(name, view.T).AsSpan(), Apply(name, matrixLeaf[rows, columns].T).Value.ToMatrix().AsSpan(), $"{name} {typeof(T).Name} leaf {rows}, {columns} T");
        }
    }

    private static void BinaryValues<T>(string name) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var x = Vector(Inputs(4, 30, 3)).ConvertChecked<T>();
        var y = Vector(Inputs(5, 30, 3).Reverse().ToArray()).ConvertChecked<T>();
        var tape = new Tape();
        var xn = tape.Leaf(x);
        var yn = tape.Constant(y);
        AssertSameBits(Expected<T>(name, x, y).ToArray(), Apply(name, xn, yn).Value.ToArray(), $"{name} {typeof(T).Name} vector");
        AssertSameBits(Expected<T>(name, x[2..^5], y[4..^3]).ToArray(), Apply(name, xn[2..^5], yn[4..^3]).Value.ToArray(), $"{name} {typeof(T).Name} subvector");
        var a = Matrix(Inputs(6, 70, 3), 12, 10).ConvertChecked<T>();
        var b = Matrix(Inputs(7, 70, 3), 10, 12).ConvertChecked<T>();
        var an = tape.Leaf(a);
        var bn = tape.Constant(b);
        AssertSameBits(Expected<T>(name, a, b.T).AsSpan(), Apply(name, an, bn.T).Value.ToMatrix().AsSpan(), $"{name} {typeof(T).Name} A, Bᵀ");
        AssertSameBits(Expected<T>(name, a.T, b).AsSpan(), Apply(name, an.T, bn).Value.ToMatrix().AsSpan(), $"{name} {typeof(T).Name} Aᵀ, B");
        AssertSameBits(Expected<T>(name, a[1..8, 2..6], b[3..7, 4..11].T).AsSpan(), Apply(name, an[1..8, 2..6], bn[3..7, 4..11].T).Value.ToMatrix().AsSpan(),
            $"{name} {typeof(T).Name} parts");
    }

    private static void ScalarValues<T>(string name, double s) where T : unmanaged, IFloatingPointIeee754<T>
    {
        T rounded = T.CreateTruncating(s);
        var x = Vector(Inputs(8, 30, 3)).ConvertChecked<T>();
        var a = Matrix(Inputs(9, 70, 3), 12, 10).ConvertChecked<T>();
        var tape = new Tape();
        var sn = tape.Leaf(s);
        var xn = tape.Leaf(x);
        var an = tape.Constant(a);
        string context = $"{name} {typeof(T).Name} s={s:R}";
        AssertSameBits(Expected<T>(name, x, rounded).ToArray(), Apply(name, xn, sn).Value.ToArray(), context);
        AssertSameBits(Expected<T>(name, x[1..^1], rounded).ToArray(), Apply(name, xn[1..^1], s).Value.ToArray(), context);
        AssertSameBits(Expected<T>(name, a, rounded).AsSpan(), Apply(name, an, sn).Value.ToMatrix().AsSpan(), context);
        AssertSameBits(Expected<T>(name, a[2..9, 1..8].T, rounded).AsSpan(), Apply(name, an[2..9, 1..8].T, sn).Value.ToMatrix().AsSpan(), context);
    }

    internal static LinearAlgebra.Vector<double> Vector(double[] values) => new(values);

    internal static Matrix<double> Matrix(double[] values, int rows, int columns)
    {
        var result = new Matrix<double>(rows, columns);
        var span = result.AsSpan();
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = values[i % values.Length];
        }

        return result;
    }
}
