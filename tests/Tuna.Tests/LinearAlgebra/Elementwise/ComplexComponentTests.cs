using System.Numerics;
using System.Runtime.InteropServices;

namespace Tuna.Tests.LinearAlgebra;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

/// <summary>要素型の変わる演算 Real・Imaginary・Magnitude・Phase（Complex → double）</summary>
[Trait("LinearAlgebra", "XMath")]
public class ComplexComponentTests
{
    private static readonly double Tiny = double.Epsilon;

    // 符号付きの 0、溢れ・アンダーフローの境界、非正規化数、∞・NaN を含む
    private static readonly Complex[] Special =
    [
        new(0, 0), new(-0.0, 0), new(0, -0.0), new(-0.0, -0.0),
        new(3, 4), new(-3, 4), new(3, -4), new(-1, 0), new(-1, -0.0),
        new(1e308, 1e308), new(-1e308, 1e308), new(double.MaxValue, double.MaxValue),
        new(1e-308, 1e-308), new(3 * Tiny, 4 * Tiny), new(Tiny, Tiny), new(1, 1e-300), new(1e300, 1),
        new(double.PositiveInfinity, 0), new(0, double.NegativeInfinity),
        new(double.PositiveInfinity, double.NaN), new(double.NaN, double.NegativeInfinity),
        new(double.NaN, 1), new(1, double.NaN), new(double.NaN, double.NaN),
        new(double.PositiveInfinity, double.PositiveInfinity), new(double.NegativeInfinity, double.NegativeInfinity),
    ];

    public static TheoryData<string> Functions => new() { "Real", "Imaginary", "Magnitude", "Phase" };

    private static double Scalar(string function, Complex z) => function switch
    {
        "Real" => z.Real,
        "Imaginary" => z.Imaginary,
        "Magnitude" => z.Magnitude,
        _ => z.Phase,
    };

    private static Vector<double> Allocate(string function, ReadOnlyVectorSpan<Complex> x) => function switch
    {
        "Real" => XMath.Real(x),
        "Imaginary" => XMath.Imaginary(x),
        "Magnitude" => XMath.Magnitude(x),
        _ => XMath.Phase(x),
    };

    private static VectorSpan<double> Into(string function, ReadOnlyVectorSpan<Complex> x, VectorSpan<double> destination) => function switch
    {
        "Real" => XMath.Real(x, destination),
        "Imaginary" => XMath.Imaginary(x, destination),
        "Magnitude" => XMath.Magnitude(x, destination),
        _ => XMath.Phase(x, destination),
    };

    private static Matrix<double> Allocate(string function, ReadOnlyMatrixSpan<Complex> x) => function switch
    {
        "Real" => XMath.Real(x),
        "Imaginary" => XMath.Imaginary(x),
        "Magnitude" => XMath.Magnitude(x),
        _ => XMath.Phase(x),
    };

    private static MatrixSpan<double> Into(string function, ReadOnlyMatrixSpan<Complex> x, MatrixSpan<double> destination) => function switch
    {
        "Real" => XMath.Real(x, destination),
        "Imaginary" => XMath.Imaginary(x, destination),
        "Magnitude" => XMath.Magnitude(x, destination),
        _ => XMath.Phase(x, destination),
    };

    private static void AssertBits(double expected, double actual, string context) =>
        Assert.True(BitConverter.DoubleToInt64Bits(expected) == BitConverter.DoubleToInt64Bits(actual),
            $"{context}: expected {expected:R}, actual {actual:R}");

    [Theory]
    [MemberData(nameof(Functions))]
    public void VectorResultsMatchScalarPropertiesBitForBit(string function)
    {
        var x = new Vector<Complex>(Special);
        var allocated = Allocate(function, x);
        var destination = new Vector<double>(Special.Length);
        var returned = Into(function, x, destination);
        Assert.Equal(Special.Length, allocated.Length);
        for (int i = 0; i < Special.Length; i++)
        {
            AssertBits(Scalar(function, Special[i]), allocated[i], $"{function}({Special[i]})");
            AssertBits(allocated[i], destination[i], $"{function}({Special[i]}) destination");
            AssertBits(allocated[i], returned[i], $"{function}({Special[i]}) returned");
        }
    }

    [Fact]
    public void MagnitudeAvoidsOverflowAndUnderflowAndFollowsIeeeSpecialCases()
    {
        var m = XMath.Magnitude(new Vector<Complex>(
        [
            new(3, 4), new(1e308, 1e308), new(3 * Tiny, 4 * Tiny), new(1e-200, 1e-200),
            new(double.PositiveInfinity, double.NaN), new(double.NaN, double.NegativeInfinity), new(double.NaN, 1), new(-0.0, -0.0),
        ]));
        Assert.Equal(5, m[0]);
        AssertBits(1e308 * Math.Sqrt(2), m[1], "|1e308 + 1e308i|");
        Assert.Equal(5 * Tiny, m[2]);
        AssertBits(1e-200 * Math.Sqrt(2), m[3], "|1e-200 + 1e-200i|");
        Assert.Equal(double.PositiveInfinity, m[4]);
        Assert.Equal(double.PositiveInfinity, m[5]);
        Assert.True(double.IsNaN(m[6]));
        AssertBits(0.0, m[7], "|-0 - 0i|");
    }

    /// <summary>
    /// 大きさの SIMD 版（DenseElementwise.Magnitude.cs）は double.Hypot の分岐をレーンの選択にしたもので、Complex.Magnitude とビット単位で一致する。
    /// 分岐の境目（バイアス付き指数 523・1523 の前後、指数の差 54・55、指数が等しい組、非正規化数、0・∞・NaN との組）を総当たりで組み合わせ、
    /// SIMD の本体と端数の両方を通す。列が連続した部分行列（列ごとに SIMD）も確かめる
    /// </summary>
    [Fact]
    public void MagnitudeMatchesComplexMagnitudeBitwiseAcrossScalingBoundaries()
    {
        var random = new Random(17);
        var parts = new List<double> { 0, -0.0, Tiny, 3 * Tiny, 2.2250738585072014e-308, double.MaxValue, double.PositiveInfinity, double.NegativeInfinity, double.NaN, 1, -1 };
        foreach (int exponent in new[] { 1, 2, 52, 53, 54, 55, 56, 107, 108, 522, 523, 524, 577, 578, 1022, 1023, 1024, 1468, 1469, 1522, 1523, 1524, 2045, 2046 })
        {
            parts.Add(Math.ScaleB(1 + random.NextDouble(), exponent - 1023));
            parts.Add(-Math.ScaleB(1 + random.NextDouble(), exponent - 1023));
        }

        // 非正規化数の各桁
        for (int bit = 0; bit < 52; bit += 7)
        {
            parts.Add(BitConverter.Int64BitsToDouble((1L << bit) | (long)random.Next()));
        }

        var values = new List<Complex>();
        foreach (double re in parts)
        {
            foreach (double im in parts)
            {
                values.Add(new Complex(re, im));
            }
        }

        // 指数の差を -60〜60 で選んだ乱数の組
        for (int i = 0; i < 20_000; i++)
        {
            double re = Math.ScaleB(random.NextDouble() + 0.5, random.Next(-1074, 1024));
            double im = Math.ScaleB(re * (random.NextDouble() + 0.5), random.Next(-60, 61)) * (random.Next(2) == 0 ? -1 : 1);
            values.Add(new Complex(re, im));
        }

        for (int shift = 0; shift < 8; shift++)
        {
            var x = values.Skip(shift).ToArray();
            var m = XMath.Magnitude(new Vector<Complex>(x));
            for (int i = 0; i < x.Length; i++)
            {
                AssertBits(x[i].Magnitude, m[i], $"|{x[i]}| (shift {shift})");
            }
        }

        int rows = 37, columns = values.Count / rows;
        var a = Matrix<Complex>.FromColumnMajor(rows, columns, values.Take(rows * columns).ToArray());
        var view = a[1..36, 2..];
        var result = XMath.Magnitude(view);
        for (int j = 0; j < view.Columns; j++)
        {
            for (int i = 0; i < view.Rows; i++)
            {
                AssertBits(view[i, j].Magnitude, result[i, j], $"submatrix [{i}, {j}] = |{view[i, j]}|");
            }
        }
    }

    [Fact]
    public void PhaseDistinguishesSignedZeros()
    {
        var p = XMath.Phase(new Vector<Complex>([new(-1, 0), new(-1, -0.0), new(0, 0), new(-0.0, 0), new(0, -1)]));
        Assert.Equal(Math.PI, p[0]);
        Assert.Equal(-Math.PI, p[1]);
        AssertBits(0.0, p[2], "phase(+0 + 0i)");
        Assert.Equal(Math.PI, p[3]);
        Assert.Equal(-Math.PI / 2, p[4]);
    }

    [Theory]
    [MemberData(nameof(Functions))]
    public void MatrixViewsKeepShapeAndPreserveDestinationPadding(string function)
    {
        var values = Enumerable.Range(0, 20).Select(i => new Complex(i - 7.5, 0.25 * i * i - 3)).ToArray();
        var a = Matrix<Complex>.FromColumnMajor(4, 5, values);
        for (int k = 0; k < 4; k++)
        {
            // 連続・転置・部分行列・転置の部分行列
            ReadOnlyMatrixSpan<Complex> view = k switch { 0 => a, 1 => a.T, 2 => a[1..3, 1..5], _ => a.T[1..4, 0..2] };
            var expected = Allocate(function, view);
            Assert.Equal((view.Rows, view.Columns), (expected.Rows, expected.Columns));

            // 出力先は1つおきの要素（間の要素は書き換えない）と、転置したビュー
            var storage = Enumerable.Repeat(-99.0, 2 * view.Rows * view.Columns).ToArray();
            var strided = new MatrixSpan<double>(storage, [view.Rows, view.Columns], [2, 2 * view.Rows]);
            Into(function, view, strided);
            var transposedStorage = new double[view.Rows * view.Columns];
            var transposed = new MatrixSpan<double>(transposedStorage, view.Columns, view.Rows).T;
            Into(function, view, transposed);
            for (int j = 0; j < view.Columns; j++)
            {
                for (int i = 0; i < view.Rows; i++)
                {
                    AssertBits(Scalar(function, view[i, j]), expected[i, j], $"{function} view {k} [{i}, {j}]");
                    AssertBits(expected[i, j], strided[i, j], $"{function} view {k} strided [{i}, {j}]");
                    AssertBits(expected[i, j], transposed[i, j], $"{function} view {k} transposed [{i}, {j}]");
                }
            }

            for (int i = 1; i < storage.Length; i += 2)
            {
                Assert.Equal(-99.0, storage[i]);
            }
        }
    }

    [Fact]
    public void RowColumnAndDiagonalViewsWorkAsVectors()
    {
        var a = Matrix<Complex>.FromColumnMajor(3, 3, Enumerable.Range(0, 9).Select(i => new Complex(i, -i)).ToArray());
        Assert.Equal(new[] { 1.0, 4, 7 }, XMath.Real(a.Row(1)).ToArray());
        Assert.Equal(new[] { 0.0, -4, -8 }, XMath.Imaginary(a.Diagonal()).ToArray());
        var result = new Matrix<double>(3, 2);
        XMath.Magnitude(a.Column(2), result.Column(1));
        Assert.Equal(new[] { 0.0, 0, 0 }, result.Column(0).ToArray());
        Assert.Equal(new[] { 6, 7, 8 }.Select(v => double.Hypot(v, v)), result.Column(1).ToArray());
    }

    [Theory]
    [MemberData(nameof(Functions))]
    public void EmptyInputsGiveEmptyResults(string function)
    {
        Assert.Equal(0, Allocate(function, new Vector<Complex>(0)).Length);
        var m = Allocate(function, new Matrix<Complex>(0, 3));
        Assert.Equal((0, 3), (m.Rows, m.Columns));
        m = Allocate(function, new Matrix<Complex>(2, 0));
        Assert.Equal((2, 0), (m.Rows, m.Columns));
    }

    [Theory]
    [MemberData(nameof(Functions))]
    public void ShapeMismatchesAreRejected(string function)
    {
        Assert.Throws<ArgumentException>(() => Into(function, new Vector<Complex>(3), new Vector<double>(4)));
        Assert.Throws<ArgumentException>(() => Into(function, new Matrix<Complex>(2, 3), new Matrix<double>(3, 2)));
    }

    [Theory]
    [MemberData(nameof(Functions))]
    public void DestinationsSharingTheInputBytesAreRejectedBeforeWriting(string function)
    {
        var storage = new Complex[] { new(1, 2), new(3, 4), new(5, 6), new(7, 8) };
        var original = storage.ToArray();

        // 実部の位置（double で 0・2・4・6 番目）を出力先にすると、入力の要素と重なる
        Assert.Throws<ArgumentException>(() => Into(function, new ReadOnlyVectorSpan<Complex>(storage),
            new VectorSpan<double>(MemoryMarshal.Cast<Complex, double>(storage.AsSpan()), [4], [2])));
        // 2×2 の行列を、同じ配列の前半の double へ書く
        Assert.Throws<ArgumentException>(() => Into(function, new MatrixSpan<Complex>(storage, 2, 2),
            new MatrixSpan<double>(MemoryMarshal.Cast<Complex, double>(storage.AsSpan()), 2, 2)));
        Assert.Equal(original, storage);
    }

    [Theory]
    [MemberData(nameof(Functions))]
    public void InterleavedViewsThatShareNoBytesAreAllowed(string function)
    {
        // 入力は複素数の 0・2 番目（double で 0〜1 と 4〜5 番目）。出力先が間の 2・6 番目か 3・7 番目なら重ならない
        var storage = new Complex[] { new(3, 4), new(-1, -1), new(-6, 8), new(-1, -1) };
        var expected = new[] { Scalar(function, storage[0]), Scalar(function, storage[2]) };
        foreach (int start in new[] { 2, 3 })
        {
            var doubles = MemoryMarshal.Cast<Complex, double>(storage.AsSpan());
            var destination = new VectorSpan<double>(doubles[start..], [2], [4]);
            Into(function, new ReadOnlyVectorSpan<Complex>(storage, [2], [2]), destination);
            Assert.Equal(expected, destination.ToArray());
            Assert.Equal(new Complex(3, 4), storage[0]);
            Assert.Equal(new Complex(-6, 8), storage[2]);
        }

        // 出力先の1つが入力の要素に入れば例外（double で 1 番目は第 0 要素の虚部、5 番目は第 2 要素の虚部）
        Assert.Throws<ArgumentException>(() => Into(function, new ReadOnlyVectorSpan<Complex>(storage, [2], [2]),
            new VectorSpan<double>(MemoryMarshal.Cast<Complex, double>(storage.AsSpan())[1..], [2], [4])));
    }
}
