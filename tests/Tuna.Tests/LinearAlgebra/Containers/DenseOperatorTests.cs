using System.Numerics;
using Tuna.LinearAlgebra;

namespace Tuna.Tests.LinearAlgebra;

public class DenseOperatorTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(257)]
    public void OperatorsMatchScalarArithmeticAcrossVectorWidths(int count)
    {
        Check<double>(count);
        Check<float>(count);
        Check<int>(count);
        Check<uint>(count);
        Check<byte>(count);
        Check<sbyte>(count);
        Check<short>(count);
        Check<ushort>(count);
        Check<long>(count);
        Check<ulong>(count);
        Check<Half>(count);
        Check<Int128>(count);
        Check<decimal>(count);
    }

    private static void Check<T>(int count) where T : unmanaged, INumberBase<T>
    {
        var a = Enumerable.Range(0, count).Select(i => T.CreateTruncating(i - 17)).ToArray();
        var b = Enumerable.Range(0, count).Select(i => T.CreateTruncating(23 - i)).ToArray();
        T scale = T.CreateChecked(3);
        var x = a.ToVector();
        var y = b.ToVector();
        var matrixA = x.AsMatrix(1, count);
        var matrixB = y.AsMatrix(1, count);
        var sum = a.Zip(b, (v, w) => unchecked(v + w)).ToArray();
        var difference = a.Zip(b, (v, w) => unchecked(v - w)).ToArray();
        var product = a.Select(v => unchecked(v * scale)).ToArray();
        Assert.Equal(sum, (x + y).ToArray());
        Assert.Equal(difference, (x - y).ToArray());
        Assert.Equal(product, (scale * x).ToArray());
        Assert.Equal(sum, (matrixA + matrixB).AsSpan().ToArray());
        Assert.Equal(difference, (matrixA - matrixB).AsSpan().ToArray());
        Assert.Equal(product, (matrixA * scale).AsSpan().ToArray());
        Assert.Equal(a, x.ToArray());
        Assert.Equal(b, y.ToArray());
    }

    [Fact]
    public void FloatingPointOperatorsPreserveSpecialValuesAndSignedZero()
    {
        double[] values = [0.0, -0.0, double.PositiveInfinity, double.NegativeInfinity,
            double.NaN, double.Epsilon, -double.Epsilon, double.MaxValue, -double.MaxValue];
        var a = values.SelectMany(x => values.Select(_ => x)).ToArray();
        var b = values.SelectMany(_ => values).ToArray();
        var x = a.ToVector();
        var y = b.ToVector();
        AssertBits(a.Zip(b, (v, w) => v + w).ToArray(), (x + y).ToArray());
        AssertBits(a.Zip(b, (v, w) => v - w).ToArray(), (x - y).ToArray());
        foreach (double scale in values)
        {
            AssertBits(a.Select(v => v * scale).ToArray(), (scale * x).ToArray());
        }
    }

    private static void AssertBits(double[] expected, double[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (double.IsNaN(expected[i]))
            {
                Assert.True(double.IsNaN(actual[i]));
            }
            else
            {
                Assert.Equal(BitConverter.DoubleToInt64Bits(expected[i]), BitConverter.DoubleToInt64Bits(actual[i]));
            }
        }
    }

    [Fact]
    public void ComplexAdditionAndSubtractionDoNotMixRealAndImaginaryParts()
    {
        Complex[] a = Enumerable.Range(0, 33).Select(i => new Complex(i, -2 * i)).ToArray();
        Complex[] b = Enumerable.Range(0, 33).Select(i => new Complex(-3 * i, i)).ToArray();
        a[0] = new Complex(double.PositiveInfinity, 2);
        b[0] = new Complex(1, double.PositiveInfinity);
        a[1] = new Complex(4, double.NaN);
        var x = a.ToVector();
        var y = b.ToVector();
        Assert.Equal(a.Zip(b, (v, w) => v + w), (x + y).ToArray());
        Assert.Equal(a.Zip(b, (v, w) => v - w), (x - y).ToArray());
        Assert.Equal(a.Zip(b, (v, w) => v + w), (x.AsMatrix(3, 11) + y.AsMatrix(3, 11)).AsSpan().ToArray());
        Assert.Equal(a.Zip(b, (v, w) => v - w), (x.AsMatrix(3, 11) - y.AsMatrix(3, 11)).AsSpan().ToArray());
        var finite = a.Skip(2).ToArray().ToVector();
        var scalar = new Complex(2, -1);
        Assert.Equal(finite.ToArray().Select(v => v * scalar), (finite * scalar).ToArray());
        Assert.Equal(finite.ToArray().Select(v => v * scalar), (finite.AsMatrix(1, finite.Length) * scalar).AsSpan().ToArray());
    }
}
