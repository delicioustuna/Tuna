using System.Numerics;
using System.Runtime.InteropServices;
using CsCheck;

namespace Tuna.Tests.LinearAlgebra;

using Tuna.LinearAlgebra;

public class WedgeProductTests
{
    [Fact]
    public void WedgeIsAlternatingAndBilinear()
    {
        Gen.Select(Gen.Int[0, 12], Gen.Int[0, 100000]).Sample((n, seed) =>
        {
            var random = new Random(seed);
            var u = Vector<int>.Generate(n, _ => random.Next(-8, 9));
            var v = Vector<int>.Generate(n, _ => random.Next(-8, 9));
            var w = Vector<int>.Generate(n, _ => random.Next(-8, 9));
            var uv = XMath.Wedge(u, v);
            var vu = XMath.Wedge(v, u);

            Assert.Equal(n, uv.Rows);
            Assert.Equal(n, uv.Columns);
            for (int j = 0; j < n; j++)
            {
                Assert.Equal(0, uv[j, j]);
                for (int i = 0; i < n; i++)
                {
                    Assert.Equal(u[i] * v[j] - v[i] * u[j], uv[i, j]);
                    Assert.Equal(-uv[j, i], uv[i, j]);
                    Assert.Equal(-vu[i, j], uv[i, j]);
                }
            }

            Assert.Equal(new int[n, n], XMath.Wedge(u, u).ToArray());
            Assert.Equal((uv + XMath.Wedge(w, v)).ToArray(), XMath.Wedge(u + w, v).ToArray());
            Assert.Equal((uv + XMath.Wedge(u, w)).ToArray(), XMath.Wedge(u, v + w).ToArray());
        }, iter: 200, seed: "1760c2055002");
    }

    [Fact]
    public void WedgeWritesOnlyTheTransposedDestinationWithADisjointStridedInput()
    {
        var storage = Matrix<double>.Generate(4, 5, (i, j) => i == 0 ? j : -100);
        var u = storage.Row(0)[1..4].AsReadOnly();
        var v = new Vector<double>([4, 5, 6]);
        var destination = storage[1..4, 1..4].T;

        var result = XMath.Wedge(u, v, destination);

        Assert.Equal(new double[,] { { 0, -3, -6 }, { 3, 0, -3 }, { 6, 3, 0 } }, result.ToArray());
        Assert.Equal([0.0, 1, 2, 3, 4], storage.Row(0).ToArray());
        Assert.Equal([0.0, -100, -100, -100], storage.Column(0).ToArray());
        Assert.Equal([4.0, -100, -100, -100], storage.Column(4).ToArray());
        Assert.Equal(3, storage[1, 2]);

        var cross = XMath.Cross(u, v);
        Assert.Equal([cross[0], cross[1], cross[2]], new[] { result[1, 2], result[2, 0], result[0, 1] });
    }

    [Fact]
    public void WedgeRejectsInvalidLengthsShapesAndOverlapBeforeWriting()
    {
        var u = new Vector<double>([1, 2]);
        var shorter = new Vector<double>([3]);
        var destination = new Matrix<double>(new double[,] { { 1, 2 }, { 3, 4 } });
        var wrongShape = new Matrix<double>(2, 1);

        Assert.Throws<ArgumentException>("v", () => XMath.Wedge(u, shorter));
        Assert.Throws<ArgumentException>("v", () => XMath.Wedge(u, shorter, destination));
        Assert.Throws<ArgumentException>("destination", () => XMath.Wedge(u, u, wrongShape));
        Assert.Throws<ArgumentException>("destination", () => { XMath.Wedge(destination.Row(0), u, destination.T); });
        Assert.Throws<ArgumentException>("destination", () => { XMath.Wedge(u, destination.Column(1), destination); });
        Assert.Equal(new double[,] { { 1, 2 }, { 3, 4 } }, destination.ToArray());
    }

    [Fact]
    public void WedgeHandlesEmptyAndOversizedResults()
    {
        var empty = new Vector<double>(0);
        var result = XMath.Wedge(empty, empty);
        Assert.Equal(0, result.Rows);
        Assert.Equal(0, result.Columns);
        Assert.Same(result, XMath.Wedge(empty, empty, result));
        Assert.Throws<ArgumentException>("destination", () => XMath.Wedge(empty, empty, new Matrix<double>(0, 1)));

        var large = new Vector<double>(46_341);
        Assert.Throws<OverflowException>(() => XMath.Wedge(large, large));
    }

    [Fact]
    public void WedgeUsesOrdinaryComplexProducts()
    {
        Complex i = Complex.ImaginaryOne;
        var u = new Vector<Complex>([i, 1]);
        var v = new Vector<Complex>([1, i]);

        Assert.Equal(new Complex[,] { { 0, -2 }, { 2, 0 } }, XMath.Wedge(u, v).ToArray());
    }

    [Fact]
    public void WedgePreservesCancellationAndSubnormalResults()
    {
        CheckPrecision<double>(27, 520);
        CheckPrecision<float>(13, 70);
        CheckPrecision<Half>(6, 10);
    }

    [Fact]
    public void DenseAndStridedWedgeAgreeAcrossBlockBoundaries()
    {
        CheckLayoutAgreement<double>(520);
        CheckLayoutAgreement<float>(70);
        CheckLayoutAgreement<Half>(10);
    }

    private static void CheckLayoutAgreement<T>(int largeExponent) where T : unmanaged, IFloatingPointIeee754<T>
    {
        const int n = 37;
        T large = T.ScaleB(T.One, largeExponent);
        T[] values = [T.Zero, T.NegativeZero, T.One, T.BitIncrement(T.One), -T.Epsilon, T.Epsilon,
            large, -large, T.BitDecrement(large), T.NaN, T.PositiveInfinity, T.NegativeInfinity];
        var u = Vector<T>.Generate(n, i => values[i % values.Length]);
        var v = Vector<T>.Generate(n, i => values[(5 * i + 3) % values.Length]);
        var strided = Matrix<T>.Generate(2, n, (i, j) => i == 0 ? u[j] : v[j]);
        var denseResult = XMath.Wedge(u, v).AsSpan();
        var stridedResult = XMath.Wedge(strided.Row(0), strided.Row(1)).AsSpan();

        for (int i = 0; i < denseResult.Length; i++)
        {
            if (T.IsNaN(denseResult[i]) && T.IsNaN(stridedResult[i]))
            {
                denseResult[i] = stridedResult[i] = T.NaN;
            }
        }

        Assert.Equal(MemoryMarshal.AsBytes(stridedResult).ToArray(), MemoryMarshal.AsBytes(denseResult).ToArray());
    }

    private static void CheckPrecision<T>(int precision, int largeExponent) where T : unmanaged, IFloatingPointIeee754<T>
    {
        double delta = Math.ScaleB(1, -precision);
        foreach (int exponent in new[] { 0, largeExponent })
        {
            T scale = T.CreateChecked(Math.ScaleB(1, exponent));
            var u = new Vector<T>([T.CreateChecked(Math.ScaleB(1 + delta, exponent)), scale]);
            var v = new Vector<T>([scale, T.CreateChecked(Math.ScaleB(1 - delta, exponent))]);
            T expected = T.CreateChecked(-Math.ScaleB(1, 2 * (exponent - precision)));

            Assert.Equal(new T[,] { { T.Zero, expected }, { -expected, T.Zero } }, XMath.Wedge(u, v).ToArray());
        }

        T half = T.CreateChecked(0.5);
        var tiny = XMath.Wedge(new Vector<T>([T.Epsilon, -T.Epsilon]), new Vector<T>([half, half]));
        Assert.Equal(T.Epsilon, tiny[0, 1]);
        Assert.Equal(-T.Epsilon, tiny[1, 0]);
    }

    [Fact]
    public void WedgeFollowsFloatingPointOperatorsForNonFiniteOperands()
    {
        var u = new Vector<double>([double.PositiveInfinity, 1, 0, double.NaN]);
        var v = new Vector<double>([2, 3, 0, 4]);
        var result = XMath.Wedge(u, v);

        Assert.True(double.IsNaN(result[0, 0]));
        Assert.True(double.IsNaN(result[3, 3]));
        Assert.Equal(0, BitConverter.DoubleToInt64Bits(result[1, 1]));
        Assert.Equal(double.PositiveInfinity, result[0, 1]);
        Assert.Equal(double.NegativeInfinity, result[1, 0]);
        Assert.True(double.IsNaN(result[0, 2]));
        Assert.True(double.IsNaN(result[2, 0]));
        Assert.True(double.IsNaN(result[1, 3]));
    }
}
