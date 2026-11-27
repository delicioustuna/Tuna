using System.Numerics;

namespace Tuna.Tests.LinearAlgebra;

using Tuna.LinearAlgebra;

public class CrossProductTests
{
    [Fact]
    public void CrossSupportsIntegerContainersAndInPlaceDestination()
    {
        var u = new Vector<int>([1, 2, 3]);
        var v = new Vector<int>([4, 5, 6]);

        Assert.Equal([-3, 6, -3], XMath.Cross(u, v).ToArray());
        Assert.Same(u, XMath.Cross(u, v, u));
        Assert.Equal([-3, 6, -3], u.ToArray());
        Assert.Equal([4, 5, 6], v.ToArray());
    }

    [Fact]
    public void CrossWritesOnlyTheStridedDestinationWhenItOverlapsBothInputs()
    {
        double[] storage = [1, 101, 2, 102, 3, 103, 4, 104, 5, 105, 6];
        var u = new ReadOnlyVectorSpan<double>(storage, [3], [2]);
        var v = new ReadOnlyVectorSpan<double>(storage.AsSpan(6), [3], [2]);
        var destination = new VectorSpan<double>(storage.AsSpan(2), [3], [2]);

        var result = XMath.Cross(u, v, destination);

        Assert.Equal([-3.0, 6.0, -3.0], result.ToArray());
        Assert.Equal([1.0, 101.0, -3.0, 102.0, 6.0, 103.0, -3.0, 104.0, 5.0, 105.0, 6.0], storage);
    }

    [Fact]
    public void CrossUsesOrdinaryComplexProducts()
    {
        Complex i = Complex.ImaginaryOne;
        var u = new Vector<Complex>([i, 1, -i]);
        var v = new Vector<Complex>([1, i, 2]);

        Assert.Equal([1, -3 * i, -2], XMath.Cross(u, v).ToArray());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(2)]
    [InlineData(4)]
    public void CrossRejectsInvalidLengthsBeforeWriting(int length)
    {
        var u = new Vector<double>([1, 2, 3]);
        var invalid = new Vector<double>(length);
        var destination = new Vector<double>([10, 20, 30]);

        Assert.Throws<ArgumentException>("u", () => XMath.Cross(invalid, u));
        Assert.Throws<ArgumentException>("v", () => XMath.Cross(u, invalid, destination));
        Assert.Throws<ArgumentException>("destination", () => XMath.Cross(u, u, invalid));
        Assert.Equal([10.0, 20.0, 30.0], destination.ToArray());
        Assert.Equal(new double[length], invalid.ToArray());
    }

    [Fact]
    public void CrossPreservesSmallDifferencesBetweenLargeProducts()
    {
        CheckCancellation<double>(27, 0);
        CheckCancellation<double>(27, 520);
        CheckCancellation<float>(13, 0);
        CheckCancellation<float>(13, 70);
        CheckCancellation<Half>(6, 0);
        CheckCancellation<Half>(6, 10);
    }

    [Fact]
    public void CrossPreservesSubnormalResults()
    {
        CheckSubnormal<double>();
        CheckSubnormal<float>();
        CheckSubnormal<Half>();
    }

    [Fact]
    public void CrossFollowsFloatingPointOperatorsForNonFiniteOperands()
    {
        var u = new Vector<double>([double.PositiveInfinity, 0, 1]);
        var v = new Vector<double>([2, 3, 0]);
        var result = XMath.Cross(u, v);

        Assert.Equal(-3, result[0]);
        Assert.True(double.IsNaN(result[1]));
        Assert.Equal(double.PositiveInfinity, result[2]);

        var narrow = XMath.Cross(new Vector<float>([float.MaxValue, float.PositiveInfinity, 0]),
            new Vector<float>([1, float.MaxValue, 0]));
        Assert.True(float.IsNaN(narrow[2]));

        u[0] = double.NaN;
        Assert.True(double.IsNaN(XMath.Cross(u, v)[2]));
    }

    private static void CheckCancellation<T>(int precision, int exponent) where T : unmanaged, IFloatingPointIeee754<T>
    {
        double delta = Math.ScaleB(1, -precision);
        T scale = T.CreateChecked(Math.ScaleB(1, exponent));
        var u = new Vector<T>([T.CreateChecked(Math.ScaleB(1 + delta, exponent)), scale, T.Zero]);
        var v = new Vector<T>([scale, T.CreateChecked(Math.ScaleB(1 - delta, exponent)), T.Zero]);
        T expected = T.CreateChecked(-Math.ScaleB(1, 2 * (exponent - precision)));

        Assert.Equal([T.Zero, T.Zero, expected], XMath.Cross(u, v).ToArray());
    }

    private static void CheckSubnormal<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        T half = T.CreateChecked(0.5);
        var u = new Vector<T>([T.Epsilon, -T.Epsilon, T.Zero]);
        var v = new Vector<T>([half, half, T.Zero]);

        Assert.Equal(T.Epsilon, XMath.Cross(u, v)[2]);
    }
}
