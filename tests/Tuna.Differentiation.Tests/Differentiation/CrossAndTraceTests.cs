using System.Numerics;
using Tuna.Differentiation;

namespace Tuna.Tests.Differentiation;

using Tuna.LinearAlgebra;

public class CrossAndTraceTests
{
    [Fact]
    public void CrossPullbackAccumulatesIntoTheOriginalVectorSlices()
    {
        CheckCrossPullback<double>();
        CheckCrossPullback<float>();
    }

    private static void CheckCrossPullback<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        var tape = new Tape();
        var u = tape.Leaf(Vector<T>.Generate(5, i => T.CreateChecked(i)));
        var v = tape.Leaf(Vector<T>.Generate(4, i => T.CreateChecked(i + 4)));
        var dz = tape.Constant(new Vector<T>([T.CreateChecked(2), T.CreateChecked(5), -T.One]));
        var cross = XMath.Cross(u[1..4], v[..3]);

        Assert.Equal([-3.0, 6.0, -3.0], cross.Value.ToArray().Select(double.CreateTruncating));
        tape.Backward(XMath.Dot(cross, dz));
        Assert.Equal([0.0, -35.0, 16.0, 10.0, 0.0], tape.GetGradient(u).ToArray().Select(double.CreateTruncating));
        Assert.Equal([17.0, -7.0, -1.0, 0.0], tape.GetGradient(v).ToArray().Select(double.CreateTruncating));
    }

    [Fact]
    public void CrossAccumulatesBothContributionsWhenTheInputsShareALeaf()
    {
        var tape = new Tape();
        var u = tape.Leaf(new Vector<double>([1, 2, 3]));
        var constant = tape.Constant(new Vector<double>([1, 0, 0]));

        Assert.False(XMath.Cross(constant, constant).RequiresGradient);
        tape.Backward(XMath.Sum(XMath.Cross(u, u)));
        Assert.Equal([0.0, 0.0, 0.0], tape.GetGradient(u).ToArray());
    }

    [Fact]
    public void CrossRejectsInvalidLengthsAndForeignNodes()
    {
        var tape = new Tape();
        var u = tape.Leaf(new Vector<double>([1, 2, 3]));
        var invalid = tape.Leaf(new Vector<double>(2));
        var foreign = new Tape().Leaf(new Vector<double>(3));

        Assert.Throws<ArgumentException>("u", () => XMath.Cross(invalid, u));
        Assert.Throws<ArgumentException>("v", () => XMath.Cross(u, invalid));
        Assert.Throws<InvalidOperationException>(() => XMath.Cross(u, foreign));

        tape.Backward(XMath.Sum(XMath.Cross(u, u)));
        Assert.Equal([0.0, 0.0, 0.0], tape.GetGradient(u).ToArray());
    }

    [Fact]
    public void TracePullbackOnlyUpdatesTheDiagonalOfEachView()
    {
        var tape = new Tape();
        var matrix = Matrix<float>.Generate(4, 5, (i, j) => 10 * i + j);
        var a = tape.Leaf(matrix);
        var trace = XMath.Trace(a[1..4, 2..4].T);

        Assert.Equal(35.0, trace.Value);
        tape.Backward(2 * trace + 3 * XMath.Trace(a));
        var expected = Matrix<float>.Generate(4, 5, (i, j) => i == j ? 3 : 0);
        expected[1, 2] = 2;
        expected[2, 3] = 2;
        Assert.Equal(expected.ToArray(), tape.GetGradient(a).ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TraceOfABroadcastMatrixAccumulatesOncePerDiagonalEntry(bool transpose)
    {
        var tape = new Tape();
        var x = tape.Leaf(new Vector<double>([1, 2, 3, 4]));
        var matrix = x.Broadcast(2);
        var trace = XMath.Trace(transpose ? matrix.T : matrix);

        Assert.Equal(3, trace.Value);
        tape.Backward(5 * trace);
        Assert.Equal([5.0, 5.0, 0.0, 0.0], tape.GetGradient(x).ToArray());
    }

    [Fact]
    public void TraceOfAnEmptySliceContributesZeroGradient()
    {
        var tape = new Tape();
        var a = tape.Leaf(new Matrix<double>(2, 3));
        var trace = XMath.Trace(a[1..1, ..]);

        Assert.Equal(0, trace.Value);
        tape.Backward(trace);
        Assert.Equal(new double[2, 3], tape.GetGradient(a).ToArray());
    }
}
