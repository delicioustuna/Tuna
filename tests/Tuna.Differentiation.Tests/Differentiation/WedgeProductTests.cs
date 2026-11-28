using System.Numerics;
using Tuna.Differentiation;

namespace Tuna.Tests.Differentiation;

using Tuna.LinearAlgebra;

public class WedgeProductTests
{
    [Fact]
    public void WedgeBackpropagatesAnAsymmetricAdjointIntoBothSlices()
    {
        CheckPullback<double>();
        CheckPullback<float>();
    }

    private static void CheckPullback<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        var tape = new Tape();
        var u = tape.Leaf(Vector<T>.Generate(5, i => T.CreateChecked(i)));
        var v = tape.Leaf(Vector<T>.Generate(4, i => T.CreateChecked(i + 4)));
        int[,] weights = { { 100, 2, -3 }, { 7, 200, 5 }, { -1, -2, 300 } };
        var g = Matrix<T>.Generate(3, 3, (i, j) => T.CreateChecked(weights[i, j]));
        var wedge = XMath.Wedge(u[1..4], v[..3]);

        Assert.Equal(XMath.Wedge(u[1..4].Value, v[..3].Value).ToArray(), wedge.Value.ToArray());
        tape.Backward(XMath.Sum(XMath.ColumnSums(XMath.Hadamard(wedge, tape.Constant(g)))));
        Assert.Equal([0.0, -37.0, 62.0, -27.0, 0.0], tape.GetGradient(u).ToArray().Select(double.CreateTruncating));
        Assert.Equal([16.0, -26.0, 12.0, 0.0], tape.GetGradient(v).ToArray().Select(double.CreateTruncating));
        Assert.Equal(g.ToArray(), tape.GetGradient(wedge).ToArray());
    }

    [Fact]
    public void WedgeCancelsTheSymmetricAdjointBeforeMultiplying()
    {
        var tape = new Tape();
        var u = tape.Leaf(new Vector<double>(3));
        var v = tape.Constant(new Vector<double>([2, 3, 4]));
        double large = double.MaxValue;
        var g = tape.Constant(new Matrix<double>(new double[,] { { large, large, 1 }, { large, large, 0 }, { 0, 0, large } }));
        var wedge = XMath.Wedge(u, v);

        tape.Backward(XMath.Sum(XMath.ColumnSums(XMath.Hadamard(wedge, g))));
        Assert.Equal([4.0, 0.0, -2.0], tape.GetGradient(u).ToArray());
    }

    [Fact]
    public void WedgeWithOneLeafRecordsGradientsInEitherOperand()
    {
        var tape = new Tape();
        var u = tape.Constant(new Vector<double>([1, 2]));
        var v = tape.Leaf(new Vector<double>([3, 4]));
        var uv = XMath.Wedge(u, v);
        var vu = XMath.Wedge(v, u);

        Assert.False(XMath.Wedge(u, u).RequiresGradient);
        tape.Backward(XMath.Trace(uv[..1, 1..]) + 2 * XMath.Trace(vu[..1, 1..]));
        Assert.Equal([2.0, -1.0], tape.GetGradient(v).ToArray());
    }

    [Fact]
    public void WedgeAccumulatesOverlappingSlicesAndCancelsIdenticalSlices()
    {
        var tape = new Tape();
        var x = tape.Leaf(new Vector<double>([1, 2, 3]));
        var same = XMath.Wedge(x[..2], x[..2]);
        var overlapping = XMath.Wedge(x[..2], x[1..]);

        tape.Backward(1e100 * XMath.Trace(same[..1, 1..]) + XMath.Trace(overlapping[..1, 1..]));
        Assert.Equal([3.0, -4.0, 1.0], tape.GetGradient(x).ToArray());
    }

    [Fact]
    public void WedgeOfTheSameParameterPreservesItsAccumulatedGradient()
    {
        var parameter = Parameter<double>.Vector(2);
        parameter.Value[0] = 1;
        parameter.Value[1] = 2;
        parameter.Gradient.Fill(1);
        var tape = new Tape();
        var wedge = XMath.Wedge(tape.UseVector(parameter), tape.UseVector(parameter));

        tape.Backward(1e100 * XMath.Trace(wedge[..1, 1..]));
        Assert.Equal([1.0, 1.0], parameter.Gradient.ToArray());
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void WedgeOfTheSameParameterPropagatesNonFiniteInput(double nonFinite)
    {
        var parameter = Parameter<double>.Vector(2);
        parameter.Value[0] = nonFinite;
        parameter.Value[1] = 2;
        parameter.Gradient.Fill(1);
        var tape = new Tape();
        var wedge = XMath.Wedge(tape.UseVector(parameter), tape.UseVector(parameter));

        tape.Backward(1e100 * XMath.Trace(wedge[..1, 1..]));
        Assert.Equal(1, parameter.Gradient[0]);
        Assert.True(double.IsNaN(parameter.Gradient[1]));
    }

    [Theory]
    [InlineData(double.NaN, false)]
    [InlineData(double.PositiveInfinity, true)]
    [InlineData(double.NegativeInfinity, false)]
    public void WedgeOfTheSameInputPropagatesNonFiniteOffDiagonalAdjoint(double adjoint, bool lower)
    {
        var tape = new Tape();
        var u = tape.Leaf(new Vector<double>([1, 2]));
        var wedge = XMath.Wedge(u, u);

        var entry = lower ? wedge[1.., ..1] : wedge[..1, 1..];
        tape.Backward(adjoint * XMath.Trace(entry));
        Assert.All(tape.GetGradient(u).ToArray(), value => Assert.True(double.IsNaN(value)));
    }

    [Fact]
    public void WedgeUsesOnlyOffDiagonalAdjoints()
    {
        var tape = new Tape();
        var u = tape.Leaf(new Vector<double>([1, 2]));
        var v = tape.Leaf(new Vector<double>([3, 4]));
        var g = tape.Constant(Matrix<double>.FromRowMajor(2, 2, [double.NaN, 1, 0, double.PositiveInfinity]));
        var wedge = XMath.Wedge(u, v);

        tape.Backward(XMath.Sum(XMath.ColumnSums(XMath.Hadamard(wedge, g))));
        Assert.Equal([4.0, -3.0], tape.GetGradient(u).ToArray());
        Assert.Equal([-2.0, 1.0], tape.GetGradient(v).ToArray());
    }

    [Fact]
    public void UnreachableWedgeWithNonFiniteInputsLeavesGradientsIntact()
    {
        var tape = new Tape();
        var u = tape.Leaf(new Vector<double>([1, 2, 3]));
        var v = tape.Leaf(new Vector<double>([double.PositiveInfinity, double.NaN, double.NegativeInfinity]));
        var wedge = XMath.Wedge(u, v);

        Assert.True(double.IsNaN(wedge.Value[0, 1]));
        tape.Backward(XMath.Sum(u));
        Assert.Equal([1.0, 1.0, 1.0], tape.GetGradient(u).ToArray());
        Assert.Equal(new double[3], tape.GetGradient(v).ToArray());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void WedgeOfEmptyOrSingleElementVectorsHasZeroGradient(int n)
    {
        var tape = new Tape();
        var u = tape.Leaf(Vector<double>.Generate(n, i => i + 1));
        var v = tape.Leaf(Vector<double>.Generate(n, i => i + 2));
        var wedge = XMath.Wedge(u, v);

        Assert.Equal(new double[n, n], wedge.Value.ToArray());
        tape.Backward(XMath.Trace(wedge));
        Assert.Equal(new double[n], tape.GetGradient(u).ToArray());
        Assert.Equal(new double[n], tape.GetGradient(v).ToArray());
    }

    [Fact]
    public void WedgeRejectsInvalidLengthsForeignNodesAndStaleNodes()
    {
        var tape = new Tape();
        var u = tape.Leaf(new Vector<double>(2));
        var invalid = tape.Leaf(new Vector<double>(1));
        var foreign = new Tape().Leaf(new Vector<double>(2));

        Assert.Throws<ArgumentException>("v", () => XMath.Wedge(u, invalid));
        Assert.Throws<InvalidOperationException>(() => XMath.Wedge(u, foreign));
        Assert.Throws<InvalidOperationException>(() => XMath.Wedge(u, default));

        tape.Backward(XMath.Trace(XMath.Wedge(u, u)));
        Assert.Throws<InvalidOperationException>(() => XMath.Wedge(u, u));
        tape.Reset();
        Assert.Throws<InvalidOperationException>(() => XMath.Wedge(u, u));
    }
}
