using System.Numerics;
using Tuna.Differentiation;

namespace Tuna.Tests.Differentiation;

using Tuna.LinearAlgebra;

public class HypotDerivativeTests
{
    [Theory]
    [InlineData(3e200, 4e200, 0.6, 0.8)]
    [InlineData(-3e-200, 4e-200, -0.6, 0.8)]
    [InlineData(double.MaxValue, -double.MaxValue, 0.7071067811865475, -0.7071067811865475)]
    [InlineData(double.Epsilon, -double.Epsilon, 0.7071067811865475, -0.7071067811865475)]
    [InlineData(-0.0, 0.0, 0.0, 0.0)]
    [InlineData(double.PositiveInfinity, 2.0, 1.0, 0.0)]
    [InlineData(2.0, double.NegativeInfinity, 0.0, -1.0)]
    [InlineData(double.NegativeInfinity, double.NaN, -1.0, 0.0)]
    [InlineData(double.NaN, double.PositiveInfinity, 0.0, 1.0)]
    [InlineData(double.PositiveInfinity, double.NegativeInfinity, double.NaN, double.NaN)]
    [InlineData(double.NaN, 2.0, double.NaN, double.NaN)]
    [InlineData(0.0, double.NaN, double.NaN, double.NaN)]
    public void ScalarPartialsFollowTheBoundaryRules(double a, double b, double dx, double dy)
    {
        var tape = new Tape();
        var x = tape.Leaf(a);
        var y = tape.Leaf(b);
        var result = XMath.Hypot(x, y);
        DifferentiationAssert.SameNumber(double.Hypot(a, b), result.Value);

        tape.Backward(result);
        AssertPartial(dx, tape.GetGradient(x), 1e-15);
        AssertPartial(dy, tape.GetGradient(y), 1e-15);
    }

    [Theory]
    [InlineData(3, 4)]
    [InlineData(-1.25, 0.625)]
    [InlineData(0, -3)]
    public void ScalarPartialsMatchFiniteDifferencesWithMixedOperands(double x, double y)
    {
        double dx = DifferentiationAssert.Gradient(a => XMath.Hypot(a, y), x);
        double dy = DifferentiationAssert.Gradient(b => XMath.Hypot(x, b), y);
        double expectedX = DifferentiationAssert.Richardson(a => double.Hypot(a, y), x);
        double expectedY = DifferentiationAssert.Richardson(b => double.Hypot(x, b), y);
        DifferentiationAssert.Close(expectedX, dx, 2e-11, 2e-11);
        DifferentiationAssert.Close(expectedY, dy, 2e-11, 2e-11);
    }

    [Fact]
    public void ArrayPartialsFollowTheBoundaryRulesInSimdAndScalarTails()
    {
        CheckArrayPartials<double>(700, 1e-15);
        CheckArrayPartials<float>(90, 2e-7);
    }

    private static void CheckArrayPartials<T>(int exponent, double tolerance)
        where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
    {
        T three = T.CreateChecked(3), four = T.CreateChecked(4);
        const double diagonal = 0.7071067811865475;
        var cases = new (T X, T Y, double Dx, double Dy)[]
        {
            (T.ScaleB(three, exponent), T.ScaleB(four, exponent), 0.6, 0.8),
            (T.ScaleB(three, -exponent), T.ScaleB(-four, -exponent), 0.6, -0.8),
            (T.MaxValue, -T.MaxValue, diagonal, -diagonal),
            (T.Epsilon, -T.Epsilon, diagonal, -diagonal),
            (T.NegativeZero, T.Zero, 0, 0),
            (T.PositiveInfinity, three, 1, 0),
            (four, T.NegativeInfinity, 0, -1),
            (T.NegativeInfinity, T.NaN, -1, 0),
            (T.NaN, T.PositiveInfinity, 0, 1),
            (T.PositiveInfinity, T.NegativeInfinity, double.NaN, double.NaN),
            (T.NaN, T.Zero, double.NaN, double.NaN)
        };
        const int length = 49;
        var a = Vector<T>.Generate(length, i => cases[i % cases.Length].X);
        var b = Vector<T>.Generate(length, i => cases[i % cases.Length].Y);
        var tape = new Tape();
        var x = tape.Leaf(a);
        var y = tape.Leaf(b);
        tape.Backward(XMath.Sum(XMath.Hypot(x, y)));
        var dx = tape.GetGradient(x);
        var dy = tape.GetGradient(y);
        var single = new Tape();
        for (int i = 0; i < length; i++)
        {
            var expected = cases[i % cases.Length];
            AssertPartial(expected.Dx, double.CreateChecked(dx[i]), tolerance);
            AssertPartial(expected.Dy, double.CreateChecked(dy[i]), tolerance);

            single.Reset();
            var xs = single.Leaf(a[i..(i + 1)]);
            var ys = single.Leaf(b[i..(i + 1)]);
            single.Backward(XMath.Sum(XMath.Hypot(xs, ys)));
            Assert.Equal(dx[i], single.GetGradient(xs)[0]);
            Assert.Equal(dy[i], single.GetGradient(ys)[0]);
        }
    }

    [Fact]
    public void ScalarOnTheLeftWorksForVectorsAndMatrices()
    {
        CheckScalarOnTheLeft<float>();
        CheckScalarOnTheLeft<double>();
    }

    private static void CheckScalarOnTheLeft<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        T three = T.CreateChecked(3);
        var a = new Vector<T>([three, -three]);
        var tape = new Tape();
        var x = tape.Leaf(a);
        var scalar = tape.Leaf(4.0);
        var vector = XMath.Hypot(scalar, x);
        var matrix = XMath.Hypot(scalar, x.AsColumnMatrix());
        Assert.Equal(XMath.Hypot(a, T.CreateChecked(4)).ToArray(), vector.Value.ToArray());
        tape.Backward(XMath.Sum(vector) + XMath.Sum(XMath.ColumnSums(matrix)));
        var grad = tape.GetGradient(x);
        T partial = T.CreateChecked(1.2);
        Assert.Equal(partial, grad[0]);
        Assert.Equal(-partial, grad[1]);
        Assert.Equal(double.CreateChecked(T.CreateChecked(3.2)), tape.GetGradient(scalar));
    }

    [Fact]
    public void ZeroAdjointsAndPartialsSuppressNonFiniteContributions()
    {
        var tape = new Tape();
        var x = tape.Leaf(new Vector<double>([double.NaN, 0, 3]));
        var y = tape.Leaf(new Vector<double>([1, 0, 4]));
        var weights = tape.Constant(new Vector<double>([0, double.PositiveInfinity, 2]));
        tape.Backward(XMath.Dot(XMath.Hypot(x, y), weights));
        Assert.Equal(new double[] { 0, 0, 1.2 }, tape.GetGradient(x).ToArray());
        Assert.Equal(new double[] { 0, 0, 1.6 }, tape.GetGradient(y).ToArray());
    }

    private static void AssertPartial(double expected, double actual, double tolerance)
    {
        if (expected == 0 || !double.IsFinite(expected))
        {
            DifferentiationAssert.SameNumber(expected, actual);
        }
        else
        {
            DifferentiationAssert.Close(expected, actual, tolerance, tolerance);
        }
    }
}
