using Tuna.Differentiation;
using static Tuna.Tests.Differentiation.DifferentiationAssert;

namespace Tuna.Tests.Differentiation;

/// <summary>冪の値、両入力の局所微分と指数の境界</summary>
public class PowerTests
{
    private const double RelativeTolerance = SixteenEpsilon;

    [Fact]
    public void ForwardValuesPreserveCoreBitsForEveryTrackingCombination()
    {
        double[] bases = [double.NegativeInfinity, -double.MaxValue, -2, -1, -double.Epsilon,
            -0.0, 0, double.Epsilon, 0.5, 1, 2, double.MaxValue, double.PositiveInfinity, double.NaN];
        double[] exponents = [double.NegativeInfinity, -3, -2, -0.5, -double.Epsilon, -0.0,
            0, double.Epsilon, 0.5, 1, 2, 3, double.MaxValue, double.PositiveInfinity, double.NaN];
        var tape = new Tape();
        foreach (double x in bases)
        {
            foreach (double y in exponents)
            {
                double expected = XMath.Pow<double>(x, y);
                for (int tracking = 0; tracking < 4; tracking++)
                {
                    tape.Reset();
                    ScalarNode left = (tracking & 1) != 0 ? tape.Leaf(x) : x;
                    ScalarNode right = (tracking & 2) != 0 ? tape.Leaf(y) : y;
                    var result = XMath.Pow(left, right);
                    string context = $"Pow({x:R}, {y:R}), tracking={tracking}";
                    EqualBits(expected, result.Value, context: context);
                    Assert.Equal(tracking != 0, result.RequiresGradient);
                }
            }
        }
    }

    [Theory]
    [InlineData(0.125, -1.25)]
    [InlineData(0.5, 2.5)]
    [InlineData(1.0, 3.0)]
    [InlineData(2.0, 0.0)]
    [InlineData(2.0, -0.5)]
    [InlineData(4.0, 1.5)]
    public void BothPartialDerivativesAgreeWithRichardsonDifferences(double x, double y)
    {
        var tape = new Tape();
        var left = tape.Leaf(x);
        var right = tape.Leaf(y);
        tape.Backward(XMath.Pow(left, right));
        double expectedX = Richardson(v => XMath.Pow<double>(v, y), x);
        double expectedY = Richardson(v => XMath.Pow<double>(x, v), y);
        Close(expectedX, tape.GetGradient(left), relativeTolerance: 2e-8, absoluteTolerance: 2e-10, context: $"Pow({x:R}, {y:R}), dx");
        Close(expectedY, tape.GetGradient(right), relativeTolerance: 2e-8, absoluteTolerance: 2e-10, context: $"Pow({x:R}, {y:R}), dy");
    }

    [Theory]
    [InlineData(-2.0, -3.0, -0.1875)]
    [InlineData(-2.0, -2.0, 0.25)]
    [InlineData(-2.0, 0.0, 0.0)]
    [InlineData(-2.0, 1.0, 1.0)]
    [InlineData(-2.0, 2.0, -4.0)]
    [InlineData(-2.0, 3.0, 12.0)]
    public void NegativeBasesHaveIntegerPowerBaseDerivativesAndUndefinedExponentDerivatives(double x, double y, double expected)
    {
        EqualBits(expected, Gradient(v => XMath.Pow(v, y), x), context: $"Pow({x:R}, {y:R}), constant exponent");
        var tape = new Tape();
        var left = tape.Leaf(x);
        var right = tape.Leaf(y);
        tape.Backward(XMath.Pow(left, right));
        EqualBits(expected, tape.GetGradient(left), context: $"Pow({x:R}, {y:R}), dx");
        Assert.True(double.IsNaN(tape.GetGradient(right)), $"Pow({x:R}, {y:R}), dy");
    }

    [Theory]
    [InlineData(-2.0, 0.5)]
    [InlineData(-2.0, double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity, 0.5)]
    [InlineData(double.NaN, 0.0)]
    [InlineData(1.0, double.NaN)]
    public void InputsOutsideTheDerivativeDomainProduceNaNForBothPartials(double x, double y)
    {
        var tape = new Tape();
        var left = tape.Leaf(x);
        var right = tape.Leaf(y);
        tape.Backward(XMath.Pow(left, right));
        Assert.True(double.IsNaN(tape.GetGradient(left)), $"Pow({x:R}, {y:R}), dx");
        Assert.True(double.IsNaN(tape.GetGradient(right)), $"Pow({x:R}, {y:R}), dy");
    }

    [Theory]
    [InlineData(-2.0, double.NegativeInfinity, double.NaN)]
    [InlineData(0.0, 0.0, double.NaN)]
    [InlineData(0.5, double.PositiveInfinity, 0.0)]
    [InlineData(1.0, 1.0, 0.0)]
    [InlineData(2.0, 0.0, 0.0)]
    [InlineData(double.PositiveInfinity, 0.0, 0.0)]
    [InlineData(double.NegativeInfinity, double.NegativeInfinity, double.NaN)]
    public void BothSignedZerosUseTheDocumentedOneSidedBaseDerivative(double y, double expectedX, double expectedY)
    {
        foreach (double x in new double[] { 0, -0.0 })
        {
            var tape = new Tape();
            var left = tape.Leaf(x);
            var right = tape.Leaf(y);
            tape.Backward(XMath.Pow(left, right));
            SameNumber(expectedX, tape.GetGradient(left), context: $"Pow({x:R}, {y:R}), dx");
            SameNumber(expectedY, tape.GetGradient(right), context: $"Pow({x:R}, {y:R}), dy");
        }
    }

    [Theory]
    [InlineData(double.PositiveInfinity, -2.0, 0.0, 0.0)]
    [InlineData(double.PositiveInfinity, 0.0, 0.0, double.PositiveInfinity)]
    [InlineData(double.PositiveInfinity, 0.5, 0.0, double.PositiveInfinity)]
    [InlineData(double.PositiveInfinity, 1.0, 1.0, double.PositiveInfinity)]
    [InlineData(double.PositiveInfinity, 2.0, double.PositiveInfinity, double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity, 2.0, double.NegativeInfinity, double.NaN)]
    [InlineData(double.NegativeInfinity, 3.0, double.PositiveInfinity, double.NaN)]
    [InlineData(double.NegativeInfinity, -2.0, 0.0, double.NaN)]
    [InlineData(0.5, double.PositiveInfinity, 0.0, 0.0)]
    [InlineData(0.5, double.NegativeInfinity, double.NegativeInfinity, double.NegativeInfinity)]
    [InlineData(1.0, double.PositiveInfinity, double.PositiveInfinity, 0.0)]
    [InlineData(1.0, double.NegativeInfinity, double.NegativeInfinity, 0.0)]
    [InlineData(2.0, double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity)]
    [InlineData(2.0, double.NegativeInfinity, 0.0, 0.0)]
    public void InfiniteInputsUseFiniteSideLimits(double x, double y, double expectedX, double expectedY)
    {
        var tape = new Tape();
        var left = tape.Leaf(x);
        var right = tape.Leaf(y);
        tape.Backward(XMath.Pow(left, right));
        SameNumber(expectedX, tape.GetGradient(left), context: $"Pow({x:R}, {y:R}), dx");
        SameNumber(expectedY, tape.GetGradient(right), context: $"Pow({x:R}, {y:R}), dy");
    }

    [Theory]
    [InlineData(1e-300, 1.5)]
    [InlineData(1e-200, 2.0)]
    [InlineData(1e200, 2.0)]
    [InlineData(1e308, 1.5)]
    [InlineData(-1e200, 2.0)]
    public void FiniteBaseDerivativesSurviveUnderflowAndOverflowOfTheValue(double x, double y)
    {
        double expected = y * XMath.Pow<double>(x, y - 1);
        double value = XMath.Pow<double>(x, y);
        Assert.True(value == 0 || double.IsInfinity(value));
        double actual = Gradient(v => XMath.Pow(v, y), x);
        Close(expected, actual, relativeTolerance: RelativeTolerance, absoluteTolerance: 4 * double.Epsilon, context: $"Pow({x:R}, {y:R}), dx");
    }

    [Fact]
    public void AZeroRoundedValueStillHasANonzeroExponentDerivative()
    {
        double x = Math.ScaleB(1, -540);
        EqualBits(0, XMath.Pow<double>(x, 2), context: "Pow(2^-540, 2), value");
        double actual = Gradient(y => XMath.Pow(x, y), 2);
        // x²·ln x = −540·ln(2)·2^-1080。係数を先に入れて非正規化数まで残す。
        double expected = Math.ScaleB(-540 * Math.Log(2), -1080);
        EqualBits(expected, actual, context: "Pow(2^-540, 2), dy");
        Assert.True(actual < 0);
    }

    [Fact]
    public void ASharedInputAccumulatesBothPartialDerivatives()
    {
        double actual = Gradient(x => XMath.Pow(x, x), 2);
        double expected = 4 * (1 + Math.Log(2));
        Close(expected, actual, relativeTolerance: RelativeTolerance, absoluteTolerance: 0, context: "Pow(x, x) at x=2");
    }

    [Fact]
    public void OwnershipGenerationAndRecordingStateAreChecked()
    {
        var tape = new Tape();
        var other = new Tape();
        var left = tape.Leaf(2);
        var right = other.Leaf(3);
        Assert.Throws<InvalidOperationException>(() => XMath.Pow(left, right));
        tape.Reset();
        var current = tape.Leaf(3);
        Assert.Throws<InvalidOperationException>(() => XMath.Pow(left, current));
        Assert.Throws<InvalidOperationException>(() => XMath.Pow(current, left));
        tape.Backward(current);
        Assert.Throws<InvalidOperationException>(() => XMath.Pow(current, 2));
        Assert.Throws<InvalidOperationException>(() => XMath.Pow(2, current));
        Assert.False(XMath.Pow((ScalarNode)2, 3).RequiresGradient);
    }

    [Fact]
    public void RepeatedRecordingAndConstantEvaluationAllocateNothingAfterWarmup()
    {
        var tape = new Tape();
        for (int i = 0; i < 200; i++)
        {
            EvaluateRepeatedly(tape);
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1000; i++)
        {
            EvaluateRepeatedly(tape);
        }

        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }

    private static void EvaluateRepeatedly(Tape tape)
    {
        tape.Reset();
        var x = tape.Leaf(2);
        var y = tape.Leaf(3);
        tape.Backward(XMath.Pow(x, y));
        _ = XMath.Pow((ScalarNode)2, 3).Value;
    }
}
