using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using static Tuna.Tests.Differentiation.DifferentiationAssert;

namespace Tuna.Tests.Differentiation;

/// <summary>スカラーの値、四則演算と局所微分の規約</summary>
public class ScalarNodeTests
{
    private const double DifferenceRelativeTolerance = 2e-9;
    private const double DifferenceAbsoluteTolerance = 8 * double.Epsilon;

    [Fact]
    public void ConstantsAndDefaultHaveNoTape()
    {
        ScalarNode zero = default;
        ScalarNode value = 2.5;
        Assert.Equal(0, zero.Value);
        Assert.False(zero.RequiresGradient);
        Assert.Equal(2.5, value.Value);
        Assert.False(value.RequiresGradient);
        Assert.False((value * value + 1).RequiresGradient);
        Assert.False(ScalarNode.Record(7, [(value, double.NaN)]).RequiresGradient);
        Assert.False(ScalarNode.Record(7, []).RequiresGradient);
    }

    [Fact]
    public void ArithmeticPreservesTheUnrecordedValueBits()
    {
        double[] values = [0, -0.0, 1, -1, double.Epsilon, -double.Epsilon,
            double.BitIncrement(0), double.BitDecrement(0), 1e-300, -1e300,
            double.MaxValue, double.MinValue, double.PositiveInfinity, double.NegativeInfinity, double.NaN];
        var tape = new Tape();
        foreach (double a in values)
        {
            foreach (double b in values)
            {
                tape.Reset();
                var x = tape.Leaf(a);
                var y = tape.Leaf(b);
                for (int operation = 0; operation < 4; operation++)
                {
                    double expected = Evaluate(operation, a, b);
                    EqualBits(expected, Evaluate(operation, x, y).Value);
                    EqualBits(expected, Evaluate(operation, x, (ScalarNode)b).Value);
                    EqualBits(expected, Evaluate(operation, (ScalarNode)a, y).Value);
                    EqualBits(expected, Evaluate(operation, (ScalarNode)a, (ScalarNode)b).Value);
                }

                EqualBits(-a, (-x).Value);
            }
        }
    }

    [Theory]
    [InlineData(0.0, 1.25)]
    [InlineData(0.75, 1.25)]
    [InlineData(-2.5, 3.5)]
    [InlineData(1e-8, -2e-8)]
    [InlineData(1e100, 2e100)]
    public void ArithmeticGradientsAgreeWithRichardsonDifferences(double a, double b)
    {
        var tape = new Tape();
        double[] gradient = new double[2];
        for (int operation = 0; operation < 4; operation++)
        {
            tape.Reset();
            var x = tape.Leaf(a);
            var y = tape.Leaf(b);
            tape.Gradient(Evaluate(operation, x, y), new VectorSpan<double>(gradient));
            Close(Richardson(x0 => Evaluate(operation, x0, b), a), gradient[0], relativeTolerance: DifferenceRelativeTolerance, absoluteTolerance: DifferenceAbsoluteTolerance);
            Close(Richardson(y0 => Evaluate(operation, a, y0), b), gradient[1], relativeTolerance: DifferenceRelativeTolerance, absoluteTolerance: DifferenceAbsoluteTolerance);
        }
    }

    [Theory]
    [InlineData(0, 1.0, 1.0)]
    [InlineData(1, 1.0, -1.0)]
    [InlineData(2, 4.0, 2.0)]
    [InlineData(3, 0.25, -0.125)]
    public void MixedConstantsPreserveTheDerivativeBits(int operation, double leftDerivative, double rightDerivative)
    {
        var tape = new Tape();
        var left = tape.Leaf(2);
        tape.Backward(Evaluate(operation, left, (ScalarNode)4));
        EqualBits(leftDerivative, tape.GetGradient(left));
        tape.Reset();
        var right = tape.Leaf(4);
        tape.Backward(Evaluate(operation, (ScalarNode)2, right));
        EqualBits(rightDerivative, tape.GetGradient(right));
    }

    [Fact]
    public void RichardsonDifferencesHandleZeroAndRepresentableSmallSteps()
    {
        Close(1, Richardson(Math.Sin, 0), relativeTolerance: DifferenceRelativeTolerance, absoluteTolerance: DifferenceAbsoluteTolerance);
        Close(3, Richardson(x => 3 * x, double.Epsilon), relativeTolerance: DifferenceRelativeTolerance, absoluteTolerance: DifferenceAbsoluteTolerance);
        Close(1, Richardson(Math.Sin, 0, step: Math.ScaleB(1, -10)), relativeTolerance: DifferenceRelativeTolerance, absoluteTolerance: DifferenceAbsoluteTolerance);
    }

    [Fact]
    public void SharedNodesAndRepeatedEdgesAccumulateInReverseOrder()
    {
        var tape = new Tape();
        var x = tape.Leaf(3);
        var square = x * x;
        var output = square + square + x;
        double[] gradient = new double[1];
        Assert.Equal(21, tape.Gradient(output, new VectorSpan<double>(gradient)));
        Assert.Equal(13, gradient[0]);
        Assert.Equal(2, tape.GetGradient(square));
    }

    [Fact]
    public void MixedConstantsAndUnaryMinusHaveTheExpectedGradients()
    {
        var tape = new Tape();
        var x = tape.Leaf(2);
        var output = -x + 3 * x - 5 / x + x / 4 + 7;
        tape.Backward(output);
        Assert.Equal(-1 + 3 + 5.0 / 4 + 0.25, tape.GetGradient(x));
    }

    [Theory]
    [InlineData(1000, 600)]
    [InlineData(-1000, -600)]
    [InlineData(1000, 1000)]
    [InlineData(-1000, -1000)]
    public void DivisionDoesNotSquareAnExtremeDenominator(int numeratorExponent, int denominatorExponent)
    {
        double a = Math.ScaleB(1, numeratorExponent);
        double b = Math.ScaleB(1, denominatorExponent);
        var tape = new Tape();
        var x = tape.Leaf(a);
        var y = tape.Leaf(b);
        tape.Backward(x / y);
        EqualBits(Math.ScaleB(1, -denominatorExponent), tape.GetGradient(x));
        EqualBits(-Math.ScaleB(1, numeratorExponent - 2 * denominatorExponent), tape.GetGradient(y));
    }

    [Fact]
    public void SubnormalLocalDerivativesArePreserved()
    {
        var tape = new Tape();
        var x = tape.Leaf(1);
        tape.Backward(x * double.Epsilon);
        EqualBits(double.Epsilon, tape.GetGradient(x));
        tape.Reset();
        x = tape.Leaf(1);
        tape.Backward(x / double.MaxValue);
        EqualBits(1 / double.MaxValue, tape.GetGradient(x));
    }

    [Fact]
    public void ArbitraryArityCustomRuleSkipsConstantsAndSumsDuplicateInputs()
    {
        var tape = new Tape();
        var x = tape.Leaf(2);
        var y = tape.Leaf(3);
        var output = ScalarNode.Record(17, [(x, 2), (y, -3), ((ScalarNode)7, double.NaN), (x, 5)]);
        double[] gradient = new double[2];
        Assert.Equal(17, tape.Gradient(output, new VectorSpan<double>(gradient)));
        Assert.Equal(new double[] { 7, -3 }, gradient);
    }

    [Fact]
    public void CustomRulesCopyTheirInputPairsAndSupportLargeArity()
    {
        var tape = new Tape();
        var x = tape.Leaf(2);
        var pairs = new (ScalarNode Input, double Derivative)[100];
        for (int i = 0; i < pairs.Length; i++)
        {
            pairs[i] = (x, i + 1);
        }

        var output = ScalarNode.Record(17, pairs);
        Array.Clear(pairs);
        tape.Backward(output);
        Assert.Equal(5050, tape.GetGradient(x));
    }

    [Fact]
    public void UnusedRecordsDoNotPropagateNaNOrInfinity()
    {
        var tape = new Tape();
        var x = tape.Leaf(2);
        var output = 3 * x;
        _ = ScalarNode.Record(double.NaN, [(x, double.NaN)]);
        _ = ScalarNode.Record(0, [(x, double.PositiveInfinity)]);
        tape.Backward(output);
        Assert.Equal(3, tape.GetGradient(x));
    }

    [Fact]
    public void ZeroAdjointsAndZeroLocalDerivativesDoNotMakeNaN()
    {
        var tape = new Tape();
        var x = tape.Leaf(0);
        var singular = ScalarNode.Record(0, [(x, double.PositiveInfinity)]);
        tape.Backward(0 * singular);
        Assert.Equal(0, tape.GetGradient(x));
        tape.Reset();
        x = tape.Leaf(0);
        var flat = ScalarNode.Record(1, [(x, 0)]);
        var output = ScalarNode.Record(1, [(flat, double.PositiveInfinity)]);
        tape.Backward(output);
        Assert.Equal(0, tape.GetGradient(x));
        tape.Reset();
        x = tape.Leaf(0);
        flat = ScalarNode.Record(1, [(x, 0)]);
        output = ScalarNode.Record(double.NaN, [(flat, double.NaN)]);
        tape.Backward(output);
        Assert.True(double.IsNaN(tape.GetGradient(flat)));
        Assert.Equal(0, tape.GetGradient(x));
    }

    [Fact]
    public void ComparisonsUseValuesAndFollowDoubleNaNRules()
    {
        var tape = new Tape();
        var x = tape.Leaf(2);
        Assert.True(x < 3);
        Assert.True(x <= 2);
        Assert.True(3 > x);
        Assert.True(2 >= x);
        Assert.False(x < double.NaN);
        Assert.False(x > double.NaN);
        Assert.False(x <= double.NaN);
        Assert.False(x >= double.NaN);
        Assert.DoesNotContain(typeof(ScalarNode).GetMethods(), m => m.Name is "op_Equality" or "op_Inequality");
    }

    private static ScalarNode Evaluate(int operation, ScalarNode a, ScalarNode b) => operation switch
    {
        0 => a + b,
        1 => a - b,
        2 => a * b,
        3 => a / b,
        _ => throw new ArgumentOutOfRangeException(nameof(operation)),
    };

    private static double Evaluate(int operation, double a, double b) => operation switch
    {
        0 => a + b,
        1 => a - b,
        2 => a * b,
        3 => a / b,
        _ => throw new ArgumentOutOfRangeException(nameof(operation)),
    };
}
