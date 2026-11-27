using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using static Tuna.Tests.Differentiation.DifferentiationAssert;

namespace Tuna.Tests.Differentiation;

/// <summary>三角関数・双曲線正接・逆正接の値と局所微分</summary>
public class ElementaryTrigonometricTests
{
    private static readonly string[] Functions = ["Sin", "Cos", "Tan", "Tanh", "Atan"];
    private static readonly string[] TrigonometricFunctions = ["Sin", "Cos", "Tan"];

    [Fact]
    public void ForwardValuesMatchTheScalarFunctionsForTrackedAndConstantInputs()
    {
        double[] values = [0, -0.0, 1, -1, double.Epsilon, -double.Epsilon,
            1e-300, -1e300, double.MaxValue, double.MinValue,
            double.PositiveInfinity, double.NegativeInfinity, double.NaN,
            Math.PI / 2, -Math.PI / 2];
        var tape = new Tape();
        foreach (string function in Functions)
        {
            foreach (double value in values)
            {
                tape.Reset();
                var input = tape.Leaf(value);
                double expected = Evaluate(function, value);
                EqualBits(expected, Evaluate(function, input).Value, context: $"{function}({value:R}), tracked input");
                var constant = Evaluate(function, (ScalarNode)value);
                EqualBits(expected, constant.Value, context: $"{function}({value:R}), constant input");
                Assert.False(constant.RequiresGradient, $"{function}({value:R}), constant input");
            }
        }
    }

    [Fact]
    public void DoubleTanhForwardValuesAgreeAcrossScalarsAndVectors()
    {
        double[] values = [0, -0.0, double.Epsilon, -double.Epsilon, 1e-6, -0.5, 0.5, 0.5048239254415147,
            double.BitDecrement(0.75), 0.75, double.BitIncrement(0.75), 1.25, -3, 19.061547465398498,
            double.PositiveInfinity, double.NegativeInfinity, double.NaN, 0.873041391];
        var x = new Vector<double>(values);
        var tape = new Tape();
        var node = tape.Leaf(x);
        var vector = XMath.Tanh(x);
        var tracked = XMath.Tanh(node).Value;
        var approximate = XMath.Tanh(node, approximate: true).Value;
        for (int i = 0; i < values.Length; i++)
        {
            double expected = XMath.Tanh(values[i]);
            EqualBits(expected, vector[i]);
            EqualBits(expected, XMath.Tanh(tape.Leaf(values[i])).Value);
            EqualBits(expected, tracked[i]);
            EqualBits(expected, approximate[i]);
        }
    }

    public static IEnumerable<object[]> DifferenceCases()
    {
        foreach (string function in Functions)
        {
            foreach (double value in new double[] { 0, -1e-6, 1e-6, -0.5, 0.5, -1.25, 1.25 })
            {
                yield return [function, value, Math.ScaleB(1, -10)];
            }
        }

        foreach (string function in TrigonometricFunctions)
        {
            yield return [function, 1e6, Math.ScaleB(1, -10)];
            yield return [function, -1e6, Math.ScaleB(1, -10)];
        }

        yield return ["Tan", 1.4, Math.ScaleB(1, -12)];
        yield return ["Tanh", 6.0, Math.ScaleB(1, -8)];
        yield return ["Tanh", -6.0, Math.ScaleB(1, -8)];
        yield return ["Atan", 1000.0, 0.25];
        yield return ["Atan", -1000.0, 0.25];
    }

    [Theory]
    [MemberData(nameof(DifferenceCases))]
    public void GradientsAgreeWithRichardsonDifferences(string function, double value, double step)
    {
        double expected = function == "Tanh" ? Richardson(Math.Tanh, value, step) : Richardson(x => Evaluate(function, x), value, step);
        double actual = Gradient(x => Evaluate(function, x), value);
        Close(expected, actual, relativeTolerance: 2e-8, absoluteTolerance: 2e-10, context: $"{function}({value:R})");
    }

    [Theory]
    [InlineData("Sin", 1.0)]
    [InlineData("Cos", 0.0)]
    [InlineData("Tan", 1.0)]
    [InlineData("Tanh", 1.0)]
    [InlineData("Atan", 1.0)]
    public void DerivativesAtBothSignedZerosHaveTheirExactValues(string function, double expected)
    {
        EqualBits(expected, Gradient(x => Evaluate(function, x), 0), context: $"{function}(+0)");
        EqualBits(expected, Gradient(x => Evaluate(function, x), -0.0), context: $"{function}(-0)");
    }

    [Fact]
    public void NonFiniteInputsFollowTheDocumentedDerivativeRules()
    {
        foreach (string function in Functions)
        {
            Assert.True(double.IsNaN(Gradient(x => Evaluate(function, x), double.NaN)), $"{function}(NaN)");
            foreach (double value in new double[] { double.NegativeInfinity, double.PositiveInfinity })
            {
                double actual = Gradient(x => Evaluate(function, x), value);
                if (function is "Sin" or "Cos" or "Tan")
                {
                    Assert.True(double.IsNaN(actual), $"{function}({value:R})");
                }
                else
                {
                    EqualBits(0, actual, context: $"{function}({value:R})");
                }
            }
        }
    }

    [Theory]
    [InlineData(20.0, 1.6993417021166355e-17)]
    [InlineData(350.0, 3.943870617503908e-304)]
    [InlineData(355.0, 1.790514490270052e-308)]
    [InlineData(360.0, 8.12892320967e-313)]
    [InlineData(370.0, 1.675e-321)]
    [InlineData(371.0, 2.27e-322)]
    [InlineData(372.0, 3e-323)]
    [InlineData(373.0, double.Epsilon)]
    public void TanhPreservesTailDerivativesAfterTheValueRoundsToOne(double value, double expected)
    {
        // mpmath 110 桁で sech(x)² を求め、binary64 に丸めた固定値。
        EqualBits(1, Math.Tanh(value), context: $"Tanh({value:R})");
        double positive = Gradient(x => Evaluate("Tanh", x), value);
        double negative = Gradient(x => Evaluate("Tanh", x), -value);
        Assert.True(positive > 0, $"Tanh({value:R})");
        WithinUlps(expected, positive, tolerance: 3, context: $"Tanh({value:R})");
        EqualBits(positive, negative, context: $"Tanh(±{value:R})");
    }

    [Theory]
    [InlineData(1e154, 1e-308)]
    [InlineData(1e155, 1e-310)]
    [InlineData(1e160, 1e-320)]
    [InlineData(1e161, 1e-322)]
    public void AtanPreservesSubnormalDerivativesWithoutSquaringTheInput(double value, double expected)
    {
        // mpmath 110 桁で、binary64 の入力に対する 1/(1 + x²) を求めた固定値。
        double positive = Gradient(x => Evaluate("Atan", x), value);
        double negative = Gradient(x => Evaluate("Atan", x), -value);
        Assert.True(positive > 0, $"Atan({value:R})");
        WithinUlps(expected, positive, tolerance: 2, context: $"Atan({value:R})");
        EqualBits(positive, negative, context: $"Atan(±{value:R})");
    }

    [Fact]
    public void TailsUnderflowOnlyWhenTheDerivativeIsBelowTheRepresentableRange()
    {
        EqualBits(0, Gradient(x => Evaluate("Tanh", x), 374), "Tanh(374)");
        EqualBits(0, Gradient(x => Evaluate("Tanh", x), -374), "Tanh(-374)");
        EqualBits(0, Gradient(x => Evaluate("Atan", x), 1e162), context: $"Atan({1e162:R})");
        EqualBits(0, Gradient(x => Evaluate("Atan", x), -1e162), context: $"Atan({-1e162:R})");
        EqualBits(0, Gradient(x => Evaluate("Atan", x), double.MaxValue), context: $"Atan({double.MaxValue:R})");
        EqualBits(0, Gradient(x => Evaluate("Atan", x), double.MinValue), context: $"Atan({double.MinValue:R})");
    }

    [Fact]
    public void StaleInputsAndRecordingAfterBackwardThrowForEveryFunction()
    {
        var tape = new Tape();
        var stale = tape.Leaf(0.25);
        tape.Reset();
        var current = tape.Leaf(0.5);
        tape.Backward(current);
        foreach (string function in Functions)
        {
            Throws<InvalidOperationException>(() => Evaluate(function, stale), $"{function}(0.25), stale input");
            Throws<InvalidOperationException>(() => Evaluate(function, current), $"{function}(0.5), recording after backward");
            var constant = Evaluate(function, (ScalarNode)0.5);
            Assert.False(constant.RequiresGradient, $"{function}(0.5), constant input");
            EqualBits(Evaluate(function, 0.5), constant.Value, context: $"{function}(0.5), constant input");
        }
    }

    private static ScalarNode Evaluate(string function, ScalarNode value) => function switch
    {
        "Sin" => XMath.Sin(value),
        "Cos" => XMath.Cos(value),
        "Tan" => XMath.Tan(value),
        "Tanh" => XMath.Tanh(value),
        "Atan" => XMath.Atan(value),
        _ => throw new ArgumentOutOfRangeException(nameof(function)),
    };

    private static double Evaluate(string function, double value) => function switch
    {
        "Sin" => Math.Sin(value),
        "Cos" => Math.Cos(value),
        "Tan" => Math.Tan(value),
        "Tanh" => XMath.Tanh(value),
        "Atan" => Math.Atan(value),
        _ => throw new ArgumentOutOfRangeException(nameof(function)),
    };
}
