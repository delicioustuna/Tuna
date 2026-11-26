using Tuna.Differentiation;
using static Tuna.Tests.Differentiation.DifferentiationAssert;

namespace Tuna.Tests.Differentiation;

/// <summary>指数・対数を組み合わせた関数の値、安定な局所微分と定義域</summary>
public class ElementaryStableTests
{
    private static readonly string[] Functions = ["Logistic", "LogLogistic", "Logit", "Softplus", "Log1mExp"];

    [Fact]
    public void ForwardValuesHaveTheSameBitsAsScalarFunctions()
    {
        double[] values = [0, -0.0, double.Epsilon, -double.Epsilon,
            0.25, 0.5, 0.75, 1, -1, 37, -37, 700, -700, 745, -745,
            double.MaxValue, double.MinValue, double.PositiveInfinity, double.NegativeInfinity, double.NaN];
        var tape = new Tape();
        foreach (double value in values)
        {
            tape.Reset();
            var x = tape.Leaf(value);
            foreach (string function in Functions)
            {
                double expected = Evaluate(function, value);
                EqualBits(expected, Evaluate(function, x).Value, context: $"{function}({value:R}), tracked");
                var constant = Evaluate(function, (ScalarNode)value);
                EqualBits(expected, constant.Value, context: $"{function}({value:R}), constant");
                Assert.False(constant.RequiresGradient, $"{function}({value:R}), constant");
            }
        }
    }

    [Theory]
    [InlineData("Logistic", -10.0)]
    [InlineData("Logistic", -1.0)]
    [InlineData("Logistic", 0.0)]
    [InlineData("Logistic", 1.0)]
    [InlineData("Logistic", 10.0)]
    [InlineData("LogLogistic", -10.0)]
    [InlineData("LogLogistic", -1.0)]
    [InlineData("LogLogistic", 0.0)]
    [InlineData("LogLogistic", 1.0)]
    [InlineData("LogLogistic", 10.0)]
    [InlineData("Logit", 0.01)]
    [InlineData("Logit", 0.25)]
    [InlineData("Logit", 0.5)]
    [InlineData("Logit", 0.75)]
    [InlineData("Logit", 0.99)]
    [InlineData("Softplus", -10.0)]
    [InlineData("Softplus", -1.0)]
    [InlineData("Softplus", 0.0)]
    [InlineData("Softplus", 1.0)]
    [InlineData("Softplus", 10.0)]
    [InlineData("Log1mExp", -10.0)]
    [InlineData("Log1mExp", -1.0)]
    [InlineData("Log1mExp", -0.1)]
    public void GradientsAgreeWithRichardsonDifferences(string function, double value)
    {
        double step = Math.ScaleB(Math.Max(1, Math.Abs(value)), -13);
        double expected = Richardson(x => Evaluate(function, x), value, step);
        double actual = Gradient(x => Evaluate(function, x), value);
        Close(expected, actual, relativeTolerance: 2e-8, absoluteTolerance: double.Epsilon, context: $"{function}({value:R})");
    }

    [Theory]
    [InlineData("Logistic", 0.0, 0.25)]
    [InlineData("LogLogistic", 0.0, 0.5)]
    [InlineData("Logit", 0.5, 4.0)]
    [InlineData("Logit", 0.25, 5.333333333333333)]
    [InlineData("Softplus", 0.0, 0.5)]
    public void GradientsHaveTheExpectedAnalyticValues(string function, double value, double expected)
    {
        EqualBits(expected, Gradient(x => Evaluate(function, x), value), context: $"{function}({value:R})");
    }

    [Fact]
    public void Log1mExpAtNegativeLogTwoHasDerivativeNegativeOne()
    {
        double value = -Math.Log(2);
        Close(-1, Gradient(x => Evaluate("Log1mExp", x), value),
            relativeTolerance: 2e-15, absoluteTolerance: double.Epsilon, context: $"Log1mExp({value:R})");
    }

    [Theory]
    [InlineData(37.0, 8.533047625744064e-17)]
    [InlineData(700.0, 9.85967654375977e-305)]
    [InlineData(745.0, double.Epsilon)]
    public void LogisticDerivativeRemainsNonzeroInBothTails(double magnitude, double expected)
    {
        double positive = Gradient(x => Evaluate("Logistic", x), magnitude);
        Assert.True(positive > 0, $"Logistic({magnitude:R})");
        Close(expected, positive, relativeTolerance: 2e-15, absoluteTolerance: double.Epsilon, context: $"Logistic({magnitude:R})");
        EqualBits(positive, Gradient(x => Evaluate("Logistic", x), -magnitude), context: $"Logistic(±{magnitude:R})");
        if (magnitude >= 700)
        {
            EqualBits(expected, positive, context: $"Logistic({magnitude:R})");
        }
    }

    [Theory]
    [InlineData(-1e-16, -1e16)]
    [InlineData(-1e-300, -1e300)]
    [InlineData(-double.Epsilon, double.NegativeInfinity)]
    public void Log1mExpDerivativeAvoidsCancellationNearZero(double value, double expected)
    {
        double actual = Gradient(x => Evaluate("Log1mExp", x), value);
        if (double.IsInfinity(expected))
        {
            EqualBits(expected, actual, context: $"Log1mExp({value:R})");
        }
        else
        {
            Close(expected, actual, relativeTolerance: 2e-15, absoluteTolerance: double.Epsilon, context: $"Log1mExp({value:R})");
        }
    }

    [Theory]
    [InlineData(-700.0, -9.85967654375977e-305)]
    [InlineData(-745.0, -double.Epsilon)]
    public void Log1mExpDerivativePreservesTheNegativeTail(double value, double expected)
    {
        EqualBits(expected, Gradient(x => Evaluate("Log1mExp", x), value), context: $"Log1mExp({value:R})");
    }

    [Theory]
    [InlineData("Logistic", double.NegativeInfinity, 0.0)]
    [InlineData("Logistic", double.PositiveInfinity, 0.0)]
    [InlineData("LogLogistic", double.NegativeInfinity, 1.0)]
    [InlineData("LogLogistic", double.PositiveInfinity, 0.0)]
    [InlineData("Logit", 0.0, double.PositiveInfinity)]
    [InlineData("Logit", 1.0, double.PositiveInfinity)]
    [InlineData("Softplus", double.NegativeInfinity, 0.0)]
    [InlineData("Softplus", double.PositiveInfinity, 1.0)]
    [InlineData("Log1mExp", double.NegativeInfinity, 0.0)]
    [InlineData("Log1mExp", 0.0, double.NegativeInfinity)]
    public void EndpointsHaveTheDeclaredGradients(string function, double value, double expected)
    {
        // ゼロの局所微分はテープが枝を飛ばすため、葉の勾配は +0 のまま。
        EqualBits(expected, Gradient(x => Evaluate(function, x), value), context: $"{function}({value:R})");
    }

    [Fact]
    public void NegativeZeroHasTheSameEndpointGradients()
    {
        EqualBits(double.PositiveInfinity, Gradient(x => Evaluate("Logit", x), -0.0), context: "Logit(-0)");
        EqualBits(double.NegativeInfinity, Gradient(x => Evaluate("Log1mExp", x), -0.0), context: "Log1mExp(-0)");
    }

    [Theory]
    [InlineData("Logistic", double.NaN)]
    [InlineData("LogLogistic", double.NaN)]
    [InlineData("Logit", double.NaN)]
    [InlineData("Logit", -double.Epsilon)]
    [InlineData("Logit", -1.0)]
    [InlineData("Logit", 1.0000000000000002)]
    [InlineData("Logit", double.NegativeInfinity)]
    [InlineData("Logit", double.PositiveInfinity)]
    [InlineData("Softplus", double.NaN)]
    [InlineData("Log1mExp", double.NaN)]
    [InlineData("Log1mExp", double.Epsilon)]
    [InlineData("Log1mExp", 1.0)]
    [InlineData("Log1mExp", double.PositiveInfinity)]
    public void InvalidInputsHaveNaNGradients(string function, double value)
    {
        Assert.True(double.IsNaN(Gradient(x => Evaluate(function, x), value)), $"{function}({value:R})");
    }

    [Fact]
    public void ConstantsDoNotRecordAfterBackward()
    {
        var tape = new Tape();
        var x = tape.Leaf(1);
        tape.Backward(x);
        foreach (string function in Functions)
        {
            var output = Evaluate(function, (ScalarNode)0.25);
            Assert.False(output.RequiresGradient, $"{function}(0.25), constant");
            EqualBits(Evaluate(function, 0.25), output.Value, context: $"{function}(0.25), constant");
        }
    }

    [Fact]
    public void OperationsRejectStaleInputs()
    {
        var tape = new Tape();
        var stale = tape.Leaf(double.PositiveInfinity);
        tape.Reset();
        foreach (string function in Functions)
        {
            Throws<InvalidOperationException>(() => Evaluate(function, stale), $"{function}(+∞), stale input");
        }
    }

    [Fact]
    public void OperationsRejectRecordingAfterBackwardEvenForZeroDerivatives()
    {
        var tape = new Tape();
        var x = tape.Leaf(double.NegativeInfinity);
        tape.Backward(x);
        foreach (string function in Functions)
        {
            Throws<InvalidOperationException>(() => Evaluate(function, x), $"{function}(−∞), recording after backward");
        }
    }

    private static ScalarNode Evaluate(string function, ScalarNode x) => function switch
    {
        "Logistic" => XMath.Logistic(x),
        "LogLogistic" => XMath.LogLogistic(x),
        "Logit" => XMath.Logit(x),
        "Softplus" => XMath.Softplus(x),
        "Log1mExp" => XMath.Log1mExp(x),
        _ => throw new ArgumentOutOfRangeException(nameof(function)),
    };

    private static double Evaluate(string function, double x) => function switch
    {
        "Logistic" => XMath.Logistic<double>(x),
        "LogLogistic" => XMath.LogLogistic<double>(x),
        "Logit" => XMath.Logit<double>(x),
        "Softplus" => XMath.Softplus<double>(x),
        "Log1mExp" => XMath.Log1mExp<double>(x),
        _ => throw new ArgumentOutOfRangeException(nameof(function)),
    };
}
