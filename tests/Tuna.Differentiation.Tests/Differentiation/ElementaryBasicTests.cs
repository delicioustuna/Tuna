using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using static Tuna.Tests.Differentiation.DifferentiationAssert;

namespace Tuna.Tests.Differentiation;

/// <summary>初等関数の値、勾配と境界での規約</summary>
public class ElementaryBasicTests
{
    private static readonly string[] UnaryFunctions = ["Exp", "Log", "Sqrt", "Square", "Abs", "Expm1", "Log1p"];
    private static readonly string[] BinaryFunctions = ["Min", "Max"];

    [Fact]
    public void UnaryValuesPreserveTheScalarFunctionBits()
    {
        var tape = new Tape();
        foreach (double value in IeeeValues())
        {
            foreach (string function in UnaryFunctions)
            {
                tape.Reset();
                var input = tape.Leaf(value);
                double expected = Evaluate(function, value);
                EqualBits(expected, Evaluate(function, input).Value, context: $"{function}({value:R}), tracked");
                var constant = Evaluate(function, (ScalarNode)value);
                EqualBits(expected, constant.Value, context: $"{function}({value:R}), constant");
                Assert.False(constant.RequiresGradient, $"{function}({value:R}), constant");
            }
        }
    }

    [Fact]
    public void MinAndMaxValuesPreserveTheScalarFunctionBits()
    {
        var tape = new Tape();
        foreach (double a in IeeeValues())
        {
            foreach (double b in IeeeValues())
            {
                tape.Reset();
                var left = tape.Leaf(a);
                var right = tape.Leaf(b);
                foreach (string function in BinaryFunctions)
                {
                    double expected = EvaluateBinary(function, a, b);
                    EqualBits(expected, EvaluateBinary(function, left, right).Value, context: $"{function}({a:R}, {b:R}), tracked");
                    EqualBits(expected, EvaluateBinary(function, left, (ScalarNode)b).Value, context: $"{function}({a:R}, {b:R}), constant right");
                    EqualBits(expected, EvaluateBinary(function, (ScalarNode)a, right).Value, context: $"{function}({a:R}, {b:R}), constant left");
                    var constant = EvaluateBinary(function, (ScalarNode)a, (ScalarNode)b);
                    EqualBits(expected, constant.Value, context: $"{function}({a:R}, {b:R}), constants");
                    Assert.False(constant.RequiresGradient, $"{function}({a:R}, {b:R}), constants");
                }
            }
        }
    }

    [Theory]
    [InlineData("Exp", -2.0)]
    [InlineData("Exp", 0.0)]
    [InlineData("Exp", 2.0)]
    [InlineData("Log", 0.25)]
    [InlineData("Log", 2.0)]
    [InlineData("Sqrt", 0.25)]
    [InlineData("Sqrt", 2.0)]
    [InlineData("Square", -2.0)]
    [InlineData("Square", 0.0)]
    [InlineData("Square", 2.0)]
    [InlineData("Abs", -2.0)]
    [InlineData("Abs", 2.0)]
    [InlineData("Expm1", -2.0)]
    [InlineData("Expm1", 0.0)]
    [InlineData("Expm1", 2.0)]
    [InlineData("Log1p", -0.5)]
    [InlineData("Log1p", 0.0)]
    [InlineData("Log1p", 2.0)]
    public void UnaryGradientsAgreeWithRichardsonDifferences(string function, double value)
    {
        double step = Math.ScaleB(Math.Max(1, Math.Abs(value)), -10);
        double expected = Richardson(x => Reference(function, x), value, step);
        double actual = Gradient(x => Evaluate(function, x), value);
        Close(expected, actual, relativeTolerance: 2e-10, absoluteTolerance: 2e-10, context: $"{function}({value:R})");
    }

    [Theory]
    [InlineData("Min", -2.0, 3.0)]
    [InlineData("Min", 3.0, -2.0)]
    [InlineData("Max", -2.0, 3.0)]
    [InlineData("Max", 3.0, -2.0)]
    public void MinAndMaxGradientsAgreeWithRichardsonDifferences(string function, double a, double b)
    {
        var tape = new Tape();
        var left = tape.Leaf(a);
        var right = tape.Leaf(b);
        double[] gradient = new double[2];
        tape.Gradient(EvaluateBinary(function, left, right), new VectorSpan<double>(gradient));
        double leftStep = Math.ScaleB(Math.Max(1, Math.Abs(a)), -10);
        double rightStep = Math.ScaleB(Math.Max(1, Math.Abs(b)), -10);
        Close(Richardson(x => ReferenceBinary(function, x, b), a, leftStep), gradient[0],
            relativeTolerance: 2e-10, absoluteTolerance: 2e-10, context: $"{function}({a:R}, {b:R}), left");
        Close(Richardson(y => ReferenceBinary(function, a, y), b, rightStep), gradient[1],
            relativeTolerance: 2e-10, absoluteTolerance: 2e-10, context: $"{function}({a:R}, {b:R}), right");
    }

    [Fact]
    public void LogAndSqrtUseTheRightLimitAtBothSignedZeros()
    {
        foreach (double zero in new[] { 0.0, -0.0 })
        {
            AssertGradient("Log", zero, double.PositiveInfinity);
            AssertGradient("Sqrt", zero, double.PositiveInfinity);
        }

        AssertGradient("Log", double.PositiveInfinity, 0);
        AssertGradient("Sqrt", double.PositiveInfinity, 0);
        AssertGradient("Log", -1, double.NaN);
        AssertGradient("Sqrt", -1, double.NaN);
        AssertGradient("Log", double.NegativeInfinity, double.NaN);
        AssertGradient("Sqrt", double.NegativeInfinity, double.NaN);
    }

    [Fact]
    public void AbsUsesTheRightDerivativeAtBothSignedZeros()
    {
        AssertGradient("Abs", 0, 1);
        AssertGradient("Abs", -0.0, 1);
        AssertGradient("Abs", double.NegativeInfinity, -1);
        AssertGradient("Abs", double.PositiveInfinity, 1);
        AssertGradient("Abs", double.NaN, double.NaN);
    }

    [Fact]
    public void SquareKeepsItsFiniteDerivativeWhenTheValueOverflows()
    {
        var tape = new Tape();
        var input = tape.Leaf(1e200);
        var output = XMath.Square(input);
        Assert.Equal(double.PositiveInfinity, output.Value);
        tape.Backward(output);
        Assert.Equal(2e200, tape.GetGradient(input));
    }

    [Fact]
    public void Expm1RetainsItsDerivativeWhenTheValueRoundsToMinusOne()
    {
        var tape = new Tape();
        var input = tape.Leaf(-40);
        var output = XMath.Expm1(input);
        Assert.Equal(-1, output.Value);
        tape.Backward(output);
        Assert.Equal(Math.Exp(-40), tape.GetGradient(input));
        AssertGradient("Expm1", double.PositiveInfinity, double.PositiveInfinity);
        AssertGradient("Expm1", double.NegativeInfinity, 0);
    }

    [Fact]
    public void ExpUsesItsValueAsTheDerivativeAtInfinity()
    {
        AssertGradient("Exp", double.PositiveInfinity, double.PositiveInfinity);
        AssertGradient("Exp", double.NegativeInfinity, 0);
    }

    [Fact]
    public void Log1pRespectsItsDomainAndTheRightLimit()
    {
        AssertGradient("Log1p", -1, double.PositiveInfinity);
        AssertGradient("Log1p", double.BitDecrement(-1), double.NaN);
        AssertGradient("Log1p", double.NegativeInfinity, double.NaN);
        AssertGradient("Log1p", double.PositiveInfinity, 0);
        AssertGradient("Log1p", 0, 1);
        AssertGradient("Log1p", -0.0, 1);
    }

    [Fact]
    public void NaNUnaryInputsHaveNaNGradients()
    {
        foreach (string function in UnaryFunctions)
        {
            AssertGradient(function, double.NaN, double.NaN);
        }
    }

    [Fact]
    public void MinAndMaxTiesSendTheGradientToTheLeftInput()
    {
        double[] values = [-2, 0, -0.0, 2, double.NegativeInfinity, double.PositiveInfinity];
        foreach (double value in values)
        {
            foreach (string function in BinaryFunctions)
            {
                AssertBinaryGradient(function, value, value, 1, 0);
            }
        }

        foreach (string function in BinaryFunctions)
        {
            AssertBinaryGradient(function, 0.0, -0.0, 1, 0);
            AssertBinaryGradient(function, -0.0, 0.0, 1, 0);
        }
    }

    [Fact]
    public void MinAndMaxSkipConstantEdgesButKeepNaNDerivatives()
    {
        foreach (string function in BinaryFunctions)
        {
            var tape = new Tape();
            var input = tape.Leaf(2);
            tape.Backward(EvaluateBinary(function, input, (ScalarNode)2));
            SameNumber(1, tape.GetGradient(input), context: $"{function}(2, 2), tracked left");

            tape.Reset();
            input = tape.Leaf(2);
            tape.Backward(EvaluateBinary(function, (ScalarNode)2, input));
            SameNumber(0, tape.GetGradient(input), context: $"{function}(2, 2), tracked right");

            tape.Reset();
            input = tape.Leaf(2);
            tape.Backward(EvaluateBinary(function, input, (ScalarNode)double.NaN));
            Assert.True(double.IsNaN(tape.GetGradient(input)), $"{function}(2, NaN), tracked left");

            tape.Reset();
            input = tape.Leaf(2);
            tape.Backward(EvaluateBinary(function, (ScalarNode)double.NaN, input));
            Assert.True(double.IsNaN(tape.GetGradient(input)), $"{function}(NaN, 2), tracked right");

            AssertBinaryGradient(function, double.NaN, 2, double.NaN, double.NaN);
            AssertBinaryGradient(function, 2, double.NaN, double.NaN, double.NaN);
        }
    }

    [Fact]
    public void UnaryFunctionsRejectStaleNodesAndRecordingAfterBackward()
    {
        var tape = new Tape();
        var stale = tape.Leaf(2);
        tape.Reset();
        foreach (string function in UnaryFunctions)
        {
            Throws<InvalidOperationException>(() => Evaluate(function, stale), $"{function}(2), stale input");
        }

        var input = tape.Leaf(2);
        tape.Backward(input);
        foreach (string function in UnaryFunctions)
        {
            Throws<InvalidOperationException>(() => Evaluate(function, input), $"{function}(2), recording after backward");
            Assert.False(Evaluate(function, (ScalarNode)2).RequiresGradient, $"{function}(2), constant");
        }
    }

    [Fact]
    public void MinAndMaxRejectStaleForeignAndCompletedTapeInputs()
    {
        var tape = new Tape();
        var stale = tape.Leaf(2);
        tape.Reset();
        var input = tape.Leaf(3);
        var foreign = new Tape().Leaf(4);
        foreach (string function in BinaryFunctions)
        {
            Throws<InvalidOperationException>(() => EvaluateBinary(function, stale, input), $"{function}(2, 3), stale left input");
            Throws<InvalidOperationException>(() => EvaluateBinary(function, input, stale), $"{function}(3, 2), stale right input");
            Throws<InvalidOperationException>(() => EvaluateBinary(function, input, foreign), $"{function}(3, 4), foreign right input");
            Throws<InvalidOperationException>(() => EvaluateBinary(function, foreign, input), $"{function}(4, 3), foreign left input");
        }

        tape.Backward(input);
        foreach (string function in BinaryFunctions)
        {
            Throws<InvalidOperationException>(() => EvaluateBinary(function, input, (ScalarNode)4), $"{function}(3, 4), recording after backward");
            Throws<InvalidOperationException>(() => EvaluateBinary(function, (ScalarNode)4, input), $"{function}(4, 3), recording after backward");
            Assert.False(EvaluateBinary(function, (ScalarNode)2, (ScalarNode)4).RequiresGradient, $"{function}(2, 4), constants");
        }
    }

    private static double[] IeeeValues() =>
        [0.0, -0.0, double.Epsilon, -double.Epsilon, double.BitIncrement(1), double.BitDecrement(-1),
        1, -1, 2, -2, 1e-300, -1e300, double.MaxValue, double.MinValue,
        double.PositiveInfinity, double.NegativeInfinity, double.NaN,
        BitConverter.Int64BitsToDouble(0x7ff8000000000123),
        BitConverter.Int64BitsToDouble(unchecked((long)0xfff8000000000456))];

    private static ScalarNode Evaluate(string function, ScalarNode x) => function switch
    {
        "Exp" => XMath.Exp(x),
        "Log" => XMath.Log(x),
        "Sqrt" => XMath.Sqrt(x),
        "Square" => XMath.Square(x),
        "Abs" => XMath.Abs(x),
        "Expm1" => XMath.Expm1(x),
        "Log1p" => XMath.Log1p(x),
        _ => throw new ArgumentOutOfRangeException(nameof(function)),
    };

    private static double Evaluate(string function, double x) => function switch
    {
        "Exp" => Math.Exp(x),
        "Log" => XMath.Log<double>(x),
        "Sqrt" => XMath.Sqrt<double>(x),
        "Square" => XMath.Square<double>(x),
        "Abs" => XMath.Abs<double>(x),
        "Expm1" => XMath.Expm1<double>(x),
        "Log1p" => XMath.Log1p<double>(x),
        _ => throw new ArgumentOutOfRangeException(nameof(function)),
    };

    private static double Reference(string function, double x) => function switch
    {
        "Exp" => Math.Exp(x),
        "Log" => Math.Log(x),
        "Sqrt" => Math.Sqrt(x),
        "Square" => x * x,
        "Abs" => Math.Abs(x),
        "Expm1" => Math.Exp(x) - 1,
        "Log1p" => Math.Log(1 + x),
        _ => throw new ArgumentOutOfRangeException(nameof(function)),
    };

    private static ScalarNode EvaluateBinary(string function, ScalarNode left, ScalarNode right) => function switch
    {
        "Min" => XMath.Min(left, right),
        "Max" => XMath.Max(left, right),
        _ => throw new ArgumentOutOfRangeException(nameof(function)),
    };

    private static double EvaluateBinary(string function, double left, double right) => function switch
    {
        "Min" => XMath.Min<double>(left, right),
        "Max" => XMath.Max<double>(left, right),
        _ => throw new ArgumentOutOfRangeException(nameof(function)),
    };

    private static double ReferenceBinary(string function, double left, double right) => function switch
    {
        "Min" => Math.Min(left, right),
        "Max" => Math.Max(left, right),
        _ => throw new ArgumentOutOfRangeException(nameof(function)),
    };

    private static void AssertGradient(string function, double value, double expected) =>
        SameNumber(expected, Gradient(x => Evaluate(function, x), value), context: $"{function}({value:R})");

    private static void AssertBinaryGradient(string function, double a, double b, double dl, double dr)
    {
        var tape = new Tape();
        var left = tape.Leaf(a);
        var right = tape.Leaf(b);
        tape.Backward(EvaluateBinary(function, left, right));
        SameNumber(dl, tape.GetGradient(left), context: $"{function}({a:R}, {b:R}), left");
        SameNumber(dr, tape.GetGradient(right), context: $"{function}({a:R}, {b:R}), right");
    }
}
