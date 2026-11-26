using Tuna;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using static Tuna.Tests.Differentiation.DifferentiationAssert;

namespace Tuna.Tests.Differentiation;

/// <summary>指数の和・差と縮約の値、局所微分、無限大の規約</summary>
public class LogReductionTests
{
    private static readonly string[] BinaryFunctions = ["LogAddExp", "LogSubExp"];

    [Fact]
    public void BinaryValuesPreserveCoreBits()
    {
        double[] arguments = [double.NegativeInfinity, -double.MaxValue, -745, -40, -1, -double.Epsilon,
            -0.0, 0.0, double.Epsilon, 1, 40, 745, double.MaxValue, double.PositiveInfinity, double.NaN];
        var tape = new Tape();
        foreach (string function in BinaryFunctions)
        {
            foreach (double a in arguments)
            {
                foreach (double b in arguments)
                {
                    tape.Reset();
                    var x = tape.Leaf(a);
                    var y = tape.Leaf(b);
                    double expected = Evaluate(function, a, b);
                    EqualBits(expected, Evaluate(function, x, y).Value, context: $"{function}({a:R}, {b:R}), tracked");
                    var constant = Evaluate(function, (ScalarNode)a, b);
                    EqualBits(expected, constant.Value, context: $"{function}({a:R}, {b:R}), constant");
                    Assert.False(constant.RequiresGradient);
                }
            }
        }
    }

    [Theory]
    [InlineData("LogAddExp", -10.0, -9.0)]
    [InlineData("LogAddExp", 0.0, 0.0)]
    [InlineData("LogAddExp", 1.0, -3.0)]
    [InlineData("LogAddExp", 20.0, 19.0)]
    [InlineData("LogSubExp", -9.0, -10.0)]
    [InlineData("LogSubExp", 0.0, -0.1)]
    [InlineData("LogSubExp", 3.0, -1.0)]
    [InlineData("LogSubExp", 20.0, 19.0)]
    public void BinaryGradientsAgreeWithRichardsonDifferences(string function, double a, double b)
    {
        var tape = new Tape();
        var x = tape.Leaf(a);
        var y = tape.Leaf(b);
        tape.Backward(Evaluate(function, x, y));
        double dx = Richardson(v => Evaluate(function, v, b), a, Math.ScaleB(Math.Max(1, Math.Abs(a)), -12));
        double dy = Richardson(v => Evaluate(function, a, v), b, Math.ScaleB(Math.Max(1, Math.Abs(b)), -12));
        Close(dx, tape.GetGradient(x), relativeTolerance: 2e-8, absoluteTolerance: 1e-11, context: $"{function}, x");
        Close(dy, tape.GetGradient(y), relativeTolerance: 2e-8, absoluteTolerance: 1e-11, context: $"{function}, y");
    }

    [Theory]
    [InlineData("LogAddExp", 2.0, 2.0, 0.5, 0.5)]
    [InlineData("LogAddExp", double.PositiveInfinity, double.PositiveInfinity, 0.5, 0.5)]
    [InlineData("LogAddExp", double.NegativeInfinity, double.NegativeInfinity, 0.5, 0.5)]
    [InlineData("LogAddExp", double.PositiveInfinity, 0.0, 1.0, 0.0)]
    [InlineData("LogAddExp", double.NegativeInfinity, 0.0, 0.0, 1.0)]
    [InlineData("LogAddExp", double.NaN, 0.0, double.NaN, double.NaN)]
    [InlineData("LogSubExp", 2.0, 2.0, double.PositiveInfinity, double.NegativeInfinity)]
    [InlineData("LogSubExp", -0.0, 0.0, double.PositiveInfinity, double.NegativeInfinity)]
    [InlineData("LogSubExp", double.PositiveInfinity, 0.0, 1.0, 0.0)]
    [InlineData("LogSubExp", 0.0, double.NegativeInfinity, 1.0, 0.0)]
    [InlineData("LogSubExp", double.PositiveInfinity, double.NegativeInfinity, 1.0, 0.0)]
    [InlineData("LogSubExp", double.PositiveInfinity, double.PositiveInfinity, double.NaN, double.NaN)]
    [InlineData("LogSubExp", double.NegativeInfinity, double.NegativeInfinity, double.NaN, double.NaN)]
    [InlineData("LogSubExp", 0.0, 1.0, double.NaN, double.NaN)]
    [InlineData("LogSubExp", double.NaN, 0.0, double.NaN, double.NaN)]
    public void BinaryBoundariesHaveTheDeclaredGradients(string function, double a, double b, double dx, double dy)
    {
        var tape = new Tape();
        var x = tape.Leaf(a);
        var y = tape.Leaf(b);
        tape.Backward(Evaluate(function, x, y));
        SameNumber(dx, tape.GetGradient(x), context: $"{function}, x");
        SameNumber(dy, tape.GetGradient(y), context: $"{function}, y");
    }

    [Theory]
    [InlineData("LogAddExp")]
    [InlineData("LogSubExp")]
    public void BinaryTailsPreserveSubnormalDerivatives(string function)
    {
        var tape = new Tape();
        var x = tape.Leaf(0);
        var y = tape.Leaf(-745);
        tape.Backward(Evaluate(function, x, y));
        EqualBits(1, tape.GetGradient(x), context: $"{function}, x");
        double expected = function == "LogAddExp" ? double.Epsilon : -double.Epsilon;
        EqualBits(expected, tape.GetGradient(y), context: $"{function}, y");
    }

    [Theory]
    [InlineData("LogAddExp")]
    [InlineData("LogSubExp")]
    public void BinaryConstantsAndRepeatedInputsAccumulateOnlyTrackedContributions(string function)
    {
        var tape = new Tape();
        var x = tape.Leaf(1);
        var y = tape.Leaf(0);
        tape.Backward(Evaluate(function, x, y));
        double dx = tape.GetGradient(x), dy = tape.GetGradient(y);

        tape.Reset();
        x = tape.Leaf(1);
        tape.Backward(Evaluate(function, x, (ScalarNode)0));
        EqualBits(dx, tape.GetGradient(x), context: $"{function}, constant y");

        tape.Reset();
        y = tape.Leaf(0);
        tape.Backward(Evaluate(function, (ScalarNode)1, y));
        EqualBits(dy, tape.GetGradient(y), context: $"{function}, constant x");

        if (function == "LogAddExp")
        {
            tape.Reset();
            x = tape.Leaf(1000);
            tape.Backward(XMath.LogAddExp(x, x));
            EqualBits(1, tape.GetGradient(x), context: "LogAddExp(x, x)");
        }
    }

    [Theory]
    [InlineData("LogAddExp")]
    [InlineData("LogSubExp")]
    public void BinaryRecordingRejectsInvalidTapeStates(string function)
    {
        var tape = new Tape();
        var x = tape.Leaf(1);
        var other = new Tape().Leaf(0);
        Assert.Throws<InvalidOperationException>(() => Evaluate(function, x, other));
        tape.Backward(Evaluate(function, x, (ScalarNode)0));
        Assert.Throws<InvalidOperationException>(() => Evaluate(function, x, (ScalarNode)0));
        tape.Reset();
        Assert.Throws<InvalidOperationException>(() => Evaluate(function, x, (ScalarNode)0));
    }

    [Fact]
    public void LogSumExpValuesPreserveCoreBitsForSmallAndPooledInputs()
    {
        double[][] cases = [[], [0], [-0.0, 0.0], [0, -40], [double.MaxValue, double.MaxValue],
            [double.NegativeInfinity, double.NegativeInfinity], [double.PositiveInfinity, 0, double.PositiveInfinity],
            [double.PositiveInfinity, double.NaN], [double.NaN, double.NegativeInfinity],
            Enumerable.Range(0, 257).Select(i => (i - 128) / 3.0).ToArray()];
        var tape = new Tape();
        foreach (double[] values in cases)
        {
            tape.Reset();
            var tracked = values.Select(tape.Leaf).ToArray();
            var constants = values.Select(value => (ScalarNode)value).ToArray();
            double expected = XMath.LogSumExp<double>(new ReadOnlyVectorSpan<double>(values));
            EqualBits(expected, XMath.LogSumExp(tracked).Value, context: $"LogSumExp, tracked, length={values.Length}");
            var constant = XMath.LogSumExp(constants);
            EqualBits(expected, constant.Value, context: $"LogSumExp, constant, length={values.Length}");
            Assert.False(constant.RequiresGradient);
        }
    }

    [Fact]
    public void LogSumExpUsesTheNormalizerBeforeItRoundsIntoAHugeMaximum()
    {
        var tape = new Tape();
        var x = tape.Leaf(double.MaxValue);
        var y = tape.Leaf(double.MaxValue);
        tape.Backward(XMath.LogSumExp([x, y, (ScalarNode)double.MaxValue]));
        EqualBits(1.0 / 3, tape.GetGradient(x), context: "LogSumExp, huge x");
        EqualBits(1.0 / 3, tape.GetGradient(y), context: "LogSumExp, huge y");
    }

    [Theory]
    [InlineData(2, -744.1, 1)]
    [InlineData(2, -743.9, 1)]
    [InlineData(3, -743.5, 1)]
    [InlineData(4, -743.0, 1)]
    [InlineData(8, -742.5, 1)]
    [InlineData(256, -738.7, 1)]
    [InlineData(2, -744.8, 0)]
    [InlineData(3, -744.1, 0)]
    [InlineData(2, -743.0, 2)]
    [InlineData(3, -743.0, 1)]
    [InlineData(4, -742.0, 3)]
    [InlineData(256, -737.0, 7)]
    public void LogSumExpNormalizesBeforeRoundingSubnormalDerivatives(int maxima, double tail, int expectedUlps)
    {
        // binary64 の入力を厳密に変換し、mpmath 1.4.1 の 1100 桁で求めた局所微分。
        var tape = new Tape();
        var inputs = new ScalarNode[maxima + 1];
        for (int i = 0; i < maxima; i++)
        {
            inputs[i] = 0;
        }

        var x = tape.Leaf(tail);
        inputs[^1] = x;
        tape.Backward(XMath.LogSumExp(inputs));
        EqualBits(expectedUlps * double.Epsilon, tape.GetGradient(x),
            context: $"LogSumExp, maxima={maxima}, tail={tail:R}");
    }

    [Fact]
    public void LogSumExpDistributesInfiniteMaximaAndPropagatesNaN()
    {
        double[][] cases = [[double.PositiveInfinity, 0, double.PositiveInfinity],
            [double.NegativeInfinity, double.NegativeInfinity], [0, double.NaN, double.PositiveInfinity]];
        double[][] expected = [[0.5, 0, 0.5], [0.5, 0.5], [double.NaN, double.NaN, double.NaN]];
        var tape = new Tape();
        for (int c = 0; c < cases.Length; c++)
        {
            tape.Reset();
            var inputs = cases[c].Select(tape.Leaf).ToArray();
            tape.Backward(XMath.LogSumExp(inputs));
            for (int i = 0; i < inputs.Length; i++)
            {
                SameNumber(expected[c][i], tape.GetGradient(inputs[i]), context: $"LogSumExp, case={c}, input={i}");
            }
        }
    }

    [Fact]
    public void LogSumExpHandlesEmptySingleAndRepeatedInputs()
    {
        var empty = XMath.LogSumExp(ReadOnlySpan<ScalarNode>.Empty);
        EqualBits(double.NegativeInfinity, empty.Value, context: "LogSumExp, empty");
        Assert.False(empty.RequiresGradient);
        var tape = new Tape();
        var x = tape.Leaf(-1000);
        tape.Backward(XMath.LogSumExp([x]));
        EqualBits(1, tape.GetGradient(x), context: "LogSumExp, single");

        tape.Reset();
        x = tape.Leaf(-1000);
        tape.Backward(XMath.LogSumExp([x, x, (ScalarNode)(-1000)]));
        EqualBits(2.0 / 3, tape.GetGradient(x), context: "LogSumExp, repeated");
    }

    [Fact]
    public void LogSumExpGradientsAgreeWithRichardsonDifferences()
    {
        double[] values = [-2, 0.5, 1];
        var tape = new Tape();
        var inputs = values.Select(tape.Leaf).ToArray();
        tape.Backward(XMath.LogSumExp(inputs));
        for (int i = 0; i < inputs.Length; i++)
        {
            int index = i;
            double expected = Richardson(value =>
            {
                double original = values[index];
                values[index] = value;
                double result = XMath.LogSumExp<double>(new ReadOnlyVectorSpan<double>(values));
                values[index] = original;
                return result;
            }, values[i], Math.ScaleB(1, -12));
            Close(expected, tape.GetGradient(inputs[i]), relativeTolerance: 2e-8, absoluteTolerance: 1e-11,
                context: $"LogSumExp, input={i}");
        }
    }

    [Fact]
    public void LogSumExpRejectsInvalidTapeStates()
    {
        var tape = new Tape();
        var x = tape.Leaf(1);
        var other = new Tape().Leaf(0);
        Assert.Throws<InvalidOperationException>(() => XMath.LogSumExp([x, other]));
        tape.Backward(XMath.LogSumExp([x, (ScalarNode)0]));
        Assert.Throws<InvalidOperationException>(() => XMath.LogSumExp([x]));
        tape.Reset();
        Assert.Throws<InvalidOperationException>(() => XMath.LogSumExp([x]));
    }

    [Theory]
    [InlineData(3)]
    [InlineData(257)]
    public void LogReductionsReuseTheTapeAndScratchWithoutAllocating(int count)
    {
        var tape = new Tape();
        var inputs = new ScalarNode[count];
        void EvaluateModel()
        {
            tape.Reset();
            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i] = tape.Leaf((i - count / 2) / 8.0);
            }

            tape.Backward(XMath.LogSumExp(inputs) + XMath.LogAddExp(inputs[0], inputs[^1])
                + XMath.LogSubExp(inputs[^1], inputs[0]));
        }

        for (int i = 0; i < 8; i++)
        {
            EvaluateModel();
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 16; i++)
        {
            EvaluateModel();
        }

        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }

    private static ScalarNode Evaluate(string function, ScalarNode x, ScalarNode y) => function switch
    {
        "LogAddExp" => XMath.LogAddExp(x, y),
        "LogSubExp" => XMath.LogSubExp(x, y),
        _ => throw new ArgumentOutOfRangeException(nameof(function))
    };

    private static double Evaluate(string function, double x, double y) => function switch
    {
        "LogAddExp" => XMath.LogAddExp<double>(x, y),
        "LogSubExp" => XMath.LogSubExp<double>(x, y),
        _ => throw new ArgumentOutOfRangeException(nameof(function))
    };
}
