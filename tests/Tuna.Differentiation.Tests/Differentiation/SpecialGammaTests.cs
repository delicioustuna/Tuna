using Tuna.Differentiation;
using static Tuna.Tests.Differentiation.DifferentiationAssert;

namespace Tuna.Tests.Differentiation;

/// <summary>ガンマ・ベータ関数の値、実数の定義域と安定な局所微分</summary>
public class SpecialGammaTests
{
    [Theory]
    [InlineData("LogGamma")]
    [InlineData("Digamma")]
    public void UnaryForwardValuesHaveTheSameBitsAsCore(string function)
    {
        double[] values = [double.NaN, double.NegativeInfinity, -10, -2.5, -1.5, -1, -0.5, -1e-310, -0.0,
            0.0, double.Epsilon, 1e-310, 0.125, 1, 1.4616321449683622, 2, 10, 1e300, double.MaxValue, double.PositiveInfinity];
        var tape = new Tape();
        foreach (double value in values)
        {
            tape.Reset();
            double expected = Evaluate(function, value);
            EqualBits(expected, Evaluate(function, tape.Leaf(value)).Value, context: $"{function}({value:R}), tracked");
            var constant = Evaluate(function, (ScalarNode)value);
            EqualBits(expected, constant.Value, context: $"{function}({value:R}), constant");
            Assert.False(constant.RequiresGradient);
        }
    }

    [Theory]
    [InlineData("LogGamma", -2.25)]
    [InlineData("LogGamma", -0.125)]
    [InlineData("LogGamma", 0.125)]
    [InlineData("LogGamma", 1.25)]
    [InlineData("LogGamma", 10.0)]
    [InlineData("Digamma", -2.25)]
    [InlineData("Digamma", -0.125)]
    [InlineData("Digamma", 0.125)]
    [InlineData("Digamma", 1.25)]
    [InlineData("Digamma", 10.0)]
    public void UnaryGradientsAgreeWithRichardsonDifferences(string function, double value)
    {
        double expected = Richardson(x => Evaluate(function, x), value, Math.ScaleB(Math.Abs(value), -13));
        double actual = Gradient(x => Evaluate(function, x), value);
        Close(expected, actual, relativeTolerance: 2e-9, absoluteTolerance: 2e-13, context: $"{function}({value:R})");
    }

    [Theory]
    [InlineData("LogGamma", 0.0, double.NegativeInfinity)]
    [InlineData("LogGamma", -0.0, double.PositiveInfinity)]
    [InlineData("LogGamma", -1.0, double.NaN)]
    [InlineData("LogGamma", double.PositiveInfinity, double.PositiveInfinity)]
    [InlineData("LogGamma", double.NegativeInfinity, double.NaN)]
    [InlineData("LogGamma", double.NaN, double.NaN)]
    [InlineData("Digamma", 0.0, double.PositiveInfinity)]
    [InlineData("Digamma", -1.0, double.NaN)]
    [InlineData("Digamma", double.PositiveInfinity, 0.0)]
    [InlineData("Digamma", double.NegativeInfinity, double.NaN)]
    [InlineData("Digamma", double.NaN, double.NaN)]
    public void UnaryEndpointsHaveTheDocumentedGradients(string function, double value, double expected)
    {
        // Digamma の両符号のゼロは同じ期待値なので、InlineData を重複させず両方を評価する。
        double[] values = function == "Digamma" && value == 0 ? [0.0, -0.0] : [value];
        foreach (double input in values)
        {
            SameNumber(expected, Gradient(x => Evaluate(function, x), input), context: $"{function}({input:R})");
        }
    }

    [Fact]
    public void LogBetaForwardValuesHaveTheSameBitsAsCore()
    {
        double[] values = [double.NaN, double.NegativeInfinity, -2, -1.5, -0.5, -0.0, 0.0,
            double.Epsilon, 0.125, 1, 2, 1e300, double.MaxValue, double.PositiveInfinity];
        var tape = new Tape();
        foreach (double a in values)
        {
            foreach (double b in values)
            {
                tape.Reset();
                var x = tape.Leaf(a);
                var y = tape.Leaf(b);
                double expected = XMath.Special.LogBeta(a, b);
                EqualBits(expected, XMath.Special.LogBeta(x, y).Value, context: $"LogBeta({a:R}, {b:R}), tracked");
                EqualBits(expected, XMath.Special.LogBeta(x, (ScalarNode)b).Value, context: $"LogBeta({a:R}, {b:R}), left");
                EqualBits(expected, XMath.Special.LogBeta((ScalarNode)a, y).Value, context: $"LogBeta({a:R}, {b:R}), right");
                var constant = XMath.Special.LogBeta((ScalarNode)a, (ScalarNode)b);
                EqualBits(expected, constant.Value, context: $"LogBeta({a:R}, {b:R}), constant");
                Assert.False(constant.RequiresGradient);
            }
        }
    }

    [Theory]
    [InlineData(0.125, 0.25)]
    [InlineData(1.0, 2.0)]
    [InlineData(10.0, 30.0)]
    [InlineData(-0.125, 1.5)]
    [InlineData(-1.5, -0.25)]
    [InlineData(-2.25, 0.5)]
    public void LogBetaGradientsAgreeWithRichardsonDifferences(double a, double b)
    {
        double da = Gradient(x => XMath.Special.LogBeta(x, (ScalarNode)b), a);
        double db = Gradient(y => XMath.Special.LogBeta((ScalarNode)a, y), b);
        double expectedA = Richardson(x => XMath.Special.LogBeta(x, b), a, Math.ScaleB(Math.Abs(a), -13));
        double expectedB = Richardson(y => XMath.Special.LogBeta(a, y), b, Math.ScaleB(Math.Abs(b), -13));
        Close(expectedA, da, relativeTolerance: 3e-9, absoluteTolerance: 2e-12, context: $"LogBeta({a:R}, {b:R}), da");
        Close(expectedB, db, relativeTolerance: 3e-9, absoluteTolerance: 2e-12, context: $"LogBeta({a:R}, {b:R}), db");
    }

    [Theory]
    [InlineData(-1.5)]
    [InlineData(-0.5)]
    [InlineData(-1e-310)]
    [InlineData(1e-310)]
    [InlineData(1e-154)]
    [InlineData(1.0)]
    [InlineData(1e300)]
    [InlineData(double.MaxValue)]
    public void LogBetaWithSecondArgumentOnePreservesTheReciprocalIdentity(double a)
    {
        double expected = -1 / a;
        double actual = Gradient(x => XMath.Special.LogBeta(x, (ScalarNode)1), a);
        if (double.IsInfinity(expected))
        {
            EqualBits(expected, actual, context: $"LogBeta({a:R}, 1), da");
        }
        else
        {
            WithinUlps(expected, actual, 4, context: $"LogBeta({a:R}, 1), da");
        }
    }

    [Theory]
    [InlineData(-0.5, 2.0)]
    [InlineData(-1.5, 4.0)]
    [InlineData(-2.5, 6.0)]
    public void LogBetaReflectionKeepsExactZeros(double a, double b)
    {
        EqualBits(0, Gradient(x => XMath.Special.LogBeta(x, (ScalarNode)b), a), context: $"LogBeta({a:R}, {b:R}), da");
    }

    [Fact]
    public void LogBetaDerivativesRemainFiniteWhenTheSumOverflows()
    {
        var tape = new Tape();
        var a = tape.Leaf(double.MaxValue);
        var b = tape.Leaf(double.MaxValue);
        var result = XMath.Special.LogBeta(a, b);
        tape.Backward(result);
        WithinUlps(-Math.Log(2), tape.GetGradient(a), 2, context: "LogBeta(max, max), da");
        EqualBits(tape.GetGradient(a), tape.GetGradient(b), context: "LogBeta(max, max), symmetry");
    }

    [Theory]
    [InlineData(-1.0, true, 106)]
    [InlineData(-1.0, false, 104)]
    [InlineData(-2.0, true, 104)]
    [InlineData(-2.0, false, 102)]
    public void LogBetaKeepsTinyShiftsOnBothSidesOfGammaPoles(double pole, bool above, int inverseSquareExponent)
    {
        double a = above ? Math.BitIncrement(pole) : Math.BitDecrement(pole);
        foreach (double b in new[] { double.Epsilon, -double.Epsilon })
        {
            // 極までの距離は2の整数乗。微分の主項 -b/(a-pole)² の残りは double の丸めより小さい。
            double expected = Math.ScaleB(-b, inverseSquareExponent);
            WithinUlps(expected, Gradient(x => XMath.Special.LogBeta(x, (ScalarNode)b), a), 4,
                context: $"LogBeta({a:R}, {b:R}), tracked da");
        }
    }

    [Theory]
    [InlineData(-0.01, 1.4844653848010512e43, -2.800916901822812e-17, 99.4062136959444)]
    [InlineData(-0.002, 7.854680242344467e216, -1.350283835508655e-17, 499.41948965006196)]
    public void LogBetaRecoversNonzeroPartialsAfterSevereReflectionCancellation(double a, double b, double expected, double conditionScale)
    {
        // mpmath 1.4.1、1100/1180桁で一致する期待値と max(1, |ψ(a)|, |ψ(a+b)|)。
        // 拡張精度の丸めの尺度を使い、通常精度の項どうしの引き算で0になる退行を検出する。
        double absoluteTolerance = 64 * Math.ScaleB(1.0, -104) * conditionScale;
        double actual = Gradient(x => XMath.Special.LogBeta(x, (ScalarNode)b), a);
        Assert.True(double.IsFinite(actual) && actual < 0);
        Close(expected, actual, relativeTolerance: 0, absoluteTolerance: absoluteTolerance,
            context: $"LogBeta({a:R}, {b:R}), tracked da");
    }

    [Theory]
    [InlineData(0.0, 2.0, double.NaN, double.NaN)]
    [InlineData(-1.0, 2.0, double.NaN, double.NaN)]
    [InlineData(-0.5, -0.5, double.NaN, double.NaN)]
    [InlineData(double.NaN, 2.0, double.NaN, double.NaN)]
    [InlineData(double.NegativeInfinity, 2.0, double.NaN, double.NaN)]
    [InlineData(double.PositiveInfinity, 2.0, 0.0, double.NegativeInfinity)]
    [InlineData(2.0, double.PositiveInfinity, double.NegativeInfinity, 0.0)]
    [InlineData(double.PositiveInfinity, double.PositiveInfinity, double.NaN, double.NaN)]
    public void LogBetaPolesAndInfinitiesHaveTheDocumentedGradients(double a, double b, double expectedA, double expectedB)
    {
        var tape = new Tape();
        var x = tape.Leaf(a);
        var y = tape.Leaf(b);
        tape.Backward(XMath.Special.LogBeta(x, y));
        SameNumber(expectedA, tape.GetGradient(x), context: $"LogBeta({a:R}, {b:R}), da");
        SameNumber(expectedB, tape.GetGradient(y), context: $"LogBeta({a:R}, {b:R}), db");
    }

    [Fact]
    public void RecordingChecksAreSharedWithArithmetic()
    {
        var tape = new Tape();
        var stale = tape.Leaf(1.5);
        tape.Reset();
        foreach (string function in new[] { "LogGamma", "Digamma" })
        {
            Throws<InvalidOperationException>(() => Evaluate(function, stale), $"{function}, stale");
        }

        var current = tape.Leaf(1.5);
        var foreign = new Tape().Leaf(2.5);
        Throws<InvalidOperationException>(() => XMath.Special.LogBeta(stale, current), "LogBeta, stale");
        Throws<InvalidOperationException>(() => XMath.Special.LogBeta(current, foreign), "LogBeta, different tapes");
        tape.Backward(current);
        foreach (string function in new[] { "LogGamma", "Digamma" })
        {
            Throws<InvalidOperationException>(() => Evaluate(function, current), $"{function}, after backward");
        }

        Throws<InvalidOperationException>(() => XMath.Special.LogBeta(current, (ScalarNode)2), "LogBeta, after backward");
    }

    [Fact]
    public void RepeatedGammaAndBetaModelAllocatesNothingAfterWarmup()
    {
        var tape = new Tape();
        for (int i = 0; i < 32; i++)
        {
            EvaluateModel(tape);
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 128; i++)
        {
            EvaluateModel(tape);
        }

        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }

    private static void EvaluateModel(Tape tape)
    {
        tape.Reset();
        var a = tape.Leaf(1.5);
        var b = tape.Leaf(2.5);
        var result = XMath.Special.LogGamma(a) + XMath.Special.Digamma(b) + XMath.Special.LogBeta(a, b);
        tape.Backward(result);
    }

    private static ScalarNode Evaluate(string function, ScalarNode x) => function switch
    {
        "LogGamma" => XMath.Special.LogGamma(x),
        "Digamma" => XMath.Special.Digamma(x),
        _ => throw new ArgumentOutOfRangeException(nameof(function))
    };

    private static double Evaluate(string function, double x) => function switch
    {
        "LogGamma" => XMath.Special.LogGamma(x),
        "Digamma" => XMath.Special.Digamma(x),
        _ => throw new ArgumentOutOfRangeException(nameof(function))
    };
}
