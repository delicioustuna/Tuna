using Tuna;
using Tuna.Differentiation;
using static Tuna.Tests.Differentiation.DifferentiationAssert;

namespace Tuna.Tests.Differentiation;

/// <summary>誤差関数と正規分布の値・局所微分・記録の検証</summary>
public class SpecialNormalTests
{
    [Theory]
    [InlineData("Erf")]
    [InlineData("Erfc")]
    [InlineData("NormalPdf")]
    [InlineData("NormalLogPdf")]
    [InlineData("NormalCdf")]
    [InlineData("NormalSf")]
    [InlineData("NormalLogCdf")]
    [InlineData("NormalLogSf")]
    public void ForwardValuesHaveTheSameBitsAsCoreFunctions(string function)
    {
        double[] values = [-double.MaxValue, -1e154, -100, -38.6, -27.299, -1, -double.Epsilon, -0.0,
            0.0, double.Epsilon, 1, 27.299, 38.6, 100, 1e154, double.MaxValue,
            double.NegativeInfinity, double.PositiveInfinity, double.NaN];
        var tape = new Tape();
        foreach (double value in values)
        {
            tape.Reset();
            var x = tape.Leaf(value);
            double expected = Evaluate(function, value);
            EqualBits(expected, Evaluate(function, x).Value, context: $"{function}({value:R}), tracked");
            var constant = Evaluate(function, (ScalarNode)value);
            EqualBits(expected, constant.Value, context: $"{function}({value:R}), constant");
            Assert.False(constant.RequiresGradient, $"{function}({value:R}), constant");
        }
    }

    [Theory]
    [InlineData("Erf", -3.0)]
    [InlineData("Erf", 0.0)]
    [InlineData("Erf", 0.7)]
    [InlineData("Erfc", -0.3)]
    [InlineData("Erfc", 0.0)]
    [InlineData("Erfc", 3.0)]
    [InlineData("NormalPdf", -3.0)]
    [InlineData("NormalPdf", 0.0)]
    [InlineData("NormalPdf", 0.7)]
    [InlineData("NormalLogPdf", -3.0)]
    [InlineData("NormalLogPdf", 0.0)]
    [InlineData("NormalLogPdf", 0.7)]
    [InlineData("NormalCdf", -3.0)]
    [InlineData("NormalCdf", 0.0)]
    [InlineData("NormalCdf", 0.7)]
    [InlineData("NormalSf", -0.3)]
    [InlineData("NormalSf", 0.0)]
    [InlineData("NormalSf", 3.0)]
    [InlineData("NormalLogCdf", -20.0)]
    [InlineData("NormalLogCdf", 0.0)]
    [InlineData("NormalLogCdf", 3.0)]
    [InlineData("NormalLogSf", -3.0)]
    [InlineData("NormalLogSf", 0.0)]
    [InlineData("NormalLogSf", 20.0)]
    public void GradientsAgreeWithRichardsonDifferences(string function, double value)
    {
        double step = Math.ScaleB(Math.Max(1, Math.Abs(value)), -12);
        double expected = Richardson(x => Evaluate(function, x), value, step);
        double actual = Gradient(x => Evaluate(function, x), value);
        Close(expected, actual, relativeTolerance: 2e-8, absoluteTolerance: 2e-12, context: $"{function}({value:R})");
    }

    [Theory]
    [InlineData("Erf", 1.1283791670955126)]
    [InlineData("Erfc", -1.1283791670955126)]
    [InlineData("NormalPdf", 0.0)]
    [InlineData("NormalLogPdf", 0.0)]
    [InlineData("NormalCdf", 0.3989422804014327)]
    [InlineData("NormalSf", -0.3989422804014327)]
    [InlineData("NormalLogCdf", 0.7978845608028654)]
    [InlineData("NormalLogSf", -0.7978845608028654)]
    public void GradientsAtZeroHaveTheAnalyticValues(string function, double expected)
    {
        EqualBits(expected, Gradient(x => Evaluate(function, x), 0), context: $"{function}(0)");
        EqualBits(expected, Gradient(x => Evaluate(function, x), -0.0), context: $"{function}(-0)");
    }

    [Theory]
    [InlineData("Erf", double.NegativeInfinity, 0.0)]
    [InlineData("Erf", double.PositiveInfinity, 0.0)]
    [InlineData("Erfc", double.NegativeInfinity, 0.0)]
    [InlineData("Erfc", double.PositiveInfinity, 0.0)]
    [InlineData("NormalPdf", double.NegativeInfinity, 0.0)]
    [InlineData("NormalPdf", double.PositiveInfinity, 0.0)]
    [InlineData("NormalLogPdf", double.NegativeInfinity, double.PositiveInfinity)]
    [InlineData("NormalLogPdf", double.PositiveInfinity, double.NegativeInfinity)]
    [InlineData("NormalCdf", double.NegativeInfinity, 0.0)]
    [InlineData("NormalCdf", double.PositiveInfinity, 0.0)]
    [InlineData("NormalSf", double.NegativeInfinity, 0.0)]
    [InlineData("NormalSf", double.PositiveInfinity, 0.0)]
    [InlineData("NormalLogCdf", double.NegativeInfinity, double.PositiveInfinity)]
    [InlineData("NormalLogCdf", double.PositiveInfinity, 0.0)]
    [InlineData("NormalLogSf", double.NegativeInfinity, 0.0)]
    [InlineData("NormalLogSf", double.PositiveInfinity, double.NegativeInfinity)]
    public void InfiniteInputsHaveTheDocumentedGradients(string function, double value, double expected)
    {
        // ゼロの辺はテープが飛ばすため、葉に残る勾配は +0 とする。
        EqualBits(expected, Gradient(x => Evaluate(function, x), value), context: $"{function}({value:R})");
    }

    [Theory]
    [InlineData("Erf")]
    [InlineData("Erfc")]
    [InlineData("NormalPdf")]
    [InlineData("NormalLogPdf")]
    [InlineData("NormalCdf")]
    [InlineData("NormalSf")]
    [InlineData("NormalLogCdf")]
    [InlineData("NormalLogSf")]
    public void NaNGradientsAndTapeLifetimesArePreserved(string function)
    {
        Assert.True(double.IsNaN(Gradient(x => Evaluate(function, x), double.NaN)), $"{function}(NaN)");
        var tape = new Tape();
        var old = tape.Leaf(0.5);
        tape.Reset();
        Throws<InvalidOperationException>(() => Evaluate(function, old), $"{function}: stale generation");
        var current = tape.Leaf(0.5);
        tape.Backward(current);
        Throws<InvalidOperationException>(() => Evaluate(function, current), $"{function}: completed backward");
    }

    [Theory]
    [InlineData(27.298)]
    [InlineData(27.299)]
    public void ErrorFunctionGradientsRemainAfterTheUnscaledExponentialUnderflows(double value)
    {
        Assert.Equal(0, Math.Exp(-value * value));
        EqualBits(double.Epsilon, Gradient(XMath.Special.Erf, value), context: $"Erf({value:R})");
        EqualBits(-double.Epsilon, Gradient(XMath.Special.Erfc, value), context: $"Erfc({value:R})");
        EqualBits(double.Epsilon, Gradient(XMath.Special.Erf, -value), context: $"Erf({-value:R})");
    }

    [Theory]
    [InlineData(38.6, -4.4e-323)]
    [InlineData(38.65, -double.Epsilon)]
    public void DensityGradientsRemainAfterTheDensityRoundsToZero(double value, double expected)
    {
        Assert.Equal(0, XMath.Special.NormalPdf(value));
        EqualBits(expected, Gradient(XMath.Special.NormalPdf, value), context: $"NormalPdf({value:R})");
        EqualBits(-expected, Gradient(XMath.Special.NormalPdf, -value), context: $"NormalPdf({-value:R})");
    }

    [Theory]
    [InlineData(134217728.0)]
    [InlineData(1e154)]
    [InlineData(1e200)]
    [InlineData(double.MaxValue)]
    public void LogTailGradientsMatchTheRoundedAsymptote(double magnitude)
    {
        EqualBits(magnitude, Gradient(XMath.Special.NormalLogCdf, -magnitude), context: $"NormalLogCdf({-magnitude:R})");
        EqualBits(-magnitude, Gradient(XMath.Special.NormalLogSf, magnitude), context: $"NormalLogSf({magnitude:R})");
    }

    private static ScalarNode Evaluate(string function, ScalarNode x) => function switch
    {
        "Erf" => XMath.Special.Erf(x),
        "Erfc" => XMath.Special.Erfc(x),
        "NormalPdf" => XMath.Special.NormalPdf(x),
        "NormalLogPdf" => XMath.Special.NormalLogPdf(x),
        "NormalCdf" => XMath.Special.NormalCdf(x),
        "NormalSf" => XMath.Special.NormalSf(x),
        "NormalLogCdf" => XMath.Special.NormalLogCdf(x),
        "NormalLogSf" => XMath.Special.NormalLogSf(x),
        _ => throw new ArgumentOutOfRangeException(nameof(function)),
    };

    private static double Evaluate(string function, double x) => function switch
    {
        "Erf" => XMath.Special.Erf(x),
        "Erfc" => XMath.Special.Erfc(x),
        "NormalPdf" => XMath.Special.NormalPdf(x),
        "NormalLogPdf" => XMath.Special.NormalLogPdf(x),
        "NormalCdf" => XMath.Special.NormalCdf(x),
        "NormalSf" => XMath.Special.NormalSf(x),
        "NormalLogCdf" => XMath.Special.NormalLogCdf(x),
        "NormalLogSf" => XMath.Special.NormalLogSf(x),
        _ => throw new ArgumentOutOfRangeException(nameof(function)),
    };
}
