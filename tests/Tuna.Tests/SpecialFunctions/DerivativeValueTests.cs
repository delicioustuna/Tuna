using static Tuna.Tests.ReferenceData.ReferencePrecision;

namespace Tuna.Tests;

/// <summary>特殊関数の導関数値の端点・裾と符号付きゼロ</summary>
public class DerivativeValueTests
{
    [Theory]
    [InlineData("Erf", double.NegativeInfinity, 0.0)]
    [InlineData("Erf", double.PositiveInfinity, 0.0)]
    [InlineData("Erf", double.NaN, double.NaN)]
    [InlineData("NormalPdf", double.NegativeInfinity, 0.0)]
    [InlineData("NormalPdf", double.PositiveInfinity, 0.0)]
    [InlineData("NormalPdf", double.NaN, double.NaN)]
    [InlineData("NormalHazard", double.NegativeInfinity, 0.0)]
    [InlineData("NormalHazard", double.PositiveInfinity, double.PositiveInfinity)]
    [InlineData("NormalHazard", double.NaN, double.NaN)]
    [InlineData("NormalReverseHazard", double.NegativeInfinity, double.PositiveInfinity)]
    [InlineData("NormalReverseHazard", double.PositiveInfinity, 0.0)]
    [InlineData("NormalReverseHazard", double.NaN, double.NaN)]
    public void NormalDerivativeEndpointsHaveTheDocumentedValues(string function, double value, double expected)
    {
        double actual = function switch
        {
            "Erf" => XMath.Special.ErfDerivative(value),
            "NormalPdf" => XMath.Special.NormalPdfDerivative(value),
            "NormalHazard" => XMath.Special.NormalHazard(value),
            "NormalReverseHazard" => XMath.Special.NormalReverseHazard(value),
            _ => throw new ArgumentOutOfRangeException(nameof(function))
        };
        EqualValue(expected, actual, $"{function}({value:R})");
    }

    [Fact]
    public void DensityDerivativePreservesTheSignOfZero()
    {
        EqualBits(-0.0, XMath.Special.NormalPdfDerivative(0.0), context: "NormalPdfDerivative(+0)");
        EqualBits(0.0, XMath.Special.NormalPdfDerivative(-0.0), context: "NormalPdfDerivative(-0)");
    }

    [Theory]
    [InlineData(27.298)]
    [InlineData(27.299)]
    public void ErrorFunctionDerivativeRemainsAfterTheExponentialUnderflows(double value)
    {
        Assert.Equal(0, Math.Exp(-value * value));
        EqualBits(double.Epsilon, XMath.Special.ErfDerivative(value), context: $"ErfDerivative({value:R})");
        EqualBits(double.Epsilon, XMath.Special.ErfDerivative(-value), context: $"ErfDerivative({-value:R})");
    }

    [Theory]
    [InlineData(38.6, -4.4e-323)]
    [InlineData(38.65, -double.Epsilon)]
    public void DensityDerivativeRemainsAfterTheDensityRoundsToZero(double value, double expected)
    {
        Assert.Equal(0, XMath.Special.NormalPdf(value));
        EqualBits(expected, XMath.Special.NormalPdfDerivative(value), context: $"NormalPdfDerivative({value:R})");
        EqualBits(-expected, XMath.Special.NormalPdfDerivative(-value), context: $"NormalPdfDerivative({-value:R})");
    }

    [Theory]
    [InlineData(134217728.0)]
    [InlineData(1e154)]
    [InlineData(1e200)]
    [InlineData(double.MaxValue)]
    public void NormalHazardsMatchTheRoundedAsymptote(double magnitude)
    {
        EqualBits(magnitude, XMath.Special.NormalHazard(magnitude), context: $"NormalHazard({magnitude:R})");
        EqualBits(magnitude, XMath.Special.NormalReverseHazard(-magnitude), context: $"NormalReverseHazard({-magnitude:R})");
    }

    [Fact]
    public void LogBetaPreservesSubnormalDerivativesAndTinyShifts()
    {
        SpecialFunctionAssert.WithinUlps(-1e-320, XMath.Special.LogBetaDerivative(1e300, 1e-20), 1);
        double a = 1e-310;
        double b = double.Epsilon;
        foreach (double x in new[] { a, -a })
        {
            double expected = -(b / x) / (x + b);
            double actual = XMath.Special.LogBetaDerivative(x, b);
            SpecialFunctionAssert.Close(expected, actual, relativeTolerance: 2e-15);
        }
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
            // 極までの距離は2の整数乗。主項の残りは double の丸めより小さい。
            double expected = Math.ScaleB(-b, inverseSquareExponent);
            SpecialFunctionAssert.WithinUlps(expected, XMath.Special.LogBetaDerivative(a, b), 4);
        }
    }

    [Theory]
    [InlineData(-0.5, 2.0)]
    [InlineData(-1.5, 4.0)]
    [InlineData(-2.5, 6.0)]
    public void LogBetaReflectionKeepsExactZeros(double a, double b)
    {
        Assert.Equal(0, XMath.Special.LogBetaDerivative(a, b));
    }

    [Theory]
    [InlineData(0.0, 2.0, double.NaN, double.NaN)]
    [InlineData(-1.0, 2.0, double.NaN, double.NaN)]
    [InlineData(-0.5, -0.5, double.NaN, double.NaN)]
    [InlineData(double.NaN, 2.0, double.NaN, double.NaN)]
    [InlineData(double.NegativeInfinity, 2.0, double.NaN, double.NaN)]
    [InlineData(double.PositiveInfinity, 2.0, -0.0, double.NegativeInfinity)]
    [InlineData(2.0, double.PositiveInfinity, double.NegativeInfinity, -0.0)]
    [InlineData(double.PositiveInfinity, -0.5, 0.0, double.NegativeInfinity)]
    [InlineData(double.PositiveInfinity, double.PositiveInfinity, double.NaN, double.NaN)]
    public void LogBetaPolesAndInfinitiesHaveTheDocumentedPartials(double a, double b, double expectedA, double expectedB)
    {
        EqualValue(expectedA, XMath.Special.LogBetaDerivative(a, b), $"LogBetaDerivative({a:R}, {b:R})");
        EqualValue(expectedB, XMath.Special.LogBetaDerivative(b, a), $"LogBetaDerivative({b:R}, {a:R})");
    }

    private static void EqualValue(double expected, double actual, string context)
    {
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual), context);
        }
        else
        {
            EqualBits(expected, actual, context: context);
        }
    }
}
