using Xunit.Abstractions;

namespace Tuna.Tests;

/// <summary>ガンマ系の負の零点と、通常域・端点の高精度参照との照合</summary>
[Trait("SpecialFunction", "Gamma")]
public class GammaValueReferenceTests(ITestOutputHelper output)
{
    private const double RelativeTolerance = 16 * 2.220446049250313e-16;
    private const ulong UlpTolerance = 4;

    [Theory]
    [InlineData("Digamma")]
    [InlineData("LogGamma")]
    public void ValuesMatchIndependentHighPrecisionReferences(string function)
    {
        int count = 0;
        int zeroCount = 0;
        int extendedZeroCount = 0;
        int boundaryCount = 0;
        double maximumRelativeError = 0;
        double worstInput = double.NaN;
        ulong maximumUlps = 0;
        foreach (var reference in GammaValueReferences.Rows.Where(row => row.Function == function))
        {
            double actual = Evaluate(function, reference.X);
            double expected = reference.Expected;
            string context = $"{function}({reference.X:R}), {reference.Region}";
            count++;
            zeroCount += reference.Region.StartsWith("negative-zero", StringComparison.Ordinal) ? 1 : 0;
            extendedZeroCount += reference.Region == "negative-zero-extended-range" ? 1 : 0;
            boundaryCount += reference.Region == "local-expansion-boundary" ? 1 : 0;
            if (double.IsInfinity(expected))
            {
                Assert.True(actual == expected, $"{context}: expected {expected:R}, actual {actual:R}");
                continue;
            }

            Assert.True(double.IsFinite(actual), $"{context}: expected {expected:R}, actual {actual:R}");
            double relativeError = expected == 0 ? (actual == 0 ? 0 : double.PositiveInfinity) : Math.Abs((actual - expected) / expected);
            ulong ulps = UlpDistance(expected, actual);
            maximumUlps = Math.Max(maximumUlps, ulps);
            if (double.IsNormal(expected) && (double.IsNaN(worstInput) || relativeError > maximumRelativeError))
            {
                maximumRelativeError = relativeError;
                worstInput = reference.X;
            }

            Assert.True(relativeError <= RelativeTolerance || ulps <= UlpTolerance,
                $"{context}: expected {expected:R}, actual {actual:R}, relative error {relativeError:E17}, ULP distance {ulps}");
        }

        string coverage = $"{function}: samples={count}, negative-zero samples={zeroCount}, extended-zero samples={extendedZeroCount}, local-expansion boundary samples={boundaryCount}";
        Assert.True(count > 0 && zeroCount > 0 && extendedZeroCount > 0, coverage);
        Assert.True(function != "LogGamma" || boundaryCount > 0, coverage);
        output.WriteLine(FormattableString.Invariant(
            $"{function}: samples={count}, negative-zero samples={zeroCount}, max relative error (normal references)={maximumRelativeError:E17} at x={worstInput:R}, max ULP distance (all finite references)={maximumUlps}"));
        output.WriteLine($"{function}: local-expansion boundary samples={boundaryCount}");
        output.WriteLine($"{function}: extended-zero samples={extendedZeroCount}");
    }

    [Theory]
    [InlineData("Digamma", 0.0, double.NegativeInfinity)]
    [InlineData("Digamma", -0.0, double.PositiveInfinity)]
    [InlineData("Digamma", -1.0, double.NaN)]
    [InlineData("Digamma", -12.0, double.NaN)]
    [InlineData("Digamma", double.NegativeInfinity, double.NaN)]
    [InlineData("Digamma", double.PositiveInfinity, double.PositiveInfinity)]
    [InlineData("Digamma", double.NaN, double.NaN)]
    [InlineData("LogGamma", -1.0, double.PositiveInfinity)]
    [InlineData("LogGamma", -12.0, double.PositiveInfinity)]
    [InlineData("LogGamma", double.NegativeInfinity, double.PositiveInfinity)]
    [InlineData("LogGamma", double.PositiveInfinity, double.PositiveInfinity)]
    [InlineData("LogGamma", double.NaN, double.NaN)]
    public void PolesAndNonFiniteInputsKeepTheExistingConventions(string function, double x, double expected)
    {
        double actual = Evaluate(function, x);
        Assert.True(double.IsNaN(expected) ? double.IsNaN(actual) : actual == expected,
            $"{function}({x:R}): expected {expected:R}, actual {actual:R}");
    }

    [Fact]
    public void BothSignedZerosRemainLogGammaPoles()
    {
        foreach (double x in new double[] { 0.0, -0.0 })
        {
            Assert.Equal(double.PositiveInfinity, XMath.Special.LogGamma(x));
        }
    }

    private static double Evaluate(string function, double x) => function switch
    {
        "Digamma" => XMath.Special.Digamma(x),
        "LogGamma" => XMath.Special.LogGamma(x),
        _ => throw new ArgumentOutOfRangeException(nameof(function))
    };

    private static ulong UlpDistance(double first, double second)
    {
        ulong a = OrderedBits(first), b = OrderedBits(second);
        return a >= b ? a - b : b - a;
    }

    private static ulong OrderedBits(double value)
    {
        ulong bits = BitConverter.DoubleToUInt64Bits(value);
        const ulong sign = 1UL << 63;
        return (bits & sign) == 0 ? sign + bits : sign - (bits & ~sign);
    }
}
