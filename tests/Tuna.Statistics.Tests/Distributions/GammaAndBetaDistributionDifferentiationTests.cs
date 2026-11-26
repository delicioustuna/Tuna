using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Statistics;

public class GammaAndBetaDistributionDifferentiationTests
{
    // mpmath 1.4.1、110 桁。入力は double の値をそのまま変換した。
    [Theory]
    [InlineData(false, 0.4, 1.7, 0.2, 0, 0.4213183810888453, -0.1660899653979239, -3.5882352941176467)]
    [InlineData(false, 0.4, 1.7, 0.2, 1, -2.138308883582195, -0.2162034919815341, 1.8377296818430398)]
    [InlineData(false, 0.4, 1.7, 0.2, 2, 1.8456504456804825, 0.1866129231362081, -1.5862098466577685)]
    [InlineData(false, 3.75, 2.1, 6.3, 0, -0.0839250999436866, -0.3571428571428572, -0.03968253968253965)]
    [InlineData(false, 3.75, 2.1, 6.3, 1, -0.5320364877606806, -0.8146939724520412, 0.27156465748401376)]
    [InlineData(false, 3.75, 2.1, 6.3, 2, 0.3620536983250724, 0.5544036405677608, -0.18480121352258694)]
    [InlineData(false, 28.0, 1.125, 45.0, 0, 0.37463836576338677, 10.666666666666666, -0.28888888888888886)]
    [InlineData(false, 28.0, 1.125, 45.0, 1, -0.008571704841504733, -0.25482747691787516, 0.006370686922946879)]
    [InlineData(false, 28.0, 1.125, 45.0, 2, 0.43466290432930554, 12.922056144969691, -0.32305140362424223)]
    [InlineData(false, 100.0, 1.0, 220.0, 0, 0.7934656936142741, 120.0, -0.55)]
    [InlineData(false, 100.0, 1.0, 220.0, 1, -3.513863530061333e-20, -5.3392272862574426e-18, 2.4269214937533828e-20)]
    [InlineData(false, 100.0, 1.0, 220.0, 2, 0.8015593165043778, 121.79492281418621, -0.5536132855190282)]
    [InlineData(false, 1.0, 1.0, 900.0, 0, 7.379610428225844, 899.0, -1.0)]
    [InlineData(false, 1.0, 1.0, 900.0, 1, 0.0, 0.0, 0.0)]
    [InlineData(false, 1.0, 1.0, 900.0, 2, 7.3807203075034336, 900.0, -1.0)]
    [InlineData(true, 0.4, 1.7, 0.2, 0, 4.701450708081387, -4.764705882352941, 35.49999999999999)]
    [InlineData(true, 0.4, 1.7, 0.2, 1, 4.802301778586773, -5.320397110440566, 45.2233754387448)]
    [InlineData(true, 0.4, 1.7, 0.2, 2, -0.00011464062336820944, 0.00012700860325500385, -0.0010795731276675327)]
    [InlineData(true, 3.75, 2.1, 6.3, 0, -2.281149677279906, 1.626984126984127, -0.701058201058201)]
    [InlineData(true, 3.75, 2.1, 6.3, 1, 0.0019345003259587771, -0.0012543624021599655, 0.0004181208007199885)]
    [InlineData(true, 3.75, 2.1, 6.3, 2, -2.5631271116792127, 1.6619745355968383, -0.5539915118656128)]
    [InlineData(true, 28.0, 1.125, 45.0, 0, -7.003120542464486, 24.866666666666667, -0.6438888888888888)]
    [InlineData(true, 28.0, 1.125, 45.0, 1, 3.1275136277292814e-74, -1.1049116872248835e-73, 2.7622792180622087e-75)]
    [InlineData(true, 28.0, 1.125, 45.0, 2, -7.0388645778037, 24.867433567204785, -0.6216858391801195)]
    [InlineData(true, 100.0, 1.0, 220.0, 0, -9.99378939909045, 99.99545454545455, -0.4590702479338843)]
    [InlineData(true, 100.0, 1.0, 220.0, 1, 0.0, 0.0, 0.0)]
    [InlineData(true, 100.0, 1.0, 220.0, 2, -10.003789844699028, 99.99549955194077, -0.45452499796336715)]
    [InlineData(true, 1.0, 1.0, 900.0, 0, -6.225179098422778, 0.9988888888888889, -0.002220987654320988)]
    [InlineData(true, 1.0, 1.0, 900.0, 1, 0.008032747247992985, -0.0011111111111111111, 1.234567901234568e-06)]
    [InlineData(true, 1.0, 1.0, 900.0, 2, -7.225456893342568, 0.9994445473251008, -0.0011104939414723343)]
    public void DerivativesMatchReference(bool inverse, double shape, double scale, double x, int operation,
        double expectedShape, double expectedScale, double expectedX)
    {
        var tape = new Tape();
        var a = tape.Leaf(shape);
        var s = tape.Leaf(scale);
        var observation = tape.Leaf(x);
        IContinuousDistribution distribution = inverse ? new InverseGammaDistribution(shape, scale) : new GammaDistribution(shape, scale);
        var output = (inverse, operation) switch
        {
            (false, 0) => GammaDistribution.LogPdf(a, s, observation),
            (false, 1) => GammaDistribution.LogCdf(a, s, observation),
            (false, _) => GammaDistribution.LogSf(a, s, observation),
            (true, 0) => InverseGammaDistribution.LogPdf(a, s, observation),
            (true, 1) => InverseGammaDistribution.LogCdf(a, s, observation),
            _ => InverseGammaDistribution.LogSf(a, s, observation),
        };
        double expected = operation switch
        {
            0 => distribution.LogPdf(x),
            1 => distribution.LogCdf(x),
            _ => distribution.LogSf(x),
        };
        Assert.Equal(BitConverter.DoubleToInt64Bits(expected), BitConverter.DoubleToInt64Bits(output.Value));
        tape.Backward(output);
        Close(expectedShape, tape.GetGradient(a));
        Close(expectedScale, tape.GetGradient(s));
        Close(expectedX, tape.GetGradient(observation));
    }

    [Fact]
    public void UpperTailRetainsSlopeWhenLogarithmsWouldCancel()
    {
        var tape = new Tape();
        var x = tape.Leaf(1e250);
        var output = GammaDistribution.LogSf(1, 2, x);
        Assert.Equal(-5e249, output.Value);
        tape.Backward(output);
        Close(-0.5, tape.GetGradient(x));
    }

    [Fact]
    public void LowerTailRetainsGradientWhenStandardizedValueUnderflows()
    {
        var tape = new Tape();
        var shape = tape.Leaf(1);
        var scale = tape.Leaf(1e200);
        var x = tape.Leaf(1e-200);
        var output = GammaDistribution.LogCdf(shape, scale, x);
        tape.Backward(output);
        Close(-921.4568215327167, tape.GetGradient(shape));
        Close(-1e-200, tape.GetGradient(scale));
        Close(1e200, tape.GetGradient(x));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void StridedBatchRecordsOnlyAggregatedDerivatives(bool inverse)
    {
        double[] storage = [0.5, double.NaN, 1.2, double.NaN, 2.4, double.NaN];
        ReadOnlyVectorSpan<double> observations = new(storage, [3], [2]);
        var tape = new Tape();
        var shape = tape.Leaf(2.3);
        var scale = tape.Leaf(1.7);
        var scalarSum = (ScalarNode)0;
        for (int i = 0; i < observations.Length; i++)
        {
            scalarSum += inverse ? InverseGammaDistribution.LogPdf(shape, scale, observations[i])
                : GammaDistribution.LogPdf(shape, scale, observations[i]);
        }

        var batch = inverse ? InverseGammaDistribution.LogPdf(shape, scale, observations)
            : GammaDistribution.LogPdf(shape, scale, observations);
        Assert.Equal(BitConverter.DoubleToInt64Bits(scalarSum.Value), BitConverter.DoubleToInt64Bits(batch.Value));
        tape.Backward(scalarSum);
        double dShape = tape.GetGradient(shape), dScale = tape.GetGradient(scale);

        tape.Reset();
        shape = tape.Leaf(2.3);
        scale = tape.Leaf(1.7);
        batch = inverse ? InverseGammaDistribution.LogPdf(shape, scale, observations)
            : GammaDistribution.LogPdf(shape, scale, observations);
        Array.Fill(storage, double.NaN);
        tape.Backward(batch);
        Close(dShape, tape.GetGradient(shape));
        Close(dScale, tape.GetGradient(scale));
    }

    [Fact]
    public void ChiSquaredReusesGammaWithHalfDegreesOfFreedom()
    {
        var tape = new Tape();
        var nu = tape.Leaf(3.5);
        var x = tape.Leaf(0.03);
        var chiSquared = ChiSquaredDistribution.LogCdf(nu, x);
        double expected = new ChiSquaredDistribution(3.5).LogCdf(0.03);
        Assert.Equal(BitConverter.DoubleToInt64Bits(expected), BitConverter.DoubleToInt64Bits(chiSquared.Value));
        tape.Backward(chiSquared);
        Close(0.5 * XMath.Special.LogRegularizedGammaPShapeDerivative(1.75, 0.015), tape.GetGradient(nu));
    }

    // mpmath 1.4.1、上側は変数を反転し、対応する下側の積分から計算した。
    [Theory]
    [InlineData(0.625, 1.5, 0.3, 0.7491321858436646, 0.10723130767572514, -1.9642857142857144, 1.8557087546617086, -2.6719151490519946)]
    [InlineData(3.0, 7.0, 0.23, -0.14070771609068766, 0.11760348983384644, 0.9034443817052508, 8.090426535983937, -4.231699079176592)]
    [InlineData(400.0, 160.0, 0.13, -1.7033911939463988, 1.1157360334834134, 2886.472148541114, 2894.720655877281, -9.521691313334836e-218)]
    [InlineData(150.0, 290.0, 0.91, 0.9840289962884736, -1.990463469983273, -3047.3748473748483, 7.611436453671556e-186, -3059.136517811365)]
    public void BetaDensityAndVariateTailDerivativesMatchReference(double alpha, double beta, double x,
        double expectedAlpha, double expectedBeta, double expectedX, double expectedLower, double expectedUpper)
    {
        var tape = new Tape();
        var observation = tape.Leaf(x);
        var a = tape.Leaf(alpha);
        var b = tape.Leaf(beta);
        var result = BetaDistribution.LogPdf(a, b, observation);
        tape.Backward(result);
        Close(expectedAlpha, tape.GetGradient(a));
        Close(expectedBeta, tape.GetGradient(b));
        Close(expectedX, tape.GetGradient(observation));

        foreach (bool upper in new[] { false, true })
        {
            tape.Reset();
            observation = tape.Leaf(x);
            result = upper ? BetaDistribution.LogSf(alpha, beta, observation) : BetaDistribution.LogCdf(alpha, beta, observation);
            tape.Backward(result);
            Close(upper ? expectedUpper : expectedLower, tape.GetGradient(observation));
        }
    }

    [Fact]
    public void BetaTailRejectsShapeGradients()
    {
        var tape = new Tape();
        var shape = tape.Leaf(2);
        Assert.Throws<NotSupportedException>(() => BetaDistribution.LogCdf(shape, 2, 0.5));
        Assert.Throws<NotSupportedException>(() => BetaDistribution.LogSf(2, shape, 0.5));
    }

    [Theory]
    [InlineData(1, 1e-20, -1)]
    [InlineData(1.0000000001, 1e-12, 99.0000082740371)]
    public void GammaDensityRetainsTheSmallVariateScore(double shape, double observation, double expected)
    {
        var tape = new Tape();
        var x = tape.Leaf(observation);
        var density = GammaDistribution.LogPdf(shape, 1, x);
        tape.Backward(density);
        Close(expected, tape.GetGradient(x));
    }

    [Fact]
    public void ChiSquaredWithTwoDegreesOfFreedomHasConstantVariateScore()
    {
        var tape = new Tape();
        var x = tape.Leaf(1e-18);
        var density = ChiSquaredDistribution.LogPdf(2, x);
        tape.Backward(density);
        Assert.Equal(-0.5, tape.GetGradient(x));
    }

    [Theory]
    [InlineData(0.36)]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(double.NaN)]
    [InlineData(-1)]
    public void BetaBatchPreservesScalarContributions(double observation)
    {
        var tape = new Tape();
        var alpha = tape.Leaf(1.4);
        var beta = tape.Leaf(2.1);
        double[] observations = [0.28, observation, 0.81];
        ScalarNode scalar = 0;
        foreach (double x in observations)
        {
            scalar += BetaDistribution.LogPdf(alpha, beta, x);
        }

        var batch = BetaDistribution.LogPdf(alpha, beta, new ReadOnlyVectorSpan<double>(observations));
        Assert.Equal(BitConverter.DoubleToInt64Bits(scalar.Value), BitConverter.DoubleToInt64Bits(batch.Value));
        tape.Backward(scalar);
        double expectedAlpha = tape.GetGradient(alpha), expectedBeta = tape.GetGradient(beta);

        tape.Reset();
        alpha = tape.Leaf(1.4);
        beta = tape.Leaf(2.1);
        batch = BetaDistribution.LogPdf(alpha, beta, new ReadOnlyVectorSpan<double>(observations));
        tape.Backward(batch);
        Verify(expectedAlpha, tape.GetGradient(alpha));
        Verify(expectedBeta, tape.GetGradient(beta));

        static void Verify(double expected, double actual)
        {
            if (double.IsFinite(expected))
            {
                Close(expected, actual);
            }
            else
            {
                Assert.Equal(expected, actual);
            }
        }
    }

    private static void Close(double expected, double actual)
    {
        if (expected == 0)
        {
            Assert.Equal(0, actual);
        }
        else
        {
            Assert.InRange(Math.Abs((expected - actual) / expected), 0, 2e-12);
        }
    }
}
