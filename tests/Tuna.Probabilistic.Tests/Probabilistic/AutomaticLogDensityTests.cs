using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using Tuna.Probabilistic.Inference;
using Tuna.RandomSource;
using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Probabilistic;

[Trait("Probabilistic", "Mcmc")]
public class AutomaticLogDensityTests
{
    [Fact]
    public void ScalarAndVectorModelsAgreeDuringParallelEvaluation()
    {
        var scalar = LogDensities.FromAutomaticDifferentiation(p => -0.5 * (p[0] * p[0] + p[1] * p[1]));
        var vector = LogDensities.FromAutomaticDifferentiation((_, p) => -0.5 * XMath.SumOfSquares(p));
        var elements = LogDensities.FromAutomaticDifferentiation((_, p) => -0.5 * (p[0] * p[0] + p[1] * p[1]));

        Parallel.For(0, 64, i =>
        {
            double[] point = [i * 0.5, i * -0.25];
            Span<double> gradient = stackalloc double[2];
            foreach (var density in new[] { scalar, vector, elements })
            {
                double expected = -0.5 * (point[0] * point[0] + point[1] * point[1]);
                double value = density.LogDensity(new ReadOnlyVectorSpan<double>(point), new VectorSpan<double>(gradient));
                Assert.Equal(expected, value);
                Assert.Equal(expected, density.LogDensity(new ReadOnlyVectorSpan<double>(point)));
                Assert.Equal(-point[0], gradient[0]);
                Assert.Equal(-point[1], gradient[1]);
            }
        });
    }

    [Fact]
    public void FailedEvaluationReturnsTapeInAUsableState()
    {
        var density = LogDensities.FromAutomaticDifferentiation(p =>
        {
            var square = p[0] * p[0];
            if (p[0].Value < 0)
            {
                throw new InvalidOperationException();
            }

            return square;
        });
        Assert.Throws<InvalidOperationException>(() => density.LogDensity(new ReadOnlyVectorSpan<double>([-1])));
        Span<double> gradient = stackalloc double[1];
        Assert.Equal(4.0, density.LogDensity(new ReadOnlyVectorSpan<double>([2]), new VectorSpan<double>(gradient)));
        Assert.Equal(4.0, gradient[0]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void HamiltonianSamplersRejectProposalsWithInvalidDistributionParameters(bool useNuts)
    {
        int invalidParameters = 0;
        var density = LogDensities.FromAutomaticDifferentiation(p =>
        {
            ScalarNode scale = XMath.Exp(p[0]);
            if (!(scale.Value > 0 && double.IsFinite(scale.Value)))
            {
                invalidParameters++;
            }

            return NormalDistribution.LogPdf(0, 1, p[0]) + NormalDistribution.LogPdf(0, scale, 0.3);
        });
        var explicitlyGuarded = LogDensities.FromAutomaticDifferentiation(p =>
        {
            ScalarNode scale = XMath.Exp(p[0]);
            if (!(scale.Value > 0 && double.IsFinite(scale.Value)))
            {
                return double.NegativeInfinity;
            }

            return NormalDistribution.LogPdf(0, 1, p[0]) + NormalDistribution.LogPdf(0, scale, 0.3);
        });

        var expected = Sample(explicitlyGuarded);
        var actual = Sample(density);
        Assert.True(invalidParameters > 0);
        Assert.True(actual.DivergenceCount > 0);
        Assert.Equal(expected.DivergenceCount, actual.DivergenceCount);
        Assert.True(expected.Draws.AsSpan().SequenceEqual(actual.Draws.AsSpan()));

        HamiltonianSamplingResult Sample(IDifferentiableLogDensity target) => useNuts
            ? new NoUTurnSampler(new() { Chains = 1, Warmup = 0, Draws = 4, Seed = 29, InitialStepSize = 1e6 }).Sample(target, [0])
            : new HamiltonianMonteCarlo(new() { Chains = 1, Warmup = 0, Draws = 4, Seed = 29, InitialStepSize = 1e6, Steps = 1 }).Sample(target, [0]);
    }

    [Fact]
    public void HamiltonianSamplerPropagatesModelErrorsAndRequiresAValidInitialPoint()
    {
        var sampler = new NoUTurnSampler(new() { Chains = 1, Warmup = 0, Draws = 1, Seed = 29, InitialStepSize = 1 });
        var badIndex = LogDensities.FromAutomaticDifferentiation((_, p) => p[0].Value == 0 ? -p[0] * p[0] : p[1]);
        Assert.Throws<ArgumentOutOfRangeException>(() => sampler.Sample(badIndex, [0]));

        var invalidScale = LogDensities.FromAutomaticDifferentiation(p => NormalDistribution.LogPdf(0, XMath.Exp(p[0]), 0.3));
        var error = Assert.Throws<ArgumentException>(() => sampler.Sample(invalidScale, [1000]));
        Assert.Equal("initialPoints", error.ParamName);
        var cause = Assert.IsType<DistributionParameterOutOfRangeException>(error.InnerException);
        Assert.Equal("standardDeviation", cause.ParamName);
        Assert.Equal(double.PositiveInfinity, cause.ActualValue);
    }

    [Fact]
    public void EightSchoolsPosteriorRanksAreCalibrated()
    {
        double[] standardErrors = [15, 10, 16, 11, 9, 11, 10, 18];
        const int replicates = 64, warmup = 192, draws = 252, thin = 12, bins = 7;
        var random = new Pcg64DxsmRandom(20261123);
        var standardNormal = new NormalDistribution(0, 1);
        var counts = new int[3, bins];
        var observations = new double[standardErrors.Length];
        var density = LogDensities.FromAutomaticDifferentiation(p =>
        {
            ScalarNode mean = p[0], tau = XMath.Exp(p[1]);
            ScalarNode value = NormalDistribution.LogPdf(0, 5, mean) + NormalDistribution.LogPdf(0, 5, tau) + p[1];
            if (!double.IsFinite(value.Value))
            {
                return value;
            }

            for (int j = 0; j < observations.Length; j++)
            {
                value += NormalDistribution.LogPdf(0, 1, p[j + 2])
                    + NormalDistribution.LogPdf(mean + tau * p[j + 2], standardErrors[j], observations[j]);
            }

            return value;
        });
        int divergences = 0;
        for (int i = 0; i < replicates; i++)
        {
            double mean = 5 * standardNormal.Sample(random);
            double tau = 5 * Math.Abs(standardNormal.Sample(random));
            double school = 0;
            for (int j = 0; j < observations.Length; j++)
            {
                double theta = mean + tau * standardNormal.Sample(random);
                observations[j] = theta + standardErrors[j] * standardNormal.Sample(random);
                if (j == 0)
                {
                    school = theta;
                }
            }

            var options = new NoUTurnSamplerOptions
            {
                Chains = 1,
                Warmup = warmup,
                Draws = draws,
                Seed = 8157UL + (ulong)i,
                TargetAcceptance = 0.95,
            };
            HamiltonianSamplingResult result = new NoUTurnSampler(options).Sample(density, new double[standardErrors.Length + 2]);
            divergences += result.DivergenceCount;
            var sampledMean = result.Parameter(0);
            var sampledLogTau = result.Parameter(1);
            var sampledZ = result.Parameter(2);
            int meanRank = 0, tauRank = 0, schoolRank = 0;
            // 21 標本から得られる順位を7等分するため、各区間の端は整数の順位では一致しない。
            // 端での無作為化により、独立な事後標本なら各区間の確率は 1/7 になる。
            for (int j = thin - 1; j < draws; j += thin)
            {
                double sampleMean = sampledMean[j, 0];
                double sampleTau = Math.Exp(sampledLogTau[j, 0]);
                meanRank += sampleMean < mean ? 1 : 0;
                tauRank += sampleTau < tau ? 1 : 0;
                schoolRank += sampleMean + sampleTau * sampledZ[j, 0] < school ? 1 : 0;
            }

            counts[0, RankBin(meanRank)]++;
            counts[1, RankBin(tauRank)]++;
            counts[2, RankBin(schoolRank)]++;
        }

        Assert.InRange(divergences, 0, 4);
        for (int parameter = 0; parameter < counts.GetLength(0); parameter++)
        {
            double expected = (double)replicates / bins, chiSquare = 0;
            for (int bin = 0; bin < bins; bin++)
            {
                double difference = counts[parameter, bin] - expected;
                chiSquare += difference * difference / expected;
            }

            Assert.True(chiSquare <= 22.46, $"Parameter {parameter}: chi-square {chiSquare:R}");
        }

        int RankBin(int rank) => (int)((rank + random.NextDouble()) * bins / (draws / thin + 1));
    }
}
