using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using Tuna.Probabilistic.Inference;
using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Probabilistic;

/// <summary>モデルを逆モードで微分し、その勾配を NUTS に渡す</summary>
[Trait("Probabilistic", "Mcmc")]
public class AutomaticDifferentiationTests
{
    [Theory]
    [InlineData(0.0, 0.0)]
    [InlineData(1.0, -2.0)]
    [InlineData(3.5, -31.0)]
    [InlineData(-5.0, 40.0)]
    public void CorrelatedNormalGradientAgreesWithAnalyticGradient(double x, double y)
    {
        var target = new ReverseCorrelatedNormal();
        var reference = new TargetDensities.CorrelatedNormal(1, -2, 1, 10, 0.9);
        double[] point = [x, y];
        var actual = new double[2];
        var expected = new double[2];
        double value = target.LogDensity(new ReadOnlyVectorSpan<double>(point), new VectorSpan<double>(actual));
        double expectedValue = reference.LogDensity(new ReadOnlyVectorSpan<double>(point), new VectorSpan<double>(expected));
        Assert.Equal(BitConverter.DoubleToInt64Bits(expectedValue), BitConverter.DoubleToInt64Bits(value));
        Assert.Equal(BitConverter.DoubleToInt64Bits(value), BitConverter.DoubleToInt64Bits(target.LogDensity(new ReadOnlyVectorSpan<double>(point))));
        for (int i = 0; i < 2; i++)
        {
            Assert.True(Math.Abs(actual[i] - expected[i]) <= 2e-14 * Math.Max(1, Math.Abs(expected[i])), $"gradient[{i}]: {actual[i]:R} != {expected[i]:R}");
        }
    }

    [Fact]
    public void NutsRecoversCorrelatedNormalFromReverseModeGradient()
    {
        // このモデルは1本のテープを再利用するので、チェーンは逐次に動かす。
        var options = new NoUTurnSamplerOptions { Seed = 91, Metric = HamiltonianMetric.Dense, MaxDegreeOfParallelism = 1 };
        var target = new ReverseCorrelatedNormal();
        HamiltonianSamplingResult result = new NoUTurnSampler(options).Sample(target, [0.0, 0.0]);
        var reference = new TargetDensities.CorrelatedNormal(1, -2, 1, 10, 0.9);
        McmcAssert.Moments(result, [1, -2], reference.Covariance);
        Assert.Equal(0, result.DivergenceCount);
        Assert.Equal(0, result.MaxTreeDepthHits);
    }

    [Fact]
    public void NutsSamplesRobustRegressionFromReverseModeGradient()
    {
        var target = new RobustRegression(
            [-3, -2, -1, 0, 1, 2, 3],
            [-0.42, -0.04, 0.52, 1.1, 1.39, 2.02, 6.8]);
        double[] initial = [0, 0, 0];
        Assert.InRange(LogDensities.GradientError(target, initial), 0, 1e-7);

        var options = new NoUTurnSamplerOptions
        {
            Chains = 1,
            Warmup = 128,
            Draws = 64,
            Seed = 20261123,
            TargetAcceptance = 0.92,
        };
        HamiltonianSamplingResult result = new NoUTurnSampler(options).Sample(target, initial);
        Assert.Equal(64, result.DrawCount);
        Assert.Equal(0, result.DivergenceCount);

        for (int parameter = 0; parameter < initial.Length; parameter++)
        {
            var draws = result.Parameter(parameter);
            for (int i = 0; i < draws.Rows; i++)
            {
                Assert.True(double.IsFinite(draws[i, 0]));
            }
        }
    }

    private sealed class RobustRegression(double[] features, double[] responses) : IDifferentiableLogDensity
    {
        private readonly Tape tape = new();
        private readonly Vector<double> featureValues = new(features);
        private readonly Vector<double> responseValues = new(responses);

        public double LogDensity(ReadOnlyVectorSpan<double> theta, VectorSpan<double> gradient)
        {
            tape.Reset();
            ReadOnlySpan<ScalarNode> inputs = tape.Leaves(theta);
            ScalarNode intercept = inputs[0], slope = inputs[1], scale = XMath.Exp(inputs[2]);
            ScalarNode density = NormalDistribution.LogPdf(0, 10, intercept)
                + NormalDistribution.LogPdf(0, 10, slope)
                + CauchyDistribution.LogPdf(0, 5, scale) + inputs[2];
            var locations = intercept + slope * tape.Constant(featureValues);
            density += StudentTDistribution.LogPdf(4, locations, scale, responseValues);
            return tape.Gradient(density, gradient);
        }
    }

    /// <summary>同じモデルを、勾配つきの葉と記録しない定数の両方で評価する</summary>
    private sealed class ReverseCorrelatedNormal : IDifferentiableLogDensity
    {
        private readonly Tape tape = new();

        public double LogDensity(ReadOnlyVectorSpan<double> theta) => Model(theta[0], theta[1]).Value;

        public double LogDensity(ReadOnlyVectorSpan<double> theta, VectorSpan<double> gradient)
        {
            tape.Reset();
            ReadOnlySpan<ScalarNode> inputs = tape.Leaves(theta);
            return tape.Gradient(Model(inputs[0], inputs[1]), gradient);
        }

        private static ScalarNode Model(ScalarNode x, ScalarNode y)
        {
            // 参照モデルと同じ (x − μ)/σ の順に計算し、値のビット列をそろえるため、尺度 1 の除算も残す。
            ScalarNode a = (x - 1) / 1;
            ScalarNode b = (y + 2) / 10;
            return -0.5 * (a * a - 2 * 0.9 * a * b + b * b) / (1 - 0.9 * 0.9);
        }
    }
}
