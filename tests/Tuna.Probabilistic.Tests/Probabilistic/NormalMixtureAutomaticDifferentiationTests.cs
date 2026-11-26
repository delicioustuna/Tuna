using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using Tuna.Probabilistic.Inference;

namespace Tuna.Tests.Probabilistic;

/// <summary>特殊関数と対数和の逆伝播を使うモデルを NUTS に接続する</summary>
[Trait("Probabilistic", "Mcmc")]
public class NormalMixtureAutomaticDifferentiationTests
{
    [Theory]
    [InlineData(0.0, 0.0)]
    [InlineData(-2.0, 3.0)]
    [InlineData(10.0, -20.0)]
    public void ShiftedNormalMixtureHasTheAnalyticGradient(double x, double y)
    {
        var target = new ReverseNormalMixture();
        double[] point = [x, y];
        double[] gradient = new double[2];
        double value = target.LogDensity(new ReadOnlyVectorSpan<double>(point), new VectorSpan<double>(gradient));
        Assert.Equal(BitConverter.DoubleToInt64Bits(target.LogDensity(new ReadOnlyVectorSpan<double>(point))),
            BitConverter.DoubleToInt64Bits(value));
        double expectedValue = XMath.Special.NormalLogPdf(x) - 0.5 + Math.Log(Math.Cosh(x)) + XMath.Special.NormalLogPdf(y);
        Assert.True(Math.Abs(value - expectedValue) <= 2e-14 * Math.Max(1, Math.Abs(expectedValue)));
        Assert.True(Math.Abs(gradient[0] - (-x + Math.Tanh(x))) <= 2e-14 * Math.Max(1, Math.Abs(x)));
        Assert.True(Math.Abs(gradient[1] + y) <= 2e-14 * Math.Max(1, Math.Abs(y)));
    }

    [Fact]
    public void NutsSamplesUsingSpecialFunctionsAndLogSumExp()
    {
        var options = new NoUTurnSamplerOptions { Seed = 193, MaxDegreeOfParallelism = 1 };
        HamiltonianSamplingResult result = new NoUTurnSampler(options).Sample(new ReverseNormalMixture(), [0.0, 0.0]);
        McmcAssert.Moments(result, [0, 0], [2, 1]);
        Assert.Equal(0, result.DivergenceCount);
        Assert.Equal(0, result.MaxTreeDepthHits);
    }

    private sealed class ReverseNormalMixture : IDifferentiableLogDensity
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
            // N(-1, 1) と N(1, 1) を等分に混ぜ、入力によって異なる重みと勾配の蓄積を通す。
            ScalarNode first = XMath.Special.NormalLogPdf(x + 1) - Math.Log(2);
            ScalarNode second = XMath.Special.NormalLogPdf(x - 1) - Math.Log(2);
            ReadOnlySpan<ScalarNode> components = [first, second];
            return XMath.LogSumExp(components) + XMath.Special.NormalLogPdf(y);
        }
    }
}
