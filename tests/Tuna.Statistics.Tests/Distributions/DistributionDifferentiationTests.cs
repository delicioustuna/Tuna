using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using Tuna.Statistics.Distributions;
using Xunit.Abstractions;

namespace Tuna.Tests.Statistics;

public class DistributionDifferentiationTests(ITestOutputHelper output)
{
    private const double RelativeTolerance = 16 * 2.220446049250313e-16;

    [Theory]
    [InlineData(0, "LogPdf")]
    [InlineData(1, "LogCdf")]
    [InlineData(2, "LogSf")]
    public void NormalLocalDerivativesMatchMpmath(int index, string function)
    {
        var tape = new Tape();
        double maximumRelativeError = 0;
        foreach (var row in DistributionGradients.Normal[index])
        {
            tape.Reset();
            var mean = tape.Leaf(row.Mean);
            var scale = tape.Leaf(row.Scale);
            var x = tape.Leaf(row.X);
            var result = NormalValue(function, mean, scale, x);
            double expected = NormalValue(function, new NormalDistribution(row.Mean, row.Scale), row.X);
            AssertSameBits(expected, result.Value);
            var constant = NormalValue(function, row.Mean, row.Scale, row.X);
            AssertSameBits(expected, constant.Value);
            Assert.False(constant.RequiresGradient);
            tape.Backward(result);

            string context = $"{function}({row.Mean:R}, {row.Scale:R}, {row.X:R})";
            maximumRelativeError = Math.Max(maximumRelativeError, CheckDerivative(row.DMean, tape.GetGradient(mean), $"μ, {context}"));
            maximumRelativeError = Math.Max(maximumRelativeError, CheckDerivative(row.DScale, tape.GetGradient(scale), $"σ, {context}"));
            maximumRelativeError = Math.Max(maximumRelativeError, CheckDerivative(row.DX, tape.GetGradient(x), $"x, {context}"));
        }

        output.WriteLine($"{function}: {3 * DistributionGradients.Normal[index].Length} partial derivatives, maximum relative error (normal references): {maximumRelativeError:E3}");
    }

    [Fact]
    public void NormalBatchMatchesMpmathForStridedObservations()
    {
        var tape = new Tape();
        double maximumRelativeError = 0;
        foreach (var row in DistributionGradients.NormalSums)
        {
            tape.Reset();
            var mean = tape.Leaf(row.Mean);
            var scale = tape.Leaf(row.Scale);
            var storage = new double[2 * row.Observations.Length];
            var distribution = new NormalDistribution(row.Mean, row.Scale);
            double expected = 0;
            for (int i = 0; i < row.Observations.Length; i++)
            {
                storage[2 * i] = row.Observations[i];
                storage[2 * i + 1] = double.NaN;
                expected += distribution.LogPdf(row.Observations[i]);
            }

            ReadOnlyVectorSpan<double> observations = storage.Length == 0
                ? default
                : new(storage, [row.Observations.Length], [2]);
            var result = NormalDistribution.LogPdf(mean, scale, observations);
            AssertSameBits(expected, result.Value);
            var constant = NormalDistribution.LogPdf(row.Mean, row.Scale, observations);
            AssertSameBits(expected, constant.Value);
            Assert.False(constant.RequiresGradient);
            // 順伝播後に観測を書き換えても、記録には集計済みの局所微分が入っている。
            Array.Fill(storage, double.NaN);
            tape.Backward(result);

            string context = $"Normal sum ({row.Mean:R}, {row.Scale:R}, [{string.Join(", ", row.Observations)}])";
            maximumRelativeError = Math.Max(maximumRelativeError, CheckDerivative(row.DMean, tape.GetGradient(mean), $"μ, {context}"));
            maximumRelativeError = Math.Max(maximumRelativeError, CheckDerivative(row.DScale, tape.GetGradient(scale), $"σ, {context}"));
        }

        output.WriteLine($"Normal sum: {2 * DistributionGradients.NormalSums.Length} partial derivatives, maximum relative error (normal references): {maximumRelativeError:E3}");
    }

    [Fact]
    public void BernoulliLocalDerivativesMatchMpmathIncludingBoundaries()
    {
        var tape = new Tape();
        double maximumRelativeError = 0;
        foreach (var row in DistributionGradients.Bernoulli)
        {
            var distribution = new BernoulliDistribution(row.Probability);
            double expected = 0;
            foreach (long k in row.Observations)
            {
                expected += distribution.LogPmf(k);
            }

            tape.Reset();
            var probability = tape.Leaf(row.Probability);
            var observations = new ReadOnlyVectorSpan<long>(row.Observations);
            var batch = BernoulliDistribution.LogPmf(probability, observations);
            AssertSameBits(expected, batch.Value);
            AssertSameBits(expected, BernoulliDistribution.LogPmf(row.Probability, observations).Value);
            tape.Backward(batch);
            string context = $"Bernoulli sum ({row.Probability:R}, [{string.Join(", ", row.Observations)}])";
            maximumRelativeError = Math.Max(maximumRelativeError, CheckDerivative(row.Derivative, tape.GetGradient(probability), context));

            if (row.Observations is [long k0])
            {
                Check(BernoulliDistribution.LogPmf, row.Probability, distribution.LogPmf(k0), k0, row.Derivative);
                Check(BernoulliDistribution.LogCdf, row.Probability, distribution.LogCdf(k0), k0, k0 == 0 ? row.Derivative : 0);
                Check(BernoulliDistribution.LogSf, row.Probability, distribution.LogSf(k0 - 1), k0 - 1, k0 == 1 ? row.Derivative : 0);
            }
        }

        output.WriteLine($"Bernoulli: maximum relative error (normal references): {maximumRelativeError:E3}");

        void Check(Func<ScalarNode, long, ScalarNode> operation, double p, double expected, long k, double derivative)
        {
            tape.Reset();
            var probability = tape.Leaf(p);
            var result = operation(probability, k);
            AssertSameBits(expected, result.Value);
            tape.Backward(result);
            maximumRelativeError = Math.Max(maximumRelativeError, CheckDerivative(derivative, tape.GetGradient(probability), $"{operation.Method.Name}({p:R}, {k})"));
        }
    }

    [Theory]
    [InlineData("LogPdf", 0.8)]
    [InlineData("LogCdf", -36)]
    [InlineData("LogCdf", 34)]
    [InlineData("LogSf", -34)]
    [InlineData("LogSf", 36)]
    public void NormalGradientsMatchCentralDifferences(string function, double observation)
    {
        double[] arguments = [0.3, 1.2, observation];
        var tape = new Tape();
        var result = NormalValue(function, tape.Leaf(arguments[0]), tape.Leaf(arguments[1]), tape.Leaf(arguments[2]));
        Span<double> gradients = stackalloc double[3];
        tape.Gradient(result, new VectorSpan<double>(gradients));
        for (int i = 0; i < arguments.Length; i++)
        {
            int index = i;
            double expected = CentralDifference(x =>
            {
                double mean = index == 0 ? x : arguments[0];
                double scale = index == 1 ? x : arguments[1];
                double q = index == 2 ? x : arguments[2];
                return NormalValue(function, new NormalDistribution(mean, scale), q);
            }, arguments[i]);
            Assert.InRange(Math.Abs((gradients[i] - expected) / expected), 0, 2e-6);
        }
    }

    [Fact]
    public void BernoulliBatchGradientMatchesCentralDifferences()
    {
        long[] observations = [1, 0, 0, 1, 0];
        const double probability = 0.2;
        var tape = new Tape();
        var p = tape.Leaf(probability);
        var result = BernoulliDistribution.LogPmf(p, new ReadOnlyVectorSpan<long>(observations));
        tape.Backward(result);
        double expected = CentralDifference(
            value => observations.Sum(new BernoulliDistribution(value).LogPmf), probability);
        Assert.InRange(Math.Abs(tape.GetGradient(p) - expected), 0, 1e-9);
    }

    [Fact]
    public void BatchTapeAllocationDoesNotGrowWithObservationCount()
    {
        var tape = new Tape();
        var normal = new Vector<double>(8192);
        var bernoulli = new Vector<long>(8192);
        Evaluate(1);

        long before = GC.GetAllocatedBytesForCurrentThread();
        Evaluate(8192);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(0, allocated);

        void Evaluate(int count)
        {
            tape.Reset();
            var mean = tape.Leaf(0);
            var scale = tape.Leaf(1);
            var probability = tape.Leaf(0.5);
            var value = NormalDistribution.LogPdf(mean, scale, normal.AsSpan()[..count])
                + BernoulliDistribution.LogPmf(probability, bernoulli.AsSpan()[..count]);
            tape.Backward(value);
        }
    }

    [Fact]
    public void DistributionOverloadsUseExistingArgumentAndTapeValidation()
    {
        var tape = new Tape();
        var mean = tape.Leaf(0);
        var secondTape = new Tape();
        Assert.Throws<DistributionParameterOutOfRangeException>(() => NormalDistribution.LogPdf(mean, 0, 1));
        Assert.Throws<DistributionParameterOutOfRangeException>(() => BernoulliDistribution.LogPmf(tape.Leaf(double.NaN), 0));
        Assert.Throws<InvalidOperationException>(() => NormalDistribution.LogCdf(mean, secondTape.Leaf(1), 0));
        tape.Reset();
        Assert.Throws<InvalidOperationException>(() => NormalDistribution.LogSf(mean, 1, 0));
    }

    [Theory]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(double.NaN)]
    public void NormalNonfiniteObservationsFollowTheTailLimits(double observation)
    {
        var distribution = new NormalDistribution(0, 1);
        foreach (string function in new[] { "LogPdf", "LogCdf", "LogSf" })
        {
            var tape = new Tape();
            var mean = tape.Leaf(0);
            var scale = tape.Leaf(1);
            var x = tape.Leaf(observation);
            var result = NormalValue(function, mean, scale, x);
            AssertSameBits(NormalValue(function, distribution, observation), result.Value);
            tape.Backward(result);
            if (double.IsNaN(observation))
            {
                Assert.True(double.IsNaN(tape.GetGradient(mean)));
                Assert.True(double.IsNaN(tape.GetGradient(scale)));
                Assert.True(double.IsNaN(tape.GetGradient(x)));
                continue;
            }

            double derivative = double.IsNegativeInfinity(result.Value) ? observation : 0;
            Assert.Equal(derivative, tape.GetGradient(mean));
            Assert.Equal(derivative == 0 ? 0 : double.PositiveInfinity, tape.GetGradient(scale));
            Assert.Equal(-derivative, tape.GetGradient(x));
        }
    }

    private static ScalarNode NormalValue(string function, ScalarNode mean, ScalarNode scale, ScalarNode x) => function switch
    {
        "LogPdf" => NormalDistribution.LogPdf(mean, scale, x),
        "LogCdf" => NormalDistribution.LogCdf(mean, scale, x),
        "LogSf" => NormalDistribution.LogSf(mean, scale, x),
        _ => throw new ArgumentOutOfRangeException(nameof(function)),
    };

    private static double NormalValue(string function, NormalDistribution distribution, double x) => function switch
    {
        "LogPdf" => distribution.LogPdf(x),
        "LogCdf" => distribution.LogCdf(x),
        "LogSf" => distribution.LogSf(x),
        _ => throw new ArgumentOutOfRangeException(nameof(function)),
    };

    private static double CentralDifference(Func<double, double> function, double x)
    {
        double step = 2e-5 * Math.Max(1, Math.Abs(x));
        double coarse = (function(x + step) - function(x - step)) / (2 * step);
        double fine = (function(x + 0.5 * step) - function(x - 0.5 * step)) / step;
        return (4 * fine - coarse) / 3;
    }

    internal static double CheckDerivative(double expected, double actual, string context)
    {
        if (double.IsInfinity(expected))
        {
            Assert.True(expected == actual, $"{context}: expected {expected:R}, got {actual:R}");
            return 0;
        }

        double error = Math.Abs(expected - actual);
        double magnitude = Math.Abs(expected);
        double ulp = magnitude - Math.BitDecrement(magnitude);
        Assert.True(double.IsFinite(actual) && error <= Math.Max(RelativeTolerance * magnitude, 4 * ulp),
            $"{context}: expected {expected:R}, got {actual:R}");
        return double.IsNormal(expected) ? error / magnitude : 0;
    }

    internal static void AssertSameBits(double expected, double actual) =>
        Assert.Equal(BitConverter.DoubleToInt64Bits(expected), BitConverter.DoubleToInt64Bits(actual));
}
