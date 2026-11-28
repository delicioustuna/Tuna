using Numerics = System.Numerics;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Statistics;

public class PointwiseDistributionTests
{
    private static readonly double[] RealLimits = [double.NegativeInfinity, -double.MaxValue, double.NaN, double.MaxValue, double.PositiveInfinity];
    private static readonly double[] PositiveLimits = [-1, 0, double.NaN, double.Epsilon, double.PositiveInfinity];
    private static readonly double[] UnitLimits = [-1, 0, double.NaN, 1, 2];
    private static readonly long[] CountLimits = [long.MinValue, 0, 1, 9, long.MaxValue];

    private static readonly Dictionary<string, Distribution<double>> Continuous = new()
    {
        [nameof(NormalDistribution)] = new([-0.4, 1.2], [0.2, 0.5, 0.8], RealLimits,
            static (p, x) => NormalDistribution.LogPdf(p[0], p[1], x),
            static (p, x) => NormalDistribution.LogPdf(p[0], p[1], x),
            static (p, x) => NormalDistribution.LogPdfPointwise(p[0], p[1], x)),
        [nameof(StudentTDistribution)] = new([4.3, -0.4, 1.2], [0.2, 0.5, 0.8], RealLimits,
            static (p, x) => StudentTDistribution.LogPdf(p[0], p[1], p[2], x),
            static (p, x) => StudentTDistribution.LogPdf(p[0], p[1], p[2], x),
            static (p, x) => StudentTDistribution.LogPdfPointwise(p[0], p[1], p[2], x)),
        [nameof(CauchyDistribution)] = new([-0.4, 1.2], [0.2, 0.5, 0.8], RealLimits,
            static (p, x) => CauchyDistribution.LogPdf(p[0], p[1], x),
            static (p, x) => CauchyDistribution.LogPdf(p[0], p[1], x),
            static (p, x) => CauchyDistribution.LogPdfPointwise(p[0], p[1], x)),
        [nameof(GammaDistribution)] = new([1.4, 1.2], [0.2, 0.5, 0.8], PositiveLimits,
            static (p, x) => GammaDistribution.LogPdf(p[0], p[1], x),
            static (p, x) => GammaDistribution.LogPdf(p[0], p[1], x),
            static (p, x) => GammaDistribution.LogPdfPointwise(p[0], p[1], x)),
        [nameof(InverseGammaDistribution)] = new([1.4, 1.2], [0.2, 0.5, 0.8], PositiveLimits,
            static (p, x) => InverseGammaDistribution.LogPdf(p[0], p[1], x),
            static (p, x) => InverseGammaDistribution.LogPdf(p[0], p[1], x),
            static (p, x) => InverseGammaDistribution.LogPdfPointwise(p[0], p[1], x)),
        [nameof(ChiSquaredDistribution)] = new([3.4], [0.2, 0.5, 0.8], PositiveLimits,
            static (p, x) => ChiSquaredDistribution.LogPdf(p[0], x),
            static (p, x) => ChiSquaredDistribution.LogPdf(p[0], x),
            static (p, x) => ChiSquaredDistribution.LogPdfPointwise(p[0], x)),
        [nameof(BetaDistribution)] = new([1.4, 2.2], [0.2, 0.5, 0.8], UnitLimits,
            static (p, x) => BetaDistribution.LogPdf(p[0], p[1], x),
            static (p, x) => BetaDistribution.LogPdf(p[0], p[1], x),
            static (p, x) => BetaDistribution.LogPdfPointwise(p[0], p[1], x)),
        [nameof(ExponentialDistribution)] = new([1.4], [0.2, 0.5, 0.8], PositiveLimits,
            static (p, x) => ExponentialDistribution.LogPdf(p[0], x),
            static (p, x) => ExponentialDistribution.LogPdf(p[0], x),
            static (p, x) => ExponentialDistribution.LogPdfPointwise(p[0], x)),
        [nameof(LogisticDistribution)] = new([-0.4, 1.2], [0.2, 0.5, 0.8], RealLimits,
            static (p, x) => LogisticDistribution.LogPdf(p[0], p[1], x),
            static (p, x) => LogisticDistribution.LogPdf(p[0], p[1], x),
            static (p, x) => LogisticDistribution.LogPdfPointwise(p[0], p[1], x)),
        [nameof(LaplaceDistribution)] = new([-0.4, 1.2], [0.2, 0.5, 0.8], RealLimits,
            static (p, x) => LaplaceDistribution.LogPdf(p[0], p[1], x),
            static (p, x) => LaplaceDistribution.LogPdf(p[0], p[1], x),
            static (p, x) => LaplaceDistribution.LogPdfPointwise(p[0], p[1], x)),
        [nameof(GumbelDistribution)] = new([-0.4, 1.2], [0.2, 0.5, 0.8], RealLimits,
            static (p, x) => GumbelDistribution.LogPdf(p[0], p[1], x),
            static (p, x) => GumbelDistribution.LogPdf(p[0], p[1], x),
            static (p, x) => GumbelDistribution.LogPdfPointwise(p[0], p[1], x)),
        [nameof(WeibullDistribution)] = new([1.4, 1.2], [0.2, 0.5, 0.8], PositiveLimits,
            static (p, x) => WeibullDistribution.LogPdf(p[0], p[1], x),
            static (p, x) => WeibullDistribution.LogPdf(p[0], p[1], x),
            static (p, x) => WeibullDistribution.LogPdfPointwise(p[0], p[1], x)),
        [nameof(UniformDistribution)] = new([0, 1], [0.2, 0.5, 0.8], [-1, 0, double.NaN, 1, 2],
            static (p, x) => UniformDistribution.LogPdf(p[0], p[1], x),
            static (p, x) => UniformDistribution.LogPdf(p[0], p[1], x),
            static (p, x) => UniformDistribution.LogPdfPointwise(p[0], p[1], x)),
        [nameof(LogNormalDistribution)] = new([-0.4, 1.2], [0.2, 0.5, 0.8], PositiveLimits,
            static (p, x) => LogNormalDistribution.LogPdf(p[0], p[1], x),
            static (p, x) => LogNormalDistribution.LogPdf(p[0], p[1], x),
            static (p, x) => LogNormalDistribution.LogPdfPointwise(p[0], p[1], x)),
        [nameof(FDistribution)] = new([3.4, 5.2], [0.2, 0.5, 0.8], PositiveLimits,
            static (p, x) => FDistribution.LogPdf(p[0], p[1], x),
            static (p, x) => FDistribution.LogPdf(p[0], p[1], x),
            static (p, x) => FDistribution.LogPdfPointwise(p[0], p[1], x)),
    };

    private static readonly Dictionary<string, Distribution<long>> Discrete = new()
    {
        [nameof(BernoulliDistribution)] = new([0.4], [0, 1, 0], CountLimits,
            static (p, x) => BernoulliDistribution.LogPmf(p[0], x),
            static (p, x) => BernoulliDistribution.LogPmf(p[0], x),
            static (p, x) => BernoulliDistribution.LogPmfPointwise(p[0], x)),
        [nameof(PoissonDistribution)] = new([2.4], [0, 2, 3], CountLimits,
            static (p, x) => PoissonDistribution.LogPmf(p[0], x),
            static (p, x) => PoissonDistribution.LogPmf(p[0], x),
            static (p, x) => PoissonDistribution.LogPmfPointwise(p[0], x)),
        [nameof(GeometricDistribution)] = new([0.4], [0, 2, 3], CountLimits,
            static (p, x) => GeometricDistribution.LogPmf(p[0], x),
            static (p, x) => GeometricDistribution.LogPmf(p[0], x),
            static (p, x) => GeometricDistribution.LogPmfPointwise(p[0], x)),
        [nameof(BinomialDistribution)] = new([0.4], [0, 2, 3], CountLimits,
            static (p, x) => BinomialDistribution.LogPmf(9, p[0], x),
            static (p, x) => BinomialDistribution.LogPmf(9, p[0], x),
            static (p, x) => BinomialDistribution.LogPmfPointwise(9, p[0], x)),
        [nameof(NegativeBinomialDistribution)] = new([2.4, 0.4], [0, 2, 3], CountLimits,
            static (p, x) => NegativeBinomialDistribution.LogPmf(p[0], p[1], x),
            static (p, x) => NegativeBinomialDistribution.LogPmf(p[0], p[1], x),
            static (p, x) => NegativeBinomialDistribution.LogPmfPointwise(p[0], p[1], x)),
    };

    public static TheoryData<string> ContinuousNames => [.. Continuous.Keys];
    public static TheoryData<string> DiscreteNames => [.. Discrete.Keys];

    [Theory]
    [MemberData(nameof(ContinuousNames))]
    public void ContinuousBatchesMatchSingleObservations(string name) => CheckDistribution(Continuous[name]);

    [Theory]
    [MemberData(nameof(DiscreteNames))]
    public void DiscreteBatchesMatchSingleObservations(string name) => CheckDistribution(Discrete[name]);

    [Fact]
    public void MixedInputsPropagateThroughEarlierScalarAndVectorOperations()
    {
        var tape = new Tape();
        var x = tape.Leaf(new Vector<double>([9, 1, 2, 8]));
        ScalarNode scale = XMath.Sum(x[1..3]);
        var observations = new ReadOnlyVectorSpan<double>([0, -10, 1, -10], [2], [2]);
        var means = x[1..3] * 2.0;
        ScalarNode sum = NormalDistribution.LogPdf(means, scale, observations);
        var pointwise = NormalDistribution.LogPdfPointwise(means, scale, observations);
        var weights = tape.Constant(new Vector<double>([2, -3]));
        tape.Backward(2 * sum + XMath.Dot(pointwise, weights));

        double dScale = 4 * ((4.0 / 9 - 1) / 3) - ((1.0 - 1) / 3);
        var gradient = tape.GetGradient(x);
        Close(0, gradient[0]);
        Close(dScale + 4 * 2 * (-2.0 / 9), gradient[1]);
        Close(dScale + 2 * (3.0 / 9), gradient[2]);
        Close(0, gradient[3]);
    }

    [Fact]
    public void ConstantsAndTapeOwnershipFollowArrayNodeRules()
    {
        var tape = new Tape();
        var observations = new Vector<double>([0.5, 2]);
        Assert.Throws<ArgumentException>(() => NormalDistribution.LogPdfPointwise(0, 1, observations));
        var constants = NormalDistribution.LogPdfPointwise(0, 1, observations, tape);
        Assert.False(constants.RequiresGradient);
        Assert.Equal(new NormalDistribution(0, 1).LogPdf(0.5), constants.Value[0]);

        var mean = tape.Leaf(new Vector<double>([0.1, 0.2]));
        ScalarNode foreignScale = new Tape().Leaf(1);
        Assert.Throws<InvalidOperationException>(() => NormalDistribution.LogPdf(mean, foreignScale, observations));
        Assert.Throws<InvalidOperationException>(() => NormalDistribution.LogPdfPointwise(mean, 1, observations, new Tape()));
        var shortScale = tape.Leaf(new Vector<double>([1]));
        Assert.Throws<ArgumentException>("standardDeviation", () => NormalDistribution.LogPdfPointwise(mean, shortScale, observations));
        tape.Backward(NormalDistribution.LogPdf(mean, 1, observations));
        Assert.Throws<InvalidOperationException>(() => NormalDistribution.LogPdf(mean, 1, observations));
        tape.Reset();
        Assert.Throws<InvalidOperationException>(() => NormalDistribution.LogPdfPointwise(mean, 1, observations));
    }

    [Fact]
    public void EmptyNormalBatchStillValidatesSharedScale()
    {
        var tape = new Tape();
        var emptyMean = tape.Leaf(new Vector<double>(0));
        Assert.Throws<DistributionParameterOutOfRangeException>(() => NormalDistribution.LogPdf(emptyMean, -1, default));
        Assert.Throws<DistributionParameterOutOfRangeException>(() => NormalDistribution.LogPdfPointwise(0, -1, default, tape));
    }

    [Fact]
    public void EmptyUniformBatchValidatesBothBoundsWhenSharedAndEachSuppliedBound()
    {
        var tape = new Tape();
        var empty = tape.Leaf(new Vector<double>(0));
        Assert.Throws<DistributionParameterOutOfRangeException>(() => UniformDistribution.LogPdfPointwise(2, 1, default, tape));
        Assert.Throws<DistributionParameterOutOfRangeException>(() => UniformDistribution.LogPdfPointwise(-double.MaxValue, double.MaxValue, default, tape));
        Assert.Throws<DistributionParameterOutOfRangeException>(() => UniformDistribution.LogPdf(double.NegativeInfinity, empty, default));
        Assert.Throws<DistributionParameterOutOfRangeException>(() => UniformDistribution.LogPdfPointwise(empty, double.PositiveInfinity, default));
        Assert.Equal(0, UniformDistribution.LogPdf(double.MaxValue, empty, default).Value);
        Assert.Equal(0, UniformDistribution.LogPdfPointwise(empty, -double.MaxValue, default).Length);
    }

    [Fact]
    public void EmptyBinomialBatchValidatesSharedTrialsAndProbability()
    {
        var tape = new Tape();
        var emptyProbability = tape.Leaf(new Vector<double>(0));
        var emptyTrials = new Vector<long>(0);
        Assert.Throws<ArgumentOutOfRangeException>("trials", () => BinomialDistribution.LogPmf(-1, emptyProbability, default));
        Assert.Throws<ArgumentOutOfRangeException>("trials", () => BinomialDistribution.LogPmfPointwise(BinomialDistribution.MaxTrials + 1, emptyProbability, default));
        Assert.Throws<DistributionParameterOutOfRangeException>(() => BinomialDistribution.LogPmf(emptyTrials, -0.1, default));
        Assert.Throws<DistributionParameterOutOfRangeException>(() => BinomialDistribution.LogPmfPointwise(emptyTrials, -0.1, default, tape));
        Assert.Equal(0, BinomialDistribution.LogPmf(emptyTrials, emptyProbability, default).Value);
        Assert.Equal(0, BinomialDistribution.LogPmfPointwise(emptyTrials, 0.5, default, tape).Length);
    }

    [Fact]
    public void FailedBatchCanBeRetriedWithExistingParameterGradients()
    {
        var tape = new Tape();
        var parameter = Parameter<double>.Vector(3);
        new double[] { 0, 1, 2 }.CopyTo(parameter.Value);
        parameter.Gradient.Fill(10);
        var observations = new Vector<double>([1, 3, 6]);
        var invalidScales = new Vector<double>([2, -1, 2]);
        for (int repetition = 0; repetition < 2; repetition++)
        {
            tape.Reset();
            var means = tape.UseVector(parameter);
            var scales = tape.Leaf(invalidScales);
            var beforeFailure = NormalDistribution.LogPdf(means, 2, observations);
            Assert.Throws<DistributionParameterOutOfRangeException>(() => NormalDistribution.LogPdf(means, scales, observations));
            Assert.Throws<DistributionParameterOutOfRangeException>(() => NormalDistribution.LogPdfPointwise(means, scales, observations));
            var afterFailure = NormalDistribution.LogPdfPointwise(means, 2, observations);
            tape.Backward(beforeFailure + XMath.Sum(afterFailure));
        }

        Assert.Equal([11d, 12d, 14d], parameter.Gradient.ToArray());
    }

    [Fact]
    public void StudentTDegreesOfFreedomAtInfinityUseTheNormalLimit()
    {
        var tape = new Tape();
        var observations = new Vector<double>([0, 1.2]);
        var degrees = tape.Leaf(new Vector<double>([double.PositiveInfinity, double.PositiveInfinity]));
        var locations = tape.Constant(new Vector<double>([0.4, 0.8]));
        var scale = tape.Leaf(1.7);
        var normal = NormalDistribution.LogPdf(locations, scale, observations);
        var student = StudentTDistribution.LogPdfPointwise(degrees, locations, scale, observations);
        Assert.Equal(normal.Value, student.Value[0] + student.Value[1]);
        tape.Backward(XMath.Sum(student) - normal);
        Assert.Equal(0, tape.GetGradient(degrees)[0]);
        Assert.Equal(0, tape.GetGradient(degrees)[1]);
        Close(0, tape.GetGradient(scale));
    }

    [Fact]
    public void BernoulliKeepsOneSidedParameterDerivativesAtZeroAndOne()
    {
        var tape = new Tape();
        var probabilities = tape.Leaf(new Vector<double>([0, 1, 0, 1]));
        var counts = new Vector<long>([0, 1, 1, 0]);
        var values = BernoulliDistribution.LogPmfPointwise(probabilities, counts);
        Assert.Equal([0, 0, double.NegativeInfinity, double.NegativeInfinity], values.Value.ToArray());
        tape.Backward(BernoulliDistribution.LogPmf(probabilities, counts));
        Assert.Equal([-1, 1, double.PositiveInfinity, double.NegativeInfinity], tape.GetGradient(probabilities).ToArray());
    }

    [Fact]
    public void BinomialUsesOneExactIntegerTrialCountPerObservation()
    {
        long[] trials = [0, 5, 10, BinomialDistribution.MaxTrials];
        long[] counts = [0, 2, 4, BinomialDistribution.MaxTrials / 2];
        double[] probabilities = [1, 0.4, 0.7, 0.5];
        var tape = new Tape();
        var p = tape.Leaf(new Vector<double>(probabilities));
        var observations = new Vector<long>(counts);
        var trialCounts = new Vector<long>(trials);
        var values = BinomialDistribution.LogPmfPointwise(trialCounts, p, observations);
        var sum = BinomialDistribution.LogPmf(trialCounts, p, observations);
        double expectedSum = 0;
        for (int i = 0; i < counts.Length; i++)
        {
            double value = new BinomialDistribution(trials[i], probabilities[i]).LogPmf(counts[i]);
            Assert.Equal(value, values.Value[i]);
            expectedSum += value;
        }

        Assert.Equal(expectedSum, sum.Value);
        tape.Backward(sum - XMath.Sum(values));
        for (int i = 0; i < counts.Length; i++)
        {
            Assert.Equal(0, tape.GetGradient(p)[i]);
        }

        tape.Reset();
        var shared = tape.Leaf(0.6);
        var shortTrials = new Vector<long>([0, 5]);
        Assert.Throws<ArgumentException>(() => BinomialDistribution.LogPmf(shortTrials, shared, observations));
        trialCounts[3] = BinomialDistribution.MaxTrials + 1;
        Assert.Throws<ArgumentOutOfRangeException>(() => BinomialDistribution.LogPmfPointwise(trialCounts, shared, observations));
        trialCounts[3] = BinomialDistribution.MaxTrials;
        tape.Backward(BinomialDistribution.LogPmf(trialCounts, shared, observations));
        Assert.True(double.IsFinite(tape.GetGradient(shared)));
    }

    [Fact]
    public void CategoricalUsesColumnWeightsAndBackpropagatesThroughViews()
    {
        var storage = new Matrix<double>(new double[,] { { 2, 3, 4 }, { 5, 1, 0.5 }, { 1, 6, 2 }, { 7, 8, 9 } });
        var observations = new Vector<long>([2, 0, 1]);
        var upstream = new Vector<double>([2, -3, 0.5]);
        var tape = new Tape();
        var leaf = tape.Leaf(storage);
        var weights = leaf.T[.., ..3];
        var values = CategoricalDistribution.LogPmfPointwise(weights, observations);
        var sum = CategoricalDistribution.LogPmf(weights, observations);
        double expectedSum = 0;
        for (int i = 0; i < observations.Length; i++)
        {
            var point = storage.Row(i).ToArray();
            ScalarNode[] scalar = [.. point.Select(static weight => (ScalarNode)weight)];
            double value = CategoricalDistribution.LogPmf(scalar, observations[i]).Value;
            Assert.Equal(value, values.Value[i]);
            expectedSum += value;
        }

        Assert.Equal(expectedSum, sum.Value);
        tape.Backward(sum + XMath.Dot(values, tape.Constant(upstream)));
        var gradient = tape.GetGradient(leaf);
        for (int i = 0; i < observations.Length; i++)
        {
            var column = storage.Row(i).ToArray();
            for (int j = 0; j < column.Length; j++)
            {
                const double h = 1e-5;
                double value = column[j];
                column[j] = value + h;
                double plus = new CategoricalDistribution(column).LogPmf(observations[i]);
                column[j] = value - h;
                double minus = new CategoricalDistribution(column).LogPmf(observations[i]);
                column[j] = value;
                Close((1 + upstream[i]) * (plus - minus) / (2 * h), gradient[i, j], 3e-8);
            }

            Assert.Equal(0, gradient[3, i]);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CategoricalMasksInfinitePartialsAndReducesSharedWeights(bool broadcast)
    {
        var tape = new Tape();
        var weights = tape.Leaf(new Vector<double>([1, 0, 3]));
        var observations = new Vector<long>([1, 2, long.MaxValue]);
        var values = broadcast
            ? CategoricalDistribution.LogPmfPointwise(weights.Broadcast(3), observations)
            : CategoricalDistribution.LogPmfPointwise(weights, observations);
        Assert.Equal(double.NegativeInfinity, values.Value[0]);
        Close(Math.Log(0.75), values.Value[1]);
        Assert.Equal(double.NegativeInfinity, values.Value[2]);
        tape.Backward(XMath.Dot(values, tape.Constant(new Vector<double>([0, 2, 5]))));
        var gradient = tape.GetGradient(weights);
        Assert.Equal(-0.5, gradient[0]);
        Assert.Equal(-0.5, gradient[1]);
        Close(1.0 / 6, gradient[2]);
    }

    [Fact]
    public void CategoricalSharedVectorMatchesTheScalarWeightLikelihood()
    {
        var weights = new Vector<double>([1e-18, 1, 3e-18]);
        var observations = new Vector<long>([1, 1, 1]);
        var reference = new Tape();
        var scalarWeights = reference.Leaves(weights);
        var expected = CategoricalDistribution.LogPmf(scalarWeights, observations);
        reference.Backward(expected);

        var tape = new Tape();
        var sharedWeights = tape.Leaf(weights);
        var result = CategoricalDistribution.LogPmf(sharedWeights, observations);
        Assert.Equal(expected.Value, result.Value);
        tape.Backward(result);
        var gradient = tape.GetGradient(sharedWeights);
        for (int i = 0; i < weights.Length; i++)
        {
            Assert.Equal(reference.GetGradient(scalarWeights[i]), gradient[i]);
        }

        tape.Reset();
        var sharedConstants = tape.Constant(weights);
        var values = CategoricalDistribution.LogPmfPointwise(sharedConstants, observations);
        Assert.False(values.RequiresGradient);
        double point = CategoricalDistribution.LogPmf([(ScalarNode)weights[0], (ScalarNode)weights[1], (ScalarNode)weights[2]], 1).Value;
        Assert.Equal([point, point, point], values.Value.ToArray());
    }

    [Fact]
    public void CategoricalValidatesSharedWeightsWithEmptyObservations()
    {
        var tape = new Tape();
        var weights = tape.Leaf(new Vector<double>([1, 2]));
        var emptyWeights = tape.Leaf(new Vector<double>(0));
        Assert.Throws<ArgumentException>("weights", () => CategoricalDistribution.LogPmf(emptyWeights, default));
        Assert.Throws<ArgumentException>("weights", () => CategoricalDistribution.LogPmfPointwise(emptyWeights, default));
        var sum = CategoricalDistribution.LogPmf(weights, default);
        Assert.Equal(0, sum.Value);
        Assert.Equal(0, CategoricalDistribution.LogPmfPointwise(weights, default).Length);
        tape.Backward(sum);
        Assert.Equal([0d, 0d], tape.GetGradient(weights).ToArray());
    }

    [Theory]
    [InlineData(-1, 1)]
    [InlineData(double.NaN, 1)]
    [InlineData(0, 0)]
    [InlineData(double.MaxValue, double.MaxValue)]
    public void CategoricalRejectsInvalidWeights(double first, double second)
    {
        var tape = new Tape();
        var observations = new Vector<long>([0, 1]);
        var weights = tape.Leaf(new Matrix<double>(new double[,] { { 1, first }, { 1, second } }));
        Assert.Throws<DistributionParameterOutOfRangeException>(() => CategoricalDistribution.LogPmf(weights, observations));
        Assert.Throws<DistributionParameterOutOfRangeException>(() => CategoricalDistribution.LogPmfPointwise(weights, observations));
        Assert.Throws<ArgumentException>("weights", () => CategoricalDistribution.LogPmf(weights[.., ..1], observations));
        Assert.Throws<ArgumentException>(() => CategoricalDistribution.LogPmfPointwise(weights[..0, ..], observations));
        var shared = tape.Leaf(new Vector<double>([first, second]));
        Assert.Throws<DistributionParameterOutOfRangeException>(() => CategoricalDistribution.LogPmf(shared, observations));
        Assert.Throws<DistributionParameterOutOfRangeException>(() => CategoricalDistribution.LogPmfPointwise(shared, observations));
        Assert.Throws<DistributionParameterOutOfRangeException>(() => CategoricalDistribution.LogPmf(shared, default));
        Assert.Throws<DistributionParameterOutOfRangeException>(() => CategoricalDistribution.LogPmfPointwise(shared, default));

        var empty = weights[.., ..0];
        var sum = CategoricalDistribution.LogPmf(empty, default);
        var values = CategoricalDistribution.LogPmfPointwise(empty, default);
        Assert.Equal(0, sum.Value);
        Assert.Equal(0, values.Length);
        tape.Backward(sum);
        Assert.Equal(0, tape.GetGradient(weights)[0, 0]);
    }

    private static void CheckDistribution<T>(Distribution<T> distribution) where T : unmanaged, Numerics.INumberBase<T>
    {
        CheckBatch(distribution, distribution.Observations, shared: -1, finiteDifferences: true);
        CheckBatch(distribution, distribution.Observations, shared: distribution.Parameters.Length - 1, finiteDifferences: false);
        CheckBatch(distribution, distribution.Boundaries, shared: -1, finiteDifferences: false);
        CheckInvalidSharedParametersOnEmptyBatch(distribution);

        var tape = new Tape();
        var inputs = distribution.Parameters.Select(static value => (ScalarSetNode<double>)value).ToArray();
        var observations = new Vector<T>(distribution.Observations);
        double first = distribution.Parameters[0];
        inputs[0] = tape.Leaf(new Vector<double>([first, double.NaN, first]));
        Assert.Throws<DistributionParameterOutOfRangeException>(() => distribution.Sum(inputs, observations));
        Assert.Throws<DistributionParameterOutOfRangeException>(() => distribution.Pointwise(inputs, observations));
        var empty = tape.Leaf(new Vector<double>(0));
        inputs[0] = empty;
        Assert.Throws<ArgumentException>(() => distribution.Sum(inputs, observations));
        Assert.Throws<ArgumentException>(() => distribution.Pointwise(inputs, observations));

        var sum = distribution.Sum(inputs, default);
        var pointwise = distribution.Pointwise(inputs, default);
        Assert.Equal(0, sum.Value);
        Assert.Equal(0, pointwise.Length);
        tape.Backward(sum);
        Assert.Equal(0, tape.GetGradient(empty).Length);
    }

    private static void CheckInvalidSharedParametersOnEmptyBatch<T>(Distribution<T> distribution) where T : unmanaged, Numerics.INumberBase<T>
    {
        var tape = new Tape();
        ScalarSetNode<double> empty = tape.Leaf(new Vector<double>(0));
        ScalarSetNode<double> invalid = tape.Leaf(double.NaN);
        var inputs = new ScalarSetNode<double>[distribution.Parameters.Length];
        inputs.AsSpan().Fill(empty);
        for (int i = 0; i < inputs.Length; i++)
        {
            inputs[i] = invalid;
            Assert.Throws<DistributionParameterOutOfRangeException>(() => distribution.Sum(inputs, default));
            Assert.Throws<DistributionParameterOutOfRangeException>(() => distribution.Pointwise(inputs, default));
            inputs[i] = empty;
        }
    }

    private static void CheckBatch<T>(Distribution<T> distribution, T[] data, int shared, bool finiteDifferences)
        where T : unmanaged, Numerics.INumberBase<T>
    {
        double[][] parameters = distribution.Parameters.Select((p, j) => data.Select((_, i) => j == shared ? p : p + 0.05 * (i % 3 - 1)).ToArray()).ToArray();
        var observations = new Vector<T>(data);
        var weights = Vector<double>.Generate(data.Length, static i => i == 2 ? 0 : 0.5 - i);
        var referenceTape = new Tape();
        double[][] derivatives = parameters.Select(p => new double[p.Length]).ToArray();
        double[] expectedValues = new double[data.Length];
        double expectedSum = 0;
        for (int i = 0; i < data.Length; i++)
        {
            referenceTape.Reset();
            double[] values = parameters.Select(p => p[i]).ToArray();
            var nodes = values.Select(referenceTape.Leaf).ToArray();
            ScalarNode reference = distribution.Single(nodes, data[i]);
            referenceTape.Backward(reference);
            expectedValues[i] = reference.Value;
            expectedSum += reference.Value;
            for (int j = 0; j < nodes.Length; j++)
            {
                double derivative = referenceTape.GetGradient(nodes[j]);
                derivatives[j][i] = derivative;
                if (finiteDifferences)
                {
                    const double h = 1e-5;
                    ScalarNode[] constants = [.. values.Select(static v => (ScalarNode)v)];
                    constants[j] = values[j] + h;
                    double plus = distribution.Single(constants, data[i]).Value;
                    constants[j] = values[j] - h;
                    double minus = distribution.Single(constants, data[i]).Value;
                    Close((plus - minus) / (2 * h), derivative, 3e-8);
                }
            }
        }

        var tape = new Tape();
        CheckGradients(pointwise: false);
        CheckGradients(pointwise: true);

        void CheckGradients(bool pointwise)
        {
            tape.Reset();
            var inputs = new ScalarSetNode<double>[parameters.Length];
            var vectors = new VectorNode<double>[parameters.Length];
            ScalarNode scalar = default;
            for (int j = 0; j < inputs.Length; j++)
            {
                if (j == shared)
                {
                    scalar = tape.Leaf(parameters[j][0]);
                    inputs[j] = scalar;
                }
                else
                {
                    vectors[j] = tape.Leaf(new Vector<double>(parameters[j]));
                    inputs[j] = vectors[j];
                }
            }

            if (pointwise)
            {
                var result = distribution.Pointwise(inputs, observations);
                Assert.Equal(expectedValues, result.Value.ToArray());
                tape.Backward(XMath.Dot(result, tape.Constant(weights)));
            }
            else
            {
                var result = distribution.Sum(inputs, observations);
                Assert.Equal(expectedSum, result.Value);
                tape.Backward(result);
            }

            for (int j = 0; j < inputs.Length; j++)
            {
                double expectedGradient = 0;
                for (int i = 0; i < data.Length; i++)
                {
                    double local = derivatives[j][i];
                    double contribution = pointwise ? (weights[i] == 0 || local == 0 ? 0 : weights[i] * local) : local;
                    if (j == shared)
                    {
                        expectedGradient += contribution;
                    }
                    else
                    {
                        Close(contribution, tape.GetGradient(vectors[j])[i]);
                    }
                }

                if (j == shared)
                {
                    Close(expectedGradient, tape.GetGradient(scalar));
                }
            }
        }
    }

    private static void Close(double expected, double actual, double tolerance = 3e-13)
    {
        if (!double.IsFinite(expected))
        {
            Assert.Equal(expected, actual);
            return;
        }

        double error = tolerance * Math.Max(1, Math.Abs(expected));
        Assert.InRange(actual, expected - error, expected + error);
    }

    private delegate ScalarNode Single<T>(ReadOnlySpan<ScalarNode> parameters, T observation);

    private delegate ScalarNode Sum<T>(ReadOnlySpan<ScalarSetNode<double>> parameters, ReadOnlyVectorSpan<T> observations)
        where T : unmanaged, Numerics.INumberBase<T>;

    private delegate VectorNode<double> Pointwise<T>(ReadOnlySpan<ScalarSetNode<double>> parameters, ReadOnlyVectorSpan<T> observations)
        where T : unmanaged, Numerics.INumberBase<T>;

    private sealed record Distribution<T>(double[] Parameters, T[] Observations, T[] Boundaries, Single<T> Single, Sum<T> Sum, Pointwise<T> Pointwise)
        where T : unmanaged, Numerics.INumberBase<T>;
}
