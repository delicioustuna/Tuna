using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Statistics;

public class RemainingDistributionDifferentiationTests
{
    [Theory]
    [InlineData("Exponential", 1.1, 0, 1.3)]
    [InlineData("Logistic", 0.2, 1.4, -0.9)]
    [InlineData("Laplace", -0.9, 0.8, 1.2)]
    [InlineData("Gumbel", 0.45, 0.97, 0.8)]
    [InlineData("Weibull", 1.3, 2.7, 1.25)]
    [InlineData("Uniform", -2.2, 3.1, 1.7)]
    [InlineData("LogNormal", -0.25, 1.9, 2.5)]
    [InlineData("F", 3.6, 5.2, 1.5)]
    public void ContinuousLogProbabilitiesMatchValuesAndCenteredDifferences(string name, double first, double second, double x)
    {
        double[] point = [first, second, x];
        for (int kind = 0; kind < 3; kind++)
        {
            int operation = kind;
            Verify(point, p => Continuous(name, p, operation), p => ContinuousValue(name, p, operation),
                i => !(name == "F" && operation != 0 && i < 2));
        }
    }

    [Theory]
    [InlineData("Poisson", 3.4, 0, 4)]
    [InlineData("Binomial", 0.57, 0, 4)]
    [InlineData("Geometric", 0.27, 0, 3)]
    [InlineData("NegativeBinomial", 2.7, 0.42, 5)]
    public void DiscreteLogProbabilitiesMatchValuesAndCenteredDifferences(string name, double first, double second, long k)
    {
        double[] point = [first, second];
        for (int kind = 0; kind < 3; kind++)
        {
            int operation = kind;
            Verify(point, p => Discrete(name, p, k, operation), p => DiscreteValue(name, p, k, operation),
                i => !(name == "NegativeBinomial" && operation != 0 && i == 0));
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void CategoricalWeightsMatchValuesAndCenteredDifferences(int kind)
    {
        double[] weights = [0.4, 1.7, 0.22, 0.5];
        Verify(weights, p => Categorical(p, 1, kind), p => EvaluateDiscrete(new CategoricalDistribution(p), 1, kind), _ => true);
    }

    [Fact]
    public void CategoricalLikelihoodPreservesObservationOrderAndGradient()
    {
        double[] weights = [1.1, 2.2, 0.18];
        long[] observations = [2, 0, 1, 1, 2, 0];
        Verify(weights, p => CategoricalDistribution.LogPmf(p, new ReadOnlyVectorSpan<long>(observations)), p =>
        {
            var distribution = new CategoricalDistribution(p);
            double result = 0;
            foreach (long k in observations)
            {
                result += distribution.LogPmf(k);
            }

            return result;
        }, _ => true);
    }

    [Fact]
    public void UnsupportedTailShapeDerivativesAreReported()
    {
        var tape = new Tape();
        var shape = tape.Leaf(2);
        Assert.Throws<NotSupportedException>(() => FDistribution.LogSf(5, shape, 1));
        Assert.Throws<NotSupportedException>(() => NegativeBinomialDistribution.LogCdf(shape, 0.7, 2));
    }

    [Fact]
    public void DeepLogisticTailRetainsVariateGradient()
    {
        var tape = new Tape();
        double scale = Math.ScaleB(1, -800);
        var x = tape.Leaf(750 * scale);
        var value = LogisticDistribution.LogCdf(0, scale, x);
        tape.Backward(value);
        double expected = Math.Exp(-750 + 800 * Math.Log(2));
        Assert.InRange(Math.Abs(tape.GetGradient(x) / expected - 1), 0, 2e-13);
    }

    [Fact]
    public void GumbelScaleScoreStaysFiniteWhenTheUnscaledProductOverflows()
    {
        var tape = new Tape();
        var scale = tape.Leaf(1e305);
        const double observation = -7.06e307;
        var density = GumbelDistribution.LogPdf(0, scale, observation);
        tape.Backward(density);
        double z = observation / scale.Value;
        double expected = -z * (Math.Exp(-z) / scale.Value);
        Assert.InRange(Math.Abs(tape.GetGradient(scale) / expected - 1), 0, 2e-13);
    }

    [Fact]
    public void WeibullLowerTailRetainsGradientsWhenThePowerUnderflows()
    {
        var tape = new Tape();
        var shape = tape.Leaf(2);
        var scale = tape.Leaf(1e200);
        var x = tape.Leaf(1e-200);
        var value = WeibullDistribution.LogCdf(shape, scale, x);
        Assert.True(double.IsFinite(value.Value));
        tape.Backward(value);
        Assert.InRange(Math.Abs(tape.GetGradient(x) / 2e200 - 1), 0, 8e-16);
        Assert.InRange(Math.Abs(tape.GetGradient(scale) / -2e-200 - 1), 0, 8e-16);
        Assert.InRange(Math.Abs(tape.GetGradient(shape) + 400 * Math.Log(10)), 0, 2e-12);
    }

    [Fact]
    public void PoissonScoreRetainsTheLowBitOfTheCount()
    {
        var tape = new Tape();
        const long count = (1L << 55) + 1;
        var rate = tape.Leaf(1L << 55);
        var value = PoissonDistribution.LogPmf(rate, count);
        tape.Backward(value);
        Assert.Equal(Math.ScaleB(1, -55), tape.GetGradient(rate));
    }

    [Fact]
    public void CategoricalDominantWeightUsesTheComplementBeforeRounding()
    {
        var tape = new Tape();
        var dominant = tape.Leaf(1);
        var small = tape.Leaf(1e-200);
        var value = CategoricalDistribution.LogPmf([dominant, small], 0);
        Assert.Equal(-1e-200, value.Value);
        tape.Backward(value);
        Assert.Equal(1e-200, tape.GetGradient(dominant));
        Assert.Equal(-1, tape.GetGradient(small));
    }

    [Fact]
    public void CategoricalReportsOverflowOfFiniteWeightsAsAParameterError()
    {
        var tape = new Tape();
        var weight = tape.Leaf(Math.Exp(709.5));
        Assert.True(double.IsFinite(weight.Value));
        double secondWeight = weight.Value;
        Assert.Throws<DistributionParameterOutOfRangeException>(() => CategoricalDistribution.LogPmf([weight, secondWeight], 1));
        Assert.Throws<ArgumentException>(() => CategoricalDistribution.LogPmf([], 1));
    }

    [Theory]
    [InlineData(568.06, 2.34, 8.94e7, -6.154082632960866e-7)]
    [InlineData(217.63, 1.21, 8.28e5, 5.042346753166296e-6)]
    public void FNumeratorScoreRetainsPrecisionInTheUpperTail(double numerator, double denominator, double x, double expected)
    {
        var tape = new Tape();
        var degrees = tape.Leaf(numerator);
        tape.Backward(FDistribution.LogPdf(degrees, denominator, x));
        Assert.InRange(Math.Abs(tape.GetGradient(degrees) / expected - 1), 0, 5e-12);
    }

    [Fact]
    public void ExponentialLowerTailRetainsGradientsAtSmallFiniteArguments()
    {
        var tape = new Tape();
        var rate = tape.Leaf(1e-100);
        var x = tape.Leaf(1e-190);
        tape.Backward(ExponentialDistribution.LogCdf(rate, x));
        Assert.Equal(1 / rate.Value, tape.GetGradient(rate));
        Assert.Equal(1 / x.Value, tape.GetGradient(x));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NegativeBinomialBatchScoresMatchScalarScoresAcrossCountRanges(bool differentiateShape)
    {
        long[] counts = [0, 1, 127, 63, -1, 0, 63, 127, 1, 64, 64];
        var tape = new Tape();
        ScalarNode shape = differentiateShape ? tape.Leaf(2.6) : 2.6;
        var probability = tape.Leaf(0.74);
        ScalarNode scalar = 0;
        foreach (long k in counts)
        {
            scalar += NegativeBinomialDistribution.LogPmf(shape, probability, k);
        }

        var batch = NegativeBinomialDistribution.LogPmf(shape, probability, new ReadOnlyVectorSpan<long>(counts));
        Assert.Equal(scalar.Value, batch.Value);
        var expected = new double[differentiateShape ? 2 : 1];
        tape.Gradient(scalar, expected);

        tape.Reset();
        shape = differentiateShape ? tape.Leaf(2.6) : 2.6;
        probability = tape.Leaf(0.74);
        batch = NegativeBinomialDistribution.LogPmf(shape, probability, new ReadOnlyVectorSpan<long>(counts));
        var actual = new double[expected.Length];
        tape.Gradient(batch, actual);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.InRange(Math.Abs(actual[i] / expected[i] - 1), 0, 2e-15);
        }
    }

    [Fact]
    public void BatchScoresMatchTheScalarScoresOnStridedObservations()
    {
        double[] storage = [0.3, double.NaN, 1.2, double.NaN, 2.8];
        var tape = new Tape();
        var shape = tape.Leaf(2);
        var scale = tape.Leaf(1.4);
        var scalar = WeibullDistribution.LogPdf(shape, scale, 0.3)
            + WeibullDistribution.LogPdf(shape, scale, 1.2) + WeibullDistribution.LogPdf(shape, scale, 2.8);
        ReadOnlyVectorSpan<double> observations = new(storage, [3], [2]);
        var batch = WeibullDistribution.LogPdf(shape, scale, observations);
        Assert.Equal(scalar.Value, batch.Value);
        tape.Backward(scalar);
        double dShape = tape.GetGradient(shape), dScale = tape.GetGradient(scale);
        tape.Reset();
        shape = tape.Leaf(2);
        scale = tape.Leaf(1.4);
        batch = WeibullDistribution.LogPdf(shape, scale, observations);
        Array.Fill(storage, double.NaN);
        tape.Backward(batch);
        Assert.InRange(Math.Abs(dShape - tape.GetGradient(shape)), 0, 1e-14);
        Assert.InRange(Math.Abs(dScale - tape.GetGradient(scale)), 0, 1e-14);
    }

    internal static ScalarNode Continuous(string name, ReadOnlySpan<ScalarNode> p, int kind)
    {
        var a = p[0];
        var b = p[1];
        var x = p[^1];
        return (name, kind) switch
        {
            ("Exponential", 0) => ExponentialDistribution.LogPdf(a, x),
            ("Exponential", 1) => ExponentialDistribution.LogCdf(a, x),
            ("Exponential", _) => ExponentialDistribution.LogSf(a, x),
            ("Logistic", 0) => LogisticDistribution.LogPdf(a, b, x),
            ("Logistic", 1) => LogisticDistribution.LogCdf(a, b, x),
            ("Logistic", _) => LogisticDistribution.LogSf(a, b, x),
            ("Laplace", 0) => LaplaceDistribution.LogPdf(a, b, x),
            ("Laplace", 1) => LaplaceDistribution.LogCdf(a, b, x),
            ("Laplace", _) => LaplaceDistribution.LogSf(a, b, x),
            ("Gumbel", 0) => GumbelDistribution.LogPdf(a, b, x),
            ("Gumbel", 1) => GumbelDistribution.LogCdf(a, b, x),
            ("Gumbel", _) => GumbelDistribution.LogSf(a, b, x),
            ("Weibull", 0) => WeibullDistribution.LogPdf(a, b, x),
            ("Weibull", 1) => WeibullDistribution.LogCdf(a, b, x),
            ("Weibull", _) => WeibullDistribution.LogSf(a, b, x),
            ("Uniform", 0) => UniformDistribution.LogPdf(a, b, x),
            ("Uniform", 1) => UniformDistribution.LogCdf(a, b, x),
            ("Uniform", _) => UniformDistribution.LogSf(a, b, x),
            ("LogNormal", 0) => LogNormalDistribution.LogPdf(a, b, x),
            ("LogNormal", 1) => LogNormalDistribution.LogCdf(a, b, x),
            ("LogNormal", _) => LogNormalDistribution.LogSf(a, b, x),
            ("F", 0) => FDistribution.LogPdf(a, b, x),
            ("F", 1) => FDistribution.LogCdf(a.Value, b.Value, x),
            ("F", _) => FDistribution.LogSf(a.Value, b.Value, x),
            _ => throw new ArgumentException(nameof(name)),
        };
    }

    private static double ContinuousValue(string name, double[] p, int kind)
    {
        IContinuousDistribution distribution = name switch
        {
            "Exponential" => new ExponentialDistribution(p[0]),
            "Logistic" => new LogisticDistribution(p[0], p[1]),
            "Laplace" => new LaplaceDistribution(p[0], p[1]),
            "Gumbel" => new GumbelDistribution(p[0], p[1]),
            "Weibull" => new WeibullDistribution(p[0], p[1]),
            "Uniform" => new UniformDistribution(p[0], p[1]),
            "LogNormal" => new LogNormalDistribution(p[0], p[1]),
            "F" => new FDistribution(p[0], p[1]),
            _ => throw new ArgumentException(nameof(name)),
        };
        return kind switch
        {
            0 => distribution.LogPdf(p[2]),
            1 => distribution.LogCdf(p[2]),
            _ => distribution.LogSf(p[2]),
        };
    }

    internal static ScalarNode Discrete(string name, ReadOnlySpan<ScalarNode> p, long k, int kind, long trials = 10) => (name, kind) switch
    {
        ("Poisson", 0) => PoissonDistribution.LogPmf(p[0], k),
        ("Poisson", 1) => PoissonDistribution.LogCdf(p[0], k),
        ("Poisson", _) => PoissonDistribution.LogSf(p[0], k),
        ("Binomial", 0) => BinomialDistribution.LogPmf(trials, p[0], k),
        ("Binomial", 1) => BinomialDistribution.LogCdf(trials, p[0], k),
        ("Binomial", _) => BinomialDistribution.LogSf(trials, p[0], k),
        ("Geometric", 0) => GeometricDistribution.LogPmf(p[0], k),
        ("Geometric", 1) => GeometricDistribution.LogCdf(p[0], k),
        ("Geometric", _) => GeometricDistribution.LogSf(p[0], k),
        ("NegativeBinomial", 0) => NegativeBinomialDistribution.LogPmf(p[0], p[1], k),
        ("NegativeBinomial", 1) => NegativeBinomialDistribution.LogCdf(p[0].Value, p[1], k),
        ("NegativeBinomial", _) => NegativeBinomialDistribution.LogSf(p[0].Value, p[1], k),
        _ => throw new ArgumentException(nameof(name)),
    };

    internal static ScalarNode Categorical(ReadOnlySpan<ScalarNode> weights, long k, int kind) => kind switch
    {
        0 => CategoricalDistribution.LogPmf(weights, k),
        1 => CategoricalDistribution.LogCdf(weights, k),
        _ => CategoricalDistribution.LogSf(weights, k),
    };

    private static double DiscreteValue(string name, double[] p, long k, int kind)
    {
        IDiscreteDistribution distribution = name switch
        {
            "Poisson" => new PoissonDistribution(p[0]),
            "Binomial" => new BinomialDistribution(10, p[0]),
            "Geometric" => new GeometricDistribution(p[0]),
            "NegativeBinomial" => new NegativeBinomialDistribution(p[0], p[1]),
            _ => throw new ArgumentException(nameof(name)),
        };
        return EvaluateDiscrete(distribution, k, kind);
    }

    private static double EvaluateDiscrete(IDiscreteDistribution distribution, long k, int kind) => kind switch
    {
        0 => distribution.LogPmf(k),
        1 => distribution.LogCdf(k),
        _ => distribution.LogSf(k),
    };

    private static void Verify(double[] point, Func<ReadOnlySpan<ScalarNode>, ScalarNode> model, Func<double[], double> reference,
        Func<int, bool> supportsGradient)
    {
        var tape = new Tape();
        var leaves = tape.Leaves(new ReadOnlyVectorSpan<double>(point));
        var result = model(leaves);
        Assert.Equal(BitConverter.DoubleToInt64Bits(reference(point)), BitConverter.DoubleToInt64Bits(result.Value));
        tape.Backward(result);
        for (int i = 0; i < point.Length; i++)
        {
            if (!supportsGradient(i))
            {
                continue;
            }

            double original = point[i];
            const double step = 1e-5;
            point[i] = original + step;
            double plus = reference(point);
            point[i] = original - step;
            double minus = reference(point);
            point[i] = original;
            double difference = (plus - minus) / (2 * step);
            double actual = tape.GetGradient(leaves[i]);
            Assert.True(Math.Abs(actual - difference) <= 1e-8 * Math.Max(1, Math.Abs(difference)),
                $"Parameter {i}: expected {difference:R}, got {actual:R}");
        }
    }
}
