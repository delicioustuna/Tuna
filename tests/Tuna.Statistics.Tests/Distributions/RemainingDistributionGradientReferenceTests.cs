using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using Tuna.Statistics.Distributions;
using Tuna.Tests.ReferenceData;
using Xunit.Abstractions;

namespace Tuna.Tests.Statistics;

internal enum DistributionGradientFunction
{
    Density,
    LogCdf,
    LogSf,
    LogLikelihood,
}

internal readonly record struct DistributionDerivativeReference(double Expected, double Scale = 0);

internal readonly record struct DistributionGradientCase<T>(
    DistributionGradientFunction Function,
    double[] Inputs,
    T[] Observations,
    DistributionDerivativeReference?[] Derivatives,
    long Trials = 0);

public class RemainingDistributionGradientReferenceTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData("Exponential")]
    [InlineData("Logistic")]
    [InlineData("Laplace")]
    [InlineData("Gumbel")]
    [InlineData("Weibull")]
    [InlineData("Uniform")]
    [InlineData("LogNormal")]
    [InlineData("F")]
    public void ContinuousGradientsMatchMpmath(string distribution) =>
        Verify(distribution, RemainingDistributionGradients.Continuous(distribution), Continuous);

    [Theory]
    [InlineData("Poisson")]
    [InlineData("Geometric")]
    [InlineData("Binomial")]
    [InlineData("NegativeBinomial")]
    [InlineData("Categorical")]
    public void DiscreteGradientsMatchMpmath(string distribution) =>
        Verify(distribution, RemainingDistributionGradients.Discrete(distribution), Discrete);

    private void Verify<T>(string distribution, IEnumerable<DistributionGradientCase<T>> cases,
        Func<string, DistributionGradientCase<T>, ScalarNode[], ScalarNode> evaluate)
    {
        var tape = new Tape();
        var statistics = new Dictionary<(DistributionGradientFunction Function, int Argument), ErrorStatistics>();
        var failures = new List<string>();
        foreach (var sample in cases)
        {
            tape.Reset();
            Assert.Equal(sample.Inputs.Length, sample.Derivatives.Length);
            var inputs = new ScalarNode[sample.Inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i] = sample.Derivatives[i].HasValue ? tape.Leaf(sample.Inputs[i]) : sample.Inputs[i];
            }

            tape.Backward(evaluate(distribution, sample, inputs));
            for (int i = 0; i < inputs.Length; i++)
            {
                if (sample.Derivatives[i] is not { } expected)
                {
                    continue;
                }

                var key = (sample.Function, i);
                if (!statistics.TryGetValue(key, out var errors))
                {
                    errors = new ErrorStatistics();
                    statistics.Add(key, errors);
                }

                double actual = tape.GetGradient(inputs[i]);
                if (!errors.Accepts(expected, actual, sample.Function == DistributionGradientFunction.LogLikelihood))
                {
                    string arguments = string.Join(", ", sample.Inputs.Select(static x => x.ToString("R")));
                    string observations = string.Join(", ", sample.Observations.Take(8));
                    failures.Add(FormattableString.Invariant(
                        $"{sample.Function} ({arguments}), trials={sample.Trials}, observations=[{observations}] ({sample.Observations.Length}), argument {i}: expected {expected.Expected:R}, actual {actual:R}, M={expected.Scale:R}"));
                }
            }
        }

        foreach (var ((function, argument), errors) in statistics.OrderBy(entry => entry.Key))
        {
            output.WriteLine(FormattableString.Invariant(
                $"{distribution} {function} arg {argument}: N={errors.Count}, max relative={errors.MaxRelative:E8}, max ULP={errors.MaxUlps}, max |error|/M={errors.MaxScaled:E8}, scaled only={errors.ScaledOnly}"));
        }

        Assert.Equal(4, statistics.Keys.Select(key => key.Function).Distinct().Count());
        Assert.True(failures.Count == 0, $"{distribution}: {failures.Count} gradient mismatches\n{string.Join("\n", failures)}");
    }

    private static ScalarNode Continuous(string distribution, DistributionGradientCase<double> sample, ScalarNode[] inputs)
    {
        if (sample.Function != DistributionGradientFunction.LogLikelihood)
        {
            return RemainingDistributionDifferentiationTests.Continuous(distribution, inputs, (int)sample.Function);
        }

        var observations = new ReadOnlyVectorSpan<double>(sample.Observations);
        return distribution switch
        {
            "Exponential" => ExponentialDistribution.LogPdf(inputs[0], observations),
            "Logistic" => LogisticDistribution.LogPdf(inputs[0], inputs[1], observations),
            "Laplace" => LaplaceDistribution.LogPdf(inputs[0], inputs[1], observations),
            "Gumbel" => GumbelDistribution.LogPdf(inputs[0], inputs[1], observations),
            "Weibull" => WeibullDistribution.LogPdf(inputs[0], inputs[1], observations),
            "Uniform" => UniformDistribution.LogPdf(inputs[0], inputs[1], observations),
            "LogNormal" => LogNormalDistribution.LogPdf(inputs[0], inputs[1], observations),
            "F" => FDistribution.LogPdf(inputs[0], inputs[1], observations),
            _ => throw new ArgumentOutOfRangeException(nameof(distribution)),
        };
    }

    private static ScalarNode Discrete(string distribution, DistributionGradientCase<long> sample, ScalarNode[] inputs)
    {
        if (sample.Function != DistributionGradientFunction.LogLikelihood)
        {
            long k = sample.Observations[0];
            return distribution == "Categorical"
                ? RemainingDistributionDifferentiationTests.Categorical(inputs, k, (int)sample.Function)
                : RemainingDistributionDifferentiationTests.Discrete(distribution, inputs, k, (int)sample.Function, sample.Trials);
        }

        var observations = new ReadOnlyVectorSpan<long>(sample.Observations);
        return distribution switch
        {
            "Poisson" => PoissonDistribution.LogPmf(inputs[0], observations),
            "Geometric" => GeometricDistribution.LogPmf(inputs[0], observations),
            "Binomial" => BinomialDistribution.LogPmf(sample.Trials, inputs[0], observations),
            "NegativeBinomial" => NegativeBinomialDistribution.LogPmf(inputs[0], inputs[1], observations),
            "Categorical" => CategoricalDistribution.LogPmf(inputs, observations),
            _ => throw new ArgumentOutOfRangeException(nameof(distribution)),
        };
    }

    private sealed class ErrorStatistics
    {
        internal int Count { get; private set; }
        internal int ScaledOnly { get; private set; }
        internal double MaxRelative { get; private set; }
        internal ulong MaxUlps { get; private set; }
        internal double MaxScaled { get; private set; }

        internal bool Accepts(DistributionDerivativeReference reference, double actual, bool sum)
        {
            Count++;
            double expected = reference.Expected;
            if (!double.IsFinite(expected) || !double.IsFinite(actual))
            {
                return actual.Equals(expected);
            }

            double error = Math.Abs(actual - expected);
            double relative = expected == 0 ? (error == 0 ? 0 : double.PositiveInfinity) : error / Math.Abs(expected);
            ulong ulps = ReferencePrecision.UlpDistance(actual, expected);
            if (double.IsNormal(expected))
            {
                MaxRelative = Math.Max(MaxRelative, relative);
            }

            MaxUlps = Math.Max(MaxUlps, ulps);

            bool accepted = relative <= ReferencePrecision.SixteenEpsilon || ulps <= 4;
            if (reference.Scale == 0)
            {
                return accepted;
            }

            double scaled = error / reference.Scale;
            MaxScaled = Math.Max(MaxScaled, scaled);
            bool scaledAccepted = scaled <= (sum ? 64 : 16) * ReferencePrecision.Epsilon;
            if (!accepted && scaledAccepted)
            {
                ScaledOnly++;
            }

            return accepted || scaledAccepted;
        }
    }
}
