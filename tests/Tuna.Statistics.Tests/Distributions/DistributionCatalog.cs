using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Statistics;

/// <summary>生成した参照値の分布名とパラメータから分布を作る</summary>
internal static class DistributionCatalog
{
    internal static IContinuousDistribution Continuous(string name, double[] p) => name switch
    {
        "Normal" => new NormalDistribution(p[0], p[1]),
        "LogNormal" => new LogNormalDistribution(p[0], p[1]),
        "Exponential" => new ExponentialDistribution(p[0]),
        "Uniform" => new UniformDistribution(p[0], p[1]),
        "Cauchy" => new CauchyDistribution(p[0], p[1]),
        "Laplace" => new LaplaceDistribution(p[0], p[1]),
        "Logistic" => new LogisticDistribution(p[0], p[1]),
        "Gamma" => new GammaDistribution(p[0], p[1]),
        "ChiSquared" => new ChiSquaredDistribution(p[0]),
        "Beta" => new BetaDistribution(p[0], p[1]),
        "StudentT" => new StudentTDistribution(p[0], p[1], p[2]),
        "Weibull" => new WeibullDistribution(p[0], p[1]),
        "Gumbel" => new GumbelDistribution(p[0], p[1]),
        "InverseGamma" => new InverseGammaDistribution(p[0], p[1]),
        "F" => new FDistribution(p[0], p[1]),
        "LocationScaleBeta" => new LocationScaleDistribution<BetaDistribution>(new BetaDistribution(p[0], p[1]), p[2], p[3]),
        "LocationScaleGamma" => new LocationScaleDistribution<GammaDistribution>(new GammaDistribution(p[0], p[1]), p[2], p[3]),
        "TruncatedNormal" => new TruncatedDistribution<NormalDistribution>(new NormalDistribution(p[0], p[1]), p[2], p[3]),
        "TruncatedLaplace" => new TruncatedDistribution<LaplaceDistribution>(new LaplaceDistribution(p[0], p[1]), p[2], p[3]),
        "TruncatedGamma" => new TruncatedDistribution<GammaDistribution>(new GammaDistribution(p[0], p[1]), p[2], p[3]),
        "TruncatedBeta" => new TruncatedDistribution<BetaDistribution>(new BetaDistribution(p[0], p[1]), p[2], p[3]),
        "TruncatedStudentT" => new TruncatedDistribution<StudentTDistribution>(new StudentTDistribution(p[0], p[1], p[2]), p[3], p[4]),
        "TruncatedCauchy" => new TruncatedDistribution<CauchyDistribution>(new CauchyDistribution(p[0], p[1]), p[2], p[3]),
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, null),
    };

    internal static IDiscreteDistribution Discrete(string name, double[] p) => name switch
    {
        "Bernoulli" => new BernoulliDistribution(p[0]),
        "Binomial" => new BinomialDistribution(checked((long)p[0]), p[1]),
        "Poisson" => new PoissonDistribution(p[0]),
        "Geometric" => new GeometricDistribution(p[0]),
        "NegativeBinomial" => new NegativeBinomialDistribution(p[0], p[1]),
        "DiscreteUniform" => new DiscreteUniformDistribution(checked((long)p[0]), checked((long)p[1])),
        "Categorical" => new CategoricalDistribution(p),
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, null),
    };

    /// <summary>テストの表示名に使う「分布名(パラメータ)」</summary>
    internal static string Describe(string name, double[] p) =>
        $"{name}({string.Join(", ", p.Select(v => v.ToString("R", System.Globalization.CultureInfo.InvariantCulture)))})";
}

internal static class DistributionAssert
{
    /// <summary>相対誤差で比べる。許容誤差が0か期待値が有限でなければ完全な一致を求める</summary>
    internal static void Close(double expected, double actual, double tolerance, string context)
    {
        if (tolerance == 0 || !double.IsFinite(expected))
        {
            Assert.True(expected.Equals(actual), $"{context}: expected {expected:R}, actual {actual:R}");
            return;
        }

        double error = Math.Abs(actual - expected) / Math.Abs(expected);
        Assert.True(error <= tolerance,
            $"{context}: expected {expected:R}, actual {actual:R}, relative error {error:E2}, tolerance {tolerance:E2}");
    }
}
