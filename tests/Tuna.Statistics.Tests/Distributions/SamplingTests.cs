using Tuna.RandomSource;
using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Statistics;

// 乱数生成の検証。シードを固定するので結果は決定的で、偶然の失敗は起きない。
// (1) 自前の累積分布関数に対する適合度（連続は Kolmogorov–Smirnov、離散は χ²）を有意水準 1e-4 で、
// (2) 平均を標準誤差の5倍以内で、(3) 一括生成が単発生成の繰り返しと一致することを確かめる。
// 生成法を変えて同じシードの列が変わっても (1)(2) は成り立つ。
public class SamplingTests
{
    private const int Count = 200_000;
    private const double SignificanceLevel = 1e-4;

    private static IContinuousDistribution Continuous(string name) => name switch
    {
        "Normal" => new NormalDistribution(1.5, 2),
        "LogNormal" => new LogNormalDistribution(0.3, 0.8),
        "Exponential" => new ExponentialDistribution(2.5),
        "Uniform" => new UniformDistribution(-1, 3),
        "GammaSmallShape" => new GammaDistribution(0.3, 2),
        "GammaUnitShape" => new GammaDistribution(1, 1),
        "Gamma" => new GammaDistribution(7.5, 0.5),
        "ChiSquared" => new ChiSquaredDistribution(3),
        "BetaSmall" => new BetaDistribution(0.4, 0.7),
        "Beta" => new BetaDistribution(2, 5),
        "BetaShapesBelowOne" => new BetaDistribution(0.2, 0.3),
        "StudentTCauchy" => new StudentTDistribution(1),
        "StudentT" => new StudentTDistribution(3.5, 1, 2),
        "StudentTSmall" => new StudentTDistribution(0.2),
        "StudentTInfinite" => new StudentTDistribution(double.PositiveInfinity, -1, 3),
        "Cauchy" => new CauchyDistribution(1, 2),
        "Laplace" => new LaplaceDistribution(-1, 0.5),
        "Logistic" => new LogisticDistribution(2, 1.5),
        "WeibullSmallShape" => new WeibullDistribution(0.7, 2),
        "Weibull" => new WeibullDistribution(5, 1),
        "Gumbel" => new GumbelDistribution(2, 1.5),
        "InverseGammaSmallShape" => new InverseGammaDistribution(0.6, 1),
        "InverseGamma" => new InverseGammaDistribution(4.5, 2),
        "FSmall" => new FDistribution(0.7, 1.5),
        "F" => new FDistribution(5, 12),
        "LocationScaleBeta" => new LocationScaleDistribution<BetaDistribution>(new BetaDistribution(2, 5), -3, 4),
        "TruncatedNormalTail" => new TruncatedDistribution<NormalDistribution>(new NormalDistribution(1, 2), 5, double.PositiveInfinity),
        "TruncatedGamma" => new TruncatedDistribution<GammaDistribution>(new GammaDistribution(0.5, 1), 0, 2),
        "TruncatedCauchy" => new TruncatedDistribution<CauchyDistribution>(new CauchyDistribution(0, 1), -1, 30),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    private static IDiscreteDistribution Discrete(string name) => name switch
    {
        "Bernoulli" => new BernoulliDistribution(0.3),
        "BinomialInversion" => new BinomialDistribution(20, 0.3),
        "Binomial" => new BinomialDistribution(1000, 0.4),
        "BinomialHighProbability" => new BinomialDistribution(1000, 0.9),
        "BinomialLarge" => new BinomialDistribution(1L << 40, 0.25),
        "PoissonInversion" => new PoissonDistribution(3.5),
        "Poisson" => new PoissonDistribution(250),
        "PoissonLarge" => new PoissonDistribution(1e7),
        "Geometric" => new GeometricDistribution(0.2),
        "NegativeBinomial" => new NegativeBinomialDistribution(3.5, 0.3),
        "NegativeBinomialOverdispersed" => new NegativeBinomialDistribution(0.5, 0.01),
        "DiscreteUniform" => new DiscreteUniformDistribution(-3, 7),
        "Categorical" => new CategoricalDistribution(0.1, 0, 0.5, 0.4),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    public static TheoryData<string> ContinuousNames =>
    [
        "Normal", "LogNormal", "Exponential", "Uniform", "GammaSmallShape", "GammaUnitShape", "Gamma", "ChiSquared",
        "BetaSmall", "Beta", "BetaShapesBelowOne", "StudentTCauchy", "StudentT", "StudentTSmall", "StudentTInfinite",
        "Cauchy", "Laplace", "Logistic", "WeibullSmallShape", "Weibull", "Gumbel", "InverseGammaSmallShape", "InverseGamma",
        "FSmall", "F", "LocationScaleBeta", "TruncatedNormalTail", "TruncatedGamma", "TruncatedCauchy",
    ];

    public static TheoryData<string> DiscreteNames =>
    [
        "Bernoulli", "BinomialInversion", "Binomial", "BinomialHighProbability", "BinomialLarge",
        "PoissonInversion", "Poisson", "PoissonLarge", "Geometric", "NegativeBinomial", "NegativeBinomialOverdispersed",
        "DiscreteUniform", "Categorical",
    ];

    [Theory]
    [MemberData(nameof(ContinuousNames))]
    public void ContinuousSamplesFollowTheDistribution(string name)
    {
        IContinuousDistribution distribution = Continuous(name);
        var random = new Pcg64DxsmRandom(20261012, StableStream(name));
        double[] samples = new double[Count];
        distribution.Sample(random, samples);

        // Kolmogorov–Smirnov: D = sup |Fₙ(x) - F(x)|
        Array.Sort(samples);
        double d = 0;
        for (int i = 0; i < Count; i++)
        {
            double f = distribution.Cdf(samples[i]);
            d = Math.Max(d, Math.Max(f - (double)i / Count, (i + 1.0) / Count - f));
        }

        double pValue = XMath.Special.KolmogorovSf(Math.Sqrt(Count) * d);
        Assert.True(pValue > SignificanceLevel, $"KS D = {d}, p = {pValue}");

        AssertMean(samples, distribution.Mean, distribution.Variance);
    }

    [Theory]
    [MemberData(nameof(DiscreteNames))]
    public void DiscreteSamplesFollowTheDistribution(string name)
    {
        IDiscreteDistribution distribution = Discrete(name);
        var random = new Pcg64DxsmRandom(20261012, StableStream(name));
        long[] samples = new long[Count];
        distribution.Sample(random, samples);
        Assert.All(samples, k => Assert.True(k >= distribution.SupportMinimum && k <= distribution.SupportMaximum));

        // χ²: 期待度数が5以上になるように両裾をまとめた区間 [lower, upper] ごとに数える
        var counts = new Dictionary<long, long>();
        foreach (long k in samples)
        {
            counts[k] = counts.GetValueOrDefault(k) + 1;
        }

        var bins = new List<(long Upper, double Probability)>();
        long start = distribution.InverseCdf(1e-6);
        long end = distribution.InverseSf(1e-6);
        double accumulated = distribution.Cdf(start - 1);
        for (long k = start; k <= end; k++)
        {
            accumulated += distribution.Pmf(k);
            if (accumulated * Count >= 5 && distribution.Sf(k) * Count >= 5)
            {
                bins.Add((k, accumulated));
                accumulated = 0;
            }
        }

        double chiSquared = 0;
        long previousUpper = long.MinValue;
        double used = 0;
        for (int b = 0; b <= bins.Count; b++)
        {
            bool last = b == bins.Count;
            long upper = last ? long.MaxValue : bins[b].Upper;
            double probability = last ? 1 - used : bins[b].Probability;
            used += last ? 0 : probability;
            long observed = counts.Where(pair => pair.Key > previousUpper && pair.Key <= upper).Sum(pair => pair.Value);
            double expected = probability * Count;
            chiSquared += (observed - expected) * (observed - expected) / expected;
            previousUpper = upper;
        }

        int degreesOfFreedom = bins.Count;
        double pValue = XMath.Special.RegularizedGammaQ(0.5 * degreesOfFreedom, 0.5 * chiSquared);
        Assert.True(pValue > SignificanceLevel, $"χ² = {chiSquared}, df = {degreesOfFreedom}, p = {pValue}");

        AssertMean(samples.Select(k => (double)k).ToArray(), distribution.Mean, distribution.Variance);
    }

    private static void AssertMean(double[] samples, double mean, double variance)
    {
        if (!double.IsFinite(mean) || !double.IsFinite(variance))
        {
            return;
        }

        double sampleMean = samples.Average();
        double tolerance = 5 * Math.Sqrt(variance / samples.Length);
        Assert.True(Math.Abs(sampleMean - mean) <= tolerance, $"mean {sampleMean}, expected {mean} ± {tolerance}");
    }

    // string.GetHashCode はプロセスごとに変わるので、名前から決まった値を作る
    private static ulong StableStream(string name)
    {
        ulong hash = 14695981039346656037;
        foreach (char c in name)
        {
            hash = (hash ^ c) * 1099511628211;
        }

        return hash;
    }

    [Theory]
    [MemberData(nameof(ContinuousNames))]
    public void ContinuousBulkSamplingMatchesRepeatedSingleSamples(string name)
    {
        IContinuousDistribution distribution = Continuous(name);
        var single = new Philox4x32Random(7, 3);
        var bulk = new Philox4x32Random(7, 3);
        double[] values = new double[257];
        distribution.Sample(bulk, values);
        foreach (double value in values)
        {
            Assert.Equal(distribution.Sample(single), value);
        }

        Assert.Equal(single.NextUInt64(), bulk.NextUInt64());
    }

    [Theory]
    [MemberData(nameof(DiscreteNames))]
    public void DiscreteBulkSamplingMatchesRepeatedSingleSamples(string name)
    {
        IDiscreteDistribution distribution = Discrete(name);
        var single = new Philox4x32Random(7, 3);
        var bulk = new Philox4x32Random(7, 3);
        long[] values = new long[257];
        distribution.Sample(bulk, values);
        foreach (long value in values)
        {
            Assert.Equal(distribution.Sample(single), value);
        }

        Assert.Equal(single.NextUInt64(), bulk.NextUInt64());
    }

    [Fact]
    public void SamplingIsReproducibleForTheSameSeed()
    {
        var first = new Pcg64DxsmRandom(42);
        var second = new Pcg64DxsmRandom(42);
        var gamma = new GammaDistribution(2.5, 1.5);
        var poisson = new PoissonDistribution(40);
        for (int i = 0; i < 1000; i++)
        {
            Assert.Equal(gamma.Sample(first), gamma.Sample(second));
            Assert.Equal(poisson.Sample(first), poisson.Sample(second));
        }
    }

    [Fact]
    public void LargePoissonAndBinomialMeansAreAccurate()
    {
        // 平均が大きいと鞍点展開を使わない受理判定は桁落ちで崩れる。平均を標準誤差の5倍以内で確かめる
        var random = new Pcg64DxsmRandom(99);
        var poisson = new PoissonDistribution(1e15);
        double[] values = new double[20000];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = poisson.Sample(random);
        }

        AssertMean(values, poisson.Mean, poisson.Variance);

        var binomial = new BinomialDistribution(BinomialDistribution.MaxTrials, 0.3);
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = binomial.Sample(random);
        }

        AssertMean(values, binomial.Mean, binomial.Variance);
    }

    [Fact]
    public void GammaWithTinyShapeIsSampledInLogSpace()
    {
        // 形状 1e-3 のガンマ乱数は大半が 1e-300 未満になる。ベータ分布は対数で比をとるので 0/0 にならず、
        // 0 か 1 の近く（丸めると 0 か 1）に集まる。0 か 1 に丸められた値があるので KS 検定は使えず、x ≤ 1/2 の割合を二項分布の5σで確かめる
        var random = new Pcg64DxsmRandom(5);
        var beta = new BetaDistribution(1e-3, 2e-3);
        const int count = 100_000;
        int lower = 0;
        for (int i = 0; i < count; i++)
        {
            double x = beta.Sample(random);
            Assert.InRange(x, 0, 1);
            lower += x <= 0.5 ? 1 : 0;
        }

        double p = beta.Cdf(0.5);
        Assert.True(Math.Abs(lower - count * p) <= 5 * Math.Sqrt(count * p * (1 - p)), $"{lower} of {count}, expected p = {p}");

        var gamma = new GammaDistribution(1e-3, 1e300);
        for (int i = 0; i < 1000; i++)
        {
            Assert.True(gamma.Sample(random) >= 0);
        }
    }

    /// <summary>決まった64ビット語の列を返す乱数源</summary>
    private sealed class ScriptedSource(params ulong[] words) : IRandomSource
    {
        private int _index;

        public ulong NextUInt64() => words[_index++];
        public uint NextUInt32() => throw new NotSupportedException();
        public uint NextUInt32(uint exclusiveMax) => throw new NotSupportedException();
        public ulong NextUInt64(ulong exclusiveMax) => throw new NotSupportedException();
        public int NextInt32() => throw new NotSupportedException();
        public int NextInt32(int exclusiveMax) => throw new NotSupportedException();
        public int NextInt32(int inclusiveMin, int exclusiveMax) => throw new NotSupportedException();
        public long NextInt64() => throw new NotSupportedException();
        public long NextInt64(long exclusiveMax) => throw new NotSupportedException();
        public long NextInt64(long inclusiveMin, long exclusiveMax) => throw new NotSupportedException();
        public double NextDouble() => throw new NotSupportedException();
        public double NextDoubleOpen() => throw new NotSupportedException();
        public float NextSingle() => throw new NotSupportedException();
        public void Fill(Span<uint> destination) => throw new NotSupportedException();
        public void Fill(Span<ulong> destination) => throw new NotSupportedException();
        public void Fill(Span<double> destination) => throw new NotSupportedException();
        public void Fill(Span<float> destination) => throw new NotSupportedException();
        public void Fill(Span<byte> destination) => throw new NotSupportedException();
    }

    [Theory]
    // p = 1/2 の2進展開は 0.1000…: 先頭64ビットが 2⁶³ 未満なら U < p
    [InlineData(0.5, new ulong[] { 0x7FFF_FFFF_FFFF_FFFF }, 1)]
    [InlineData(0.5, new ulong[] { 0x8000_0000_0000_0000 }, 0)]
    // p = 1/3 の倍精度値は 0.0101…01（53ビット）で、先頭64ビット 0x5555_5555_5555_5400 で展開が終わる
    [InlineData(1.0 / 3, new ulong[] { 0x5555_5555_5555_53FF }, 1)]
    [InlineData(1.0 / 3, new ulong[] { 0x5555_5555_5555_5400 }, 0)]
    // p = 2⁻²⁰/3 の展開は64ビットを超える（0x0000_0555_5555_5555, 0x5540_0000_0000_0000）。先頭が等しければ次の語で決める
    [InlineData(3.178914388020833e-07, new ulong[] { 0x0000_0555_5555_5555, 0x553F_FFFF_FFFF_FFFF }, 1)]
    [InlineData(3.178914388020833e-07, new ulong[] { 0x0000_0555_5555_5555, 0x5540_0000_0000_0000 }, 0)]
    [InlineData(3.178914388020833e-07, new ulong[] { 0x0000_0555_5555_5554 }, 1)]
    [InlineData(3.178914388020833e-07, new ulong[] { 0x0000_0555_5555_5556 }, 0)]
    // p = 2⁻¹⁰⁰: 先頭64ビットが0でなければ偽
    [InlineData(7.8886090522101181e-31, new ulong[] { 1 }, 0)]
    public void BernoulliComparesBinaryExpansionsExactly(double p, ulong[] words, long expected)
    {
        Assert.Equal(expected, new BernoulliDistribution(p).Sample(new ScriptedSource(words)));
    }
}
