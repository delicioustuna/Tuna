using Tuna.RandomSource;
using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Statistics;

// 離散分布の規約（パラメータの検証、NaN・台の外・逆関数の端点）と、退化した場合・分布どうしの関係。値の精度は DiscreteReferenceTests で確かめる。
public class DiscreteDistributionTests
{
    private static IDiscreteDistribution Create(string name) => name switch
    {
        "Bernoulli" => new BernoulliDistribution(0.3),
        "Binomial" => new BinomialDistribution(1000, 0.4),
        "BinomialHighProbability" => new BinomialDistribution(50, 0.97),
        "Poisson" => new PoissonDistribution(7),
        "PoissonLarge" => new PoissonDistribution(1e12),
        "Geometric" => new GeometricDistribution(0.2),
        "NegativeBinomial" => new NegativeBinomialDistribution(3.5, 0.3),
        "DiscreteUniform" => new DiscreteUniformDistribution(-3, 7),
        "Categorical" => new CategoricalDistribution(0, 0.1, 0, 0.5, 0.4, 0),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    public static TheoryData<string> Names =>
    [
        "Bernoulli", "Binomial", "BinomialHighProbability", "Poisson", "PoissonLarge", "Geometric", "NegativeBinomial",
        "DiscreteUniform", "Categorical",
    ];

    [Theory]
    [MemberData(nameof(Names))]
    public void InvalidProbabilitiesThrowInInverses(string name)
    {
        IDiscreteDistribution d = Create(name);
        foreach (double p in new[] { double.NaN, -0.1, 1.0000000000000002, double.PositiveInfinity })
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => d.InverseCdf(p));
            Assert.Throws<ArgumentOutOfRangeException>(() => d.InverseSf(p));
        }

        foreach (double logP in new[] { double.NaN, double.Epsilon, double.PositiveInfinity })
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => d.InverseLogCdf(logP));
            Assert.Throws<ArgumentOutOfRangeException>(() => d.InverseLogSf(logP));
        }
    }

    [Theory]
    [MemberData(nameof(Names))]
    public void InversesReturnTheSupportAtTheEndpoints(string name)
    {
        IDiscreteDistribution d = Create(name);
        Assert.Equal(d.SupportMinimum, d.InverseCdf(0));
        Assert.Equal(d.SupportMaximum, d.InverseCdf(1));
        Assert.Equal(d.SupportMinimum, d.InverseSf(1));
        Assert.Equal(d.SupportMaximum, d.InverseSf(0));
        Assert.Equal(d.SupportMinimum, d.InverseLogCdf(double.NegativeInfinity));
        Assert.Equal(d.SupportMaximum, d.InverseLogCdf(0));
        Assert.Equal(d.SupportMinimum, d.InverseLogSf(0));
        Assert.Equal(d.SupportMaximum, d.InverseLogSf(double.NegativeInfinity));
    }

    [Theory]
    [MemberData(nameof(Names))]
    public void ProbabilitiesBeyondTheSupportAreExact(string name)
    {
        IDiscreteDistribution d = Create(name);
        foreach (long below in new[] { long.MinValue, d.SupportMinimum - 1 })
        {
            Assert.Equal(0.0, d.Pmf(below));
            Assert.Equal(double.NegativeInfinity, d.LogPmf(below));
            Assert.Equal(0.0, d.Cdf(below));
            Assert.Equal(1.0, d.Sf(below));
            Assert.Equal(double.NegativeInfinity, d.LogCdf(below));
            Assert.Equal(0.0, d.LogSf(below));
        }

        // 上に有界でない分布でも P(X > long.MaxValue) は double では 0（対数は有限の負の値になりうる）
        Assert.Equal(1.0, d.Cdf(long.MaxValue));
        Assert.Equal(0.0, d.Sf(long.MaxValue));
        Assert.Equal(0.0, d.LogCdf(long.MaxValue));
        if (d.SupportMaximum < long.MaxValue)
        {
            Assert.Equal(0.0, d.Pmf(d.SupportMaximum + 1));
            Assert.Equal(double.NegativeInfinity, d.LogPmf(d.SupportMaximum + 1));
            Assert.Equal(1.0, d.Cdf(d.SupportMaximum));
            Assert.Equal(0.0, d.Sf(d.SupportMaximum));
            Assert.Equal(double.NegativeInfinity, d.LogSf(d.SupportMaximum));
        }
    }

    [Theory]
    [MemberData(nameof(Names))]
    public void CdfIsTheSumOfThePmfAndInversesAreConsistent(string name)
    {
        IDiscreteDistribution d = Create(name);
        long start = d.InverseCdf(1e-12);
        long end = d.InverseSf(1e-12);
        if (end - start > 200)
        {
            start = d.InverseCdf(0.4);
            end = start + 200;
        }

        double cumulative = d.Cdf(start - 1);
        for (long k = start; k <= end; k++)
        {
            cumulative += d.Pmf(k);
            Assert.Equal(d.Cdf(k), cumulative, 1e-12);
            Assert.Equal(1.0, d.Cdf(k) + d.Sf(k), 1e-15);

            // Cdf(k) の少し内側の p では k が Cdf(k) ≥ p を満たす最小の値（p = Cdf(k) ちょうどでは、
            // double に丸めた p と真の Cdf(k) の大小で答えが k と k + 1 に分かれるので使わない）
            double p = d.Cdf(k);
            if (p > 0 && p < 1 && d.Pmf(k) > 1e-9 * p)
            {
                Assert.Equal(k, d.InverseCdf(p * (1 - 1e-12)));
                Assert.Equal(k, d.InverseLogCdf(d.LogCdf(k) - 1e-12));
            }

            double q = d.Sf(k);
            if (q > 0 && q < 1 && d.Pmf(k) > 1e-9 * q)
            {
                Assert.Equal(k, d.InverseSf(q * (1 + 1e-12)));
                Assert.Equal(k, d.InverseLogSf(d.LogSf(k) + 1e-12));
            }
        }

        Assert.Equal(d.InverseCdf(0.5), d.Median);
    }

    [Theory]
    [MemberData(nameof(Names))]
    public void SampleRejectsNullRandomSource(string name)
    {
        IDiscreteDistribution d = Create(name);
        Assert.Throws<ArgumentNullException>(() => d.Sample(null!));
        Assert.Throws<ArgumentNullException>(() => d.Sample(null!, new long[1]));
    }

    [Fact]
    public void ConstructorsValidateParameters()
    {
        foreach (double p in new[] { -0.1, 1.1, double.NaN, double.PositiveInfinity })
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new BernoulliDistribution(p));
            Assert.Throws<ArgumentOutOfRangeException>(() => new BinomialDistribution(10, p));
            Assert.Throws<ArgumentOutOfRangeException>(() => new GeometricDistribution(p));
            Assert.Throws<ArgumentOutOfRangeException>(() => new NegativeBinomialDistribution(2, p));
        }

        // 幾何・負の二項は成功確率 0 を許さない（台が空になる）
        Assert.Throws<ArgumentOutOfRangeException>(() => new GeometricDistribution(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NegativeBinomialDistribution(2, 0));

        Assert.Throws<ArgumentOutOfRangeException>(() => new BinomialDistribution(-1, 0.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BinomialDistribution(BinomialDistribution.MaxTrials + 1, 0.5));
        _ = new BinomialDistribution(BinomialDistribution.MaxTrials, 0.5);

        foreach (double rate in new[] { -1, double.NaN, double.PositiveInfinity, PoissonDistribution.MaxRate * 1.0000001 })
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new PoissonDistribution(rate));
        }

        foreach (double r in new[] { 0, -1, double.NaN, double.PositiveInfinity, NegativeBinomialDistribution.MaxSuccesses * 2 })
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new NegativeBinomialDistribution(r, 0.5));
        }

        Assert.Throws<ArgumentOutOfRangeException>(() => NegativeBinomialDistribution.FromMean(-1, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => NegativeBinomialDistribution.FromMean(1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DiscreteUniformDistribution(1, 0));

        Assert.Throws<ArgumentException>(() => new CategoricalDistribution(ReadOnlySpan<double>.Empty));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CategoricalDistribution(1, -0.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CategoricalDistribution(1, double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CategoricalDistribution(1, double.PositiveInfinity));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CategoricalDistribution(0, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CategoricalDistribution(double.MaxValue, double.MaxValue));
        Assert.Throws<ArgumentException>(() => CategoricalDistribution.FromLogWeights(ReadOnlySpan<double>.Empty));
        Assert.Throws<ArgumentOutOfRangeException>(() => CategoricalDistribution.FromLogWeights([0, double.NaN]));
        Assert.Throws<ArgumentOutOfRangeException>(() => CategoricalDistribution.FromLogWeights([0, double.PositiveInfinity]));
        Assert.Throws<ArgumentOutOfRangeException>(() => CategoricalDistribution.FromLogWeights([double.NegativeInfinity]));
    }

    [Fact]
    public void DegenerateDistributionsPutAllMassOnOnePoint()
    {
        var random = new Pcg64DxsmRandom(1);
        IDiscreteDistribution[] cases =
        [
            new BernoulliDistribution(0), new BernoulliDistribution(1), new BinomialDistribution(0, 0.3),
            new BinomialDistribution(12, 0), new BinomialDistribution(12, 1), new PoissonDistribution(0),
            new GeometricDistribution(1), new NegativeBinomialDistribution(2.5, 1), new DiscreteUniformDistribution(-4, -4),
            new CategoricalDistribution(0, 0, 3, 0),
        ];
        long[] points = [0, 1, 0, 0, 12, 0, 0, 0, -4, 2];
        for (int i = 0; i < cases.Length; i++)
        {
            IDiscreteDistribution d = cases[i];
            long point = points[i];
            Assert.Equal(1.0, d.Pmf(point));
            Assert.Equal(0.0, d.LogPmf(point));
            Assert.Equal(0.0, d.Pmf(point + 1));
            Assert.Equal(1.0, d.Cdf(point));
            Assert.Equal(0.0, d.Sf(point));
            Assert.Equal(point, d.SupportMinimum);
            Assert.Equal(point, d.SupportMaximum);
            Assert.Equal(point, d.Median);
            Assert.Equal(point, d.InverseCdf(0.3));
            Assert.Equal(point, d.InverseSf(0.3));
            Assert.Equal((double)point, d.Mean);
            Assert.Equal(0.0, d.Variance);
            Assert.Equal(0.0, d.Entropy);
            for (int j = 0; j < 20; j++)
            {
                Assert.Equal(point, d.Sample(random));
            }
        }
    }

    [Fact]
    public void DiscreteUniformCoversTheWholeLongRange()
    {
        var d = new DiscreteUniformDistribution(long.MinValue, long.MaxValue);
        Assert.Equal(-0.5, d.Mean);
        Assert.Equal(Math.Pow(2, -64), d.Pmf(0));
        Assert.Equal(0.5, d.Cdf(-1));
        Assert.Equal(0.5, d.Sf(-1));

        // 個数が 2⁵³ を超えると Cdf は double の丸めで 2⁶⁴·2⁻⁵⁴ = 1024 個ずつ同じ値になる。逆関数は丸めた Cdf に対して最小の k を返す
        long median = d.Median;
        Assert.InRange(median, -1024, -1);
        Assert.True(d.Cdf(median) >= 0.5);
        Assert.True(d.Cdf(median - 1) < 0.5);
        Assert.Equal(64 * Math.Log(2), d.Entropy, 1e-13);
        Assert.Equal(long.MinValue, d.InverseCdf(0));
        Assert.Equal(long.MaxValue, d.InverseCdf(1));

        var random = new Philox4x32Random(11);
        long negative = 0;
        for (int i = 0; i < 1000; i++)
        {
            if (d.Sample(random) < 0)
            {
                negative++;
            }
        }

        Assert.InRange(negative, 400, 600);
    }

    [Fact]
    public void CategoricalNormalizesWeightsAndAcceptsLogWeights()
    {
        var d = new CategoricalDistribution(2, 0, 6);
        Assert.Equal(3, d.CategoryCount);
        Assert.Equal([0.25, 0, 0.75], d.Probabilities.ToArray());
        Assert.Equal(0, d.SupportMinimum);
        Assert.Equal(2, d.SupportMaximum);
        Assert.Equal(2, d.Median);
        Assert.Equal(1.5, d.Mean);

        // 対数の重みは最大値を引いてから戻すので、exp が溢れる大きさでもよい
        // （1000 + ln 2 の丸め誤差 ulp(1000)/2 ≈ 6e-14 がそのまま確率の相対誤差になる）
        var fromLog = CategoricalDistribution.FromLogWeights([1000 + Math.Log(2), double.NegativeInfinity, 1000 + Math.Log(6)]);
        Assert.Equal(0.25, fromLog.Probabilities[0], 1e-13);
        Assert.Equal(0.0, fromLog.Probabilities[1]);
        Assert.Equal(0.75, fromLog.Probabilities[2], 1e-13);

        // 重み 0 のカテゴリは出ない
        var random = new Pcg64DxsmRandom(3);
        var counts = new int[3];
        for (int i = 0; i < 10_000; i++)
        {
            counts[d.Sample(random)]++;
        }

        Assert.Equal(0, counts[1]);
        Assert.InRange(counts[0], 2_300, 2_700);
    }

    [Fact]
    public void CategoricalHigherMomentsStayFiniteForTinyVariance()
    {
        // 分散 2e-308 では σ³ と σ⁴ がアンダーフローする。歪度 0、超過尖度 1/σ² - 3 = 5e307
        var d = new CategoricalDistribution(1e-308, 1, 1e-308);
        Assert.Equal(2e-308, d.Variance, 1e-322);
        Assert.Equal(0.0, d.Skewness);
        Assert.Equal(5e307, d.ExcessKurtosis, 5e307 * 1e-14);
    }

    [Fact]
    public void GeometricIsNegativeBinomialWithOneSuccess()
    {
        var geometric = new GeometricDistribution(0.2);
        var negativeBinomial = new NegativeBinomialDistribution(1, 0.2);
        foreach (long k in new long[] { 0, 1, 7, 100, 3000 })
        {
            Assert.Equal(geometric.Pmf(k), negativeBinomial.Pmf(k), geometric.Pmf(k) * 1e-14);
            Assert.Equal(geometric.LogSf(k), negativeBinomial.LogSf(k), Math.Abs(geometric.LogSf(k)) * 1e-14);
        }

        Assert.Equal(geometric.Mean, negativeBinomial.Mean, 1e-14);
        Assert.Equal(geometric.Entropy, negativeBinomial.Entropy, 1e-14);
        Assert.Equal(geometric.InverseCdf(0.9), negativeBinomial.InverseCdf(0.9));
    }

    [Fact]
    public void NegativeBinomialFromMean()
    {
        var d = NegativeBinomialDistribution.FromMean(12, 3);
        Assert.Equal(3.0, d.Successes);
        Assert.Equal(0.2, d.Probability, 1e-16);
        Assert.Equal(12.0, d.Mean, 1e-13);

        var zero = NegativeBinomialDistribution.FromMean(0, 3);
        Assert.Equal(1.0, zero.Probability);
        Assert.Equal(1.0, zero.Pmf(0));
    }

    [Fact]
    public void NegativeBinomialSamplesSaturateWhenTheMixingRateOverflows()
    {
        // p = 1e-320 ではガンマ・ポアソン混合の平均 G(1 - p)/p が +∞ に溢れる。正規近似の ∞ + √∞·z が NaN になって 0 を返さず、long.MaxValue で飽和する
        var d = new NegativeBinomialDistribution(1e15, 1e-320);
        long[] samples = new long[64];
        d.Sample(new Pcg64DxsmRandom(1, 2), samples);
        Assert.All(samples, k => Assert.Equal(long.MaxValue, k));
    }

    [Theory]
    [InlineData(1e-320, 1e-310, 9.99988867182686e-11, 9.99988867182689e299, 9.99994433575852e149, 2.0000111329102726e160, double.PositiveInfinity)]
    [InlineData(1e-20, 1e-160, 1e140, 9.999999999999999e299, 1e150, 2e10, 6e20)]
    [InlineData(1.0, 1e-160, 1e160, double.PositiveInfinity, 1e160, 2.0, 6.0)]
    [InlineData(1e-310, 0.9999999999999999, 0.0, 0.0, 1.0536712127723492e-163, 9.490626562425171e162, double.PositiveInfinity)]
    [InlineData(3.5, 0.25, 10.5, 42.0, 6.48074069840786, 1.0801234497346435, 1.7380952380952381)]
    public void NegativeBinomialMomentsAvoidIntermediateOverflowAndUnderflow(
        double r, double p, double mean, double variance, double standardDeviation, double skewness, double excessKurtosis)
    {
        // p ≈ 10⁻³¹⁰ では p² が0に、p = 10⁻¹⁶⁰ では非正規化数に落ちるが、r が小さければ分散 r(1 - p)/p² は表せる。
        // p = 1 - 2⁻⁵³ と r = 10⁻³¹⁰ では r(1 - p) がアンダーフローするが、標準偏差 √(r(1 - p)) は表せる。
        // 期待値は入力の double の値（1 - p も double の値）から mpmath で求めた値で、表せない大きさのものは ±∞ と0
        var d = new NegativeBinomialDistribution(r, p);
        void Check(double expected, double actual, string name)
        {
            if (expected == 0)
            {
                Assert.Equal(0.0, actual);
            }
            else
            {
                DistributionAssert.Close(expected, actual, 1e-14, name);
            }
        }

        Check(mean, d.Mean, "Mean");
        Check(variance, d.Variance, "Variance");
        Check(standardDeviation, d.StandardDeviation, "StandardDeviation");
        Check(skewness, d.Skewness, "Skewness");
        Check(excessKurtosis, d.ExcessKurtosis, "ExcessKurtosis");
    }

    [Fact]
    public void GeometricVarianceKeepsItsDigitsWhenTheSquareIsSubnormal()
    {
        // p² ≈ 5.6·10⁻³⁰⁹ は非正規化数で、2乗してから割ると 4 ulp ずれる。期待値は 1/p²（1 - p は1に丸まる）を有理数で求めて丸めた値
        double p = 7.458340731200208e-155;
        DistributionAssert.Close(1.7976931348623151e308, new GeometricDistribution(p).Variance, 2.3e-16, "Geometric");
        DistributionAssert.Close(1.7976931348623151e308, new NegativeBinomialDistribution(1, p).Variance, 2.3e-16, "NegativeBinomial");
    }

    [Theory]
    [InlineData(1e-310, 1L, -714.4945260087142)]
    [InlineData(1e-310, 1000L, -1413.8563146670815)]
    [InlineData(5e-324, 2L, -746.5195134630611)]
    public void NegativeBinomialLogPmfKeepsItsDigitsForSubnormalSuccesses(double r, long k, double expected)
    {
        // r が非正規化数では S(r) の 1/r がオーバーフローし、r/(r + k) も非正規化数に落ちて桁落ちする（r = 5·10⁻³²⁴ では 0 になる）。p = 1/2、期待値は mpmath で算出。
        DistributionAssert.Close(expected, new NegativeBinomialDistribution(r, 0.5).LogPmf(k), 1e-15, $"r = {r}, k = {k}");
    }

    [Theory]
    [InlineData(2.5, 1e-310, 715.5313267376592)]
    [InlineData(0.5, 5e-324, 744.5306818512953)]
    [InlineData(1e-320, 1e-310, 7.8099951e-315)]
    [InlineData(1e-320, 1e-5, 8.555999e-317)]
    [InlineData(1e-310, 0.5, 4.963286043759591e-308)]
    [InlineData(0.5, 0.9999999999999999, 2.1332906413727453e-15)]
    [InlineData(1.0, 1e-310, 714.8013788281542)]
    [InlineData(1e6, 1e-160, 376.7403083579007)]
    public void NegativeBinomialEntropyForExtremeParameters(double r, double p, double expected)
    {
        // q/p が溢れる p、積分の端の t が 0・+∞ になる p・r、確率の和の項がアンダーフローする r、分散が溢れる正規分布に近い場合（r = 10⁶）。
        // 期待値は入力の double の値から、積分表示を mpmath の数値積分で求めた値（中程度のパラメータで直接の和と 10⁻³⁸ まで一致）。
        // 結果が非正規化数のときは、その間隔の数個分まで許す
        var d = new NegativeBinomialDistribution(r, p);
        var task = Task.Run(() => d.Entropy);
        Assert.True(task.Wait(TimeSpan.FromSeconds(30)), "Entropy did not finish");
        DistributionAssert.Close(expected, task.Result, Math.Max(1e-14, 4 * double.Epsilon / expected), $"r = {r}, p = {p}");
    }

    [Fact]
    public void GeometricEntropyStaysFiniteWhenTheOddsOverflow()
    {
        // (1 - p)/p が溢れる p でも、エントロピー (-(1 - p) ln(1 - p) - p ln p)/p ≈ 1 - ln p は表せる（期待値は mpmath で求めた値）
        DistributionAssert.Close(714.8013788281542, new GeometricDistribution(1e-310).Entropy, 1e-15, "Entropy");
    }

    [Fact]
    public void BernoulliIsBinomialWithOneTrial()
    {
        var bernoulli = new BernoulliDistribution(0.3);
        var binomial = new BinomialDistribution(1, 0.3);
        for (long k = -1; k <= 2; k++)
        {
            Assert.Equal(binomial.Pmf(k), bernoulli.Pmf(k), 1e-16);
            Assert.Equal(binomial.Cdf(k), bernoulli.Cdf(k), 1e-16);
        }

        Assert.Equal(binomial.Variance, bernoulli.Variance, 1e-16);
        Assert.Equal(binomial.Skewness, bernoulli.Skewness, 1e-15);
        Assert.Equal(binomial.Entropy, bernoulli.Entropy, 1e-15);
    }
}
