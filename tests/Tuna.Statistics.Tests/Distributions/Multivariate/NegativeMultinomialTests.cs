using Tuna.RandomSource;
using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Statistics;

// 負の多項分布: 負の二項分布との一致（失敗が1種類の場合と周辺分布）、すべての成分が0の点、全体の確率の和とモーメント、
// 確率0の成分、乱数の飽和、パラメータの検証。
public class NegativeMultinomialTests
{
    [Theory]
    [InlineData(3.0, 0.4)]
    [InlineData(0.5, 0.9)]
    [InlineData(1e6, 0.3)]
    [InlineData(2.5, 1e-3)]
    public void OneFailureKindIsNegativeBinomial(double r, double p0)
    {
        var negativeBinomial = new NegativeBinomialDistribution(r, p0);
        var d = new NegativeMultinomialDistribution(r, p0, 1 - p0);
        double mean = negativeBinomial.Mean;
        foreach (long k in new[] { 0L, 1L, 10L, (long)mean, (long)(2 * mean) + 3, 100000L })
        {
            DistributionAssert.Close(negativeBinomial.LogPmf(k), d.LogPmf(new[] { k }), 1e-13, $"k = {k}");
        }
    }

    [Fact]
    public void MarginalsAreNegativeBinomial()
    {
        // Xᵢ の周辺分布は成功確率 p₀/(p₀ + pᵢ) の負の二項分布。もう一方の成分について和をとる（裾は 0.3ᵏ 程度で減る）
        double r = 2.5;
        double p0 = 0.5;
        double[] p = [0.2, 0.3];
        var d = new NegativeMultinomialDistribution(r, p0, p);
        for (int i = 0; i < 2; i++)
        {
            var marginal = new NegativeBinomialDistribution(r, p0 / (p0 + p[i]));
            for (long k = 0; k <= 20; k++)
            {
                double sum = 0;
                long[] x = new long[2];
                x[i] = k;
                for (long other = 0; other <= 500; other++)
                {
                    x[1 - i] = other;
                    sum += d.Pmf(x);
                }

                DistributionAssert.Close(marginal.Pmf(k), sum, 1e-13, $"X{i} = {k}");
            }
        }
    }

    [Fact]
    public void AllZeroPointIsTheSuccessProbabilityToThePowerR()
    {
        var d = new NegativeMultinomialDistribution(3.5, 0.3, 0.3, 0.4);
        DistributionAssert.Close(3.5 * Math.Log(0.3), d.LogPmf(new long[] { 0, 0 }), 1e-15, "p₀ = 0.3");

        // p₀ = 1 - 2⁻⁴⁰ では ln p₀ を失敗の確率の和から求めて、対数の相対精度を保つ
        double q = Math.ScaleB(1, -41);
        var nearOne = new NegativeMultinomialDistribution(1000, 1 - 2 * q, q, q);
        DistributionAssert.Close(1000 * XMath.Log1p(-2 * q), nearOne.LogPmf(new long[] { 0, 0 }), 1e-14, "p₀ = 1 - 2⁻⁴⁰");
    }

    [Theory]
    [InlineData(3.0, 0.6, new double[] { 0.25, 0.15 })]
    [InlineData(0.4, 0.7, new double[] { 0.1, 0.1, 0.1 })]
    public void ProbabilitiesSumToOneAndMatchTheMoments(double r, double p0, double[] p)
    {
        // 各成分を 0..limit の箱の中で数える。箱の外の確率は 0.3^limit 程度で無視できる
        var d = new NegativeMultinomialDistribution(r, p0, p);
        int limit = p.Length == 2 ? 200 : 60;
        MultinomialTests.AssertMomentsByEnumeration(d, Box(limit, p.Length), 1e-12);
    }

    [Fact]
    public void MomentsStayFiniteWhenTheOddsOverflow()
    {
        // p₀ = 10⁻³¹⁰ では pᵢ/p₀ が溢れるが、r = 10⁻³²⁰ を掛けた平均・分散は表せる（期待値は入力の double の値から mpmath で求めた値）
        var d = new NegativeMultinomialDistribution(1e-320, 1e-310, 1.0);
        DistributionAssert.Close(9.99988867182686e-11, d.GetMean()[0], 1e-14, "Mean");
        DistributionAssert.Close(9.99988867182689e299, d.GetCovariance()[0, 0], 1e-14, "Variance");
        DistributionAssert.Close(new NegativeBinomialDistribution(1e-320, 1e-310).Mean, d.GetMean()[0], 1e-14, "NegativeBinomial.Mean");

        // 確率0の成分は常に0なので、平均と共分散の行・列は0（0·∞ の NaN にしない）。表せない分散は +∞
        var zero = new NegativeMultinomialDistribution(1, 1e-310, 1.0, 0.0);
        var mean = zero.GetMean();
        var covariance = zero.GetCovariance();
        Assert.Equal(double.PositiveInfinity, mean[0]);
        Assert.Equal(0.0, mean[1]);
        Assert.Equal(double.PositiveInfinity, covariance[0, 0]);
        Assert.Equal(0.0, covariance[0, 1]);
        Assert.Equal(0.0, covariance[1, 0]);
        Assert.Equal(0.0, covariance[1, 1]);
    }

    [Fact]
    public void ZeroProbabilityFailureKindsStayEmpty()
    {
        var d = new NegativeMultinomialDistribution(2, 0.5, 0.5, 0.0);
        Assert.Equal(double.NegativeInfinity, d.LogPmf(new long[] { 1, 1 }));
        DistributionAssert.Close(new NegativeBinomialDistribution(2, 0.5).LogPmf(4), d.LogPmf(new long[] { 4, 0 }), 1e-14, "LogPmf");
        var samples = d.Sample(new Pcg64DxsmRandom(8, 1), 500);
        for (int j = 0; j < samples.Columns; j++)
        {
            Assert.Equal(0L, samples[1, j]);
        }

        // p₀ = 1 では失敗が起きない
        var certain = new NegativeMultinomialDistribution(4, 1.0, 0.0, 0.0);
        Assert.Equal(0.0, certain.LogPmf(new long[] { 0, 0 }));
        Assert.Equal(double.NegativeInfinity, certain.LogPmf(new long[] { 0, 1 }));
        var fixedSamples = certain.Sample(new Pcg64DxsmRandom(8, 2), 50);
        for (int j = 0; j < fixedSamples.Columns; j++)
        {
            Assert.Equal(new long[] { 0, 0 }, fixedSamples.Column(j).ToArray());
        }
    }

    [Fact]
    public void SamplesSaturateWhenTheMixingRateOverflows()
    {
        // ポアソン乱数の平均 G pᵢ/p₀ が倍精度で溢れる成分は long.MaxValue になる（負の二項分布と同じ）
        var d = new NegativeMultinomialDistribution(1e15, 1e-320, 0.5, 0.5);
        var samples = d.Sample(new Pcg64DxsmRandom(9, 9), 64);
        for (int j = 0; j < samples.Columns; j++)
        {
            Assert.Equal(new[] { long.MaxValue, long.MaxValue }, samples.Column(j).ToArray());
        }
    }

    [Fact]
    public void InvalidParametersThrow()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new NegativeMultinomialDistribution(0, 0.5, 0.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NegativeMultinomialDistribution(-1, 0.5, 0.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NegativeMultinomialDistribution(double.NaN, 0.5, 0.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NegativeMultinomialDistribution(double.PositiveInfinity, 0.5, 0.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NegativeMultinomialDistribution(Math.BitIncrement(NegativeMultinomialDistribution.MaxSuccesses), 0.5, 0.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NegativeMultinomialDistribution(1, 0.0, 1.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NegativeMultinomialDistribution(1, 1.5, -0.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NegativeMultinomialDistribution(1, double.NaN, 0.5));
        Assert.Throws<ArgumentException>(() => new NegativeMultinomialDistribution(1, 1.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NegativeMultinomialDistribution(1, 0.5, 0.75, -0.25));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NegativeMultinomialDistribution(1, 0.5, double.PositiveInfinity));
        Assert.Throws<ArgumentException>(() => new NegativeMultinomialDistribution(1, 0.5, 0.4));
        Assert.Throws<ArgumentException>(() => new NegativeMultinomialDistribution(1, 0.5, 0.5, 0.5));
        _ = new NegativeMultinomialDistribution(NegativeMultinomialDistribution.MaxSuccesses, 0.5, 0.5);
    }

    private static IEnumerable<long[]> Box(int limit, int k)
    {
        if (k == 0)
        {
            yield return [];
            yield break;
        }

        for (long first = 0; first <= limit; first++)
        {
            foreach (long[] rest in Box(limit, k - 1))
            {
                yield return [first, .. rest];
            }
        }
    }
}
