using Tuna.RandomSource;
using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Statistics;

// ディリクレ多項分布: ベータ二項分布・一様な分布・カテゴリカル分布・多項分布との一致、全体の確率の和とモーメント、
// 集中度が非正規化数の場合、退化した場合、パラメータの検証。
public class DirichletMultinomialTests
{
    [Theory]
    [InlineData(10L, 0.5, 2.0)]
    [InlineData(100L, 3.0, 7.0)]
    [InlineData(1000L, 0.1, 0.2)]
    [InlineData(50L, 100.0, 200.0)]
    public void TwoComponentsMatchBetaBinomial(long n, double a, double b)
    {
        // P(X₁ = k) = C(n, k) B(k + a, n - k + b)/B(a, b)
        var d = new DirichletMultinomialDistribution(n, a, b);
        foreach (long k in new[] { 0L, 1L, n / 3, n / 2, n - 1, n })
        {
            double expected = XMath.Special.LogBinomial(n, k) + XMath.Special.LogBeta(k + a, n - k + b) - XMath.Special.LogBeta(a, b);
            DistributionAssert.Close(expected, d.LogPmf(new[] { k, n - k }), 1e-12, $"k = {k}");
        }
    }

    [Fact]
    public void UnitConcentrationsGiveUniformCompositions()
    {
        // α = (1, 1, 1) では和が n の C(n + 2, 2) 通りの点がすべて同じ確率
        var d = new DirichletMultinomialDistribution(6, 1, 1, 1);
        double expected = -Math.Log(28);
        foreach (long[] x in MultinomialTests.Compositions(6, 3))
        {
            DistributionAssert.Close(expected, d.LogPmf(x), 1e-14, $"x = [{string.Join(", ", x)}]");
        }
    }

    [Fact]
    public void OneTrialIsCategorical()
    {
        double[] alpha = [0.3, 2.0, 0.7, 5.0];
        var d = new DirichletMultinomialDistribution(1, alpha);
        for (int j = 0; j < alpha.Length; j++)
        {
            long[] x = new long[alpha.Length];
            x[j] = 1;
            DistributionAssert.Close(Math.Log(alpha[j] / d.Concentration), d.LogPmf(x), 1e-14, $"category {j}");
        }
    }

    [Theory]
    [InlineData(new double[] { 1e40, 1e200 })]
    [InlineData(new double[] { 2.3e45, 1e40 })]
    [InlineData(new double[] { 1e300, 1.0, 1e100 })]
    public void OneTrialWithLargeAsymmetricConcentrationsIsCategorical(double[] alpha)
    {
        // n = 1 の確率は αⱼ/α₀。Uwⱼ ≈ αⱼ が大きいので、Uwⱼ を丸めると bd0 に αⱼε²/2（10⁴⁰ で 10⁸）の誤差が入る。
        // 期待値の対数は、確率が 1 に近ければ他の成分の和 q から ln(1 - q/α₀) で求める
        var d = new DirichletMultinomialDistribution(1, alpha);
        for (int j = 0; j < alpha.Length; j++)
        {
            long[] x = new long[alpha.Length];
            x[j] = 1;
            double others = alpha.Where((_, i) => i != j).Sum();
            double expected = alpha[j] > others ? XMath.Log1p(-others / d.Concentration) : Math.Log(alpha[j] / d.Concentration);
            DistributionAssert.Close(expected, d.LogPmf(x), 1e-14, $"category {j}");
        }
    }

    [Fact]
    public void LargeConcentrationsApproachTheMultinomial()
    {
        // α = cp で c → ∞ の極限は Mult(n, p)。差は相対で n²/c 程度
        double[] p = [0.2, 0.3, 0.5];
        var multinomial = new MultinomialDistribution(20, p);
        var d = new DirichletMultinomialDistribution(20, p.Select(v => 1e18 * v).ToArray());
        foreach (long[] x in MultinomialTests.Compositions(20, 3))
        {
            DistributionAssert.Close(multinomial.LogPmf(x), d.LogPmf(x), 1e-13, $"x = [{string.Join(", ", x)}]");
        }
    }

    [Theory]
    [InlineData(7L, new double[] { 0.5, 1.5, 2.0 })]
    [InlineData(10L, new double[] { 0.01, 0.02, 0.05 })]
    [InlineData(12L, new double[] { 100.0, 200.0 })]
    [InlineData(5L, new double[] { 1e-3 })]
    public void ProbabilitiesSumToOneAndMatchTheMoments(long n, double[] alpha)
    {
        var d = new DirichletMultinomialDistribution(n, alpha);
        MultinomialTests.AssertMomentsByEnumeration(d, MultinomialTests.Compositions(n, alpha.Length), 1e-13);
    }

    [Theory]
    [InlineData(15L)]
    [InlineData(1_000_000_000_000_000L)]
    public void SubnormalConcentrationsPutTheMassOnTheVertices(long n)
    {
        // α → 0 では確率ベクトルがほぼ頂点に集まり、P(X = n eᵢ) → αᵢ/α₀。
        // n = 10¹⁵ では α₀/(n + α₀) が 0 に丸まるが、確率に要る α₀wᵢ/(n + α₀) は表せる
        var d = new DirichletMultinomialDistribution(n, 1e-310, 2e-310, 3e-310);
        double a0 = d.Concentration;
        for (int i = 0; i < 3; i++)
        {
            long[] vertex = new long[3];
            vertex[i] = n;
            DistributionAssert.Close(Math.Log(d.Alpha[i] / a0), d.LogPmf(vertex), 1e-14, $"vertex {i}");
        }

        // 頂点以外の点の確率は α 程度で、有限な対数を持つ
        double inside = d.LogPmf(new long[] { n - 1, 1, 0 });
        Assert.True(double.IsFinite(inside) && inside < -700, $"LogPmf {inside}");

        const int count = 6000;
        var samples = d.Sample(new Pcg64DxsmRandom(7, 3), count);
        long[] hits = new long[3];
        for (int j = 0; j < count; j++)
        {
            int nonzero = 0;
            for (int i = 0; i < 3; i++)
            {
                if (samples[i, j] != 0)
                {
                    Assert.Equal(n, samples[i, j]);
                    hits[i]++;
                    nonzero++;
                }
            }

            Assert.Equal(1, nonzero);
        }

        for (int i = 0; i < 3; i++)
        {
            double p = d.Alpha[i] / a0;
            Assert.True(Math.Abs(hits[i] - count * p) <= 5 * Math.Sqrt(count * p * (1 - p)), $"vertex {i}: {hits[i]} of {count}");
        }
    }

    [Fact]
    public void MomentsKeepTheirDigitsWhenTheRatiosAreSubnormal()
    {
        // α₂/α₀ ≈ 3·10⁻³²³ は非正規化数だが、n = 2⁵³ を掛けた平均と共分散は正規化数（期待値は入力の double の値から mpmath で求めた値）
        var d = new DirichletMultinomialDistribution(1L << 53, 30, 1e-320);
        DistributionAssert.Close(3.002366326412384e-306, d.GetMean()[1], 1e-14, "Mean");
        var covariance = d.GetCovariance();
        DistributionAssert.Close(8.723519915393923e-292, covariance[1, 1], 1e-14, "Variance");
        DistributionAssert.Close(-8.723519915393923e-292, covariance[0, 1], 1e-14, "Covariance");
    }

    [Fact]
    public void DegenerateDistributionsUseNoRandomness()
    {
        // n = 0 は乱数を使わずに0のベクトルを返す
        var d = new DirichletMultinomialDistribution(0, 0.3, 4.0);
        var random = new Pcg64DxsmRandom(5, 6);
        var sample = d.Sample(random, 10);
        Assert.Equal(new Pcg64DxsmRandom(5, 6).NextUInt64(), random.NextUInt64());
        for (int j = 0; j < sample.Columns; j++)
        {
            Assert.Equal(new long[] { 0, 0 }, sample.Column(j).ToArray());
        }

        Assert.Equal(0.0, d.LogPmf(new long[] { 0, 0 }));
        Assert.Equal(0.0, new DirichletMultinomialDistribution(9, 0.25).LogPmf(new long[] { 9 }));
    }

    [Fact]
    public void InvalidParametersThrow()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new DirichletMultinomialDistribution(-1, 1.0, 1.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DirichletMultinomialDistribution(DirichletMultinomialDistribution.MaxTrials + 1, 1.0, 1.0));
        Assert.Throws<ArgumentException>(() => new DirichletMultinomialDistribution(3));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DirichletMultinomialDistribution(3, 1.0, 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DirichletMultinomialDistribution(3, 1.0, -1.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DirichletMultinomialDistribution(3, 1.0, double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DirichletMultinomialDistribution(3, 1.0, double.PositiveInfinity));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DirichletMultinomialDistribution(3, 1e308, 1e308));
        _ = new DirichletMultinomialDistribution(DirichletMultinomialDistribution.MaxTrials, 1.0, 1.0);
    }
}
