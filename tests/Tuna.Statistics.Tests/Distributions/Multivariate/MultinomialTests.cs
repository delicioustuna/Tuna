using Tuna.RandomSource;
using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Statistics;

// 多項分布: 二項分布・カテゴリカル分布との一致、全体の確率の和とモーメント、確率0の成分と確率が1に近い成分、退化した場合、パラメータの検証。
public class MultinomialTests
{
    /// <summary>成分が <paramref name="k"/> 個で和が <paramref name="total"/> の非負整数ベクトルをすべて列挙する</summary>
    internal static IEnumerable<long[]> Compositions(long total, int k)
    {
        if (k == 1)
        {
            yield return [total];
            yield break;
        }

        for (long first = 0; first <= total; first++)
        {
            foreach (long[] rest in Compositions(total - first, k - 1))
            {
                yield return [first, .. rest];
            }
        }
    }

    /// <summary>確率の和、平均、共分散を列挙した点から求めて、分布のモーメントと比べる</summary>
    internal static void AssertMomentsByEnumeration(IMultivariateDiscreteDistribution d, IEnumerable<long[]> points, double tolerance)
    {
        int k = d.Dimension;
        double total = 0;
        double[] mean = new double[k];
        double[] second = new double[k * k];
        foreach (long[] x in points)
        {
            double p = d.Pmf(x);
            total += p;
            for (int i = 0; i < k; i++)
            {
                mean[i] += p * x[i];
                for (int j = 0; j < k; j++)
                {
                    second[i + j * k] += p * x[i] * x[j];
                }
            }
        }

        Assert.True(Math.Abs(total - 1) <= tolerance, $"total {total}");
        var expectedMean = d.GetMean();
        var expectedCovariance = d.GetCovariance();
        for (int i = 0; i < k; i++)
        {
            Assert.True(Math.Abs(mean[i] - expectedMean[i]) <= tolerance * Math.Max(1, Math.Abs(expectedMean[i])), $"mean[{i}] {mean[i]} vs {expectedMean[i]}");
            for (int j = 0; j < k; j++)
            {
                double covariance = second[i + j * k] - mean[i] * mean[j];
                double scale = Math.Max(1, Math.Sqrt(Math.Abs(expectedCovariance[i, i] * expectedCovariance[j, j])));
                Assert.True(Math.Abs(covariance - expectedCovariance[i, j]) <= tolerance * scale,
                    $"covariance[{i}, {j}] {covariance} vs {expectedCovariance[i, j]}");
            }
        }
    }

    [Theory]
    [InlineData(10L, 0.3)]
    [InlineData(1000L, 0.01)]
    [InlineData(1000000L, 0.5)]
    [InlineData(1000000000000L, 0.25)]
    public void TwoComponentsMatchBinomial(long n, double p)
    {
        var binomial = new BinomialDistribution(n, p);
        var multinomial = new MultinomialDistribution(n, p, 1 - p);
        foreach (long k in new[] { 0L, 1L, (long)(n * p), n / 3, n - 1, n })
        {
            DistributionAssert.Close(binomial.LogPmf(k), multinomial.LogPmf(new[] { k, n - k }), 1e-13, $"k = {k}");
        }
    }

    [Fact]
    public void OneTrialIsCategorical()
    {
        double[] p = [0.1, 0.0, 0.6, 0.3];
        var categorical = new CategoricalDistribution(p);
        var multinomial = new MultinomialDistribution(1, p);
        for (int j = 0; j < p.Length; j++)
        {
            long[] x = new long[p.Length];
            x[j] = 1;
            DistributionAssert.Close(categorical.LogPmf(j), multinomial.LogPmf(x), 1e-15, $"category {j}");
        }

        Assert.Equal(0.0, multinomial.Pmf(new long[] { 1, 0, 1, 0 }));
    }

    [Theory]
    [InlineData(7L, new double[] { 0.1, 0.2, 0.3, 0.4 })]
    [InlineData(12L, new double[] { 0.05, 0.9, 0.05 })]
    [InlineData(5L, new double[] { 0.0, 0.5, 0.5 })]
    public void ProbabilitiesSumToOneAndMatchTheMoments(long n, double[] p)
    {
        var d = new MultinomialDistribution(n, p);
        AssertMomentsByEnumeration(d, Compositions(n, p.Length), 1e-13);
    }

    [Fact]
    public void ZeroProbabilityComponentsMustBeEmpty()
    {
        var d = new MultinomialDistribution(9, 0.0, 0.25, 0.0, 0.75);
        Assert.Equal(double.NegativeInfinity, d.LogPmf(new long[] { 1, 2, 0, 6 }));
        Assert.Equal(double.NegativeInfinity, d.LogPmf(new long[] { 0, 2, 1, 6 }));
        DistributionAssert.Close(new BinomialDistribution(9, 0.25).LogPmf(3), d.LogPmf(new long[] { 0, 3, 0, 6 }), 1e-14, "LogPmf");
        var samples = d.Sample(new Pcg64DxsmRandom(4, 4), 500);
        for (int j = 0; j < samples.Columns; j++)
        {
            Assert.Equal(0L, samples[0, j]);
            Assert.Equal(0L, samples[2, j]);
            Assert.Equal(9L, samples[1, j] + samples[3, j]);
        }
    }

    [Fact]
    public void ProbabilityNearOneKeepsTheRelativeAccuracyOfTheLogarithm()
    {
        // p₁ = 1 - 2⁻⁴⁰ はちょうど表せる。すべてが第1成分に入る確率の対数 n ln p₁ は -n·2⁻⁴⁰ 程度
        double q = Math.ScaleB(1, -40);
        var d = new MultinomialDistribution(1000, 1 - q, q);
        DistributionAssert.Close(1000 * XMath.Log1p(-q), d.LogPmf(new long[] { 1000, 0 }), 1e-14, "LogPmf");

        // 分散 npᵢ(1 - pᵢ) の 1 - pᵢ は他の成分の和から求める
        var concentrated = new MultinomialDistribution(10, 1e-20, 1e-20, 1.0);
        DistributionAssert.Close(10 * 2e-20, concentrated.GetCovariance()[2, 2], 1e-15, "Variance");
    }

    [Fact]
    public void DegenerateDistributionsUseNoRandomness()
    {
        // n = 0 と1成分の分布は値が決まっていて、乱数を使わない
        foreach (var d in new[] { new MultinomialDistribution(0, 0.3, 0.7), new MultinomialDistribution(7, 1.0), new MultinomialDistribution(5, 0, 1, 0) })
        {
            var random = new Pcg64DxsmRandom(5, 6);
            var sample = d.Sample(random, 10);
            Assert.Equal(new Pcg64DxsmRandom(5, 6).NextUInt64(), random.NextUInt64());
            long[] expected = d.GetMean().ToArray().Select(v => (long)v).ToArray();
            for (int j = 0; j < sample.Columns; j++)
            {
                Assert.Equal(expected, sample.Column(j).ToArray());
            }

            Assert.Equal(0.0, d.LogPmf(expected));
        }

        Assert.Equal(double.NegativeInfinity, new MultinomialDistribution(7, 1.0).LogPmf(new long[] { 6 }));
    }

    [Fact]
    public void ProbabilitiesAreNormalizedWithinTheTolerance()
    {
        const double eps = 2.220446049250313e-16;

        // 総和のずれは 16Kε まで受け付け、総和で割った値を使う
        var d = new MultinomialDistribution(3, 0.2, 0.3, 0.5 + 24 * eps);
        double sum = 0;
        foreach (double p in d.Probabilities)
        {
            sum += p;
        }

        Assert.True(Math.Abs(sum - 1) <= 2 * eps, $"sum {sum}");
        Assert.Throws<ArgumentException>(() => new MultinomialDistribution(3, 0.2, 0.3, 0.5 + 60 * eps));
    }

    [Fact]
    public void InvalidParametersThrow()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new MultinomialDistribution(-1, 0.5, 0.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => new MultinomialDistribution(MultinomialDistribution.MaxTrials + 1, 0.5, 0.5));
        Assert.Throws<ArgumentException>(() => new MultinomialDistribution(3));
        Assert.Throws<ArgumentOutOfRangeException>(() => new MultinomialDistribution(3, 1.5, -0.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => new MultinomialDistribution(3, 0.5, double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => new MultinomialDistribution(3, 0.5, double.PositiveInfinity));
        Assert.Throws<ArgumentException>(() => new MultinomialDistribution(3, 0.5, 0.4));
        Assert.Throws<ArgumentException>(() => new MultinomialDistribution(3, 0.0, 0.0));
        Assert.Throws<ArgumentException>(() => new MultinomialDistribution(3, 2.0, 3.0));
        _ = new MultinomialDistribution(MultinomialDistribution.MaxTrials, 0.5, 0.5);
    }
}
