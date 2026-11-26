using Tuna.RandomSource;
using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Statistics;

// ディリクレ分布: 2成分のベータ分布との一致、単体の境界と総和の許容幅の規約、モーメント、小さな α の乱数、パラメータの検証。
public class DirichletTests
{
    [Theory]
    [InlineData(2.0, 3.0, 0.25)]
    [InlineData(0.5, 0.5, 0.375)]
    [InlineData(1e6, 2e6, 0.3330078125)]
    [InlineData(0.01, 5.0, 0.0009765625)]
    [InlineData(1.5, 1e8, 1.4901161193847656e-08)]
    public void TwoComponentsMatchBeta(double a, double b, double x)
    {
        // x と 1 - x の和がちょうど1になる点（x は2の累乗の分数）
        var beta = new BetaDistribution(a, b);
        var dirichlet = new DirichletDistribution(a, b);
        double expected = beta.LogPdf(x);
        DistributionAssert.Close(expected, dirichlet.LogPdf(new[] { x, 1 - x }), 4e-15, "LogPdf");
        DistributionAssert.Close(new BetaDistribution(b, a).LogPdf(1 - x), new DirichletDistribution(b, a).LogPdf(new[] { 1 - x, x }), 4e-15, "swapped");
    }

    [Fact]
    public void UniformDirichletHasTheReciprocalVolumeOfTheSimplex()
    {
        // α = 1 は先頭の K - 1 座標について密度 (K - 1)!
        var d = new DirichletDistribution(1, 1, 1, 1);
        DistributionAssert.Close(Math.Log(6), d.LogPdf(new[] { 0.125, 0.25, 0.375, 0.25 }), 1e-15, "LogPdf");
        DistributionAssert.Close(Math.Log(6), d.LogPdf(new[] { 0.0, 0.0, 0.5, 0.5 }), 1e-15, "boundary");
    }

    [Fact]
    public void BoundaryUsesTheLimitOfEachFactor()
    {
        // xᵢ = 0 での xᵢ^(αᵢ - 1) は αᵢ = 1 なら1、αᵢ < 1 なら +∞、αᵢ > 1 なら0。+∞ と0が同時なら NaN
        double[] x = [0.0, 0.5, 0.5];
        DistributionAssert.Close(Math.Log(7.5), new DirichletDistribution(1, 2, 3).LogPdf(x), 1e-15, "αᵢ = 1");
        Assert.Equal(double.PositiveInfinity, new DirichletDistribution(0.5, 2, 3).LogPdf(x));
        Assert.Equal(double.PositiveInfinity, new DirichletDistribution(0.5, 2, 3).Pdf(x));
        Assert.Equal(double.NegativeInfinity, new DirichletDistribution(2, 2, 3).LogPdf(x));
        Assert.Equal(0.0, new DirichletDistribution(2, 2, 3).Pdf(x));
        Assert.True(double.IsNaN(new DirichletDistribution(0.5, 2, 3).LogPdf(new[] { 0.0, 0.0, 1.0 })));

        // 頂点
        Assert.Equal(double.NegativeInfinity, new DirichletDistribution(2, 2, 3).LogPdf(new[] { 1.0, 0.0, 0.0 }));
        Assert.Equal(0.0, new DirichletDistribution(1, 1).LogPdf(new[] { 1.0, 0.0 }));
    }

    [Fact]
    public void PointsOffTheSimplexHaveZeroDensity()
    {
        var d = new DirichletDistribution(2, 3, 0.5);
        double[] onSimplex = [0.3, 0.3, 0.4];
        double inside = d.LogPdf(onSimplex);
        Assert.True(double.IsFinite(inside));

        // 総和のずれが 16Kε 以内なら x/Σx で評価し、その外は台の外
        double[] slightlyOff = [0.3, 0.3, 0.4 + 4e-15];
        Assert.InRange(Math.Abs(slightlyOff.Sum() - 1), 1e-15, 48 * 2.220446049250313e-16);
        DistributionAssert.Close(inside, d.LogPdf(slightlyOff), 1e-13, "within the tolerance");
        Assert.Equal(double.NegativeInfinity, d.LogPdf(new[] { 0.3, 0.3, 0.4 + 1e-10 }));
        Assert.Equal(double.NegativeInfinity, d.LogPdf(new[] { 0.3, 0.3, 0.3 }));
        Assert.Equal(double.NegativeInfinity, d.LogPdf(new[] { 0.6, 0.6, -0.2 }));
        Assert.Equal(double.NegativeInfinity, d.LogPdf(new[] { 1.5, -0.5, 0.0 }));
        Assert.Equal(double.NegativeInfinity, d.LogPdf(new[] { -0.0, -0.0, -0.0 }));
    }

    [Fact]
    public void MomentsMatchTheClosedForms()
    {
        double[] alpha = [2, 3, 0.5];
        var d = new DirichletDistribution(alpha);
        double a0 = 5.5;
        Assert.Equal(a0, d.Concentration);
        Assert.Equal(alpha, d.Alpha.ToArray());
        var mean = d.GetMean();
        var covariance = d.GetCovariance();
        for (int i = 0; i < 3; i++)
        {
            DistributionAssert.Close(alpha[i] / a0, mean[i], 1e-15, "Mean");
            double rowSum = 0;
            for (int j = 0; j < 3; j++)
            {
                double expected = alpha[i] * ((i == j ? a0 : 0) - alpha[j]) / (a0 * a0 * (a0 + 1));
                DistributionAssert.Close(expected, covariance[i, j], 1e-15, "Covariance");
                Assert.Equal(covariance[i, j], covariance[j, i]);
                rowSum += covariance[i, j];
            }

            Assert.True(Math.Abs(rowSum) <= 1e-16, $"row sum {rowSum}");
        }

        // 1 - αᵢ/α₀ を引き算せずに求めるので、他の成分が小さくても分散が相対精度を保つ
        var concentrated = new DirichletDistribution(1e20, 1, 1);
        DistributionAssert.Close(2e-20 / (1e20 + 3), concentrated.GetCovariance()[0, 0], 1e-14, "Variance");
    }

    [Fact]
    public void SamplesWithTinyConcentrationsStayOnTheSimplex()
    {
        // ガンマ乱数がアンダーフローするほど小さな α でも 0/0 にならず、和は1
        foreach (var d in new[] { new DirichletDistribution(1e-3, 1e-3, 1e-3), new DirichletDistribution(1e-30, 1e-30), new DirichletDistribution(1e6, 1e-6) })
        {
            var samples = d.Sample(new Pcg64DxsmRandom(9, 9), 2000);
            for (int j = 0; j < samples.Columns; j++)
            {
                double sum = 0;
                for (int i = 0; i < samples.Rows; i++)
                {
                    Assert.InRange(samples[i, j], 0.0, 1.0);
                    sum += samples[i, j];
                }

                Assert.True(Math.Abs(sum - 1) <= 4e-16 * samples.Rows, $"sum {sum}");
            }
        }
    }

    [Fact]
    public void ConcentrationsSoSmallThatEveryLogGammaOverflowsGiveAVertex()
    {
        // α = 1e-310 では全成分の ln U / α が溢れて −∞ になる。極限の値（−ln U/α が最小の成分が 1、他は 0）を返し、NaN にしない
        var d = new DirichletDistribution(1e-310, 1e-310, 1e-310);
        var single = d.Sample(new Pcg64DxsmRandom(17, 3));
        var samples = d.Sample(new Pcg64DxsmRandom(17, 3), 2000);
        Assert.Equal(single.ToArray(), new[] { samples[0, 0], samples[1, 0], samples[2, 0] });
        var counts = new int[3];
        for (int j = 0; j < samples.Columns; j++)
        {
            int ones = 0;
            for (int i = 0; i < 3; i++)
            {
                Assert.True(samples[i, j] == 0 || samples[i, j] == 1, $"{samples[i, j]}");
                if (samples[i, j] == 1)
                {
                    ones++;
                    counts[i]++;
                }
            }

            Assert.Equal(1, ones);
        }

        // 同じ α なら頂点は対称に選ばれる（2000 回で各 1/3、標準偏差は約 21）
        Assert.All(counts, c => Assert.InRange(c, 560, 780));
    }

    [Fact]
    public void InvalidParametersThrow()
    {
        Assert.Throws<ArgumentException>(() => new DirichletDistribution(1.0));
        Assert.Throws<ArgumentException>(() => new DirichletDistribution());
        Assert.Throws<ArgumentOutOfRangeException>(() => new DirichletDistribution(1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DirichletDistribution(1, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DirichletDistribution(1, double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DirichletDistribution(1, double.PositiveInfinity));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DirichletDistribution(1e308, 1e308));
    }
}
