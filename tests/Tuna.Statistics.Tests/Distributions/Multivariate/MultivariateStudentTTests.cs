using Tuna.LinearAlgebra;
using Tuna.RandomSource;
using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Statistics;

// 多変量 t 分布: 1変量の t 分布・多変量正規分布（ν = ∞ と大きな ν）との一致、モーメントの規約、パラメータの検証。
public class MultivariateStudentTTests
{
    private static ReadOnlyMatrixSpan<double> Square(params double[] values) => MultivariateContractTests.Square(values);

    private static MultivariateStudentTDistribution Create(double nu) =>
        new(nu, MultivariateContractTests.Mean3, Square(MultivariateContractTests.Covariance3));

    [Theory]
    [InlineData(1.0, 0.5, 2.0, 0.5)]
    [InlineData(1.0, 0.5, 2.0, -1e10)]
    [InlineData(2.5, -1.0, 0.25, 3.0)]
    [InlineData(30.0, 0.0, 1.0, 1.5)]
    [InlineData(1e10, 0.0, 1.0, 1.5)]
    [InlineData(0.3, 2.0, 3.0, 1e100)]
    public void OneDimensionMatchesStudentT(double nu, double location, double scale, double x)
    {
        var univariate = new StudentTDistribution(nu, location, scale);
        var multivariate = new MultivariateStudentTDistribution(nu, new[] { location }, Square(scale * scale));
        DistributionAssert.Close(univariate.LogPdf(x), multivariate.LogPdf(new[] { x }), 4e-15, "LogPdf");
    }

    [Theory]
    [InlineData(3.5)]
    [InlineData(1.0)]
    [InlineData(0.4)]
    public void OneDimensionalSamplesFromTheFactorMatchStudentT(double nu)
    {
        var univariate = new StudentTDistribution(nu, -2, 0.75);
        var multivariate = MultivariateStudentTDistribution.FromCholeskyFactor(nu, new[] { -2.0 }, Square(0.75));
        var a = new Pcg64DxsmRandom(12, 1);
        var b = new Pcg64DxsmRandom(12, 1);
        for (int k = 0; k < 200; k++)
        {
            Assert.Equal(univariate.Sample(a), multivariate.Sample(b)[0]);
        }
    }

    [Fact]
    public void TinyDegreesOfFreedomSamplesAreNeverNaN()
    {
        // ν が極小だと z √(ν/V) が溢れる。先に Lz を求めてから掛けるので、成分は ±∞ か有限で NaN にならない（単独の版と複数の版）
        var t = new MultivariateStudentTDistribution(0.001, new double[2], Square(1, 0, 0, 1));
        var single = t.Sample(new Pcg64DxsmRandom(17, 3));
        Assert.All(single.ToArray(), v => Assert.False(double.IsNaN(v)));
        var samples = t.Sample(new Pcg64DxsmRandom(17, 3), 1000);
        Assert.All(samples.AsSpan().ToArray(), v => Assert.False(double.IsNaN(v)));
        Assert.Equal(single.ToArray(), new[] { samples[0, 0], samples[1, 0] });

        // 非対角がある尺度でも、成分ごとの符号は Lz の符号
        var correlated = new MultivariateStudentTDistribution(0.001, new[] { 1.0, -1.0 }, Square(2, 0.5, 0.5, 1));
        Assert.All(correlated.Sample(new Pcg64DxsmRandom(5, 1), 1000).AsSpan().ToArray(), v => Assert.False(double.IsNaN(v)));
    }

    [Fact]
    public void InfiniteDegreesOfFreedomMatchMultivariateNormal()
    {
        var t = Create(double.PositiveInfinity);
        var normal = new MultivariateNormalDistribution(MultivariateContractTests.Mean3, Square(MultivariateContractTests.Covariance3));
        foreach (double[] x in new[] { new[] { 0.0, 0.0, 0.0 }, new[] { 3.0, -1.0, 2.0 }, new[] { -300.0, 10.0, 0.5 } })
        {
            Assert.Equal(normal.LogPdf(x), t.LogPdf(x));
        }

        Assert.Equal(normal.GetMean().ToArray(), t.GetMean().ToArray());
        Assert.Equal(normal.GetCovariance().AsSpan().ToArray(), t.GetCovariance().AsSpan().ToArray());
        Assert.Equal(normal.Sample(new Pcg64DxsmRandom(4, 4), 20).AsSpan().ToArray(), t.Sample(new Pcg64DxsmRandom(4, 4), 20).AsSpan().ToArray());
    }

    [Fact]
    public void LargeDegreesOfFreedomApproachMultivariateNormal()
    {
        // ln p_ν - ln p_∞ = O(1/ν)。正規化定数を対数ガンマ関数の差で求めると ν = 1e12 では 1e-4 程度の誤差が出る
        var normal = Create(double.PositiveInfinity);
        foreach (double nu in new[] { 1e12, 1e15, 1e300 })
        {
            var t = Create(nu);
            foreach (double[] x in new[] { new[] { 0.0, 0.0, 0.0 }, new[] { 3.0, -1.0, 2.0 } })
            {
                double expected = normal.LogPdf(x);
                Assert.True(Math.Abs(t.LogPdf(x) - expected) <= 100 / nu + 1e-14 * Math.Abs(expected), $"ν = {nu}: {t.LogPdf(x)} vs {expected}");
            }
        }
    }

    [Fact]
    public void TailBeyondTheOverflowOfTheSquaredNormIsFinite()
    {
        // ‖z‖² が溢れる点でも、ln p = c - ((ν + d)/2) ln(‖z‖²/ν) + O(ν/‖z‖²) を対数のまま求める
        var t = new MultivariateStudentTDistribution(3, new[] { 0.0, 0.0 }, Square(1, 0, 0, 1));
        double logPdf = t.LogPdf(new[] { 1e200, 1e200 });
        double expected = t.LogPdf(new[] { 1e100, 1e100 }) - 2.5 * Math.Log(1e200);
        DistributionAssert.Close(expected, logPdf, 1e-14, "LogPdf");
        Assert.Equal(double.NegativeInfinity, t.LogPdf(new[] { double.PositiveInfinity, 0.0 }));
    }

    [Fact]
    public void MomentsFollowTheUnivariateConventions()
    {
        // ν > 2: ν/(ν - 2)·S、1 < ν ≤ 2: 平均は μ・分散は +∞・共分散は NaN、ν ≤ 1: 平均も NaN
        var t = Create(3.5);
        Assert.Equal(MultivariateContractTests.Mean3, t.GetMean().ToArray());
        var covariance = t.GetCovariance();
        for (int k = 0; k < 9; k++)
        {
            DistributionAssert.Close(3.5 / 1.5 * MultivariateContractTests.Covariance3[k], covariance.AsSpan()[k], 1e-15, "Covariance");
        }

        var heavy = Create(2);
        Assert.Equal(MultivariateContractTests.Mean3, heavy.GetMean().ToArray());
        var heavyCovariance = heavy.GetCovariance();
        Assert.Equal(double.PositiveInfinity, heavyCovariance[1, 1]);
        Assert.True(double.IsNaN(heavyCovariance[1, 0]));

        var cauchy = Create(1);
        Assert.All(cauchy.GetMean().ToArray(), v => Assert.True(double.IsNaN(v)));
        Assert.All(cauchy.GetCovariance().AsSpan().ToArray(), v => Assert.True(double.IsNaN(v)));
        Assert.Equal(MultivariateContractTests.Covariance3, cauchy.Scale.ToArray().Cast<double>().ToArray());
        Assert.Equal(1.0, cauchy.DegreesOfFreedom);
    }

    [Fact]
    public void InvalidParametersThrow()
    {
        var scale = MultivariateContractTests.Covariance3;
        double[] location = MultivariateContractTests.Mean3;
        Assert.Throws<ArgumentOutOfRangeException>(() => new MultivariateStudentTDistribution(0, location, Square(scale)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new MultivariateStudentTDistribution(-1, location, Square(scale)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new MultivariateStudentTDistribution(double.NaN, location, Square(scale)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new MultivariateStudentTDistribution(2, new[] { 0.0, double.NaN, 0.0 }, Square(scale)));
        Assert.Throws<ArgumentException>(() => new MultivariateStudentTDistribution(2, ReadOnlySpan<double>.Empty, Square()));
        Assert.Throws<ArgumentException>(() => new MultivariateStudentTDistribution(2, location, Square(1, 0, 0, 1)));
        Assert.Throws<ArgumentException>(() => new MultivariateStudentTDistribution(2, new[] { 0.0, 0.0 }, Square(1, 2, 2, 1)));
        Assert.Throws<ArgumentOutOfRangeException>(() => MultivariateStudentTDistribution.FromCholeskyFactor(2, new[] { 0.0 }, Square(0.0)));
        Assert.Throws<ArgumentOutOfRangeException>(() => MultivariateStudentTDistribution.FromCholeskyFactor(0, new[] { 0.0 }, Square(1.0)));
    }
}
