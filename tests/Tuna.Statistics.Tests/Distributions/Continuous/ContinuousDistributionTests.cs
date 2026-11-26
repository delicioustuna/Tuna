using Tuna.RandomSource;
using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Statistics;

// 連続分布の規約（パラメータの検証、NaN・台の外・逆関数の端点）と、分布どうしの関係。値の精度は ContinuousReferenceTests で確かめる。
public class ContinuousDistributionTests
{
    private static IContinuousDistribution Create(string name) => name switch
    {
        "Normal" => new NormalDistribution(1.5, 2),
        "LogNormal" => new LogNormalDistribution(0.3, 0.8),
        "Exponential" => new ExponentialDistribution(2.5),
        "Uniform" => new UniformDistribution(-1, 3),
        "Cauchy" => new CauchyDistribution(1, 2),
        "Laplace" => new LaplaceDistribution(-1, 0.5),
        "Logistic" => new LogisticDistribution(2, 1.5),
        "GammaSmallShape" => new GammaDistribution(0.3, 2),
        "Gamma" => new GammaDistribution(7.5, 0.5),
        "ChiSquared" => new ChiSquaredDistribution(3),
        "BetaSmall" => new BetaDistribution(0.4, 0.7),
        "Beta" => new BetaDistribution(2, 5),
        "StudentT" => new StudentTDistribution(3.5, 1, 2),
        "StudentTInfinite" => new StudentTDistribution(double.PositiveInfinity, -1, 3),
        "WeibullSmallShape" => new WeibullDistribution(0.7, 2),
        "Weibull" => new WeibullDistribution(5, 1),
        "Gumbel" => new GumbelDistribution(-1, 2),
        "InverseGammaSmallShape" => new InverseGammaDistribution(0.4, 3),
        "InverseGamma" => new InverseGammaDistribution(5.5, 2),
        "FSmall" => new FDistribution(0.8, 1.5),
        "F" => new FDistribution(4, 9),
        "LocationScaleBeta" => new LocationScaleDistribution<BetaDistribution>(new BetaDistribution(2, 5), -3, 4),
        "LocationScaleGamma" => new LocationScaleDistribution<GammaDistribution>(new GammaDistribution(0.3, 2), 1, 0.5),
        "TruncatedNormal" => new TruncatedDistribution<NormalDistribution>(new NormalDistribution(1, 2), -1, 4),
        "HalfNormal" => new TruncatedDistribution<NormalDistribution>(NormalDistribution.Standard, 0, double.PositiveInfinity),
        "TruncatedGamma" => new TruncatedDistribution<GammaDistribution>(new GammaDistribution(0.3, 2), double.NegativeInfinity, 1.5),
        "TruncatedBeta" => new TruncatedDistribution<BetaDistribution>(new BetaDistribution(0.5, 0.5), 0.2, 3),
        "TruncatedCauchy" => new TruncatedDistribution<CauchyDistribution>(new CauchyDistribution(1, 2), double.NegativeInfinity, 0),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    public static TheoryData<string> Names =>
    [
        "Normal", "LogNormal", "Exponential", "Uniform", "Cauchy", "Laplace", "Logistic", "GammaSmallShape", "Gamma",
        "ChiSquared", "BetaSmall", "Beta", "StudentT", "StudentTInfinite", "WeibullSmallShape", "Weibull",
        "Gumbel", "InverseGammaSmallShape", "InverseGamma", "FSmall", "F", "LocationScaleBeta", "LocationScaleGamma",
        "TruncatedNormal", "HalfNormal", "TruncatedGamma", "TruncatedBeta", "TruncatedCauchy",
    ];

    [Theory]
    [MemberData(nameof(Names))]
    public void NaNArgumentsGiveNaN(string name)
    {
        IContinuousDistribution d = Create(name);
        Assert.Equal(double.NaN, d.Pdf(double.NaN));
        Assert.Equal(double.NaN, d.LogPdf(double.NaN));
        Assert.Equal(double.NaN, d.Cdf(double.NaN));
        Assert.Equal(double.NaN, d.Sf(double.NaN));
        Assert.Equal(double.NaN, d.LogCdf(double.NaN));
        Assert.Equal(double.NaN, d.LogSf(double.NaN));
        Assert.Equal(double.NaN, d.InverseCdf(double.NaN));
        Assert.Equal(double.NaN, d.InverseSf(double.NaN));
        Assert.Equal(double.NaN, d.InverseLogCdf(double.NaN));
        Assert.Equal(double.NaN, d.InverseLogSf(double.NaN));
    }

    [Theory]
    [MemberData(nameof(Names))]
    public void ProbabilitiesOutsideTheDomainGiveNaN(string name)
    {
        IContinuousDistribution d = Create(name);
        foreach (double p in new[] { -0.1, -double.Epsilon, 1.0000000000000002, double.PositiveInfinity, double.NegativeInfinity })
        {
            Assert.Equal(double.NaN, d.InverseCdf(p));
            Assert.Equal(double.NaN, d.InverseSf(p));
        }

        foreach (double logP in new[] { double.Epsilon, 0.5, double.PositiveInfinity })
        {
            Assert.Equal(double.NaN, d.InverseLogCdf(logP));
            Assert.Equal(double.NaN, d.InverseLogSf(logP));
        }
    }

    [Theory]
    [MemberData(nameof(Names))]
    public void InversesReturnTheSupportAtTheEndpoints(string name)
    {
        IContinuousDistribution d = Create(name);
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
        IContinuousDistribution d = Create(name);
        foreach (double below in new[] { double.NegativeInfinity, d.SupportMinimum - 1 })
        {
            if (double.IsFinite(d.SupportMinimum) || double.IsNegativeInfinity(below))
            {
                Assert.Equal(0.0, d.Pdf(below));
                Assert.Equal(double.NegativeInfinity, d.LogPdf(below));
                Assert.Equal(0.0, d.Cdf(below));
                Assert.Equal(1.0, d.Sf(below));
                Assert.Equal(double.NegativeInfinity, d.LogCdf(below));
                Assert.Equal(0.0, d.LogSf(below));
            }
        }

        foreach (double above in new[] { double.PositiveInfinity, d.SupportMaximum + 1 })
        {
            if (double.IsFinite(d.SupportMaximum) || double.IsPositiveInfinity(above))
            {
                Assert.Equal(0.0, d.Pdf(above));
                Assert.Equal(double.NegativeInfinity, d.LogPdf(above));
                Assert.Equal(1.0, d.Cdf(above));
                Assert.Equal(0.0, d.Sf(above));
                Assert.Equal(0.0, d.LogCdf(above));
                Assert.Equal(double.NegativeInfinity, d.LogSf(above));
            }
        }
    }

    [Theory]
    [MemberData(nameof(Names))]
    public void MedianIsTheHalfQuantile(string name)
    {
        IContinuousDistribution d = Create(name);
        double half = d.InverseCdf(0.5);
        Assert.Equal(half, d.Median, Math.Abs(half) * 1e-14);
    }

    [Theory]
    [MemberData(nameof(Names))]
    public void InversesRoundTrip(string name)
    {
        IContinuousDistribution d = Create(name);
        foreach (double p in new[] { 1e-300, 1e-12, 0.01, 0.3, 0.5, 0.8, 0.999 })
        {
            double x = d.InverseCdf(p);
            double y = d.InverseSf(p);
            Assert.Equal(x, d.InverseLogCdf(Math.Log(p)), Math.Abs(x) * 1e-12);
            Assert.Equal(y, d.InverseLogSf(Math.Log(p)), Math.Abs(y) * 1e-12);

            // x の丸め（相対 ε/2）は確率を |x|·f(x)·ε/2 だけ動かす。台の端に丸められた点（一様分布の確率 1e-300 など）は除く
            if (x > d.SupportMinimum && x < d.SupportMaximum)
            {
                Assert.Equal(p, d.Cdf(x), p * 1e-12 + Math.Abs(x) * d.Pdf(x) * 4e-16);
            }

            if (y > d.SupportMinimum && y < d.SupportMaximum)
            {
                Assert.Equal(p, d.Sf(y), p * 1e-12 + Math.Abs(y) * d.Pdf(y) * 4e-16);
            }
        }
    }

    [Theory]
    [MemberData(nameof(Names))]
    public void SampleRejectsNullRandomSource(string name)
    {
        IContinuousDistribution d = Create(name);
        Assert.Throws<ArgumentNullException>(() => d.Sample(null!));
        Assert.Throws<ArgumentNullException>(() => d.Sample(null!, new double[1]));
    }

    [Fact]
    public void ConstructorsValidateParameters()
    {
        double[] notFinite = [double.NaN, double.PositiveInfinity, double.NegativeInfinity];
        double[] notPositive = [0, -1, double.NaN, double.PositiveInfinity];
        foreach (double v in notFinite)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new NormalDistribution(v, 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => new LogNormalDistribution(v, 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => new UniformDistribution(v, 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => new UniformDistribution(0, v));
            Assert.Throws<ArgumentOutOfRangeException>(() => new CauchyDistribution(v, 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => new LaplaceDistribution(v, 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => new LogisticDistribution(v, 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => new StudentTDistribution(3, v, 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => new GumbelDistribution(v, 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => new LocationScaleDistribution<NormalDistribution>(NormalDistribution.Standard, v, 1));
        }

        foreach (double v in notPositive)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new NormalDistribution(0, v));
            Assert.Throws<ArgumentOutOfRangeException>(() => new LogNormalDistribution(0, v));
            Assert.Throws<ArgumentOutOfRangeException>(() => new ExponentialDistribution(v));
            Assert.Throws<ArgumentOutOfRangeException>(() => ExponentialDistribution.FromScale(v));
            Assert.Throws<ArgumentOutOfRangeException>(() => new CauchyDistribution(0, v));
            Assert.Throws<ArgumentOutOfRangeException>(() => new LaplaceDistribution(0, v));
            Assert.Throws<ArgumentOutOfRangeException>(() => new LogisticDistribution(0, v));
            Assert.Throws<ArgumentOutOfRangeException>(() => new GammaDistribution(v, 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => new GammaDistribution(1, v));
            Assert.Throws<ArgumentOutOfRangeException>(() => GammaDistribution.FromShapeRate(1, v));
            Assert.Throws<ArgumentOutOfRangeException>(() => new ChiSquaredDistribution(v));
            Assert.Throws<ArgumentOutOfRangeException>(() => new BetaDistribution(v, 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => new BetaDistribution(1, v));
            Assert.Throws<ArgumentOutOfRangeException>(() => new StudentTDistribution(3, 0, v));
            Assert.Throws<ArgumentOutOfRangeException>(() => new WeibullDistribution(v, 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => new WeibullDistribution(1, v));
            Assert.Throws<ArgumentOutOfRangeException>(() => new GumbelDistribution(0, v));
            Assert.Throws<ArgumentOutOfRangeException>(() => new InverseGammaDistribution(v, 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => new InverseGammaDistribution(1, v));
            Assert.Throws<ArgumentOutOfRangeException>(() => new FDistribution(v, 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => new FDistribution(1, v));
            Assert.Throws<ArgumentOutOfRangeException>(() => new LocationScaleDistribution<NormalDistribution>(NormalDistribution.Standard, 0, v));
        }

        Assert.Throws<ArgumentNullException>(() => new LocationScaleDistribution<IContinuousDistribution>(null!, 0, 1));

        // 切断の区間は a < b で、元の分布の台と交わり、確率が 0 でない（ln Z が -∞ にならない）こと
        var normal = NormalDistribution.Standard;
        Assert.Throws<ArgumentNullException>(() => new TruncatedDistribution<IContinuousDistribution>(null!, 0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TruncatedDistribution<NormalDistribution>(normal, double.NaN, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TruncatedDistribution<NormalDistribution>(normal, 0, double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TruncatedDistribution<NormalDistribution>(normal, 1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TruncatedDistribution<NormalDistribution>(normal, 2, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TruncatedDistribution<NormalDistribution>(normal, double.PositiveInfinity, double.PositiveInfinity));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TruncatedDistribution<GammaDistribution>(new GammaDistribution(2, 1), -2, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TruncatedDistribution<UniformDistribution>(new UniformDistribution(0, 1), 1, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TruncatedDistribution<WeibullDistribution>(new WeibullDistribution(10, 1), 1e40, double.PositiveInfinity));
        _ = new TruncatedDistribution<NormalDistribution>(normal, 1e10, double.PositiveInfinity);

        // t 分布の自由度だけは +∞ を許す
        Assert.Throws<ArgumentOutOfRangeException>(() => new StudentTDistribution(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new StudentTDistribution(double.NaN));
        _ = new StudentTDistribution(double.PositiveInfinity);

        // 一様分布は幅が正で有限
        Assert.Throws<ArgumentOutOfRangeException>(() => new UniformDistribution(1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new UniformDistribution(2, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new UniformDistribution(-double.MaxValue, double.MaxValue));
    }

    [Fact]
    public void MomentsThatDoNotExistAreNaNOrInfinite()
    {
        var cauchy = new CauchyDistribution(0, 1);
        Assert.Equal(double.NaN, cauchy.Mean);
        Assert.Equal(double.NaN, cauchy.Variance);
        Assert.Equal(double.NaN, cauchy.StandardDeviation);
        Assert.Equal(double.NaN, cauchy.Skewness);
        Assert.Equal(double.NaN, cauchy.ExcessKurtosis);

        var heavy = new StudentTDistribution(0.5);
        Assert.Equal(double.NaN, heavy.Mean);
        Assert.Equal(double.NaN, heavy.Variance);

        var t = new StudentTDistribution(1.5);
        Assert.Equal(0.0, t.Mean);
        Assert.Equal(double.PositiveInfinity, t.Variance);
        Assert.Equal(double.PositiveInfinity, t.StandardDeviation);
        Assert.Equal(double.NaN, t.Skewness);
        Assert.Equal(double.NaN, t.ExcessKurtosis);

        var t3 = new StudentTDistribution(3);
        Assert.Equal(3.0, t3.Variance);
        Assert.Equal(double.NaN, t3.Skewness);
        Assert.Equal(double.PositiveInfinity, t3.ExcessKurtosis);

        // 台が正の分布は、右の裾で発散するモーメントが +∞。平均が発散すると分散などは存在しない
        AssertMoments(new InverseGammaDistribution(0.8, 1), double.PositiveInfinity, double.NaN, double.NaN, double.NaN);
        AssertMoments(new InverseGammaDistribution(1.5, 1), 2, double.PositiveInfinity, double.NaN, double.NaN);
        AssertMoments(new InverseGammaDistribution(2.5, 1.5), 1, 2, double.PositiveInfinity, double.PositiveInfinity);
        AssertMoments(new InverseGammaDistribution(3.5, 2.5), 1, 2.0 / 3, 4 * Math.Sqrt(1.5) / 0.5, double.PositiveInfinity);
        AssertMoments(new FDistribution(5, 2), double.PositiveInfinity, double.NaN, double.NaN, double.NaN);
        AssertMoments(new FDistribution(5, 3), 3, double.PositiveInfinity, double.NaN, double.NaN);
        AssertMoments(new FDistribution(4, 6), 1.5, 2 * 1.5 * 1.5 * 8 / 4 / 2, double.PositiveInfinity, double.NaN);
        AssertMoments(new FDistribution(4, 8), 4.0 / 3, 2 * 64 * 10.0 / (4 * 36 * 4), 14 * Math.Sqrt(32) / (2 * Math.Sqrt(40)), double.PositiveInfinity);

        // 切断は、区間が無限の側で元の分布のモーメントが発散すれば、その側の無限大。有界な区間ならすべて有限
        AssertMoments(Truncate(new CauchyDistribution(0, 1), 0, double.PositiveInfinity), double.PositiveInfinity, double.NaN, double.NaN, double.NaN);
        AssertMoments(Truncate(new CauchyDistribution(0, 1), double.NegativeInfinity, 0), double.NegativeInfinity, double.NaN, double.NaN, double.NaN);
        AssertMoments(Truncate(new InverseGammaDistribution(0.8, 1), 1, double.PositiveInfinity), double.PositiveInfinity, double.NaN, double.NaN, double.NaN);
        var tLeft = Truncate(new StudentTDistribution(1.5), double.NegativeInfinity, 0);
        Assert.True(double.IsFinite(tLeft.Mean) && tLeft.Mean < 0);
        Assert.Equal(double.PositiveInfinity, tLeft.Variance);
        Assert.Equal(double.NaN, tLeft.Skewness);
        var tRight = Truncate(new StudentTDistribution(3), 1, double.PositiveInfinity);
        Assert.True(double.IsFinite(tRight.Variance) && tRight.Variance > 0);
        Assert.Equal(double.PositiveInfinity, tRight.Skewness);
        Assert.Equal(double.PositiveInfinity, tRight.ExcessKurtosis);
        var tLeft3 = Truncate(new StudentTDistribution(3), double.NegativeInfinity, -1);
        Assert.Equal(double.NegativeInfinity, tLeft3.Skewness);
        var bounded = Truncate(new CauchyDistribution(0, 1), -1, 3);
        Assert.True(double.IsFinite(bounded.Mean) && double.IsFinite(bounded.Variance) && double.IsFinite(bounded.Skewness)
            && double.IsFinite(bounded.ExcessKurtosis));

        static TruncatedDistribution<IContinuousDistribution> Truncate(IContinuousDistribution d, double a, double b) => new(d, a, b);

        static void AssertMoments(IContinuousDistribution d, double mean, double variance, double skewness, double excessKurtosis)
        {
            string name = d.GetType().Name;
            Assert.True(Close(mean, d.Mean), $"{name} mean {d.Mean}");
            Assert.True(Close(variance, d.Variance), $"{name} variance {d.Variance}");
            Assert.True(Close(Math.Sqrt(variance), d.StandardDeviation), $"{name} standard deviation {d.StandardDeviation}");
            Assert.True(Close(skewness, d.Skewness), $"{name} skewness {d.Skewness}");
            Assert.True(excessKurtosis.Equals(d.ExcessKurtosis), $"{name} excess kurtosis {d.ExcessKurtosis}");
        }

        static bool Close(double expected, double actual) =>
            expected.Equals(actual) || Math.Abs(actual - expected) <= 4e-15 * Math.Abs(expected);
    }

    /// <summary>
    /// 逆ガンマ分布は 1/X がガンマ分布、F 分布は d₁X/(d₁X + d₂) がベータ分布、ガンベル分布は e^(-(X - μ)/β) が指数分布
    /// </summary>
    [Fact]
    public void NewDistributionsAgreeWithTheirDefinitions()
    {
        var inverseGamma = new InverseGammaDistribution(2.5, 3);
        var gamma = GammaDistribution.FromShapeRate(2.5, 3);
        foreach (double x in new[] { 0.05, 0.4, 1.0, 2.5, 30.0 })
        {
            Assert.Equal(gamma.Sf(1 / x), inverseGamma.Cdf(x), 1e-15);
            Assert.Equal(gamma.Cdf(1 / x), inverseGamma.Sf(x), 1e-15);
            Assert.Equal(gamma.Pdf(1 / x) / (x * x), inverseGamma.Pdf(x), 1e-14 * inverseGamma.Pdf(x));
        }

        foreach (double p in new[] { 1e-10, 0.2, 0.5, 0.9 })
        {
            Assert.Equal(1 / gamma.InverseSf(p), inverseGamma.InverseCdf(p), 1e-14 * inverseGamma.InverseCdf(p));
        }

        var f = new FDistribution(3, 7);
        var beta = new BetaDistribution(1.5, 3.5);
        foreach (double x in new[] { 0.01, 0.5, 1.0, 3.0, 20.0 })
        {
            double u = 3 * x / (3 * x + 7);
            Assert.Equal(beta.Cdf(u), f.Cdf(x), 1e-15);
            Assert.Equal(beta.Pdf(u) * 21 / ((3 * x + 7) * (3 * x + 7)), f.Pdf(x), 1e-14 * f.Pdf(x));
        }

        // F(1, ν) は t(ν) の2乗
        var f1 = new FDistribution(1, 4.5);
        var t = new StudentTDistribution(4.5);
        foreach (double a in new[] { 0.1, 1.0, 3.0, 50.0 })
        {
            Assert.Equal(2 * t.Sf(a), f1.Sf(a * a), 1e-15);
        }

        var gumbel = new GumbelDistribution(1.5, 2);
        var exponential = new ExponentialDistribution(1);
        foreach (double x in new[] { -5.0, 0.0, 1.5, 4.0, 30.0 })
        {
            double e = Math.Exp(-(x - 1.5) / 2);
            Assert.Equal(exponential.Sf(e), gumbel.Cdf(x), 1e-15);
            Assert.Equal(exponential.Cdf(e), gumbel.Sf(x), 1e-15);
        }

        Assert.Equal(1.5 + 2 * XMath.Special.EulerGamma, gumbel.Mean, 1e-15);
        Assert.Equal(gumbel.InverseCdf(0.5), gumbel.Median, 1e-15);
    }

    /// <summary>位置・尺度の変換は、位置と尺度を持つ分布の型と同じ値になる。モーメントと乱数も μ + σY</summary>
    [Fact]
    public void LocationScaleMatchesBuiltInLocationAndScale()
    {
        var shifted = new LocationScaleDistribution<StudentTDistribution>(new StudentTDistribution(3.5), -2, 0.75);
        var t = new StudentTDistribution(3.5, -2, 0.75);
        Assert.Equal(-2.0, shifted.Location);
        Assert.Equal(0.75, shifted.Scale);
        Assert.Equal(3.5, shifted.Distribution.DegreesOfFreedom);
        foreach (double x in new[] { -1e10, -30.0, -2.5, -2.0, -1.0, 4.0, 1e20 })
        {
            // t 分布の密度は exp(LogPdf) なので、経路が違うと相対誤差 |ln f|·ε 程度ずれる
            Assert.Equal(t.Pdf(x), shifted.Pdf(x), 1e-15 * (1 - t.LogPdf(x)) * t.Pdf(x));
            Assert.Equal(t.LogPdf(x), shifted.LogPdf(x), 1e-14 * Math.Abs(t.LogPdf(x)));
            Assert.Equal(t.Cdf(x), shifted.Cdf(x), 1e-15 * t.Cdf(x));
            Assert.Equal(t.Sf(x), shifted.Sf(x), 1e-15 * t.Sf(x));
            Assert.Equal(t.LogCdf(x), shifted.LogCdf(x), 1e-14 * Math.Abs(t.LogCdf(x)));
            Assert.Equal(t.LogSf(x), shifted.LogSf(x), 1e-14 * Math.Abs(t.LogSf(x)));
        }

        foreach (double p in new[] { 1e-100, 0.01, 0.5, 0.9 })
        {
            Assert.Equal(t.InverseCdf(p), shifted.InverseCdf(p), 1e-14 * Math.Abs(t.InverseCdf(p)));
            Assert.Equal(t.InverseSf(p), shifted.InverseSf(p), 1e-14 * Math.Abs(t.InverseSf(p)));
            Assert.Equal(t.InverseLogCdf(Math.Log(p)), shifted.InverseLogCdf(Math.Log(p)), 1e-14 * Math.Abs(t.InverseCdf(p)));
            Assert.Equal(t.InverseLogSf(Math.Log(p)), shifted.InverseLogSf(Math.Log(p)), 1e-14 * Math.Abs(t.InverseSf(p)));
        }

        Assert.Equal(t.Mean, shifted.Mean);
        Assert.Equal(t.Variance, shifted.Variance, 1e-15 * t.Variance);
        Assert.Equal(t.StandardDeviation, shifted.StandardDeviation, 1e-15 * t.StandardDeviation);
        Assert.Equal(t.Skewness, shifted.Skewness);
        Assert.Equal(t.ExcessKurtosis, shifted.ExcessKurtosis, 1e-15 * t.ExcessKurtosis);
        Assert.Equal(t.Entropy, shifted.Entropy, 1e-15);
        Assert.Equal(t.Median, shifted.Median);

        // 台の端も変換する（ガンマ分布の台 [0, ∞) が [1, ∞) になる）
        var gamma = new LocationScaleDistribution<GammaDistribution>(new GammaDistribution(2, 1), 1, 3);
        Assert.Equal(1.0, gamma.SupportMinimum);
        Assert.Equal(double.PositiveInfinity, gamma.SupportMaximum);
        Assert.Equal(0.0, gamma.Cdf(1));
        Assert.Equal(0.0, gamma.Pdf(0.5));

        var a = new Philox4x32Random(11);
        var b = new Philox4x32Random(11);
        var standard = new GammaDistribution(2, 1);
        for (int i = 0; i < 100; i++)
        {
            Assert.Equal(1 + 3 * standard.Sample(a), gamma.Sample(b));
        }
    }

    /// <summary>
    /// 切断の閉形式: 半正規分布、指数分布の無記憶性（[a, ∞) に切断すると a + 元の分布）、一様分布の部分区間。
    /// 区間が台全体なら元の分布と同じ値で、乱数は逆関数法
    /// </summary>
    [Fact]
    public void TruncationAgreesWithClosedForms()
    {
        var normal = NormalDistribution.Standard;
        var half = new TruncatedDistribution<NormalDistribution>(normal, 0, double.PositiveInfinity);
        Assert.Equal(0.0, half.LowerBound);
        Assert.Equal(double.PositiveInfinity, half.UpperBound);
        Assert.Equal(0.5, half.Probability);
        Assert.Equal(-Math.Log(2), half.LogProbability);
        AssertClose(Math.Sqrt(2 / Math.PI), half.Mean, 2e-15);
        AssertClose(1 - 2 / Math.PI, half.Variance, 2e-15);
        // √2(4 - π)/(π - 2)^1.5 と 8(π - 3)/(π - 2)²（double で計算すると π - 3 が桁落ちするので mpmath の値）。
        // 超過尖度は m₄/m₂² - 3 の差で 4 倍程度に拡大する
        AssertClose(0.99527174643115604, half.Skewness, 4e-15);
        AssertClose(0.86917730360597412, half.ExcessKurtosis, 2e-14);
        AssertClose(0.5 * Math.Log(Math.PI * Math.E / 2), half.Entropy, 2e-15);
        AssertClose(normal.InverseCdf(0.75), half.Median, 1e-15);
        foreach (double x in new[] { 1e-300, 1e-20, 1e-3, 0.3, 2.0, 9.0 })
        {
            // 0 の近くでも F(x) - F(0) の桁落ちなしに求まる
            AssertClose(XMath.Special.Erf(x / Math.Sqrt(2)), half.Cdf(x), 1e-15);
            AssertClose(2 * normal.Sf(x), half.Sf(x), 1e-15);
            AssertClose(2 * normal.Pdf(x), half.Pdf(x), 1e-15);
        }

        // 端の近くの分位点（F(0) + pZ は 1/2 に丸まる）も相対精度を保つ: x ≈ p·√(π/2)
        foreach (double p in new[] { 1e-300, 1e-20 })
        {
            AssertClose(p * Math.Sqrt(Math.PI / 2), half.InverseCdf(p), 1e-15);
            double logP = Math.Log(p);
            AssertClose(Math.Exp(logP) * Math.Sqrt(Math.PI / 2), half.InverseLogCdf(logP), 1e-15);
        }

        var exponential = new ExponentialDistribution(2);
        var tail = new TruncatedDistribution<ExponentialDistribution>(exponential, 3, double.PositiveInfinity);
        AssertClose(Math.Exp(-6), tail.Probability, 1e-15);
        foreach (double x in new[] { 3 + 1e-12, 3.1, 4.0, 10.0, 300.0 })
        {
            AssertClose(exponential.Cdf(x - 3), tail.Cdf(x), 2e-15);
            AssertClose(exponential.Sf(x - 3), tail.Sf(x), 2e-15);
            AssertClose(exponential.Pdf(x - 3), tail.Pdf(x), 2e-15);
            AssertClose(exponential.LogSf(x - 3), tail.LogSf(x), 2e-15);
        }

        foreach (double p in new[] { 1e-300, 1e-10, 0.3, 0.9 })
        {
            AssertClose(3 + exponential.InverseCdf(p), tail.InverseCdf(p), 1e-15);
            AssertClose(3 + exponential.InverseSf(p), tail.InverseSf(p), 1e-15);
        }

        AssertClose(3.5, tail.Mean, 1e-15);
        AssertClose(0.25, tail.Variance, 1e-15);
        AssertClose(2, tail.Skewness, 1e-14);
        AssertClose(6, tail.ExcessKurtosis, 1e-14);
        AssertClose(1 - Math.Log(2), tail.Entropy, 1e-15);

        var part = new TruncatedDistribution<UniformDistribution>(new UniformDistribution(0, 10), 2, 5);
        var uniform = new UniformDistribution(2, 5);
        Assert.Equal(2.0, part.SupportMinimum);
        Assert.Equal(5.0, part.SupportMaximum);
        foreach (double x in new[] { 2.5, 3.0, 4.75 })
        {
            AssertClose(uniform.Cdf(x), part.Cdf(x), 1e-15);
            AssertClose(uniform.Pdf(x), part.Pdf(x), 1e-15);
        }

        AssertClose(3.5, part.Mean, 1e-15);
        AssertClose(0.75, part.Variance, 1e-15);
        Assert.Equal(0, part.Skewness, 1e-14);
        AssertClose(-1.2, part.ExcessKurtosis, 1e-14);
        AssertClose(Math.Log(3), part.Entropy, 1e-15);

        // 台の端で密度が発散する: Beta(1, 0.1) を [1/2, 1] に切断すると 1 - V/2、Beta(0.1, 1) を [0, 1/2] に切断すると V/2（V ~ Beta(0.1, 1)）。
        // 1 の近くでは点の丸めで密度が大きくずれる（0.1 の冪では、1 から 1e-16 以内に確率の 2.5 % がある）
        var power = new BetaDistribution(0.1, 1);
        var nearOne = new TruncatedDistribution<BetaDistribution>(new BetaDistribution(1, 0.1), 0.5, 1);
        var nearZero = new TruncatedDistribution<BetaDistribution>(power, 0, 0.5);
        AssertClose(1 - 0.5 * power.Mean, nearOne.Mean, 2e-15);
        AssertClose(0.5 * power.Mean, nearZero.Mean, 2e-15);
        foreach (var reflected in new[] { nearOne, nearZero })
        {
            AssertClose(0.25 * power.Variance, reflected.Variance, 4e-15);
            AssertClose(power.ExcessKurtosis, reflected.ExcessKurtosis, 1e-14);
            AssertClose(power.Entropy - Math.Log(2), reflected.Entropy, 4e-15);
        }

        AssertClose(-power.Skewness, nearOne.Skewness, 4e-15);
        AssertClose(power.Skewness, nearZero.Skewness, 4e-15);

        // 区間が台全体なら元の分布と同じ（ガンマ分布の台の外の下限は台の端 0 に丸める）
        var gamma = new GammaDistribution(2.5, 1.5);
        var whole = new TruncatedDistribution<GammaDistribution>(gamma, -5, double.PositiveInfinity);
        Assert.Equal(0.0, whole.SupportMinimum);
        Assert.Equal(1.0, whole.Probability);
        Assert.Equal(gamma.Mean, whole.Mean);
        Assert.Equal(gamma.Variance, whole.Variance);
        Assert.Equal(gamma.Skewness, whole.Skewness);
        Assert.Equal(gamma.ExcessKurtosis, whole.ExcessKurtosis);
        Assert.Equal(gamma.Entropy, whole.Entropy);
        foreach (double x in new[] { 1e-10, 0.5, 4.0, 40.0 })
        {
            Assert.Equal(gamma.Pdf(x), whole.Pdf(x));
            Assert.Equal(gamma.Cdf(x), whole.Cdf(x));
            Assert.Equal(gamma.Sf(x), whole.Sf(x));
        }

        var a = new Philox4x32Random(5);
        var b = new Philox4x32Random(5);
        for (int i = 0; i < 100; i++)
        {
            Assert.Equal(tail.InverseCdf(a.NextDoubleOpen()), tail.Sample(b));
        }

        static void AssertClose(double expected, double actual, double relative) =>
            Assert.True(Math.Abs(actual - expected) <= relative * Math.Abs(expected), $"expected {expected:R}, actual {actual:R}");
    }

    [Fact]
    public void InfiniteDegreesOfFreedomIsTheNormalDistribution()
    {
        var t = new StudentTDistribution(double.PositiveInfinity, -1, 3);
        var normal = new NormalDistribution(-1, 3);
        foreach (double x in new[] { -200.0, -7.0, -1.0, 0.5, 4.0, 60.0 })
        {
            Assert.Equal(normal.Pdf(x), t.Pdf(x));
            Assert.Equal(normal.LogPdf(x), t.LogPdf(x));
            Assert.Equal(normal.Cdf(x), t.Cdf(x));
            Assert.Equal(normal.Sf(x), t.Sf(x));
            Assert.Equal(normal.LogCdf(x), t.LogCdf(x));
            Assert.Equal(normal.LogSf(x), t.LogSf(x));
        }

        foreach (double p in new[] { 1e-200, 0.01, 0.5, 0.9 })
        {
            Assert.Equal(normal.InverseCdf(p), t.InverseCdf(p));
            Assert.Equal(normal.InverseSf(p), t.InverseSf(p));
            Assert.Equal(normal.InverseLogCdf(Math.Log(p)), t.InverseLogCdf(Math.Log(p)));
            Assert.Equal(normal.InverseLogSf(Math.Log(p)), t.InverseLogSf(Math.Log(p)));
        }

        Assert.Equal(normal.Mean, t.Mean);
        Assert.Equal(normal.Variance, t.Variance);
        Assert.Equal(0.0, t.ExcessKurtosis);
        Assert.Equal(normal.Entropy, t.Entropy);

        var a = new Philox4x32Random(5);
        var b = new Philox4x32Random(5);
        for (int i = 0; i < 100; i++)
        {
            Assert.Equal(normal.Sample(a), t.Sample(b));
        }
    }

    [Fact]
    public void ChiSquaredIsGammaWithHalfTheDegreesOfFreedom()
    {
        var chi = new ChiSquaredDistribution(7);
        var gamma = new GammaDistribution(3.5, 2);
        Assert.Equal(7.0, chi.DegreesOfFreedom);
        foreach (double x in new[] { 1e-8, 0.5, 7.0, 30.0 })
        {
            Assert.Equal(gamma.Pdf(x), chi.Pdf(x));
            Assert.Equal(gamma.Cdf(x), chi.Cdf(x));
            Assert.Equal(gamma.LogSf(x), chi.LogSf(x));
        }

        Assert.Equal(gamma.InverseCdf(0.3), chi.InverseCdf(0.3));
        Assert.Equal(gamma.Entropy, chi.Entropy);
    }

    [Fact]
    public void AlternativeParameterizationsAgree()
    {
        var byScale = ExponentialDistribution.FromScale(4);
        Assert.Equal(0.25, byScale.Rate);
        Assert.Equal(4.0, byScale.Scale);

        var byRate = GammaDistribution.FromShapeRate(2.5, 4);
        Assert.Equal(0.25, byRate.Scale);
        Assert.Equal(4.0, byRate.Rate);
        Assert.Equal(new GammaDistribution(2.5, 0.25).Cdf(1.3), byRate.Cdf(1.3));

        Assert.Equal(new NormalDistribution(0, 1).Cdf(0.7), NormalDistribution.Standard.Cdf(0.7));
    }

    [Fact]
    public void SpecialCasesAgreeWithSimplerDistributions()
    {
        // t(1) = コーシー、ワイブル(1, λ) = 指数(1/λ)、ガンマ(1, θ) = 指数(1/θ)、ベータ(1, 1) = 一様
        var t1 = new StudentTDistribution(1, 2, 3);
        var cauchy = new CauchyDistribution(2, 3);
        var weibull = new WeibullDistribution(1, 4);
        var gamma = new GammaDistribution(1, 4);
        var exponential = new ExponentialDistribution(0.25);
        var beta = new BetaDistribution(1, 1);
        var uniform = new UniformDistribution(0, 1);
        foreach (double x in new[] { 1e-6, 0.3, 1.0, 5.0, 40.0 })
        {
            Assert.Equal(cauchy.Cdf(x), t1.Cdf(x), 1e-15);
            Assert.Equal(cauchy.Pdf(x), t1.Pdf(x), 1e-15);
            Assert.Equal(exponential.Cdf(x), weibull.Cdf(x), 1e-15);
            Assert.Equal(exponential.LogPdf(x), weibull.LogPdf(x), 1e-14);
            Assert.Equal(exponential.Sf(x), gamma.Sf(x), 1e-15);
        }

        foreach (double u in new[] { 1e-9, 0.25, 0.75 })
        {
            Assert.Equal(uniform.Cdf(u), beta.Cdf(u), 1e-15);
            Assert.Equal(uniform.InverseSf(u), beta.InverseSf(u), 1e-15);
        }

        Assert.Equal(exponential.Entropy, weibull.Entropy, 1e-15);
        Assert.Equal(exponential.Variance, weibull.Variance, 1e-14);
    }

    [Fact]
    public void DensitiesAtTheBoundaryOfTheSupport()
    {
        // 形状 < 1 で発散、= 1 で有限、> 1 で 0
        Assert.Equal(double.PositiveInfinity, new GammaDistribution(0.5, 1).Pdf(0));
        Assert.Equal(0.5, new GammaDistribution(1, 2).Pdf(0));
        Assert.Equal(0.0, new GammaDistribution(2, 1).Pdf(0));
        Assert.Equal(double.PositiveInfinity, new WeibullDistribution(0.5, 1).Pdf(0));
        Assert.Equal(0.5, new WeibullDistribution(1, 2).Pdf(0));
        Assert.Equal(0.0, new WeibullDistribution(3, 1).Pdf(0));
        Assert.Equal(double.PositiveInfinity, new BetaDistribution(0.5, 2).Pdf(0));
        Assert.Equal(2.0, new BetaDistribution(1, 2).Pdf(0));
        Assert.Equal(0.0, new BetaDistribution(2, 2).Pdf(1));
        Assert.Equal(double.PositiveInfinity, new BetaDistribution(2, 0.5).Pdf(1));
        Assert.Equal(0.0, new LogNormalDistribution(0, 1).Pdf(0));
        Assert.Equal(2.5, new ExponentialDistribution(2.5).Pdf(0));
        Assert.Equal(0.25, new UniformDistribution(-1, 3).Pdf(-1));
        Assert.Equal(0.25, new UniformDistribution(-1, 3).Pdf(3));
    }

    [Fact]
    public void TailsBeyondUnderflowStayFiniteInLogSpace()
    {
        var normal = new NormalDistribution(1.5, 2);
        Assert.Equal(0.0, normal.Sf(80));
        Assert.Equal(-774.870788040974, normal.LogSf(80), 1e-10);
        Assert.Equal(80.0, normal.InverseLogSf(normal.LogSf(80)), 1e-12);

        var t = new StudentTDistribution(3.5);
        Assert.True(double.IsFinite(t.LogSf(1e200)));
        Assert.Equal(1e200, t.InverseLogSf(t.LogSf(1e200)), 1e188);

        // 形状が小さいと台の端の近くに確率が集まる（P(X ≤ 1e-300) ≈ 0.5）
        var gamma = new GammaDistribution(1e-3, 1);
        Assert.InRange(gamma.Cdf(1e-300), 0.4, 0.6);
        Assert.Equal(1e-300, gamma.InverseCdf(gamma.Cdf(1e-300)), 1e-312);
        Assert.Equal(1e-300, gamma.InverseLogCdf(gamma.LogCdf(1e-300)), 1e-312);

        // 対数確率が -700 から ln(最小の正規化数) までは、どの形状も InverseSf（InverseRegularizedGammaQ の深い裾）で求める。
        // 期待値は mpmath 1.4.1 で 60 桁で解いた値（条件数は 2 程度）
        var large = new GammaDistribution(150, 1);
        Assert.Equal(1155.9814164099166, large.InverseLogSf(-705), 1155.98 * 1e-14);
        Assert.Equal(1159.8714898802241, large.InverseLogSf(-708.39), 1159.87 * 1e-14);

        // 確率・(x/λ)ᵏ が非正規化数か0に丸まる裾も対数のまま求める
        var weibull = new WeibullDistribution(2, 1);
        Assert.Equal(2 * Math.Log(1e-308), weibull.LogCdf(1e-308), 1e-12);
        Assert.Equal(1e-308, weibull.InverseLogCdf(2 * Math.Log(1e-308)), 1e-321);
        double logTiny = Math.Log(double.Epsilon) - Math.Log(1e308);
        var uniform = new UniformDistribution(-1e308, 0);
        Assert.Equal(logTiny, uniform.LogSf(-double.Epsilon), 1e-12);
        Assert.Equal(-double.Epsilon, uniform.InverseLogSf(logTiny));
        Assert.Equal(logTiny, new UniformDistribution(0, 1e308).LogCdf(double.Epsilon), 1e-12);
        Assert.Equal(double.Epsilon, new UniformDistribution(0, 1e308).InverseLogCdf(logTiny));
    }

    /// <summary>
    /// 位置 -1e308・尺度 1e308 では x - μ と μ + σz の途中だけが溢れる。標準化した値（z = 2）の分布と同じになる
    /// </summary>
    [Fact]
    public void ExtremeLocationAndScaleDoNotOverflowIntermediates()
    {
        (IContinuousDistribution Extreme, IContinuousDistribution Standard)[] pairs =
        [
            (new NormalDistribution(-1e308, 1e308), new NormalDistribution(0, 1)),
            (new CauchyDistribution(-1e308, 1e308), new CauchyDistribution(0, 1)),
            (new LaplaceDistribution(-1e308, 1e308), new LaplaceDistribution(0, 1)),
            (new LogisticDistribution(-1e308, 1e308), new LogisticDistribution(0, 1)),
            (new StudentTDistribution(4, -1e308, 1e308), new StudentTDistribution(4)),
        ];
        foreach (var (d, standard) in pairs)
        {
            string name = d.GetType().Name;
            Assert.True(standard.Cdf(2) == d.Cdf(1e308), name);
            Assert.True(standard.Sf(2) == d.Sf(1e308), name);
            Assert.True(standard.LogCdf(2) == d.LogCdf(1e308), name);
            Assert.True(standard.LogSf(2) == d.LogSf(1e308), name);
            Assert.Equal(standard.LogPdf(2) - Math.Log(1e308), d.LogPdf(1e308), 1e-12);
            // 密度は非正規化数（t 分布は exp(LogPdf) なので相対 |ln f|·ε 程度の誤差を許す）
            Assert.Equal(standard.Pdf(2) / 1e308, d.Pdf(1e308), Math.Max(4 * double.Epsilon, 1e-12 * standard.Pdf(2) / 1e308));
            Assert.Equal(standard.Pdf(1) / 1e308, d.Pdf(0), Math.Max(4 * double.Epsilon, 1e-12 * standard.Pdf(1) / 1e308));

            // 逆関数は有限の 1e308 に戻る（条件数は |x|/σ 程度なので相対誤差は 1e-15 程度）
            Assert.Equal(1e308, d.InverseCdf(standard.Cdf(2)), 1e294);
            Assert.Equal(1e308, d.InverseSf(standard.Sf(2)), 1e294);
            Assert.Equal(1e308, d.InverseLogCdf(standard.LogCdf(2)), 1e294);
            Assert.Equal(1e308, d.InverseLogSf(standard.LogSf(2)), 1e294);
            Assert.Equal(-1e308, d.InverseCdf(0.5));
            Assert.Equal(double.NegativeInfinity, d.InverseCdf(0));
            Assert.Equal(double.PositiveInfinity, d.InverseCdf(1));
        }

        // 表現できない値は ∞ のまま
        Assert.Equal(double.PositiveInfinity, new NormalDistribution(1e308, 1e308).InverseCdf(0.99));
    }

    /// <summary>
    /// コーシー分布と自由度の小さい t 分布の裾の分位点は、標準化した値が溢れても尺度が小さければ有限。
    /// 期待値は mpmath 1.4.1 で 120 桁で解いた値。対数確率からの逆関数と t 分布は条件数が 700 以上なので相対 1e-12、
    /// コーシー分布の線形の確率からの逆関数は条件数が 2〜5 なので相対 1e-14 で比べる
    /// </summary>
    [Fact]
    public void SmallScaleTailQuantilesDoNotOverflow()
    {
        static void Near(double expected, double actual, double relative) => Assert.Equal(expected, actual, Math.Abs(expected) * relative);

        var cauchy = new CauchyDistribution(0, 1e-10);
        Near(-1.5663077530567481e302, cauchy.InverseLogCdf(-720), 1e-12);
        Near(1.5663077530567481e302, cauchy.InverseLogSf(-720), 1e-12);
        Near(-3.1830988618379166e299, cauchy.InverseCdf(1e-310), 1e-14);
        Near(3.1830988618379166e299, cauchy.InverseSf(1e-310), 1e-14);
        Assert.Equal(double.NegativeInfinity, cauchy.InverseLogCdf(double.NegativeInfinity));
        Assert.Equal(double.NegativeInfinity, cauchy.InverseCdf(0));

        // 尺度の積 3.4e308 は溢れるが、位置を足すと表現できる
        Near(-1.7000020166548533e308, new CauchyDistribution(1.7e308, 1e-10).InverseCdf(9.362e-320), 1e-14);
        // 尺度も確率も最小の非正規化数なら -1/π
        Near(-1 / Math.PI, new CauchyDistribution(0, double.Epsilon).InverseCdf(double.Epsilon), 1e-14);
        Assert.Equal(double.NegativeInfinity, new CauchyDistribution(0, 1).InverseCdf(1e-310));

        Near(-1.5663077530567481e302, new StudentTDistribution(1, 0, 1e-10).InverseLogCdf(-720), 1e-12);
        var heavy = new StudentTDistribution(0.5, 0, 1e-100);
        Near(-2.8040521362235204e246, heavy.InverseLogCdf(-400), 1e-12);
        Near(-1.0284911563163401e299, heavy.InverseCdf(1e-200), 1e-12);
        Near(1.0284911563163401e299, heavy.InverseSf(1e-200), 1e-12);
        Assert.Equal(double.NegativeInfinity, heavy.InverseLogCdf(double.NegativeInfinity));
        Near(1.3930506441411899e234, new StudentTDistribution(2, 0, 1e-200).InverseLogSf(-2000), 1e-12);
        Near(-7.5082859345825310e197, new StudentTDistribution(0.2, 0, 1e-300).InverseCdf(1e-100), 1e-12);
    }

    /// <summary>
    /// 極小の形状では Q(k, y) ≈ k·E₁(y)。y = x/θ が正規化数で表せなくても ln Γ(1 + k) ≈ -γk を失わない
    /// </summary>
    [Fact]
    public void TinyGammaShapeKeepsTheEulerTerm()
    {
        double k = 1e-308;
        var gamma = new GammaDistribution(k, 1e308);
        double logY = Math.Log(1e-308) - Math.Log(1e308);
        double expected = k * (-logY - XMath.Special.EulerGamma);
        Assert.Equal(expected, gamma.Sf(1e-308), expected * 1e-14);
        Assert.Equal(-expected, gamma.LogCdf(1e-308), expected * 1e-14);
        Assert.Equal(Math.Log(expected), gamma.LogSf(1e-308), 1e-12);
        Assert.Equal(1e-308, gamma.InverseSf(expected), 1e-308 * 1e-11);
        Assert.Equal(1e-308, gamma.InverseLogSf(Math.Log(expected)), 1e-308 * 1e-8);

        // 分位点 x/θ が正規化数で小さい範囲（logQ ≈ -703）。y について線形に解くと 192·2⁻⁴⁰⁰ ≈ 7.4e-119 で止まっていた。
        // 期待値は mpmath 1.4.1 で Q = k·E₁(y)（k が極小なので O(k) の項は相対 1e-300 以下）を 60 桁で解いた値。
        // 条件数は |logQ·(ln y + γ)| ≈ 3e5 なので、許容誤差を相対 1e-9 にする
        Assert.Equal(1e100, gamma.InverseLogSf(-703.0258440439491), 1e100 * 1e-9);
        Assert.Equal(1.379490e72, gamma.InverseLogSf(-702.9), 1.379490e72 * 1e-6);
        Assert.Equal(5.611174e190, gamma.InverseLogSf(-703.6), 5.611174e190 * 1e-6);

        // y = 1 では Q = k·E₁(1)
        double e1 = 0.21938393439552027 * k;
        Assert.Equal(e1, gamma.Sf(1e308), e1 * 1e-14);
        Assert.Equal(1e308, gamma.InverseSf(e1), 1e308 * 1e-13);
        Assert.Equal(1.0, XMath.Special.InverseRegularizedGammaQ(k, e1), 1e-13);
    }
}
