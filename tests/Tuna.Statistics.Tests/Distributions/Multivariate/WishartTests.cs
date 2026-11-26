using Tuna.LinearAlgebra;
using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Statistics;

// ウィシャート分布と逆ウィシャート分布: 1次元のガンマ分布・逆ガンマ分布との一致、逆行列の関係、
// 大きな自由度での安定性、モーメントの規約、評価点の台の判定、パラメータの検証。
public class WishartTests
{
    private static ReadOnlyMatrixSpan<double> Square(params double[] values) => MultivariateContractTests.Square(values);

    [Theory]
    [InlineData(3.5, 2.0, 5.0)]
    [InlineData(3.5, 2.0, 0.01)]
    [InlineData(0.5, 1.0, 1e-5)]
    [InlineData(1e12, 2.0, 2e12)]
    [InlineData(1e12, 2.0, 2.000004e12)]
    public void OneDimensionalWishartIsGamma(double nu, double variance, double x)
    {
        // W₁(ν, σ²) = Gamma(ν/2, 2σ²)
        // 許容誤差には x の丸め（ウィシャート分布はコレスキー因子 √x を経由する）が条件数 |x ∂ln p/∂x| = |k - 1 - x/θ| 倍に拡大される分を含める
        var gamma = new GammaDistribution(nu / 2, 2 * variance);
        var wishart = new WishartDistribution(nu, Square(variance));
        double expected = gamma.LogPdf(x);
        double kappa = Math.Abs(nu / 2 - 1 - x / (2 * variance)) / Math.Abs(expected);
        DistributionAssert.Close(expected, wishart.LogPdf(Square(x)), 1e-14 + 8e-16 * kappa, "LogPdf");
    }

    [Theory]
    [InlineData(3.5, 2.0, 5.0)]
    [InlineData(3.5, 2.0, 0.05)]
    [InlineData(0.5, 1.0, 1e5)]
    [InlineData(1e12, 2.0, 2e-12)]
    [InlineData(1e12, 2.0, 2.000004e-12)]
    public void OneDimensionalInverseWishartIsInverseGamma(double nu, double scale, double x)
    {
        // W⁻¹₁(ν, ψ) = InverseGamma(ν/2, ψ/2)
        var inverseGamma = new InverseGammaDistribution(nu / 2, scale / 2);
        var inverseWishart = new InverseWishartDistribution(nu, Square(scale));
        double expected = inverseGamma.LogPdf(x);
        double kappa = Math.Abs(nu / 2 + 1 - scale / (2 * x)) / Math.Abs(expected);
        DistributionAssert.Close(expected, inverseWishart.LogPdf(Square(x)), 1e-14 + 8e-16 * kappa, "LogPdf");
    }

    private static double[] Inverse2(double[] a)
    {
        double determinant = a[0] * a[3] - a[1] * a[2];
        return [a[3] / determinant, -a[1] / determinant, -a[2] / determinant, a[0] / determinant];
    }

    [Theory]
    [InlineData(2.5, 0.5, 1.0)]
    [InlineData(7.0, 1.0, -3.0)]
    [InlineData(1.2, 4.0, 0.25)]
    public void InverseWishartDensityIsTheWishartDensityOfTheInverse(double nu, double a, double b)
    {
        // X ~ W⁻¹(ν, Ψ) ⇔ X⁻¹ ~ W(ν, Ψ⁻¹)、p_X(X) = p_{X⁻¹}(X⁻¹) |X|^-(p+1)
        double[] psi = [2.0, 0.5, 0.5, 1.0];
        double[] x = [a + 1, b / 4, b / 4, a];
        double logDeterminant = Math.Log(x[0] * x[3] - x[1] * x[2]);
        var inverseWishart = new InverseWishartDistribution(nu, Square(psi));
        var wishart = new WishartDistribution(nu, Square(Inverse2(psi)));
        double expected = wishart.LogPdf(Square(Inverse2(x))) - 3 * logDeterminant;
        DistributionAssert.Close(expected, inverseWishart.LogPdf(Square(x)), 1e-13, "LogPdf");
    }

    [Fact]
    public void LargeDegreesOfFreedomStayAccurate()
    {
        // X = νΣ（最頻値の近く）。ν ln|X| と ln Γₚ(ν/2) は 1e13 程度で、密度の対数は数十。
        // 対角の Σ・X での密度を、Γ₂(a) = √π Γ(a) Γ(a - 1/2) を使ってガンマ分布の密度で書き直したものと比べる
        double nu = 1e12;
        var wishart = new WishartDistribution(nu, Square(2, 0, 0, 0.5));
        double x0 = 2e12;
        double x1 = 0.5e12;
        double expected = new GammaDistribution(nu / 2, 4).LogPdf(x0) + new GammaDistribution((nu - 1) / 2, 1).LogPdf(x1)
            - 0.5 * Math.Log(x0) - 0.5 * Math.Log(Math.PI);
        double actual = wishart.LogPdf(Square(x0, 0, 0, x1));
        Assert.True(double.IsFinite(actual));
        Assert.True(Math.Abs(actual - expected) <= 1e-12 * Math.Abs(expected), $"{actual} vs {expected}");
    }

    [Fact]
    public void MomentsFollowTheConventions()
    {
        var scale = Square(MultivariateContractTests.Covariance3);
        double[] s = MultivariateContractTests.Covariance3;
        var wishart = new WishartDistribution(4.5, scale);
        var mean = wishart.GetMean();
        var variance = wishart.GetVariance();
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Assert.Equal(4.5 * s[i + 3 * j], mean[i, j]);
                DistributionAssert.Close(4.5 * (s[i + 3 * j] * s[i + 3 * j] + s[i + 3 * i] * s[j + 3 * j]), variance[i, j], 1e-15, "Variance");
            }
        }

        // 逆ウィシャート: ν > p + 3 で平均と分散、p + 1 < ν ≤ p + 3 は分散が +∞、ν ≤ p + 1 は平均の対角が +∞・非対角が NaN
        var inverse = new InverseWishartDistribution(9.5, scale);
        DistributionAssert.Close(s[1] / 5.5, inverse.GetMean()[1, 0], 1e-15, "Mean");
        double m = 5.5;
        DistributionAssert.Close((7.5 * s[1] * s[1] + m * s[0] * s[4]) / (6.5 * m * m * 3.5), inverse.GetVariance()[1, 0], 1e-15, "Variance");

        var heavy = new InverseWishartDistribution(5.5, scale);
        Assert.True(double.IsFinite(heavy.GetMean()[0, 1]));
        Assert.All(heavy.GetVariance().AsSpan().ToArray(), v => Assert.Equal(double.PositiveInfinity, v));

        var heavier = new InverseWishartDistribution(3.5, scale);
        Assert.Equal(double.PositiveInfinity, heavier.GetMean()[2, 2]);
        Assert.True(double.IsNaN(heavier.GetMean()[2, 0]));
        Assert.All(heavier.GetVariance().AsSpan().ToArray(), v => Assert.True(double.IsNaN(v)));

        Assert.Equal(4.5, wishart.DegreesOfFreedom);
        Assert.Equal(3, inverse.Dimension);
        Assert.Equal(s, inverse.Scale.ToArray().Cast<double>().ToArray());
    }

    public static TheoryData<string> Names => ["Wishart", "InverseWishart"];

    private static IMatrixContinuousDistribution Create(string name) => name == "Wishart"
        ? new WishartDistribution(3.5, Square(2, 0.5, 0.5, 1))
        : new InverseWishartDistribution(3.5, Square(2, 0.5, 0.5, 1));

    [Theory]
    [MemberData(nameof(Names))]
    public void PointsOutsideThePositiveDefiniteConeHaveZeroDensity(string name)
    {
        var d = Create(name);
        Assert.True(double.IsFinite(d.LogPdf(Square(1, 0.2, 0.2, 0.5))));
        Assert.Equal(double.NegativeInfinity, d.LogPdf(Square(1, 0.2, 0.3, 0.5)));
        Assert.Equal(double.NegativeInfinity, d.LogPdf(Square(1, 2, 2, 1)));
        Assert.Equal(double.NegativeInfinity, d.LogPdf(Square(1, 1, 1, 1)));
        Assert.Equal(double.NegativeInfinity, d.LogPdf(Square(-1, 0, 0, -1)));
        Assert.Equal(double.NegativeInfinity, d.LogPdf(Square(0, 0, 0, 0)));
        Assert.Equal(0.0, d.Pdf(Square(1, 2, 2, 1)));

        // 丸め程度の非対称は許し、下三角を使う
        Assert.Equal(d.LogPdf(Square(1, 0.2, 0.2, 0.5)), d.LogPdf(Square(1, 0.2, 0.2 + 1e-12, 0.5)));
    }

    [Fact]
    public void FarPointsHaveZeroDensityWithoutNaN()
    {
        // tr(Σ⁻¹X)・tr(ΨX⁻¹) が溢れる点
        var wishart = new WishartDistribution(3.5, Square(1e-300, 0, 0, 1e-300));
        Assert.Equal(double.NegativeInfinity, wishart.LogPdf(Square(1e300, 0, 0, 1e300)));
        var inverse = new InverseWishartDistribution(3.5, Square(1e300, 0, 0, 1e300));
        Assert.Equal(double.NegativeInfinity, inverse.LogPdf(Square(1e-300, 0, 0, 1e-300)));
    }

    [Theory]
    [MemberData(nameof(Names))]
    public void InvalidParametersThrow(string name)
    {
        Func<double, double[], object> create = name == "Wishart"
            ? (nu, s) => new WishartDistribution(nu, Square(s))
            : (nu, s) => new InverseWishartDistribution(nu, Square(s));
        double[] scale = [2, 0.5, 0.5, 1];
        Assert.Throws<ArgumentOutOfRangeException>(() => create(1, scale));
        Assert.Throws<ArgumentOutOfRangeException>(() => create(0.5, scale));
        Assert.Throws<ArgumentOutOfRangeException>(() => create(double.PositiveInfinity, scale));
        Assert.Throws<ArgumentOutOfRangeException>(() => create(double.NaN, scale));
        Assert.Throws<ArgumentException>(() => create(3, []));
        Assert.Throws<ArgumentException>(() => create(3, [2, 0.5, 0.4, 1]));
        Assert.Throws<ArgumentException>(() => create(3, [1, 2, 2, 1]));
        Assert.Throws<ArgumentException>(() => create(3, [1, 1, 1, 1]));
        Assert.Throws<ArgumentOutOfRangeException>(() => create(3, [1, double.NaN, double.NaN, 1]));
        Assert.Throws<ArgumentOutOfRangeException>(() => create(3, [1, 0, 0, double.PositiveInfinity]));
        Assert.Throws<ArgumentException>(() => name == "Wishart"
            ? new WishartDistribution(3, new ReadOnlyMatrixSpan<double>(new double[6], 2, 3))
            : new InverseWishartDistribution(3, new ReadOnlyMatrixSpan<double>(new double[6], 2, 3)));

        // ν は p - 1 より大きければ実数でよい
        Assert.True(double.IsFinite(((IMatrixContinuousDistribution)create(1.000001, scale)).LogPdf(Square(1, 0.2, 0.2, 0.5))));
    }
}
