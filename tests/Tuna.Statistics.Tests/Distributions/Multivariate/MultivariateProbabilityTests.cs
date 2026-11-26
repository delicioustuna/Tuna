using Tuna.Integration;
using Tuna.LinearAlgebra;
using Tuna.RandomSource;
using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Statistics;

// 多変量正規分布・多変量 t 分布の累積分布関数と超直方体の確率: 1変量・2変量の閉じた式、直交確率の閉じた式（1/8 + Σasin ρ/(4π)、相関 1/2 の 1/(n+1)）、
// 包除原理との一致、変数の並べ替えに依らないこと、ν = ∞ の多変量正規分布との一致、scipy（Genz の方法）の参照値、標本との一致、端の規約、パラメータの検証。
public class MultivariateProbabilityTests
{
    private static ReadOnlyMatrixSpan<double> Square(params double[] values) => MultivariateContractTests.Square(values);

    private static IntegrationOptions Tight(double absolute = 1e-12) => new() { AbsoluteTolerance = absolute, RelativeTolerance = 0 };

    private static IntegrationOptions Budget(double absolute) => new() { AbsoluteTolerance = absolute, RelativeTolerance = 0, MaxEvaluations = 1 << 24 };

    // 準モンテカルロ積分の標準誤差は1σの見積もり。5σ と、参照値の誤差の分だけの余裕で比べる
    private static void AssertWithin(double expected, IntegrationResult result, double margin, string context)
    {
        Assert.True(result.Converged, $"{context}: {result}");
        Assert.True(Math.Abs(result.Value - expected) <= 5 * result.ErrorEstimate + margin, $"{context}: {result.Value} vs {expected}, error {result.ErrorEstimate}");
    }

    private static readonly double[] Correlation3 = [1, 0.5, 0.3, 0.5, 1, 0.4, 0.3, 0.4, 1];

    [Theory]
    [InlineData(0.0, 1.0, 0.3)]
    [InlineData(-1.0, 2.0, -5.0)]
    [InlineData(2.0, 0.5, 2.4)]
    [InlineData(0.0, 1.0, 40.0)]
    public void OneDimensionalIsTheNormalCdf(double mean, double sd, double x)
    {
        var d = new MultivariateNormalDistribution(new[] { mean }, Square(sd * sd));
        var result = d.Cdf(new[] { x }, Tight());
        Assert.True(result.Converged);
        DistributionAssert.Close(XMath.Special.NormalCdf((x - mean) / sd), result.Value, 1e-14, "Cdf");
        // 上の裾（x = 40）の区間の確率は倍精度で表せない（約 1e-334）ので、絶対誤差で比べる
        double expected = XMath.Special.NormalCdf((x + 0.5 - mean) / sd) - XMath.Special.NormalCdf((x - 1 - mean) / sd);
        Assert.InRange(d.Probability(new[] { x - 1 }, new[] { x + 0.5 }, Tight()).Value, expected - 1e-300, expected + 1e-14);
    }

    [Theory]
    [InlineData(0.5, 0.5, 1.0)]
    [InlineData(-0.3, -1.2, 0.4)]
    [InlineData(0.9, 0.7, 0.7)]
    [InlineData(0.99, -2.0, 1.5)]
    [InlineData(0.0, 1.0, -1.0)]
    public void BivariateMatchesTheBivariateNormalCdf(double rho, double x, double y)
    {
        var d = new MultivariateNormalDistribution(new double[2], Square(1, rho, rho, 1));
        var result = d.Cdf(new[] { x, y }, Tight());
        Assert.True(result.Converged);
        DistributionAssert.Close(XMath.Special.BivariateNormalCdf(x, y, rho), result.Value, 1e-11, "Cdf");
    }

    [Fact]
    public void ThreeDimensionalMatchesTheDocumentedValue()
    {
        // 3 次元多変量正規分布 CDF の参照値 0.60700631164716123
        var d = new MultivariateNormalDistribution(new double[3], Square(Correlation3));
        var result = d.Cdf(new[] { 0.5, 1.0, 1.5 }, Tight());
        Assert.True(result.Converged);
        DistributionAssert.Close(0.60700631164716123, result.Value, 1e-12, "Cdf");
    }

    [Fact]
    public void OrthantProbabilityHasTheClosedForm()
    {
        // 3変量の直交確率 P(X ≤ 0) = 1/8 + (asin ρ₁₂ + asin ρ₁₃ + asin ρ₂₃)/(4π)
        var d = new MultivariateNormalDistribution(new double[3], Square(Correlation3));
        double expected = 0.125 + (Math.Asin(0.5) + Math.Asin(0.3) + Math.Asin(0.4)) / (4 * Math.PI);
        DistributionAssert.Close(expected, d.Cdf(new double[3], Tight()).Value, 1e-12, "Cdf");
    }

    [Theory]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    public void EquicorrelatedOrthantHasTheReciprocalFormula(int n)
    {
        // 相関 1/2 の n 変量では P(X ≤ 0) = 1/(n + 1)。4変量は直積の求積、5変量以上は準モンテカルロ積分
        var sigma = new double[n * n];
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                sigma[i + j * n] = i == j ? 1.0 : 0.5;
            }
        }

        var d = new MultivariateNormalDistribution(new double[n], Square(sigma));
        AssertWithin(1.0 / (n + 1), d.Cdf(new double[n], Budget(1e-7)), 1e-12, "Cdf");
    }

    [Fact]
    public void FourVariateMatchesTheScipyReference()
    {
        // scipy.stats.multivariate_normal.cdf（Genz の方法）の値。scipy の誤差は 1e-8 程度
        double[] correlation4 = [1, 0.3, 0.2, 0.1, 0.3, 1, 0.4, 0.2, 0.2, 0.4, 1, 0.3, 0.1, 0.2, 0.3, 1];
        var d = new MultivariateNormalDistribution(new double[4], Square(correlation4));
        AssertWithin(0.20796539645347964, d.Cdf(new[] { 0.2, -0.4, 0.9, 1.1 }, Budget(1e-8)), 1e-8, "Cdf");
    }

    [Fact]
    public void RectangleProbabilityIsTheAlternatingSumOfTheCdf()
    {
        // 包除原理: P(a < X ≤ b) は端の8点の累積分布関数の交代和
        var d = new MultivariateNormalDistribution(new[] { 0.5, -1.0, 0.0 }, Square(Correlation3));
        double[] a = [-0.5, -1.5, -0.25];
        double[] b = [0.5, 0.75, 1.5];
        double sum = 0;
        for (int mask = 0; mask < 8; mask++)
        {
            var corner = new double[3];
            double sign = 1;
            for (int i = 0; i < 3; i++)
            {
                bool lowerEnd = (mask >> i & 1) == 1;
                corner[i] = lowerEnd ? a[i] : b[i];
                if (lowerEnd)
                {
                    sign = -sign;
                }
            }

            sum += sign * d.Cdf(corner, Tight()).Value;
        }

        var rectangle = d.Probability(a, b, Tight());
        Assert.True(rectangle.Converged);
        DistributionAssert.Close(sum, rectangle.Value, 1e-10, "Probability");
    }

    [Fact]
    public void ProbabilityIsInvariantUnderPermutationOfTheVariables()
    {
        // 並べ替え（ピボット）は確率を変えない: 変数を入れ替えた分布と区間で同じ値になる
        var d = new MultivariateNormalDistribution(new[] { 0.2, -0.4, 1.0 }, Square(Correlation3));
        double[] lower = [-1.0, -0.5, -2.0];
        double[] upper = [0.8, 2.0, 0.5];
        var permuted = new MultivariateNormalDistribution(new[] { -0.4, 1.0, 0.2 },
            Square(1, 0.4, 0.5, 0.4, 1, 0.3, 0.5, 0.3, 1));
        double[] permutedLower = [-0.5, -2.0, -1.0];
        double[] permutedUpper = [2.0, 0.5, 0.8];
        DistributionAssert.Close(d.Probability(lower, upper, Tight()).Value,
            permuted.Probability(permutedLower, permutedUpper, Tight()).Value, 1e-10, "Probability");
    }

    [Fact]
    public void ProbabilityAgreesWithSamples()
    {
        // 標本（100,000 個）の割合と比べる。割合の標準誤差は約 0.0014
        var d = new MultivariateNormalDistribution(new[] { 0.2, -0.4, 1.0 }, Square(Correlation3));
        double[] lower = [-1.0, -0.5, -2.0];
        double[] upper = [0.8, 2.0, 0.5];
        var samples = d.Sample(new Pcg64DxsmRandom(17, 3), 100_000);
        int inside = 0;
        for (int j = 0; j < samples.Columns; j++)
        {
            bool all = true;
            for (int i = 0; i < 3; i++)
            {
                all &= samples[i, j] > lower[i] && samples[i, j] <= upper[i];
            }

            if (all)
            {
                inside++;
            }
        }

        double frequency = (double)inside / samples.Columns;
        Assert.InRange(frequency, d.Probability(lower, upper, Tight()).Value - 0.01, d.Probability(lower, upper, Tight()).Value + 0.01);
    }

    [Fact]
    public void FullRangeHasProbabilityOne()
    {
        var d = new MultivariateNormalDistribution(new double[3], Square(Correlation3));
        double[] upper = [double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity];
        DistributionAssert.Close(1, d.Cdf(upper, Tight()).Value, 1e-12, "Cdf");
        Assert.Equal(0, d.Cdf(new[] { double.NegativeInfinity, 0, 0 }, Tight()).Value);
    }

    [Fact]
    public void EmptyAndNaNRectanglesFollowTheConventions()
    {
        var d = new MultivariateNormalDistribution(new double[2], Square(1, 0.5, 0.5, 1));

        // 区間が空（下端 ≥ 上端）なら確率 0、評価の回数 0
        var empty = d.Probability(new[] { 0.0, -1.0 }, new[] { 0.0, 1.0 });
        Assert.Equal(0, empty.Value);
        Assert.Equal(0, empty.Evaluations);
        Assert.Equal(IntegrationTermination.Tolerance, empty.Termination);
        Assert.Equal(0, d.Cdf(new[] { double.NegativeInfinity, -1.0 }).Value);

        // NaN の端は NaN（分布の密度の規約と同じ）
        var nan = d.Cdf(new[] { double.NaN, 0.0 });
        Assert.True(double.IsNaN(nan.Value));
        Assert.Equal(IntegrationTermination.NonFiniteValue, nan.Termination);
    }

    [Fact]
    public void NaNIsFoundInEveryComponentBeforeTheEmptyInterval()
    {
        // 空区間の成分が NaN の成分より前にあっても NaN（成分の並び順で結果が変わらない）
        var d = new MultivariateNormalDistribution(new double[2], Square(1, 0, 0, 1));
        foreach (var upper in new[] { new[] { double.NegativeInfinity, double.NaN }, new[] { double.NaN, double.NegativeInfinity } })
        {
            var result = d.Cdf(upper);
            Assert.True(double.IsNaN(result.Value));
            Assert.Equal(IntegrationTermination.NonFiniteValue, result.Termination);
        }

        Assert.True(double.IsNaN(d.Probability(new[] { 0.0, double.NaN }, new[] { 0.0, 1.0 }).Value));
    }

    [Fact]
    public void TinyPositiveVarianceIsUsedAsItIs()
    {
        // 正規化数でない正の分散（1e-310）も置き換えない: 標準偏差1つ分の累積確率は Φ(1)
        double v = 1e-310;
        var d = new MultivariateNormalDistribution(new[] { 0.0 }, Square(v));
        DistributionAssert.Close(XMath.Special.NormalCdf(1), d.Cdf(new[] { Math.Sqrt(v) }).Value, 1e-14, "Cdf");
        var d2 = new MultivariateNormalDistribution(new double[2], Square(v, 0, 0, 1));
        DistributionAssert.Close(0.5 * XMath.Special.NormalCdf(1), d2.Cdf(new[] { Math.Sqrt(v), 0.0 }, Tight()).Value, 1e-12, "Cdf");
    }

    [Fact]
    public void InvalidArgumentsThrow()
    {
        var d = new MultivariateNormalDistribution(new double[2], Square(1, 0.5, 0.5, 1));
        Assert.Throws<ArgumentException>(() => d.Cdf(new[] { 0.0 }));
        Assert.Throws<ArgumentException>(() => d.Probability(new[] { 0.0 }, new[] { 1.0, 1.0 }));
        Assert.Throws<ArgumentException>(() => d.Probability(new[] { 0.0, 0.0 }, new[] { 1.0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => d.Cdf(new[] { 0.0, 0.0 }, new IntegrationOptions { AbsoluteTolerance = -1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => d.Cdf(new[] { 0.0, 0.0 }, new IntegrationOptions { RelativeTolerance = double.NaN }));
    }

    [Fact]
    public void StudentTWithOneDimensionIsTheUnivariateStudentT()
    {
        var t = new MultivariateStudentTDistribution(3.5, new[] { 1.0 }, Square(4));
        var univariate = new StudentTDistribution(3.5, 1, 2);
        foreach (double x in new[] { -6.0, 0.5, 1.0, 7.5 })
        {
            DistributionAssert.Close(univariate.Cdf(x), t.Cdf(new[] { x }, Tight()).Value, 1e-10, "Cdf");
        }
    }

    [Theory]
    [InlineData(3.5, 0.3342726437483571)]
    [InlineData(1.0, 0.30477546339247474)]
    [InlineData(30.0, 0.3477429046335602)]
    public void ThreeDimensionalMatchesTheScipyReference(double nu, double expected)
    {
        // scipy.stats.multivariate_t.cdf の値（位置 μ = (1, -2, 0.5)、尺度 S、点 (1.5, -1, 1)）。scipy の誤差は 1e-8 程度
        double[] scale = [2, 0.3, -0.4, 0.3, 1, 0.2, -0.4, 0.2, 1.5];
        var t = new MultivariateStudentTDistribution(nu, MultivariateContractTests.Mean3, Square(scale));
        AssertWithin(expected, t.Cdf(new[] { 1.5, -1.0, 1.0 }, Budget(1e-7)), 1e-7, "Cdf");
    }

    [Fact]
    public void FourDimensionalMatchesTheScipyReference()
    {
        // ν = 4 の4変量（準モンテカルロ積分の次元5）。scipy の値は 0.19894527（実行による差 3e-9）
        double[] correlation4 = [1, 0.3, 0.2, 0.1, 0.3, 1, 0.4, 0.2, 0.2, 0.4, 1, 0.3, 0.1, 0.2, 0.3, 1];
        var t = new MultivariateStudentTDistribution(4.0, new double[4], Square(correlation4));
        AssertWithin(0.198945278, t.Cdf(new[] { 0.2, -0.4, 0.9, 1.1 }, Budget(1e-7)), 1e-7, "Cdf");
    }

    [Fact]
    public void InfiniteDegreesOfFreedomIsTheMultivariateNormal()
    {
        var normal = new MultivariateNormalDistribution(MultivariateContractTests.Mean3, Square(MultivariateContractTests.Covariance3));
        var t = new MultivariateStudentTDistribution(double.PositiveInfinity, MultivariateContractTests.Mean3, Square(MultivariateContractTests.Covariance3));
        double[] upper = [1.5, -1.0, 1.0];
        double[] lower = [-0.5, -3.0, -1.0];
        Assert.Equal(normal.Cdf(upper, Tight()).Value, t.Cdf(upper, Tight()).Value);
        Assert.Equal(normal.Probability(lower, upper, Tight()).Value, t.Probability(lower, upper, Tight()).Value);
    }

    [Fact]
    public void StudentTProbabilityAgreesWithSamples()
    {
        // 標本（100,000 個）の割合と比べる。自由度 3 の裾があるので、標本の割合の標準誤差は約 0.0015
        var t = new MultivariateStudentTDistribution(3.0, MultivariateContractTests.Mean3, Square(MultivariateContractTests.Covariance3));
        double[] lower = [-1.0, -3.0, -0.5];
        double[] upper = [2.0, -1.0, 2.5];
        var samples = t.Sample(new Pcg64DxsmRandom(8, 5), 100_000);
        int inside = 0;
        for (int j = 0; j < samples.Columns; j++)
        {
            bool all = true;
            for (int i = 0; i < 3; i++)
            {
                all &= samples[i, j] > lower[i] && samples[i, j] <= upper[i];
            }

            if (all)
            {
                inside++;
            }
        }

        double frequency = (double)inside / samples.Columns;
        double expected = t.Probability(lower, upper, new IntegrationOptions { AbsoluteTolerance = 1e-4, RelativeTolerance = 0 }).Value;
        Assert.InRange(frequency, expected - 0.01, expected + 0.01);
    }

    [Theory]
    [InlineData(0.01)]
    [InlineData(0.001)]
    public void SmallDegreesOfFreedomKeepTheMassWhereTheScaleUnderflows(double nu)
    {
        // ν が小さいと、カイ二乗の分位点（尺度 s）が 0 に丸まる u の範囲の測度は無視できない（ν = 0.01 で約 3%）。
        // 全領域は 1、対角の尺度行列の直交確率は s によらず 1/4、1次元の全領域も 1
        var t = new MultivariateStudentTDistribution(nu, new double[2], Square(1, 0, 0, 1));
        DistributionAssert.Close(1, t.Cdf(new[] { double.PositiveInfinity, double.PositiveInfinity }).Value, 1e-8, "全領域");
        DistributionAssert.Close(0.25, t.Cdf(new double[2]).Value, 1e-8, "直交確率");
        var univariate = new MultivariateStudentTDistribution(nu, new[] { 0.0 }, Square(1));
        DistributionAssert.Close(1, univariate.Cdf(new[] { double.PositiveInfinity }).Value, 1e-8, "1次元の全領域");
        DistributionAssert.Close(0.5, univariate.Cdf(new[] { 0.0 }).Value, 1e-8, "1次元の中央");
    }

    [Fact]
    public void InvalidStudentTArgumentsThrow()
    {
        var t = new MultivariateStudentTDistribution(3, new double[2], Square(1, 0.5, 0.5, 1));
        Assert.Throws<ArgumentException>(() => t.Cdf(new[] { 0.0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => t.Cdf(new[] { 0.0, 0.0 }, new IntegrationOptions { AbsoluteTolerance = double.PositiveInfinity }));
    }
}
