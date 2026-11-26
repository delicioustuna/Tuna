using Tuna.LinearAlgebra;
using Tuna.RandomSource;
using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Statistics;

// 多変量正規分布と行列正規分布: 1変量の正規分布・ベクトル化した多変量正規分布との一致、コレスキー因子からの作成、
// 共分散の検証、x - μ が溢れる場合の計算し直し。
public class MultivariateNormalTests
{
    private static ReadOnlyMatrixSpan<double> Square(params double[] values) => MultivariateContractTests.Square(values);

    [Theory]
    [InlineData(0.5, 1.5, 0.5)]
    [InlineData(0.5, 1.5, 5.0)]
    [InlineData(-3.0, 0.01, 2.0)]
    [InlineData(1e10, 3e9, -1e11)]
    public void OneDimensionMatchesNormal(double mean, double standardDeviation, double x)
    {
        var normal = new NormalDistribution(mean, standardDeviation);
        var multivariate = new MultivariateNormalDistribution(new[] { mean }, Square(standardDeviation * standardDeviation));
        DistributionAssert.Close(normal.LogPdf(x), multivariate.LogPdf(new[] { x }), 2e-15, "LogPdf");
    }

    [Fact]
    public void OneDimensionalSamplesFromTheFactorMatchNormal()
    {
        var normal = new NormalDistribution(-2, 0.75);
        var multivariate = MultivariateNormalDistribution.FromCholeskyFactor(new[] { -2.0 }, Square(0.75));
        var a = new Pcg64DxsmRandom(11, 1);
        var b = new Pcg64DxsmRandom(11, 1);
        for (int k = 0; k < 200; k++)
        {
            Assert.Equal(normal.Sample(a), multivariate.Sample(b)[0]);
        }
    }

    [Fact]
    public void DiagonalCovarianceIsAProductOfNormals()
    {
        var d = new MultivariateNormalDistribution(new[] { 1.0, -2.0, 0.5 }, Square(4, 0, 0, 0, 0.25, 0, 0, 0, 9));
        double[] x = [2.5, -1.0, -6.0];
        double expected = new NormalDistribution(1, 2).LogPdf(2.5) + new NormalDistribution(-2, 0.5).LogPdf(-1.0)
            + new NormalDistribution(0.5, 3).LogPdf(-6.0);
        DistributionAssert.Close(expected, d.LogPdf(x), 1e-15, "LogPdf");
    }

    [Fact]
    public void FromCholeskyFactorMatchesTheCovarianceConstructor()
    {
        // 上三角は読まない（NaN でもよい）
        double[] factor = [1.5, 0.2, -0.3, double.NaN, 0.8, 0.1, double.NaN, double.NaN, 1.2];
        var fromFactor = MultivariateNormalDistribution.FromCholeskyFactor(MultivariateContractTests.Mean3, Square(factor));
        var covariance = fromFactor.GetCovariance();
        var fromCovariance = new MultivariateNormalDistribution(MultivariateContractTests.Mean3, covariance);

        Assert.Equal(0.0, fromFactor.CholeskyFactor[0, 1]);
        Assert.Equal(0.0, fromFactor.CholeskyFactor[1, 2]);
        Assert.Equal(1.5 * 1.5, covariance[0, 0]);
        Assert.Equal(1.5 * 0.2, covariance[1, 0]);
        Assert.Equal(covariance[1, 0], covariance[0, 1]);
        DistributionAssert.Close(2 * (Math.Log(1.5) + Math.Log(0.8) + Math.Log(1.2)), fromFactor.LogDeterminant, 1e-15, "LogDeterminant");
        DistributionAssert.Close(fromFactor.LogDeterminant, fromCovariance.LogDeterminant, 1e-14, "LogDeterminant");
        foreach (double[] x in new[] { new[] { 0.0, 0.0, 0.0 }, new[] { 3.0, -1.0, 2.0 } })
        {
            DistributionAssert.Close(fromFactor.LogPdf(x), fromCovariance.LogPdf(x), 1e-14, "LogPdf");
        }
    }

    [Fact]
    public void LowerTriangleIsUsedWithinTheSymmetryTolerance()
    {
        var symmetric = new MultivariateNormalDistribution(new[] { 0.0, 0.0 }, Square(2, 0.3, 0.3, 1));
        var nearlySymmetric = new MultivariateNormalDistribution(new[] { 0.0, 0.0 }, Square(2, 0.3, 0.3 + 1e-12, 1));
        Assert.Equal(0.3, nearlySymmetric.Covariance[0, 1]);
        Assert.Equal(symmetric.LogPdf(new[] { 1.0, -1.0 }), nearlySymmetric.LogPdf(new[] { 1.0, -1.0 }));
    }

    [Fact]
    public void InvalidParametersThrow()
    {
        double[] mean = [0.0, 0.0];
        Assert.Throws<ArgumentException>(() => new MultivariateNormalDistribution(ReadOnlySpan<double>.Empty, Square()));
        Assert.Throws<ArgumentException>(() => new MultivariateNormalDistribution(mean, Square(1, 0, 0, 0, 1, 0, 0, 0, 1)));
        Assert.Throws<ArgumentException>(() => new MultivariateNormalDistribution(mean, new ReadOnlyMatrixSpan<double>(new double[6], 2, 3)));
        Assert.Throws<ArgumentException>(() => new MultivariateNormalDistribution(mean, Square(1, 0.5, 0.4, 1)));
        Assert.Throws<ArgumentException>(() => new MultivariateNormalDistribution(mean, Square(1, 2, 2, 1)));
        Assert.Throws<ArgumentException>(() => new MultivariateNormalDistribution(mean, Square(1, 1, 1, 1)));
        Assert.Throws<ArgumentException>(() => new MultivariateNormalDistribution(mean, Square(0, 0, 0, 1)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new MultivariateNormalDistribution(new[] { 0.0, double.NaN }, Square(1, 0, 0, 1)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new MultivariateNormalDistribution(new[] { double.PositiveInfinity, 0.0 }, Square(1, 0, 0, 1)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new MultivariateNormalDistribution(mean, Square(1, 0, 0, double.PositiveInfinity)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new MultivariateNormalDistribution(mean, Square(1, double.NaN, double.NaN, 1)));

        Assert.Throws<ArgumentOutOfRangeException>(() => MultivariateNormalDistribution.FromCholeskyFactor(mean, Square(1, 0, 0, 0)));
        Assert.Throws<ArgumentOutOfRangeException>(() => MultivariateNormalDistribution.FromCholeskyFactor(mean, Square(1, 0, 0, -1)));
        Assert.Throws<ArgumentOutOfRangeException>(() => MultivariateNormalDistribution.FromCholeskyFactor(mean, Square(1, double.NaN, 0, 1)));
        Assert.Throws<ArgumentOutOfRangeException>(() => MultivariateNormalDistribution.FromCholeskyFactor(mean, Square(1e200, 0, 0, 1)));
        Assert.Throws<ArgumentException>(() => MultivariateNormalDistribution.FromCholeskyFactor(mean, Square(1)));
    }

    /// <summary>
    /// x₂ - μ₂ が溢れるが ‖L⁻¹(x - μ)‖² は有限な点。μ・Σ・x を 1/2・1/4・1/2 にした分布では差が溢れず、
    /// 密度の対数は d ln 2 だけ大きい（L も 1/2 になるだけなので z は同じ）
    /// </summary>
    private static readonly double[] OverflowMean = [0.0, -0.95e308];
    private static readonly double[] OverflowCovariance = [1.0, 1e154, 1e154, 1.7e308];
    private static readonly double[] OverflowPoint = [1.8e154, 0.95e308];

    private static double HalvedLogPdf() =>
        new MultivariateNormalDistribution(OverflowMean.Select(v => v / 2).ToArray(), Square(OverflowCovariance.Select(v => v / 4).ToArray()))
            .LogPdf(OverflowPoint.Select(v => v / 2).ToArray()) - 2 * Math.Log(2);

    [Fact]
    public void OverflowingDifferenceIsComputedFromHalves()
    {
        Assert.Equal(double.PositiveInfinity, OverflowPoint[1] - OverflowMean[1]);
        double expected = HalvedLogPdf();
        Assert.True(double.IsFinite(expected) && expected < -1e308);
        var d = new MultivariateNormalDistribution(OverflowMean, Square(OverflowCovariance));
        DistributionAssert.Close(expected, d.LogPdf(OverflowPoint), 1e-15, "MultivariateNormal");

        var t = new MultivariateStudentTDistribution(double.PositiveInfinity, OverflowMean, Square(OverflowCovariance));
        DistributionAssert.Close(expected, t.LogPdf(OverflowPoint), 1e-15, "MultivariateStudentT(∞)");

        var matrix = new MatrixNormalDistribution(new ReadOnlyMatrixSpan<double>(OverflowMean, 2, 1), Square(OverflowCovariance), Square(1));
        DistributionAssert.Close(expected, matrix.LogPdf(new ReadOnlyMatrixSpan<double>(OverflowPoint, 2, 1)), 1e-15, "MatrixNormal");
    }

    // ---------------------------------------------------------------- 行列正規分布

    [Fact]
    public void MatrixNormalWithOneColumnMatchesMultivariateNormal()
    {
        var matrix = new MatrixNormalDistribution(new ReadOnlyMatrixSpan<double>(MultivariateContractTests.Mean3, 3, 1),
            Square(MultivariateContractTests.Covariance3), Square(1));
        var vector = new MultivariateNormalDistribution(MultivariateContractTests.Mean3, Square(MultivariateContractTests.Covariance3));
        foreach (double[] x in new[] { new[] { 0.0, 0.0, 0.0 }, new[] { 3.0, -1.0, 2.0 }, new[] { -30.0, 10.0, 0.5 } })
        {
            DistributionAssert.Close(vector.LogPdf(x), matrix.LogPdf(new ReadOnlyMatrixSpan<double>(x, 3, 1)), 1e-15, "LogPdf");
        }
    }

    [Fact]
    public void MatrixNormalMatchesTheMultivariateNormalOfTheVectorization()
    {
        // vec(X) ~ N(vec(M), V ⊗ U)
        var d = (MatrixNormalDistribution)MultivariateContractTests.Matrix("MatrixNormal");
        int n = d.Rows;
        int p = d.Columns;
        var kronecker = new Matrix<double>(n * p, n * p);
        for (int k = 0; k < n * p; k++)
        {
            for (int l = 0; l < n * p; l++)
            {
                kronecker[k, l] = d.ColumnCovariance[k / n, l / n] * d.RowCovariance[k % n, l % n];
            }
        }

        var vector = new MultivariateNormalDistribution(d.GetMean().AsVector(), kronecker);
        var random = new Pcg64DxsmRandom(5, 1);
        for (int s = 0; s < 20; s++)
        {
            var x = d.Sample(random);
            if (s % 2 == 1)
            {
                // 裾の点
                x.AsSpan()[s % 6] += 10 * s;
            }

            double expected = vector.LogPdf(x.AsVector());
            DistributionAssert.Close(expected, d.LogPdf(x), 1e-13, "LogPdf");
        }
    }

    [Fact]
    public void MatrixNormalIsInvariantUnderReciprocalScalingOfTheCovariances()
    {
        var d = (MatrixNormalDistribution)MultivariateContractTests.Matrix("MatrixNormal");
        var u = d.RowCovariance.ToArray();
        var v = d.ColumnCovariance.ToArray();
        var scaled = new MatrixNormalDistribution(d.Mean, (Matrix<double>)Scale(u, 3.0), (Matrix<double>)Scale(v, 1 / 3.0));
        var random = new Pcg64DxsmRandom(5, 2);
        for (int s = 0; s < 10; s++)
        {
            var x = d.Sample(random);
            DistributionAssert.Close(d.LogPdf(x), scaled.LogPdf(x), 1e-14, "LogPdf");
        }
    }

    private static double[,] Scale(double[,] values, double factor)
    {
        var result = (double[,])values.Clone();
        for (int i = 0; i < result.GetLength(0); i++)
        {
            for (int j = 0; j < result.GetLength(1); j++)
            {
                result[i, j] *= factor;
            }
        }

        return result;
    }

    [Fact]
    public void MatrixNormalInvalidParametersThrow()
    {
        double[] values = [1.0, 2.0];
        Assert.Throws<ArgumentException>(() => new MatrixNormalDistribution(new ReadOnlyMatrixSpan<double>([], 0, 2), Square(), Square(1, 0, 0, 1)));
        Assert.Throws<ArgumentException>(() => new MatrixNormalDistribution(new ReadOnlyMatrixSpan<double>(values, 2, 1), Square(1), Square(1)));
        Assert.Throws<ArgumentException>(() => new MatrixNormalDistribution(new ReadOnlyMatrixSpan<double>(values, 2, 1), Square(1, 0, 0, 1), Square(1, 0, 0, 1)));
        Assert.Throws<ArgumentException>(() => new MatrixNormalDistribution(new ReadOnlyMatrixSpan<double>(values, 2, 1), Square(1, 0.5, 0.4, 1), Square(1)));
        Assert.Throws<ArgumentException>(() => new MatrixNormalDistribution(new ReadOnlyMatrixSpan<double>(values, 2, 1), Square(1, 0, 0, 1), Square(-1)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new MatrixNormalDistribution(new ReadOnlyMatrixSpan<double>([1.0, double.NaN], 2, 1),
            Square(1, 0, 0, 1), Square(1)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new MatrixNormalDistribution(new ReadOnlyMatrixSpan<double>(values, 2, 1), Square(1, 0, 0, 1), Square(double.PositiveInfinity)));
    }

    [Fact]
    public void MatrixNormalExposesItsParameters()
    {
        var d = (MatrixNormalDistribution)MultivariateContractTests.Matrix("MatrixNormal");
        Assert.Equal(3, d.Rows);
        Assert.Equal(2, d.Columns);
        Assert.Equal(-1.0, d.Mean[2, 0]);
        Assert.Equal(d.Mean.ToArray(), d.GetMean().ToArray());
        Assert.Equal(MultivariateContractTests.Covariance3, d.RowCovariance.ToArray().Cast<double>().ToArray());
        Assert.Equal(0.3, d.ColumnCovariance[1, 0]);
    }
}
