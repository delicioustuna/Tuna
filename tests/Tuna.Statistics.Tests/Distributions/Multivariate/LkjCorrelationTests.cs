using Tuna.LinearAlgebra;
using Tuna.RandomSource;
using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Statistics;

// LKJ 分布: 2次元のベータ分布との一致、η = 1 の 3 × 3 で相関行列の集合の体積、大きな η での安定性、
// 評価点の台の判定、標本の対角、パラメータの検証。
public class LkjCorrelationTests
{
    private static ReadOnlyMatrixSpan<double> Square(params double[] values) => MultivariateContractTests.Square(values);

    [Theory]
    [InlineData(1.0, 0.3)]
    [InlineData(0.5, -0.99)]
    [InlineData(3.0, 0.0)]
    [InlineData(1e6, 1e-4)]
    [InlineData(0.05, 0.999)]
    public void TwoDimensionsMatchBeta(double eta, double r)
    {
        // d = 2 では (r + 1)/2 ~ Beta(η, η) で、密度は r について 1/2 倍
        double expected = new BetaDistribution(eta, eta).LogPdf(0.5 * (r + 1)) - Math.Log(2);
        DistributionAssert.Close(expected, new LkjCorrelationDistribution(2, eta).LogPdf(Square(1, r, r, 1)), 1e-13, "LogPdf");
    }

    [Fact]
    public void UniformThreeByThreeIsTheReciprocalVolumeOfTheElliptope()
    {
        // 3 × 3 の相関行列の集合の体積は π²/2
        var d = new LkjCorrelationDistribution(3, 1);
        double expected = -Math.Log(Math.PI * Math.PI / 2);
        DistributionAssert.Close(expected, d.LogPdf(Matrix<double>.Identity(3)), 1e-15, "identity");
        DistributionAssert.Close(expected, d.LogPdf(Square(1, 0.4, -0.2, 0.4, 1, 0.3, -0.2, 0.3, 1)), 1e-15, "interior");
    }

    [Fact]
    public void LargeShapeStaysFinite()
    {
        // η が大きいと分布は単位行列に集中する。正規化定数は大きな η でも桁落ちせずに求まる
        foreach (double eta in new[] { 1e8, 1e300 })
        {
            var d = new LkjCorrelationDistribution(5, eta);
            double atIdentity = d.LogPdf(Matrix<double>.Identity(5));
            Assert.True(double.IsFinite(atIdentity) && atIdentity > 0, $"η = {eta}: {atIdentity}");
        }

        // d = 2 では (η/π)^(1/2)·(1 + O(1/η)) に近づく: ln c = ln(√π Γ(η)/Γ(η + 1/2)) ≈ (ln π - ln η)/2
        var two = new LkjCorrelationDistribution(2, 1e12);
        DistributionAssert.Close(0.5 * (Math.Log(1e12) - Math.Log(Math.PI)), two.LogPdf(Matrix<double>.Identity(2)), 1e-12, "d = 2");
    }

    [Fact]
    public void PointsOutsideTheCorrelationMatricesHaveZeroDensity()
    {
        var d = new LkjCorrelationDistribution(3, 1.5);
        double inside = d.LogPdf(Square(1, 0.4, -0.2, 0.4, 1, 0.3, -0.2, 0.3, 1));
        Assert.True(double.IsFinite(inside));

        // 対角が 1 から √ε 以内なら1として扱い、その外は台の外
        Assert.Equal(inside, d.LogPdf(Square(1 + 1e-12, 0.4, -0.2, 0.4, 1, 0.3, -0.2, 0.3, 1 - 1e-12)));
        Assert.Equal(double.NegativeInfinity, d.LogPdf(Square(1.001, 0.4, -0.2, 0.4, 1, 0.3, -0.2, 0.3, 1)));
        Assert.Equal(double.NegativeInfinity, d.LogPdf(Square(1, 0.4, -0.2, 0.5, 1, 0.3, -0.2, 0.3, 1)));
        Assert.Equal(double.NegativeInfinity, d.LogPdf(Square(1, -0.6, -0.6, -0.6, 1, -0.6, -0.6, -0.6, 1)));
        Assert.Equal(double.NegativeInfinity, d.LogPdf(Square(1, 1, 0, 1, 1, 0, 0, 0, 1)));
        Assert.Equal(double.NegativeInfinity, d.LogPdf(Square(1, 1.5, 0, 1.5, 1, 0, 0, 0, 1)));
        Assert.Equal(0.0, d.Pdf(Square(1, 1.5, 0, 1.5, 1, 0, 0, 0, 1)));
    }

    [Theory]
    [InlineData(2, 0.5)]
    [InlineData(4, 1.5)]
    [InlineData(6, 1e4)]
    public void SamplesHaveUnitDiagonal(int dimension, double eta)
    {
        var d = new LkjCorrelationDistribution(dimension, eta);
        var random = new Pcg64DxsmRandom(10, 10);
        for (int k = 0; k < 200; k++)
        {
            var x = d.Sample(random);
            for (int i = 0; i < dimension; i++)
            {
                Assert.Equal(1.0, x[i, i]);
                for (int j = 0; j < i; j++)
                {
                    Assert.InRange(x[i, j], -1.0, 1.0);
                }
            }
        }

        Assert.Equal(Matrix<double>.Identity(dimension).AsSpan().ToArray(), d.GetMean().AsSpan().ToArray());
        Assert.Equal(dimension, d.Dimension);
        Assert.Equal(eta, d.Shape);
    }

    [Fact]
    public void InvalidParametersThrow()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new LkjCorrelationDistribution(1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LkjCorrelationDistribution(0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LkjCorrelationDistribution(3, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LkjCorrelationDistribution(3, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LkjCorrelationDistribution(3, double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LkjCorrelationDistribution(3, double.PositiveInfinity));
    }
}
