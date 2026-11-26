using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Statistics;

public class MultivariateDistributionDifferentiationTests
{
    [Fact]
    public void MultivariateNormalCholeskyGradientMatchesCenteredDifferences()
    {
        Verify([0.2, -0.3, 1.4, 0.27, 0.9, 0.8, -1.5],
            p => MultivariateNormalDistribution.LogPdfCholesky(p[..2], p.Slice(2, 3), p[5..]),
            p => MultivariateNormalDistribution.FromCholeskyFactor(
                new ReadOnlyVectorSpan<double>(p[..2]), new Matrix<double>(new double[,] { { p[2], 0 }, { p[3], p[4] } })).LogPdf(p[5..]));
    }

    [Fact]
    public void NormalCholeskyGradientRetainsCancellationAtSubnormalScale()
    {
        var tape = new Tape();
        var factor = tape.Leaf(double.Epsilon);
        var result = MultivariateNormalDistribution.LogPdfCholesky([0], [factor], [double.Epsilon]);
        Assert.True(double.IsFinite(result.Value));
        tape.Backward(result);
        Assert.Equal(0, tape.GetGradient(factor));
    }

    [Fact]
    public void DirichletGradientMatchesCenteredDifferencesAlongTheSimplex()
    {
        Verify([0.7, 2.4, 3.1, 0.17, 0.25],
            p => DirichletDistribution.LogPdf(p[..3], [p[3], p[4], 1 - p[3] - p[4]]),
            p => new DirichletDistribution(p[..3]).LogPdf(new Vector<double>([p[3], p[4], 1 - p[3] - p[4]])));
    }

    [Fact]
    public void LkjStrictLowerTriangleGradientMatchesCenteredDifferences()
    {
        Verify([1.7, 0.13, -0.28, 0.62],
            p => LkjCorrelationDistribution.LogPdf(3, p[0], p[1..]),
            p => new LkjCorrelationDistribution(3, p[0]).LogPdf(new Matrix<double>(
                new double[,] { { 1, p[1], p[2] }, { p[1], 1, p[3] }, { p[2], p[3], 1 } })));
    }

    [Fact]
    public void WishartPackedMatrixGradientsMatchCenteredDifferences()
    {
        Verify([5.7, 2.2, 0.3, 1.5, -0.45, 0.62, 1.9, 3.4, 0.8, 2.1, 0.2, 0.4, 3.2],
            p => WishartDistribution.LogPdf(3, p[0], p.Slice(1, 6), p[7..]),
            p => new WishartDistribution(p[0], Symmetric(p.AsSpan(1, 6))).LogPdf(Symmetric(p.AsSpan(7))));
    }

    [Fact]
    public void RepeatedPackedDensitiesReuseRecordingStorage()
    {
        var tape = new Tape();
        for (int i = 0; i < 4; i++)
        {
            Evaluate();
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        Evaluate();
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);

        void Evaluate()
        {
            tape.Reset();
            ScalarNode shape = tape.Leaf(1.7);
            ScalarNode entry = tape.Leaf(0.3);
            ScalarNode result = LkjCorrelationDistribution.LogPdf(2, shape, [entry])
                + WishartDistribution.LogPdf(2, 5, [1.0, 0.2, 1.3], [2.0, entry, 3.0])
                + DirichletDistribution.LogPdf([shape, 2], [entry, 1 - entry])
                + MultivariateNormalDistribution.LogPdfCholesky([entry], [1], [0]);
            tape.Backward(result);
        }
    }

    [Fact]
    public void LkjCholeskyDensityIncludesTheJacobianToCorrelationCoordinates()
    {
        var tape = new Tape();
        ScalarNode shape = tape.Leaf(1.7);
        ScalarNode diagonal = tape.Leaf(0.8);
        var factor = LkjCorrelationDistribution.LogPdfCholesky(3, shape, [1, 0.6, diagonal, 0, 0, 1]);
        var correlation = LkjCorrelationDistribution.LogPdf(3, shape, [0.6, 0, 0]);
        Assert.InRange(Math.Abs(factor.Value - correlation.Value - Math.Log(0.8)), 0, 2e-15);
        tape.Backward(factor);
        Assert.InRange(Math.Abs(tape.GetGradient(diagonal) - 3), 0, 2e-15);
    }

    private static Matrix<double> Symmetric(ReadOnlySpan<double> p) =>
        new(new double[,] { { p[0], p[1], p[3] }, { p[1], p[2], p[4] }, { p[3], p[4], p[5] } });

    private static void Verify(double[] point, Func<ReadOnlySpan<ScalarNode>, ScalarNode> model, Func<double[], double> reference)
    {
        var tape = new Tape();
        var leaves = tape.Leaves(new ReadOnlyVectorSpan<double>(point));
        ScalarNode result = model(leaves);
        double expected = reference(point);
        Assert.Equal(BitConverter.DoubleToInt64Bits(expected), BitConverter.DoubleToInt64Bits(result.Value));
        tape.Backward(result);

        for (int i = 0; i < point.Length; i++)
        {
            const double step = 1e-5;
            double original = point[i];
            point[i] = original + step;
            double plus = reference(point);
            point[i] = original - step;
            double minus = reference(point);
            point[i] = original;

            double difference = (plus - minus) / (2 * step);
            double gradient = tape.GetGradient(leaves[i]);
            Assert.InRange(Math.Abs(gradient - difference) / Math.Max(1, Math.Abs(difference)), 0, 1e-8);
        }
    }
}
