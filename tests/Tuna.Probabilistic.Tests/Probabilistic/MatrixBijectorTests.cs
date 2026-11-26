using Tuna.LinearAlgebra;
using Tuna.Probabilistic.Inference;

namespace Tuna.Tests.Probabilistic;

public class MatrixBijectorTests
{
    [Theory]
    [InlineData(4, 5)]
    [InlineData(3, -10)]
    [InlineData(2, 19)]
    [InlineData(6, 18)]
    public void CorrelationInverseUsesTheTailNormAtStrongCorrelations(int order, double value)
    {
        var bijector = MatrixBijectors.CorrelationCholesky(order);
        var point = new double[bijector.UnconstrainedDimension];
        Array.Fill(point, value);
        var factor = new double[bijector.ConstrainedDimension];
        var recovered = new double[point.Length];
        bijector.Forward(point, factor);
        bijector.Inverse(factor, recovered);
        Assert.All(recovered, coordinate => Assert.InRange(Math.Abs(coordinate - value), 0, 2e-14));
    }

    [Fact]
    public void CorrelationFactorLkjDensityCombinesWithTheTransform()
    {
        var transform = MatrixBijectors.CorrelationCholesky(3);
        var prior = LogDensities.FromAutomaticDifferentiation(
            factor => Tuna.Statistics.Distributions.LkjCorrelationDistribution.LogPdfCholesky(3, 1.7, factor));
        var density = LogDensities.FromAutomaticDifferentiation(
            p => LogDensities.RecordTransformed(prior, transform, p));
        Assert.InRange(LogDensities.GradientError(density, [0.2, -0.39, 0.61]), 0, 1e-8);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void RoundTripAndGradientMatchFiniteDifferences(int transform)
    {
        IBijector bijector = transform switch
        {
            0 => MatrixBijectors.Covariance(3),
            1 => MatrixBijectors.CovarianceCholesky(3),
            _ => MatrixBijectors.CorrelationCholesky(3),
        };
        double[] point = [-0.4, 0.31, 0.12, -0.25, 0.73, 0.19];
        Array.Resize(ref point, bijector.UnconstrainedDimension);
        double[] weights = [0.1, -0.5, 0.2, 1.1, -0.7, 0.3];
        var constrained = new double[bijector.ConstrainedDimension];
        var recovered = new double[point.Length];
        var gradient = new double[point.Length];
        double jacobian = bijector.Forward(point, constrained);
        bijector.Inverse(constrained, recovered);
        for (int i = 0; i < point.Length; i++)
        {
            Assert.InRange(Math.Abs(recovered[i] - point[i]), 0, 1e-14);
        }

        if (transform == 0)
        {
            double expected = 3 * Math.Log(2) + 4 * point[0] + 3 * point[2] + 2 * point[5];
            Assert.InRange(Math.Abs(expected - jacobian), 0, 1e-14);
        }
        else if (transform == 2)
        {
            Assert.Equal(1, constrained[0]);
            Assert.InRange(Math.Abs(constrained[1] * constrained[1] + constrained[2] * constrained[2] - 1), 0, 1e-14);
            Assert.InRange(Math.Abs(constrained[3] * constrained[3] + constrained[4] * constrained[4] + constrained[5] * constrained[5] - 1), 0, 1e-14);
        }

        bijector.PullBack(point, weights, gradient);
        for (int i = 0; i < point.Length; i++)
        {
            double old = point[i];
            const double h = 1e-5;
            point[i] = old + h;
            double plus = Evaluate();
            point[i] = old - h;
            double minus = Evaluate();
            point[i] = old;
            Assert.InRange(Math.Abs(gradient[i] - (plus - minus) / (2 * h)), 0, 1e-8);
        }

        // 同じ制約変換を自動微分モデルの一部として使う。
        var linear = LogDensities.FromAutomaticDifferentiation(x =>
        {
            var value = x[0] * weights[0];
            for (int j = 1; j < weights.Length; j++)
            {
                value += x[j] * weights[j];
            }

            return value;
        });
        var transformed = LogDensities.FromAutomaticDifferentiation(x => LogDensities.RecordTransformed(linear, bijector, x));
        Assert.InRange(LogDensities.GradientError(transformed, point), 0, 1e-7);

        double Evaluate()
        {
            double value = bijector.Forward(point, constrained);
            for (int i = 0; i < constrained.Length; i++)
            {
                value += weights[i] * constrained[i];
            }

            return value;
        }
    }
}
