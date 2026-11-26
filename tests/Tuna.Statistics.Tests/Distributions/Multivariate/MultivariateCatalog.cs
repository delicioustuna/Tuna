using Tuna.LinearAlgebra;
using Tuna.Statistics.Distributions;

namespace Tuna.Tests.Statistics;

/// <summary>
/// 参照値のパラメータ（tools/DistributionReferences/generate_multivariate_references.py の詰め方）から多変量分布を作る。
/// 行列は列優先で、次元・試行回数を表す整数と離散分布の評価点も double で持つ。
/// </summary>
internal static class MultivariateCatalog
{
    internal static object Create(string name, double[] p)
    {
        switch (name)
        {
            case "MultivariateNormal":
                {
                    int d = (int)p[0];
                    return new MultivariateNormalDistribution(p.AsSpan(1, d), Square(p, 1 + d, d));
                }

            case "MultivariateStudentT":
                {
                    int d = (int)p[1];
                    return new MultivariateStudentTDistribution(p[0], p.AsSpan(2, d), Square(p, 2 + d, d));
                }

            case "Wishart":
                return new WishartDistribution(p[0], Square(p, 2, (int)p[1]));
            case "InverseWishart":
                return new InverseWishartDistribution(p[0], Square(p, 2, (int)p[1]));
            case "Dirichlet":
                return new DirichletDistribution(p);
            case "MatrixNormal":
                {
                    int n = (int)p[0];
                    int c = (int)p[1];
                    return new MatrixNormalDistribution(
                        new ReadOnlyMatrixSpan<double>(p.AsSpan(2, n * c), n, c), Square(p, 2 + n * c, n), Square(p, 2 + n * c + n * n, c));
                }

            case "Lkj":
                return new LkjCorrelationDistribution((int)p[0], p[1]);
            case "Multinomial":
                return new MultinomialDistribution((long)p[0], p.AsSpan(1));
            case "DirichletMultinomial":
                return new DirichletMultinomialDistribution((long)p[0], p.AsSpan(1));
            case "NegativeMultinomial":
                return new NegativeMultinomialDistribution(p[0], p[1], p.AsSpan(2));
            default:
                throw new ArgumentOutOfRangeException(nameof(name), name, null);
        }
    }

    /// <summary>評価点（列優先）での密度の対数</summary>
    internal static double LogPdf(object distribution, double[] point) => distribution switch
    {
        IMultivariateContinuousDistribution v => v.LogPdf(point),
        IMatrixContinuousDistribution m => m.LogPdf(new ReadOnlyMatrixSpan<double>(point, m.Rows, m.Columns)),
        _ => throw new ArgumentException("Unknown distribution type.", nameof(distribution)),
    };

    /// <summary>評価点（整数の値を double で持つ）での確率の対数</summary>
    internal static double LogPmf(object distribution, double[] point) =>
        ((IMultivariateDiscreteDistribution)distribution).LogPmf(point.Select(v => checked((long)v)).ToArray());

    private static ReadOnlyMatrixSpan<double> Square(double[] p, int offset, int n) => new(p.AsSpan(offset, n * n), n, n);
}
