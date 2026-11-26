using System.Buffers;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public sealed partial class MultivariateNormalDistribution
{
    /// <summary>
    /// Records a multivariate normal log density parameterized by the covariance Cholesky factor.
    /// <para>共分散の Cholesky 因子を指定し、多変量正規分布の対数密度とその微分を記録します。</para>
    /// </summary>
    /// <param name="mean">The finite mean vector. / 有限の平均ベクトル。</param>
    /// <param name="lowerFactor">The lower-triangular factor, packed by rows including its positive diagonal. / 正の対角を含む下三角成分を行の順に並べた因子。</param>
    /// <param name="x">The observation. / 観測値。</param>
    /// <returns>The log density node. / 対数密度を表すノード。</returns>
    /// <exception cref="ArgumentException">The dimensions disagree or the mean is empty. / 次元が一致しないか平均が空の場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">The mean or factor is nonfinite or the diagonal is nonpositive. / 平均や因子が有限でないか、対角が正でない場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdfCholesky(ReadOnlySpan<ScalarNode> mean, ReadOnlySpan<ScalarNode> lowerFactor, ReadOnlySpan<ScalarNode> x)
    {
        int n = mean.Length;
        if (n == 0 || x.Length != n || lowerFactor.Length != PackedLogPdf.PackedLength(n))
        {
            throw new ArgumentException("The mean, observation, and packed factor dimensions must agree.");
        }

        int length = checked(n * n + 5 * n + lowerFactor.Length);
        double[]? rented = length > MultivariateMath.StackLimit ? ArrayPool<double>.Shared.Rent(length) : null;
        Span<double> buffer = rented is null ? stackalloc double[length] : rented.AsSpan(0, length);
        try
        {
            var factor = buffer[..(n * n)];
            var vectors = buffer[factor.Length..];
            var m = vectors[..n];
            var observations = vectors.Slice(n, n);
            var z = vectors.Slice(2 * n, n);
            var dMean = vectors.Slice(3 * n, n);
            var dX = vectors.Slice(4 * n, n);
            var dFactor = vectors[(5 * n)..];

            int p = 0;
            for (int i = 0; i < n; i++)
            {
                m[i] = mean[i].Value;
                if (!double.IsFinite(m[i]))
                {
                    throw new DistributionParameterOutOfRangeException(nameof(mean), m[i], "The mean entries must be finite.");
                }

                observations[i] = x[i].Value;
                for (int j = 0; j <= i; j++)
                {
                    double element = lowerFactor[p++].Value;
                    if (!double.IsFinite(element) || (i == j && !(element > 0)))
                    {
                        throw new DistributionParameterOutOfRangeException(nameof(lowerFactor), element, "The factor must be finite with a positive diagonal.");
                    }

                    factor[j * n + i] = element;
                }
            }

            double status = MultivariateMath.Standardize(new ReadOnlyVectorSpan<double>(observations), m, factor, z);
            double value = status == 0
                ? -n * DistributionMath.LnSqrt2Pi - MultivariateMath.SumLogDiagonal(factor, n) - MultivariateMath.HalfSquaredNorm(z)
                : double.IsNaN(status) ? double.NaN : double.NegativeInfinity;
            if (!(PackedLogPdf.RequiresGradient(mean) || PackedLogPdf.RequiresGradient(lowerFactor) || PackedLogPdf.RequiresGradient(x)))
            {
                return value;
            }

            if (status != 0)
            {
                dMean.Fill(double.NaN);
                dFactor.Fill(double.NaN);
                dX.Fill(double.NaN);
            }
            else
            {
                z.CopyTo(dMean);
                PackedLogPdf.SolveTranspose(factor, dMean, dX);
                p = 0;
                for (int i = 0; i < n; i++)
                {
                    double numerator = dX[i], diagonal = factor[i * n + i];
                    dX[i] = -dMean[i];
                    for (int j = 0; j < i; j++)
                    {
                        dFactor[p++] = double.IsInfinity(dMean[i]) && double.IsFinite(numerator)
                            ? DistributionMath.ScaledRatio([numerator, z[j]], [diagonal]) : dMean[i] * z[j];
                    }

                    // 除算より先に差を取り、対角が小さい場合の ∞ - ∞ を避ける。
                    dFactor[p++] = Math.FusedMultiplyAdd(numerator, z[i], -1) / diagonal;
                }
            }

            return PackedLogPdf.Record(value, mean, dMean, lowerFactor, dFactor, x, dX);
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<double>.Shared.Return(rented);
            }
        }
    }
}
