using System.Buffers;
using Tuna.Differentiation;

namespace Tuna.Statistics.Distributions;

public sealed partial class LkjCorrelationDistribution
{
    /// <summary>
    /// Records the LKJ log density of a correlation matrix and its derivatives.
    /// <para>相関行列に対する LKJ 分布の対数密度とその微分を記録します。</para>
    /// </summary>
    /// <param name="dimension">The matrix order, at least two. / 2 以上の行列次数。</param>
    /// <param name="shape">The positive finite shape. / 正で有限の形状。</param>
    /// <param name="correlations">The strict lower triangle packed by rows. The diagonal is one. / 対角を除く下三角成分を行の順に並べた相関。対角は 1。</param>
    /// <returns>The log density node. / 対数密度を表すノード。</returns>
    /// <exception cref="ArgumentOutOfRangeException">The dimension is less than two. / 次元が 2 未満の場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">The shape is outside its domain. / 形状が定義域外の場合。</exception>
    /// <exception cref="ArgumentException">The number of correlations does not match the matrix order. / 相関の数が行列次数と一致しない場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(int dimension, ScalarNode shape, ReadOnlySpan<ScalarNode> correlations)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(dimension, 2);
        double eta = shape.Value;
        DistributionArgument.PositiveFinite(eta, nameof(shape), forDifferentiation: true);
        if (correlations.Length != PackedLogPdf.PackedLength(dimension - 1))
        {
            throw new ArgumentException("The packed correlations must match the matrix order.", nameof(correlations));
        }

        int size = checked(dimension * dimension);
        int length = checked(2 * size + correlations.Length);
        double[]? rented = length > MultivariateMath.StackLimit ? ArrayPool<double>.Shared.Rent(length) : null;
        Span<double> buffer = rented is null ? stackalloc double[length] : rented.AsSpan(0, length);
        try
        {
            Span<double> factor = buffer[..size];
            var gradient = buffer[(2 * size)..];
            int p = 0;
            bool nan = false, finite = true;
            for (int row = 0; row < dimension; row++)
            {
                for (int j = 0; j < row; j++)
                {
                    double element = correlations[p++].Value;
                    nan |= double.IsNaN(element);
                    finite &= double.IsFinite(element);
                    factor[j * dimension + row] = factor[row * dimension + j] = element;
                }

                factor[row * dimension + row] = 1;
            }

            bool onSupport = finite && MultivariateMath.FactorInPlace(factor, dimension);
            double logDeterminant = onSupport ? FactorLogDeterminant(factor, dimension) : double.NegativeInfinity;
            double value = nan ? double.NaN : onSupport
                ? (eta - 1) * logDeterminant + LogNormalization(dimension, eta) : double.NegativeInfinity;
            if (!shape.RequiresGradient && !PackedLogPdf.RequiresGradient(correlations))
            {
                return value;
            }

            double dShape = nan ? double.NaN : 0;
            if (onSupport)
            {
                dShape = shape.RequiresGradient ? ShapeDerivative(dimension, eta, logDeterminant) : 0;

                var inverse = buffer.Slice(size, size);
                if (eta != 1 && PackedLogPdf.RequiresGradient(correlations))
                {
                    PackedLogPdf.Inverse(factor, dimension, inverse);
                    p = 0;
                    for (int row = 1; row < dimension; row++)
                    {
                        for (int j = 0; j < row; j++)
                        {
                            gradient[p++] = (eta - 1) * (2 * inverse[j * dimension + row]);
                        }
                    }
                }
                else
                {
                    gradient.Clear();
                }
            }
            else
            {
                gradient.Fill(dShape);
            }

            return PackedLogPdf.Record(value, [shape], [dShape], correlations, gradient);
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<double>.Shared.Return(rented);
            }
        }
    }

    /// <summary>
    /// Records the LKJ log density in its correlation Cholesky coordinates.
    /// <para>相関行列の Cholesky 因子を座標とする LKJ 分布の対数密度とその微分を記録します。</para>
    /// </summary>
    /// <param name="dimension">The matrix order, at least two. / 2 以上の行列次数。</param>
    /// <param name="shape">The positive finite shape. / 正で有限の形状。</param>
    /// <param name="lowerFactor">The lower triangle packed by rows, including the diagonal. / 対角を含む下三角成分を行の順に並べた因子。</param>
    /// <returns>The log density node, including the log Jacobian from the factor to the correlation matrix. / 因子から相関行列への対数ヤコビアンを含む対数密度ノード。</returns>
    /// <remarks>
    /// <para>Each row must have unit norm and a positive diagonal entry. Only the diagonal factor entries have nonzero local derivatives within the support.</para>
    /// <para>各行のノルムは 1 で対角成分が正です。台の内側では因子の対角成分だけが非ゼロの局所微分を持ちます。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">The dimension is less than two. / 次元が 2 未満の場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">The shape is outside its domain. / 形状が定義域外の場合。</exception>
    /// <exception cref="ArgumentException">The number of entries does not match the matrix order. / 成分数が行列次数と一致しない場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdfCholesky(int dimension, ScalarNode shape, ReadOnlySpan<ScalarNode> lowerFactor)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(dimension, 2);
        double eta = shape.Value;
        DistributionArgument.PositiveFinite(eta, nameof(shape), forDifferentiation: true);
        int length = PackedLogPdf.PackedLength(dimension);
        if (lowerFactor.Length != length)
        {
            throw new ArgumentException("The packed factor must match the matrix order.", nameof(lowerFactor));
        }

        bool onSupport = true, nan = false;
        int index = 0;
        double logDeterminant = 0, logJacobian = 0;
        for (int row = 0; row < dimension; row++)
        {
            double normSquared = 0;
            for (int column = 0; column <= row; column++)
            {
                double entry = lowerFactor[index++].Value;
                nan |= double.IsNaN(entry);
                normSquared = Math.FusedMultiplyAdd(entry, entry, normSquared);
            }

            double diagonal = lowerFactor[index - 1].Value;
            bool positive = diagonal > 0;
            onSupport &= positive && Math.Abs(normSquared - 1) <= (row + 1) * MultivariateMath.SymmetryTolerance;
            if (positive && row > 0)
            {
                double logarithm = Math.Log(diagonal);
                logDeterminant += 2 * logarithm;
                logJacobian += (dimension - 1 - row) * logarithm;
            }
        }

        double value = nan ? double.NaN : onSupport
            ? (eta - 1) * logDeterminant + logJacobian + LogNormalization(dimension, eta)
            : double.NegativeInfinity;
        if (!shape.RequiresGradient && !PackedLogPdf.RequiresGradient(lowerFactor))
        {
            return value;
        }

        double[]? rented = length > MultivariateMath.StackLimit ? ArrayPool<double>.Shared.Rent(length) : null;
        Span<double> gradient = rented is null ? stackalloc double[length] : rented.AsSpan(0, length);
        try
        {
            gradient.Fill(nan ? double.NaN : 0);
            if (onSupport)
            {
                int diagonal = 0;
                for (int row = 1; row < dimension; row++)
                {
                    diagonal += row + 1;
                    gradient[diagonal] = (dimension - 1 - row + 2 * (eta - 1)) / lowerFactor[diagonal].Value;
                }
            }

            double dShape = nan ? double.NaN : onSupport && shape.RequiresGradient
                ? ShapeDerivative(dimension, eta, logDeterminant) : 0;
            return PackedLogPdf.Record(value, [shape], [dShape], lowerFactor, gradient);
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<double>.Shared.Return(rented);
            }
        }
    }

    private static double ShapeDerivative(int dimension, double eta, double logDeterminant)
    {
        double derivative = logDeterminant;
        for (int k = 1; k < dimension; k++)
        {
            double beta = eta + 0.5 * (dimension - k - 1);
            derivative -= (dimension - k) * XMath.Special.LogBetaDerivative(beta, 0.5);
        }

        return derivative;
    }
}
