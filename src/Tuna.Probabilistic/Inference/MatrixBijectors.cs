using System.Buffers;
using Tuna.LinearAlgebra;

namespace Tuna.Probabilistic.Inference;

/// <summary>
/// Creates constraints for covariance matrices and Cholesky factors.
/// <para>共分散行列と Cholesky 因子の制約変換を作成します。</para>
/// </summary>
/// <remarks>
/// <para>The constrained vector packs the lower triangle by rows, including the diagonal.</para>
/// <para>制約付きベクトルは、対角を含む下三角成分を行の順に格納します。</para>
/// </remarks>
public static class MatrixBijectors
{
    /// <summary>
    /// Creates a positive-definite covariance matrix transform.
    /// <para>正定値の共分散行列への変換を作成します。</para>
    /// </summary>
    /// <param name="order">The positive matrix order. / 正の行列次数。</param>
    /// <returns>A transform with one parameter per lower-triangular entry. / 下三角成分ごとに 1 つのパラメータを持つ変換。</returns>
    /// <remarks>
    /// <para>The inverse returns NaN for matrices that do not admit a positive-diagonal Cholesky factor.</para>
    /// <para>正の対角を持つ Cholesky 因子が存在しない行列の逆変換は NaN になります。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">The order is nonpositive or the matrix is too large. / 次数が正でないか、行列が大きすぎる場合。</exception>
    public static IBijector Covariance(int order) => new CovarianceTransform(order, factorOnly: false);

    /// <summary>
    /// Creates a transform to a lower-triangular factor with positive diagonal.
    /// <para>正の対角を持つ下三角の Cholesky 因子への変換を作成します。</para>
    /// </summary>
    /// <param name="order">The positive matrix order. / 正の行列次数。</param>
    /// <returns>A transform with one parameter per lower-triangular entry. / 下三角成分ごとに 1 つのパラメータを持つ変換。</returns>
    /// <exception cref="ArgumentOutOfRangeException">The order is nonpositive or the packed factor is too large. / 次数が正でないか、因子が大きすぎる場合。</exception>
    public static IBijector CovarianceCholesky(int order) => new CovarianceTransform(order, factorOnly: true);

    /// <summary>
    /// Creates a transform to a correlation matrix's Cholesky factor.
    /// <para>相関行列の Cholesky 因子への変換を作成します。</para>
    /// </summary>
    /// <param name="order">The positive matrix order. / 正の行列次数。</param>
    /// <returns>A transform whose parameters represent strict lower-triangular entries. / 狭義下三角成分にパラメータを持つ変換。</returns>
    /// <remarks>
    /// <para>The log Jacobian uses the strict lower-triangular entries as independent coordinates.</para>
    /// <para>対角を除く下三角成分を独立座標としてヤコビアンを計算します。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">The order is nonpositive or the packed factor is too large. / 次数が正でないか、因子が大きすぎる場合。</exception>
    public static IBijector CorrelationCholesky(int order) => new CorrelationTransform(order);

    private static int PackedLength(int order)
    {
        long length = (long)order * (order + 1L) / 2;
        if (order <= 0 || length > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(order));
        }

        return (int)length;
    }

    private static int RowStart(int row) => (int)((long)row * (row + 1L) / 2);

    private sealed class CovarianceTransform : IBijector
    {
        private readonly int order;
        private readonly bool factorOnly;

        internal CovarianceTransform(int order, bool factorOnly)
        {
            if (!factorOnly && (long)order * order > int.MaxValue)
            {
                throw new ArgumentOutOfRangeException(nameof(order));
            }

            this.order = order;
            this.factorOnly = factorOnly;
            ConstrainedDimension = PackedLength(order);
        }

        public int UnconstrainedDimension => ConstrainedDimension;
        public int ConstrainedDimension { get; }

        public double Forward(ReadOnlyVectorSpan<double> unconstrained, VectorSpan<double> constrained)
        {
            Bijection.RequireLengths(this, unconstrained.Length, constrained.Length);
            if (factorOnly)
            {
                double jacobian = 0;
                for (int row = 0; row < order; row++)
                {
                    int start = RowStart(row);
                    for (int j = 0; j < row; j++)
                    {
                        constrained[start + j] = unconstrained[start + j];
                    }

                    double z = unconstrained[start + row];
                    constrained[start + row] = Math.Exp(z);
                    jacobian += z;
                }

                return jacobian;
            }

            double result = order * Math.Log(2);
            for (int row = 0; row < order; row++)
            {
                int start = RowStart(row);
                for (int column = 0; column <= row; column++)
                {
                    double value = 0;
                    for (int j = 0; j <= column; j++)
                    {
                        value += Factor(unconstrained, row, j) * Factor(unconstrained, column, j);
                    }

                    constrained[start + column] = value;
                }

                result += (order + 1 - row) * unconstrained[start + row];
            }

            return result;
        }

        public void Inverse(ReadOnlyVectorSpan<double> constrained, VectorSpan<double> unconstrained)
        {
            Bijection.RequireLengths(this, unconstrained.Length, constrained.Length);
            if (factorOnly)
            {
                for (int row = 0; row < order; row++)
                {
                    int start = RowStart(row);
                    for (int j = 0; j < row; j++)
                    {
                        unconstrained[start + j] = constrained[start + j];
                    }

                    unconstrained[start + row] = Math.Log(constrained[start + row]);
                }

                return;
            }

            var matrix = new Matrix<double>(order, order);
            for (int row = 0; row < order; row++)
            {
                for (int j = 0; j <= row; j++)
                {
                    matrix[row, j] = matrix[j, row] = constrained[RowStart(row) + j];
                }
            }

            if (!CholeskyDecomposition.TryCompute(matrix, out var factor))
            {
                unconstrained.Fill(double.NaN);
                return;
            }

            for (int row = 0; row < order; row++)
            {
                int start = RowStart(row);
                for (int j = 0; j < row; j++)
                {
                    unconstrained[start + j] = factor.L[row, j];
                }

                unconstrained[start + row] = Math.Log(factor.L[row, row]);
            }
        }

        public void PullBack(ReadOnlyVectorSpan<double> unconstrained, ReadOnlyVectorSpan<double> constrainedGradient, VectorSpan<double> unconstrainedGradient)
        {
            Bijection.RequireLengths(this, unconstrained.Length, constrainedGradient.Length);
            Bijection.RequireLengths(this, unconstrainedGradient.Length, constrainedGradient.Length);
            for (int row = 0; row < order; row++)
            {
                int start = RowStart(row);
                for (int column = 0; column <= row; column++)
                {
                    double gradient = constrainedGradient[start + column];
                    if (!factorOnly)
                    {
                        // 下三角の独立成分に対して、d(LLᵀ) = dL Lᵀ + L dLᵀ を縮約する。
                        gradient = 2 * constrainedGradient[start + row] * Factor(unconstrained, row, column);
                        for (int j = column; j < order; j++)
                        {
                            if (j != row)
                            {
                                int packed = RowStart(Math.Max(row, j)) + Math.Min(row, j);
                                gradient += constrainedGradient[packed] * Factor(unconstrained, j, column);
                            }
                        }
                    }

                    if (row == column)
                    {
                        gradient = gradient * Factor(unconstrained, row, row) + (factorOnly ? 1 : order + 1 - row);
                    }

                    unconstrainedGradient[start + column] = gradient;
                }
            }
        }

        private static double Factor(ReadOnlyVectorSpan<double> values, int row, int column)
        {
            double value = values[RowStart(row) + column];
            return row == column ? Math.Exp(value) : value;
        }
    }

    private sealed class CorrelationTransform(int order) : IBijector
    {
        public int ConstrainedDimension { get; } = PackedLength(order);
        public int UnconstrainedDimension => ConstrainedDimension - order;

        public double Forward(ReadOnlyVectorSpan<double> unconstrained, VectorSpan<double> constrained)
        {
            Bijection.RequireLengths(this, unconstrained.Length, constrained.Length);
            double jacobian = 0;
            for (int row = 0; row < order; row++)
            {
                int start = RowStart(row), free = start - row;
                double remaining = 1;
                for (int j = 0; j < row; j++)
                {
                    double z = unconstrained[free + j];
                    double logSech = LogSech(z);
                    constrained[start + j] = remaining * Math.Tanh(z);
                    remaining *= Math.Exp(logSech);
                    jacobian += (row + 1 - j) * logSech;
                }

                constrained[start + row] = remaining;
            }

            return jacobian;
        }

        public void Inverse(ReadOnlyVectorSpan<double> constrained, VectorSpan<double> unconstrained)
        {
            Bijection.RequireLengths(this, unconstrained.Length, constrained.Length);
            for (int row = 0; row < order; row++)
            {
                int start = RowStart(row), free = start - row;
                double tail = constrained[start + row];
                if (!(tail >= 0 && double.IsFinite(tail)))
                {
                    unconstrained.Fill(double.NaN);
                    return;
                }

                for (int j = row - 1; j >= 0; j--)
                {
                    double entry = constrained[start + j];
                    double norm = double.Hypot(entry, tail);
                    double ratio = entry / tail;
                    // asinh(Lᵢⱼ / ‖Lᵢ,ⱼ₊₁:ᵢ‖) は tanh が 1 に丸まっても因子から復元できる。
                    unconstrained[free + j] = double.IsFinite(ratio) ? Math.Asinh(ratio)
                        : tail == 0 ? entry == 0 ? 0 : Math.CopySign(double.PositiveInfinity, entry)
                        : Math.CopySign(Math.Log(norm) + XMath.Log1p<double>(Math.Abs(entry) / norm) - Math.Log(tail), entry);
                    tail = norm;
                }

                if (!(Math.Abs(tail - 1) <= 1e-12))
                {
                    unconstrained.Fill(double.NaN);
                    return;
                }
            }
        }

        public void PullBack(ReadOnlyVectorSpan<double> unconstrained, ReadOnlyVectorSpan<double> constrainedGradient, VectorSpan<double> unconstrainedGradient)
        {
            Bijection.RequireLengths(this, unconstrained.Length, constrainedGradient.Length);
            Bijection.RequireLengths(this, unconstrainedGradient.Length, constrainedGradient.Length);
            double[]? rented = order > Bijection.StackLimit ? ArrayPool<double>.Shared.Rent(order) : null;
            Span<double> prefix = rented is null ? stackalloc double[order] : rented.AsSpan(0, order);
            try
            {
                for (int row = 1; row < order; row++)
                {
                    int start = RowStart(row), free = start - row;
                    prefix[0] = 1;
                    for (int j = 0; j < row; j++)
                    {
                        prefix[j + 1] = prefix[j] * Math.Exp(LogSech(unconstrained[free + j]));
                    }

                    double suffix = constrainedGradient[start + row] * prefix[row];
                    for (int j = row - 1; j >= 0; j--)
                    {
                        double z = unconstrained[free + j], rho = Math.Tanh(z);
                        double weight = constrainedGradient[start + j] * prefix[j];
                        unconstrainedGradient[free + j] = weight * Math.Exp(2 * LogSech(z))
                            - rho * (suffix + row + 1 - j);
                        suffix += weight * rho;
                    }
                }
            }
            finally
            {
                if (rented is not null)
                {
                    ArrayPool<double>.Shared.Return(rented);
                }
            }
        }

        private static double LogSech(double input)
        {
            double magnitude = Math.Abs(input);
            if (magnitude < 0.5)
            {
                double correlation = Math.Tanh(input);
                return 0.5 * XMath.Log1p<double>(-correlation * correlation);
            }

            return Math.Log(2) - magnitude - XMath.Log1p<double>(Math.Exp(-2 * magnitude));
        }
    }
}
