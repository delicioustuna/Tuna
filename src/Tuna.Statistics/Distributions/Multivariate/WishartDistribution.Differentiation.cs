using System.Buffers;
using Tuna.Differentiation;

namespace Tuna.Statistics.Distributions;

public sealed partial class WishartDistribution
{
    /// <summary>
    /// Records the Wishart log density and its derivatives.
    /// <para>ウィシャート分布の対数密度とその微分を記録します。</para>
    /// </summary>
    /// <param name="order">The positive matrix order. / 正の行列次数。</param>
    /// <param name="degreesOfFreedom">The finite degrees of freedom, greater than the order minus one. / 行列次数から 1 を引いた値より大きい有限の自由度。</param>
    /// <param name="scale">The positive-definite scale matrix, packed by rows of the lower triangle. / 下三角成分を行の順に並べた正定値の尺度行列。</param>
    /// <param name="x">The observation, in the same packed format. / 尺度行列と同じ形式で格納した観測行列。</param>
    /// <returns>The log density node. / 対数密度を表すノード。</returns>
    /// <exception cref="ArgumentOutOfRangeException">The order is not positive. / 次数が正でない場合。</exception>
    /// <exception cref="ArgumentException">The matrix dimensions disagree. / 次元が一致しない場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">The degrees of freedom or scale is outside its domain. / 自由度または尺度行列が定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(int order, ScalarNode degreesOfFreedom, ReadOnlySpan<ScalarNode> scale, ReadOnlySpan<ScalarNode> x)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(order);
        int packed = PackedLogPdf.PackedLength(order);
        if (scale.Length != packed || x.Length != packed)
        {
            throw new ArgumentException("The packed scale and observation must match the order.");
        }

        double nu = degreesOfFreedom.Value;
        if (!(nu > order - 1 && double.IsFinite(nu)))
        {
            throw new DistributionParameterOutOfRangeException(nameof(degreesOfFreedom), nu, "The degrees of freedom must be finite and greater than the order minus one.");
        }

        int size = checked(order * order);
        int length = checked(5 * size + 2 * packed);
        double[]? rented = length > MultivariateMath.StackLimit ? ArrayPool<double>.Shared.Rent(length) : null;
        Span<double> buffer = rented is null ? stackalloc double[length] : rented.AsSpan(0, length);
        try
        {
            Span<double> l = buffer[..size];
            Span<double> observed = buffer.Slice(size, size);
            Span<double> whitened = buffer.Slice(2 * size, size);
            Span<double> scaleInverse = buffer.Slice(3 * size, size);
            Span<double> observedInverse = buffer.Slice(4 * size, size);
            Span<double> dScale = buffer.Slice(5 * size, packed);
            Span<double> dX = buffer[(5 * size + packed)..];

            if (!ReadSymmetric(scale, order, l, out _))
            {
                throw new DistributionParameterOutOfRangeException(nameof(scale), null, "The scale entries must be finite.");
            }

            if (!MultivariateMath.FactorInPlace(l, order))
            {
                throw new DistributionParameterOutOfRangeException(nameof(scale), null, "The scale must be positive definite.");
            }

            bool onSupport = ReadSymmetric(x, order, observed, out bool nan) && MultivariateMath.FactorInPlace(observed, order);
            double value;
            if (onSupport)
            {
                observed.CopyTo(whitened);
                double halfLogDet = MultivariateMath.SumLogDiagonal(l, order);
                double logNormalization = LogNormalization(order, halfLogDet);
                value = FactorLogPdf(nu, l, order, logNormalization, whitened);
            }
            else
            {
                value = nan ? double.NaN : double.NegativeInfinity;
            }

            bool differentiateScale = PackedLogPdf.RequiresGradient(scale);
            bool differentiateX = PackedLogPdf.RequiresGradient(x);
            if (!degreesOfFreedom.RequiresGradient && !differentiateScale && !differentiateX)
            {
                return value;
            }

            double dNu = nan ? double.NaN : 0;
            if (!onSupport)
            {
                dScale.Fill(dNu);
                dX.Fill(dNu);
                return PackedLogPdf.Record(value, [degreesOfFreedom], [dNu], scale, dScale, x, dX);
            }

            if (degreesOfFreedom.RequiresGradient)
            {
                for (int i = 0; i < order; i++)
                {
                    int diagonal = i * order + i;
                    double entry = whitened[diagonal];
                    double logRatio = entry >= DistributionMath.MinNormal && double.IsFinite(entry)
                        ? Math.Log(entry) : Math.Log(observed[diagonal]) - Math.Log(l[diagonal]);
                    double halfDegree = 0.5 * (nu - i);
                    dNu += logRatio - 0.5 * (DistributionMath.Ln2 + XMath.Special.Digamma(halfDegree));
                }
            }

            dScale.Clear();
            dX.Clear();
            if (differentiateScale || differentiateX)
            {
                PackedLogPdf.Inverse(l, order, scaleInverse);
            }

            if (differentiateX)
            {
                PackedLogPdf.Inverse(observed, order, observedInverse);
                FillObservationGradient(nu, order, scaleInverse, observedInverse, dX);
            }

            if (differentiateScale)
            {
                FillScaleGradient(nu, order, l, whitened, scaleInverse, dScale);
            }

            return PackedLogPdf.Record(value, [degreesOfFreedom], [dNu], scale, dScale, x, dX);
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<double>.Shared.Return(rented);
            }
        }
    }

    private static bool ReadSymmetric(ReadOnlySpan<ScalarNode> inputs, int order, Span<double> destination, out bool nan)
    {
        nan = false;
        bool finite = true;
        int index = 0;
        for (int row = 0; row < order; row++)
        {
            for (int j = 0; j <= row; j++)
            {
                double value = inputs[index++].Value;
                nan |= double.IsNaN(value);
                finite &= double.IsFinite(value);
                destination[j * order + row] = destination[row * order + j] = value;
            }
        }

        return finite;
    }

    private static void FillObservationGradient(double nu, int order, ReadOnlySpan<double> inverseScale,
        ReadOnlySpan<double> inverseObservation, Span<double> gradient)
    {
        // 独立な下三角成分のうち、非対角成分は対称な 2 要素を変える。
        int index = 0;
        for (int row = 0; row < order; row++)
        {
            for (int column = 0; column <= row; column++)
            {
                int entry = column * order + row;
                double difference = Math.FusedMultiplyAdd(nu - order - 1, inverseObservation[entry], -inverseScale[entry]);
                gradient[index++] = (row == column ? 0.5 : 1) * difference;
            }
        }
    }

    private static void FillScaleGradient(double nu, int order, ReadOnlySpan<double> factor, Span<double> whitened,
        ReadOnlySpan<double> inverseScale, Span<double> gradient)
    {
        // A = L⁻¹ C から B = L⁻ᵀ A = Σ⁻¹ C を計算する。Σ⁻¹ X Σ⁻¹ = B Bᵀ。
        for (int j = 0; j < order; j++)
        {
            PackedLogPdf.SolveTranspose(factor, whitened.Slice(j * order, order));
        }

        int index = 0;
        for (int row = 0; row < order; row++)
        {
            for (int column = 0; column <= row; column++)
            {
                double gram = 0;
                for (int j = 0; j < order; j++)
                {
                    gram += whitened[j * order + row] * whitened[j * order + column];
                }

                double difference = Math.FusedMultiplyAdd(-nu, inverseScale[column * order + row], gram);
                gradient[index++] = (row == column ? 0.5 : 1) * difference;
            }
        }
    }
}
