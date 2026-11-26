using System.Buffers;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public sealed partial class DirichletDistribution
{
    /// <summary>
    /// Records the Dirichlet log density and its derivatives.
    /// <para>ディリクレ分布の対数密度とその微分を記録します。</para>
    /// </summary>
    /// <param name="alpha">The positive finite concentrations with a finite sum. / 各成分と総和が正で有限の集中度。</param>
    /// <param name="x">The simplex coordinates. / 単体上の座標。</param>
    /// <returns>The log density node. / 対数密度を表すノード。</returns>
    /// <remarks>
    /// <para>Within the support tolerance, the observation derivatives include normalization by the sum of its coordinates.</para>
    /// <para>台の許容幅内の観測では、座標の総和による正規化を含めて微分します。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The vector lengths disagree or are less than two. / 長さが一致しないか、成分が 2 個未満の場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A concentration or their sum is outside its domain. / 集中度の成分または総和が範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ReadOnlySpan<ScalarNode> alpha, ReadOnlySpan<ScalarNode> x)
    {
        int n = alpha.Length;
        if (n < 2 || x.Length != n)
        {
            throw new ArgumentException("The concentrations and observation must have the same length, at least two.");
        }

        int length = checked(6 * n + 1);
        double[]? rented = length > MultivariateMath.StackLimit ? ArrayPool<double>.Shared.Rent(length) : null;
        Span<double> buffer = rented is null ? stackalloc double[length] : rented.AsSpan(0, length);
        try
        {
            var a = buffer[..n];
            var coordinates = buffer.Slice(n, n);
            var remaining = buffer.Slice(2 * n, n);
            var dAlpha = buffer.Slice(3 * n, n);
            var dX = buffer.Slice(4 * n, n);
            var suffix = buffer[(5 * n)..];
            suffix[n] = 0;
            var exponents = new CompensatedSum();
            for (int i = n - 1; i >= 0; i--)
            {
                double entry = alpha[i].Value;
                DistributionArgument.PositiveFinite(entry, nameof(alpha), forDifferentiation: true);
                a[i] = entry;
                suffix[i] = suffix[i + 1] + entry;
                exponents.AddDifference(entry, 1);
                coordinates[i] = x[i].Value;
            }

            if (!double.IsFinite(suffix[0]))
            {
                throw new DistributionParameterOutOfRangeException(nameof(alpha), suffix[0], "The sum of the concentrations must be finite.");
            }

            double value = LogPdfCore(new ReadOnlyVectorSpan<double>(coordinates), remaining, a, suffix, out double sum, out bool onSupport);
            if (!(PackedLogPdf.RequiresGradient(alpha) || PackedLogPdf.RequiresGradient(x)))
            {
                return value;
            }

            if (!onSupport || double.IsNaN(value))
            {
                double derivative = double.IsNaN(value) ? double.NaN : 0;
                dAlpha.Fill(derivative);
                dX.Fill(derivative);
            }
            else
            {
                double previous = 0, logSum = Math.Log(sum), normalization = exponents.Value / sum;
                for (int i = 0; i < n; i++)
                {
                    double entry = coordinates[i], exponent = a[i] - 1;
                    dX[i] = (exponent == 0 ? 0 : exponent / entry) - normalization;
                    dAlpha[i] = !alpha[i].RequiresGradient ? 0 : entry == 0 ? double.NegativeInfinity
                        : Math.Log(entry) - logSum - XMath.Special.LogBetaDerivative(a[i], previous + suffix[i + 1]);
                    previous += a[i];
                }
            }

            return PackedLogPdf.Record(value, alpha, dAlpha, x, dX);
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
