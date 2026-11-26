using System.Buffers;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public sealed partial class CategoricalDistribution
{
    /// <summary>
    /// Records the categorical log mass and its weight derivatives.
    /// <para>カテゴリカル分布の対数質量と重みに関する微分を記録します。</para>
    /// </summary>
    /// <param name="weights">Nonnegative finite weights with a finite positive sum. / 総和が正で有限の、非負かつ有限の重み。</param>
    /// <param name="k">The integer category. / 整数のカテゴリ。</param>
    /// <returns>The log mass node. / 対数質量を表すノード。</returns>
    /// <exception cref="ArgumentException">No category weights are supplied. / カテゴリの重みが空の場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A weight or their sum is outside its domain. / 重みまたは総和が定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPmf(ReadOnlySpan<ScalarNode> weights, long k) => RecordProbability(weights, k, LogProbability.Density);

    /// <summary>
    /// Records the categorical log cumulative probability and its weight derivatives.
    /// <para>カテゴリカル分布の対数累積確率と重みに関する微分を記録します。</para>
    /// </summary>
    /// <param name="weights">Nonnegative finite weights with a finite positive sum. / 総和が正で有限の、非負かつ有限の重み。</param>
    /// <param name="k">The integer threshold included in the lower tail. / 下側の事象に含まれる整数のしきい値。</param>
    /// <returns>The log cumulative probability node. / 対数累積確率を表すノード。</returns>
    /// <exception cref="ArgumentException">No category weights are supplied. / カテゴリの重みが空の場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A weight or their sum is outside its domain. / 重みまたは総和が定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogCdf(ReadOnlySpan<ScalarNode> weights, long k) => RecordProbability(weights, k, LogProbability.LowerTail);

    /// <summary>
    /// Records the categorical log survival probability and its weight derivatives.
    /// <para>カテゴリカル分布の対数生存確率と重みに関する微分を記録します。</para>
    /// </summary>
    /// <param name="weights">Nonnegative finite weights with a finite positive sum. / 総和が正で有限の、非負かつ有限の重み。</param>
    /// <param name="k">The integer threshold excluded from the upper tail. / 上側の事象に含まれない整数のしきい値。</param>
    /// <returns>The log survival probability node. / 対数生存確率を表すノード。</returns>
    /// <exception cref="ArgumentException">No category weights are supplied. / カテゴリの重みが空の場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A weight or their sum is outside its domain. / 重みまたは総和が定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogSf(ReadOnlySpan<ScalarNode> weights, long k) => RecordProbability(weights, k, LogProbability.UpperTail);

    /// <summary>
    /// Records the categorical log likelihood of independent observations as a single operation.
    /// <para>カテゴリカル分布の独立な観測値の対数尤度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="weights">Nonnegative finite weights with a finite positive sum. / 総和が正で有限の、非負かつ有限の重み。</param>
    /// <param name="observations">The observations, summed in input order. / 入力順に集計する観測値。</param>
    /// <returns>The log likelihood node. / 対数尤度を表すノード。</returns>
    /// <exception cref="ArgumentException">No category weights are supplied. / カテゴリの重みが空の場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A weight or their sum is outside its domain. / 重みまたは総和が定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPmf(ReadOnlySpan<ScalarNode> weights, ReadOnlyVectorSpan<long> observations)
    {
        double total = ValidateWeights(weights);
        int n = weights.Length;
        int length = checked(2 * n);
        double[]? rented = length > MultivariateMath.StackLimit ? ArrayPool<double>.Shared.Rent(length) : null;
        Span<double> work = rented is null ? stackalloc double[length] : rented.AsSpan(0, length);
        try
        {
            Span<double> logMass = work[..n], counts = work[n..];
            counts.Clear();
            for (int j = 0; j < n; j++)
            {
                double p = weights[j].Value / total;
                logMass[j] = p <= 0.5 ? Math.Log(p) : RecordValue(weights, total, j, j + 1, LogProbability.Density).Value;
            }

            double value = 0;
            int validCount = 0;
            for (int i = 0; i < observations.Length; i++)
            {
                long k = observations[i];
                if (k < 0 || k >= n)
                {
                    value += double.NegativeInfinity;
                }
                else
                {
                    value += logMass[(int)k];
                    counts[(int)k]++;
                    validCount++;
                }
            }

            if (!PackedLogPdf.RequiresGradient(weights))
            {
                return value;
            }

            for (int j = 0; j < n; j++)
            {
                double weight = weights[j].Value, count = counts[j];
                if (count == 0)
                {
                    counts[j] = -(validCount / total);
                }
                else if (count == validCount && weight / total > 0.5)
                {
                    var (_, _, rest) = RecordValue(weights, total, j, j + 1, LogProbability.Density);
                    counts[j] = DistributionMath.ScaledRatio([count, rest], [weight, total]);
                }
                else
                {
                    counts[j] = Math.FusedMultiplyAdd(-validCount, weight / total, count) / weight;
                }
            }

            return PackedLogPdf.Record(value, weights, counts, default, default);
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<double>.Shared.Return(rented);
            }
        }
    }

    private static ScalarNode RecordProbability(ReadOnlySpan<ScalarNode> weights, long k, LogProbability kind)
    {
        double total = ValidateWeights(weights);
        int n = weights.Length;
        int index = k < 0 ? 0 : k >= n ? n : (int)k;
        int end = k < 0 ? 0 : k >= n ? n : index + 1;
        int start = kind == LogProbability.Density ? index : kind == LogProbability.UpperTail ? end : 0;
        if (kind == LogProbability.UpperTail)
        {
            end = n;
        }

        var (value, selected, rest) = RecordValue(weights, total, start, end, kind);
        if (!PackedLogPdf.RequiresGradient(weights))
        {
            return value;
        }

        double[]? rented = n > MultivariateMath.StackLimit ? ArrayPool<double>.Shared.Rent(n) : null;
        Span<double> gradient = rented is null ? stackalloc double[n] : rented.AsSpan(0, n);
        try
        {
            if (start == end || end - start == n)
            {
                gradient.Clear();
            }
            else
            {
                gradient.Fill(-1 / total);
                gradient[start..end].Fill(selected == 0 ? double.PositiveInfinity
                    : DistributionMath.ScaledRatio([rest], [selected, total]));
            }

            return PackedLogPdf.Record(value, weights, gradient, default, default);
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<double>.Shared.Return(rented);
            }
        }
    }

    private static (double Value, double Selected, double Complement) RecordValue(ReadOnlySpan<ScalarNode> weights, double total,
        int start, int end, LogProbability kind)
    {
        if (start == end)
        {
            return (double.NegativeInfinity, 0, total);
        }

        if (end - start == weights.Length)
        {
            return (0, total, 0);
        }

        double inside = 0, insideCompensation = 0, outside = 0, outsideCompensation = 0;
        double selected = 0, rest = 0;
        for (int i = start; i < end; i++)
        {
            int j = kind == LogProbability.UpperTail ? end - 1 - (i - start) : i;
            double weight = weights[j].Value;
            selected += weight;
            Accumulate(ref inside, ref insideCompensation, weight / total);
        }

        for (int i = 0; i < weights.Length; i++)
        {
            int j = kind == LogProbability.LowerTail ? weights.Length - 1 - i : i;
            if (j < start || j >= end)
            {
                double weight = weights[j].Value;
                rest += weight;
                Accumulate(ref outside, ref outsideCompensation, weight / total);
            }
        }

        double p = Math.Min(1, inside + insideCompensation), q = Math.Min(1, outside + outsideCompensation);
        return (p <= 0.5 ? Math.Log(p) : DistributionMath.Log1p(-q), selected, rest);
    }

    private static double ValidateWeights(ReadOnlySpan<ScalarNode> weights)
    {
        if (weights.IsEmpty)
        {
            throw new ArgumentException("At least one category is required.", nameof(weights));
        }

        double total = 0;
        foreach (ScalarNode node in weights)
        {
            double weight = node.Value;
            DistributionArgument.NonNegativeFinite(weight, nameof(weights), forDifferentiation: true);
            total += weight;
        }

        DistributionArgument.PositiveFinite(total, nameof(weights), forDifferentiation: true);
        return total;
    }
}
