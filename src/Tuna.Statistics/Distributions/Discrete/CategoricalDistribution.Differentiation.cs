using System.Buffers;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public sealed partial class CategoricalDistribution
{
    /// <summary>
    /// Records the sum of categorical log masses with shared or per-observation weights as a single operation.
    /// <para>共通または観測ごとのカテゴリ重みを用いて、カテゴリカル分布の対数尤度の合計を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="weights">A shared weight vector or a matrix with one weight column per observation. / 共通の重みベクトル、または観測ごとの重みを列に並べた行列。</param>
    /// <param name="observations">The observed category indices. / 観測されたカテゴリのインデックス。</param>
    /// <returns>The log likelihood in observation order (zero for empty observations). / 入力順に集計した対数尤度。観測値が空の場合は 0。</returns>
    /// <exception cref="ArgumentException">The column count differs from the observation count, the shared weights are empty, or observations are supplied with no category rows. / 列数が観測数と一致しない場合、共通の重みが空の場合、または観測値があるのにカテゴリの行がない場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A weight is negative or non-finite, or the sum of the shared weights or a weight column is nonpositive or non-finite. / 負または非有限の重みがある場合、または共通の重みや各列の合計が正で有限でない場合。</exception>
    /// <exception cref="InvalidOperationException">The weight node is invalid or requires a gradient after backpropagation. / 重みのノードが無効、または微分対象のノードのテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPmf(VectorSetNode<double> weights, ReadOnlyVectorSpan<long> observations)
    {
        var bound = BindLikelihoodWeights(weights, observations.Length, out Tape tape);
        return tape.RecordPointwiseSum<double, CategoricalLikelihood>(observations.Length, [bound.Input], new(bound, observations));
    }

    /// <summary>
    /// Records one categorical log mass per observation with shared or per-observation weights as a single operation.
    /// <para>共通または観測ごとのカテゴリ重みを用いて、カテゴリカル分布の観測ごとの対数確率質量を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="weights">A shared weight vector or a matrix with one weight column per observation. / 共通の重みベクトル、または観測ごとの重みを列に並べた行列。</param>
    /// <param name="observations">The observed category indices. / 観測されたカテゴリのインデックス。</param>
    /// <returns>The log masses in observation order. / 入力順に並べた観測ごとの対数確率質量。</returns>
    /// <exception cref="ArgumentException">The column count differs from the observation count, the shared weights are empty, or observations are supplied with no category rows. / 列数が観測数と一致しない場合、共通の重みが空の場合、または観測値があるのにカテゴリの行がない場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A weight is negative or non-finite, or the sum of the shared weights or a weight column is nonpositive or non-finite. / 負または非有限の重みがある場合、または共通の重みや各列の合計が正で有限でない場合。</exception>
    /// <exception cref="InvalidOperationException">The weight node is invalid or requires a gradient after backpropagation. / 重みのノードが無効、または微分対象のノードのテープが逆伝播済みの場合。</exception>
    public static VectorNode<double> LogPmfPointwise(VectorSetNode<double> weights, ReadOnlyVectorSpan<long> observations)
    {
        var bound = BindLikelihoodWeights(weights, observations.Length, out Tape tape);
        return tape.RecordPointwiseVector<double, CategoricalLikelihood>(observations.Length, [bound.Input], new(bound, observations));
    }

    private static BoundVectorSet<double> BindLikelihoodWeights(VectorSetNode<double> weights, int length, out Tape tape)
    {
        Tape? owner = null;
        var bound = weights.Bind(length, ref owner, nameof(weights));
        if (!bound.IsShared && bound.VectorLength == 0 && length != 0)
        {
            throw new ArgumentException("At least one category is required when observations are supplied.", nameof(weights));
        }

        tape = owner!;
        return bound;
    }

    private readonly ref struct CategoricalLikelihood(BoundVectorSet<double> weights, ReadOnlyVectorSpan<long> observations) : IPointwiseOperation<double>
    {
        private readonly BoundVectorSet<double> weightVectors = weights;
        private readonly ReadOnlyVectorSpan<long> categories = observations;

        public double Forward(scoped PointwiseOutput<double> output)
        {
            Span<double> values = output.Values;
            Span<double> derivatives = output.Derivatives(0);
            int rows = weightVectors.VectorLength;
            double sharedTotal = 0;
            if (weightVectors.IsShared)
            {
                var shared = new WeightValues(weightVectors.SharedValue);
                sharedTotal = ValidateWeights(shared);
                if (output.Sum)
                {
                    return SumLogMasses(shared, categories, sharedTotal, derivatives);
                }
            }

            double sum = 0;
            for (int i = 0; i < categories.Length; i++)
            {
                var point = new WeightValues(weightVectors[i]);
                double total = weightVectors.IsShared ? sharedTotal : ValidateWeights(point);
                var (start, end) = CategoryRange(rows, categories[i], LogProbability.Density);
                var (value, selected, rest) = RecordValue(point, total, start, end, LogProbability.Density);
                if (output.Sum)
                {
                    sum += value;
                }
                else
                {
                    values[i] = value;
                }

                if (!derivatives.IsEmpty)
                {
                    FillWeightGradient(derivatives.Slice(i * rows, rows), start, end, selected, rest, total);
                }
            }

            return sum;
        }
    }

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
        var values = new WeightValues(weights);
        double total = ValidateWeights(values);
        int length = PackedLogPdf.RequiresGradient(weights) ? weights.Length : 0;
        double[]? rented = length > MultivariateMath.StackLimit ? ArrayPool<double>.Shared.Rent(length) : null;
        Span<double> derivatives = rented is null ? stackalloc double[length] : rented.AsSpan(0, length);
        try
        {
            double value = SumLogMasses(values, observations, total, derivatives);
            return derivatives.IsEmpty ? value : PackedLogPdf.Record(value, weights, derivatives, default, default);
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<double>.Shared.Return(rented);
            }
        }
    }

    private static double SumLogMasses(WeightValues weights, ReadOnlyVectorSpan<long> observations, double total, Span<double> derivatives)
    {
        derivatives.Clear();
        if (observations.IsEmpty)
        {
            return 0;
        }

        int n = weights.Length;
        double[]? rented = n > MultivariateMath.StackLimit ? ArrayPool<double>.Shared.Rent(n) : null;
        Span<double> logMass = rented is null ? stackalloc double[n] : rented.AsSpan(0, n);
        try
        {
            for (int j = 0; j < n; j++)
            {
                double p = weights[j] / total;
                logMass[j] = p <= 0.5 ? Math.Log(p) : RecordValue(weights, total, j, j + 1, LogProbability.Density).Value;
            }

            bool requiresGradient = !derivatives.IsEmpty;
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
                    if (requiresGradient)
                    {
                        derivatives[(int)k]++;
                        validCount++;
                    }
                }
            }

            for (int j = 0; j < derivatives.Length; j++)
            {
                double weight = weights[j], count = derivatives[j];
                if (count == 0)
                {
                    derivatives[j] = -(validCount / total);
                }
                else if (count == validCount && weight / total > 0.5)
                {
                    var (_, _, rest) = RecordValue(weights, total, j, j + 1, LogProbability.Density);
                    derivatives[j] = DistributionMath.ScaledRatio([count, rest], [weight, total]);
                }
                else
                {
                    derivatives[j] = Math.FusedMultiplyAdd(-validCount, weight / total, count) / weight;
                }
            }

            return value;
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
        var values = new WeightValues(weights);
        double total = ValidateWeights(values);
        int n = weights.Length;
        var (start, end) = CategoryRange(n, k, kind);
        var (value, selected, rest) = RecordValue(values, total, start, end, kind);
        if (!PackedLogPdf.RequiresGradient(weights))
        {
            return value;
        }

        double[]? rented = n > MultivariateMath.StackLimit ? ArrayPool<double>.Shared.Rent(n) : null;
        Span<double> gradient = rented is null ? stackalloc double[n] : rented.AsSpan(0, n);
        try
        {
            FillWeightGradient(gradient, start, end, selected, rest, total);
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

    private static (int Start, int End) CategoryRange(int count, long k, LogProbability kind)
    {
        int index = (int)Math.Clamp(k, 0, count);
        int end = (int)Math.Clamp(k < count ? k + 1 : count, 0, count);
        return kind switch
        {
            LogProbability.Density => (index, end),
            LogProbability.UpperTail => (end, count),
            _ => (0, end),
        };
    }

    private static void FillWeightGradient(Span<double> gradient, int start, int end, double selected, double rest, double total)
    {
        if (start == end || end - start == gradient.Length)
        {
            gradient.Clear();
            return;
        }

        gradient.Fill(-1 / total);
        gradient[start..end].Fill(selected == 0 ? double.PositiveInfinity : DistributionMath.ScaledRatio([rest], [selected, total]));
    }

    private static (double Value, double Selected, double Complement) RecordValue(WeightValues weights, double total,
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
            double weight = weights[j];
            selected += weight;
            Accumulate(ref inside, ref insideCompensation, weight / total);
        }

        for (int i = 0; i < weights.Length; i++)
        {
            int j = kind == LogProbability.LowerTail ? weights.Length - 1 - i : i;
            if (j < start || j >= end)
            {
                double weight = weights[j];
                rest += weight;
                Accumulate(ref outside, ref outsideCompensation, weight / total);
            }
        }

        double p = Math.Min(1, inside + insideCompensation), q = Math.Min(1, outside + outsideCompensation);
        return (p <= 0.5 ? Math.Log(p) : DistributionMath.Log1p(-q), selected, rest);
    }

    private static double ValidateWeights(WeightValues weights)
    {
        if (weights.Length == 0)
        {
            throw new ArgumentException("At least one category is required.", nameof(weights));
        }

        double total = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            double value = weights[i];
            DistributionArgument.NonNegativeFinite(value, nameof(weights), forDifferentiation: true);
            total += value;
        }

        DistributionArgument.PositiveFinite(total, nameof(weights), forDifferentiation: true);
        return total;
    }

    private readonly ref struct WeightValues
    {
        private readonly ReadOnlySpan<ScalarNode> nodes;
        private readonly ReadOnlyVectorSpan<double> values;

        internal WeightValues(ReadOnlySpan<ScalarNode> nodes) => this.nodes = nodes;

        internal WeightValues(ReadOnlyVectorSpan<double> values) => this.values = values;

        internal int Length => nodes.IsEmpty ? values.Length : nodes.Length;

        internal double this[int index] => nodes.IsEmpty ? values[index] : nodes[index].Value;
    }
}
