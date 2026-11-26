using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public readonly partial struct BinomialDistribution
{
    /// <summary>
    /// Records the binomial log mass and its derivatives.
    /// <para>二項分布の対数質量とその微分を記録します。</para>
    /// </summary>
    /// <param name="trials">The integer trial count. / 整数の試行回数。</param>
    /// <param name="probability">The success probability. / 成功確率。</param>
    /// <param name="k">The integer observation. / 整数の観測値。</param>
    /// <returns>The log mass node. / 対数質量を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">The success probability is outside its distribution domain. / 成功確率が分布の定義域外の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">The trial count is outside its supported range. / 試行回数が対応範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPmf(long trials, ScalarNode probability, long k) =>
        ScalarDistributionLogProbability.RecordMass(new BinomialDistribution(trials, probability.Value, forDifferentiation: true), probability, default, k, LogProbability.Density, Gradient);

    /// <summary>
    /// Records the binomial log cumulative probability and its derivatives.
    /// <para>二項分布の対数累積確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="trials">The integer trial count. / 整数の試行回数。</param>
    /// <param name="probability">The success probability. / 成功確率。</param>
    /// <param name="k">The integer observation. / 整数の観測値。</param>
    /// <returns>The log cumulative probability node. / 対数累積確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">The success probability is outside its distribution domain. / 成功確率が分布の定義域外の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">The trial count is outside its supported range. / 試行回数が対応範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogCdf(long trials, ScalarNode probability, long k) =>
        ScalarDistributionLogProbability.RecordMass(new BinomialDistribution(trials, probability.Value, forDifferentiation: true), probability, default, k, LogProbability.LowerTail, Gradient);

    /// <summary>
    /// Records the binomial log survival probability and its derivatives.
    /// <para>二項分布の対数生存確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="trials">The integer trial count. / 整数の試行回数。</param>
    /// <param name="probability">The success probability. / 成功確率。</param>
    /// <param name="k">The integer observation. / 整数の観測値。</param>
    /// <returns>The log survival probability node. / 対数生存確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">The success probability is outside its distribution domain. / 成功確率が分布の定義域外の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">The trial count is outside its supported range. / 試行回数が対応範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogSf(long trials, ScalarNode probability, long k) =>
        ScalarDistributionLogProbability.RecordMass(new BinomialDistribution(trials, probability.Value, forDifferentiation: true), probability, default, k, LogProbability.UpperTail, Gradient);

    /// <summary>
    /// Records the binomial log likelihood of independent observations as a single operation.
    /// <para>二項分布の対数尤度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="trials">The integer trial count. / 整数の試行回数。</param>
    /// <param name="probability">The success probability. / 成功確率。</param>
    /// <param name="observations">The observations, summed in input order. / 入力順に集計する観測値。</param>
    /// <returns>The log likelihood node. / 対数尤度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">The success probability is outside its distribution domain. / 成功確率が分布の定義域外の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">The trial count is outside its supported range. / 試行回数が対応範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPmf(long trials, ScalarNode probability, ReadOnlyVectorSpan<long> observations) =>
        ScalarDistributionLogProbability.RecordMassSum(new BinomialDistribution(trials, probability.Value, forDifferentiation: true), probability, default, observations, Gradient);

    private static (double, double) Gradient(BinomialDistribution d, long k, double value, LogProbability kind)
    {
        long n = d._trials;
        if (k < 0 || k > n || n == 0 || (kind != LogProbability.Density && k == n))
        {
            return default;
        }

        double p = d._p, q = d._q;
        if (kind == LogProbability.Density)
        {
            double score = p == 0 ? k == 0 ? -n : double.PositiveInfinity
                : q == 0 ? k == n ? n : double.NegativeInfinity
                : Math.FusedMultiplyAdd(-n, p, k) / p / q;
            return (score, 0);
        }

        bool lower = kind == LogProbability.LowerTail;
        if (p == 0 || q == 0)
        {
            return (p == 0 ? lower ? k == 0 ? -n : 0 : double.PositiveInfinity
                : lower ? double.NegativeInfinity : k == n - 1 ? n : 0, 0);
        }

        double a = k + 1.0, b = n - k;
        var (mantissa, exponent) = p <= 0.5 ? XMath.Special.ScaledBetaTailPrefixRatio(a, b, p, 0, value, lower)
            : XMath.Special.ScaledBetaTailPrefixRatio(b, a, q, 0, value, !lower);
        return ((lower ? -1 : 1) * DistributionMath.ScaledRatio([], [p, q], mantissa, exponent), 0);
    }
}
