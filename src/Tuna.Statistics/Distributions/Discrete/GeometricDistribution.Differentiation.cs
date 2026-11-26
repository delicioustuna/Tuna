using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public readonly partial struct GeometricDistribution
{
    /// <summary>
    /// Records the geometric log mass and its derivatives.
    /// <para>幾何分布の対数質量とその微分を記録します。</para>
    /// </summary>
    /// <param name="probability">The success probability. / 成功確率。</param>
    /// <param name="k">The integer observation. / 整数の観測値。</param>
    /// <returns>The log mass node. / 対数質量を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPmf(ScalarNode probability, long k) =>
        ScalarDistributionLogProbability.RecordMass(new GeometricDistribution(probability.Value, forDifferentiation: true), probability, default, k, LogProbability.Density, Gradient);

    /// <summary>
    /// Records the geometric log cumulative probability and its derivatives.
    /// <para>幾何分布の対数累積確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="probability">The success probability. / 成功確率。</param>
    /// <param name="k">The integer observation. / 整数の観測値。</param>
    /// <returns>The log cumulative probability node. / 対数累積確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogCdf(ScalarNode probability, long k) =>
        ScalarDistributionLogProbability.RecordMass(new GeometricDistribution(probability.Value, forDifferentiation: true), probability, default, k, LogProbability.LowerTail, Gradient);

    /// <summary>
    /// Records the geometric log survival probability and its derivatives.
    /// <para>幾何分布の対数生存確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="probability">The success probability. / 成功確率。</param>
    /// <param name="k">The integer observation. / 整数の観測値。</param>
    /// <returns>The log survival probability node. / 対数生存確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogSf(ScalarNode probability, long k) =>
        ScalarDistributionLogProbability.RecordMass(new GeometricDistribution(probability.Value, forDifferentiation: true), probability, default, k, LogProbability.UpperTail, Gradient);

    /// <summary>
    /// Records the geometric log likelihood of independent observations as a single operation.
    /// <para>幾何分布の対数尤度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="probability">The success probability. / 成功確率。</param>
    /// <param name="observations">The observations, summed in input order. / 入力順に集計する観測値。</param>
    /// <returns>The log likelihood node. / 対数尤度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPmf(ScalarNode probability, ReadOnlyVectorSpan<long> observations) =>
        ScalarDistributionLogProbability.RecordMassSum(new GeometricDistribution(probability.Value, forDifferentiation: true), probability, default, observations, Gradient);

    private static (double, double) Gradient(GeometricDistribution d, long k, double value, LogProbability kind)
    {
        if (k < 0)
        {
            return default;
        }

        double p = d._p, q = 1 - p;
        if (kind == LogProbability.Density)
        {
            return (k == 0 ? 1 / p : q == 0 ? double.NegativeInfinity
                : Math.FusedMultiplyAdd(-(k + 1.0), p, 1) / p / q, 0);
        }

        if (kind == LogProbability.UpperTail)
        {
            return (-(k + 1.0) / q, 0);
        }

        if (k == 0)
        {
            return (1 / p, 0);
        }

        if (q == 0)
        {
            return default;
        }

        double count = k + 1.0;
        double t = -count * d._logQ;
        if (t <= 0.5)
        {
            double factor = t < DistributionMath.MinNormal ? 1 : t / DistributionMath.Expm1(t);
            return (factor / -d._logQ / q, 0);
        }

        var exponent = XMath.Special.PreciseLogComplementPower(p, k);
        return (DistributionMath.ExponentialRatio(exponent, [count], [-DistributionMath.Expm1(-t)]), 0);
    }
}
