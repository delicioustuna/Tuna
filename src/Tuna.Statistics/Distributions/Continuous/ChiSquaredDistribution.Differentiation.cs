using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public readonly partial struct ChiSquaredDistribution
{
    /// <summary>
    /// Records the chi-squared log density and its derivatives.
    /// <para>カイ二乗分布の対数密度とその微分を記録します。</para>
    /// </summary>
    /// <param name="degreesOfFreedom">The positive finite degrees of freedom whose half is nonzero. / 半分が 0 に丸まらない、正で有限の自由度。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log density node. / 対数密度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">The degrees of freedom are outside their domain. / 自由度が範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarNode degreesOfFreedom, ScalarNode x) =>
        GammaLogProbability.Record(new ChiSquaredDistribution(degreesOfFreedom.Value, forDifferentiation: true)._gamma, degreesOfFreedom, 2, x,
            LogProbability.Density, inverse: false, halfShape: true);

    /// <summary>
    /// Records the chi-squared log cumulative probability and its derivatives.
    /// <para>カイ二乗分布の対数累積確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="degreesOfFreedom">The positive finite degrees of freedom whose half is nonzero. / 半分が 0 に丸まらない、正で有限の自由度。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log cumulative probability node. / 対数累積確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">The degrees of freedom are outside their domain. / 自由度が範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogCdf(ScalarNode degreesOfFreedom, ScalarNode x) =>
        GammaLogProbability.Record(new ChiSquaredDistribution(degreesOfFreedom.Value, forDifferentiation: true)._gamma, degreesOfFreedom, 2, x,
            LogProbability.LowerTail, inverse: false, halfShape: true);

    /// <summary>
    /// Records the chi-squared log survival probability and its derivatives.
    /// <para>カイ二乗分布の対数生存確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="degreesOfFreedom">The positive finite degrees of freedom whose half is nonzero. / 半分が 0 に丸まらない、正で有限の自由度。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log survival probability node. / 対数生存確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">The degrees of freedom are outside their domain. / 自由度が範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogSf(ScalarNode degreesOfFreedom, ScalarNode x) =>
        GammaLogProbability.Record(new ChiSquaredDistribution(degreesOfFreedom.Value, forDifferentiation: true)._gamma, degreesOfFreedom, 2, x,
            LogProbability.UpperTail, inverse: false, halfShape: true);

    /// <summary>
    /// Records the chi-squared log likelihood of independent observations as a single operation.
    /// <para>カイ二乗分布に従う独立な観測値の対数尤度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="degreesOfFreedom">The positive finite degrees of freedom whose half is nonzero. / 半分が 0 に丸まらない、正で有限の自由度。</param>
    /// <param name="observations">The observations, summed in input order. / 入力順に集計する観測値。</param>
    /// <returns>The log likelihood node. / 対数尤度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">The degrees of freedom are outside their domain. / 自由度が範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarNode degreesOfFreedom, ReadOnlyVectorSpan<double> observations) =>
        GammaLogProbability.RecordSum(new ChiSquaredDistribution(degreesOfFreedom.Value, forDifferentiation: true)._gamma, degreesOfFreedom, 2,
            observations, inverse: false, halfShape: true);
}
