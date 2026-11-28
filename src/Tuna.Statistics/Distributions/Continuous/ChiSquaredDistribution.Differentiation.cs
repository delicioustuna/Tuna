using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public readonly partial struct ChiSquaredDistribution
{
    /// <summary>
    /// Records the sum of chi-squared log densities with shared or per-observation parameters as a single operation.
    /// <para>共通または観測ごとのパラメータを用いて、カイ二乗分布の対数尤度の合計を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="degreesOfFreedom">Degrees of freedom. / 自由度。</param>
    /// <param name="observations">The observations. / 観測値のベクトル。</param>
    /// <returns>The log likelihood in observation order (zero for empty observations). / 入力順に集計した対数尤度。観測値が空の場合は 0。</returns>
    /// <exception cref="ArgumentException">A vector parameter has a different length from the observations. / パラメータベクトルの長さが観測数と一致しない場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is invalid, the tapes differ, or backpropagation has completed for a differentiable input. / ノードが無効、テープが異なる、または微分対象のノードのテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarSetNode<double> degreesOfFreedom,
        ReadOnlyVectorSpan<double> observations) =>
        PointwiseLogLikelihood.RecordSum(observations, EvaluateLikelihoodPoint, ValidateSharedLikelihoodParameters, degreesOfFreedom);

    /// <summary>
    /// Records the chi-squared log density for each observation as a single operation.
    /// <para>カイ二乗分布の観測ごとの対数密度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="degreesOfFreedom">Degrees of freedom. / 自由度。</param>
    /// <param name="observations">The observations. / 観測値のベクトル。</param>
    /// <param name="tape">The tape, required when all parameters are scalar constants. / テープ。すべてのパラメータがスカラー定数の場合は必須です。</param>
    /// <returns>The log densities in observation order. / 入力順に並べた観測ごとの対数密度。</returns>
    /// <exception cref="ArgumentException">A vector parameter has a different length from the observations, or no tape is available. / パラメータベクトルの長さが観測数と一致しない場合、またはテープが未指定の場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is invalid, the tapes differ, or backpropagation has completed for a differentiable input. / ノードが無効、テープが異なる、または微分対象のノードのテープが逆伝播済みの場合。</exception>
    public static VectorNode<double> LogPdfPointwise(ScalarSetNode<double> degreesOfFreedom,
        ReadOnlyVectorSpan<double> observations, Tape? tape = null) =>
        PointwiseLogLikelihood.RecordVector(observations, EvaluateLikelihoodPoint, ValidateSharedLikelihoodParameters, degreesOfFreedom, tape: tape);

    private static void ValidateSharedLikelihoodParameters(double? degreesOfFreedom, double? unusedSecond, double? unusedThird) =>
        _ = new ChiSquaredDistribution(degreesOfFreedom ?? 2, forDifferentiation: true);

    private static LikelihoodEvaluation EvaluateLikelihoodPoint(double observation, double degreesOfFreedom, double _, double __, LikelihoodGradients gradients) =>
        GammaLogProbability.Density(new ChiSquaredDistribution(degreesOfFreedom, forDifferentiation: true)._gamma, degreesOfFreedom, 2,
            observation, gradients, inverse: false, halfShape: true);

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
