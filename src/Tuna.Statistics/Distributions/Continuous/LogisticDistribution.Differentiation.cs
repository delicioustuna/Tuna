using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public readonly partial struct LogisticDistribution
{
    /// <summary>
    /// Records the sum of logistic log densities with shared or per-observation parameters as a single operation.
    /// <para>共通または観測ごとのパラメータを用いて、ロジスティック分布の対数尤度の合計を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="location">Location. / 位置パラメータ。</param>
    /// <param name="scale">Scale. / 尺度パラメータ。</param>
    /// <param name="observations">The observations. / 観測値のベクトル。</param>
    /// <returns>The log likelihood in observation order (zero for empty observations). / 入力順に集計した対数尤度。観測値が空の場合は 0。</returns>
    /// <exception cref="ArgumentException">A vector parameter has a different length from the observations. / パラメータベクトルの長さが観測数と一致しない場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is invalid, the tapes differ, or backpropagation has completed for a differentiable input. / ノードが無効、テープが異なる、または微分対象のノードのテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarSetNode<double> location, ScalarSetNode<double> scale,
        ReadOnlyVectorSpan<double> observations) =>
        PointwiseLogLikelihood.RecordSum(observations, EvaluateLikelihoodPoint, ValidateSharedLikelihoodParameters, location, scale);

    /// <summary>
    /// Records the logistic log density for each observation as a single operation.
    /// <para>ロジスティック分布の観測ごとの対数密度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="location">Location. / 位置パラメータ。</param>
    /// <param name="scale">Scale. / 尺度パラメータ。</param>
    /// <param name="observations">The observations. / 観測値のベクトル。</param>
    /// <param name="tape">The tape, required when all parameters are scalar constants. / テープ。すべてのパラメータがスカラー定数の場合は必須です。</param>
    /// <returns>The log densities in observation order. / 入力順に並べた観測ごとの対数密度。</returns>
    /// <exception cref="ArgumentException">A vector parameter has a different length from the observations, or no tape is available. / パラメータベクトルの長さが観測数と一致しない場合、またはテープが未指定の場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is invalid, the tapes differ, or backpropagation has completed for a differentiable input. / ノードが無効、テープが異なる、または微分対象のノードのテープが逆伝播済みの場合。</exception>
    public static VectorNode<double> LogPdfPointwise(ScalarSetNode<double> location, ScalarSetNode<double> scale,
        ReadOnlyVectorSpan<double> observations, Tape? tape = null) =>
        PointwiseLogLikelihood.RecordVector(observations, EvaluateLikelihoodPoint, ValidateSharedLikelihoodParameters, location, scale, tape: tape);

    private static void ValidateSharedLikelihoodParameters(double? location, double? scale, double? unusedThird) =>
        _ = new LogisticDistribution(location ?? 0, scale ?? 1, forDifferentiation: true);

    private static LikelihoodEvaluation EvaluateLikelihoodPoint(double observation, double location, double scale, double _, LikelihoodGradients gradients) =>
        PointwiseLogLikelihood.Density(new LogisticDistribution(location, scale, forDifferentiation: true), observation, gradients, Gradient);

    /// <summary>
    /// Records the logistic log density and its derivatives.
    /// <para>ロジスティック分布の対数密度とその微分を記録します。</para>
    /// </summary>
    /// <param name="location">The finite location. / 有限の位置。</param>
    /// <param name="scale">The positive finite scale. / 正で有限の尺度。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log density node. / 対数密度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarNode location, ScalarNode scale, ScalarNode x) =>
        ScalarDistributionLogProbability.Record(new LogisticDistribution(location.Value, scale.Value, forDifferentiation: true), location, scale, x, LogProbability.Density, Gradient);

    /// <summary>
    /// Records the logistic log cumulative probability and its derivatives.
    /// <para>ロジスティック分布の対数累積確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="location">The finite location. / 有限の位置。</param>
    /// <param name="scale">The positive finite scale. / 正で有限の尺度。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log cumulative probability node. / 対数累積確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogCdf(ScalarNode location, ScalarNode scale, ScalarNode x) =>
        ScalarDistributionLogProbability.Record(new LogisticDistribution(location.Value, scale.Value, forDifferentiation: true), location, scale, x, LogProbability.LowerTail, Gradient);

    /// <summary>
    /// Records the logistic log survival probability and its derivatives.
    /// <para>ロジスティック分布の対数生存確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="location">The finite location. / 有限の位置。</param>
    /// <param name="scale">The positive finite scale. / 正で有限の尺度。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log survival probability node. / 対数生存確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogSf(ScalarNode location, ScalarNode scale, ScalarNode x) =>
        ScalarDistributionLogProbability.Record(new LogisticDistribution(location.Value, scale.Value, forDifferentiation: true), location, scale, x, LogProbability.UpperTail, Gradient);

    /// <summary>
    /// Records the logistic log likelihood of independent observations as a single operation.
    /// <para>ロジスティック分布の対数尤度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="location">The finite location. / 有限の位置。</param>
    /// <param name="scale">The positive finite scale. / 正で有限の尺度。</param>
    /// <param name="observations">The observations, summed in input order. / 入力順に集計する観測値。</param>
    /// <returns>The log likelihood node. / 対数尤度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarNode location, ScalarNode scale, ReadOnlyVectorSpan<double> observations) =>
        ScalarDistributionLogProbability.RecordSum(new LogisticDistribution(location.Value, scale.Value, forDifferentiation: true), location, scale, observations, Gradient);

    private static (double, double, double) Gradient(LogisticDistribution d, double x, double value, LogProbability kind)
    {
        double z = d.Standardize(x);
        if (kind == LogProbability.Density)
        {
            return ScalarDistributionLogProbability.LocationScaleDensity(z, d._scale, -Math.Tanh(0.5 * z));
        }

        bool upper = kind == LogProbability.UpperTail;
        double argument = upper ? -z : z;
        double correction = argument > 0 ? DistributionMath.StandardizeCorrection(x, d._location, d._scale, z) : 0;
        var exponent = argument >= 0 ? (-argument, upper ? correction : -correction) : (0, 0);
        return ScalarDistributionLogProbability.LocationScaleTail(z, d._scale, exponent, upper,
            denominator: 1 + Math.Exp(-Math.Abs(argument)));
    }
}
