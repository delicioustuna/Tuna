using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public readonly partial struct UniformDistribution
{
    /// <summary>
    /// Records the sum of uniform log densities with shared or per-observation parameters as a single operation.
    /// <para>共通または観測ごとのパラメータを用いて、一様分布の対数尤度の合計を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="lower">Lower bound. / 下端。</param>
    /// <param name="upper">Upper bound. / 上端。</param>
    /// <param name="observations">The observations. / 観測値のベクトル。</param>
    /// <returns>The log likelihood in observation order (zero for empty observations). / 入力順に集計した対数尤度。観測値が空の場合は 0。</returns>
    /// <exception cref="ArgumentException">A vector parameter has a different length from the observations. / パラメータベクトルの長さが観測数と一致しない場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is invalid, the tapes differ, or backpropagation has completed for a differentiable input. / ノードが無効、テープが異なる、または微分対象のノードのテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarSetNode<double> lower, ScalarSetNode<double> upper,
        ReadOnlyVectorSpan<double> observations) =>
        PointwiseLogLikelihood.RecordSum(observations, EvaluateLikelihoodPoint, ValidateSharedLikelihoodParameters, lower, upper);

    /// <summary>
    /// Records the uniform log density for each observation as a single operation.
    /// <para>一様分布の観測ごとの対数密度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="lower">Lower bound. / 下端。</param>
    /// <param name="upper">Upper bound. / 上端。</param>
    /// <param name="observations">The observations. / 観測値のベクトル。</param>
    /// <param name="tape">The tape, required when all parameters are scalar constants. / テープ。すべてのパラメータがスカラー定数の場合は必須です。</param>
    /// <returns>The log densities in observation order. / 入力順に並べた観測ごとの対数密度。</returns>
    /// <exception cref="ArgumentException">A vector parameter has a different length from the observations, or no tape is available. / パラメータベクトルの長さが観測数と一致しない場合、またはテープが未指定の場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is invalid, the tapes differ, or backpropagation has completed for a differentiable input. / ノードが無効、テープが異なる、または微分対象のノードのテープが逆伝播済みの場合。</exception>
    public static VectorNode<double> LogPdfPointwise(ScalarSetNode<double> lower, ScalarSetNode<double> upper,
        ReadOnlyVectorSpan<double> observations, Tape? tape = null) =>
        PointwiseLogLikelihood.RecordVector(observations, EvaluateLikelihoodPoint, ValidateSharedLikelihoodParameters, lower, upper, tape: tape);

    private static void ValidateSharedLikelihoodParameters(double? lower, double? upper, double? unusedThird)
    {
        if (lower.HasValue && upper.HasValue)
        {
            _ = new UniformDistribution(lower.Value, upper.Value, forDifferentiation: true);
            return;
        }

        DistributionArgument.Finite(lower.GetValueOrDefault(), nameof(lower), forDifferentiation: true);
        DistributionArgument.Finite(upper.GetValueOrDefault(), nameof(upper), forDifferentiation: true);
    }

    private static LikelihoodEvaluation EvaluateLikelihoodPoint(double observation, double lower, double upper, double _, LikelihoodGradients gradients) =>
        PointwiseLogLikelihood.Density(new UniformDistribution(lower, upper, forDifferentiation: true), observation, gradients, Gradient);

    /// <summary>
    /// Records the uniform log density and its derivatives.
    /// <para>一様分布の対数密度とその微分を記録します。</para>
    /// </summary>
    /// <param name="lower">The finite lower bound. / 有限の下端。</param>
    /// <param name="upper">The finite upper bound. / 有限の上端。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log density node. / 対数密度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarNode lower, ScalarNode upper, ScalarNode x) =>
        ScalarDistributionLogProbability.Record(new UniformDistribution(lower.Value, upper.Value, forDifferentiation: true), lower, upper, x, LogProbability.Density, Gradient);

    /// <summary>
    /// Records the uniform log cumulative probability and its derivatives.
    /// <para>一様分布の対数累積確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="lower">The finite lower bound. / 有限の下端。</param>
    /// <param name="upper">The finite upper bound. / 有限の上端。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log cumulative probability node. / 対数累積確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogCdf(ScalarNode lower, ScalarNode upper, ScalarNode x) =>
        ScalarDistributionLogProbability.Record(new UniformDistribution(lower.Value, upper.Value, forDifferentiation: true), lower, upper, x, LogProbability.LowerTail, Gradient);

    /// <summary>
    /// Records the uniform log survival probability and its derivatives.
    /// <para>一様分布の対数生存確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="lower">The finite lower bound. / 有限の下端。</param>
    /// <param name="upper">The finite upper bound. / 有限の上端。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log survival probability node. / 対数生存確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogSf(ScalarNode lower, ScalarNode upper, ScalarNode x) =>
        ScalarDistributionLogProbability.Record(new UniformDistribution(lower.Value, upper.Value, forDifferentiation: true), lower, upper, x, LogProbability.UpperTail, Gradient);

    /// <summary>
    /// Records the uniform log likelihood of independent observations as a single operation.
    /// <para>一様分布の対数尤度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="lower">The finite lower bound. / 有限の下端。</param>
    /// <param name="upper">The finite upper bound. / 有限の上端。</param>
    /// <param name="observations">The observations, summed in input order. / 入力順に集計する観測値。</param>
    /// <returns>The log likelihood node. / 対数尤度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarNode lower, ScalarNode upper, ReadOnlyVectorSpan<double> observations) =>
        ScalarDistributionLogProbability.RecordSum(new UniformDistribution(lower.Value, upper.Value, forDifferentiation: true), lower, upper, observations, Gradient);

    private static (double, double, double) Gradient(UniformDistribution d, double x, double value, LogProbability kind)
    {
        if (double.IsNaN(x))
        {
            return (double.NaN, double.NaN, double.NaN);
        }

        if (x < d._lower || x > d._upper)
        {
            return default;
        }

        double inverseWidth = 1 / d._width;
        if (kind == LogProbability.Density)
        {
            return (inverseWidth, -inverseWidth, 0);
        }

        double left = x - d._lower, right = d._upper - x;
        if (kind == LogProbability.LowerTail)
        {
            return (left == 0 ? double.NegativeInfinity : -DistributionMath.ScaledRatio([right], [left, d._width]),
                -inverseWidth, 1 / left);
        }

        return (inverseWidth, right == 0 ? double.PositiveInfinity : DistributionMath.ScaledRatio([left], [right, d._width]), -1 / right);
    }
}
