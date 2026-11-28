using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public readonly partial struct ExponentialDistribution
{
    /// <summary>
    /// Records the sum of exponential log densities with shared or per-observation parameters as a single operation.
    /// <para>共通または観測ごとのパラメータを用いて、指数分布の対数尤度の合計を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="rate">Rate. / 率。</param>
    /// <param name="observations">The observations. / 観測値のベクトル。</param>
    /// <returns>The log likelihood in observation order (zero for empty observations). / 入力順に集計した対数尤度。観測値が空の場合は 0。</returns>
    /// <exception cref="ArgumentException">A vector parameter has a different length from the observations. / パラメータベクトルの長さが観測数と一致しない場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is invalid, the tapes differ, or backpropagation has completed for a differentiable input. / ノードが無効、テープが異なる、または微分対象のノードのテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarSetNode<double> rate,
        ReadOnlyVectorSpan<double> observations) =>
        PointwiseLogLikelihood.RecordSum(observations, EvaluateLikelihoodPoint, ValidateSharedLikelihoodParameters, rate);

    /// <summary>
    /// Records the exponential log density for each observation as a single operation.
    /// <para>指数分布の観測ごとの対数密度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="rate">Rate. / 率。</param>
    /// <param name="observations">The observations. / 観測値のベクトル。</param>
    /// <param name="tape">The tape, required when all parameters are scalar constants. / テープ。すべてのパラメータがスカラー定数の場合は必須です。</param>
    /// <returns>The log densities in observation order. / 入力順に並べた観測ごとの対数密度。</returns>
    /// <exception cref="ArgumentException">A vector parameter has a different length from the observations, or no tape is available. / パラメータベクトルの長さが観測数と一致しない場合、またはテープが未指定の場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is invalid, the tapes differ, or backpropagation has completed for a differentiable input. / ノードが無効、テープが異なる、または微分対象のノードのテープが逆伝播済みの場合。</exception>
    public static VectorNode<double> LogPdfPointwise(ScalarSetNode<double> rate,
        ReadOnlyVectorSpan<double> observations, Tape? tape = null) =>
        PointwiseLogLikelihood.RecordVector(observations, EvaluateLikelihoodPoint, ValidateSharedLikelihoodParameters, rate, tape: tape);

    private static void ValidateSharedLikelihoodParameters(double? rate, double? unusedSecond, double? unusedThird) =>
        _ = new ExponentialDistribution(rate ?? 1, forDifferentiation: true);

    private static LikelihoodEvaluation EvaluateLikelihoodPoint(double observation, double rate, double _, double __, LikelihoodGradients gradients) =>
        PointwiseLogLikelihood.Density(new ExponentialDistribution(rate, forDifferentiation: true), observation, gradients, Gradient);

    /// <summary>
    /// Records the exponential log density and its derivatives.
    /// <para>指数分布の対数密度とその微分を記録します。</para>
    /// </summary>
    /// <param name="rate">The positive finite rate. / 正で有限の率。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log density node. / 対数密度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarNode rate, ScalarNode x) =>
        ScalarDistributionLogProbability.Record(new ExponentialDistribution(rate.Value, forDifferentiation: true), rate, default, x, LogProbability.Density, Gradient);

    /// <summary>
    /// Records the exponential log cumulative probability and its derivatives.
    /// <para>指数分布の対数累積確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="rate">The positive finite rate. / 正で有限の率。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log cumulative probability node. / 対数累積確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogCdf(ScalarNode rate, ScalarNode x) =>
        ScalarDistributionLogProbability.Record(new ExponentialDistribution(rate.Value, forDifferentiation: true), rate, default, x, LogProbability.LowerTail, Gradient);

    /// <summary>
    /// Records the exponential log survival probability and its derivatives.
    /// <para>指数分布の対数生存確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="rate">The positive finite rate. / 正で有限の率。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log survival probability node. / 対数生存確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogSf(ScalarNode rate, ScalarNode x) =>
        ScalarDistributionLogProbability.Record(new ExponentialDistribution(rate.Value, forDifferentiation: true), rate, default, x, LogProbability.UpperTail, Gradient);

    /// <summary>
    /// Records the exponential log likelihood of independent observations as a single operation.
    /// <para>指数分布の対数尤度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="rate">The positive finite rate. / 正で有限の率。</param>
    /// <param name="observations">The observations, summed in input order. / 入力順に集計する観測値。</param>
    /// <returns>The log likelihood node. / 対数尤度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarNode rate, ReadOnlyVectorSpan<double> observations) =>
        ScalarDistributionLogProbability.RecordSum(new ExponentialDistribution(rate.Value, forDifferentiation: true), rate, default, observations, Gradient);

    private static (double, double, double) Gradient(ExponentialDistribution d, double x, double value, LogProbability kind)
    {
        if (double.IsNaN(x))
        {
            return (double.NaN, 0, double.NaN);
        }

        double rate = d._rate;
        if (x < 0)
        {
            return default;
        }

        if (kind != LogProbability.LowerTail)
        {
            double numerator = kind == LogProbability.Density ? Math.FusedMultiplyAdd(-rate, x, 1) : double.NegativeInfinity;
            return (double.IsFinite(numerator) ? numerator / rate : -x, 0, -rate);
        }

        double t = rate * x;
        if (double.IsPositiveInfinity(t))
        {
            return default;
        }

        if (t <= 0.5)
        {
            double factor = t < NumericConstants.MinNormal ? 1 : t / DistributionMath.Expm1(t);
            return (factor / rate, 0, factor / x);
        }

        var exponent = (-t, -Math.FusedMultiplyAdd(rate, x, -t));
        double denominator = -DistributionMath.Expm1(-t);
        return (DistributionMath.ExponentialRatio(exponent, [x], [denominator]), 0,
            DistributionMath.ExponentialRatio(exponent, [rate], [denominator]));
    }
}
