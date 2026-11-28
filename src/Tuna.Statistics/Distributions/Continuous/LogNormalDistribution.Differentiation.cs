using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public readonly partial struct LogNormalDistribution
{
    /// <summary>
    /// Records the sum of lognormal log densities with shared or per-observation parameters as a single operation.
    /// <para>共通または観測ごとのパラメータを用いて、対数正規分布の対数尤度の合計を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="mu">Mean in log coordinates. / 対数座標の平均。</param>
    /// <param name="sigma">Standard deviation in log coordinates. / 対数座標の標準偏差。</param>
    /// <param name="observations">The observations. / 観測値のベクトル。</param>
    /// <returns>The log likelihood in observation order (zero for empty observations). / 入力順に集計した対数尤度。観測値が空の場合は 0。</returns>
    /// <exception cref="ArgumentException">A vector parameter has a different length from the observations. / パラメータベクトルの長さが観測数と一致しない場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is invalid, the tapes differ, or backpropagation has completed for a differentiable input. / ノードが無効、テープが異なる、または微分対象のノードのテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarSetNode<double> mu, ScalarSetNode<double> sigma,
        ReadOnlyVectorSpan<double> observations) =>
        PointwiseLogLikelihood.RecordSum(observations, EvaluateLikelihoodPoint, ValidateSharedLikelihoodParameters, mu, sigma);

    /// <summary>
    /// Records the lognormal log density for each observation as a single operation.
    /// <para>対数正規分布の観測ごとの対数密度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="mu">Mean in log coordinates. / 対数座標の平均。</param>
    /// <param name="sigma">Standard deviation in log coordinates. / 対数座標の標準偏差。</param>
    /// <param name="observations">The observations. / 観測値のベクトル。</param>
    /// <param name="tape">The tape, required when all parameters are scalar constants. / テープ。すべてのパラメータがスカラー定数の場合は必須です。</param>
    /// <returns>The log densities in observation order. / 入力順に並べた観測ごとの対数密度。</returns>
    /// <exception cref="ArgumentException">A vector parameter has a different length from the observations, or no tape is available. / パラメータベクトルの長さが観測数と一致しない場合、またはテープが未指定の場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is invalid, the tapes differ, or backpropagation has completed for a differentiable input. / ノードが無効、テープが異なる、または微分対象のノードのテープが逆伝播済みの場合。</exception>
    public static VectorNode<double> LogPdfPointwise(ScalarSetNode<double> mu, ScalarSetNode<double> sigma,
        ReadOnlyVectorSpan<double> observations, Tape? tape = null) =>
        PointwiseLogLikelihood.RecordVector(observations, EvaluateLikelihoodPoint, ValidateSharedLikelihoodParameters, mu, sigma, tape: tape);

    private static void ValidateSharedLikelihoodParameters(double? mu, double? sigma, double? unusedThird) =>
        _ = new LogNormalDistribution(mu ?? 0, sigma ?? 1, forDifferentiation: true);

    private static LikelihoodEvaluation EvaluateLikelihoodPoint(double observation, double mu, double sigma, double _, LikelihoodGradients gradients) =>
        PointwiseLogLikelihood.Density(new LogNormalDistribution(mu, sigma, forDifferentiation: true), observation, gradients, Gradient);

    /// <summary>
    /// Records the lognormal log density and its derivatives.
    /// <para>対数正規分布の対数密度とその微分を記録します。</para>
    /// </summary>
    /// <param name="mu">The finite mean in log coordinates. / 対数座標での有限の平均。</param>
    /// <param name="sigma">The positive finite standard deviation in log coordinates. / 対数座標での正で有限の標準偏差。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log density node. / 対数密度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarNode mu, ScalarNode sigma, ScalarNode x) =>
        ScalarDistributionLogProbability.Record(new LogNormalDistribution(mu.Value, sigma.Value, forDifferentiation: true), mu, sigma, x, LogProbability.Density, Gradient);

    /// <summary>
    /// Records the lognormal log cumulative probability and its derivatives.
    /// <para>対数正規分布の対数累積確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="mu">The finite mean in log coordinates. / 対数座標での有限の平均。</param>
    /// <param name="sigma">The positive finite standard deviation in log coordinates. / 対数座標での正で有限の標準偏差。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log cumulative probability node. / 対数累積確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogCdf(ScalarNode mu, ScalarNode sigma, ScalarNode x) =>
        ScalarDistributionLogProbability.Record(new LogNormalDistribution(mu.Value, sigma.Value, forDifferentiation: true), mu, sigma, x, LogProbability.LowerTail, Gradient);

    /// <summary>
    /// Records the lognormal log survival probability and its derivatives.
    /// <para>対数正規分布の対数生存確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="mu">The finite mean in log coordinates. / 対数座標での有限の平均。</param>
    /// <param name="sigma">The positive finite standard deviation in log coordinates. / 対数座標での正で有限の標準偏差。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log survival probability node. / 対数生存確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogSf(ScalarNode mu, ScalarNode sigma, ScalarNode x) =>
        ScalarDistributionLogProbability.Record(new LogNormalDistribution(mu.Value, sigma.Value, forDifferentiation: true), mu, sigma, x, LogProbability.UpperTail, Gradient);

    /// <summary>
    /// Records the lognormal log likelihood of independent observations as a single operation.
    /// <para>対数正規分布の対数尤度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="mu">The finite mean in log coordinates. / 対数座標での有限の平均。</param>
    /// <param name="sigma">The positive finite standard deviation in log coordinates. / 対数座標での正で有限の標準偏差。</param>
    /// <param name="observations">The observations, summed in input order. / 入力順に集計する観測値。</param>
    /// <returns>The log likelihood node. / 対数尤度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarNode mu, ScalarNode sigma, ReadOnlyVectorSpan<double> observations) =>
        ScalarDistributionLogProbability.RecordSum(new LogNormalDistribution(mu.Value, sigma.Value, forDifferentiation: true), mu, sigma, observations, Gradient);

    private static (double, double, double) Gradient(LogNormalDistribution d, double x, double value, LogProbability kind)
    {
        if (double.IsNaN(x))
        {
            return (double.NaN, double.NaN, double.NaN);
        }

        if (x < 0)
        {
            return default;
        }

        double log = Math.Log(x);
        double z = (log - d._mu) / d._sigma;
        var normal = new NormalDistribution(d._mu, d._sigma);
        if (kind == LogProbability.Density)
        {
            double mean = normal.LogPdfMeanDerivative(log, z);
            double dx = double.IsPositiveInfinity(x) ? -0.0 : -(1 + mean) / x;
            return (mean, normal.LogPdfScaleDerivative(log, z), dx);
        }

        bool upper = kind == LogProbability.UpperTail;
        var (location, scale) = normal.LogTailDerivatives(log, z, upper, requiresScale: true);
        double sign = upper ? -1 : 1;
        double dX = double.IsPositiveInfinity(x) || (x == 0 && upper) ? 0 : sign * location / x;
        if (location == 0 && x > 0 && double.IsFinite(x) && Math.Abs(z) < 56)
        {
            double logSlope = -(0.5 * z) * z - XMath.Constants.LnSqrtTwoPi - value;
            dX = sign * Math.Exp(logSlope - Math.Log(d._sigma) - log);
        }

        return (-sign * location, -sign * scale, dX);
    }
}
