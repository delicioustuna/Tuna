using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public readonly partial struct StudentTDistribution
{
    /// <summary>
    /// Records the sum of Student's t log densities with shared or per-observation parameters as a single operation.
    /// <para>共通または観測ごとのパラメータを用いて、スチューデントの t 分布の対数尤度の合計を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="degreesOfFreedom">Degrees of freedom. / 自由度。</param>
    /// <param name="location">Location. / 位置パラメータ。</param>
    /// <param name="scale">Scale. / 尺度パラメータ。</param>
    /// <param name="observations">The observations. / 観測値のベクトル。</param>
    /// <returns>The log likelihood in observation order (zero for empty observations). / 入力順に集計した対数尤度。観測値が空の場合は 0。</returns>
    /// <exception cref="ArgumentException">A vector parameter has a different length from the observations. / パラメータベクトルの長さが観測数と一致しない場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is invalid, the tapes differ, or backpropagation has completed for a differentiable input. / ノードが無効、テープが異なる、または微分対象のノードのテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarSetNode<double> degreesOfFreedom, ScalarSetNode<double> location, ScalarSetNode<double> scale,
        ReadOnlyVectorSpan<double> observations) =>
        PointwiseLogLikelihood.RecordSum(observations, EvaluateLikelihoodPoint, ValidateSharedLikelihoodParameters, degreesOfFreedom, location, scale);

    /// <summary>
    /// Records the Student's t log density for each observation as a single operation.
    /// <para>スチューデントの t 分布の観測ごとの対数密度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="degreesOfFreedom">Degrees of freedom. / 自由度。</param>
    /// <param name="location">Location. / 位置パラメータ。</param>
    /// <param name="scale">Scale. / 尺度パラメータ。</param>
    /// <param name="observations">The observations. / 観測値のベクトル。</param>
    /// <param name="tape">The tape, required when all parameters are scalar constants. / テープ。すべてのパラメータがスカラー定数の場合は必須です。</param>
    /// <returns>The log densities in observation order. / 入力順に並べた観測ごとの対数密度。</returns>
    /// <exception cref="ArgumentException">A vector parameter has a different length from the observations, or no tape is available. / パラメータベクトルの長さが観測数と一致しない場合、またはテープが未指定の場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is invalid, the tapes differ, or backpropagation has completed for a differentiable input. / ノードが無効、テープが異なる、または微分対象のノードのテープが逆伝播済みの場合。</exception>
    public static VectorNode<double> LogPdfPointwise(ScalarSetNode<double> degreesOfFreedom, ScalarSetNode<double> location, ScalarSetNode<double> scale,
        ReadOnlyVectorSpan<double> observations, Tape? tape = null) =>
        PointwiseLogLikelihood.RecordVector(observations, EvaluateLikelihoodPoint, ValidateSharedLikelihoodParameters, degreesOfFreedom, location, scale, tape: tape);

    private static void ValidateSharedLikelihoodParameters(double? degreesOfFreedom, double? location, double? scale) =>
        _ = new StudentTDistribution(degreesOfFreedom ?? 1, location ?? 0, scale ?? 1, forDifferentiation: true);

    private static LikelihoodEvaluation EvaluateLikelihoodPoint(double observation, double degreesOfFreedom, double location, double scale,
        LikelihoodGradients gradients) =>
        new StudentTDistribution(degreesOfFreedom, location, scale, forDifferentiation: true).DensityPoint(observation,
            (gradients & LikelihoodGradients.First) != 0, (gradients & LikelihoodGradients.Second) != 0, (gradients & LikelihoodGradients.Third) != 0);

    /// <summary>
    /// Records the Student's t log density and its derivatives with respect to the degrees of freedom, location, scale, and variate.
    /// <para>スチューデントの t 分布の対数密度と、自由度・位置・尺度・評価点に関する微分を記録します。</para>
    /// </summary>
    /// <param name="degreesOfFreedom">The positive degrees of freedom <c>ν</c>. / 正の自由度 <c>ν</c>。</param>
    /// <param name="location">The finite location <c>μ</c>. / 有限の位置 <c>μ</c>。</param>
    /// <param name="scale">The positive finite scale <c>σ</c>. / 正で有限の尺度 <c>σ</c>。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log density node. / 対数密度を表すノード。</returns>
    /// <remarks>
    /// <para>Positive infinity for the degrees of freedom gives the normal limit with derivative zero in that argument.</para>
    /// <para>自由度が正の無限大の場合は正規分布の極限を使い、自由度の局所微分は 0 です。</para>
    /// </remarks>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">An input node is stale, the input nodes belong to different tapes, or the tape has already run backward. / 失効したノードがある場合、ノードのテープが異なる場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarNode degreesOfFreedom, ScalarNode location, ScalarNode scale, ScalarNode x)
    {
        var distribution = new StudentTDistribution(degreesOfFreedom.Value, location.Value, scale.Value, forDifferentiation: true);
        double observation = x.Value;
        bool requiresNu = degreesOfFreedom.RequiresGradient;
        bool requiresLocation = location.RequiresGradient || x.RequiresGradient;
        bool requiresScale = scale.RequiresGradient;
        var point = distribution.DensityPoint(observation, requiresNu, requiresLocation, requiresScale);
        double locationDerivative = point.Second.Value;
        return requiresNu || requiresLocation || requiresScale
            ? ScalarNode.Record(point.Value, [(degreesOfFreedom, point.First.Value), (location, locationDerivative), (scale, point.Third.Value), (x, -locationDerivative)])
            : point.Value;
    }

    private LikelihoodEvaluation DensityPoint(double observation, bool requiresNu, bool requiresLocation, bool requiresScale)
    {
        double t = Standardize(observation);
        double value = LogPdfStandardized(t);
        if (!requiresNu && !requiresLocation && !requiresScale)
        {
            return new(value);
        }

        if (IsNormal)
        {
            var normal = new NormalDistribution(_location, _scale);
            double locationDerivative = requiresLocation ? normal.LogPdfMeanDerivative(observation, t) : 0;
            double scaleDerivative = requiresScale ? normal.LogPdfScaleDerivative(observation, t) : 0;
            return new(value, default, new(locationDerivative), new(scaleDerivative));
        }

        var gradient = new StudentTLogPdfGradient(_degreesOfFreedom, _location, _scale, requiresNu);
        var (dLocation, dScale) = gradient.LocationScale(observation, t, requiresLocation, requiresScale);
        var dNu = requiresNu ? gradient.DegreesOfFreedom(observation, t) : default;
        return new(value, dNu, dLocation, dScale);
    }

    /// <summary>
    /// Records the Student's t log likelihood of independent observations as a single operation.
    /// <para>スチューデントの t 分布に従う独立な観測値の対数尤度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="degreesOfFreedom">The positive degrees of freedom <c>ν</c>. Positive infinity gives the normal limit. / 正の自由度 <c>ν</c>。正の無限大では正規分布の極限を使います。</param>
    /// <param name="location">The finite location <c>μ</c>. / 有限の位置 <c>μ</c>。</param>
    /// <param name="scale">The positive finite scale <c>σ</c>. / 正で有限の尺度 <c>σ</c>。</param>
    /// <param name="observations">The observations. / 観測値のベクトル。</param>
    /// <returns>The log likelihood node for the observations. / 観測全体の対数尤度を表すノード。</returns>
    /// <remarks>
    /// <para>Values are summed in input order. An empty vector gives zero for the value and gradient. The tape stores only the aggregated partial derivatives.</para>
    /// <para>対数密度を入力順に合計します。空のベクトルでは値と勾配がともに 0 です。テープには各パラメータの偏微分の合計だけを保存します。</para>
    /// </remarks>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">An input node is stale, the input nodes belong to different tapes, or the tape has already run backward. / 失効したノードがある場合、ノードのテープが異なる場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarNode degreesOfFreedom, ScalarNode location, ScalarNode scale, ReadOnlyVectorSpan<double> observations)
    {
        var distribution = new StudentTDistribution(degreesOfFreedom.Value, location.Value, scale.Value, forDifferentiation: true);
        bool requiresNu = degreesOfFreedom.RequiresGradient;
        bool requiresLocation = location.RequiresGradient;
        bool requiresScale = scale.RequiresGradient;
        double value, dNu = 0, dLocation, dScale;
        if (distribution.IsNormal)
        {
            var normal = new NormalDistribution(distribution._location, distribution._scale);
            (value, dLocation, dScale) = normal.LogPdfSum(observations, requiresLocation, requiresScale);
        }
        else
        {
            (value, dNu, dLocation, dScale) = distribution.LogPdfSum(observations, requiresNu, requiresLocation, requiresScale);
        }

        return requiresNu || requiresLocation || requiresScale
            ? ScalarNode.Record(value, [(degreesOfFreedom, dNu), (location, dLocation), (scale, dScale)])
            : value;
    }

    private (double Value, double Nu, double Location, double Scale) LogPdfSum(
        ReadOnlyVectorSpan<double> observations, bool requiresNu, bool requiresLocation, bool requiresScale)
    {
        if (observations.Length == 0)
        {
            return default;
        }

        var gradient = new StudentTLogPdfGradient(_degreesOfFreedom, _location, _scale, requiresNu);
        var nuSum = new ScaledDerivativeSum();
        var locationSum = new ScaledDerivativeSum();
        var scaleSum = new ScaledDerivativeSum();
        double value = 0, constant = LogPdfConstant(), logScale = Math.Log(_scale);
        for (int i = 0; i < observations.Length; i++)
        {
            double x = observations[i];
            double t = Standardize(x);
            value += LogPdfValue(t, constant, logScale);
            if (requiresLocation || requiresScale)
            {
                var (dLocation, dScale) = gradient.LocationScale(x, t, requiresLocation, requiresScale);
                locationSum.Add(dLocation);
                scaleSum.Add(dScale);
            }

            if (requiresNu)
            {
                nuSum.Add(gradient.DegreesOfFreedom(x, t));
            }
        }

        return (value, nuSum.Value, locationSum.Value, scaleSum.Value);
    }
}
