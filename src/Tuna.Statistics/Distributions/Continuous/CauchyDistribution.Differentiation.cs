using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public readonly partial struct CauchyDistribution
{
    /// <summary>
    /// Records the sum of Cauchy log densities with shared or per-observation parameters as a single operation.
    /// <para>共通または観測ごとのパラメータを用いて、コーシー分布の対数尤度の合計を 1 回の演算として記録します。</para>
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
    /// Records the Cauchy log density for each observation as a single operation.
    /// <para>コーシー分布の観測ごとの対数密度を 1 回の演算として記録します。</para>
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
        _ = new CauchyDistribution(location ?? 0, scale ?? 1, forDifferentiation: true);

    private static LikelihoodEvaluation EvaluateLikelihoodPoint(double observation, double location, double scale, double _, LikelihoodGradients gradients) =>
        new CauchyDistribution(location, scale, forDifferentiation: true).DensityPoint(observation,
            (gradients & LikelihoodGradients.First) != 0, (gradients & LikelihoodGradients.Second) != 0);

    /// <summary>
    /// Records the Cauchy log density and its derivatives with respect to the location, scale, and variate.
    /// <para>コーシー分布の対数密度と、位置・尺度・評価点に関する微分を記録します。</para>
    /// </summary>
    /// <param name="location">The finite location <c>x₀</c>. / 有限の位置 <c>x₀</c>。</param>
    /// <param name="scale">The positive finite scale <c>γ</c>. / 正で有限の尺度 <c>γ</c>。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log density node. / 対数密度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">An input node is stale, the input nodes belong to different tapes, or the tape has already run backward. / 失効したノードがある場合、ノードのテープが異なる場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarNode location, ScalarNode scale, ScalarNode x)
    {
        var distribution = new CauchyDistribution(location.Value, scale.Value, forDifferentiation: true);
        double observation = x.Value;
        bool requiresLocation = location.RequiresGradient || x.RequiresGradient;
        bool requiresScale = scale.RequiresGradient;
        var point = distribution.DensityPoint(observation, requiresLocation, requiresScale);
        double locationDerivative = point.First.Value;
        return requiresLocation || requiresScale
            ? ScalarNode.Record(point.Value, [(location, locationDerivative), (scale, point.Second.Value), (x, -locationDerivative)])
            : point.Value;
    }

    private LikelihoodEvaluation DensityPoint(double observation, bool requiresLocation, bool requiresScale)
    {
        double t = Standardize(observation);
        double value = LogPdfValue(t, -XMath.Constants.LnPi - Math.Log(_scale));
        if (!requiresLocation && !requiresScale)
        {
            return new(value);
        }

        var gradient = new StudentTLogPdfGradient(1, _location, _scale);
        var (dLocation, dScale) = gradient.LocationScale(observation, t, requiresLocation, requiresScale);
        return new(value, dLocation, dScale);
    }

    /// <summary>
    /// Records the Cauchy log likelihood of independent observations as a single operation.
    /// <para>コーシー分布に従う独立な観測値の対数尤度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="location">The finite location <c>x₀</c>. / 有限の位置 <c>x₀</c>。</param>
    /// <param name="scale">The positive finite scale <c>γ</c>. / 正で有限の尺度 <c>γ</c>。</param>
    /// <param name="observations">The observations. / 観測値のベクトル。</param>
    /// <returns>The log likelihood node for the observations. / 観測全体の対数尤度を表すノード。</returns>
    /// <remarks>
    /// <para>Values are summed in input order. An empty vector gives zero for the value and gradient. The tape stores only the aggregated partial derivatives.</para>
    /// <para>対数密度を入力順に合計します。空のベクトルでは値と勾配がともに 0 です。テープには各パラメータの偏微分の合計だけを保存します。</para>
    /// </remarks>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">An input node is stale, the input nodes belong to different tapes, or the tape has already run backward. / 失効したノードがある場合、ノードのテープが異なる場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarNode location, ScalarNode scale, ReadOnlyVectorSpan<double> observations)
    {
        var distribution = new CauchyDistribution(location.Value, scale.Value, forDifferentiation: true);
        var gradient = new StudentTLogPdfGradient(1, distribution._location, distribution._scale);
        bool requiresLocation = location.RequiresGradient;
        bool requiresScale = scale.RequiresGradient;
        var locationSum = new ScaledDerivativeSum();
        var scaleSum = new ScaledDerivativeSum();
        double value = 0, constant = -XMath.Constants.LnPi - Math.Log(distribution._scale);
        for (int i = 0; i < observations.Length; i++)
        {
            double x = observations[i];
            double t = distribution.Standardize(x);
            value += LogPdfValue(t, constant);
            if (requiresLocation || requiresScale)
            {
                var (dLocation, dScale) = gradient.LocationScale(x, t, requiresLocation, requiresScale);
                locationSum.Add(dLocation);
                scaleSum.Add(dScale);
            }
        }

        return requiresLocation || requiresScale
            ? ScalarNode.Record(value, [(location, locationSum.Value), (scale, scaleSum.Value)])
            : value;
    }
}
