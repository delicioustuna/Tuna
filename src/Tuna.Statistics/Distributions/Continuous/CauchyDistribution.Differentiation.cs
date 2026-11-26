using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public readonly partial struct CauchyDistribution
{
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
        double t = distribution.Standardize(observation);
        double value = LogPdfValue(t, -DistributionMath.LnPi - Math.Log(distribution._scale));
        bool requiresLocation = location.RequiresGradient || x.RequiresGradient;
        bool requiresScale = scale.RequiresGradient;
        if (!requiresLocation && !requiresScale)
        {
            return value;
        }

        var gradient = new StudentTLogPdfGradient(1, distribution._location, distribution._scale);
        var (dLocation, dScale) = gradient.LocationScale(observation, t, requiresLocation, requiresScale);
        double locationDerivative = dLocation.Value;
        return ScalarNode.Record(value, [(location, locationDerivative), (scale, dScale.Value), (x, -locationDerivative)]);
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
        double value = 0, constant = -DistributionMath.LnPi - Math.Log(distribution._scale);
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
