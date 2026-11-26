using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public readonly partial struct StudentTDistribution
{
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
        double t = distribution.Standardize(observation);
        double value = distribution.LogPdfStandardized(t);
        bool requiresNu = degreesOfFreedom.RequiresGradient;
        bool requiresLocation = location.RequiresGradient || x.RequiresGradient;
        bool requiresScale = scale.RequiresGradient;
        if (!requiresNu && !requiresLocation && !requiresScale)
        {
            return value;
        }

        double dNu = 0, dLocation, dScale;
        if (distribution.IsNormal)
        {
            var normal = new NormalDistribution(distribution._location, distribution._scale);
            dLocation = requiresLocation ? normal.LogPdfMeanDerivative(observation, t) : 0;
            dScale = requiresScale ? normal.LogPdfScaleDerivative(observation, t) : 0;
        }
        else
        {
            var gradient = new StudentTLogPdfGradient(distribution._degreesOfFreedom, distribution._location, distribution._scale, requiresNu);
            var (locationDerivative, scaleDerivative) = gradient.LocationScale(observation, t, requiresLocation, requiresScale);
            (dLocation, dScale) = (locationDerivative.Value, scaleDerivative.Value);
            dNu = requiresNu ? gradient.DegreesOfFreedom(observation, t).Value : 0;
        }

        return ScalarNode.Record(value, [(degreesOfFreedom, dNu), (location, dLocation), (scale, dScale), (x, -dLocation)]);
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
