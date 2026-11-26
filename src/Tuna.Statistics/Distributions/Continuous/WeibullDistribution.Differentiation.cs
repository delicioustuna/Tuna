using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public readonly partial struct WeibullDistribution
{
    /// <summary>
    /// Records the Weibull log density and its derivatives.
    /// <para>ワイブル分布の対数密度とその微分を記録します。</para>
    /// </summary>
    /// <param name="shape">The positive finite shape. / 正で有限の形状。</param>
    /// <param name="scale">The positive finite scale. / 正で有限の尺度。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log density node. / 対数密度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarNode shape, ScalarNode scale, ScalarNode x) =>
        ScalarDistributionLogProbability.Record(new WeibullDistribution(shape.Value, scale.Value, forDifferentiation: true), shape, scale, x, LogProbability.Density, Gradient);

    /// <summary>
    /// Records the Weibull log cumulative probability and its derivatives.
    /// <para>ワイブル分布の対数累積確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="shape">The positive finite shape. / 正で有限の形状。</param>
    /// <param name="scale">The positive finite scale. / 正で有限の尺度。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log cumulative probability node. / 対数累積確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogCdf(ScalarNode shape, ScalarNode scale, ScalarNode x) =>
        ScalarDistributionLogProbability.Record(new WeibullDistribution(shape.Value, scale.Value, forDifferentiation: true), shape, scale, x, LogProbability.LowerTail, Gradient);

    /// <summary>
    /// Records the Weibull log survival probability and its derivatives.
    /// <para>ワイブル分布の対数生存確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="shape">The positive finite shape. / 正で有限の形状。</param>
    /// <param name="scale">The positive finite scale. / 正で有限の尺度。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log survival probability node. / 対数生存確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogSf(ScalarNode shape, ScalarNode scale, ScalarNode x) =>
        ScalarDistributionLogProbability.Record(new WeibullDistribution(shape.Value, scale.Value, forDifferentiation: true), shape, scale, x, LogProbability.UpperTail, Gradient);

    /// <summary>
    /// Records the Weibull log likelihood of independent observations as a single operation.
    /// <para>ワイブル分布の対数尤度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="shape">The positive finite shape. / 正で有限の形状。</param>
    /// <param name="scale">The positive finite scale. / 正で有限の尺度。</param>
    /// <param name="observations">The observations, summed in input order. / 入力順に集計する観測値。</param>
    /// <returns>The log likelihood node. / 対数尤度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarNode shape, ScalarNode scale, ReadOnlyVectorSpan<double> observations) =>
        ScalarDistributionLogProbability.RecordSum(new WeibullDistribution(shape.Value, scale.Value, forDifferentiation: true), shape, scale, observations, Gradient);

    private static (double, double, double) Gradient(WeibullDistribution d, double x, double value, LogProbability kind)
    {
        if (double.IsNaN(x))
        {
            return (double.NaN, double.NaN, double.NaN);
        }

        double a = d._shape, scale = d._scale;
        if (x < 0)
        {
            return default;
        }

        if (x == 0)
        {
            double edge = a == 1 ? -1 / scale : a < 1 ? double.NegativeInfinity : 0;
            return kind switch
            {
                LogProbability.Density => (double.NegativeInfinity, -a / scale,
                    a > 1 ? double.PositiveInfinity : edge),
                LogProbability.LowerTail => (double.NegativeInfinity, -a / scale, double.PositiveInfinity),
                _ => (0, 0, edge),
            };
        }

        if (double.IsPositiveInfinity(x))
        {
            double edge = a == 1 ? -1 / scale : a > 1 ? double.NegativeInfinity : 0;
            return kind == LogProbability.LowerTail ? default : (double.NegativeInfinity, double.PositiveInfinity, edge);
        }

        double logX = Math.Log(x), logScale = Math.Log(scale);
        double ratio = x / scale;
        double logRatio = Math.Abs(ratio - 1) < 0.5
            ? DistributionMath.Log1p((x - scale) / scale)
            : ratio >= DistributionMath.MinNormal && double.IsFinite(ratio) ? Math.Log(ratio) : logX - logScale;
        double logPower = a * logRatio;
        double power = Math.Exp(logPower);
        if (kind != LogProbability.Density)
        {
            bool upper = kind == LogProbability.UpperTail;
            if (!upper && power > 4096)
            {
                return default;
            }

            if (!upper && power <= 0.5)
            {
                double factor = power < DistributionMath.MinNormal ? 1 : power / DistributionMath.Expm1(power);
                return (logRatio * factor, -DistributionMath.ScaledRatio([a, factor], [scale]),
                    DistributionMath.ScaledRatio([a, factor], [x]));
            }

            var exponent = Math.Abs(logPower) is > 8 and < 8192 || !upper && power > 4
                ? XMath.Special.PreciseLogPowerRatio(x, scale, a) : (Hi: logPower, Lo: 0.0);
            if (upper)
            {
                double shape = logRatio == 0 ? 0
                    : -Math.CopySign(DistributionMath.ExponentialRatio(exponent, [Math.Abs(logRatio)], []), logRatio);
                return (shape, DistributionMath.ExponentialRatio(exponent, [a], [scale]),
                    -DistributionMath.ExponentialRatio(exponent, [a], [x]));
            }

            var precisePower = power > 4 ? XMath.Special.PreciseExp(exponent) : (Hi: power, Lo: 0.0);
            var decay = (-precisePower.Hi, -precisePower.Lo);
            double denominator = -DistributionMath.Expm1(-precisePower.Hi);
            return (DistributionMath.ExponentialRatio(decay, [logRatio, precisePower.Hi], [denominator]),
                -DistributionMath.ExponentialRatio(decay, [a, precisePower.Hi], [denominator, scale]),
                DistributionMath.ExponentialRatio(decay, [a, precisePower.Hi], [denominator, x]));
        }

        double delta = -DistributionMath.Expm1(logPower);
        double dDensityShape = 1 / a + delta * logRatio;
        double dScale = double.IsFinite(delta) ? -DistributionMath.ScaledRatio([a, delta], [scale])
            : Math.Exp(logPower + Math.Log(a) - logScale);
        double dX = a == 1 ? -1 / scale
            : power < 0.5 && a is >= 0.5 and <= 2 ? Math.FusedMultiplyAdd(-a, power, a - 1) / x
            : double.IsFinite(delta) ? Math.FusedMultiplyAdd(a, delta, -1) / x
            : -Math.Exp(logPower + Math.Log(a) - logX);
        return (dDensityShape, dScale, dX);
    }
}
