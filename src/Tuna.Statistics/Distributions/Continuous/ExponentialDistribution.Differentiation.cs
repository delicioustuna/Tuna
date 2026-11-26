using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public readonly partial struct ExponentialDistribution
{
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
            double factor = t < DistributionMath.MinNormal ? 1 : t / DistributionMath.Expm1(t);
            return (factor / rate, 0, factor / x);
        }

        var exponent = (-t, -Math.FusedMultiplyAdd(rate, x, -t));
        double denominator = -DistributionMath.Expm1(-t);
        return (DistributionMath.ExponentialRatio(exponent, [x], [denominator]), 0,
            DistributionMath.ExponentialRatio(exponent, [rate], [denominator]));
    }
}
