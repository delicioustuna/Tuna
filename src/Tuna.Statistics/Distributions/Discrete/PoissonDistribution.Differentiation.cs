using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public readonly partial struct PoissonDistribution
{
    /// <summary>
    /// Records the sum of Poisson log masses with shared or per-observation parameters as a single operation.
    /// <para>共通または観測ごとのパラメータを用いて、ポアソン分布の対数尤度の合計を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="rate">Rate. / 率。</param>
    /// <param name="observations">The integer observations. / 整数の観測値。</param>
    /// <returns>The log likelihood in observation order (zero for empty observations). / 入力順に集計した対数尤度。観測値が空の場合は 0。</returns>
    /// <exception cref="ArgumentException">A vector parameter has a different length from the observations. / パラメータベクトルの長さが観測数と一致しない場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is invalid, the tapes differ, or backpropagation has completed for a differentiable input. / ノードが無効、テープが異なる、または微分対象のノードのテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPmf(ScalarSetNode<double> rate,
        ReadOnlyVectorSpan<long> observations) =>
        PointwiseLogLikelihood.RecordSum(observations, EvaluateLikelihoodPoint, ValidateSharedLikelihoodParameters, rate);

    /// <summary>
    /// Records the Poisson log mass for each observation as a single operation.
    /// <para>ポアソン分布の観測ごとの対数確率質量を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="rate">Rate. / 率。</param>
    /// <param name="observations">The integer observations. / 整数の観測値。</param>
    /// <param name="tape">The tape, required when all parameters are scalar constants. / テープ。すべてのパラメータがスカラー定数の場合は必須です。</param>
    /// <returns>The log masses in observation order. / 入力順に並べた観測ごとの対数確率質量。</returns>
    /// <exception cref="ArgumentException">A vector parameter has a different length from the observations, or no tape is available. / パラメータベクトルの長さが観測数と一致しない場合、またはテープが未指定の場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is invalid, the tapes differ, or backpropagation has completed for a differentiable input. / ノードが無効、テープが異なる、または微分対象のノードのテープが逆伝播済みの場合。</exception>
    public static VectorNode<double> LogPmfPointwise(ScalarSetNode<double> rate,
        ReadOnlyVectorSpan<long> observations, Tape? tape = null) =>
        PointwiseLogLikelihood.RecordVector(observations, EvaluateLikelihoodPoint, ValidateSharedLikelihoodParameters, rate, tape: tape);

    private static void ValidateSharedLikelihoodParameters(double? rate, double? unusedSecond, double? unusedThird) =>
        _ = new PoissonDistribution(rate ?? 0, forDifferentiation: true);

    private static LikelihoodEvaluation EvaluateLikelihoodPoint(long observation, double rate, double _, double __, LikelihoodGradients gradients) =>
        PointwiseLogLikelihood.Mass(new PoissonDistribution(rate, forDifferentiation: true), observation, gradients, Gradient);

    /// <summary>
    /// Records the Poisson log mass and its derivatives.
    /// <para>ポアソン分布の対数質量とその微分を記録します。</para>
    /// </summary>
    /// <param name="rate">The rate in the distribution domain. / 分布の定義域内の率。</param>
    /// <param name="k">The integer observation. / 整数の観測値。</param>
    /// <returns>The log mass node. / 対数質量を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPmf(ScalarNode rate, long k) =>
        ScalarDistributionLogProbability.RecordMass(new PoissonDistribution(rate.Value, forDifferentiation: true), rate, default, k, LogProbability.Density, Gradient);

    /// <summary>
    /// Records the Poisson log cumulative probability and its derivatives.
    /// <para>ポアソン分布の対数累積確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="rate">The rate in the distribution domain. / 分布の定義域内の率。</param>
    /// <param name="k">The integer observation. / 整数の観測値。</param>
    /// <returns>The log cumulative probability node. / 対数累積確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogCdf(ScalarNode rate, long k) =>
        ScalarDistributionLogProbability.RecordMass(new PoissonDistribution(rate.Value, forDifferentiation: true), rate, default, k, LogProbability.LowerTail, Gradient);

    /// <summary>
    /// Records the Poisson log survival probability and its derivatives.
    /// <para>ポアソン分布の対数生存確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="rate">The rate in the distribution domain. / 分布の定義域内の率。</param>
    /// <param name="k">The integer observation. / 整数の観測値。</param>
    /// <returns>The log survival probability node. / 対数生存確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogSf(ScalarNode rate, long k) =>
        ScalarDistributionLogProbability.RecordMass(new PoissonDistribution(rate.Value, forDifferentiation: true), rate, default, k, LogProbability.UpperTail, Gradient);

    /// <summary>
    /// Records the Poisson log likelihood of independent observations as a single operation.
    /// <para>ポアソン分布の対数尤度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="rate">The rate in the distribution domain. / 分布の定義域内の率。</param>
    /// <param name="observations">The observations, summed in input order. / 入力順に集計する観測値。</param>
    /// <returns>The log likelihood node. / 対数尤度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPmf(ScalarNode rate, ReadOnlyVectorSpan<long> observations) =>
        ScalarDistributionLogProbability.RecordMassSum(new PoissonDistribution(rate.Value, forDifferentiation: true), rate, default, observations, Gradient);

    private static (double, double) Gradient(PoissonDistribution d, long k, double value, LogProbability kind)
    {
        if (k < 0)
        {
            return default;
        }

        double rate = d._rate;
        if (rate == 0)
        {
            return kind switch
            {
                LogProbability.Density => (k == 0 ? -1 : double.PositiveInfinity, 0),
                LogProbability.LowerTail => (k == 0 ? -1 : 0, 0),
                _ => (double.PositiveInfinity, 0),
            };
        }

        if (kind == LogProbability.Density)
        {
            // k が 2⁵³ を超えても、λ との差を求めるまで整数の下位ビットを保つ。
            double difference = Math.FusedMultiplyAdd(32768, k >> 15, -rate) + (k & 32767);
            return (difference / rate, 0);
        }

        bool upperGamma = kind == LogProbability.LowerTail;
        var (mantissa, exponent) = XMath.Special.ScaledGammaTailPrefixRatio(k + 1.0, rate, value, upperGamma);
        return ((upperGamma ? -1 : 1) * DistributionMath.ScaledRatio([], [rate], mantissa, exponent), 0);
    }
}
