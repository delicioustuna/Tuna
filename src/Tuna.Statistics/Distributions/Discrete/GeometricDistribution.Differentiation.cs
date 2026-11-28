using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public readonly partial struct GeometricDistribution
{
    /// <summary>
    /// Records the sum of geometric log masses with shared or per-observation parameters as a single operation.
    /// <para>共通または観測ごとのパラメータを用いて、幾何分布の対数尤度の合計を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="probability">Success probability. / 成功確率。</param>
    /// <param name="observations">The integer observations. / 整数の観測値。</param>
    /// <returns>The log likelihood in observation order (zero for empty observations). / 入力順に集計した対数尤度。観測値が空の場合は 0。</returns>
    /// <exception cref="ArgumentException">A vector parameter has a different length from the observations. / パラメータベクトルの長さが観測数と一致しない場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is invalid, the tapes differ, or backpropagation has completed for a differentiable input. / ノードが無効、テープが異なる、または微分対象のノードのテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPmf(ScalarSetNode<double> probability,
        ReadOnlyVectorSpan<long> observations) =>
        PointwiseLogLikelihood.RecordSum(observations, EvaluateLikelihoodPoint, ValidateSharedLikelihoodParameters, probability);

    /// <summary>
    /// Records the geometric log mass for each observation as a single operation.
    /// <para>幾何分布の観測ごとの対数確率質量を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="probability">Success probability. / 成功確率。</param>
    /// <param name="observations">The integer observations. / 整数の観測値。</param>
    /// <param name="tape">The tape, required when all parameters are scalar constants. / テープ。すべてのパラメータがスカラー定数の場合は必須です。</param>
    /// <returns>The log masses in observation order. / 入力順に並べた観測ごとの対数確率質量。</returns>
    /// <exception cref="ArgumentException">A vector parameter has a different length from the observations, or no tape is available. / パラメータベクトルの長さが観測数と一致しない場合、またはテープが未指定の場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is invalid, the tapes differ, or backpropagation has completed for a differentiable input. / ノードが無効、テープが異なる、または微分対象のノードのテープが逆伝播済みの場合。</exception>
    public static VectorNode<double> LogPmfPointwise(ScalarSetNode<double> probability,
        ReadOnlyVectorSpan<long> observations, Tape? tape = null) =>
        PointwiseLogLikelihood.RecordVector(observations, EvaluateLikelihoodPoint, ValidateSharedLikelihoodParameters, probability, tape: tape);

    private static void ValidateSharedLikelihoodParameters(double? probability, double? unusedSecond, double? unusedThird) =>
        _ = new GeometricDistribution(probability ?? 1, forDifferentiation: true);

    private static LikelihoodEvaluation EvaluateLikelihoodPoint(long observation, double probability, double _, double __, LikelihoodGradients gradients) =>
        PointwiseLogLikelihood.Mass(new GeometricDistribution(probability, forDifferentiation: true), observation, gradients, Gradient);

    /// <summary>
    /// Records the geometric log mass and its derivatives.
    /// <para>幾何分布の対数質量とその微分を記録します。</para>
    /// </summary>
    /// <param name="probability">The success probability. / 成功確率。</param>
    /// <param name="k">The integer observation. / 整数の観測値。</param>
    /// <returns>The log mass node. / 対数質量を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPmf(ScalarNode probability, long k) =>
        ScalarDistributionLogProbability.RecordMass(new GeometricDistribution(probability.Value, forDifferentiation: true), probability, default, k, LogProbability.Density, Gradient);

    /// <summary>
    /// Records the geometric log cumulative probability and its derivatives.
    /// <para>幾何分布の対数累積確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="probability">The success probability. / 成功確率。</param>
    /// <param name="k">The integer observation. / 整数の観測値。</param>
    /// <returns>The log cumulative probability node. / 対数累積確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogCdf(ScalarNode probability, long k) =>
        ScalarDistributionLogProbability.RecordMass(new GeometricDistribution(probability.Value, forDifferentiation: true), probability, default, k, LogProbability.LowerTail, Gradient);

    /// <summary>
    /// Records the geometric log survival probability and its derivatives.
    /// <para>幾何分布の対数生存確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="probability">The success probability. / 成功確率。</param>
    /// <param name="k">The integer observation. / 整数の観測値。</param>
    /// <returns>The log survival probability node. / 対数生存確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogSf(ScalarNode probability, long k) =>
        ScalarDistributionLogProbability.RecordMass(new GeometricDistribution(probability.Value, forDifferentiation: true), probability, default, k, LogProbability.UpperTail, Gradient);

    /// <summary>
    /// Records the geometric log likelihood of independent observations as a single operation.
    /// <para>幾何分布の対数尤度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="probability">The success probability. / 成功確率。</param>
    /// <param name="observations">The observations, summed in input order. / 入力順に集計する観測値。</param>
    /// <returns>The log likelihood node. / 対数尤度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPmf(ScalarNode probability, ReadOnlyVectorSpan<long> observations) =>
        ScalarDistributionLogProbability.RecordMassSum(new GeometricDistribution(probability.Value, forDifferentiation: true), probability, default, observations, Gradient);

    private static (double, double) Gradient(GeometricDistribution d, long k, double value, LogProbability kind)
    {
        if (k < 0)
        {
            return default;
        }

        double p = d._p, q = 1 - p;
        if (kind == LogProbability.Density)
        {
            return (k == 0 ? 1 / p : q == 0 ? double.NegativeInfinity
                : Math.FusedMultiplyAdd(-(k + 1.0), p, 1) / p / q, 0);
        }

        if (kind == LogProbability.UpperTail)
        {
            return (-(k + 1.0) / q, 0);
        }

        if (k == 0)
        {
            return (1 / p, 0);
        }

        if (q == 0)
        {
            return default;
        }

        double count = k + 1.0;
        double t = -count * d._logQ;
        if (t <= 0.5)
        {
            double factor = t < NumericConstants.MinNormal ? 1 : t / DistributionMath.Expm1(t);
            return (factor / -d._logQ / q, 0);
        }

        var exponent = XMath.Special.PreciseLogComplementPower(p, k);
        return (DistributionMath.ExponentialRatio(exponent, [count], [-DistributionMath.Expm1(-t)]), 0);
    }
}
