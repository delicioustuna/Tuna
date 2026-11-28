using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public readonly partial struct BernoulliDistribution
{
    /// <summary>
    /// Records the sum of Bernoulli log masses with shared or per-observation parameters as a single operation.
    /// <para>共通または観測ごとのパラメータを用いて、ベルヌーイ分布の対数尤度の合計を 1 回の演算として記録します。</para>
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
    /// Records the Bernoulli log mass for each observation as a single operation.
    /// <para>ベルヌーイ分布の観測ごとの対数確率質量を 1 回の演算として記録します。</para>
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
        _ = new BernoulliDistribution(probability ?? 0, forDifferentiation: true);

    private static LikelihoodEvaluation EvaluateLikelihoodPoint(long observation, double probability, double _, double __, LikelihoodGradients gradients)
    {
        var distribution = new BernoulliDistribution(probability, forDifferentiation: true);
        return new(distribution.LogPmf(observation),
            new((gradients & LikelihoodGradients.First) != 0 ? distribution.LogPmfProbabilityDerivative(observation) : 0));
    }

    /// <summary>
    /// Records the Bernoulli log probability mass and its derivative with respect to the success probability.
    /// <para>ベルヌーイ分布の対数確率質量と成功確率に関する微分を記録します。</para>
    /// </summary>
    /// <param name="probability">The success probability <c>p</c>, between 0 and 1. / 0 以上 1 以下の成功確率 <c>p</c>。</param>
    /// <param name="k">The observed value. / 観測値。</param>
    /// <returns>The log probability mass node. / 対数確率質量を表すノード。</returns>
    /// <remarks>
    /// <para>The derivative takes its one-sided limit at <c>p = 0</c> or <c>p = 1</c>. Observations outside {0, 1} have derivative zero.</para>
    /// <para><c>p</c> が 0 または 1 のときは片側極限を微分値として用います。<c>k</c> が 0 と 1 以外の場合、微分は 0 です。</para>
    /// </remarks>
    /// <exception cref="DistributionParameterOutOfRangeException">The success probability is outside [0, 1]. / 成功確率が 0 以上 1 以下の範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">The probability node is stale or its tape has already run backward. / 成功確率のノードが失効しているか、テープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPmf(ScalarNode probability, long k)
    {
        var distribution = new BernoulliDistribution(probability.Value, forDifferentiation: true);
        double value = distribution.LogPmf(k);
        if (!probability.RequiresGradient)
        {
            return value;
        }

        return ScalarNode.Record(value, [(probability, distribution.LogPmfProbabilityDerivative(k))]);
    }

    private double LogPmfProbabilityDerivative(long k) => k switch
    {
        0 => -1 / _q,
        1 => 1 / Math.Abs(_p),
        _ => 0,
    };

    /// <summary>
    /// Records the Bernoulli log likelihood of independent observations as a single operation.
    /// <para>ベルヌーイ分布に従う独立な観測値の対数尤度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="probability">The success probability <c>p</c>, between 0 and 1. / 0 以上 1 以下の成功確率 <c>p</c>。</param>
    /// <param name="observations">The observations. / 観測値のベクトル。</param>
    /// <returns>The log likelihood node for the observations. / 観測全体の対数尤度を表すノード。</returns>
    /// <remarks>
    /// <para>Observations outside {0, 1} contribute negative infinity to the value and zero to the gradient. An empty vector gives zero for both.</para>
    /// <para>0 と 1 以外の観測値では対数尤度が負の無限大となり、その観測値から勾配への寄与は 0 になります。空のベクトルでは値と勾配がともに 0 です。</para>
    /// </remarks>
    /// <exception cref="DistributionParameterOutOfRangeException">The success probability is outside [0, 1]. / 成功確率が 0 以上 1 以下の範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">The probability node is stale or its tape has already run backward. / 成功確率のノードが失効しているか、テープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPmf(ScalarNode probability, ReadOnlyVectorSpan<long> observations)
    {
        var distribution = new BernoulliDistribution(probability.Value, forDifferentiation: true);
        bool requiresGradient = probability.RequiresGradient;
        double logSuccess = Math.Log(distribution._p);
        double logFailure = distribution.LogQ;
        double value = 0;
        int successes = 0, failures = 0;
        for (int i = 0; i < observations.Length; i++)
        {
            long k = observations[i];
            switch (k)
            {
                case 0:
                    value += logFailure;
                    failures++;
                    break;
                case 1:
                    value += logSuccess;
                    successes++;
                    break;
                default:
                    value = double.NegativeInfinity;
                    break;
            }
        }

        return requiresGradient
            ? ScalarNode.Record(value, [(probability, distribution.LogLikelihoodDerivative(successes, failures))])
            : value;
    }

    private double LogLikelihoodDerivative(int successes, int failures)
    {
        if (_p == 0)
        {
            return successes > 0 ? double.PositiveInfinity : -failures;
        }

        if (_q == 0)
        {
            return failures > 0 ? double.NegativeInfinity : successes;
        }

        // 和 n₁/p − n₀/(1 − p) の桁落ちを避ける。
        double numerator = Math.FusedMultiplyAdd(-(successes + failures), _p, successes);
        return numerator / (_p * _q);
    }

    /// <summary>
    /// Records the Bernoulli log cumulative probability and its derivative with respect to the success probability.
    /// <para>ベルヌーイ分布の対数累積確率と成功確率に関する微分を記録します。</para>
    /// </summary>
    /// <param name="probability">The success probability <c>p</c>, between 0 and 1. / 0 以上 1 以下の成功確率 <c>p</c>。</param>
    /// <param name="k">The threshold in <c>P(X ≤ k)</c>. / 下側確率 <c>P(X ≤ k)</c> のしきい値。</param>
    /// <returns>The log cumulative probability node. / 対数累積確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">The success probability is outside [0, 1]. / 成功確率が 0 以上 1 以下の範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">The probability node is stale or its tape has already run backward. / 成功確率のノードが失効しているか、テープが逆伝播済みの場合。</exception>
    public static ScalarNode LogCdf(ScalarNode probability, long k)
    {
        if (k == 0)
        {
            return LogPmf(probability, 0);
        }

        double value = new BernoulliDistribution(probability.Value, forDifferentiation: true).LogCdf(k);
        return ScalarNode.Record(value, [(probability, 0)]);
    }

    /// <summary>
    /// Records the Bernoulli log survival probability and its derivative with respect to the success probability.
    /// <para>ベルヌーイ分布の対数上側確率と成功確率に関する微分を記録します。</para>
    /// </summary>
    /// <param name="probability">The success probability <c>p</c>, between 0 and 1. / 0 以上 1 以下の成功確率 <c>p</c>。</param>
    /// <param name="k">The threshold in <c>P(X &gt; k)</c>. / 上側確率 <c>P(X &gt; k)</c> のしきい値。</param>
    /// <returns>The log survival probability node. / 対数上側確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">The success probability is outside [0, 1]. / 成功確率が 0 以上 1 以下の範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">The probability node is stale or its tape has already run backward. / 成功確率のノードが失効しているか、テープが逆伝播済みの場合。</exception>
    public static ScalarNode LogSf(ScalarNode probability, long k)
    {
        if (k == 0)
        {
            return LogPmf(probability, 1);
        }

        double value = new BernoulliDistribution(probability.Value, forDifferentiation: true).LogSf(k);
        return ScalarNode.Record(value, [(probability, 0)]);
    }
}
