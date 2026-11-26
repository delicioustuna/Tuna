using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public readonly partial struct BernoulliDistribution
{
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

        double derivative = k switch
        {
            0 => -1 / distribution._q,
            1 => 1 / Math.Abs(distribution._p),
            _ => 0,
        };
        return ScalarNode.Record(value, [(probability, derivative)]);
    }

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
