using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public readonly partial struct NegativeBinomialDistribution
{
    /// <summary>
    /// Records the negative binomial log mass and its derivatives.
    /// <para>負の二項分布の対数質量とその微分を記録します。</para>
    /// </summary>
    /// <param name="successes">The success count in the distribution domain. / 分布の定義域内の成功回数。</param>
    /// <param name="probability">The success probability. / 成功確率。</param>
    /// <param name="k">The integer observation. / 整数の観測値。</param>
    /// <returns>The log mass node. / 対数質量を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPmf(ScalarNode successes, ScalarNode probability, long k)
    {
        var d = new NegativeBinomialDistribution(successes.Value, probability.Value, forDifferentiation: true);
        double value = d.LogPmf(k);
        if (!successes.RequiresGradient && !probability.RequiresGradient)
        {
            return value;
        }

        double dr = k >= 0 && successes.RequiresGradient ? d.ShapeDerivative(k, Math.Log(d._p)) : 0;
        double dp = k >= 0 && probability.RequiresGradient ? d.ProbabilityDerivative(k) : 0;
        return ScalarNode.Record(value, [(successes, dr), (probability, dp)]);
    }

    /// <summary>
    /// Records the negative binomial log cumulative probability and its derivatives.
    /// <para>負の二項分布の対数累積確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="successes">The success count in the distribution domain, supplied as a constant. / 定数として渡す分布の定義域内の成功回数。</param>
    /// <param name="probability">The success probability. / 成功確率。</param>
    /// <param name="k">The integer observation. / 整数の観測値。</param>
    /// <returns>The log cumulative probability node. / 対数累積確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    /// <exception cref="NotSupportedException">A shape parameter requires a gradient. / 形状パラメータが勾配を必要とする場合。</exception>
    public static ScalarNode LogCdf(ScalarNode successes, ScalarNode probability, long k)
    {
        var distribution = new NegativeBinomialDistribution(successes.Value, probability.Value, forDifferentiation: true);
        if (successes.RequiresGradient)
        {
            throw new NotSupportedException("The log tail derivatives with respect to the shape parameters are unavailable.");
        }

        return ScalarDistributionLogProbability.RecordMass(distribution, successes, probability, k, LogProbability.LowerTail, TailGradient);
    }

    /// <summary>
    /// Records the negative binomial log survival probability and its derivatives.
    /// <para>負の二項分布の対数生存確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="successes">The success count in the distribution domain, supplied as a constant. / 定数として渡す分布の定義域内の成功回数。</param>
    /// <param name="probability">The success probability. / 成功確率。</param>
    /// <param name="k">The integer observation. / 整数の観測値。</param>
    /// <returns>The log survival probability node. / 対数生存確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    /// <exception cref="NotSupportedException">A shape parameter requires a gradient. / 形状パラメータが勾配を必要とする場合。</exception>
    public static ScalarNode LogSf(ScalarNode successes, ScalarNode probability, long k)
    {
        var distribution = new NegativeBinomialDistribution(successes.Value, probability.Value, forDifferentiation: true);
        if (successes.RequiresGradient)
        {
            throw new NotSupportedException("The log tail derivatives with respect to the shape parameters are unavailable.");
        }

        return ScalarDistributionLogProbability.RecordMass(distribution, successes, probability, k, LogProbability.UpperTail, TailGradient);
    }

    /// <summary>
    /// Records the negative binomial log likelihood of independent observations as a single operation.
    /// <para>負の二項分布の対数尤度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="successes">The success count in the distribution domain. / 分布の定義域内の成功回数。</param>
    /// <param name="probability">The success probability. / 成功確率。</param>
    /// <param name="observations">The observations, summed in input order. / 入力順に集計する観測値。</param>
    /// <returns>The log likelihood node. / 対数尤度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPmf(ScalarNode successes, ScalarNode probability, ReadOnlyVectorSpan<long> observations)
    {
        var d = new NegativeBinomialDistribution(successes.Value, probability.Value, forDifferentiation: true);
        bool differentiateShape = successes.RequiresGradient, differentiateProbability = probability.RequiresGradient;
        double logP = differentiateShape ? Math.Log(d._p) : 0;
        // 小さい観測値の出現回数を数え、同じ形状微分を繰り返し計算しない。
        Span<int> counts = differentiateShape ? stackalloc int[64] : [];
        counts.Clear();
        double value = 0;
        var shapeSum = new ScaledDerivativeSum();
        var probabilitySum = new ScaledDerivativeSum();
        for (int i = 0; i < observations.Length; i++)
        {
            long k = observations[i];
            value += d.LogPmf(k);
            if (k < 0)
            {
                continue;
            }

            if (differentiateShape)
            {
                if (k < counts.Length)
                {
                    counts[(int)k]++;
                }
                else
                {
                    shapeSum.Add(new(d.ShapeDerivative(k, logP)));
                }
            }

            if (differentiateProbability)
            {
                probabilitySum.Add(new(d.ProbabilityDerivative(k)));
            }
        }

        for (int k = 0; k < counts.Length; k++)
        {
            int count = counts[k];
            if (count == 0)
            {
                continue;
            }

            double derivative = d.ShapeDerivative(k, logP);
            double contribution = derivative * count;
            shapeSum.Add(double.IsInfinity(contribution) && double.IsFinite(derivative)
                ? new(Math.ScaleB(derivative, -32) * count, 32) : new(contribution));
        }

        return differentiateShape || differentiateProbability
            ? ScalarNode.Record(value, [(successes, shapeSum.Value), (probability, probabilitySum.Value)]) : value;
    }

    private double ShapeDerivative(long k, double logP) =>
        logP - (k == 0 ? 0 : XMath.Special.LogBetaDerivative(_r, k));

    private double ProbabilityDerivative(long k) =>
        k == 0 ? _r / _p : _q == 0 ? double.NegativeInfinity : Math.FusedMultiplyAdd(-k, _p, _r * _q) / _p / _q;

    private static (double, double) TailGradient(NegativeBinomialDistribution d, long k, double value, LogProbability kind)
    {
        if (k < 0)
        {
            return default;
        }

        double r = d._r, p = d._p, q = d._q;
        bool upper = kind == LogProbability.UpperTail;
        if (q == 0)
        {
            return (0, upper ? double.NegativeInfinity : k == 0 ? r : 0);
        }

        var (mantissa, exponent) = p <= 0.5 ? XMath.Special.ScaledBetaTailPrefixRatio(r, k + 1.0, p, 0, value, upper)
            : XMath.Special.ScaledBetaTailPrefixRatio(k + 1.0, r, q, 0, value, !upper);
        return (0, (upper ? -1 : 1) * DistributionMath.ScaledRatio([], [p, q], mantissa, exponent));
    }
}
