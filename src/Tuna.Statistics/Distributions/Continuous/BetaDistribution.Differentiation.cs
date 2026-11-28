using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public readonly partial struct BetaDistribution
{
    /// <summary>
    /// Records the sum of beta log densities with shared or per-observation parameters as a single operation.
    /// <para>共通または観測ごとのパラメータを用いて、ベータ分布の対数尤度の合計を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="alpha">First shape. / 第 1 形状パラメータ。</param>
    /// <param name="beta">Second shape. / 第 2 形状パラメータ。</param>
    /// <param name="observations">The observations. / 観測値のベクトル。</param>
    /// <returns>The log likelihood in observation order (zero for empty observations). / 入力順に集計した対数尤度。観測値が空の場合は 0。</returns>
    /// <exception cref="ArgumentException">A vector parameter has a different length from the observations. / パラメータベクトルの長さが観測数と一致しない場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is invalid, the tapes differ, or backpropagation has completed for a differentiable input. / ノードが無効、テープが異なる、または微分対象のノードのテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarSetNode<double> alpha, ScalarSetNode<double> beta,
        ReadOnlyVectorSpan<double> observations) =>
        PointwiseLogLikelihood.RecordSum(observations, EvaluateLikelihoodPoint, ValidateSharedLikelihoodParameters, alpha, beta);

    /// <summary>
    /// Records the beta log density for each observation as a single operation.
    /// <para>ベータ分布の観測ごとの対数密度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="alpha">First shape. / 第 1 形状パラメータ。</param>
    /// <param name="beta">Second shape. / 第 2 形状パラメータ。</param>
    /// <param name="observations">The observations. / 観測値のベクトル。</param>
    /// <param name="tape">The tape, required when all parameters are scalar constants. / テープ。すべてのパラメータがスカラー定数の場合は必須です。</param>
    /// <returns>The log densities in observation order. / 入力順に並べた観測ごとの対数密度。</returns>
    /// <exception cref="ArgumentException">A vector parameter has a different length from the observations, or no tape is available. / パラメータベクトルの長さが観測数と一致しない場合、またはテープが未指定の場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is invalid, the tapes differ, or backpropagation has completed for a differentiable input. / ノードが無効、テープが異なる、または微分対象のノードのテープが逆伝播済みの場合。</exception>
    public static VectorNode<double> LogPdfPointwise(ScalarSetNode<double> alpha, ScalarSetNode<double> beta,
        ReadOnlyVectorSpan<double> observations, Tape? tape = null) =>
        PointwiseLogLikelihood.RecordVector(observations, EvaluateLikelihoodPoint, ValidateSharedLikelihoodParameters, alpha, beta, tape: tape);

    private static void ValidateSharedLikelihoodParameters(double? alpha, double? beta, double? unusedThird) =>
        _ = new BetaDistribution(alpha ?? 1, beta ?? 1, forDifferentiation: true);

    private static LikelihoodEvaluation EvaluateLikelihoodPoint(double observation, double alpha, double beta, double _, LikelihoodGradients gradients)
    {
        var distribution = new BetaDistribution(alpha, beta, forDifferentiation: true);
        double value = distribution.LogPdf(observation);
        if (gradients == LikelihoodGradients.None)
        {
            return new(value);
        }

        var (da, db, _) = distribution.LogPdfDerivatives(observation,
            (gradients & LikelihoodGradients.First) != 0, (gradients & LikelihoodGradients.Second) != 0, requiresX: false);
        return new(value, new(da), new(db));
    }

    /// <summary>
    /// Records the beta log density and its derivatives.
    /// <para>ベータ分布の対数密度とその微分を記録します。</para>
    /// </summary>
    /// <param name="alpha">The first positive finite shape. / 正で有限の第 1 形状パラメータ。</param>
    /// <param name="beta">The second positive finite shape. / 正で有限の第 2 形状パラメータ。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log density node. / 対数密度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A shape is outside its domain. / 形状パラメータが範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarNode alpha, ScalarNode beta, ScalarNode x)
    {
        var distribution = new BetaDistribution(alpha.Value, beta.Value, forDifferentiation: true);
        double value = distribution.LogPdf(x.Value);
        if (!alpha.RequiresGradient && !beta.RequiresGradient && !x.RequiresGradient)
        {
            return value;
        }

        var (da, db, dx) = distribution.LogPdfDerivatives(x.Value, alpha.RequiresGradient, beta.RequiresGradient);
        return ScalarNode.Record(value, [(alpha, da), (beta, db), (x, dx)]);
    }

    /// <summary>
    /// Records the beta log likelihood of independent observations as a single operation.
    /// <para>ベータ分布に従う独立な観測値の対数尤度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="alpha">The first positive finite shape. / 正で有限の第 1 形状パラメータ。</param>
    /// <param name="beta">The second positive finite shape. / 正で有限の第 2 形状パラメータ。</param>
    /// <param name="observations">The observations, summed in input order. / 入力順に集計する観測値。</param>
    /// <returns>The log likelihood node. / 対数尤度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A shape is outside its domain. / 形状パラメータが範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarNode alpha, ScalarNode beta, ReadOnlyVectorSpan<double> observations)
    {
        var distribution = new BetaDistribution(alpha.Value, beta.Value, forDifferentiation: true);
        bool requiresAlpha = alpha.RequiresGradient, requiresBeta = beta.RequiresGradient;
        double alphaNormalization = requiresAlpha && observations.Length != 0
            ? XMath.Special.LogBetaDerivative(distribution._alpha, distribution._beta) : 0;
        double betaNormalization = requiresBeta && observations.Length != 0
            ? XMath.Special.LogBetaDerivative(distribution._beta, distribution._alpha) : 0;
        double value = 0;
        var alphaSum = new ScaledDerivativeSum();
        var betaSum = new ScaledDerivativeSum();
        for (int i = 0; i < observations.Length; i++)
        {
            double x = observations[i];
            value += distribution.LogPdf(x);
            if (double.IsNaN(x))
            {
                alphaSum.Add(new(double.NaN));
                betaSum.Add(new(double.NaN));
            }
            else if (x is >= 0 and <= 1)
            {
                if (requiresAlpha)
                {
                    alphaSum.Add(new(x == 0 ? double.NegativeInfinity : Math.Log(x) - alphaNormalization));
                }

                if (requiresBeta)
                {
                    betaSum.Add(new(x == 1 ? double.NegativeInfinity : DistributionMath.Log1p(-x) - betaNormalization));
                }
            }
        }

        return requiresAlpha || requiresBeta
            ? ScalarNode.Record(value, [(alpha, alphaSum.Value), (beta, betaSum.Value)])
            : value;
    }

    /// <summary>
    /// Records the beta log cumulative probability and its derivative with respect to the variate.
    /// <para>ベータ分布の対数累積確率と評価点に関する微分を記録します。</para>
    /// </summary>
    /// <param name="alpha">The first positive finite shape, supplied as a constant. / 定数として渡す正で有限の第 1 形状パラメータ。</param>
    /// <param name="beta">The second positive finite shape, supplied as a constant. / 定数として渡す正で有限の第 2 形状パラメータ。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log cumulative probability node. / 対数累積確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A shape is outside its domain. / 形状パラメータが範囲外の場合。</exception>
    /// <exception cref="NotSupportedException">A shape node requires a gradient. / 形状パラメータのノードが勾配を必要とする場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale or its tape has run backward. / ノードが失効している場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogCdf(ScalarNode alpha, ScalarNode beta, ScalarNode x) => RecordLogTail(alpha, beta, x, upper: false);

    /// <summary>
    /// Records the beta log survival probability and its derivative with respect to the variate.
    /// <para>ベータ分布の対数生存確率と評価点に関する微分を記録します。</para>
    /// </summary>
    /// <param name="alpha">The first positive finite shape, supplied as a constant. / 定数として渡す正で有限の第 1 形状パラメータ。</param>
    /// <param name="beta">The second positive finite shape, supplied as a constant. / 定数として渡す正で有限の第 2 形状パラメータ。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log survival probability node. / 対数生存確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A shape is outside its domain. / 形状パラメータが範囲外の場合。</exception>
    /// <exception cref="NotSupportedException">A shape node requires a gradient. / 形状パラメータのノードが勾配を必要とする場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale or its tape has run backward. / ノードが失効している場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogSf(ScalarNode alpha, ScalarNode beta, ScalarNode x) => RecordLogTail(alpha, beta, x, upper: true);

    private (double Alpha, double Beta, double X) LogPdfDerivatives(double x, bool requiresAlpha, bool requiresBeta, bool requiresX = true)
    {
        if (double.IsNaN(x))
        {
            return (double.NaN, double.NaN, double.NaN);
        }

        if (x < 0 || x > 1)
        {
            return default;
        }

        double da = requiresAlpha ? Math.Log(x) - XMath.Special.LogBetaDerivative(_alpha, _beta) : 0;
        double db = requiresBeta ? DistributionMath.Log1p(-x) - XMath.Special.LogBetaDerivative(_beta, _alpha) : 0;
        if (x == 0)
        {
            return (double.NegativeInfinity, db, _alpha < 1 ? double.NegativeInfinity : _alpha > 1 ? double.PositiveInfinity : 1 - _beta);
        }

        if (x == 1)
        {
            return (da, double.NegativeInfinity, _beta < 1 ? double.PositiveInfinity : _beta > 1 ? double.NegativeInfinity : _alpha - 1);
        }

        if (!requiresX)
        {
            return (da, db, 0);
        }

        double y = 1 - x;
        double left = _alpha * y, right = _beta * x;
        double remainder = Math.FusedMultiplyAdd(_alpha, y, -left) - Math.FusedMultiplyAdd(_beta, x, -right);
        remainder = Math.FusedMultiplyAdd(_alpha, CompensatedSum.DifferenceError(1, x, y), remainder);
        double centered = left - right;
        double adjustment = Math.FusedMultiplyAdd(2, x, -1);
        double numerator = centered + adjustment;
        numerator += remainder + CompensatedSum.DifferenceError(left, right, centered)
            + CompensatedSum.DifferenceError(centered, -adjustment, numerator);
        return (da, db, numerator / y / x);
    }

    private static ScalarNode RecordLogTail(ScalarNode alpha, ScalarNode beta, ScalarNode x, bool upper)
    {
        var distribution = new BetaDistribution(alpha.Value, beta.Value, forDifferentiation: true);
        if (alpha.RequiresGradient || beta.RequiresGradient)
        {
            throw new NotSupportedException("Beta log tail derivatives with respect to the shape parameters are unavailable.");
        }

        double observation = x.Value;
        double value = upper ? distribution.LogSf(observation) : distribution.LogCdf(observation);
        if (!x.RequiresGradient)
        {
            return value;
        }

        double gradient = observation switch
        {
            < 0 or > 1 => 0,
            0 => upper ? -EndpointDensity(distribution._alpha, distribution._beta) : double.PositiveInfinity,
            1 => upper ? double.NegativeInfinity : EndpointDensity(distribution._beta, distribution._alpha),
            _ when double.IsNaN(observation) => double.NaN,
            _ => (upper ? -1 : 1) * Math.Exp(XMath.Special.LogBetaTailPrefixRatio(
                distribution._alpha, distribution._beta, observation, value, upper)
                - Math.Log(observation) - DistributionMath.Log1p(-observation)),
        };
        return ScalarNode.Record(value, [(x, gradient)]);
    }
}
