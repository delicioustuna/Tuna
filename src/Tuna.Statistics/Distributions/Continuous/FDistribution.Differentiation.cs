using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

public readonly partial struct FDistribution
{
    /// <summary>
    /// Records the sum of F distribution log densities with shared or per-observation parameters as a single operation.
    /// <para>共通または観測ごとのパラメータを用いて、F 分布の対数尤度の合計を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="numeratorDegreesOfFreedom">Numerator degrees of freedom. / 分子の自由度。</param>
    /// <param name="denominatorDegreesOfFreedom">Denominator degrees of freedom. / 分母の自由度。</param>
    /// <param name="observations">The observations. / 観測値のベクトル。</param>
    /// <returns>The log likelihood in observation order (zero for empty observations). / 入力順に集計した対数尤度。観測値が空の場合は 0。</returns>
    /// <exception cref="ArgumentException">A vector parameter has a different length from the observations. / パラメータベクトルの長さが観測数と一致しない場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is invalid, the tapes differ, or backpropagation has completed for a differentiable input. / ノードが無効、テープが異なる、または微分対象のノードのテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarSetNode<double> numeratorDegreesOfFreedom, ScalarSetNode<double> denominatorDegreesOfFreedom,
        ReadOnlyVectorSpan<double> observations) =>
        PointwiseLogLikelihood.RecordSum(observations, EvaluateLikelihoodPoint, ValidateSharedLikelihoodParameters, numeratorDegreesOfFreedom, denominatorDegreesOfFreedom);

    /// <summary>
    /// Records the F distribution log density for each observation as a single operation.
    /// <para>F 分布の観測ごとの対数密度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="numeratorDegreesOfFreedom">Numerator degrees of freedom. / 分子の自由度。</param>
    /// <param name="denominatorDegreesOfFreedom">Denominator degrees of freedom. / 分母の自由度。</param>
    /// <param name="observations">The observations. / 観測値のベクトル。</param>
    /// <param name="tape">The tape, required when all parameters are scalar constants. / テープ。すべてのパラメータがスカラー定数の場合は必須です。</param>
    /// <returns>The log densities in observation order. / 入力順に並べた観測ごとの対数密度。</returns>
    /// <exception cref="ArgumentException">A vector parameter has a different length from the observations, or no tape is available. / パラメータベクトルの長さが観測数と一致しない場合、またはテープが未指定の場合。</exception>
    /// <exception cref="DistributionParameterOutOfRangeException">A distribution parameter is outside its domain. / 分布パラメータが定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is invalid, the tapes differ, or backpropagation has completed for a differentiable input. / ノードが無効、テープが異なる、または微分対象のノードのテープが逆伝播済みの場合。</exception>
    public static VectorNode<double> LogPdfPointwise(ScalarSetNode<double> numeratorDegreesOfFreedom, ScalarSetNode<double> denominatorDegreesOfFreedom,
        ReadOnlyVectorSpan<double> observations, Tape? tape = null) =>
        PointwiseLogLikelihood.RecordVector(observations, EvaluateLikelihoodPoint, ValidateSharedLikelihoodParameters, numeratorDegreesOfFreedom, denominatorDegreesOfFreedom, tape: tape);

    private static void ValidateSharedLikelihoodParameters(double? numeratorDegreesOfFreedom, double? denominatorDegreesOfFreedom, double? unusedThird) =>
        _ = new FDistribution(numeratorDegreesOfFreedom ?? 1, denominatorDegreesOfFreedom ?? 1, forDifferentiation: true);

    private static LikelihoodEvaluation EvaluateLikelihoodPoint(double observation, double numeratorDegreesOfFreedom, double denominatorDegreesOfFreedom,
        double _, LikelihoodGradients gradients)
    {
        var distribution = new FDistribution(numeratorDegreesOfFreedom, denominatorDegreesOfFreedom, forDifferentiation: true);
        double value = distribution.LogPdf(observation);
        if (gradients == LikelihoodGradients.None)
        {
            return new(value);
        }

        var (da, db, _) = distribution.DensityGradient(observation,
            (gradients & LikelihoodGradients.First) != 0, (gradients & LikelihoodGradients.Second) != 0, differentiateX: false);
        return new(value, new(da), new(db));
    }

    /// <summary>
    /// Records the F log density and its derivatives.
    /// <para>F 分布の対数密度とその微分を記録します。</para>
    /// </summary>
    /// <param name="numeratorDegreesOfFreedom">The positive finite numerator degrees of freedom. / 正で有限の分子の自由度。</param>
    /// <param name="denominatorDegreesOfFreedom">The positive finite denominator degrees of freedom. / 正で有限の分母の自由度。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log density node. / 対数密度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarNode numeratorDegreesOfFreedom, ScalarNode denominatorDegreesOfFreedom, ScalarNode x)
    {
        var d = new FDistribution(numeratorDegreesOfFreedom.Value, denominatorDegreesOfFreedom.Value, forDifferentiation: true);
        double value = d.LogPdf(x.Value);
        if (!numeratorDegreesOfFreedom.RequiresGradient && !denominatorDegreesOfFreedom.RequiresGradient && !x.RequiresGradient)
        {
            return value;
        }

        var (da, db, dx) = d.DensityGradient(x.Value,
            numeratorDegreesOfFreedom.RequiresGradient, denominatorDegreesOfFreedom.RequiresGradient, x.RequiresGradient);
        return ScalarNode.Record(value, [(numeratorDegreesOfFreedom, da), (denominatorDegreesOfFreedom, db), (x, dx)]);
    }

    /// <summary>
    /// Records the F log cumulative probability and its derivatives.
    /// <para>F 分布の対数累積確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="numeratorDegreesOfFreedom">The positive finite numerator degrees of freedom, supplied as a constant. / 定数として渡す正で有限の分子の自由度。</param>
    /// <param name="denominatorDegreesOfFreedom">The positive finite denominator degrees of freedom, supplied as a constant. / 定数として渡す正で有限の分母の自由度。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log cumulative probability node. / 対数累積確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    /// <exception cref="NotSupportedException">A shape parameter requires a gradient. / 形状パラメータが勾配を必要とする場合。</exception>
    public static ScalarNode LogCdf(ScalarNode numeratorDegreesOfFreedom, ScalarNode denominatorDegreesOfFreedom, ScalarNode x)
    {
        var distribution = new FDistribution(numeratorDegreesOfFreedom.Value, denominatorDegreesOfFreedom.Value, forDifferentiation: true);
        if (numeratorDegreesOfFreedom.RequiresGradient || denominatorDegreesOfFreedom.RequiresGradient)
        {
            throw new NotSupportedException("The log tail derivatives with respect to the shape parameters are unavailable.");
        }

        return ScalarDistributionLogProbability.Record(distribution, numeratorDegreesOfFreedom, denominatorDegreesOfFreedom, x, LogProbability.LowerTail, TailGradient);
    }

    /// <summary>
    /// Records the F log survival probability and its derivatives.
    /// <para>F 分布の対数生存確率とその微分を記録します。</para>
    /// </summary>
    /// <param name="numeratorDegreesOfFreedom">The positive finite numerator degrees of freedom, supplied as a constant. / 定数として渡す正で有限の分子の自由度。</param>
    /// <param name="denominatorDegreesOfFreedom">The positive finite denominator degrees of freedom, supplied as a constant. / 定数として渡す正で有限の分母の自由度。</param>
    /// <param name="x">The variate. / 評価点。</param>
    /// <returns>The log survival probability node. / 対数生存確率を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    /// <exception cref="NotSupportedException">A shape parameter requires a gradient. / 形状パラメータが勾配を必要とする場合。</exception>
    public static ScalarNode LogSf(ScalarNode numeratorDegreesOfFreedom, ScalarNode denominatorDegreesOfFreedom, ScalarNode x)
    {
        var distribution = new FDistribution(numeratorDegreesOfFreedom.Value, denominatorDegreesOfFreedom.Value, forDifferentiation: true);
        if (numeratorDegreesOfFreedom.RequiresGradient || denominatorDegreesOfFreedom.RequiresGradient)
        {
            throw new NotSupportedException("The log tail derivatives with respect to the shape parameters are unavailable.");
        }

        return ScalarDistributionLogProbability.Record(distribution, numeratorDegreesOfFreedom, denominatorDegreesOfFreedom, x, LogProbability.UpperTail, TailGradient);
    }

    /// <summary>
    /// Records the F log likelihood of independent observations as a single operation.
    /// <para>F 分布の対数尤度を 1 回の演算として記録します。</para>
    /// </summary>
    /// <param name="numeratorDegreesOfFreedom">The positive finite numerator degrees of freedom. / 正で有限の分子の自由度。</param>
    /// <param name="denominatorDegreesOfFreedom">The positive finite denominator degrees of freedom. / 正で有限の分母の自由度。</param>
    /// <param name="observations">The observations, summed in input order. / 入力順に集計する観測値。</param>
    /// <returns>The log likelihood node. / 対数尤度を表すノード。</returns>
    /// <exception cref="DistributionParameterOutOfRangeException">A parameter is outside its distribution domain. / パラメータが分布の定義域外の場合。</exception>
    /// <exception cref="InvalidOperationException">A node is stale, belongs to a different tape, or its tape has run backward. / ノードが失効している場合、別のテープに属する場合、またはテープが逆伝播済みの場合。</exception>
    public static ScalarNode LogPdf(ScalarNode numeratorDegreesOfFreedom, ScalarNode denominatorDegreesOfFreedom, ReadOnlyVectorSpan<double> observations)
    {
        var d = new FDistribution(numeratorDegreesOfFreedom.Value, denominatorDegreesOfFreedom.Value, forDifferentiation: true);
        bool requiresGradient = numeratorDegreesOfFreedom.RequiresGradient || denominatorDegreesOfFreedom.RequiresGradient;
        double? dA = numeratorDegreesOfFreedom.RequiresGradient && !observations.IsEmpty
            ? XMath.Special.LogBetaDerivative(d.A, d.B) : null;
        double? dB = denominatorDegreesOfFreedom.RequiresGradient && !observations.IsEmpty
            ? XMath.Special.LogBetaDerivative(d.B, d.A) : null;
        double value = 0;
        var numeratorSum = new ScaledDerivativeSum();
        var denominatorSum = new ScaledDerivativeSum();
        for (int i = 0; i < observations.Length; i++)
        {
            double x = observations[i];
            value += d.LogPdf(x);
            if (requiresGradient)
            {
                var (da, db, _) = d.DensityGradient(x, dA, dB, differentiateX: false);
                numeratorSum.Add(new(da));
                denominatorSum.Add(new(db));
            }
        }

        return requiresGradient
            ? ScalarNode.Record(value, [(numeratorDegreesOfFreedom, numeratorSum.Value), (denominatorDegreesOfFreedom, denominatorSum.Value)])
            : value;
    }

    private (double, double, double) DensityGradient(double x, bool differentiateA, bool differentiateB, bool differentiateX)
    {
        double? dA = differentiateA ? XMath.Special.LogBetaDerivative(A, B) : null;
        double? dB = differentiateB ? XMath.Special.LogBetaDerivative(B, A) : null;
        return DensityGradient(x, dA, dB, differentiateX);
    }

    private (double, double, double) DensityGradient(double x, double? dA, double? dB, bool differentiateX)
    {
        if (double.IsNaN(x))
        {
            return (double.NaN, double.NaN, double.NaN);
        }

        if (x < 0)
        {
            return default;
        }

        double a = A, b = B;
        if (x == 0)
        {
            return (double.NegativeInfinity, a == 1 ? 0 : -0.5 * dB.GetValueOrDefault() - a / _denominator,
                a < 1 ? double.NegativeInfinity : a > 1 ? double.PositiveInfinity : -1 - 2 / _denominator);
        }

        if (double.IsPositiveInfinity(x))
        {
            return (b == 1 ? 0 : -0.5 * dA.GetValueOrDefault() - b / _numerator, double.NegativeInfinity, -0.0);
        }

        bool smallUpper = SmallCoordinate(x, out double small);
        double centered = smallUpper ? Math.FusedMultiplyAdd(a, small, -b * (1 - small))
            : Math.FusedMultiplyAdd(-b, small, a * (1 - small));
        double da = 0, db = 0, dx = 0;
        if (dA.HasValue || dB.HasValue)
        {
            double logSmall = small >= NumericConstants.MinNormal ? Math.Log(small) : LogSmallCoordinate(LogR(x), smallUpper);
            // 大きい側の座標の対数は、小さい側の座標から計算して桁落ちを避ける。
            double logLarge = DistributionMath.Log1p(-small);
            if (dA is double dBetaA)
            {
                double logP = smallUpper ? logLarge : logSmall;
                da = 0.5 * (logP - dBetaA) + centered / _numerator;
            }

            if (dB is double dBetaB)
            {
                double logQ = smallUpper ? logSmall : logLarge;
                db = 0.5 * (logQ - dBetaB) - centered / _denominator;
            }
        }

        if (differentiateX)
        {
            dx = !smallUpper && a == 1
                ? -(1 - small) * (1 + 2 / _denominator)
                : (!smallUpper && a is >= 0.5 and <= 2
                    ? Math.FusedMultiplyAdd(-a, small, a - 1) - b * small : centered - 1) / x;
        }

        return (da, db, dx);
    }

    private static (double, double, double) TailGradient(FDistribution d, double x, double value, LogProbability kind)
    {
        if (double.IsNaN(x))
        {
            return (double.NaN, double.NaN, double.NaN);
        }

        if (x < 0 || double.IsPositiveInfinity(x))
        {
            return default;
        }

        double a = d.A, b = d.B;
        if (x == 0)
        {
            return kind == LogProbability.LowerTail ? (0, 0, double.PositiveInfinity)
                : (0, 0, a < 1 ? double.NegativeInfinity : a == 1 ? -1 : 0);
        }

        bool smallUpper = d.SmallCoordinate(x, out double small);
        bool upperTail = kind == LogProbability.UpperTail;
        var (smallShape, otherShape) = smallUpper ? (b, a) : (a, b);
        double sign = upperTail ? -1 : 1;
        (double Mantissa, long Exponent) ratio;
        if (small >= NumericConstants.MinNormal)
        {
            double correction = d.CoordinateCorrection(x, small, smallUpper);
            ratio = XMath.Special.ScaledBetaTailPrefixRatio(smallShape, otherShape, small, correction, value,
                upperTail != smallUpper);
        }
        else
        {
            if (upperTail == smallUpper)
            {
                return (0, 0, sign * (smallShape / x));
            }

            var (hi, lo) = XMath.Special.PreciseLogProductRatio(d._numerator, x, d._denominator);
            ratio = XMath.Special.ScaledBetaTailPrefixRatioFromLog(smallShape, otherShape, smallUpper ? (-hi, -lo) : (hi, lo), value);
        }

        return (0, 0, sign * DistributionMath.ScaledRatio([], [x], ratio.Mantissa, ratio.Exponent));
    }

    private double CoordinateCorrection(double x, double coordinate, bool upper)
    {
        double product = _numerator * x;
        double total = product + _denominator;
        if (!double.IsFinite(total) || product < NumericConstants.MinNormal)
        {
            return 0;
        }

        double productError = Math.FusedMultiplyAdd(_numerator, x, -product);
        double displaced = total - product;
        double totalError = (product - (total - displaced)) + (_denominator - displaced) + productError;
        double selected = upper ? _denominator : product;
        double error = Math.FusedMultiplyAdd(-coordinate, total, selected);
        return ((error + (upper ? 0 : productError)) - coordinate * totalError) / total;
    }
}
