using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

internal enum LogProbability
{
    Density,
    LowerTail,
    UpperTail,
}

internal static class GammaLogProbability
{
    internal static LikelihoodEvaluation Density<TDistribution>(TDistribution distribution, double shape, double scale,
        double observation, LikelihoodGradients gradients, bool inverse, bool halfShape = false)
        where TDistribution : struct, IContinuousDistribution
    {
        double value = distribution.LogPdf(observation);
        if (gradients == LikelihoodGradients.None)
        {
            return new(value);
        }

        double factor = halfShape ? 0.5 : 1;
        var (dShape, dScale, _) = Derivatives(shape * factor, scale, observation, value, LogProbability.Density, inverse,
            (gradients & LikelihoodGradients.First) != 0);
        return new(value, new(factor * dShape), new(dScale));
    }

    internal static ScalarNode Record<TDistribution>(
        TDistribution distribution, ScalarNode shape, ScalarNode scale, ScalarNode x, LogProbability kind, bool inverse, bool halfShape = false)
        where TDistribution : struct, IContinuousDistribution
    {
        double observation = x.Value;
        double value = kind switch
        {
            LogProbability.Density => distribution.LogPdf(observation),
            LogProbability.LowerTail => distribution.LogCdf(observation),
            _ => distribution.LogSf(observation),
        };
        bool shapeGradient = shape.RequiresGradient;
        if (!shapeGradient && !scale.RequiresGradient && !x.RequiresGradient)
        {
            return value;
        }

        double shapeFactor = halfShape ? 0.5 : 1;
        var (dShape, dScale, dX) = Derivatives(shape.Value * shapeFactor, scale.Value, observation, value, kind, inverse, shapeGradient);
        return ScalarNode.Record(value, [(shape, shapeFactor * dShape), (scale, dScale), (x, dX)]);
    }

    internal static ScalarNode RecordSum<TDistribution>(
        TDistribution distribution, ScalarNode shape, ScalarNode scale, ReadOnlyVectorSpan<double> observations, bool inverse, bool halfShape = false)
        where TDistribution : struct, IContinuousDistribution
    {
        double shapeFactor = halfShape ? 0.5 : 1;
        double a = shape.Value * shapeFactor, s = scale.Value;
        bool shapeGradient = shape.RequiresGradient, scaleGradient = scale.RequiresGradient;
        double value = 0;
        var dShape = new ScaledDerivativeSum();
        var dScale = new ScaledDerivativeSum();
        for (int i = 0; i < observations.Length; i++)
        {
            double x = observations[i];
            double density = distribution.LogPdf(x);
            value += density;
            if (shapeGradient || scaleGradient)
            {
                var (da, ds, _) = Derivatives(a, s, x, density, LogProbability.Density, inverse, shapeGradient);
                dShape.Add(new(da));
                dScale.Add(new(ds));
            }
        }

        return shapeGradient || scaleGradient
            ? ScalarNode.Record(value, [(shape, shapeFactor * dShape.Value), (scale, dScale.Value)])
            : value;
    }

    private static (double Shape, double Scale, double X) Derivatives(
        double shape, double scale, double x, double value, LogProbability kind, bool inverse, bool shapeGradient)
    {
        if (double.IsNaN(x))
        {
            return (double.NaN, double.NaN, double.NaN);
        }

        bool upper = inverse ? kind == LogProbability.LowerTail : kind == LogProbability.UpperTail;
        if (!(x > 0) || double.IsPositiveInfinity(x))
        {
            return AtEndpoint(shape, scale, x, kind, inverse, upper);
        }

        double numerator = inverse ? scale : x;
        double denominator = inverse ? x : scale;
        double y = numerator / denominator;
        double logY = y >= NumericConstants.MinNormal && double.IsFinite(y)
            ? Math.Log(y) : Math.Log(numerator) - Math.Log(denominator);

        if (kind == LogProbability.Density)
        {
            // 差を取る前に両方を同じ 2 の冪で拡大し、非正規化数の丸めで残差が失われるのを防ぐ。
            if (Math.Min(numerator, denominator) < NumericConstants.MinNormal && Math.Max(numerator, denominator) < 1e292)
            {
                numerator = Math.ScaleB(numerator, 52);
                denominator = Math.ScaleB(denominator, 52);
            }

            double dShape = 0;
            if (shapeGradient)
            {
                dShape = y >= NumericConstants.MinNormal && double.IsFinite(y)
                    ? XMath.Special.GammaShapePrefixDerivative(shape, y, lower: false)
                        + DistributionMath.Log1p(Math.FusedMultiplyAdd(-y, denominator, numerator) / numerator)
                    : logY - XMath.Special.Digamma(shape);
            }

            double difference = Math.FusedMultiplyAdd(-shape, denominator, numerator);
            double centered = double.IsFinite(difference) ? difference / denominator : y - shape;
            double dScale = double.IsFinite(centered) ? centered / scale
                : DistributionMath.ScaledRatio([difference], [denominator, scale]);
            double dX = double.IsFinite(centered) ? (inverse ? centered - 1 : -centered - 1) / x
                : Math.CopySign(DistributionMath.ScaledRatio([difference], [denominator, x]), inverse ? 1 : -1);
            if (!inverse && shape is >= 0.5 and <= 2 && double.IsFinite(y))
            {
                // a − 1 はこの範囲で正確。先に引くことで y が小さい場合の評価点に関する勾配を保つ。
                if (shape == 1)
                {
                    dX = -1 / scale;
                }
                else
                {
                    double slope = Math.FusedMultiplyAdd(shape - 1, denominator, -numerator);
                    if (double.IsFinite(slope))
                    {
                        dX = DistributionMath.ScaledRatio([slope], [denominator, x]);
                    }
                }
            }

            return (dShape, inverse ? -dScale : dScale, dX);
        }

        double dTailShape = 0, logSlope;
        if (y < NumericConstants.MinNormal)
        {
            // 下側は P(a,y) ≈ yᵃ / Γ(a+1)。y を丸めず、対数の比から微分を求める。
            double logP = DistributionMath.LogRegularizedGammaPFromLog(shape, logY);
            double lowerShape = logY - XMath.Special.Digamma(shape + 1);
            double logTailRatio = upper ? logP - value : 0;
            logSlope = Math.Log(shape) + logTailRatio;
            dTailShape = upper ? Math.Exp(logTailRatio + Math.Log(-lowerShape)) : lowerShape;
        }
        else if (double.IsPositiveInfinity(y))
        {
            double fraction = Math.FusedMultiplyAdd(-shape, denominator, numerator) / numerator;
            logSlope = upper ? logY + Math.Log(fraction) : double.NegativeInfinity;
            dTailShape = upper ? logY - XMath.Special.Digamma(shape) : -0.0;
        }
        else
        {
            logSlope = XMath.Special.LogGammaTailPrefixRatio(shape, y, value, upper);
            if (shapeGradient)
            {
                dTailShape = upper ? XMath.Special.LogRegularizedGammaQShapeDerivative(shape, y)
                    : XMath.Special.LogRegularizedGammaPShapeDerivative(shape, y);
            }
        }

        double sign = upper == inverse ? 1 : -1;
        return (dTailShape, -sign * Math.Exp(logSlope - Math.Log(scale)), sign * Math.Exp(logSlope - Math.Log(x)));
    }

    private static (double Shape, double Scale, double X) AtEndpoint(double shape, double scale, double x,
        LogProbability kind, bool inverse, bool upper)
    {
        if (x < 0)
        {
            return default;
        }

        if (kind == LogProbability.Density)
        {
            if (x == 0)
            {
                return inverse ? (double.PositiveInfinity, double.NegativeInfinity, double.PositiveInfinity)
                    : (double.NegativeInfinity, -shape / scale, shape < 1 ? double.NegativeInfinity
                        : shape > 1 ? double.PositiveInfinity : -1 / scale);
            }

            return inverse ? (double.NegativeInfinity, shape / scale, -0.0)
                : (double.PositiveInfinity, double.PositiveInfinity, -1 / scale);
        }

        if (x == 0)
        {
            if (inverse)
            {
                return upper ? (double.PositiveInfinity, double.NegativeInfinity, double.PositiveInfinity) : default;
            }

            if (!upper)
            {
                return (double.NegativeInfinity, -shape / scale, double.PositiveInfinity);
            }

            return (0, 0, shape < 1 ? double.NegativeInfinity : shape == 1 ? -1 / scale : -0.0);
        }

        if (inverse)
        {
            return upper ? default : (double.NegativeInfinity, shape / scale, -0.0);
        }

        return upper ? (double.PositiveInfinity, double.PositiveInfinity, -1 / scale) : default;
    }
}
