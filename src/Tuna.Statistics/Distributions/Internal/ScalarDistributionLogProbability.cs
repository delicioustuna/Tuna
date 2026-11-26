using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Statistics.Distributions;

internal static class ScalarDistributionLogProbability
{
    internal static ScalarNode Record<T>(T distribution, ScalarNode first, ScalarNode second, ScalarNode x, LogProbability kind,
        Func<T, double, double, LogProbability, (double First, double Second, double X)> gradient)
        where T : struct, IContinuousDistribution
    {
        double observation = x.Value;
        double value = kind switch
        {
            LogProbability.Density => distribution.LogPdf(observation),
            LogProbability.LowerTail => distribution.LogCdf(observation),
            _ => distribution.LogSf(observation),
        };
        if (!first.RequiresGradient && !second.RequiresGradient && !x.RequiresGradient)
        {
            return value;
        }

        var (da, db, dx) = gradient(distribution, observation, value, kind);
        return ScalarNode.Record(value, [(first, da), (second, db), (x, dx)]);
    }

    internal static ScalarNode RecordSum<T>(T distribution, ScalarNode first, ScalarNode second, ReadOnlyVectorSpan<double> observations,
        Func<T, double, double, LogProbability, (double First, double Second, double X)> gradient)
        where T : struct, IContinuousDistribution
    {
        bool requiresGradient = first.RequiresGradient || second.RequiresGradient;
        double value = 0;
        var firstSum = new ScaledDerivativeSum();
        var secondSum = new ScaledDerivativeSum();
        for (int i = 0; i < observations.Length; i++)
        {
            double x = observations[i];
            double density = distribution.LogPdf(x);
            value += density;
            if (requiresGradient)
            {
                var (da, db, _) = gradient(distribution, x, density, LogProbability.Density);
                firstSum.Add(new(da));
                secondSum.Add(new(db));
            }
        }

        return requiresGradient ? ScalarNode.Record(value, [(first, firstSum.Value), (second, secondSum.Value)]) : value;
    }

    internal static ScalarNode RecordMass<T>(T distribution, ScalarNode first, ScalarNode second, long k, LogProbability kind,
        Func<T, long, double, LogProbability, (double First, double Second)> gradient)
        where T : struct, IDiscreteDistribution
    {
        double value = kind switch
        {
            LogProbability.Density => distribution.LogPmf(k),
            LogProbability.LowerTail => distribution.LogCdf(k),
            _ => distribution.LogSf(k),
        };
        if (!first.RequiresGradient && !second.RequiresGradient)
        {
            return value;
        }

        var (da, db) = gradient(distribution, k, value, kind);
        return ScalarNode.Record(value, [(first, da), (second, db)]);
    }

    internal static ScalarNode RecordMassSum<T>(T distribution, ScalarNode first, ScalarNode second, ReadOnlyVectorSpan<long> observations,
        Func<T, long, double, LogProbability, (double First, double Second)> gradient)
        where T : struct, IDiscreteDistribution
    {
        bool requiresGradient = first.RequiresGradient || second.RequiresGradient;
        double value = 0;
        var firstSum = new ScaledDerivativeSum();
        var secondSum = new ScaledDerivativeSum();
        for (int i = 0; i < observations.Length; i++)
        {
            long k = observations[i];
            double mass = distribution.LogPmf(k);
            value += mass;
            if (requiresGradient)
            {
                var (da, db) = gradient(distribution, k, mass, LogProbability.Density);
                firstSum.Add(new(da));
                secondSum.Add(new(db));
            }
        }

        return requiresGradient ? ScalarNode.Record(value, [(first, firstSum.Value), (second, secondSum.Value)]) : value;
    }

    internal static (double Location, double Scale, double X) LocationScaleDensity(double z, double scale, double slope)
    {
        double dx = slope / scale;
        double numerator = slope == 0 ? 1 : Math.FusedMultiplyAdd(z, slope, 1);
        // ガンベル分布の左裾では z と傾きの積だけがオーバーフローすることがある。
        double ds = double.IsInfinity(numerator) && double.IsFinite(z) && double.IsFinite(slope)
            ? -DistributionMath.ScaledRatio([z, slope], [scale]) : -numerator / scale;
        return (-dx, ds, dx);
    }

    internal static (double Location, double Scale, double X) LocationScaleTail(double z, double scale,
        (double Hi, double Lo) exponent, bool negative, double numerator = 1, double denominator = 1)
    {
        double sign = negative ? -1 : 1;
        double dx = sign * DistributionMath.ExponentialRatio(exponent, [numerator], [denominator, scale]);
        double ds = z == 0 || double.IsNegativeInfinity(exponent.Hi) ? 0
            : double.IsNormal(dx) || !double.IsFinite(z) || double.IsPositiveInfinity(exponent.Hi) ? -z * dx
            : -sign * DistributionMath.ExponentialRatio(exponent, [numerator, z], [denominator, scale]);
        return (-dx, ds, dx);
    }
}
