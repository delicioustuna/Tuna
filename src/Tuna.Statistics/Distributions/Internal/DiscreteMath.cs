namespace Tuna.Statistics.Distributions;

internal static class DiscreteMath
{
    // For p > 0.5, test Sf(k) <= 1 - p (which is exact) to preserve relative accuracy in the upper tail.
    internal static long InverseCdf<TDistribution>(in TDistribution distribution, double p, long guess)
        where TDistribution : IDiscreteDistribution
    {
        DistributionArgument.Probability(p, nameof(p));
        TDistribution d = distribution;
        if (p == 0)
        {
            return d.SupportMinimum;
        }

        if (p == 1)
        {
            return d.SupportMaximum;
        }

        if (p <= 0.5)
        {
            return DistributionMath.SmallestSatisfying(k => d.Cdf(k) >= p, guess, d.SupportMinimum, d.SupportMaximum);
        }

        double q = 1 - p;
        return DistributionMath.SmallestSatisfying(k => d.Sf(k) <= q, guess, d.SupportMinimum, d.SupportMaximum);
    }

    internal static long InverseSf<TDistribution>(in TDistribution distribution, double q, long guess)
        where TDistribution : IDiscreteDistribution
    {
        DistributionArgument.Probability(q, nameof(q));
        TDistribution d = distribution;
        if (q == 1)
        {
            return d.SupportMinimum;
        }

        if (q == 0)
        {
            return d.SupportMaximum;
        }

        if (q < 0.5)
        {
            return DistributionMath.SmallestSatisfying(k => d.Sf(k) <= q, guess, d.SupportMinimum, d.SupportMaximum);
        }

        double p = 1 - q;
        return DistributionMath.SmallestSatisfying(k => d.Cdf(k) >= p, guess, d.SupportMinimum, d.SupportMaximum);
    }

    internal static long InverseLogCdf<TDistribution>(in TDistribution distribution, double logP, long guess)
        where TDistribution : IDiscreteDistribution
    {
        DistributionArgument.LogProbability(logP, nameof(logP));
        TDistribution d = distribution;
        if (double.IsNegativeInfinity(logP))
        {
            return d.SupportMinimum;
        }

        if (logP == 0)
        {
            return d.SupportMaximum;
        }

        if (logP <= -XMath.Constants.Ln2)
        {
            return DistributionMath.SmallestSatisfying(k => d.LogCdf(k) >= logP, guess, d.SupportMinimum, d.SupportMaximum);
        }

        double q = -DistributionMath.Expm1(logP);
        return DistributionMath.SmallestSatisfying(k => d.Sf(k) <= q, guess, d.SupportMinimum, d.SupportMaximum);
    }

    internal static long InverseLogSf<TDistribution>(in TDistribution distribution, double logQ, long guess)
        where TDistribution : IDiscreteDistribution
    {
        DistributionArgument.LogProbability(logQ, nameof(logQ));
        TDistribution d = distribution;
        if (logQ == 0)
        {
            return d.SupportMinimum;
        }

        if (double.IsNegativeInfinity(logQ))
        {
            return d.SupportMaximum;
        }

        if (logQ <= -XMath.Constants.Ln2)
        {
            return DistributionMath.SmallestSatisfying(k => d.LogSf(k) <= logQ, guess, d.SupportMinimum, d.SupportMaximum);
        }

        double p = -DistributionMath.Expm1(logQ);
        return DistributionMath.SmallestSatisfying(k => d.Cdf(k) >= p, guess, d.SupportMinimum, d.SupportMaximum);
    }

    // Second-order Cornish–Fisher initial guess clamped to [minimum, maximum].
    internal static long Guess(double mean, double standardDeviation, double skewness, double z, long minimum, long maximum)
    {
        if (!double.IsFinite(z))
        {
            return z > 0 ? maximum : minimum;
        }

        double value = mean + standardDeviation * (z + (z * z - 1) * skewness / 6);
        if (!(value > minimum))
        {
            return minimum;
        }

        if (!(value < maximum))
        {
            return maximum;
        }

        return (long)Math.Floor(value);
    }

    // Second-order Edgeworth negentropy expansion for near-Gaussian lattice distributions; omitted terms are O(max(γ₁², |γ₂|)³).
    // Takes logVariance so distributions whose variance overflows still work.
    internal static double EntropyExpansion(double logVariance, double skewness, double excessKurtosis)
    {
        double s2 = skewness * skewness;
        double negentropy = s2 / 12 + excessKurtosis * excessKurtosis / 48 + 7 * s2 * s2 / 48 - s2 * excessKurtosis / 8;
        return 0.5 * (NumericConstants.LnTau + 1 + logVariance) - negentropy;
    }

    // Threshold on max(γ₁², |γ₂|) below which omitted Edgeworth terms fall under double rounding error.
    internal const double EntropyExpansionLimit = 1e-5;

    // Sums -p_k ln p_k outward from the mode with Neumaier compensation, stopping each tail once the geometric tail bound <= ε/4 * |sum|.
    internal static double EntropySum(Func<long, double> logProbability, long mode, long minimum, long maximum)
    {
        double sum = 0;
        double compensation = 0;
        void Add(double term)
        {
            double t = sum + term;
            compensation += Math.Abs(sum) >= Math.Abs(term) ? sum - t + term : term - t + sum;
            sum = t;
        }

        double previous = double.NaN;
        for (long k = mode; k <= maximum; k++)
        {
            double logP = logProbability(k);
            double term = logP == double.NegativeInfinity ? 0 : -Math.Exp(logP) * logP;
            Add(term);
            if (k > mode + 1 && Converged(term, previous, sum))
            {
                break;
            }

            previous = term;
            if (k == long.MaxValue)
            {
                break;
            }
        }

        previous = double.NaN;
        for (long k = mode - 1; k >= minimum; k--)
        {
            double logP = logProbability(k);
            double term = logP == double.NegativeInfinity ? 0 : -Math.Exp(logP) * logP;
            Add(term);
            if (k < mode - 2 && Converged(term, previous, sum))
            {
                break;
            }

            previous = term;
        }

        return sum + compensation;
    }

    private static bool Converged(double term, double previous, double sum)
    {
        if (term == 0)
        {
            return true;
        }

        double ratio = term / previous;
        if (!(ratio < 1))
        {
            return false;
        }

        // Geometric series tail bound assuming non-increasing ratio.
        return term * ratio / (1 - ratio) <= 0.25 * NumericConstants.MachineEpsilon * Math.Abs(sum);
    }
}
