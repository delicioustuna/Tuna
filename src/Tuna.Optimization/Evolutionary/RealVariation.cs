using Tuna.RandomSource;

namespace Tuna.Optimization.Evolutionary;

internal static class RealVariation
{
    // Bounded Simulated Binary Crossover (SBX, Deb & Agrawal 1995).
    // Truncates each child's spread factor β to [lower, upper] independently via β_max = 1 + 2·margin / spread using the same uniform u.
    public static void SimulatedBinaryCrossover(ReadOnlySpan<double> parent1, ReadOnlySpan<double> parent2,
        Span<double> child1, Span<double> child2, ReadOnlySpan<double> lower, ReadOnlySpan<double> upper,
        double probability, double distributionIndex, Philox4x32Random random)
    {
        parent1.CopyTo(child1);
        parent2.CopyTo(child2);
        if (random.NextDouble() >= probability)
        {
            return;
        }

        Cross(parent1, parent2, child1, child2, lower, upper, distributionIndex + 1, random);
    }

    public static void Cross(ReadOnlySpan<double> parent1, ReadOnlySpan<double> parent2, Span<double> child1, Span<double> child2,
        ReadOnlySpan<double> lower, ReadOnlySpan<double> upper, double exponent, Philox4x32Random random)
    {
        for (int d = 0; d < child1.Length; d++)
        {
            if (random.NextDouble() >= 0.5)
            {
                continue;
            }

            double x1 = parent1[d], x2 = parent2[d];
            if (x1 == x2)
            {
                continue;
            }

            CrossGene(x1, x2, lower[d], upper[d], exponent, random, out child1[d], out child2[d]);
        }
    }

    public static void CrossGene(double x1, double x2, double lower, double upper, double exponent, Philox4x32Random random,
        out double child1, out double child2)
    {
        double y1 = Math.Min(x1, x2), y2 = Math.Max(x1, x2), spread = y2 - y1;
        double u = random.NextDouble();
        double lowerBeta = SpreadFactor(u, 1 + 2 * (y1 - lower) / spread, exponent);
        double upperBeta = SpreadFactor(u, 1 + 2 * (upper - y2) / spread, exponent);
        double middle = y1 + 0.5 * spread;
        double a = Math.Clamp(middle - 0.5 * lowerBeta * spread, lower, upper);
        double b = Math.Clamp(middle + 0.5 * upperBeta * spread, lower, upper);
        if (random.NextDouble() < 0.5)
        {
            (a, b) = (b, a);
        }

        child1 = a;
        child2 = b;
    }

    private static double SpreadFactor(double u, double maxBeta, double exponent)
    {
        double alpha = 2 - Math.Pow(maxBeta, -exponent);
        double t = u * alpha;
        return t <= 1 ? Math.Pow(t, 1 / exponent) : Math.Pow(2 - t, -1 / exponent);
    }

    // Bounded polynomial mutation (Deb & Goyal 1996): truncates left/right tails to [lower, upper] with probability 1/2 each.
    public static void PolynomialMutation(Span<double> x, ReadOnlySpan<double> lower, ReadOnlySpan<double> upper,
        double probability, double distributionIndex, Philox4x32Random random)
    {
        double exponent = distributionIndex + 1;
        for (int d = 0; d < x.Length; d++)
        {
            if (random.NextDouble() >= probability)
            {
                continue;
            }

            if (upper[d] - lower[d] == 0)
            {
                continue;
            }

            x[d] = MutateGene(x[d], lower[d], upper[d], exponent, random);
        }
    }

    public static double MutateGene(double y, double lower, double upper, double exponent, Philox4x32Random random)
    {
        double width = upper - lower;
        double u = random.NextDouble();
        double delta;
        if (u < 0.5)
        {
            double below = (y - lower) / width;
            double value = 2 * u + (1 - 2 * u) * Math.Pow(1 - below, exponent);
            delta = Math.Pow(value, 1 / exponent) - 1;
        }
        else
        {
            double above = (upper - y) / width;
            double value = 2 * (1 - u) + 2 * (u - 0.5) * Math.Pow(1 - above, exponent);
            delta = 1 - Math.Pow(value, 1 / exponent);
        }

        return Math.Clamp(y + delta * width, lower, upper);
    }

    public static int Tournament(ReadOnlySpan<int> rank, int size, Philox4x32Random random)
    {
        int winner = random.NextInt32(rank.Length);
        for (int k = 1; k < size; k++)
        {
            int challenger = random.NextInt32(rank.Length);
            if (rank[challenger] < rank[winner])
            {
                winner = challenger;
            }
        }

        return winner;
    }

    public static bool IsBetter(double a, double b) => !double.IsNaN(a) && (double.IsNaN(b) || a < b);

    public static int Compare(double a, int indexA, double b, int indexB)
    {
        if (IsBetter(a, b))
        {
            return -1;
        }

        if (IsBetter(b, a))
        {
            return 1;
        }

        return indexA.CompareTo(indexB);
    }
}
