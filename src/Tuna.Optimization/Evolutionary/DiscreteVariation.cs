using Tuna.RandomSource;

namespace Tuna.Optimization.Evolutionary;

internal static class DiscreteVariation
{
    // Expand integer bounds by ±0.5 so boundary integers occupy the same unit rounding cell as interior integers.
    public static void CrossIntegers(ReadOnlySpan<long> parent1, ReadOnlySpan<long> parent2, Span<long> child1, Span<long> child2,
        ReadOnlySpan<long> lower, ReadOnlySpan<long> upper, double exponent, Philox4x32Random random)
    {
        for (int d = 0; d < child1.Length; d++)
        {
            if (random.NextDouble() >= 0.5)
            {
                continue;
            }

            long x1 = parent1[d], x2 = parent2[d];
            if (x1 == x2)
            {
                continue;
            }

            long lo = lower[d];
            ulong width = unchecked((ulong)(upper[d] - lo));
            RealVariation.CrossGene(Offset(x1, lo), Offset(x2, lo), -0.5, width + 0.5, exponent, random, out double a, out double b);
            child1[d] = FromOffset(a, lo, width);
            child2[d] = FromOffset(b, lo, width);
        }
    }

    public static void MutateIntegers(Span<long> x, ReadOnlySpan<long> lower, ReadOnlySpan<long> upper,
        double probability, double exponent, Philox4x32Random random)
    {
        for (int d = 0; d < x.Length; d++)
        {
            if (random.NextDouble() >= probability)
            {
                continue;
            }

            long lo = lower[d];
            ulong width = unchecked((ulong)(upper[d] - lo));
            if (width == 0)
            {
                continue;
            }

            x[d] = FromOffset(RealVariation.MutateGene(Offset(x[d], lo), -0.5, width + 0.5, exponent, random), lo, width);
        }
    }

    // Offset from lower is exact in double because width ≤ 2^53.
    private static double Offset(long x, long lower) => unchecked((ulong)(x - lower));

    private static long FromOffset(double z, long lower, ulong width)
    {
        double k = Math.Clamp(Math.Round(z), 0, width);
        return unchecked(lower + (long)(ulong)k);
    }

    public static void UniformCrossover<T>(Span<T> child1, Span<T> child2, Philox4x32Random random)
    {
        for (int d = 0; d < child1.Length; d++)
        {
            if (random.NextDouble() < 0.5)
            {
                (child1[d], child2[d]) = (child2[d], child1[d]);
            }
        }
    }

    public static void FlipBits(Span<bool> x, double probability, Philox4x32Random random)
    {
        for (int d = 0; d < x.Length; d++)
        {
            if (random.NextDouble() < probability)
            {
                x[d] = !x[d];
            }
        }
    }

    public static void MutateCategories(Span<int> x, ReadOnlySpan<int> counts, double probability, Philox4x32Random random)
    {
        for (int d = 0; d < x.Length; d++)
        {
            if (random.NextDouble() >= probability || counts[d] < 2)
            {
                continue;
            }

            // Pick uniformly among the remaining counts[d] - 1 categories (returns 0 without drawing when count == 2).
            int other = random.NextInt32(counts[d] - 1);
            x[d] = other >= x[d] ? other + 1 : other;
        }
    }

    // Order Crossover (OX, Davis 1985).
    public static void OrderCrossover(ReadOnlySpan<int> parent1, ReadOnlySpan<int> parent2, Span<int> child1, Span<int> child2,
        Philox4x32Random random, Span<bool> scratch)
    {
        int n = parent1.Length;
        int i = random.NextInt32(n), j = random.NextInt32(n);
        int start = Math.Min(i, j), end = Math.Max(i, j);
        OrderCrossover(parent1, parent2, child1, start, end, scratch);
        OrderCrossover(parent2, parent1, child2, start, end, scratch);
    }

    public static void OrderCrossover(ReadOnlySpan<int> donor, ReadOnlySpan<int> other, Span<int> child, int start, int end, Span<bool> scratch)
    {
        int n = donor.Length;
        Span<bool> used = scratch[..n];
        used.Clear();
        for (int k = start; k <= end; k++)
        {
            child[k] = donor[k];
            used[donor[k]] = true;
        }

        int position = end + 1 == n ? 0 : end + 1;
        int read = position;
        for (int t = 0; t < n; t++)
        {
            int value = other[read];
            read = read + 1 == n ? 0 : read + 1;
            if (used[value])
            {
                continue;
            }

            child[position] = value;
            position = position + 1 == n ? 0 : position + 1;
        }
    }

    public static void Invert(Span<int> x, double probability, Philox4x32Random random)
    {
        if (random.NextDouble() >= probability || x.Length < 2)
        {
            return;
        }

        int i = random.NextInt32(x.Length);
        int j = random.NextInt32(x.Length - 1);
        if (j >= i)
        {
            j++;
        }

        x[Math.Min(i, j)..(Math.Max(i, j) + 1)].Reverse();
    }
}
