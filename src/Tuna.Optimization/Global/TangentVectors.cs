namespace Tuna.Optimization.Global;

// Immutable gradient/subgradient vector helpers for McCormick and GradientDual; null represents the zero vector (constant).
internal static class TangentVectors
{
    public static double[][] Units(int dimension)
    {
        var units = new double[dimension][];
        for (int i = 0; i < dimension; i++)
        {
            units[i] = new double[dimension];
            units[i][i] = 1;
        }

        return units;
    }

    public static double[]? Scale(double[]? s, double factor)
    {
        if (s is null || factor == 0)
        {
            return null;
        }

        if (factor == 1)
        {
            return s;
        }

        var r = new double[s.Length];
        for (int i = 0; i < r.Length; i++)
        {
            r[i] = factor * s[i];
        }

        return r;
    }

    public static double[]? Combine(double[]? a, double alpha, double[]? b, double beta)
    {
        if (a is null || alpha == 0)
        {
            return Scale(b, beta);
        }

        if (b is null || beta == 0)
        {
            return Scale(a, alpha);
        }

        if (a.Length != b.Length)
        {
            throw new ArgumentException("The operands were created with different numbers of variables.");
        }

        var r = new double[a.Length];
        for (int i = 0; i < r.Length; i++)
        {
            r[i] = alpha * a[i] + beta * b[i];
        }

        return r;
    }

    public static double Sign(double v) => v > 0 ? 1 : v < 0 ? -1 : 0;
}
