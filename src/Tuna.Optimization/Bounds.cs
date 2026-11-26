namespace Tuna.Optimization;

internal static class Bounds
{
    // Validates and copies bounds; omitting one side defaults to ±∞, and omitting both when allowEmpty is true returns null.
    public static void Validate(ReadOnlySpan<double> lowerBounds, ReadOnlySpan<double> upperBounds, int dimension, bool allowEmpty,
        out double[]? lower, out double[]? upper)
    {
        if (lowerBounds.IsEmpty && upperBounds.IsEmpty && allowEmpty)
        {
            lower = null;
            upper = null;
            return;
        }

        if ((!lowerBounds.IsEmpty && lowerBounds.Length != dimension) || (!upperBounds.IsEmpty && upperBounds.Length != dimension))
        {
            throw new ArgumentException($"The bounds must have length {dimension}.", lowerBounds.Length != dimension ? nameof(lowerBounds) : nameof(upperBounds));
        }

        lower = new double[dimension];
        upper = new double[dimension];
        for (int i = 0; i < dimension; i++)
        {
            double lo = lowerBounds.IsEmpty ? double.NegativeInfinity : lowerBounds[i];
            double hi = upperBounds.IsEmpty ? double.PositiveInfinity : upperBounds[i];
            if (double.IsNaN(lo) || double.IsNaN(hi))
            {
                throw new ArgumentException($"Bound {i} must not be NaN.", double.IsNaN(lo) ? nameof(lowerBounds) : nameof(upperBounds));
            }

            if (lo > hi || lo == double.PositiveInfinity || hi == double.NegativeInfinity)
            {
                throw new ArgumentException($"Lower bound {i} must not exceed the upper bound, and the box must not be empty.", nameof(lowerBounds));
            }

            lower[i] = lo;
            upper[i] = hi;
        }
    }

    public static double Clamp(double x, double lo, double hi) => x < lo ? lo : x > hi ? hi : x;
}
