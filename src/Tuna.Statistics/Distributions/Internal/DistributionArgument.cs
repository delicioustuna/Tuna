namespace Tuna.Statistics.Distributions;

internal static class DistributionArgument
{
    internal static void Finite(double value, string name, bool forDifferentiation = false)
    {
        if (!double.IsFinite(value))
        {
            throw ParameterOutOfRange(forDifferentiation, name, value, "The value must be finite.");
        }
    }

    internal static void PositiveFinite(double value, string name, bool forDifferentiation = false)
    {
        if (!(value > 0 && value < double.PositiveInfinity))
        {
            throw ParameterOutOfRange(forDifferentiation, name, value, "The value must be positive and finite.");
        }
    }

    internal static void NonNegativeFinite(double value, string name, bool forDifferentiation = false)
    {
        if (!(value >= 0 && value < double.PositiveInfinity))
        {
            throw ParameterOutOfRange(forDifferentiation, name, value, "The value must be non-negative and finite.");
        }
    }

    internal static void PositiveOrInfinity(double value, string name, bool forDifferentiation = false)
    {
        if (!(value > 0))
        {
            throw ParameterOutOfRange(forDifferentiation, name, value, "The value must be positive or positive infinity.");
        }
    }

    internal static void Probability(double value, string name, bool forDifferentiation = false)
    {
        if (!(value >= 0 && value <= 1))
        {
            throw ParameterOutOfRange(forDifferentiation, name, value, "The probability must be in [0, 1].");
        }
    }

    internal static ArgumentOutOfRangeException ParameterOutOfRange(bool forDifferentiation, string name, object? value, string message) =>
        forDifferentiation ? new DistributionParameterOutOfRangeException(name, value, message) : new ArgumentOutOfRangeException(name, value, message);

    internal static void LogProbability(double value, string name)
    {
        if (!(value <= 0))
        {
            throw new ArgumentOutOfRangeException(name, value, "The log probability must be non-positive.");
        }
    }
}
