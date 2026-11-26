namespace Tuna.Statistics.Distributions;

internal readonly struct ScaledDerivative(double mantissa, int exponent = 0)
{
    internal readonly double Mantissa = mantissa;
    internal readonly int Exponent = exponent;

    internal double Value => Exponent == 0 ? Mantissa : Math.ScaleB(Mantissa, Exponent);
}
