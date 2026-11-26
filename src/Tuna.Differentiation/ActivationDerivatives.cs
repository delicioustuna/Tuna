using System.Numerics;
using System.Runtime.CompilerServices;

namespace Tuna.Differentiation;

internal static class ActivationDerivatives
{
    // Exact integer shift (x + 64 has no rounding) and exp(-64) for keeping negative-tail exponentials normal until the final multiply.
    internal const double TailShift = 64;
    internal const double TailUnshift = 1.603810890548638e-28;

    internal const double SeluScale = 1.0507009873554804934193349852946;
    internal const double SeluScaleAlpha = 1.7580993408473768599402175208123;

    private const double InverseSqrt2 = 0.70710678118654752440084436210485;
    private const double InverseSqrt2Pi = 0.39894228040143267793994605993438;

    // Double-double representation of k = 2*sqrt(2/pi) and c = 0.044715 for z = k*(x + c*x^3).
    private const double GeluTanhScaleHigh = 1.5957691216057308;
    private const double GeluTanhScaleLow = -9.96930880911092e-17;
    private const double GeluTanhCubicHigh = 0.044715;
    private const double GeluTanhCubicLow = 2.1960211427085595e-18;

    // Exact GELU derivative Phi(x) + x*phi(x), splitting x^2 = hi + lo via FMA to avoid exponential argument rounding.
    internal static double Gelu(double x)
    {
        if (double.IsNaN(x))
        {
            return double.NaN;
        }

        if (x > 10)
        {
            return 1;
        }

        if (x < -39)
        {
            return -0.0;
        }

        double hi = x * x;
        double lo = Math.FusedMultiplyAdd(x, x, -hi);
        if (x >= 0)
        {
            double gaussian = Math.Exp(-0.5 * hi) * (1 - 0.5 * lo);
            return 0.5 * XMath.Special.Erfc(-x * InverseSqrt2) + x * gaussian * InverseSqrt2Pi;
        }

        double bracket = 0.5 * XMath.Special.Erfcx(-x * InverseSqrt2) + x * InverseSqrt2Pi;
        // Shift exponent by +64 below -37.5 so exp(-x^2/2) stays normal until the final product.
        return x < -37.5
            ? Math.Exp(TailShift - 0.5 * hi) * (1 - 0.5 * lo) * bracket * TailUnshift
            : Math.Exp(-0.5 * hi) * (1 - 0.5 * lo) * bracket;
    }

    // Derivative of tanh-approximated GELU x*sigma(z), with z = k*(x + c*x^3) evaluated in double-double to preserve exp(-|z|) accuracy.
    internal static double GeluTanh(double x)
    {
        if (double.IsNaN(x))
        {
            return double.NaN;
        }

        if (x > 10)
        {
            return 1;
        }

        if (x < -25)
        {
            return -0.0;
        }

        double square = x * x, squareLow = Math.FusedMultiplyAdd(x, x, -square);
        double cube = x * square, cubeLow = Math.FusedMultiplyAdd(x, square, -cube) + x * squareLow;
        double cubic = GeluTanhCubicHigh * cube;
        double cubicLow = Math.FusedMultiplyAdd(GeluTanhCubicHigh, cube, -cubic) + (GeluTanhCubicHigh * cubeLow + GeluTanhCubicLow * cube);
        double sum = x + cubic;
        double carry = sum - x;
        double sumLow = (x - (sum - carry)) + (cubic - carry) + cubicLow;
        double z = GeluTanhScaleHigh * sum;
        double zLow = Math.FusedMultiplyAdd(GeluTanhScaleHigh, sum, -z) + (GeluTanhScaleHigh * sumLow + GeluTanhScaleLow * sum);
        double w = x * GeluTanhScaleHigh * (1 + 3 * GeluTanhCubicHigh * square);
        if (z >= 0)
        {
            double e = Math.Exp(-z) * (1 - zLow);
            double onePlus = 1 + e;
            return (onePlus + w * e) / (onePlus * onePlus);
        }

        // For z < -700, 1 + exp(z) rounds to 1; shift by +64 to avoid premature subnormal underflow.
        if (z < -700)
        {
            return Math.Exp(z + TailShift) * (1 + zLow) * (1 + w) * TailUnshift;
        }

        double t = Math.Exp(z) * (1 + zLow);
        double denominator = 1 + t;
        return t * (denominator + w) / (denominator * denominator);
    }
}

// Negative-tail threshold where exp(x) approaches subnormal range (double: -708, float: -87).
internal static class ActivationTail<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    public static T Below
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => typeof(T) == typeof(float) ? T.CreateTruncating(-87f) : T.CreateTruncating(-708.0);
    }

    public static T Shift
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => T.CreateTruncating(ActivationDerivatives.TailShift);
    }

    public static T Unshift
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => typeof(T) == typeof(float) ? T.CreateTruncating(1.6038109e-28f) : T.CreateTruncating(ActivationDerivatives.TailUnshift);
    }
}

internal readonly struct ReluDerivative<T> : IUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Invoke(T x) => T.IsNaN(x) ? T.NaN : x > T.Zero ? T.One : T.Zero;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Invoke(Vector<T> x)
    {
        var r = Vector.ConditionalSelect(Vector.GreaterThan(x, Vector<T>.Zero), Vector<T>.One, Vector<T>.Zero);
        return Vector.ConditionalSelect(ElementwiseKernels<T>.IsNaN(x), new Vector<T>(T.NaN), r);
    }
}

internal readonly struct LeakyReluDerivative<T> : IBinaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Invoke(T x, T s) => T.IsNaN(x) ? T.NaN : x > T.Zero ? T.One : s;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Invoke(Vector<T> x, Vector<T> s)
    {
        var r = Vector.ConditionalSelect(Vector.GreaterThan(x, Vector<T>.Zero), Vector<T>.One, s);
        return Vector.ConditionalSelect(ElementwiseKernels<T>.IsNaN(x), new Vector<T>(T.NaN), r);
    }
}

// Shifts negative-tail exponent by an integer s (exact for |x| < 2^23; beyond that both exp(x) and exp(x + s) underflow to 0).
internal readonly struct ExponentialLinearArgument<T> : IBinaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Invoke(T x, T s) => x < ActivationTail<T>.Below ? x + s : x;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Invoke(Vector<T> x, Vector<T> s)
        => Vector.ConditionalSelect(Vector.LessThan(x, new Vector<T>(ActivationTail<T>.Below)), x + s, x);
}

// In the negative tail, c * t * exp(-Shift) recovers a * exp(x) without premature underflow.
internal readonly struct ExponentialLinearDerivative<T> : IParameterizedKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Invoke(T x, T t, T a, T c, T p)
    {
        if (x > T.Zero)
        {
            return p;
        }

        return x < ActivationTail<T>.Below ? c * t * ActivationTail<T>.Unshift : a * t;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Invoke(Vector<T> x, Vector<T> t, Vector<T> a, Vector<T> c, Vector<T> p)
    {
        var r = Vector.ConditionalSelect(Vector.LessThan(x, new Vector<T>(ActivationTail<T>.Below)), c * t * new Vector<T>(ActivationTail<T>.Unshift), a * t);
        return Vector.ConditionalSelect(Vector.GreaterThan(x, Vector<T>.Zero), p, r);
    }
}

internal readonly struct SiluArgument<T> : IUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Invoke(T x) => x < ActivationTail<T>.Below ? x + ActivationTail<T>.Shift : -T.Abs(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Invoke(Vector<T> x)
        => Vector.ConditionalSelect(Vector.LessThan(x, new Vector<T>(ActivationTail<T>.Below)), x + new Vector<T>(ActivationTail<T>.Shift), -Vector.Abs(x));
}

// Uses t = exp(-|x|) (or exp(x + Shift) in the negative tail). Grouping (1 + x) + t keeps 1 + x exact on [-2, -0.5] near the derivative root x ≈ -1.2785.
internal readonly struct SiluDerivative<T> : IBinaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Invoke(T x, T t)
    {
        if (x < ActivationTail<T>.Below)
        {
            return T.IsZero(t) ? T.Zero : (T.One + x) * t * ActivationTail<T>.Unshift;
        }

        T onePlus = T.One + t;
        T denominator = onePlus * onePlus;
        if (x >= T.Zero)
        {
            T product = T.IsZero(t) ? T.Zero : x * t;
            return (onePlus + product) / denominator;
        }

        return t * ((T.One + x) + t) / denominator;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Invoke(Vector<T> x, Vector<T> t)
    {
        var zero = Vector.Equals(t, Vector<T>.Zero);
        var onePlus = Vector<T>.One + t;
        var denominator = onePlus * onePlus;
        var positive = (onePlus + Vector.ConditionalSelect(zero, Vector<T>.Zero, x * t)) / denominator;
        var negative = t * ((Vector<T>.One + x) + t) / denominator;
        var tail = Vector.ConditionalSelect(zero, Vector<T>.Zero, (Vector<T>.One + x) * t * new Vector<T>(ActivationTail<T>.Unshift));
        var r = Vector.ConditionalSelect(Vector.GreaterThanOrEqual(x, Vector<T>.Zero), positive, negative);
        return Vector.ConditionalSelect(Vector.LessThan(x, new Vector<T>(ActivationTail<T>.Below)), tail, r);
    }
}
