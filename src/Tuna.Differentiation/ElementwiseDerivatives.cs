using System.Numerics;
using System.Runtime.CompilerServices;

namespace Tuna.Differentiation;

internal readonly struct LogDerivative<T> : IUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Invoke(T x) => T.IsZero(x) ? T.PositiveInfinity : x < T.Zero ? T.NaN : T.One / x;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Invoke(Vector<T> x)
    {
        var r = Vector.ConditionalSelect(Vector.Equals(x, Vector<T>.Zero), new Vector<T>(T.PositiveInfinity), Vector<T>.One / x);
        return Vector.ConditionalSelect(Vector.LessThan(x, Vector<T>.Zero), new Vector<T>(T.NaN), r);
    }
}

internal readonly struct SqrtDerivative<T> : IBinaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    private static T Half
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => T.One / (T.One + T.One);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Invoke(T x, T y) => T.IsZero(x) ? T.PositiveInfinity : Half / y;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Invoke(Vector<T> x, Vector<T> y)
        => Vector.ConditionalSelect(Vector.Equals(x, Vector<T>.Zero), new Vector<T>(T.PositiveInfinity), new Vector<T>(Half) / y);
}

internal readonly struct TwiceKernel<T> : IUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Invoke(T x) => x + x;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Invoke(Vector<T> x) => x + x;
}

internal readonly struct AbsDerivative<T> : IUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Invoke(T x) => T.IsNaN(x) ? T.NaN : x < T.Zero ? T.NegativeOne : T.One;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Invoke(Vector<T> x)
    {
        var r = Vector.ConditionalSelect(Vector.LessThan(x, Vector<T>.Zero), new Vector<T>(T.NegativeOne), Vector<T>.One);
        return Vector.ConditionalSelect(ElementwiseKernels<T>.IsNaN(x), new Vector<T>(T.NaN), r);
    }
}

internal readonly struct Log1pDerivative<T> : IUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Invoke(T x) => x < T.NegativeOne ? T.NaN : T.One / (T.One + x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Invoke(Vector<T> x)
        => Vector.ConditionalSelect(Vector.LessThan(x, new Vector<T>(T.NegativeOne)), new Vector<T>(T.NaN), Vector<T>.One / (Vector<T>.One + x));
}

internal readonly struct NegativeAbsKernel<T> : IUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Invoke(T x) => -T.Abs(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Invoke(Vector<T> x) => -Vector.Abs(x);
}

// Computes sigma(x)*sigma(-x) as t/(1 + t)^2 from t = exp(-|x|) to preserve positive-tail accuracy.
internal readonly struct LogisticDerivativeFromExp<T> : IUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Invoke(T t)
    {
        T denominator = T.One + t;
        return t / (denominator * denominator);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Invoke(Vector<T> t)
    {
        var denominator = Vector<T>.One + t;
        return t / (denominator * denominator);
    }
}

internal readonly struct LogitDerivative<T> : IUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Invoke(T p)
    {
        if (!(p >= T.Zero && p <= T.One))
        {
            return T.NaN;
        }

        return T.IsZero(p) || p == T.One ? T.PositiveInfinity : T.One / (p * (T.One - p));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Invoke(Vector<T> p)
    {
        var r = Vector<T>.One / (p * (Vector<T>.One - p));
        r = Vector.ConditionalSelect(Vector.Equals(p, Vector<T>.Zero) | Vector.Equals(p, Vector<T>.One), new Vector<T>(T.PositiveInfinity), r);
        var inside = Vector.GreaterThanOrEqual(p, Vector<T>.Zero) & Vector.LessThanOrEqual(p, Vector<T>.One);
        return Vector.ConditionalSelect(inside, r, new Vector<T>(T.NaN));
    }
}

// Uses t = exp(x) for x < -ln 2 and m = expm1(-x) otherwise to avoid cancellation and overflow.
internal readonly struct Log1mExpDerivative<T> : ITernaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    private static T NegativeLn2 => T.CreateTruncating(-0.6931471805599453);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Invoke(T x, T t, T m)
    {
        if (T.IsNaN(x) || x > T.Zero)
        {
            return T.NaN;
        }

        if (T.IsZero(x))
        {
            return T.NegativeInfinity;
        }

        return x < NegativeLn2 ? -t / (T.One - t) : T.NegativeOne / m;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Invoke(Vector<T> x, Vector<T> t, Vector<T> m)
    {
        var r = Vector.ConditionalSelect(Vector.LessThan(x, new Vector<T>(NegativeLn2)), -t / (Vector<T>.One - t), new Vector<T>(T.NegativeOne) / m);
        r = Vector.ConditionalSelect(Vector.Equals(x, Vector<T>.Zero), new Vector<T>(T.NegativeInfinity), r);
        return Vector.ConditionalSelect(Vector.GreaterThan(x, Vector<T>.Zero) | ElementwiseKernels<T>.IsNaN(x), new Vector<T>(T.NaN), r);
    }
}

internal readonly struct InverseKernel<T> : IUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Invoke(T x) => T.One / x;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Invoke(Vector<T> x) => Vector<T>.One / x;
}

// Computes -1/x^2 from y = 1/x as -(y*y) to avoid premature x^2 overflow.
internal readonly struct ReciprocalDerivative<T> : IUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Invoke(T y) => -(y * y);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Invoke(Vector<T> y) => -(y * y);
}

internal readonly struct MinLeftDerivative<T> : IBinaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Invoke(T x, T y) => T.IsNaN(x) || T.IsNaN(y) ? T.NaN : x <= y ? T.One : T.Zero;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Invoke(Vector<T> x, Vector<T> y)
        => SelectNaN(x, y, Vector.ConditionalSelect(Vector.LessThanOrEqual(x, y), Vector<T>.One, Vector<T>.Zero));

    internal static Vector<T> SelectNaN(Vector<T> x, Vector<T> y, Vector<T> r)
        => Vector.ConditionalSelect(ElementwiseKernels<T>.IsNaN(x) | ElementwiseKernels<T>.IsNaN(y), new Vector<T>(T.NaN), r);
}

internal readonly struct MinRightDerivative<T> : IBinaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Invoke(T x, T y) => T.IsNaN(x) || T.IsNaN(y) ? T.NaN : x <= y ? T.Zero : T.One;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Invoke(Vector<T> x, Vector<T> y)
        => MinLeftDerivative<T>.SelectNaN(x, y, Vector.ConditionalSelect(Vector.LessThanOrEqual(x, y), Vector<T>.Zero, Vector<T>.One));
}

internal readonly struct MaxLeftDerivative<T> : IBinaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Invoke(T x, T y) => T.IsNaN(x) || T.IsNaN(y) ? T.NaN : x >= y ? T.One : T.Zero;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Invoke(Vector<T> x, Vector<T> y)
        => MinLeftDerivative<T>.SelectNaN(x, y, Vector.ConditionalSelect(Vector.GreaterThanOrEqual(x, y), Vector<T>.One, Vector<T>.Zero));
}

internal readonly struct MaxRightDerivative<T> : IBinaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Invoke(T x, T y) => T.IsNaN(x) || T.IsNaN(y) ? T.NaN : x >= y ? T.Zero : T.One;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> Invoke(Vector<T> x, Vector<T> y)
        => MinLeftDerivative<T>.SelectNaN(x, y, Vector.ConditionalSelect(Vector.GreaterThanOrEqual(x, y), Vector<T>.Zero, Vector<T>.One));
}
