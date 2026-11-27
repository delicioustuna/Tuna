using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Tuna.Differentiation;

internal static class ElementwiseKernels<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    // Skips elements where dy or d is exact 0 so 0 * ∞ and 0 * NaN do not corrupt gradients (matching ScalarNode).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void MaskedProduct(ReadOnlySpan<T> dy, ReadOnlySpan<T> d, Span<T> g, bool overwrite)
    {
        int n = g.Length;
        dy = dy[..n];
        d = d[..n];
        int i = 0;
        if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported)
        {
            ref T pdy = ref MemoryMarshal.GetReference(dy);
            ref T pd = ref MemoryMarshal.GetReference(d);
            ref T pg = ref MemoryMarshal.GetReference(g);
            int last = n - Vector<T>.Count;
            if (overwrite)
            {
                for (; i <= last; i += Vector<T>.Count)
                {
                    var a = Vector.LoadUnsafe(ref pdy, (nuint)i);
                    var b = Vector.LoadUnsafe(ref pd, (nuint)i);
                    var skip = Vector.Equals(a, Vector<T>.Zero) | Vector.Equals(b, Vector<T>.Zero);
                    Vector.ConditionalSelect(skip, Vector<T>.Zero, a * b).StoreUnsafe(ref pg, (nuint)i);
                }
            }
            else
            {
                for (; i <= last; i += Vector<T>.Count)
                {
                    var a = Vector.LoadUnsafe(ref pdy, (nuint)i);
                    var b = Vector.LoadUnsafe(ref pd, (nuint)i);
                    var old = Vector.LoadUnsafe(ref pg, (nuint)i);
                    var skip = Vector.Equals(a, Vector<T>.Zero) | Vector.Equals(b, Vector<T>.Zero);
                    Vector.ConditionalSelect(skip, old, old + a * b).StoreUnsafe(ref pg, (nuint)i);
                }
            }
        }

        for (; i < n; i++)
        {
            T a = dy[i], b = d[i];
            if (T.IsZero(a) || T.IsZero(b))
            {
                if (overwrite)
                {
                    g[i] = T.Zero;
                }
            }
            else
            {
                g[i] = overwrite ? a * b : g[i] + a * b;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void MaskedScale(ReadOnlySpan<T> dy, T d, Span<T> g, bool overwrite)
    {
        if (T.IsZero(d))
        {
            if (overwrite)
            {
                g.Clear();
            }

            return;
        }

        int n = g.Length;
        dy = dy[..n];
        int i = 0;
        if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported)
        {
            ref T pdy = ref MemoryMarshal.GetReference(dy);
            ref T pg = ref MemoryMarshal.GetReference(g);
            var b = new Vector<T>(d);
            int last = n - Vector<T>.Count;
            // Write a * b directly (not 0 + a * b) on overwrite so -0 underflow signs match the scalar tail.
            if (overwrite)
            {
                for (; i <= last; i += Vector<T>.Count)
                {
                    var a = Vector.LoadUnsafe(ref pdy, (nuint)i);
                    Vector.ConditionalSelect(Vector.Equals(a, Vector<T>.Zero), Vector<T>.Zero, a * b).StoreUnsafe(ref pg, (nuint)i);
                }
            }
            else
            {
                for (; i <= last; i += Vector<T>.Count)
                {
                    var a = Vector.LoadUnsafe(ref pdy, (nuint)i);
                    var old = Vector.LoadUnsafe(ref pg, (nuint)i);
                    Vector.ConditionalSelect(Vector.Equals(a, Vector<T>.Zero), old, old + a * b).StoreUnsafe(ref pg, (nuint)i);
                }
            }
        }

        for (; i < n; i++)
        {
            T a = dy[i];
            if (T.IsZero(a))
            {
                if (overwrite)
                {
                    g[i] = T.Zero;
                }
            }
            else
            {
                g[i] = overwrite ? a * d : g[i] + a * d;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void AddScaled(ReadOnlySpan<T> x, T s, Span<T> g, bool overwrite)
    {
        int n = g.Length;
        x = x[..n];
        int i = 0;
        if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported)
        {
            ref T px = ref MemoryMarshal.GetReference(x);
            ref T pg = ref MemoryMarshal.GetReference(g);
            var vs = new Vector<T>(s);
            int last = n - Vector<T>.Count;
            if (overwrite)
            {
                for (; i <= last; i += Vector<T>.Count)
                {
                    (Vector.LoadUnsafe(ref px, (nuint)i) * vs).StoreUnsafe(ref pg, (nuint)i);
                }
            }
            else
            {
                for (; i <= last; i += Vector<T>.Count)
                {
                    (Vector.LoadUnsafe(ref pg, (nuint)i) + Vector.LoadUnsafe(ref px, (nuint)i) * vs).StoreUnsafe(ref pg, (nuint)i);
                }
            }
        }

        for (; i < n; i++)
        {
            g[i] = overwrite ? x[i] * s : g[i] + x[i] * s;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Map<TKernel>(ReadOnlySpan<T> x, Span<T> d) where TKernel : IUnaryKernel<T>
    {
        int n = d.Length;
        x = x[..n];
        int i = 0;
        if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported)
        {
            ref T px = ref MemoryMarshal.GetReference(x);
            ref T pd = ref MemoryMarshal.GetReference(d);
            int last = n - Vector<T>.Count;
            for (; i <= last; i += Vector<T>.Count)
            {
                TKernel.Invoke(Vector.LoadUnsafe(ref px, (nuint)i)).StoreUnsafe(ref pd, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            d[i] = TKernel.Invoke(x[i]);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Map<TKernel>(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> d) where TKernel : IBinaryKernel<T>
    {
        int n = d.Length;
        x = x[..n];
        y = y[..n];
        int i = 0;
        if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported)
        {
            ref T px = ref MemoryMarshal.GetReference(x);
            ref T py = ref MemoryMarshal.GetReference(y);
            ref T pd = ref MemoryMarshal.GetReference(d);
            int last = n - Vector<T>.Count;
            for (; i <= last; i += Vector<T>.Count)
            {
                TKernel.Invoke(Vector.LoadUnsafe(ref px, (nuint)i), Vector.LoadUnsafe(ref py, (nuint)i)).StoreUnsafe(ref pd, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            d[i] = TKernel.Invoke(x[i], y[i]);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Map<TKernel>(ReadOnlySpan<T> x, T s, Span<T> d) where TKernel : IBinaryKernel<T>
    {
        int n = d.Length;
        x = x[..n];
        int i = 0;
        if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported)
        {
            ref T px = ref MemoryMarshal.GetReference(x);
            ref T pd = ref MemoryMarshal.GetReference(d);
            var vs = new Vector<T>(s);
            int last = n - Vector<T>.Count;
            for (; i <= last; i += Vector<T>.Count)
            {
                TKernel.Invoke(Vector.LoadUnsafe(ref px, (nuint)i), vs).StoreUnsafe(ref pd, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            d[i] = TKernel.Invoke(x[i], s);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Map<TKernel>(ReadOnlySpan<T> x, ReadOnlySpan<T> y, ReadOnlySpan<T> z, Span<T> d) where TKernel : ITernaryKernel<T>
    {
        int n = d.Length;
        x = x[..n];
        y = y[..n];
        z = z[..n];
        int i = 0;
        if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported)
        {
            ref T px = ref MemoryMarshal.GetReference(x);
            ref T py = ref MemoryMarshal.GetReference(y);
            ref T pz = ref MemoryMarshal.GetReference(z);
            ref T pd = ref MemoryMarshal.GetReference(d);
            int last = n - Vector<T>.Count;
            for (; i <= last; i += Vector<T>.Count)
            {
                TKernel.Invoke(Vector.LoadUnsafe(ref px, (nuint)i), Vector.LoadUnsafe(ref py, (nuint)i), Vector.LoadUnsafe(ref pz, (nuint)i))
                    .StoreUnsafe(ref pd, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            d[i] = TKernel.Invoke(x[i], y[i], z[i]);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Map<TKernel>(ReadOnlySpan<T> x, T s, ReadOnlySpan<T> z, Span<T> d) where TKernel : ITernaryKernel<T>
    {
        int n = d.Length;
        x = x[..n];
        z = z[..n];
        int i = 0;
        if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported)
        {
            ref T px = ref MemoryMarshal.GetReference(x);
            ref T pz = ref MemoryMarshal.GetReference(z);
            ref T pd = ref MemoryMarshal.GetReference(d);
            var vs = new Vector<T>(s);
            int last = n - Vector<T>.Count;
            for (; i <= last; i += Vector<T>.Count)
            {
                TKernel.Invoke(Vector.LoadUnsafe(ref px, (nuint)i), vs, Vector.LoadUnsafe(ref pz, (nuint)i))
                    .StoreUnsafe(ref pd, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            d[i] = TKernel.Invoke(x[i], s, z[i]);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Map<TKernel>(ReadOnlySpan<T> x, ReadOnlySpan<T> y, T a, T b, T c, Span<T> d) where TKernel : IParameterizedKernel<T>
    {
        int n = d.Length;
        x = x[..n];
        y = y[..n];
        int i = 0;
        if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported)
        {
            ref T px = ref MemoryMarshal.GetReference(x);
            ref T py = ref MemoryMarshal.GetReference(y);
            ref T pd = ref MemoryMarshal.GetReference(d);
            var va = new Vector<T>(a);
            var vb = new Vector<T>(b);
            var vc = new Vector<T>(c);
            int last = n - Vector<T>.Count;
            for (; i <= last; i += Vector<T>.Count)
            {
                TKernel.Invoke(Vector.LoadUnsafe(ref px, (nuint)i), Vector.LoadUnsafe(ref py, (nuint)i), va, vb, vc).StoreUnsafe(ref pd, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            d[i] = TKernel.Invoke(x[i], y[i], a, b, c);
        }
    }

    // Computes -x/y^2 as -(z/y) where z = x/y, falling back to exponent-separated scaling when z is subnormal.
    // A length-1 y span broadcasts the scalar divisor.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void DenominatorDerivatives(ReadOnlySpan<T> x, ReadOnlySpan<T> y, ReadOnlySpan<T> z, Span<T> d)
    {
        int n = d.Length;
        x = x[..n];
        z = z[..n];
        bool broadcast = y.Length != n;
        int i = 0;
        if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported)
        {
            ref T py = ref MemoryMarshal.GetReference(y);
            ref T pz = ref MemoryMarshal.GetReference(z);
            ref T pd = ref MemoryMarshal.GetReference(d);
            var scalar = broadcast ? new Vector<T>(y[0]) : default;
            var smallestNormal = new Vector<T>(SmallestNormal);
            int last = n - Vector<T>.Count;
            for (; i <= last; i += Vector<T>.Count)
            {
                var q = Vector.LoadUnsafe(ref pz, (nuint)i);
                var b = broadcast ? scalar : Vector.LoadUnsafe(ref py, (nuint)i);
                (-(q / b)).StoreUnsafe(ref pd, (nuint)i);
                var magnitude = Vector.Abs(q);
                var subnormal = Vector.LessThan(magnitude, smallestNormal) & Vector.GreaterThan(magnitude, Vector<T>.Zero);
                if (!Vector.EqualsAll(subnormal, Vector<T>.Zero))
                {
                    for (int j = i; j < i + Vector<T>.Count; j++)
                    {
                        if (T.IsSubnormal(z[j]))
                        {
                            d[j] = DenominatorDerivative(x[j], broadcast ? y[0] : y[j]);
                        }
                    }
                }
            }
        }

        for (; i < n; i++)
        {
            T b = broadcast ? y[0] : y[i];
            d[i] = T.IsSubnormal(z[i]) ? DenominatorDerivative(x[i], b) : -(z[i] / b);
        }
    }

    private static T DenominatorDerivative(T x, T y)
        => T.CreateTruncating(ScalarNode.DenominatorDerivative(double.CreateTruncating(x), double.CreateTruncating(y)));

    private static T SmallestNormal => typeof(T) == typeof(float) ? T.CreateTruncating(1.17549435E-38f) : T.CreateTruncating(2.2250738585072014E-308);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<T> IsNaN(Vector<T> x) => ~Vector.Equals(x, x);
}

internal interface IUnaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    static abstract T Invoke(T x);

    static abstract Vector<T> Invoke(Vector<T> x);
}

internal interface IBinaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    static abstract T Invoke(T x, T y);

    static abstract Vector<T> Invoke(Vector<T> x, Vector<T> y);
}

internal interface ITernaryKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    static abstract T Invoke(T x, T y, T z);

    static abstract Vector<T> Invoke(Vector<T> x, Vector<T> y, Vector<T> z);
}

internal interface IParameterizedKernel<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    static abstract T Invoke(T x, T y, T a, T b, T c);

    static abstract Vector<T> Invoke(Vector<T> x, Vector<T> y, Vector<T> a, Vector<T> b, Vector<T> c);
}
