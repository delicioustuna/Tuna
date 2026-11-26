using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Tuna.Differentiation;

internal static class LossKernels<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void NegatedMagnitudes(ReadOnlySpan<T> x, Span<T> d)
    {
        int n = d.Length;
        x = x[..n];
        int i = 0;
        if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported)
        {
            ref T px = ref MemoryMarshal.GetReference(x);
            ref T pd = ref MemoryMarshal.GetReference(d);
            var sign = new Vector<T>(T.NegativeZero);
            int last = n - Vector<T>.Count;
            for (; i <= last; i += Vector<T>.Count)
            {
                (Vector.LoadUnsafe(ref px, (nuint)i) | sign).StoreUnsafe(ref pd, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            d[i] = -T.Abs(x[i]);
        }
    }

    // Expects s = softplus(-|logits|) on entry and overwrites s with y*softplus(-logits) + (1 - y)*softplus(logits), zeroing exact-zero target terms.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void LogisticLosses(ReadOnlySpan<T> logits, ReadOnlySpan<T> targets, Span<T> s)
    {
        int n = s.Length;
        logits = logits[..n];
        targets = targets[..n];
        int i = 0;
        if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported)
        {
            ref T px = ref MemoryMarshal.GetReference(logits);
            ref T py = ref MemoryMarshal.GetReference(targets);
            ref T ps = ref MemoryMarshal.GetReference(s);
            int last = n - Vector<T>.Count;
            for (; i <= last; i += Vector<T>.Count)
            {
                var x = Vector.LoadUnsafe(ref px, (nuint)i);
                var softplus = Vector.LoadUnsafe(ref ps, (nuint)i);
                var y = Vector.LoadUnsafe(ref py, (nuint)i);
                var complement = Vector<T>.One - y;
                var nonnegative = Vector.GreaterThanOrEqual(x, Vector<T>.Zero);
                var positive = Vector.ConditionalSelect(nonnegative, x + softplus, softplus);
                var negative = Vector.ConditionalSelect(nonnegative, softplus, softplus - x);
                var first = Vector.ConditionalSelect(Vector.Equals(y, Vector<T>.Zero), Vector<T>.Zero, y * negative);
                var second = Vector.ConditionalSelect(Vector.Equals(complement, Vector<T>.Zero), Vector<T>.Zero, complement * positive);
                (first + second).StoreUnsafe(ref ps, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            T x = logits[i], softplus = s[i], y = targets[i], complement = T.One - y;
            bool nonnegative = x >= T.Zero;
            T positive = nonnegative ? x + softplus : softplus;
            T negative = nonnegative ? softplus : softplus - x;
            s[i] = (T.IsZero(y) ? T.Zero : y * negative) + (T.IsZero(complement) ? T.Zero : complement * positive);
        }
    }

    // Expects p = sigma(-|logits|) on entry and overwrites p with (1 - y)*sigma(logits) - y*sigma(-logits) to preserve accuracy in the tails.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void LogisticDerivatives(ReadOnlySpan<T> logits, ReadOnlySpan<T> targets, Span<T> p)
    {
        int n = p.Length;
        logits = logits[..n];
        targets = targets[..n];
        int i = 0;
        if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported)
        {
            ref T px = ref MemoryMarshal.GetReference(logits);
            ref T py = ref MemoryMarshal.GetReference(targets);
            ref T pp = ref MemoryMarshal.GetReference(p);
            int last = n - Vector<T>.Count;
            for (; i <= last; i += Vector<T>.Count)
            {
                var x = Vector.LoadUnsafe(ref px, (nuint)i);
                var logistic = Vector.LoadUnsafe(ref pp, (nuint)i);
                var y = Vector.LoadUnsafe(ref py, (nuint)i);
                var complement = Vector<T>.One - logistic;
                var nonnegative = Vector.GreaterThanOrEqual(x, Vector<T>.Zero);
                var positive = Vector.ConditionalSelect(nonnegative, complement, logistic);
                var negative = Vector.ConditionalSelect(nonnegative, logistic, complement);
                ((Vector<T>.One - y) * positive - y * negative).StoreUnsafe(ref pp, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            T x = logits[i], logistic = p[i], y = targets[i], complement = T.One - logistic;
            bool nonnegative = x >= T.Zero;
            T positive = nonnegative ? complement : logistic;
            T negative = nonnegative ? logistic : complement;
            p[i] = (T.One - y) * positive - y * negative;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void TargetTerms(ReadOnlySpan<T> targets, ReadOnlySpan<T> logSoftmax, Span<T> terms)
    {
        int n = terms.Length;
        targets = targets[..n];
        logSoftmax = logSoftmax[..n];
        int i = 0;
        if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported)
        {
            ref T pt = ref MemoryMarshal.GetReference(targets);
            ref T pl = ref MemoryMarshal.GetReference(logSoftmax);
            ref T pd = ref MemoryMarshal.GetReference(terms);
            int last = n - Vector<T>.Count;
            for (; i <= last; i += Vector<T>.Count)
            {
                var t = Vector.LoadUnsafe(ref pt, (nuint)i);
                var term = Vector<T>.Zero - t * Vector.LoadUnsafe(ref pl, (nuint)i);
                Vector.ConditionalSelect(Vector.Equals(t, Vector<T>.Zero), Vector<T>.Zero, term).StoreUnsafe(ref pd, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            T t = targets[i];
            terms[i] = T.IsZero(t) ? T.Zero : T.Zero - t * logSoftmax[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void LabelLosses(ReadOnlySpan<T> logSoftmax, ReadOnlySpan<int> labels, int classes, Span<T> losses)
    {
        for (int n = 0; n < losses.Length; n++)
        {
            losses[n] = T.Zero - logSoftmax[labels[n] + n * classes];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void SubtractLabels(Span<T> weights, ReadOnlySpan<int> labels, int classes)
    {
        for (int n = 0; n < labels.Length; n++)
        {
            weights[labels[n] + n * classes] -= T.One;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static bool ContainsNaN(ReadOnlySpan<T> x)
    {
        int n = x.Length;
        int i = 0;
        if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported)
        {
            ref T px = ref MemoryMarshal.GetReference(x);
            var found = Vector<T>.Zero;
            int last = n - Vector<T>.Count;
            for (; i <= last; i += Vector<T>.Count)
            {
                var v = Vector.LoadUnsafe(ref px, (nuint)i);
                found |= ~Vector.Equals(v, v);
            }

            if (Vector.AsVectorByte(found) != Vector<byte>.Zero)
            {
                return true;
            }
        }

        for (; i < n; i++)
        {
            if (T.IsNaN(x[i]))
            {
                return true;
            }
        }

        return false;
    }
}
