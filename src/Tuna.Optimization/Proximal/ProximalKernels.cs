using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Tuna.LinearAlgebra;
using Simd = System.Numerics.Vector;
using SimdVector = System.Numerics.Vector<double>;

namespace Tuna.Optimization.Proximal;

// Elementwise proximal kernels (SIMD on contiguous spans; supports exact in-place aliasing) and simplex projection threshold.
internal static class ProximalKernels
{
    private static bool TryDense(ReadOnlyVectorSpan<double> v, VectorSpan<double> destination, out ReadOnlySpan<double> source, out Span<double> target)
    {
        if ((v.Stride == 1 || v.Length <= 1) && (destination.AsReadOnly().Stride == 1 || destination.Length <= 1))
        {
            source = v.Data[..v.Length];
            target = destination.Data[..destination.Length];
            return true;
        }

        source = default;
        target = default;
        return false;
    }

    private static double Soft(double x, double threshold) => Math.Max(x - threshold, 0) + Math.Min(x + threshold, 0);

    public static void SoftThreshold(ReadOnlyVectorSpan<double> v, double threshold, double scale, VectorSpan<double> destination)
    {
        if (!TryDense(v, destination, out var x, out var y))
        {
            for (int k = 0; k < v.Length; k++)
            {
                destination[k] = scale * Soft(v[k], threshold);
            }

            return;
        }

        ref double a = ref MemoryMarshal.GetReference(x);
        ref double b = ref MemoryMarshal.GetReference(y);
        int n = x.Length, i = 0;
        if (Simd.IsHardwareAccelerated)
        {
            var t = new SimdVector(threshold);
            var s = new SimdVector(scale);
            for (; i <= n - SimdVector.Count; i += SimdVector.Count)
            {
                var value = Simd.LoadUnsafe(ref a, (nuint)i);
                var shrunk = Simd.Max(value - t, SimdVector.Zero) + Simd.Min(value + t, SimdVector.Zero);
                Simd.StoreUnsafe((shrunk * s), ref b, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            y[i] = scale * Soft(x[i], threshold);
        }
    }

    public static void SoftThreshold(ReadOnlyVectorSpan<double> v, double threshold, ReadOnlySpan<double> weights, VectorSpan<double> destination)
    {
        if (!TryDense(v, destination, out var x, out var y))
        {
            for (int k = 0; k < v.Length; k++)
            {
                destination[k] = Soft(v[k], threshold * weights[k]);
            }

            return;
        }

        ref double a = ref MemoryMarshal.GetReference(x);
        ref double b = ref MemoryMarshal.GetReference(y);
        ref double w = ref MemoryMarshal.GetReference(weights);
        int n = x.Length, i = 0;
        if (Simd.IsHardwareAccelerated)
        {
            var t = new SimdVector(threshold);
            for (; i <= n - SimdVector.Count; i += SimdVector.Count)
            {
                var value = Simd.LoadUnsafe(ref a, (nuint)i);
                var level = t * Simd.LoadUnsafe(ref w, (nuint)i);
                Simd.StoreUnsafe((Simd.Max(value - level, SimdVector.Zero) + Simd.Min(value + level, SimdVector.Zero)), ref b, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            y[i] = Soft(x[i], threshold * weights[i]);
        }
    }

    // Subtract reference before shift so large v values do not absorb shift into rounding error.
    public static void PositivePart(ReadOnlyVectorSpan<double> v, double reference, double shift, VectorSpan<double> destination)
    {
        if (!TryDense(v, destination, out var x, out var y))
        {
            for (int k = 0; k < v.Length; k++)
            {
                destination[k] = Math.Max((v[k] - reference) - shift, 0);
            }

            return;
        }

        ref double a = ref MemoryMarshal.GetReference(x);
        ref double b = ref MemoryMarshal.GetReference(y);
        int n = x.Length, i = 0;
        if (Simd.IsHardwareAccelerated)
        {
            var r = new SimdVector(reference);
            var s = new SimdVector(shift);
            for (; i <= n - SimdVector.Count; i += SimdVector.Count)
            {
                Simd.StoreUnsafe(Simd.Max((Simd.LoadUnsafe(ref a, (nuint)i) - r) - s, SimdVector.Zero), ref b, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            y[i] = Math.Max((x[i] - reference) - shift, 0);
        }
    }

    // Subtract reference before shift so large |v| values do not absorb shift into rounding error.
    public static void ShiftedSoftThreshold(ReadOnlyVectorSpan<double> v, double reference, double shift, VectorSpan<double> destination)
    {
        for (int k = 0; k < v.Length; k++)
        {
            double x = v[k];
            destination[k] = Math.Max((x - reference) - shift, 0) + Math.Min((x + reference) + shift, 0);
        }
    }

    public static void Clamp(ReadOnlyVectorSpan<double> v, ReadOnlySpan<double> lower, ReadOnlySpan<double> upper, VectorSpan<double> destination)
    {
        if (!TryDense(v, destination, out var x, out var y))
        {
            for (int k = 0; k < v.Length; k++)
            {
                destination[k] = Math.Min(Math.Max(v[k], lower[k]), upper[k]);
            }

            return;
        }

        ref double a = ref MemoryMarshal.GetReference(x);
        ref double b = ref MemoryMarshal.GetReference(y);
        ref double lo = ref MemoryMarshal.GetReference(lower);
        ref double hi = ref MemoryMarshal.GetReference(upper);
        int n = x.Length, i = 0;
        if (Simd.IsHardwareAccelerated)
        {
            for (; i <= n - SimdVector.Count; i += SimdVector.Count)
            {
                var value = Simd.Max(Simd.LoadUnsafe(ref a, (nuint)i), Simd.LoadUnsafe(ref lo, (nuint)i));
                Simd.StoreUnsafe(Simd.Min(value, Simd.LoadUnsafe(ref hi, (nuint)i)), ref b, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            y[i] = Math.Min(Math.Max(x[i], lower[i]), upper[i]);
        }
    }

    public static void Scale(ReadOnlyVectorSpan<double> v, double factor, VectorSpan<double> destination)
    {
        if (!TryDense(v, destination, out var x, out var y))
        {
            for (int k = 0; k < v.Length; k++)
            {
                destination[k] = factor * v[k];
            }

            return;
        }

        if (!Unsafe.AreSame(ref MemoryMarshal.GetReference(x), ref MemoryMarshal.GetReference(y)))
        {
            x.CopyTo(y);
        }

        if (factor != 1)
        {
            Blas.Scale(factor, y);
        }
    }

    public static double Norm2(ReadOnlyVectorSpan<double> v)
    {
        if (v.Stride == 1 || v.Length <= 1)
        {
            return Blas.Norm2(v.Data[..v.Length]);
        }

        double scale = 0;
        for (int i = 0; i < v.Length; i++)
        {
            scale = Math.Max(scale, Math.Abs(v[i]));
        }

        if (scale == 0 || !double.IsFinite(scale))
        {
            return scale;
        }

        double sum = 0;
        for (int i = 0; i < v.Length; i++)
        {
            double r = v[i] / scale;
            sum += r * r;
        }

        return scale * Math.Sqrt(sum);
    }

    public static double SumAbs(ReadOnlyVectorSpan<double> v)
    {
        if (v.Stride == 1 || v.Length <= 1)
        {
            return Blas.SumAbs(v.Data[..v.Length]);
        }

        double sum = 0;
        for (int i = 0; i < v.Length; i++)
        {
            sum += Math.Abs(v[i]);
        }

        return sum;
    }

    public static double SimplexReference(ReadOnlyVectorSpan<double> y, bool absolute)
    {
        double max = absolute ? Math.Abs(y[0]) : y[0];
        for (int i = 1; i < y.Length; i++)
        {
            max = Math.Max(max, absolute ? Math.Abs(y[i]) : y[i]);
        }

        return max;
    }

    // Condat (2016) / Michelot (1986) expected-O(n) simplex projection threshold, returned as τ − reference.
    // Shifting by reference = max(y) keeps τ − reference in [−total, 0) so large coordinates do not lose total's digits;
    // clamping z = y − reference below at −total excludes inactive coordinates and avoids ∞ − ∞ on overflow.
    public static double SimplexThreshold(ReadOnlyVectorSpan<double> y, double reference, double total, bool absolute, ref double[]? buffer)
    {
        int n = y.Length;
        if (buffer is null || buffer.Length < 2 * n)
        {
            buffer = new double[2 * n];
        }

        Span<double> active = buffer.AsSpan(0, n);
        Span<double> deferred = buffer.AsSpan(n, n);
        int count = 0, deferredCount = 0;
        double first = Math.Max((absolute ? Math.Abs(y[0]) : y[0]) - reference, -total);
        active[count++] = first;
        double rho = first - total;
        for (int i = 1; i < n; i++)
        {
            double yi = Math.Max((absolute ? Math.Abs(y[i]) : y[i]) - reference, -total);
            if (!(yi > rho))
            {
                continue;
            }

            double candidate = rho + (yi - rho) / (count + 1);
            if (candidate > yi - total)
            {
                active[count++] = yi;
                rho = candidate;
            }
            else
            {
                active[..count].CopyTo(deferred[deferredCount..]);
                deferredCount += count;
                active[0] = yi;
                count = 1;
                rho = yi - total;
            }
        }

        for (int k = 0; k < deferredCount; k++)
        {
            double yk = deferred[k];
            if (yk > rho)
            {
                active[count++] = yk;
                rho += (yk - rho) / count;
            }
        }

        while (true)
        {
            int size = count, write = 0;
            for (int k = 0; k < count; k++)
            {
                double yk = active[k];
                if (yk <= rho && size > 1)
                {
                    size--;
                    rho += (rho - yk) / size;
                }
                else
                {
                    active[write++] = yk;
                }
            }

            if (write == count)
            {
                return rho;
            }

            count = write;
        }
    }
}
