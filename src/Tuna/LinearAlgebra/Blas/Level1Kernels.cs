using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Tuna.LinearAlgebra;

// 連続 double 領域に対する BLAS Level 1 相当および分解向け融合 SIMD カーネル。
internal static class Level1Kernels
{
    // 平方和のオーバーフロー・アンダーフロー回避のためスケーリング経路へ切り替える閾値。
    private const double SumSquaresHigh = 1e150;
    private const double SumSquaresLow = 1e-150;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static double Dot(ReadOnlySpan<double> x, ReadOnlySpan<double> y)
    {
        Debug.Assert(y.Length >= x.Length);
        int n = x.Length;
        ref double px = ref MemoryMarshal.GetReference(x);
        ref double py = ref MemoryMarshal.GetReference(y);
        int i = 0;
        double sum = 0;
        if (Vector512.IsHardwareAccelerated && n >= 8)
        {
            var s0 = Vector512<double>.Zero;
            var s1 = Vector512<double>.Zero;
            var s2 = Vector512<double>.Zero;
            var s3 = Vector512<double>.Zero;
            for (; i <= n - 32; i += 32)
            {
                s0 = Vector512.FusedMultiplyAdd(Vector512.LoadUnsafe(ref px, (nuint)i), Vector512.LoadUnsafe(ref py, (nuint)i), s0);
                s1 = Vector512.FusedMultiplyAdd(Vector512.LoadUnsafe(ref px, (nuint)(i + 8)), Vector512.LoadUnsafe(ref py, (nuint)(i + 8)), s1);
                s2 = Vector512.FusedMultiplyAdd(Vector512.LoadUnsafe(ref px, (nuint)(i + 16)), Vector512.LoadUnsafe(ref py, (nuint)(i + 16)), s2);
                s3 = Vector512.FusedMultiplyAdd(Vector512.LoadUnsafe(ref px, (nuint)(i + 24)), Vector512.LoadUnsafe(ref py, (nuint)(i + 24)), s3);
            }

            for (; i <= n - 8; i += 8)
            {
                s0 = Vector512.FusedMultiplyAdd(Vector512.LoadUnsafe(ref px, (nuint)i), Vector512.LoadUnsafe(ref py, (nuint)i), s0);
            }

            sum = Vector512.Sum((s0 + s1) + (s2 + s3));
        }
        else if (Vector256.IsHardwareAccelerated && n >= 4)
        {
            var s0 = Vector256<double>.Zero;
            var s1 = Vector256<double>.Zero;
            var s2 = Vector256<double>.Zero;
            var s3 = Vector256<double>.Zero;
            for (; i <= n - 16; i += 16)
            {
                s0 = Vector256.MultiplyAddEstimate(Vector256.LoadUnsafe(ref px, (nuint)i), Vector256.LoadUnsafe(ref py, (nuint)i), s0);
                s1 = Vector256.MultiplyAddEstimate(Vector256.LoadUnsafe(ref px, (nuint)(i + 4)), Vector256.LoadUnsafe(ref py, (nuint)(i + 4)), s1);
                s2 = Vector256.MultiplyAddEstimate(Vector256.LoadUnsafe(ref px, (nuint)(i + 8)), Vector256.LoadUnsafe(ref py, (nuint)(i + 8)), s2);
                s3 = Vector256.MultiplyAddEstimate(Vector256.LoadUnsafe(ref px, (nuint)(i + 12)), Vector256.LoadUnsafe(ref py, (nuint)(i + 12)), s3);
            }

            for (; i <= n - 4; i += 4)
            {
                s0 = Vector256.MultiplyAddEstimate(Vector256.LoadUnsafe(ref px, (nuint)i), Vector256.LoadUnsafe(ref py, (nuint)i), s0);
            }

            sum = Vector256.Sum((s0 + s1) + (s2 + s3));
        }

        for (; i < n; i++)
        {
            sum += Unsafe.Add(ref px, i) * Unsafe.Add(ref py, i);
        }

        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Axpy(double a, ReadOnlySpan<double> x, Span<double> y)
    {
        Debug.Assert(y.Length >= x.Length);
        if (a == 0)
        {
            return;
        }

        int n = x.Length;
        ref double px = ref MemoryMarshal.GetReference(x);
        ref double py = ref MemoryMarshal.GetReference(y);
        int i = 0;
        if (Vector512.IsHardwareAccelerated && n >= 8)
        {
            var va = Vector512.Create(a);
            for (; i <= n - 16; i += 16)
            {
                var y0 = Vector512.FusedMultiplyAdd(va, Vector512.LoadUnsafe(ref px, (nuint)i), Vector512.LoadUnsafe(ref py, (nuint)i));
                var y1 = Vector512.FusedMultiplyAdd(va, Vector512.LoadUnsafe(ref px, (nuint)(i + 8)), Vector512.LoadUnsafe(ref py, (nuint)(i + 8)));
                y0.StoreUnsafe(ref py, (nuint)i);
                y1.StoreUnsafe(ref py, (nuint)(i + 8));
            }

            for (; i <= n - 8; i += 8)
            {
                Vector512.FusedMultiplyAdd(va, Vector512.LoadUnsafe(ref px, (nuint)i), Vector512.LoadUnsafe(ref py, (nuint)i)).StoreUnsafe(ref py, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && n >= 4)
        {
            var va = Vector256.Create(a);
            for (; i <= n - 8; i += 8)
            {
                var y0 = Vector256.MultiplyAddEstimate(va, Vector256.LoadUnsafe(ref px, (nuint)i), Vector256.LoadUnsafe(ref py, (nuint)i));
                var y1 = Vector256.MultiplyAddEstimate(va, Vector256.LoadUnsafe(ref px, (nuint)(i + 4)), Vector256.LoadUnsafe(ref py, (nuint)(i + 4)));
                y0.StoreUnsafe(ref py, (nuint)i);
                y1.StoreUnsafe(ref py, (nuint)(i + 4));
            }

            for (; i <= n - 4; i += 4)
            {
                Vector256.MultiplyAddEstimate(va, Vector256.LoadUnsafe(ref px, (nuint)i), Vector256.LoadUnsafe(ref py, (nuint)i)).StoreUnsafe(ref py, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            Unsafe.Add(ref py, i) += a * Unsafe.Add(ref px, i);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Scale(double a, Span<double> x)
    {
        int n = x.Length;
        ref double px = ref MemoryMarshal.GetReference(x);
        int i = 0;
        if (Vector512.IsHardwareAccelerated && n >= 8)
        {
            var va = Vector512.Create(a);
            for (; i <= n - 8; i += 8)
            {
                (va * Vector512.LoadUnsafe(ref px, (nuint)i)).StoreUnsafe(ref px, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && n >= 4)
        {
            var va = Vector256.Create(a);
            for (; i <= n - 4; i += 4)
            {
                (va * Vector256.LoadUnsafe(ref px, (nuint)i)).StoreUnsafe(ref px, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            Unsafe.Add(ref px, i) *= a;
        }
    }

    internal static void Multiply(Span<double> x, ReadOnlySpan<double> w)
    {
        Debug.Assert(w.Length >= x.Length);
        int n = x.Length;
        ref double px = ref MemoryMarshal.GetReference(x);
        ref double pw = ref MemoryMarshal.GetReference(w);
        int i = 0;
        if (Vector512.IsHardwareAccelerated && n >= 8)
        {
            for (; i <= n - 8; i += 8)
            {
                (Vector512.LoadUnsafe(ref pw, (nuint)i) * Vector512.LoadUnsafe(ref px, (nuint)i)).StoreUnsafe(ref px, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && n >= 4)
        {
            for (; i <= n - 4; i += 4)
            {
                (Vector256.LoadUnsafe(ref pw, (nuint)i) * Vector256.LoadUnsafe(ref px, (nuint)i)).StoreUnsafe(ref px, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            Unsafe.Add(ref px, i) *= Unsafe.Add(ref pw, i);
        }
    }

    // 平面回転 (x, y) <- (c*x + s*y, c*y - s*x)。
    internal static void Rotate(Span<double> x, Span<double> y, double c, double s)
    {
        Debug.Assert(y.Length >= x.Length);
        int n = x.Length;
        ref double px = ref MemoryMarshal.GetReference(x);
        ref double py = ref MemoryMarshal.GetReference(y);
        int i = 0;
        if (Vector512.IsHardwareAccelerated && n >= 8)
        {
            var vc = Vector512.Create(c);
            var vs = Vector512.Create(s);
            for (; i <= n - 8; i += 8)
            {
                var xv = Vector512.LoadUnsafe(ref px, (nuint)i);
                var yv = Vector512.LoadUnsafe(ref py, (nuint)i);
                Vector512.FusedMultiplyAdd(vc, xv, vs * yv).StoreUnsafe(ref px, (nuint)i);
                Vector512.FusedMultiplyAdd(vc, yv, -(vs * xv)).StoreUnsafe(ref py, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && n >= 4)
        {
            var vc = Vector256.Create(c);
            var vs = Vector256.Create(s);
            for (; i <= n - 4; i += 4)
            {
                var xv = Vector256.LoadUnsafe(ref px, (nuint)i);
                var yv = Vector256.LoadUnsafe(ref py, (nuint)i);
                Vector256.MultiplyAddEstimate(vc, xv, vs * yv).StoreUnsafe(ref px, (nuint)i);
                Vector256.MultiplyAddEstimate(vc, yv, -(vs * xv)).StoreUnsafe(ref py, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            double xi = Unsafe.Add(ref px, i);
            double yi = Unsafe.Add(ref py, i);
            Unsafe.Add(ref px, i) = c * xi + s * yi;
            Unsafe.Add(ref py, i) = c * yi - s * xi;
        }
    }

    // x^T z を返しつつ y += a * x を 1 パスで実行する（対称行列ベクトル積用融合演算）。
    internal static double DotAxpy(ReadOnlySpan<double> x, ReadOnlySpan<double> z, double a, Span<double> y)
    {
        Debug.Assert(z.Length >= x.Length && y.Length >= x.Length);
        int n = x.Length;
        ref double px = ref MemoryMarshal.GetReference(x);
        ref double pz = ref MemoryMarshal.GetReference(z);
        ref double py = ref MemoryMarshal.GetReference(y);
        int i = 0;
        double sum = 0;
        if (Vector512.IsHardwareAccelerated && n >= 8)
        {
            var va = Vector512.Create(a);
            var s0 = Vector512<double>.Zero;
            var s1 = Vector512<double>.Zero;
            for (; i <= n - 16; i += 16)
            {
                var x0 = Vector512.LoadUnsafe(ref px, (nuint)i);
                var x1 = Vector512.LoadUnsafe(ref px, (nuint)(i + 8));
                s0 = Vector512.FusedMultiplyAdd(x0, Vector512.LoadUnsafe(ref pz, (nuint)i), s0);
                s1 = Vector512.FusedMultiplyAdd(x1, Vector512.LoadUnsafe(ref pz, (nuint)(i + 8)), s1);
                Vector512.FusedMultiplyAdd(va, x0, Vector512.LoadUnsafe(ref py, (nuint)i)).StoreUnsafe(ref py, (nuint)i);
                Vector512.FusedMultiplyAdd(va, x1, Vector512.LoadUnsafe(ref py, (nuint)(i + 8))).StoreUnsafe(ref py, (nuint)(i + 8));
            }

            for (; i <= n - 8; i += 8)
            {
                var x0 = Vector512.LoadUnsafe(ref px, (nuint)i);
                s0 = Vector512.FusedMultiplyAdd(x0, Vector512.LoadUnsafe(ref pz, (nuint)i), s0);
                Vector512.FusedMultiplyAdd(va, x0, Vector512.LoadUnsafe(ref py, (nuint)i)).StoreUnsafe(ref py, (nuint)i);
            }

            sum = Vector512.Sum(s0 + s1);
        }
        else if (Vector256.IsHardwareAccelerated && n >= 4)
        {
            var va = Vector256.Create(a);
            var s0 = Vector256<double>.Zero;
            var s1 = Vector256<double>.Zero;
            for (; i <= n - 8; i += 8)
            {
                var x0 = Vector256.LoadUnsafe(ref px, (nuint)i);
                var x1 = Vector256.LoadUnsafe(ref px, (nuint)(i + 4));
                s0 = Vector256.MultiplyAddEstimate(x0, Vector256.LoadUnsafe(ref pz, (nuint)i), s0);
                s1 = Vector256.MultiplyAddEstimate(x1, Vector256.LoadUnsafe(ref pz, (nuint)(i + 4)), s1);
                Vector256.MultiplyAddEstimate(va, x0, Vector256.LoadUnsafe(ref py, (nuint)i)).StoreUnsafe(ref py, (nuint)i);
                Vector256.MultiplyAddEstimate(va, x1, Vector256.LoadUnsafe(ref py, (nuint)(i + 4))).StoreUnsafe(ref py, (nuint)(i + 4));
            }

            for (; i <= n - 4; i += 4)
            {
                var x0 = Vector256.LoadUnsafe(ref px, (nuint)i);
                s0 = Vector256.MultiplyAddEstimate(x0, Vector256.LoadUnsafe(ref pz, (nuint)i), s0);
                Vector256.MultiplyAddEstimate(va, x0, Vector256.LoadUnsafe(ref py, (nuint)i)).StoreUnsafe(ref py, (nuint)i);
            }

            sum = Vector256.Sum(s0 + s1);
        }

        for (; i < n; i++)
        {
            double xi = Unsafe.Add(ref px, i);
            sum += xi * Unsafe.Add(ref pz, i);
            Unsafe.Add(ref py, i) += a * xi;
        }

        return sum;
    }

    // 連続 4 列に対する DotAxpy 融合演算（x・y のメモリアクセスを 4 列で共有）。
    internal static void DotAxpy4(
        int n, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> x, double t0, double t1, double t2, double t3, Span<double> y,
        out double d0, out double d1, out double d2, out double d3)
    {
        Debug.Assert(a.Length >= 3 * lda + n && x.Length >= n && y.Length >= n);
        ref double pa = ref MemoryMarshal.GetReference(a);
        ref double p1 = ref Unsafe.Add(ref pa, lda);
        ref double p2 = ref Unsafe.Add(ref pa, 2 * lda);
        ref double p3 = ref Unsafe.Add(ref pa, 3 * lda);
        ref double px = ref MemoryMarshal.GetReference(x);
        ref double py = ref MemoryMarshal.GetReference(y);
        int i = 0;
        double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        if (Vector512.IsHardwareAccelerated && n >= 8)
        {
            var v0 = Vector512.Create(t0);
            var v1 = Vector512.Create(t1);
            var v2 = Vector512.Create(t2);
            var v3 = Vector512.Create(t3);
            var a0 = Vector512<double>.Zero;
            var a1 = Vector512<double>.Zero;
            var a2 = Vector512<double>.Zero;
            var a3 = Vector512<double>.Zero;
            for (; i <= n - 8; i += 8)
            {
                var xv = Vector512.LoadUnsafe(ref px, (nuint)i);
                var c0 = Vector512.LoadUnsafe(ref pa, (nuint)i);
                var c1 = Vector512.LoadUnsafe(ref p1, (nuint)i);
                var c2 = Vector512.LoadUnsafe(ref p2, (nuint)i);
                var c3 = Vector512.LoadUnsafe(ref p3, (nuint)i);
                a0 = Vector512.FusedMultiplyAdd(c0, xv, a0);
                a1 = Vector512.FusedMultiplyAdd(c1, xv, a1);
                a2 = Vector512.FusedMultiplyAdd(c2, xv, a2);
                a3 = Vector512.FusedMultiplyAdd(c3, xv, a3);
                var yv = Vector512.FusedMultiplyAdd(v0, c0, Vector512.LoadUnsafe(ref py, (nuint)i));
                yv = Vector512.FusedMultiplyAdd(v1, c1, yv);
                yv = Vector512.FusedMultiplyAdd(v2, c2, yv);
                Vector512.FusedMultiplyAdd(v3, c3, yv).StoreUnsafe(ref py, (nuint)i);
            }

            s0 = Vector512.Sum(a0);
            s1 = Vector512.Sum(a1);
            s2 = Vector512.Sum(a2);
            s3 = Vector512.Sum(a3);
        }
        else if (Vector256.IsHardwareAccelerated && n >= 4)
        {
            var v0 = Vector256.Create(t0);
            var v1 = Vector256.Create(t1);
            var v2 = Vector256.Create(t2);
            var v3 = Vector256.Create(t3);
            var a0 = Vector256<double>.Zero;
            var a1 = Vector256<double>.Zero;
            var a2 = Vector256<double>.Zero;
            var a3 = Vector256<double>.Zero;
            for (; i <= n - 4; i += 4)
            {
                var xv = Vector256.LoadUnsafe(ref px, (nuint)i);
                var c0 = Vector256.LoadUnsafe(ref pa, (nuint)i);
                var c1 = Vector256.LoadUnsafe(ref p1, (nuint)i);
                var c2 = Vector256.LoadUnsafe(ref p2, (nuint)i);
                var c3 = Vector256.LoadUnsafe(ref p3, (nuint)i);
                a0 = Vector256.MultiplyAddEstimate(c0, xv, a0);
                a1 = Vector256.MultiplyAddEstimate(c1, xv, a1);
                a2 = Vector256.MultiplyAddEstimate(c2, xv, a2);
                a3 = Vector256.MultiplyAddEstimate(c3, xv, a3);
                var yv = Vector256.MultiplyAddEstimate(v0, c0, Vector256.LoadUnsafe(ref py, (nuint)i));
                yv = Vector256.MultiplyAddEstimate(v1, c1, yv);
                yv = Vector256.MultiplyAddEstimate(v2, c2, yv);
                Vector256.MultiplyAddEstimate(v3, c3, yv).StoreUnsafe(ref py, (nuint)i);
            }

            s0 = Vector256.Sum(a0);
            s1 = Vector256.Sum(a1);
            s2 = Vector256.Sum(a2);
            s3 = Vector256.Sum(a3);
        }

        for (; i < n; i++)
        {
            double xi = Unsafe.Add(ref px, i);
            double c0 = Unsafe.Add(ref pa, i);
            double c1 = Unsafe.Add(ref p1, i);
            double c2 = Unsafe.Add(ref p2, i);
            double c3 = Unsafe.Add(ref p3, i);
            s0 += c0 * xi;
            s1 += c1 * xi;
            s2 += c2 * xi;
            s3 += c3 * xi;
            Unsafe.Add(ref py, i) += t0 * c0 + t1 * c1 + t2 * c2 + t3 * c3;
        }

        d0 = s0;
        d1 = s1;
        d2 = s2;
        d3 = s3;
    }

    // 連続 4 列 A[:, 0..3] とベクトル x の内積。
    internal static void Dot4(int n, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> x, out double r0, out double r1, out double r2, out double r3)
    {
        Debug.Assert(a.Length >= 3 * lda + n && x.Length >= n);
        ref double p0 = ref MemoryMarshal.GetReference(a);
        ref double p1 = ref Unsafe.Add(ref p0, lda);
        ref double p2 = ref Unsafe.Add(ref p0, 2 * lda);
        ref double p3 = ref Unsafe.Add(ref p0, 3 * lda);
        ref double px = ref MemoryMarshal.GetReference(x);
        int i = 0;
        r0 = r1 = r2 = r3 = 0;
        if (Vector512.IsHardwareAccelerated && n >= 8)
        {
            var s0 = Vector512<double>.Zero;
            var s1 = Vector512<double>.Zero;
            var s2 = Vector512<double>.Zero;
            var s3 = Vector512<double>.Zero;
            for (; i <= n - 8; i += 8)
            {
                var xv = Vector512.LoadUnsafe(ref px, (nuint)i);
                s0 = Vector512.FusedMultiplyAdd(Vector512.LoadUnsafe(ref p0, (nuint)i), xv, s0);
                s1 = Vector512.FusedMultiplyAdd(Vector512.LoadUnsafe(ref p1, (nuint)i), xv, s1);
                s2 = Vector512.FusedMultiplyAdd(Vector512.LoadUnsafe(ref p2, (nuint)i), xv, s2);
                s3 = Vector512.FusedMultiplyAdd(Vector512.LoadUnsafe(ref p3, (nuint)i), xv, s3);
            }

            r0 = Vector512.Sum(s0);
            r1 = Vector512.Sum(s1);
            r2 = Vector512.Sum(s2);
            r3 = Vector512.Sum(s3);
        }
        else if (Vector256.IsHardwareAccelerated && n >= 4)
        {
            var s0 = Vector256<double>.Zero;
            var s1 = Vector256<double>.Zero;
            var s2 = Vector256<double>.Zero;
            var s3 = Vector256<double>.Zero;
            for (; i <= n - 4; i += 4)
            {
                var xv = Vector256.LoadUnsafe(ref px, (nuint)i);
                s0 = Vector256.MultiplyAddEstimate(Vector256.LoadUnsafe(ref p0, (nuint)i), xv, s0);
                s1 = Vector256.MultiplyAddEstimate(Vector256.LoadUnsafe(ref p1, (nuint)i), xv, s1);
                s2 = Vector256.MultiplyAddEstimate(Vector256.LoadUnsafe(ref p2, (nuint)i), xv, s2);
                s3 = Vector256.MultiplyAddEstimate(Vector256.LoadUnsafe(ref p3, (nuint)i), xv, s3);
            }

            r0 = Vector256.Sum(s0);
            r1 = Vector256.Sum(s1);
            r2 = Vector256.Sum(s2);
            r3 = Vector256.Sum(s3);
        }

        for (; i < n; i++)
        {
            double xi = Unsafe.Add(ref px, i);
            r0 += Unsafe.Add(ref p0, i) * xi;
            r1 += Unsafe.Add(ref p1, i) * xi;
            r2 += Unsafe.Add(ref p2, i) * xi;
            r3 += Unsafe.Add(ref p3, i) * xi;
        }
    }

    // 連続 4 列 A[:, c] += s_c * x。
    internal static void AddToColumns4(int n, double s0, double s1, double s2, double s3, ReadOnlySpan<double> x, Span<double> a, int lda)
    {
        Debug.Assert(a.Length >= 3 * lda + n && x.Length >= n);
        ref double p0 = ref MemoryMarshal.GetReference(a);
        ref double p1 = ref Unsafe.Add(ref p0, lda);
        ref double p2 = ref Unsafe.Add(ref p0, 2 * lda);
        ref double p3 = ref Unsafe.Add(ref p0, 3 * lda);
        ref double px = ref MemoryMarshal.GetReference(x);
        int i = 0;
        if (Vector512.IsHardwareAccelerated && n >= 8)
        {
            var v0 = Vector512.Create(s0);
            var v1 = Vector512.Create(s1);
            var v2 = Vector512.Create(s2);
            var v3 = Vector512.Create(s3);
            for (; i <= n - 8; i += 8)
            {
                var xv = Vector512.LoadUnsafe(ref px, (nuint)i);
                Vector512.FusedMultiplyAdd(v0, xv, Vector512.LoadUnsafe(ref p0, (nuint)i)).StoreUnsafe(ref p0, (nuint)i);
                Vector512.FusedMultiplyAdd(v1, xv, Vector512.LoadUnsafe(ref p1, (nuint)i)).StoreUnsafe(ref p1, (nuint)i);
                Vector512.FusedMultiplyAdd(v2, xv, Vector512.LoadUnsafe(ref p2, (nuint)i)).StoreUnsafe(ref p2, (nuint)i);
                Vector512.FusedMultiplyAdd(v3, xv, Vector512.LoadUnsafe(ref p3, (nuint)i)).StoreUnsafe(ref p3, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && n >= 4)
        {
            var v0 = Vector256.Create(s0);
            var v1 = Vector256.Create(s1);
            var v2 = Vector256.Create(s2);
            var v3 = Vector256.Create(s3);
            for (; i <= n - 4; i += 4)
            {
                var xv = Vector256.LoadUnsafe(ref px, (nuint)i);
                Vector256.MultiplyAddEstimate(v0, xv, Vector256.LoadUnsafe(ref p0, (nuint)i)).StoreUnsafe(ref p0, (nuint)i);
                Vector256.MultiplyAddEstimate(v1, xv, Vector256.LoadUnsafe(ref p1, (nuint)i)).StoreUnsafe(ref p1, (nuint)i);
                Vector256.MultiplyAddEstimate(v2, xv, Vector256.LoadUnsafe(ref p2, (nuint)i)).StoreUnsafe(ref p2, (nuint)i);
                Vector256.MultiplyAddEstimate(v3, xv, Vector256.LoadUnsafe(ref p3, (nuint)i)).StoreUnsafe(ref p3, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            double xi = Unsafe.Add(ref px, i);
            Unsafe.Add(ref p0, i) += s0 * xi;
            Unsafe.Add(ref p1, i) += s1 * xi;
            Unsafe.Add(ref p2, i) += s2 * xi;
            Unsafe.Add(ref p3, i) += s3 * xi;
        }
    }

    // y += Σ_{c=0..3} t_c * A[:, c]。
    internal static void AddColumns4(int n, ReadOnlySpan<double> a, int lda, double t0, double t1, double t2, double t3, Span<double> y)
    {
        Debug.Assert(a.Length >= 3 * lda + n && y.Length >= n);
        ref double p0 = ref MemoryMarshal.GetReference(a);
        ref double p1 = ref Unsafe.Add(ref p0, lda);
        ref double p2 = ref Unsafe.Add(ref p0, 2 * lda);
        ref double p3 = ref Unsafe.Add(ref p0, 3 * lda);
        ref double py = ref MemoryMarshal.GetReference(y);
        int i = 0;
        if (Vector512.IsHardwareAccelerated && n >= 8)
        {
            var v0 = Vector512.Create(t0);
            var v1 = Vector512.Create(t1);
            var v2 = Vector512.Create(t2);
            var v3 = Vector512.Create(t3);
            for (; i <= n - 8; i += 8)
            {
                var yv = Vector512.FusedMultiplyAdd(v0, Vector512.LoadUnsafe(ref p0, (nuint)i), Vector512.LoadUnsafe(ref py, (nuint)i));
                yv = Vector512.FusedMultiplyAdd(v1, Vector512.LoadUnsafe(ref p1, (nuint)i), yv);
                yv = Vector512.FusedMultiplyAdd(v2, Vector512.LoadUnsafe(ref p2, (nuint)i), yv);
                Vector512.FusedMultiplyAdd(v3, Vector512.LoadUnsafe(ref p3, (nuint)i), yv).StoreUnsafe(ref py, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && n >= 4)
        {
            var v0 = Vector256.Create(t0);
            var v1 = Vector256.Create(t1);
            var v2 = Vector256.Create(t2);
            var v3 = Vector256.Create(t3);
            for (; i <= n - 4; i += 4)
            {
                var yv = Vector256.MultiplyAddEstimate(v0, Vector256.LoadUnsafe(ref p0, (nuint)i), Vector256.LoadUnsafe(ref py, (nuint)i));
                yv = Vector256.MultiplyAddEstimate(v1, Vector256.LoadUnsafe(ref p1, (nuint)i), yv);
                yv = Vector256.MultiplyAddEstimate(v2, Vector256.LoadUnsafe(ref p2, (nuint)i), yv);
                Vector256.MultiplyAddEstimate(v3, Vector256.LoadUnsafe(ref p3, (nuint)i), yv).StoreUnsafe(ref py, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            Unsafe.Add(ref py, i) += t0 * Unsafe.Add(ref p0, i) + t1 * Unsafe.Add(ref p1, i) + t2 * Unsafe.Add(ref p2, i) + t3 * Unsafe.Add(ref p3, i);
        }
    }

    // z += a * x + b * y（対称ランク 2 更新用）。
    internal static void Axpy2(double a, ReadOnlySpan<double> x, double b, ReadOnlySpan<double> y, Span<double> z)
    {
        Debug.Assert(y.Length >= x.Length && z.Length >= x.Length);
        int n = x.Length;
        ref double px = ref MemoryMarshal.GetReference(x);
        ref double py = ref MemoryMarshal.GetReference(y);
        ref double pz = ref MemoryMarshal.GetReference(z);
        int i = 0;
        if (Vector512.IsHardwareAccelerated && n >= 8)
        {
            var va = Vector512.Create(a);
            var vb = Vector512.Create(b);
            for (; i <= n - 8; i += 8)
            {
                var t = Vector512.FusedMultiplyAdd(va, Vector512.LoadUnsafe(ref px, (nuint)i), Vector512.LoadUnsafe(ref pz, (nuint)i));
                Vector512.FusedMultiplyAdd(vb, Vector512.LoadUnsafe(ref py, (nuint)i), t).StoreUnsafe(ref pz, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && n >= 4)
        {
            var va = Vector256.Create(a);
            var vb = Vector256.Create(b);
            for (; i <= n - 4; i += 4)
            {
                var t = Vector256.MultiplyAddEstimate(va, Vector256.LoadUnsafe(ref px, (nuint)i), Vector256.LoadUnsafe(ref pz, (nuint)i));
                Vector256.MultiplyAddEstimate(vb, Vector256.LoadUnsafe(ref py, (nuint)i), t).StoreUnsafe(ref pz, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            Unsafe.Add(ref pz, i) += a * Unsafe.Add(ref px, i) + b * Unsafe.Add(ref py, i);
        }
    }

    // LAPACK dlasr 'R','V' 相当。隣接列ペアに平面回転を連鎖適用する（直前の結果を保持し各列 1 回の読み書きに抑える）。
    internal static void RotateSequence(Span<double> z, int ld, int rows, int start, ReadOnlySpan<double> c, ReadOnlySpan<double> s, bool forward)
    {
        int count = c.Length;
        if (count == 0 || rows == 0)
        {
            return;
        }

        // 行を数百行ずつのタイルに分け、持ち越す列だけを作業領域（L1）に置く。
        // 各列はタイルごとに連続領域として1回読み1回書く
        const int Tile = 2048;
        Span<double> carry = stackalloc double[Math.Min(Tile, rows)];

        // 処理した行数だけ進める（r0 += Tile は、行数が int の上限に近いと最後のタイルの後に負へ折り返す）
        for (int r0 = 0, rr; r0 < rows; r0 += rr)
        {
            rr = Math.Min(Tile, rows - r0);
            var cbuf = carry[..rr];
            if (forward)
            {
                z.Slice(r0 + start * ld, rr).CopyTo(cbuf);
                for (int j = 0; j < count; j++)
                {
                    var x = z.Slice(r0 + (start + j) * ld, rr);
                    var y = z.Slice(r0 + (start + j + 1) * ld, rr);
                    RotateCarry(cbuf, y, x, c[j], s[j], forward: true);
                }

                cbuf.CopyTo(z.Slice(r0 + (start + count) * ld, rr));
            }
            else
            {
                z.Slice(r0 + (start + count) * ld, rr).CopyTo(cbuf);
                for (int j = count - 1; j >= 0; j--)
                {
                    var x = z.Slice(r0 + (start + j) * ld, rr);
                    var y = z.Slice(r0 + (start + j + 1) * ld, rr);
                    RotateCarry(cbuf, x, y, c[j], s[j], forward: false);
                }

                cbuf.CopyTo(z.Slice(r0 + start * ld, rr));
            }
        }
    }

    // RotateSequence の 1 回転分（前進時は carry=x/input=y、後退時は carry=y/input=x）。
    private static void RotateCarry(Span<double> carry, ReadOnlySpan<double> input, Span<double> output, double c, double s, bool forward)
    {
        int n = carry.Length;
        ref double pc = ref MemoryMarshal.GetReference(carry);
        ref double pi = ref MemoryMarshal.GetReference(input);
        ref double po = ref MemoryMarshal.GetReference(output);
        double so = forward ? s : -s;
        double sc = forward ? -s : s;
        int i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            var vc = Vector512.Create(c);
            var vso = Vector512.Create(so);
            var vsc = Vector512.Create(sc);
            for (; i <= n - 8; i += 8)
            {
                var k = Vector512.LoadUnsafe(ref pc, (nuint)i);
                var v = Vector512.LoadUnsafe(ref pi, (nuint)i);
                Vector512.FusedMultiplyAdd(vc, k, vso * v).StoreUnsafe(ref po, (nuint)i);
                Vector512.FusedMultiplyAdd(vc, v, vsc * k).StoreUnsafe(ref pc, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var vc = Vector256.Create(c);
            var vso = Vector256.Create(so);
            var vsc = Vector256.Create(sc);
            for (; i <= n - 4; i += 4)
            {
                var k = Vector256.LoadUnsafe(ref pc, (nuint)i);
                var v = Vector256.LoadUnsafe(ref pi, (nuint)i);
                (vc * k + vso * v).StoreUnsafe(ref po, (nuint)i);
                (vc * v + vsc * k).StoreUnsafe(ref pc, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            double k = Unsafe.Add(ref pc, i);
            double v = Unsafe.Add(ref pi, i);
            Unsafe.Add(ref po, i) = c * k + so * v;
            Unsafe.Add(ref pc, i) = c * v + sc * k;
        }
    }

    // 平方和 Σ x_i^2（Dot(x, x) とビット一致、各要素 1 回のみ読み込み）。
    internal static double SumOfSquares(ReadOnlySpan<double> x)
    {
        int n = x.Length;
        ref double px = ref MemoryMarshal.GetReference(x);
        int i = 0;
        double sum = 0;
        if (Vector512.IsHardwareAccelerated && n >= 8)
        {
            var s0 = Vector512<double>.Zero;
            var s1 = Vector512<double>.Zero;
            var s2 = Vector512<double>.Zero;
            var s3 = Vector512<double>.Zero;
            for (; i <= n - 32; i += 32)
            {
                var v0 = Vector512.LoadUnsafe(ref px, (nuint)i);
                var v1 = Vector512.LoadUnsafe(ref px, (nuint)(i + 8));
                var v2 = Vector512.LoadUnsafe(ref px, (nuint)(i + 16));
                var v3 = Vector512.LoadUnsafe(ref px, (nuint)(i + 24));
                s0 = Vector512.FusedMultiplyAdd(v0, v0, s0);
                s1 = Vector512.FusedMultiplyAdd(v1, v1, s1);
                s2 = Vector512.FusedMultiplyAdd(v2, v2, s2);
                s3 = Vector512.FusedMultiplyAdd(v3, v3, s3);
            }

            for (; i <= n - 8; i += 8)
            {
                var v = Vector512.LoadUnsafe(ref px, (nuint)i);
                s0 = Vector512.FusedMultiplyAdd(v, v, s0);
            }

            sum = Vector512.Sum((s0 + s1) + (s2 + s3));
        }
        else if (Vector256.IsHardwareAccelerated && n >= 4)
        {
            var s0 = Vector256<double>.Zero;
            var s1 = Vector256<double>.Zero;
            var s2 = Vector256<double>.Zero;
            var s3 = Vector256<double>.Zero;
            for (; i <= n - 16; i += 16)
            {
                var v0 = Vector256.LoadUnsafe(ref px, (nuint)i);
                var v1 = Vector256.LoadUnsafe(ref px, (nuint)(i + 4));
                var v2 = Vector256.LoadUnsafe(ref px, (nuint)(i + 8));
                var v3 = Vector256.LoadUnsafe(ref px, (nuint)(i + 12));
                s0 = Vector256.MultiplyAddEstimate(v0, v0, s0);
                s1 = Vector256.MultiplyAddEstimate(v1, v1, s1);
                s2 = Vector256.MultiplyAddEstimate(v2, v2, s2);
                s3 = Vector256.MultiplyAddEstimate(v3, v3, s3);
            }

            for (; i <= n - 4; i += 4)
            {
                var v = Vector256.LoadUnsafe(ref px, (nuint)i);
                s0 = Vector256.MultiplyAddEstimate(v, v, s0);
            }

            sum = Vector256.Sum((s0 + s1) + (s2 + s3));
        }

        for (; i < n; i++)
        {
            double v = Unsafe.Add(ref px, i);
            sum += v * v;
        }

        return sum;
    }

    // ユークリッドノルム ||x||_2（オーバーフロー・アンダーフロー回避）。
    internal static double Norm2(ReadOnlySpan<double> x) => Norm2(x, SumOfSquares(x));

    // 計算済み平方和 ss から ||x||_2 を求める（オーバーフロー・アンダーフロー範囲なら最大絶対値で再計算）。
    internal static double Norm2(ReadOnlySpan<double> x, double ss)
    {
        if (ss > SumSquaresLow && ss < SumSquaresHigh)
        {
            return Math.Sqrt(ss);
        }

        if (ss == 0 && !ContainsNonZero(x))
        {
            return 0;
        }

        if (double.IsNaN(ss))
        {
            return double.NaN;
        }

        // 最大絶対値で正規化してから平方和を取る（LAPACK dnrm2 と同等の安全な経路）
        double scale = 0;
        foreach (double v in x)
        {
            scale = Math.Max(scale, Math.Abs(v));
        }

        if (double.IsPositiveInfinity(scale))
        {
            return double.PositiveInfinity;
        }

        double inv = 1.0 / scale;
        double sum = 0;
        if (double.IsFinite(inv))
        {
            foreach (double v in x)
            {
                double t = v * inv;
                sum += t * t;
            }
        }
        else
        {
            foreach (double v in x)
            {
                double t = v / scale;
                sum += t * t;
            }
        }

        return scale * Math.Sqrt(sum);
    }

    private static bool ContainsNonZero(ReadOnlySpan<double> x)
    {
        foreach (double v in x)
        {
            if (v != 0)
            {
                return true;
            }
        }

        return false;
    }

    internal static int IndexOfMaxAbs(ReadOnlySpan<double> x)
    {
        int index = 0;
        double max = -1;
        for (int i = 0; i < x.Length; i++)
        {
            double v = Math.Abs(x[i]);
            if (v > max)
            {
                max = v;
                index = i;
            }
        }

        return index;
    }

    // B <- A^T（キャッシュ効率のためブロック転置）。
    internal static unsafe void Transpose(int m, int n, ReadOnlySpan<double> a, int lda, Span<double> b, int ldb)
    {
        if (m <= 0 || n <= 0)
        {
            return;
        }

        Debug.Assert(a.Length >= (n - 1) * lda + m && b.Length >= (m - 1) * ldb + n);
        fixed (double* pa = a)
        fixed (double* pb = b)
        {
            const int Block = 32;
            bool simd = Gemm.Use512;
            var one = Vector512.Create(1.0);

            // 処理した列数・行数だけ進める（j0 += Block と Math.Min(n, j0 + Block) は、n が int の上限に近いと負へ折り返す）
            for (int j0 = 0, jw; j0 < n; j0 += jw)
            {
                jw = Math.Min(Block, n - j0);
                int j1 = j0 + jw;
                for (int i0 = 0, iw; i0 < m; i0 += iw)
                {
                    iw = Math.Min(Block, m - i0);
                    int i1 = i0 + iw;
                    int j = j0;
                    if (simd)
                    {
                        // 8×8 のタイルをレジスタ内で転置する（列 j..j+7 の行 i..i+7 → b の列 i..i+7 の行 j..j+7）
                        for (; j <= j1 - 8; j += 8)
                        {
                            int i = i0;
                            for (; i <= i1 - 8; i += 8)
                            {
                                Gemm.Transpose8(pa + i + (nint)j * lda, lda, pb + j + (nint)i * ldb, ldb, one);
                            }

                            for (; i < i1; i++)
                            {
                                double* src = pa + i + (nint)j * lda;
                                double* dst = pb + j + (nint)i * ldb;
                                for (int jj = 0; jj < 8; jj++)
                                {
                                    dst[jj] = src[(nint)jj * lda];
                                }
                            }
                        }
                    }

                    for (; j < j1; j++)
                    {
                        double* src = pa + (nint)j * lda;
                        double* dst = pb + j;
                        for (int i = i0; i < i1; i++)
                        {
                            dst[(nint)i * ldb] = src[i];
                        }
                    }
                }
            }
        }
    }

    internal static void Swap(Span<double> x, Span<double> y)
    {
        Debug.Assert(y.Length >= x.Length);
        for (int i = 0; i < x.Length; i++)
        {
            (x[i], y[i]) = (y[i], x[i]);
        }
    }

    internal static void Copy(int m, int n, ReadOnlySpan<double> a, int lda, Span<double> b, int ldb)
    {
        for (int j = 0; j < n; j++)
        {
            a.Slice(j * lda, m).CopyTo(b.Slice(j * ldb, m));
        }
    }
}
