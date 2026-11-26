using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Tuna.LinearAlgebra;

// 実数型（float / double / プリミティブ整数型）の連続領域向け SIMD ベクトル演算（BLAS Level 1/2 内部カーネル）。
// 浮動小数点型は FMA 対応時に 1 回丸めの積和を行い、整数型は unchecked 相当の折り返し演算を行う。
internal static class RealVectorOps
{
    // 256/128 ビット FMA の利用可否（AVX-512 環境では 512 ビット FMA が常に利用可能）。
    private static bool HasFma => Fma.IsSupported || AdvSimd.IsSupported;

    // ---- 積和（浮動小数点型は FMA、他は積と和） ----

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector512<T> MulAdd<T>(Vector512<T> a, Vector512<T> b, Vector512<T> c)
    {
        if (typeof(T) == typeof(double))
        {
            return Vector512.FusedMultiplyAdd(a.AsDouble(), b.AsDouble(), c.AsDouble()).As<double, T>();
        }

        if (typeof(T) == typeof(float))
        {
            return Vector512.FusedMultiplyAdd(a.AsSingle(), b.AsSingle(), c.AsSingle()).As<float, T>();
        }

        return a * b + c;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<T> MulAdd<T>(Vector256<T> a, Vector256<T> b, Vector256<T> c)
    {
        if (typeof(T) == typeof(double) && HasFma)
        {
            return Vector256.FusedMultiplyAdd(a.AsDouble(), b.AsDouble(), c.AsDouble()).As<double, T>();
        }

        if (typeof(T) == typeof(float) && HasFma)
        {
            return Vector256.FusedMultiplyAdd(a.AsSingle(), b.AsSingle(), c.AsSingle()).As<float, T>();
        }

        return a * b + c;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector128<T> MulAdd<T>(Vector128<T> a, Vector128<T> b, Vector128<T> c)
    {
        if (typeof(T) == typeof(double) && HasFma)
        {
            return Vector128.FusedMultiplyAdd(a.AsDouble(), b.AsDouble(), c.AsDouble()).As<double, T>();
        }

        if (typeof(T) == typeof(float) && HasFma)
        {
            return Vector128.FusedMultiplyAdd(a.AsSingle(), b.AsSingle(), c.AsSingle()).As<float, T>();
        }

        return a * b + c;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T MulAdd<T>(T a, T b, T c)
        where T : INumberBase<T>
    {
        if (typeof(T) == typeof(double))
        {
            double r = Math.FusedMultiplyAdd(Unsafe.As<T, double>(ref a), Unsafe.As<T, double>(ref b), Unsafe.As<T, double>(ref c));
            return Unsafe.As<double, T>(ref r);
        }

        if (typeof(T) == typeof(float))
        {
            float r = MathF.FusedMultiplyAdd(Unsafe.As<T, float>(ref a), Unsafe.As<T, float>(ref b), Unsafe.As<T, float>(ref c));
            return Unsafe.As<float, T>(ref r);
        }

        return (a * b) + c;
    }

    // ---- レベル1 ----

    internal static T Dot<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> y)
        where T : unmanaged, INumberBase<T>
    {
        int n = x.Length;
        ref T px = ref MemoryMarshal.GetReference(x);
        ref T py = ref MemoryMarshal.GetReference(y);
        int i = 0;
        T sum = T.Zero;
        if (Vector512.IsHardwareAccelerated && Vector512<T>.IsSupported && n >= Vector512<T>.Count)
        {
            int w = Vector512<T>.Count;
            var a0 = Vector512<T>.Zero;
            var a1 = Vector512<T>.Zero;
            var a2 = Vector512<T>.Zero;
            var a3 = Vector512<T>.Zero;
            for (; i <= n - 4 * w; i += 4 * w)
            {
                a0 = MulAdd(Vector512.LoadUnsafe(ref px, (nuint)i), Vector512.LoadUnsafe(ref py, (nuint)i), a0);
                a1 = MulAdd(Vector512.LoadUnsafe(ref px, (nuint)(i + w)), Vector512.LoadUnsafe(ref py, (nuint)(i + w)), a1);
                a2 = MulAdd(Vector512.LoadUnsafe(ref px, (nuint)(i + 2 * w)), Vector512.LoadUnsafe(ref py, (nuint)(i + 2 * w)), a2);
                a3 = MulAdd(Vector512.LoadUnsafe(ref px, (nuint)(i + 3 * w)), Vector512.LoadUnsafe(ref py, (nuint)(i + 3 * w)), a3);
            }

            for (; i <= n - w; i += w)
            {
                a0 = MulAdd(Vector512.LoadUnsafe(ref px, (nuint)i), Vector512.LoadUnsafe(ref py, (nuint)i), a0);
            }

            sum = Vector512.Sum((a0 + a1) + (a2 + a3));
        }
        else if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && n >= Vector256<T>.Count)
        {
            int w = Vector256<T>.Count;
            var a0 = Vector256<T>.Zero;
            var a1 = Vector256<T>.Zero;
            var a2 = Vector256<T>.Zero;
            var a3 = Vector256<T>.Zero;
            for (; i <= n - 4 * w; i += 4 * w)
            {
                a0 = MulAdd(Vector256.LoadUnsafe(ref px, (nuint)i), Vector256.LoadUnsafe(ref py, (nuint)i), a0);
                a1 = MulAdd(Vector256.LoadUnsafe(ref px, (nuint)(i + w)), Vector256.LoadUnsafe(ref py, (nuint)(i + w)), a1);
                a2 = MulAdd(Vector256.LoadUnsafe(ref px, (nuint)(i + 2 * w)), Vector256.LoadUnsafe(ref py, (nuint)(i + 2 * w)), a2);
                a3 = MulAdd(Vector256.LoadUnsafe(ref px, (nuint)(i + 3 * w)), Vector256.LoadUnsafe(ref py, (nuint)(i + 3 * w)), a3);
            }

            for (; i <= n - w; i += w)
            {
                a0 = MulAdd(Vector256.LoadUnsafe(ref px, (nuint)i), Vector256.LoadUnsafe(ref py, (nuint)i), a0);
            }

            sum = Vector256.Sum((a0 + a1) + (a2 + a3));
        }
        else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && n >= Vector128<T>.Count)
        {
            int w = Vector128<T>.Count;
            var a0 = Vector128<T>.Zero;
            var a1 = Vector128<T>.Zero;
            for (; i <= n - 2 * w; i += 2 * w)
            {
                a0 = MulAdd(Vector128.LoadUnsafe(ref px, (nuint)i), Vector128.LoadUnsafe(ref py, (nuint)i), a0);
                a1 = MulAdd(Vector128.LoadUnsafe(ref px, (nuint)(i + w)), Vector128.LoadUnsafe(ref py, (nuint)(i + w)), a1);
            }

            for (; i <= n - w; i += w)
            {
                a0 = MulAdd(Vector128.LoadUnsafe(ref px, (nuint)i), Vector128.LoadUnsafe(ref py, (nuint)i), a0);
            }

            sum = Vector128.Sum(a0 + a1);
        }

        for (; i < n; i++)
        {
            sum = MulAdd(Unsafe.Add(ref px, i), Unsafe.Add(ref py, i), sum);
        }

        return sum;
    }

    internal static void Axpy<T>(T a, ReadOnlySpan<T> x, Span<T> y)
        where T : unmanaged, INumberBase<T>
    {
        int n = x.Length;
        ref T px = ref MemoryMarshal.GetReference(x);
        ref T py = ref MemoryMarshal.GetReference(y);
        int i = 0;
        if (Vector512.IsHardwareAccelerated && Vector512<T>.IsSupported && n >= Vector512<T>.Count)
        {
            int w = Vector512<T>.Count;
            var va = Vector512.Create(a);
            for (; i <= n - 2 * w; i += 2 * w)
            {
                MulAdd(va, Vector512.LoadUnsafe(ref px, (nuint)i), Vector512.LoadUnsafe(ref py, (nuint)i)).StoreUnsafe(ref py, (nuint)i);
                MulAdd(va, Vector512.LoadUnsafe(ref px, (nuint)(i + w)), Vector512.LoadUnsafe(ref py, (nuint)(i + w))).StoreUnsafe(ref py, (nuint)(i + w));
            }

            for (; i <= n - w; i += w)
            {
                MulAdd(va, Vector512.LoadUnsafe(ref px, (nuint)i), Vector512.LoadUnsafe(ref py, (nuint)i)).StoreUnsafe(ref py, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && n >= Vector256<T>.Count)
        {
            int w = Vector256<T>.Count;
            var va = Vector256.Create(a);
            for (; i <= n - 2 * w; i += 2 * w)
            {
                MulAdd(va, Vector256.LoadUnsafe(ref px, (nuint)i), Vector256.LoadUnsafe(ref py, (nuint)i)).StoreUnsafe(ref py, (nuint)i);
                MulAdd(va, Vector256.LoadUnsafe(ref px, (nuint)(i + w)), Vector256.LoadUnsafe(ref py, (nuint)(i + w))).StoreUnsafe(ref py, (nuint)(i + w));
            }

            for (; i <= n - w; i += w)
            {
                MulAdd(va, Vector256.LoadUnsafe(ref px, (nuint)i), Vector256.LoadUnsafe(ref py, (nuint)i)).StoreUnsafe(ref py, (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && n >= Vector128<T>.Count)
        {
            int w = Vector128<T>.Count;
            var va = Vector128.Create(a);
            for (; i <= n - w; i += w)
            {
                MulAdd(va, Vector128.LoadUnsafe(ref px, (nuint)i), Vector128.LoadUnsafe(ref py, (nuint)i)).StoreUnsafe(ref py, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            Unsafe.Add(ref py, i) = MulAdd(a, Unsafe.Add(ref px, i), Unsafe.Add(ref py, i));
        }
    }

    internal static void Scale<T>(T a, Span<T> x)
        where T : unmanaged, INumberBase<T>
    {
        int n = x.Length;
        ref T px = ref MemoryMarshal.GetReference(x);
        int i = 0;
        if (Vector512.IsHardwareAccelerated && Vector512<T>.IsSupported && n >= Vector512<T>.Count)
        {
            int w = Vector512<T>.Count;
            var va = Vector512.Create(a);
            for (; i <= n - w; i += w)
            {
                (va * Vector512.LoadUnsafe(ref px, (nuint)i)).StoreUnsafe(ref px, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && n >= Vector256<T>.Count)
        {
            int w = Vector256<T>.Count;
            var va = Vector256.Create(a);
            for (; i <= n - w; i += w)
            {
                (va * Vector256.LoadUnsafe(ref px, (nuint)i)).StoreUnsafe(ref px, (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && n >= Vector128<T>.Count)
        {
            int w = Vector128<T>.Count;
            var va = Vector128.Create(a);
            for (; i <= n - w; i += w)
            {
                (va * Vector128.LoadUnsafe(ref px, (nuint)i)).StoreUnsafe(ref px, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            Unsafe.Add(ref px, i) = a * Unsafe.Add(ref px, i);
        }
    }

    internal static T SumAbs<T>(ReadOnlySpan<T> x)
        where T : unmanaged, INumberBase<T>
    {
        int n = x.Length;
        ref T px = ref MemoryMarshal.GetReference(x);
        int i = 0;
        T sum = T.Zero;
        if (Vector512.IsHardwareAccelerated && Vector512<T>.IsSupported && n >= Vector512<T>.Count)
        {
            int w = Vector512<T>.Count;
            var a0 = Vector512<T>.Zero;
            var a1 = Vector512<T>.Zero;
            for (; i <= n - 2 * w; i += 2 * w)
            {
                a0 += Vector512.Abs(Vector512.LoadUnsafe(ref px, (nuint)i));
                a1 += Vector512.Abs(Vector512.LoadUnsafe(ref px, (nuint)(i + w)));
            }

            for (; i <= n - w; i += w)
            {
                a0 += Vector512.Abs(Vector512.LoadUnsafe(ref px, (nuint)i));
            }

            sum = Vector512.Sum(a0 + a1);
        }
        else if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && n >= Vector256<T>.Count)
        {
            int w = Vector256<T>.Count;
            var a0 = Vector256<T>.Zero;
            var a1 = Vector256<T>.Zero;
            for (; i <= n - 2 * w; i += 2 * w)
            {
                a0 += Vector256.Abs(Vector256.LoadUnsafe(ref px, (nuint)i));
                a1 += Vector256.Abs(Vector256.LoadUnsafe(ref px, (nuint)(i + w)));
            }

            for (; i <= n - w; i += w)
            {
                a0 += Vector256.Abs(Vector256.LoadUnsafe(ref px, (nuint)i));
            }

            sum = Vector256.Sum(a0 + a1);
        }
        else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && n >= Vector128<T>.Count)
        {
            int w = Vector128<T>.Count;
            var a0 = Vector128<T>.Zero;
            for (; i <= n - w; i += w)
            {
                a0 += Vector128.Abs(Vector128.LoadUnsafe(ref px, (nuint)i));
            }

            sum = Vector128.Sum(a0);
        }

        for (; i < n; i++)
        {
            sum += T.Abs(Unsafe.Add(ref px, i));
        }

        return sum;
    }

    // 絶対値が最大の最初のインデックス（空なら -1、NaN を含む場合は最初の NaN）。
    internal static int IndexOfMaxAbs<T>(ReadOnlySpan<T> x)
        where T : unmanaged, INumberBase<T>
    {
        int n = x.Length;
        if (n == 0)
        {
            return -1;
        }

        // 最大値を SIMD で求め、その値を持つ最初の添字を探す（NaN は別扱い）
        ref T px = ref MemoryMarshal.GetReference(x);
        int i = 0;
        T max = T.Zero;
        if (Vector512.IsHardwareAccelerated && Vector512<T>.IsSupported && n >= Vector512<T>.Count)
        {
            int w = Vector512<T>.Count;
            var vm = Vector512<T>.Zero;
            for (; i <= n - w; i += w)
            {
                vm = Vector512.Max(vm, Vector512.Abs(Vector512.LoadUnsafe(ref px, (nuint)i)));
            }

            for (int k = 0; k < w; k++)
            {
                max = T.MaxMagnitudeNumber(max, vm.GetElement(k));
            }
        }
        else if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && n >= Vector256<T>.Count)
        {
            int w = Vector256<T>.Count;
            var vm = Vector256<T>.Zero;
            for (; i <= n - w; i += w)
            {
                vm = Vector256.Max(vm, Vector256.Abs(Vector256.LoadUnsafe(ref px, (nuint)i)));
            }

            for (int k = 0; k < w; k++)
            {
                max = T.MaxMagnitudeNumber(max, vm.GetElement(k));
            }
        }

        for (; i < n; i++)
        {
            max = T.MaxMagnitudeNumber(max, T.Abs(Unsafe.Add(ref px, i)));
        }

        for (int j = 0; j < n; j++)
        {
            T v = Unsafe.Add(ref px, j);
            if (T.IsNaN(v))
            {
                return j;
            }

            if (T.Abs(v) == max)
            {
                return j;
            }
        }

        return 0;
    }

    // 平面回転 (x, y) <- (c*x + s*y, c*y - s*x)。
    internal static void Rotate<T>(Span<T> x, Span<T> y, T c, T s)
        where T : unmanaged, INumberBase<T>
    {
        int n = x.Length;
        ref T px = ref MemoryMarshal.GetReference(x);
        ref T py = ref MemoryMarshal.GetReference(y);
        int i = 0;
        if (Vector512.IsHardwareAccelerated && Vector512<T>.IsSupported && n >= Vector512<T>.Count)
        {
            int w = Vector512<T>.Count;
            var vc = Vector512.Create(c);
            var vs = Vector512.Create(s);
            for (; i <= n - w; i += w)
            {
                var xv = Vector512.LoadUnsafe(ref px, (nuint)i);
                var yv = Vector512.LoadUnsafe(ref py, (nuint)i);
                MulAdd(vs, yv, vc * xv).StoreUnsafe(ref px, (nuint)i);
                (vc * yv - vs * xv).StoreUnsafe(ref py, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && n >= Vector256<T>.Count)
        {
            int w = Vector256<T>.Count;
            var vc = Vector256.Create(c);
            var vs = Vector256.Create(s);
            for (; i <= n - w; i += w)
            {
                var xv = Vector256.LoadUnsafe(ref px, (nuint)i);
                var yv = Vector256.LoadUnsafe(ref py, (nuint)i);
                MulAdd(vs, yv, vc * xv).StoreUnsafe(ref px, (nuint)i);
                (vc * yv - vs * xv).StoreUnsafe(ref py, (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && n >= Vector128<T>.Count)
        {
            int w = Vector128<T>.Count;
            var vc = Vector128.Create(c);
            var vs = Vector128.Create(s);
            for (; i <= n - w; i += w)
            {
                var xv = Vector128.LoadUnsafe(ref px, (nuint)i);
                var yv = Vector128.LoadUnsafe(ref py, (nuint)i);
                MulAdd(vs, yv, vc * xv).StoreUnsafe(ref px, (nuint)i);
                (vc * yv - vs * xv).StoreUnsafe(ref py, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            T xi = Unsafe.Add(ref px, i);
            T yi = Unsafe.Add(ref py, i);
            Unsafe.Add(ref px, i) = MulAdd(s, yi, c * xi);
            Unsafe.Add(ref py, i) = (c * yi) - (s * xi);
        }
    }

    // ---- Level 2 用融合カーネル ----

    // y += a * col と Σ col[i] * x[i] を 1 パスで同時計算する（対称行列ベクトル積用）。
    internal static T AxpyDot<T>(T a, ReadOnlySpan<T> col, ReadOnlySpan<T> x, Span<T> y)
        where T : unmanaged, INumberBase<T>
    {
        int n = col.Length;
        ref T pc = ref MemoryMarshal.GetReference(col);
        ref T px = ref MemoryMarshal.GetReference(x);
        ref T py = ref MemoryMarshal.GetReference(y);
        int i = 0;
        T sum = T.Zero;
        if (Vector512.IsHardwareAccelerated && Vector512<T>.IsSupported && n >= Vector512<T>.Count)
        {
            int w = Vector512<T>.Count;
            var va = Vector512.Create(a);
            var s0 = Vector512<T>.Zero;
            var s1 = Vector512<T>.Zero;
            for (; i <= n - 2 * w; i += 2 * w)
            {
                var c0 = Vector512.LoadUnsafe(ref pc, (nuint)i);
                var c1 = Vector512.LoadUnsafe(ref pc, (nuint)(i + w));
                MulAdd(va, c0, Vector512.LoadUnsafe(ref py, (nuint)i)).StoreUnsafe(ref py, (nuint)i);
                MulAdd(va, c1, Vector512.LoadUnsafe(ref py, (nuint)(i + w))).StoreUnsafe(ref py, (nuint)(i + w));
                s0 = MulAdd(c0, Vector512.LoadUnsafe(ref px, (nuint)i), s0);
                s1 = MulAdd(c1, Vector512.LoadUnsafe(ref px, (nuint)(i + w)), s1);
            }

            for (; i <= n - w; i += w)
            {
                var c0 = Vector512.LoadUnsafe(ref pc, (nuint)i);
                MulAdd(va, c0, Vector512.LoadUnsafe(ref py, (nuint)i)).StoreUnsafe(ref py, (nuint)i);
                s0 = MulAdd(c0, Vector512.LoadUnsafe(ref px, (nuint)i), s0);
            }

            sum = Vector512.Sum(s0 + s1);
        }
        else if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && n >= Vector256<T>.Count)
        {
            int w = Vector256<T>.Count;
            var va = Vector256.Create(a);
            var s0 = Vector256<T>.Zero;
            var s1 = Vector256<T>.Zero;
            for (; i <= n - 2 * w; i += 2 * w)
            {
                var c0 = Vector256.LoadUnsafe(ref pc, (nuint)i);
                var c1 = Vector256.LoadUnsafe(ref pc, (nuint)(i + w));
                MulAdd(va, c0, Vector256.LoadUnsafe(ref py, (nuint)i)).StoreUnsafe(ref py, (nuint)i);
                MulAdd(va, c1, Vector256.LoadUnsafe(ref py, (nuint)(i + w))).StoreUnsafe(ref py, (nuint)(i + w));
                s0 = MulAdd(c0, Vector256.LoadUnsafe(ref px, (nuint)i), s0);
                s1 = MulAdd(c1, Vector256.LoadUnsafe(ref px, (nuint)(i + w)), s1);
            }

            for (; i <= n - w; i += w)
            {
                var c0 = Vector256.LoadUnsafe(ref pc, (nuint)i);
                MulAdd(va, c0, Vector256.LoadUnsafe(ref py, (nuint)i)).StoreUnsafe(ref py, (nuint)i);
                s0 = MulAdd(c0, Vector256.LoadUnsafe(ref px, (nuint)i), s0);
            }

            sum = Vector256.Sum(s0 + s1);
        }
        else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && n >= Vector128<T>.Count)
        {
            int w = Vector128<T>.Count;
            var va = Vector128.Create(a);
            var s0 = Vector128<T>.Zero;
            for (; i <= n - w; i += w)
            {
                var c0 = Vector128.LoadUnsafe(ref pc, (nuint)i);
                MulAdd(va, c0, Vector128.LoadUnsafe(ref py, (nuint)i)).StoreUnsafe(ref py, (nuint)i);
                s0 = MulAdd(c0, Vector128.LoadUnsafe(ref px, (nuint)i), s0);
            }

            sum = Vector128.Sum(s0);
        }

        for (; i < n; i++)
        {
            T c = Unsafe.Add(ref pc, i);
            Unsafe.Add(ref py, i) = MulAdd(a, c, Unsafe.Add(ref py, i));
            sum = MulAdd(c, Unsafe.Add(ref px, i), sum);
        }

        return sum;
    }

    // 4 列の内積 r_k = Σ A[i, k] * x[i]（x の読み込みを 4 列で共有）。
    internal static unsafe void Dot4<T>(int n, T* a, nint lda, T* x, out T r0, out T r1, out T r2, out T r3)
        where T : unmanaged, INumberBase<T>
    {
        T* c0 = a;
        T* c1 = a + lda;
        T* c2 = a + (2 * lda);
        T* c3 = a + (3 * lda);
        int i = 0;
        T s0 = T.Zero, s1 = T.Zero, s2 = T.Zero, s3 = T.Zero;
        if (Vector512.IsHardwareAccelerated && Vector512<T>.IsSupported && n >= Vector512<T>.Count)
        {
            int w = Vector512<T>.Count;
            Vector512<T> a0 = default, a1 = default, a2 = default, a3 = default;
            for (; i <= n - w; i += w)
            {
                var xv = Vector512.Load(x + i);
                a0 = MulAdd(Vector512.Load(c0 + i), xv, a0);
                a1 = MulAdd(Vector512.Load(c1 + i), xv, a1);
                a2 = MulAdd(Vector512.Load(c2 + i), xv, a2);
                a3 = MulAdd(Vector512.Load(c3 + i), xv, a3);
            }

            s0 = Vector512.Sum(a0);
            s1 = Vector512.Sum(a1);
            s2 = Vector512.Sum(a2);
            s3 = Vector512.Sum(a3);
        }
        else if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && n >= Vector256<T>.Count)
        {
            int w = Vector256<T>.Count;
            Vector256<T> a0 = default, a1 = default, a2 = default, a3 = default;
            for (; i <= n - w; i += w)
            {
                var xv = Vector256.Load(x + i);
                a0 = MulAdd(Vector256.Load(c0 + i), xv, a0);
                a1 = MulAdd(Vector256.Load(c1 + i), xv, a1);
                a2 = MulAdd(Vector256.Load(c2 + i), xv, a2);
                a3 = MulAdd(Vector256.Load(c3 + i), xv, a3);
            }

            s0 = Vector256.Sum(a0);
            s1 = Vector256.Sum(a1);
            s2 = Vector256.Sum(a2);
            s3 = Vector256.Sum(a3);
        }
        else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && n >= Vector128<T>.Count)
        {
            int w = Vector128<T>.Count;
            Vector128<T> a0 = default, a1 = default, a2 = default, a3 = default;
            for (; i <= n - w; i += w)
            {
                var xv = Vector128.Load(x + i);
                a0 = MulAdd(Vector128.Load(c0 + i), xv, a0);
                a1 = MulAdd(Vector128.Load(c1 + i), xv, a1);
                a2 = MulAdd(Vector128.Load(c2 + i), xv, a2);
                a3 = MulAdd(Vector128.Load(c3 + i), xv, a3);
            }

            s0 = Vector128.Sum(a0);
            s1 = Vector128.Sum(a1);
            s2 = Vector128.Sum(a2);
            s3 = Vector128.Sum(a3);
        }

        for (; i < n; i++)
        {
            T xi = x[i];
            s0 = MulAdd(c0[i], xi, s0);
            s1 = MulAdd(c1[i], xi, s1);
            s2 = MulAdd(c2[i], xi, s2);
            s3 = MulAdd(c3[i], xi, s3);
        }

        r0 = s0;
        r1 = s1;
        r2 = s2;
        r3 = s3;
    }

    // y += Σ_{k=0..3} t_k * A[:, k]（y の読み書きを 4 列で共有）。
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static unsafe void AddColumns4<T>(int n, T* a, nint lda, T t0, T t1, T t2, T t3, T* y)
        where T : unmanaged, INumberBase<T>
    {
        T* c0 = a;
        T* c1 = a + lda;
        T* c2 = a + (2 * lda);
        T* c3 = a + (3 * lda);
        int i = 0;
        if (Vector512.IsHardwareAccelerated && Vector512<T>.IsSupported && n >= Vector512<T>.Count)
        {
            int w = Vector512<T>.Count;
            var v0 = Vector512.Create(t0);
            var v1 = Vector512.Create(t1);
            var v2 = Vector512.Create(t2);
            var v3 = Vector512.Create(t3);
            for (; i <= n - w; i += w)
            {
                var acc = Vector512.Load(y + i);
                acc = MulAdd(v0, Vector512.Load(c0 + i), acc);
                acc = MulAdd(v1, Vector512.Load(c1 + i), acc);
                acc = MulAdd(v2, Vector512.Load(c2 + i), acc);
                acc = MulAdd(v3, Vector512.Load(c3 + i), acc);
                acc.Store(y + i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && n >= Vector256<T>.Count)
        {
            int w = Vector256<T>.Count;
            var v0 = Vector256.Create(t0);
            var v1 = Vector256.Create(t1);
            var v2 = Vector256.Create(t2);
            var v3 = Vector256.Create(t3);
            for (; i <= n - w; i += w)
            {
                var acc = Vector256.Load(y + i);
                acc = MulAdd(v0, Vector256.Load(c0 + i), acc);
                acc = MulAdd(v1, Vector256.Load(c1 + i), acc);
                acc = MulAdd(v2, Vector256.Load(c2 + i), acc);
                acc = MulAdd(v3, Vector256.Load(c3 + i), acc);
                acc.Store(y + i);
            }
        }
        else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && n >= Vector128<T>.Count)
        {
            int w = Vector128<T>.Count;
            var v0 = Vector128.Create(t0);
            var v1 = Vector128.Create(t1);
            var v2 = Vector128.Create(t2);
            var v3 = Vector128.Create(t3);
            for (; i <= n - w; i += w)
            {
                var acc = Vector128.Load(y + i);
                acc = MulAdd(v0, Vector128.Load(c0 + i), acc);
                acc = MulAdd(v1, Vector128.Load(c1 + i), acc);
                acc = MulAdd(v2, Vector128.Load(c2 + i), acc);
                acc = MulAdd(v3, Vector128.Load(c3 + i), acc);
                acc.Store(y + i);
            }
        }

        for (; i < n; i++)
        {
            T acc = y[i];
            acc = MulAdd(t0, c0[i], acc);
            acc = MulAdd(t1, c1[i], acc);
            acc = MulAdd(t2, c2[i], acc);
            acc = MulAdd(t3, c3[i], acc);
            y[i] = acc;
        }
    }

    // 2 列分の AxpyDot を融合し、x と y のメモリアクセスを共有する。
    internal static unsafe void AxpyDot2<T>(int n, T ta, T* ca, T tb, T* cb, T* x, T* y, out T sa, out T sb)
        where T : unmanaged, INumberBase<T>
    {
        int i = 0;
        T s0 = T.Zero, s1 = T.Zero;
        if (Vector512.IsHardwareAccelerated && Vector512<T>.IsSupported && n >= Vector512<T>.Count)
        {
            int w = Vector512<T>.Count;
            var va = Vector512.Create(ta);
            var vb = Vector512.Create(tb);
            var a0 = Vector512<T>.Zero;
            var a1 = Vector512<T>.Zero;
            for (; i <= n - w; i += w)
            {
                var c0 = Vector512.Load(ca + i);
                var c1 = Vector512.Load(cb + i);
                var xv = Vector512.Load(x + i);
                MulAdd(vb, c1, MulAdd(va, c0, Vector512.Load(y + i))).Store(y + i);
                a0 = MulAdd(c0, xv, a0);
                a1 = MulAdd(c1, xv, a1);
            }

            s0 = Vector512.Sum(a0);
            s1 = Vector512.Sum(a1);
        }
        else if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && n >= Vector256<T>.Count)
        {
            int w = Vector256<T>.Count;
            var va = Vector256.Create(ta);
            var vb = Vector256.Create(tb);
            var a0 = Vector256<T>.Zero;
            var a1 = Vector256<T>.Zero;
            for (; i <= n - w; i += w)
            {
                var c0 = Vector256.Load(ca + i);
                var c1 = Vector256.Load(cb + i);
                var xv = Vector256.Load(x + i);
                MulAdd(vb, c1, MulAdd(va, c0, Vector256.Load(y + i))).Store(y + i);
                a0 = MulAdd(c0, xv, a0);
                a1 = MulAdd(c1, xv, a1);
            }

            s0 = Vector256.Sum(a0);
            s1 = Vector256.Sum(a1);
        }
        else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && n >= Vector128<T>.Count)
        {
            int w = Vector128<T>.Count;
            var va = Vector128.Create(ta);
            var vb = Vector128.Create(tb);
            var a0 = Vector128<T>.Zero;
            var a1 = Vector128<T>.Zero;
            for (; i <= n - w; i += w)
            {
                var c0 = Vector128.Load(ca + i);
                var c1 = Vector128.Load(cb + i);
                var xv = Vector128.Load(x + i);
                MulAdd(vb, c1, MulAdd(va, c0, Vector128.Load(y + i))).Store(y + i);
                a0 = MulAdd(c0, xv, a0);
                a1 = MulAdd(c1, xv, a1);
            }

            s0 = Vector128.Sum(a0);
            s1 = Vector128.Sum(a1);
        }

        for (; i < n; i++)
        {
            T c0 = ca[i];
            T c1 = cb[i];
            y[i] = MulAdd(tb, c1, MulAdd(ta, c0, y[i]));
            s0 = MulAdd(c0, x[i], s0);
            s1 = MulAdd(c1, x[i], s1);
        }

        sa = s0;
        sb = s1;
    }
}
