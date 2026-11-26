using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Tuna.LinearAlgebra;

// 連続領域に対する要素ごとの SIMD 演算。BLAS の実装や契約とは独立させる。
internal static partial class DenseElementwise
{
    // 出力先は入力と完全一致してよい（各要素を読んでから書き込む）。部分的な重なりは呼び出し側で除外する。
    internal static void Add<T>(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> result, bool subtract)
        where T : unmanaged, INumberBase<T>
    {
        DenseStorage.RequireLength(left.Length, right.Length);
        DenseStorage.RequireLength(left.Length, result.Length);
        if (typeof(T) == typeof(Complex))
        {
            // 加減算は実部・虚部ごと。複素乗算に由来する 0 * Infinity を作らない。
            Binary<double>(MemoryMarshal.Cast<T, double>(left), MemoryMarshal.Cast<T, double>(right),
                MemoryMarshal.Cast<T, double>(result), subtract);
        }
        else
        {
            Binary<T>(left, right, result, subtract);
        }
    }

    // SIMD 演算のみの直接ループなので、PGO を活かすため AggressiveOptimization は付けない。
    private static void Binary<T>(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> result, bool subtract)
        where T : unmanaged, INumberBase<T>
    {
        ref T a = ref MemoryMarshal.GetReference(left);
        ref T b = ref MemoryMarshal.GetReference(right);
        ref T c = ref MemoryMarshal.GetReference(result);
        int n = left.Length;
        int i = 0;
        if (Vector512.IsHardwareAccelerated && Vector512<T>.IsSupported)
        {
            int width = Vector512<T>.Count;
            for (; i <= n - width; i += width)
            {
                var x = Vector512.LoadUnsafe(ref a, (nuint)i);
                var y = Vector512.LoadUnsafe(ref b, (nuint)i);
                (subtract ? x - y : x + y).StoreUnsafe(ref c, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported)
        {
            int width = Vector256<T>.Count;
            for (; i <= n - width; i += width)
            {
                var x = Vector256.LoadUnsafe(ref a, (nuint)i);
                var y = Vector256.LoadUnsafe(ref b, (nuint)i);
                (subtract ? x - y : x + y).StoreUnsafe(ref c, (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported)
        {
            int width = Vector128<T>.Count;
            for (; i <= n - width; i += width)
            {
                var x = Vector128.LoadUnsafe(ref a, (nuint)i);
                var y = Vector128.LoadUnsafe(ref b, (nuint)i);
                (subtract ? x - y : x + y).StoreUnsafe(ref c, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            result[i] = subtract ? unchecked(left[i] - right[i]) : unchecked(left[i] + right[i]);
        }
    }

    internal static void Scale<T>(ReadOnlySpan<T> source, T scalar, Span<T> result)
        where T : unmanaged, INumberBase<T>
    {
        DenseStorage.RequireLength(source.Length, result.Length);
        if (typeof(T) == typeof(Complex))
        {
            ScaleComplex(MemoryMarshal.Cast<T, Complex>(source), Unsafe.As<T, Complex>(ref scalar), MemoryMarshal.Cast<T, Complex>(result));
            return;
        }

        ref T a = ref MemoryMarshal.GetReference(source);
        ref T c = ref MemoryMarshal.GetReference(result);
        int n = source.Length;
        int i = 0;
        if (Vector512.IsHardwareAccelerated && Vector512<T>.IsSupported)
        {
            int width = Vector512<T>.Count;
            var scale = Vector512.Create(scalar);
            for (; i <= n - width; i += width)
            {
                (Vector512.LoadUnsafe(ref a, (nuint)i) * scale).StoreUnsafe(ref c, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported)
        {
            int width = Vector256<T>.Count;
            var scale = Vector256.Create(scalar);
            for (; i <= n - width; i += width)
            {
                (Vector256.LoadUnsafe(ref a, (nuint)i) * scale).StoreUnsafe(ref c, (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported)
        {
            int width = Vector128<T>.Count;
            var scale = Vector128.Create(scalar);
            for (; i <= n - width; i += width)
            {
                (Vector128.LoadUnsafe(ref a, (nuint)i) * scale).StoreUnsafe(ref c, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            result[i] = unchecked(source[i] * scalar);
        }
    }

    // 軟判定しきい値 sign(x) * max(|x| - threshold, 0)。threshold >= 0 は呼び出し側で保証する。
    internal static void SoftThreshold<T>(ReadOnlySpan<T> source, T threshold, Span<T> result)
        where T : unmanaged, INumber<T>
    {
        DenseStorage.RequireLength(source.Length, result.Length);
        SoftThresholdLines(in MemoryMarshal.GetReference(source), 0, threshold, ref MemoryMarshal.GetReference(result), 0, source.Length, 1);
    }

    // 選択されないレーンの x ± threshold はオーバーフローしても破棄される。SoftThresholdOperator<T> とビット単位で一致させる。
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void SoftThresholdLines<T>(ref readonly T source, int sourceStride, T threshold, ref T result, int resultStride, int length, int count)
        where T : unmanaged, INumber<T>
    {
        var op = new SoftThresholdOperator<T>(threshold);
        for (int k = 0; k < count; k++)
        {
            ref T a = ref Unsafe.Add(ref Unsafe.AsRef(in source), (nint)k * sourceStride);
            ref T c = ref Unsafe.Add(ref result, (nint)k * resultStride);
            int i = 0;
            if (Vector512.IsHardwareAccelerated && Vector512<T>.IsSupported)
            {
                int width = Vector512<T>.Count;
                var t = Vector512.Create(threshold);
                for (; i <= length - width; i += width)
                {
                    var x = Vector512.LoadUnsafe(ref a, (nuint)i);
                    var below = Vector512.IsNegative(x) & Vector512.LessThan(x + t, Vector512<T>.Zero);
                    var rest = Vector512.ConditionalSelect(below, x + t, x & Vector512.IsNaN(x));
                    Vector512.ConditionalSelect(Vector512.GreaterThan(x, t), x - t, rest).StoreUnsafe(ref c, (nuint)i);
                }
            }
            else if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported)
            {
                int width = Vector256<T>.Count;
                var t = Vector256.Create(threshold);
                for (; i <= length - width; i += width)
                {
                    var x = Vector256.LoadUnsafe(ref a, (nuint)i);
                    var below = Vector256.IsNegative(x) & Vector256.LessThan(x + t, Vector256<T>.Zero);
                    var rest = Vector256.ConditionalSelect(below, x + t, x & Vector256.IsNaN(x));
                    Vector256.ConditionalSelect(Vector256.GreaterThan(x, t), x - t, rest).StoreUnsafe(ref c, (nuint)i);
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported)
            {
                int width = Vector128<T>.Count;
                var t = Vector128.Create(threshold);
                for (; i <= length - width; i += width)
                {
                    var x = Vector128.LoadUnsafe(ref a, (nuint)i);
                    var below = Vector128.IsNegative(x) & Vector128.LessThan(x + t, Vector128<T>.Zero);
                    var rest = Vector128.ConditionalSelect(below, x + t, x & Vector128.IsNaN(x));
                    Vector128.ConditionalSelect(Vector128.GreaterThan(x, t), x - t, rest).StoreUnsafe(ref c, (nuint)i);
                }
            }

            // 端と SIMD 非対応の型（decimal・Half など）
            for (; i < length; i++)
            {
                Unsafe.Add(ref c, i) = op.Invoke(Unsafe.Add(ref a, i));
            }
        }
    }
}
