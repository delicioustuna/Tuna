using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Tuna.LinearAlgebra;

internal static partial class DenseReductions
{
    // ---------- 位置の検索 ----------

    // 最初の NaN の位置（なければ -1）
    internal static int IndexOfNaN<T>(ReadOnlySpan<T> x) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ref T r = ref MemoryMarshal.GetReference(x);
        int n = x.Length, i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            for (int width = Vector512<T>.Count; i <= n - width; i += width)
            {
                var v = Vector512.LoadUnsafe(ref r, (nuint)i);
                ulong nan = (~Vector512.Equals(v, v)).ExtractMostSignificantBits();
                if (nan != 0)
                {
                    return i + BitOperations.TrailingZeroCount(nan);
                }
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            for (int width = Vector256<T>.Count; i <= n - width; i += width)
            {
                var v = Vector256.LoadUnsafe(ref r, (nuint)i);
                uint nan = (~Vector256.Equals(v, v)).ExtractMostSignificantBits();
                if (nan != 0)
                {
                    return i + BitOperations.TrailingZeroCount(nan);
                }
            }
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            for (int width = Vector128<T>.Count; i <= n - width; i += width)
            {
                var v = Vector128.LoadUnsafe(ref r, (nuint)i);
                uint nan = (~Vector128.Equals(v, v)).ExtractMostSignificantBits();
                if (nan != 0)
                {
                    return i + BitOperations.TrailingZeroCount(nan);
                }
            }
        }

        for (; i < n; i++)
        {
            if (T.IsNaN(Unsafe.Add(ref r, i)))
            {
                return i;
            }
        }

        return -1;
    }

    private static int CountNaN<T>(ReadOnlySpan<T> x) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ref T r = ref MemoryMarshal.GetReference(x);
        int n = x.Length, i = 0, count = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            for (int width = Vector512<T>.Count; i <= n - width; i += width)
            {
                var v = Vector512.LoadUnsafe(ref r, (nuint)i);
                count += BitOperations.PopCount((~Vector512.Equals(v, v)).ExtractMostSignificantBits());
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            for (int width = Vector256<T>.Count; i <= n - width; i += width)
            {
                var v = Vector256.LoadUnsafe(ref r, (nuint)i);
                count += BitOperations.PopCount((~Vector256.Equals(v, v)).ExtractMostSignificantBits());
            }
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            for (int width = Vector128<T>.Count; i <= n - width; i += width)
            {
                var v = Vector128.LoadUnsafe(ref r, (nuint)i);
                count += BitOperations.PopCount((~Vector128.Equals(v, v)).ExtractMostSignificantBits());
            }
        }

        for (; i < n; i++)
        {
            if (T.IsNaN(Unsafe.Add(ref r, i)))
            {
                count++;
            }
        }

        return count;
    }
}
