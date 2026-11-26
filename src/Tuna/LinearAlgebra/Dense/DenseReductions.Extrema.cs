using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Tuna.LinearAlgebra;

internal static partial class DenseReductions
{
    // ---------- 最大・最小 ----------

    private static bool TryExtremumReal<T, TOperator>(ReadOnlySpan<T> x, out T result) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (typeof(TOperator) == typeof(MaximumOperator<T>))
        {
            result = Reduce<T, MaximumKernel<T>>(x[1..], x[0]);
        }
        else if (typeof(TOperator) == typeof(MinimumOperator<T>))
        {
            result = Reduce<T, MinimumKernel<T>>(x[1..], x[0]);
        }
        else if (typeof(TOperator) == typeof(MaximumNumberOperator<T>))
        {
            result = Reduce<T, MaximumNumberKernel<T>>(x[1..], x[0]);
        }
        else if (typeof(TOperator) == typeof(MinimumNumberOperator<T>))
        {
            result = Reduce<T, MinimumNumberKernel<T>>(x[1..], x[0]);
        }
        else
        {
            result = T.Zero;
            return false;
        }

        return true;
    }

    // initial に x[0], x[1], … を順に TKernel で畳み込んだ値。TKernel は順序によらず一意な結果を返す演算であること。
    // ベクトルで畳み込んだ後、レーンを順に畳み込み、最後に端数を畳み込む
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static T Reduce<T, TKernel>(ReadOnlySpan<T> x, T initial)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TKernel : IFoldKernel<T>
    {
        int threads = ParallelThreads(x.Length);
        if (threads > 1)
        {
            return ReduceParallel<T, TKernel>(x, initial, threads);
        }

        ref T r = ref MemoryMarshal.GetReference(x);
        int n = x.Length, i = 0;
        T result = initial;
        if (Vector512.IsHardwareAccelerated && n >= 4 * Vector512<T>.Count)
        {
            int width = Vector512<T>.Count;
            var m0 = Vector512.Create(initial);
            var m1 = m0;
            var m2 = m0;
            var m3 = m0;
            for (; i <= n - 4 * width; i += 4 * width)
            {
                m0 = TKernel.Invoke(m0, Vector512.LoadUnsafe(ref r, (nuint)i));
                m1 = TKernel.Invoke(m1, Vector512.LoadUnsafe(ref r, (nuint)(i + width)));
                m2 = TKernel.Invoke(m2, Vector512.LoadUnsafe(ref r, (nuint)(i + 2 * width)));
                m3 = TKernel.Invoke(m3, Vector512.LoadUnsafe(ref r, (nuint)(i + 3 * width)));
            }

            var m = TKernel.Invoke(TKernel.Invoke(m0, m1), TKernel.Invoke(m2, m3));
            for (int k = 0; k < width; k++)
            {
                result = TKernel.Invoke(result, m.GetElement(k));
            }
        }
        else if (Vector256.IsHardwareAccelerated && n >= 4 * Vector256<T>.Count)
        {
            int width = Vector256<T>.Count;
            var m0 = Vector256.Create(initial);
            var m1 = m0;
            var m2 = m0;
            var m3 = m0;
            for (; i <= n - 4 * width; i += 4 * width)
            {
                m0 = TKernel.Invoke(m0, Vector256.LoadUnsafe(ref r, (nuint)i));
                m1 = TKernel.Invoke(m1, Vector256.LoadUnsafe(ref r, (nuint)(i + width)));
                m2 = TKernel.Invoke(m2, Vector256.LoadUnsafe(ref r, (nuint)(i + 2 * width)));
                m3 = TKernel.Invoke(m3, Vector256.LoadUnsafe(ref r, (nuint)(i + 3 * width)));
            }

            var m = TKernel.Invoke(TKernel.Invoke(m0, m1), TKernel.Invoke(m2, m3));
            for (int k = 0; k < width; k++)
            {
                result = TKernel.Invoke(result, m.GetElement(k));
            }
        }
        else if (Vector128.IsHardwareAccelerated && n >= 4 * Vector128<T>.Count)
        {
            int width = Vector128<T>.Count;
            var m0 = Vector128.Create(initial);
            var m1 = m0;
            var m2 = m0;
            var m3 = m0;
            for (; i <= n - 4 * width; i += 4 * width)
            {
                m0 = TKernel.Invoke(m0, Vector128.LoadUnsafe(ref r, (nuint)i));
                m1 = TKernel.Invoke(m1, Vector128.LoadUnsafe(ref r, (nuint)(i + width)));
                m2 = TKernel.Invoke(m2, Vector128.LoadUnsafe(ref r, (nuint)(i + 2 * width)));
                m3 = TKernel.Invoke(m3, Vector128.LoadUnsafe(ref r, (nuint)(i + 3 * width)));
            }

            var m = TKernel.Invoke(TKernel.Invoke(m0, m1), TKernel.Invoke(m2, m3));
            for (int k = 0; k < width; k++)
            {
                result = TKernel.Invoke(result, m.GetElement(k));
            }
        }

        for (; i < n; i++)
        {
            result = TKernel.Invoke(result, Unsafe.Add(ref r, i));
        }

        return result;
    }

    private static bool TryRowExtremaReal<T, TOperator>(int rows, int columns, ref readonly T a, int columnStride, Span<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (typeof(TOperator) == typeof(MaximumOperator<T>))
        {
            RowExtrema<T, MaximumKernel<T>>(rows, columns, in a, columnStride, destination);
        }
        else if (typeof(TOperator) == typeof(MinimumOperator<T>))
        {
            RowExtrema<T, MinimumKernel<T>>(rows, columns, in a, columnStride, destination);
        }
        else if (typeof(TOperator) == typeof(MaximumNumberOperator<T>))
        {
            RowExtrema<T, MaximumNumberKernel<T>>(rows, columns, in a, columnStride, destination);
        }
        else if (typeof(TOperator) == typeof(MinimumNumberOperator<T>))
        {
            RowExtrema<T, MinimumNumberKernel<T>>(rows, columns, in a, columnStride, destination);
        }
        else
        {
            return false;
        }

        return true;
    }

    // result[i] = a[i, 0]、続いて j = 1, 2, … の順に result[i] = TKernel(result[i], a[i, j])（StridedKernels.RowExtrema と同じ順）
    private static void RowExtrema<T, TKernel>(int rows, int columns, ref readonly T a, int columnStride, Span<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TKernel : IFoldKernel<T>
    {
        MemoryMarshal.CreateReadOnlySpan(in a, rows).CopyTo(destination);
        ref T d = ref MemoryMarshal.GetReference(destination);
        for (int j = 1; j < columns; j++)
        {
            ref T c = ref Unsafe.Add(ref Unsafe.AsRef(in a), (nint)j * columnStride);
            int i = 0;
            if (Vector512.IsHardwareAccelerated)
            {
                for (int width = Vector512<T>.Count; i <= rows - width; i += width)
                {
                    TKernel.Invoke(Vector512.LoadUnsafe(ref d, (nuint)i), Vector512.LoadUnsafe(ref c, (nuint)i)).StoreUnsafe(ref d, (nuint)i);
                }
            }
            else if (Vector256.IsHardwareAccelerated)
            {
                for (int width = Vector256<T>.Count; i <= rows - width; i += width)
                {
                    TKernel.Invoke(Vector256.LoadUnsafe(ref d, (nuint)i), Vector256.LoadUnsafe(ref c, (nuint)i)).StoreUnsafe(ref d, (nuint)i);
                }
            }
            else if (Vector128.IsHardwareAccelerated)
            {
                for (int width = Vector128<T>.Count; i <= rows - width; i += width)
                {
                    TKernel.Invoke(Vector128.LoadUnsafe(ref d, (nuint)i), Vector128.LoadUnsafe(ref c, (nuint)i)).StoreUnsafe(ref d, (nuint)i);
                }
            }

            for (; i < rows; i++)
            {
                Unsafe.Add(ref d, i) = TKernel.Invoke(Unsafe.Add(ref d, i), Unsafe.Add(ref c, i));
            }
        }
    }
}
