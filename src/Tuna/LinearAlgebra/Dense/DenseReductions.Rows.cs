using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Tuna.LinearAlgebra;

internal static partial class DenseReductions
{
    // ---------- 行ごとの和 ----------

    private static bool TryRowSumsReal<T, TOperator>(int rows, int columns, ref readonly T a, int columnStride, ReadOnlySpan<T> parameters, Span<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (typeof(TOperator) == typeof(FirstOperator<T>))
        {
            RowSums<T, ElementKernel<T>>(rows, columns, in a, columnStride, parameters, destination);
        }
        else if (typeof(TOperator) == typeof(SquareOperator<T>))
        {
            RowSums<T, SquareKernel<T>>(rows, columns, in a, columnStride, parameters, destination);
        }
        else if (typeof(TOperator) == typeof(IgnoreNaNOperator<T>))
        {
            RowSums<T, IgnoreNaNKernel<T>>(rows, columns, in a, columnStride, parameters, destination);
        }
        else if (typeof(TOperator) == typeof(DeviationOperator<T>))
        {
            RowSums<T, DeviationKernel<T>>(rows, columns, in a, columnStride, parameters, destination);
        }
        else if (typeof(TOperator) == typeof(SquaredDeviationOperator<T>))
        {
            RowSums<T, SquaredDeviationKernel<T>>(rows, columns, in a, columnStride, parameters, destination);
        }
        else if (typeof(TOperator) == typeof(ScaledSquareByFactorOperator<T>))
        {
            RowSums<T, ScaledSquareKernel<T>>(rows, columns, in a, columnStride, parameters, destination);
        }
        else
        {
            return false;
        }

        return true;
    }

    // RowPairwiseSums と同じ木構造で行ごとの総和を計算する（行を RowTile 単位に分割し、列内の連続行を SIMD レーンに割り当てる）。
    // exponentiate が true のときは exp(a[i, j] - parameters[i])（差が 0 の項は 0）を計算し、一致数を counts に加算する。
    private static void RowSums<T, TKernel>(int rows, int columns, ref readonly T a, int columnStride, ReadOnlySpan<T> parameters, Span<T> destination,
        bool exponentiate = false, Span<int> counts = default)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TKernel : ITermKernel<T>
    {
        if (rows == 0)
        {
            return;
        }

        int blocks = (int)(((long)columns + Block - 1) / Block);
        int levels = blocks <= 1 ? 0 : BitOperations.Log2((uint)blocks) + 1;
        int tile = Math.Min(rows, RowTile);
        T[] rented = ArrayPool<T>.Shared.Rent(checked(tile * (levels + 2)));
        try
        {
            // first += tile は rows が int.MaxValue 付近でオーバーフローするため実処理行数 count で進める
            for (int first = 0, count; first < rows; first += count)
            {
                count = Math.Min(tile, rows - first);
                ref readonly T start = ref Unsafe.Add(ref Unsafe.AsRef(in a), first);
                var p = parameters.IsEmpty ? parameters : parameters.Slice(first, count);
                var result = destination.Slice(first, count);
                var zeros = exponentiate ? counts.Slice(first, count) : counts;
                var terms = rented.AsSpan(tile * (levels + 1), count);
                if (blocks <= 1)
                {
                    result.Clear();
                    for (int j = 0; j < columns; j++)
                    {
                        AddColumn<T, TKernel>(result, in Unsafe.Add(ref Unsafe.AsRef(in start), (nint)j * columnStride), p, exponentiate, terms, zeros);
                    }

                    continue;
                }

                // 作業領域: 先頭の count 個がブロックの和、続いて段ごとの部分和
                var sum = rented.AsSpan(0, count);
                int depth = 0;
                for (int begin = 0, block = 0; begin < columns; block++)
                {
                    int end = columns - begin <= Block ? columns : begin + Block;
                    sum.Clear();
                    for (int j = begin; j < end; j++)
                    {
                        AddColumn<T, TKernel>(sum, in Unsafe.Add(ref Unsafe.AsRef(in start), (nint)j * columnStride), p, exponentiate, terms, zeros);
                    }

                    for (int k = block; (k & 1) != 0; k >>= 1)
                    {
                        AddLeft(rented.AsSpan(count * --depth + count, count), sum);
                    }

                    sum.CopyTo(rented.AsSpan(count * depth++ + count, count));
                    begin = end;
                }

                rented.AsSpan(count * --depth + count, count).CopyTo(result);
                while (depth > 0)
                {
                    AddLeft(rented.AsSpan(count * --depth + count, count), result);
                }
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(rented);
        }
    }

    // 1 列分の項を加算する（exponentiate 時は exp(column[i] - parameters[i]) を求め、差が 0 の行は zeros[i] をインクリメントする）。
    private static void AddColumn<T, TKernel>(Span<T> sum, ref readonly T column, ReadOnlySpan<T> parameters, bool exponentiate, Span<T> terms, Span<int> zeros)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TKernel : ITermKernel<T>
    {
        if (exponentiate)
        {
            SubtractRowsInto(MemoryMarshal.CreateReadOnlySpan(in column, sum.Length), parameters, terms);
            ReplaceZeroWithNegativeInfinity(terms, zeros);
            ExpInPlace(terms);
            AddTerms<T, ElementKernel<T>>(sum, in MemoryMarshal.GetReference(terms), default);
        }
        else
        {
            AddTerms<T, TKernel>(sum, in column, parameters);
        }
    }

    // sum[i] = sum[i] + TKernel(column[i], parameters[i])（parameters が空なら引数は 0）
    private static void AddTerms<T, TKernel>(Span<T> sum, ref readonly T column, ReadOnlySpan<T> parameters)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TKernel : ITermKernel<T>
    {
        ref T s = ref MemoryMarshal.GetReference(sum);
        ref T c = ref Unsafe.AsRef(in column);
        bool hasParameters = !parameters.IsEmpty;
        ref T p = ref MemoryMarshal.GetReference(parameters);
        int n = sum.Length, i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            for (int width = Vector512<T>.Count; i <= n - width; i += width)
            {
                var q = hasParameters ? Vector512.LoadUnsafe(ref p, (nuint)i) : Vector512<T>.Zero;
                (Vector512.LoadUnsafe(ref s, (nuint)i) + TKernel.Invoke(Vector512.LoadUnsafe(ref c, (nuint)i), q)).StoreUnsafe(ref s, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            for (int width = Vector256<T>.Count; i <= n - width; i += width)
            {
                var q = hasParameters ? Vector256.LoadUnsafe(ref p, (nuint)i) : Vector256<T>.Zero;
                (Vector256.LoadUnsafe(ref s, (nuint)i) + TKernel.Invoke(Vector256.LoadUnsafe(ref c, (nuint)i), q)).StoreUnsafe(ref s, (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            for (int width = Vector128<T>.Count; i <= n - width; i += width)
            {
                var q = hasParameters ? Vector128.LoadUnsafe(ref p, (nuint)i) : Vector128<T>.Zero;
                (Vector128.LoadUnsafe(ref s, (nuint)i) + TKernel.Invoke(Vector128.LoadUnsafe(ref c, (nuint)i), q)).StoreUnsafe(ref s, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            Unsafe.Add(ref s, i) += TKernel.Invoke(Unsafe.Add(ref c, i), hasParameters ? Unsafe.Add(ref p, i) : T.Zero);
        }
    }

    // sum[i] = left[i] + sum[i]
    private static void AddLeft<T>(ReadOnlySpan<T> left, Span<T> sum) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ref T s = ref MemoryMarshal.GetReference(sum);
        ref T l = ref MemoryMarshal.GetReference(left);
        int n = sum.Length, i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            for (int width = Vector512<T>.Count; i <= n - width; i += width)
            {
                (Vector512.LoadUnsafe(ref l, (nuint)i) + Vector512.LoadUnsafe(ref s, (nuint)i)).StoreUnsafe(ref s, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            for (int width = Vector256<T>.Count; i <= n - width; i += width)
            {
                (Vector256.LoadUnsafe(ref l, (nuint)i) + Vector256.LoadUnsafe(ref s, (nuint)i)).StoreUnsafe(ref s, (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            for (int width = Vector128<T>.Count; i <= n - width; i += width)
            {
                (Vector128.LoadUnsafe(ref l, (nuint)i) + Vector128.LoadUnsafe(ref s, (nuint)i)).StoreUnsafe(ref s, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            Unsafe.Add(ref s, i) = Unsafe.Add(ref l, i) + Unsafe.Add(ref s, i);
        }
    }
}
