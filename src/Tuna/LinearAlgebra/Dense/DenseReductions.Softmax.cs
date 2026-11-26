using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Tuna.LinearAlgebra;

internal static partial class DenseReductions
{
    // Σ exp(x[i] - shift)（差が 0 の項は 0）と一致数を TermChunk 単位で計算する。
    // shift が有限なら x[i] - shift == 0 と x[i] == shift は同値のため、入力を再走査せず作業領域上で一致数を数えられる。
    [SkipLocalsInit]
    private static T SumOfShiftedExp<T>(ReadOnlySpan<T> x, T shift, out int count) where T : unmanaged, IFloatingPointIeee754<T>
    {
        int threads = ParallelThreads(x.Length);
        if (threads > 1)
        {
            return SumOfShiftedExpParallel(x, shift, threads, out count);
        }

        var tree = new SumTree<T>(stackalloc T[SumTree<T>.Capacity]);
        Span<T> terms = stackalloc T[TermChunk];
        AddShiftedExpBlocks(ref tree, x, shift, terms, out count);
        return tree.Result();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AddShiftedExpBlocks<T>(ref SumTree<T> tree, ReadOnlySpan<T> x, T shift, scoped Span<T> terms, out int count)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        count = 0;
        // start += TermChunk は x.Length が int.MaxValue 付近でオーバーフローするため実処理長 length で進める
        for (int start = 0, length; start < x.Length; start += length)
        {
            length = Math.Min(TermChunk, x.Length - start);
            var chunk = terms[..length];
            SubtractInto(x.Slice(start, length), shift, chunk);
            count += ReplaceZeroWithNegativeInfinity(chunk);
            ExpInPlace(chunk);
            AddBlocks(ref tree, length, new UnaryTerm<T, ElementKernel<T>>(in MemoryMarshal.GetReference(terms), T.Zero));
        }
    }

    // destination[i] = exp(x[i] - shift)（指数は DenseElementwise.Exp とビット一致する SIMD 実装を使用）。
    private static void ShiftedExp<T>(ReadOnlySpan<T> x, T shift, Span<T> destination) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var terms = destination[..x.Length];
        SubtractInto(x, shift, terms);
        ExpInPlace(terms);
    }

    // destination[i] = x[i] - shift
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void SubtractInto<T>(ReadOnlySpan<T> x, T shift, Span<T> destination) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ref T s = ref MemoryMarshal.GetReference(x);
        ref T d = ref MemoryMarshal.GetReference(destination);
        int n = x.Length, i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            for (int width = Vector512<T>.Count; i <= n - width; i += width)
            {
                (Vector512.LoadUnsafe(ref s, (nuint)i) - Vector512.Create(shift)).StoreUnsafe(ref d, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            for (int width = Vector256<T>.Count; i <= n - width; i += width)
            {
                (Vector256.LoadUnsafe(ref s, (nuint)i) - Vector256.Create(shift)).StoreUnsafe(ref d, (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            for (int width = Vector128<T>.Count; i <= n - width; i += width)
            {
                (Vector128.LoadUnsafe(ref s, (nuint)i) - Vector128.Create(shift)).StoreUnsafe(ref d, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            Unsafe.Add(ref d, i) = Unsafe.Add(ref s, i) - shift;
        }
    }

    // destination[i] = (x[i] - first) - second（引き算ごとに丸めるので、2回に分けて引いたものと同じビット）
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void SubtractTwiceInto<T>(ReadOnlySpan<T> x, T first, T second, Span<T> destination) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ref T s = ref MemoryMarshal.GetReference(x);
        ref T d = ref MemoryMarshal.GetReference(destination);
        int n = x.Length, i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            for (int width = Vector512<T>.Count; i <= n - width; i += width)
            {
                (Vector512.LoadUnsafe(ref s, (nuint)i) - Vector512.Create(first) - Vector512.Create(second)).StoreUnsafe(ref d, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            for (int width = Vector256<T>.Count; i <= n - width; i += width)
            {
                (Vector256.LoadUnsafe(ref s, (nuint)i) - Vector256.Create(first) - Vector256.Create(second)).StoreUnsafe(ref d, (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            for (int width = Vector128<T>.Count; i <= n - width; i += width)
            {
                (Vector128.LoadUnsafe(ref s, (nuint)i) - Vector128.Create(first) - Vector128.Create(second)).StoreUnsafe(ref d, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            Unsafe.Add(ref d, i) = Unsafe.Add(ref s, i) - first - second;
        }
    }

    // destination[i] = exp(destination[i])（DenseElementwise.Exp とビット一致）。
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ExpInPlace<T>(Span<T> destination) where T : unmanaged, IFloatingPointIeee754<T>
    {
        DenseElementwise.TryMap<T, ExpOperator<T>>(destination, destination, default);
    }

    // 差が ±0 の要素を -∞ に置換してから exp を適用し、ShiftedExpOperator の「差が 0 なら 0」をビット一致で再現する。

    // destination[i] が ±0 なら −∞ に置き換え、置き換えた個数を返す
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static int ReplaceZeroWithNegativeInfinity<T>(Span<T> destination) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ref T d = ref MemoryMarshal.GetReference(destination);
        int n = destination.Length, i = 0, count = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            for (int width = Vector512<T>.Count; i <= n - width; i += width)
            {
                var v = Vector512.LoadUnsafe(ref d, (nuint)i);
                var zero = Vector512.Equals(v, Vector512<T>.Zero);
                Vector512.ConditionalSelect(zero, Vector512.Create(T.NegativeInfinity), v).StoreUnsafe(ref d, (nuint)i);
                count += BitOperations.PopCount(zero.ExtractMostSignificantBits());
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            for (int width = Vector256<T>.Count; i <= n - width; i += width)
            {
                var v = Vector256.LoadUnsafe(ref d, (nuint)i);
                var zero = Vector256.Equals(v, Vector256<T>.Zero);
                Vector256.ConditionalSelect(zero, Vector256.Create(T.NegativeInfinity), v).StoreUnsafe(ref d, (nuint)i);
                count += BitOperations.PopCount(zero.ExtractMostSignificantBits());
            }
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            for (int width = Vector128<T>.Count; i <= n - width; i += width)
            {
                var v = Vector128.LoadUnsafe(ref d, (nuint)i);
                var zero = Vector128.Equals(v, Vector128<T>.Zero);
                Vector128.ConditionalSelect(zero, Vector128.Create(T.NegativeInfinity), v).StoreUnsafe(ref d, (nuint)i);
                count += BitOperations.PopCount(zero.ExtractMostSignificantBits());
            }
        }

        for (; i < n; i++)
        {
            if (Unsafe.Add(ref d, i) == T.Zero)
            {
                Unsafe.Add(ref d, i) = T.NegativeInfinity;
                count++;
            }
        }

        return count;
    }

    // destination[i] が ±0 なら -∞ に置換し counts[i] を 1 増やす。比較マスク（全ビット 1 = -1）を int32 レーンに揃えて減算でカウントする。
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ReplaceZeroWithNegativeInfinity<T>(Span<T> destination, Span<int> counts) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ref T d = ref MemoryMarshal.GetReference(destination);
        ref int c = ref MemoryMarshal.GetReference(counts);
        int n = destination.Length, i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            for (int width = Vector512<int>.Count; i <= n - width; i += width)
            {
                var v = Vector512.LoadUnsafe(ref d, (nuint)i);
                var zero = Vector512.Equals(v, Vector512<T>.Zero);
                Vector512.ConditionalSelect(zero, Vector512.Create(T.NegativeInfinity), v).StoreUnsafe(ref d, (nuint)i);
                Vector512<int> found;
                if (Vector512<T>.Count == Vector512<int>.Count)
                {
                    found = zero.As<T, int>();
                }
                else
                {
                    nuint next = (nuint)(i + Vector512<T>.Count);
                    var w = Vector512.LoadUnsafe(ref d, next);
                    var wzero = Vector512.Equals(w, Vector512<T>.Zero);
                    Vector512.ConditionalSelect(wzero, Vector512.Create(T.NegativeInfinity), w).StoreUnsafe(ref d, next);
                    found = Vector512.Narrow(zero.As<T, long>(), wzero.As<T, long>());
                }

                (Vector512.LoadUnsafe(ref c, (nuint)i) - found).StoreUnsafe(ref c, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            for (int width = Vector256<int>.Count; i <= n - width; i += width)
            {
                var v = Vector256.LoadUnsafe(ref d, (nuint)i);
                var zero = Vector256.Equals(v, Vector256<T>.Zero);
                Vector256.ConditionalSelect(zero, Vector256.Create(T.NegativeInfinity), v).StoreUnsafe(ref d, (nuint)i);
                Vector256<int> found;
                if (Vector256<T>.Count == Vector256<int>.Count)
                {
                    found = zero.As<T, int>();
                }
                else
                {
                    nuint next = (nuint)(i + Vector256<T>.Count);
                    var w = Vector256.LoadUnsafe(ref d, next);
                    var wzero = Vector256.Equals(w, Vector256<T>.Zero);
                    Vector256.ConditionalSelect(wzero, Vector256.Create(T.NegativeInfinity), w).StoreUnsafe(ref d, next);
                    found = Vector256.Narrow(zero.As<T, long>(), wzero.As<T, long>());
                }

                (Vector256.LoadUnsafe(ref c, (nuint)i) - found).StoreUnsafe(ref c, (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            for (int width = Vector128<int>.Count; i <= n - width; i += width)
            {
                var v = Vector128.LoadUnsafe(ref d, (nuint)i);
                var zero = Vector128.Equals(v, Vector128<T>.Zero);
                Vector128.ConditionalSelect(zero, Vector128.Create(T.NegativeInfinity), v).StoreUnsafe(ref d, (nuint)i);
                Vector128<int> found;
                if (Vector128<T>.Count == Vector128<int>.Count)
                {
                    found = zero.As<T, int>();
                }
                else
                {
                    nuint next = (nuint)(i + Vector128<T>.Count);
                    var w = Vector128.LoadUnsafe(ref d, next);
                    var wzero = Vector128.Equals(w, Vector128<T>.Zero);
                    Vector128.ConditionalSelect(wzero, Vector128.Create(T.NegativeInfinity), w).StoreUnsafe(ref d, next);
                    found = Vector128.Narrow(zero.As<T, long>(), wzero.As<T, long>());
                }

                (Vector128.LoadUnsafe(ref c, (nuint)i) - found).StoreUnsafe(ref c, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            if (Unsafe.Add(ref d, i) == T.Zero)
            {
                Unsafe.Add(ref d, i) = T.NegativeInfinity;
                Unsafe.Add(ref c, i)++;
            }
        }
    }

    // destination[i] = x[i] - shifts[i]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void SubtractRowsInto<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> shifts, Span<T> destination) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ref T s = ref MemoryMarshal.GetReference(x);
        ref T q = ref MemoryMarshal.GetReference(shifts);
        ref T d = ref MemoryMarshal.GetReference(destination);
        int n = x.Length, i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            for (int width = Vector512<T>.Count; i <= n - width; i += width)
            {
                (Vector512.LoadUnsafe(ref s, (nuint)i) - Vector512.LoadUnsafe(ref q, (nuint)i)).StoreUnsafe(ref d, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            for (int width = Vector256<T>.Count; i <= n - width; i += width)
            {
                (Vector256.LoadUnsafe(ref s, (nuint)i) - Vector256.LoadUnsafe(ref q, (nuint)i)).StoreUnsafe(ref d, (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            for (int width = Vector128<T>.Count; i <= n - width; i += width)
            {
                (Vector128.LoadUnsafe(ref s, (nuint)i) - Vector128.LoadUnsafe(ref q, (nuint)i)).StoreUnsafe(ref d, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            Unsafe.Add(ref d, i) = Unsafe.Add(ref s, i) - Unsafe.Add(ref q, i);
        }
    }

    // destination[i] = (x[i] - first[i]) - second[i]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void SubtractRowsTwiceInto<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> first, ReadOnlySpan<T> second, Span<T> destination) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ref T s = ref MemoryMarshal.GetReference(x);
        ref T q = ref MemoryMarshal.GetReference(first);
        ref T r = ref MemoryMarshal.GetReference(second);
        ref T d = ref MemoryMarshal.GetReference(destination);
        int n = x.Length, i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            for (int width = Vector512<T>.Count; i <= n - width; i += width)
            {
                (Vector512.LoadUnsafe(ref s, (nuint)i) - Vector512.LoadUnsafe(ref q, (nuint)i) - Vector512.LoadUnsafe(ref r, (nuint)i)).StoreUnsafe(ref d, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            for (int width = Vector256<T>.Count; i <= n - width; i += width)
            {
                (Vector256.LoadUnsafe(ref s, (nuint)i) - Vector256.LoadUnsafe(ref q, (nuint)i) - Vector256.LoadUnsafe(ref r, (nuint)i)).StoreUnsafe(ref d, (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            for (int width = Vector128<T>.Count; i <= n - width; i += width)
            {
                (Vector128.LoadUnsafe(ref s, (nuint)i) - Vector128.LoadUnsafe(ref q, (nuint)i) - Vector128.LoadUnsafe(ref r, (nuint)i)).StoreUnsafe(ref d, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            Unsafe.Add(ref d, i) = Unsafe.Add(ref s, i) - Unsafe.Add(ref q, i) - Unsafe.Add(ref r, i);
        }
    }

    // 各列 j について destination[:, j] = x[:, j] - first（exponentiate なら、その指数）。second が空でなければ (x[:, j] - first) - second
    private static void SubtractRows<T>(int rows, int columns, ref readonly T x, int xColumnStride, ReadOnlySpan<T> first, ReadOnlySpan<T> second,
        ref T destination, int destinationColumnStride, bool exponentiate)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        for (int j = 0; j < columns; j++)
        {
            var source = MemoryMarshal.CreateReadOnlySpan(in Unsafe.Add(ref Unsafe.AsRef(in x), (nint)j * xColumnStride), rows);
            var target = MemoryMarshal.CreateSpan(ref Unsafe.Add(ref destination, (nint)j * destinationColumnStride), rows);
            if (second.IsEmpty)
            {
                SubtractRowsInto(source, first, target);
            }
            else
            {
                SubtractRowsTwiceInto(source, first, second, target);
            }

            if (exponentiate)
            {
                ExpInPlace(target);
            }
        }
    }

    // 各列 j について destination[:, j] /= divisors
    private static void DivideRows<T>(int rows, int columns, ref T destination, int columnStride, ReadOnlySpan<T> divisors) where T : unmanaged, IFloatingPointIeee754<T>
    {
        for (int j = 0; j < columns; j++)
        {
            DivideRowsInPlace(MemoryMarshal.CreateSpan(ref Unsafe.Add(ref destination, (nint)j * columnStride), rows), divisors);
        }
    }

    // destination[i] /= divisors[i]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void DivideRowsInPlace<T>(Span<T> destination, ReadOnlySpan<T> divisors) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ref T d = ref MemoryMarshal.GetReference(destination);
        ref T q = ref MemoryMarshal.GetReference(divisors);
        int n = destination.Length, i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            for (int width = Vector512<T>.Count; i <= n - width; i += width)
            {
                (Vector512.LoadUnsafe(ref d, (nuint)i) / Vector512.LoadUnsafe(ref q, (nuint)i)).StoreUnsafe(ref d, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            for (int width = Vector256<T>.Count; i <= n - width; i += width)
            {
                (Vector256.LoadUnsafe(ref d, (nuint)i) / Vector256.LoadUnsafe(ref q, (nuint)i)).StoreUnsafe(ref d, (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            for (int width = Vector128<T>.Count; i <= n - width; i += width)
            {
                (Vector128.LoadUnsafe(ref d, (nuint)i) / Vector128.LoadUnsafe(ref q, (nuint)i)).StoreUnsafe(ref d, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            Unsafe.Add(ref d, i) /= Unsafe.Add(ref q, i);
        }
    }

    // StridedKernels.ColumnSoftmax の1列分: 最大値を引いて指数をとり、その和で割る
    private static void Softmax<T>(ReadOnlySpan<T> x, Span<T> destination) where T : unmanaged, IFloatingPointIeee754<T>
    {
        T max = Reduce<T, MaximumKernel<T>>(x[1..], x[0]);
        ShiftedExp(x, max, destination);
        T sum = Sum<T, UnaryTerm<T, ElementKernel<T>>>(destination.Length, new(in MemoryMarshal.GetReference(destination), T.Zero));
        DivideInPlace(destination, sum);
    }

    // destination[i] /= divisor（正しく丸める除算なので、ベクトルでもスカラーと同じビット）
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void DivideInPlace<T>(Span<T> destination, T divisor) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ref T d = ref MemoryMarshal.GetReference(destination);
        int n = destination.Length, i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            for (int width = Vector512<T>.Count; i <= n - width; i += width)
            {
                (Vector512.LoadUnsafe(ref d, (nuint)i) / Vector512.Create(divisor)).StoreUnsafe(ref d, (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            for (int width = Vector256<T>.Count; i <= n - width; i += width)
            {
                (Vector256.LoadUnsafe(ref d, (nuint)i) / Vector256.Create(divisor)).StoreUnsafe(ref d, (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            for (int width = Vector128<T>.Count; i <= n - width; i += width)
            {
                (Vector128.LoadUnsafe(ref d, (nuint)i) / Vector128.Create(divisor)).StoreUnsafe(ref d, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            destination[i] /= divisor;
        }
    }
}
