using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Tuna.LinearAlgebra;

// ソート（全順序 -∞ < ... < -0 < +0 < ... < +∞、NaN は末尾に元順保持）と分位点（Hyndman & Fan 1996 定義 7）。
// double / float / Half は BitsKey の整数キーで並べ替え・選択を行う。
internal static partial class StridedKernels
{
    internal static void Sort<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination) where T : unmanaged, INumber<T>
    {
        RequireLength(destination, x.Length);
        RequireElementwiseAlias(x.AsMatrix(), destination.AsMatrix());
        if (typeof(T) == typeof(double))
        {
            SortByBits<T, long>(x, destination);
        }
        else if (typeof(T) == typeof(float))
        {
            SortByBits<T, int>(x, destination);
        }
        else if (typeof(T) == typeof(Half))
        {
            SortByBits<T, short>(x, destination);
        }
        else
        {
            SortByValue(x, destination);
        }
    }

    internal static void ArgSort<T>(ReadOnlyVectorSpan<T> x, Span<int> destination) where T : unmanaged, INumber<T>
    {
        RequireLength(destination, x.Length);
        var input = x.AsMatrix();
        if (StridedLayout.OverlapsAcrossTypes(
            in input.Reference, input.Rows, input.Columns, input.RowStride, input.ColumnStride,
            in MemoryMarshal.GetReference(destination), destination.Length, 1, 1, 0))
        {
            throw new ArgumentException("The destination must not overlap the input.", nameof(destination));
        }

        if (typeof(T) == typeof(double))
        {
            ArgSortByBits<T, long>(x, destination);
        }
        else if (typeof(T) == typeof(float))
        {
            ArgSortByBits<T, int>(x, destination);
        }
        else if (typeof(T) == typeof(Half))
        {
            ArgSortByBits<T, short>(x, destination);
        }
        else
        {
            ArgSortByValue(x, destination);
        }
    }

    internal static T Quantile<T>(ReadOnlyVectorSpan<T> x, double probability, bool ignoreNaN) where T : unmanaged, IFloatingPointIeee754<T>
    {
        RequireProbability(probability);
        if (typeof(T) == typeof(double))
        {
            return Quantile<T, long, BitsKey<T, long>>(x, probability, ignoreNaN);
        }

        if (typeof(T) == typeof(float))
        {
            return Quantile<T, int, BitsKey<T, int>>(x, probability, ignoreNaN);
        }

        if (typeof(T) == typeof(Half))
        {
            return Quantile<T, short, BitsKey<T, short>>(x, probability, ignoreNaN);
        }

        return Quantile<T, T, TotalOrderKey<T>>(x, probability, ignoreNaN);
    }

    internal static void Quantiles<T>(ReadOnlyVectorSpan<T> x, ReadOnlySpan<double> probabilities, bool ignoreNaN, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        RequireLength(destination, probabilities.Length);
        foreach (double p in probabilities)
        {
            RequireProbability(p, nameof(probabilities));
        }

        RequireDisjoint(x.AsMatrix(), destination.AsMatrix());
        var output = destination.AsMatrix().AsReadOnly();
        if (StridedLayout.OverlapsAcrossTypes(
            in MemoryMarshal.GetReference(probabilities), probabilities.Length, 1, 1, 0,
            in output.Reference, output.Rows, output.Columns, output.RowStride, output.ColumnStride))
        {
            throw new ArgumentException("The destination must not overlap the inputs.", nameof(destination));
        }

        if (typeof(T) == typeof(double))
        {
            Quantiles<T, long, BitsKey<T, long>>(x, probabilities, ignoreNaN, destination);
        }
        else if (typeof(T) == typeof(float))
        {
            Quantiles<T, int, BitsKey<T, int>>(x, probabilities, ignoreNaN, destination);
        }
        else if (typeof(T) == typeof(Half))
        {
            Quantiles<T, short, BitsKey<T, short>>(x, probabilities, ignoreNaN, destination);
        }
        else
        {
            Quantiles<T, T, TotalOrderKey<T>>(x, probabilities, ignoreNaN, destination);
        }
    }

    internal static void ColumnQuantiles<T>(ReadOnlyMatrixSpan<T> a, double probability, bool ignoreNaN, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        RequireProbability(probability);
        RequireLength(destination, a.Columns);
        RequireDisjoint(a, destination.AsMatrix());
        if (typeof(T) == typeof(double))
        {
            ColumnQuantiles<T, long, BitsKey<T, long>>(a, probability, ignoreNaN, destination);
        }
        else if (typeof(T) == typeof(float))
        {
            ColumnQuantiles<T, int, BitsKey<T, int>>(a, probability, ignoreNaN, destination);
        }
        else if (typeof(T) == typeof(Half))
        {
            ColumnQuantiles<T, short, BitsKey<T, short>>(a, probability, ignoreNaN, destination);
        }
        else
        {
            ColumnQuantiles<T, T, TotalOrderKey<T>>(a, probability, ignoreNaN, destination);
        }
    }

    internal static void RowQuantiles<T>(ReadOnlyMatrixSpan<T> a, double probability, bool ignoreNaN, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        ColumnQuantiles(a.T, probability, ignoreNaN, destination);

    private static void RequireProbability(double probability, string name = "probability")
    {
        if (!(probability >= 0 && probability <= 1))
        {
            throw new ArgumentOutOfRangeException(name, probability, "The probability must be in [0, 1].");
        }
    }

    // ---------- 並べ替え ----------

    private static void SortByBits<T, TKey>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, INumber<T>
        where TKey : unmanaged, IBinaryInteger<TKey>, ISignedNumber<TKey>, IMinMaxValue<TKey>
    {
        // 出力先が連続なら、作業領域を使わずに出力先の上で鍵に写して並べる（出力先は入力そのものか、入力と重ならない）
        if (destination.AsReadOnly().Stride == 1)
        {
            SortContiguousByBits<T, TKey>(x, destination, copy: !Overlaps(x.AsMatrix(), destination.AsMatrix(), out _));
            return;
        }

        int length = x.Length;
        TKey[] keys = ArrayPool<TKey>.Shared.Rent(Math.Max(1, length));
        T[] nans = ArrayPool<T>.Shared.Rent(Math.Max(1, length));
        try
        {
            int n = 0, nanCount = 0;
            for (int i = 0; i < length; i++)
            {
                T value = x.ElementAt(i);
                if (T.IsNaN(value))
                {
                    nans[nanCount++] = value;
                }
                else
                {
                    keys[n++] = BitsKey<T, TKey>.ToKey(value);
                }
            }

            // 鍵の整数の大小が全順序と一致するので、整数の既定の比較で並べ替える
            keys.AsSpan(0, n).Sort();
            for (int i = 0; i < n; i++)
            {
                destination[i] = BitsKey<T, TKey>.FromKey(keys[i]);
            }

            for (int i = 0; i < nanCount; i++)
            {
                destination[n + i] = nans[i];
            }
        }
        finally
        {
            ArrayPool<TKey>.Shared.Return(keys);
            ArrayPool<T>.Shared.Return(nans);
        }
    }

    // 連続出力先上で整数キー変換・ソート・逆変換をインプレース実行する（NaN は元の順で末尾に配置）。
    private static void SortContiguousByBits<T, TKey>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination, bool copy)
        where T : unmanaged, INumber<T>
        where TKey : unmanaged, IBinaryInteger<TKey>, ISignedNumber<TKey>, IMinMaxValue<TKey>
    {
        if (copy)
        {
            x.CopyTo(destination);
        }

        var values = destination.Data;
        int n = values.Length;
        int first = IndexOfNaN(values);
        T[]? nans = null;
        try
        {
            int nanCount = 0;
            if (first >= 0)
            {
                nans = ArrayPool<T>.Shared.Rent(n - first);
                int count = first;
                for (int i = first; i < n; i++)
                {
                    T value = values[i];
                    if (T.IsNaN(value))
                    {
                        nans[nanCount++] = value;
                    }
                    else
                    {
                        values[count++] = value;
                    }
                }

                n = count;
            }

            // BitsKey の写像（負の数は符号以外のビットを反転する）は自分自身の逆なので、同じ関数で値に戻す
            var keys = MemoryMarshal.Cast<T, TKey>(values[..n]);
            FlipNegatives(keys);
            keys.Sort();
            FlipNegatives(keys);
            nans.AsSpan(0, nanCount).CopyTo(values[n..]);
        }
        finally
        {
            if (nans is not null)
            {
                ArrayPool<T>.Shared.Return(nans);
            }
        }
    }

    // 最初の NaN のインデックス（なければ -1）。
    private static int IndexOfNaN<T>(ReadOnlySpan<T> values) where T : unmanaged, INumber<T>
    {
        if (typeof(T) == typeof(double))
        {
            return DenseReductions.IndexOfNaN(MemoryMarshal.Cast<T, double>(values));
        }

        if (typeof(T) == typeof(float))
        {
            return DenseReductions.IndexOfNaN(MemoryMarshal.Cast<T, float>(values));
        }

        for (int i = 0; i < values.Length; i++)
        {
            if (T.IsNaN(values[i]))
            {
                return i;
            }
        }

        return -1;
    }

    // 浮動小数点ビット列と全順序整数キーをインプレース相互変換する（BitsKey と同一写像）。
    private static void FlipNegatives<TKey>(Span<TKey> keys)
        where TKey : unmanaged, IBinaryInteger<TKey>, ISignedNumber<TKey>, IMinMaxValue<TKey>
    {
        int shift = Unsafe.SizeOf<TKey>() * 8 - 1;
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated && System.Numerics.Vector<TKey>.IsSupported)
        {
            var mask = new System.Numerics.Vector<TKey>(TKey.MaxValue);
            var vectors = MemoryMarshal.Cast<TKey, System.Numerics.Vector<TKey>>(keys);
            for (int k = 0; k < vectors.Length; k++)
            {
                var v = vectors[k];
                vectors[k] = v ^ ((v >> shift) & mask);
            }

            i = vectors.Length * System.Numerics.Vector<TKey>.Count;
        }

        for (; i < keys.Length; i++)
        {
            TKey b = keys[i];
            keys[i] = b ^ ((b >> shift) & TKey.MaxValue);
        }
    }

    private static void SortByValue<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination) where T : unmanaged, INumber<T>
    {
        int length = x.Length;
        T[] values = ArrayPool<T>.Shared.Rent(Math.Max(1, length));
        T[] nans = ArrayPool<T>.Shared.Rent(Math.Max(1, length));
        try
        {
            int n = 0, nanCount = 0;
            for (int i = 0; i < length; i++)
            {
                T value = x.ElementAt(i);
                if (T.IsNaN(value))
                {
                    nans[nanCount++] = value;
                }
                else
                {
                    values[n++] = value;
                }
            }

            values.AsSpan(0, n).Sort(new KeyComparer<T, NumberOrder<T>>());
            for (int i = 0; i < n; i++)
            {
                destination[i] = values[i];
            }

            for (int i = 0; i < nanCount; i++)
            {
                destination[n + i] = nans[i];
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(values);
            ArrayPool<T>.Shared.Return(nans);
        }
    }

    // ArgSort の位置の列 order には、NaN でない位置を前から、NaN の位置を後ろから置く

    private static void ArgSortByBits<T, TKey>(ReadOnlyVectorSpan<T> x, Span<int> destination)
        where T : unmanaged, INumber<T>
        where TKey : unmanaged, IBinaryInteger<TKey>, ISignedNumber<TKey>, IMinMaxValue<TKey>
    {
        int length = x.Length;
        TKey[] keys = ArrayPool<TKey>.Shared.Rent(Math.Max(1, length));
        int[] order = ArrayPool<int>.Shared.Rent(Math.Max(1, length));
        try
        {
            int n = 0, nanStart = length;
            for (int i = 0; i < length; i++)
            {
                T value = x.ElementAt(i);
                if (T.IsNaN(value))
                {
                    order[--nanStart] = i;
                    continue;
                }

                keys[n] = BitsKey<T, TKey>.ToKey(value);
                order[n++] = i;
            }

            var sortedKeys = keys.AsSpan(0, n);
            var sortedOrder = order.AsSpan(0, n);
            sortedKeys.Sort(sortedOrder);
            StabilizeRuns(sortedKeys, sortedOrder, new KeyComparer<TKey, IntegerOrder<TKey>>());
            WriteOrder(order.AsSpan(0, length), n, destination);
        }
        finally
        {
            ArrayPool<TKey>.Shared.Return(keys);
            ArrayPool<int>.Shared.Return(order);
        }
    }

    private static void ArgSortByValue<T>(ReadOnlyVectorSpan<T> x, Span<int> destination) where T : unmanaged, INumber<T>
    {
        int length = x.Length;
        T[] values = ArrayPool<T>.Shared.Rent(Math.Max(1, length));
        int[] order = ArrayPool<int>.Shared.Rent(Math.Max(1, length));
        try
        {
            int n = 0, nanStart = length;
            for (int i = 0; i < length; i++)
            {
                T value = x.ElementAt(i);
                if (T.IsNaN(value))
                {
                    order[--nanStart] = i;
                    continue;
                }

                values[n] = value;
                order[n++] = i;
            }

            var sortedValues = values.AsSpan(0, n);
            var sortedOrder = order.AsSpan(0, n);
            var comparer = new KeyComparer<T, NumberOrder<T>>();
            sortedValues.Sort(sortedOrder, comparer);
            StabilizeRuns(sortedValues, sortedOrder, comparer);
            WriteOrder(order.AsSpan(0, length), n, destination);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(values);
            ArrayPool<int>.Shared.Return(order);
        }
    }

    // 非安定ソート後に等値キー区間を元インデックス昇順に整列し、ArgSort の安定性を保証する。
    private static void StabilizeRuns<TKey, TComparer>(Span<TKey> keys, Span<int> order, TComparer comparer)
        where TComparer : IComparer<TKey>
    {
        int start = 0;
        while (start < keys.Length)
        {
            int end = start + 1;
            while (end < keys.Length && comparer.Compare(keys[start], keys[end]) == 0)
            {
                end++;
            }

            if (end - start > 1)
            {
                order[start..end].Sort();
            }

            start = end;
        }
    }

    // ソート済み非 NaN 位置に続き、末尾に逆順格納された NaN 位置を元の順で書き出す。
    private static void WriteOrder(ReadOnlySpan<int> order, int count, Span<int> destination)
    {
        order[..count].CopyTo(destination);
        for (int k = count, i = order.Length - 1; k < order.Length; k++, i--)
        {
            destination[k] = order[i];
        }
    }

    // 値そのものの全順序（-0 < +0、NaN は事前除外）。
    private readonly struct NumberOrder<T> : IKeyOrder<T> where T : INumber<T>
    {
        public static bool Less(T x, T y) => x < y || (x == y && T.IsNegative(x) && !T.IsNegative(y));
    }

    private readonly struct IntegerOrder<TKey> : IKeyOrder<TKey> where TKey : IBinaryInteger<TKey>
    {
        public static bool Less(TKey x, TKey y) => x < y;
    }

    // ---------- 分位点 ----------

    private static T Quantile<T, TKey, TOrder>(ReadOnlyVectorSpan<T> x, double probability, bool ignoreNaN)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TKey : unmanaged
        where TOrder : IOrderKey<T, TKey>
    {
        TKey[] rented = ArrayPool<TKey>.Shared.Rent(Math.Max(1, x.Length));
        try
        {
            return Quantile<T, TKey, TOrder>(x, probability, ignoreNaN, rented);
        }
        finally
        {
            ArrayPool<TKey>.Shared.Return(rented);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ColumnQuantiles<T, TKey, TOrder>(ReadOnlyMatrixSpan<T> a, double probability, bool ignoreNaN, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TKey : unmanaged
        where TOrder : IOrderKey<T, TKey>
    {
        TKey[] rented = ArrayPool<TKey>.Shared.Rent(Math.Max(1, a.Rows));
        try
        {
            for (int j = 0; j < a.Columns; j++)
            {
                destination[j] = Quantile<T, TKey, TOrder>(a.Column(j), probability, ignoreNaN, rented);
            }
        }
        finally
        {
            ArrayPool<TKey>.Shared.Return(rented);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static T Quantile<T, TKey, TOrder>(ReadOnlyVectorSpan<T> x, double probability, bool ignoreNaN, Span<TKey> work)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TKey : unmanaged
        where TOrder : IOrderKey<T, TKey>
    {
        if (!CollectKeys<T, TKey, TOrder>(x, ignoreNaN, work, out int n, out T nan))
        {
            return nan;
        }

        if (n == 0)
        {
            return T.NaN;
        }

        var keys = work[..n];
        QuantilePosition(n, probability, out int j, out double g);
        long budget = SelectBudget(n);
        ulong random = 0;
        Select<TKey, TOrder>(keys, 0, n - 1, j, ref budget, ref random);
        T lower = TOrder.FromKey(keys[j]);
        if (g == 0)
        {
            return lower;
        }

        // j より後ろ（すべて keys[j] 以上）の最小が xⱼ₊₁
        TKey upper = keys[j + 1];
        for (int i = j + 2; i < n; i++)
        {
            if (TOrder.Less(keys[i], upper))
            {
                upper = keys[i];
            }
        }

        return Interpolate(lower, TOrder.FromKey(upper), g);
    }

    private static void Quantiles<T, TKey, TOrder>(ReadOnlyVectorSpan<T> x, ReadOnlySpan<double> probabilities, bool ignoreNaN, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TKey : unmanaged
        where TOrder : IOrderKey<T, TKey>
    {
        TKey[] rented = ArrayPool<TKey>.Shared.Rent(Math.Max(1, x.Length));
        try
        {
            bool collected = CollectKeys<T, TKey, TOrder>(x, ignoreNaN, rented, out int n, out T nan);
            if (!collected || n == 0)
            {
                T value = collected ? T.NaN : nan;
                for (int k = 0; k < probabilities.Length; k++)
                {
                    destination[k] = value;
                }

                return;
            }

            // 確率が複数なら、選択を繰り返すより一度並べ替えるほうが速い。整数の鍵は大小が全順序と一致するので、整数の既定の比較で並べ替える
            var keys = rented.AsSpan(0, n);
            if (typeof(TKey) == typeof(T))
            {
                keys.Sort(new KeyComparer<TKey, TOrder>());
            }
            else
            {
                keys.Sort();
            }

            for (int k = 0; k < probabilities.Length; k++)
            {
                QuantilePosition(n, probabilities[k], out int j, out double g);
                T lower = TOrder.FromKey(keys[j]);
                destination[k] = g == 0 ? lower : Interpolate(lower, TOrder.FromKey(keys[j + 1]), g);
            }
        }
        finally
        {
            ArrayPool<TKey>.Shared.Return(rented);
        }
    }

    // 非 NaN 要素のキーを work[0..count) に格納する（ignoreNaN が false で NaN を検出したら false）。
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool CollectKeys<T, TKey, TOrder>(ReadOnlyVectorSpan<T> x, bool ignoreNaN, Span<TKey> work, out int count, out T nan)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TKey : unmanaged
        where TOrder : IOrderKey<T, TKey>
    {
        count = 0;
        nan = T.Zero;
        for (int i = 0; i < x.Length; i++)
        {
            T value = x.ElementAt(i);
            if (T.IsNaN(value))
            {
                if (ignoreNaN)
                {
                    continue;
                }

                nan = value;
                return false;
            }

            work[count++] = TOrder.ToKey(value);
        }

        return true;
    }

    // 定義 7 の位置 h = (n - 1) * p の整数部 j と小数部 g（p == 1 時は j = n - 1, g = 0）。
    private static void QuantilePosition(int n, double probability, out int j, out double g)
    {
        double h = (n - 1) * probability;
        j = Math.Min((int)Math.Floor(h), n - 1);
        g = h - j;
    }

    // [a, b] の線形補間（g < 1/2 は a + g*(b - a)、g >= 1/2 は b - (1 - g)*(b - a) で端点範囲・単調性を維持。float / Half は double で計算）。
    private static T Interpolate<T>(T a, T b, double g) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (g == 0.5)
        {
            return Midpoint(a, b);
        }

        if (typeof(T) == typeof(float) || typeof(T) == typeof(Half))
        {
            return T.CreateTruncating(Lerp(double.CreateTruncating(a), double.CreateTruncating(b), g));
        }

        return Lerp(a, b, T.CreateTruncating(g));
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static T Lerp<T>(T a, T b, T g) where T : IFloatingPointIeee754<T>
    {
        if (a == b)
        {
            return a;
        }

        // 無限大の端は極限の値（-∞ と +∞ の間は NaN。中央値と同じ）
        if (T.IsNegativeInfinity(a))
        {
            return T.IsPositiveInfinity(b) ? T.NaN : a;
        }

        if (T.IsPositiveInfinity(b))
        {
            return b;
        }

        T d = b - a;

        // 差がオーバーフローするときは (1 - g)a + gb（絶対値は |a|・|b| 以下なのでオーバーフローしない）
        if (T.IsInfinity(d))
        {
            return (T.One - g) * a + g * b;
        }

        return g < T.One / (T.One + T.One) ? a + g * d : b - (T.One - g) * d;
    }
}
