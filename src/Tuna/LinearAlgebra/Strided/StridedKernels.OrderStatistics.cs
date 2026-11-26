using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Tuna.LinearAlgebra;

internal interface IKeyOrder<TKey>
{
    static abstract bool Less(TKey x, TKey y);
}

// 値と全順序キー（-∞ < ... < -0 < +0 < ... < +∞、NaN は事前除外）の相互変換。
internal interface IOrderKey<T, TKey> : IKeyOrder<TKey>
{
    static abstract TKey ToKey(T value);

    static abstract T FromKey(TKey key);
}

// IEEE 754 ビット列の負数のみ符号以外を反転した符号付き整数キー（整数の大小が全順序と一致し、変換は自己逆写像）。
internal readonly struct BitsKey<T, TKey> : IOrderKey<T, TKey>
    where T : unmanaged
    where TKey : unmanaged, IBinaryInteger<TKey>, ISignedNumber<TKey>, IMinMaxValue<TKey>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TKey Flip(TKey bits) => bits ^ ((bits >> (Unsafe.SizeOf<TKey>() * 8 - 1)) & TKey.MaxValue);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static TKey ToKey(T value) => Flip(Unsafe.BitCast<T, TKey>(value));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T FromKey(TKey key) => Unsafe.BitCast<TKey, T>(Flip(key));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool Less(TKey x, TKey y) => x < y;
}

// double / float / Half 以外の型向け全順序キー（-0 < +0）。
internal readonly struct TotalOrderKey<T> : IOrderKey<T, T> where T : IFloatingPointIeee754<T>
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T ToKey(T value) => value;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T FromKey(T key) => key;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool Less(T x, T y) => x < y || (x == y && T.IsNegative(x) && !T.IsNegative(y));
}

internal readonly struct KeyComparer<TKey, TOrder> : IComparer<TKey>
    where TKey : unmanaged
    where TOrder : IKeyOrder<TKey>
{
    public int Compare(TKey x, TKey y) => TOrder.Less(x, y) ? -1 : TOrder.Less(y, x) ? 1 : 0;
}

// 中央値計算（Floyd-Rivest 選択アルゴリズム + 予算超過時のソートフォールバックによる最悪 O(n log n) 保証）。
// double / float / Half は BitsKey で整数比較し、キー関数を即時最適化ループへインライン展開する。
internal static partial class StridedKernels
{
    // Floyd & Rivest が標本から範囲を絞る長さの下限（論文の値）
    private const int SelectSampleThreshold = 600;

    internal static T Median<T>(ReadOnlyVectorSpan<T> x, bool ignoreNaN) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (typeof(T) == typeof(double))
        {
            return Median<T, long, BitsKey<T, long>>(x, ignoreNaN);
        }

        if (typeof(T) == typeof(float))
        {
            return Median<T, int, BitsKey<T, int>>(x, ignoreNaN);
        }

        if (typeof(T) == typeof(Half))
        {
            return Median<T, short, BitsKey<T, short>>(x, ignoreNaN);
        }

        return Median<T, T, TotalOrderKey<T>>(x, ignoreNaN);
    }

    internal static void ColumnMedians<T>(ReadOnlyMatrixSpan<T> a, bool ignoreNaN, VectorSpan<T> destination) where T : unmanaged, IFloatingPointIeee754<T>
    {
        RequireLength(destination, a.Columns);
        RequireDisjoint(a, destination.AsMatrix());
        if (typeof(T) == typeof(double))
        {
            ColumnMedians<T, long, BitsKey<T, long>>(a, ignoreNaN, destination);
        }
        else if (typeof(T) == typeof(float))
        {
            ColumnMedians<T, int, BitsKey<T, int>>(a, ignoreNaN, destination);
        }
        else if (typeof(T) == typeof(Half))
        {
            ColumnMedians<T, short, BitsKey<T, short>>(a, ignoreNaN, destination);
        }
        else
        {
            ColumnMedians<T, T, TotalOrderKey<T>>(a, ignoreNaN, destination);
        }
    }

    internal static void RowMedians<T>(ReadOnlyMatrixSpan<T> a, bool ignoreNaN, VectorSpan<T> destination) where T : unmanaged, IFloatingPointIeee754<T>
    {
        RequireLength(destination, a.Rows);
        RequireDisjoint(a, destination.AsMatrix());
        if (typeof(T) == typeof(double))
        {
            RowMedians<T, long, BitsKey<T, long>>(a, ignoreNaN, destination);
        }
        else if (typeof(T) == typeof(float))
        {
            RowMedians<T, int, BitsKey<T, int>>(a, ignoreNaN, destination);
        }
        else if (typeof(T) == typeof(Half))
        {
            RowMedians<T, short, BitsKey<T, short>>(a, ignoreNaN, destination);
        }
        else
        {
            RowMedians<T, T, TotalOrderKey<T>>(a, ignoreNaN, destination);
        }
    }

    private static T Median<T, TKey, TOrder>(ReadOnlyVectorSpan<T> x, bool ignoreNaN)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TKey : unmanaged
        where TOrder : IOrderKey<T, TKey>
    {
        TKey[] rented = ArrayPool<TKey>.Shared.Rent(Math.Max(1, x.Length));
        try
        {
            return Median<T, TKey, TOrder>(x, ignoreNaN, rented);
        }
        finally
        {
            ArrayPool<TKey>.Shared.Return(rented);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ColumnMedians<T, TKey, TOrder>(ReadOnlyMatrixSpan<T> a, bool ignoreNaN, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TKey : unmanaged
        where TOrder : IOrderKey<T, TKey>
    {
        TKey[] rented = ArrayPool<TKey>.Shared.Rent(Math.Max(1, a.Rows));
        try
        {
            for (int j = 0; j < a.Columns; j++)
            {
                destination[j] = Median<T, TKey, TOrder>(a.Column(j), ignoreNaN, rented);
            }
        }
        finally
        {
            ArrayPool<TKey>.Shared.Return(rented);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void RowMedians<T, TKey, TOrder>(ReadOnlyMatrixSpan<T> a, bool ignoreNaN, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TKey : unmanaged
        where TOrder : IOrderKey<T, TKey>
    {
        TKey[] rented = ArrayPool<TKey>.Shared.Rent(Math.Max(1, a.Columns));
        try
        {
            for (int i = 0; i < a.Rows; i++)
            {
                destination[i] = Median<T, TKey, TOrder>(a.Row(i), ignoreNaN, rented);
            }
        }
        finally
        {
            ArrayPool<TKey>.Shared.Return(rented);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static T Median<T, TKey, TOrder>(ReadOnlyVectorSpan<T> x, bool ignoreNaN, Span<TKey> work)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TKey : unmanaged
        where TOrder : IOrderKey<T, TKey>
    {
        int n = 0;
        for (int i = 0; i < x.Length; i++)
        {
            T value = x.ElementAt(i);
            if (T.IsNaN(value))
            {
                if (ignoreNaN)
                {
                    continue;
                }

                return value;
            }

            work[n++] = TOrder.ToKey(value);
        }

        if (n == 0)
        {
            return T.NaN;
        }

        var keys = work[..n];
        int k = n / 2;
        long budget = SelectBudget(n);
        ulong random = 0;
        Select<TKey, TOrder>(keys, 0, n - 1, k, ref budget, ref random);
        T upper = TOrder.FromKey(keys[k]);
        if ((n & 1) != 0)
        {
            return upper;
        }

        // 偶数なら、k より前（すべて keys[k] 以下）の最大が中央のもう1つ
        TKey lower = keys[0];
        for (int i = 1; i < k; i++)
        {
            if (TOrder.Less(lower, keys[i]))
            {
                lower = keys[i];
            }
        }

        return Midpoint(TOrder.FromKey(lower), upper);
    }

    // 選択の手間（分割した範囲の長さの合計）の上限。乱数・整列済み・山形・のこぎり・少ない種類などの並びは 1.6n〜3n 程度
    // （長さ 1000 前後では 6n 程度まで）で収まり、上限に届くのは選択法を狙った入力（McIlroy の敵対的な比較で作ったものなど）だけ
    private static long SelectBudget(int n) => 8L * n + 1024;

    // 中央の2つ a ≤ b の中点。(a + b)/2 がオーバーフローするときだけ a/2 + b/2 にする（ふつうは足してから半分にして、非正規化数で丸めを増やさない）。
    // -∞ と +∞ の中点は NaN
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static T Midpoint<T>(T a, T b) where T : IFloatingPointIeee754<T>
    {
        T two = T.One + T.One;
        T midpoint = (a + b) / two;
        return T.IsInfinity(midpoint) && T.IsFinite(a) && T.IsFinite(b) ? a / two + b / two : midpoint;
    }

    // Floyd & Rivest (Algorithm 489, CACM 1975) の選択法で keys[k] を確定する。
    // 偏った並び（山形など）を避けるため標本は固定シード擬似乱数で全体から抽出し、予算超過時は残余区間をソートする。
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Select<TKey, TOrder>(Span<TKey> keys, int left, int right, int k, ref long budget, ref ulong random)
        where TKey : unmanaged
        where TOrder : IKeyOrder<TKey>
    {
        while (right > left)
        {
            if (budget < 0)
            {
                keys[left..(right + 1)].Sort(new KeyComparer<TKey, TOrder>());
                return;
            }

            budget -= right - left + 1;
            if (right - left > SelectSampleThreshold)
            {
                int n = right - left + 1;
                int i = k - left + 1;
                double z = Math.Log(n);
                double s = 0.5 * Math.Exp(2 * z / 3);
                double sd = 0.5 * Math.Sqrt(z * s * (n - s) / n) * Math.Sign(i - n / 2);
                int newLeft = Math.Max(left, (int)Math.Floor(k - i * s / n + sd));
                int newRight = Math.Min(right, (int)Math.Floor(k + (n - i) * s / n + sd));
                for (int q = newLeft; q <= newRight; q++)
                {
                    // Knuth の MMIX の線形合同法。上位 32 ビットを [0, n) に写す
                    random = random * 6364136223846793005UL + 1442695040888963407UL;
                    int r = left + (int)(((random >> 32) * (ulong)n) >> 32);
                    (keys[q], keys[r]) = (keys[r], keys[q]);
                }

                Select<TKey, TOrder>(keys, newLeft, newRight, k, ref budget, ref random);
            }

            // keys[k] を軸 t にして [left, right] を分ける。両端に t 以下と t 以上を置いて、内側の走査の番兵にする
            TKey t = keys[k];
            int lo = left, hi = right;
            (keys[left], keys[k]) = (keys[k], keys[left]);
            if (TOrder.Less(t, keys[right]))
            {
                (keys[right], keys[left]) = (keys[left], keys[right]);
            }

            while (lo < hi)
            {
                (keys[lo], keys[hi]) = (keys[hi], keys[lo]);
                lo++;
                hi--;
                while (TOrder.Less(keys[lo], t))
                {
                    lo++;
                }

                while (TOrder.Less(t, keys[hi]))
                {
                    hi--;
                }
            }

            if (!TOrder.Less(keys[left], t) && !TOrder.Less(t, keys[left]))
            {
                (keys[left], keys[hi]) = (keys[hi], keys[left]);
            }
            else
            {
                hi++;
                (keys[hi], keys[right]) = (keys[right], keys[hi]);
            }

            // t は hi に来た。k を含む側だけを続ける
            if (hi <= k)
            {
                left = hi + 1;
            }

            if (k <= hi)
            {
                right = hi - 1;
            }
        }
    }
}
