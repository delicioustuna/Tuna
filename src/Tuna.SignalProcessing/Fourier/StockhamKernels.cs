using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Tuna.SignalProcessing;

internal interface IFourierDirection
{
    static abstract bool IsInverse { get; }
}

internal readonly struct ForwardDirection : IFourierDirection
{
    public static bool IsInverse => false;
}

internal readonly struct InverseDirection : IFourierDirection
{
    public static bool IsInverse => true;
}

internal interface ITwiddleUse
{
    static abstract bool Enabled { get; }
}

internal readonly struct WithTwiddles : ITwiddleUse
{
    public static bool Enabled => true;
}

internal readonly struct WithoutTwiddles : ITwiddleUse
{
    public static bool Enabled => false;
}

internal interface ITwiddleTable
{
    double Real(int j1, int k2);

    double Imaginary(int j1, int k2);
}

internal readonly struct PackedTwiddles(Complex[] table, int radix) : ITwiddleTable
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Real(int j1, int k2) => table[(j1 - 1) * (radix - 1) + k2 - 1].Real;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Imaginary(int j1, int k2) => table[(j1 - 1) * (radix - 1) + k2 - 1].Imaginary;
}

// 第 1 段 SIMD 用に j1 を 4 要素単位でパックした回転因子表。k2 ごとのストリーム分離による L1 キャッシュ衝突を避ける。
internal readonly struct FirstTwiddles(Complex[] table, int radix) : ITwiddleTable
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Real(int j1, int k2) => table[StockhamKernels.FirstIndex(j1, radix) + StockhamKernels.FirstBlock * (k2 - 1)].Real;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double Imaginary(int j1, int k2) => table[StockhamKernels.FirstIndex(j1, radix) + StockhamKernels.FirstBlock * (k2 - 1)].Imaginary;
}

internal interface IRadixKernel
{
    static abstract void Columns<TD, TT, TL, TW>(
        ref Complex x, ref Complex y, nint s, nint count, int m, TW table, int j1First, int j1Last)
        where TD : IFourierDirection
        where TT : ITwiddleUse
        where TL : unmanaged, IComplexLanes<TL>
        where TW : ITwiddleTable;

    static abstract void First<TD, TL>(ref Complex x, ref Complex y, int m, Complex[] table, int j1First, int j1Last)
        where TD : IFourierDirection
        where TL : unmanaged, IComplexLanes<TL>;
}

internal static class StockhamKernels
{
    private const double Sin120 = 0.8660254037844386467637231707529361834715;

    private const double Cos72 = 0.3090169943749474241022934171828190588602;

    private const double Cos144 = -0.8090169943749474241022934171828190588602;

    private const double Sin72 = 0.9510565162951535721164393333793821434057;

    private const double Sin144 = 0.5877852522924731291687059546390727685976;

    // 蝶演算は SIMD 幅ごとに NoInlining メソッドへ分離し、JIT のインライン展開上限超過を防ぐ。
    internal static void Run<TD>(StockhamStage stage, ref Complex input, ref Complex output, nint stride, int part = 0, int parts = 1)
        where TD : IFourierDirection
    {
        int m = stage.M;
        int j1First = 0, j1Last = m;
        nint qFirst = 0, count = stride;
        if (parts > 1)
        {
            if (stride == 1)
            {
                // 第 1 段の j1 方向 SIMD に合わせ、分割境界を FirstBlock (4) の倍数に揃える。
                j1First = (int)Boundary(m, part, parts, FirstBlock);
                j1Last = (int)Boundary(m, part + 1, parts, FirstBlock);
            }
            else if (m >= parts)
            {
                j1First = (int)Boundary(m, part, parts, 1);
                j1Last = (int)Boundary(m, part + 1, parts, 1);
            }
            else
            {
                qFirst = Boundary(stride, part, parts, FirstBlock);
                count = Boundary(stride, part + 1, parts, FirstBlock) - qFirst;
            }

            if (j1First >= j1Last || count <= 0)
            {
                return;
            }
        }

        ref Complex x = ref Unsafe.Add(ref input, qFirst);
        ref Complex y = ref Unsafe.Add(ref output, qFirst);
        switch (stage.Radix)
        {
            case 2:
                RunRadix<TD, Radix2Kernel>(stage, ref x, ref y, stride, count, j1First, j1Last);
                break;
            case 3:
                RunRadix<TD, Radix3Kernel>(stage, ref x, ref y, stride, count, j1First, j1Last);
                break;
            case 4:
                RunRadix<TD, Radix4Kernel>(stage, ref x, ref y, stride, count, j1First, j1Last);
                break;
            case 5:
                RunRadix<TD, Radix5Kernel>(stage, ref x, ref y, stride, count, j1First, j1Last);
                break;
            case 8:
                RunRadix<TD, Radix8Kernel>(stage, ref x, ref y, stride, count, j1First, j1Last);
                break;
            default:
                GenericOdd<TD>(stage, ref x, ref y, stride, count, j1First, j1Last);
                break;
        }
    }

    private static nint Boundary(nint total, int part, int parts, int unit) =>
        part >= parts ? total : total * part / parts / unit * unit;

    internal static bool HasKernel(int radix) => radix is 2 or 3 or 4 or 5 or 8;

    internal const int FirstBlock = 4;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int FirstIndex(int j1, int radix) => (j1 / FirstBlock) * (radix - 1) * FirstBlock + j1 % FirstBlock;

    internal static int FirstLength(int m, int radix) => (m + FirstBlock - 1) / FirstBlock * FirstBlock * (radix - 1);

    private static void RunRadix<TD, TK>(StockhamStage stage, ref Complex x, ref Complex y, nint s, nint count, int j1First, int j1Last)
        where TD : IFourierDirection
        where TK : IRadixKernel
    {
        int m = stage.M;
        if (stage.FirstTwiddles is { } first)
        {
            if (s == 1)
            {
                int full = j1First;
                if (Lanes4.IsAccelerated)
                {
                    full += (j1Last - j1First) & ~3;
                    if (full > j1First)
                    {
                        TK.First<TD, Lanes4>(ref x, ref y, m, first, j1First, full);
                    }
                }
                else if (Lanes2.IsAccelerated)
                {
                    full += (j1Last - j1First) & ~1;
                    if (full > j1First)
                    {
                        TK.First<TD, Lanes2>(ref x, ref y, m, first, j1First, full);
                    }
                }

                TK.First<TD, Lanes1>(ref x, ref y, m, first, Math.Max(full, 1), j1Last);

                // j1 = 0 に回転因子 1+0i を掛けると -0.0 の符号や無限大の扱いが変わるため、乗算なし経路で上書きする。
                if (j1First == 0)
                {
                    TK.Columns<TD, WithoutTwiddles, Lanes1, FirstTwiddles>(ref x, ref y, 1, 1, m, default, 0, 1);
                }

                return;
            }

            RunColumns<TD, TK, FirstTwiddles>(ref x, ref y, s, count, m, new FirstTwiddles(first, stage.Radix), j1First, j1Last);
            return;
        }

        RunColumns<TD, TK, PackedTwiddles>(ref x, ref y, s, count, m, new PackedTwiddles(stage.Twiddles, stage.Radix), j1First, j1Last);
    }

    private static void RunColumns<TD, TK, TW>(ref Complex x, ref Complex y, nint s, nint count, int m, TW table, int j1First, int j1Last)
        where TD : IFourierDirection
        where TK : IRadixKernel
        where TW : ITwiddleTable
    {
        int start = Math.Max(j1First, 1);
        if (Lanes4.IsAccelerated && count >= 4)
        {
            if (j1First == 0)
            {
                TK.Columns<TD, WithoutTwiddles, Lanes4, TW>(ref x, ref y, s, count, m, table, 0, 1);
            }

            TK.Columns<TD, WithTwiddles, Lanes4, TW>(ref x, ref y, s, count, m, table, start, j1Last);
        }
        else if (Lanes2.IsAccelerated && count >= 2)
        {
            if (j1First == 0)
            {
                TK.Columns<TD, WithoutTwiddles, Lanes2, TW>(ref x, ref y, s, count, m, table, 0, 1);
            }

            TK.Columns<TD, WithTwiddles, Lanes2, TW>(ref x, ref y, s, count, m, table, start, j1Last);
        }
        else
        {
            if (j1First == 0)
            {
                TK.Columns<TD, WithoutTwiddles, Lanes1, TW>(ref x, ref y, s, count, m, table, 0, 1);
            }

            TK.Columns<TD, WithTwiddles, Lanes1, TW>(ref x, ref y, s, count, m, table, start, j1Last);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static TL Rotate<TD, TL>(TL x)
        where TD : IFourierDirection
        where TL : unmanaged, IComplexLanes<TL> =>
        TD.IsInverse ? TL.MultiplyI(x) : TL.MultiplyMinusI(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static TL Twiddle<TD, TL>(TL x, TL re, TL im)
        where TD : IFourierDirection
        where TL : unmanaged, IComplexLanes<TL> =>
        TL.Add(TL.Product(x, re), TL.Product(TD.IsInverse ? TL.MultiplyMinusI(x) : TL.MultiplyI(x), im));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static TL TwiddleEach<TD, TL>(TL x, ref Complex w)
        where TD : IFourierDirection
        where TL : unmanaged, IComplexLanes<TL> =>
        TD.IsInverse ? TL.MultiplyEachConjugate(x, TL.Load(in w)) : TL.MultiplyEach(x, TL.Load(in w));

    private readonly struct Radix2Kernel : IRadixKernel
    {
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void Columns<TD, TT, TL, TW>(
            ref Complex x, ref Complex y, nint s, nint count, int m, TW table, int j1First, int j1Last)
            where TD : IFourierDirection
            where TT : ITwiddleUse
            where TL : unmanaged, IComplexLanes<TL>
            where TW : ITwiddleTable
        {
            nint xs = s * m;
            nint last = count - TL.Count;
            TL r1 = default, i1 = default;
            for (int j1 = j1First; j1 < j1Last; j1++)
            {
                ref Complex xj = ref Unsafe.Add(ref x, s * j1);
                ref Complex yj = ref Unsafe.Add(ref y, s * 2 * j1);
                if (TT.Enabled)
                {
                    r1 = TL.Broadcast(table.Real(j1, 1));
                    i1 = TL.Broadcast(table.Imaginary(j1, 1));
                }

                for (nint q = 0; ; q += TL.Count)
                {
                    q = Math.Min(q, last);
                    ref Complex xq = ref Unsafe.Add(ref xj, q);
                    ref Complex yq = ref Unsafe.Add(ref yj, q);
                    TL a0 = TL.Load(in xq);
                    TL a1 = TL.Load(in Unsafe.Add(ref xq, xs));
                    TL b1 = TL.Subtract(a0, a1);
                    TL.Store(ref yq, TL.Add(a0, a1));
                    TL.Store(ref Unsafe.Add(ref yq, s), TT.Enabled ? Twiddle<TD, TL>(b1, r1, i1) : b1);

                    if (q == last)
                    {
                        break;
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void First<TD, TL>(ref Complex x, ref Complex y, int m, Complex[] table, int j1First, int j1Last)
            where TD : IFourierDirection
            where TL : unmanaged, IComplexLanes<TL>
        {
            ref Complex w = ref MemoryMarshal.GetArrayDataReference(table);
            for (int j1 = j1First; j1 < j1Last; j1 += TL.Count)
            {
                ref Complex xj = ref Unsafe.Add(ref x, j1);
                ref Complex yj = ref Unsafe.Add(ref y, 2 * j1);
                ref Complex wj = ref Unsafe.Add(ref w, FirstIndex(j1, 2));
                TL a0 = TL.Load(in xj);
                TL a1 = TL.Load(in Unsafe.Add(ref xj, m));
                TL.StoreLanes(ref yj, 2, TL.Add(a0, a1));
                TL.StoreLanes(ref Unsafe.Add(ref yj, 1), 2, TwiddleEach<TD, TL>(TL.Subtract(a0, a1), ref wj));
            }
        }
    }

    // ---- 基数 3 ----

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Radix3Core<TD, TL>(TL a0, TL a1, TL a2, out TL b0, out TL b1, out TL b2)
        where TD : IFourierDirection
        where TL : unmanaged, IComplexLanes<TL>
    {
        // b1 = a0 − (a1 + a2)/2 ∓ i·(√3/2)·(a1 − a2)、b2 は虚部の項の符号が逆
        TL t1 = TL.Add(a1, a2);
        TL t2 = TL.Subtract(a1, a2);
        TL m1 = TL.Add(a0, TL.Scale(t1, -0.5));
        TL m2 = TL.Scale(Rotate<TD, TL>(t2), Sin120);
        b0 = TL.Add(a0, t1);
        b1 = TL.Add(m1, m2);
        b2 = TL.Subtract(m1, m2);
    }

    private readonly struct Radix3Kernel : IRadixKernel
    {
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void Columns<TD, TT, TL, TW>(
            ref Complex x, ref Complex y, nint s, nint count, int m, TW table, int j1First, int j1Last)
            where TD : IFourierDirection
            where TT : ITwiddleUse
            where TL : unmanaged, IComplexLanes<TL>
            where TW : ITwiddleTable
        {
            nint xs = s * m;
            nint last = count - TL.Count;
            TL r1 = default, i1 = default, r2 = default, i2 = default;
            for (int j1 = j1First; j1 < j1Last; j1++)
            {
                ref Complex xj = ref Unsafe.Add(ref x, s * j1);
                ref Complex yj = ref Unsafe.Add(ref y, s * 3 * j1);
                if (TT.Enabled)
                {
                    r1 = TL.Broadcast(table.Real(j1, 1));
                    i1 = TL.Broadcast(table.Imaginary(j1, 1));
                    r2 = TL.Broadcast(table.Real(j1, 2));
                    i2 = TL.Broadcast(table.Imaginary(j1, 2));
                }

                for (nint q = 0; ; q += TL.Count)
                {
                    q = Math.Min(q, last);
                    ref Complex xq = ref Unsafe.Add(ref xj, q);
                    ref Complex yq = ref Unsafe.Add(ref yj, q);
                    Radix3Core<TD, TL>(
                        TL.Load(in xq), TL.Load(in Unsafe.Add(ref xq, xs)), TL.Load(in Unsafe.Add(ref xq, 2 * xs)),
                        out TL b0, out TL b1, out TL b2);
                    TL.Store(ref yq, b0);
                    TL.Store(ref Unsafe.Add(ref yq, s), TT.Enabled ? Twiddle<TD, TL>(b1, r1, i1) : b1);
                    TL.Store(ref Unsafe.Add(ref yq, 2 * s), TT.Enabled ? Twiddle<TD, TL>(b2, r2, i2) : b2);

                    if (q == last)
                    {
                        break;
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void First<TD, TL>(ref Complex x, ref Complex y, int m, Complex[] table, int j1First, int j1Last)
            where TD : IFourierDirection
            where TL : unmanaged, IComplexLanes<TL>
        {
            ref Complex w = ref MemoryMarshal.GetArrayDataReference(table);
            for (int j1 = j1First; j1 < j1Last; j1 += TL.Count)
            {
                ref Complex xj = ref Unsafe.Add(ref x, j1);
                ref Complex yj = ref Unsafe.Add(ref y, 3 * j1);
                ref Complex wj = ref Unsafe.Add(ref w, FirstIndex(j1, 3));
                Radix3Core<TD, TL>(
                    TL.Load(in xj), TL.Load(in Unsafe.Add(ref xj, m)), TL.Load(in Unsafe.Add(ref xj, 2 * m)),
                    out TL b0, out TL b1, out TL b2);
                TL.StoreLanes(ref yj, 3, b0);
                TL.StoreLanes(ref Unsafe.Add(ref yj, 1), 3, TwiddleEach<TD, TL>(b1, ref wj));
                TL.StoreLanes(ref Unsafe.Add(ref yj, 2), 3, TwiddleEach<TD, TL>(b2, ref Unsafe.Add(ref wj, 4)));
            }
        }
    }

    // ---- 基数 4 ----

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Radix4Core<TD, TL>(TL a0, TL a1, TL a2, TL a3, out TL b0, out TL b1, out TL b2, out TL b3)
        where TD : IFourierDirection
        where TL : unmanaged, IComplexLanes<TL>
    {
        // b1 = (a0 − a2) ∓ i·(a1 − a3)、b3 = (a0 − a2) ± i·(a1 − a3)
        TL t0 = TL.Add(a0, a2);
        TL t1 = TL.Subtract(a0, a2);
        TL t2 = TL.Add(a1, a3);
        TL t3 = Rotate<TD, TL>(TL.Subtract(a1, a3));
        b0 = TL.Add(t0, t2);
        b1 = TL.Add(t1, t3);
        b2 = TL.Subtract(t0, t2);
        b3 = TL.Subtract(t1, t3);
    }

    private readonly struct Radix4Kernel : IRadixKernel
    {
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void Columns<TD, TT, TL, TW>(
            ref Complex x, ref Complex y, nint s, nint count, int m, TW table, int j1First, int j1Last)
            where TD : IFourierDirection
            where TT : ITwiddleUse
            where TL : unmanaged, IComplexLanes<TL>
            where TW : ITwiddleTable
        {
            nint xs = s * m;
            nint last = count - TL.Count;
            TL r1 = default, i1 = default, r2 = default, i2 = default, r3 = default, i3 = default;
            for (int j1 = j1First; j1 < j1Last; j1++)
            {
                ref Complex xj = ref Unsafe.Add(ref x, s * j1);
                ref Complex yj = ref Unsafe.Add(ref y, s * 4 * j1);
                if (TT.Enabled)
                {
                    r1 = TL.Broadcast(table.Real(j1, 1));
                    i1 = TL.Broadcast(table.Imaginary(j1, 1));
                    r2 = TL.Broadcast(table.Real(j1, 2));
                    i2 = TL.Broadcast(table.Imaginary(j1, 2));
                    r3 = TL.Broadcast(table.Real(j1, 3));
                    i3 = TL.Broadcast(table.Imaginary(j1, 3));
                }

                for (nint q = 0; ; q += TL.Count)
                {
                    q = Math.Min(q, last);
                    ref Complex xq = ref Unsafe.Add(ref xj, q);
                    ref Complex yq = ref Unsafe.Add(ref yj, q);
                    Radix4Core<TD, TL>(
                        TL.Load(in xq), TL.Load(in Unsafe.Add(ref xq, xs)),
                        TL.Load(in Unsafe.Add(ref xq, 2 * xs)), TL.Load(in Unsafe.Add(ref xq, 3 * xs)),
                        out TL b0, out TL b1, out TL b2, out TL b3);
                    TL.Store(ref yq, b0);
                    TL.Store(ref Unsafe.Add(ref yq, s), TT.Enabled ? Twiddle<TD, TL>(b1, r1, i1) : b1);
                    TL.Store(ref Unsafe.Add(ref yq, 2 * s), TT.Enabled ? Twiddle<TD, TL>(b2, r2, i2) : b2);
                    TL.Store(ref Unsafe.Add(ref yq, 3 * s), TT.Enabled ? Twiddle<TD, TL>(b3, r3, i3) : b3);

                    if (q == last)
                    {
                        break;
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void First<TD, TL>(ref Complex x, ref Complex y, int m, Complex[] table, int j1First, int j1Last)
            where TD : IFourierDirection
            where TL : unmanaged, IComplexLanes<TL>
        {
            ref Complex w = ref MemoryMarshal.GetArrayDataReference(table);
            for (int j1 = j1First; j1 < j1Last; j1 += TL.Count)
            {
                ref Complex xj = ref Unsafe.Add(ref x, j1);
                ref Complex yj = ref Unsafe.Add(ref y, 4 * j1);
                ref Complex wj = ref Unsafe.Add(ref w, FirstIndex(j1, 4));
                Radix4Core<TD, TL>(
                    TL.Load(in xj), TL.Load(in Unsafe.Add(ref xj, m)),
                    TL.Load(in Unsafe.Add(ref xj, 2 * m)), TL.Load(in Unsafe.Add(ref xj, 3 * m)),
                    out TL b0, out TL b1, out TL b2, out TL b3);
                TL.StoreLanes(ref yj, 4, b0);
                TL.StoreLanes(ref Unsafe.Add(ref yj, 1), 4, TwiddleEach<TD, TL>(b1, ref wj));
                TL.StoreLanes(ref Unsafe.Add(ref yj, 2), 4, TwiddleEach<TD, TL>(b2, ref Unsafe.Add(ref wj, 4)));
                TL.StoreLanes(ref Unsafe.Add(ref yj, 3), 4, TwiddleEach<TD, TL>(b3, ref Unsafe.Add(ref wj, 8)));
            }
        }
    }

    // ---- 基数 5 ----

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Radix5Core<TD, TL>(TL a0, TL a1, TL a2, TL a3, TL a4, out TL b0, out TL b1, out TL b2, out TL b3, out TL b4)
        where TD : IFourierDirection
        where TL : unmanaged, IComplexLanes<TL>
    {
        // 対称な組 (a1, a4)・(a2, a3) の和と差に分け、実部側 r と虚部側 i を作る
        TL t1 = TL.Add(a1, a4);
        TL t2 = TL.Add(a2, a3);
        TL d1 = TL.Subtract(a1, a4);
        TL d2 = TL.Subtract(a2, a3);
        TL r1 = TL.Add(TL.Add(a0, TL.Scale(t1, Cos72)), TL.Scale(t2, Cos144));
        TL r2 = TL.Add(TL.Add(a0, TL.Scale(t1, Cos144)), TL.Scale(t2, Cos72));
        TL i1 = Rotate<TD, TL>(TL.Add(TL.Scale(d1, Sin72), TL.Scale(d2, Sin144)));
        TL i2 = Rotate<TD, TL>(TL.Subtract(TL.Scale(d1, Sin144), TL.Scale(d2, Sin72)));
        b0 = TL.Add(TL.Add(a0, t1), t2);
        b1 = TL.Add(r1, i1);
        b2 = TL.Add(r2, i2);
        b3 = TL.Subtract(r2, i2);
        b4 = TL.Subtract(r1, i1);
    }

    private readonly struct Radix5Kernel : IRadixKernel
    {
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void Columns<TD, TT, TL, TW>(
            ref Complex x, ref Complex y, nint s, nint count, int m, TW table, int j1First, int j1Last)
            where TD : IFourierDirection
            where TT : ITwiddleUse
            where TL : unmanaged, IComplexLanes<TL>
            where TW : ITwiddleTable
        {
            nint xs = s * m;
            nint last = count - TL.Count;
            TL r1 = default, i1 = default, r2 = default, i2 = default, r3 = default, i3 = default, r4 = default, i4 = default;
            for (int j1 = j1First; j1 < j1Last; j1++)
            {
                ref Complex xj = ref Unsafe.Add(ref x, s * j1);
                ref Complex yj = ref Unsafe.Add(ref y, s * 5 * j1);
                if (TT.Enabled)
                {
                    r1 = TL.Broadcast(table.Real(j1, 1));
                    i1 = TL.Broadcast(table.Imaginary(j1, 1));
                    r2 = TL.Broadcast(table.Real(j1, 2));
                    i2 = TL.Broadcast(table.Imaginary(j1, 2));
                    r3 = TL.Broadcast(table.Real(j1, 3));
                    i3 = TL.Broadcast(table.Imaginary(j1, 3));
                    r4 = TL.Broadcast(table.Real(j1, 4));
                    i4 = TL.Broadcast(table.Imaginary(j1, 4));
                }

                for (nint q = 0; ; q += TL.Count)
                {
                    q = Math.Min(q, last);
                    ref Complex xq = ref Unsafe.Add(ref xj, q);
                    ref Complex yq = ref Unsafe.Add(ref yj, q);
                    Radix5Core<TD, TL>(
                        TL.Load(in xq), TL.Load(in Unsafe.Add(ref xq, xs)), TL.Load(in Unsafe.Add(ref xq, 2 * xs)),
                        TL.Load(in Unsafe.Add(ref xq, 3 * xs)), TL.Load(in Unsafe.Add(ref xq, 4 * xs)),
                        out TL b0, out TL b1, out TL b2, out TL b3, out TL b4);
                    TL.Store(ref yq, b0);
                    TL.Store(ref Unsafe.Add(ref yq, s), TT.Enabled ? Twiddle<TD, TL>(b1, r1, i1) : b1);
                    TL.Store(ref Unsafe.Add(ref yq, 2 * s), TT.Enabled ? Twiddle<TD, TL>(b2, r2, i2) : b2);
                    TL.Store(ref Unsafe.Add(ref yq, 3 * s), TT.Enabled ? Twiddle<TD, TL>(b3, r3, i3) : b3);
                    TL.Store(ref Unsafe.Add(ref yq, 4 * s), TT.Enabled ? Twiddle<TD, TL>(b4, r4, i4) : b4);

                    if (q == last)
                    {
                        break;
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void First<TD, TL>(ref Complex x, ref Complex y, int m, Complex[] table, int j1First, int j1Last)
            where TD : IFourierDirection
            where TL : unmanaged, IComplexLanes<TL>
        {
            ref Complex w = ref MemoryMarshal.GetArrayDataReference(table);
            for (int j1 = j1First; j1 < j1Last; j1 += TL.Count)
            {
                ref Complex xj = ref Unsafe.Add(ref x, j1);
                ref Complex yj = ref Unsafe.Add(ref y, 5 * j1);
                ref Complex wj = ref Unsafe.Add(ref w, FirstIndex(j1, 5));
                Radix5Core<TD, TL>(
                    TL.Load(in xj), TL.Load(in Unsafe.Add(ref xj, m)), TL.Load(in Unsafe.Add(ref xj, 2 * m)),
                    TL.Load(in Unsafe.Add(ref xj, 3 * m)), TL.Load(in Unsafe.Add(ref xj, 4 * m)),
                    out TL b0, out TL b1, out TL b2, out TL b3, out TL b4);
                TL.StoreLanes(ref yj, 5, b0);
                TL.StoreLanes(ref Unsafe.Add(ref yj, 1), 5, TwiddleEach<TD, TL>(b1, ref wj));
                TL.StoreLanes(ref Unsafe.Add(ref yj, 2), 5, TwiddleEach<TD, TL>(b2, ref Unsafe.Add(ref wj, 4)));
                TL.StoreLanes(ref Unsafe.Add(ref yj, 3), 5, TwiddleEach<TD, TL>(b3, ref Unsafe.Add(ref wj, 8)));
                TL.StoreLanes(ref Unsafe.Add(ref yj, 4), 5, TwiddleEach<TD, TL>(b4, ref Unsafe.Add(ref wj, 12)));
            }
        }
    }

    // ---- 基数 8 ----

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Radix8Core<TD, TL>(
        TL a0, TL a1, TL a2, TL a3, TL a4, TL a5, TL a6, TL a7,
        out TL b0, out TL b1, out TL b2, out TL b3, out TL b4, out TL b5, out TL b6, out TL b7)
        where TD : IFourierDirection
        where TL : unmanaged, IComplexLanes<TL>
    {
        // 偶数番の4点の DFT E と奇数番の4点の DFT O から、b_k = E_k + ω₈^k·O_k、b_(k+4) = E_k − ω₈^k·O_k（k ∈ [0, 4)）。
        // ω₈ = (1 ∓ i)/√2 なので ω₈·O = (O + R(O))/√2、ω₈²·O = R(O)、ω₈³·O = (R(O) − O)/√2（R は ∓i 倍）
        Radix4Core<TD, TL>(a0, a2, a4, a6, out TL e0, out TL e1, out TL e2, out TL e3);
        Radix4Core<TD, TL>(a1, a3, a5, a7, out TL o0, out TL o1, out TL o2, out TL o3);
        TL w1 = TL.Scale(TL.Add(o1, Rotate<TD, TL>(o1)), XMath.Constants.OneOverSqrt2);
        TL w2 = Rotate<TD, TL>(o2);
        TL w3 = TL.Scale(TL.Subtract(Rotate<TD, TL>(o3), o3), XMath.Constants.OneOverSqrt2);
        b0 = TL.Add(e0, o0);
        b1 = TL.Add(e1, w1);
        b2 = TL.Add(e2, w2);
        b3 = TL.Add(e3, w3);
        b4 = TL.Subtract(e0, o0);
        b5 = TL.Subtract(e1, w1);
        b6 = TL.Subtract(e2, w2);
        b7 = TL.Subtract(e3, w3);
    }

    private readonly struct Radix8Kernel : IRadixKernel
    {
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void Columns<TD, TT, TL, TW>(
            ref Complex x, ref Complex y, nint s, nint count, int m, TW table, int j1First, int j1Last)
            where TD : IFourierDirection
            where TT : ITwiddleUse
            where TL : unmanaged, IComplexLanes<TL>
            where TW : ITwiddleTable
        {
            nint xs = s * m;
            nint last = count - TL.Count;
            TL r1 = default, i1 = default, r2 = default, i2 = default, r3 = default, i3 = default, r4 = default, i4 = default;
            TL r5 = default, i5 = default, r6 = default, i6 = default, r7 = default, i7 = default;
            for (int j1 = j1First; j1 < j1Last; j1++)
            {
                ref Complex xj = ref Unsafe.Add(ref x, s * j1);
                ref Complex yj = ref Unsafe.Add(ref y, s * 8 * j1);
                if (TT.Enabled)
                {
                    r1 = TL.Broadcast(table.Real(j1, 1));
                    i1 = TL.Broadcast(table.Imaginary(j1, 1));
                    r2 = TL.Broadcast(table.Real(j1, 2));
                    i2 = TL.Broadcast(table.Imaginary(j1, 2));
                    r3 = TL.Broadcast(table.Real(j1, 3));
                    i3 = TL.Broadcast(table.Imaginary(j1, 3));
                    r4 = TL.Broadcast(table.Real(j1, 4));
                    i4 = TL.Broadcast(table.Imaginary(j1, 4));
                    r5 = TL.Broadcast(table.Real(j1, 5));
                    i5 = TL.Broadcast(table.Imaginary(j1, 5));
                    r6 = TL.Broadcast(table.Real(j1, 6));
                    i6 = TL.Broadcast(table.Imaginary(j1, 6));
                    r7 = TL.Broadcast(table.Real(j1, 7));
                    i7 = TL.Broadcast(table.Imaginary(j1, 7));
                }

                for (nint q = 0; ; q += TL.Count)
                {
                    q = Math.Min(q, last);
                    ref Complex xq = ref Unsafe.Add(ref xj, q);
                    ref Complex yq = ref Unsafe.Add(ref yj, q);
                    Radix8Core<TD, TL>(
                        TL.Load(in xq), TL.Load(in Unsafe.Add(ref xq, xs)), TL.Load(in Unsafe.Add(ref xq, 2 * xs)),
                        TL.Load(in Unsafe.Add(ref xq, 3 * xs)), TL.Load(in Unsafe.Add(ref xq, 4 * xs)),
                        TL.Load(in Unsafe.Add(ref xq, 5 * xs)), TL.Load(in Unsafe.Add(ref xq, 6 * xs)),
                        TL.Load(in Unsafe.Add(ref xq, 7 * xs)),
                        out TL b0, out TL b1, out TL b2, out TL b3, out TL b4, out TL b5, out TL b6, out TL b7);
                    TL.Store(ref yq, b0);
                    TL.Store(ref Unsafe.Add(ref yq, s), TT.Enabled ? Twiddle<TD, TL>(b1, r1, i1) : b1);
                    TL.Store(ref Unsafe.Add(ref yq, 2 * s), TT.Enabled ? Twiddle<TD, TL>(b2, r2, i2) : b2);
                    TL.Store(ref Unsafe.Add(ref yq, 3 * s), TT.Enabled ? Twiddle<TD, TL>(b3, r3, i3) : b3);
                    TL.Store(ref Unsafe.Add(ref yq, 4 * s), TT.Enabled ? Twiddle<TD, TL>(b4, r4, i4) : b4);
                    TL.Store(ref Unsafe.Add(ref yq, 5 * s), TT.Enabled ? Twiddle<TD, TL>(b5, r5, i5) : b5);
                    TL.Store(ref Unsafe.Add(ref yq, 6 * s), TT.Enabled ? Twiddle<TD, TL>(b6, r6, i6) : b6);
                    TL.Store(ref Unsafe.Add(ref yq, 7 * s), TT.Enabled ? Twiddle<TD, TL>(b7, r7, i7) : b7);

                    if (q == last)
                    {
                        break;
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void First<TD, TL>(ref Complex x, ref Complex y, int m, Complex[] table, int j1First, int j1Last)
            where TD : IFourierDirection
            where TL : unmanaged, IComplexLanes<TL>
        {
            ref Complex w = ref MemoryMarshal.GetArrayDataReference(table);
            for (int j1 = j1First; j1 < j1Last; j1 += TL.Count)
            {
                ref Complex xj = ref Unsafe.Add(ref x, j1);
                ref Complex yj = ref Unsafe.Add(ref y, 8 * j1);
                ref Complex wj = ref Unsafe.Add(ref w, FirstIndex(j1, 8));
                Radix8Core<TD, TL>(
                    TL.Load(in xj), TL.Load(in Unsafe.Add(ref xj, m)), TL.Load(in Unsafe.Add(ref xj, 2 * m)),
                    TL.Load(in Unsafe.Add(ref xj, 3 * m)), TL.Load(in Unsafe.Add(ref xj, 4 * m)),
                    TL.Load(in Unsafe.Add(ref xj, 5 * m)), TL.Load(in Unsafe.Add(ref xj, 6 * m)),
                    TL.Load(in Unsafe.Add(ref xj, 7 * m)),
                    out TL b0, out TL b1, out TL b2, out TL b3, out TL b4, out TL b5, out TL b6, out TL b7);
                TL.StoreLanes(ref yj, 8, b0);
                TL.StoreLanes(ref Unsafe.Add(ref yj, 1), 8, TwiddleEach<TD, TL>(b1, ref wj));
                TL.StoreLanes(ref Unsafe.Add(ref yj, 2), 8, TwiddleEach<TD, TL>(b2, ref Unsafe.Add(ref wj, 4)));
                TL.StoreLanes(ref Unsafe.Add(ref yj, 3), 8, TwiddleEach<TD, TL>(b3, ref Unsafe.Add(ref wj, 8)));
                TL.StoreLanes(ref Unsafe.Add(ref yj, 4), 8, TwiddleEach<TD, TL>(b4, ref Unsafe.Add(ref wj, 12)));
                TL.StoreLanes(ref Unsafe.Add(ref yj, 5), 8, TwiddleEach<TD, TL>(b5, ref Unsafe.Add(ref wj, 16)));
                TL.StoreLanes(ref Unsafe.Add(ref yj, 6), 8, TwiddleEach<TD, TL>(b6, ref Unsafe.Add(ref wj, 20)));
                TL.StoreLanes(ref Unsafe.Add(ref yj, 7), 8, TwiddleEach<TD, TL>(b7, ref Unsafe.Add(ref wj, 24)));
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TL TwiddleComplex<TD, TL>(TL x, Complex w)
        where TD : IFourierDirection
        where TL : unmanaged, IComplexLanes<TL> =>
        TD.IsInverse ? TL.MultiplyConjugate(x, w) : TL.Multiply(x, w);

    private static void GenericOdd<TD>(StockhamStage stage, ref Complex x, ref Complex y, nint s, nint count, int j1First, int j1Last)
        where TD : IFourierDirection
    {
        nint q4 = Lanes4.IsAccelerated ? count - count % 4 : 0;
        nint q2 = Lanes2.IsAccelerated ? q4 + (count - q4) / 2 * 2 : q4;
        GenericOddPass<TD, Lanes4>(stage, ref x, ref y, s, 0, q4, j1First, j1Last);
        GenericOddPass<TD, Lanes2>(stage, ref x, ref y, s, q4, q2, j1First, j1Last);
        GenericOddPass<TD, Lanes1>(stage, ref x, ref y, s, q2, count, j1First, j1Last);
    }

    private static void GenericOddPass<TD, TL>(
        StockhamStage stage, ref Complex x, ref Complex y, nint s, nint first, nint last, int j1First, int j1Last)
        where TD : IFourierDirection
        where TL : unmanaged, IComplexLanes<TL>
    {
        if (first >= last)
        {
            return;
        }

        int p = stage.Radix;
        int m = stage.M;
        int h = p / 2;
        Complex[] w = stage.Twiddles;
        double[] cos = stage.Cos!;
        double[] sin = stage.Sin!;
        Span<TL> sums = stackalloc TL[h];
        Span<TL> differences = stackalloc TL[h];
        nint xs = s * m;
        for (int j1 = j1First; j1 < j1Last; j1++)
        {
            ref Complex xj = ref Unsafe.Add(ref x, s * j1);
            ref Complex yj = ref Unsafe.Add(ref y, s * p * j1);
            int t = (p - 1) * (j1 - 1);
            for (nint q = first; q < last; q += TL.Count)
            {
                ref Complex xq = ref Unsafe.Add(ref xj, q);
                ref Complex yq = ref Unsafe.Add(ref yj, q);
                TL a0 = TL.Load(in xq);
                TL b0 = a0;
                for (int j = 1; j <= h; j++)
                {
                    TL aj = TL.Load(in Unsafe.Add(ref xq, j * xs));
                    TL ap = TL.Load(in Unsafe.Add(ref xq, (p - j) * xs));
                    sums[j - 1] = TL.Add(aj, ap);
                    differences[j - 1] = TL.Subtract(aj, ap);
                    b0 = TL.Add(b0, sums[j - 1]);
                }

                TL.Store(ref yq, b0);
                for (int k = 1; k <= h; k++)
                {
                    // re = a0 + Σ_j (a_j + a_{p−j})·cos(2πjk/p)、im = Σ_j (a_j − a_{p−j})·sin(2πjk/p)
                    TL re = TL.Add(a0, TL.Scale(sums[0], cos[k]));
                    TL im = TL.Scale(differences[0], sin[k]);
                    int r = k;
                    for (int j = 2; j <= h; j++)
                    {
                        r += k;
                        if (r >= p)
                        {
                            r -= p;
                        }

                        re = TL.Add(re, TL.Scale(sums[j - 1], cos[r]));
                        im = TL.Add(im, TL.Scale(differences[j - 1], sin[r]));
                    }

                    TL rotated = Rotate<TD, TL>(im);
                    TL bk = TL.Add(re, rotated);
                    TL bpk = TL.Subtract(re, rotated);
                    if (j1 > 0)
                    {
                        bk = TwiddleComplex<TD, TL>(bk, w[t + k - 1]);
                        bpk = TwiddleComplex<TD, TL>(bpk, w[t + p - k - 1]);
                    }

                    TL.Store(ref Unsafe.Add(ref yq, k * s), bk);
                    TL.Store(ref Unsafe.Add(ref yq, (p - k) * s), bpk);
                }
            }
        }
    }
}
