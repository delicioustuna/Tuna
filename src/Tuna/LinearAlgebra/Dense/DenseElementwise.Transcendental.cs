using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Tuna.LinearAlgebra;

// 指数関数（Tang 1989 テーブル法）および対数関数（区間中心 c での分解法）の SIMD 実装。
// スカラーと全 SIMD 幅で同一の演算順序（FMA 含む）を用い、レイアウトや ISA に依存せずビット単位で再現する。
internal static partial class DenseElementwise
{
    private const double ExpShift = 6755399441055744.0; // 0x1.8p52。足すと最近接の整数に丸まり、下位のビットに整数が入る
    private const double ExpExactLimit = 708;           // |x| ≤ 708 なら e^x は正規化数で、表引きの式がそのまま使える
    private const double ExpZeroBelow = -746;           // x ≤ -746 なら e^x < 2^-1075 で、0 に丸まる
    private const double ExpInfinityAbove = 710;        // x ≥ 710 なら e^x は double の最大値を超える
    private const int ExpTableMask = 127;
    private const ulong LogOffset = 0x3FE6000000000000; // z の範囲 [0x1.6p-1, 0x1.6p0) の下端
    private const ulong LogTableMask = 127;
    private const ulong MinNormalBits = 0x0010000000000000;
    private const ulong PositiveNormalRange = 0x7FE0000000000000; // 正の正規化数は (bits - MinNormalBits) がこれ未満
    private const long DoubleShiftBits = 0x4338000000000000;     // 0x1.8p52 のビット列。整数を足して引き戻すと double になる
    private const ulong QuietBit = 1UL << 51;                    // NaN を静かにするビット

    // e^r - 1 ≈ r + r²·(1/2 + r/6 + r²/24 + r³/120 + r⁴/720)。|r| ≤ ln2/256 で打ち切り誤差は 2^-68 未満
    private const double ExpC2 = 0.5, ExpC3 = 1.0 / 6, ExpC4 = 1.0 / 24, ExpC5 = 1.0 / 120, ExpC6 = 1.0 / 720;

    // ln(1 + r) - r ≈ r²·(-1/2 + r/3 - r²/4 + … + r⁷/9)。|r| ≤ 2^-7 で打ち切り誤差は相対 2^-66 未満
    private const double LogA2 = -0.5, LogA3 = 1.0 / 3, LogA4 = -0.25, LogA5 = 0.2, LogA6 = -1.0 / 6, LogA7 = 1.0 / 7, LogA8 = -0.125, LogA9 = 1.0 / 9;

    private static readonly double Ln2Hi = BitConverter.UInt64BitsToDouble(Ln2HiBits);
    private static readonly double Ln2Lo = BitConverter.UInt64BitsToDouble(Ln2LoBits);
    private static readonly double ExpLn2HiN = BitConverter.UInt64BitsToDouble(ExpLn2HiNBits);
    private static readonly double ExpLn2LoN = BitConverter.UInt64BitsToDouble(ExpLn2LoNBits);
    private static readonly double ExpInvLn2N = BitConverter.UInt64BitsToDouble(ExpInvLn2NBits);

    // AggressiveOptimization メソッドの JIT 前に static readonly フィールドを初期化し、定数畳み込みと境界チェック除去を有効にする。
    static DenseElementwise()
    {
        RuntimeHelpers.RunClassConstructor(typeof(ElementwiseTables).TypeHandle);
        RuntimeHelpers.RunClassConstructor(typeof(ScalarLanes).TypeHandle);
    }

    // ---------- 指数関数 ----------

    // float・double はベクトル実装とビット一致する e^x を返し、それ以外の型は T.Exp に委譲する（縮約との整合性維持）。
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static T Exp<T>(T x) where T : IExponentialFunctions<T>
    {
        if (typeof(T) == typeof(double))
        {
            double y = Exp(Unsafe.As<T, double>(ref x));
            return Unsafe.As<double, T>(ref y);
        }

        if (typeof(T) == typeof(float))
        {
            float y = (float)Exp((double)Unsafe.As<T, float>(ref x));
            return Unsafe.As<float, T>(ref y);
        }

        return T.Exp(x);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static double Exp(double x)
    {
        if (x <= ExpZeroBelow)
        {
            return 0;
        }

        if (x >= ExpInfinityAbove)
        {
            return double.PositiveInfinity;
        }

        if (double.IsNaN(x))
        {
            return Quiet(x);
        }

        if (!(Math.Abs(x) <= ExpExactLimit))
        {
            return Math.Exp(x);
        }

        double z = Math.FusedMultiplyAdd(x, ExpInvLn2N, ExpShift);
        double kd = z - ExpShift;
        ulong ki = BitConverter.DoubleToUInt64Bits(z);
        double r = Math.FusedMultiplyAdd(kd, -ExpLn2HiN, x);
        r = Math.FusedMultiplyAdd(kd, -ExpLn2LoN, r);
        int j = (int)ki & ExpTableMask;
        ref ulong table = ref MemoryMarshal.GetReference(ExpTableBits);
        double scale = BitConverter.UInt64BitsToDouble(Unsafe.Add(ref table, 2 * j) + (ki << 45));
        double tail = BitConverter.UInt64BitsToDouble(Unsafe.Add(ref table, 2 * j + 1));
        double r2 = r * r;
        double q = Math.FusedMultiplyAdd(Math.FusedMultiplyAdd(Math.FusedMultiplyAdd(Math.FusedMultiplyAdd(ExpC6, r, ExpC5), r, ExpC4), r, ExpC3), r, ExpC2);
        double p = Math.FusedMultiplyAdd(r2, q, r);
        return Math.FusedMultiplyAdd(scale, tail + p, scale);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static Vector128<double> Exp(Vector128<double> x)
    {
        var z = Vector128.FusedMultiplyAdd(x, Vector128.Create(ExpInvLn2N), Vector128.Create(ExpShift));
        var kd = z - Vector128.Create(ExpShift);
        var ki = z.AsUInt64();
        var r = Vector128.FusedMultiplyAdd(kd, Vector128.Create(-ExpLn2HiN), x);
        r = Vector128.FusedMultiplyAdd(kd, Vector128.Create(-ExpLn2LoN), r);
        var index = (ki & Vector128.Create((ulong)ExpTableMask)) << 1;
        var scale = (Gather(ExpTableBits, index) + (ki << 45)).AsDouble();
        var tail = Gather(ExpTableBits, index + Vector128<ulong>.One).AsDouble();
        var r2 = r * r;
        var q = Vector128.FusedMultiplyAdd(Vector128.FusedMultiplyAdd(Vector128.FusedMultiplyAdd(Vector128.FusedMultiplyAdd(
            Vector128.Create(ExpC6), r, Vector128.Create(ExpC5)), r, Vector128.Create(ExpC4)), r, Vector128.Create(ExpC3)), r, Vector128.Create(ExpC2));
        var p = Vector128.FusedMultiplyAdd(r2, q, r);
        var y = Vector128.FusedMultiplyAdd(scale, tail + p, scale);
        y = Vector128.ConditionalSelect(Vector128.LessThanOrEqual(x, Vector128.Create(ExpZeroBelow)), Vector128<double>.Zero, y);
        y = Vector128.ConditionalSelect(Vector128.GreaterThanOrEqual(x, Vector128.Create(ExpInfinityAbove)), Vector128.Create(double.PositiveInfinity), y);
        var nan = ~Vector128.Equals(x, x);
        y = Vector128.ConditionalSelect(nan, (x.AsUInt64() | Vector128.Create(QuietBit)).AsDouble(), y);
        var special = ~Vector128.LessThanOrEqual(Vector128.Abs(x), Vector128.Create(ExpExactLimit))
            & Vector128.GreaterThan(x, Vector128.Create(ExpZeroBelow)) & Vector128.LessThan(x, Vector128.Create(ExpInfinityAbove));
        return Vector128.EqualsAll(special.AsUInt64(), Vector128<ulong>.Zero) ? y : FallbackLanes<MathExp>(x, y, special);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static Vector256<double> Exp(Vector256<double> x)
    {
        var z = Vector256.FusedMultiplyAdd(x, Vector256.Create(ExpInvLn2N), Vector256.Create(ExpShift));
        var kd = z - Vector256.Create(ExpShift);
        var ki = z.AsUInt64();
        var r = Vector256.FusedMultiplyAdd(kd, Vector256.Create(-ExpLn2HiN), x);
        r = Vector256.FusedMultiplyAdd(kd, Vector256.Create(-ExpLn2LoN), r);
        var index = (ki & Vector256.Create((ulong)ExpTableMask)) << 1;
        var scale = (Gather(ExpTableBits, index) + (ki << 45)).AsDouble();
        var tail = Gather(ExpTableBits, index + Vector256<ulong>.One).AsDouble();
        var r2 = r * r;
        var q = Vector256.FusedMultiplyAdd(Vector256.FusedMultiplyAdd(Vector256.FusedMultiplyAdd(Vector256.FusedMultiplyAdd(
            Vector256.Create(ExpC6), r, Vector256.Create(ExpC5)), r, Vector256.Create(ExpC4)), r, Vector256.Create(ExpC3)), r, Vector256.Create(ExpC2));
        var p = Vector256.FusedMultiplyAdd(r2, q, r);
        var y = Vector256.FusedMultiplyAdd(scale, tail + p, scale);
        y = Vector256.ConditionalSelect(Vector256.LessThanOrEqual(x, Vector256.Create(ExpZeroBelow)), Vector256<double>.Zero, y);
        y = Vector256.ConditionalSelect(Vector256.GreaterThanOrEqual(x, Vector256.Create(ExpInfinityAbove)), Vector256.Create(double.PositiveInfinity), y);
        var nan = ~Vector256.Equals(x, x);
        y = Vector256.ConditionalSelect(nan, (x.AsUInt64() | Vector256.Create(QuietBit)).AsDouble(), y);
        var special = ~Vector256.LessThanOrEqual(Vector256.Abs(x), Vector256.Create(ExpExactLimit))
            & Vector256.GreaterThan(x, Vector256.Create(ExpZeroBelow)) & Vector256.LessThan(x, Vector256.Create(ExpInfinityAbove));
        return Vector256.EqualsAll(special.AsUInt64(), Vector256<ulong>.Zero) ? y : FallbackLanes<MathExp>(x, y, special);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static Vector512<double> Exp(Vector512<double> x)
    {
        var z = Vector512.FusedMultiplyAdd(x, Vector512.Create(ExpInvLn2N), Vector512.Create(ExpShift));
        var kd = z - Vector512.Create(ExpShift);
        var ki = z.AsUInt64();
        var r = Vector512.FusedMultiplyAdd(kd, Vector512.Create(-ExpLn2HiN), x);
        r = Vector512.FusedMultiplyAdd(kd, Vector512.Create(-ExpLn2LoN), r);
        var index = (ki & Vector512.Create((ulong)ExpTableMask)) << 1;
        var scale = (Gather(ExpTableBits, index) + (ki << 45)).AsDouble();
        var tail = Gather(ExpTableBits, index + Vector512<ulong>.One).AsDouble();
        var r2 = r * r;
        var q = Vector512.FusedMultiplyAdd(Vector512.FusedMultiplyAdd(Vector512.FusedMultiplyAdd(Vector512.FusedMultiplyAdd(
            Vector512.Create(ExpC6), r, Vector512.Create(ExpC5)), r, Vector512.Create(ExpC4)), r, Vector512.Create(ExpC3)), r, Vector512.Create(ExpC2));
        var p = Vector512.FusedMultiplyAdd(r2, q, r);
        var y = Vector512.FusedMultiplyAdd(scale, tail + p, scale);
        y = Vector512.ConditionalSelect(Vector512.LessThanOrEqual(x, Vector512.Create(ExpZeroBelow)), Vector512<double>.Zero, y);
        y = Vector512.ConditionalSelect(Vector512.GreaterThanOrEqual(x, Vector512.Create(ExpInfinityAbove)), Vector512.Create(double.PositiveInfinity), y);
        var nan = ~Vector512.Equals(x, x);
        y = Vector512.ConditionalSelect(nan, (x.AsUInt64() | Vector512.Create(QuietBit)).AsDouble(), y);
        var special = ~Vector512.LessThanOrEqual(Vector512.Abs(x), Vector512.Create(ExpExactLimit))
            & Vector512.GreaterThan(x, Vector512.Create(ExpZeroBelow)) & Vector512.LessThan(x, Vector512.Create(ExpInfinityAbove));
        return Vector512.EqualsAll(special.AsUInt64(), Vector512<ulong>.Zero) ? y : FallbackLanes<MathExp>(x, y, special);
    }

    // ---------- 対数関数 ----------

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static double Log(double x)
    {
        ulong ix = BitConverter.DoubleToUInt64Bits(x);
        if (x == 0)
        {
            return double.NegativeInfinity;
        }

        if (x == double.PositiveInfinity)
        {
            return double.PositiveInfinity;
        }

        if (x < 0)
        {
            return double.NaN;
        }

        if (double.IsNaN(x))
        {
            return Quiet(x);
        }

        if (ix - MinNormalBits >= PositiveNormalRange)
        {
            return Math.Log(x);
        }

        ulong tmp = ix - LogOffset;
        int i = (int)((tmp >> 45) & LogTableMask);
        long k = (long)tmp >> 52;
        double z = BitConverter.UInt64BitsToDouble(ix - (tmp & (0xFFFUL << 52)));
        double invc = BitConverter.UInt64BitsToDouble(Unsafe.Add(ref MemoryMarshal.GetReference(LogInverseCenterBits), i));
        double chi = BitConverter.UInt64BitsToDouble(Unsafe.Add(ref MemoryMarshal.GetReference(LogCenterHighBits), i));
        double clo = BitConverter.UInt64BitsToDouble(Unsafe.Add(ref MemoryMarshal.GetReference(LogCenterLowBits), i));
        double kd = BitConverter.Int64BitsToDouble(k + DoubleShiftBits) - ExpShift;

        // z/c - 1 = (p - 1) + pLo を丸めなしで持つ（p - 1 は Sterbenz の補題で厳密、pLo は積の丸め誤差）
        double p = z * invc;
        double pLo = Math.FusedMultiplyAdd(z, invc, -p);
        double r = p - 1;
        // w = k·Ln2Hi + ln c の上位は丸めなし。hi + lo = w + r は丸めなしの分解（|w| ≥ |r| か w = 0）
        double w = Math.FusedMultiplyAdd(kd, Ln2Hi, chi);
        double hi = w + r;
        double lo = w - hi + r;
        double r2 = r * r;
        double q = Math.FusedMultiplyAdd(Math.FusedMultiplyAdd(Math.FusedMultiplyAdd(Math.FusedMultiplyAdd(Math.FusedMultiplyAdd(Math.FusedMultiplyAdd(
            Math.FusedMultiplyAdd(LogA9, r, LogA8), r, LogA7), r, LogA6), r, LogA5), r, LogA4), r, LogA3), r, LogA2);
        // ln(1 + r + pLo) ≈ ln(1 + r) + pLo·(1 - r)。|pLo| ≤ 2^-53 なので、pLo·r を落とすと最大 0.25 ulp の誤差になる
        double low = lo + (Math.FusedMultiplyAdd(-pLo, r, pLo) + Math.FusedMultiplyAdd(kd, Ln2Lo, clo));
        return hi + Math.FusedMultiplyAdd(r2, q, low);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static Vector128<double> Log(Vector128<double> x)
    {
        var ix = x.AsUInt64();
        var tmp = ix - Vector128.Create(LogOffset);
        var i = Vector128.ShiftRightLogical(tmp, 45) & Vector128.Create(LogTableMask);
        var k = Vector128.ShiftRightArithmetic(tmp.AsInt64(), 52);
        var z = (ix - (tmp & Vector128.Create(0xFFFUL << 52))).AsDouble();
        var invc = Gather(LogInverseCenterBits, i).AsDouble();
        var chi = Gather(LogCenterHighBits, i).AsDouble();
        var clo = Gather(LogCenterLowBits, i).AsDouble();
        var kd = (k + Vector128.Create(DoubleShiftBits)).AsDouble() - Vector128.Create(ExpShift);
        var p = z * invc;
        var pLo = Vector128.FusedMultiplyAdd(z, invc, -p);
        var r = p - Vector128<double>.One;
        var w = Vector128.FusedMultiplyAdd(kd, Vector128.Create(Ln2Hi), chi);
        var hi = w + r;
        var lo = w - hi + r;
        var r2 = r * r;
        var q = Vector128.FusedMultiplyAdd(Vector128.FusedMultiplyAdd(Vector128.FusedMultiplyAdd(Vector128.FusedMultiplyAdd(
            Vector128.FusedMultiplyAdd(Vector128.FusedMultiplyAdd(Vector128.FusedMultiplyAdd(
            Vector128.Create(LogA9), r, Vector128.Create(LogA8)), r, Vector128.Create(LogA7)), r, Vector128.Create(LogA6)), r, Vector128.Create(LogA5)),
            r, Vector128.Create(LogA4)), r, Vector128.Create(LogA3)), r, Vector128.Create(LogA2));
        var low = lo + (Vector128.FusedMultiplyAdd(-pLo, r, pLo) + Vector128.FusedMultiplyAdd(kd, Vector128.Create(Ln2Lo), clo));
        var y = hi + Vector128.FusedMultiplyAdd(r2, q, low);
        y = Vector128.ConditionalSelect(Vector128.Equals(x, Vector128<double>.Zero), Vector128.Create(double.NegativeInfinity), y);
        var infinite = Vector128.Equals(x, Vector128.Create(double.PositiveInfinity));
        y = Vector128.ConditionalSelect(infinite, x, y);
        var negative = Vector128.LessThan(x, Vector128<double>.Zero);
        var nan = ~Vector128.Equals(x, x);
        y = Vector128.ConditionalSelect(negative, Vector128.Create(double.NaN), y);
        y = Vector128.ConditionalSelect(nan, (ix | Vector128.Create(QuietBit)).AsDouble(), y);
        var special = Vector128.GreaterThanOrEqual(ix - Vector128.Create(MinNormalBits), Vector128.Create(PositiveNormalRange)).AsDouble()
            & ~(Vector128.Equals(x, Vector128<double>.Zero) | infinite | negative | nan);
        return Vector128.EqualsAll(special.AsUInt64(), Vector128<ulong>.Zero) ? y : FallbackLanes<MathLog>(x, y, special);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static Vector256<double> Log(Vector256<double> x)
    {
        var ix = x.AsUInt64();
        var tmp = ix - Vector256.Create(LogOffset);
        var i = Vector256.ShiftRightLogical(tmp, 45) & Vector256.Create(LogTableMask);
        var k = Vector256.ShiftRightArithmetic(tmp.AsInt64(), 52);
        var z = (ix - (tmp & Vector256.Create(0xFFFUL << 52))).AsDouble();
        var invc = Gather(LogInverseCenterBits, i).AsDouble();
        var chi = Gather(LogCenterHighBits, i).AsDouble();
        var clo = Gather(LogCenterLowBits, i).AsDouble();
        var kd = (k + Vector256.Create(DoubleShiftBits)).AsDouble() - Vector256.Create(ExpShift);
        var p = z * invc;
        var pLo = Vector256.FusedMultiplyAdd(z, invc, -p);
        var r = p - Vector256<double>.One;
        var w = Vector256.FusedMultiplyAdd(kd, Vector256.Create(Ln2Hi), chi);
        var hi = w + r;
        var lo = w - hi + r;
        var r2 = r * r;
        var q = Vector256.FusedMultiplyAdd(Vector256.FusedMultiplyAdd(Vector256.FusedMultiplyAdd(Vector256.FusedMultiplyAdd(
            Vector256.FusedMultiplyAdd(Vector256.FusedMultiplyAdd(Vector256.FusedMultiplyAdd(
            Vector256.Create(LogA9), r, Vector256.Create(LogA8)), r, Vector256.Create(LogA7)), r, Vector256.Create(LogA6)), r, Vector256.Create(LogA5)),
            r, Vector256.Create(LogA4)), r, Vector256.Create(LogA3)), r, Vector256.Create(LogA2));
        var low = lo + (Vector256.FusedMultiplyAdd(-pLo, r, pLo) + Vector256.FusedMultiplyAdd(kd, Vector256.Create(Ln2Lo), clo));
        var y = hi + Vector256.FusedMultiplyAdd(r2, q, low);
        y = Vector256.ConditionalSelect(Vector256.Equals(x, Vector256<double>.Zero), Vector256.Create(double.NegativeInfinity), y);
        var infinite = Vector256.Equals(x, Vector256.Create(double.PositiveInfinity));
        y = Vector256.ConditionalSelect(infinite, x, y);
        var negative = Vector256.LessThan(x, Vector256<double>.Zero);
        var nan = ~Vector256.Equals(x, x);
        y = Vector256.ConditionalSelect(negative, Vector256.Create(double.NaN), y);
        y = Vector256.ConditionalSelect(nan, (ix | Vector256.Create(QuietBit)).AsDouble(), y);
        var special = Vector256.GreaterThanOrEqual(ix - Vector256.Create(MinNormalBits), Vector256.Create(PositiveNormalRange)).AsDouble()
            & ~(Vector256.Equals(x, Vector256<double>.Zero) | infinite | negative | nan);
        return Vector256.EqualsAll(special.AsUInt64(), Vector256<ulong>.Zero) ? y : FallbackLanes<MathLog>(x, y, special);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static Vector512<double> Log(Vector512<double> x)
    {
        var ix = x.AsUInt64();
        var tmp = ix - Vector512.Create(LogOffset);
        var i = Vector512.ShiftRightLogical(tmp, 45) & Vector512.Create(LogTableMask);
        var k = Vector512.ShiftRightArithmetic(tmp.AsInt64(), 52);
        var z = (ix - (tmp & Vector512.Create(0xFFFUL << 52))).AsDouble();
        var invc = Gather(LogInverseCenterBits, i).AsDouble();
        var chi = Gather(LogCenterHighBits, i).AsDouble();
        var clo = Gather(LogCenterLowBits, i).AsDouble();
        var kd = (k + Vector512.Create(DoubleShiftBits)).AsDouble() - Vector512.Create(ExpShift);
        var p = z * invc;
        var pLo = Vector512.FusedMultiplyAdd(z, invc, -p);
        var r = p - Vector512<double>.One;
        var w = Vector512.FusedMultiplyAdd(kd, Vector512.Create(Ln2Hi), chi);
        var hi = w + r;
        var lo = w - hi + r;
        var r2 = r * r;
        var q = Vector512.FusedMultiplyAdd(Vector512.FusedMultiplyAdd(Vector512.FusedMultiplyAdd(Vector512.FusedMultiplyAdd(
            Vector512.FusedMultiplyAdd(Vector512.FusedMultiplyAdd(Vector512.FusedMultiplyAdd(
            Vector512.Create(LogA9), r, Vector512.Create(LogA8)), r, Vector512.Create(LogA7)), r, Vector512.Create(LogA6)), r, Vector512.Create(LogA5)),
            r, Vector512.Create(LogA4)), r, Vector512.Create(LogA3)), r, Vector512.Create(LogA2));
        var low = lo + (Vector512.FusedMultiplyAdd(-pLo, r, pLo) + Vector512.FusedMultiplyAdd(kd, Vector512.Create(Ln2Lo), clo));
        var y = hi + Vector512.FusedMultiplyAdd(r2, q, low);
        y = Vector512.ConditionalSelect(Vector512.Equals(x, Vector512<double>.Zero), Vector512.Create(double.NegativeInfinity), y);
        var infinite = Vector512.Equals(x, Vector512.Create(double.PositiveInfinity));
        y = Vector512.ConditionalSelect(infinite, x, y);
        var negative = Vector512.LessThan(x, Vector512<double>.Zero);
        var nan = ~Vector512.Equals(x, x);
        y = Vector512.ConditionalSelect(negative, Vector512.Create(double.NaN), y);
        y = Vector512.ConditionalSelect(nan, (ix | Vector512.Create(QuietBit)).AsDouble(), y);
        var special = Vector512.GreaterThanOrEqual(ix - Vector512.Create(MinNormalBits), Vector512.Create(PositiveNormalRange)).AsDouble()
            & ~(Vector512.Equals(x, Vector512<double>.Zero) | infinite | negative | nan);
        return Vector512.EqualsAll(special.AsUInt64(), Vector512<ulong>.Zero) ? y : FallbackLanes<MathLog>(x, y, special);
    }

    // ---------- 特殊入力 ----------

    private static double Quiet(double x) => BitConverter.UInt64BitsToDouble(BitConverter.DoubleToUInt64Bits(x) | QuietBit);

    private interface IMathFunction
    {
        static abstract double Invoke(double x);
    }

    private readonly struct MathExp : IMathFunction
    {
        public static double Invoke(double x) => Math.Exp(x);
    }

    private readonly struct MathLog : IMathFunction
    {
        public static double Invoke(double x) => Math.Log(x);
    }

    // テーブル式が使えない有限入力（非正規化領域・オーバーフロー境界付近）のレーンのみ Math.Exp / Math.Log へフォールバックする。
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static Vector128<double> FallbackLanes<TMath>(Vector128<double> x, Vector128<double> y, Vector128<double> special) where TMath : IMathFunction
    {
        Span<double> result = stackalloc double[Vector128<double>.Count];
        y.CopyTo(result);
        for (uint mask = special.ExtractMostSignificantBits(); mask != 0; mask &= mask - 1)
        {
            int i = BitOperations.TrailingZeroCount(mask);
            result[i] = TMath.Invoke(x.GetElement(i));
        }

        return Vector128.Create<double>(result);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static Vector256<double> FallbackLanes<TMath>(Vector256<double> x, Vector256<double> y, Vector256<double> special) where TMath : IMathFunction
    {
        Span<double> result = stackalloc double[Vector256<double>.Count];
        y.CopyTo(result);
        for (uint mask = special.ExtractMostSignificantBits(); mask != 0; mask &= mask - 1)
        {
            int i = BitOperations.TrailingZeroCount(mask);
            result[i] = TMath.Invoke(x.GetElement(i));
        }

        return Vector256.Create<double>(result);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static Vector512<double> FallbackLanes<TMath>(Vector512<double> x, Vector512<double> y, Vector512<double> special) where TMath : IMathFunction
    {
        Span<double> result = stackalloc double[Vector512<double>.Count];
        y.CopyTo(result);
        for (ulong mask = special.ExtractMostSignificantBits(); mask != 0; mask &= mask - 1)
        {
            int i = BitOperations.TrailingZeroCount(mask);
            result[i] = TMath.Invoke(x.GetElement(i));
        }

        return Vector512.Create<double>(result);
    }

    // ---------- テーブル参照 ----------

    // インデックスは呼び出し側でマスク済み。AVX2 gather 非対応時はスカラー読み込みにフォールバックする。
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector128<ulong> Gather(ReadOnlySpan<ulong> table, Vector128<ulong> index)
    {
        if (Avx2.IsSupported)
        {
            fixed (ulong* p = table)
            {
                return Avx2.GatherVector128(p, index.AsInt64(), 8);
            }
        }

        ref ulong t = ref MemoryMarshal.GetReference(table);
        return Vector128.Create(Unsafe.Add(ref t, (nint)index.GetElement(0)), Unsafe.Add(ref t, (nint)index.GetElement(1)));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<ulong> Gather(ReadOnlySpan<ulong> table, Vector256<ulong> index)
    {
        if (Avx2.IsSupported)
        {
            fixed (ulong* p = table)
            {
                return Avx2.GatherVector256(p, index.AsInt64(), 8);
            }
        }

        return Vector256.Create(Gather(table, index.GetLower()), Gather(table, index.GetUpper()));
    }

    // .NET 10 では AVX-512 gather が未提供のため AVX2 gather を 2 回に分けて実行する
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<ulong> Gather(ReadOnlySpan<ulong> table, Vector512<ulong> index) =>
        Vector512.Create(Gather(table, index.GetLower()), Gather(table, index.GetUpper()));
}
