using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Tuna.LinearAlgebra;

// float 向けの高速近似要素演算（テーブル引きを行わず float のまま多項式で計算、誤差は数 ulp）。
// ISingleLanes 上の共通式により、スカラー・各 SIMD 幅・レイアウトによらずビット単位で一致する。
internal static partial class DenseElementwise
{
    // e^x = 2^n·e^r。n = round(x/ln 2)、r = x - n·ln 2（|r| ≤ ln2/2 より少し大きい程度）。
    // e^r ≈ 1 + r + r²·(1/2 + r/6 + r²/24 + r³/120 + r⁴/720 + r⁵/5040)（テイラー展開の7次。打ち切り誤差は相対 5·10⁻⁹ 未満）
    private const float SingleExpShift = 12582912f;                 // 0x1.8p23。足すと最近接の整数に丸まり、下位のビットに整数が入る
    private const float SingleInvLn2 = 1.44269504088896341f;
    private const float SingleLn2Hi = 0.693145751953125f;           // ln 2 の上位。仮数の下位 9 ビットが 0 なので、|n| < 2^9 で n·SingleLn2Hi は丸めなし
    private const float SingleLn2Lo = 1.42860682030941723e-6f;      // ln 2 - SingleLn2Hi
    private const float SingleExpHigh = 88.8f;                      // e^x は x > 88.723 でオーバーフローする。それより大きい x はここに切り詰めても ∞
    private const float SingleExpLow = -150f;                       // e^x は x < -103.98 で 0 に丸まる。-150 で切り詰めれば 2^n の指数が正規化数に収まる
    private const float SingleExpC2 = 0.5f, SingleExpC3 = 1f / 6, SingleExpC4 = 1f / 24, SingleExpC5 = 1f / 120, SingleExpC6 = 1f / 720, SingleExpC7 = 1f / 5040;

    // ln(1 + u) = 2·atanh(s)、s = u/(2 + u)。0 ≤ u ≤ 1 で s ≤ 1/3 なので、2s·(1 + s²/3 + s⁴/5 + … + s¹⁴/15) の打ち切り誤差は相対 1.4·10⁻⁹ 未満
    private const float SingleLog1pA3 = 1f / 3, SingleLog1pA5 = 1f / 5, SingleLog1pA7 = 1f / 7, SingleLog1pA9 = 1f / 9,
        SingleLog1pA11 = 1f / 11, SingleLog1pA13 = 1f / 13, SingleLog1pA15 = 1f / 15;

    // ---------- スカラー実装（ベクトル実装の各レーンとビット単位で一致） ----------
    // インライン展開で JIT 展開上限を超えないよう NoInlining とし、代わりに AggressiveOptimization で即時最適化する。
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    internal static float ApproximateExp(float x) => InvokeSingle<ApproximateExpFunction>(x);

    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    internal static float ApproximateLogistic(float x) => InvokeSingle<ApproximateLogisticFunction>(x);

    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    internal static float ApproximateSilu(float x) => InvokeSingle<ApproximateSiluFunction>(x);

    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    internal static float ApproximateSoftplus(float x) => InvokeSingle<ApproximateSoftplusFunction>(x);

    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    internal static float ApproximateGelu(float x) => InvokeSingle<ApproximateGeluFunction>(x);

    // SIMD 対応環境では Vector128 の先頭レーンで計算し、汎用レジスタと SIMD レジスタ間の往復を避ける。
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float InvokeSingle<TFunction>(float x) where TFunction : ISingleLanesFunction =>
        Vector128.IsHardwareAccelerated
            ? TFunction.Invoke(new SingleLanes128(Vector128.CreateScalar(x))).Value.ToScalar()
            : TFunction.Invoke(new SingleScalarLanes(x)).Value;

    // float は近似カーネル、double は高精度カーネルで計算する。
    private static void MapApproximate<T, TSingle, TExact>(ref T x, int xStride, ref T destination, int destinationStride, int length, int count)
        where T : unmanaged
        where TSingle : ISingleLanesFunction
        where TExact : struct, IExactUnaryKernel<double>
    {
        if (typeof(T) == typeof(float))
        {
            MapLines(ref Unsafe.As<T, float>(ref x), xStride, ref Unsafe.As<T, float>(ref destination), destinationStride, length, count,
                default(SingleLanesKernel<TSingle>));
        }
        else
        {
            MapWidened(ref x, xStride, ref destination, destinationStride, length, count, default(TExact));
        }
    }

    // ---------- float スカラーおよび各 SIMD 幅の共通演算 ----------
    // 比較は全ビット 1（真）または 0（偽）のマスクを返す。
    private interface ISingleLanes<TSelf> where TSelf : struct, ISingleLanes<TSelf>
    {
        static abstract TSelf Create(float value);
        static abstract TSelf operator +(TSelf x, TSelf y);
        static abstract TSelf operator -(TSelf x, TSelf y);
        static abstract TSelf operator *(TSelf x, TSelf y);
        static abstract TSelf operator /(TSelf x, TSelf y);
        static abstract TSelf operator -(TSelf x);
        static abstract TSelf Abs(TSelf x);
        static abstract TSelf FusedMultiplyAdd(TSelf x, TSelf y, TSelf z);

        // NaN 以外の x を [low, high] にクランプする（NaN の振る舞いは命令依存のため呼び出し側で復元する）。
        static abstract TSelf ClampNumber(TSelf x, TSelf low, TSelf high);

        // t = 0x1.8p23 + n（整数 -217 <= n <= 128）から p * 2^n を計算する（2 のべき乗を 2 回に分けて乗算）。
        static abstract TSelf ScaleByShiftedExponent(TSelf p, TSelf t);

        static abstract TSelf Equal(TSelf x, TSelf y);
        static abstract TSelf LessThan(TSelf x, TSelf y);
        static abstract TSelf GreaterThan(TSelf x, TSelf y);
        static abstract TSelf GreaterThanOrEqual(TSelf x, TSelf y);
        static abstract TSelf Select(TSelf mask, TSelf ifTrue, TSelf ifFalse);
    }

    // SIMD 非対応環境向けスカラー実装（分岐予測ミスを避けるためビット演算で選択する）。
    private readonly struct SingleScalarLanes(float value) : ISingleLanes<SingleScalarLanes>
    {
        public float Value => value;
        public static SingleScalarLanes Create(float value) => new(value);
        public static SingleScalarLanes operator +(SingleScalarLanes x, SingleScalarLanes y) => new(x.Value + y.Value);
        public static SingleScalarLanes operator -(SingleScalarLanes x, SingleScalarLanes y) => new(x.Value - y.Value);
        public static SingleScalarLanes operator *(SingleScalarLanes x, SingleScalarLanes y) => new(x.Value * y.Value);
        public static SingleScalarLanes operator /(SingleScalarLanes x, SingleScalarLanes y) => new(x.Value / y.Value);
        public static SingleScalarLanes operator -(SingleScalarLanes x) => new(-x.Value);
        public static SingleScalarLanes Abs(SingleScalarLanes x) => new(Math.Abs(x.Value));
        public static SingleScalarLanes FusedMultiplyAdd(SingleScalarLanes x, SingleScalarLanes y, SingleScalarLanes z) =>
            new(MathF.FusedMultiplyAdd(x.Value, y.Value, z.Value));
        public static SingleScalarLanes ClampNumber(SingleScalarLanes x, SingleScalarLanes low, SingleScalarLanes high) =>
            new(Math.Min(Math.Max(x.Value, low.Value), high.Value));

        public static SingleScalarLanes ScaleByShiftedExponent(SingleScalarLanes p, SingleScalarLanes t)
        {
            int n = unchecked(BitConverter.SingleToInt32Bits(t.Value) - BitConverter.SingleToInt32Bits(SingleExpShift));
            int n1 = n >> 1;
            int n2 = unchecked(n - n1);
            float s1 = BitConverter.Int32BitsToSingle(unchecked((n1 + 127) << 23));
            float s2 = BitConverter.Int32BitsToSingle(unchecked((n2 + 127) << 23));
            return new(p.Value * s1 * s2);
        }

        public static SingleScalarLanes Equal(SingleScalarLanes x, SingleScalarLanes y) => Mask(x.Value == y.Value);
        public static SingleScalarLanes LessThan(SingleScalarLanes x, SingleScalarLanes y) => Mask(x.Value < y.Value);
        public static SingleScalarLanes GreaterThan(SingleScalarLanes x, SingleScalarLanes y) => Mask(x.Value > y.Value);
        public static SingleScalarLanes GreaterThanOrEqual(SingleScalarLanes x, SingleScalarLanes y) => Mask(x.Value >= y.Value);
        public static SingleScalarLanes Select(SingleScalarLanes mask, SingleScalarLanes ifTrue, SingleScalarLanes ifFalse)
        {
            int m = BitConverter.SingleToInt32Bits(mask.Value);
            return new(BitConverter.Int32BitsToSingle((m & BitConverter.SingleToInt32Bits(ifTrue.Value)) | (~m & BitConverter.SingleToInt32Bits(ifFalse.Value))));
        }

        private static SingleScalarLanes Mask(bool condition) => new(BitConverter.Int32BitsToSingle(-(condition ? 1 : 0)));
    }

    private readonly struct SingleLanes128(Vector128<float> value) : ISingleLanes<SingleLanes128>
    {
        public Vector128<float> Value => value;
        public static SingleLanes128 Create(float value) => new(Vector128.Create(value));
        public static SingleLanes128 operator +(SingleLanes128 x, SingleLanes128 y) => new(x.Value + y.Value);
        public static SingleLanes128 operator -(SingleLanes128 x, SingleLanes128 y) => new(x.Value - y.Value);
        public static SingleLanes128 operator *(SingleLanes128 x, SingleLanes128 y) => new(x.Value * y.Value);
        public static SingleLanes128 operator /(SingleLanes128 x, SingleLanes128 y) => new(x.Value / y.Value);
        public static SingleLanes128 operator -(SingleLanes128 x) => new(-x.Value);
        public static SingleLanes128 Abs(SingleLanes128 x) => new(Vector128.Abs(x.Value));
        public static SingleLanes128 FusedMultiplyAdd(SingleLanes128 x, SingleLanes128 y, SingleLanes128 z) =>
            new(Vector128.FusedMultiplyAdd(x.Value, y.Value, z.Value));
        public static SingleLanes128 ClampNumber(SingleLanes128 x, SingleLanes128 low, SingleLanes128 high) =>
            new(Vector128.MinNative(Vector128.MaxNative(x.Value, low.Value), high.Value));

        public static SingleLanes128 ScaleByShiftedExponent(SingleLanes128 p, SingleLanes128 t)
        {
            var n = t.Value.AsInt32() - Vector128.Create(SingleExpShift).AsInt32();
            var n1 = Vector128.ShiftRightArithmetic(n, 1);
            var n2 = n - n1;
            var s1 = Vector128.ShiftLeft(n1 + Vector128.Create(127), 23).AsSingle();
            var s2 = Vector128.ShiftLeft(n2 + Vector128.Create(127), 23).AsSingle();
            return new(p.Value * s1 * s2);
        }

        public static SingleLanes128 Equal(SingleLanes128 x, SingleLanes128 y) => new(Vector128.Equals(x.Value, y.Value));
        public static SingleLanes128 LessThan(SingleLanes128 x, SingleLanes128 y) => new(Vector128.LessThan(x.Value, y.Value));
        public static SingleLanes128 GreaterThan(SingleLanes128 x, SingleLanes128 y) => new(Vector128.GreaterThan(x.Value, y.Value));
        public static SingleLanes128 GreaterThanOrEqual(SingleLanes128 x, SingleLanes128 y) => new(Vector128.GreaterThanOrEqual(x.Value, y.Value));
        public static SingleLanes128 Select(SingleLanes128 mask, SingleLanes128 ifTrue, SingleLanes128 ifFalse) =>
            new(Vector128.ConditionalSelect(mask.Value, ifTrue.Value, ifFalse.Value));
    }

    private readonly struct SingleLanes256(Vector256<float> value) : ISingleLanes<SingleLanes256>
    {
        public Vector256<float> Value => value;
        public static SingleLanes256 Create(float value) => new(Vector256.Create(value));
        public static SingleLanes256 operator +(SingleLanes256 x, SingleLanes256 y) => new(x.Value + y.Value);
        public static SingleLanes256 operator -(SingleLanes256 x, SingleLanes256 y) => new(x.Value - y.Value);
        public static SingleLanes256 operator *(SingleLanes256 x, SingleLanes256 y) => new(x.Value * y.Value);
        public static SingleLanes256 operator /(SingleLanes256 x, SingleLanes256 y) => new(x.Value / y.Value);
        public static SingleLanes256 operator -(SingleLanes256 x) => new(-x.Value);
        public static SingleLanes256 Abs(SingleLanes256 x) => new(Vector256.Abs(x.Value));
        public static SingleLanes256 FusedMultiplyAdd(SingleLanes256 x, SingleLanes256 y, SingleLanes256 z) =>
            new(Vector256.FusedMultiplyAdd(x.Value, y.Value, z.Value));
        public static SingleLanes256 ClampNumber(SingleLanes256 x, SingleLanes256 low, SingleLanes256 high) =>
            new(Vector256.MinNative(Vector256.MaxNative(x.Value, low.Value), high.Value));

        public static SingleLanes256 ScaleByShiftedExponent(SingleLanes256 p, SingleLanes256 t)
        {
            var n = t.Value.AsInt32() - Vector256.Create(SingleExpShift).AsInt32();
            var n1 = Vector256.ShiftRightArithmetic(n, 1);
            var n2 = n - n1;
            var s1 = Vector256.ShiftLeft(n1 + Vector256.Create(127), 23).AsSingle();
            var s2 = Vector256.ShiftLeft(n2 + Vector256.Create(127), 23).AsSingle();
            return new(p.Value * s1 * s2);
        }

        public static SingleLanes256 Equal(SingleLanes256 x, SingleLanes256 y) => new(Vector256.Equals(x.Value, y.Value));
        public static SingleLanes256 LessThan(SingleLanes256 x, SingleLanes256 y) => new(Vector256.LessThan(x.Value, y.Value));
        public static SingleLanes256 GreaterThan(SingleLanes256 x, SingleLanes256 y) => new(Vector256.GreaterThan(x.Value, y.Value));
        public static SingleLanes256 GreaterThanOrEqual(SingleLanes256 x, SingleLanes256 y) => new(Vector256.GreaterThanOrEqual(x.Value, y.Value));
        public static SingleLanes256 Select(SingleLanes256 mask, SingleLanes256 ifTrue, SingleLanes256 ifFalse) =>
            new(Vector256.ConditionalSelect(mask.Value, ifTrue.Value, ifFalse.Value));
    }

    private readonly struct SingleLanes512(Vector512<float> value) : ISingleLanes<SingleLanes512>
    {
        public Vector512<float> Value => value;
        public static SingleLanes512 Create(float value) => new(Vector512.Create(value));
        public static SingleLanes512 operator +(SingleLanes512 x, SingleLanes512 y) => new(x.Value + y.Value);
        public static SingleLanes512 operator -(SingleLanes512 x, SingleLanes512 y) => new(x.Value - y.Value);
        public static SingleLanes512 operator *(SingleLanes512 x, SingleLanes512 y) => new(x.Value * y.Value);
        public static SingleLanes512 operator /(SingleLanes512 x, SingleLanes512 y) => new(x.Value / y.Value);
        public static SingleLanes512 operator -(SingleLanes512 x) => new(-x.Value);
        public static SingleLanes512 Abs(SingleLanes512 x) => new(Vector512.Abs(x.Value));
        public static SingleLanes512 FusedMultiplyAdd(SingleLanes512 x, SingleLanes512 y, SingleLanes512 z) =>
            new(Vector512.FusedMultiplyAdd(x.Value, y.Value, z.Value));
        public static SingleLanes512 ClampNumber(SingleLanes512 x, SingleLanes512 low, SingleLanes512 high) =>
            new(Vector512.MinNative(Vector512.MaxNative(x.Value, low.Value), high.Value));

        public static SingleLanes512 ScaleByShiftedExponent(SingleLanes512 p, SingleLanes512 t)
        {
            var n = t.Value.AsInt32() - Vector512.Create(SingleExpShift).AsInt32();
            var n1 = Vector512.ShiftRightArithmetic(n, 1);
            var n2 = n - n1;
            var s1 = Vector512.ShiftLeft(n1 + Vector512.Create(127), 23).AsSingle();
            var s2 = Vector512.ShiftLeft(n2 + Vector512.Create(127), 23).AsSingle();
            return new(p.Value * s1 * s2);
        }

        public static SingleLanes512 Equal(SingleLanes512 x, SingleLanes512 y) => new(Vector512.Equals(x.Value, y.Value));
        public static SingleLanes512 LessThan(SingleLanes512 x, SingleLanes512 y) => new(Vector512.LessThan(x.Value, y.Value));
        public static SingleLanes512 GreaterThan(SingleLanes512 x, SingleLanes512 y) => new(Vector512.GreaterThan(x.Value, y.Value));
        public static SingleLanes512 GreaterThanOrEqual(SingleLanes512 x, SingleLanes512 y) => new(Vector512.GreaterThanOrEqual(x.Value, y.Value));
        public static SingleLanes512 Select(SingleLanes512 mask, SingleLanes512 ifTrue, SingleLanes512 ifFalse) =>
            new(Vector512.ConditionalSelect(mask.Value, ifTrue.Value, ifFalse.Value));
    }

    private interface ISingleLanesFunction
    {
        static abstract TLanes Invoke<TLanes>(TLanes x) where TLanes : struct, ISingleLanes<TLanes>;
    }

    private readonly struct SingleLanesKernel<TFunction> : IExactUnaryKernel<float> where TFunction : ISingleLanesFunction
    {
        [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
        public float Invoke(float x) => InvokeSingle<TFunction>(x);
        public Vector128<float> Invoke(Vector128<float> x) => TFunction.Invoke(new SingleLanes128(x)).Value;
        public Vector256<float> Invoke(Vector256<float> x) => TFunction.Invoke(new SingleLanes256(x)).Value;
        public Vector512<float> Invoke(Vector512<float> x) => TFunction.Invoke(new SingleLanes512(x)).Value;
    }

    // ---------- 近似関数 ----------

    // e^x = p * 2^n の p ≈ e^r と t = 0x1.8p23 + n を求める（x は [-150, 88.8] にクランプ）。
    // ScaleByShiftedExponent(m * p, t) とすれば e^x が非正規化数となる領域でも m * e^x の精度を保てる。
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (TLanes P, TLanes T) ApproximateExpParts<TLanes>(TLanes x) where TLanes : struct, ISingleLanes<TLanes>
    {
        var clamped = TLanes.ClampNumber(x, TLanes.Create(SingleExpLow), TLanes.Create(SingleExpHigh));
        var shift = TLanes.Create(SingleExpShift);
        var t = TLanes.FusedMultiplyAdd(clamped, TLanes.Create(SingleInvLn2), shift);
        var n = t - shift;
        var r = TLanes.FusedMultiplyAdd(n, TLanes.Create(-SingleLn2Hi), clamped);
        r = TLanes.FusedMultiplyAdd(n, TLanes.Create(-SingleLn2Lo), r);
        var q = TLanes.FusedMultiplyAdd(TLanes.Create(SingleExpC7), r, TLanes.Create(SingleExpC6));
        q = TLanes.FusedMultiplyAdd(q, r, TLanes.Create(SingleExpC5));
        q = TLanes.FusedMultiplyAdd(q, r, TLanes.Create(SingleExpC4));
        q = TLanes.FusedMultiplyAdd(q, r, TLanes.Create(SingleExpC3));
        q = TLanes.FusedMultiplyAdd(q, r, TLanes.Create(SingleExpC2));
        var p = TLanes.FusedMultiplyAdd(q * r, r, r) + TLanes.Create(1);
        return (p, t);
    }

    // NaN レーンは元の入力ビット（符号・ペイロード）をそのまま復元する。
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TLanes KeepNaN<TLanes>(TLanes x, TLanes y) where TLanes : struct, ISingleLanes<TLanes> =>
        TLanes.Select(TLanes.Equal(x, x), y, x);

    // e^x（x >= 88.8 は ∞、x <= -150 は 0）。
    private readonly struct ApproximateExpFunction : ISingleLanesFunction
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TLanes Invoke<TLanes>(TLanes x) where TLanes : struct, ISingleLanes<TLanes>
        {
            var (p, t) = ApproximateExpParts(x);
            return KeepNaN(x, TLanes.ScaleByShiftedExponent(p, t));
        }
    }

    // m * σ(z): e = e^{-|z|}, d = 1 + e として z >= 0 なら m/d、それ以外は m*e/d（z < -150 は m*0）。
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TLanes ScaledLogistic<TLanes>(TLanes z, TLanes m) where TLanes : struct, ISingleLanes<TLanes>
    {
        var negative = -TLanes.Abs(z);
        var (p, t) = ApproximateExpParts(negative);
        var e = TLanes.ScaleByShiftedExponent(p, t);
        var me = TLanes.Select(TLanes.LessThan(negative, TLanes.Create(SingleExpLow)), m * TLanes.Create(0), TLanes.ScaleByShiftedExponent(m * p, t));
        var numerator = TLanes.Select(TLanes.GreaterThanOrEqual(z, TLanes.Create(0)), m, me);
        return numerator / (TLanes.Create(1) + e);
    }

    // σ(x) = 1 / (1 + e^-x)
    private readonly struct ApproximateLogisticFunction : ISingleLanesFunction
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TLanes Invoke<TLanes>(TLanes x) where TLanes : struct, ISingleLanes<TLanes> =>
            KeepNaN(x, ScaledLogistic(x, TLanes.Create(1)));
    }

    // x * σ(x)（x = -∞ は -0）。
    private readonly struct ApproximateSiluFunction : ISingleLanesFunction
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TLanes Invoke<TLanes>(TLanes x) where TLanes : struct, ISingleLanes<TLanes>
        {
            var y = TLanes.Select(TLanes.Equal(x, TLanes.Create(float.NegativeInfinity)), TLanes.Create(-0f), ScaledLogistic(x, x));
            return KeepNaN(x, y);
        }
    }

    // ln(1 + u) (0 <= u <= 1): 2s = 2u/(2 + u), w = s^2 として 2s + 2s*w*(1/3 + ... + w^6/15)。
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TLanes ApproximateLog1pUnit<TLanes>(TLanes u) where TLanes : struct, ISingleLanes<TLanes>
    {
        var s2 = (u + u) / (TLanes.Create(2) + u);
        var w = s2 * s2 * TLanes.Create(0.25f);
        var q = TLanes.FusedMultiplyAdd(TLanes.Create(SingleLog1pA15), w, TLanes.Create(SingleLog1pA13));
        q = TLanes.FusedMultiplyAdd(q, w, TLanes.Create(SingleLog1pA11));
        q = TLanes.FusedMultiplyAdd(q, w, TLanes.Create(SingleLog1pA9));
        q = TLanes.FusedMultiplyAdd(q, w, TLanes.Create(SingleLog1pA7));
        q = TLanes.FusedMultiplyAdd(q, w, TLanes.Create(SingleLog1pA5));
        q = TLanes.FusedMultiplyAdd(q, w, TLanes.Create(SingleLog1pA3));
        return TLanes.FusedMultiplyAdd(s2 * w, q, s2);
    }

    // ln(1 + e^x)
    private readonly struct ApproximateSoftplusFunction : ISingleLanesFunction
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TLanes Invoke<TLanes>(TLanes x) where TLanes : struct, ISingleLanes<TLanes>
        {
            var (p, t) = ApproximateExpParts(-TLanes.Abs(x));
            var log = ApproximateLog1pUnit(TLanes.ScaleByShiftedExponent(p, t));
            return KeepNaN(x, TLanes.Select(TLanes.GreaterThan(x, TLanes.Create(0)), x + log, log));
        }
    }

    // tanh 近似 GELU: x * σ(z), z = c * FMA(k * x^2, x, x)（x = -∞ は -0）。
    private readonly struct ApproximateGeluFunction : ISingleLanesFunction
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TLanes Invoke<TLanes>(TLanes x) where TLanes : struct, ISingleLanes<TLanes>
        {
            var z = TLanes.Create((float)XMath.GeluTanhScale) * TLanes.FusedMultiplyAdd(TLanes.Create((float)XMath.GeluTanhCubic) * x * x, x, x);
            var y = TLanes.Select(TLanes.Equal(x, TLanes.Create(float.NegativeInfinity)), TLanes.Create(-0f), ScaledLogistic(z, x));
            return KeepNaN(x, y);
        }
    }
}
