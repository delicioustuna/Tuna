using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Tuna.LinearAlgebra;

// 指数・対数を含む複合要素演算のディスパッチと ILanes ベースの共通実装。
// XMath / XMath.Special のスカラー定義と同じ四則演算順序を保ち、Math.Exp・Math.Log のみを自前実装に置き換える。
internal static partial class DenseElementwise
{
    // float は原則 double に拡張して計算後に丸める（Approximate* の float のみ float のまま計算する）。
    private static bool TryMapTranscendental<T, TOperator>(ref T x, int xStride, ref T destination, int destinationStride, int length, int count, TOperator op)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (typeof(TOperator) == typeof(ExpOperator<T>))
        {
            MapWidened(ref x, xStride, ref destination, destinationStride, length, count, default(LanesKernel<ExpFunction>));
        }
        else if (typeof(TOperator) == typeof(StaticUnaryOperator<T, XMath.Operators.Log<T>>))
        {
            MapWidened(ref x, xStride, ref destination, destinationStride, length, count, default(LanesKernel<LogFunction>));
        }
        else if (typeof(TOperator) == typeof(StaticUnaryOperator<T, XMath.Operators.Log1p<T>>))
        {
            MapWidened(ref x, xStride, ref destination, destinationStride, length, count, default(LanesKernel<Log1pFunction>));
        }
        else if (typeof(TOperator) == typeof(StaticUnaryOperator<T, XMath.Operators.Expm1<T>>))
        {
            MapWidened(ref x, xStride, ref destination, destinationStride, length, count, default(LanesKernel<Expm1Function>));
        }
        else if (typeof(TOperator) == typeof(StaticUnaryOperator<T, XMath.Operators.Tanh<T>>))
        {
            MapWidened(ref x, xStride, ref destination, destinationStride, length, count, default(LanesKernel<TanhFunction>));
        }
        else if (typeof(TOperator) == typeof(StaticUnaryOperator<T, XMath.Operators.Logistic<T>>))
        {
            MapWidened(ref x, xStride, ref destination, destinationStride, length, count, default(LanesKernel<LogisticFunction>));
        }
        else if (typeof(TOperator) == typeof(StaticUnaryOperator<T, XMath.Operators.Logit<T>>))
        {
            MapWidened(ref x, xStride, ref destination, destinationStride, length, count, default(LanesKernel<LogitFunction>));
        }
        else if (typeof(TOperator) == typeof(StaticUnaryOperator<T, XMath.Operators.LogLogistic<T>>))
        {
            MapWidened(ref x, xStride, ref destination, destinationStride, length, count, default(LanesKernel<LogLogisticFunction>));
        }
        else if (typeof(TOperator) == typeof(StaticUnaryOperator<T, XMath.Operators.Softplus<T>>))
        {
            MapWidened(ref x, xStride, ref destination, destinationStride, length, count, default(LanesKernel<SoftplusFunction>));
        }
        else if (typeof(TOperator) == typeof(StaticUnaryOperator<T, XMath.Operators.Silu<T>>))
        {
            MapWidened(ref x, xStride, ref destination, destinationStride, length, count, default(LanesKernel<SiluFunction>));
        }
        else if (typeof(TOperator) == typeof(StaticUnaryOperator<T, XMath.Operators.ApproximateExp<T>>))
        {
            MapApproximate<T, ApproximateExpFunction, LanesKernel<ExpFunction>>(ref x, xStride, ref destination, destinationStride, length, count);
        }
        else if (typeof(TOperator) == typeof(StaticUnaryOperator<T, XMath.Operators.ApproximateTanh<T>>))
        {
            MapApproximate<T, ApproximateTanhFunction, LanesKernel<TanhFunction>>(ref x, xStride, ref destination, destinationStride, length, count);
        }
        else if (typeof(TOperator) == typeof(StaticUnaryOperator<T, XMath.Operators.ApproximateLogistic<T>>))
        {
            MapApproximate<T, ApproximateLogisticFunction, LanesKernel<LogisticFunction>>(ref x, xStride, ref destination, destinationStride, length, count);
        }
        else if (typeof(TOperator) == typeof(StaticUnaryOperator<T, XMath.Operators.ApproximateSilu<T>>))
        {
            MapApproximate<T, ApproximateSiluFunction, LanesKernel<SiluFunction>>(ref x, xStride, ref destination, destinationStride, length, count);
        }
        else if (typeof(TOperator) == typeof(StaticUnaryOperator<T, XMath.Operators.ApproximateSoftplus<T>>))
        {
            MapApproximate<T, ApproximateSoftplusFunction, LanesKernel<SoftplusFunction>>(ref x, xStride, ref destination, destinationStride, length, count);
        }
        else if (typeof(TOperator) == typeof(StaticUnaryOperator<T, XMath.Operators.ApproximateGelu<T>>))
        {
            // double は高精度な指数で tanh 近似式を計算し、float のみ近似ロジスティック関数を使う
            MapApproximate<T, ApproximateGeluFunction, LanesKernel<GeluTanhFunction>>(ref x, xStride, ref destination, destinationStride, length, count);
        }
        else if (typeof(TOperator) == typeof(StaticUnaryOperator<T, XMath.Operators.Selu<T>>))
        {
            MapWidened(ref x, xStride, ref destination, destinationStride, length, count, default(LanesKernel<SeluFunction>));
        }
        else if (typeof(TOperator) == typeof(XMath.Operators.Elu<T>))
        {
            // スカラー定義と同様に alpha を double に拡張して計算する
            double alpha = double.CreateTruncating(Unsafe.As<TOperator, XMath.Operators.Elu<T>>(ref op).Alpha);
            MapWidened(ref x, xStride, ref destination, destinationStride, length, count, new EluKernel(alpha));
        }
        else
        {
            return false;
        }

        return true;
    }

    private static void MapWidened<T, TKernel>(ref T x, int xStride, ref T destination, int destinationStride, int length, int count, TKernel kernel)
        where T : unmanaged
        where TKernel : IExactUnaryKernel<double>
    {
        if (typeof(T) == typeof(double))
        {
            MapLines(ref Unsafe.As<T, double>(ref x), xStride, ref Unsafe.As<T, double>(ref destination), destinationStride, length, count, kernel);
        }
        else
        {
            MapLines(ref Unsafe.As<T, float>(ref x), xStride, ref Unsafe.As<T, float>(ref destination), destinationStride, length, count,
                new WidenedKernel<TKernel>(kernel));
        }
    }

    // float 入力を double に拡張してカーネルを適用し、結果を float へ丸めるラッパー（ループへインライン展開する）。
    private readonly struct WidenedKernel<TKernel>(TKernel kernel) : IExactUnaryKernel<float> where TKernel : IExactUnaryKernel<double>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Invoke(float x) => (float)kernel.Invoke(x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector128<float> Invoke(Vector128<float> x)
        {
            var (lower, upper) = Vector128.Widen(x);
            return Vector128.Narrow(kernel.Invoke(lower), kernel.Invoke(upper));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector256<float> Invoke(Vector256<float> x)
        {
            var (lower, upper) = Vector256.Widen(x);
            return Vector256.Narrow(kernel.Invoke(lower), kernel.Invoke(upper));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector512<float> Invoke(Vector512<float> x)
        {
            var (lower, upper) = Vector512.Widen(x);
            return Vector512.Narrow(kernel.Invoke(lower), kernel.Invoke(upper));
        }
    }

    // ---------- 複合演算向け共通抽象 ----------
    // AggressiveOptimization ループへ確実に展開するため全メンバーに AggressiveInlining を付ける。比較は全ビット 1/0 マスクを返す。
    private interface ILanes<TSelf> where TSelf : struct, ILanes<TSelf>
    {
        static abstract TSelf Create(double value);
        static abstract TSelf operator +(TSelf x, TSelf y);
        static abstract TSelf operator -(TSelf x, TSelf y);
        static abstract TSelf operator *(TSelf x, TSelf y);
        static abstract TSelf operator /(TSelf x, TSelf y);
        static abstract TSelf operator -(TSelf x);
        static abstract TSelf operator &(TSelf x, TSelf y);
        static abstract TSelf Abs(TSelf x);
        static abstract TSelf CopySign(TSelf magnitude, TSelf sign);
        static abstract bool AllTrue(TSelf mask);
        static abstract TSelf Exp(TSelf x);
        static abstract TSelf ExpNormal(TSelf x);
        static abstract TSelf Log(TSelf x);
        static abstract TSelf Hypot(TSelf x, TSelf y);
        static abstract TSelf Equal(TSelf x, TSelf y);
        static abstract TSelf LessThan(TSelf x, TSelf y);
        static abstract TSelf GreaterThan(TSelf x, TSelf y);
        static abstract TSelf GreaterThanOrEqual(TSelf x, TSelf y);
        static abstract TSelf LessThanOrEqual(TSelf x, TSelf y);
        static abstract TSelf Select(TSelf mask, TSelf ifTrue, TSelf ifFalse);
    }

    private readonly struct ScalarLanes(double value) : ILanes<ScalarLanes>
    {
        private static readonly double True = BitConverter.UInt64BitsToDouble(ulong.MaxValue);

        public double Value => value;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarLanes Create(double value) => new(value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarLanes operator +(ScalarLanes x, ScalarLanes y) => new(x.Value + y.Value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarLanes operator -(ScalarLanes x, ScalarLanes y) => new(x.Value - y.Value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarLanes operator *(ScalarLanes x, ScalarLanes y) => new(x.Value * y.Value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarLanes operator /(ScalarLanes x, ScalarLanes y) => new(x.Value / y.Value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarLanes operator -(ScalarLanes x) => new(-x.Value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarLanes operator &(ScalarLanes x, ScalarLanes y) =>
            new(BitConverter.UInt64BitsToDouble(BitConverter.DoubleToUInt64Bits(x.Value) & BitConverter.DoubleToUInt64Bits(y.Value)));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarLanes Abs(ScalarLanes x) => new(Math.Abs(x.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarLanes CopySign(ScalarLanes magnitude, ScalarLanes sign) => new(Math.CopySign(magnitude.Value, sign.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool AllTrue(ScalarLanes mask) => BitConverter.DoubleToUInt64Bits(mask.Value) == ulong.MaxValue;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarLanes Exp(ScalarLanes x) => new(DenseElementwise.Exp(x.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarLanes ExpNormal(ScalarLanes x) => new(DenseElementwise.ExpNormal(x.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarLanes Log(ScalarLanes x) => new(DenseElementwise.Log(x.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarLanes Hypot(ScalarLanes x, ScalarLanes y) => new(double.Hypot(x.Value, y.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarLanes Equal(ScalarLanes x, ScalarLanes y) => new(x.Value == y.Value ? True : 0);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarLanes LessThan(ScalarLanes x, ScalarLanes y) => new(x.Value < y.Value ? True : 0);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarLanes GreaterThan(ScalarLanes x, ScalarLanes y) => new(x.Value > y.Value ? True : 0);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarLanes GreaterThanOrEqual(ScalarLanes x, ScalarLanes y) => new(x.Value >= y.Value ? True : 0);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarLanes LessThanOrEqual(ScalarLanes x, ScalarLanes y) => new(x.Value <= y.Value ? True : 0);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarLanes Select(ScalarLanes mask, ScalarLanes ifTrue, ScalarLanes ifFalse) =>
            BitConverter.DoubleToUInt64Bits(mask.Value) != 0 ? ifTrue : ifFalse;
    }

    private readonly struct Lanes128(Vector128<double> value) : ILanes<Lanes128>
    {
        public Vector128<double> Value => value;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes128 Create(double value) => new(Vector128.Create(value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes128 operator +(Lanes128 x, Lanes128 y) => new(x.Value + y.Value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes128 operator -(Lanes128 x, Lanes128 y) => new(x.Value - y.Value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes128 operator *(Lanes128 x, Lanes128 y) => new(x.Value * y.Value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes128 operator /(Lanes128 x, Lanes128 y) => new(x.Value / y.Value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes128 operator -(Lanes128 x) => new(-x.Value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes128 operator &(Lanes128 x, Lanes128 y) => new(x.Value & y.Value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes128 Abs(Lanes128 x) => new(Vector128.Abs(x.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes128 CopySign(Lanes128 magnitude, Lanes128 sign) => new(Vector128.CopySign(magnitude.Value, sign.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool AllTrue(Lanes128 mask) => Vector128.EqualsAll(mask.Value.AsUInt64(), Vector128.Create(ulong.MaxValue));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes128 Exp(Lanes128 x) => new(DenseElementwise.Exp(x.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes128 ExpNormal(Lanes128 x) => new(DenseElementwise.ExpNormal(x.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes128 Log(Lanes128 x) => new(DenseElementwise.Log(x.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes128 Hypot(Lanes128 x, Lanes128 y) => new(DenseElementwise.Hypot(x.Value, y.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes128 Equal(Lanes128 x, Lanes128 y) => new(Vector128.Equals(x.Value, y.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes128 LessThan(Lanes128 x, Lanes128 y) => new(Vector128.LessThan(x.Value, y.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes128 GreaterThan(Lanes128 x, Lanes128 y) => new(Vector128.GreaterThan(x.Value, y.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes128 GreaterThanOrEqual(Lanes128 x, Lanes128 y) => new(Vector128.GreaterThanOrEqual(x.Value, y.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes128 LessThanOrEqual(Lanes128 x, Lanes128 y) => new(Vector128.LessThanOrEqual(x.Value, y.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes128 Select(Lanes128 mask, Lanes128 ifTrue, Lanes128 ifFalse) => new(Vector128.ConditionalSelect(mask.Value, ifTrue.Value, ifFalse.Value));
    }

    private readonly struct Lanes256(Vector256<double> value) : ILanes<Lanes256>
    {
        public Vector256<double> Value => value;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes256 Create(double value) => new(Vector256.Create(value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes256 operator +(Lanes256 x, Lanes256 y) => new(x.Value + y.Value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes256 operator -(Lanes256 x, Lanes256 y) => new(x.Value - y.Value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes256 operator *(Lanes256 x, Lanes256 y) => new(x.Value * y.Value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes256 operator /(Lanes256 x, Lanes256 y) => new(x.Value / y.Value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes256 operator -(Lanes256 x) => new(-x.Value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes256 operator &(Lanes256 x, Lanes256 y) => new(x.Value & y.Value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes256 Abs(Lanes256 x) => new(Vector256.Abs(x.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes256 CopySign(Lanes256 magnitude, Lanes256 sign) => new(Vector256.CopySign(magnitude.Value, sign.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool AllTrue(Lanes256 mask) => Vector256.EqualsAll(mask.Value.AsUInt64(), Vector256.Create(ulong.MaxValue));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes256 Exp(Lanes256 x) => new(DenseElementwise.Exp(x.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes256 ExpNormal(Lanes256 x) => new(DenseElementwise.ExpNormal(x.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes256 Log(Lanes256 x) => new(DenseElementwise.Log(x.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes256 Hypot(Lanes256 x, Lanes256 y) => new(DenseElementwise.Hypot(x.Value, y.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes256 Equal(Lanes256 x, Lanes256 y) => new(Vector256.Equals(x.Value, y.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes256 LessThan(Lanes256 x, Lanes256 y) => new(Vector256.LessThan(x.Value, y.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes256 GreaterThan(Lanes256 x, Lanes256 y) => new(Vector256.GreaterThan(x.Value, y.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes256 GreaterThanOrEqual(Lanes256 x, Lanes256 y) => new(Vector256.GreaterThanOrEqual(x.Value, y.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes256 LessThanOrEqual(Lanes256 x, Lanes256 y) => new(Vector256.LessThanOrEqual(x.Value, y.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes256 Select(Lanes256 mask, Lanes256 ifTrue, Lanes256 ifFalse) => new(Vector256.ConditionalSelect(mask.Value, ifTrue.Value, ifFalse.Value));
    }

    private readonly struct Lanes512(Vector512<double> value) : ILanes<Lanes512>
    {
        public Vector512<double> Value => value;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes512 Create(double value) => new(Vector512.Create(value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes512 operator +(Lanes512 x, Lanes512 y) => new(x.Value + y.Value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes512 operator -(Lanes512 x, Lanes512 y) => new(x.Value - y.Value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes512 operator *(Lanes512 x, Lanes512 y) => new(x.Value * y.Value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes512 operator /(Lanes512 x, Lanes512 y) => new(x.Value / y.Value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes512 operator -(Lanes512 x) => new(-x.Value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes512 operator &(Lanes512 x, Lanes512 y) => new(x.Value & y.Value);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes512 Abs(Lanes512 x) => new(Vector512.Abs(x.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes512 CopySign(Lanes512 magnitude, Lanes512 sign) => new(Vector512.CopySign(magnitude.Value, sign.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool AllTrue(Lanes512 mask) => Vector512.EqualsAll(mask.Value.AsUInt64(), Vector512.Create(ulong.MaxValue));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes512 Exp(Lanes512 x) => new(DenseElementwise.Exp(x.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes512 ExpNormal(Lanes512 x) => new(DenseElementwise.ExpNormal(x.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes512 Log(Lanes512 x) => new(DenseElementwise.Log(x.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes512 Hypot(Lanes512 x, Lanes512 y) => new(DenseElementwise.Hypot(x.Value, y.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes512 Equal(Lanes512 x, Lanes512 y) => new(Vector512.Equals(x.Value, y.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes512 LessThan(Lanes512 x, Lanes512 y) => new(Vector512.LessThan(x.Value, y.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes512 GreaterThan(Lanes512 x, Lanes512 y) => new(Vector512.GreaterThan(x.Value, y.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes512 GreaterThanOrEqual(Lanes512 x, Lanes512 y) => new(Vector512.GreaterThanOrEqual(x.Value, y.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes512 LessThanOrEqual(Lanes512 x, Lanes512 y) => new(Vector512.LessThanOrEqual(x.Value, y.Value));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Lanes512 Select(Lanes512 mask, Lanes512 ifTrue, Lanes512 ifFalse) => new(Vector512.ConditionalSelect(mask.Value, ifTrue.Value, ifFalse.Value));
    }

    private interface ILanesFunction
    {
        static abstract TLanes Invoke<TLanes>(TLanes x) where TLanes : struct, ILanes<TLanes>;
    }

    private readonly struct LanesKernel<TFunction> : IExactUnaryKernel<double> where TFunction : ILanesFunction
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Invoke(double x) => TFunction.Invoke(new ScalarLanes(x)).Value;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector128<double> Invoke(Vector128<double> x) => TFunction.Invoke(new Lanes128(x)).Value;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector256<double> Invoke(Vector256<double> x) => TFunction.Invoke(new Lanes256(x)).Value;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector512<double> Invoke(Vector512<double> x) => TFunction.Invoke(new Lanes512(x)).Value;
    }

    // ---------- 複合演算 ----------

    private readonly struct ExpFunction : ILanesFunction
    {
        public static TLanes Invoke<TLanes>(TLanes x) where TLanes : struct, ILanes<TLanes> => TLanes.Exp(x);
    }

    private readonly struct LogFunction : ILanesFunction
    {
        public static TLanes Invoke<TLanes>(TLanes x) where TLanes : struct, ILanes<TLanes> => TLanes.Log(x);
    }

    // XMath.Special.Log1p: u = 1 + x; u == 1 ? x : u >= 2/ε ? ln(u) : ln(u) * x / (u - 1)
    private readonly struct Log1pFunction : ILanesFunction
    {
        public static TLanes Invoke<TLanes>(TLanes x) where TLanes : struct, ILanes<TLanes> => Log1p(x);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TLanes Log1p<TLanes>(TLanes x) where TLanes : struct, ILanes<TLanes>
    {
        var one = TLanes.Create(1);
        var u = one + x;
        var log = TLanes.Log(u);
        return Log1pFinish(x, u, log);
    }

    // u = 1 + x, log = ln(u) からの Log1p 補正項
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TLanes Log1pFinish<TLanes>(TLanes x, TLanes u, TLanes log) where TLanes : struct, ILanes<TLanes>
    {
        var one = TLanes.Create(1);
        var y = TLanes.Select(TLanes.GreaterThanOrEqual(u, TLanes.Create(2 / NumericConstants.MachineEpsilon)), log, log * x / (u - one));
        return TLanes.Select(TLanes.Equal(u, one), x, y);
    }

    // XMath.Special.Expm1: |x| < 1e-5 は多項式、それ以外は u = e^x から補正して計算する
    private readonly struct Expm1Function : ILanesFunction
    {
        public static TLanes Invoke<TLanes>(TLanes x) where TLanes : struct, ILanes<TLanes> => Expm1(x);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TLanes Expm1<TLanes>(TLanes x) where TLanes : struct, ILanes<TLanes>
    {
        var one = TLanes.Create(1);
        var small = x + x * x * (TLanes.Create(0.5) + x / TLanes.Create(6));
        var u = TLanes.Exp(x);
        var um1 = u - one;
        var y = TLanes.Select(TLanes.GreaterThanOrEqual(u, TLanes.Create(1 / NumericConstants.MachineEpsilon)), um1, um1 * x / TLanes.Log(u));
        y = TLanes.Select(TLanes.Equal(um1, TLanes.Create(-1)), TLanes.Create(-1), y);
        y = TLanes.Select(TLanes.Equal(u, one), x, y);
        return TLanes.Select(TLanes.LessThan(TLanes.Abs(x), TLanes.Create(1e-5)), small, y);
    }

    internal static double Tanh(double x) => TanhFunction.Invoke(new ScalarLanes(x)).Value;

    private readonly struct TanhFunction : ILanesFunction
    {
        private const double PolynomialLimit = 0.75;
        private const double SaturationLimit = 20;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TLanes Invoke<TLanes>(TLanes x) where TLanes : struct, ILanes<TLanes>
        {
            var magnitude = TLanes.Abs(x);
            var small = TLanes.LessThanOrEqual(magnitude, TLanes.Create(PolynomialLimit));
            if (TLanes.AllTrue(small))
            {
                var (numerator, denominator) = NearZeroCorrection(magnitude);
                return TLanes.CopySign(magnitude - numerator / denominator, x);
            }

            if (TLanes.AllTrue(TLanes.GreaterThanOrEqual(magnitude, TLanes.Create(SaturationLimit))))
            {
                return TLanes.CopySign(TLanes.Create(1), x);
            }

            var result = TLanes.CopySign(MixedRange(magnitude), x);
            return TLanes.Select(TLanes.Equal(x, x), result, x);
        }

        // float の上下半分に指数の表引きを展開すると JIT のインライン予算を超える。
        [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
        private static TLanes MixedRange<TLanes>(TLanes magnitude) where TLanes : struct, ILanes<TLanes>
        {
            var one = TLanes.Create(1);
            // tanh(20) は 1 に丸まる。指数の引数を [-40, 0] に収める。
            var limit = TLanes.Create(SaturationLimit);
            var bounded = TLanes.Select(TLanes.LessThan(magnitude, limit), magnitude, limit);
            var e = TLanes.ExpNormal(-(bounded + bounded));
            var small = TLanes.LessThanOrEqual(magnitude, TLanes.Create(PolynomialLimit));
            var (p, q) = NearZeroCorrection(bounded);
            // 補正項の分子と分母を選択し、混在するベクトルでも除算は 1 回にまとめる。
            var origin = TLanes.Select(small, bounded, one);
            var numerator = TLanes.Select(small, p, e + e);
            var denominator = TLanes.Select(small, q, one + e);
            return origin - numerator / denominator;
        }

        // tanh(x)/x の [4/4] Padé 近似を x - x³ P(x²)/Q(x²) の形で評価し、原点付近の桁落ちを防ぐ。
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static (TLanes Numerator, TLanes Denominator) NearZeroCorrection<TLanes>(TLanes x) where TLanes : struct, ILanes<TLanes>
        {
            var z = x * x;
            var p = (TLanes.Create(44) * z + TLanes.Create(12870)) * z + TLanes.Create(810810);
            var q = ((TLanes.Create(45) * z + TLanes.Create(13860)) * z + TLanes.Create(945945)) * z + TLanes.Create(16216200);
            return (x * z * (p * z + TLanes.Create(11486475)), q * z + TLanes.Create(34459425));
        }
    }

    // XMath.Logistic: x >= 0 ? 1/(1 + e^-x) : e^x/(1 + e^x)
    private readonly struct LogisticFunction : ILanesFunction
    {
        public static TLanes Invoke<TLanes>(TLanes x) where TLanes : struct, ILanes<TLanes> => Logistic(x);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TLanes Logistic<TLanes>(TLanes x) where TLanes : struct, ILanes<TLanes>
    {
        var one = TLanes.Create(1);
        var nonnegative = TLanes.GreaterThanOrEqual(x, TLanes.Create(0));
        var e = TLanes.Exp(TLanes.Select(nonnegative, -x, x));
        return TLanes.Select(nonnegative, one / (one + e), e / (one + e));
    }

    // XMath.Logit: p in [1/4, 3/4] なら Log1p((2p - 1)/(1 - p))、それ以外は ln(p/(1 - p))（対数評価は 1 回）
    private readonly struct LogitFunction : ILanesFunction
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TLanes Invoke<TLanes>(TLanes p) where TLanes : struct, ILanes<TLanes>
        {
            var one = TLanes.Create(1);
            var middle = TLanes.GreaterThanOrEqual(p, TLanes.Create(0.25)) & TLanes.LessThanOrEqual(p, TLanes.Create(0.75));
            var t = (TLanes.Create(2) * p - one) / (one - p);
            var u = one + t;
            var log = TLanes.Log(TLanes.Select(middle, u, p / (one - p)));
            return TLanes.Select(middle, Log1pFinish(t, u, log), log);
        }
    }

    // XMath.Softplus: x > 0 ? x + Log1p(e^-x) : Log1p(e^x)
    private readonly struct SoftplusFunction : ILanesFunction
    {
        public static TLanes Invoke<TLanes>(TLanes x) where TLanes : struct, ILanes<TLanes> => Softplus(x);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TLanes Softplus<TLanes>(TLanes x) where TLanes : struct, ILanes<TLanes>
    {
        var positive = TLanes.GreaterThan(x, TLanes.Create(0));
        var log = Log1p(TLanes.Exp(TLanes.Select(positive, -x, x)));
        return TLanes.Select(positive, x + log, log);
    }

    private readonly struct LogLogisticFunction : ILanesFunction
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TLanes Invoke<TLanes>(TLanes x) where TLanes : struct, ILanes<TLanes> => -Softplus(-x);
    }

    // XMath.Silu: x = -∞ は -0、x < -709 はスケーリングしてアンダーフローを回避（指数評価は 1 回）
    private readonly struct SiluFunction : ILanesFunction
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TLanes Invoke<TLanes>(TLanes x) where TLanes : struct, ILanes<TLanes>
        {
            var tail = TLanes.LessThan(x, TLanes.Create(XMath.SiluTailBelow));
            var e = TLanes.Exp(TLanes.Select(tail, x + TLanes.Create(XMath.TailExpShift), -x));
            var y = TLanes.Select(tail, x * e * TLanes.Create(XMath.TailExpUnshift), x / (TLanes.Create(1) + e));
            return TLanes.Select(TLanes.Equal(x, TLanes.Create(double.NegativeInfinity)), TLanes.Create(-0.0), y);
        }
    }

    // XMath.ApproximateGelu: z = c * (x + k * x^3), x = -∞ は -0、それ以外は x * Logistic(z)
    private readonly struct GeluTanhFunction : ILanesFunction
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TLanes Invoke<TLanes>(TLanes x) where TLanes : struct, ILanes<TLanes>
        {
            var z = TLanes.Create(XMath.GeluTanhScale) * (x + TLanes.Create(XMath.GeluTanhCubic) * x * x * x);
            var y = x * Logistic(z);
            return TLanes.Select(TLanes.Equal(x, TLanes.Create(double.NegativeInfinity)), TLanes.Create(-0.0), y);
        }
    }

    private readonly struct SeluFunction : ILanesFunction
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TLanes Invoke<TLanes>(TLanes x) where TLanes : struct, ILanes<TLanes>
        {
            var negative = TLanes.Create(XMath.SeluAlpha) * Expm1(x);
            return TLanes.Create(XMath.SeluScale) * TLanes.Select(TLanes.GreaterThan(x, TLanes.Create(0)), x, negative);
        }
    }

    // XMath.Elu（alpha は double に拡張済みの値）
    private readonly struct EluKernel(double alpha) : IExactUnaryKernel<double>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Invoke(double x) => Invoke(new ScalarLanes(x)).Value;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector128<double> Invoke(Vector128<double> x) => Invoke(new Lanes128(x)).Value;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector256<double> Invoke(Vector256<double> x) => Invoke(new Lanes256(x)).Value;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector512<double> Invoke(Vector512<double> x) => Invoke(new Lanes512(x)).Value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private TLanes Invoke<TLanes>(TLanes x) where TLanes : struct, ILanes<TLanes> =>
            TLanes.Select(TLanes.GreaterThan(x, TLanes.Create(0)), x, TLanes.Create(alpha) * Expm1(x));
    }
}
