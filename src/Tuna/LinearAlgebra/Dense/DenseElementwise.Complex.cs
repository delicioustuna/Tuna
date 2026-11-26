using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Tuna.LinearAlgebra;

// 複素数（Complex）の要素ごとの SIMD 実装。実部・虚部をデインターリーブして ILanes 上の共通式で計算し、
// .NET 10 の Complex 演算子とビット単位で一致させる（NaN 成分の符号・ペイロードは未規定）。
internal static partial class DenseElementwise
{
    // Complex 演算子 op に対応する SIMD 実装があれば destination[i] = op.Invoke(x[i]) を計算して true を返す。
    private static bool TryMapComplex<TOperator>(ref Complex x, int xStride, ref Complex destination, int destinationStride, int length, int count, TOperator op)
    {
        if (typeof(TOperator) == typeof(NegateOperator<Complex>))
        {
            MapComplexLines<NegateFunction>(ref x, xStride, default, ref destination, destinationStride, length, count);
        }
        else if (typeof(TOperator) == typeof(StaticUnaryOperator<Complex, XMath.Operators.Abs<Complex>>))
        {
            MapComplexLines<AbsFunction>(ref x, xStride, default, ref destination, destinationStride, length, count);
        }
        else if (typeof(TOperator) == typeof(StaticUnaryOperator<Complex, XMath.Operators.Square<Complex>>))
        {
            MapComplexLines<SquareFunction>(ref x, xStride, default, ref destination, destinationStride, length, count);
        }
        else if (typeof(TOperator) == typeof(StaticUnaryOperator<Complex, XMath.Operators.Reciprocal<Complex>>))
        {
            MapComplexLines<ReciprocalFunction>(ref x, xStride, default, ref destination, destinationStride, length, count);
        }
        else if (typeof(TOperator) == typeof(AddScalarOperator<Complex>))
        {
            MapComplexLines<AddFunction>(ref x, xStride, Unsafe.As<TOperator, AddScalarOperator<Complex>>(ref op).Scalar,
                ref destination, destinationStride, length, count);
        }
        else if (typeof(TOperator) == typeof(SubtractScalarOperator<Complex>))
        {
            MapComplexLines<SubtractFunction>(ref x, xStride, Unsafe.As<TOperator, SubtractScalarOperator<Complex>>(ref op).Scalar,
                ref destination, destinationStride, length, count);
        }
        else if (typeof(TOperator) == typeof(ScaleOperator<Complex>))
        {
            MapComplexLines<MultiplyFunction>(ref x, xStride, Unsafe.As<TOperator, ScaleOperator<Complex>>(ref op).Scalar,
                ref destination, destinationStride, length, count);
        }
        else if (typeof(TOperator) == typeof(XMath.Operators.DivideScalar<Complex>))
        {
            MapComplexLines<DivideFunction>(ref x, xStride, Unsafe.As<TOperator, XMath.Operators.DivideScalar<Complex>>(ref op).Y,
                ref destination, destinationStride, length, count);
        }
        else
        {
            return false;
        }

        return true;
    }

    // Complex 二項演算子に対応する SIMD 実装があれば destination[i] = op.Invoke(x[i], y[i]) を計算して true を返す。
    private static bool TryZipComplex<TOperator>(ref Complex x, int xStride, ref Complex y, int yStride, ref Complex destination, int destinationStride, int length, int count)
    {
        if (typeof(TOperator) == typeof(AddOperator<Complex>))
        {
            ZipComplexLines<AddFunction>(ref x, xStride, ref y, yStride, ref destination, destinationStride, length, count);
        }
        else if (typeof(TOperator) == typeof(SubtractOperator<Complex>))
        {
            ZipComplexLines<SubtractFunction>(ref x, xStride, ref y, yStride, ref destination, destinationStride, length, count);
        }
        else if (typeof(TOperator) == typeof(MultiplyOperator<Complex>))
        {
            ZipComplexLines<MultiplyFunction>(ref x, xStride, ref y, yStride, ref destination, destinationStride, length, count);
        }
        else if (typeof(TOperator) == typeof(StaticBinaryOperator<Complex, XMath.Operators.Divide<Complex>>))
        {
            ZipComplexLines<DivideFunction>(ref x, xStride, ref y, yStride, ref destination, destinationStride, length, count);
        }
        else
        {
            return false;
        }

        return true;
    }

    // destination[i] = x[i] * scalar（Complex 乗算と同じ値）。
    internal static void ScaleComplex(ReadOnlySpan<Complex> x, Complex scalar, Span<Complex> destination)
    {
        DenseStorage.RequireLength(x.Length, destination.Length);
        MapComplexLines<MultiplyFunction>(ref MemoryMarshal.GetReference(x), 0, scalar, ref MemoryMarshal.GetReference(destination), 0, x.Length, 1);
    }

    // 単一区間は MapComplexLine、複数区間は MapManyComplexLines へ振り分ける（単項演算では y 未使用）。
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void MapComplexLines<TFunction>(ref Complex x, int xStride, Complex y, ref Complex destination, int destinationStride, int length, int count)
        where TFunction : IComplexFunction
    {
        if (count == 1)
        {
            MapComplexLine<TFunction>(ref x, y, ref destination, length);
        }
        else
        {
            MapManyComplexLines<TFunction>(ref x, xStride, y, ref destination, destinationStride, length, count);
        }
    }

    // 単一区間向け（レジスタ退避を避けるため本体を直接記述する）。
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static void MapComplexLine<TFunction>(ref Complex x, Complex y, ref Complex destination, int length)
        where TFunction : IComplexFunction
    {
        ref Complex xk = ref x;
        ref Complex ck = ref destination;
        ref double a = ref Unsafe.As<Complex, double>(ref xk);
        ref double c = ref Unsafe.As<Complex, double>(ref ck);
        int i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            var yr = new Lanes512(Vector512.Create(y.Real));
            var yi = new Lanes512(Vector512.Create(y.Imaginary));
            for (int width = Vector512<double>.Count; i <= length - width; i += width)
            {
                var (xr, xi) = Deinterleave(Vector512.LoadUnsafe(ref a, (nuint)(2 * i)), Vector512.LoadUnsafe(ref a, (nuint)(2 * i + width)));
                TFunction.Invoke(new Lanes512(xr), new Lanes512(xi), yr, yi, out var zr, out var zi);
                var (first, second) = Interleave(zr.Value, zi.Value);
                first.StoreUnsafe(ref c, (nuint)(2 * i));
                second.StoreUnsafe(ref c, (nuint)(2 * i + width));
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var yr = new Lanes256(Vector256.Create(y.Real));
            var yi = new Lanes256(Vector256.Create(y.Imaginary));
            for (int width = Vector256<double>.Count; i <= length - width; i += width)
            {
                var (xr, xi) = Deinterleave(Vector256.LoadUnsafe(ref a, (nuint)(2 * i)), Vector256.LoadUnsafe(ref a, (nuint)(2 * i + width)));
                TFunction.Invoke(new Lanes256(xr), new Lanes256(xi), yr, yi, out var zr, out var zi);
                var (first, second) = Interleave(zr.Value, zi.Value);
                first.StoreUnsafe(ref c, (nuint)(2 * i));
                second.StoreUnsafe(ref c, (nuint)(2 * i + width));
            }
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            var yr = new Lanes128(Vector128.Create(y.Real));
            var yi = new Lanes128(Vector128.Create(y.Imaginary));
            for (int width = Vector128<double>.Count; i <= length - width; i += width)
            {
                var (xr, xi) = Deinterleave(Vector128.LoadUnsafe(ref a, (nuint)(2 * i)), Vector128.LoadUnsafe(ref a, (nuint)(2 * i + width)));
                TFunction.Invoke(new Lanes128(xr), new Lanes128(xi), yr, yi, out var zr, out var zi);
                var (first, second) = Interleave(zr.Value, zi.Value);
                first.StoreUnsafe(ref c, (nuint)(2 * i));
                second.StoreUnsafe(ref c, (nuint)(2 * i + width));
            }
        }

        for (; i < length; i++)
        {
            Complex xValue = Unsafe.Add(ref xk, i);
            TFunction.Invoke(new ScalarLanes(xValue.Real), new ScalarLanes(xValue.Imaginary), new ScalarLanes(y.Real), new ScalarLanes(y.Imaginary), out var zr, out var zi);
            Unsafe.Add(ref ck, i) = new Complex(zr.Value, zi.Value);
        }
    }

    // 複数区間向け。
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static void MapManyComplexLines<TFunction>(ref Complex x, int xStride, Complex y, ref Complex destination, int destinationStride, int length, int count)
        where TFunction : IComplexFunction
    {
        for (int k = 0; k < count; k++)
        {
            ref Complex xk = ref Unsafe.Add(ref x, (nint)k * xStride);
            ref Complex ck = ref Unsafe.Add(ref destination, (nint)k * destinationStride);
            ref double a = ref Unsafe.As<Complex, double>(ref xk);
            ref double c = ref Unsafe.As<Complex, double>(ref ck);
            int i = 0;
            if (Vector512.IsHardwareAccelerated)
            {
                var yr = new Lanes512(Vector512.Create(y.Real));
                var yi = new Lanes512(Vector512.Create(y.Imaginary));
                for (int width = Vector512<double>.Count; i <= length - width; i += width)
                {
                    var (xr, xi) = Deinterleave(Vector512.LoadUnsafe(ref a, (nuint)(2 * i)), Vector512.LoadUnsafe(ref a, (nuint)(2 * i + width)));
                    TFunction.Invoke(new Lanes512(xr), new Lanes512(xi), yr, yi, out var zr, out var zi);
                    var (first, second) = Interleave(zr.Value, zi.Value);
                    first.StoreUnsafe(ref c, (nuint)(2 * i));
                    second.StoreUnsafe(ref c, (nuint)(2 * i + width));
                }
            }
            else if (Vector256.IsHardwareAccelerated)
            {
                var yr = new Lanes256(Vector256.Create(y.Real));
                var yi = new Lanes256(Vector256.Create(y.Imaginary));
                for (int width = Vector256<double>.Count; i <= length - width; i += width)
                {
                    var (xr, xi) = Deinterleave(Vector256.LoadUnsafe(ref a, (nuint)(2 * i)), Vector256.LoadUnsafe(ref a, (nuint)(2 * i + width)));
                    TFunction.Invoke(new Lanes256(xr), new Lanes256(xi), yr, yi, out var zr, out var zi);
                    var (first, second) = Interleave(zr.Value, zi.Value);
                    first.StoreUnsafe(ref c, (nuint)(2 * i));
                    second.StoreUnsafe(ref c, (nuint)(2 * i + width));
                }
            }
            else if (Vector128.IsHardwareAccelerated)
            {
                var yr = new Lanes128(Vector128.Create(y.Real));
                var yi = new Lanes128(Vector128.Create(y.Imaginary));
                for (int width = Vector128<double>.Count; i <= length - width; i += width)
                {
                    var (xr, xi) = Deinterleave(Vector128.LoadUnsafe(ref a, (nuint)(2 * i)), Vector128.LoadUnsafe(ref a, (nuint)(2 * i + width)));
                    TFunction.Invoke(new Lanes128(xr), new Lanes128(xi), yr, yi, out var zr, out var zi);
                    var (first, second) = Interleave(zr.Value, zi.Value);
                    first.StoreUnsafe(ref c, (nuint)(2 * i));
                    second.StoreUnsafe(ref c, (nuint)(2 * i + width));
                }
            }

            for (; i < length; i++)
            {
                Complex xValue = Unsafe.Add(ref xk, i);
                TFunction.Invoke(new ScalarLanes(xValue.Real), new ScalarLanes(xValue.Imaginary), new ScalarLanes(y.Real), new ScalarLanes(y.Imaginary), out var zr, out var zi);
                Unsafe.Add(ref ck, i) = new Complex(zr.Value, zi.Value);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ZipComplexLines<TFunction>(ref Complex x, int xStride, ref Complex y, int yStride, ref Complex destination, int destinationStride, int length, int count)
        where TFunction : IComplexFunction
    {
        for (int k = 0; k < count; k++)
        {
            ref Complex xk = ref Unsafe.Add(ref x, (nint)k * xStride);
            ref Complex yk = ref Unsafe.Add(ref y, (nint)k * yStride);
            ref Complex ck = ref Unsafe.Add(ref destination, (nint)k * destinationStride);
            ref double a = ref Unsafe.As<Complex, double>(ref xk);
            ref double b = ref Unsafe.As<Complex, double>(ref yk);
            ref double c = ref Unsafe.As<Complex, double>(ref ck);
            int i = 0;
            if (Vector512.IsHardwareAccelerated)
            {
                for (int width = Vector512<double>.Count; i <= length - width; i += width)
                {
                    var (xr, xi) = Deinterleave(Vector512.LoadUnsafe(ref a, (nuint)(2 * i)), Vector512.LoadUnsafe(ref a, (nuint)(2 * i + width)));
                    var (yr, yi) = Deinterleave(Vector512.LoadUnsafe(ref b, (nuint)(2 * i)), Vector512.LoadUnsafe(ref b, (nuint)(2 * i + width)));
                    TFunction.Invoke(new Lanes512(xr), new Lanes512(xi), new Lanes512(yr), new Lanes512(yi), out var zr, out var zi);
                    var (first, second) = Interleave(zr.Value, zi.Value);
                    first.StoreUnsafe(ref c, (nuint)(2 * i));
                    second.StoreUnsafe(ref c, (nuint)(2 * i + width));
                }
            }
            else if (Vector256.IsHardwareAccelerated)
            {
                for (int width = Vector256<double>.Count; i <= length - width; i += width)
                {
                    var (xr, xi) = Deinterleave(Vector256.LoadUnsafe(ref a, (nuint)(2 * i)), Vector256.LoadUnsafe(ref a, (nuint)(2 * i + width)));
                    var (yr, yi) = Deinterleave(Vector256.LoadUnsafe(ref b, (nuint)(2 * i)), Vector256.LoadUnsafe(ref b, (nuint)(2 * i + width)));
                    TFunction.Invoke(new Lanes256(xr), new Lanes256(xi), new Lanes256(yr), new Lanes256(yi), out var zr, out var zi);
                    var (first, second) = Interleave(zr.Value, zi.Value);
                    first.StoreUnsafe(ref c, (nuint)(2 * i));
                    second.StoreUnsafe(ref c, (nuint)(2 * i + width));
                }
            }
            else if (Vector128.IsHardwareAccelerated)
            {
                for (int width = Vector128<double>.Count; i <= length - width; i += width)
                {
                    var (xr, xi) = Deinterleave(Vector128.LoadUnsafe(ref a, (nuint)(2 * i)), Vector128.LoadUnsafe(ref a, (nuint)(2 * i + width)));
                    var (yr, yi) = Deinterleave(Vector128.LoadUnsafe(ref b, (nuint)(2 * i)), Vector128.LoadUnsafe(ref b, (nuint)(2 * i + width)));
                    TFunction.Invoke(new Lanes128(xr), new Lanes128(xi), new Lanes128(yr), new Lanes128(yi), out var zr, out var zi);
                    var (first, second) = Interleave(zr.Value, zi.Value);
                    first.StoreUnsafe(ref c, (nuint)(2 * i));
                    second.StoreUnsafe(ref c, (nuint)(2 * i + width));
                }
            }

            for (; i < length; i++)
            {
                Complex xValue = Unsafe.Add(ref xk, i);
                Complex yValue = Unsafe.Add(ref yk, i);
                TFunction.Invoke(new ScalarLanes(xValue.Real), new ScalarLanes(xValue.Imaginary), new ScalarLanes(yValue.Real), new ScalarLanes(yValue.Imaginary), out var zr, out var zi);
                Unsafe.Add(ref ck, i) = new Complex(zr.Value, zi.Value);
            }
        }
    }

    // ---------- 実部・虚部への分解と戻し ----------

    // first = [r0 i0 r1 i1 r2 i2 r3 i3]、second = [r4 i4 … r7 i7] を、実部 [r0 … r7] と虚部 [i0 … i7] に分ける
    private static (Vector512<double> Real, Vector512<double> Imaginary) Deinterleave(Vector512<double> first, Vector512<double> second)
    {
        var order = Vector512.Create(0L, 2, 4, 6, 1, 3, 5, 7);
        var a = Vector512.Shuffle(first, order);
        var b = Vector512.Shuffle(second, order);
        return (Vector512.Create(a.GetLower(), b.GetLower()), Vector512.Create(a.GetUpper(), b.GetUpper()));
    }

    private static (Vector512<double> First, Vector512<double> Second) Interleave(Vector512<double> real, Vector512<double> imaginary)
    {
        var order = Vector512.Create(0L, 4, 1, 5, 2, 6, 3, 7);
        return (Vector512.Shuffle(Vector512.Create(real.GetLower(), imaginary.GetLower()), order),
            Vector512.Shuffle(Vector512.Create(real.GetUpper(), imaginary.GetUpper()), order));
    }

    private static (Vector256<double> Real, Vector256<double> Imaginary) Deinterleave(Vector256<double> first, Vector256<double> second)
    {
        var order = Vector256.Create(0L, 2, 1, 3);
        var a = Vector256.Shuffle(first, order);
        var b = Vector256.Shuffle(second, order);
        return (Vector256.Create(a.GetLower(), b.GetLower()), Vector256.Create(a.GetUpper(), b.GetUpper()));
    }

    private static (Vector256<double> First, Vector256<double> Second) Interleave(Vector256<double> real, Vector256<double> imaginary)
    {
        var order = Vector256.Create(0L, 2, 1, 3);
        return (Vector256.Shuffle(Vector256.Create(real.GetLower(), imaginary.GetLower()), order),
            Vector256.Shuffle(Vector256.Create(real.GetUpper(), imaginary.GetUpper()), order));
    }

    private static (Vector128<double> Real, Vector128<double> Imaginary) Deinterleave(Vector128<double> first, Vector128<double> second) =>
        (Vector128.Create(first.ToScalar(), second.ToScalar()), Vector128.Create(first.GetElement(1), second.GetElement(1)));

    private static (Vector128<double> First, Vector128<double> Second) Interleave(Vector128<double> real, Vector128<double> imaginary) =>
        (Vector128.Create(real.ToScalar(), imaginary.ToScalar()), Vector128.Create(real.GetElement(1), imaginary.GetElement(1)));

    // ---------- 演算式 ----------
    // 各実装には AggressiveInlining を付け、AggressiveOptimization ループへ確実にインライン展開させる。
    private interface IComplexFunction
    {
        static abstract void Invoke<TLanes>(TLanes xr, TLanes xi, TLanes yr, TLanes yi, out TLanes zr, out TLanes zi) where TLanes : struct, ILanes<TLanes>;
    }

    private readonly struct AddFunction : IComplexFunction
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Invoke<TLanes>(TLanes xr, TLanes xi, TLanes yr, TLanes yi, out TLanes zr, out TLanes zi) where TLanes : struct, ILanes<TLanes>
        {
            zr = xr + yr;
            zi = xi + yi;
        }
    }

    private readonly struct SubtractFunction : IComplexFunction
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Invoke<TLanes>(TLanes xr, TLanes xi, TLanes yr, TLanes yi, out TLanes zr, out TLanes zi) where TLanes : struct, ILanes<TLanes>
        {
            zr = xr - yr;
            zi = xi - yi;
        }
    }

    // 成分ごとの符号反転（0 - x ではなく符号ビット反転）。
    private readonly struct NegateFunction : IComplexFunction
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Invoke<TLanes>(TLanes xr, TLanes xi, TLanes yr, TLanes yi, out TLanes zr, out TLanes zi) where TLanes : struct, ILanes<TLanes>
        {
            zr = -xr;
            zi = -xi;
        }
    }

    private readonly struct MultiplyFunction : IComplexFunction
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Invoke<TLanes>(TLanes xr, TLanes xi, TLanes yr, TLanes yi, out TLanes zr, out TLanes zi) where TLanes : struct, ILanes<TLanes>
        {
            zr = xr * yr - xi * yi;
            zi = xi * yr + xr * yi;
        }
    }

    private readonly struct SquareFunction : IComplexFunction
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Invoke<TLanes>(TLanes xr, TLanes xi, TLanes yr, TLanes yi, out TLanes zr, out TLanes zi) where TLanes : struct, ILanes<TLanes> =>
            MultiplyFunction.Invoke(xr, xi, xr, xi, out zr, out zi);
    }

    // Smith の複素数除算。|yi| < |yr| の分岐前に被演算子を選択し除算回数をスカラーと同じ 3 回に抑える。
    // 虚部は減算と加算で NaN の符号が異なるため両方を計算して選択する。
    private readonly struct DivideFunction : IComplexFunction
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Invoke<TLanes>(TLanes xr, TLanes xi, TLanes yr, TLanes yi, out TLanes zr, out TLanes zi) where TLanes : struct, ILanes<TLanes>
        {
            var first = TLanes.LessThan(TLanes.Abs(yi), TLanes.Abs(yr));
            var p = TLanes.Select(first, yi, yr);
            var q = TLanes.Select(first, yr, yi);
            var t = p / q;
            var d = q + p * t;
            var real = TLanes.Select(first, xr + xi * t, xi + xr * t);
            var imaginary = TLanes.Select(first, xi - xr * t, -xr + xi * t);
            zr = real / d;
            zi = imaginary / d;
        }
    }

    // XMath.Reciprocal: Complex.One / x（Complex.Reciprocal と異なり 0 を特別扱いしない）。
    private readonly struct ReciprocalFunction : IComplexFunction
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Invoke<TLanes>(TLanes xr, TLanes xi, TLanes yr, TLanes yi, out TLanes zr, out TLanes zi) where TLanes : struct, ILanes<TLanes> =>
            DivideFunction.Invoke(TLanes.Create(1), TLanes.Create(0), xr, xi, out zr, out zi);
    }

    private readonly struct AbsFunction : IComplexFunction
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Invoke<TLanes>(TLanes xr, TLanes xi, TLanes yr, TLanes yi, out TLanes zr, out TLanes zi) where TLanes : struct, ILanes<TLanes>
        {
            zr = TLanes.Hypot(xr, xi);
            zi = TLanes.Create(0);
        }
    }
}
