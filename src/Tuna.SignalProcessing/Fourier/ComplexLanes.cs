using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Tuna.SignalProcessing;

// SIMD レーン抽象化。積和命令を使わず複素数積を x·wr + (i·x)·wi の順に統一することで、
// Lanes1/2/4 どの幅や端数処理でもビット単位で同一の結果を保証する。
internal interface IComplexLanes<TSelf> where TSelf : unmanaged, IComplexLanes<TSelf>
{
    static abstract int Count { get; }

    static abstract bool IsAccelerated { get; }

    static abstract TSelf Load(ref readonly Complex source);

    static abstract void Store(ref Complex destination, TSelf value);

    static abstract TSelf Add(TSelf x, TSelf y);

    static abstract TSelf Subtract(TSelf x, TSelf y);

    static abstract TSelf Scale(TSelf x, double factor);

    static abstract TSelf Broadcast(double value);

    static abstract TSelf Product(TSelf x, TSelf y);

    static abstract void StoreLanes(ref Complex destination, nint stride, TSelf value);

    static abstract TSelf MultiplyEach(TSelf x, TSelf w);

    static abstract TSelf MultiplyEachConjugate(TSelf x, TSelf w);

    static abstract TSelf Multiply(TSelf x, Complex w);

    static abstract TSelf MultiplyConjugate(TSelf x, Complex w);

    static abstract TSelf MultiplyI(TSelf x);

    static abstract TSelf MultiplyMinusI(TSelf x);

    static abstract TSelf Conjugate(TSelf x);

    static abstract TSelf Reverse(TSelf x);
}

internal readonly struct Lanes1(Vector128<double> value) : IComplexLanes<Lanes1>
{
    private readonly Vector128<double> value = value;

    public static int Count => 1;

    public static bool IsAccelerated => Vector128.IsHardwareAccelerated;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes1 Load(ref readonly Complex source) =>
        new(Vector128.LoadUnsafe(ref Unsafe.As<Complex, double>(ref Unsafe.AsRef(in source))));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store(ref Complex destination, Lanes1 value) =>
        value.value.StoreUnsafe(ref Unsafe.As<Complex, double>(ref destination));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void StoreLanes(ref Complex destination, nint stride, Lanes1 value)
    {
        Store(ref destination, value);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes1 Add(Lanes1 x, Lanes1 y) => new(x.value + y.value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes1 Subtract(Lanes1 x, Lanes1 y) => new(x.value - y.value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes1 Scale(Lanes1 x, double factor) => new(x.value * Vector128.Create(factor));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes1 Broadcast(double value) => new(Vector128.Create(value));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes1 Product(Lanes1 x, Lanes1 y) => new(x.value * y.value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes1 Multiply(Lanes1 x, Complex w) =>
        new(x.value * Vector128.Create(w.Real) + MultiplyI(x).value * Vector128.Create(w.Imaginary));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes1 MultiplyConjugate(Lanes1 x, Complex w) =>
        new(x.value * Vector128.Create(w.Real) + MultiplyMinusI(x).value * Vector128.Create(w.Imaginary));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes1 MultiplyEach(Lanes1 x, Lanes1 w) =>
        new(x.value * Vector128.Shuffle(w.value, Vector128.Create(0L, 0)) + MultiplyI(x).value * Vector128.Shuffle(w.value, Vector128.Create(1L, 1)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes1 MultiplyEachConjugate(Lanes1 x, Lanes1 w) =>
        new(x.value * Vector128.Shuffle(w.value, Vector128.Create(0L, 0)) + MultiplyMinusI(x).value * Vector128.Shuffle(w.value, Vector128.Create(1L, 1)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes1 MultiplyI(Lanes1 x) => new(Vector128.Xor(Swap(x.value), Vector128.Create(-0.0, 0.0)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes1 MultiplyMinusI(Lanes1 x) => new(Vector128.Xor(Swap(x.value), Vector128.Create(0.0, -0.0)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes1 Conjugate(Lanes1 x) => new(Vector128.Xor(x.value, Vector128.Create(0.0, -0.0)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes1 Reverse(Lanes1 x) => x;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<double> Swap(Vector128<double> v) => Vector128.Shuffle(v, Vector128.Create(1L, 0));
}

internal readonly struct Lanes2(Vector256<double> value) : IComplexLanes<Lanes2>
{
    private readonly Vector256<double> value = value;

    public static int Count => 2;

    public static bool IsAccelerated => Vector256.IsHardwareAccelerated;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes2 Load(ref readonly Complex source) =>
        new(Vector256.LoadUnsafe(ref Unsafe.As<Complex, double>(ref Unsafe.AsRef(in source))));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store(ref Complex destination, Lanes2 value) =>
        value.value.StoreUnsafe(ref Unsafe.As<Complex, double>(ref destination));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void StoreLanes(ref Complex destination, nint stride, Lanes2 value)
    {
        value.value.GetLower().StoreUnsafe(ref Unsafe.As<Complex, double>(ref destination));
        value.value.GetUpper().StoreUnsafe(ref Unsafe.As<Complex, double>(ref Unsafe.Add(ref destination, stride)));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes2 Add(Lanes2 x, Lanes2 y) => new(x.value + y.value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes2 Subtract(Lanes2 x, Lanes2 y) => new(x.value - y.value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes2 Scale(Lanes2 x, double factor) => new(x.value * Vector256.Create(factor));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes2 Broadcast(double value) => new(Vector256.Create(value));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes2 Product(Lanes2 x, Lanes2 y) => new(x.value * y.value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes2 Multiply(Lanes2 x, Complex w) =>
        new(x.value * Vector256.Create(w.Real) + MultiplyI(x).value * Vector256.Create(w.Imaginary));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes2 MultiplyConjugate(Lanes2 x, Complex w) =>
        new(x.value * Vector256.Create(w.Real) + MultiplyMinusI(x).value * Vector256.Create(w.Imaginary));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes2 MultiplyEach(Lanes2 x, Lanes2 w) =>
        new(x.value * Vector256.Shuffle(w.value, Vector256.Create(0L, 0, 2, 2)) + MultiplyI(x).value * Vector256.Shuffle(w.value, Vector256.Create(1L, 1, 3, 3)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes2 MultiplyEachConjugate(Lanes2 x, Lanes2 w) =>
        new(x.value * Vector256.Shuffle(w.value, Vector256.Create(0L, 0, 2, 2)) + MultiplyMinusI(x).value * Vector256.Shuffle(w.value, Vector256.Create(1L, 1, 3, 3)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes2 MultiplyI(Lanes2 x) => new(Vector256.Xor(Swap(x.value), Vector256.Create(-0.0, 0.0, -0.0, 0.0)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes2 MultiplyMinusI(Lanes2 x) => new(Vector256.Xor(Swap(x.value), Vector256.Create(0.0, -0.0, 0.0, -0.0)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes2 Conjugate(Lanes2 x) => new(Vector256.Xor(x.value, Vector256.Create(0.0, -0.0, 0.0, -0.0)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes2 Reverse(Lanes2 x) => new(Vector256.Create(x.value.GetUpper(), x.value.GetLower()));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<double> Swap(Vector256<double> v) => Vector256.Shuffle(v, Vector256.Create(1L, 0, 3, 2));
}

internal readonly struct Lanes4(Vector512<double> value) : IComplexLanes<Lanes4>
{
    private readonly Vector512<double> value = value;

    public static int Count => 4;

    public static bool IsAccelerated => Vector512.IsHardwareAccelerated;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes4 Load(ref readonly Complex source) =>
        new(Vector512.LoadUnsafe(ref Unsafe.As<Complex, double>(ref Unsafe.AsRef(in source))));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store(ref Complex destination, Lanes4 value) =>
        value.value.StoreUnsafe(ref Unsafe.As<Complex, double>(ref destination));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void StoreLanes(ref Complex destination, nint stride, Lanes4 value)
    {
        var lower = value.value.GetLower();
        var upper = value.value.GetUpper();
        lower.GetLower().StoreUnsafe(ref Unsafe.As<Complex, double>(ref destination));
        lower.GetUpper().StoreUnsafe(ref Unsafe.As<Complex, double>(ref Unsafe.Add(ref destination, stride)));
        upper.GetLower().StoreUnsafe(ref Unsafe.As<Complex, double>(ref Unsafe.Add(ref destination, 2 * stride)));
        upper.GetUpper().StoreUnsafe(ref Unsafe.As<Complex, double>(ref Unsafe.Add(ref destination, 3 * stride)));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes4 Add(Lanes4 x, Lanes4 y) => new(x.value + y.value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes4 Subtract(Lanes4 x, Lanes4 y) => new(x.value - y.value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes4 Scale(Lanes4 x, double factor) => new(x.value * Vector512.Create(factor));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes4 Broadcast(double value) => new(Vector512.Create(value));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes4 Product(Lanes4 x, Lanes4 y) => new(x.value * y.value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes4 Multiply(Lanes4 x, Complex w) =>
        new(x.value * Vector512.Create(w.Real) + MultiplyI(x).value * Vector512.Create(w.Imaginary));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes4 MultiplyConjugate(Lanes4 x, Complex w) =>
        new(x.value * Vector512.Create(w.Real) + MultiplyMinusI(x).value * Vector512.Create(w.Imaginary));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes4 MultiplyEach(Lanes4 x, Lanes4 w) =>
        new(x.value * Vector512.Shuffle(w.value, Vector512.Create(0L, 0, 2, 2, 4, 4, 6, 6)) + MultiplyI(x).value * Vector512.Shuffle(w.value, Vector512.Create(1L, 1, 3, 3, 5, 5, 7, 7)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes4 MultiplyEachConjugate(Lanes4 x, Lanes4 w) =>
        new(x.value * Vector512.Shuffle(w.value, Vector512.Create(0L, 0, 2, 2, 4, 4, 6, 6)) + MultiplyMinusI(x).value * Vector512.Shuffle(w.value, Vector512.Create(1L, 1, 3, 3, 5, 5, 7, 7)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes4 MultiplyI(Lanes4 x) => new(Vector512.Xor(Swap(x.value), Vector512.Create(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes4 MultiplyMinusI(Lanes4 x) => new(Vector512.Xor(Swap(x.value), Vector512.Create(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes4 Conjugate(Lanes4 x) => new(Vector512.Xor(x.value, Vector512.Create(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes4 Reverse(Lanes4 x) => new(Vector512.Shuffle(x.value, Vector512.Create(6L, 7, 4, 5, 2, 3, 0, 1)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<double> Swap(Vector512<double> v) => Vector512.Shuffle(v, Vector512.Create(1L, 0, 3, 2, 5, 4, 7, 6));
}
