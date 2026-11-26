using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Tuna.LinearAlgebra;

// 要素型ごとのベクトル演算ディスパッチ（Complex は ComplexVectorOps、それ以外は RealVectorOps）。
internal static class VectorOps
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ReadOnlySpan<Complex> AsComplex<T>(ReadOnlySpan<T> x) =>
        MemoryMarshal.CreateReadOnlySpan(ref Unsafe.As<T, Complex>(ref MemoryMarshal.GetReference(x)), x.Length);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Span<Complex> AsComplex<T>(Span<T> x) =>
        MemoryMarshal.CreateSpan(ref Unsafe.As<T, Complex>(ref MemoryMarshal.GetReference(x)), x.Length);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Complex ToComplex<T>(T v) => Unsafe.As<T, Complex>(ref v);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T FromComplex<T>(Complex v) => Unsafe.As<Complex, T>(ref v);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static T Conjugate<T>(T v)
        where T : unmanaged, INumberBase<T> =>
        typeof(T) == typeof(Complex) ? FromComplex<T>(Complex.Conjugate(ToComplex(v))) : v;

    // 複素数は虚部を 0 にする（エルミート行列の対角成分用）。
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static T RealPart<T>(T v)
        where T : unmanaged, INumberBase<T> =>
        typeof(T) == typeof(Complex) ? FromComplex<T>(new Complex(ToComplex(v).Real, 0)) : v;

    // Σ op(x[i]) * y[i]（conjugateX は実数型では無視）。
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static T Dot<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> y, bool conjugateX = false)
        where T : unmanaged, INumberBase<T> =>
        typeof(T) == typeof(Complex)
            ? FromComplex<T>(ComplexVectorOps.Dot(AsComplex(x), AsComplex(y), conjugateX))
            : RealVectorOps.Dot(x, y);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Axpy<T>(T a, ReadOnlySpan<T> x, Span<T> y)
        where T : unmanaged, INumberBase<T>
    {
        if (typeof(T) == typeof(Complex))
        {
            ComplexVectorOps.Axpy(ToComplex(a), AsComplex(x), AsComplex(y));
        }
        else
        {
            RealVectorOps.Axpy(a, x, y);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Scale<T>(T a, Span<T> x)
        where T : unmanaged, INumberBase<T>
    {
        if (typeof(T) == typeof(Complex))
        {
            ComplexVectorOps.Scale(ToComplex(a), AsComplex(x));
        }
        else
        {
            RealVectorOps.Scale(a, x);
        }
    }

    // y += a * col と Σ op(col[i]) * x[i] の融合計算。
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static T AxpyDot<T>(T a, ReadOnlySpan<T> col, ReadOnlySpan<T> x, Span<T> y, bool conjugateColumn)
        where T : unmanaged, INumberBase<T> =>
        typeof(T) == typeof(Complex)
            ? FromComplex<T>(ComplexVectorOps.AxpyDot(ToComplex(a), AsComplex(col), AsComplex(x), AsComplex(y), conjugateColumn))
            : RealVectorOps.AxpyDot(a, col, x, y);
}
