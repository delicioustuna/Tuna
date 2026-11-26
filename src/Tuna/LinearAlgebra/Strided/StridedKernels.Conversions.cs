using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Tuna.LinearAlgebra;

// 要素型を変換する単項演算子（Complex -> double など）。
internal interface IConvertingOperator<TSource, TResult>
{
    TResult Invoke(TSource x);
}

internal readonly struct RealPartOperator : IConvertingOperator<Complex, double>
{
    public double Invoke(Complex x) => x.Real;
}

internal readonly struct ImaginaryPartOperator : IConvertingOperator<Complex, double>
{
    public double Invoke(Complex x) => x.Imaginary;
}

internal readonly struct MagnitudeOperator : IConvertingOperator<Complex, double>
{
    public double Invoke(Complex x) => x.Magnitude;
}

internal readonly struct PhaseOperator : IConvertingOperator<Complex, double>
{
    public double Invoke(Complex x) => x.Phase;
}

internal static partial class StridedKernels
{
    // 要素型が異なる要素ごと演算（インプレース不可のため領域共有はすべて例外）。
    internal static void MapTo<TSource, TResult, TOperator>(ReadOnlyMatrixSpan<TSource> x, MatrixSpan<TResult> destination, TOperator op)
        where TSource : unmanaged, INumberBase<TSource>
        where TResult : unmanaged, INumberBase<TResult>
        where TOperator : IConvertingOperator<TSource, TResult>
    {
        if (x.Rows != destination.Rows || x.Columns != destination.Columns)
        {
            throw new ArgumentException("Matrix dimensions must agree.");
        }

        var output = destination.AsReadOnly();
        if (StridedLayout.OverlapsAcrossTypes(
            in x.Reference, x.Rows, x.Columns, x.RowStride, x.ColumnStride,
            in output.Reference, output.Rows, output.Columns, output.RowStride, output.ColumnStride))
        {
            throw new ArgumentException("The destination must not overlap the input.", nameof(destination));
        }

        if (x.TryGetDense(out var xs) && destination.TryGetDense(out var ds))
        {
            if (typeof(TOperator) == typeof(MagnitudeOperator))
            {
                DenseElementwise.Magnitude(MemoryMarshal.Cast<TSource, Complex>(xs), MemoryMarshal.Cast<TResult, double>(ds));
                return;
            }

            for (int i = 0; i < ds.Length; i++)
            {
                ds[i] = op.Invoke(xs[i]);
            }

            return;
        }

        // Magnitude は列連続なら列単位で SIMD 実装を使う（Complex.Magnitude とビット単位で一致）
        if (typeof(TOperator) == typeof(MagnitudeOperator) && x.Rows > 1 && x.RowStride == 1 && output.RowStride == 1)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                DenseElementwise.Magnitude(
                    MemoryMarshal.Cast<TSource, Complex>(MemoryMarshal.CreateReadOnlySpan(in x.ElementAt(0, j), x.Rows)),
                    MemoryMarshal.Cast<TResult, double>(MemoryMarshal.CreateSpan(ref Unsafe.AsRef(in output.ElementAt(0, j)), x.Rows)));
            }

            return;
        }

        for (int j = 0; j < x.Columns; j++)
        {
            for (int i = 0; i < x.Rows; i++)
            {
                Unsafe.AsRef(in output.ElementAt(i, j)) = op.Invoke(x.ElementAt(i, j));
            }
        }
    }
}
