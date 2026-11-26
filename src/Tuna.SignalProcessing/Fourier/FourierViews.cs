using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Tuna.SignalProcessing;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

internal static class FourierViews
{
    private const long BatchBytes = 512 * 1024;

    private const double RowWork = 8;

    private const int BlockRows = 4;

    internal static ReadOnlyMatrixSpan<T> AsRow<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, INumberBase<T> =>
        new(in x.Reference, x.Length == 0 ? 0 : 1, x.Length, 0, x.Stride);

    internal static MatrixSpan<T> AsRow<T>(VectorSpan<T> x) where T : unmanaged, INumberBase<T>
    {
        var view = x.AsReadOnly();
        return MatrixSpan<T>.Writable(AsRow(view));
    }

    internal static void ComplexRows<TD>(FourierPlan plan, ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination, double scale)
        where TD : IFourierDirection
    {
        RequireShape(x.Rows, x.Columns, destination.Rows, destination.Columns);
        RequireLength(plan.Length, x.Columns);
        var output = destination.AsReadOnly();
        if (StridedLayout.Overlaps(
                in x.Reference, x.Rows, x.Columns, x.RowStride, x.ColumnStride,
                in output.Reference, output.Rows, output.Columns, output.RowStride, output.ColumnStride, out bool exact) && !exact)
        {
            throw new ArgumentException("The destination partially overlaps the input. Only an exact alias is allowed.", nameof(destination));
        }

        int rows = x.Rows;
        int n = x.Columns;
        if (rows == 0 || n == 0)
        {
            return;
        }

        if (x.TryGetDense(out var xs) && destination.TryGetDense(out var ds))
        {
            Complex[] work = Rent(plan.WorkLength(rows));
            plan.Execute<TD>(ref MemoryMarshal.GetReference(Writable(xs)), ref MemoryMarshal.GetReference(ds), work, rows);
            LaneLoops.Scale(ds, scale);
            ArrayPool<Complex>.Shared.Return(work);
            return;
        }

        int block = BlockSize(rows, (long)n * Unsafe.SizeOf<Complex>(), out bool interleaved);
        bool single = IsRowContiguous(x.Columns, x.ColumnStride) && IsRowContiguous(output.Columns, output.ColumnStride);
        int threads = RowThreads(rows, n, single ? 1 : block);
        if (threads > 1)
        {
            ComplexRowsParallel<TD>(plan, x, destination, scale, single, block, interleaved, threads);
            return;
        }

        ComplexRowRange<TD>(plan, x, destination, 0, rows, scale, single, block, interleaved);
    }

    private static void ComplexRowsParallel<TD>(
        FourierPlan plan, ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination, double scale, bool single, int block, bool interleaved, int threads)
        where TD : IFourierDirection =>
        ParallelRows(x, destination, threads, single ? 1 : block, (xv, dv, first, count) =>
            ComplexRowRange<TD>(plan, xv, dv, first, count, scale, single, block, interleaved));

    private static void ComplexRowRange<TD>(
        FourierPlan plan, ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination, int first, int count, double scale, bool single, int block, bool interleaved)
        where TD : IFourierDirection
    {
        var output = destination.AsReadOnly();
        int n = x.Columns;
        if (single)
        {
            Complex[] work = Rent(plan.WorkLength(1));
            for (int i = first; i < first + count; i++)
            {
                ref Complex source = ref Unsafe.AsRef(in x.ElementAt(i, 0));
                ref Complex target = ref Unsafe.AsRef(in output.ElementAt(i, 0));
                plan.Execute<TD>(ref source, ref target, work, 1);
                LaneLoops.Scale(MemoryMarshal.CreateSpan(ref target, n), scale);
            }

            ArrayPool<Complex>.Shared.Return(work);
            return;
        }

        Complex[] buffer = Rent(checked(block * n));
        Complex[] workBuffer = Rent(plan.WorkLength(interleaved ? block : 1));
        for (int start = first; start < first + count; start += block)
        {
            int rows = Math.Min(block, first + count - start);
            Gather(x, start, rows, buffer, interleaved);
            if (interleaved)
            {
                plan.Execute<TD>(ref buffer[0], ref buffer[0], workBuffer, rows);
            }
            else
            {
                for (int i = 0; i < rows; i++)
                {
                    plan.Execute<TD>(ref buffer[i * n], ref buffer[i * n], workBuffer, 1);
                }
            }

            LaneLoops.Scale(buffer.AsSpan(0, rows * n), scale);
            Scatter<Complex>(buffer, start, rows, destination, interleaved);
        }

        ArrayPool<Complex>.Shared.Return(workBuffer);
        ArrayPool<Complex>.Shared.Return(buffer);
    }

    internal static void RealForwardRows(RealFourierPlan plan, ReadOnlyMatrixSpan<double> x, MatrixSpan<Complex> destination, double scale)
    {
        RequireShape(x.Rows, plan.SpectrumLength, destination.Rows, destination.Columns);
        RequireLength(plan.Length, x.Columns);
        var output = destination.AsReadOnly();
        RequireDisjoint(in x.Reference, x.Extent, in output.Reference, output.Extent);
        int rows = x.Rows;
        int n = x.Columns;
        int spectrum = plan.SpectrumLength;
        if (rows == 0 || n == 0)
        {
            return;
        }

        if (x.TryGetDense(out var xs) && destination.TryGetDense(out var ds))
        {
            Complex[] work = Rent(plan.ForwardWorkLength(rows));
            plan.ForwardCore(ref MemoryMarshal.GetReference(Writable(xs)), ref MemoryMarshal.GetReference(ds), work, rows);
            LaneLoops.Scale(ds, scale);
            ArrayPool<Complex>.Shared.Return(work);
            return;
        }

        int block = BlockSize(rows, (long)spectrum * Unsafe.SizeOf<Complex>(), out bool interleaved);
        bool single = IsRowContiguous(x.Columns, x.ColumnStride) && IsRowContiguous(output.Columns, output.ColumnStride);
        int threads = RowThreads(rows, n, single ? 1 : block);
        if (threads > 1)
        {
            RealForwardRowsParallel(plan, x, destination, scale, single, block, interleaved, threads);
            return;
        }

        RealForwardRowRange(plan, x, destination, 0, rows, scale, single, block, interleaved);
    }

    private static void RealForwardRowsParallel(
        RealFourierPlan plan, ReadOnlyMatrixSpan<double> x, MatrixSpan<Complex> destination, double scale, bool single, int block, bool interleaved, int threads) =>
        ParallelRows(x, destination, threads, single ? 1 : block, (xv, dv, first, count) =>
            RealForwardRowRange(plan, xv, dv, first, count, scale, single, block, interleaved));

    private static void RealForwardRowRange(
        RealFourierPlan plan, ReadOnlyMatrixSpan<double> x, MatrixSpan<Complex> destination, int first, int count, double scale, bool single, int block, bool interleaved)
    {
        var output = destination.AsReadOnly();
        int n = x.Columns;
        int spectrum = plan.SpectrumLength;
        if (single)
        {
            Complex[] work = Rent(plan.ForwardWorkLength(1));
            for (int i = first; i < first + count; i++)
            {
                ref Complex target = ref Unsafe.AsRef(in output.ElementAt(i, 0));
                plan.ForwardCore(ref Unsafe.AsRef(in x.ElementAt(i, 0)), ref target, work, 1);
                LaneLoops.Scale(MemoryMarshal.CreateSpan(ref target, spectrum), scale);
            }

            ArrayPool<Complex>.Shared.Return(work);
            return;
        }

        double[] gathered = ArrayPool<double>.Shared.Rent(checked(block * n));
        Complex[] result = Rent(checked(block * spectrum));
        Complex[] workBuffer = Rent(plan.ForwardWorkLength(interleaved ? block : 1));
        for (int start = first; start < first + count; start += block)
        {
            int rows = Math.Min(block, first + count - start);
            Gather(x, start, rows, gathered, interleaved);
            if (interleaved)
            {
                plan.ForwardCore(ref gathered[0], ref result[0], workBuffer, rows);
            }
            else
            {
                for (int i = 0; i < rows; i++)
                {
                    plan.ForwardCore(ref gathered[i * n], ref result[i * spectrum], workBuffer, 1);
                }
            }

            LaneLoops.Scale(result.AsSpan(0, rows * spectrum), scale);
            Scatter<Complex>(result, start, rows, destination, interleaved);
        }

        ArrayPool<Complex>.Shared.Return(workBuffer);
        ArrayPool<Complex>.Shared.Return(result);
        ArrayPool<double>.Shared.Return(gathered);
    }

    internal static void RealInverseRows(RealFourierPlan plan, ReadOnlyMatrixSpan<Complex> x, MatrixSpan<double> destination, double scale)
    {
        RequireShape(x.Rows, plan.Length, destination.Rows, destination.Columns);
        if (x.Columns != plan.SpectrumLength)
        {
            throw new ArgumentException(
                $"The spectrum must have {plan.SpectrumLength} elements (length / 2 + 1) for a signal of length {plan.Length}.", nameof(x));
        }

        var output = destination.AsReadOnly();
        RequireDisjoint(in x.Reference, x.Extent, in output.Reference, output.Extent);
        int rows = x.Rows;
        int n = plan.Length;
        int spectrum = x.Columns;
        if (rows == 0 || n == 0)
        {
            return;
        }

        if (x.TryGetDense(out var xs) && destination.TryGetDense(out var ds))
        {
            Complex[] work = Rent(plan.InverseWorkLength(rows));
            plan.InverseCore(ref MemoryMarshal.GetReference(Writable(xs)), ref MemoryMarshal.GetReference(ds), work, rows);
            LaneLoops.Scale(ds, scale);
            ArrayPool<Complex>.Shared.Return(work);
            return;
        }

        int block = BlockSize(rows, (long)spectrum * Unsafe.SizeOf<Complex>(), out bool interleaved);
        bool single = IsRowContiguous(x.Columns, x.ColumnStride) && IsRowContiguous(output.Columns, output.ColumnStride);
        int threads = RowThreads(rows, n, single ? 1 : block);
        if (threads > 1)
        {
            RealInverseRowsParallel(plan, x, destination, scale, single, block, interleaved, threads);
            return;
        }

        RealInverseRowRange(plan, x, destination, 0, rows, scale, single, block, interleaved);
    }

    private static void RealInverseRowsParallel(
        RealFourierPlan plan, ReadOnlyMatrixSpan<Complex> x, MatrixSpan<double> destination, double scale, bool single, int block, bool interleaved, int threads) =>
        ParallelRows(x, destination, threads, single ? 1 : block, (xv, dv, first, count) =>
            RealInverseRowRange(plan, xv, dv, first, count, scale, single, block, interleaved));

    private static void RealInverseRowRange(
        RealFourierPlan plan, ReadOnlyMatrixSpan<Complex> x, MatrixSpan<double> destination, int first, int count, double scale, bool single, int block, bool interleaved)
    {
        var output = destination.AsReadOnly();
        int n = plan.Length;
        int spectrum = x.Columns;
        if (single)
        {
            Complex[] work = Rent(plan.InverseWorkLength(1));
            for (int i = first; i < first + count; i++)
            {
                ref double target = ref Unsafe.AsRef(in output.ElementAt(i, 0));
                plan.InverseCore(ref Unsafe.AsRef(in x.ElementAt(i, 0)), ref target, work, 1);
                LaneLoops.Scale(MemoryMarshal.CreateSpan(ref target, n), scale);
            }

            ArrayPool<Complex>.Shared.Return(work);
            return;
        }

        Complex[] gathered = Rent(checked(block * spectrum));
        double[] result = ArrayPool<double>.Shared.Rent(checked(block * n));
        Complex[] workBuffer = Rent(plan.InverseWorkLength(interleaved ? block : 1));
        for (int start = first; start < first + count; start += block)
        {
            int rows = Math.Min(block, first + count - start);
            Gather(x, start, rows, gathered, interleaved);
            if (interleaved)
            {
                plan.InverseCore(ref gathered[0], ref result[0], workBuffer, rows);
            }
            else
            {
                for (int i = 0; i < rows; i++)
                {
                    plan.InverseCore(ref gathered[i * spectrum], ref result[i * n], workBuffer, 1);
                }
            }

            LaneLoops.Scale(result.AsSpan(0, rows * n), scale);
            Scatter<double>(result, start, rows, destination, interleaved);
        }

        ArrayPool<Complex>.Shared.Return(workBuffer);
        ArrayPool<double>.Shared.Return(result);
        ArrayPool<Complex>.Shared.Return(gathered);
    }

    internal static int RowThreads(int rows, int n, int unit)
    {
        int threads = Parallelism.Degree(RowWork * rows * n * Math.Log2(n + 1), Parallelism.Grain);
        return Math.Min(threads, (rows + unit - 1) / unit);
    }

    internal delegate void RowRange<TIn, TOut>(ReadOnlyMatrixSpan<TIn> x, MatrixSpan<TOut> destination, int first, int count)
        where TIn : unmanaged, INumberBase<TIn>
        where TOut : unmanaged, INumberBase<TOut>;

    internal static unsafe void ParallelRows<TIn, TOut>(ReadOnlyMatrixSpan<TIn> x, MatrixSpan<TOut> destination, int threads, int unit, RowRange<TIn, TOut> body)
        where TIn : unmanaged, INumberBase<TIn>
        where TOut : unmanaged, INumberBase<TOut>
    {
        var output = destination.AsReadOnly();
        int rows = x.Rows;
        int units = (rows + unit - 1) / unit;
        fixed (TIn* inputPointer = &Unsafe.AsRef(in x.Reference))
        fixed (TOut* outputPointer = &Unsafe.AsRef(in output.Reference))
        {
            nint source = (nint)inputPointer, target = (nint)outputPointer;
            int columns = x.Columns, rowStride = x.RowStride, columnStride = x.ColumnStride;
            int outputColumns = output.Columns, outputRowStride = output.RowStride, outputColumnStride = output.ColumnStride;
            Parallelism.ForStatic(threads, part =>
            {
                int first = Math.Min(rows, (int)((long)units * part / threads) * unit);
                int end = Math.Min(rows, (int)((long)units * (part + 1) / threads) * unit);
                if (first >= end)
                {
                    return;
                }

                var input = new ReadOnlyMatrixSpan<TIn>(in Unsafe.AsRef<TIn>((void*)source), rows, columns, rowStride, columnStride);
                var view = new ReadOnlyMatrixSpan<TOut>(in Unsafe.AsRef<TOut>((void*)target), rows, outputColumns, outputRowStride, outputColumnStride);
                body(input, MatrixSpan<TOut>.Writable(view), first, end - first);
            });
        }
    }

    private static int BlockSize(int rows, long rowBytes, out bool interleaved)
    {
        long fitting = BatchBytes / rowBytes / BlockRows * BlockRows;
        int block = rows <= fitting ? rows : (int)Math.Max(fitting, Math.Min(BlockRows, rows));
        interleaved = block * rowBytes <= BatchBytes;
        return block;
    }

    private static void Gather<T>(ReadOnlyMatrixSpan<T> x, int first, int count, Span<T> buffer, bool interleaved)
        where T : unmanaged, INumberBase<T>
    {
        int n = x.Columns;
        nint rowStride = x.RowStride;
        nint columnStride = x.ColumnStride;
        ref T origin = ref Unsafe.Add(ref Unsafe.AsRef(in x.Reference), first * rowStride);
        for (int j = 0; j < n; j++)
        {
            ref T column = ref Unsafe.Add(ref origin, j * columnStride);
            if (interleaved && (rowStride == 1 || count == 1))
            {
                MemoryMarshal.CreateReadOnlySpan(ref column, count).CopyTo(buffer.Slice(j * count, count));
                continue;
            }

            for (int i = 0; i < count; i++)
            {
                buffer[interleaved ? i + count * j : i * n + j] = Unsafe.Add(ref column, i * rowStride);
            }
        }
    }

    private static void Scatter<T>(ReadOnlySpan<T> buffer, int first, int count, MatrixSpan<T> destination, bool interleaved)
        where T : unmanaged, INumberBase<T>
    {
        var output = destination.AsReadOnly();
        int n = output.Columns;
        nint rowStride = output.RowStride;
        nint columnStride = output.ColumnStride;
        ref T origin = ref Unsafe.Add(ref Unsafe.AsRef(in output.Reference), first * rowStride);
        for (int j = 0; j < n; j++)
        {
            ref T column = ref Unsafe.Add(ref origin, j * columnStride);
            if (interleaved && (rowStride == 1 || count == 1))
            {
                buffer.Slice(j * count, count).CopyTo(MemoryMarshal.CreateSpan(ref column, count));
                continue;
            }

            for (int i = 0; i < count; i++)
            {
                Unsafe.Add(ref column, i * rowStride) = buffer[interleaved ? i + count * j : i * n + j];
            }
        }
    }

    private static bool IsRowContiguous(int columns, int columnStride) => columns <= 1 || columnStride == 1;

    private static Span<T> Writable<T>(ReadOnlySpan<T> span) =>
        MemoryMarshal.CreateSpan(ref MemoryMarshal.GetReference(span), span.Length);

    private static Complex[] Rent(int length) => ArrayPool<Complex>.Shared.Rent(Math.Max(length, 1));

    private static void RequireShape(int rows, int columns, int destinationRows, int destinationColumns)
    {
        if (rows != destinationRows || columns != destinationColumns)
        {
            throw new ArgumentException(
                $"The destination must be {rows} x {columns}, but it is {destinationRows} x {destinationColumns}.", "destination");
        }
    }

    private static void RequireLength(int planLength, int length)
    {
        if (planLength != length)
        {
            throw new ArgumentException($"The plan is for length {planLength}, but the input has length {length}.", "x");
        }
    }

    private static void RequireDisjoint<TInput, TOutput>(ref readonly TInput input, int inputExtent, ref readonly TOutput output, int outputExtent)
    {
        if (inputExtent == 0 || outputExtent == 0)
        {
            return;
        }

        nint offset = Unsafe.ByteOffset(
            ref Unsafe.As<TInput, byte>(ref Unsafe.AsRef(in input)), ref Unsafe.As<TOutput, byte>(ref Unsafe.AsRef(in output)));
        nint inputBytes = (nint)inputExtent * Unsafe.SizeOf<TInput>();
        nint outputBytes = (nint)outputExtent * Unsafe.SizeOf<TOutput>();

        // output 先頭が input 先頭から offset バイト離れているため、[-outputBytes, inputBytes) なら重なる。
        if (offset < inputBytes && -offset < outputBytes)
        {
            throw new ArgumentException("The destination must not overlap the input.", "destination");
        }
    }
}
