using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

internal sealed partial class ArrayStore<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    private const byte RequiresGradientFlag = 1;
    private const byte TouchedFlag = 2;
    private const byte ExternalGradientFlag = 4;
    private const byte PooledFlag = 8;
    // Allows alternating between two distinct shapes (up to 2x peak) plus slack for small buffers without reallocation.
    private const int RetentionFactor = 3;

    private object?[] sources = [];
    private T[]?[] gradients = [];
    private T[]?[] cachedValues = [];
    private T[]?[] cachedGradients = [];
    private int[] rows = [];
    private int[] columns = [];
    private byte[] flags = [];
    private int count;
    private Dictionary<int, SpareArrays>? spares;
    private readonly Stack<SpareArrays> emptySpares = new();
    private SpareArrays[] trimOrder = [];
    // Retained at peak length and excluded from RetentionFactor because broadcast backward passes can expand n to n^2.
    private readonly T[][] scratch = [[], [], [], [], [], [], [], [], []];
    private DecompositionPool? decompositions;
    private long borrowed;
    private long peakBorrowed;
    private long retained;
    private long epoch;
    private int cachedCount;
    private bool reviewRetention;

    public void Reset()
    {
        // Clear live references so external containers and Parameter<T> buffers are not held past Reset; pooled buffers stay in cachedValues/cachedGradients.
        Array.Clear(sources, 0, count);
        Array.Clear(gradients, 0, count);
        peakBorrowed = Math.Max(peakBorrowed, borrowed);
        borrowed = 0;
        if (count > cachedCount)
        {
            cachedCount = count;
        }
        else if (count < cachedCount)
        {
            reviewRetention = true;
        }

        // Skip scanning on steady-state replays; flags still reflect the just-finished recording.
        if (reviewRetention)
        {
            ReleaseUnused();
        }

        decompositions?.Reset();
        count = 0;
        epoch++;
        ResetOperations();
    }

    public DecompositionPool Decompositions => decompositions ??= new();

    public bool RequiresGradient(int slot) => (flags[slot] & RequiresGradientFlag) != 0;

    public bool Touched(int slot) => (flags[slot] & TouchedFlag) != 0;

    public int AddPooled(int rowCount, int columnCount, bool requiresGradient)
    {
        // Validate element count before adding a slot so overflow leaves the store unmodified.
        int length = checked(rowCount * columnCount);
        int slot = AddSlot(rowCount, columnCount, requiresGradient ? (byte)(PooledFlag | RequiresGradientFlag) : PooledFlag);
        sources[slot] = Borrow(ref cachedValues[slot], length);
        return slot;
    }

    public int AddConstant(object container, int rowCount, int columnCount)
    {
        int slot = AddSlot(rowCount, columnCount, 0);
        NoteUnpooled(slot);
        sources[slot] = container;
        return slot;
    }

    public int AddFactor(Matrix<T> factor, bool requiresGradient) => AddFactor(factor, factor.Rows, factor.Columns, requiresGradient);

    public int AddFactor(T[] values, bool requiresGradient) => AddFactor(values, values.Length, 1, requiresGradient);

    // References decomposition factor storage directly; keeps cachedGradients for same-length reuse across recordings.
    private int AddFactor(object factor, int rowCount, int columnCount, bool requiresGradient)
    {
        int slot = AddSlot(rowCount, columnCount, requiresGradient ? RequiresGradientFlag : (byte)0);
        if (cachedValues[slot] is not null)
        {
            reviewRetention = true;
        }

        sources[slot] = factor;
        return slot;
    }

    public int AddParameter(T[] value, T[] gradient, int rowCount, int columnCount)
    {
        int slot = AddSlot(rowCount, columnCount, RequiresGradientFlag | ExternalGradientFlag);
        NoteUnpooled(slot);
        sources[slot] = value;
        gradients[slot] = gradient;
        return slot;
    }

    public Span<T> Values(int slot)
    {
        object source = sources[slot]!;
        if (source.GetType() == typeof(T[]))
        {
            return Unsafe.As<T[]>(source);
        }

        if (source is Matrix<T> matrix)
        {
            return matrix.AsSpan();
        }

        return source is LinearAlgebra.Vector<T> vector ? vector.AsSpan() : Unsafe.As<Tensor<T>>(source).AsSpan();
    }

    public ReadOnlyMatrixSpan<T> MatrixView(scoped in ArrayView view) => MatrixOf(Values(view.Slot), view);

    public ReadOnlyVectorSpan<T> VectorView(scoped in ArrayView view) => new(Values(view.Slot).Slice(Offset(view), view.RowCount));

    public MatrixSpan<T> WritableMatrix(int slot) => new(Values(slot), rows[slot], columns[slot]);

    public VectorSpan<T> WritableVector(int slot) => new(Values(slot));

    public ReadOnlyMatrixSpan<T> GradientMatrix(scoped in ArrayView view) => MatrixOf(ReadableGradient(view.Slot), view);

    public ReadOnlyVectorSpan<T> GradientVector(scoped in ArrayView view) => new(ReadableGradient(view.Slot).Slice(Offset(view), view.RowCount));

    public int CopyGradient(int slot, bool transposed, VectorSpan<double> destination, int start, int columnsPerMatrix)
    {
        ReadOnlySpan<T> gradient = ReadableGradient(slot);
        if (!transposed)
        {
            for (int i = 0; i < gradient.Length; i++)
            {
                destination[start + i] = double.CreateTruncating(gradient[i]);
            }

            return gradient.Length;
        }

        if (gradient.IsEmpty)
        {
            return 0;
        }

        // Each leaf matrix is copied in logical column order, including tensors stored as consecutive transposed matrices.
        int leafRows = columnsPerMatrix == 0 ? columns[slot] : columnsPerMatrix;
        int leafColumns = rows[slot];
        int matrixLength = leafRows * leafColumns;
        for (int offset = 0; offset < gradient.Length; offset += matrixLength)
        {
            for (int j = 0; j < leafColumns; j++)
            {
                for (int i = 0; i < leafRows; i++)
                {
                    destination[start++] = double.CreateTruncating(gradient[offset + j + i * leafColumns]);
                }
            }
        }

        return gradient.Length;
    }

    public int Length(int slot) => rows[slot] * columns[slot];

    public Span<T> OutputGradient(int slot) => gradients[slot].AsSpan(0, Length(slot));

    public void Backward(in ArrayRecord record, double[] adjoints)
    {
        switch (record.Operation)
        {
            case ArrayOperation.MatrixMultiply:
                MatrixMultiplyBackward(record);
                break;
            case ArrayOperation.MatrixVectorMultiply:
                MatrixVectorMultiplyBackward(record);
                break;
            case ArrayOperation.Sum:
                SumBackward(record, adjoints[record.Output]);
                break;
            case ArrayOperation.Dot:
                DotBackward(record, adjoints[record.Output]);
                break;
            case ArrayOperation.UserOperation:
                UserOperationBackward(record);
                break;
            case ArrayOperation.Embedding:
                EmbeddingBackward(record);
                break;
            default:
                ElementwiseBackward(record, adjoints);
                break;
        }
    }

    private void MatrixMultiplyBackward(in ArrayRecord record)
    {
        int output = record.Output;
        if (!Touched(output))
        {
            return;
        }

        var c = ArrayView.Whole(output, rows[output], columns[output]);
        var dc = Operand(gradients[output]!, c);
        ArrayView a = record.Left, b = record.Right;
        if (RequiresGradient(a.Slot))
        {
            AddProduct(a, dc, ValueOperand(b, RightValues).Transpose(), b.Columns);
        }

        if (RequiresGradient(b.Slot))
        {
            AddProduct(b, ValueOperand(a, LeftValues).Transpose(), dc, a.Rows);
        }
    }

    private void MatrixVectorMultiplyBackward(in ArrayRecord record)
    {
        int output = record.Output;
        if (!Touched(output))
        {
            return;
        }

        ReadOnlySpan<T> dy = gradients[output].AsSpan(0, rows[output]);
        ArrayView a = record.Left, x = record.Right;
        if (RequiresGradient(a.Slot))
        {
            ReadOnlySpan<T> xValues = Values(x.Slot).Slice(Offset(x), x.RowCount);
            if (a.Broadcast)
            {
                Span<T> product = Scratch(Contribution, a.Length);
                product.Clear();
                AddOuterProduct(a.Rows, a.Columns, dy, xValues, product, Math.Max(1, a.Rows));
                AddReduced(a, BeginGradient(a, out bool reduce), product, reduce, negate: false);
            }
            else
            {
                Span<T> target = BeginGradient(a, out bool overwrite).Slice(Offset(a));
                if (overwrite)
                {
                    target.Clear();
                }

                ReadOnlySpan<T> left = a.Transposed ? xValues : dy, right = a.Transposed ? dy : xValues;
                AddOuterProduct(a.RowCount, a.ColumnCount, left, right, target, LeadingDimension(a.Slot));
            }
        }

        if (RequiresGradient(x.Slot))
        {
            // Materialize broadcast views into column-major scratch because BLAS does not accept zero column stride.
            ReadOnlySpan<T> aValues;
            int m, n, ld;
            bool transposed;
            if (a.Broadcast)
            {
                aValues = Materialize(a, LeftValues);
                (m, n, ld, transposed) = (a.Rows, a.Columns, Math.Max(1, a.Rows), false);
            }
            else
            {
                aValues = Values(a.Slot).Slice(Offset(a));
                (m, n, ld, transposed) = (a.RowCount, a.ColumnCount, LeadingDimension(a.Slot), a.Transposed);
            }

            Span<T> target = BeginGradient(x, out bool overwrite).Slice(Offset(x), x.RowCount);
            // Detect zero multipliers before non-transposed Gemv, which may skip those columns and lose 0 * non-finite = NaN.
            int skipped = transposed && m > 0 ? NextZero(dy.Slice(0, n), 0) : n;
            Gemv(!transposed, m, n, aValues, ld, dy, overwrite, target);
            if (skipped < n)
            {
                AddSkippedColumns(skipped, m, n, aValues, ld, dy, target);
            }
        }
    }

    private static void AddOuterProduct(int m, int n, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> a, int lda)
    {
        int skipped = FirstSkippedOuterColumn(m, n, x, y);
        Ger(m, n, x, y, a, lda);
        if (skipped < n)
        {
            AddSkippedOuterColumns(skipped, m, n, x, y, a, lda);
        }
    }

    private ReadOnlySpan<T> Materialize(scoped in ArrayView view, int index)
    {
        Span<T> copy = Scratch(index, view.Length);
        MatrixView(view).CopyTo(new MatrixSpan<T>(copy, view.Rows, view.Columns));
        return copy;
    }

    // Materializes broadcast views into scratch to avoid allocating a temporary copy inside core triangular solves.
    public ReadOnlyMatrixSpan<T> CoefficientView(scoped in ArrayView view, int index, bool transpose)
    {
        ReadOnlyMatrixSpan<T> matrix = transpose ? MatrixView(view).T : MatrixView(view);
        if (!view.Broadcast)
        {
            return matrix;
        }

        var copy = new MatrixSpan<T>(Scratch(index, view.Length), matrix.Rows, matrix.Columns);
        matrix.CopyTo(copy);
        return copy.AsReadOnly();
    }

    private BlasOperand ValueOperand(scoped in ArrayView view, int index)
        => view.Broadcast ? new(Materialize(view, index), false, Math.Max(1, view.Rows)) : Operand(Values(view.Slot), view);

    private void SumBackward(in ArrayRecord record, double adjoint)
    {
        T g = FromDouble(adjoint);
        ArrayView x = record.Left;
        if (T.IsZero(g) || !RequiresGradient(x.Slot))
        {
            return;
        }

        Span<T> target = BeginGradient(x, out bool overwrite).Slice(Offset(x), x.RowCount);
        if (overwrite)
        {
            target.Fill(g);
        }
        else
        {
            XMath.Add(new ReadOnlyVectorSpan<T>(target), g, new VectorSpan<T>(target));
        }
    }

    private void DotBackward(in ArrayRecord record, double adjoint)
    {
        T g = FromDouble(adjoint);
        if (T.IsZero(g))
        {
            return;
        }

        ArrayView x = record.Left, y = record.Right;
        if (RequiresGradient(x.Slot))
        {
            AddScaled(x, g, Values(y.Slot).Slice(Offset(y), y.RowCount));
        }

        if (RequiresGradient(y.Slot))
        {
            AddScaled(y, g, Values(x.Slot).Slice(Offset(x), x.RowCount));
        }
    }

    // Uses a dedicated non-FMA kernel instead of Blas.Axpy for ISA bit-determinism and stable JIT codegen.
    private void AddScaled(in ArrayView target, T g, ReadOnlySpan<T> source)
    {
        Span<T> destination = BeginGradient(target, out bool overwrite).Slice(Offset(target), target.RowCount);
        ElementwiseKernels<T>.AddScaled(source, g, destination, overwrite);
    }

    // Avoids ref indexing into T[][] to bypass per-call array-variance checks.
    public Span<T> Scratch(int index, int length)
    {
        T[] buffer = scratch[index];
        if (buffer.Length < length)
        {
            buffer = GC.AllocateUninitializedArray<T>(length);
            scratch[index] = buffer;
        }

        return buffer.AsSpan(0, length);
    }

    private void AddProduct(in ArrayView target, BlasOperand x, BlasOperand y, int k)
    {
        if (target.Broadcast)
        {
            Span<T> product = Scratch(Contribution, target.Length);
            Gemm(x, y, target.Rows, target.Columns, k, overwrite: true, product, Math.Max(1, target.Rows));
            AddReduced(target, BeginGradient(target, out bool reduce), product, reduce, negate: false);
            return;
        }

        Span<T> destination = BeginGradient(target, out bool overwrite).Slice(Offset(target));
        int ld = LeadingDimension(target.Slot);
        if (target.Transposed)
        {
            Gemm(y.Transpose(), x.Transpose(), target.RowCount, target.ColumnCount, k, overwrite, destination, ld);
        }
        else
        {
            Gemm(x, y, target.RowCount, target.ColumnCount, k, overwrite, destination, ld);
        }
    }

    // Sets overwrite = true only on the first contribution covering an entire pooled slot; Parameter<T> gradients always accumulate.
    private Span<T> BeginGradient(scoped in ArrayView view, out bool overwrite)
    {
        int slot = view.Slot;
        byte flag = flags[slot];
        overwrite = false;
        if ((flag & TouchedFlag) != 0)
        {
            return gradients[slot];
        }

        flags[slot] = (byte)(flag | TouchedFlag);
        if ((flag & ExternalGradientFlag) != 0)
        {
            return gradients[slot];
        }

        T[] gradient = Borrow(ref cachedGradients[slot], Length(slot));
        gradients[slot] = gradient;
        if (view.StorageLength == gradient.Length)
        {
            overwrite = true;
        }
        else
        {
            gradient.AsSpan().Clear();
        }

        return gradient;
    }

    private Span<T> ReadableGradient(int slot)
    {
        if ((flags[slot] & (TouchedFlag | ExternalGradientFlag)) == 0)
        {
            T[] gradient = Borrow(ref cachedGradients[slot], Length(slot));
            gradient.AsSpan().Clear();
            gradients[slot] = gradient;
            flags[slot] |= TouchedFlag;
        }

        return gradients[slot];
    }

    private int AddSlot(int rowCount, int columnCount, byte flag)
    {
        int slot = count;
        if (slot == flags.Length)
        {
            Grow();
        }

        rows[slot] = rowCount;
        columns[slot] = columnCount;
        flags[slot] = flag;
        count = slot + 1;
        return slot;
    }

    private void Grow()
    {
        int capacity = (int)Math.Min(Array.MaxLength, Math.Max(16L, 2L * flags.Length));
        if (capacity == flags.Length)
        {
            throw new OutOfMemoryException("Too many array nodes in one tape.");
        }

        // Resize into locals before committing so OOM leaves all parallel arrays with matching lengths.
        var newSources = sources;
        var newGradients = gradients;
        var newCachedValues = cachedValues;
        var newCachedGradients = cachedGradients;
        var newRows = rows;
        var newColumns = columns;
        var newFlags = flags;
        Array.Resize(ref newSources, capacity);
        Array.Resize(ref newGradients, capacity);
        Array.Resize(ref newCachedValues, capacity);
        Array.Resize(ref newCachedGradients, capacity);
        Array.Resize(ref newRows, capacity);
        Array.Resize(ref newColumns, capacity);
        Array.Resize(ref newFlags, capacity);
        sources = newSources;
        gradients = newGradients;
        cachedValues = newCachedValues;
        cachedGradients = newCachedGradients;
        rows = newRows;
        columns = newColumns;
        flags = newFlags;
    }

    private T[] Borrow(ref T[]? cached, int length)
    {
        borrowed += length;
        T[]? array = cached;
        if (array is not null && array.Length == length)
        {
            return array;
        }

        if (array is not null)
        {
            Return(array);
        }

        array = Rent(length);
        cached = array;
        return array;
    }

    private T[] Rent(int length)
    {
        if (length == 0)
        {
            return [];
        }

        if (spares is not null && spares.TryGetValue(length, out var spare) && spare.Arrays.TryPop(out var array))
        {
            spare.LastUse = epoch;
            return array;
        }

        reviewRetention = true;
        retained += length;
        return GC.AllocateUninitializedArray<T>(length);
    }

    private void Return(T[] array)
    {
        if (array.Length == 0)
        {
            return;
        }

        reviewRetention = true;
        spares ??= new();
        if (!spares.TryGetValue(array.Length, out var spare))
        {
            spare = emptySpares.TryPop(out var empty) ? empty : new SpareArrays();
            spares.Add(array.Length, spare);
        }

        spare.Arrays.Push(array);
        spare.LastUse = epoch;
    }

    private void NoteUnpooled(int slot)
    {
        if (cachedValues[slot] is not null || cachedGradients[slot] is not null)
        {
            reviewRetention = true;
        }
    }

    private void ReleaseUnused()
    {
        for (int slot = 0; slot < count; slot++)
        {
            byte flag = flags[slot];
            if ((flag & PooledFlag) == 0 && cachedValues[slot] is { } values)
            {
                cachedValues[slot] = null;
                Return(values);
            }

            // Retain cached gradient buffers for same-length pooled slots even if Backward was not called in this recording.
            if (cachedGradients[slot] is { } gradient
                && (flag & (TouchedFlag | ExternalGradientFlag)) != TouchedFlag
                && ((flag & PooledFlag) == 0 || gradient.Length != Length(slot)))
            {
                cachedGradients[slot] = null;
                Return(gradient);
            }
        }

        for (int slot = count; slot < cachedCount; slot++)
        {
            if (cachedValues[slot] is { } values)
            {
                cachedValues[slot] = null;
                Return(values);
            }

            if (cachedGradients[slot] is { } gradient)
            {
                cachedGradients[slot] = null;
                Return(gradient);
            }
        }

        cachedCount = count;
        TrimSpares();
        reviewRetention = false;
    }

    private void TrimSpares()
    {
        if (spares is null)
        {
            return;
        }

        long limit = RetentionFactor * peakBorrowed;
        if (retained > limit)
        {
            if (trimOrder.Length < spares.Count)
            {
                trimOrder = new SpareArrays[spares.Count];
            }

            int candidates = 0;
            foreach (var spare in spares.Values)
            {
                if (spare.Arrays.Count > 0)
                {
                    trimOrder[candidates++] = spare;
                }
            }

            Span<SpareArrays> order = trimOrder.AsSpan(0, candidates);
            order.Sort(static (x, y) => x.LastUse.CompareTo(y.LastUse));
            foreach (var spare in order)
            {
                while (retained > limit && spare.Arrays.TryPop(out var array))
                {
                    retained -= array.Length;
                }

                if (retained <= limit)
                {
                    break;
                }
            }

            order.Clear();
        }

        foreach (var (length, spare) in spares)
        {
            if (spare.Arrays.Count == 0 && spare.LastUse != epoch)
            {
                spares.Remove(length);
                emptySpares.Push(spare);
            }
        }
    }

    private int LeadingDimension(int slot) => Math.Max(1, rows[slot]);

    private sealed class SpareArrays
    {
        public readonly Stack<T[]> Arrays = new();
        public long LastUse;
    }

    // Empty slices may carry an out-of-bounds start offset; clamp to 0 since no elements are accessed.
    private int Offset(in ArrayView view)
        => view.RowCount == 0 || view.ColumnCount == 0 ? 0 : view.RowStart + view.ColumnStart * rows[view.Slot];

    private ReadOnlyMatrixSpan<T> MatrixOf(Span<T> storage, scoped in ArrayView view)
    {
        int slot = view.Slot;
        if (view.Broadcast)
        {
            var repeated = new ReadOnlyVectorSpan<T>(storage.Slice(Offset(view), view.RowCount)).Broadcast(view.ColumnCount);
            return view.Transposed ? repeated.T : repeated;
        }

        var matrix = new ReadOnlyMatrixSpan<T>(storage, rows[slot], columns[slot]);
        if (view.RowCount != rows[slot] || view.ColumnCount != columns[slot])
        {
            matrix = matrix[view.RowStart..(view.RowStart + view.RowCount), view.ColumnStart..(view.ColumnStart + view.ColumnCount)];
        }

        return view.Transposed ? matrix.T : matrix;
    }

    private BlasOperand Operand(T[] storage, scoped in ArrayView view) => Operand((Span<T>)storage, view);

    private BlasOperand Operand(Span<T> storage, scoped in ArrayView view)
        => new(storage[Offset(view)..], view.Transposed, LeadingDimension(view.Slot));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static T FromDouble(double value)
    {
        if (typeof(T) == typeof(double))
        {
            return Unsafe.BitCast<double, T>(value);
        }

        if (typeof(T) == typeof(float))
        {
            return Unsafe.BitCast<float, T>((float)value);
        }

        return T.CreateTruncating(value);
    }

    private readonly ref struct BlasOperand(ReadOnlySpan<T> data, bool transposed, int leadingDimension)
    {
        public readonly ReadOnlySpan<T> Data = data;
        public readonly bool Transposed = transposed;
        public readonly int Ld = leadingDimension;

        public BlasOperand Transpose() => new(Data, !Transposed, Ld);
    }

    private static void Gemm(BlasOperand x, BlasOperand y, int m, int n, int k, bool overwrite, Span<T> c, int ldc)
    {
        var ta = x.Transposed ? BlasTranspose.Transpose : BlasTranspose.None;
        var tb = y.Transposed ? BlasTranspose.Transpose : BlasTranspose.None;
        if (typeof(T) == typeof(double))
        {
            Blas.Gemm(ta, tb, m, n, k, 1.0, MemoryMarshal.Cast<T, double>(x.Data), x.Ld, MemoryMarshal.Cast<T, double>(y.Data), y.Ld,
                overwrite ? 0.0 : 1.0, MemoryMarshal.Cast<T, double>(c), ldc);
        }
        else
        {
            Blas.Gemm(ta, tb, m, n, k, 1f, MemoryMarshal.Cast<T, float>(x.Data), x.Ld, MemoryMarshal.Cast<T, float>(y.Data), y.Ld,
                overwrite ? 0f : 1f, MemoryMarshal.Cast<T, float>(c), ldc);
        }
    }

    private static void Gemv(bool transpose, int m, int n, ReadOnlySpan<T> a, int lda, ReadOnlySpan<T> x, bool overwrite, Span<T> y)
    {
        var trans = transpose ? BlasTranspose.Transpose : BlasTranspose.None;
        if (typeof(T) == typeof(double))
        {
            Blas.Gemv(trans, m, n, 1.0, MemoryMarshal.Cast<T, double>(a), lda, MemoryMarshal.Cast<T, double>(x), overwrite ? 0.0 : 1.0,
                MemoryMarshal.Cast<T, double>(y));
        }
        else
        {
            Blas.Gemv(trans, m, n, 1f, MemoryMarshal.Cast<T, float>(a), lda, MemoryMarshal.Cast<T, float>(x), overwrite ? 0f : 1f,
                MemoryMarshal.Cast<T, float>(y));
        }
    }

    // Detects the first zero y[j] before Blas.Ger so columns skipped by BLAS can still propagate IEEE 754 0 * non-finite = NaN;
    // scanning before BLAS avoids re-reading cache-cold vectors afterward.
    private static int FirstSkippedOuterColumn(int m, int n, ReadOnlySpan<T> x, ReadOnlySpan<T> y)
    {
        int j = NextZero(y.Slice(0, n), 0);
        return m == 0 || j == n || !ContainsNonFinite(x.Slice(0, m)) ? n : j;
    }

    // Adds x * y[j] for zero y[j] columns skipped by Blas.Ger; products are 0 or NaN, so re-adding to unskipped columns is idempotent.
    private static void AddSkippedOuterColumns(int first, int m, int n, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> a, int lda)
    {
        y = y.Slice(0, n);
        for (int j = first; j < n; j = NextZero(y, j + 1))
        {
            T t = y[j];
            Span<T> column = a.Slice(j * lda, m);
            for (int i = 0; i < m; i++)
            {
                column[i] += x[i] * t;
            }
        }
    }

    // Restores IEEE 754 0 * non-finite = NaN for columns skipped by non-transposed Blas.Gemv when x[j] == 0.
    private static void AddSkippedColumns(int first, int m, int n, ReadOnlySpan<T> a, int lda, ReadOnlySpan<T> x, Span<T> y)
    {
        x = x.Slice(0, n);
        for (int j = first; j < n; j = NextZero(x, j + 1))
        {
            ReadOnlySpan<T> column = a.Slice(j * lda, m);
            if (ContainsNonFinite(column))
            {
                T t = x[j];
                for (int i = 0; i < m; i++)
                {
                    y[i] += column[i] * t;
                }
            }
        }
    }

    private static int NextZero(ReadOnlySpan<T> values, int start)
    {
        int i = start;
        if (System.Numerics.Vector.IsHardwareAccelerated && System.Numerics.Vector<T>.IsSupported)
        {
            ref T origin = ref MemoryMarshal.GetReference(values);
            int last = values.Length - System.Numerics.Vector<T>.Count;
            while (i <= last
                && !System.Numerics.Vector.EqualsAny(System.Numerics.Vector.LoadUnsafe(ref origin, (nuint)i), System.Numerics.Vector<T>.Zero))
            {
                i += System.Numerics.Vector<T>.Count;
            }
        }

        for (; i < values.Length; i++)
        {
            if (T.IsZero(values[i]))
            {
                return i;
            }
        }

        return values.Length;
    }

    // v - v == 0 iff every lane is finite; +/-Infinity and NaN yield NaN.
    private static bool ContainsNonFinite(ReadOnlySpan<T> values)
    {
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated && System.Numerics.Vector<T>.IsSupported)
        {
            ref T origin = ref MemoryMarshal.GetReference(values);
            int last = values.Length - System.Numerics.Vector<T>.Count;
            for (; i <= last; i += System.Numerics.Vector<T>.Count)
            {
                var v = System.Numerics.Vector.LoadUnsafe(ref origin, (nuint)i);
                if (!System.Numerics.Vector.EqualsAll(v - v, System.Numerics.Vector<T>.Zero))
                {
                    return true;
                }
            }
        }

        for (; i < values.Length; i++)
        {
            if (!T.IsFinite(values[i]))
            {
                return true;
            }
        }

        return false;
    }

    private static void Ger(int m, int n, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> a, int lda)
    {
        if (typeof(T) == typeof(double))
        {
            Blas.Ger(m, n, 1.0, MemoryMarshal.Cast<T, double>(x), MemoryMarshal.Cast<T, double>(y), MemoryMarshal.Cast<T, double>(a), lda);
        }
        else
        {
            Blas.Ger(m, n, 1f, MemoryMarshal.Cast<T, float>(x), MemoryMarshal.Cast<T, float>(y), MemoryMarshal.Cast<T, float>(a), lda);
        }
    }
}
