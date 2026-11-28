namespace Tuna.Differentiation;

// Contiguous submatrix view into a column-major slot (vectors are 1-column).
// RowStart/ColumnStart/RowCount/ColumnCount are in storage coordinates; Transposed swaps logical rows/columns.
// Broadcast repeats a vector slice along ColumnCount columns (stride 0) when count > 1 (or rows when Transposed).
internal readonly struct ArrayView(int slot, int rowStart, int columnStart, int rowCount, int columnCount, bool transposed, bool broadcast = false)
{
    public readonly int Slot = slot;
    public readonly int RowStart = rowStart;
    public readonly int ColumnStart = columnStart;
    public readonly int RowCount = rowCount;
    public readonly int ColumnCount = columnCount;
    public readonly bool Transposed = transposed;
    public readonly bool Broadcast = broadcast;

    public int Rows => Transposed ? ColumnCount : RowCount;

    public int Columns => Transposed ? RowCount : ColumnCount;

    public int Length => RowCount * ColumnCount;

    public int StorageLength => Broadcast ? RowCount : RowCount * ColumnCount;

    public static ArrayView Whole(int slot, int rows, int columns) => new(slot, 0, 0, rows, columns, false);

    public ArrayView Transpose() => new(Slot, RowStart, ColumnStart, RowCount, ColumnCount, !Transposed, Broadcast);

    public ArrayView Slice(Range rows, Range columns)
    {
        var (row, rowCount) = rows.GetOffsetAndLength(Rows);
        var (column, columnCount) = columns.GetOffsetAndLength(Columns);
        if (Broadcast)
        {
            // Storage rows index vector elements and storage columns hold the repeat count.
            return Transposed
                ? Repeat(RowStart + column, columnCount, rowCount, true)
                : Repeat(RowStart + row, rowCount, columnCount, false);
        }

        return Transposed
            ? new(Slot, RowStart + column, ColumnStart + row, columnCount, rowCount, true)
            : new(Slot, RowStart + row, ColumnStart + column, rowCount, columnCount, false);
    }

    public ArrayView Slice(Range range)
    {
        var (start, length) = range.GetOffsetAndLength(RowCount);
        return new(Slot, RowStart + start, ColumnStart, length, 1, false);
    }

    public ArrayView Repeat(int count, bool asRows) => Repeat(RowStart, RowCount, count, asRows);

    private ArrayView Repeat(int start, int length, int count, bool asRows) => new(Slot, start, ColumnStart, length, count, asRows, count > 1);
}

// Enum ordering is relied upon by range-based dispatch in ArrayStore:
// unary, activations, binary, array-scalar, reductions, softmax, losses, linear algebra.
// Column and row reductions alternate so parity selects the axis.
internal enum ArrayOperation : byte
{
    MatrixMultiply,
    MatrixVectorMultiply,
    Outer,
    Cross,
    Wedge,
    Sum,
    Dot,
    Trace,

    Negate,
    Exp,
    Log,
    Sqrt,
    Square,
    Abs,
    Expm1,
    Log1p,
    Tanh,
    Logistic,
    LogLogistic,
    Logit,
    Softplus,
    Log1mExp,
    Reciprocal,

    Relu,
    LeakyRelu,
    Elu,
    Selu,
    Gelu,
    GeluTanh,
    Silu,

    Add,
    Subtract,
    Hadamard,
    Divide,
    Min,
    Max,
    Pow,
    Hypot,
    LogAddExp,
    LogSubExp,

    AddScalar,
    SubtractScalar,
    MultiplyScalar,
    DivideScalar,
    MinScalar,
    MaxScalar,
    PowScalar,
    HypotScalar,
    LogAddExpScalar,
    LogSubExpScalar,
    ScalarSubtract,

    SumOfSquares,
    Mean,
    LogSumExp,

    ColumnSums,
    RowSums,
    ColumnSumsOfSquares,
    RowSumsOfSquares,
    ColumnMeans,
    RowMeans,
    ColumnLogSumExp,
    RowLogSumExp,

    ColumnSoftmax,
    RowSoftmax,
    ColumnLogSoftmax,
    RowLogSoftmax,

    // Loss records store a scratch-slot index in ScalarInput (P - onehot for labels, log-softmax L for targets, -1 for logistic).
    SoftmaxCrossEntropyLabels,
    SoftmaxCrossEntropyTargets,
    LogisticCrossEntropy,

    // Linear algebra records (double only) store a DecompositionPool index in ScalarInput (-1 for triangular solves).
    // Multi-output decompositions store the primary slot in Output and auxiliary slots in DecompositionInfo.Second/Third.
    LuSolveVector,
    LuSolveMatrix,
    LuSolveTransposedVector,
    LuSolveTransposedMatrix,
    LuLogAbsDeterminant,
    SolveLowerTriangularVector,
    SolveLowerTriangularMatrix,
    SolveUpperTriangularVector,
    SolveUpperTriangularMatrix,
    Cholesky,
    CholeskyLogDeterminant,
    CholeskySolveVector,
    CholeskySolveMatrix,
    CholeskyInverseQuadraticForm,
    Qr,
    SymmetricEigen,
    SingularValueDecomposition,
    SvdSolve,
    SvdPseudoInverse,

    // ScalarInput is the index of the user operation in ArrayStore.
    UserOperation,

    // ScalarInput holds the slot containing the copied embedding indices.
    Embedding,

    PointwiseSum,
    PointwiseVector,
}

// Position is the scalar nodeCount when recorded; backward processes this record before scalar node Position - 1.
// Scalar holds the unrounded parameter (e.g. a scalar operand, activation parameter, or retained SVD rank). ScalarInput holds either
// a scalar node index (-1 if constant), a cached forward slot (losses) or a decomposition or user-operation index.
internal struct ArrayRecord
{
    public ArrayOperation Operation;
    public bool Single;
    public int Position;
    public int Output;
    public int ScalarInput;
    public double Scalar;
    public ArrayView Left;
    public ArrayView Right;
}

internal readonly struct ArrayLeaf(bool single, int slot, bool transposed, int columnsPerMatrix)
{
    public readonly bool Single = single;
    public readonly int Slot = slot;
    public readonly bool Transposed = transposed;
    public readonly int ColumnsPerMatrix = columnsPerMatrix;
}
