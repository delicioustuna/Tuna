using System.Numerics;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

internal readonly record struct TensorView(
    int StorageRows, int StorageColumns, int StorageCount,
    int FirstRow, int FirstColumn, int FirstMatrix, int RowCount, int ColumnCount, int Count,
    bool Transposed, bool Repeated)
{
    internal int Rows => Transposed ? ColumnCount : RowCount;
    internal int Columns => Transposed ? RowCount : ColumnCount;
    internal bool IsEmpty => RowCount == 0 || ColumnCount == 0 || Count == 0;
    internal bool IsWhole => !Repeated && FirstRow == 0 && FirstColumn == 0 && FirstMatrix == 0
        && RowCount == StorageRows && ColumnCount == StorageColumns && Count == StorageCount;
    internal bool CoversRepeatedSource => Repeated && FirstRow == 0 && FirstColumn == 0
        && RowCount == StorageRows && ColumnCount == StorageColumns;

    // An empty tensor can have more columns across its matrices than fit in a matrix dimension.
    internal int FlatColumns => StorageRows == 0
        ? (int)Math.Min((long)StorageColumns * StorageCount, int.MaxValue)
        : checked(StorageColumns * StorageCount);

    internal static TensorView Whole(int rows, int columns, int count, bool repeated = false) =>
        new(rows, columns, count, 0, 0, 0, rows, columns, count, Transposed: false, repeated);

    internal TensorView Slice(Range rows, Range columns, Range matrices)
    {
        var (firstRow, rowCount) = (Transposed ? columns : rows).GetOffsetAndLength(RowCount);
        var (firstColumn, columnCount) = (Transposed ? rows : columns).GetOffsetAndLength(ColumnCount);
        var (firstMatrix, count) = matrices.GetOffsetAndLength(Count);
        return this with
        {
            FirstRow = FirstRow + firstRow,
            FirstColumn = FirstColumn + firstColumn,
            FirstMatrix = FirstMatrix + firstMatrix,
            RowCount = rowCount,
            ColumnCount = columnCount,
            Count = count,
        };
    }

    internal ReadOnlyTensorSpan<T> Read<T>(ReadOnlyMatrixSpan<T> storage)
        where T : unmanaged, INumberBase<T>
    {
        if (Repeated)
        {
            var matrix = storage[FirstRow..(FirstRow + RowCount), FirstColumn..(FirstColumn + ColumnCount)];
            return (Transposed ? matrix.T : matrix).Broadcast(Count);
        }

        var tensor = new ReadOnlyTensorSpan<T>(storage.Data, StorageRows, StorageColumns, StorageCount);
        if (RowCount != StorageRows || ColumnCount != StorageColumns || Count != StorageCount)
        {
            tensor = tensor[FirstRow..(FirstRow + RowCount), FirstColumn..(FirstColumn + ColumnCount), FirstMatrix..(FirstMatrix + Count)];
        }

        return Transposed ? tensor.T : tensor;
    }

    internal TensorSpan<T> Write<T>(MatrixSpan<T> storage)
        where T : unmanaged, INumberBase<T>
    {
        var tensor = new TensorSpan<T>(storage.Data, StorageRows, StorageColumns, StorageCount);
        if (RowCount != StorageRows || ColumnCount != StorageColumns || Count != StorageCount)
        {
            tensor = tensor[FirstRow..(FirstRow + RowCount), FirstColumn..(FirstColumn + ColumnCount), FirstMatrix..(FirstMatrix + Count)];
        }

        return Transposed ? tensor.T : tensor;
    }

    internal MatrixSpan<T> RepeatedSource<T>(MatrixSpan<T> storage)
        where T : unmanaged, INumberBase<T>
    {
        var source = storage[FirstRow..(FirstRow + RowCount), FirstColumn..(FirstColumn + ColumnCount)];
        return Transposed ? source.T : source;
    }
}
