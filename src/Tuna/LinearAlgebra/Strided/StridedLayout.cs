using System.Runtime.CompilerServices;

namespace Tuna.LinearAlgebra;

// 2 階ビューの Lengths / Strides を ReadOnlySpan<nint> として公開するための固定長バッファ。
[InlineArray(2)]
internal struct NintPair
{
    private nint element0;

    // AggressiveOptimization メソッド内でも確実にインライン展開させる。
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static NintPair Create(nint first, nint second)
    {
        NintPair pair = default;
        pair[0] = first;
        pair[1] = second;
        return pair;
    }
}

// 2 階ストライドビューの形状検証・ストライド正規化（長さ <= 1 は 0、TensorSpan 互換）・重なり判定。
internal static class StridedLayout
{
    internal static int Normalize(int length, int stride) => length <= 1 ? 0 : stride;

    internal static int Extent(int rows, int columns, int rowStride, int columnStride) =>
        rows == 0 || columns == 0 ? 0 : checked((rows - 1) * rowStride + (columns - 1) * columnStride + 1);

    // 形状とストライドを検証して正規化する（writable 時はストライド 0 のブロードキャストなど非単射配置を拒否）。
    internal static (int Rows, int Columns, int RowStride, int ColumnStride) Validate(
        ReadOnlySpan<nint> lengths, ReadOnlySpan<nint> strides, int rank, bool writable)
    {
        if (lengths.Length != rank || strides.Length != rank)
        {
            throw new ArgumentException($"Lengths and strides must have exactly {rank} elements.");
        }

        int rows = ToInt32(lengths[0], nameof(lengths));
        int columns = rank == 2 ? ToInt32(lengths[1], nameof(lengths)) : 1;
        int rowStride = Normalize(rows, ToInt32(strides[0], nameof(strides)));
        int columnStride = rank == 2 ? Normalize(columns, ToInt32(strides[1], nameof(strides))) : 0;
        if (writable && !IsInjective(rows, columns, rowStride, columnStride))
        {
            throw new ArgumentException("A writable view must not map two indices to the same element.", nameof(strides));
        }

        Extent(rows, columns, rowStride, columnStride);
        return (rows, columns, rowStride, columnStride);
    }

    private static int ToInt32(nint value, string name)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(value, name);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(value, int.MaxValue, name);
        return (int)value;
    }

    // 入れ子条件（StridedBatchLayout.IsNested）で単射性を判定し、Tensor.GetMatrix との整合性を保つ。
    private static bool IsInjective(int rows, int columns, int rowStride, int columnStride) =>
        StridedBatchLayout.IsNested(rows, columns, 1, rowStride, columnStride, 0);

    // 入力と書き込み先（単射ビュー前提）が要素を共有するか判定する（範囲重複時のみ要素単位で検証。exact は全添字一致）。
    internal static bool Overlaps<T>(
        ref readonly T source, int rows, int columns, int rowStride, int columnStride,
        ref readonly T destination, int destinationRows, int destinationColumns, int destinationRowStride, int destinationColumnStride,
        out bool exact)
    {
        exact = false;
        int sourceExtent = Extent(rows, columns, rowStride, columnStride);
        int destinationExtent = Extent(destinationRows, destinationColumns, destinationRowStride, destinationColumnStride);
        if (sourceExtent == 0 || destinationExtent == 0)
        {
            return false;
        }

        int size = Unsafe.SizeOf<T>();
        nint byteOffset = Unsafe.ByteOffset(in destination, in source);
        if (byteOffset >= (nint)destinationExtent * size || byteOffset + (nint)sourceExtent * size <= 0)
        {
            return false;
        }

        if (byteOffset % size != 0)
        {
            return true;
        }

        long offset = byteOffset / size;
        exact = offset == 0 && rows == destinationRows && columns == destinationColumns
            && rowStride == destinationRowStride && columnStride == destinationColumnStride;
        if (exact)
        {
            return true;
        }

        for (int j = 0; j < columns; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                long position = offset + (long)i * rowStride + (long)j * columnStride;
                if (Contains(position, destinationRows, destinationColumns, destinationRowStride, destinationColumnStride))
                {
                    return true;
                }
            }
        }

        return false;
    }

    // 異型ビュー間のバイト重複を判定する（要素サイズが整数倍かつ境界整列時は要素単位、それ以外は範囲重複で保守的に判定）。
    internal static bool OverlapsAcrossTypes<TSource, TDestination>(
        ref readonly TSource source, int rows, int columns, int rowStride, int columnStride,
        ref readonly TDestination destination, int destinationRows, int destinationColumns, int destinationRowStride, int destinationColumnStride)
    {
        int sourceExtent = Extent(rows, columns, rowStride, columnStride);
        int destinationExtent = Extent(destinationRows, destinationColumns, destinationRowStride, destinationColumnStride);
        if (sourceExtent == 0 || destinationExtent == 0)
        {
            return false;
        }

        int sourceSize = Unsafe.SizeOf<TSource>();
        int destinationSize = Unsafe.SizeOf<TDestination>();
        nint byteOffset = Unsafe.ByteOffset(
            in Unsafe.As<TDestination, byte>(ref Unsafe.AsRef(in destination)),
            in Unsafe.As<TSource, byte>(ref Unsafe.AsRef(in source)));
        if (byteOffset >= (nint)destinationExtent * destinationSize || byteOffset + (nint)sourceExtent * sourceSize <= 0)
        {
            return false;
        }

        if (sourceSize % destinationSize != 0 || byteOffset % destinationSize != 0)
        {
            return true;
        }

        long offset = byteOffset / destinationSize;
        int parts = sourceSize / destinationSize;
        for (int j = 0; j < columns; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                long position = offset + ((long)i * rowStride + (long)j * columnStride) * parts;
                for (int p = 0; p < parts; p++)
                {
                    if (Contains(position + p, destinationRows, destinationColumns, destinationRowStride, destinationColumnStride))
                    {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    // 入れ子配置の単射ビューに position が含まれるか O(1) で判定する（大ストライド軸の添字は商で一意に決まる）。
    private static bool Contains(long position, int rows, int columns, int rowStride, int columnStride)
    {
        if (position < 0)
        {
            return false;
        }

        if (rows <= 1 && columns <= 1)
        {
            return position == 0;
        }

        if (rows <= 1 || columns <= 1)
        {
            var (stride, length) = rows <= 1 ? (columnStride, columns) : (rowStride, rows);
            return position % stride == 0 && position / stride < length;
        }

        var (smallStride, smallLength, largeStride, largeLength) = rowStride <= columnStride
            ? (rowStride, rows, columnStride, columns)
            : (columnStride, columns, rowStride, rows);
        long large = position / largeStride;
        long remainder = position - large * largeStride;
        return large < largeLength && remainder % smallStride == 0 && remainder / smallStride < smallLength;
    }
}

[InlineArray(3)]
internal struct NintTriple
{
    private nint element0;

    internal static NintTriple Create(nint first, nint second, nint third)
    {
        NintTriple triple = default;
        triple[0] = first;
        triple[1] = second;
        triple[2] = third;
        return triple;
    }
}

// 3 階 Tensor（最終軸がバッチ軸）ビューの形状検証・入れ子単射性チェック・重なり判定。
internal static class StridedBatchLayout
{
    internal static int Extent(int rows, int columns, int count, int rowStride, int columnStride, int matrixStride) =>
        rows == 0 || columns == 0 || count == 0
            ? 0
            : checked((rows - 1) * rowStride + (columns - 1) * columnStride + (count - 1) * matrixStride + 1);

    internal const int SupportedRank = 3;

    // 未対応階数は ArgumentException と区別して NotSupportedException を送出する。
    internal static void RequireSupportedRank(int rank)
    {
        if (rank != SupportedRank)
        {
            throw new NotSupportedException(
                $"Tensor supports rank {SupportedRank} only, but rank {rank} was requested. " +
                "Use a scalar for rank 0, Vector<T> for rank 1 and Matrix<T> for rank 2; rank 4 and above are not supported yet.");
        }
    }

    internal static (int Rows, int Columns, int MatrixCount) ValidateLengths(ReadOnlySpan<nint> lengths)
    {
        RequireSupportedRank(lengths.Length);
        return (ToInt32(lengths[0], nameof(lengths)), ToInt32(lengths[1], nameof(lengths)), ToInt32(lengths[2], nameof(lengths)));
    }

    internal static (int Rows, int Columns, int Count, int RowStride, int ColumnStride, int MatrixStride) Validate(
        ReadOnlySpan<nint> lengths, ReadOnlySpan<nint> strides, bool writable)
    {
        if (lengths.Length != strides.Length)
        {
            throw new ArgumentException("Lengths and strides must have the same number of elements.", nameof(strides));
        }

        var (rows, columns, count) = ValidateLengths(lengths);
        int rowStride = StridedLayout.Normalize(rows, ToInt32(strides[0], nameof(strides)));
        int columnStride = StridedLayout.Normalize(columns, ToInt32(strides[1], nameof(strides)));
        int matrixStride = StridedLayout.Normalize(count, ToInt32(strides[2], nameof(strides)));
        if (writable && !IsNested(rows, columns, count, rowStride, columnStride, matrixStride))
        {
            throw new ArgumentException("A writable view must not map two indices to the same element.", nameof(strides));
        }

        Extent(rows, columns, count, rowStride, columnStride, matrixStride);
        return (rows, columns, count, rowStride, columnStride, matrixStride);
    }

    private static int ToInt32(nint value, string name)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(value, name);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(value, int.MaxValue, name);
        return (int)value;
    }

    private static int Sort(int rows, int columns, int count, int rowStride, int columnStride, int matrixStride,
        Span<(long Stride, long Length)> dims)
    {
        int used = 0;
        if (rows > 1)
        {
            dims[used++] = (rowStride, rows);
        }

        if (columns > 1)
        {
            dims[used++] = (columnStride, columns);
        }

        if (count > 1)
        {
            dims[used++] = (matrixStride, count);
        }

        for (int i = 1; i < used; i++)
        {
            for (int j = i; j > 0 && dims[j].Stride < dims[j - 1].Stride; j--)
            {
                (dims[j], dims[j - 1]) = (dims[j - 1], dims[j]);
            }
        }

        return used;
    }

    // ストライド昇順の各次元が入れ子条件（単射性の十分条件）を満たすか判定する。
    internal static bool IsNested(int rows, int columns, int count, int rowStride, int columnStride, int matrixStride)
    {
        // 要素のない配置では、長さ 0 でない次元のストライドが 0 でも同じ要素を指す添字の組はない
        if (rows == 0 || columns == 0 || count == 0)
        {
            return true;
        }

        Span<(long Stride, long Length)> dims = stackalloc (long, long)[3];
        int used = Sort(rows, columns, count, rowStride, columnStride, matrixStride, dims);
        long reach = 0;
        for (int i = 0; i < used; i++)
        {
            if (dims[i].Stride <= reach)
            {
                return false;
            }

            reach += dims[i].Stride * (dims[i].Length - 1);
        }

        return true;
    }

    // 3 階ビュー間の要素重複を判定する（出力先は IsNested 前提）。
    internal static bool Overlaps<T>(
        ref readonly T source, int rows, int columns, int count, int rowStride, int columnStride, int matrixStride,
        ref readonly T destination, int destinationRows, int destinationColumns, int destinationCount,
        int destinationRowStride, int destinationColumnStride, int destinationMatrixStride,
        out bool exact)
    {
        exact = false;
        int sourceExtent = Extent(rows, columns, count, rowStride, columnStride, matrixStride);
        int destinationExtent = Extent(destinationRows, destinationColumns, destinationCount,
            destinationRowStride, destinationColumnStride, destinationMatrixStride);
        if (sourceExtent == 0 || destinationExtent == 0)
        {
            return false;
        }

        int size = Unsafe.SizeOf<T>();
        nint byteOffset = Unsafe.ByteOffset(in destination, in source);
        if (byteOffset >= (nint)destinationExtent * size || byteOffset + (nint)sourceExtent * size <= 0)
        {
            return false;
        }

        if (byteOffset % size != 0)
        {
            return true;
        }

        long offset = byteOffset / size;
        exact = offset == 0 && rows == destinationRows && columns == destinationColumns && count == destinationCount
            && rowStride == destinationRowStride && columnStride == destinationColumnStride && matrixStride == destinationMatrixStride;
        if (exact)
        {
            return true;
        }

        Span<(long Stride, long Length)> dims = stackalloc (long, long)[3];
        int used = Sort(destinationRows, destinationColumns, destinationCount,
            destinationRowStride, destinationColumnStride, destinationMatrixStride, dims);
        for (int k = 0; k < count; k++)
        {
            for (int j = 0; j < columns; j++)
            {
                for (int i = 0; i < rows; i++)
                {
                    long position = offset + (long)i * rowStride + (long)j * columnStride + (long)k * matrixStride;
                    if (Contains(position, dims[..used]))
                    {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    // 入れ子配置に position が含まれるか O(1) で判定する（大ストライド軸から順に添字を決定）。
    private static bool Contains(long position, ReadOnlySpan<(long Stride, long Length)> dims)
    {
        if (position < 0)
        {
            return false;
        }

        for (int i = dims.Length - 1; i >= 0; i--)
        {
            long index = position / dims[i].Stride;
            if (index >= dims[i].Length)
            {
                return false;
            }

            position -= index * dims[i].Stride;
        }

        return position == 0;
    }
}
