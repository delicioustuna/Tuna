using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Tuna.NeuralNetwork;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

internal static class Guard
{
    internal static void RequireElementType<T>()
    {
        if (typeof(T) != typeof(float) && typeof(T) != typeof(double))
        {
            throw new NotSupportedException($"The element type {typeof(T).Name} is not supported. Use float or double.");
        }
    }

    internal static void RequireShape(int expectedRows, int expectedColumns, int rows, int columns, string name)
    {
        if (rows != expectedRows || columns != expectedColumns)
        {
            throw new ArgumentException($"Expected a {expectedRows}×{expectedColumns} matrix but got {rows}×{columns}.", name);
        }
    }

    internal static void RequireLength(int expected, int actual, string name)
    {
        if (expected != actual)
        {
            throw new ArgumentException($"Expected length {expected} but got {actual}.", name);
        }
    }

    internal static void RequireOptionalLength(int expected, int actual, string name)
    {
        if (actual != 0 && actual != expected)
        {
            throw new ArgumentException($"Expected length {expected} (or 0 to omit) but got {actual}.", name);
        }
    }

    internal static void RequirePositive(int value, string name)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(value, 1, name);
    }

    // allowExact が true の場合は同一要素・同一インデックス参照（インプレース）のみ許可する。
    internal static void RequireNoOverlap<T>(ReadOnlyMatrixSpan<T> source, MatrixSpan<T> destination, bool allowExact, string name)
        where T : unmanaged, INumberBase<T>
    {
        if (Overlaps(source, destination.AsReadOnly(), out bool exact) && !(allowExact && exact))
        {
            throw new ArgumentException("The destination must not overlap the input.", name);
        }
    }

    // 列内が連続かつ列ストライドが一致するビュー同士は O(1) で重なりを判定し、それ以外は StridedLayout.Overlaps に委譲する。
    internal static bool Overlaps<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b, out bool exact)
        where T : unmanaged, INumberBase<T>
    {
        exact = false;
        if (a.Rows == 0 || a.Columns == 0 || b.Rows == 0 || b.Columns == 0)
        {
            return false;
        }

        int ld = b.ColumnStride;
        if (a.RowStride == 1 && b.RowStride == 1 && a.Columns > 1 && b.Columns > 1 && a.ColumnStride == ld && a.Rows <= ld && b.Rows <= ld)
        {
            nint bytes = Unsafe.ByteOffset(in b.Reference, in a.Reference);
            if (bytes % Unsafe.SizeOf<T>() == 0)
            {
                long offset = bytes / Unsafe.SizeOf<T>();
                exact = offset == 0 && a.Rows == b.Rows && a.Columns == b.Columns;
                if (exact)
                {
                    return true;
                }

                // a の第 j 列は b の第 q + j 列の行 [r, r + a.Rows)（ld を超えた分は第 q + j + 1 列の行 [0, r + a.Rows − ld)）
                long q = offset >= 0 ? offset / ld : -((-offset + ld - 1) / ld), r = offset - q * ld;
                int aColumns = a.Columns, bColumns = b.Columns;
                bool primary = r < b.Rows && q < bColumns && q + aColumns > 0;
                bool wrapped = r + a.Rows > ld && q + 1 < bColumns && q + 1 + aColumns > 0;
                return primary || wrapped;
            }
        }

        return StridedLayout.Overlaps(in a.Reference, a.Rows, a.Columns, a.RowStride, a.ColumnStride,
            in b.Reference, b.Rows, b.Columns, b.RowStride, b.ColumnStride, out exact);
    }

    internal static void RequireNoOverlap<T>(ReadOnlyVectorSpan<T> source, MatrixSpan<T> destination, string name)
        where T : unmanaged, INumberBase<T>
    {
        var d = destination.AsReadOnly();
        if (StridedLayout.Overlaps(in source.Reference, source.Length, 1, source.Stride, 0,
                in d.Reference, d.Rows, d.Columns, d.RowStride, d.ColumnStride, out _))
        {
            throw new ArgumentException("The destination must not overlap the input.", name);
        }
    }

    internal static void RequireNoOverlap<T>(ReadOnlyMatrixSpan<T> source, TensorSpan<T> destination, string name)
        where T : unmanaged, INumberBase<T> =>
        RequireNoOverlap(in source.Reference, source.Rows, source.Columns, source.RowStride, source.ColumnStride, source.Data,
            destination, name);

    internal static void RequireNoOverlap<T>(ReadOnlyVectorSpan<T> source, TensorSpan<T> destination, string name)
        where T : unmanaged, INumberBase<T> =>
        RequireNoOverlap(in source.Reference, source.Length, 1, source.Stride, 0, source.Data, destination, name);

    // テンソルが入れ子ストライド構造なら要素位置で精密判定し、それ以外は先頭〜末尾の領域重なりで保守的に判定する。
    private static void RequireNoOverlap<T>(ref readonly T source, int rows, int columns, int rowStride, int columnStride,
        ReadOnlySpan<T> sourceData, TensorSpan<T> destination, string name)
        where T : unmanaged, INumberBase<T>
    {
        var d = destination.AsReadOnly();
        bool overlaps = StridedBatchLayout.IsNested(d.Rows, d.Columns, d.MatrixCount, d.RowStride, d.ColumnStride, d.MatrixStride)
            ? StridedBatchLayout.Overlaps(in source, rows, columns, 1, rowStride, columnStride, 0,
                in d.Reference, d.Rows, d.Columns, d.MatrixCount, d.RowStride, d.ColumnStride, d.MatrixStride, out _)
            : sourceData.Overlaps(d.Data);
        if (overlaps)
        {
            throw new ArgumentException("The destination must not overlap the input.", name);
        }
    }
}

// 計算カーネルは列内が連続（RowStride == 1）であることを前提とし、非連続ビューのみ ArrayPool へ詰め直す。
internal static class Dense
{
    internal static ReadOnlyMatrixSpan<T> Columns<T>(ReadOnlyMatrixSpan<T> x, ref T[]? rented) where T : unmanaged, INumberBase<T>
    {
        if (x.Rows <= 1 || x.RowStride == 1)
        {
            return x;
        }

        rented = ArrayPool<T>.Shared.Rent(x.Rows * x.Columns);
        var copy = new MatrixSpan<T>(rented.AsSpan(0, x.Rows * x.Columns), x.Rows, x.Columns);
        x.CopyTo(copy);
        return copy;
    }

    internal static ReadOnlyVectorSpan<T> Contiguous<T>(ReadOnlyVectorSpan<T> x, ref T[]? rented) where T : unmanaged, INumberBase<T>
    {
        if (x.Length <= 1 || x.Stride == 1)
        {
            return x;
        }

        rented = ArrayPool<T>.Shared.Rent(x.Length);
        var copy = new VectorSpan<T>(rented.AsSpan(0, x.Length));
        x.CopyTo(copy);
        return copy;
    }

    internal static T[] ToPinnedArray<T>(ReadOnlyMatrixSpan<T> x) where T : unmanaged, INumberBase<T>
    {
        var array = GC.AllocateUninitializedArray<T>(x.Rows * x.Columns, pinned: true);
        x.CopyTo(new MatrixSpan<T>(array, x.Rows, x.Columns));
        return array;
    }

    internal static T[] ToPinnedArray<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, INumberBase<T>
    {
        var array = GC.AllocateUninitializedArray<T>(x.Length, pinned: true);
        x.CopyTo(new VectorSpan<T>(array));
        return array;
    }

    internal static void Return<T>(T[]? rented)
    {
        if (rented is not null)
        {
            ArrayPool<T>.Shared.Return(rented);
        }
    }

    // fixed 文で固定するため、読み取り専用ビューの先頭参照を書き換え可能な ref として取得する。
    internal static ref T Ref<T>(ReadOnlyMatrixSpan<T> x) where T : unmanaged, INumberBase<T> => ref Unsafe.AsRef(in x.Reference);

    internal static ref T Ref<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, INumberBase<T> => ref Unsafe.AsRef(in x.Reference);

    internal static ref T Ref<T>(MatrixSpan<T> x) where T : unmanaged, INumberBase<T> => ref Unsafe.AsRef(in x.AsReadOnly().Reference);
}

// 並列ラムダには ref struct をキャプチャできないため、ポインタ経由でビューを再構築する。
internal static unsafe class Views
{
    internal static ReadOnlyMatrixSpan<T> Read<T>(T* p, int rows, int columns, int columnStride) where T : unmanaged, INumberBase<T> =>
        new(in Unsafe.AsRef<T>(p), rows, columns, StridedLayout.Normalize(rows, 1), StridedLayout.Normalize(columns, columnStride));

    internal static MatrixSpan<T> Write<T>(T* p, int rows, int columns, int columnStride) where T : unmanaged, INumberBase<T> =>
        new(ref Unsafe.AsRef<T>(p), rows, columns, StridedLayout.Normalize(rows, 1), StridedLayout.Normalize(columns, columnStride));

    internal static Span<T> Span<T>(T* p, int length) where T : unmanaged => new(p, length);
}

internal static class Simd
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<T> Fma<T>(Vector256<T> a, Vector256<T> b, Vector256<T> c)
    {
        if (typeof(T) == typeof(float))
        {
            return Vector256.FusedMultiplyAdd(a.AsSingle(), b.AsSingle(), c.AsSingle()).As<float, T>();
        }

        return Vector256.FusedMultiplyAdd(a.AsDouble(), b.AsDouble(), c.AsDouble()).As<double, T>();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector512<T> Fma<T>(Vector512<T> a, Vector512<T> b, Vector512<T> c)
    {
        if (typeof(T) == typeof(float))
        {
            return Vector512.FusedMultiplyAdd(a.AsSingle(), b.AsSingle(), c.AsSingle()).As<float, T>();
        }

        return Vector512.FusedMultiplyAdd(a.AsDouble(), b.AsDouble(), c.AsDouble()).As<double, T>();
    }

    internal static bool Wide => Vector512.IsHardwareAccelerated;
}

internal static class Split
{
    internal static int Threads(double work) => Parallelism.Degree(work, Parallelism.Grain);

    internal static (int Start, int End) Range(int count, int chunks, int index) =>
        ((int)((long)count * index / chunks), (int)((long)count * (index + 1) / chunks));

    internal static int Chunks(int count, int threads, int minimum)
    {
        if (threads <= 1 || count <= minimum)
        {
            return 1;
        }

        return Math.Max(1, Math.Min(threads * 2, count / minimum));
    }

    // 値型のワーク構造体で区間を処理し、呼び出しごとのクロージャやデリゲートのアロケーションを避ける。
    internal static unsafe void For<TWork>(int count, int chunks, int threads, ref TWork work)
        where TWork : unmanaged, IRangeWork
    {
        if (chunks <= 1)
        {
            work.Invoke(0, count);
            return;
        }

        fixed (TWork* state = &work)
        {
            var ranges = new Ranges<TWork>(state, count, chunks);
            Parallelism.For(chunks, threads, ref ranges);
        }
    }

    internal static void For(int count, int chunks, int threads, Action<int, int> body)
    {
        if (chunks <= 1)
        {
            body(0, count);
            return;
        }

        Parallelism.For(chunks, threads, i =>
        {
            var (start, end) = Range(count, chunks, i);
            if (end > start)
            {
                body(start, end);
            }
        });
    }

    internal interface IRangeWork
    {
        void Invoke(int start, int end);
    }

    private readonly unsafe struct Ranges<TWork>(TWork* work, int count, int chunks) : IParallelWork
        where TWork : unmanaged, IRangeWork
    {
        public void Invoke(int index)
        {
            var (start, end) = Range(count, chunks, index);
            if (end > start)
            {
                work->Invoke(start, end);
            }
        }
    }
}
