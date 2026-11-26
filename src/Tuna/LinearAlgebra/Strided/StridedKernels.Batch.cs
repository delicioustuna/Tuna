using System.Numerics;
using System.Runtime.CompilerServices;

namespace Tuna.LinearAlgebra;

// 3 階行列バッチに対する演算。要素演算は連結可能なら単一 2 階呼び出しにまとめ、積・求解は小規模行列をバッチ並列化する。
// 重なり判定はバッチ全体で行い、異なるインデックスの行列間の重複も検出する。
internal static partial class StridedKernels
{
    // 1 行列あたりの積和回数がこれ未満ならバッチ単位で並列化し、各行列内部は逐次実行する。
    internal const double SmallMatrixWork = 4 * Parallelism.Grain;

    internal static void Copy<T>(ReadOnlyTensorSpan<T> source, TensorSpan<T> destination)
        where T : unmanaged, INumberBase<T> =>
        Map(source, destination, default(IdentityOperator<T>));

    internal static void SumMatrices<T>(ReadOnlyTensorSpan<T> x, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        if (destination.Rows != x.Rows || destination.Columns != x.Columns)
        {
            throw new ArgumentException("Destination shape must agree.", nameof(destination));
        }

        RequireDisjoint(x, destination.Unsqueeze(2));
        if (destination.IsEmpty)
        {
            return;
        }

        if (x.MatrixCount == 0)
        {
            destination.Fill(T.Zero);
            return;
        }

        // 行列内の要素を縦に並べ、バッチ軸を列として既存の行方向の縮約に渡す。
        if ((x.Rows == 1 || x.RowStride == 1) && (x.Columns == 1 || x.ColumnStride == x.Rows) &&
            destination.TryGetDense(out var dense))
        {
            var matrices = new ReadOnlyMatrixSpan<T>(in x.Reference, dense.Length, x.MatrixCount, 1, x.MatrixStride);
            RowSumsOf<T, FirstOperator<T>>(matrices, default, dense);
            return;
        }

        for (int j = 0; j < x.Columns; j++)
        {
            var columns = new ReadOnlyMatrixSpan<T>(in x.ElementAt(0, j, 0), x.Rows, x.MatrixCount, x.RowStride, x.MatrixStride);
            RowSums(columns, destination.Column(j));
        }
    }

    internal static void Map<T, TOperator>(ReadOnlyTensorSpan<T> x, TensorSpan<T> destination, TOperator op)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulUnaryOperator<T>
    {
        RequireSameShape(x, destination);
        RequireElementwiseAlias(x, destination);
        if (x.TryAsMatrix(out var xm) && destination.AsReadOnly().TryAsMatrix(out var dm))
        {
            Map(xm, MatrixSpan<T>.Writable(dm), op);
            return;
        }

        var output = destination.AsReadOnly();
        for (int k = 0; k < x.MatrixCount; k++)
        {
            Map(x.Matrix(k), MatrixSpan<T>.Writable(output.Matrix(k)), op);
        }
    }

    internal static void Zip<T, TOperator>(ReadOnlyTensorSpan<T> x, ReadOnlyTensorSpan<T> y, TensorSpan<T> destination, TOperator op)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulBinaryOperator<T>
    {
        RequireSameShape(x, y);
        RequireSameShape(x, destination);
        RequireElementwiseAlias(x, destination);
        RequireElementwiseAlias(y, destination);
        if (x.TryAsMatrix(out var xm) && y.TryAsMatrix(out var ym) && destination.AsReadOnly().TryAsMatrix(out var dm))
        {
            Zip(xm, ym, MatrixSpan<T>.Writable(dm), op);
            return;
        }

        var output = destination.AsReadOnly();
        for (int k = 0; k < x.MatrixCount; k++)
        {
            Zip(x.Matrix(k), y.Matrix(k), MatrixSpan<T>.Writable(output.Matrix(k)), op);
        }
    }

    internal static void Scale<T>(ReadOnlyTensorSpan<T> x, T scalar, TensorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        RequireSameShape(x, destination);
        RequireElementwiseAlias(x, destination);
        if (x.TryAsMatrix(out var xm) && destination.AsReadOnly().TryAsMatrix(out var dm))
        {
            Scale(xm, scalar, MatrixSpan<T>.Writable(dm));
            return;
        }

        var output = destination.AsReadOnly();
        for (int k = 0; k < x.MatrixCount; k++)
        {
            Scale(x.Matrix(k), scalar, MatrixSpan<T>.Writable(output.Matrix(k)));
        }
    }

    internal static unsafe void Multiply<T>(ReadOnlyTensorSpan<T> a, ReadOnlyTensorSpan<T> b, TensorSpan<T> c)
        where T : unmanaged, INumberBase<T>
    {
        int m = a.Rows, k = a.Columns, n = b.Columns, count = a.MatrixCount;
        if (b.Rows != k)
        {
            throw new ArgumentException("Inner dimensions must agree.");
        }

        if (b.MatrixCount != count)
        {
            throw new ArgumentException("Both batches must have the same number of matrices.");
        }

        if (c.Rows != m || c.Columns != n || c.MatrixCount != count)
        {
            throw new ArgumentException("Destination shape must agree.", "destination");
        }

        RequireDisjoint(a, c);
        RequireDisjoint(b, c);
        if (count == 0)
        {
            return;
        }

        // 全行列に同一の左行列をブロードキャスト乗算する場合は、横に連結した単一の行列積へまとめる（計算経路が変わらないときのみ）。
        if ((a.MatrixStride == 0 || count == 1) && b.TryAsMatrix(out var bm) && SameColumnPath<T>(m, n, bm.Columns, k) && c.AsReadOnly().TryAsMatrix(out var cm))
        {
            Multiply(a.Matrix(0), bm, MatrixSpan<T>.Writable(cm));
            return;
        }

        fixed (T* pa = &Unsafe.AsRef(in a.Reference))
        fixed (T* pb = &Unsafe.AsRef(in b.Reference))
        fixed (T* pc = &Unsafe.AsRef(in c.AsReadOnly().Reference))
        {
            var work = new BatchMultiplyWork<T>((nint)pa, (nint)pb, (nint)pc, Layout.Of(a), Layout.Of(b), Layout.Of(c.AsReadOnly()), count);
            double workPerMatrix = (double)m * n * k;
            if (workPerMatrix >= SmallMatrixWork)
            {
                work.MultiplyRange(0, count);
                return;
            }

            int threads = Parallelism.Degree(count * workPerMatrix, Parallelism.Grain);
            if (threads <= 1)
            {
                using var scope = Parallelism.Sequential();
                work.MultiplyRange(0, count);
                return;
            }

            work.Chunks = Math.Min(count, 4 * threads);
            Parallelism.For(work.Chunks, threads, ref work);
        }
    }

    private struct BatchMultiplyWork<T>(nint a, nint b, nint c, Layout left, Layout right, Layout output, int count) : IParallelWork
        where T : unmanaged, INumberBase<T>
    {
        internal int Chunks;

        public readonly void Invoke(int chunk)
        {
            using var scope = Parallelism.Sequential();
            MultiplyRange((int)((long)count * chunk / Chunks), (int)((long)count * (chunk + 1) / Chunks));
        }

        internal readonly void MultiplyRange(int start, int end)
        {
            for (int i = start; i < end; i++)
            {
                // 形状と重なりはバッチ全体で検証済み。
                MultiplyUnchecked(left.Matrix<T>(a, i), right.Matrix<T>(b, i), MatrixSpan<T>.Writable(output.Matrix<T>(c, i)));
            }
        }
    }

    // 三角行列バッチの求解 A[k] X[k] = B[k]（右辺 1 列は TRSV 経路、2 列以上は TRSM 経路）。
    internal static unsafe void SolveTriangular<T>(ReadOnlyTensorSpan<T> a, ReadOnlyTensorSpan<T> b, TensorSpan<T> x, bool lower)
        where T : unmanaged, INumberBase<T>
    {
        RequireSystemShape(a, b, x);
        RequireDisjoint(a, x);
        CopyRightHandSide(b, x);
        int n = a.Rows;
        if (x.IsEmpty)
        {
            return;
        }

        fixed (T* pa = &Unsafe.AsRef(in a.Reference))
        fixed (T* px = &Unsafe.AsRef(in x.AsReadOnly().Reference))
        {
            var (sa, sx) = ((nint)pa, (nint)px);
            var (la, lx) = (Layout.Of(a), Layout.Of(x.AsReadOnly()));
            bool vector = VectorPath(x.Columns);
            ForEachMatrix(x.MatrixCount, (double)n * n * x.Columns / 2, (start, end) =>
            {
                for (int i = start; i < end; i++)
                {
                    SolveTriangularInPlace(la.Matrix<T>(sa, i), MatrixSpan<T>.Writable(lx.Matrix<T>(sx, i)), lower, vector);
                }
            });
        }
    }

    // 部分ピボット付き LU 分解によるバッチ求解（いずれかが特異なら最初の番号で ArgumentException を送出）。
    internal static unsafe void Solve(ReadOnlyTensorSpan<double> a, ReadOnlyTensorSpan<double> b, TensorSpan<double> x)
    {
        RequireSystemShape(a, b, x);
        RequireDisjoint(a, x);
        CopyRightHandSide(b, x);
        int n = a.Rows;
        int count = x.MatrixCount;
        if (n == 0 || count == 0)
        {
            return;
        }

        // 係数行列がブロードキャスト（MatrixStride == 0）なら 1 回だけ分解し、各行列を同一経路で解く（ビット一致）。
        if (a.MatrixStride == 0)
        {
            if (!TrySolveBroadcast(a.Matrix(0), x))
            {
                throw new ArgumentException("The matrix at index 0 is singular.", nameof(a));
            }

            return;
        }

        int singular = int.MaxValue;
        fixed (double* pa = &Unsafe.AsRef(in a.Reference))
        fixed (double* px = &Unsafe.AsRef(in x.AsReadOnly().Reference))
        {
            var (sa, sx) = ((nint)pa, (nint)px);
            var (la, lx) = (Layout.Of(a), Layout.Of(x.AsReadOnly()));
            bool vector = VectorPath(x.Columns);
            ForEachMatrix(count, (double)n * n * (2.0 / 3 * n + 2 * x.Columns), (start, end) =>
            {
                var lu = GC.AllocateUninitializedArray<double>(checked(n * n));
                var pivots = new int[n];
                for (int i = start; i < end; i++)
                {
                    CopyDense(la.Matrix<double>(sa, i), lu);
                    if (LuKernels.Factor(n, n, lu, n, pivots) != 0)
                    {
                        InterlockedMin(ref singular, i);
                        continue;
                    }

                    SolveFactoredInPlace(n, lu, pivots, MatrixSpan<double>.Writable(lx.Matrix<double>(sx, i)), vector);
                }
            });
        }

        if (singular != int.MaxValue)
        {
            throw new ArgumentException($"The matrix at index {singular} is singular.", nameof(a));
        }
    }

    // 共通係数 a を 1 回だけ LU 分解してバッチ全右辺を解く（特異なら何もせず false）。
    private static unsafe bool TrySolveBroadcast(ReadOnlyMatrixSpan<double> a, TensorSpan<double> x)
    {
        int n = a.Rows;
        var lu = GC.AllocateUninitializedArray<double>(checked(n * n));
        var pivots = new int[n];
        CopyDense(a, lu);
        if (LuKernels.Factor(n, n, lu, n, pivots) != 0)
        {
            return false;
        }

        fixed (double* px = &Unsafe.AsRef(in x.AsReadOnly().Reference))
        {
            nint sx = (nint)px;
            var lx = Layout.Of(x.AsReadOnly());
            bool vector = VectorPath(x.Columns);
            ForEachMatrix(x.MatrixCount, (double)n * n * 2 * x.Columns, (start, end) =>
            {
                for (int i = start; i < end; i++)
                {
                    SolveFactoredInPlace(n, lu, pivots, MatrixSpan<double>.Writable(lx.Matrix<double>(sx, i)), vector);
                }
            });
        }

        return true;
    }

    // 右辺 1 列は TRSV、2 列以上は TRSM を選ぶ（1 列での TRSM 呼び出しコスト回避）。
    internal static bool VectorPath(int columns) => columns == 1;

    // workPerMatrix < SmallMatrixWork なら行列範囲を並列化（内部逐次）、それ以上なら行列を順次実行し内部並列化に委ねる。
    internal static void ForEachMatrix(int count, double workPerMatrix, Action<int, int> body)
    {
        if (count == 0)
        {
            return;
        }

        if (workPerMatrix >= SmallMatrixWork)
        {
            body(0, count);
            return;
        }

        int threads = Parallelism.Degree(count * workPerMatrix, Parallelism.Grain);
        if (threads <= 1)
        {
            using var scope = Parallelism.Sequential();
            body(0, count);
            return;
        }

        int chunks = Math.Min(count, 4 * threads);
        Parallelism.For(chunks, threads, c =>
        {
            using var scope = Parallelism.Sequential();
            body((int)((long)count * c / chunks), (int)((long)count * (c + 1) / chunks));
        });
    }

    private static void InterlockedMin(ref int target, int value)
    {
        int current = Volatile.Read(ref target);
        while (value < current)
        {
            int previous = Interlocked.CompareExchange(ref target, value, current);
            if (previous == current)
            {
                return;
            }

            current = previous;
        }
    }

    // 並列区間内でビューを再構築するための形状・ストライド情報（ref struct はキャプチャ不可のため）。
    private readonly record struct Layout(int Rows, int Columns, int RowStride, int ColumnStride, int MatrixStride)
    {
        internal static Layout Of<T>(ReadOnlyTensorSpan<T> view) where T : unmanaged, INumberBase<T> =>
            new(view.Rows, view.Columns, view.RowStride, view.ColumnStride, view.MatrixStride);

        internal unsafe ReadOnlyMatrixSpan<T> Matrix<T>(nint origin, int index) where T : unmanaged, INumberBase<T> =>
            new(in Unsafe.AsRef<T>((T*)origin + (nint)index * MatrixStride), Rows, Columns, RowStride, ColumnStride);
    }

    private static void RequireSystemShape<T>(ReadOnlyTensorSpan<T> a, ReadOnlyTensorSpan<T> b, TensorSpan<T> x)
        where T : unmanaged, INumberBase<T>
    {
        if (a.Rows != a.Columns)
        {
            throw new ArgumentException("The coefficient matrices must be square.", nameof(a));
        }

        if (b.Rows != a.Rows)
        {
            throw new ArgumentException("The right-hand sides must have as many rows as the coefficient matrices.", nameof(b));
        }

        if (b.MatrixCount != a.MatrixCount)
        {
            throw new ArgumentException("Both batches must have the same number of matrices.", nameof(b));
        }

        if (x.Rows != b.Rows || x.Columns != b.Columns || x.MatrixCount != b.MatrixCount)
        {
            throw new ArgumentException("Destination shape must agree.", "destination");
        }
    }

    internal static void CopyRightHandSide<T>(ReadOnlyTensorSpan<T> b, TensorSpan<T> x)
        where T : unmanaged, INumberBase<T>
    {
        if (Overlaps(b, x, out bool exact))
        {
            if (!exact)
            {
                throw new ArgumentException("The destination partially overlaps the right-hand sides. Only an exact alias is allowed.", "destination");
            }

            return;
        }

        Copy(b, x);
    }

    // 分解の入力を出力先へコピーする（完全一致なら無処理、部分重複は例外）。
    internal static void CopyInput<T>(ReadOnlyTensorSpan<T> source, TensorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        if (Overlaps(source, destination, out bool exact))
        {
            if (!exact)
            {
                throw new ArgumentException("The destination partially overlaps the input. Only an exact alias is allowed.", nameof(destination));
            }

            return;
        }

        Copy(source, destination);
    }

    private static void RequireSameShape<T>(ReadOnlyTensorSpan<T> x, ReadOnlyTensorSpan<T> y) where T : unmanaged, INumberBase<T>
    {
        if (x.Rows != y.Rows || x.Columns != y.Columns || x.MatrixCount != y.MatrixCount)
        {
            throw new ArgumentException("Batch dimensions must agree.");
        }
    }

    private static void RequireElementwiseAlias<T>(ReadOnlyTensorSpan<T> input, TensorSpan<T> destination) where T : unmanaged, INumberBase<T>
    {
        if (Overlaps(input, destination, out bool exact) && !exact)
        {
            throw new ArgumentException("The destination partially overlaps an input. Only an exact alias is allowed.", nameof(destination));
        }
    }

    internal static void RequireDisjoint<T>(ReadOnlyTensorSpan<T> input, TensorSpan<T> destination) where T : unmanaged, INumberBase<T>
    {
        if (Overlaps(input, destination, out _))
        {
            throw new ArgumentException("The destination must not overlap the inputs.", nameof(destination));
        }
    }

    private static bool Overlaps<T>(ReadOnlyTensorSpan<T> input, TensorSpan<T> destination, out bool exact) where T : unmanaged, INumberBase<T>
    {
        var output = destination.AsReadOnly();
        return StridedBatchLayout.Overlaps(
            in input.Reference, input.Rows, input.Columns, input.MatrixCount, input.RowStride, input.ColumnStride, input.MatrixStride,
            in output.Reference, output.Rows, output.Columns, output.MatrixCount, output.RowStride, output.ColumnStride, output.MatrixStride,
            out exact);
    }
}
