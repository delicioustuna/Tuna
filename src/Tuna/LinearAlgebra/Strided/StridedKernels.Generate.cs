using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;

namespace Tuna.LinearAlgebra;

// 複数のスレッドから順不同に呼ぶ。作業の範囲ごとにコピーし、範囲の中では同じ値を参照で使い続ける（演算の状態は読むだけにする契約。
// 行 (i, j) の要素を生成するソース（並列実行時は外部状態の書き換え不可）。
internal interface IEntrySource<T> where T : unmanaged, INumberBase<T>
{
    T Evaluate(int row, int column);
}

internal struct IndexSource<T, TOperator>(TOperator op) : IEntrySource<T>
    where T : unmanaged, INumberBase<T>
    where TOperator : IStatefulIndexOperator<T>
{
    private TOperator op = op;

    public T Evaluate(int row, int column) => op.Invoke(row, column);
}

internal readonly struct StaticIndexOperator<T, TOperator> : IStatefulIndexOperator<T>
    where T : unmanaged, INumberBase<T>
    where TOperator : IIndexOperator<T>
{
    public T Invoke(int row, int column) => TOperator.Invoke(row, column);
}

internal readonly struct DelegateIndexOperator<T>(Func<int, int, T> op) : IStatefulIndexOperator<T>
    where T : unmanaged, INumberBase<T>
{
    private readonly Func<int, int, T> generate = op ?? throw new ArgumentNullException(nameof(op));

    public T Invoke(int row, int column) => generate(row, column);
}

internal readonly struct StaticVectorIndexOperator<T, TOperator> : IStatefulVectorIndexOperator<T>
    where T : unmanaged, INumberBase<T>
    where TOperator : IVectorIndexOperator<T>
{
    public T Invoke(int index) => TOperator.Invoke(index);
}

internal readonly struct DelegateVectorIndexOperator<T>(Func<int, T> op) : IStatefulVectorIndexOperator<T>
    where T : unmanaged, INumberBase<T>
{
    private readonly Func<int, T> generate = op ?? throw new ArgumentNullException(nameof(op));

    public T Invoke(int index) => generate(index);
}

internal readonly struct StaticTensorIndexOperator<T, TOperator> : IStatefulTensorIndexOperator<T>
    where T : unmanaged, INumberBase<T>
    where TOperator : ITensorIndexOperator<T>
{
    public T Invoke(ReadOnlySpan<int> indices) => TOperator.Invoke(indices);
}

internal readonly struct DelegateTensorIndexOperator<T>(Func<ReadOnlySpan<int>, T> op) : IStatefulTensorIndexOperator<T>
    where T : unmanaged, INumberBase<T>
{
    private readonly Func<ReadOnlySpan<int>, T> generate = op ?? throw new ArgumentNullException(nameof(op));

    public T Invoke(ReadOnlySpan<int> indices) => generate(indices);
}

// 2 点演算 op(x[:, i], y[:, j]) のソース（並列実行のため呼び出し元で fixed したポインタを保持）。
internal unsafe struct PairwiseSource<T, TOperator> : IEntrySource<T>
    where T : unmanaged, INumberBase<T>
    where TOperator : IStatefulPairwiseOperator<T>
{
    private readonly T* x;
    private readonly T* y;
    private readonly int dimension;
    private readonly int xRowStride;
    private readonly int xColumnStride;
    private readonly int yRowStride;
    private readonly int yColumnStride;
    private TOperator op;

    public PairwiseSource(T* x, int xRowStride, int xColumnStride, T* y, int yRowStride, int yColumnStride, int dimension, TOperator op)
    {
        this.x = x;
        this.y = y;
        this.dimension = dimension;
        this.xRowStride = xRowStride;
        this.xColumnStride = xColumnStride;
        this.yRowStride = yRowStride;
        this.yColumnStride = yColumnStride;
        this.op = op;
    }

    public T Evaluate(int row, int column) => op.Invoke(
        new ReadOnlyVectorSpan<T>(in Unsafe.AsRef<T>(x + (nint)row * xColumnStride), dimension, xRowStride),
        new ReadOnlyVectorSpan<T>(in Unsafe.AsRef<T>(y + (nint)column * yColumnStride), dimension, yRowStride));
}

// 行列を要素ごとに生成する計算本体（XMath.Pairwise・PairwiseSymmetric・Generate・GenerateBand・GenerateSymmetric）。
// 各要素は他の要素と独立に1回ずつ求めるので、列の分け方やスレッド数によらず結果は同じ。
internal static partial class StridedKernels
{
    internal static void Generate<T, TOperator>(VectorSpan<T> destination, TOperator op)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulVectorIndexOperator<T>
    {
        var column = destination.AsMatrix();
        for (int i = 0; i < column.Rows; i++)
        {
            Element(column, i, 0) = op.Invoke(i);
        }
    }

    internal static void Generate<T, TOperator>(TensorSpan<T> destination, TOperator op)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulTensorIndexOperator<T>
    {
        if (destination.IsEmpty)
        {
            return;
        }

        Span<int> indices = stackalloc int[destination.Rank];
        for (int k = 0; k < destination.MatrixCount; k++)
        {
            indices[2] = k;
            var matrix = destination.GetMatrix(k);
            for (int j = 0; j < matrix.Columns; j++)
            {
                indices[1] = j;
                for (int i = 0; i < matrix.Rows; i++)
                {
                    indices[0] = i;
                    Element(matrix, i, j) = op.Invoke(indices);
                }
            }
        }
    }

    // 生成 1 要素あたりの計算量目安（積和換算。未知のコールバックに対して保守的に並列化閾値を決める）。
    internal const double GeneratedElementWork = 64;

    // 列ペア演算 destination[i, j] = op(x[:, i], y[:, j])（symmetric 時は下三角のみ計算して上三角へ複製）。
    internal static unsafe void Pairwise<T, TOperator>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, MatrixSpan<T> destination, TOperator op, bool symmetric = false)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulPairwiseOperator<T>
    {
        if (x.Rows != y.Rows)
        {
            throw new ArgumentException("Both inputs must have the same number of rows (the dimension of each point).");
        }

        if (destination.Rows != x.Columns || destination.Columns != y.Columns)
        {
            throw new ArgumentException("Destination shape must agree.", nameof(destination));
        }

        RequireDisjoint(x, destination);
        RequireDisjoint(y, destination);
        if (destination.Rows == 0 || destination.Columns == 0)
        {
            return;
        }

        fixed (T* px = &Unsafe.AsRef(in x.Reference))
        fixed (T* py = &Unsafe.AsRef(in y.Reference))
        {
            var source = new PairwiseSource<T, TOperator>(px, x.RowStride, x.ColumnStride, py, y.RowStride, y.ColumnStride, x.Rows, op);
            int n = destination.Rows;
            Generate<T, PairwiseSource<T, TOperator>>(destination, n - 1, destination.Columns - 1, symmetric,
                4.0 * x.Rows + GeneratedElementWork, source);
        }
    }

    // 帯（-lowerBandwidth <= j - i <= upperBandwidth）の要素を生成し帯外を 0 埋めする。
    // 対称時は列 j の担当が (i, j) と (j, i) の両方を書き込むため、並列実行でも書き込み領域は重複しない。
    internal static unsafe void Generate<T, TSource>(
        MatrixSpan<T> destination, int lowerBandwidth, int upperBandwidth, bool symmetric, double elementWork, TSource source)
        where T : unmanaged, INumberBase<T>
        where TSource : IEntrySource<T>
    {
        int rows = destination.Rows, columns = destination.Columns;
        if (rows == 0 || columns == 0)
        {
            return;
        }

        int lower = Math.Min(lowerBandwidth, rows - 1);
        int upper = symmetric ? lower : Math.Min(upperBandwidth, columns - 1);
        double evaluated = 0;
        for (int j = 0; j < columns; j++)
        {
            evaluated += Math.Max(0, Math.Min(rows - 1, j + lower) - (symmetric ? j : Math.Max(0, j - upper)) + 1);
        }

        int threads = Parallelism.Degree(evaluated * elementWork, Parallelism.Grain);
        if (threads <= 1)
        {
            GenerateColumns(destination, 0, columns, lower, upper, symmetric, ref source);
            return;
        }

        var view = destination.AsReadOnly();
        int rowStride = view.RowStride, columnStride = view.ColumnStride;
        int chunks = Math.Min(columns, 8 * threads);
        // 演算が投げた例外は、逐次の場合と同じく包まずに伝える（最初に捕まえた1つ。残りの範囲は求めない）
        ExceptionDispatchInfo? failure = null;
        fixed (T* p = &Unsafe.AsRef(in view.Reference))
        {
            nint origin = (nint)p;
            Parallelism.For(chunks, threads, c =>
            {
                if (Volatile.Read(ref failure) is not null)
                {
                    return;
                }

                try
                {
                    var local = source;
                    var d = new MatrixSpan<T>(ref Unsafe.AsRef<T>((T*)origin), rows, columns, rowStride, columnStride);
                    GenerateColumns(d, (int)((long)columns * c / chunks), (int)((long)columns * (c + 1) / chunks), lower, upper, symmetric, ref local);
                }
                catch (Exception e)
                {
                    Interlocked.CompareExchange(ref failure, ExceptionDispatchInfo.Capture(e), null);
                }
            });
        }

        failure?.Throw();
    }

    private static void GenerateColumns<T, TSource>(MatrixSpan<T> d, int c0, int c1, int lower, int upper, bool symmetric, ref TSource source)
        where T : unmanaged, INumberBase<T>
        where TSource : IEntrySource<T>
    {
        int rows = d.Rows;
        for (int j = c0; j < c1; j++)
        {
            // 帯の上の外 [0, top)、（対称なら写しで埋まる [top, j)）、求める範囲 [first, last]、帯の下の外 (last, rows)
            int top = Math.Min(rows, Math.Max(0, j - (symmetric ? lower : upper)));
            int first = symmetric ? j : top;
            int last = Math.Min(rows - 1, j + lower);
            for (int i = 0; i < top; i++)
            {
                Element(d, i, j) = T.Zero;
            }

            for (int i = first; i <= last; i++)
            {
                T v = source.Evaluate(i, j);
                Element(d, i, j) = v;
                if (symmetric && i != j)
                {
                    Element(d, j, i) = v;
                }
            }

            for (int i = Math.Max(first, last + 1); i < rows; i++)
            {
                Element(d, i, j) = T.Zero;
            }
        }
    }
}
