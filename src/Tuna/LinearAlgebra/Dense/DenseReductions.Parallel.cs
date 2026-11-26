using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Tuna.LinearAlgebra;

internal static partial class DenseReductions
{
    // ---------- 長いベクトルの並列化 ----------
    //
    // 和の木は長さだけで決まるので、2 の冪個のブロック（ParallelChunk 要素）からなる区間の和は、その区間だけを足しても木の部分木と同じビットになる。
    // 固定チャンク単位（2^ParallelChunkLevel ブロック = 65536 要素）で並列縮約し、同一の加算木に結合するため
    // スレッド数によらず逐次計算とビット単位で一致する。
    private const int ParallelChunkLevel = 10;

    private const int ParallelChunk = Block << ParallelChunkLevel;

    // 並列化の最小要素数（これ未満では起動コストとキャッシュ移動が上回る）。
    private const int ParallelLength = 8 * ParallelChunk;

    private static int ParallelThreads(int length) => length < ParallelLength ? 1 : Parallelism.Degree(length, ParallelChunk);

    private static unsafe T SumUnary<T, TKernel>(ref readonly T x, int length, T parameter)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TKernel : ITermKernel<T>
    {
        int threads = ParallelThreads(length);
        if (threads <= 1)
        {
            return Sum<T, UnaryTerm<T, TKernel>>(length, new(in x, parameter));
        }

        fixed (T* p = &Unsafe.AsRef(in x))
        {
            return SumParallel<T, UnaryTerm<T, TKernel>, UnarySource<T, TKernel>>(length, new(p, parameter), threads);
        }
    }

    private static unsafe T SumProduct<T>(ref readonly T x, ref readonly T y, int length) where T : unmanaged, IFloatingPointIeee754<T>
    {
        int threads = ParallelThreads(length);
        if (threads <= 1)
        {
            return Sum<T, ProductTerm<T>>(length, new(in x, in y));
        }

        fixed (T* px = &Unsafe.AsRef(in x))
        fixed (T* py = &Unsafe.AsRef(in y))
        {
            return SumParallel<T, ProductTerm<T>, ProductSource<T>>(length, new(px, py), threads);
        }
    }

    [SkipLocalsInit]
    private static unsafe T SumParallel<T, TTerm, TSource>(int length, TSource source, int threads)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TTerm : ITerm<T>, allows ref struct
        where TSource : unmanaged, ITermSource<T, TTerm>
    {
        int chunks = length / ParallelChunk;
        T[] partial = ArrayPool<T>.Shared.Rent(chunks);
        try
        {
            fixed (T* pp = partial)
            {
                var work = new ChunkSumWork<T, TTerm, TSource>(source, pp);
                Parallelism.For(chunks, threads, ref work);
            }

            var tree = new SumTree<T>(stackalloc T[SumTree<T>.Capacity]);
            for (int c = 0; c < chunks; c++)
            {
                tree.Push(partial[c], ParallelChunkLevel);
            }

            nint done = (nint)chunks * ParallelChunk;
            AddBlocks(ref tree, length - (int)done, source.At(done));
            return tree.Result();
        }
        finally
        {
            ArrayPool<T>.Shared.Return(partial);
        }
    }

    [SkipLocalsInit]
    private static T ChunkSum<T, TTerm, TSource>(TSource source, int chunk)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TTerm : ITerm<T>, allows ref struct
        where TSource : unmanaged, ITermSource<T, TTerm>
    {
        var tree = new SumTree<T>(stackalloc T[SumTree<T>.Capacity]);
        AddBlocks(ref tree, ParallelChunk, source.At((nint)chunk * ParallelChunk));
        return tree.Result();
    }

    // 固定チャンクごとに指数和と最大値一致数を並列計算し、逐次実装と同一の加算木へ結合する。
    [SkipLocalsInit]
    private static unsafe T SumOfShiftedExpParallel<T>(ReadOnlySpan<T> x, T shift, int threads, out int count)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int chunks = x.Length / ParallelChunk;
        var partial = ArrayPool<ShiftedExpPartial<T>>.Shared.Rent(chunks);
        try
        {
            fixed (T* p = x)
            fixed (ShiftedExpPartial<T>* pp = partial)
            {
                var work = new ShiftedExpChunkWork<T>(p, shift, pp);
                Parallelism.For(chunks, threads, ref work);
            }

            var tree = new SumTree<T>(stackalloc T[SumTree<T>.Capacity]);
            count = 0;
            for (int c = 0; c < chunks; c++)
            {
                tree.Push(partial[c].Sum, ParallelChunkLevel);
                count += partial[c].Count;
            }

            // 末尾の端数区間は 1 チャンクにまとめず、ブロック単位で同じ木へ加算する。
            nint done = (nint)chunks * ParallelChunk;
            Span<T> terms = stackalloc T[TermChunk];
            AddShiftedExpBlocks(ref tree, x[(int)done..], shift, terms, out int tailCount);
            count += tailCount;
            return tree.Result();
        }
        finally
        {
            ArrayPool<ShiftedExpPartial<T>>.Shared.Return(partial);
        }
    }

    private readonly struct ShiftedExpPartial<T>(T sum, int count) where T : unmanaged
    {
        internal T Sum { get; } = sum;
        internal int Count { get; } = count;
    }

    private readonly unsafe struct ShiftedExpChunkWork<T>(T* x, T shift, ShiftedExpPartial<T>* partial) : IParallelWork
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        public void Invoke(int index)
        {
            T sum = SumOfShiftedExp(new ReadOnlySpan<T>(x + (nint)index * ParallelChunk, ParallelChunk), shift, out int count);
            partial[index] = new(sum, count);
        }
    }

    // 順序不変かつ initial に対して冪等な演算（最大・最小）をチャンクごとに並列縮約する。
    private static unsafe T ReduceParallel<T, TKernel>(ReadOnlySpan<T> x, T initial, int threads)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TKernel : IFoldKernel<T>
    {
        int chunks = (int)(((long)x.Length + ParallelChunk - 1) / ParallelChunk);
        T[] partial = ArrayPool<T>.Shared.Rent(chunks);
        try
        {
            fixed (T* p = x)
            fixed (T* pp = partial)
            {
                ReduceChunks<T, TKernel>(p, x.Length, initial, pp, chunks, threads);
            }

            T result = initial;
            for (int c = 0; c < chunks; c++)
            {
                result = TKernel.Invoke(result, partial[c]);
            }

            return result;
        }
        finally
        {
            ArrayPool<T>.Shared.Return(partial);
        }
    }

    private static unsafe void ReduceChunks<T, TKernel>(T* x, int length, T initial, T* partial, int chunks, int threads)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TKernel : IFoldKernel<T>
    {
        var work = new ReduceChunkWork<T, TKernel>(x, length, initial, partial);
        Parallelism.For(chunks, threads, ref work);
    }

    private readonly unsafe struct ChunkSumWork<T, TTerm, TSource>(TSource source, T* partial) : IParallelWork
        where T : unmanaged, IFloatingPointIeee754<T>
        where TTerm : ITerm<T>, allows ref struct
        where TSource : unmanaged, ITermSource<T, TTerm>
    {
        public void Invoke(int index) => partial[index] = ChunkSum<T, TTerm, TSource>(source, index);
    }

    private readonly unsafe struct ReduceChunkWork<T, TKernel>(T* x, int length, T initial, T* partial) : IParallelWork
        where T : unmanaged, IFloatingPointIeee754<T>
        where TKernel : IFoldKernel<T>
    {
        public void Invoke(int index)
        {
            nint start = (nint)index * ParallelChunk;
            partial[index] = Reduce<T, TKernel>(new ReadOnlySpan<T>(x + start, (int)Math.Min(ParallelChunk, length - start)), initial);
        }
    }

    // 並列ワーカーへ渡すためポインタで区間先頭を参照するファクトリ。
    private interface ITermSource<T, TTerm>
        where T : unmanaged
        where TTerm : ITerm<T>, allows ref struct
    {
        TTerm At(nint offset);
    }

    private readonly unsafe struct UnarySource<T, TKernel>(T* x, T parameter) : ITermSource<T, UnaryTerm<T, TKernel>>
        where T : unmanaged, IFloatingPointIeee754<T>
        where TKernel : ITermKernel<T>
    {
        public UnaryTerm<T, TKernel> At(nint offset) => new(in x[offset], parameter);
    }

    private readonly unsafe struct ProductSource<T>(T* x, T* y) : ITermSource<T, ProductTerm<T>>
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        public ProductTerm<T> At(nint offset) => new(in x[offset], in y[offset]);
    }
}
