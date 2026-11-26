using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Tuna.LinearAlgebra;

// BLAS Level 1 の並列実行。集約は長さのみで決まる固定チャンク順で加算し、スレッド数によらずビット一致を保つ。
internal static unsafe class ParallelLevel1<T>
    where T : unmanaged, INumberBase<T>
{
    private const double Grain = 32 * 1024; // 1 スレッドあたりの最小要素数（8 バイト換算）

    private static int Threads(long n) => Parallelism.PlannedDegree((double)n * Unsafe.SizeOf<T>() / 8.0, Grain);

    // スレッド数設定に依存せず長さのみで決まるチャンク化対象判定。
    private static bool IsLong(long n) => (double)n * Unsafe.SizeOf<T>() / 8.0 >= 2 * Grain;

    private static int Chunks(int n) => ChunkedSum.Chunks((long)n * Unsafe.SizeOf<T>());

    // [0, n) を 64 要素境界で parts 分割した t 番目の範囲。
    private static (int Lo, int Hi) Range(int n, int parts, int t)
    {
        int lo = t == 0 ? 0 : Math.Min(n, (int)((long)n * t / parts) & ~63);
        int hi = t == parts - 1 ? n : Math.Min(n, (int)((long)n * (t + 1) / parts) & ~63);
        return (lo, hi);
    }

    internal static T Dot(ReadOnlySpan<T> x, ReadOnlySpan<T> y, bool conjugateX)
    {
        int n = x.Length;
        if (!IsLong(2L * n))
        {
            return VectorOps.Dot(x, y, conjugateX);
        }

        fixed (T* px = x)
        fixed (T* py = y)
        {
            return ChunkedSum.Run<T, DotTerm>(n, Chunks(n), Threads(2L * n), new DotTerm(px, py, conjugateX));
        }
    }

    private readonly struct DotTerm(T* x, T* y, bool conjugateX) : IChunkTerm<T>
    {
        public T Sum(int start, int length) =>
            VectorOps.Dot(new ReadOnlySpan<T>(x + start, length), new ReadOnlySpan<T>(y + start, length), conjugateX);
    }

    internal static void Axpy(T a, ReadOnlySpan<T> x, Span<T> y)
    {
        int n = x.Length;
        int threads = Threads(3L * n);
        if (threads <= 1)
        {
            VectorOps.Axpy(a, x, y);
            return;
        }

        fixed (T* px = x)
        fixed (T* py = y)
        {
            AxpyParallel(n, threads, a, px, py);
        }
    }

    private static void AxpyParallel(int n, int threads, T a, T* xx, T* yy)
    {
        var work = new AxpyWork(n, threads, a, xx, yy);
        Parallelism.ForStatic(threads, ref work);
    }

    private readonly struct AxpyWork(int n, int threads, T a, T* x, T* y) : IParallelWork
    {
        public void Invoke(int t)
        {
            var (lo, hi) = Range(n, threads, t);
            VectorOps.Axpy(a, new ReadOnlySpan<T>(x + lo, hi - lo), new Span<T>(y + lo, hi - lo));
        }
    }

    internal static void Scale(T a, Span<T> x)
    {
        int n = x.Length;
        int threads = Threads(2L * n);
        if (threads <= 1)
        {
            VectorOps.Scale(a, x);
            return;
        }

        fixed (T* px = x)
        {
            ScaleParallel(n, threads, a, px);
        }
    }

    private static void ScaleParallel(int n, int threads, T a, T* xx)
    {
        var work = new ScaleWork(n, threads, a, xx);
        Parallelism.ForStatic(threads, ref work);
    }

    private readonly struct ScaleWork(int n, int threads, T a, T* x) : IParallelWork
    {
        public void Invoke(int t)
        {
            var (lo, hi) = Range(n, threads, t);
            VectorOps.Scale(a, new Span<T>(x + lo, hi - lo));
        }
    }

    internal static T SumAbs(ReadOnlySpan<T> x)
    {
        int n = x.Length;
        if (!IsLong(n))
        {
            return RealVectorOps.SumAbs(x);
        }

        fixed (T* px = x)
        {
            return ChunkedSum.Run<T, SumAbsTerm>(n, Chunks(n), Threads(n), new SumAbsTerm(px));
        }
    }

    private readonly struct SumAbsTerm(T* x) : IChunkTerm<T>
    {
        public T Sum(int start, int length) => RealVectorOps.SumAbs(new ReadOnlySpan<T>(x + start, length));
    }

    internal static void Rotate(Span<T> x, Span<T> y, T c, T s)
    {
        int n = x.Length;
        int threads = Threads(4L * n);
        if (threads <= 1)
        {
            RealVectorOps.Rotate(x, y, c, s);
            return;
        }

        fixed (T* px = x)
        fixed (T* py = y)
        {
            RotateParallel(n, threads, px, py, c, s);
        }
    }

    private static void RotateParallel(int n, int threads, T* xx, T* yy, T c, T s)
    {
        var work = new RotateWork(n, threads, xx, yy, c, s);
        Parallelism.ForStatic(threads, ref work);
    }

    private readonly struct RotateWork(int n, int threads, T* x, T* y, T c, T s) : IParallelWork
    {
        public void Invoke(int t)
        {
            var (lo, hi) = Range(n, threads, t);
            RealVectorOps.Rotate(new Span<T>(x + lo, hi - lo), new Span<T>(y + lo, hi - lo), c, s);
        }
    }

    internal static void Swap(Span<T> x, Span<T> y)
    {
        int n = x.Length;
        int threads = Threads(4L * n);
        if (threads <= 1)
        {
            BlasCheck.Swap(x, y);
            return;
        }

        fixed (T* px = x)
        fixed (T* py = y)
        {
            SwapParallel(n, threads, px, py);
        }
    }

    private static void SwapParallel(int n, int threads, T* xx, T* yy)
    {
        var work = new SwapWork(n, threads, xx, yy);
        Parallelism.ForStatic(threads, ref work);
    }

    private readonly struct SwapWork(int n, int threads, T* x, T* y) : IParallelWork
    {
        public void Invoke(int t)
        {
            var (lo, hi) = Range(n, threads, t);
            BlasCheck.Swap(new Span<T>(x + lo, hi - lo), new Span<T>(y + lo, hi - lo));
        }
    }

    internal static void Copy(ReadOnlySpan<T> x, Span<T> y)
    {
        int n = x.Length;
        int threads = Threads(2L * n);
        if (threads <= 1)
        {
            x.CopyTo(y);
            return;
        }

        fixed (T* px = x)
        fixed (T* py = y)
        {
            CopyParallel(n, threads, px, py);
        }
    }

    private static void CopyParallel(int n, int threads, T* xx, T* yy)
    {
        var work = new CopyWork(n, threads, xx, yy);
        Parallelism.ForStatic(threads, ref work);
    }

    private readonly struct CopyWork(int n, int threads, T* x, T* y) : IParallelWork
    {
        public void Invoke(int t)
        {
            var (lo, hi) = Range(n, threads, t);
            new ReadOnlySpan<T>(x + lo, hi - lo).CopyTo(new Span<T>(y + lo, hi - lo));
        }
    }
}

// 並列ノルム計算（長さのみで決まる固定チャンク順で平方和を加算し、スレッド数によらずビット一致）。
internal static unsafe class ParallelLevel1Norms
{
    private const double Grain = 32 * 1024;

    // 倍精度 ||x||_2（オーバーフロー・アンダーフロー範囲は Level1Kernels.Norm2 側でスケーリング再計算）。
    internal static double Norm2(ReadOnlySpan<double> x)
    {
        int n = x.Length;
        if (n < 2 * Grain)
        {
            return Level1Kernels.Norm2(x);
        }

        double sumOfSquares;
        fixed (double* px = x)
        {
            sumOfSquares = ChunkedSum.Run<double, SumOfSquaresTerm>(
                n, ChunkedSum.Chunks((long)n * sizeof(double)), Parallelism.PlannedDegree(n, Grain), new SumOfSquaresTerm(px));
        }

        return Level1Kernels.Norm2(x, sumOfSquares);
    }

    // 単精度 ||x||_2（倍精度で累積）。
    internal static float Norm2(ReadOnlySpan<float> x)
    {
        int n = x.Length;
        if (n / 2.0 < 2 * Grain)
        {
            return BlasNorms.Norm2(x);
        }

        fixed (float* px = x)
        {
            return (float)Math.Sqrt(ChunkedSum.Run<double, SingleSumOfSquaresTerm>(
                n, ChunkedSum.Chunks((long)n * sizeof(float)), Parallelism.PlannedDegree(n / 2.0, Grain), new SingleSumOfSquaresTerm(px)));
        }
    }

    internal static double Norm2(ReadOnlySpan<Complex> x) =>
        Norm2(MemoryMarshal.CreateReadOnlySpan(ref Unsafe.As<Complex, double>(ref MemoryMarshal.GetReference(x)), 2 * x.Length));

    private readonly struct SumOfSquaresTerm(double* x) : IChunkTerm<double>
    {
        public double Sum(int start, int length) => Level1Kernels.SumOfSquares(new ReadOnlySpan<double>(x + start, length));
    }

    private readonly struct SingleSumOfSquaresTerm(float* x) : IChunkTerm<double>
    {
        public double Sum(int start, int length) => BlasNorms.SumOfSquares(new ReadOnlySpan<float>(x + start, length));
    }
}

internal interface IChunkTerm<TResult>
{
    TResult Sum(int start, int length);
}

// 長さのみで決まる固定チャンクに分割し、区間順に加算することでスレッド数非依存な集約を行う。
internal static unsafe class ChunkedSum
{
    private const int ChunkBytes = 16 * 1024; // チャンク最小サイズ（MaxChunks 到達までは 2 倍未満）
    private const int MaxChunks = 768;        // 部分和を stackalloc に収める上限

    // 3 * 2^k 個にすることで 2/3/4/6/8/12/16 スレッドへ均等割り当てできる。
    internal static int Chunks(long bytes)
    {
        int chunks = 3;
        while (chunks < MaxChunks && bytes / chunks >= 2 * ChunkBytes)
        {
            chunks *= 2;
        }

        return chunks;
    }

    // c 番目のチャンク先頭（16 要素境界に切り下げ）。
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Start(int length, int chunks, int c) => c == chunks ? length : (int)((long)length * c / chunks) & ~15;

    // term が参照する領域は完了まで呼び出し元で固定すること。
    [SkipLocalsInit]
    internal static TResult Run<TResult, TTerm>(int length, int chunks, int threads, TTerm term)
        where TResult : unmanaged, INumberBase<TResult>
        where TTerm : unmanaged, IChunkTerm<TResult>
    {
        TResult sum;
        if (threads <= 1)
        {
            int end = Start(length, chunks, 1);
            sum = term.Sum(0, end);
            for (int c = 1; c < chunks; c++)
            {
                int start = end;
                end = Start(length, chunks, c + 1);
                sum += term.Sum(start, end - start);
            }

            return sum;
        }

        // 同じ区間を毎回同じスレッドに割り当てる（繰り返し使うベクトルが各コアのキャッシュに残る）
        TResult* partial = stackalloc TResult[chunks];
        int parts = Math.Min(threads, chunks);
        var work = new Work<TResult, TTerm>(term, partial, length, chunks, parts);
        Parallelism.ForStatic(parts, ref work);
        sum = partial[0];
        for (int c = 1; c < chunks; c++)
        {
            sum += partial[c];
        }

        return sum;
    }

    private readonly struct Work<TResult, TTerm>(TTerm term, TResult* partial, int length, int chunks, int parts) : IParallelWork
        where TResult : unmanaged, INumberBase<TResult>
        where TTerm : unmanaged, IChunkTerm<TResult>
    {
        public void Invoke(int index)
        {
            int last = chunks * (index + 1) / parts;
            for (int c = chunks * index / parts; c < last; c++)
            {
                int start = Start(length, chunks, c);
                partial[c] = term.Sum(start, Start(length, chunks, c + 1) - start);
            }
        }
    }
}
