namespace Tuna.LinearAlgebra;

// 分解パネル向けの並列行列ベクトル積（BLAS dgemv / dsymv 相当）。
internal static class MatrixVector
{
    // 1 スレッドあたりの最小要素数（約 8us 相当）。
    internal const double Grain = 16 * 1024;

    // y = A x（行分割で並列化）。
    internal static unsafe void Multiply(int m, int n, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> x, Span<double> y)
    {
        int threads = Parallelism.Degree((double)m * n, Grain);
        if (threads <= 1 || m < 64 * threads)
        {
            MultiplySequential(m, n, a, lda, x, y);
            return;
        }

        // 各スレッドの行数を8の倍数にそろえ、キャッシュラインを共有しないようにする
        int rows = (m / threads + 7) / 8 * 8;
        int blocks = (m + rows - 1) / rows;
        fixed (double* pa = a)
        fixed (double* px = x)
        fixed (double* py = y)
        {
            var work = new MultiplyWork(m, n, rows, new SharedSpan(pa, a.Length), lda, new SharedSpan(px, x.Length), new SharedSpan(py, y.Length));
            Parallelism.For(blocks, threads, ref work);
        }
    }

    private readonly struct MultiplyWork(int m, int n, int rows, SharedSpan a, int lda, SharedSpan x, SharedSpan y) : IParallelWork
    {
        public void Invoke(int t)
        {
            int i0 = t * rows;
            MultiplySequential(Math.Min(rows, m - i0), n, a.Span[i0..], lda, x.Span, y.Span[i0..]);
        }
    }

    // y = A^T x（列分割で並列化）。
    internal static unsafe void MultiplyTransposed(int m, int n, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> x, Span<double> y)
    {
        int threads = Parallelism.Degree((double)m * n, Grain);
        if (threads <= 1 || n < 16 * threads)
        {
            MultiplyTransposedSequential(m, n, a, lda, x, y);
            return;
        }

        int cols = (n / threads + 7) / 8 * 8;
        int blocks = (n + cols - 1) / cols;
        fixed (double* pa = a)
        fixed (double* px = x)
        fixed (double* py = y)
        {
            var work = new MultiplyTransposedWork(m, n, cols, new SharedSpan(pa, a.Length), lda, new SharedSpan(px, x.Length), new SharedSpan(py, y.Length));
            Parallelism.For(blocks, threads, ref work);
        }
    }

    private readonly struct MultiplyTransposedWork(int m, int n, int cols, SharedSpan a, int lda, SharedSpan x, SharedSpan y) : IParallelWork
    {
        public void Invoke(int t)
        {
            int j0 = t * cols;
            MultiplyTransposedSequential(m, Math.Min(cols, n - j0), a.Span[(j0 * lda)..], lda, x.Span, y.Span[j0..]);
        }
    }

    // y = α A x（下三角格納の n×n 対称行列。列範囲ごとの部分和を次数依存の固定順で合算）。
    internal static unsafe void SymmetricMultiply(int n, double alpha, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> x, Span<double> y)
    {
        const int Columns = 256;
        if (n <= Columns)
        {
            SymmetricMultiplySequential(0, n, n, alpha, a, lda, x, y);
            return;
        }

        // 部分和の境界と合計順を次数だけで決め、逐次でも同じ順で足す。
        int parts = 1 + (n - 1) / Columns;
        int threads = Parallelism.Degree((double)n * n / 2, Grain);
        double[] partial = WorkArrayPool<double>.Rent(checked((parts - 1) * n));
        try
        {
            fixed (double* pa = a)
            fixed (double* px = x)
            fixed (double* py = y)
            fixed (double* pp = partial)
            {
                var work = new SymmetricWork(
                    n, Columns, alpha, new SharedSpan(pa, a.Length), lda, new SharedSpan(px, x.Length), new SharedSpan(py, y.Length), new SharedSpan(pp, partial.Length));
                Parallelism.For(parts, threads, ref work);
            }

            for (int t = 1; t < parts; t++)
            {
                int start = t * Columns;
                Level1Kernels.Axpy(1.0, partial.AsSpan((t - 1) * n + start, n - start), y[start..n]);
            }
        }
        finally
        {
            WorkArrayPool<double>.Return(partial);
        }
    }

    private readonly struct SymmetricWork(int n, int columns, double alpha, SharedSpan a, int lda, SharedSpan x, SharedSpan y, SharedSpan partial) : IParallelWork
    {
        public void Invoke(int t)
        {
            var dest = t == 0 ? y.Span[..n] : partial.Span.Slice((t - 1) * n, n);
            SymmetricMultiplySequential(t * columns, Math.Min((t + 1) * columns, n), n, alpha, a.Span, lda, x.Span, dest);
        }
    }

    internal static void MultiplySequential(int m, int n, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> x, Span<double> y)
    {
        y = y[..m];
        y.Clear();
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            Level1Kernels.AddColumns4(m, a[(j * lda)..], lda, x[j], x[j + 1], x[j + 2], x[j + 3], y);
        }

        for (; j < n; j++)
        {
            Level1Kernels.Axpy(x[j], a.Slice(j * lda, m), y);
        }
    }

    internal static void MultiplyTransposedSequential(int m, int n, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> x, Span<double> y)
    {
        x = x[..m];
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            Level1Kernels.Dot4(m, a[(j * lda)..], lda, x, out y[j], out y[j + 1], out y[j + 2], out y[j + 3]);
        }

        for (; j < n; j++)
        {
            y[j] = Level1Kernels.Dot(a.Slice(j * lda, m), x);
        }
    }

    // 列 [j0, j1) の寄与のみ y[j0:n] に書き込む（j0 は 4 の倍数、事前にゼロクリア済み）。
    internal static void SymmetricMultiplySequential(int j0, int j1, int n, double alpha, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> x, Span<double> y)
    {
        y = y[..n];
        y[j0..].Clear();
        int j = j0;

        // 4列ずつ: 対角ブロック（4×4）はスカラーで、その下の行は4列まとめて読む
        for (; j + 4 <= j1; j += 4)
        {
            double x0 = x[j], x1 = x[j + 1], x2 = x[j + 2], x3 = x[j + 3];
            double a00 = a[j + j * lda], a10 = a[j + 1 + j * lda], a20 = a[j + 2 + j * lda], a30 = a[j + 3 + j * lda];
            double a11 = a[j + 1 + (j + 1) * lda], a21 = a[j + 2 + (j + 1) * lda], a31 = a[j + 3 + (j + 1) * lda];
            double a22 = a[j + 2 + (j + 2) * lda], a32 = a[j + 3 + (j + 2) * lda];
            double a33 = a[j + 3 + (j + 3) * lda];
            double y0 = a00 * x0 + a10 * x1 + a20 * x2 + a30 * x3;
            double y1 = a10 * x0 + a11 * x1 + a21 * x2 + a31 * x3;
            double y2 = a20 * x0 + a21 * x1 + a22 * x2 + a32 * x3;
            double y3 = a30 * x0 + a31 * x1 + a32 * x2 + a33 * x3;
            int below = n - j - 4;
            double d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            if (below > 0)
            {
                Level1Kernels.DotAxpy4(
                    below, a[(j + 4 + j * lda)..], lda, x[(j + 4)..], alpha * x0, alpha * x1, alpha * x2, alpha * x3, y[(j + 4)..],
                    out d0, out d1, out d2, out d3);
            }

            y[j] += alpha * (y0 + d0);
            y[j + 1] += alpha * (y1 + d1);
            y[j + 2] += alpha * (y2 + d2);
            y[j + 3] += alpha * (y3 + d3);
        }

        for (; j < j1; j++)
        {
            double t1 = alpha * x[j];
            var col = a.Slice(j + 1 + j * lda, n - j - 1);
            double t2 = Level1Kernels.DotAxpy(col, x[(j + 1)..], t1, y[(j + 1)..]);
            y[j] += t1 * a[j + j * lda] + alpha * t2;
        }
    }
}
