namespace Tuna.LinearAlgebra;

// 二段階縮約の第 2 段（帯行列のバルジ追跡）に共通するパイプライン並列実行と反射の保存・ブロック逆変換。
internal static class BulgeChasing
{
    private const double PipelineGrain = 64.0 * 1024 * 1024; // 並列化閾値（n^2 * b 換算）

    internal delegate void SweepBody(int sweep, Span<double> work, int[]? progress);

    internal static int KernelCount(int n, int b, int s) => s > n - 2 ? 0 : (n - 2 - s) / b + 1;

    // スイープ s の k 番目処理はスイープ s-1 の k+2 番目完了後なら領域が重ならない。
    // スイープを番号順に取得してパイプライン並列化するため、逐次フォールバック時でもデッドロックせず演算順序も一致する。
    internal static void Run(int n, int b, int sweeps, int workLength, SweepBody body)
    {
        if (sweeps <= 0)
        {
            return;
        }

        int threads = Parallelism.Degree((double)n * n * b, PipelineGrain);
        threads = Math.Min(threads, sweeps / (2 * b));
        if (threads <= 1)
        {
            double[] rented = WorkArrayPool<double>.Rent(workLength);
            for (int s = 0; s < sweeps; s++)
            {
                body(s, rented.AsSpan(0, workLength), null);
            }

            WorkArrayPool<double>.Return(rented);
            return;
        }

        var progress = new int[sweeps];
        int[] next = [0];
        double[] buffers = WorkArrayPool<double>.Rent(checked(threads * workLength));
        try
        {
            Parallelism.For(threads, threads, worker =>
            {
                var work = buffers.AsSpan(worker * workLength, workLength);
                for (int s = Interlocked.Increment(ref next[0]) - 1; s < sweeps; s = Interlocked.Increment(ref next[0]) - 1)
                {
                    body(s, work, progress);
                }
            });
        }
        finally
        {
            WorkArrayPool<double>.Return(buffers);
        }
    }

    // 直前スイープが kernel + 2 番目まで完了するのを待機する。
    internal static void WaitFor(int[]? progress, int n, int b, int s, int kernel)
    {
        if (progress is null || s == 0)
        {
            return;
        }

        int needed = Math.Min(kernel + 3, KernelCount(n, b, s - 1));
        if (Volatile.Read(ref progress[s - 1]) >= needed)
        {
            return;
        }

        var spinner = default(SpinWait);
        while (Volatile.Read(ref progress[s - 1]) < needed)
        {
            spinner.SpinOnce(sleep1Threshold: -1);
        }
    }

    internal static void Publish(int[]? progress, int s, int done)
    {
        if (progress is not null)
        {
            Volatile.Write(ref progress[s], done);
        }
    }
}

// バルジ追跡で生成した Householder 反射（スイープ s の k 番目は行 s + 1 + kb から長さ b 以下）を保持し、ブロック適用する。
internal sealed class BandReflectors
{
    private const int GroupSize = 64; // 1 つのブロック反射にまとめるスイープ数

    private readonly long[] offsets;
    private readonly double[] v;
    private readonly double[] tau;

    internal BandReflectors(int n, int b, int sweeps)
    {
        N = n;
        B = b;
        Sweeps = sweeps;
        offsets = new long[sweeps + 1];
        for (int s = 0; s < sweeps; s++)
        {
            offsets[s + 1] = offsets[s] + BulgeChasing.KernelCount(n, b, s);
        }

        v = new double[offsets[sweeps] * b];
        tau = new double[offsets[sweeps]];
    }

    internal int N { get; }

    internal int B { get; }

    internal int Sweeps { get; }

    internal void Store(int s, int kernel, ReadOnlySpan<double> u, double t)
    {
        long index = offsets[s] + kernel;
        var dest = v.AsSpan((int)(index * B), B);
        u.CopyTo(dest);
        dest[u.Length..].Clear();
        tau[index] = t;
    }

    // Q = H_00 H_01 ... を C に左から適用する。連続 GroupSize 本のスイープをブロック反射 I - VTV^T にまとめ、
    // 交換不可能な依存関係を守るためグループは降順、グループ内は上ブロックから順に適用する（列分割のため 1 スレッドと一致）。
    internal unsafe void Apply(int columns, Span<double> c, int ldc)
    {
        int n = N;
        int b = B;
        if (Sweeps <= 0 || columns <= 0)
        {
            return;
        }

        int g = GroupSize;
        int h = b + g - 1;
        int maxKernels = BulgeChasing.KernelCount(n, b, 0);
        int threads = Parallelism.Degree((double)n * n * columns, 4 * Parallelism.Grain);
        threads = Math.Max(1, Math.Min(threads, columns / 16));
        var bounds = new int[threads + 1];
        for (int t = 0; t < threads; t++)
        {
            bounds[t] = (int)((long)columns * t / threads) & ~7;
        }

        bounds[threads] = columns;
        double[] vAll = WorkArrayPool<double>.Rent(checked(maxKernels * h * g));
        double[] yAll = WorkArrayPool<double>.Rent(checked(maxKernels * h * g));
        double[] tAll = WorkArrayPool<double>.Rent(checked(maxKernels * g * g));
        double[] tauAll = WorkArrayPool<double>.Rent(checked(maxKernels * g));
        double[] wAll = WorkArrayPool<double>.Rent(checked(threads * g * columns));
        try
        {
            fixed (double* pc = c)
            {
                var sc = new SharedSpan(pc, c.Length);
                int groups = (Sweeps + g - 1) / g;
                for (int group = groups - 1; group >= 0; group--)
                {
                    int s0 = group * g;
                    int members = Math.Min(g, Sweeps - s0);
                    int kernels = BulgeChasing.KernelCount(n, b, s0);

                    // 各位置のブロック反射の V と、各スレッドで T を掛け直さないよう Y = VT を作る
                    Parallelism.For(kernels, Parallelism.Degree((double)kernels * h * g * g, 4 * Parallelism.Grain), k =>
                    {
                        int r0 = s0 + 1 + k * b;
                        int rows = Math.Min(h, n - r0);
                        int width = Math.Min(members, rows);
                        var vk = vAll.AsSpan(k * h * g, rows * width);
                        var tauk = tauAll.AsSpan(k * g, width);
                        vk.Clear();
                        for (int j = 0; j < width; j++)
                        {
                            int s = s0 + j;
                            if (k >= BulgeChasing.KernelCount(n, b, s))
                            {
                                tauk[j] = 0;
                                vk[j + j * rows] = 1;
                                continue;
                            }

                            long index = offsets[s] + k;
                            int len = Math.Min(b, rows - j);
                            v.AsSpan((int)(index * b), len).CopyTo(vk.Slice(j + j * rows, len));
                            tauk[j] = tau[index];
                        }

                        var tk = tAll.AsSpan(k * g * g, width * width);
                        Householder.FormT(rows, width, vk, rows, tauk, tk, width);
                        var yk = yAll.AsSpan(k * h * g, rows * width);
                        yk.Clear();
                        Gemm.Multiply(false, false, rows, width, width, 1.0, vk, rows, tk, width, yk, rows);
                    });

                    // C ← C - Y(VᵀC)
                    Parallelism.ForStatic(threads, t =>
                    {
                        int lo = bounds[t];
                        int hi = bounds[t + 1];
                        if (hi <= lo)
                        {
                            return;
                        }

                        var w = wAll.AsSpan(t * g * columns, g * columns);
                        for (int k = 0; k < kernels; k++)
                        {
                            int r0 = s0 + 1 + k * b;
                            int rows = Math.Min(h, n - r0);
                            int width = Math.Min(members, rows);
                            var ck = sc.Span[(r0 + lo * ldc)..];
                            var wk = w[..(width * (hi - lo))];
                            wk.Clear();
                            Gemm.Multiply(true, false, width, hi - lo, rows, 1.0, vAll.AsSpan(k * h * g, rows * width), rows, ck, ldc, wk, width);
                            Gemm.Multiply(false, false, rows, hi - lo, width, -1.0, yAll.AsSpan(k * h * g, rows * width), rows, wk, width, ck, ldc);
                        }
                    });
                }
            }
        }
        finally
        {
            WorkArrayPool<double>.Return(vAll);
            WorkArrayPool<double>.Return(yAll);
            WorkArrayPool<double>.Return(tAll);
            WorkArrayPool<double>.Return(tauAll);
            WorkArrayPool<double>.Return(wAll);
        }
    }
}
