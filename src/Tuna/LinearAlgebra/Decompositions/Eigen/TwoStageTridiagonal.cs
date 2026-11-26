namespace Tuna.LinearAlgebra;

// 対称行列の二段階三重対角化（LAPACK dsytrd_2stage 相当、下三角を使用）。
internal static class TwoStageTridiagonal
{
    internal const int Bandwidth = 64;
    private const int Width = 128;

    internal static int SweepCount(int n) => Math.Max(n - 2, 0);

    // d に対角（長さ n）、e に副対角（長さ n-1）。
    // 第 1 段の反射は A[b:n, 0:n-b] に QR 分解と同じ形で格納し、係数は tau（長さ n-b）に格納する。
    // reflectors が指定されていれば第 2 段の反射も保存する。
    internal static void Reduce(int n, int b, Span<double> a, int lda, Span<double> d, Span<double> e, Span<double> tau, BandReflectors? reflectors)
    {
        ReduceToBand(n, b, a, lda, tau);
        int ldab = 2 * b + 1;
        double[] band = WorkArrayPool<double>.Rent(checked(n * ldab));
        try
        {
            var ab = band.AsSpan(0, n * ldab);
            ab.Clear();
            for (int j = 0; j < n; j++)
            {
                int len = Math.Min(b + 1, n - j);
                a.Slice(j + j * lda, len).CopyTo(ab[(j * ldab)..]);
            }

            BandToTridiagonal(n, b, ab, d, e, reflectors);
        }
        finally
        {
            WorkArrayPool<double>.Return(band);
        }
    }

    /// <summary>
    /// 第1段：下三角を帯幅 <paramref name="b"/> の帯行列に縮約する（帯の外に反射、<paramref name="tau"/> は長さ <c>n - b</c>）
    /// </summary>
    internal static void ReduceToBand(int n, int b, Span<double> a, int lda, Span<double> tau)
    {
        if (n <= b + 1)
        {
            tau[..Math.Max(n - b, 0)].Clear();
            return;
        }

        tau[..(n - b)].Clear();
        int mMax = n - b;
        double[] rented = WorkArrayPool<double>.Rent(checked(7 * mMax * b + 2 * b * b));
        try
        {
            var buffer = rented.AsSpan();
            for (int j = 0; j < n - b; j += b)
            {
                int r0 = j + b;
                int m = n - r0;
                int k = Math.Min(m, b);
                var panel = a[(r0 + j * lda)..];
                var v = buffer[..(m * k)];
                var t = buffer.Slice(m * k, k * k);
                var rest = buffer[(m * k + k * k)..];

                // 帯の下の列ブロックをQR分解する
                if (m >= b)
                {
                    QrKernels.FactorPanel(m, b, panel, lda, tau[j..], v, t, formBlock: true);
                }
                else
                {
                    QrKernels.Factor(m, b, panel, lda, tau[j..]);
                    Householder.ExtractV(m, k, panel, lda, v, m);
                    Householder.FormT(m, k, v, m, tau[j..], t, k);
                }

                if (m <= 1)
                {
                    continue;
                }

                // A22 ← QᵀA22Q（Q = I - VTVᵀ）を A22 ← A22 - WVᵀ - VWᵀ として計算する。
                // Y = VT、X = A22·Y、W = X - ½V(YᵀX)
                var y = rest[..(m * k)];
                var x = rest.Slice(m * k, m * k);
                var g = rest.Slice(2 * m * k, k * k);
                y.Clear();
                Gemm.Multiply(false, false, m, k, k, 1.0, v, m, t, k, y, m);
                var a22 = a[(r0 + r0 * lda)..];
                SymmetricMultiply(m, k, a22, lda, y, m, x, m);
                g.Clear();
                Gemm.Multiply(true, false, k, k, m, 1.0, y, m, x, m, g, k);
                Gemm.Multiply(false, false, m, k, k, -0.5, v, m, g, k, x, m);
                SymmetricRank2kUpdate(m, k, v, m, x, m, a22, lda, rest.Slice(2 * m * k + k * k, 4 * m * k));
            }
        }
        finally
        {
            WorkArrayPool<double>.Return(rented);
        }
    }

    // 出力の行ブロックごとに、その左の行ブロック・対角ブロック・下の列ブロックの寄与を行列積で求める。
    // A は下三角で与えた m × m 対称行列、Y・X は m × k。
    /// <summary>
    /// <c>X = A·Y</c>
    /// </summary>
    private static unsafe void SymmetricMultiply(int m, int k, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> y, int ldy, Span<double> x, int ldx)
    {
        int blocks = (m + Width - 1) / Width;
        int threads = Parallelism.Degree((double)m * m * k, 4 * Parallelism.Grain);
        double[] diag = WorkArrayPool<double>.Rent(checked(Math.Max(threads, 1) * Width * Width + 1));
        try
        {
            fixed (double* pa = a)
            fixed (double* py = y)
            fixed (double* px = x)
            fixed (double* pd = diag)
            {
                var sa = new SharedSpan(pa, a.Length);
                var sy = new SharedSpan(py, y.Length);
                var sx = new SharedSpan(px, x.Length);
                var sd = new SharedSpan(pd, diag.Length);
                int[] next = [0];
                Parallelism.For(Math.Max(threads, 1), threads, worker =>
                {
                    var aa = sa.Span;
                    var yy = sy.Span;
                    var xx = sx.Span;
                    var full = sd.Span.Slice(worker * Width * Width, Width * Width);
                    for (int blk = Interlocked.Increment(ref next[0]) - 1; blk < blocks; blk = Interlocked.Increment(ref next[0]) - 1)
                    {
                        int c0 = blk * Width;
                        int w = Math.Min(Width, m - c0);
                        int c1 = c0 + w;
                        var xb = xx[c0..];
                        for (int j = 0; j < k; j++)
                        {
                            xb.Slice(j * ldx, w).Clear();
                        }

                        // 左の行ブロック A[c0:c1, 0:c0]
                        if (c0 > 0)
                        {
                            Gemm.Multiply(false, false, w, k, c0, 1.0, aa[c0..], lda, yy, ldy, xb, ldx);
                        }

                        // 対角ブロック（下三角を対称に展開する）
                        for (int j = 0; j < w; j++)
                        {
                            for (int i = j; i < w; i++)
                            {
                                double value = aa[c0 + i + (c0 + j) * lda];
                                full[i + j * w] = value;
                                full[j + i * w] = value;
                            }
                        }

                        Gemm.Multiply(false, false, w, k, w, 1.0, full, w, yy[c0..], ldy, xb, ldx);

                        // 下の列ブロック A[c1:m, c0:c1]ᵀ
                        if (c1 < m)
                        {
                            Gemm.Multiply(true, false, w, k, m - c1, 1.0, aa[(c1 + c0 * lda)..], lda, yy[c1..], ldy, xb, ldx);
                        }
                    }
                });
            }
        }
        finally
        {
            WorkArrayPool<double>.Return(diag);
        }
    }

    // [V W]·[W V]ᵀ を列ブロックごとに1回の行列積で引く（対角ブロック内の上三角も書き換わる）。
    // C は m × m、V・W は m × k。
    /// <summary>
    /// <c>C ← C - WVᵀ - VWᵀ</c> の下三角
    /// </summary>
    private static unsafe void SymmetricRank2kUpdate(
        int m, int k, ReadOnlySpan<double> v, int ldv, ReadOnlySpan<double> w, int ldw, Span<double> c, int ldc, Span<double> work)
    {
        var vw = work[..(2 * m * k)];
        var wv = work.Slice(2 * m * k, 2 * m * k);
        Level1Kernels.Copy(m, k, v, ldv, vw, m);
        Level1Kernels.Copy(m, k, w, ldw, vw[(m * k)..], m);
        Level1Kernels.Copy(m, k, w, ldw, wv, m);
        Level1Kernels.Copy(m, k, v, ldv, wv[(m * k)..], m);
        int blocks = (m + Width - 1) / Width;
        int threads = Parallelism.Degree((double)m * m * k, 4 * Parallelism.Grain);
        fixed (double* pvw = vw)
        fixed (double* pwv = wv)
        fixed (double* pc = c)
        {
            var svw = new SharedSpan(pvw, vw.Length);
            var swv = new SharedSpan(pwv, wv.Length);
            var sc = new SharedSpan(pc, c.Length);

            // 下の列ほど短いので、長い先頭の列ブロックから割り当てる
            Parallelism.For(blocks, threads, blk =>
            {
                int c0 = blk * Width;
                int width = Math.Min(Width, m - c0);
                Gemm.Multiply(false, true, m - c0, width, 2 * k, -1.0, svw.Span[c0..], m, swv.Span[c0..], m, sc.Span[(c0 + c0 * ldc)..], ldc);
            });
        }
    }

    // LAPACK dsb2st_kernels の下三角版と同じ手順
    /// <summary>
    /// 第2段：帯幅 <paramref name="b"/> の下三角帯行列 <paramref name="ab"/>（要素 <c>(r, c)</c> は <c>ab[r - c + c·(2b+1)]</c>、 <c>r - c ≤ 2b</c> まで使える）をバルジ追跡で三重対角化する
    /// </summary>
    internal static unsafe void BandToTridiagonal(int n, int b, Span<double> ab, Span<double> d, Span<double> e, BandReflectors? reflectors)
    {
        int ld = 2 * b;
        fixed (double* pab = ab)
        {
            var sab = new SharedSpan(pab, ab.Length);
            BulgeChasing.Run(n, b, SweepCount(n), 4 * b + 4, (s, work, progress) => Sweep(n, b, s, sab.Span, ld, work, reflectors, progress));
        }

        for (int i = 0; i < n; i++)
        {
            d[i] = ab[i * (ld + 1)];
            if (i + 1 < n)
            {
                e[i] = ab[1 + i * (ld + 1)];
            }
        }
    }

    // 帯行列の要素 (r, c) は ab[r + c·ld]（ld = 2b）。
    /// <summary>
    /// スイープ <paramref name="s"/>：列 s の帯の中の要素を消し、できたバルジを対角に沿って送る
    /// </summary>
    private static void Sweep(int n, int b, int s, Span<double> ab, int ld, Span<double> work, BandReflectors? reflectors, int[]? progress)
    {
        var v = work[..b];
        var vNext = work.Slice(b, b);
        var tmp = work.Slice(2 * b, b + 1);

        // 種類1：列 s の行 s+1..ed を反射で [β, 0, …] にし、対角ブロックに両側から掛ける
        BulgeChasing.WaitFor(progress, n, b, s, 0);
        int st = s + 1;
        int ed = Math.Min(s + b, n - 1);
        int lm = ed - st + 1;
        double alpha = ab[st + s * ld];
        var x = ab.Slice(st + 1 + s * ld, lm - 1);
        double tau = Householder.Generate(ref alpha, x);
        ab[st + s * ld] = alpha;
        v[0] = 1;
        x.CopyTo(v[1..]);
        x.Clear();
        reflectors?.Store(s, 0, v[..lm], tau);
        ApplyTwoSided(lm, v[..lm], tau, ab[(st + st * ld)..], ld, tmp);
        BulgeChasing.Publish(progress, s, 1);

        for (int kernel = 1; ; kernel++)
        {
            // 種類2：直前の反射を下のブロックに右から掛け、ブロックの先頭列を次の反射で消して残りの列に左から掛ける
            int j1 = ed + 1;
            int j2 = Math.Min(ed + b, n - 1);
            int ln = ed - st + 1;
            int lm2 = j2 - j1 + 1;
            if (lm2 <= 0)
            {
                break;
            }

            BulgeChasing.WaitFor(progress, n, b, s, kernel);
            var block = ab[(j1 + st * ld)..];
            Householder.ApplyRight(v[..ln], tau, lm2, ln, block, ld, tmp);
            double alpha2 = block[0];
            var x2 = block.Slice(1, lm2 - 1);
            double tau2 = Householder.Generate(ref alpha2, x2);
            block[0] = alpha2;
            vNext[0] = 1;
            x2.CopyTo(vNext[1..]);
            x2.Clear();
            if (ln > 1)
            {
                Householder.ApplyLeft(vNext[..lm2], tau2, lm2, ln - 1, block[ld..], ld);
            }

            reflectors?.Store(s, kernel, vNext[..lm2], tau2);

            // 種類3：次の対角ブロックに両側から掛ける
            st = j1;
            ed = j2;
            tau = tau2;
            vNext[..lm2].CopyTo(v);
            ApplyTwoSided(lm2, v[..lm2], tau, ab[(st + st * ld)..], ld, tmp);
            BulgeChasing.Publish(progress, s, kernel + 1);
        }
    }

    // H = I - τvvᵀ、A は下三角で与えた m × m 対称行列、LAPACK dlarfy
    /// <summary>
    /// <c>A ← HAH</c>
    /// </summary>
    private static void ApplyTwoSided(int m, ReadOnlySpan<double> v, double tau, Span<double> a, int lda, Span<double> work)
    {
        if (tau == 0 || m <= 0)
        {
            return;
        }

        // w = τAv、w ← w - ½τ(wᵀv)v、A ← A - vwᵀ - wvᵀ
        var w = work[..m];
        MatrixVector.SymmetricMultiplySequential(0, m, m, tau, a, lda, v, w);
        double alpha = -0.5 * tau * Level1Kernels.Dot(w, v);
        Level1Kernels.Axpy(alpha, v, w);
        for (int j = 0; j < m; j++)
        {
            Level1Kernels.Axpy2(-w[j], v[j..], -v[j], w[j..], a.Slice(j + j * lda, m - j));
        }
    }

    /// <summary>
    /// 第1段の反射の積 <c>Q₁</c> を <c>C</c>（<c>n × columns</c>）に左から掛ける
    /// </summary>
    internal static void ApplyQ1(int n, int b, int columns, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> tau, Span<double> c, int ldc)
    {
        if (n > b + 1)
        {
            QrKernels.ApplyQ(n - b, columns, n - b, a[b..], lda, tau, c[b..], ldc);
        }
    }
}
