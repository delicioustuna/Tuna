namespace Tuna.LinearAlgebra;

// 二段階の上二重対角化（LAPACK dgebrd_2stage 相当、m >= n）。
internal static class TwoStageBidiagonal
{
    internal const int Bandwidth = 64;

    internal static int SweepCount(int n) => Math.Max(n - 2, 0);

    // d に対角（長さ n）、e に優対角（長さ n-1）。
    // 第 1 段の左反射は A の対角より下に（係数は tauq、長さ n）、右反射は p（(n-b)×(n-b) 列優先、逆変換不要なら空可。係数は taup、長さ n-b）に格納する。
    // left / right が指定されていれば第 2 段の左右反射も保存する。
    internal static unsafe void Reduce(
        int m, int n, int b, Span<double> a, int lda, Span<double> d, Span<double> e, Span<double> tauq, Span<double> p, Span<double> taup,
        BandReflectors? left, BandReflectors? right)
    {
        ReduceToBand(m, n, b, a, lda, tauq, p, Math.Max(n - b, 1), taup);

        // 上三角帯行列を、対角の上 2b-1 本・下 b-1 本まで入る帯の形式に移す。要素 (r, c) は band[o + r + c·ld]
        int ldab = 3 * b;
        int ld = ldab - 1;
        int o = 2 * b - 1;
        double[] rented = WorkArrayPool<double>.Rent(checked(n * ldab + o + 1));
        try
        {
            var band = rented.AsSpan(0, n * ldab + o + 1);
            band.Clear();
            for (int c = 0; c < n; c++)
            {
                int r0 = Math.Max(0, c - b);
                a.Slice(r0 + c * lda, c - r0 + 1).CopyTo(band[(o + r0 + c * ld)..]);
            }

            fixed (double* pb = band)
            {
                var sb = new SharedSpan(pb + o, band.Length - o);
                BulgeChasing.Run(n, b, SweepCount(n), 4 * b + 4, (s, work, progress) => Sweep(n, b, s, sb.Span, ld, work, left, right, progress));
            }

            for (int i = 0; i < n; i++)
            {
                d[i] = band[o + i + i * ld];
                if (i + 1 < n)
                {
                    e[i] = band[o + i + (i + 1) * ld];
                }
            }
        }
        finally
        {
            WorkArrayPool<double>.Return(rented);
        }
    }

    /// <summary>
    /// 第1段：<c>m × n</c> の <c>A</c> を上側の帯幅 <paramref name="b"/> の上三角帯行列に縮約する
    /// </summary>
    internal static void ReduceToBand(int m, int n, int b, Span<double> a, int lda, Span<double> tauq, Span<double> p, int ldp, Span<double> taup)
    {
        int rowsMax = Math.Max(n - b, 1);
        double[] rented = WorkArrayPool<double>.Rent(checked(m * b + 3 * rowsMax * b + 2 * b * b));
        try
        {
            var buffer = rented.AsSpan();
            var v = buffer[..(m * b)];
            var t = buffer.Slice(m * b, b * b);
            var pt = buffer.Slice(m * b + b * b, rowsMax * b);
            var v2 = buffer.Slice(m * b + b * b + rowsMax * b, rowsMax * b);
            var t2 = buffer.Slice(m * b + b * b + 2 * rowsMax * b, b * b);
            if (n > b)
            {
                taup[..(n - b)].Clear();
            }

            for (int j = 0; j < n; j += b)
            {
                int jb = Math.Min(b, n - j);
                int rows = m - j;

                // 列ブロック A[j:m, j:j+jb] をQR分解し、右の列に左から掛ける
                QrKernels.FactorPanel(rows, jb, a[(j + j * lda)..], lda, tauq[j..], v, t, formBlock: true);
                if (j + jb < n)
                {
                    Householder.ApplyBlockLeft(true, rows, n - j - jb, jb, v, rows, t, jb, a[(j + (j + jb) * lda)..], lda);
                }

                // 帯の右の行ブロック A[j:j+jb, j+b:n] を、転置してQR分解する（LQ分解）
                int rc = n - j - b;
                if (rc <= 0)
                {
                    continue;
                }

                int k2 = Math.Min(rc, jb);
                Level1Kernels.Transpose(jb, rc, a[(j + (j + b) * lda)..], lda, pt, rc);
                if (rc >= jb)
                {
                    QrKernels.FactorPanel(rc, jb, pt, rc, taup[j..], v2, t2, formBlock: true);
                }
                else
                {
                    QrKernels.Factor(rc, jb, pt, rc, taup[j..]);
                    Householder.ExtractV(rc, k2, pt, rc, v2, rc);
                    Householder.FormT(rc, k2, v2, rc, taup[j..], t2, k2);
                }

                // 行ブロックは [Rᵀ 0]（Rᵀ は下三角）になる。帯の外の 0 はこの後読まないので書かない。
                // 反射は（逆変換に使うなら）p の列 j.. に QR と同じ形で移す
                for (int i = 0; i < Math.Min(rc, jb); i++)
                {
                    var column = a.Slice(j + (j + b + i) * lda, jb);
                    column[..i].Clear();
                    for (int q = i; q < jb; q++)
                    {
                        column[q] = pt[i + q * rc];
                    }
                }

                if (!p.IsEmpty)
                {
                    for (int q = 0; q < k2; q++)
                    {
                        pt.Slice(q + 1 + q * rc, rc - q - 1).CopyTo(p[(j + q + 1 + (j + q) * ldp)..]);
                    }
                }

                // 下の行に右から掛ける
                if (j + jb < m)
                {
                    Householder.ApplyBlockRight(false, m - j - jb, rc, k2, v2, rc, t2, k2, a[(j + jb + (j + b) * lda)..], lda);
                }
            }
        }
        finally
        {
            WorkArrayPool<double>.Return(rented);
        }
    }

    // 帯行列の要素 (r, c) は band[r + c·ld]（-(2b-1) ≤ r - c ≤ b-1）。
    /// <summary>
    /// スイープ <paramref name="s"/>：行 s の帯の中の要素を消し、できたバルジを対角に沿って送る
    /// </summary>
    private static void Sweep(int n, int b, int s, Span<double> band, int ld, Span<double> work, BandReflectors? left, BandReflectors? right, int[]? progress)
    {
        var u = work[..b];
        var w = work.Slice(b, b);
        var tmp = work.Slice(2 * b, 2 * b);
        for (int kernel = 0; ; kernel++)
        {
            int st = s + 1 + kernel * b;
            if (st > n - 1)
            {
                break;
            }

            int ed = Math.Min(st + b - 1, n - 1);
            int lm = ed - st + 1;
            int r = kernel == 0 ? s : st - b;
            BulgeChasing.WaitFor(progress, n, b, s, kernel);

            // 行 r の列 st..ed を右からの反射で [β, 0, …] にし、その列に要素を持つ行 r+1..ed に掛ける
            for (int i = 0; i < lm; i++)
            {
                u[i] = band[r + (st + i) * ld];
            }

            double alpha = u[0];
            double tauR = Householder.Generate(ref alpha, u[1..lm]);
            band[r + st * ld] = alpha;
            for (int i = 1; i < lm; i++)
            {
                band[r + (st + i) * ld] = 0;
            }

            u[0] = 1;
            Householder.ApplyRight(u[..lm], tauR, ed - r, lm, band[(r + 1 + st * ld)..], ld, tmp);
            right?.Store(s, kernel, u[..lm], tauR);

            // 対角の下にできた列 st の要素を左からの反射で消し、行 st..ed の右の要素に掛ける
            var x = band.Slice(st + 1 + st * ld, lm - 1);
            double beta = band[st + st * ld];
            double tauL = Householder.Generate(ref beta, x);
            band[st + st * ld] = beta;
            w[0] = 1;
            x.CopyTo(w[1..]);
            x.Clear();
            int cmax = Math.Min(ed + b, n - 1);
            if (cmax > st)
            {
                Householder.ApplyLeft(w[..lm], tauL, lm, cmax - st, band[(st + (st + 1) * ld)..], ld);
            }

            left?.Store(s, kernel, w[..lm], tauL);
            BulgeChasing.Publish(progress, s, kernel + 1);
        }
    }

    /// <summary>
    /// 第1段の右の反射の積 <c>P₁</c> を <c>C</c>（<c>n × columns</c>）に左から掛ける
    /// </summary>
    internal static void ApplyP1(int n, int b, int columns, ReadOnlySpan<double> p, ReadOnlySpan<double> taup, Span<double> c, int ldc)
    {
        if (n > b + 1)
        {
            QrKernels.ApplyQ(n - b, columns, n - b, p, n - b, taup, c[b..], ldc);
        }
    }
}
