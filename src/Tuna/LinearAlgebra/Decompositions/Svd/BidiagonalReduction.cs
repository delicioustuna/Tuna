namespace Tuna.LinearAlgebra;

// LAPACK dgebrd 相当、m ≥ n
/// <summary>
/// 上二重対角化 <c>A = QBPᵀ</c>
/// </summary>
internal static class BidiagonalReduction
{
    // d に対角（長さ n）、 e に優対角（長さ n-1）、対角より下に左の反射、A[i, i+2:n] に右の反射を格納する。
    /// <summary>
    /// <c>m × n</c>（<c>m ≥ n</c>）の <c>A</c> を上二重対角化する
    /// </summary>
    internal static unsafe void Reduce(int m, int n, Span<double> a, int lda, Span<double> d, Span<double> e, Span<double> tauq, Span<double> taup)
    {
        int nb = BlockSize;
        int blocked = n > Crossover ? n - Crossover : 0;
        blocked -= blocked % nb;
        double[] rented = WorkArrayPool<double>.Rent(checked(m + n + (m + n) * nb));
        var u = rented.AsSpan(0, n);
        var work = rented.AsSpan(n, m);
        ColumnTeam? team = null;
        BidiagonalPanel? panel = null;
        try
        {
            fixed (double* pa = a)
            fixed (double* pr = rented)
            {
                // 先頭から nb 列ずつ反射を作り（dlabrd）、残りを A ← A - VYᵀ - XUᵀ の行列積で更新する。
                // パネルと更新は、逐次でも次数だけで決まる同じ列区間で行う（並列数によらない結果にする）
                int i = 0;
                if (blocked > 0)
                {
                    int threads = Parallelism.PlannedDegree((double)m * n, MatrixVector.Grain);
                    team = ColumnTeam.Rent(threads, m);
                    panel = BidiagonalPanel.Rent(team, m, n, nb);
                    for (; i < blocked; i += nb)
                    {
                        int mm = m - i;
                        int nn = n - i;
                        var sub = a[(i + i * lda)..];
                        team.Partition(nn, nb);
                        panel.Reduce(mm, nn, nb, pa + i + (nint)i * lda, lda, d[i..], e[i..], tauq[i..], taup[i..], pr + m + n, mm, pr + m + n + mm * nb, nn);

                        // 各列区間の担当スレッドが、その列だけを更新する（次のパネルでも同じ列を読む）
                        var update = new TrailingWork(team.Ranges, nb, mm, nn, pa + i + (nint)i * lda, lda, pr + m + n, pr + m + n + mm * nb);
                        team.For(ref update);

                        for (int j = 0; j < nb; j++)
                        {
                            sub[j + j * lda] = d[i + j];
                            sub[j + (j + 1) * lda] = e[i + j];
                        }
                    }
                }

                var rest = a[(i + i * lda)..];
                ReduceUnblocked(m - i, n - i, rest, lda, d[i..], e[i..], tauq[i..], taup[i..], u, work);
            }
        }
        finally
        {
            WorkArrayPool<double>.Return(rented);
            if (panel is not null)
            {
                BidiagonalPanel.Return(panel);
            }

            if (team is not null)
            {
                ColumnTeam.Return(team);
            }
        }
    }

    // x は ldx = mm、y は ldy = nn。
    /// <summary>
    /// 列区間 <c>t</c> の担当の列だけを <c>A ← A - VYᵀ - XUᵀ</c> で更新する作業（<c>a</c> はパネルの先頭）
    /// </summary>
    private readonly unsafe struct TrailingWork(ColumnRanges ranges, int nb, int mm, int nn, double* a, int lda, double* x, double* y) : IParallelWork
    {
        public void Invoke(int t)
        {
            var (lo, hi) = ranges.Range(t, nb);
            if (hi <= lo)
            {
                return;
            }

            int cols = hi - lo;
            int rows = mm - nb;
            var c = new Span<double>(a + nb + (nint)lo * lda, (cols - 1) * lda + rows);
            Gemm.Multiply(
                false, true, rows, cols, nb, -1.0, new ReadOnlySpan<double>(a + nb, (nb - 1) * lda + rows), lda,
                new ReadOnlySpan<double>(y + lo, (nb - 1) * nn + cols), nn, c, lda);
            Gemm.Multiply(
                false, false, rows, cols, nb, -1.0, new ReadOnlySpan<double>(x + nb, (nb - 1) * mm + rows), mm,
                new ReadOnlySpan<double>(a + (nint)lo * lda, (cols - 1) * lda + nb), lda, c, lda);
        }
    }

    // LAPACK dgebrd の既定と同じ 32
    /// <summary>
    /// ブロック幅
    /// </summary>
    internal const int BlockSize = 32;

    // LAPACK の既定と同じ 128
    /// <summary>
    /// 残りの列がこれ以下になったら非ブロック版で処理する
    /// </summary>
    internal const int Crossover = 128;

    // LAPACK dgebd2、m ≥ n
    /// <summary>
    /// 非ブロック版
    /// </summary>
    private static void ReduceUnblocked(
        int m, int n, Span<double> a, int lda, Span<double> d, Span<double> e, Span<double> tauq, Span<double> taup,
        Span<double> u, Span<double> work)
    {
        for (int i = 0; i < n; i++)
        {
            // 左の反射 H(i) で A[i+1:m, i] を消去
            int ii = i + i * lda;
            double alpha = a[ii];
            double tq = Householder.Generate(ref alpha, a.Slice(ii + 1, m - i - 1));
            tauq[i] = tq;
            d[i] = alpha;
            if (i < n - 1 && tq != 0)
            {
                a[ii] = 1;
                Householder.ApplyLeft(a.Slice(ii, m - i), tq, m - i, n - i - 1, a[(ii + lda)..], lda);
            }

            a[ii] = d[i];

            if (i < n - 1)
            {
                // 右の反射 G(i) で A[i, i+2:n] を消去（行は飛び飛びなので連続領域にコピーして扱う）
                int len = n - i - 1;
                var row = u[..len];
                for (int j = 0; j < len; j++)
                {
                    row[j] = a[i + (i + 1 + j) * lda];
                }

                double beta = row[0];
                double tp = Householder.Generate(ref beta, row[1..]);
                taup[i] = tp;
                e[i] = beta;
                row[0] = 1;
                Householder.ApplyRight(row, tp, m - i - 1, len, a[(i + 1 + (i + 1) * lda)..], lda, work);
                a[i + (i + 1) * lda] = beta;
                for (int j = 1; j < len; j++)
                {
                    a[i + (i + 1 + j) * lda] = row[j];
                }
            }
            else
            {
                taup[i] = 0;
            }
        }
    }

    /// <summary>
    /// 左の反射から <c>Q</c> の先頭 n 列（<c>m × n</c>）を作る
    /// </summary>
    internal static void FormQ(int m, int n, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> tauq, Span<double> q, int ldq) =>
        QrKernels.FormQ(m, n, n, a, lda, tauq, q, ldq);

    // LAPACK dormbr 'P','L','N'
    /// <summary>
    /// 右の反射の積 <c>P</c>（<c>n × n</c>）を <c>C</c>（<c>n × columns</c>）に左から掛ける
    /// </summary>
    internal static void ApplyP(int n, int columns, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> taup, Span<double> c, int ldc)
    {
        if (n <= 1)
        {
            return;
        }

        // P = diag(1, P')。行に格納された反射を列に並べ替えて QR と同じ形にする
        int k = n - 1;
        double[] rented = WorkArrayPool<double>.Rent(checked(k * k));
        var t = rented.AsSpan(0, k * k);
        for (int col = 0; col < k; col++)
        {
            for (int r = col; r < k; r++)
            {
                t[r + col * k] = a[col + (1 + r) * lda];
            }
        }

        QrKernels.ApplyQ(k, columns, k, t, k, taup, c[1..], ldc);
        WorkArrayPool<double>.Return(rented);
    }

    /// <summary>
    /// 右の反射から <c>P</c>（<c>n × n</c>）を作る
    /// </summary>
    internal static void FormP(int n, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> taup, Span<double> p, int ldp)
    {
        p[..n].Clear();
        p[0] = 1;
        for (int j = 1; j < n; j++)
        {
            p[j * ldp] = 0;
        }

        if (n <= 1)
        {
            return;
        }

        // 行に格納された反射を列に並べ替えると、QR と同じ形（(n-1) × (n-1)）になる
        int k = n - 1;
        double[] rented = WorkArrayPool<double>.Rent(checked(k * k));
        var t = rented.AsSpan(0, k * k);
        for (int c = 0; c < k; c++)
        {
            for (int r = c; r < k; r++)
            {
                t[r + c * k] = a[c + (1 + r) * lda];
            }
        }

        QrKernels.FormQ(k, k, k, t, k, taup, p[(1 + ldp)..], ldp);
        WorkArrayPool<double>.Return(rented);
    }
}
