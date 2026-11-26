namespace Tuna.LinearAlgebra;

// LAPACK dsytrd 相当、下三角を使用
/// <summary>
/// 対称行列の三重対角化 <c>A = QTQᵀ</c>
/// </summary>
internal static class TridiagonalReduction
{
    // d に対角（長さ n）、e に副対角（長さ n-1）、 A[i+2:n, i] に反射ベクトル（先頭の1は省略）、tau に係数（長さ n-1）を格納する。
    /// <summary>
    /// 下三角部分から三重対角化する
    /// </summary>
    internal static unsafe void Reduce(int n, Span<double> a, int lda, Span<double> d, Span<double> e, Span<double> tau)
    {
        if (n == 0)
        {
            return;
        }

        int nb = BlockSize;
        int blocked = n > Crossover ? n - Crossover : 0;
        blocked -= blocked % nb;
        int threads = blocked > 0 ? Parallelism.PlannedDegree((double)n * n / 2, MatrixVector.Grain) : 1;
        double[] rented = WorkArrayPool<double>.Rent(checked(n * (nb + 1)));
        ColumnTeam? team = null;
        try
        {
            fixed (double* pa = a)
            {
                // 先頭の列ブロックごとに、パネルの反射と W を作り、残りを階数 2nb の更新 A ← A - VWᵀ - WVᵀ で更新する
                int i = 0;
                if (blocked > 0)
                {
                    team = ColumnTeam.Rent(threads, n);
                    for (; i < blocked; i += nb)
                    {
                        int m = n - i;
                        var sub = a[(i + i * lda)..];
                        var w = rented.AsSpan(0, m * nb);
                        team.Partition(m, nb, triangular: true);
                        ReducePanel(m, nb, sub, lda, e[i..], tau[i..], w, m, team, pa + i + (nint)i * lda);

                        // 各列区間の担当スレッドが、その列だけを更新する（次のパネルでも同じ列を読む）
                        fixed (double* pw = rented)
                        {
                            var work = new TrailingWork(team.Ranges, nb, m - nb, m, pa + i + (nint)i * lda, lda, pw);
                            team.For(ref work);
                        }

                        for (int j = 0; j < nb; j++)
                        {
                            sub[j + 1 + j * lda] = e[i + j];
                            d[i + j] = sub[j + j * lda];
                        }
                    }
                }

                ReduceUnblocked(n - i, a[(i + i * lda)..], lda, d[i..], e[i..], tau[i..], rented.AsSpan(0, n - i));
            }
        }
        finally
        {
            WorkArrayPool<double>.Return(rented);
            if (team is not null)
            {
                ColumnTeam.Return(team);
            }
        }
    }

    // w は ldw = m。
    /// <summary>
    /// 列区間 <c>t</c> の担当の列だけを <c>A ← A - VWᵀ - WVᵀ</c> で更新する作業（<c>a</c> はパネルの先頭）
    /// </summary>
    private readonly unsafe struct TrailingWork(ColumnRanges ranges, int nb, int rest, int m, double* a, int lda, double* w) : IParallelWork
    {
        public void Invoke(int t)
        {
            var (lo, hi) = ranges.Range(t, nb);
            if (hi <= lo)
            {
                return;
            }

            int j0 = lo - nb;
            int len = rest - j0;
            UpdateTrailing(
                len, hi - lo, nb,
                new ReadOnlySpan<double>(a + lo, (nb - 1) * lda + len), lda,
                new ReadOnlySpan<double>(w + lo, (nb - 1) * m + len), m,
                new Span<double>(a + lo + (nint)lo * lda, (hi - lo - 1) * lda + len), lda);
        }
    }

    // LAPACK dsytrd の既定と同じ 32
    /// <summary>
    /// ブロック幅
    /// </summary>
    internal const int BlockSize = 32;

    // LAPACK の既定と同じ 128
    /// <summary>
    /// 残りがこれ以下になったら非ブロック版で処理する
    /// </summary>
    internal const int Crossover = 128;

    // 対称行列とベクトルの積は team の列区間ごとに計算する（pa は固定済みの a）
    // LAPACK dlatrd、下三角
    // 反射ベクトルの先頭成分（副対角）は1に置き換えたまま返す（副対角の値は e）。
    /// <summary>
    /// 先頭 <paramref name="nb"/> 列の反射を作り、<c>A ← A - VWᵀ - WVᵀ</c> の <c>W</c>（<c>m × nb</c>）を求める
    /// </summary>
    private static unsafe void ReducePanel(int m, int nb, Span<double> a, int lda, Span<double> e, Span<double> tau, Span<double> w, int ldw, ColumnTeam team, double* pa)
    {
        for (int i = 0; i < nb; i++)
        {
            var col = a.Slice(i + i * lda, m - i);

            // A(i:m, i) ← A(i:m, i) - A(i:m, 0:i) W(i, 0:i)ᵀ - W(i:m, 0:i) A(i, 0:i)ᵀ
            for (int j = 0; j < i; j++)
            {
                Level1Kernels.Axpy2(-w[i + j * ldw], a.Slice(i + j * lda, m - i), -a[i + j * lda], w.Slice(i + j * ldw, m - i), col);
            }

            if (i + 1 >= m)
            {
                tau[i] = 0;
                continue;
            }

            double alpha = col[1];
            double taui = Householder.Generate(ref alpha, col[2..]);
            e[i] = alpha;
            tau[i] = taui;
            col[1] = 1;

            int len = m - i - 1;
            var v = col[1..];
            var wi = w.Slice(i + 1 + i * ldw, len);
            if (taui == 0)
            {
                wi.Clear();
                continue;
            }

            // W(i+1:m, i) = τ (A22 v - V (Wᵀv) - W (Vᵀv))、A22 はパネル開始時点の値
            fixed (double* pw = wi)
            {
                team.SymmetricMultiply(pa, lda, m, i + 1, pa + i + 1 + (nint)i * lda, pw);
            }

            var tmp = w.Slice(i * ldw, i);
            for (int j = 0; j < i; j++)
            {
                tmp[j] = Level1Kernels.Dot(w.Slice(i + 1 + j * ldw, len), v);
            }

            for (int j = 0; j < i; j++)
            {
                Level1Kernels.Axpy(-tmp[j], a.Slice(i + 1 + j * lda, len), wi);
            }

            for (int j = 0; j < i; j++)
            {
                tmp[j] = Level1Kernels.Dot(a.Slice(i + 1 + j * lda, len), v);
            }

            for (int j = 0; j < i; j++)
            {
                Level1Kernels.Axpy(-tmp[j], w.Slice(i + 1 + j * ldw, len), wi);
            }

            Level1Kernels.Scale(taui, wi);
            double alpha2 = -0.5 * taui * Level1Kernels.Dot(wi, v);
            Level1Kernels.Axpy(alpha2, v, wi);
        }
    }

    // 列ブロックごとに行列積で更新するので、対角ブロック内の上三角も書き換わる。
    // C は n × columns の部分、 V・W は n × k。
    /// <summary>
    /// <c>C ← C - VWᵀ - WVᵀ</c> の下三角のうち先頭 <paramref name="columns"/> 列
    /// </summary>
    private static void UpdateTrailing(int n, int columns, int k, ReadOnlySpan<double> v, int ldv, ReadOnlySpan<double> w, int ldw, Span<double> c, int ldc)
    {
        const int Width = 128;

        // 処理した列数だけ進める（j += Width は、列数が int の上限に近いと最後のブロックの後に負へ折り返す）
        for (int j = 0, jb; j < columns; j += jb)
        {
            jb = Math.Min(Width, columns - j);
            var cj = c[(j + j * ldc)..];
            Gemm.Multiply(false, true, n - j, jb, k, -1.0, v[j..], ldv, w[j..], ldw, cj, ldc);
            Gemm.Multiply(false, true, n - j, jb, k, -1.0, w[j..], ldw, v[j..], ldv, cj, ldc);
        }
    }

    // LAPACK dsytd2、下三角
    /// <summary>
    /// 非ブロック版
    /// </summary>
    private static void ReduceUnblocked(int n, Span<double> a, int lda, Span<double> d, Span<double> e, Span<double> tau, Span<double> work)
    {
        for (int i = 0; i < n - 1; i++)
        {
            int sub = i + 1 + i * lda;
            int len = n - i - 1;
            double alpha = a[sub];
            double taui = Householder.Generate(ref alpha, a.Slice(sub + 1, len - 1));
            e[i] = alpha;
            if (taui != 0)
            {
                a[sub] = 1;
                var v = a.Slice(sub, len);
                var a22 = a[(i + 1 + (i + 1) * lda)..];
                var w = work[..len];

                // w = τ A22 v
                MatrixVector.SymmetricMultiply(len, taui, a22, lda, v, w);

                // w ← w - (τ/2)(wᵀv) v
                double alpha2 = -0.5 * taui * Level1Kernels.Dot(w, v);
                Level1Kernels.Axpy(alpha2, v, w);

                // A22 ← A22 - v wᵀ - w vᵀ（下三角）
                for (int j = 0; j < len; j++)
                {
                    Level1Kernels.Axpy2(-w[j], v[j..], -v[j], w[j..], a22.Slice(j + j * lda, len - j));
                }

                a[sub] = e[i];
            }

            d[i] = a[i + i * lda];
            tau[i] = taui;
        }

        d[n - 1] = a[(n - 1) + (n - 1) * lda];
    }

    // LAPACK dormtr 'L','L','N'
    /// <summary>
    /// <see cref="Reduce"/> の反射の積 <c>Q</c> を <c>C</c>（<c>n × columns</c>）に左から掛ける
    /// </summary>
    internal static void ApplyQ(int n, int columns, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> tau, Span<double> c, int ldc)
    {
        // Q = diag(1, Q')。C の先頭行は変わらない
        if (n > 1)
        {
            QrKernels.ApplyQ(n - 1, columns, n - 1, a[1..], lda, tau, c[1..], ldc);
        }
    }

    // LAPACK dorgtr、下三角
    /// <summary>
    /// <see cref="Reduce"/> の反射から直交行列 <c>Q</c>（<c>n × n</c>）を作る
    /// </summary>
    internal static void FormQ(int n, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> tau, Span<double> q, int ldq)
    {
        if (n == 0)
        {
            return;
        }

        // Q = diag(1, Q'), Q' は A[1:n, 0:n-1] に QR と同じ形で格納された n-1 個の反射の積
        q[..n].Clear();
        q[0] = 1;
        for (int j = 1; j < n; j++)
        {
            q[j * ldq] = 0;
        }

        if (n > 1)
        {
            QrKernels.FormQ(n - 1, n - 1, n - 1, a[1..], lda, tau, q[(1 + ldq)..], ldq);
        }
    }
}
