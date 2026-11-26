namespace Tuna.LinearAlgebra;

// LAPACK dgeqrf / dorgqr 相当、ブロック化
/// <summary>
/// ハウスホルダーQR分解の計算本体
/// </summary>
internal static class QrKernels
{
    /// <summary>
    /// ブロック幅
    /// </summary>
    internal const int BlockSize = 32;

    // パネルごとに残りの行列全体を2回読み書きするので、ブロック幅が狭いと行列がキャッシュに収まらない大きさでメモリ帯域に律速される。
    // 幅を広げるとパネルの分解が重くなるので、パネルは再帰的に分解して行列積にする（FactorRecursive）
    /// <summary>
    /// 大きな行列でのブロック幅（<see cref="LargeThreshold"/> 以上で 64、その2倍以上で 128）
    /// </summary>
    internal const int LargeBlockSize = 64;

    /// <summary>
    /// 反射の数がこれ以上なら <see cref="LargeBlockSize"/> を使う（未満の結果はブロック幅 32 のときと変わらない）
    /// </summary>
    internal const int LargeThreshold = 2048;

    // 分解と違って逐次のパネル分解がないので、広いほど Q の読み書きが減る。
    /// <summary>
    /// <see cref="FormQ"/> の大きな行列でのブロック幅
    /// </summary>
    private const int FormQLargeBlockSize = 128;

    /// <summary>
    /// 再帰的なパネル分解で、これ以下の幅は非ブロック版で分解する
    /// </summary>
    private const int RecursiveLeaf = 16;

    // 並列版では呼び出し元のスレッドが次のパネルを分解する間に他のスレッドが残りを更新するので、
    // 幅を広げすぎると 2048 程度ではパネルの分解を待つ時間が増える
    /// <summary>
    /// 反射 <paramref name="k"/> 個の分解のブロック幅
    /// </summary>
    internal static int FactorBlockSize(int k) => k >= 2 * LargeThreshold ? 2 * LargeBlockSize : k >= LargeThreshold ? LargeBlockSize : BlockSize;

    // LAPACK の既定の切り替え点 nx と同じ 128
    /// <summary>
    /// 反射の数がこれ以下なら非ブロック版を使う
    /// </summary>
    internal const int Crossover = 128;

    // 上三角に R、対角より下に反射ベクトル（先頭の1は省略）、 tau に係数 τ を格納する。
    /// <summary>
    /// <c>m × n</c> 行列 <c>A</c> をその場で分解する
    /// </summary>
    internal static void Factor(int m, int n, Span<double> a, int lda, Span<double> tau)
    {
        int k = Math.Min(m, n);
        int nb = FactorBlockSize(k);
        if (k <= Crossover || n <= nb)
        {
            FactorUnblocked(m, n, a, lda, tau);
            return;
        }

        int threads = Math.Min(Parallelism.Degree((double)m * n * k, 4 * Parallelism.Grain), (n - nb) / 32);
        if (threads > 1)
        {
            FactorParallel(threads, m, n, a, lda, tau);
            return;
        }

        double[] vbuf = WorkArrayPool<double>.Rent(checked(m * nb));
        double[] tbuf = WorkArrayPool<double>.Rent(checked(nb * nb));
        try
        {
            for (int j = 0; j < k; j += nb)
            {
                int jb = Math.Min(nb, k - j);
                int rows = m - j;
                FactorPanel(rows, jb, a[(j + j * lda)..], lda, tau[j..], vbuf, tbuf, formBlock: j + jb < n);
                if (j + jb < n)
                {
                    Householder.ApplyBlockLeft(true, rows, n - j - jb, jb, vbuf, rows, tbuf, jb, a[(j + (j + jb) * lda)..], lda);
                }
            }
        }
        finally
        {
            WorkArrayPool<double>.Return(vbuf);
            WorkArrayPool<double>.Return(tbuf);
        }
    }

    // パネルの分解（列ごとの反射で、行列積にならない）は逐次にしか進められないので、前のパネルによる残りの更新と重ねる。
    // 列の分担はパネルごとに区切り直すが、境界は少しずつしか動かないので、各列はほぼ同じスレッドのキャッシュに残る。
    // 各列の計算は1スレッドの場合と同じなので、結果も一致する
    // 各スレッドが担当の列にブロック反射を掛け、 0 番（呼び出し元）は次のパネルの列を先に更新して分解しておく（先読み）。
    /// <summary>
    /// <see cref="Factor"/> の並列版
    /// </summary>
    private static unsafe void FactorParallel(int threads, int m, int n, Span<double> a, int lda, Span<double> tau)
    {
        int k = Math.Min(m, n);
        int nb = FactorBlockSize(k);
        var team = ColumnTeam.Rent(threads, 0);
        double[] buffers = WorkArrayPool<double>.Rent(checked(2 * (m * nb + nb * nb)));
        try
        {
            fixed (double* pa = a)
            fixed (double* ptau = tau)
            fixed (double* pbuf = buffers)
            {
                double* sa = pa;
                double* stau = ptau;
                double* sbuf = pbuf;

                // 先頭のパネルを分解する。V と T は2組の領域を交互に使う
                int jb0 = Math.Min(nb, k);
                FactorPanel(m, jb0, sa, lda, stau, sbuf, sbuf + m * nb);
                int current = 0;
                for (int j = 0; j < k; j += nb)
                {
                    int jb = Math.Min(nb, k - j);
                    int next = j + jb;
                    if (next >= n)
                    {
                        break;
                    }

                    int nextWidth = next < k ? Math.Min(nb, k - next) : 0;
                    int rows = m - j;
                    double* v = sbuf + current * (m * nb + nb * nb);
                    double* t = v + m * nb;
                    double* vNext = sbuf + (1 - current) * (m * nb + nb * nb);

                    // 次のパネルは分解の手間を見込んで重みを3倍にし、0 番に割り当てる
                    team.Partition(n - next, nextWidth, leadWeight: 3.0);
                    var work = new FactorStepWork(team.Ranges, m, nb, j, rows, jb, next, nextWidth, sa, lda, stau, v, t, vNext);
                    Parallelism.ForStatic(threads, ref work);
                    current = 1 - current;
                }
            }
        }
        finally
        {
            WorkArrayPool<double>.Return(buffers);
            ColumnTeam.Return(team);
        }
    }

    /// <summary>
    /// パネル <c>j</c> のブロック反射を担当の列に掛け、0 番は次のパネルを先に更新して分解する作業
    /// </summary>
    private readonly unsafe struct FactorStepWork(
        ColumnRanges ranges, int m, int nb, int j, int rows, int jb, int next, int nextWidth, double* a, int lda, double* tau, double* v, double* t, double* vNext)
        : IParallelWork
    {
        public void Invoke(int th)
        {
            var (lo, hi) = ranges.Range(th, 0);
            if (th == 0 && nextWidth > 0)
            {
                ApplyBlock(rows, jb, v, t, a + j + (nint)next * lda, lda, nextWidth);
                FactorPanel(m - next, nextWidth, a + next + (nint)next * lda, lda, tau + next, vNext, vNext + m * nb);
                lo = Math.Max(lo, nextWidth);
            }

            if (hi > lo)
            {
                ApplyBlock(rows, jb, v, t, a + j + (nint)(next + lo) * lda, lda, hi - lo);
            }
        }
    }

    // 幅が BlockSize 以下なら非ブロック版、それより広ければ再帰版で分解する
    // t、jb × jb。
    // v、rows × jb。
    /// <summary>
    /// パネル（<c>rows × jb</c>）を分解し、<paramref name="formBlock"/> なら Vと Tも作る
    /// </summary>
    internal static unsafe void FactorPanel(int rows, int jb, Span<double> panel, int lda, Span<double> tau, Span<double> v, Span<double> t, bool formBlock)
    {
        fixed (double* pp = panel)
        fixed (double* ptau = tau)
        fixed (double* pv = v)
        fixed (double* pt = t)
        {
            FactorPanel(rows, jb, pp, lda, ptau, pv, pt, formBlock);
        }
    }

    /// <inheritdoc cref="FactorPanel(int, int, Span{double}, int, Span{double}, Span{double}, Span{double}, bool)"/>
    private static unsafe void FactorPanel(int rows, int jb, double* panel, int lda, double* tau, double* v, double* t, bool formBlock = true)
    {
        if (jb > BlockSize)
        {
            // 再帰版は V と T を分解と同時に作る
            FactorRecursive(rows, jb, panel, lda, tau, v, rows, t, jb);
            return;
        }

        FactorUnblocked(rows, jb, new Span<double>(panel, (jb - 1) * lda + rows), lda, new Span<double>(tau, jb));
        if (formBlock)
        {
            FormBlock(rows, jb, panel, lda, tau, v, t);
        }
    }

    // 列を半分に分け、左半分を分解して右半分にブロック反射を掛けてから、右半分の下の部分を分解する。
    // T は T = [T₁, -T₁(V₁ᵀV₂)T₂; 0, T₂] で組み立てる。演算のほとんどが行列積になるので、
    // パネルの列ごとに残りのパネル全体を読み直す非ブロック版より速い
    // Elmroth–Gustavson、LAPACK dgeqrt3
    // m × n の A をその場で分解し（m ≥ n）、 単位下台形の V（m × n）と H₀⋯Hₙ₋₁ = I - VTVᵀ となる上三角の T（n × n、下三角は0）も作る。
    /// <summary>
    /// 再帰的なパネル分解
    /// </summary>
    private static unsafe void FactorRecursive(int m, int n, double* a, int lda, double* tau, double* v, int ldv, double* t, int ldt)
    {
        if (n <= RecursiveLeaf)
        {
            FactorUnblocked(m, n, new Span<double>(a, (n - 1) * lda + m), lda, new Span<double>(tau, n));
            var vs = new Span<double>(v, (n - 1) * ldv + m);
            Householder.ExtractV(m, n, new ReadOnlySpan<double>(a, (n - 1) * lda + m), lda, vs, ldv);
            Householder.FormT(m, n, vs, ldv, new ReadOnlySpan<double>(tau, n), new Span<double>(t, (n - 1) * ldt + n), ldt);
            return;
        }

        int n1 = n / 2;
        int n2 = n - n1;

        // 左半分を分解し、右半分に (I - V₁T₁V₁ᵀ)ᵀ を掛ける
        FactorRecursive(m, n1, a, lda, tau, v, ldv, t, ldt);
        Householder.ApplyBlockLeft(
            true, m, n2, n1, new ReadOnlySpan<double>(v, (n1 - 1) * ldv + m), ldv, new ReadOnlySpan<double>(t, (n1 - 1) * ldt + n1), ldt,
            new Span<double>(a + (nint)n1 * lda, (n2 - 1) * lda + m), lda);

        // 右半分の下の部分を分解する。V₂ の上 n1 行は0
        double* v2 = v + (nint)n1 * ldv;
        double* t2 = t + n1 + (nint)n1 * ldt;
        FactorRecursive(m - n1, n2, a + n1 + (nint)n1 * lda, lda, tau + n1, v2 + n1, ldv, t2, ldt);
        for (int j = 0; j < n2; j++)
        {
            new Span<double>(v2 + (nint)j * ldv, n1).Clear();
        }

        for (int j = 0; j < n1; j++)
        {
            new Span<double>(t + n1 + (nint)j * ldt, n2).Clear();
        }

        // T₁₂ = -T₁ (V₁ᵀV₂) T₂（V₂ の上 n1 行は0なので、V₁ᵀV₂ は n1 行目以降だけで計算する）
        double[] rented = WorkArrayPool<double>.Rent(checked(2 * n1 * n2));
        var w = rented.AsSpan(0, n1 * n2);
        var tw = rented.AsSpan(n1 * n2, n1 * n2);
        w.Clear();
        tw.Clear();
        Gemm.Multiply(
            true, false, n1, n2, m - n1, 1.0, new ReadOnlySpan<double>(v + n1, (n1 - 1) * ldv + m - n1), ldv,
            new ReadOnlySpan<double>(v2 + n1, (n2 - 1) * ldv + m - n1), ldv, w, n1);
        Gemm.Multiply(false, false, n1, n2, n1, 1.0, new ReadOnlySpan<double>(t, (n1 - 1) * ldt + n1), ldt, w, n1, tw, n1);
        var t12 = new Span<double>(t + (nint)n1 * ldt, (n2 - 1) * ldt + n1);
        for (int j = 0; j < n2; j++)
        {
            t12.Slice(j * ldt, n1).Clear();
        }

        Gemm.Multiply(false, false, n1, n2, n2, -1.0, tw, n1, new ReadOnlySpan<double>(t2, (n2 - 1) * ldt + n2), ldt, t12, ldt);
        WorkArrayPool<double>.Return(rented);
    }

    // t、jb × jb。
    // v。
    /// <summary>
    /// 分解済みのパネル（<c>rows × jb</c>）から Vと Tを作る
    /// </summary>
    private static unsafe void FormBlock(int rows, int jb, double* panel, int lda, double* tau, double* v, double* t)
    {
        var vs = new Span<double>(v, rows * jb);
        Householder.ExtractV(rows, jb, new ReadOnlySpan<double>(panel, (jb - 1) * lda + rows), lda, vs, rows);
        Householder.FormT(rows, jb, vs, rows, new ReadOnlySpan<double>(tau, jb), new Span<double>(t, jb * jb), jb);
    }

    // C は rows × cols。
    /// <summary>
    /// <c>C ← (I - VTᵀVᵀ)C</c>
    /// </summary>
    private static unsafe void ApplyBlock(int rows, int jb, double* v, double* t, double* c, int ldc, int cols) =>
        Householder.ApplyBlockLeft(
            true, rows, cols, jb, new ReadOnlySpan<double>(v, rows * jb), rows, new ReadOnlySpan<double>(t, jb * jb), jb,
            new Span<double>(c, (cols - 1) * ldc + rows), ldc);

    // LAPACK dgeqr2
    /// <summary>
    /// 非ブロック版
    /// </summary>
    internal static void FactorUnblocked(int m, int n, Span<double> a, int lda, Span<double> tau)
    {
        int k = Math.Min(m, n);
        for (int i = 0; i < k; i++)
        {
            int ii = i + i * lda;
            double alpha = a[ii];
            double t = Householder.Generate(ref alpha, a.Slice(ii + 1, m - i - 1));
            tau[i] = t;
            if (i < n - 1 && t != 0)
            {
                a[ii] = 1;
                Householder.ApplyLeft(a.Slice(ii, m - i), t, m - i, n - i - 1, a[(ii + lda)..], lda);
            }

            a[ii] = alpha;
        }
    }

    // k は反射の個数（k ≤ qColumns ≤ m）。
    /// <summary>
    /// <see cref="Factor"/> の結果から直交行列 <c>Q</c> の先頭 <paramref name="qColumns"/> 列（<c>m × qColumns</c>）を作る
    /// </summary>
    internal static void FormQ(int m, int qColumns, int k, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> tau, Span<double> q, int ldq)
    {
        for (int j = 0; j < qColumns; j++)
        {
            var qj = q.Slice(j * ldq, m);
            qj.Clear();
            qj[j] = 1;
        }

        if (k == 0)
        {
            return;
        }

        int nb = k >= LargeThreshold ? FormQLargeBlockSize : BlockSize;
        if (k <= Crossover)
        {
            FormQUnblocked(m, qColumns, k, a, lda, tau, q, ldq);
            return;
        }

        ApplyBlocks(m, qColumns, k, nb, a, lda, tau, q, ldq, formQ: true);
    }

    // LAPACK dormqr 'L','N'
    /// <summary>
    /// <see cref="Factor"/> の反射の積 <c>Q = H₀H₁⋯Hₖ₋₁</c>（<c>m × m</c>）を <c>C</c>（<c>m × n</c>）に左から掛ける
    /// </summary>
    internal static void ApplyQ(int m, int n, int k, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> tau, Span<double> c, int ldc)
    {
        if (k == 0 || n == 0)
        {
            return;
        }

        const int nb = 64;
        if (k <= Crossover)
        {
            double[] vbuf = WorkArrayPool<double>.Rent(m);
            var v = vbuf.AsSpan(0, m);
            for (int i = k - 1; i >= 0; i--)
            {
                int len = m - i;
                v[0] = 1;
                a.Slice(i + 1 + i * lda, len - 1).CopyTo(v[1..]);
                Householder.ApplyLeft(v, tau[i], len, n, c[i..], ldc);
            }

            WorkArrayPool<double>.Return(vbuf);
            return;
        }

        ApplyBlocks(m, n, k, nb, a, lda, tau, c, ldc, formQ: false);
    }

    // 複数のスレッドで計算する場合は、まず全ブロックの V・T を分担して作り、
    // 次に C の列を各スレッドに割り当てて、各スレッドが自分の列に全ブロックを掛ける（同期は2回だけで、
    // C の担当部分は各コアのキャッシュに残る）。列ごとの計算は1スレッドと同じなので、結果も一致する
    // formQ なら C は単位行列から始めた Q で、ブロック j は列 j 以降にだけ掛ける。
    /// <summary>
    /// 反射を <paramref name="nb"/> 本ずつのブロック <c>I - VTVᵀ</c> にまとめ、後ろのブロックから順に <c>C</c> に左から掛ける
    /// </summary>
    private static unsafe void ApplyBlocks(
        int m, int n, int k, int nb, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> tau, Span<double> c, int ldc, bool formQ)
    {
        int blocks = (k + nb - 1) / nb;
        int threads = Parallelism.Degree((double)m * n * k, 4 * Parallelism.Grain);
        threads = Math.Min(threads, n / 16);
        if (threads <= 1)
        {
            double[] vbuf = WorkArrayPool<double>.Rent(checked(m * nb));
            double[] tbuf = WorkArrayPool<double>.Rent(checked(nb * nb));
            try
            {
                for (int b = blocks - 1; b >= 0; b--)
                {
                    int j = b * nb;
                    int jb = Math.Min(nb, k - j);
                    int rows = m - j;
                    Householder.ExtractV(rows, jb, a[(j + j * lda)..], lda, vbuf, rows);
                    Householder.FormT(rows, jb, vbuf, rows, tau[j..], tbuf, jb);
                    int first = formQ ? j : 0;
                    Householder.ApplyBlockLeft(false, rows, n - first, jb, vbuf, rows, tbuf, jb, c[(j + first * ldc)..], ldc);
                }
            }
            finally
            {
                WorkArrayPool<double>.Return(vbuf);
                WorkArrayPool<double>.Return(tbuf);
            }

            return;
        }

        // ブロック b の V は vOffset[b] から（rows × jb）、T は b·nb² から
        long[] vOffset = WorkArrayPool<long>.Rent(blocks + 1);
        int[] bounds = WorkArrayPool<int>.Rent(threads + 1);
        double[]? vAll = null;
        double[]? tAll = null;
        try
        {
            vOffset[0] = 0;
            for (int b = 0; b < blocks; b++)
            {
                vOffset[b + 1] = vOffset[b] + (long)(m - b * nb) * Math.Min(nb, k - b * nb);
            }

            // 列 c に掛かるブロックの演算量に比例した重みで列を分ける（formQ では列 c にはブロック j ≤ c だけが掛かる）
            {
                double total = 0;
                for (int col = 0; col < n; col++)
                {
                    total += Weight(vOffset, col, nb, blocks, formQ);
                }

                bounds[0] = 0;
                double sum = 0;
                int t = 1;
                for (int col = 0; col < n && t < threads; col++)
                {
                    sum += Weight(vOffset, col, nb, blocks, formQ);
                    while (t < threads && sum >= total * t / threads)
                    {
                        bounds[t] = Math.Max(bounds[t - 1], (col + 8) & ~7);
                        t++;
                    }
                }

                for (; t <= threads; t++)
                {
                    bounds[t] = n;
                }

                for (t = 1; t < threads; t++)
                {
                    bounds[t] = Math.Min(bounds[t], n);
                }
            }

            vAll = WorkArrayPool<double>.Rent((int)vOffset[blocks]);
            tAll = WorkArrayPool<double>.Rent(checked(blocks * nb * nb));
            fixed (double* pa = a)
            fixed (double* ptau = tau)
            fixed (double* pc = c)
            fixed (double* pv = vAll)
            fixed (double* pt = tAll)
            fixed (long* po = vOffset)
            fixed (int* pb = bounds)
            {
                var form = new FormBlocksWork(m, k, nb, pa, a.Length, lda, ptau, pv, pt, po);
                Parallelism.For(blocks, threads, ref form);
                var apply = new ApplyBlocksWork(m, k, nb, blocks, formQ, pb, po, pv, pt, pc, ldc);
                Parallelism.ForStatic(threads, ref apply);
            }
        }
        finally
        {
            if (vAll is not null)
            {
                WorkArrayPool<double>.Return(vAll);
            }

            if (tAll is not null)
            {
                WorkArrayPool<double>.Return(tAll);
            }

            WorkArrayPool<long>.Return(vOffset);
            WorkArrayPool<int>.Return(bounds);
        }

        static double Weight(long[] vOffset, int col, int nb, int blocks, bool formQ) =>
            formQ ? vOffset[Math.Min(blocks, col / nb + 1)] : vOffset[blocks];
    }

    /// <summary>
    /// ブロック <c>b</c> の V・T を作る作業
    /// </summary>
    private readonly unsafe struct FormBlocksWork(int m, int k, int nb, double* a, int aLength, int lda, double* tau, double* v, double* t, long* vOffset)
        : IParallelWork
    {
        public void Invoke(int b)
        {
            int j = b * nb;
            int jb = Math.Min(nb, k - j);
            int rows = m - j;
            var vb = new Span<double>(v + vOffset[b], rows * jb);
            Householder.ExtractV(rows, jb, new ReadOnlySpan<double>(a + j + (nint)j * lda, aLength - j - j * lda), lda, vb, rows);
            Householder.FormT(rows, jb, vb, rows, new ReadOnlySpan<double>(tau + j, jb), new Span<double>(t + (nint)b * nb * nb, nb * nb), jb);
        }
    }

    /// <summary>
    /// 担当の列 <c>[bounds[t], bounds[t + 1])</c> に全ブロックを後ろから掛ける作業 <c>t</c>
    /// </summary>
    private readonly unsafe struct ApplyBlocksWork(int m, int k, int nb, int blocks, bool formQ, int* bounds, long* vOffset, double* v, double* t, double* c, int ldc)
        : IParallelWork
    {
        public void Invoke(int index)
        {
            int lo = bounds[index];
            int hi = bounds[index + 1];
            for (int b = blocks - 1; b >= 0; b--)
            {
                int j = b * nb;
                int first = formQ ? Math.Max(lo, j) : lo;
                if (first >= hi)
                {
                    continue;
                }

                int jb = Math.Min(nb, k - j);
                int rows = m - j;
                Householder.ApplyBlockLeft(
                    false, rows, hi - first, jb, new ReadOnlySpan<double>(v + vOffset[b], rows * jb), rows,
                    new ReadOnlySpan<double>(t + (nint)b * nb * nb, nb * nb), jb,
                    new Span<double>(c + j + (nint)first * ldc, (hi - first - 1) * ldc + rows), ldc);
            }
        }
    }

    // LAPACK dorg2r に相当
    // Q は単位行列で初期化済み
    /// <summary>
    /// 非ブロック版
    /// </summary>
    private static void FormQUnblocked(int m, int qColumns, int k, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> tau, Span<double> q, int ldq)
    {
        double[] rented = WorkArrayPool<double>.Rent(m);
        var v = rented.AsSpan(0, m);
        for (int i = k - 1; i >= 0; i--)
        {
            int len = m - i;
            v[0] = 1;
            a.Slice(i + 1 + i * lda, len - 1).CopyTo(v[1..]);
            Householder.ApplyLeft(v, tau[i], len, qColumns - i, q[(i + i * ldq)..], ldq);
        }

        WorkArrayPool<double>.Return(rented);
    }
}
