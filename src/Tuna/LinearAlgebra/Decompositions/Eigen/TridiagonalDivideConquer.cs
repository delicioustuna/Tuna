using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Tuna.LinearAlgebra;

// 三重対角行列を半分に分けて再帰的に解き、階数1の修正 D + ρzzᵀ の固有値問題（永年方程式）で統合する。
// 統合では LAPACK dlaed2 と同じ減次（小さい zᵢ と近接した固有値の組）を行い、
// Gu–Eisenstat の方法で z を再計算して固有ベクトルの直交性を保つ。
// 固有ベクトルの更新は上半分・下半分の非零構造を使い分けた2つの行列積で行う。
// Cuppen 法、LAPACK dstedc / dlaed0–dlaed4 相当
/// <summary>
/// 対称三重対角行列の固有値分解の分割統治法
/// </summary>
internal static class TridiagonalDivideConquer
{
    // LAPACK の SMLSIZ と同じ 25
    /// <summary>
    /// これ以下の大きさの部分問題は QL 法で解く
    /// </summary>
    internal const int LeafSize = 25;

    /// <summary>
    /// 三重対角行列（対角 <paramref name="d"/>、副対角 <paramref name="e"/>（長さ n-1 以上））の固有値を昇順に <paramref name="d"/> へ、固有ベクトルを <paramref name="q"/>（<c>n × n</c>）へ求める
    /// </summary>
    /// <returns>
    /// 収束したら真。
    /// </returns>
    internal static bool Solve(int n, Span<double> d, Span<double> e, Span<double> q, int ldq)
    {
        for (int j = 0; j < n; j++)
        {
            var col = q.Slice(j * ldq, n);
            col.Clear();
            col[j] = 1;
        }

        if (n <= 1)
        {
            return true;
        }

        // 減次の閾値を行列の尺度によらないものにするため、最大絶対値を [1, 2) に正規化して解く（e は終了時に戻す）
        double scale = Scaling.UnitFactor(d[..n], e[..(n - 1)]);
        if (scale != 1)
        {
            Level1Kernels.Scale(scale, d[..n]);
            Level1Kernels.Scale(scale, e[..(n - 1)]);
        }

        double[] work = WorkArrayPool<double>.Rent(checked(2 * n * n + 8 * n));
        int[] iwork = WorkArrayPool<int>.Rent(checked(4 * n));
        try
        {
            int threads = Parallelism.Degree((double)n * n * n, Parallelism.Grain);
            return threads > 1 && n > 4 * LeafSize
                ? SolveParallel(n, d, e, q, ldq, work, iwork, threads)
                : SolveRange(0, n, d, e, q, ldq, work, iwork);
        }
        finally
        {
            WorkArrayPool<double>.Return(work);
            WorkArrayPool<int>.Return(iwork);
            if (scale != 1)
            {
                Level1Kernels.Scale(1.0 / scale, d[..n]);
                Level1Kernels.Scale(1.0 / scale, e[..(n - 1)]);
            }
        }
    }

    // 最上段の統合だけは単独で実行し、その中の永年方程式と行列積を並列化する。
    /// <summary>
    /// 再帰の上の数段を幅優先に展開し、同じ深さの部分問題（とその統合）を別々のスレッドで解く
    /// </summary>
    private static unsafe bool SolveParallel(int n, Span<double> d, Span<double> e, Span<double> q, int ldq, double[] work, int[] iwork, int threads)
    {
        // スレッド数の2倍以上の部分問題に分ける（大きさの偏りを動的な割り当てでならす）
        int depth = 1;
        while ((1 << depth) < 2 * threads)
        {
            depth++;
        }

        // 段 t の部分問題（lo, size の組、高々 2^t 個）を nodes の 2(2^t - 1) から並べる。
        // 分割した部分問題だけを同じ段の先頭に詰め直し（統合で使う）、分割しない部分問題は葉として leaves に並べる
        int capacity = (2 << depth) - 1;
        int[] buffer = WorkArrayPool<int>.Rent(checked(4 * capacity));
        try
        {
            fixed (int* nodes = buffer)
            fixed (double* pd = d)
            fixed (double* pe = e)
            fixed (double* pq = q)
            {
                int* leaves = nodes + 2 * capacity;
                int leafCount = 0;
                Span<int> counts = stackalloc int[depth + 1];
                nodes[0] = 0;
                nodes[1] = n;
                counts[0] = 1;
                for (int t = 0; t < depth; t++)
                {
                    int* current = Level(nodes, t);
                    int* next = Level(nodes, t + 1);
                    int split = 0;
                    int children = 0;
                    for (int i = 0; i < counts[t]; i++)
                    {
                        int lo = current[2 * i];
                        int size = current[2 * i + 1];
                        if (size <= LeafSize)
                        {
                            leaves[2 * leafCount] = lo;
                            leaves[2 * leafCount + 1] = size;
                            leafCount++;
                            continue;
                        }

                        // SolveRange と同じ分割と対角の修正
                        int n1 = size / 2;
                        int cut = lo + n1;
                        double rho = e[cut - 1];
                        d[cut - 1] -= Math.Abs(rho);
                        d[cut] -= Math.Abs(rho);
                        current[2 * split] = lo;
                        current[2 * split + 1] = size;
                        split++;
                        next[2 * children] = lo;
                        next[2 * children + 1] = n1;
                        next[2 * children + 2] = cut;
                        next[2 * children + 3] = size - n1;
                        children += 2;
                    }

                    counts[t] = split;
                    counts[t + 1] = children;
                }

                new ReadOnlySpan<int>(Level(nodes, depth), 2 * counts[depth]).CopyTo(new Span<int>(leaves + 2 * leafCount, 2 * counts[depth]));
                leafCount += counts[depth];
                int failed = 0;

                // 各スレッドは自分の作業領域を使う
                var solve = new RangeWork(pd, d.Length, pe, e.Length, pq, q.Length, ldq, leaves, merge: false, &failed);
                Parallelism.For(leafCount, threads, ref solve);
                for (int t = depth - 1; t >= 1 && failed == 0; t--)
                {
                    var merging = new RangeWork(pd, d.Length, pe, e.Length, pq, q.Length, ldq, Level(nodes, t), merge: true, &failed);
                    if (counts[t] >= threads)
                    {
                        Parallelism.For(counts[t], threads, ref merging);
                    }
                    else
                    {
                        // スレッドより部分問題が少ない段は1つずつ統合し、統合の中を並列化する
                        for (int i = 0; i < counts[t]; i++)
                        {
                            merging.Invoke(i);
                        }
                    }
                }

                if (failed != 0)
                {
                    return false;
                }
            }
        }
        finally
        {
            WorkArrayPool<int>.Return(buffer);
        }

        return Merge(n, n / 2, d[..n], q, ldq, e[n / 2 - 1], work, iwork);

        static int* Level(int* nodes, int t) => nodes + 2 * ((1 << t) - 1);
    }

    // 統合の結果は（Merge が常に真を返すので）失敗を伝えない。
    /// <summary>
    /// <paramref name="nodes"/> の i 番目の部分問題を解く（<paramref name="merge"/> なら2つの半分を統合する）作業
    /// </summary>
    private readonly unsafe struct RangeWork(
        double* d, int dLength, double* e, int eLength, double* q, int qLength, int ldq, int* nodes, bool merge, int* failed) : IParallelWork
    {
        public void Invoke(int i)
        {
            int lo = nodes[2 * i];
            int size = nodes[2 * i + 1];
            double[] w = WorkArrayPool<double>.Rent(checked(2 * size * size + 8 * size));
            int[] iw = WorkArrayPool<int>.Rent(checked(4 * size));
            try
            {
                var dd = new Span<double>(d, dLength);
                var qq = new Span<double>(q, qLength);
                bool ok = merge
                    ? Merge(size, size / 2, dd.Slice(lo, size), qq[(lo + lo * ldq)..], ldq, e[lo + size / 2 - 1], w, iw)
                    : SolveRange(lo, size, dd, new Span<double>(e, eLength), qq, ldq, w, iw);
                if (!ok && !merge)
                {
                    Volatile.Write(ref *failed, 1);
                }
            }
            finally
            {
                WorkArrayPool<double>.Return(w);
                WorkArrayPool<int>.Return(iw);
            }
        }
    }

    // q の対応する対角ブロックに固有ベクトルを書く。
    /// <summary>
    /// 添字 <c>[lo, lo+size)</c> の部分問題を解く
    /// </summary>
    private static bool SolveRange(int lo, int size, Span<double> d, Span<double> e, Span<double> q, int ldq, double[] work, int[] iwork)
    {
        var qb = q[(lo + lo * ldq)..];
        if (size <= LeafSize)
        {
            var el = work.AsSpan(0, size);
            e.Slice(lo, size - 1).CopyTo(el);
            return TridiagonalEigen.QlImplicit(size, d.Slice(lo, size), el, qb, ldq, true);
        }

        int n1 = size / 2;
        int cut = lo + n1;

        // T = diag(T1', T2') + |ρ|·wwᵀ（w は境界の2成分が 1 と sign(ρ)）
        double rho = e[cut - 1];
        d[cut - 1] -= Math.Abs(rho);
        d[cut] -= Math.Abs(rho);

        if (!SolveRange(lo, n1, d, e, q, ldq, work, iwork) || !SolveRange(cut, size - n1, d, e, q, ldq, work, iwork))
        {
            return false;
        }

        return Merge(size, n1, d.Slice(lo, size), qb, ldq, rho, work, iwork);
    }

    // LAPACK dlaed1–dlaed3
    /// <summary>
    /// 固有値・固有ベクトルが求まった2つの部分問題（各半分で昇順）を、階数1の修正で統合する
    /// </summary>
    private static unsafe bool Merge(int n, int n1, Span<double> d, Span<double> q, int ldq, double rho, double[] work, int[] iwork)
    {
        int n2 = n - n1;

        // 作業領域の割り当て
        var ws = work.AsSpan();
        var z = ws[..n];
        var w = ws.Slice(2 * n, n);
        var lambda = ws.Slice(3 * n, n);
        var dsort = ws.Slice(4 * n, n);
        var zsort = ws.Slice(5 * n, n);
        var newValues = ws.Slice(6 * n, n);
        var big = ws.Slice(8 * n, 2 * n * n);
        var perm = iwork.AsSpan(0, n);
        var nondeflated = iwork.AsSpan(n, n);
        var deflated = iwork.AsSpan(2 * n, n);
        var coltype = iwork.AsSpan(3 * n, n);

        // z = Qᵀw（上のブロックの最終行と下のブロックの先頭行）
        for (int j = 0; j < n1; j++)
        {
            z[j] = q[(n1 - 1) + j * ldq];
        }

        for (int j = n1; j < n; j++)
        {
            z[j] = q[n1 + j * ldq];
        }

        if (rho < 0)
        {
            Level1Kernels.Scale(-1.0, z[n1..]);
        }

        // ‖z‖ = √2 なので正規化して ρ に繰り込む
        Level1Kernels.Scale(1.0 / Math.Sqrt(2.0), z);
        rho = Math.Abs(2 * rho);

        // 2つの昇順の列を併合した順序
        {
            int a = 0;
            int b = n1;
            for (int t = 0; t < n; t++)
            {
                perm[t] = (b >= n || (a < n1 && d[a] <= d[b])) ? a++ : b++;
            }
        }

        for (int j = 0; j < n; j++)
        {
            coltype[j] = j < n1 ? 1 : 3;
        }

        // 減次の閾値
        double zmax = 0;
        double dmax = 0;
        for (int j = 0; j < n; j++)
        {
            zmax = Math.Max(zmax, Math.Abs(z[j]));
            dmax = Math.Max(dmax, Math.Abs(d[j]));
        }

        double tol = 8 * NumericConstants.UnitRoundoff * Math.Max(dmax, zmax);
        int k = 0;
        int nd = 0;
        if (rho * zmax <= tol)
        {
            // すべて減次（統合は並べ替えだけ）
            for (int t = 0; t < n; t++)
            {
                deflated[nd++] = perm[t];
            }
        }
        else
        {
            int pj = -1;
            for (int t = 0; t < n; t++)
            {
                int nj = perm[t];
                if (rho * Math.Abs(z[nj]) <= tol)
                {
                    deflated[nd++] = nj;
                    continue;
                }

                if (pj < 0)
                {
                    pj = nj;
                    continue;
                }

                // 近接した固有値の組: 回転で z[pj] を消せるなら pj を減次する
                double s = z[pj];
                double c = z[nj];
                double tau = Householder.Hypot(c, s);
                double tdiff = d[nj] - d[pj];
                c /= tau;
                s = -s / tau;
                if (Math.Abs(tdiff * c * s) <= tol)
                {
                    z[nj] = tau;
                    z[pj] = 0;
                    if (coltype[nj] != coltype[pj])
                    {
                        coltype[nj] = 2;
                    }

                    coltype[pj] = 4;
                    Level1Kernels.Rotate(q.Slice(pj * ldq, n), q.Slice(nj * ldq, n), c, s);
                    double tt = d[pj] * c * c + d[nj] * s * s;
                    d[nj] = d[pj] * s * s + d[nj] * c * c;
                    d[pj] = tt;
                    deflated[nd++] = pj;
                    pj = nj;
                }
                else
                {
                    nondeflated[k++] = pj;
                    pj = nj;
                }
            }

            if (pj >= 0)
            {
                nondeflated[k++] = pj;
            }
        }

        if (k > 0)
        {
            // 永年方程式 1 + ρ Σ zᵢ²/(δᵢ - λ) = 0 の根と差 δⱼ - λᵢ（列 i）
            var delta = big[..(k * k)];
            for (int i = 0; i < k; i++)
            {
                dsort[i] = d[nondeflated[i]];
                zsort[i] = z[nondeflated[i]];
            }

            // 各根・各成分・各列の計算は独立なので、範囲ごとに別のスレッドで行う（作業領域は配列の位置で受け渡す）
            int threads = Parallelism.Degree(30.0 * k * k, Parallelism.Grain);
            int chunks = threads > 1 ? Math.Min(k, 4 * threads) : 1;
            int chunk = (k + chunks - 1) / chunks;
            var layout = new MergeLayout(k, chunk, 4 * n, 5 * n, 8 * n, 2 * n, 3 * n);
            fixed (double* pw = work)
            {
                var roots = new SecularRootsWork(layout, pw, rho);
                Parallelism.For(chunks, threads, ref roots);

                // Gu–Eisenstat: 計算した根に対して厳密に成り立つ z を作り直す（行 i の範囲ごと）
                var recompute = new RecomputeZWork(layout, pw);
                Parallelism.For(chunks, threads, ref recompute);
            }

            // U の行を列の型（1: 上だけ、2: 両方、3: 下だけ）の順に並べ替える（LAPACK dlaed3 と同じ）。
            // 上の n1 行の積は行 [0, c12)、下の n2 行の積は行 [c1, k) の連続した範囲だけを使う
            int c1 = 0;
            int c12 = 0;
            {
                int fill = 0;
                for (int pass = 1; pass <= 3; pass++)
                {
                    for (int i = 0; i < k; i++)
                    {
                        int type = coltype[nondeflated[i]];
                        if (type == pass)
                        {
                            perm[fill++] = i;
                        }
                    }

                    if (pass == 1)
                    {
                        c1 = fill;
                    }
                    else if (pass == 2)
                    {
                        c12 = fill;
                    }
                }
            }

            int[] rowOrder = WorkArrayPool<int>.Rent(k);
            perm[..k].CopyTo(rowOrder);

            // 修正後の固有ベクトル（列 j）: uᵢ = ẑᵢ / (δᵢ - λⱼ) を正規化し、行を型の順に並べて delta を上書き
            fixed (double* pw = work)
            fixed (int* po = rowOrder)
            {
                var vectors = new VectorsWork(layout, pw, po);
                Parallelism.For(chunks, threads, ref vectors);
            }

            // 新しい固有ベクトル = Q（非減次の列、型の順）× U
            var newQ = big.Slice(k * k, n * k);
            MultiplyGrouped(n1, 0, 0, c12, k, rowOrder, nondeflated, q, ldq, delta, newQ, n);
            MultiplyGrouped(n2, n1, c1, k, k, rowOrder, nondeflated, q, ldq, delta, newQ, n);
            WorkArrayPool<int>.Return(rowOrder);

            // 減次した列を後ろへ、非減次の新しい列を前へ（固有値の昇順に並べ直す）
            for (int i = 0; i < k; i++)
            {
                newValues[i] = lambda[i];
            }

            for (int i = 0; i < nd; i++)
            {
                newValues[k + i] = d[deflated[i]];
            }

            // 減次した列の内容を退避してから書き戻す
            var deflatedCols = big.Slice(k * k + n * k, Math.Max(nd, 0) * n);
            if (deflatedCols.Length < nd * n)
            {
                throw new InvalidOperationException("Workspace too small.");
            }

            for (int i = 0; i < nd; i++)
            {
                q.Slice(deflated[i] * ldq, n).CopyTo(deflatedCols.Slice(i * n, n));
            }

            // 全体を昇順に並べる
            var order = iwork.AsSpan(0, n);
            SortIndices(newValues[..n], order);
            for (int t = 0; t < n; t++)
            {
                int src = order[t];
                d[t] = newValues[src];
                var dst = q.Slice(t * ldq, n);
                if (src < k)
                {
                    newQ.Slice(src * n, n).CopyTo(dst);
                }
                else
                {
                    deflatedCols.Slice((src - k) * n, n).CopyTo(dst);
                }
            }
        }
        else
        {
            // すべて減次: 固有値と列を昇順に並べ替えるだけ
            var tmp = big[..(n * n)];
            for (int t = 0; t < n; t++)
            {
                q.Slice(deflated[t] * ldq, n).CopyTo(tmp.Slice(t * n, n));
                newValues[t] = d[deflated[t]];
            }

            var order = iwork.AsSpan(0, n);
            SortIndices(newValues[..n], order);
            for (int t = 0; t < n; t++)
            {
                d[t] = newValues[order[t]];
                tmp.Slice(order[t] * n, n).CopyTo(q.Slice(t * ldq, n));
            }
        }

        return true;
    }

    // 作業領域（Merge の work）の位置
    /// <summary>
    /// 統合の並列区間に渡す、非減次の個数 <c>k</c>・範囲の幅と作業領域の各部分の位置
    /// </summary>
    private readonly record struct MergeLayout(int K, int Chunk, int Dsort, int Zsort, int Delta, int W, int Lambda);

    /// <summary>
    /// 範囲 <c>c</c> の根 <c>λᵢ</c> と差 <c>δⱼ - λᵢ</c> を求める作業
    /// </summary>
    private readonly unsafe struct SecularRootsWork(MergeLayout layout, double* work, double rho) : IParallelWork
    {
        public void Invoke(int c)
        {
            int k = layout.K;
            int i0 = c * layout.Chunk;
            int i1 = Math.Min(k, i0 + layout.Chunk);
            double[] shifted = WorkArrayPool<double>.Rent(k);
            var ds = new ReadOnlySpan<double>(work + layout.Dsort, k);
            var zs = new ReadOnlySpan<double>(work + layout.Zsort, k);
            for (int i = i0; i < i1; i++)
            {
                work[layout.Lambda + i] = SecularRoot(k, i, ds, zs, rho, new Span<double>(work + layout.Delta + i * k, k), shifted.AsSpan(0, k));
            }

            WorkArrayPool<double>.Return(shifted);
        }
    }

    /// <summary>
    /// 範囲 <c>c</c> の行の ẑ を作り直す作業
    /// </summary>
    private readonly unsafe struct RecomputeZWork(MergeLayout layout, double* work) : IParallelWork
    {
        public void Invoke(int c)
        {
            int k = layout.K;
            int i0 = c * layout.Chunk;
            int i1 = Math.Min(k, i0 + layout.Chunk);
            if (i0 >= i1)
            {
                return;
            }

            var ds = new ReadOnlySpan<double>(work + layout.Dsort, k);
            var dl = new ReadOnlySpan<double>(work + layout.Delta, k * k);
            var wv = new Span<double>(work + layout.W, k);
            for (int i = i0; i < i1; i++)
            {
                wv[i] = dl[i + i * k];
            }

            for (int j = 0; j < k; j++)
            {
                // i ≠ j の範囲を2つに分けてベクトル化する
                var colj = dl.Slice(j * k, k);
                double dj = ds[j];
                int split = Math.Clamp(j, i0, i1);
                ScaleByRatio(wv[i0..split], colj[i0..split], ds[i0..split], dj);
                int after = Math.Clamp(j + 1, i0, i1);
                ScaleByRatio(wv[after..i1], colj[after..i1], ds[after..i1], dj);
            }

            var zsort = new ReadOnlySpan<double>(work + layout.Zsort, k);
            for (int i = i0; i < i1; i++)
            {
                wv[i] = Math.CopySign(Math.Sqrt(Math.Max(-wv[i], 0)), zsort[i]);
            }
        }
    }

    /// <summary>
    /// 範囲 <c>c</c> の列の固有ベクトルを求め、行を型の順に並べて差の列を上書きする作業
    /// </summary>
    private readonly unsafe struct VectorsWork(MergeLayout layout, double* work, int* rowOrder) : IParallelWork
    {
        public void Invoke(int c)
        {
            int k = layout.K;
            int j0 = c * layout.Chunk;
            int j1 = Math.Min(k, j0 + layout.Chunk);
            var wv = new ReadOnlySpan<double>(work + layout.W, k);
            double[] tmp = WorkArrayPool<double>.Rent(k);
            var u = tmp.AsSpan(0, k);
            for (int j = j0; j < j1; j++)
            {
                var col = new Span<double>(work + layout.Delta + j * k, k);
                Divide(wv, col, u);

                double scale = 1.0 / Level1Kernels.Norm2(u);
                for (int r = 0; r < k; r++)
                {
                    col[r] = u[rowOrder[r]] * scale;
                }
            }

            WorkArrayPool<double>.Return(tmp);
        }
    }

    // r は型の順の行番号で、列は nondeflated[rowOrder[r]]。
    /// <summary>
    /// 行 <c>[rowOffset, rowOffset+rows)</c> について <c>newQ ← Q[:, 非減次の列 r ∈ [r0, r1)]·U[r0:r1, :]</c>
    /// </summary>
    private static void MultiplyGrouped(
        int rows, int rowOffset, int r0, int r1, int k, int[] rowOrder, ReadOnlySpan<int> nondeflated,
        ReadOnlySpan<double> q, int ldq, ReadOnlySpan<double> u, Span<double> newQ, int ldNew)
    {
        var target = newQ[rowOffset..];
        for (int j = 0; j < k; j++)
        {
            target.Slice(j * ldNew, rows).Clear();
        }

        int m = r1 - r0;
        if (m == 0 || rows == 0)
        {
            return;
        }

        double[] a = WorkArrayPool<double>.Rent(checked(rows * m));
        try
        {
            for (int p = 0; p < m; p++)
            {
                q.Slice(rowOffset + nondeflated[rowOrder[r0 + p]] * ldq, rows).CopyTo(a.AsSpan(p * rows, rows));
            }

            Gemm.Multiply(false, false, rows, k, m, 1.0, a, rows, u[r0..], k, target, ldNew);
        }
        finally
        {
            WorkArrayPool<double>.Return(a);
        }
    }

    /// <summary>
    /// 値の昇順に並べた添字（挿入ソートではなく配列のソートを使う）
    /// </summary>
    private static void SortIndices(ReadOnlySpan<double> values, Span<int> order)
    {
        int n = values.Length;
        double[] keys = WorkArrayPool<double>.Rent(n);
        try
        {
            for (int i = 0; i < n; i++)
            {
                order[i] = i;
                keys[i] = values[i];
            }

            keys.AsSpan(0, n).Sort(order[..n]);
        }
        finally
        {
            WorkArrayPool<double>.Return(keys);
        }
    }

    // 根に近い極を原点にとって τ = λ - δ_org を求め、delta[j] = δⱼ - λ を高い相対精度で返す
    // 両隣の極による有理近似（middle way 法）の反復を、挟み込みの二分法で保護する。
    /// <summary>
    /// 永年方程式 <c>f(λ) = 1/ρ + Σⱼ zⱼ²/(δⱼ - λ) = 0</c> の i 番目の根（<c>δ</c> は狭義単調増加、<c>ρ &gt; 0</c>）
    /// </summary>
    internal static double SecularRoot(
        int k, int i, ReadOnlySpan<double> dd, ReadOnlySpan<double> z, double rho, Span<double> delta, Span<double> shifted,
        ReadOnlySpan<double> roots = default)
    {
        double rhoinv = 1.0 / rho;
        if (k == 1)
        {
            double t = rho * z[0] * z[0];
            delta[0] = -t;
            return dd[0] + t;
        }

        int org;
        double lo;
        double hi;
        double guess = double.NaN;
        double zz = 0;
        for (int j = 0; j < k; j++)
        {
            zz += z[j] * z[j];
        }

        if (i < k - 1)
        {
            double gap = Difference(dd, roots, i + 1, i);
            double mid = 0.5 * gap;

            // 区間の中点での符号で、根に近い側の極を原点に選ぶ
            Shift(k, dd, roots, i, shifted);

            double fmid = rhoinv + Evaluate(k, i, shifted, z, mid, out _, out _, out _, out _);

            // 初期値（LAPACK dlaed4 と同じ）: 両隣の極以外の項を中点での値で定数 c とみなし、
            // c + a/(-t) + b/(gap - t) = 0（t は極 i からの距離）の区間内の根をとる
            double a = z[i] * z[i];
            double b = z[i + 1] * z[i + 1];
            double c = fmid - a / (shifted[i] - mid) - b / (shifted[i + 1] - mid);
            if (fmid >= 0)
            {
                org = i;
                lo = 0;
                hi = mid;
            }
            else
            {
                org = i + 1;
                lo = -(gap - mid);
                hi = 0;
                Shift(k, dd, roots, i + 1, shifted);
            }

            // 原点を org にとった座標で c·(D1 - t)(D2 - t) + a·(D2 - t) + b·(D1 - t) = 0
            double d1 = shifted[i];
            double d2 = shifted[i + 1];
            guess = SolveQuadraticInBracket(c, -(c * (d1 + d2) + a + b), c * d1 * d2 + a * d2 + b * d1, lo, hi);
        }
        else
        {
            org = k - 1;
            lo = 0;
            hi = rho * zz;
            Shift(k, dd, roots, k - 1, shifted);
        }

        // 初期値が求まらなければ区間の中点から始める
        double tau = guess > lo && guess < hi ? guess : 0.5 * (lo + hi);
        double eps = 2 * NumericConstants.UnitRoundoff;
        for (int iter = 0; iter < 100; iter++)
        {
            double f = rhoinv + Evaluate(k, i, shifted, z, tau, out double psi, out double dpsi, out double phi, out double dphi);
            double errbound = 8 * (phi - psi) + 2 * rhoinv + 3 * Math.Abs(tau) * (dpsi + dphi);
            if (Math.Abs(f) <= eps * errbound)
            {
                break;
            }

            if (f > 0)
            {
                hi = tau;
            }
            else
            {
                lo = tau;
            }

            if (hi - lo <= eps * Math.Max(Math.Abs(lo), Math.Abs(hi)))
            {
                break;
            }

            double next = double.NaN;
            if (i < k - 1)
            {
                // ψ ≈ a + s1/(D1 - t)、φ ≈ b + s2/(D2 - t) と近似して c + s1/(D1 - t) + s2/(D2 - t) = 0 を解く
                double d1 = shifted[i] - tau;
                double d2 = shifted[i + 1] - tau;
                double s1 = dpsi * d1 * d1;
                double s2 = dphi * d2 * d2;
                double c = f - s1 / d1 - s2 / d2;

                // 未知数を η = t - tau（現在値からの補正）にとる: c(d1 - η)(d2 - η) + s1(d2 - η) + s2(d1 - η) = 0
                double qa = c;
                double qb = -(c * (d1 + d2) + s1 + s2);
                double qc = c * d1 * d2 + s1 * d2 + s2 * d1;
                next = tau + SolveQuadraticInBracket(qa, qb, qc, lo - tau, hi - tau);
            }
            else
            {
                // ψ ≈ a + s1/(D1 - t)、c + s1/(D1 - t) = 0
                double d1 = shifted[k - 1] - tau;
                double s1 = dpsi * d1 * d1;
                double c = f - s1 / d1;
                if (c > 0)
                {
                    next = tau + (d1 + s1 / c);
                }
            }

            if (!(next > lo && next < hi))
            {
                next = 0.5 * (lo + hi);
            }

            tau = next;
        }

        Offset(k, shifted, tau, delta);
        return dd[org] + tau;
    }

    // roots を与えた場合は Difference と同じく積の形。
    /// <summary>
    /// <c>shifted[j] = dd[j] - dd[org]</c>
    /// </summary>
    private static void Shift(int k, ReadOnlySpan<double> dd, ReadOnlySpan<double> roots, int org, Span<double> shifted)
    {
        ref double pd = ref MemoryMarshal.GetReference(dd);
        ref double ps = ref MemoryMarshal.GetReference(shifted);
        int j = 0;
        if (roots.IsEmpty)
        {
            double o = dd[org];
            if (Vector512.IsHardwareAccelerated)
            {
                var vo = Vector512.Create(o);
                for (; j <= k - 8; j += 8)
                {
                    (Vector512.LoadUnsafe(ref pd, (nuint)j) - vo).StoreUnsafe(ref ps, (nuint)j);
                }
            }

            for (; j < k; j++)
            {
                Unsafe.Add(ref ps, j) = Unsafe.Add(ref pd, j) - o;
            }
        }
        else
        {
            ref double pr = ref MemoryMarshal.GetReference(roots);
            double o = roots[org];
            if (Vector512.IsHardwareAccelerated)
            {
                var vo = Vector512.Create(o);
                for (; j <= k - 8; j += 8)
                {
                    var r = Vector512.LoadUnsafe(ref pr, (nuint)j);
                    ((r - vo) * (r + vo)).StoreUnsafe(ref ps, (nuint)j);
                }
            }

            for (; j < k; j++)
            {
                double r = Unsafe.Add(ref pr, j);
                Unsafe.Add(ref ps, j) = (r - o) * (r + o);
            }
        }
    }

    /// <summary>
    /// <c>w[i] *= num[i] / (den[i] - d)</c>
    /// </summary>
    private static void ScaleByRatio(Span<double> w, ReadOnlySpan<double> num, ReadOnlySpan<double> den, double d)
    {
        int n = w.Length;
        ref double pw = ref MemoryMarshal.GetReference(w);
        ref double pn = ref MemoryMarshal.GetReference(num);
        ref double pd = ref MemoryMarshal.GetReference(den);
        int i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            var vd = Vector512.Create(d);
            for (; i <= n - 8; i += 8)
            {
                var r = Vector512.LoadUnsafe(ref pn, (nuint)i) / (Vector512.LoadUnsafe(ref pd, (nuint)i) - vd);
                (Vector512.LoadUnsafe(ref pw, (nuint)i) * r).StoreUnsafe(ref pw, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            Unsafe.Add(ref pw, i) *= Unsafe.Add(ref pn, i) / (Unsafe.Add(ref pd, i) - d);
        }
    }

    /// <summary>
    /// <c>dest[i] = num[i] / den[i]</c>
    /// </summary>
    private static void Divide(ReadOnlySpan<double> num, ReadOnlySpan<double> den, Span<double> dest)
    {
        int n = dest.Length;
        ref double pn = ref MemoryMarshal.GetReference(num);
        ref double pd = ref MemoryMarshal.GetReference(den);
        ref double po = ref MemoryMarshal.GetReference(dest);
        int i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            for (; i <= n - 8; i += 8)
            {
                (Vector512.LoadUnsafe(ref pn, (nuint)i) / Vector512.LoadUnsafe(ref pd, (nuint)i)).StoreUnsafe(ref po, (nuint)i);
            }
        }

        for (; i < n; i++)
        {
            Unsafe.Add(ref po, i) = Unsafe.Add(ref pn, i) / Unsafe.Add(ref pd, i);
        }
    }

    /// <summary>
    /// <c>dest[j] = src[j] - t</c>
    /// </summary>
    private static void Offset(int k, ReadOnlySpan<double> src, double t, Span<double> dest)
    {
        ref double ps = ref MemoryMarshal.GetReference(src);
        ref double pd = ref MemoryMarshal.GetReference(dest);
        int j = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            var vt = Vector512.Create(t);
            for (; j <= k - 8; j += 8)
            {
                (Vector512.LoadUnsafe(ref ps, (nuint)j) - vt).StoreUnsafe(ref pd, (nuint)j);
            }
        }

        for (; j < k; j++)
        {
            Unsafe.Add(ref pd, j) = Unsafe.Add(ref ps, j) - t;
        }
    }

    // roots を与えた場合は dd = roots² とみなして (rₐ - r_b)(rₐ + r_b) で桁落ちなく計算する（特異値の永年方程式用）
    /// <summary>
    /// <c>dd[a] - dd[b]</c>
    /// </summary>
    private static double Difference(ReadOnlySpan<double> dd, ReadOnlySpan<double> roots, int a, int b) =>
        roots.IsEmpty ? dd[a] - dd[b] : (roots[a] - roots[b]) * (roots[a] + roots[b]);

    /// <summary>
    /// <c>qa·η² + qb·η + qc = 0</c> の根のうち (lo, hi) に入るもの
    /// </summary>
    /// <returns>
    /// なければ NaN。
    /// </returns>
    private static double SolveQuadraticInBracket(double qa, double qb, double qc, double lo, double hi)
    {
        if (qa == 0)
        {
            return qb != 0 ? -qc / qb : double.NaN;
        }

        double disc = qb * qb - 4 * qa * qc;
        if (disc < 0)
        {
            disc = 0;
        }

        double sq = Math.Sqrt(disc);
        double temp = -0.5 * (qb + Math.CopySign(sq, qb));
        double r1 = temp / qa;
        double r2 = temp != 0 ? qc / temp : double.NaN;
        bool in1 = r1 > lo && r1 < hi;
        bool in2 = r2 > lo && r2 < hi;
        if (in1 && in2)
        {
            return Math.Abs(r1) < Math.Abs(r2) ? r1 : r2;
        }

        return in1 ? r1 : in2 ? r2 : double.NaN;
    }

    /// <summary>
    /// <c>Σⱼ zⱼ²/(Δⱼ - τ)</c> を左側（<c>j ≤ i</c>）の ψ と右側の φ に分けて、導関数とともに求める
    /// </summary>
    private static double Evaluate(int k, int i, ReadOnlySpan<double> shifted, ReadOnlySpan<double> z, double tau, out double psi, out double dpsi, out double phi, out double dphi)
    {
        SumTerms(shifted[..(i + 1)], z[..(i + 1)], tau, out psi, out dpsi);
        SumTerms(shifted[(i + 1)..k], z[(i + 1)..k], tau, out phi, out dphi);
        return psi + phi;
    }

    /// <summary>
    /// <c>Σ zⱼ²/(Δⱼ - τ)</c> と <c>Σ zⱼ²/(Δⱼ - τ)²</c>
    /// </summary>
    private static void SumTerms(ReadOnlySpan<double> shifted, ReadOnlySpan<double> z, double tau, out double sum, out double dsum)
    {
        int n = shifted.Length;
        ref double ps = ref MemoryMarshal.GetReference(shifted);
        ref double pz = ref MemoryMarshal.GetReference(z);
        int j = 0;
        double s = 0;
        double ds = 0;
        if (Vector512.IsHardwareAccelerated && n >= 8)
        {
            var vt = Vector512.Create(tau);
            var acc = Vector512<double>.Zero;
            var dacc = Vector512<double>.Zero;
            for (; j <= n - 8; j += 8)
            {
                var zj = Vector512.LoadUnsafe(ref pz, (nuint)j);
                var r = zj / (Vector512.LoadUnsafe(ref ps, (nuint)j) - vt);
                acc = Vector512.FusedMultiplyAdd(zj, r, acc);
                dacc = Vector512.FusedMultiplyAdd(r, r, dacc);
            }

            s = Vector512.Sum(acc);
            ds = Vector512.Sum(dacc);
        }
        else if (Vector256.IsHardwareAccelerated && n >= 4)
        {
            var vt = Vector256.Create(tau);
            var acc = Vector256<double>.Zero;
            var dacc = Vector256<double>.Zero;
            for (; j <= n - 4; j += 4)
            {
                var zj = Vector256.LoadUnsafe(ref pz, (nuint)j);
                var r = zj / (Vector256.LoadUnsafe(ref ps, (nuint)j) - vt);
                acc += zj * r;
                dacc += r * r;
            }

            s = Vector256.Sum(acc);
            ds = Vector256.Sum(dacc);
        }

        for (; j < n; j++)
        {
            double zj = Unsafe.Add(ref pz, j);
            double r = zj / (Unsafe.Add(ref ps, j) - tau);
            s += zj * r;
            ds += r * r;
        }

        sum = s;
        dsum = ds;
    }
}
