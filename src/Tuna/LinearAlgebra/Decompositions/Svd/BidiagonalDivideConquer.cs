namespace Tuna.LinearAlgebra;

// 上二重対角行列の分割統治 SVD（LAPACK dbdsdc / dlasd0–dlasd4 相当、Gu–Eisenstat 直交性回復付き）。
internal static class BidiagonalDivideConquer
{
    private const double Epsilon = 1.1102230246251565e-16;

    // LAPACK の SMLSIZ と同じ 25。これ以下の部分問題は QR 法（BidiagonalSvd）で解く。
    internal const int LeafSize = 25;

    // n × n 上二重対角行列（対角 d、優対角 e）を B = U·diag(σ)·Vᵀ に分解し、d を特異値（降順）で上書きする。
    internal static bool Solve(int n, Span<double> d, ReadOnlySpan<double> e, Span<double> u, int ldu, Span<double> v, int ldv)
    {
        if (n == 0)
        {
            return true;
        }

        // 再帰の途中では列が1つ多い部分問題の優対角 e[lo+size-1] も使うので長さ n の作業用にコピーする
        double[] ee = WorkArrayPool<double>.Rent(n);
        try
        {
            ee.AsSpan(0, n).Clear();
            e[..(n - 1)].CopyTo(ee);

            // LAPACK dbdsdc と同じく最大絶対値を [1, 2) に正規化して解く。
            // 小さい行列では永年方程式の dⱼ² - σ² などの途中の積がアンダーフローし、特異値と減次の判定を損なう
            double scale = Scaling.UnitFactor(d[..n], ee.AsSpan(0, n - 1));
            if (scale != 1)
            {
                Level1Kernels.Scale(scale, d[..n]);
                Level1Kernels.Scale(scale, ee.AsSpan(0, n - 1));
            }

            int threads = Parallelism.Degree((double)n * n * n, Parallelism.Grain);
            bool ok = threads > 1 && n > 4 * LeafSize
                ? SolveParallel(n, d, ee, u, ldu, v, ldv, threads)
                : SolveNode(0, n, 0, d, ee, u, ldu, v, ldv);
            if (!ok)
            {
                return false;
            }

            if (scale != 1)
            {
                Level1Kernels.Scale(1.0 / scale, d[..n]);
            }
        }
        finally
        {
            WorkArrayPool<double>.Return(ee);
        }

        // 降順にする
        d[..n].Reverse();
        for (int j = 0; j < n / 2; j++)
        {
            Level1Kernels.Swap(u.Slice(j * ldu, n), u.Slice((n - 1 - j) * ldu, n));
            Level1Kernels.Swap(v.Slice(j * ldv, n), v.Slice((n - 1 - j) * ldv, n));
        }

        return true;
    }

    // 最上段の統合だけは単独で実行し、その中の永年方程式と行列積を並列化する。
    /// <summary>
    /// 再帰の上の数段を幅優先に展開し、同じ深さの部分問題（とその統合）を別々のスレッドで解く
    /// </summary>
    private static unsafe bool SolveParallel(int n, Span<double> d, Span<double> e, Span<double> u, int ldu, Span<double> v, int ldv, int threads)
    {
        int depth = 1;
        while ((1 << depth) < 2 * threads)
        {
            depth++;
        }

        // 部分問題は (lo, n, sqre)。分割は SolveNode と同じ（中央の行 k = lo + n/2 は統合で使う）。
        // 段 t の部分問題（高々 2^t 個）を nodes の 3(2^t - 1) から並べ、分割した部分問題だけを同じ段の先頭に詰め直す（統合で使う）。
        // 分割しない部分問題は葉として leaves に並べる
        int capacity = (2 << depth) - 1;
        int[] buffer = WorkArrayPool<int>.Rent(checked(6 * capacity));
        int failed = 0;
        try
        {
            fixed (int* nodes = buffer)
            fixed (double* pd = d)
            fixed (double* pe = e)
            fixed (double* pu = u)
            fixed (double* pv = v)
            {
                int* leaves = nodes + 3 * capacity;
                int leafCount = 0;
                Span<int> counts = stackalloc int[depth + 1];
                nodes[0] = 0;
                nodes[1] = n;
                nodes[2] = 0;
                counts[0] = 1;
                for (int t = 0; t < depth; t++)
                {
                    int* current = Level(nodes, t);
                    int* next = Level(nodes, t + 1);
                    int split = 0;
                    int children = 0;
                    for (int i = 0; i < counts[t]; i++)
                    {
                        int lo = current[3 * i];
                        int size = current[3 * i + 1];
                        int sqre = current[3 * i + 2];
                        if (size <= LeafSize)
                        {
                            Set(leaves, leafCount++, lo, size, sqre);
                            continue;
                        }

                        int nl = size / 2;
                        int k = lo + nl;
                        Set(current, split++, lo, size, sqre);
                        Set(next, children++, lo, nl, 1);
                        Set(next, children++, k + 1, size - nl - 1, sqre);
                    }

                    counts[t] = split;
                    counts[t + 1] = children;
                }

                new ReadOnlySpan<int>(Level(nodes, depth), 3 * counts[depth]).CopyTo(new Span<int>(leaves + 3 * leafCount, 3 * counts[depth]));
                leafCount += counts[depth];
                var system = new NodeSystem(pd, d.Length, pe, e.Length, pu, u.Length, ldu, pv, v.Length, ldv);
                var solve = new SolveNodesWork(system, leaves, &failed);
                Parallelism.For(leafCount, threads, ref solve);
                for (int t = depth - 1; t >= 1 && failed == 0; t--)
                {
                    var merge = new MergeNodesWork(system, Level(nodes, t));
                    if (counts[t] >= threads)
                    {
                        Parallelism.For(counts[t], threads, ref merge);
                    }
                    else
                    {
                        // スレッドより部分問題が少ない段は1つずつ統合し、統合の中を並列化する
                        for (int i = 0; i < counts[t]; i++)
                        {
                            merge.Invoke(i);
                        }
                    }
                }
            }
        }
        finally
        {
            WorkArrayPool<int>.Return(buffer);
        }

        if (failed != 0)
        {
            return false;
        }

        Merge(0, n / 2, n - n / 2 - 1, 0, d[n / 2], e[n / 2], d, u, ldu, v, ldv);
        return true;

        static int* Level(int* nodes, int t) => nodes + 3 * ((1 << t) - 1);

        static void Set(int* list, int i, int lo, int size, int sqre)
        {
            list[3 * i] = lo;
            list[3 * i + 1] = size;
            list[3 * i + 2] = sqre;
        }
    }

    /// <summary>
    /// 並列区間に渡す、固定済みの二重対角行列と特異ベクトルの領域
    /// </summary>
    private readonly unsafe struct NodeSystem(double* d, int dLength, double* e, int eLength, double* u, int uLength, int ldu, double* v, int vLength, int ldv)
    {
        internal Span<double> D => new(d, dLength);

        internal Span<double> E => new(e, eLength);

        internal Span<double> U => new(u, uLength);

        internal Span<double> V => new(v, vLength);

        internal int Ldu => ldu;

        internal int Ldv => ldv;
    }

    /// <summary>
    /// <paramref name="nodes"/> の i 番目の部分問題（lo, n, sqre）を解く作業
    /// </summary>
    private readonly unsafe struct SolveNodesWork(NodeSystem system, int* nodes, int* failed) : IParallelWork
    {
        public void Invoke(int i)
        {
            if (!SolveNode(nodes[3 * i], nodes[3 * i + 1], nodes[3 * i + 2], system.D, system.E, system.U, system.Ldu, system.V, system.Ldv))
            {
                Volatile.Write(ref *failed, 1);
            }
        }
    }

    /// <summary>
    /// <paramref name="nodes"/> の i 番目の部分問題を、解いた2つの子から統合する作業
    /// </summary>
    private readonly unsafe struct MergeNodesWork(NodeSystem system, int* nodes) : IParallelWork
    {
        public void Invoke(int i)
        {
            int lo = nodes[3 * i];
            int size = nodes[3 * i + 1];
            int sqre = nodes[3 * i + 2];
            int nl = size / 2;
            int k = lo + nl;
            var dd = system.D;
            Merge(lo, nl, size - nl - 1, sqre, dd[k], system.E[k], dd, system.U, system.Ldu, system.V, system.Ldv);
        }
    }

    // 特異値（昇順）を d[lo..] に、 左特異ベクトルを U の対角ブロック（n × n）、右特異ベクトルを V の対角ブロック（(n+sqre)²）に書く。
    // sqre = 1 のとき V の最後の列は零特異値に対応する（B の零空間）。
    /// <summary>
    /// 行 <c>[lo, lo+n)</c>、列 <c>[lo, lo+n+sqre)</c> の部分問題を解く
    /// </summary>
    private static bool SolveNode(int lo, int n, int sqre, Span<double> d, Span<double> e, Span<double> u, int ldu, Span<double> v, int ldv)
    {
        if (n <= LeafSize)
        {
            return SolveLeaf(lo, n, sqre, d, e, u, ldu, v, ldv);
        }

        int nl = n / 2;
        int k = lo + nl;
        int nr = n - nl - 1;
        double alpha = d[k];
        double beta = e[k];
        if (!SolveNode(lo, nl, 1, d, e, u, ldu, v, ldv) || !SolveNode(k + 1, nr, sqre, d, e, u, ldu, v, ldv))
        {
            return false;
        }

        Merge(lo, nl, nr, sqre, alpha, beta, d, u, ldu, v, ldv);
        return true;
    }

    // LAPACK dlasdq
    // 列が1つ多い場合は回転で正方の上二重対角に直す。
    /// <summary>
    /// 小さい部分問題を QR 法で解く
    /// </summary>
    private static bool SolveLeaf(int lo, int n, int sqre, Span<double> d, Span<double> e, Span<double> u, int ldu, Span<double> v, int ldv)
    {
        int m = n + sqre;
        var ub = u[(lo + lo * ldu)..];
        var vb = v[(lo + lo * ldv)..];
        for (int j = 0; j < n; j++)
        {
            ub.Slice(j * ldu, n).Clear();
            ub[j + j * ldu] = 1;
        }

        for (int j = 0; j < m; j++)
        {
            vb.Slice(j * ldv, m).Clear();
            vb[j + j * ldv] = 1;
        }

        Span<double> dd = stackalloc double[n];
        Span<double> ee = stackalloc double[n];
        d.Slice(lo, n).CopyTo(dd);
        ee.Clear();
        e.Slice(lo, n - 1 + sqre).CopyTo(ee);
        if (sqre == 1)
        {
            // 右からの回転で n × (n+1) の上二重対角を下二重対角（最後の列は0）にする
            for (int i = 0; i < n; i++)
            {
                BidiagonalSvd.Rotation(dd[i], ee[i], out double c, out double s, out double r);
                dd[i] = r;
                if (i < n - 1)
                {
                    ee[i] = s * dd[i + 1];
                    dd[i + 1] *= c;
                }
                else
                {
                    ee[i] = 0;
                }

                Level1Kernels.Rotate(vb.Slice(i * ldv, m), vb.Slice((i + 1) * ldv, m), c, s);
            }

            // 左からの回転で上二重対角に戻す
            for (int i = 0; i < n - 1; i++)
            {
                BidiagonalSvd.Rotation(dd[i], ee[i], out double c, out double s, out double r);
                dd[i] = r;
                ee[i] = s * dd[i + 1];
                dd[i + 1] *= c;
                Level1Kernels.Rotate(ub.Slice(i * ldu, n), ub.Slice((i + 1) * ldu, n), c, s);
            }

            ee[n - 1] = 0;
        }

        if (!BidiagonalSvd.Solve(n, dd, ee, ub, n, ldu, vb, m, ldv))
        {
            return false;
        }

        // 昇順にする（V の最後の列（零空間）はそのまま）
        for (int j = 0; j < n / 2; j++)
        {
            (dd[j], dd[n - 1 - j]) = (dd[n - 1 - j], dd[j]);
            Level1Kernels.Swap(ub.Slice(j * ldu, n), ub.Slice((n - 1 - j) * ldu, n));
            Level1Kernels.Swap(vb.Slice(j * ldv, m), vb.Slice((n - 1 - j) * ldv, m));
        }

        dd.CopyTo(d.Slice(lo, n));
        return true;
    }

    // LAPACK dlasd1–dlasd3
    /// <summary>
    /// 上下の部分問題の特異値分解と中央の行 <c>(α, β)</c> から、行 <c>[lo, lo+n)</c> の部分問題の分解を作る
    /// </summary>
    private static void Merge(int lo, int nl, int nr, int sqre, double alpha, double beta, Span<double> d, Span<double> u, int ldu, Span<double> v, int ldv)
    {
        int n = nl + 1 + nr;
        int m = n + sqre;
        int k = lo + nl;
        int vTopRows = nl + 1;
        int vBottomRows = m - vTopRows;

        // 基底の列（V の列番号で表す。子の特異値の列は U でも同じ番号）: 0 番目は上側の零空間の列（k）で、U では中央の行の単位ベクトル
        int[] rentedInt = WorkArrayPool<int>.Rent(checked(4 * n));
        int nwork = 8 * n;
        double[] rented = WorkArrayPool<double>.Rent(nwork);
        try
        {
            var cols = rentedInt.AsSpan(0, n);
            var type = rentedInt.AsSpan(n, n);
            var nondeflated = rentedInt.AsSpan(2 * n, n);
            var deflated = rentedInt.AsSpan(3 * n, n);
            var z = rented.AsSpan(0, n);
            var sv = rented.AsSpan(n, n);
            var ws = rented.AsSpan(2 * n, 6 * n);

            // z = (α·V₁ の最終行, β·V₂ の先頭行)。上下の零空間の列を回転で1つにまとめる
            double za = alpha * v[k + k * ldv];
            if (sqre == 1)
            {
                int nullCol = lo + n;
                double zb = beta * v[k + 1 + nullCol * ldv];
                double r = Householder.Hypot(za, zb);
                double c = r == 0 ? 1 : za / r;
                double s = r == 0 ? 0 : zb / r;
                Level1Kernels.Rotate(v.Slice(lo + k * ldv, m), v.Slice(lo + nullCol * ldv, m), c, s);
                za = r;
            }

            // 子の特異値の列を併合して昇順に並べる。型: 1 = 上側だけ、2 = 両側、3 = 下側だけ
            cols[0] = k;
            type[0] = sqre == 1 ? 2 : 1;
            z[0] = za;
            sv[0] = 0;
            {
                int a = lo, b = k + 1, t = 1;
                while (a < k || b <= k + nr)
                {
                    bool takeLeft = b > k + nr || (a < k && d[a] <= d[b]);
                    int c = takeLeft ? a++ : b++;
                    cols[t] = c;
                    type[t] = takeLeft ? 1 : 3;
                    sv[t] = d[c];
                    z[t] = takeLeft ? alpha * v[k + c * ldv] : beta * v[k + 1 + c * ldv];
                    t++;
                }
            }

            // 減次
            double dmax = 0;
            for (int t = 1; t < n; t++)
            {
                dmax = Math.Max(dmax, sv[t]);
            }

            double tol = 8 * Epsilon * Math.Max(dmax, Math.Max(Math.Abs(alpha), Math.Abs(beta)));
            if (Math.Abs(z[0]) <= tol)
            {
                z[0] = tol;
            }

            int nk = 0;
            int nd = 0;
            nondeflated[nk++] = 0;
            int prev = -1;
            for (int t = 1; t < n; t++)
            {
                if (Math.Abs(z[t]) <= tol)
                {
                    deflated[nd++] = t;
                    continue;
                }

                if (prev < 0)
                {
                    prev = t;
                    continue;
                }

                if (Math.Abs(sv[t] - sv[prev]) <= tol)
                {
                    // 近接した特異値の組: 回転で z[prev] を消して prev を減次する
                    double s = z[prev];
                    double c = z[t];
                    double tau = Householder.Hypot(c, s);
                    c /= tau;
                    s = -s / tau;
                    z[t] = tau;
                    z[prev] = 0;
                    Level1Kernels.Rotate(u.Slice(lo + cols[prev] * ldu, n), u.Slice(lo + cols[t] * ldu, n), c, s);
                    Level1Kernels.Rotate(v.Slice(lo + cols[prev] * ldv, m), v.Slice(lo + cols[t] * ldv, m), c, s);
                    if (type[t] != type[prev])
                    {
                        type[t] = 2;
                        type[prev] = 2;
                    }

                    deflated[nd++] = prev;
                    prev = t;
                }
                else
                {
                    nondeflated[nk++] = prev;
                    prev = t;
                }
            }

            if (prev >= 0)
            {
                nondeflated[nk++] = prev;
            }

            // 零に近すぎる特異値は永年方程式の極が重ならないよう tol/2 まで離す
            if (nk >= 2 && sv[nondeflated[1]] <= 0.5 * tol)
            {
                sv[nondeflated[1]] = 0.5 * tol;
            }

            SolveSecular(lo, n, m, k, nk, nd, vTopRows, vBottomRows, cols, type, nondeflated, deflated, z, sv, d, u, ldu, v, ldv, ws);
        }
        finally
        {
            WorkArrayPool<double>.Return(rented);
            WorkArrayPool<int>.Return(rentedInt);
        }
    }

    // LAPACK dlasd3
    /// <summary>
    /// 永年方程式を解き、特異ベクトルを更新して昇順に並べ直す
    /// </summary>
    private static void SolveSecular(
        int lo, int n, int m, int k, int nk, int nd, int vTopRows, int vBottomRows,
        ReadOnlySpan<int> cols, ReadOnlySpan<int> type, ReadOnlySpan<int> nondeflated, ReadOnlySpan<int> deflated,
        ReadOnlySpan<double> z, ReadOnlySpan<double> sv, Span<double> d, Span<double> u, int ldu, Span<double> v, int ldv, Span<double> ws)
    {
        var roots = ws[..nk];
        var squares = ws.Slice(n, nk);
        var zs = ws.Slice(2 * n, nk);
        var zhat = ws.Slice(3 * n, nk);
        var sigma = ws.Slice(4 * n, nk);
        var shifted = ws.Slice(5 * n, nk);
        for (int i = 0; i < nk; i++)
        {
            int t = nondeflated[i];
            roots[i] = sv[t];
            squares[i] = sv[t] * sv[t];
            zs[i] = z[t];
        }

        // delta[j, i] = dⱼ² - σᵢ²、σᵢ² は列 i の根
        // 末尾の nk + nd 個は並べ替えの鍵に使う
        int bigSize = 2 * nk * nk + (n + m) * nk + (n + m) * nd + nk + nd;
        double[] bigBuffer = WorkArrayPool<double>.Rent(bigSize);
        int[] selection = WorkArrayPool<int>.Rent(checked(3 * nk + nd));
        try
        {
            var big = bigBuffer.AsSpan(0, bigSize);
            var delta = big[..(nk * nk)];
            var um = big.Slice(nk * nk, nk * nk);
            var newU = big.Slice(2 * nk * nk, n * nk);
            var newV = big.Slice(2 * nk * nk + n * nk, m * nk);
            var oldU = big.Slice(2 * nk * nk + (n + m) * nk, n * nd);
            var oldV = big.Slice(2 * nk * nk + (n + m) * nk + n * nd, m * nd);

            SolveSecularParallel(nk, n, ws, bigBuffer);

            // 新しい U = [U の基底]·U_M。上側の行は型1・2、下側の行は型2・3の列だけが寄与し、中央の行 k は u₀
            var top = selection.AsSpan(0, nk);
            var bottom = selection.AsSpan(nk, nk);
            int topCount = Select(1, nk, type, nondeflated, top, bottom, out int bottomCount);
            newU.Clear();
            MultiplySelected(nk, rows: k - lo, rowStart: lo, top[..topCount], cols, nondeflated, u, ldu, um, newU, n, rowOffset: 0);
            MultiplySelected(nk, rows: n - (k - lo) - 1, rowStart: k + 1, bottom[..bottomCount], cols, nondeflated, u, ldu, um, newU, n, rowOffset: k - lo + 1);
            for (int i = 0; i < nk; i++)
            {
                newU[(k - lo) + i * n] = um[i * nk];
            }

            // 新しい V = [V の基底]·V_M。基底の0番（零空間の列）は型 type[0]
            topCount = Select(0, nk, type, nondeflated, top, bottom, out bottomCount);
            newV.Clear();
            MultiplySelected(nk, rows: vTopRows, rowStart: lo, top[..topCount], cols, nondeflated, v, ldv, delta, newV, m, rowOffset: 0);
            MultiplySelected(nk, rows: vBottomRows, rowStart: k + 1, bottom[..bottomCount], cols, nondeflated, v, ldv, delta, newV, m, rowOffset: vTopRows);

            // 減次した列を退避する
            for (int i = 0; i < nd; i++)
            {
                int c = cols[deflated[i]];
                u.Slice(lo + c * ldu, n).CopyTo(oldU.Slice(i * n, n));
                v.Slice(lo + c * ldv, m).CopyTo(oldV.Slice(i * m, m));
            }

            // 全体を昇順に並べて書き戻す（V の列 lo+n（sqre = 1 の零空間）はそのまま）
            int total = nk + nd;
            var values = big.Slice(bigSize - total, total);
            for (int i = 0; i < nk; i++)
            {
                values[i] = sigma[i];
            }

            for (int i = 0; i < nd; i++)
            {
                values[nk + i] = sv[deflated[i]];
            }

            var order = selection.AsSpan(2 * nk, total);
            for (int i = 0; i < total; i++)
            {
                order[i] = i;
            }

            values.Sort(order);
            for (int t = 0; t < total; t++)
            {
                int src = order[t];
                int dst = lo + t;
                d[dst] = values[t];
                if (src < nk)
                {
                    newU.Slice(src * n, n).CopyTo(u.Slice(lo + dst * ldu, n));
                    newV.Slice(src * m, m).CopyTo(v.Slice(lo + dst * ldv, m));
                }
                else
                {
                    oldU.Slice((src - nk) * n, n).CopyTo(u.Slice(lo + dst * ldu, n));
                    oldV.Slice((src - nk) * m, m).CopyTo(v.Slice(lo + dst * ldv, m));
                }
            }
        }
        finally
        {
            WorkArrayPool<double>.Return(bigBuffer);
            WorkArrayPool<int>.Return(selection);
        }
    }

    // top は型1・2（上側の行が寄与する）、bottom は型2・3 の列を、first 番目以降の非減次の列から選ぶ。
    /// <summary>
    /// 統合で上側・下側の行に寄与する M の行番号を選ぶ
    /// </summary>
    /// <returns>
    /// <paramref name="top"/> に選んだ個数。
    /// </returns>
    private static int Select(
        int first, int nk, ReadOnlySpan<int> type, ReadOnlySpan<int> nondeflated, Span<int> top, Span<int> bottom, out int bottomCount)
    {
        int topCount = 0;
        bottomCount = 0;
        for (int j = first; j < nk; j++)
        {
            int t = nondeflated[j];
            if (type[t] != 3)
            {
                top[topCount++] = j;
            }

            if (type[t] != 1)
            {
                bottom[bottomCount++] = j;
            }
        }

        return topCount;
    }

    // 各根・各成分・各列の計算は独立なので、範囲ごとに別のスレッドで行う（作業領域は配列の位置で受け渡す）。
    /// <summary>
    /// 永年方程式の根（<c>sigma</c>）、Gu–Eisenstat の <c>ẑ</c>、M の特異ベクトル（<c>delta</c> に V_M、<c>um</c> に U_M）を求める
    /// </summary>
    /// <param name="nk">デフレーション後の次数。</param>
    /// <param name="n">行列の次数。</param>
    /// <param name="ws">作業配列（<c>roots, squares, zs, zhat, sigma, shifted</c> が長さ n ずつ並ぶ）。</param>
    /// <param name="big">先頭に <c>delta</c>（<c>nk × nk</c>）、続けて <c>um</c>（<c>nk × nk</c>）。</param>
    private static unsafe void SolveSecularParallel(int nk, int n, Span<double> ws, double[] big)
    {
        double[] w = WorkArrayPool<double>.Rent(checked(5 * nk));
        ws.Slice(0, nk).CopyTo(w.AsSpan(0, nk));
        ws.Slice(n, nk).CopyTo(w.AsSpan(nk, nk));
        ws.Slice(2 * n, nk).CopyTo(w.AsSpan(2 * nk, nk));
        try
        {
            // w = [roots, squares, zs, zhat, sigma]
            int threads = Parallelism.Degree(30.0 * nk * nk, Parallelism.Grain);
            int chunks = threads > 1 ? Math.Min(nk, 4 * threads) : 1;
            int chunk = (nk + chunks - 1) / chunks;
            fixed (double* pw = w)
            fixed (double* pb = big)
            {
                var roots = new SingularRootsWork(nk, chunk, pw, pb);
                Parallelism.For(chunks, threads, ref roots);

                // Gu–Eisenstat: 計算した根に対して厳密に成り立つ z を作り直す（成分 j の範囲ごと）
                var recompute = new RecomputeZWork(nk, chunk, pw, pb);
                Parallelism.For(chunks, threads, ref recompute);

                // M の特異ベクトル: vⱼ = ẑⱼ/(dⱼ² - σ²)、u₀ = -1, uⱼ = dⱼẑⱼ/(dⱼ² - σ²)（列 i、正規化）
                var vectors = new SingularVectorsWork(nk, chunk, pw, pb);
                Parallelism.For(chunks, threads, ref vectors);
            }

            w.AsSpan(3 * nk, nk).CopyTo(ws.Slice(3 * n, nk));
            w.AsSpan(4 * nk, nk).CopyTo(ws.Slice(4 * n, nk));
        }
        finally
        {
            WorkArrayPool<double>.Return(w);
        }
    }

    // w = [roots, squares, zs, zhat, sigma]（長さ nk ずつ）、big の先頭に delta、続けて um（nk × nk ずつ）。
    /// <summary>
    /// 範囲 <c>c</c> の根 <c>σᵢ</c> と差 <c>dⱼ² - σᵢ²</c> を求める作業
    /// </summary>
    private readonly unsafe struct SingularRootsWork(int nk, int chunk, double* w, double* big) : IParallelWork
    {
        public void Invoke(int c)
        {
            int i0 = c * chunk;
            int i1 = Math.Min(nk, i0 + chunk);
            double[] shifted = WorkArrayPool<double>.Rent(nk);
            var roots = new ReadOnlySpan<double>(w, nk);
            var squares = new ReadOnlySpan<double>(w + nk, nk);
            var zs = new ReadOnlySpan<double>(w + 2 * nk, nk);
            for (int i = i0; i < i1; i++)
            {
                double lambda = TridiagonalDivideConquer.SecularRoot(nk, i, squares, zs, 1.0, new Span<double>(big + i * nk, nk), shifted.AsSpan(0, nk), roots);
                w[4 * nk + i] = Math.Sqrt(Math.Max(lambda, 0));
            }

            WorkArrayPool<double>.Return(shifted);
        }
    }

    /// <summary>
    /// 範囲 <c>c</c> の成分の ẑ を作り直す作業
    /// </summary>
    private readonly unsafe struct RecomputeZWork(int nk, int chunk, double* w, double* big) : IParallelWork
    {
        public void Invoke(int c)
        {
            int j0 = c * chunk;
            int j1 = Math.Min(nk, j0 + chunk);
            var roots = new ReadOnlySpan<double>(w, nk);
            var zs = new ReadOnlySpan<double>(w + 2 * nk, nk);
            var zhat = new Span<double>(w + 3 * nk, nk);
            var delta = new ReadOnlySpan<double>(big, nk * nk);
            for (int j = j0; j < j1; j++)
            {
                zhat[j] = delta[j + j * nk];
            }

            for (int i = 0; i < nk; i++)
            {
                for (int j = j0; j < j1; j++)
                {
                    if (i != j)
                    {
                        zhat[j] *= delta[j + i * nk] / ((roots[j] - roots[i]) * (roots[j] + roots[i]));
                    }
                }
            }

            for (int j = j0; j < j1; j++)
            {
                zhat[j] = Math.CopySign(Math.Sqrt(Math.Max(-zhat[j], 0)), zs[j]);
            }
        }
    }

    /// <summary>
    /// 範囲 <c>c</c> の列の M の特異ベクトルを求める作業（delta を V_M で上書きし、um に U_M を書く）
    /// </summary>
    private readonly unsafe struct SingularVectorsWork(int nk, int chunk, double* w, double* big) : IParallelWork
    {
        public void Invoke(int c)
        {
            int i0 = c * chunk;
            int i1 = Math.Min(nk, i0 + chunk);
            var roots = new ReadOnlySpan<double>(w, nk);
            var zhat = new ReadOnlySpan<double>(w + 3 * nk, nk);
            for (int i = i0; i < i1; i++)
            {
                var vc = new Span<double>(big + i * nk, nk);
                var uc = new Span<double>(big + nk * nk + i * nk, nk);
                for (int j = 0; j < nk; j++)
                {
                    double q = zhat[j] / vc[j];
                    vc[j] = q;
                    uc[j] = roots[j] * q;
                }

                uc[0] = -1;
                Level1Kernels.Scale(1.0 / Level1Kernels.Norm2(vc), vc);
                Level1Kernels.Scale(1.0 / Level1Kernels.Norm2(uc), uc);
            }
        }
    }

    // mvec、nk × nk。
    // selected は M の行番号。
    /// <summary>
    /// 行 <c>[rowStart, rowStart+rows)</c> について、選んだ基底の列と M の特異ベクトルの積を <paramref name="target"/> の行 <c>rowOffset..</c> に書く
    /// </summary>
    private static void MultiplySelected(
        int nk, int rows, int rowStart, ReadOnlySpan<int> selected, ReadOnlySpan<int> cols, ReadOnlySpan<int> nondeflated,
        ReadOnlySpan<double> basis, int ldb, ReadOnlySpan<double> mvec, Span<double> target, int ldt, int rowOffset)
    {
        int count = selected.Length;
        if (count == 0 || rows == 0)
        {
            return;
        }

        double[] rented = WorkArrayPool<double>.Rent(checked(rows * count + count * nk));
        try
        {
            var a = rented.AsSpan(0, rows * count);
            var b = rented.AsSpan(rows * count, count * nk);
            for (int c = 0; c < count; c++)
            {
                basis.Slice(rowStart + cols[nondeflated[selected[c]]] * ldb, rows).CopyTo(a.Slice(c * rows, rows));
            }

            // 選んだ行を列ごとに集める（読み書きとも列の中で連続に近い順にする）
            var sel = selected;
            for (int i = 0; i < nk; i++)
            {
                var src = mvec.Slice(i * nk, nk);
                var dst = b.Slice(i * count, count);
                for (int c = 0; c < count; c++)
                {
                    dst[c] = src[sel[c]];
                }
            }

            Gemm.Multiply(false, false, rows, nk, count, 1.0, a, rows, b, count, target[rowOffset..], ldt);
        }
        finally
        {
            WorkArrayPool<double>.Return(rented);
        }
    }
}
