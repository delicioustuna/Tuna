namespace Tuna.LinearAlgebra;

internal static partial class HessenbergQr
{
    // LAPACK dlaqr3
    // 未収束部分 [l, n] の右下 nw × nw の窓をシューア分解し、 窓の外との結合（スパイク）が無視できる固有値を収縮させる。
    // 収縮できない固有値は窓の上側へ並べ替える。
    /// <summary>
    /// 積極的早期収縮
    /// </summary>
    /// <param name="nn">行列全体の次数。</param>
    /// <param name="low">固有ベクトルを更新する行の先頭。</param>
    /// <param name="high">固有ベクトルを更新する行の末尾。</param>
    /// <param name="l">処理範囲の先頭。</param>
    /// <param name="n">行列の次数。</param>
    /// <param name="nw">デフレーション窓の大きさ。</param>
    /// <param name="h">ヘッセンベルグ行列。</param>
    /// <param name="wr">固有値の実部の出力先。</param>
    /// <param name="wi">固有値の虚部の出力先。</param>
    /// <param name="z">固有ベクトルの出力先。</param>
    /// <param name="vectors">固有ベクトルも更新する場合は true。</param>
    /// <param name="exshift">累積したシフト。</param>
    /// <param name="shifts">収縮しなかった窓の固有値（前半に実部、後半に虚部。それぞれ先頭 <paramref name="undeflated"/> 個）。</param>
    /// <param name="undeflated">デフレーションされずに残った固有値の数。</param>
    /// <returns>
    /// 収縮した固有値の数（窓の下端から）。
    /// </returns>
    private static int AggressiveEarlyDeflation(
        int nn, int low, int high, int l, int n, int nw, Span<double> h, Span<double> wr, Span<double> wi, Span<double> z, bool vectors,
        double exshift, Span<double> shifts, out int undeflated)
    {
        int ld = nn;
        int cap = shifts.Length / 2;
        int kwtop = n - nw + 1;
        double s = kwtop > l ? h[kwtop + (kwtop - 1) * ld] : 0.0;
        undeflated = 0;

        int size = nw * nw;
        double[] buffer = WorkArrayPool<double>.Rent(checked(2 * size + 4 * nw));
        try
        {
            var t = buffer.AsSpan(0, size);
            var u = buffer.AsSpan(size, size);
            var twr = buffer.AsSpan(2 * size, nw);
            var twi = buffer.AsSpan(2 * size + nw, nw);
            var work = buffer.AsSpan(2 * size + 2 * nw, 2 * nw);
            t.Clear();
            u.Clear();
            for (int j = 0; j < nw; j++)
            {
                int rows = Math.Min(j + 2, nw);
                h.Slice(kwtop + (kwtop + j) * ld, rows).CopyTo(t.Slice(j * nw, rows));
                u[j + j * nw] = 1;
            }

            // 窓のシューア分解 T = UᵀWU（窓は高々数百次なので逐次に実行する。並列区間の起動の方が高くつく）
            bool converged;
            using (Parallelism.Sequential())
            {
                converged = Solve(nw, 0, nw - 1, t, twr, twi, u, true, backSubstitute: false);
            }

            if (!converged)
            {
                return 0;
            }

            for (int j = 0; j < nw - 2; j++)
            {
                t.Slice(j * nw + j + 2, nw - j - 2).Clear();
            }

            // 下端から収縮を判定する。収縮できないブロックは窓の上側（ilst）へ移す
            double smlnum = 2.2250738585072014e-308 * ((high - low + 1) / Epsilon);
            int ns = nw;
            int ilst = 0;
            while (ilst < ns)
            {
                bool pair = ns >= 2 && t[ns - 1 + (ns - 2) * nw] != 0;
                if (!pair)
                {
                    double foo = Math.Abs(t[ns - 1 + (ns - 1) * nw]);
                    if (foo == 0)
                    {
                        foo = Math.Abs(s);
                    }

                    if (Math.Abs(s * u[(ns - 1) * nw]) <= Math.Max(smlnum, Epsilon * foo))
                    {
                        ns--;
                    }
                    else
                    {
                        if (!SchurBlocks.MoveUp(nw, t, nw, u, nw, 0, nw, ns - 1, ilst))
                        {
                            break;
                        }

                        ilst++;
                    }
                }
                else
                {
                    double foo = Math.Abs(t[ns - 1 + (ns - 1) * nw]) +
                        Math.Sqrt(Math.Abs(t[ns - 1 + (ns - 2) * nw])) * Math.Sqrt(Math.Abs(t[ns - 2 + (ns - 1) * nw]));
                    if (foo == 0)
                    {
                        foo = Math.Abs(s);
                    }

                    if (Math.Max(Math.Abs(s * u[(ns - 1) * nw]), Math.Abs(s * u[(ns - 2) * nw])) <= Math.Max(smlnum, Epsilon * foo))
                    {
                        ns -= 2;
                    }
                    else
                    {
                        if (!SchurBlocks.MoveUp(nw, t, nw, u, nw, 0, nw, ns - 2, ilst))
                        {
                            break;
                        }

                        ilst += 2;
                    }
                }
            }

            if (ns == 0)
            {
                s = 0;
            }

            // 収縮しなかった固有値（次の掃引のシフト候補）
            for (int i = 0; i < ns;)
            {
                if (i + 1 < ns && t[i + 1 + i * nw] != 0)
                {
                    double a = t[i + i * nw], b = t[i + (i + 1) * nw], c = t[i + 1 + i * nw], d = t[i + 1 + (i + 1) * nw];
                    double pp = 0.5 * (a - d);
                    double qq = pp * pp + b * c;
                    double rq = Math.Sqrt(Math.Abs(qq));
                    if (qq >= 0)
                    {
                        double zr = pp >= 0 ? pp + rq : pp - rq;
                        shifts[i] = d + zr;
                        shifts[i + 1] = zr != 0 ? d - b * c / zr : d + zr;
                        shifts[cap + i] = 0;
                        shifts[cap + i + 1] = 0;
                    }
                    else
                    {
                        shifts[i] = d + pp;
                        shifts[i + 1] = d + pp;
                        shifts[cap + i] = rq;
                        shifts[cap + i + 1] = -rq;
                    }

                    i += 2;
                }
                else
                {
                    shifts[i] = t[i + i * nw];
                    shifts[cap + i] = 0;
                    i++;
                }
            }

            undeflated = ns;
            int nd = nw - ns;
            if (nd == 0 && s != 0)
            {
                return 0;
            }

            if (ns > 1 && s != 0)
            {
                // スパイク s·U[0, 0:ns] を反射で第1成分に集め、窓の未収縮部分をヘッセンベルグ形に戻す
                var v = work[..ns];
                for (int j = 0; j < ns; j++)
                {
                    v[j] = u[j * nw];
                }

                double beta = v[0];
                double tau = Householder.Generate(ref beta, v[1..]);
                v[0] = 1;
                var tmp = work[nw..];
                Householder.ApplyLeft(v, tau, ns, nw, t, nw);
                Householder.ApplyRight(v, tau, ns, ns, t, nw, tmp);
                Householder.ApplyRight(v, tau, nw, ns, u, nw, tmp);

                var tauH = buffer.AsSpan(2 * size + 2 * nw, nw);
                HessenbergReduction.Reduce(nw, 0, ns - 1, t, nw, tauH);
                var hv = buffer.AsSpan(2 * size + 3 * nw, nw);
                for (int i = 0; i < ns - 2; i++)
                {
                    int len = ns - 1 - i;
                    hv[0] = 1;
                    t.Slice(i + 2 + i * nw, len - 1).CopyTo(hv[1..]);
                    Householder.ApplyRight(hv[..len], tauH[i], nw, len, u[((i + 1) * nw)..], nw, twr);
                }

                for (int j = 0; j < nw - 2; j++)
                {
                    t.Slice(j * nw + j + 2, nw - j - 2).Clear();
                }
            }

            // 窓を書き戻し、窓の外の部分と Z を U で更新する
            if (kwtop > l)
            {
                h[kwtop + (kwtop - 1) * ld] = s * u[0];
            }

            for (int j = 0; j < nw; j++)
            {
                int rows = Math.Min(j + 2, nw);
                t.Slice(j * nw, rows).CopyTo(h.Slice(kwtop + (kwtop + j) * ld, rows));
            }

            int rowTop = vectors ? 0 : l;
            int above = kwtop - rowTop;
            int right = vectors ? nn - n - 1 : 0;
            int zRows = vectors ? high - low + 1 : 0;
            ApplyWindow(h, ld, z, u, nw, kwtop, n, rowTop, above, right, low, zRows);

            // 収縮した固有値を確定する（2×2 ブロックは標準形にする）
            int rowEnd = vectors ? nn : n + 1;
            for (int i = kwtop + ns; i <= n;)
            {
                if (i < n && h[i + 1 + i * ld] != 0)
                {
                    SchurBlocks.StandardizeBlock(
                        i, h, ld, rowTop, rowEnd, z, ld, low, zRows, out double r1, out double i1, out double r2, out double i2);
                    h[i + i * ld] += exshift;
                    h[i + 1 + (i + 1) * ld] += exshift;
                    wr[i] = r1 + exshift;
                    wi[i] = i1;
                    wr[i + 1] = r2 + exshift;
                    wi[i + 1] = i2;
                    i += 2;
                }
                else
                {
                    h[i + i * ld] += exshift;
                    wr[i] = h[i + i * ld];
                    wi[i] = 0;
                    i++;
                }
            }

            return nd;
        }
        finally
        {
            WorkArrayPool<double>.Return(buffer);
        }
    }

    /// <summary>
    /// AED の窓の外の更新で、1つの作業に割り当てる行（列）の数
    /// </summary>
    private const int WindowUpdateChunk = 96;

    // 3つの行列積は互いに独立なので、行（右の列は列）を区切った作業にまとめて1つの並列区間で実行する （行列積ごとに並列化すると、窓のシューア分解の間に休止した作業スレッドを3回起こすことになる）。
    // 窓の直交行列 U を窓の外へ適用する: 窓の上の行 H[rowTop:kwtop, kwtop:n] ← H·U、
    // 右の列 H[kwtop:n, n+1:] ← Uᵀ·H、Z[low:high, kwtop:n] ← Z·U。
    // 3つの行列積は互いに独立なので、行（右の列は列）を区切った作業にまとめて1つの並列区間で実行する
    // （行列積ごとに並列化すると、窓のシューア分解の間に休止した作業スレッドを3回起こすことになる）
    /// <summary>
    /// 窓の直交変換を、窓の外の行列と固有ベクトルに適用する
    /// </summary>
    private static unsafe void ApplyWindow(
        Span<double> h, int ld, Span<double> z, ReadOnlySpan<double> u, int nw, int kwtop, int n, int rowTop, int above, int right, int low, int zRows)
    {
        int aboveTasks = (above + WindowUpdateChunk - 1) / WindowUpdateChunk;
        int rightTasks = (right + WindowUpdateChunk - 1) / WindowUpdateChunk;
        int zTasks = (zRows + WindowUpdateChunk - 1) / WindowUpdateChunk;
        int tasks = aboveTasks + rightTasks + zTasks;
        if (tasks == 0)
        {
            return;
        }

        int threads = Parallelism.PlannedDegree((double)(above + right + zRows) * nw * nw, 4 * Parallelism.Grain);
        double[] tmpBuffer = WorkArrayPool<double>.Rent(checked(threads * WindowUpdateChunk * nw));
        fixed (double* ph = h)
        fixed (double* pz = z)
        fixed (double* pu = u)
        fixed (double* pt = tmpBuffer)
        {
            double* hh = ph, zz = pz, uu = pu, tt = pt;
            int hLength = h.Length, zLength = z.Length, uLength = nw * nw;
            void Task(int task, int slot)
            {
                var uSpan = new ReadOnlySpan<double>(uu, uLength);
                var tmp = new Span<double>(tt + ((nint)slot * WindowUpdateChunk * nw), WindowUpdateChunk * nw);
                if (task < aboveTasks)
                {
                    int r0 = rowTop + (task * WindowUpdateChunk);
                    int rows = Math.Min(WindowUpdateChunk, above - (task * WindowUpdateChunk));
                    var hs = new Span<double>(hh, hLength)[(r0 + kwtop * ld)..];
                    tmp[..(rows * nw)].Clear();
                    Gemm.Multiply(false, false, rows, nw, nw, 1.0, hs, ld, uSpan, nw, tmp, rows);
                    Level1Kernels.Copy(rows, nw, tmp, rows, hs, ld);
                }
                else if (task < aboveTasks + rightTasks)
                {
                    int c0 = n + 1 + ((task - aboveTasks) * WindowUpdateChunk);
                    int cols = Math.Min(WindowUpdateChunk, right - ((task - aboveTasks) * WindowUpdateChunk));
                    var hs = new Span<double>(hh, hLength)[(kwtop + c0 * ld)..];
                    tmp[..(nw * cols)].Clear();
                    Gemm.Multiply(true, false, nw, cols, nw, 1.0, uSpan, nw, hs, ld, tmp, nw);
                    Level1Kernels.Copy(nw, cols, tmp, nw, hs, ld);
                }
                else
                {
                    int r0 = low + ((task - aboveTasks - rightTasks) * WindowUpdateChunk);
                    int rows = Math.Min(WindowUpdateChunk, zRows - ((task - aboveTasks - rightTasks) * WindowUpdateChunk));
                    var zs = new Span<double>(zz, zLength)[(r0 + kwtop * ld)..];
                    tmp[..(rows * nw)].Clear();
                    Gemm.Multiply(false, false, rows, nw, nw, 1.0, zs, ld, uSpan, nw, tmp, rows);
                    Level1Kernels.Copy(rows, nw, tmp, rows, zs, ld);
                }
            }

            if (threads <= 1)
            {
                for (int task = 0; task < tasks; task++)
                {
                    Task(task, 0);
                }
            }
            else
            {
                // 作業 t をスレッド t mod threads が順に受け持つ（一時領域はスレッドごと）
                Parallelism.ForStatic(threads, t =>
                {
                    using (Parallelism.Sequential())
                    {
                        for (int task = t; task < tasks; task += threads)
                        {
                            Task(task, t);
                        }
                    }
                });
            }
        }

        WorkArrayPool<double>.Return(tmpBuffer);
    }
}
