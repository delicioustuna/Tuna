namespace Tuna.LinearAlgebra;

// 実シューア形 T（列優先、n × n）と直交行列 Q を同時に更新する。
// Q の更新は行 [qRow, qRow + qRows) だけに行う。
// LAPACK dlanv2 / dlasy2 / dlaexc 相当
/// <summary>
/// 実シューア形の対角ブロック（1×1 と 2×2）の操作
/// </summary>
internal static class SchurBlocks
{
    // LAPACK dlanv2
    // 固有値が実数なら上三角（c = 0）、複素数なら対角成分が等しく bc < 0 の形にする。
    // [a b; c d] ← [cs -sn; sn cs]ᵀ [a b; c d] [cs -sn; sn cs]。
    /// <summary>
    /// 2×2 の実行列 <c>[a b; c d]</c> を標準形に変換する回転 <c>(cs, sn)</c> を求める
    /// </summary>
    internal static void Standardize(
        ref double a, ref double b, ref double c, ref double d,
        out double rt1r, out double rt1i, out double rt2r, out double rt2i, out double cs, out double sn)
    {
        const double Multiplier = 4;
        double safmn2 = Math.ScaleB(1.0, (int)(Math.Log2(NumericConstants.MinNormal / NumericConstants.MachineEpsilon) / 2));
        double safmx2 = 1 / safmn2;
        if (c == 0)
        {
            cs = 1;
            sn = 0;
        }
        else if (b == 0)
        {
            // 行と列を入れ替える
            cs = 0;
            sn = 1;
            (a, d) = (d, a);
            b = -c;
            c = 0;
        }
        else if (a - d == 0 && Math.Sign(b) != Math.Sign(c))
        {
            cs = 1;
            sn = 0;
        }
        else
        {
            double temp = a - d;
            double p = 0.5 * temp;
            double bcmax = Math.Max(Math.Abs(b), Math.Abs(c));
            double bcmis = Math.Min(Math.Abs(b), Math.Abs(c)) * Math.CopySign(1.0, b) * Math.CopySign(1.0, c);
            double scale = Math.Max(Math.Abs(p), bcmax);
            double z = p / scale * p + bcmax / scale * bcmis;
            if (z >= Multiplier * NumericConstants.MachineEpsilon)
            {
                // 実固有値
                z = p + Math.CopySign(Math.Sqrt(scale) * Math.Sqrt(z), p);
                a = d + z;
                d -= bcmax / z * bcmis;
                double tau = Householder.Hypot(c, z);
                cs = z / tau;
                sn = c / tau;
                b -= c;
                c = 0;
            }
            else
            {
                // 複素固有値、またはほぼ等しい実固有値。対角成分を等しくする
                int count = 0;
                double sigma = b + c;
                while (true)
                {
                    count++;
                    scale = Math.Max(Math.Abs(temp), Math.Abs(sigma));
                    if (scale >= safmx2)
                    {
                        sigma *= safmn2;
                        temp *= safmn2;
                        if (count <= 20)
                        {
                            continue;
                        }
                    }

                    if (scale <= safmn2)
                    {
                        sigma *= safmx2;
                        temp *= safmx2;
                        if (count <= 20)
                        {
                            continue;
                        }
                    }

                    break;
                }

                p = 0.5 * temp;
                double tau = Householder.Hypot(sigma, temp);
                cs = Math.Sqrt(0.5 * (1 + Math.Abs(sigma) / tau));
                sn = -(p / (tau * cs)) * Math.CopySign(1.0, sigma);

                double aa = a * cs + b * sn;
                double bb = -a * sn + b * cs;
                double cc = c * cs + d * sn;
                double dd = -c * sn + d * cs;
                a = aa * cs + cc * sn;
                b = bb * cs + dd * sn;
                c = -aa * sn + cc * cs;
                d = -bb * sn + dd * cs;
                temp = 0.5 * (a + d);
                a = temp;
                d = temp;
                if (c != 0)
                {
                    if (b != 0)
                    {
                        if (Math.Sign(b) == Math.Sign(c))
                        {
                            // 実固有値: 上三角にする
                            double sab = Math.Sqrt(Math.Abs(b));
                            double sac = Math.Sqrt(Math.Abs(c));
                            p = Math.CopySign(sab * sac, c);
                            tau = 1 / Math.Sqrt(Math.Abs(b + c));
                            a = temp + p;
                            d = temp - p;
                            b -= c;
                            c = 0;
                            double cs1 = sab * tau;
                            double sn1 = sac * tau;
                            temp = cs * cs1 - sn * sn1;
                            sn = cs * sn1 + sn * cs1;
                            cs = temp;
                        }
                    }
                    else
                    {
                        b = -c;
                        c = 0;
                        temp = cs;
                        cs = -sn;
                        sn = temp;
                    }
                }
            }
        }

        rt1r = a;
        rt2r = d;
        if (c == 0)
        {
            rt1i = 0;
            rt2i = 0;
        }
        else
        {
            rt1i = Math.Sqrt(Math.Abs(b)) * Math.Sqrt(Math.Abs(c));
            rt2i = -rt1i;
        }
    }

    // 行の更新は列 [j+2, colEnd)、列の更新は行 [rowStart, j) に行う。
    /// <summary>
    /// <c>T</c> の行・列 <c>j, j+1</c> の 2×2 ブロックを標準形にし、<c>T</c> の残りと <c>Q</c> に同じ回転を掛ける
    /// </summary>
    internal static void StandardizeBlock(
        int j, Span<double> t, int ldt, int rowStart, int colEnd, Span<double> q, int ldq, int qRow, int qRows,
        out double rt1r, out double rt1i, out double rt2r, out double rt2i)
    {
        int j2 = j + 1;
        double a = t[j + j * ldt], b = t[j + j2 * ldt], c = t[j2 + j * ldt], d = t[j2 + j2 * ldt];
        Standardize(ref a, ref b, ref c, ref d, out rt1r, out rt1i, out rt2r, out rt2i, out double cs, out double sn);
        t[j + j * ldt] = a;
        t[j + j2 * ldt] = b;
        t[j2 + j * ldt] = c;
        t[j2 + j2 * ldt] = d;
        for (int k = j + 2; k < colEnd; k++)
        {
            double x = t[j + k * ldt], y = t[j2 + k * ldt];
            t[j + k * ldt] = cs * x + sn * y;
            t[j2 + k * ldt] = cs * y - sn * x;
        }

        if (j > rowStart)
        {
            Level1Kernels.Rotate(t.Slice(rowStart + j * ldt, j - rowStart), t.Slice(rowStart + j2 * ldt, j - rowStart), cs, sn);
        }

        if (qRows > 0)
        {
            Level1Kernels.Rotate(q.Slice(qRow + j * ldq, qRows), q.Slice(qRow + j2 * ldq, qRows), cs, sn);
        }
    }

    // LAPACK dlaexc
    // T は全体（n × n）を更新する。
    /// <summary>
    /// 隣接する対角ブロック <c>T₁₁</c>（<c>n1 × n1</c>、先頭 <paramref name="j1"/>）と <c>T₂₂</c>（<c>n2 × n2</c>）を 直交相似変換で入れ替える
    /// </summary>
    /// <returns>
    /// 入れ替えたら真。条件が悪く入れ替えると誤差が大きくなる場合は何もせず偽。
    /// </returns>
    internal static bool Swap(int n, Span<double> t, int ldt, Span<double> q, int ldq, int qRow, int qRows, int j1, int n1, int n2)
    {
        if (n1 == 0 || n2 == 0 || j1 + n1 >= n)
        {
            return true;
        }

        int j2 = j1 + 1;
        int j3 = j1 + 2;
        if (n1 == 1 && n2 == 1)
        {
            // 1×1 同士: 回転1つ
            double t11 = t[j1 + j1 * ldt];
            double t22 = t[j2 + j2 * ldt];
            BidiagonalSvd.Rotation(t[j1 + j2 * ldt], t22 - t11, out double cs, out double sn, out _);
            for (int k = j3; k < n; k++)
            {
                double v1 = t[j1 + k * ldt], v2 = t[j2 + k * ldt];
                t[j1 + k * ldt] = cs * v1 + sn * v2;
                t[j2 + k * ldt] = cs * v2 - sn * v1;
            }

            if (j1 > 0)
            {
                Level1Kernels.Rotate(t.Slice(j1 * ldt, j1), t.Slice(j2 * ldt, j1), cs, sn);
            }

            t[j1 + j1 * ldt] = t22;
            t[j2 + j2 * ldt] = t11;
            if (qRows > 0)
            {
                Level1Kernels.Rotate(q.Slice(qRow + j1 * ldq, qRows), q.Slice(qRow + j2 * ldq, qRows), cs, sn);
            }

            return true;
        }

        // 2×2 を含む場合: シルベスター方程式 T₁₁X - XT₂₂ = γT₁₂ を解き、[-X; γI] の QR の Q で入れ替える
        int nd = n1 + n2;
        Span<double> dm = stackalloc double[16];
        for (int c = 0; c < nd; c++)
        {
            for (int r = 0; r < nd; r++)
            {
                dm[r + c * 4] = t[j1 + r + (j1 + c) * ldt];
            }
        }

        double dnorm = 0;
        for (int c = 0; c < nd; c++)
        {
            for (int r = 0; r < nd; r++)
            {
                dnorm = Math.Max(dnorm, Math.Abs(dm[r + c * 4]));
            }
        }

        double smlnum = NumericConstants.MinNormal / NumericConstants.MachineEpsilon;
        double thresh = Math.Max(10 * NumericConstants.MachineEpsilon * dnorm, smlnum);
        Span<double> x = stackalloc double[4];
        SolveSylvester(n1, n2, dm, 4, dm[(n1 + n1 * 4)..], 4, dm[(n1 * 4)..], 4, out double scale, x, 2);

        Span<double> u = stackalloc double[3];
        Span<double> u2 = stackalloc double[3];
        Span<double> work = stackalloc double[4];
        if (n1 == 1 && n2 == 2)
        {
            u[0] = scale;
            u[1] = x[0];
            u[2] = x[2];
            double beta = u[2];
            double tau = Householder.Generate(ref beta, u[..2]);
            u[2] = 1;
            double t11 = t[j1 + j1 * ldt];

            // 小さい行列で試して誤差を確認する
            ApplyLeft3(u, tau, 3, dm, 4);
            ApplyRight3(u, tau, 3, dm, 4);
            if (Math.Max(Math.Max(Math.Abs(dm[2]), Math.Abs(dm[2 + 4])), Math.Abs(dm[2 + 8] - t11)) > thresh)
            {
                return false;
            }

            ApplyLeft3(u, tau, n - j1, t[(j1 + j1 * ldt)..], ldt);
            ApplyRight3(u, tau, j2 + 1, t[(j1 * ldt)..], ldt);
            t[j3 + j1 * ldt] = 0;
            t[j3 + j2 * ldt] = 0;
            t[j3 + j3 * ldt] = t11;
            if (qRows > 0)
            {
                ApplyRight3(u, tau, qRows, q[(qRow + j1 * ldq)..], ldq);
            }
        }
        else if (n1 == 2 && n2 == 1)
        {
            u[0] = -x[0];
            u[1] = -x[1];
            u[2] = scale;
            double beta = u[0];
            double tau = Householder.Generate(ref beta, u[1..]);
            u[0] = 1;
            double t33 = t[j3 + j3 * ldt];

            ApplyLeft3(u, tau, 3, dm, 4);
            ApplyRight3(u, tau, 3, dm, 4);
            if (Math.Max(Math.Max(Math.Abs(dm[1]), Math.Abs(dm[2])), Math.Abs(dm[0] - t33)) > thresh)
            {
                return false;
            }

            ApplyRight3(u, tau, j3 + 1, t[(j1 * ldt)..], ldt);
            ApplyLeft3(u, tau, n - j1 - 1, t[(j1 + j2 * ldt)..], ldt);
            t[j1 + j1 * ldt] = t33;
            t[j2 + j1 * ldt] = 0;
            t[j3 + j1 * ldt] = 0;
            if (qRows > 0)
            {
                ApplyRight3(u, tau, qRows, q[(qRow + j1 * ldq)..], ldq);
            }
        }
        else
        {
            // 2×2 同士: 反射を2つ使う
            u[0] = -x[0];
            u[1] = -x[1];
            u[2] = scale;
            double beta = u[0];
            double tau1 = Householder.Generate(ref beta, u[1..]);
            u[0] = 1;
            double temp = -tau1 * (x[2] + u[1] * x[3]);
            u2[0] = -temp * u[1] - x[3];
            u2[1] = -temp * u[2];
            u2[2] = scale;
            beta = u2[0];
            double tau2 = Householder.Generate(ref beta, u2[1..]);
            u2[0] = 1;

            ApplyLeft3(u, tau1, 4, dm, 4);
            ApplyRight3(u, tau1, 4, dm, 4);
            ApplyLeft3(u2, tau2, 4, dm[1..], 4);
            ApplyRight3(u2, tau2, 4, dm[4..], 4);
            double err = Math.Max(Math.Max(Math.Abs(dm[2]), Math.Abs(dm[2 + 4])), Math.Max(Math.Abs(dm[3]), Math.Abs(dm[3 + 4])));
            if (err > thresh)
            {
                return false;
            }

            int j4 = j1 + 3;
            ApplyLeft3(u, tau1, n - j1, t[(j1 + j1 * ldt)..], ldt);
            ApplyRight3(u, tau1, j4 + 1, t[(j1 * ldt)..], ldt);
            ApplyLeft3(u2, tau2, n - j1, t[(j2 + j1 * ldt)..], ldt);
            ApplyRight3(u2, tau2, j4 + 1, t[(j2 * ldt)..], ldt);
            t[j3 + j1 * ldt] = 0;
            t[j3 + j2 * ldt] = 0;
            t[j4 + j1 * ldt] = 0;
            t[j4 + j2 * ldt] = 0;
            if (qRows > 0)
            {
                ApplyRight3(u, tau1, qRows, q[(qRow + j1 * ldq)..], ldq);
                ApplyRight3(u2, tau2, qRows, q[(qRow + j2 * ldq)..], ldq);
            }
        }

        // 入れ替えた 2×2 ブロックを標準形に戻す
        if (n2 == 2)
        {
            StandardizeBlock(j1, t, ldt, 0, n, q, ldq, qRow, qRows, out _, out _, out _, out _);
        }

        if (n1 == 2)
        {
            int k = j1 + n2;
            StandardizeBlock(k, t, ldt, 0, n, q, ldq, qRow, qRows, out _, out _, out _, out _);
        }

        return true;
    }

    // LAPACK dtrexc の上向き
    /// <summary>
    /// ブロック（先頭 <paramref name="from"/>）を、上にあるブロックと順に入れ替えて先頭 <paramref name="to"/> まで移す
    /// </summary>
    /// <returns>
    /// 移動できたら真。入れ替えに失敗した場合や、移動中のブロックが 1×1 に分かれた場合は偽（それまでの入れ替えは残る）。
    /// </returns>
    internal static bool MoveUp(int n, Span<double> t, int ldt, Span<double> q, int ldq, int qRow, int qRows, int from, int to)
    {
        int size = from + 1 < n && t[from + 1 + from * ldt] != 0 ? 2 : 1;
        int here = from;
        while (here > to)
        {
            int above = here >= 2 && t[here - 1 + (here - 2) * ldt] != 0 ? 2 : 1;
            if (here - above < to)
            {
                above = 1;
            }

            if (!Swap(n, t, ldt, q, ldq, qRow, qRows, here - above, above, size))
            {
                return false;
            }

            here -= above;
            if (size == 2 && t[here + 1 + here * ldt] == 0)
            {
                return false;
            }
        }

        return true;
    }

    // LAPACK dlasy2
    // クロネッカー積の形の連立一次方程式を完全ピボット選択のガウス消去で解き、小さすぎるピボットは摂動する。
    // TL は n1 × n1、TR は n2 × n2、n1, n2 ≤ 2。
    /// <summary>
    /// 小さいシルベスター方程式 <c>TL·X - X·TR = scale·B</c>を解く
    /// </summary>
    internal static void SolveSylvester(
        int n1, int n2, ReadOnlySpan<double> tl, int ldtl, ReadOnlySpan<double> tr, int ldtr, ReadOnlySpan<double> b, int ldb,
        out double scale, Span<double> x, int ldx)
    {
        int size = n1 * n2;
        Span<double> m = stackalloc double[16];
        Span<double> rhs = stackalloc double[4];
        m.Clear();
        double smin = 0;
        for (int i = 0; i < n1; i++)
        {
            for (int k = 0; k < n1; k++)
            {
                smin = Math.Max(smin, Math.Abs(tl[i + k * ldtl]));
            }
        }

        for (int i = 0; i < n2; i++)
        {
            for (int k = 0; k < n2; k++)
            {
                smin = Math.Max(smin, Math.Abs(tr[i + k * ldtr]));
            }
        }

        double smlnum = NumericConstants.MinNormal / NumericConstants.MachineEpsilon;
        smin = Math.Max(NumericConstants.MachineEpsilon * smin, smlnum);

        // 未知数 X[i, j] の番号は i + j·n1
        for (int j = 0; j < n2; j++)
        {
            for (int i = 0; i < n1; i++)
            {
                int row = i + j * n1;
                rhs[row] = b[i + j * ldb];
                for (int k = 0; k < n1; k++)
                {
                    m[row + (k + j * n1) * 4] += tl[i + k * ldtl];
                }

                for (int k = 0; k < n2; k++)
                {
                    m[row + (i + k * n1) * 4] -= tr[k + j * ldtr];
                }
            }
        }

        // 完全ピボット選択のガウス消去
        Span<int> colPerm = stackalloc int[4];
        for (int i = 0; i < size; i++)
        {
            colPerm[i] = i;
        }

        scale = 1;
        for (int k = 0; k < size; k++)
        {
            int pr = k, pc = k;
            double best = -1;
            for (int c = k; c < size; c++)
            {
                for (int r = k; r < size; r++)
                {
                    double v = Math.Abs(m[r + c * 4]);
                    if (v > best)
                    {
                        best = v;
                        pr = r;
                        pc = c;
                    }
                }
            }

            if (pr != k)
            {
                for (int c = 0; c < size; c++)
                {
                    (m[k + c * 4], m[pr + c * 4]) = (m[pr + c * 4], m[k + c * 4]);
                }

                (rhs[k], rhs[pr]) = (rhs[pr], rhs[k]);
            }

            if (pc != k)
            {
                for (int r = 0; r < size; r++)
                {
                    (m[r + k * 4], m[r + pc * 4]) = (m[r + pc * 4], m[r + k * 4]);
                }

                (colPerm[k], colPerm[pc]) = (colPerm[pc], colPerm[k]);
            }

            if (Math.Abs(m[k + k * 4]) < smin)
            {
                m[k + k * 4] = smin;
            }

            for (int r = k + 1; r < size; r++)
            {
                double f = m[r + k * 4] / m[k + k * 4];
                if (f == 0)
                {
                    continue;
                }

                for (int c = k + 1; c < size; c++)
                {
                    m[r + c * 4] -= f * m[k + c * 4];
                }

                rhs[r] -= f * rhs[k];
            }
        }

        // オーバーフローを避ける縮尺
        double bmax = 0;
        for (int i = 0; i < size; i++)
        {
            bmax = Math.Max(bmax, Math.Abs(rhs[i]));
        }

        if (8 * smlnum * bmax > Math.Abs(m[size - 1 + (size - 1) * 4]))
        {
            scale = 0.125 / bmax;
            for (int i = 0; i < size; i++)
            {
                rhs[i] *= scale;
            }
        }

        Span<double> sol = stackalloc double[4];
        for (int k = size - 1; k >= 0; k--)
        {
            double sum = rhs[k];
            for (int c = k + 1; c < size; c++)
            {
                sum -= m[k + c * 4] * sol[c];
            }

            sol[k] = sum / m[k + k * 4];
        }

        for (int k = 0; k < size; k++)
        {
            int idx = colPerm[k];
            x[idx % n1 + idx / n1 * ldx] = sol[k];
        }
    }

    /// <summary>
    /// 長さ3の反射 <c>H = I - τuuᵀ</c> を <c>C</c>（3 × n、先頭3行）に左から掛ける
    /// </summary>
    private static void ApplyLeft3(ReadOnlySpan<double> u, double tau, int n, Span<double> c, int ldc)
    {
        if (tau == 0)
        {
            return;
        }

        double u0 = u[0], u1 = u[1], u2 = u[2];
        for (int j = 0; j < n; j++)
        {
            int o = j * ldc;
            double s = u0 * c[o] + u1 * c[o + 1] + u2 * c[o + 2];
            s *= tau;
            c[o] -= s * u0;
            c[o + 1] -= s * u1;
            c[o + 2] -= s * u2;
        }
    }

    /// <summary>
    /// 長さ3の反射 <c>H = I - τuuᵀ</c> を <c>C</c>（m × 3、先頭3列）に右から掛ける
    /// </summary>
    private static void ApplyRight3(ReadOnlySpan<double> u, double tau, int m, Span<double> c, int ldc)
    {
        if (tau == 0)
        {
            return;
        }

        double u0 = u[0], u1 = u[1], u2 = u[2];
        var c0 = c[..m];
        var c1 = c.Slice(ldc, m);
        var c2 = c.Slice(2 * ldc, m);
        for (int i = 0; i < m; i++)
        {
            double s = tau * (c0[i] * u0 + c1[i] * u1 + c2[i] * u2);
            c0[i] -= s * u0;
            c1[i] -= s * u1;
            c2[i] -= s * u2;
        }
    }
}
