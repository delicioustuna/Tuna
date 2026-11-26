namespace Tuna.LinearAlgebra;

// Demmel–Kahan の零シフト QR と相対的な収束判定により、二重対角行列の特異値を高い相対精度で求める。
// 回転は U と V の列に作用させる（Vᵀ の行への作用と同値）。
// LAPACK dbdsqr 相当の陰的零シフト／シフト付き QR 法
/// <summary>
/// 上二重対角行列の特異値分解
/// </summary>
internal static class BidiagonalSvd
{
    private const double Epsilon = 1.1102230246251565e-16;
    private const double SafeMin = 2.2250738585072014e-308;
    private const int MaxIterationFactor = 6;

    /// <summary>
    /// <paramref name="d"/>（長さ n）と <paramref name="e"/>（長さ n-1）を特異値（降順）で上書きし、 <c>U</c>（<c>uRows × n</c>）と <c>V</c>（<c>vRows × n</c>）の列に回転を作用させる
    /// </summary>
    /// <returns>
    /// 収束したら真。
    /// </returns>
    internal static bool Solve(int n, Span<double> d, Span<double> e, Span<double> u, int uRows, int ldu, Span<double> v, int vRows, int ldv)
    {
        if (n == 0)
        {
            return true;
        }

        bool vectors = uRows > 0 || vRows > 0;
        bool ok = true;
        if (n > 1)
        {
            double[] rented = WorkArrayPool<double>.Rent(checked(4 * n));
            try
            {
                ok = Iterate(n, d, e, u, uRows, ldu, v, vRows, ldv, rented.AsSpan(0, 4 * n), vectors);
            }
            finally
            {
                WorkArrayPool<double>.Return(rented);
            }
        }

        // 特異値を非負にする
        for (int i = 0; i < n; i++)
        {
            if (d[i] < 0)
            {
                d[i] = -d[i];
                if (vRows > 0)
                {
                    Level1Kernels.Scale(-1.0, v.Slice(i * ldv, vRows));
                }
            }
        }

        // 降順に並べ替える（選択ソート）
        for (int i = 0; i < n - 1; i++)
        {
            int k = i;
            double max = d[i];
            for (int j = i + 1; j < n; j++)
            {
                if (d[j] > max)
                {
                    k = j;
                    max = d[j];
                }
            }

            if (k != i)
            {
                d[k] = d[i];
                d[i] = max;
                if (uRows > 0)
                {
                    Level1Kernels.Swap(u.Slice(i * ldu, uRows), u.Slice(k * ldu, uRows));
                }

                if (vRows > 0)
                {
                    Level1Kernels.Swap(v.Slice(i * ldv, vRows), v.Slice(k * ldv, vRows));
                }
            }
        }

        return ok;
    }

    private static bool Iterate(int n, Span<double> d, Span<double> e, Span<double> u, int uRows, int ldu, Span<double> v, int vRows, int ldv, Span<double> work, bool vectors)
    {
        double eps = Epsilon;
        double unfl = SafeMin;
        double tolmul = Math.Max(10.0, Math.Min(100.0, Math.Pow(eps, -0.125)));
        double tol = tolmul * eps;

        // 最大特異値の近似
        double smax = 0;
        for (int i = 0; i < n; i++)
        {
            smax = Math.Max(smax, Math.Abs(d[i]));
        }

        for (int i = 0; i < n - 1; i++)
        {
            smax = Math.Max(smax, Math.Abs(e[i]));
        }

        // 最小特異値の下界の近似（相対精度の閾値）
        double sminoa = Math.Abs(d[0]);
        if (sminoa != 0)
        {
            double mu = sminoa;
            for (int i = 1; i < n; i++)
            {
                mu = Math.Abs(d[i]) * (mu / (mu + Math.Abs(e[i - 1])));
                sminoa = Math.Min(sminoa, mu);
                if (sminoa == 0)
                {
                    break;
                }
            }
        }

        sminoa /= Math.Sqrt(n);
        double thresh = Math.Max(tol * sminoa, MaxIterationFactor * (n * (n * unfl)));

        long maxIterations = (long)MaxIterationFactor * n * n;
        long iterations = 0;
        int oldll = -1;
        int oldm = -1;
        int idir = 0;
        int m = n - 1;
        var cs1 = work[..n];
        var sn1 = work.Slice(n, n);
        var cs2 = work.Slice(2 * n, n);
        var sn2 = work.Slice(3 * n, n);

        while (m > 0)
        {
            if (iterations > maxIterations)
            {
                return false;
            }

            // 分離した下端のブロック [ll, m] を探す
            double smaxBlock = Math.Abs(d[m]);
            int ll = -1;
            bool split = false;
            for (int lll = 1; lll <= m; lll++)
            {
                ll = m - lll;
                double abss = Math.Abs(d[ll]);
                double abse = Math.Abs(e[ll]);
                if (abse <= thresh)
                {
                    split = true;
                    break;
                }

                smaxBlock = Math.Max(smaxBlock, Math.Max(abss, abse));
            }

            if (split)
            {
                e[ll] = 0;
                if (ll == m - 1)
                {
                    m--;
                    continue;
                }
            }
            else
            {
                ll = -1;
            }

            ll++;

            // 2×2 ブロックは直接解く
            if (ll == m - 1)
            {
                Svd2x2(d[m - 1], e[m - 1], d[m], out double sigmn, out double sigmx, out double sinr, out double cosr, out double sinl, out double cosl);
                d[m - 1] = sigmx;
                e[m - 1] = 0;
                d[m] = sigmn;
                if (vRows > 0)
                {
                    Level1Kernels.Rotate(v.Slice((m - 1) * ldv, vRows), v.Slice(m * ldv, vRows), cosr, sinr);
                }

                if (uRows > 0)
                {
                    Level1Kernels.Rotate(u.Slice((m - 1) * ldu, uRows), u.Slice(m * ldu, uRows), cosl, sinl);
                }

                m -= 2;
                continue;
            }

            // 新しいブロックなら追跡の向きを決める（大きい側から小さい側へ）
            if (ll > oldm || m < oldll)
            {
                idir = Math.Abs(d[ll]) >= Math.Abs(d[m]) ? 1 : 2;
            }

            // 収束判定
            double sminl;
            if (idir == 1)
            {
                if (Math.Abs(e[m - 1]) <= tol * Math.Abs(d[m]))
                {
                    e[m - 1] = 0;
                    continue;
                }

                double mu = Math.Abs(d[ll]);
                sminl = mu;
                bool converged = false;
                for (int lll = ll; lll < m; lll++)
                {
                    if (Math.Abs(e[lll]) <= tol * mu)
                    {
                        e[lll] = 0;
                        converged = true;
                        break;
                    }

                    mu = Math.Abs(d[lll + 1]) * (mu / (mu + Math.Abs(e[lll])));
                    sminl = Math.Min(sminl, mu);
                }

                if (converged)
                {
                    continue;
                }
            }
            else
            {
                if (Math.Abs(e[ll]) <= tol * Math.Abs(d[ll]))
                {
                    e[ll] = 0;
                    continue;
                }

                double mu = Math.Abs(d[m]);
                sminl = mu;
                bool converged = false;
                for (int lll = m - 1; lll >= ll; lll--)
                {
                    if (Math.Abs(e[lll]) <= tol * mu)
                    {
                        e[lll] = 0;
                        converged = true;
                        break;
                    }

                    mu = Math.Abs(d[lll]) * (mu / (mu + Math.Abs(e[lll])));
                    sminl = Math.Min(sminl, mu);
                }

                if (converged)
                {
                    continue;
                }
            }

            oldll = ll;
            oldm = m;

            // シフトの計算（相対精度を損なうほど大きければ零シフト）
            double shift;
            if (n * tol * (sminl / smaxBlock) <= Math.Max(eps, 0.01 * tol))
            {
                shift = 0;
            }
            else
            {
                double sll;
                if (idir == 1)
                {
                    sll = Math.Abs(d[ll]);
                    shift = SingularValues2x2Min(d[m - 1], e[m - 1], d[m]);
                }
                else
                {
                    sll = Math.Abs(d[m]);
                    shift = SingularValues2x2Min(d[ll], e[ll], d[ll + 1]);
                }

                if (sll > 0 && (shift / sll) * (shift / sll) < eps)
                {
                    shift = 0;
                }
            }

            iterations += m - ll;
            int count = m - ll;

            if (shift == 0)
            {
                if (idir == 1)
                {
                    double cs = 1;
                    double oldcs = 1;
                    double sn = 0;
                    double oldsn = 0;
                    for (int i = ll; i < m; i++)
                    {
                        Rotation(d[i] * cs, e[i], out cs, out sn, out double r);
                        if (i > ll)
                        {
                            e[i - 1] = oldsn * r;
                        }

                        Rotation(oldcs * r, d[i + 1] * sn, out oldcs, out oldsn, out d[i]);
                        cs1[i - ll] = cs;
                        sn1[i - ll] = sn;
                        cs2[i - ll] = oldcs;
                        sn2[i - ll] = oldsn;
                    }

                    double h = d[m] * cs;
                    d[m] = h * oldcs;
                    e[m - 1] = h * oldsn;
                    if (vectors)
                    {
                        ApplyForward(v, vRows, ldv, ll, cs1[..count], sn1[..count]);
                        ApplyForward(u, uRows, ldu, ll, cs2[..count], sn2[..count]);
                    }

                    if (Math.Abs(e[m - 1]) <= thresh)
                    {
                        e[m - 1] = 0;
                    }
                }
                else
                {
                    double cs = 1;
                    double oldcs = 1;
                    double sn = 0;
                    double oldsn = 0;
                    for (int i = m; i > ll; i--)
                    {
                        Rotation(d[i] * cs, e[i - 1], out cs, out sn, out double r);
                        if (i < m)
                        {
                            e[i] = oldsn * r;
                        }

                        Rotation(oldcs * r, d[i - 1] * sn, out oldcs, out oldsn, out d[i]);
                        cs1[i - ll - 1] = cs;
                        sn1[i - ll - 1] = -sn;
                        cs2[i - ll - 1] = oldcs;
                        sn2[i - ll - 1] = -oldsn;
                    }

                    double h = d[ll] * cs;
                    d[ll] = h * oldcs;
                    e[ll] = h * oldsn;
                    if (vectors)
                    {
                        ApplyBackward(v, vRows, ldv, ll, cs2[..count], sn2[..count]);
                        ApplyBackward(u, uRows, ldu, ll, cs1[..count], sn1[..count]);
                    }

                    if (Math.Abs(e[ll]) <= thresh)
                    {
                        e[ll] = 0;
                    }
                }
            }
            else
            {
                if (idir == 1)
                {
                    double f = (Math.Abs(d[ll]) - shift) * (Math.CopySign(1.0, d[ll]) + shift / d[ll]);
                    double g = e[ll];
                    for (int i = ll; i < m; i++)
                    {
                        Rotation(f, g, out double cosr, out double sinr, out double r);
                        if (i > ll)
                        {
                            e[i - 1] = r;
                        }

                        f = cosr * d[i] + sinr * e[i];
                        e[i] = cosr * e[i] - sinr * d[i];
                        g = sinr * d[i + 1];
                        d[i + 1] = cosr * d[i + 1];
                        Rotation(f, g, out double cosl, out double sinl, out r);
                        d[i] = r;
                        f = cosl * e[i] + sinl * d[i + 1];
                        d[i + 1] = cosl * d[i + 1] - sinl * e[i];
                        if (i < m - 1)
                        {
                            g = sinl * e[i + 1];
                            e[i + 1] = cosl * e[i + 1];
                        }

                        cs1[i - ll] = cosr;
                        sn1[i - ll] = sinr;
                        cs2[i - ll] = cosl;
                        sn2[i - ll] = sinl;
                    }

                    e[m - 1] = f;
                    if (vectors)
                    {
                        ApplyForward(v, vRows, ldv, ll, cs1[..count], sn1[..count]);
                        ApplyForward(u, uRows, ldu, ll, cs2[..count], sn2[..count]);
                    }

                    if (Math.Abs(e[m - 1]) <= thresh)
                    {
                        e[m - 1] = 0;
                    }
                }
                else
                {
                    double f = (Math.Abs(d[m]) - shift) * (Math.CopySign(1.0, d[m]) + shift / d[m]);
                    double g = e[m - 1];
                    for (int i = m; i > ll; i--)
                    {
                        Rotation(f, g, out double cosr, out double sinr, out double r);
                        if (i < m)
                        {
                            e[i] = r;
                        }

                        f = cosr * d[i] + sinr * e[i - 1];
                        e[i - 1] = cosr * e[i - 1] - sinr * d[i];
                        g = sinr * d[i - 1];
                        d[i - 1] = cosr * d[i - 1];
                        Rotation(f, g, out double cosl, out double sinl, out r);
                        d[i] = r;
                        f = cosl * e[i - 1] + sinl * d[i - 1];
                        d[i - 1] = cosl * d[i - 1] - sinl * e[i - 1];
                        if (i > ll + 1)
                        {
                            g = sinl * e[i - 2];
                            e[i - 2] = cosl * e[i - 2];
                        }

                        cs1[i - ll - 1] = cosr;
                        sn1[i - ll - 1] = -sinr;
                        cs2[i - ll - 1] = cosl;
                        sn2[i - ll - 1] = -sinl;
                    }

                    e[ll] = f;
                    if (Math.Abs(e[ll]) <= thresh)
                    {
                        e[ll] = 0;
                    }

                    if (vectors)
                    {
                        ApplyBackward(u, uRows, ldu, ll, cs1[..count], sn1[..count]);
                        ApplyBackward(v, vRows, ldv, ll, cs2[..count], sn2[..count]);
                    }
                }
            }
        }

        return true;
    }

    // LAPACK dlasr 'R','V','F'
    /// <summary>
    /// 列 <c>(start+j, start+j+1)</c> に回転 <c>j = 0, 1, …</c> を順に作用させる
    /// </summary>
    private static void ApplyForward(Span<double> z, int rows, int ldz, int start, ReadOnlySpan<double> c, ReadOnlySpan<double> s) =>
        Level1Kernels.RotateSequence(z, ldz, rows, start, c, s, forward: true);

    // LAPACK dlasr 'R','V','B'
    /// <summary>
    /// 列 <c>(start+j, start+j+1)</c> に回転 <c>j = …, 1, 0</c> を逆順に作用させる
    /// </summary>
    private static void ApplyBackward(Span<double> z, int rows, int ldz, int start, ReadOnlySpan<double> c, ReadOnlySpan<double> s) =>
        Level1Kernels.RotateSequence(z, ldz, rows, start, c, s, forward: false);

    // [c s; -s c]·[f; g] = [r; 0] となる平面回転。
    /// <summary>
    /// LAPACK dlartg
    /// </summary>
    internal static void Rotation(double f, double g, out double c, out double s, out double r)
    {
        if (g == 0)
        {
            c = 1;
            s = 0;
            r = f;
            return;
        }

        if (f == 0)
        {
            c = 0;
            s = Math.CopySign(1.0, g);
            r = Math.Abs(g);
            return;
        }

        double f1 = Math.Abs(f);
        double g1 = Math.Abs(g);
        const double rtmin = 1.4916681462400413e-154;
        const double rtmax = 4.7403759540545887e+153;
        if (f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax)
        {
            double dd = Math.Sqrt(f * f + g * g);
            c = f1 / dd;
            r = Math.CopySign(dd, f);
            s = g / r;
        }
        else
        {
            double u = Math.Min(4.49423283715579e+307, Math.Max(SafeMin, Math.Max(f1, g1)));
            double fs = f / u;
            double gs = g / u;
            double dd = Math.Sqrt(fs * fs + gs * gs);
            c = Math.Abs(fs) / dd;
            r = Math.CopySign(dd, f);
            s = gs / r;
            r *= u;
        }
    }

    // [f g; 0 h] の小さい方の特異値。
    /// <summary>
    /// LAPACK dlas2
    /// </summary>
    internal static double SingularValues2x2Min(double f, double g, double h)
    {
        double fa = Math.Abs(f);
        double ga = Math.Abs(g);
        double ha = Math.Abs(h);
        double fhmn = Math.Min(fa, ha);
        double fhmx = Math.Max(fa, ha);
        if (fhmn == 0)
        {
            return 0;
        }

        if (ga < fhmx)
        {
            double asv = 1 + fhmn / fhmx;
            double at = (fhmx - fhmn) / fhmx;
            double au = (ga / fhmx) * (ga / fhmx);
            double c = 2 / (Math.Sqrt(asv * asv + au) + Math.Sqrt(at * at + au));
            return fhmn * c;
        }
        else
        {
            double au = fhmx / ga;
            if (au == 0)
            {
                return (fhmn * fhmx) / ga;
            }

            double asv = 1 + fhmn / fhmx;
            double at = (fhmx - fhmn) / fhmx;
            double c = 1 / (Math.Sqrt(1 + (asv * au) * (asv * au)) + Math.Sqrt(1 + (at * au) * (at * au)));
            double ssmin = (fhmn * c) * au;
            return ssmin + ssmin;
        }
    }

    // 上三角 [f g; 0 h] の特異値分解 [csl snl; -snl csl]·[f g; 0 h]·[csr -snr; snr csr] = diag(ssmax, ssmin)。
    /// <summary>
    /// LAPACK dlasv2
    /// </summary>
    internal static void Svd2x2(double f, double g, double h, out double ssmin, out double ssmax, out double snr, out double csr, out double snl, out double csl)
    {
        const double eps = 2.220446049250313e-16;
        double ft = f;
        double fa = Math.Abs(ft);
        double ht = h;
        double ha = Math.Abs(h);
        int pmax = 1;
        bool swap = ha > fa;
        if (swap)
        {
            pmax = 3;
            (ft, ht) = (ht, ft);
            (fa, ha) = (ha, fa);
        }

        double gt = g;
        double ga = Math.Abs(gt);
        double clt, crt, slt, srt;
        if (ga == 0)
        {
            ssmin = ha;
            ssmax = fa;
            clt = 1;
            crt = 1;
            slt = 0;
            srt = 0;
        }
        else
        {
            bool gasmal = true;
            ssmin = 0;
            ssmax = 0;
            clt = crt = slt = srt = 0;
            if (ga > fa)
            {
                pmax = 2;
                if (fa / ga < eps)
                {
                    gasmal = false;
                    ssmax = ga;
                    ssmin = ha > 1 ? fa / (ga / ha) : (fa / ga) * ha;
                    clt = 1;
                    slt = ht / gt;
                    srt = 1;
                    crt = ft / gt;
                }
            }

            if (gasmal)
            {
                double dd = fa - ha;
                double l = dd == fa ? 1.0 : dd / fa;
                double mm0 = gt / ft;
                double t = 2 - l;
                double mm = mm0 * mm0;
                double tt = t * t;
                double s = Math.Sqrt(tt + mm);
                double r = l == 0 ? Math.Abs(mm0) : Math.Sqrt(l * l + mm);
                double a = 0.5 * (s + r);
                ssmin = ha / a;
                ssmax = fa * a;
                if (mm == 0)
                {
                    if (l == 0)
                    {
                        t = Math.CopySign(2.0, ft) * Math.CopySign(1.0, gt);
                    }
                    else
                    {
                        t = gt / Math.CopySign(dd, ft) + mm0 / t;
                    }
                }
                else
                {
                    t = (mm0 / (s + t) + mm0 / (r + l)) * (1 + a);
                }

                l = Math.Sqrt(t * t + 4);
                crt = 2 / l;
                srt = t / l;
                clt = (crt + srt * mm0) / a;
                slt = (ht / ft) * srt / a;
            }
        }

        if (swap)
        {
            csl = srt;
            snl = crt;
            csr = slt;
            snr = clt;
        }
        else
        {
            csl = clt;
            snl = slt;
            csr = crt;
            snr = srt;
        }

        double tsign = pmax switch
        {
            1 => Math.CopySign(1.0, csr) * Math.CopySign(1.0, csl) * Math.CopySign(1.0, f),
            2 => Math.CopySign(1.0, snr) * Math.CopySign(1.0, csl) * Math.CopySign(1.0, g),
            _ => Math.CopySign(1.0, snr) * Math.CopySign(1.0, snl) * Math.CopySign(1.0, h),
        };
        ssmax = Math.CopySign(ssmax, tsign);
        ssmin = Math.CopySign(ssmin, tsign * Math.CopySign(1.0, f) * Math.CopySign(1.0, h));
    }
}
