namespace Tuna.LinearAlgebra;

internal static partial class HessenbergQr
{
    // 各固有ベクトル（複素共役の組は2列）の後退代入は自分の列だけを書き換えるので、固有値ごとに別のスレッドで解ける。
    // 他の列の解で上書きされる対角・優対角は転置した写し、副対角は別の配列から読む
    /// <summary>
    /// 実シューア形の後退代入で固有ベクトルを求め、<c>Z</c> に逆変換を掛ける
    /// </summary>
    private static unsafe void BackSubstitute(int nn, int low, int high, Span<double> h, Span<double> wr, Span<double> wi, Span<double> z, double norm)
    {
        int ld = nn;
        if (norm == 0.0)
        {
            return;
        }

        // 行 i の内積を連続領域で取るため、上三角部分の転置を作っておく（後退代入で書き換わる前の値だけを読む）
        double[] transposed = WorkArrayPool<double>.Rent(checked(nn * nn + nn));
        var ht = transposed.AsSpan(0, nn * nn);
        var sub = transposed.AsSpan(nn * nn, nn);
        for (int j = 0; j < nn; j++)
        {
            for (int i = 0; i <= j; i++)
            {
                ht[j + i * ld] = h[i + j * ld];
            }

            sub[j] = j + 1 < nn ? h[j + 1 + j * ld] : 0;
        }

        // 列 n の仕事量は n² に比例するので、大きい列から動的に割り当てる
        int threads = Parallelism.Degree((double)nn * nn * nn / 6, 4 * Parallelism.Grain);
        fixed (double* ph = h)
        fixed (double* pwr = wr)
        fixed (double* pwi = wi)
        {
            var sh = new SharedSpan(ph, h.Length);
            var swr = new SharedSpan(pwr, wr.Length);
            var swi = new SharedSpan(pwi, wi.Length);
            Parallelism.For(nn, threads, k => SolveVector(nn - 1 - k, sh.Span, ld, swr.Span, swi.Span, transposed.AsSpan(0, nn * nn), transposed.AsSpan(nn * nn, nn), norm));
        }

        // 上三角部分だけを残して X とし、Z ← Z·X（ヘッセンベルグ化の逆変換。積は転置の写しに使った領域に作る）
        for (int j = 0; j < nn; j++)
        {
            h.Slice(j + 1 + j * ld, nn - j - 1).Clear();
        }

        var result = transposed.AsSpan(0, nn * nn);
        result.Clear();
        Gemm.Multiply(false, false, nn, nn, nn, 1.0, z, ld, h, ld, result, ld);
        result.CopyTo(z);
        WorkArrayPool<double>.Return(transposed);
    }

    // 組の先頭（wi[n] > 0）では何もしない。
    /// <summary>
    /// 固有値 <paramref name="n"/> の固有ベクトル（複素共役の組なら <c>n-1</c> 列が実部、<c>n</c> 列が虚部）を後退代入で求め、 その列に書く
    /// </summary>
    private static void SolveVector(int n, Span<double> h, int ld, ReadOnlySpan<double> wr, ReadOnlySpan<double> wi, ReadOnlySpan<double> ht, ReadOnlySpan<double> sub, double norm)
    {
        double eps = NumericConstants.MachineEpsilon;
        double p, q, r = 0, s = 0, zz = 0, t, w, x, y;
        p = wr[n];
        q = wi[n];

        if (q == 0)
        {
            // 実固有ベクトル
            int l = n;
            h[n + n * ld] = 1.0;
            for (int i = n - 1; i >= 0; i--)
            {
                w = ht[i + i * ld] - p;
                r = Level1Kernels.Dot(ht.Slice(l + i * ld, n - l + 1), h.Slice(l + n * ld, n - l + 1));

                if (wi[i] < 0.0)
                {
                    zz = w;
                    s = r;
                }
                else
                {
                    l = i;
                    if (wi[i] == 0.0)
                    {
                        h[i + n * ld] = w != 0.0 ? -r / w : -r / (eps * norm);
                    }
                    else
                    {
                        x = ht[(i + 1) + i * ld];
                        y = sub[i];
                        q = (wr[i] - p) * (wr[i] - p) + wi[i] * wi[i];
                        t = (x * s - zz * r) / q;
                        h[i + n * ld] = t;
                        h[(i + 1) + n * ld] = Math.Abs(x) > Math.Abs(zz) ? (-r - w * t) / x : (-s - y * t) / zz;
                    }

                    // オーバーフロー対策
                    t = Math.Abs(h[i + n * ld]);
                    if ((eps * t) * t > 1)
                    {
                        for (int j = i; j <= n; j++)
                        {
                            h[j + n * ld] /= t;
                        }
                    }
                }
            }
        }
        else if (q < 0)
        {
            // 複素固有ベクトル（n-1 列が実部、n 列が虚部）
            int l = n - 1;
            double cr, ci;
            if (Math.Abs(h[n + (n - 1) * ld]) > Math.Abs(h[(n - 1) + n * ld]))
            {
                h[(n - 1) + (n - 1) * ld] = q / h[n + (n - 1) * ld];
                h[(n - 1) + n * ld] = -(h[n + n * ld] - p) / h[n + (n - 1) * ld];
            }
            else
            {
                ComplexDivide(0.0, -h[(n - 1) + n * ld], h[(n - 1) + (n - 1) * ld] - p, q, out cr, out ci);
                h[(n - 1) + (n - 1) * ld] = cr;
                h[(n - 1) + n * ld] = ci;
            }

            h[n + (n - 1) * ld] = 0.0;
            h[n + n * ld] = 1.0;
            for (int i = n - 2; i >= 0; i--)
            {
                var row = ht.Slice(l + i * ld, n - l + 1);
                double ra = Level1Kernels.Dot(row, h.Slice(l + (n - 1) * ld, n - l + 1));
                double sa = Level1Kernels.Dot(row, h.Slice(l + n * ld, n - l + 1));

                w = ht[i + i * ld] - p;
                if (wi[i] < 0.0)
                {
                    zz = w;
                    r = ra;
                    s = sa;
                }
                else
                {
                    l = i;
                    if (wi[i] == 0)
                    {
                        ComplexDivide(-ra, -sa, w, q, out cr, out ci);
                        h[i + (n - 1) * ld] = cr;
                        h[i + n * ld] = ci;
                    }
                    else
                    {
                        x = ht[(i + 1) + i * ld];
                        y = sub[i];
                        double vr = (wr[i] - p) * (wr[i] - p) + wi[i] * wi[i] - q * q;
                        double vi = (wr[i] - p) * 2.0 * q;
                        if (vr == 0.0 && vi == 0.0)
                        {
                            vr = eps * norm * (Math.Abs(w) + Math.Abs(q) + Math.Abs(x) + Math.Abs(y) + Math.Abs(zz));
                        }

                        ComplexDivide(x * r - zz * ra + q * sa, x * s - zz * sa - q * ra, vr, vi, out cr, out ci);
                        h[i + (n - 1) * ld] = cr;
                        h[i + n * ld] = ci;
                        if (Math.Abs(x) > Math.Abs(zz) + Math.Abs(q))
                        {
                            h[(i + 1) + (n - 1) * ld] = (-ra - w * h[i + (n - 1) * ld] + q * h[i + n * ld]) / x;
                            h[(i + 1) + n * ld] = (-sa - w * h[i + n * ld] - q * h[i + (n - 1) * ld]) / x;
                        }
                        else
                        {
                            ComplexDivide(-r - y * h[i + (n - 1) * ld], -s - y * h[i + n * ld], zz, q, out cr, out ci);
                            h[(i + 1) + (n - 1) * ld] = cr;
                            h[(i + 1) + n * ld] = ci;
                        }
                    }

                    // オーバーフロー対策
                    t = Math.Max(Math.Abs(h[i + (n - 1) * ld]), Math.Abs(h[i + n * ld]));
                    if ((eps * t) * t > 1)
                    {
                        for (int j = i; j <= n; j++)
                        {
                            h[j + (n - 1) * ld] /= t;
                            h[j + n * ld] /= t;
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// 複素数の除算 <c>(xr + i·xi)/(yr + i·yi)</c>（Smith の方法）
    /// </summary>
    private static void ComplexDivide(double xr, double xi, double yr, double yi, out double cr, out double ci)
    {
        double r, d;
        if (Math.Abs(yr) > Math.Abs(yi))
        {
            r = yi / yr;
            d = yr + r * yi;
            cr = (xr + r * xi) / d;
            ci = (xi - r * xr) / d;
        }
        else
        {
            r = yr / yi;
            d = yi + r * yr;
            cr = (r * xr + xi) / d;
            ci = (r * xi - xr) / d;
        }
    }
}
