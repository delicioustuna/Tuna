using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Tuna.LinearAlgebra;

// LAPACK dgebal / dgehrd / dorghr / dgebak 相当
/// <summary>
/// 一般の正方行列の平衡化とヘッセンベルグ化
/// </summary>
internal static class HessenbergReduction
{
    // LAPACK dgebal 'B'
    // 孤立した固有値を左上と右下へ移し、残りの [ilo, ihi] の行と列のノルムを揃える。
    /// <summary>
    /// 置換と2の冪の対角相似変換で行列を平衡化する
    /// </summary>
    /// <param name="n">行列の次数。</param>
    /// <param name="a">係数行列。</param>
    /// <param name="lda">係数行列の列の間隔。</param>
    /// <param name="scale">長さ n。置換の相手（<c>i &lt; ilo</c> または <c>i &gt; ihi</c>）または倍率（<c>ilo ≤ i ≤ ihi</c>）。</param>
    /// <param name="ilo">平衡化後の処理範囲の先頭。</param>
    /// <param name="ihi">平衡化後の処理範囲の末尾。</param>
    internal static void Balance(int n, Span<double> a, int lda, Span<double> scale, out int ilo, out int ihi)
    {
        int k = 0;
        int l = n - 1;
        if (n == 0)
        {
            ilo = 0;
            ihi = -1;
            return;
        }

        // 行のうち非対角成分がすべて0のものを下へ寄せる
        bool noconv = true;
        while (noconv && l >= 0)
        {
            noconv = false;
            for (int i = l; i >= 0; i--)
            {
                bool canSwap = true;
                for (int j = 0; j <= l; j++)
                {
                    if (i != j && a[i + j * lda] != 0)
                    {
                        canSwap = false;
                        break;
                    }
                }

                if (canSwap)
                {
                    scale[l] = i;
                    if (i != l)
                    {
                        Level1Kernels.Swap(a.Slice(i * lda, l + 1), a.Slice(l * lda, l + 1));
                        SwapRows(a, lda, i, l, k, n);
                    }

                    noconv = true;
                    if (l == 0)
                    {
                        // 残った1×1 の部分の倍率は1（0 始まりの添字では置換先 0 と区別できないため明示する）
                        scale[0] = 1;
                        ilo = 0;
                        ihi = 0;
                        return;
                    }

                    l--;
                }
            }
        }

        // 列のうち非対角成分がすべて0のものを左へ寄せる
        noconv = true;
        while (noconv)
        {
            noconv = false;
            for (int j = k; j <= l; j++)
            {
                bool canSwap = true;
                for (int i = k; i <= l; i++)
                {
                    if (i != j && a[i + j * lda] != 0)
                    {
                        canSwap = false;
                        break;
                    }
                }

                if (canSwap)
                {
                    scale[k] = j;
                    if (j != k)
                    {
                        Level1Kernels.Swap(a.Slice(j * lda, l + 1), a.Slice(k * lda, l + 1));
                        SwapRows(a, lda, j, k, k, n);
                    }

                    noconv = true;
                    k++;
                }
            }
        }

        for (int i = k; i <= l; i++)
        {
            scale[i] = 1;
        }

        ilo = k;
        ihi = l;
        if (l > k)
        {
            Scale(n, a, lda, scale, k, l);
        }
    }

    // dgebal は行と列を1つずつその場で拡大縮小し、そのたびに行のノルム（列優先の配置では飛び飛びの読み出し）を計算し直す。
    // ここでは元の A と、その行を連続に並べた転置 Aᵀ を保ち、現在の倍率 d で重み付けたノルム
    // （‖A[:, i] ⊘ d‖·dᵢ、‖A[i, :] ⊙ d‖ / dᵢ）を連続した読み出しだけで求め、最後に1回だけ A を拡大縮小する。
    // 倍率は2の冪なので、ノルムも最終結果も逐次に拡大縮小した場合と同じ値になる
    // scale[k..l] に D の対角を掛け込む。
    /// <summary>
    /// 2の冪の対角相似変換 <c>D⁻¹AD</c> で <c>[k, l]</c> の行と列のノルムを揃える（dgebal の後半）
    /// </summary>
    private static void Scale(int n, Span<double> a, int lda, Span<double> scale, int k, int l)
    {
        const double radix = 2.0;
        double sfmin1 = NumericConstants.MinNormal / (NumericConstants.MachineEpsilon * radix);
        double sfmax1 = 1.0 / sfmin1;
        double sfmin2 = sfmin1 * radix;
        double sfmax2 = 1.0 / sfmin2;

        // at の列 i は A の行 i の列 [k, n)（長さ n - k）
        int w = n - k;
        int count = l - k + 1;
        double[] rented = WorkArrayPool<double>.Rent(checked(w * count + 2 * n));
        try
        {
            var at = rented.AsSpan(0, w * count);
            var d = rented.AsSpan(w * count, n);
            var inv = rented.AsSpan(w * count + n, n);
            d.Fill(1);
            inv.Fill(1);
            Level1Kernels.Transpose(count, w, a[(k + k * lda)..], lda, at, w);

            bool noconv = true;
            int sweeps = 0;
            while (noconv && sweeps++ < 100)
            {
                noconv = false;
                for (int i = k; i <= l; i++)
                {
                    // 列 i: 行 [k, l] のノルムと行 [0, l] の最大値。行 i: 列 [k, l] のノルムと列 [k, n) の最大値
                    var col = a.Slice(i * lda, l + 1);
                    WeightedNormMax(col[k..], inv[k..(l + 1)], out double c, out double ca);
                    ca = Math.Max(ca, WeightedMax(col[..k], inv[..k]));
                    var row = at.Slice((i - k) * w, w);
                    WeightedNormMax(row[..count], d[k..(l + 1)], out double r, out double ra);
                    ra = Math.Max(ra, WeightedMax(row[count..], d[(l + 1)..]));
                    c *= d[i];
                    ca *= d[i];
                    r *= inv[i];
                    ra *= inv[i];
                    if (c == 0 || r == 0)
                    {
                        continue;
                    }

                    if (double.IsNaN(c + ca + r + ra))
                    {
                        // dgebal と同じく、ここまでの倍率を適用して終える
                        Apply(n, a, lda, d, inv, k, l);
                        return;
                    }

                    double g = r / radix;
                    double f = 1;
                    double s = c + r;
                    while (c < g && Math.Max(f, Math.Max(c, ca)) < sfmax2 && Math.Min(r, Math.Min(g, ra)) > sfmin2)
                    {
                        f *= radix;
                        c *= radix;
                        ca *= radix;
                        r /= radix;
                        g /= radix;
                        ra /= radix;
                    }

                    g = c / radix;
                    while (g >= r && Math.Max(r, ra) < sfmax2 && Math.Min(Math.Min(f, c), Math.Min(g, ca)) > sfmin2)
                    {
                        f /= radix;
                        c /= radix;
                        g /= radix;
                        ca /= radix;
                        r *= radix;
                        ra *= radix;
                    }

                    if (c + r >= 0.95 * s)
                    {
                        continue;
                    }

                    if (f < 1 && scale[i] < 1 && f * scale[i] <= sfmin1)
                    {
                        continue;
                    }

                    if (f > 1 && scale[i] > 1 && scale[i] >= sfmax1 / f)
                    {
                        continue;
                    }

                    scale[i] *= f;
                    d[i] *= f;
                    inv[i] /= f;
                    noconv = true;
                }
            }

            Apply(n, a, lda, d, inv, k, l);
        }
        finally
        {
            WorkArrayPool<double>.Return(rented);
        }
    }

    // dgebal と同じ範囲: 行 i（k ≤ i ≤ l）は列 [k, n)、列 j（k ≤ j ≤ l）は行 [0, l]。
    /// <summary>
    /// <c>A ← D⁻¹AD</c>
    /// </summary>
    private static void Apply(int n, Span<double> a, int lda, ReadOnlySpan<double> d, ReadOnlySpan<double> inv, int k, int l)
    {
        for (int j = k; j < n; j++)
        {
            Level1Kernels.Multiply(a.Slice(k + j * lda, l - k + 1), inv.Slice(k, l - k + 1));

            if (j <= l && d[j] != 1)
            {
                Level1Kernels.Scale(d[j], a.Slice(j * lda, l + 1));
            }
        }
    }

    /// <summary>
    /// <c>x ⊙ w</c> の2ノルムと最大絶対値
    /// </summary>
    private static void WeightedNormMax(ReadOnlySpan<double> x, ReadOnlySpan<double> w, out double norm, out double max)
    {
        double ss = 0;
        double mx = 0;
        int i = 0;
        if (Vector512.IsHardwareAccelerated && x.Length >= 8)
        {
            var vs = Vector512<double>.Zero;
            var vm = Vector512<double>.Zero;
            ref double px = ref MemoryMarshal.GetReference(x);
            ref double pw = ref MemoryMarshal.GetReference(w);
            for (; i <= x.Length - 8; i += 8)
            {
                var t = Vector512.LoadUnsafe(ref px, (nuint)i) * Vector512.LoadUnsafe(ref pw, (nuint)i);
                vs = Vector512.FusedMultiplyAdd(t, t, vs);
                vm = Vector512.Max(vm, Vector512.Abs(t));
            }

            ss = Vector512.Sum(vs);
            for (int q = 0; q < 8; q++)
            {
                mx = Math.Max(mx, vm.GetElement(q));
            }
        }

        for (; i < x.Length; i++)
        {
            double t = x[i] * w[i];
            ss += t * t;
            mx = Math.Max(mx, Math.Abs(t));
        }

        max = mx;
        if ((ss > 1e-150 && ss < 1e150) || mx == 0 || !double.IsFinite(mx))
        {
            norm = mx == 0 || double.IsFinite(mx) ? Math.Sqrt(ss) : mx;
            return;
        }

        // 平方和がアンダーフロー・オーバーフローしうるので、最大絶対値で正規化して計算し直す
        double scaleInv = 1.0 / mx;
        double sum = 0;
        for (i = 0; i < x.Length; i++)
        {
            double t = x[i] * w[i] * scaleInv;
            sum += t * t;
        }

        norm = mx * Math.Sqrt(sum);
    }

    /// <summary>
    /// <c>x ⊙ w</c> の最大絶対値
    /// </summary>
    private static double WeightedMax(ReadOnlySpan<double> x, ReadOnlySpan<double> w)
    {
        double mx = 0;
        for (int i = 0; i < x.Length; i++)
        {
            mx = Math.Max(mx, Math.Abs(x[i] * w[i]));
        }

        return mx;
    }

    private static void SwapRows(Span<double> a, int lda, int r1, int r2, int colStart, int colEnd)
    {
        for (int j = colStart; j < colEnd; j++)
        {
            (a[r1 + j * lda], a[r2 + j * lda]) = (a[r2 + j * lda], a[r1 + j * lda]);
        }
    }

    // LAPACK dgebak 'B','R'
    /// <summary>
    /// 平衡化の逆変換を固有ベクトル（<c>V</c> の行）に作用させる
    /// </summary>
    internal static void BackTransform(int n, int columns, ReadOnlySpan<double> scale, int ilo, int ihi, Span<double> v, int ldv)
    {
        for (int i = ilo; i <= ihi; i++)
        {
            double s = scale[i];
            if (s != 1)
            {
                for (int j = 0; j < columns; j++)
                {
                    v[i + j * ldv] *= s;
                }
            }
        }

        for (int ii = 0; ii < n; ii++)
        {
            int i = ii;
            if (i >= ilo && i <= ihi)
            {
                continue;
            }

            if (i < ilo)
            {
                i = ilo - 1 - ii;
            }

            int k = (int)scale[i];
            if (k == i)
            {
                continue;
            }

            SwapRows(v, ldv, i, k, 0, columns);
        }
    }

    // LAPACK dgehrd の既定と同じ 32
    /// <summary>
    /// ブロック幅
    /// </summary>
    internal const int BlockSize = 32;

    // LAPACK の既定と同じ 128
    /// <summary>
    /// 残りがこれ以下になったら非ブロック版で処理する
    /// </summary>
    internal const int Crossover = 128;

    // 先頭から BlockSize 列ずつ反射を作り（dlahr2）、右からの更新は A ← A - YVᵀ、
    // 左からの更新はブロック反射 (I - VTVᵀ)ᵀ の行列積で行う。最後の Crossover 列は非ブロック版（dgehd2）
    // LAPACK dgehrd
    // 反射は A[i+2:ihi+1, i] に、係数は tau に格納。
    /// <summary>
    /// <c>[ilo, ihi]</c> の部分をヘッセンベルグ化する
    /// </summary>
    internal static unsafe void Reduce(int n, int ilo, int ihi, Span<double> a, int lda, Span<double> tau)
    {
        int nb = BlockSize;
        int rows = ihi + 1;
        int p = ilo;
        if (ihi - 1 - Crossover >= ilo)
        {
            int threads = Parallelism.PlannedDegree((double)rows * (ihi - ilo), MatrixVector.Grain);
            var team = new ColumnTeam(threads, rows);
            double[] buffer = WorkArrayPool<double>.Rent(checked(2 * rows * nb + nb * nb));
            try
            {
                fixed (double* pa = a)
                fixed (double* pb = buffer)
                {
                    var y = buffer.AsSpan(0, rows * nb);
                    var v = buffer.AsSpan(rows * nb, rows * nb);
                    var t = buffer.AsSpan(2 * rows * nb, nb * nb);
                    for (; p <= ihi - 1 - Crossover; p += nb)
                    {
                        // 列 p 以降を分担する（列 p から数える）
                        team.Partition(n - p, nb);
                        ReducePanel(ihi, p, nb, a, lda, tau[p..], t, nb, y, rows, team, pa);

                        // A[0:p+1, p+1:p+nb] ← A - Y V₁ᵀ（V₁ は単位下三角）
                        for (int j = 0; j < nb - 1; j++)
                        {
                            // (Y V₁ᵀ)[:, j] = Y[:, j] + Σ_{q<j} Y[:, q] V₁[j, q]、V₁[j, q] = A[p+1+j, p+q]
                            var col = a.Slice((p + 1 + j) * lda, p + 1);
                            Level1Kernels.Axpy(-1.0, y.Slice(j * rows, p + 1), col);
                            for (int q = 0; q < j; q++)
                            {
                                Level1Kernels.Axpy(-a[p + 1 + j + (p + q) * lda], y.Slice(q * rows, p + 1), col);
                            }
                        }

                        int vr = ihi - p;
                        Householder.ExtractV(vr, nb, a[(p + 1 + p * lda)..], lda, v, vr);

                        // A[0:ihi+1, p+nb:ihi+1] ← A - Y V₂ᵀ（V₂ は p+nb 行目以降。先頭の1を一時的に置く）
                        int last = p + nb + (p + nb - 1) * lda;
                        double ei = a[last];
                        a[last] = 1;
                        // 各スレッドが担当の列について、右からの更新と左からの更新を続けて行う
                        int p0 = p;
                        int ldaLocal = lda;
                        double* paa = pa;
                        double* py = pb;
                        double* pv = pb + rows * nb;
                        double* pt = pb + 2 * rows * nb;
                        team.For(th =>
                        {
                            var (lo, hi) = team.Range(th, nb);
                            if (hi <= lo)
                            {
                                return;
                            }

                            int right = Math.Min(hi, ihi - p0 + 1) - lo;
                            if (right > 0)
                            {
                                Gemm.Multiply(
                                    false, true, rows, right, nb, -1.0, new ReadOnlySpan<double>(py, rows * nb), rows,
                                    new ReadOnlySpan<double>(paa + p0 + lo + (nint)p0 * ldaLocal, (nb - 1) * ldaLocal + right), ldaLocal,
                                    new Span<double>(paa + (nint)(p0 + lo) * ldaLocal, (right - 1) * ldaLocal + rows), ldaLocal);
                            }

                            Householder.ApplyBlockLeft(
                                true, vr, hi - lo, nb, new ReadOnlySpan<double>(pv, vr * nb), vr, new ReadOnlySpan<double>(pt, nb * nb), nb,
                                new Span<double>(paa + p0 + 1 + (nint)(p0 + lo) * ldaLocal, (hi - lo - 1) * ldaLocal + vr), ldaLocal);
                        });
                        a[last] = ei;
                    }
                }
            }
            finally
            {
                WorkArrayPool<double>.Return(buffer);
            }
        }

        double[] rented = WorkArrayPool<double>.Rent(n);
        var work = rented.AsSpan(0, n);
        for (int i = p; i < ihi - 1; i++)
        {
            int sub = i + 1 + i * lda;
            int len = ihi - i;
            double alpha = a[sub];
            double tv = Householder.Generate(ref alpha, a.Slice(sub + 1, len - 1));
            tau[i] = tv;
            if (tv != 0)
            {
                a[sub] = 1;
                var v = a.Slice(sub, len);

                // A[0:ihi+1, i+1:ihi+1] ← A H
                Householder.ApplyRight(v, tv, ihi + 1, len, a[((i + 1) * lda)..], lda, work);

                // A[i+1:ihi+1, i+1:n] ← H A
                Householder.ApplyLeft(v, tv, len, n - i - 1, a[(i + 1 + (i + 1) * lda)..], lda);
            }

            a[sub] = alpha;
        }

        WorkArrayPool<double>.Return(rented);
    }

    // 残りの列との行列ベクトル積は team の列区間ごとに計算する（pa は固定済みの a）
    // LAPACK dlahr2
    // 各列は作る直前にそれまでの反射で更新する。
    // H = I - VTVᵀ の T（nb × nb、上三角）と Y = AVT（(ihi+1) × nb）を返す。
    // A[p+nb, p+nb-1] 以外の副対角は確定した値で、反射ベクトルは副対角より下に格納する。
    /// <summary>
    /// 列 <c>[p, p+nb)</c> の反射を作る
    /// </summary>
    private static unsafe void ReducePanel(
        int ihi, int p, int nb, Span<double> a, int lda, Span<double> tau, Span<double> t, int ldt, Span<double> y, int ldy, ColumnTeam team, double* pa)
    {
        int r = ihi - p;
        var w = t.Slice((nb - 1) * ldt, nb);
        double ei = 0;
        t[..(nb * ldt)].Clear();
        for (int ii = 0; ii < nb; ii++)
        {
            int c = p + ii;
            var col = a.Slice(p + 1 + c * lda, r);
            if (ii > 0)
            {
                // A[p+1:ihi+1, c] ← A - Y[p+1:, 0:ii] A[p+ii, p:p+ii]ᵀ（右からの更新）
                for (int j = 0; j < ii; j++)
                {
                    Level1Kernels.Axpy(-a[p + ii + (p + j) * lda], y.Slice(p + 1 + j * ldy, r), col);
                }

                // 左から (I - VTVᵀ)ᵀ を掛ける。b₁ = col[0:ii]、b₂ = col[ii:]
                var b1 = col[..ii];
                var b2 = col[ii..];
                var wi = w[..ii];

                // w = V₁ᵀb₁ + V₂ᵀb₂（V₁ は単位下三角、V₁[q, j] = A[p+1+q, p+j]）
                for (int j = 0; j < ii; j++)
                {
                    double sum = b1[j];
                    for (int q = j + 1; q < ii; q++)
                    {
                        sum += a[p + 1 + q + (p + j) * lda] * b1[q];
                    }

                    wi[j] = sum + Level1Kernels.Dot(a.Slice(p + 1 + ii + (p + j) * lda, r - ii), b2);
                }

                // w = Tᵀw（下から更新）
                for (int j = ii - 1; j >= 0; j--)
                {
                    double sum = 0;
                    for (int q = 0; q <= j; q++)
                    {
                        sum += t[q + j * ldt] * wi[q];
                    }

                    wi[j] = sum;
                }

                // b₂ ← b₂ - V₂w
                for (int j = 0; j < ii; j++)
                {
                    Level1Kernels.Axpy(-wi[j], a.Slice(p + 1 + ii + (p + j) * lda, r - ii), b2);
                }

                // b₁ ← b₁ - V₁w（下から更新）
                for (int q = ii - 1; q >= 0; q--)
                {
                    double sum = wi[q];
                    for (int j = 0; j < q; j++)
                    {
                        sum += a[p + 1 + q + (p + j) * lda] * wi[j];
                    }

                    b1[q] -= sum;
                }

                a[p + ii + (c - 1) * lda] = ei;
            }

            // A[p+ii+2:ihi+1, c] を消去する反射
            int sub = p + ii + 1 + c * lda;
            double alpha = a[sub];
            double tv = Householder.Generate(ref alpha, a.Slice(sub + 1, r - ii - 1));
            tau[ii] = tv;
            ei = alpha;
            a[sub] = 1;
            var v = a.Slice(sub, r - ii);

            // Y[p+1:, ii] = τ (A[p+1:, c+1:ihi+1] v - Y[p+1:, 0:ii] (V₂ᵀv))
            var yi = y.Slice(p + 1 + ii * ldy, r);
            fixed (double* py = yi)
            {
                team.Multiply(pa + p + 1 + (nint)p * lda, lda, r, ii + 1, ihi - p + 1, pa + sub, py);
            }

            var ti = t.Slice(ii * ldt, ii);
            for (int j = 0; j < ii; j++)
            {
                ti[j] = Level1Kernels.Dot(a.Slice(p + 1 + ii + (p + j) * lda, r - ii), v);
            }

            for (int j = 0; j < ii; j++)
            {
                Level1Kernels.Axpy(-ti[j], y.Slice(p + 1 + j * ldy, r), yi);
            }

            Level1Kernels.Scale(tv, yi);

            // T[0:ii, ii] = -τ T[0:ii, 0:ii] (V₂ᵀv)、T[ii, ii] = τ
            for (int j = 0; j < ii; j++)
            {
                double sum = 0;
                for (int q = j; q < ii; q++)
                {
                    sum += t[j + q * ldt] * ti[q];
                }

                ti[j] = -tv * sum;
            }

            t[ii + ii * ldt] = tv;
        }

        a[p + nb + (p + nb - 1) * lda] = ei;

        // Y[0:p+1, :] = (A[0:p+1, p+1:p+nb+1] V₁ + A[0:p+1, p+nb+1:ihi+1] V₂) T
        int top = p + 1;
        for (int j = 0; j < nb; j++)
        {
            // (A₁V₁)[:, j] = A[:, p+1+j] + Σ_{q>j} A[:, p+1+q] V₁[q, j]
            var yj = y.Slice(j * ldy, top);
            a.Slice((p + 1 + j) * lda, top).CopyTo(yj);
            for (int q = j + 1; q < nb; q++)
            {
                Level1Kernels.Axpy(a[p + 1 + q + (p + j) * lda], a.Slice((p + 1 + q) * lda, top), yj);
            }
        }

        if (ihi > p + nb)
        {
            Gemm.Multiply(false, false, top, nb, ihi - p - nb, 1.0, a[((p + nb + 1) * lda)..], lda, a[(p + nb + 1 + p * lda)..], lda, y, ldy);
        }

        // Y[0:p+1, :] ← Y T（右の列から）
        for (int j = nb - 1; j >= 0; j--)
        {
            var yj = y.Slice(j * ldy, top);
            Level1Kernels.Scale(t[j + j * ldt], yj);
            for (int q = 0; q < j; q++)
            {
                Level1Kernels.Axpy(t[q + j * ldt], y.Slice(q * ldy, top), yj);
            }
        }
    }

    // LAPACK dorghr
    /// <summary>
    /// <see cref="Reduce"/> の反射から直交行列 <c>Q</c>（<c>n × n</c>）を作る
    /// </summary>
    internal static void FormQ(int n, int ilo, int ihi, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> tau, Span<double> q, int ldq)
    {
        for (int j = 0; j < n; j++)
        {
            var qj = q.Slice(j * ldq, n);
            qj.Clear();
            qj[j] = 1;
        }

        int nh = ihi - ilo;
        if (nh <= 0)
        {
            return;
        }

        QrKernels.FormQ(nh, nh, nh - 1, a[(ilo + 1 + ilo * lda)..], lda, tau[ilo..], q[(ilo + 1 + (ilo + 1) * ldq)..], ldq);
    }
}
