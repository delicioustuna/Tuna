namespace Tuna.LinearAlgebra;

// Householder 変換 H = I - τ v v^T および compact WY ブロック表現 I - V T V^T。
internal static class Householder
{
    // 2^1021: |α - β| <= 2|β| がオーバーフローまたは逆数が非正規化数となりうる境界。
    private const double LargeBeta = 2.2471164185778949e307;

    // LAPACK dlarfg 相当。H*[α; x] = [β; 0] の H = I - τ v v^T を生成し、alpha を β、x を v[1:] で上書きして τ を返す。
    internal static double Generate(ref double alpha, Span<double> x)
    {
        if (x.Length == 0)
        {
            return 0;
        }

        double xnorm = Level1Kernels.Norm2(x);
        if (xnorm == 0)
        {
            return 0;
        }

        double beta = -Math.CopySign(Hypot(alpha, xnorm), alpha);
        double smallNum = NumericConstants.MinNormal / NumericConstants.MachineEpsilon;
        int rescaled = 0;
        if (Math.Abs(beta) < smallNum)
        {
            // β が非正規数の範囲にあると 1/(α-β) が不正確になるので、拡大してから計算し直す
            double rsafmn = 1.0 / smallNum;
            do
            {
                rescaled++;
                Level1Kernels.Scale(rsafmn, x);
                beta *= rsafmn;
                alpha *= rsafmn;
            }
            while (Math.Abs(beta) < smallNum && rescaled < 20);

            xnorm = Level1Kernels.Norm2(x);
            beta = -Math.CopySign(Hypot(alpha, xnorm), alpha);
        }

        double tau;
        if (Math.Abs(beta) < LargeBeta)
        {
            tau = (beta - alpha) / beta;
            Level1Kernels.Scale(1.0 / (alpha - beta), x);
        }
        else
        {
            // |α - β| = |α| + |β| は |β| の2倍まで大きくなり、オーバーフローするか逆数が非正規数になる。
            // α/β ≤ 0 なので τ = 1 - α/β は桁落ちしない。v は4分の1にした差で割る（2の冪の倍率は丸め誤差を生まない）
            tau = 1 - alpha / beta;
            double quarterDifference = 0.25 * alpha - 0.25 * beta;
            Level1Kernels.Scale(0.25, x);
            Level1Kernels.Scale(1.0 / quarterDifference, x);
        }

        for (int j = 0; j < rescaled; j++)
        {
            beta *= smallNum;
        }

        alpha = beta;
        return tau;
    }

    internal static double Hypot(double a, double b)
    {
        a = Math.Abs(a);
        b = Math.Abs(b);
        double w = Math.Max(a, b);
        double z = Math.Min(a, b);
        if (z == 0 || double.IsInfinity(w))
        {
            return w;
        }

        double r = z / w;
        return w * Math.Sqrt(1 + r * r);
    }

    // C <- (I - τ v v^T) C（v は先頭要素 1 を明示格納した長さ m）。
    internal static void ApplyLeft(ReadOnlySpan<double> v, double tau, int m, int n, Span<double> c, int ldc)
    {
        if (tau == 0)
        {
            return;
        }

        v = v[..m];
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            var cj = c[(j * ldc)..];
            Level1Kernels.Dot4(m, cj, ldc, v, out double w0, out double w1, out double w2, out double w3);
            Level1Kernels.AddToColumns4(m, -tau * w0, -tau * w1, -tau * w2, -tau * w3, v, cj, ldc);
        }

        for (; j < n; j++)
        {
            var cj = c.Slice(j * ldc, m);
            double w = tau * Level1Kernels.Dot(v, cj);
            Level1Kernels.Axpy(-w, v, cj);
        }
    }

    // C <- C (I - τ v v^T)（v は長さ n、work は長さ m 以上）。
    internal static void ApplyRight(ReadOnlySpan<double> v, double tau, int m, int n, Span<double> c, int ldc, Span<double> work)
    {
        if (tau == 0)
        {
            return;
        }

        // w = C v
        var w = work[..m];
        w.Clear();
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            Level1Kernels.AddColumns4(m, c[(j * ldc)..], ldc, v[j], v[j + 1], v[j + 2], v[j + 3], w);
        }

        for (; j < n; j++)
        {
            Level1Kernels.Axpy(v[j], c.Slice(j * ldc, m), w);
        }

        // C -= τ w vᵀ
        for (j = 0; j + 4 <= n; j += 4)
        {
            Level1Kernels.AddToColumns4(m, -tau * v[j], -tau * v[j + 1], -tau * v[j + 2], -tau * v[j + 3], w, c[(j * ldc)..], ldc);
        }

        for (; j < n; j++)
        {
            Level1Kernels.Axpy(-tau * v[j], w, c.Slice(j * ldc, m));
        }
    }

    // LAPACK dlarft（前進・列方向）相当。単位下台形 V（m×k）から H_1...H_k = I - V T V^T の上三角 T（k×k）を構築する。
    internal static void FormT(int m, int k, ReadOnlySpan<double> v, int ldv, ReadOnlySpan<double> tau, Span<double> t, int ldt)
    {
        // G = VᵀV を行列積で求める（V は対角より上が0、対角が1の明示的な形なので、列全体の内積でよい）。
        // T の上三角に G の上三角を置き、列ごとに T[0:i, i] = -τᵢ T[0:i, 0:i] G[0:i, i] とする
        for (int i = 0; i < k; i++)
        {
            t.Slice(i * ldt, k).Clear();
        }

        if (k > 1)
        {
            Gemm.Multiply(true, false, k, k, m, 1.0, v, ldv, v, ldv, t, ldt);
        }

        for (int i = 0; i < k; i++)
        {
            double ti = tau[i];
            var tcol = t.Slice(i * ldt, k);
            if (ti == 0)
            {
                tcol[..i].Clear();
            }
            else
            {
                // T[0:i, i] = T[0:i, 0:i] (-τᵢ G[0:i, i])（上三角との積を上から順に）
                for (int p = 0; p < i; p++)
                {
                    tcol[p] *= -ti;
                }

                for (int p = 0; p < i; p++)
                {
                    double sum = 0;
                    for (int q = p; q < i; q++)
                    {
                        sum += t[p + q * ldt] * tcol[q];
                    }

                    tcol[p] = sum;
                }
            }

            tcol[i] = ti;
        }

        // 下三角（G の残り）を0に戻す
        for (int i = 0; i < k; i++)
        {
            t.Slice(i + 1 + i * ldt, k - i - 1).Clear();
        }
    }

    // LAPACK dlarfb（左・前進・列方向）相当。C <- (I - V op(T) V^T) C（V: m×k, C: m×n）。
    internal static void ApplyBlockLeft(
        bool transpose, int m, int n, int k,
        ReadOnlySpan<double> v, int ldv, ReadOnlySpan<double> t, int ldt, Span<double> c, int ldc)
    {
        if (m <= 0 || n <= 0 || k <= 0)
        {
            return;
        }

        double[] rented = WorkArrayPool<double>.Rent(checked(2 * k * n));
        var w = rented.AsSpan(0, k * n);
        var tw = rented.AsSpan(k * n, k * n);
        w.Clear();
        tw.Clear();

        // W = Vᵀ C（k × n）
        Gemm.Multiply(true, false, k, n, m, 1.0, v, ldv, c, ldc, w, k);

        // TW = op(T) W（T の下三角は0なので行列積でよい）
        Gemm.Multiply(transpose, false, k, n, k, 1.0, t, ldt, w, k, tw, k);

        // C -= V TW
        Gemm.Multiply(false, false, m, n, k, -1.0, v, ldv, tw, k, c, ldc);
        WorkArrayPool<double>.Return(rented);
    }

    // C <- C (I - V op(T) V^T)（V: n×k, C: m×n）。
    internal static void ApplyBlockRight(
        bool transpose, int m, int n, int k,
        ReadOnlySpan<double> v, int ldv, ReadOnlySpan<double> t, int ldt, Span<double> c, int ldc)
    {
        if (m <= 0 || n <= 0 || k <= 0)
        {
            return;
        }

        double[] rented = WorkArrayPool<double>.Rent(checked(m * k));
        var w = rented.AsSpan(0, m * k);
        w.Clear();

        // W = C V（m × k）
        Gemm.Multiply(false, false, m, k, n, 1.0, c, ldc, v, ldv, w, m);

        // W = W op(T)  ⇔  Wᵀ = op(T)ᵀ Wᵀ
        MultiplyTriangularRight(transpose, m, k, t, ldt, w, m);

        // C -= W Vᵀ
        Gemm.Multiply(false, true, m, n, k, -1.0, w, m, v, ldv, c, ldc);
        WorkArrayPool<double>.Return(rented);
    }

    // W <- W * op(T)（上三角 T: k×k, W: m×k）。
    private static void MultiplyTriangularRight(bool transpose, int m, int k, ReadOnlySpan<double> t, int ldt, Span<double> w, int ldw)
    {
        if (!transpose)
        {
            // (WT)[:, j] = Σ_{p≤j} W[:, p] T[p, j]。右の列から更新する
            for (int j = k - 1; j >= 0; j--)
            {
                var wj = w.Slice(j * ldw, m);
                Level1Kernels.Scale(t[j + j * ldt], wj);
                for (int p = 0; p < j; p++)
                {
                    Level1Kernels.Axpy(t[p + j * ldt], w.Slice(p * ldw, m), wj);
                }
            }
        }
        else
        {
            // (WTᵀ)[:, j] = Σ_{p≥j} W[:, p] T[j, p]。左の列から更新する
            for (int j = 0; j < k; j++)
            {
                var wj = w.Slice(j * ldw, m);
                Level1Kernels.Scale(t[j + j * ldt], wj);
                for (int p = j + 1; p < k; p++)
                {
                    Level1Kernels.Axpy(t[j + p * ldt], w.Slice(p * ldw, m), wj);
                }
            }
        }
    }

    // 対角下の反射ベクトルから単位下台形 V（m×k）を明示的に構築する。
    internal static void ExtractV(int m, int k, ReadOnlySpan<double> a, int lda, Span<double> v, int ldv)
    {
        for (int j = 0; j < k; j++)
        {
            var vj = v.Slice(j * ldv, m);
            vj[..Math.Min(j, m)].Clear();
            if (j < m)
            {
                vj[j] = 1;
                a.Slice(j + 1 + j * lda, m - j - 1).CopyTo(vj[(j + 1)..]);
            }
        }
    }
}
