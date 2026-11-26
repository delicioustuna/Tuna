using System.Numerics;

namespace Tuna.LinearAlgebra;

// 複素行列積 C <- op(A)*op(B) の 4M 法実装（実部・虚部を実行列に展開して倍精度 Gemm へ委譲）。
// 実部和を別々に累積するため打ち消し前にオーバーフローしうる。非有限結果が生じた列は呼び出し元で直接複素積へフォールバックする。
internal static class GemmComplex
{
    // 作業配列 1 つあたりの要素数目安（倍精度で 8 MB）。
    private const int Budget = 1 << 20;

    // 4M 法の分割コストが見合う規模か判定する（小規模は呼び出し元の列ごと GEMV に委譲）。
    internal static bool Prefers(int m, int n, int k) =>
        m >= 8 && n >= 8 && k >= 8 && (double)m * n * k >= 128 * 32 * 32 && m <= MaxRows;

    // 作業配列（最大 256*m 要素）が配列上限を超えない最大行数。
    private const int MaxRows = 1 << 22;

    // C <- op(A)*op(B) を上書き計算し、全要素が有限なら true を返す。
    internal static bool Multiply(
        bool transA, bool transB, int m, int n, int k,
        ReadOnlySpan<Complex> a, int lda, ReadOnlySpan<Complex> b, int ldb, Span<Complex> c, int ldc)
    {
        int kb = Math.Min(k, Math.Max(128, Math.Min(Budget / (2 * m), Budget / 256)));
        int nb = Math.Min(n, Math.Max(64, Math.Min(Budget / (2 * m), Budget / (4 * kb))));
        double[] aw = WorkArrayPool<double>.Rent(checked(2 * m * kb));
        double[] bw = WorkArrayPool<double>.Rent(checked(4 * kb * nb));
        double[] cw = WorkArrayPool<double>.Rent(checked(2 * m * nb));
        bool finite = true;
        try
        {
            for (int j0 = 0; j0 < n; j0 += nb)
            {
                int w = Math.Min(nb, n - j0);
                var block = cw.AsSpan(0, 2 * m * w);
                block.Clear();
                for (int p0 = 0; p0 < k; p0 += kb)
                {
                    int d = Math.Min(kb, k - p0);
                    SplitA(transA, a, lda, m, p0, d, aw);
                    SplitB(transB, b, ldb, p0, j0, d, w, bw);

                    // [Re C | Im C] (m × 2w) += [Re A | Im A] (m × 2d) · [[Re B, Im B], [-Im B, Re B]] (2d × 2w)
                    Gemm.Multiply(transA, transB, m, 2 * w, 2 * d, 1.0,
                        aw, transA ? 2 * d : m, bw, transB ? 2 * w : 2 * d, block, m);
                }

                for (int jj = 0; jj < w; jj++)
                {
                    var re = block.Slice(jj * m, m);
                    var im = block.Slice((w + jj) * m, m);
                    var dst = c.Slice((j0 + jj) * ldc, m);
                    for (int i = 0; i < m; i++)
                    {
                        dst[i] = new Complex(re[i], im[i]);
                        finite &= double.IsFinite(re[i]) & double.IsFinite(im[i]);
                    }
                }
            }

            return finite;
        }
        finally
        {
            WorkArrayPool<double>.Return(aw);
            WorkArrayPool<double>.Return(bw);
            WorkArrayPool<double>.Return(cw);
        }
    }

    // op(A)[:, p0..p0+d] を [Re A | Im A] へ展開する（転置時は格納方向を維持した 2d × m として書き込み、転置は Gemm に任せる）。
    private static void SplitA(bool transA, ReadOnlySpan<Complex> a, int lda, int m, int p0, int d, Span<double> dest)
    {
        if (!transA)
        {
            for (int q = 0; q < d; q++)
            {
                var src = a.Slice((p0 + q) * lda, m);
                var re = dest.Slice(q * m, m);
                var im = dest.Slice((d + q) * m, m);
                for (int i = 0; i < m; i++)
                {
                    re[i] = src[i].Real;
                    im[i] = src[i].Imaginary;
                }
            }

            return;
        }

        // op(A)[i, p] = a[p + i·lda]
        for (int i = 0; i < m; i++)
        {
            var src = a.Slice(p0 + i * lda, d);
            var column = dest.Slice(i * 2 * d, 2 * d);
            for (int q = 0; q < d; q++)
            {
                column[q] = src[q].Real;
                column[d + q] = src[q].Imaginary;
            }
        }
    }

    // op(B)[p0..p0+d, j0..j0+w] を [[Re B, Im B], [-Im B, Re B]] へ展開する（転置時は 2w × 2d で出力）。
    private static void SplitB(bool transB, ReadOnlySpan<Complex> b, int ldb, int p0, int j0, int d, int w, Span<double> dest)
    {
        if (!transB)
        {
            int ld = 2 * d;
            for (int jj = 0; jj < w; jj++)
            {
                var src = b.Slice(p0 + (j0 + jj) * ldb, d);
                var left = dest.Slice(jj * ld, ld);
                var right = dest.Slice((w + jj) * ld, ld);
                for (int q = 0; q < d; q++)
                {
                    double re = src[q].Real, im = src[q].Imaginary;
                    left[q] = re;
                    left[d + q] = -im;
                    right[q] = im;
                    right[d + q] = re;
                }
            }

            return;
        }

        // op(B)[p, j] = b[j + p·ldb]。転置した作業配列の第 q 列（q < d）は [Re B[q, :], Im B[q, :]]、第 d + q 列は [-Im B[q, :], Re B[q, :]]
        int ldt = 2 * w;
        for (int q = 0; q < d; q++)
        {
            var src = b.Slice(j0 + (p0 + q) * ldb, w);
            var top = dest.Slice(q * ldt, ldt);
            var bottom = dest.Slice((d + q) * ldt, ldt);
            for (int jj = 0; jj < w; jj++)
            {
                double re = src[jj].Real, im = src[jj].Imaginary;
                top[jj] = re;
                top[w + jj] = im;
                bottom[jj] = -im;
                bottom[w + jj] = re;
            }
        }
    }
}
