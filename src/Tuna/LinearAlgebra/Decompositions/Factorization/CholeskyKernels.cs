using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Tuna.LinearAlgebra;

// LAPACK dpotrf 相当、下三角・ブロック化
/// <summary>
/// コレスキー分解の計算本体
/// </summary>
internal static class CholeskyKernels
{
    /// <summary>
    /// これ以下の次数は非ブロック版で分解する
    /// </summary>
    internal const int LeafSize = 32;

    // A₁₁ = L₁₁L₁₁ᵀ、L₂₁ = A₂₁L₁₁⁻ᵀ、A₂₂ ← A₂₂ - L₂₁L₂₁ᵀ のブロック化。
    // 128次以上は右視的ブロック版、それより小さい行列とパネル内部は再帰版を使う。逐次と並列で分け方を変えない
    // 上三角部分は作業領域として書き換わる。
    /// <summary>
    /// 下三角部分を <c>L</c> で上書きする
    /// </summary>
    /// <returns>
    /// 成功なら0、正定値でなければ失敗した列番号 + 1。
    /// </returns>
    internal static int Factor(int n, Span<double> a, int lda)
    {
        if (n <= LeafSize)
        {
            return FactorUnblocked(n, a, lda);
        }

        if (n >= BlockedMinimum)
        {
            int threads = Parallelism.PlannedDegree((double)n * n * n / 3, 4 * Parallelism.Grain);
            return FactorBlocked(threads, n, a, lda);
        }

        return FactorRecursive(n, a, lda);
    }

    /// <summary>
    /// これ以上の次数で右視的ブロック版を使う（逐次も同じ手順）
    /// </summary>
    private const int BlockedMinimum = 128;

    // パネルの分解（三角方程式を含む）は1つの作業が逐次に行うので、n が小さいうちは細くして 待ち時間を減らし、大きくなったら行列積の内側の次元を確保するために広げる（4 コアでの計測で決めた）。
    /// <summary>
    /// 分解のパネル幅
    /// </summary>
    private static int BlockWidth(int n) => n < 900 ? 32 : 64;

    // 再帰版は大きな行列積で効率がよいが、並列にすると行列積ごとに同期が要り、データが毎回別のコアへ移る。
    // この版では列区間とパネルの幅を次数だけで決め、逐次でも同じ順序で各要素を計算する
    // 固定した列区間の更新と次のパネルの分解を重ねる（先読み）。
    /// <summary>
    /// 右視的ブロック版
    /// </summary>
    private static unsafe int FactorBlocked(int threads, int n, Span<double> a, int lda)
    {
        int nb = BlockWidth(n);
        int failed = FactorPanel(n, 0, Math.Min(nb, n), a, lda);
        if (failed != 0)
        {
            return failed;
        }

        fixed (double* pa = a)
        {
            for (int j = 0; j < n; j += nb)
            {
                int jb = Math.Min(nb, n - j);
                int next = j + jb;
                if (next >= n)
                {
                    break;
                }

                int nextWidth = Math.Min(nb, n - next);
                int remaining = n - next - nextWidth;
                int tasks = 1 + (remaining + UpdateBlock - 1) / UpdateBlock;
                var work = new FactorWork(n, j, jb, next, nextWidth, new SharedSpan(pa, a.Length), lda);
                Parallelism.For(tasks, threads, ref work);
                if (work.Failed != 0)
                {
                    return work.Failed;
                }
            }
        }

        return 0;
    }

    /// <summary>
    /// 次のパネルを先読みし、残りの固定幅の列区間を更新する
    /// </summary>
    private struct FactorWork(int n, int j, int jb, int next, int nextWidth, SharedSpan a, int lda) : IParallelWork
    {
        internal int Failed;

        public void Invoke(int index)
        {
            if (index == 0)
            {
                UpdateColumns(n, j, jb, next, next + nextWidth, a.Span, lda);
                Failed = FactorPanel(n, next, nextWidth, a.Span, lda);
                return;
            }

            int c0 = next + nextWidth + (index - 1) * UpdateBlock;
            UpdateColumns(n, j, jb, c0, Math.Min(n, c0 + UpdateBlock), a.Span, lda);
        }
    }

    /// <summary>
    /// 列 <c>[c0, c1)</c> の下三角部分を、パネル <c>[j, j+jb)</c> の L で更新する（<c>A[c0:n, c0:c1] -= L[c0:n] L[c0:c1]ᵀ</c>）
    /// </summary>
    private static void UpdateColumns(int n, int j, int jb, int c0, int c1, Span<double> a, int lda) =>
        Gemm.MultiplyLower(n - c0, c1 - c0, jb, -1.0, a[(c0 + j * lda)..], lda, a[(c0 + j * lda)..], lda, a[(c0 + c0 * lda)..], lda);

    // 対角ブロックの分解と、その下の三角方程式。
    /// <summary>
    /// 列 <c>[j, j+jb)</c> のパネルを分解する
    /// </summary>
    /// <returns>
    /// 成功なら0、正定値でなければ失敗した列番号 + 1。
    /// </returns>
    private static int FactorPanel(int n, int j, int jb, Span<double> a, int lda)
    {
        var diag = a[(j + j * lda)..];
        int info = FactorRecursive(jb, diag, lda);
        if (info != 0)
        {
            return j + info;
        }

        int below = n - j - jb;
        if (below > 0)
        {
            SolveTransposedRight(below, jb, diag, lda, diag[jb..], lda);
        }

        return 0;
    }

    /// <summary>
    /// 再帰的ブロック版（1スレッド、または並列版のパネル用）
    /// </summary>
    private static int FactorRecursive(int n, Span<double> a, int lda)
    {
        if (n <= LeafSize)
        {
            return FactorUnblocked(n, a, lda);
        }

        int n1 = Math.Min(n - 1, (n / 2 + 7) / 8 * 8);
        int n2 = n - n1;
        int info = FactorRecursive(n1, a, lda);
        if (info != 0)
        {
            return info;
        }

        // A21 ← A21 L11⁻ᵀ
        SolveTransposedRight(n2, n1, a, lda, a[n1..], lda);

        // A22 ← A22 - A21 A21ᵀ（下三角だけを列ブロックごとに更新）
        UpdateTrailing(n2, n1, a[n1..], lda, a[(n1 + n1 * lda)..], lda);

        info = FactorRecursive(n2, a[(n1 + n1 * lda)..], lda);
        return info == 0 ? 0 : n1 + info;
    }

    /// <summary>
    /// 非ブロック版（左視的・列指向。列 j を、それより左の列の 4 列ずつの積和で更新してから正規化する）
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static unsafe int FactorUnblocked(int n, Span<double> a, int lda)
    {
        nint ld = lda;
        fixed (double* pa = a)
        {
            for (int j = 0; j < n; j++)
            {
                // A[j:n, j] -= A[j:n, 0:j] · A[j, 0:j]ᵀ
                int rows = n - j;
                double* col = pa + j + (j * ld);
                double* left = pa + j;
                int p = 0;
                for (; p <= j - 4; p += 4)
                {
                    double* ap = left + (p * ld);
                    RealVectorOps.AddColumns4(rows, ap, ld, -ap[0], -ap[ld], -ap[2 * ld], -ap[3 * ld], col);
                }

                for (; p < j; p++)
                {
                    double* ap = left + (p * ld);
                    double t = -ap[0];
                    for (int i = 0; i < rows; i++)
                    {
                        col[i] += t * ap[i];
                    }
                }

                double ajj = col[0];
                if (!(ajj > 0) || double.IsPositiveInfinity(ajj))
                {
                    return j + 1;
                }

                ajj = Math.Sqrt(ajj);
                col[0] = ajj;
                if (rows > 1)
                {
                    Level1Kernels.Scale(1.0 / ajj, new Span<double>(col + 1, rows - 1));
                }
            }
        }

        return 0;
    }

    // L は k × k 下三角、B は m × k。
    /// <summary>
    /// <c>B ← B L⁻ᵀ</c>
    /// </summary>
    internal static unsafe void SolveTransposedRight(int m, int k, ReadOnlySpan<double> l, int ldl, Span<double> b, int ldb)
    {
        if (m <= SolveRows)
        {
            SolveTransposedRightSequential(m, k, l, ldl, b, ldb);
            return;
        }

        int blocks = 1 + (m - 1) / SolveRows;
        int threads = Parallelism.Degree((double)m * k * k / 2, Parallelism.Grain);
        fixed (double* pl = l)
        fixed (double* pb = b)
        {
            var work = new SolveRightWork(m, k, new SharedSpan(pl, l.Length), ldl, new SharedSpan(pb, b.Length), ldb);
            Parallelism.For(blocks, threads, ref work);
        }
    }

    /// <summary>
    /// SIMD の端数の位置を並列数によらず固定する行数
    /// </summary>
    private const int SolveRows = 128;

    private readonly struct SolveRightWork(int m, int k, SharedSpan l, int ldl, SharedSpan b, int ldb) : IParallelWork
    {
        public void Invoke(int index)
        {
            int first = index * SolveRows;
            SolveTransposedRightSequential(Math.Min(SolveRows, m - first), k, l.Span, ldl, b.Span[first..], ldb);
        }
    }

    /// <summary>
    /// 固定した行区間の三角方程式を、列方向に再帰分割して解く
    /// </summary>
    private static void SolveTransposedRightSequential(int m, int k, ReadOnlySpan<double> l, int ldl, Span<double> b, int ldb)
    {
        if (k > 16)
        {
            // L = [L₁₁ 0; L₂₁ L₂₂] と分けて X₁ = B₁L₁₁⁻ᵀ、X₂ = (B₂ - X₁L₂₁ᵀ)L₂₂⁻ᵀ
            int k1 = k / 2;
            SolveTransposedRightSequential(m, k1, l, ldl, b, ldb);
            Gemm.Multiply(false, true, m, k - k1, k1, -1.0, b, ldb, l[k1..], ldl, b[(k1 * ldb)..], ldb);
            SolveTransposedRightSequential(m, k - k1, l[(k1 + k1 * ldl)..], ldl, b[(k1 * ldb)..], ldb);
            return;
        }

        // X Lᵀ = B ⇔ X[:, c] = (B[:, c] - Σ_{p<c} L[c, p] X[:, p]) / L[c, c]
        int done = SolveLeafVector(m, k, l, ldl, b, ldb);
        if (done == m)
        {
            return;
        }

        b = b[done..];
        m -= done;
        for (int c = 0; c < k; c++)
        {
            var bc = b.Slice(c * ldb, m);
            for (int p = 0; p < c; p++)
            {
                Level1Kernels.Axpy(-l[c + p * ldl], b.Slice(p * ldb, m), bc);
            }

            Level1Kernels.Scale(1.0 / l[c + c * ldl], bc);
        }
    }

    // 1つの行ブロックの k 列をレジスタ（またはスタック）に置いたまま前進代入するので、B の各要素は1回ずつ読み書きする。
    // 8 または 4 行。
    /// <summary>
    /// <see cref="SolveTransposedRight"/> の葉（<c>k ≤ 16</c>）を、行を SIMD の幅ずつまとめて解く
    /// </summary>
    /// <returns>
    /// 解いた行数（先頭から。残りの行は呼び出し元が解く）。
    /// </returns>
    private static unsafe int SolveLeafVector(int m, int k, ReadOnlySpan<double> l, int ldl, Span<double> b, int ldb)
    {
        int width = Vector512.IsHardwareAccelerated ? 8 : Vector256.IsHardwareAccelerated ? 4 : 0;
        if (width == 0 || m < width || k > 16)
        {
            return 0;
        }

        // 係数 -L[c, p] と 1 / L[c, c] を先に並べておく
        double* coef = stackalloc double[16 * 16];
        double* inv = stackalloc double[16];
        for (int c = 0; c < k; c++)
        {
            inv[c] = 1.0 / l[c + c * ldl];
            for (int p = 0; p < c; p++)
            {
                coef[c * 16 + p] = -l[c + p * ldl];
            }
        }

        int rows = m - m % width;
        nint ld = ldb;
        fixed (double* pb = b)
        {
            if (width == 8)
            {
                Vector512<double>* x = stackalloc Vector512<double>[16];
                for (int i = 0; i < rows; i += 8)
                {
                    double* bi = pb + i;
                    for (int c = 0; c < k; c++)
                    {
                        var v = Vector512.Load(bi + c * ld);
                        double* row = coef + c * 16;
                        for (int p = 0; p < c; p++)
                        {
                            v = Vector512.FusedMultiplyAdd(Vector512.Create(row[p]), x[p], v);
                        }

                        v *= inv[c];
                        x[c] = v;
                        v.Store(bi + c * ld);
                    }
                }
            }
            else
            {
                Vector256<double>* x = stackalloc Vector256<double>[16];
                for (int i = 0; i < rows; i += 4)
                {
                    double* bi = pb + i;
                    for (int c = 0; c < k; c++)
                    {
                        var v = Vector256.Load(bi + c * ld);
                        double* row = coef + c * 16;
                        for (int p = 0; p < c; p++)
                        {
                            v = RealVectorOps.MulAdd(Vector256.Create(row[p]), x[p], v);
                        }

                        v *= inv[c];
                        x[c] = v;
                        v.Store(bi + c * ld);
                    }
                }
            }
        }

        return rows;
    }

    /// <summary>
    /// コレスキー因子 <c>L</c>（<c>n × n</c> 下三角）で <c>LLᵀX = B</c> を解き、<c>B</c>（<c>n × nrhs</c>、列優先）を解で上書きする
    /// </summary>
    /// <param name="n">行列の次数。</param>
    /// <param name="l">下三角因子 L。</param>
    /// <param name="ldl">下三角因子の列の間隔。</param>
    /// <param name="nrhs">右辺の列数。</param>
    /// <param name="b">右辺の値。</param>
    /// <param name="ldb">右辺の列の間隔。</param>
    /// <param name="vector">真なら各列を前進・後退代入（ベクトルの右辺の経路）で、偽なら再帰的な TRSM（行列の右辺の経路）で解く。</param>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Solve(int n, ReadOnlySpan<double> l, int ldl, int nrhs, Span<double> b, int ldb, bool vector)
    {
        if (n == 0 || nrhs == 0)
        {
            return;
        }

        if (!vector)
        {
            TriangularKernels.SolveLeft(lower: true, transposeA: false, unitDiagonal: false, n, nrhs, l, ldl, b, ldb);
            TriangularKernels.SolveLeft(lower: false, transposeA: true, unitDiagonal: false, n, nrhs, l, ldl, b, ldb);
            return;
        }

        for (int c = 0; c < nrhs; c++)
        {
            var x = b.Slice(c * ldb, n);

            // Ly = b（列指向の前進代入）
            for (int j = 0; j < n; j++)
            {
                x[j] /= l[j + j * ldl];
                Level1Kernels.Axpy(-x[j], l.Slice(j + 1 + j * ldl, n - j - 1), x[(j + 1)..]);
            }

            // Lᵀx = y（列の内積による後退代入）
            for (int j = n - 1; j >= 0; j--)
            {
                x[j] = (x[j] - Level1Kernels.Dot(l.Slice(j + 1 + j * ldl, n - j - 1), x[(j + 1)..])) / l[j + j * ldl];
            }
        }
    }

    /// <summary>
    /// <paramref name="x"/> を <c>LLᵀx</c> で上書きする（<c>A = LLᵀ</c> の掛け算。<c>‖A‖₁</c> の推定に使う。<c>O(n²)</c>、逐次）
    /// </summary>
    internal static void MultiplyFactorProduct(int n, ReadOnlySpan<double> l, int ldl, Span<double> x)
    {
        double[] rented = WorkArrayPool<double>.Rent(n);
        var y = rented.AsSpan(0, n);

        // y = Lᵀx（L の列と x の内積）
        for (int j = 0; j < n; j++)
        {
            y[j] = Level1Kernels.Dot(l.Slice(j + j * ldl, n - j), x[j..]);
        }

        // x = Ly（L の列に沿って足す）
        x.Clear();
        for (int j = 0; j < n; j++)
        {
            Level1Kernels.Axpy(y[j], l.Slice(j + j * ldl, n - j), x[j..]);
        }

        WorkArrayPool<double>.Return(rented);
    }

    // 列 j と w に平面回転 (Lⱼ, w) ← (cLⱼ + sw, cw − sLⱼ)（r = hypot(lⱼⱼ, wⱼ)、c = lⱼⱼ/r、s = wⱼ/r）を掛けて
    // wⱼ を 0 にする。回転は直交なので [L, w][L, w]ᵀ は変わらず、新しい対角は r > 0。wⱼ = 0 の列は回転しない
    // w は作業領域として壊す。
    /// <summary>
    /// <c>LLᵀ + wwᵀ</c> の因子に <paramref name="l"/> をその場で書き換える
    /// </summary>
    internal static void UpdateRankOne(int n, Span<double> l, int ldl, Span<double> w)
    {
        for (int j = 0; j < n; j++)
        {
            double wj = w[j];
            if (wj == 0)
            {
                continue;
            }

            double ljj = l[j + j * ldl];
            double r = double.Hypot(ljj, wj);
            double c = ljj / r;
            double s = wj / r;
            l[j + j * ldl] = r;
            w[j] = 0;
            Level1Kernels.Rotate(l.Slice(j + 1 + j * ldl, n - j - 1), w[(j + 1)..], c, s);
        }
    }

    // p = L⁻¹x、α = √(1 − ‖p‖²) とし、[α; p] の成分を下から順に第1成分へ集める平面回転を求める（‖p‖ < 1 が正定値の条件）。
    // 同じ回転を [0ᵀ; Lᵀ] の行（L の列）に下から掛けると、第1行が xᵀ、残りが新しい Lᵀ になる
    // （[0ᵀ; Lᵀ] の積 LLᵀ = xxᵀ + L'L'ᵀ。G. W. Stewart, "The effects of rounding error on an algorithm for downdating a Cholesky factorization",
    // IMA J. Appl. Math. 23 (1979) 203–213 の方法）。新しい対角は cⱼlⱼⱼ なので、書き換える前に正で有限かを確かめる
    // 正定値でなければ何も書かずに偽を返す。
    /// <summary>
    /// <c>LLᵀ − xxᵀ</c> が正定値なら、その因子に <paramref name="l"/> をその場で書き換えて真を返す
    /// </summary>
    /// <param name="n">行列の次数。</param>
    /// <param name="l">下三角因子 L。</param>
    /// <param name="ldl">下三角因子の列の間隔。</param>
    /// <param name="work">長さ <c>3n</c> の作業領域。先頭の <c>n</c> 個に <c>x</c> を入れて渡す。</param>
    internal static bool TryDowndateRankOne(int n, Span<double> l, int ldl, Span<double> work)
    {
        var p = work[..n];
        var cosines = work.Slice(n, n);
        var sines = work.Slice(2 * n, n);
        TriangularKernels.SolveLeft(lower: true, transposeA: false, unitDiagonal: false, n, 1, l, ldl, p, n);
        double norm = Level1Kernels.Norm2(p);
        if (!(norm < 1))
        {
            return false;
        }

        double v = Math.Sqrt((1 - norm) * (1 + norm));
        for (int j = n - 1; j >= 0; j--)
        {
            double r = double.Hypot(v, p[j]);
            cosines[j] = v / r;
            sines[j] = p[j] / r;
            v = r;
            double diagonal = cosines[j] * l[j + j * ldl];
            if (!(diagonal > 0 && double.IsFinite(diagonal)))
            {
                return false;
            }
        }

        // z（第1行。最後は x になる）と L の列を下から回転する。列 j の時点で z の第 j 成分より上は 0
        var z = p;
        z.Clear();
        for (int j = n - 1; j >= 0; j--)
        {
            Level1Kernels.Rotate(z[j..], l.Slice(j + j * ldl, n - j), cosines[j], sines[j]);
        }

        return true;
    }

    /// <summary>
    /// 逆行列を求めるときの列のブロック幅
    /// </summary>
    private const int InverseBlock = 64;

    // M = L⁻¹ を c の下三角に列のブロックごとの三角方程式（右辺は単位行列の列、上の 0 の行は飛ばす）で求め、
    // その場で MᵀM の下三角に置き換えてから（dlauum 相当）、上三角へ写す。n × n の作業領域は使わない。
    // 分け方は次数だけで決まり、行列積と三角方程式の各列の結果は一緒に計算する列の数にもスレッド数にもよらないので、結果はスレッド数によらない。
    // LAPACK dpotri 相当
    /// <summary>
    /// コレスキー因子 <c>L</c> から <c>A⁻¹ = L⁻ᵀL⁻¹</c> を求め、<paramref name="c"/>（<c>n × n</c>、列優先）に対称行列として書く
    /// </summary>
    internal static void Inverse(int n, ReadOnlySpan<double> l, int ldl, Span<double> c, int ldc)
    {
        if (n == 0)
        {
            return;
        }

        int blocks = (n + InverseBlock - 1) / InverseBlock;
        int threads = Math.Min(blocks, Parallelism.Degree((double)n * n * n / 6, Parallelism.Grain));
        if (threads > 1)
        {
            InverseSolveParallel(blocks, threads, n, l, ldl, c, ldc);
        }
        else
        {
            for (int b = 0; b < blocks; b++)
            {
                InverseSolveBlock(b, n, l, ldl, c, ldc);
            }
        }

        GramOfLower(n, c, ldc);
        for (int j = 1; j < n; j++)
        {
            for (int i = 0; i < j; i++)
            {
                c[i + j * ldc] = c[j + i * ldc];
            }
        }
    }

    // 対角ブロックの上側は 0 になる（上三角の部分で、最後に写すときに上書きされる）。
    /// <summary>
    /// <c>M[j:, J] = L[j:, j:]⁻¹ E</c> を <paramref name="c"/> の同じ位置に求める（<c>E</c> は単位行列の列 <c>J</c> のうち行 <c>j</c> 以降）
    /// </summary>
    private static void InverseSolveBlock(int b, int n, ReadOnlySpan<double> l, int ldl, Span<double> c, int ldc)
    {
        int j = b * InverseBlock;
        int jb = Math.Min(InverseBlock, n - j);
        for (int k = 0; k < jb; k++)
        {
            var column = c.Slice(j + (j + k) * ldc, n - j);
            column.Clear();
            column[k] = 1;
        }

        TriangularKernels.SolveLeft(lower: true, transposeA: false, unitDiagonal: false, n - j, jb, l[(j + j * ldl)..], ldl, c[(j + j * ldc)..], ldc);
    }

    /// <summary>
    /// 列のブロックを並列に解く（値型の作業で、呼び出しごとに捕捉やデリゲートを確保しない）
    /// </summary>
    private static unsafe void InverseSolveParallel(int blocks, int threads, int n, ReadOnlySpan<double> l, int ldl, Span<double> c, int ldc)
    {
        fixed (double* pl = l)
        fixed (double* pc = c)
        {
            var work = new InverseSolveWork(n, new SharedSpan(pl, l.Length), ldl, new SharedSpan(pc, c.Length), ldc);
            Parallelism.For(blocks, threads, ref work);
        }
    }

    private readonly struct InverseSolveWork(int n, SharedSpan l, int ldl, SharedSpan c, int ldc) : IParallelWork
    {
        public void Invoke(int index)
        {
            using var scope = Parallelism.Sequential();
            InverseSolveBlock(index, n, l.Span, ldl, c.Span, ldc);
        }
    }

    // 幅 InverseBlock の行のブロック I = [i, i + ib) ごとに、上から順に
    // (TᵀT)[I, :i+ib] = T[I, I]ᵀT[I, :i+ib] + T[i+ib:, I]ᵀT[i+ib:, :i+ib] で置き換える（下の行だけを読むので、上から順に置き換えられる）。
    // 対角ブロック T[I, I] は上側を 0 にした写しを作り、対角ブロック自身と、その左の幅 InverseBlock の列の範囲ごとに、
    // 写しとの行列積で置き換える。これらは互いに独立なので並列に計算する。作業領域は n によらず、写しと各範囲の元の値の分（それぞれ 32 KB 以下）だけ
    // LAPACK dlauum 'L' 相当
    // 上三角部分は作業領域として書き換わる。
    /// <summary>
    /// 下三角行列 <c>T</c>（下三角部分だけを読む）をその場で <c>TᵀT</c> の下三角に置き換える
    /// </summary>
    private static unsafe void GramOfLower(int n, Span<double> a, int lda)
    {
        const int Nb = InverseBlock;

        // 対角ブロック T[I, I] の写し（n によらず 32 KB）
        double[] rented = WorkArrayPool<double>.Rent(checked(Nb * Nb));
        try
        {
            fixed (double* pd = rented)
            fixed (double* pa = a)
            {
                var sd = new SharedSpan(pd, Nb * Nb);
                var sa = new SharedSpan(pa, a.Length);
                // 処理した行数だけ進める（i += Nb は、n が int の上限に近いと最後のブロックの後に負へ折り返す）
                for (int i = 0, ib; i < n; i += ib)
                {
                    ib = Math.Min(Nb, n - i);
                    int tasks = (i + Nb - 1) / Nb + 1;
                    int threads = Parallelism.Degree((double)ib * (i + ib) * (n - i), Parallelism.Grain);
                    var d = sd.Span[..(ib * ib)];
                    d.Clear();
                    for (int k = 0; k < ib; k++)
                    {
                        a.Slice(i + k + (i + k) * lda, ib - k).CopyTo(d.Slice(k + k * ib, ib - k));
                    }

                    // 作業がスレッド数の半分以下なら、作業を順に計算して各行列積の中で並列化する
                    if (threads > 1 && 2 * tasks > threads)
                    {
                        GramOfLowerRowParallel(tasks, Math.Min(tasks, threads), n, i, ib, sa, lda, sd);
                    }
                    else
                    {
                        for (int t = 0; t < tasks; t++)
                        {
                            GramOfLowerRowTask(t, n, i, ib, sa.Span, lda, d);
                        }
                    }
                }
            }
        }
        finally
        {
            WorkArrayPool<double>.Return(rented);
        }
    }

    /// <summary>
    /// 行のブロックの作業を並列に計算する（値型の作業で、呼び出しごとに捕捉やデリゲートを確保しない）
    /// </summary>
    private static void GramOfLowerRowParallel(int tasks, int threads, int n, int i, int ib, SharedSpan a, int lda, SharedSpan diagonal)
    {
        var work = new GramRowWork(n, i, ib, a, lda, diagonal);
        Parallelism.For(tasks, threads, ref work);
    }

    private readonly struct GramRowWork(int n, int i, int ib, SharedSpan a, int lda, SharedSpan diagonal) : IParallelWork
    {
        public void Invoke(int index)
        {
            using var scope = Parallelism.Sequential();
            GramOfLowerRowTask(index, n, i, ib, a.Span, lda, diagonal.Span);
        }
    }

    // 0 番は対角ブロック、それ以外は左の列の範囲 [(t - 1)·Nb, t·Nb)（diagonal は上側を 0 にした T[I, I] の写し、列の間隔 ib）。
    /// <summary>
    /// 行のブロック <c>I = [i, i + ib)</c> の作業 <paramref name="t"/> 番
    /// </summary>
    private static void GramOfLowerRowTask(int t, int n, int i, int ib, Span<double> a, int lda, ReadOnlySpan<double> diagonal)
    {
        int below = n - i - ib;
        var tail = a[(i + ib + i * lda)..];
        if (t == 0)
        {
            var aii = a[(i + i * lda)..];
            for (int k = 0; k < ib; k++)
            {
                aii.Slice(k * lda, ib).Clear();
            }

            Gemm.MultiplyLowerTransposed(ib, ib, ib, 1.0, diagonal, ib, diagonal, ib, aii, lda);
            Gemm.MultiplyLowerTransposed(ib, ib, below, 1.0, tail, lda, tail, lda, aii, lda);
            return;
        }

        int c0 = (t - 1) * InverseBlock;
        int w = Math.Min(InverseBlock, i - c0);
        var block = a[(i + c0 * lda)..];

        // T[I, I]ᵀ との積は、元の値を小さい作業領域に写してから行列積で求める
        double[] rented = WorkArrayPool<double>.Rent(checked(ib * w));
        try
        {
            var old = rented.AsSpan(0, ib * w);
            for (int k = 0; k < w; k++)
            {
                var column = block.Slice(k * lda, ib);
                column.CopyTo(old.Slice(k * ib, ib));
                column.Clear();
            }

            Gemm.Multiply(true, false, ib, w, ib, 1.0, diagonal, ib, old, ib, block, lda);
        }
        finally
        {
            WorkArrayPool<double>.Return(rented);
        }

        Gemm.Multiply(true, false, ib, w, below, 1.0, tail, lda, a[(i + ib + c0 * lda)..], lda, block, lda);
    }

    // A は m × k、C は m × m。
    /// <summary>
    /// <c>C ← C - AAᵀ</c> の下三角部分
    /// </summary>
    internal static unsafe void UpdateTrailing(int m, int k, ReadOnlySpan<double> a, int lda, Span<double> c, int ldc)
    {
        int panels = (m + UpdateBlock - 1) / UpdateBlock;
        int threads = Parallelism.Degree((double)m * m * k / 2, Parallelism.Grain);
        fixed (double* pa = a)
        fixed (double* pc = c)
        {
            var work = new UpdateWork(m, k, new SharedSpan(pa, a.Length), lda, new SharedSpan(pc, c.Length), ldc);
            Parallelism.For(panels, threads, ref work);
        }
    }

    // 行列積の形を並列数によらず固定する。
    /// <summary>
    /// 下三角の更新の列幅
    /// </summary>
    private const int UpdateBlock = 128;

    private readonly struct UpdateWork(int m, int k, SharedSpan a, int lda, SharedSpan c, int ldc) : IParallelWork
    {
        public void Invoke(int index)
        {
            int j = index * UpdateBlock;
            int width = Math.Min(UpdateBlock, m - j);
            Gemm.MultiplyLower(m - j, width, k, -1.0, a.Span[j..], lda, a.Span[j..], lda, c.Span[(j + j * ldc)..], ldc);
        }
    }
}
