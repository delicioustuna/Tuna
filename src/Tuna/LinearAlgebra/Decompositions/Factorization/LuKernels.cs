namespace Tuna.LinearAlgebra;

// 列を半分ずつに分け、左半分を再帰的に分解してから、右半分に行の交換・三角方程式（TriangularKernels）・
// 行列積による更新を行い、右下を再帰的に分解する。更新の大部分が行列積になる。
// ピボットは各列で絶対値が最大の要素（同じ値なら上の行）。分け方は形状だけで決まるので、結果はスレッド数によらない。
// LAPACK dgetrf2 と同じ再帰的な列の分割
/// <summary>
/// 部分ピボット付き LU 分解 <c>PA = LU</c> の計算本体
/// </summary>
internal static class LuKernels
{
    /// <summary>
    /// これ以下の列数は右視的な非ブロック版で分解する
    /// </summary>
    private const int LeafSize = 16;

    /// <summary>
    /// <c>m × n</c>（<c>m ≥ n</c>）の行列を分解し、<c>L</c>（単位下三角、対角は格納しない）と <c>U</c> で上書きする
    /// </summary>
    /// <param name="m">行数。</param>
    /// <param name="n">行列の次数。</param>
    /// <param name="a">係数行列。</param>
    /// <param name="lda">係数行列の列の間隔。</param>
    /// <param name="pivots">長さ <c>n</c>。第 <c>i</c> 段で行 <c>i</c> と行 <c>pivots[i]</c> を交換した（先頭からの添字）。</param>
    /// <returns>
    /// 0 なら正則、正なら最初に 0 になったピボットの段の番号 + 1（分解は最後まで続ける）。
    /// </returns>
    internal static int Factor(int m, int n, Span<double> a, int lda, Span<int> pivots)
    {
        if (n <= LeafSize)
        {
            return FactorUnblocked(m, n, a, lda, pivots);
        }

        int n1 = Math.Min(n - 1, (n / 2 + 7) / 8 * 8);
        int n2 = n - n1;
        int info = Factor(m, n1, a, lda, pivots);

        // 右の列に左の列で決めた行の交換を施し、A₁₂ ← L₁₁⁻¹A₁₂、A₂₂ ← A₂₂ - A₂₁A₁₂
        var right = a[(n1 * lda)..];
        SwapRows(right, lda, n2, pivots, 0, n1);
        TriangularKernels.SolveLeft(lower: true, transposeA: false, unitDiagonal: true, n1, n2, a, lda, right, lda);
        Gemm.Multiply(false, false, m - n1, n2, n1, -1.0, a[n1..], lda, right, lda, right[n1..], lda);

        int info2 = Factor(m - n1, n2, right[n1..], lda, pivots[n1..]);
        if (info == 0 && info2 != 0)
        {
            info = info2 + n1;
        }

        for (int i = n1; i < n; i++)
        {
            pivots[i] += n1;
        }

        // 左の列にも右下で決めた行の交換を施す
        SwapRows(a, lda, n1, pivots, n1, n);
        return info;
    }

    /// <summary>
    /// 右視的な非ブロック版（列ごとにピボットを選び、残りの列を axpy で更新する）
    /// </summary>
    private static int FactorUnblocked(int m, int n, Span<double> a, int lda, Span<int> pivots)
    {
        int info = 0;
        for (int j = 0; j < n; j++)
        {
            var column = a.Slice(j * lda, m);
            int p = j + Level1Kernels.IndexOfMaxAbs(column[j..]);
            pivots[j] = p;
            double pivot = column[p];
            if (pivot != 0)
            {
                if (p != j)
                {
                    for (int c = 0; c < n; c++)
                    {
                        (a[j + c * lda], a[p + c * lda]) = (a[p + c * lda], a[j + c * lda]);
                    }
                }

                if (Math.Abs(pivot) >= NumericConstants.MinNormal)
                {
                    Level1Kernels.Scale(1.0 / pivot, column[(j + 1)..]);
                }
                else
                {
                    for (int i = j + 1; i < m; i++)
                    {
                        column[i] /= pivot;
                    }
                }
            }
            else if (info == 0)
            {
                info = j + 1;
            }

            for (int c = j + 1; c < n; c++)
            {
                Level1Kernels.Axpy(-a[j + c * lda], column[(j + 1)..], a.Slice(j + 1 + c * lda, m - j - 1));
            }
        }

        return info;
    }

    // 行の交換は列どうしが独立なので、 列の範囲に分けて並列に行っても結果は変わらない。
    /// <summary>
    /// 1スレッドあたりの最小の交換の回数（1回は数ns なので約 10µs 分）
    /// </summary>
    private const double SwapGrain = 4 * 1024;

    // LAPACK dlaswp
    // backward が真なら段を逆の順（end - 1 から start）にたどり、置換の逆を施す。
    /// <summary>
    /// 列 <c>0 ≤ c &lt; columns</c> の行を、段 <c>start ≤ i &lt; end</c> の順に <c>i ↔ pivots[i]</c> と交換する
    /// </summary>
    internal static unsafe void SwapRows(Span<double> a, int lda, int columns, ReadOnlySpan<int> pivots, int start, int end, bool backward = false)
    {
        int threads = columns > 1 && end > start ? Parallelism.Degree((double)columns * (end - start), SwapGrain) : 1;
        if (threads <= 1)
        {
            SwapColumns(a, lda, 0, columns, pivots, start, end, backward);
            return;
        }

        fixed (double* pa = a)
        fixed (int* pp = pivots)
        {
            var work = new SwapRowsWork(new SharedSpan(pa, a.Length), lda, columns, pp, pivots.Length, start, end, backward, threads);
            Parallelism.For(threads, threads, ref work);
        }
    }

    /// <summary>
    /// 列を <c>threads</c> 個の範囲に分けて行を交換する作業（閉包を作らない値型）
    /// </summary>
    private readonly unsafe struct SwapRowsWork(
        SharedSpan a, int lda, int columns, int* pivots, int count, int start, int end, bool backward, int threads) : IParallelWork
    {
        public void Invoke(int index)
        {
            int c0 = (int)((long)columns * index / threads);
            int c1 = (int)((long)columns * (index + 1) / threads);
            SwapColumns(a.Span, lda, c0, c1, new ReadOnlySpan<int>(pivots, count), start, end, backward);
        }
    }

    /// <summary>
    /// 列 <c>[c0, c1)</c> について <see cref="SwapRows"/> を行う
    /// </summary>
    private static void SwapColumns(Span<double> a, int lda, int c0, int c1, ReadOnlySpan<int> pivots, int start, int end, bool backward)
    {
        for (int c = c0; c < c1; c++)
        {
            var column = a[(c * lda)..];
            if (backward)
            {
                for (int i = end - 1; i >= start; i--)
                {
                    int p = pivots[i];
                    if (p != i)
                    {
                        (column[i], column[p]) = (column[p], column[i]);
                    }
                }

                continue;
            }

            for (int i = start; i < end; i++)
            {
                int p = pivots[i];
                if (p != i)
                {
                    (column[i], column[p]) = (column[p], column[i]);
                }
            }
        }
    }

    // 転置は LAPACK dgetrs の 'T' と同じ順（Uᵀ、Lᵀ の順に解いてから、行の交換を逆の順に施す。Aᵀ = UᵀLᵀP）。
    // transpose が真なら AᵀX = B。
    /// <summary>
    /// 分解済みの <c>n × n</c> の係数で <c>AX = B</c>を解き、 <c>B</c>（<c>n × nrhs</c>、列優先）を解で上書きする
    /// </summary>
    /// <param name="n">行列の次数。</param>
    /// <param name="lu">LU 分解の因子。</param>
    /// <param name="ldlu">LU 分解の因子の列の間隔。</param>
    /// <param name="pivots">行の交換を表すピボット。</param>
    /// <param name="nrhs">右辺の列数。</param>
    /// <param name="b">右辺の値。</param>
    /// <param name="ldb">右辺の列の間隔。</param>
    /// <param name="vector">真なら各列を TRSV（ベクトルの右辺の経路）で、偽なら TRSM（行列の右辺の経路）で解く。</param>
    /// <param name="transpose">転置した係数行列について解く場合は true。</param>
    internal static void Solve(
        int n, ReadOnlySpan<double> lu, int ldlu, ReadOnlySpan<int> pivots, int nrhs, Span<double> b, int ldb, bool vector, bool transpose = false)
    {
        if (n == 0 || nrhs == 0)
        {
            return;
        }

        if (transpose)
        {
            SolveTransposed(n, lu, ldlu, pivots, nrhs, b, ldb, vector);
            return;
        }

        SwapRows(b, ldb, nrhs, pivots, 0, n);
        if (vector)
        {
            for (int c = 0; c < nrhs; c++)
            {
                var x = b.Slice(c * ldb, n);
                ParallelLevel2<double>.Trsv(BlasTriangle.Lower, BlasTranspose.None, BlasDiagonal.Unit, n, lu, ldlu, x);
                ParallelLevel2<double>.Trsv(BlasTriangle.Upper, BlasTranspose.None, BlasDiagonal.NonUnit, n, lu, ldlu, x);
            }

            return;
        }

        TriangularKernels.SolveLeft(lower: true, transposeA: false, unitDiagonal: true, n, nrhs, lu, ldlu, b, ldb);
        TriangularKernels.SolveLeft(lower: false, transposeA: false, unitDiagonal: false, n, nrhs, lu, ldlu, b, ldb);
    }

    private static void SolveTransposed(int n, ReadOnlySpan<double> lu, int ldlu, ReadOnlySpan<int> pivots, int nrhs, Span<double> b, int ldb, bool vector)
    {
        if (vector)
        {
            for (int c = 0; c < nrhs; c++)
            {
                var x = b.Slice(c * ldb, n);
                ParallelLevel2<double>.Trsv(BlasTriangle.Upper, BlasTranspose.Transpose, BlasDiagonal.NonUnit, n, lu, ldlu, x);
                ParallelLevel2<double>.Trsv(BlasTriangle.Lower, BlasTranspose.Transpose, BlasDiagonal.Unit, n, lu, ldlu, x);
            }
        }
        else
        {
            // lower は op(A) の三角（Uᵀ は下三角、Lᵀ は上三角）
            TriangularKernels.SolveLeft(lower: true, transposeA: true, unitDiagonal: false, n, nrhs, lu, ldlu, b, ldb);
            TriangularKernels.SolveLeft(lower: false, transposeA: true, unitDiagonal: true, n, nrhs, lu, ldlu, b, ldb);
        }

        SwapRows(b, ldb, nrhs, pivots, 0, n, backward: true);
    }
}
