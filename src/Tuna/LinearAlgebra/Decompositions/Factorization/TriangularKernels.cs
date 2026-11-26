using System.Runtime.CompilerServices;

namespace Tuna.LinearAlgebra;

// 係数を再帰的に半分ずつに分け（X₁ = A₁₁⁻¹B₁、B₂ ← B₂ - A₂₁X₁、X₂ = A₂₂⁻¹B₂）、更新の大部分を行列積（GEMM）で行う。
// 分け方は次数だけで決まり、GEMM の各要素の積和の順序は C の列数にもスレッド数にもよらないので、
// 右辺の各列の解は、一緒に解く列の数にもスレッド数にもよらずビット単位で同じになる。
// 対角に 0 があっても検査しない（解は ±∞ や NaN を含む。BLAS の TRSM と同じ）。
/// <summary>
/// 三角行列を係数とする複数右辺の連立一次方程式 <c>op(A)·X = B</c>（BLAS dtrsm の左側版に相当、倍精度）
/// </summary>
internal static class TriangularKernels
{
    /// <summary>
    /// これ以下の次数は列ごとの前進・後退代入で解く
    /// </summary>
    private const int LeafSize = 16;

    /// <summary>
    /// 列の範囲で並列化するときの、1スレッドあたりの最小の列数（これより少なければ、GEMM の内部の並列化に任せる）
    /// </summary>
    private const int ParallelColumns = 64;

    /// <summary>
    /// <c>op(A)·X = B</c> を解き、<c>B</c>（<c>n × nrhs</c>、列優先）を解で上書きする
    /// </summary>
    /// <param name="lower">真なら <c>op(A)</c> は下三角、偽なら上三角（反対側の三角部分は読まない）。</param>
    /// <param name="transposeA">真なら <c>op(A)</c> は格納された行列 <c>S</c> の転置（<c>S</c> は反対側の三角）。</param>
    /// <param name="unitDiagonal">真なら対角を 1 とみなし、格納された値を読まない。</param>
    /// <param name="n">次数。</param>
    /// <param name="nrhs">右辺の列数。</param>
    /// <param name="a">係数 <c>S</c>（列優先、列の間隔 <paramref name="lda"/>）。</param>
    /// <param name="lda">係数の列の間隔。</param>
    /// <param name="b">右辺（解で上書きする）。</param>
    /// <param name="ldb">右辺の列の間隔。</param>
    internal static unsafe void SolveLeft(
        bool lower, bool transposeA, bool unitDiagonal, int n, int nrhs, ReadOnlySpan<double> a, int lda, Span<double> b, int ldb)
    {
        if (n <= 0 || nrhs <= 0)
        {
            return;
        }

        // 右辺の各列は独立に解けるので、列が十分多ければ列の範囲ごとに別のスレッドで解く
        int threads = Math.Min(Parallelism.Degree((double)n * n * nrhs / 2, Parallelism.Grain), nrhs / ParallelColumns);
        if (threads > 1)
        {
            SolveColumnsParallel(threads, lower, transposeA, unitDiagonal, n, nrhs, a, lda, b, ldb);
            return;
        }

        Solve(lower, transposeA, unitDiagonal, n, nrhs, a, lda, b, ldb);
    }

    /// <summary>
    /// 右辺の列の範囲ごとに別のスレッドで解く（値型の作業で、呼び出しごとに捕捉やデリゲートを確保しない）
    /// </summary>
    private static unsafe void SolveColumnsParallel(
        int threads, bool lower, bool transposeA, bool unitDiagonal, int n, int nrhs, ReadOnlySpan<double> a, int lda, Span<double> b, int ldb)
    {
        int columns = (nrhs + threads - 1) / threads;
        fixed (double* pa = a)
        fixed (double* pb = b)
        {
            var work = new SolveColumnsWork(lower, transposeA, unitDiagonal, n, nrhs, columns,
                new SharedSpan(pa, a.Length), lda, new SharedSpan(pb, b.Length), ldb);
            Parallelism.For(threads, threads, ref work);
        }
    }

    /// <summary>
    /// 右辺の列 <c>[t·columns, (t + 1)·columns)</c> を解く作業 <c>t</c>
    /// </summary>
    private readonly struct SolveColumnsWork(
        bool lower, bool transposeA, bool unitDiagonal, int n, int nrhs, int columns, SharedSpan a, int lda, SharedSpan b, int ldb) : IParallelWork
    {
        public void Invoke(int index)
        {
            int j0 = index * columns;
            int count = Math.Min(columns, nrhs - j0);
            if (count > 0)
            {
                Solve(lower, transposeA, unitDiagonal, n, count, a.Span, lda, b.Span[(j0 * ldb)..], ldb);
            }
        }
    }

    private static void Solve(
        bool lower, bool transposeA, bool unitDiagonal, int n, int nrhs, ReadOnlySpan<double> a, int lda, Span<double> b, int ldb)
    {
        if (n <= LeafSize)
        {
            SolveLeaf(lower, transposeA, unitDiagonal, n, nrhs, a, lda, b, ldb);
            return;
        }

        int n1 = Math.Min(n - 1, (n / 2 + 7) / 8 * 8);
        int n2 = n - n1;
        var a22 = a[(n1 + n1 * lda)..];
        if (lower)
        {
            // X₁ = A₁₁⁻¹B₁、B₂ ← B₂ - A₂₁X₁、X₂ = A₂₂⁻¹B₂
            Solve(lower, transposeA, unitDiagonal, n1, nrhs, a, lda, b, ldb);
            if (transposeA)
            {
                // A₂₁ = (S₁₂)ᵀ
                Gemm.Multiply(true, false, n2, nrhs, n1, -1.0, a[(n1 * lda)..], lda, b, ldb, b[n1..], ldb);
            }
            else
            {
                Gemm.Multiply(false, false, n2, nrhs, n1, -1.0, a[n1..], lda, b, ldb, b[n1..], ldb);
            }

            Solve(lower, transposeA, unitDiagonal, n2, nrhs, a22, lda, b[n1..], ldb);
        }
        else
        {
            // X₂ = A₂₂⁻¹B₂、B₁ ← B₁ - A₁₂X₂、X₁ = A₁₁⁻¹B₁
            Solve(lower, transposeA, unitDiagonal, n2, nrhs, a22, lda, b[n1..], ldb);
            if (transposeA)
            {
                // A₁₂ = (S₂₁)ᵀ
                Gemm.Multiply(true, false, n1, nrhs, n2, -1.0, a[n1..], lda, b[n1..], ldb, b, ldb);
            }
            else
            {
                Gemm.Multiply(false, false, n1, nrhs, n2, -1.0, a[(n1 * lda)..], lda, b[n1..], ldb, b, ldb);
            }

            Solve(lower, transposeA, unitDiagonal, n1, nrhs, a, lda, b, ldb);
        }
    }

    /// <summary>
    /// 列ごとの前進・後退代入（<c>S</c> の列に沿った axpy か内積で解く）
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void SolveLeaf(
        bool lower, bool transposeA, bool unitDiagonal, int n, int nrhs, ReadOnlySpan<double> a, int lda, Span<double> b, int ldb)
    {
        for (int c = 0; c < nrhs; c++)
        {
            var x = b.Slice(c * ldb, n);
            if (!transposeA)
            {
                if (lower)
                {
                    // S は下三角。前進代入で、解が決まった成分の寄与を S の列に沿って引く
                    for (int j = 0; j < n; j++)
                    {
                        if (x[j] != 0)
                        {
                            if (!unitDiagonal)
                            {
                                x[j] /= a[j + j * lda];
                            }

                            Level1Kernels.Axpy(-x[j], a.Slice(j + 1 + j * lda, n - j - 1), x[(j + 1)..]);
                        }
                    }
                }
                else
                {
                    // S は上三角。後退代入
                    for (int j = n - 1; j >= 0; j--)
                    {
                        if (x[j] != 0)
                        {
                            if (!unitDiagonal)
                            {
                                x[j] /= a[j + j * lda];
                            }

                            Level1Kernels.Axpy(-x[j], a.Slice(j * lda, j), x);
                        }
                    }
                }
            }
            else if (lower)
            {
                // op(A) = Sᵀ は下三角（S は上三角）。前進代入で、S の列との内積を引く
                for (int j = 0; j < n; j++)
                {
                    double t = x[j] - Level1Kernels.Dot(a.Slice(j * lda, j), x);
                    x[j] = unitDiagonal ? t : t / a[j + j * lda];
                }
            }
            else
            {
                // op(A) = Sᵀ は上三角（S は下三角）。後退代入
                for (int j = n - 1; j >= 0; j--)
                {
                    double t = x[j] - Level1Kernels.Dot(a.Slice(j + 1 + j * lda, n - j - 1), x[(j + 1)..]);
                    x[j] = unitDiagonal ? t : t / a[j + j * lda];
                }
            }
        }
    }
}
