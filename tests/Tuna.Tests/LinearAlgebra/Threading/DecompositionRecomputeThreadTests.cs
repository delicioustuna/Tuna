using Tuna.LinearAlgebra;
using static Tuna.Tests.LinearAlgebra.LinearAlgebraTestUtilities;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>分解し直した結果と LU の転置の求解は、並列数によらず同じビットになる</summary>
[Trait("LinearAlgebra", "Parallelism")]
[Collection(nameof(ParallelismCollection))]
public class DecompositionRecomputeThreadTests
{
    /// <summary>
    /// 300 次・1100 次の LU は再帰の右側で行の交換・三角方程式・行列積を並列にし、257 列の右辺は列の範囲ごとに並列に解く。
    /// Cholesky の分解し直しも右視的ブロック版の並列経路を通る
    /// </summary>
    [Theory]
    [InlineData(300)]
    [InlineData(1100)]
    public void RecomputeAndTransposedSolvesDoNotDependOnThreadCount(int n)
    {
        var general = Random(n, n, 990 + n);
        var spd = RandomPositiveDefinite(n, 991 + n);
        var b = Random(n, 257, 992 + n);
        double[,]? luFactors = null, transposed = null, transposedVector = null, choleskyFactor = null;
        foreach (int threads in new[] { 1, 2, 3, 4 })
        {
            int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
            try
            {
                LinearAlgebraSettings.MaxDegreeOfParallelism = threads;
                var lu = Random(n, n, 993 + n).Lu();
                lu.Recompute(general);
                var x = lu.SolveTransposed(b).ToArray();
                var v = Matrix<double>.FromColumnMajor(n, 1, lu.SolveTransposed(b.Column(0)).AsSpan()).ToArray();
                var u = lu.U.ToArray();
                var cholesky = CholeskyDecomposition.Compute(RandomPositiveDefinite(n, 994 + n));
                cholesky.Recompute(spd);
                var l = cholesky.L.ToArray();
                if (threads == 1)
                {
                    (luFactors, transposed, transposedVector, choleskyFactor) = (u, x, v, l);
                    continue;
                }

                Assert.Equal(luFactors, u);
                Assert.Equal(transposed, x);
                Assert.Equal(transposedVector, v);
                Assert.Equal(choleskyFactor, l);
            }
            finally
            {
                LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
            }
        }
    }

    /// <summary>
    /// QR は 300 次でパネルの分解と更新を重ねる並列版と、Q を作るブロックの並列版を通る。
    /// 対称固有値分解と特異値分解は列の分担を固定した縮約と並列の分割統治法を通り、700 次では列区間が 12 個になる。
    /// 特異値分解の縦長（先に QR 分解）と横長（転置して解く）も含める
    /// </summary>
    [Theory]
    [InlineData(300)]
    [InlineData(700)]
    public void OrthogonalRecomputeDoesNotDependOnThreadCount(int n)
    {
        var general = Random(n, n, 1990 + n);
        var symmetric = RandomSymmetric(n, 1991 + n);
        var tall = Random(n + n / 2 + n / 4, n / 2, 1992 + n);
        double[,]? q = null, r = null, vectors = null, u = null, v = null, tallU = null, wideV = null;
        double[]? values = null, singular = null;
        foreach (int threads in new[] { 1, 2, 3, 4 })
        {
            int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
            try
            {
                LinearAlgebraSettings.MaxDegreeOfParallelism = threads;
                var qr = Random(n, n, 1993 + n).Qr();
                qr.Recompute(general);
                var eig = RandomSymmetric(n, 1994 + n).SymmetricEigen();
                eig.Recompute(symmetric);
                var svd = Random(n, n, 1995 + n).Svd();
                svd.Recompute(general);
                var tallSvd = Random(tall.Rows, tall.Columns, 1996 + n).Svd();
                tallSvd.Recompute(tall);
                var wideSvd = Random(tall.Columns, tall.Rows, 1997 + n).Svd();
                wideSvd.Recompute(tall.T);
                var result = (qr.Q.ToArray(), qr.R.ToArray(), eig.Values, eig.Vectors!.ToArray(), svd.Values, svd.U!.ToArray(), svd.V!.ToArray(),
                    tallSvd.U!.ToArray(), wideSvd.V!.ToArray());
                if (threads == 1)
                {
                    (q, r, values, vectors, singular, u, v, tallU, wideV) = result;
                    continue;
                }

                Assert.Equal(q, result.Item1);
                Assert.Equal(r, result.Item2);
                Assert.Equal(values, result.Item3);
                Assert.Equal(vectors, result.Item4);
                Assert.Equal(singular, result.Item5);
                Assert.Equal(u, result.Item6);
                Assert.Equal(v, result.Item7);
                Assert.Equal(tallU, result.Item8);
                Assert.Equal(wideV, result.Item9);
            }
            finally
            {
                LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
            }
        }
    }
}
