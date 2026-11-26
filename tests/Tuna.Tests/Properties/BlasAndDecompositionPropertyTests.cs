using CsCheck;
using Tuna.LinearAlgebra;
using Tuna.Tests.LinearAlgebra;

namespace Tuna.Tests.Properties;

[Trait("Category", "Property")]
public class BlasAndDecompositionPropertyTests
{
    private const double Sentinel = -777777.5;

    [Fact]
    public void BlasGemm_MatchesNaiveReferenceAndPreservesPadding()
    {
        var shapeGen = Gen.Select(Gen.Int[0, 35], Gen.Int[0, 35], Gen.Int[0, 35]);
        var padGen = Gen.Select(Gen.Int[0, 4], Gen.Int[0, 4], Gen.Int[0, 4]);
        var scalarGen = Gen.Select(Gen.Int[0, 5], Gen.Int[0, 5]);
        var gen = Gen.Select(Gen.Bool, Gen.Bool, shapeGen, padGen, scalarGen, Gen.Int[0, 100000]);

        gen.Sample((tA, tB, shape, pads, scalars, dataSeed) =>
        {
            var (m, n, k) = shape;
            var (padA, padB, padC) = pads;
            var (alphaKind, betaKind) = scalars;
            var transA = tA ? BlasTranspose.Transpose : BlasTranspose.None;
            var transB = tB ? BlasTranspose.Transpose : BlasTranspose.None;
            int aRows = transA == BlasTranspose.None ? m : k;
            int aCols = transA == BlasTranspose.None ? k : m;
            int bRows = transB == BlasTranspose.None ? k : n;
            int bCols = transB == BlasTranspose.None ? n : k;
            int lda = Math.Max(1, aRows) + padA;
            int ldb = Math.Max(1, bRows) + padB;
            int ldc = Math.Max(1, m) + padC;

            var rng = new Random(dataSeed);
            double alpha = PickScalar(alphaKind, rng);
            double beta = PickScalar(betaKind, rng);

            var a = new double[Math.Max(1, lda * aCols)];
            var b = new double[Math.Max(1, ldb * bCols)];
            var cActual = new double[Math.Max(1, ldc * n)];
            var cExpected = new double[cActual.Length];

            for (int i = 0; i < a.Length; i++)
            {
                a[i] = 2.0 * rng.NextDouble() - 1.0;
            }

            for (int i = 0; i < b.Length; i++)
            {
                b[i] = 2.0 * rng.NextDouble() - 1.0;
            }

            Array.Fill(cActual, Sentinel);
            Array.Fill(cExpected, Sentinel);

            for (int j = 0; j < n; j++)
            {
                for (int i = 0; i < m; i++)
                {
                    double v = 2.0 * rng.NextDouble() - 1.0;
                    cActual[i + j * ldc] = v;
                    cExpected[i + j * ldc] = v;
                }
            }

            Blas.Gemm(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, cActual, ldc);

            for (int j = 0; j < n; j++)
            {
                for (int i = 0; i < m; i++)
                {
                    double sum = 0.0;
                    for (int p = 0; p < k; p++)
                    {
                        double aVal = transA == BlasTranspose.None ? a[i + p * lda] : a[p + i * lda];
                        double bVal = transB == BlasTranspose.None ? b[p + j * ldb] : b[j + p * ldb];
                        sum += aVal * bVal;
                    }

                    double cPrev = beta == 0.0 ? 0.0 : beta * cExpected[i + j * ldc];
                    cExpected[i + j * ldc] = alpha * sum + cPrev;
                }
            }

            double tol = 1e-11 * (Math.Abs(alpha) * k + Math.Abs(beta) + 1.0);
            for (int j = 0; j < n; j++)
            {
                for (int i = 0; i < m; i++)
                {
                    double diff = Math.Abs(cActual[i + j * ldc] - cExpected[i + j * ldc]);
                    Assert.True(diff <= tol, $"Gemm mismatch at ({i},{j}): diff={diff:E2}, tol={tol:E2}");
                }

                for (int i = m; i < ldc; i++)
                {
                    Assert.Equal(Sentinel, cActual[i + j * ldc]);
                }
            }
        }, iter: 200, seed: "456789abcdef");
    }

    [Fact]
    public void BlasTrsv_SolvesTriangularSystemAccurately()
    {
        var gen = Gen.Select(
            Gen.Bool,
            Gen.Bool,
            Gen.Bool,
            Gen.Int[1, 24],
            Gen.Int[0, 4],
            Gen.Int[0, 100000]);

        gen.Sample((isUpper, isTrans, isUnit, n, pad, dataSeed) =>
        {
            var uplo = isUpper ? BlasTriangle.Upper : BlasTriangle.Lower;
            var trans = isTrans ? BlasTranspose.Transpose : BlasTranspose.None;
            var diag = isUnit ? BlasDiagonal.Unit : BlasDiagonal.NonUnit;
            int lda = n + pad;

            var rng = new Random(dataSeed);
            var a = new double[lda * n];
            for (int j = 0; j < n; j++)
            {
                for (int i = 0; i < n; i++)
                {
                    bool inTriangle = isUpper ? i <= j : i >= j;
                    if (!inTriangle)
                    {
                        // 反対側の三角領域にノイズを入れて、参照しないことも同時に検証する
                        a[i + j * lda] = 1e6;
                    }
                    else if (i == j)
                    {
                        a[i + j * lda] = isUnit ? 1e6 : (rng.Next(2) == 0 ? 1.0 : -1.0) * (n + 1.0 + rng.NextDouble());
                    }
                    else
                    {
                        a[i + j * lda] = 2.0 * rng.NextDouble() - 1.0;
                    }
                }
            }

            var xTrue = new double[n];
            for (int i = 0; i < n; i++)
            {
                xTrue[i] = 2.0 * rng.NextDouble() - 1.0;
            }

            // b = op(A) * xTrue を独立に計算する
            var b = new double[n];
            for (int i = 0; i < n; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < n; j++)
                {
                    int row = isTrans ? j : i;
                    int col = isTrans ? i : j;
                    bool inTriangle = isUpper ? row <= col : row >= col;
                    if (!inTriangle)
                    {
                        continue;
                    }

                    double aVal = (row == col && isUnit) ? 1.0 : a[row + col * lda];
                    sum += aVal * xTrue[j];
                }

                b[i] = sum;
            }

            var xSolved = (double[])b.Clone();
            Blas.Trsv(uplo, trans, diag, n, a, lda, xSolved);

            double maxErr = 0.0;
            for (int i = 0; i < n; i++)
            {
                maxErr = Math.Max(maxErr, Math.Abs(xSolved[i] - xTrue[i]));
            }

            Assert.True(maxErr <= 1e-10, $"Trsv error {maxErr:E2} exceeded tolerance for n={n}, uplo={uplo}, trans={trans}, diag={diag}");
        }, iter: 150, seed: "56789abcdef0");
    }

    [Fact]
    public void QrDecomposition_AlgebraicInvariantsAndRecomputeConsistency()
    {
        var gen = Gen.Select(Gen.Int[1, 20], Gen.Int[1, 20], Gen.Bool, Gen.Int[0, 100000]);

        gen.Sample((m, n, rankDeficient, dataSeed) =>
        {
            var a = LinearAlgebraTestUtilities.Random(m, n, dataSeed);
            if (rankDeficient && n >= 2)
            {
                for (int i = 0; i < m; i++)
                {
                    a[i, n - 1] = 2.0 * a[i, 0];
                }
            }

            var b = LinearAlgebraTestUtilities.Random(m, n, dataSeed + 1);

            var qr = QrDecomposition.Compute(a);
            var qrLazy = QrDecomposition.ComputeFactors(a);

            double scale = LinearAlgebraTestUtilities.Norm(a);
            var reconstructed = LinearAlgebraTestUtilities.MultiplyNaive(qr.Q, qr.R);
            LinearAlgebraTestUtilities.AssertRatio(LinearAlgebraTestUtilities.Difference(a, reconstructed), scale, Math.Max(m, n), "‖A - QR‖");
            LinearAlgebraTestUtilities.AssertRatio(LinearAlgebraTestUtilities.OrthogonalityError(qr.Q), 1.0, Math.Max(m, n), "‖QᵀQ - I‖");
            LinearAlgebraTestUtilities.AssertUpperTriangular(qr.R);

            Assert.True(LinearAlgebraTestUtilities.Difference(qr.Q, qrLazy.Q) <= 1e-13);
            Assert.True(LinearAlgebraTestUtilities.Difference(qr.R, qrLazy.R) <= 1e-13);

            qr.Recompute(b);
            var qrB = QrDecomposition.Compute(b);
            Assert.True(LinearAlgebraTestUtilities.Difference(qr.Q, qrB.Q) <= 1e-13);
            Assert.True(LinearAlgebraTestUtilities.Difference(qr.R, qrB.R) <= 1e-13);
        }, iter: 120, seed: "6789abcdef01");
    }

    [Fact]
    public void LuDecomposition_AlgebraicInvariantsSolveAndRecompute()
    {
        var gen = Gen.Select(Gen.Int[1, 20], Gen.Int[0, 100000]);

        gen.Sample((n, dataSeed) =>
        {
            // 対角優位寄りにして条件数を安定させつつ、一般の非対称行列を生成する
            var a = LinearAlgebraTestUtilities.Random(n, n, dataSeed);
            var bMat = LinearAlgebraTestUtilities.Random(n, n, dataSeed + 1);
            for (int i = 0; i < n; i++)
            {
                a[i, i] += n;
                bMat[i, i] += n;
            }

            var lu = LuDecomposition.Compute(a);
            Assert.False(lu.IsSingular);

            var pa = new Matrix<double>(n, n);
            int[] perm = lu.Permutation;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    pa[i, j] = a[perm[i], j];
                }
            }

            var product = LinearAlgebraTestUtilities.MultiplyNaive(lu.L, lu.U);
            double scale = LinearAlgebraTestUtilities.Norm(a);
            LinearAlgebraTestUtilities.AssertRatio(LinearAlgebraTestUtilities.Difference(pa, product), scale, n, "‖PA - LU‖");
            LinearAlgebraTestUtilities.AssertLowerTriangular(lu.L);
            LinearAlgebraTestUtilities.AssertUpperTriangular(lu.U);

            var rhs = LinearAlgebraTestUtilities.Random(n, 1, dataSeed + 2).Column(0).ToVector();
            var x = lu.Solve(rhs);
            var ax = LinearAlgebraTestUtilities.MultiplyNaive(a, Matrix<double>.FromColumnMajor(n, 1, x.AsSpan()));
            var rhsMat = Matrix<double>.FromColumnMajor(n, 1, rhs.AsSpan());
            LinearAlgebraTestUtilities.AssertRatio(LinearAlgebraTestUtilities.Difference(ax, rhsMat), scale * LinearAlgebraTestUtilities.Norm(Matrix<double>.FromColumnMajor(n, 1, x.AsSpan())), n, "‖Ax - b‖");

            var xt = lu.SolveTransposed(rhs);
            var atx = LinearAlgebraTestUtilities.MultiplyNaive(a, Matrix<double>.FromColumnMajor(n, 1, xt.AsSpan()), transposeA: true);
            LinearAlgebraTestUtilities.AssertRatio(LinearAlgebraTestUtilities.Difference(atx, rhsMat), scale * LinearAlgebraTestUtilities.Norm(Matrix<double>.FromColumnMajor(n, 1, xt.AsSpan())), n, "‖Aᵀx - b‖");

            lu.Recompute(bMat);
            var luB = LuDecomposition.Compute(bMat);
            Assert.True(LinearAlgebraTestUtilities.Difference(lu.L, luB.L) <= 1e-13);
            Assert.True(LinearAlgebraTestUtilities.Difference(lu.U, luB.U) <= 1e-13);
        }, iter: 120, seed: "789abcdef012");
    }

    [Fact]
    public void CholeskyDecomposition_InvariantsRank1UpdateDowndateAndRecompute()
    {
        var gen = Gen.Select(Gen.Int[1, 18], Gen.Int[0, 100000]);

        gen.Sample((n, dataSeed) =>
        {
            var a = LinearAlgebraTestUtilities.RandomPositiveDefinite(n, dataSeed);
            var b = LinearAlgebraTestUtilities.RandomPositiveDefinite(n, dataSeed + 1);
            var chol = CholeskyDecomposition.Compute(a);

            Assert.True(chol.IsPositiveDefinite);
            LinearAlgebraTestUtilities.AssertLowerTriangular(chol.L);

            double scale = LinearAlgebraTestUtilities.Norm(a);
            var llt = LinearAlgebraTestUtilities.MultiplyNaive(chol.L, chol.L, transposeB: true);
            LinearAlgebraTestUtilities.AssertRatio(LinearAlgebraTestUtilities.Difference(a, llt), scale, n, "‖A - LLᵀ‖");

            // ランク 1 更新 A + xxᵀ の一致と、続くダウンデートによる元の A への復元
            var u = LinearAlgebraTestUtilities.Random(n, 1, dataSeed + 2).Column(0).ToVector();
            var aUpdated = a.Clone();
            for (int j = 0; j < n; j++)
            {
                for (int i = 0; i < n; i++)
                {
                    aUpdated[i, j] += u[i] * u[j];
                }
            }

            chol.Update(u);
            var lltUpdated = LinearAlgebraTestUtilities.MultiplyNaive(chol.L, chol.L, transposeB: true);
            LinearAlgebraTestUtilities.AssertRatio(LinearAlgebraTestUtilities.Difference(aUpdated, lltUpdated), LinearAlgebraTestUtilities.Norm(aUpdated), n, "‖(A + uuᵀ) - L'L'ᵀ‖");

            Assert.True(chol.TryDowndate(u));
            var lltRestored = LinearAlgebraTestUtilities.MultiplyNaive(chol.L, chol.L, transposeB: true);
            LinearAlgebraTestUtilities.AssertRatio(LinearAlgebraTestUtilities.Difference(a, lltRestored), scale, n, "‖A - L_restored L_restoredᵀ‖");

            chol.Recompute(b);
            var cholB = CholeskyDecomposition.Compute(b);
            Assert.True(LinearAlgebraTestUtilities.Difference(chol.L, cholB.L) <= 1e-13);
        }, iter: 120, seed: "89abcdef0123");
    }

    [Fact]
    public void SingularValueDecomposition_InvariantsAndRecomputeConsistency()
    {
        var gen = Gen.Select(Gen.Int[1, 16], Gen.Int[1, 16], Gen.Int[0, 100000]);

        gen.Sample((m, n, dataSeed) =>
        {
            var a = LinearAlgebraTestUtilities.Random(m, n, dataSeed);
            var b = LinearAlgebraTestUtilities.Random(m, n, dataSeed + 1);
            var svd = SingularValueDecomposition.Compute(a);

            int k = Math.Min(m, n);
            Assert.Equal(k, svd.Values.Length);
            for (int i = 0; i < k; i++)
            {
                Assert.True(svd.Values[i] >= 0.0);
                if (i > 0)
                {
                    Assert.True(svd.Values[i - 1] >= svd.Values[i]);
                }
            }

            Assert.NotNull(svd.U);
            Assert.NotNull(svd.V);
            var sigma = Matrix<double>.FromDiagonal(svd.Values);
            var us = LinearAlgebraTestUtilities.MultiplyNaive(svd.U, sigma);
            var reconstructed = LinearAlgebraTestUtilities.MultiplyNaive(us, svd.V, transposeB: true);

            double scale = LinearAlgebraTestUtilities.Norm(a);
            LinearAlgebraTestUtilities.AssertRatio(LinearAlgebraTestUtilities.Difference(a, reconstructed), scale, Math.Max(m, n), "‖A - UΣVᵀ‖");
            LinearAlgebraTestUtilities.AssertRatio(LinearAlgebraTestUtilities.OrthogonalityError(svd.U), 1.0, Math.Max(m, n), "‖UᵀU - I‖");
            LinearAlgebraTestUtilities.AssertRatio(LinearAlgebraTestUtilities.OrthogonalityError(svd.V), 1.0, Math.Max(m, n), "‖VᵀV - I‖");

            svd.Recompute(b);
            var svdB = SingularValueDecomposition.Compute(b);
            for (int i = 0; i < k; i++)
            {
                Assert.True(Math.Abs(svd.Values[i] - svdB.Values[i]) <= 1e-12 * (svdB.Values[0] + 1.0));
            }
        }, iter: 100, seed: "9abcdef01234");
    }

    private static double PickScalar(int kind, Random rng) => kind switch
    {
        0 => 0.0,
        1 => 1.0,
        2 => -1.0,
        _ => 2.0 * rng.NextDouble() - 1.0
    };
}
