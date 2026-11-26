using Tuna.LinearAlgebra;
using static Tuna.Tests.LinearAlgebra.LinearAlgebraTestUtilities;

namespace Tuna.Tests.LinearAlgebra;

[Trait("LinearAlgebra", "Decompositions")]
public class DecompositionViewTests
{
    [Fact]
    public void ComputesAllDecompositionsFromStridedAndTransposedViews()
    {
        var padded = Random(12, 10, 401);
        var slice = padded[2..8, 1..7]; // 6x6 submatrix view (MatrixSpan)
        var sliceMat = slice.ToMatrix();

        // Symmetric SPD view (A^T A + 5 I)
        var spdPadded = new Matrix<double>(10, 10);
        var spd = sliceMat.T * sliceMat + 5.0 * Matrix<double>.Identity(6);
        spd.AsMatrixSpan().CopyTo(spdPadded[2..8, 2..8]);
        var spdSlice = spdPadded[2..8, 2..8];

        // Cholesky (MatrixSpan, ReadOnlyMatrixSpan, Compute, TryCompute, destination)
        var cholExpected = spd.Cholesky();
        Assert.Equal(cholExpected.L.ToArray(), spdSlice.Cholesky().L.ToArray());
        Assert.Equal(cholExpected.L.ToArray(), spdSlice.AsReadOnly().Cholesky().L.ToArray());
        Assert.True(CholeskyDecomposition.TryCompute(spdSlice.AsReadOnly(), out var cholTry));
        Assert.Equal(cholExpected.L.ToArray(), cholTry.L.ToArray());

        var cholDest = new Matrix<double>(6, 6);
        var cholWithDest = CholeskyDecomposition.Compute(spdSlice.AsReadOnly(), cholDest);
        Assert.Same(cholDest, cholWithDest.L);
        Assert.Equal(cholExpected.L.ToArray(), cholDest.ToArray());

        // LU
        var luExpected = sliceMat.Lu();
        Assert.Equal(luExpected.L.ToArray(), slice.Lu().L.ToArray());
        Assert.Equal(luExpected.U.ToArray(), slice.AsReadOnly().Lu().U.ToArray());
        Assert.Equal(luExpected.Permutation, LuDecomposition.Compute(slice.AsReadOnly()).Permutation);

        // QR (rectangular submatrix + transpose)
        var rectSlice = padded[1..10, 2..7]; // 9x5
        var rectMat = rectSlice.ToMatrix();
        var qrExpected = rectMat.Qr();
        Assert.Equal(qrExpected.Q.ToArray(), rectSlice.Qr().Q.ToArray());
        Assert.Equal(qrExpected.R.ToArray(), rectSlice.AsReadOnly().Qr().R.ToArray());
        Assert.Equal(qrExpected.R.ToArray(), QrDecomposition.ComputeFactors(rectSlice.AsReadOnly()).R.ToArray());

        // SymmetricEigen
        var symExpected = spd.SymmetricEigen();
        var symFromSpan = spdSlice.SymmetricEigen();
        Assert.Equal(symExpected.Values, symFromSpan.Values);
        Assert.Equal(symExpected.Vectors!.ToArray(), symFromSpan.Vectors!.ToArray());
        Assert.Equal(symExpected.Values, spdSlice.AsReadOnly().SymmetricEigen(computeVectors: false).Values);

        // General Eigen (using transposed view)
        var eigExpected = sliceMat.T.ToMatrix().Eigen();
        var eigFromTranspose = slice.T.Eigen();
        Assert.Equal(eigExpected.Values, eigFromTranspose.Values);
        Assert.Equal(eigExpected.Vectors!.ToArray(), eigFromTranspose.Vectors!.ToArray());
        Assert.Equal(eigExpected.Values, slice.T.AsReadOnly().Eigen(computeVectors: false).Values);

        // SVD (both tall and wide views)
        var svdTallExpected = rectMat.Svd();
        var svdTallView = rectSlice.Svd();
        Assert.Equal(svdTallExpected.Values, svdTallView.Values);
        Assert.Equal(svdTallExpected.U!.ToArray(), svdTallView.U!.ToArray());
        Assert.Equal(svdTallExpected.V!.ToArray(), svdTallView.V!.ToArray());

        var svdWideExpected = rectMat.T.ToMatrix().Svd();
        var svdWideView = rectSlice.T.AsReadOnly().Svd();
        Assert.Equal(svdWideExpected.Values, svdWideView.Values);
        Assert.Equal(svdWideExpected.U!.ToArray(), svdWideView.U!.ToArray());
        Assert.Equal(svdWideExpected.V!.ToArray(), svdWideView.V!.ToArray());
    }

    [Fact]
    public void RejectsNonSquareViewsInSquareDecompositions()
    {
        var rect = new Matrix<double>(4, 3);
        Assert.Throws<ArgumentException>(() => rect.AsMatrixSpan().AsReadOnly().Cholesky());
        Assert.Throws<ArgumentException>(() => CholeskyDecomposition.TryCompute(rect.AsMatrixSpan().AsReadOnly(), out _));
        Assert.Throws<ArgumentException>(() => rect.AsMatrixSpan().AsReadOnly().Lu());
        Assert.Throws<ArgumentException>(() => rect.AsMatrixSpan().AsReadOnly().SymmetricEigen());
        Assert.Throws<ArgumentException>(() => rect.AsMatrixSpan().AsReadOnly().Eigen());
    }
}
