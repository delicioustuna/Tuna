using Tuna.LinearAlgebra;
using static Tuna.Tests.LinearAlgebra.LinearAlgebraTestUtilities;

namespace Tuna.Tests.LinearAlgebra;

// 合否は LAPACK のテストと同じ判定比（残差 / (‖A‖·n·ε) ≤ 30）。
[Trait("LinearAlgebra", "Cholesky")]
public class CholeskyDecompositionTests
{
    /// <summary>乱数の正定値行列で <c>‖A - LLᵀ‖ / (‖A‖nε)</c> が小さく、<c>L</c> は対角が正の下三角</summary>
    [Theory]
    [MemberData(nameof(Sizes), MemberType = typeof(LinearAlgebraTestUtilities))]
    public void ReconstructsRandomPositiveDefinite(int n)
    {
        var a = RandomPositiveDefinite(n, 100 + n);
        var l = a.Cholesky().L;
        AssertLowerTriangular(l);
        for (int i = 0; i < n; i++)
        {
            Assert.True(l[i, i] > 0);
        }

        AssertRatio(Difference(a, MultiplyNaive(l, l, transposeB: true)), Norm(a), n, "A - LLᵀ");
    }

    /// <summary>上三角部分は参照しない（NaN を入れても結果が変わらない）</summary>
    [Fact]
    public void ReadsOnlyLowerTriangle()
    {
        int n = 70;
        var a = RandomPositiveDefinite(n, 7);
        var b = a.Clone();
        for (int j = 1; j < n; j++)
        {
            for (int i = 0; i < j; i++)
            {
                b[i, j] = double.NaN;
            }
        }

        Assert.Equal(a.Cholesky().L.ToArray(), b.Cholesky().L.ToArray());
    }

    /// <summary>条件数 1.6e13 のヒルベルト行列（n=10）でも後退誤差は小さい</summary>
    [Fact]
    public void HilbertMatrixHasSmallBackwardError()
    {
        int n = 10;
        var a = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                a[i, j] = 1.0 / (i + j + 1);
            }
        }

        var l = a.Cholesky().L;
        AssertRatio(Difference(a, MultiplyNaive(l, l, transposeB: true)), Norm(a), n, "A - LLᵀ");
    }

    /// <summary>対角行列では <c>L</c> は対角の平方根に厳密に一致する</summary>
    [Fact]
    public void DiagonalMatrixGivesExactSquareRoots()
    {
        var a = Matrix<double>.FromDiagonal([4, 9, 2, 1e-300, 1e300]);
        var l = a.Cholesky().L;
        Assert.Equal(2.0, l[0, 0]);
        Assert.Equal(3.0, l[1, 1]);
        Assert.Equal(Math.Sqrt(2.0), l[2, 2]);
        Assert.Equal(1e-150, l[3, 3], 1e-165);
        Assert.Equal(1e150, l[4, 4], 1e135);
    }

    /// <summary>非正定値（不定、半正定値、NaN を含む）は失敗として報告する</summary>
    [Fact]
    public void RejectsMatricesThatAreNotPositiveDefinite()
    {
        var indefinite = new Matrix<double>(new double[,] { { 1, 2 }, { 2, 1 } });
        Assert.False(CholeskyDecomposition.TryCompute(indefinite, out _));
        Assert.Throws<ArgumentException>(() => indefinite.Cholesky());

        var semidefinite = new Matrix<double>(new double[,] { { 1, 1 }, { 1, 1 } });
        Assert.False(CholeskyDecomposition.TryCompute(semidefinite, out _));

        var nan = new Matrix<double>(new double[,] { { double.NaN } });
        Assert.False(CholeskyDecomposition.TryCompute(nan, out _));

        // ブロック化された経路の途中（n > 64）で負になる場合
        var large = RandomPositiveDefinite(100, 3);
        large[90, 90] = -1;
        Assert.False(CholeskyDecomposition.TryCompute(large, out _));

        Assert.Throws<ArgumentException>(() => new Matrix<double>(2, 3).Cholesky());
    }

    /// <summary>連立方程式の解と行列式</summary>
    [Fact]
    public void SolvesAndComputesDeterminant()
    {
        var a = new Matrix<double>(new double[,] { { 4, 12, -16 }, { 12, 37, -43 }, { -16, -43, 98 } });
        var chol = a.Cholesky();
        Assert.Equal(new double[,] { { 2, 0, 0 }, { 6, 1, 0 }, { -8, 5, 3 } }, chol.L.ToArray());
        Assert.Equal(36.0, chol.Determinant, 1e-12);
        Assert.Equal(Math.Log(36.0), chol.LogDeterminant, 1e-14);
        var x = chol.Solve([1, 2, 3]);
        var ax = a * x.ToVector();
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(i + 1.0, ax[i], 1e-12);
        }
    }

    /// <summary>0次と1次</summary>
    [Fact]
    public void HandlesTrivialSizes()
    {
        Assert.Equal(0, new Matrix<double>(0, 0).Cholesky().L.Rows);
        Assert.Equal(3.0, new Matrix<double>(new double[,] { { 9 } }).Cholesky().L[0, 0]);
    }
}
