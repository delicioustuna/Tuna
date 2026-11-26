using Tuna.LinearAlgebra;
using static Tuna.Tests.LinearAlgebra.LinearAlgebraTestUtilities;

namespace Tuna.Tests.LinearAlgebra;

// 同じ分解オブジェクトでの再計算（Recompute）および LU 転置求解・destination 逆行列を検証する。
[Trait("LinearAlgebra", "Recompute")]
public class RecomputeTests
{
    public static TheoryData<int> RecomputeSizes => new() { 0, 1, 2, 7, 16, 65, 129, 300 };

    // Recompute の結果が新規分解とビット単位で一致し、L が同じ領域のまま更新されること
    [Theory]
    [MemberData(nameof(RecomputeSizes))]
    public void CholeskyRecomputeMatchesFreshDecomposition(int n)
    {
        var first = RandomPositiveDefinite(n, 1000 + n);
        var second = RandomPositiveDefinite(n, 2000 + n);
        var cholesky = CholeskyDecomposition.Compute(first);
        var l = cholesky.L;

        Assert.True(cholesky.TryRecompute(second));
        var fresh = CholeskyDecomposition.Compute(second);
        Assert.True(cholesky.IsPositiveDefinite);
        Assert.Same(l, cholesky.L);
        Assert.Equal(fresh.L.ToArray(), l.ToArray());
        Assert.Equal(fresh.LogDeterminant, cholesky.LogDeterminant);
        Assert.Equal(fresh.EstimateReciprocalConditionNumber(), cholesky.EstimateReciprocalConditionNumber());

        // 元の行列は変えない
        Assert.Equal(RandomPositiveDefinite(n, 2000 + n).ToArray(), second.ToArray());
    }

    // 入力にビュー（部分行列・転置）を渡せること、また L 自身を渡すとインプレースで分解されること
    [Fact]
    public void CholeskyRecomputeAcceptsViewsAndItsOwnStorage()
    {
        int n = 70;
        var a = RandomPositiveDefinite(n, 11);
        var expected = CholeskyDecomposition.Compute(a).L.ToArray();
        var cholesky = CholeskyDecomposition.Compute(RandomPositiveDefinite(n, 12));

        var padded = new Matrix<double>(n + 3, n + 5);
        a.AsMatrixSpan().CopyTo(padded[2..(n + 2), 4..(n + 4)]);
        cholesky.Recompute(padded[2..(n + 2), 4..(n + 4)]);
        Assert.Equal(expected, cholesky.L.ToArray());

        var other = RandomPositiveDefinite(n, 13);
        cholesky.Recompute(other.T);
        Assert.Equal(CholeskyDecomposition.Compute(other.T.ToMatrix()).L.ToArray(), cholesky.L.ToArray());

        // 下三角部分に新しい行列を書き込んでから、L 自身を再分解する
        var l = cholesky.L;
        for (int j = 0; j < n; j++)
        {
            for (int i = j; i < n; i++)
            {
                l[i, j] = a[i, j];
            }
        }

        cholesky.Recompute(l);
        Assert.Equal(expected, cholesky.L.ToArray());
    }

    // Compute に出力先を渡した場合、Recompute でも同じ出力先が書き換わること
    [Fact]
    public void CholeskyRecomputeWritesIntoTheDestinationOfCompute()
    {
        int n = 20;
        var destination = new Matrix<double>(n, n);
        var cholesky = CholeskyDecomposition.Compute(RandomPositiveDefinite(n, 21), destination);
        var a = RandomPositiveDefinite(n, 22);
        cholesky.Recompute(a);
        Assert.Same(destination, cholesky.L);
        Assert.Equal(CholeskyDecomposition.Compute(a).L.ToArray(), destination.ToArray());
    }

    /// <summary>失敗すると、次に成功するまで因子を使う操作は例外、行列式は NaN になる</summary>
    [Fact]
    public void CholeskyRecomputeFailureInvalidatesUntilNextSuccess()
    {
        int n = 6;
        var good = RandomPositiveDefinite(n, 31);
        var indefinite = good.Clone();
        indefinite[3, 3] = -1;
        var cholesky = CholeskyDecomposition.Compute(good);
        var b = Random(n, 2, 32);
        var x = new Matrix<double>(n, 2);
        var v = new Tuna.LinearAlgebra.Vector<double>(n);

        Assert.False(cholesky.TryRecompute(indefinite));
        Assert.False(cholesky.IsPositiveDefinite);
        Assert.True(double.IsNaN(cholesky.Determinant));
        Assert.True(double.IsNaN(cholesky.LogDeterminant));
        Assert.Throws<InvalidOperationException>(() => cholesky.Solve(b));
        Assert.Throws<InvalidOperationException>(() => cholesky.Solve(b, x));
        Assert.Throws<InvalidOperationException>(() => cholesky.Solve(b.Column(0)));
        Assert.Throws<InvalidOperationException>(() => cholesky.Solve((ReadOnlySpan<double>)new double[n]));
        Assert.Throws<InvalidOperationException>(() => cholesky.Inverse());
        Assert.Throws<InvalidOperationException>(() => cholesky.Inverse(new Matrix<double>(n, n)));
        Assert.Throws<InvalidOperationException>(() => cholesky.EstimateReciprocalConditionNumber());
        Assert.Throws<InvalidOperationException>(() => cholesky.Refine(good, b, x));
        Assert.Throws<InvalidOperationException>(() => cholesky.Update(v));
        Assert.Throws<InvalidOperationException>(() => cholesky.TryDowndate(v));
        Assert.Throws<InvalidOperationException>(() => cholesky.TryAppend(new Tuna.LinearAlgebra.Vector<double>(n + 1), out _));
        Assert.Throws<ArgumentException>(() => cholesky.Recompute(indefinite));
        Assert.False(cholesky.IsPositiveDefinite);

        Assert.True(cholesky.TryRecompute(good));
        Assert.True(cholesky.IsPositiveDefinite);
        var fresh = CholeskyDecomposition.Compute(good);
        Assert.Equal(fresh.L.ToArray(), cholesky.L.ToArray());
        Assert.Equal(fresh.Solve(b).ToArray(), cholesky.Solve(b).ToArray());
        Assert.Equal(fresh.EstimateReciprocalConditionNumber(), cholesky.EstimateReciprocalConditionNumber());
    }

    /// <summary>次数が違う、正方でない、L と一部だけ重なる入力は例外にし、分解は変えない</summary>
    [Fact]
    public void CholeskyRecomputeRejectsBadShapesAndPartialOverlap()
    {
        int n = 8;
        var a = RandomPositiveDefinite(n, 41);
        var cholesky = CholeskyDecomposition.Compute(a);
        var before = cholesky.L.ToArray();
        Assert.Throws<ArgumentException>(() => cholesky.TryRecompute(RandomPositiveDefinite(n + 1, 42)));
        Assert.Throws<ArgumentException>(() => cholesky.TryRecompute(Random(n, n + 1, 43)));
        Assert.Throws<ArgumentException>(() => cholesky.TryRecompute(cholesky.L.T));
        Assert.Equal(before, cholesky.L.ToArray());
        Assert.True(cholesky.IsPositiveDefinite);
    }

    /// <summary>LU を分解し直した結果は新しく分解した結果とビット単位で同じで、取得済みの L・U は前のまま</summary>
    [Theory]
    [MemberData(nameof(RecomputeSizes))]
    public void LuRecomputeMatchesFreshDecomposition(int n)
    {
        var first = Random(n, n, 3000 + n);
        var second = Random(n, n, 4000 + n);
        var lu = first.Lu();
        var oldL = lu.L.ToArray();
        var oldU = lu.U.ToArray();
        var heldL = lu.L;

        lu.Recompute(second);
        var fresh = second.Lu();
        Assert.Equal(fresh.IsSingular, lu.IsSingular);
        Assert.Equal(fresh.L.ToArray(), lu.L.ToArray());
        Assert.Equal(fresh.U.ToArray(), lu.U.ToArray());
        Assert.Equal(fresh.Permutation, lu.Permutation);
        Assert.Equal(fresh.Determinant, lu.Determinant);
        Assert.Equal(fresh.EstimateReciprocalConditionNumber(), lu.EstimateReciprocalConditionNumber());
        Assert.Equal(oldL, heldL.ToArray());
        if (n > 0)
        {
            Assert.NotEqual(oldU, lu.U.ToArray());
        }

        var b = Random(n, 3, 5000 + n);
        Assert.Equal(fresh.Solve(b).ToArray(), lu.Solve(b).ToArray());
        Assert.Equal(Random(n, n, 4000 + n).ToArray(), second.ToArray());
    }

    /// <summary>LU は特異でも分解し直し、IsSingular を新しい分解に合わせる。ビューと、overwriteInput で渡した領域そのものも受ける</summary>
    [Fact]
    public void LuRecomputeTracksSingularityAndAcceptsViews()
    {
        int n = 40;
        var regular = Random(n, n, 51);
        // 列が 0 なら、その段のピボットは厳密に 0 になる
        var singular = regular.Clone();
        for (int i = 0; i < n; i++)
        {
            singular[i, 7] = 0;
        }

        var lu = regular.Lu();
        lu.Recompute(singular);
        Assert.True(lu.IsSingular);
        Assert.Throws<InvalidOperationException>(() => lu.Solve(Random(n, 1, 52)));
        Assert.Equal(0, lu.DeterminantSign);

        lu.Recompute(regular.T);
        Assert.False(lu.IsSingular);
        var transposed = regular.T.ToMatrix().Lu();
        Assert.Equal(transposed.U.ToArray(), lu.U.ToArray());

        var storage = regular.Clone();
        var inPlace = LuDecomposition.Compute(storage, overwriteInput: true);
        Random(n, n, 53).AsMatrixSpan().CopyTo(storage);
        inPlace.Recompute(storage);
        Assert.Equal(Random(n, n, 53).Lu().U.ToArray(), inPlace.U.ToArray());

        Assert.Throws<ArgumentException>(() => lu.Recompute(Random(n + 1, n + 1, 54)));
        Assert.Throws<ArgumentException>(() => inPlace.Recompute(storage.T));
    }

    public static TheoryData<int, int> TransposedShapes => new() { { 1, 1 }, { 17, 3 }, { 65, 130 }, { 300, 7 }, { 300, 200 } };

    /// <summary>転置の求解 <c>AᵀX = B</c> の残差は小さく、その場で解いても同じ</summary>
    [Theory]
    [MemberData(nameof(TransposedShapes))]
    public void LuSolveTransposedHasSmallResiduals(int n, int nrhs)
    {
        var a = Random(n, n, 600 + n);
        var b = Random(n, nrhs, 610 + n);
        var lu = a.Lu();
        var at = a.T.ToMatrix();

        var x = lu.SolveTransposed(b);
        AssertRatio(Difference(MultiplyNaive(at, x), b), Math.Max(Norm(a) * Norm(x), Norm(b)), n, "AᵀX = B");
        var inPlace = b.Clone();
        lu.SolveTransposed(inPlace, inPlace);
        Assert.Equal(x.ToArray(), inPlace.ToArray());
        var rowMajor = new Matrix<double>(nrhs, n);
        lu.SolveTransposed(b, rowMajor.T);
        Assert.Equal(x.ToArray(), rowMajor.T.ToMatrix().ToArray());

        var v = lu.SolveTransposed(b.Column(0));
        var xv = Matrix<double>.FromColumnMajor(n, 1, v.AsSpan());
        AssertRatio(Difference(MultiplyNaive(at, xv), b[.., 0..1].ToMatrix()), Math.Max(Norm(a) * Norm(xv), Norm(b)), n, "Aᵀx = b");
        var destination = new Tuna.LinearAlgebra.Vector<double>(n);
        lu.SolveTransposed(b.Column(0), destination);
        Assert.Equal(v.ToArray(), destination.ToArray());
    }

    /// <summary>特異な分解での転置の求解・逆行列は例外にし、出力先を変えない。形状が合わなければ例外</summary>
    [Fact]
    public void LuTransposedSolveAndInverseRejectSingularAndBadShapes()
    {
        var singular = new Matrix<double>(new double[,] { { 1, 2 }, { 2, 4 } });
        var lu = singular.Lu();
        Assert.Throws<InvalidOperationException>(() => lu.SolveTransposed(new Matrix<double>(2, 1)));
        var destination = Matrix<double>.Identity(2);
        destination[0, 1] = 5;
        Assert.Throws<InvalidOperationException>(() => lu.Inverse(destination));
        Assert.Equal(new double[,] { { 1, 5 }, { 0, 1 } }, destination.ToArray());

        var regular = Random(3, 3, 71).Lu();
        Assert.Throws<ArgumentException>(() => regular.SolveTransposed(new Matrix<double>(4, 1)));
        Assert.Throws<ArgumentException>(() => regular.SolveTransposed(new Matrix<double>(3, 2), new Matrix<double>(3, 1)));
        Assert.Throws<ArgumentException>(() => regular.Inverse(new Matrix<double>(3, 2)));
    }

    /// <summary>
    /// overwriteInput で分解した LU では、渡した行列が因子の領域になる。そこを出力先にする求解・転置の求解・逆行列は書く前に例外にし、因子を変えない。
    /// 右辺が因子の領域にあっても、出力先が別なら解ける
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(70)]
    public void LuDestinationsRejectOverlapWithFactorStorage(int n)
    {
        var a = Random(n, n, 800 + n);
        if (n == 1)
        {
            a[0, 0] = 2;
        }

        var storage = a.Clone();
        var lu = LuDecomposition.Compute(storage, overwriteInput: true);
        var factors = storage.Clone();
        double determinant = lu.Determinant;
        var b = Random(n, n, 810 + n);

        Assert.Throws<ArgumentException>(() => lu.Inverse(storage));
        Assert.Throws<ArgumentException>(() => lu.Inverse(storage.T));
        Assert.Throws<ArgumentException>(() => lu.Solve(b, storage));
        Assert.Throws<ArgumentException>(() => lu.SolveTransposed(b, storage));
        Assert.Throws<ArgumentException>(() => lu.Solve(b.Column(0), storage.Column(n - 1)));
        Assert.Throws<ArgumentException>(() => lu.SolveTransposed(b.Column(0), storage.Column(n - 1)));
        Assert.Throws<ArgumentException>(() => lu.SolveTransposed(storage.Column(0), storage.Column(0)));
        var wide = new Matrix<double>(n, n + 1);
        storage.AsMatrixSpan().CopyTo(wide[.., 0..n]);
        Assert.Throws<ArgumentException>(() => lu.Solve(wide[.., 0..n], storage));
        Assert.Equal(factors.ToArray(), storage.ToArray());
        Assert.Equal(determinant, lu.Determinant);

        var fresh = a.Lu();
        Assert.Equal(fresh.Inverse().ToArray(), lu.Inverse().ToArray());
        var x = new Tuna.LinearAlgebra.Vector<double>(n);
        lu.Solve(storage.Column(0), x);
        Assert.Equal(fresh.Solve(factors.Column(0)).ToArray(), x.ToArray());
        lu.SolveTransposed(storage, b);
        Assert.Equal(fresh.SolveTransposed(factors).ToArray(), b.ToArray());
        Assert.Equal(factors.ToArray(), storage.ToArray());
    }

    // LU の destination 逆行列が新規割り当てオーバーロードとビット単位で一致すること（転置ビュー出力先を含む）
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(65)]
    [InlineData(200)]
    public void LuInverseWithDestinationMatchesAllocatingInverse(int n)
    {
        var lu = Random(n, n, 700 + n).Lu();
        var expected = lu.Inverse().ToArray();
        var destination = Random(n, n, 710 + n);
        lu.Inverse(destination);
        Assert.Equal(expected, destination.ToArray());

        var transposed = new Matrix<double>(n, n);
        lu.Inverse(transposed.T);
        Assert.Equal(expected, transposed.T.ToMatrix().ToArray());

        var padded = new Matrix<double>(n + 2, n + 1);
        lu.Inverse(padded[1..(n + 1), 1..(n + 1)]);
        Assert.Equal(expected, padded[1..(n + 1), 1..(n + 1)].ToMatrix().ToArray());
    }
}
