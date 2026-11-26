using Tuna.LinearAlgebra;
using DenseVector = Tuna.LinearAlgebra.Vector<double>;
using static Tuna.Tests.LinearAlgebra.LinearAlgebraTestUtilities;

namespace Tuna.Tests.LinearAlgebra;

// 逆行列の destination オーバーロード、既存因子からの構築、ランク 1 更新・ダウンデート、行と列の追加を検証する。
// 判定基準は他の分解テストと同じ残差比（残差 / (‖A‖·n·ε) ≤ 30）。
[Trait("LinearAlgebra", "Cholesky")]
public class CholeskyUpdateTests
{
    public static TheoryData<int> UpdateSizes => new() { 1, 2, 3, 16, 17, 64, 65, 130 };

    private static Matrix<double> Outer(DenseVector x)
    {
        int n = x.Length;
        var m = new Matrix<double>(n, n);
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                m[i, j] = x[i] * x[j];
            }
        }

        return m;
    }

    private static DenseVector RandomVector(int n, int seed, double scale = 1)
    {
        var random = new Random(seed);
        var x = new DenseVector(n);
        for (int i = 0; i < n; i++)
        {
            x[i] = scale * (2 * random.NextDouble() - 1);
        }

        return x;
    }

    private static Matrix<double> Reconstruct(CholeskyDecomposition chol) => MultiplyNaive(chol.L, chol.L, transposeB: true);

    // ---- 逆行列の destination オーバーロード ----

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(130)]
    [InlineData(300)]
    public void InverseIntoDestinationIsBitwiseEqualToAllocatingInverse(int n)
    {
        var chol = RandomPositiveDefinite(n, 700 + n).Cholesky();
        var expected = chol.Inverse();

        var destination = new Matrix<double>(n, n);
        destination.AsSpan().Fill(double.NaN);
        var returned = chol.Inverse(destination);
        Assert.Equal(expected.AsSpan().ToArray(), destination.AsSpan().ToArray());
        Assert.Equal(n, returned.Rows);

        // 転置ビュー（結果は対称なので直接書く）と、間隔のある部分行列（作業領域から写す）
        var transposed = new Matrix<double>(n, n);
        chol.Inverse(transposed.T);
        Assert.Equal(expected.AsSpan().ToArray(), transposed.AsSpan().ToArray());
        var large = new Matrix<double>(n + 3, n + 2);
        large.AsSpan().Fill(-7);
        chol.Inverse(large[1..(n + 1), 2..]);
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                Assert.Equal(expected[i, j], large[i + 1, j + 2]);
            }
        }

        Assert.Equal(-7, large[0, 0]);
        Assert.Equal(-7, large[n + 2, n + 1]);
    }

    [Fact]
    public void InverseIntoDestinationRejectsWrongShapeAndOverlapWithFactor()
    {
        var chol = RandomPositiveDefinite(6, 1).Cholesky();
        Assert.Throws<ArgumentException>(() => { chol.Inverse(new Matrix<double>(6, 5)); });
        Assert.Throws<ArgumentException>(() => { chol.Inverse(chol.L); });
    }

    [Fact]
    public void BatchInverseIntoDestinationIsBitwiseEqualToAllocatingInverse()
    {
        int n = 40, count = 5;
        var a = new Tensor<double>(n, n, count);
        for (int k = 0; k < count; k++)
        {
            var m = RandomPositiveDefinite(n, 50 + k);
            m.AsMatrixSpan().CopyTo(a.GetMatrix(k));
        }

        // 1つは正定値でない（結果は NaN）
        a.GetMatrix(3).Diagonal().Fill(-1);
        Assert.False(BatchCholeskyDecomposition.TryCompute(a, out var chol));
        var expected = chol.Inverse();
        var destination = new Tensor<double>(n, n, count);
        chol.Inverse(destination);
        Assert.Equal(expected.AsSpan().ToArray(), destination.AsSpan().ToArray());
        Assert.True(double.IsNaN(destination[0, 0, 3]));
        Assert.Throws<ArgumentException>(() => { chol.Inverse(new Tensor<double>(n, n, count - 1)); });
    }

    // ---- 既存の因子から ----

    [Fact]
    public void FromFactorSolvesLikeTheOriginalDecomposition()
    {
        int n = 50;
        var a = RandomPositiveDefinite(n, 3);
        var chol = a.Cholesky();
        var factor = chol.L.Clone();
        for (int j = 1; j < n; j++)
        {
            for (int i = 0; i < j; i++)
            {
                factor[i, j] = double.NaN;  // 上三角は読まない
            }
        }

        var copy = CholeskyDecomposition.FromFactor(factor);
        var b = RandomVector(n, 4);
        Assert.Equal(chol.Solve(b).ToArray(), copy.Solve(b).ToArray());
        Assert.Equal(chol.L.AsSpan().ToArray(), copy.L.AsSpan().ToArray());
        Assert.Equal(chol.LogDeterminant, copy.LogDeterminant);

        // ‖A‖₁ も推定するので、条件数の逆数の推定は元の推定以上で、通常は同じ
        double original = chol.EstimateReciprocalConditionNumber();
        double estimated = copy.EstimateReciprocalConditionNumber();
        Assert.InRange(estimated, original * (1 - 1e-12), 3 * original);
    }

    [Fact]
    public void FromFactorRejectsInvalidFactors()
    {
        Assert.Throws<ArgumentException>(() => CholeskyDecomposition.FromFactor(new Matrix<double>(2, 3)));
        var zeroDiagonal = Matrix<double>.Identity(3);
        zeroDiagonal[1, 1] = 0;
        Assert.Throws<ArgumentException>(() => CholeskyDecomposition.FromFactor(zeroDiagonal));
        var nonFinite = Matrix<double>.Identity(3);
        nonFinite[2, 0] = double.PositiveInfinity;
        Assert.Throws<ArgumentException>(() => CholeskyDecomposition.FromFactor(nonFinite));
        Assert.Equal(0, CholeskyDecomposition.FromFactor(new Matrix<double>(0, 0)).L.Rows);
    }

    // ---- ランク1の更新・ダウンデート ----

    [Theory]
    [MemberData(nameof(UpdateSizes))]
    public void UpdateGivesTheFactorOfTheUpdatedMatrix(int n)
    {
        var a = RandomPositiveDefinite(n, 10 + n);
        var x = RandomVector(n, 20 + n, scale: Math.Sqrt(n));
        var chol = a.Cholesky();
        chol.Update(x);
        var updated = a + Outer(x);
        AssertLowerTriangular(chol.L);
        for (int i = 0; i < n; i++)
        {
            Assert.True(chol.L[i, i] > 0);
        }

        AssertRatio(Difference(updated, Reconstruct(chol)), Norm(updated), n, "A + xxᵀ - L'L'ᵀ");

        // 因子は一意なので、分解し直した因子と丸め誤差の範囲で一致する
        AssertRatio(Difference(updated.Cholesky().L, chol.L), Norm(chol.L), n, "L' - chol(A + xxᵀ)");
    }

    [Theory]
    [MemberData(nameof(UpdateSizes))]
    public void DowndateUndoesUpdate(int n)
    {
        var a = RandomPositiveDefinite(n, 30 + n);
        var x = RandomVector(n, 40 + n, scale: Math.Sqrt(n));
        var chol = (a + Outer(x)).Cholesky();
        Assert.True(chol.TryDowndate(x));
        AssertRatio(Difference(a, Reconstruct(chol)), Norm(a), n, "A - L'L'ᵀ");
        AssertRatio(Difference(a.Cholesky().L, chol.L), Norm(chol.L), n, "L' - chol(A)");
    }

    [Fact]
    public void DowndateThatLosesPositiveDefinitenessLeavesTheFactorUnchanged()
    {
        int n = 20;
        var a = RandomPositiveDefinite(n, 5);
        var chol = a.Cholesky();
        var before = chol.L.AsSpan().ToArray();

        // A − xxᵀ は x = 2·√a₀₀ e₀ で (0, 0) 成分が負になる
        var x = new DenseVector(n);
        x[0] = 2 * Math.Sqrt(a[0, 0]);
        Assert.False(chol.TryDowndate(x));
        Assert.Equal(before, chol.L.AsSpan().ToArray());

        // 境界: A − xxᵀ が特異（x = L の第 0 列、‖L⁻¹x‖ = 1）でも偽
        var column = chol.L.Column(0).ToVector();
        Assert.False(chol.TryDowndate(column));
        Assert.Equal(before, chol.L.AsSpan().ToArray());
    }

    [Fact]
    public void UpdateAcceptsSparseVectorsAndViewsOfTheFactor()
    {
        int n = 30;
        var a = RandomPositiveDefinite(n, 8);
        var chol = a.Cholesky();

        // 後半だけが非零（前半の列は回転しない）
        var x = new DenseVector(n);
        x[20] = 3;
        x[25] = -1;
        chol.Update(x);
        AssertRatio(Difference(a + Outer(x), Reconstruct(chol)), Norm(a), n, "sparse update");

        // L 自身の列を渡してもよい（先に写す）
        var expected = (a + Outer(x));
        var column = chol.L.Column(3).ToVector();
        chol.Update(chol.L.Column(3));
        AssertRatio(Difference(expected + Outer(column), Reconstruct(chol)), Norm(expected), n, "aliased update");
    }

    [Fact]
    public void UpdateRejectsWrongLengthAndNonFiniteValues()
    {
        var chol = RandomPositiveDefinite(4, 2).Cholesky();
        Assert.Throws<ArgumentException>(() => chol.Update(new DenseVector(3)));
        Assert.Throws<ArgumentException>(() => chol.TryDowndate(new DenseVector(5)));
        var x = new DenseVector(4);
        x[1] = double.NaN;
        Assert.Throws<ArgumentException>(() => chol.Update(x));
        Assert.Throws<ArgumentException>(() => chol.TryDowndate(x));
    }

    [Fact]
    public void ConditionEstimateFollowsTheUpdatedMatrix()
    {
        int n = 40;
        var a = RandomPositiveDefinite(n, 9);
        var x = RandomVector(n, 10, scale: 30);
        var chol = a.Cholesky();
        chol.Update(x);
        double expected = (a + Outer(x)).Cholesky().EstimateReciprocalConditionNumber();
        Assert.InRange(chol.EstimateReciprocalConditionNumber(), expected * (1 - 1e-10), 3 * expected);
    }

    // ---- 行と列の追加 ----

    [Theory]
    [MemberData(nameof(UpdateSizes))]
    public void AppendMatchesTheDecompositionOfTheExtendedMatrix(int n)
    {
        var a = RandomPositiveDefinite(n + 1, 60 + n);
        var leading = a[..n, ..n].ToMatrix().Cholesky();
        var before = leading.L.AsSpan().ToArray();
        var extended = leading.Append(a.Column(n));
        Assert.Equal(before, leading.L.AsSpan().ToArray());  // 元の分解は変えない
        Assert.Equal(n + 1, extended.L.Rows);
        AssertLowerTriangular(extended.L);
        AssertRatio(Difference(a, Reconstruct(extended)), Norm(a), n + 1, "A - L'L'ᵀ");
        AssertRatio(Difference(a.Cholesky().L, extended.L), Norm(extended.L), n + 1, "L' - chol(A)");
        for (int j = 0; j < n; j++)
        {
            for (int i = j; i < n; i++)
            {
                Assert.Equal(leading.L[i, j], extended.L[i, j]);  // 左上はそのまま写す
            }
        }
    }

    [Fact]
    public void AppendGrowsFromAnEmptyDecompositionAndRejectsIndefiniteExtensions()
    {
        var chol = new Matrix<double>(0, 0).Cholesky();
        chol = chol.Append(new DenseVector([4.0]));
        Assert.Equal(2, chol.L[0, 0]);
        chol = chol.Append(new DenseVector([2.0, 5.0]));
        Assert.Equal(new double[,] { { 2, 0 }, { 1, 2 } }, chol.L.ToArray());

        // [[4, 2, 6], [2, 5, 3], [6, 3, c]] は c ≤ 9 で正定値でない
        Assert.False(chol.TryAppend(new DenseVector([6.0, 3.0, 9.0]), out _));
        Assert.Throws<ArgumentException>(() => chol.Append(new DenseVector([6.0, 3.0, 9.0])));
        Assert.True(chol.TryAppend(new DenseVector([6.0, 3.0, 10.0]), out var extended));
        Assert.Equal(1, extended.L[2, 2], 15);
        Assert.Throws<ArgumentException>(() => chol.Append(new DenseVector(2)));
        Assert.Throws<ArgumentException>(() => chol.Append(new DenseVector([1.0, double.NaN, 3.0])));
    }
}
