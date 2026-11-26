using System.Numerics;
using static Tuna.Tests.LinearAlgebra.LinearAlgebraTestUtilities;

namespace Tuna.Tests.LinearAlgebra;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む
using Tuna.LinearAlgebra;

// 連立一次方程式（三角行列・LU 分解・コレスキー分解の行列の右辺と逆行列）。
// 合否は LAPACK のテストと同じ判定比（‖AX - B‖ / (‖A‖‖X‖nε) ≤ 30）。
[Trait("LinearAlgebra", "Solve")]
public class SolveTests
{
    /// <summary>ブロック化（葉 16）・並列化（列 64 ずつ）の境界をまたぐ次数</summary>
    public static TheoryData<int> SolveSizes => new() { 0, 1, 2, 5, 16, 17, 33, 64, 65, 100, 129, 300 };

    /// <summary>右辺の列数（並列化の境界 64 列をまたぐ）</summary>
    public static TheoryData<int, int> SolveShapes => new() { { 1, 1 }, { 17, 3 }, { 65, 130 }, { 300, 7 }, { 300, 200 } };

    /// <summary>対角が大きい（条件のよい）三角行列。反対側の三角部分には NaN を入れ、読まないことも確かめる</summary>
    private static Matrix<double> RandomTriangular(int n, bool lower, int seed)
    {
        var a = Random(n, n, seed);
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                if (i == j)
                {
                    a[i, j] = (a[i, j] >= 0 ? 1 : -1) * (n + 1);
                }
                else if (lower ? i < j : i > j)
                {
                    a[i, j] = double.NaN;
                }
            }
        }

        return a;
    }

    /// <summary>NaN を入れた反対側の三角部分を 0 にした行列（残差の計算用）</summary>
    private static Matrix<double> Triangle(Matrix<double> a, bool lower)
    {
        var t = new Matrix<double>(a.Rows, a.Columns);
        for (int j = 0; j < a.Columns; j++)
        {
            for (int i = 0; i < a.Rows; i++)
            {
                if (lower ? i >= j : i <= j)
                {
                    t[i, j] = a[i, j];
                }
            }
        }

        return t;
    }

    private static Matrix<double> Transposed(Matrix<double> a) => a.T.ToMatrix();

    /// <summary><c>‖AX - B‖ / (‖A‖‖X‖nε)</c> が閾値以下</summary>
    private static void AssertSolves(Matrix<double> a, Matrix<double> x, Matrix<double> b, string what)
    {
        var residual = Difference(MultiplyNaive(a, x), b);
        AssertRatio(residual, Math.Max(Norm(a) * Norm(x), Norm(b)), a.Rows, what);
    }

    [Theory]
    [MemberData(nameof(SolveShapes))]
    public void TriangularSolvesHaveSmallResiduals(int n, int nrhs)
    {
        var lower = RandomTriangular(n, lower: true, 10 + n);
        var upper = RandomTriangular(n, lower: false, 20 + n);
        var b = Random(n, nrhs, 30 + n);

        AssertSolves(Triangle(lower, true), XMath.SolveLowerTriangular(lower, b), b, "Lx = b");
        AssertSolves(Triangle(upper, false), XMath.SolveUpperTriangular(upper, b), b, "Ux = b");

        // 転置は転置ビューで渡す（Lᵀ は上三角、Uᵀ は下三角）
        AssertSolves(Transposed(Triangle(lower, true)), XMath.SolveUpperTriangular(lower.T, b), b, "Lᵀx = b");
        AssertSolves(Transposed(Triangle(upper, false)), XMath.SolveLowerTriangular(upper.T, b), b, "Uᵀx = b");
    }

    [Theory]
    [MemberData(nameof(SolveSizes))]
    public void TriangularVectorSolvesHaveSmallResiduals(int n)
    {
        var lower = RandomTriangular(n, lower: true, 40 + n);
        var b = Random(n, 1, 50 + n);
        var x = XMath.SolveLowerTriangular(lower, b.Column(0).ToVector());
        AssertSolves(Triangle(lower, true), Matrix<double>.FromColumnMajor(n, 1, x.AsSpan()), b, "Lx = b (vector)");
        var y = XMath.SolveUpperTriangular(lower.T, b.Column(0).ToVector());
        AssertSolves(Transposed(Triangle(lower, true)), Matrix<double>.FromColumnMajor(n, 1, y.AsSpan()), b, "Lᵀx = b (vector)");
    }

    /// <summary>行列の右辺の各列の解は、一緒に解く列の数によらずビット単位で同じ</summary>
    [Theory]
    [InlineData(100)]
    [InlineData(300)]
    public void MatrixSolveColumnsDoNotDependOnColumnCount(int n)
    {
        var lower = RandomTriangular(n, lower: true, 60 + n);
        var b = Random(n, 150, 70 + n);
        var all = XMath.SolveLowerTriangular(lower, b);
        var allT = XMath.SolveUpperTriangular(lower.T, b);
        foreach (int j in new[] { 0, 63, 64, 149 })
        {
            var one = XMath.SolveLowerTriangular(lower, b[.., j..(j + 1)]);
            var oneT = XMath.SolveUpperTriangular(lower.T, b[.., j..(j + 1)]);
            for (int i = 0; i < n; i++)
            {
                Assert.Equal(all[i, j], one[i, 0]);
                Assert.Equal(allT[i, j], oneT[i, 0]);
            }
        }
    }

    /// <summary>係数・右辺・出力先がストライド付きのビュー（転置・部分行列・間引き）でも、連続な行列と同じ解になる</summary>
    [Fact]
    public void TriangularSolvesAcceptStridedViews()
    {
        int n = 40;
        var lower = RandomTriangular(n, lower: true, 80);
        var b = Random(n, 5, 81);
        var expected = XMath.SolveLowerTriangular(lower, b);

        // 係数を行・列とも 1 つおきに並べた大きな行列の中に置く（BLAS に渡せない配置）
        var big = new Matrix<double>(2 * n, 2 * n);
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                big[2 * i, 2 * j] = lower[i, j];
            }
        }

        var sparseView = new ReadOnlyMatrixSpan<double>(big.AsSpan(), [n, n], [2, 4 * n]);

        // 出力先は転置ビュー（列のストライドが 1）
        var destination = new Matrix<double>(5, n);
        XMath.SolveLowerTriangular(sparseView, b, destination.T);
        Assert.Equal(expected.ToArray(), destination.T.ToArray());

        // 右辺が部分行列
        var wide = Random(n + 3, 9, 82);
        var sub = wide[3.., 2..7];
        var x = XMath.SolveLowerTriangular(lower, sub);
        Assert.Equal(XMath.SolveLowerTriangular(lower, sub.ToMatrix()).ToArray(), x.ToArray());
    }

    /// <summary>出力先が右辺と同じならその場で解き、別の領域に解いた結果とビット単位で同じ</summary>
    [Fact]
    public void TriangularSolveInPlaceMatchesOutOfPlace()
    {
        int n = 70;
        var upper = RandomTriangular(n, lower: false, 90);
        var b = Random(n, 4, 91);
        var expected = XMath.SolveUpperTriangular(upper, b);
        var inPlace = b.Clone();
        XMath.SolveUpperTriangular(upper, inPlace, inPlace);
        Assert.Equal(expected.ToArray(), inPlace.ToArray());

        var v = b.Column(1).ToVector();
        var expectedVector = XMath.SolveUpperTriangular(upper, v);
        XMath.SolveUpperTriangular(upper, v, v);
        Assert.Equal(expectedVector.ToArray(), v.ToArray());
    }

    [Fact]
    public void TriangularSolveRejectsOverlapAndBadShapes()
    {
        var a = RandomTriangular(6, lower: true, 92);
        var b = Random(6, 3, 93);

        // 出力先が係数と重なる、右辺と部分的に重なる
        Assert.Throws<ArgumentException>(() => XMath.SolveLowerTriangular(a, a[.., 0..3], a[.., 3..6]));
        var both = Random(6, 4, 94);
        Assert.Throws<ArgumentException>(() => XMath.SolveLowerTriangular(a, both[.., 0..3], both[.., 1..4]));

        // 形状
        Assert.Throws<ArgumentException>(() => XMath.SolveLowerTriangular(Random(6, 5, 95), b));
        Assert.Throws<ArgumentException>(() => XMath.SolveLowerTriangular(a, Random(5, 3, 96)));
        Assert.Throws<ArgumentException>(() => XMath.SolveLowerTriangular(a, b, new Matrix<double>(6, 2)));
    }

    /// <summary>汎用の要素型（float・Complex）も TRSV の経路で解ける</summary>
    [Fact]
    public void TriangularSolvesSupportGenericElementTypes()
    {
        var lf = new Matrix<float>(new float[,] { { 2, 0, 0 }, { 1, 4, 0 }, { -1, 2, 5 } });
        var bf = new Matrix<float>(new float[,] { { 2, 4 }, { 5, 6 }, { 6, 7 } });
        var xf = XMath.SolveLowerTriangular(lf, bf);
        var check = XMath.Multiply(lf, xf);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                Assert.Equal(bf[i, j], check[i, j], 1e-5f);
            }
        }

        var lc = new Matrix<Complex>(new Complex[,] { { new(1, 1), 0 }, { new(0, 2), new(3, -1) } });
        var bc = new Vector<Complex>([new(1, 0), new(2, 3)]);
        var xc = XMath.SolveLowerTriangular(lc, bc);
        var r = XMath.Multiply(lc, xc);
        Assert.True(Complex.Abs(r[0] - bc[0]) < 1e-14);
        Assert.True(Complex.Abs(r[1] - bc[1]) < 1e-14);

        // 上三角の転置ビュー（複素数は共役を取らない転置）
        var yc = XMath.SolveUpperTriangular(lc.T, bc);
        var rt = XMath.Multiply(lc.T, yc);
        Assert.True(Complex.Abs(rt[0] - bc[0]) < 1e-14);
        Assert.True(Complex.Abs(rt[1] - bc[1]) < 1e-14);
    }

    [Theory]
    [MemberData(nameof(Sizes), MemberType = typeof(LinearAlgebraTestUtilities))]
    public void LuReconstructsPermutedMatrix(int n)
    {
        var a = Random(n, n, 200 + n);
        var lu = a.Lu();
        Assert.False(lu.IsSingular);
        var l = lu.L;
        var u = lu.U;
        AssertUpperTriangular(u);
        AssertLowerTriangular(l);
        for (int i = 0; i < n; i++)
        {
            Assert.Equal(1.0, l[i, i]);
        }

        // 部分ピボットなので |lᵢⱼ| ≤ 1
        for (int j = 0; j < n; j++)
        {
            for (int i = j + 1; i < n; i++)
            {
                Assert.True(Math.Abs(l[i, j]) <= 1);
            }
        }

        var p = lu.Permutation;
        Assert.Equal(Enumerable.Range(0, n), p.Order());
        var pa = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                pa[i, j] = a[p[i], j];
            }
        }

        AssertRatio(Difference(pa, MultiplyNaive(l, u)), Norm(a), n, "PA - LU");
    }

    [Theory]
    [MemberData(nameof(SolveShapes))]
    public void LuSolvesHaveSmallResiduals(int n, int nrhs)
    {
        var a = Random(n, n, 300 + n);
        var b = Random(n, nrhs, 310 + n);
        var lu = a.Lu();
        AssertSolves(a, lu.Solve(b), b, "AX = B");
        AssertSolves(a, XMath.Solve(a, b), b, "XMath.Solve");
        var x = lu.Solve(b.Column(0));
        AssertSolves(a, Matrix<double>.FromColumnMajor(n, 1, x.AsSpan()), b[.., 0..1].ToMatrix(), "Ax = b");
        Assert.Equal(x.ToArray(), XMath.Solve(a, b.Column(0)).ToArray());
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(65)]
    [InlineData(200)]
    public void InverseTimesMatrixIsIdentity(int n)
    {
        var a = Random(n, n, 400 + n);
        var inverse = XMath.Inverse(a);
        Assert.Equal(inverse.ToArray(), a.Lu().Inverse().ToArray());
        var product = MultiplyNaive(a, inverse);
        AssertRatio(Difference(product, Matrix<double>.Identity(n)), Norm(a) * Norm(inverse), n, "A·A⁻¹ - I");
    }

    [Fact]
    public void LuDeterminantHasCorrectSign()
    {
        var swap = new Matrix<double>(new double[,] { { 0, 1 }, { 1, 0 } });
        Assert.Equal(-1.0, swap.Lu().Determinant);
        Assert.Equal(-1, swap.Lu().DeterminantSign);
        Assert.Equal(0.0, swap.Lu().LogAbsDeterminant);

        var a = new Matrix<double>(new double[,] { { 2, -1, 0 }, { -1, 2, -1 }, { 0, -1, 2 } });
        Assert.Equal(4.0, a.Lu().Determinant, 1e-14);

        var b = new Matrix<double>(new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 10 } });
        var lu = b.Lu();
        Assert.Equal(-3.0, lu.Determinant, 1e-13);
        Assert.Equal(-1, lu.DeterminantSign);
        Assert.Equal(Math.Log(3), lu.LogAbsDeterminant, 1e-14);

        // 大きな行列で行列式が溢れても、対数は有限
        int n = 300;
        var big = Matrix<double>.Identity(n);
        for (int i = 0; i < n; i++)
        {
            big[i, i] = i % 2 == 0 ? 1e10 : -1e10;
        }

        var bigLu = big.Lu();
        Assert.Equal(n * Math.Log(1e10), bigLu.LogAbsDeterminant, 1e-9);
        Assert.Equal(1, bigLu.DeterminantSign);
        Assert.True(double.IsPositiveInfinity(bigLu.Determinant));
    }

    /// <summary>先頭のピボットが極端に小さくても、行の交換で精度よく解ける</summary>
    [Fact]
    public void LuPivotsAwayFromTinyLeadingEntry()
    {
        var a = new Matrix<double>(new double[,] { { 1e-20, 1 }, { 1, 1 } });
        var x = XMath.Solve(a, new Vector<double>([1.0, 2.0]));
        Assert.Equal(1.0, x[0], 1e-15);
        Assert.Equal(1.0, x[1], 1e-15);
    }

    [Fact]
    public void SingularMatricesAreReported()
    {
        var a = new Matrix<double>(new double[,] { { 1, 2, 3 }, { 2, 4, 6 }, { 1, 0, 1 } });
        var lu = a.Lu();
        Assert.True(lu.IsSingular);
        Assert.Equal(0.0, lu.Determinant);
        Assert.Equal(0, lu.DeterminantSign);
        Assert.True(double.IsNegativeInfinity(lu.LogAbsDeterminant));
        Assert.Throws<InvalidOperationException>(() => lu.Solve(new Vector<double>([1.0, 2.0, 3.0])));
        Assert.Throws<InvalidOperationException>(() => lu.Inverse());
        Assert.Throws<ArgumentException>(() => XMath.Solve(a, new Vector<double>([1.0, 2.0, 3.0])));
        Assert.Throws<ArgumentException>(() => XMath.Inverse(a));
        Assert.Throws<ArgumentException>(() => XMath.Solve(new Matrix<double>(3, 3), Matrix<double>.Identity(3)));
    }

    [Fact]
    public void LuCanOverwriteInputAndRejectsNonSquare()
    {
        var a = Random(20, 20, 500);
        var copy = a.Clone();
        var lu = LuDecomposition.Compute(a, overwriteInput: true);
        var b = Random(20, 2, 501);
        AssertSolves(copy, lu.Solve(b), b, "overwriteInput");
        Assert.Throws<ArgumentException>(() => Random(3, 4, 502).Lu());
        Assert.Equal(0, new Matrix<double>(0, 0).Lu().Order);
    }

    [Fact]
    public void SolveRejectsOverlapWithCoefficient()
    {
        var a = Random(4, 8, 503);
        Assert.Throws<ArgumentException>(() => XMath.Solve(a[.., 0..4], a[.., 4..6], a[.., 2..4]));
        Assert.Throws<ArgumentException>(() => XMath.Inverse(a[.., 0..4], a[.., 2..6]));
    }

    [Theory]
    [MemberData(nameof(SolveShapes))]
    public void CholeskyMatrixSolveHasSmallResidual(int n, int nrhs)
    {
        var a = RandomPositiveDefinite(n, 600 + n);
        var b = Random(n, nrhs, 610 + n);
        var chol = a.Cholesky();
        var x = chol.Solve(b);
        AssertSolves(a, x, b, "AX = B (Cholesky)");

        // 行列の右辺の各列は、一緒に解く列の数によらない
        var first = chol.Solve(b[.., 0..1]);
        for (int i = 0; i < n; i++)
        {
            Assert.Equal(x[i, 0], first[i, 0]);
        }

        // ベクトルのビュー版は配列版と同じ計算
        var v = chol.Solve(b.Column(nrhs - 1));
        Assert.Equal(chol.Solve(b.Column(nrhs - 1).ToArray()), v.ToArray());
    }

    [Fact]
    public void CholeskySolveSupportsInPlaceAndRejectsOverlapWithFactor()
    {
        var a = RandomPositiveDefinite(30, 620);
        var chol = a.Cholesky();
        var b = Random(30, 3, 621);
        var expected = chol.Solve(b);
        var inPlace = b.Clone();
        chol.Solve(inPlace, inPlace);
        Assert.Equal(expected.ToArray(), inPlace.ToArray());
        Assert.Throws<ArgumentException>(() => chol.Solve(b, chol.L[.., 0..3]));
        Assert.Throws<ArgumentException>(() => chol.Solve(Random(29, 3, 622)));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(130)]
    [InlineData(300)]
    public void CholeskyInverseIsSymmetricInverse(int n)
    {
        var a = RandomPositiveDefinite(n, 700 + n);
        var inverse = a.Cholesky().Inverse();
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < j; i++)
            {
                Assert.Equal(inverse[j, i], inverse[i, j]);
            }
        }

        AssertRatio(Difference(MultiplyNaive(a, inverse), Matrix<double>.Identity(n)), Norm(a) * Norm(inverse), n, "A·A⁻¹ - I");
    }
}
