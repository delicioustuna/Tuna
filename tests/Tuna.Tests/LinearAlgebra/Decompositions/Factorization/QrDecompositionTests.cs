using Tuna.LinearAlgebra;
using static Tuna.Tests.LinearAlgebra.LinearAlgebraTestUtilities;

namespace Tuna.Tests.LinearAlgebra;

// 合否は LAPACK のテストと同じ判定比（残差 / (‖A‖·max(m, n)·ε) ≤ 30）。
[Trait("LinearAlgebra", "QR")]
public class QrDecompositionTests
{
    /// <summary>正方行列で <c>A = QR</c>、<c>QᵀQ = I</c>、<c>R</c> は上三角</summary>
    [Theory]
    [MemberData(nameof(Sizes), MemberType = typeof(LinearAlgebraTestUtilities))]
    public void FactorsSquareMatrix(int n) => Check(Random(n, n, 200 + n));

    /// <summary>縦長・横長の行列（ブロック幅をまたぐ形）</summary>
    [Theory]
    [InlineData(10, 3)]
    [InlineData(3, 10)]
    [InlineData(100, 33)]
    [InlineData(33, 100)]
    [InlineData(200, 65)]
    [InlineData(65, 200)]
    [InlineData(1, 7)]
    [InlineData(7, 1)]
    public void FactorsRectangularMatrix(int m, int n) => Check(Random(m, n, m * 1000 + n));

    /// <summary>ランク落ち（0の列、同じ列の繰り返し）でも <c>Q</c> は直交</summary>
    [Fact]
    public void HandlesRankDeficientMatrix()
    {
        var a = Random(80, 70, 5);
        for (int i = 0; i < 80; i++)
        {
            a[i, 3] = 0;
            a[i, 40] = a[i, 10];
        }

        Check(a);
        Check(new Matrix<double>(50, 40));
    }

    /// <summary>極端に大きい・小さい値でもオーバーフロー・アンダーフローしない</summary>
    [Theory]
    [InlineData(1e300)]
    [InlineData(1e-300)]
    [InlineData(4.9406564584124654e-324)]
    public void HandlesExtremeScaling(double scale)
    {
        var a = scale * Random(40, 40, 11);
        Check(a);
    }

    /// <summary>最小二乗解が正規方程式 <c>AᵀAx = Aᵀb</c> を満たす</summary>
    [Fact]
    public void SolvesLeastSquares()
    {
        int m = 60;
        int n = 20;
        var a = Random(m, n, 9);
        var b = new double[m];
        var random = new Random(10);
        for (int i = 0; i < m; i++)
        {
            b[i] = random.NextDouble();
        }

        var x = a.Qr().Solve(b);
        var r = (a * x.ToVector()).ToArray();
        for (int i = 0; i < m; i++)
        {
            r[i] -= b[i];
        }

        // 残差は列空間と直交する: ‖Aᵀr‖ ≤ c·‖A‖·‖r‖·ε
        double rnorm = Math.Sqrt(r.Sum(v => v * v));
        for (int j = 0; j < n; j++)
        {
            double atr = 0;
            for (int i = 0; i < m; i++)
            {
                atr += a[i, j] * r[i];
            }

            Assert.True(Math.Abs(atr) <= 30 * m * Epsilon * Norm(a) * Math.Max(rnorm, 1), $"column {j}: {atr}");
        }
    }

    /// <summary>0行・0列</summary>
    [Fact]
    public void HandlesEmptyMatrix()
    {
        var qr = new Matrix<double>(0, 0).Qr();
        Assert.Equal(0, qr.Q.Rows);
        Assert.Equal(0, qr.R.Columns);
        Assert.Equal(5, new Matrix<double>(5, 0).Qr().Q.Rows);
    }

    private static void Check(Matrix<double> a)
    {
        int m = a.Rows;
        int n = a.Columns;
        int k = Math.Min(m, n);
        var qr = a.Qr();
        Assert.Equal(m, qr.Q.Rows);
        Assert.Equal(k, qr.Q.Columns);
        Assert.Equal(k, qr.R.Rows);
        Assert.Equal(n, qr.R.Columns);
        AssertUpperTriangular(qr.R);
        AssertRatio(Difference(a, MultiplyNaive(qr.Q, qr.R)), Norm(a), Math.Max(m, n), "A - QR");
        AssertRatio(OrthogonalityError(qr.Q), 1, m, "QᵀQ - I");

        // 分解だけを先に計算して後から Q, R を作っても同じ結果
        var lazy = QrDecomposition.ComputeFactors(a);
        Assert.Equal(qr.R.ToArray(), lazy.R.ToArray());
        Assert.Equal(qr.Q.ToArray(), lazy.Q.ToArray());
    }

    [Fact]
    public void SolvesVectorAndMatrixRightHandSides()
    {
        int m = 24;
        int n = 7;
        var a = Random(m, n, 31);
        var bMat = Random(m, 4, 32);
        var qr = a.Qr();

        // Vector<double> オーバーロードが ReadOnlySpan<double> オーバーロードと一致する
        Vector<double> b0 = bMat.Column(0).ToVector();
        double[] expected0 = qr.Solve(b0.AsSpan());
        Vector<double> actual0 = qr.Solve(b0);
        Assert.Equal(expected0, actual0.ToArray());

        // ストライド付きビュー入力と出力先（destination）の再利用
        var transposedB = bMat.T.ToMatrix();
        var rowView = transposedB.Row(1);
        var dstVec = new Vector<double>(n);
        qr.Solve(rowView, dstVec);
        Assert.Equal(qr.Solve(bMat.Column(1).ToArray()), dstVec.ToArray());

        // 複数右辺の行列オーバーロードと、入力に重なる destination（b[0..n, ..]）
        Matrix<double> solutions = qr.Solve(bMat);
        Assert.Equal((n, 4), (solutions.Rows, solutions.Columns));
        for (int col = 0; col < 4; col++)
        {
            Assert.Equal(qr.Solve(bMat.Column(col).ToArray()), solutions.Column(col).ToArray());
        }

        var aliased = bMat.AsMatrixSpan().ToMatrix();
        qr.Solve(aliased, aliased[0..n, ..]);
        Assert.Equal(solutions.ToArray(), aliased[0..n, ..].ToArray());

        // 転置ビュー出力先（非連続ストライド）
        var dstTransposed = new Matrix<double>(4, n);
        qr.Solve(bMat, dstTransposed.T);
        Assert.Equal(solutions.ToArray(), dstTransposed.T.ToArray());

        // 因子ストレージとの重なりは例外になる
        var overwritten = a.AsMatrixSpan().ToMatrix();
        var qrOverwrite = QrDecomposition.Compute(overwritten, overwriteInput: true);
        Assert.Throws<ArgumentException>(() => qrOverwrite.Solve(b0, overwritten.Column(0)[0..n]));
    }
}
