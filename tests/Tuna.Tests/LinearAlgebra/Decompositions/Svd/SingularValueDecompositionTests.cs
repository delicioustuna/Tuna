using Tuna.LinearAlgebra;
using static Tuna.Tests.LinearAlgebra.LinearAlgebraTestUtilities;

namespace Tuna.Tests.LinearAlgebra;

// 合否は LAPACK のテストと同じ判定比（残差 / (‖A‖·max(m, n)·ε) ≤ 30）。
// 有限参照値は mpmath 1.4.1、60 桁で生成し binary64 に丸めた。
[Trait("LinearAlgebra", "SVD")]
public class SingularValueDecompositionTests
{
    /// <summary>正方行列で <c>A = UΣVᵀ</c>、<c>U</c>, <c>V</c> は正規直交、特異値は降順で非負</summary>
    [Theory]
    [MemberData(nameof(Sizes), MemberType = typeof(LinearAlgebraTestUtilities))]
    public void DecomposesSquareMatrix(int n) => Check(Random(n, n, 400 + n));

    /// <summary>縦長（QR を先に行う経路を含む）・横長の行列</summary>
    [Theory]
    [InlineData(10, 3)]
    [InlineData(3, 10)]
    [InlineData(15, 10)]
    [InlineData(10, 15)]
    [InlineData(200, 40)]
    [InlineData(40, 200)]
    [InlineData(130, 100)]
    [InlineData(1, 9)]
    [InlineData(9, 1)]
    public void DecomposesRectangularMatrix(int m, int n) => Check(Random(m, n, m * 1000 + n));

    /// <summary>指定した特異値 <c>1, 10⁻¹, …</c>（10桁にわたる）を持つ行列で、特異値の誤差が <c>‖A‖ε</c> 程度</summary>
    [Theory]
    [InlineData(20)]
    [InlineData(120)]
    public void RecoversPrescribedSingularValues(int n)
    {
        var sigma = new double[n];
        for (int i = 0; i < n; i++)
        {
            sigma[i] = Math.Pow(10, -10.0 * i / (n - 1));
        }

        var u = RandomOrthogonal(n, 31);
        var v = RandomOrthogonal(n, 32);
        var a = MultiplyNaive(MultiplyNaive(u, Matrix<double>.FromDiagonal(sigma)), v, transposeB: true);
        var svd = Check(a);
        for (int i = 0; i < n; i++)
        {
            Assert.True(Math.Abs(svd.Values[i] - sigma[i]) <= Threshold * n * Epsilon, $"{i}: {svd.Values[i]} vs {sigma[i]}");
        }
    }

    // フランク行列 n=12（ヘッセンベルグ形で最小特異値 1.1e-8）
    [Fact]
    public void MatchesFrankReference()
    {
        double[] expected =
        [
            47.73601651957576, 18.028147108362287, 10.79075108181094, 7.622462792490768, 5.898032314737142,
            4.826980899966061, 4.000138007674308, 3.211231299249944, 2.423965346545281, 1.6386865052482353,
            0.8693957405862269, 1.1186305109237777e-08,
        ];
        var svd = Check(Frank(12));
        for (int i = 0; i < 12; i++)
        {
            Assert.True(Math.Abs(svd.Values[i] - expected[i]) <= 4 * 12 * Epsilon * expected[0], $"{i}: {svd.Values[i]} vs {expected[i]}");
        }

        Assert.Equal(expected[0], svd.Values[0], 4 * Epsilon * expected[0]);
    }

    // 段階的に小さくなる上二重対角行列。二重対角行列の QR 法は各特異値を高い相対精度で求める。
    [Fact]
    public void GradedBidiagonalHasHighRelativeAccuracy()
    {
        double[] expected =
        [
            1.414213564140862, 0.00012247448725823134, 1.1547005390608457e-08, 1.118033989183133e-12,
            1.0954451153085366e-16, 4.082482890463342e-21,
        ];
        double[] d = [1, 1e-4, 1e-8, 1e-12, 1e-16, 1e-20];
        double[] e = [1, 1e-4, 1e-8, 1e-12, 1e-16];
        var a = new Matrix<double>(6, 6);
        for (int i = 0; i < 6; i++)
        {
            a[i, i] = d[i];
            if (i < 5)
            {
                a[i, i + 1] = e[i];
            }
        }

        var svd = a.Svd();
        for (int i = 0; i < 6; i++)
        {
            Assert.True(Math.Abs(svd.Values[i] - expected[i]) <= 30 * 6 * Epsilon * expected[i], $"{i}: {svd.Values[i]} vs {expected[i]}");
        }
    }

    /// <summary>対角行列（負や0を含む）の特異値は絶対値を降順に並べたものに厳密に一致する</summary>
    [Fact]
    public void DiagonalMatrixGivesExactValues()
    {
        var svd = Check(Matrix<double>.FromDiagonal([3, -7, 0, 1e-300, -2]));
        Assert.Equal(new double[] { 7, 3, 2, 1e-300, 0 }, svd.Values);
    }

    /// <summary>ランク落ちの行列で数値ランクが正しい</summary>
    [Fact]
    public void DetectsNumericalRank()
    {
        var b = Random(60, 5, 13);
        var c = Random(5, 40, 14);
        var a = MultiplyNaive(b, c);
        var svd = Check(a);
        Assert.Equal(5, svd.Rank);
        Assert.Equal(0, new Matrix<double>(4, 3).Svd().Rank);
    }

    /// <summary>
    /// 分割統治法の減次が多く起こる行列（低ランク、重複した特異値）と、再帰が深くなる大きさでも
    /// 残差と直交性が保たれる
    /// </summary>
    [Theory]
    [InlineData(0, 150)]
    [InlineData(1, 160)]
    [InlineData(2, 300)]
    public void HandlesDeflationHeavyAndLargeMatrices(int kind, int n)
    {
        Matrix<double> a;
        if (kind == 0)
        {
            // ランク 7
            a = MultiplyNaive(Random(n, 7, 61), Random(7, n, 62));
        }
        else if (kind == 1)
        {
            // 特異値が 1, 2, 3 の3種類だけ
            var sigma = Enumerable.Range(0, n).Select(i => 1.0 + i % 3).ToArray();
            a = MultiplyNaive(MultiplyNaive(RandomOrthogonal(n, 63), Matrix<double>.FromDiagonal(sigma)), RandomOrthogonal(n, 64), transposeB: true);
        }
        else
        {
            a = Random(n, n, 65);
        }

        var svd = Check(a);
        if (kind == 0)
        {
            Assert.Equal(7, svd.Rank);
        }
        else if (kind == 1)
        {
            Assert.All(svd.Values, s => Assert.True(Math.Abs(s - Math.Round(s)) <= Threshold * n * Epsilon * 3, $"{s}"));
        }
    }

    /// <summary>極端な縮尺でもオーバーフロー・アンダーフローしない</summary>
    [Theory]
    [InlineData(1e300)]
    [InlineData(1e-300)]
    [InlineData(4.9406564584124654e-324)]
    public void HandlesExtremeScaling(double scale)
    {
        var a = scale * Random(30, 25, 15);
        var svd = Check(a);
        Assert.True(svd.Values[0] > 0 && double.IsFinite(svd.Values[0]));
    }

    /// <summary>
    /// 特異ベクトルも求める分割統治法（n &gt; 25）でも、特異値と数値的なランクは縮尺どおりに変わる。
    /// 小さい行列では永年方程式の dⱼ² - σ² などがアンダーフローしていたので、二重対角行列を正規化してから解く
    /// </summary>
    [Theory]
    [InlineData(64, 64, -1000)]
    [InlineData(64, 64, -460)]
    [InlineData(128, 64, -460)]
    [InlineData(64, 128, -460)]
    [InlineData(64, 64, -300)]
    [InlineData(64, 64, 1000)]
    public void DivideAndConquerScalesExactlyWithPowersOfTwo(int m, int n, int exponent)
    {
        // 階数 min(m, n) - 8 の行列（縦長に作り、最後の8列は前の列の和）で、零特異値とランクの判定も確かめる
        int rows = Math.Max(m, n);
        int columns = Math.Min(m, n);
        var tall = Random(rows, columns, 21);
        for (int j = columns - 8; j < columns; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                tall[i, j] = tall[i, j - (columns - 8)] + tall[i, j - (columns - 9)];
            }
        }

        var a = m >= n ? tall : XMath.Transpose(tall);

        double s = Math.ScaleB(1.0, exponent);
        var v1 = a.Svd().Values;
        var svd = (s * a).Svd();
        Assert.Equal(a.Svd().Rank, svd.Rank);
        Assert.Equal(Math.Min(m, n) - 8, svd.Rank);
        for (int i = 0; i < v1.Length; i++)
        {
            Assert.Equal(v1[i], svd.Values[i] / s, Threshold * Math.Max(m, n) * Epsilon * v1[0]);
        }

        AssertRatio(OrthogonalityError(svd.U!), 1, m, "UᵀU - I");
        AssertRatio(OrthogonalityError(svd.V!), 1, n, "VᵀV - I");
    }

    /// <summary>特異値だけの計算は特異ベクトルと同時に求めた値と一致する</summary>
    [Theory]
    [InlineData(70, 70)]
    [InlineData(150, 50)]
    [InlineData(50, 150)]
    public void ValuesOnlyMatchesFullDecomposition(int m, int n)
    {
        var a = Random(m, n, 16);
        var full = a.Svd();
        var only = a.Svd(false);
        Assert.Null(only.U);
        Assert.Null(only.V);
        for (int i = 0; i < full.Values.Length; i++)
        {
            Assert.Equal(full.Values[i], only.Values[i], 30 * Math.Max(m, n) * Epsilon * full.Values[0]);
        }
    }

    /// <summary>0次、1次、零行列</summary>
    [Fact]
    public void HandlesTrivialMatrices()
    {
        Assert.Empty(new Matrix<double>(0, 0).Svd().Values);
        Assert.Empty(new Matrix<double>(3, 0).Svd().Values);
        Assert.Equal(new double[] { 5 }, new Matrix<double>(new double[,] { { -5 } }).Svd().Values);
        Check(new Matrix<double>(6, 4));
    }

    internal static Matrix<double> Frank(int n)
    {
        var a = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = Math.Max(i - 1, 0); j < n; j++)
            {
                a[i, j] = n - Math.Max(i, j);
            }
        }

        return a;
    }

    private static SingularValueDecomposition Check(Matrix<double> a)
    {
        int m = a.Rows;
        int n = a.Columns;
        int k = Math.Min(m, n);
        var svd = a.Svd();
        var u = svd.U!;
        var v = svd.V!;
        Assert.Equal(k, svd.Values.Length);
        Assert.Equal(m, u.Rows);
        Assert.Equal(k, u.Columns);
        Assert.Equal(n, v.Rows);
        Assert.Equal(k, v.Columns);
        for (int i = 0; i < k; i++)
        {
            Assert.True(svd.Values[i] >= 0);
            if (i > 0)
            {
                Assert.True(svd.Values[i - 1] >= svd.Values[i]);
            }
        }

        var usv = MultiplyNaive(MultiplyNaive(u, Matrix<double>.FromDiagonal(svd.Values)), v, transposeB: true);
        AssertRatio(Difference(a, usv), Norm(a), Math.Max(m, n), "A - UΣVᵀ");
        AssertRatio(OrthogonalityError(u), 1, m, "UᵀU - I");
        AssertRatio(OrthogonalityError(v), 1, n, "VᵀV - I");
        return svd;
    }

    // 100 ε σ_max ≈ 2.22e-14 なので σ_min=1e-15 は数値ランクに含めない。
    // ベクトルの省略で元の最大次元を失って min(m,n) ε に変わる不具合を検出する。
    [Trait("Category", "CoreAudit20261102")]
    [Theory]
    [InlineData(100, 2, true)]
    [InlineData(100, 2, false)]
    [InlineData(2, 100, true)]
    [InlineData(2, 100, false)]
    public void NumericalRankUsesOriginalDimensionsWhenVectorsAreOmitted(int rows, int columns, bool computeVectors)
    {
        var a = new Matrix<double>(rows, columns);
        a[0, 0] = 1;
        a[1, 1] = 1e-15;
        var svd = a.Svd(computeVectors);
        Assert.Equal(1.0, svd.Values[0], 1e-14);
        Assert.Equal(1e-15, svd.Values[1], 1e-28);
        Assert.Equal(1, svd.Rank);
    }
}
