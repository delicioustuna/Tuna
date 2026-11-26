using System.Numerics;
using Tuna.LinearAlgebra;
using static Tuna.Tests.LinearAlgebra.LinearAlgebraTestUtilities;

namespace Tuna.Tests.LinearAlgebra;

// 合否は LAPACK のテストと同じ判定比（‖AV - VD‖ / (‖A‖·n·ε) ≤ 30、固有ベクトルはノルム1）。
// 有限参照値は mpmath 1.4.1、60 桁で生成し binary64 に丸めた。
[Trait("LinearAlgebra", "Eigen")]
public class EigenDecompositionTests
{
    /// <summary>乱数の非対称行列（複素固有値を含む）で <c>‖AV - VD‖</c> が小さい</summary>
    [Theory]
    [MemberData(nameof(Sizes), MemberType = typeof(LinearAlgebraTestUtilities))]
    public void DecomposesRandomMatrix(int n) => Check(Random(n, n, 500 + n));

    /// <summary>
    /// 積極的早期収縮と複数バルジの掃引が働く大きさ（n ≥ 75）で、密集した固有値、ほぼ欠陥のある行列、
    /// 既にヘッセンベルグ形の行列でも <c>‖AV - VD‖</c> が小さい
    /// </summary>
    [Theory]
    [InlineData(0, 300)]
    [InlineData(1, 160)]
    [InlineData(2, 140)]
    [InlineData(3, 120)]
    public void DecomposesLargeAndStructuredMatrices(int kind, int n)
    {
        var random = new Random(70 + kind);
        var a = new Matrix<double>(n, n);
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                a[i, j] = kind switch
                {
                    0 => 2 * random.NextDouble() - 1,
                    1 => (i == j ? Math.Floor(3 * random.NextDouble()) : 0) + 1e-8 * (2 * random.NextDouble() - 1),
                    2 => j >= i ? 1 : (i == j + 1 ? 1e-3 : 0),
                    _ => j >= i - 1 ? 2 * random.NextDouble() - 1 : 0,
                };
            }
        }

        Check(a);
    }

    /// <summary>巡回置換行列の固有値は1の n 乗根</summary>
    [Theory]
    [InlineData(2)]
    [InlineData(7)]
    [InlineData(32)]
    public void CyclicPermutationGivesRootsOfUnity(int n)
    {
        var a = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++)
        {
            a[(i + 1) % n, i] = 1;
        }

        var evd = Check(a);
        foreach (var lambda in evd.Values)
        {
            Assert.Equal(1.0, lambda.Magnitude, 100 * n * Epsilon);
            var power = Complex.Pow(lambda, n);
            Assert.True(Complex.Abs(power - 1) <= 100 * n * n * Epsilon, $"{lambda}^{n} = {power}");
        }

        // 共役な組は虚部が正のものが先
        for (int i = 0; i < n; i++)
        {
            if (evd.Values[i].Imaginary > 0)
            {
                Assert.Equal(Complex.Conjugate(evd.Values[i]), evd.Values[i + 1]);
            }
        }
    }

    /// <summary>上三角行列（平衡化の置換だけで分解される）の固有値は対角成分に厳密に一致する</summary>
    [Fact]
    public void TriangularMatrixGivesDiagonalExactly()
    {
        int n = 20;
        var a = Random(n, n, 41);
        for (int j = 0; j < n; j++)
        {
            for (int i = j + 1; i < n; i++)
            {
                a[i, j] = 0;
            }
        }

        var evd = Check(a);
        var expected = Enumerable.Range(0, n).Select(i => a[i, i]).Order().ToArray();
        var actual = evd.Values.Select(v => v.Real).Order().ToArray();
        Assert.Equal(expected, actual);
        Assert.All(evd.Values, v => Assert.Equal(0.0, v.Imaginary));
    }

    // フランク行列 n=12。固有値は正の実数で、小さいものほど条件が悪い。
    [Fact]
    public void MatchesFrankReference()
    {
        double[] expected =
        [
            0.031028060644010015, 0.049507429185278305, 0.08122765924040504, 0.14364651976922047,
            0.2847497205584782, 0.6435053190048554, 1.553988709132107, 3.5118559485807572,
            6.9615330855671225, 12.311077400868527, 20.19898864587708, 32.228891501572164,
        ];
        var evd = Check(SingularValueDecompositionTests.Frank(12));
        var actual = evd.Values.OrderBy(v => v.Real).ToArray();
        for (int i = 0; i < 12; i++)
        {
            // 大きい固有値は条件がよいので ε 程度、最小のものでも 1e-8 程度の相対誤差に収まる
            double tol = i >= 6 ? 1000 * Epsilon * expected[i] : 1e-6 * expected[i];
            Assert.True(Complex.Abs(actual[i] - expected[i]) <= tol, $"{i}: {actual[i]} vs {expected[i]}");
        }
    }

    /// <summary>対称行列では対称用の分解と同じ実固有値を返す</summary>
    [Fact]
    public void SymmetricInputMatchesSymmetricSolver()
    {
        var a = RandomSymmetric(60, 42);
        var general = Check(a).Values.Select(v => v.Real).Order().ToArray();
        var symmetric = a.SymmetricEigen(false).Values;
        for (int i = 0; i < 60; i++)
        {
            Assert.Equal(symmetric[i], general[i], 30 * 60 * Epsilon * Math.Abs(symmetric[^1]));
        }
    }

    /// <summary>
    /// 成分の大きさが極端に異なる行列（<c>DAD⁻¹</c>、<c>D = diag(2^{10i})</c>）でも平衡化により同じ固有値が求まる
    /// </summary>
    [Fact]
    public void BalancingHandlesBadlyScaledMatrix()
    {
        int n = 12;
        var a = Random(n, n, 43);
        var scaled = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                scaled[i, j] = Math.ScaleB(a[i, j], 10 * (i - j));
            }
        }

        var expected = a.Eigen(false).Values.OrderBy(v => v.Real).ThenBy(v => v.Imaginary).ToArray();
        var evd = scaled.Eigen();
        var actual = evd.Values.OrderBy(v => v.Real).ThenBy(v => v.Imaginary).ToArray();
        for (int i = 0; i < n; i++)
        {
            Assert.True(Complex.Abs(actual[i] - expected[i]) <= 1e-12, $"{i}: {actual[i]} vs {expected[i]}");
        }

        // 固有対の残差は各固有ベクトル（ノルム1）について相対的に小さい
        Assert.True(EigenResidual(scaled, evd) <= 1e-10 * Norm(scaled));
    }

    /// <summary>2×2 の回転行列の固有値は <c>cos θ ± i sin θ</c></summary>
    [Fact]
    public void RotationMatrixHasComplexPair()
    {
        double theta = 0.3;
        var a = new Matrix<double>(new double[,] { { Math.Cos(theta), -Math.Sin(theta) }, { Math.Sin(theta), Math.Cos(theta) } });
        var evd = Check(a);
        Assert.Equal(Math.Cos(theta), evd.Values[0].Real, 1e-15);
        Assert.Equal(Math.Sin(theta), evd.Values[0].Imaginary, 1e-15);
        Assert.Equal(-Math.Sin(theta), evd.Values[1].Imaginary, 1e-15);
    }

    /// <summary>ジョルダンブロック（欠陥行列）の固有値は <c>ε^(1/k)</c> 程度の誤差で求まる</summary>
    [Fact]
    public void DefectiveMatrixEigenvaluesAreAccurateToRootEpsilon()
    {
        int n = 4;
        var a = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++)
        {
            a[i, i] = 2;
            if (i + 1 < n)
            {
                a[i, i + 1] = 1;
            }
        }

        var evd = a.Eigen();
        foreach (var lambda in evd.Values)
        {
            Assert.True(Complex.Abs(lambda - 2) <= 10 * Math.Pow(Epsilon, 1.0 / n), $"{lambda}");
        }
    }

    /// <summary>固有値だけの計算は固有ベクトルと同時に求めた値と一致する</summary>
    [Fact]
    public void ValuesOnlyMatchesFullDecomposition()
    {
        var a = Random(80, 80, 44);
        var full = a.Eigen().Values.OrderBy(v => v.Real).ThenBy(v => v.Imaginary).ToArray();
        var evd = a.Eigen(false);
        Assert.Null(evd.Vectors);
        var only = evd.Values.OrderBy(v => v.Real).ThenBy(v => v.Imaginary).ToArray();
        for (int i = 0; i < 80; i++)
        {
            Assert.True(Complex.Abs(full[i] - only[i]) <= 1e-12, $"{i}: {full[i]} vs {only[i]}");
        }
    }

    /// <summary>極端な縮尺でもオーバーフロー・アンダーフローしない</summary>
    [Theory]
    [InlineData(1e300)]
    [InlineData(1e-300)]
    public void HandlesExtremeScaling(double scale)
    {
        var a = Random(25, 25, 45);
        var expected = a.Eigen(false).Values.OrderBy(v => v.Real).ThenBy(v => v.Imaginary).ToArray();
        var actual = (scale * a).Eigen().Values.OrderBy(v => v.Real).ThenBy(v => v.Imaginary).ToArray();
        for (int i = 0; i < 25; i++)
        {
            Assert.True(Complex.Abs(actual[i] / scale - expected[i]) <= 1e-12, $"{i}: {actual[i] / scale} vs {expected[i]}");
        }
    }

    /// <summary>0次、1次、零行列、単位行列</summary>
    [Fact]
    public void HandlesTrivialMatrices()
    {
        Assert.Empty(new Matrix<double>(0, 0).Eigen().Values);
        Assert.Equal(new Complex(-3, 0), new Matrix<double>(new double[,] { { -3 } }).Eigen().Values[0]);
        Assert.All(Check(new Matrix<double>(5, 5)).Values, v => Assert.Equal(Complex.Zero, v));
        Assert.All(Check(Matrix<double>.Identity(5)).Values, v => Assert.Equal(Complex.One, v));
        Assert.Throws<ArgumentException>(() => new Matrix<double>(2, 3).Eigen());
    }

    private static EigenDecomposition Check(Matrix<double> a)
    {
        int n = a.Rows;
        var evd = a.Eigen();
        Assert.NotNull(evd.Vectors);
        for (int j = 0; j < n; j++)
        {
            var v = evd.GetVector(j);
            double norm = Math.Sqrt(v.Sum(c => c.Real * c.Real + c.Imaginary * c.Imaginary));
            Assert.Equal(1.0, norm, 1e-13);
        }

        AssertRatio(EigenResidual(a, evd), Norm(a), n, "AV - VD");
        return evd;
    }
}
