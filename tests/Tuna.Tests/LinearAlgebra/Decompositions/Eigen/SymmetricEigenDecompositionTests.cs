using Tuna.LinearAlgebra;
using static Tuna.Tests.LinearAlgebra.LinearAlgebraTestUtilities;

namespace Tuna.Tests.LinearAlgebra;

// 合否は LAPACK のテストと同じ判定比（残差 / (‖A‖·n·ε) ≤ 30）。
// 有限参照値は mpmath 1.4.1、60 桁で生成し binary64 に丸めた（入力も binary64 に丸めた値を正確に変換）。
[Trait("LinearAlgebra", "SymmetricEigen")]
public class SymmetricEigenDecompositionTests
{
    /// <summary>乱数の対称行列で <c>‖AV - VΛ‖</c>、<c>‖VᵀV - I‖</c> が小さく、固有値は昇順</summary>
    [Theory]
    [MemberData(nameof(Sizes), MemberType = typeof(LinearAlgebraTestUtilities))]
    public void DecomposesRandomSymmetric(int n) => Check(RandomSymmetric(n, 300 + n));

    /// <summary>
    /// 三重対角テプリッツ行列 <c>tridiag(-1, 2, -1)</c> の固有値 <c>2 - 2cos(kπ/(n+1))</c>
    /// </summary>
    [Theory]
    [InlineData(5)]
    [InlineData(50)]
    [InlineData(150)]
    public void MatchesAnalyticTridiagonalSpectrum(int n)
    {
        var a = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++)
        {
            a[i, i] = 2;
            if (i + 1 < n)
            {
                a[i + 1, i] = -1;
                a[i, i + 1] = -1;
            }
        }

        var evd = Check(a);
        for (int k = 1; k <= n; k++)
        {
            double expected = 2 - 2 * Math.Cos(k * Math.PI / (n + 1));
            Assert.True(Math.Abs(evd.Values[k - 1] - expected) <= Threshold * n * Epsilon * 4, $"k={k}: {evd.Values[k - 1]} vs {expected}");
        }
    }

    // ヒルベルト行列 n=10（条件数 1.6e13）。固有値の絶対誤差は ‖A‖ε 程度で、小さい固有値は相対的には不正確でよい。
    [Fact]
    public void MatchesHilbertReference()
    {
        double[] expected =
        [
            1.0932524334974553e-13, 2.2667455503810733e-11, 2.1474388217975424e-09, 1.2289677387429186e-07,
            4.729689293190096e-06, 0.0001287496142763734, 0.0025308907686700287, 0.035741816271639235,
            0.3429295484835091, 1.7519196702651776,
        ];
        int n = 10;
        var a = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                a[i, j] = 1.0 / (i + j + 1);
            }
        }

        var evd = Check(a);
        for (int i = 0; i < n; i++)
        {
            Assert.True(Math.Abs(evd.Values[i] - expected[i]) <= 4 * n * Epsilon * expected[^1], $"{i}: {evd.Values[i]} vs {expected[i]}");
        }

        Assert.Equal(expected[^1], evd.Values[^1], 4 * Epsilon * expected[^1]);
    }

    // 対称パスカル行列 n=8（整数成分）
    [Fact]
    public void MatchesPascalReference()
    {
        double[] expected =
        [
            0.00022008514614167818, 0.0067202144402930325, 0.0837302458580416, 0.511891554249057,
            1.9535387753505649, 11.943115534324663, 148.8047753363649, 4543.696008254266,
        ];
        int n = 8;
        var a = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double c = 1;
                for (int k = 1; k <= i; k++)
                {
                    c = c * (j + k) / k;
                }

                a[i, j] = Math.Round(c);
            }
        }

        var evd = Check(a);
        for (int i = 0; i < n; i++)
        {
            Assert.True(Math.Abs(evd.Values[i] - expected[i]) <= 4 * n * Epsilon * expected[^1], $"{i}: {evd.Values[i]} vs {expected[i]}");
        }
    }

    /// <summary>重複・密集した固有値でも固有ベクトルは直交する</summary>
    [Theory]
    [InlineData(40)]
    [InlineData(130)]
    public void HandlesRepeatedAndClusteredEigenvalues(int n)
    {
        var spectrum = new double[n];
        for (int i = 0; i < n; i++)
        {
            spectrum[i] = i < n / 2 ? 1.0 : 1.0 + (i - n / 2) * 1e-12;
        }

        spectrum[^1] = 3.0;
        var evd = Check(SymmetricWithSpectrum(spectrum, 21));
        Array.Sort(spectrum);
        for (int i = 0; i < n; i++)
        {
            Assert.True(Math.Abs(evd.Values[i] - spectrum[i]) <= Threshold * n * Epsilon * 3, $"{i}");
        }
    }

    /// <summary>
    /// 分割統治法の減次が多く起こる行列: ウィルキンソン行列 W⁺（固有値がほぼ対で重なる）、
    /// 対角行列を弱く結合したもの（副対角が ε 程度）、副対角に0を含むもの
    /// </summary>
    [Theory]
    [InlineData(0, 101)]
    [InlineData(1, 120)]
    [InlineData(2, 300)]
    public void HandlesDeflationHeavyTridiagonals(int kind, int n)
    {
        var a = new Matrix<double>(n, n);
        var random = new Random(77 + kind);
        for (int i = 0; i < n; i++)
        {
            a[i, i] = kind switch
            {
                0 => Math.Abs(i - (n - 1) / 2),
                1 => i % 4,
                _ => random.NextDouble(),
            };
            if (i + 1 < n)
            {
                double off = kind switch
                {
                    0 => 1,
                    1 => 1e-15 * random.NextDouble(),
                    _ => i % 37 == 36 ? 0 : random.NextDouble(),
                };
                a[i + 1, i] = off;
                a[i, i + 1] = off;
            }
        }

        Check(a);
    }

    /// <summary>大きな次数（分割統治法の再帰が深い）でも残差と直交性が保たれる</summary>
    [Fact]
    public void DecomposesLargeRandomSymmetric() => Check(RandomSymmetric(400, 9));

    /// <summary>上三角部分は参照しない</summary>
    [Fact]
    public void ReadsOnlyLowerTriangle()
    {
        int n = 50;
        var a = RandomSymmetric(n, 8);
        var b = a.Clone();
        for (int j = 1; j < n; j++)
        {
            for (int i = 0; i < j; i++)
            {
                b[i, j] = double.NaN;
            }
        }

        Assert.Equal(a.SymmetricEigen().Values, b.SymmetricEigen().Values);
    }

    /// <summary>極端な縮尺では固有値が縮尺どおりに変わる（2の冪なら丸め誤差もない）</summary>
    [Theory]
    [InlineData(-1000)]
    [InlineData(1000)]
    public void ScalesExactlyWithPowersOfTwo(int exponent)
    {
        var a = RandomSymmetric(30, 4);
        double s = Math.ScaleB(1.0, exponent);
        var scaled = s * a;
        var v1 = a.SymmetricEigen(false).Values;
        var v2 = scaled.SymmetricEigen(false).Values;
        for (int i = 0; i < v1.Length; i++)
        {
            Assert.Equal(v1[i], v2[i] / s, 30 * 30 * Epsilon * Math.Abs(v1[^1]));
        }
    }

    /// <summary>
    /// 固有ベクトルも求める分割統治法（n &gt; 25）でも、固有値は縮尺どおりに変わる。
    /// 統合の減次の閾値が三重対角行列の大きさに依存しないよう、正規化してから解く（‖A‖ が 1 より小さいと固有値を失っていた）
    /// </summary>
    [Theory]
    [InlineData(-1000)]
    [InlineData(-480)]
    [InlineData(-300)]
    [InlineData(-50)]
    [InlineData(-30)]
    [InlineData(20)]
    [InlineData(1000)]
    public void DivideAndConquerScalesExactlyWithPowersOfTwo(int exponent)
    {
        // n = 200 は並列の経路（n > 100）も通る。2^-30（要素が 1e-9 程度）でも以前は相対 1e-10 の誤差があった
        int n = 200;
        var a = RandomSymmetric(n, 5);
        double s = Math.ScaleB(1.0, exponent);
        var v1 = a.SymmetricEigen().Values;
        var evd = (s * a).SymmetricEigen();
        double tolerance = Threshold * n * Epsilon * Math.Max(Math.Abs(v1[0]), Math.Abs(v1[^1]));
        for (int i = 0; i < n; i++)
        {
            Assert.Equal(v1[i], evd.Values[i] / s, tolerance);
        }

        AssertRatio(OrthogonalityError(evd.Vectors!), 1, n, "VᵀV - I");
    }

    /// <summary>固有値だけの計算は固有ベクトルと同時に求めた値と一致する</summary>
    [Fact]
    public void ValuesOnlyMatchesFullDecomposition()
    {
        var a = RandomSymmetric(90, 12);
        var full = a.SymmetricEigen();
        var only = a.SymmetricEigen(false);
        Assert.Null(only.Vectors);
        for (int i = 0; i < 90; i++)
        {
            Assert.Equal(full.Values[i], only.Values[i], 30 * 90 * Epsilon * 10);
        }
    }

    /// <summary>零行列、単位行列、対角行列（昇順に並べ替えられる）、0次と1次</summary>
    [Fact]
    public void HandlesTrivialMatrices()
    {
        Assert.All(new Matrix<double>(5, 5).SymmetricEigen().Values, v => Assert.Equal(0.0, v));
        Assert.All(Matrix<double>.Identity(5).SymmetricEigen().Values, v => Assert.Equal(1.0, v));
        Assert.Equal(new double[] { -2, 1, 3 }, Matrix<double>.FromDiagonal([3, -2, 1]).SymmetricEigen().Values);
        Assert.Empty(new Matrix<double>(0, 0).SymmetricEigen().Values);
        Assert.Equal(new double[] { 7 }, new Matrix<double>(new double[,] { { 7 } }).SymmetricEigen().Values);
        Assert.Throws<ArgumentException>(() => new Matrix<double>(2, 3).SymmetricEigen());
    }

    private static SymmetricEigenDecomposition Check(Matrix<double> a)
    {
        int n = a.Rows;
        var evd = a.SymmetricEigen();
        var v = evd.Vectors!;
        for (int i = 1; i < n; i++)
        {
            Assert.True(evd.Values[i - 1] <= evd.Values[i]);
        }

        var av = MultiplyNaive(a, v);
        var vl = MultiplyNaive(v, Matrix<double>.FromDiagonal(evd.Values));
        AssertRatio(Difference(av, vl), Norm(a), n, "AV - VΛ");
        AssertRatio(OrthogonalityError(v), 1, n, "VᵀV - I");
        return evd;
    }
}
