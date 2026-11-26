using Tuna.LinearAlgebra;
using static Tuna.Tests.LinearAlgebra.LinearAlgebraTestUtilities;

namespace Tuna.Tests.LinearAlgebra;

// 反射が 2048 個以上の大きな行列ではブロック幅を広げ、パネルを再帰的に分解する。
// 行列積で残差を作ると時間がかかるので、符号を乱数にしたベクトルへの作用で確かめる（O(mn)）。
[Trait("LinearAlgebra", "QR")]
[Collection(nameof(ParallelismCollection))]
public class QrLargeDecompositionTests
{
    /// <summary>
    /// ブロック幅 64（端数のパネルは再帰版の奇数幅）・縦長・横長で、1スレッドと全コアの結果が一致し、
    /// <c>A = QR</c>、<c>QᵀQ = I</c>
    /// </summary>
    [Theory]
    [InlineData(2085, 2085)]
    [InlineData(2300, 2050)]
    [InlineData(2050, 2300)]
    public void FactorsLargeMatrix(int m, int n)
    {
        var a = Random(m, n, m + 7 * n);
        int saved = LinearAlgebraSettings.MaxDegreeOfParallelism;
        QrDecomposition one;
        QrDecomposition many;
        try
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = 1;
            one = a.Qr();
            LinearAlgebraSettings.MaxDegreeOfParallelism = Math.Max(2, Environment.ProcessorCount);
            many = a.Qr();
        }
        finally
        {
            LinearAlgebraSettings.MaxDegreeOfParallelism = saved;
        }

        Check(a, one);
        Assert.Equal(one.R.ToArray(), many.R.ToArray());
        Assert.Equal(one.Q.ToArray(), many.Q.ToArray());
    }

    /// <summary>ブロック幅 128（反射 4096 個以上）</summary>
    [Fact]
    public void FactorsVeryLargeMatrix()
    {
        var a = Random(4100, 4096, 41);
        Check(a, a.Qr());
    }

    /// <summary>ランク落ち（0の列、同じ列の繰り返し）でも再帰的なパネル分解の <c>Q</c> は直交</summary>
    [Fact]
    public void HandlesRankDeficientLargeMatrix()
    {
        var a = Random(2100, 2048, 6);
        for (int i = 0; i < a.Rows; i++)
        {
            a[i, 5] = 0;
            a[i, 70] = a[i, 20];
            a[i, 2047] = 0;
        }

        Check(a, a.Qr());
    }

    private static void Check(Matrix<double> a, QrDecomposition qr)
    {
        int m = a.Rows;
        int n = a.Columns;
        int k = Math.Min(m, n);
        var q = qr.Q;
        var r = qr.R;
        Assert.Equal(m, q.Rows);
        Assert.Equal(k, q.Columns);
        AssertUpperTriangular(r);
        var random = new Random(m * 3 + n);
        double norm = Norm(a);
        for (int probe = 0; probe < 3; probe++)
        {
            // ‖Ax - Q(Rx)‖ / ‖x‖
            var x = Signs(n, random);
            var residual = Subtract(Apply(a, x), Apply(q, Apply(r, x)));
            AssertRatio(Length(residual) / Length(x), norm, Math.Max(m, n), "A - QR");

            // ‖Qᵀ(Qy) - y‖ / ‖y‖
            var y = Signs(k, random);
            var orthogonality = Subtract(ApplyTransposed(q, Apply(q, y)), y);
            AssertRatio(Length(orthogonality) / Length(y), 1, m, "QᵀQ - I");
        }
    }

    private static double[] Signs(int n, Random random) => Enumerable.Range(0, n).Select(_ => random.Next(2) == 0 ? -1.0 : 1.0).ToArray();

    private static double[] Apply(Matrix<double> a, double[] x)
    {
        var y = new double[a.Rows];
        for (int j = 0; j < a.Columns; j++)
        {
            for (int i = 0; i < a.Rows; i++)
            {
                y[i] += a[i, j] * x[j];
            }
        }

        return y;
    }

    private static double[] ApplyTransposed(Matrix<double> a, double[] x)
    {
        var y = new double[a.Columns];
        for (int j = 0; j < a.Columns; j++)
        {
            double sum = 0;
            for (int i = 0; i < a.Rows; i++)
            {
                sum += a[i, j] * x[i];
            }

            y[j] = sum;
        }

        return y;
    }

    private static double[] Subtract(double[] x, double[] y) => x.Zip(y, (p, q) => p - q).ToArray();

    private static double Length(double[] x) => Math.Sqrt(x.Sum(v => v * v));

    // ‖[scale,scale]‖₂=√2 scale は有限。反射の beta-alpha だけが先に溢れる。
    // 因子の符号には依存せず、単位ノルムと縮尺した再構成を固定入力で確認する。
    [Trait("Category", "CoreAudit20261102")]
    [Theory]
    [InlineData(1e300)]
    [InlineData(8e307)]
    [InlineData(1e308)]
    public void NearMaximumFiniteColumnKeepsFiniteOrthogonalFactors(double scale)
    {
        var a = new Matrix<double>(new double[,] { { scale }, { scale } });
        var qr = a.Qr();
        Assert.True(double.IsFinite(qr.R[0, 0]));
        Assert.True(double.IsFinite(qr.Q[0, 0]) && double.IsFinite(qr.Q[1, 0]));
        Assert.Equal(1.0, qr.Q[0, 0] * qr.Q[0, 0] + qr.Q[1, 0] * qr.Q[1, 0], 2e-14);
        Assert.Equal(1.0, qr.Q[0, 0] * (qr.R[0, 0] / scale), 2e-14);
        Assert.Equal(1.0, qr.Q[1, 0] * (qr.R[0, 0] / scale), 2e-14);
    }

    // 溢れを避ける縮尺は大きい列だけに掛ける。独立な小さい列・要素は0へ丸めず、そのまま R と解に残る
    [Theory]
    [InlineData(1e300, 1e-300)]
    [InlineData(1e-300, 1e300)]
    [InlineData(1e308, 1e-300)]
    [InlineData(1e-308, 1e308)]
    public void LargeColumnScalingKeepsIndependentSmallDiagonal(double d0, double d1)
    {
        var qr = Matrix<double>.FromDiagonal([d0, d1]).Qr();
        Assert.Equal(d0, Math.Abs(qr.R[0, 0]));
        Assert.Equal(d1, Math.Abs(qr.R[1, 1]));
        Assert.Equal([1.0, 1.0], qr.Solve([d0, d1]));
    }

    // 溢れない入力は縮めないので、溢れる大きさに近い列の中の小さい要素も R にそのまま残る（上三角なので R = A）
    [Theory]
    [InlineData(2, 1e-320)]
    [InlineData(2, 1e-307)]
    [InlineData(128, 1e-307)]
    [InlineData(2048, 1e-307)]
    public void SmallEntryInsideLargeColumnIsKeptWithoutOverflow(int rows, double small)
    {
        var a = new Matrix<double>(rows, 2);
        a[0, 0] = 1;
        a[0, 1] = 1e308;
        a[1, 1] = small;
        var qr = a.Qr();
        Assert.Equal(1.0, qr.R[0, 0]);
        Assert.Equal(1e308, qr.R[0, 1]);
        Assert.Equal(small, qr.R[1, 1]);
    }

    // 縮めずに分解すると反射の更新が溢れる入力（2列目の w = τ(c₀ + v·c₁) ≈ 2.4e308）。溢れた列だけ縮めてやり直す
    [Theory]
    [InlineData(1e308, 1.4142135623730951e308, 0.0)]
    [InlineData(5e307, 1.0606601717798211e308, 3.5355339059327378e307)]
    public void OverflowingReflectionUpdateFallsBackToColumnScaling(double a11, double r01, double r11)
    {
        var a = new Matrix<double>(new double[,] { { 1e308, 1e308 }, { 1e308, a11 } });
        var qr = a.Qr();
        Assert.Equal(1.4142135623730951e308, Math.Abs(qr.R[0, 0]), 1e294);
        Assert.Equal(r01, Math.Abs(qr.R[0, 1]), 1e294);
        Assert.Equal(r11, Math.Abs(qr.R[1, 1]), 1e294);
        foreach (double q in qr.Q.ToArray())
        {
            Assert.Equal(Math.Sqrt(0.5), Math.Abs(q), 1e-15);
        }
    }

    [Fact]
    public void ColumnScalingFallbackKeepsSmallEntryInOtherColumn()
    {
        // 1・2列目だけが溢れて縮められ、3列目の非正規数はそのまま残る
        var a = new Matrix<double>(new double[,] { { 1e308, 1e308, 0 }, { 1e308, 1e308, 0 }, { 0, 0, 1e-320 } });
        var qr = a.Qr();
        Assert.Equal(1.4142135623730951e308, Math.Abs(qr.R[0, 0]), 1e294);
        Assert.Equal(1.4142135623730951e308, Math.Abs(qr.R[0, 1]), 1e294);
        Assert.True(Math.Abs(qr.R[1, 1]) < 1e294);
        Assert.Equal(1e-320, Math.Abs(qr.R[2, 2]));
    }

    [Fact]
    public void NearMaximumFiniteColumnsKeepFiniteFactors()
    {
        var a = new Matrix<double>(new double[,] { { 1e308, 1e308 }, { 1e308, -1e308 } });
        var qr = a.Qr();
        Assert.Equal(Math.Sqrt(2) * 1e308, Math.Abs(qr.R[0, 0]), 1e294);
        Assert.Equal(Math.Sqrt(2) * 1e308, Math.Abs(qr.R[1, 1]), 1e294);
        Assert.True(Math.Abs(qr.R[0, 1]) < 1e294);
        foreach (double q in qr.Q.ToArray())
        {
            Assert.Equal(Math.Sqrt(0.5), Math.Abs(q), 1e-15);
        }
    }

    // 反射の更新がオーバーフローして再試行する経路で、逐次作成・遅延作成と入力上書きの全組み合わせを検証する（横長・ランク落ち・微小な別列も含む）。
    [Trait("Category", "IntegrationCoverage20261102")]
    [Theory]
    [InlineData(0, false, false)]
    [InlineData(0, false, true)]
    [InlineData(0, true, false)]
    [InlineData(0, true, true)]
    [InlineData(1, false, false)]
    [InlineData(1, false, true)]
    [InlineData(1, true, false)]
    [InlineData(1, true, true)]
    [InlineData(2, false, false)]
    [InlineData(2, false, true)]
    [InlineData(2, true, false)]
    [InlineData(2, true, true)]
    [InlineData(3, false, false)]
    [InlineData(3, false, true)]
    [InlineData(3, true, false)]
    [InlineData(3, true, true)]
    public void OverflowRetryPreservesFactorsAcrossComputationAndOverwriteModes(int sample, bool factorsOnly, bool overwrite)
    {
        double[,] values = sample switch
        {
            0 => new double[,] { { 1e308, 1e308 }, { 1e308, 1e308 } },
            1 => new double[,] { { 1e308, 1e308 }, { 1e308, 5e307 } },
            2 => new double[,] { { 1e308, 1e308, 0 }, { 1e308, 1e308, 0 }, { 0, 0, 1e-320 } },
            3 => new double[,] { { 1e308, 1e308, 2 }, { 1e308, 5e307, -1 } },
            _ => throw new ArgumentOutOfRangeException(nameof(sample))
        };
        var a = new Matrix<double>(values);
        var qr = factorsOnly ? QrDecomposition.ComputeFactors(a, overwrite) : QrDecomposition.Compute(a, overwrite);
        var q = qr.Q;
        var r = qr.R;
        Assert.All(q.ToArray().Cast<double>(), value => Assert.True(double.IsFinite(value)));
        Assert.All(r.ToArray().Cast<double>(), value => Assert.True(double.IsFinite(value)));
        if (!overwrite)
        {
            Assert.Equal(values, a.ToArray());
        }

        if (sample == 2)
        {
            Assert.Equal(1e-320, Math.Abs(r[2, 2]));
        }

        for (int i = 0; i < q.Columns; i++)
        {
            for (int j = 0; j < q.Columns; j++)
            {
                double dot = 0;
                for (int k = 0; k < q.Rows; k++)
                {
                    dot += q[k, i] * q[k, j];
                }

                Assert.True(Math.Abs(dot - (i == j ? 1 : 0)) <= 2e-12);
            }
        }

        // 行列全体のノルムで小さい列の誤りを隠さず、各列を割ってから再構成する。
        for (int j = 0; j < r.Columns; j++)
        {
            double scale = 0;
            for (int i = 0; i < q.Rows; i++)
            {
                scale = Math.Max(scale, Math.Abs(values[i, j]));
            }

            for (int i = 0; i < q.Rows; i++)
            {
                double actual = 0;
                for (int k = 0; k < q.Columns; k++)
                {
                    actual += q[i, k] * (r[k, j] / scale);
                }

                Assert.True(Math.Abs(actual - values[i, j] / scale) <= 2e-12,
                    $"sample {sample}, ({i},{j}): scaled reconstruction {actual:R}");
            }
        }
    }

    // A*[1/4,1/2] = [7.5e307,5e307]。Aの分解には再試行が必要だが、係数・右辺・解は有限。
    [Trait("Category", "IntegrationCoverage20261102")]
    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void OverflowRetryKeepsSolveAndOverlappingRightHandSideUsable(bool factorsOnly, bool overwrite)
    {
        var a = new Matrix<double>(new double[,] { { 1e308, 1e308 }, { 1e308, 5e307 } });
        var qr = factorsOnly ? QrDecomposition.ComputeFactors(a, overwrite) : QrDecomposition.Compute(a, overwrite);
        double[] b = [7.5e307, 5e307];
        Check(qr.Solve(b));
        double[] destination = new double[2];
        qr.Solve(b, destination);
        Check(destination);
        Assert.Equal(new[] { 7.5e307, 5e307 }, b);
        double[] shared = (double[])b.Clone();
        qr.Solve(shared, shared);
        Check(shared);

        static void Check(double[] actual)
        {
            Assert.Equal(0.25, actual[0], 2e-12);
            Assert.Equal(0.5, actual[1], 2e-12);
        }
    }
}
