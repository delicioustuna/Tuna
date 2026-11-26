using static Tuna.Tests.LinearAlgebra.LinearAlgebraTestUtilities;

namespace Tuna.Tests.LinearAlgebra;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む
using Tuna.LinearAlgebra;

// 条件数の逆数の推定（LU・コレスキー・バッチのコレスキー）と反復改良。
// 推定は逆行列から求めた真の値と比べる（推定は ‖A⁻¹‖₁ の下からの推定なので、条件数の逆数は真の値以上で、通常は3倍以内）。
[Trait("LinearAlgebra", "Condition")]
public class ConditionTests
{
    /// <summary>逆行列から求めた 1-ノルムの条件数の逆数</summary>
    private static double ExactReciprocalCondition(Matrix<double> a) =>
        1 / (Norm1(a) * Norm1(a.Lu().Inverse()));

    private static double Norm1(Matrix<double> a)
    {
        double norm = 0;
        for (int j = 0; j < a.Columns; j++)
        {
            double sum = 0;
            for (int i = 0; i < a.Rows; i++)
            {
                sum += Math.Abs(a[i, j]);
            }

            norm = Math.Max(norm, sum);
        }

        return norm;
    }

    /// <summary>ヒルベルト行列 1/(i+j+1) に lcm(1, …, 2n-1) を掛けた整数の行列（要素も積和も倍精度で厳密に表せる）</summary>
    private static Matrix<double> ScaledHilbert(int n)
    {
        long lcm = 1;
        for (long k = 2; k <= 2 * n - 1; k++)
        {
            lcm = lcm / Gcd(lcm, k) * k;
        }

        var a = new Matrix<double>(n, n);
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                a[i, j] = lcm / (i + j + 1);
            }
        }

        return a;
    }

    private static long Gcd(long a, long b) => b == 0 ? a : Gcd(b, a % b);

    /// <summary>厳密な解 1, 2, …, n と、それに対する厳密な右辺</summary>
    private static (Vector<double> Solution, Vector<double> RightHandSide) ExactSystem(Matrix<double> a)
    {
        int n = a.Rows;
        var x = new Vector<double>(n);
        var b = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            x[i] = i + 1;
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                b[i] += a[i, j] * x[j];
            }
        }

        return (x, b);
    }

    private static double RelativeError(Vector<double> x, Vector<double> expected)
    {
        double error = 0, scale = 0;
        for (int i = 0; i < x.Length; i++)
        {
            error = Math.Max(error, Math.Abs(x[i] - expected[i]));
            scale = Math.Max(scale, Math.Abs(expected[i]));
        }

        return error / scale;
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 2)]
    [InlineData(5, 3)]
    [InlineData(40, 4)]
    [InlineData(200, 5)]
    public void EstimatesAreCloseToExactReciprocalConditionNumbers(int n, int seed)
    {
        var a = Random(n, n, 600 + seed);
        double exact = ExactReciprocalCondition(a);
        double estimate = a.Lu().EstimateReciprocalConditionNumber();
        Assert.InRange(estimate, exact * (1 - 1e-10), 3 * exact);

        var spd = RandomPositiveDefinite(n, 610 + seed);
        double exactSpd = ExactReciprocalCondition(spd);
        double estimateSpd = spd.Cholesky().EstimateReciprocalConditionNumber();
        Assert.InRange(estimateSpd, exactSpd * (1 - 1e-10), 3 * exactSpd);
    }

    [Theory]
    [InlineData(1e-2)]
    [InlineData(1e-8)]
    [InlineData(1e-14)]
    public void DetectsIllConditionedSymmetricMatrices(double smallest)
    {
        // 固有値 1 … smallest の対称正定値行列（2-ノルムの条件数 1/smallest）。
        // 比べる真の値も逆行列から求めるので、悪条件ほど真の値自体に条件数 × ε 程度の誤差がある
        var eigenvalues = Enumerable.Range(0, 30).Select(i => Math.Pow(smallest, i / 29.0)).ToArray();
        var a = SymmetricWithSpectrum(eigenvalues, 620);
        double exact = ExactReciprocalCondition(a);
        Assert.InRange(a.Cholesky().EstimateReciprocalConditionNumber(), exact * 0.9, 3 * exact);
        Assert.InRange(a.Lu().EstimateReciprocalConditionNumber(), exact * 0.9, 3 * exact);
    }

    [Fact]
    public void DiagonalAndSpecialMatrices()
    {
        Assert.Equal(1.0, Matrix<double>.Identity(7).Lu().EstimateReciprocalConditionNumber());
        Assert.Equal(1.0, Matrix<double>.Identity(7).Cholesky().EstimateReciprocalConditionNumber());

        // 対角行列では推定は厳密
        var d = Matrix<double>.FromDiagonal([4, 1e-6, 2]);
        Assert.Equal(1e-6 / 4, d.Lu().EstimateReciprocalConditionNumber(), 1e-20);
        Assert.Equal(1e-6 / 4, d.Cholesky().EstimateReciprocalConditionNumber(), 1e-20);

        // 次数 0 は 1、特異は 0、NaN を含めば NaN
        Assert.Equal(1.0, new Matrix<double>(0, 0).Lu().EstimateReciprocalConditionNumber());
        Assert.Equal(1.0, new Matrix<double>(0, 0).Cholesky().EstimateReciprocalConditionNumber());
        Assert.Equal(0.0, new Matrix<double>(3, 3).Lu().EstimateReciprocalConditionNumber());
        var withNaN = Random(3, 3, 630);
        withNaN[1, 2] = double.NaN;
        Assert.True(double.IsNaN(withNaN.Lu().EstimateReciprocalConditionNumber()));

        // 数値的に特異（推定の途中で溢れる）なら 0
        var nearlySingular = new Matrix<double>(2, 2);
        nearlySingular[0, 0] = 1;
        nearlySingular[1, 0] = 1;
        nearlySingular[0, 1] = 1;
        nearlySingular[1, 1] = 1 + 1e-300;
        Assert.Equal(0.0, nearlySingular.Lu().EstimateReciprocalConditionNumber());
    }

    [Fact]
    public void CholeskyEstimateUsesOnlyTheLowerTriangleOfTheInput()
    {
        var a = RandomPositiveDefinite(20, 640);
        double expected = a.Cholesky().EstimateReciprocalConditionNumber();
        var garbage = a.Clone();
        for (int j = 1; j < 20; j++)
        {
            for (int i = 0; i < j; i++)
            {
                garbage[i, j] = 1e10;
            }
        }

        Assert.Equal(expected, garbage.Cholesky().EstimateReciprocalConditionNumber());

        // 入力の領域に L を上書きしても、分解の前の 1-ノルムで推定する
        var overwritten = a.Clone();
        Assert.Equal(expected, CholeskyDecomposition.Compute(overwritten, overwritten).EstimateReciprocalConditionNumber());
        var lu = LuDecomposition.Compute(a.Clone(), overwriteInput: true);
        Assert.Equal(a.Lu().EstimateReciprocalConditionNumber(), lu.EstimateReciprocalConditionNumber());
    }

    [Theory]
    [InlineData(3, 7)]
    [InlineData(40, 5)]
    [InlineData(170, 3)]
    public void BatchEstimatesMatchSingleMatrixEstimates(int n, int count)
    {
        var matrices = Enumerable.Range(0, count).Select(k => RandomPositiveDefinite(n, 650 + k)).ToArray();
        matrices[1][0, 0] = -1;
        Assert.False(BatchCholeskyDecomposition.TryCompute(Tensor<double>.FromMatrices(matrices), out var chol));
        var estimates = chol.EstimateReciprocalConditionNumbers();
        Assert.Equal(count, estimates.Length);
        Assert.True(double.IsNaN(estimates[1]));
        for (int k = 0; k < count; k++)
        {
            if (k != 1)
            {
                Assert.Equal(matrices[k].Cholesky().EstimateReciprocalConditionNumber(), estimates[k]);
            }
        }

        Assert.Throws<ArgumentException>(() => chol.EstimateReciprocalConditionNumbers(new Vector<double>(count + 1)));
    }

    [Theory]
    [InlineData(8)]
    [InlineData(10)]
    public void RefinementRemovesTheConditionNumberFromTheError(int n)
    {
        // ヒルベルト行列（n = 10 で条件数 3.5×10¹³）。改良しない解の相対誤差は条件数 × ε 程度
        var a = ScaledHilbert(n);
        var (expected, b) = ExactSystem(a);

        var lu = a.Lu();
        var x = lu.Solve(b);
        double before = RelativeError(x, expected);
        Assert.True(before > 1e-10, $"before = {before}");
        double backwardError = lu.Refine(a, b, x);
        Assert.InRange(RelativeError(x, expected), 0, 1e-15);
        Assert.InRange(backwardError, 0, 1e-15);

        var chol = a.Cholesky();
        var y = chol.Solve(b);
        Assert.True(RelativeError(y, expected) > 1e-10);
        Assert.InRange(chol.Refine(a, b, y), 0, 1e-15);
        Assert.InRange(RelativeError(y, expected), 0, 1e-15);
    }

    [Fact]
    public void RefinementOfWellConditionedSystemsKeepsAccurateSolutions()
    {
        var a = Random(50, 50, 660);
        var b = Random(50, 1, 661).Column(0).ToVector();
        var lu = a.Lu();
        var x = lu.Solve(b);
        Assert.InRange(lu.Refine(a, b, x), 0, 1e-15);
        var residual = XMath.Subtract(b, XMath.Multiply(a, x));
        Assert.InRange(XMath.NormInfinity<double>(residual), 0, 1e-13);
    }

    [Fact]
    public void ZeroIterationsOnlyMeasureTheBackwardError()
    {
        var a = ScaledHilbert(10);
        var (_, b) = ExactSystem(a);
        var lu = a.Lu();
        var x = lu.Solve(b);
        var unchanged = x.Clone();
        double error = lu.Refine(a, b, x, maxIterations: 0);
        Assert.Equal(unchanged.ToArray(), x.ToArray());
        Assert.True(error > 0);
        Assert.True(lu.Refine(a, b, x) <= error);
    }

    [Fact]
    public void MatrixRefinementMatchesColumnByColumnRefinement()
    {
        var a = ScaledHilbert(9);
        var b = Random(9, 4, 670);
        var lu = a.Lu();
        var x = lu.Solve(b);
        var columns = x.Clone();
        double error = lu.Refine(a, b, x);
        double max = 0;
        for (int c = 0; c < 4; c++)
        {
            var column = columns.Column(c).ToVector();
            max = Math.Max(max, lu.Refine(a, b.Column(c), column));
            Assert.Equal(column.ToArray(), x.Column(c).ToArray());
        }

        Assert.Equal(max, error);

        // コレスキーは係数の下三角部分だけを読む。転置ビューの係数は写してから読むが、結果は同じ
        var chol = a.Cholesky();
        var y = chol.Solve(b);
        var z = y.Clone();
        var upperGarbage = a.Clone();
        for (int j = 1; j < 9; j++)
        {
            for (int i = 0; i < j; i++)
            {
                upperGarbage[i, j] = double.NaN;
            }
        }

        chol.Refine(a, b, y);
        chol.Refine(upperGarbage, b, z);
        Assert.Equal(y.ToArray(), z.ToArray());

        var w = lu.Solve(b);
        lu.Refine(a.T.ToMatrix().T, b, w);
        Assert.Equal(x.ToArray(), w.ToArray());
    }

    [Fact]
    public void RefinementRejectsInvalidArguments()
    {
        var a = Random(4, 4, 680);
        var b = Random(4, 2, 681);
        var lu = a.Lu();
        var x = lu.Solve(b);
        Assert.Throws<ArgumentException>(() => lu.Refine(Random(3, 3, 682), b, x));
        Assert.Throws<ArgumentException>(() => lu.Refine(a, b, new Matrix<double>(4, 3)));
        Assert.Throws<ArgumentException>(() => lu.Refine(a, b, b));
        Assert.Throws<ArgumentException>(() => lu.Refine(a, a, a));
        Assert.Throws<ArgumentOutOfRangeException>(() => lu.Refine(a, b, x, -1));
        Assert.Throws<InvalidOperationException>(() => new Matrix<double>(4, 4).Lu().Refine(a, b, x));

        var chol = RandomPositiveDefinite(4, 683).Cholesky();
        Assert.Throws<ArgumentException>(() => chol.Refine(a, b, chol.L[.., 0..2]));

        // 入力の領域を因子の格納に使った LU では、解をその領域に置けない
        var overwritten = a.Clone();
        var luInPlace = LuDecomposition.Compute(overwritten, overwriteInput: true);
        Assert.Throws<ArgumentException>(() => luInPlace.Refine(a, b, overwritten[.., 0..2]));
        Assert.Equal(0.0, new Matrix<double>(0, 0).Lu().Refine(new Matrix<double>(0, 0), new Matrix<double>(0, 2), new Matrix<double>(0, 2)));
    }

    // A=[scale], b=[scale] の厳密解は1。残差と |A*x| が有限でも、
    // |b|+|A*x| だけが溢れて後退誤差を0と誤判定する問題を、尺度が小さい対照と比較する。
    [Trait("Category", "CoreAuditFollowup20261102")]
    [Theory]
    [InlineData(false, false, 3e307, 2.0, 0.3333333333333333)]
    [InlineData(false, false, 6e307, 2.0, 0.3333333333333333)]
    [InlineData(false, false, 3e307, 1.5, 0.2)]
    [InlineData(false, false, 1e308, 1.5, 0.2)]
    [InlineData(false, true, 3e307, 2.0, 0.3333333333333333)]
    [InlineData(false, true, 6e307, 2.0, 0.3333333333333333)]
    [InlineData(false, true, 3e307, 1.5, 0.2)]
    [InlineData(false, true, 1e308, 1.5, 0.2)]
    [InlineData(true, false, 3e307, 2.0, 0.3333333333333333)]
    [InlineData(true, false, 6e307, 2.0, 0.3333333333333333)]
    [InlineData(true, false, 3e307, 1.5, 0.2)]
    [InlineData(true, false, 1e308, 1.5, 0.2)]
    [InlineData(true, true, 3e307, 2.0, 0.3333333333333333)]
    [InlineData(true, true, 6e307, 2.0, 0.3333333333333333)]
    [InlineData(true, true, 3e307, 1.5, 0.2)]
    [InlineData(true, true, 1e308, 1.5, 0.2)]
    public void BackwardErrorRemainsScaleInvariantWhenDenominatorOverflows(
        bool useCholesky, bool matrixRightHandSide, double scale, double initial, double expected)
    {
        var a = Matrix<double>.FromDiagonal([scale]);
        double error;
        if (matrixRightHandSide)
        {
            var b = new Matrix<double>(new double[,] { { scale, scale } });
            var x = new Matrix<double>(new double[,] { { initial, 1.0 } });
            error = useCholesky ? a.Cholesky().Refine(a, b, x, maxIterations: 0)
                : a.Lu().Refine(a, b, x, maxIterations: 0);
            Assert.Equal(initial, x[0, 0]);
            Assert.Equal(1.0, x[0, 1]);
        }
        else
        {
            var b = new Vector<double>([scale]);
            var x = new Vector<double>([initial]);
            error = useCholesky ? a.Cholesky().Refine(a, b, x, maxIterations: 0)
                : a.Lu().Refine(a, b, x, maxIterations: 0);
            Assert.Equal(initial, x[0]);
        }

        Assert.Equal(expected, error, 2e-14);
    }

    // 1×1の非零行列の条件数は尺度によらず厳密に1。逆ノルム自体は有限。
    [Trait("Category", "AuditRegression")]
    [Trait("Category", "CoreAuditIdentities20261102")]
    [Theory]
    [InlineData("LU", 1e+300)]
    [InlineData("LU", 1.7e+308)]
    [InlineData("LU", 1.7976931348623155e+308)]
    [InlineData("LU", 1.7976931348623157e+308)]
    [InlineData("Cholesky", 1e+300)]
    [InlineData("Cholesky", 1.7e+308)]
    [InlineData("Cholesky", 1.7976931348623155e+308)]
    [InlineData("Cholesky", 1.7976931348623157e+308)]
    [InlineData("BatchCholesky", 1e+300)]
    [InlineData("BatchCholesky", 1.7e+308)]
    [InlineData("BatchCholesky", 1.7976931348623155e+308)]
    [InlineData("BatchCholesky", 1.7976931348623157e+308)]
    public void ReciprocalConditionRemainsOneNearTheLargestFiniteScale(string decomposition, double scale)
    {
        var a = Matrix<double>.FromDiagonal([scale]);
        double actual = decomposition switch
        {
            "LU" => a.Lu().EstimateReciprocalConditionNumber(),
            "Cholesky" => a.Cholesky().EstimateReciprocalConditionNumber(),
            "BatchCholesky" => BatchCholeskyDecomposition.Compute(Tensor<double>.FromMatrices([a])).EstimateReciprocalConditionNumbers()[0],
            _ => throw new ArgumentException(nameof(decomposition))
        };
        SpecialFunctionAssert.Close(1.0, actual, relativeTolerance: 2e-12);
    }
}
