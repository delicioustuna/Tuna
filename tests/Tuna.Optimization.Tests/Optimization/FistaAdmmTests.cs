using Tuna.LinearAlgebra;
using Tuna.Optimization;
using Tuna.Optimization.Gradient;
using Tuna.Optimization.Proximal;
using static Tuna.Tests.Optimization.GradientTestFunctions;
using static Tuna.Tests.Optimization.ProximalOperatorTests;

namespace Tuna.Tests.Optimization;

[Trait("Optimization", "Proximal")]
public class FistaAdmmTests
{
    // ---------- FISTA ----------

    [Fact]
    public void FistaSatisfiesTheL1OptimalityConditions()
    {
        // ½‖Ax − b‖² + λ‖x‖₁ の最適性: 非零の成分で Aᵀ(Ax − b) = −λ sign(x)、零の成分で |Aᵀ(Ax − b)| ≤ λ
        const int m = 40, n = 60;
        const double lambda = 0.5;
        var (a, b) = SparseRegression(m, n, seed: 21);
        var f = new LeastSquaresObjective(a, b);
        var fista = new Fista(new FistaOptions { GradientTolerance = 1e-10, MaxIterations = 100000 });
        var result = fista.Minimize(f, new L1Penalty(lambda), new double[n]);
        Assert.Equal(MinimizationTermination.GradientTolerance, result.Termination);

        var gradient = new double[n];
        f.Evaluate(result.Point, gradient);
        int nonzero = 0;
        for (int i = 0; i < n; i++)
        {
            double x = result.Point[i];
            if (x != 0)
            {
                nonzero++;
                Assert.Equal(-lambda * Math.Sign(x), gradient[i], 1e-8);
            }
            else
            {
                Assert.True(Math.Abs(gradient[i]) <= lambda + 1e-8, $"component {i}: {gradient[i]}");
            }
        }

        Assert.InRange(nonzero, 1, m);
        Assert.Equal(f.Evaluate(result.Point) + lambda * result.Point.ToArray().Sum(Math.Abs), result.Value, 1e-12);
        Assert.True(fista.LastStep > 0);
    }

    [Fact]
    public void FistaSolvesNonNegativeLeastSquares()
    {
        const int m = 30, n = 12;
        var a = RandomMatrix(m, n, 31);
        var b = RandomVector(m, 32, scale: 3);
        var f = new LeastSquaresObjective(a, b);
        var result = new Fista(new FistaOptions { GradientTolerance = 1e-10 }).Minimize(f, new NonNegativeConstraint(), new double[n]);
        Assert.Equal(MinimizationTermination.GradientTolerance, result.Termination);

        // KKT: x ≥ 0、∇f ≥ 0、xᵢ∇fᵢ = 0
        var gradient = new double[n];
        f.Evaluate(result.Point, gradient);
        int active = 0;
        for (int i = 0; i < n; i++)
        {
            Assert.True(result.Point[i] >= 0);
            if (result.Point[i] == 0)
            {
                active++;
                Assert.True(gradient[i] >= -1e-8, $"component {i}: {gradient[i]}");
            }
            else
            {
                Assert.Equal(0.0, gradient[i], 1e-8);
            }
        }

        Assert.True(active > 0);
    }

    [Fact]
    public void AdaptiveRestartSpeedsUpAStronglyConvexProblem()
    {
        // 強凸な問題では、運動量を捨てない FISTA は振動して遅くなる
        const int n = 50;
        var p = RandomPositiveDefinite(n, 0.05, 41);
        var q = RandomVector(n, 42);
        var f = new QuadraticObjective(p, q);
        var options = new FistaOptions { GradientTolerance = 1e-8, MaxIterations = 200000 };
        var plain = new Fista(new FistaOptions { GradientTolerance = 1e-8, MaxIterations = 200000, AdaptiveRestart = false })
            .Minimize(f, new L1Penalty(0.1), new double[n]);
        var restarted = new Fista(options);
        var fast = restarted.Minimize(f, new L1Penalty(0.1), new double[n]);
        Assert.True(plain.Converged && fast.Converged);
        Assert.True(fast.Iterations < plain.Iterations, $"restart {fast.Iterations}, plain {plain.Iterations}");
        Assert.True(restarted.Restarts > 0);
        Assert.True(MaxAbsDifference(fast.Point, plain.Point) < 1e-6);
    }

    [Fact]
    public void FistaStopsAndReportsWhy()
    {
        var f = new LeastSquaresObjective(RandomMatrix(10, 5, 1), RandomVector(10, 2));
        var limited = new Fista(new FistaOptions { GradientTolerance = 0, MaxIterations = 3 }).Minimize(f, new L1Penalty(0.1), new double[5]);
        Assert.Equal(MinimizationTermination.MaxIterations, limited.Termination);
        Assert.Equal(3, limited.Iterations);

        var evaluations = new Fista(new FistaOptions { GradientTolerance = 0, MaxEvaluations = 10 }).Minimize(f, new L1Penalty(0.1), new double[5]);
        Assert.Equal(MinimizationTermination.MaxEvaluations, evaluations.Termination);
        Assert.True(evaluations.Evaluations <= 10);

        var nonFinite = new Fista().Minimize((x, g) => double.NaN, new ZeroFunction(), [1.0]);
        Assert.Equal(MinimizationTermination.NonFiniteValue, nonFinite.Termination);
        Assert.Equal(0, nonFinite.Iterations);

        // 初期の歩幅を与えれば勾配の見積もりの評価がなくなる
        var given = new Fista(new FistaOptions { InitialStep = 0.01 }).Minimize(f, new L1Penalty(0.1), new double[5]);
        var estimated = new Fista().Minimize(f, new L1Penalty(0.1), new double[5]);
        Assert.True(given.Converged && estimated.Converged);
        Assert.True(MaxAbsDifference(given.Point, estimated.Point) < 1e-5);

        Assert.Throws<ArgumentOutOfRangeException>(() => new Fista(new FistaOptions { BacktrackingFactor = 1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Fista(new FistaOptions { InitialStep = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Fista(new FistaOptions { MaxIterations = 0 }));
        Assert.Throws<ArgumentException>(() => new Fista().Minimize(f, new L1Penalty(0.1), Array.Empty<double>()));
        Assert.Throws<ArgumentException>(() => new Fista().Minimize(f, new L1Penalty(0.1), [double.NaN, 0, 0, 0, 0]));
    }

    // 初期ステップ幅の推定も評価回数上限に含めることを検証する。
    [Theory]
    [InlineData(1L)]
    [InlineData(2L)]
    [InlineData(3L)]
    public void FistaNeverExceedsTheEvaluationBudget(long budget)
    {
        int calls = 0;
        var result = new Fista(new FistaOptions { MaxEvaluations = budget }).Minimize(
            (ReadOnlyVectorSpan<double> x, VectorSpan<double> g) =>
            {
                calls++;
                g[0] = x[0];
                return 0.5 * x[0] * x[0];
            },
            new ZeroFunction(), [1.0]);
        Assert.True(calls <= budget, $"budget={budget}, calls={calls}");
        Assert.Equal(calls, result.Evaluations);
        Assert.Equal(MinimizationTermination.MaxEvaluations, result.Termination);
    }

    // ---------- ADMM ----------

    [Theory]
    [InlineData(1.0, true)]
    [InlineData(1.6, true)]
    [InlineData(1.0, false)]
    public void ConsensusAdmmMatchesFistaOnTheL1Problem(double relaxation, bool adaptive)
    {
        const int m = 40, n = 60;
        const double lambda = 0.5;
        var (a, b) = SparseRegression(m, n, seed: 21);
        var reference = new Fista(new FistaOptions { GradientTolerance = 1e-11, MaxIterations = 100000 })
            .Minimize(new LeastSquaresObjective(a, b), new L1Penalty(lambda), new double[n]);

        var admm = new Admm(new AdmmOptions { AbsoluteTolerance = 1e-10, RelativeTolerance = 1e-10, OverRelaxation = relaxation, AdaptivePenalty = adaptive, MaxIterations = 100000 });
        var result = admm.Minimize(new LeastSquaresObjective(a, b), new L1Penalty(lambda), new double[n]);
        Assert.True(result.Converged, $"{result.Termination} after {result.Iterations}");
        Assert.True(MaxAbsDifference(result.Z, reference.Point) < 1e-7, $"difference {MaxAbsDifference(result.Z, reference.Point)}");
        Assert.True(MaxAbsDifference(result.X, result.Z) < 1e-8);
        Assert.Equal(reference.Value, result.Value, 1e-7);
    }

    [Fact]
    public void SplitAdmmDenoisesWithTotalVariation()
    {
        // ½‖x − b‖² + λ‖Dx‖₁（D は隣どうしの差）。双対 min ½‖Dᵀw − b‖²（‖w‖∞ ≤ λ）を FISTA で解き、x = b − Dᵀw と比べる
        const int n = 80;
        const double lambda = 0.8;
        var random = new Random(51);
        var signal = new double[n];
        for (int i = 0; i < n; i++)
        {
            signal[i] = (i / 20 % 2 == 0 ? 1.0 : -1.0) + 0.3 * (2 * random.NextDouble() - 1);
        }

        var d = new Matrix<double>(n - 1, n);
        for (int i = 0; i < n - 1; i++)
        {
            d[i, i] = -1;
            d[i, i + 1] = 1;
        }

        var identity = Matrix<double>.Identity(n);
        var subproblem = QuadraticAdmmSubproblem.LeastSquares(identity, signal, d);
        var admm = new Admm(new AdmmOptions { AbsoluteTolerance = 1e-10, RelativeTolerance = 1e-10, OverRelaxation = 1.5, MaxIterations = 100000 });
        var result = admm.Minimize(subproblem, new L1Penalty(lambda), signal);
        Assert.True(result.Converged, $"{result.Termination} after {result.Iterations}");

        var dt = XMath.Transpose(d);
        var box = new BoxConstraint(Enumerable.Repeat(-lambda, n - 1).ToArray(), Enumerable.Repeat(lambda, n - 1).ToArray());
        var dualResult = new Fista(new FistaOptions { GradientTolerance = 1e-12, MaxIterations = 200000 })
            .Minimize(new LeastSquaresObjective(dt, signal), box, new double[n - 1]);
        Assert.True(dualResult.Converged);
        var w = dualResult.Point;
        var expected = new double[n];
        for (int i = 0; i < n; i++)
        {
            double dtw = (i > 0 ? w[i - 1] : 0) - (i < n - 1 ? w[i] : 0);
            expected[i] = signal[i] - dtw;
        }

        Assert.True(MaxAbsDifference(result.X, expected) < 1e-7, $"difference {MaxAbsDifference(result.X, expected)}");

        // 全変動の解は区分的に一定で、雑音による細かい段差はほとんど残らない
        int jumps = 0;
        for (int i = 0; i + 1 < n; i++)
        {
            jumps += Math.Abs(result.X[i + 1] - result.X[i]) > 1e-6 ? 1 : 0;
        }

        Assert.True(jumps < 20, $"jumps {jumps}");
    }

    [Fact]
    public void SplitAdmmSolvesABoxConstrainedQuadraticLikeLbfgsB()
    {
        // ½xᵀPx + qᵀx、−0.3 ≤ x ≤ 0.3 を D = I の分割形式で解き、L-BFGS-B と比べる
        const int n = 20;
        var p = RandomPositiveDefinite(n, 0.5, 61);
        var q = RandomVector(n, 62, scale: 5);
        var lower = Enumerable.Repeat(-0.3, n).ToArray();
        var upper = Enumerable.Repeat(0.3, n).ToArray();
        var subproblem = new QuadraticAdmmSubproblem(p, q, Matrix<double>.Identity(n));
        var admm = new Admm(new AdmmOptions { AbsoluteTolerance = 1e-10, RelativeTolerance = 1e-10, Penalty = 1e-3, MaxIterations = 100000 });
        var result = admm.Minimize(subproblem, new BoxConstraint(lower, upper), new double[n]);
        Assert.True(result.Converged, $"{result.Termination} after {result.Iterations}");
        Assert.NotEqual(1e-3, result.Penalty);   // 罰則の調整が働いた

        var negated = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            negated[i] = -q[i];
        }

        var reference = new LbfgsB(lower, upper, new LbfgsBOptions { GradientTolerance = 1e-9, FunctionTolerance = 0 })
            .Minimize(new Quadratic(p, negated), new double[n]);
        Assert.True(MaxAbsDifference(result.Z, reference.Point) < 1e-6, $"difference {MaxAbsDifference(result.Z, reference.Point)}");
    }

    [Fact]
    public void AdmmStopsAndValidates()
    {
        var f = new LeastSquaresObjective(RandomMatrix(10, 5, 1), RandomVector(10, 2));
        var limited = new Admm(new AdmmOptions { MaxIterations = 2, AbsoluteTolerance = 0, RelativeTolerance = 0 }).Minimize(f, new L1Penalty(0.1), new double[5]);
        Assert.Equal(MinimizationTermination.MaxIterations, limited.Termination);
        Assert.Equal(2, limited.Iterations);
        Assert.False(limited.Converged);

        Assert.Throws<ArgumentOutOfRangeException>(() => new Admm(new AdmmOptions { Penalty = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Admm(new AdmmOptions { OverRelaxation = 2 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Admm(new AdmmOptions { PenaltyBalance = 1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Admm(new AdmmOptions { PenaltyFactor = 1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Admm(new AdmmOptions { MaxIterations = 0 }));
        Assert.Throws<ArgumentException>(() => new Admm().Minimize(f, new L1Penalty(0.1), Array.Empty<double>()));
        var subproblem = new QuadraticAdmmSubproblem(Matrix<double>.Identity(3), new double[3], Matrix<double>.Identity(3));
        Assert.Throws<ArgumentException>(() => new Admm().Minimize(subproblem, new ZeroFunction(), new double[2]));
        Assert.Throws<ArgumentException>(() => new QuadraticAdmmSubproblem(Matrix<double>.Identity(3), new double[3], new Matrix<double>(2, 4)));
    }

    // ---------- 補助 ----------

    // 疎な真の係数（8 個が非零）と雑音から作る回帰の問題
    private static (Matrix<double> A, Vector<double> B) SparseRegression(int m, int n, int seed)
    {
        var a = RandomMatrix(m, n, seed);
        var random = new Random(seed + 1);
        var truth = new double[n];
        for (int k = 0; k < 8; k++)
        {
            truth[random.Next(n)] = 4 * random.NextDouble() - 2;
        }

        var b = new Vector<double>(m);
        for (int i = 0; i < m; i++)
        {
            double sum = 0.05 * (2 * random.NextDouble() - 1);
            for (int j = 0; j < n; j++)
            {
                sum += a[i, j] * truth[j];
            }

            b[i] = sum;
        }

        return (a, b);
    }
}
