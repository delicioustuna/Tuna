using Tuna.LinearAlgebra;
using Tuna.Optimization;
using Tuna.Optimization.Gradient;
using static Tuna.Tests.Optimization.GradientTestFunctions;

namespace Tuna.Tests.Optimization;

[Trait("Optimization", "Gradient")]
public class LbfgsBTests
{
    // ---------- 引数の検証 ----------

    [Fact]
    public void RejectsInvalidBounds()
    {
        Assert.Throws<ArgumentException>(() => new LbfgsB([], []));
        Assert.Throws<ArgumentException>(() => new LbfgsB([0.0, 0.0], [1.0]));
        Assert.Throws<ArgumentException>(() => new LbfgsB([double.NaN], [1.0]));
        Assert.Throws<ArgumentException>(() => new LbfgsB([0.0], [double.NaN]));
        Assert.Throws<ArgumentException>(() => new LbfgsB([2.0], [1.0]));
        Assert.Throws<ArgumentException>(() => new LbfgsB([double.PositiveInfinity], []));
        Assert.Throws<ArgumentException>(() => new LbfgsB([], [double.NegativeInfinity]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LbfgsB([0.0], [1.0], new LbfgsBOptions { HistorySize = 0 }));
    }

    [Fact]
    public void EmptySideMeansInfinite()
    {
        var onlyLower = new LbfgsB([0.0, -1.0], []);
        Assert.Equal([0.0, -1.0], onlyLower.LowerBounds.ToArray());
        Assert.Equal([double.PositiveInfinity, double.PositiveInfinity], onlyLower.UpperBounds.ToArray());
        var onlyUpper = new LbfgsB([], [3.0]);
        Assert.Equal([double.NegativeInfinity], onlyUpper.LowerBounds.ToArray());
        Assert.Equal(1, onlyUpper.Dimension);
    }

    // ---------- 収束 ----------

    [Theory]
    [InlineData(5)]
    [InlineData(200)]
    public void MinimizesSeparableQuadraticAtTheClampedCenter(int n)
    {
        // Σ ½aᵢ(xᵢ − cᵢ)² の最小点は clamp(c, l, u)。中心の多くを範囲の外に置き、コーシー点の経路に折れ点をたくさん作る
        var random = new Random(11);
        var weight = new double[n];
        var center = new double[n];
        var lower = new double[n];
        var upper = new double[n];
        for (int i = 0; i < n; i++)
        {
            weight[i] = Math.Pow(10, 2 * random.NextDouble() - 1);
            center[i] = 6 * random.NextDouble() - 3;
            lower[i] = -1 - random.NextDouble();
            upper[i] = 1 + random.NextDouble();
        }

        var solver = new LbfgsB(lower, upper, new LbfgsBOptions { GradientTolerance = 1e-7, FunctionTolerance = 0 });
        var result = solver.Minimize(
            (x, g) =>
            {
                double sum = 0;
                for (int i = 0; i < n; i++)
                {
                    double r = x[i] - center[i];
                    g[i] = weight[i] * r;
                    sum += 0.5 * weight[i] * r * r;
                }

                return sum;
            },
            new double[n]);

        Assert.Equal(MinimizationTermination.GradientTolerance, result.Termination);
        for (int i = 0; i < n; i++)
        {
            // 自由な変数の誤差は |∂f/∂xᵢ| / aᵢ ≤ 1e-7 / 0.1
            double expected = Math.Clamp(center[i], lower[i], upper[i]);
            Assert.Equal(expected, result.Point[i], 1e-6);
        }
    }

    [Theory]
    [InlineData(1.0)]
    [InlineData(1e-2)]
    public void SatisfiesKktConditionsOnBoundedQuadratic(double shift)
    {
        const int n = 40;
        var a = RandomPositiveDefinite(n, shift, seed: 21);
        var b = RandomVector(n, seed: 22, scale: 20);
        var lower = new double[n];
        var upper = new double[n];
        for (int i = 0; i < n; i++)
        {
            lower[i] = i % 3 == 0 ? double.NegativeInfinity : -1;
            upper[i] = i % 5 == 0 ? double.PositiveInfinity : 1;
        }

        var f = new Quadratic(a, b);
        var solver = new LbfgsB(lower, upper, new LbfgsBOptions { GradientTolerance = 1e-6, FunctionTolerance = 0, MaxIterations = 20000 });
        var result = solver.Minimize(f, new double[n]);
        Assert.Equal(MinimizationTermination.GradientTolerance, result.Termination);

        // 射影勾配 r = x − P(x − ∇f)。強凸（係数 μ ≥ shift）で ∇f が L-リプシッツなら ‖x − x*‖ ≤ (1 + L)/μ · ‖r‖
        var gradient = new double[n];
        f.Evaluate(result.Point, gradient);
        int active = 0;
        double residual = 0;
        for (int i = 0; i < n; i++)
        {
            double x = result.Point[i];
            Assert.InRange(x, lower[i], upper[i]);
            double r = x - Math.Clamp(x - gradient[i], lower[i], upper[i]);
            residual += r * r;
            if (x == lower[i] || x == upper[i])
            {
                active++;
            }
        }

        Assert.True(active > 0, "no active bound");
        Assert.True(Math.Sqrt(residual) <= Math.Sqrt(n) * 1e-6, $"residual {Math.Sqrt(residual)}");

        // 参照解は射影勾配法を十分に回して作る
        double lipschitz = FrobeniusNorm(a);
        var reference = ProjectedGradientReference(f, lower, upper, lipschitz, iterations: shift < 0.1 ? 400000 : 20000);
        double error = Math.Sqrt(SquaredDistance(result.Point, reference));
        Assert.True(error <= (1 + lipschitz) / shift * Math.Sqrt(residual) + 1e-9, $"error {error}");
    }

    [Fact]
    public void MinimizesBoundedRosenbrockOnTheUpperBound()
    {
        // x₁ ≤ 0.5 のもとでの最小点は (0.5, 0.25)（値 0.25、∂f/∂x₁ = −1 で上限が効く）
        var solver = new LbfgsB([-2.0, -2.0], [0.5, 2.0], new LbfgsBOptions { GradientTolerance = 1e-10, FunctionTolerance = 0 });
        var result = solver.Minimize(new Rosenbrock(), [-1.2, 1.0]);
        Assert.True(result.Converged, result.Termination.ToString());
        Assert.Equal(0.5, result.Point[0]);
        Assert.Equal(0.25, result.Point[1], 1e-9);
        Assert.Equal(0.25, result.Value, 1e-12);
    }

    [Fact]
    public void MinimizesLinearFunctionAtACorner()
    {
        // 曲率の組が一つも溜まらない（yᵀs = 0）ので、射影した最急降下の経路だけで角まで進む
        var cost = new[] { 1.0, -2.0, 0.5, -0.25 };
        var solver = new LbfgsB([-1.0, -1.0, -1.0, -1.0], [2.0, 3.0, 4.0, 5.0], new LbfgsBOptions { GradientTolerance = 0 });
        var result = solver.Minimize(
            (x, g) =>
            {
                double sum = 0;
                for (int i = 0; i < 4; i++)
                {
                    g[i] = cost[i];
                    sum += cost[i] * x[i];
                }

                return sum;
            },
            [0.0, 0.0, 0.0, 0.0]);

        Assert.Equal(MinimizationTermination.GradientTolerance, result.Termination);
        Assert.Equal([-1.0, 3.0, -1.0, 5.0], result.Point.ToArray());
        Assert.Equal(0, solver.HistoryCount);
    }

    [Fact]
    public void FixedVariablesStayAndTheRestSolveTheReducedProblem()
    {
        // 下限 = 上限の変数は動かず、残りは ½xᵀAx − bᵀx の縮約問題 A_FF x_F = b_F − A_FC x_C を解く
        const int n = 12;
        var a = RandomPositiveDefinite(n, 1.0, seed: 31);
        var b = RandomVector(n, seed: 32);
        var lower = new double[n];
        var upper = new double[n];
        var fixedValue = new double?[n];
        for (int i = 0; i < n; i++)
        {
            if (i % 4 == 1)
            {
                fixedValue[i] = 0.3 * i - 1;
                lower[i] = upper[i] = fixedValue[i]!.Value;
            }
            else
            {
                lower[i] = double.NegativeInfinity;
                upper[i] = double.PositiveInfinity;
            }
        }

        int[] freeIndex = Enumerable.Range(0, n).Where(i => fixedValue[i] is null).ToArray();
        var reduced = new Matrix<double>(freeIndex.Length, freeIndex.Length);
        var rhs = new Vector<double>(freeIndex.Length);
        for (int p = 0; p < freeIndex.Length; p++)
        {
            int i = freeIndex[p];
            double sum = b[i];
            for (int j = 0; j < n; j++)
            {
                if (fixedValue[j] is double v)
                {
                    sum -= a[i, j] * v;
                }
            }

            rhs[p] = sum;
            for (int q = 0; q < freeIndex.Length; q++)
            {
                reduced[p, q] = a[i, freeIndex[q]];
            }
        }

        var expected = reduced.Cholesky().Solve(rhs);
        var solver = new LbfgsB(lower, upper, new LbfgsBOptions { GradientTolerance = 1e-7, FunctionTolerance = 0 });
        var result = solver.Minimize(new Quadratic(a, b), new double[n]);
        Assert.Equal(MinimizationTermination.GradientTolerance, result.Termination);
        for (int i = 0; i < n; i++)
        {
            if (fixedValue[i] is double v)
            {
                Assert.Equal(v, result.Point[i]);
            }
        }

        for (int p = 0; p < freeIndex.Length; p++)
        {
            Assert.Equal(expected[p], result.Point[freeIndex[p]], 1e-6);
        }
    }

    [Fact]
    public void WithoutBoundsFollowsTheSameIteratesAsLbfgs()
    {
        // 範囲がなければ、コーシー点から部分空間の準ニュートン点へ進む方向は L-BFGS の −H∇f と（丸めを除いて）同じ
        const int n = 6;
        var lower = Enumerable.Repeat(double.NegativeInfinity, n).ToArray();
        var upper = Enumerable.Repeat(double.PositiveInfinity, n).ToArray();
        var start = new[] { -1.2, 1.0, -1.2, 1.0, 0.5, -0.5 };
        for (int iterations = 1; iterations <= 12; iterations++)
        {
            var bounded = new LbfgsB(lower, upper, new LbfgsBOptions { HistorySize = 4, MaxIterations = iterations, GradientTolerance = 0, FunctionTolerance = 0 })
                .Minimize(new Rosenbrock(), start);
            var free = new Lbfgs(n, new LbfgsOptions { HistorySize = 4, MaxIterations = iterations, GradientTolerance = 0, FunctionTolerance = 0 })
                .Minimize(new Rosenbrock(), start);
            Assert.Equal(free.Evaluations, bounded.Evaluations);
            Assert.True(MaxAbsDifference(free.Point, bounded.Point) < 1e-9, $"iteration {iterations}");
        }

        var full = new LbfgsB(lower, upper, new LbfgsBOptions { GradientTolerance = 1e-10, FunctionTolerance = 0 }).Minimize(new Rosenbrock(), start);
        Assert.True(full.Converged);
        for (int i = 0; i < n; i++)
        {
            Assert.Equal(1.0, full.Point[i], 1e-8);
        }
    }

    [Fact]
    public void SolvesALargeBoundedProblem()
    {
        // n = 1000 の三重対角の二次形式 Σ (xᵢ − cᵢ)² + ½Σ (xᵢ₊₁ − xᵢ)²、範囲 [0, 1]
        const int n = 1000;
        var center = RandomVector(n, seed: 41, scale: 1.5);
        var lower = new double[n];
        var upper = Enumerable.Repeat(1.0, n).ToArray();
        var solver = new LbfgsB(lower, upper, new LbfgsBOptions { GradientTolerance = 1e-6, FunctionTolerance = 0 });
        var gradient = new double[n];
        double Evaluate(ReadOnlyVectorSpan<double> x, VectorSpan<double> g)
        {
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                double r = x[i] - center[i];
                sum += r * r;
                g[i] = 2 * r;
            }

            for (int i = 0; i + 1 < n; i++)
            {
                double d = x[i + 1] - x[i];
                sum += 0.5 * d * d;
                g[i] -= d;
                g[i + 1] += d;
            }

            return sum;
        }

        var result = solver.Minimize(Evaluate, new double[n]);
        Assert.Equal(MinimizationTermination.GradientTolerance, result.Termination);
        Assert.True(result.Evaluations < 200, $"evaluations {result.Evaluations}");
        Evaluate(result.Point, gradient);
        int atLower = 0, atUpper = 0;
        for (int i = 0; i < n; i++)
        {
            double x = result.Point[i];
            Assert.InRange(x, 0.0, 1.0);
            Assert.True(Math.Abs(x - Math.Clamp(x - gradient[i], 0, 1)) <= 1e-6);
            atLower += x == 0 ? 1 : 0;
            atUpper += x == 1 ? 1 : 0;
        }

        Assert.True(atLower > 100 && atUpper > 5, $"active {atLower} {atUpper}");
    }

    [Fact]
    public void FindsTheBoundaryMinimumOfAConcaveFunction()
    {
        // −Σxᵢ² は範囲の内部に極小がない。曲率が負の組は捨てられ、各変数は近いほうではなく勾配の向きの端へ行く
        var solver = new LbfgsB([-1.0, -1.0, -1.0], [2.0, 2.0, 2.0], new LbfgsBOptions { GradientTolerance = 0 });
        var result = solver.Minimize(
            (x, g) =>
            {
                double sum = 0;
                for (int i = 0; i < 3; i++)
                {
                    g[i] = -2 * x[i];
                    sum -= x[i] * x[i];
                }

                return sum;
            },
            [0.1, -0.2, 0.3]);

        Assert.Equal(MinimizationTermination.GradientTolerance, result.Termination);
        Assert.Equal([2.0, -1.0, 2.0], result.Point.ToArray());
        Assert.Equal(0, solver.HistoryCount);
    }

    // ---------- 範囲の扱い ----------

    [Fact]
    public void EveryRequestedPointIsFeasible()
    {
        const int n = 8;
        var lower = new double[n];
        var upper = new double[n];
        for (int i = 0; i < n; i++)
        {
            lower[i] = -0.5 + 0.1 * i;
            upper[i] = lower[i] + 0.3 + 0.05 * i;
        }

        var solver = new LbfgsB(lower, upper, new LbfgsBOptions { HistorySize = 3, GradientTolerance = 1e-9, FunctionTolerance = 0 });
        solver.Start(Enumerable.Repeat(5.0, n).ToArray());
        var f = new Rosenbrock();
        var gradient = new double[n];
        int count = 0;
        while (!solver.IsFinished)
        {
            ReadOnlyVectorSpan<double> x = solver.Ask();
            for (int i = 0; i < n; i++)
            {
                Assert.InRange(x[i], lower[i], upper[i]);
            }

            if (count++ == 0)
            {
                // 初期点は範囲に射影してから評価する
                Assert.Equal(upper, x.ToArray());
            }

            solver.Tell(f.Evaluate(x, gradient), gradient);
        }

        Assert.True(solver.GetResult().Converged, solver.Termination.ToString());
    }

    [Fact]
    public void ConvergenceUsesTheProjectedGradient()
    {
        // f = x₀ + x₁ は (0, 0) で勾配 (1, 1) だが、下限に接していて射影勾配は 0
        var solver = new LbfgsB([0.0, 0.0], [1.0, 1.0]);
        var result = solver.Minimize(
            (x, g) =>
            {
                g[0] = g[1] = 1;
                return x[0] + x[1];
            },
            [-3.0, 0.0]);

        Assert.Equal(MinimizationTermination.GradientTolerance, result.Termination);
        Assert.Equal(0, result.Iterations);
        Assert.Equal(1, result.Evaluations);
        Assert.Equal(0.0, result.GradientNorm);
        Assert.Equal([0.0, 0.0], result.Point.ToArray());
    }

    [Fact]
    public void AskTellMatchesMinimize()
    {
        var options = new LbfgsBOptions { HistorySize = 4, GradientTolerance = 1e-10, FunctionTolerance = 0 };
        double[] lower = [-2.0, 0.2, -2.0], upper = [0.5, 2.0, 2.0];
        var expected = new LbfgsB(lower, upper, options).Minimize(new Rosenbrock(), [-1.0, 2.0, 0.5]);

        var solver = new LbfgsB(lower, upper, options);
        solver.Start([-1.0, 2.0, 0.5]);
        var gradient = new double[3];
        var f = new Rosenbrock();
        while (!solver.IsFinished)
        {
            ReadOnlyVectorSpan<double> x = solver.Ask();
            solver.Tell(f.Evaluate(x, gradient), gradient);
        }

        var result = solver.GetResult();
        Assert.Equal(expected.Point.ToArray(), result.Point.ToArray());
        Assert.Equal(expected.Evaluations, result.Evaluations);
        Assert.Equal(expected.Termination, result.Termination);
    }

    // ---------- 補助 ----------

    private static double FrobeniusNorm(Matrix<double> a)
    {
        double sum = 0;
        for (int j = 0; j < a.Columns; j++)
        {
            for (int i = 0; i < a.Rows; i++)
            {
                sum += a[i, j] * a[i, j];
            }
        }

        return Math.Sqrt(sum);
    }

    private static double SquaredDistance(ReadOnlyVectorSpan<double> a, ReadOnlySpan<double> b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }

        return sum;
    }

    // 歩幅 1/L の射影勾配法（遅いが単純で、L-BFGS-B と独立な参照解）
    private static double[] ProjectedGradientReference(Quadratic f, double[] lower, double[] upper, double lipschitz, int iterations)
    {
        int n = lower.Length;
        var x = new double[n];
        var g = new double[n];
        for (int k = 0; k < iterations; k++)
        {
            f.Evaluate(x, g);
            for (int i = 0; i < n; i++)
            {
                x[i] = Math.Clamp(x[i] - g[i] / lipschitz, lower[i], upper[i]);
            }
        }

        return x;
    }
}
