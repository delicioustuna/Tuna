using Tuna.LinearAlgebra;
using Tuna.Optimization;
using Tuna.Optimization.Gradient;
using static Tuna.Tests.Optimization.GradientTestFunctions;

namespace Tuna.Tests.Optimization;

[Trait("Optimization", "Gradient")]
public class LbfgsTests
{
    // ---------- 引数の検証 ----------

    public static TheoryData<LbfgsOptions> InvalidOptions => new()
    {
        new() { HistorySize = 0 },
        new() { GradientTolerance = -1 },
        new() { GradientTolerance = double.NaN },
        new() { FunctionTolerance = -1 },
        new() { StepTolerance = -1 },
        new() { MaxIterations = 0 },
        new() { MaxEvaluations = 0 },
        new() { SufficientDecrease = 0 },
        new() { SufficientDecrease = 0.5 },
        new() { Curvature = 1e-5 },
        new() { Curvature = 1 },
        new() { MaxLineSearchEvaluations = 0 },
    };

    [Theory]
    [MemberData(nameof(InvalidOptions))]
    public void RejectsInvalidOptions(LbfgsOptions options)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Lbfgs(2, options));
    }

    [Fact]
    public void RejectsInvalidArguments()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Lbfgs(0));
        var solver = new Lbfgs(2);
        Assert.Throws<InvalidOperationException>(() => solver.Ask());
        Assert.Throws<InvalidOperationException>(() => solver.Tell(0, new double[2]));
        Assert.Throws<InvalidOperationException>(() => _ = solver.Point.Length);
        Assert.Throws<ArgumentException>(() => solver.Start([1.0]));
        Assert.Throws<ArgumentException>(() => solver.Start([1.0, double.NaN]));
        solver.Start([1.0, 2.0]);
        Assert.Throws<ArgumentException>(() => solver.Tell(0, new double[3]));
    }

    // ---------- 収束 ----------

    [Theory]
    [InlineData(2)]
    [InlineData(10)]
    [InlineData(100)]
    public void MinimizesRosenbrock(int n)
    {
        var start = new double[n];
        for (int i = 0; i < n; i++)
        {
            start[i] = i % 2 == 0 ? -1.2 : 1.0;
        }

        var solver = new Lbfgs(n, new LbfgsOptions { GradientTolerance = 1e-9, FunctionTolerance = 0, MaxIterations = 5000 });
        MinimizationResult result = solver.Minimize(new Rosenbrock(), start);
        Assert.Equal(MinimizationTermination.GradientTolerance, result.Termination);
        Assert.True(result.Converged);
        Assert.True(result.GradientNorm <= 1e-9);
        for (int i = 0; i < n; i++)
        {
            Assert.Equal(1.0, result.Point[i], 1e-7);
        }

        Assert.True(result.Value < 1e-14, $"value {result.Value}");
        Assert.True(result.Evaluations < 60L * n + 200, $"evaluations {result.Evaluations}");
    }

    [Theory]
    [InlineData(1.0, 1e-7)]
    [InlineData(0.1, 1e-6)]
    [InlineData(1e-3, 1e-4)]
    public void MinimizesQuadratic(double shift, double tolerance)
    {
        // A = MᵀM + shift·I（shift = 1e-3 で条件数はおよそ 1e5）。値の丸め誤差のため、悪条件の問題ほど届く勾配の大きさは大きい
        const int n = 60;
        var a = RandomPositiveDefinite(n, shift, seed: 3);
        var b = RandomVector(n, seed: 4);
        var exact = a.Cholesky().Solve(b);
        var solver = new Lbfgs(n, new LbfgsOptions { GradientTolerance = tolerance, FunctionTolerance = 0, MaxIterations = 20000 });
        var result = solver.Minimize(new Quadratic(a, b), new double[n]);
        Assert.Equal(MinimizationTermination.GradientTolerance, result.Termination);

        // ‖x − x*‖₂ = ‖A⁻¹∇f‖₂ ≤ ‖∇f‖₂ / λ_min ≤ √n‖∇f‖∞ / shift
        double bound = Math.Sqrt(n) * result.GradientNorm / shift;
        double error = 0;
        for (int i = 0; i < n; i++)
        {
            error += (result.Point[i] - exact[i]) * (result.Point[i] - exact[i]);
        }

        Assert.True(Math.Sqrt(error) <= bound, $"error {Math.Sqrt(error)} > {bound}");
    }

    [Fact]
    public void AcceptedStepsSatisfyStrongWolfeConditions()
    {
        const double mu = 1e-4, eta = 0.9;
        var solver = new Lbfgs(4, new LbfgsOptions { SufficientDecrease = mu, Curvature = eta, GradientTolerance = 1e-8, FunctionTolerance = 0 });
        var f = new Rosenbrock();
        solver.Start([-1.2, 1.0, -1.2, 1.0]);
        var gradient = new double[4];
        double value = f.Evaluate(solver.Ask(), gradient);
        solver.Tell(value, gradient);
        int checkedSteps = 0;
        while (!solver.IsFinished)
        {
            double[] x0 = solver.Point.ToArray(), g0 = solver.Gradient.ToArray();
            double f0 = solver.Value;
            int iteration = solver.Iterations;
            while (!solver.IsFinished && solver.Iterations == iteration)
            {
                value = f.Evaluate(solver.Ask(), gradient);
                solver.Tell(value, gradient);
            }

            if (solver.Iterations == iteration)
            {
                break;
            }

            // s = x₁ − x₀ は αd に等しいので、条件は s で書ける
            double slope0 = 0, slope1 = 0;
            for (int i = 0; i < 4; i++)
            {
                double s = solver.Point[i] - x0[i];
                slope0 += g0[i] * s;
                slope1 += solver.Gradient[i] * s;
            }

            Assert.True(slope0 < 0);
            Assert.True(solver.Value <= f0 + mu * slope0, $"sufficient decrease at iteration {iteration}");
            Assert.True(Math.Abs(slope1) <= eta * Math.Abs(slope0) * (1 + 1e-12), $"curvature at iteration {iteration}");
            checkedSteps++;
        }

        Assert.Equal(MinimizationTermination.GradientTolerance, solver.Termination);
        Assert.True(checkedSteps > 20);
    }

    [Fact]
    public void AskTellMatchesMinimize()
    {
        var options = new LbfgsOptions { HistorySize = 5, GradientTolerance = 1e-10, FunctionTolerance = 0 };
        var expected = new Lbfgs(3, options).Minimize(new Rosenbrock(), [-1.0, 2.0, 0.5]);

        var solver = new Lbfgs(3, options);
        solver.Start([-1.0, 2.0, 0.5]);
        var gradient = new double[3];
        var f = new Rosenbrock();
        while (!solver.IsFinished)
        {
            ReadOnlyVectorSpan<double> x = solver.Ask();
            Assert.Equal(x.ToArray(), solver.Ask().ToArray());   // Tell までは同じ点
            double value = f.Evaluate(x, gradient);
            solver.Tell(value, gradient);
        }

        var result = solver.GetResult();
        Assert.Equal(expected.Point.ToArray(), result.Point.ToArray());
        Assert.Equal(expected.Value, result.Value);
        Assert.Equal(expected.Iterations, result.Iterations);
        Assert.Equal(expected.Evaluations, result.Evaluations);
        Assert.Equal(expected.Termination, result.Termination);
        Assert.Throws<InvalidOperationException>(() => solver.Ask());
    }

    [Fact]
    public void DelegateOverloadMatchesStruct()
    {
        var f = new Rosenbrock();
        var a = new Lbfgs(2).Minimize(f, [-1.2, 1.0]);
        var b = new Lbfgs(2).Minimize((x, g) => f.Evaluate(x, g), [-1.2, 1.0]);
        Assert.Equal(a.Point.ToArray(), b.Point.ToArray());
        Assert.Equal(a.Evaluations, b.Evaluations);
    }

    [Fact]
    public void RestartingDiscardsState()
    {
        var solver = new Lbfgs(2);
        var first = solver.Minimize(new Rosenbrock(), [-1.2, 1.0]);
        var second = solver.Minimize(new Rosenbrock(), [-1.2, 1.0]);
        Assert.Equal(first.Point.ToArray(), second.Point.ToArray());
        Assert.Equal(first.Evaluations, second.Evaluations);
    }

    // ---------- 有限でない値・止まる理由 ----------

    [Fact]
    public void ShrinksStepsIntoTheDomain()
    {
        // f(x) = x² − ln x（x > 0）。x ≤ 0 では NaN。最小は 1/√2
        var solver = new Lbfgs(1, new LbfgsOptions { GradientTolerance = 1e-12, FunctionTolerance = 0 });
        var result = solver.Minimize((x, g) =>
        {
            if (x[0] <= 0)
            {
                g[0] = double.NaN;
                return double.NaN;
            }

            g[0] = 2 * x[0] - 1 / x[0];
            return x[0] * x[0] - Math.Log(x[0]);
        }, [30.0]);
        Assert.Equal(MinimizationTermination.GradientTolerance, result.Termination);
        Assert.Equal(1 / Math.Sqrt(2), result.Point[0], 1e-12);
    }

    [Fact]
    public void StopsOnNonFiniteInitialValue()
    {
        var result = new Lbfgs(2).Minimize((x, g) => double.PositiveInfinity, [1.0, 1.0]);
        Assert.Equal(MinimizationTermination.NonFiniteValue, result.Termination);
        Assert.Equal(1, result.Evaluations);
        Assert.Equal(0, result.Iterations);
    }

    [Fact]
    public void StopsAtStationaryInitialPoint()
    {
        var result = new Lbfgs(2).Minimize(new Rosenbrock(), [1.0, 1.0]);
        Assert.Equal(MinimizationTermination.GradientTolerance, result.Termination);
        Assert.Equal(0, result.Iterations);
        Assert.Equal(1, result.Evaluations);
    }

    [Fact]
    public void StopsAtIterationAndEvaluationLimits()
    {
        var byIterations = new Lbfgs(2, new LbfgsOptions { MaxIterations = 3, FunctionTolerance = 0 }).Minimize(new Rosenbrock(), [-1.2, 1.0]);
        Assert.Equal(MinimizationTermination.MaxIterations, byIterations.Termination);
        Assert.Equal(3, byIterations.Iterations);

        var byEvaluations = new Lbfgs(2, new LbfgsOptions { MaxEvaluations = 7, FunctionTolerance = 0 }).Minimize(new Rosenbrock(), [-1.2, 1.0]);
        Assert.Equal(MinimizationTermination.MaxEvaluations, byEvaluations.Termination);
        Assert.Equal(7, byEvaluations.Evaluations);
        Assert.True(byEvaluations.Value < new Rosenbrock().Evaluate(new double[] { -1.2, 1.0 }));
    }

    [Fact]
    public void StopsOnFunctionAndStepTolerances()
    {
        var byFunction = new Lbfgs(2, new LbfgsOptions { GradientTolerance = 0, FunctionTolerance = 1e-3 }).Minimize(new Rosenbrock(), [-1.2, 1.0]);
        Assert.Equal(MinimizationTermination.FunctionTolerance, byFunction.Termination);

        var byStep = new Lbfgs(2, new LbfgsOptions { GradientTolerance = 0, FunctionTolerance = 0, StepTolerance = 1e-2 }).Minimize(new Rosenbrock(), [-1.2, 1.0]);
        Assert.Equal(MinimizationTermination.StepTolerance, byStep.Termination);
    }

    [Fact]
    public void ReportsLineSearchFailureForWrongGradient()
    {
        // 勾配の符号を誤った関数: 報告される方向は降下方向だが、実際には値が増える
        var result = new Lbfgs(2).Minimize((x, g) =>
        {
            g[0] = -2 * x[0];
            g[1] = -2 * x[1];
            return x[0] * x[0] + x[1] * x[1];
        }, [1.0, 2.0]);
        Assert.Equal(MinimizationTermination.LineSearchFailed, result.Termination);
        Assert.Equal(0, result.Iterations);
        Assert.Equal(new[] { 1.0, 2.0 }, result.Point.ToArray());
    }

    [Fact]
    public void ReachesThePrecisionLimitWithZeroTolerances()
    {
        var result = new Lbfgs(2, new LbfgsOptions { GradientTolerance = 0, FunctionTolerance = 0 }).Minimize(new Rosenbrock(), [-1.2, 1.0]);
        Assert.True(result.Termination is MinimizationTermination.LineSearchFailed or MinimizationTermination.GradientTolerance);
        Assert.Equal(1.0, result.Point[0], 1e-8);
        Assert.Equal(1.0, result.Point[1], 1e-8);
    }

    // ---------- 1次元の関数での直線探索（Moré & Thuente 1994 の検定関数） ----------

    public static TheoryData<int, double> LineSearchFunctions => new()
    {
        { 1, 1e-3 }, { 1, 1e-1 }, { 1, 1e1 }, { 1, 1e3 },
        { 2, 1e-3 }, { 2, 1e-1 }, { 2, 1e1 }, { 2, 1e3 },
        { 3, 1e-3 }, { 3, 1e-1 }, { 3, 1e1 }, { 3, 1e3 },
    };

    [Theory]
    [MemberData(nameof(LineSearchFunctions))]
    public void FirstStepSatisfiesStrongWolfeOnTestFunctions(int function, double scale)
    {
        // φ(scale·x) を x ∈ ℝ の関数とみなし、0 から始める（最初の反復の歩幅の初期値は 1/|d| なので、最初の試行点は φ の歩幅 scale）
        const double mu = 1e-3, eta = 0.1;
        Func<double, (double, double)> phi = function switch
        {
            1 => a => (-a / (a * a + 2), (a * a - 2) / ((a * a + 2) * (a * a + 2))),
            2 => a => (Math.Pow(a + 0.004, 5) - 2 * Math.Pow(a + 0.004, 4), 5 * Math.Pow(a + 0.004, 4) - 8 * Math.Pow(a + 0.004, 3)),
            _ => Oscillating,
        };
        double stretch = scale;   // x の空間の最初の試行点は |x| = 1 なので、φ の歩幅では scale
        var solver = new Lbfgs(1, new LbfgsOptions
        {
            SufficientDecrease = mu,
            Curvature = eta,
            GradientTolerance = 0,
            MaxIterations = 1,
            MaxLineSearchEvaluations = 40,
        });
        var result = solver.Minimize((x, g) =>
        {
            var (value, derivative) = phi(x[0] * stretch);
            g[0] = derivative * stretch;
            return value;
        }, [0.0]);
        Assert.Equal(1, result.Iterations);
        double alpha = result.Point[0] * stretch;
        var (fa, ga) = phi(alpha);
        var (f0, g0) = phi(0);
        Assert.True(fa <= f0 + mu * alpha * g0, $"sufficient decrease at α = {alpha}");
        Assert.True(Math.Abs(ga) <= eta * Math.Abs(g0) * (1 + 1e-9), $"curvature at α = {alpha}: {ga}");
    }

    // 論文の関数 3（β = 0.01、l = 39）
    private static (double, double) Oscillating(double a)
    {
        const double beta = 0.01, l = 39;
        double phi0, dphi0;
        if (a <= 1 - beta)
        {
            phi0 = 1 - a;
            dphi0 = -1;
        }
        else if (a >= 1 + beta)
        {
            phi0 = a - 1;
            dphi0 = 1;
        }
        else
        {
            phi0 = (a - 1) * (a - 1) / (2 * beta) + beta / 2;
            dphi0 = (a - 1) / beta;
        }

        double w = l * Math.PI / 2;
        return (phi0 + 2 * (1 - beta) / (l * Math.PI) * Math.Sin(w * a), dphi0 + (1 - beta) * Math.Cos(w * a));
    }
}
