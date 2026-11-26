using Tuna.Optimization;
using Tuna.Optimization.Gradient;

namespace Tuna.Tests.Optimization;

[Trait("Optimization", "Gradient")]
public class LineSearchTerminationTests
{
    [Theory]
    [InlineData(false, 0.0)]
    [InlineData(true, 0.0)]
    [InlineData(false, 1e-8)]
    [InlineData(true, 1e-8)]
    public void FunctionToleranceCanBeDisabledWhenImprovementsRoundAway(bool bounded, double tolerance)
    {
        const double gradientTolerance = 1e-12;
        LineSearchMinimizer solver = bounded
            ? new LbfgsB([-10.0, -10.0], [10.0, 10.0], new LbfgsBOptions
            {
                FunctionTolerance = tolerance,
                GradientTolerance = gradientTolerance,
                MaxIterations = 1,
            })
            : new Lbfgs(2, new LbfgsOptions
            {
                FunctionTolerance = tolerance,
                GradientTolerance = gradientTolerance,
                MaxIterations = 1,
            });

        // The analytic gradient is nonzero, but the constant makes the decrease round to zero.
        var result = solver.Minimize((x, g) =>
        {
            g[0] = x[0];
            g[1] = 2 * x[1];
            return 1e20 + 0.5 * x[0] * x[0] + x[1] * x[1];
        }, [2.0, 3.0]);

        Assert.True(result.GradientNorm > gradientTolerance);
        Assert.Equal(1, result.Iterations);
        Assert.Equal(tolerance == 0 ? MinimizationTermination.MaxIterations : MinimizationTermination.FunctionTolerance,
            result.Termination);
        Assert.Equal(tolerance > 0, result.Converged);
    }
}
