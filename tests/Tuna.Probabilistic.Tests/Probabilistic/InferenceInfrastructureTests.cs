using Tuna.LinearAlgebra;
using Tuna.Optimization;
using Tuna.Probabilistic.Inference;

namespace Tuna.Tests.Probabilistic;

// 対数密度のラッパーと変数変換、評価不能な値の扱い、サンプリング結果の形状と要約を検証する。
[Trait("Probabilistic", "Mcmc")]
public class InferenceInfrastructureTests
{
    private sealed class Potential : IDifferentiableObjectiveFunction
    {
        public double Evaluate(ReadOnlyVectorSpan<double> x) => 0.5 * x[0] * x[0] + 2 * x[1] * x[1];

        public double Evaluate(ReadOnlyVectorSpan<double> x, VectorSpan<double> gradient)
        {
            gradient[0] = x[0];
            gradient[1] = 4 * x[1];
            return Evaluate(x);
        }
    }

    [Fact]
    public void PotentialsAndObjectivesAreNegated()
    {
        double[] x = [0.3, -0.7];
        var g = new double[2];
        PotentialLogDensity<Potential> density = LogDensities.FromPotential(new Potential());
        Assert.Equal(-(0.5 * 0.09 + 2 * 0.49), density.LogDensity(x, g), 15);
        Assert.Equal(-0.3, g[0], 15);
        Assert.Equal(2.8, g[1], 15);
        Assert.Equal(density.LogDensity(x), density.LogDensity(x, g));

        var target = new TargetDensities.CorrelatedNormal(0, 0, 1, 2, 0.5);
        NegativeDifferentiableLogDensity<TargetDensities.CorrelatedNormal> objective = LogDensities.ToDifferentiableObjective(target);
        var h = new double[2];
        Assert.Equal(-target.LogDensity(x, h), objective.Evaluate(x, g));
        Assert.Equal(-h[0], g[0]);
        Assert.Equal(-h[1], g[1]);
        Assert.Equal(-target.LogDensity(x), LogDensities.ToObjective(target).Evaluate(x));

        // MAP 推定（−ln π の最小化）
        var minimum = new Tuna.Optimization.Gradient.Lbfgs(2).Minimize(
            LogDensities.ToDifferentiableObjective(new TargetDensities.IndependentNormal([3, -1], [2, 0.5])), [0.0, 0.0]);
        Assert.Equal(3, minimum.Point[0], 4);
        Assert.Equal(-1, minimum.Point[1], 4);
    }

    [Fact]
    public void GradientErrorDetectsWrongGradients()
    {
        var target = new TargetDensities.Banana(3, 0.05);
        Assert.True(LogDensities.GradientError(target, [1.0, 2.0, -0.5]) < 1e-7);
        IDifferentiableLogDensity wrong = LogDensities.FromDelegate((x, g) =>
        {
            g[0] = x[0];
            return -0.5 * x[0] * x[0];
        });
        Assert.True(LogDensities.GradientError(wrong, [1.5]) > 0.5);
        Assert.Throws<ArgumentNullException>(() => LogDensities.FromDelegate((Func<ReadOnlyVectorSpan<double>, double>)null!));
        Assert.Throws<ArgumentNullException>(() => LogDensities.FromPotential<Potential>(null!));
        Assert.Throws<ArgumentNullException>(() => LogDensities.ToObjective<ILogDensity>(null!));
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void UnevaluablePointsAreNeverVisited(double bad)
    {
        // x > 1 で評価できない（NaN・+∞）か密度 0（−∞）の正規分布。サンプラーは x > 1 へ移らない
        double LogDensity(ReadOnlyVectorSpan<double> x) => x[0] > 1 ? bad : -0.5 * x[0] * x[0];
        RandomWalkMetropolisResult walk = new RandomWalkMetropolis(new RandomWalkMetropolisOptions { Seed = 91, Draws = 500 }).Sample(LogDensity, [0.0]);
        SliceSamplingResult slice = new SliceSampler(new SliceSamplerOptions { Seed = 91, Draws = 500 }).Sample(LogDensity, [0.0]);
        HamiltonianSamplingResult nuts = new NoUTurnSampler(new NoUTurnSamplerOptions { Seed = 91, Draws = 500 })
            .Sample((x, g) =>
            {
                g[0] = -x[0];
                return LogDensity(x);
            }, [0.0]);
        foreach (SamplingResult result in new SamplingResult[] { walk, slice, nuts })
        {
            Assert.All(result.Draws.AsSpan().ToArray(), v => Assert.True(v <= 1));
            Assert.All(result.LogDensities.AsSpan().ToArray(), v => Assert.True(double.IsFinite(v)));
        }

        Assert.True(nuts.DivergenceCount > 0);
    }

    [Fact]
    public void ResultViewsShareTheDraws()
    {
        var target = new TargetDensities.CorrelatedNormal(1, -2, 1, 10, 0.9);
        RandomWalkMetropolisResult result = new RandomWalkMetropolis(new RandomWalkMetropolisOptions { Seed = 92, Draws = 300, Chains = 3 })
            .Sample(target, [0.0, 0.0]);
        Assert.Equal(2, result.Dimension);
        Assert.Equal(300, result.DrawCount);
        Assert.Equal(3, result.ChainCount);
        ReadOnlyMatrixSpan<double> second = result.Parameter(1);
        Assert.Equal(300, second.Rows);
        Assert.Equal(3, second.Columns);
        MatrixSpan<double> chain = result.Chain(2);
        Assert.Equal(2, chain.Rows);
        Assert.Equal(300, chain.Columns);
        double sum = 0, sumSquares = 0;
        for (int c = 0; c < 3; c++)
        {
            for (int t = 0; t < 300; t++)
            {
                Assert.Equal(result.Draws[1, t, c], second[t, c]);
                Assert.Equal(result.Draws[0, t, c], result.Chain(c)[0, t]);
                sum += result.Draws[1, t, c];
                sumSquares += result.Draws[1, t, c] * result.Draws[1, t, c];
            }
        }

        double mean = sum / 900;
        Assert.Equal(mean, result.Mean()[1], 12);
        Assert.Equal((sumSquares - 900 * mean * mean) / 899, result.Covariance()[1, 1], 9);
        Assert.Equal(result.Covariance()[0, 1], result.Covariance()[1, 0]);
        Assert.Equal(result.TotalEvaluations, result.Evaluations.Sum());
        Assert.Throws<ArgumentOutOfRangeException>(() => result.Parameter(2));
        Assert.Throws<ArgumentOutOfRangeException>(() => result.Chain(-1));

        PosteriorSummary summary = result.Summarize();
        Assert.Equal("theta[1]", summary[1].Name);
        Assert.Equal(mean, summary[1].Mean, 12);
        Assert.Equal(ConvergenceDiagnostics.RHat(second), summary[1].RHat);
        Assert.Equal(Math.Min(summary[0].BulkEffectiveSampleSize, Math.Min(summary[0].TailEffectiveSampleSize,
            Math.Min(summary[1].BulkEffectiveSampleSize, summary[1].TailEffectiveSampleSize))), summary.MinEffectiveSampleSize);
    }

    [Fact]
    public void LogDensitiesCanBeUsedAsObjectivesForEvolutionarySearch()
    {
        // 遺伝的アルゴリズムなど値だけを使う手法へ: −ln π を最小化すると最頻値に着く
        var objective = LogDensities.ToObjective(new TargetDensities.TwoModes(2, 3));
        Assert.True(objective.Evaluate(new[] { 3.0, 3.0 }) < objective.Evaluate(new[] { 0.0, 0.0 }));
        Assert.Equal(objective.Evaluate(new[] { 3.0, 3.0 }), objective.Evaluate(new[] { -3.0, -3.0 }), 12);
    }
}
