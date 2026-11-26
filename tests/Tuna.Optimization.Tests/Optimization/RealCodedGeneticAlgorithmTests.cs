using Tuna.LinearAlgebra;
using Tuna.Optimization;
using Tuna.Optimization.Evolutionary;

namespace Tuna.Tests.Optimization;

[Trait("Optimization", "GeneticAlgorithm")]
public class RealCodedGeneticAlgorithmTests
{
    private static double Sphere(ReadOnlyVectorSpan<double> x)
    {
        double sum = 0;
        for (int i = 0; i < x.Length; i++)
        {
            sum += x[i] * x[i];
        }

        return sum;
    }

    private static double Rosenbrock(ReadOnlyVectorSpan<double> x)
    {
        double sum = 0;
        for (int i = 0; i + 1 < x.Length; i++)
        {
            sum += 100 * Math.Pow(x[i + 1] - x[i] * x[i], 2) + Math.Pow(1 - x[i], 2);
        }

        return sum;
    }

    private static double Rastrigin(ReadOnlyVectorSpan<double> x)
    {
        double sum = 10 * x.Length;
        for (int i = 0; i < x.Length; i++)
        {
            sum += x[i] * x[i] - 10 * Math.Cos(2 * Math.PI * x[i]);
        }

        return sum;
    }

    private readonly struct SphereObjective : IObjectiveFunction
    {
        public double Evaluate(ReadOnlyVectorSpan<double> x) => Sphere(x);
    }

    private static double[] Fill(int n, double value) => Enumerable.Repeat(value, n).ToArray();

    private static double[] Evaluate(ReadOnlyMatrixSpan<double> batch, Func<ReadOnlyVectorSpan<double>, double> f)
    {
        var values = new double[batch.Columns];
        for (int j = 0; j < batch.Columns; j++)
        {
            values[j] = f(batch.Column(j));
        }

        return values;
    }

    // ---------- 引数の検証 ----------

    [Fact]
    public void RejectsInvalidBounds()
    {
        Assert.Throws<ArgumentException>(() => new RealCodedGeneticAlgorithm([], []));
        Assert.Throws<ArgumentException>(() => new RealCodedGeneticAlgorithm([0, 0], [1]));
        Assert.Throws<ArgumentException>(() => new RealCodedGeneticAlgorithm([double.NaN], [1]));
        Assert.Throws<ArgumentException>(() => new RealCodedGeneticAlgorithm([0], [double.PositiveInfinity]));
        Assert.Throws<ArgumentException>(() => new RealCodedGeneticAlgorithm([1], [0]));
        Assert.Throws<ArgumentException>(() => new RealCodedGeneticAlgorithm([-1e308], [1e308]));   // 幅が有限でない
    }

    public static TheoryData<RealCodedGeneticAlgorithmOptions> InvalidOptions => new()
    {
        new() { PopulationSize = 1, EliteCount = 0 },
        new() { PopulationSize = 10, EliteCount = -1 },
        new() { PopulationSize = 10, EliteCount = 10 },
        new() { PopulationSize = 10, TournamentSize = 0 },
        new() { PopulationSize = 10, TournamentSize = 11 },
        new() { CrossoverProbability = -0.1 },
        new() { CrossoverProbability = 1.5 },
        new() { CrossoverProbability = double.NaN },
        new() { CrossoverDistributionIndex = -1 },
        new() { CrossoverDistributionIndex = double.PositiveInfinity },
        new() { MutationProbability = 2 },
        new() { MutationProbability = double.NaN },
        new() { MutationDistributionIndex = double.NaN },
        new() { MaxGenerations = 0 },
        new() { TargetFitness = double.NaN },
        new() { StagnationGenerations = 0 },
    };

    [Theory]
    [MemberData(nameof(InvalidOptions))]
    public void RejectsInvalidOptions(RealCodedGeneticAlgorithmOptions options)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new RealCodedGeneticAlgorithm([0, 0], [1, 1], options));
    }

    [Fact]
    public void RejectsAPopulationLargerThanTheStorage()
    {
        var options = new RealCodedGeneticAlgorithmOptions { PopulationSize = int.MaxValue / 2 + 1 };
        Assert.Throws<ArgumentOutOfRangeException>(() => new RealCodedGeneticAlgorithm([0, 0], [1, 1], options));
    }

    // ---------- ask/tell の手順 ----------

    [Fact]
    public void AskReturnsTheSameCandidatesUntilTell()
    {
        var ga = new RealCodedGeneticAlgorithm([-1, -1, -1], [1, 1, 1], new() { PopulationSize = 8, Seed = 1 });
        var first = ga.Ask().ToMatrix();
        var second = ga.Ask().ToMatrix();
        Assert.Equal(3, first.Rows);
        Assert.Equal(8, first.Columns);
        Assert.Equal(first.AsSpan().ToArray(), second.AsSpan().ToArray());

        ga.Tell(Evaluate(ga.Ask(), Sphere));
        var offspring = ga.Ask();
        Assert.Equal(8 - 2, offspring.Columns);   // 既定のエリート数 2 を除いた数
    }

    [Fact]
    public void TellRequiresAPendingAskAndOneValuePerCandidate()
    {
        var ga = new RealCodedGeneticAlgorithm([0], [1], new() { PopulationSize = 4, Seed = 1 });
        Assert.Throws<InvalidOperationException>(() => ga.Tell(new double[4]));
        Assert.Throws<InvalidOperationException>(() => ga.BestFitness);
        Assert.Throws<InvalidOperationException>(() => ga.Population.Columns);
        ga.Ask();
        Assert.Throws<ArgumentException>(() => ga.Tell(new double[3]));
        ga.Tell([0.4, 0.3, 0.2, 0.1]);
        Assert.Throws<InvalidOperationException>(() => ga.Tell([0, 0]));   // 次の Ask の前
        Assert.Equal(1, ga.Generations);
        Assert.Equal(4, ga.Evaluations);
        Assert.Equal(0.1, ga.BestFitness);
    }

    [Fact]
    public void AskAfterTerminationThrows()
    {
        var ga = new RealCodedGeneticAlgorithm([0], [1], new() { PopulationSize = 4, MaxGenerations = 1, Seed = 1 });
        ga.Tell(Evaluate(ga.Ask(), Sphere));
        Assert.True(ga.IsFinished);
        Assert.Equal(GeneticAlgorithmTermination.MaxGenerations, ga.Termination);
        Assert.Throws<InvalidOperationException>(() => ga.Ask());
    }

    [Fact]
    public void PopulationHoldsTheElitesFollowedByTheEvaluatedOffspring()
    {
        var ga = new RealCodedGeneticAlgorithm([-2, -2], [2, 2], new() { PopulationSize = 10, EliteCount = 3, Seed = 4 });
        ga.Tell(Evaluate(ga.Ask(), Sphere));
        var previous = ga.Population.ToMatrix();
        var previousFitness = ga.PopulationFitness.ToArray();
        int[] best = Enumerable.Range(0, 10).OrderBy(i => previousFitness[i]).ThenBy(i => i).Take(3).ToArray();

        var offspring = ga.Ask().ToMatrix();
        var values = Evaluate(offspring, Sphere);
        ga.Tell(values);
        for (int k = 0; k < 3; k++)
        {
            Assert.Equal(previous.Column(best[k]).ToVector().ToArray(), ga.Population.Column(k).ToVector().ToArray());
            Assert.Equal(previousFitness[best[k]], ga.PopulationFitness[k]);
        }

        for (int k = 0; k < 7; k++)
        {
            Assert.Equal(offspring.Column(k).ToVector().ToArray(), ga.Population.Column(3 + k).ToVector().ToArray());
            Assert.Equal(values[k], ga.PopulationFitness[3 + k]);
        }
    }

    // ---------- 再現性 ----------

    [Fact]
    public void SameSeedReproducesTheCandidatesBitForBit()
    {
        var options = new RealCodedGeneticAlgorithmOptions { PopulationSize = 13, EliteCount = 1, MaxGenerations = 15, Seed = 20261013 };
        var a = new RealCodedGeneticAlgorithm(Fill(4, -3), Fill(4, 3), options);
        var b = new RealCodedGeneticAlgorithm(Fill(4, -3), Fill(4, 3), options);
        while (!a.IsFinished)
        {
            var batchA = a.Ask();
            var batchB = b.Ask();
            Assert.Equal(batchA.ToMatrix().AsSpan().ToArray(), batchB.ToMatrix().AsSpan().ToArray());
            a.Tell(Evaluate(batchA, Rastrigin));
            b.Tell(Evaluate(batchB, Rastrigin));
        }

        Assert.True(b.IsFinished);
        Assert.Equal(a.BestFitness, b.BestFitness);
    }

    [Fact]
    public void DifferentSeedsGiveDifferentInitialPopulations()
    {
        var a = new RealCodedGeneticAlgorithm([0, 0], [1, 1], new() { PopulationSize = 4, Seed = 1 });
        var b = new RealCodedGeneticAlgorithm([0, 0], [1, 1], new() { PopulationSize = 4, Seed = 2 });
        Assert.NotEqual(a.Ask().ToMatrix().AsSpan().ToArray(), b.Ask().ToMatrix().AsSpan().ToArray());
    }

    [Fact]
    public void OmittedSeedIsReportedAndReproduces()
    {
        var a = new RealCodedGeneticAlgorithm([0, 0], [1, 1], new() { PopulationSize = 4 });
        var b = new RealCodedGeneticAlgorithm([0, 0], [1, 1], new() { PopulationSize = 4, Seed = a.Seed });
        Assert.Equal(a.Ask().ToMatrix().AsSpan().ToArray(), b.Ask().ToMatrix().AsSpan().ToArray());
    }

    [Fact]
    public void RunMatchesTheEquivalentAskTellLoop()
    {
        var options = new RealCodedGeneticAlgorithmOptions { PopulationSize = 20, MaxGenerations = 30, Seed = 8 };
        var run = new RealCodedGeneticAlgorithm(Fill(3, -5), Fill(3, 5), options).Run(new SphereObjective());

        var manual = new RealCodedGeneticAlgorithm(Fill(3, -5), Fill(3, 5), options);
        while (!manual.IsFinished)
        {
            manual.Tell(Evaluate(manual.Ask(), Sphere));
        }

        Assert.Equal(manual.BestFitness, run.BestFitness);
        Assert.Equal(manual.BestPoint.ToVector().ToArray(), run.BestPoint.ToArray());

        var viaDelegate = new RealCodedGeneticAlgorithm(Fill(3, -5), Fill(3, 5), options).Run(Sphere);
        Assert.Equal(run, viaDelegate with { BestPoint = run.BestPoint });
        Assert.Equal(run.BestPoint.ToArray(), viaDelegate.BestPoint.ToArray());
    }

    // ---------- 範囲・エリート・NaN ----------

    [Fact]
    public void CandidatesStayInsideTheBoundsAndFixedVariablesKeepTheirValue()
    {
        double[] lower = [-1, 2.5, 0, -1e-300], upper = [1, 2.5, 1e6, 1e-300];
        var ga = new RealCodedGeneticAlgorithm(lower, upper,
            new() { PopulationSize = 30, MaxGenerations = 40, MutationProbability = 1, CrossoverProbability = 1, Seed = 3 });
        while (!ga.IsFinished)
        {
            var batch = ga.Ask();
            for (int j = 0; j < batch.Columns; j++)
            {
                for (int d = 0; d < lower.Length; d++)
                {
                    Assert.InRange(batch[d, j], lower[d], upper[d]);
                }

                Assert.Equal(2.5, batch[1, j]);
            }

            ga.Tell(Evaluate(batch, x => Math.Abs(x[0] - 0.3) + Math.Abs(x[2] - 7)));
        }
    }

    [Fact]
    public void ElitismKeepsThePopulationBestFromGettingWorse()
    {
        var ga = new RealCodedGeneticAlgorithm(Fill(5, -5.12), Fill(5, 5.12), new() { PopulationSize = 20, EliteCount = 1, MaxGenerations = 60, Seed = 12 });
        double previous = double.PositiveInfinity;
        while (!ga.IsFinished)
        {
            ga.Tell(Evaluate(ga.Ask(), Rastrigin));
            double populationBest = ga.PopulationFitness.ToArray().Min();
            Assert.True(populationBest <= previous);
            Assert.Equal(populationBest, ga.BestFitness);
            previous = populationBest;
        }

        Assert.Equal(20 + 59L * 19, ga.Evaluations);
    }

    [Fact]
    public void WithoutElitesTheBestPointEverEvaluatedIsKept()
    {
        var ga = new RealCodedGeneticAlgorithm(Fill(2, -1), Fill(2, 1), new() { PopulationSize = 6, EliteCount = 0, MaxGenerations = 30, Seed = 2 });
        double best = double.PositiveInfinity;
        double[] bestPoint = [];
        while (!ga.IsFinished)
        {
            var batch = ga.Ask();
            var values = Evaluate(batch, Sphere);
            for (int j = 0; j < values.Length; j++)
            {
                if (values[j] < best)
                {
                    best = values[j];
                    bestPoint = batch.Column(j).ToVector().ToArray();
                }
            }

            ga.Tell(values);
            Assert.Equal(best, ga.BestFitness);
            Assert.Equal(bestPoint, ga.BestPoint.ToVector().ToArray());
        }
    }

    [Fact]
    public void NaNIsTreatedAsTheWorstValue()
    {
        // x₀ < 0 の点は評価できない（NaN）とする。最良値は常に数値で、エリートに NaN は残らない
        var ga = new RealCodedGeneticAlgorithm([-1, -1], [1, 1], new() { PopulationSize = 16, EliteCount = 4, MaxGenerations = 25, Seed = 6 });
        while (!ga.IsFinished)
        {
            ga.Tell(Evaluate(ga.Ask(), x => x[0] < 0 ? double.NaN : Sphere(x)));
            Assert.False(double.IsNaN(ga.BestFitness));
            for (int k = 0; k < 4 && ga.Generations > 1; k++)
            {
                Assert.False(double.IsNaN(ga.PopulationFitness[k]));
            }
        }

        Assert.True(ga.BestPoint[0] >= 0);
    }

    [Fact]
    public void AllNaNRunsToTheGenerationLimitWithNaNAsTheBest()
    {
        var result = new RealCodedGeneticAlgorithm([0], [1], new() { PopulationSize = 4, MaxGenerations = 5, Seed = 1 })
            .Run(_ => double.NaN);
        Assert.True(double.IsNaN(result.BestFitness));
        Assert.Equal(5, result.Generations);
        Assert.Equal(GeneticAlgorithmTermination.MaxGenerations, result.Termination);
    }

    [Fact]
    public void NegativeInfinityCountsAsTheBestValue()
    {
        var result = new RealCodedGeneticAlgorithm([0], [1], new() { PopulationSize = 6, MaxGenerations = 3, Seed = 1 })
            .Run(x => x[0] > 0.5 ? double.NegativeInfinity : x[0]);
        Assert.Equal(double.NegativeInfinity, result.BestFitness);
        Assert.True(result.BestPoint[0] > 0.5);
    }

    // ---------- 終了条件 ----------

    [Fact]
    public void StopsWhenTheTargetIsReached()
    {
        var result = new RealCodedGeneticAlgorithm(Fill(2, -5), Fill(2, 5), new() { PopulationSize = 30, MaxGenerations = 1000, TargetFitness = 1e-3, Seed = 1 })
            .Run(Sphere);
        Assert.Equal(GeneticAlgorithmTermination.TargetReached, result.Termination);
        Assert.True(result.BestFitness <= 1e-3);
        Assert.True(result.Generations < 1000);
    }

    [Fact]
    public void StopsAfterTheGivenNumberOfGenerationsWithoutImprovement()
    {
        var result = new RealCodedGeneticAlgorithm([0], [1], new() { PopulationSize = 5, MaxGenerations = 100, StagnationGenerations = 4, Seed = 1 })
            .Run(_ => 1.0);
        Assert.Equal(GeneticAlgorithmTermination.Stagnation, result.Termination);
        Assert.Equal(1 + 4, result.Generations);
        Assert.Equal(5 + 4L * 3, result.Evaluations);
    }

    [Fact]
    public void TargetIsCheckedBeforeTheGenerationLimit()
    {
        var result = new RealCodedGeneticAlgorithm([0], [1], new() { PopulationSize = 4, MaxGenerations = 1, TargetFitness = 2, Seed = 1 })
            .Run(_ => 1.0);
        Assert.Equal(GeneticAlgorithmTermination.TargetReached, result.Termination);
    }

    [Fact]
    public void RunOnAFinishedAlgorithmReturnsWithoutEvaluating()
    {
        var ga = new RealCodedGeneticAlgorithm([0], [1], new() { PopulationSize = 4, MaxGenerations = 1, Seed = 1 });
        ga.Run(Sphere);
        int calls = 0;
        var result = ga.Run(x =>
        {
            calls++;
            return Sphere(x);
        });
        Assert.Equal(0, calls);
        Assert.Equal(1, result.Generations);
    }

    [Fact]
    public void ObjectiveExceptionsLeaveTheCandidatesPending()
    {
        var ga = new RealCodedGeneticAlgorithm([0, 0], [1, 1], new() { PopulationSize = 4, Seed = 1 });
        var before = ga.Ask().ToMatrix().AsSpan().ToArray();
        Assert.Throws<InvalidOperationException>(() => ga.Run(_ => throw new InvalidOperationException("broken")));
        Assert.Equal(0, ga.Generations);
        Assert.Equal(before, ga.Ask().ToMatrix().AsSpan().ToArray());
    }

    [Fact]
    public void RunHonorsCancellation()
    {
        var ga = new RealCodedGeneticAlgorithm([0], [1], new() { PopulationSize = 4, Seed = 1 });
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        Assert.Throws<OperationCanceledException>(() => ga.Run(Sphere, cancellation.Token));
    }

    // ---------- 収束（固定シード） ----------

    public static TheoryData<ulong> Seeds => new() { 1, 2, 3 };

    [Theory]
    [MemberData(nameof(Seeds))]
    public void MinimizesStandardBenchmarkFunctions(ulong seed)
    {
        var sphere = new RealCodedGeneticAlgorithm(Fill(10, -5), Fill(10, 5), new() { PopulationSize = 60, MaxGenerations = 300, Seed = seed })
            .Run(new SphereObjective());
        Assert.True(sphere.BestFitness < 1e-2, $"sphere {sphere.BestFitness}");

        var rosenbrock = new RealCodedGeneticAlgorithm(Fill(2, -2), Fill(2, 2), new() { PopulationSize = 60, MaxGenerations = 300, Seed = seed })
            .Run(Rosenbrock);
        Assert.True(rosenbrock.BestFitness < 1e-2, $"rosenbrock {rosenbrock.BestFitness}");
        Assert.Equal(1.0, rosenbrock.BestPoint[0], 0.1);
        Assert.Equal(1.0, rosenbrock.BestPoint[1], 0.2);

        var rastrigin = new RealCodedGeneticAlgorithm(Fill(5, -5.12), Fill(5, 5.12), new() { PopulationSize = 100, MaxGenerations = 300, Seed = seed })
            .Run(Rastrigin);
        Assert.True(rastrigin.BestFitness < 0.1, $"rastrigin {rastrigin.BestFitness}");
    }
}
