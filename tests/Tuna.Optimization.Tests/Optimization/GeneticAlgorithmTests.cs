using Tuna.LinearAlgebra;
using Tuna.Optimization.Evolutionary;
using Tuna.RandomSource;

namespace Tuna.Tests.Optimization;

[Trait("Optimization", "GeneticAlgorithm")]
public class GeneticAlgorithmTests
{
    private static readonly GenomeLayout Mixed = new(
        realLowerBounds: [-1, 0.25], realUpperBounds: [1, 0.25],
        integerLowerBounds: [long.MinValue, -3, 7], integerUpperBounds: [long.MinValue + 5, 3, 7],
        bitCount: 5, categoryCounts: [1, 4, 2], permutationLengths: [3, 1, 6]);

    /// <summary>個体の区画ごとの値をすべて連ねた文字列にする（再現性の比較用。実数はビット列で比べる）</summary>
    private static string Describe(GenomeView g) =>
        string.Join(",", g.Reals.ToArray().Select(v => BitConverter.DoubleToInt64Bits(v).ToString())) + "|" +
        string.Join(",", g.Integers.ToArray()) + "|" + string.Join(",", g.Bits.ToArray()) + "|" +
        string.Join(",", g.Categories.ToArray()) + "|" + string.Join(",", g.Permutations.ToArray());

    private static string Describe(GenomeBatch batch)
    {
        var parts = new string[batch.Count];
        for (int j = 0; j < batch.Count; j++)
        {
            parts[j] = Describe(batch[j]);
        }

        return string.Join(";", parts);
    }

    /// <summary>区画を混ぜた個体の、決まった値（整数・カテゴリ・二値・順列の並び）からの外れ具合</summary>
    private static double MixedObjective(GenomeView g)
    {
        double sum = Math.Abs(g.Reals[0] - 0.3);
        sum += Math.Abs((double)(g.Integers[1] - 2));
        sum += g.Categories[1] == 3 ? 0 : 1;
        foreach (bool bit in g.Bits)
        {
            sum += bit ? 0 : 1;
        }

        ReadOnlySpan<int> tour = g.Permutation(2);
        for (int k = 0; k < tour.Length; k++)
        {
            sum += tour[k] == k ? 0 : 1;
        }

        return sum;
    }

    private static double[] Evaluate(GenomeBatch batch, Func<GenomeView, double> f)
    {
        var values = new double[batch.Count];
        for (int j = 0; j < batch.Count; j++)
        {
            values[j] = f(batch[j]);
        }

        return values;
    }

    // ---------- ゲノムの構成 ----------

    [Fact]
    public void LayoutReportsItsSections()
    {
        Assert.Equal(2, Mixed.RealCount);
        Assert.Equal(3, Mixed.IntegerCount);
        Assert.Equal(5, Mixed.BitCount);
        Assert.Equal(3, Mixed.CategoryCount);
        Assert.Equal(3, Mixed.PermutationCount);
        Assert.Equal(10, Mixed.PermutationElementCount);
        Assert.Equal([0, 3, 4], Enumerable.Range(0, 3).Select(Mixed.PermutationOffset));
        Assert.Equal(2 + 3 + 5 + 3 + 3, Mixed.MutationUnitCount);
        Assert.Equal([1, 4, 2], Mixed.CategoryCounts.ToArray());
    }

    [Fact]
    public void LayoutRejectsInvalidSections()
    {
        Assert.Throws<ArgumentException>(() => new GenomeLayout());
        Assert.Throws<ArgumentException>(() => new GenomeLayout(realLowerBounds: [0, 0], realUpperBounds: [1]));
        Assert.Throws<ArgumentException>(() => new GenomeLayout(realLowerBounds: [double.NaN], realUpperBounds: [1]));
        Assert.Throws<ArgumentException>(() => new GenomeLayout(realLowerBounds: [1], realUpperBounds: [0]));
        Assert.Throws<ArgumentException>(() => new GenomeLayout(realLowerBounds: [-1e308], realUpperBounds: [1e308]));
        Assert.Throws<ArgumentException>(() => new GenomeLayout(integerLowerBounds: [0, 0], integerUpperBounds: [1]));
        Assert.Throws<ArgumentException>(() => new GenomeLayout(integerLowerBounds: [1], integerUpperBounds: [0]));
        Assert.Throws<ArgumentException>(() => new GenomeLayout(integerLowerBounds: [0], integerUpperBounds: [(1L << 53) + 1]));
        Assert.Throws<ArgumentException>(() => new GenomeLayout(integerLowerBounds: [long.MinValue], integerUpperBounds: [long.MaxValue]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new GenomeLayout(bitCount: -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new GenomeLayout(categoryCounts: [3, 0]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new GenomeLayout(permutationLengths: [0]));

        var widest = new GenomeLayout(integerLowerBounds: [-(1L << 52)], integerUpperBounds: [1L << 52]);   // 幅 2⁵³ は可
        Assert.Equal(1, widest.IntegerCount);
    }

    [Fact]
    public void RejectsANullLayoutAndPopulationsLargerThanTheStorage()
    {
        Assert.Throws<ArgumentNullException>(() => new GeneticAlgorithm(null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => new GeneticAlgorithm(new GenomeLayout(bitCount: 1 << 30), new() { PopulationSize = 4 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new GeneticAlgorithm(Mixed, new() { EliteCount = 100 }));
    }

    // ---------- 乱数の割り当て ----------

    /// <summary>
    /// 初期個体群は文書の割り当てのとおり: 候補 i はストリーム ID 0（初期化・世代 0）、ブロック番号 i·2³² から、
    /// 実数（一様実数）→ 整数（下限 + [0, 幅] の整数乱数）→ 二値（一様実数が 1/2 未満なら真）→ カテゴリ（[0, k) の整数乱数）→ 順列（Fisher–Yates）の順に乱数を使う
    /// </summary>
    [Fact]
    public void InitialPopulationFollowsTheDocumentedStreamAssignment()
    {
        const ulong seed = 20261013;
        var ga = new GeneticAlgorithm(Mixed, new() { PopulationSize = 6, Seed = seed });
        var batch = ga.Ask();
        Assert.Equal(6, batch.Count);
        var random = new Philox4x32Random(0);
        for (int i = 0; i < 6; i++)
        {
            random.RestoreState(new Philox4x32State(seed, 0, (ulong)i << 32, 0));
            GenomeView g = batch[i];
            for (int d = 0; d < Mixed.RealCount; d++)
            {
                double lo = Mixed.RealLowerBounds[d], hi = Mixed.RealUpperBounds[d];
                Assert.Equal(Math.Min(lo + (hi - lo) * random.NextDouble(), hi), g.Reals[d]);
            }

            for (int d = 0; d < Mixed.IntegerCount; d++)
            {
                long lo = Mixed.IntegerLowerBounds[d];
                ulong width = unchecked((ulong)(Mixed.IntegerUpperBounds[d] - lo));
                Assert.Equal(unchecked(lo + (long)random.NextUInt64(width + 1)), g.Integers[d]);
            }

            for (int d = 0; d < Mixed.BitCount; d++)
            {
                Assert.Equal(random.NextDouble() < 0.5, g.Bits[d]);
            }

            for (int d = 0; d < Mixed.CategoryCount; d++)
            {
                Assert.Equal(random.NextInt32(Mixed.CategoryCounts[d]), g.Categories[d]);
            }

            for (int k = 0; k < Mixed.PermutationCount; k++)
            {
                int[] expected = Enumerable.Range(0, Mixed.PermutationLengths[k]).ToArray();
                for (int m = expected.Length - 1; m > 0; m--)
                {
                    int j = random.NextInt32(m + 1);
                    (expected[m], expected[j]) = (expected[j], expected[m]);
                }

                Assert.Equal(expected, g.Permutation(k).ToArray());
            }
        }
    }

    /// <summary>評価用のシードは、ストリーム ID 4·2⁵⁶ + 世代、ブロック 0 の Philox の出力（語 0 が下位 32 ビット）で、乱数源の状態を変えない</summary>
    [Fact]
    public void EvaluationSeedFollowsTheDocumentedFormulaWithoutConsumingRandomness()
    {
        static ulong Expected(ulong seed, int generation)
        {
            var block = Philox4x32Random.GenerateBlock(seed, (4UL << 56) | (uint)generation, 0);
            return ((ulong)block.Word1 << 32) | block.Word0;
        }

        var options = new GeneticAlgorithmOptions { PopulationSize = 6, Seed = 77 };
        var reader = new GeneticAlgorithm(Mixed, options);
        var plain = new GeneticAlgorithm(Mixed, options);
        for (int generation = 0; generation < 4; generation++)
        {
            Assert.Equal(Expected(77, generation), reader.EvaluationSeed);
            Assert.Equal(reader.EvaluationSeed, reader.EvaluationSeed);
            var batch = reader.Ask();
            Assert.Equal(Expected(77, generation), reader.EvaluationSeed);   // 評価待ちの世代のシード
            Assert.Equal(Describe(plain.Ask()), Describe(batch));
            reader.Tell(Evaluate(batch, MixedObjective));
            plain.Tell(Evaluate(plain.Ask(), MixedObjective));
        }

        Assert.NotEqual(Expected(77, 0), Expected(77, 1));
        var real = new RealCodedGeneticAlgorithm([0.0], [1.0], new() { Seed = 77 });
        Assert.Equal(Expected(77, 0), real.EvaluationSeed);
    }

    /// <summary>実数の遺伝子だけの構成では、RealCodedGeneticAlgorithm と同じ候補を作る（エリートを評価し直す場合も）</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RealOnlyLayoutMatchesTheRealCodedAlgorithm(bool reevaluate)
    {
        double[] lower = [-5, 2.5, -1], upper = [5, 2.5, 3];
        var real = new RealCodedGeneticAlgorithm(lower, upper,
            new() { PopulationSize = 11, EliteCount = 2, TournamentSize = 3, MaxGenerations = 25, Seed = 5, ReevaluateElites = reevaluate });
        var general = new GeneticAlgorithm(new GenomeLayout(realLowerBounds: lower, realUpperBounds: upper),
            new() { PopulationSize = 11, EliteCount = 2, TournamentSize = 3, MaxGenerations = 25, Seed = 5, ReevaluateElites = reevaluate });
        while (!real.IsFinished)
        {
            var a = real.Ask().ToMatrix();
            var b = general.Ask().Reals.ToMatrix();
            Assert.Equal(a.AsSpan().ToArray(), b.AsSpan().ToArray());
            var values = new double[a.Columns];
            for (int j = 0; j < a.Columns; j++)
            {
                var x = a.Column(j);
                values[j] = x[0] * x[0] + Math.Sin(3 * x[2]);
            }

            real.Tell(values);
            general.Tell(values);
        }

        Assert.True(general.IsFinished);
        Assert.Equal(real.BestPoint.ToVector().ToArray(), general.BestGenome.Reals.ToArray());
    }

    [Fact]
    public void OmittedMutationProbabilityIsOneOverTheMutationUnits()
    {
        var implicitRate = new GeneticAlgorithm(Mixed, new() { PopulationSize = 9, Seed = 3, MaxGenerations = 10 });
        var explicitRate = new GeneticAlgorithm(Mixed, new() { PopulationSize = 9, Seed = 3, MaxGenerations = 10, MutationProbability = 1.0 / 16 });
        while (!implicitRate.IsFinished)
        {
            var a = implicitRate.Ask();
            Assert.Equal(Describe(explicitRate.Ask()), Describe(a));
            var values = Evaluate(a, MixedObjective);
            implicitRate.Tell(values);
            explicitRate.Tell(values);
        }
    }

    // ---------- 手順 ----------

    [Fact]
    public void SameSeedReproducesMixedCandidates()
    {
        var options = new GeneticAlgorithmOptions { PopulationSize = 13, EliteCount = 1, MaxGenerations = 20, CrossoverProbability = 1, Seed = 99 };
        var a = new GeneticAlgorithm(Mixed, options);
        var b = new GeneticAlgorithm(Mixed, options);
        while (!a.IsFinished)
        {
            var batchA = a.Ask();
            var batchB = b.Ask();
            Assert.Equal(Describe(batchA), Describe(batchB));
            Assert.Equal(Describe(batchA), Describe(a.Ask()));   // Tell までは同じ候補
            a.Tell(Evaluate(batchA, MixedObjective));
            b.Tell(Evaluate(batchB, MixedObjective));
        }

        Assert.Equal(a.BestFitness, b.BestFitness);
    }

    [Fact]
    public void CandidatesAreAlwaysValidAndFixedGenesKeepTheirValues()
    {
        var ga = new GeneticAlgorithm(Mixed, new() { PopulationSize = 20, MaxGenerations = 40, CrossoverProbability = 1, MutationProbability = 1, Seed = 8 });
        while (!ga.IsFinished)
        {
            var batch = ga.Ask();
            for (int j = 0; j < batch.Count; j++)
            {
                GenomeView g = batch[j];
                Assert.InRange(g.Reals[0], -1, 1);
                Assert.Equal(0.25, g.Reals[1]);
                Assert.InRange(g.Integers[0], long.MinValue, long.MinValue + 5);
                Assert.InRange(g.Integers[1], -3, 3);
                Assert.Equal(7, g.Integers[2]);
                Assert.Equal(0, g.Categories[0]);
                Assert.InRange(g.Categories[1], 0, 3);
                Assert.InRange(g.Categories[2], 0, 1);
                for (int k = 0; k < Mixed.PermutationCount; k++)
                {
                    Assert.Equal(Enumerable.Range(0, Mixed.PermutationLengths[k]), g.Permutation(k).ToArray().Order());
                }
            }

            ga.Tell(Evaluate(batch, MixedObjective));
        }
    }

    [Fact]
    public void PopulationHoldsTheElitesFollowedByTheOffspring()
    {
        var ga = new GeneticAlgorithm(Mixed, new() { PopulationSize = 10, EliteCount = 3, Seed = 4 });
        ga.Tell(Evaluate(ga.Ask(), MixedObjective));
        var previousFitness = ga.PopulationFitness.ToArray();
        string[] elites = Enumerable.Range(0, 10).OrderBy(i => previousFitness[i]).ThenBy(i => i).Take(3)
            .Select(i => Describe(ga.Population[i])).ToArray();

        var offspring = ga.Ask();
        Assert.Equal(7, offspring.Count);
        string children = Describe(offspring);
        var values = Evaluate(offspring, MixedObjective);
        ga.Tell(values);
        for (int k = 0; k < 3; k++)
        {
            Assert.Equal(elites[k], Describe(ga.Population[k]));
        }

        Assert.Equal(values, ga.PopulationFitness.ToArray()[3..]);
        Assert.EndsWith(children, Describe(ga.Population));
    }

    /// <summary>エリートを評価し直す設定では、候補の先頭がエリートの複製で、その値は新しい評価に置き換わる</summary>
    [Fact]
    public void ReevaluatedElitesComeFirstAndTakeTheNewValues()
    {
        var ga = new GeneticAlgorithm(Mixed, new() { PopulationSize = 8, EliteCount = 3, Seed = 6, ReevaluateElites = true, MaxGenerations = 4 });
        ga.Tell(Evaluate(ga.Ask(), MixedObjective));
        var fitness = ga.PopulationFitness.ToArray();
        int[] best = Enumerable.Range(0, 8).OrderBy(i => fitness[i]).ThenBy(i => i).Take(3).ToArray();
        string[] elites = best.Select(i => Describe(ga.Population[i])).ToArray();

        var batch = ga.Ask();
        Assert.Equal(8, batch.Count);
        for (int k = 0; k < 3; k++)
        {
            Assert.Equal(elites[k], Describe(batch[k]));
        }

        double[] values = [100, 101, 102, 0, 1, 2, 3, 4];   // エリートには前より悪い値が付く
        ga.Tell(values);
        Assert.Equal(values, ga.PopulationFitness.ToArray());
        Assert.Equal(16, ga.Evaluations);
        while (!ga.IsFinished)
        {
            ga.Tell(Evaluate(ga.Ask(), MixedObjective));
        }

        Assert.Equal(8 * 4, ga.Evaluations);
    }

    [Fact]
    public void BatchesAndGenomesExposeTheirSections()
    {
        var ga = new GeneticAlgorithm(new GenomeLayout(bitCount: 3, permutationLengths: [4]), new() { PopulationSize = 4, Seed = 1 });
        var batch = ga.Ask();
        Assert.Equal(0, batch.Reals.Rows);
        Assert.Equal(4, batch.Reals.Columns);
        Assert.Throws<ArgumentOutOfRangeException>(() => { _ = ga.Ask()[4]; });
        Assert.Throws<ArgumentOutOfRangeException>(() => { _ = ga.Ask()[-1]; });
        Assert.Equal(3, batch[0].Bits.Length);
        Assert.True(batch[0].Reals.IsEmpty && batch[0].Integers.IsEmpty && batch[0].Categories.IsEmpty);

        Genome copy = batch[2].ToGenome();
        int[] permutation = copy.Permutation(0).ToArray();
        ga.Tell([3, 2, 1, 0]);
        ga.Tell(Evaluate(ga.Ask(), g => g.Bits[0] ? 0 : 1));
        Assert.Equal(permutation, copy.Permutation(0).ToArray());   // コピーは後の世代の影響を受けない
        Assert.Equal(copy.Permutations.ToArray(), copy.AsView().Permutations.ToArray());
        Assert.Same(ga.Layout, copy.Layout);
    }

    [Fact]
    public void RunMatchesTheAskTellLoopForDelegatesAndStructs()
    {
        var options = new GeneticAlgorithmOptions { PopulationSize = 12, MaxGenerations = 15, Seed = 21 };
        var viaDelegate = new GeneticAlgorithm(Mixed, options).Run(MixedObjective);
        var viaStruct = new GeneticAlgorithm(Mixed, options).Run(new MixedStruct());
        var manual = new GeneticAlgorithm(Mixed, options);
        while (!manual.IsFinished)
        {
            manual.Tell(Evaluate(manual.Ask(), MixedObjective));
        }

        Assert.Equal(manual.BestFitness, viaDelegate.BestFitness);
        Assert.Equal(manual.BestFitness, viaStruct.BestFitness);
        Assert.Equal(Describe(manual.BestGenome), Describe(viaDelegate.BestPoint.AsView()));
        Assert.Equal(Describe(viaStruct.BestPoint.AsView()), Describe(viaDelegate.BestPoint.AsView()));
        Assert.Equal((manual.Generations, manual.Evaluations, manual.Termination),
            (viaDelegate.Generations, viaDelegate.Evaluations, viaDelegate.Termination));
    }

    private readonly struct MixedStruct : IGenomeObjective
    {
        public double Evaluate(GenomeView genome) => MixedObjective(genome);
    }

    [Fact]
    public void ProtocolErrorsAreReported()
    {
        var ga = new GeneticAlgorithm(Mixed, new() { PopulationSize = 4, MaxGenerations = 1 });
        Assert.Throws<InvalidOperationException>(() => ga.BestFitness);
        Assert.Throws<InvalidOperationException>(() => ga.Tell([1, 2, 3, 4]));
        Assert.Throws<ArgumentException>(() => ga.Tell(Evaluate(ga.Ask(), MixedObjective).AsSpan(1)));
        ga.Tell(Evaluate(ga.Ask(), MixedObjective));
        Assert.Equal(GeneticAlgorithmTermination.MaxGenerations, ga.Termination);
        Assert.Throws<InvalidOperationException>(() => ga.Ask());
    }

    // ---------- 収束（固定シード） ----------

    public static TheoryData<ulong> Seeds => new() { 1, 2, 3 };

    [Theory]
    [MemberData(nameof(Seeds))]
    public void FindsTheOptimumOfDiscreteProblems(ulong seed)
    {
        long[] target = [-37, 0, 12, 99, -100, 5, 63, -8];
        var integers = new GeneticAlgorithm(
            new GenomeLayout(integerLowerBounds: Enumerable.Repeat(-100L, 8).ToArray(), integerUpperBounds: Enumerable.Repeat(100L, 8).ToArray()),
            new() { PopulationSize = 40, MaxGenerations = 300, TargetFitness = 0, Seed = seed })
            .Run(g =>
            {
                double sum = 0;
                for (int i = 0; i < 8; i++)
                {
                    sum += Math.Pow(g.Integers[i] - target[i], 2);
                }

                return sum;
            });
        Assert.Equal(GeneticAlgorithmTermination.TargetReached, integers.Termination);
        Assert.Equal(target, integers.BestPoint.Integers.ToArray());

        var oneMax = new GeneticAlgorithm(new GenomeLayout(bitCount: 64), new() { PopulationSize = 40, MaxGenerations = 300, TargetFitness = 0, Seed = seed })
            .Run(g => g.Bits.Count(false));
        Assert.Equal(GeneticAlgorithmTermination.TargetReached, oneMax.Termination);

        int[] counts = [2, 3, 5, 7, 11, 3, 3, 4, 9, 2, 6, 5];
        int[] wanted = counts.Select((k, i) => (i * 7 + 3) % k).ToArray();
        var categories = new GeneticAlgorithm(new GenomeLayout(categoryCounts: counts), new() { PopulationSize = 40, MaxGenerations = 300, TargetFitness = 0, Seed = seed })
            .Run(g =>
            {
                int mismatches = 0;
                for (int i = 0; i < counts.Length; i++)
                {
                    if (g.Categories[i] != wanted[i])
                    {
                        mismatches++;
                    }
                }

                return mismatches;
            });
        Assert.Equal(wanted, categories.BestPoint.Categories.ToArray());
    }

    /// <summary>円周上の 12 都市（番号と円周上の順を混ぜて置く）の巡回路。最短は円周の順で、長さは 2n·sin(π/n)</summary>
    [Theory]
    [MemberData(nameof(Seeds))]
    public void FindsTheShortestTourOnACircle(ulong seed)
    {
        const int n = 12;
        double[] x = new double[n], y = new double[n];
        for (int i = 0; i < n; i++)
        {
            double angle = 2 * Math.PI * (i * 5 % n) / n;
            (x[i], y[i]) = (Math.Cos(angle), Math.Sin(angle));
        }

        double shortest = 2 * n * Math.Sin(Math.PI / n);
        var result = new GeneticAlgorithm(new GenomeLayout(permutationLengths: [n]),
            new() { PopulationSize = 60, MaxGenerations = 400, TargetFitness = shortest * (1 + 1e-12), Seed = seed })
            .Run(g =>
            {
                ReadOnlySpan<int> tour = g.Permutation(0);
                double length = 0;
                for (int i = 0; i < n; i++)
                {
                    int a = tour[i], b = tour[(i + 1) % n];
                    length += Math.Sqrt((x[a] - x[b]) * (x[a] - x[b]) + (y[a] - y[b]) * (y[a] - y[b]));
                }

                return length;
            });
        Assert.Equal(GeneticAlgorithmTermination.TargetReached, result.Termination);
    }

    [Theory]
    [MemberData(nameof(Seeds))]
    public void FindsTheOptimumOfAMixedProblem(ulong seed)
    {
        var result = new GeneticAlgorithm(Mixed, new() { PopulationSize = 40, MaxGenerations = 300, Seed = seed }).Run(MixedObjective);
        GenomeView best = result.BestPoint.AsView();
        Assert.True(result.BestFitness < 0.01, $"best {result.BestFitness}");
        Assert.Equal(2, best.Integers[1]);
        Assert.Equal(3, best.Categories[1]);
        Assert.All(best.Bits.ToArray(), bit => Assert.True(bit));
        Assert.Equal(Enumerable.Range(0, 6), best.Permutation(2).ToArray());
    }

    // ---------- 強化学習: 表形式の方策の探索（滑る FrozenLake 4×4） ----------

    /// <summary>
    /// 4×4 の凍った湖（S: 開始、F: 氷、H: 穴、G: ゴール）。行動（0: 左、1: 下、2: 右、3: 上）は 1/3 ずつの確率で、意図した向きとその左右の向きに進む。
    /// 方策は状態ごとの行動（カテゴリの遺伝子 16 個）で、評価はゴールに着いたエピソードの割合（100 歩で打ち切り）。
    /// </summary>
    private static class FrozenLake
    {
        public const int States = 16, Horizon = 100;
        private const string Map = "SFFFFHFHFFFHHFFG";

        private static int Move(int state, int action)
        {
            int row = state / 4, column = state % 4;
            switch (action)
            {
                case 0: column = Math.Max(column - 1, 0); break;
                case 1: row = Math.Min(row + 1, 3); break;
                case 2: column = Math.Min(column + 1, 3); break;
                default: row = Math.Max(row - 1, 0); break;
            }

            return row * 4 + column;
        }

        private static bool Terminal(int state) => Map[state] is 'H' or 'G';

        /// <summary>乱数源 <paramref name="random"/> の環境で1エピソード実行し、ゴールに着けば 1</summary>
        public static double Episode(ReadOnlySpan<int> policy, Philox4x32Random random)
        {
            int state = 0;
            for (int t = 0; t < Horizon; t++)
            {
                state = Move(state, (policy[state] + random.NextInt32(3) + 3) % 4);
                if (Terminal(state))
                {
                    return Map[state] == 'G' ? 1 : 0;
                }
            }

            return 0;
        }

        /// <summary><see cref="Horizon"/> 歩以内にゴールに着く確率（状態の分布を前へ進めて正確に求める）</summary>
        public static double SuccessProbability(ReadOnlySpan<int> policy)
        {
            var p = new double[States];
            p[0] = 1;
            double success = 0;
            for (int t = 0; t < Horizon; t++)
            {
                var q = new double[States];
                for (int s = 0; s < States; s++)
                {
                    if (p[s] == 0 || Terminal(s))
                    {
                        continue;
                    }

                    for (int slip = -1; slip <= 1; slip++)
                    {
                        q[Move(s, (policy[s] + slip + 4) % 4)] += p[s] / 3;
                    }
                }

                for (int s = 0; s < States; s++)
                {
                    if (Map[s] == 'G')
                    {
                        success += q[s];
                    }

                    if (Terminal(s))
                    {
                        q[s] = 0;
                    }
                }

                p = q;
            }

            return success;
        }

        /// <summary>価値反復で求めた（打ち切りのない）最適な方策</summary>
        public static int[] ValueIterationPolicy()
        {
            var value = new double[States];
            var policy = new int[States];
            for (int iteration = 0; iteration < 2000; iteration++)
            {
                for (int s = 0; s < States; s++)
                {
                    if (Terminal(s))
                    {
                        value[s] = Map[s] == 'G' ? 1 : 0;
                        continue;
                    }

                    double best = -1;
                    for (int a = 0; a < 4; a++)
                    {
                        double v = 0;
                        for (int slip = -1; slip <= 1; slip++)
                        {
                            v += value[Move(s, (a + slip + 4) % 4)] / 3;
                        }

                        if (v > best + 1e-15)
                        {
                            best = v;
                            policy[s] = a;
                        }
                    }

                    value[s] = best;
                }
            }

            return policy;
        }
    }

    /// <summary>
    /// 世代の評価用のシードから全候補に同じエピソードの列（共通乱数）を作り、20 エピソードの成功率の符号を反転して最小化する
    /// </summary>
    private static GeneticAlgorithm SearchFrozenLakePolicy(ulong seed, bool reevaluate)
    {
        var ga = new GeneticAlgorithm(new GenomeLayout(categoryCounts: Enumerable.Repeat(4, FrozenLake.States).ToArray()),
            new() { PopulationSize = 50, MaxGenerations = 60, Seed = seed, ReevaluateElites = reevaluate });
        var values = new double[50];
        while (!ga.IsFinished)
        {
            var batch = ga.Ask();
            ulong evaluationSeed = ga.EvaluationSeed;
            for (int j = 0; j < batch.Count; j++)
            {
                double successes = 0;
                for (int episode = 0; episode < 20; episode++)
                {
                    successes += FrozenLake.Episode(batch[j].Categories, new Philox4x32Random(evaluationSeed, (ulong)episode));
                }

                values[j] = -successes / 20;
            }

            ga.Tell(values.AsSpan(0, batch.Count));
        }

        return ga;
    }

    [Fact]
    public void NoisyPolicySearchIsReproducibleThroughTheEvaluationSeed()
    {
        var a = SearchFrozenLakePolicy(9, reevaluate: true);
        var b = SearchFrozenLakePolicy(9, reevaluate: true);
        Assert.Equal(a.BestFitness, b.BestFitness);
        Assert.Equal(a.BestGenome.Categories.ToArray(), b.BestGenome.Categories.ToArray());
        Assert.Equal(a.PopulationFitness.ToArray(), b.PopulationFitness.ToArray());
    }

    /// <summary>
    /// 雑音のある評価でも、見つけた方策の正確な成功確率は、価値反復の方策（100 歩で打ち切ると 0.740）の 80% 以上（シード 1〜20 の 40 回の試行では最小 0.639）。
    /// 記録された最良値（20 エピソードの成功率）は運のよい評価に偏るので、正確な確率より大きい
    /// </summary>
    [Theory]
    [InlineData(1UL, false)]
    [InlineData(2UL, false)]
    [InlineData(3UL, false)]
    [InlineData(1UL, true)]
    [InlineData(2UL, true)]
    [InlineData(3UL, true)]
    public void FindsANearOptimalPolicyFromNoisyEpisodes(ulong seed, bool reevaluate)
    {
        double optimal = FrozenLake.SuccessProbability(FrozenLake.ValueIterationPolicy());
        Assert.Equal(0.740, optimal, 0.001);

        var ga = SearchFrozenLakePolicy(seed, reevaluate);
        double found = FrozenLake.SuccessProbability(ga.BestGenome.Categories);
        Assert.True(found >= 0.8 * optimal, $"found {found}, optimal {optimal}");
        Assert.True(-ga.BestFitness >= found, $"recorded {-ga.BestFitness}, exact {found}");
    }
}
