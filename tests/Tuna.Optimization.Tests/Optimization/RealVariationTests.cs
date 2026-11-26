using System.Reflection;
using Tuna.Optimization.Evolutionary;
using Tuna.RandomSource;

namespace Tuna.Tests.Optimization;

/// <summary>
/// 実数値 GA の遺伝的操作（内部の RealVariation）が、導出のもとにした確率分布にしたがうことを確認する。
/// 期待値は SBX・多項式突然変異の分布関数を手で積分した閉形式。標本数 n の経験確率の標準偏差は高々 1/(2√n) なので、その 5 倍を許容する。
/// </summary>
[Trait("Optimization", "GeneticAlgorithm")]
public class RealVariationTests
{
    private const int Samples = 200_000;
    private static readonly double Tolerance = 5 * 0.5 / Math.Sqrt(Samples);

    private delegate void Crossover(ReadOnlySpan<double> parent1, ReadOnlySpan<double> parent2, Span<double> child1, Span<double> child2,
        ReadOnlySpan<double> lower, ReadOnlySpan<double> upper, double probability, double distributionIndex, Philox4x32Random random);

    private delegate void Mutation(Span<double> x, ReadOnlySpan<double> lower, ReadOnlySpan<double> upper,
        double probability, double distributionIndex, Philox4x32Random random);

    private delegate int Tournament(ReadOnlySpan<int> rank, int size, Philox4x32Random random);

    private static T Internal<T>(string name) where T : Delegate =>
        typeof(RealCodedGeneticAlgorithm).Assembly.GetType("Tuna.Optimization.Evolutionary.RealVariation")!
            .GetMethod(name, BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)!
            .CreateDelegate<T>();

    private static readonly Crossover Sbx = Internal<Crossover>("SimulatedBinaryCrossover");
    private static readonly Mutation Polynomial = Internal<Mutation>("PolynomialMutation");
    private static readonly Tournament Select = Internal<Tournament>("Tournament");

    /// <summary>範囲が十分広ければ、広がり係数 β = |c₂ − c₁| / |p₂ − p₁| の分布関数は β ≤ 1 で β^(η+1)/2、β &gt; 1 で 1 − β^−(η+1)/2</summary>
    [Fact]
    public void WideBoundCrossoverFollowsTheSpreadFactorDistribution()
    {
        var random = new Philox4x32Random(20261013);
        double[] lower = [-1e9], upper = [1e9], c1 = new double[1], c2 = new double[1];
        int crossed = 0, atHalf = 0, atOne = 0, atTwo = 0;
        for (int i = 0; i < Samples; i++)
        {
            Sbx([0.0], [1.0], c1, c2, lower, upper, 1, 2, random);
            if (c1[0] == 0 && c2[0] == 1)
            {
                continue;   // この遺伝子は交叉しなかった（確率 1/2）
            }

            crossed++;
            double beta = Math.Abs(c2[0] - c1[0]);
            Assert.Equal(1.0, c1[0] + c2[0], 1e-12);   // 子は中点に対称
            if (beta <= 0.5)
            {
                atHalf++;
            }

            if (beta <= 1)
            {
                atOne++;
            }

            if (beta <= 2)
            {
                atTwo++;
            }
        }

        Assert.Equal(0.5, (double)crossed / Samples, Tolerance);
        Assert.Equal(0.5 * Math.Pow(0.5, 3), (double)atHalf / crossed, Tolerance);
        Assert.Equal(0.5, (double)atOne / crossed, Tolerance);
        Assert.Equal(1 - 0.5 * Math.Pow(2, -3), (double)atTwo / crossed, Tolerance);
    }

    /// <summary>
    /// 範囲に近い側の子は β ≤ β_max に切り詰めた分布にしたがう。親 0.1, 0.3、範囲 [0, 1]、η = 2 では β_max = 2 で、
    /// P(β ≤ 1/2) = (0.5^3 / 2) / (1 − 2^−3 / 2)
    /// </summary>
    [Fact]
    public void BoundedCrossoverTruncatesTheSpreadFactorAndStaysInRange()
    {
        var random = new Philox4x32Random(7);
        double[] lower = [0], upper = [1], c1 = new double[1], c2 = new double[1];
        int crossed = 0, small = 0;
        for (int i = 0; i < Samples; i++)
        {
            Sbx([0.1], [0.3], c1, c2, lower, upper, 1, 2, random);
            Assert.InRange(c1[0], 0, 1);
            Assert.InRange(c2[0], 0, 1);
            if (c1[0] == 0.1 && c2[0] == 0.3)
            {
                continue;
            }

            crossed++;
            double beta = (0.2 - Math.Min(c1[0], c2[0])) / 0.1;
            if (beta <= 0.5)
            {
                small++;
            }
        }

        Assert.Equal(0.5 * Math.Pow(0.5, 3) / (1 - 0.5 * Math.Pow(2, -3)), (double)small / crossed, Tolerance);
    }

    [Fact]
    public void CrossoverWithZeroProbabilityCopiesTheParents()
    {
        var random = new Philox4x32Random(1);
        double[] c1 = new double[3], c2 = new double[3];
        Sbx([0.1, 0.2, 0.3], [0.9, 0.8, 0.7], c1, c2, [0, 0, 0], [1, 1, 1], 0, 15, random);
        Assert.Equal([0.1, 0.2, 0.3], c1);
        Assert.Equal([0.9, 0.8, 0.7], c2);
    }

    /// <summary>
    /// y = 1/2、範囲 [0, 1]、η = 1 では変化量 δ の分布関数は、負の側で ((1 + δ)² − 1/4) / (3/4) / 2、正の側で 1/2 + (1 − (1 − δ)² ) / (3/4) / 2
    /// </summary>
    [Fact]
    public void MutationFollowsTheTruncatedPolynomialDistribution()
    {
        var random = new Philox4x32Random(99);
        double[] lower = [0], upper = [1], x = new double[1];
        int quarterBelow = 0, below = 0, quarterAbove = 0;
        for (int i = 0; i < Samples; i++)
        {
            x[0] = 0.5;
            Polynomial(x, lower, upper, 1, 1, random);
            Assert.InRange(x[0], 0, 1);
            double delta = x[0] - 0.5;
            if (delta <= -0.25)
            {
                quarterBelow++;
            }

            if (delta <= 0)
            {
                below++;
            }

            if (delta <= 0.25)
            {
                quarterAbove++;
            }
        }

        Assert.Equal(0.5 * (0.5625 - 0.25) / 0.75, (double)quarterBelow / Samples, Tolerance);
        Assert.Equal(0.5, (double)below / Samples, Tolerance);
        Assert.Equal(0.5 + 0.5 * (1 - 0.5625) / 0.75, (double)quarterAbove / Samples, Tolerance);
    }

    /// <summary>上限に近い点では正の側が [0, δ₂] に切り詰められる。y = 0.9、η = 1 で P(δ ≤ 0.05) = 1/2 + (1 − 0.95²) / (1 − 0.9²) / 2</summary>
    [Fact]
    public void MutationNearABoundTruncatesThatSide()
    {
        var random = new Philox4x32Random(5);
        double[] lower = [0], upper = [1], x = new double[1];
        int count = 0;
        for (int i = 0; i < Samples; i++)
        {
            x[0] = 0.9;
            Polynomial(x, lower, upper, 1, 1, random);
            Assert.InRange(x[0], 0, 1);
            if (x[0] - 0.9 <= 0.05)
            {
                count++;
            }
        }

        Assert.Equal(0.5 + 0.5 * (1 - 0.95 * 0.95) / (1 - 0.9 * 0.9), (double)count / Samples, Tolerance);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    public void MutationAtABoundStaysInRange(double start)
    {
        var random = new Philox4x32Random(3);
        double[] lower = [0], upper = [1], x = new double[1];
        for (int i = 0; i < 10_000; i++)
        {
            x[0] = start;
            Polynomial(x, lower, upper, 1, 20, random);
            Assert.InRange(x[0], 0, 1);
        }
    }

    [Fact]
    public void MutationLeavesFixedVariablesUnchanged()
    {
        var random = new Philox4x32Random(3);
        double[] x = [2.5, 0.5];
        for (int i = 0; i < 1000; i++)
        {
            Polynomial(x, [2.5, 0], [2.5, 1], 1, 20, random);
            Assert.Equal(2.5, x[0]);
        }
    }

    /// <summary>大きさ k のトーナメント（復元抽出）で最良の個体が勝つ確率は 1 − (1 − 1/N)^k</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    public void TournamentPicksTheBestWithTheExpectedProbability(int size)
    {
        var random = new Philox4x32Random(11);
        int[] rank = [2, 0, 3, 1];
        int wins = 0;
        for (int i = 0; i < Samples; i++)
        {
            if (Select(rank, size, random) == 1)
            {
                wins++;
            }
        }

        Assert.Equal(1 - Math.Pow(0.75, size), (double)wins / Samples, Tolerance);
    }
}
