using System.Reflection;
using Tuna.Optimization.Evolutionary;
using Tuna.RandomSource;

namespace Tuna.Tests.Optimization;

/// <summary>
/// 整数・二値・カテゴリ・順列の遺伝的操作（内部の DiscreteVariation）が、導出のもとにした確率分布にしたがうことを確認する。
/// 期待値は SBX・多項式突然変異の分布関数を丸めの区間で積分した閉形式と、一様な選び方の組み合わせの数から求める。
/// 標本数 n の経験確率の標準偏差は高々 1/(2√n) なので、その 5 倍を許容する。
/// </summary>
[Trait("Optimization", "GeneticAlgorithm")]
public class DiscreteVariationTests
{
    private const int Samples = 200_000;
    private static readonly double Tolerance = 5 * 0.5 / Math.Sqrt(Samples);

    private delegate void IntegerCrossover(ReadOnlySpan<long> parent1, ReadOnlySpan<long> parent2, Span<long> child1, Span<long> child2,
        ReadOnlySpan<long> lower, ReadOnlySpan<long> upper, double exponent, Philox4x32Random random);

    private delegate void IntegerMutation(Span<long> x, ReadOnlySpan<long> lower, ReadOnlySpan<long> upper,
        double probability, double exponent, Philox4x32Random random);

    private delegate void Swap<T>(Span<T> child1, Span<T> child2, Philox4x32Random random);

    private delegate void BitMutation(Span<bool> x, double probability, Philox4x32Random random);

    private delegate void CategoryMutation(Span<int> x, ReadOnlySpan<int> counts, double probability, Philox4x32Random random);

    private delegate void RandomOrderCrossover(ReadOnlySpan<int> parent1, ReadOnlySpan<int> parent2, Span<int> child1, Span<int> child2,
        Philox4x32Random random, Span<bool> scratch);

    private delegate void FixedOrderCrossover(ReadOnlySpan<int> donor, ReadOnlySpan<int> other, Span<int> child, int start, int end, Span<bool> scratch);

    private delegate void Inversion(Span<int> x, double probability, Philox4x32Random random);

    private static readonly Type Variation =
        typeof(GeneticAlgorithm).Assembly.GetType("Tuna.Optimization.Evolutionary.DiscreteVariation")!;

    private static T Internal<T>(string name, params Type[] parameters) where T : Delegate =>
        Variation.GetMethod(name, BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic, parameters)!.CreateDelegate<T>();

    private static readonly IntegerCrossover CrossIntegers = Internal<IntegerCrossover>("CrossIntegers",
        typeof(ReadOnlySpan<long>), typeof(ReadOnlySpan<long>), typeof(Span<long>), typeof(Span<long>),
        typeof(ReadOnlySpan<long>), typeof(ReadOnlySpan<long>), typeof(double), typeof(Philox4x32Random));

    private static readonly IntegerMutation MutateIntegers = Internal<IntegerMutation>("MutateIntegers",
        typeof(Span<long>), typeof(ReadOnlySpan<long>), typeof(ReadOnlySpan<long>), typeof(double), typeof(double), typeof(Philox4x32Random));

    private static readonly Swap<int> UniformCrossover = Variation
        .GetMethod("UniformCrossover", BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)!
        .MakeGenericMethod(typeof(int)).CreateDelegate<Swap<int>>();

    private static readonly BitMutation FlipBits = Internal<BitMutation>("FlipBits",
        typeof(Span<bool>), typeof(double), typeof(Philox4x32Random));

    private static readonly CategoryMutation MutateCategories = Internal<CategoryMutation>("MutateCategories",
        typeof(Span<int>), typeof(ReadOnlySpan<int>), typeof(double), typeof(Philox4x32Random));

    private static readonly RandomOrderCrossover OrderCrossover = Internal<RandomOrderCrossover>("OrderCrossover",
        typeof(ReadOnlySpan<int>), typeof(ReadOnlySpan<int>), typeof(Span<int>), typeof(Span<int>), typeof(Philox4x32Random), typeof(Span<bool>));

    private static readonly FixedOrderCrossover OrderCrossoverAt = Internal<FixedOrderCrossover>("OrderCrossover",
        typeof(ReadOnlySpan<int>), typeof(ReadOnlySpan<int>), typeof(Span<int>), typeof(int), typeof(int), typeof(Span<bool>));

    private static readonly Inversion Invert = Internal<Inversion>("Invert", typeof(Span<int>), typeof(double), typeof(Philox4x32Random));

    // ---------- 整数 ----------

    /// <summary>広がり係数 β の分布関数（β ≤ 1 で β^e/2、β &gt; 1 で 1 − β^−e/2）を β_max で切り詰めて正規化したもの</summary>
    private static double TruncatedSpreadCdf(double beta, double maxBeta, double exponent)
    {
        static double G(double b, double e) => b <= 1 ? 0.5 * Math.Pow(b, e) : 1 - 0.5 * Math.Pow(b, -e);
        beta = Math.Clamp(beta, 0, maxBeta);
        return G(beta, exponent) / G(maxBeta, exponent);
    }

    /// <summary>
    /// 親 3 と 7、範囲 [0, 10]（連続値の範囲 [−1/2, 21/2]）、η = 2 の整数の SBX で、小さい側の子が k になる確率。
    /// 交叉しない場合（確率 1/2）は親の 3。交叉した場合、小さい側の子は m − βs/2（m = 5、s = 4）を丸めた値で、
    /// k に丸まるのは β が (2(m − k − 1/2)/s, 2(m − k + 1/2)/s] にあるとき（β_max = 1 + 2(3 + 1/2)/4）
    /// </summary>
    [Fact]
    public void IntegerCrossoverFollowsTheRoundedSpreadFactorDistribution()
    {
        const double exponent = 3, m = 5, s = 4, maxBeta = 1 + 2 * 3.5 / 4;
        var random = new Philox4x32Random(20261013);
        long[] c1 = new long[1], c2 = new long[1];
        var counts = new int[11];
        for (int i = 0; i < Samples; i++)
        {
            c1[0] = 3;   // 交叉は子に親を写してから呼ぶ
            c2[0] = 7;
            CrossIntegers([3], [7], c1, c2, [0], [10], exponent, random);
            Assert.InRange(c1[0], 0, 10);
            Assert.InRange(c2[0], 0, 10);
            counts[Math.Min(c1[0], c2[0])]++;
        }

        for (int k = 0; k <= 5; k++)
        {
            double crossed = TruncatedSpreadCdf(2 * (m - k + 0.5) / s, maxBeta, exponent) - TruncatedSpreadCdf(2 * (m - k - 0.5) / s, maxBeta, exponent);
            double expected = 0.5 * crossed + (k == 3 ? 0.5 : 0);
            Assert.Equal(expected, (double)counts[k] / Samples, Tolerance);
        }

        Assert.Equal(0, counts.AsSpan(6).ToArray().Sum());
    }

    /// <summary>多項式突然変異の変化量 δ（幅で正規化）の分布関数。負の側を [−below, 0]、正の側を [0, above] に切り詰め、それぞれに 1/2 を割り当てる</summary>
    private static double TruncatedPolynomialCdf(double delta, double below, double above, double exponent)
    {
        delta = Math.Clamp(delta, -below, above);
        if (delta <= 0)
        {
            double floor = Math.Pow(1 - below, exponent);
            return (Math.Pow(1 + delta, exponent) - floor) / (2 * (1 - floor));
        }

        double ceiling = Math.Pow(1 - above, exponent);
        return 1 - (Math.Pow(1 - delta, exponent) - ceiling) / (2 * (1 - ceiling));
    }

    /// <summary>
    /// 値 2、範囲 [0, 10]（連続値の範囲 [−1/2, 21/2]、幅 11）、η = 1 の整数の多項式突然変異で、結果が k になる確率は
    /// δ が ((k − 1/2 − 2)/11, (k + 1/2 − 2)/11] にある確率
    /// </summary>
    [Fact]
    public void IntegerMutationFollowsTheRoundedPolynomialDistribution()
    {
        const double exponent = 2, width = 11, below = 2.5 / width, above = 8.5 / width;
        var random = new Philox4x32Random(99);
        long[] x = new long[1];
        var counts = new int[11];
        for (int i = 0; i < Samples; i++)
        {
            x[0] = 2;
            MutateIntegers(x, [0], [10], 1, exponent, random);
            Assert.InRange(x[0], 0, 10);
            counts[x[0]]++;
        }

        for (int k = 0; k <= 10; k++)
        {
            double expected = TruncatedPolynomialCdf((k + 0.5 - 2) / width, below, above, exponent)
                - TruncatedPolynomialCdf((k - 0.5 - 2) / width, below, above, exponent);
            Assert.Equal(expected, (double)counts[k] / Samples, Tolerance);
        }
    }

    [Fact]
    public void IntegerOperatorsStayInsideExtremeBoundsAndKeepFixedValues()
    {
        var random = new Philox4x32Random(5);
        long[] lower = [long.MinValue, long.MaxValue - 3, -(1L << 52), 42], upper = [long.MinValue + 2, long.MaxValue, 1L << 52, 42];
        long[] p1 = [long.MinValue, long.MaxValue, -(1L << 52), 42], p2 = [long.MinValue + 2, long.MaxValue - 3, 1L << 52, 42];
        long[] c1 = new long[4], c2 = new long[4];
        for (int i = 0; i < 20_000; i++)
        {
            p1.CopyTo(c1);
            p2.CopyTo(c2);
            CrossIntegers(p1, p2, c1, c2, lower, upper, 3, random);
            MutateIntegers(c1, lower, upper, 1, 21, random);
            MutateIntegers(c2, lower, upper, 1, 1, random);
            for (int d = 0; d < 4; d++)
            {
                Assert.InRange(c1[d], lower[d], upper[d]);
                Assert.InRange(c2[d], lower[d], upper[d]);
            }

            Assert.Equal(42, c1[3]);
            Assert.Equal(42, c2[3]);
        }
    }

    // ---------- 二値・カテゴリ ----------

    [Fact]
    public void UniformCrossoverSwapsEachGeneWithProbabilityOneHalf()
    {
        var random = new Philox4x32Random(17);
        int[] c1 = new int[3], c2 = new int[3];
        var swapped = new int[3];
        int bothSwapped = 0;
        for (int i = 0; i < Samples; i++)
        {
            int[] a = [0, 0, 0], b = [1, 1, 1];
            a.CopyTo(c1);
            b.CopyTo(c2);
            UniformCrossover(c1, c2, random);
            for (int d = 0; d < 3; d++)
            {
                Assert.Equal(1, c1[d] + c2[d]);   // 値は入れ替わるだけ
                swapped[d] += c1[d];
            }

            if (c1[0] == 1 && c1[1] == 1)
            {
                bothSwapped++;
            }
        }

        foreach (int count in swapped)
        {
            Assert.Equal(0.5, (double)count / Samples, Tolerance);
        }

        Assert.Equal(0.25, (double)bothSwapped / Samples, Tolerance);   // 遺伝子ごとに独立
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.125)]
    [InlineData(1.0)]
    public void BitFlipChangesEachBitWithTheGivenProbability(double probability)
    {
        var random = new Philox4x32Random(23);
        bool[] x = new bool[2];
        int flipped = 0;
        for (int i = 0; i < Samples; i++)
        {
            x[0] = false;
            x[1] = true;
            FlipBits(x, probability, random);
            if (x[0])
            {
                flipped++;
            }

            if (!x[1])
            {
                flipped++;
            }
        }

        Assert.Equal(probability, flipped / (2.0 * Samples), Tolerance);
    }

    /// <summary>変えると決まった遺伝子は、今と異なる k − 1 個のカテゴリのどれかへ等しい確率で変わる。カテゴリが1つの遺伝子は変わらない</summary>
    [Fact]
    public void CategoryMutationMovesToAnotherCategoryUniformly()
    {
        var random = new Philox4x32Random(31);
        const double probability = 0.4;
        int[] counts = [5, 2, 1];
        int[] x = new int[3];
        var five = new int[5];
        int twoChanged = 0;
        for (int i = 0; i < Samples; i++)
        {
            x[0] = 3;
            x[1] = 1;
            x[2] = 0;
            MutateCategories(x, counts, probability, random);
            Assert.InRange(x[0], 0, 4);
            five[x[0]]++;
            if (x[1] == 0)
            {
                twoChanged++;
            }

            Assert.Equal(0, x[2]);
        }

        for (int k = 0; k < 5; k++)
        {
            Assert.Equal(k == 3 ? 1 - probability : probability / 4, (double)five[k] / Samples, Tolerance);
        }

        Assert.Equal(probability, (double)twoChanged / Samples, Tolerance);
    }

    // ---------- 順列 ----------

    /// <summary>
    /// 教科書でよく使われる例（親 123|4567|89 と 452|1876|93）。区間を写し、残りを2番目の切れ目の後ろから相手の親の順で埋める:
    /// 子は 218|4567|93 と 345|1876|92（ここでは 0 から数えた値）
    /// </summary>
    [Fact]
    public void OrderCrossoverReproducesTheTextbookExample()
    {
        int[] p1 = [0, 1, 2, 3, 4, 5, 6, 7, 8], p2 = [3, 4, 1, 0, 7, 6, 5, 8, 2];
        int[] child = new int[9];
        bool[] scratch = new bool[9];
        OrderCrossoverAt(p1, p2, child, 3, 6, scratch);
        Assert.Equal([1, 0, 7, 3, 4, 5, 6, 8, 2], child);
        OrderCrossoverAt(p2, p1, child, 3, 6, scratch);
        Assert.Equal([2, 3, 4, 0, 7, 6, 5, 8, 1], child);
    }

    [Fact]
    public void OrderCrossoverHandlesSegmentsAtTheEnds()
    {
        int[] p1 = [0, 1, 2, 3, 4], p2 = [4, 3, 2, 1, 0];
        int[] child = new int[5];
        bool[] scratch = new bool[5];
        OrderCrossoverAt(p1, p2, child, 0, 4, scratch);   // 全体を写す
        Assert.Equal(p1, child);
        OrderCrossoverAt(p1, p2, child, 3, 4, scratch);   // 区間が末尾まで: 先頭から相手の順で埋める
        Assert.Equal([2, 1, 0, 3, 4], child);
        OrderCrossoverAt(p1, p2, child, 0, 0, scratch);   // 長さ 1: 位置 1 から、相手の位置 1 以降の順（3, 2, 1, 4）で埋める
        Assert.Equal([0, 3, 2, 1, 4], child);
    }

    [Fact]
    public void RandomOrderCrossoverAlwaysGivesPermutations()
    {
        var random = new Philox4x32Random(41);
        int[] p1 = new int[8], p2 = new int[8], c1 = new int[8], c2 = new int[8];
        bool[] scratch = new bool[8];
        for (int i = 0; i < 20_000; i++)
        {
            random.Permutation(p1);
            random.Permutation(p2);
            OrderCrossover(p1, p2, c1, c2, random, scratch);
            Assert.Equal(Enumerable.Range(0, 8), c1.Order());
            Assert.Equal(Enumerable.Range(0, 8), c2.Order());
        }
    }

    /// <summary>反転は異なる2点 i &lt; j を一様に選ぶので、長さ 5 では 10 組のそれぞれが確率 1/10。逆にする区間の両端は、元と値が変わる最初と最後の位置</summary>
    [Fact]
    public void InversionReversesAUniformlyChosenSegment()
    {
        var random = new Philox4x32Random(43);
        int[] x = new int[5];
        var pairs = new int[5, 5];
        for (int s = 0; s < Samples; s++)
        {
            for (int k = 0; k < 5; k++)
            {
                x[k] = k;
            }

            Invert(x, 1, random);
            int i = Enumerable.Range(0, 5).First(k => x[k] != k), j = Enumerable.Range(0, 5).Last(k => x[k] != k);
            for (int k = 0; k < 5; k++)
            {
                Assert.Equal(k < i || k > j ? k : i + j - k, x[k]);
            }

            pairs[i, j]++;
        }

        for (int i = 0; i < 5; i++)
        {
            for (int j = i + 1; j < 5; j++)
            {
                Assert.Equal(0.1, (double)pairs[i, j] / Samples, Tolerance);
            }
        }
    }

    [Fact]
    public void InversionRespectsTheProbabilityAndShortPermutations()
    {
        var random = new Philox4x32Random(47);
        int changed = 0;
        for (int s = 0; s < Samples; s++)
        {
            int[] x = [0, 1, 2];
            Invert(x, 0.3, random);
            if (x[0] != 0 || x[2] != 2)
            {
                changed++;
            }
        }

        Assert.Equal(0.3, (double)changed / Samples, Tolerance);
        int[] two = [0, 1], one = [0];
        Invert(two, 1, random);
        Assert.Equal([1, 0], two);
        Invert(one, 1, random);
        Assert.Equal([0], one);
    }

    /// <summary>順列の初期化（コアの Fisher–Yates 法 <c>Permutation</c>）は 4! = 24 通りのそれぞれが確率 1/24</summary>
    [Fact]
    public void ShuffleIsUniformOverPermutations()
    {
        var random = new Philox4x32Random(53);
        int[] x = new int[4];
        var counts = new Dictionary<int, int>();
        for (int s = 0; s < Samples; s++)
        {
            random.Permutation(x);
            int key = ((x[0] * 4 + x[1]) * 4 + x[2]) * 4 + x[3];
            counts[key] = counts.GetValueOrDefault(key) + 1;
        }

        Assert.Equal(24, counts.Count);
        foreach (int count in counts.Values)
        {
            Assert.Equal(1.0 / 24, (double)count / Samples, Tolerance);
        }
    }
}
