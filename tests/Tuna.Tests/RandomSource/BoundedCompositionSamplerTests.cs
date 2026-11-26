using System.Numerics;
using Tuna.RandomSource;

namespace Tuna.Tests.RandomSource;

/// <summary>
/// 上下限と総和が決まった整数の組の一様な選択: 組の数 <c>N</c> が多倍長の数え上げと一致し、どの組もカイ二乗で見分けられない精度で確率 <c>1/N</c> で出て、
/// 制約を常に満たし、乱数を <c>NextDouble</c> で <c>n - 1</c> 個だけ使うこと
/// </summary>
public class BoundedCompositionSamplerTests
{
    /// <summary>
    /// 前の成分 <c>0, …, count - 1</c> で和が t になる組の数（t = 0, …, limit）を多倍長で数える。
    /// 漸化式 <c>M(t) = Σ_(v=0)^(cᵢ) M'(t - v)</c> を累積和の差で求める（整数なので引き算は正確）
    /// </summary>
    private static BigInteger[] PrefixCounts(long[] widths, int count, long limit)
    {
        var counts = new BigInteger[limit + 1];
        counts[0] = 1;
        var cumulative = new BigInteger[limit + 2];
        for (int i = 0; i < count; i++)
        {
            for (int t = 0; t <= limit; t++)
            {
                cumulative[t + 1] = cumulative[t] + counts[t];
            }

            for (long t = 0; t <= limit; t++)
            {
                counts[t] = cumulative[t + 1] - cumulative[Math.Max(0, t - widths[i])];
            }
        }

        return counts;
    }

    private static long[] Widths(long[] lower, long[] upper) => lower.Zip(upper, (l, u) => u - l).ToArray();

    // 和 m の組の数は、zᵢ = cᵢ - yᵢ で和 Σcᵢ - m の組の数に等しいので、小さいほうを数える
    private static BigInteger Count(long[] lower, long[] upper, long total)
    {
        var widths = Widths(lower, upper);
        long m = total - lower.Sum();
        m = Math.Min(m, widths.Sum() - m);
        return PrefixCounts(widths, lower.Length, m)[m];
    }

    private static void AssertSatisfies(long[] lower, long[] upper, long total, long[] x)
    {
        Int128 sum = 0;
        for (int i = 0; i < x.Length; i++)
        {
            Assert.InRange(x[i], lower[i], upper[i]);
            sum += x[i];
        }

        Assert.Equal((Int128)total, sum);
    }

    private static IEnumerable<long[]> Enumerate(long[] lower, long[] upper, long total)
    {
        var x = new long[lower.Length];
        return Recurse(0, total);

        IEnumerable<long[]> Recurse(int k, long remaining)
        {
            if (k == x.Length - 1)
            {
                if (remaining >= lower[k] && remaining <= upper[k])
                {
                    x[k] = remaining;
                    yield return x.ToArray();
                }

                yield break;
            }

            for (long v = lower[k]; v <= upper[k]; v++)
            {
                x[k] = v;
                foreach (var tuple in Recurse(k + 1, remaining - v))
                {
                    yield return tuple;
                }
            }
        }
    }

    [Fact]
    public void LogCountMatchesExactCounts()
    {
        var random = new Random(3);
        var cases = new List<(long[] Lower, long[] Upper, long Total)>
        {
            ([0], [5], 3),                                  // 1成分
            ([2, 2, 2], [2, 2, 2], 6),                      // 組は1つ
            ([0, 0, 0], [3, 3, 3], 0),                      // 和が下限の和
            ([0, 0, 0], [3, 3, 3], 9),                      // 和が上限の和
            ([-3, 1, 0, 5], [4, 1, 10, 9], 12),             // 幅 0 の成分と負の下限
            ([0, 0, 0, 0], [10, 2, 7, 4], 18),              // 補った組の側（和が Σcᵢ の半分より大きい）
            ([0, 0], [1000, 1000], 1000),
            ([.. Enumerable.Repeat(0L, 50)], [.. Enumerable.Repeat(200L, 50)], 4000),
            ([.. Enumerable.Repeat(0L, 300)], [.. Enumerable.Repeat(1L, 300)], 150),
            ([.. Enumerable.Repeat(0L, 30)], [.. Enumerable.Repeat(1000L, 30)], 29000),
        };
        for (int c = 0; c < 20; c++)
        {
            int n = random.Next(1, 12);
            long[] lower = Enumerable.Range(0, n).Select(_ => (long)random.Next(-50, 50)).ToArray();
            long[] upper = lower.Select(l => l + random.Next(0, 40)).ToArray();
            cases.Add((lower, upper, lower.Sum() + random.NextInt64(upper.Sum() - lower.Sum() + 1)));
        }

        foreach (var (lower, upper, total) in cases)
        {
            var sampler = new BoundedCompositionSampler(lower, upper, total);
            double expected = BigInteger.Log(Count(lower, upper, total));
            Assert.True(Math.Abs(sampler.LogCount - expected) <= 1e-13 * Math.Max(1, expected),
                $"n={lower.Length} s={total}: expected {expected:R}, actual {sampler.LogCount:R}");
            Assert.Equal(lower.Length, sampler.Length);
            Assert.Equal(total, sampler.Total);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EveryTupleIsEquallyLikely(bool philox)
    {
        RandomGenerator source = philox ? new Philox4x32Random(99) : new Pcg64DxsmRandom(99);
        var cases = new (long[] Lower, long[] Upper, long Total)[]
        {
            ([0, -1, 2], [3, 2, 4], 5),
            ([0, 0, 0, 0], [10, 2, 7, 4], 18),              // 補った組の側
            ([1, 1, 1, 1, 1], [3, 3, 3, 3, 3], 9),
            ([0, 5, 0], [6, 5, 6], 11),                     // 幅 0 の成分
            ([-2, -2], [7, 3], 1),
        };
        foreach (var (lower, upper, total) in cases)
        {
            var tuples = Enumerate(lower, upper, total).Select(t => string.Join(",", t)).ToArray();
            Assert.Equal(Count(lower, upper, total), new BigInteger(tuples.Length));
            var sampler = new BoundedCompositionSampler(lower, upper, total);
            int draws = tuples.Length * 300;
            var counts = tuples.ToDictionary(t => t, _ => 0);
            var x = new long[lower.Length];
            for (int k = 0; k < draws; k++)
            {
                sampler.Sample(source, x);
                string key = string.Join(",", x);
                Assert.True(counts.ContainsKey(key), key);
                counts[key]++;
            }

            double expected = (double)draws / tuples.Length;
            double chiSquare = counts.Values.Sum(c => (c - expected) * (c - expected) / expected);
            Assert.All(counts.Values, c => Assert.True(c > 0));
            Assert.True(chiSquare < RandomSourceExtensionsTests.ChiSquareCritical(Math.Max(3, tuples.Length - 1)),
                $"[{string.Join(",", lower)}]..[{string.Join(",", upper)}] s={total}: {tuples.Length} tuples, χ² = {chiSquare}");
        }
    }

    [Fact]
    public void MarginalsMatchExactProbabilitiesForLargeCounts()
    {
        // 組が数え上げられない大きさ（約 5 × 10¹⁷ 通り）で、最初と最後の成分の周辺分布を多倍長の比 N₁(m - v)/N₀(m) と比べる
        long[] lower = [.. Enumerable.Range(0, 20).Select(i => (long)(i % 3 - 1))];
        long[] upper = [.. lower.Select((l, i) => l + 5 + 3 * (i % 4))];
        long total = lower.Sum() + 70;
        var widths = Widths(lower, upper);
        var sampler = new BoundedCompositionSampler(lower, upper, total);
        var source = new Pcg64DxsmRandom(5);
        const int draws = 60000;
        var first = new int[widths[0] + 1];
        var last = new int[widths[^1] + 1];
        var x = new long[lower.Length];
        for (int k = 0; k < draws; k++)
        {
            sampler.Sample(source, x);
            AssertSatisfies(lower, upper, total, x);
            first[x[0] - lower[0]]++;
            last[x[^1] - lower[^1]]++;
        }

        CheckMarginal(first, PrefixCounts(widths[1..], widths.Length - 1, 70));
        CheckMarginal(last, PrefixCounts(widths[..^1], widths.Length - 1, 70));

        void CheckMarginal(int[] observed, BigInteger[] rest)
        {
            // P(y = v) = (残りの成分で和 m - v になる組の数)/N。期待度数が小さい値はまとめる
            BigInteger all = BigInteger.Zero;
            for (int v = 0; v < observed.Length; v++)
            {
                all += rest[70 - v];
            }

            double chiSquare = 0, pooledObserved = 0, pooledExpected = 0;
            int cells = 0;
            for (int v = 0; v < observed.Length; v++)
            {
                double p = rest[70 - v].IsZero ? 0 : Math.Exp(BigInteger.Log(rest[70 - v]) - BigInteger.Log(all));
                double expected = p * draws;
                if (expected < 20)
                {
                    pooledObserved += observed[v];
                    pooledExpected += expected;
                    continue;
                }

                chiSquare += (observed[v] - expected) * (observed[v] - expected) / expected;
                cells++;
            }

            if (pooledExpected >= 5)
            {
                chiSquare += (pooledObserved - pooledExpected) * (pooledObserved - pooledExpected) / pooledExpected;
                cells++;
            }

            Assert.True(chiSquare < RandomSourceExtensionsTests.ChiSquareCritical(Math.Max(3, cells - 1)), $"χ² = {chiSquare} ({cells} cells)");
        }
    }

    [Fact]
    public void SamplesUseNMinusOneUniformsAndReproduce()
    {
        long[] lower = [0, -4, 3, 0, 0, 10];
        long[] upper = [9, 4, 3, 100, 2, 20];
        var sampler = new BoundedCompositionSampler(lower, upper, 40);
        var source = new Philox4x32Random(17);
        var reference = new Philox4x32Random(17);
        var again = new Philox4x32Random(17);
        var x = new long[6];
        var y = new long[6];
        for (int k = 0; k < 100; k++)
        {
            sampler.Sample(source, x);
            sampler.Sample(again, y);
            Assert.Equal(x, y);
            AssertSatisfies(lower, upper, 40, x);
            for (int i = 0; i < 5; i++)
            {
                reference.NextDouble();
            }

            Assert.Equal(reference.GetState(), source.GetState());
        }

        // 1成分と、組が1つの場合（それでも n - 1 個使う）
        var single = new BoundedCompositionSampler([-5], [5], 2);
        var one = new long[1];
        single.Sample(source, one);
        Assert.Equal(2, one[0]);
        Assert.Equal(0.0, single.LogCount);
        Assert.Equal(reference.GetState(), source.GetState());
        var fixedTuple = new BoundedCompositionSampler([1, 2, 3], [1, 2, 3], 6);
        var three = new long[3];
        fixedTuple.Sample(source, three);
        Assert.Equal(new long[] { 1, 2, 3 }, three);
        Assert.Equal(0.0, fixedTuple.LogCount);
        reference.NextDouble();
        reference.NextDouble();
        Assert.Equal(reference.GetState(), source.GetState());
    }

    [Fact]
    public void ExtremeBoundsDoNotOverflow()
    {
        var source = new Pcg64DxsmRandom(1);
        var x = new long[2];

        // 幅が long を超える成分（2⁶⁴ - 1）と、Σlᵢ が long を下回る下限
        long[] lower = [long.MinValue, -5];
        long[] upper = [long.MaxValue, 5];
        var low = new BoundedCompositionSampler(lower, upper, long.MinValue + 3);
        Assert.Equal(Math.Log(9), low.LogCount, 14);
        var seen = new HashSet<long>();
        for (int k = 0; k < 2000; k++)
        {
            low.Sample(source, x);
            AssertSatisfies(lower, upper, long.MinValue + 3, x);
            seen.Add(x[0]);
        }

        Assert.Equal(9, seen.Count);

        // Σuᵢ が long を超える上限（補った組の側）
        lower = [-5, long.MinValue];
        upper = [5, long.MaxValue];
        var high = new BoundedCompositionSampler(lower, upper, long.MaxValue - 2);
        Assert.Equal(Math.Log(8), high.LogCount, 14);
        seen.Clear();
        for (int k = 0; k < 2000; k++)
        {
            high.Sample(source, x);
            AssertSatisfies(lower, upper, long.MaxValue - 2, x);
            seen.Add(x[0]);
        }

        Assert.Equal(8, seen.Count);
    }

    [Fact]
    public void InvalidArgumentsAreRejected()
    {
        Assert.Throws<ArgumentException>(() => new BoundedCompositionSampler([], [], 0));
        Assert.Throws<ArgumentException>(() => new BoundedCompositionSampler([0, 0], [1], 0));
        Assert.Throws<ArgumentException>(() => new BoundedCompositionSampler([0, 2], [1, 1], 1));
        Assert.Throws<ArgumentException>(() => new BoundedCompositionSampler([0, 1], [3, 4], 0));      // s < Σlᵢ
        Assert.Throws<ArgumentException>(() => new BoundedCompositionSampler([0, 1], [3, 4], 8));      // s > Σuᵢ
        Assert.Throws<ArgumentException>(() => new BoundedCompositionSampler([long.MaxValue, long.MaxValue], [long.MaxValue, long.MaxValue], long.MaxValue));

        // 簡約した和 m' が int に収まらない、あるいは表（約 n·m' 個）が配列に収まらない
        Assert.Throws<ArgumentOutOfRangeException>(() => new BoundedCompositionSampler([0, 0], [long.MaxValue, long.MaxValue], long.MaxValue / 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BoundedCompositionSampler(new long[1000], [.. Enumerable.Repeat(10_000_000L, 1000)], 5_000_000));

        var sampler = new BoundedCompositionSampler([0, 0], [3, 3], 3);
        Assert.Throws<ArgumentException>(() => sampler.Sample(new Pcg64DxsmRandom(1), new long[3]));
        Assert.Throws<ArgumentNullException>(() => sampler.Sample(null!, new long[2]));
    }
}
