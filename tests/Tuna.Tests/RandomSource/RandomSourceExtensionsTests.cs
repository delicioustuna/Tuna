using Tuna.RandomSource;

namespace Tuna.Tests.RandomSource;

/// <summary>Fisher–Yates のシャッフルと順列: 値と消費が <c>NextInt32(i + 1)</c>（i = n - 1, …, 1）の繰り返しと同じで、並びが一様であること</summary>
public class RandomSourceExtensionsTests
{
    private static RandomGenerator Create(bool philox, ulong seed = 123, ulong stream = 0) =>
        philox ? new Philox4x32Random(seed, stream) : new Pcg64DxsmRandom(seed, stream);

    /// <summary>定義どおりの参照: i = n - 1, …, 1 の順に j = NextInt32(i + 1) を引いて入れ替える</summary>
    private static void ReferenceShuffle<T>(IRandomSource random, T[] values)
    {
        for (int i = values.Length - 1; i > 0; i--)
        {
            int j = random.NextInt32(i + 1);
            (values[i], values[j]) = (values[j], values[i]);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ShuffleMatchesTheDefinitionAndConsumesOneDrawPerPosition(bool philox)
    {
        var source = Create(philox);
        var reference = Create(philox);
        foreach (int n in Enumerable.Range(0, 40).Concat([100, 1000, 4097]))
        {
            int[] expected = Enumerable.Range(0, n).Select(i => 3 * i - 7).ToArray();
            int[] actual = expected.ToArray();
            ReferenceShuffle(reference, expected);
            source.Shuffle(actual.AsSpan());
            Assert.Equal(expected, actual);

            int[] permutation = new int[n];
            int[] identity = Enumerable.Range(0, n).ToArray();
            ReferenceShuffle(reference, identity);
            source.Permutation(permutation);
            Assert.Equal(identity, permutation);
            Assert.Equal(reference.NextUInt64(), source.NextUInt64());
        }

        // 参照型の要素
        string[] words = ["a", "b", "c", "d", "e"];
        string[] expectedWords = words.ToArray();
        ReferenceShuffle(reference, expectedWords);
        source.Shuffle(words.AsSpan());
        Assert.Equal(expectedWords, words);
    }

    [Fact]
    public void ShuffleThroughTheInterfaceUsesOnlyNextInt32()
    {
        // RandomGenerator 以外の乱数源は、NextInt32(i + 1) を n - 1 回呼ぶだけ
        var counting = new CountingSource(new Pcg64DxsmRandom(7));
        var direct = new Pcg64DxsmRandom(7);
        int[] viaInterface = Enumerable.Range(0, 257).ToArray();
        int[] viaGenerator = viaInterface.ToArray();
        counting.Shuffle(viaInterface.AsSpan());
        direct.Shuffle(viaGenerator.AsSpan());
        Assert.Equal(viaGenerator, viaInterface);
        Assert.Equal(256, counting.BoundedInt32Calls);
        Assert.Equal(256, counting.TotalCalls);

        int[] permutation = new int[10];
        counting.Permutation(permutation);
        Assert.Equal(256 + 9, counting.TotalCalls);
        Assert.Equal(Enumerable.Range(0, 10), permutation.Order());

        counting.Shuffle(Span<int>.Empty);
        counting.Shuffle(new int[1].AsSpan());
        Assert.Equal(256 + 9, counting.TotalCalls);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PermutationsAreUniform(bool philox)
    {
        // 長さ 4 の 24 通りと長さ 6 の 720 通りの出現回数をカイ二乗で確かめる（有意水準 1e-6）
        var source = Create(philox, 2026);
        foreach (int n in new[] { 4, 6 })
        {
            int outcomes = Enumerable.Range(1, n).Aggregate(1, (a, b) => a * b);
            int draws = outcomes * 200;
            var counts = new Dictionary<string, int>();
            int[] permutation = new int[n];
            for (int k = 0; k < draws; k++)
            {
                source.Permutation(permutation);
                Assert.Equal(Enumerable.Range(0, n), permutation.Order());
                string key = string.Join(",", permutation);
                counts[key] = counts.GetValueOrDefault(key) + 1;
            }

            Assert.Equal(outcomes, counts.Count);
            double expected = (double)draws / outcomes;
            double chiSquare = counts.Values.Sum(c => (c - expected) * (c - expected) / expected);
            Assert.True(chiSquare < ChiSquareCritical(outcomes - 1), $"n={n}: χ² = {chiSquare}");
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ShuffleDoesNotAllocateAndRejectsNull(bool philox)
    {
        var source = Create(philox);
        double[] values = new double[1024];
        int[] permutation = new int[1024];
        for (int i = 0; i < 10; i++)
        {
            source.Shuffle(values.AsSpan());
            source.Permutation(permutation);
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10; i++)
        {
            source.Shuffle(values.AsSpan());
            source.Permutation(permutation);
        }

        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);

        IRandomSource missing = null!;
        Assert.Throws<ArgumentNullException>(() => missing.Shuffle(new int[3].AsSpan()));
        Assert.Throws<ArgumentNullException>(() => missing.Permutation(new int[3]));
    }

    /// <summary>自由度 k のカイ二乗分布の上側 1e-6 点（Wilson–Hilferty の近似。k ≥ 3 で十分正確）</summary>
    internal static double ChiSquareCritical(int k)
    {
        const double z = 4.753424;
        double a = 2.0 / (9 * k);
        return k * Math.Pow(1 - a + z * Math.Sqrt(a), 3);
    }

    /// <summary>呼び出しの回数を数えて別の乱数源に任せる（RandomGenerator でない乱数源）</summary>
    private sealed class CountingSource(IRandomSource inner) : IRandomSource
    {
        public int TotalCalls { get; private set; }
        public int BoundedInt32Calls { get; private set; }

        private T Count<T>(T value)
        {
            TotalCalls++;
            return value;
        }

        public uint NextUInt32() => Count(inner.NextUInt32());
        public ulong NextUInt64() => Count(inner.NextUInt64());
        public uint NextUInt32(uint exclusiveMax) => Count(inner.NextUInt32(exclusiveMax));
        public ulong NextUInt64(ulong exclusiveMax) => Count(inner.NextUInt64(exclusiveMax));
        public int NextInt32() => Count(inner.NextInt32());
        public int NextInt32(int exclusiveMax)
        {
            BoundedInt32Calls++;
            return Count(inner.NextInt32(exclusiveMax));
        }
        public int NextInt32(int inclusiveMin, int exclusiveMax) => Count(inner.NextInt32(inclusiveMin, exclusiveMax));
        public long NextInt64() => Count(inner.NextInt64());
        public long NextInt64(long exclusiveMax) => Count(inner.NextInt64(exclusiveMax));
        public long NextInt64(long inclusiveMin, long exclusiveMax) => Count(inner.NextInt64(inclusiveMin, exclusiveMax));
        public double NextDouble() => Count(inner.NextDouble());
        public double NextDoubleOpen() => Count(inner.NextDoubleOpen());
        public float NextSingle() => Count(inner.NextSingle());
        public void Fill(Span<uint> destination)
        {
            TotalCalls++;
            inner.Fill(destination);
        }
        public void Fill(Span<ulong> destination)
        {
            TotalCalls++;
            inner.Fill(destination);
        }
        public void Fill(Span<double> destination)
        {
            TotalCalls++;
            inner.Fill(destination);
        }
        public void Fill(Span<float> destination)
        {
            TotalCalls++;
            inner.Fill(destination);
        }
        public void Fill(Span<byte> destination)
        {
            TotalCalls++;
            inner.Fill(destination);
        }
    }
}
