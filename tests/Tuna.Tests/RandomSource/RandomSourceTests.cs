using System.Buffers.Binary;
using System.Numerics;
using Tuna.RandomSource;

namespace Tuna.Tests.RandomSource;

public class RandomSourceTests
{
    private static RandomGenerator Create(bool philox, ulong seed = 123, ulong stream = 0) =>
        philox ? new Philox4x32Random(seed, stream) : new Pcg64DxsmRandom(seed, stream);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BulkGenerationMatchesScalarCallsAndPartitioning(bool philox)
    {
        var scalar = Create(philox);
        var bulk = Create(philox);
        // 奇数の位置からでも語の消費規約を維持する。
        Assert.Equal(scalar.NextUInt32(), bulk.NextUInt32());
        ulong[] words = new ulong[1031];
        bulk.Fill(words.AsSpan(0, 17));
        bulk.Fill(words.AsSpan(17));
        foreach (ulong word in words)
        {
            Assert.Equal(scalar.NextUInt64(), word);
        }

        uint[] small = new uint[101];
        bulk.Fill(small);
        foreach (uint word in small)
        {
            Assert.Equal(scalar.NextUInt32(), word);
        }

        double[] doubles = new double[101];
        bulk.Fill(doubles);
        foreach (double value in doubles)
        {
            Assert.Equal(scalar.NextDouble(), value);
        }

        float[] singles = new float[101];
        bulk.Fill(singles);
        foreach (float value in singles)
        {
            Assert.Equal(scalar.NextSingle(), value);
        }

        Assert.Equal(scalar.NextUInt64(), bulk.NextUInt64());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BytesAreLittleEndianAndTailConsumesOneWord(bool philox)
    {
        for (int length = 0; length <= 33; length++)
        {
            var scalar = Create(philox);
            var bulk = Create(philox);
            byte[] expected = new byte[((length + 7) / 8) * 8];
            for (int i = 0; i < expected.Length; i += 8)
            {
                BinaryPrimitives.WriteUInt64LittleEndian(expected.AsSpan(i), scalar.NextUInt64());
            }

            byte[] actual = new byte[length];
            bulk.Fill(actual);
            Assert.Equal(expected[..length], actual);
            Assert.Equal(scalar.NextUInt64(), bulk.NextUInt64());
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BoundsCoverExtremeSignedRangesAndFloatingIntervals(bool philox)
    {
        var source = Create(philox);
        for (int i = 0; i < 10000; i++)
        {
            Assert.InRange(source.NextInt32(), 0, int.MaxValue - 1);
            Assert.InRange(source.NextInt64(), 0, long.MaxValue - 1);
            Assert.InRange(source.NextInt32(int.MinValue, int.MaxValue), int.MinValue, int.MaxValue - 1);
            Assert.InRange(source.NextInt64(long.MinValue, long.MaxValue), long.MinValue, long.MaxValue - 1);
            Assert.InRange(source.NextInt32(-20, -3), -20, -4);
            Assert.InRange(source.NextInt64(-20, -3), -20, -4);
            Assert.InRange(source.NextUInt32(uint.MaxValue), 0U, uint.MaxValue - 1);
            Assert.InRange(source.NextUInt64(ulong.MaxValue), 0UL, ulong.MaxValue - 1);
            Assert.InRange(source.NextDouble(), 0, Math.BitDecrement(1.0));
            Assert.InRange(source.NextDoubleOpen(), 1.0 / (1UL << 53), Math.BitDecrement(1.0));
            Assert.InRange(source.NextSingle(), 0, MathF.BitDecrement(1.0f));
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EmptyOperationsAndInvalidArgumentsDoNotConsumeRandomness(bool philox)
    {
        var source = Create(philox);
        source.Fill(Span<byte>.Empty);
        source.Fill(Span<uint>.Empty);
        source.Fill(Span<ulong>.Empty);
        source.Fill(Span<double>.Empty);
        source.Fill(Span<float>.Empty);
        Assert.Equal(0, source.NextInt32(0));
        Assert.Equal(0, source.NextInt32(1));
        Assert.Equal(7, source.NextInt32(7, 7));
        Assert.Equal(int.MinValue, source.NextInt32(int.MinValue, int.MinValue + 1));
        Assert.Equal(0L, source.NextInt64(0));
        Assert.Equal(0L, source.NextInt64(1));
        Assert.Equal(long.MinValue, source.NextInt64(long.MinValue, long.MinValue + 1));
        Assert.Equal(0U, source.NextUInt32(1));
        Assert.Equal(0UL, source.NextUInt64(1));
        Assert.Throws<ArgumentOutOfRangeException>(() => source.NextInt32(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => source.NextInt32(1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => source.NextInt64(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => source.NextInt64(1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => source.NextUInt32(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => source.NextUInt64(0));
        Assert.Equal(Create(philox).NextUInt64(), source.NextUInt64());
    }

    // DXSMは下位語を1に固定すると上位語から出力への全単射。
    // テスト入力はBigIntegerによる逆写像で構成し、変換の端点を確実に通す。
    private static Pcg64DxsmRandom WithFirstOutput(ulong output)
    {
        BigInteger modulus = BigInteger.One << 64;
        BigInteger inverse = BigInteger.ModPow(new BigInteger(0xDA942042E4DD58B5UL), (BigInteger.One << 63) - 1, modulus);
        ulong product = output ^ (output >> 48);
        ulong folded = (ulong)((product * inverse) % modulus);
        ulong upper = folded ^ (folded >> 32);
        var source = new Pcg64DxsmRandom(new Pcg64DxsmState((UInt128)upper << 64 | 1, 1));
        var saved = source.GetState();
        Assert.Equal(output, source.NextUInt64());
        source.RestoreState(saved);
        return source;
    }

    [Fact]
    public void FloatingEndpointsAreExactAndOpenIntervalRejectsZero()
    {
        Assert.Equal(0, WithFirstOutput(0).NextDouble());
        Assert.Equal(Math.BitDecrement(1.0), WithFirstOutput(ulong.MaxValue).NextDouble());
        Assert.Equal(Math.BitDecrement(1.0), WithFirstOutput(ulong.MaxValue).NextDoubleOpen());
        Assert.Equal(0, WithFirstOutput(0).NextSingle());
        Assert.Equal(MathF.BitDecrement(1.0f), WithFirstOutput(ulong.MaxValue).NextSingle());
        var source = WithFirstOutput(0);
        var reference = new Pcg64DxsmRandom(source.GetState());
        double expected;
        do
        {
            expected = reference.NextDouble();
        } while (expected == 0);
        Assert.Equal(expected, source.NextDoubleOpen());
        Assert.Equal(reference.GetState(), source.GetState());
    }

    [Fact]
    public void BoundedConversionRejectsTheIncompleteRemainderRegion()
    {
        var source = WithFirstOutput(0);
        var reference = new Pcg64DxsmRandom(source.GetState());
        reference.NextUInt64(); // bound=3では、0の入力は棄却が必要。
        ulong value;
        do
        {
            value = reference.NextUInt64();
        } while (value == 0);
        ulong expected = (ulong)((new BigInteger(value) * 3) >> 64);
        Assert.Equal(expected, source.NextUInt64(3));
        Assert.Equal(reference.GetState(), source.GetState());
        var source32 = WithFirstOutput(0);
        var reference32 = new Pcg64DxsmRandom(source32.GetState());
        uint word;
        do
        {
            word = reference32.NextUInt32();
        } while (word == 0);
        Assert.Equal((uint)(((ulong)word * 3) >> 32), source32.NextUInt32(3));
        Assert.Equal(reference32.GetState(), source32.GetState());
        Assert.Equal(2UL, WithFirstOutput(ulong.MaxValue).NextUInt64(3));
        Assert.Equal(2U, WithFirstOutput(ulong.MaxValue).NextUInt32(3));
        Assert.Equal(long.MaxValue - 1, WithFirstOutput(ulong.MaxValue).NextInt64(long.MinValue, long.MaxValue));
        Assert.Equal(int.MaxValue - 1, WithFirstOutput(ulong.MaxValue).NextInt32(int.MinValue, int.MaxValue));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SharedInstanceProducesExactlyTheSerialMultiset(bool philox)
    {
        const int workers = 8, perWorker = 4096;
        var source = Create(philox);
        ulong[] actual = new ulong[workers * perWorker];
        Parallel.For(0, workers, worker =>
        {
            int offset = worker * perWorker;
            for (int i = 0; i < perWorker / 2; i++)
            {
                actual[offset + i] = source.NextUInt64();
            }

            source.Fill(actual.AsSpan(offset + perWorker / 2, perWorker / 2));
        });
        ulong[] expected = new ulong[actual.Length];
        var serial = Create(philox);
        serial.Fill(expected);
        Assert.Equal(serial.NextUInt64(), source.NextUInt64());
        Array.Sort(expected);
        Array.Sort(actual);
        Assert.Equal(expected, actual);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LogicalStreamIdsReproduceAcrossParallelismLevels(bool philox)
    {
        const int streams = 37;
        ulong[][] expected = Enumerable.Range(0, streams).Select(i =>
        {
            ulong[] values = new ulong[73];
            Create(philox, 31, (ulong)i).Fill(values);
            return values;
        }).ToArray();
        foreach (int degree in new[] { 1, 2, 8 })
        {
            ulong[][] actual = new ulong[streams][];
            Parallel.For(0, streams, new ParallelOptions { MaxDegreeOfParallelism = degree }, i =>
            {
                int id = streams - 1 - i;
                actual[id] = new ulong[73];
                Create(philox, 31, (ulong)id).Fill(actual[id]);
            });
            for (int i = 0; i < streams; i++)
            {
                Assert.Equal(expected[i], actual[i]);
            }
        }

        Assert.Equal(streams, expected.Select(values => values[0]).Distinct().Count());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GenerationDoesNotAllocateAfterWarmup(bool philox)
    {
        var source = Create(philox);
        double[] values = new double[1024];
        for (int i = 0; i < 100; i++)
        {
            source.Fill(values);
            source.NextUInt64();
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 100; i++)
        {
            source.Fill(values);
            source.NextUInt64();
        }

        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }
}
