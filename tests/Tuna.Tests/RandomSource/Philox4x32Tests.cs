using Tuna.RandomSource;

namespace Tuna.Tests.RandomSource;

public class Philox4x32Tests
{
    // Random123 v1.14.0 tests/kat_vectors の数値。出典・ライセンスは本体の THIRD_PARTY_NOTICES.txt。
    [Theory]
    [InlineData(0UL, 0UL, 0UL, 0x6627E8D5U, 0xE169C58DU, 0xBC57AC4CU, 0x9B00DBD8U)]
    [InlineData(ulong.MaxValue, ulong.MaxValue, ulong.MaxValue, 0x408F276DU, 0x41C83B0EU, 0xA20BC7C6U, 0x6D5451FDU)]
    [InlineData(0x299F31D0A4093822UL, 0x0370734413198A2EUL, 0x85A308D3243F6A88UL,
        0xD16CFE09U, 0x94FDCCEBU, 0x5001E420U, 0x24126EA1U)]
    public void MatchesPublishedKnownAnswers(ulong seed, ulong stream, ulong counter, uint a, uint b, uint c, uint d)
    {
        Assert.Equal(new Philox4x32Block(a, b, c, d), Philox4x32Random.GenerateBlock(seed, stream, counter));
        var source = new Philox4x32Random(new Philox4x32State(seed, stream, counter, 0));
        Assert.Equal(new[] { a, b, c, d }, Enumerable.Range(0, 4).Select(_ => source.NextUInt32()).ToArray());
    }

    [Fact]
    public void UInt64CombinesConsecutiveWordsLowFirstAcrossBlockBoundary()
    {
        var source = new Philox4x32Random(42, 123);
        var first = Philox4x32Random.GenerateBlock(42, 123, 0);
        var second = Philox4x32Random.GenerateBlock(42, 123, 1);
        source.Seek(0, 3);
        Assert.Equal(first.Word3 | ((ulong)second.Word0 << 32), source.NextUInt64());
        Assert.Equal(new Philox4x32State(42, 123, 1, 1), source.GetState());
    }

    [Fact]
    public void CounterCarriesWithoutChangingStream()
    {
        var source = new Philox4x32Random(7, 9);
        source.Seek(uint.MaxValue, 3);
        source.NextUInt32();
        Assert.Equal(Philox4x32Random.GenerateBlock(7, 9, 1UL << 32).Word0, source.NextUInt32());
        Assert.Equal(9UL, source.GetState().StreamId);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void RestoresEveryBufferedPosition(int consumed)
    {
        var source = new Philox4x32Random(7, 15);
        for (int i = 0; i < consumed; i++)
        {
            source.NextUInt32();
        }

        var saved = source.GetState();
        var copy = new Philox4x32Random(saved);
        ulong[] expected = new ulong[33];
        source.Fill(expected);
        Assert.Equal(expected, Enumerable.Range(0, expected.Length).Select(_ => copy.NextUInt64()).ToArray());
        source.RestoreState(saved);
        ulong[] actual = new ulong[33];
        source.Fill(actual);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void ExhaustionDoesNotWrapIntoAnotherStreamOrConsumeAnIncompleteUInt64()
    {
        var source = new Philox4x32Random(0);
        source.Seek(ulong.MaxValue, 3);
        var saved = source.GetState();
        Assert.Throws<InvalidOperationException>(() => source.NextUInt64());
        Assert.Equal(saved, source.GetState());
        Assert.Equal(Philox4x32Random.GenerateBlock(0, 0, ulong.MaxValue).Word3, source.NextUInt32());
        Assert.Throws<InvalidOperationException>(() => source.NextUInt32());
        var restored = new Philox4x32Random(source.GetState());
        Assert.Throws<InvalidOperationException>(() => restored.NextUInt64());
        restored.Seek(0);
        Assert.Equal(0x6627E8D5U, restored.NextUInt32());
    }

    /// <summary>ブロック単位でまとめて書く一括生成は、どの語の位置から始めても1個ずつ生成した値と状態に一致する</summary>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void BulkFillMatchesOneAtATimeFromEveryWordPosition(int wordIndex)
    {
        // ブロック番号の下位 32 ビットの桁上がりをまたぐ位置も含める
        foreach (ulong blockIndex in new[] { 1000UL, (1UL << 32) - 3 })
        {
            foreach (int length in new[] { 0, 1, 2, 3, 4, 5, 7, 8, 9, 33, 1000 })
            {
                var start = new Philox4x32State(11, 5, blockIndex, wordIndex);
                var bulk = new Philox4x32Random(start);
                var single = new Philox4x32Random(start);
                var words = new uint[length];
                bulk.Fill(words);
                Assert.Equal(Enumerable.Range(0, length).Select(_ => single.NextUInt32()).ToArray(), words);
                Assert.Equal(single.GetState(), bulk.GetState());

                var longs = new ulong[length];
                bulk.Fill(longs);
                Assert.Equal(Enumerable.Range(0, length).Select(_ => single.NextUInt64()).ToArray(), longs);
                Assert.Equal(single.GetState(), bulk.GetState());

                var doubles = new double[length];
                bulk.Fill(doubles);
                Assert.Equal(Enumerable.Range(0, length).Select(_ => single.NextDouble()).ToArray(), doubles);
                Assert.Equal(single.GetState(), bulk.GetState());
                Assert.Equal(single.NextUInt32(), bulk.NextUInt32());
            }
        }
    }

    /// <summary>
    /// ストリームの末尾では、一括生成も1個ずつと同じ位置で例外を出し、それまでの出力と状態の進行を残す。
    /// 生成できなかった位置から先の要素は書き換えない（語3から始まる対の持ち越しの語も残さない）
    /// </summary>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void BulkFillStopsAtExhaustionLikeOneAtATime(int wordIndex)
    {
        var start = new Philox4x32State(3, 8, ulong.MaxValue - 5, wordIndex);
        var bulk = new Philox4x32Random(start);
        var single = new Philox4x32Random(start);
        var doubles = Enumerable.Repeat(-1.0, 40).ToArray();
        Assert.Throws<InvalidOperationException>(() => bulk.Fill(doubles));
        var expected = TakeUntilExhausted(single.NextDouble);
        Assert.Equal(expected.Concat(Enumerable.Repeat(-1.0, 40 - expected.Count)), doubles);
        Assert.Equal(single.GetState(), bulk.GetState());

        bulk = new Philox4x32Random(start);
        single = new Philox4x32Random(start);
        var pairs = Enumerable.Repeat(0xDEADBEEFDEADBEEFUL, 40).ToArray();
        Assert.Throws<InvalidOperationException>(() => bulk.Fill(pairs));
        var expectedPairs = TakeUntilExhausted(single.NextUInt64);
        Assert.Equal(expectedPairs.Concat(Enumerable.Repeat(0xDEADBEEFDEADBEEFUL, 40 - expectedPairs.Count)), pairs);
        Assert.Equal(single.GetState(), bulk.GetState());

        bulk = new Philox4x32Random(start);
        single = new Philox4x32Random(start);
        var words = Enumerable.Repeat(0xDEADBEEFU, 40).ToArray();
        Assert.Throws<InvalidOperationException>(() => bulk.Fill(words));
        var expectedWords = TakeUntilExhausted(single.NextUInt32);
        Assert.Equal(expectedWords.Concat(Enumerable.Repeat(0xDEADBEEFU, 40 - expectedWords.Count)), words);
        Assert.Equal(single.GetState(), bulk.GetState());
    }

    private static List<T> TakeUntilExhausted<T>(Func<T> next)
    {
        var values = new List<T>();
        try
        {
            while (true)
            {
                values.Add(next());
            }
        }
        catch (InvalidOperationException)
        {
            return values;
        }
    }

    [Fact]
    public void InvalidStateAndSeekLeaveGeneratorUnchanged()
    {
        var source = new Philox4x32Random(123);
        var before = source.GetState();
        Assert.Throws<ArgumentException>(() => source.RestoreState(before with { Version = 2 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => source.RestoreState(before with { WordIndex = -1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => source.RestoreState(before with { WordIndex = 5 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => source.Seek(0, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => source.Seek(0, 4));
        Assert.Equal(before, source.GetState());
    }

    [Fact]
    public void StatelessAccessIsIndependentOfExecutionOrder()
    {
        var expected = Enumerable.Range(0, 1024).Select(i => Philox4x32Random.GenerateBlock(17, 3, (ulong)i)).ToArray();
        var actual = new Philox4x32Block[expected.Length];
        Parallel.For(0, actual.Length, i => actual[actual.Length - 1 - i] =
            Philox4x32Random.GenerateBlock(17, 3, (ulong)(actual.Length - 1 - i)));
        Assert.Equal(expected, actual);
    }
}
