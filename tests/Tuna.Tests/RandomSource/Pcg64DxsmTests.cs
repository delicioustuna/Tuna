using System.Text.Json;
using Tuna.RandomSource;

namespace Tuna.Tests.RandomSource;

public class Pcg64DxsmTests
{
    private static JsonDocument References() => JsonDocument.Parse(
        typeof(Pcg64DxsmTests).Assembly.GetManifestResourceStream("Tuna.Tests.RandomSource.pcg64dxsm-reference.json")!);

    [Fact]
    public void SeedExpansionAndRawOutputMatchIndependentNumpyReference()
    {
        using var document = References();
        foreach (var entry in document.RootElement.GetProperty("Seeded").EnumerateArray())
        {
            var source = new Pcg64DxsmRandom(entry.GetProperty("Seed").GetUInt64(), entry.GetProperty("Stream").GetUInt64());
            Assert.Equal(UInt128.Parse(entry.GetProperty("State").GetString()!), source.GetState().State);
            Assert.Equal(UInt128.Parse(entry.GetProperty("Increment").GetString()!), source.GetState().Increment);
            foreach (var value in entry.GetProperty("Values").EnumerateArray())
            {
                Assert.Equal(value.GetUInt64(), source.NextUInt64());
            }
        }
    }

    [Fact]
    public void RawStatesAndLargeAdvancesMatchIndependentNumpyReference()
    {
        using var document = References();
        foreach (var entry in document.RootElement.GetProperty("Direct").EnumerateArray())
        {
            var state = new Pcg64DxsmState(UInt128.Parse(entry.GetProperty("State").GetString()!),
                UInt128.Parse(entry.GetProperty("Increment").GetString()!));
            var source = new Pcg64DxsmRandom(state);
            source.Advance(UInt128.Parse(entry.GetProperty("Advance").GetString()!));
            Assert.Equal(UInt128.Parse(entry.GetProperty("AdvancedState").GetString()!), source.GetState().State);
            foreach (var value in entry.GetProperty("Values").EnumerateArray())
            {
                Assert.Equal(value.GetUInt64(), source.NextUInt64());
            }
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(257)]
    [InlineData(10000)]
    public void AdvanceAgreesWithSequentialConsumption(int count)
    {
        var sequential = new Pcg64DxsmRandom(314159, 27);
        var skipped = new Pcg64DxsmRandom(314159, 27);
        for (int i = 0; i < count; i++)
        {
            sequential.NextUInt64();
        }

        skipped.Advance((uint)count);
        Assert.Equal(sequential.GetState(), skipped.GetState());
        Assert.Equal(sequential.NextUInt64(), skipped.NextUInt64());
    }

    [Fact]
    public void FullPeriodAdvanceReturnsToStartingState()
    {
        var source = new Pcg64DxsmRandom(17);
        var saved = source.GetState();
        source.Advance(UInt128.MaxValue);
        source.NextUInt64();
        Assert.Equal(saved, source.GetState());
    }

    [Fact]
    public void RestoresMixedCallsWithoutHiddenHalfWordCache()
    {
        var source = new Pcg64DxsmRandom(21, 34);
        source.NextUInt32();
        source.NextDoubleOpen();
        var saved = source.GetState();
        var copy = new Pcg64DxsmRandom(saved);
        Assert.Equal(source.NextUInt32(), copy.NextUInt32());
        Assert.Equal(source.NextDouble(), copy.NextDouble());
        Assert.Equal(source.NextInt64(long.MinValue, long.MaxValue), copy.NextInt64(long.MinValue, long.MaxValue));
        source.RestoreState(saved);
        Assert.Equal(new Pcg64DxsmRandom(saved).NextUInt64(), source.NextUInt64());
    }

    [Fact]
    public void RejectsInvalidStateBeforeMutation()
    {
        var source = new Pcg64DxsmRandom(21);
        var saved = source.GetState();
        Assert.Throws<ArgumentException>(() => source.RestoreState(saved with { Increment = 0 }));
        Assert.Throws<ArgumentException>(() => source.RestoreState(saved with { Increment = 2 }));
        Assert.Throws<ArgumentException>(() => source.RestoreState(saved with { Version = 0 }));
        Assert.Equal(saved, source.GetState());
    }
}
