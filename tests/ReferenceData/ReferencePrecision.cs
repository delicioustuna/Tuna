using Xunit;

namespace Tuna.Tests.ReferenceData;

internal static class ReferencePrecision
{
    internal const double Epsilon = 2.220446049250313e-16;
    internal const double SixteenEpsilon = 16 * Epsilon;

    internal static void EqualBits(double expected, double actual, string? context = null)
    {
        long expectedBits = BitConverter.DoubleToInt64Bits(expected);
        long actualBits = BitConverter.DoubleToInt64Bits(actual);
        Assert.True(expectedBits == actualBits,
            $"{context}: expected {expected:R} (0x{expectedBits:X16}), actual {actual:R} (0x{actualBits:X16})");
    }

    internal static ulong UlpDistance(double first, double second)
    {
        // 正負のゼロを同じ点へ写す。
        ulong firstOrdered = OrderedBits(first);
        ulong secondOrdered = OrderedBits(second);
        return firstOrdered >= secondOrdered ? firstOrdered - secondOrdered : secondOrdered - firstOrdered;
    }

    private static ulong OrderedBits(double value)
    {
        ulong bits = BitConverter.DoubleToUInt64Bits(value);
        const ulong sign = 1UL << 63;
        return (bits & sign) == 0 ? sign + bits : sign - (bits & ~sign);
    }
}
