using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Tuna.LinearAlgebra;
using static Tuna.XMath;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>
/// 中央値が、並べ替えて中央をとる定義とビット単位で一致すること。並べ替えは全順序（-0 &lt; +0）で、要素数が偶数なら中央の2つの中点
/// <c>(a + b)/2</c>（溢れるときは <c>a/2 + b/2</c>）。NaN は、ignoreNaN なら除き、そうでなければ最初の NaN を返す。
/// double・float・Half は整数の鍵で、NFloat は汎用の経路で選ぶ。長さ 601 を超えると Floyd–Rivest の標本からの絞り込みを通る。
/// </summary>
[Trait("LinearAlgebra", "XMath")]
public class MedianTests
{
    private static readonly int[] Lengths = [.. Enumerable.Range(0, 41), 63, 64, 65, 599, 600, 601, 602, 603, 604, 1000, 1001, 4097, 10000];

    private static readonly int[] LongLengths = [65536, 100001];

    private static int Compare<T>(T x, T y) where T : IFloatingPointIeee754<T> =>
        x < y ? -1 : x > y ? 1 : T.IsNegative(x) == T.IsNegative(y) ? 0 : T.IsNegative(x) ? -1 : 1;

    private static T ReferenceMedian<T>(T[] x, bool ignoreNaN) where T : IFloatingPointIeee754<T>
    {
        if (!ignoreNaN)
        {
            foreach (T value in x)
            {
                if (T.IsNaN(value))
                {
                    return value;
                }
            }
        }

        var values = x.Where(v => !T.IsNaN(v)).ToArray();
        if (values.Length == 0)
        {
            return T.NaN;
        }

        Array.Sort(values, Compare);
        int k = values.Length / 2;
        if (values.Length % 2 == 1)
        {
            return values[k];
        }

        T a = values[k - 1], b = values[k], two = T.One + T.One;
        T midpoint = (a + b) / two;
        return T.IsInfinity(midpoint) && T.IsFinite(a) && T.IsFinite(b) ? a / two + b / two : midpoint;
    }

    private static long Bits<T>(T value) where T : unmanaged => Unsafe.SizeOf<T>() switch
    {
        8 => Unsafe.BitCast<T, long>(value),
        4 => Unsafe.BitCast<T, int>(value),
        _ => Unsafe.BitCast<T, short>(value),
    };

    private static void AssertBits<T>(string label, T expected, T actual) where T : unmanaged, IFloatingPointIeee754<T>
    {
        bool same = T.IsNaN(expected) ? T.IsNaN(actual) : Bits(expected) == Bits(actual);
        Assert.True(same, $"{label}: expected {expected} ({Bits(expected):X}), actual {actual} ({Bits(actual):X})");
    }

    private static T From<T>(double value) where T : IFloatingPointIeee754<T> => T.CreateSaturating(value);

    /// <summary>並び（乱数・昇順・降順・定数・少ない種類と ±0・山形・のこぎり）と特別な値（NaN・±∞・最大値付近・非正規化数）</summary>
    private static IEnumerable<(string Name, T[] Values)> Patterns<T>(int n, int seed) where T : IFloatingPointIeee754<T>, IMinMaxValue<T>
    {
        var random = new Random(seed);
        T[] Make(Func<int, T> value) => Enumerable.Range(0, n).Select(value).ToArray();
        yield return ("random", Make(_ => From<T>((2 * random.NextDouble() - 1) * Math.ScaleB(1, random.Next(-8, 9)))));
        yield return ("ascending", Make(i => From<T>(i - n / 3)));
        yield return ("descending", Make(i => From<T>(n - i)));
        yield return ("constant", Make(_ => From<T>(3.5)));
        T[] pool = [-T.One, T.NegativeZero, T.Zero, T.One, From<T>(0.5)];
        yield return ("ties", Make(_ => pool[random.Next(pool.Length)]));
        yield return ("organ pipe", Make(i => From<T>(Math.Min(i, n - 1 - i))));
        yield return ("sawtooth", Make(i => From<T>(i % 7)));
        yield return ("huge", Make(_ => T.MaxValue * From<T>(random.NextDouble() < 0.5 ? 1 : 0.75 + random.NextDouble() / 4)));
        yield return ("subnormal", Make(_ => T.Epsilon * From<T>(random.Next(-5, 6))));
        yield return ("infinities", Make(i => i % 3 == 0 ? T.NegativeInfinity : i % 3 == 1 ? T.PositiveInfinity : From<T>(random.Next(-3, 4))));
        if (n > 0)
        {
            var oneNaN = Make(_ => From<T>(random.NextDouble()));
            oneNaN[random.Next(n)] = T.NaN;
            yield return ("one NaN", oneNaN);
            yield return ("many NaN", Make(_ => random.NextDouble() < 0.3 ? T.NaN : From<T>(random.Next(-50, 50))));
            yield return ("all NaN", Make(_ => T.NaN));
        }
    }

    private static void CheckMedians<T>(int maximumLength) where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
    {
        foreach (int n in Lengths.Concat(LongLengths).Where(n => n <= maximumLength))
        {
            foreach (var (name, x) in Patterns<T>(n, n))
            {
                if (n > 10000 && name is not ("random" or "ties" or "organ pipe"))
                {
                    continue;
                }

                var original = x.ToArray();
                var strided = new Matrix<T>(3, n);
                for (int i = 0; i < n; i++)
                {
                    strided[1, i] = x[i];
                }

                foreach (bool ignoreNaN in new[] { false, true })
                {
                    string label = $"{typeof(T).Name} n={n} {name} ignoreNaN={ignoreNaN}";
                    T expected = ReferenceMedian(x, ignoreNaN);
                    AssertBits($"Median {label}", expected, Median(x.ToVector(), ignoreNaN));
                    AssertBits($"Median strided {label}", expected, Median(strided.Row(1), ignoreNaN));
                    if (!ignoreNaN)
                    {
                        AssertBits($"Median default {label}", expected, Median(x.ToVector()));
                    }
                }

                Assert.Equal(original.Select(Bits), x.Select(Bits));
            }
        }
    }

    [Fact]
    public void MedianMatchesTheSortedDefinitionForDouble() => CheckMedians<double>(int.MaxValue);

    [Fact]
    public void MedianMatchesTheSortedDefinitionForSingle() => CheckMedians<float>(int.MaxValue);

    // Half は整数の鍵（short）、NFloat は値そのものを全順序で比べる汎用の経路
    [Fact]
    public void MedianMatchesTheSortedDefinitionForHalfAndGenericTypes()
    {
        CheckMedians<Half>(4097);
        CheckMedians<NFloat>(4097);
    }

    [Fact]
    public void ColumnAndRowMediansMatchTheVectorMedian()
    {
        foreach (var (rows, columns) in new[] { (0, 3), (3, 0), (1, 1), (5, 4), (64, 7), (7, 64), (601, 3), (1001, 2) })
        {
            var a = new Matrix<double>(rows, columns);
            int seed = 1;
            for (int j = 0; j < columns; j++)
            {
                var patterns = Patterns<double>(rows, seed++).ToArray();
                var values = patterns[j % patterns.Length].Values;
                for (int i = 0; i < rows; i++)
                {
                    a[i, j] = values[i];
                }
            }

            foreach (bool ignoreNaN in new[] { false, true })
            {
                string label = $"{rows}×{columns} ignoreNaN={ignoreNaN}";
                var columnMedians = ColumnMedians(a, ignoreNaN);
                var rowMedians = RowMedians(a.T, ignoreNaN);
                for (int j = 0; j < columns; j++)
                {
                    double[] column = Enumerable.Range(0, rows).Select(i => a[i, j]).ToArray();
                    AssertBits($"ColumnMedians {label} [{j}]", ReferenceMedian(column, ignoreNaN), columnMedians[j]);
                    AssertBits($"RowMedians {label} [{j}]", ReferenceMedian(column, ignoreNaN), rowMedians[j]);
                }

                var destination = new Tuna.LinearAlgebra.Vector<double>(columns);
                Assert.Same(destination, ColumnMedians(a, ignoreNaN, destination));
                Assert.Equal(columnMedians.ToArray().Select(Bits), destination.ToArray().Select(Bits));
                var rowDestination = new Tuna.LinearAlgebra.Vector<double>(columns);
                Assert.Same(rowDestination, RowMedians(a.T, ignoreNaN, rowDestination));
                Assert.Equal(columnMedians.ToArray().Select(Bits), rowDestination.ToArray().Select(Bits));
            }

            Assert.Equal(ColumnMedians(a, ignoreNaN: false).ToArray().Select(Bits), ColumnMedians(a).ToArray().Select(Bits));
            Assert.Equal(RowMedians(a, ignoreNaN: false).ToArray().Select(Bits), RowMedians(a).ToArray().Select(Bits));
            var plain = new Tuna.LinearAlgebra.Vector<double>(rows);
            Assert.Same(plain, RowMedians(a, plain));
        }

        Assert.Throws<ArgumentException>(() => { ColumnMedians(new Matrix<double>(2, 3), new Tuna.LinearAlgebra.Vector<double>(2)); });
        Assert.Throws<ArgumentException>(() => { RowMedians(new Matrix<double>(2, 3), true, new Tuna.LinearAlgebra.Vector<double>(3)); });
    }

    [Fact]
    public void MedianPropagatesTheFirstNaNAndHandlesSmallCases()
    {
        double first = BitConverter.Int64BitsToDouble(0x7FF8_0000_0000_0042), second = BitConverter.Int64BitsToDouble(0x7FF8_0000_0000_0099);
        Assert.Equal(0x7FF8_0000_0000_0042, BitConverter.DoubleToInt64Bits(Median(new double[] { 1, first, second, 2 }.ToVector())));
        Assert.Equal(1.5, Median(new double[] { 1, first, second, 2 }.ToVector(), ignoreNaN: true));
        Assert.True(double.IsNaN(Median(new Tuna.LinearAlgebra.Vector<double>(0))));
        Assert.True(double.IsNaN(Median(new[] { double.NaN, double.NaN }.ToVector(), ignoreNaN: true)));
        Assert.Equal(2.0, Median(new double[] { 3, 1, 2 }.ToVector()));
        Assert.Equal(2.5, Median(new double[] { 4, 1, 3, 2 }.ToVector()));
        Assert.Equal(double.MaxValue, Median(new[] { double.MaxValue, double.MaxValue }.ToVector()));
        Assert.Equal(0.875 * double.MaxValue, Median(new[] { double.MaxValue, 0.75 * double.MaxValue }.ToVector()));
        Assert.True(double.IsNaN(Median(new[] { double.NegativeInfinity, double.PositiveInfinity }.ToVector())));
        Assert.Equal(double.NegativeInfinity, Median(new[] { double.NegativeInfinity, double.NegativeInfinity, 0 }.ToVector()));
        Assert.True(double.IsNegative(Median(new[] { 0.0, -0.0, -0.0 }.ToVector())));
        Assert.Equal(0x0, BitConverter.DoubleToInt64Bits(Median(new[] { 0.0, -0.0 }.ToVector())));
    }
}
