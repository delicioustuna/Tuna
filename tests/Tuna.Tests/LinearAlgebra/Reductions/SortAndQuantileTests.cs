using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Tuna.LinearAlgebra;
using static Tuna.XMath;

namespace Tuna.Tests.LinearAlgebra;

/// <summary>
/// 並べ替えと分位点が、全順序（-0 &lt; +0、NaN は最後に元の順）で並べた定義とビット単位で一致すること。ArgSort は安定。
/// 分位点は Hyndman–Fan の定義 7（h = (n - 1)p、j = ⌊h⌋、g = h - j、Q = xⱼ + g(xⱼ₊₁ - xⱼ)）で、g = 1/2 は中央値と同じ中点。
/// double・float・Half は整数の鍵、NFloat と整数型は値を比べる汎用の経路を通る。
/// </summary>
[Trait("LinearAlgebra", "XMath")]
public class SortAndQuantileTests
{
    private static readonly int[] Lengths = [.. Enumerable.Range(0, 20), 31, 32, 33, 64, 65, 600, 601, 602, 1000, 4097];

    private static readonly double[] Probabilities = [0, 1e-9, 0.1, 0.25, 1.0 / 3, 0.5, 0.6, 0.75, 0.9, 0.99, 1 - 1e-12, 1];

    private static int Compare<T>(T x, T y) where T : INumber<T> =>
        x < y ? -1 : x > y ? 1 : T.IsNegative(x) == T.IsNegative(y) ? 0 : T.IsNegative(x) ? -1 : 1;

    private static long Bits<T>(T value) where T : unmanaged => Unsafe.SizeOf<T>() switch
    {
        8 => Unsafe.BitCast<T, long>(value),
        4 => Unsafe.BitCast<T, int>(value),
        _ => Unsafe.BitCast<T, short>(value),
    };

    private static void AssertBits<T>(string label, T expected, T actual) where T : unmanaged, INumber<T>
    {
        bool same = T.IsNaN(expected) ? T.IsNaN(actual) : Bits(expected) == Bits(actual);
        Assert.True(same, $"{label}: expected {expected} ({Bits(expected):X}), actual {actual} ({Bits(actual):X})");
    }

    private static T From<T>(double value) where T : INumber<T> => T.CreateSaturating(value);

    private static T[] ReferenceSort<T>(T[] x) where T : INumber<T>
    {
        var values = x.Where(v => !T.IsNaN(v)).ToArray();
        Array.Sort(values, Compare);
        return [.. values, .. x.Where(T.IsNaN)];
    }

    // OrderBy は安定なので、等しい値は元の位置の順になる
    private static int[] ReferenceArgSort<T>(T[] x) where T : INumber<T> =>
    [
        .. Enumerable.Range(0, x.Length).Where(i => !T.IsNaN(x[i])).OrderBy(i => x[i], Comparer<T>.Create(Compare)),
        .. Enumerable.Range(0, x.Length).Where(i => T.IsNaN(x[i])),
    ];

    /// <summary>定義 7 を並べ替えた値から直接求める参照（補間の式は実装の注釈のとおりに書き下す）</summary>
    private static T ReferenceQuantile<T>(T[] x, double p, bool ignoreNaN) where T : unmanaged, IFloatingPointIeee754<T>
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

        var sorted = x.Where(v => !T.IsNaN(v)).ToArray();
        if (sorted.Length == 0)
        {
            return T.NaN;
        }

        Array.Sort(sorted, Compare);
        double h = (sorted.Length - 1) * p;
        int j = Math.Min((int)Math.Floor(h), sorted.Length - 1);
        double g = h - j;
        T a = sorted[j];
        if (g == 0)
        {
            return a;
        }

        T b = sorted[j + 1], two = T.One + T.One;
        if (g == 0.5)
        {
            T midpoint = (a + b) / two;
            return T.IsInfinity(midpoint) && T.IsFinite(a) && T.IsFinite(b) ? a / two + b / two : midpoint;
        }

        if (typeof(T) == typeof(float) || typeof(T) == typeof(Half))
        {
            return T.CreateTruncating(Lerp(double.CreateTruncating(a), double.CreateTruncating(b), g));
        }

        return Lerp(a, b, T.CreateTruncating(g));

        static TValue Lerp<TValue>(TValue a, TValue b, TValue g) where TValue : IFloatingPointIeee754<TValue>
        {
            if (a == b)
            {
                return a;
            }

            if (TValue.IsNegativeInfinity(a))
            {
                return TValue.IsPositiveInfinity(b) ? TValue.NaN : a;
            }

            if (TValue.IsPositiveInfinity(b))
            {
                return b;
            }

            TValue d = b - a;
            if (TValue.IsInfinity(d))
            {
                return (TValue.One - g) * a + g * b;
            }

            return g < TValue.CreateTruncating(0.5) ? a + g * d : b - (TValue.One - g) * d;
        }
    }

    /// <summary>並び（乱数・昇順・降順・定数・少ない種類と ±0・のこぎり）と特別な値（NaN・±∞・最大値付近・非正規化数）</summary>
    private static IEnumerable<(string Name, T[] Values)> Patterns<T>(int n, int seed) where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
    {
        var random = new Random(seed);
        T[] Make(Func<int, T> value) => Enumerable.Range(0, n).Select(value).ToArray();
        yield return ("random", Make(_ => From<T>((2 * random.NextDouble() - 1) * Math.ScaleB(1, random.Next(-8, 9)))));
        yield return ("ascending", Make(i => From<T>(i - n / 3)));
        yield return ("descending", Make(i => From<T>(n - i)));
        yield return ("constant", Make(_ => From<T>(3.5)));
        T[] pool = [-T.One, T.NegativeZero, T.Zero, T.One, From<T>(0.5)];
        yield return ("ties", Make(_ => pool[random.Next(pool.Length)]));
        yield return ("sawtooth", Make(i => From<T>(i % 7)));
        yield return ("huge", Make(_ => T.MaxValue * From<T>(random.NextDouble() < 0.5 ? 1 : -0.75 - random.NextDouble() / 4)));
        yield return ("subnormal", Make(_ => T.Epsilon * From<T>(random.Next(-5, 6))));
        yield return ("infinities", Make(i => i % 3 == 0 ? T.NegativeInfinity : i % 3 == 1 ? T.PositiveInfinity : From<T>(random.Next(-3, 4))));
        if (n > 0)
        {
            // NaN は符号と仮数のビットを変えて、元の順に並ぶことを確かめられるようにする
            int payload = 0;
            T NextNaN() => BitConverterNaN<T>(++payload, payload % 2 == 0);
            yield return ("many NaN", Make(_ => random.NextDouble() < 0.3 ? NextNaN() : From<T>(random.Next(-50, 50))));
            yield return ("all NaN", Make(_ => NextNaN()));
        }
    }

    private static T BitConverterNaN<T>(int payload, bool negative) where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (typeof(T) == typeof(double))
        {
            return Unsafe.BitCast<long, T>((negative ? unchecked((long)0xFFF8_0000_0000_0000) : 0x7FF8_0000_0000_0000) | (long)(uint)payload);
        }

        if (typeof(T) == typeof(float))
        {
            return Unsafe.BitCast<int, T>((negative ? unchecked((int)0xFFC0_0000) : 0x7FC0_0000) | payload);
        }

        if (typeof(T) == typeof(Half))
        {
            return Unsafe.BitCast<short, T>((short)((negative ? unchecked((short)0xFE00) : 0x7E00) | (payload & 0x1FF)));
        }

        return T.NaN;
    }

    private static void CheckSort<T>(int maximumLength) where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
    {
        foreach (int n in Lengths.Where(n => n <= maximumLength))
        {
            foreach (var (name, x) in Patterns<T>(n, n + 17))
            {
                string label = $"{typeof(T).Name} n={n} {name}";
                var original = x.ToArray();
                var expected = ReferenceSort(x);
                Assert.Equal(expected.Select(Bits), Sort(x.ToVector()).ToArray().Select(Bits));

                // 行列の行（ストライド 3）から、別の行へ（作業領域を通る）と、連続した出力先へ（出力先の上で並べる）
                var strided = new Matrix<T>(3, n);
                for (int i = 0; i < n; i++)
                {
                    strided[1, i] = x[i];
                }

                Assert.Equal(expected.Select(Bits), Sort(strided.Row(1), strided.Row(2)).ToArray().Select(Bits));
                var contiguous = new Tuna.LinearAlgebra.Vector<T>(n);
                Assert.Equal(expected.Select(Bits), Sort(strided.Row(1), contiguous).ToArray().Select(Bits));

                // 出力先が入力そのもの・その場の並べ替え（連続とストライド付き）
                var vector = x.ToVector();
                Sort(vector, vector);
                Assert.Equal(expected.Select(Bits), vector.ToArray().Select(Bits));
                vector = x.ToVector();
                InPlace.Sort(vector);
                Assert.Equal(expected.Select(Bits), vector.ToArray().Select(Bits));
                InPlace.Sort(strided.Row(1));
                Assert.Equal(expected.Select(Bits), strided.Row(1).ToArray().Select(Bits));

                // 入力と1つずれた出力先は n >= 2 で部分的に重なる。拒否する場合は入力・出力とも変更しない。
                // n <= 1 は重ならないので、同じ配列の中でも書き込める。
                var shifted = new Tuna.LinearAlgebra.Vector<T>(n + 1);
                for (int i = 0; i < n; i++)
                {
                    shifted[i] = x[i];
                }

                void SortShifted(bool forward)
                {
                    VectorSpan<T> whole = shifted;
                    if (forward)
                    {
                        Sort(whole[..n], whole[1..]);
                    }
                    else
                    {
                        Sort(whole[1..], whole[..n]);
                    }
                }

                if (n >= 2)
                {
                    var before = shifted.ToArray();
                    Assert.Throws<ArgumentException>(() => SortShifted(true));
                    Assert.Equal(before.Select(Bits), shifted.ToArray().Select(Bits));
                }
                else
                {
                    SortShifted(true);
                    Assert.Equal(expected.Select(Bits), shifted.ToArray()[1..].Select(Bits));
                }

                for (int i = 0; i < n; i++)
                {
                    shifted[i + 1] = x[i];
                }

                if (n >= 2)
                {
                    var before = shifted.ToArray();
                    Assert.Throws<ArgumentException>(() => SortShifted(false));
                    Assert.Equal(before.Select(Bits), shifted.ToArray().Select(Bits));
                }
                else
                {
                    SortShifted(false);
                    Assert.Equal(expected.Select(Bits), shifted.ToArray()[..n].Select(Bits));
                }

                var order = ArgSort(x.ToVector());
                Assert.True(ReferenceArgSort(x).SequenceEqual(order), label);
                Assert.Equal(expected.Select(Bits), order.Select(i => Bits(x[i])));
                Assert.Equal(original.Select(Bits), x.Select(Bits));
            }
        }
    }

    [Fact]
    public void SortAndArgSortFollowTheTotalOrderForDouble() => CheckSort<double>(int.MaxValue);

    [Fact]
    public void SortAndArgSortFollowTheTotalOrderForSingleHalfAndNFloat()
    {
        CheckSort<float>(int.MaxValue);
        CheckSort<Half>(4097);
        CheckSort<NFloat>(4097);
    }

    [Fact]
    public void SortAndArgSortHandleIntegersAndSmallCases()
    {
        var random = new Random(5);
        foreach (int n in new[] { 0, 1, 2, 7, 100, 1001 })
        {
            int[] ints = Enumerable.Range(0, n).Select(_ => random.Next(-20, 20)).ToArray();
            int[] sorted = ints.Order().ToArray();
            Assert.Equal(sorted, Sort(ints.ToVector()).ToArray());
            Assert.Equal(ReferenceArgSort(ints), ArgSort(ints.ToVector()));
            long[] longs = ints.Select(v => v * (long.MaxValue / 64)).ToArray();
            Assert.Equal(longs.Order().ToArray(), Sort(longs.ToVector()).ToArray());
            Assert.Equal(ReferenceArgSort(longs), ArgSort(longs.ToVector()));
        }

        // -0 は +0 の前（等しくはない）、NaN は最後に元の順。ArgSort は等しい値で元の位置の順
        double nanA = BitConverter.Int64BitsToDouble(0x7FF8_0000_0000_0042), nanB = BitConverter.Int64BitsToDouble(unchecked((long)0xFFF8_0000_0000_0099));
        double[] x = [0.0, nanA, -0.0, 1, double.NegativeInfinity, nanB, -0.0, 1, double.PositiveInfinity];
        Assert.Equal(new[] { double.NegativeInfinity, -0.0, -0.0, 0.0, 1, 1, double.PositiveInfinity, nanA, nanB }.Select(BitConverter.DoubleToInt64Bits),
            Sort(x.ToVector()).ToArray().Select(BitConverter.DoubleToInt64Bits));
        Assert.Equal(new[] { 4, 2, 6, 0, 3, 7, 8, 1, 5 }, ArgSort(x.ToVector()));

        var destination = new int[x.Length];
        Assert.True(Unsafe.AreSame(ref destination[0], ref MemoryMarshal.GetReference(ArgSort(x.ToVector(), destination))));
        var sortDestination = new Tuna.LinearAlgebra.Vector<double>(x.Length);
        Assert.Equal(Sort(x.ToVector()).ToArray().Select(BitConverter.DoubleToInt64Bits),
            Sort(x.ToVector(), sortDestination).ToArray().Select(BitConverter.DoubleToInt64Bits));
        Assert.Throws<ArgumentException>(() => { Sort(x.ToVector(), new Tuna.LinearAlgebra.Vector<double>(3)); });
        Assert.Throws<ArgumentException>(() => { ArgSort(x.ToVector(), new int[3]); });
        Assert.Empty(ArgSort(new Tuna.LinearAlgebra.Vector<double>(0)));
    }

    [Fact]
    public void ArgSortRejectsDestinationOverlappingItsInput()
    {
        // 型が違っても、出力先が入力と重なれば書き込む前に拒否する。番兵も含めて変更しない。
        float[] storage = [float.NaN, 2, 1, float.NaN];
        var original = storage.ToArray();
        Assert.Throws<ArgumentException>(() =>
        {
            var input = new ReadOnlyVectorSpan<float>(storage.AsSpan(0, 3));
            ArgSort(input, MemoryMarshal.Cast<float, int>(storage.AsSpan(0, 3)));
        });
        Assert.Equal(original.Select(Bits), storage.Select(Bits));

        var random = new Random(9);
        foreach (int n in new[] { 1, 2, 17, 600, 1001 })
        {
            float[] values = Enumerable.Range(0, n).Select(_ => random.NextDouble() < 0.2 ? float.NaN : random.Next(-5, 5)).ToArray();
            float[] shared = [.. values, float.NaN];
            foreach (int offset in n == 1 ? new[] { 0 } : new[] { 0, 1 })
            {
                var before = shared.ToArray();
                Assert.Throws<ArgumentException>(() =>
                {
                    var input = new ReadOnlyVectorSpan<float>(shared.AsSpan(0, n));
                    ArgSort(input, MemoryMarshal.Cast<float, int>(shared.AsSpan(offset, n)));
                });
                Assert.Equal(before.Select(Bits), shared.Select(Bits));
            }

            // 整数は値を比べる経路を通る。出力先が入力そのもの
            int[] integers = values.Select(v => float.IsNaN(v) ? 7 : (int)v).ToArray();
            var beforeIntegers = integers.ToArray();
            Assert.Throws<ArgumentException>(() => { ArgSort(new ReadOnlyVectorSpan<int>(integers), integers); });
            Assert.Equal(beforeIntegers, integers);
        }

        // 同じ配列でも領域が重ならなければ出力できる。
        float[] disjoint = [float.NaN, 2, 1, 0, 0, 0];
        Assert.Equal(new[] { 2, 1, 0 }, ArgSort(new ReadOnlyVectorSpan<float>(disjoint.AsSpan(0, 3)),
            MemoryMarshal.Cast<float, int>(disjoint.AsSpan(3, 3))).ToArray());
        Assert.Equal(original[..3].Select(Bits), disjoint[..3].Select(Bits));
    }

    private static void CheckQuantiles<T>(int maximumLength) where T : unmanaged, IFloatingPointIeee754<T>, IMinMaxValue<T>
    {
        foreach (int n in Lengths.Where(n => n <= maximumLength))
        {
            foreach (var (name, x) in Patterns<T>(n, n))
            {
                var original = x.ToArray();
                var strided = new Matrix<T>(3, n);
                for (int i = 0; i < n; i++)
                {
                    strided[1, i] = x[i];
                }

                foreach (bool ignoreNaN in new[] { false, true })
                {
                    string label = $"{typeof(T).Name} n={n} {name} ignoreNaN={ignoreNaN}";
                    var batch = Quantiles(x.ToVector(), Probabilities, ignoreNaN);
                    for (int k = 0; k < Probabilities.Length; k++)
                    {
                        double p = Probabilities[k];
                        T expected = ReferenceQuantile(x, p, ignoreNaN);
                        AssertBits($"Quantile p={p} {label}", expected, Quantile(x.ToVector(), p, ignoreNaN));
                        AssertBits($"Quantile strided p={p} {label}", expected, Quantile(strided.Row(1), p, ignoreNaN));
                        AssertBits($"Quantiles p={p} {label}", expected, batch[k]);
                        if (!ignoreNaN)
                        {
                            AssertBits($"Quantile default p={p} {label}", expected, Quantile(x.ToVector(), p));
                        }
                    }

                    AssertBits($"Quantile(0.5) = Median {label}", Median(x.ToVector(), ignoreNaN), Quantile(x.ToVector(), 0.5, ignoreNaN));
                }

                Assert.Equal(original.Select(Bits), x.Select(Bits));
            }
        }
    }

    [Fact]
    public void QuantileMatchesTheType7DefinitionForDouble() => CheckQuantiles<double>(int.MaxValue);

    [Fact]
    public void QuantileMatchesTheType7DefinitionForSingleHalfAndNFloat()
    {
        CheckQuantiles<float>(int.MaxValue);
        CheckQuantiles<Half>(1000);
        CheckQuantiles<NFloat>(1000);
    }

    [Fact]
    public void QuantileMatchesKnownValuesAndStaysMonotone()
    {
        // numpy.quantile（method="linear"）と R の quantile（type = 7）の値
        var x = new double[] { 4, 1, 3, 2 }.ToVector();
        Assert.Equal(new[] { 1, 1.75, 2.5, 3.25, 4 }, Quantiles(x, [0, 0.25, 0.5, 0.75, 1]).ToArray());
        var ten = Enumerable.Range(1, 10).Select(v => (double)v).ToArray().ToVector();
        Assert.Equal(9.1, Quantile(ten, 0.9));
        Assert.Equal(1.09, Quantile(ten, 0.01), 15);
        Assert.Equal(5.5, Quantile(ten, 0.5));
        Assert.Equal(7.0, Quantile(new double[] { 7 }.ToVector(), 0.3));

        // 補間した値は隣の2点の間に収まり、p について単調
        var random = new Random(11);
        var values = Enumerable.Range(0, 257).Select(_ => Math.Exp(20 * random.NextDouble() - 10) * (random.Next(2) == 0 ? -1 : 1)).ToArray();
        var sorted = values.Order().ToArray();
        double[] grid = Enumerable.Range(0, 10001).Select(i => i / 10000.0).ToArray();
        var q = Quantiles(values.ToVector(), grid).ToArray();
        for (int i = 0; i < grid.Length; i++)
        {
            double h = 256 * grid[i];
            int j = Math.Min((int)Math.Floor(h), 255);
            Assert.InRange(q[i], sorted[j], sorted[j + 1]);
            if (i > 0)
            {
                Assert.True(q[i - 1] <= q[i], $"p={grid[i]}");
            }
        }

        // 無限大と、差が溢れる区間
        Assert.True(double.IsNaN(Quantile(new[] { double.NegativeInfinity, double.PositiveInfinity }.ToVector(), 0.25)));
        Assert.Equal(double.NegativeInfinity, Quantile(new[] { double.NegativeInfinity, 0 }.ToVector(), 0.75));
        Assert.Equal(double.PositiveInfinity, Quantile(new[] { 0, double.PositiveInfinity }.ToVector(), 0.25));
        double wide = Quantile(new[] { double.MaxValue, -double.MaxValue }.ToVector(), 0.25);       // 差 b - a は溢れる
        Assert.True(Math.Abs(wide + 0.5 * double.MaxValue) <= 1e-15 * double.MaxValue, $"{wide:R}");
        Assert.Equal(0.875 * double.MaxValue, Quantile(new[] { double.MaxValue, 0.75 * double.MaxValue }.ToVector(), 0.5));

        // NaN と空
        double first = BitConverter.Int64BitsToDouble(0x7FF8_0000_0000_0042);
        var withNaN = new[] { 1, first, 3, double.NaN }.ToVector();
        Assert.Equal(0x7FF8_0000_0000_0042, BitConverter.DoubleToInt64Bits(Quantile(withNaN, 0.3)));
        Assert.All(Quantiles(withNaN, [0.1, 0.9]).ToArray(), v => Assert.Equal(0x7FF8_0000_0000_0042, BitConverter.DoubleToInt64Bits(v)));
        Assert.Equal(1.6, Quantile(withNaN, 0.3, ignoreNaN: true), 15);
        Assert.True(double.IsNaN(Quantile(new Tuna.LinearAlgebra.Vector<double>(0), 0.5)));
        Assert.All(Quantiles(new[] { double.NaN }.ToVector(), [0.1, 0.9], ignoreNaN: true).ToArray(), v => Assert.True(double.IsNaN(v)));
        Assert.Empty(Quantiles(x, []).ToArray());
    }

    [Fact]
    public void QuantilesRejectDestinationOverlappingTheirProbabilities()
    {
        // 確率との部分的な重なり・完全な一致は、書き込む前に拒否する。
        var x = new double[] { 10, 20, 30 }.ToVector();
        double[] storage = [0.25, 0.5, 0.75, 0.0];
        var original = storage.ToArray();
        Assert.Throws<ArgumentException>(() => { Quantiles(x, storage.AsSpan(0, 3), new VectorSpan<double>(storage.AsSpan(1, 3))); });
        Assert.Equal(original, storage);
        Assert.Throws<ArgumentException>(() => { Quantiles(x, storage.AsSpan(1, 3), new VectorSpan<double>(storage.AsSpan(0, 3))); });
        Assert.Equal(original, storage);
        double[] same = [0.75, 0.25];
        Assert.Throws<ArgumentException>(() => { Quantiles(x, same, new VectorSpan<double>(same)); });
        Assert.Equal(new[] { 0.75, 0.25 }, same);

        // 出力の要素型が float でも、double の確率と同じ記憶域なら拒否する。
        var single = new float[] { 10, 20, 30 }.ToVector();
        Assert.Throws<ArgumentException>(() =>
        {
            var output = new VectorSpan<float>(MemoryMarshal.Cast<double, float>(same.AsSpan())[..same.Length]);
            Quantiles(single, same, output);
        });
        Assert.Equal(new[] { 0.75, 0.25 }, same);

        // 同じ配列内の隣り合う、重ならない領域への出力は認める。
        double[] disjoint = [0.25, 0.75, 0, 0];
        Quantiles(x, disjoint.AsSpan(0, 2), new VectorSpan<double>(disjoint.AsSpan(2, 2)));
        Assert.Equal(new[] { 0.25, 0.75, 15, 25 }, disjoint);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void QuantilesRejectDestinationOverlappingTheirInput(bool ignoreNaN)
    {
        double[] storage = [30, 10, 20, 40];
        var original = storage.ToArray();
        foreach (var (inputOffset, outputOffset) in new[] { (0, 0), (0, 1), (1, 0) })
        {
            Assert.Throws<ArgumentException>(() =>
            {
                var input = new ReadOnlyVectorSpan<double>(storage.AsSpan(inputOffset, 3));
                var output = new VectorSpan<double>(storage.AsSpan(outputOffset, 3));
                Quantiles(input, [0.25, 0.5, 0.75], ignoreNaN, output);
            });
            Assert.Equal(original, storage);
        }
    }

    [Fact]
    public void ColumnAndRowQuantilesMatchTheVectorQuantile()
    {
        foreach (var (rows, columns) in new[] { (0, 3), (3, 0), (1, 1), (5, 4), (64, 7), (601, 3) })
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

            foreach (double p in new[] { 0, 0.1, 0.5, 0.75, 1 })
            {
                foreach (bool ignoreNaN in new[] { false, true })
                {
                    string label = $"{rows}×{columns} p={p} ignoreNaN={ignoreNaN}";
                    var columnQuantiles = ColumnQuantiles(a, p, ignoreNaN);
                    var rowQuantiles = RowQuantiles(a.T, p, ignoreNaN);
                    for (int j = 0; j < columns; j++)
                    {
                        double[] column = Enumerable.Range(0, rows).Select(i => a[i, j]).ToArray();
                        AssertBits($"ColumnQuantiles {label} [{j}]", ReferenceQuantile(column, p, ignoreNaN), columnQuantiles[j]);
                        AssertBits($"RowQuantiles {label} [{j}]", ReferenceQuantile(column, p, ignoreNaN), rowQuantiles[j]);
                    }

                    var destination = new Tuna.LinearAlgebra.Vector<double>(columns);
                    Assert.Same(destination, ColumnQuantiles(a, p, ignoreNaN, destination));
                    Assert.Equal(columnQuantiles.ToArray().Select(Bits), destination.ToArray().Select(Bits));
                    var rowDestination = new Tuna.LinearAlgebra.Vector<double>(columns);
                    Assert.Same(rowDestination, RowQuantiles(a.T, p, ignoreNaN, rowDestination));
                    if (!ignoreNaN)
                    {
                        Assert.Equal(columnQuantiles.ToArray().Select(Bits), ColumnQuantiles(a, p).ToArray().Select(Bits));
                        Assert.Equal(columnQuantiles.ToArray().Select(Bits), RowQuantiles(a.T, p).ToArray().Select(Bits));
                    }
                }
            }
        }

        Assert.Throws<ArgumentException>(() => { ColumnQuantiles(new Matrix<double>(2, 3), 0.5, new Tuna.LinearAlgebra.Vector<double>(2)); });
        Assert.Throws<ArgumentException>(() => { RowQuantiles(new Matrix<double>(2, 3), 0.5, true, new Tuna.LinearAlgebra.Vector<double>(3)); });
        Assert.Throws<ArgumentException>(() => { Quantiles(new double[] { 1, 2 }.ToVector(), [0.5], new Tuna.LinearAlgebra.Vector<double>(2)); });
    }

    [Fact]
    public void QuantileRejectsProbabilitiesOutsideTheUnitInterval()
    {
        var x = new double[] { 1, 2, 3 }.ToVector();
        foreach (double p in new[] { -1e-300, -0.5, 1 + 1e-15, 2, double.NaN, double.PositiveInfinity, double.NegativeInfinity })
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => Quantile(x, p));
            Assert.Throws<ArgumentOutOfRangeException>(() => Quantile(new Tuna.LinearAlgebra.Vector<double>(0), p, ignoreNaN: true));
            Assert.Throws<ArgumentOutOfRangeException>(() => { Quantiles(x, [0.5, p]); });
            Assert.Throws<ArgumentOutOfRangeException>(() => { ColumnQuantiles(new Matrix<double>(2, 2), p); });
            Assert.Throws<ArgumentOutOfRangeException>(() => { RowQuantiles(new Matrix<double>(2, 2), p, ignoreNaN: true); });
        }

        Assert.Equal(1.0, Quantile(x, -0.0));
    }
}
