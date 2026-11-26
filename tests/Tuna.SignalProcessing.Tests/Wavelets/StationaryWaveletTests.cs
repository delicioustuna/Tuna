using Tuna.LinearAlgebra;
using Tuna.SignalProcessing;
using static Tuna.Tests.SignalProcessing.WaveletAssertions;

namespace Tuna.Tests.SignalProcessing;

/// <summary>定常ウェーブレット変換（SWT）と最大重複離散ウェーブレット変換（MODWT）</summary>
[Trait("Wavelets", "Stationary")]
public class StationaryWaveletTests
{
    public static TheoryData<string> Names()
    {
        var data = new TheoryData<string>();
        foreach (string name in DiscreteWaveletTests.AllNames())
        {
            data.Add(name);
        }

        return data;
    }

    public static TheoryData<string> OrthogonalNames()
    {
        var data = new TheoryData<string>();
        foreach (string name in DiscreteWaveletTests.AllNames())
        {
            if (DiscreteWavelet.FromName(name).IsOrthogonal)
            {
                data.Add(name);
            }
        }

        return data;
    }

    // ---- SWT ----

    [Fact]
    public void StationaryMatchesPyWavelets()
    {
        foreach (var (name, levels, x, expected) in WaveletExtensionReferences.Stationary)
        {
            var w = DiscreteWavelet.FromName(name);
            var c = Wavelets.Stationary(x, w, levels);
            Assert.Equal(levels + 1, c.Rows);
            for (int row = 0; row <= levels; row++)
            {
                Close(expected[row], c.Row(row), Tolerance(name, MaxAbs(expected[row])), $"{name} J={levels} row {row}");
            }
        }
    }

    [Fact]
    public void InverseStationaryMatchesPyWavelets()
    {
        // 係数は変換の結果でない乱数なので、逆変換の式そのものを確かめる
        foreach (var (name, coefficients, expected) in WaveletExtensionReferences.InverseStationary)
        {
            var w = DiscreteWavelet.FromName(name);
            var c = Matrix<double>.FromRowMajor(coefficients.Length, coefficients[0].Length, coefficients.SelectMany(r => r).ToArray());
            Close(expected, Wavelets.InverseStationary(c, w), Tolerance(name, MaxAbs(expected)), $"{name} iswt");
        }
    }

    [Theory]
    [MemberData(nameof(Names))]
    public void StationaryRoundTrips(string name)
    {
        var w = DiscreteWavelet.FromName(name);
        foreach (int n in new[] { 1, 2, 7, 37, 64 })
        {
            var x = RandomSignal(n, n);
            int maxLevel = 31 - int.LeadingZeroCount(n);
            for (int levels = 0; levels <= maxLevel; levels++)
            {
                var c = Wavelets.Stationary(x, w, levels);
                Close(x, Wavelets.InverseStationary(c, w), 1e-11, $"{name} n={n} J={levels}");
            }
        }
    }

    [Theory]
    [InlineData("db2", 13, 3)]
    [InlineData("bior2.2", 21, 4)]
    [InlineData("coif2", 6, 2)]
    [InlineData("haar", 11, 3)]
    public void StationaryMatchesDefinitionForAnyLength(string name, int n, int levels)
    {
        // PyWavelets は n が 2^levels で割り切れることを求めるが、定義の式は任意の n で意味を持つ
        var w = DiscreteWavelet.FromName(name);
        var x = RandomSignal(n, 7);
        var c = Wavelets.Stationary(x, w, levels);
        int f = w.FilterLength;
        var a = (double[])x.Clone();
        for (int j = 1; j <= levels; j++)
        {
            int s = 1 << (j - 1);
            var next = new double[n];
            var d = new double[n];
            for (int t = 0; t < n; t++)
            {
                for (int l = 0; l < f; l++)
                {
                    double v = a[Modulo(t + s * (f / 2 - l), n)];
                    next[t] += w.DecompositionLow[l] * v;
                    d[t] += w.DecompositionHigh[l] * v;
                }
            }

            Close(d, c.Row(levels + 1 - j), 1e-14, $"{name} detail {j}");
            a = next;
        }

        Close(a, c.Row(0), 1e-13, $"{name} approximation");
    }

    [Theory]
    [InlineData("db3", 24, 3)]
    [InlineData("bior3.5", 17, 2)]
    public void StationaryIsShiftEquivariant(string name, int n, int levels)
    {
        // 入力を巡回して1つずらすと、係数もちょうど1つずれる（各値の計算の手順が同じなのでビット単位で）
        var w = DiscreteWavelet.FromName(name);
        var x = RandomSignal(n, 3);
        var shifted = new double[n];
        for (int t = 0; t < n; t++)
        {
            shifted[t] = x[Modulo(t - 1, n)];
        }

        var c = Wavelets.Stationary(x, w, levels);
        var cs = Wavelets.Stationary(shifted, w, levels);
        for (int row = 0; row <= levels; row++)
        {
            for (int t = 0; t < n; t++)
            {
                Assert.Equal(BitConverter.DoubleToInt64Bits(c[row, Modulo(t - 1, n)]), BitConverter.DoubleToInt64Bits(cs[row, t]));
            }
        }
    }

    [Fact]
    public void StationaryDestinationsMayAliasTheInput()
    {
        var w = DiscreteWavelet.FromName("sym4");
        var x = RandomSignal(40, 11);
        var expected = Wavelets.Stationary(x, w, 3);

        // 出力先の最後の行に入力を置く
        var destination = new Matrix<double>(4, 40);
        for (int t = 0; t < 40; t++)
        {
            destination[3, t] = x[t];
        }

        Wavelets.Stationary(destination.Row(3), w, destination);
        Identical(expected, destination, "swt in place");

        var back = expected.Clone();
        Wavelets.InverseStationary(back, w, back.Row(1));
        Identical(Wavelets.InverseStationary(expected, w), back.Row(1), "iswt in place");
    }

    [Fact]
    public void StationaryValidatesArguments()
    {
        var w = DiscreteWavelet.Daubechies(2);
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.Stationary(new double[8], w, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.Stationary(new double[8], w, -1));
        Assert.Throws<ArgumentException>(() => Wavelets.Stationary(Array.Empty<double>(), w, 0));
        Assert.Throws<ArgumentNullException>(() => Wavelets.Stationary(new double[8], null!, 1));
        Assert.Throws<ArgumentException>(() => Wavelets.Stationary(new double[8], w, new Matrix<double>(3, 7)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.InverseStationary(new Matrix<double>(4, 7), w));
        Assert.Throws<ArgumentException>(() => Wavelets.InverseStationary(new Matrix<double>(2, 8), w, new double[7]));
        Assert.Equal(1, Wavelets.Stationary(new double[] { 2.5 }, w, 0).Rows);
    }

    // ---- MODWT ----

    [Fact]
    public void MaximalOverlapMatchesReference()
    {
        foreach (var (name, levels, x, expected) in WaveletExtensionReferences.MaximalOverlap)
        {
            var w = DiscreteWavelet.FromName(name);
            var c = Wavelets.MaximalOverlap(x, w, levels);
            for (int row = 0; row <= levels; row++)
            {
                Close(expected[row], c.Row(row), Tolerance(name, MaxAbs(expected[row])), $"{name} J={levels} row {row}");
            }
        }
    }

    [Fact]
    public void HaarMaximalOverlapIsHalfTheDifference()
    {
        // Percival–Walden の第5章: Haar の W̃_{1,t} = (X_t − X_{t−1})/2、Ṽ_{1,t} = (X_t + X_{t−1})/2
        var x = RandomSignal(19, 5);
        var c = Wavelets.MaximalOverlap(x, DiscreteWavelet.Haar, 1);
        for (int t = 0; t < x.Length; t++)
        {
            double previous = x[Modulo(t - 1, x.Length)];
            Assert.Equal((x[t] - previous) / 2, c[1, t], 1e-15);
            Assert.Equal((x[t] + previous) / 2, c[0, t], 1e-15);
        }
    }

    [Theory]
    [MemberData(nameof(OrthogonalNames))]
    public void MaximalOverlapPreservesEnergyAndRoundTrips(string name)
    {
        var w = DiscreteWavelet.FromName(name);
        foreach (int n in new[] { 1, 5, 64, 100 })
        {
            var x = RandomSignal(n, n + 1);
            double energy = x.Sum(v => v * v);
            int maxLevel = 31 - int.LeadingZeroCount(n);
            for (int levels = 0; levels <= Math.Min(maxLevel, 5); levels++)
            {
                var c = Wavelets.MaximalOverlap(x, w, levels);
                double sum = 0;
                foreach (double v in c.AsSpan())
                {
                    sum += v * v;
                }

                Assert.True(Math.Abs(sum - energy) <= 1e-13 * energy, $"{name} n={n} J={levels}: {sum:R} vs {energy:R}");
                Close(x, Wavelets.InverseMaximalOverlap(c, w), 1e-13, $"{name} n={n} J={levels}");
            }
        }
    }

    [Theory]
    [InlineData("db4", 64, 4)]
    [InlineData("sym6", 50, 3)]
    [InlineData("coif1", 33, 5)]
    [InlineData("haar", 16, 4)]
    public void MultiresolutionComponentsAddUpToTheSignal(string name, int n, int levels)
    {
        var w = DiscreteWavelet.FromName(name);
        var x = RandomSignal(n, 9);
        var mra = Wavelets.MaximalOverlapMultiresolution(x, w, levels);
        var c = Wavelets.MaximalOverlap(x, w, levels);
        for (int t = 0; t < n; t++)
        {
            double sum = 0;
            for (int row = 0; row <= levels; row++)
            {
                sum += mra[row, t];
            }

            Assert.True(Math.Abs(sum - x[t]) <= 1e-13, $"{name} t={t}: {sum:R} vs {x[t]:R}");
        }

        // 各成分は、その段だけを残した係数の逆変換
        for (int row = 0; row <= levels; row++)
        {
            var only = new Matrix<double>(levels + 1, n);
            c.Row(row).CopyTo(only.Row(row));
            Close(ToArray(Wavelets.InverseMaximalOverlap(only, w)), mra.Row(row), 1e-14, $"{name} component {row}");
        }
    }

    [Fact]
    public void MaximalOverlapValidatesArguments()
    {
        Assert.Throws<ArgumentException>(() => Wavelets.MaximalOverlap(new double[16], DiscreteWavelet.Biorthogonal(2, 2), 2));
        Assert.Throws<ArgumentException>(() => Wavelets.InverseMaximalOverlap(new Matrix<double>(3, 16), DiscreteWavelet.FromName("rbio1.3")));
        Assert.Throws<ArgumentException>(() => Wavelets.MaximalOverlapMultiresolution(new double[16], DiscreteWavelet.FromName("bior1.1"), 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.MaximalOverlap(new double[15], DiscreteWavelet.Haar, 4));

        // destination オーバーロードは新規割り当てオーバーロードと同じ結果を返す
        var x = RandomSignal(30, 2);
        var w = DiscreteWavelet.Daubechies(3);
        var destination = new Matrix<double>(3, 30);
        Wavelets.MaximalOverlap(x, w, destination);
        Identical(Wavelets.MaximalOverlap(x, w, 2), destination, "modwt destination");
        var back = new double[30];
        Wavelets.InverseMaximalOverlap(destination, w, back);
        Identical(Wavelets.InverseMaximalOverlap(destination, w), back, "imodwt destination");
    }

    private static int Modulo(int value, int modulus) => ((value % modulus) + modulus) % modulus;
}
