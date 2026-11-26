using Tuna.LinearAlgebra;
using Tuna.SignalProcessing;

namespace Tuna.Tests.SignalProcessing;

/// <summary>離散ウェーブレット変換: フィルタの性質、PyWavelets との照合、完全再構成、多段の分解、引数の検査</summary>
[Trait("Wavelets", "Discrete")]
public class DiscreteWaveletTests
{
    private static readonly WaveletBoundary[] Boundaries =
    [
        WaveletBoundary.Symmetric, WaveletBoundary.Periodization, WaveletBoundary.Zero,
        WaveletBoundary.Constant, WaveletBoundary.Reflect, WaveletBoundary.Periodic,
    ];

    public static IEnumerable<string> AllNames()
    {
        yield return "haar";
        for (int n = 1; n <= 20; n++)
        {
            yield return $"db{n}";
        }

        for (int n = 2; n <= 20; n++)
        {
            yield return $"sym{n}";
        }

        for (int n = 1; n <= 5; n++)
        {
            yield return $"coif{n}";
        }

        foreach (string orders in new[] { "1.1", "1.3", "1.5", "2.2", "2.4", "2.6", "2.8", "3.1", "3.3", "3.5", "3.7", "3.9", "4.4", "5.5", "6.8" })
        {
            yield return $"bior{orders}";
            yield return $"rbio{orders}";
        }
    }

    public static TheoryData<string> Names()
    {
        var data = new TheoryData<string>();
        foreach (string name in AllNames())
        {
            data.Add(name);
        }

        return data;
    }

    private static double[] RandomSignal(int length, int seed)
    {
        var random = new Random(seed);
        var x = new double[length];
        for (int i = 0; i < length; i++)
        {
            x[i] = 2 * random.NextDouble() - 1;
        }

        return x;
    }

    /// <summary>
    /// PyWavelets との照合の許容誤差（期待値の最大の大きさに対する比）。PyWavelets の Symlet と bior4.4・5.5・6.8 の表は
    /// 定義式の値と最大 1.5e−11 違うので緩める
    /// </summary>
    private static double ReferenceTolerance(string name, ReadOnlySpan<double> expected)
    {
        double scale = 0;
        foreach (double v in expected)
        {
            scale = Math.Max(scale, Math.Abs(v));
        }

        bool tabulated = name.StartsWith("sym", StringComparison.Ordinal) || name.EndsWith("4.4", StringComparison.Ordinal)
            || name.EndsWith("5.5", StringComparison.Ordinal) || name.EndsWith("6.8", StringComparison.Ordinal);
        return (tabulated ? 1e-10 : 1e-13) * Math.Max(scale, 1);
    }

    private static void AssertClose(ReadOnlySpan<double> expected, ReadOnlyVectorSpan<double> actual, double tolerance, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (!(Math.Abs(expected[i] - actual[i]) <= tolerance))
            {
                Assert.Fail($"{label}[{i}]: expected {expected[i]:R}, actual {actual[i]:R}");
            }
        }
    }

    // ---- フィルタ ----

    [Theory]
    [MemberData(nameof(Names))]
    public void FiltersArePerfectReconstructionFilterBanks(string name)
    {
        var w = DiscreteWavelet.FromName(name);
        Assert.Equal(name, w.Name);
        int f = w.FilterLength;
        Assert.Equal(f, w.DecompositionLow.Length);
        Assert.Equal(f, w.ReconstructionHigh.Length);
        Assert.True(Math.Abs(Sum(w.DecompositionLow) - Math.Sqrt(2)) < 4e-15, $"{name}: sum {Sum(w.DecompositionLow):R}");
        Assert.True(Math.Abs(Sum(w.ReconstructionLow) - Math.Sqrt(2)) < 4e-15, $"{name}: sum {Sum(w.ReconstructionLow):R}");

        // 双直交性: 分解の低域（逆順）と再構成の低域を 2m ずらした内積が δ_m（直交ウェーブレットでは h の直交性）
        for (int m = -(f / 2); m <= f / 2; m++)
        {
            double sum = 0;
            for (int k = 0; k < f; k++)
            {
                int j = k + 2 * m;
                if (j >= 0 && j < f)
                {
                    sum += w.DecompositionLow[f - 1 - k] * w.ReconstructionLow[j];
                }
            }

            Assert.True(Math.Abs(sum - (m == 0 ? 1 : 0)) < 2e-15, $"{name}: shift {m}: {sum:R}");
        }

        if (w.IsOrthogonal)
        {
            for (int k = 0; k < f; k++)
            {
                Assert.Equal(w.ReconstructionLow[k], w.DecompositionLow[f - 1 - k]);
            }
        }
    }

    [Theory]
    [MemberData(nameof(Names))]
    public void DecompositionWaveletHasItsVanishingMoments(string name)
    {
        var w = DiscreteWavelet.FromName(name);
        var g = w.DecompositionHigh;
        double center = (g.Length - 1) / 2.0;
        for (int p = 0; p < w.VanishingMoments; p++)
        {
            double sum = 0, magnitude = 0;
            for (int k = 0; k < g.Length; k++)
            {
                double term = Math.Pow(k - center, p) * g[k];
                sum += term;
                magnitude += Math.Abs(term);
            }

            Assert.True(Math.Abs(sum) <= 1e-12 * magnitude, $"{name}: moment {p} = {sum:R}");
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void CoifletScalingFunctionHasVanishingMoments(int order)
    {
        // 添字 −2·order … 4·order − 1 で Σ k^p h_k = 0（p = 1, …, 2·order − 1）
        var h = DiscreteWavelet.Coiflet(order).ReconstructionLow;
        for (int p = 1; p < 2 * order; p++)
        {
            double sum = 0, magnitude = 0;
            for (int i = 0; i < h.Length; i++)
            {
                double term = Math.Pow(i - 2 * order, p) * h[i];
                sum += term;
                magnitude += Math.Abs(term);
            }

            Assert.True(Math.Abs(sum) <= 1e-13 * magnitude, $"coif{order}: moment {p} = {sum:R}");
        }
    }

    [Fact]
    public void KnownFilters()
    {
        // db2 の閉じた形 (1 ± √3)/(4√2)、(3 ± √3)/(4√2)
        double r = Math.Sqrt(3), s = 4 * Math.Sqrt(2);
        AssertClose([(1 + r) / s, (3 + r) / s, (3 - r) / s, (1 - r) / s], DiscreteWavelet.Daubechies(2).ReconstructionLow.ToArray(), 4e-16, "db2");
        AssertClose([Math.Sqrt(0.5), Math.Sqrt(0.5)], DiscreteWavelet.Haar.ReconstructionLow.ToArray(), 0, "haar");

        // LeGall 5/3（bior2.2）: 分解 √2·(−1, 2, 6, 2, −1)/8、再構成 √2·(1, 2, 1)/4
        double q = Math.Sqrt(2);
        AssertClose([0, -q / 8, q / 4, 3 * q / 4, q / 4, -q / 8], DiscreteWavelet.Biorthogonal(2, 2).DecompositionLow.ToArray(), 4e-16, "bior2.2 dec");
        AssertClose([0, q / 4, q / 2, q / 4, 0, 0], DiscreteWavelet.Biorthogonal(2, 2).ReconstructionLow.ToArray(), 4e-16, "bior2.2 rec");
    }

    [Fact]
    public void FactoriesAndNames()
    {
        Assert.Same(DiscreteWavelet.Daubechies(4), DiscreteWavelet.FromName("db4"));
        Assert.Same(DiscreteWavelet.Symlet(8), DiscreteWavelet.FromName(" SYM8 "));
        Assert.Same(DiscreteWavelet.Coiflet(3), DiscreteWavelet.FromName("coif3"));
        Assert.Same(DiscreteWavelet.Biorthogonal(4, 4), DiscreteWavelet.FromName("bior4.4"));
        Assert.Same(DiscreteWavelet.ReverseBiorthogonal(3, 9), DiscreteWavelet.FromName("rbio3.9"));
        Assert.Same(DiscreteWavelet.Haar, DiscreteWavelet.FromName("haar"));
        Assert.Equal("db1", DiscreteWavelet.Daubechies(1).Name);
        Assert.Equal(4, DiscreteWavelet.Biorthogonal(4, 4).VanishingMoments);
        Assert.Equal(9, DiscreteWavelet.ReverseBiorthogonal(3, 9).VanishingMoments);
        Assert.Equal(6, DiscreteWavelet.Coiflet(3).VanishingMoments);
        Assert.False(DiscreteWavelet.Biorthogonal(2, 2).IsOrthogonal);
        Assert.True(DiscreteWavelet.Symlet(5).IsOrthogonal);

        foreach (string bad in new[] { "db0", "db21", "sym1", "sym21", "coif6", "bior2.3", "rbio7.7", "dmey", "db", "bior", "bior2", "" })
        {
            Assert.Throws<ArgumentException>(() => DiscreteWavelet.FromName(bad));
        }

        Assert.Throws<ArgumentOutOfRangeException>(() => DiscreteWavelet.Daubechies(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => DiscreteWavelet.Symlet(21));
        Assert.Throws<ArgumentOutOfRangeException>(() => DiscreteWavelet.Coiflet(6));
        Assert.Throws<ArgumentOutOfRangeException>(() => DiscreteWavelet.Biorthogonal(2, 3));
    }

    // ---- PyWavelets との照合 ----

    [Fact]
    public void SingleLevelMatchesPyWavelets()
    {
        foreach (var (name, boundary, x, expectedA, expectedD) in PyWaveletsReferences.Single)
        {
            var w = DiscreteWavelet.FromName(name);
            var (a, d) = Wavelets.Forward(x, w, boundary);
            double scale = x.Max(Math.Abs);
            string label = $"{name} {boundary} n={x.Length}";
            AssertClose(expectedA, a, ReferenceTolerance(name, expectedA), label + " a");
            AssertClose(expectedD, d, ReferenceTolerance(name, expectedD), label + " d");

            // 逆変換は PyWavelets の idwt と同じ長さで、先頭 n 個が元の信号
            var back = Wavelets.Inverse(a, d, w, boundary);
            Assert.Equal(Wavelets.InverseLength(a.Length, w, boundary), back.Length);
            Assert.True(back.Length == x.Length || back.Length == x.Length + 1, label);
            AssertClose(x, ((ReadOnlyVectorSpan<double>)back)[..x.Length], 1e-12 * scale, label + " inverse");
        }
    }

    [Fact]
    public void MultilevelMatchesPyWavelets()
    {
        foreach (var (name, boundary, levels, x, expected) in PyWaveletsReferences.Multilevel)
        {
            var w = DiscreteWavelet.FromName(name);
            var decomposition = levels < 0 ? Wavelets.Decompose(x, w, boundary) : Wavelets.Decompose(x, w, levels, boundary);
            string label = $"{name} {boundary} n={x.Length}";
            Assert.Equal(expected.Length - 1, decomposition.Levels);
            AssertClose(expected[0], decomposition.Approximation, ReferenceTolerance(name, expected[0]), label + " cA");
            for (int i = 0; i < decomposition.Levels; i++)
            {
                AssertClose(expected[i + 1], decomposition.Details[i], ReferenceTolerance(name, expected[i + 1]), $"{label} cD{decomposition.Levels - i}");
                Assert.Same(decomposition.Details[i], decomposition.Detail(decomposition.Levels - i));
            }

            var back = Wavelets.Reconstruct(decomposition);
            AssertClose(x, back, 1e-12, label + " reconstruct");
        }
    }

    [Fact]
    public void MaxLevelMatchesPyWavelets()
    {
        var byLength = new Dictionary<int, DiscreteWavelet>
        {
            [2] = DiscreteWavelet.Haar,
            [4] = DiscreteWavelet.Daubechies(2),
            [6] = DiscreteWavelet.Daubechies(3),
            [20] = DiscreteWavelet.Daubechies(10),
            [40] = DiscreteWavelet.Daubechies(20),
        };
        foreach (var (length, filterLength, expected) in PyWaveletsReferences.MaxLevels)
        {
            Assert.Equal(expected, Wavelets.MaxLevel(length, byLength[filterLength]));
        }
    }

    // ---- 完全再構成・エネルギー ----

    [Theory]
    [MemberData(nameof(Names))]
    public void InverseReconstructsTheSignalForEveryBoundary(string name)
    {
        var w = DiscreteWavelet.FromName(name);
        foreach (var boundary in Boundaries)
        {
            foreach (int n in new[] { 1, 2, 3, 4, 5, 8, 9, 17, 32, 41 })
            {
                if (boundary == WaveletBoundary.Reflect && n < 2)
                {
                    continue;
                }

                var x = RandomSignal(n, 31 * n + (int)boundary);
                var (a, d) = Wavelets.Forward(x, w, boundary);
                Assert.Equal(Wavelets.CoefficientLength(n, w, boundary), a.Length);
                var back = Wavelets.Inverse(a, d, w, boundary);
                AssertClose(x, ((ReadOnlyVectorSpan<double>)back)[..n], 1e-12, $"{name} {boundary} n={n}");
            }
        }
    }

    [Theory]
    [InlineData("haar")]
    [InlineData("db6")]
    [InlineData("sym9")]
    [InlineData("coif4")]
    public void PeriodizationIsOrthogonalForOrthogonalWavelets(string name)
    {
        var w = DiscreteWavelet.FromName(name);
        foreach (int n in new[] { 2, 8, 30, 64 })
        {
            var x = RandomSignal(n, n);
            var (a, d) = Wavelets.Forward(x, w, WaveletBoundary.Periodization);
            Assert.Equal(n / 2, a.Length);
            double energy = x.Sum(v => v * v);
            double coefficients = a.AsSpan().ToArray().Sum(v => v * v) + d.AsSpan().ToArray().Sum(v => v * v);
            Assert.Equal(energy, coefficients, 1e-12 * energy);
        }
    }

    [Fact]
    public void PolynomialsHaveZeroInteriorDetails()
    {
        // 次数 < 消失モーメントの多項式は、境界の影響を受けない詳細係数が 0
        foreach (string name in new[] { "db4", "sym6", "coif2", "bior3.5" })
        {
            var w = DiscreteWavelet.FromName(name);
            int n = 64;
            var x = new double[n];
            for (int i = 0; i < n; i++)
            {
                x[i] = Math.Pow((i - 30) / 10.0, w.VanishingMoments - 1) - 0.5 * i / 10.0;
            }

            var (_, d) = Wavelets.Forward(x, w, WaveletBoundary.Zero);
            double scale = x.Max(Math.Abs);
            int f = w.FilterLength;
            for (int k = f / 2; k < (n - 1) / 2; k++)
            {
                Assert.True(Math.Abs(d[k]) < 1e-11 * scale, $"{name} d[{k}] = {d[k]:R}");
            }
        }
    }

    // ---- 多段 ----

    [Fact]
    public void DecomposeAndReconstructRoundTrip()
    {
        foreach (string name in new[] { "haar", "db5", "sym7", "coif2", "bior4.4", "rbio2.6" })
        {
            var w = DiscreteWavelet.FromName(name);
            foreach (var boundary in Boundaries)
            {
                foreach (int n in new[] { 2, 13, 64, 101 })
                {
                    var x = RandomSignal(n, n + name.Length);
                    var decomposition = Wavelets.Decompose(x, w, boundary);
                    Assert.Equal(Wavelets.MaxLevel(n, w), decomposition.Levels);
                    Assert.Equal(n, decomposition.Length);
                    Assert.Same(w, decomposition.Wavelet);
                    Assert.Equal(boundary, decomposition.Boundary);
                    AssertClose(x, Wavelets.Reconstruct(decomposition), 1e-12, $"{name} {boundary} n={n}");

                    // 段数を最大より多くしても再構成できる（反射は近似係数が1つになる段から先へは進めない。長さ 2 のフィルタだけが1つまで縮む）
                    if (boundary == WaveletBoundary.Reflect && w.FilterLength == 2)
                    {
                        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.Decompose(x, w, 40, boundary));
                        continue;
                    }

                    var deep = Wavelets.Decompose(x, w, decomposition.Levels + 2, boundary);
                    AssertClose(x, Wavelets.Reconstruct(deep), 1e-11, $"{name} {boundary} n={n} deep");
                }
            }
        }
    }

    [Fact]
    public void ZeroLevelsCopiesTheSignal()
    {
        double[] x = [1, 2, 3];
        var decomposition = Wavelets.Decompose(x, DiscreteWavelet.Haar, 0);
        Assert.Equal(0, decomposition.Levels);
        AssertClose(x, decomposition.Approximation, 0, "approximation");
        AssertClose(x, Wavelets.Reconstruct(decomposition), 0, "reconstruct");
    }

    [Fact]
    public void ModifiedCoefficientsAreUsedInReconstruction()
    {
        // 詳細係数を 0 にすると、Haar では隣り合う2点ずつの平均に置き換わる
        double[] x = [1, 3, 2, 6, 5, 5, 0, 4];
        var decomposition = Wavelets.Decompose(x, DiscreteWavelet.Haar, 1);
        decomposition.Detail(1).AsSpan().Clear();
        AssertClose([2, 2, 4, 4, 5, 5, 2, 2], Wavelets.Reconstruct(decomposition), 1e-15, "smoothed");
    }

    // ---- ビューと出力先 ----

    [Fact]
    public void StridedViewsAndDestinationsMatchDenseResultsBitwise()
    {
        var w = DiscreteWavelet.Symlet(4);
        var m = Matrix<double>.FromColumnMajor(3, 20, RandomSignal(60, 5));
        var row = m.Row(1);
        var dense = row.ToVector();
        var (a, d) = Wavelets.Forward(dense, w, WaveletBoundary.Reflect);
        var (sa, sd) = Wavelets.Forward(row, w, WaveletBoundary.Reflect);
        Assert.Equal(a.AsSpan().ToArray(), sa.AsSpan().ToArray());
        Assert.Equal(d.AsSpan().ToArray(), sd.AsSpan().ToArray());

        int length = a.Length;
        var target = new Matrix<double>(2, length);
        Wavelets.Forward(row, w, WaveletBoundary.Reflect, target.Row(0), target.Row(1));
        Assert.Equal(a.AsSpan().ToArray(), target.Row(0).ToArray());
        Assert.Equal(d.AsSpan().ToArray(), target.Row(1).ToArray());

        var back = Wavelets.Inverse(a, d, w, WaveletBoundary.Reflect);
        var output = new Vector<double>(back.Length + 3);
        var returned = Wavelets.Inverse(target.Row(0), target.Row(1), w, WaveletBoundary.Reflect, output.AsSpan()[1..(1 + back.Length)]);
        Assert.Equal(back.AsSpan().ToArray(), returned.ToArray());
        Assert.Equal(0, output[0]);

        // 出力先が係数と重なっていてもよい（その場で上書き）
        var coefficients = new Vector<double>(2 * length);
        a.AsSpan().CopyTo(coefficients.AsSpan()[..length]);
        d.AsSpan().CopyTo(coefficients.AsSpan()[length..]);
        var inPlace = coefficients.AsSpan()[..back.Length];
        Wavelets.Inverse(coefficients.AsSpan()[..length], coefficients.AsSpan()[length..], w, WaveletBoundary.Reflect, inPlace);
        Assert.Equal(back.AsSpan().ToArray(), inPlace.ToArray());
    }

    [Fact]
    public void InvalidArgumentsThrow()
    {
        var w = DiscreteWavelet.Daubechies(2);
        Assert.Throws<ArgumentException>(() => Wavelets.Forward(ReadOnlySpan<double>.Empty, w));
        Assert.Throws<ArgumentException>(() => Wavelets.Forward(new double[] { 1 }, w, WaveletBoundary.Reflect));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.Forward(new double[] { 1, 2 }, w, (WaveletBoundary)17));
        Assert.Throws<ArgumentNullException>(() => Wavelets.Forward(new double[] { 1, 2 }, null!));
        Assert.Throws<ArgumentException>(() => Wavelets.Inverse(new double[] { 1, 2, 3 }, new double[] { 1, 2 }, w));
        Assert.Throws<ArgumentException>(() => Wavelets.Inverse(new double[] { 1 }, new double[] { 1 }, DiscreteWavelet.Daubechies(4)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.Decompose(new double[] { 1, 2 }, w, -1));

        var x = new double[] { 1, 2, 3, 4, 5 };
        var shared = new double[8];
        Assert.Throws<ArgumentException>(() => Wavelets.Forward(x, w, WaveletBoundary.Symmetric, shared.AsSpan(0, 4), shared.AsSpan(2, 4)));
        Assert.Throws<ArgumentException>(() => Wavelets.Forward(x, w, WaveletBoundary.Symmetric, new double[3], new double[4]));
        Assert.Throws<ArgumentException>(() => Wavelets.Inverse(new double[] { 1, 2 }, new double[] { 1, 2 }, w, WaveletBoundary.Symmetric, new double[3]));
    }

    [Fact]
    public void LengthsMatchTheDefinitions()
    {
        var w = DiscreteWavelet.Daubechies(4); // F = 8
        Assert.Equal(7, Wavelets.CoefficientLength(7, w, WaveletBoundary.Symmetric));
        Assert.Equal(4, Wavelets.CoefficientLength(7, w, WaveletBoundary.Periodization));
        Assert.Equal(8, Wavelets.InverseLength(7, w, WaveletBoundary.Symmetric));
        Assert.Equal(8, Wavelets.InverseLength(4, w, WaveletBoundary.Periodization));
        Assert.Equal(0, Wavelets.MaxLevel(0, w));
        Assert.Equal(0, Wavelets.MaxLevel(6, w));
        Assert.Equal(1, Wavelets.MaxLevel(14, w));
    }

    private static double Sum(ReadOnlySpan<double> values)
    {
        double sum = 0;
        foreach (double v in values)
        {
            sum += v;
        }

        return sum;
    }
}
