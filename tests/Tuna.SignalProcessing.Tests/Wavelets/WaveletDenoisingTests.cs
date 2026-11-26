using Tuna.LinearAlgebra;
using Tuna.SignalProcessing;
using static Tuna.Tests.SignalProcessing.WaveletAssertions;

namespace Tuna.Tests.SignalProcessing;

/// <summary>ウェーブレット係数のしきい値処理によるノイズ除去</summary>
[Trait("Wavelets", "Denoising")]
public class WaveletDenoisingTests
{
    [Fact]
    public void DenoiseMatchesScikitImage()
    {
        foreach (var (name, rule, shrinkage, levels, noiseLevel, x, expected) in WaveletExtensionReferences.Denoise)
        {
            var w = DiscreteWavelet.FromName(name);
            Vector<double> y;
            if (!double.IsNaN(noiseLevel))
            {
                int l = levels < 0 ? Math.Max(Wavelets.MaxLevel(x.Length, w) - 3, 1) : levels;
                y = Wavelets.Denoise(x, w, rule, shrinkage, l, noiseLevel);
            }
            else if (levels >= 0)
            {
                y = Wavelets.Denoise(x, w, rule, shrinkage, levels);
            }
            else if (rule == WaveletThresholdRule.Bayes && shrinkage == WaveletShrinkage.Soft)
            {
                y = Wavelets.Denoise(x, w);
            }
            else
            {
                y = Wavelets.Denoise(x, w, rule, shrinkage);
            }

            Close(expected, y, Tolerance(name, 1), $"{name} {rule} {shrinkage} n={x.Length}");
        }
    }

    [Fact]
    public void Denoise2DMatchesScikitImage()
    {
        foreach (var (name, rule, shrinkage, levels, x, expected) in WaveletExtensionReferences.Denoise2D)
        {
            var w = DiscreteWavelet.FromName(name);
            var matrix = new Matrix<double>(x);
            var y = levels >= 0 ? Wavelets.Denoise2D(matrix, w, rule, shrinkage, levels)
                : rule == WaveletThresholdRule.Bayes && shrinkage == WaveletShrinkage.Soft ? Wavelets.Denoise2D(matrix, w)
                : Wavelets.Denoise2D(matrix, w, rule, shrinkage);
            Close(expected, y, Tolerance(name, 1), $"{name} {rule} {shrinkage} {x.GetLength(0)}x{x.GetLength(1)}");
        }
    }

    [Fact]
    public void EstimateNoiseMatchesScikitImage()
    {
        // scikit-image の estimate_sigma は db2 の1段の詳細係数（対称拡張）から推定する
        foreach (var (x, sigma) in WaveletExtensionReferences.NoiseEstimates)
        {
            var (_, detail) = Wavelets.Forward(x, DiscreteWavelet.Daubechies(2));
            Assert.Equal(sigma, Wavelets.EstimateNoise(detail), 1e-15 * Math.Max(sigma, 1));
        }

        // 中央値（偶数個は中央の2つの平均）、0 の係数は除く
        Assert.Equal(1.5 / 0.6744897501960817, Wavelets.EstimateNoise(new double[] { 0, -1, 2, 0, -3, 1 }), 1e-15);
        Assert.Equal(2.0 / 0.6744897501960817, Wavelets.EstimateNoise(new double[] { 0, -1, 2, -3 }), 1e-15);
        Assert.Equal(0, Wavelets.EstimateNoise(new double[] { 0, 0, 0 }));
        Assert.Equal(0, Wavelets.EstimateNoise(Array.Empty<double>()));
        Assert.Equal(1.0 / 0.6744897501960817, Wavelets.EstimateNoise(Matrix<double>.FromRowMajor(2, 2, [0, 1, -1, 1])), 1e-15);
        Assert.Throws<ArgumentException>(() => Wavelets.EstimateNoise(new double[] { 1, double.NaN }));
    }

    [Fact]
    public void ThresholdRules()
    {
        Assert.Equal(0, Wavelets.UniversalThreshold(2.0, 1));
        Assert.Equal(0.3 * Math.Sqrt(2 * Math.Log(1000)), Wavelets.UniversalThreshold(0.3, 1000), 1e-15);
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.UniversalThreshold(-1, 10));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.UniversalThreshold(double.NaN, 10));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.UniversalThreshold(1, 0));

        // 平均(d²) = 5、σ = 1 なら 1/√(5 − 1) = 0.5
        Assert.Equal(0.5, Wavelets.BayesThreshold(new double[] { 1, -3 }, 1.0), 1e-15);
        Assert.Equal(0.5, Wavelets.BayesThreshold(Matrix<double>.FromRowMajor(1, 2, [1, -3]), 1.0), 1e-15);

        // 係数がノイズより小さければ、しきい値は σ²/√ε で、係数をすべて 0 にする
        Assert.Equal(4.0 / Math.Sqrt(Math.Pow(2, -52)), Wavelets.BayesThreshold(new double[] { 1, -1 }, 2.0), 1e-3);
        Assert.Equal(0, Wavelets.BayesThreshold(new double[] { 1, -1 }, 0.0));
        Assert.Throws<ArgumentException>(() => Wavelets.BayesThreshold(Array.Empty<double>(), 1.0));
    }

    [Fact]
    public void ShrinkSoftAndHard()
    {
        var soft = new double[] { -3, -1, -0.5, 0, 0.5, 1, 3, double.NaN };
        Wavelets.Shrink(soft, 1.0, WaveletShrinkage.Soft);
        Assert.Equal([-2, 0, 0, 0, 0, 0, 2, double.NaN], soft);

        // ハードしきい値は |x| < T を 0 にし、|x| = T は残す（PyWavelets と同じ）
        var hard = new double[] { -3, -1, -0.5, 0, 0.5, 1, 3, double.NaN };
        Wavelets.Shrink(hard, 1.0, WaveletShrinkage.Hard);
        Assert.Equal([-3, -1, 0, 0, 0, 1, 3, double.NaN], hard);

        var matrix = Matrix<double>.FromRowMajor(2, 2, [-2, 0.5, 1.5, -0.25]);
        Wavelets.Shrink(matrix, 1.0, WaveletShrinkage.Hard);
        Assert.Equal(new double[,] { { -2, 0 }, { 1.5, 0 } }, matrix.ToArray());
        Wavelets.Shrink(matrix, 1.0, WaveletShrinkage.Soft);
        Assert.Equal(new double[,] { { -1, 0 }, { 0.5, 0 } }, matrix.ToArray());

        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.Shrink(new double[2], -1, WaveletShrinkage.Hard));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.Shrink(new double[2], double.NaN, WaveletShrinkage.Hard));
        Assert.ThrowsAny<ArgumentException>(() => Wavelets.Shrink(new double[2], -1, WaveletShrinkage.Soft));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.Shrink(new double[2], 1, (WaveletShrinkage)2));
    }

    [Fact]
    public void DenoiseLeavesNoiseFreeSignalsAndReducesNoise()
    {
        // 区分的に定数の信号の Haar の最も細かい詳細係数は、段差の1か所を除いて 0。σ の推定はその1つだけから決まる
        var steps = new double[64];
        for (int t = 0; t < 64; t++)
        {
            steps[t] = t < 21 ? 0.25 : -0.5;
        }

        Close(steps, Wavelets.Denoise(steps, DiscreteWavelet.Haar, WaveletThresholdRule.Bayes, WaveletShrinkage.Soft, 2, 0.0), 1e-15, "σ = 0");

        // 雑音を加えた滑らかな信号の誤差は小さくなる
        var random = new Random(5);
        var clean = new double[1024];
        var noisy = new double[1024];
        for (int t = 0; t < clean.Length; t++)
        {
            clean[t] = Math.Sin(2 * Math.PI * 3 * t / clean.Length) + (t > 600 ? 0.5 : 0);
            noisy[t] = clean[t] + 0.2 * Gaussian(random);
        }

        foreach (var rule in new[] { WaveletThresholdRule.Bayes, WaveletThresholdRule.Universal })
        {
            var denoised = Wavelets.Denoise(noisy, DiscreteWavelet.Symlet(8), rule, WaveletShrinkage.Soft);
            Assert.True(Rms(denoised, clean) < 0.5 * Rms(new Vector<double>(noisy), clean), $"{rule}: {Rms(denoised, clean)}");
        }
    }

    [Fact]
    public void DenoiseValidatesArguments()
    {
        var w = DiscreteWavelet.Daubechies(2);
        Assert.Throws<ArgumentException>(() => Wavelets.Denoise(new double[] { 1, double.NaN, 3 }, w));
        Assert.Throws<ArgumentException>(() => Wavelets.Denoise2D(Matrix<double>.FromRowMajor(1, 2, [1, double.PositiveInfinity]), w));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.Denoise(new double[16], w, WaveletThresholdRule.Bayes, WaveletShrinkage.Soft, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.Denoise(new double[16], w, (WaveletThresholdRule)5, WaveletShrinkage.Soft));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.Denoise(new double[16], w, WaveletThresholdRule.Bayes, WaveletShrinkage.Soft, 2, -0.1));
        Assert.Throws<ArgumentException>(() => Wavelets.Denoise(Array.Empty<double>(), w));
        Assert.Throws<ArgumentNullException>(() => Wavelets.Denoise(new double[4], null!));
    }

    private static double Gaussian(Random random) =>
        Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble());

    private static double Rms(ReadOnlyVectorSpan<double> x, double[] reference)
    {
        double sum = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            sum += (x[i] - reference[i]) * (x[i] - reference[i]);
        }

        return Math.Sqrt(sum / reference.Length);
    }
}
