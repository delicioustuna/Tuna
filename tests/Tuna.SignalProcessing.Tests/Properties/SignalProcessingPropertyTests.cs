using CsCheck;
using Tuna.LinearAlgebra;
using Tuna.SignalProcessing;
using Complex = System.Numerics.Complex;

namespace Tuna.SignalProcessing.Tests.Properties;

[Trait("Category", "Property")]
public class SignalProcessingPropertyTests
{
    private static readonly FourierNormalization[] Normalizations =
    [
        FourierNormalization.Backward,
        FourierNormalization.Forward,
        FourierNormalization.Orthonormal
    ];

    private static readonly ConvolutionMode[] ConvolutionModes =
    [
        ConvolutionMode.Full,
        ConvolutionMode.Same,
        ConvolutionMode.Valid
    ];

    private static readonly DiscreteWavelet[] TestWavelets =
    [
        DiscreteWavelet.Haar,
        DiscreteWavelet.Daubechies(2),
        DiscreteWavelet.Daubechies(4),
        DiscreteWavelet.Symlet(4),
        DiscreteWavelet.Coiflet(2),
        DiscreteWavelet.Biorthogonal(2, 2)
    ];

    private static readonly WaveletBoundary[] WaveletBoundaries =
    [
        WaveletBoundary.Symmetric,
        WaveletBoundary.Periodic,
        WaveletBoundary.Zero
    ];

    [Fact]
    public void ComplexFft_RoundTripParsevalLinearityAndDestinationOverload()
    {
        var gen = Gen.Select(Gen.Int[1, 140], Gen.Int[0, 2], Gen.Int[0, 100000]);

        gen.Sample((length, normIdx, dataSeed) =>
        {
            var norm = Normalizations[normIdx];
            var rng = new Random(dataSeed);

            var x = new Vector<Complex>(length);
            var y = new Vector<Complex>(length);
            for (int i = 0; i < length; i++)
            {
                x[i] = new Complex(2.0 * rng.NextDouble() - 1.0, 2.0 * rng.NextDouble() - 1.0);
                y[i] = new Complex(2.0 * rng.NextDouble() - 1.0, 2.0 * rng.NextDouble() - 1.0);
            }

            // 1. 往復不変条件: Inverse(Forward(x)) ≈ x
            var freq = Fourier.Forward(x, norm);
            var roundTrip = Fourier.Inverse(freq, norm);
            double tol = 1e-10 * (length + 1);
            for (int i = 0; i < length; i++)
            {
                Assert.True((roundTrip[i] - x[i]).Magnitude <= tol, $"FFT round-trip mismatch at {i} (n={length}, norm={norm})");
            }

            // 2. destination オーバーロードの一致
            var freqDst = new Vector<Complex>(length);
            Fourier.Forward(x, norm, freqDst);
            for (int i = 0; i < length; i++)
            {
                Assert.True((freqDst[i] - freq[i]).Magnitude <= 1e-13);
            }

            // 3. Parseval の等式（Orthonormal 正規化ではエネルギーが厳密保存される）
            if (norm == FourierNormalization.Orthonormal)
            {
                double energyTime = 0.0;
                double energyFreq = 0.0;
                for (int i = 0; i < length; i++)
                {
                    energyTime += x[i].Real * x[i].Real + x[i].Imaginary * x[i].Imaginary;
                    energyFreq += freq[i].Real * freq[i].Real + freq[i].Imaginary * freq[i].Imaginary;
                }

                Assert.True(Math.Abs(energyTime - energyFreq) <= 1e-10 * (energyTime + 1.0));
            }

            // 4. 線形性: FFT(a*x + b*y) ≈ a*FFT(x) + b*FFT(y)
            var a = new Complex(2.0 * rng.NextDouble() - 1.0, 2.0 * rng.NextDouble() - 1.0);
            var b = new Complex(2.0 * rng.NextDouble() - 1.0, 2.0 * rng.NextDouble() - 1.0);
            var combined = new Vector<Complex>(length);
            for (int i = 0; i < length; i++)
            {
                combined[i] = a * x[i] + b * y[i];
            }

            var freqCombined = Fourier.Forward(combined, norm);
            var freqY = Fourier.Forward(y, norm);
            for (int i = 0; i < length; i++)
            {
                var expected = a * freq[i] + b * freqY[i];
                Assert.True((freqCombined[i] - expected).Magnitude <= tol * 2.0);
            }
        }, iter: 150, seed: "a1b2c3d4e5f6");
    }

    [Fact]
    public void RealFft_RoundTripAndMatchesComplexFftOneSidedSpectrum()
    {
        var gen = Gen.Select(Gen.Int[1, 140], Gen.Int[0, 2], Gen.Int[0, 100000]);

        gen.Sample((length, normIdx, dataSeed) =>
        {
            var norm = Normalizations[normIdx];
            var rng = new Random(dataSeed);
            var xReal = new Vector<double>(length);
            var xComplex = new Vector<Complex>(length);
            for (int i = 0; i < length; i++)
            {
                double v = 2.0 * rng.NextDouble() - 1.0;
                xReal[i] = v;
                xComplex[i] = new Complex(v, 0.0);
            }

            var realSpectrum = Fourier.ForwardReal(xReal, norm);
            var complexSpectrum = Fourier.Forward(xComplex, norm);
            int expectedBins = length / 2 + 1;
            Assert.Equal(expectedBins, realSpectrum.Length);

            double tol = 1e-10 * (length + 1);
            for (int i = 0; i < expectedBins; i++)
            {
                Assert.True((realSpectrum[i] - complexSpectrum[i]).Magnitude <= tol);
            }

            var reconstructed = Fourier.InverseReal(realSpectrum, length, norm);
            Assert.Equal(length, reconstructed.Length);
            for (int i = 0; i < length; i++)
            {
                Assert.True(Math.Abs(reconstructed[i] - xReal[i]) <= tol);
            }
        }, iter: 150, seed: "b2c3d4e5f6a1");
    }

    [Fact]
    public void Convolve_FftMatchesDirectAndIsCommutative()
    {
        var gen = Gen.Select(Gen.Int[1, 65], Gen.Int[1, 65], Gen.Int[0, 2], Gen.Int[0, 100000]);

        gen.Sample((lenX, lenY, modeIdx, dataSeed) =>
        {
            var mode = ConvolutionModes[modeIdx];
            var rng = new Random(dataSeed);
            var x = new Vector<double>(lenX);
            var y = new Vector<double>(lenY);
            for (int i = 0; i < lenX; i++)
            {
                x[i] = 2.0 * rng.NextDouble() - 1.0;
            }

            for (int i = 0; i < lenY; i++)
            {
                y[i] = 2.0 * rng.NextDouble() - 1.0;
            }

            // Valid モードは長い方の入力を第 1 引数にする必要がある場合があるため、長さに応じて順序を揃える
            if (mode == ConvolutionMode.Valid && lenX < lenY)
            {
                (x, y) = (y, x);
            }

            var direct = XMath.Convolve<double>(x, y, mode, ConvolutionMethod.Direct);
            var fft = XMath.Convolve<double>(x, y, mode, ConvolutionMethod.Fourier);
            Assert.Equal(direct.Length, fft.Length);

            double tol = 1e-10 * (x.Length + y.Length);
            for (int i = 0; i < direct.Length; i++)
            {
                Assert.True(Math.Abs(direct[i] - fft[i]) <= tol, $"Convolve Direct vs Fft mismatch at {i} (mode={mode})");
            }

            // Full モードでは畳み込みの可換性 convolve(x, y) == convolve(y, x) が成り立つ
            if (mode == ConvolutionMode.Full)
            {
                var swapped = XMath.Convolve<double>(y, x, mode, ConvolutionMethod.Fourier);
                Assert.Equal(direct.Length, swapped.Length);
                for (int i = 0; i < direct.Length; i++)
                {
                    Assert.True(Math.Abs(direct[i] - swapped[i]) <= tol);
                }
            }
        }, iter: 150, seed: "c3d4e5f6a1b2");
    }

    [Fact]
    public void Wavelets_MultiLevelDecomposeAndReconstruct_PerfectReconstruction()
    {
        var gen = Gen.Select(
            Gen.Int[4, 96],
            Gen.Int[0, TestWavelets.Length - 1],
            Gen.Int[0, WaveletBoundaries.Length - 1],
            Gen.Int[1, 4],
            Gen.Int[0, 100000]);

        gen.Sample((length, waveletIdx, boundaryIdx, requestedLevels, dataSeed) =>
        {
            var wavelet = TestWavelets[waveletIdx];
            var boundary = WaveletBoundaries[boundaryIdx];
            int maxLevel = Wavelets.MaxLevel(length, wavelet);
            if (maxLevel < 1)
            {
                return;
            }

            int levels = Math.Min(requestedLevels, maxLevel);
            var rng = new Random(dataSeed);
            var x = new Vector<double>(length);
            for (int i = 0; i < length; i++)
            {
                x[i] = 2.0 * rng.NextDouble() - 1.0;
            }

            var decomp = Wavelets.Decompose(x, wavelet, levels, boundary);
            var reconstructed = Wavelets.Reconstruct(decomp);

            Assert.Equal(length, reconstructed.Length);
            for (int i = 0; i < length; i++)
            {
                Assert.True(
                    Math.Abs(reconstructed[i] - x[i]) <= 1e-9,
                    $"Wavelet reconstruction mismatch at {i} (n={length}, wavelet={wavelet.Name}, boundary={boundary}, levels={levels})");
            }
        }, iter: 150, seed: "d4e5f6a1b2c3");
    }
}
