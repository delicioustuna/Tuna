using Tuna.LinearAlgebra;
using Tuna.SignalProcessing;
using Complex = System.Numerics.Complex;
using static Tuna.Tests.SignalProcessing.FourierReference;

namespace Tuna.Tests.SignalProcessing;

/// <summary>chirp-z 変換とズーム FFT: SciPy との照合、二重倍精度の直接和との誤差、DFT との一致、例外</summary>
[Trait("Fourier", "ChirpZ")]
public class ChirpZTests
{
    private static void AssertClose(ReadOnlySpan<Complex> expected, ReadOnlyVectorSpan<Complex> actual, double relative, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = 0;
        foreach (var value in expected)
        {
            scale = Math.Max(scale, value.Magnitude);
        }

        for (int k = 0; k < expected.Length; k++)
        {
            double error = (actual[k] - expected[k]).Magnitude;
            Assert.True(error <= relative * Math.Max(scale, 1), $"{label} [{k}]: {actual[k]} vs {expected[k]} (error {error:E2})");
        }
    }

    private static void AssertBitwiseEqual(ReadOnlyVectorSpan<Complex> expected, ReadOnlyVectorSpan<Complex> actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int k = 0; k < expected.Length; k++)
        {
            if (BitConverter.DoubleToInt64Bits(expected[k].Real) != BitConverter.DoubleToInt64Bits(actual[k].Real)
                || BitConverter.DoubleToInt64Bits(expected[k].Imaginary) != BitConverter.DoubleToInt64Bits(actual[k].Imaginary))
            {
                Assert.Fail($"{label}[{k}] expected {expected[k]}, actual {actual[k]}");
            }
        }
    }

    // ---- SciPy との照合 ----

    [Fact]
    public void ChirpZMatchesSciPy()
    {
        foreach (var (x, count, ratio, start, expected) in SciPyReferences.ChirpZ)
        {
            var actual = Fourier.ChirpZ(x, count, ratio, start);
            AssertClose(expected, actual, 1e-12, $"czt n={x.Length} m={count}");
        }
    }

    [Fact]
    public void ZoomMatchesSciPy()
    {
        foreach (var (x, first, step, count, spacing, expected) in SciPyReferences.Zoom)
        {
            var actual = Fourier.Zoom(x, first, step, count, spacing);
            AssertClose(expected, actual, 1e-12, $"zoom_fft n={x.Length} m={count}");
        }
    }

    // ---- DFT との一致 ----

    [Theory]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(64)]
    [InlineData(1024)]
    public void ZoomOverTheWholeCircleIsTheDft(int n)
    {
        // f₀ = 0、Δf = 1/n（2 の累乗なので正確）、n 点は DFT そのもの
        var x = RandomComplex(n, n);
        var expected = Fourier.Forward(x);
        AssertClose(expected.AsSpan(), Fourier.Zoom(x, 0, 1.0 / n, n), 8 * (1 + Math.Log2(n)) * Epsilon, $"n={n}");
    }

    [Theory]
    [InlineData(5)]
    [InlineData(64)]
    [InlineData(100)]
    public void ChirpZOnTheUnitCircleIsTheDft(int n)
    {
        var x = RandomComplex(n, 3 * n);
        var w = Complex.FromPolarCoordinates(1, -2 * Math.PI / n);
        AssertClose(Fourier.Forward(x).AsSpan(), Fourier.ChirpZ(x, n, w, Complex.One), 1e-12, $"n={n}");
    }

    // ---- 二重倍精度の直接和との誤差 ----

    /// <summary>θ（[−π, π]）の sin と cos（二重倍精度、テイラー級数）</summary>
    private static (DoubleDouble Sin, DoubleDouble Cos) SinCos(DoubleDouble theta)
    {
        var x2 = theta * theta;
        DoubleDouble sin = theta, cos = 1.0, sinTerm = theta, cosTerm = 1.0;
        for (int i = 1; i < 60; i++)
        {
            sinTerm = -(sinTerm * x2 / (2.0 * i) / (2.0 * i + 1));
            cosTerm = -(cosTerm * x2 / (2.0 * i - 1) / (2.0 * i));
            sin += sinTerm;
            cos += cosTerm;
            if (Math.Abs(sinTerm.Hi) < 1e-40 && Math.Abs(cosTerm.Hi) < 1e-40)
            {
                break;
            }
        }

        return (sin, cos);
    }

    /// <summary>a·b を誤差なしの二重倍精度で</summary>
    private static DoubleDouble Product(double a, double b)
    {
        double p = a * b;
        return new DoubleDouble(p, Math.FusedMultiplyAdd(a, b, -p));
    }

    /// <summary>
    /// X_k = Σ_j x_j·exp(−2πi·(f₀d + k·Δfd)·j) を二重倍精度で直接計算する（f₀d と Δfd は double の値そのもの）。
    /// 各 k で1標本あたりの回転 r = exp(−2πi·φ_k) を二重倍精度で作り、r^j を掛け算で進める（誤差は j·1e−32 程度）
    /// </summary>
    private static (DoubleDouble Re, DoubleDouble Im)[] DirectZoom(Complex[] x, double start, double step, int count)
    {
        var result = new (DoubleDouble Re, DoubleDouble Im)[count];
        for (int k = 0; k < count; k++)
        {
            var turns = new DoubleDouble(start) + Product(step, k);
            turns -= Math.Round(turns.Hi);
            var (sin, cos) = SinCos(DoubleDouble.Pi * (turns * 2.0));
            DoubleDouble rr = cos, ri = -sin;
            DoubleDouble pr = 1.0, pi = 0.0, sr = 0.0, si = 0.0;
            for (int j = 0; j < x.Length; j++)
            {
                DoubleDouble xr = x[j].Real, xi = x[j].Imaginary;
                sr += xr * pr - xi * pi;
                si += xr * pi + xi * pr;
                (pr, pi) = (pr * rr - pi * ri, pr * ri + pi * rr);
            }

            result[k] = (sr, si);
        }

        return result;
    }

    [Theory]
    [InlineData(64, 50, 0.1234, 0.0001)]
    [InlineData(1000, 300, 0.3, -0.00037)]
    [InlineData(2048, 600, -0.4, 0.00123)]
    [InlineData(4096, 64, 0.0123456789, 0.25 / 4096)]
    public void ZoomAccuracyAgainstDoubleDoubleSum(int n, int count, double start, double step)
    {
        var x = RandomComplex(n, n + count);
        var expected = DirectZoom(x, start, step, count);
        var actual = Fourier.Zoom(x, start, step, count);
        double error = RelativeRmsError(actual.AsSpan(), expected);
        double bound = 4 * (1 + Math.Log2(Fourier.NextFastLength(n + count - 1)));
        Assert.True(error <= bound, $"n={n} m={count}: {error:F2} ε > {bound:F2} ε");
    }

    // ---- 実数版・出力先・周期性 ----

    [Fact]
    public void RealInputMatchesComplexInputBitwise()
    {
        var real = RandomReal(77, 5);
        var complex = real.Select(v => new Complex(v, 0)).ToArray();
        AssertBitwiseEqual(Fourier.Zoom(complex, 0.05, 0.001, 40), Fourier.Zoom(real, 0.05, 0.001, 40), "real");
    }

    [Fact]
    public void SampleSpacingScalesTheFrequencies()
    {
        // f·d が正確に同じ値になる組（d = 0.5）なら、ビット単位で一致する
        var x = RandomReal(50, 6);
        AssertBitwiseEqual(Fourier.Zoom(x, 0.1, 0.002, 30, 1), Fourier.Zoom(x, 0.2, 0.004, 30, 0.5), "spacing");
    }

    [Fact]
    public void FrequenciesArePeriodicInTheSampleRate()
    {
        var x = RandomComplex(40, 7);
        var baseband = Fourier.Zoom(x, 0.125, 0.0078125, 20);
        AssertClose(baseband.AsSpan(), Fourier.Zoom(x, 3.125, 0.0078125, 20), 1e-14, "f + 3");
        AssertClose(baseband.AsSpan(), Fourier.Zoom(x, -0.875, 0.0078125, 20), 1e-14, "f − 1");
    }

    [Fact]
    public void DestinationMayAliasTheInput()
    {
        var x = new Vector<Complex>(RandomComplex(32, 8));
        var expected = Fourier.Zoom(x, 0.1, 0.01, 32);
        var expectedChirp = Fourier.ChirpZ(x, 32, new Complex(0.99, -0.05), new Complex(1, 0.1));
        var alias = x.AsVectorSpan().ToVector();
        Fourier.Zoom(alias, 0.1, 0.01, 1, alias);
        AssertBitwiseEqual(expected, alias, "zoom");
        var aliasChirp = x.AsVectorSpan().ToVector();
        Fourier.ChirpZ(aliasChirp, new Complex(0.99, -0.05), new Complex(1, 0.1), aliasChirp);
        AssertBitwiseEqual(expectedChirp, aliasChirp, "chirp-z");
    }

    // ---- 空の入力と例外 ----

    [Fact]
    public void EmptyInputsAndOutputs()
    {
        Assert.Equal(0, Fourier.Zoom(new double[] { 1, 2 }, 0, 0.1, 0).Length);
        var zeros = Fourier.Zoom(Array.Empty<double>(), 0, 0.1, 3);
        Assert.All(zeros.AsSpan().ToArray(), value => Assert.Equal(Complex.Zero, value));
        Assert.Equal(0, Fourier.ChirpZ(new Complex[] { 1 }, 0, Complex.One, Complex.One).Length);
        Assert.All(Fourier.ChirpZ(Array.Empty<Complex>(), 2, Complex.One, Complex.One).AsSpan().ToArray(), value => Assert.Equal(Complex.Zero, value));

        // 長さ 1 の入力は x_0（丸め誤差の範囲で）
        var single = Fourier.Zoom(new double[] { 2.5 }, 0.3, 0.1, 4);
        Assert.All(single.AsSpan().ToArray(), value => Assert.True((value - 2.5).Magnitude < 4 * Epsilon * 2.5, value.ToString()));
    }

    [Fact]
    public void RejectsInvalidArguments()
    {
        var x = new double[] { 1, 2, 3 };
        Assert.Throws<ArgumentOutOfRangeException>(() => Fourier.Zoom(x, double.NaN, 0.1, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fourier.Zoom(x, 0, double.PositiveInfinity, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fourier.Zoom(x, 0, 0.1, 4, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fourier.Zoom(x, 0, 0.1, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fourier.Zoom(x, 1e300, 0.1, 4, 1e10));
        var c = new Complex[] { 1, 2 };
        Assert.Throws<ArgumentOutOfRangeException>(() => Fourier.ChirpZ(c, 3, Complex.Zero, Complex.One));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fourier.ChirpZ(c, 3, Complex.One, new Complex(double.NaN, 0)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fourier.ChirpZ(c, -1, Complex.One, Complex.One));
    }
}
