using Tuna.LinearAlgebra;
using Tuna.SignalProcessing;
using Complex = System.Numerics.Complex;
using static Tuna.Tests.SignalProcessing.FourierReference;

namespace Tuna.Tests.SignalProcessing;

// FFT の誤差: 二重倍精度の直接 DFT と比べた相対 RMS 誤差 ‖y − y_ref‖₂ / ‖y_ref‖₂。
// 判定上限 2·(1 + log₂ n)·ε で実装誤りや回転因子生成の精度劣化を検出する。
[Trait("Fourier", "Accuracy")]
public class FourierAccuracyTests
{
    private static double Bound(int n) => 2 * (1 + Math.Log2(Math.Max(n, 1)));

    /// <summary>1〜256 の全長（基数 2・3・4・5、一般の奇数基数 7〜31、Bluestein 法の全経路を通る）</summary>
    [Fact]
    public void ComplexTransformsMatchDirectDftForAllLengthsUpTo256()
    {
        var failures = new List<string>();
        for (int n = 1; n <= 256; n++)
        {
            CheckComplex(n, failures);
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Theory]
    [InlineData(289)]   // 17²（一般の奇数基数を2段）
    [InlineData(343)]   // 7³
    [InlineData(512)]
    [InlineData(625)]   // 5⁴
    [InlineData(729)]   // 3⁶
    [InlineData(961)]   // 31²（直接扱う最大の素数）
    [InlineData(997)]   // 素数（Bluestein 法）
    [InlineData(1000)]
    [InlineData(1024)]
    [InlineData(1031)]  // 素数
    [InlineData(2048)]
    [InlineData(2187)]  // 3⁷
    [InlineData(2310)]  // 2·3·5·7·11
    [InlineData(4096)]
    [InlineData(4099)]  // 素数
    public void ComplexTransformsMatchDirectDftForSelectedLengths(int n)
    {
        var failures = new List<string>();
        CheckComplex(n, failures);
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Fact]
    public void RealTransformsMatchDirectDftForAllLengthsUpTo256()
    {
        var failures = new List<string>();
        for (int n = 1; n <= 256; n++)
        {
            CheckReal(n, failures);
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Theory]
    [InlineData(1000)]
    [InlineData(1024)]
    [InlineData(1031)]
    [InlineData(2062)]  // 2 × 素数（半分の長さが Bluestein 法）
    [InlineData(4096)]
    public void RealTransformsMatchDirectDftForSelectedLengths(int n)
    {
        var failures = new List<string>();
        CheckReal(n, failures);
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    /// <summary>大きな長さは往復の誤差とパーセバルの等式で調べる（直接 DFT は O(n²) で遅い）</summary>
    [Theory]
    [InlineData(1 << 16)]
    [InlineData(1 << 20)]
    [InlineData(531441)]   // 3¹²
    [InlineData(390625)]   // 5⁸
    [InlineData(65537)]    // 素数
    [InlineData(131074)]   // 2 × 65537
    [InlineData(1000003)]  // 素数
    public void LargeLengthsRoundTripAndSatisfyParseval(int n)
    {
        var x = RandomComplex(n, n);
        var spectrum = Fourier.Forward(new ReadOnlyVectorSpan<Complex>(x));
        var back = Fourier.Inverse(spectrum);

        double error = 0, norm = 0, energy = 0;
        for (int i = 0; i < n; i++)
        {
            error += Complex.Abs(back[i] - x[i]) * Complex.Abs(back[i] - x[i]);
            norm += x[i].Real * x[i].Real + x[i].Imaginary * x[i].Imaginary;
            energy += spectrum[i].Real * spectrum[i].Real + spectrum[i].Imaginary * spectrum[i].Imaginary;
        }

        double roundTrip = Math.Sqrt(error / norm) / Epsilon;
        Assert.True(roundTrip <= 2 * Bound(n), $"n = {n}: round trip {roundTrip:F2} ε");
        Assert.True(Math.Abs(energy / n / norm - 1) <= Bound(n) * Epsilon * 4, $"n = {n}: Parseval {energy / n / norm - 1:E3}");

        var real = RandomReal(n, n + 1);
        var realBack = Fourier.InverseReal(Fourier.ForwardReal(new ReadOnlyVectorSpan<double>(real)), n);
        double realError = 0, realNorm = 0;
        for (int i = 0; i < n; i++)
        {
            realError += (realBack[i] - real[i]) * (realBack[i] - real[i]);
            realNorm += real[i] * real[i];
        }

        double realRoundTrip = Math.Sqrt(realError / realNorm) / Epsilon;
        Assert.True(realRoundTrip <= 2 * Bound(n), $"n = {n}: real round trip {realRoundTrip:F2} ε");
    }

    private static void CheckComplex(int n, List<string> failures)
    {
        var x = RandomComplex(n, n);
        var forward = Fourier.Forward(new ReadOnlyVectorSpan<Complex>(x));
        double e = RelativeRmsError(forward.AsSpan(), Dft(x, inverse: false));
        if (!(e <= Bound(n)))
        {
            failures.Add($"forward n = {n}: {e:F2} ε");
        }

        var inverse = Fourier.Inverse(new ReadOnlyVectorSpan<Complex>(x));
        double ei = RelativeRmsError(inverse.AsSpan(), Dft(x, inverse: true), 1.0 / n);
        if (!(ei <= Bound(n)))
        {
            failures.Add($"inverse n = {n}: {ei:F2} ε");
        }
    }

    private static void CheckReal(int n, List<string> failures)
    {
        var x = RandomReal(n, n);
        var spectrum = Fourier.ForwardReal(new ReadOnlyVectorSpan<double>(x));
        var reference = Dft(x.Select(v => new Complex(v, 0)).ToArray(), inverse: false);
        double e = RelativeRmsError(spectrum.AsSpan(), reference.AsSpan(0, n / 2 + 1));
        if (!(e <= Bound(n)))
        {
            failures.Add($"forward real n = {n}: {e:F2} ε");
        }

        // 逆変換: エルミート対称に広げたスペクトルの直接 DFT（虚部は 0 になる）と比べる
        var half = RandomComplex(n / 2 + 1, n + 7);
        half[0] = new Complex(half[0].Real, 0);
        if (n % 2 == 0)
        {
            half[n / 2] = new Complex(half[n / 2].Real, 0);
        }

        var full = new Complex[n];
        for (int k = 0; k < n; k++)
        {
            full[k] = k <= n / 2 ? half[k] : Complex.Conjugate(half[n - k]);
        }

        var signal = Fourier.InverseReal(new ReadOnlyVectorSpan<Complex>(half), n);
        var expected = Dft(full, inverse: true);
        double ei = RelativeRmsError(signal.AsSpan().ToArray().Select(v => new Complex(v, 0)).ToArray(), expected, 1.0 / n);
        if (!(ei <= Bound(n)))
        {
            failures.Add($"inverse real n = {n}: {ei:F2} ε");
        }
    }
}
