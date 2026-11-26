using Complex = System.Numerics.Complex;
using Tuna.LinearAlgebra;
using Tuna.SignalProcessing;

namespace Tuna.Tests.SignalProcessing;

/// <summary>連続ウェーブレット変換: 正規化、定義式（直接の和）との照合、換算係数、逆変換、ビューと引数の検査</summary>
[Trait("Wavelets", "Continuous")]
public class ContinuousWaveletTests
{
    public static IEnumerable<ContinuousWavelet> AllWavelets()
    {
        yield return ContinuousWavelet.Morlet();
        yield return ContinuousWavelet.Morlet(10);
        foreach (int m in new[] { 1, 2, 3, 4, 6, 9 })
        {
            yield return ContinuousWavelet.Paul(m);
        }

        foreach (int m in new[] { 1, 2, 3, 4, 6, 9 })
        {
            yield return ContinuousWavelet.DerivativeOfGaussian(m);
        }
    }

    public static TheoryData<string> Names()
    {
        var data = new TheoryData<string>();
        foreach (var wavelet in AllWavelets())
        {
            data.Add(wavelet.Name);
        }

        return data;
    }

    private static ContinuousWavelet FromName(string name) => AllWavelets().Single(w => w.Name == name);

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

    // ---- 基底 ----

    /// <summary>∫f(ω) dω（台形則）。負の周波数で 0 のウェーブレットは ω = e^v と置いて、ω = 0 の角を避ける</summary>
    private static Complex SpectralIntegral(ContinuousWavelet wavelet, Func<double, Complex> f)
    {
        var sum = Complex.Zero;
        if (wavelet.FourierTransform(-1) == Complex.Zero)
        {
            const double h = 1.0 / 256;
            for (double v = -50; v <= 7; v += h)
            {
                double omega = Math.Exp(v);
                sum += f(omega) * (omega * h);
            }
        }
        else
        {
            const double h = 1.0 / 128;
            for (double omega = -60; omega <= 60; omega += h)
            {
                sum += f(omega) * h;
            }
        }

        return sum;
    }

    [Theory]
    [MemberData(nameof(Names))]
    public void WaveletsHaveUnitEnergy(string name)
    {
        var w = FromName(name);
        // 周波数領域: ∫|ψ̂₀(ω)|² dω = 1
        double spectral = SpectralIntegral(w, omega => Complex.Abs(w.FourierTransform(omega)) * Complex.Abs(w.FourierTransform(omega))).Real;
        Assert.Equal(1, spectral, 1e-12);

        // 時間領域: ∫|ψ₀(η)|² dη = 1（Paul は |η|^(−2(m+1)) で減衰するので広く取る）
        double range = name == "Paul(1)" ? 4000 : 400, temporal = 0;
        const double step = 1.0 / 64;
        for (double eta = -range; eta <= range; eta += step)
        {
            double magnitude = Complex.Abs(w.Evaluate(eta));
            temporal += magnitude * magnitude * step;
        }

        // Morlet の ψ₀ は負の周波数を切り捨てていないので、ψ̂₀ との差（ω₀ = 6 で 1e−17 程度）はここでは見えない
        Assert.Equal(1, temporal, name == "Paul(1)" ? 1e-10 : 1e-12);
    }

    [Theory]
    [MemberData(nameof(Names))]
    public void EvaluateIsTheInverseFourierTransform(string name)
    {
        // ψ₀(η) = (2π)^(−1/2)·∫ψ̂₀(ω)·e^(iωη) dω（位相の規約、Paul の i^m と奇数次の DOG の符号を確かめる）
        var w = FromName(name);
        double tolerance = name.StartsWith("Morlet", StringComparison.Ordinal) ? 1e-8 : 1e-12;
        foreach (double eta in new[] { 0.0, 0.3, -0.7, 1.5, -2.5, 4.0 })
        {
            var expected = SpectralIntegral(w, omega => w.FourierTransform(omega) * Complex.FromPolarCoordinates(1, omega * eta)) / Math.Sqrt(2 * Math.PI);
            var actual = w.Evaluate(eta);
            Assert.True(Complex.Abs(actual - expected) <= tolerance, $"{name} at η = {eta}: {actual} vs {expected}");
        }
    }

    [Fact]
    public void FactorsMatchTorrenceCompo()
    {
        // Torrence–Compo (1998) の表1・表2
        Assert.Equal(1.033, ContinuousWavelet.Morlet().FourierFactor, 1e-3);
        Assert.Equal(4 * Math.PI / 9, ContinuousWavelet.Paul().FourierFactor, 1e-15);
        Assert.Equal(2 * Math.PI / Math.Sqrt(2.5), ContinuousWavelet.MexicanHat.FourierFactor, 1e-15);
        Assert.Equal(Math.Sqrt(2), ContinuousWavelet.Morlet().EFoldingTime, 1e-15);
        Assert.Equal(1 / Math.Sqrt(2), ContinuousWavelet.Paul().EFoldingTime, 1e-15);
        Assert.Equal(Math.Sqrt(2), ContinuousWavelet.DerivativeOfGaussian(6).EFoldingTime, 1e-15);

        // ψ₀(0): Morlet π^(−1/4) = 0.751、Paul(4) 1.079、DOG(2) 0.867、DOG(6) 0.884
        Assert.Equal(0.751, ContinuousWavelet.Morlet().Evaluate(0).Real, 1e-3);
        Assert.Equal(1.079, ContinuousWavelet.Paul().Evaluate(0).Real, 1e-3);
        Assert.Equal(0.867, ContinuousWavelet.MexicanHat.Evaluate(0).Real, 1e-3);
        Assert.Equal(0.884, ContinuousWavelet.DerivativeOfGaussian(6).Evaluate(0).Real, 1e-3);
    }

    [Fact]
    public void NamesAndFactories()
    {
        Assert.Equal("Morlet(6)", ContinuousWavelet.Morlet().Name);
        Assert.Equal("Morlet(5.5)", ContinuousWavelet.Morlet(5.5).ToString());
        Assert.Equal("Paul(4)", ContinuousWavelet.Paul().Name);
        Assert.Equal("DOG(2)", ContinuousWavelet.MexicanHat.Name);
        Assert.Same(ContinuousWavelet.MexicanHat, ContinuousWavelet.DerivativeOfGaussian(2));
        Assert.Equal("DOG(7)", ContinuousWavelet.DerivativeOfGaussian(7).Name);
    }

    // ---- 変換 ----

    /// <summary>定義式 W(s, n) = (δt/s)^(1/2)·Σ_j x_j·ψ₀*((j − n)·δt/s) を、変換の長さ L で周期化した核で直接計算する</summary>
    private static Complex[,] DirectTransform(double[] x, double[] scales, ContinuousWavelet wavelet, double dt, int length, int periods)
    {
        int n = x.Length;
        var result = new Complex[scales.Length, n];
        for (int j = 0; j < scales.Length; j++)
        {
            double s = scales[j];
            var kernel = new Complex[length];
            for (int d = 0; d < length; d++)
            {
                var sum = Complex.Zero;
                for (int m = -periods; m <= periods; m++)
                {
                    sum += Complex.Conjugate(wavelet.Evaluate((d + (double)m * length) * dt / s));
                }

                kernel[d] = sum * Math.Sqrt(dt / s);
            }

            for (int t = 0; t < n; t++)
            {
                var sum = Complex.Zero;
                for (int i = 0; i < n; i++)
                {
                    sum += x[i] * kernel[((i - t) % length + length) % length];
                }

                result[j, t] = sum;
            }
        }

        return result;
    }

    [Theory]
    [MemberData(nameof(Names))]
    public void TransformMatchesTheDefinition(string name)
    {
        var w = FromName(name);
        if (name == "Paul(1)" || name == "Paul(2)")
        {
            return; // 核の周期化の和が遅く収束するので、直接の和を正確に作れない
        }

        // ψ̂₀ がナイキスト周波数で無視できる大きさになるスケール（s/δt ≥ 20）だけを使う
        const double dt = 0.5;
        double[] scales = [10, 13, 21];
        var x = RandomSignal(150, 3);
        double tolerance = name.StartsWith("Morlet", StringComparison.Ordinal) ? 1e-8 : 1e-12;
        foreach (var padding in new[] { WaveletPadding.Zero, WaveletPadding.Periodic })
        {
            int length = padding == WaveletPadding.Zero ? Fourier.NextFastLength(2 * x.Length - 1) : x.Length;
            var expected = DirectTransform(x, scales, w, dt, length, name.StartsWith("Paul", StringComparison.Ordinal) ? 2000 : 3);
            var actual = Wavelets.Continuous(x, scales, w, dt, padding);
            Assert.Equal(scales.Length, actual.Rows);
            Assert.Equal(x.Length, actual.Columns);
            double scale = 0, error = 0;
            for (int j = 0; j < scales.Length; j++)
            {
                for (int t = 0; t < x.Length; t++)
                {
                    scale = Math.Max(scale, Complex.Abs(expected[j, t]));
                    error = Math.Max(error, Complex.Abs(actual[j, t] - expected[j, t]));
                }
            }

            Assert.True(error <= tolerance * scale, $"{name} {padding}: error {error:E2}, scale {scale:E2}");
        }
    }

    [Theory]
    [MemberData(nameof(Names))]
    public void SinePeaksAtItsPeriod(string name)
    {
        // 周期 P の正弦波の時間平均のパワー |W|² は、周期 FourierFactor·s = P のスケールで最大になる（Fourier factor の定義）
        var w = FromName(name);
        const int n = 4096;
        const double dt = 0.25, period = 64 * dt;
        var x = new double[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = Math.Sin(2 * Math.PI * i * dt / period);
        }

        const double fraction = 1.0 / 128;
        var scales = Wavelets.GeometricScales(period / w.FourierFactor * 0.5, fraction, 256);
        var coefficients = Wavelets.Continuous(x, scales, w, dt, WaveletPadding.Periodic);
        int best = -1;
        double bestPower = double.NegativeInfinity;
        for (int j = 0; j < scales.Length; j++)
        {
            double power = 0;
            for (int t = 0; t < n; t++)
            {
                power += Complex.Abs(coefficients[j, t]) * Complex.Abs(coefficients[j, t]);
            }

            if (power > bestPower)
            {
                bestPower = power;
                best = j;
            }
        }

        double peakPeriod = w.FourierFactor * scales[best];
        Assert.InRange(peakPeriod / period, Math.Pow(2, -fraction), Math.Pow(2, fraction));
    }

    [Fact]
    public void RealWaveletsGiveRealCoefficients()
    {
        var x = RandomSignal(300, 8);
        var scales = Wavelets.GeometricScales(1, 0.25, 30);
        foreach (int m in new[] { 1, 2, 3, 6 })
        {
            var coefficients = Wavelets.Continuous(x, scales, ContinuousWavelet.DerivativeOfGaussian(m), 0.5);
            double real = 0, imaginary = 0;
            for (int j = 0; j < scales.Length; j++)
            {
                for (int t = 0; t < x.Length; t++)
                {
                    real = Math.Max(real, Math.Abs(coefficients[j, t].Real));
                    imaginary = Math.Max(imaginary, Math.Abs(coefficients[j, t].Imaginary));
                }
            }

            Assert.True(imaginary <= 1e-14 * real, $"DOG({m}): {imaginary:E2} vs {real:E2}");
        }
    }

    [Fact]
    public void ScalesAndConeOfInfluence()
    {
        var scales = Wavelets.GeometricScales(0.5, 0.25, 9);
        for (int j = 0; j < 9; j++)
        {
            Assert.Equal(0.5 * Math.Pow(2, j * 0.25), scales[j]);
        }

        Assert.Equal(2.0, scales[8]);
        Assert.Equal(new[] { 0.5 }, Wavelets.GeometricScales(0.5, 0.25, 1).ToArray());

        var coi = Wavelets.ConeOfInfluence(6, ContinuousWavelet.Morlet(), 2);
        double r = 2 / Math.Sqrt(2);
        Assert.Equal(new[] { 0, r, 2 * r, 2 * r, r, 0 }, coi.ToArray());
        Assert.Equal(new[] { 0.0 }, Wavelets.ConeOfInfluence(1, ContinuousWavelet.Paul(), 1).ToArray());
        Assert.Equal(new[] { 0, Math.Sqrt(2), 0 }, Wavelets.ConeOfInfluence(3, ContinuousWavelet.Paul(), 1).ToArray(), new ToleranceComparer(1e-15));
    }

    private sealed class ToleranceComparer(double tolerance) : IEqualityComparer<double>
    {
        public bool Equals(double x, double y) => Math.Abs(x - y) <= tolerance;
        public int GetHashCode(double obj) => 0;
    }

    // ---- 逆変換 ----

    private static double RelativeError(Vector<double> actual, double[] expected)
    {
        double error = 0, norm = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            error += (actual[i] - expected[i]) * (actual[i] - expected[i]);
            norm += expected[i] * expected[i];
        }

        return Math.Sqrt(error / norm);
    }

    [Theory]
    [InlineData("Morlet(6)", 1e-9)]
    [InlineData("Paul(4)", 1e-6)]
    [InlineData("Paul(3)", 1e-5)]
    [InlineData("DOG(2)", 1e-5)]
    [InlineData("DOG(6)", 1e-9)]
    public void InverseRecoversASinusoidAtTheMiddleOfTheScales(string name, double bound)
    {
        // スケールの範囲の中央の周期の余弦波は（スケールの離散化の波打ちを除いて）ちょうど戻る。端の補い方にはよらない
        var w = FromName(name);
        const int n = 512;
        const double dt = 0.5;
        var scales = Wavelets.GeometricScales(2 * dt / w.FourierFactor, 0.125, 81); // 周期 2δt〜2^11·δt
        var x = new double[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = Math.Cos(2 * Math.PI * i / 64.0 + 1);
        }

        foreach (var padding in new[] { WaveletPadding.Periodic, WaveletPadding.Zero })
        {
            var back = Wavelets.InverseContinuous(Wavelets.Continuous(x, scales, w, dt, padding), scales, w, dt);
            if (padding == WaveletPadding.Periodic)
            {
                Assert.True(RelativeError(back, x) <= bound, $"{name}: relative error {RelativeError(back, x):E2}");
            }
            else
            {
                // 0 で埋めると端の近くは戻らないので、中央だけを比べる
                double error = 0;
                for (int i = 192; i < 320; i++)
                {
                    error = Math.Max(error, Math.Abs(back[i] - x[i]));
                }

                Assert.True(error <= 1e-3, $"{name} (zero padding): error {error:E2}");
            }
        }
    }

    [Theory]
    [InlineData("Morlet(6)", 1e-4)]
    [InlineData("Paul(4)", 0.003)]
    [InlineData("DOG(2)", 0.02)]
    [InlineData("DOG(6)", 1e-4)]
    public void InverseReconstructsABroadbandSignal(string name, double bound)
    {
        // スケールの範囲の内側の周期だけを含む信号は近似的に戻る（Torrence–Compo の第3節 i）。範囲の端に近い周期ほど小さくなる
        var w = FromName(name);
        const int n = 1024;
        var x = new double[n];
        foreach (int cycles in new[] { 64, 50, 20, 8 }) // 周期 16〜128
        {
            for (int i = 0; i < n; i++)
            {
                x[i] += Math.Cos(2 * Math.PI * cycles * i / n + cycles);
            }
        }

        var scales = Wavelets.GeometricScales(2 / w.FourierFactor, 0.125, 90);
        var back = Wavelets.InverseContinuous(Wavelets.Continuous(x, scales, w, 1, WaveletPadding.Periodic), scales, w, 1);
        Assert.True(RelativeError(back, x) <= bound, $"{name}: relative error {RelativeError(back, x):F4}");
    }

    [Fact]
    public void UnevenScalesAreWeightedByTheirLogarithmicSpacing()
    {
        // 等比でないスケールでも、範囲の中央の周期は近似的に戻る
        var w = ContinuousWavelet.Paul();
        var scales = new List<double>();
        for (double s = 1; s < 400; s *= scales.Count % 2 == 0 ? 1.06 : 1.12)
        {
            scales.Add(s);
        }

        double middle = w.FourierFactor * Math.Sqrt(scales[0] * scales[^1]);
        var x = new double[2048];
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = Math.Sin(2 * Math.PI * i / middle);
        }

        var back = Wavelets.InverseContinuous(Wavelets.Continuous(x, scales.ToArray(), w, 1), scales.ToArray(), w, 1);
        double error = 0;
        for (int i = 800; i < 1248; i++)
        {
            error = Math.Max(error, Math.Abs(back[i] - x[i]));
        }

        Assert.True(error <= 2e-4, $"error {error:E2}");
    }

    // ---- ビューと出力先 ----

    private static Complex[] Flatten(ReadOnlyMatrixSpan<Complex> m)
    {
        var result = new Complex[m.Rows * m.Columns];
        for (int j = 0; j < m.Columns; j++)
        {
            for (int i = 0; i < m.Rows; i++)
            {
                result[j * m.Rows + i] = m[i, j];
            }
        }

        return result;
    }

    [Fact]
    public void StridedViewsAndDestinationsMatchDenseResultsBitwise()
    {
        var w = ContinuousWavelet.Paul();
        var m = Matrix<double>.FromColumnMajor(3, 40, RandomSignal(120, 5));
        var row = m.Row(2);
        var dense = row.ToVector();
        var scaleMatrix = Matrix<double>.FromColumnMajor(2, 12, Wavelets.GeometricScales(0.5, 0.5, 24).ToArray());
        var scaleRow = scaleMatrix.Row(1);
        var denseScales = scaleRow.ToVector();

        var expected = Wavelets.Continuous(dense, denseScales, w, 0.25);
        var actual = Wavelets.Continuous(row, scaleRow, w, 0.25);
        Assert.Equal(Flatten(expected), Flatten(actual));

        // 出力先は大きな行列の一部（行が飛び飛び）
        var big = new Matrix<Complex>(20, 50);
        var target = big[3..15, 5..45];
        var returned = Wavelets.Continuous(row, scaleRow, w, 0.25, WaveletPadding.Zero, target);
        Assert.Equal(Flatten(expected), Flatten(returned));
        Assert.Equal(Complex.Zero, big[2, 5]);

        // 逆変換: 転置のビュー（列が飛び飛び）からも同じ結果
        var back = Wavelets.InverseContinuous(expected, denseScales, w, 0.25);
        var transposed = expected.T.ToMatrix();
        var fromView = Wavelets.InverseContinuous(transposed.T, scaleRow, w, 0.25);
        Assert.Equal(back.ToArray(), fromView.ToArray());
        var output = new Vector<double>(45);
        var written = Wavelets.InverseContinuous(target, scaleRow, w, 0.25, output[2..42]);
        Assert.Equal(back.ToArray(), written.ToArray());
        Assert.Equal(0, output[1]);
    }

    [Fact]
    public void InvalidArgumentsThrow()
    {
        var w = ContinuousWavelet.Morlet();
        double[] x = [1, 2, 3, 4];
        double[] scales = [1, 2];
        Assert.Throws<ArgumentNullException>(() => Wavelets.Continuous(x, scales, null!, 1));
        Assert.Throws<ArgumentException>(() => Wavelets.Continuous(ReadOnlySpan<double>.Empty, scales, w, 1));
        Assert.Throws<ArgumentException>(() => Wavelets.Continuous(x, ReadOnlySpan<double>.Empty, w, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.Continuous(x, new double[] { 1, 0 }, w, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.Continuous(x, new double[] { double.NaN }, w, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.Continuous(x, new double[] { double.PositiveInfinity }, w, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.Continuous(x, scales, w, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.Continuous(x, scales, w, double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.Continuous(x, scales, w, 1, (WaveletPadding)5));
        Assert.Throws<ArgumentException>(() => Wavelets.Continuous(x, scales, w, 1, WaveletPadding.Zero, new Matrix<Complex>(2, 3)));

        var c = Wavelets.Continuous(x, scales, w, 1);
        Assert.Throws<ArgumentException>(() => Wavelets.InverseContinuous(c[..1, ..], new double[] { 1 }, w, 1));
        Assert.Throws<ArgumentException>(() => Wavelets.InverseContinuous(c, new double[] { 2, 1 }, w, 1));
        Assert.Throws<ArgumentException>(() => Wavelets.InverseContinuous(c, new double[] { 1, 2, 3 }, w, 1));
        Assert.Throws<ArgumentException>(() => Wavelets.InverseContinuous(c, scales, ContinuousWavelet.DerivativeOfGaussian(3), 1));
        Assert.Throws<ArgumentException>(() => Wavelets.InverseContinuous(c, scales, w, 1, new double[3]));

        Assert.Throws<ArgumentOutOfRangeException>(() => ContinuousWavelet.Morlet(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => ContinuousWavelet.Morlet(double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => ContinuousWavelet.Paul(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => ContinuousWavelet.Paul(101));
        Assert.Throws<ArgumentOutOfRangeException>(() => ContinuousWavelet.DerivativeOfGaussian(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => ContinuousWavelet.DerivativeOfGaussian(101));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.GeometricScales(0, 0.25, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.GeometricScales(1, -0.25, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.GeometricScales(1, 0.25, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.GeometricScales(1, 100, 20));
        Assert.Throws<ArgumentOutOfRangeException>(() => Wavelets.ConeOfInfluence(0, w, 1));
        Assert.Throws<ArgumentNullException>(() => Wavelets.ConeOfInfluence(4, null!, 1));
    }

    [Fact]
    public void HighOrdersStayFinite()
    {
        // 階乗やガンマ関数があふれないこと
        foreach (var w in new[] { ContinuousWavelet.Paul(100), ContinuousWavelet.DerivativeOfGaussian(100) })
        {
            foreach (double v in new[] { -800.0, -30, -1, 0, 0.5, 10, 100, 700, 1e6 })
            {
                Assert.True(double.IsFinite(w.FourierTransform(v).Real) && double.IsFinite(w.FourierTransform(v).Imaginary), $"{w.Name} ψ̂({v})");
                Assert.True(double.IsFinite(w.Evaluate(v).Real) && double.IsFinite(w.Evaluate(v).Imaginary), $"{w.Name} ψ({v})");
            }
        }
    }
}
