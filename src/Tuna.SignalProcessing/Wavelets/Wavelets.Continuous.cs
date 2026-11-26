using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Tuna.SignalProcessing;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

public static partial class Wavelets
{
    /// <summary>
    /// Computes the continuous wavelet transform (CWT) of a signal at the specified scales using zero padding.
    /// <para>ゼロパディングを用いて、指定したスケールにおける入力信号の連続ウェーブレット変換（CWT）を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal of length at least 1. / 長さ 1 以上の入力信号。</param>
    /// <param name="scales">The positive finite scales corresponding to each output row. / 各出力行に対応する正の有限スケール。</param>
    /// <param name="wavelet">The mother wavelet to use. / 使用するマザーウェーブレット。</param>
    /// <param name="sampleSpacing">The positive finite sample spacing delta-t. / 正の有限の標本間隔 δt。</param>
    /// <returns>A newly allocated <c>scales.Length x x.Length</c> complex matrix with scales along rows and time samples along columns. / スケールを行、時刻を列に持つ <c>scales.Length x x.Length</c> の新しい複素行列。</returns>
    /// <remarks>
    /// <para>
    /// Uses <see cref="WaveletPadding.Zero"/> and does not subtract the signal mean.
    /// Each scale <c>s</c> shares the same units as <paramref name="sampleSpacing"/>, with equivalent Fourier period <see cref="ContinuousWavelet.FourierFactor"/><c> * s</c>.
    /// </para>
    /// <para>
    /// 境界には <see cref="WaveletPadding.Zero"/> を使用し、信号の平均値は差し引きません。
    /// スケール <c>s</c> は <paramref name="sampleSpacing"/> と同じ単位を持ち、対応するフーリエ周期は <see cref="ContinuousWavelet.FourierFactor"/><c> * s</c> です。
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="wavelet"/> is <see langword="null"/>. / <paramref name="wavelet"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="x"/> or <paramref name="scales"/> is empty. / <paramref name="x"/> または <paramref name="scales"/> が空の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">Any scale or <paramref name="sampleSpacing"/> is not positive and finite. / スケールまたは <paramref name="sampleSpacing"/> に正の有限値でない値が含まれる場合。</exception>
    public static Matrix<Complex> Continuous(
        ReadOnlyVectorSpan<double> x, ReadOnlyVectorSpan<double> scales, ContinuousWavelet wavelet, double sampleSpacing) =>
        Continuous(x, scales, wavelet, sampleSpacing, WaveletPadding.Zero);

    /// <summary>
    /// Computes the continuous wavelet transform (CWT) of a signal at the specified scales and padding mode.
    /// <para>指定したスケールとパディング方式で、入力信号の連続ウェーブレット変換（CWT）を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal of length at least 1. / 長さ 1 以上の入力信号。</param>
    /// <param name="scales">The positive finite scales corresponding to each output row. / 各出力行に対応する正の有限スケール。</param>
    /// <param name="wavelet">The mother wavelet to use. / 使用するマザーウェーブレット。</param>
    /// <param name="sampleSpacing">The positive finite sample spacing delta-t. / 正の有限の標本間隔 δt。</param>
    /// <param name="padding">The padding mode applied at the signal edges. / 信号端に適用するパディング方式。</param>
    /// <returns>A newly allocated <c>scales.Length x x.Length</c> complex matrix with scales along rows and time samples along columns. / スケールを行、時刻を列に持つ <c>scales.Length x x.Length</c> の新しい複素行列。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="wavelet"/> is <see langword="null"/>. / <paramref name="wavelet"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="x"/> or <paramref name="scales"/> is empty. / <paramref name="x"/> または <paramref name="scales"/> が空の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">Any scale or <paramref name="sampleSpacing"/> is not positive and finite, or <paramref name="padding"/> is not a defined value. / スケールや <paramref name="sampleSpacing"/> が正の有限値でない場合、または <paramref name="padding"/> が定義外の値の場合。</exception>
    public static Matrix<Complex> Continuous(
        ReadOnlyVectorSpan<double> x, ReadOnlyVectorSpan<double> scales, ContinuousWavelet wavelet, double sampleSpacing,
        WaveletPadding padding)
    {
        RequireContinuous(x.Length, nameof(x), scales, wavelet, sampleSpacing, padding);
        var result = Matrix<Complex>.Uninitialized(scales.Length, x.Length);
        ContinuousCore(x, scales, wavelet, sampleSpacing, padding, result);
        return result;
    }

    /// <summary>
    /// Computes the continuous wavelet transform (CWT) of a signal and writes the complex coefficients into <paramref name="destination"/>.
    /// <para>連続ウェーブレット変換（CWT）を計算し、複素係数を <paramref name="destination"/> へ書き込みます。</para>
    /// </summary>
    /// <param name="x">The input signal of length at least 1. / 長さ 1 以上の入力信号。</param>
    /// <param name="scales">The positive finite scales corresponding to each output row. / 各出力行に対応する正の有限スケール。</param>
    /// <param name="wavelet">The mother wavelet to use. / 使用するマザーウェーブレット。</param>
    /// <param name="sampleSpacing">The positive finite sample spacing delta-t. / 正の有限の標本間隔 δt。</param>
    /// <param name="padding">The padding mode applied at the signal edges. / 信号端に適用するパディング方式。</param>
    /// <param name="destination">The destination matrix with <c>scales.Length</c> rows and <c>x.Length</c> columns. / 行数 <c>scales.Length</c>、列数 <c>x.Length</c> の出力先行列。</param>
    /// <returns><paramref name="destination"/> containing the complex wavelet coefficients. / 複素係数を書き込んだ <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="wavelet"/> is <see langword="null"/>. / <paramref name="wavelet"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="x"/> or <paramref name="scales"/> is empty, or <paramref name="destination"/> does not have shape <c>scales.Length x x.Length</c>. / <paramref name="x"/> や <paramref name="scales"/> が空の場合、または <paramref name="destination"/> の形状が一致しない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">Any scale or <paramref name="sampleSpacing"/> is not positive and finite, or <paramref name="padding"/> is not a defined value. / スケールや <paramref name="sampleSpacing"/> が正の有限値でない場合、または <paramref name="padding"/> が定義外の値の場合。</exception>
    public static MatrixSpan<Complex> Continuous(
        ReadOnlyVectorSpan<double> x, ReadOnlyVectorSpan<double> scales, ContinuousWavelet wavelet, double sampleSpacing,
        WaveletPadding padding, MatrixSpan<Complex> destination)
    {
        RequireContinuous(x.Length, nameof(x), scales, wavelet, sampleSpacing, padding);
        if (destination.Rows != scales.Length || destination.Columns != x.Length)
        {
            throw new ArgumentException(
                $"The destination must be {scales.Length} x {x.Length}, but it is {destination.Rows} x {destination.Columns}.", nameof(destination));
        }

        ContinuousCore(x, scales, wavelet, sampleSpacing, padding, destination);
        return destination;
    }

    /// <summary>
    /// Generates a geometric progression of scales <c>s_j = s0 * 2^(j * delta_j)</c> for <c>j = 0 .. count - 1</c>.
    /// <para>等比数列のスケール <c>s_j = s0 * 2^(j * delta_j)</c>（<c>j = 0 .. count - 1</c>）を生成します。</para>
    /// </summary>
    /// <param name="smallest">The smallest scale <c>s0</c>, positive and finite (e.g., <c>2 * dt / FourierFactor</c> for a minimum period of <c>2 * dt</c>). / 正の有限の最小スケール <c>s0</c>（最小周期を <c>2 * dt</c> にする場合は <c>2 * dt / FourierFactor</c>）。</param>
    /// <param name="octaveFraction">The scale spacing per octave <c>delta_j</c>, positive and finite (e.g., 0.25; smaller values improve inverse accuracy). / 1 オクターブあたりのスケール間隔 <c>delta_j</c>（正の有限値。例: 0.25。小さいほど逆変換の精度が向上します）。</param>
    /// <param name="count">The number of scales to generate, at least 1. / 1 以上のスケール数。</param>
    /// <returns>A newly allocated vector containing the geometric scales. / 生成された等比スケールを格納した新しいベクトル。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="smallest"/> or <paramref name="octaveFraction"/> is not positive and finite, <paramref name="count"/> is less than 1, or the largest scale overflows. / <paramref name="smallest"/> や <paramref name="octaveFraction"/> が正の有限値でない場合、<paramref name="count"/> が 1 未満の場合、または最大スケールがオーバーフローする場合。</exception>
    public static Vector<double> GeometricScales(double smallest, double octaveFraction, int count)
    {
        RequirePositive(smallest, nameof(smallest));
        RequirePositive(octaveFraction, nameof(octaveFraction));
        ArgumentOutOfRangeException.ThrowIfLessThan(count, 1);
        var result = Vector<double>.Uninitialized(count);
        var values = result.AsSpan();
        for (int j = 0; j < count; j++)
        {
            values[j] = smallest * Math.Pow(2, j * octaveFraction);
        }

        if (!double.IsFinite(values[count - 1]))
        {
            throw new ArgumentOutOfRangeException(nameof(count), count, "The largest scale overflows.");
        }

        return result;
    }

    /// <summary>
    /// Computes the cone of influence (COI) scale boundary at each sample index for a continuous wavelet transform.
    /// <para>連続ウェーブレット変換の各時刻において境界の影響を受けるスケールの目安（Cone of Influence）を計算します。</para>
    /// </summary>
    /// <param name="length">The signal length n, at least 1. / 1 以上の信号の長さ n。</param>
    /// <param name="wavelet">The continuous wavelet used for the transform. / 変換に使用する連続ウェーブレット。</param>
    /// <param name="sampleSpacing">The positive finite sample spacing delta-t. / 正の有限の標本間隔 δt。</param>
    /// <returns>A vector whose i-th element is <c>delta_t * min(i, n - 1 - i) / EFoldingTime</c>; scales above this value are affected by edge effects. / 時刻 i に対する <c>delta_t * min(i, n - 1 - i) / EFoldingTime</c> を格納したベクトル（この値を超えるスケールは境界の影響を受けます）。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="wavelet"/> is <see langword="null"/>. / <paramref name="wavelet"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is less than 1 or <paramref name="sampleSpacing"/> is not positive and finite. / <paramref name="length"/> が 1 未満、または <paramref name="sampleSpacing"/> が正の有限値でない場合。</exception>
    public static Vector<double> ConeOfInfluence(int length, ContinuousWavelet wavelet, double sampleSpacing)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(length, 1);
        ArgumentNullException.ThrowIfNull(wavelet);
        RequirePositive(sampleSpacing, nameof(sampleSpacing));
        double eFolding = wavelet.EFoldingTime;
        var result = Vector<double>.Uninitialized(length);
        var values = result.AsSpan();
        for (int i = 0; i < length; i++)
        {
            values[i] = sampleSpacing * Math.Min(i, length - 1 - i) / eFolding;
        }

        return result;
    }

    /// <summary>
    /// Approximately reconstructs a real-valued signal from its continuous wavelet transform coefficients.
    /// <para>連続ウェーブレット変換の係数から実数信号を近似的に再構成します。</para>
    /// </summary>
    /// <param name="coefficients">The <c>scales.Length x n</c> complex CWT coefficient matrix. / <c>scales.Length x n</c> の複素 CWT 係数行列。</param>
    /// <param name="scales">At least two positive, finite, strictly increasing scales matching those used in the forward transform. / 順変換と同じ 2 つ以上の狭義単調増加な正の有限スケール。</param>
    /// <param name="wavelet">The mother wavelet used in the forward transform (must have a non-zero summed real response; odd-order DOG wavelets cannot be inverted). / 順変換に使用したマザーウェーブレット（実部応答の和が 0 になる奇数次 DOG などは使用できません）。</param>
    /// <param name="sampleSpacing">The positive finite sample spacing delta-t used in the forward transform. / 順変換に使用した正の有限の標本間隔 δt。</param>
    /// <returns>A newly allocated vector containing the reconstructed signal, with length equal to the column count of <paramref name="coefficients"/>. / 係数行列の列数と同じ長さの再構成信号を格納した新しいベクトル。</returns>
    /// <remarks>
    /// <para>
    /// Signal mean and period components outside the covered scale range are not recovered, and periods near the scale boundaries are attenuated.
    /// </para>
    /// <para>
    /// スケール範囲外の周期成分および信号の平均値は復元されず、スケール範囲の端に近い周期成分は減衰します。
    /// </para>
    /// </remarks>
    public static Vector<double> InverseContinuous(
        ReadOnlyMatrixSpan<Complex> coefficients, ReadOnlyVectorSpan<double> scales, ContinuousWavelet wavelet, double sampleSpacing)
    {
        var gains = InverseGains(coefficients, scales, wavelet, sampleSpacing);
        var result = Vector<double>.Uninitialized(coefficients.Columns);
        InverseContinuousCore(coefficients, gains, result.AsVectorSpan());
        return result;
    }

    /// <summary>
    /// Approximately reconstructs a real-valued signal from its continuous wavelet transform coefficients into <paramref name="destination"/>.
    /// <para>連続ウェーブレット変換の係数から実数信号を近似的に再構成し、<paramref name="destination"/> へ書き込みます。</para>
    /// </summary>
    /// <param name="coefficients">The <c>scales.Length x n</c> complex CWT coefficient matrix. / <c>scales.Length x n</c> の複素 CWT 係数行列。</param>
    /// <param name="scales">At least two positive, finite, strictly increasing scales matching those used in the forward transform. / 順変換と同じ 2 つ以上の狭義単調増加な正の有限スケール。</param>
    /// <param name="wavelet">The mother wavelet used in the forward transform. / 順変換に使用したマザーウェーブレット。</param>
    /// <param name="sampleSpacing">The positive finite sample spacing delta-t used in the forward transform. / 順変換に使用した正の有限の標本間隔 δt。</param>
    /// <param name="destination">The destination span with length equal to the column count of <paramref name="coefficients"/>. / 係数行列の列数と同じ長さの出力先スパン。</param>
    /// <returns><paramref name="destination"/> containing the reconstructed signal. / 再構成した信号を書き込んだ <paramref name="destination"/>。</returns>
    public static VectorSpan<double> InverseContinuous(
        ReadOnlyMatrixSpan<Complex> coefficients, ReadOnlyVectorSpan<double> scales, ContinuousWavelet wavelet, double sampleSpacing,
        VectorSpan<double> destination)
    {
        var gains = InverseGains(coefficients, scales, wavelet, sampleSpacing);
        RequireLength(coefficients.Columns, destination.Length, nameof(destination));
        InverseContinuousCore(coefficients, gains, destination);
        return destination;
    }

    private const double Negligible = 1e-200;

    private const double ScaleWork = 256;

    private static void ContinuousCore(
        ReadOnlyVectorSpan<double> x, ReadOnlyVectorSpan<double> scales, ContinuousWavelet wavelet, double sampleSpacing,
        WaveletPadding padding, MatrixSpan<Complex> destination)
    {
        int n = x.Length;
        int length = TransformLength(n, padding);
        var plan = FourierPlan.Create(length);
        Complex[] spectrum = ArrayPool<Complex>.Shared.Rent(length);
        for (int i = 0; i < n; i++)
        {
            spectrum[i] = new Complex(x[i], 0);
        }

        spectrum.AsSpan(n, length - n).Clear();
        VectorSpan<Complex> spectrumView = spectrum.AsSpan(0, length);
        plan.Forward(spectrumView, spectrumView);

        double[] ratios = new double[scales.Length];
        for (int j = 0; j < ratios.Length; j++)
        {
            ratios[j] = scales[j] / sampleSpacing;
        }

        int threads = Math.Min(Parallelism.Degree(ScaleWork * ratios.Length * length, Parallelism.Grain), ratios.Length);
        if (threads > 1)
        {
            ContinuousScalesParallel(plan, spectrum, ratios, wavelet, n, destination, threads);
        }
        else
        {
            ContinuousScales(plan, spectrum, ratios, wavelet, n, destination, 0, ratios.Length);
        }

        ArrayPool<Complex>.Shared.Return(spectrum);
    }

    private static void ContinuousScales(
        FourierPlan plan, Complex[] spectrum, double[] ratios, ContinuousWavelet wavelet, int n, MatrixSpan<Complex> destination, int first, int end)
    {
        int length = plan.Length;
        Complex[] buffer = ArrayPool<Complex>.Shared.Rent(length);
        var work = buffer.AsSpan(0, length);
        VectorSpan<Complex> workView = work;
        for (int j = first; j < end; j++)
        {
            double ratio = ratios[j];
            double norm = Math.Sqrt(2 * Math.PI * ratio);
            double step = 2 * Math.PI * ratio / length;
            for (int k = 0; k < length; k++)
            {
                // Torrence–Compo (1998) eq. (6); split the Nyquist bin evenly between positive and negative frequencies.
                var psi = 2 * k == length
                    ? 0.5 * (wavelet.FourierTransform(step * k) + wavelet.FourierTransform(-step * k))
                    : wavelet.FourierTransform(step * (k <= length / 2 ? k : k - length));
                // Flush negligible tails to zero to prevent slow subnormal arithmetic inside the inverse FFT.
                double re = Math.Abs(psi.Real) < Negligible ? 0 : psi.Real;
                double im = Math.Abs(psi.Imaginary) < Negligible ? 0 : psi.Imaginary;
                work[k] = spectrum[k] * new Complex(re * norm, -im * norm);
            }

            plan.Inverse(workView, workView);
            var row = destination.Row(j);
            for (int i = 0; i < n; i++)
            {
                row[i] = work[i];
            }
        }

        ArrayPool<Complex>.Shared.Return(buffer);
    }

    // Separate method avoids allocating a closure on the sequential path.
    private static unsafe void ContinuousScalesParallel(
        FourierPlan plan, Complex[] spectrum, double[] ratios, ContinuousWavelet wavelet, int n, MatrixSpan<Complex> destination, int threads)
    {
        var view = destination.AsReadOnly();
        int rows = view.Rows, columns = view.Columns, rowStride = view.RowStride, columnStride = view.ColumnStride;
        fixed (Complex* pointer = &Unsafe.AsRef(in view.Reference))
        {
            nint address = (nint)pointer;
            Parallelism.ForStatic(threads, part =>
            {
                var target = MatrixSpan<Complex>.Writable(
                    new ReadOnlyMatrixSpan<Complex>(in Unsafe.AsRef<Complex>((void*)address), rows, columns, rowStride, columnStride));
                ContinuousScales(
                    plan, spectrum, ratios, wavelet, n, target,
                    (int)((long)ratios.Length * part / threads), (int)((long)ratios.Length * (part + 1) / threads));
            });
        }
    }

    // Computes per-scale reconstruction weights calibrated by averaging the discrete scale response over one log-scale step.
    private static double[] InverseGains(
        ReadOnlyMatrixSpan<Complex> coefficients, ReadOnlyVectorSpan<double> scales, ContinuousWavelet wavelet, double sampleSpacing)
    {
        int count = scales.Length;
        if (count < 2)
        {
            throw new ArgumentException("The inverse transform needs at least two scales.", nameof(scales));
        }

        RequireContinuous(coefficients.Columns, nameof(coefficients), scales, wavelet, sampleSpacing, WaveletPadding.Zero);
        if (coefficients.Rows != count)
        {
            throw new ArgumentException($"The coefficients have {coefficients.Rows} rows for {count} scales.", nameof(coefficients));
        }

        for (int j = 1; j < count; j++)
        {
            if (!(scales[j] > scales[j - 1]))
            {
                throw new ArgumentException("The scales must be strictly increasing.", nameof(scales));
            }
        }

        var weights = new double[count];
        for (int j = 0; j < count; j++)
        {
            weights[j] = j == 0 ? Math.Log(scales[1] / scales[0])
                : j == count - 1 ? Math.Log(scales[j] / scales[j - 1])
                : 0.5 * Math.Log(scales[j + 1] / scales[j - 1]);
        }

        double peak = 2 * Math.PI / wavelet.FourierFactor;
        double reference = peak / Math.Sqrt(scales[0]) / Math.Sqrt(scales[count - 1]);
        if (!(reference > 0) || !double.IsFinite(reference))
        {
            throw new ArgumentException($"The Fourier factor of the {wavelet.Name} wavelet must be positive and finite.", nameof(wavelet));
        }

        // Average the summed response across one log-scale interval to smooth discrete-scale ripple.
        const int samples = 64;
        double width = weights[count / 2];
        double response = 0, magnitude = 0;
        for (int t = 0; t < samples; t++)
        {
            double omega = reference * Math.Exp(width * ((t + 0.5) / samples - 0.5));
            for (int j = 0; j < count; j++)
            {
                var positive = wavelet.FourierTransform(scales[j] * omega);
                var negative = wavelet.FourierTransform(-scales[j] * omega);
                response += weights[j] * (positive.Real + negative.Real);
                magnitude += weights[j] * (positive.Magnitude + negative.Magnitude);
            }
        }

        if (!(Math.Abs(response) > 1e-8 * magnitude))
        {
            throw new ArgumentException(
                $"The {wavelet.Name} wavelet cannot be inverted: the real part of its summed response is zero.", nameof(wavelet));
        }

        double factor = 2 * samples / (Math.Sqrt(2 * Math.PI / sampleSpacing) * response);
        var gains = new double[count];
        for (int j = 0; j < count; j++)
        {
            gains[j] = weights[j] / Math.Sqrt(scales[j]) * factor;
        }

        return gains;
    }

    private static void InverseContinuousCore(ReadOnlyMatrixSpan<Complex> coefficients, double[] gains, VectorSpan<double> destination)
    {
        for (int i = 0; i < coefficients.Columns; i++)
        {
            double sum = 0;
            for (int j = 0; j < gains.Length; j++)
            {
                sum += gains[j] * coefficients[j, i].Real;
            }

            destination[i] = sum;
        }
    }

    private static int TransformLength(int length, WaveletPadding padding) =>
        padding == WaveletPadding.Zero ? FourierPlan.NextFastLength(checked(2 * length - 1)) : length;

    private static void RequireContinuous(
        int length, string lengthName, ReadOnlyVectorSpan<double> scales, ContinuousWavelet wavelet, double sampleSpacing,
        WaveletPadding padding)
    {
        ArgumentNullException.ThrowIfNull(wavelet);
        if (length < 1)
        {
            throw new ArgumentException("The signal must not be empty.", lengthName);
        }

        if (scales.Length < 1)
        {
            throw new ArgumentException("At least one scale is needed.", nameof(scales));
        }

        for (int j = 0; j < scales.Length; j++)
        {
            if (!(scales[j] > 0) || !double.IsFinite(scales[j]))
            {
                throw new ArgumentOutOfRangeException(nameof(scales), scales[j], "Every scale must be positive and finite.");
            }
        }

        RequirePositive(sampleSpacing, nameof(sampleSpacing));
        if (padding is not (WaveletPadding.Zero or WaveletPadding.Periodic))
        {
            throw new ArgumentOutOfRangeException(nameof(padding), padding, "Unknown padding.");
        }
    }

    private static void RequirePositive(double value, string name)
    {
        if (!(value > 0) || !double.IsFinite(value))
        {
            throw new ArgumentOutOfRangeException(name, value, "The value must be positive and finite.");
        }
    }
}
