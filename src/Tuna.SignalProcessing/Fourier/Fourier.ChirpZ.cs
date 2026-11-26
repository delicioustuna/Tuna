using System.Buffers;
using System.Numerics;

namespace Tuna.SignalProcessing;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

public static partial class Fourier
{
    /// <summary>
    /// Evaluates the chirp-z transform of a complex signal along a logarithmic spiral contour <c>z_k = start · ratio^(-k)</c>.
    /// <para>複素平面上の対数らせん点列 <c>z_k = start · ratio^(−k)</c> に沿ってチャープ z 変換を評価します。</para>
    /// </summary>
    /// <param name="x">The input complex signal. / 入力の複素数信号。</param>
    /// <param name="count">The number of output points (non-negative). / 評価する出力点数（0 以上）。</param>
    /// <param name="ratio">The inverse contour step ratio <c>w</c> (nonzero and finite). / 点列の公比の逆数 <c>w</c>（0 以外の有限値）。</param>
    /// <param name="start">The initial contour point <c>a</c> (nonzero and finite). / 点列の初項 <c>a</c>（0 以外の有限値）。</param>
    /// <returns>A new vector of length <paramref name="count"/> containing <c>X[k] = Σ_j x[j] · z_k^(-j)</c>. / <c>X[k] = Σ_j x[j] · z_k^(−j)</c> を格納した長さ <paramref name="count"/> の新しいベクトル。</returns>
    /// <remarks>
    /// <para>Reduces to the standard DFT when <c>ratio = exp(-2πi/n)</c>, <c>start = 1</c>, and <c>count = n</c>. When <c>|ratio|</c> deviates from 1 on long inputs, intermediate chirp powers may overflow.</para>
    /// <para><c>ratio = exp(−2πi/n)</c>、<c>start = 1</c>、<c>count = n</c> のとき通常の DFT と一致します。長い入力で <c>|ratio|</c> が 1 から大きく離れると中間計算がオーバーフローする可能性があります。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative, or <paramref name="ratio"/> or <paramref name="start"/> is zero or non-finite. / <paramref name="count"/> が負の場合、または <paramref name="ratio"/>・<paramref name="start"/> が 0 または非有限の場合。</exception>
    public static Vector<Complex> ChirpZ(ReadOnlyVectorSpan<Complex> x, int count, Complex ratio, Complex start)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        var result = Vector<Complex>.Uninitialized(count);
        ChirpZ(x, ratio, start, result);
        return result;
    }

    /// <summary>
    /// Evaluates the chirp-z transform along <c>z_k = start · ratio^(-k)</c> and writes the result to <paramref name="destination"/>.
    /// <para>複素平面上の点列 <c>z_k = start · ratio^(−k)</c> に沿ってチャープ z 変換を評価し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex signal. / 入力の複素数信号。</param>
    /// <param name="ratio">The inverse contour step ratio <c>w</c> (nonzero and finite). / 点列の公比の逆数 <c>w</c>（0 以外の有限値）。</param>
    /// <param name="start">The initial contour point <c>a</c> (nonzero and finite). / 点列の初項 <c>a</c>（0 以外の有限値）。</param>
    /// <param name="destination">The destination span whose length determines the number of evaluated points. May overlap <paramref name="x"/>. / 出力先（その長さが評価点数になります。<paramref name="x"/> と重なっても構いません）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="ratio"/> or <paramref name="start"/> is zero or non-finite. / <paramref name="ratio"/> または <paramref name="start"/> が 0 または非有限の場合。</exception>
    public static VectorSpan<Complex> ChirpZ(ReadOnlyVectorSpan<Complex> x, Complex ratio, Complex start, VectorSpan<Complex> destination)
    {
        RequireNonzeroFinite(ratio, nameof(ratio));
        RequireNonzeroFinite(start, nameof(start));
        int n = x.Length, m = destination.Length;
        if (BluesteinTrivial(n, destination))
        {
            return destination;
        }

        var logRatio = Complex.Log(ratio);
        var logStart = Complex.Log(start);
        Complex[] buffer = RentBluestein(n, m, out int length);
        var u = buffer.AsSpan(0, length);
        var v = buffer.AsSpan(length, length);
        var output = buffer.AsSpan(2 * length, m);
        v.Clear();
        for (int t = 0; t < Math.Max(n, m); t++)
        {
            // Bluestein 恒等式 jk = (j² + k² − (k − j)²)/2 による入力・核・出力チャープ因子
            double half = 0.5 * t * t;
            if (t < n)
            {
                u[t] = x[t] * Complex.Exp(new Complex(
                    logRatio.Real * half - logStart.Real * t, logRatio.Imaginary * half - logStart.Imaginary * t));
            }

            PlaceKernel(v, t, n, m, Complex.Exp(new Complex(-logRatio.Real * half, -logRatio.Imaginary * half)));
            if (t < m)
            {
                output[t] = Complex.Exp(new Complex(logRatio.Real * half, logRatio.Imaginary * half));
            }
        }

        Convolve(buffer, length, n, destination);
        return destination;
    }

    /// <summary>
    /// Evaluates the discrete Fourier transform of a complex signal on an arbitrary uniformly spaced frequency grid (zoom FFT).
    /// <para>任意の等間隔周波数グリッド上で複素数信号の離散フーリエ変換を評価します（ズーム FFT）。</para>
    /// </summary>
    /// <param name="x">The input complex signal. / 入力の複素数信号。</param>
    /// <param name="firstFrequency">The starting frequency <c>f₀</c> (finite). / 開始周波数 <c>f₀</c>（有限値）。</param>
    /// <param name="frequencyStep">The spacing <c>Δf</c> between adjacent frequencies (finite, may be negative). / 周波数間隔 <c>Δf</c>（有限値、負も可）。</param>
    /// <param name="count">The number of frequency points (non-negative). / 評価する周波数点数（0 以上）。</param>
    /// <param name="sampleSpacing">The sample spacing <c>d</c> (positive and finite). Frequencies are in units of <c>1/d</c>. / 標本間隔 <c>d</c>（正の有限値）。周波数の単位は <c>1/d</c> です。</param>
    /// <returns>A new vector containing <c>Σ_j x[j] · exp(-2πi · f_k · j · d)</c> at <c>f_k = firstFrequency + k · frequencyStep</c>. / <c>f_k = firstFrequency + k · frequencyStep</c> における <c>Σ_j x[j] · exp(−2πi · f_k · j · d)</c> を格納した新しいベクトル。</returns>
    /// <remarks>
    /// <para>Uses the unit-circle chirp-z transform (Rabiner–Schafer–Rader 1969) in O((n + count) log(n + count)) time with compensated phase accumulation so accuracy remains comparable to FFT even for long inputs.</para>
    /// <para>単位円上のチャープ z 変換（Rabiner–Schafer–Rader 1969）により O((n + count) log(n + count)) で計算します。位相を補償加算して評価するため、長い信号でも FFT と同水準の精度を保ちます。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative, any frequency parameter is non-finite, <paramref name="sampleSpacing"/> is not positive and finite, or <c>frequency · sampleSpacing</c> overflows. / <paramref name="count"/> が負、周波数引数が非有限、<paramref name="sampleSpacing"/> が正の有限値でない、または周波数と標本間隔の積がオーバーフローする場合。</exception>
    public static Vector<Complex> Zoom(
        ReadOnlyVectorSpan<Complex> x, double firstFrequency, double frequencyStep, int count, double sampleSpacing = 1)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        var result = Vector<Complex>.Uninitialized(count);
        Zoom(x, firstFrequency, frequencyStep, sampleSpacing, result);
        return result;
    }

    /// <summary>
    /// Evaluates the discrete Fourier transform of a real signal on an arbitrary uniformly spaced frequency grid (zoom FFT).
    /// <para>任意の等間隔周波数グリッド上で実数信号の離散フーリエ変換を評価します（ズーム FFT）。</para>
    /// </summary>
    /// <param name="x">The input real signal. / 入力の実数信号。</param>
    /// <param name="firstFrequency">The starting frequency <c>f₀</c> (finite). / 開始周波数 <c>f₀</c>（有限値）。</param>
    /// <param name="frequencyStep">The spacing <c>Δf</c> between adjacent frequencies (finite, may be negative). / 周波数間隔 <c>Δf</c>（有限値、負も可）。</param>
    /// <param name="count">The number of frequency points (non-negative). / 評価する周波数点数（0 以上）。</param>
    /// <param name="sampleSpacing">The sample spacing <c>d</c> (positive and finite). / 標本間隔 <c>d</c>（正の有限値）。</param>
    /// <returns>A new vector containing <c>Σ_j x[j] · exp(-2πi · f_k · j · d)</c> at <c>f_k = firstFrequency + k · frequencyStep</c>. / <c>f_k = firstFrequency + k · frequencyStep</c> における変換値を格納した新しいベクトル。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative, any frequency parameter is non-finite, <paramref name="sampleSpacing"/> is not positive and finite, or <c>frequency · sampleSpacing</c> overflows. / 引数が範囲外または非有限の場合、あるいは周波数と標本間隔の積がオーバーフローする場合。</exception>
    public static Vector<Complex> Zoom(
        ReadOnlyVectorSpan<double> x, double firstFrequency, double frequencyStep, int count, double sampleSpacing = 1)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        var result = Vector<Complex>.Uninitialized(count);
        Zoom(x, firstFrequency, frequencyStep, sampleSpacing, result);
        return result;
    }

    /// <summary>
    /// Evaluates the discrete Fourier transform of a complex signal on a uniformly spaced frequency grid and writes the result to <paramref name="destination"/>.
    /// <para>等間隔周波数グリッド上で複素数信号の離散フーリエ変換を評価し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex signal. / 入力の複素数信号。</param>
    /// <param name="firstFrequency">The starting frequency <c>f₀</c> (finite). / 開始周波数 <c>f₀</c>（有限値）。</param>
    /// <param name="frequencyStep">The spacing <c>Δf</c> between adjacent frequencies (finite, may be negative). / 周波数間隔 <c>Δf</c>（有限値、負も可）。</param>
    /// <param name="sampleSpacing">The sample spacing <c>d</c> (positive and finite). / 標本間隔 <c>d</c>（正の有限値）。</param>
    /// <param name="destination">The destination span whose length determines the number of evaluated frequencies. May overlap <paramref name="x"/>. / 出力先（その長さが評価点数になります。<paramref name="x"/> と重なっても構いません）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentOutOfRangeException">Any frequency parameter is non-finite, <paramref name="sampleSpacing"/> is not positive and finite, or <c>frequency · sampleSpacing</c> overflows. / 引数が非有限・範囲外の場合、または周波数と標本間隔の積がオーバーフローする場合。</exception>
    public static VectorSpan<Complex> Zoom(
        ReadOnlyVectorSpan<Complex> x, double firstFrequency, double frequencyStep, double sampleSpacing, VectorSpan<Complex> destination)
    {
        var (start, step) = ZoomTurns(firstFrequency, frequencyStep, sampleSpacing);
        int n = x.Length;
        if (BluesteinTrivial(n, destination))
        {
            return destination;
        }

        Complex[] buffer = RentBluestein(n, destination.Length, out int length);
        var factors = buffer.AsSpan(0, n);
        ZoomFactors(buffer, length, n, destination.Length, start, step);
        for (int j = 0; j < n; j++)
        {
            factors[j] = x[j] * factors[j];
        }

        Convolve(buffer, length, n, destination);
        return destination;
    }

    /// <summary>
    /// Evaluates the discrete Fourier transform of a real signal on a uniformly spaced frequency grid and writes the result to <paramref name="destination"/>.
    /// <para>等間隔周波数グリッド上で実数信号の離散フーリエ変換を評価し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real signal. / 入力の実数信号。</param>
    /// <param name="firstFrequency">The starting frequency <c>f₀</c> (finite). / 開始周波数 <c>f₀</c>（有限値）。</param>
    /// <param name="frequencyStep">The spacing <c>Δf</c> between adjacent frequencies (finite, may be negative). / 周波数間隔 <c>Δf</c>（有限値、負も可）。</param>
    /// <param name="sampleSpacing">The sample spacing <c>d</c> (positive and finite). / 標本間隔 <c>d</c>（正の有限値）。</param>
    /// <param name="destination">The destination span whose length determines the number of evaluated frequencies. May overlap <paramref name="x"/>. / 出力先（その長さが評価点数になります。<paramref name="x"/> と重なっても構いません）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentOutOfRangeException">Any frequency parameter is non-finite, <paramref name="sampleSpacing"/> is not positive and finite, or <c>frequency · sampleSpacing</c> overflows. / 引数が非有限・範囲外の場合、または周波数と標本間隔の積がオーバーフローする場合。</exception>
    public static VectorSpan<Complex> Zoom(
        ReadOnlyVectorSpan<double> x, double firstFrequency, double frequencyStep, double sampleSpacing, VectorSpan<Complex> destination)
    {
        var (start, step) = ZoomTurns(firstFrequency, frequencyStep, sampleSpacing);
        int n = x.Length;
        if (BluesteinTrivial(n, destination))
        {
            return destination;
        }

        Complex[] buffer = RentBluestein(n, destination.Length, out int length);
        var factors = buffer.AsSpan(0, n);
        ZoomFactors(buffer, length, n, destination.Length, start, step);
        for (int j = 0; j < n; j++)
        {
            factors[j] = new Complex(x[j] * factors[j].Real, x[j] * factors[j].Imaginary);
        }

        Convolve(buffer, length, n, destination);
        return destination;
    }

    private static (double Start, double Step) ZoomTurns(double firstFrequency, double frequencyStep, double sampleSpacing)
    {
        if (!double.IsFinite(firstFrequency))
        {
            throw new ArgumentOutOfRangeException(nameof(firstFrequency), firstFrequency, "The frequency must be finite.");
        }

        if (!double.IsFinite(frequencyStep))
        {
            throw new ArgumentOutOfRangeException(nameof(frequencyStep), frequencyStep, "The frequency step must be finite.");
        }

        if (!(sampleSpacing > 0) || !double.IsFinite(sampleSpacing))
        {
            throw new ArgumentOutOfRangeException(nameof(sampleSpacing), sampleSpacing, "The sample spacing must be positive and finite.");
        }

        double start = firstFrequency * sampleSpacing, step = frequencyStep * sampleSpacing;
        if (!double.IsFinite(start) || !double.IsFinite(step))
        {
            throw new ArgumentOutOfRangeException(nameof(sampleSpacing), sampleSpacing, "The frequency times the sample spacing overflows.");
        }

        return (start, step);
    }

    // 回転数 Δfd·t²/2 と f₀d·t を mod 1 の二重倍精度で漸化的に累積し、長系列での位相誤差蓄積を抑える。
    private static void ZoomFactors(Complex[] buffer, int length, int n, int m, double start, double step)
    {
        var factors = buffer.AsSpan(0, n);
        var v = buffer.AsSpan(length, length);
        var output = buffer.AsSpan(2 * length, m);
        v.Clear();

        double half = Wrap(0.5 * step), offset = Wrap(start);
        double chirpHigh = 0, chirpLow = 0, shiftHigh = 0, shiftLow = 0;
        for (int t = 0; t < Math.Max(n, m); t++)
        {
            var e = UnitTurn(chirpHigh + chirpLow);
            PlaceKernel(v, t, n, m, e);
            if (t < m)
            {
                output[t] = Complex.Conjugate(e);
            }

            if (t < n)
            {
                double sum = shiftHigh + chirpHigh;
                double back = sum - shiftHigh;
                double error = (shiftHigh - (sum - back)) + (chirpHigh - back) + shiftLow + chirpLow;
                factors[t] = UnitTurn(-(Wrap(sum) + error));
            }

            // (t + 1)² = t² + (2t + 1) の増分を FMA で無誤差分割して加算する。
            double odd = 2.0 * t + 1;
            double product = half * odd;
            AddTurns(ref chirpHigh, ref chirpLow, product, Math.FusedMultiplyAdd(half, odd, -product));
            AddTurns(ref shiftHigh, ref shiftLow, offset, 0);
        }
    }

    private static void AddTurns(ref double high, ref double low, double addHigh, double addLow)
    {
        double sum = high + addHigh;
        double back = sum - high;
        double error = (high - (sum - back)) + (addHigh - back) + low + addLow;
        sum -= Math.Round(sum);
        high = sum + error;
        low = error - (high - sum);
    }

    private static void PlaceKernel(Span<Complex> v, int t, int n, int m, Complex value)
    {
        if (t < m)
        {
            v[t] = value;
        }

        if (t > 0 && t < n)
        {
            v[v.Length - t] = value;
        }
    }

    private static bool BluesteinTrivial(int n, VectorSpan<Complex> destination)
    {
        if (destination.Length == 0)
        {
            return true;
        }

        if (n == 0)
        {
            destination.Fill(Complex.Zero);
            return true;
        }

        return false;
    }

    private static Complex[] RentBluestein(int n, int m, out int length)
    {
        length = FourierPlan.NextFastLength(checked(n + m - 1));
        return ArrayPool<Complex>.Shared.Rent(checked(2 * length + m + FourierPlan.Create(length).WorkLength(1)));
    }

    private static void Convolve(Complex[] buffer, int length, int n, VectorSpan<Complex> destination)
    {
        int m = destination.Length;
        var plan = FourierPlan.Create(length);
        var u = buffer.AsSpan(0, length);
        var v = buffer.AsSpan(length, length);
        var output = buffer.AsSpan(2 * length, m);
        var work = buffer.AsSpan(2 * length + m);
        u[n..].Clear();
        plan.Execute<ForwardDirection>(ref u[0], ref u[0], work, 1);
        plan.Execute<ForwardDirection>(ref v[0], ref v[0], work, 1);
        for (int k = 0; k < length; k++)
        {
            u[k] *= v[k];
        }

        plan.Execute<InverseDirection>(ref u[0], ref u[0], work, 1);
        double scale = 1.0 / length;
        for (int k = 0; k < m; k++)
        {
            var value = output[k] * u[k];
            destination[k] = new Complex(value.Real * scale, value.Imaginary * scale);
        }

        ArrayPool<Complex>.Shared.Return(buffer);
    }

    private static Complex UnitTurn(double turns)
    {
        var (sin, cos) = double.SinCosPi(2 * turns);
        return new Complex(cos, sin);
    }

    private static double Wrap(double value) => value - Math.Round(value);

    private static void RequireNonzeroFinite(Complex value, string name)
    {
        if (value == Complex.Zero || !double.IsFinite(value.Real) || !double.IsFinite(value.Imaginary))
        {
            throw new ArgumentOutOfRangeException(name, value, "The value must be nonzero and finite.");
        }
    }
}
