using System.Globalization;
using System.Numerics;

namespace Tuna.SignalProcessing;

/// <summary>
/// Represents a mother wavelet basis (<c>ψ₀</c> and its Fourier transform <c>ψ̂₀</c>) for continuous wavelet transforms.
/// <para>連続ウェーブレット変換の基底（母ウェーブレット <c>ψ₀</c> とそのフーリエ変換 <c>ψ̂₀</c>）を表します。</para>
/// </summary>
/// <remarks>
/// <para>Follows the unit-energy normalization of Torrence–Compo (1998, Table 1): <c>∫|ψ₀(η)|² dη = 1</c> and <c>ψ̂₀(ω) = (2π)^(-1/2) ∫ψ₀(η) e^(-iωη) dη</c>. At scale <c>s</c> and sample spacing <c>δt</c>, <see cref="Wavelets"/> multiplies <c>ψ̂₀(sω)</c> by <c>(2πs / δt)^(1/2)</c>. Custom wavelets can be defined by subclassing; continuous transforms and reconstructions invoke <see cref="FourierTransform"/>, while cone-of-influence calculations use <see cref="EFoldingTime"/>.</para>
/// <para>Torrence–Compo (1998, Table 1) の単位エネルギー正規化（<c>∫|ψ₀(η)|² dη = 1</c>、<c>ψ̂₀(ω) = (2π)^(−1/2) ∫ψ₀(η) e^(−iωη) dη</c>）に従います。スケール <c>s</c>・標本間隔 <c>δt</c> の変換では <c>ψ̂₀(sω)</c> に <c>(2πs / δt)^(1/2)</c> を掛けます。派生クラスで独自のウェーブレットを定義でき、変換と逆変換は <see cref="FourierTransform"/>、影響円錐の計算は <see cref="EFoldingTime"/> を使用します。</para>
/// </remarks>
public abstract class ContinuousWavelet
{
    // e^x が非正規化数となる境界（非正規化演算による性能低下を防ぐため 0 に切り捨てる）。
    private const double MinimumExponent = -708;

    /// <summary>
    /// Initializes a new instance of <see cref="ContinuousWavelet"/> with the specified display name.
    /// <para>指定した名前で <see cref="ContinuousWavelet"/> を初期化します。</para>
    /// </summary>
    /// <param name="name">The wavelet name. / ウェーブレット名。</param>
    /// <exception cref="ArgumentNullException"><paramref name="name"/> is <see langword="null"/>. / <paramref name="name"/> が <see langword="null"/> の場合。</exception>
    protected ContinuousWavelet(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        Name = name;
    }

    /// <summary>
    /// Gets the wavelet name (for example, <c>Morlet(6)</c>, <c>Paul(4)</c>, or <c>DOG(2)</c>).
    /// <para>ウェーブレット名（<c>Morlet(6)</c>、<c>Paul(4)</c>、<c>DOG(2)</c> など）を取得します。</para>
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Evaluates the Fourier transform <c>ψ̂₀(ω)</c> at the dimensionless scaled angular frequency <paramref name="angularFrequency"/>.
    /// <para>無次元角周波数 <paramref name="angularFrequency"/> におけるフーリエ変換 <c>ψ̂₀(ω)</c> を評価します。</para>
    /// </summary>
    /// <param name="angularFrequency">The dimensionless angular frequency <c>s·ω</c>. / スケールを掛けた無次元角周波数 <c>s·ω</c>。</param>
    /// <returns>The value <c>ψ̂₀(angularFrequency)</c>. / <c>ψ̂₀(angularFrequency)</c> の値。</returns>
    public abstract Complex FourierTransform(double angularFrequency);

    /// <summary>
    /// Evaluates the mother wavelet <c>ψ₀(η)</c> at the dimensionless time <paramref name="time"/>.
    /// <para>無次元時刻 <paramref name="time"/> における母ウェーブレット <c>ψ₀(η)</c> を評価します。</para>
    /// </summary>
    /// <param name="time">The dimensionless time <c>η = t / s</c>. / 無次元時刻 <c>η = t / s</c>。</param>
    /// <returns>The value <c>ψ₀(time)</c>. / <c>ψ₀(time)</c> の値。</returns>
    public abstract Complex Evaluate(double time);

    /// <summary>
    /// Gets the scale-to-Fourier-period conversion factor (<c>period = FourierFactor · s</c>).
    /// <para>スケールからフーリエ周期への換算係数（<c>周期 = FourierFactor · s</c>）を取得します。</para>
    /// </summary>
    public abstract double FourierFactor { get; }

    /// <summary>
    /// Gets the e-folding time ratio <c>τ_s / s</c> at which edge effects decay by a factor of <c>e^(-2)</c>.
    /// <para>境界効果のパワーが <c>e^(−2)</c> に減衰する e-folding 時間とスケールの比 <c>τ_s / s</c> を取得します。</para>
    /// </summary>
    public abstract double EFoldingTime { get; }

    /// <inheritdoc/>
    public override string ToString() => Name;

    /// <summary>
    /// Returns the default complex Morlet wavelet with central angular frequency <c>ω₀ = 6</c>.
    /// <para>既定の中心角周波数 <c>ω₀ = 6</c> の複素 Morlet ウェーブレットを返します。</para>
    /// </summary>
    /// <returns>The Morlet wavelet with <c>ω₀ = 6</c>. / 中心角周波数 <c>ω₀ = 6</c> の複素 Morlet ウェーブレット。</returns>
    public static ContinuousWavelet Morlet() => Morlet(6);

    /// <summary>
    /// Returns the complex Morlet wavelet with the specified central angular frequency.
    /// <para>指定した中心角周波数の複素 Morlet ウェーブレットを返します。</para>
    /// </summary>
    /// <param name="centralFrequency">The central angular frequency <c>ω₀</c> (positive and finite). / 中心角周波数 <c>ω₀</c>（正の有限値）。</param>
    /// <returns>The complex Morlet wavelet for <paramref name="centralFrequency"/>. / 指定した中心角周波数の複素 Morlet ウェーブレット。</returns>
    /// <remarks>
    /// <para>Uses the analytic spectrum <c>ψ̂₀(ω) = π^(-1/4) H(ω) exp(-(ω - ω₀)² / 2)</c> truncated to <c>ω &gt; 0</c> (Torrence–Compo 1998), while <see cref="Evaluate"/> returns the untruncated form <c>π^(-1/4) exp(iω₀η - η² / 2)</c>. The discrepancy is ~1e-9 at <c>ω₀ = 6</c> and grows for <c>ω₀ &lt; 5</c>.</para>
    /// <para>変換では Torrence–Compo (1998) と同様に負の周波数を 0 とした <c>ψ̂₀(ω)</c> を用い、<see cref="Evaluate"/> は切り捨てのない <c>ψ₀(η)</c> を返します。両者の差は <c>ω₀ = 6</c> で約 1e−9 であり、<c>ω₀ &lt; 5</c> では大きくなります。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="centralFrequency"/> is not positive and finite. / <paramref name="centralFrequency"/> が正の有限値でない場合。</exception>
    public static ContinuousWavelet Morlet(double centralFrequency)
    {
        if (!(centralFrequency > 0) || !double.IsFinite(centralFrequency))
        {
            throw new ArgumentOutOfRangeException(nameof(centralFrequency), centralFrequency, "The central frequency must be positive and finite.");
        }

        return new MorletWavelet(centralFrequency);
    }

    /// <summary>
    /// Returns the default 4th-order complex Paul wavelet.
    /// <para>既定の 4 次複素 Paul ウェーブレットを返します。</para>
    /// </summary>
    /// <returns>The 4th-order Paul wavelet. / 4 次 Paul ウェーブレット。</returns>
    public static ContinuousWavelet Paul() => Paul(4);

    /// <summary>
    /// Returns the complex Paul wavelet of the specified order.
    /// <para>指定した次数の複素 Paul ウェーブレットを返します。</para>
    /// </summary>
    /// <param name="order">The wavelet order, from 1 to 100. / 次数（1〜100）。</param>
    /// <returns>The Paul wavelet of order <paramref name="order"/>. / 指定した次数の Paul ウェーブレット。</returns>
    /// <remarks>
    /// <para>Uses <c>ψ̂₀(ω) = 2^m / √(m·(2m - 1)!) · H(ω) · ω^m e^(-ω)</c> (Torrence–Compo 1998, Table 1) and the matching inverse <c>ψ₀(η) = 2^m m! / √(π·(2m)!) · (1 - iη)^(-(m + 1))</c> without the extra <c>i^m</c> factor from Table 1.</para>
    /// <para><c>ψ̂₀(ω)</c> は Torrence–Compo (1998, Table 1) の定義に従い、<see cref="Evaluate"/> はその正確な逆フーリエ変換に一致させています（Table 1 の <c>ψ₀</c> にある余分な <c>i^m</c> 因子を除いています）。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="order"/> is outside <c>[1, 100]</c>. / <paramref name="order"/> が 1〜100 の範囲外の場合。</exception>
    public static ContinuousWavelet Paul(int order)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(order, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(order, 100);
        return new PaulWavelet(order);
    }

    /// <summary>
    /// Gets the Mexican hat (Ricker) wavelet, equivalent to <see cref="DerivativeOfGaussian"/> of order 2.
    /// <para>Mexican hat（Ricker）ウェーブレット（2 次 <see cref="DerivativeOfGaussian"/> と同一）を取得します。</para>
    /// </summary>
    public static ContinuousWavelet MexicanHat { get; } = new DerivativeOfGaussianWavelet(2);

    /// <summary>
    /// Returns the real-valued derivative-of-Gaussian (DOG) wavelet of the specified order.
    /// <para>指定した階数のガウス関数微分（DOG）実数ウェーブレットを返します。</para>
    /// </summary>
    /// <param name="order">The derivative order, from 1 to 100. / 微分階数（1〜100）。</param>
    /// <returns>The DOG wavelet of order <paramref name="order"/>. / 指定した階数の DOG ウェーブレット。</returns>
    /// <remarks>
    /// <para>Uses <c>ψ̂₀(ω) = -i^m / √Γ(m + 1/2) · ω^m exp(-ω² / 2)</c> (Torrence–Compo 1998, Table 1) and the exact corresponding time-domain form <c>ψ₀(η) = (-1)^(m + 1) He_m(η) exp(-η² / 2) / √Γ(m + 1/2)</c>. Odd orders have <c>ψ₀(0) = 0</c> and cannot be inverted via delta-function reconstruction.</para>
    /// <para><c>ψ̂₀(ω)</c> は Torrence–Compo (1998, Table 1) に従い、<see cref="Evaluate"/> はその正確な逆変換に符号を合わせています。実数ウェーブレットのため実数信号の変換は実数となり、奇数階では <c>ψ₀(0) = 0</c> のためデルタ関数による逆変換はできません。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="order"/> is outside <c>[1, 100]</c>. / <paramref name="order"/> が 1〜100 の範囲外の場合。</exception>
    public static ContinuousWavelet DerivativeOfGaussian(int order)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(order, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(order, 100);
        return order == 2 ? MexicanHat : new DerivativeOfGaussianWavelet(order);
    }

    private sealed class MorletWavelet(double centralFrequency)
        : ContinuousWavelet($"Morlet({centralFrequency.ToString("R", CultureInfo.InvariantCulture)})")
    {
        private static readonly double Normalization = Math.Pow(Math.PI, -0.25);

        public override Complex FourierTransform(double angularFrequency)
        {
            if (!(angularFrequency > 0))
            {
                return Complex.Zero;
            }

            double exponent = -0.5 * (angularFrequency - centralFrequency) * (angularFrequency - centralFrequency);
            return exponent < MinimumExponent ? Complex.Zero : new Complex(Normalization * Math.Exp(exponent), 0);
        }

        public override Complex Evaluate(double time)
        {
            double envelope = Normalization * Math.Exp(-0.5 * time * time);
            double phase = centralFrequency * time;
            return new Complex(envelope * Math.Cos(phase), envelope * Math.Sin(phase));
        }

        public override double FourierFactor { get; } =
            4 * Math.PI / (centralFrequency + Math.Sqrt(2 + centralFrequency * centralFrequency));

        public override double EFoldingTime => Math.Sqrt(2);
    }

    private sealed class PaulWavelet : ContinuousWavelet
    {
        private readonly int order;

        // 階乗のオーバーフローと中間積の非正規化を避けるため、平方根の逐次積で正規化定数を計算する。
        private readonly double spectralNormalization;
        private readonly double temporalNormalization;

        public PaulWavelet(int order) : base($"Paul({order.ToString(CultureInfo.InvariantCulture)})")
        {
            this.order = order;
            double spectral = Math.Sqrt(2), temporal = 1 / Math.Sqrt(Math.PI);
            for (int k = 1; k <= order; k++)
            {
                spectral *= Math.Sqrt(2.0 / ((2.0 * k - 1) * k));
                temporal *= Math.Sqrt(2.0 * k / (2.0 * k - 1));
            }

            spectralNormalization = spectral;
            temporalNormalization = temporal;
            FourierFactor = 4 * Math.PI / (2 * order + 1);
        }

        public override Complex FourierTransform(double angularFrequency)
        {
            if (!(angularFrequency > 0) || -angularFrequency < MinimumExponent)
            {
                return Complex.Zero;
            }

            return new Complex(spectralNormalization * Math.Pow(angularFrequency, order) * Math.Exp(-angularFrequency), 0);
        }

        public override Complex Evaluate(double time)
        {
            double magnitude = temporalNormalization / Math.Pow(double.Hypot(1, time), order + 1);
            return Complex.FromPolarCoordinates(magnitude, (order + 1) * Math.Atan(time));
        }

        public override double FourierFactor { get; }

        public override double EFoldingTime => 1 / Math.Sqrt(2);
    }

    private sealed class DerivativeOfGaussianWavelet : ContinuousWavelet
    {
        private readonly int order;
        private readonly double normalization;

        public DerivativeOfGaussianWavelet(int order) : base($"DOG({order.ToString(CultureInfo.InvariantCulture)})")
        {
            this.order = order;
            double gamma = Math.Sqrt(Math.PI);
            for (int k = 0; k < order; k++)
            {
                gamma *= k + 0.5;
            }

            normalization = 1 / Math.Sqrt(gamma);
            FourierFactor = 2 * Math.PI / Math.Sqrt(order + 0.5);
        }

        public override Complex FourierTransform(double angularFrequency)
        {
            double exponent = -0.5 * angularFrequency * angularFrequency;
            if (!(exponent >= MinimumExponent))
            {
                return Complex.Zero;
            }

            double value = normalization * Math.Pow(angularFrequency, order) * Math.Exp(exponent);
            return (order % 4) switch
            {
                0 => new Complex(-value, 0),
                1 => new Complex(0, -value),
                2 => new Complex(value, 0),
                _ => new Complex(0, value),
            };
        }

        public override Complex Evaluate(double time)
        {
            double decay = Math.Exp(-0.5 * time * time);
            if (decay == 0)
            {
                return Complex.Zero;
            }

            // 確率論の Hermite 多項式の漸化式: He_{k+1}(η) = η·He_k(η) − k·He_{k−1}(η)
            double previous = 1, current = time;
            for (int k = 1; k < order; k++)
            {
                (previous, current) = (current, time * current - k * previous);
            }

            double sign = order % 2 == 0 ? -1 : 1;
            return new Complex(sign * normalization * current * decay, 0);
        }

        public override double FourierFactor { get; }

        public override double EFoldingTime => Math.Sqrt(2);
    }
}
