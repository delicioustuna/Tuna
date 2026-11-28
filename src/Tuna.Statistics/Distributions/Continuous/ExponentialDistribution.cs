using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the exponential distribution parameterized by rate <c>λ</c> on <c>[0, ∞)</c>.
/// <para>率パラメーター <c>λ</c> でパラメーター化された <c>[0, ∞)</c> 上の指数分布を表します。</para>
/// </summary>
public readonly partial struct ExponentialDistribution : IContinuousDistribution
{
    private readonly double _rate;

    /// <summary>
    /// Initializes a new instance of <see cref="ExponentialDistribution"/> with the specified rate parameter.
    /// <para>率パラメーターを指定して <see cref="ExponentialDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="rate">The rate parameter <c>λ</c> (positive and finite; the mean is <c>1/λ</c>). / 率パラメーター <c>λ</c>（正かつ有限。平均は <c>1/λ</c>）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rate"/> is not positive and finite. / <paramref name="rate"/> が正の有限値でない場合。</exception>
    public ExponentialDistribution(double rate)
        : this(rate, forDifferentiation: false)
    {
    }

    private ExponentialDistribution(double rate, bool forDifferentiation)
    {
        DistributionArgument.PositiveFinite(rate, nameof(rate), forDifferentiation);
        _rate = rate;
    }

    /// <summary>
    /// Creates an <see cref="ExponentialDistribution"/> from its scale (mean) parameter <c>θ = 1/λ</c>.
    /// <para>尺度（平均）パラメーター <c>θ = 1/λ</c> を指定して <see cref="ExponentialDistribution"/> を生成します。</para>
    /// </summary>
    /// <param name="scale">The scale parameter <c>θ</c> (positive, with finite reciprocal). / 尺度パラメーター <c>θ</c>（正で、逆数が有限）。</param>
    /// <returns>An exponential distribution with rate <c>1 / scale</c>. / 率 <c>1 / scale</c> の指数分布。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="scale"/> is not positive and finite, or <c>1 / scale</c> overflows. / <paramref name="scale"/> が正の有限値でない、または <c>1 / scale</c> がオーバーフローする場合。</exception>
    public static ExponentialDistribution FromScale(double scale)
    {
        DistributionArgument.PositiveFinite(scale, nameof(scale));
        return new ExponentialDistribution(1 / scale);
    }

    /// <summary>
    /// Gets the rate parameter <c>λ</c>.
    /// <para>率パラメーター <c>λ</c> を取得します。</para>
    /// </summary>
    public double Rate => _rate;

    /// <summary>
    /// Gets the scale parameter <c>1/λ</c>.
    /// <para>尺度パラメーター <c>1/λ</c> を取得します。</para>
    /// </summary>
    public double Scale => 1 / _rate;

    /// <inheritdoc/>
    public double Mean => 1 / _rate;

    /// <inheritdoc/>
    public double Variance => 1 / (_rate * _rate);

    /// <inheritdoc/>
    public double StandardDeviation => 1 / _rate;

    /// <inheritdoc/>
    public double Skewness => 2;

    /// <inheritdoc/>
    public double ExcessKurtosis => 6;

    /// <inheritdoc/>
    public double Entropy => 1 - Math.Log(_rate);

    /// <inheritdoc/>
    public double Median => XMath.Constants.Ln2 / _rate;

    /// <inheritdoc/>
    public double SupportMinimum => 0;

    /// <inheritdoc/>
    public double SupportMaximum => double.PositiveInfinity;

    /// <inheritdoc/>
    public double Pdf(double x) => x < 0 ? 0 : _rate * Math.Exp(-_rate * x);

    /// <inheritdoc/>
    public double LogPdf(double x) => x < 0 ? double.NegativeInfinity : Math.Log(_rate) - _rate * x;

    /// <inheritdoc/>
    public double Cdf(double x) => x <= 0 ? 0 : -DistributionMath.Expm1(-_rate * x);

    /// <inheritdoc/>
    public double Sf(double x) => x <= 0 ? 1 : Math.Exp(-_rate * x);

    /// <inheritdoc/>
    public double LogCdf(double x) => x <= 0 ? double.NegativeInfinity : DistributionMath.Log1mExp(-_rate * x);

    /// <inheritdoc/>
    public double LogSf(double x) => x <= 0 ? 0 : -_rate * x;

    /// <inheritdoc/>
    public double InverseCdf(double p) => p >= 0 && p <= 1 ? -DistributionMath.Log1p(-p) / _rate : double.NaN;

    /// <inheritdoc/>
    public double InverseSf(double q) => q >= 0 && q <= 1 ? -Math.Log(q) / _rate : double.NaN;

    /// <inheritdoc/>
    public double InverseLogCdf(double logP) => logP <= 0 ? -DistributionMath.Log1mExp(logP) / _rate : double.NaN;

    /// <inheritdoc/>
    public double InverseLogSf(double logQ) => logQ <= 0 ? -logQ / _rate : double.NaN;

    /// <inheritdoc/>
    public double Sample(IRandomSource random)
    {
        ArgumentNullException.ThrowIfNull(random);
        return Ziggurat.StandardExponential(random) / _rate;
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, Span<double> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = Ziggurat.StandardExponential(random) / _rate;
        }
    }
}
