using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents a location-scale transformation <c>X = μ + σY</c> of a continuous distribution.
/// <para>連続分布に位置・尺度変換 <c>X = μ + σY</c> を適用した分布を表します。</para>
/// </summary>
/// <typeparam name="TDistribution">The type of the underlying continuous distribution of <c>Y</c>. / 元の連続分布（<c>Y</c> の分布）の型。</typeparam>
/// <remarks>
/// <para>
/// The mean is <c>μ + σE[Y]</c>, the variance is <c>σ²Var[Y]</c>, skewness and excess kurtosis are unchanged, and differential entropy increases by <c>ln σ</c>.
/// When <typeparamref name="TDistribution"/> is a value type, evaluations dispatch without boxing.
/// </para>
/// <para>
/// 平均は <c>μ + σE[Y]</c>、分散は <c>σ²Var[Y]</c>、歪度および超過尖度は不変で、微分エントロピーは <c>ln σ</c> だけ増加します。
/// <typeparamref name="TDistribution"/> が値型の場合、ボックス化なしで元の分布を評価します。
/// </para>
/// </remarks>
public readonly struct LocationScaleDistribution<TDistribution> : IContinuousDistribution
    where TDistribution : IContinuousDistribution
{
    private readonly TDistribution _distribution;
    private readonly double _location;
    private readonly double _scale;

    /// <summary>
    /// Initializes a new instance of <see cref="LocationScaleDistribution{TDistribution}"/>.
    /// <para>元の分布と位置・尺度パラメーターを指定して <see cref="LocationScaleDistribution{TDistribution}"/> を初期化します。</para>
    /// </summary>
    /// <param name="distribution">The underlying continuous distribution. / 元の連続分布。</param>
    /// <param name="location">The location parameter <c>μ</c> (finite). / 位置パラメーター <c>μ</c>（有限）。</param>
    /// <param name="scale">The scale parameter <c>σ</c> (positive and finite). / 尺度パラメーター <c>σ</c>（正かつ有限）。</param>
    /// <exception cref="ArgumentNullException"><paramref name="distribution"/> is <see langword="null"/>. / <paramref name="distribution"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="location"/> is not finite, or <paramref name="scale"/> is not positive and finite. / <paramref name="location"/> が有限でない、または <paramref name="scale"/> が正の有限値でない場合。</exception>
    public LocationScaleDistribution(TDistribution distribution, double location, double scale)
    {
        ArgumentNullException.ThrowIfNull(distribution);
        DistributionArgument.Finite(location, nameof(location));
        DistributionArgument.PositiveFinite(scale, nameof(scale));
        _distribution = distribution;
        _location = location;
        _scale = scale;
    }

    /// <summary>
    /// Gets the underlying distribution.
    /// <para>元の連続分布を取得します。</para>
    /// </summary>
    public TDistribution Distribution => _distribution;

    /// <summary>
    /// Gets the location parameter <c>μ</c>.
    /// <para>位置パラメーター <c>μ</c> を取得します。</para>
    /// </summary>
    public double Location => _location;

    /// <summary>
    /// Gets the scale parameter <c>σ</c>.
    /// <para>尺度パラメーター <c>σ</c> を取得します。</para>
    /// </summary>
    public double Scale => _scale;

    /// <inheritdoc/>
    public double Mean => Unstandardize(_distribution.Mean);

    /// <inheritdoc/>
    public double Variance => _scale * (_scale * _distribution.Variance);

    /// <inheritdoc/>
    public double StandardDeviation => _scale * _distribution.StandardDeviation;

    /// <inheritdoc/>
    public double Skewness => _distribution.Skewness;

    /// <inheritdoc/>
    public double ExcessKurtosis => _distribution.ExcessKurtosis;

    /// <inheritdoc/>
    public double Entropy => _distribution.Entropy + Math.Log(_scale);

    /// <inheritdoc/>
    public double Median => Unstandardize(_distribution.Median);

    /// <inheritdoc/>
    public double SupportMinimum => Unstandardize(_distribution.SupportMinimum);

    /// <inheritdoc/>
    public double SupportMaximum => Unstandardize(_distribution.SupportMaximum);

    private double Standardize(double x) => DistributionMath.Standardize(x, _location, _scale);

    private double Unstandardize(double z) => DistributionMath.Unstandardize(_location, _scale, z);

    /// <inheritdoc/>
    public double Pdf(double x)
    {
        double z = Standardize(x);
        double f = _distribution.Pdf(z);
        double density = f / _scale;
        if (f >= DistributionMath.MinNormal && density >= DistributionMath.MinNormal && density < double.PositiveInfinity)
        {
            return density;
        }

        // Fall back to log-space when f or f/σ is subnormal/underflows (e.g., tiny σ compensating for tiny f), preserving ∞ and NaN.
        return f < double.PositiveInfinity ? Math.Exp(_distribution.LogPdf(z) - Math.Log(_scale)) : density;
    }

    /// <inheritdoc/>
    public double LogPdf(double x) => _distribution.LogPdf(Standardize(x)) - Math.Log(_scale);

    /// <inheritdoc/>
    public double Cdf(double x) => _distribution.Cdf(Standardize(x));

    /// <inheritdoc/>
    public double Sf(double x) => _distribution.Sf(Standardize(x));

    /// <inheritdoc/>
    public double LogCdf(double x) => _distribution.LogCdf(Standardize(x));

    /// <inheritdoc/>
    public double LogSf(double x) => _distribution.LogSf(Standardize(x));

    /// <inheritdoc/>
    public double InverseCdf(double p) => Unstandardize(_distribution.InverseCdf(p));

    /// <inheritdoc/>
    public double InverseSf(double q) => Unstandardize(_distribution.InverseSf(q));

    /// <inheritdoc/>
    public double InverseLogCdf(double logP) => Unstandardize(_distribution.InverseLogCdf(logP));

    /// <inheritdoc/>
    public double InverseLogSf(double logQ) => Unstandardize(_distribution.InverseLogSf(logQ));

    /// <inheritdoc/>
    public double Sample(IRandomSource random)
    {
        ArgumentNullException.ThrowIfNull(random);
        return Unstandardize(_distribution.Sample(random));
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, Span<double> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        _distribution.Sample(random, destination);
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = Unstandardize(destination[i]);
        }
    }
}
