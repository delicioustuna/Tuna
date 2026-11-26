using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the Laplace (double exponential) distribution with location <c>μ</c> and scale <c>b</c>.
/// <para>位置 <c>μ</c>、尺度 <c>b</c> を持つラプラス分布（二重指数分布）を表します。</para>
/// </summary>
public readonly partial struct LaplaceDistribution : IContinuousDistribution
{
    private readonly double _location;
    private readonly double _scale;

    /// <summary>
    /// Initializes a new instance of <see cref="LaplaceDistribution"/> with the specified location and scale.
    /// <para>位置および尺度パラメーターを指定して <see cref="LaplaceDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="location">The location parameter <c>μ</c> (finite). / 位置パラメーター <c>μ</c>（有限）。</param>
    /// <param name="scale">The scale parameter <c>b</c> (positive and finite). / 尺度パラメーター <c>b</c>（正かつ有限）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="location"/> is not finite, or <paramref name="scale"/> is not positive and finite. / <paramref name="location"/> が有限でない、または <paramref name="scale"/> が正の有限値でない場合。</exception>
    public LaplaceDistribution(double location, double scale)
        : this(location, scale, forDifferentiation: false)
    {
    }

    private LaplaceDistribution(double location, double scale, bool forDifferentiation)
    {
        DistributionArgument.Finite(location, nameof(location), forDifferentiation);
        DistributionArgument.PositiveFinite(scale, nameof(scale), forDifferentiation);
        _location = location;
        _scale = scale;
    }

    /// <summary>
    /// Gets the location parameter <c>μ</c>.
    /// <para>位置パラメーター <c>μ</c> を取得します。</para>
    /// </summary>
    public double Location => _location;

    /// <summary>
    /// Gets the scale parameter <c>b</c>.
    /// <para>尺度パラメーター <c>b</c> を取得します。</para>
    /// </summary>
    public double Scale => _scale;

    /// <inheritdoc/>
    public double Mean => _location;

    /// <inheritdoc/>
    public double Variance => 2 * _scale * _scale;

    /// <inheritdoc/>
    public double StandardDeviation => Math.Sqrt(2) * _scale;

    /// <inheritdoc/>
    public double Skewness => 0;

    /// <inheritdoc/>
    public double ExcessKurtosis => 3;

    /// <inheritdoc/>
    public double Entropy => 1 + DistributionMath.Ln2 + Math.Log(_scale);

    /// <inheritdoc/>
    public double Median => _location;

    /// <inheritdoc/>
    public double SupportMinimum => double.NegativeInfinity;

    /// <inheritdoc/>
    public double SupportMaximum => double.PositiveInfinity;

    private double Standardize(double x) => DistributionMath.Standardize(x, _location, _scale);

    /// <inheritdoc/>
    public double Pdf(double x) => 0.5 * Math.Exp(-Math.Abs(Standardize(x))) / _scale;

    /// <inheritdoc/>
    public double LogPdf(double x) => -Math.Abs(Standardize(x)) - DistributionMath.Ln2 - Math.Log(_scale);

    /// <inheritdoc/>
    public double Cdf(double x)
    {
        double t = Standardize(x);
        return t < 0 ? 0.5 * Math.Exp(t) : 1 - 0.5 * Math.Exp(-t);
    }

    /// <inheritdoc/>
    public double Sf(double x)
    {
        double t = Standardize(x);
        return t > 0 ? 0.5 * Math.Exp(-t) : 1 - 0.5 * Math.Exp(t);
    }

    /// <inheritdoc/>
    public double LogCdf(double x)
    {
        double t = Standardize(x);
        return t < 0 ? t - DistributionMath.Ln2 : DistributionMath.Log1p(-0.5 * Math.Exp(-t));
    }

    /// <inheritdoc/>
    public double LogSf(double x)
    {
        double t = Standardize(x);
        return t > 0 ? -t - DistributionMath.Ln2 : DistributionMath.Log1p(-0.5 * Math.Exp(t));
    }

    /// <inheritdoc/>
    public double InverseCdf(double p)
    {
        if (!(p >= 0 && p <= 1))
        {
            return double.NaN;
        }

        return p <= 0.5
            ? DistributionMath.Unstandardize(_location, _scale, Math.Log(2 * p))
            : DistributionMath.Unstandardize(_location, _scale, -Math.Log(2 * (1 - p)));
    }

    /// <inheritdoc/>
    public double InverseSf(double q)
    {
        if (!(q >= 0 && q <= 1))
        {
            return double.NaN;
        }

        return q <= 0.5
            ? DistributionMath.Unstandardize(_location, _scale, -Math.Log(2 * q))
            : DistributionMath.Unstandardize(_location, _scale, Math.Log(2 * (1 - q)));
    }

    /// <inheritdoc/>
    public double InverseLogCdf(double logP)
    {
        if (!(logP <= 0))
        {
            return double.NaN;
        }

        return logP <= -DistributionMath.Ln2
            ? DistributionMath.Unstandardize(_location, _scale, logP + DistributionMath.Ln2)
            : DistributionMath.Unstandardize(_location, _scale, -(DistributionMath.Ln2 + DistributionMath.Log1mExp(logP)));
    }

    /// <inheritdoc/>
    public double InverseLogSf(double logQ)
    {
        if (!(logQ <= 0))
        {
            return double.NaN;
        }

        return logQ <= -DistributionMath.Ln2
            ? DistributionMath.Unstandardize(_location, _scale, -(logQ + DistributionMath.Ln2))
            : DistributionMath.Unstandardize(_location, _scale, DistributionMath.Ln2 + DistributionMath.Log1mExp(logQ));
    }

    private double SampleCore(IRandomSource random)
    {
        double u = random.NextDoubleOpen();
        return u < 0.5
            ? DistributionMath.Unstandardize(_location, _scale, Math.Log(2 * u))
            : DistributionMath.Unstandardize(_location, _scale, -Math.Log(2 * (1 - u)));
    }

    /// <inheritdoc/>
    public double Sample(IRandomSource random)
    {
        ArgumentNullException.ThrowIfNull(random);
        return SampleCore(random);
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, Span<double> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = SampleCore(random);
        }
    }
}
