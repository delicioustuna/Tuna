using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the logistic distribution with location <c>μ</c> and scale <c>s</c>.
/// <para>位置 <c>μ</c>、尺度 <c>s</c> を持つロジスティック分布を表します。</para>
/// </summary>
public readonly partial struct LogisticDistribution : IContinuousDistribution
{
    private readonly double _location;
    private readonly double _scale;

    /// <summary>
    /// Initializes a new instance of <see cref="LogisticDistribution"/> with the specified location and scale.
    /// <para>位置および尺度パラメーターを指定して <see cref="LogisticDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="location">The location parameter <c>μ</c> (finite). / 位置パラメーター <c>μ</c>（有限）。</param>
    /// <param name="scale">The scale parameter <c>s</c> (positive and finite). / 尺度パラメーター <c>s</c>（正かつ有限）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="location"/> is not finite, or <paramref name="scale"/> is not positive and finite. / <paramref name="location"/> が有限でない、または <paramref name="scale"/> が正の有限値でない場合。</exception>
    public LogisticDistribution(double location, double scale)
        : this(location, scale, forDifferentiation: false)
    {
    }

    private LogisticDistribution(double location, double scale, bool forDifferentiation)
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
    /// Gets the scale parameter <c>s</c>.
    /// <para>尺度パラメーター <c>s</c> を取得します。</para>
    /// </summary>
    public double Scale => _scale;

    /// <inheritdoc/>
    public double Mean => _location;

    /// <inheritdoc/>
    public double Variance => Math.PI * Math.PI / 3 * _scale * _scale;

    /// <inheritdoc/>
    public double StandardDeviation => Math.PI / XMath.Constants.Sqrt3 * _scale;

    /// <inheritdoc/>
    public double Skewness => 0;

    /// <inheritdoc/>
    public double ExcessKurtosis => 1.2;

    /// <inheritdoc/>
    public double Entropy => Math.Log(_scale) + 2;

    /// <inheritdoc/>
    public double Median => _location;

    /// <inheritdoc/>
    public double SupportMinimum => double.NegativeInfinity;

    /// <inheritdoc/>
    public double SupportMaximum => double.PositiveInfinity;

    private double Standardize(double x) => DistributionMath.Standardize(x, _location, _scale);

    /// <inheritdoc/>
    public double Pdf(double x)
    {
        double e = Math.Exp(-Math.Abs(Standardize(x)));
        double denominator = 1 + e;
        return e / (denominator * denominator) / _scale;
    }

    /// <inheritdoc/>
    public double LogPdf(double x)
    {
        double t = Math.Abs(Standardize(x));
        return -t - 2 * DistributionMath.Log1p(Math.Exp(-t)) - Math.Log(_scale);
    }

    /// <inheritdoc/>
    public double Cdf(double x) => StandardCdf(Standardize(x));

    /// <inheritdoc/>
    public double Sf(double x) => StandardCdf(-Standardize(x));

    /// <inheritdoc/>
    public double LogCdf(double x) => -DistributionMath.Log1pExp(-Standardize(x));

    /// <inheritdoc/>
    public double LogSf(double x) => -DistributionMath.Log1pExp(Standardize(x));

    private static double StandardCdf(double t)
    {
        if (t < 0)
        {
            double e = Math.Exp(t);
            return e / (1 + e);
        }

        return 1 / (1 + Math.Exp(-t));
    }

    /// <inheritdoc/>
    public double InverseCdf(double p) => p >= 0 && p <= 1 ? DistributionMath.Unstandardize(_location, _scale, DistributionMath.Logit(p)) : double.NaN;

    /// <inheritdoc/>
    public double InverseSf(double q) => q >= 0 && q <= 1 ? DistributionMath.Unstandardize(_location, _scale, -DistributionMath.Logit(q)) : double.NaN;

    /// <inheritdoc/>
    public double InverseLogCdf(double logP) =>
        logP <= 0 ? DistributionMath.Unstandardize(_location, _scale, logP - DistributionMath.Log1mExp(logP)) : double.NaN;

    /// <inheritdoc/>
    public double InverseLogSf(double logQ) =>
        logQ <= 0 ? DistributionMath.Unstandardize(_location, _scale, -(logQ - DistributionMath.Log1mExp(logQ))) : double.NaN;

    private double SampleCore(IRandomSource random) => DistributionMath.Unstandardize(_location, _scale, DistributionMath.Logit(random.NextDoubleOpen()));

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
