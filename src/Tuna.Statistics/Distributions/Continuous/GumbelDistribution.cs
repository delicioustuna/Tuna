using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the Gumbel (Type I extreme value for maxima) distribution with location <c>μ</c> and scale <c>β</c>.
/// <para>位置 <c>μ</c>、尺度 <c>β</c> を持つガンベル分布（最大値の I 型極値分布）を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// This is the right-skewed maximum extreme value distribution with CDF <c>exp(-exp(-(x - μ)/β))</c>.
/// The minimum extreme value distribution corresponds to <c>-X</c>.
/// </para>
/// <para>
/// 累積分布関数が <c>exp(-exp(-(x - μ)/β))</c> で与えられる右歪みの最大値極値分布です。
/// 最小値の極値分布は <c>-X</c> に対応します。
/// </para>
/// </remarks>
public readonly partial struct GumbelDistribution : IContinuousDistribution
{
    // 12 * sqrt(6) * zeta(3) / pi^3
    private const double StandardSkewness = 1.1395470994046486;

    // -ln(ln 2)
    private const double StandardMedian = 0.36651292058166435;

    private readonly double _location;
    private readonly double _scale;

    /// <summary>
    /// Initializes a new instance of <see cref="GumbelDistribution"/> with the specified location and scale.
    /// <para>位置および尺度パラメーターを指定して <see cref="GumbelDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="location">The location parameter <c>μ</c> (finite; also the mode). / 位置パラメーター <c>μ</c>（有限、最頻値）。</param>
    /// <param name="scale">The scale parameter <c>β</c> (positive and finite). / 尺度パラメーター <c>β</c>（正かつ有限）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="location"/> is not finite, or <paramref name="scale"/> is not positive and finite. / <paramref name="location"/> が有限でない、または <paramref name="scale"/> が正の有限値でない場合。</exception>
    public GumbelDistribution(double location, double scale)
        : this(location, scale, forDifferentiation: false)
    {
    }

    private GumbelDistribution(double location, double scale, bool forDifferentiation)
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
    /// Gets the scale parameter <c>β</c>.
    /// <para>尺度パラメーター <c>β</c> を取得します。</para>
    /// </summary>
    public double Scale => _scale;

    /// <inheritdoc/>
    public double Mean => DistributionMath.Unstandardize(_location, _scale, XMath.Special.EulerGamma);

    /// <inheritdoc/>
    public double Variance => 1.6449340668482264 * _scale * _scale;

    /// <inheritdoc/>
    public double StandardDeviation => 1.282549830161864 * _scale;

    /// <inheritdoc/>
    public double Skewness => StandardSkewness;

    /// <inheritdoc/>
    public double ExcessKurtosis => 2.4;

    /// <inheritdoc/>
    public double Entropy => Math.Log(_scale) + (XMath.Special.EulerGamma + 1);

    /// <inheritdoc/>
    public double Median => DistributionMath.Unstandardize(_location, _scale, StandardMedian);

    /// <inheritdoc/>
    public double SupportMinimum => double.NegativeInfinity;

    /// <inheritdoc/>
    public double SupportMaximum => double.PositiveInfinity;

    private double Standardize(double x) => DistributionMath.Standardize(x, _location, _scale);

    /// <inheritdoc/>
    public double Pdf(double x)
    {
        double z = Standardize(x);
        double t = Math.Exp(-z);
        // exp(-t) underflows to 0 for t > 746; guard avoids Inf * 0 = NaN when t = +Inf.
        double standard = t > 746 ? 0 : t * Math.Exp(-t);
        if (standard >= DistributionMath.MinNormal)
        {
            return standard / _scale;
        }

        return double.IsNaN(standard) ? double.NaN : Math.Exp(LogPdf(x));
    }

    /// <inheritdoc/>
    public double LogPdf(double x)
    {
        double z = Standardize(x);
        double t = Math.Exp(-z);
        return double.IsPositiveInfinity(t) ? double.NegativeInfinity : -z - t - Math.Log(_scale);
    }

    /// <inheritdoc/>
    public double Cdf(double x) => Math.Exp(-Math.Exp(-Standardize(x)));

    /// <inheritdoc/>
    public double Sf(double x) => -DistributionMath.Expm1(-Math.Exp(-Standardize(x)));

    /// <inheritdoc/>
    public double LogCdf(double x) => -Math.Exp(-Standardize(x));

    /// <inheritdoc/>
    public double LogSf(double x)
    {
        double z = Standardize(x);
        double t = Math.Exp(-z);
        // When t is subnormal, ln(1 - exp(-t)) = ln(t) + O(t) = -z within floating-point precision.
        return t >= DistributionMath.MinNormal ? DistributionMath.Log1mExp(-t) : -z;
    }

    private double FromNegativeLogP(double negativeLogP) =>
        DistributionMath.Unstandardize(_location, _scale, -Math.Log(negativeLogP));

    /// <inheritdoc/>
    public double InverseCdf(double p) => p >= 0 && p <= 1 ? FromNegativeLogP(-Math.Log(p)) : double.NaN;

    /// <inheritdoc/>
    public double InverseSf(double q) => q >= 0 && q <= 1 ? FromNegativeLogP(-DistributionMath.Log1p(-q)) : double.NaN;

    /// <inheritdoc/>
    public double InverseLogCdf(double logP) => logP <= 0 ? FromNegativeLogP(-logP) : double.NaN;

    /// <inheritdoc/>
    public double InverseLogSf(double logQ)
    {
        if (!(logQ <= 0))
        {
            return double.NaN;
        }

        // For logQ < -40, -ln(-ln(1 - e^logQ)) = -logQ - e^logQ/2 + ... agrees with -logQ within ε even if e^logQ underflows.
        return logQ < -40
            ? DistributionMath.Unstandardize(_location, _scale, -logQ)
            : FromNegativeLogP(-DistributionMath.Log1mExp(logQ));
    }

    private double SampleCore(IRandomSource random) => FromNegativeLogP(-Math.Log(random.NextDoubleOpen()));

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
