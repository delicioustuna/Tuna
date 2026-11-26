using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the Cauchy distribution with location <c>x₀</c> and scale <c>γ</c>.
/// <para>位置 <c>x₀</c>、尺度 <c>γ</c> を持つコーシー分布を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// The mean, variance, standard deviation, skewness, and excess kurtosis are undefined and return <see cref="double.NaN"/>.
/// </para>
/// <para>
/// 平均、分散、標準偏差、歪度、および超過尖度は定義されないため <see cref="double.NaN"/> を返します。
/// </para>
/// </remarks>
public readonly partial struct CauchyDistribution : IContinuousDistribution
{
    private readonly double _location;
    private readonly double _scale;

    /// <summary>
    /// Initializes a new instance of <see cref="CauchyDistribution"/> with the specified location and scale.
    /// <para>位置および尺度パラメーターを指定して <see cref="CauchyDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="location">The location parameter <c>x₀</c> (finite). / 位置パラメーター <c>x₀</c>（有限）。</param>
    /// <param name="scale">The scale parameter <c>γ</c> (positive and finite). / 尺度パラメーター <c>γ</c>（正かつ有限）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="location"/> is not finite, or <paramref name="scale"/> is not positive and finite. / <paramref name="location"/> が有限でない、または <paramref name="scale"/> が正の有限値でない場合。</exception>
    public CauchyDistribution(double location, double scale)
        : this(location, scale, forDifferentiation: false)
    {
    }

    private CauchyDistribution(double location, double scale, bool forDifferentiation)
    {
        DistributionArgument.Finite(location, nameof(location), forDifferentiation);
        DistributionArgument.PositiveFinite(scale, nameof(scale), forDifferentiation);
        _location = location;
        _scale = scale;
    }

    /// <summary>
    /// Gets the location parameter <c>x₀</c>.
    /// <para>位置パラメーター <c>x₀</c> を取得します。</para>
    /// </summary>
    public double Location => _location;

    /// <summary>
    /// Gets the scale parameter <c>γ</c>.
    /// <para>尺度パラメーター <c>γ</c> を取得します。</para>
    /// </summary>
    public double Scale => _scale;

    /// <inheritdoc/>
    public double Mean => double.NaN;

    /// <inheritdoc/>
    public double Variance => double.NaN;

    /// <inheritdoc/>
    public double StandardDeviation => double.NaN;

    /// <inheritdoc/>
    public double Skewness => double.NaN;

    /// <inheritdoc/>
    public double ExcessKurtosis => double.NaN;

    /// <inheritdoc/>
    public double Entropy => Math.Log(4 * Math.PI) + Math.Log(_scale);

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
        double t = Math.Abs(Standardize(x));
        double peak = 1 / (Math.PI * _scale);
        if (peak < DistributionMath.MinNormal)
        {
            // When πγ overflows or 1/(πγ) is subnormal, compute standard density first and divide by γ once.
            double standard = t > 1e150 ? 1 / Math.PI / t / t : 1 / (Math.PI * (1 + t * t));
            return standard / _scale;
        }

        return t > 1e150 ? peak / t / t : peak / (1 + t * t);
    }

    /// <inheritdoc/>
    public double LogPdf(double x)
    {
        return LogPdfValue(Standardize(x), -DistributionMath.LnPi - Math.Log(_scale));
    }

    private static double LogPdfValue(double t, double constant) => constant - LogOnePlusSquare(Math.Abs(t));

    // Overflow-safe ln(1 + t²) for t >= 0.
    private static double LogOnePlusSquare(double t) =>
        t > 1e150 ? 2 * Math.Log(t) : t > 1 ? 2 * Math.Log(t) + DistributionMath.Log1p(1 / (t * t)) : DistributionMath.Log1p(t * t);

    /// <inheritdoc/>
    public double Cdf(double x) => Math.Atan2(1, -Standardize(x)) / Math.PI;

    /// <inheritdoc/>
    public double Sf(double x) => Math.Atan2(1, Standardize(x)) / Math.PI;

    /// <inheritdoc/>
    public double LogCdf(double x)
    {
        double t = Standardize(x);
        return t <= 0 ? Math.Log(Math.Atan2(1, -t) / Math.PI) : DistributionMath.Log1p(-Math.Atan2(1, t) / Math.PI);
    }

    /// <inheritdoc/>
    public double LogSf(double x)
    {
        double t = Standardize(x);
        return t >= 0 ? Math.Log(Math.Atan2(1, t) / Math.PI) : DistributionMath.Log1p(-Math.Atan2(1, -t) / Math.PI);
    }

    private static double StandardQuantile(double p)
    {
        if (p < 0.25)
        {
            // Explicit zero check so p = -0.0 (from -expm1(0)) still returns -∞.
            return p == 0 ? double.NegativeInfinity : -1 / double.TanPi(p);
        }

        if (p > 0.75)
        {
            return 1 / double.TanPi(1 - p);
        }

        return double.TanPi(p - 0.5);
    }

    private double Quantile(double p, bool upper)
    {
        if (p > 0 && p < DistributionMath.MinNormal)
        {
            // For subnormal p, tan(πp) = πp within relative ε; scale by 2^64 before unstandardizing to avoid intermediate overflow.
            double m = -1 / (Math.PI * Math.ScaleB(p, 64));
            return DistributionMath.UnstandardizeScaled(_location, _scale, upper ? -m : m, 64);
        }

        double z = StandardQuantile(p);
        return DistributionMath.Unstandardize(_location, _scale, upper ? -z : z);
    }

    /// <inheritdoc/>
    public double InverseCdf(double p) => p >= 0 && p <= 1 ? Quantile(p, false) : double.NaN;

    /// <inheritdoc/>
    public double InverseSf(double q) => q >= 0 && q <= 1 ? Quantile(q, true) : double.NaN;

    /// <inheritdoc/>
    public double InverseLogCdf(double logP)
    {
        if (!(logP <= 0))
        {
            return double.NaN;
        }

        if (logP > -DistributionMath.Ln2)
        {
            return InverseSf(-DistributionMath.Expm1(logP));
        }

        if (logP >= DistributionMath.LogProbabilityLinearLimit)
        {
            return InverseCdf(Math.Exp(logP));
        }

        // Asymptotic tail -1/tan(πp) ≈ -1/(πp) evaluated in log-space to avoid overflow when scale is small.
        return DistributionMath.UnstandardizeFromLog(_location, _scale, true, -DistributionMath.LnPi - logP);
    }

    /// <inheritdoc/>
    public double InverseLogSf(double logQ)
    {
        if (!(logQ <= 0))
        {
            return double.NaN;
        }

        if (logQ > -DistributionMath.Ln2)
        {
            return InverseCdf(-DistributionMath.Expm1(logQ));
        }

        if (logQ >= DistributionMath.LogProbabilityLinearLimit)
        {
            return InverseSf(Math.Exp(logQ));
        }

        return DistributionMath.UnstandardizeFromLog(_location, _scale, false, -DistributionMath.LnPi - logQ);
    }

    /// <inheritdoc/>
    public double Sample(IRandomSource random)
    {
        ArgumentNullException.ThrowIfNull(random);
        return DistributionMath.Unstandardize(_location, _scale, double.TanPi(random.NextDoubleOpen() - 0.5));
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, Span<double> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = DistributionMath.Unstandardize(_location, _scale, double.TanPi(random.NextDoubleOpen() - 0.5));
        }
    }
}
