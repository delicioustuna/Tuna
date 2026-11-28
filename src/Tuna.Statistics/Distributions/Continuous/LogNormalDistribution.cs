using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the log-normal distribution where <c>ln X ~ N(μ, σ²)</c>.
/// <para><c>ln X ~ N(μ, σ²)</c> を満たす対数正規分布を表します。</para>
/// </summary>
public readonly partial struct LogNormalDistribution : IContinuousDistribution
{
    private readonly double _mu;
    private readonly double _sigma;

    /// <summary>
    /// Initializes a new instance of <see cref="LogNormalDistribution"/> with the mean and standard deviation of the underlying normal distribution.
    /// <para>対数変換後の正規分布の平均と標準偏差を指定して <see cref="LogNormalDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="mu">The mean <c>μ</c> of <c>ln X</c> (finite). / <c>ln X</c> の平均 <c>μ</c>（有限）。</param>
    /// <param name="sigma">The standard deviation <c>σ</c> of <c>ln X</c> (positive and finite). / <c>ln X</c> の標準偏差 <c>σ</c>（正かつ有限）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="mu"/> is not finite, or <paramref name="sigma"/> is not positive and finite. / <paramref name="mu"/> が有限でない、または <paramref name="sigma"/> が正の有限値でない場合。</exception>
    public LogNormalDistribution(double mu, double sigma)
        : this(mu, sigma, forDifferentiation: false)
    {
    }

    private LogNormalDistribution(double mu, double sigma, bool forDifferentiation)
    {
        DistributionArgument.Finite(mu, nameof(mu), forDifferentiation);
        DistributionArgument.PositiveFinite(sigma, nameof(sigma), forDifferentiation);
        _mu = mu;
        _sigma = sigma;
    }

    /// <summary>
    /// Gets the mean <c>μ</c> of <c>ln X</c>.
    /// <para><c>ln X</c> の平均 <c>μ</c> を取得します。</para>
    /// </summary>
    public double Mu => _mu;

    /// <summary>
    /// Gets the standard deviation <c>σ</c> of <c>ln X</c>.
    /// <para><c>ln X</c> の標準偏差 <c>σ</c> を取得します。</para>
    /// </summary>
    public double Sigma => _sigma;

    /// <inheritdoc/>
    public double Mean => Math.Exp(_mu + 0.5 * _sigma * _sigma);

    /// <inheritdoc/>
    public double Variance
    {
        get
        {
            double s2 = _sigma * _sigma;
            return DistributionMath.Expm1(s2) * Math.Exp(2 * _mu + s2);
        }
    }

    /// <inheritdoc/>
    public double StandardDeviation
    {
        get
        {
            double s2 = _sigma * _sigma;
            return Math.Sqrt(DistributionMath.Expm1(s2)) * Math.Exp(_mu + 0.5 * s2);
        }
    }

    /// <inheritdoc/>
    public double Skewness
    {
        get
        {
            double w = DistributionMath.Expm1(_sigma * _sigma);
            return (w + 3) * Math.Sqrt(w);
        }
    }

    /// <inheritdoc/>
    public double ExcessKurtosis
    {
        get
        {
            // Express e^(4σ²) + 2e^(3σ²) + 3e^(2σ²) - 6 in w = expm1(σ²) to avoid cancellation for small σ.
            double w = DistributionMath.Expm1(_sigma * _sigma);
            return w * (16 + w * (15 + w * (6 + w)));
        }
    }

    /// <inheritdoc/>
    public double Entropy => _mu + 0.5 + XMath.Constants.LnSqrtTwoPi + Math.Log(_sigma);

    /// <inheritdoc/>
    public double Median => Math.Exp(_mu);

    /// <inheritdoc/>
    public double SupportMinimum => 0;

    /// <inheritdoc/>
    public double SupportMaximum => double.PositiveInfinity;

    private double Standardize(double x) => (Math.Log(x) - _mu) / _sigma;

    /// <inheritdoc/>
    public double Pdf(double x)
    {
        if (double.IsNaN(x))
        {
            return double.NaN;
        }

        if (x <= 0 || double.IsPositiveInfinity(x))
        {
            return 0;
        }

        // Fall back to log-space when NormalPdf underflows even though dividing by σx recovers a normal float.
        double value = XMath.Special.NormalPdf(Standardize(x)) / _sigma / x;
        return value >= NumericConstants.MinNormal && value < double.PositiveInfinity ? value : Math.Exp(LogPdf(x));
    }

    /// <inheritdoc/>
    public double LogPdf(double x)
    {
        if (double.IsNaN(x))
        {
            return double.NaN;
        }

        if (x <= 0 || double.IsPositiveInfinity(x))
        {
            return double.NegativeInfinity;
        }

        double logX = Math.Log(x);
        return XMath.Special.NormalLogPdf((logX - _mu) / _sigma) - Math.Log(_sigma) - logX;
    }

    /// <inheritdoc/>
    public double Cdf(double x) => x <= 0 ? 0 : XMath.Special.NormalCdf(Standardize(x));

    /// <inheritdoc/>
    public double Sf(double x) => x <= 0 ? 1 : XMath.Special.NormalSf(Standardize(x));

    /// <inheritdoc/>
    public double LogCdf(double x) => x <= 0 ? double.NegativeInfinity : XMath.Special.NormalLogCdf(Standardize(x));

    /// <inheritdoc/>
    public double LogSf(double x) => x <= 0 ? 0 : XMath.Special.NormalLogSf(Standardize(x));

    /// <inheritdoc/>
    public double InverseCdf(double p) => Math.Exp(_mu + _sigma * XMath.Special.InverseNormalCdf(p));

    /// <inheritdoc/>
    public double InverseSf(double q) => Math.Exp(_mu + _sigma * XMath.Special.InverseNormalSf(q));

    /// <inheritdoc/>
    public double InverseLogCdf(double logP) => Math.Exp(_mu + _sigma * XMath.Special.InverseNormalLogCdf(logP));

    /// <inheritdoc/>
    public double InverseLogSf(double logQ) => Math.Exp(_mu + _sigma * XMath.Special.InverseNormalLogSf(logQ));

    /// <inheritdoc/>
    public double Sample(IRandomSource random)
    {
        ArgumentNullException.ThrowIfNull(random);
        return Math.Exp(_mu + _sigma * Ziggurat.StandardNormal(random));
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, Span<double> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = Math.Exp(_mu + _sigma * Ziggurat.StandardNormal(random));
        }
    }
}
