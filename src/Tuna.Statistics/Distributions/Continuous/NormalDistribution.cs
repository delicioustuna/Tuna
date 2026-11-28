using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the normal (Gaussian) distribution <c>N(μ, σ²)</c>.
/// <para>正規分布（ガウス分布）<c>N(μ, σ²)</c> を表します。</para>
/// </summary>
public readonly partial struct NormalDistribution : IContinuousDistribution
{
    private readonly double _mean;
    private readonly double _standardDeviation;

    /// <summary>
    /// Initializes a new instance of <see cref="NormalDistribution"/> with the specified mean and standard deviation.
    /// <para>平均と標準偏差を指定して <see cref="NormalDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="mean">The mean <c>μ</c> (finite). / 平均 <c>μ</c>（有限）。</param>
    /// <param name="standardDeviation">The standard deviation <c>σ</c> (positive and finite). / 標準偏差 <c>σ</c>（正かつ有限）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="mean"/> is not finite, or <paramref name="standardDeviation"/> is not positive and finite. / <paramref name="mean"/> が有限でない、または <paramref name="standardDeviation"/> が正の有限値でない場合。</exception>
    public NormalDistribution(double mean, double standardDeviation)
        : this(mean, standardDeviation, forDifferentiation: false)
    {
    }

    private NormalDistribution(double mean, double standardDeviation, bool forDifferentiation)
    {
        DistributionArgument.Finite(mean, nameof(mean), forDifferentiation);
        DistributionArgument.PositiveFinite(standardDeviation, nameof(standardDeviation), forDifferentiation);
        _mean = mean;
        _standardDeviation = standardDeviation;
    }

    /// <summary>
    /// Gets the standard normal distribution <c>N(0, 1)</c>.
    /// <para>標準正規分布 <c>N(0, 1)</c> を取得します。</para>
    /// </summary>
    public static NormalDistribution Standard => new(0, 1);

    /// <inheritdoc/>
    public double Mean => _mean;

    /// <inheritdoc/>
    public double StandardDeviation => _standardDeviation;

    /// <inheritdoc/>
    public double Variance => _standardDeviation * _standardDeviation;

    /// <inheritdoc/>
    public double Skewness => 0;

    /// <inheritdoc/>
    public double ExcessKurtosis => 0;

    /// <inheritdoc/>
    public double Entropy => 0.5 + XMath.Constants.LnSqrtTwoPi + Math.Log(_standardDeviation);

    /// <inheritdoc/>
    public double Median => _mean;

    /// <inheritdoc/>
    public double SupportMinimum => double.NegativeInfinity;

    /// <inheritdoc/>
    public double SupportMaximum => double.PositiveInfinity;

    private double Standardize(double x) => DistributionMath.Standardize(x, _mean, _standardDeviation);

    /// <inheritdoc/>
    public double Pdf(double x) => XMath.Special.NormalPdf(Standardize(x)) / _standardDeviation;

    /// <inheritdoc/>
    public double LogPdf(double x) => LogPdfValue(Standardize(x), Math.Log(_standardDeviation));

    private static double LogPdfValue(double z, double logScale) => XMath.Special.NormalLogPdf(z) - logScale;

    /// <inheritdoc/>
    public double Cdf(double x) => XMath.Special.NormalCdf(Standardize(x));

    /// <inheritdoc/>
    public double Sf(double x) => XMath.Special.NormalSf(Standardize(x));

    /// <inheritdoc/>
    public double LogCdf(double x) => XMath.Special.NormalLogCdf(Standardize(x));

    /// <inheritdoc/>
    public double LogSf(double x) => XMath.Special.NormalLogSf(Standardize(x));

    /// <inheritdoc/>
    public double InverseCdf(double p) => DistributionMath.Unstandardize(_mean, _standardDeviation, XMath.Special.InverseNormalCdf(p));

    /// <inheritdoc/>
    public double InverseSf(double q) => DistributionMath.Unstandardize(_mean, _standardDeviation, XMath.Special.InverseNormalSf(q));

    /// <inheritdoc/>
    public double InverseLogCdf(double logP) => DistributionMath.Unstandardize(_mean, _standardDeviation, XMath.Special.InverseNormalLogCdf(logP));

    /// <inheritdoc/>
    public double InverseLogSf(double logQ) => DistributionMath.Unstandardize(_mean, _standardDeviation, XMath.Special.InverseNormalLogSf(logQ));

    /// <inheritdoc/>
    public double Sample(IRandomSource random)
    {
        ArgumentNullException.ThrowIfNull(random);
        return DistributionMath.Unstandardize(_mean, _standardDeviation, Ziggurat.StandardNormal(random));
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, Span<double> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = DistributionMath.Unstandardize(_mean, _standardDeviation, Ziggurat.StandardNormal(random));
        }
    }
}
