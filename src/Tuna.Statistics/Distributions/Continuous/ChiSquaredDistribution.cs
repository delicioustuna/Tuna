using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the chi-squared distribution <c>χ²(ν)</c> (equivalent to a gamma distribution with shape <c>ν/2</c> and scale 2).
/// <para>カイ二乗分布 <c>χ²(ν)</c>（形状 <c>ν/2</c>、尺度 2 のガンマ分布と同値）を表します。</para>
/// </summary>
public readonly partial struct ChiSquaredDistribution : IContinuousDistribution
{
    private readonly GammaDistribution _gamma;

    /// <summary>
    /// Initializes a new instance of <see cref="ChiSquaredDistribution"/> with the specified degrees of freedom.
    /// <para>自由度を指定して <see cref="ChiSquaredDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="degreesOfFreedom">The degrees of freedom <c>ν</c> (positive and finite; need not be an integer). / 自由度 <c>ν</c>（正かつ有限。整数でなくてもよい）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="degreesOfFreedom"/> is not positive and finite. / <paramref name="degreesOfFreedom"/> が正の有限値でない場合。</exception>
    public ChiSquaredDistribution(double degreesOfFreedom)
        : this(degreesOfFreedom, forDifferentiation: false)
    {
    }

    private ChiSquaredDistribution(double degreesOfFreedom, bool forDifferentiation)
    {
        DistributionArgument.PositiveFinite(degreesOfFreedom, nameof(degreesOfFreedom), forDifferentiation);
        _gamma = new GammaDistribution(0.5 * degreesOfFreedom, 2, forDifferentiation);
    }

    /// <summary>
    /// Gets the degrees of freedom <c>ν</c>.
    /// <para>自由度 <c>ν</c> を取得します。</para>
    /// </summary>
    public double DegreesOfFreedom => 2 * _gamma.Shape;

    /// <inheritdoc/>
    public double Mean => _gamma.Mean;

    /// <inheritdoc/>
    public double Variance => _gamma.Variance;

    /// <inheritdoc/>
    public double StandardDeviation => _gamma.StandardDeviation;

    /// <inheritdoc/>
    public double Skewness => _gamma.Skewness;

    /// <inheritdoc/>
    public double ExcessKurtosis => _gamma.ExcessKurtosis;

    /// <inheritdoc/>
    public double Entropy => _gamma.Entropy;

    /// <inheritdoc/>
    public double Median => _gamma.Median;

    /// <inheritdoc/>
    public double SupportMinimum => 0;

    /// <inheritdoc/>
    public double SupportMaximum => double.PositiveInfinity;

    /// <inheritdoc/>
    public double Pdf(double x) => _gamma.Pdf(x);

    /// <inheritdoc/>
    public double LogPdf(double x) => _gamma.LogPdf(x);

    /// <inheritdoc/>
    public double Cdf(double x) => _gamma.Cdf(x);

    /// <inheritdoc/>
    public double Sf(double x) => _gamma.Sf(x);

    /// <inheritdoc/>
    public double LogCdf(double x) => _gamma.LogCdf(x);

    /// <inheritdoc/>
    public double LogSf(double x) => _gamma.LogSf(x);

    /// <inheritdoc/>
    public double InverseCdf(double p) => _gamma.InverseCdf(p);

    /// <inheritdoc/>
    public double InverseSf(double q) => _gamma.InverseSf(q);

    /// <inheritdoc/>
    public double InverseLogCdf(double logP) => _gamma.InverseLogCdf(logP);

    /// <inheritdoc/>
    public double InverseLogSf(double logQ) => _gamma.InverseLogSf(logQ);

    /// <inheritdoc/>
    public double Sample(IRandomSource random) => _gamma.Sample(random);

    /// <inheritdoc/>
    public void Sample(IRandomSource random, Span<double> destination) => _gamma.Sample(random, destination);
}
