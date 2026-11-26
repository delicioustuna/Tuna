using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the Poisson distribution with rate (mean) <c>λ</c>.
/// <para>平均（発生率）<c>λ</c> を持つポアソン分布を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// When <c>λ = 0</c>, the distribution is degenerate at 0, and <see cref="Skewness"/> and <see cref="ExcessKurtosis"/> return <see cref="double.NaN"/>.
/// </para>
/// <para>
/// <c>λ = 0</c> の場合は常に 0 をとる退化した分布となり、<see cref="Skewness"/> および <see cref="ExcessKurtosis"/> は <see cref="double.NaN"/> を返します。
/// </para>
/// </remarks>
public readonly partial struct PoissonDistribution : IDiscreteDistribution
{
    /// <summary>
    /// The maximum supported rate (<c>10¹⁸</c>, keeping samples within <see cref="long"/> range).
    /// <para>サポートされる平均の上限（<c>10¹⁸</c>、標本が <see cref="long"/> に収まる範囲）。</para>
    /// </summary>
    public const double MaxRate = Sampling.MaxPoissonRate;

    private readonly double _rate;

    /// <summary>
    /// Initializes a new instance of <see cref="PoissonDistribution"/> with the specified rate.
    /// <para>平均（率）パラメーターを指定して <see cref="PoissonDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="rate">The rate (mean) <c>λ</c> (<c>0 ≤ λ ≤ 10¹⁸</c>; 0 yields a degenerate distribution at 0). / 平均 <c>λ</c>（<c>0 ≤ λ ≤ 10¹⁸</c>。0 は常に 0 をとる退化分布）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rate"/> is <see cref="double.NaN"/> or outside <c>[0, MaxRate]</c>. / <paramref name="rate"/> が <see cref="double.NaN"/> または <c>[0, MaxRate]</c> の範囲外の場合。</exception>
    public PoissonDistribution(double rate)
        : this(rate, forDifferentiation: false)
    {
    }

    private PoissonDistribution(double rate, bool forDifferentiation)
    {
        DistributionArgument.NonNegativeFinite(rate, nameof(rate), forDifferentiation);
        if (forDifferentiation && rate > MaxRate)
        {
            throw new DistributionParameterOutOfRangeException(nameof(rate), rate, "The rate must not exceed MaxRate.");
        }

        ArgumentOutOfRangeException.ThrowIfGreaterThan(rate, MaxRate);
        _rate = rate;
    }

    /// <summary>
    /// Gets the rate (mean) parameter <c>λ</c>.
    /// <para>平均（率）パラメーター <c>λ</c> を取得します。</para>
    /// </summary>
    public double Rate => _rate;

    /// <inheritdoc/>
    public double Mean => _rate;

    /// <inheritdoc/>
    public double Variance => _rate;

    /// <inheritdoc/>
    public double StandardDeviation => Math.Sqrt(_rate);

    /// <inheritdoc/>
    public double Skewness => _rate > 0 ? 1 / Math.Sqrt(_rate) : double.NaN;

    /// <inheritdoc/>
    public double ExcessKurtosis => _rate > 0 ? 1 / _rate : double.NaN;

    /// <inheritdoc/>
    public double Entropy
    {
        get
        {
            if (_rate == 0)
            {
                return 0;
            }

            if (1 / _rate <= DiscreteMath.EntropyExpansionLimit)
            {
                return DiscreteMath.EntropyExpansion(Math.Log(_rate), Skewness, ExcessKurtosis);
            }

            var self = this;
            return DiscreteMath.EntropySum(self.LogPmf, (long)Math.Floor(_rate), 0, long.MaxValue);
        }
    }

    /// <inheritdoc/>
    public long Median => InverseCdf(0.5);

    /// <inheritdoc/>
    public long SupportMinimum => 0;

    /// <inheritdoc/>
    public long SupportMaximum => _rate == 0 ? 0 : long.MaxValue;

    /// <inheritdoc/>
    public double Pmf(long k) => Math.Exp(LogPmf(k));

    /// <inheritdoc/>
    public double LogPmf(long k) => k < 0 ? double.NegativeInfinity : DistributionMath.LogPoissonProbability(k, _rate);

    /// <summary>
    /// Evaluates the cumulative distribution function (CDF), <c>P(X ≤ k)</c>.
    /// <para>累積分布関数（CDF）<c>P(X ≤ k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns>The cumulative probability <c>P(X ≤ k)</c>. / 累積確率 <c>P(X ≤ k)</c>。</returns>
    public double Cdf(long k) => k < 0 ? 0 : XMath.Special.RegularizedGammaQ(k + 1.0, _rate);

    /// <summary>
    /// Evaluates the survival function, <c>P(X &gt; k)</c>.
    /// <para>生存関数（上側累積確率）<c>P(X &gt; k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns>The survival probability <c>P(X &gt; k)</c>. / 生存確率 <c>P(X &gt; k)</c>。</returns>
    public double Sf(long k) => k < 0 ? 1 : XMath.Special.RegularizedGammaP(k + 1.0, _rate);

    /// <summary>
    /// Evaluates the natural logarithm of the cumulative distribution function, <c>ln P(X ≤ k)</c>.
    /// <para>累積分布関数の自然対数 <c>ln P(X ≤ k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns><c>ln P(X ≤ k)</c>, or <see cref="double.NegativeInfinity"/> when the probability is 0. / <c>ln P(X ≤ k)</c>。確率が 0 の場合は <see cref="double.NegativeInfinity"/>。</returns>
    public double LogCdf(long k) => k < 0 ? double.NegativeInfinity : XMath.Special.LogRegularizedGammaQ(k + 1.0, _rate);

    /// <summary>
    /// Evaluates the natural logarithm of the survival function, <c>ln P(X &gt; k)</c>.
    /// <para>生存関数の自然対数 <c>ln P(X &gt; k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns><c>ln P(X &gt; k)</c>, or <see cref="double.NegativeInfinity"/> when the probability is 0. / <c>ln P(X &gt; k)</c>。確率が 0 の場合は <see cref="double.NegativeInfinity"/>。</returns>
    public double LogSf(long k) => k < 0 ? 0 : XMath.Special.LogRegularizedGammaP(k + 1.0, _rate);

    private long Guess(double z) =>
        _rate == 0 ? 0 : DiscreteMath.Guess(_rate, Math.Sqrt(_rate), 1 / Math.Sqrt(_rate), z, 0, long.MaxValue);

    /// <inheritdoc/>
    public long InverseCdf(double p) => DiscreteMath.InverseCdf(this, p, Guess(XMath.Special.InverseNormalCdf(p)));

    /// <inheritdoc/>
    public long InverseSf(double q) => DiscreteMath.InverseSf(this, q, Guess(XMath.Special.InverseNormalSf(q)));

    /// <inheritdoc/>
    public long InverseLogCdf(double logP) => DiscreteMath.InverseLogCdf(this, logP, Guess(XMath.Special.InverseNormalLogCdf(logP)));

    /// <inheritdoc/>
    public long InverseLogSf(double logQ) => DiscreteMath.InverseLogSf(this, logQ, Guess(XMath.Special.InverseNormalLogSf(logQ)));

    /// <inheritdoc/>
    public long Sample(IRandomSource random)
    {
        ArgumentNullException.ThrowIfNull(random);
        return Sampling.Poisson(random, _rate);
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, Span<long> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = Sampling.Poisson(random, _rate);
        }
    }
}
