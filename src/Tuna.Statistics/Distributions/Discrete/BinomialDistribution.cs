using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the binomial distribution <c>B(n, p)</c> for the number of successes in <c>n</c> independent Bernoulli trials.
/// <para><c>n</c> 回の独立なベルヌーイ試行における成功回数の二項分布 <c>B(n, p)</c> を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// When the distribution is degenerate (<c>n = 0</c>, <c>p = 0</c>, or <c>p = 1</c>), <see cref="Skewness"/> and <see cref="ExcessKurtosis"/> return <see cref="double.NaN"/>.
/// </para>
/// <para>
/// 退化した分布（<c>n = 0</c>、<c>p = 0</c>、または <c>p = 1</c>）では、<see cref="Skewness"/> および <see cref="ExcessKurtosis"/> は <see cref="double.NaN"/> を返します。
/// </para>
/// </remarks>
public readonly partial struct BinomialDistribution : IDiscreteDistribution
{
    /// <summary>
    /// The maximum supported number of trials (<c>2⁵³</c>, the largest exact integer range in <see cref="double"/>).
    /// <para>サポートされる試行回数の上限（<c>2⁵³</c>、<see cref="double"/> で正確に表現可能な整数範囲）。</para>
    /// </summary>
    public const long MaxTrials = 1L << 53;

    private readonly long _trials;
    private readonly double _p;
    private readonly double _q;

    /// <summary>
    /// Initializes a new instance of <see cref="BinomialDistribution"/> with the specified trial count and success probability.
    /// <para>試行回数と成功確率を指定して <see cref="BinomialDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="trials">The number of trials <c>n</c> (<c>0 ≤ n ≤ 2⁵³</c>). / 試行回数 <c>n</c>（<c>0 ≤ n ≤ 2⁵³</c>）。</param>
    /// <param name="probability">The success probability <c>p</c> in <c>[0, 1]</c>. / 成功確率 <c>p</c>（<c>0 ≤ p ≤ 1</c>）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="trials"/> is outside <c>[0, MaxTrials]</c>, or <paramref name="probability"/> is <see cref="double.NaN"/> or outside <c>[0, 1]</c>. / <paramref name="trials"/> が <c>[0, MaxTrials]</c> の範囲外、または <paramref name="probability"/> が <see cref="double.NaN"/> もしくは <c>[0, 1]</c> の範囲外の場合。</exception>
    public BinomialDistribution(long trials, double probability)
        : this(trials, probability, forDifferentiation: false)
    {
    }

    private BinomialDistribution(long trials, double probability, bool forDifferentiation)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(trials);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(trials, MaxTrials);
        DistributionArgument.Probability(probability, nameof(probability), forDifferentiation);
        _trials = trials;
        _p = probability;
        _q = 1 - probability;
    }

    /// <summary>
    /// Gets the number of trials <c>n</c>.
    /// <para>試行回数 <c>n</c> を取得します。</para>
    /// </summary>
    public long Trials => _trials;

    /// <summary>
    /// Gets the success probability <c>p</c>.
    /// <para>成功確率 <c>p</c> を取得します。</para>
    /// </summary>
    public double Probability => _p;

    private bool IsDegenerate => _trials == 0 || _p == 0 || _q == 0;

    /// <inheritdoc/>
    public double Mean => _trials * _p;

    /// <inheritdoc/>
    public double Variance => _trials * _p * _q;

    /// <inheritdoc/>
    public double StandardDeviation => Math.Sqrt(Variance);

    /// <inheritdoc/>
    public double Skewness => IsDegenerate ? double.NaN : (_q - _p) / Math.Sqrt(Variance);

    /// <inheritdoc/>
    public double ExcessKurtosis => IsDegenerate ? double.NaN : (1 - 6 * _p * _q) / Variance;

    /// <inheritdoc/>
    public double Entropy
    {
        get
        {
            if (IsDegenerate)
            {
                return 0;
            }

            double skewness = Skewness, kurtosis = ExcessKurtosis;
            if (Math.Max(skewness * skewness, Math.Abs(kurtosis)) <= DiscreteMath.EntropyExpansionLimit)
            {
                return DiscreteMath.EntropyExpansion(Math.Log(Variance), skewness, kurtosis);
            }

            var self = this;
            long mode = (long)Math.Floor((_trials + 1.0) * _p);
            return DiscreteMath.EntropySum(self.LogPmf, Math.Min(mode, _trials), 0, _trials);
        }
    }

    /// <inheritdoc/>
    public long Median => InverseCdf(0.5);

    /// <inheritdoc/>
    public long SupportMinimum => _q == 0 ? _trials : 0;

    /// <inheritdoc/>
    public long SupportMaximum => _p == 0 ? 0 : _trials;

    /// <inheritdoc/>
    public double Pmf(long k) => Math.Exp(LogPmf(k));

    /// <inheritdoc/>
    public double LogPmf(long k) =>
        k < 0 || k > _trials ? double.NegativeInfinity : DistributionMath.LogBinomialProbability(k, _trials, _p, _q);

    /// <summary>
    /// Evaluates the cumulative distribution function (CDF), <c>P(X ≤ k)</c>.
    /// <para>累積分布関数（CDF）<c>P(X ≤ k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns>The cumulative probability <c>P(X ≤ k)</c>. / 累積確率 <c>P(X ≤ k)</c>。</returns>
    public double Cdf(long k)
    {
        if (k < 0)
        {
            return 0;
        }

        if (k >= _trials)
        {
            return 1;
        }

        double a = k + 1.0, b = (double)(_trials - k);
        return _p <= 0.5
            ? XMath.Special.RegularizedIncompleteBetaComplement(a, b, _p)
            : XMath.Special.RegularizedIncompleteBeta(b, a, _q);
    }

    /// <summary>
    /// Evaluates the survival function, <c>P(X &gt; k)</c>.
    /// <para>生存関数（上側累積確率）<c>P(X &gt; k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns>The survival probability <c>P(X &gt; k)</c>. / 生存確率 <c>P(X &gt; k)</c>。</returns>
    public double Sf(long k)
    {
        if (k < 0)
        {
            return 1;
        }

        if (k >= _trials)
        {
            return 0;
        }

        double a = k + 1.0, b = (double)(_trials - k);
        return _p <= 0.5
            ? XMath.Special.RegularizedIncompleteBeta(a, b, _p)
            : XMath.Special.RegularizedIncompleteBetaComplement(b, a, _q);
    }

    /// <summary>
    /// Evaluates the natural logarithm of the cumulative distribution function, <c>ln P(X ≤ k)</c>.
    /// <para>累積分布関数の自然対数 <c>ln P(X ≤ k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns><c>ln P(X ≤ k)</c>, or <see cref="double.NegativeInfinity"/> when the probability is 0. / <c>ln P(X ≤ k)</c>。確率が 0 の場合は <see cref="double.NegativeInfinity"/>。</returns>
    public double LogCdf(long k)
    {
        if (k < 0)
        {
            return double.NegativeInfinity;
        }

        if (k >= _trials)
        {
            return 0;
        }

        double a = k + 1.0, b = (double)(_trials - k);
        return _p <= 0.5
            ? XMath.Special.LogRegularizedIncompleteBetaComplement(a, b, _p)
            : XMath.Special.LogRegularizedIncompleteBeta(b, a, _q);
    }

    /// <summary>
    /// Evaluates the natural logarithm of the survival function, <c>ln P(X &gt; k)</c>.
    /// <para>生存関数の自然対数 <c>ln P(X &gt; k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns><c>ln P(X &gt; k)</c>, or <see cref="double.NegativeInfinity"/> when the probability is 0. / <c>ln P(X &gt; k)</c>。確率が 0 の場合は <see cref="double.NegativeInfinity"/>。</returns>
    public double LogSf(long k)
    {
        if (k < 0)
        {
            return 0;
        }

        if (k >= _trials)
        {
            return double.NegativeInfinity;
        }

        double a = k + 1.0, b = (double)(_trials - k);
        return _p <= 0.5
            ? XMath.Special.LogRegularizedIncompleteBeta(a, b, _p)
            : XMath.Special.LogRegularizedIncompleteBetaComplement(b, a, _q);
    }

    private long Guess(double z) =>
        IsDegenerate ? SupportMinimum : DiscreteMath.Guess(Mean, StandardDeviation, Skewness, z, SupportMinimum, SupportMaximum);

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
        return Sampling.Binomial(random, _trials, _p, _q);
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, Span<long> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = Sampling.Binomial(random, _trials, _p, _q);
        }
    }
}
