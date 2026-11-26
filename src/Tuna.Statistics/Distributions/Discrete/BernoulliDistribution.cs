using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the Bernoulli distribution taking value 1 with probability <c>p</c> and 0 with probability <c>1 - p</c>.
/// <para>確率 <c>p</c> で 1、確率 <c>1 - p</c> で 0 をとるベルヌーイ分布を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Sampling produces 1 with exact probability <c>p</c> even when <c>p</c> is not an integer multiple of <c>2⁻⁵³</c>.
/// When <c>p</c> is 0 or 1, <see cref="Skewness"/> and <see cref="ExcessKurtosis"/> return <see cref="double.NaN"/>.
/// </para>
/// <para>
/// サンプリングで 1 が生成される確率は、<c>p</c> が <c>2⁻⁵³</c> の整数倍でない場合でも厳密に <c>p</c> です。
/// <c>p</c> が 0 または 1 のとき、<see cref="Skewness"/> および <see cref="ExcessKurtosis"/> は <see cref="double.NaN"/> を返します。
/// </para>
/// </remarks>
public readonly partial struct BernoulliDistribution : IDiscreteDistribution
{
    private readonly double _p;
    private readonly double _q;

    /// <summary>
    /// Initializes a new instance of <see cref="BernoulliDistribution"/> with the specified success probability.
    /// <para>成功確率を指定して <see cref="BernoulliDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="probability">The success (value 1) probability <c>p</c> in <c>[0, 1]</c>. / 成功（値 1）の確率 <c>p</c>（<c>0 ≤ p ≤ 1</c>）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="probability"/> is <see cref="double.NaN"/> or outside <c>[0, 1]</c>. / <paramref name="probability"/> が <see cref="double.NaN"/> または <c>[0, 1]</c> の範囲外の場合。</exception>
    public BernoulliDistribution(double probability)
        : this(probability, forDifferentiation: false)
    {
    }

    private BernoulliDistribution(double probability, bool forDifferentiation)
    {
        DistributionArgument.Probability(probability, nameof(probability), forDifferentiation);
        _p = probability;
        _q = 1 - probability;
    }

    /// <summary>
    /// Gets the success probability <c>p</c>.
    /// <para>成功確率 <c>p</c> を取得します。</para>
    /// </summary>
    public double Probability => _p;

    private double LogQ => _p < 0.5 ? DistributionMath.Log1p(-_p) : Math.Log(_q);

    /// <inheritdoc/>
    public double Mean => _p;

    /// <inheritdoc/>
    public double Variance => _p * _q;

    /// <inheritdoc/>
    public double StandardDeviation => Math.Sqrt(_p * _q);

    /// <inheritdoc/>
    public double Skewness => _p > 0 && _q > 0 ? (_q - _p) / Math.Sqrt(_p * _q) : double.NaN;

    /// <inheritdoc/>
    public double ExcessKurtosis => _p > 0 && _q > 0 ? (1 - 6 * _p * _q) / (_p * _q) : double.NaN;

    /// <inheritdoc/>
    public double Entropy => (_q > 0 ? -_q * LogQ : 0) + (_p > 0 ? -_p * Math.Log(_p) : 0);

    /// <inheritdoc/>
    public long Median => _p <= 0.5 ? 0 : 1;

    /// <inheritdoc/>
    public long SupportMinimum => _q == 0 ? 1 : 0;

    /// <inheritdoc/>
    public long SupportMaximum => _p == 0 ? 0 : 1;

    /// <inheritdoc/>
    public double Pmf(long k) => k == 0 ? _q : k == 1 ? _p : 0;

    /// <inheritdoc/>
    public double LogPmf(long k) => k == 0 ? LogQ : k == 1 ? Math.Log(_p) : double.NegativeInfinity;

    /// <summary>
    /// Evaluates the cumulative distribution function (CDF), <c>P(X ≤ k)</c>.
    /// <para>累積分布関数（CDF）<c>P(X ≤ k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns>The cumulative probability <c>P(X ≤ k)</c>. / 累積確率 <c>P(X ≤ k)</c>。</returns>
    public double Cdf(long k) => k < 0 ? 0 : k == 0 ? _q : 1;

    /// <summary>
    /// Evaluates the survival function, <c>P(X &gt; k)</c>.
    /// <para>生存関数（上側累積確率）<c>P(X &gt; k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns>The survival probability <c>P(X &gt; k)</c>. / 生存確率 <c>P(X &gt; k)</c>。</returns>
    public double Sf(long k) => k < 0 ? 1 : k == 0 ? _p : 0;

    /// <summary>
    /// Evaluates the natural logarithm of the cumulative distribution function, <c>ln P(X ≤ k)</c>.
    /// <para>累積分布関数の自然対数 <c>ln P(X ≤ k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns><c>ln P(X ≤ k)</c>, or <see cref="double.NegativeInfinity"/> when the probability is 0. / <c>ln P(X ≤ k)</c>。確率が 0 の場合は <see cref="double.NegativeInfinity"/>。</returns>
    public double LogCdf(long k) => k < 0 ? double.NegativeInfinity : k == 0 ? LogQ : 0;

    /// <summary>
    /// Evaluates the natural logarithm of the survival function, <c>ln P(X &gt; k)</c>.
    /// <para>生存関数の自然対数 <c>ln P(X &gt; k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns><c>ln P(X &gt; k)</c>, or <see cref="double.NegativeInfinity"/> when the probability is 0. / <c>ln P(X &gt; k)</c>。確率が 0 の場合は <see cref="double.NegativeInfinity"/>。</returns>
    public double LogSf(long k) => k < 0 ? 0 : k == 0 ? Math.Log(_p) : double.NegativeInfinity;

    /// <inheritdoc/>
    public long InverseCdf(double p) => DiscreteMath.InverseCdf(this, p, 0);

    /// <inheritdoc/>
    public long InverseSf(double q) => DiscreteMath.InverseSf(this, q, 0);

    /// <inheritdoc/>
    public long InverseLogCdf(double logP) => DiscreteMath.InverseLogCdf(this, logP, 0);

    /// <inheritdoc/>
    public long InverseLogSf(double logQ) => DiscreteMath.InverseLogSf(this, logQ, 0);

    /// <inheritdoc/>
    public long Sample(IRandomSource random)
    {
        ArgumentNullException.ThrowIfNull(random);
        return Sampling.Bernoulli(random, _p) ? 1 : 0;
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, Span<long> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = Sampling.Bernoulli(random, _p) ? 1 : 0;
        }
    }
}
