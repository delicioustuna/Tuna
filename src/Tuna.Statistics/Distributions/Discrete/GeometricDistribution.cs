using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the geometric distribution for the number of failures before the first success.
/// <para>初めて成功するまでの失敗回数を表す幾何分布です。</para>
/// </summary>
/// <remarks>
/// <para>
/// The support starts at 0 and corresponds to <see cref="NegativeBinomialDistribution"/> with <c>r = 1</c>.
/// Samples exceeding <see cref="long.MaxValue"/> saturate to <see cref="long.MaxValue"/>.
/// When <c>p = 1</c>, <see cref="Skewness"/> and <see cref="ExcessKurtosis"/> return <see cref="double.NaN"/>.
/// </para>
/// <para>
/// 台は 0 から始まり、<see cref="NegativeBinomialDistribution"/> で成功回数を 1 とした場合に一致します。
/// サンプリング結果が <see cref="long.MaxValue"/> を超える場合は <see cref="long.MaxValue"/> に丸められます。
/// <c>p = 1</c> のとき、<see cref="Skewness"/> および <see cref="ExcessKurtosis"/> は <see cref="double.NaN"/> を返します。
/// </para>
/// </remarks>
public readonly partial struct GeometricDistribution : IDiscreteDistribution
{
    private readonly double _p;
    private readonly double _logQ;

    /// <summary>
    /// Initializes a new instance of <see cref="GeometricDistribution"/> with the specified success probability.
    /// <para>成功確率を指定して <see cref="GeometricDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="probability">The per-trial success probability <c>p</c> in <c>(0, 1]</c>. / 各試行の成功確率 <c>p</c>（<c>0 &lt; p ≤ 1</c>）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="probability"/> is <see cref="double.NaN"/> or outside <c>(0, 1]</c>. / <paramref name="probability"/> が <see cref="double.NaN"/> または <c>(0, 1]</c> の範囲外の場合。</exception>
    public GeometricDistribution(double probability)
        : this(probability, forDifferentiation: false)
    {
    }

    private GeometricDistribution(double probability, bool forDifferentiation)
    {
        if (!(probability > 0 && probability <= 1))
        {
            throw DistributionArgument.ParameterOutOfRange(forDifferentiation, nameof(probability), probability, "The probability must be in (0, 1].");
        }

        _p = probability;
        _logQ = DistributionMath.Log1p(-probability);
    }

    /// <summary>
    /// Gets the per-trial success probability <c>p</c>.
    /// <para>各試行の成功確率 <c>p</c> を取得します。</para>
    /// </summary>
    public double Probability => _p;

    private double Q => 1 - _p;

    /// <inheritdoc/>
    public double Mean => Q / _p;

    // Divide by p twice to avoid subnormal underflow of p² when p < 1.5e-154.
    /// <inheritdoc/>
    public double Variance => Q / _p / _p;

    /// <inheritdoc/>
    public double StandardDeviation => Math.Sqrt(Q) / _p;

    /// <inheritdoc/>
    public double Skewness => _p < 1 ? (2 - _p) / Math.Sqrt(Q) : double.NaN;

    /// <inheritdoc/>
    public double ExcessKurtosis => _p < 1 ? 6 + _p * _p / Q : double.NaN;

    // Divide ln(1 - p) by p before multiplying by (1 - p) to avoid overflow of (1 - p)/p near subnormal p.
    /// <inheritdoc/>
    public double Entropy => _p < 1 ? -Q * (_logQ / _p) - Math.Log(_p) : 0;

    /// <inheritdoc/>
    public long Median => InverseCdf(0.5);

    /// <inheritdoc/>
    public long SupportMinimum => 0;

    /// <inheritdoc/>
    public long SupportMaximum => _p == 1 ? 0 : long.MaxValue;

    private double LogSurvival(long k) => ((double)k + 1) * _logQ;

    /// <inheritdoc/>
    public double Pmf(long k) => Math.Exp(LogPmf(k));

    /// <inheritdoc/>
    public double LogPmf(long k) => k < 0 ? double.NegativeInfinity : k == 0 ? Math.Log(_p) : Math.Log(_p) + k * _logQ;

    /// <summary>
    /// Evaluates the cumulative distribution function (CDF), <c>P(X ≤ k)</c>.
    /// <para>累積分布関数（CDF）<c>P(X ≤ k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns>The cumulative probability <c>P(X ≤ k)</c>. / 累積確率 <c>P(X ≤ k)</c>。</returns>
    public double Cdf(long k) => k < 0 ? 0 : -DistributionMath.Expm1(LogSurvival(k));

    /// <summary>
    /// Evaluates the survival function, <c>P(X &gt; k)</c>.
    /// <para>生存関数（上側累積確率）<c>P(X &gt; k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns>The survival probability <c>P(X &gt; k)</c>. / 生存確率 <c>P(X &gt; k)</c>。</returns>
    public double Sf(long k) => k < 0 ? 1 : Math.Exp(LogSurvival(k));

    /// <summary>
    /// Evaluates the natural logarithm of the cumulative distribution function, <c>ln P(X ≤ k)</c>.
    /// <para>累積分布関数の自然対数 <c>ln P(X ≤ k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns><c>ln P(X ≤ k)</c>, or <see cref="double.NegativeInfinity"/> when the probability is 0. / <c>ln P(X ≤ k)</c>。確率が 0 の場合は <see cref="double.NegativeInfinity"/>。</returns>
    public double LogCdf(long k) => k < 0 ? double.NegativeInfinity : DistributionMath.Log1mExp(LogSurvival(k));

    /// <summary>
    /// Evaluates the natural logarithm of the survival function, <c>ln P(X &gt; k)</c>.
    /// <para>生存関数の自然対数 <c>ln P(X &gt; k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns><c>ln P(X &gt; k)</c>, or <see cref="double.NegativeInfinity"/> when the probability is 0. / <c>ln P(X &gt; k)</c>。確率が 0 の場合は <see cref="double.NegativeInfinity"/>。</returns>
    public double LogSf(long k) => k < 0 ? 0 : LogSurvival(k);

    private long Guess(double logQ)
    {
        if (_p == 1)
        {
            return 0;
        }

        double value = Math.Ceiling(logQ / _logQ) - 1;
        return value <= 0 ? 0 : value >= 9.2e18 ? long.MaxValue : (long)value;
    }

    /// <inheritdoc/>
    public long InverseCdf(double p) => DiscreteMath.InverseCdf(this, p, p >= 0 && p <= 1 ? Guess(DistributionMath.Log1p(-p)) : 0);

    /// <inheritdoc/>
    public long InverseSf(double q) => DiscreteMath.InverseSf(this, q, q >= 0 && q <= 1 ? Guess(Math.Log(q)) : 0);

    /// <inheritdoc/>
    public long InverseLogCdf(double logP) => DiscreteMath.InverseLogCdf(this, logP, logP <= 0 ? Guess(DistributionMath.Log1mExp(logP)) : 0);

    /// <inheritdoc/>
    public long InverseLogSf(double logQ) => DiscreteMath.InverseLogSf(this, logQ, logQ <= 0 ? Guess(logQ) : 0);

    private long SampleCore(IRandomSource random)
    {
        if (_p == 1)
        {
            return 0;
        }

        double value = Math.Floor(Ziggurat.StandardExponential(random) / -_logQ);
        return value >= 9.2233720368547758e18 ? long.MaxValue : (long)value;
    }

    /// <inheritdoc/>
    public long Sample(IRandomSource random)
    {
        ArgumentNullException.ThrowIfNull(random);
        return SampleCore(random);
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, Span<long> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = SampleCore(random);
        }
    }
}
