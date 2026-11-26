using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the discrete uniform distribution over the closed integer range <c>[a, b]</c>.
/// <para>閉区間 <c>[a, b]</c> 上の整数を等確率でとる離散一様分布を表します。</para>
/// </summary>
public readonly struct DiscreteUniformDistribution : IDiscreteDistribution
{
    private readonly long _lower;
    private readonly long _upper;

    /// <summary>
    /// Initializes a new instance of <see cref="DiscreteUniformDistribution"/> over <c>[lower, upper]</c>.
    /// <para>閉区間 <c>[lower, upper]</c> を指定して <see cref="DiscreteUniformDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="lower">The inclusive lower bound <c>a</c>. / 下端 <c>a</c>（含む）。</param>
    /// <param name="upper">The inclusive upper bound <c>b</c> (<c>a ≤ b</c>). / 上端 <c>b</c>（含む、<c>a ≤ b</c>）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="upper"/> is less than <paramref name="lower"/>. / <paramref name="upper"/> が <paramref name="lower"/> より小さい場合。</exception>
    public DiscreteUniformDistribution(long lower, long upper)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(upper, lower);
        _lower = lower;
        _upper = upper;
    }

    /// <summary>
    /// Gets the inclusive lower bound <c>a</c>.
    /// <para>下端 <c>a</c>（含む）を取得します。</para>
    /// </summary>
    public long Lower => _lower;

    /// <summary>
    /// Gets the inclusive upper bound <c>b</c>.
    /// <para>上端 <c>b</c>（含む）を取得します。</para>
    /// </summary>
    public long Upper => _upper;

    private double Count => (double)Offset(_upper) + 1;

    private ulong Offset(long k) => unchecked((ulong)k - (ulong)_lower);

    // Use Int128 to prevent long overflow in lower + upper before halving.
    /// <inheritdoc/>
    public double Mean => (double)((Int128)_lower + _upper) * 0.5;

    /// <inheritdoc/>
    public double Variance
    {
        get
        {
            double n = Count;
            return (n - 1) * (n + 1) / 12;
        }
    }

    /// <inheritdoc/>
    public double StandardDeviation => Math.Sqrt(Variance);

    /// <inheritdoc/>
    public double Skewness => _lower < _upper ? 0 : double.NaN;

    /// <inheritdoc/>
    public double ExcessKurtosis
    {
        get
        {
            if (_lower == _upper)
            {
                return double.NaN;
            }

            double n = Count;
            double n2 = n * n;
            return -1.2 * (n2 + 1) / ((n - 1) * (n + 1));
        }
    }

    /// <inheritdoc/>
    public double Entropy => Math.Log(Count);

    /// <inheritdoc/>
    public long Median => InverseCdf(0.5);

    /// <inheritdoc/>
    public long SupportMinimum => _lower;

    /// <inheritdoc/>
    public long SupportMaximum => _upper;

    /// <inheritdoc/>
    public double Pmf(long k) => k >= _lower && k <= _upper ? 1 / Count : 0;

    /// <inheritdoc/>
    public double LogPmf(long k) => k >= _lower && k <= _upper ? -Math.Log(Count) : double.NegativeInfinity;

    /// <summary>
    /// Evaluates the cumulative distribution function (CDF), <c>P(X ≤ k)</c>.
    /// <para>累積分布関数（CDF）<c>P(X ≤ k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns>The cumulative probability <c>P(X ≤ k)</c>. / 累積確率 <c>P(X ≤ k)</c>。</returns>
    public double Cdf(long k) => k < _lower ? 0 : k >= _upper ? 1 : ((double)Offset(k) + 1) / Count;

    /// <summary>
    /// Evaluates the survival function, <c>P(X &gt; k)</c>.
    /// <para>生存関数（上側累積確率）<c>P(X &gt; k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns>The survival probability <c>P(X &gt; k)</c>. / 生存確率 <c>P(X &gt; k)</c>。</returns>
    public double Sf(long k) => k < _lower ? 1 : k >= _upper ? 0 : (double)(Offset(_upper) - Offset(k)) / Count;

    /// <summary>
    /// Evaluates the natural logarithm of the cumulative distribution function, <c>ln P(X ≤ k)</c>.
    /// <para>累積分布関数の自然対数 <c>ln P(X ≤ k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns><c>ln P(X ≤ k)</c>, or <see cref="double.NegativeInfinity"/> when the probability is 0. / <c>ln P(X ≤ k)</c>。確率が 0 の場合は <see cref="double.NegativeInfinity"/>。</returns>
    public double LogCdf(long k)
    {
        double p = Cdf(k);
        return p <= 0.5 ? Math.Log(p) : DistributionMath.Log1p(-Sf(k));
    }

    /// <summary>
    /// Evaluates the natural logarithm of the survival function, <c>ln P(X &gt; k)</c>.
    /// <para>生存関数の自然対数 <c>ln P(X &gt; k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns><c>ln P(X &gt; k)</c>, or <see cref="double.NegativeInfinity"/> when the probability is 0. / <c>ln P(X &gt; k)</c>。確率が 0 の場合は <see cref="double.NegativeInfinity"/>。</returns>
    public double LogSf(long k)
    {
        double q = Sf(k);
        return q <= 0.5 ? Math.Log(q) : DistributionMath.Log1p(-Cdf(k));
    }

    private long Guess(double p)
    {
        if (!(p > 0))
        {
            return _lower;
        }

        if (!(p < 1))
        {
            return _upper;
        }

        double offset = Math.Ceiling(p * Count) - 1;
        return offset <= 0 ? _lower : offset >= (double)Offset(_upper) ? _upper : unchecked(_lower + (long)(ulong)offset);
    }

    /// <inheritdoc/>
    public long InverseCdf(double p) => DiscreteMath.InverseCdf(this, p, Guess(p));

    /// <inheritdoc/>
    public long InverseSf(double q) => DiscreteMath.InverseSf(this, q, Guess(1 - q));

    /// <inheritdoc/>
    public long InverseLogCdf(double logP) => DiscreteMath.InverseLogCdf(this, logP, Guess(Math.Exp(logP)));

    /// <inheritdoc/>
    public long InverseLogSf(double logQ) => DiscreteMath.InverseLogSf(this, logQ, Guess(-DistributionMath.Expm1(logQ)));

    private long SampleCore(IRandomSource random)
    {
        ulong width = unchecked(Offset(_upper) + 1);
        ulong offset = width == 0 ? random.NextUInt64() : random.NextUInt64(width);
        return unchecked(_lower + (long)offset);
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
