using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the negative binomial distribution for the number of failures before <c>r</c> successes.
/// <para><c>r</c> 回の成功に至るまでの失敗回数を表す負の二項分布です。</para>
/// </summary>
/// <remarks>
/// <para>
/// The success parameter <c>r</c> may be any positive real number up to <see cref="MaxSuccesses"/> (modeling overdispersed counts as a gamma–Poisson mixture).
/// The PMF is <c>Γ(k + r) / (Γ(r) k!) · pʳ (1 - p)ᵏ</c> for <c>k ≥ 0</c>.
/// When <c>p = 1</c>, <see cref="Skewness"/> and <see cref="ExcessKurtosis"/> return <see cref="double.NaN"/>.
/// </para>
/// <para>
/// 成功回数パラメーター <c>r</c> は <see cref="MaxSuccesses"/> 以下の任意の正の実数を指定できます（ガンマ・ポアソン混合として過分散な計数データをモデル化できます）。
/// 確率質量関数は <c>k ≥ 0</c> に対して <c>Γ(k + r) / (Γ(r) k!) · pʳ (1 - p)ᵏ</c> です。
/// <c>p = 1</c> のとき、<see cref="Skewness"/> および <see cref="ExcessKurtosis"/> は <see cref="double.NaN"/> を返します。
/// </para>
/// </remarks>
public readonly partial struct NegativeBinomialDistribution : IDiscreteDistribution
{
    /// <summary>
    /// The maximum supported value for <see cref="Successes"/> (<c>2⁵²</c>, keeping <c>r + k</c> representable in <see cref="double"/>).
    /// <para><see cref="Successes"/> の上限（<c>2⁵²</c>。<c>r + k</c> を <see cref="double"/> で扱うための範囲）。</para>
    /// </summary>
    public const double MaxSuccesses = 4503599627370496;

    private readonly double _r;
    private readonly double _p;
    private readonly double _q;

    /// <summary>
    /// Initializes a new instance of <see cref="NegativeBinomialDistribution"/> with the specified success count and success probability.
    /// <para>成功回数と成功確率を指定して <see cref="NegativeBinomialDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="successes">The number of successes <c>r</c> (<c>0 &lt; r ≤ 2⁵²</c>; need not be an integer). / 成功回数 <c>r</c>（<c>0 &lt; r ≤ 2⁵²</c>、非整数も可）。</param>
    /// <param name="probability">The per-trial success probability <c>p</c> in <c>(0, 1]</c>. / 各試行の成功確率 <c>p</c>（<c>0 &lt; p ≤ 1</c>）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="successes"/> is outside <c>(0, MaxSuccesses]</c>, or <paramref name="probability"/> is <see cref="double.NaN"/> or outside <c>(0, 1]</c>. / <paramref name="successes"/> が <c>(0, MaxSuccesses]</c> の範囲外、または <paramref name="probability"/> が <see cref="double.NaN"/> もしくは <c>(0, 1]</c> の範囲外の場合。</exception>
    public NegativeBinomialDistribution(double successes, double probability)
        : this(successes, probability, forDifferentiation: false)
    {
    }

    private NegativeBinomialDistribution(double successes, double probability, bool forDifferentiation)
    {
        DistributionArgument.PositiveFinite(successes, nameof(successes), forDifferentiation);
        if (forDifferentiation && successes > MaxSuccesses)
        {
            throw new DistributionParameterOutOfRangeException(nameof(successes), successes, "The success count must not exceed MaxSuccesses.");
        }

        ArgumentOutOfRangeException.ThrowIfGreaterThan(successes, MaxSuccesses);

        if (!(probability > 0 && probability <= 1))
        {
            throw DistributionArgument.ParameterOutOfRange(forDifferentiation, nameof(probability), probability, "The probability must be in (0, 1].");
        }

        _r = successes;
        _p = probability;
        _q = 1 - probability;
    }

    /// <summary>
    /// Creates a <see cref="NegativeBinomialDistribution"/> from mean <c>μ</c> and dispersion shape <c>r</c> (variance <c>μ + μ²/r</c>).
    /// <para>平均 <c>μ</c> と形状パラメーター <c>r</c>（分散 <c>μ + μ²/r</c>）から <see cref="NegativeBinomialDistribution"/> を生成します。</para>
    /// </summary>
    /// <param name="mean">The mean <c>μ</c> (non-negative and finite). / 平均 <c>μ</c>（非負かつ有限）。</param>
    /// <param name="successes">The shape parameter <c>r</c> (<c>0 &lt; r ≤ 2⁵²</c>). / 形状パラメーター <c>r</c>（<c>0 &lt; r ≤ 2⁵²</c>）。</param>
    /// <returns>A negative binomial distribution with the specified mean and shape. / 指定した平均と形状を持つ負の二項分布。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="mean"/> is negative or non-finite, or <paramref name="successes"/> is outside <c>(0, MaxSuccesses]</c>. / <paramref name="mean"/> が負または非有限、あるいは <paramref name="successes"/> が <c>(0, MaxSuccesses]</c> の範囲外の場合。</exception>
    public static NegativeBinomialDistribution FromMean(double mean, double successes)
    {
        DistributionArgument.NonNegativeFinite(mean, nameof(mean));
        DistributionArgument.PositiveFinite(successes, nameof(successes));
        return new NegativeBinomialDistribution(successes, successes / (successes + mean));
    }

    /// <summary>
    /// Gets the success count (shape) parameter <c>r</c>.
    /// <para>成功回数（形状）パラメーター <c>r</c> を取得します。</para>
    /// </summary>
    public double Successes => _r;

    /// <summary>
    /// Gets the per-trial success probability <c>p</c>.
    /// <para>各試行の成功確率 <c>p</c> を取得します。</para>
    /// </summary>
    public double Probability => _p;

    /// <inheritdoc/>
    public double Mean => _r * _q / _p;

    // ScaledRatio avoids subnormal underflow/overflow of p² when p < 1.5e-154.
    /// <inheritdoc/>
    public double Variance => DistributionMath.ScaledRatio([_r, _q], [_p, _p]);

    // Separate square roots prevent underflow of r*(1 - p) when r is tiny and p is near 1.
    /// <inheritdoc/>
    public double StandardDeviation => Math.Sqrt(_r) * Math.Sqrt(_q) / _p;

    /// <inheritdoc/>
    public double Skewness => _q > 0 ? (2 - _p) / (Math.Sqrt(_r) * Math.Sqrt(_q)) : double.NaN;

    /// <inheritdoc/>
    public double ExcessKurtosis => _q > 0 ? 6 / _r + DistributionMath.ScaledRatio([_p, _p], [_r, _q]) : double.NaN;

    /// <inheritdoc/>
    public double Entropy
    {
        get
        {
            if (_q == 0)
            {
                return 0;
            }

            double skewness = Skewness, kurtosis = ExcessKurtosis;
            if (Math.Max(skewness * skewness, Math.Abs(kurtosis)) <= DiscreteMath.EntropyExpansionLimit)
            {
                // Pass log-variance directly because rq/p² can overflow for tiny p (expansion is used when rq >= 1e5).
                return DiscreteMath.EntropyExpansion(Math.Log(_r * _q) - 2 * Math.Log(_p), skewness, kurtosis);
            }

            if (_r == 1)
            {
                return -_q * (DistributionMath.Log1p(-_p) / _p) - Math.Log(_p);
            }

            // Skip direct PMF summation for r < 1 (where p_k scales with r and log-space rounding amplifies via exp) or wide support.
            double terms = Mean + 40 * StandardDeviation + 40 / -DistributionMath.Log1p(-_p);
            if (_r > 1 && terms <= 2e5)
            {
                var self = this;
                long mode = (long)Math.Floor((_r - 1) * _q / _p);
                return DiscreteMath.EntropySum(self.LogPmf, mode, 0, long.MaxValue);
            }

            return EntropyIntegral();
        }
    }

    // Evaluates entropy via the integral representation of ln Γ and the PGF G(e^-t) using the trapezoidal rule with Neumaier summation.
    private double EntropyIntegral()
    {
        double r = _r, p = _p, q = _q;
        double logP = Math.Log(p);
        double logQ = DistributionMath.Log1p(-p);
        double logR = Math.Log(r);

        double logOdds = logQ - logP;

        // Integrand (divided by r) behaves as -(r - 1)(q/p)t as t -> 0 and decays as e^(-min(r, 1)t) as t -> Inf.
        double lower = -Math.Max(logOdds, 0) - 2 * Math.Log(r + 1) - 40;
        double upper = Math.Log(45) - Math.Min(logR, 0) + 1;
        const double step = 0.15;

        // Factor out r so subnormal r retains accuracy; compute nodes from index k to avoid cumulative step drift.
        double sum = 0, compensation = 0;
        int count = (int)((upper - lower) / step);
        for (int k = 0; k <= count; k++)
        {
            double u = lower + k * step;

            double t = Math.Exp(u);
            double weight, log1pX;
            if (t * (r + 1) < 1e-20)
            {
                // Small-t limit: (e^-t - e^-rt)/(1 - e^-t) -> r - 1, 1 - e^-t -> t (relative error <= (r+1)t/2).
                weight = r - 1;
                log1pX = DistributionMath.Log1p(Math.Exp(u + logOdds));
            }
            else
            {
                double em1 = DistributionMath.Expm1(-t);

                double rt = t < double.PositiveInfinity ? r * t : Math.Exp(u + logR);
                weight = t < 1
                    ? Math.Exp(-t) * DistributionMath.Expm1(-(r - 1) * t) / em1
                    : (Math.Exp(-t) - Math.Exp(-rt)) / -em1;

                double x = q * -em1 / p;
                log1pX = x < double.PositiveInfinity ? DistributionMath.Log1p(x) : logOdds + Math.Log(-em1);
            }

            // ((1 + x)^(-r) - 1)/r -> -ln(1 + x) when r*ln(1 + x) < 1e-20.
            double rl = r * log1pX;
            double term = weight * (rl < 1e-20 ? -log1pX : DistributionMath.Expm1(-rl) / r);
            double next = sum + term;
            compensation += Math.Abs(sum) >= Math.Abs(term) ? sum - next + term : term - next + sum;
            sum = next;
        }

        return r * (-logP - q * (logQ / p) + step * (sum + compensation));
    }

    /// <inheritdoc/>
    public long Median => InverseCdf(0.5);

    /// <inheritdoc/>
    public long SupportMinimum => 0;

    /// <inheritdoc/>
    public long SupportMaximum => _q == 0 ? 0 : long.MaxValue;

    /// <inheritdoc/>
    public double Pmf(long k) => Math.Exp(LogPmf(k));

    /// <inheritdoc/>
    public double LogPmf(long k)
    {
        if (k < 0)
        {
            return double.NegativeInfinity;
        }

        if (k == 0)
        {
            return _r * Math.Log(_p);
        }

        if (_q == 0)
        {
            return double.NegativeInfinity;
        }

        double failures = k;
        double n = _r + failures;
        if (_r >= 1)
        {
            return Math.Log(_r / n) + DistributionMath.LogBinomialProbabilityCore(_r, failures, n, _p, _q);
        }

        // For r < 1, r/n can become subnormal and 1/r in S(r) can overflow; group -S(r) - (1/2) ln(n/r) via StirlingRemainderShifted.
        return DistributionMath.StirlingRemainder(n) - DistributionMath.StirlingRemainderShifted(_r, failures) - DistributionMath.StirlingRemainder(failures)
            - 0.5 * (NumericConstants.LnTau + Math.Log(failures)) - DistributionMath.Bd0(_r, n * _p) - DistributionMath.Bd0(failures, n * _q);
    }

    /// <summary>
    /// Evaluates the cumulative distribution function (CDF), <c>P(X ≤ k)</c>.
    /// <para>累積分布関数（CDF）<c>P(X ≤ k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns>The cumulative probability <c>P(X ≤ k)</c>. / 累積確率 <c>P(X ≤ k)</c>。</returns>
    public double Cdf(long k) => k < 0 ? 0 : XMath.Special.RegularizedIncompleteBeta(_r, k + 1.0, _p);

    /// <summary>
    /// Evaluates the survival function, <c>P(X &gt; k)</c>.
    /// <para>生存関数（上側累積確率）<c>P(X &gt; k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns>The survival probability <c>P(X &gt; k)</c>. / 生存確率 <c>P(X &gt; k)</c>。</returns>
    public double Sf(long k) => k < 0 ? 1 : XMath.Special.RegularizedIncompleteBetaComplement(_r, k + 1.0, _p);

    /// <summary>
    /// Evaluates the natural logarithm of the cumulative distribution function, <c>ln P(X ≤ k)</c>.
    /// <para>累積分布関数の自然対数 <c>ln P(X ≤ k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns><c>ln P(X ≤ k)</c>, or <see cref="double.NegativeInfinity"/> when the probability is 0. / <c>ln P(X ≤ k)</c>。確率が 0 の場合は <see cref="double.NegativeInfinity"/>。</returns>
    public double LogCdf(long k) => k < 0 ? double.NegativeInfinity : XMath.Special.LogRegularizedIncompleteBeta(_r, k + 1.0, _p);

    /// <summary>
    /// Evaluates the natural logarithm of the survival function, <c>ln P(X &gt; k)</c>.
    /// <para>生存関数の自然対数 <c>ln P(X &gt; k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns><c>ln P(X &gt; k)</c>, or <see cref="double.NegativeInfinity"/> when the probability is 0. / <c>ln P(X &gt; k)</c>。確率が 0 の場合は <see cref="double.NegativeInfinity"/>。</returns>
    public double LogSf(long k) => k < 0 ? 0 : XMath.Special.LogRegularizedIncompleteBetaComplement(_r, k + 1.0, _p);

    private long Guess(double z) =>
        _q == 0 ? 0 : DiscreteMath.Guess(Mean, StandardDeviation, Skewness, z, 0, long.MaxValue);

    /// <inheritdoc/>
    public long InverseCdf(double p) => DiscreteMath.InverseCdf(this, p, Guess(XMath.Special.InverseNormalCdf(p)));

    /// <inheritdoc/>
    public long InverseSf(double q) => DiscreteMath.InverseSf(this, q, Guess(XMath.Special.InverseNormalSf(q)));

    /// <inheritdoc/>
    public long InverseLogCdf(double logP) => DiscreteMath.InverseLogCdf(this, logP, Guess(XMath.Special.InverseNormalLogCdf(logP)));

    /// <inheritdoc/>
    public long InverseLogSf(double logQ) => DiscreteMath.InverseLogSf(this, logQ, Guess(XMath.Special.InverseNormalLogSf(logQ)));

    private long SampleCore(IRandomSource random)
    {
        if (_q == 0)
        {
            return 0;
        }

        double logOdds = DistributionMath.Log1p(-_p) - Math.Log(_p);
        double rate = Math.Exp(Sampling.LogStandardGamma(random, _r) + logOdds);
        return Sampling.Poisson(random, rate);
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
