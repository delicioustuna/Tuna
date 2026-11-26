using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents a continuous distribution conditioned (truncated) to a specified interval.
/// <para>連続分布を指定した区間に制限した切断分布（条件付き分布）を表します。</para>
/// </summary>
/// <typeparam name="TDistribution">The type of the underlying continuous distribution. / 元の連続分布の型。</typeparam>
/// <remarks>
/// <para>
/// The density is restricted to the intersection of the truncation interval with the base support and renormalized by the interval probability.
/// If the interval covers the entire base support, moments and entropy delegate directly to the base distribution; otherwise they are computed via double-exponential numerical quadrature.
/// If one endpoint is infinite and the corresponding moment of the base distribution is not finite, the truncated moment may return infinity on the unbounded side.
/// </para>
/// <para>
/// 元の分布を指定区間と台の共通部分に制限し、その区間の確率で密度を正規化します。
/// 区間が元の台全体を含む場合は元の分布の値をそのまま返し、それ以外では二重指数関数型数値積分でモーメントとエントロピーを計算します。
/// 区間の片側が無限で元の分布の対応するモーメントが有限でない場合は、その側の無限大を返すことがあります。
/// </para>
/// </remarks>
public readonly struct TruncatedDistribution<TDistribution> : IContinuousDistribution
    where TDistribution : IContinuousDistribution
{
    // Threshold below which sums/differences of probabilities switch to log-space.
    private const double LinearLimit = 1e-280;

    // Trigger density quadrature when subtraction loses more than 3 bits (difference < 1/8 of minuend).
    private const double CancellationRatio = 0.125;

    // ln(CancellationRatio) and ln(1 - CancellationRatio).
    private const double LogCancellationRatio = -2.0794415416798357;
    private const double LogCancellationLimit = -0.13353139262452263;

    private readonly TDistribution _distribution;
    private readonly double _lowerBound;
    private readonly double _upperBound;

    private readonly double _lower;
    private readonly double _upper;

    private readonly double _lowerCdf;
    private readonly double _lowerSf;
    private readonly double _lowerLogCdf;
    private readonly double _lowerLogSf;
    private readonly double _upperCdf;
    private readonly double _upperSf;
    private readonly double _upperLogCdf;
    private readonly double _upperLogSf;

    // Interval probability Z (0 if subnormal/underflowed) and ln Z.
    private readonly double _mass;
    private readonly double _logMass;

    /// <summary>
    /// Initializes a new instance of <see cref="TruncatedDistribution{TDistribution}"/>.
    /// <para>元の分布と切断区間を指定して <see cref="TruncatedDistribution{TDistribution}"/> を初期化します。</para>
    /// </summary>
    /// <param name="distribution">The underlying continuous distribution. / 元の連続分布。</param>
    /// <param name="lower">The lower bound <c>a</c> (<see cref="double.NegativeInfinity"/> for no lower truncation). / 下限 <c>a</c>（<see cref="double.NegativeInfinity"/> の場合は下側を切断しない）。</param>
    /// <param name="upper">The upper bound <c>b</c> with <c>a &lt; b</c> (<see cref="double.PositiveInfinity"/> for no upper truncation). / 上限 <c>b</c>（<c>a &lt; b</c>。<see cref="double.PositiveInfinity"/> の場合は上側を切断しない）。</param>
    /// <exception cref="ArgumentNullException"><paramref name="distribution"/> is <see langword="null"/>. / <paramref name="distribution"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">The interval is empty or contains <see cref="double.NaN"/>, does not intersect the base support, or has zero probability. / 区間が空または <see cref="double.NaN"/> を含む、元の分布の台と交わらない、または区間確率が 0 の場合。</exception>
    public TruncatedDistribution(TDistribution distribution, double lower, double upper)
    {
        ArgumentNullException.ThrowIfNull(distribution);
        if (double.IsNaN(lower))
        {
            throw new ArgumentOutOfRangeException(nameof(lower), lower, "The bound must not be NaN.");
        }

        if (!(lower < upper))
        {
            throw new ArgumentOutOfRangeException(nameof(upper), upper, "The upper bound must be greater than the lower bound.");
        }

        double lo = Math.Max(lower, distribution.SupportMinimum);
        double hi = Math.Min(upper, distribution.SupportMaximum);
        if (!(lo < hi))
        {
            throw new ArgumentOutOfRangeException(nameof(lower), lower, "The interval must intersect the support of the distribution.");
        }

        _distribution = distribution;
        _lowerBound = lower;
        _upperBound = upper;
        _lower = lo;
        _upper = hi;
        if (lo == double.NegativeInfinity)
        {
            (_lowerCdf, _lowerSf, _lowerLogCdf, _lowerLogSf) = (0, 1, double.NegativeInfinity, 0);
        }
        else
        {
            (_lowerCdf, _lowerSf) = (distribution.Cdf(lo), distribution.Sf(lo));
            (_lowerLogCdf, _lowerLogSf) = (distribution.LogCdf(lo), distribution.LogSf(lo));
        }

        if (hi == double.PositiveInfinity)
        {
            (_upperCdf, _upperSf, _upperLogCdf, _upperLogSf) = (1, 0, 0, double.NegativeInfinity);
        }
        else
        {
            (_upperCdf, _upperSf) = (distribution.Cdf(hi), distribution.Sf(hi));
            (_upperLogCdf, _upperLogSf) = (distribution.LogCdf(hi), distribution.LogSf(hi));
        }

        _mass = 0;
        _logMass = 0;
        double mass = BaseMass(_lowerCdf, _lowerSf, _lowerLogCdf, _lowerLogSf, _upperCdf, _upperSf, _upperLogCdf, _upperLogSf,
            lo, hi, out double logMass);
        if (!(logMass > double.NegativeInfinity))
        {
            throw new ArgumentOutOfRangeException(nameof(lower), lower, "The probability of the interval must be positive.");
        }

        _mass = mass;
        _logMass = logMass;
    }

    /// <summary>
    /// Gets the underlying distribution.
    /// <para>元の連続分布を取得します。</para>
    /// </summary>
    public TDistribution Distribution => _distribution;

    /// <summary>
    /// Gets the specified lower bound <c>a</c> (for the effective support lower bound, see <see cref="SupportMinimum"/>).
    /// <para>指定された下限 <c>a</c> を取得します（実際の台の下限は <see cref="SupportMinimum"/> を参照してください）。</para>
    /// </summary>
    public double LowerBound => _lowerBound;

    /// <summary>
    /// Gets the specified upper bound <c>b</c> (for the effective support upper bound, see <see cref="SupportMaximum"/>).
    /// <para>指定された上限 <c>b</c> を取得します（実際の台の上限は <see cref="SupportMaximum"/> を参照してください）。</para>
    /// </summary>
    public double UpperBound => _upperBound;

    /// <summary>
    /// Gets the probability that the underlying distribution falls within the truncation interval.
    /// <para>元の分布において切断区間内に入る確率を取得します。</para>
    /// </summary>
    /// <value>The interval probability <c>P(a ≤ Y ≤ b)</c>, or 0 if it underflows. / 区間確率 <c>P(a ≤ Y ≤ b)</c>。アンダーフローする場合は 0。</value>
    public double Probability => _mass > 0 ? _mass : Math.Exp(_logMass);

    /// <summary>
    /// Gets the natural logarithm of the probability that the underlying distribution falls within the truncation interval.
    /// <para>元の分布において切断区間内に入る確率の自然対数を取得します。</para>
    /// </summary>
    /// <value>The log-probability <c>ln P(a ≤ Y ≤ b)</c>, which remains finite even when <see cref="Probability"/> underflows. / 対数区間確率 <c>ln P(a ≤ Y ≤ b)</c>。<see cref="Probability"/> がアンダーフローしても有限値を保持します。</value>
    public double LogProbability => _logMass;

    /// <inheritdoc/>
    public double SupportMinimum => _lower;

    /// <inheritdoc/>
    public double SupportMaximum => _upper;

    /// <inheritdoc/>
    public double Median => InverseCdf(0.5);

    private bool IsUntruncated => _lower == _distribution.SupportMinimum && _upper == _distribution.SupportMaximum;

    // 1 if unbounded above, -1 if unbounded below, 0 if bounded (both infinite only occurs when untruncated).
    private int UnboundedSide => _upper == double.PositiveInfinity ? 1 : _lower == double.NegativeInfinity ? -1 : 0;

    // Smallest moment order (1..4) that is non-finite on the unbounded side, or 5 if all are finite or the interval is bounded.
    private int DivergentOrder
    {
        get
        {
            if (UnboundedSide == 0)
            {
                return 5;
            }

            if (!double.IsFinite(_distribution.Mean))
            {
                return 1;
            }

            if (!double.IsFinite(_distribution.Variance))
            {
                return 2;
            }

            if (!double.IsFinite(_distribution.Skewness))
            {
                return 3;
            }

            if (!double.IsFinite(_distribution.ExcessKurtosis))
            {
                return 4;
            }

            return 5;
        }
    }

    /// <value>The truncated mean; returns positive or negative infinity on the unbounded side if the base mean is not finite. / 切断後の平均。元の平均が有限でなく切断区間の片側が無限の場合は、その側の無限大。</value>
    /// <inheritdoc/>
    public double Mean
    {
        get
        {
            if (IsUntruncated)
            {
                return _distribution.Mean;
            }

            if (DivergentOrder == 1)
            {
                return UnboundedSide * double.PositiveInfinity;
            }

            return ComputeMean();
        }
    }

    /// <inheritdoc/>
    public double Variance
    {
        get
        {
            if (IsUntruncated)
            {
                return _distribution.Variance;
            }

            int divergent = DivergentOrder;
            if (divergent <= 2)
            {
                return divergent == 1 ? double.NaN : double.PositiveInfinity;
            }

            Span<double> central = stackalloc double[5];
            CentralMoments(2, central);
            return central[2];
        }
    }

    /// <inheritdoc/>
    public double StandardDeviation => IsUntruncated ? _distribution.StandardDeviation : Math.Sqrt(Variance);

    /// <inheritdoc/>
    public double Skewness
    {
        get
        {
            if (IsUntruncated)
            {
                return _distribution.Skewness;
            }

            int divergent = DivergentOrder;
            if (divergent <= 2)
            {
                return double.NaN;
            }

            if (divergent == 3)
            {
                return UnboundedSide * double.PositiveInfinity;
            }

            Span<double> central = stackalloc double[5];
            CentralMoments(3, central);
            return central[3] / (central[2] * Math.Sqrt(central[2]));
        }
    }

    /// <inheritdoc/>
    public double ExcessKurtosis
    {
        get
        {
            if (IsUntruncated)
            {
                return _distribution.ExcessKurtosis;
            }

            int divergent = DivergentOrder;
            if (divergent <= 2)
            {
                return double.NaN;
            }

            if (divergent <= 4)
            {
                return double.PositiveInfinity;
            }

            Span<double> central = stackalloc double[5];
            CentralMoments(4, central);
            return central[4] / (central[2] * central[2]) - 3;
        }
    }

    /// <inheritdoc/>
    public double Entropy
    {
        get
        {
            if (IsUntruncated)
            {
                return _distribution.Entropy;
            }

            var integrand = new EntropyIntegrand(this);
            Span<double> sums = stackalloc double[2];
            Integrate(ref integrand, sums);
            return sums[1] / sums[0] + Math.Log(sums[0]);
        }
    }

    /// <inheritdoc/>
    public double Pdf(double x)
    {
        if (double.IsNaN(x))
        {
            return double.NaN;
        }

        if (x < _lower || x > _upper)
        {
            return 0;
        }

        double f = _distribution.Pdf(x);
        if (_mass > 0 && f >= DistributionMath.MinNormal)
        {
            double density = f / _mass;
            if (density >= DistributionMath.MinNormal && density < double.PositiveInfinity)
            {
                return density;
            }
        }

        // Fall back to log-space when f or Z is subnormal/underflowed; preserve +∞.
        return f < double.PositiveInfinity ? Math.Exp(_distribution.LogPdf(x) - _logMass) : f;
    }

    /// <inheritdoc/>
    public double LogPdf(double x)
    {
        if (double.IsNaN(x))
        {
            return double.NaN;
        }

        if (x < _lower || x > _upper)
        {
            return double.NegativeInfinity;
        }

        return _distribution.LogPdf(x) - _logMass;
    }

    /// <inheritdoc/>
    public double Cdf(double x)
    {
        if (double.IsNaN(x))
        {
            return double.NaN;
        }

        if (x <= _lower)
        {
            return 0;
        }

        if (x >= _upper)
        {
            return 1;
        }

        return Fraction(LowerMass(x, out double logMass), logMass);
    }

    /// <inheritdoc/>
    public double Sf(double x)
    {
        if (double.IsNaN(x))
        {
            return double.NaN;
        }

        if (x <= _lower)
        {
            return 1;
        }

        if (x >= _upper)
        {
            return 0;
        }

        return Fraction(UpperMass(x, out double logMass), logMass);
    }

    /// <inheritdoc/>
    public double LogCdf(double x)
    {
        if (double.IsNaN(x))
        {
            return double.NaN;
        }

        if (x <= _lower)
        {
            return double.NegativeInfinity;
        }

        if (x >= _upper)
        {
            return 0;
        }

        LowerMass(x, out double logMass);
        double logFraction = logMass - _logMass;

        // For probabilities above 1/2, ln(1 - Sf(x)) avoids precision loss near 1.
        return logFraction > -DistributionMath.Ln2 ? DistributionMath.Log1p(-Sf(x)) : Math.Min(0, logFraction);
    }

    /// <inheritdoc/>
    public double LogSf(double x)
    {
        if (double.IsNaN(x))
        {
            return double.NaN;
        }

        if (x <= _lower)
        {
            return 0;
        }

        if (x >= _upper)
        {
            return double.NegativeInfinity;
        }

        UpperMass(x, out double logMass);
        double logFraction = logMass - _logMass;
        return logFraction > -DistributionMath.Ln2 ? DistributionMath.Log1p(-Cdf(x)) : Math.Min(0, logFraction);
    }

    /// <inheritdoc/>
    public double InverseCdf(double p)
    {
        if (!(p >= 0 && p <= 1))
        {
            return double.NaN;
        }

        if (p == 0)
        {
            return _lower;
        }

        if (p == 1)
        {
            return _upper;
        }

        return p <= 0.5 ? LowerQuantile(p, Math.Log(p)) : UpperQuantile(1 - p, DistributionMath.Log1p(-p));
    }

    /// <inheritdoc/>
    public double InverseSf(double q)
    {
        if (!(q >= 0 && q <= 1))
        {
            return double.NaN;
        }

        if (q == 0)
        {
            return _upper;
        }

        if (q == 1)
        {
            return _lower;
        }

        return q <= 0.5 ? UpperQuantile(q, Math.Log(q)) : LowerQuantile(1 - q, DistributionMath.Log1p(-q));
    }

    /// <inheritdoc/>
    public double InverseLogCdf(double logP)
    {
        if (!(logP <= 0))
        {
            return double.NaN;
        }

        if (logP == double.NegativeInfinity)
        {
            return _lower;
        }

        if (logP == 0)
        {
            return _upper;
        }

        return logP <= -DistributionMath.Ln2
            ? LowerQuantile(Math.Exp(logP), logP)
            : UpperQuantile(-DistributionMath.Expm1(logP), DistributionMath.Log1mExp(logP));
    }

    /// <inheritdoc/>
    public double InverseLogSf(double logQ)
    {
        if (!(logQ <= 0))
        {
            return double.NaN;
        }

        if (logQ == double.NegativeInfinity)
        {
            return _upper;
        }

        if (logQ == 0)
        {
            return _lower;
        }

        return logQ <= -DistributionMath.Ln2
            ? UpperQuantile(Math.Exp(logQ), logQ)
            : LowerQuantile(-DistributionMath.Expm1(logQ), DistributionMath.Log1mExp(logQ));
    }

    /// <inheritdoc/>
    public double Sample(IRandomSource random)
    {
        ArgumentNullException.ThrowIfNull(random);
        return SampleCore(random);
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, Span<double> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = SampleCore(random);
        }
    }

    // Inverse transform sampling; for u > 1/2, 1 - u is exact and evaluated from the upper tail.
    private double SampleCore(IRandomSource random)
    {
        double u = random.NextDoubleOpen();
        return u <= 0.5 ? LowerQuantile(u, Math.Log(u)) : UpperQuantile(1 - u, DistributionMath.Log1p(-u));
    }

    private double Fraction(double mass, double logMass) =>
        Math.Min(1, mass > 0 && _mass > 0 ? mass / _mass : Math.Exp(logMass - _logMass));

    private double LowerMass(double x, out double logMass) =>
        BaseMass(_lowerCdf, _lowerSf, _lowerLogCdf, _lowerLogSf,
            _distribution.Cdf(x), _distribution.Sf(x), double.NaN, double.NaN, _lower, x, out logMass);

    private double UpperMass(double x, out double logMass) =>
        BaseMass(_distribution.Cdf(x), _distribution.Sf(x), double.NaN, double.NaN,
            _upperCdf, _upperSf, _upperLogCdf, _upperLogSf, x, _upper, out logMass);

    // Computes P(u < Y <= v) via the smaller minuend of F(v)-F(u) or S(u)-S(v), switching to log-space
    // below LinearLimit and to density quadrature when cancellation exceeds CancellationRatio.
    // Returns 0 in linear space when subnormal/underflowed.
    private double BaseMass(double cdfU, double sfU, double logCdfU, double logSfU,
        double cdfV, double sfV, double logCdfV, double logSfV, double u, double v, out double logMass)
    {
        bool cdfSide = cdfV <= sfU;
        double minuend = cdfSide ? cdfV : sfU;
        if (minuend >= LinearLimit)
        {
            double mass = cdfSide ? cdfV - cdfU : sfU - sfV;
            if (mass >= CancellationRatio * minuend)
            {
                logMass = Math.Log(mass);
                return mass;
            }
        }
        else
        {
            double logMinuend, logSubtrahend;
            if (cdfSide)
            {
                logMinuend = double.IsNaN(logCdfV) ? _distribution.LogCdf(v) : logCdfV;
                logSubtrahend = double.IsNaN(logCdfU) ? _distribution.LogCdf(u) : logCdfU;
            }
            else
            {
                logMinuend = double.IsNaN(logSfU) ? _distribution.LogSf(u) : logSfU;
                logSubtrahend = double.IsNaN(logSfV) ? _distribution.LogSf(v) : logSfV;
            }

            double difference = logSubtrahend - logMinuend;
            if (difference <= LogCancellationLimit || double.IsNaN(difference))
            {
                logMass = logMinuend + DistributionMath.Log1mExp(difference);
                return 0;
            }
        }

        // Fall back to direct density quadrature on narrow intervals where subtraction suffers cancellation.
        return NarrowMass(u, v, out logMass);
    }

    private double NarrowMass(double u, double v, out double logMass)
    {
        if (!(u < v))
        {
            logMass = double.NegativeInfinity;
            return 0;
        }

        // Shift by midpoint log-density in extreme tails to prevent integrand underflow.
        double shift = _distribution.LogPdf(0.5 * u + 0.5 * v);
        if (!(shift < DistributionMath.LogProbabilityLinearLimit))
        {
            shift = 0;
        }

        var integrand = new BaseDensityIntegrand(_distribution, shift);
        Span<double> sums = stackalloc double[1];
        double total = 0;

        // Split at the base median to handle non-smooth peaks (e.g., Laplace).
        double median = _distribution.Median;
        double minimum = _distribution.SupportMinimum, maximum = _distribution.SupportMaximum;
        if (u < median && median < v)
        {
            DoubleExponentialQuadrature.Finite(ref integrand, u, median, minimum, maximum, sums);
            total += sums[0];
            DoubleExponentialQuadrature.Finite(ref integrand, median, v, minimum, maximum, sums);
            total += sums[0];
        }
        else
        {
            DoubleExponentialQuadrature.Finite(ref integrand, u, v, minimum, maximum, sums);
            total = sums[0];
        }

        // Compute linear mass directly without exp(logMass) to avoid ε|ln| amplification.
        logMass = Math.Log(total) + shift;
        double linear = total * Math.Exp(shift);
        return linear >= DistributionMath.MinNormal ? linear : 0;
    }

    private double LowerQuantile(double p, double logP)
    {
        double target = p * _mass;
        double x;
        bool refine;
        if (target >= LinearLimit)
        {
            // Use InverseSf(S(a) - pZ) when F(x) > 1/2; since pZ <= Z/2 <= S(a)/2, subtraction does not cancel.
            double cdf = _lowerCdf + target;
            double reference;
            if (cdf <= 0.5)
            {
                x = _distribution.InverseCdf(cdf);
                reference = _lowerCdf;
            }
            else
            {
                x = _distribution.InverseSf(_lowerSf - target);
                reference = _lowerSf;
            }

            refine = reference * CancellationRatio > target;
        }
        else
        {
            double logTarget = logP + _logMass;
            double logCdf = XMath.LogAddExp(_lowerLogCdf, logTarget);
            double logReference;
            if (logCdf <= -DistributionMath.Ln2)
            {
                x = _distribution.InverseLogCdf(logCdf);
                logReference = _lowerLogCdf;
            }
            else
            {
                x = _distribution.InverseLogSf(_lowerLogSf + DistributionMath.Log1mExp(logTarget - _lowerLogSf));
                logReference = _lowerLogSf;
            }

            refine = logTarget - logReference < LogCancellationRatio;
        }

        x = Math.Clamp(x, _lower, _upper);
        return refine ? Refine(x, p, logP, upper: false) : x;
    }

    private double UpperQuantile(double q, double logQ)
    {
        double target = q * _mass;
        double x;
        bool refine;
        if (target >= LinearLimit)
        {
            double sf = _upperSf + target;
            double reference;
            if (sf <= 0.5)
            {
                x = _distribution.InverseSf(sf);
                reference = _upperSf;
            }
            else
            {
                x = _distribution.InverseCdf(_upperCdf - target);
                reference = _upperCdf;
            }

            refine = reference * CancellationRatio > target;
        }
        else
        {
            double logTarget = logQ + _logMass;
            double logSf = XMath.LogAddExp(_upperLogSf, logTarget);
            double logReference;
            if (logSf <= -DistributionMath.Ln2)
            {
                x = _distribution.InverseLogSf(logSf);
                logReference = _upperLogSf;
            }
            else
            {
                x = _distribution.InverseLogCdf(_upperLogCdf + DistributionMath.Log1mExp(logTarget - _upperLogCdf));
                logReference = _upperLogCdf;
            }

            refine = logTarget - logReference < LogCancellationRatio;
        }

        x = Math.Clamp(x, _lower, _upper);
        return refine ? Refine(x, q, logQ, upper: true) : x;
    }

    // Newton refinement near truncation boundaries where adding pZ to the boundary CDF/SF loses precision.
    private double Refine(double x, double probability, double logProbability, bool upper)
    {
        for (int iteration = 0; iteration < 8; iteration++)
        {
            double logDensity = LogPdf(x);
            if (!double.IsFinite(logDensity))
            {
                break;
            }

            // Use linear residual when available (log residual leaves ε|ln p| error), otherwise log residual.
            double mass = upper ? UpperMass(x, out double logMass) : LowerMass(x, out logMass);
            double logFraction = logMass - _logMass;
            double density = Math.Exp(logDensity);
            double step;
            if (mass > 0 && _mass > 0 && probability >= DistributionMath.MinNormal && density >= DistributionMath.MinNormal)
            {
                step = (Math.Min(1, mass / _mass) - probability) / density;
            }
            else if (logFraction == double.NegativeInfinity)
            {
                step = -Math.Exp(logProbability - logDensity);
            }
            else
            {
                step = (logFraction - logProbability) * Math.Exp(logFraction - logDensity);
            }

            double next = Math.Clamp(upper ? x + step : x - step, _lower, _upper);
            if (!double.IsFinite(next))
            {
                break;
            }

            bool done = Math.Abs(next - x) <= 2 * DistributionMath.Epsilon * Math.Abs(x);
            x = next;
            if (done)
            {
                break;
            }
        }

        return x;
    }

    private double ComputeMean()
    {
        double center = Median;
        var integrand = new MomentIntegrand(this, center, 1);
        Span<double> sums = stackalloc double[2];
        Integrate(ref integrand, sums);
        return center + sums[1] / sums[0];
    }

    // Normalizing by the integrated total weight sums[0] cancels rounding bias in the density normalization.
    private void CentralMoments(int order, Span<double> central)
    {
        double mean = ComputeMean();
        var integrand = new MomentIntegrand(this, mean, order);
        Span<double> sums = stackalloc double[order + 1];
        Integrate(ref integrand, sums);
        for (int k = 2; k <= order; k++)
        {
            central[k] = sums[k] / sums[0];
        }
    }

    // Splits at the base median and truncated median; uses tanh-sinh on finite subintervals and exp-sinh on semi-infinite tails.
    private void Integrate<TIntegrand>(ref TIntegrand integrand, Span<double> sums)
        where TIntegrand : struct, IQuadratureIntegrand
    {
        int count = integrand.Count;
        Span<double> part = stackalloc double[DoubleExponentialQuadrature.MaxComponents];
        sums[..count].Clear();

        Span<double> points = stackalloc double[4];
        int n = 0;
        points[n++] = _lower;
        double first = _distribution.Median, second = Median;
        if (first > second)
        {
            (first, second) = (second, first);
        }

        if (_lower < first && first < _upper)
        {
            points[n++] = first;
        }

        if (points[n - 1] < second && second < _upper)
        {
            points[n++] = second;
        }

        points[n++] = _upper;

        for (int i = 0; i + 1 < n; i++)
        {
            double a = points[i], b = points[i + 1];
            if (double.IsFinite(a) && double.IsFinite(b))
            {
                DoubleExponentialQuadrature.Finite(ref integrand, a, b, _distribution.SupportMinimum, _distribution.SupportMaximum, part);
            }
            else
            {
                // Characteristic scale for the semi-infinite tail is the distance from the finite bound to the truncated median.
                double scale = double.IsFinite(a) ? a - _lower : _upper - b;
                if (!(scale > 0 && scale < double.PositiveInfinity))
                {
                    scale = Math.Max(1, Math.Abs(double.IsFinite(a) ? a : b));
                }

                if (double.IsFinite(a))
                {
                    DoubleExponentialQuadrature.SemiInfinite(ref integrand, a, scale, 1, part);
                }
                else
                {
                    DoubleExponentialQuadrature.SemiInfinite(ref integrand, b, scale, -1, part);
                }
            }

            for (int j = 0; j < count; j++)
            {
                sums[j] += part[j];
            }
        }
    }

    private readonly struct BaseDensityIntegrand(TDistribution distribution, double shift) : IQuadratureIntegrand
    {
        public int Count => 1;

        public double Density(double x) => shift == 0 ? distribution.Pdf(x) : Math.Exp(distribution.LogPdf(x) - shift);

        public void Evaluate(double x, double density, double mass, Span<double> values) => values[0] = mass;
    }

    private readonly struct MomentIntegrand(TruncatedDistribution<TDistribution> distribution, double center, int order)
        : IQuadratureIntegrand
    {
        public int Count => order + 1;

        public double Density(double x) => distribution.Pdf(x);

        public void Evaluate(double x, double density, double mass, Span<double> values)
        {
            double d = x - center;
            double v = mass;
            values[0] = v;
            for (int k = 1; k <= order; k++)
            {
                v *= d;
                values[k] = v;
            }
        }
    }

    private readonly struct EntropyIntegrand(TruncatedDistribution<TDistribution> distribution) : IQuadratureIntegrand
    {
        public int Count => 2;

        public double Density(double x) => distribution.Pdf(x);

        public void Evaluate(double x, double density, double mass, Span<double> values)
        {
            values[0] = mass;
            values[1] = -mass * Math.Log(density);
        }
    }
}
