using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the Fisher–Snedecor F-distribution <c>F(d₁, d₂)</c>.
/// <para>F 分布 <c>F(d₁, d₂)</c> を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Moments depend on <c>d₂</c>: <see cref="Mean"/> is finite for <c>d₂ &gt; 2</c> (otherwise <see cref="double.PositiveInfinity"/>);
/// <see cref="Variance"/> and <see cref="StandardDeviation"/> are finite for <c>d₂ &gt; 4</c>, <see cref="double.PositiveInfinity"/> for <c>2 &lt; d₂ ≤ 4</c>, and <see cref="double.NaN"/> for <c>d₂ ≤ 2</c>;
/// <see cref="Skewness"/> is finite for <c>d₂ &gt; 6</c>, <see cref="double.PositiveInfinity"/> for <c>4 &lt; d₂ ≤ 6</c>, and <see cref="double.NaN"/> for <c>d₂ ≤ 4</c>;
/// <see cref="ExcessKurtosis"/> is finite for <c>d₂ &gt; 8</c>, <see cref="double.PositiveInfinity"/> for <c>6 &lt; d₂ ≤ 8</c>, and <see cref="double.NaN"/> for <c>d₂ ≤ 6</c>.
/// </para>
/// <para>
/// モーメントの存在条件は <c>d₂</c> に依存します。<see cref="Mean"/> は <c>d₂ &gt; 2</c> で有限（それ以外は <see cref="double.PositiveInfinity"/>）、
/// <see cref="Variance"/> および <see cref="StandardDeviation"/> は <c>d₂ &gt; 4</c> で有限（<c>2 &lt; d₂ ≤ 4</c> で <see cref="double.PositiveInfinity"/>、<c>d₂ ≤ 2</c> で <see cref="double.NaN"/>）、
/// <see cref="Skewness"/> は <c>d₂ &gt; 6</c> で有限（<c>4 &lt; d₂ ≤ 6</c> で <see cref="double.PositiveInfinity"/>、<c>d₂ ≤ 4</c> で <see cref="double.NaN"/>）、
/// <see cref="ExcessKurtosis"/> は <c>d₂ &gt; 8</c> で有限（<c>6 &lt; d₂ ≤ 8</c> で <see cref="double.PositiveInfinity"/>、<c>d₂ ≤ 6</c> で <see cref="double.NaN"/>）です。
/// </para>
/// </remarks>
public readonly partial struct FDistribution : IContinuousDistribution
{
    private readonly double _numerator;
    private readonly double _denominator;

    // d₁/d₂ and ln(d₁/d₂); when the ratio is not normal/finite, coordinate transforms evaluate in log-space.
    private readonly double _ratio;
    private readonly double _logRatio;
    private readonly bool _ratioIsRegular;

    /// <summary>
    /// Initializes a new instance of <see cref="FDistribution"/> with the specified degrees of freedom.
    /// <para>分子・分母の自由度を指定して <see cref="FDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="numeratorDegreesOfFreedom">The numerator degrees of freedom <c>d₁</c> (positive and finite; need not be an integer). / 分子の自由度 <c>d₁</c>（正かつ有限。整数でなくてもよい）。</param>
    /// <param name="denominatorDegreesOfFreedom">The denominator degrees of freedom <c>d₂</c> (positive and finite; need not be an integer). / 分母の自由度 <c>d₂</c>（正かつ有限。整数でなくてもよい）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="numeratorDegreesOfFreedom"/> or <paramref name="denominatorDegreesOfFreedom"/> is not positive and finite. / <paramref name="numeratorDegreesOfFreedom"/> または <paramref name="denominatorDegreesOfFreedom"/> が正の有限値でない場合。</exception>
    public FDistribution(double numeratorDegreesOfFreedom, double denominatorDegreesOfFreedom)
        : this(numeratorDegreesOfFreedom, denominatorDegreesOfFreedom, forDifferentiation: false)
    {
    }

    private FDistribution(double numeratorDegreesOfFreedom, double denominatorDegreesOfFreedom, bool forDifferentiation)
    {
        DistributionArgument.PositiveFinite(numeratorDegreesOfFreedom, nameof(numeratorDegreesOfFreedom), forDifferentiation);
        DistributionArgument.PositiveFinite(denominatorDegreesOfFreedom, nameof(denominatorDegreesOfFreedom), forDifferentiation);
        _numerator = numeratorDegreesOfFreedom;
        _denominator = denominatorDegreesOfFreedom;
        _ratio = numeratorDegreesOfFreedom / denominatorDegreesOfFreedom;
        _logRatio = Math.Log(numeratorDegreesOfFreedom) - Math.Log(denominatorDegreesOfFreedom);
        _ratioIsRegular = _ratio >= NumericConstants.MinNormal && _ratio < double.PositiveInfinity;
    }

    /// <summary>
    /// Gets the numerator degrees of freedom <c>d₁</c>.
    /// <para>分子の自由度 <c>d₁</c> を取得します。</para>
    /// </summary>
    public double NumeratorDegreesOfFreedom => _numerator;

    /// <summary>
    /// Gets the denominator degrees of freedom <c>d₂</c>.
    /// <para>分母の自由度 <c>d₂</c> を取得します。</para>
    /// </summary>
    public double DenominatorDegreesOfFreedom => _denominator;

    private double A => 0.5 * _numerator;

    private double B => 0.5 * _denominator;

    /// <inheritdoc/>
    public double Mean => _denominator > 2 ? _denominator / (_denominator - 2) : double.PositiveInfinity;

    // Scaled ratio w = (d₂ - 2)/d₁ avoids overflow in moment formulas for large degrees of freedom.
    private double W => (_denominator - 2) / _numerator;

    /// <inheritdoc/>
    public double Variance
    {
        get
        {
            double d2 = _denominator;
            if (d2 > 4)
            {
                double m = d2 / (d2 - 2);
                return 2 * m * m * (1 + W) / (d2 - 4);
            }

            return d2 > 2 ? double.PositiveInfinity : double.NaN;
        }
    }

    /// <inheritdoc/>
    public double StandardDeviation
    {
        get
        {
            double d2 = _denominator;
            if (d2 > 4)
            {
                return d2 / (d2 - 2) * Math.Sqrt(2 * (1 + W) / (d2 - 4));
            }

            return d2 > 2 ? double.PositiveInfinity : double.NaN;
        }
    }

    /// <inheritdoc/>
    public double Skewness
    {
        get
        {
            double d2 = _denominator;
            if (d2 > 6)
            {
                double w = W;
                return (2 + w) / (d2 - 6) * Math.Sqrt(8 * (d2 - 4) / (1 + w));
            }

            return d2 > 4 ? double.PositiveInfinity : double.NaN;
        }
    }

    /// <inheritdoc/>
    public double ExcessKurtosis
    {
        get
        {
            double d2 = _denominator;
            if (d2 > 8)
            {
                double w = W;
                return 12 * (5 * d2 - 22 + (d2 - 4) * w * (w / (1 + w))) / (d2 - 6) / (d2 - 8);
            }

            return d2 > 6 ? double.PositiveInfinity : double.NaN;
        }
    }

    /// <inheritdoc/>
    public double Entropy
    {
        get
        {
            double a = A, b = B, s = a + b;
            double logarithms = XMath.Constants.LnSqrtTwoPi + 0.5 * (DistributionMath.Log1p(a / b) - Math.Log(a));
            double rational = 0.5 - 0.5 / a + 0.5 / b;
            double stirling = DistributionMath.StirlingRemainder(a) + DistributionMath.StirlingRemainder(b)
                - DistributionMath.StirlingRemainder(s);
            double digamma = (a - 1) * DistributionMath.DigammaRemainder(a) + (b + 1) * DistributionMath.DigammaRemainder(b)
                - s * DistributionMath.DigammaRemainder(s);
            return logarithms + rational + stirling + digamma;
        }
    }

    /// <inheritdoc/>
    public double Median => InverseCdf(0.5);

    /// <inheritdoc/>
    public double SupportMinimum => 0;

    /// <inheritdoc/>
    public double SupportMaximum => double.PositiveInfinity;

    // Returns the smaller of u = r/(1+r) and v = 1/(1+r) (r = d₁x/d₂) in `small`, and true if v is smaller (r > 1).
    private bool SmallCoordinate(double x, out double small)
    {
        double r = _ratio * x;
        if (_ratioIsRegular && r >= NumericConstants.MinNormal && r < double.PositiveInfinity)
        {
            if (r <= 1)
            {
                small = r / (1 + r);
                return false;
            }

            small = 1 / (1 + r);
            return true;
        }

        double logR = _logRatio + Math.Log(x);
        bool upper = logR > 0;
        small = Math.Exp(LogSmallCoordinate(logR, upper));
        return upper;
    }

    private static double LogSmallCoordinate(double logR, bool upper) =>
        upper ? -DistributionMath.Log1pExp(logR) : logR - DistributionMath.Log1pExp(logR);

    private double LogR(double x) => _logRatio + Math.Log(x);

    /// <inheritdoc/>
    public double Pdf(double x)
    {
        if (double.IsNaN(x))
        {
            return double.NaN;
        }

        if (x < 0 || double.IsPositiveInfinity(x))
        {
            return 0;
        }

        double a = A, b = B;
        if (x == 0)
        {
            return a < 1 ? double.PositiveInfinity : a == 1 ? Math.Exp(_logRatio) * b : 0;
        }

        bool upper = SmallCoordinate(x, out double small);
        if (small >= NumericConstants.MinNormal)
        {
            // Pass the more accurate (smaller) coordinate first so the second argument matches 1 - small.
            double prefix = upper
                ? XMath.Special.BetaPrefix(b, a, small, 1 - small)
                : XMath.Special.BetaPrefix(a, b, small, 1 - small);
            if (prefix >= NumericConstants.MinNormal && prefix < double.PositiveInfinity)
            {
                double f = prefix / x;
                if (f >= NumericConstants.MinNormal && f < double.PositiveInfinity)
                {
                    return f;
                }
            }
        }

        return Math.Exp(LogPdf(x));
    }

    /// <inheritdoc/>
    public double LogPdf(double x)
    {
        if (double.IsNaN(x))
        {
            return double.NaN;
        }

        if (x < 0 || double.IsPositiveInfinity(x))
        {
            return double.NegativeInfinity;
        }

        if (x == 0)
        {
            return Math.Log(Pdf(0));
        }

        double a = A, b = B;
        bool upper = SmallCoordinate(x, out double small);
        if (small >= NumericConstants.MinNormal)
        {
            double logPrefix = upper
                ? XMath.Special.LogBetaPrefix(b, a, small, 1 - small)
                : XMath.Special.LogBetaPrefix(a, b, small, 1 - small);
            return logPrefix - Math.Log(x);
        }

        // Deep tail where r or 1/r < MinNormal: ln(1 + r) and ln(1 + 1/r) are negligible relative to ln x.
        double logBeta = XMath.Special.LogBeta(a, b);
        double logX = Math.Log(x);
        return upper
            ? -b * _logRatio - (b + 1) * logX - logBeta
            : a * _logRatio + (a - 1) * logX - logBeta;
    }

    /// <inheritdoc/>
    public double Cdf(double x)
    {
        if (!(x > 0))
        {
            return double.IsNaN(x) ? double.NaN : 0;
        }

        if (double.IsPositiveInfinity(x))
        {
            return 1;
        }

        double a = A, b = B;
        bool upper = SmallCoordinate(x, out double small);
        if (upper)
        {
            return small >= NumericConstants.MinNormal
                ? XMath.Special.RegularizedIncompleteBetaComplement(b, a, small)
                : -DistributionMath.Expm1(DistributionMath.LogRegularizedIncompleteBetaFromLog(b, a, LogSmallCoordinate(LogR(x), true)));
        }

        return small >= NumericConstants.MinNormal
            ? XMath.Special.RegularizedIncompleteBeta(a, b, small)
            : Math.Exp(DistributionMath.LogRegularizedIncompleteBetaFromLog(a, b, LogSmallCoordinate(LogR(x), false)));
    }

    /// <inheritdoc/>
    public double Sf(double x)
    {
        if (!(x > 0))
        {
            return double.IsNaN(x) ? double.NaN : 1;
        }

        if (double.IsPositiveInfinity(x))
        {
            return 0;
        }

        double a = A, b = B;
        bool upper = SmallCoordinate(x, out double small);
        if (upper)
        {
            return small >= NumericConstants.MinNormal
                ? XMath.Special.RegularizedIncompleteBeta(b, a, small)
                : Math.Exp(DistributionMath.LogRegularizedIncompleteBetaFromLog(b, a, LogSmallCoordinate(LogR(x), true)));
        }

        return small >= NumericConstants.MinNormal
            ? XMath.Special.RegularizedIncompleteBetaComplement(a, b, small)
            : -DistributionMath.Expm1(DistributionMath.LogRegularizedIncompleteBetaFromLog(a, b, LogSmallCoordinate(LogR(x), false)));
    }

    /// <inheritdoc/>
    public double LogCdf(double x)
    {
        if (!(x > 0))
        {
            return double.IsNaN(x) ? double.NaN : double.NegativeInfinity;
        }

        if (double.IsPositiveInfinity(x))
        {
            return 0;
        }

        double a = A, b = B;
        bool upper = SmallCoordinate(x, out double small);
        if (upper)
        {
            return small >= NumericConstants.MinNormal
                ? XMath.Special.LogRegularizedIncompleteBetaComplement(b, a, small)
                : DistributionMath.Log1mExp(DistributionMath.LogRegularizedIncompleteBetaFromLog(b, a, LogSmallCoordinate(LogR(x), true)));
        }

        return small >= NumericConstants.MinNormal
            ? XMath.Special.LogRegularizedIncompleteBeta(a, b, small)
            : DistributionMath.LogRegularizedIncompleteBetaFromLog(a, b, LogSmallCoordinate(LogR(x), false));
    }

    /// <inheritdoc/>
    public double LogSf(double x)
    {
        if (!(x > 0))
        {
            return double.IsNaN(x) ? double.NaN : 0;
        }

        if (double.IsPositiveInfinity(x))
        {
            return double.NegativeInfinity;
        }

        double a = A, b = B;
        bool upper = SmallCoordinate(x, out double small);
        if (upper)
        {
            return small >= NumericConstants.MinNormal
                ? XMath.Special.LogRegularizedIncompleteBeta(b, a, small)
                : DistributionMath.LogRegularizedIncompleteBetaFromLog(b, a, LogSmallCoordinate(LogR(x), true));
        }

        return small >= NumericConstants.MinNormal
            ? XMath.Special.LogRegularizedIncompleteBetaComplement(a, b, small)
            : DistributionMath.Log1mExp(DistributionMath.LogRegularizedIncompleteBetaFromLog(a, b, LogSmallCoordinate(LogR(x), false)));
    }

    private double FromCoordinate(bool isV, double small)
    {
        double odds = isV ? (1 - small) / small : small / (1 - small);
        double x = odds / _ratio;
        if (_ratioIsRegular && x >= NumericConstants.MinNormal && x < double.PositiveInfinity)
        {
            return x;
        }

        return Math.Exp((isV ? -1 : 1) * (Math.Log(small) - DistributionMath.Log1p(-small)) - _logRatio);
    }

    private double FromLogCoordinate(bool isV, double logSmall)
    {
        double logOdds = logSmall - DistributionMath.Log1mExp(logSmall);
        return Math.Exp((isV ? -logOdds : logOdds) - _logRatio);
    }

    // Solves I_s(first, second) = probability; when s > 0.5, re-solves for the complementary coordinate 1 - s directly.
    private double Quantile(double first, double second, double probability, bool isV)
    {
        double s = XMath.Special.InverseRegularizedIncompleteBeta(first, second, probability);
        if (s < NumericConstants.MinNormal)
        {
            return FromLogCoordinate(isV, BetaDistribution.LowerLogQuantileFromLog(first, second, Math.Log(probability)));
        }

        if (s <= 0.5)
        {
            return FromCoordinate(isV, s);
        }

        double t = XMath.Special.InverseRegularizedIncompleteBetaComplement(second, first, probability);
        if (t < NumericConstants.MinNormal)
        {
            return FromLogCoordinate(!isV, BetaDistribution.LowerLogQuantileFromLog(second, first, DistributionMath.Log1p(-probability)));
        }

        return FromCoordinate(!isV, t);
    }

    private double QuantileFromLog(double first, double second, double logProbability, bool isV) =>
        FromLogCoordinate(isV, BetaDistribution.LowerLogQuantileFromLog(first, second, logProbability));

    /// <inheritdoc/>
    public double InverseCdf(double p) => p >= 0 && p <= 1 ? Quantile(A, B, p, false) : double.NaN;

    /// <inheritdoc/>
    public double InverseSf(double q) => q >= 0 && q <= 1 ? Quantile(B, A, q, true) : double.NaN;

    /// <inheritdoc/>
    public double InverseLogCdf(double logP)
    {
        if (!(logP <= 0))
        {
            return double.NaN;
        }

        if (logP > -XMath.Constants.Ln2)
        {
            return InverseSf(-DistributionMath.Expm1(logP));
        }

        return logP >= DistributionMath.LogProbabilityLinearLimit ? InverseCdf(Math.Exp(logP)) : QuantileFromLog(A, B, logP, false);
    }

    /// <inheritdoc/>
    public double InverseLogSf(double logQ)
    {
        if (!(logQ <= 0))
        {
            return double.NaN;
        }

        if (logQ > -XMath.Constants.Ln2)
        {
            return InverseCdf(-DistributionMath.Expm1(logQ));
        }

        return logQ >= DistributionMath.LogProbabilityLinearLimit ? InverseSf(Math.Exp(logQ)) : QuantileFromLog(B, A, logQ, true);
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

    private double SampleCore(IRandomSource random)
    {
        double logA = Sampling.LogStandardGamma(random, A);
        double logB = Sampling.LogStandardGamma(random, B);
        return Math.Exp(logA - logB - _logRatio);
    }
}
