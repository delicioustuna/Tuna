using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the location-scale Student's t-distribution <c>X = μ + σT</c>, where <c>T</c> has <c>ν</c> degrees of freedom.
/// <para>位置・尺度付きスチューデントの t 分布（<c>X = μ + σT</c>、<c>T</c> は自由度 <c>ν</c> の標準 t 分布）を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Setting <see cref="DegreesOfFreedom"/> to <see cref="double.PositiveInfinity"/> yields the normal distribution <c>N(μ, σ²)</c>.
/// Moment existence depends on <c>ν</c>: <see cref="Mean"/> is <c>μ</c> for <c>ν &gt; 1</c> (otherwise <see cref="double.NaN"/>);
/// <see cref="Variance"/> and <see cref="StandardDeviation"/> are finite for <c>ν &gt; 2</c>, <see cref="double.PositiveInfinity"/> for <c>1 &lt; ν ≤ 2</c>, and <see cref="double.NaN"/> for <c>ν ≤ 1</c>;
/// <see cref="Skewness"/> is 0 for <c>ν &gt; 3</c> (otherwise <see cref="double.NaN"/>);
/// <see cref="ExcessKurtosis"/> is finite for <c>ν &gt; 4</c>, <see cref="double.PositiveInfinity"/> for <c>2 &lt; ν ≤ 4</c>, and <see cref="double.NaN"/> for <c>ν ≤ 2</c>.
/// </para>
/// <para>
/// <see cref="DegreesOfFreedom"/> に <see cref="double.PositiveInfinity"/> を指定すると、正規分布 <c>N(μ, σ²)</c> として振る舞います。
/// モーメントの存在条件は <c>ν</c> に依存します。<see cref="Mean"/> は <c>ν &gt; 1</c> で <c>μ</c>（それ以外は <see cref="double.NaN"/>）、
/// <see cref="Variance"/> および <see cref="StandardDeviation"/> は <c>ν &gt; 2</c> で有限（<c>1 &lt; ν ≤ 2</c> で <see cref="double.PositiveInfinity"/>、<c>ν ≤ 1</c> で <see cref="double.NaN"/>）、
/// <see cref="Skewness"/> は <c>ν &gt; 3</c> で 0（それ以外は <see cref="double.NaN"/>）、
/// <see cref="ExcessKurtosis"/> は <c>ν &gt; 4</c> で有限（<c>2 &lt; ν ≤ 4</c> で <see cref="double.PositiveInfinity"/>、<c>ν ≤ 2</c> で <see cref="double.NaN"/>）です。
/// </para>
/// </remarks>
public readonly partial struct StudentTDistribution : IContinuousDistribution
{
    private readonly double _degreesOfFreedom;
    private readonly double _location;
    private readonly double _scale;

    /// <summary>
    /// Initializes a new instance of <see cref="StudentTDistribution"/> with the specified degrees of freedom, location, and scale.
    /// <para>自由度・位置・尺度を指定して <see cref="StudentTDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="degreesOfFreedom">The degrees of freedom <c>ν</c> (positive, or <see cref="double.PositiveInfinity"/> for the normal limit). / 自由度 <c>ν</c>（正、または正規極限として <see cref="double.PositiveInfinity"/>）。</param>
    /// <param name="location">The location parameter <c>μ</c> (finite). / 位置パラメーター <c>μ</c>（有限）。</param>
    /// <param name="scale">The scale parameter <c>σ</c> (positive and finite). / 尺度パラメーター <c>σ</c>（正かつ有限）。</param>
    /// <exception cref="ArgumentOutOfRangeException">Any parameter is outside its valid domain. / いずれかのパラメーターが有効な範囲外の場合。</exception>
    public StudentTDistribution(double degreesOfFreedom, double location = 0, double scale = 1)
        : this(degreesOfFreedom, location, scale, forDifferentiation: false)
    {
    }

    private StudentTDistribution(double degreesOfFreedom, double location, double scale, bool forDifferentiation)
    {
        DistributionArgument.PositiveOrInfinity(degreesOfFreedom, nameof(degreesOfFreedom), forDifferentiation);
        DistributionArgument.Finite(location, nameof(location), forDifferentiation);
        DistributionArgument.PositiveFinite(scale, nameof(scale), forDifferentiation);
        _degreesOfFreedom = degreesOfFreedom;
        _location = location;
        _scale = scale;
    }

    /// <summary>
    /// Gets the degrees of freedom <c>ν</c>.
    /// <para>自由度 <c>ν</c> を取得します。</para>
    /// </summary>
    public double DegreesOfFreedom => _degreesOfFreedom;

    /// <summary>
    /// Gets the location parameter <c>μ</c>.
    /// <para>位置パラメーター <c>μ</c> を取得します。</para>
    /// </summary>
    public double Location => _location;

    /// <summary>
    /// Gets the scale parameter <c>σ</c>.
    /// <para>尺度パラメーター <c>σ</c> を取得します。</para>
    /// </summary>
    public double Scale => _scale;

    private bool IsNormal => double.IsPositiveInfinity(_degreesOfFreedom);

    /// <inheritdoc/>
    public double Mean => _degreesOfFreedom > 1 ? _location : double.NaN;

    /// <inheritdoc/>
    public double Variance
    {
        get
        {
            double nu = _degreesOfFreedom;
            if (nu > 2)
            {
                return IsNormal ? _scale * _scale : _scale * _scale * (nu / (nu - 2));
            }

            return nu > 1 ? double.PositiveInfinity : double.NaN;
        }
    }

    /// <inheritdoc/>
    public double StandardDeviation
    {
        get
        {
            double nu = _degreesOfFreedom;
            if (nu > 2)
            {
                return IsNormal ? _scale : _scale * Math.Sqrt(nu / (nu - 2));
            }

            return nu > 1 ? double.PositiveInfinity : double.NaN;
        }
    }

    /// <inheritdoc/>
    public double Skewness => _degreesOfFreedom > 3 ? 0 : double.NaN;

    /// <inheritdoc/>
    public double ExcessKurtosis
    {
        get
        {
            double nu = _degreesOfFreedom;
            if (nu > 4)
            {
                return 6 / (nu - 4);
            }

            return nu > 2 ? double.PositiveInfinity : double.NaN;
        }
    }

    /// <inheritdoc/>
    public double Entropy
    {
        get
        {
            if (IsNormal)
            {
                return 0.5 + DistributionMath.LnSqrt2Pi + Math.Log(_scale);
            }

            double half = 0.5 * _degreesOfFreedom;
            return (half + 0.5) * DistributionMath.DigammaHalfStep(half) + DistributionMath.LnSqrt2Pi
                - DistributionMath.LogGammaHalfStepCorrection(half) + Math.Log(_scale);
        }
    }

    /// <inheritdoc/>
    public double Median => _location;

    /// <inheritdoc/>
    public double SupportMinimum => double.NegativeInfinity;

    /// <inheritdoc/>
    public double SupportMaximum => double.PositiveInfinity;

    private double Standardize(double x) => DistributionMath.Standardize(x, _location, _scale);

    /// <inheritdoc/>
    public double Pdf(double x)
    {
        if (IsNormal)
        {
            return XMath.Special.NormalPdf(Standardize(x)) / _scale;
        }

        return Math.Exp(LogPdf(x));
    }

    /// <inheritdoc/>
    public double LogPdf(double x) => LogPdfStandardized(Standardize(x));

    private double LogPdfStandardized(double t)
    {
        if (IsNormal)
        {
            return XMath.Special.NormalLogPdf(t) - Math.Log(_scale);
        }

        return LogPdfValue(t, LogPdfConstant(), Math.Log(_scale));
    }

    private double LogPdfConstant()
    {
        double nu = _degreesOfFreedom;
        if (!(nu > 0 && nu < 20))
        {
            return ComputeLogPdfConstant(nu);
        }

        // 小さい自由度の漸化式は、同じ自由度で位置だけが変わる回帰の観測間で共有できる。
        if (nu == normalizationCache.FirstDegrees)
        {
            return normalizationCache.FirstValue;
        }

        if (nu == normalizationCache.SecondDegrees)
        {
            return normalizationCache.SecondValue;
        }

        double value = ComputeLogPdfConstant(nu);
        normalizationCache = (nu, value, normalizationCache.FirstDegrees, normalizationCache.FirstValue);
        return value;
    }

    [ThreadStatic]
    private static (double FirstDegrees, double FirstValue, double SecondDegrees, double SecondValue) normalizationCache;

    private static double ComputeLogPdfConstant(double degreesOfFreedom) =>
        DistributionMath.LogGammaHalfStepCorrection(0.5 * degreesOfFreedom) - DistributionMath.LnSqrt2Pi;

    private double LogPdfValue(double t, double constant, double logScale) =>
        constant - 0.5 * (_degreesOfFreedom + 1) * LogOnePlusRatio(t) - logScale;

    // Overflow-safe ln(1 + t²/ν).
    private double LogOnePlusRatio(double t)
    {
        double a = Math.Abs(t);
        double ratio = a * a / _degreesOfFreedom;
        if (ratio < 1e300)
        {
            return ratio > 1 ? Math.Log(ratio) + DistributionMath.Log1p(1 / ratio) : DistributionMath.Log1p(ratio);
        }

        // For t²/ν >= 1e300, ln(1 + 1/ratio) is below ε.
        return double.IsNaN(a) ? double.NaN : 2 * Math.Log(a) - Math.Log(_degreesOfFreedom);
    }

    private double Tail(double a)
    {
        double nu = _degreesOfFreedom;
        double ratio = a * a / nu;
        if (ratio < 1e300)
        {
            double x = 1 / (1 + ratio);
            if (x <= 0.5)
            {
                return 0.5 * XMath.Special.RegularizedIncompleteBeta(0.5 * nu, 0.5, x);
            }

            double y = ratio / (1 + ratio);
            return 0.5 * XMath.Special.RegularizedIncompleteBetaComplement(0.5, 0.5 * nu, y);
        }

        return 0.5 * Math.Exp(LogTailFromLogX(a));
    }

    private double LogTail(double a)
    {
        double nu = _degreesOfFreedom;
        double ratio = a * a / nu;
        if (ratio < 1e300)
        {
            double x = 1 / (1 + ratio);
            if (x <= 0.5)
            {
                return XMath.Special.LogRegularizedIncompleteBeta(0.5 * nu, 0.5, x) - DistributionMath.Ln2;
            }

            double y = ratio / (1 + ratio);
            return XMath.Special.LogRegularizedIncompleteBetaComplement(0.5, 0.5 * nu, y) - DistributionMath.Ln2;
        }

        return LogTailFromLogX(a) - DistributionMath.Ln2;
    }

    private double LogTailFromLogX(double a)
    {
        double logX = Math.Log(_degreesOfFreedom) - 2 * Math.Log(a);
        return DistributionMath.LogRegularizedIncompleteBetaFromLog(0.5 * _degreesOfFreedom, 0.5, logX);
    }

    /// <inheritdoc/>
    public double Cdf(double x)
    {
        double t = Standardize(x);
        if (IsNormal)
        {
            return XMath.Special.NormalCdf(t);
        }

        if (double.IsNaN(t))
        {
            return double.NaN;
        }

        return t <= 0 ? Tail(-t) : 1 - Tail(t);
    }

    /// <inheritdoc/>
    public double Sf(double x)
    {
        double t = Standardize(x);
        if (IsNormal)
        {
            return XMath.Special.NormalSf(t);
        }

        if (double.IsNaN(t))
        {
            return double.NaN;
        }

        return t >= 0 ? Tail(t) : 1 - Tail(-t);
    }

    /// <inheritdoc/>
    public double LogCdf(double x)
    {
        double t = Standardize(x);
        if (IsNormal)
        {
            return XMath.Special.NormalLogCdf(t);
        }

        if (double.IsNaN(t))
        {
            return double.NaN;
        }

        return t <= 0 ? LogTail(-t) : DistributionMath.Log1p(-Tail(t));
    }

    /// <inheritdoc/>
    public double LogSf(double x)
    {
        double t = Standardize(x);
        if (IsNormal)
        {
            return XMath.Special.NormalLogSf(t);
        }

        if (double.IsNaN(t))
        {
            return double.NaN;
        }

        return t >= 0 ? LogTail(t) : DistributionMath.Log1p(-Tail(-t));
    }

    private double TwoSidedQuantile(double r, bool negative)
    {
        if (r >= 1)
        {
            return DistributionMath.Unstandardize(_location, _scale, negative ? -0.0 : 0.0);
        }

        if (r <= 0)
        {
            return negative ? double.NegativeInfinity : double.PositiveInfinity;
        }

        double nu = _degreesOfFreedom;
        double x = XMath.Special.InverseRegularizedIncompleteBeta(0.5 * nu, 0.5, r);
        if (x < DistributionMath.MinNormal)
        {
            return DistributionMath.UnstandardizeFromLog(_location, _scale, negative, LogTwoSidedQuantileFromLog(Math.Log(r)));
        }

        double a;
        if (x <= 0.5)
        {
            a = Math.Sqrt(nu) * Math.Sqrt((1 - x) / x);
        }
        else
        {
            // When x is near 1, solve directly for y = 1 - x to avoid cancellation.
            double y = XMath.Special.InverseRegularizedIncompleteBetaComplement(0.5, 0.5 * nu, r);
            a = Math.Sqrt(nu) * Math.Sqrt(y / (1 - y));
        }

        return DistributionMath.Unstandardize(_location, _scale, negative ? -a : a);
    }

    // Solves ln(2 P(T > a)) = logR in log-space since a can overflow for small degrees of freedom.
    private double LogTwoSidedQuantileFromLog(double logR)
    {
        double logX = BetaDistribution.LowerLogQuantileFromLog(0.5 * _degreesOfFreedom, 0.5, logR);
        return 0.5 * (Math.Log(_degreesOfFreedom) + DistributionMath.Log1p(-Math.Exp(logX)) - logX);
    }

    /// <inheritdoc/>
    public double InverseCdf(double p)
    {
        if (!(p >= 0 && p <= 1))
        {
            return double.NaN;
        }

        if (IsNormal)
        {
            return DistributionMath.Unstandardize(_location, _scale, XMath.Special.InverseNormalCdf(p));
        }

        return p <= 0.5 ? TwoSidedQuantile(2 * p, true) : TwoSidedQuantile(2 * (1 - p), false);
    }

    /// <inheritdoc/>
    public double InverseSf(double q)
    {
        if (!(q >= 0 && q <= 1))
        {
            return double.NaN;
        }

        if (IsNormal)
        {
            return DistributionMath.Unstandardize(_location, _scale, XMath.Special.InverseNormalSf(q));
        }

        return q <= 0.5 ? TwoSidedQuantile(2 * q, false) : TwoSidedQuantile(2 * (1 - q), true);
    }

    /// <inheritdoc/>
    public double InverseLogCdf(double logP)
    {
        if (!(logP <= 0))
        {
            return double.NaN;
        }

        if (IsNormal)
        {
            return DistributionMath.Unstandardize(_location, _scale, XMath.Special.InverseNormalLogCdf(logP));
        }

        if (logP > -DistributionMath.Ln2)
        {
            return InverseSf(-DistributionMath.Expm1(logP));
        }

        if (logP >= DistributionMath.LogProbabilityLinearLimit)
        {
            return InverseCdf(Math.Exp(logP));
        }

        return DistributionMath.UnstandardizeFromLog(_location, _scale, true, LogTwoSidedQuantileFromLog(logP + DistributionMath.Ln2));
    }

    /// <inheritdoc/>
    public double InverseLogSf(double logQ)
    {
        if (!(logQ <= 0))
        {
            return double.NaN;
        }

        if (IsNormal)
        {
            return DistributionMath.Unstandardize(_location, _scale, XMath.Special.InverseNormalLogSf(logQ));
        }

        if (logQ > -DistributionMath.Ln2)
        {
            return InverseCdf(-DistributionMath.Expm1(logQ));
        }

        if (logQ >= DistributionMath.LogProbabilityLinearLimit)
        {
            return InverseSf(Math.Exp(logQ));
        }

        return DistributionMath.UnstandardizeFromLog(_location, _scale, false, LogTwoSidedQuantileFromLog(logQ + DistributionMath.Ln2));
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
        double z = Ziggurat.StandardNormal(random);
        if (IsNormal)
        {
            return DistributionMath.Unstandardize(_location, _scale, z);
        }

        // V = 2G ~ χ²(ν), T = Z * sqrt(ν / V) evaluated in log-space to avoid underflow for tiny ν.
        double logV = DistributionMath.Ln2 + Sampling.LogStandardGamma(random, 0.5 * _degreesOfFreedom);
        return DistributionMath.Unstandardize(_location, _scale, z * Math.Exp(0.5 * (Math.Log(_degreesOfFreedom) - logV)));
    }
}
