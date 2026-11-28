using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the inverse gamma distribution with shape <c>α</c> and scale <c>β</c> on <c>(0, ∞)</c>, where <c>1/X</c> follows a gamma distribution with shape <c>α</c> and rate <c>β</c>.
/// <para>形状 <c>α</c>、尺度 <c>β</c> を持つ <c>(0, ∞)</c> 上の逆ガンマ分布（<c>1/X</c> が形状 <c>α</c>、率 <c>β</c> のガンマ分布に従う分布）を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Moments depend on <c>α</c>: <see cref="Mean"/> is finite for <c>α &gt; 1</c> (otherwise <see cref="double.PositiveInfinity"/>);
/// <see cref="Variance"/> and <see cref="StandardDeviation"/> are finite for <c>α &gt; 2</c>, <see cref="double.PositiveInfinity"/> for <c>1 &lt; α ≤ 2</c>, and <see cref="double.NaN"/> for <c>α ≤ 1</c>;
/// <see cref="Skewness"/> is finite for <c>α &gt; 3</c>, <see cref="double.PositiveInfinity"/> for <c>2 &lt; α ≤ 3</c>, and <see cref="double.NaN"/> for <c>α ≤ 2</c>;
/// <see cref="ExcessKurtosis"/> is finite for <c>α &gt; 4</c>, <see cref="double.PositiveInfinity"/> for <c>2 &lt; α ≤ 4</c>, and <see cref="double.NaN"/> for <c>α ≤ 2</c>.
/// </para>
/// <para>
/// モーメントの存在条件は <c>α</c> に依存します。<see cref="Mean"/> は <c>α &gt; 1</c> で有限（それ以外は <see cref="double.PositiveInfinity"/>）、
/// <see cref="Variance"/> および <see cref="StandardDeviation"/> は <c>α &gt; 2</c> で有限（<c>1 &lt; α ≤ 2</c> で <see cref="double.PositiveInfinity"/>、<c>α ≤ 1</c> で <see cref="double.NaN"/>）、
/// <see cref="Skewness"/> は <c>α &gt; 3</c> で有限（<c>2 &lt; α ≤ 3</c> で <see cref="double.PositiveInfinity"/>、<c>α ≤ 2</c> で <see cref="double.NaN"/>）、
/// <see cref="ExcessKurtosis"/> は <c>α &gt; 4</c> で有限（<c>2 &lt; α ≤ 4</c> で <see cref="double.PositiveInfinity"/>、<c>α ≤ 2</c> で <see cref="double.NaN"/>）です。
/// </para>
/// </remarks>
public readonly partial struct InverseGammaDistribution : IContinuousDistribution
{
    private readonly double _shape;
    private readonly double _scale;

    /// <summary>
    /// Initializes a new instance of <see cref="InverseGammaDistribution"/> with the specified shape and scale.
    /// <para>形状および尺度パラメーターを指定して <see cref="InverseGammaDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="shape">The shape parameter <c>α</c> (positive and finite). / 形状パラメーター <c>α</c>（正かつ有限）。</param>
    /// <param name="scale">The scale parameter <c>β</c> (positive and finite). / 尺度パラメーター <c>β</c>（正かつ有限）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="shape"/> or <paramref name="scale"/> is not positive and finite. / <paramref name="shape"/> または <paramref name="scale"/> が正の有限値でない場合。</exception>
    public InverseGammaDistribution(double shape, double scale)
        : this(shape, scale, forDifferentiation: false)
    {
    }

    private InverseGammaDistribution(double shape, double scale, bool forDifferentiation)
    {
        DistributionArgument.PositiveFinite(shape, nameof(shape), forDifferentiation);
        DistributionArgument.PositiveFinite(scale, nameof(scale), forDifferentiation);
        _shape = shape;
        _scale = scale;
    }

    /// <summary>
    /// Gets the shape parameter <c>α</c>.
    /// <para>形状パラメーター <c>α</c> を取得します。</para>
    /// </summary>
    public double Shape => _shape;

    /// <summary>
    /// Gets the scale parameter <c>β</c>.
    /// <para>尺度パラメーター <c>β</c> を取得します。</para>
    /// </summary>
    public double Scale => _scale;

    /// <inheritdoc/>
    public double Mean => _shape > 1 ? _scale / (_shape - 1) : double.PositiveInfinity;

    /// <inheritdoc/>
    public double Variance
    {
        get
        {
            if (_shape > 2)
            {
                double mean = _scale / (_shape - 1);
                return mean * (mean / (_shape - 2));
            }

            return _shape > 1 ? double.PositiveInfinity : double.NaN;
        }
    }

    /// <inheritdoc/>
    public double StandardDeviation
    {
        get
        {
            if (_shape > 2)
            {
                return _scale / (_shape - 1) / Math.Sqrt(_shape - 2);
            }

            return _shape > 1 ? double.PositiveInfinity : double.NaN;
        }
    }

    /// <inheritdoc/>
    public double Skewness
    {
        get
        {
            if (_shape > 3)
            {
                return 4 * Math.Sqrt(_shape - 2) / (_shape - 3);
            }

            return _shape > 2 ? double.PositiveInfinity : double.NaN;
        }
    }

    /// <inheritdoc/>
    public double ExcessKurtosis
    {
        get
        {
            if (_shape > 4)
            {
                return 6 * (5 * _shape - 11) / (_shape - 3) / (_shape - 4);
            }

            return _shape > 2 ? double.PositiveInfinity : double.NaN;
        }
    }

    /// <inheritdoc/>
    public double Entropy =>
        Math.Log(_scale) + (XMath.Constants.LnSqrtTwoPi - 1.5 * Math.Log(_shape) + 0.5 + 0.5 / _shape
            + DistributionMath.StirlingRemainder(_shape) + (_shape + 1) * DistributionMath.DigammaRemainder(_shape));

    /// <inheritdoc/>
    public double Median => InverseCdf(0.5);

    /// <inheritdoc/>
    public double SupportMinimum => 0;

    /// <inheritdoc/>
    public double SupportMaximum => double.PositiveInfinity;

    private double LogUpperProbability(double x)
    {
        double y = _scale / x;
        return y >= NumericConstants.MinNormal
            ? XMath.Special.LogRegularizedGammaP(_shape, y)
            : DistributionMath.LogRegularizedGammaPFromLog(_shape, Math.Log(_scale) - Math.Log(x));
    }

    /// <inheritdoc/>
    public double Pdf(double x)
    {
        if (double.IsNaN(x))
        {
            return double.NaN;
        }

        if (x <= 0 || double.IsPositiveInfinity(x))
        {
            return 0;
        }

        double y = _scale / x;
        if (y >= NumericConstants.MinNormal && y < double.PositiveInfinity)
        {
            double prefix = XMath.Special.GammaPrefix(_shape, y);
            if (prefix >= NumericConstants.MinNormal && prefix < double.PositiveInfinity)
            {
                return prefix / x;
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

        if (x <= 0 || double.IsPositiveInfinity(x))
        {
            return double.NegativeInfinity;
        }

        double y = _scale / x;
        if (double.IsPositiveInfinity(y))
        {
            return double.NegativeInfinity;
        }

        if (y >= NumericConstants.MinNormal)
        {
            return XMath.Special.LogGammaPrefix(_shape, y) - Math.Log(x);
        }

        return _shape * Math.Log(_scale) - (_shape + 1) * Math.Log(x) - y - XMath.Special.LogGamma(_shape);
    }

    /// <inheritdoc/>
    public double Cdf(double x)
    {
        if (!(x > 0))
        {
            return double.IsNaN(x) ? double.NaN : 0;
        }

        double y = _scale / x;
        return y >= NumericConstants.MinNormal ? XMath.Special.RegularizedGammaQ(_shape, y) : -DistributionMath.Expm1(LogUpperProbability(x));
    }

    /// <inheritdoc/>
    public double Sf(double x)
    {
        if (!(x > 0))
        {
            return double.IsNaN(x) ? double.NaN : 1;
        }

        double y = _scale / x;
        return y >= NumericConstants.MinNormal ? XMath.Special.RegularizedGammaP(_shape, y) : Math.Exp(LogUpperProbability(x));
    }

    /// <inheritdoc/>
    public double LogCdf(double x)
    {
        if (!(x > 0))
        {
            return double.IsNaN(x) ? double.NaN : double.NegativeInfinity;
        }

        double y = _scale / x;
        return y >= NumericConstants.MinNormal
            ? XMath.Special.LogRegularizedGammaQ(_shape, y)
            : DistributionMath.Log1mExp(LogUpperProbability(x));
    }

    /// <inheritdoc/>
    public double LogSf(double x)
    {
        if (!(x > 0))
        {
            return double.IsNaN(x) ? double.NaN : 0;
        }

        return LogUpperProbability(x);
    }

    private double FromStandard(double y) => _scale / y;

    private double FromStandardLog(double t)
    {
        double y = Math.Exp(t);
        return y >= NumericConstants.MinNormal && y < double.PositiveInfinity ? _scale / y : Math.Exp(Math.Log(_scale) - t);
    }

    /// <inheritdoc/>
    public double InverseCdf(double p)
    {
        if (!(p >= 0 && p <= 1))
        {
            return double.NaN;
        }

        double y = XMath.Special.InverseRegularizedGammaQ(_shape, p);
        if (y < NumericConstants.MinNormal && p < 1)
        {
            return FromStandardLog(GammaDistribution.StandardLowerLogQuantileFromLog(_shape, DistributionMath.Log1p(-p)));
        }

        return FromStandard(y);
    }

    /// <inheritdoc/>
    public double InverseSf(double q)
    {
        if (!(q >= 0 && q <= 1))
        {
            return double.NaN;
        }

        double y = XMath.Special.InverseRegularizedGammaP(_shape, q);
        if (y < NumericConstants.MinNormal && q > 0)
        {
            return FromStandardLog(GammaDistribution.StandardLowerLogQuantileFromLog(_shape, Math.Log(q)));
        }

        return FromStandard(y);
    }

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

        // Same tiny-shape guard as GammaDistribution.InverseLogSf since F(x) = Q(α, β/x).
        if (logP >= Math.Log(NumericConstants.MinNormal))
        {
            return InverseCdf(Math.Exp(logP));
        }

        if (logP > XMath.Special.LogRegularizedGammaQ(_shape, NumericConstants.MinNormal))
        {
            return FromStandardLog(GammaDistribution.StandardLowerLogQuantileFromLog(_shape, DistributionMath.Log1mExp(logP)));
        }

        return FromStandard(GammaDistribution.StandardUpperQuantileFromLog(_shape, logP));
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

        if (logQ >= DistributionMath.LogProbabilityLinearLimit)
        {
            return InverseSf(Math.Exp(logQ));
        }

        return double.IsNegativeInfinity(logQ)
            ? double.PositiveInfinity
            : FromStandardLog(GammaDistribution.StandardLowerLogQuantileFromLog(_shape, logQ));
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

    private double SampleCore(IRandomSource random) =>
        _shape >= 1
            ? _scale / Sampling.StandardGamma(random, _shape)
            : Math.Exp(Math.Log(_scale) - Sampling.LogStandardGamma(random, _shape));
}
