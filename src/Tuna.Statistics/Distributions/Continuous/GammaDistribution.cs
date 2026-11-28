using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the gamma distribution parameterized by shape <c>k</c> and scale <c>θ</c> on <c>[0, ∞)</c>.
/// <para>形状 <c>k</c> と尺度 <c>θ</c> でパラメーター化された <c>[0, ∞)</c> 上のガンマ分布を表します。</para>
/// </summary>
public readonly partial struct GammaDistribution : IContinuousDistribution
{
    private readonly double _shape;
    private readonly double _scale;

    /// <summary>
    /// Initializes a new instance of <see cref="GammaDistribution"/> with the specified shape and scale.
    /// <para>形状および尺度パラメーターを指定して <see cref="GammaDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="shape">The shape parameter <c>k</c> (positive and finite). / 形状パラメーター <c>k</c>（正かつ有限）。</param>
    /// <param name="scale">The scale parameter <c>θ</c> (positive and finite). / 尺度パラメーター <c>θ</c>（正かつ有限）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="shape"/> or <paramref name="scale"/> is not positive and finite. / <paramref name="shape"/> または <paramref name="scale"/> が正の有限値でない場合。</exception>
    public GammaDistribution(double shape, double scale)
        : this(shape, scale, forDifferentiation: false)
    {
    }

    internal GammaDistribution(double shape, double scale, bool forDifferentiation)
    {
        DistributionArgument.PositiveFinite(shape, nameof(shape), forDifferentiation);
        DistributionArgument.PositiveFinite(scale, nameof(scale), forDifferentiation);
        _shape = shape;
        _scale = scale;
    }

    /// <summary>
    /// Creates a <see cref="GammaDistribution"/> from shape <c>k</c> and rate <c>β = 1/θ</c>.
    /// <para>形状 <c>k</c> と率 <c>β = 1/θ</c> を指定して <see cref="GammaDistribution"/> を生成します。</para>
    /// </summary>
    /// <param name="shape">The shape parameter <c>k</c> (positive and finite). / 形状パラメーター <c>k</c>（正かつ有限）。</param>
    /// <param name="rate">The rate parameter <c>β = 1/θ</c> (positive, with finite reciprocal). / 率パラメーター <c>β = 1/θ</c>（正で、逆数が有限）。</param>
    /// <returns>A gamma distribution with shape <paramref name="shape"/> and scale <c>1 / rate</c>. / 形状 <paramref name="shape"/>、尺度 <c>1 / rate</c> のガンマ分布。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="shape"/> or <paramref name="rate"/> is not positive and finite, or <c>1 / rate</c> overflows. / <paramref name="shape"/> または <paramref name="rate"/> が正の有限値でない、または <c>1 / rate</c> がオーバーフローする場合。</exception>
    public static GammaDistribution FromShapeRate(double shape, double rate)
    {
        DistributionArgument.PositiveFinite(rate, nameof(rate));
        return new GammaDistribution(shape, 1 / rate);
    }

    /// <summary>
    /// Gets the shape parameter <c>k</c>.
    /// <para>形状パラメーター <c>k</c> を取得します。</para>
    /// </summary>
    public double Shape => _shape;

    /// <summary>
    /// Gets the scale parameter <c>θ</c>.
    /// <para>尺度パラメーター <c>θ</c> を取得します。</para>
    /// </summary>
    public double Scale => _scale;

    /// <summary>
    /// Gets the rate parameter <c>1/θ</c>.
    /// <para>率パラメーター <c>1/θ</c> を取得します。</para>
    /// </summary>
    public double Rate => 1 / _scale;

    /// <inheritdoc/>
    public double Mean => _shape * _scale;

    /// <inheritdoc/>
    public double Variance => _shape * _scale * _scale;

    /// <inheritdoc/>
    public double StandardDeviation => Math.Sqrt(_shape) * _scale;

    /// <inheritdoc/>
    public double Skewness => 2 / Math.Sqrt(_shape);

    /// <inheritdoc/>
    public double ExcessKurtosis => 6 / _shape;

    /// <inheritdoc/>
    public double Entropy => Math.Log(_scale) + StandardEntropy(_shape);

    internal static double StandardEntropy(double shape)
    {
        if (shape < 1)
        {
            return shape + XMath.Special.LogGamma(shape) + (1 - shape) * XMath.Special.Digamma(shape);
        }

        return 0.5 * (NumericConstants.LnTau + Math.Log(shape)) + 0.5 - 0.5 / shape
            + DistributionMath.StirlingRemainder(shape) + (shape - 1) * DistributionMath.DigammaRemainder(shape);
    }

    /// <inheritdoc/>
    public double Median => InverseCdf(0.5);

    /// <inheritdoc/>
    public double SupportMinimum => 0;

    /// <inheritdoc/>
    public double SupportMaximum => double.PositiveInfinity;

    private static bool IsRegular(double y) => y >= NumericConstants.MinNormal && y < double.PositiveInfinity;

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

        if (x == 0)
        {
            return _shape < 1 ? double.PositiveInfinity : _shape == 1 ? 1 / _scale : 0;
        }

        double y = x / _scale;
        if (IsRegular(y))
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

        if (x < 0 || double.IsPositiveInfinity(x))
        {
            return double.NegativeInfinity;
        }

        if (x == 0)
        {
            return _shape < 1 ? double.PositiveInfinity : _shape == 1 ? -Math.Log(_scale) : double.NegativeInfinity;
        }

        double y = x / _scale;
        if (IsRegular(y))
        {
            return XMath.Special.LogGammaPrefix(_shape, y) - Math.Log(x);
        }

        double logY = Math.Log(x) - Math.Log(_scale);
        return (_shape - 1) * logY - y - XMath.Special.LogGamma(_shape) - Math.Log(_scale);
    }

    private double LogLowerProbability(double x)
    {
        double y = x / _scale;
        return y >= NumericConstants.MinNormal
            ? XMath.Special.LogRegularizedGammaP(_shape, y)
            : DistributionMath.LogRegularizedGammaPFromLog(_shape, Math.Log(x) - Math.Log(_scale));
    }

    /// <inheritdoc/>
    public double Cdf(double x)
    {
        if (!(x > 0))
        {
            return double.IsNaN(x) ? double.NaN : 0;
        }

        double y = x / _scale;
        return y >= NumericConstants.MinNormal ? XMath.Special.RegularizedGammaP(_shape, y) : Math.Exp(LogLowerProbability(x));
    }

    /// <inheritdoc/>
    public double Sf(double x)
    {
        if (!(x > 0))
        {
            return double.IsNaN(x) ? double.NaN : 1;
        }

        double y = x / _scale;
        return y >= NumericConstants.MinNormal ? XMath.Special.RegularizedGammaQ(_shape, y) : -DistributionMath.Expm1(LogLowerProbability(x));
    }

    /// <inheritdoc/>
    public double LogCdf(double x)
    {
        if (!(x > 0))
        {
            return double.IsNaN(x) ? double.NaN : double.NegativeInfinity;
        }

        return LogLowerProbability(x);
    }

    /// <inheritdoc/>
    public double LogSf(double x)
    {
        if (!(x > 0))
        {
            return double.IsNaN(x) ? double.NaN : 0;
        }

        double y = x / _scale;
        return y >= NumericConstants.MinNormal
            ? XMath.Special.LogRegularizedGammaQ(_shape, y)
            : DistributionMath.Log1mExp(LogLowerProbability(x));
    }

    /// <inheritdoc/>
    public double InverseCdf(double p)
    {
        if (!(p >= 0 && p <= 1))
        {
            return double.NaN;
        }

        double y = XMath.Special.InverseRegularizedGammaP(_shape, p);
        if (y < NumericConstants.MinNormal && p > 0)
        {
            return LowerQuantileFromLog(Math.Log(p));
        }

        return _scale * y;
    }

    /// <inheritdoc/>
    public double InverseSf(double q)
    {
        if (!(q >= 0 && q <= 1))
        {
            return double.NaN;
        }

        double y = XMath.Special.InverseRegularizedGammaQ(_shape, q);
        if (y < NumericConstants.MinNormal && q < 1)
        {
            return LowerQuantileFromLog(DistributionMath.Log1p(-q));
        }

        return _scale * y;
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

        if (logP >= DistributionMath.LogProbabilityLinearLimit)
        {
            return InverseCdf(Math.Exp(logP));
        }

        return LowerQuantileFromLog(logP);
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

        // For tiny shapes, Q(k, y) ≈ k*E₁(y) is tiny even for tiny y (e.g., k = 1e-308, logQ = -703 => y = 1e-208);
        // linear-space solver StandardUpperQuantileFromLog cannot reach near 0, so route through InverseSf or lower-tail log solve.
        if (logQ >= Math.Log(NumericConstants.MinNormal))
        {
            return InverseSf(Math.Exp(logQ));
        }

        if (logQ > XMath.Special.LogRegularizedGammaQ(_shape, NumericConstants.MinNormal))
        {
            return LowerQuantileFromLog(DistributionMath.Log1mExp(logQ));
        }

        return _scale * StandardUpperQuantileFromLog(_shape, logQ);
    }

    private double LowerQuantileFromLog(double logP)
    {
        if (double.IsNegativeInfinity(logP))
        {
            return 0;
        }

        double t = StandardLowerLogQuantileFromLog(_shape, logP);
        return Math.Exp(t + Math.Log(_scale));
    }

    // Solves ln P(shape, e^t) = logP for t = ln(x/scale).
    internal static double StandardLowerLogQuantileFromLog(double shape, double logP)
    {
        double guess = (logP + XMath.Special.LogGamma1p(shape)) / shape;
        if (shape > 1)
        {
            double approximation = DistributionMath.WilsonHilferty(shape, XMath.Special.InverseNormalLogCdf(logP));
            if (approximation > 0)
            {
                guess = Math.Max(guess, Math.Log(approximation));
            }
        }

        double logMinNormal = Math.Log(NumericConstants.MinNormal);
        return DistributionMath.SolveIncreasing(
            (double t, out double derivative) =>
            {
                double value = DistributionMath.LogRegularizedGammaPFromLog(shape, t);
                derivative = t < logMinNormal
                    ? shape
                    : Math.Exp(XMath.Special.LogGammaPrefix(shape, Math.Exp(t)) - value);
                return value;
            },
            logP, guess, 1 / Math.Sqrt(Math.Max(shape, 1)), upper: Math.Log(shape) + 1);
    }

    // Solves ln Q(shape, y) = logQ for y in the deep upper tail.
    internal static double StandardUpperQuantileFromLog(double shape, double logQ)
    {
        if (double.IsNegativeInfinity(logQ))
        {
            return double.PositiveInfinity;
        }

        // Asymptotic initial guess from Q(k, y) ≈ y^(k-1) e^(-y) / Γ(k).
        double guess = -logQ + (shape - 1) * Math.Log(-logQ) - XMath.Special.LogGamma(shape);
        if (shape > 1)
        {
            double approximation = DistributionMath.WilsonHilferty(shape, XMath.Special.InverseNormalLogSf(logQ));
            if (approximation > guess || !(guess > 0))
            {
                guess = approximation;
            }
        }

        if (!(guess > 0))
        {
            guess = -logQ;
        }

        return DistributionMath.SolveIncreasing(
            (double y, out double derivative) =>
            {
                double value = -XMath.Special.LogRegularizedGammaQ(shape, y);
                derivative = Math.Exp(XMath.Special.LogGammaPrefix(shape, y) - Math.Log(y) + value);
                return value;
            },
            -logQ, guess, Math.Max(1, Math.Sqrt(shape)), lower: 0);
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
            ? _scale * Sampling.StandardGamma(random, _shape)
            : Math.Exp(Sampling.LogStandardGamma(random, _shape) + Math.Log(_scale));
}
