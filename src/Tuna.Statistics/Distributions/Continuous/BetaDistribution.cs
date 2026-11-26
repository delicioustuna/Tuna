using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the beta distribution on <c>[0, 1]</c> with shape parameters <c>α</c> and <c>β</c>.
/// <para>形状パラメーター <c>α</c>、<c>β</c> を持つ <c>[0, 1]</c> 上のベータ分布を表します。</para>
/// </summary>
public readonly partial struct BetaDistribution : IContinuousDistribution
{
    private readonly double _alpha;
    private readonly double _beta;

    /// <summary>
    /// Initializes a new instance of <see cref="BetaDistribution"/> with the specified shape parameters.
    /// <para>2 つの形状パラメーターを指定して <see cref="BetaDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="alpha">The first shape parameter <c>α</c> (positive and finite). / 第 1 形状パラメーター <c>α</c>（正かつ有限）。</param>
    /// <param name="beta">The second shape parameter <c>β</c> (positive and finite). / 第 2 形状パラメーター <c>β</c>（正かつ有限）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="alpha"/> or <paramref name="beta"/> is not positive and finite. / <paramref name="alpha"/> または <paramref name="beta"/> が正の有限値でない場合。</exception>
    public BetaDistribution(double alpha, double beta)
        : this(alpha, beta, forDifferentiation: false)
    {
    }

    private BetaDistribution(double alpha, double beta, bool forDifferentiation)
    {
        DistributionArgument.PositiveFinite(alpha, nameof(alpha), forDifferentiation);
        DistributionArgument.PositiveFinite(beta, nameof(beta), forDifferentiation);
        _alpha = alpha;
        _beta = beta;
    }

    /// <summary>
    /// Gets the first shape parameter <c>α</c>.
    /// <para>第 1 形状パラメーター <c>α</c> を取得します。</para>
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Gets the second shape parameter <c>β</c>.
    /// <para>第 2 形状パラメーター <c>β</c> を取得します。</para>
    /// </summary>
    public double Beta => _beta;

    /// <inheritdoc/>
    public double Mean => _alpha / (_alpha + _beta);

    /// <inheritdoc/>
    public double Variance
    {
        get
        {
            double s = _alpha + _beta;
            return _alpha / s * (_beta / s) / (s + 1);
        }
    }

    /// <inheritdoc/>
    public double StandardDeviation => Math.Sqrt(Variance);

    /// <inheritdoc/>
    public double Skewness
    {
        get
        {
            double s = _alpha + _beta;
            return 2 * (_beta - _alpha) / (s + 2) * Math.Sqrt((s + 1) / _alpha / _beta);
        }
    }

    /// <inheritdoc/>
    public double ExcessKurtosis
    {
        get
        {
            double s = _alpha + _beta;
            double d = _alpha - _beta;
            return 6 * (d / _alpha * (d / _beta) * ((s + 1) / (s + 2)) / (s + 3) - 1 / (s + 3));
        }
    }

    /// <inheritdoc/>
    public double Entropy
    {
        get
        {
            double a = _alpha, b = _beta, s = a + b;
            double logarithms = 0.5 * (DistributionMath.LnTwoPi + Math.Log(a / s) + Math.Log(b / s) - Math.Log(s));
            double rational = 0.5 - 0.5 / a - 0.5 / b + 1 / s;
            double stirling = DistributionMath.StirlingRemainder(a) + DistributionMath.StirlingRemainder(b)
                - DistributionMath.StirlingRemainder(s);
            double digamma = (a - 1) * DistributionMath.DigammaRemainder(a) + (b - 1) * DistributionMath.DigammaRemainder(b)
                - (s - 2) * DistributionMath.DigammaRemainder(s);
            return logarithms + rational + stirling + digamma;
        }
    }

    /// <inheritdoc/>
    public double Median => InverseCdf(0.5);

    /// <inheritdoc/>
    public double SupportMinimum => 0;

    /// <inheritdoc/>
    public double SupportMaximum => 1;

    private static double EndpointDensity(double near, double far) =>
        near < 1 ? double.PositiveInfinity : near == 1 ? far : 0;

    /// <inheritdoc/>
    public double Pdf(double x)
    {
        if (double.IsNaN(x))
        {
            return double.NaN;
        }

        if (x < 0 || x > 1)
        {
            return 0;
        }

        if (x == 0)
        {
            return EndpointDensity(_alpha, _beta);
        }

        if (x == 1)
        {
            return EndpointDensity(_beta, _alpha);
        }

        double y = 1 - x;
        double prefix = XMath.Special.BetaPrefix(_alpha, _beta, x, y);
        if (prefix >= DistributionMath.MinNormal && prefix < double.PositiveInfinity && x >= DistributionMath.MinNormal)
        {
            return prefix / x / y;
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

        if (x < 0 || x > 1)
        {
            return double.NegativeInfinity;
        }

        if (x == 0)
        {
            return Math.Log(EndpointDensity(_alpha, _beta));
        }

        if (x == 1)
        {
            return Math.Log(EndpointDensity(_beta, _alpha));
        }

        double y = 1 - x;
        return XMath.Special.LogBetaPrefix(_alpha, _beta, x, y) - Math.Log(x) - DistributionMath.Log1p(-x);
    }

    /// <inheritdoc/>
    public double Cdf(double x) => x <= 0 ? 0 : x >= 1 ? 1 : XMath.Special.RegularizedIncompleteBeta(_alpha, _beta, x);

    /// <inheritdoc/>
    public double Sf(double x) => x <= 0 ? 1 : x >= 1 ? 0 : XMath.Special.RegularizedIncompleteBetaComplement(_alpha, _beta, x);

    /// <inheritdoc/>
    public double LogCdf(double x) =>
        x <= 0 ? double.NegativeInfinity : x >= 1 ? 0 : XMath.Special.LogRegularizedIncompleteBeta(_alpha, _beta, x);

    /// <inheritdoc/>
    public double LogSf(double x) =>
        x <= 0 ? 0 : x >= 1 ? double.NegativeInfinity : XMath.Special.LogRegularizedIncompleteBetaComplement(_alpha, _beta, x);

    /// <inheritdoc/>
    public double InverseCdf(double p)
    {
        if (!(p >= 0 && p <= 1))
        {
            return double.NaN;
        }

        double x = XMath.Special.InverseRegularizedIncompleteBeta(_alpha, _beta, p);
        if (x < DistributionMath.MinNormal && p > 0)
        {
            return Math.Exp(LowerLogQuantileFromLog(_alpha, _beta, Math.Log(p)));
        }

        return x;
    }

    /// <inheritdoc/>
    public double InverseSf(double q)
    {
        if (!(q >= 0 && q <= 1))
        {
            return double.NaN;
        }

        double x = XMath.Special.InverseRegularizedIncompleteBetaComplement(_alpha, _beta, q);
        if (x < DistributionMath.MinNormal && q < 1)
        {
            return Math.Exp(LowerLogQuantileFromLog(_alpha, _beta, DistributionMath.Log1p(-q)));
        }

        return x;
    }

    /// <inheritdoc/>
    public double InverseLogCdf(double logP)
    {
        if (!(logP <= 0))
        {
            return double.NaN;
        }

        if (logP > -DistributionMath.Ln2)
        {
            return InverseSf(-DistributionMath.Expm1(logP));
        }

        if (logP >= DistributionMath.LogProbabilityLinearLimit)
        {
            return InverseCdf(Math.Exp(logP));
        }

        return Math.Exp(LowerLogQuantileFromLog(_alpha, _beta, logP));
    }

    /// <inheritdoc/>
    public double InverseLogSf(double logQ)
    {
        if (!(logQ <= 0))
        {
            return double.NaN;
        }

        if (logQ > -DistributionMath.Ln2)
        {
            return InverseCdf(-DistributionMath.Expm1(logQ));
        }

        if (logQ >= DistributionMath.LogProbabilityLinearLimit)
        {
            return InverseSf(Math.Exp(logQ));
        }

        // In extreme upper tails, solve for ln(1 - x) via symmetry Beta(β, α).
        return -DistributionMath.Expm1(LowerLogQuantileFromLog(_beta, _alpha, logQ));
    }

    // Solves ln I_x(a, b) = logP for t = ln x in the deep lower tail.
    internal static double LowerLogQuantileFromLog(double a, double b, double logP)
    {
        if (double.IsNegativeInfinity(logP))
        {
            return double.NegativeInfinity;
        }

        double logMinNormal = Math.Log(DistributionMath.MinNormal);
        // Asymptotic initial guess from I_x(a, b) ≈ x^a / (a B(a, b)).
        double guess = Math.Min((logP + Math.Log(a) + XMath.Special.LogBeta(a, b)) / a, -DistributionMath.Epsilon);
        return DistributionMath.SolveIncreasing(
            (double t, out double derivative) =>
            {
                double value = DistributionMath.LogRegularizedIncompleteBetaFromLog(a, b, t);
                if (t < logMinNormal)
                {
                    derivative = a;
                }
                else
                {
                    double x = Math.Exp(t);
                    double y = 1 - x;
                    derivative = Math.Exp(XMath.Special.LogBetaPrefix(a, b, x, y) - DistributionMath.Log1p(-x) - value);
                }

                return value;
            },
            logP, guess, 1 / Math.Max(a, 1), upper: 0);
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
        double logA = Sampling.LogStandardGamma(random, _alpha);
        double logB = Sampling.LogStandardGamma(random, _beta);
        return 1 / (1 + Math.Exp(logB - logA));
    }
}
