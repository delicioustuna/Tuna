using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the continuous uniform distribution <c>U(a, b)</c> on <c>[a, b]</c>.
/// <para><c>[a, b]</c> 上の連続一様分布 <c>U(a, b)</c> を表します。</para>
/// </summary>
public readonly partial struct UniformDistribution : IContinuousDistribution
{
    private readonly double _lower;
    private readonly double _upper;
    private readonly double _width;

    /// <summary>
    /// Initializes a new instance of <see cref="UniformDistribution"/> over the specified interval.
    /// <para>区間の下端と上端を指定して <see cref="UniformDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="lower">The lower bound <c>a</c> (finite). / 下端 <c>a</c>（有限）。</param>
    /// <param name="upper">The upper bound <c>b</c> (<c>a &lt; b</c> with finite width <c>b - a</c>). / 上端 <c>b</c>（<c>a &lt; b</c> かつ幅 <c>b - a</c> が有限）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="lower"/> or <paramref name="upper"/> is not finite, or <c>upper - lower</c> is not positive and finite. / <paramref name="lower"/> または <paramref name="upper"/> が有限でない、または <c>upper - lower</c> が正の有限値でない場合。</exception>
    public UniformDistribution(double lower, double upper)
        : this(lower, upper, forDifferentiation: false)
    {
    }

    private UniformDistribution(double lower, double upper, bool forDifferentiation)
    {
        DistributionArgument.Finite(lower, nameof(lower), forDifferentiation);
        DistributionArgument.Finite(upper, nameof(upper), forDifferentiation);
        double width = upper - lower;
        if (!(width > 0 && double.IsFinite(width)))
        {
            throw DistributionArgument.ParameterOutOfRange(forDifferentiation, nameof(upper), upper, "The upper bound must exceed the lower bound by a finite width.");
        }

        _lower = lower;
        _upper = upper;
        _width = width;
    }

    /// <summary>
    /// Gets the lower bound <c>a</c>.
    /// <para>下端 <c>a</c> を取得します。</para>
    /// </summary>
    public double Lower => _lower;

    /// <summary>
    /// Gets the upper bound <c>b</c>.
    /// <para>上端 <c>b</c> を取得します。</para>
    /// </summary>
    public double Upper => _upper;

    /// <inheritdoc/>
    public double Mean => 0.5 * _lower + 0.5 * _upper;

    /// <inheritdoc/>
    public double Variance => _width * _width / 12;

    /// <inheritdoc/>
    public double StandardDeviation => _width / Math.Sqrt(12);

    /// <inheritdoc/>
    public double Skewness => 0;

    /// <inheritdoc/>
    public double ExcessKurtosis => -1.2;

    /// <inheritdoc/>
    public double Entropy => Math.Log(_width);

    /// <inheritdoc/>
    public double Median => Mean;

    /// <inheritdoc/>
    public double SupportMinimum => _lower;

    /// <inheritdoc/>
    public double SupportMaximum => _upper;

    /// <inheritdoc/>
    public double Pdf(double x) => double.IsNaN(x) ? double.NaN : x >= _lower && x <= _upper ? 1 / _width : 0;

    /// <inheritdoc/>
    public double LogPdf(double x) => double.IsNaN(x) ? double.NaN : x >= _lower && x <= _upper ? -Math.Log(_width) : double.NegativeInfinity;

    /// <inheritdoc/>
    public double Cdf(double x) => x <= _lower ? 0 : x >= _upper ? 1 : double.IsNaN(x) ? double.NaN : (x - _lower) / _width;

    /// <inheritdoc/>
    public double Sf(double x) => x <= _lower ? 1 : x >= _upper ? 0 : double.IsNaN(x) ? double.NaN : (_upper - x) / _width;

    /// <inheritdoc/>
    public double LogCdf(double x)
    {
        double p = Cdf(x);
        if (p < NumericConstants.MinNormal && x > _lower)
        {
            // Take logs before dividing when the ratio underflows or becomes subnormal on very wide intervals.
            return Math.Log(x - _lower) - Math.Log(_width);
        }

        // Use Log1p(-Sf(x)) above 0.5 to avoid rounding p near 1.
        return p <= 0.5 ? Math.Log(p) : DistributionMath.Log1p(-Sf(x));
    }

    /// <inheritdoc/>
    public double LogSf(double x)
    {
        double q = Sf(x);
        if (q < NumericConstants.MinNormal && x < _upper)
        {
            return Math.Log(_upper - x) - Math.Log(_width);
        }

        return q <= 0.5 ? Math.Log(q) : DistributionMath.Log1p(-Cdf(x));
    }

    /// <inheritdoc/>
    public double InverseCdf(double p)
    {
        if (!(p >= 0 && p <= 1))
        {
            return double.NaN;
        }

        return p <= 0.5 ? _lower + p * _width : _upper - (1 - p) * _width;
    }

    /// <inheritdoc/>
    public double InverseSf(double q)
    {
        if (!(q >= 0 && q <= 1))
        {
            return double.NaN;
        }

        return q <= 0.5 ? _upper - q * _width : _lower + (1 - q) * _width;
    }

    /// <inheritdoc/>
    public double InverseLogCdf(double logP)
    {
        if (!(logP <= 0))
        {
            return double.NaN;
        }

        return logP > -XMath.Constants.Ln2 ? InverseSf(-DistributionMath.Expm1(logP)) : _lower + Offset(logP);
    }

    /// <inheritdoc/>
    public double InverseLogSf(double logQ)
    {
        if (!(logQ <= 0))
        {
            return double.NaN;
        }

        return logQ > -XMath.Constants.Ln2 ? InverseCdf(-DistributionMath.Expm1(logQ)) : _upper - Offset(logQ);
    }

    private double Offset(double logP)
    {
        double p = Math.Exp(logP);
        return p >= NumericConstants.MinNormal ? p * _width : Math.Exp(logP + Math.Log(_width));
    }

    /// <inheritdoc/>
    public double Sample(IRandomSource random)
    {
        ArgumentNullException.ThrowIfNull(random);
        return _lower + random.NextDouble() * _width;
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, Span<double> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = _lower + random.NextDouble() * _width;
        }
    }
}
