using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the Weibull distribution with shape <c>k</c> and scale <c>λ</c> on <c>[0, ∞)</c>.
/// <para>形状 <c>k</c>、尺度 <c>λ</c> を持つ <c>[0, ∞)</c> 上のワイブル分布を表します。</para>
/// </summary>
public readonly partial struct WeibullDistribution : IContinuousDistribution
{
    private readonly double _shape;
    private readonly double _scale;

    /// <summary>
    /// Initializes a new instance of <see cref="WeibullDistribution"/> with the specified shape and scale.
    /// <para>形状および尺度パラメーターを指定して <see cref="WeibullDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="shape">The shape parameter <c>k</c> (positive and finite). / 形状パラメーター <c>k</c>（正かつ有限）。</param>
    /// <param name="scale">The scale parameter <c>λ</c> (positive and finite). / 尺度パラメーター <c>λ</c>（正かつ有限）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="shape"/> or <paramref name="scale"/> is not positive and finite. / <paramref name="shape"/> または <paramref name="scale"/> が正の有限値でない場合。</exception>
    public WeibullDistribution(double shape, double scale)
        : this(shape, scale, forDifferentiation: false)
    {
    }

    private WeibullDistribution(double shape, double scale, bool forDifferentiation)
    {
        DistributionArgument.PositiveFinite(shape, nameof(shape), forDifferentiation);
        DistributionArgument.PositiveFinite(scale, nameof(scale), forDifferentiation);
        _shape = shape;
        _scale = scale;
    }

    /// <summary>
    /// Gets the shape parameter <c>k</c>.
    /// <para>形状パラメーター <c>k</c> を取得します。</para>
    /// </summary>
    public double Shape => _shape;

    /// <summary>
    /// Gets the scale parameter <c>λ</c>.
    /// <para>尺度パラメーター <c>λ</c> を取得します。</para>
    /// </summary>
    public double Scale => _scale;

    // Minimum shape k for evaluating central moments via power series in ε = 1/k.
    private const double SeriesShape = 8;

    /// <inheritdoc/>
    public double Mean => _scale * XMath.Special.Gamma(1 + 1 / _shape);

    /// <inheritdoc/>
    public double Variance => Math.Exp(LogVariance());

    /// <inheritdoc/>
    public double StandardDeviation => Math.Exp(0.5 * LogVariance());

    private double LogVariance()
    {
        double epsilon = 1 / _shape;
        double logGamma1 = XMath.Special.LogGamma(1 + epsilon);
        if (_shape >= SeriesShape)
        {
            return 2 * (Math.Log(_scale) + logGamma1 + Math.Log(epsilon)) + Math.Log(Series(Mu2Series, epsilon));
        }

        return 2 * (Math.Log(_scale) + logGamma1) + LogExpm1(LogMomentRatio(2, epsilon));
    }

    /// <inheritdoc/>
    public double Skewness
    {
        get
        {
            double epsilon = 1 / _shape;
            if (_shape >= SeriesShape)
            {
                double s2 = Series(Mu2Series, epsilon);
                return Series(Mu3Series, epsilon) / (s2 * Math.Sqrt(s2));
            }

            double a2 = LogMomentRatio(2, epsilon);
            double a3 = LogMomentRatio(3, epsilon);
            if (a3 < 700)
            {
                double mu2 = DistributionMath.Expm1(a2);
                return (DistributionMath.Expm1(a3) - 3 * mu2) / (mu2 * Math.Sqrt(mu2));
            }

            return Math.Exp(a3 - 1.5 * LogExpm1(a2)) * (1 - 3 * Math.Exp(a2 - a3) + 2 * Math.Exp(-a3));
        }
    }

    /// <inheritdoc/>
    public double ExcessKurtosis
    {
        get
        {
            double epsilon = 1 / _shape;
            if (_shape >= SeriesShape)
            {
                double s2 = Series(Mu2Series, epsilon);
                return Series(Kappa4Series, epsilon) / (s2 * s2);
            }

            double a2 = LogMomentRatio(2, epsilon);
            double a3 = LogMomentRatio(3, epsilon);
            double a4 = LogMomentRatio(4, epsilon);
            if (a4 < 700)
            {
                double mu2 = DistributionMath.Expm1(a2);
                double mu4 = DistributionMath.Expm1(a4) - 4 * DistributionMath.Expm1(a3) + 6 * mu2;
                return mu4 / (mu2 * mu2) - 3;
            }

            return Math.Exp(a4 - 2 * LogExpm1(a2))
                * (1 - 4 * Math.Exp(a3 - a4) + 6 * Math.Exp(a2 - a4) - 3 * Math.Exp(-a4)) - 3;
        }
    }

    // Computes A_m = ln Γ(1 + mε) - m ln Γ(1 + ε). For mε <= 1, substitutes Abramowitz & Stegun 6.1.41
    // so the canceling linear term x(1 - γ) drops out analytically, reducing rounding error.
    private static double LogMomentRatio(int m, double epsilon)
    {
        if (m * epsilon > 1)
        {
            return XMath.Special.LogGamma(1 + m * epsilon) - m * XMath.Special.LogGamma(1 + epsilon);
        }

        double sum = 0;
        double mPower = m;
        double epsilonPower = epsilon;
        for (int j = 2; j < ZetaMinusOne.Length; j++)
        {
            mPower *= m;
            epsilonPower *= epsilon;
            double term = ZetaMinusOne[j] * (mPower - m) * epsilonPower / j;
            sum += (j & 1) == 0 ? term : -term;
            if (term <= 1e-17 * Math.Abs(sum))
            {
                break;
            }
        }

        return m * DistributionMath.Log1p(epsilon) - DistributionMath.Log1p(m * epsilon) + sum;
    }

    // Precomputed ζ(j) - 1 for j >= 2 (indices 0 and 1 unused).
    private static readonly double[] ZetaMinusOne = CreateZetaMinusOne(72);

    private static double[] CreateZetaMinusOne(int count)
    {
        var values = new double[count];
        for (int j = 2; j < count; j++)
        {
            values[j] = XMath.Special.ZetaMinusOne(j);
        }

        return values;
    }

    private static double LogExpm1(double a) => a > 30 ? a + DistributionMath.Log1p(-Math.Exp(-a)) : Math.Log(DistributionMath.Expm1(a));

    private static double Series(double[] coefficients, double epsilon)
    {
        double sum = 0;
        for (int n = coefficients.Length - 1; n >= 0; n--)
        {
            sum = coefficients[n] + epsilon * sum;
        }

        return sum;
    }

    /// <inheritdoc/>
    public double Entropy => XMath.Special.EulerGamma * (1 - 1 / _shape) + Math.Log(_scale) - Math.Log(_shape) + 1;

    /// <inheritdoc/>
    public double Median => Quantile(XMath.Constants.Ln2);

    /// <inheritdoc/>
    public double SupportMinimum => 0;

    /// <inheritdoc/>
    public double SupportMaximum => double.PositiveInfinity;

    private double Power(double x)
    {
        double z = x / _scale;
        if (z >= NumericConstants.MinNormal && z < double.PositiveInfinity)
        {
            return Math.Pow(z, _shape);
        }

        return Math.Exp(_shape * (Math.Log(x) - Math.Log(_scale)));
    }

    private double LogPower(double x)
    {
        double z = x / _scale;
        return z >= NumericConstants.MinNormal && z < double.PositiveInfinity
            ? _shape * Math.Log(z)
            : _shape * (Math.Log(x) - Math.Log(_scale));
    }

    private double QuantileFromLog(double logW)
    {
        double value = _scale * Math.Exp(logW / _shape);
        return value >= NumericConstants.MinNormal && value < double.PositiveInfinity
            ? value
            : Math.Exp(Math.Log(_scale) + logW / _shape);
    }

    private double Quantile(double w)
    {
        double value = _scale * Math.Pow(w, 1 / _shape);
        if (value >= NumericConstants.MinNormal && value < double.PositiveInfinity || w == 0 || double.IsPositiveInfinity(w))
        {
            return value;
        }

        return Math.Exp(Math.Log(_scale) + Math.Log(w) / _shape);
    }

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

        double z = x / _scale;
        double value = _shape / _scale * Math.Pow(z, _shape - 1) * Math.Exp(-Math.Pow(z, _shape));
        return double.IsFinite(value) && value >= NumericConstants.MinNormal ? value : Math.Exp(LogPdf(x));
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

        double logZ = Math.Log(x) - Math.Log(_scale);
        return Math.Log(_shape) - Math.Log(_scale) + (_shape - 1) * logZ - Power(x);
    }

    /// <inheritdoc/>
    public double Cdf(double x) => x <= 0 ? 0 : -DistributionMath.Expm1(-Power(x));

    /// <inheritdoc/>
    public double Sf(double x) => x <= 0 ? 1 : Math.Exp(-Power(x));

    /// <inheritdoc/>
    public double LogCdf(double x)
    {
        if (x <= 0)
        {
            return double.NegativeInfinity;
        }

        double w = Power(x);
        // When w = (x/λ)^k is subnormal, ln(1 - e^-w) = ln w - w/2 + ... agrees with ln w.
        return w < NumericConstants.MinNormal ? LogPower(x) : DistributionMath.Log1mExp(-w);
    }

    /// <inheritdoc/>
    public double LogSf(double x) => x <= 0 ? 0 : -Power(x);

    /// <inheritdoc/>
    public double InverseCdf(double p) => p >= 0 && p <= 1 ? Quantile(-DistributionMath.Log1p(-p)) : double.NaN;

    /// <inheritdoc/>
    public double InverseSf(double q) => q >= 0 && q <= 1 ? Quantile(-Math.Log(q)) : double.NaN;

    /// <inheritdoc/>
    public double InverseLogCdf(double logP)
    {
        if (!(logP <= 0))
        {
            return double.NaN;
        }

        // In deep lower tails, w = -ln(1 - p) ≈ p, so ln w = logP within relative error p/2.
        return logP < DistributionMath.LogProbabilityLinearLimit ? QuantileFromLog(logP) : Quantile(-DistributionMath.Log1mExp(logP));
    }

    /// <inheritdoc/>
    public double InverseLogSf(double logQ) => logQ <= 0 ? Quantile(-logQ) : double.NaN;

    /// <inheritdoc/>
    public double Sample(IRandomSource random)
    {
        ArgumentNullException.ThrowIfNull(random);
        return Quantile(Ziggurat.StandardExponential(random));
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, Span<double> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = Quantile(Ziggurat.StandardExponential(random));
        }
    }

    // Power-series coefficients in ε = 1/k for central moments μ₂ = ε² S₂(ε), μ₃ = ε³ S₃(ε),
    // and fourth cumulant κ₄ = μ₄ - 3μ₂² = ε⁴ S₄(ε) of Y = E^ε / Γ(1 + ε), precomputed in high precision.
    private static readonly double[] Mu2Series =
    [
        1.6449340668482264, -2.4041138063191886, 5.1410353601279064, -1.0176175231454812e+1, 2.0375465474724988e+1,
        -4.0743987816707965e+1, 8.1487923935610386e+1, -1.629749962158658e+2, 3.2594955505022998e+2,
        -6.5189880012601991e+2, 1.3037973960750809e+3, -2.6075946557735663e+3, 5.2151892206713645e+3,
        -1.0430378380759998e+4, 2.086075672113352e+4, -4.1721513415343382e+4, 8.3443026812737934e+4,
        -1.6688605361351009e+5, 3.3377210721904304e+5, -6.6754421443276801e+5, 1.3350884288619906e+6,
        -2.6701768577216177e+6, 5.3403537154416597e+6, -1.0680707430882269e+7, 2.1361414861763837e+7,
        -4.2722829723527208e+7, 8.5445659447054104e+7, -1.70891318894108e+8, 3.4178263778821586e+8,
        -6.8356527557643164e+8, 1.3671305511528632e+9, -2.7342611023057264e+9,
    ];
    private static readonly double[] Mu3Series =
    [
        -2.4041138063191886, 1.785833335623378e+1, -6.6699310959652428e+1, 2.323896571292965e+2,
        -7.5804008492662132e+2, 2.3973143473647702e+3, -7.4373010743405658e+3, 2.2802364571492472e+4,
        -6.938718212164848e+4, 2.1012062047525699e+5, -6.3427831461377458e+5, 1.9106653202771667e+6,
        -5.7476529166960938e+6, 1.7274266967466218e+7, -5.1885408795815224e+7, 1.5578142936251712e+8,
        -4.6759467482019022e+8, 1.4032847690996882e+9, -4.2108557533381217e+9, 1.2634570087233753e+10,
        -3.7907715818851975e+10, 1.1373115842492528e+11, -3.4120949699261644e+11, 1.0236605340851834e+12,
        -3.0710456879776976e+12, 9.2132652346386065e+12, -2.7640052044218675e+13, 8.2920668811599481e+13,
        -2.4876303179019197e+14, 7.4629114607762286e+14, -2.2388775396413524e+15, 6.7166408217326097e+15,
        -2.0149938870802311e+16, 6.0449849423596964e+16, -1.8134961389314255e+17, 5.4404897292408837e+17,
        -1.6321471812615226e+18, 4.8964420687629871e+18, -1.4689327256245656e+19, 4.4067983868650141e+19,
        -1.3220395580577645e+20, 3.966118758169809e+20, -1.1898356442502451e+21, 3.569506966349339e+21,
    ];
    private static readonly double[] Kappa4Series =
    [
        6.4939394022668291, -9.7227836654016863e+1, 6.9612939956217106e+2, -3.8062605155762083e+3,
        1.8546076265985141e+4, -8.4489645880466041e+4, 3.6946474421288269e+5, -1.5732016645227046e+6,
        6.579874458391235e+6, -2.7181513968749655e+7, 1.1131122731454638e+8, -4.5299366222581262e+8,
        1.8351973874475863e+9, -7.4103907467941345e+9, 2.9850192614274768e+10, -1.2002623159991697e+11,
        4.8198029479417813e+11, -1.933544935274701e+12, 7.7510456733308494e+12, -3.1054768542330666e+13,
        1.2437080527016158e+14, -4.979383563896742e+14, 1.9931187052972874e+15, -7.9765703876259999e+15,
        3.1918567663454482e+16, -1.2771112773834725e+17, 5.1095507953513246e+17, -2.0441520182156761e+18,
        8.1776031610951934e+18, -3.271339788378679e+19, 1.3086254720017123e+20, -5.2347705568414699e+20,
        2.0939888231533098e+21, -8.3761970933751873e+21, 3.3505513774768382e+22, -1.3402423130075645e+23,
        5.3610345380366752e+23, -2.1444334010074469e+24, 8.5777923613891529e+24, -3.4311345717595441e+25,
        1.3724591168641704e+26, -5.4898523319360624e+26, 2.195945692117893e+27, -8.7837970465012621e+27,
        3.5135231020092648e+28, -1.4054105258263037e+29, 5.6216459583729457e+29, -2.2486595398694848e+30,
        8.9946416290388323e+30, -3.5978576924837955e+31, 1.4391433892539959e+32, -5.7565744937974143e+32,
        2.3026300785533945e+33, -9.2105211573168632e+33, 3.6842087158577307e+34, -1.4736835622223879e+35,
        5.8947344765274381e+35, -2.3578938589023412e+36, 9.4315756404834627e+36, -3.7726303176556145e+37,
    ];
}
