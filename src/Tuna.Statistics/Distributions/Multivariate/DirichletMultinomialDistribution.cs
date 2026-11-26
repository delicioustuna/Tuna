using Tuna.LinearAlgebra;
using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the Dirichlet-multinomial compound distribution over non-negative integer vectors summing to <c>n</c>.
/// <para>確率ベクトルがディリクレ分布に従う複合分布（ディリクレ多項分布）を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// The support consists of non-negative integer vectors summing to <c>n</c>.
/// For 2 categories, the first component follows a beta-binomial distribution; as <c>α₀ → ∞</c> with fixed <c>αᵢ/α₀</c>, the distribution approaches <see cref="MultinomialDistribution"/>.
/// </para>
/// <para>
/// 台は非負整数成分からなり総和が <c>n</c> のベクトルです。
/// カテゴリ数が 2 のとき第 1 成分はベータ二項分布に一致し、<c>αᵢ/α₀</c> を固定して <c>α₀ → ∞</c> とすると <see cref="MultinomialDistribution"/> に収束します。
/// </para>
/// </remarks>
public sealed class DirichletMultinomialDistribution : IMultivariateDiscreteDistribution
{
    /// <summary>
    /// The maximum supported number of trials (<c>2⁵³</c>, matching <see cref="BinomialDistribution.MaxTrials"/>).
    /// <para>サポートされる試行回数の上限（<c>2⁵³</c>、<see cref="BinomialDistribution.MaxTrials"/> と同一）。</para>
    /// </summary>
    public const long MaxTrials = BinomialDistribution.MaxTrials;

    private readonly long _trials;
    private readonly double[] _alpha;
    private readonly double[] _others;
    private readonly double _concentration;

    /// <summary>
    /// Initializes a new instance of <see cref="DirichletMultinomialDistribution"/> with the specified trial count and concentration parameters.
    /// <para>試行回数と集中度パラメーターを指定して <see cref="DirichletMultinomialDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="trials">The number of trials <c>n</c> (<c>0 ≤ n ≤ 2⁵³</c>). / 試行回数 <c>n</c>（<c>0 ≤ n ≤ 2⁵³</c>）。</param>
    /// <param name="alpha">The positive finite concentration parameters <c>αᵢ</c> (at least 1 element, with a finite sum). / 各成分の集中度パラメーター <c>αᵢ</c>（1 個以上の正の有限値で、総和も有限）。</param>
    /// <exception cref="ArgumentException"><paramref name="alpha"/> is empty. / <paramref name="alpha"/> が空の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="trials"/> is outside <c>[0, MaxTrials]</c>, any element of <paramref name="alpha"/> is not positive and finite, or the sum of concentrations overflows. / <paramref name="trials"/> が <c>[0, MaxTrials]</c> の範囲外、正の有限値でない集中度がある場合、または総和が有限でない場合。</exception>
    public DirichletMultinomialDistribution(long trials, params ReadOnlySpan<double> alpha)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(trials);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(trials, MaxTrials);
        if (alpha.IsEmpty)
        {
            throw new ArgumentException("At least one category is required.", nameof(alpha));
        }

        foreach (double value in alpha)
        {
            DistributionArgument.PositiveFinite(value, nameof(alpha));
        }

        _trials = trials;
        _alpha = alpha.ToArray();
        double[] suffix;
        (suffix, _others) = MultivariateMath.SuffixAndOthers(_alpha);
        _concentration = suffix[0];
        if (!double.IsFinite(_concentration))
        {
            throw new ArgumentOutOfRangeException(nameof(alpha), _concentration, "The sum of the concentrations must be finite.");
        }
    }

    /// <inheritdoc/>
    public int Dimension => _alpha.Length;

    /// <summary>
    /// Gets the number of trials <c>n</c>.
    /// <para>試行回数 <c>n</c> を取得します。</para>
    /// </summary>
    public long Trials => _trials;

    /// <summary>
    /// Gets a read-only span of the concentration parameters <c>α</c>.
    /// <para>集中度パラメーター <c>α</c> の読み取り専用スパンを取得します。</para>
    /// </summary>
    public ReadOnlySpan<double> Alpha => _alpha;

    /// <summary>
    /// Gets the sum of the concentration parameters <c>α₀ = Σᵢ αᵢ</c>.
    /// <para>集中度パラメーターの総和 <c>α₀ = Σᵢ αᵢ</c> を取得します。</para>
    /// </summary>
    public double Concentration => _concentration;

    /// <inheritdoc/>
    public Vector<double> GetMean()
    {
        var mean = Vector<double>.Uninitialized(Dimension);
        for (int i = 0; i < Dimension; i++)
        {
            mean[i] = DistributionMath.ScaledRatio([_trials, _alpha[i]], [_concentration]);
        }

        return mean;
    }

    /// <inheritdoc/>
    public Matrix<double> GetCovariance()
    {
        int k = Dimension;
        double n = _trials;
        double a0 = _concentration;
        var covariance = Matrix<double>.Uninitialized(k, k);
        for (int j = 0; j < k; j++)
        {
            for (int i = 0; i < k; i++)
            {
                covariance[i, j] = i == j
                    ? DistributionMath.ScaledRatio([n, _alpha[j], _others[j], n + a0], [a0, a0, 1 + a0])
                    : -DistributionMath.ScaledRatio([n, _alpha[i], _alpha[j], n + a0], [a0, a0, 1 + a0]);
            }
        }

        return covariance;
    }

    /// <inheritdoc/>
    public double Pmf(ReadOnlyVectorSpan<long> x) => Math.Exp(LogPmf(x));

    /// <inheritdoc/>
    public double LogPmf(ReadOnlyVectorSpan<long> x)
    {
        RequireLength(x.Length, nameof(x));
        return LogPmfCore(x);
    }

    /// <inheritdoc/>
    public void LogPmf(ReadOnlyMatrixSpan<long> x, VectorSpan<double> destination)
    {
        RequireLength(x.Rows, nameof(x));
        if (destination.Length != x.Columns)
        {
            throw new ArgumentException("The length must equal the number of points.", nameof(destination));
        }

        MultivariateMath.RequireDisjoint(x, destination);
        for (int j = 0; j < x.Columns; j++)
        {
            destination[j] = LogPmfCore(x.Column(j));
        }
    }

    private double LogPmfCore(ReadOnlyVectorSpan<long> x)
    {
        if (!MultivariateMath.IsComposition(x, _trials))
        {
            return double.NegativeInfinity;
        }

        if (_trials == 0)
        {
            return 0;
        }

        // Decompose log-PMF into Stirling remainders + shifted concentration terms + Loader deviance terms bd0.
        // For α₀ < 1, separate -ln α₀ + Σ ln α_i (pairing the first positive component as ln(α_i/α₀)) to avoid cancellation of large -ln α terms.
        double n = _trials;
        double a0 = _concentration;
        double total = a0 + n;
        double t = n / total;
        double u = a0 / total;
        bool small = a0 < 1;
        double divisor = small ? a0 : 1;
        double counts = DistributionMath.StirlingRemainder(n);
        double logs = 0.5 * (DistributionMath.LnTwoPi + Math.Log(n));
        double concentrations = (small ? DistributionMath.StirlingRemainderShiftedWithoutLog(a0, n) : DistributionMath.StirlingRemainderShifted(a0, n))
            - DistributionMath.StirlingRemainder(total);
        // For x_i = 0, bd0(0, T*α_i) + bd0(α_i, U*α_i) = α_i * ln(1 + n/α₀).
        double ratio = n / a0;
        double empty = double.IsFinite(ratio) ? DistributionMath.Log1p(ratio) : Math.Log(n) - Math.Log(a0);
        double deviance = 0;
        for (int i = 0; i < x.Length; i++)
        {
            double a = _alpha[i];
            if (x[i] == 0)
            {
                deviance += a * empty;
                continue;
            }

            double count = x[i];
            double w = a + count;
            double share = w / total;
            counts -= DistributionMath.StirlingRemainder(count);
            logs -= 0.5 * (DistributionMath.LnTwoPi + Math.Log(count));
            if (small)
            {
                concentrations += DistributionMath.StirlingRemainder(w) - DistributionMath.StirlingRemainderShiftedWithoutLog(a, count) + Math.Log(a / divisor);
                divisor = 1;
            }
            else
            {
                concentrations += DistributionMath.StirlingRemainder(w) - DistributionMath.StirlingRemainderShifted(a, count);
            }

            deviance += DistributionMath.Bd0(count, n * share) + ConcentrationDeviance(a, count, a0, share, t, u);
        }

        return counts + logs + concentrations - deviance;
    }

    // Evaluates bd0(α_i, U*w_i) without quadratic rounding error when U*w_i ≈ α_i for huge α_i by writing
    // d_i = U*w_i - α_i = x_i*U - α_i*T and calling -α_i * log1pmx(d_i / α_i).
    private static double ConcentrationDeviance(double a, double count, double a0, double share, double t, double u)
    {
        double m = a0 * share;
        if (m < DistributionMath.MinNormal)
        {
            return a * (Math.Log(a / a0) - Math.Log(share)) + (m - a);
        }

        if (u >= DistributionMath.MinNormal && m >= 0.5 * a && m <= 2 * a)
        {
            return -a * DistributionMath.Log1pmx((count * u - a * t) / a);
        }

        return DistributionMath.Bd0(a, m);
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, VectorSpan<long> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        RequireLength(destination.Length, nameof(destination));
        int k = Dimension;
        Span<double> weights = k <= MultivariateMath.StackLimit ? stackalloc double[k] : new double[k];
        Span<double> logUniform = k <= MultivariateMath.StackLimit ? stackalloc double[k] : new double[k];
        Span<double> suffix = k < MultivariateMath.StackLimit ? stackalloc double[k + 1] : new double[k + 1];
        SampleCore(random, weights, logUniform, suffix, destination);
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, MatrixSpan<long> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        RequireLength(destination.Rows, nameof(destination));
        int k = Dimension;
        Span<double> weights = k <= MultivariateMath.StackLimit ? stackalloc double[k] : new double[k];
        Span<double> logUniform = k <= MultivariateMath.StackLimit ? stackalloc double[k] : new double[k];
        Span<double> suffix = k < MultivariateMath.StackLimit ? stackalloc double[k + 1] : new double[k + 1];
        for (int j = 0; j < destination.Columns; j++)
        {
            SampleCore(random, weights, logUniform, suffix, destination.Column(j));
        }
    }

    private void SampleCore(IRandomSource random, Span<double> weights, Span<double> logUniform, Span<double> suffix, VectorSpan<long> destination)
    {
        if (_trials == 0)
        {
            destination.Fill(0);
            return;
        }

        Sampling.DirichletWeights(random, _alpha, weights, logUniform);
        suffix[^1] = 0;
        for (int i = weights.Length - 1; i >= 0; i--)
        {
            suffix[i] = suffix[i + 1] + weights[i];
        }

        Sampling.Multinomial(random, _trials, weights, suffix, destination);
    }

    private void RequireLength(int length, string name)
    {
        if (length != Dimension)
        {
            throw new ArgumentException("The length must equal the dimension.", name);
        }
    }
}
