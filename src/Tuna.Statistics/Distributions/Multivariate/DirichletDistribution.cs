using Tuna.LinearAlgebra;
using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the Dirichlet distribution <c>Dir(α)</c> over the probability simplex.
/// <para>確率単体上のディリクレ分布 <c>Dir(α)</c> を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Evaluation points accept all <c>K</c> components; density is with respect to Lebesgue measure on the first <c>K - 1</c> coordinates (<c>√K</c> times the surface-measure density on the simplex).
/// Points must satisfy <c>xᵢ ≥ 0</c> and <c>|Σᵢ xᵢ - 1| ≤ 16Kε</c>; points outside the tolerance are treated as outside the support (density 0), while points within tolerance are evaluated at <c>x / Σᵢ xᵢ</c>.
/// On the boundary (<c>xᵢ = 0</c>), <c>xᵢ^(αᵢ - 1)</c> evaluates to 1 for <c>αᵢ = 1</c>, <see cref="double.PositiveInfinity"/> for <c>αᵢ &lt; 1</c>, and 0 for <c>αᵢ &gt; 1</c>; conflicting <c>+∞</c> and 0 factors return <see cref="double.NaN"/>.
/// </para>
/// <para>
/// 評価点は <c>K</c> 成分すべてを受け取り、密度は先頭 <c>K - 1</c> 座標のルベーグ測度に関して返します（単体上の面積測度に対する密度の <c>√K</c> 倍）。
/// 台は <c>xᵢ ≥ 0</c> かつ <c>|Σᵢ xᵢ - 1| ≤ 16Kε</c> とみなし、許容幅外の点は正規化せず台の外（密度 0）、許容幅内の点は <c>x / Σᵢ xᵢ</c> で評価します。
/// 境界（<c>xᵢ = 0</c>）では <c>xᵢ^(αᵢ - 1)</c> の極限に従い <c>αᵢ = 1</c> で 1、<c>αᵢ &lt; 1</c> で <see cref="double.PositiveInfinity"/>、<c>αᵢ &gt; 1</c> で 0 とし、<c>+∞</c> と 0 が同時に現れる場合は <see cref="double.NaN"/> を返します。
/// </para>
/// </remarks>
public sealed partial class DirichletDistribution : IMultivariateContinuousDistribution
{
    private readonly double[] _alpha;
    private readonly double[] _suffix;
    private readonly double _concentration;

    /// <summary>
    /// Initializes a new instance of <see cref="DirichletDistribution"/> with the specified concentration parameters.
    /// <para>集中度パラメーターを指定して <see cref="DirichletDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="alpha">The positive finite concentration parameters <c>αᵢ</c> (at least 2 elements, with a finite sum). / 各成分の集中度パラメーター <c>αᵢ</c>（2 個以上の正の有限値で、総和も有限）。</param>
    /// <exception cref="ArgumentException"><paramref name="alpha"/> has fewer than 2 elements. / <paramref name="alpha"/> の要素数が 2 未満の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">Any element is not positive and finite, or the sum of concentrations overflows. / 正の有限値でない要素がある場合、または総和が有限でない場合。</exception>
    public DirichletDistribution(params ReadOnlySpan<double> alpha)
    {
        if (alpha.Length < 2)
        {
            throw new ArgumentException("At least two components are required.", nameof(alpha));
        }

        foreach (double value in alpha)
        {
            DistributionArgument.PositiveFinite(value, nameof(alpha));
        }

        _alpha = alpha.ToArray();

        // _suffix[k] = α_k + ... + α_{K-1}, _suffix[K] = 0.
        _suffix = new double[_alpha.Length + 1];
        for (int k = _alpha.Length - 1; k >= 0; k--)
        {
            _suffix[k] = _suffix[k + 1] + _alpha[k];
        }

        _concentration = _suffix[0];
        if (!double.IsFinite(_concentration))
        {
            throw new ArgumentOutOfRangeException(nameof(alpha), _concentration, "The sum of the concentrations must be finite.");
        }
    }

    /// <inheritdoc/>
    public int Dimension => _alpha.Length;

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
            mean[i] = _alpha[i] / _concentration;
        }

        return mean;
    }

    /// <inheritdoc/>
    public Matrix<double> GetCovariance()
    {
        int k = Dimension;
        double a0 = _concentration;
        var covariance = Matrix<double>.Uninitialized(k, k);
        double prefix = 0;
        for (int j = 0; j < k; j++)
        {
            double pj = _alpha[j] / a0;
            for (int i = 0; i < k; i++)
            {
                // Compute (α₀ - α_j) via prefix + suffix[j + 1] to avoid cancellation when α_j ≈ α₀.
                covariance[i, j] = i == j
                    ? pj * ((prefix + _suffix[j + 1]) / a0) / (a0 + 1)
                    : -(_alpha[i] / a0) * pj / (a0 + 1);
            }

            prefix += _alpha[j];
        }

        return covariance;
    }

    /// <inheritdoc/>
    public double Pdf(ReadOnlyVectorSpan<double> x) => Math.Exp(LogPdf(x));

    /// <inheritdoc/>
    public double LogPdf(ReadOnlyVectorSpan<double> x)
    {
        RequireLength(x.Length, nameof(x));
        int k = Dimension;
        Span<double> remaining = k <= MultivariateMath.StackLimit ? stackalloc double[k] : new double[k];
        return LogPdfCore(x, remaining);
    }

    /// <inheritdoc/>
    public void LogPdf(ReadOnlyMatrixSpan<double> x, VectorSpan<double> destination)
    {
        RequireLength(x.Rows, nameof(x));
        if (destination.Length != x.Columns)
        {
            throw new ArgumentException("The length must equal the number of points.", nameof(destination));
        }

        MultivariateMath.RequireDisjoint(x, destination);
        int k = Dimension;
        Span<double> remaining = k <= MultivariateMath.StackLimit ? stackalloc double[k] : new double[k];
        for (int j = 0; j < x.Columns; j++)
        {
            destination[j] = LogPdfCore(x.Column(j), remaining);
        }
    }

    private double LogPdfCore(ReadOnlyVectorSpan<double> x, Span<double> remaining) =>
        LogPdfCore(x, remaining, _alpha, _suffix, out _, out _);

    private static double LogPdfCore(ReadOnlyVectorSpan<double> x, Span<double> remaining,
        ReadOnlySpan<double> alpha, ReadOnlySpan<double> suffix, out double sum, out bool onSupport)
    {
        int k = alpha.Length;
        sum = 0;
        onSupport = false;
        for (int i = 0; i < k; i++)
        {
            if (double.IsNaN(x[i]))
            {
                return double.NaN;
            }
        }

        bool boundary = false;
        for (int i = k - 1; i >= 0; i--)
        {
            double value = x[i];
            if (!(value >= 0 && value < double.PositiveInfinity))
            {
                return double.NegativeInfinity;
            }

            boundary |= value == 0;
            sum += value;
            remaining[i] = sum;
        }

        if (!(Math.Abs(sum - 1) <= 16 * k * DistributionMath.Epsilon))
        {
            return double.NegativeInfinity;
        }

        onSupport = true;
        if (boundary)
        {
            return BoundaryLogPdf(x, sum, alpha);
        }

        // Factor Dirichlet log-density as stick-breaking Beta prefixes to avoid Stirling cancellation for large α.
        double result = k * Math.Log(sum);
        for (int i = 0; i < k - 1; i++)
        {
            double a = alpha[i];
            double b = suffix[i + 1];
            double value = x[i];
            double rest = remaining[i + 1];
            double v = value <= rest ? value / remaining[i] : rest / remaining[i];
            result += value <= rest
                ? XMath.Special.LogBetaPrefix(a, b, v, 1 - v)
                : XMath.Special.LogBetaPrefix(b, a, v, 1 - v);
        }

        for (int i = 0; i < k; i++)
        {
            result -= Math.Log(x[i]);
        }

        return result;
    }

    private static double BoundaryLogPdf(ReadOnlyVectorSpan<double> x, double sum, ReadOnlySpan<double> alpha)
    {
        double logSum = Math.Log(sum);
        double result = 0;
        bool positiveInfinity = false;
        bool negativeInfinity = false;
        for (int i = 0; i < alpha.Length; i++)
        {
            double exponent = alpha[i] - 1;
            if (x[i] == 0)
            {
                positiveInfinity |= exponent < 0;
                negativeInfinity |= exponent > 0;
                continue;
            }

            result += exponent * (Math.Log(x[i]) - logSum);
        }

        if (positiveInfinity)
        {
            return negativeInfinity ? double.NaN : double.PositiveInfinity;
        }

        return negativeInfinity ? double.NegativeInfinity : result - XMath.Special.LogBeta(alpha);
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, VectorSpan<double> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        RequireLength(destination.Length, nameof(destination));
        int k = Dimension;
        Span<double> weights = k <= MultivariateMath.StackLimit ? stackalloc double[k] : new double[k];
        Span<double> logUniform = k <= MultivariateMath.StackLimit ? stackalloc double[k] : new double[k];
        SampleCore(random, weights, logUniform, destination);
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, MatrixSpan<double> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        RequireLength(destination.Rows, nameof(destination));
        int k = Dimension;
        Span<double> weights = k <= MultivariateMath.StackLimit ? stackalloc double[k] : new double[k];
        Span<double> logUniform = k <= MultivariateMath.StackLimit ? stackalloc double[k] : new double[k];
        for (int j = 0; j < destination.Columns; j++)
        {
            SampleCore(random, weights, logUniform, destination.Column(j));
        }
    }

    private void SampleCore(IRandomSource random, Span<double> weights, Span<double> logUniform, VectorSpan<double> destination)
    {
        double sum = Sampling.DirichletWeights(random, _alpha, weights, logUniform);
        for (int i = 0; i < weights.Length; i++)
        {
            destination[i] = weights[i] / sum;
        }
    }

    private void RequireLength(int length, string name)
    {
        if (length != Dimension)
        {
            throw new ArgumentException("The length must equal the dimension.", name);
        }
    }
}
