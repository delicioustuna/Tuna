using Tuna.LinearAlgebra;
using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the negative multinomial distribution for category failure counts before <c>r</c> successes.
/// <para><c>r</c> 回の成功に至るまでの種類別失敗回数を表す負の多項分布です。</para>
/// </summary>
/// <remarks>
/// <para>
/// The success parameter <c>r</c> may be any positive real number up to <see cref="MaxSuccesses"/> (modeling correlated overdispersed counts as a gamma–Poisson mixture <c>G ~ Gamma(r, 1)</c>, <c>Xᵢ ~ Poisson(G pᵢ / p₀)</c>).
/// The support consists of non-negative integer vectors. For components whose conditional mean exceeds <c>10¹⁸</c>, samples use a rounded normal approximation saturated at <see cref="long.MaxValue"/>.
/// </para>
/// <para>
/// 成功回数パラメーター <c>r</c> は <see cref="MaxSuccesses"/> 以下の任意の正の実数を指定できます（ガンマ・ポアソン混合 <c>G ~ Gamma(r, 1)</c>、<c>Xᵢ ~ Poisson(G pᵢ / p₀)</c> として相関のある過分散計数データをモデル化できます）。
/// 台は非負整数ベクトルです。条件付き平均が <c>10¹⁸</c> を超える成分の標本は正規近似の丸め値（<see cref="long.MaxValue"/> で飽和）となります。
/// </para>
/// </remarks>
public sealed class NegativeMultinomialDistribution : IMultivariateDiscreteDistribution
{
    /// <summary>
    /// The maximum supported value for <see cref="Successes"/> (<c>2⁵²</c>, matching <see cref="NegativeBinomialDistribution.MaxSuccesses"/>).
    /// <para><see cref="Successes"/> の上限（<c>2⁵²</c>、<see cref="NegativeBinomialDistribution.MaxSuccesses"/> と同一）。</para>
    /// </summary>
    public const double MaxSuccesses = NegativeBinomialDistribution.MaxSuccesses;

    private readonly double _r;
    private readonly double _p0;
    private readonly double[] _p;
    private readonly double _logSuccess;
    private readonly double[] _logOdds;

    /// <summary>
    /// Initializes a new instance of <see cref="NegativeMultinomialDistribution"/> with the specified success count, success probability, and failure probabilities.
    /// <para>成功回数、成功確率、および種類別の失敗確率を指定して <see cref="NegativeMultinomialDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="successes">The number of successes <c>r</c> (<c>0 &lt; r ≤ 2⁵²</c>; need not be an integer). / 成功回数 <c>r</c>（<c>0 &lt; r ≤ 2⁵²</c>、非整数も可）。</param>
    /// <param name="successProbability">The per-trial success probability <c>p₀</c> in <c>(0, 1]</c>. / 各試行の成功確率 <c>p₀</c>（<c>0 &lt; p₀ ≤ 1</c>）。</param>
    /// <param name="failureProbabilities">The non-negative finite failure probabilities <c>pᵢ</c> (at least 1 element; <c>p₀ + Σpᵢ</c> within <c>16(K + 1)ε</c> of 1, renormalized internally). / 各種類の失敗確率 <c>pᵢ</c>（1 個以上の非負有限値。<c>p₀ + Σpᵢ</c> が 1 から <c>16(K + 1)ε</c> 以内であり、内部で正規化されます）。</param>
    /// <exception cref="ArgumentException"><paramref name="failureProbabilities"/> is empty, or the total probability deviates from 1 beyond tolerance. / <paramref name="failureProbabilities"/> が空、または確率の総和が 1 から許容誤差を超えて外れている場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="successes"/> or <paramref name="successProbability"/> is out of range, or any failure probability is negative or non-finite. / <paramref name="successes"/>・<paramref name="successProbability"/> が範囲外、または失敗確率に負・非有限の要素が含まれる場合。</exception>
    public NegativeMultinomialDistribution(double successes, double successProbability, params ReadOnlySpan<double> failureProbabilities)
    {
        DistributionArgument.PositiveFinite(successes, nameof(successes));
        ArgumentOutOfRangeException.ThrowIfGreaterThan(successes, MaxSuccesses);
        if (!(successProbability > 0 && successProbability <= 1))
        {
            throw new ArgumentOutOfRangeException(nameof(successProbability), successProbability, "The probability must be in (0, 1].");
        }

        if (failureProbabilities.IsEmpty)
        {
            throw new ArgumentException("At least one kind of failure is required.", nameof(failureProbabilities));
        }

        var all = new double[failureProbabilities.Length + 1];
        all[0] = successProbability;
        failureProbabilities.CopyTo(all.AsSpan(1));
        all = MultivariateMath.Probabilities(all, nameof(failureProbabilities));

        _r = successes;
        _p0 = all[0];
        _p = all[1..];
        // Compute ln p₀ = log1p(-Σ pᵢ) when p₀ > 0.5 to avoid rounding loss near 1.
        double failure = 0;
        foreach (double value in _p)
        {
            failure += value;
        }

        _logSuccess = _p0 > 0.5 ? DistributionMath.Log1p(-failure) : Math.Log(_p0);
        _logOdds = new double[_p.Length];
        for (int i = 0; i < _p.Length; i++)
        {
            _logOdds[i] = Math.Log(_p[i]) - _logSuccess;
        }
    }

    /// <inheritdoc/>
    public int Dimension => _p.Length;

    /// <summary>
    /// Gets the success count (shape) parameter <c>r</c>.
    /// <para>成功回数（形状）パラメーター <c>r</c> を取得します。</para>
    /// </summary>
    public double Successes => _r;

    /// <summary>
    /// Gets the normalized success probability <c>p₀</c>.
    /// <para>正規化された成功確率 <c>p₀</c> を取得します。</para>
    /// </summary>
    public double SuccessProbability => _p0;

    /// <summary>
    /// Gets a read-only span of the normalized failure probabilities <c>p</c>.
    /// <para>正規化された各種類の失敗確率 <c>p</c> の読み取り専用スパンを取得します。</para>
    /// </summary>
    public ReadOnlySpan<double> FailureProbabilities => _p;

    /// <inheritdoc/>
    public Vector<double> GetMean()
    {
        var mean = Vector<double>.Uninitialized(Dimension);
        for (int i = 0; i < Dimension; i++)
        {
            mean[i] = DistributionMath.ScaledRatio([_r, _p[i]], [_p0]);
        }

        return mean;
    }

    /// <inheritdoc/>
    public Matrix<double> GetCovariance()
    {
        int k = Dimension;
        var covariance = Matrix<double>.Uninitialized(k, k);
        for (int j = 0; j < k; j++)
        {
            for (int i = 0; i < k; i++)
            {
                covariance[i, j] = DistributionMath.ScaledRatio([_r, _p[i], i == j ? _p[j] + _p0 : _p[j]], [_p0, _p0]);
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
        double failures = 0;
        for (int i = 0; i < x.Length; i++)
        {
            long value = x[i];
            if (value < 0 || (value > 0 && _p[i] == 0))
            {
                return double.NegativeInfinity;
            }

            failures += value;
        }

        if (failures == 0)
        {
            return _r * _logSuccess;
        }

        double total = _r + failures;
        // Group S(r) + (1/2) ln(total / r) via StirlingRemainderShifted to prevent overflow for subnormal r.
        double stirling = DistributionMath.StirlingRemainder(total) - DistributionMath.StirlingRemainderShifted(_r, failures);
        double logs = 0;
        double deviance = DistributionMath.Bd0(_r, total * _p0);
        for (int i = 0; i < x.Length; i++)
        {
            double m = total * _p[i];
            if (x[i] == 0)
            {
                deviance += m;
                continue;
            }

            double count = x[i];
            stirling -= DistributionMath.StirlingRemainder(count);
            logs -= 0.5 * (DistributionMath.LnTwoPi + Math.Log(count));
            deviance += DistributionMath.Bd0(count, m);
        }

        return stirling + logs - deviance;
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, VectorSpan<long> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        RequireLength(destination.Length, nameof(destination));
        SampleCore(random, destination);
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, MatrixSpan<long> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        RequireLength(destination.Rows, nameof(destination));
        for (int j = 0; j < destination.Columns; j++)
        {
            SampleCore(random, destination.Column(j));
        }
    }

    private void SampleCore(IRandomSource random, VectorSpan<long> destination)
    {
        double logGamma = Sampling.LogStandardGamma(random, _r);
        for (int i = 0; i < _logOdds.Length; i++)
        {
            destination[i] = Sampling.Poisson(random, Math.Exp(logGamma + _logOdds[i]));
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
