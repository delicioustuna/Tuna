using Tuna.LinearAlgebra;
using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the multinomial distribution <c>Mult(n, p)</c> for category counts across <c>n</c> independent trials.
/// <para><c>n</c> 回の独立試行における各結果の出現回数を表す多項分布 <c>Mult(n, p)</c> です。</para>
/// </summary>
/// <remarks>
/// <para>
/// The support consists of non-negative integer vectors summing to <c>n</c>; all other points have probability 0.
/// </para>
/// <para>
/// 台は非負整数成分からなり総和が <c>n</c> のベクトルであり、それ以外の点では確率 0 を返します。
/// </para>
/// </remarks>
public sealed class MultinomialDistribution : IMultivariateDiscreteDistribution
{
    /// <summary>
    /// The maximum supported number of trials (<c>2⁵³</c>, matching <see cref="BinomialDistribution.MaxTrials"/>).
    /// <para>サポートされる試行回数の上限（<c>2⁵³</c>、<see cref="BinomialDistribution.MaxTrials"/> と同一）。</para>
    /// </summary>
    public const long MaxTrials = BinomialDistribution.MaxTrials;

    private readonly long _trials;
    private readonly double[] _p;
    private readonly double[] _suffix;
    private readonly double[] _others;

    /// <summary>
    /// Initializes a new instance of <see cref="MultinomialDistribution"/> with the specified trial count and category probabilities.
    /// <para>試行回数と各結果の確率を指定して <see cref="MultinomialDistribution"/> を初期化します。</para>
    /// </summary>
    /// <param name="trials">The number of trials <c>n</c> (<c>0 ≤ n ≤ 2⁵³</c>). / 試行回数 <c>n</c>（<c>0 ≤ n ≤ 2⁵³</c>）。</param>
    /// <param name="probabilities">The non-negative finite outcome probabilities <c>pᵢ</c> (at least 1 element; sum within <c>16Kε</c> of 1, renormalized internally). / 各結果の確率 <c>pᵢ</c>（1 個以上の非負有限値。総和が 1 から <c>16Kε</c> 以内であり、内部で正規化されます）。</param>
    /// <exception cref="ArgumentException"><paramref name="probabilities"/> is empty or its sum deviates from 1 beyond tolerance. / <paramref name="probabilities"/> が空、または総和が 1 から許容誤差を超えて外れている場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="trials"/> is outside <c>[0, MaxTrials]</c>, or any probability is negative or non-finite. / <paramref name="trials"/> が <c>[0, MaxTrials]</c> の範囲外、または負・非有限の確率が含まれる場合。</exception>
    public MultinomialDistribution(long trials, params ReadOnlySpan<double> probabilities)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(trials);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(trials, MaxTrials);
        _trials = trials;
        _p = MultivariateMath.Probabilities(probabilities, nameof(probabilities));
        (_suffix, _others) = MultivariateMath.SuffixAndOthers(_p);
    }

    /// <inheritdoc/>
    public int Dimension => _p.Length;

    /// <summary>
    /// Gets the number of trials <c>n</c>.
    /// <para>試行回数 <c>n</c> を取得します。</para>
    /// </summary>
    public long Trials => _trials;

    /// <summary>
    /// Gets a read-only span of the normalized outcome probabilities <c>p</c>.
    /// <para>正規化された各結果の確率 <c>p</c> の読み取り専用スパンを取得します。</para>
    /// </summary>
    public ReadOnlySpan<double> Probabilities => _p;

    /// <inheritdoc/>
    public Vector<double> GetMean()
    {
        var mean = Vector<double>.Uninitialized(Dimension);
        for (int i = 0; i < Dimension; i++)
        {
            mean[i] = _trials * _p[i];
        }

        return mean;
    }

    /// <inheritdoc/>
    public Matrix<double> GetCovariance()
    {
        int k = Dimension;
        double n = _trials;
        var covariance = Matrix<double>.Uninitialized(k, k);
        for (int j = 0; j < k; j++)
        {
            for (int i = 0; i < k; i++)
            {
                covariance[i, j] = i == j ? n * _p[i] * _others[i] : -(n * _p[i]) * _p[j];
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

        // Accumulate Stirling remainders, log(2π*count)/2, and Loader deviance bd0 separately so single-bin outcomes cancel to exact 0.
        double n = _trials;
        double stirling = DistributionMath.StirlingRemainder(n);
        double logs = 0.5 * (DistributionMath.LnTwoPi + Math.Log(n));
        double deviance = 0;
        for (int i = 0; i < x.Length; i++)
        {
            double m = n * _p[i];
            if (x[i] == 0)
            {
                deviance += m;
                continue;
            }

            if (_p[i] == 0)
            {
                return double.NegativeInfinity;
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
        Sampling.Multinomial(random, _trials, _p, _suffix, destination);
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, MatrixSpan<long> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        RequireLength(destination.Rows, nameof(destination));
        for (int j = 0; j < destination.Columns; j++)
        {
            Sampling.Multinomial(random, _trials, _p, _suffix, destination.Column(j));
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
