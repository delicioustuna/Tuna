using Tuna.RandomSource;

namespace Tuna.Statistics.Distributions;

/// <summary>
/// Represents the categorical distribution over <c>0, 1, …, K - 1</c> with probabilities proportional to the given weights.
/// <para>与えられた重みに比例する確率で <c>0, 1, …, K - 1</c> の値をとるカテゴリカル分布を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Category probabilities and alias sampling tables are copied and precomputed at construction time; instances are immutable and thread-safe.
/// </para>
/// <para>
/// 各カテゴリの確率およびエイリアス法サンプリング用のテーブルは構築時にコピー・前計算されるため、インスタンスは不変かつスレッドセーフです。
/// </para>
/// </remarks>
public sealed partial class CategoricalDistribution : IDiscreteDistribution
{
    private readonly double[] _probabilities;
    private readonly double[] _logProbabilities;
    private readonly double[] _cumulative;
    private readonly double[] _survival;
    private readonly double[] _aliasProbability;
    private readonly int[] _alias;
    private readonly long _supportMinimum;
    private readonly long _supportMaximum;

    /// <summary>
    /// Initializes a new instance of <see cref="CategoricalDistribution"/> from non-negative category weights.
    /// <para>非負の重みを指定して <see cref="CategoricalDistribution"/> を初期化します。確率は重みを総和で正規化した値になります。</para>
    /// </summary>
    /// <param name="weights">The non-negative finite weights for each category (at least one element, with a positive finite sum). / 各カテゴリの重み（1 個以上の非負有限値で、総和が正かつ有限）。</param>
    /// <exception cref="ArgumentException"><paramref name="weights"/> is empty. / <paramref name="weights"/> が空の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">Any weight is negative or non-finite, or the sum of weights is zero or non-finite. / いずれかの重みが負または非有限、あるいは総和が 0 または非有限の場合。</exception>
    public CategoricalDistribution(params ReadOnlySpan<double> weights)
    {
        if (weights.IsEmpty)
        {
            throw new ArgumentException("At least one category is required.", nameof(weights));
        }

        double total = 0;
        foreach (double weight in weights)
        {
            if (!(weight >= 0 && weight < double.PositiveInfinity))
            {
                throw new ArgumentOutOfRangeException(nameof(weights), weight, "Each weight must be non-negative and finite.");
            }

            total += weight;
        }

        if (!(total > 0 && total < double.PositiveInfinity))
        {
            throw new ArgumentOutOfRangeException(nameof(weights), total, "The total weight must be positive and finite.");
        }

        _probabilities = new double[weights.Length];
        for (int i = 0; i < weights.Length; i++)
        {
            _probabilities[i] = weights[i] / total;
        }

        _logProbabilities = LogProbabilities(_probabilities);
        (_cumulative, _survival) = CumulativeSums(_probabilities);
        (_aliasProbability, _alias) = BuildAlias(_probabilities);
        _supportMinimum = Array.FindIndex(_probabilities, x => x > 0);
        _supportMaximum = Array.FindLastIndex(_probabilities, x => x > 0);
    }

    /// <summary>
    /// Creates a <see cref="CategoricalDistribution"/> from log-weights without intermediate overflow or underflow.
    /// <para>重みの自然対数から <see cref="CategoricalDistribution"/> を生成します（対数が極端に大きい場合や小さい場合でもオーバーフローしません）。</para>
    /// </summary>
    /// <param name="logWeights">The log-weights for each category (<see cref="double.NegativeInfinity"/> denotes weight 0; at least one element must be finite). / 各カテゴリの対数重み（<see cref="double.NegativeInfinity"/> は重み 0、少なくとも 1 個は有限）。</param>
    /// <returns>A categorical distribution with probabilities computed via log-sum-exp normalization. / 正規化された確率を持つカテゴリカル分布。</returns>
    /// <exception cref="ArgumentException"><paramref name="logWeights"/> is empty. / <paramref name="logWeights"/> が空の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">Any element is <see cref="double.NaN"/> or <see cref="double.PositiveInfinity"/>, or all elements are <see cref="double.NegativeInfinity"/>. / いずれかの要素が <see cref="double.NaN"/> または <see cref="double.PositiveInfinity"/>、あるいはすべて <see cref="double.NegativeInfinity"/> の場合。</exception>
    public static CategoricalDistribution FromLogWeights(ReadOnlySpan<double> logWeights)
    {
        if (logWeights.IsEmpty)
        {
            throw new ArgumentException("At least one category is required.", nameof(logWeights));
        }

        double max = double.NegativeInfinity;
        foreach (double logWeight in logWeights)
        {
            if (double.IsNaN(logWeight) || double.IsPositiveInfinity(logWeight))
            {
                throw new ArgumentOutOfRangeException(nameof(logWeights), logWeight, "Each log weight must be finite or negative infinity.");
            }

            max = Math.Max(max, logWeight);
        }

        if (double.IsNegativeInfinity(max))
        {
            throw new ArgumentOutOfRangeException(nameof(logWeights), max, "At least one log weight must be finite.");
        }

        var weights = new double[logWeights.Length];
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = Math.Exp(logWeights[i] - max);
        }

        return new CategoricalDistribution(weights);
    }

    /// <summary>
    /// Gets the number of categories <c>K</c>.
    /// <para>カテゴリ数 <c>K</c> を取得します。</para>
    /// </summary>
    public int CategoryCount => _probabilities.Length;

    /// <summary>
    /// Gets a read-only span of the normalized category probabilities.
    /// <para>正規化された各カテゴリの確率の読み取り専用スパンを取得します。</para>
    /// </summary>
    public ReadOnlySpan<double> Probabilities => _probabilities;

    /// <inheritdoc/>
    public double Mean
    {
        get
        {
            double sum = 0;
            for (int k = 0; k < _probabilities.Length; k++)
            {
                sum += k * _probabilities[k];
            }

            return sum;
        }
    }

    private double CentralMoment(int order)
    {
        double mean = Mean;
        double sum = 0;
        for (int k = 0; k < _probabilities.Length; k++)
        {
            sum += Math.Pow(k - mean, order) * _probabilities[k];
        }

        return sum;
    }

    /// <inheritdoc/>
    public double Variance => CentralMoment(2);

    /// <inheritdoc/>
    public double StandardDeviation => Math.Sqrt(Variance);

    /// <inheritdoc/>
    public double Skewness
    {
        get
        {
            double variance = Variance;
            if (!(variance > 0))
            {
                return double.NaN;
            }

            // Divide sequentially by variance and sqrt(variance) when σ³ would underflow or become subnormal.
            double cube = variance * Math.Sqrt(variance);
            return cube >= DistributionMath.MinNormal ? CentralMoment(3) / cube : CentralMoment(3) / variance / Math.Sqrt(variance);
        }
    }

    /// <inheritdoc/>
    public double ExcessKurtosis
    {
        get
        {
            double variance = Variance;
            if (!(variance > 0))
            {
                return double.NaN;
            }

            double square = variance * variance;
            return (square >= DistributionMath.MinNormal ? CentralMoment(4) / square : CentralMoment(4) / variance / variance) - 3;
        }
    }

    /// <inheritdoc/>
    public double Entropy
    {
        get
        {
            double sum = 0;
            for (int k = 0; k < _probabilities.Length; k++)
            {
                if (_probabilities[k] > 0)
                {
                    sum -= _probabilities[k] * _logProbabilities[k];
                }
            }

            return sum;
        }
    }

    /// <inheritdoc/>
    public long Median => InverseCdf(0.5);

    /// <inheritdoc/>
    public long SupportMinimum => _supportMinimum;

    /// <inheritdoc/>
    public long SupportMaximum => _supportMaximum;

    /// <inheritdoc/>
    public double Pmf(long k) => k >= 0 && k < _probabilities.Length ? _probabilities[k] : 0;

    /// <inheritdoc/>
    public double LogPmf(long k) => k >= 0 && k < _probabilities.Length ? _logProbabilities[k] : double.NegativeInfinity;

    /// <summary>
    /// Evaluates the cumulative distribution function (CDF), <c>P(X ≤ k)</c>.
    /// <para>累積分布関数（CDF）<c>P(X ≤ k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns>The cumulative probability <c>P(X ≤ k)</c>. / 累積確率 <c>P(X ≤ k)</c>。</returns>
    public double Cdf(long k) => k < 0 ? 0 : k >= _probabilities.Length - 1 ? 1 : _cumulative[k];

    /// <summary>
    /// Evaluates the survival function, <c>P(X &gt; k)</c>.
    /// <para>生存関数（上側累積確率）<c>P(X &gt; k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns>The survival probability <c>P(X &gt; k)</c>. / 生存確率 <c>P(X &gt; k)</c>。</returns>
    public double Sf(long k) => k < 0 ? 1 : k >= _probabilities.Length - 1 ? 0 : _survival[k];

    /// <summary>
    /// Evaluates the natural logarithm of the cumulative distribution function, <c>ln P(X ≤ k)</c>.
    /// <para>累積分布関数の自然対数 <c>ln P(X ≤ k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns><c>ln P(X ≤ k)</c>, or <see cref="double.NegativeInfinity"/> when the probability is 0. / <c>ln P(X ≤ k)</c>。確率が 0 の場合は <see cref="double.NegativeInfinity"/>。</returns>
    public double LogCdf(long k)
    {
        double p = Cdf(k);
        return p <= 0.5 ? Math.Log(p) : DistributionMath.Log1p(-Sf(k));
    }

    /// <summary>
    /// Evaluates the natural logarithm of the survival function, <c>ln P(X &gt; k)</c>.
    /// <para>生存関数の自然対数 <c>ln P(X &gt; k)</c> を評価します。</para>
    /// </summary>
    /// <param name="k">The integer evaluation point. / 整数の評価点。</param>
    /// <returns><c>ln P(X &gt; k)</c>, or <see cref="double.NegativeInfinity"/> when the probability is 0. / <c>ln P(X &gt; k)</c>。確率が 0 の場合は <see cref="double.NegativeInfinity"/>。</returns>
    public double LogSf(long k)
    {
        double q = Sf(k);
        return q <= 0.5 ? Math.Log(q) : DistributionMath.Log1p(-Cdf(k));
    }

    /// <inheritdoc/>
    public long InverseCdf(double p) => DiscreteMath.InverseCdf(this, p, 0);

    /// <inheritdoc/>
    public long InverseSf(double q) => DiscreteMath.InverseSf(this, q, 0);

    /// <inheritdoc/>
    public long InverseLogCdf(double logP) => DiscreteMath.InverseLogCdf(this, logP, 0);

    /// <inheritdoc/>
    public long InverseLogSf(double logQ) => DiscreteMath.InverseLogSf(this, logQ, 0);

    private long SampleCore(IRandomSource random)
    {
        int column = random.NextInt32(_probabilities.Length);
        return Sampling.Bernoulli(random, _aliasProbability[column]) ? column : _alias[column];
    }

    /// <inheritdoc/>
    public long Sample(IRandomSource random)
    {
        ArgumentNullException.ThrowIfNull(random);
        return SampleCore(random);
    }

    /// <inheritdoc/>
    public void Sample(IRandomSource random, Span<long> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = SampleCore(random);
        }
    }

    // For probabilities > 0.5, compute ln(1 - sum_{j != k} p_j) via compensated sum to preserve accuracy near 1.
    private static double[] LogProbabilities(double[] probabilities)
    {
        var logProbabilities = new double[probabilities.Length];
        for (int k = 0; k < probabilities.Length; k++)
        {
            if (probabilities[k] <= 0.5)
            {
                logProbabilities[k] = Math.Log(probabilities[k]);
                continue;
            }

            double sum = 0, compensation = 0;
            for (int j = 0; j < probabilities.Length; j++)
            {
                if (j != k)
                {
                    Accumulate(ref sum, ref compensation, probabilities[j]);
                }
            }

            logProbabilities[k] = DistributionMath.Log1p(-(sum + compensation));
        }

        return logProbabilities;
    }

    // Prefix and suffix cumulative sums using Neumaier compensated summation.
    private static (double[] Cumulative, double[] Survival) CumulativeSums(double[] probabilities)
    {
        int n = probabilities.Length;
        var cumulative = new double[n];
        var survival = new double[n];
        double sum = 0, compensation = 0;
        for (int k = 0; k < n; k++)
        {
            Accumulate(ref sum, ref compensation, probabilities[k]);
            cumulative[k] = Math.Min(1, sum + compensation);
        }

        sum = 0;
        compensation = 0;
        for (int k = n - 1; k >= 0; k--)
        {
            survival[k] = Math.Min(1, sum + compensation);
            Accumulate(ref sum, ref compensation, probabilities[k]);
        }

        return (cumulative, survival);
    }

    private static void Accumulate(ref double sum, ref double compensation, double term)
    {
        double t = sum + term;
        compensation += Math.Abs(sum) >= Math.Abs(term) ? sum - t + term : term - t + sum;
        sum = t;
    }

    // Vose (1991) alias method table construction.
    private static (double[] Probability, int[] Alias) BuildAlias(double[] probabilities)
    {
        int n = probabilities.Length;
        var probability = new double[n];
        var alias = new int[n];
        var scaled = new double[n];
        var small = new Stack<int>();
        var large = new Stack<int>();
        for (int i = 0; i < n; i++)
        {
            scaled[i] = probabilities[i] * n;
            (scaled[i] < 1 ? small : large).Push(i);
        }

        while (small.Count > 0 && large.Count > 0)
        {
            int less = small.Pop();
            int more = large.Pop();
            probability[less] = scaled[less];
            alias[less] = more;
            scaled[more] = scaled[more] + scaled[less] - 1;
            (scaled[more] < 1 ? small : large).Push(more);
        }

        // Remaining columns due to rounding get probability 1 (self), except zero-probability categories which delegate to the mode.
        int largest = Array.IndexOf(probabilities, probabilities.Max());
        foreach (int i in large.Concat(small))
        {
            bool positive = probabilities[i] > 0;
            probability[i] = positive ? 1 : 0;
            alias[i] = positive ? i : largest;
        }

        return (probability, alias);
    }
}
