namespace Tuna.RandomSource;

/// <summary>
/// Samples integer compositions satisfying per-component lower and upper bounds and a fixed total from an approximately uniform distribution.
/// <para>各成分の上下限および総和の制約を満たす整数の組を、近似的な一様分布からサンプリングします。</para>
/// </summary>
/// <remarks>
/// <para>Samples from the finite set <c>{x ∈ ℤⁿ : lᵢ ≤ xᵢ ≤ uᵢ, Σxᵢ = s}</c>. Due to the <c>2⁻⁵³</c> resolution of <see cref="IRandomSource.NextDouble"/> and floating-point rounding in the dynamic programming table, the resulting distribution is approximately uniform; extremely rare compositions may have probability rounded to zero or slightly perturbed.</para>
/// <para>With <c>m = s - Σlᵢ</c>, <c>cᵢ = uᵢ - lᵢ</c>, and <c>m' = min(m, Σcᵢ - m)</c>, construction takes <c>O(n·m')</c> time and memory, and each sample takes <c>O(Σ min(cᵢ, m'))</c> time. Instances are immutable after construction, and <see cref="Sample(IRandomSource, Span{long})"/> is thread-safe across concurrent calls.</para>
/// <para>対象集合は <c>{x ∈ ℤⁿ : lᵢ ≤ xᵢ ≤ uᵢ, Σxᵢ = s}</c> です。<see cref="IRandomSource.NextDouble"/> の刻み幅 <c>2⁻⁵³</c> と動的計画法テーブルの浮動小数点丸めのため、厳密な一様分布ではなく近似的な一様分布となり、極めて確率の低い組は出現しないかわずかに偏る場合があります。</para>
/// <para><c>m = s - Σlᵢ</c>、<c>cᵢ = uᵢ - lᵢ</c>、<c>m' = min(m, Σcᵢ - m)</c> とすると、構築の計算量およびメモリは <c>O(n·m')</c>、1 回のサンプリングは <c>O(Σ min(cᵢ, m'))</c> です。構築後は不変であり、<see cref="Sample(IRandomSource, Span{long})"/> は並行呼び出しに対してスレッドセーフです。</para>
/// </remarks>
public sealed class BoundedCompositionSampler
{
    private readonly long[] _lower;
    private readonly long[] _upper;

    // 各成分の幅 cᵢ = uᵢ - lᵢ を簡約後の和 m' で打ち切った値。
    private readonly int[] _widths;

    // 行 k + 1（k = 0, …, n - 2）の Nₖ₊₁(t) を行ごとの 2 のべき乗スケールで格納した平坦配列。
    private readonly double[] _table;
    private readonly int[] _rowOffsets;
    private readonly int[] _rowLengths;

    // 補集合側（zᵢ = cᵢ - yᵢ）で簡約している場合は true。
    private readonly bool _complement;

    // 簡約後の総和 m' = min(m, Σcᵢ - m)。
    private readonly int _reducedTotal;

    /// <summary>
    /// Initializes a new <see cref="BoundedCompositionSampler"/> and precomputes the counting table for the specified bounds and total.
    /// <para>指定した上下限および総和に対する計数テーブルを前計算し、<see cref="BoundedCompositionSampler"/> の新しいインスタンスを初期化します。</para>
    /// </summary>
    /// <param name="lower">The inclusive lower bounds <c>lᵢ</c> for each component. / 各成分の包括的下限 <c>lᵢ</c>。</param>
    /// <param name="upper">The inclusive upper bounds <c>uᵢ</c> for each component (<c>lᵢ ≤ uᵢ</c>). / 各成分の包括的上限 <c>uᵢ</c>（<c>lᵢ ≤ uᵢ</c>）。</param>
    /// <param name="total">The required sum <c>s</c> of all components (<c>Σlᵢ ≤ s ≤ Σuᵢ</c>). / 全成分の総和 <c>s</c>（<c>Σlᵢ ≤ s ≤ Σuᵢ</c>）。</param>
    /// <exception cref="ArgumentException"><paramref name="lower"/> is empty, <paramref name="lower"/> and <paramref name="upper"/> have different lengths, <c>lower[i] &gt; upper[i]</c> for some index, or <paramref name="total"/> lies outside <c>[Σlᵢ, Σuᵢ]</c>. / <paramref name="lower"/> が空の場合、<paramref name="lower"/> と <paramref name="upper"/> の長さが異なる場合、ある成分で <c>lower[i] &gt; upper[i]</c> の場合、または <paramref name="total"/> が <c>[Σlᵢ, Σuᵢ]</c> の範囲外の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">The required counting table size exceeds <see cref="Array.MaxLength"/> or the reduced total cannot fit in a 32-bit signed integer. / 必要な計数テーブルのサイズが <see cref="Array.MaxLength"/> を超えるか、簡約後の総和が 32 ビット符号付き整数に収まらない場合。</exception>
    public BoundedCompositionSampler(ReadOnlySpan<long> lower, ReadOnlySpan<long> upper, long total)
    {
        if (lower.IsEmpty)
        {
            throw new ArgumentException("At least one component is required.", nameof(lower));
        }

        if (lower.Length != upper.Length)
        {
            throw new ArgumentException("The lower and upper bounds must have the same length.", nameof(upper));
        }

        int n = lower.Length;
        Int128 lowerSum = 0, upperSum = 0;
        for (int i = 0; i < n; i++)
        {
            if (lower[i] > upper[i])
            {
                throw new ArgumentException($"The lower bound exceeds the upper bound at index {i}.", nameof(upper));
            }

            lowerSum += lower[i];
            upperSum += upper[i];
        }

        if (total < lowerSum || total > upperSum)
        {
            throw new ArgumentException("The total must lie between the sums of the lower and upper bounds.", nameof(total));
        }

        _lower = lower.ToArray();
        _upper = upper.ToArray();
        Total = total;
        Length = n;

        // 簡約問題 0 ≤ yᵢ ≤ cᵢ, Σyᵢ = m。m > Σcᵢ / 2 のときは補集合側 zᵢ = cᵢ - yᵢ（和 Σcᵢ - m）へ変換してテーブル幅を抑える。
        UInt128 capacity = (UInt128)(upperSum - lowerSum);
        UInt128 m = (UInt128)(total - lowerSum);
        _complement = m > capacity - m;
        UInt128 reduced = _complement ? capacity - m : m;
        if (reduced >= int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(total), "The table for the given bounds and total is too large.");
        }

        int mr = (int)reduced;
        _reducedTotal = mr;
        _widths = new int[n];
        for (int i = 0; i < n; i++)
        {
            ulong width = unchecked((ulong)upper[i] - (ulong)lower[i]);
            _widths[i] = width < (ulong)mr ? (int)width : mr;
        }

        // 行 k の幅 Wₖ = min(m', cₖ + … + cₙ₋₁)。テーブルには行 1..n を格納し、行 0 は総数計算時に m' の位置のみ評価する。
        _rowLengths = new int[n];
        _rowOffsets = new int[n];
        long size = 0;
        long suffix = 0;
        for (int k = n - 1; k >= 0; k--)
        {
            // 行 k + 1 の長さ（行 n は和が 0 の 1 要素のみ）。
            _rowLengths[k] = (int)Math.Min(suffix, mr) + 1;
            suffix = Math.Min(suffix + _widths[k], mr);
            size += _rowLengths[k];
        }

        if (size > Array.MaxLength)
        {
            throw new ArgumentOutOfRangeException(nameof(total), "The table for the given bounds and total is too large.");
        }

        _table = new double[size];
        long offset = 0;
        for (int k = 0; k < n; k++)
        {
            _rowOffsets[k] = (int)offset;
            offset += _rowLengths[k];
        }

        LogCount = Build();
    }

    /// <summary>
    /// Gets the number of components <c>n</c>.
    /// <para>成分の数 <c>n</c> を取得します。</para>
    /// </summary>
    public int Length { get; }

    /// <summary>
    /// Gets the required total sum <c>s</c>.
    /// <para>全成分の総和 <c>s</c> を取得します。</para>
    /// </summary>
    public long Total { get; }

    /// <summary>
    /// Gets the natural logarithm <c>ln N</c> of the number of feasible compositions.
    /// <para>条件を満たす組の総数の自然対数 <c>ln N</c> を取得します。</para>
    /// </summary>
    public double LogCount { get; }

    /// <summary>
    /// Draws one composition and writes its components to <paramref name="destination"/>.
    /// <para>条件を満たす組を 1 つサンプリングし、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="random">The random source used for sampling. / サンプリングに使用する乱数源。</param>
    /// <param name="destination">The destination span of length <see cref="Length"/>. / 長さ <see cref="Length"/> の出力先スパン。</param>
    /// <remarks>
    /// <para>Consumes exactly <c>Length - 1</c> values from <see cref="IRandomSource.NextDouble"/> in component order <c>k = 0, 1, …, Length - 2</c>, including components that have only one feasible value.</para>
    /// <para>成分 <c>k = 0, 1, …, Length - 2</c> の順に <see cref="IRandomSource.NextDouble"/> を 1 個ずつ（計 <c>Length - 1</c> 個）消費します。取り得る値が 1 通りのみの成分でも 1 個消費します。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="random"/> is <see langword="null"/>. / <paramref name="random"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <see cref="Length"/>. / <paramref name="destination"/> の長さが <see cref="Length"/> と等しくない場合。</exception>
    public void Sample(IRandomSource random, Span<long> destination)
    {
        ArgumentNullException.ThrowIfNull(random);
        if (destination.Length != Length)
        {
            throw new ArgumentException("The destination length must equal the number of components.", nameof(destination));
        }

        int n = Length;
        int t = _reducedTotal;
        for (int k = 0; k < n - 1; k++)
        {
            // yₖ = v の重みは Nₖ₊₁(t - v)（v = lo..hi）。合計を求めた後、同じ順序で累積しながら target を超える最初の v を選択する。
            var row = _table.AsSpan(_rowOffsets[k], _rowLengths[k]);
            int lo = Math.Max(0, t - (row.Length - 1));
            int hi = Math.Min(_widths[k], t);
            double sum = 0;
            for (int v = lo; v <= hi; v++)
            {
                sum += row[t - v];
            }

            double target = random.NextDouble() * sum;
            int chosen = hi;
            double cumulative = 0;
            for (int v = lo; v <= hi; v++)
            {
                cumulative += row[t - v];
                if (target < cumulative)
                {
                    chosen = v;
                    break;
                }
            }

            destination[k] = Component(k, chosen);
            t -= chosen;
        }

        destination[n - 1] = Component(n - 1, t);
    }

    private long Component(int k, int y) => _complement ? _upper[k] - y : _lower[k] + y;

    // 行 n - 1 から行 1 まで後ろ向きに計数テーブルを構築し、組の総数の自然対数 ln N₀(m') を返す。
    private double Build()
    {
        int n = Length;
        int mr = _reducedTotal;

        // 行 n: Nₙ(0) = 1。各行の格納値は真の計数値の 2^(-exponent) 倍。
        double[] next = [1.0];
        int nextLength = 1;
        long exponent = 0;
        var prefix = new double[Math.Max(1, mr + 1)];
        var suffix = new double[Math.Max(1, mr + 1)];
        double[] current = new double[Math.Max(1, mr + 1)];
        for (int k = n - 1; k >= 0; k--)
        {
            // 行 k + 1（next）をテーブルに格納する。行 n はテーブル末尾の行 _rowLengths[n - 1]。
            next.AsSpan(0, nextLength).CopyTo(_table.AsSpan(_rowOffsets[k], _rowLengths[k]));
            if (k == 0)
            {
                break;
            }

            int length = _rowLengths[k - 1];
            WindowSums(next.AsSpan(0, nextLength), _widths[k], current.AsSpan(0, length), prefix, suffix);

            // 最大値が [1, 2) に収まるよう 2 のべき乗でスケーリングする（サンプリングでは行内の比のみを使うため、行ごとに独立にスケーリングできる）。
            double max = 0;
            for (int t = 0; t < length; t++)
            {
                max = Math.Max(max, current[t]);
            }

            int scale = Math.ILogB(max);
            for (int t = 0; t < length; t++)
            {
                current[t] = Math.ScaleB(current[t], -scale);
            }

            exponent += scale;
            if (next.Length < length)
            {
                next = new double[Math.Max(1, mr + 1)];
            }

            current.AsSpan(0, length).CopyTo(next);
            nextLength = length;
        }

        // N₀(m') = Σ_(v) N₁(m' - v)（行 1 はテーブルの行 0）。
        var first = _table.AsSpan(_rowOffsets[0], _rowLengths[0]);
        int low = Math.Max(0, mr - (first.Length - 1));
        int high = Math.Min(_widths[0], mr);
        double sum = 0, compensation = 0;
        for (int v = low; v <= high; v++)
        {
            Add(ref sum, ref compensation, first[mr - v]);
        }

        return Math.Log(sum + compensation) + exponent * Math.Log(2);
    }

    // 長さ width + 1 のブロックごとに前方・後方の累積和を補償付きで計算する（van Herk–Gil–Werman 法）。
    // 任意の窓和 result[t] = Σ_(s = max(0, t - width))^(min(t, values.Length - 1)) values[s] が高々 2 ブロックの加算で表せるため、減算による桁落ちを避けられる。
    private static void WindowSums(ReadOnlySpan<double> values, int width, Span<double> result, Span<double> prefix, Span<double> suffix)
    {
        int last = values.Length - 1;
        int block = width + 1;
        for (int start = 0; start <= last; start += block)
        {
            int end = Math.Min(start + block - 1, last);
            double sum = 0, compensation = 0;
            for (int s = start; s <= end; s++)
            {
                Add(ref sum, ref compensation, values[s]);
                prefix[s] = sum + compensation;
            }

            sum = 0;
            compensation = 0;
            for (int s = end; s >= start; s--)
            {
                Add(ref sum, ref compensation, values[s]);
                suffix[s] = sum + compensation;
            }
        }

        for (int t = 0; t < result.Length; t++)
        {
            int low = Math.Max(0, t - width), high = Math.Min(t, last);

            // 窓はブロック先頭開始（prefix のみ）、ブロック末尾終了（suffix のみ）、または 2 ブロック跨ぎ（suffix + prefix）のいずれかになる。
            if (low % block == 0)
            {
                result[t] = prefix[high];
            }
            else if (high == last || high % block == block - 1)
            {
                result[t] = low / block == high / block ? suffix[low] : suffix[low] + prefix[high];
            }
            else
            {
                result[t] = suffix[low] + prefix[high];
            }
        }
    }

    // Neumaier 補償加算（丸め誤差を compensation に蓄積する）。
    private static void Add(ref double sum, ref double compensation, double value)
    {
        double next = sum + value;
        compensation += Math.Abs(sum) >= Math.Abs(value) ? (sum - next) + value : (value - next) + sum;
        sum = next;
    }
}
