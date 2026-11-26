using Tuna.Integration.Internal;
using Tuna.LinearAlgebra;

namespace Tuna.Integration;

/// <summary>
/// Provides adaptive numerical integration over multidimensional hyperrectangles.
/// <para>多次元の超直方体上の適応型数値積分を提供します。</para>
/// </summary>
/// <remarks>
/// <para>Infinite bounds are supported on any axis. If the main contribution lies far from the origin or has a scale very different from 1, shift or rescale the variables before integrating. If the running product of axis widths falls outside the supported range, reorder or rescale the axes. For high-dimensional or non-smooth integrands, consider <see cref="QuasiMonteCarlo"/>.</para>
/// <para>The integrand is evaluated on the calling thread. If an evaluation or accumulated sum becomes non-finite, integration terminates with <see cref="IntegrationTermination.NonFiniteValue"/>.</para>
/// <para>各軸に無限大の境界を指定できます。主な寄与が原点から離れている場合やスケールが 1 と大きく異なる場合は、事前に変数を平行移動またはスケーリングしてください。軸幅の累積積が許容範囲を外れる場合は、軸の順序やスケールを変更してください。高次元または滑らかでない被積分関数には <see cref="QuasiMonteCarlo"/> も検討できます。</para>
/// <para>被積分関数は呼び出し元スレッドで評価され、評価値または積分値・誤差の和が有限値でなくなると <see cref="IntegrationTermination.NonFiniteValue"/> で終了します。</para>
/// </remarks>
public static class Cubature
{
    /// <summary>
    /// The maximum supported number of dimensions.
    /// <para>サポートされる最大次元数。</para>
    /// </summary>
    public const int MaxDimension = 20;

    private const double RelativeMinimumWidth = 1024 * GaussKronrod.MachineEpsilon;
    private const double AbsoluteMinimumWidth = 1e-300;

    // 軸スコアが相対的にこの範囲内なら同率とみなし、幅の広い軸を分割する
    private const double DifferenceTie = 1e-12;

    // 3次元以下は軸ごとの15点 Gauss–Kronrod 直積則（15^d 点）、4次元以上は Genz–Malik 則を使用
    private const int ProductRuleMaxDimension = 3;

    // 各軸の幅（t 空間）を先頭軸から順に掛けた累積積がオーバーフローまたは 0 にアンダーフローする領域を防ぐため、
    // 細分化後の体積も正規化数の最小値（2.2e-308）を上回るよう余裕を持った上下限を設ける
    private const double MinimumVolume = 1e-250;
    private const double MaximumVolume = 1e250;

    /// <summary>
    /// Computes the definite integral over the hyperrectangle <c>Π [lowerᵢ, upperᵢ]</c>.
    /// <para>超直方体 <c>Π [lowerᵢ, upperᵢ]</c> 上の定積分を計算します。</para>
    /// </summary>
    /// <param name="integrand">The integrand function. / 被積分関数。</param>
    /// <param name="lower">The lower integration bounds for each axis (may include ±∞). / 各軸の下端（±∞ を指定可能）。</param>
    /// <param name="upper">The upper integration bounds for each axis (may include ±∞). / 各軸の上端（±∞ を指定可能）。</param>
    /// <param name="options">Integration options, or <see langword="null"/> to use default settings. / 積分の設定。<see langword="null"/> の場合は既定値を使用します。</param>
    /// <returns>The integral value, error estimate, evaluation count, and termination reason. Returns 0 without evaluating the integrand if any axis has equal bounds; reverses the sign for each axis whose lower bound exceeds its upper bound. / 積分値、誤差見積もり、評価回数、および終了理由。上下端が等しい軸がある場合は被積分関数を評価せずに 0 を返し、上下端が逆順の軸ごとに符号を反転します。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="integrand"/> is <see langword="null"/>. / <paramref name="integrand"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="lower"/> and <paramref name="upper"/> have different lengths or contain <see cref="double.NaN"/>. / <paramref name="lower"/> と <paramref name="upper"/> の長さが異なる場合、または <see cref="double.NaN"/> を含む場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">The dimension is outside <c>[1, MaxDimension]</c>, any running product of transformed axis widths is outside <c>[1e-250, 1e250]</c> (infinite half-lines have width 1 and <c>(-∞, ∞)</c> has width 4), a tolerance is negative or non-finite, or the evaluation limit is smaller than the initial region count (<c>3^k</c> for <c>k</c> doubly infinite axes) times the rule point count. / 次元が 1 以上 <see cref="MaxDimension"/> 以下の範囲外の場合、先頭軸から順に掛けた軸幅の積のいずれかが 1e-250 以上 1e250 以下の範囲外の場合（片側無限の軸幅は 1、<c>(-∞, ∞)</c> の軸幅は 4）、許容誤差が負または有限値でない場合、あるいは評価回数の上限が初期領域数（<c>(-∞, ∞)</c> の軸 k 本に対し 3^k）× 1回の求積則の点数未満の場合。</exception>
    public static IntegrationResult Integrate(Func<ReadOnlyVectorSpan<double>, double> integrand, ReadOnlySpan<double> lower, ReadOnlySpan<double> upper,
        IntegrationOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(integrand);
        return Integrate(new DelegateMultivariateIntegrand(integrand), lower, upper, options);
    }

    /// <summary>
    /// Computes the definite integral over the hyperrectangle <c>Π [lowerᵢ, upperᵢ]</c>.
    /// <para>超直方体 <c>Π [lowerᵢ, upperᵢ]</c> 上の定積分を計算します。</para>
    /// </summary>
    /// <typeparam name="TIntegrand">The type of the integrand. / 被積分関数の型。</typeparam>
    /// <inheritdoc cref="Integrate(Func{ReadOnlyVectorSpan{double}, double}, ReadOnlySpan{double}, ReadOnlySpan{double}, IntegrationOptions?)"/>
    public static IntegrationResult Integrate<TIntegrand>(TIntegrand integrand, ReadOnlySpan<double> lower, ReadOnlySpan<double> upper,
        IntegrationOptions? options = null)
        where TIntegrand : IMultivariateIntegrand
    {
        ArgumentNullException.ThrowIfNull(integrand);
        options ??= IntegrationOptions.Default;
        options.Validate();
        if (lower.Length != upper.Length)
        {
            throw new ArgumentException("The lower and upper limits must have the same length.", nameof(upper));
        }

        int d = lower.Length;
        ArgumentOutOfRangeException.ThrowIfLessThan(d, 1, nameof(lower));
        ArgumentOutOfRangeException.ThrowIfGreaterThan(d, MaxDimension, nameof(lower));

        var kinds = new AxisKind[d];
        var origins = new double[d];
        var t0 = new double[d];
        var t1 = new double[d];
        double sign = 1;
        bool empty = false;
        for (int i = 0; i < d; i++)
        {
            double a = lower[i], b = upper[i];
            if (double.IsNaN(a))
            {
                throw new ArgumentException("The lower limits must not be NaN.", nameof(lower));
            }

            if (double.IsNaN(b))
            {
                throw new ArgumentException("The upper limits must not be NaN.", nameof(upper));
            }

            if (a > b)
            {
                (a, b) = (b, a);
                sign = -sign;
            }

            empty |= a == b;
            (kinds[i], origins[i], t0[i], t1[i]) = (double.IsNegativeInfinity(a), double.IsPositiveInfinity(b)) switch
            {
                (false, false) => (AxisKind.Finite, 0.0, a, b),
                (false, true) => (AxisKind.UpperTail, a, 0.0, 1.0),
                (true, false) => (AxisKind.LowerTail, b, 0.0, 1.0),
                (true, true) => (AxisKind.FullLine, 0.0, -1.0, 3.0),
            };
        }

        if (empty)
        {
            return new IntegrationResult(0, 0, 0, IntegrationTermination.Tolerance);
        }

        double volume = 1;
        for (int i = 0; i < d; i++)
        {
            volume *= t1[i] - t0[i];
            if (!(volume >= MinimumVolume && volume <= MaximumVolume))
            {
                throw new ArgumentOutOfRangeException(nameof(upper),
                    "The running products of the axis widths must lie between 1e-250 and 1e250; reorder or rescale the variables.");
            }
        }

        // 最初の領域の数（(−∞, ∞) の軸 k 本で 3^k）から評価の回数を求め、領域を作る前に上限を確かめる
        ICubatureRule rule = d <= ProductRuleMaxDimension ? new ProductGaussKronrod(d) : new GenzMalik(d);
        long initialRegions = 1;
        foreach (var kind in kinds)
        {
            if (kind == AxisKind.FullLine)
            {
                initialRegions *= 3;
            }
        }

        options.ValidateBudget(initialRegions * rule.Points);
        var regions = InitialRegions(kinds, t0, t1);
        var mapped = new Mapped<TIntegrand>(integrand, kinds, origins);
        var run = new Adaptive<TIntegrand>(mapped, rule);
        var result = run.Integrate(regions, options);
        return result with { Value = sign * result.Value };
    }

    // (−∞, ∞) の軸は t の区間 [−1, 0]（下側の裾）、[0, 1]（上側の裾）、[1, 3]（[−1, 1]）の3つに分割し直積を作る。
    // 変換の継ぎ目が領域の内部に入らないため、求積則は継ぎ目で評価しない。
    private static List<(double[] Lower, double[] Upper)> InitialRegions(AxisKind[] kinds, double[] t0, double[] t1)
    {
        var regions = new List<(double[] Lower, double[] Upper)> { ((double[])t0.Clone(), (double[])t1.Clone()) };
        for (int i = 0; i < kinds.Length; i++)
        {
            if (kinds[i] != AxisKind.FullLine)
            {
                continue;
            }

            var split = new List<(double[] Lower, double[] Upper)>(regions.Count * 3);
            foreach (var (lower, upper) in regions)
            {
                foreach (var (a, b) in (ReadOnlySpan<(double, double)>)[(-1.0, 0.0), (0.0, 1.0), (1.0, 3.0)])
                {
                    var l = (double[])lower.Clone();
                    var u = (double[])upper.Clone();
                    (l[i], u[i]) = (a, b);
                    split.Add((l, u));
                }
            }

            regions = split;
        }

        return regions;
    }

    private enum AxisKind : byte
    {
        Finite,
        LowerTail,
        UpperTail,
        FullLine,
    }

    // 軸ごとの変数変換を適用した被積分関数 f(x(t))·Π dxᵢ/dtᵢ
    private struct Mapped<T> : IBoxIntegrand where T : IMultivariateIntegrand
    {
        private T f;
        private readonly AxisKind[] kinds;
        private readonly double[] origins;
        private readonly double[] x;
        private readonly bool identity;

        public Mapped(T f, AxisKind[] kinds, double[] origins)
        {
            this.f = f;
            this.kinds = kinds;
            this.origins = origins;
            x = new double[kinds.Length];
            identity = Array.TrueForAll(kinds, k => k == AxisKind.Finite);
        }

        public double Evaluate(Span<double> t)
        {
            if (identity)
            {
                return f.Evaluate(new ReadOnlyVectorSpan<double>(t));
            }

            for (int i = 0; i < x.Length; i++)
            {
                double ti = t[i];
                x[i] = kinds[i] switch
                {
                    AxisKind.UpperTail => origins[i] + (1 - ti) / ti,
                    AxisKind.LowerTail => origins[i] - (1 - ti) / ti,
                    AxisKind.FullLine => ti < 0 ? -1 + (1 + ti) / ti : ti <= 1 ? 1 + (1 - ti) / ti : ti - 2,
                    _ => ti,
                };
            }

            double value = f.Evaluate(x);

            // ヤコビアンの積のオーバーフローを避けるため、1軸ずつ除算する（値が 0 なら 0 のまま維持される）
            for (int i = 0; i < x.Length; i++)
            {
                double ti = t[i];
                switch (kinds[i])
                {
                    case AxisKind.UpperTail:
                    case AxisKind.LowerTail:
                        value = value / ti / ti;
                        break;
                    case AxisKind.FullLine when ti <= 1:
                        value = value / ti / ti;
                        break;
                }
            }

            return value;
        }
    }

    private sealed record Region(double[] Lower, double[] Upper, double Value, double Error, int SplitAxis);

    // 誤差見積もりが最大の領域から順に二分割する大域的適応型求積
    private sealed class Adaptive<T>(Mapped<T> f, ICubatureRule rule) where T : IMultivariateIntegrand
    {
        private Mapped<T> f = f;
        private readonly PriorityQueue<Region, double> heap = new();
        private readonly List<Region> frozen = [];
        private CompensatedSum value;
        private CompensatedSum heapError;
        private CompensatedSum frozenError;
        private bool tooSmall;
        private long evaluations;

        public IntegrationResult Integrate(List<(double[] Lower, double[] Upper)> regions, IntegrationOptions options)
        {
            foreach (var (lower, upper) in regions)
            {
                if (!Apply(lower, upper))
                {
                    return NonFinite();
                }
            }

            while (true)
            {
                double target = Math.Max(options.AbsoluteTolerance, options.RelativeTolerance * Math.Abs(value.Value));
                if (heapError.Value + frozenError.Value <= target)
                {
                    Recompute();
                    target = Math.Max(options.AbsoluteTolerance, options.RelativeTolerance * Math.Abs(value.Value));
                    if (heapError.Value + frozenError.Value <= target)
                    {
                        return Result(IntegrationTermination.Tolerance);
                    }
                }

                if (frozenError.Value > target || heap.Count == 0)
                {
                    return Result(tooSmall ? IntegrationTermination.IntervalTooSmall : IntegrationTermination.RoundoffLimited);
                }

                if (evaluations > options.MaxEvaluations - 2 * rule.Points)
                {
                    return Result(IntegrationTermination.MaxEvaluations);
                }

                var worst = heap.Dequeue();
                heapError.Add(-worst.Error);
                int axis = worst.SplitAxis;
                if (axis < 0)
                {
                    tooSmall = true;
                    frozen.Add(worst);
                    frozenError.Add(worst.Error);
                    continue;
                }

                value.Add(-worst.Value);
                double middle = 0.5 * worst.Lower[axis] + 0.5 * worst.Upper[axis];
                var leftUpper = (double[])worst.Upper.Clone();
                leftUpper[axis] = middle;
                var rightLower = (double[])worst.Lower.Clone();
                rightLower[axis] = middle;
                if (!Apply(worst.Lower, leftUpper) || !Apply(rightLower, worst.Upper))
                {
                    return NonFinite();
                }
            }
        }

        private bool Apply(double[] lower, double[] upper)
        {
            evaluations += rule.Points;
            if (!rule.Apply(ref f, lower, upper, out var estimate))
            {
                return false;
            }

            var region = new Region(lower, upper, estimate.Value, estimate.Error, SplitAxis(lower, upper, rule.AxisScores));
            value.Add(region.Value);
            if (estimate.AtRoundoff)
            {
                frozen.Add(region);
                frozenError.Add(region.Error);
            }
            else
            {
                heap.Enqueue(region, -region.Error);
                heapError.Add(region.Error);
            }

            return true;
        }

        // 軸スコアが最大の軸を選ぶ（同程度なら幅が最大の軸）。分割可能な軸がなければ -1。
        private static int SplitAxis(double[] lower, double[] upper, ReadOnlySpan<double> differences)
        {
            int best = -1;
            double bestDifference = 0, bestWidth = 0;
            for (int i = 0; i < lower.Length; i++)
            {
                double t0 = lower[i], t1 = upper[i];
                double middle = 0.5 * t0 + 0.5 * t1;
                double width = t1 - t0;
                if (!(middle > t0 && middle < t1) || width <= RelativeMinimumWidth * Math.Max(Math.Abs(t0), Math.Abs(t1)) || width <= AbsoluteMinimumWidth)
                {
                    continue;
                }

                double difference = differences[i];
                if (best < 0 || difference > bestDifference * (1 + DifferenceTie)
                    || (difference >= bestDifference * (1 - DifferenceTie) && width > bestWidth))
                {
                    best = i;
                    bestDifference = difference;
                    bestWidth = width;
                }
            }

            return best;
        }

        private void Recompute()
        {
            value = default;
            heapError = default;
            frozenError = default;
            foreach (var (region, _) in heap.UnorderedItems)
            {
                value.Add(region.Value);
                heapError.Add(region.Error);
            }

            foreach (var region in frozen)
            {
                value.Add(region.Value);
                frozenError.Add(region.Error);
            }
        }

        private IntegrationResult Result(IntegrationTermination termination)
        {
            Recompute();
            double total = value.Value;
            double error = heapError.Value + frozenError.Value;
            if (!double.IsFinite(total) || !double.IsFinite(error))
            {
                return NonFinite();
            }

            return new IntegrationResult(total, error, evaluations, termination);
        }

        private IntegrationResult NonFinite() => new(double.NaN, double.NaN, evaluations, IntegrationTermination.NonFiniteValue);
    }
}
