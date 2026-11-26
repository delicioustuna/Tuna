using Tuna.LinearAlgebra;
using Tuna.Optimization.Gradient;

namespace Tuna.Optimization.Global;

/// <summary>
/// Configures options for <see cref="BranchAndBound"/>.
/// <para><see cref="BranchAndBound"/> の設定オプションを表します。</para>
/// </summary>
/// <remarks>
/// <para>Option values are validated by the <see cref="BranchAndBound"/> constructor.</para>
/// <para>各設定値の妥当性は <see cref="BranchAndBound"/> のコンストラクターで検証されます。</para>
/// </remarks>
public sealed class BranchAndBoundOptions
{
    /// <summary>
    /// Gets the absolute optimality gap tolerance between upper and lower bounds.
    /// <para>上界と下界の差に対する絶対許容誤差を取得します。</para>
    /// </summary>
    /// <value>A non-negative finite number. The default is <c>1e-6</c>. Combined with <see cref="RelativeTolerance"/> via <c>max(AbsoluteTolerance, RelativeTolerance · |upper|)</c>. / 0 以上の有限値。既定値は <c>1e-6</c>。<c>max(AbsoluteTolerance, RelativeTolerance · |上界|)</c> で判定します。</value>
    public double AbsoluteTolerance { get; init; } = 1e-6;

    /// <summary>
    /// Gets the relative optimality gap tolerance scaled by <c>|upper|</c>.
    /// <para>上界の絶対値に対する相対許容誤差を取得します。</para>
    /// </summary>
    /// <value>A non-negative finite number. The default is <c>1e-6</c>. / 0 以上の有限値。既定値は <c>1e-6</c>。</value>
    public double RelativeTolerance { get; init; } = 1e-6;

    /// <summary>
    /// Gets the maximum number of boxes (branch-and-bound nodes) that may be split.
    /// <para>分割するボックス（ノード）数の上限を取得します。</para>
    /// </summary>
    /// <value>An integer greater than or equal to 1. The default is 100,000. / 1 以上の整数。既定値は 100,000。</value>
    public long MaxNodes { get; init; } = 100_000;

    /// <summary>
    /// Gets a value indicating whether McCormick affine underestimators are used alongside interval bounds.
    /// <para>下界計算に区間演算と併せて McCormick 緩和のアフィン下界を使用するかどうかを取得します。</para>
    /// </summary>
    /// <value><see langword="true"/> (the default) to take the tighter of interval and McCormick affine bounds; <see langword="false"/> to use interval arithmetic only. / 区間下界と McCormick アフィン下界の大きい方を使用する場合は <see langword="true"/>（既定値）、区間演算のみの場合は <see langword="false"/>。</value>
    public bool UseRelaxation { get; init; } = true;

    /// <summary>
    /// Gets a value indicating whether <see cref="LbfgsB"/> local search (using <see cref="GradientDual"/>) is run from the root midpoint and from boxes that improve the upper bound.
    /// <para>初期ボックスの中点および上界を更新したボックスの中点から <see cref="LbfgsB"/> による局所最小化（勾配は <see cref="GradientDual"/>）を実行するかどうかを取得します。</para>
    /// </summary>
    /// <value><see langword="true"/> (the default) to enable local search; otherwise, <see langword="false"/>. / 局所探索を有効にする場合は <see langword="true"/>（既定値）、無効にする場合は <see langword="false"/>。</value>
    public bool LocalSearch { get; init; } = true;

    /// <summary>
    /// Gets optional <see cref="LbfgsBOptions"/> for local search, or <see langword="null"/> for defaults.
    /// <para>局所探索に使用する <see cref="LbfgsBOptions"/> を取得します（<see langword="null"/> の場合は既定値）。</para>
    /// </summary>
    public LbfgsBOptions? LocalSearchOptions { get; init; }

    /// <summary>
    /// Gets the minimum relative box width (relative to the initial box) below which a box is no longer split.
    /// <para>初期ボックス幅に対する相対幅がこの値以下になったボックスを分割対象外とするしきい値を取得します。</para>
    /// </summary>
    /// <value>A value in <c>[0, 1)</c>. The default is <c>1e-12</c>. Unsplit boxes are set aside and their lower bounds remain reflected in <see cref="BranchAndBoundResult.LowerBound"/>. / <c>[0, 1)</c> の値。既定値は <c>1e-12</c>。分割されないボックスは脇に置かれ、その下界は <see cref="BranchAndBoundResult.LowerBound"/> に反映されます。</value>
    public double MinimumRelativeWidth { get; init; } = 1e-12;
}

/// <summary>
/// Specifies the reason <see cref="BranchAndBound"/> terminated.
/// <para><see cref="BranchAndBound"/> の終了理由を表します。</para>
/// </summary>
public enum BranchAndBoundTermination
{
    /// <summary>
    /// The gap between upper and lower bounds fell within tolerance (global minimum certified within tolerance).
    /// <para>上界と下界の差が許容誤差以下になりました（許容誤差内で大域的最小値が保証されました）。</para>
    /// </summary>
    GapTolerance,

    /// <summary>
    /// The maximum number of split boxes (<see cref="BranchAndBoundOptions.MaxNodes"/>) was reached.
    /// <para>分割したボックス数が上限（<see cref="BranchAndBoundOptions.MaxNodes"/>）に達しました。</para>
    /// </summary>
    MaxNodes,

    /// <summary>
    /// All remaining boxes became narrower than <see cref="BranchAndBoundOptions.MinimumRelativeWidth"/> before the gap closed.
    /// <para>残存するすべてのボックスが分割下限幅未満となり、ギャップが許容誤差まで縮まりませんでした。</para>
    /// </summary>
    MinimumWidth,

    /// <summary>
    /// The function was undefined everywhere on the box or no finite objective value was found.
    /// <para>ボックス全体で関数が未定義（空区間）であったか、有限値を返す点が見つかりませんでした。</para>
    /// </summary>
    NoFeasiblePoint,
}

/// <summary>
/// Represents the result of global minimization by <see cref="BranchAndBound"/>.
/// <para><see cref="BranchAndBound"/> による大域的最小化の実行結果を表します。</para>
/// </summary>
/// <param name="Point">A copy of the best feasible point found (or the initial box midpoint if none was found). / 見つかった最良点のコピー（見つからなかった場合は初期ボックスの中点）。</param>
/// <param name="Value">The objective value (upper bound) at <paramref name="Point"/>, or <see cref="double.NaN"/> if no finite point was found. / <paramref name="Point"/> における目的関数値（上界）。有限点が見つからなかった場合は <see cref="double.NaN"/>。</param>
/// <param name="LowerBound">A certified lower bound on the global minimum over the box (minimum over remaining, set-aside, and pruned boxes). / ボックス上の最小値に対する下界（残存・保留・枝刈りした全ボックスの下界の最小値）。</param>
/// <param name="Nodes">The number of split boxes. / 分割したボックス数。</param>
/// <param name="LocalSearches">The number of local searches performed. / 実行した局所探索の回数。</param>
/// <param name="Termination">The reason the search terminated. / 終了理由。</param>
public sealed record BranchAndBoundResult(
    Vector<double> Point, double Value, double LowerBound, long Nodes, int LocalSearches, BranchAndBoundTermination Termination)
{
    /// <summary>
    /// Gets the optimality gap <c>Value − LowerBound</c>.
    /// <para>上界と下界のギャップ <c>Value − LowerBound</c> を取得します。</para>
    /// </summary>
    public double Gap => Value - LowerBound;

    /// <summary>
    /// Gets a value indicating whether the global minimum was certified within tolerance.
    /// <para>許容誤差内で大域的最適性が保証されて終了したかどうかを取得します。</para>
    /// </summary>
    /// <value><see langword="true"/> if <see cref="Termination"/> is <see cref="BranchAndBoundTermination.GapTolerance"/>; otherwise, <see langword="false"/>. / <see cref="Termination"/> が <see cref="BranchAndBoundTermination.GapTolerance"/> の場合は <see langword="true"/>、それ以外は <see langword="false"/>。</value>
    public bool Converged => Termination == BranchAndBoundTermination.GapTolerance;
}

/// <summary>
/// Globally minimizes a factorable function over a finite box using spatial branch-and-bound with interval arithmetic and McCormick relaxations.
/// <para>区間演算と McCormick 緩和を用いた空間分枝限定法により、有限ボックス上で分解可能な関数を大域的に最小化します。</para>
/// </summary>
/// <remarks>
/// <para>Explores points within the box where the function is defined. Interval bounds use outwardly rounded arithmetic, while McCormick affine lower bounds use standard floating-point arithmetic. Instances are not thread-safe and must not be called concurrently.</para>
/// <para>ボックス内の関数定義域を探索します。区間下界は外向き丸めで計算し、McCormick 緩和のアフィン下界は通常の浮動小数点丸めで計算します。インスタンスはスレッドセーフではないため、複数スレッドから同時に呼び出さないでください。</para>
/// </remarks>
public sealed class BranchAndBound
{
    private readonly int n;
    private readonly double[] lower;
    private readonly double[] upper;
    private readonly double absoluteTolerance;
    private readonly double relativeTolerance;
    private readonly long maxNodes;
    private readonly bool useRelaxation;
    private readonly bool localSearch;
    private readonly double minimumWidth;
    private readonly LbfgsB? local;

    /// <summary>
    /// Initializes a new spatial branch-and-bound solver over the box <c>[lowerBounds, upperBounds]</c>.
    /// <para>ボックス <c>[lowerBounds, upperBounds]</c> 上の空間分枝限定法ソルバーを初期化します。</para>
    /// </summary>
    /// <param name="lowerBounds">Finite lower bounds for each variable (non-empty). / 各変数の有限な下限（空でないこと）。</param>
    /// <param name="upperBounds">Finite upper bounds for each variable (at least the corresponding lower bound). / 各変数の有限な上限（対応する下限以上）。</param>
    /// <param name="options">Optional solver configuration, or <see langword="null"/> for defaults. / オプション設定。<see langword="null"/> の場合は既定値を使用します。</param>
    /// <exception cref="ArgumentException">The bounds are empty, differ in length, contain non-finite values, or have a lower bound exceeding an upper bound. / 範囲が空または長さ不一致、非有限値を含む、あるいは下限が上限を超えている場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option value in <paramref name="options"/> is out of range. / <paramref name="options"/> の設定値が有効範囲外の場合。</exception>
    public BranchAndBound(ReadOnlySpan<double> lowerBounds, ReadOnlySpan<double> upperBounds, BranchAndBoundOptions? options = null)
    {
        if (lowerBounds.IsEmpty || lowerBounds.Length != upperBounds.Length)
        {
            throw new ArgumentException("The bounds must be non-empty and have the same length.", nameof(lowerBounds));
        }

        for (int i = 0; i < lowerBounds.Length; i++)
        {
            if (!double.IsFinite(lowerBounds[i]) || !double.IsFinite(upperBounds[i]) || lowerBounds[i] > upperBounds[i])
            {
                throw new ArgumentException($"Bound {i} must be finite with the lower bound not above the upper bound.", nameof(lowerBounds));
            }
        }

        options ??= new BranchAndBoundOptions();
        if (!(options.AbsoluteTolerance >= 0 && double.IsFinite(options.AbsoluteTolerance)))
        {
            throw new ArgumentOutOfRangeException(nameof(options.AbsoluteTolerance), "The tolerance must be non-negative and finite.");
        }

        if (!(options.RelativeTolerance >= 0 && double.IsFinite(options.RelativeTolerance)))
        {
            throw new ArgumentOutOfRangeException(nameof(options.RelativeTolerance), "The tolerance must be non-negative and finite.");
        }

        ArgumentOutOfRangeException.ThrowIfLessThan(options.MaxNodes, 1L, nameof(options.MaxNodes));
        if (!(options.MinimumRelativeWidth >= 0 && options.MinimumRelativeWidth < 1))
        {
            throw new ArgumentOutOfRangeException(nameof(options.MinimumRelativeWidth), "The minimum relative width must lie in [0, 1).");
        }

        n = lowerBounds.Length;
        lower = lowerBounds.ToArray();
        upper = upperBounds.ToArray();
        absoluteTolerance = options.AbsoluteTolerance;
        relativeTolerance = options.RelativeTolerance;
        maxNodes = options.MaxNodes;
        useRelaxation = options.UseRelaxation;
        localSearch = options.LocalSearch;
        minimumWidth = options.MinimumRelativeWidth;
        local = localSearch ? new LbfgsB(lower, upper, options.LocalSearchOptions) : null;
    }

    /// <summary>
    /// Gets the number of variables.
    /// <para>変数の数を取得します。</para>
    /// </summary>
    public int Dimension => n;

    /// <summary>
    /// Globally minimizes the specified factorable function over the configured box.
    /// <para>指定した分解可能な関数を設定ボックス内で大域的に最小化します。</para>
    /// </summary>
    /// <typeparam name="TFunction">The factorable function type. / 分解可能関数の型。</typeparam>
    /// <param name="function">The factorable objective function (evaluated using <see cref="Interval"/>, <see cref="McCormick"/>, and <see cref="GradientDual"/>). / 分解可能な目的関数（<see cref="Interval"/>・<see cref="McCormick"/>・<see cref="GradientDual"/> で評価されます）。</param>
    /// <param name="cancellationToken">A cancellation token checked before each box split. / 各ボックス分割の前に確認されるキャンセルトークン。</param>
    /// <returns>The global minimization result containing the best point, upper and lower bounds, node and local search counts, and termination reason. / 最良点、上界、下界、ノード数、局所探索回数、および終了理由を含む大域的最小化結果。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="function"/> is <see langword="null"/>. / <paramref name="function"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was canceled. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public BranchAndBoundResult Minimize<TFunction>(TFunction function, CancellationToken cancellationToken = default)
        where TFunction : IFactorableFunction
    {
        ArgumentNullException.ThrowIfNull(function);
        return new Search<TFunction>(this, function, cancellationToken).Run();
    }

    private sealed class Box(double[] lower, double[] upper)
    {
        public double[] Lower { get; } = lower;

        public double[] Upper { get; } = upper;
    }

    private sealed class Search<TFunction> where TFunction : IFactorableFunction
    {
        private readonly BranchAndBound owner;
        private readonly TFunction function;
        private readonly CancellationToken cancellationToken;
        private readonly int n;
        private readonly double[][] units;
        private readonly Interval[] intervals;
        private readonly McCormick[] relaxations;
        private readonly GradientDual[] constants;
        private readonly double[] point;
        private readonly double[] corner;
        private readonly double[] best;
        private readonly FactorableObjective<TFunction>? objective;
        private double incumbent = double.PositiveInfinity;
        private int localSearches;

        public Search(BranchAndBound owner, TFunction function, CancellationToken cancellationToken)
        {
            this.owner = owner;
            this.function = function;
            this.cancellationToken = cancellationToken;
            n = owner.n;
            units = TangentVectors.Units(n);
            intervals = new Interval[n];
            relaxations = new McCormick[n];
            constants = new GradientDual[n];
            point = new double[n];
            corner = new double[n];
            best = new double[n];
            objective = owner.localSearch ? new FactorableObjective<TFunction>(function, n) : null;
        }

        public BranchAndBoundResult Run()
        {
            var root = new Box((double[])owner.lower.Clone(), (double[])owner.upper.Clone());
            Center(root, best);
            double rootBound = Bound(root, double.NegativeInfinity, out bool empty);
            if (empty)
            {
                return Result(double.PositiveInfinity, 0, BranchAndBoundTermination.NoFeasiblePoint);
            }

            Probe(root, alwaysSearch: true);
            var queue = new PriorityQueue<Box, double>();
            queue.Enqueue(root, rootBound);
            double discarded = double.PositiveInfinity, setAside = double.PositiveInfinity;
            long nodes = 0;
            var termination = BranchAndBoundTermination.GapTolerance;
            while (queue.TryPeek(out Box? box, out double bound))
            {
                if (bound >= Threshold())
                {
                    break;
                }

                if (nodes >= owner.maxNodes)
                {
                    termination = BranchAndBoundTermination.MaxNodes;
                    break;
                }

                cancellationToken.ThrowIfCancellationRequested();
                queue.Dequeue();
                nodes++;
                int k = SplitDimension(box);
                double middle = k < 0 ? double.NaN : Interval.MidpointOf(box.Lower[k], box.Upper[k]);
                if (k < 0 || !(middle > box.Lower[k] && middle < box.Upper[k]))
                {
                    setAside = Math.Min(setAside, bound);
                    continue;
                }

                var left = new Box((double[])box.Lower.Clone(), (double[])box.Upper.Clone());
                left.Upper[k] = middle;
                var right = new Box(box.Lower, box.Upper);
                right.Lower[k] = middle;
                foreach (Box child in (ReadOnlySpan<Box>)[left, right])
                {
                    double childBound = Bound(child, bound, out bool childEmpty);
                    if (childEmpty)
                    {
                        continue;
                    }

                    Probe(child, alwaysSearch: false);
                    if (childBound >= Threshold())
                    {
                        discarded = Math.Min(discarded, childBound);
                    }
                    else
                    {
                        queue.Enqueue(child, childBound);
                    }
                }
            }

            double lowerBound = Math.Min(discarded, setAside);
            if (queue.TryPeek(out _, out double remaining))
            {
                lowerBound = Math.Min(lowerBound, remaining);
            }

            if (termination == BranchAndBoundTermination.GapTolerance)
            {
                if (double.IsPositiveInfinity(incumbent))
                {
                    termination = BranchAndBoundTermination.NoFeasiblePoint;
                }
                else if (setAside < Threshold())
                {
                    termination = BranchAndBoundTermination.MinimumWidth;
                }
            }

            return Result(Math.Min(lowerBound, incumbent), nodes, termination);
        }

        private BranchAndBoundResult Result(double lowerBound, long nodes, BranchAndBoundTermination termination) =>
            new(new Vector<double>(best), double.IsPositiveInfinity(incumbent) ? double.NaN : incumbent, lowerBound, nodes, localSearches, termination);

        private double Threshold() =>
            double.IsPositiveInfinity(incumbent)
                ? double.PositiveInfinity
                : incumbent - Math.Max(owner.absoluteTolerance, owner.relativeTolerance * Math.Abs(incumbent));

        private int SplitDimension(Box box)
        {
            int k = -1;
            double widest = owner.minimumWidth;
            for (int i = 0; i < n; i++)
            {
                double full = owner.upper[i] - owner.lower[i];
                if (full > 0)
                {
                    double relative = (box.Upper[i] - box.Lower[i]) / full;
                    if (relative > widest)
                    {
                        widest = relative;
                        k = i;
                    }
                }
            }

            return k;
        }

        private static void Center(Box box, double[] destination)
        {
            for (int i = 0; i < destination.Length; i++)
            {
                destination[i] = Interval.MidpointOf(box.Lower[i], box.Upper[i]);
            }
        }

        private double Bound(Box box, double parent, out bool empty)
        {
            for (int i = 0; i < n; i++)
            {
                intervals[i] = new Interval(box.Lower[i], box.Upper[i]);
            }

            double bound;
            if (!owner.useRelaxation)
            {
                Interval range = function.Evaluate<Interval>(intervals);
                empty = range.IsEmpty;
                bound = range.Lower;
            }
            else
            {
                Center(box, point);
                McCormick m = Relax(point);
                empty = m.Range.IsEmpty;
                bound = m.Range.Lower;
                double affine = AffineMinimum(m, box, point, corner);
                if (affine > double.NegativeInfinity)
                {
                    bound = Math.Max(bound, affine);
                    for (int i = 0; i < n; i++)
                    {
                        point[i] = Interval.MidpointOf(point[i], corner[i]);
                    }

                    bound = Math.Max(bound, AffineMinimum(Relax(point), box, point, corner));
                }
            }

            return empty ? double.PositiveInfinity : Math.Max(bound, parent);
        }

        private McCormick Relax(double[] reference)
        {
            for (int i = 0; i < n; i++)
            {
                relaxations[i] = McCormick.Variable(intervals[i], reference[i], units[i]);
            }

            return function.Evaluate<McCormick>(relaxations);
        }

        // Minimum of cv(p) + sᵀ(x − p) over box; writes achieving corner to minimizer.
        private double AffineMinimum(McCormick m, Box box, double[] reference, double[] minimizer)
        {
            double value = m.Convex;
            if (m.Range.IsEmpty || !double.IsFinite(value))
            {
                return double.NegativeInfinity;
            }

            ReadOnlySpan<double> s = m.ConvexSubgradient;
            for (int i = 0; i < n; i++)
            {
                double si = s.IsEmpty ? 0 : s[i];
                if (si > 0)
                {
                    value += si * (box.Lower[i] - reference[i]);
                    minimizer[i] = box.Lower[i];
                }
                else if (si < 0)
                {
                    value += si * (box.Upper[i] - reference[i]);
                    minimizer[i] = box.Upper[i];
                }
                else if (si == 0)
                {
                    minimizer[i] = reference[i];
                }
                else
                {
                    return double.NegativeInfinity;
                }
            }

            return double.IsNaN(value) ? double.NegativeInfinity : value;
        }

        private void Probe(Box box, bool alwaysSearch)
        {
            Center(box, point);
            for (int i = 0; i < n; i++)
            {
                constants[i] = point[i];
            }

            double value = function.Evaluate<GradientDual>(constants).Value;
            if (!double.IsFinite(value))
            {
                return;
            }

            bool improved = value < incumbent;
            if (improved)
            {
                incumbent = value;
                point.CopyTo(best, 0);
            }

            if ((improved || alwaysSearch) && objective is not null)
            {
                localSearches++;
                MinimizationResult result = owner.local!.Minimize(objective, point, cancellationToken);
                if (double.IsFinite(result.Value) && result.Value < incumbent)
                {
                    incumbent = result.Value;
                    result.Point.AsSpan().CopyTo(best);
                }
            }
        }
    }
}
