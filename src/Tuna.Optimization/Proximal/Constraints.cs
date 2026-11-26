using Tuna.LinearAlgebra;

namespace Tuna.Optimization.Proximal;

/// <summary>
/// Represents the indicator function of the non-negative orthant <c>{x | x ≥ 0}</c>.
/// <para>非負象限 <c>{x | x ≥ 0}</c> の指示関数を表します。</para>
/// </summary>
/// <remarks>
/// <para>Its proximal operator is componentwise projection <c>max(v, 0)</c>.</para>
/// <para>近接写像は成分ごとの射影 <c>max(v, 0)</c> です。</para>
/// </remarks>
public sealed class NonNegativeConstraint : IProximalOperator
{
    /// <inheritdoc/>
    public void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)
    {
        ProximalChecks.Check(v, step, destination);
        ProximalKernels.PositivePart(v, 0, 0, destination);
    }

    /// <inheritdoc/>
    public double Evaluate(ReadOnlyVectorSpan<double> x)
    {
        for (int i = 0; i < x.Length; i++)
        {
            if (!(x[i] >= 0))
            {
                return double.PositiveInfinity;
            }
        }

        return 0;
    }
}

/// <summary>
/// Represents the indicator function of a box constraint <c>{x | l ≤ x ≤ u}</c>.
/// <para>ボックス制約 <c>{x | l ≤ x ≤ u}</c> の指示関数を表します。</para>
/// </summary>
/// <remarks>
/// <para>Its proximal operator is componentwise clamping <c>min(max(v, l), u)</c>.</para>
/// <para>近接写像は成分ごとの射影 <c>min(max(v, l), u)</c> です。</para>
/// </remarks>
public sealed class BoxConstraint : IProximalOperator
{
    private readonly double[] lower;
    private readonly double[] upper;

    /// <summary>
    /// Initializes a box constraint from componentwise lower and upper bounds.
    /// <para>成分ごとの下限および上限からボックス制約を初期化します。</para>
    /// </summary>
    /// <param name="lowerBounds">Lower bounds for each variable (may include <see cref="double.NegativeInfinity"/>). Pass an empty span for no lower bounds. / 各成分の下限（<see cref="double.NegativeInfinity"/> も指定可能）。空の場合は全成分が −∞。</param>
    /// <param name="upperBounds">Upper bounds for each variable (may include <see cref="double.PositiveInfinity"/>, at least the corresponding lower bound). Pass an empty span for no upper bounds. / 各成分の上限（<see cref="double.PositiveInfinity"/> も指定可能、対応する下限以上）。空の場合は全成分が +∞。</param>
    /// <exception cref="ArgumentException">Both spans are empty, their lengths differ, a bound is <see cref="double.NaN"/>, or a lower bound exceeds an upper bound. / 両方が空、長さが異なる、<see cref="double.NaN"/> を含む、または下限が上限を超えている場合。</exception>
    public BoxConstraint(ReadOnlySpan<double> lowerBounds, ReadOnlySpan<double> upperBounds)
    {
        if (lowerBounds.IsEmpty && upperBounds.IsEmpty)
        {
            throw new ArgumentException("At least one of the bounds must be given.", nameof(lowerBounds));
        }

        int n = Math.Max(lowerBounds.Length, upperBounds.Length);
        Bounds.Validate(lowerBounds, upperBounds, n, allowEmpty: false, out double[]? lo, out double[]? hi);
        lower = lo!;
        upper = hi!;
    }

    /// <summary>
    /// Gets the lower bounds for each component.
    /// <para>各成分の下限を取得します。</para>
    /// </summary>
    public ReadOnlySpan<double> LowerBounds => lower;

    /// <summary>
    /// Gets the upper bounds for each component.
    /// <para>各成分の上限を取得します。</para>
    /// </summary>
    public ReadOnlySpan<double> UpperBounds => upper;

    /// <inheritdoc/>
    public void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)
    {
        ProximalChecks.Check(v, step, destination, lower.Length);
        ProximalKernels.Clamp(v, lower, upper, destination);
    }

    /// <inheritdoc/>
    public double Evaluate(ReadOnlyVectorSpan<double> x)
    {
        ProximalChecks.Length(x.Length, lower.Length);
        for (int i = 0; i < x.Length; i++)
        {
            if (!(x[i] >= lower[i] && x[i] <= upper[i]))
            {
                return double.PositiveInfinity;
            }
        }

        return 0;
    }
}

/// <summary>
/// Represents the indicator function of the origin-centered L2 ball <c>{x | ‖x‖₂ ≤ r}</c>.
/// <para>原点を中心とする半径 <c>r</c> の L2 球 <c>{x | ‖x‖₂ ≤ r}</c> の指示関数を表します。</para>
/// </summary>
/// <remarks>
/// <para>Its proximal operator is radial projection <c>v · min(1, r / ‖v‖₂)</c>.</para>
/// <para>近接写像は L2 球への射影 <c>v · min(1, r / ‖v‖₂)</c> です。</para>
/// </remarks>
public sealed class L2BallConstraint : IProximalOperator
{
    /// <summary>
    /// Initializes an L2 ball constraint with the specified radius.
    /// <para>指定した半径の L2 球制約を初期化します。</para>
    /// </summary>
    /// <param name="radius">The ball radius <c>r</c> (non-negative and finite). / 半径 <c>r</c>（0 以上の有限値）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="radius"/> is negative or non-finite. / <paramref name="radius"/> が負または有限値でない場合。</exception>
    public L2BallConstraint(double radius)
    {
        ProximalChecks.NonNegative(radius, nameof(radius));
        Radius = radius;
    }

    /// <summary>
    /// Gets the radius <c>r</c> of the L2 ball.
    /// <para>L2 球の半径 <c>r</c> を取得します。</para>
    /// </summary>
    public double Radius { get; }

    /// <inheritdoc/>
    public void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)
    {
        ProximalChecks.Check(v, step, destination);
        double norm = ProximalKernels.Norm2(v);
        ProximalKernels.Scale(v, norm > Radius ? Radius / norm : 1.0, destination);
    }

    /// <inheritdoc/>
    public double Evaluate(ReadOnlyVectorSpan<double> x) =>
        ProximalKernels.Norm2(x) <= Radius * (1 + ProximalChecks.FeasibilityTolerance) ? 0 : double.PositiveInfinity;
}

/// <summary>
/// Represents the indicator function of the L1 ball <c>{x | ‖x‖₁ ≤ r}</c>.
/// <para>L1 球 <c>{x | ‖x‖₁ ≤ r}</c> の指示関数を表します。</para>
/// </summary>
/// <remarks>
/// <para>Its proximal operator projects onto the L1 ball in expected O(n) time via Condat's simplex threshold algorithm. Instances reuse an internal scratch buffer and are not thread-safe.</para>
/// <para>近接写像は Condat の単体射影アルゴリズムにより期待計算量 O(n) で L1 球へ射影します（球外では軟判定しきい値処理を適用します）。内部の作業バッファを再利用するためスレッドセーフではありません。</para>
/// </remarks>
public sealed class L1BallConstraint : IProximalOperator
{
    private double[]? buffer;

    /// <summary>
    /// Initializes an L1 ball constraint with the specified radius.
    /// <para>指定した半径の L1 球制約を初期化します。</para>
    /// </summary>
    /// <param name="radius">The ball radius <c>r</c> (positive and finite). / 半径 <c>r</c>（正の有限値）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="radius"/> is not positive and finite. / <paramref name="radius"/> が正の有限値でない場合。</exception>
    public L1BallConstraint(double radius)
    {
        if (!(radius > 0 && double.IsFinite(radius)))
        {
            throw new ArgumentOutOfRangeException(nameof(radius), "The radius must be positive and finite.");
        }

        Radius = radius;
    }

    /// <summary>
    /// Gets the radius <c>r</c> of the L1 ball.
    /// <para>L1 球の半径 <c>r</c> を取得します。</para>
    /// </summary>
    public double Radius { get; }

    /// <inheritdoc/>
    public void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)
    {
        ProximalChecks.Check(v, step, destination);
        if (v.Length == 0 || ProximalKernels.SumAbs(v) <= Radius)
        {
            ProximalKernels.Scale(v, 1.0, destination);
            return;
        }

        // Compute τ relative to max(|v|) so large coordinates do not wash out Radius; clamp τ ≥ 0.
        double reference = ProximalKernels.SimplexReference(v, absolute: true);
        double tau = ProximalKernels.SimplexThreshold(v, reference, Radius, absolute: true, ref buffer);
        ProximalKernels.ShiftedSoftThreshold(v, reference, Math.Max(tau, -reference), destination);
    }

    /// <inheritdoc/>
    public double Evaluate(ReadOnlyVectorSpan<double> x) =>
        ProximalKernels.SumAbs(x) <= Radius * (1 + ProximalChecks.FeasibilityTolerance) ? 0 : double.PositiveInfinity;
}

/// <summary>
/// Represents the indicator function of the simplex <c>{x | x ≥ 0, Σx = s}</c>.
/// <para>単体 <c>{x | x ≥ 0, Σx = s}</c> の指示関数を表します。</para>
/// </summary>
/// <remarks>
/// <para>Its proximal operator projects onto the simplex <c>max(v − τ, 0)</c> in expected O(n) time via Condat's algorithm. Instances reuse an internal scratch buffer and are not thread-safe.</para>
/// <para>近接写像は Condat のアルゴリズムにより期待計算量 O(n) で単体への射影 <c>max(v − τ, 0)</c> を計算します。内部の作業バッファを再利用するためスレッドセーフではありません。</para>
/// </remarks>
public sealed class SimplexConstraint : IProximalOperator
{
    private double[]? buffer;

    /// <summary>
    /// Initializes a simplex constraint with the specified coordinate sum.
    /// <para>指定した総和の単体制約を初期化します。</para>
    /// </summary>
    /// <param name="total">The target sum <c>s</c> of coordinates (positive and finite; default is 1). / 成分の総和 <c>s</c>（正の有限値、既定値は 1）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="total"/> is not positive and finite. / <paramref name="total"/> が正の有限値でない場合。</exception>
    public SimplexConstraint(double total = 1)
    {
        if (!(total > 0 && double.IsFinite(total)))
        {
            throw new ArgumentOutOfRangeException(nameof(total), "The total must be positive and finite.");
        }

        Total = total;
    }

    /// <summary>
    /// Gets the target sum <c>s</c> of the simplex coordinates.
    /// <para>単体の成分総和 <c>s</c> を取得します。</para>
    /// </summary>
    public double Total { get; }

    /// <exception cref="ArgumentException"><paramref name="v"/> is empty. / <paramref name="v"/> が空の場合。</exception>
    /// <inheritdoc/>
    public void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)
    {
        ProximalChecks.Check(v, step, destination);
        if (v.Length == 0)
        {
            throw new ArgumentException("The simplex needs at least one component.", nameof(v));
        }

        // Compute τ relative to max(v) so large coordinates do not wash out Total.
        double reference = ProximalKernels.SimplexReference(v, absolute: false);
        double tau = ProximalKernels.SimplexThreshold(v, reference, Total, absolute: false, ref buffer);
        ProximalKernels.PositivePart(v, reference, tau, destination);
    }

    /// <inheritdoc/>
    public double Evaluate(ReadOnlyVectorSpan<double> x)
    {
        double sum = 0;
        for (int i = 0; i < x.Length; i++)
        {
            if (!(x[i] >= 0))
            {
                return double.PositiveInfinity;
            }

            sum += x[i];
        }

        return x.Length > 0 && Math.Abs(sum - Total) <= ProximalChecks.FeasibilityTolerance * Math.Max(Total, sum) ? 0 : double.PositiveInfinity;
    }
}
