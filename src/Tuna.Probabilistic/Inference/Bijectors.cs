using System.Buffers;
using Tuna.LinearAlgebra;

namespace Tuna.Probabilistic.Inference;

/// <summary>
/// Represents a smooth, invertible transformation between an unconstrained space <c>ℝⁿ</c> and a constrained parameter space.
/// <para>制約のない空間 <c>ℝⁿ</c> と制約付きパラメータ空間の間の滑らかな可逆写像（制約変換）を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Samplers operate in unconstrained space <c>y ∈ ℝⁿ</c> and map points to constrained space via <c>x = f(y)</c>.
/// To transform a constrained log-density <c>ln p(x)</c>, the log absolute Jacobian determinant is added: <c>ln p(f(y)) + ln|det J_f(y)|</c> (via <see cref="TransformedLogDensity{TDensity, TBijector}"/>; transformations follow the Stan Reference Manual conventions).
/// </para>
/// <para>
/// For transforms whose constrained dimension exceeds the unconstrained dimension (such as a <c>K</c>-component simplex summing to 1), the density and Jacobian are defined with respect to the leading <c>K − 1</c> coordinates.
/// </para>
/// <para>
/// サンプラーは制約のない空間 <c>y ∈ ℝⁿ</c> 上で動作し、<c>x = f(y)</c> により制約付き空間へ写像します。
/// 制約付き空間の対数密度 <c>ln p(x)</c> を変換する際は、ヤコビアン行列式の絶対値の自然対数を加えて <c>ln p(f(y)) + ln|det J_f(y)|</c> とします（<see cref="TransformedLogDensity{TDensity, TBijector}"/>。各変換は Stan Reference Manual の定義に従います）。
/// </para>
/// <para>
/// 単体（和が 1 の <c>K</c> 成分）のように制約付き空間の次元が制約なし空間より大きい変換では、密度とヤコビアンは先頭の <c>K − 1</c> 成分に対して定義されます。
/// </para>
/// </remarks>
public interface IBijector
{
    /// <summary>
    /// Gets the dimension of the unconstrained space.
    /// <para>制約のない空間の次元を取得します。</para>
    /// </summary>
    int UnconstrainedDimension { get; }

    /// <summary>
    /// Gets the dimension (number of coordinates) of the constrained space.
    /// <para>制約付き空間の次元（成分数）を取得します。</para>
    /// </summary>
    int ConstrainedDimension { get; }

    /// <summary>
    /// Maps an unconstrained point to the constrained space and returns the log absolute Jacobian determinant.
    /// <para>制約のない点を制約付き空間へ写像し、ヤコビアン行列式の絶対値の自然対数を返します。</para>
    /// </summary>
    /// <param name="unconstrained">The unconstrained point <c>y</c>. / 制約のない点 <c>y</c>。</param>
    /// <param name="constrained">The destination for the constrained point <c>f(y)</c>. / 変換後の点 <c>f(y)</c> の書き込み先。</param>
    /// <returns>The log absolute Jacobian determinant <c>ln|det J_f(y)|</c>. / ヤコビアン行列式の絶対値の自然対数 <c>ln|det J_f(y)|</c>。</returns>
    double Forward(ReadOnlyVectorSpan<double> unconstrained, VectorSpan<double> constrained);

    /// <summary>
    /// Maps a constrained point back to the unconstrained space.
    /// <para>制約付きの点を制約のない空間へ逆写像します。</para>
    /// </summary>
    /// <param name="constrained">The constrained point <c>x</c>. / 逆変換する点 <c>x</c>。</param>
    /// <param name="unconstrained">The destination for <c>f⁻¹(x)</c>. Writes <c>±∞</c> on constraint boundaries and <see cref="double.NaN"/> outside the domain. / 逆像 <c>f⁻¹(x)</c> の書き込み先。境界上では <c>±∞</c>、定義域外では <see cref="double.NaN"/> を書き込みます。</param>
    void Inverse(ReadOnlyVectorSpan<double> constrained, VectorSpan<double> unconstrained);

    /// <summary>
    /// Pulls a constrained-space log-density gradient back to the unconstrained space, including the Jacobian gradient term.
    /// <para>制約付き空間の勾配にヤコビアン項の勾配を加えて制約のない空間へ引き戻します。</para>
    /// </summary>
    /// <param name="unconstrained">The unconstrained point <c>y</c>. / 変換前の点 <c>y</c>。</param>
    /// <param name="constrainedGradient">The gradient <c>∇ₓ ln p(x)</c> evaluated at <c>x = f(y)</c>. / 変換後の点における勾配 <c>∇ₓ ln p(x)</c>。</param>
    /// <param name="unconstrainedGradient">The destination for <c>J_f(y)ᵀ ∇ₓ ln p + ∇_y ln|det J_f(y)|</c>. / 変換後の勾配 <c>J_f(y)ᵀ ∇ₓ ln p + ∇_y ln|det J_f(y)|</c> の書き込み先。</param>
    void PullBack(ReadOnlyVectorSpan<double> unconstrained, ReadOnlyVectorSpan<double> constrainedGradient, VectorSpan<double> unconstrainedGradient);
}

/// <summary>
/// Constrains each coordinate independently to an open interval, half-line, or the entire real line.
/// <para>各成分を独立に開区間、半直線、または実数全体へ制約する変換です。</para>
/// </summary>
/// <remarks>
/// <para>
/// Bounded intervals use <c>x = a + (b − a)·σ(y)</c> with <c>ln|J| = ln(b − a) + ln σ(y) + ln σ(−y)</c>.
/// Lower-only bounds use <c>x = a + eʸ</c>, upper-only bounds use <c>x = b − eʸ</c> (both with <c>ln|J| = y</c>), and unbounded coordinates use the identity map.
/// Instances are immutable and thread-safe.
/// </para>
/// <para>
/// 両側有界区間では <c>x = a + (b − a)·σ(y)</c>、<c>ln|J| = ln(b − a) + ln σ(y) + ln σ(−y)</c> を用います。
/// 下限のみでは <c>x = a + eʸ</c>、上限のみでは <c>x = b − eʸ</c>（いずれも <c>ln|J| = y</c>）、上下限ともに無限大の成分は恒等写像となります。
/// インスタンスは不変であり、スレッドセーフです。
/// </para>
/// </remarks>
public sealed class BoxBijector : IBijector
{
    private readonly double[] lower;
    private readonly double[] upper;

    /// <summary>
    /// Initializes a new instance of <see cref="BoxBijector"/> with per-coordinate lower and upper bounds.
    /// <para>成分ごとの下限と上限を指定して <see cref="BoxBijector"/> の新しいインスタンスを初期化します。</para>
    /// </summary>
    /// <param name="lowerBounds">The per-coordinate lower bounds (<see cref="double.NegativeInfinity"/> for unbounded). / 成分ごとの下限（制約なしの場合は <see cref="double.NegativeInfinity"/>）。</param>
    /// <param name="upperBounds">The per-coordinate upper bounds (<see cref="double.PositiveInfinity"/> for unbounded). / 成分ごとの上限（制約なしの場合は <see cref="double.PositiveInfinity"/>）。</param>
    /// <exception cref="ArgumentException">The spans differ in length or are empty, contain <see cref="double.NaN"/>, violate <c>lower &lt; upper</c>, have <c>lower = +∞</c> or <c>upper = −∞</c>, or have a non-finite width <c>b − a</c> for finite bounds. / 長さが異なるか空の場合、<see cref="double.NaN"/> を含む場合、下限が上限以上の場合、下限が <c>+∞</c> または上限が <c>−∞</c> の場合、あるいは有限な上下限の差 <c>b − a</c> がオーバーフローする場合。</exception>
    public BoxBijector(ReadOnlySpan<double> lowerBounds, ReadOnlySpan<double> upperBounds)
    {
        if (lowerBounds.Length != upperBounds.Length || lowerBounds.IsEmpty)
        {
            throw new ArgumentException("The bounds must have the same positive length.", nameof(upperBounds));
        }

        for (int i = 0; i < lowerBounds.Length; i++)
        {
            double a = lowerBounds[i], b = upperBounds[i];
            if (double.IsNaN(a) || double.IsNaN(b) || !(a < b) || a == double.PositiveInfinity || b == double.NegativeInfinity)
            {
                throw new ArgumentException($"Bound {i} must satisfy lower < upper without NaN.", nameof(lowerBounds));
            }

            if (double.IsFinite(a) && double.IsFinite(b) && !double.IsFinite(b - a))
            {
                throw new ArgumentException($"The width of bound {i} must be finite.", nameof(upperBounds));
            }
        }

        lower = lowerBounds.ToArray();
        upper = upperBounds.ToArray();
    }

    /// <summary>
    /// Creates a <see cref="BoxBijector"/> that constrains all coordinates to be strictly positive <c>(0, +∞)</c>.
    /// <para>すべての成分を正の値 <c>(0, +∞)</c> に制約する <see cref="BoxBijector"/> を生成します。</para>
    /// </summary>
    /// <param name="dimension">The number of coordinates (1 or greater). / 成分数（1 以上）。</param>
    /// <returns>A <see cref="BoxBijector"/> with lower bound 0 and upper bound <see cref="double.PositiveInfinity"/>. / 下限 0・上限 <see cref="double.PositiveInfinity"/> の <see cref="BoxBijector"/>。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="dimension"/> is less than 1. / <paramref name="dimension"/> が 1 未満の場合。</exception>
    public static BoxBijector Positive(int dimension) => Uniform(dimension, 0, double.PositiveInfinity);

    /// <summary>
    /// Creates a <see cref="BoxBijector"/> that applies the same interval <c>(lowerBound, upperBound)</c> to all coordinates.
    /// <para>すべての成分に同一の区間 <c>(lowerBound, upperBound)</c> を適用する <see cref="BoxBijector"/> を生成します。</para>
    /// </summary>
    /// <param name="dimension">The number of coordinates (1 or greater). / 成分数（1 以上）。</param>
    /// <param name="lowerBound">The common lower bound. / 共通の下限。</param>
    /// <param name="upperBound">The common upper bound. / 共通の上限。</param>
    /// <returns>A <see cref="BoxBijector"/> with identical bounds for every coordinate. / 全成分が同じ上下限を持つ <see cref="BoxBijector"/>。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="dimension"/> is less than 1. / <paramref name="dimension"/> が 1 未満の場合。</exception>
    /// <exception cref="ArgumentException">The bounds are invalid. / 上下限の指定が不正な場合。</exception>
    public static BoxBijector Uniform(int dimension, double lowerBound, double upperBound)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(dimension, 1);
        var a = new double[dimension];
        var b = new double[dimension];
        Array.Fill(a, lowerBound);
        Array.Fill(b, upperBound);
        return new BoxBijector(a, b);
    }

    /// <inheritdoc/>
    public int UnconstrainedDimension => lower.Length;

    /// <inheritdoc/>
    public int ConstrainedDimension => lower.Length;

    /// <summary>
    /// Gets a copy of the per-coordinate lower bounds.
    /// <para>成分ごとの下限のコピーを取得します。</para>
    /// </summary>
    /// <value>A newly allocated array containing the lower bounds. / 下限値をコピーした新しい配列。</value>
    public double[] LowerBounds => (double[])lower.Clone();

    /// <summary>
    /// Gets a copy of the per-coordinate upper bounds.
    /// <para>成分ごとの上限のコピーを取得します。</para>
    /// </summary>
    /// <value>A newly allocated array containing the upper bounds. / 上限値をコピーした新しい配列。</value>
    public double[] UpperBounds => (double[])upper.Clone();

    /// <inheritdoc/>
    public double Forward(ReadOnlyVectorSpan<double> unconstrained, VectorSpan<double> constrained)
    {
        Bijection.RequireLengths(this, unconstrained.Length, constrained.Length);
        double logJacobian = 0;
        for (int i = 0; i < lower.Length; i++)
        {
            double y = unconstrained[i], a = lower[i], b = upper[i];
            bool hasLower = double.IsFinite(a), hasUpper = double.IsFinite(b);
            if (hasLower && hasUpper)
            {
                double width = b - a;

                // a + width·σ(y) では上限付近で 1 − σ(y) が桁落ちして b に丸められるため、y > 0 では上限からの差で計算する
                constrained[i] = y > 0 ? b - width * XMath.Logistic(-y) : a + width * XMath.Logistic(y);
                logJacobian += Math.Log(width) + XMath.LogLogistic(y) + XMath.LogLogistic(-y);
            }
            else if (hasLower)
            {
                constrained[i] = a + Math.Exp(y);
                logJacobian += y;
            }
            else if (hasUpper)
            {
                constrained[i] = b - Math.Exp(y);
                logJacobian += y;
            }
            else
            {
                constrained[i] = y;
            }
        }

        return logJacobian;
    }

    /// <inheritdoc/>
    public void Inverse(ReadOnlyVectorSpan<double> constrained, VectorSpan<double> unconstrained)
    {
        Bijection.RequireLengths(this, unconstrained.Length, constrained.Length);
        for (int i = 0; i < lower.Length; i++)
        {
            double x = constrained[i], a = lower[i], b = upper[i];
            bool hasLower = double.IsFinite(a), hasUpper = double.IsFinite(b);
            unconstrained[i] = (hasLower, hasUpper) switch
            {
                (true, true) => x < a || x > b ? double.NaN : Math.Log(x - a) - Math.Log(b - x),
                (true, false) => x < a ? double.NaN : Math.Log(x - a),
                (false, true) => x > b ? double.NaN : Math.Log(b - x),
                _ => x,
            };
        }
    }

    /// <inheritdoc/>
    public void PullBack(ReadOnlyVectorSpan<double> unconstrained, ReadOnlyVectorSpan<double> constrainedGradient, VectorSpan<double> unconstrainedGradient)
    {
        Bijection.RequireLengths(this, unconstrained.Length, constrainedGradient.Length);
        Bijection.RequireLengths(this, unconstrainedGradient.Length, constrainedGradient.Length);
        for (int i = 0; i < lower.Length; i++)
        {
            double y = unconstrained[i], g = constrainedGradient[i], a = lower[i], b = upper[i];
            bool hasLower = double.IsFinite(a), hasUpper = double.IsFinite(b);
            if (hasLower && hasUpper)
            {
                double p = XMath.Logistic(y), q = XMath.Logistic(-y);
                // dx/dy = (b − a)σ(y)σ(−y)、d ln|J|/dy = σ(−y) − σ(y)
                unconstrainedGradient[i] = (b - a) * p * q * g + (q - p);
            }
            else if (hasLower)
            {
                unconstrainedGradient[i] = Math.Exp(y) * g + 1;
            }
            else if (hasUpper)
            {
                unconstrainedGradient[i] = -Math.Exp(y) * g + 1;
            }
            else
            {
                unconstrainedGradient[i] = g;
            }
        }
    }
}

// 棒折り変換: zₖ = σ(yₖ − ln(K − k))、xₖ = zₖ·sₖ、sₖ₊₁ = sₖ(1 − zₖ)（s₁ = 1）、x_K = s_K、ln|J| = Σₖ [ln zₖ + ln(1 − zₖ) + ln sₖ]
/// <summary>
/// Maps <c>ℝ^(K−1)</c> onto the <c>K</c>-component probability simplex via stick-breaking.
/// <para>棒折り変換により <c>ℝ^(K−1)</c> を <c>K</c> 成分の確率単体へ写像します。</para>
/// </summary>
/// <remarks>
/// <para>The unconstrained dimension is <c>K − 1</c> and the constrained dimension is <c>K</c>. The origin <c>y = 0</c> maps to the center of the simplex <c>(1/K, …, 1/K)</c>.</para>
/// <para>制約のない空間の次元は <c>K − 1</c>、制約付き空間の次元は <c>K</c> です。原点 <c>y = 0</c> は単体の中心 <c>(1/K, …, 1/K)</c> に写像されます。</para>
/// </remarks>
public sealed class SimplexBijector : IBijector
{
    private readonly int size;

    /// <summary>
    /// Initializes a new instance of <see cref="SimplexBijector"/> for a simplex of the specified size.
    /// <para>指定した成分数の単体に対する <see cref="SimplexBijector"/> の新しいインスタンスを初期化します。</para>
    /// </summary>
    /// <param name="size">The number of simplex components (2 or greater). / 単体の成分数（2 以上）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="size"/> is less than 2. / <paramref name="size"/> が 2 未満の場合。</exception>
    public SimplexBijector(int size)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(size, 2);
        this.size = size;
    }

    /// <inheritdoc/>
    public int UnconstrainedDimension => size - 1;

    /// <inheritdoc/>
    public int ConstrainedDimension => size;

    /// <inheritdoc/>
    public double Forward(ReadOnlyVectorSpan<double> unconstrained, VectorSpan<double> constrained)
    {
        Bijection.RequireLengths(this, unconstrained.Length, constrained.Length);
        // 残りの棒を乗算で更新すると小さい成分がアンダーフローするため対数で累積する
        double logStick = 0, logJacobian = 0;
        for (int k = 0; k < size - 1; k++)
        {
            double u = unconstrained[k] - Math.Log(size - 1 - k);
            double logZ = XMath.LogLogistic(u), logOneMinusZ = XMath.LogLogistic(-u);
            constrained[k] = Math.Exp(logStick + logZ);
            logJacobian += logZ + logOneMinusZ + logStick;
            logStick += logOneMinusZ;
        }

        constrained[size - 1] = Math.Exp(logStick);
        return logJacobian;
    }

    /// <remarks>
    /// <para>All components must be non-negative and sum to 1 within <c>1e-8</c> (minor normalization drift is absorbed via ratios). Zero components map to <see cref="double.NegativeInfinity"/>; invalid inputs write <see cref="double.NaN"/>.</para>
    /// <para>全成分が 0 以上で和が 1 から <c>1e-8</c> 以内である必要があります（微小なずれは比で吸収します）。0 の成分は <see cref="double.NegativeInfinity"/> となり、負または <see cref="double.NaN"/> の成分を含む場合は <see cref="double.NaN"/> を書き込みます。</para>
    /// </remarks>
    /// <inheritdoc/>
    public void Inverse(ReadOnlyVectorSpan<double> constrained, VectorSpan<double> unconstrained)
    {
        Bijection.RequireLengths(this, unconstrained.Length, constrained.Length);
        double total = 0;
        bool valid = true;
        for (int k = 0; k < size; k++)
        {
            double x = constrained[k];
            valid &= x >= 0 && double.IsFinite(x);
            total += x;
        }

        if (!valid || !(Math.Abs(total - 1) <= 1e-8))
        {
            for (int k = 0; k < size - 1; k++)
            {
                unconstrained[k] = double.NaN;
            }

            return;
        }

        // yₖ = ln xₖ − ln Σⱼ₍ⱼ>ₖ₎ xⱼ + ln(K − k)。1 − Σⱼ₍ⱼ≤ₖ₎ xⱼ とすると桁落ちするため末尾から累積する
        double rest = constrained[size - 1];
        for (int k = size - 2; k >= 0; k--)
        {
            double x = constrained[k];
            unconstrained[k] = Math.Log(x) - Math.Log(rest) + Math.Log(size - 1 - k);
            rest += x;
        }
    }

    /// <inheritdoc/>
    public void PullBack(ReadOnlyVectorSpan<double> unconstrained, ReadOnlyVectorSpan<double> constrainedGradient, VectorSpan<double> unconstrainedGradient)
    {
        Bijection.RequireLengths(this, unconstrained.Length, constrainedGradient.Length);
        if (unconstrainedGradient.Length != size - 1)
        {
            throw new ArgumentException($"The unconstrained gradient must have length {size - 1}.", nameof(unconstrainedGradient));
        }

        // ∂xₖ/∂yⱼ = xₖ(1 − zₖ)（j = k）、−xₖzⱼ（j < k）。末尾からの和 Tⱼ = Σₖ₍ₖ>ⱼ₎ gₖxₖ を用いる
        double logStick = 0;
        Span<double> x = size <= 256 ? stackalloc double[size] : new double[size];
        Span<double> z = size <= 256 ? stackalloc double[size] : new double[size];
        for (int k = 0; k < size - 1; k++)
        {
            double u = unconstrained[k] - Math.Log(size - 1 - k);
            z[k] = XMath.Logistic(u);
            x[k] = Math.Exp(logStick + XMath.LogLogistic(u));
            logStick += XMath.LogLogistic(-u);
        }

        x[size - 1] = Math.Exp(logStick);
        double tail = constrainedGradient[size - 1] * x[size - 1];
        for (int j = size - 2; j >= 0; j--)
        {
            double oneMinusZ = XMath.Logistic(-(unconstrained[j] - Math.Log(size - 1 - j)));
            double fromDensity = x[j] * oneMinusZ * constrainedGradient[j] - z[j] * tail;
            // ln|J| の微分: ln zⱼ + ln(1 − zⱼ) から 1 − 2zⱼ、後続の K − 2 − j 個の ln sₖ から −zⱼ ずつ
            double fromJacobian = oneMinusZ - z[j] - (size - 2 - j) * z[j];
            unconstrainedGradient[j] = fromDensity + fromJacobian;
            tail += constrainedGradient[j] * x[j];
        }
    }
}

/// <summary>
/// Maps <c>ℝ^K</c> onto strictly increasing vectors <c>x₁ &lt; x₂ &lt; … &lt; x_K</c> via <c>x₁ = y₁</c>, <c>xₖ = xₖ₋₁ + e^{yₖ}</c>.
/// <para><c>x₁ = y₁</c>、<c>xₖ = xₖ₋₁ + e^{yₖ}</c> により <c>ℝ^K</c> を狭義単調増加ベクトル <c>x₁ &lt; x₂ &lt; … &lt; x_K</c> へ写像します。</para>
/// </summary>
/// <remarks>
/// <para>The log absolute Jacobian determinant is <c>ln|J| = Σₖ₌₂ yₖ</c>. Useful for ordered-logit cutpoints or mixture component means to prevent label switching.</para>
/// <para>ヤコビアン行列式の絶対値の自然対数は <c>ln|J| = Σₖ₌₂ yₖ</c> です。順序ロジットの閾値や、ラベルスイッチングを防ぐ混合分布の平均パラメータなどに利用できます。</para>
/// </remarks>
public sealed class OrderedBijector : IBijector
{
    private readonly int size;

    /// <summary>
    /// Initializes a new instance of <see cref="OrderedBijector"/> for a vector of the specified length.
    /// <para>指定した成分数の <see cref="OrderedBijector"/> の新しいインスタンスを初期化します。</para>
    /// </summary>
    /// <param name="size">The vector dimension (1 or greater). / ベクトルの成分数（1 以上）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="size"/> is less than 1. / <paramref name="size"/> が 1 未満の場合。</exception>
    public OrderedBijector(int size)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(size, 1);
        this.size = size;
    }

    /// <inheritdoc/>
    public int UnconstrainedDimension => size;

    /// <inheritdoc/>
    public int ConstrainedDimension => size;

    /// <inheritdoc/>
    public double Forward(ReadOnlyVectorSpan<double> unconstrained, VectorSpan<double> constrained)
    {
        Bijection.RequireLengths(this, unconstrained.Length, constrained.Length);
        double x = unconstrained[0], logJacobian = 0;
        constrained[0] = x;
        for (int k = 1; k < size; k++)
        {
            x += Math.Exp(unconstrained[k]);
            constrained[k] = x;
            logJacobian += unconstrained[k];
        }

        return logJacobian;
    }

    /// <remarks>
    /// <para>Adjacent differences equal to zero map to <see cref="double.NegativeInfinity"/>; negative or <see cref="double.NaN"/> differences map to <see cref="double.NaN"/>.</para>
    /// <para>隣接成分の差が 0 の場合は <see cref="double.NegativeInfinity"/>、負または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/> を書き込みます。</para>
    /// </remarks>
    /// <inheritdoc/>
    public void Inverse(ReadOnlyVectorSpan<double> constrained, VectorSpan<double> unconstrained)
    {
        Bijection.RequireLengths(this, unconstrained.Length, constrained.Length);
        unconstrained[0] = constrained[0];
        for (int k = 1; k < size; k++)
        {
            double gap = constrained[k] - constrained[k - 1];
            unconstrained[k] = gap < 0 ? double.NaN : Math.Log(gap);
        }
    }

    /// <inheritdoc/>
    public void PullBack(ReadOnlyVectorSpan<double> unconstrained, ReadOnlyVectorSpan<double> constrainedGradient, VectorSpan<double> unconstrainedGradient)
    {
        Bijection.RequireLengths(this, unconstrained.Length, constrainedGradient.Length);
        Bijection.RequireLengths(this, unconstrainedGradient.Length, constrainedGradient.Length);
        // xₖ は yⱼ（j ≤ k）に依存し、∂xₖ/∂y₁ = 1、∂xₖ/∂yⱼ = e^{yⱼ}（2 ≤ j ≤ k）のため末尾から累積する
        double tail = 0;
        for (int j = size - 1; j >= 0; j--)
        {
            tail += constrainedGradient[j];
            unconstrainedGradient[j] = j == 0 ? tail : Math.Exp(unconstrained[j]) * tail + 1;
        }
    }
}

/// <summary>
/// Combines multiple bijectors by applying each to a contiguous block of coordinates in both unconstrained and constrained spaces.
/// <para>複数の制約変換を、制約なし空間および制約付き空間の連続したブロックに順番に適用する複合変換です。</para>
/// </summary>
/// <remarks>
/// <para>For example, <c>new CompositeBijector(BoxBijector.Positive(2), new SimplexBijector(3))</c> constrains the first 2 coordinates to be positive and the next 3 to form a simplex (unconstrained dimension 4, constrained dimension 5).</para>
/// <para>例: <c>new CompositeBijector(BoxBijector.Positive(2), new SimplexBijector(3))</c> は、先頭 2 成分を正の値、続く 3 成分を確率単体に制約します（制約なし次元 4、制約付き次元 5）。</para>
/// </remarks>
public sealed class CompositeBijector : IBijector
{
    private readonly IBijector[] parts;

    /// <summary>
    /// Initializes a new instance of <see cref="CompositeBijector"/> from the specified component bijectors in order.
    /// <para>指定した制約変換を順番に並べて <see cref="CompositeBijector"/> の新しいインスタンスを初期化します。</para>
    /// </summary>
    /// <param name="parts">The component bijectors to concatenate. / 連結する制約変換の配列。</param>
    /// <exception cref="ArgumentNullException"><paramref name="parts"/> is <see langword="null"/>. / <paramref name="parts"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="parts"/> is empty or contains a <see langword="null"/> element. / <paramref name="parts"/> が空の場合、または <see langword="null"/> 要素を含む場合。</exception>
    public CompositeBijector(params IBijector[] parts)
    {
        ArgumentNullException.ThrowIfNull(parts);
        if (parts.Length == 0 || Array.IndexOf(parts, null) >= 0)
        {
            throw new ArgumentException("At least one non-null bijector is required.", nameof(parts));
        }

        this.parts = (IBijector[])parts.Clone();
        foreach (IBijector part in this.parts)
        {
            UnconstrainedDimension += part.UnconstrainedDimension;
            ConstrainedDimension += part.ConstrainedDimension;
        }
    }

    /// <inheritdoc/>
    public int UnconstrainedDimension { get; }

    /// <inheritdoc/>
    public int ConstrainedDimension { get; }

    /// <inheritdoc/>
    public double Forward(ReadOnlyVectorSpan<double> unconstrained, VectorSpan<double> constrained)
    {
        Bijection.RequireLengths(this, unconstrained.Length, constrained.Length);
        double logJacobian = 0;
        int u = 0, c = 0;
        foreach (IBijector part in parts)
        {
            int nu = part.UnconstrainedDimension, nc = part.ConstrainedDimension;
            logJacobian += part.Forward(unconstrained[u..(u + nu)], constrained[c..(c + nc)]);
            u += nu;
            c += nc;
        }

        return logJacobian;
    }

    /// <inheritdoc/>
    public void Inverse(ReadOnlyVectorSpan<double> constrained, VectorSpan<double> unconstrained)
    {
        Bijection.RequireLengths(this, unconstrained.Length, constrained.Length);
        int u = 0, c = 0;
        foreach (IBijector part in parts)
        {
            int nu = part.UnconstrainedDimension, nc = part.ConstrainedDimension;
            part.Inverse(constrained[c..(c + nc)], unconstrained[u..(u + nu)]);
            u += nu;
            c += nc;
        }
    }

    /// <inheritdoc/>
    public void PullBack(ReadOnlyVectorSpan<double> unconstrained, ReadOnlyVectorSpan<double> constrainedGradient, VectorSpan<double> unconstrainedGradient)
    {
        Bijection.RequireLengths(this, unconstrained.Length, constrainedGradient.Length);
        Bijection.RequireLengths(this, unconstrainedGradient.Length, constrainedGradient.Length);
        int u = 0, c = 0;
        foreach (IBijector part in parts)
        {
            int nu = part.UnconstrainedDimension, nc = part.ConstrainedDimension;
            part.PullBack(unconstrained[u..(u + nu)], constrainedGradient[c..(c + nc)], unconstrainedGradient[u..(u + nu)]);
            u += nu;
            c += nc;
        }
    }
}

/// <summary>
/// Wraps a constrained-space log-density with an <see cref="IBijector"/> to form the unconstrained log-density <c>ln p(f(y)) + ln|det J_f(y)|</c>.
/// <para>制約付き空間の対数密度に <see cref="IBijector"/> を適用し、制約なし空間の対数密度 <c>ln p(f(y)) + ln|det J_f(y)|</c> を構成します。</para>
/// </summary>
/// <typeparam name="TDensity">The constrained-space log-density type. / 制約付き空間の対数密度の型。</typeparam>
/// <typeparam name="TBijector">The bijector type. / 制約変換の型。</typeparam>
/// <remarks>
/// <para>Thread-safe for concurrent evaluation whenever both the underlying density and bijector are thread-safe. Use <see cref="SamplingResult.Transform{TBijector}(TBijector)"/> to map posterior draws back to constrained space.</para>
/// <para>元の対数密度と制約変換がともにスレッドセーフであれば、この型も並列評価に対応します。標本を制約付き空間へ戻すには <see cref="SamplingResult.Transform{TBijector}(TBijector)"/> を使用してください。</para>
/// </remarks>
public readonly struct TransformedLogDensity<TDensity, TBijector> : ILogDensity
    where TDensity : ILogDensity
    where TBijector : IBijector
{
    private readonly TDensity density;
    private readonly TBijector bijector;

    /// <summary>
    /// Initializes a new instance of <see cref="TransformedLogDensity{TDensity, TBijector}"/>.
    /// <para><see cref="TransformedLogDensity{TDensity, TBijector}"/> の新しいインスタンスを初期化します。</para>
    /// </summary>
    /// <param name="density">The constrained-space log-density. / 制約付き空間の対数密度。</param>
    /// <param name="bijector">The bijector mapping unconstrained points to constrained space. / 制約なし空間から制約付き空間への変換。</param>
    /// <exception cref="ArgumentNullException"><paramref name="density"/> or <paramref name="bijector"/> is <see langword="null"/>. / <paramref name="density"/> または <paramref name="bijector"/> が <see langword="null"/> の場合。</exception>
    public TransformedLogDensity(TDensity density, TBijector bijector)
    {
        ArgumentNullException.ThrowIfNull(density);
        ArgumentNullException.ThrowIfNull(bijector);
        this.density = density;
        this.bijector = bijector;
    }

    /// <inheritdoc/>
    public double LogDensity(ReadOnlyVectorSpan<double> theta)
    {
        int k = bijector.ConstrainedDimension;
        double[]? rented = k > Bijection.StackLimit ? ArrayPool<double>.Shared.Rent(k) : null;
        Span<double> buffer = rented is null ? stackalloc double[k] : rented.AsSpan(0, k);
        try
        {
            var x = new VectorSpan<double>(buffer);
            double logJacobian = bijector.Forward(theta, x);
            return density.LogDensity(x) + logJacobian;
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<double>.Shared.Return(rented);
            }
        }
    }
}

/// <summary>
/// Wraps a constrained-space differentiable log-density with an <see cref="IBijector"/> to form an unconstrained <see cref="IDifferentiableLogDensity"/>.
/// <para>制約付き空間の微分可能な対数密度に <see cref="IBijector"/> を適用し、制約なし空間の <see cref="IDifferentiableLogDensity"/> を構成します。</para>
/// </summary>
/// <typeparam name="TDensity">The constrained-space differentiable log-density type. / 制約付き空間の微分可能な対数密度の型。</typeparam>
/// <typeparam name="TBijector">The bijector type. / 制約変換の型。</typeparam>
/// <remarks>
/// <para>Gradients are pulled back to unconstrained space via <see cref="IBijector.PullBack"/>. Thread-safety requirements are identical to <see cref="TransformedLogDensity{TDensity, TBijector}"/>.</para>
/// <para>勾配は <see cref="IBijector.PullBack"/> によって制約なし空間へ変換されます。スレッドセーフの条件は <see cref="TransformedLogDensity{TDensity, TBijector}"/> と同じです。</para>
/// </remarks>
public readonly struct TransformedDifferentiableLogDensity<TDensity, TBijector> : IDifferentiableLogDensity
    where TDensity : IDifferentiableLogDensity
    where TBijector : IBijector
{
    private readonly TDensity density;
    private readonly TBijector bijector;

    /// <summary>
    /// Initializes a new instance of <see cref="TransformedDifferentiableLogDensity{TDensity, TBijector}"/>.
    /// <para><see cref="TransformedDifferentiableLogDensity{TDensity, TBijector}"/> の新しいインスタンスを初期化します。</para>
    /// </summary>
    /// <param name="density">The constrained-space differentiable log-density. / 制約付き空間の微分可能な対数密度。</param>
    /// <param name="bijector">The bijector mapping unconstrained points to constrained space. / 制約なし空間から制約付き空間への変換。</param>
    /// <exception cref="ArgumentNullException"><paramref name="density"/> or <paramref name="bijector"/> is <see langword="null"/>. / <paramref name="density"/> または <paramref name="bijector"/> が <see langword="null"/> の場合。</exception>
    public TransformedDifferentiableLogDensity(TDensity density, TBijector bijector)
    {
        ArgumentNullException.ThrowIfNull(density);
        ArgumentNullException.ThrowIfNull(bijector);
        this.density = density;
        this.bijector = bijector;
    }

    /// <inheritdoc/>
    public double LogDensity(ReadOnlyVectorSpan<double> theta) => new TransformedLogDensity<TDensity, TBijector>(density, bijector).LogDensity(theta);

    /// <inheritdoc/>
    public double LogDensity(ReadOnlyVectorSpan<double> theta, VectorSpan<double> gradient)
    {
        int k = bijector.ConstrainedDimension;
        double[]? rented = 2 * k > Bijection.StackLimit ? ArrayPool<double>.Shared.Rent(2 * k) : null;
        Span<double> buffer = rented is null ? stackalloc double[2 * k] : rented.AsSpan(0, 2 * k);
        try
        {
            var x = new VectorSpan<double>(buffer[..k]);
            var gx = new VectorSpan<double>(buffer[k..]);
            gx.Fill(0);
            double logJacobian = bijector.Forward(theta, x);
            double value = density.LogDensity(x, gx) + logJacobian;
            bijector.PullBack(theta, gx, gradient);
            return value;
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<double>.Shared.Return(rented);
            }
        }
    }
}

internal static class Bijection
{
    public const int StackLimit = 256;

    public static void RequireLengths(IBijector bijector, int unconstrained, int constrained)
    {
        if (unconstrained != bijector.UnconstrainedDimension || constrained != bijector.ConstrainedDimension)
        {
            throw new ArgumentException(
                $"The unconstrained and constrained vectors must have lengths {bijector.UnconstrainedDimension} and {bijector.ConstrainedDimension}.");
        }
    }
}
