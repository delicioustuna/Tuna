using Tuna.LinearAlgebra;

namespace Tuna.Optimization.Proximal;

/// <summary>
/// Represents the identically zero function, whose proximal operator is the identity mapping.
/// <para>恒等的に 0 を返す関数を表します（近接写像は恒等写像です）。ADMM で一方の項が存在しない場合などに使用します。</para>
/// </summary>
public sealed class ZeroFunction : IProximalOperator
{
    /// <inheritdoc/>
    public void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)
    {
        ProximalChecks.Check(v, step, destination);
        ProximalKernels.Scale(v, 1.0, destination);
    }

    /// <inheritdoc/>
    public double Evaluate(ReadOnlyVectorSpan<double> x) => 0;
}

/// <summary>
/// Represents the (optionally weighted) L1 norm penalty <c>h(x) = λ Σ wᵢ|xᵢ|</c>.
/// <para>L1 ノルム罰則 <c>h(x) = λ Σ wᵢ|xᵢ|</c> を表します。</para>
/// </summary>
/// <remarks>
/// <para>Its proximal operator is componentwise soft thresholding <c>sign(v) max(|v| − tλwᵢ, 0)</c>.</para>
/// <para>近接写像は成分ごとの軟判定しきい値処理 <c>sign(v) max(|v| − tλwᵢ, 0)</c> です。</para>
/// </remarks>
public sealed class L1Penalty : IProximalOperator
{
    private readonly double lambda;
    private readonly double[]? weights;

    /// <summary>
    /// Initializes an unweighted L1 penalty with uniform weight <paramref name="lambda"/>.
    /// <para>すべての成分に共通の係数 <paramref name="lambda"/> を適用する L1 罰則を初期化します。</para>
    /// </summary>
    /// <param name="lambda">The regularization strength λ (non-negative and finite). / 正則化係数 λ（0 以上の有限値）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="lambda"/> is negative or non-finite. / <paramref name="lambda"/> が負または有限値でない場合。</exception>
    public L1Penalty(double lambda)
    {
        ProximalChecks.NonNegative(lambda, nameof(lambda));
        this.lambda = lambda;
    }

    /// <summary>
    /// Initializes a weighted L1 penalty with componentwise weights <c>λwᵢ</c>.
    /// <para>成分ごとの重み <c>λwᵢ</c> を持つ L1 罰則を初期化します。変数の次元は重みベクトルの長さで固定されます。</para>
    /// </summary>
    /// <param name="lambda">The global regularization strength λ (non-negative and finite). / 全体の正則化係数 λ（0 以上の有限値）。</param>
    /// <param name="weights">Componentwise weights <c>wᵢ</c> (non-empty, non-negative, and finite). / 成分ごとの重み <c>wᵢ</c>（長さ 1 以上、各要素は 0 以上の有限値）。</param>
    /// <exception cref="ArgumentException"><paramref name="weights"/> is empty. / <paramref name="weights"/> が空の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="lambda"/> or an element of <paramref name="weights"/> is negative or non-finite. / <paramref name="lambda"/> または <paramref name="weights"/> の要素が負または有限値でない場合。</exception>
    public L1Penalty(double lambda, ReadOnlySpan<double> weights)
        : this(lambda)
    {
        if (weights.IsEmpty)
        {
            throw new ArgumentException("The weights must not be empty.", nameof(weights));
        }

        foreach (double w in weights)
        {
            ProximalChecks.NonNegative(w, nameof(weights));
        }

        this.weights = weights.ToArray();
    }

    /// <summary>
    /// Gets the global regularization strength λ.
    /// <para>全体の正則化係数 λ を取得します。</para>
    /// </summary>
    public double Lambda => lambda;

    /// <inheritdoc/>
    public void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)
    {
        ProximalChecks.Check(v, step, destination, weights?.Length ?? -1);
        if (weights is null)
        {
            ProximalKernels.SoftThreshold(v, step * lambda, 1.0, destination);
        }
        else
        {
            ProximalKernels.SoftThreshold(v, step * lambda, weights, destination);
        }
    }

    /// <inheritdoc/>
    public double Evaluate(ReadOnlyVectorSpan<double> x)
    {
        ProximalChecks.Length(x.Length, weights?.Length ?? -1);
        if (weights is null)
        {
            return lambda * ProximalKernels.SumAbs(x);
        }

        double sum = 0;
        for (int i = 0; i < x.Length; i++)
        {
            sum += Math.Abs(x[i]) * weights[i];
        }

        return lambda * sum;
    }
}

/// <summary>
/// Represents the elastic net penalty <c>h(x) = λ₁‖x‖₁ + (λ₂/2)‖x‖₂²</c>.
/// <para>Elastic Net 罰則 <c>h(x) = λ₁‖x‖₁ + (λ₂/2)‖x‖₂²</c> を表します。</para>
/// </summary>
/// <remarks>
/// <para>Its proximal operator applies soft thresholding with threshold <c>tλ₁</c> scaled by <c>1 / (1 + tλ₂)</c>.</para>
/// <para>近接写像は、しきい値 <c>tλ₁</c> の軟判定しきい値処理に係数 <c>1 / (1 + tλ₂)</c> を掛けたものです。</para>
/// </remarks>
public sealed class ElasticNetPenalty : IProximalOperator
{
    /// <summary>
    /// Initializes an elastic net penalty with the specified L1 and squared-L2 coefficients.
    /// <para>指定した L1 係数および L2 二乗係数で Elastic Net 罰則を初期化します。</para>
    /// </summary>
    /// <param name="l1">The L1 coefficient λ₁ (non-negative and finite). / L1 係数 λ₁（0 以上の有限値）。</param>
    /// <param name="l2">The squared-L2 coefficient λ₂ (non-negative and finite). / L2 二乗係数 λ₂（0 以上の有限値）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="l1"/> or <paramref name="l2"/> is negative or non-finite. / <paramref name="l1"/> または <paramref name="l2"/> が負または有限値でない場合。</exception>
    public ElasticNetPenalty(double l1, double l2)
    {
        ProximalChecks.NonNegative(l1, nameof(l1));
        ProximalChecks.NonNegative(l2, nameof(l2));
        L1 = l1;
        L2 = l2;
    }

    /// <summary>
    /// Gets the L1 coefficient λ₁.
    /// <para>L1 係数 λ₁ を取得します。</para>
    /// </summary>
    public double L1 { get; }

    /// <summary>
    /// Gets the squared-L2 coefficient λ₂.
    /// <para>L2 二乗係数 λ₂ を取得します。</para>
    /// </summary>
    public double L2 { get; }

    /// <inheritdoc/>
    public void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)
    {
        ProximalChecks.Check(v, step, destination);
        ProximalKernels.SoftThreshold(v, step * L1, 1 / (1 + step * L2), destination);
    }

    /// <inheritdoc/>
    public double Evaluate(ReadOnlyVectorSpan<double> x)
    {
        double norm = ProximalKernels.Norm2(x);
        return L1 * ProximalKernels.SumAbs(x) + 0.5 * L2 * norm * norm;
    }
}

/// <summary>
/// Represents the unsquared L2 norm penalty <c>h(x) = λ‖x‖₂</c>.
/// <para>二乗しない L2 ノルム罰則 <c>h(x) = λ‖x‖₂</c> を表します。</para>
/// </summary>
/// <remarks>
/// <para>Its proximal operator is block soft thresholding <c>max(1 − tλ / ‖v‖₂, 0) v</c>.</para>
/// <para>近接写像はベクトル全体のブロック縮小 <c>max(1 − tλ / ‖v‖₂, 0) v</c> です。</para>
/// </remarks>
public sealed class L2NormPenalty : IProximalOperator
{
    /// <summary>
    /// Initializes an L2 norm penalty with coefficient <paramref name="lambda"/>.
    /// <para>指定した係数 <paramref name="lambda"/> で L2 ノルム罰則を初期化します。</para>
    /// </summary>
    /// <param name="lambda">The regularization strength λ (non-negative and finite). / 正則化係数 λ（0 以上の有限値）。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="lambda"/> is negative or non-finite. / <paramref name="lambda"/> が負または有限値でない場合。</exception>
    public L2NormPenalty(double lambda)
    {
        ProximalChecks.NonNegative(lambda, nameof(lambda));
        Lambda = lambda;
    }

    /// <summary>
    /// Gets the regularization strength λ.
    /// <para>正則化係数 λ を取得します。</para>
    /// </summary>
    public double Lambda { get; }

    /// <inheritdoc/>
    public void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)
    {
        ProximalChecks.Check(v, step, destination);
        ProximalChecks.BlockShrink(v, destination, step * Lambda);
    }

    /// <inheritdoc/>
    public double Evaluate(ReadOnlyVectorSpan<double> x) => Lambda * ProximalKernels.Norm2(x);
}

/// <summary>
/// Represents the group lasso penalty over contiguous variable groups <c>h(x) = λ Σ_g √|g| ‖x_g‖₂</c>.
/// <para>連続する変数グループに対するグループ Lasso 罰則 <c>h(x) = λ Σ_g √|g| ‖x_g‖₂</c> を表します。</para>
/// </summary>
/// <remarks>
/// <para>Its proximal operator applies block soft thresholding independently to each group, weighted by <c>√|g|</c>.</para>
/// <para>近接写像は、各グループに対して重み <c>√|g|</c> を用いたブロック縮小を独立に適用します。</para>
/// </remarks>
public sealed class GroupLassoPenalty : IProximalOperator
{
    private readonly int[] starts;
    private readonly double[] weights;
    private readonly int dimension;

    /// <summary>
    /// Initializes a group lasso penalty with contiguous groups of the specified sizes.
    /// <para>先頭から順に指定したサイズで分割したグループに対するグループ Lasso 罰則を初期化します。</para>
    /// </summary>
    /// <param name="lambda">The regularization strength λ (non-negative and finite). / 正則化係数 λ（0 以上の有限値）。</param>
    /// <param name="groupSizes">Sizes of each contiguous group (non-empty; each size at least 1; their sum defines the vector dimension). / 各グループの要素数（空でないこと。各サイズは 1 以上で、合計がベクトルの次元になります）。</param>
    /// <exception cref="ArgumentException"><paramref name="groupSizes"/> is empty. / <paramref name="groupSizes"/> が空の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="lambda"/> is negative or non-finite, or a group size is less than 1. / <paramref name="lambda"/> が負または有限値でない、あるいはグループサイズが 1 未満の場合。</exception>
    public GroupLassoPenalty(double lambda, ReadOnlySpan<int> groupSizes)
    {
        ProximalChecks.NonNegative(lambda, nameof(lambda));
        if (groupSizes.IsEmpty)
        {
            throw new ArgumentException("At least one group is required.", nameof(groupSizes));
        }

        Lambda = lambda;
        starts = new int[groupSizes.Length + 1];
        weights = new double[groupSizes.Length];
        for (int k = 0; k < groupSizes.Length; k++)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(groupSizes[k], 1, nameof(groupSizes));
            starts[k + 1] = checked(starts[k] + groupSizes[k]);
            weights[k] = Math.Sqrt(groupSizes[k]);
        }

        dimension = starts[^1];
    }

    /// <summary>
    /// Gets the regularization strength λ.
    /// <para>正則化係数 λ を取得します。</para>
    /// </summary>
    public double Lambda { get; }

    /// <summary>
    /// Gets the number of groups.
    /// <para>グループ数を取得します。</para>
    /// </summary>
    public int GroupCount => weights.Length;

    /// <inheritdoc/>
    public void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)
    {
        ProximalChecks.Check(v, step, destination, dimension);
        for (int k = 0; k < weights.Length; k++)
        {
            var range = starts[k]..starts[k + 1];
            ProximalChecks.BlockShrink(v[range], destination[range], step * Lambda * weights[k]);
        }
    }

    /// <inheritdoc/>
    public double Evaluate(ReadOnlyVectorSpan<double> x)
    {
        ProximalChecks.Length(x.Length, dimension);
        double sum = 0;
        for (int k = 0; k < weights.Length; k++)
        {
            sum += weights[k] * ProximalKernels.Norm2(x[starts[k]..starts[k + 1]]);
        }

        return Lambda * sum;
    }
}

internal static class ProximalChecks
{
    public const double FeasibilityTolerance = 1e-10;

    public static void Check(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination, int dimension = -1)
    {
        if (!(step > 0))
        {
            throw new ArgumentOutOfRangeException(nameof(step), "The step must be positive.");
        }

        if (destination.Length != v.Length)
        {
            throw new ArgumentException("The destination must have the same length as the point.", nameof(destination));
        }

        Length(v.Length, dimension);
    }

    public static void Length(int length, int dimension)
    {
        if (dimension >= 0 && length != dimension)
        {
            throw new ArgumentException($"The vector length must be {dimension}.");
        }
    }

    public static void NonNegative(double value, string name)
    {
        if (!(value >= 0 && double.IsFinite(value)))
        {
            throw new ArgumentOutOfRangeException(name, "The value must be non-negative and finite.");
        }
    }

    // destination = max(1 − threshold/‖v‖₂, 0) v（ベクトル全体を縮める）
    public static void BlockShrink(ReadOnlyVectorSpan<double> v, VectorSpan<double> destination, double threshold)
    {
        double norm = ProximalKernels.Norm2(v);
        ProximalKernels.Scale(v, norm > threshold ? 1 - threshold / norm : 0, destination);
    }
}
