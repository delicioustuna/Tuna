using System.Numerics;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Optimization.Stochastic;

/// <summary>
/// Implements the AdaMax optimizer, an infinity-norm variant of Adam.
/// <para>Adam の 2 次モーメントを無限大ノルムに置き換えた AdaMax オプティマイザーです。</para>
/// </summary>
/// <remarks>
/// <para>Updates <c>m ← β₁m + (1 − β₁)g</c>, <c>u ← max(β₂u, |g|)</c>, and <c>x ← x − (η / (1 − β₁ᵗ)) m / u</c> componentwise. Components with <c>u == 0</c> are left unchanged.</para>
/// <para>成分ごとに <c>m ← β₁m + (1 − β₁)g</c>、<c>u ← max(β₂u, |g|)</c>、<c>x ← x − (η / (1 − β₁ᵗ)) m / u</c> と更新します。<c>u == 0</c> の成分は更新しません。</para>
/// </remarks>
public sealed class AdaMax : StochasticGradientOptimizer
{
    private readonly double beta1;
    private readonly double beta2;
    private readonly double[] first;
    private readonly double[] norm;

    /// <summary>
    /// Initializes a new <see cref="AdaMax"/> optimizer for the specified parameter dimension.
    /// <para>指定したパラメーター次元に対する <see cref="AdaMax"/> オプティマイザーを初期化します。</para>
    /// </summary>
    /// <param name="dimension">The number of parameters (at least 1). / パラメーター数（1 以上）。</param>
    /// <param name="options">Optional solver configuration, or <see langword="null"/> for defaults. / オプション設定。<see langword="null"/> の場合は既定値を使用します。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="dimension"/> is less than 1, or an option value in <paramref name="options"/> is out of range. / <paramref name="dimension"/> が 1 未満、または <paramref name="options"/> の設定値が有効範囲外の場合。</exception>
    public AdaMax(int dimension, AdaMaxOptions? options = null)
        : base(dimension, options ??= new AdaMaxOptions())
    {
        CheckCoefficient(options.Beta1, nameof(options.Beta1));
        CheckCoefficient(options.Beta2, nameof(options.Beta2));
        beta1 = options.Beta1;
        beta2 = options.Beta2;
        first = new double[dimension];
        norm = new double[dimension];
    }

    private protected override void Update(VectorSpan<double> parameters, ReadOnlySpan<double> gradient, double learningRate, long step)
    {
        double rate = learningRate / (1 - Math.Pow(beta1, step));
        UpdateCore(parameters, gradient, first, norm, rate, beta1, beta2);
    }

    internal static void UpdateCore<T>(VectorSpan<T> parameters, ReadOnlySpan<T> gradient, Span<T> first, Span<T> norm,
        T rate, double beta1, double beta2)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        T firstDecay = T.CreateTruncating(beta1);
        T firstWeight = T.CreateTruncating(1 - beta1);
        T normDecay = T.CreateTruncating(beta2);
        for (int i = 0; i < gradient.Length; i++)
        {
            T g = gradient[i];
            T m = firstDecay * first[i] + firstWeight * g;
            T u = T.Max(normDecay * norm[i], T.Abs(g));
            first[i] = m;
            norm[i] = u;
            if (u > T.Zero)
            {
                parameters[i] -= rate * m / u;
            }
        }
    }

    private protected override void ResetState()
    {
        Array.Clear(first);
        Array.Clear(norm);
    }
}

/// <summary>
/// Updates a parameter set with the AdaMax optimizer.
/// <para>AdaMax でパラメーター一式を更新します。</para>
/// </summary>
/// <typeparam name="T">The element type (<see langword="float"/> or <see langword="double"/>). / 要素型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
public sealed class AdaMax<T> : StochasticGradientOptimizer<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    private readonly double beta1;
    private readonly double beta2;
    private readonly T[][] first;
    private readonly T[][] norm;

    /// <summary>
    /// Initializes an optimizer bound to the specified parameter set.
    /// <para>指定したパラメーター一式を更新するオプティマイザーを初期化します。</para>
    /// </summary>
    /// <param name="parameters">The parameters to update. / 更新するパラメーター一式。</param>
    /// <param name="options">Optimizer settings, or <see langword="null"/> for the defaults. / 設定。<see langword="null"/> の場合は既定値を使用します。</param>
    /// <exception cref="ArgumentNullException"><paramref name="parameters"/> is <see langword="null"/>. / <paramref name="parameters"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option is out of range. / 設定値が有効範囲外の場合。</exception>
    /// <exception cref="NotSupportedException">The element type is neither <see langword="float"/> nor <see langword="double"/>. / 要素型が <see langword="float"/> でも <see langword="double"/> でもない場合。</exception>
    public AdaMax(ParameterSet<T> parameters, AdaMaxOptions? options = null)
        : base(parameters, options ??= new AdaMaxOptions())
    {
        CheckCoefficient(options.Beta1, nameof(options.Beta1));
        CheckCoefficient(options.Beta2, nameof(options.Beta2));
        beta1 = options.Beta1;
        beta2 = options.Beta2;
        first = AllocateState();
        norm = AllocateState();
    }

    private protected override void Update(double rate, long step, GradientClip clip)
    {
        T correctedRate = T.CreateTruncating(rate / (1 - Math.Pow(beta1, step)));
        for (int i = 0; i < Parameters.Count; i++)
        {
            var parameter = Parameters[i];
            var gradient = PrepareGradient(parameter, rate, clip);
            AdaMax.UpdateCore<T>(parameter.Value, gradient, first[i], norm[i], correctedRate, beta1, beta2);
        }
    }

    private protected override void ResetState()
    {
        ClearState(first);
        ClearState(norm);
    }
}
