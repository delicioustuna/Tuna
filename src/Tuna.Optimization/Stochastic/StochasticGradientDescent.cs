using System.Numerics;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Optimization.Stochastic;

/// <summary>
/// Implements stochastic gradient descent (SGD) with optional classical or Nesterov momentum.
/// <para>モーメンタム（Polyak 型および Nesterov 型）に対応した確率的勾配降下法（SGD）オプティマイザーです。</para>
/// </summary>
/// <remarks>
/// <para>Updates <c>x ← x − ηg</c> without momentum, or <c>v ← μv + g</c> followed by <c>x ← x − ηv</c> (or <c>x ← x − η(g + μv)</c> with Nesterov momentum), starting from <c>v = 0</c>. When the learning rate changes across steps, the new learning rate scales the accumulated velocity <c>v</c> as well.</para>
/// <para>モーメンタムなしでは <c>x ← x − ηg</c>、モーメンタム μ を指定すると <c>v = 0</c> から <c>v ← μv + g</c>、<c>x ← x − ηv</c>（Nesterov 型では <c>x ← x − η(g + μv)</c>）で更新します。学習率が途中で変化する場合、蓄積済みの速度 <c>v</c> にも新しい学習率が適用されます。</para>
/// </remarks>
public sealed class StochasticGradientDescent : StochasticGradientOptimizer
{
    private readonly double momentum;
    private readonly bool nesterov;
    private readonly double[]? velocity;

    /// <summary>
    /// Initializes a new <see cref="StochasticGradientDescent"/> optimizer for the specified parameter dimension.
    /// <para>指定したパラメーター次元に対する <see cref="StochasticGradientDescent"/> オプティマイザーを初期化します。</para>
    /// </summary>
    /// <param name="dimension">The number of parameters (at least 1). / パラメーター数（1 以上）。</param>
    /// <param name="options">Optional solver configuration, or <see langword="null"/> for defaults. / オプション設定。<see langword="null"/> の場合は既定値を使用します。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="dimension"/> is less than 1, or an option value in <paramref name="options"/> is out of range. / <paramref name="dimension"/> が 1 未満、または <paramref name="options"/> の設定値が有効範囲外の場合。</exception>
    public StochasticGradientDescent(int dimension, StochasticGradientDescentOptions? options = null)
        : base(dimension, options ??= new StochasticGradientDescentOptions())
    {
        CheckCoefficient(options.Momentum, nameof(options.Momentum));
        momentum = options.Momentum;
        nesterov = options.Nesterov && momentum > 0;
        velocity = momentum > 0 ? new double[dimension] : null;
    }

    /// <summary>
    /// Gets the momentum coefficient μ.
    /// <para>モーメンタム係数 μ を取得します。</para>
    /// </summary>
    public double Momentum => momentum;

    /// <summary>
    /// Gets a value indicating whether Nesterov momentum is active.
    /// <para>Nesterov モーメンタムが有効かどうかを取得します。</para>
    /// </summary>
    public bool Nesterov => nesterov;

    private protected override void Update(VectorSpan<double> parameters, ReadOnlySpan<double> gradient, double learningRate, long step)
    {
        UpdateCore(parameters, gradient, velocity, learningRate, momentum, nesterov);
    }

    internal static void UpdateCore<T>(VectorSpan<T> parameters, ReadOnlySpan<T> gradient, Span<T> velocity,
        T learningRate, double momentum, bool nesterov)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (velocity.IsEmpty)
        {
            for (int i = 0; i < gradient.Length; i++)
            {
                parameters[i] -= learningRate * gradient[i];
            }

            return;
        }

        T coefficient = T.CreateTruncating(momentum);
        for (int i = 0; i < gradient.Length; i++)
        {
            T v = coefficient * velocity[i] + gradient[i];
            velocity[i] = v;
            parameters[i] -= learningRate * (nesterov ? gradient[i] + coefficient * v : v);
        }
    }

    private protected override void ResetState()
    {
        if (velocity is not null)
        {
            Array.Clear(velocity);
        }
    }
}

/// <summary>
/// Updates a parameter set with stochastic gradient descent, with optional momentum.
/// <para>確率的勾配降下法でパラメーター一式を更新します。モーメンタムも使用できます。</para>
/// </summary>
/// <typeparam name="T">The element type (<see langword="float"/> or <see langword="double"/>). / 要素型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
public sealed class StochasticGradientDescent<T> : StochasticGradientOptimizer<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    private readonly double momentum;
    private readonly bool nesterov;
    private readonly T[][]? velocity;

    /// <summary>
    /// Initializes an optimizer bound to the specified parameter set.
    /// <para>指定したパラメーター一式を更新するオプティマイザーを初期化します。</para>
    /// </summary>
    /// <param name="parameters">The parameters to update. / 更新するパラメーター一式。</param>
    /// <param name="options">Optimizer settings, or <see langword="null"/> for the defaults. / 設定。<see langword="null"/> の場合は既定値を使用します。</param>
    /// <exception cref="ArgumentNullException"><paramref name="parameters"/> is <see langword="null"/>. / <paramref name="parameters"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option is out of range. / 設定値が有効範囲外の場合。</exception>
    /// <exception cref="NotSupportedException">The element type is neither <see langword="float"/> nor <see langword="double"/>. / 要素型が <see langword="float"/> でも <see langword="double"/> でもない場合。</exception>
    public StochasticGradientDescent(ParameterSet<T> parameters, StochasticGradientDescentOptions? options = null)
        : base(parameters, options ??= new StochasticGradientDescentOptions())
    {
        CheckCoefficient(options.Momentum, nameof(options.Momentum));
        momentum = options.Momentum;
        nesterov = options.Nesterov && momentum > 0;
        velocity = momentum > 0 ? AllocateState() : null;
    }

    /// <inheritdoc cref="StochasticGradientDescent.Momentum"/>
    public double Momentum => momentum;

    /// <inheritdoc cref="StochasticGradientDescent.Nesterov"/>
    public bool Nesterov => nesterov;

    private protected override void Update(double rate, long step, GradientClip clip)
    {
        T learningRate = T.CreateTruncating(rate);
        for (int i = 0; i < Parameters.Count; i++)
        {
            var parameter = Parameters[i];
            var gradient = PrepareGradient(parameter, rate, clip);
            StochasticGradientDescent.UpdateCore<T>(parameter.Value, gradient, velocity?[i], learningRate, momentum, nesterov);
        }
    }

    private protected override void ResetState() => ClearState(velocity);
}
