using System.Numerics;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Optimization.Stochastic;

/// <summary>
/// Implements the RMSProp optimizer with optional centering and momentum.
/// <para>中心化およびモーメンタムに対応した RMSProp オプティマイザーです。</para>
/// </summary>
/// <remarks>
/// <para>Maintains an exponential moving average of squared gradients <c>s ← ρs + (1 − ρ)g²</c> and updates <c>x ← x − ηg / (√s + ε)</c> componentwise without bias correction. When <see cref="RmsPropOptions.Centered"/> is <see langword="true"/>, uses <c>√max(s − m², 0) + ε</c>; when <see cref="RmsPropOptions.Momentum"/> is positive, applies momentum to the scaled step <c>b ← μb + g / r</c>.</para>
/// <para>勾配二乗の指数移動平均 <c>s ← ρs + (1 − ρ)g²</c> を保持し、バイアス補正なしで成分ごとに <c>x ← x − ηg / (√s + ε)</c> と更新します。<see cref="RmsPropOptions.Centered"/> が <see langword="true"/> の場合は <c>√max(s − m², 0) + ε</c> を分母に用い、<see cref="RmsPropOptions.Momentum"/> が正の場合はスケーリング後のステップ <c>b ← μb + g / r</c> にモーメンタムを適用します。</para>
/// </remarks>
public sealed class RmsProp : StochasticGradientOptimizer
{
    private readonly double decay;
    private readonly double epsilon;
    private readonly double momentum;
    private readonly double[] square;
    private readonly double[]? mean;
    private readonly double[]? buffer;

    /// <summary>
    /// Initializes a new <see cref="RmsProp"/> optimizer for the specified parameter dimension.
    /// <para>指定したパラメーター次元に対する <see cref="RmsProp"/> オプティマイザーを初期化します。</para>
    /// </summary>
    /// <param name="dimension">The number of parameters (at least 1). / パラメーター数（1 以上）。</param>
    /// <param name="options">Optional solver configuration, or <see langword="null"/> for defaults. / オプション設定。<see langword="null"/> の場合は既定値を使用します。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="dimension"/> is less than 1, or an option value in <paramref name="options"/> is out of range. / <paramref name="dimension"/> が 1 未満、または <paramref name="options"/> の設定値が有効範囲外の場合。</exception>
    public RmsProp(int dimension, RmsPropOptions? options = null)
        : base(dimension, options ??= new RmsPropOptions())
    {
        CheckCoefficient(options.Decay, nameof(options.Decay));
        CheckEpsilon(options.Epsilon, nameof(options.Epsilon));
        CheckCoefficient(options.Momentum, nameof(options.Momentum));
        decay = options.Decay;
        epsilon = options.Epsilon;
        momentum = options.Momentum;
        square = new double[dimension];
        mean = options.Centered ? new double[dimension] : null;
        buffer = momentum > 0 ? new double[dimension] : null;
    }

    private protected override void Update(VectorSpan<double> parameters, ReadOnlySpan<double> gradient, double learningRate, long step)
    {
        UpdateCore(parameters, gradient, square, mean, buffer, learningRate, decay, epsilon, momentum);
    }

    internal static void UpdateCore<T>(VectorSpan<T> parameters, ReadOnlySpan<T> gradient, Span<T> square, Span<T> mean, Span<T> buffer,
        T learningRate, double decay, T epsilon, T momentum)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        T coefficient = T.CreateTruncating(decay);
        T rest = T.CreateTruncating(1 - decay);
        for (int i = 0; i < gradient.Length; i++)
        {
            T g = gradient[i];
            T s = coefficient * square[i] + rest * g * g;
            square[i] = s;
            if (!mean.IsEmpty)
            {
                T m = coefficient * mean[i] + rest * g;
                mean[i] = m;
                s = T.Max(s - m * m, T.Zero);
            }

            T scaled = g / (T.Sqrt(s) + epsilon);
            if (!buffer.IsEmpty)
            {
                scaled += momentum * buffer[i];
                buffer[i] = scaled;
            }

            parameters[i] -= learningRate * scaled;
        }
    }

    private protected override void ResetState()
    {
        Array.Clear(square);
        if (mean is not null)
        {
            Array.Clear(mean);
        }

        if (buffer is not null)
        {
            Array.Clear(buffer);
        }
    }
}

/// <summary>
/// Updates a parameter set with RMSProp, with optional centering and momentum.
/// <para>RMSProp でパラメーター一式を更新します。中心化とモーメンタムも使用できます。</para>
/// </summary>
/// <typeparam name="T">The element type (<see langword="float"/> or <see langword="double"/>). / 要素型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
public sealed class RmsProp<T> : StochasticGradientOptimizer<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    private readonly double decay;
    private readonly T epsilon;
    private readonly T momentum;
    private readonly T[][] square;
    private readonly T[][]? mean;
    private readonly T[][]? buffer;

    /// <summary>
    /// Initializes an optimizer bound to the specified parameter set.
    /// <para>指定したパラメーター一式を更新するオプティマイザーを初期化します。</para>
    /// </summary>
    /// <param name="parameters">The parameters to update. / 更新するパラメーター一式。</param>
    /// <param name="options">Optimizer settings, or <see langword="null"/> for the defaults. / 設定。<see langword="null"/> の場合は既定値を使用します。</param>
    /// <exception cref="ArgumentNullException"><paramref name="parameters"/> is <see langword="null"/>. / <paramref name="parameters"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option is out of range. / 設定値が有効範囲外の場合。</exception>
    /// <exception cref="NotSupportedException">The element type is neither <see langword="float"/> nor <see langword="double"/>. / 要素型が <see langword="float"/> でも <see langword="double"/> でもない場合。</exception>
    public RmsProp(ParameterSet<T> parameters, RmsPropOptions? options = null)
        : base(parameters, options ??= new RmsPropOptions())
    {
        CheckCoefficient(options.Decay, nameof(options.Decay));
        CheckCoefficient(options.Momentum, nameof(options.Momentum));
        CheckEpsilon(options.Epsilon, nameof(options.Epsilon));
        decay = options.Decay;
        epsilon = T.CreateTruncating(options.Epsilon);
        momentum = T.CreateTruncating(options.Momentum);
        square = AllocateState();
        mean = options.Centered ? AllocateState() : null;
        buffer = options.Momentum > 0 ? AllocateState() : null;
    }

    private protected override void Update(double rate, long step, GradientClip clip)
    {
        T learningRate = T.CreateTruncating(rate);
        for (int i = 0; i < Parameters.Count; i++)
        {
            var parameter = Parameters[i];
            var gradient = PrepareGradient(parameter, rate, clip);
            RmsProp.UpdateCore<T>(parameter.Value, gradient, square[i], mean?[i], buffer?[i], learningRate, decay, epsilon, momentum);
        }
    }

    private protected override void ResetState()
    {
        ClearState(square);
        ClearState(mean);
        ClearState(buffer);
    }
}
