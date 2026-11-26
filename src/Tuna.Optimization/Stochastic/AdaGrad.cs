using System.Numerics;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Optimization.Stochastic;

/// <summary>
/// Implements the diagonal AdaGrad optimizer.
/// <para>対角型 AdaGrad オプティマイザーです。</para>
/// </summary>
/// <remarks>
/// <para>Accumulates squared gradients <c>G ← G + g²</c> and updates <c>x ← x − ηg / (√G + ε)</c> componentwise, monotonically decreasing the per-coordinate effective learning rate.</para>
/// <para>勾配の二乗和 <c>G ← G + g²</c> を蓄積し、成分ごとに <c>x ← x − ηg / (√G + ε)</c> と更新します。各成分の実効学習率は単調に減少します。</para>
/// </remarks>
public sealed class AdaGrad : StochasticGradientOptimizer
{
    private readonly double epsilon;
    private readonly double initialAccumulator;
    private readonly double[] accumulator;

    /// <summary>
    /// Initializes a new <see cref="AdaGrad"/> optimizer for the specified parameter dimension.
    /// <para>指定したパラメーター次元に対する <see cref="AdaGrad"/> オプティマイザーを初期化します。</para>
    /// </summary>
    /// <param name="dimension">The number of parameters (at least 1). / パラメーター数（1 以上）。</param>
    /// <param name="options">Optional solver configuration, or <see langword="null"/> for defaults. / オプション設定。<see langword="null"/> の場合は既定値を使用します。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="dimension"/> is less than 1, or an option value in <paramref name="options"/> is out of range. / <paramref name="dimension"/> が 1 未満、または <paramref name="options"/> の設定値が有効範囲外の場合。</exception>
    public AdaGrad(int dimension, AdaGradOptions? options = null)
        : base(dimension, options ??= new AdaGradOptions())
    {
        CheckEpsilon(options.Epsilon, nameof(options.Epsilon));
        if (!(options.InitialAccumulator >= 0 && double.IsFinite(options.InitialAccumulator)))
        {
            throw new ArgumentOutOfRangeException(nameof(options.InitialAccumulator), "The initial accumulator must be non-negative and finite.");
        }

        epsilon = options.Epsilon;
        initialAccumulator = options.InitialAccumulator;
        accumulator = new double[dimension];
        ResetState();
    }

    private protected override void Update(VectorSpan<double> parameters, ReadOnlySpan<double> gradient, double learningRate, long step)
    {
        UpdateCore(parameters, gradient, accumulator, learningRate, epsilon);
    }

    internal static void UpdateCore<T>(VectorSpan<T> parameters, ReadOnlySpan<T> gradient, Span<T> accumulator,
        T learningRate, T epsilon)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        for (int i = 0; i < gradient.Length; i++)
        {
            T g = gradient[i];
            T sum = accumulator[i] + g * g;
            accumulator[i] = sum;
            parameters[i] -= learningRate * g / (T.Sqrt(sum) + epsilon);
        }
    }

    private protected override void ResetState() => Array.Fill(accumulator, initialAccumulator);
}

/// <summary>
/// Updates a parameter set with the AdaGrad optimizer.
/// <para>AdaGrad でパラメーター一式を更新します。</para>
/// </summary>
/// <typeparam name="T">The element type (<see langword="float"/> or <see langword="double"/>). / 要素型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
public sealed class AdaGrad<T> : StochasticGradientOptimizer<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    private readonly T epsilon;
    private readonly T initialAccumulator;
    private readonly T[][] accumulator;

    /// <summary>
    /// Initializes an optimizer bound to the specified parameter set.
    /// <para>指定したパラメーター一式を更新するオプティマイザーを初期化します。</para>
    /// </summary>
    /// <param name="parameters">The parameters to update. / 更新するパラメーター一式。</param>
    /// <param name="options">Optimizer settings, or <see langword="null"/> for the defaults. / 設定。<see langword="null"/> の場合は既定値を使用します。</param>
    /// <exception cref="ArgumentNullException"><paramref name="parameters"/> is <see langword="null"/>. / <paramref name="parameters"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option is out of range. / 設定値が有効範囲外の場合。</exception>
    /// <exception cref="NotSupportedException">The element type is neither <see langword="float"/> nor <see langword="double"/>. / 要素型が <see langword="float"/> でも <see langword="double"/> でもない場合。</exception>
    public AdaGrad(ParameterSet<T> parameters, AdaGradOptions? options = null)
        : base(parameters, options ??= new AdaGradOptions())
    {
        CheckEpsilon(options.Epsilon, nameof(options.Epsilon));
        epsilon = T.CreateTruncating(options.Epsilon);
        initialAccumulator = T.CreateTruncating(options.InitialAccumulator);
        if (!(options.InitialAccumulator >= 0 && T.IsFinite(initialAccumulator)))
        {
            throw new ArgumentOutOfRangeException(nameof(options.InitialAccumulator), "The initial accumulator must be non-negative and finite in the parameter element type.");
        }

        accumulator = AllocateState();
        ResetState();
    }

    private protected override void Update(double rate, long step, GradientClip clip)
    {
        T learningRate = T.CreateTruncating(rate);
        for (int i = 0; i < Parameters.Count; i++)
        {
            var parameter = Parameters[i];
            var gradient = PrepareGradient(parameter, rate, clip);
            AdaGrad.UpdateCore<T>(parameter.Value, gradient, accumulator[i], learningRate, epsilon);
        }
    }

    private protected override void ResetState()
    {
        foreach (var entries in accumulator)
        {
            Array.Fill(entries, initialAccumulator);
        }
    }
}
