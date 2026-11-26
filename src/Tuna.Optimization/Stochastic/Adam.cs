using System.Numerics;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Optimization.Stochastic;

/// <summary>
/// Implements the Adam optimizer, supporting AMSGrad and decoupled weight decay (AdamW).
/// <para>Adam オプティマイザーです（AMSGrad および切り離した重み減衰による AdamW に対応）。</para>
/// </summary>
/// <remarks>
/// <para>Enable AMSGrad via <see cref="AdamOptions.Amsgrad"/>. For AdamW, set <see cref="StochasticGradientOptions.WeightDecay"/> to a positive value and <see cref="StochasticGradientOptions.DecoupledWeightDecay"/> to <see langword="true"/>.</para>
/// <para>AMSGrad を使用する場合は <see cref="AdamOptions.Amsgrad"/> を <see langword="true"/> に設定します。AdamW として使用するには <see cref="StochasticGradientOptions.WeightDecay"/> に正の値を設定し、<see cref="StochasticGradientOptions.DecoupledWeightDecay"/> を <see langword="true"/> にしてください。</para>
/// </remarks>
public sealed class Adam : StochasticGradientOptimizer
{
    private readonly double beta1;
    private readonly double beta2;
    private readonly double epsilon;
    private readonly double[] first;
    private readonly double[] second;
    private readonly double[]? maximum;

    /// <summary>
    /// Initializes a new <see cref="Adam"/> optimizer for the specified parameter dimension.
    /// <para>指定したパラメーター次元に対する <see cref="Adam"/> オプティマイザーを初期化します。</para>
    /// </summary>
    /// <param name="dimension">The number of parameters (at least 1). / パラメーター数（1 以上）。</param>
    /// <param name="options">Optional solver configuration, or <see langword="null"/> for defaults. / オプション設定。<see langword="null"/> の場合は既定値を使用します。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="dimension"/> is less than 1, or an option value in <paramref name="options"/> is out of range. / <paramref name="dimension"/> が 1 未満、または <paramref name="options"/> の設定値が有効範囲外の場合。</exception>
    public Adam(int dimension, AdamOptions? options = null)
        : base(dimension, options ??= new AdamOptions())
    {
        CheckCoefficient(options.Beta1, nameof(options.Beta1));
        CheckCoefficient(options.Beta2, nameof(options.Beta2));
        CheckEpsilon(options.Epsilon, nameof(options.Epsilon));
        beta1 = options.Beta1;
        beta2 = options.Beta2;
        epsilon = options.Epsilon;
        first = new double[dimension];
        second = new double[dimension];
        maximum = options.Amsgrad ? new double[dimension] : null;
    }

    private protected override void Update(VectorSpan<double> parameters, ReadOnlySpan<double> gradient, double learningRate, long step)
    {
        double correction1 = 1 - Math.Pow(beta1, step);
        double correction2 = 1 - Math.Pow(beta2, step);
        UpdateCore(parameters, gradient, first, second, maximum, learningRate, beta1, beta2, epsilon, correction1, correction2);
    }

    internal static void UpdateCore<T>(VectorSpan<T> parameters, ReadOnlySpan<T> gradient, Span<T> first, Span<T> second, Span<T> maximum,
        T learningRate, double beta1, double beta2, T epsilon, T correction1, T correction2)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        T firstDecay = T.CreateTruncating(beta1);
        T firstWeight = T.CreateTruncating(1 - beta1);
        T secondDecay = T.CreateTruncating(beta2);
        T secondWeight = T.CreateTruncating(1 - beta2);
        for (int i = 0; i < gradient.Length; i++)
        {
            T g = gradient[i];
            T m = firstDecay * first[i] + firstWeight * g;
            T v = secondDecay * second[i] + secondWeight * g * g;
            first[i] = m;
            second[i] = v;
            if (!maximum.IsEmpty)
            {
                v = T.Max(maximum[i], v);
                maximum[i] = v;
            }

            parameters[i] -= learningRate * (m / correction1) / (T.Sqrt(v / correction2) + epsilon);
        }
    }

    private protected override void ResetState()
    {
        Array.Clear(first);
        Array.Clear(second);
        if (maximum is not null)
        {
            Array.Clear(maximum);
        }
    }
}

/// <summary>
/// Updates a parameter set with Adam, including optional AMSGrad or decoupled weight decay.
/// <para>Adam でパラメーター一式を更新します。AMSGrad と切り離し重み減衰も使用できます。</para>
/// </summary>
/// <typeparam name="T">The element type (<see langword="float"/> or <see langword="double"/>). / 要素型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
public sealed class Adam<T> : StochasticGradientOptimizer<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    private readonly double beta1;
    private readonly double beta2;
    private readonly T epsilon;
    private readonly T[][] first;
    private readonly T[][] second;
    private readonly T[][]? maximum;

    /// <summary>
    /// Initializes an optimizer bound to the specified parameter set.
    /// <para>指定したパラメーター一式を更新するオプティマイザーを初期化します。</para>
    /// </summary>
    /// <param name="parameters">The parameters to update. / 更新するパラメーター一式。</param>
    /// <param name="options">Optimizer settings, or <see langword="null"/> for the defaults. / 設定。<see langword="null"/> の場合は既定値を使用します。</param>
    /// <exception cref="ArgumentNullException"><paramref name="parameters"/> is <see langword="null"/>. / <paramref name="parameters"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option is out of range. / 設定値が有効範囲外の場合。</exception>
    /// <exception cref="NotSupportedException">The element type is neither <see langword="float"/> nor <see langword="double"/>. / 要素型が <see langword="float"/> でも <see langword="double"/> でもない場合。</exception>
    public Adam(ParameterSet<T> parameters, AdamOptions? options = null)
        : base(parameters, options ??= new AdamOptions())
    {
        CheckCoefficient(options.Beta1, nameof(options.Beta1));
        CheckCoefficient(options.Beta2, nameof(options.Beta2));
        CheckEpsilon(options.Epsilon, nameof(options.Epsilon));
        beta1 = options.Beta1;
        beta2 = options.Beta2;
        epsilon = T.CreateTruncating(options.Epsilon);
        first = AllocateState();
        second = AllocateState();
        maximum = options.Amsgrad ? AllocateState() : null;
    }

    private protected override void Update(double rate, long step, GradientClip clip)
    {
        T learningRate = T.CreateTruncating(rate);
        T correction1 = T.CreateTruncating(1 - Math.Pow(beta1, step));
        T correction2 = T.CreateTruncating(1 - Math.Pow(beta2, step));
        for (int i = 0; i < Parameters.Count; i++)
        {
            var parameter = Parameters[i];
            var gradient = PrepareGradient(parameter, rate, clip);
            Adam.UpdateCore<T>(parameter.Value, gradient, first[i], second[i], maximum?[i],
                learningRate, beta1, beta2, epsilon, correction1, correction2);
        }
    }

    private protected override void ResetState()
    {
        ClearState(first);
        ClearState(second);
        ClearState(maximum);
    }
}
