using System.Numerics;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;

namespace Tuna.Optimization.Stochastic;

/// <summary>
/// Implements the NAdam optimizer (Adam with Nesterov momentum and constant <c>β₁</c>).
/// <para>Nesterov モーメンタムを組み込んだ NAdam オプティマイザーです（モーメンタム係数 <c>β₁</c> は一定）。</para>
/// </summary>
/// <remarks>
/// <para>Uses lookahead first-moment estimate <c>m̂ = β₁m / (1 − β₁ᵗ⁺¹) + (1 − β₁)g / (1 − β₁ᵗ)</c> with constant <c>β₁</c> (without Dozat's per-step momentum decay schedule <c>μₜ</c>).</para>
/// <para>一定の係数 <c>β₁</c> を用い、先読み1次モーメント <c>m̂ = β₁m / (1 − β₁ᵗ⁺¹) + (1 − β₁)g / (1 − β₁ᵗ)</c> で更新します（ステップごとにモーメンタム係数 <c>μₜ</c> を変化させるスケジュールは使用しません）。</para>
/// </remarks>
public sealed class NAdam : StochasticGradientOptimizer
{
    private readonly double beta1;
    private readonly double beta2;
    private readonly double epsilon;
    private readonly double[] first;
    private readonly double[] second;

    /// <summary>
    /// Initializes a new <see cref="NAdam"/> optimizer for the specified parameter dimension.
    /// <para>指定したパラメーター次元に対する <see cref="NAdam"/> オプティマイザーを初期化します。</para>
    /// </summary>
    /// <param name="dimension">The number of parameters (at least 1). / パラメーター数（1 以上）。</param>
    /// <param name="options">Optional solver configuration, or <see langword="null"/> for defaults. / オプション設定。<see langword="null"/> の場合は既定値を使用します。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="dimension"/> is less than 1, or an option value in <paramref name="options"/> is out of range. / <paramref name="dimension"/> が 1 未満、または <paramref name="options"/> の設定値が有効範囲外の場合。</exception>
    public NAdam(int dimension, NAdamOptions? options = null)
        : base(dimension, options ??= new NAdamOptions())
    {
        CheckCoefficient(options.Beta1, nameof(options.Beta1));
        CheckCoefficient(options.Beta2, nameof(options.Beta2));
        CheckEpsilon(options.Epsilon, nameof(options.Epsilon));
        beta1 = options.Beta1;
        beta2 = options.Beta2;
        epsilon = options.Epsilon;
        first = new double[dimension];
        second = new double[dimension];
    }

    private protected override void Update(VectorSpan<double> parameters, ReadOnlySpan<double> gradient, double learningRate, long step)
    {
        double power = Math.Pow(beta1, step);
        double correctionNext = 1 - power * beta1;
        double correctionNow = 1 - power;
        double correction2 = 1 - Math.Pow(beta2, step);
        UpdateCore(parameters, gradient, first, second, learningRate, beta1, beta2, epsilon, correctionNext, correctionNow, correction2);
    }

    internal static void UpdateCore<T>(VectorSpan<T> parameters, ReadOnlySpan<T> gradient, Span<T> first, Span<T> second,
        T learningRate, double beta1, double beta2, T epsilon, T correctionNext, T correctionNow, T correction2)
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
            T lookahead = firstDecay * m / correctionNext + firstWeight * g / correctionNow;
            parameters[i] -= learningRate * lookahead / (T.Sqrt(v / correction2) + epsilon);
        }
    }

    private protected override void ResetState()
    {
        Array.Clear(first);
        Array.Clear(second);
    }
}

/// <summary>
/// Updates a parameter set with the NAdam optimizer.
/// <para>NAdam でパラメーター一式を更新します。</para>
/// </summary>
/// <typeparam name="T">The element type (<see langword="float"/> or <see langword="double"/>). / 要素型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
public sealed class NAdam<T> : StochasticGradientOptimizer<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    private readonly double beta1;
    private readonly double beta2;
    private readonly T epsilon;
    private readonly T[][] first;
    private readonly T[][] second;

    /// <summary>
    /// Initializes an optimizer bound to the specified parameter set.
    /// <para>指定したパラメーター一式を更新するオプティマイザーを初期化します。</para>
    /// </summary>
    /// <param name="parameters">The parameters to update. / 更新するパラメーター一式。</param>
    /// <param name="options">Optimizer settings, or <see langword="null"/> for the defaults. / 設定。<see langword="null"/> の場合は既定値を使用します。</param>
    /// <exception cref="ArgumentNullException"><paramref name="parameters"/> is <see langword="null"/>. / <paramref name="parameters"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option is out of range. / 設定値が有効範囲外の場合。</exception>
    /// <exception cref="NotSupportedException">The element type is neither <see langword="float"/> nor <see langword="double"/>. / 要素型が <see langword="float"/> でも <see langword="double"/> でもない場合。</exception>
    public NAdam(ParameterSet<T> parameters, NAdamOptions? options = null)
        : base(parameters, options ??= new NAdamOptions())
    {
        CheckCoefficient(options.Beta1, nameof(options.Beta1));
        CheckCoefficient(options.Beta2, nameof(options.Beta2));
        CheckEpsilon(options.Epsilon, nameof(options.Epsilon));
        beta1 = options.Beta1;
        beta2 = options.Beta2;
        epsilon = T.CreateTruncating(options.Epsilon);
        first = AllocateState();
        second = AllocateState();
    }

    private protected override void Update(double rate, long step, GradientClip clip)
    {
        T learningRate = T.CreateTruncating(rate);
        double power = Math.Pow(beta1, step);
        T correctionNext = T.CreateTruncating(1 - power * beta1);
        T correctionNow = T.CreateTruncating(1 - power);
        T correction2 = T.CreateTruncating(1 - Math.Pow(beta2, step));
        for (int i = 0; i < Parameters.Count; i++)
        {
            var parameter = Parameters[i];
            var gradient = PrepareGradient(parameter, rate, clip);
            NAdam.UpdateCore<T>(parameter.Value, gradient, first[i], second[i],
                learningRate, beta1, beta2, epsilon, correctionNext, correctionNow, correction2);
        }
    }

    private protected override void ResetState()
    {
        ClearState(first);
        ClearState(second);
    }
}
