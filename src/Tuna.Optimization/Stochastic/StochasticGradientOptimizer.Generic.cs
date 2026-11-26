using System.Numerics;
using Tuna.Differentiation;

namespace Tuna.Optimization.Stochastic;

/// <summary>
/// Updates a fixed set of learnable parameters using their accumulated gradients.
/// <para>逆伝播で加算された勾配を使用して、初期化時に指定した学習パラメーターを更新します。</para>
/// </summary>
/// <typeparam name="T">The parameter and optimizer state element type (<see langword="float"/> or <see langword="double"/>). / パラメーターとオプティマイザーの状態の要素型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
/// <remarks>
/// <para>Call <see cref="Step"/> after backpropagation. The optimizer and its parameters must be used sequentially.</para>
/// <para>逆伝播の完了後に <see cref="Step"/> を呼び出します。同じオプティマイザーと学習パラメーターは、複数のスレッドから同時に操作しないでください。</para>
/// </remarks>
public abstract class StochasticGradientOptimizer<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    private readonly T[] work;
    private readonly double coupledWeightDecay;
    private readonly double decoupledWeightDecay;
    private readonly double clipNorm;
    private double learningRate;

    private protected StochasticGradientOptimizer(ParameterSet<T> parameters, StochasticGradientOptions options)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        if (typeof(T) != typeof(float) && typeof(T) != typeof(double))
        {
            throw new NotSupportedException("Parameter optimizers support float and double.");
        }

        StochasticGradientOptimizer.ValidateOptions(options);
        Parameters = parameters;
        learningRate = options.LearningRate;
        Schedule = options.Schedule;
        coupledWeightDecay = options.DecoupledWeightDecay ? 0 : options.WeightDecay;
        decoupledWeightDecay = options.DecoupledWeightDecay ? options.WeightDecay : 0;
        clipNorm = options.GradientClipNorm ?? double.PositiveInfinity;

        int length = 0;
        if (coupledWeightDecay > 0 || double.IsFinite(clipNorm))
        {
            for (int i = 0; i < parameters.Count; i++)
            {
                length = Math.Max(length, parameters[i].Length);
            }
        }

        work = length == 0 ? [] : new T[length];
    }

    /// <summary>
    /// Gets the parameter set bound to this optimizer.
    /// <para>このオプティマイザーが更新するパラメーター一式を取得します。</para>
    /// </summary>
    public ParameterSet<T> Parameters { get; }

    /// <summary>
    /// Gets the total number of parameter elements.
    /// <para>パラメーターの総要素数を取得します。</para>
    /// </summary>
    public long Dimension => Parameters.Length;

    /// <summary>
    /// Gets the number of completed updates since initialization or the last reset.
    /// <para>初期化または直近のリセットからの更新回数を取得します。</para>
    /// </summary>
    public long StepCount { get; private set; }

    /// <summary>
    /// Gets or sets the base learning rate.
    /// <para>学習率スケジュールを適用する前の基準学習率を取得または設定します。</para>
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">The value is nonpositive or nonfinite. / 設定値が 0 以下または有限でない場合。</exception>
    public double LearningRate
    {
        get => learningRate;
        set
        {
            StochasticGradientOptimizer.CheckLearningRate(value, nameof(value));
            learningRate = value;
        }
    }

    /// <summary>
    /// Gets the learning rate schedule, or <see langword="null"/> for a constant learning rate.
    /// <para>学習率スケジュールを取得します。固定の学習率の場合は <see langword="null"/> です。</para>
    /// </summary>
    public ILearningRateSchedule? Schedule { get; }

    /// <summary>
    /// Gets the learning rate for the next update, including the schedule multiplier.
    /// <para>学習率スケジュールの倍率を反映した次回の学習率を取得します。</para>
    /// </summary>
    public double CurrentLearningRate => learningRate * (Schedule?.Multiplier(StepCount) ?? 1.0);

    /// <summary>
    /// Updates all bound parameter values in place using their accumulated gradients.
    /// <para>加算された勾配に基づき、パラメーターの値をまとめてインプレースで更新します。</para>
    /// </summary>
    /// <remarks>
    /// <para>When clipping is enabled, a single Euclidean norm across the full set determines the scale before any weight decay. Gradient buffers are preserved for explicit clearing with <see cref="ParameterSet{T}.ZeroGradients"/>.</para>
    /// <para>勾配クリッピングは重み減衰の適用前に行い、パラメーター一式の勾配の L2 ノルムから共通の倍率を求めます。勾配バッファは保持され、<see cref="ParameterSet{T}.ZeroGradients"/> で明示的にクリアできます。</para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">The effective learning rate is negative or rounds to a nonfinite value in <typeparamref name="T"/>. / 適用する学習率が負、または <typeparamref name="T"/> に丸めると有限でない場合。</exception>
    public void Step()
    {
        double rate = CurrentLearningRate;
        if (!(rate >= 0 && T.IsFinite(T.CreateTruncating(rate))))
        {
            throw new InvalidOperationException($"The learning rate {rate} at step {StepCount} is out of range.");
        }

        long step = checked(StepCount + 1);
        Update(rate, step, GetGradientClip());
        StepCount = step;
    }

    /// <summary>
    /// Resets the update count and optimizer state.
    /// <para>更新回数とモーメントなどのオプティマイザーの状態を初期化します。</para>
    /// </summary>
    public void Reset()
    {
        ResetState();
        StepCount = 0;
    }

    private protected abstract void Update(double rate, long step, GradientClip clip);

    private protected abstract void ResetState();

    private protected T[][] AllocateState()
    {
        var state = new T[Parameters.Count][];
        for (int i = 0; i < state.Length; i++)
        {
            int length = Parameters[i].Length;
            state[i] = length == 0 ? [] : new T[length];
        }

        return state;
    }

    private protected static void ClearState(T[][]? state)
    {
        if (state is not null)
        {
            foreach (var entries in state)
            {
                Array.Clear(entries);
            }
        }
    }

    private protected static void CheckCoefficient(double value, string name)
    {
        StochasticGradientOptimizer.CheckCoefficient(value, name);
        if (!(T.CreateTruncating(value) < T.One))
        {
            throw new ArgumentOutOfRangeException(name, "The coefficient must be less than 1 in the parameter element type.");
        }
    }

    private protected static void CheckEpsilon(double value, string name)
    {
        T rounded = T.CreateTruncating(value);
        if (!(rounded > T.Zero && T.IsFinite(rounded)))
        {
            throw new ArgumentOutOfRangeException(name, "The epsilon must be positive and finite in the parameter element type.");
        }
    }

    private protected ReadOnlySpan<T> PrepareGradient(Parameter<T> parameter, double rate, GradientClip clip)
    {
        var values = parameter.Value;
        var gradient = parameter.Gradient;
        if (decoupledWeightDecay > 0)
        {
            double keep = 1 - rate * decoupledWeightDecay;
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = T.CreateTruncating(double.CreateChecked(values[i]) * keep);
            }
        }

        if (coupledWeightDecay == 0 && !clip.IsActive)
        {
            return gradient;
        }

        for (int i = 0; i < gradient.Length; i++)
        {
            double adjusted = clip.Apply(double.CreateChecked(gradient[i]));
            if (coupledWeightDecay > 0)
            {
                adjusted += coupledWeightDecay * double.CreateChecked(values[i]);
            }

            work[i] = T.CreateTruncating(adjusted);
        }

        return work.AsSpan(0, gradient.Length);
    }

    private GradientClip GetGradientClip()
    {
        if (double.IsPositiveInfinity(clipNorm))
        {
            return default;
        }

        double squared = 0;
        for (int i = 0; i < Parameters.Count; i++)
        {
            squared = GradientClip.SumOfSquares<T>(Parameters[i].Gradient, squared);
        }

        if (GradientClip.TryCreate(squared, clipNorm, out GradientClip clip))
        {
            return clip;
        }

        double max = 0;
        for (int i = 0; i < Parameters.Count; i++)
        {
            max = GradientClip.MaxAbs<T>(Parameters[i].Gradient, max);
        }

        if (!(max > 0))
        {
            return default;
        }

        double scaled = 0;
        if (double.IsFinite(max))
        {
            for (int i = 0; i < Parameters.Count; i++)
            {
                scaled = GradientClip.ScaledSumOfSquares<T>(Parameters[i].Gradient, max, scaled);
            }
        }

        return GradientClip.CreateScaled(max, scaled, clipNorm);
    }
}
