using Tuna.LinearAlgebra;

namespace Tuna.Optimization.Stochastic;

/// <summary>
/// Provides a base class for stochastic gradient optimizers that update parameters from supplied gradients.
/// <para>与えられた勾配からパラメーターを更新する確率的勾配法オプティマイザーの基底クラスです。</para>
/// </summary>
/// <remarks>
/// <para><see cref="Step"/> updates parameters in place; <see cref="Reset"/> clears optimizer state such as moments and step count. Updates are deterministic for identical sequences of values and gradients. Instances are not thread-safe and must not be called concurrently.</para>
/// <para><see cref="Step"/> はパラメーターをインプレースで更新し、<see cref="Reset"/> はモーメントやステップ回数などの内部状態を初期化します。同一の値・勾配列に対して更新結果は決定的に一致します。インスタンスはスレッドセーフではないため、複数スレッドから同時に呼び出さないでください。</para>
/// </remarks>
public abstract class StochasticGradientOptimizer
{
    private protected readonly int n;
    private readonly double[] work;
    private readonly double weightDecay;
    private readonly bool decoupledWeightDecay;
    private readonly double clipNorm;
    private double learningRate;

    private protected StochasticGradientOptimizer(int dimension, StochasticGradientOptions options)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(dimension, 1);
        ValidateOptions(options);

        n = dimension;
        work = new double[dimension];
        learningRate = options.LearningRate;
        Schedule = options.Schedule;
        weightDecay = options.WeightDecay;
        decoupledWeightDecay = options.DecoupledWeightDecay;
        clipNorm = options.GradientClipNorm ?? double.PositiveInfinity;
    }

    /// <summary>
    /// Gets the number of parameters.
    /// <para>パラメーターの数を取得します。</para>
    /// </summary>
    public int Dimension => n;

    /// <summary>
    /// Gets the number of updates performed so far (reset to 0 by <see cref="Reset"/>).
    /// <para>これまでに行った更新回数を取得します（<see cref="Reset"/> で 0 に戻ります）。</para>
    /// </summary>
    public long StepCount { get; private set; }

    /// <summary>
    /// Gets or sets the base learning rate (may be modified between steps).
    /// <para>基準学習率を取得または設定します（更新の合間に変更できます）。</para>
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">The assigned value is not positive and finite. / 設定した値が正の有限値でない場合。</exception>
    public double LearningRate
    {
        get => learningRate;
        set
        {
            CheckLearningRate(value, nameof(value));
            learningRate = value;
        }
    }

    /// <summary>
    /// Gets the learning-rate schedule, or <see langword="null"/> if constant.
    /// <para>学習率スケジュールを取得します（一定の場合は <see langword="null"/>）。</para>
    /// </summary>
    public ILearningRateSchedule? Schedule { get; }

    /// <summary>
    /// Gets the effective learning rate that will be used on the next <see cref="Step"/> (<see cref="LearningRate"/> multiplied by the schedule multiplier).
    /// <para>次の <see cref="Step"/> で使用される実効学習率（<see cref="LearningRate"/> × スケジュール倍率）を取得します。</para>
    /// </summary>
    public double CurrentLearningRate => learningRate * (Schedule?.Multiplier(StepCount) ?? 1.0);

    /// <summary>
    /// Performs a single in-place parameter update using the specified gradient.
    /// <para>指定した勾配を用いてパラメーターをインプレースで1回更新します。</para>
    /// </summary>
    /// <param name="parameters">The parameter vector to update in place (length <see cref="Dimension"/>). / インプレース更新対象のパラメーターベクトル（長さ <see cref="Dimension"/>）。</param>
    /// <param name="gradient">The (possibly stochastic) gradient at <paramref name="parameters"/> (length <see cref="Dimension"/>, non-overlapping with <paramref name="parameters"/>). / <paramref name="parameters"/> における（確率的）勾配ベクトル（長さ <see cref="Dimension"/>、<paramref name="parameters"/> と領域は非重複）。</param>
    /// <exception cref="ArgumentException">The length of <paramref name="parameters"/> or <paramref name="gradient"/> does not match <see cref="Dimension"/>. / <paramref name="parameters"/> または <paramref name="gradient"/> の長さが <see cref="Dimension"/> と一致しない場合。</exception>
    /// <exception cref="InvalidOperationException">The learning-rate schedule returned a negative or non-finite multiplier. / 学習率スケジュールが負または有限でない倍率を返した場合。</exception>
    public void Step(VectorSpan<double> parameters, ReadOnlyVectorSpan<double> gradient)
    {
        if (parameters.Length != n)
        {
            throw new ArgumentException($"The parameter length must be {n}.", nameof(parameters));
        }

        if (gradient.Length != n)
        {
            throw new ArgumentException($"The gradient length must be {n}.", nameof(gradient));
        }

        double eta = CurrentLearningRate;
        if (!(eta >= 0 && double.IsFinite(eta)))
        {
            throw new InvalidOperationException($"The schedule gave the learning rate {eta} at step {StepCount}.");
        }

        double squared = 0;
        for (int i = 0; i < n; i++)
        {
            double gi = gradient[i];
            work[i] = gi;
            squared += gi * gi;
        }

        if (!GradientClip.TryCreate(squared, clipNorm, out GradientClip clip))
        {
            double max = GradientClip.MaxAbs<double>(work, 0);
            if (max > 0)
            {
                double sum = double.IsFinite(max) ? GradientClip.ScaledSumOfSquares<double>(work, max, 0) : 0;
                clip = GradientClip.CreateScaled(max, sum, clipNorm);
            }
        }

        if (clip.IsActive)
        {
            for (int i = 0; i < n; i++)
            {
                work[i] = clip.Apply(work[i]);
            }
        }

        if (weightDecay > 0)
        {
            if (decoupledWeightDecay)
            {
                double keep = 1 - eta * weightDecay;
                for (int i = 0; i < n; i++)
                {
                    parameters[i] *= keep;
                }
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    work[i] += weightDecay * parameters[i];
                }
            }
        }

        StepCount++;
        Update(parameters, work, eta, StepCount);
    }

    /// <summary>
    /// Resets the step count and all internal optimizer state (such as moment estimates).
    /// <para>更新回数とオプティマイザーの内部状態（モーメントなど）を初期化します。</para>
    /// </summary>
    public void Reset()
    {
        StepCount = 0;
        ResetState();
    }

    /// <summary>
    /// Minimizes a delegate objective function for up to <paramref name="maxIterations"/> updates.
    /// <para>デリゲート目的関数を評価しながら、最大 <paramref name="maxIterations"/> 回の更新で最小化します。</para>
    /// </summary>
    /// <param name="objective">The objective delegate returning the scalar value and writing the gradient to its second parameter. / 目的関数値を返し、第2引数に勾配を書き込むデリゲート。</param>
    /// <param name="initialPoint">A finite initial point of length <see cref="Dimension"/>. / 長さ <see cref="Dimension"/> の有限の初期点。</param>
    /// <param name="maxIterations">The maximum number of parameter updates (non-negative). / 更新回数の上限（0 以上）。</param>
    /// <param name="gradientTolerance">Stops with <see cref="MinimizationTermination.GradientTolerance"/> when the infinity norm of the gradient is at most this value (non-negative; default is 0). / 勾配の最大ノルムがこの値以下になったときに <see cref="MinimizationTermination.GradientTolerance"/> で終了します（0 以上、既定値は 0）。</param>
    /// <param name="cancellationToken">A cancellation token checked before each objective evaluation. / 各目的関数評価の前に確認されるキャンセルトークン。</param>
    /// <returns>A result containing the point satisfying <paramref name="gradientTolerance"/> or otherwise the best evaluated point; terminates with <see cref="MinimizationTermination.NonFiniteValue"/> if any value or gradient is non-finite. / 勾配許容誤差を満たした点（未達の場合は評価した中で目的関数値が最小の点）を含む結果。非有限値が生じた場合は <see cref="MinimizationTermination.NonFiniteValue"/> で終了します。</returns>
    /// <remarks>
    /// <para>Calls <see cref="Reset"/> before starting minimization.</para>
    /// <para>最小化の開始前に <see cref="Reset"/> を呼び出して内部状態を初期化します。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="objective"/> is <see langword="null"/>. / <paramref name="objective"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoint"/> does not have length <see cref="Dimension"/> or contains a non-finite value. / <paramref name="initialPoint"/> の長さが <see cref="Dimension"/> と一致しない、または有限でない値を含む場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxIterations"/> or <paramref name="gradientTolerance"/> is negative (or <see cref="double.NaN"/>). / <paramref name="maxIterations"/> または <paramref name="gradientTolerance"/> が負（または <see cref="double.NaN"/>）の場合。</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was canceled. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public MinimizationResult Minimize(Func<ReadOnlyVectorSpan<double>, VectorSpan<double>, double> objective, ReadOnlySpan<double> initialPoint,
        int maxIterations, double gradientTolerance = 0, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(objective);
        return Minimize(new DelegateObjective(objective), initialPoint, maxIterations, gradientTolerance, cancellationToken);
    }

    /// <summary>
    /// Minimizes a differentiable objective function for up to <paramref name="maxIterations"/> updates.
    /// <para>微分可能な目的関数を評価しながら、最大 <paramref name="maxIterations"/> 回の更新で最小化します。</para>
    /// </summary>
    /// <typeparam name="TObjective">The differentiable objective function type. / 微分可能な目的関数の型。</typeparam>
    /// <param name="objective">The objective function computing value and gradient. / 目的関数値と勾配を計算する目的関数。</param>
    /// <param name="initialPoint">A finite initial point of length <see cref="Dimension"/>. / 長さ <see cref="Dimension"/> の有限の初期点。</param>
    /// <param name="maxIterations">The maximum number of parameter updates (non-negative). / 更新回数の上限（0 以上）。</param>
    /// <param name="gradientTolerance">Stops with <see cref="MinimizationTermination.GradientTolerance"/> when the infinity norm of the gradient is at most this value (non-negative; default is 0). / 勾配の最大ノルムがこの値以下になったときに <see cref="MinimizationTermination.GradientTolerance"/> で終了します（0 以上、既定値は 0）。</param>
    /// <param name="cancellationToken">A cancellation token checked before each objective evaluation. / 各目的関数評価の前に確認されるキャンセルトークン。</param>
    /// <returns>A result containing the point satisfying <paramref name="gradientTolerance"/> or otherwise the best evaluated point; terminates with <see cref="MinimizationTermination.NonFiniteValue"/> if any value or gradient is non-finite. / 勾配許容誤差を満たした点（未達の場合は評価した中で目的関数値が最小の点）を含む結果。非有限値が生じた場合は <see cref="MinimizationTermination.NonFiniteValue"/> で終了します。</returns>
    /// <remarks>
    /// <para>Calls <see cref="Reset"/> before starting minimization.</para>
    /// <para>最小化の開始前に <see cref="Reset"/> を呼び出して内部状態を初期化します。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="objective"/> is <see langword="null"/>. / <paramref name="objective"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoint"/> does not have length <see cref="Dimension"/> or contains a non-finite value. / <paramref name="initialPoint"/> の長さが <see cref="Dimension"/> と一致しない、または有限でない値を含む場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxIterations"/> or <paramref name="gradientTolerance"/> is negative (or <see cref="double.NaN"/>). / <paramref name="maxIterations"/> または <paramref name="gradientTolerance"/> が負（または <see cref="double.NaN"/>）の場合。</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was canceled. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public MinimizationResult Minimize<TObjective>(TObjective objective, ReadOnlySpan<double> initialPoint,
        int maxIterations, double gradientTolerance = 0, CancellationToken cancellationToken = default)
        where TObjective : IDifferentiableObjectiveFunction
    {
        ArgumentNullException.ThrowIfNull(objective);
        if (initialPoint.Length != n)
        {
            throw new ArgumentException($"The initial point length must be {n}.", nameof(initialPoint));
        }

        foreach (double v in initialPoint)
        {
            if (!double.IsFinite(v))
            {
                throw new ArgumentException("The initial point must be finite.", nameof(initialPoint));
            }
        }

        ArgumentOutOfRangeException.ThrowIfNegative(maxIterations);
        if (!(gradientTolerance >= 0))
        {
            throw new ArgumentOutOfRangeException(nameof(gradientTolerance), "The gradient tolerance must be non-negative.");
        }

        Reset();
        var x = initialPoint.ToArray();
        var gradient = new double[n];
        var best = new double[n];
        double bestValue = double.NaN, bestNorm = double.NaN;
        long evaluations = 0;
        var termination = MinimizationTermination.MaxIterations;
        for (int k = 0; ; k++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Array.Clear(gradient);
            double value = objective.Evaluate(x, gradient);
            evaluations++;
            double norm = 0;
            foreach (double gi in gradient)
            {
                norm = Math.Max(norm, Math.Abs(gi));
            }

            if (!double.IsFinite(value) || !double.IsFinite(norm))
            {
                if (evaluations == 1)
                {
                    x.CopyTo(best, 0);
                    bestValue = value;
                    bestNorm = double.NaN;
                }

                termination = MinimizationTermination.NonFiniteValue;
                break;
            }

            if (norm <= gradientTolerance || !(value >= bestValue))
            {
                x.CopyTo(best, 0);
                bestValue = value;
                bestNorm = norm;
            }

            if (norm <= gradientTolerance)
            {
                termination = MinimizationTermination.GradientTolerance;
                break;
            }

            if (k >= maxIterations)
            {
                break;
            }

            Step(x, gradient);
        }

        return new MinimizationResult(new Vector<double>(best), bestValue, bestNorm, (int)Math.Min(StepCount, int.MaxValue), evaluations, termination);
    }

    private protected abstract void Update(VectorSpan<double> parameters, ReadOnlySpan<double> gradient, double learningRate, long step);

    private protected abstract void ResetState();

    internal static void ValidateOptions(StochasticGradientOptions options)
    {
        CheckLearningRate(options.LearningRate, nameof(options.LearningRate));
        if (!(options.WeightDecay >= 0 && double.IsFinite(options.WeightDecay)))
        {
            throw new ArgumentOutOfRangeException(nameof(options.WeightDecay), "The weight decay must be non-negative and finite.");
        }

        if (options.GradientClipNorm is { } clip && !(clip > 0))
        {
            throw new ArgumentOutOfRangeException(nameof(options.GradientClipNorm), "The clipping norm must be positive.");
        }
    }

    internal static void CheckLearningRate(double value, string name)
    {
        if (!(value > 0 && double.IsFinite(value)))
        {
            throw new ArgumentOutOfRangeException(name, "The learning rate must be positive and finite.");
        }
    }

    internal static void CheckCoefficient(double value, string name)
    {
        if (!(value >= 0 && value < 1))
        {
            throw new ArgumentOutOfRangeException(name, "The coefficient must lie in [0, 1).");
        }
    }

    internal static void CheckEpsilon(double value, string name)
    {
        if (!(value > 0 && double.IsFinite(value)))
        {
            throw new ArgumentOutOfRangeException(name, "The epsilon must be positive and finite.");
        }
    }
}
