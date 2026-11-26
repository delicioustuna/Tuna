using Tuna.LinearAlgebra;

namespace Tuna.Optimization.Gradient;

/// <summary>
/// Provides a base class for gradient-based minimizers using line search.
/// <para>勾配と直線探索を用いて目的関数を最小化するオプティマイザーの基底クラスです。</para>
/// </summary>
/// <remarks>
/// <para>For ask-tell external evaluation, initialize with <see cref="Start"/> and feed values and gradients for each point returned by <see cref="Ask"/> into <see cref="Tell"/>. To evaluate an objective directly, call <see cref="Minimize{TObjective}(TObjective, ReadOnlySpan{double}, CancellationToken)"/>. If the initial point yields a non-finite value or gradient, minimization terminates with <see cref="MinimizationTermination.NonFiniteValue"/>; subsequent non-finite trials shrink the step length. The last accepted iterate is always the best evaluated point. Convergence is checked after each accepted step in the order: gradient norm, relative function change, and relative step size. Instances are not thread-safe and must not be called concurrently.</para>
/// <para>外部で評価を行う場合は <see cref="Start"/> で初期化し、<see cref="Ask"/> が返す各試行点の目的関数値と勾配を <see cref="Tell"/> に渡します。目的関数を直接渡す場合は <see cref="Minimize{TObjective}(TObjective, ReadOnlySpan{double}, CancellationToken)"/> を使用します。初期点の評価値または勾配が有限値でない場合は <see cref="MinimizationTermination.NonFiniteValue"/> で終了し、それ以降の試行点で有限値が得られない場合はステップ幅を縮小します。最後に採用した反復点が常に最良点になります。各採用ステップ後に勾配ノルム、目的関数値の相対変化、相対移動量の順で収束を判定します。インスタンスはスレッドセーフではないため、複数スレッドから同時に呼び出さないでください。</para>
/// </remarks>
public abstract class LineSearchMinimizer
{
    private protected const double UnboundedStep = double.PositiveInfinity;

    private protected readonly int n;
    private readonly int maxIterations;
    private readonly long maxEvaluations;
    private readonly double gradientTolerance;
    private readonly double functionTolerance;
    private readonly double? stepTolerance;
    private readonly int maxLineSearchEvaluations;
    private readonly MoreThuenteLineSearch lineSearch;

    private protected double[] x;
    private protected double[] g;
    private protected double f;
    private protected readonly double[] direction;
    private double[] trial;
    private double[] trialGradient;
    private double trialValue;
    private double[] best;
    private double[] bestGradient;
    private double bestValue;
    private readonly double[] stepBuffer;
    private readonly double[] gradientChange;
    private double gradientNorm = double.NaN;
    private IEnumerator<bool>? routine;
    private bool awaiting;
    private bool hasPoint;

    private protected LineSearchMinimizer(int dimension, LbfgsOptions options)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(dimension, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.HistorySize, 1, nameof(options.HistorySize));
        if (!(options.GradientTolerance >= 0))
        {
            throw new ArgumentOutOfRangeException(nameof(options.GradientTolerance), "The gradient tolerance must be non-negative.");
        }

        if (!(options.FunctionTolerance >= 0))
        {
            throw new ArgumentOutOfRangeException(nameof(options.FunctionTolerance), "The function tolerance must be non-negative.");
        }

        if (options.StepTolerance is { } xtol && !(xtol >= 0))
        {
            throw new ArgumentOutOfRangeException(nameof(options.StepTolerance), "The step tolerance must be non-negative.");
        }

        ArgumentOutOfRangeException.ThrowIfLessThan(options.MaxIterations, 1, nameof(options.MaxIterations));
        if (options.MaxEvaluations is { } maxEval)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(maxEval, 1L, nameof(options.MaxEvaluations));
        }

        if (!(options.SufficientDecrease > 0 && options.SufficientDecrease < 0.5))
        {
            throw new ArgumentOutOfRangeException(nameof(options.SufficientDecrease), "The sufficient decrease coefficient must lie in (0, 1/2).");
        }

        if (!(options.Curvature > options.SufficientDecrease && options.Curvature < 1))
        {
            throw new ArgumentOutOfRangeException(nameof(options.Curvature), "The curvature coefficient must lie in (SufficientDecrease, 1).");
        }

        ArgumentOutOfRangeException.ThrowIfLessThan(options.MaxLineSearchEvaluations, 1, nameof(options.MaxLineSearchEvaluations));

        n = dimension;
        maxIterations = options.MaxIterations;
        maxEvaluations = options.MaxEvaluations ?? long.MaxValue;
        gradientTolerance = options.GradientTolerance;
        functionTolerance = options.FunctionTolerance;
        stepTolerance = options.StepTolerance;
        maxLineSearchEvaluations = options.MaxLineSearchEvaluations;
        lineSearch = new MoreThuenteLineSearch(options.SufficientDecrease, options.Curvature);
        x = new double[n];
        g = new double[n];
        direction = new double[n];
        trial = new double[n];
        trialGradient = new double[n];
        best = new double[n];
        bestGradient = new double[n];
        stepBuffer = new double[n];
        gradientChange = new double[n];
    }

    /// <summary>
    /// Gets the number of variables.
    /// <para>変数の数を取得します。</para>
    /// </summary>
    public int Dimension => n;

    /// <summary>
    /// Gets the number of completed iterations (accepted steps).
    /// <para>反復回数（採用したステップ数）を取得します。</para>
    /// </summary>
    public int Iterations { get; private set; }

    /// <summary>
    /// Gets the total number of objective evaluations (calls to <see cref="Tell"/>).
    /// <para>目的関数の評価回数（<see cref="Tell"/> の呼び出し回数）を取得します。</para>
    /// </summary>
    public long Evaluations { get; private set; }

    /// <summary>
    /// Gets a value indicating whether minimization has terminated.
    /// <para>最小化が終了したかどうかを取得します。</para>
    /// </summary>
    /// <value><see langword="true"/> if minimization has terminated; otherwise, <see langword="false"/>. Call <see cref="Start"/> to begin a new run. / 終了している場合は <see langword="true"/>、それ以外は <see langword="false"/>。新しく開始するには <see cref="Start"/> を呼び出します。</value>
    public bool IsFinished => Termination != MinimizationTermination.None;

    /// <summary>
    /// Gets the reason minimization terminated.
    /// <para>最小化の終了理由を取得します。</para>
    /// </summary>
    /// <value>The termination reason, or <see cref="MinimizationTermination.None"/> if minimization has not finished. / 終了理由。終了前は <see cref="MinimizationTermination.None"/>。</value>
    public MinimizationTermination Termination { get; private set; }

    /// <summary>
    /// Gets the current iterate.
    /// <para>現在の反復点を取得します。</para>
    /// </summary>
    /// <value>A view of the current iterate (the final iterate after termination), valid until the next call to <see cref="Tell"/>. / 現在の反復点へのビュー（終了後は最終反復点）。次の <see cref="Tell"/> 呼び出しまで有効です。</value>
    /// <exception cref="InvalidOperationException">The initial point has not been evaluated yet. / 初期点の評価がまだ完了していない場合。</exception>
    public ReadOnlyVectorSpan<double> Point => hasPoint ? x : throw NotEvaluated();

    /// <summary>
    /// Gets the objective value at the current iterate.
    /// <para>現在の反復点における目的関数値を取得します。</para>
    /// </summary>
    /// <exception cref="InvalidOperationException">The initial point has not been evaluated yet. / 初期点の評価がまだ完了していない場合。</exception>
    public double Value => hasPoint ? f : throw NotEvaluated();

    /// <summary>
    /// Gets the gradient at the current iterate.
    /// <para>現在の反復点における勾配を取得します。</para>
    /// </summary>
    /// <value>A view of the gradient, valid until the next call to <see cref="Tell"/>. / 勾配へのビュー。次の <see cref="Tell"/> 呼び出しまで有効です。</value>
    /// <exception cref="InvalidOperationException">The initial point has not been evaluated yet. / 初期点の評価がまだ完了していない場合。</exception>
    public ReadOnlyVectorSpan<double> Gradient => hasPoint ? g : throw NotEvaluated();

    /// <summary>
    /// Gets the infinity norm of the gradient (or projected gradient <c>x − P(x − ∇f)</c> for <see cref="LbfgsB"/>) at the current iterate.
    /// <para>現在の反復点における勾配（<see cref="LbfgsB"/> では射影勾配 <c>x − P(x − ∇f)</c>）の最大ノルムを取得します。</para>
    /// </summary>
    /// <exception cref="InvalidOperationException">The initial point has not been evaluated yet. / 初期点の評価がまだ完了していない場合。</exception>
    public double GradientNorm => hasPoint ? gradientNorm : throw NotEvaluated();

    /// <summary>
    /// Starts a new minimization run from the specified initial point.
    /// <para>指定した初期点から新しい最小化を開始します。</para>
    /// </summary>
    /// <param name="initialPoint">A finite initial point of length <see cref="Dimension"/>. Projected onto the bounds by <see cref="LbfgsB"/>. / 長さが <see cref="Dimension"/> に等しい有限の初期点。<see cref="LbfgsB"/> では範囲制約内に射影されます。</param>
    /// <remarks>
    /// <para>Resets any in-progress state and clears the curvature history.</para>
    /// <para>進行中の反復状態と曲率履歴はすべてリセットされます。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="initialPoint"/> does not have length <see cref="Dimension"/> or contains a non-finite value. / <paramref name="initialPoint"/> の長さが <see cref="Dimension"/> と一致しない、または有限でない成分を含む場合。</exception>
    public void Start(ReadOnlySpan<double> initialPoint)
    {
        if (initialPoint.Length != n)
        {
            throw new ArgumentException($"The initial point must have length {n}.", nameof(initialPoint));
        }

        foreach (double value in initialPoint)
        {
            if (!double.IsFinite(value))
            {
                throw new ArgumentException("The initial point must be finite.", nameof(initialPoint));
            }
        }

        initialPoint.CopyTo(trial);
        Prepare(trial);
        ClearHistory();
        Iterations = 0;
        Evaluations = 0;
        Termination = MinimizationTermination.None;
        hasPoint = false;
        gradientNorm = double.NaN;
        awaiting = false;
        routine = Run().GetEnumerator();
        Resume();
    }

    /// <summary>
    /// Returns the next trial point to evaluate.
    /// <para>次に評価する試行点を取得します。</para>
    /// </summary>
    /// <returns>A view of the next trial point, valid until the next call to <see cref="Tell"/>. Repeated calls before <see cref="Tell"/> return the same point. / 次に評価する点へのビュー。次の <see cref="Tell"/> 呼び出しまで有効で、その間の再呼び出しでは同じ点を返します。</returns>
    /// <exception cref="InvalidOperationException"><see cref="Start"/> has not been called, or minimization has already terminated. / <see cref="Start"/> が呼ばれていない、または既に終了している場合。</exception>
    public ReadOnlyVectorSpan<double> Ask()
    {
        if (!awaiting)
        {
            throw new InvalidOperationException(IsFinished ? "The minimizer has finished." : "Call Start first.");
        }

        return trial;
    }

    /// <summary>
    /// Reports the objective value and gradient for the trial point returned by <see cref="Ask"/> and advances the solver.
    /// <para><see cref="Ask"/> が返した試行点での目的関数値と勾配を報告し、探索を1ステップ進めます。</para>
    /// </summary>
    /// <param name="value">The objective value. <see cref="double.NaN"/> and infinities are treated as failed evaluations (terminating immediately on the initial point). / 目的関数値。<see cref="double.NaN"/> や無限大は評価失敗として扱われます（初期点の場合は即座に終了します）。</param>
    /// <param name="gradient">The gradient vector of length <see cref="Dimension"/> (copied internally). / 長さが <see cref="Dimension"/> の勾配ベクトル（内部にコピーされます）。</param>
    /// <exception cref="InvalidOperationException"><see cref="Start"/> has not been called, or minimization has already terminated. / <see cref="Start"/> が呼ばれていない、または既に終了している場合。</exception>
    /// <exception cref="ArgumentException">The length of <paramref name="gradient"/> does not match <see cref="Dimension"/>. / <paramref name="gradient"/> の長さが <see cref="Dimension"/> と一致しない場合。</exception>
    public void Tell(double value, ReadOnlyVectorSpan<double> gradient)
    {
        if (!awaiting)
        {
            throw new InvalidOperationException(IsFinished ? "The minimizer has finished." : "Call Start first.");
        }

        if (gradient.Length != n)
        {
            throw new ArgumentException($"The gradient must have length {n}.", nameof(gradient));
        }

        gradient.CopyTo(trialGradient);
        Accept(value);
    }

    /// <summary>
    /// Minimizes the specified delegate objective function starting from <paramref name="initialPoint"/>.
    /// <para>指定した初期点からデリゲート目的関数を最小化します。</para>
    /// </summary>
    /// <param name="objective">The objective delegate returning the scalar value and writing the gradient to its second parameter. / 目的関数値を返し、第2引数に勾配を書き込むデリゲート。</param>
    /// <param name="initialPoint">A finite initial point of length <see cref="Dimension"/>. / 長さが <see cref="Dimension"/> に等しい有限の初期点。</param>
    /// <param name="cancellationToken">A cancellation token checked before each objective evaluation. / 各目的関数評価の前に確認されるキャンセルトークン。</param>
    /// <returns>The minimization result containing a copy of the final iterate, objective value, gradient norm, iteration and evaluation counts, and termination reason. / 最終反復点のコピー、目的関数値、勾配ノルム、反復回数・評価回数、終了理由を含む最小化結果。</returns>
    /// <remarks>
    /// <para>Exceptions thrown by <paramref name="objective"/> are propagated directly; the pending trial point remains active so execution can be resumed with <see cref="Ask"/> and <see cref="Tell"/>.</para>
    /// <para><paramref name="objective"/> から送出された例外はそのまま伝播します。評価待ちの試行点は保持されるため、<see cref="Ask"/> と <see cref="Tell"/> で探索を継続できます。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="objective"/> is <see langword="null"/>. / <paramref name="objective"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoint"/> does not have length <see cref="Dimension"/> or contains a non-finite value. / <paramref name="initialPoint"/> の長さが <see cref="Dimension"/> と一致しない、または有限でない成分を含む場合。</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was canceled. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public MinimizationResult Minimize(Func<ReadOnlyVectorSpan<double>, VectorSpan<double>, double> objective, ReadOnlySpan<double> initialPoint,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(objective);
        return Minimize(new DelegateObjective(objective), initialPoint, cancellationToken);
    }

    /// <summary>
    /// Minimizes the specified differentiable objective function starting from <paramref name="initialPoint"/>.
    /// <para>指定した初期点から微分可能な目的関数を最小化します。</para>
    /// </summary>
    /// <typeparam name="TObjective">The differentiable objective function type. / 微分可能な目的関数の型。</typeparam>
    /// <param name="objective">The objective function computing both value and gradient. / 評価点で目的関数値と勾配を計算する目的関数。</param>
    /// <param name="initialPoint">A finite initial point of length <see cref="Dimension"/>. / 長さが <see cref="Dimension"/> に等しい有限の初期点。</param>
    /// <param name="cancellationToken">A cancellation token checked before each objective evaluation. / 各目的関数評価の前に確認されるキャンセルトークン。</param>
    /// <returns>The minimization result containing a copy of the final iterate, objective value, gradient norm, iteration and evaluation counts, and termination reason. / 最終反復点のコピー、目的関数値、勾配ノルム、反復回数・評価回数、終了理由を含む最小化結果。</returns>
    /// <remarks>
    /// <para>Exceptions thrown by <paramref name="objective"/> are propagated directly; the pending trial point remains active so execution can be resumed with <see cref="Ask"/> and <see cref="Tell"/>.</para>
    /// <para><paramref name="objective"/> から送出された例外はそのまま伝播します。評価待ちの試行点は保持されるため、<see cref="Ask"/> と <see cref="Tell"/> で探索を継続できます。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="objective"/> is <see langword="null"/>. / <paramref name="objective"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoint"/> does not have length <see cref="Dimension"/> or contains a non-finite value. / <paramref name="initialPoint"/> の長さが <see cref="Dimension"/> と一致しない、または有限でない成分を含む場合。</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was canceled. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public MinimizationResult Minimize<TObjective>(TObjective objective, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)
        where TObjective : IDifferentiableObjectiveFunction
    {
        ArgumentNullException.ThrowIfNull(objective);
        Start(initialPoint);
        while (awaiting)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Array.Clear(trialGradient);
            double value = objective.Evaluate(trial, trialGradient);
            Accept(value);
        }

        return GetResult();
    }

    /// <summary>
    /// Gets a snapshot of the current minimization result.
    /// <para>現在の最小化結果のスナップショットを取得します。</para>
    /// </summary>
    /// <returns>A minimization result containing a copy of the current iterate, value, gradient norm, counts, and termination reason (<see cref="MinimizationTermination.None"/> if still running). / 現在の反復点のコピー、目的関数値、勾配ノルム、反復・評価回数、および終了理由（実行中の場合は <see cref="MinimizationTermination.None"/>）を含む最小化結果。</returns>
    /// <exception cref="InvalidOperationException">The initial point has not been evaluated yet. / 初期点の評価がまだ完了していない場合。</exception>
    public MinimizationResult GetResult()
    {
        if (!hasPoint)
        {
            throw NotEvaluated();
        }

        return new MinimizationResult(new Vector<double>(x), f, gradientNorm, Iterations, Evaluations, Termination);
    }

    private protected virtual void Prepare(Span<double> point)
    {
    }

    private protected abstract bool HasHistory { get; }

    private protected abstract void ClearHistory();

    private protected abstract void UpdateHistory(ReadOnlySpan<double> s, ReadOnlySpan<double> y);

    private protected abstract bool ComputeDirection(Span<double> d);

    private protected virtual double MaxStep(ReadOnlySpan<double> d) => UnboundedStep;

    private protected virtual void TrialPoint(double alpha, ReadOnlySpan<double> d, Span<double> destination)
    {
        for (int i = 0; i < n; i++)
        {
            destination[i] = x[i] + alpha * d[i];
        }
    }

    private protected virtual double ComputeGradientNorm()
    {
        double norm = 0;
        for (int i = 0; i < n; i++)
        {
            norm = Math.Max(norm, Math.Abs(g[i]));
        }

        return norm;
    }

    private void Accept(double value)
    {
        trialValue = value;
        Evaluations++;
        awaiting = false;
        Resume();
    }

    private void Resume()
    {
        if (routine!.MoveNext())
        {
            awaiting = true;
        }
        else
        {
            routine.Dispose();
            routine = null;
        }
    }

    private static InvalidOperationException NotEvaluated() => new("The initial point has not been evaluated yet.");

    private static bool AllFinite(ReadOnlySpan<double> values)
    {
        foreach (double v in values)
        {
            if (!double.IsFinite(v))
            {
                return false;
            }
        }

        return true;
    }

    // 1つの評価を求めるたびに yield する手順。trial に評価してほしい点を置いてから yield し、再開後に trialValue・trialGradient を読む
    private IEnumerable<bool> Run()
    {
        yield return true;
        (x, trial) = (trial, x);
        (g, trialGradient) = (trialGradient, g);
        f = trialValue;
        hasPoint = true;
        if (!double.IsFinite(f) || !AllFinite(g))
        {
            gradientNorm = double.NaN;
            Termination = MinimizationTermination.NonFiniteValue;
            yield break;
        }

        gradientNorm = ComputeGradientNorm();
        if (gradientNorm <= gradientTolerance)
        {
            Termination = MinimizationTermination.GradientTolerance;
            yield break;
        }

        while (true)
        {
            if (Iterations >= maxIterations)
            {
                Termination = MinimizationTermination.MaxIterations;
                yield break;
            }

            bool hasDirection = ComputeDirection(direction);
            double slope = hasDirection ? Blas.Dot(g, direction) : double.NaN;
            if (!(slope < 0))
            {
                if (HasHistory)
                {
                    ClearHistory();
                    continue;
                }

                Termination = MinimizationTermination.LineSearchFailed;
                yield break;
            }

            double maxStep = MaxStep(direction);
            double initialStep = HasHistory ? 1.0 : 1.0 / Blas.Norm2(direction);
            lineSearch.Start(f, slope, Math.Min(initialStep, maxStep), 0, maxStep);
            bool accepted = false;
            bool useBest = false;
            for (int trials = 0; ; trials++)
            {
                if (Evaluations >= maxEvaluations)
                {
                    Termination = MinimizationTermination.MaxEvaluations;
                    yield break;
                }

                if (trials >= maxLineSearchEvaluations)
                {
                    accepted = useBest = lineSearch.BestSatisfiesSufficientDecrease;
                    break;
                }

                TrialPoint(lineSearch.Step, direction, trial);
                yield return true;

                LineSearchStatus status;
                if (!double.IsFinite(trialValue) || !AllFinite(trialGradient))
                {
                    status = lineSearch.RejectNonFinite();
                }
                else
                {
                    status = lineSearch.Next(trialValue, Blas.Dot(trialGradient, direction));
                    if (lineSearch.BestIsLatest)
                    {
                        trial.CopyTo(best, 0);
                        trialGradient.CopyTo(bestGradient, 0);
                        bestValue = trialValue;
                    }
                }

                if (status is LineSearchStatus.Converged or LineSearchStatus.MaximumStep)
                {
                    accepted = true;
                    break;
                }

                if (status != LineSearchStatus.Continue)
                {
                    accepted = useBest = lineSearch.BestSatisfiesSufficientDecrease;
                    break;
                }
            }

            if (!accepted)
            {
                if (HasHistory)
                {
                    ClearHistory();
                    continue;
                }

                Termination = MinimizationTermination.LineSearchFailed;
                yield break;
            }

            double previous = f;
            if (useBest)
            {
                (best, trial) = (trial, best);
                (bestGradient, trialGradient) = (trialGradient, bestGradient);
                trialValue = bestValue;
            }

            double moved = 0, size = 0;
            for (int i = 0; i < n; i++)
            {
                stepBuffer[i] = trial[i] - x[i];
                gradientChange[i] = trialGradient[i] - g[i];
                moved = Math.Max(moved, Math.Abs(stepBuffer[i]));
                size = Math.Max(size, Math.Abs(trial[i]));
            }

            (x, trial) = (trial, x);
            (g, trialGradient) = (trialGradient, g);
            f = trialValue;
            Iterations++;
            UpdateHistory(stepBuffer, gradientChange);
            gradientNorm = ComputeGradientNorm();

            if (gradientNorm <= gradientTolerance)
            {
                Termination = MinimizationTermination.GradientTolerance;
                yield break;
            }

            if (functionTolerance > 0 && previous - f <= functionTolerance * Math.Max(Math.Max(Math.Abs(previous), Math.Abs(f)), 1.0))
            {
                Termination = MinimizationTermination.FunctionTolerance;
                yield break;
            }

            if (stepTolerance is { } xtol && moved <= xtol * Math.Max(1.0, size))
            {
                Termination = MinimizationTermination.StepTolerance;
                yield break;
            }
        }
    }
}
