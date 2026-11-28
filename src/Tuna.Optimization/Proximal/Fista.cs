using Tuna.LinearAlgebra;

namespace Tuna.Optimization.Proximal;

/// <summary>
/// Configures options for <see cref="Fista"/>.
/// <para><see cref="Fista"/> の設定オプションを表します。</para>
/// </summary>
/// <remarks>
/// <para>Option values are validated by the <see cref="Fista"/> constructor.</para>
/// <para>各設定値の妥当性は <see cref="Fista"/> のコンストラクターで検証されます。</para>
/// </remarks>
public sealed class FistaOptions
{
    /// <summary>
    /// Gets the convergence tolerance on the infinity norm of the gradient mapping <c>(y − x⁺) / t</c>.
    /// <para>勾配写像 <c>(y − x⁺) / t</c> の最大ノルムに対する収束許容誤差を取得します。</para>
    /// </summary>
    /// <value>A non-negative finite number. The default is <c>1e-6</c>. / 0 以上の有限値。既定値は <c>1e-6</c>。</value>
    public double GradientTolerance { get; init; } = 1e-6;

    /// <summary>
    /// Gets the maximum number of iterations.
    /// <para>反復回数の上限を取得します。</para>
    /// </summary>
    /// <value>An integer greater than or equal to 1. The default is 10000. / 1 以上の整数。既定値は 10000。</value>
    public int MaxIterations { get; init; } = 10000;

    /// <summary>
    /// Gets the optional maximum number of evaluations of the smooth term.
    /// <para>滑らかな項の評価回数の上限を取得します。</para>
    /// </summary>
    /// <value>An integer greater than or equal to 1, or <see langword="null"/> (the default) for no limit. / 1 以上の整数、または無制限を表す <see langword="null"/>（既定値）。</value>
    public long? MaxEvaluations { get; init; }

    /// <summary>
    /// Gets the optional initial proximal gradient step size.
    /// <para>近接勾配法の初期ステップ幅を取得します。</para>
    /// </summary>
    /// <value>A positive finite number, or <see langword="null"/> (the default) to estimate the inverse Lipschitz constant near the initial point. / 正の有限値、または初期点近傍で推定したリプシッツ定数の逆数を使用する <see langword="null"/>（既定値）。</value>
    /// <remarks>
    /// <para>The probe gradient evaluation used for estimation counts toward <see cref="MaxEvaluations"/> and is skipped if the budget is already exhausted after the initial evaluation.</para>
    /// <para>ステップ幅の推定に使用する勾配評価も <see cref="MaxEvaluations"/> に計上されます。初期点評価時点で上限に達している場合は推定を行いません。</para>
    /// </remarks>
    public double? InitialStep { get; init; }

    /// <summary>
    /// Gets the multiplicative step-size shrinkage factor used during backtracking.
    /// <para>バックトラッキング時にステップ幅へ掛ける縮小倍率を取得します。</para>
    /// </summary>
    /// <value>A value strictly between 0 and 1. The default is 0.5. / 0 より大きく 1 未満の値。既定値は 0.5。</value>
    public double BacktrackingFactor { get; init; } = 0.5;

    /// <summary>
    /// Gets a value indicating whether gradient-based adaptive momentum restart (O'Donoghue &amp; Candès 2015) is enabled.
    /// <para>勾配に基づく適応的再始動（O'Donoghue &amp; Candès 2015）を有効にするかどうかを取得します。</para>
    /// </summary>
    /// <value><see langword="true"/> (the default) to restart momentum when overshoot is detected; otherwise, <see langword="false"/>. / オーバーシュート検出時に運動量をリセットする場合は <see langword="true"/>（既定値）、無効にする場合は <see langword="false"/>。</value>
    public bool AdaptiveRestart { get; init; } = true;
}

/// <summary>
/// Minimizes the sum of a smooth convex function and a proximable convex function using FISTA.
/// <para>FISTA を用いて、滑らかな凸関数と近接写像を持つ凸関数の和を最小化します。</para>
/// </summary>
/// <remarks>
/// <para>Constrained problems can be expressed by using a set indicator as the proximal term. For composite penalties involving a linear transform <c>h(Dx)</c>, use <see cref="Admm"/>. Instances are not thread-safe and must not be called concurrently.</para>
/// <para>制約付き最適化は、非平滑項に集合の指示関数（近接写像が射影）を指定することで表現できます。線形変換を含む <c>h(Dx)</c> 形式には <see cref="Admm"/> を使用してください。インスタンスはスレッドセーフではないため、複数スレッドから同時に呼び出さないでください。</para>
/// </remarks>
public sealed class Fista
{
    private readonly double gradientTolerance;
    private readonly int maxIterations;
    private readonly long maxEvaluations;
    private readonly double? initialStep;
    private readonly double backtracking;
    private readonly bool restart;

    /// <summary>
    /// Initializes a new FISTA solver with the specified options.
    /// <para>指定した設定で FISTA ソルバーを初期化します。</para>
    /// </summary>
    /// <param name="options">Optional solver configuration, or <see langword="null"/> for defaults. / オプション設定。<see langword="null"/> の場合は既定値を使用します。</param>
    /// <exception cref="ArgumentOutOfRangeException">An option value in <paramref name="options"/> is out of range. / <paramref name="options"/> の設定値が有効範囲外の場合。</exception>
    public Fista(FistaOptions? options = null)
    {
        options ??= new FistaOptions();
        if (!(options.GradientTolerance >= 0))
        {
            throw new ArgumentOutOfRangeException(nameof(options.GradientTolerance), "The gradient tolerance must be non-negative.");
        }

        ArgumentOutOfRangeException.ThrowIfLessThan(options.MaxIterations, 1, nameof(options.MaxIterations));
        if (options.MaxEvaluations is { } maxEval)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(maxEval, 1L, nameof(options.MaxEvaluations));
        }

        if (options.InitialStep is { } t && !(t > 0 && double.IsFinite(t)))
        {
            throw new ArgumentOutOfRangeException(nameof(options.InitialStep), "The initial step must be positive and finite.");
        }

        if (!(options.BacktrackingFactor > 0 && options.BacktrackingFactor < 1))
        {
            throw new ArgumentOutOfRangeException(nameof(options.BacktrackingFactor), "The backtracking factor must lie in (0, 1).");
        }

        gradientTolerance = options.GradientTolerance;
        maxIterations = options.MaxIterations;
        maxEvaluations = options.MaxEvaluations ?? long.MaxValue;
        initialStep = options.InitialStep;
        backtracking = options.BacktrackingFactor;
        restart = options.AdaptiveRestart;
    }

    /// <summary>
    /// Gets the final step size <c>t</c> used in the most recent call to <c>Minimize</c> (<c>1/t</c> estimates the Lipschitz constant).
    /// <para>直近の <c>Minimize</c> 呼び出しで使用した最終ステップ幅 <c>t</c>（<c>1/t</c> がリプシッツ定数の推定値）を取得します。</para>
    /// </summary>
    public double LastStep { get; private set; } = double.NaN;

    /// <summary>
    /// Gets the number of adaptive momentum restarts performed in the most recent call to <c>Minimize</c>.
    /// <para>直近の <c>Minimize</c> 呼び出しで行われた適応的再始動の回数を取得します。</para>
    /// </summary>
    public int Restarts { get; private set; }

    /// <summary>
    /// Minimizes the composite objective <c>f(x) + h(x)</c>.
    /// <para>複合目的関数 <c>f(x) + h(x)</c> を最小化します。</para>
    /// </summary>
    /// <typeparam name="TSmooth">The smooth differentiable objective type. / 滑らかな微分可能目的関数の型。</typeparam>
    /// <typeparam name="TProximal">The proximal operator type for <c>h</c>. / 非平滑項 <c>h</c> の近接写像の型。</typeparam>
    /// <param name="smooth">The smooth convex objective <c>f</c>. / 滑らかな凸目的関数 <c>f</c>。</param>
    /// <param name="proximal">The proximable convex function <c>h</c>. / 近接写像を持つ凸関数 <c>h</c>。</param>
    /// <param name="initialPoint">A finite non-empty initial point (need not lie in the domain of <c>h</c>). / 空でない有限の初期点（<c>h</c> の定義域外でも可）。</param>
    /// <param name="cancellationToken">A cancellation token checked at each iteration. / 反復ごとに確認されるキャンセルトークン。</param>
    /// <returns>A result containing the last accepted point <c>x⁺</c>, composite value <c>f(x⁺) + h(x⁺)</c>, gradient mapping infinity norm, evaluation count of <c>f</c>, and termination reason (<see cref="MinimizationTermination.LineSearchFailed"/> if backtracking reaches underflow, <see cref="MinimizationTermination.NonFiniteValue"/> if <c>f</c> or its gradient is non-finite). / 最後に採用した点 <c>x⁺</c>、複合目的関数値 <c>f(x⁺) + h(x⁺)</c>、勾配写像の最大ノルム、<c>f</c> の評価回数、および終了理由を含む結果（バックトラッキング不能時は <see cref="MinimizationTermination.LineSearchFailed"/>、非有限値発生時は <see cref="MinimizationTermination.NonFiniteValue"/>）。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="smooth"/> or <paramref name="proximal"/> is <see langword="null"/>. / <paramref name="smooth"/> または <paramref name="proximal"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoint"/> is empty or contains a non-finite value. / <paramref name="initialPoint"/> が空、または有限でない値を含む場合。</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was canceled. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public MinimizationResult Minimize<TSmooth, TProximal>(TSmooth smooth, TProximal proximal, ReadOnlySpan<double> initialPoint,
        CancellationToken cancellationToken = default)
        where TSmooth : IDifferentiableObjectiveFunction
        where TProximal : IProximalOperator
    {
        ArgumentNullException.ThrowIfNull(smooth);
        ArgumentNullException.ThrowIfNull(proximal);
        if (initialPoint.IsEmpty)
        {
            throw new ArgumentException("The initial point must not be empty.", nameof(initialPoint));
        }

        foreach (double v in initialPoint)
        {
            if (!double.IsFinite(v))
            {
                throw new ArgumentException("The initial point must be finite.", nameof(initialPoint));
            }
        }

        int n = initialPoint.Length;
        var x = initialPoint.ToArray();       // 直前の点 xₖ
        var y = initialPoint.ToArray();       // 外挿点
        var gy = new double[n];
        var trial = new double[n];            // x⁺
        var work = new double[n];
        var trialGradient = new double[n];
        long evaluations = 0;
        Restarts = 0;

        double fy = EvaluateWithGradient(smooth, y, gy, ref evaluations);
        if (!double.IsFinite(fy) || !AllFinite(gy))
        {
            LastStep = double.NaN;
            return new MinimizationResult(new Vector<double>(x), fy + proximal.Evaluate(x), double.NaN, 0, evaluations, MinimizationTermination.NonFiniteValue);
        }

        // 歩幅の見積もりも評価の上限に数える。残りがなければ見積もらない（1歩も進めずに上限で止まる）
        double step = initialStep ?? (evaluations < maxEvaluations ? EstimateStep(smooth, y, gy, work, trial, ref evaluations) : 1);
        double theta = 1;
        double fx = double.NaN, mappingNorm = double.NaN;
        int iterations = 0;
        var termination = MinimizationTermination.MaxIterations;
        bool hasPoint = false;
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (iterations >= maxIterations)
            {
                break;
            }

            // 後退判定つきの近接勾配の一歩
            double ftrial;
            while (true)
            {
                if (evaluations >= maxEvaluations)
                {
                    termination = MinimizationTermination.MaxEvaluations;
                    goto Done;
                }

                for (int i = 0; i < n; i++)
                {
                    work[i] = y[i] - step * gy[i];
                }

                proximal.Apply(work, step, trial);
                ftrial = smooth.Evaluate(trial);
                evaluations++;
                double linear = 0, squared = 0;
                for (int i = 0; i < n; i++)
                {
                    double d = trial[i] - y[i];
                    linear += gy[i] * d;
                    squared += d * d;
                }

                if (double.IsFinite(ftrial))
                {
                    double excess = ftrial - fy - linear - squared / (2 * step);
                    double magnitude = Math.Abs(fy) + Math.Abs(ftrial);
                    if (excess <= 4 * NumericConstants.MachineEpsilon * magnitude)
                    {
                        break;
                    }

                    // 差が値の丸め誤差に埋もれうる大きさなら、勾配の差による判定 (∇f(x⁺) − ∇f(y))ᵀ(x⁺ − y) ≤ ‖x⁺ − y‖²/t に切り替える
                    // （Becker, Candès &amp; Grant 2011 §5.3。二次関数では値による判定と同じ）
                    if (excess <= NumericConstants.SqrtMachineEpsilon * magnitude && evaluations < maxEvaluations)
                    {
                        double value = EvaluateWithGradient(smooth, trial, trialGradient, ref evaluations);
                        if (double.IsFinite(value) && Curvature(trialGradient, gy, trial, y) <= squared / step)
                        {
                            break;
                        }
                    }
                }

                step *= backtracking;
                if (!(step > 0) || squared == 0)
                {
                    termination = MinimizationTermination.LineSearchFailed;
                    goto Done;
                }
            }

            // 勾配写像の大きさと再始動の判定
            double norm = 0, alignment = 0;
            for (int i = 0; i < n; i++)
            {
                double mapping = (y[i] - trial[i]) / step;
                norm = Math.Max(norm, Math.Abs(mapping));
                alignment += mapping * (trial[i] - x[i]);
            }

            (x, trial) = (trial, x);           // x ← x⁺、trial は直前の点を持つ
            fx = ftrial;
            mappingNorm = norm;
            hasPoint = true;
            iterations++;
            if (norm <= gradientTolerance)
            {
                termination = MinimizationTermination.GradientTolerance;
                break;
            }

            if (restart && alignment > 0)
            {
                theta = 1;
                Restarts++;
                x.CopyTo(y, 0);
            }
            else
            {
                double next = 0.5 * (1 + Math.Sqrt(1 + 4 * theta * theta));
                double momentum = (theta - 1) / next;
                for (int i = 0; i < n; i++)
                {
                    y[i] = x[i] + momentum * (x[i] - trial[i]);
                }

                theta = next;
            }

            if (evaluations >= maxEvaluations)
            {
                termination = MinimizationTermination.MaxEvaluations;
                break;
            }

            fy = EvaluateWithGradient(smooth, y, gy, ref evaluations);
            if (!double.IsFinite(fy) || !AllFinite(gy))
            {
                termination = MinimizationTermination.NonFiniteValue;
                break;
            }
        }

    Done:
        LastStep = step;
        if (!hasPoint)
        {
            // 1歩も進めなかった: 初期点のまま（外挿点は初期点と同じ）
            fx = fy;
        }

        return new MinimizationResult(new Vector<double>(x), fx + proximal.Evaluate(x), mappingNorm, iterations, evaluations, termination);
    }

    /// <summary>
    /// Minimizes the composite objective <c>f(x) + h(x)</c> using a delegate for the smooth term.
    /// <para>滑らかな項をデリゲートで指定して複合目的関数 <c>f(x) + h(x)</c> を最小化します。</para>
    /// </summary>
    /// <typeparam name="TProximal">The proximal operator type for <c>h</c>. / 非平滑項 <c>h</c> の近接写像の型。</typeparam>
    /// <param name="smooth">The delegate returning the smooth objective value and writing its gradient to the second parameter. / 目的関数値を返し、第2引数に勾配を書き込む滑らかな凸関数 <c>f</c> のデリゲート。</param>
    /// <param name="proximal">The proximable convex function <c>h</c>. / 近接写像を持つ凸関数 <c>h</c>。</param>
    /// <param name="initialPoint">A finite non-empty initial point (need not lie in the domain of <c>h</c>). / 空でない有限の初期点（<c>h</c> の定義域外でも可）。</param>
    /// <param name="cancellationToken">A cancellation token checked at each iteration. / 反復ごとに確認されるキャンセルトークン。</param>
    /// <returns>A result containing the last accepted point <c>x⁺</c>, composite value <c>f(x⁺) + h(x⁺)</c>, gradient mapping infinity norm, evaluation count of <c>f</c>, and termination reason. / 最後に採用した点 <c>x⁺</c>、複合目的関数値 <c>f(x⁺) + h(x⁺)</c>、勾配写像の最大ノルム、<c>f</c> の評価回数、および終了理由を含む結果。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="smooth"/> or <paramref name="proximal"/> is <see langword="null"/>. / <paramref name="smooth"/> または <paramref name="proximal"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoint"/> is empty or contains a non-finite value. / <paramref name="initialPoint"/> が空、または有限でない値を含む場合。</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was canceled. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public MinimizationResult Minimize<TProximal>(Func<ReadOnlyVectorSpan<double>, VectorSpan<double>, double> smooth, TProximal proximal,
        ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)
        where TProximal : IProximalOperator
    {
        ArgumentNullException.ThrowIfNull(smooth);
        return Minimize(new DelegateObjective(smooth), proximal, initialPoint, cancellationToken);
    }

    private static double EvaluateWithGradient<TSmooth>(TSmooth smooth, double[] x, double[] gradient, ref long evaluations)
        where TSmooth : IDifferentiableObjectiveFunction
    {
        Array.Clear(gradient);
        evaluations++;
        return smooth.Evaluate(x, gradient);
    }

    // 勾配の向きに少し動かした点との勾配の差から L ≈ ‖Δ∇f‖/‖Δx‖ を見積もり、1/L を返す
    private static double EstimateStep<TSmooth>(TSmooth smooth, double[] x, double[] gradient, double[] probe, double[] probeGradient, ref long evaluations)
        where TSmooth : IDifferentiableObjectiveFunction
    {
        double gnorm = Blas.Norm2(gradient), xnorm = Blas.Norm2(x);
        if (!(gnorm > 0))
        {
            return 1;
        }

        double h = 1e-4 * Math.Max(1, xnorm) / gnorm;
        for (int i = 0; i < x.Length; i++)
        {
            probe[i] = x[i] - h * gradient[i];
        }

        EvaluateWithGradient(smooth, probe, probeGradient, ref evaluations);
        double change = 0;
        for (int i = 0; i < x.Length; i++)
        {
            double d = probeGradient[i] - gradient[i];
            change += d * d;
        }

        double lipschitz = Math.Sqrt(change) / (h * gnorm);
        return lipschitz > 0 && double.IsFinite(lipschitz) ? 1 / lipschitz : 1;
    }

    // (a − b)ᵀ(x − y)
    private static double Curvature(double[] a, double[] b, double[] x, double[] y)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += (a[i] - b[i]) * (x[i] - y[i]);
        }

        return sum;
    }

    private static bool AllFinite(double[] values)
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
}
