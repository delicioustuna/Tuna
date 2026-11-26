using Tuna.LinearAlgebra;

namespace Tuna.Optimization;

/// <summary>
/// Specifies the reason a gradient-based or proximal minimization terminated.
/// <para>勾配法または近接勾配法の終了理由を表します。</para>
/// </summary>
public enum MinimizationTermination
{
    /// <summary>
    /// The optimizer has not terminated yet.
    /// <para>まだ終了していません。</para>
    /// </summary>
    None,

    /// <summary>
    /// The infinity norm of the gradient (or projected gradient for bound-constrained problems) fell within tolerance.
    /// <para>勾配（範囲制約付き問題では射影勾配）の最大ノルムが許容誤差以下になりました。</para>
    /// </summary>
    GradientTolerance,

    /// <summary>
    /// The relative decrease in the objective value over one iteration fell within tolerance.
    /// <para>1反復あたりの目的関数値の相対的な減少量が許容誤差以下になりました。</para>
    /// </summary>
    FunctionTolerance,

    /// <summary>
    /// The step size between consecutive iterates fell within tolerance.
    /// <para>1反復あたりの点の移動量が許容誤差以下になりました。</para>
    /// </summary>
    StepTolerance,

    /// <summary>
    /// Both primal and dual residuals of ADMM fell within tolerance.
    /// <para>ADMM の主残差と双対残差がともに許容誤差以下になりました。</para>
    /// </summary>
    ResidualTolerance,

    /// <summary>
    /// The maximum number of iterations was reached.
    /// <para>反復回数が上限に達しました。</para>
    /// </summary>
    MaxIterations,

    /// <summary>
    /// The maximum number of objective evaluations was reached.
    /// <para>目的関数の評価回数が上限に達しました。</para>
    /// </summary>
    MaxEvaluations,

    /// <summary>
    /// Line search failed to find a step length satisfying the acceptance conditions (including finite-precision stagnation or gradient inconsistency).
    /// <para>直線探索が条件を満たすステップ幅を見つけられませんでした（有限精度での停滞や、勾配と関数値変化の不整合を含みます）。</para>
    /// </summary>
    LineSearchFailed,

    /// <summary>
    /// An objective value, gradient, or intermediate evaluation was non-finite.
    /// <para>目的関数値や勾配などの評価値が有限値ではありませんでした。</para>
    /// </summary>
    NonFiniteValue,
}

/// <summary>
/// Represents the result of a minimization run.
/// <para>最小化の実行結果を表します。</para>
/// </summary>
/// <param name="Point">The returned solution point (copied). For line-search methods this is the last accepted iterate (best evaluated); for deterministic stochastic-gradient runs this is the best evaluated point. / 返される解の点（コピー）。直線探索手法では最後に採用した反復点（評価した中で最良）、確率的勾配法の決定的な最小化では評価した中で最小値の点。</param>
/// <param name="Value">The objective value at <paramref name="Point"/>. / <paramref name="Point"/> における目的関数値。</param>
/// <param name="GradientNorm">The infinity norm of the gradient (or projected gradient for bound-constrained problems) at <paramref name="Point"/>. / <paramref name="Point"/> における勾配（範囲制約付き問題では射影勾配）の最大ノルム。</param>
/// <param name="Iterations">The number of completed iterations (accepted steps). / 反復回数（採用したステップ数）。</param>
/// <param name="Evaluations">The total number of objective evaluations. / 目的関数の評価回数。</param>
/// <param name="Termination">The reason minimization terminated. / 終了理由。</param>
public sealed record MinimizationResult(
    Vector<double> Point, double Value, double GradientNorm, int Iterations, long Evaluations, MinimizationTermination Termination)
{
    /// <summary>
    /// Gets a value indicating whether minimization terminated by satisfying a convergence tolerance.
    /// <para>収束許容誤差の条件を満たして終了したかどうかを取得します。</para>
    /// </summary>
    /// <value><see langword="true"/> if minimization satisfied a convergence tolerance; otherwise, <see langword="false"/>. / 収束許容誤差の条件を満たして終了した場合は <see langword="true"/>、それ以外は <see langword="false"/>。</value>
    public bool Converged => Termination is MinimizationTermination.GradientTolerance
        or MinimizationTermination.FunctionTolerance or MinimizationTermination.StepTolerance or MinimizationTermination.ResidualTolerance;
}
