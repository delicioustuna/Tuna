namespace Tuna.Optimization.Gradient;

/// <summary>
/// Configures options for <see cref="Lbfgs"/> and <see cref="LbfgsB"/>.
/// <para><see cref="Lbfgs"/> および <see cref="LbfgsB"/> の設定オプションを表します。</para>
/// </summary>
/// <remarks>
/// <para>Option values are validated when passed to the optimizer constructor.</para>
/// <para>各設定値の妥当性はオプティマイザーのコンストラクターで検証されます。</para>
/// </remarks>
public class LbfgsOptions
{
    /// <summary>
    /// Gets the maximum number of curvature pairs <c>(s, y)</c> retained in history.
    /// <para>履歴として保持する曲率ペア <c>(s, y)</c> の最大数 <c>m</c> を取得します。</para>
    /// </summary>
    /// <value>An integer greater than or equal to 1. The default is 10; typical values are between 3 and 20. / 1 以上の整数。既定値は 10（通常は 3〜20 が目安）。</value>
    public int HistorySize { get; init; } = 10;

    /// <summary>
    /// Gets the convergence tolerance on the infinity norm of the gradient (or projected gradient for <see cref="LbfgsB"/>).
    /// <para>勾配（<see cref="LbfgsB"/> では射影勾配）の最大ノルムに対する収束許容誤差を取得します。</para>
    /// </summary>
    /// <value>A non-negative finite number. The default is <c>1e-5</c>. / 0 以上の有限値。既定値は <c>1e-5</c>。</value>
    public double GradientTolerance { get; init; } = 1e-5;

    /// <summary>
    /// Gets the convergence tolerance on the relative decrease in the objective value per iteration.
    /// <para>1反復あたりの目的関数値の相対減少に対する収束許容誤差を取得します。</para>
    /// </summary>
    /// <value>A non-negative finite number. The default is <c>1e7</c> times machine epsilon (approximately <c>2.2e-9</c>). Set to 0 to disable this criterion. / 0 以上の有限値。既定値はマシンイプシロンの 10⁷ 倍（約 <c>2.2e-9</c>）。0 を指定するとこの判定を無効にします。</value>
    /// <remarks>
    /// <para>Convergence is signaled when <c>(fₖ − fₖ₊₁) / max(|fₖ|, |fₖ₊₁|, 1)</c> is less than or equal to this value.</para>
    /// <para><c>(fₖ − fₖ₊₁) / max(|fₖ|, |fₖ₊₁|, 1)</c> がこの値以下になったときに収束と判定します。</para>
    /// </remarks>
    public double FunctionTolerance { get; init; } = 1e7 * (double.BitIncrement(1.0) - 1.0);

    /// <summary>
    /// Gets the optional convergence tolerance on the relative step size per iteration.
    /// <para>1反復あたりの相対ステップ幅に対する収束許容誤差を取得します。</para>
    /// </summary>
    /// <value>A non-negative finite number, or <see langword="null"/> (the default) to disable this criterion. Convergence is signaled when <c>‖xₖ₊₁ − xₖ‖∞ ≤ StepTolerance · max(1, ‖xₖ₊₁‖∞)</c>. / 0 以上の有限値、またはこの判定を無効にする <see langword="null"/>（既定値）。<c>‖xₖ₊₁ − xₖ‖∞ ≤ StepTolerance · max(1, ‖xₖ₊₁‖∞)</c> で収束と判定します。</value>
    public double? StepTolerance { get; init; }

    /// <summary>
    /// Gets the maximum number of iterations.
    /// <para>反復回数の上限を取得します。</para>
    /// </summary>
    /// <value>An integer greater than or equal to 1. The default is 1000. / 1 以上の整数。既定値は 1000。</value>
    public int MaxIterations { get; init; } = 1000;

    /// <summary>
    /// Gets the optional maximum number of objective function evaluations.
    /// <para>目的関数の評価回数の上限を取得します。</para>
    /// </summary>
    /// <value>An integer greater than or equal to 1, or <see langword="null"/> (the default) for no limit. / 1 以上の整数、または無制限を表す <see langword="null"/>（既定値）。</value>
    public long? MaxEvaluations { get; init; }

    /// <summary>
    /// Gets the sufficient decrease (Armijo) coefficient μ used by the line search.
    /// <para>直線探索の十分減少（Armijo）条件の係数 μ を取得します。</para>
    /// </summary>
    /// <value>A value strictly between 0 and 0.5. The default is <c>1e-4</c>. / 0 より大きく 0.5 未満の値。既定値は <c>1e-4</c>。</value>
    public double SufficientDecrease { get; init; } = 1e-4;

    /// <summary>
    /// Gets the curvature coefficient η for the strong Wolfe condition <c>|φ′(α)| ≤ η|φ′(0)|</c>.
    /// <para>直線探索の強 Wolfe 曲率条件 <c>|φ′(α)| ≤ η|φ′(0)|</c> の係数 η を取得します。</para>
    /// </summary>
    /// <value>A value strictly between <see cref="SufficientDecrease"/> and 1. The default is <c>0.9</c>. / <see cref="SufficientDecrease"/> より大きく 1 未満の値。既定値は <c>0.9</c>。</value>
    public double Curvature { get; init; } = 0.9;

    /// <summary>
    /// Gets the maximum number of objective evaluations allowed in a single line search.
    /// <para>1回の直線探索で許容する目的関数評価回数の上限を取得します。</para>
    /// </summary>
    /// <value>An integer greater than or equal to 1. The default is 20. / 1 以上の整数。既定値は 20。</value>
    public int MaxLineSearchEvaluations { get; init; } = 20;
}

/// <summary>
/// Configures options for <see cref="LbfgsB"/>.
/// <para><see cref="LbfgsB"/> の設定オプションを表します（設定項目は <see cref="LbfgsOptions"/> と同一です）。</para>
/// </summary>
public sealed class LbfgsBOptions : LbfgsOptions
{
}
