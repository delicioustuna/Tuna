namespace Tuna.Integration;

/// <summary>
/// Specifies the reason why numerical integration terminated.
/// <para>数値積分の終了理由を表します。</para>
/// </summary>
public enum IntegrationTermination
{
    /// <summary>
    /// The error estimate (or standard error for quasi-Monte Carlo integration) met the requested tolerance.
    /// <para>誤差見積もり（準モンテカルロ積分では標準誤差）が許容誤差以下になりました。</para>
    /// </summary>
    Tolerance,

    /// <summary>
    /// The maximum number of integrand evaluations was reached.
    /// <para>被積分関数の評価回数が上限に達しました。</para>
    /// </summary>
    MaxEvaluations,

    /// <summary>
    /// Rounding error prevented the requested tolerance from being reached.
    /// <para>丸め誤差により許容誤差に到達できませんでした。</para>
    /// </summary>
    /// <remarks>
    /// <para>This can occur when positive and negative contributions largely cancel or when the relative tolerance is close to machine precision.</para>
    /// <para>被積分関数の正負の寄与が大きく相殺される場合や、相対許容誤差が計算機イプシロンに近い場合に発生します。</para>
    /// </remarks>
    RoundoffLimited,

    /// <summary>
    /// The interval or region with the largest error could no longer be subdivided.
    /// <para>誤差の大きい区間または領域をこれ以上分割できませんでした。</para>
    /// </summary>
    /// <remarks>
    /// <para>This can occur near singularities or discontinuities that were not provided as breakpoints.</para>
    /// <para>区切り点として指定していない特異点や不連続点の近傍で発生することがあります。</para>
    /// </remarks>
    IntervalTooSmall,

    /// <summary>
    /// The integrand returned a non-finite value or an intermediate sum became non-finite.
    /// <para>被積分関数の戻り値または積分の途中結果が有限値でなくなりました。</para>
    /// </summary>
    /// <remarks>
    /// <para>Both the integral value and the error estimate (or standard error) are <see cref="double.NaN"/>.</para>
    /// <para>積分値および誤差見積もり（準モンテカルロ積分では標準誤差）はいずれも <see cref="double.NaN"/> になります。</para>
    /// </remarks>
    NonFiniteValue,
}

/// <summary>
/// Represents the result of deterministic numerical integration (<see cref="Quadrature"/> or <see cref="Cubature"/>).
/// <para>決定的な数値積分（<see cref="Quadrature"/> または <see cref="Cubature"/>）の結果を表します。</para>
/// </summary>
/// <param name="Value">The computed integral value. / 計算された積分値。</param>
/// <param name="ErrorEstimate">The estimated absolute error summed over all subregions; not a strict upper bound. / 絶対誤差の見積もり（全小区間の見積もりの和。厳密な上界ではありません）。</param>
/// <param name="Evaluations">The total number of integrand evaluations performed. / 被積分関数を評価した総回数。</param>
/// <param name="Termination">The reason integration terminated. / 積分の終了理由。</param>
public readonly record struct IntegrationResult(double Value, double ErrorEstimate, long Evaluations, IntegrationTermination Termination)
{
    /// <summary>
    /// Gets a value indicating whether integration converged within the requested tolerance.
    /// <para>積分が許容誤差内に収束したかどうかを取得します。</para>
    /// </summary>
    /// <value><see langword="true"/> if <see cref="Termination"/> is <see cref="IntegrationTermination.Tolerance"/>; otherwise, <see langword="false"/>. / <see cref="Termination"/> が <see cref="IntegrationTermination.Tolerance"/> の場合は <see langword="true"/>、それ以外は <see langword="false"/>。</value>
    public bool Converged => Termination == IntegrationTermination.Tolerance;
}
