namespace Tuna.Integration;

/// <summary>
/// Specifies options for deterministic numerical integration (<see cref="Quadrature"/> and <see cref="Cubature"/>).
/// <para>決定的な数値積分（<see cref="Quadrature"/> および <see cref="Cubature"/>）の設定を表します。</para>
/// </summary>
/// <remarks>
/// <para>Integration converges when the error estimate is less than or equal to <c>max(AbsoluteTolerance, RelativeTolerance * |value|)</c>. Specify <see cref="AbsoluteTolerance"/> when the integral is close to zero.</para>
/// <para>誤差見積もりが <c>max(AbsoluteTolerance, RelativeTolerance * |value|)</c> 以下になると収束とみなします。積分値が 0 に近い場合は <see cref="AbsoluteTolerance"/> も設定してください。</para>
/// </remarks>
public sealed class IntegrationOptions
{
    /// <summary>
    /// Gets the absolute error tolerance.
    /// <para>絶対許容誤差を取得します。</para>
    /// </summary>
    /// <value>A finite, non-negative value. The default is 0. / 0 以上の有限値。既定値は 0 です。</value>
    public double AbsoluteTolerance { get; init; }

    /// <summary>
    /// Gets the relative error tolerance.
    /// <para>相対許容誤差を取得します。</para>
    /// </summary>
    /// <value>A finite, non-negative value. The default is 1e-8. / 0 以上の有限値。既定値は 1e-8 です。</value>
    public double RelativeTolerance { get; init; } = 1e-8;

    /// <summary>
    /// Gets the maximum number of integrand evaluations.
    /// <para>被積分関数の評価回数の上限を取得します。</para>
    /// </summary>
    /// <value>A count large enough to evaluate all initial intervals or regions at least once. The default is 1,000,000. / すべての初期区間または初期領域を 1 回以上評価できる回数。既定値は 1,000,000 です。</value>
    public long MaxEvaluations { get; init; } = 1_000_000;

    internal static readonly IntegrationOptions Default = new();

    internal void Validate()
    {
        if (!(AbsoluteTolerance >= 0 && double.IsFinite(AbsoluteTolerance)))
        {
            throw new ArgumentOutOfRangeException(nameof(AbsoluteTolerance), "The absolute tolerance must be finite and non-negative.");
        }

        if (!(RelativeTolerance >= 0 && double.IsFinite(RelativeTolerance)))
        {
            throw new ArgumentOutOfRangeException(nameof(RelativeTolerance), "The relative tolerance must be finite and non-negative.");
        }
    }

    internal void ValidateBudget(long initialEvaluations)
    {
        if (MaxEvaluations < initialEvaluations)
        {
            throw new ArgumentOutOfRangeException(nameof(MaxEvaluations),
                $"The evaluation limit must allow the initial {initialEvaluations} evaluations.");
        }
    }
}
