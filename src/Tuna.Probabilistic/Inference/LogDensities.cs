using Tuna.LinearAlgebra;
using Tuna.Optimization;

namespace Tuna.Probabilistic.Inference;

/// <summary>
/// Provides helpers for creating log-densities from delegates, converting between log-densities and optimization objectives, and checking analytic gradients.
/// <para>デリゲートからの対数密度の生成、対数密度と最適化目的関数の相互変換、および勾配の検証を行うユーティリティを提供します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Because MAP estimation minimizes <c>−ln π(θ)</c>, a differentiable log-density can be converted with <see cref="ToDifferentiableObjective{TDensity}(TDensity)"/> and passed to optimizers such as <see cref="Tuna.Optimization.Gradient.Lbfgs"/>.
/// Conversely, a potential energy objective <c>U(θ) = −ln π(θ)</c> (including <see cref="Tuna.Optimization.Global.FactorableObjective{TFunction}"/> or <see cref="FiniteDifferenceGradient"/>) can be wrapped with <see cref="FromPotential{TObjective}(TObjective)"/> for sampling.
/// </para>
/// <para>
/// MAP 推定は <c>−ln π(θ)</c> の最小化に相当するため、<see cref="ToDifferentiableObjective{TDensity}(TDensity)"/> で <see cref="Tuna.Optimization.Gradient.Lbfgs"/> などの最適化手法に渡せます。
/// 逆に、最適化の目的関数として定義したポテンシャル <c>U(θ) = −ln π(θ)</c>（<see cref="Tuna.Optimization.Global.FactorableObjective{TFunction}"/> や <see cref="FiniteDifferenceGradient"/> でラップしたものを含む）は、<see cref="FromPotential{TObjective}(TObjective)"/> でサンプラーに渡せます。
/// </para>
/// </remarks>
public static partial class LogDensities
{
    /// <summary>
    /// Wraps a log-density delegate as an <see cref="ILogDensity"/>.
    /// <para>対数密度を計算するデリゲートを <see cref="ILogDensity"/> としてラップします。</para>
    /// </summary>
    /// <param name="logDensity">The delegate that evaluates the log-density. / 対数密度を評価するデリゲート。</param>
    /// <returns>An <see cref="ILogDensity"/> wrapper around <paramref name="logDensity"/>. / <paramref name="logDensity"/> の <see cref="ILogDensity"/> ラッパー。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="logDensity"/> is <see langword="null"/>. / <paramref name="logDensity"/> が <see langword="null"/> の場合。</exception>
    public static ILogDensity FromDelegate(Func<ReadOnlyVectorSpan<double>, double> logDensity)
    {
        ArgumentNullException.ThrowIfNull(logDensity);
        return new DelegateLogDensity(logDensity);
    }

    /// <summary>
    /// Wraps a value-and-gradient delegate as an <see cref="IDifferentiableLogDensity"/>.
    /// <para>対数密度を返し第 2 引数へ勾配を書き込むデリゲートを <see cref="IDifferentiableLogDensity"/> としてラップします。</para>
    /// </summary>
    /// <param name="logDensity">The delegate that evaluates the log-density and writes its gradient to the second argument. / 対数密度を返し、第 2 引数に勾配を書き込むデリゲート。</param>
    /// <returns>An <see cref="IDifferentiableLogDensity"/> wrapper around <paramref name="logDensity"/>. / <paramref name="logDensity"/> の <see cref="IDifferentiableLogDensity"/> ラッパー。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="logDensity"/> is <see langword="null"/>. / <paramref name="logDensity"/> が <see langword="null"/> の場合。</exception>
    public static IDifferentiableLogDensity FromDelegate(Func<ReadOnlyVectorSpan<double>, VectorSpan<double>, double> logDensity)
    {
        ArgumentNullException.ThrowIfNull(logDensity);
        return new DelegateDifferentiableLogDensity(logDensity);
    }

    /// <summary>
    /// Wraps a differentiable potential energy objective <c>U(θ) = −ln π(θ)</c> as a log-density <c>ln π(θ) = −U(θ)</c>.
    /// <para>ポテンシャル <c>U(θ) = −ln π(θ)</c> を表す微分可能な目的関数を、対数密度 <c>ln π(θ) = −U(θ)</c> としてラップします。</para>
    /// </summary>
    /// <typeparam name="TObjective">The potential objective type. / ポテンシャル目的関数の型。</typeparam>
    /// <param name="potential">The potential objective function to negate. / 符号を反転するポテンシャル目的関数。</param>
    /// <returns>A log-density wrapper negating the value and gradient of <paramref name="potential"/>. / <paramref name="potential"/> の値と勾配の符号を反転する対数密度ラッパー。</returns>
    /// <remarks>
    /// <para>Objective value <see cref="double.PositiveInfinity"/> maps to <see cref="double.NegativeInfinity"/> (zero density), <see cref="double.NaN"/> remains <see cref="double.NaN"/>, and the gradient is negated.</para>
    /// <para>目的関数値の <see cref="double.PositiveInfinity"/> は <see cref="double.NegativeInfinity"/>（密度 0）に写像され、<see cref="double.NaN"/> はそのまま保持されます。勾配も符号が反転されます。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="potential"/> is <see langword="null"/>. / <paramref name="potential"/> が <see langword="null"/> の場合。</exception>
    public static PotentialLogDensity<TObjective> FromPotential<TObjective>(TObjective potential)
        where TObjective : IDifferentiableObjectiveFunction
    {
        ArgumentNullException.ThrowIfNull(potential);
        return new PotentialLogDensity<TObjective>(potential);
    }

    /// <summary>
    /// Wraps a log-density as a minimization objective <c>−ln π(θ)</c> for derivative-free optimizers.
    /// <para>対数密度を最小化目的関数 <c>−ln π(θ)</c> としてラップします（勾配を使わない最適化手法向け）。</para>
    /// </summary>
    /// <typeparam name="TDensity">The log-density type. / 対数密度の型。</typeparam>
    /// <param name="density">The log-density to negate. / 符号を反転する対数密度。</param>
    /// <returns>An objective function wrapper returning <c>−ln π(θ)</c>. / <c>−ln π(θ)</c> を返す目的関数ラッパー。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="density"/> is <see langword="null"/>. / <paramref name="density"/> が <see langword="null"/> の場合。</exception>
    public static NegativeLogDensity<TDensity> ToObjective<TDensity>(TDensity density)
        where TDensity : ILogDensity
    {
        ArgumentNullException.ThrowIfNull(density);
        return new NegativeLogDensity<TDensity>(density);
    }

    /// <summary>
    /// Wraps a differentiable log-density as a minimization objective <c>−ln π(θ)</c> for gradient-based optimizers such as MAP estimation.
    /// <para>微分可能な対数密度を、勾配法による MAP 推定などのための最小化目的関数 <c>−ln π(θ)</c> としてラップします。</para>
    /// </summary>
    /// <typeparam name="TDensity">The differentiable log-density type. / 微分可能な対数密度の型。</typeparam>
    /// <param name="density">The differentiable log-density to negate. / 符号を反転する微分可能な対数密度。</param>
    /// <returns>A differentiable objective function wrapper returning <c>−ln π(θ)</c> and its negated gradient. / <c>−ln π(θ)</c> と反転した勾配を返す微分可能な目的関数ラッパー。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="density"/> is <see langword="null"/>. / <paramref name="density"/> が <see langword="null"/> の場合。</exception>
    public static NegativeDifferentiableLogDensity<TDensity> ToDifferentiableObjective<TDensity>(TDensity density)
        where TDensity : IDifferentiableLogDensity
    {
        ArgumentNullException.ThrowIfNull(density);
        return new NegativeDifferentiableLogDensity<TDensity>(density);
    }

    /// <summary>
    /// Compares the analytic gradient of a log-density against central finite differences.
    /// <para>対数密度の解析勾配と中心差分による数値微分を比較します。</para>
    /// </summary>
    /// <typeparam name="TDensity">The differentiable log-density type. / 微分可能な対数密度の型。</typeparam>
    /// <param name="density">The differentiable log-density to check. / 検証する微分可能な対数密度。</param>
    /// <param name="theta">The point at which to compare gradients. / 勾配を比較する点。</param>
    /// <returns>The maximum componentwise scaled error <c>maxᵢ |gᵢ − dᵢ| / max(1, |gᵢ|, |dᵢ|)</c>. / 成分ごとの相対・絶対混合誤差の最大値 <c>maxᵢ |gᵢ − dᵢ| / max(1, |gᵢ|, |dᵢ|)</c>。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="density"/> is <see langword="null"/>. / <paramref name="density"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="theta"/> is empty. / <paramref name="theta"/> が空の場合。</exception>
    public static double GradientError<TDensity>(TDensity density, ReadOnlySpan<double> theta)
        where TDensity : IDifferentiableLogDensity
    {
        ArgumentNullException.ThrowIfNull(density);
        return GradientCheck.MaximumError(new NegativeDifferentiableLogDensity<TDensity>(density), theta);
    }

    // NaN と +∞ を密度 0 (-∞) へ正規化する
    internal static double Sanitize(double value) => value < double.PositiveInfinity ? value : double.NegativeInfinity;

    private sealed class DelegateLogDensity(Func<ReadOnlyVectorSpan<double>, double> function) : ILogDensity
    {
        public double LogDensity(ReadOnlyVectorSpan<double> theta) => function(theta);
    }

    private sealed class DelegateDifferentiableLogDensity(Func<ReadOnlyVectorSpan<double>, VectorSpan<double>, double> function)
        : IDifferentiableLogDensity
    {
        public double LogDensity(ReadOnlyVectorSpan<double> theta, VectorSpan<double> gradient) => function(theta, gradient);
    }
}

/// <summary>
/// Wraps a potential objective <c>U(θ) = −ln π(θ)</c> as an <see cref="IDifferentiableLogDensity"/> returning <c>−U(θ)</c>. Created via <see cref="LogDensities.FromPotential{TObjective}(TObjective)"/>.
/// <para>ポテンシャル目的関数 <c>U(θ) = −ln π(θ)</c> を対数密度 <c>−U(θ)</c> として扱うラッパーです。<see cref="LogDensities.FromPotential{TObjective}(TObjective)"/> で生成します。</para>
/// </summary>
/// <typeparam name="TObjective">The potential objective type. / ポテンシャル目的関数の型。</typeparam>
public readonly struct PotentialLogDensity<TObjective> : IDifferentiableLogDensity
    where TObjective : IDifferentiableObjectiveFunction
{
    private readonly TObjective potential;

    internal PotentialLogDensity(TObjective potential) => this.potential = potential;

    /// <inheritdoc/>
    public double LogDensity(ReadOnlyVectorSpan<double> theta) => -potential.Evaluate(theta);

    /// <inheritdoc/>
    public double LogDensity(ReadOnlyVectorSpan<double> theta, VectorSpan<double> gradient)
    {
        double value = potential.Evaluate(theta, gradient);
        for (int i = 0; i < gradient.Length; i++)
        {
            gradient[i] = -gradient[i];
        }

        return -value;
    }
}

/// <summary>
/// Wraps an <see cref="ILogDensity"/> as a minimization objective <c>−ln π(θ)</c>. Created via <see cref="LogDensities.ToObjective{TDensity}(TDensity)"/>.
/// <para><see cref="ILogDensity"/> を最小化目的関数 <c>−ln π(θ)</c> として扱うラッパーです。<see cref="LogDensities.ToObjective{TDensity}(TDensity)"/> で生成します。</para>
/// </summary>
/// <typeparam name="TDensity">The log-density type. / 対数密度の型。</typeparam>
/// <remarks>
/// <para>Zero-density points (<see cref="double.NegativeInfinity"/>) map to <see cref="double.PositiveInfinity"/>, and <see cref="double.NaN"/> remains <see cref="double.NaN"/>.</para>
/// <para>密度 0 の点（<see cref="double.NegativeInfinity"/>）は <see cref="double.PositiveInfinity"/> に写像され、<see cref="double.NaN"/> はそのまま保持されます。</para>
/// </remarks>
public readonly struct NegativeLogDensity<TDensity> : IObjectiveFunction
    where TDensity : ILogDensity
{
    private readonly TDensity density;

    internal NegativeLogDensity(TDensity density) => this.density = density;

    /// <inheritdoc/>
    public double Evaluate(ReadOnlyVectorSpan<double> x) => -density.LogDensity(x);
}

/// <summary>
/// Wraps an <see cref="IDifferentiableLogDensity"/> as a differentiable minimization objective <c>−ln π(θ)</c>. Created via <see cref="LogDensities.ToDifferentiableObjective{TDensity}(TDensity)"/>.
/// <para><see cref="IDifferentiableLogDensity"/> を微分可能な最小化目的関数 <c>−ln π(θ)</c> として扱うラッパーです。<see cref="LogDensities.ToDifferentiableObjective{TDensity}(TDensity)"/> で生成します。</para>
/// </summary>
/// <typeparam name="TDensity">The differentiable log-density type. / 微分可能な対数密度の型。</typeparam>
/// <remarks>
/// <para>Zero-density points (<see cref="double.NegativeInfinity"/>) map to <see cref="double.PositiveInfinity"/>, <see cref="double.NaN"/> remains <see cref="double.NaN"/>, and the gradient is negated.</para>
/// <para>密度 0 の点（<see cref="double.NegativeInfinity"/>）は <see cref="double.PositiveInfinity"/> に写像され、<see cref="double.NaN"/> はそのまま保持されます。勾配も符号が反転されます。</para>
/// </remarks>
public readonly struct NegativeDifferentiableLogDensity<TDensity> : IDifferentiableObjectiveFunction
    where TDensity : IDifferentiableLogDensity
{
    private readonly TDensity density;

    internal NegativeDifferentiableLogDensity(TDensity density) => this.density = density;

    /// <inheritdoc/>
    public double Evaluate(ReadOnlyVectorSpan<double> x) => -density.LogDensity(x);

    /// <inheritdoc/>
    public double Evaluate(ReadOnlyVectorSpan<double> x, VectorSpan<double> gradient)
    {
        double value = density.LogDensity(x, gradient);
        for (int i = 0; i < gradient.Length; i++)
        {
            gradient[i] = -gradient[i];
        }

        return -value;
    }
}
