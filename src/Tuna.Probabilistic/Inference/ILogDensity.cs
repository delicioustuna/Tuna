using Tuna.LinearAlgebra;

namespace Tuna.Probabilistic.Inference;

/// <summary>
/// Represents an unnormalized log-density function on an unconstrained real vector space.
/// <para>制約のない実ベクトル空間上の対数密度関数を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Implementations must be deterministic, returning the same value for the same input.
/// When running multiple chains in parallel, implementations must support concurrent evaluation from multiple threads.
/// For constrained parameters, wrap the density with <see cref="TransformedLogDensity{TDensity, TBijector}"/>.
/// </para>
/// <para>
/// 実装は決定的であり、同じ入力に対して同じ値を返す必要があります。
/// チェーンを並列実行する場合は、複数スレッドからの同時呼び出しに対応している必要があります。
/// 制約のあるパラメータには <see cref="TransformedLogDensity{TDensity, TBijector}"/> を使用してください。
/// </para>
/// </remarks>
public interface ILogDensity
{
    /// <summary>
    /// Evaluates the log-density at the specified point.
    /// <para>指定した点の対数密度を評価します。</para>
    /// </summary>
    /// <param name="theta">The point to evaluate. Valid only for the duration of the call; copy if retention is needed. / 評価する点。呼び出し中のみ有効なビューのため、保持する場合はコピーしてください。</param>
    /// <returns>The log-density up to an additive constant; <see cref="double.NegativeInfinity"/> for zero-density points. Samplers treat <see cref="double.NaN"/> and <see cref="double.PositiveInfinity"/> as zero density. / 正規化定数を除いた対数密度。密度 0 の点は <see cref="double.NegativeInfinity"/>。サンプラーは <see cref="double.NaN"/> と <see cref="double.PositiveInfinity"/> も密度 0 として扱います。</returns>
    double LogDensity(ReadOnlyVectorSpan<double> theta);
}

/// <summary>
/// Represents a differentiable unnormalized log-density function on an unconstrained real vector space.
/// <para>勾配を計算できる、制約のない実ベクトル空間上の対数密度関数を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// In Hamiltonian samplers, points whose gradient contains <see cref="double.NaN"/> or infinity are treated as divergent transitions.
/// Analytic gradients can be checked against numerical differentiation using <see cref="LogDensities.GradientError{TDensity}(TDensity, ReadOnlySpan{double})"/>.
/// The default <see cref="ILogDensity.LogDensity"/> implementation allocates a temporary gradient buffer; implement it explicitly if value-only evaluation is cheaper.
/// </para>
/// <para>
/// ハミルトニアン法では、勾配に <see cref="double.NaN"/> または無限大を含む点は発散として扱われます。
/// 解析勾配は <see cref="LogDensities.GradientError{TDensity}(TDensity, ReadOnlySpan{double})"/> で数値微分と検証できます。
/// 既定の <see cref="ILogDensity.LogDensity"/> 実装は一時的な勾配バッファを確保するため、値のみを安価に評価できる型は明示的に実装してください。
/// </para>
/// </remarks>
public interface IDifferentiableLogDensity : ILogDensity
{
    /// <summary>
    /// Evaluates the log-density and its gradient at the specified point.
    /// <para>指定した点の対数密度と勾配を評価します。</para>
    /// </summary>
    /// <param name="theta">The point to evaluate. Valid only for the duration of the call; copy if retention is needed. / 評価する点。呼び出し中のみ有効なビューのため、保持する場合はコピーしてください。</param>
    /// <param name="gradient">The destination buffer for the gradient, with the same length as <paramref name="theta"/> and not overlapping it. / 勾配の書き込み先（長さは <paramref name="theta"/> と同じで、重複しない領域）。</param>
    /// <returns>The log-density up to an additive constant; <see cref="double.NegativeInfinity"/> for zero-density points. Samplers treat <see cref="double.NaN"/> and <see cref="double.PositiveInfinity"/> as zero density. / 正規化定数を除いた対数密度。密度 0 の点は <see cref="double.NegativeInfinity"/>。サンプラーは <see cref="double.NaN"/> と <see cref="double.PositiveInfinity"/> も密度 0 として扱います。</returns>
    double LogDensity(ReadOnlyVectorSpan<double> theta, VectorSpan<double> gradient);

    double ILogDensity.LogDensity(ReadOnlyVectorSpan<double> theta)
    {
        var gradient = new double[theta.Length];
        return LogDensity(theta, gradient);
    }
}
