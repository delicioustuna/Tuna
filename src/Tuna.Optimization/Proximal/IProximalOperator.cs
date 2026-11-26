using Tuna.LinearAlgebra;

namespace Tuna.Optimization.Proximal;

/// <summary>
/// Represents a convex function <c>h</c> (including indicator functions of convex sets) with a computable proximal operator.
/// <para>近接写像を計算できる凸関数 <c>h</c>（凸集合の指示関数を含む）を表します。</para>
/// </summary>
/// <remarks>
/// <para>The proximal operator is defined as <c>prox_{t·h}(v) = argmin_x [h(x) + ‖x − v‖² / (2t)]</c> for <c>t &gt; 0</c>; for an indicator function it reduces to Euclidean projection onto the set. <see cref="Apply"/> supports aliasing when <c>v</c> and <c>destination</c> reference the exact same memory region; partially overlapping spans produce undefined results. Implementations that reuse internal scratch buffers are not thread-safe.</para>
/// <para>近接写像は正のステップ幅 <c>t &gt; 0</c> に対して <c>prox_{t·h}(v) = argmin_x [h(x) + ‖x − v‖² / (2t)]</c> で定義され、集合の指示関数ではその集合へのユークリッド射影になります。<see cref="Apply"/> は入力 <c>v</c> と出力先 <c>destination</c> が完全に同一領域である場合（インプレース）にも対応しますが、部分的な重複時の動作は未定義です。内部の作業用バッファを共有する実装はスレッドセーフではありません。</para>
/// </remarks>
public interface IProximalOperator
{
    /// <summary>
    /// Computes <c>prox_{step·h}(v)</c> and writes the result to <paramref name="destination"/>.
    /// <para><c>prox_{step·h}(v)</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="v">The input point. / 入力点。</param>
    /// <param name="step">The positive proximal step parameter <c>t</c>. / 正のステップ幅 <c>t</c>。</param>
    /// <param name="destination">The destination span with the same length as <paramref name="v"/>. / 結果の書き込み先。<paramref name="v"/> と同じ長さ。</param>
    void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination);

    /// <summary>
    /// Evaluates the convex function <c>h</c> at the specified point.
    /// <para>指定した点で凸関数 <c>h</c> を評価します。</para>
    /// </summary>
    /// <param name="x">The point to evaluate. / 評価点。</param>
    /// <returns>The value <c>h(x)</c>. For set indicators, returns 0 inside the set (within feasibility tolerance) and <see cref="double.PositiveInfinity"/> outside. / 関数値 <c>h(x)</c>。集合の指示関数の場合は集合内（丸め誤差の許容範囲を含む）で 0、集合外で <see cref="double.PositiveInfinity"/>。</returns>
    double Evaluate(ReadOnlyVectorSpan<double> x);
}
