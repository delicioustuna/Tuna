using Tuna.LinearAlgebra;

namespace Tuna.Optimization;

/// <summary>
/// Represents a differentiable objective function to minimize over ℝⁿ.
/// <para>ℝⁿ 上で最小化する、勾配を計算できる目的関数を表します。</para>
/// </summary>
/// <remarks>
/// <para>Can be passed to gradient-based optimizers such as <see cref="Gradient.Lbfgs"/> and <see cref="Proximal.Fista"/>, as well as value-only optimizers such as genetic algorithms. Implement <see cref="IObjectiveFunction.Evaluate(ReadOnlyVectorSpan{double})"/> explicitly to avoid allocating a temporary gradient buffer when evaluating only the value. Use <see cref="FiniteDifferenceGradient"/> for finite-difference approximation, <see cref="Global.FactorableObjective{TFunction}"/> for automatic differentiation of factorable expressions, and <see cref="GradientCheck"/> to verify analytic gradients.</para>
/// <para>値と勾配を使用する <see cref="Gradient.Lbfgs"/> や <see cref="Proximal.Fista"/> に加え、値のみを使用する遺伝的アルゴリズムにも渡せます。値だけを独立に計算できる場合は <see cref="IObjectiveFunction.Evaluate(ReadOnlyVectorSpan{double})"/> を明示的に実装することで一時的な勾配バッファのアロケーションを避けられます。勾配の近似には <see cref="FiniteDifferenceGradient"/>、分解可能式の自動微分には <see cref="Global.FactorableObjective{TFunction}"/>、解析勾配の検証には <see cref="GradientCheck"/> を利用できます。</para>
/// </remarks>
public interface IDifferentiableObjectiveFunction : IObjectiveFunction
{
    /// <summary>
    /// Evaluates the objective function at the specified point and writes its gradient.
    /// <para>指定した点で目的関数を評価し、勾配を書き込みます。</para>
    /// </summary>
    /// <param name="x">The point to evaluate. Valid only for the duration of the call; copy if retention is needed. / 評価点。呼び出し中のみ有効なビューのため、保持する場合はコピーが必要です。</param>
    /// <param name="gradient">The destination span for the gradient, with the same length as <paramref name="x"/> and non-overlapping with <paramref name="x"/>. / 勾配の書き込み先。<paramref name="x"/> と同じ長さで、領域は重複しません。</param>
    /// <returns>The objective value at <paramref name="x"/>. / <paramref name="x"/> での目的関数値。</returns>
    double Evaluate(ReadOnlyVectorSpan<double> x, VectorSpan<double> gradient);

    double IObjectiveFunction.Evaluate(ReadOnlyVectorSpan<double> x)
    {
        var gradient = new double[x.Length];
        return Evaluate(x, gradient);
    }
}
