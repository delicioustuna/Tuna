using Tuna.LinearAlgebra;

namespace Tuna.Optimization;

/// <summary>
/// Represents an objective function to minimize over ℝⁿ.
/// <para>ℝⁿ 上で最小化する目的関数を表します。</para>
/// </summary>
/// <remarks>
/// <para>Smaller objective values are preferred; negate the objective value to maximize. Handling of <see cref="double.NaN"/> and infinities follows the conventions of each optimizer. For reproducible runs, evaluating the same point should return the same value; stochastic objectives can use a deterministic per-evaluation seed such as <see cref="Evolutionary.GeneticAlgorithm.EvaluationSeed"/>.</para>
/// <para>値が小さい点を優先します。最大化する場合は目的関数値の符号を反転してください。<see cref="double.NaN"/> や無限大の扱いは各最適化手法の仕様に従います。結果を再現するには、同じ点の評価で常に同じ値を返してください。確率的な評価では <see cref="Evolutionary.GeneticAlgorithm.EvaluationSeed"/> などの評価用シードを利用できます。</para>
/// </remarks>
public interface IObjectiveFunction
{
    /// <summary>
    /// Evaluates the objective function at the specified point.
    /// <para>指定した点で目的関数を評価します。</para>
    /// </summary>
    /// <param name="x">The point to evaluate. Valid only for the duration of the call; copy if retention is needed. / 評価点。呼び出し中のみ有効なビューのため、保持する場合はコピーが必要です。</param>
    /// <returns>The objective value at <paramref name="x"/>, or <see cref="double.NaN"/> if the point cannot be evaluated. / <paramref name="x"/> での目的関数値。評価できない点の場合は <see cref="double.NaN"/>。</returns>
    double Evaluate(ReadOnlyVectorSpan<double> x);
}
