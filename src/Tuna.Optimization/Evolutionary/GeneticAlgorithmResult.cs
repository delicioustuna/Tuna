using Tuna.LinearAlgebra;

namespace Tuna.Optimization.Evolutionary;

/// <summary>
/// Specifies the reason a genetic algorithm terminated.
/// <para>遺伝的アルゴリズムの終了理由を表します。</para>
/// </summary>
public enum GeneticAlgorithmTermination
{
    /// <summary>
    /// The algorithm has not terminated yet.
    /// <para>まだ終了していません。</para>
    /// </summary>
    None,

    /// <summary>
    /// The evaluated generation count reached <see cref="GeneticAlgorithmOptions.MaxGenerations"/>.
    /// <para>評価した世代数が上限（<see cref="GeneticAlgorithmOptions.MaxGenerations"/>）に達しました。</para>
    /// </summary>
    MaxGenerations,

    /// <summary>
    /// An objective value at or below <see cref="GeneticAlgorithmOptions.TargetFitness"/> was reached.
    /// <para>目標値（<see cref="GeneticAlgorithmOptions.TargetFitness"/>）以下の目的関数値が得られました。</para>
    /// </summary>
    TargetReached,

    /// <summary>
    /// Best fitness did not strictly improve for <see cref="GeneticAlgorithmOptions.StagnationGenerations"/> consecutive generations.
    /// <para>指定された世代数（<see cref="GeneticAlgorithmOptions.StagnationGenerations"/>）続けて最良値が改善しませんでした。</para>
    /// </summary>
    Stagnation,
}

/// <summary>
/// Represents the result of a <see cref="GeneticAlgorithm"/> run.
/// <para>遺伝的アルゴリズムの実行結果を表します。</para>
/// </summary>
/// <typeparam name="TPoint">The representation type of the best solution (for example, <see cref="Genome"/> for <see cref="GeneticAlgorithm"/>). / 最良解の表現型（<see cref="GeneticAlgorithm"/> では <see cref="Genome"/>）。</typeparam>
/// <param name="BestPoint">A copy of the best candidate evaluated so far. / これまでに評価した候補のうち最良の解（コピー）。</param>
/// <param name="BestFitness">The objective value at <paramref name="BestPoint"/>, or <see cref="double.NaN"/> if all evaluations were <see cref="double.NaN"/>. / <paramref name="BestPoint"/> の目的関数値。全評価が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</param>
/// <param name="Generations">The number of evaluated generations (including the initial population). / 評価した世代数（初期個体群を含む）。</param>
/// <param name="Evaluations">The total number of candidate objective evaluations. / 目的関数の総評価回数。</param>
/// <param name="Termination">The reason the algorithm terminated. / 終了理由。</param>
public sealed record GeneticAlgorithmResult<TPoint>(
    TPoint BestPoint, double BestFitness, int Generations, long Evaluations, GeneticAlgorithmTermination Termination);

/// <summary>
/// Represents the result of a <see cref="RealCodedGeneticAlgorithm"/> run.
/// <para>実数値遺伝的アルゴリズムの実行結果を表します。</para>
/// </summary>
/// <param name="BestPoint">A copy of the best point evaluated so far. / これまでに評価した点のうち最良の点（コピー）。</param>
/// <param name="BestFitness">The objective value at <paramref name="BestPoint"/>, or <see cref="double.NaN"/> if all evaluations were <see cref="double.NaN"/>. / <paramref name="BestPoint"/> の目的関数値。全評価が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</param>
/// <param name="Generations">The number of evaluated generations (including the initial population). / 評価した世代数（初期個体群を含む）。</param>
/// <param name="Evaluations">The total number of candidate objective evaluations. / 目的関数の総評価回数。</param>
/// <param name="Termination">The reason the algorithm terminated. / 終了理由。</param>
public sealed record GeneticAlgorithmResult(
    Vector<double> BestPoint, double BestFitness, int Generations, long Evaluations, GeneticAlgorithmTermination Termination);
