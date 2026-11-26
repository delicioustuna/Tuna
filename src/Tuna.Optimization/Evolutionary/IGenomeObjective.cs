namespace Tuna.Optimization.Evolutionary;

/// <summary>
/// Represents an objective function over genomes to minimize with <see cref="GeneticAlgorithm"/>.
/// <para><see cref="GeneticAlgorithm"/> で最小化するゲノム上の目的関数を表します。</para>
/// </summary>
/// <remarks>
/// <para>Smaller objective values are preferred. <see cref="double.NaN"/> is treated as the worst possible value and <see cref="double.NegativeInfinity"/> as the best.</para>
/// <para>目的関数値は小さいほど良いとみなされます。<see cref="double.NaN"/> は最悪の値、<see cref="double.NegativeInfinity"/> は最良の値として扱われます。</para>
/// </remarks>
public interface IGenomeObjective
{
    /// <summary>
    /// Evaluates the objective function for the specified genome.
    /// <para>指定したゲノムの目的関数値を評価します。</para>
    /// </summary>
    /// <param name="genome">The candidate genome view, valid only for the duration of the call; call <see cref="GenomeView.ToGenome"/> to retain a copy. / 評価対象のゲノムビュー。呼び出し中のみ有効なため、保持する場合は <see cref="GenomeView.ToGenome"/> でコピーしてください。</param>
    /// <returns>The objective value (smaller is better), or <see cref="double.NaN"/> if the candidate could not be evaluated. / 目的関数値（小さいほど良い）。評価できない候補の場合は <see cref="double.NaN"/>。</returns>
    double Evaluate(GenomeView genome);
}
