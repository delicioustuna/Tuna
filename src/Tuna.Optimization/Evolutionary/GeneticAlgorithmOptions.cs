namespace Tuna.Optimization.Evolutionary;

/// <summary>
/// Configures options for <see cref="GeneticAlgorithm"/> and <see cref="RealCodedGeneticAlgorithm"/>.
/// <para><see cref="GeneticAlgorithm"/> および <see cref="RealCodedGeneticAlgorithm"/> の設定オプションを表します。</para>
/// </summary>
/// <remarks>
/// <para>Option values are validated by the algorithm constructor.</para>
/// <para>各設定値の妥当性はアルゴリズムのコンストラクターで検証されます。</para>
/// </remarks>
public class GeneticAlgorithmOptions
{
    /// <summary>
    /// Gets the population size.
    /// <para>個体数を取得します。</para>
    /// </summary>
    /// <value>An integer greater than or equal to 2. The default is 100. / 2 以上の整数。既定値は 100。</value>
    public int PopulationSize { get; init; } = 100;

    /// <summary>
    /// Gets the number of top individuals carried unchanged into the next generation.
    /// <para>次世代へそのまま引き継ぐエリート個体数を取得します。</para>
    /// </summary>
    /// <value>An integer in <c>[0, PopulationSize − 1]</c>. The default is 2. Elites are not re-evaluated unless <see cref="ReevaluateElites"/> is <see langword="true"/>. / <c>[0, PopulationSize − 1]</c> の整数。既定値は 2。<see cref="ReevaluateElites"/> が <see langword="true"/> の場合を除き再評価されません。</value>
    public int EliteCount { get; init; } = 2;

    /// <summary>
    /// Gets the tournament size for parent selection.
    /// <para>親選択に用いるトーナメントサイズを取得します。</para>
    /// </summary>
    /// <value>An integer in <c>[1, PopulationSize]</c>. The default is 2; 1 selects uniformly at random. / <c>[1, PopulationSize]</c> の整数。既定値は 2（1 は一様ランダム選択）。</value>
    public int TournamentSize { get; init; } = 2;

    /// <summary>
    /// Gets the probability of applying crossover to a selected pair of parents.
    /// <para>選択した親ペアに交叉を適用する確率を取得します。</para>
    /// </summary>
    /// <value>A value in <c>[0, 1]</c>. The default is 0.9. When crossover occurs, each section applies its operator (SBX for reals/integers, uniform for bits/categories, order crossover for permutations); otherwise children copy their parents. / <c>[0, 1]</c> の値。既定値は 0.9。交叉時は各区画の演算子（実数・整数は SBX、二値・カテゴリは一様交叉、順列は順序交叉）を適用し、交叉しない場合は親を複製します。</value>
    public double CrossoverProbability { get; init; } = 0.9;

    /// <summary>
    /// Gets the SBX distribution index <c>η_c</c> for real and integer genes.
    /// <para>実数・整数遺伝子の SBX 分布指数 <c>η_c</c> を取得します。</para>
    /// </summary>
    /// <value>A non-negative finite number. The default is 15; larger values keep offspring closer to parents. / 0 以上の有限値。既定値は 15（大きいほど親の近傍に子が生成されます）。</value>
    public double CrossoverDistributionIndex { get; init; } = 15;

    /// <summary>
    /// Gets the optional per-unit mutation probability (per real/integer/binary/categorical gene or per permutation).
    /// <para>突然変異単位（実数・整数・二値・カテゴリの各遺伝子、または各順列）ごとの突然変異確率を取得します。</para>
    /// </summary>
    /// <value>A value in <c>[0, 1]</c>, or <see langword="null"/> (the default) to use <c>1 / MutationUnitCount</c> (<c>1 / Dimension</c> for real-coded GA). / <c>[0, 1]</c> の値、または <c>1 / MutationUnitCount</c>（実数値 GA では <c>1 / Dimension</c>）を使用する <see langword="null"/>（既定値）。</value>
    public double? MutationProbability { get; init; }

    /// <summary>
    /// Gets the polynomial mutation distribution index <c>η_m</c> for real and integer genes.
    /// <para>実数・整数遺伝子の多項式突然変異の分布指数 <c>η_m</c> を取得します。</para>
    /// </summary>
    /// <value>A non-negative finite number. The default is 20; larger values produce smaller perturbations. / 0 以上の有限値。既定値は 20（大きいほど摂動が小さくなります）。</value>
    public double MutationDistributionIndex { get; init; } = 20;

    /// <summary>
    /// Gets the maximum number of generations to evaluate (including the initial population).
    /// <para>評価する世代数の上限（初期個体群を含む）を取得します。</para>
    /// </summary>
    /// <value>An integer greater than or equal to 1. The default is 100. / 1 以上の整数。既定値は 100。</value>
    public int MaxGenerations { get; init; } = 100;

    /// <summary>
    /// Gets the optional target fitness threshold at or below which the algorithm terminates.
    /// <para>この値以下の目的関数値が得られたときに終了する目標値を取得します。</para>
    /// </summary>
    /// <value>A non-<see cref="double.NaN"/> number, or <see langword="null"/> (the default) to disable this stopping criterion. / <see cref="double.NaN"/> 以外の数値、またはこの停止条件を無効にする <see langword="null"/>（既定値）。</value>
    public double? TargetFitness { get; init; }

    /// <summary>
    /// Gets the optional number of consecutive generations without strict improvement in best fitness before terminating.
    /// <para>最良値の厳密な改善がない連続世代数による停滞終了の上限を取得します。</para>
    /// </summary>
    /// <value>An integer greater than or equal to 1, or <see langword="null"/> (the default) to disable stagnation stopping. / 1 以上の整数、または停滞判定を無効にする <see langword="null"/>（既定値）。</value>
    public int? StagnationGenerations { get; init; }

    /// <summary>
    /// Gets the optional random seed.
    /// <para>乱数シードを取得します。</para>
    /// </summary>
    /// <value>An explicit seed, or <see langword="null"/> (the default) to draw a seed from the OS (inspectable via <see cref="GeneticAlgorithm.Seed"/>). / 明示的なシード値、または OS から自動取得する <see langword="null"/>（既定値。取得した値は <see cref="GeneticAlgorithm.Seed"/> で確認できます）。</value>
    public ulong? Seed { get; init; }

    /// <summary>
    /// Gets a value indicating whether elites carried to the next generation are re-evaluated.
    /// <para>次世代へ引き継ぐエリート個体も再評価するかどうかを取得します。</para>
    /// </summary>
    /// <value><see langword="true"/> to include elite copies at the start of every generation's candidate batch (useful for noisy objectives); otherwise, <see langword="false"/> (the default). / 各世代の候補バッチ先頭にエリート複製を含めて再評価する場合は <see langword="true"/>（ノイズのある目的関数向け）、それ以外は <see langword="false"/>（既定値）。</value>
    public bool ReevaluateElites { get; init; }
}

/// <summary>
/// Configures options for <see cref="RealCodedGeneticAlgorithm"/>.
/// <para><see cref="RealCodedGeneticAlgorithm"/> の設定オプションを表します（設定項目は <see cref="GeneticAlgorithmOptions"/> と同一です）。</para>
/// </summary>
public sealed class RealCodedGeneticAlgorithmOptions : GeneticAlgorithmOptions
{
}
