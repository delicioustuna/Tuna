using Tuna.LinearAlgebra;
using Tuna.RandomSource;

namespace Tuna.Optimization.Evolutionary;

/// <summary>
/// Minimizes an objective function over a bounded real-valued box using a real-coded genetic algorithm.
/// <para>有界な実数ボックス領域上で目的関数を最小化する実数値遺伝的アルゴリズムです。</para>
/// </summary>
/// <remarks>
/// <para>Use the ask-tell loop (<see cref="Ask"/> and <see cref="Tell"/>) for external or parallel evaluation, or <see cref="Run{TObjective}(TObjective, CancellationToken)"/> for synchronous evaluation. Smaller fitness values are preferred, <see cref="double.NaN"/> is ranked worst, and ties break by smaller column index. Identical seeds, options, bounds, and evaluation sequences reproduce identical candidate sequences (matching a real-only <see cref="GeneticAlgorithm"/>). Instances are not thread-safe except that <see cref="EvaluationSeed"/> may be read concurrently while evaluating candidates.</para>
/// <para>外部評価や並列評価を行う場合は <see cref="Ask"/> と <see cref="Tell"/> を、同期評価には <see cref="Run{TObjective}(TObjective, CancellationToken)"/> を使用します。目的関数値は小さいほど優先され、<see cref="double.NaN"/> は最悪、同値時は列インデックスの小さい候補を優先します。同一のシード・設定・範囲・評価値列からは常に同一の候補列が再現されます（実数のみの <see cref="GeneticAlgorithm"/> と一致）。インスタンス操作は単一スレッドで行ってください（ただし候補の評価中に <see cref="EvaluationSeed"/> を並列読み取りすることは安全です）。</para>
/// </remarks>
public sealed class RealCodedGeneticAlgorithm
{
    private readonly GeneticAlgorithm core;

    /// <summary>
    /// Initializes a new real-coded genetic algorithm over the box <c>[lowerBounds[i], upperBounds[i]]</c>.
    /// <para>ボックス領域 <c>[lowerBounds[i], upperBounds[i]]</c> 上で最小化を行う実数値遺伝的アルゴリズムを初期化します。</para>
    /// </summary>
    /// <param name="lowerBounds">Finite lower bounds for each variable (non-empty). / 各変数の有限な下限（空でないこと）。</param>
    /// <param name="upperBounds">Finite upper bounds for each variable (at least the corresponding lower bound, with finite width). Variables with equal bounds stay fixed. / 各変数の有限な上限（対応する下限以上かつ幅が有限）。下限と等しい変数は固定されます。</param>
    /// <param name="options">Optional algorithm configuration, or <see langword="null"/> for defaults. / オプション設定。<see langword="null"/> の場合は既定値を使用します。</param>
    /// <exception cref="ArgumentException">The bounds are empty, differ in length, contain non-finite values, have a lower bound exceeding an upper bound, or have non-finite width. / 範囲が空または長さ不一致、非有限値を含む、下限が上限を超える、あるいは幅が有限でない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option value is out of range, or dimension times population size exceeds <see cref="int.MaxValue"/>. / 設定値が有効範囲外、または次元 × 個体数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public RealCodedGeneticAlgorithm(ReadOnlySpan<double> lowerBounds, ReadOnlySpan<double> upperBounds, RealCodedGeneticAlgorithmOptions? options = null)
    {
        if (lowerBounds.Length == 0 || lowerBounds.Length != upperBounds.Length)
        {
            throw new ArgumentException("The bounds must be non-empty and have the same length.", nameof(upperBounds));
        }

        GenomeLayout.ValidateRealBounds(lowerBounds, upperBounds, nameof(lowerBounds), nameof(upperBounds));
        core = new GeneticAlgorithm(new GenomeLayout(realLowerBounds: lowerBounds, realUpperBounds: upperBounds), options);
    }

    /// <summary>
    /// Gets the number of variables.
    /// <para>変数の数を取得します。</para>
    /// </summary>
    public int Dimension => core.Layout.RealCount;

    /// <summary>
    /// Gets the random seed used by this instance (pass the same seed in options to reproduce the candidate sequence).
    /// <para>このインスタンスで使用している乱数シードを取得します（同じシードを設定に渡すと候補列を再現できます）。</para>
    /// </summary>
    public ulong Seed => core.Seed;

    /// <summary>
    /// Gets the number of completed generations (including the initial population).
    /// <para>評価を完了した世代数（初期個体群を含む）を取得します。</para>
    /// </summary>
    public int Generations => core.Generations;

    /// <summary>
    /// Gets the total number of candidate evaluations reported to <see cref="Tell"/>.
    /// <para><see cref="Tell"/> に報告された候補評価回数の累計を取得します。</para>
    /// </summary>
    public long Evaluations => core.Evaluations;

    /// <summary>
    /// Gets a value indicating whether the algorithm has terminated.
    /// <para>最小化が終了したかどうかを取得します。</para>
    /// </summary>
    /// <value><see langword="true"/> if terminated (subsequent calls to <see cref="Ask"/> throw); otherwise, <see langword="false"/>. / 終了している場合は <see langword="true"/>（以降の <see cref="Ask"/> 呼び出しは例外を送出します）、それ以外は <see langword="false"/>。</value>
    public bool IsFinished => core.IsFinished;

    /// <summary>
    /// Gets the reason the algorithm terminated.
    /// <para>終了理由を取得します。</para>
    /// </summary>
    /// <value>The termination reason, or <see cref="GeneticAlgorithmTermination.None"/> while running. / 終了理由。実行中は <see cref="GeneticAlgorithmTermination.None"/>。</value>
    public GeneticAlgorithmTermination Termination => core.Termination;

    /// <inheritdoc cref="GeneticAlgorithm.EvaluationSeed"/>
    public ulong EvaluationSeed => core.EvaluationSeed;

    /// <summary>
    /// Gets the best objective value evaluated so far.
    /// <para>これまでに評価した候補の最良目的関数値を取得します。</para>
    /// </summary>
    /// <value>The minimum evaluated objective value, or <see cref="double.NaN"/> if all evaluations were <see cref="double.NaN"/>. / 最小の目的関数値。すべての評価が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</value>
    /// <exception cref="InvalidOperationException">No generation has been evaluated yet. / まだ1世代も評価されていない場合。</exception>
    public double BestFitness => core.BestFitness;

    /// <summary>
    /// Gets a view of the best point evaluated so far (retained even when <see cref="GeneticAlgorithmOptions.EliteCount"/> is 0).
    /// <para>これまでに評価した最良点へのビューを取得します（エリート数 0 の設定でも保持されます）。</para>
    /// </summary>
    /// <value>A view valid until the next call to <see cref="Tell"/>. / 次の <see cref="Tell"/> 呼び出しまで有効なビュー。</value>
    /// <exception cref="InvalidOperationException">No generation has been evaluated yet. / まだ1世代も評価されていない場合。</exception>
    public ReadOnlyVectorSpan<double> BestPoint => new(core.BestGenome.Reals);

    /// <summary>
    /// Gets the population from the most recently evaluated generation as a <see cref="Dimension"/> × population-size matrix view (one individual per column).
    /// <para>最後に評価を完了した世代の個体群を <see cref="Dimension"/> × 個体数の行列ビュー（1列 = 1個体）として取得します。</para>
    /// </summary>
    /// <value>A matrix view (leading columns are elites, followed by offspring), valid until the next call to <see cref="Tell"/>. / 先頭列にエリート、残りに子を格納した行列ビュー。次の <see cref="Tell"/> 呼び出しまで有効です。</value>
    /// <exception cref="InvalidOperationException">No generation has been evaluated yet. / まだ1世代も評価されていない場合。</exception>
    public ReadOnlyMatrixSpan<double> Population => core.Population.Reals;

    /// <summary>
    /// Gets the objective values corresponding to each column of <see cref="Population"/>.
    /// <para><see cref="Population"/> の各列に対応する目的関数値を取得します。</para>
    /// </summary>
    /// <exception cref="InvalidOperationException">No generation has been evaluated yet. / まだ1世代も評価されていない場合。</exception>
    public ReadOnlySpan<double> PopulationFitness => core.PopulationFitness;

    /// <summary>
    /// Returns the candidate points to evaluate for the current generation as a <see cref="Dimension"/> × candidate-count matrix view.
    /// <para>現在の世代で評価すべき候補点を <see cref="Dimension"/> × 候補数の行列ビューとして取得します。</para>
    /// </summary>
    /// <returns>A matrix view with one candidate per column. Count equals population size on the first generation (or always when <see cref="GeneticAlgorithmOptions.ReevaluateElites"/> is <see langword="true"/>, with elites first), and <c>PopulationSize − EliteCount</c> on subsequent generations. Valid until the next <see cref="Tell"/>; repeated calls before <see cref="Tell"/> return the same candidates. / 1列に1候補を格納する行列ビュー。候補数は初回（または <see cref="GeneticAlgorithmOptions.ReevaluateElites"/> が <see langword="true"/> のとき常時）が個体数、それ以降は <c>PopulationSize − EliteCount</c> です。次の <see cref="Tell"/> まで有効で、その間の再呼び出しでは同じ候補を返します。</returns>
    /// <exception cref="InvalidOperationException">The algorithm has already terminated. / 既に終了している場合。</exception>
    public ReadOnlyMatrixSpan<double> Ask() => core.Ask().Reals;

    /// <summary>
    /// Reports objective values for the candidate columns returned by <see cref="Ask"/> and advances to the next generation.
    /// <para><see cref="Ask"/> が返した候補列の目的関数値を順に報告し、世代を1つ進めます。</para>
    /// </summary>
    /// <param name="values">Objective values aligned with the candidate columns returned by <see cref="Ask"/>. <see cref="double.NaN"/> is ranked worst. / <see cref="Ask"/> が返した候補列に対応する目的関数値。<see cref="double.NaN"/> は最悪値として扱われます。</param>
    /// <remarks>
    /// <para>After advancing the generation, termination criteria are checked in the order: target fitness (<see cref="GeneticAlgorithmOptions.TargetFitness"/>), maximum generations, and stagnation.</para>
    /// <para>世代更新後、目標値到達（<see cref="GeneticAlgorithmOptions.TargetFitness"/>）・最大世代数・停滞の順に終了条件を判定します。</para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">No pending batch from <see cref="Ask"/> exists. / 対応する <see cref="Ask"/> が呼ばれていない場合。</exception>
    /// <exception cref="ArgumentException">The length of <paramref name="values"/> does not match the candidate count. / <paramref name="values"/> の長さが候補数と一致しない場合。</exception>
    public void Tell(ReadOnlySpan<double> values) => core.Tell(values);

    /// <summary>
    /// Runs the genetic algorithm to termination using the specified delegate objective.
    /// <para>指定したデリゲート目的関数を用いて終了条件まで実数値遺伝的アルゴリズムを実行します。</para>
    /// </summary>
    /// <param name="objective">The objective delegate to minimize. / 最小化する目的関数デリゲート。</param>
    /// <param name="cancellationToken">A cancellation token checked between generations. / 世代の区切りで確認されるキャンセルトークン。</param>
    /// <returns>The result containing a copy of the best point, best fitness, generation and evaluation counts, and termination reason. / 最良点のコピー、最良目的関数値、世代数、評価回数、および終了理由を含む結果。</returns>
    /// <remarks>
    /// <para>Exceptions thrown by <paramref name="objective"/> propagate directly; the pending generation remains active and can be retrieved again with <see cref="Ask"/>. If already finished, returns the existing result without further evaluations.</para>
    /// <para><paramref name="objective"/> から送出された例外はそのまま伝播します。評価待ちの世代は保持されるため <see cref="Ask"/> で同じ候補を再取得できます。既に終了している場合は評価を行わずに結果を返します。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="objective"/> is <see langword="null"/>. / <paramref name="objective"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was canceled. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public GeneticAlgorithmResult Run(Func<ReadOnlyVectorSpan<double>, double> objective, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(objective);
        return Run(new DelegateObjective(objective), cancellationToken);
    }

    /// <summary>
    /// Runs the genetic algorithm to termination using the specified objective function.
    /// <para>指定した目的関数を用いて終了条件まで実数値遺伝的アルゴリズムを実行します。</para>
    /// </summary>
    /// <typeparam name="TObjective">The objective function type. / 目的関数の型。</typeparam>
    /// <param name="objective">The objective function to minimize. / 最小化する目的関数。</param>
    /// <param name="cancellationToken">A cancellation token checked between generations. / 世代の区切りで確認されるキャンセルトークン。</param>
    /// <returns>The result containing a copy of the best point, best fitness, generation and evaluation counts, and termination reason. / 最良点のコピー、最良目的関数値、世代数、評価回数、および終了理由を含む結果。</returns>
    /// <remarks>
    /// <para>Exceptions thrown by <paramref name="objective"/> propagate directly; the pending generation remains active and can be retrieved again with <see cref="Ask"/>. If already finished, returns the existing result without further evaluations.</para>
    /// <para><paramref name="objective"/> から送出された例外はそのまま伝播します。評価待ちの世代は保持されるため <see cref="Ask"/> で同じ候補を再取得できます。既に終了している場合は評価を行わずに結果を返します。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="objective"/> is <see langword="null"/>. / <paramref name="objective"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was canceled. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public GeneticAlgorithmResult Run<TObjective>(TObjective objective, CancellationToken cancellationToken = default)
        where TObjective : IObjectiveFunction
    {
        ArgumentNullException.ThrowIfNull(objective);
        var result = core.Run(new RealsObjective<TObjective>(objective), cancellationToken);
        return new GeneticAlgorithmResult(new ReadOnlyVectorSpan<double>(result.BestPoint.Reals).ToVector(),
            result.BestFitness, result.Generations, result.Evaluations, result.Termination);
    }

    private readonly struct RealsObjective<TObjective>(TObjective objective) : IGenomeObjective
        where TObjective : IObjectiveFunction
    {
        public double Evaluate(GenomeView genome) => objective.Evaluate(new ReadOnlyVectorSpan<double>(genome.Reals));
    }

    private readonly struct DelegateObjective(Func<ReadOnlyVectorSpan<double>, double> objective) : IObjectiveFunction
    {
        public double Evaluate(ReadOnlyVectorSpan<double> x) => objective(x);
    }
}
