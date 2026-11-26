using Tuna.RandomSource;

namespace Tuna.Optimization.Evolutionary;

/// <summary>
/// Minimizes an objective function over mixed-type genomes (real, integer, binary, categorical, and permutation genes) using a genetic algorithm.
/// <para>実数・整数・二値・カテゴリ・順列から成る混合型ゲノム上で目的関数を最小化する遺伝的アルゴリズムです。</para>
/// </summary>
/// <remarks>
/// <para>Use the ask-tell loop (<see cref="Ask"/> and <see cref="Tell"/>) for external or parallel evaluation, or <see cref="Run{TObjective}(TObjective, CancellationToken)"/> for synchronous evaluation. Smaller fitness values are preferred, <see cref="double.NaN"/> is ranked worst, and ties break by smaller column index. Identical seeds, options, layouts, and evaluation sequences reproduce identical candidate sequences (matching <see cref="RealCodedGeneticAlgorithm"/> when only real genes are present). Instances are not thread-safe except that <see cref="EvaluationSeed"/> may be read concurrently while evaluating candidates.</para>
/// <para>外部評価や並列評価を行う場合は <see cref="Ask"/> と <see cref="Tell"/> を、同期評価には <see cref="Run{TObjective}(TObjective, CancellationToken)"/> を使用します。目的関数値は小さいほど優先され、<see cref="double.NaN"/> は最悪、同値時は列インデックスの小さい候補を優先します。同一のシード・設定・レイアウト・評価値列からは常に同一の候補列が再現されます（実数遺伝子のみの場合は <see cref="RealCodedGeneticAlgorithm"/> と一致）。インスタンス操作は単一スレッドで行ってください（ただし候補の評価中に <see cref="EvaluationSeed"/> を並列読み取りすることは安全です）。</para>
/// </remarks>
public sealed class GeneticAlgorithm
{
    private enum Purpose : ulong { Initialization = 0, Selection = 1, Crossover = 2, Mutation = 3, Evaluation = 4 }

    private readonly int populationSize;
    private readonly int eliteCount;
    private readonly int tournamentSize;
    private readonly double crossoverProbability;
    private readonly double crossoverExponent;
    private readonly double mutationProbability;
    private readonly double mutationIndex;
    private readonly int maxGenerations;
    private readonly double? targetFitness;
    private readonly int? stagnationGenerations;
    private readonly bool reevaluateElites;
    private readonly Philox4x32Random random;

    private GenomeStore population;
    private GenomeStore next;
    private readonly GenomeStore candidates;
    private readonly GenomeStore spare;
    private readonly GenomeStore best;
    private double[] fitness;
    private double[] nextFitness;
    private readonly int[] order;
    private readonly int[] rank;
    private readonly bool[] scratch;
    private double bestFitness = double.NaN;
    private int candidateCount;
    private bool pending;
    private int stagnantGenerations;

    /// <summary>
    /// Initializes a new genetic algorithm over the specified genome layout.
    /// <para>指定したゲノム構成上で最小化を行う遺伝的アルゴリズムを初期化します。</para>
    /// </summary>
    /// <param name="layout">The genome layout defining gene sections and bounds. / 遺伝子区画と範囲を定義するゲノム構成。</param>
    /// <param name="options">Optional algorithm configuration, or <see langword="null"/> for defaults. / オプション設定。<see langword="null"/> の場合は既定値を使用します。</param>
    /// <exception cref="ArgumentNullException"><paramref name="layout"/> is <see langword="null"/>. / <paramref name="layout"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option value is out of range, or the gene count of any section times <see cref="GeneticAlgorithmOptions.PopulationSize"/> exceeds <see cref="int.MaxValue"/>. / 設定値が有効範囲外、またはいずれかの区画の遺伝子数 × 個体数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public GeneticAlgorithm(GenomeLayout layout, GeneticAlgorithmOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(layout);
        options ??= new GeneticAlgorithmOptions();
        populationSize = options.PopulationSize;
        ArgumentOutOfRangeException.ThrowIfLessThan(populationSize, 2, nameof(options.PopulationSize));
        long largest = Math.Max(Math.Max(Math.Max(layout.RealCount, layout.IntegerCount), Math.Max(layout.BitCount, layout.CategoryCount)),
            layout.PermutationElementCount);
        if (largest * populationSize > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(options.PopulationSize), "The genes in a section times the population size must not exceed int.MaxValue.");
        }

        eliteCount = options.EliteCount;
        ArgumentOutOfRangeException.ThrowIfNegative(eliteCount, nameof(options.EliteCount));
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(eliteCount, populationSize, nameof(options.EliteCount));
        tournamentSize = options.TournamentSize;
        ArgumentOutOfRangeException.ThrowIfLessThan(tournamentSize, 1, nameof(options.TournamentSize));
        ArgumentOutOfRangeException.ThrowIfGreaterThan(tournamentSize, populationSize, nameof(options.TournamentSize));
        crossoverProbability = Probability(options.CrossoverProbability, nameof(options.CrossoverProbability));
        crossoverExponent = DistributionIndex(options.CrossoverDistributionIndex, nameof(options.CrossoverDistributionIndex)) + 1;
        mutationProbability = options.MutationProbability is { } pm
            ? Probability(pm, nameof(options.MutationProbability))
            : 1.0 / layout.MutationUnitCount;
        mutationIndex = DistributionIndex(options.MutationDistributionIndex, nameof(options.MutationDistributionIndex));
        maxGenerations = options.MaxGenerations;
        ArgumentOutOfRangeException.ThrowIfLessThan(maxGenerations, 1, nameof(options.MaxGenerations));
        if (options.TargetFitness is { } target && double.IsNaN(target))
        {
            throw new ArgumentOutOfRangeException(nameof(options.TargetFitness), "The target fitness must not be NaN.");
        }

        targetFitness = options.TargetFitness;
        if (options.StagnationGenerations is { } stagnation)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(stagnation, 1, nameof(options.StagnationGenerations));
        }

        stagnationGenerations = options.StagnationGenerations;
        reevaluateElites = options.ReevaluateElites;

        Layout = layout;
        random = options.Seed is { } seed ? new Philox4x32Random(seed) : new Philox4x32Random();
        Seed = random.GetState().Seed;
        population = new GenomeStore(layout, populationSize);
        next = new GenomeStore(layout, populationSize);
        candidates = new GenomeStore(layout, populationSize);
        spare = new GenomeStore(layout, 1);
        best = new GenomeStore(layout, 1);
        fitness = new double[populationSize];
        nextFitness = new double[populationSize];
        order = new int[populationSize];
        rank = new int[populationSize];
        int longest = 0;
        foreach (int length in layout.PermutationLengths)
        {
            longest = Math.Max(longest, length);
        }

        scratch = new bool[longest];
    }

    /// <summary>
    /// Gets the genome layout.
    /// <para>ゲノム構成を取得します。</para>
    /// </summary>
    public GenomeLayout Layout { get; }

    /// <summary>
    /// Gets the random seed used by this instance (pass the same seed in options to reproduce the candidate sequence).
    /// <para>このインスタンスで使用している乱数シードを取得します（同じシードを設定に渡すと候補列を再現できます）。</para>
    /// </summary>
    public ulong Seed { get; }

    /// <summary>
    /// Gets the number of completed generations (including the initial population).
    /// <para>評価を完了した世代数（初期個体群を含む）を取得します。</para>
    /// </summary>
    public int Generations { get; private set; }

    /// <summary>
    /// Gets the total number of candidate evaluations reported to <see cref="Tell"/>.
    /// <para><see cref="Tell"/> に報告された候補評価回数の累計を取得します。</para>
    /// </summary>
    public long Evaluations { get; private set; }

    /// <summary>
    /// Gets a value indicating whether the algorithm has terminated.
    /// <para>最小化が終了したかどうかを取得します。</para>
    /// </summary>
    /// <value><see langword="true"/> if terminated (subsequent calls to <see cref="Ask"/> throw); otherwise, <see langword="false"/>. / 終了している場合は <see langword="true"/>（以降の <see cref="Ask"/> 呼び出しは例外を送出します）、それ以外は <see langword="false"/>。</value>
    public bool IsFinished => Termination != GeneticAlgorithmTermination.None;

    /// <summary>
    /// Gets the reason the algorithm terminated.
    /// <para>終了理由を取得します。</para>
    /// </summary>
    /// <value>The termination reason, or <see cref="GeneticAlgorithmTermination.None"/> while running. / 終了理由。実行中は <see cref="GeneticAlgorithmTermination.None"/>。</value>
    public GeneticAlgorithmTermination Termination { get; private set; }

    /// <summary>
    /// Gets a deterministic random seed derived from <see cref="Seed"/> and the current generation for stochastic candidate evaluation.
    /// <para>確率的な候補評価に使用するため、<see cref="Seed"/> と現在の世代番号から決定的に導出した乱数シードを取得します。</para>
    /// </summary>
    /// <value>A deterministic seed for the pending generation (or next generation returned by <see cref="Ask"/>). Safe to read concurrently during evaluation without disturbing the internal generator state. / 評価待ち世代（または次の <see cref="Ask"/> の世代）用の決定的シード。内部乱数生成器の状態に影響せず、評価中に並列読み取り可能です。</value>
    /// <remarks>
    /// <para>Use this seed directly for common random numbers across a generation, or combine it with the candidate column index as a stream ID for independent per-candidate streams.</para>
    /// <para>世代内で共通の乱数を使う場合はこの値をそのままシードにし、候補ごとに独立した乱数ストリームが必要な場合は列インデックスをストリーム ID として組み合わせてください。</para>
    /// </remarks>
    public ulong EvaluationSeed
    {
        get
        {
            Philox4x32Block block = Philox4x32Random.GenerateBlock(Seed, ((ulong)Purpose.Evaluation << 56) | (uint)Generations, 0);
            return ((ulong)block.Word1 << 32) | block.Word0;
        }
    }

    /// <summary>
    /// Gets the best objective value evaluated so far.
    /// <para>これまでに評価した候補の最良目的関数値を取得します。</para>
    /// </summary>
    /// <value>The minimum evaluated objective value, or <see cref="double.NaN"/> if all evaluations were <see cref="double.NaN"/>. / 最小の目的関数値。すべての評価が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</value>
    /// <exception cref="InvalidOperationException">No generation has been evaluated yet. / まだ1世代も評価されていない場合。</exception>
    public double BestFitness
    {
        get
        {
            RequireEvaluated();
            return bestFitness;
        }
    }

    /// <summary>
    /// Gets a view of the best genome evaluated so far (retained even when <see cref="GeneticAlgorithmOptions.EliteCount"/> is 0).
    /// <para>これまでに評価した最良ゲノムへのビューを取得します（エリート数 0 の設定でも保持されます）。</para>
    /// </summary>
    /// <value>A view valid until the next call to <see cref="Tell"/>. / 次の <see cref="Tell"/> 呼び出しまで有効なビュー。</value>
    /// <exception cref="InvalidOperationException">No generation has been evaluated yet. / まだ1世代も評価されていない場合。</exception>
    public GenomeView BestGenome
    {
        get
        {
            RequireEvaluated();
            return best.View(0);
        }
    }

    /// <summary>
    /// Gets the population from the most recently evaluated generation.
    /// <para>最後に評価を完了した世代の個体群を取得します。</para>
    /// </summary>
    /// <value>A batch view (leading columns are elites, followed by offspring), valid until the next call to <see cref="Tell"/>. / 先頭列にエリート、残りに子を格納したバッチビュー。次の <see cref="Tell"/> 呼び出しまで有効です。</value>
    /// <exception cref="InvalidOperationException">No generation has been evaluated yet. / まだ1世代も評価されていない場合。</exception>
    public GenomeBatch Population
    {
        get
        {
            RequireEvaluated();
            return population.Batch(populationSize);
        }
    }

    /// <summary>
    /// Gets the objective values corresponding to each individual in <see cref="Population"/>.
    /// <para><see cref="Population"/> の各個体に対応する目的関数値を取得します。</para>
    /// </summary>
    /// <exception cref="InvalidOperationException">No generation has been evaluated yet. / まだ1世代も評価されていない場合。</exception>
    public ReadOnlySpan<double> PopulationFitness
    {
        get
        {
            RequireEvaluated();
            return fitness;
        }
    }

    /// <summary>
    /// Returns the batch of candidate genomes to evaluate for the current generation.
    /// <para>現在の世代で評価すべき候補ゲノムのバッチを取得します。</para>
    /// </summary>
    /// <returns>A batch view with one candidate per column. Count equals population size on the first generation (or always when <see cref="GeneticAlgorithmOptions.ReevaluateElites"/> is <see langword="true"/>, with elites first), and <c>PopulationSize − EliteCount</c> on subsequent generations. Valid until the next <see cref="Tell"/>; repeated calls before <see cref="Tell"/> return the same candidates. / 1列に1候補を格納するバッチビュー。候補数は初回（または <see cref="GeneticAlgorithmOptions.ReevaluateElites"/> が <see langword="true"/> のとき常時）が個体数、それ以降は <c>PopulationSize − EliteCount</c> です。次の <see cref="Tell"/> まで有効で、その間の再呼び出しでは同じ候補を返します。</returns>
    /// <exception cref="InvalidOperationException">The algorithm has already terminated. / 既に終了している場合。</exception>
    public GenomeBatch Ask()
    {
        if (IsFinished)
        {
            throw new InvalidOperationException("The genetic algorithm has already terminated.");
        }

        if (!pending)
        {
            if (Generations == 0)
            {
                Initialize();
            }
            else
            {
                Reproduce();
            }

            pending = true;
        }

        return candidates.Batch(candidateCount);
    }

    /// <summary>
    /// Reports objective values for the candidates returned by <see cref="Ask"/> and advances to the next generation.
    /// <para><see cref="Ask"/> が返した候補の目的関数値を順に報告し、世代を1つ進めます。</para>
    /// </summary>
    /// <param name="values">Objective values aligned with the candidates returned by <see cref="Ask"/>. <see cref="double.NaN"/> is ranked worst. / <see cref="Ask"/> が返した候補に対応する目的関数値。<see cref="double.NaN"/> は最悪値として扱われます。</param>
    /// <remarks>
    /// <para>After advancing the generation, termination criteria are checked in the order: target fitness, maximum generations, and stagnation.</para>
    /// <para>世代更新後、目標値到達・最大世代数・停滞の順に終了条件を判定します。</para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">No pending batch from <see cref="Ask"/> exists. / 対応する <see cref="Ask"/> が呼ばれていない場合。</exception>
    /// <exception cref="ArgumentException">The length of <paramref name="values"/> does not match the candidate count. / <paramref name="values"/> の長さが候補数と一致しない場合。</exception>
    public void Tell(ReadOnlySpan<double> values)
    {
        if (!pending)
        {
            throw new InvalidOperationException("Call Ask before Tell.");
        }

        if (values.Length != candidateCount)
        {
            throw new ArgumentException($"Expected {candidateCount} values, one per candidate.", nameof(values));
        }

        int carried = Generations == 0 || reevaluateElites ? 0 : eliteCount;
        for (int k = 0; k < carried; k++)
        {
            population.CopyTo(order[k], next, k);
            nextFitness[k] = fitness[order[k]];
        }

        candidates.CopyLeadingTo(candidateCount, next, carried);
        values.CopyTo(nextFitness.AsSpan(carried));
        (population, next) = (next, population);
        (fitness, nextFitness) = (nextFitness, fitness);
        pending = false;
        Generations++;
        Evaluations += candidateCount;

        for (int i = 0; i < populationSize; i++)
        {
            order[i] = i;
        }

        order.AsSpan().Sort(new FitnessOrder(fitness));
        for (int r = 0; r < populationSize; r++)
        {
            rank[order[r]] = r;
        }

        double generationBest = fitness[order[0]];
        if (Generations == 1 || RealVariation.IsBetter(generationBest, bestFitness))
        {
            bestFitness = generationBest;
            population.CopyTo(order[0], best, 0);
            stagnantGenerations = 0;
        }
        else
        {
            stagnantGenerations++;
        }

        if (targetFitness is { } goal && bestFitness <= goal)
        {
            Termination = GeneticAlgorithmTermination.TargetReached;
        }
        else if (Generations >= maxGenerations)
        {
            Termination = GeneticAlgorithmTermination.MaxGenerations;
        }
        else if (stagnationGenerations is { } limit && stagnantGenerations >= limit)
        {
            Termination = GeneticAlgorithmTermination.Stagnation;
        }
    }

    /// <summary>
    /// Runs the genetic algorithm to termination using the specified delegate objective.
    /// <para>指定したデリゲート目的関数を用いて終了条件まで遺伝的アルゴリズムを実行します。</para>
    /// </summary>
    /// <param name="objective">The objective delegate to minimize. / 最小化する目的関数デリゲート。</param>
    /// <param name="cancellationToken">A cancellation token checked between generations. / 世代の区切りで確認されるキャンセルトークン。</param>
    /// <returns>The result containing a copy of the best genome, best fitness, generation and evaluation counts, and termination reason. / 最良ゲノムのコピー、最良目的関数値、世代数、評価回数、および終了理由を含む結果。</returns>
    /// <remarks>
    /// <para>Exceptions thrown by <paramref name="objective"/> propagate directly; the pending generation remains active and can be retrieved again with <see cref="Ask"/>. If already finished, returns the existing result without further evaluations.</para>
    /// <para><paramref name="objective"/> から送出された例外はそのまま伝播します。評価待ちの世代は保持されるため <see cref="Ask"/> で同じ候補を再取得できます。既に終了している場合は評価を行わずに結果を返します。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="objective"/> is <see langword="null"/>. / <paramref name="objective"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was canceled. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public GeneticAlgorithmResult<Genome> Run(Func<GenomeView, double> objective, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(objective);
        return Run(new DelegateObjective(objective), cancellationToken);
    }

    /// <summary>
    /// Runs the genetic algorithm to termination using the specified genome objective.
    /// <para>指定したゲノム目的関数を用いて終了条件まで遺伝的アルゴリズムを実行します。</para>
    /// </summary>
    /// <typeparam name="TObjective">The genome objective type. / ゲノム目的関数の型。</typeparam>
    /// <param name="objective">The objective function to minimize. / 最小化する目的関数。</param>
    /// <param name="cancellationToken">A cancellation token checked between generations. / 世代の区切りで確認されるキャンセルトークン。</param>
    /// <returns>The result containing a copy of the best genome, best fitness, generation and evaluation counts, and termination reason. / 最良ゲノムのコピー、最良目的関数値、世代数、評価回数、および終了理由を含む結果。</returns>
    /// <remarks>
    /// <para>Exceptions thrown by <paramref name="objective"/> propagate directly; the pending generation remains active and can be retrieved again with <see cref="Ask"/>. If already finished, returns the existing result without further evaluations.</para>
    /// <para><paramref name="objective"/> から送出された例外はそのまま伝播します。評価待ちの世代は保持されるため <see cref="Ask"/> で同じ候補を再取得できます。既に終了している場合は評価を行わずに結果を返します。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="objective"/> is <see langword="null"/>. / <paramref name="objective"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was canceled. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public GeneticAlgorithmResult<Genome> Run<TObjective>(TObjective objective, CancellationToken cancellationToken = default)
        where TObjective : IGenomeObjective
    {
        ArgumentNullException.ThrowIfNull(objective);
        var values = new double[populationSize];
        while (!IsFinished)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var batch = Ask();
            for (int j = 0; j < batch.Count; j++)
            {
                values[j] = objective.Evaluate(batch[j]);
            }

            Tell(values.AsSpan(0, batch.Count));
        }

        return new GeneticAlgorithmResult<Genome>(best.View(0).ToGenome(), bestFitness, Generations, Evaluations, Termination);
    }

    private void Initialize()
    {
        GenomeLayout layout = Layout;
        for (int i = 0; i < populationSize; i++)
        {
            Position(Purpose.Initialization, i);
            Span<double> reals = candidates.RealsOf(i);
            ReadOnlySpan<double> realLower = layout.RealLowerBounds, realUpper = layout.RealUpperBounds;
            for (int d = 0; d < reals.Length; d++)
            {
                double u = random.NextDouble();
                reals[d] = Math.Min(realLower[d] + (realUpper[d] - realLower[d]) * u, realUpper[d]);
            }

            Span<long> integers = candidates.IntegersOf(i);
            ReadOnlySpan<long> integerLower = layout.IntegerLowerBounds, integerUpper = layout.IntegerUpperBounds;
            for (int d = 0; d < integers.Length; d++)
            {
                ulong width = unchecked((ulong)(integerUpper[d] - integerLower[d]));
                integers[d] = unchecked(integerLower[d] + (long)random.NextUInt64(width + 1));
            }

            Span<bool> bits = candidates.BitsOf(i);
            for (int d = 0; d < bits.Length; d++)
            {
                bits[d] = random.NextDouble() < 0.5;
            }

            Span<int> categories = candidates.CategoriesOf(i);
            ReadOnlySpan<int> counts = layout.CategoryCounts;
            for (int d = 0; d < categories.Length; d++)
            {
                categories[d] = random.NextInt32(counts[d]);
            }

            Span<int> permutations = candidates.PermutationsOf(i);
            for (int k = 0; k < layout.PermutationCount; k++)
            {
                random.Permutation(permutations.Slice(layout.PermutationOffset(k), layout.PermutationLengths[k]));
            }
        }

        candidateCount = populationSize;
    }

    private void Reproduce()
    {
        int children = populationSize - eliteCount;
        int offset = 0;
        if (reevaluateElites)
        {
            for (int k = 0; k < eliteCount; k++)
            {
                population.CopyTo(order[k], candidates, k);
            }

            offset = eliteCount;
        }

        for (int pair = 0; 2 * pair < children; pair++)
        {
            Position(Purpose.Selection, pair);
            int first = RealVariation.Tournament(rank, tournamentSize, random);
            int second = RealVariation.Tournament(rank, tournamentSize, random);
            bool hasSecond = 2 * pair + 1 < children;
            Position(Purpose.Crossover, pair);
            Cross(first, second, offset + 2 * pair, hasSecond ? candidates : spare, hasSecond ? offset + 2 * pair + 1 : 0);
        }

        for (int c = 0; c < children; c++)
        {
            Position(Purpose.Mutation, c);
            Mutate(offset + c);
        }

        candidateCount = offset + children;
    }

    private void Cross(int first, int second, int column1, GenomeStore store2, int column2)
    {
        population.CopyTo(first, candidates, column1);
        population.CopyTo(second, store2, column2);
        if (random.NextDouble() >= crossoverProbability)
        {
            return;
        }

        GenomeLayout layout = Layout;
        RealVariation.Cross(population.RealsOf(first), population.RealsOf(second), candidates.RealsOf(column1), store2.RealsOf(column2),
            layout.RealLowerBounds, layout.RealUpperBounds, crossoverExponent, random);
        DiscreteVariation.CrossIntegers(population.IntegersOf(first), population.IntegersOf(second), candidates.IntegersOf(column1), store2.IntegersOf(column2),
            layout.IntegerLowerBounds, layout.IntegerUpperBounds, crossoverExponent, random);
        DiscreteVariation.UniformCrossover(candidates.BitsOf(column1), store2.BitsOf(column2), random);
        DiscreteVariation.UniformCrossover(candidates.CategoriesOf(column1), store2.CategoriesOf(column2), random);
        Span<int> parent1 = population.PermutationsOf(first), parent2 = population.PermutationsOf(second);
        Span<int> child1 = candidates.PermutationsOf(column1), child2 = store2.PermutationsOf(column2);
        for (int k = 0; k < layout.PermutationCount; k++)
        {
            int start = layout.PermutationOffset(k), length = layout.PermutationLengths[k];
            DiscreteVariation.OrderCrossover(parent1.Slice(start, length), parent2.Slice(start, length),
                child1.Slice(start, length), child2.Slice(start, length), random, scratch);
        }
    }

    private void Mutate(int column)
    {
        GenomeLayout layout = Layout;
        RealVariation.PolynomialMutation(candidates.RealsOf(column), layout.RealLowerBounds, layout.RealUpperBounds,
            mutationProbability, mutationIndex, random);
        DiscreteVariation.MutateIntegers(candidates.IntegersOf(column), layout.IntegerLowerBounds, layout.IntegerUpperBounds,
            mutationProbability, mutationIndex + 1, random);
        DiscreteVariation.FlipBits(candidates.BitsOf(column), mutationProbability, random);
        DiscreteVariation.MutateCategories(candidates.CategoriesOf(column), layout.CategoryCounts, mutationProbability, random);
        Span<int> permutations = candidates.PermutationsOf(column);
        for (int k = 0; k < layout.PermutationCount; k++)
        {
            DiscreteVariation.Invert(permutations.Slice(layout.PermutationOffset(k), layout.PermutationLengths[k]), mutationProbability, random);
        }
    }

    // Counter-based Philox positioning isolates RNG streams by (purpose, generation, index) for deterministic reproducibility.
    private void Position(Purpose purpose, int index) =>
        random.RestoreState(new Philox4x32State(Seed, ((ulong)purpose << 56) | (uint)Generations, (ulong)(uint)index << 32, 0));

    private void RequireEvaluated()
    {
        if (Generations == 0)
        {
            throw new InvalidOperationException("No generation has been evaluated yet.");
        }
    }

    private static double Probability(double value, string name)
    {
        if (!(value >= 0 && value <= 1))
        {
            throw new ArgumentOutOfRangeException(name, value, "The probability must be between 0 and 1.");
        }

        return value;
    }

    private static double DistributionIndex(double value, string name)
    {
        if (!(value >= 0) || !double.IsFinite(value))
        {
            throw new ArgumentOutOfRangeException(name, value, "The distribution index must be finite and non-negative.");
        }

        return value;
    }

    // Total order (ties broken by index) makes sorting deterministic independent of sort stability.
    private readonly struct FitnessOrder(double[] fitness) : IComparer<int>
    {
        public int Compare(int a, int b) => RealVariation.Compare(fitness[a], a, fitness[b], b);
    }

    private readonly struct DelegateObjective(Func<GenomeView, double> objective) : IGenomeObjective
    {
        public double Evaluate(GenomeView genome) => objective(genome);
    }
}
