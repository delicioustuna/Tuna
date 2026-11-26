using Tuna.LinearAlgebra;
using Tuna.RandomSource;

namespace Tuna.Probabilistic.Inference;

/// <summary>
/// Configures <see cref="RandomWalkMetropolis"/>.
/// <para><see cref="RandomWalkMetropolis"/> の設定オプションを表します。</para>
/// </summary>
public sealed class RandomWalkMetropolisOptions : MultiChainSamplerOptions
{
    /// <summary>
    /// Gets or initializes the target acceptance rate during warmup (in <c>(0, 1)</c>).
    /// <para>ウォームアップ期間中の目標受理率を取得または設定します（<c>(0, 1)</c>）。</para>
    /// </summary>
    /// <value>A value in <c>(0, 1)</c>, or <see langword="null"/> (default) for 0.44 in 1 dimension and 0.234 in 2 or more dimensions. / <c>(0, 1)</c> の値、または <see langword="null"/>（既定。1 次元で 0.44、2 次元以上で 0.234）。</value>
    public double? TargetAcceptance { get; init; }

    /// <summary>
    /// Gets or initializes the initial proposal scale factor <c>s</c> (positive finite value; defaults to <c>2.38 / √d</c> when <see langword="null"/>). Used unchanged when <see cref="SamplerOptions.Warmup"/> is 0.
    /// <para>提案分布の初期スケール係数 <c>s</c> を取得または設定します（正の有限値。<see langword="null"/> の既定値は <c>2.38 / √d</c>）。ウォームアップなしの場合はこの値がそのまま使用されます。</para>
    /// </summary>
    public double? ProposalScale { get; init; }

    /// <summary>
    /// Gets or initializes the initial proposal shape matrix <c>C</c> (<c>d × d</c> symmetric positive definite; defaults to the identity matrix when <see langword="null"/>). Proposals are drawn as <c>θ + s·L·z</c> where <c>C = LLᵀ</c>, giving covariance <c>s²C</c>.
    /// <para>提案分布の初期形状行列 <c>C</c> を取得または設定します（<c>d × d</c> の対称正定値行列。<see langword="null"/> の既定値は単位行列）。提案は <c>θ + s·L·z</c>（<c>C = LLᵀ</c>）で生成され、提案共分散は <c>s²C</c> です。</para>
    /// </summary>
    public Matrix<double>? InitialCovariance { get; init; }

    /// <summary>
    /// Gets or initializes a value indicating whether the proposal shape is updated from regularized sample covariance at the end of each warmup window (default is <see langword="true"/>).
    /// <para>ウォームアップの各ウィンドウ終端で、正則化した標本共分散から提案分布の形状を更新するかどうかを取得または設定します（既定値 <see langword="true"/>）。</para>
    /// </summary>
    /// <value><see langword="true"/> to adapt the proposal shape per window; <see langword="false"/> to keep the initial shape fixed and adapt only the scale. / <see langword="true"/> の場合はウィンドウごとに形状を更新し、<see langword="false"/> の場合は形状を固定してスケールのみを適応させます。</value>
    public bool AdaptCovariance { get; init; } = true;

    /// <summary>
    /// Gets or initializes the warmup adaptation window schedule.
    /// <para>ウォームアップ期間の適応ウィンドウ設定を取得または設定します。</para>
    /// </summary>
    public WarmupWindowOptions Windows { get; init; } = new();
}

/// <summary>
/// Samples from a log-density using an adaptive Gaussian random-walk Metropolis algorithm.
/// <para>正規分布の提案を用いる適応ランダムウォーク・メトロポリス法により、対数密度からサンプリングします。</para>
/// </summary>
/// <remarks>
/// <para>Gradients are not required and discontinuous log-densities are supported. Proposal scale and covariance are adapted during warmup and frozen during sampling.</para>
/// <para>勾配は不要であり、不連続な対数密度にも適用できます。提案分布のスケールと共分散はウォームアップ期間中に適応させ、サンプリング期間中は固定します。</para>
/// </remarks>
public sealed class RandomWalkMetropolis
{
    private const double OptimalScaleNumerator = 2.38;

    private readonly SamplerSettings settings;
    private readonly int chains;
    private readonly double? targetAcceptance;
    private readonly double? proposalScale;
    private readonly Matrix<double>? initialCovariance;
    private readonly bool adaptCovariance;
    private readonly WarmupWindowOptions windows;

    /// <summary>
    /// Initializes a new instance of <see cref="RandomWalkMetropolis"/> with the specified options.
    /// <para>指定した設定で <see cref="RandomWalkMetropolis"/> の新しいインスタンスを初期化します。</para>
    /// </summary>
    /// <param name="options">The sampler options, or <see langword="null"/> for defaults. / サンプラー設定（<see langword="null"/> の場合は既定値）。</param>
    /// <exception cref="ArgumentNullException"><see cref="RandomWalkMetropolisOptions.Windows"/> is <see langword="null"/>. / <see cref="RandomWalkMetropolisOptions.Windows"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option value is out of its valid range. / 設定値が有効範囲外の場合。</exception>
    /// <exception cref="ArgumentException"><see cref="RandomWalkMetropolisOptions.InitialCovariance"/> is not square. / <see cref="RandomWalkMetropolisOptions.InitialCovariance"/> が正方行列でない場合。</exception>
    public RandomWalkMetropolis(RandomWalkMetropolisOptions? options = null)
    {
        options ??= new RandomWalkMetropolisOptions();
        settings = SamplerSettings.Validate(options);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.Chains, 1, nameof(options.Chains));
        chains = options.Chains;
        if (options.TargetAcceptance is { } target && !(target > 0 && target < 1))
        {
            throw new ArgumentOutOfRangeException(nameof(options.TargetAcceptance), "The target acceptance must lie in (0, 1).");
        }

        if (options.ProposalScale is { } scale && !(scale > 0 && double.IsFinite(scale)))
        {
            throw new ArgumentOutOfRangeException(nameof(options.ProposalScale), "The proposal scale must be positive and finite.");
        }

        if (options.InitialCovariance is { IsSquare: false })
        {
            throw new ArgumentException("The initial covariance must be square.", nameof(options));
        }

        targetAcceptance = options.TargetAcceptance;
        proposalScale = options.ProposalScale;
        initialCovariance = options.InitialCovariance?.Clone();
        adaptCovariance = options.AdaptCovariance;
        windows = options.Windows ?? throw new ArgumentNullException(nameof(options.Windows));
        _ = new WarmupSchedule(settings.Warmup, windows);
    }

    /// <summary>
    /// Gets the number of independent chains.
    /// <para>独立したチェーンの数を取得します。</para>
    /// </summary>
    public int Chains => chains;

    /// <summary>
    /// Gets the random seed used for sampling (configured value or obtained from the OS).
    /// <para>サンプリングに使用する乱数シード（指定値、または省略時に OS から取得した値）を取得します。</para>
    /// </summary>
    public ulong Seed => settings.Seed;

    /// <summary>
    /// Generates posterior draws starting all chains from the specified initial point.
    /// <para>すべてのチェーンを指定した初期点から開始して標本を生成します。</para>
    /// </summary>
    /// <param name="logDensity">The delegate evaluating the log-density. / 対数密度を評価するデリゲート。</param>
    /// <param name="initialPoint">The initial point shared by all chains (length equals parameter dimension). / 全チェーン共通の初期点（長さはパラメータ次元）。</param>
    /// <param name="cancellationToken">A token checked at each transition to cancel sampling. / 遷移ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws and sampler diagnostics. / ウォームアップ後の標本とサンプラー診断情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="logDensity"/> is <see langword="null"/>. / <paramref name="logDensity"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoint"/> is empty or non-finite, its log-density is non-finite, or <see cref="RandomWalkMetropolisOptions.InitialCovariance"/> has mismatched dimensions or is not positive definite. / <paramref name="initialPoint"/> が空または非有限、初期点の対数密度が非有限、または初期共分散行列の次元不一致や非正定値の場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public RandomWalkMetropolisResult Sample(Func<ReadOnlyVectorSpan<double>, double> logDensity, ReadOnlySpan<double> initialPoint,
        CancellationToken cancellationToken = default) =>
        Sample(LogDensities.FromDelegate(logDensity), initialPoint, cancellationToken);

    /// <summary>
    /// Generates posterior draws starting each chain from its corresponding column of <paramref name="initialPoints"/>.
    /// <para>チェーンごとに指定した初期点から標本を生成します。</para>
    /// </summary>
    /// <param name="logDensity">The delegate evaluating the log-density. / 対数密度を評価するデリゲート。</param>
    /// <param name="initialPoints">The matrix of initial points with shape <c>[dimension × chains]</c>. / 各チェーンの初期点を列に並べた行列（形状 <c>[次元 × チェーン数]</c>）。</param>
    /// <param name="cancellationToken">A token checked at each transition to cancel sampling. / 遷移ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws and sampler diagnostics. / ウォームアップ後の標本とサンプラー診断情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="logDensity"/> is <see langword="null"/>. / <paramref name="logDensity"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoints"/> is empty, has a column count different from <see cref="Chains"/>, contains non-finite values, has non-finite log-density, or <see cref="RandomWalkMetropolisOptions.InitialCovariance"/> is invalid. / <paramref name="initialPoints"/> が空または列数が <see cref="Chains"/> と異なる場合、非有限値を含む場合、初期点の対数密度が非有限の場合、または初期共分散行列が不正な場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public RandomWalkMetropolisResult Sample(Func<ReadOnlyVectorSpan<double>, double> logDensity, ReadOnlyMatrixSpan<double> initialPoints,
        CancellationToken cancellationToken = default) =>
        Sample(LogDensities.FromDelegate(logDensity), initialPoints, cancellationToken);

    /// <summary>
    /// Generates posterior draws starting all chains from the specified initial point.
    /// <para>すべてのチェーンを指定した初期点から開始して標本を生成します。</para>
    /// </summary>
    /// <typeparam name="TDensity">The log-density type. / 対数密度の型。</typeparam>
    /// <param name="density">The target log-density. / 対象分布の対数密度。</param>
    /// <param name="initialPoint">The initial point shared by all chains (length equals parameter dimension). / 全チェーン共通の初期点（長さはパラメータ次元）。</param>
    /// <param name="cancellationToken">A token checked at each transition to cancel sampling. / 遷移ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws and sampler diagnostics. / ウォームアップ後の標本とサンプラー診断情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="density"/> is <see langword="null"/>. / <paramref name="density"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoint"/> is empty or non-finite, its log-density is non-finite, or <see cref="RandomWalkMetropolisOptions.InitialCovariance"/> has mismatched dimensions or is not positive definite. / <paramref name="initialPoint"/> が空または非有限、初期点の対数密度が非有限、または初期共分散行列の次元不一致や非正定値の場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public RandomWalkMetropolisResult Sample<TDensity>(TDensity density, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)
        where TDensity : ILogDensity =>
        Run(density, initialPoint.Length, InitialPoints.Broadcast(initialPoint, chains), cancellationToken);

    /// <summary>
    /// Generates posterior draws starting each chain from its corresponding column of <paramref name="initialPoints"/>.
    /// <para>チェーンごとに指定した初期点から標本を生成します。</para>
    /// </summary>
    /// <typeparam name="TDensity">The log-density type. / 対数密度の型。</typeparam>
    /// <param name="density">The target log-density. / 対象分布の対数密度。</param>
    /// <param name="initialPoints">The matrix of initial points with shape <c>[dimension × chains]</c>. / 各チェーンの初期点を列に並べた行列（形状 <c>[次元 × チェーン数]</c>）。</param>
    /// <param name="cancellationToken">A token checked at each transition to cancel sampling. / 遷移ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws and sampler diagnostics. / ウォームアップ後の標本とサンプラー診断情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="density"/> is <see langword="null"/>. / <paramref name="density"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoints"/> is empty, has a column count different from <see cref="Chains"/>, contains non-finite values, has non-finite log-density, or <see cref="RandomWalkMetropolisOptions.InitialCovariance"/> is invalid. / <paramref name="initialPoints"/> が空または列数が <see cref="Chains"/> と異なる場合、非有限値を含む場合、初期点の対数密度が非有限の場合、または初期共分散行列が不正な場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public RandomWalkMetropolisResult Sample<TDensity>(TDensity density, ReadOnlyMatrixSpan<double> initialPoints, CancellationToken cancellationToken = default)
        where TDensity : ILogDensity =>
        Run(density, initialPoints.Rows, InitialPoints.Columns(initialPoints, chains), cancellationToken);

    private RandomWalkMetropolisResult Run<TDensity>(TDensity density, int d, double[] starts, CancellationToken cancellationToken)
        where TDensity : ILogDensity
    {
        ArgumentNullException.ThrowIfNull(density);
        if (initialCovariance is not null && initialCovariance.Rows != d)
        {
            throw new ArgumentException($"The initial covariance must be {d} x {d}.", "options");
        }

        settings.RequireStorage(d, chains);
        var draws = new Tensor<double>(d, settings.Draws, chains);
        var logDensities = new Matrix<double>(settings.Draws, chains);
        var covariances = new Tensor<double>(d, d, chains);
        var evaluations = new long[chains];
        var acceptance = new double[chains];
        var scales = new double[chains];
        ChainRunner.Run(chains, settings.MaxDegreeOfParallelism, cancellationToken, c =>
        {
            var chain = new Chain<TDensity>(this, density, d, c);
            chain.Run(starts.AsSpan(c * d, d), draws.GetMatrix(c), logDensities.Column(c), cancellationToken);
            evaluations[c] = chain.Evaluations;
            acceptance[c] = chain.AcceptanceRate;
            scales[c] = chain.Scale;
            chain.WriteCovariance(covariances.GetMatrix(c));
        });

        return new RandomWalkMetropolisResult(draws, logDensities, evaluations, settings.Seed, acceptance, scales, covariances);
    }

    private sealed class Chain<TDensity>(RandomWalkMetropolis sampler, TDensity density, int d, int index)
        where TDensity : ILogDensity
    {
        private readonly Philox4x32Random random = RandomStreams.Create(sampler.settings.Seed, StreamPurpose.Transition, index);
        private readonly CovarianceShape shape = new(d, dense: true);
        private readonly double[] current = new double[d];
        private readonly double[] proposal = new double[d];
        private readonly double[] normal = new double[d];
        private readonly double[] step = new double[d];
        private double currentLogDensity;
        private bool lastAccepted;
        private long accepted;

        public long Evaluations { get; private set; }

        public double AcceptanceRate => (double)accepted / sampler.settings.SamplingIterations;

        public double Scale { get; private set; }

        public void WriteCovariance(MatrixSpan<double> destination)
        {
            var dense = new double[d * d];
            shape.WriteDense(dense);
            double s2 = Scale * Scale;
            for (int j = 0; j < d; j++)
            {
                for (int i = 0; i < d; i++)
                {
                    destination[i, j] = s2 * dense[j * d + i];
                }
            }
        }

        public void Run(ReadOnlySpan<double> start, MatrixSpan<double> draws, VectorSpan<double> logDensities, CancellationToken cancellationToken)
        {
            SamplerSettings settings = sampler.settings;
            if (sampler.initialCovariance is { } c0 && !shape.TrySet(c0.AsSpan()))
            {
                throw new ArgumentException("The initial covariance must be symmetric positive definite.", "options");
            }

            start.CopyTo(current);
            currentLogDensity = Evaluate(current);
            InitialPoints.RequireValid(current, currentLogDensity, index);
            double optimal = OptimalScaleNumerator / Math.Sqrt(d);
            Scale = sampler.proposalScale ?? optimal;
            double target = sampler.targetAcceptance ?? (d == 1 ? 0.44 : 0.234);
            var schedule = new WarmupSchedule(settings.Warmup, sampler.windows);
            var adaptation = new DualAveraging(target, centerFactor: 1, gamma: 1);
            adaptation.Restart(Scale);
            var estimator = new CovarianceEstimator(d, dense: true);
            var estimate = new double[d * d];
            for (int t = 0; t < settings.Warmup; t++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                double acceptance = Transition();
                Scale = adaptation.Update(acceptance);
                if (sampler.adaptCovariance && schedule.InWindow(t))
                {
                    estimator.Add(current);
                }

                if (sampler.adaptCovariance && schedule.EndsWindow(t))
                {
                    estimator.WriteRegularized(estimate);
                    estimator.Reset();
                    if (shape.TrySet(estimate))
                    {
                        Scale = optimal;
                    }

                    adaptation.Restart(Scale);
                }
            }

            if (settings.Warmup > 0 && double.IsFinite(adaptation.Averaged))
            {
                Scale = adaptation.Averaged;
            }

            int kept = 0;
            for (long t = 0; t < settings.SamplingIterations; t++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Transition();
                if (lastAccepted)
                {
                    accepted++;
                }

                if ((t + 1) % settings.Thin == 0)
                {
                    current.CopyTo(draws.Column(kept).Data);
                    logDensities[kept] = currentLogDensity;
                    kept++;
                }
            }
        }

        private double Transition()
        {
            Gaussian.Fill(random, normal);
            shape.MultiplyFactor(normal, step);
            for (int i = 0; i < d; i++)
            {
                proposal[i] = current[i] + Scale * step[i];
            }

            double proposed = Evaluate(proposal);
            double logRatio = proposed - currentLogDensity;
            double logU = Math.Log(random.NextDoubleOpen());
            lastAccepted = logU < logRatio;
            if (lastAccepted)
            {
                proposal.CopyTo(current, 0);
                currentLogDensity = proposed;
            }

            return logRatio >= 0 ? 1 : Math.Exp(logRatio);
        }

        private double Evaluate(double[] point)
        {
            Evaluations++;
            return LogDensities.Sanitize(density.LogDensity(point));
        }
    }
}

/// <summary>
/// Represents the draws and proposal diagnostics produced by <see cref="RandomWalkMetropolis"/>.
/// <para><see cref="RandomWalkMetropolis"/> のサンプリング結果と提案分布の診断情報を表します。</para>
/// </summary>
public sealed class RandomWalkMetropolisResult : SamplingResult
{
    private readonly double[] acceptanceRates;
    private readonly double[] proposalScales;

    internal RandomWalkMetropolisResult(Tensor<double> draws, Matrix<double> logDensities, long[] evaluations, ulong seed,
        double[] acceptanceRates, double[] proposalScales, Tensor<double> proposalCovariances)
        : base(draws, logDensities, evaluations, seed)
    {
        this.acceptanceRates = acceptanceRates;
        this.proposalScales = proposalScales;
        ProposalCovariances = proposalCovariances;
    }

    /// <summary>
    /// Gets the post-warmup proposal acceptance rate for each chain.
    /// <para>チェーンごとのウォームアップ後の提案受理率を取得します。</para>
    /// </summary>
    public IReadOnlyList<double> AcceptanceRates => acceptanceRates;

    /// <summary>
    /// Gets the fixed post-warmup proposal scale factor <c>s</c> for each chain.
    /// <para>チェーンごとのウォームアップ後に固定された提案スケール係数 <c>s</c> を取得します。</para>
    /// </summary>
    public IReadOnlyList<double> ProposalScales => proposalScales;

    /// <summary>
    /// Gets the post-warmup proposal covariance <c>s²C</c> for each chain with shape <c>[dimension, dimension, chain]</c>.
    /// <para>チェーンごとのウォームアップ後の提案共分散 <c>s²C</c>（形状 <c>[次元, 次元, チェーン]</c>）を取得します。</para>
    /// </summary>
    /// <remarks>
    /// <para>Individual chain slices can be reused as <see cref="RandomWalkMetropolisOptions.InitialCovariance"/> (with <see cref="RandomWalkMetropolisOptions.ProposalScale"/> set to 1) in subsequent runs.</para>
    /// <para>各チェーンの行列は、再実行時の <see cref="RandomWalkMetropolisOptions.InitialCovariance"/>（<see cref="RandomWalkMetropolisOptions.ProposalScale"/> = 1）として再利用できます。</para>
    /// </remarks>
    public Tensor<double> ProposalCovariances { get; }
}
