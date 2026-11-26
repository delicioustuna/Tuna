using Tuna.LinearAlgebra;
using Tuna.RandomSource;

namespace Tuna.Probabilistic.Inference;

/// <summary>
/// Configures <see cref="EnsembleSampler"/>.
/// <para><see cref="EnsembleSampler"/> の設定オプションを表します。</para>
/// </summary>
public sealed class EnsembleSamplerOptions : SamplerOptions
{
    /// <summary>
    /// Gets or initializes the number of walkers <c>K</c> (2 or greater; defaults to <c>max(2d, 8)</c> when <see langword="null"/>, where <c>d</c> is the dimension).
    /// <para>ウォーカー数 <c>K</c> を取得または設定します（2 以上。<see langword="null"/> の既定値は <c>max(2d, 8)</c>、<c>d</c> は次元）。</para>
    /// </summary>
    /// <remarks>
    /// <para>Must be at least <c>d + 1</c> at sampling time so walkers are not confined to a lower-dimensional affine subspace; <c>2d</c> or more is recommended.</para>
    /// <para>サンプリング時には次元 <c>d + 1</c> 以上である必要があります（初期点が張るアフィン部分空間から出られないため）。<c>2d</c> 以上を推奨します。</para>
    /// </remarks>
    public int? Walkers { get; init; }

    /// <summary>
    /// Gets or initializes the stretch-move scale parameter <c>a</c> (finite and greater than 1; default is 2 per Goodman &amp; Weare 2010). Larger values propose longer moves and lower acceptance rates.
    /// <para>ストレッチ移動のスケールパラメータ <c>a</c> を取得または設定します（1 より大きい有限値、既定値 2、Goodman &amp; Weare 2010）。大きいほど移動幅が広がり受理率は下がります。</para>
    /// </summary>
    public double StretchScale { get; init; } = 2;

    /// <summary>
    /// Gets or initializes the Gaussian standard deviation used to scatter walkers around a single initial point (positive finite value; default is <c>10⁻³</c>). Points with non-finite log-density are redrawn.
    /// <para>単一の初期点から開始する際にウォーカーを周囲に配置する正規分布の標準偏差を取得または設定します（正の有限値、既定値 <c>10⁻³</c>）。対数密度が非有限の点は再生成されます。</para>
    /// </summary>
    public double InitialSpread { get; init; } = 1e-3;
}

/// <summary>
/// Samples from a log-density using the affine-invariant stretch-move ensemble sampler (Goodman &amp; Weare 2010).
/// <para>ストレッチ移動によるアフィン不変アンサンブルサンプラー（Goodman &amp; Weare 2010）により、対数密度からサンプリングします。</para>
/// </summary>
/// <remarks>
/// <para>
/// Gradients are not required. The third axis of <see cref="SamplingResult.Draws"/> indexes walkers, and each iteration updates all walkers once.
/// Because walkers interact, <see cref="ConvergenceDiagnostics.RHat"/> serves only as a heuristic. For strongly multimodal targets, consider <see cref="ParallelTempering"/>.
/// </para>
/// <para>
/// 勾配は不要です。<see cref="SamplingResult.Draws"/> の第 3 軸がウォーカーに対応し、1 反復で全ウォーカーを 1 回ずつ更新します。
/// ウォーカー同士が相互作用するため、<see cref="ConvergenceDiagnostics.RHat"/> は参考値として扱ってください。多峰性の強い分布には <see cref="ParallelTempering"/> も利用できます。
/// </para>
/// </remarks>
public sealed class EnsembleSampler
{
    private const int MaxInitializationAttempts = 1000;

    private readonly SamplerSettings settings;
    private readonly int? walkers;
    private readonly double stretchScale;
    private readonly double initialSpread;

    /// <summary>
    /// Initializes a new instance of <see cref="EnsembleSampler"/> with the specified options.
    /// <para>指定した設定で <see cref="EnsembleSampler"/> の新しいインスタンスを初期化します。</para>
    /// </summary>
    /// <param name="options">The sampler options, or <see langword="null"/> for defaults. / サンプラー設定（<see langword="null"/> の場合は既定値）。</param>
    /// <exception cref="ArgumentOutOfRangeException">An option value is out of its valid range. / 設定値が有効範囲外の場合。</exception>
    public EnsembleSampler(EnsembleSamplerOptions? options = null)
    {
        options ??= new EnsembleSamplerOptions();
        settings = SamplerSettings.Validate(options);
        if (options.Walkers is { } k)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(k, 2, nameof(options.Walkers));
        }

        if (!(options.StretchScale > 1 && double.IsFinite(options.StretchScale)))
        {
            throw new ArgumentOutOfRangeException(nameof(options.StretchScale), "The stretch scale must be greater than 1 and finite.");
        }

        if (!(options.InitialSpread > 0 && double.IsFinite(options.InitialSpread)))
        {
            throw new ArgumentOutOfRangeException(nameof(options.InitialSpread), "The initial spread must be positive and finite.");
        }

        walkers = options.Walkers;
        stretchScale = options.StretchScale;
        initialSpread = options.InitialSpread;
    }

    /// <summary>
    /// Gets the configured number of walkers, or <see langword="null"/> if determined automatically from the dimension.
    /// <para>設定されたウォーカー数を取得します（次元から自動決定する場合は <see langword="null"/>）。</para>
    /// </summary>
    public int? Walkers => walkers;

    /// <summary>
    /// Gets the random seed used for sampling (configured value or obtained from the OS).
    /// <para>サンプリングに使用する乱数シード（指定値、または省略時に OS から取得した値）を取得します。</para>
    /// </summary>
    public ulong Seed => settings.Seed;

    /// <summary>
    /// Generates posterior draws by scattering walkers around the specified initial point.
    /// <para>指定した初期点の周囲にウォーカーを配置して標本を生成します。</para>
    /// </summary>
    /// <param name="logDensity">The delegate evaluating the log-density. / 対数密度を評価するデリゲート。</param>
    /// <param name="initialPoint">The center point around which walkers are scattered with standard deviation <see cref="EnsembleSamplerOptions.InitialSpread"/>. / ウォーカーを標準偏差 <see cref="EnsembleSamplerOptions.InitialSpread"/> で配置する中心点。</param>
    /// <param name="cancellationToken">A token checked at each sweep to cancel sampling. / 掃引ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws and acceptance diagnostics. / ウォームアップ後の標本と受理率診断情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="logDensity"/> is <see langword="null"/>. / <paramref name="logDensity"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoint"/> is empty or non-finite, the walker count is less than <c>dimension + 1</c>, or scattered points repeatedly yield non-finite log-density. / <paramref name="initialPoint"/> が空または非有限、ウォーカー数が <c>次元 + 1</c> 未満、または配置した点の対数密度が繰り返し非有限となる場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public EnsembleSamplingResult Sample(Func<ReadOnlyVectorSpan<double>, double> logDensity, ReadOnlySpan<double> initialPoint,
        CancellationToken cancellationToken = default) =>
        Sample(LogDensities.FromDelegate(logDensity), initialPoint, cancellationToken);

    /// <summary>
    /// Generates posterior draws starting each walker from its corresponding column of <paramref name="initialPoints"/>.
    /// <para>ウォーカーごとに指定した初期点から標本を生成します。</para>
    /// </summary>
    /// <param name="logDensity">The delegate evaluating the log-density. / 対数密度を評価するデリゲート。</param>
    /// <param name="initialPoints">The matrix of initial points with shape <c>[dimension × walkers]</c>. When <see cref="EnsembleSamplerOptions.Walkers"/> is <see langword="null"/>, the column count sets the walker count. Points must span the full space so walkers are not confined to a lower-dimensional affine subspace. / 各ウォーカーの初期点を列に並べた行列（形状 <c>[次元 × ウォーカー数]</c>）。<see cref="EnsembleSamplerOptions.Walkers"/> が <see langword="null"/> の場合は列数がウォーカー数になります。</param>
    /// <param name="cancellationToken">A token checked at each sweep to cancel sampling. / 掃引ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws and acceptance diagnostics. / ウォームアップ後の標本と受理率診断情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="logDensity"/> is <see langword="null"/>. / <paramref name="logDensity"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoints"/> is empty, has a column count conflicting with <see cref="Walkers"/> or less than <c>dimension + 1</c>, contains non-finite values, or has non-finite log-density. / <paramref name="initialPoints"/> が空、列数が設定ウォーカー数と不一致または <c>次元 + 1</c> 未満、非有限値を含む、または初期点の対数密度が非有限の場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public EnsembleSamplingResult Sample(Func<ReadOnlyVectorSpan<double>, double> logDensity, ReadOnlyMatrixSpan<double> initialPoints,
        CancellationToken cancellationToken = default) =>
        Sample(LogDensities.FromDelegate(logDensity), initialPoints, cancellationToken);

    /// <summary>
    /// Generates posterior draws by scattering walkers around the specified initial point.
    /// <para>指定した初期点の周囲にウォーカーを配置して標本を生成します。</para>
    /// </summary>
    /// <typeparam name="TDensity">The log-density type. / 対数密度の型。</typeparam>
    /// <param name="density">The target log-density. / 対象分布の対数密度。</param>
    /// <param name="initialPoint">The center point around which walkers are scattered with standard deviation <see cref="EnsembleSamplerOptions.InitialSpread"/>. / ウォーカーを標準偏差 <see cref="EnsembleSamplerOptions.InitialSpread"/> で配置する中心点。</param>
    /// <param name="cancellationToken">A token checked at each sweep to cancel sampling. / 掃引ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws and acceptance diagnostics. / ウォームアップ後の標本と受理率診断情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="density"/> is <see langword="null"/>. / <paramref name="density"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoint"/> is empty or non-finite, the walker count is less than <c>dimension + 1</c>, or scattered points repeatedly yield non-finite log-density. / <paramref name="initialPoint"/> が空または非有限、ウォーカー数が <c>次元 + 1</c> 未満、または配置した点の対数密度が繰り返し非有限となる場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public EnsembleSamplingResult Sample<TDensity>(TDensity density, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)
        where TDensity : ILogDensity
    {
        ArgumentNullException.ThrowIfNull(density);
        if (initialPoint.IsEmpty)
        {
            throw new ArgumentException("The initial point must not be empty.", nameof(initialPoint));
        }

        int d = initialPoint.Length;
        int k = RequireWalkers(walkers ?? Math.Max(2 * d, 8), d);
        var starts = new double[d * k];
        var startValues = new double[k];
        var evaluations = new long[k];
        double[] center = initialPoint.ToArray();
        foreach (double v in center)
        {
            if (!double.IsFinite(v))
            {
                throw new ArgumentException("The initial point must be finite.", nameof(initialPoint));
            }
        }

        ChainRunner.Run(k, settings.MaxDegreeOfParallelism, cancellationToken, w =>
        {
            Philox4x32Random random = RandomStreams.Create(settings.Seed, StreamPurpose.Initialization, w);
            Span<double> point = starts.AsSpan(w * d, d);
            for (int attempt = 0; attempt < MaxInitializationAttempts; attempt++)
            {
                Gaussian.Fill(random, point);
                for (int i = 0; i < d; i++)
                {
                    point[i] = center[i] + initialSpread * point[i];
                }

                evaluations[w]++;
                double value = LogDensities.Sanitize(density.LogDensity(point));
                if (double.IsFinite(value))
                {
                    startValues[w] = value;
                    return;
                }
            }

            throw new ArgumentException(
                $"The log density was not finite at {MaxInitializationAttempts} points scattered around the initial point (walker {w}).", nameof(initialPoint));
        });

        return Run(density, d, k, starts, startValues, evaluations, cancellationToken);
    }

    /// <summary>
    /// Generates posterior draws starting each walker from its corresponding column of <paramref name="initialPoints"/>.
    /// <para>ウォーカーごとに指定した初期点から標本を生成します。</para>
    /// </summary>
    /// <typeparam name="TDensity">The log-density type. / 対数密度の型。</typeparam>
    /// <param name="density">The target log-density. / 対象分布の対数密度。</param>
    /// <param name="initialPoints">The matrix of initial points with shape <c>[dimension × walkers]</c>. When <see cref="EnsembleSamplerOptions.Walkers"/> is <see langword="null"/>, the column count sets the walker count. / 各ウォーカーの初期点を列に並べた行列（形状 <c>[次元 × ウォーカー数]</c>）。<see cref="EnsembleSamplerOptions.Walkers"/> が <see langword="null"/> の場合は列数がウォーカー数になります。</param>
    /// <param name="cancellationToken">A token checked at each sweep to cancel sampling. / 掃引ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws and acceptance diagnostics. / ウォームアップ後の標本と受理率診断情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="density"/> is <see langword="null"/>. / <paramref name="density"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoints"/> is empty, has a column count conflicting with <see cref="Walkers"/> or less than <c>dimension + 1</c>, contains non-finite values, or has non-finite log-density. / <paramref name="initialPoints"/> が空、列数が設定ウォーカー数と不一致または <c>次元 + 1</c> 未満、非有限値を含む、または初期点の対数密度が非有限の場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public EnsembleSamplingResult Sample<TDensity>(TDensity density, ReadOnlyMatrixSpan<double> initialPoints, CancellationToken cancellationToken = default)
        where TDensity : ILogDensity
    {
        ArgumentNullException.ThrowIfNull(density);
        int k = walkers ?? initialPoints.Columns;
        double[] starts = InitialPoints.Columns(initialPoints, k, "walkers");
        int d = initialPoints.Rows;
        RequireWalkers(k, d);
        var startValues = new double[k];
        var evaluations = new long[k];
        for (int w = 0; w < k; w++)
        {
            ReadOnlySpan<double> point = starts.AsSpan(w * d, d);
            evaluations[w]++;
            startValues[w] = LogDensities.Sanitize(density.LogDensity(point));
            InitialPoints.RequireValid(point, startValues[w], w);
        }

        return Run(density, d, k, starts, startValues, evaluations, cancellationToken);
    }

    private static int RequireWalkers(int k, int d)
    {
        if (k < d + 1)
        {
            throw new ArgumentException($"The number of walkers ({k}) must be at least the dimension plus one ({d + 1}).", "options");
        }

        return k;
    }

    private EnsembleSamplingResult Run<TDensity>(TDensity density, int d, int k, double[] positions, double[] values, long[] evaluations,
        CancellationToken cancellationToken)
        where TDensity : ILogDensity
    {
        settings.RequireStorage(d, k);
        var draws = new Tensor<double>(d, settings.Draws, k);
        var logDensities = new Matrix<double>(settings.Draws, k);
        var randoms = new Philox4x32Random[k];
        for (int w = 0; w < k; w++)
        {
            randoms[w] = RandomStreams.Create(settings.Seed, StreamPurpose.Transition, w);
        }

        var proposals = new double[d * k];
        var accepted = new long[k];
        bool sampling = false;
        int half = k / 2;
        double a = stretchScale;

        void Move(int w, int partnerStart, int partnerCount)
        {
            Philox4x32Random random = randoms[w];
            int j = partnerStart + random.NextInt32(partnerCount);
            double z = (a - 1) * random.NextDouble() + 1;
            z = z * z / a;
            Span<double> x = positions.AsSpan(w * d, d);
            ReadOnlySpan<double> partner = positions.AsSpan(j * d, d);
            Span<double> y = proposals.AsSpan(w * d, d);
            for (int i = 0; i < d; i++)
            {
                y[i] = partner[i] + z * (x[i] - partner[i]);
            }

            evaluations[w]++;
            double proposed = LogDensities.Sanitize(density.LogDensity(y));
            double logRatio = (d - 1) * Math.Log(z) + proposed - values[w];
            if (Math.Log(random.NextDoubleOpen()) < logRatio)
            {
                y.CopyTo(x);
                values[w] = proposed;
                if (sampling)
                {
                    accepted[w]++;
                }
            }
        }

        long total = settings.Warmup + settings.SamplingIterations;
        int kept = 0;
        for (long t = 0; t < total; t++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            sampling = t >= settings.Warmup;
            ChainRunner.Run(half, settings.MaxDegreeOfParallelism, cancellationToken, w => Move(w, half, k - half));
            ChainRunner.Run(k - half, settings.MaxDegreeOfParallelism, cancellationToken, w => Move(half + w, 0, half));
            if (sampling && (t - settings.Warmup + 1) % settings.Thin == 0)
            {
                for (int w = 0; w < k; w++)
                {
                    positions.AsSpan(w * d, d).CopyTo(draws.GetVector(kept, w).Data);
                    logDensities[kept, w] = values[w];
                }

                kept++;
            }
        }

        var rates = new double[k];
        for (int w = 0; w < k; w++)
        {
            rates[w] = (double)accepted[w] / settings.SamplingIterations;
        }

        return new EnsembleSamplingResult(draws, logDensities, evaluations, settings.Seed, rates);
    }
}

/// <summary>
/// Represents the draws and acceptance diagnostics produced by <see cref="EnsembleSampler"/> (<see cref="SamplingResult.ChainCount"/> equals the number of walkers).
/// <para><see cref="EnsembleSampler"/> のサンプリング結果と受理率診断情報を表します（<see cref="SamplingResult.ChainCount"/> はウォーカー数に対応します）。</para>
/// </summary>
public sealed class EnsembleSamplingResult : SamplingResult
{
    private readonly double[] acceptanceRates;

    internal EnsembleSamplingResult(Tensor<double> draws, Matrix<double> logDensities, long[] evaluations, ulong seed, double[] acceptanceRates)
        : base(draws, logDensities, evaluations, seed)
    {
        this.acceptanceRates = acceptanceRates;
    }

    /// <summary>
    /// Gets the post-warmup move acceptance rate for each walker (roughly 0.2–0.5 indicates an appropriate <see cref="EnsembleSamplerOptions.StretchScale"/>).
    /// <para>ウォーカーごとのウォームアップ後の移動受理率を取得します（おおむね 0.2〜0.5 であれば <see cref="EnsembleSamplerOptions.StretchScale"/> は妥当です）。</para>
    /// </summary>
    public IReadOnlyList<double> AcceptanceRates => acceptanceRates;

    /// <summary>
    /// Gets the mean acceptance rate across all walkers.
    /// <para>全ウォーカーの平均受理率を取得します。</para>
    /// </summary>
    public double MeanAcceptanceRate => acceptanceRates.Average();
}
