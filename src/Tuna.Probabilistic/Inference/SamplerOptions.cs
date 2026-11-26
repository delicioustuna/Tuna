namespace Tuna.Probabilistic.Inference;

/// <summary>
/// Provides common configuration options for MCMC samplers.
/// <para>MCMC サンプラーに共通の設定オプションを提供します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Option values are validated by each sampler's constructor.
/// Each chain runs <see cref="Warmup"/> adaptation and burn-in transitions (discarded), followed by <see cref="Draws"/> × <see cref="Thin"/> sampling transitions retaining every <see cref="Thin"/>-th state.
/// </para>
/// <para>
/// 設定値は各サンプラーのコンストラクターで検証されます。
/// 各チェーンは <see cref="Warmup"/> 回のウォームアップ遷移（適応およびバーンイン、標本には含めない）の後、<see cref="Draws"/> × <see cref="Thin"/> 回の遷移を実行し、<see cref="Thin"/> 回ごとに 1 標本を保持します。
/// </para>
/// </remarks>
public abstract class SamplerOptions
{
    /// <summary>
    /// Gets or initializes the number of warmup transitions per chain (0 or greater; default is 1000).
    /// <para>チェーンごとのウォームアップ遷移回数を取得または設定します（0 以上、既定値 1000）。</para>
    /// </summary>
    /// <remarks>
    /// <para>During warmup, sampler parameters such as step size and proposal covariance are adapted and then frozen. Warmup draws are excluded from the result.</para>
    /// <para>ウォームアップ期間中にステップ幅や提案共分散などを適応させ、終了後に固定します。この期間の標本は結果に含まれません。</para>
    /// </remarks>
    public int Warmup { get; init; } = 1000;

    /// <summary>
    /// Gets or initializes the number of retained draws per chain (1 or greater; default is 1000).
    /// <para>チェーンごとに保持する標本数を取得または設定します（1 以上、既定値 1000）。</para>
    /// </summary>
    public int Draws { get; init; } = 1000;

    /// <summary>
    /// Gets or initializes the thinning interval (1 or greater; default is 1). After warmup, one draw is kept every <see cref="Thin"/> transitions.
    /// <para>間引き間隔を取得または設定します（1 以上、既定値 1）。ウォームアップ後、<see cref="Thin"/> 回の遷移ごとに 1 標本を保持します。</para>
    /// </summary>
    public int Thin { get; init; } = 1;

    /// <summary>
    /// Gets or initializes the random number generator seed. If <see langword="null"/> (default), a seed is obtained from the OS and exposed via <see cref="SamplingResult.Seed"/>.
    /// <para>乱数シードを取得または設定します。<see langword="null"/>（既定）の場合は OS からシードを取得し、<see cref="SamplingResult.Seed"/> で参照できます。</para>
    /// </summary>
    /// <remarks>
    /// <para>Given the same seed, options, initial points, and log-density, draws are bitwise identical regardless of <see cref="MaxDegreeOfParallelism"/>.</para>
    /// <para>同一のシード・設定・初期点・対数密度であれば、<see cref="MaxDegreeOfParallelism"/> によらずビット単位で同一の標本が得られます。</para>
    /// </remarks>
    public ulong? Seed { get; init; }

    /// <summary>
    /// Gets or initializes the maximum number of chains (or walkers in <see cref="EnsembleSampler"/>) evaluated concurrently (1 or greater, or -1 for unbounded; default is 1).
    /// <para>同時に実行するチェーン数（<see cref="EnsembleSampler"/> では同時評価するウォーカー数）の上限を取得または設定します（1 以上、または無制限の -1。既定値 1）。</para>
    /// </summary>
    /// <remarks>
    /// <para>When set to 2 or greater (or -1), the target log-density must be safe for concurrent calls from multiple threads.</para>
    /// <para>2 以上（または -1）を指定する場合、対数密度は複数スレッドからの同時呼び出しに対してスレッドセーフである必要があります。</para>
    /// </remarks>
    public int MaxDegreeOfParallelism { get; init; } = 1;
}

/// <summary>
/// Provides configuration options for samplers that run multiple independent chains.
/// <para>独立した複数のチェーンを実行するサンプラーの設定オプションを提供します。</para>
/// </summary>
public abstract class MultiChainSamplerOptions : SamplerOptions
{
    /// <summary>
    /// Gets or initializes the number of independent chains (1 or greater; default is 4). Four or more chains from dispersed initial points are recommended for convergence diagnostics such as <see cref="ConvergenceDiagnostics.RHat"/>.
    /// <para>独立したチェーンの数を取得または設定します（1 以上、既定値 4）。<see cref="ConvergenceDiagnostics.RHat"/> などの収束診断には、互いに離れた初期点から 4 本以上実行することを推奨します。</para>
    /// </summary>
    public int Chains { get; init; } = 4;
}

/// <summary>
/// Configures the windowed warmup adaptation schedule used by <see cref="NoUTurnSampler"/>, <see cref="HamiltonianMonteCarlo"/>, <see cref="RandomWalkMetropolis"/>, <see cref="SliceSampler"/>, and <see cref="ParallelTempering"/>.
/// <para><see cref="NoUTurnSampler"/>、<see cref="HamiltonianMonteCarlo"/>、<see cref="RandomWalkMetropolis"/>、<see cref="SliceSampler"/>、および <see cref="ParallelTempering"/> で使用するウォームアップ適応ウィンドウの区切りを設定します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Warmup is partitioned into an initial fast buffer, exponentially doubling slow adaptation windows, and a terminal fast buffer (following Stan's adaptation schedule, with the same defaults 75, 25, and 50).
/// Only step size or scale is adapted during the initial and terminal buffers. At the end of each window, the metric, proposal covariance, or slice directions are updated from the window's draws and step-size adaptation is restarted. If the next doubled window would not fit before the terminal buffer, the current window is extended to meet it.
/// </para>
/// <para>
/// If the sum of the three settings exceeds the warmup length, the initial buffer defaults to 15% of warmup, the terminal buffer to 10%, and the remainder forms the first window. When warmup is fewer than 20 transitions, metric adaptation is skipped and only step size or scale is adapted.
/// </para>
/// <para>
/// ウォームアップ期間を「初期バッファ → 倍々に拡張するウィンドウ → 終端バッファ」に分割します（Stan の自動適応スケジュールと同一の構成・既定値 75、25、50）。
/// 初期バッファと終端バッファではステップ幅（またはスケール）のみを適応させます。各ウィンドウの終端でそのウィンドウ内の標本共分散から計量・提案共分散・スライス方向を更新し、ステップ幅の適応を再開します。次のウィンドウが終端バッファの手前に収まらない場合は、現在のウィンドウを終端バッファ直前まで延長します。
/// </para>
/// <para>
/// 3 つの合計がウォームアップ回数を超える場合、初期バッファを 15%、終端バッファを 10%、残りを最初のウィンドウとして配分します。ウォームアップ回数が 20 未満の場合、計量などの適応は行わずステップ幅（またはスケール）のみを適応させます。
/// </para>
/// </remarks>
public sealed class WarmupWindowOptions
{
    /// <summary>
    /// Gets or initializes the length of the initial buffer in transitions (0 or greater; default is 75).
    /// <para>初期バッファの遷移回数を取得または設定します（0 以上、既定値 75）。</para>
    /// </summary>
    public int InitialBuffer { get; init; } = 75;

    /// <summary>
    /// Gets or initializes the length of the first adaptation window in transitions (1 or greater; default is 25). Subsequent windows double in length.
    /// <para>最初の適応ウィンドウの遷移回数を取得または設定します（1 以上、既定値 25）。以降のウィンドウは倍々に長くなります。</para>
    /// </summary>
    public int BaseWindow { get; init; } = 25;

    /// <summary>
    /// Gets or initializes the length of the terminal buffer in transitions (0 or greater; default is 50).
    /// <para>終端バッファの遷移回数を取得または設定します（0 以上、既定値 50）。</para>
    /// </summary>
    public int TerminalBuffer { get; init; } = 50;
}
