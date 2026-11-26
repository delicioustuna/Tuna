using Tuna.LinearAlgebra;
using Tuna.RandomSource;

namespace Tuna.Probabilistic.Inference;

/// <summary>
/// Configures <see cref="ParallelTempering"/>.
/// <para><see cref="ParallelTempering"/> の設定オプションを表します。</para>
/// </summary>
public sealed class ParallelTemperingOptions : MultiChainSamplerOptions
{
    /// <summary>
    /// Gets or initializes the number of temperatures <c>K</c> in the ladder from inverse temperature 1 down to <see cref="MinimumInverseTemperature"/> (2 or greater; default is 8).
    /// <para>逆温度 1（目標分布）から <see cref="MinimumInverseTemperature"/> までの温度ラダーの段数 <c>K</c> を取得または設定します（2 以上、既定値 8）。</para>
    /// </summary>
    /// <remarks>
    /// <para>Increasing <see cref="Temperatures"/> much beyond twice <see cref="ParallelTemperingResult.CommunicationBarriers"/> (<c>2Λ</c>) yields diminishing gains in round-trip rate (Syed et al. 2022).</para>
    /// <para>結果の <see cref="ParallelTemperingResult.CommunicationBarriers"/>（大域的通信障壁 <c>Λ</c>）の約 2 倍を大きく超えて増やしても、往復頻度の改善は限定的です（Syed et al. 2022）。</para>
    /// </remarks>
    public int Temperatures { get; init; } = 8;

    /// <summary>
    /// Gets or initializes the inverse temperature <c>β_min</c> of the hottest replica.
    /// <para>最も高温のレプリカの逆温度 <c>β_min</c> を取得または設定します。</para>
    /// </summary>
    /// <value>A value in <c>[0, 1)</c> (must be strictly positive when tempering the whole density). Defaults to 0.01 for whole-density tempering and 0 for likelihood-only tempering when <see langword="null"/>. / <c>[0, 1)</c> の値（密度全体を焼きなます場合は正）。<see langword="null"/>（既定）の場合、密度全体では 0.01、尤度のみでは 0 です。</value>
    public double? MinimumInverseTemperature { get; init; }

    /// <summary>
    /// Gets or initializes a value indicating whether intermediate inverse temperatures are adapted during warmup (default is <see langword="true"/>).
    /// <para>ウォームアップ期間中に中間の逆温度を適応させるかどうかを取得または設定します（既定値 <see langword="true"/>）。</para>
    /// </summary>
    /// <value><see langword="true"/> to adjust intermediate inverse temperatures at each window end to equalize adjacent swap rejection rates (Syed et al. 2022); endpoint inverse temperatures remain fixed. / <see langword="true"/> の場合、各ウィンドウ終端で隣接温度間の交換棄却率が揃うように中間の逆温度を調整します（Syed et al. 2022）。両端の逆温度は固定されます。</value>
    public bool AdaptLadder { get; init; } = true;

    /// <summary>
    /// Gets or initializes the number of local random-walk Metropolis transitions per temperature between swap attempts (1 or greater; default is 1).
    /// <para>交換試行の間に各温度で実行するランダムウォーク・メトロポリス遷移の回数を取得または設定します（1 以上、既定値 1）。</para>
    /// </summary>
    public int LocalSteps { get; init; } = 1;

    /// <summary>
    /// Gets or initializes the target acceptance rate for local random-walk Metropolis updates at each temperature (same convention as <see cref="RandomWalkMetropolisOptions.TargetAcceptance"/>; defaults to 0.44 in 1D and 0.234 in 2D or higher).
    /// <para>各温度のランダムウォーク・メトロポリス法における目標受理率を取得または設定します（<see cref="RandomWalkMetropolisOptions.TargetAcceptance"/> と同じ仕様。<see langword="null"/> の既定値は 1 次元で 0.44、2 次元以上で 0.234）。</para>
    /// </summary>
    public double? TargetAcceptance { get; init; }

    /// <summary>
    /// Gets or initializes a value indicating whether each temperature's proposal covariance shape is adapted at the end of each warmup window (default is <see langword="true"/>).
    /// <para>ウォームアップの各ウィンドウ終端で、各温度の提案共分散形状を標本共分散から更新するかどうかを取得または設定します（既定値 <see langword="true"/>）。</para>
    /// </summary>
    public bool AdaptCovariance { get; init; } = true;

    /// <summary>
    /// Gets or initializes the warmup adaptation window schedule.
    /// <para>ウォームアップ期間の適応ウィンドウ設定を取得または設定します。</para>
    /// </summary>
    public WarmupWindowOptions Windows { get; init; } = new();
}

/// <summary>
/// Samples from a log-density using non-reversible parallel tempering (replica exchange) across a ladder of inverse temperatures.
/// <para>逆温度ラダー上で非可逆レプリカ交換を行うパラレルテンパリングサンプラーです。</para>
/// </summary>
/// <remarks>
/// <para>
/// Supports both whole-density tempering <c>π(θ)^β</c> and likelihood-only tempering <c>p(θ)·L(θ)^β</c>, returning draws from the target inverse temperature <c>β = 1</c>.
/// Points with non-finite log-density (or non-finite prior/likelihood) are treated as zero density at all temperatures.
/// <see cref="SamplingResult.Evaluations"/> counts one evaluation per point (combining prior and likelihood in likelihood tempering, and skipping likelihood evaluation when the prior is <see cref="double.NegativeInfinity"/>).
/// </para>
/// <para>
/// 密度全体を <c>π(θ)^β</c> とする方法と、尤度のみを <c>p(θ)·L(θ)^β</c> とする方法の両方に対応し、逆温度 <c>β = 1</c> の標本を返します。
/// 対数密度（尤度のみの場合は事前分布または尤度）が非有限の点はすべての温度で密度 0 として扱われます。
/// <see cref="SamplingResult.Evaluations"/> は 1 点の評価を 1 回と数えます（尤度のみの場合は事前分布と尤度をまとめて 1 回とし、事前分布が <see cref="double.NegativeInfinity"/> の点では尤度を評価しません）。
/// </para>
/// </remarks>
public sealed class ParallelTempering
{
    private const double OptimalScaleNumerator = 2.38;
    private const double DefaultMinimumPower = 0.01;
    // 棄却率がほぼ 0 の区間でも逆温度の狭義単調性を保つための下限
    private const double MinimumRejection = 1e-3;

    private readonly SamplerSettings settings;
    private readonly int chains;
    private readonly int temperatures;
    private readonly double? minimumInverseTemperature;
    private readonly bool adaptLadder;
    private readonly int localSteps;
    private readonly double? targetAcceptance;
    private readonly bool adaptCovariance;
    private readonly WarmupWindowOptions windows;

    /// <summary>
    /// Initializes a new instance of <see cref="ParallelTempering"/> with the specified options.
    /// <para>指定した設定で <see cref="ParallelTempering"/> の新しいインスタンスを初期化します。</para>
    /// </summary>
    /// <param name="options">The sampler options, or <see langword="null"/> for defaults. / サンプラー設定（<see langword="null"/> の場合は既定値）。</param>
    /// <exception cref="ArgumentNullException"><see cref="ParallelTemperingOptions.Windows"/> is <see langword="null"/>. / <see cref="ParallelTemperingOptions.Windows"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option value is out of its valid range. / 設定値が有効範囲外の場合。</exception>
    public ParallelTempering(ParallelTemperingOptions? options = null)
    {
        options ??= new ParallelTemperingOptions();
        settings = SamplerSettings.Validate(options);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.Chains, 1, nameof(options.Chains));
        ArgumentOutOfRangeException.ThrowIfLessThan(options.Temperatures, 2, nameof(options.Temperatures));
        if (options.MinimumInverseTemperature is { } minimum && !(minimum >= 0 && minimum < 1))
        {
            throw new ArgumentOutOfRangeException(nameof(options.MinimumInverseTemperature), "The minimum inverse temperature must lie in [0, 1).");
        }

        ArgumentOutOfRangeException.ThrowIfLessThan(options.LocalSteps, 1, nameof(options.LocalSteps));
        if (options.TargetAcceptance is { } target && !(target > 0 && target < 1))
        {
            throw new ArgumentOutOfRangeException(nameof(options.TargetAcceptance), "The target acceptance must lie in (0, 1).");
        }

        chains = options.Chains;
        temperatures = options.Temperatures;
        minimumInverseTemperature = options.MinimumInverseTemperature;
        adaptLadder = options.AdaptLadder;
        localSteps = options.LocalSteps;
        targetAcceptance = options.TargetAcceptance;
        adaptCovariance = options.AdaptCovariance;
        windows = options.Windows ?? throw new ArgumentNullException(nameof(options.Windows));
        _ = new WarmupSchedule(settings.Warmup, windows);
    }

    /// <summary>
    /// Gets the number of independent replica-exchange chains.
    /// <para>独立に実行するレプリカ交換チェーンの数を取得します。</para>
    /// </summary>
    public int Chains => chains;

    /// <summary>
    /// Gets the number of temperatures in each ladder.
    /// <para>各温度ラダーの温度数を取得します。</para>
    /// </summary>
    public int Temperatures => temperatures;

    /// <summary>
    /// Gets the random seed used for sampling (configured value or obtained from the OS).
    /// <para>サンプリングに使用する乱数シード（指定値、または省略時に OS から取得した値）を取得します。</para>
    /// </summary>
    public ulong Seed => settings.Seed;

    /// <summary>
    /// Generates posterior draws using whole-density tempering <c>π(θ)^β</c>, starting all chains and temperatures from the specified initial point.
    /// <para>対数密度全体を温度調整（<c>π(θ)^β</c>）し、すべてのチェーン・温度を指定した初期点から開始して標本を生成します。</para>
    /// </summary>
    /// <param name="logDensity">The delegate evaluating the target log-density. / 対象分布の対数密度を評価するデリゲート。</param>
    /// <param name="initialPoint">The initial point shared by all chains and temperatures. / 全チェーン・全温度で共通の初期点。</param>
    /// <param name="cancellationToken">A token checked at each transition to cancel sampling. / 遷移ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws at inverse temperature 1 and replica-exchange diagnostics. / 逆温度 1 のウォームアップ後標本とレプリカ交換診断情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="logDensity"/> is <see langword="null"/>. / <paramref name="logDensity"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoint"/> is empty or non-finite, or its log-density is non-finite. / <paramref name="initialPoint"/> が空または非有限、あるいは初期点の対数密度が非有限の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><see cref="ParallelTemperingOptions.MinimumInverseTemperature"/> is 0. / <see cref="ParallelTemperingOptions.MinimumInverseTemperature"/> が 0 の場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public ParallelTemperingResult Sample(Func<ReadOnlyVectorSpan<double>, double> logDensity, ReadOnlySpan<double> initialPoint,
        CancellationToken cancellationToken = default) =>
        Sample(LogDensities.FromDelegate(logDensity), initialPoint, cancellationToken);

    /// <summary>
    /// Generates posterior draws using whole-density tempering <c>π(θ)^β</c>, starting each chain from its corresponding column of <paramref name="initialPoints"/>.
    /// <para>対数密度全体を温度調整（<c>π(θ)^β</c>）し、チェーンごとに指定した初期点から標本を生成します。</para>
    /// </summary>
    /// <param name="logDensity">The delegate evaluating the target log-density. / 対象分布の対数密度を評価するデリゲート。</param>
    /// <param name="initialPoints">The matrix of initial points with shape <c>[dimension × chains]</c>; all temperatures in a chain start at its column. / 各チェーンの初期点を列に並べた行列（形状 <c>[次元 × チェーン数]</c>）。各チェーンの全温度はその列から開始します。</param>
    /// <param name="cancellationToken">A token checked at each transition to cancel sampling. / 遷移ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws at inverse temperature 1 and replica-exchange diagnostics. / 逆温度 1 のウォームアップ後標本とレプリカ交換診断情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="logDensity"/> is <see langword="null"/>. / <paramref name="logDensity"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoints"/> is empty, has a column count different from <see cref="Chains"/>, contains non-finite values, or has non-finite log-density. / <paramref name="initialPoints"/> が空または列数が <see cref="Chains"/> と異なる場合、非有限値を含む場合、あるいは初期点の対数密度が非有限の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><see cref="ParallelTemperingOptions.MinimumInverseTemperature"/> is 0. / <see cref="ParallelTemperingOptions.MinimumInverseTemperature"/> が 0 の場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public ParallelTemperingResult Sample(Func<ReadOnlyVectorSpan<double>, double> logDensity, ReadOnlyMatrixSpan<double> initialPoints,
        CancellationToken cancellationToken = default) =>
        Sample(LogDensities.FromDelegate(logDensity), initialPoints, cancellationToken);

    /// <summary>
    /// Generates posterior draws using likelihood-only tempering <c>p(θ)·L(θ)^β</c>, starting all chains and temperatures from the specified initial point.
    /// <para>尤度のみを温度調整（<c>p(θ)·L(θ)^β</c>）し、すべてのチェーン・温度を指定した初期点から開始して標本を生成します。</para>
    /// </summary>
    /// <param name="logPrior">The delegate evaluating the proper log-prior <c>ln p(θ)</c> (must be normalizable for <c>β = 0</c>). / 事前分布の対数密度 <c>ln p(θ)</c>（<c>β = 0</c> でサンプリングするため正規化可能である必要があります）。</param>
    /// <param name="logLikelihood">The delegate evaluating the log-likelihood <c>ln L(θ)</c>. / 対数尤度 <c>ln L(θ)</c> を評価するデリゲート。</param>
    /// <param name="initialPoint">The initial point shared by all chains and temperatures. / 全チェーン・全温度で共通の初期点。</param>
    /// <param name="cancellationToken">A token checked at each transition to cancel sampling. / 遷移ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws at inverse temperature 1 and replica-exchange diagnostics. / 逆温度 1 のウォームアップ後標本とレプリカ交換診断情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="logPrior"/> or <paramref name="logLikelihood"/> is <see langword="null"/>. / <paramref name="logPrior"/> または <paramref name="logLikelihood"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoint"/> is empty or non-finite, or its log-prior or log-likelihood is non-finite. / <paramref name="initialPoint"/> が空または非有限、あるいは初期点の事前対数密度または対数尤度が非有限の場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public ParallelTemperingResult Sample(Func<ReadOnlyVectorSpan<double>, double> logPrior, Func<ReadOnlyVectorSpan<double>, double> logLikelihood,
        ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default) =>
        Sample(LogDensities.FromDelegate(logPrior), LogDensities.FromDelegate(logLikelihood), initialPoint, cancellationToken);

    /// <summary>
    /// Generates posterior draws using likelihood-only tempering <c>p(θ)·L(θ)^β</c>, starting each chain from its corresponding column of <paramref name="initialPoints"/>.
    /// <para>尤度のみを温度調整（<c>p(θ)·L(θ)^β</c>）し、チェーンごとに指定した初期点から標本を生成します。</para>
    /// </summary>
    /// <param name="logPrior">The delegate evaluating the proper log-prior <c>ln p(θ)</c> (must be normalizable for <c>β = 0</c>). / 事前分布の対数密度 <c>ln p(θ)</c>（<c>β = 0</c> でサンプリングするため正規化可能である必要があります）。</param>
    /// <param name="logLikelihood">The delegate evaluating the log-likelihood <c>ln L(θ)</c>. / 対数尤度 <c>ln L(θ)</c> を評価するデリゲート。</param>
    /// <param name="initialPoints">The matrix of initial points with shape <c>[dimension × chains]</c>; all temperatures in a chain start at its column. / 各チェーンの初期点を列に並べた行列（形状 <c>[次元 × チェーン数]</c>）。各チェーンの全温度はその列から開始します。</param>
    /// <param name="cancellationToken">A token checked at each transition to cancel sampling. / 遷移ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws at inverse temperature 1 and replica-exchange diagnostics. / 逆温度 1 のウォームアップ後標本とレプリカ交換診断情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="logPrior"/> or <paramref name="logLikelihood"/> is <see langword="null"/>. / <paramref name="logPrior"/> または <paramref name="logLikelihood"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoints"/> is empty, has a column count different from <see cref="Chains"/>, contains non-finite values, or has non-finite log-prior or log-likelihood. / <paramref name="initialPoints"/> が空または列数が <see cref="Chains"/> と異なる場合、非有限値を含む場合、あるいは初期点の事前対数密度または対数尤度が非有限の場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public ParallelTemperingResult Sample(Func<ReadOnlyVectorSpan<double>, double> logPrior, Func<ReadOnlyVectorSpan<double>, double> logLikelihood,
        ReadOnlyMatrixSpan<double> initialPoints, CancellationToken cancellationToken = default) =>
        Sample(LogDensities.FromDelegate(logPrior), LogDensities.FromDelegate(logLikelihood), initialPoints, cancellationToken);

    /// <summary>
    /// Generates posterior draws using whole-density tempering <c>π(θ)^β</c>, starting all chains and temperatures from the specified initial point.
    /// <para>対数密度全体を温度調整（<c>π(θ)^β</c>）し、すべてのチェーン・温度を指定した初期点から開始して標本を生成します。</para>
    /// </summary>
    /// <typeparam name="TDensity">The log-density type. / 対数密度の型。</typeparam>
    /// <param name="density">The target log-density. / 対象分布の対数密度。</param>
    /// <param name="initialPoint">The initial point shared by all chains and temperatures. / 全チェーン・全温度で共通の初期点。</param>
    /// <param name="cancellationToken">A token checked at each transition to cancel sampling. / 遷移ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws at inverse temperature 1 and replica-exchange diagnostics. / 逆温度 1 のウォームアップ後標本とレプリカ交換診断情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="density"/> is <see langword="null"/>. / <paramref name="density"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoint"/> is empty or non-finite, or its log-density is non-finite. / <paramref name="initialPoint"/> が空または非有限、あるいは初期点の対数密度が非有限の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><see cref="ParallelTemperingOptions.MinimumInverseTemperature"/> is 0. / <see cref="ParallelTemperingOptions.MinimumInverseTemperature"/> が 0 の場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public ParallelTemperingResult Sample<TDensity>(TDensity density, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)
        where TDensity : ILogDensity
    {
        ArgumentNullException.ThrowIfNull(density);
        return Run(new PowerTempering<TDensity>(density), initialPoint.Length, InitialPoints.Broadcast(initialPoint, chains), PowerMinimum(),
            cancellationToken);
    }

    /// <summary>
    /// Generates posterior draws using whole-density tempering <c>π(θ)^β</c>, starting each chain from its corresponding column of <paramref name="initialPoints"/>.
    /// <para>対数密度全体を温度調整（<c>π(θ)^β</c>）し、チェーンごとに指定した初期点から標本を生成します。</para>
    /// </summary>
    /// <typeparam name="TDensity">The log-density type. / 対数密度の型。</typeparam>
    /// <param name="density">The target log-density. / 対象分布の対数密度。</param>
    /// <param name="initialPoints">The matrix of initial points with shape <c>[dimension × chains]</c>; all temperatures in a chain start at its column. / 各チェーンの初期点を列に並べた行列（形状 <c>[次元 × チェーン数]</c>）。各チェーンの全温度はその列から開始します。</param>
    /// <param name="cancellationToken">A token checked at each transition to cancel sampling. / 遷移ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws at inverse temperature 1 and replica-exchange diagnostics. / 逆温度 1 のウォームアップ後標本とレプリカ交換診断情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="density"/> is <see langword="null"/>. / <paramref name="density"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoints"/> is empty, has a column count different from <see cref="Chains"/>, contains non-finite values, or has non-finite log-density. / <paramref name="initialPoints"/> が空または列数が <see cref="Chains"/> と異なる場合、非有限値を含む場合、あるいは初期点の対数密度が非有限の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><see cref="ParallelTemperingOptions.MinimumInverseTemperature"/> is 0. / <see cref="ParallelTemperingOptions.MinimumInverseTemperature"/> が 0 の場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public ParallelTemperingResult Sample<TDensity>(TDensity density, ReadOnlyMatrixSpan<double> initialPoints, CancellationToken cancellationToken = default)
        where TDensity : ILogDensity
    {
        ArgumentNullException.ThrowIfNull(density);
        return Run(new PowerTempering<TDensity>(density), initialPoints.Rows, InitialPoints.Columns(initialPoints, chains), PowerMinimum(),
            cancellationToken);
    }

    /// <summary>
    /// Generates posterior draws using likelihood-only tempering <c>p(θ)·L(θ)^β</c>, starting all chains and temperatures from the specified initial point.
    /// <para>尤度のみを温度調整（<c>p(θ)·L(θ)^β</c>）し、すべてのチェーン・温度を指定した初期点から開始して標本を生成します。</para>
    /// </summary>
    /// <typeparam name="TPrior">The log-prior type. / 事前対数密度の型。</typeparam>
    /// <typeparam name="TLikelihood">The log-likelihood type. / 対数尤度の型。</typeparam>
    /// <param name="logPrior">The proper log-prior <c>ln p(θ)</c> (must be normalizable for <c>β = 0</c>). / 事前分布の対数密度 <c>ln p(θ)</c>（<c>β = 0</c> でサンプリングするため正規化可能である必要があります）。</param>
    /// <param name="logLikelihood">The log-likelihood <c>ln L(θ)</c>. / 対数尤度 <c>ln L(θ)</c>。</param>
    /// <param name="initialPoint">The initial point shared by all chains and temperatures. / 全チェーン・全温度で共通の初期点。</param>
    /// <param name="cancellationToken">A token checked at each transition to cancel sampling. / 遷移ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws at inverse temperature 1 and replica-exchange diagnostics. / 逆温度 1 のウォームアップ後標本とレプリカ交換診断情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="logPrior"/> or <paramref name="logLikelihood"/> is <see langword="null"/>. / <paramref name="logPrior"/> または <paramref name="logLikelihood"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoint"/> is empty or non-finite, or its log-prior or log-likelihood is non-finite. / <paramref name="initialPoint"/> が空または非有限、あるいは初期点の事前対数密度または対数尤度が非有限の場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public ParallelTemperingResult Sample<TPrior, TLikelihood>(TPrior logPrior, TLikelihood logLikelihood, ReadOnlySpan<double> initialPoint,
        CancellationToken cancellationToken = default)
        where TPrior : ILogDensity
        where TLikelihood : ILogDensity
    {
        ArgumentNullException.ThrowIfNull(logPrior);
        ArgumentNullException.ThrowIfNull(logLikelihood);
        return Run(new LikelihoodTempering<TPrior, TLikelihood>(logPrior, logLikelihood), initialPoint.Length,
            InitialPoints.Broadcast(initialPoint, chains), minimumInverseTemperature ?? 0, cancellationToken);
    }

    /// <summary>
    /// Generates posterior draws using likelihood-only tempering <c>p(θ)·L(θ)^β</c>, starting each chain from its corresponding column of <paramref name="initialPoints"/>.
    /// <para>尤度のみを温度調整（<c>p(θ)·L(θ)^β</c>）し、チェーンごとに指定した初期点から標本を生成します。</para>
    /// </summary>
    /// <typeparam name="TPrior">The log-prior type. / 事前対数密度の型。</typeparam>
    /// <typeparam name="TLikelihood">The log-likelihood type. / 対数尤度の型。</typeparam>
    /// <param name="logPrior">The proper log-prior <c>ln p(θ)</c> (must be normalizable for <c>β = 0</c>). / 事前分布の対数密度 <c>ln p(θ)</c>（<c>β = 0</c> でサンプリングするため正規化可能である必要があります）。</param>
    /// <param name="logLikelihood">The log-likelihood <c>ln L(θ)</c>. / 対数尤度 <c>ln L(θ)</c>。</param>
    /// <param name="initialPoints">The matrix of initial points with shape <c>[dimension × chains]</c>; all temperatures in a chain start at its column. / 各チェーンの初期点を列に並べた行列（形状 <c>[次元 × チェーン数]</c>）。各チェーンの全温度はその列から開始します。</param>
    /// <param name="cancellationToken">A token checked at each transition to cancel sampling. / 遷移ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws at inverse temperature 1 and replica-exchange diagnostics. / 逆温度 1 のウォームアップ後標本とレプリカ交換診断情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="logPrior"/> or <paramref name="logLikelihood"/> is <see langword="null"/>. / <paramref name="logPrior"/> または <paramref name="logLikelihood"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoints"/> is empty, has a column count different from <see cref="Chains"/>, contains non-finite values, or has non-finite log-prior or log-likelihood. / <paramref name="initialPoints"/> が空または列数が <see cref="Chains"/> と異なる場合、非有限値を含む場合、あるいは初期点の事前対数密度または対数尤度が非有限の場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public ParallelTemperingResult Sample<TPrior, TLikelihood>(TPrior logPrior, TLikelihood logLikelihood, ReadOnlyMatrixSpan<double> initialPoints,
        CancellationToken cancellationToken = default)
        where TPrior : ILogDensity
        where TLikelihood : ILogDensity
    {
        ArgumentNullException.ThrowIfNull(logPrior);
        ArgumentNullException.ThrowIfNull(logLikelihood);
        return Run(new LikelihoodTempering<TPrior, TLikelihood>(logPrior, logLikelihood), initialPoints.Rows,
            InitialPoints.Columns(initialPoints, chains), minimumInverseTemperature ?? 0, cancellationToken);
    }

    private double PowerMinimum()
    {
        double minimum = minimumInverseTemperature ?? DefaultMinimumPower;
        if (minimum <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(ParallelTemperingOptions.MinimumInverseTemperature),
                "The minimum inverse temperature must be positive when the whole density is tempered.");
        }

        return minimum;
    }

    // β_min > 0 なら等比、β_min = 0 なら等差で初期ラダーを構成する
    internal static double[] InitialLadder(int count, double minimum)
    {
        var betas = new double[count];
        for (int k = 0; k < count; k++)
        {
            double fraction = (double)k / (count - 1);
            betas[k] = minimum > 0 ? Math.Pow(minimum, fraction) : 1 - fraction;
        }

        betas[0] = 1;
        betas[^1] = minimum;
        return betas;
    }

    // 累積交換棄却率を逆温度について区分線形補間し、等間隔になるように中間逆温度を再配置する（Syed et al. 2022）
    internal static void AdaptLadderPoints(Span<double> betas, ReadOnlySpan<double> rejections)
    {
        int count = betas.Length;
        Span<double> cumulative = count <= 256 ? stackalloc double[count] : new double[count];
        cumulative[0] = 0;
        for (int i = 1; i < count; i++)
        {
            double r = rejections[i - 1];
            cumulative[i] = cumulative[i - 1] + (double.IsFinite(r) ? Math.Max(r, MinimumRejection) : MinimumRejection);
        }

        double[] old = betas.ToArray();
        double total = cumulative[^1];
        int segment = 1;
        for (int k = 1; k < count - 1; k++)
        {
            double target = total * k / (count - 1);
            while (cumulative[segment] < target)
            {
                segment++;
            }

            double fraction = (target - cumulative[segment - 1]) / (cumulative[segment] - cumulative[segment - 1]);
            betas[k] = old[segment - 1] + fraction * (old[segment] - old[segment - 1]);
        }
    }

    private ParallelTemperingResult Run<TTempered>(TTempered tempered, int d, double[] starts, double minimum, CancellationToken cancellationToken)
        where TTempered : ITemperedDensity
    {
        if (d == 0)
        {
            throw new ArgumentException("The initial point must not be empty.", "initialPoint");
        }

        int k = temperatures;
        settings.RequireStorage(d, chains);
        var draws = new Tensor<double>(d, settings.Draws, chains);
        var logDensities = new Matrix<double>(settings.Draws, chains);
        var betas = new Matrix<double>(k, chains);
        var swapRates = new Matrix<double>(k - 1, chains);
        var localRates = new Matrix<double>(k, chains);
        var roundTrips = new long[chains];
        var evaluations = new long[chains];
        ChainRunner.Run(chains, settings.MaxDegreeOfParallelism, cancellationToken, c =>
        {
            var chain = new Chain<TTempered>(this, tempered, d, c, minimum);
            chain.Run(starts.AsSpan(c * d, d), draws.GetMatrix(c), logDensities.Column(c), cancellationToken);
            evaluations[c] = chain.Evaluations;
            roundTrips[c] = chain.RoundTrips;
            chain.WriteStatistics(betas.Column(c), swapRates.Column(c), localRates.Column(c));
        });

        return new ParallelTemperingResult(draws, logDensities, evaluations, settings.Seed, betas, swapRates, localRates, roundTrips);
    }

    // 温度調整された分布 ln π_β(θ) = r(θ) + β·ℓ(θ) の非温度調整部 r と温度調整部 ℓ
    private interface ITemperedDensity
    {
        void Evaluate(ReadOnlySpan<double> theta, out double reference, out double tempered);
    }

    private readonly struct PowerTempering<TDensity>(TDensity density) : ITemperedDensity
        where TDensity : ILogDensity
    {
        public void Evaluate(ReadOnlySpan<double> theta, out double reference, out double tempered)
        {
            reference = 0;
            tempered = LogDensities.Sanitize(density.LogDensity(theta));
        }
    }

    private readonly struct LikelihoodTempering<TPrior, TLikelihood>(TPrior prior, TLikelihood likelihood) : ITemperedDensity
        where TPrior : ILogDensity
        where TLikelihood : ILogDensity
    {
        public void Evaluate(ReadOnlySpan<double> theta, out double reference, out double tempered)
        {
            reference = LogDensities.Sanitize(prior.LogDensity(theta));
            tempered = double.IsFinite(reference) ? LogDensities.Sanitize(likelihood.LogDensity(theta)) : double.NegativeInfinity;
        }
    }

    private sealed class Rung(int d, double scale, double targetAcceptance)
    {
        public double[] State = new double[d];
        public double Reference;
        public double Tempered;
        public int Label;
        public readonly CovarianceShape Shape = new(d, dense: true);
        public readonly CovarianceEstimator Estimator = new(d, dense: true);
        public readonly DualAveraging Adaptation = new(targetAcceptance, centerFactor: 1, gamma: 1);
        public double Scale = scale;
        public long Accepted;
        public Philox4x32Random Random = null!;

        public double LogTarget(double beta) => LogTarget(Reference, Tempered, beta);

        public static double LogTarget(double reference, double tempered, double beta) =>
            double.IsFinite(reference) && double.IsFinite(tempered) ? reference + beta * tempered : double.NegativeInfinity;
    }

    private sealed class Chain<TTempered>(ParallelTempering sampler, TTempered tempered, int d, int index, double minimum)
        where TTempered : ITemperedDensity
    {
        private readonly int count = sampler.temperatures;
        private readonly double[] betas = InitialLadder(sampler.temperatures, minimum);
        private readonly Rung[] rungs = new Rung[sampler.temperatures];
        private readonly Philox4x32Random exchange = RandomStreams.Create(sampler.settings.Seed, StreamPurpose.Exchange, index);
        private readonly double[] proposal = new double[d];
        private readonly double[] normal = new double[d];
        private readonly double[] step = new double[d];
        private readonly double[] swapSums = new double[sampler.temperatures - 1];
        private readonly long[] swapAttempts = new long[sampler.temperatures - 1];
        private readonly sbyte[] lastEnd = new sbyte[sampler.temperatures];   // レプリカごとに最後に到達した端（0 = 未到達、−1 = 低温端、+1 = 高温端）

        public long Evaluations { get; private set; }

        public long RoundTrips { get; private set; }

        public void WriteStatistics(VectorSpan<double> betaDestination, VectorSpan<double> swapDestination, VectorSpan<double> localDestination)
        {
            for (int k = 0; k < count; k++)
            {
                betaDestination[k] = betas[k];
                localDestination[k] = (double)rungs[k].Accepted / (sampler.settings.SamplingIterations * sampler.localSteps);
            }

            for (int i = 0; i < count - 1; i++)
            {
                swapDestination[i] = swapAttempts[i] > 0 ? swapSums[i] / swapAttempts[i] : double.NaN;
            }
        }

        public void Run(ReadOnlySpan<double> start, MatrixSpan<double> draws, VectorSpan<double> logDensities, CancellationToken cancellationToken)
        {
            SamplerSettings settings = sampler.settings;
            double optimal = OptimalScaleNumerator / Math.Sqrt(d);
            double target = sampler.targetAcceptance ?? (d == 1 ? 0.44 : 0.234);
            Evaluate(start, out double reference, out double value);
            InitialPoints.RequireValid(start, Rung.LogTarget(reference, value, 1), index);
            for (int k = 0; k < count; k++)
            {
                var rung = new Rung(d, optimal, target)
                {
                    Reference = reference,
                    Tempered = value,
                    Label = k,
                    Random = RandomStreams.Create(settings.Seed, StreamPurpose.Transition, index, k),
                };
                start.CopyTo(rung.State);
                rung.Adaptation.Restart(optimal);
                rungs[k] = rung;
            }

            var schedule = new WarmupSchedule(settings.Warmup, sampler.windows);
            var estimate = new double[d * d];
            var rejections = new double[count - 1];
            long round = 0;
            for (int t = 0; t < settings.Warmup; t++, round++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                for (int k = 0; k < count; k++)
                {
                    Rung rung = rungs[k];
                    for (int s = 0; s < sampler.localSteps; s++)
                    {
                        rung.Scale = rung.Adaptation.Update(LocalStep(rung, betas[k], out _));
                    }
                }

                bool inWindow = schedule.InWindow(t);
                Exchange(round, record: inWindow);
                if (inWindow && sampler.adaptCovariance)
                {
                    foreach (Rung rung in rungs)
                    {
                        rung.Estimator.Add(rung.State);
                    }
                }

                if (schedule.EndsWindow(t))
                {
                    foreach (Rung rung in rungs)
                    {
                        if (sampler.adaptCovariance)
                        {
                            rung.Estimator.WriteRegularized(estimate);
                            rung.Estimator.Reset();
                            if (rung.Shape.TrySet(estimate))
                            {
                                rung.Scale = optimal;
                            }
                        }

                        rung.Adaptation.Restart(rung.Scale);
                    }

                    if (sampler.adaptLadder && count > 2)
                    {
                        for (int i = 0; i < count - 1; i++)
                        {
                            rejections[i] = swapAttempts[i] > 0 ? 1 - swapSums[i] / swapAttempts[i] : double.NaN;
                        }

                        AdaptLadderPoints(betas, rejections);
                    }

                    Array.Clear(swapSums);
                    Array.Clear(swapAttempts);
                }
            }

            foreach (Rung rung in rungs)
            {
                if (settings.Warmup > 0 && double.IsFinite(rung.Adaptation.Averaged))
                {
                    rung.Scale = rung.Adaptation.Averaged;
                }
            }

            Array.Clear(swapSums);
            Array.Clear(swapAttempts);
            int kept = 0;
            for (long t = 0; t < settings.SamplingIterations; t++, round++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                for (int k = 0; k < count; k++)
                {
                    Rung rung = rungs[k];
                    for (int s = 0; s < sampler.localSteps; s++)
                    {
                        LocalStep(rung, betas[k], out bool accepted);
                        if (accepted)
                        {
                            rung.Accepted++;
                        }
                    }
                }

                Exchange(round, record: true);
                TrackRoundTrips();
                if ((t + 1) % settings.Thin == 0)
                {
                    Rung cold = rungs[0];
                    cold.State.CopyTo(draws.Column(kept).Data);
                    logDensities[kept] = cold.LogTarget(1);
                    kept++;
                }
            }
        }

        private double LocalStep(Rung rung, double beta, out bool accepted)
        {
            Gaussian.Fill(rung.Random, normal);
            rung.Shape.MultiplyFactor(normal, step);
            for (int i = 0; i < d; i++)
            {
                proposal[i] = rung.State[i] + rung.Scale * step[i];
            }

            Evaluate(proposal, out double reference, out double value);
            double logRatio = Rung.LogTarget(reference, value, beta) - rung.LogTarget(beta);
            accepted = Math.Log(rung.Random.NextDoubleOpen()) < logRatio;
            if (accepted)
            {
                proposal.CopyTo(rung.State, 0);
                rung.Reference = reference;
                rung.Tempered = value;
            }

            return logRatio >= 0 ? 1 : Math.Exp(logRatio);
        }

        // 非可逆交換（Syed et al. 2022）: 偶数ラウンドは (0,1), (2,3), …、奇数ラウンドは (1,2), (3,4), … を試行する
        private void Exchange(long round, bool record)
        {
            for (int i = (int)(round & 1); i + 1 < count; i += 2)
            {
                Rung colder = rungs[i], hotter = rungs[i + 1];
                double logRatio = (betas[i] - betas[i + 1]) * (hotter.Tempered - colder.Tempered);
                double probability = logRatio >= 0 ? 1 : Math.Exp(logRatio);
                if (record)
                {
                    swapSums[i] += probability;
                    swapAttempts[i]++;
                }

                if (Math.Log(exchange.NextDoubleOpen()) < logRatio)
                {
                    (colder.State, hotter.State) = (hotter.State, colder.State);
                    (colder.Reference, hotter.Reference) = (hotter.Reference, colder.Reference);
                    (colder.Tempered, hotter.Tempered) = (hotter.Tempered, colder.Tempered);
                    (colder.Label, hotter.Label) = (hotter.Label, colder.Label);
                }
            }
        }

        private void TrackRoundTrips()
        {
            int hot = rungs[^1].Label;
            if (lastEnd[hot] == -1)
            {
                lastEnd[hot] = 1;
            }

            int cold = rungs[0].Label;
            if (lastEnd[cold] == 1)
            {
                RoundTrips++;
            }

            lastEnd[cold] = -1;
        }

        private void Evaluate(ReadOnlySpan<double> point, out double reference, out double value)
        {
            Evaluations++;
            tempered.Evaluate(point, out reference, out value);
        }
    }
}

/// <summary>
/// Represents the target-temperature draws and replica-exchange diagnostics produced by <see cref="ParallelTempering"/>.
/// <para><see cref="ParallelTempering"/> の逆温度 1 の標本とレプリカ交換診断情報を表します。</para>
/// </summary>
public sealed class ParallelTemperingResult : SamplingResult
{
    private readonly long[] roundTrips;

    internal ParallelTemperingResult(Tensor<double> draws, Matrix<double> logDensities, long[] evaluations, ulong seed,
        Matrix<double> inverseTemperatures, Matrix<double> swapAcceptanceRates, Matrix<double> localAcceptanceRates, long[] roundTrips)
        : base(draws, logDensities, evaluations, seed)
    {
        InverseTemperatures = inverseTemperatures;
        SwapAcceptanceRates = swapAcceptanceRates;
        LocalAcceptanceRates = localAcceptanceRates;
        this.roundTrips = roundTrips;
    }

    /// <summary>
    /// Gets the post-warmup inverse temperature ladder for each chain, with shape <c>[temperatures × chains]</c> (each column decreases from 1 to <c>β_min</c>).
    /// <para>チェーンごとのウォームアップ後に固定された逆温度ラダー（形状 <c>[温度数 × チェーン数]</c>、各列は 1 から <c>β_min</c> へ単調減少）を取得します。</para>
    /// </summary>
    public Matrix<double> InverseTemperatures { get; }

    /// <summary>
    /// Gets the post-warmup mean swap acceptance probability between adjacent temperatures <c>(i, i + 1)</c> for each chain, with shape <c>[(temperatures − 1) × chains]</c>.
    /// <para>チェーンごとのウォームアップ後の隣接温度間 <c>(i, i + 1)</c> の平均交換受理確率（形状 <c>[(温度数 − 1) × チェーン数]</c>）を取得します。</para>
    /// </summary>
    public Matrix<double> SwapAcceptanceRates { get; }

    /// <summary>
    /// Gets the post-warmup local random-walk Metropolis acceptance rate at each temperature for each chain, with shape <c>[temperatures × chains]</c>.
    /// <para>チェーンごとのウォームアップ後の各温度におけるローカルランダムウォーク・メトロポリス受理率（形状 <c>[温度数 × チェーン数]</c>）を取得します。</para>
    /// </summary>
    public Matrix<double> LocalAcceptanceRates { get; }

    /// <summary>
    /// Gets the number of post-warmup cold-to-hot-to-cold round trips for each chain.
    /// <para>チェーンごとのウォームアップ後の低温端（逆温度 1）→ 高温端 → 低温端の往復回数を取得します。</para>
    /// </summary>
    /// <remarks>
    /// <para>Zero or very few round trips indicate poor mixing across temperatures; consider increasing <see cref="ParallelTemperingOptions.Temperatures"/> or lowering <see cref="ParallelTemperingOptions.MinimumInverseTemperature"/>.</para>
    /// <para>0 または極めて少ない場合は温度間の混合が不十分であるため、<see cref="ParallelTemperingOptions.Temperatures"/> を増やすか <see cref="ParallelTemperingOptions.MinimumInverseTemperature"/> を下げることを検討してください。</para>
    /// </remarks>
    public IReadOnlyList<long> RoundTrips => roundTrips;

    /// <summary>
    /// Gets the estimated global communication barrier <c>Λ</c> (sum of adjacent swap rejection rates; Syed et al. 2022) for each chain.
    /// <para>チェーンごとの大域的通信障壁 <c>Λ</c>（隣接温度間の交換棄却率の総和、Syed et al. 2022）の推定値を取得します。</para>
    /// </summary>
    /// <remarks>
    /// <para>With non-reversible swaps and sufficiently many temperatures, the asymptotic round-trip rate approaches <c>1 / (2 + 2Λ)</c> per round, and roughly <c>2Λ</c> temperatures suffice.</para>
    /// <para>非可逆交換で温度数を十分に増やしたときの漸近往復頻度は 1 ラウンド当たり約 <c>1 / (2 + 2Λ)</c> であり、温度数はおよそ <c>2Λ</c> で十分とされます。</para>
    /// </remarks>
    public IReadOnlyList<double> CommunicationBarriers
    {
        get
        {
            int chains = SwapAcceptanceRates.Columns;
            var result = new double[chains];
            for (int c = 0; c < chains; c++)
            {
                double sum = 0;
                for (int i = 0; i < SwapAcceptanceRates.Rows; i++)
                {
                    sum += 1 - SwapAcceptanceRates[i, c];
                }

                result[c] = sum;
            }

            return result;
        }
    }
}
