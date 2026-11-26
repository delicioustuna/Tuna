using Tuna.LinearAlgebra;

namespace Tuna.Probabilistic.Inference;

/// <summary>
/// Configures <see cref="HamiltonianMonteCarlo"/>.
/// <para><see cref="HamiltonianMonteCarlo"/> の設定オプションを表します。</para>
/// </summary>
public sealed class HamiltonianMonteCarloOptions : HamiltonianSamplerOptions
{
    /// <summary>
    /// Gets or initializes the number of leapfrog steps <c>L</c> per transition (1 or greater; default is 10).
    /// <para>1 回の遷移あたりのリープフロッグ回数 <c>L</c> を取得または設定します（1 以上、既定値 10）。</para>
    /// </summary>
    public int Steps { get; init; } = 10;

    /// <summary>
    /// Gets or initializes the per-transition step-size jitter factor <c>j</c> in <c>[0, 1)</c> (default is 0).
    /// <para>遷移ごとのステップ幅の揺らぎ係数 <c>j</c>（<c>[0, 1)</c>、既定値 0）を取得または設定します。</para>
    /// </summary>
    /// <value>Each transition uses <c>ε(1 + j(2u − 1))</c>, where <c>u ~ Uniform(0, 1)</c> and <c>ε</c> is the base step size. / 各遷移のステップ幅は <c>ε(1 + j(2u − 1))</c>（<c>u ~ Uniform(0, 1)</c>、<c>ε</c> は基準ステップ幅）となります。</value>
    /// <remarks>
    /// <para>Adding jitter helps prevent periodic trajectories from stalling exploration (Neal 2011, §5.4.2.2).</para>
    /// <para>軌道長が目標分布の周期と共振して探索が停滞するのを防ぐために使用します（Neal 2011, §5.4.2.2）。</para>
    /// </remarks>
    public double StepSizeJitter { get; init; }
}

/// <summary>
/// Samples from a differentiable log-density using fixed-trajectory Hamiltonian Monte Carlo (HMC).
/// <para>固定軌道長のハミルトニアン・モンテカルロ法（HMC）により、微分可能な対数密度からサンプリングします。</para>
/// </summary>
/// <remarks>
/// <para>
/// Supply the target density and its gradient via <see cref="IDifferentiableLogDensity"/>.
/// For automatic trajectory-length selection, use <see cref="NoUTurnSampler"/>.
/// Transitions whose Hamiltonian increases beyond <see cref="HamiltonianSamplerOptions.DivergenceThreshold"/> are terminated, rejected, and recorded as divergent.
/// </para>
/// <para>
/// 対象分布の対数密度と勾配を <see cref="IDifferentiableLogDensity"/> で渡します。
/// 軌道長を自動調整したい場合は <see cref="NoUTurnSampler"/> を使用してください。
/// ハミルトニアンの増加が <see cref="HamiltonianSamplerOptions.DivergenceThreshold"/> を超えた遷移は打ち切って棄却し、発散として記録します。
/// </para>
/// </remarks>
public sealed class HamiltonianMonteCarlo
{
    private readonly HamiltonianSettings settings;
    private readonly int steps;
    private readonly double jitter;

    /// <summary>
    /// Initializes a new instance of <see cref="HamiltonianMonteCarlo"/> with the specified options.
    /// <para>指定した設定で <see cref="HamiltonianMonteCarlo"/> の新しいインスタンスを初期化します。</para>
    /// </summary>
    /// <param name="options">The sampler options, or <see langword="null"/> for defaults. / サンプラー設定（<see langword="null"/> の場合は既定値）。</param>
    /// <exception cref="ArgumentNullException"><see cref="HamiltonianSamplerOptions.Windows"/> is <see langword="null"/>. / <see cref="HamiltonianSamplerOptions.Windows"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option value is out of its valid range. / 設定値が有効範囲外の場合。</exception>
    /// <exception cref="ArgumentException"><see cref="HamiltonianSamplerOptions.InitialInverseMetric"/> is not square. / <see cref="HamiltonianSamplerOptions.InitialInverseMetric"/> が正方行列でない場合。</exception>
    public HamiltonianMonteCarlo(HamiltonianMonteCarloOptions? options = null)
    {
        options ??= new HamiltonianMonteCarloOptions();
        settings = HamiltonianSettings.Validate(options, defaultTarget: 0.65);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.Steps, 1, nameof(options.Steps));
        if (!(options.StepSizeJitter >= 0 && options.StepSizeJitter < 1))
        {
            throw new ArgumentOutOfRangeException(nameof(options.StepSizeJitter), "The step size jitter must lie in [0, 1).");
        }

        steps = options.Steps;
        jitter = options.StepSizeJitter;
    }

    /// <summary>
    /// Gets the number of independent chains.
    /// <para>独立したチェーンの数を取得します。</para>
    /// </summary>
    public int Chains => settings.Chains;

    /// <summary>
    /// Gets the random seed used for sampling (configured value or obtained from the OS).
    /// <para>サンプリングに使用する乱数シード（指定値、または省略時に OS から取得した値）を取得します。</para>
    /// </summary>
    public ulong Seed => settings.Common.Seed;

    /// <summary>
    /// Generates posterior draws starting all chains from the specified initial point.
    /// <para>すべてのチェーンを指定した初期点から開始して標本を生成します。</para>
    /// </summary>
    /// <param name="logDensity">The delegate returning the log-density and writing its gradient to the second argument. / 対数密度を返し、第 2 引数に勾配を書き込むデリゲート。</param>
    /// <param name="initialPoint">The initial point shared by all chains (length equals parameter dimension). / 全チェーン共通の初期点（長さはパラメータ次元）。</param>
    /// <param name="cancellationToken">A token checked at each transition to cancel sampling. / 遷移ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws and Hamiltonian diagnostics. / ウォームアップ後の標本とハミルトニアン診断情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="logDensity"/> is <see langword="null"/>. / <paramref name="logDensity"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoint"/> is empty or non-finite, its log-density or gradient is non-finite, or <see cref="HamiltonianSamplerOptions.InitialInverseMetric"/> has mismatched dimensions or is not positive definite. / <paramref name="initialPoint"/> が空または非有限、初期点の対数密度や勾配が非有限、または初期逆計量の次元不一致や非正定値の場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public HamiltonianSamplingResult Sample(Func<ReadOnlyVectorSpan<double>, VectorSpan<double>, double> logDensity, ReadOnlySpan<double> initialPoint,
        CancellationToken cancellationToken = default) =>
        Sample(LogDensities.FromDelegate(logDensity), initialPoint, cancellationToken);

    /// <summary>
    /// Generates posterior draws starting each chain from its corresponding column of <paramref name="initialPoints"/>.
    /// <para>チェーンごとに指定した初期点から標本を生成します。</para>
    /// </summary>
    /// <param name="logDensity">The delegate returning the log-density and writing its gradient to the second argument. / 対数密度を返し、第 2 引数に勾配を書き込むデリゲート。</param>
    /// <param name="initialPoints">The matrix of initial points with shape <c>[dimension × chains]</c>. / 各チェーンの初期点を列に並べた行列（形状 <c>[次元 × チェーン数]</c>）。</param>
    /// <param name="cancellationToken">A token checked at each transition to cancel sampling. / 遷移ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws and Hamiltonian diagnostics. / ウォームアップ後の標本とハミルトニアン診断情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="logDensity"/> is <see langword="null"/>. / <paramref name="logDensity"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoints"/> is empty, has a column count different from <see cref="Chains"/>, contains non-finite values, has non-finite log-density or gradient, or <see cref="HamiltonianSamplerOptions.InitialInverseMetric"/> is invalid. / <paramref name="initialPoints"/> が空または列数が <see cref="Chains"/> と異なる場合、非有限値を含む場合、初期点の対数密度や勾配が非有限の場合、または初期逆計量が不正な場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public HamiltonianSamplingResult Sample(Func<ReadOnlyVectorSpan<double>, VectorSpan<double>, double> logDensity, ReadOnlyMatrixSpan<double> initialPoints,
        CancellationToken cancellationToken = default) =>
        Sample(LogDensities.FromDelegate(logDensity), initialPoints, cancellationToken);

    /// <summary>
    /// Generates posterior draws starting all chains from the specified initial point.
    /// <para>すべてのチェーンを指定した初期点から開始して標本を生成します。</para>
    /// </summary>
    /// <typeparam name="TDensity">The differentiable log-density type. / 微分可能な対数密度の型。</typeparam>
    /// <param name="density">The target differentiable log-density. / 対象分布の微分可能な対数密度。</param>
    /// <param name="initialPoint">The initial point shared by all chains (length equals parameter dimension). / 全チェーン共通の初期点（長さはパラメータ次元）。</param>
    /// <param name="cancellationToken">A token checked at each transition to cancel sampling. / 遷移ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws and Hamiltonian diagnostics. / ウォームアップ後の標本とハミルトニアン診断情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="density"/> is <see langword="null"/>. / <paramref name="density"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoint"/> is empty or non-finite, its log-density or gradient is non-finite, or <see cref="HamiltonianSamplerOptions.InitialInverseMetric"/> has mismatched dimensions or is not positive definite. / <paramref name="initialPoint"/> が空または非有限、初期点の対数密度や勾配が非有限、または初期逆計量の次元不一致や非正定値の場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public HamiltonianSamplingResult Sample<TDensity>(TDensity density, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)
        where TDensity : IDifferentiableLogDensity =>
        Run(density, initialPoint.Length, InitialPoints.Broadcast(initialPoint, settings.Chains), cancellationToken);

    /// <summary>
    /// Generates posterior draws starting each chain from its corresponding column of <paramref name="initialPoints"/>.
    /// <para>チェーンごとに指定した初期点から標本を生成します。</para>
    /// </summary>
    /// <typeparam name="TDensity">The differentiable log-density type. / 微分可能な対数密度の型。</typeparam>
    /// <param name="density">The target differentiable log-density. / 対象分布の微分可能な対数密度。</param>
    /// <param name="initialPoints">The matrix of initial points with shape <c>[dimension × chains]</c>. / 各チェーンの初期点を列に並べた行列（形状 <c>[次元 × チェーン数]</c>）。</param>
    /// <param name="cancellationToken">A token checked at each transition to cancel sampling. / 遷移ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws and Hamiltonian diagnostics. / ウォームアップ後の標本とハミルトニアン診断情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="density"/> is <see langword="null"/>. / <paramref name="density"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoints"/> is empty, has a column count different from <see cref="Chains"/>, contains non-finite values, has non-finite log-density or gradient, or <see cref="HamiltonianSamplerOptions.InitialInverseMetric"/> is invalid. / <paramref name="initialPoints"/> が空または列数が <see cref="Chains"/> と異なる場合、非有限値を含む場合、初期点の対数密度や勾配が非有限の場合、または初期逆計量が不正な場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public HamiltonianSamplingResult Sample<TDensity>(TDensity density, ReadOnlyMatrixSpan<double> initialPoints, CancellationToken cancellationToken = default)
        where TDensity : IDifferentiableLogDensity =>
        Run(density, initialPoints.Rows, InitialPoints.Columns(initialPoints, settings.Chains), cancellationToken);

    private HamiltonianSamplingResult Run<TDensity>(TDensity density, int d, double[] starts, CancellationToken cancellationToken)
        where TDensity : IDifferentiableLogDensity =>
        HamiltonianDriver.Run(settings, density, d, starts, maxTreeDepth: 0,
            c => new Chain<TDensity>(density, d, c, settings, steps, jitter), cancellationToken);

    private sealed class Chain<TDensity>(TDensity density, int d, int chain, HamiltonianSettings settings, int steps, double jitter)
        : HamiltonianChain<TDensity>(density, d, chain, settings)
        where TDensity : IDifferentiableLogDensity
    {
        protected override void Transition(double epsilon)
        {
            SampleMomentum(Current);
            double h0 = Current.Hamiltonian();
            if (jitter > 0)
            {
                epsilon *= 1 + jitter * (2 * Random.NextDouble() - 1);
            }

            Trial.CopyFrom(Current);
            int taken = 0;
            bool divergent = false;
            for (int k = 0; k < steps; k++)
            {
                Leapfrog(Trial, epsilon);
                taken++;
                if (!(Trial.Hamiltonian() - h0 <= DivergenceThreshold))
                {
                    divergent = true;
                    break;
                }
            }

            double change = divergent ? double.PositiveInfinity : Trial.Hamiltonian() - h0;
            double u = Random.NextDoubleOpen();
            bool accept = !divergent && (change <= 0 || Math.Log(u) < -change);
            if (accept)
            {
                Current.CopyFrom(Trial);
            }

            LastAcceptance = divergent ? 0 : change <= 0 ? 1 : Math.Exp(-change);
            LastEnergy = accept ? Trial.Hamiltonian() : h0;
            LastDepth = 0;
            LastLeapfrogs = taken;
            LastDivergent = divergent;
        }
    }
}
