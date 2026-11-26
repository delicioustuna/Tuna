using Tuna.LinearAlgebra;

namespace Tuna.Probabilistic.Inference;

/// <summary>
/// Configures <see cref="NoUTurnSampler"/>.
/// <para><see cref="NoUTurnSampler"/> の設定オプションを表します。</para>
/// </summary>
public sealed class NoUTurnSamplerOptions : HamiltonianSamplerOptions
{
    /// <summary>
    /// Gets or initializes the maximum binary tree depth (1 to 30; default is 10). Each transition performs at most <c>2^MaxTreeDepth − 1</c> leapfrog steps.
    /// <para>二分木の深さの上限を取得または設定します（1 以上 30 以下、既定値 10）。1 遷移あたりのリープフロッグ回数は最大 <c>2^MaxTreeDepth − 1</c> 回です。</para>
    /// </summary>
    public int MaxTreeDepth { get; init; } = 10;
}

/// <summary>
/// Samples from a differentiable log-density using the multinomial No-U-Turn Sampler (NUTS).
/// <para>軌道長を自動調整する多項サンプリング版 No-U-Turn サンプラー（NUTS）により、微分可能な対数密度からサンプリングします。</para>
/// </summary>
/// <remarks>
/// <para>
/// Supply the target density and its gradient via <see cref="IDifferentiableLogDensity"/>; wrap constrained parameters with <see cref="TransformedDifferentiableLogDensity{TDensity, TBijector}"/>.
/// Trajectories whose Hamiltonian increases beyond <see cref="HamiltonianSamplerOptions.DivergenceThreshold"/> are terminated and recorded as divergent.
/// Implements Hoffman &amp; Gelman (2014) NUTS with Betancourt (2017) multinomial trajectory sampling.
/// </para>
/// <para>
/// 対象分布の対数密度と勾配を <see cref="IDifferentiableLogDensity"/> で渡します。制約付きパラメータには <see cref="TransformedDifferentiableLogDensity{TDensity, TBijector}"/> を使用してください。
/// ハミルトニアンの増加が <see cref="HamiltonianSamplerOptions.DivergenceThreshold"/> を超えた場合は軌道を打ち切り、発散として記録します。
/// Hoffman &amp; Gelman (2014) の NUTS に Betancourt (2017) の多項サンプリングを組み合わせた実装です。
/// </para>
/// </remarks>
public sealed class NoUTurnSampler
{
    private readonly HamiltonianSettings settings;
    private readonly int maxTreeDepth;

    /// <summary>
    /// Initializes a new instance of <see cref="NoUTurnSampler"/> with the specified options.
    /// <para>指定した設定で <see cref="NoUTurnSampler"/> の新しいインスタンスを初期化します。</para>
    /// </summary>
    /// <param name="options">The sampler options, or <see langword="null"/> for defaults. / サンプラー設定（<see langword="null"/> の場合は既定値）。</param>
    /// <exception cref="ArgumentNullException"><see cref="HamiltonianSamplerOptions.Windows"/> is <see langword="null"/>. / <see cref="HamiltonianSamplerOptions.Windows"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option value is out of its valid range. / 設定値が有効範囲外の場合。</exception>
    /// <exception cref="ArgumentException"><see cref="HamiltonianSamplerOptions.InitialInverseMetric"/> is not square. / <see cref="HamiltonianSamplerOptions.InitialInverseMetric"/> が正方行列でない場合。</exception>
    public NoUTurnSampler(NoUTurnSamplerOptions? options = null)
    {
        options ??= new NoUTurnSamplerOptions();
        settings = HamiltonianSettings.Validate(options, defaultTarget: 0.8);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.MaxTreeDepth, 1, nameof(options.MaxTreeDepth));
        ArgumentOutOfRangeException.ThrowIfGreaterThan(options.MaxTreeDepth, 30, nameof(options.MaxTreeDepth));
        maxTreeDepth = options.MaxTreeDepth;
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
        HamiltonianDriver.Run(settings, density, d, starts, maxTreeDepth,
            c => new Chain<TDensity>(density, d, c, settings, maxTreeDepth), cancellationToken);

    private sealed class Subtree(int dimension)
    {
        public readonly Phase Near = new(dimension);
        public readonly Phase Far = new(dimension);
        public readonly Phase Sample = new(dimension);
        public readonly double[] Rho = new double[dimension];       // 部分木内の運動量和 ρ
        public double LogWeight;                                    // ln Σ e^{H₀ − H}
        public bool Valid;
    }

    private sealed class Chain<TDensity> : HamiltonianChain<TDensity> where TDensity : IDifferentiableLogDensity
    {
        private readonly int maxDepth;
        private readonly Subtree[,] levels;     // [深さ, 0 = 前半・1 = 後半] の作業領域
        private readonly Subtree extension;
        private readonly Phase left, right;
        private readonly Phase oldEdge;
        private readonly Phase selected;
        private readonly double[] rho;
        private readonly double[] rhoWork;
        private double h0;
        private double acceptanceSum;
        private int leapfrogs;
        private bool divergent;

        public Chain(TDensity density, int d, int chain, HamiltonianSettings settings, int maxDepth)
            : base(density, d, chain, settings)
        {
            this.maxDepth = maxDepth;
            levels = new Subtree[maxDepth, 2];
            for (int k = 0; k < maxDepth; k++)
            {
                levels[k, 0] = new Subtree(d);
                levels[k, 1] = new Subtree(d);
            }

            extension = new Subtree(d);
            left = new Phase(d);
            right = new Phase(d);
            oldEdge = new Phase(d);
            selected = new Phase(d);
            rho = new double[d];
            rhoWork = new double[d];
        }

        protected override void Transition(double epsilon)
        {
            SampleMomentum(Current);
            h0 = Current.Hamiltonian();
            left.CopyFrom(Current);
            right.CopyFrom(Current);
            selected.CopyFrom(Current);
            Current.Momentum.CopyTo(rho, 0);
            double logWeight = 0;
            acceptanceSum = 0;
            leapfrogs = 0;
            divergent = false;
            int depth = 0;
            while (depth < maxDepth)
            {
                bool forward = Random.NextDouble() < 0.5;
                Phase edge = forward ? right : left;
                Phase opposite = forward ? left : right;
                oldEdge.CopyFrom(edge);
                Build(depth, forward ? epsilon : -epsilon, edge, extension);
                depth++;
                if (!extension.Valid)
                {
                    break;
                }

                // バイアス付き漸進サンプリング: 新規部分木の代表点を確率 min(1, w_new / w_old) で採用する
                double logAccept = extension.LogWeight - logWeight;
                double u = Random.NextDoubleOpen();
                if (logAccept >= 0 || Math.Log(u) < logAccept)
                {
                    selected.CopyFrom(extension.Sample);
                }

                logWeight = XMath.LogAddExp(logWeight, extension.LogWeight);
                edge.CopyFrom(extension.Far);
                for (int i = 0; i < D; i++)
                {
                    rhoWork[i] = rho[i] + extension.Near.Momentum[i];
                    rho[i] += extension.Rho[i];
                }

                bool continues = NoUTurn(left, right, rho)
                    && NoUTurn(opposite, extension.Near, rhoWork)
                    && NoUTurn(oldEdge, extension.Far, Sum(oldEdge.Momentum, extension.Rho));
                if (!continues)
                {
                    break;
                }
            }

            Current.CopyFrom(selected);
            LastAcceptance = leapfrogs > 0 ? acceptanceSum / leapfrogs : 0;
            LastEnergy = selected.Hamiltonian();
            LastDepth = depth;
            LastLeapfrogs = leapfrogs;
            LastDivergent = divergent;
        }

        private void Build(int depth, double epsilon, Phase from, Subtree result)
        {
            if (depth == 0)
            {
                Phase z = result.Near;
                z.CopyFrom(from);
                Leapfrog(z, epsilon);
                leapfrogs++;
                double change = z.Hamiltonian() - h0;
                if (!(change <= DivergenceThreshold))
                {
                    divergent = true;
                    result.Valid = false;
                    return;
                }

                acceptanceSum += change <= 0 ? 1 : Math.Exp(-change);
                result.Far.CopyFrom(z);
                result.Sample.CopyFrom(z);
                z.Momentum.CopyTo(result.Rho, 0);
                result.LogWeight = -change;
                result.Valid = true;
                return;
            }

            Subtree first = levels[depth - 1, 0], second = levels[depth - 1, 1];
            Build(depth - 1, epsilon, from, first);
            if (!first.Valid)
            {
                result.Valid = false;
                return;
            }

            Build(depth - 1, epsilon, first.Far, second);
            if (!second.Valid)
            {
                result.Valid = false;
                return;
            }

            result.Near.CopyFrom(first.Near);
            result.Far.CopyFrom(second.Far);
            result.LogWeight = XMath.LogAddExp(first.LogWeight, second.LogWeight);
            // 部分木内は重みに比例した多項サンプリングで代表点を選ぶ
            double logChooseSecond = second.LogWeight - result.LogWeight;
            result.Sample.CopyFrom(Math.Log(Random.NextDoubleOpen()) < logChooseSecond ? second.Sample : first.Sample);
            for (int i = 0; i < D; i++)
            {
                result.Rho[i] = first.Rho[i] + second.Rho[i];
            }

            result.Valid = NoUTurn(result.Near, result.Far, result.Rho)
                && NoUTurn(first.Near, second.Near, Sum(first.Rho, second.Near.Momentum))
                && NoUTurn(first.Far, second.Far, Sum(first.Far.Momentum, second.Rho));
        }

        private double[] Sum(double[] a, double[] b)
        {
            for (int i = 0; i < D; i++)
            {
                rhoWork[i] = a[i] + b[i];
            }

            return rhoWork;
        }

        // 一般化 U ターン判定（Betancourt 2017）: 両端の速度と運動量和の内積がいずれも正なら継続する
        private bool NoUTurn(Phase a, Phase b, double[] momentumSum)
        {
            double da = 0, db = 0;
            for (int i = 0; i < D; i++)
            {
                da += a.Velocity[i] * momentumSum[i];
                db += b.Velocity[i] * momentumSum[i];
            }

            return da > 0 && db > 0;
        }
    }
}
