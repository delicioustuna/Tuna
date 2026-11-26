using Tuna.LinearAlgebra;
using Tuna.RandomSource;

namespace Tuna.Probabilistic.Inference;

/// <summary>
/// Configures <see cref="SliceSampler"/>.
/// <para><see cref="SliceSampler"/> の設定オプションを表します。</para>
/// </summary>
public sealed class SliceSamplerOptions : MultiChainSamplerOptions
{
    /// <summary>
    /// Gets or initializes the initial slice bracket width <c>w</c> along each direction (positive finite value; default is 1). Used until the end of the first warmup window when adapting widths.
    /// <para>各方向の初期スライス幅 <c>w</c> を取得または設定します（正の有限値、既定値 1）。幅を適応させる場合も最初のウィンドウ終了まではこの値が使用されます。</para>
    /// </summary>
    public double InitialWidth { get; init; } = 1;

    /// <summary>
    /// Gets or initializes the maximum number of stepping-out expansions <c>m</c> per 1D slice update (1 or greater; default is 100, per Neal 2003 Fig. 3). Bounds the interval to <c>m·w</c> so flat densities still terminate.
    /// <para>1 次元スライス更新あたりのステップアウト拡張回数の上限 <c>m</c> を取得または設定します（1 以上、既定値 100、Neal 2003 Fig. 3）。区間幅を <c>m·w</c> 以内に制限し、平坦な密度でも停止するようにします。</para>
    /// </summary>
    public int MaxSteppingOut { get; init; } = 100;

    /// <summary>
    /// Gets or initializes a value indicating whether slice widths are adapted to <c>3</c> times the windowed sample standard deviation along each direction at the end of each warmup window (default is <see langword="true"/>).
    /// <para>ウォームアップの各ウィンドウ終端で、各方向のスライス幅を標本標準偏差の 3 倍へ更新するかどうかを取得または設定します（既定値 <see langword="true"/>）。</para>
    /// </summary>
    public bool AdaptWidths { get; init; } = true;

    /// <summary>
    /// Gets or initializes a value indicating whether orthogonal slice update directions are adapted during warmup (default is <see langword="true"/>).
    /// <para>ウォームアップ期間中にスライスの直交更新方向を適応させるかどうかを取得または設定します（既定値 <see langword="true"/>）。</para>
    /// </summary>
    /// <value><see langword="true"/> to use eigenvectors of the windowed sample covariance (factor slice sampling); <see langword="false"/> to update along coordinate axes. / <see langword="true"/> の場合はウィンドウごとの標本共分散の固有ベクトル（因子スライス法）、<see langword="false"/> の場合は座標軸に沿って更新します。</value>
    public bool AdaptDirections { get; init; } = true;

    /// <summary>
    /// Gets or initializes the warmup adaptation window schedule.
    /// <para>ウォームアップ期間の適応ウィンドウ設定を取得または設定します。</para>
    /// </summary>
    public WarmupWindowOptions Windows { get; init; } = new();
}

/// <summary>
/// Samples from a log-density using orthogonal stepping-out and shrinkage slice sampling.
/// <para>直交する方向に沿って区間を拡張・縮小するスライスサンプラーにより、対数密度からサンプリングします。</para>
/// </summary>
/// <remarks>
/// <para>Gradients are not required and discontinuous log-densities are supported. When enabled, slice directions and widths are adapted during warmup and frozen during sampling.</para>
/// <para>勾配は不要であり、不連続な対数密度にも適用できます。方向や区間幅の適応が有効な場合、ウォームアップ期間中に更新し、サンプリング期間中は固定します。</para>
/// </remarks>
public sealed class SliceSampler
{
    private const double WidthPerStandardDeviation = 3;
    private const int MaxShrinkage = 10_000;

    private readonly SamplerSettings settings;
    private readonly int chains;
    private readonly double initialWidth;
    private readonly int maxSteppingOut;
    private readonly bool adaptWidths;
    private readonly bool adaptDirections;
    private readonly WarmupWindowOptions windows;

    /// <summary>
    /// Initializes a new instance of <see cref="SliceSampler"/> with the specified options.
    /// <para>指定した設定で <see cref="SliceSampler"/> の新しいインスタンスを初期化します。</para>
    /// </summary>
    /// <param name="options">The sampler options, or <see langword="null"/> for defaults. / サンプラー設定（<see langword="null"/> の場合は既定値）。</param>
    /// <exception cref="ArgumentNullException"><see cref="SliceSamplerOptions.Windows"/> is <see langword="null"/>. / <see cref="SliceSamplerOptions.Windows"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">An option value is out of its valid range. / 設定値が有効範囲外の場合。</exception>
    public SliceSampler(SliceSamplerOptions? options = null)
    {
        options ??= new SliceSamplerOptions();
        settings = SamplerSettings.Validate(options);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.Chains, 1, nameof(options.Chains));
        if (!(options.InitialWidth > 0 && double.IsFinite(options.InitialWidth)))
        {
            throw new ArgumentOutOfRangeException(nameof(options.InitialWidth), "The initial width must be positive and finite.");
        }

        ArgumentOutOfRangeException.ThrowIfLessThan(options.MaxSteppingOut, 1, nameof(options.MaxSteppingOut));
        chains = options.Chains;
        initialWidth = options.InitialWidth;
        maxSteppingOut = options.MaxSteppingOut;
        adaptWidths = options.AdaptWidths;
        adaptDirections = options.AdaptDirections;
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
    /// <returns>The post-warmup draws and slice direction/width diagnostics. / ウォームアップ後の標本およびスライス方向・幅の情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="logDensity"/> is <see langword="null"/>. / <paramref name="logDensity"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoint"/> is empty or non-finite, or its log-density is non-finite. / <paramref name="initialPoint"/> が空または非有限、あるいは初期点の対数密度が非有限の場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public SliceSamplingResult Sample(Func<ReadOnlyVectorSpan<double>, double> logDensity, ReadOnlySpan<double> initialPoint,
        CancellationToken cancellationToken = default) =>
        Sample(LogDensities.FromDelegate(logDensity), initialPoint, cancellationToken);

    /// <summary>
    /// Generates posterior draws starting each chain from its corresponding column of <paramref name="initialPoints"/>.
    /// <para>チェーンごとに指定した初期点から標本を生成します。</para>
    /// </summary>
    /// <param name="logDensity">The delegate evaluating the log-density. / 対数密度を評価するデリゲート。</param>
    /// <param name="initialPoints">The matrix of initial points with shape <c>[dimension × chains]</c>. / 各チェーンの初期点を列に並べた行列（形状 <c>[次元 × チェーン数]</c>）。</param>
    /// <param name="cancellationToken">A token checked at each transition to cancel sampling. / 遷移ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws and slice direction/width diagnostics. / ウォームアップ後の標本およびスライス方向・幅の情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="logDensity"/> is <see langword="null"/>. / <paramref name="logDensity"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoints"/> is empty, has a column count different from <see cref="Chains"/>, contains non-finite values, or has non-finite log-density. / <paramref name="initialPoints"/> が空または列数が <see cref="Chains"/> と異なる場合、非有限値を含む場合、あるいは初期点の対数密度が非有限の場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public SliceSamplingResult Sample(Func<ReadOnlyVectorSpan<double>, double> logDensity, ReadOnlyMatrixSpan<double> initialPoints,
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
    /// <returns>The post-warmup draws and slice direction/width diagnostics. / ウォームアップ後の標本およびスライス方向・幅の情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="density"/> is <see langword="null"/>. / <paramref name="density"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoint"/> is empty or non-finite, or its log-density is non-finite. / <paramref name="initialPoint"/> が空または非有限、あるいは初期点の対数密度が非有限の場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public SliceSamplingResult Sample<TDensity>(TDensity density, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)
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
    /// <returns>The post-warmup draws and slice direction/width diagnostics. / ウォームアップ後の標本およびスライス方向・幅の情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="density"/> is <see langword="null"/>. / <paramref name="density"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="initialPoints"/> is empty, has a column count different from <see cref="Chains"/>, contains non-finite values, or has non-finite log-density. / <paramref name="initialPoints"/> が空または列数が <see cref="Chains"/> と異なる場合、非有限値を含む場合、あるいは初期点の対数密度が非有限の場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public SliceSamplingResult Sample<TDensity>(TDensity density, ReadOnlyMatrixSpan<double> initialPoints, CancellationToken cancellationToken = default)
        where TDensity : ILogDensity =>
        Run(density, initialPoints.Rows, InitialPoints.Columns(initialPoints, chains), cancellationToken);

    private SliceSamplingResult Run<TDensity>(TDensity density, int d, double[] starts, CancellationToken cancellationToken)
        where TDensity : ILogDensity
    {
        ArgumentNullException.ThrowIfNull(density);
        settings.RequireStorage(d, chains);
        var draws = new Tensor<double>(d, settings.Draws, chains);
        var logDensities = new Matrix<double>(settings.Draws, chains);
        var directions = new Tensor<double>(d, d, chains);
        var widths = new Matrix<double>(d, chains);
        var evaluations = new long[chains];
        ChainRunner.Run(chains, settings.MaxDegreeOfParallelism, cancellationToken, c =>
        {
            var chain = new Chain<TDensity>(this, density, d, c);
            chain.Run(starts.AsSpan(c * d, d), draws.GetMatrix(c), logDensities.Column(c), cancellationToken);
            evaluations[c] = chain.Evaluations;
            chain.WriteDirections(directions.GetMatrix(c), widths.Column(c));
        });

        return new SliceSamplingResult(draws, logDensities, evaluations, settings.Seed, directions, widths);
    }

    private sealed class Chain<TDensity>(SliceSampler sampler, TDensity density, int d, int index)
        where TDensity : ILogDensity
    {
        private readonly Philox4x32Random random = RandomStreams.Create(sampler.settings.Seed, StreamPurpose.Transition, index);
        private readonly double[] current = new double[d];
        private readonly double[] trial = new double[d];
        private readonly double[] directions = Identity(d);   // 各列が 1 つの正規直交方向（列優先）
        private readonly double[] widths = Filled(d, sampler.initialWidth);
        private double currentLogDensity;

        public long Evaluations { get; private set; }

        public void WriteDirections(MatrixSpan<double> destination, VectorSpan<double> widthDestination)
        {
            for (int j = 0; j < d; j++)
            {
                widthDestination[j] = widths[j];
                for (int i = 0; i < d; i++)
                {
                    destination[i, j] = directions[j * d + i];
                }
            }
        }

        public void Run(ReadOnlySpan<double> start, MatrixSpan<double> draws, VectorSpan<double> logDensities, CancellationToken cancellationToken)
        {
            SamplerSettings settings = sampler.settings;
            start.CopyTo(current);
            currentLogDensity = Evaluate(current);
            InitialPoints.RequireValid(current, currentLogDensity, index);
            var schedule = new WarmupSchedule(settings.Warmup, sampler.windows);
            bool adapting = sampler.adaptWidths || sampler.adaptDirections;
            var estimator = new CovarianceEstimator(d, dense: true);
            var estimate = new double[d * d];
            for (int t = 0; t < settings.Warmup; t++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Sweep();
                if (adapting && schedule.InWindow(t))
                {
                    estimator.Add(current);
                }

                if (adapting && schedule.EndsWindow(t))
                {
                    estimator.WriteRegularized(estimate);
                    estimator.Reset();
                    Adapt(estimate);
                }
            }

            int kept = 0;
            for (long t = 0; t < settings.SamplingIterations; t++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Sweep();
                if ((t + 1) % settings.Thin == 0)
                {
                    current.CopyTo(draws.Column(kept).Data);
                    logDensities[kept] = currentLogDensity;
                    kept++;
                }
            }
        }

        private void Adapt(double[] covariance)
        {
            if (sampler.adaptDirections && d > 1)
            {
                var matrix = Matrix<double>.FromColumnMajor(d, d, covariance);
                SymmetricEigenDecomposition eigen;
                try
                {
                    eigen = SymmetricEigenDecomposition.Compute(matrix);
                }
                catch (ArithmeticException)
                {
                    return;
                }

                Matrix<double> vectors = eigen.Vectors!;
                for (int j = 0; j < d; j++)
                {
                    double value = eigen.Values[j];
                    if (!(value > 0 && double.IsFinite(value)))
                    {
                        return;
                    }
                }

                for (int j = 0; j < d; j++)
                {
                    for (int i = 0; i < d; i++)
                    {
                        directions[j * d + i] = vectors[i, j];
                    }

                    if (sampler.adaptWidths)
                    {
                        widths[j] = WidthPerStandardDeviation * Math.Sqrt(eigen.Values[j]);
                    }
                }

                return;
            }

            if (sampler.adaptWidths)
            {
                // 既存の方向 v に対して、その方向の標準偏差 √(vᵀΣv) に合わせて幅を更新する
                for (int j = 0; j < d; j++)
                {
                    double variance = 0;
                    for (int a = 0; a < d; a++)
                    {
                        for (int b = 0; b < d; b++)
                        {
                            variance += directions[j * d + a] * covariance[b * d + a] * directions[j * d + b];
                        }
                    }

                    if (variance > 0 && double.IsFinite(variance))
                    {
                        widths[j] = WidthPerStandardDeviation * Math.Sqrt(variance);
                    }
                }
            }
        }

        private void Sweep()
        {
            for (int j = 0; j < d; j++)
            {
                Update(directions.AsSpan(j * d, d), widths[j]);
            }
        }

        // 1 次元スライス更新（ステップアウトと縮小: Neal 2003, Figs. 3 & 5）
        private void Update(ReadOnlySpan<double> direction, double w)
        {
            double logHeight = currentLogDensity + Math.Log(random.NextDoubleOpen());
            double lower = -w * random.NextDouble();
            double upper = lower + w;
            int m = sampler.maxSteppingOut;
            int leftSteps = (int)Math.Floor(m * random.NextDouble());
            int rightSteps = m - 1 - leftSteps;
            while (leftSteps > 0 && At(direction, lower) > logHeight)
            {
                lower -= w;
                leftSteps--;
            }

            while (rightSteps > 0 && At(direction, upper) > logHeight)
            {
                upper += w;
                rightSteps--;
            }

            for (int k = 0; k < MaxShrinkage; k++)
            {
                double t = lower + random.NextDouble() * (upper - lower);
                double value = At(direction, t);
                if (value > logHeight)
                {
                    trial.CopyTo(current, 0);
                    currentLogDensity = value;
                    return;
                }

                if (t < 0)
                {
                    lower = t;
                }
                else
                {
                    upper = t;
                }
            }

            // 縮小が上限回数で収束しない場合は現在状態を維持する
        }

        private double At(ReadOnlySpan<double> direction, double t)
        {
            for (int i = 0; i < d; i++)
            {
                trial[i] = current[i] + t * direction[i];
            }

            return Evaluate(trial);
        }

        private double Evaluate(double[] point)
        {
            Evaluations++;
            return LogDensities.Sanitize(density.LogDensity(point));
        }

        private static double[] Identity(int n)
        {
            var result = new double[n * n];
            for (int i = 0; i < n; i++)
            {
                result[i * n + i] = 1;
            }

            return result;
        }

        private static double[] Filled(int n, double value)
        {
            var result = new double[n];
            Array.Fill(result, value);
            return result;
        }
    }
}

/// <summary>
/// Represents the draws and adapted slice geometry produced by <see cref="SliceSampler"/>.
/// <para><see cref="SliceSampler"/> のサンプリング結果と適応後のスライス方向・幅を表します。</para>
/// </summary>
public sealed class SliceSamplingResult : SamplingResult
{
    internal SliceSamplingResult(Tensor<double> draws, Matrix<double> logDensities, long[] evaluations, ulong seed,
        Tensor<double> directions, Matrix<double> widths)
        : base(draws, logDensities, evaluations, seed)
    {
        Directions = directions;
        Widths = widths;
    }

    /// <summary>
    /// Gets the post-warmup orthonormal slice update directions for each chain, with shape <c>[dimension, dimension, chain]</c> (each column is one unit direction).
    /// <para>チェーンごとのウォームアップ後に固定された正規直交スライス方向（形状 <c>[次元, 次元, チェーン]</c>、各列が 1 つの単位方向ベクトル）を取得します。</para>
    /// </summary>
    public Tensor<double> Directions { get; }

    /// <summary>
    /// Gets the post-warmup slice bracket widths along each direction for each chain, with shape <c>[dimension × chain]</c>.
    /// <para>チェーンごとのウォームアップ後に固定された各方向のスライス幅（形状 <c>[次元 × チェーン]</c>）を取得します。</para>
    /// </summary>
    public Matrix<double> Widths { get; }
}
