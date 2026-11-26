using Tuna.LinearAlgebra;
using Tuna.RandomSource;

namespace Tuna.Probabilistic.Inference;

/// <summary>
/// Configures <see cref="EllipticalSliceSampler"/>.
/// <para><see cref="EllipticalSliceSampler"/> の設定オプションを表します。</para>
/// </summary>
public sealed class EllipticalSliceSamplerOptions : MultiChainSamplerOptions
{
}

/// <summary>
/// Samples from a posterior with a multivariate Gaussian prior <c>π(θ) ∝ L(θ)·N(θ; μ, Σ)</c> using elliptical slice sampling.
/// <para>正規事前分布を持つ事後分布 <c>π(θ) ∝ L(θ)·N(θ; μ, Σ)</c> から、楕円スライスサンプラーによりサンプリングします。</para>
/// </summary>
/// <remarks>
/// <para>
/// Gradients are not required and discontinuous likelihoods are supported.
/// When initial points are omitted, each chain draws from the prior until a point with finite log-likelihood is found. Warmup draws are excluded from the result.
/// </para>
/// <para>
/// 尤度の勾配は不要であり、不連続な尤度にも適用できます。
/// 初期点を省略すると、各チェーンは対数尤度が有限となる点が得られるまで事前分布からサンプリングして初期化します。ウォームアップ期間の標本は結果に含まれません。
/// </para>
/// </remarks>
public sealed class EllipticalSliceSampler
{
    private const int MaxShrinkage = 10_000;
    private const int MaxInitializationAttempts = 1000;

    private readonly SamplerSettings settings;
    private readonly int chains;

    /// <summary>
    /// Initializes a new instance of <see cref="EllipticalSliceSampler"/> with the specified options.
    /// <para>指定した設定で <see cref="EllipticalSliceSampler"/> の新しいインスタンスを初期化します。</para>
    /// </summary>
    /// <param name="options">The sampler options, or <see langword="null"/> for defaults. / サンプラー設定（<see langword="null"/> の場合は既定値）。</param>
    /// <exception cref="ArgumentOutOfRangeException">An option value is out of its valid range. / 設定値が有効範囲外の場合。</exception>
    public EllipticalSliceSampler(EllipticalSliceSamplerOptions? options = null)
    {
        options ??= new EllipticalSliceSamplerOptions();
        settings = SamplerSettings.Validate(options);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.Chains, 1, nameof(options.Chains));
        chains = options.Chains;
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
    /// Generates posterior draws initializing each chain by sampling from the Gaussian prior.
    /// <para>各チェーンを事前分布からのサンプリングで初期化し、標本を生成します。</para>
    /// </summary>
    /// <param name="logLikelihood">The delegate evaluating the log-likelihood <c>ln L(θ)</c> up to an additive constant. / 対数尤度 <c>ln L(θ)</c>（正規化定数は不要）を評価するデリゲート。</param>
    /// <param name="priorMean">The Gaussian prior mean <c>μ</c> (length equals parameter dimension). / 事前分布の平均ベクトル <c>μ</c>（長さはパラメータ次元）。</param>
    /// <param name="priorCovariance">The Gaussian prior covariance <c>Σ</c> (<c>d × d</c> symmetric positive definite; only the lower triangular part is read). / 事前分布の共分散行列 <c>Σ</c>（<c>d × d</c> の対称正定値行列。下三角部分のみを参照）。</param>
    /// <param name="cancellationToken">A token checked at each transition to cancel sampling. / 遷移ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws and log-likelihood diagnostics. / ウォームアップ後の標本と対数尤度情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="logLikelihood"/> is <see langword="null"/>. / <paramref name="logLikelihood"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="priorMean"/> is empty or non-finite, <paramref name="priorCovariance"/> has mismatched dimensions or is not positive definite, or no finite log-likelihood is found from prior draws. / 平均が空または非有限、共分散行列の次元不一致または非正定値、あるいは事前分布から有限な対数尤度の点が得られない場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public EllipticalSliceSamplingResult Sample(Func<ReadOnlyVectorSpan<double>, double> logLikelihood, ReadOnlySpan<double> priorMean,
        ReadOnlyMatrixSpan<double> priorCovariance, CancellationToken cancellationToken = default) =>
        Sample(LogDensities.FromDelegate(logLikelihood), priorMean, priorCovariance, cancellationToken);

    /// <summary>
    /// Generates posterior draws starting each chain from its corresponding column of <paramref name="initialPoints"/>.
    /// <para>チェーンごとに指定した初期点から標本を生成します。</para>
    /// </summary>
    /// <param name="logLikelihood">The delegate evaluating the log-likelihood <c>ln L(θ)</c> up to an additive constant. / 対数尤度 <c>ln L(θ)</c>（正規化定数は不要）を評価するデリゲート。</param>
    /// <param name="priorMean">The Gaussian prior mean <c>μ</c> (length equals parameter dimension). / 事前分布の平均ベクトル <c>μ</c>（長さはパラメータ次元）。</param>
    /// <param name="priorCovariance">The Gaussian prior covariance <c>Σ</c> (<c>d × d</c> symmetric positive definite; only the lower triangular part is read). / 事前分布の共分散行列 <c>Σ</c>（<c>d × d</c> の対称正定値行列。下三角部分のみを参照）。</param>
    /// <param name="initialPoints">The matrix of initial points with shape <c>[dimension × chains]</c>. / 各チェーンの初期点を列に並べた行列（形状 <c>[次元 × チェーン数]</c>）。</param>
    /// <param name="cancellationToken">A token checked at each transition to cancel sampling. / 遷移ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws and log-likelihood diagnostics. / ウォームアップ後の標本と対数尤度情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="logLikelihood"/> is <see langword="null"/>. / <paramref name="logLikelihood"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="priorMean"/> is empty or non-finite, <paramref name="priorCovariance"/> or <paramref name="initialPoints"/> has mismatched dimensions, <paramref name="priorCovariance"/> is not positive definite, or any initial point is non-finite or has non-finite log-likelihood. / 平均が空または非有限、共分散行列や初期点の次元不一致、共分散行列が非正定値、または初期点やその対数尤度が非有限の場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public EllipticalSliceSamplingResult Sample(Func<ReadOnlyVectorSpan<double>, double> logLikelihood, ReadOnlySpan<double> priorMean,
        ReadOnlyMatrixSpan<double> priorCovariance, ReadOnlyMatrixSpan<double> initialPoints, CancellationToken cancellationToken = default) =>
        Sample(LogDensities.FromDelegate(logLikelihood), priorMean, priorCovariance, initialPoints, cancellationToken);

    /// <summary>
    /// Generates posterior draws initializing each chain by sampling from the Gaussian prior.
    /// <para>各チェーンを事前分布からのサンプリングで初期化し、標本を生成します。</para>
    /// </summary>
    /// <typeparam name="TLikelihood">The log-likelihood type. / 対数尤度の型。</typeparam>
    /// <param name="logLikelihood">The target log-likelihood <c>ln L(θ)</c> up to an additive constant. / 対数尤度 <c>ln L(θ)</c>（正規化定数は不要）。</param>
    /// <param name="priorMean">The Gaussian prior mean <c>μ</c> (length equals parameter dimension). / 事前分布の平均ベクトル <c>μ</c>（長さはパラメータ次元）。</param>
    /// <param name="priorCovariance">The Gaussian prior covariance <c>Σ</c> (<c>d × d</c> symmetric positive definite; only the lower triangular part is read). / 事前分布の共分散行列 <c>Σ</c>（<c>d × d</c> の対称正定値行列。下三角部分のみを参照）。</param>
    /// <param name="cancellationToken">A token checked at each transition to cancel sampling. / 遷移ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws and log-likelihood diagnostics. / ウォームアップ後の標本と対数尤度情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="logLikelihood"/> is <see langword="null"/>. / <paramref name="logLikelihood"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="priorMean"/> is empty or non-finite, <paramref name="priorCovariance"/> has mismatched dimensions or is not positive definite, or no finite log-likelihood is found from prior draws. / 平均が空または非有限、共分散行列の次元不一致または非正定値、あるいは事前分布から有限な対数尤度の点が得られない場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public EllipticalSliceSamplingResult Sample<TLikelihood>(TLikelihood logLikelihood, ReadOnlySpan<double> priorMean,
        ReadOnlyMatrixSpan<double> priorCovariance, CancellationToken cancellationToken = default)
        where TLikelihood : ILogDensity
    {
        ArgumentNullException.ThrowIfNull(logLikelihood);
        Prior prior = Prior.Create(priorMean, priorCovariance);
        return Run(logLikelihood, prior, starts: null, cancellationToken);
    }

    /// <summary>
    /// Generates posterior draws starting each chain from its corresponding column of <paramref name="initialPoints"/>.
    /// <para>チェーンごとに指定した初期点から標本を生成します。</para>
    /// </summary>
    /// <typeparam name="TLikelihood">The log-likelihood type. / 対数尤度の型。</typeparam>
    /// <param name="logLikelihood">The target log-likelihood <c>ln L(θ)</c> up to an additive constant. / 対数尤度 <c>ln L(θ)</c>（正規化定数は不要）。</param>
    /// <param name="priorMean">The Gaussian prior mean <c>μ</c> (length equals parameter dimension). / 事前分布の平均ベクトル <c>μ</c>（長さはパラメータ次元）。</param>
    /// <param name="priorCovariance">The Gaussian prior covariance <c>Σ</c> (<c>d × d</c> symmetric positive definite; only the lower triangular part is read). / 事前分布の共分散行列 <c>Σ</c>（<c>d × d</c> の対称正定値行列。下三角部分のみを参照）。</param>
    /// <param name="initialPoints">The matrix of initial points with shape <c>[dimension × chains]</c>. / 各チェーンの初期点を列に並べた行列（形状 <c>[次元 × チェーン数]</c>）。</param>
    /// <param name="cancellationToken">A token checked at each transition to cancel sampling. / 遷移ごとに確認するキャンセル用トークン。</param>
    /// <returns>The post-warmup draws and log-likelihood diagnostics. / ウォームアップ後の標本と対数尤度情報。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="logLikelihood"/> is <see langword="null"/>. / <paramref name="logLikelihood"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="priorMean"/> is empty or non-finite, <paramref name="priorCovariance"/> or <paramref name="initialPoints"/> has mismatched dimensions, <paramref name="priorCovariance"/> is not positive definite, or any initial point is non-finite or has non-finite log-likelihood. / 平均が空または非有限、共分散行列や初期点の次元不一致、共分散行列が非正定値、または初期点やその対数尤度が非有限の場合。</exception>
    /// <exception cref="OperationCanceledException">Sampling was canceled via <paramref name="cancellationToken"/>. / <paramref name="cancellationToken"/> によりキャンセルされた場合。</exception>
    public EllipticalSliceSamplingResult Sample<TLikelihood>(TLikelihood logLikelihood, ReadOnlySpan<double> priorMean,
        ReadOnlyMatrixSpan<double> priorCovariance, ReadOnlyMatrixSpan<double> initialPoints, CancellationToken cancellationToken = default)
        where TLikelihood : ILogDensity
    {
        ArgumentNullException.ThrowIfNull(logLikelihood);
        Prior prior = Prior.Create(priorMean, priorCovariance);
        double[] starts = InitialPoints.Columns(initialPoints, chains);
        if (initialPoints.Rows != prior.Dimension)
        {
            throw new ArgumentException($"The initial points must have {prior.Dimension} rows.", nameof(initialPoints));
        }

        return Run(logLikelihood, prior, starts, cancellationToken);
    }

    private EllipticalSliceSamplingResult Run<TLikelihood>(TLikelihood logLikelihood, Prior prior, double[]? starts, CancellationToken cancellationToken)
        where TLikelihood : ILogDensity
    {
        int d = prior.Dimension;
        settings.RequireStorage(d, chains);
        var draws = new Tensor<double>(d, settings.Draws, chains);
        var logDensities = new Matrix<double>(settings.Draws, chains);
        var logLikelihoods = new Matrix<double>(settings.Draws, chains);
        var evaluations = new long[chains];
        ChainRunner.Run(chains, settings.MaxDegreeOfParallelism, cancellationToken, c =>
        {
            var chain = new Chain<TLikelihood>(settings, logLikelihood, prior, c);
            if (starts is null)
            {
                chain.InitializeFromPrior();
            }
            else
            {
                chain.Initialize(starts.AsSpan(c * d, d));
            }

            chain.Run(draws.GetMatrix(c), logDensities.Column(c), logLikelihoods.Column(c), cancellationToken);
            evaluations[c] = chain.Evaluations;
        });

        return new EllipticalSliceSamplingResult(draws, logDensities, evaluations, settings.Seed, logLikelihoods);
    }

    private sealed class Prior(double[] mean, CovarianceShape shape)
    {
        public int Dimension => mean.Length;

        public ReadOnlySpan<double> Mean => mean;

        public CovarianceShape Shape => shape;

        public static Prior Create(ReadOnlySpan<double> mean, ReadOnlyMatrixSpan<double> covariance)
        {
            if (mean.IsEmpty)
            {
                throw new ArgumentException("The prior mean must not be empty.", "priorMean");
            }

            foreach (double v in mean)
            {
                if (!double.IsFinite(v))
                {
                    throw new ArgumentException("The prior mean must be finite.", "priorMean");
                }
            }

            int d = mean.Length;
            if (covariance.Rows != d || covariance.Columns != d)
            {
                throw new ArgumentException($"The prior covariance must be {d} x {d}.", "priorCovariance");
            }

            var values = new double[d * d];
            for (int j = 0; j < d; j++)
            {
                for (int i = 0; i < d; i++)
                {
                    // 下三角部分を対称にコピーする
                    values[j * d + i] = i >= j ? covariance[i, j] : covariance[j, i];
                }
            }

            var shape = new CovarianceShape(d, dense: true);
            if (!shape.TrySet(values))
            {
                throw new ArgumentException("The prior covariance must be symmetric positive definite.", "priorCovariance");
            }

            return new Prior(mean.ToArray(), shape);
        }

        // 正規化定数を除いた事前対数密度 −½(θ − μ)ᵀΣ⁻¹(θ − μ)
        public double LogDensity(ReadOnlySpan<double> theta, Span<double> work)
        {
            for (int i = 0; i < mean.Length; i++)
            {
                work[i] = theta[i] - mean[i];
            }

            shape.SolveFactor(work, work);
            double sum = 0;
            foreach (double v in work)
            {
                sum += v * v;
            }

            return -0.5 * sum;
        }
    }

    private sealed class Chain<TLikelihood>(SamplerSettings settings, TLikelihood logLikelihood, Prior prior, int index)
        where TLikelihood : ILogDensity
    {
        private readonly Philox4x32Random random = RandomStreams.Create(settings.Seed, StreamPurpose.Transition, index);
        private readonly int d = prior.Dimension;
        private readonly double[] current = new double[prior.Dimension];
        private readonly double[] trial = new double[prior.Dimension];
        private readonly double[] normal = new double[prior.Dimension];
        private readonly double[] nu = new double[prior.Dimension];
        private double currentLogLikelihood;

        public long Evaluations { get; private set; }

        public void Initialize(ReadOnlySpan<double> start)
        {
            start.CopyTo(current);
            currentLogLikelihood = Evaluate(current);
            InitialPoints.RequireValid(current, currentLogLikelihood, index);
        }

        public void InitializeFromPrior()
        {
            Philox4x32Random initialization = RandomStreams.Create(settings.Seed, StreamPurpose.Initialization, index);
            for (int attempt = 0; attempt < MaxInitializationAttempts; attempt++)
            {
                Gaussian.Fill(initialization, normal);
                prior.Shape.MultiplyFactor(normal, current);
                for (int i = 0; i < d; i++)
                {
                    current[i] += prior.Mean[i];
                }

                currentLogLikelihood = Evaluate(current);
                if (double.IsFinite(currentLogLikelihood))
                {
                    return;
                }
            }

            throw new ArgumentException(
                $"The log likelihood was not finite at {MaxInitializationAttempts} points drawn from the prior (chain {index}).", "logLikelihood");
        }

        public void Run(MatrixSpan<double> draws, VectorSpan<double> logDensities, VectorSpan<double> logLikelihoods, CancellationToken cancellationToken)
        {
            for (int t = 0; t < settings.Warmup; t++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Transition();
            }

            int kept = 0;
            for (long t = 0; t < settings.SamplingIterations; t++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Transition();
                if ((t + 1) % settings.Thin == 0)
                {
                    current.CopyTo(draws.Column(kept).Data);
                    logLikelihoods[kept] = currentLogLikelihood;
                    logDensities[kept] = currentLogLikelihood + prior.LogDensity(current, trial);
                    kept++;
                }
            }
        }

        // Murray, Adams & MacKay (2010), Fig. 2
        private void Transition()
        {
            Gaussian.Fill(random, normal);
            prior.Shape.MultiplyFactor(normal, nu);
            double logHeight = currentLogLikelihood + Math.Log(random.NextDoubleOpen());
            double angle = 2 * Math.PI * random.NextDouble();
            double lower = angle - 2 * Math.PI, upper = angle;
            ReadOnlySpan<double> mean = prior.Mean;
            for (int k = 0; k < MaxShrinkage; k++)
            {
                double cos = Math.Cos(angle), sin = Math.Sin(angle);
                for (int i = 0; i < d; i++)
                {
                    trial[i] = mean[i] + (current[i] - mean[i]) * cos + nu[i] * sin;
                }

                double value = Evaluate(trial);
                if (value > logHeight)
                {
                    trial.CopyTo(current, 0);
                    currentLogLikelihood = value;
                    return;
                }

                if (angle < 0)
                {
                    lower = angle;
                }
                else
                {
                    upper = angle;
                }

                angle = lower + random.NextDouble() * (upper - lower);
            }

            // 縮小が上限回数で収束しない場合は現在状態を維持する
        }

        private double Evaluate(double[] point)
        {
            Evaluations++;
            return LogDensities.Sanitize(logLikelihood.LogDensity(point));
        }
    }
}

/// <summary>
/// Represents the draws and log-likelihood values produced by <see cref="EllipticalSliceSampler"/>.
/// <para><see cref="EllipticalSliceSampler"/> のサンプリング結果と対数尤度を表します。</para>
/// </summary>
/// <remarks>
/// <para><see cref="SamplingResult.LogDensities"/> stores the sum of the log-likelihood and the unnormalized Gaussian prior log-density <c>−½(θ − μ)ᵀΣ⁻¹(θ − μ)</c>.</para>
/// <para><see cref="SamplingResult.LogDensities"/> には、対数尤度と正規化定数を除いた正規事前分布の対数密度 <c>−½(θ − μ)ᵀΣ⁻¹(θ − μ)</c> の和が格納されます。</para>
/// </remarks>
public sealed class EllipticalSliceSamplingResult : SamplingResult
{
    internal EllipticalSliceSamplingResult(Tensor<double> draws, Matrix<double> logDensities, long[] evaluations, ulong seed, Matrix<double> logLikelihoods)
        : base(draws, logDensities, evaluations, seed)
    {
        LogLikelihoods = logLikelihoods;
    }

    /// <summary>
    /// Gets the log-likelihood evaluated at each retained draw, with shape <c>[draw × chain]</c>.
    /// <para>保持された各標本における対数尤度行列（形状 <c>[反復 × チェーン]</c>）を取得します。</para>
    /// </summary>
    public Matrix<double> LogLikelihoods { get; }
}
