using Tuna.LinearAlgebra;
using Tuna.RandomSource;
using Tuna.Statistics.Distributions;

namespace Tuna.Probabilistic.Inference;

/// <summary>
/// Specifies the structure of the inverse mass matrix (metric) <c>M⁻¹</c> in Hamiltonian samplers.
/// <para>ハミルトニアン法における逆計量（運動量共分散の逆行列 <c>M⁻¹</c>）の構造を指定します。</para>
/// </summary>
public enum HamiltonianMetric
{
    /// <summary>
    /// Diagonal inverse metric, adapting per-parameter scales only (default).
    /// <para>対角逆計量。パラメータごとのスケールのみを適応させます（既定）。</para>
    /// </summary>
    Diagonal,

    /// <summary>
    /// Dense inverse metric, capturing linear correlations across parameters using Cholesky factorization at <c>O(d²)</c> cost per leapfrog step.
    /// <para>密逆計量。コレスキー分解を用いてパラメータ間の線形相関も取り除きます（1 リープフロッグ当たり <c>O(d²)</c>）。</para>
    /// </summary>
    Dense,
}

/// <summary>
/// Provides common configuration options for <see cref="NoUTurnSampler"/> and <see cref="HamiltonianMonteCarlo"/>.
/// <para><see cref="NoUTurnSampler"/> および <see cref="HamiltonianMonteCarlo"/> に共通の設定オプションを提供します。</para>
/// </summary>
public abstract class HamiltonianSamplerOptions : MultiChainSamplerOptions
{
    /// <summary>
    /// Gets or initializes the target mean acceptance statistic during warmup (in <c>(0, 1)</c>). Defaults to 0.8 for <see cref="NoUTurnSampler"/> and 0.65 for <see cref="HamiltonianMonteCarlo"/>.
    /// <para>ウォームアップ期間中の目標平均受理統計量を取得または設定します（<c>(0, 1)</c>）。既定値は <see cref="NoUTurnSampler"/> で 0.8、<see cref="HamiltonianMonteCarlo"/> で 0.65 です。</para>
    /// </summary>
    /// <remarks>
    /// <para>Raising this value (for example, to 0.9–0.99) yields a smaller step size and helps eliminate divergent transitions.</para>
    /// <para>発散が発生する場合は 0.9〜0.99 程度へ引き上げるとステップ幅が小さくなり、発散の解消に有効です。</para>
    /// </remarks>
    public double? TargetAcceptance { get; init; }

    /// <summary>
    /// Gets or initializes the initial leapfrog step size.
    /// <para>リープフロッグ法の初期ステップ幅を取得または設定します。</para>
    /// </summary>
    /// <value>A positive finite value, or <see langword="null"/> (default) to automatically search for a step size whose single-step acceptance crosses 0.5. Used unchanged when <see cref="SamplerOptions.Warmup"/> is 0. / 正の有限値、または <see langword="null"/>（既定。1 回のリープフロッグ受理確率が 0.5 をまたぐ値を自動探索します）。ウォームアップなしの場合はこの値がそのまま使用されます。</value>
    public double? InitialStepSize { get; init; }

    /// <summary>
    /// Gets or initializes the structure of the inverse metric (default is <see cref="HamiltonianMetric.Diagonal"/>).
    /// <para>逆計量の構造を取得または設定します（既定値 <see cref="HamiltonianMetric.Diagonal"/>）。</para>
    /// </summary>
    public HamiltonianMetric Metric { get; init; } = HamiltonianMetric.Diagonal;

    /// <summary>
    /// Gets or initializes the initial inverse metric <c>M⁻¹</c> (<c>d × d</c> symmetric positive definite matrix; only diagonal entries are used for <see cref="HamiltonianMetric.Diagonal"/>). Defaults to the identity matrix when <see langword="null"/>.
    /// <para>逆計量 <c>M⁻¹</c> の初期値を取得または設定します（<c>d × d</c> の対称正定値行列。<see cref="HamiltonianMetric.Diagonal"/> では対角成分のみ使用）。<see langword="null"/>（既定）の場合は単位行列になります。</para>
    /// </summary>
    /// <remarks>
    /// <para>Sampling efficiency improves when <see cref="InitialInverseMetric"/> approximates the target posterior covariance.</para>
    /// <para>目標分布の共分散に近い行列を指定するほどサンプリング効率が向上します。</para>
    /// </remarks>
    public Matrix<double>? InitialInverseMetric { get; init; }

    /// <summary>
    /// Gets or initializes a value indicating whether the inverse metric is updated from regularized sample covariance at the end of each warmup window (default is <see langword="true"/>).
    /// <para>ウォームアップの各ウィンドウ終端で、正則化した標本共分散から逆計量を更新するかどうかを取得または設定します（既定値 <see langword="true"/>）。</para>
    /// </summary>
    public bool AdaptMetric { get; init; } = true;

    /// <summary>
    /// Gets or initializes the warmup adaptation window schedule.
    /// <para>ウォームアップ期間の適応ウィンドウ設定を取得または設定します。</para>
    /// </summary>
    public WarmupWindowOptions Windows { get; init; } = new();

    /// <summary>
    /// Gets or initializes the Hamiltonian increase threshold above which a trajectory is terminated and recorded as divergent (positive; default is 1000).
    /// <para>軌道を発散と判定するハミルトニアン増加量の閾値を取得または設定します（正の値、既定値 1000）。この値を超えた時点で軌道を打ち切り、発散として記録します。</para>
    /// </summary>
    public double DivergenceThreshold { get; init; } = 1000;
}

/// <summary>
/// Represents the draws and Hamiltonian diagnostics produced by <see cref="NoUTurnSampler"/> or <see cref="HamiltonianMonteCarlo"/>.
/// <para><see cref="NoUTurnSampler"/> または <see cref="HamiltonianMonteCarlo"/> のサンプリング結果と診断情報を表します。</para>
/// </summary>
public sealed class HamiltonianSamplingResult : SamplingResult
{
    private readonly double[] stepSizes;

    internal HamiltonianSamplingResult(Tensor<double> draws, Matrix<double> logDensities, long[] evaluations, ulong seed,
        double[] stepSizes, Tensor<double> inverseMetrics, Matrix<double> acceptanceStatistics, Matrix<double> energies,
        Matrix<int> treeDepths, Matrix<int> leapfrogSteps, Matrix<int> divergences, int maxTreeDepth)
        : base(draws, logDensities, evaluations, seed)
    {
        this.stepSizes = stepSizes;
        InverseMetrics = inverseMetrics;
        AcceptanceStatistics = acceptanceStatistics;
        Energies = energies;
        TreeDepths = treeDepths;
        LeapfrogSteps = leapfrogSteps;
        Divergences = divergences;
        MaxTreeDepth = maxTreeDepth;
    }

    /// <summary>
    /// Gets the fixed post-warmup step size for each chain.
    /// <para>チェーンごとのウォームアップ後に固定されたステップ幅を取得します。</para>
    /// </summary>
    public IReadOnlyList<double> StepSizes => stepSizes;

    /// <summary>
    /// Gets the post-warmup inverse metric <c>M⁻¹</c> for each chain with shape <c>[dimension, dimension, chain]</c> (diagonal matrix when using <see cref="HamiltonianMetric.Diagonal"/>).
    /// <para>チェーンごとのウォームアップ後の逆計量 <c>M⁻¹</c>（形状 <c>[次元, 次元, チェーン]</c>、対角計量では対角行列）を取得します。</para>
    /// </summary>
    /// <remarks>
    /// <para>Individual chain slices can be passed back to <see cref="HamiltonianSamplerOptions.InitialInverseMetric"/> for subsequent runs.</para>
    /// <para>各チェーンの行列は、再実行時の <see cref="HamiltonianSamplerOptions.InitialInverseMetric"/> として再利用できます。</para>
    /// </remarks>
    public Tensor<double> InverseMetrics { get; }

    /// <summary>
    /// Gets the acceptance statistic for each retained draw with shape <c>[draw × chain]</c> (for NUTS, the mean of <c>min(1, e^{H₀ − H})</c> along the trajectory).
    /// <para>保持された各標本の遷移における受理統計量（形状 <c>[反復 × チェーン]</c>、NUTS では軌道上の点の <c>min(1, e^{H₀ − H})</c> の平均）を取得します。</para>
    /// </summary>
    public Matrix<double> AcceptanceStatistics { get; }

    /// <summary>
    /// Gets the Hamiltonian (total energy) <c>−ln π(θ) + ½pᵀM⁻¹p</c> at each retained draw with shape <c>[draw × chain]</c>.
    /// <para>保持された各標本で選択された点のハミルトニアン（全エネルギー）<c>−ln π(θ) + ½pᵀM⁻¹p</c>（形状 <c>[反復 × チェーン]</c>）を取得します。</para>
    /// </summary>
    public Matrix<double> Energies { get; }

    /// <summary>
    /// Gets the binary tree depth for each retained draw with shape <c>[draw × chain]</c> (0 for static <see cref="HamiltonianMonteCarlo"/>).
    /// <para>保持された各標本の遷移における木の深さ（形状 <c>[反復 × チェーン]</c>、静的 <see cref="HamiltonianMonteCarlo"/> では 0）を取得します。</para>
    /// </summary>
    public Matrix<int> TreeDepths { get; }

    /// <summary>
    /// Gets the number of leapfrog steps (equal to gradient evaluations) for each retained draw with shape <c>[draw × chain]</c>.
    /// <para>保持された各標本の遷移におけるリープフロッグ回数（勾配評価回数と等しい、形状 <c>[反復 × チェーン]</c>）を取得します。</para>
    /// </summary>
    public Matrix<int> LeapfrogSteps { get; }

    /// <summary>
    /// Gets the divergence indicator matrix with shape <c>[draw × chain]</c> (1 indicates a divergent transition, 0 otherwise).
    /// <para>保持された各標本の遷移で発散が発生したかを示す行列（形状 <c>[反復 × チェーン]</c>、1 が発散）を取得します。</para>
    /// </summary>
    public Matrix<int> Divergences { get; }

    /// <summary>
    /// Gets the configured maximum tree depth (0 for static <see cref="HamiltonianMonteCarlo"/>).
    /// <para>設定された木の深さの上限（静的 <see cref="HamiltonianMonteCarlo"/> では 0）を取得します。</para>
    /// </summary>
    public int MaxTreeDepth { get; }

    /// <summary>
    /// Gets the total number of divergent post-warmup transitions across all chains.
    /// <para>全チェーン合計のウォームアップ後発散遷移回数を取得します。</para>
    /// </summary>
    /// <remarks>
    /// <para>Any non-zero value indicates high-curvature regions may not have been explored reliably; consider raising <see cref="HamiltonianSamplerOptions.TargetAcceptance"/> or reparameterizing the model.</para>
    /// <para>0 でない場合は曲率の高い領域を正しく探索できておらず標本が偏るおそれがあります。<see cref="HamiltonianSamplerOptions.TargetAcceptance"/> を引き上げるか、モデルを再パラメータ化してください。</para>
    /// </remarks>
    public int DivergenceCount
    {
        get
        {
            int count = 0;
            foreach (int v in Divergences.AsSpan())
            {
                count += v;
            }

            return count;
        }
    }

    /// <summary>
    /// Gets the total number of post-warmup transitions that reached <see cref="MaxTreeDepth"/> (NUTS only).
    /// <para>ウォームアップ後に木の深さが <see cref="MaxTreeDepth"/> に到達した遷移の総数を取得します（NUTS のみ）。</para>
    /// </summary>
    /// <remarks>
    /// <para>Frequent hits indicate trajectories are being truncated early, reducing sampling efficiency.</para>
    /// <para>到達回数が多い場合は軌道が途中で打ち切られており、サンプリング効率が低下しています。</para>
    /// </remarks>
    public int MaxTreeDepthHits
    {
        get
        {
            if (MaxTreeDepth == 0)
            {
                return 0;
            }

            int count = 0;
            foreach (int v in TreeDepths.AsSpan())
            {
                count += v >= MaxTreeDepth ? 1 : 0;
            }

            return count;
        }
    }

    /// <summary>
    /// Computes the Energy Bayesian Fraction of Missing Information (E-BFMI) for the specified chain.
    /// <para>指定したチェーンの E-BFMI（エネルギーのベイズ的欠損情報割合）を計算します。</para>
    /// </summary>
    /// <param name="chain">The zero-based chain index. / 0 から始まるチェーンのインデックス。</param>
    /// <returns>The E-BFMI statistic for <paramref name="chain"/>, or <see cref="double.NaN"/> if fewer than 2 draws exist. / 指定チェーンの E-BFMI。標本数が 2 未満の場合は <see cref="double.NaN"/>。</returns>
    /// <remarks>
    /// <para>Values below 0.3 indicate that momentum resampling alone is not exploring the energy distribution tails efficiently.</para>
    /// <para>0.3 を下回る場合は、運動量の再サンプリングだけではエネルギー分布の裾を十分に探索できていない目安になります。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="chain"/> is out of range. / <paramref name="chain"/> が範囲外の場合。</exception>
    public double EnergyBayesianFractionOfMissingInformation(int chain)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(chain);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(chain, ChainCount);
        return ConvergenceDiagnostics.EnergyFractionOfMissingInformation(Energies.Column(chain));
    }
}

// 位相空間の状態（位置・運動量・勾配・速度 M⁻¹p・対数密度）
internal sealed class Phase(int dimension)
{
    public readonly double[] Theta = new double[dimension];
    public readonly double[] Momentum = new double[dimension];
    public readonly double[] Gradient = new double[dimension];
    public readonly double[] Velocity = new double[dimension];
    public double LogDensity;

    public void CopyFrom(Phase other)
    {
        other.Theta.CopyTo(Theta, 0);
        other.Momentum.CopyTo(Momentum, 0);
        other.Gradient.CopyTo(Gradient, 0);
        other.Velocity.CopyTo(Velocity, 0);
        LogDensity = other.LogDensity;
    }

    public void CopyPositionFrom(Phase other)
    {
        other.Theta.CopyTo(Theta, 0);
        other.Gradient.CopyTo(Gradient, 0);
        LogDensity = other.LogDensity;
    }

    // 運動エネルギー ½pᵀM⁻¹p（Velocity が最新であること）
    public double Kinetic()
    {
        double sum = 0;
        for (int i = 0; i < Momentum.Length; i++)
        {
            sum += Momentum[i] * Velocity[i];
        }

        return 0.5 * sum;
    }

    public double Hamiltonian() => -LogDensity + Kinetic();
}

internal abstract class HamiltonianChain<TDensity> where TDensity : IDifferentiableLogDensity
{
    private readonly TDensity density;
    private readonly HamiltonianSettings settings;
    private readonly double[] normal;
    protected readonly int D;
    protected readonly Philox4x32Random Random;
    protected readonly CovarianceShape Metric;
    protected readonly Phase Current;
    protected readonly Phase Trial;

    protected HamiltonianChain(TDensity density, int dimension, int chain, HamiltonianSettings settings)
    {
        this.density = density;
        this.settings = settings;
        D = dimension;
        Random = RandomStreams.Create(settings.Common.Seed, StreamPurpose.Transition, chain);
        Metric = new CovarianceShape(dimension, settings.Metric == HamiltonianMetric.Dense);
        Current = new Phase(dimension);
        Trial = new Phase(dimension);
        normal = new double[dimension];
        ChainIndex = chain;
    }

    protected int ChainIndex { get; }

    protected double DivergenceThreshold => settings.DivergenceThreshold;

    public long Evaluations { get; private set; }

    public double StepSize { get; private set; }

    protected double LastAcceptance;
    protected double LastEnergy;
    protected int LastDepth;
    protected int LastLeapfrogs;
    protected bool LastDivergent;

    protected abstract void Transition(double epsilon);

    public void WriteInverseMetric(MatrixSpan<double> destination)
    {
        var dense = new double[D * D];
        Metric.WriteDense(dense);
        for (int j = 0; j < D; j++)
        {
            for (int i = 0; i < D; i++)
            {
                destination[i, j] = dense[j * D + i];
            }
        }
    }

    public void Run(ReadOnlySpan<double> start, HamiltonianStorage storage, CancellationToken cancellationToken)
    {
        SamplerSettings common = settings.Common;
        if (settings.InitialInverseMetric is { } initial)
        {
            var values = new double[Metric.IsDense ? D * D : D];
            for (int j = 0; j < D; j++)
            {
                if (Metric.IsDense)
                {
                    for (int i = 0; i < D; i++)
                    {
                        values[j * D + i] = initial[i, j];
                    }
                }
                else
                {
                    values[j] = initial[j, j];
                }
            }

            if (!Metric.TrySet(values))
            {
                throw new ArgumentException("The initial inverse metric must be symmetric positive definite.", "options");
            }
        }

        start.CopyTo(Current.Theta);
        try
        {
            Current.LogDensity = Evaluate(Current.Theta, Current.Gradient);
        }
        catch (DistributionParameterOutOfRangeException e)
        {
            throw new ArgumentException($"The initial point of chain {ChainIndex} produced an out-of-domain distribution parameter.", "initialPoints", e);
        }

        InitialPoints.RequireValid(Current.Theta, Current.LogDensity, ChainIndex);
        double epsilon = settings.InitialStepSize ?? FindReasonableStepSize(1.0);
        var schedule = new WarmupSchedule(common.Warmup, settings.Windows);
        var adaptation = new DualAveraging(settings.TargetAcceptance);
        adaptation.Restart(epsilon);
        var estimator = new CovarianceEstimator(D, Metric.IsDense);
        var estimate = new double[Metric.IsDense ? D * D : D];
        for (int t = 0; t < common.Warmup; t++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Transition(epsilon);
            epsilon = adaptation.Update(LastAcceptance);
            if (settings.AdaptMetric && schedule.InWindow(t))
            {
                estimator.Add(Current.Theta);
            }

            if (settings.AdaptMetric && schedule.EndsWindow(t))
            {
                estimator.WriteRegularized(estimate);
                estimator.Reset();
                Metric.TrySet(estimate);
                epsilon = FindReasonableStepSize(epsilon);
                adaptation.Restart(epsilon);
            }
        }

        if (common.Warmup > 0 && double.IsFinite(adaptation.Averaged))
        {
            epsilon = adaptation.Averaged;
        }

        StepSize = epsilon;
        int kept = 0;
        for (long t = 0; t < common.SamplingIterations; t++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Transition(epsilon);
            if ((t + 1) % common.Thin == 0)
            {
                Current.Theta.CopyTo(storage.Draws.Column(kept).Data);
                storage.LogDensities[kept] = Current.LogDensity;
                storage.Acceptance[kept] = LastAcceptance;
                storage.Energies[kept] = LastEnergy;
                storage.Depths[kept] = LastDepth;
                storage.Leapfrogs[kept] = LastLeapfrogs;
                storage.Divergences[kept] = LastDivergent ? 1 : 0;
                kept++;
            }
        }
    }

    // 運動量 p ~ N(0, M) と速度 M⁻¹p を生成する
    protected void SampleMomentum(Phase z)
    {
        Gaussian.Fill(Random, normal);
        Metric.SolveFactorTranspose(normal, z.Momentum);
        Metric.Multiply(z.Momentum, z.Velocity);
    }

    // 符号付きステップ幅 epsilon で 1 回のリープフロッグ積分を行う（非有限の密度または勾配では LogDensity = -∞）
    protected void Leapfrog(Phase z, double epsilon)
    {
        double half = 0.5 * epsilon;
        for (int i = 0; i < D; i++)
        {
            z.Momentum[i] += half * z.Gradient[i];
        }

        Metric.Multiply(z.Momentum, z.Velocity);
        for (int i = 0; i < D; i++)
        {
            z.Theta[i] += epsilon * z.Velocity[i];
        }

        try
        {
            z.LogDensity = Evaluate(z.Theta, z.Gradient);
        }
        catch (DistributionParameterOutOfRangeException)
        {
            z.LogDensity = double.NegativeInfinity;
        }

        if (z.LogDensity == double.NegativeInfinity)
        {
            return;
        }

        for (int i = 0; i < D; i++)
        {
            z.Momentum[i] += half * z.Gradient[i];
        }

        Metric.Multiply(z.Momentum, z.Velocity);
    }

    private double Evaluate(double[] theta, double[] gradient)
    {
        Evaluations++;
        Array.Clear(gradient);
        double value = LogDensities.Sanitize(density.LogDensity(theta, gradient));
        if (value == double.NegativeInfinity)
        {
            return value;
        }

        foreach (double g in gradient)
        {
            if (!double.IsFinite(g))
            {
                return double.NegativeInfinity;
            }
        }

        return value;
    }

    // 1 回のリープフロッグ受理確率が 1/2 をまたぐまでステップ幅を倍増または半減させる（Hoffman & Gelman 2014, Algorithm 4）
    private double FindReasonableStepSize(double epsilon)
    {
        Trial.CopyPositionFrom(Current);
        SampleMomentum(Trial);
        double[] momentum = (double[])Trial.Momentum.Clone();
        double h0 = Trial.Hamiltonian();
        double LogRatio(double step)
        {
            Trial.CopyPositionFrom(Current);
            momentum.CopyTo(Trial.Momentum, 0);
            Metric.Multiply(Trial.Momentum, Trial.Velocity);
            Leapfrog(Trial, step);
            double value = h0 - Trial.Hamiltonian();
            return double.IsNaN(value) ? double.NegativeInfinity : value;
        }

        double logHalf = Math.Log(0.5);
        double logRatio = LogRatio(epsilon);
        int direction = logRatio > logHalf ? 1 : -1;
        for (int k = 0; k < 100; k++)
        {
            if (!(direction == 1 ? logRatio > logHalf : logRatio < logHalf))
            {
                break;
            }

            double next = direction == 1 ? 2 * epsilon : 0.5 * epsilon;
            if (!(next > double.Epsilon && next < 1e300))
            {
                break;
            }

            epsilon = next;
            logRatio = LogRatio(epsilon);
        }

        return epsilon;
    }
}

internal sealed record HamiltonianSettings(
    SamplerSettings Common, int Chains, double TargetAcceptance, double? InitialStepSize, HamiltonianMetric Metric,
    Matrix<double>? InitialInverseMetric, bool AdaptMetric, WarmupWindowOptions Windows, double DivergenceThreshold)
{
    public static HamiltonianSettings Validate(HamiltonianSamplerOptions options, double defaultTarget)
    {
        SamplerSettings common = SamplerSettings.Validate(options);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.Chains, 1, nameof(options.Chains));
        double target = options.TargetAcceptance ?? defaultTarget;
        if (!(target > 0 && target < 1))
        {
            throw new ArgumentOutOfRangeException(nameof(options.TargetAcceptance), "The target acceptance must lie in (0, 1).");
        }

        if (options.InitialStepSize is { } step && !(step > 0 && double.IsFinite(step)))
        {
            throw new ArgumentOutOfRangeException(nameof(options.InitialStepSize), "The initial step size must be positive and finite.");
        }

        if (options.Metric is not (HamiltonianMetric.Diagonal or HamiltonianMetric.Dense))
        {
            throw new ArgumentOutOfRangeException(nameof(options.Metric));
        }

        if (options.InitialInverseMetric is { IsSquare: false })
        {
            throw new ArgumentException("The initial inverse metric must be square.", nameof(options));
        }

        if (!(options.DivergenceThreshold > 0))
        {
            throw new ArgumentOutOfRangeException(nameof(options.DivergenceThreshold), "The divergence threshold must be positive.");
        }

        WarmupWindowOptions windows = options.Windows ?? throw new ArgumentNullException(nameof(options.Windows));
        _ = new WarmupSchedule(common.Warmup, windows);
        return new HamiltonianSettings(common, options.Chains, target, options.InitialStepSize, options.Metric,
            options.InitialInverseMetric?.Clone(), options.AdaptMetric, windows, options.DivergenceThreshold);
    }

    public void RequireDimension(int d)
    {
        if (InitialInverseMetric is not null && InitialInverseMetric.Rows != d)
        {
            throw new ArgumentException($"The initial inverse metric must be {d} x {d}.", "options");
        }

        Common.RequireStorage(d, Chains);
    }
}

internal readonly ref struct HamiltonianStorage(
    MatrixSpan<double> draws, VectorSpan<double> logDensities, VectorSpan<double> acceptance, VectorSpan<double> energies,
    VectorSpan<int> depths, VectorSpan<int> leapfrogs, VectorSpan<int> divergences)
{
    public readonly MatrixSpan<double> Draws = draws;
    public readonly VectorSpan<double> LogDensities = logDensities;
    public readonly VectorSpan<double> Acceptance = acceptance;
    public readonly VectorSpan<double> Energies = energies;
    public readonly VectorSpan<int> Depths = depths;
    public readonly VectorSpan<int> Leapfrogs = leapfrogs;
    public readonly VectorSpan<int> Divergences = divergences;
}

internal static class HamiltonianDriver
{
    public static HamiltonianSamplingResult Run<TDensity>(
        HamiltonianSettings settings, TDensity density, int d, double[] starts, int maxTreeDepth,
        Func<int, HamiltonianChain<TDensity>> createChain, CancellationToken cancellationToken)
        where TDensity : IDifferentiableLogDensity
    {
        ArgumentNullException.ThrowIfNull(density);
        settings.RequireDimension(d);
        int chains = settings.Chains, n = settings.Common.Draws;
        var draws = new Tensor<double>(d, n, chains);
        var logDensities = new Matrix<double>(n, chains);
        var acceptance = new Matrix<double>(n, chains);
        var energies = new Matrix<double>(n, chains);
        var depths = new Matrix<int>(n, chains);
        var leapfrogs = new Matrix<int>(n, chains);
        var divergences = new Matrix<int>(n, chains);
        var metrics = new Tensor<double>(d, d, chains);
        var evaluations = new long[chains];
        var stepSizes = new double[chains];
        ChainRunner.Run(chains, settings.Common.MaxDegreeOfParallelism, cancellationToken, c =>
        {
            HamiltonianChain<TDensity> chain = createChain(c);
            var storage = new HamiltonianStorage(draws.GetMatrix(c), logDensities.Column(c), acceptance.Column(c), energies.Column(c),
                depths.Column(c), leapfrogs.Column(c), divergences.Column(c));
            chain.Run(starts.AsSpan(c * d, d), storage, cancellationToken);
            evaluations[c] = chain.Evaluations;
            stepSizes[c] = chain.StepSize;
            chain.WriteInverseMetric(metrics.GetMatrix(c));
        });

        return new HamiltonianSamplingResult(draws, logDensities, evaluations, settings.Common.Seed, stepSizes, metrics,
            acceptance, energies, depths, leapfrogs, divergences, maxTreeDepth);
    }
}
