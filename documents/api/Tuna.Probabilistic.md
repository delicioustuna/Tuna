# Tuna.Probabilistic API リファレンス

> このファイルは `tools/ApiDocGenerator` によって自動生成されています。手動で編集せず、`dotnet build Tuna.slnx -c Release` の後に再生成してください。
>
> `(+N overloads)` は `Vector` / `VectorSpan` / `ReadOnlyVectorSpan` 等のコンテナ・ビュー組み合わせや `double` / `float` 固定型オーバーロードの省略を表します。

## 公開型一覧

- `Tuna.Probabilistic.Inference` (43 型): [`BoxBijector`](#class-boxbijector), [`CompositeBijector`](#class-compositebijector), [`ConvergenceDiagnostics`](#static-class-convergencediagnostics), [`EllipticalSliceSampler`](#class-ellipticalslicesampler), [`EllipticalSliceSamplerOptions`](#class-ellipticalslicesampleroptions), [`EllipticalSliceSamplingResult`](#class-ellipticalslicesamplingresult), [`EnsembleSampler`](#class-ensemblesampler), [`EnsembleSamplerOptions`](#class-ensemblesampleroptions), [`EnsembleSamplingResult`](#class-ensemblesamplingresult), [`HamiltonianMetric`](#enum-hamiltonianmetric), [`HamiltonianMonteCarlo`](#class-hamiltonianmontecarlo), [`HamiltonianMonteCarloOptions`](#class-hamiltonianmontecarlooptions), [`HamiltonianSamplerOptions`](#abstract-class-hamiltoniansampleroptions), [`HamiltonianSamplingResult`](#class-hamiltoniansamplingresult), [`IBijector`](#interface-ibijector), [`IDifferentiableLogDensity`](#interface-idifferentiablelogdensity), [`ILogDensity`](#interface-ilogdensity), [`LogDensities`](#static-class-logdensities), [`MatrixBijectors`](#static-class-matrixbijectors), [`MultiChainSamplerOptions`](#abstract-class-multichainsampleroptions), [`NegativeDifferentiableLogDensity<TDensity>`](#readonly-struct-negativedifferentiablelogdensitytdensity), [`NegativeLogDensity<TDensity>`](#readonly-struct-negativelogdensitytdensity), [`NoUTurnSampler`](#class-nouturnsampler), [`NoUTurnSamplerOptions`](#class-nouturnsampleroptions), [`OrderedBijector`](#class-orderedbijector), [`ParallelTempering`](#class-paralleltempering), [`ParallelTemperingOptions`](#class-paralleltemperingoptions), [`ParallelTemperingResult`](#class-paralleltemperingresult), [`ParameterSummary`](#class-parametersummary), [`PosteriorSummary`](#class-posteriorsummary), [`PotentialLogDensity<TObjective>`](#readonly-struct-potentiallogdensitytobjective), [`RandomWalkMetropolis`](#class-randomwalkmetropolis), [`RandomWalkMetropolisOptions`](#class-randomwalkmetropolisoptions), [`RandomWalkMetropolisResult`](#class-randomwalkmetropolisresult), [`SamplerOptions`](#abstract-class-sampleroptions), [`SamplingResult`](#class-samplingresult), [`SimplexBijector`](#class-simplexbijector), [`SliceSampler`](#class-slicesampler), [`SliceSamplerOptions`](#class-slicesampleroptions), [`SliceSamplingResult`](#class-slicesamplingresult), [`TransformedDifferentiableLogDensity<TDensity, TBijector>`](#readonly-struct-transformeddifferentiablelogdensitytdensity-tbijector), [`TransformedLogDensity<TDensity, TBijector>`](#readonly-struct-transformedlogdensitytdensity-tbijector), [`WarmupWindowOptions`](#class-warmupwindowoptions)

## Tuna.Probabilistic.Inference

### `class BoxBijector`

Constrains each coordinate independently to an open interval, half-line, or the entire real line. / 各成分を独立に開区間、半直線、または実数全体へ制約する変換です。

- 継承・実装: `IBijector`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `BoxBijector(ReadOnlySpan<double> lowerBounds, ReadOnlySpan<double> upperBounds)` | Initializes a new instance of `BoxBijector` with per-coordinate lower and upper bounds. / 成分ごとの下限と上限を指定して `BoxBijector` の新しいインスタンスを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int ConstrainedDimension { get; }` | Gets the dimension (number of coordinates) of the constrained space. / 制約付き空間の次元（成分数）を取得します。 |
| `double[] LowerBounds { get; }` | Gets a copy of the per-coordinate lower bounds. / 成分ごとの下限のコピーを取得します。 |
| `int UnconstrainedDimension { get; }` | Gets the dimension of the unconstrained space. / 制約のない空間の次元を取得します。 |
| `double[] UpperBounds { get; }` | Gets a copy of the per-coordinate upper bounds. / 成分ごとの上限のコピーを取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Forward(ReadOnlyVectorSpan<double> unconstrained, VectorSpan<double> constrained)` | Maps an unconstrained point to the constrained space and returns the log absolute Jacobian determinant. / 制約のない点を制約付き空間へ写像し、ヤコビアン行列式の絶対値の自然対数を返します。 |
| `void Inverse(ReadOnlyVectorSpan<double> constrained, VectorSpan<double> unconstrained)` | Maps a constrained point back to the unconstrained space. / 制約付きの点を制約のない空間へ逆写像します。 |
| `static BoxBijector Positive(int dimension)` | Creates a `BoxBijector` that constrains all coordinates to be strictly positive `(0, +∞)`. / すべての成分を正の値 `(0, +∞)` に制約する `BoxBijector` を生成します。 |
| `void PullBack(ReadOnlyVectorSpan<double> unconstrained, ReadOnlyVectorSpan<double> constrainedGradient, VectorSpan<double> unconstrainedGradient)` | Pulls a constrained-space log-density gradient back to the unconstrained space, including the Jacobian gradient term. / 制約付き空間の勾配にヤコビアン項の勾配を加えて制約のない空間へ引き戻します。 |
| `static BoxBijector Uniform(int dimension, double lowerBound, double upperBound)` | Creates a `BoxBijector` that applies the same interval `(lowerBound, upperBound)` to all coordinates. / すべての成分に同一の区間 `(lowerBound, upperBound)` を適用する `BoxBijector` を生成します。 |

### `class CompositeBijector`

Combines multiple bijectors by applying each to a contiguous block of coordinates in both unconstrained and constrained spaces. / 複数の制約変換を、制約なし空間および制約付き空間の連続したブロックに順番に適用する複合変換です。

- 継承・実装: `IBijector`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `CompositeBijector(params IBijector[] parts)` | Initializes a new instance of `CompositeBijector` from the specified component bijectors in order. / 指定した制約変換を順番に並べて `CompositeBijector` の新しいインスタンスを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int ConstrainedDimension { get; }` | Gets the dimension (number of coordinates) of the constrained space. / 制約付き空間の次元（成分数）を取得します。 |
| `int UnconstrainedDimension { get; }` | Gets the dimension of the unconstrained space. / 制約のない空間の次元を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Forward(ReadOnlyVectorSpan<double> unconstrained, VectorSpan<double> constrained)` | Maps an unconstrained point to the constrained space and returns the log absolute Jacobian determinant. / 制約のない点を制約付き空間へ写像し、ヤコビアン行列式の絶対値の自然対数を返します。 |
| `void Inverse(ReadOnlyVectorSpan<double> constrained, VectorSpan<double> unconstrained)` | Maps a constrained point back to the unconstrained space. / 制約付きの点を制約のない空間へ逆写像します。 |
| `void PullBack(ReadOnlyVectorSpan<double> unconstrained, ReadOnlyVectorSpan<double> constrainedGradient, VectorSpan<double> unconstrainedGradient)` | Pulls a constrained-space log-density gradient back to the unconstrained space, including the Jacobian gradient term. / 制約付き空間の勾配にヤコビアン項の勾配を加えて制約のない空間へ引き戻します。 |

### `static class ConvergenceDiagnostics`

Computes MCMC convergence diagnostics, effective sample sizes, and Monte Carlo standard errors. / MCMC の収束診断、有効標本数、およびモンテカルロ標準誤差を計算します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static double BulkEffectiveSampleSize(ReadOnlyMatrixSpan<double> draws)` | Computes the rank-normalized bulk effective sample size for estimating location summaries. / 分布中心部の推定に対する順位正規化バルク有効標本数を計算します。 |
| `static double EffectiveSampleSize(ReadOnlyMatrixSpan<double> draws)` | Computes the effective sample size for estimating the posterior mean. / 事後平均の推定に対する有効標本数を計算します。 |
| `static double MeanStandardError(ReadOnlyMatrixSpan<double> draws)` | Computes the Monte Carlo standard error of the sample mean, `s / √S_eff`. / 標本平均のモンテカルロ標準誤差 `s / √S_eff` を計算します。 |
| `static double Quantile(ReadOnlyMatrixSpan<double> draws, double probability)` | Computes the sample quantile pooled across all chains using Hyndman–Fan definition 7 linear interpolation. / 全チェーンを統合した標本の分位点を Hyndman–Fan 定義 7 の線形補間で計算します。 |
| `static double QuantileEffectiveSampleSize(ReadOnlyMatrixSpan<double> draws, double probability)` | Computes the effective sample size for estimating the quantile at the specified probability. / 指定した確率の分位点推定に対する有効標本数を計算します。 |
| `static double QuantileStandardError(ReadOnlyMatrixSpan<double> draws, double probability)` | Computes the Monte Carlo standard error of the quantile at the specified probability. / 指定した確率の分位点に対するモンテカルロ標準誤差を計算します。 |
| `static double RHat(ReadOnlyMatrixSpan<double> draws)` | Computes the rank-normalized split-`R̂` convergence diagnostic. / 順位正規化した分割チェーンの収束診断 `R̂` を計算します。 |
| `static double SplitRHat(ReadOnlyMatrixSpan<double> draws)` | Computes the classical split-`R̂` convergence diagnostic without rank normalization. / 順位正規化を行わない分割チェーンの収束診断 `R̂` を計算します。 |
| `static double TailEffectiveSampleSize(ReadOnlyMatrixSpan<double> draws)` | Computes the tail effective sample size for estimating extreme quantiles. / 分布の裾の推定に対する裾有効標本数を計算します。 |

### `class EllipticalSliceSampler`

Samples from a posterior with a multivariate Gaussian prior `π(θ) ∝ L(θ)·N(θ; μ, Σ)` using elliptical slice sampling. / 正規事前分布を持つ事後分布 `π(θ) ∝ L(θ)·N(θ; μ, Σ)` から、楕円スライスサンプラーによりサンプリングします。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `EllipticalSliceSampler(EllipticalSliceSamplerOptions options = null)` | Initializes a new instance of `EllipticalSliceSampler` with the specified options. / 指定した設定で `EllipticalSliceSampler` の新しいインスタンスを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Chains { get; }` | Gets the number of independent chains. / 独立したチェーンの数を取得します。 |
| `ulong Seed { get; }` | Gets the random seed used for sampling (configured value or obtained from the OS). / サンプリングに使用する乱数シード（指定値、または省略時に OS から取得した値）を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `EllipticalSliceSamplingResult Sample(Func<ReadOnlyVectorSpan<double>, double> logLikelihood, ReadOnlySpan<double> priorMean, ReadOnlyMatrixSpan<double> priorCovariance, CancellationToken cancellationToken = default)` | Generates posterior draws initializing each chain by sampling from the Gaussian prior. / 各チェーンを事前分布からのサンプリングで初期化し、標本を生成します。 |
| `EllipticalSliceSamplingResult Sample<TLikelihood>(TLikelihood logLikelihood, ReadOnlySpan<double> priorMean, ReadOnlyMatrixSpan<double> priorCovariance, CancellationToken cancellationToken = default)` | Generates posterior draws initializing each chain by sampling from the Gaussian prior. / 各チェーンを事前分布からのサンプリングで初期化し、標本を生成します。 |
| `EllipticalSliceSamplingResult Sample(Func<ReadOnlyVectorSpan<double>, double> logLikelihood, ReadOnlySpan<double> priorMean, ReadOnlyMatrixSpan<double> priorCovariance, ReadOnlyMatrixSpan<double> initialPoints, CancellationToken cancellationToken = default)` | Generates posterior draws starting each chain from its corresponding column of `initialPoints`. / チェーンごとに指定した初期点から標本を生成します。 |
| `EllipticalSliceSamplingResult Sample<TLikelihood>(TLikelihood logLikelihood, ReadOnlySpan<double> priorMean, ReadOnlyMatrixSpan<double> priorCovariance, ReadOnlyMatrixSpan<double> initialPoints, CancellationToken cancellationToken = default)` | Generates posterior draws starting each chain from its corresponding column of `initialPoints`. / チェーンごとに指定した初期点から標本を生成します。 |

### `class EllipticalSliceSamplerOptions`

Configures `EllipticalSliceSampler`. / `EllipticalSliceSampler` の設定オプションを表します。

- 継承・実装: `MultiChainSamplerOptions`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `EllipticalSliceSamplerOptions()` |  |

### `class EllipticalSliceSamplingResult`

Represents the draws and log-likelihood values produced by `EllipticalSliceSampler`. / `EllipticalSliceSampler` のサンプリング結果と対数尤度を表します。

- 継承・実装: `SamplingResult`

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `Matrix<double> LogLikelihoods { get; }` | Gets the log-likelihood evaluated at each retained draw, with shape `[draw × chain]`. / 保持された各標本における対数尤度行列（形状 `[反復 × チェーン]`）を取得します。 |

### `class EnsembleSampler`

Samples from a log-density using the affine-invariant stretch-move ensemble sampler (Goodman & Weare 2010). / ストレッチ移動によるアフィン不変アンサンブルサンプラー（Goodman & Weare 2010）により、対数密度からサンプリングします。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `EnsembleSampler(EnsembleSamplerOptions options = null)` | Initializes a new instance of `EnsembleSampler` with the specified options. / 指定した設定で `EnsembleSampler` の新しいインスタンスを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `ulong Seed { get; }` | Gets the random seed used for sampling (configured value or obtained from the OS). / サンプリングに使用する乱数シード（指定値、または省略時に OS から取得した値）を取得します。 |
| `int? Walkers { get; }` | Gets the configured number of walkers, or `null` if determined automatically from the dimension. / 設定されたウォーカー数を取得します（次元から自動決定する場合は `null`）。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `EnsembleSamplingResult Sample(Func<ReadOnlyVectorSpan<double>, double> logDensity, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)` | Generates posterior draws by scattering walkers around the specified initial point. / 指定した初期点の周囲にウォーカーを配置して標本を生成します。 |
| `EnsembleSamplingResult Sample(Func<ReadOnlyVectorSpan<double>, double> logDensity, ReadOnlyMatrixSpan<double> initialPoints, CancellationToken cancellationToken = default)` | Generates posterior draws starting each walker from its corresponding column of `initialPoints`. / ウォーカーごとに指定した初期点から標本を生成します。 |
| `EnsembleSamplingResult Sample<TDensity>(TDensity density, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)` | Generates posterior draws by scattering walkers around the specified initial point. / 指定した初期点の周囲にウォーカーを配置して標本を生成します。 |
| `EnsembleSamplingResult Sample<TDensity>(TDensity density, ReadOnlyMatrixSpan<double> initialPoints, CancellationToken cancellationToken = default)` | Generates posterior draws starting each walker from its corresponding column of `initialPoints`. / ウォーカーごとに指定した初期点から標本を生成します。 |

### `class EnsembleSamplerOptions`

Configures `EnsembleSampler`. / `EnsembleSampler` の設定オプションを表します。

- 継承・実装: `SamplerOptions`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `EnsembleSamplerOptions()` |  |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double InitialSpread { get; set; }` | Gets or initializes the Gaussian standard deviation used to scatter walkers around a single initial point (positive finite value; default is `10⁻³`). Points with non-finite log-density are redrawn. / 単一の初期点から開始する際にウォーカーを周囲に配置する正規分布の標準偏差を取得または設定します（正の有限値、既定値 `10⁻³`）。対数密度が非有限の点は再生成されます。 |
| `double StretchScale { get; set; }` | Gets or initializes the stretch-move scale parameter `a` (finite and greater than 1; default is 2 per Goodman & Weare 2010). Larger values propose longer moves and lower acceptance rates. / ストレッチ移動のスケールパラメータ `a` を取得または設定します（1 より大きい有限値、既定値 2、Goodman & Weare 2010）。大きいほど移動幅が広がり受理率は下がります。 |
| `int? Walkers { get; set; }` | Gets or initializes the number of walkers `K` (2 or greater; defaults to `max(2d, 8)` when `null`, where `d` is the dimension). / ウォーカー数 `K` を取得または設定します（2 以上。`null` の既定値は `max(2d, 8)`、`d` は次元）。 |

### `class EnsembleSamplingResult`

Represents the draws and acceptance diagnostics produced by `EnsembleSampler` (`ChainCount` equals the number of walkers). / `EnsembleSampler` のサンプリング結果と受理率診断情報を表します（`ChainCount` はウォーカー数に対応します）。

- 継承・実装: `SamplingResult`

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `IReadOnlyList<double> AcceptanceRates { get; }` | Gets the post-warmup move acceptance rate for each walker (roughly 0.2–0.5 indicates an appropriate `StretchScale`). / ウォーカーごとのウォームアップ後の移動受理率を取得します（おおむね 0.2〜0.5 であれば `StretchScale` は妥当です）。 |
| `double MeanAcceptanceRate { get; }` | Gets the mean acceptance rate across all walkers. / 全ウォーカーの平均受理率を取得します。 |

### `enum HamiltonianMetric`

Specifies the structure of the inverse mass matrix (metric) `M⁻¹` in Hamiltonian samplers. / ハミルトニアン法における逆計量（運動量共分散の逆行列 `M⁻¹`）の構造を指定します。

- 継承・実装: `IComparable`, `IConvertible`, `IFormattable`, `ISpanFormattable`

| 値 | 説明 |
| --- | --- |
| `Diagonal` | Diagonal inverse metric, adapting per-parameter scales only (default). / 対角逆計量。パラメータごとのスケールのみを適応させます（既定）。 |
| `Dense` | Dense inverse metric, capturing linear correlations across parameters using Cholesky factorization at `O(d²)` cost per leapfrog step. / 密逆計量。コレスキー分解を用いてパラメータ間の線形相関も取り除きます（1 リープフロッグ当たり `O(d²)`）。 |

### `class HamiltonianMonteCarlo`

Samples from a differentiable log-density using fixed-trajectory Hamiltonian Monte Carlo (HMC). / 固定軌道長のハミルトニアン・モンテカルロ法（HMC）により、微分可能な対数密度からサンプリングします。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `HamiltonianMonteCarlo(HamiltonianMonteCarloOptions options = null)` | Initializes a new instance of `HamiltonianMonteCarlo` with the specified options. / 指定した設定で `HamiltonianMonteCarlo` の新しいインスタンスを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Chains { get; }` | Gets the number of independent chains. / 独立したチェーンの数を取得します。 |
| `ulong Seed { get; }` | Gets the random seed used for sampling (configured value or obtained from the OS). / サンプリングに使用する乱数シード（指定値、または省略時に OS から取得した値）を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `HamiltonianSamplingResult Sample(Func<ReadOnlyVectorSpan<double>, VectorSpan<double>, double> logDensity, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)` | Generates posterior draws starting all chains from the specified initial point. / すべてのチェーンを指定した初期点から開始して標本を生成します。 |
| `HamiltonianSamplingResult Sample(Func<ReadOnlyVectorSpan<double>, VectorSpan<double>, double> logDensity, ReadOnlyMatrixSpan<double> initialPoints, CancellationToken cancellationToken = default)` | Generates posterior draws starting each chain from its corresponding column of `initialPoints`. / チェーンごとに指定した初期点から標本を生成します。 |
| `HamiltonianSamplingResult Sample<TDensity>(TDensity density, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)` | Generates posterior draws starting all chains from the specified initial point. / すべてのチェーンを指定した初期点から開始して標本を生成します。 |
| `HamiltonianSamplingResult Sample<TDensity>(TDensity density, ReadOnlyMatrixSpan<double> initialPoints, CancellationToken cancellationToken = default)` | Generates posterior draws starting each chain from its corresponding column of `initialPoints`. / チェーンごとに指定した初期点から標本を生成します。 |

### `class HamiltonianMonteCarloOptions`

Configures `HamiltonianMonteCarlo`. / `HamiltonianMonteCarlo` の設定オプションを表します。

- 継承・実装: `HamiltonianSamplerOptions`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `HamiltonianMonteCarloOptions()` |  |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double StepSizeJitter { get; set; }` | Gets or initializes the per-transition step-size jitter factor `j` in `[0, 1)` (default is 0). / 遷移ごとのステップ幅の揺らぎ係数 `j`（`[0, 1)`、既定値 0）を取得または設定します。 |
| `int Steps { get; set; }` | Gets or initializes the number of leapfrog steps `L` per transition (1 or greater; default is 10). / 1 回の遷移あたりのリープフロッグ回数 `L` を取得または設定します（1 以上、既定値 10）。 |

### `abstract class HamiltonianSamplerOptions`

Provides common configuration options for `NoUTurnSampler` and `HamiltonianMonteCarlo`. / `NoUTurnSampler` および `HamiltonianMonteCarlo` に共通の設定オプションを提供します。

- 継承・実装: `MultiChainSamplerOptions`

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `bool AdaptMetric { get; set; }` | Gets or initializes a value indicating whether the inverse metric is updated from regularized sample covariance at the end of each warmup window (default is `true`). / ウォームアップの各ウィンドウ終端で、正則化した標本共分散から逆計量を更新するかどうかを取得または設定します（既定値 `true`）。 |
| `double DivergenceThreshold { get; set; }` | Gets or initializes the Hamiltonian increase threshold above which a trajectory is terminated and recorded as divergent (positive; default is 1000). / 軌道を発散と判定するハミルトニアン増加量の閾値を取得または設定します（正の値、既定値 1000）。この値を超えた時点で軌道を打ち切り、発散として記録します。 |
| `Matrix<double> InitialInverseMetric { get; set; }` | Gets or initializes the initial inverse metric `M⁻¹` (`d × d` symmetric positive definite matrix; only diagonal entries are used for `Diagonal`). Defaults to the identity matrix when `null`. / 逆計量 `M⁻¹` の初期値を取得または設定します（`d × d` の対称正定値行列。`Diagonal` では対角成分のみ使用）。`null`（既定）の場合は単位行列になります。 |
| `double? InitialStepSize { get; set; }` | Gets or initializes the initial leapfrog step size. / リープフロッグ法の初期ステップ幅を取得または設定します。 |
| `HamiltonianMetric Metric { get; set; }` | Gets or initializes the structure of the inverse metric (default is `Diagonal`). / 逆計量の構造を取得または設定します（既定値 `Diagonal`）。 |
| `double? TargetAcceptance { get; set; }` | Gets or initializes the target mean acceptance statistic during warmup (in `(0, 1)`). Defaults to 0.8 for `NoUTurnSampler` and 0.65 for `HamiltonianMonteCarlo`. / ウォームアップ期間中の目標平均受理統計量を取得または設定します（`(0, 1)`）。既定値は `NoUTurnSampler` で 0.8、`HamiltonianMonteCarlo` で 0.65 です。 |
| `WarmupWindowOptions Windows { get; set; }` | Gets or initializes the warmup adaptation window schedule. / ウォームアップ期間の適応ウィンドウ設定を取得または設定します。 |

### `class HamiltonianSamplingResult`

Represents the draws and Hamiltonian diagnostics produced by `NoUTurnSampler` or `HamiltonianMonteCarlo`. / `NoUTurnSampler` または `HamiltonianMonteCarlo` のサンプリング結果と診断情報を表します。

- 継承・実装: `SamplingResult`

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `Matrix<double> AcceptanceStatistics { get; }` | Gets the acceptance statistic for each retained draw with shape `[draw × chain]` (for NUTS, the mean of `min(1, e^{H₀ − H})` along the trajectory). / 保持された各標本の遷移における受理統計量（形状 `[反復 × チェーン]`、NUTS では軌道上の点の `min(1, e^{H₀ − H})` の平均）を取得します。 |
| `int DivergenceCount { get; }` | Gets the total number of divergent post-warmup transitions across all chains. / 全チェーン合計のウォームアップ後発散遷移回数を取得します。 |
| `Matrix<int> Divergences { get; }` | Gets the divergence indicator matrix with shape `[draw × chain]` (1 indicates a divergent transition, 0 otherwise). / 保持された各標本の遷移で発散が発生したかを示す行列（形状 `[反復 × チェーン]`、1 が発散）を取得します。 |
| `Matrix<double> Energies { get; }` | Gets the Hamiltonian (total energy) `−ln π(θ) + ½pᵀM⁻¹p` at each retained draw with shape `[draw × chain]`. / 保持された各標本で選択された点のハミルトニアン（全エネルギー）`−ln π(θ) + ½pᵀM⁻¹p`（形状 `[反復 × チェーン]`）を取得します。 |
| `Tensor<double> InverseMetrics { get; }` | Gets the post-warmup inverse metric `M⁻¹` for each chain with shape `[dimension, dimension, chain]` (diagonal matrix when using `Diagonal`). / チェーンごとのウォームアップ後の逆計量 `M⁻¹`（形状 `[次元, 次元, チェーン]`、対角計量では対角行列）を取得します。 |
| `Matrix<int> LeapfrogSteps { get; }` | Gets the number of leapfrog steps (equal to gradient evaluations) for each retained draw with shape `[draw × chain]`. / 保持された各標本の遷移におけるリープフロッグ回数（勾配評価回数と等しい、形状 `[反復 × チェーン]`）を取得します。 |
| `int MaxTreeDepth { get; }` | Gets the configured maximum tree depth (0 for static `HamiltonianMonteCarlo`). / 設定された木の深さの上限（静的 `HamiltonianMonteCarlo` では 0）を取得します。 |
| `int MaxTreeDepthHits { get; }` | Gets the total number of post-warmup transitions that reached `MaxTreeDepth` (NUTS only). / ウォームアップ後に木の深さが `MaxTreeDepth` に到達した遷移の総数を取得します（NUTS のみ）。 |
| `IReadOnlyList<double> StepSizes { get; }` | Gets the fixed post-warmup step size for each chain. / チェーンごとのウォームアップ後に固定されたステップ幅を取得します。 |
| `Matrix<int> TreeDepths { get; }` | Gets the binary tree depth for each retained draw with shape `[draw × chain]` (0 for static `HamiltonianMonteCarlo`). / 保持された各標本の遷移における木の深さ（形状 `[反復 × チェーン]`、静的 `HamiltonianMonteCarlo` では 0）を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double EnergyBayesianFractionOfMissingInformation(int chain)` | Computes the Energy Bayesian Fraction of Missing Information (E-BFMI) for the specified chain. / 指定したチェーンの E-BFMI（エネルギーのベイズ的欠損情報割合）を計算します。 |

### `interface IBijector`

Represents a smooth, invertible transformation between an unconstrained space `ℝⁿ` and a constrained parameter space. / 制約のない空間 `ℝⁿ` と制約付きパラメータ空間の間の滑らかな可逆写像（制約変換）を表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int ConstrainedDimension { get; }` | Gets the dimension (number of coordinates) of the constrained space. / 制約付き空間の次元（成分数）を取得します。 |
| `int UnconstrainedDimension { get; }` | Gets the dimension of the unconstrained space. / 制約のない空間の次元を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Forward(ReadOnlyVectorSpan<double> unconstrained, VectorSpan<double> constrained)` | Maps an unconstrained point to the constrained space and returns the log absolute Jacobian determinant. / 制約のない点を制約付き空間へ写像し、ヤコビアン行列式の絶対値の自然対数を返します。 |
| `void Inverse(ReadOnlyVectorSpan<double> constrained, VectorSpan<double> unconstrained)` | Maps a constrained point back to the unconstrained space. / 制約付きの点を制約のない空間へ逆写像します。 |
| `void PullBack(ReadOnlyVectorSpan<double> unconstrained, ReadOnlyVectorSpan<double> constrainedGradient, VectorSpan<double> unconstrainedGradient)` | Pulls a constrained-space log-density gradient back to the unconstrained space, including the Jacobian gradient term. / 制約付き空間の勾配にヤコビアン項の勾配を加えて制約のない空間へ引き戻します。 |

### `interface IDifferentiableLogDensity`

Represents a differentiable unnormalized log-density function on an unconstrained real vector space. / 勾配を計算できる、制約のない実ベクトル空間上の対数密度関数を表します。

- 継承・実装: `ILogDensity`

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double LogDensity(ReadOnlyVectorSpan<double> theta, VectorSpan<double> gradient)` | Evaluates the log-density and its gradient at the specified point. / 指定した点の対数密度と勾配を評価します。 |

### `interface ILogDensity`

Represents an unnormalized log-density function on an unconstrained real vector space. / 制約のない実ベクトル空間上の対数密度関数を表します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double LogDensity(ReadOnlyVectorSpan<double> theta)` | Evaluates the log-density at the specified point. / 指定した点の対数密度を評価します。 |

### `static class LogDensities`

Provides helpers for creating log-densities from delegates, converting between log-densities and optimization objectives, and checking analytic gradients. / デリゲートからの対数密度の生成、対数密度と最適化目的関数の相互変換、および勾配の検証を行うユーティリティを提供します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static IDifferentiableLogDensity FromAutomaticDifferentiation(Func<ReadOnlySpan<ScalarNode>, ScalarNode> model)` | Creates a log density that differentiates a scalar-node model when evaluated. / スカラーノードで記述したモデルから、勾配付き対数密度を作成します。 |
| `static IDifferentiableLogDensity FromAutomaticDifferentiation(Func<Tape, VectorNode<double>, ScalarNode> model)` | Creates a log density that differentiates a vector-node model when evaluated. / ベクトルノードで記述したモデルから、勾配付き対数密度を作成します。 |
| `static ILogDensity FromDelegate(Func<ReadOnlyVectorSpan<double>, double> logDensity)` | Wraps a log-density delegate as an `ILogDensity`. / 対数密度を計算するデリゲートを `ILogDensity` としてラップします。 |
| `static IDifferentiableLogDensity FromDelegate(Func<ReadOnlyVectorSpan<double>, VectorSpan<double>, double> logDensity)` | Wraps a value-and-gradient delegate as an `IDifferentiableLogDensity`. / 対数密度を返し第 2 引数へ勾配を書き込むデリゲートを `IDifferentiableLogDensity` としてラップします。 |
| `static PotentialLogDensity<TObjective> FromPotential<TObjective>(TObjective potential)` | Wraps a differentiable potential energy objective `U(θ) = −ln π(θ)` as a log-density `ln π(θ) = −U(θ)`. / ポテンシャル `U(θ) = −ln π(θ)` を表す微分可能な目的関数を、対数密度 `ln π(θ) = −U(θ)` としてラップします。 |
| `static double GradientError<TDensity>(TDensity density, ReadOnlySpan<double> theta)` | Compares the analytic gradient of a log-density against central finite differences. / 対数密度の解析勾配と中心差分による数値微分を比較します。 |
| `static ScalarNode RecordTransformed(IDifferentiableLogDensity density, IBijector bijector, ReadOnlySpan<ScalarNode> unconstrained)` | Records a constrained log density and the transform's log Jacobian as one operation. / 制約付き対数密度と制約変換の対数ヤコビアンを 1 回の演算として記録します。 |
| `static NegativeDifferentiableLogDensity<TDensity> ToDifferentiableObjective<TDensity>(TDensity density)` | Wraps a differentiable log-density as a minimization objective `−ln π(θ)` for gradient-based optimizers such as MAP estimation. / 微分可能な対数密度を、勾配法による MAP 推定などのための最小化目的関数 `−ln π(θ)` としてラップします。 |
| `static NegativeLogDensity<TDensity> ToObjective<TDensity>(TDensity density)` | Wraps a log-density as a minimization objective `−ln π(θ)` for derivative-free optimizers. / 対数密度を最小化目的関数 `−ln π(θ)` としてラップします（勾配を使わない最適化手法向け）。 |

### `static class MatrixBijectors`

Creates constraints for covariance matrices and Cholesky factors. / 共分散行列と Cholesky 因子の制約変換を作成します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static IBijector CorrelationCholesky(int order)` | Creates a transform to a correlation matrix's Cholesky factor. / 相関行列の Cholesky 因子への変換を作成します。 |
| `static IBijector Covariance(int order)` | Creates a positive-definite covariance matrix transform. / 正定値の共分散行列への変換を作成します。 |
| `static IBijector CovarianceCholesky(int order)` | Creates a transform to a lower-triangular factor with positive diagonal. / 正の対角を持つ下三角の Cholesky 因子への変換を作成します。 |

### `abstract class MultiChainSamplerOptions`

Provides configuration options for samplers that run multiple independent chains. / 独立した複数のチェーンを実行するサンプラーの設定オプションを提供します。

- 継承・実装: `SamplerOptions`

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Chains { get; set; }` | Gets or initializes the number of independent chains (1 or greater; default is 4). Four or more chains from dispersed initial points are recommended for convergence diagnostics such as `RHat`. / 独立したチェーンの数を取得または設定します（1 以上、既定値 4）。`RHat` などの収束診断には、互いに離れた初期点から 4 本以上実行することを推奨します。 |

### `readonly struct NegativeDifferentiableLogDensity<TDensity>`

Wraps an `IDifferentiableLogDensity` as a differentiable minimization objective `−ln π(θ)`. Created via `ToDifferentiableObjective`. / `IDifferentiableLogDensity` を微分可能な最小化目的関数 `−ln π(θ)` として扱うラッパーです。`ToDifferentiableObjective` で生成します。

- 継承・実装: `IDifferentiableObjectiveFunction`, `IObjectiveFunction`

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Evaluate(ReadOnlyVectorSpan<double> x)` | Evaluates the objective function at the specified point. / 指定した点で目的関数を評価します。 |
| `double Evaluate(ReadOnlyVectorSpan<double> x, VectorSpan<double> gradient)` | Evaluates the objective function at the specified point and writes its gradient. / 指定した点で目的関数を評価し、勾配を書き込みます。 |

### `readonly struct NegativeLogDensity<TDensity>`

Wraps an `ILogDensity` as a minimization objective `−ln π(θ)`. Created via `ToObjective`. / `ILogDensity` を最小化目的関数 `−ln π(θ)` として扱うラッパーです。`ToObjective` で生成します。

- 継承・実装: `IObjectiveFunction`

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Evaluate(ReadOnlyVectorSpan<double> x)` | Evaluates the objective function at the specified point. / 指定した点で目的関数を評価します。 |

### `class NoUTurnSampler`

Samples from a differentiable log-density using the multinomial No-U-Turn Sampler (NUTS). / 軌道長を自動調整する多項サンプリング版 No-U-Turn サンプラー（NUTS）により、微分可能な対数密度からサンプリングします。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `NoUTurnSampler(NoUTurnSamplerOptions options = null)` | Initializes a new instance of `NoUTurnSampler` with the specified options. / 指定した設定で `NoUTurnSampler` の新しいインスタンスを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Chains { get; }` | Gets the number of independent chains. / 独立したチェーンの数を取得します。 |
| `ulong Seed { get; }` | Gets the random seed used for sampling (configured value or obtained from the OS). / サンプリングに使用する乱数シード（指定値、または省略時に OS から取得した値）を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `HamiltonianSamplingResult Sample(Func<ReadOnlyVectorSpan<double>, VectorSpan<double>, double> logDensity, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)` | Generates posterior draws starting all chains from the specified initial point. / すべてのチェーンを指定した初期点から開始して標本を生成します。 |
| `HamiltonianSamplingResult Sample(Func<ReadOnlyVectorSpan<double>, VectorSpan<double>, double> logDensity, ReadOnlyMatrixSpan<double> initialPoints, CancellationToken cancellationToken = default)` | Generates posterior draws starting each chain from its corresponding column of `initialPoints`. / チェーンごとに指定した初期点から標本を生成します。 |
| `HamiltonianSamplingResult Sample<TDensity>(TDensity density, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)` | Generates posterior draws starting all chains from the specified initial point. / すべてのチェーンを指定した初期点から開始して標本を生成します。 |
| `HamiltonianSamplingResult Sample<TDensity>(TDensity density, ReadOnlyMatrixSpan<double> initialPoints, CancellationToken cancellationToken = default)` | Generates posterior draws starting each chain from its corresponding column of `initialPoints`. / チェーンごとに指定した初期点から標本を生成します。 |

### `class NoUTurnSamplerOptions`

Configures `NoUTurnSampler`. / `NoUTurnSampler` の設定オプションを表します。

- 継承・実装: `HamiltonianSamplerOptions`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `NoUTurnSamplerOptions()` |  |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int MaxTreeDepth { get; set; }` | Gets or initializes the maximum binary tree depth (1 to 30; default is 10). Each transition performs at most `2^MaxTreeDepth − 1` leapfrog steps. / 二分木の深さの上限を取得または設定します（1 以上 30 以下、既定値 10）。1 遷移あたりのリープフロッグ回数は最大 `2^MaxTreeDepth − 1` 回です。 |

### `class OrderedBijector`

Maps `ℝ^K` onto strictly increasing vectors `x₁ < x₂ < … < x_K` via `x₁ = y₁`, `xₖ = xₖ₋₁ + e^{yₖ}`. / `x₁ = y₁`、`xₖ = xₖ₋₁ + e^{yₖ}` により `ℝ^K` を狭義単調増加ベクトル `x₁ < x₂ < … < x_K` へ写像します。

- 継承・実装: `IBijector`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `OrderedBijector(int size)` | Initializes a new instance of `OrderedBijector` for a vector of the specified length. / 指定した成分数の `OrderedBijector` の新しいインスタンスを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int ConstrainedDimension { get; }` | Gets the dimension (number of coordinates) of the constrained space. / 制約付き空間の次元（成分数）を取得します。 |
| `int UnconstrainedDimension { get; }` | Gets the dimension of the unconstrained space. / 制約のない空間の次元を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Forward(ReadOnlyVectorSpan<double> unconstrained, VectorSpan<double> constrained)` | Maps an unconstrained point to the constrained space and returns the log absolute Jacobian determinant. / 制約のない点を制約付き空間へ写像し、ヤコビアン行列式の絶対値の自然対数を返します。 |
| `void Inverse(ReadOnlyVectorSpan<double> constrained, VectorSpan<double> unconstrained)` | Maps a constrained point back to the unconstrained space. / 制約付きの点を制約のない空間へ逆写像します。 |
| `void PullBack(ReadOnlyVectorSpan<double> unconstrained, ReadOnlyVectorSpan<double> constrainedGradient, VectorSpan<double> unconstrainedGradient)` | Pulls a constrained-space log-density gradient back to the unconstrained space, including the Jacobian gradient term. / 制約付き空間の勾配にヤコビアン項の勾配を加えて制約のない空間へ引き戻します。 |

### `class ParallelTempering`

Samples from a log-density using non-reversible parallel tempering (replica exchange) across a ladder of inverse temperatures. / 逆温度ラダー上で非可逆レプリカ交換を行うパラレルテンパリングサンプラーです。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `ParallelTempering(ParallelTemperingOptions options = null)` | Initializes a new instance of `ParallelTempering` with the specified options. / 指定した設定で `ParallelTempering` の新しいインスタンスを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Chains { get; }` | Gets the number of independent replica-exchange chains. / 独立に実行するレプリカ交換チェーンの数を取得します。 |
| `ulong Seed { get; }` | Gets the random seed used for sampling (configured value or obtained from the OS). / サンプリングに使用する乱数シード（指定値、または省略時に OS から取得した値）を取得します。 |
| `int Temperatures { get; }` | Gets the number of temperatures in each ladder. / 各温度ラダーの温度数を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `ParallelTemperingResult Sample(Func<ReadOnlyVectorSpan<double>, double> logDensity, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)` | Generates posterior draws using whole-density tempering `π(θ)^β`, starting all chains and temperatures from the specified initial point. / 対数密度全体を温度調整（`π(θ)^β`）し、すべてのチェーン・温度を指定した初期点から開始して標本を生成します。 |
| `ParallelTemperingResult Sample(Func<ReadOnlyVectorSpan<double>, double> logDensity, ReadOnlyMatrixSpan<double> initialPoints, CancellationToken cancellationToken = default)` | Generates posterior draws using whole-density tempering `π(θ)^β`, starting each chain from its corresponding column of `initialPoints`. / 対数密度全体を温度調整（`π(θ)^β`）し、チェーンごとに指定した初期点から標本を生成します。 |
| `ParallelTemperingResult Sample<TDensity>(TDensity density, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)` | Generates posterior draws using whole-density tempering `π(θ)^β`, starting all chains and temperatures from the specified initial point. / 対数密度全体を温度調整（`π(θ)^β`）し、すべてのチェーン・温度を指定した初期点から開始して標本を生成します。 |
| `ParallelTemperingResult Sample<TDensity>(TDensity density, ReadOnlyMatrixSpan<double> initialPoints, CancellationToken cancellationToken = default)` | Generates posterior draws using whole-density tempering `π(θ)^β`, starting each chain from its corresponding column of `initialPoints`. / 対数密度全体を温度調整（`π(θ)^β`）し、チェーンごとに指定した初期点から標本を生成します。 |
| `ParallelTemperingResult Sample(Func<ReadOnlyVectorSpan<double>, double> logPrior, Func<ReadOnlyVectorSpan<double>, double> logLikelihood, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)` | Generates posterior draws using likelihood-only tempering `p(θ)·L(θ)^β`, starting all chains and temperatures from the specified initial point. / 尤度のみを温度調整（`p(θ)·L(θ)^β`）し、すべてのチェーン・温度を指定した初期点から開始して標本を生成します。 |
| `ParallelTemperingResult Sample(Func<ReadOnlyVectorSpan<double>, double> logPrior, Func<ReadOnlyVectorSpan<double>, double> logLikelihood, ReadOnlyMatrixSpan<double> initialPoints, CancellationToken cancellationToken = default)` | Generates posterior draws using likelihood-only tempering `p(θ)·L(θ)^β`, starting each chain from its corresponding column of `initialPoints`. / 尤度のみを温度調整（`p(θ)·L(θ)^β`）し、チェーンごとに指定した初期点から標本を生成します。 |
| `ParallelTemperingResult Sample<TPrior, TLikelihood>(TPrior logPrior, TLikelihood logLikelihood, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)` | Generates posterior draws using likelihood-only tempering `p(θ)·L(θ)^β`, starting all chains and temperatures from the specified initial point. / 尤度のみを温度調整（`p(θ)·L(θ)^β`）し、すべてのチェーン・温度を指定した初期点から開始して標本を生成します。 |
| `ParallelTemperingResult Sample<TPrior, TLikelihood>(TPrior logPrior, TLikelihood logLikelihood, ReadOnlyMatrixSpan<double> initialPoints, CancellationToken cancellationToken = default)` | Generates posterior draws using likelihood-only tempering `p(θ)·L(θ)^β`, starting each chain from its corresponding column of `initialPoints`. / 尤度のみを温度調整（`p(θ)·L(θ)^β`）し、チェーンごとに指定した初期点から標本を生成します。 |

### `class ParallelTemperingOptions`

Configures `ParallelTempering`. / `ParallelTempering` の設定オプションを表します。

- 継承・実装: `MultiChainSamplerOptions`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `ParallelTemperingOptions()` |  |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `bool AdaptCovariance { get; set; }` | Gets or initializes a value indicating whether each temperature's proposal covariance shape is adapted at the end of each warmup window (default is `true`). / ウォームアップの各ウィンドウ終端で、各温度の提案共分散形状を標本共分散から更新するかどうかを取得または設定します（既定値 `true`）。 |
| `bool AdaptLadder { get; set; }` | Gets or initializes a value indicating whether intermediate inverse temperatures are adapted during warmup (default is `true`). / ウォームアップ期間中に中間の逆温度を適応させるかどうかを取得または設定します（既定値 `true`）。 |
| `int LocalSteps { get; set; }` | Gets or initializes the number of local random-walk Metropolis transitions per temperature between swap attempts (1 or greater; default is 1). / 交換試行の間に各温度で実行するランダムウォーク・メトロポリス遷移の回数を取得または設定します（1 以上、既定値 1）。 |
| `double? MinimumInverseTemperature { get; set; }` | Gets or initializes the inverse temperature `β_min` of the hottest replica. / 最も高温のレプリカの逆温度 `β_min` を取得または設定します。 |
| `double? TargetAcceptance { get; set; }` | Gets or initializes the target acceptance rate for local random-walk Metropolis updates at each temperature (same convention as `TargetAcceptance`; defaults to 0.44 in 1D and 0.234 in 2D or higher). / 各温度のランダムウォーク・メトロポリス法における目標受理率を取得または設定します（`TargetAcceptance` と同じ仕様。`null` の既定値は 1 次元で 0.44、2 次元以上で 0.234）。 |
| `int Temperatures { get; set; }` | Gets or initializes the number of temperatures `K` in the ladder from inverse temperature 1 down to `MinimumInverseTemperature` (2 or greater; default is 8). / 逆温度 1（目標分布）から `MinimumInverseTemperature` までの温度ラダーの段数 `K` を取得または設定します（2 以上、既定値 8）。 |
| `WarmupWindowOptions Windows { get; set; }` | Gets or initializes the warmup adaptation window schedule. / ウォームアップ期間の適応ウィンドウ設定を取得または設定します。 |

### `class ParallelTemperingResult`

Represents the target-temperature draws and replica-exchange diagnostics produced by `ParallelTempering`. / `ParallelTempering` の逆温度 1 の標本とレプリカ交換診断情報を表します。

- 継承・実装: `SamplingResult`

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `IReadOnlyList<double> CommunicationBarriers { get; }` | Gets the estimated global communication barrier `Λ` (sum of adjacent swap rejection rates; Syed et al. 2022) for each chain. / チェーンごとの大域的通信障壁 `Λ`（隣接温度間の交換棄却率の総和、Syed et al. 2022）の推定値を取得します。 |
| `Matrix<double> InverseTemperatures { get; }` | Gets the post-warmup inverse temperature ladder for each chain, with shape `[temperatures × chains]` (each column decreases from 1 to `β_min`). / チェーンごとのウォームアップ後に固定された逆温度ラダー（形状 `[温度数 × チェーン数]`、各列は 1 から `β_min` へ単調減少）を取得します。 |
| `Matrix<double> LocalAcceptanceRates { get; }` | Gets the post-warmup local random-walk Metropolis acceptance rate at each temperature for each chain, with shape `[temperatures × chains]`. / チェーンごとのウォームアップ後の各温度におけるローカルランダムウォーク・メトロポリス受理率（形状 `[温度数 × チェーン数]`）を取得します。 |
| `IReadOnlyList<long> RoundTrips { get; }` | Gets the number of post-warmup cold-to-hot-to-cold round trips for each chain. / チェーンごとのウォームアップ後の低温端（逆温度 1）→ 高温端 → 低温端の往復回数を取得します。 |
| `Matrix<double> SwapAcceptanceRates { get; }` | Gets the post-warmup mean swap acceptance probability between adjacent temperatures `(i, i + 1)` for each chain, with shape `[(temperatures − 1) × chains]`. / チェーンごとのウォームアップ後の隣接温度間 `(i, i + 1)` の平均交換受理確率（形状 `[(温度数 − 1) × チェーン数]`）を取得します。 |

### `class ParameterSummary`

Represents posterior summary statistics and convergence diagnostics for a single parameter (one row of `PosteriorSummary`). / 単一パラメータの事後分布要約統計量および収束診断を表します（`PosteriorSummary` の 1 行）。

- 継承・実装: `IEquatable<ParameterSummary>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `ParameterSummary(string Name, double Mean, double StandardDeviation, double MeanStandardError, double Quantile5, double Median, double Quantile95, double BulkEffectiveSampleSize, double TailEffectiveSampleSize, double RHat)` | Represents posterior summary statistics and convergence diagnostics for a single parameter (one row of `PosteriorSummary`). / 単一パラメータの事後分布要約統計量および収束診断を表します（`PosteriorSummary` の 1 行）。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double BulkEffectiveSampleSize { get; set; }` | The bulk effective sample size (`BulkEffectiveSampleSize`). / バルクの有効標本数（`BulkEffectiveSampleSize`）。 |
| `double Mean { get; set; }` | The sample mean pooled across all chains. / 全チェーンを統合した標本平均。 |
| `double MeanStandardError { get; set; }` | The Monte Carlo standard error of the mean (`MeanStandardError`). / 平均のモンテカルロ標準誤差（`MeanStandardError`）。 |
| `double Median { get; set; }` | The median (50th percentile). / 中央値（50% 分位点）。 |
| `string Name { get; set; }` | The parameter name. / パラメータ名。 |
| `double Quantile5 { get; set; }` | The 5th percentile (`Quantile`). / 5% 分位点（`Quantile`）。 |
| `double Quantile95 { get; set; }` | The 95th percentile. / 95% 分位点。 |
| `double RHat { get; set; }` | The rank-normalized split-`R̂` (`RHat`). / 順位正規化した分割チェーンの `R̂`（`RHat`）。 |
| `double StandardDeviation { get; set; }` | The unbiased sample standard deviation pooled across all chains (divisor `N − 1`). / 全チェーンを統合した不偏標本標準偏差（分母 `N − 1`）。 |
| `double TailEffectiveSampleSize { get; set; }` | The tail effective sample size (`TailEffectiveSampleSize`). / 裾の有効標本数（`TailEffectiveSampleSize`）。 |

### `class PosteriorSummary`

Represents a table of per-parameter posterior summary statistics and convergence diagnostics. / パラメータごとの事後分布要約統計量と収束診断の表を表します。

- 継承・実装: `IEnumerable`, `IEnumerable<ParameterSummary>`, `IReadOnlyCollection<ParameterSummary>`, `IReadOnlyList<ParameterSummary>`

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `ParameterSummary this[int index] { get; }` | Gets the summary row for the parameter at the specified index. / 指定したインデックスのパラメータの要約行を取得します。 |
| `int Count { get; }` | Gets the number of parameters. / パラメータ数を取得します。 |
| `double MaxRHat { get; }` | Gets the maximum `R̂` across all parameters, ignoring `NaN`. / 全パラメータにわたる `R̂` の最大値を取得します（`NaN` は除外）。 |
| `double MinEffectiveSampleSize { get; }` | Gets the minimum of bulk and tail effective sample sizes across all parameters, ignoring `NaN`. / 全パラメータにわたるバルクおよび裾の有効標本数の最小値を取得します（`NaN` は除外）。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static PosteriorSummary Compute(Tensor<double> draws, IReadOnlyList<string> names = null)` | Computes per-parameter posterior summary statistics and convergence diagnostics from the specified draws tensor. / 指定した標本テンソルからパラメータごとの事後分布要約と収束診断を計算します。 |
| `IEnumerator<ParameterSummary> GetEnumerator()` |  |
| `string ToString()` | Formats the posterior summary as an aligned text table. / 事後分布の要約を列を揃えたテキスト表として書式設定します。 |

### `readonly struct PotentialLogDensity<TObjective>`

Wraps a potential objective `U(θ) = −ln π(θ)` as an `IDifferentiableLogDensity` returning `−U(θ)`. Created via `FromPotential`. / ポテンシャル目的関数 `U(θ) = −ln π(θ)` を対数密度 `−U(θ)` として扱うラッパーです。`FromPotential` で生成します。

- 継承・実装: `IDifferentiableLogDensity`, `ILogDensity`

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double LogDensity(ReadOnlyVectorSpan<double> theta)` | Evaluates the log-density at the specified point. / 指定した点の対数密度を評価します。 |
| `double LogDensity(ReadOnlyVectorSpan<double> theta, VectorSpan<double> gradient)` | Evaluates the log-density and its gradient at the specified point. / 指定した点の対数密度と勾配を評価します。 |

### `class RandomWalkMetropolis`

Samples from a log-density using an adaptive Gaussian random-walk Metropolis algorithm. / 正規分布の提案を用いる適応ランダムウォーク・メトロポリス法により、対数密度からサンプリングします。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `RandomWalkMetropolis(RandomWalkMetropolisOptions options = null)` | Initializes a new instance of `RandomWalkMetropolis` with the specified options. / 指定した設定で `RandomWalkMetropolis` の新しいインスタンスを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Chains { get; }` | Gets the number of independent chains. / 独立したチェーンの数を取得します。 |
| `ulong Seed { get; }` | Gets the random seed used for sampling (configured value or obtained from the OS). / サンプリングに使用する乱数シード（指定値、または省略時に OS から取得した値）を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `RandomWalkMetropolisResult Sample(Func<ReadOnlyVectorSpan<double>, double> logDensity, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)` | Generates posterior draws starting all chains from the specified initial point. / すべてのチェーンを指定した初期点から開始して標本を生成します。 |
| `RandomWalkMetropolisResult Sample(Func<ReadOnlyVectorSpan<double>, double> logDensity, ReadOnlyMatrixSpan<double> initialPoints, CancellationToken cancellationToken = default)` | Generates posterior draws starting each chain from its corresponding column of `initialPoints`. / チェーンごとに指定した初期点から標本を生成します。 |
| `RandomWalkMetropolisResult Sample<TDensity>(TDensity density, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)` | Generates posterior draws starting all chains from the specified initial point. / すべてのチェーンを指定した初期点から開始して標本を生成します。 |
| `RandomWalkMetropolisResult Sample<TDensity>(TDensity density, ReadOnlyMatrixSpan<double> initialPoints, CancellationToken cancellationToken = default)` | Generates posterior draws starting each chain from its corresponding column of `initialPoints`. / チェーンごとに指定した初期点から標本を生成します。 |

### `class RandomWalkMetropolisOptions`

Configures `RandomWalkMetropolis`. / `RandomWalkMetropolis` の設定オプションを表します。

- 継承・実装: `MultiChainSamplerOptions`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `RandomWalkMetropolisOptions()` |  |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `bool AdaptCovariance { get; set; }` | Gets or initializes a value indicating whether the proposal shape is updated from regularized sample covariance at the end of each warmup window (default is `true`). / ウォームアップの各ウィンドウ終端で、正則化した標本共分散から提案分布の形状を更新するかどうかを取得または設定します（既定値 `true`）。 |
| `Matrix<double> InitialCovariance { get; set; }` | Gets or initializes the initial proposal shape matrix `C` (`d × d` symmetric positive definite; defaults to the identity matrix when `null`). Proposals are drawn as `θ + s·L·z` where `C = LLᵀ`, giving covariance `s²C`. / 提案分布の初期形状行列 `C` を取得または設定します（`d × d` の対称正定値行列。`null` の既定値は単位行列）。提案は `θ + s·L·z`（`C = LLᵀ`）で生成され、提案共分散は `s²C` です。 |
| `double? ProposalScale { get; set; }` | Gets or initializes the initial proposal scale factor `s` (positive finite value; defaults to `2.38 / √d` when `null`). Used unchanged when `Warmup` is 0. / 提案分布の初期スケール係数 `s` を取得または設定します（正の有限値。`null` の既定値は `2.38 / √d`）。ウォームアップなしの場合はこの値がそのまま使用されます。 |
| `double? TargetAcceptance { get; set; }` | Gets or initializes the target acceptance rate during warmup (in `(0, 1)`). / ウォームアップ期間中の目標受理率を取得または設定します（`(0, 1)`）。 |
| `WarmupWindowOptions Windows { get; set; }` | Gets or initializes the warmup adaptation window schedule. / ウォームアップ期間の適応ウィンドウ設定を取得または設定します。 |

### `class RandomWalkMetropolisResult`

Represents the draws and proposal diagnostics produced by `RandomWalkMetropolis`. / `RandomWalkMetropolis` のサンプリング結果と提案分布の診断情報を表します。

- 継承・実装: `SamplingResult`

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `IReadOnlyList<double> AcceptanceRates { get; }` | Gets the post-warmup proposal acceptance rate for each chain. / チェーンごとのウォームアップ後の提案受理率を取得します。 |
| `Tensor<double> ProposalCovariances { get; }` | Gets the post-warmup proposal covariance `s²C` for each chain with shape `[dimension, dimension, chain]`. / チェーンごとのウォームアップ後の提案共分散 `s²C`（形状 `[次元, 次元, チェーン]`）を取得します。 |
| `IReadOnlyList<double> ProposalScales { get; }` | Gets the fixed post-warmup proposal scale factor `s` for each chain. / チェーンごとのウォームアップ後に固定された提案スケール係数 `s` を取得します。 |

### `abstract class SamplerOptions`

Provides common configuration options for MCMC samplers. / MCMC サンプラーに共通の設定オプションを提供します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Draws { get; set; }` | Gets or initializes the number of retained draws per chain (1 or greater; default is 1000). / チェーンごとに保持する標本数を取得または設定します（1 以上、既定値 1000）。 |
| `int MaxDegreeOfParallelism { get; set; }` | Gets or initializes the maximum number of chains (or walkers in `EnsembleSampler`) evaluated concurrently (1 or greater, or -1 for unbounded; default is 1). / 同時に実行するチェーン数（`EnsembleSampler` では同時評価するウォーカー数）の上限を取得または設定します（1 以上、または無制限の -1。既定値 1）。 |
| `ulong? Seed { get; set; }` | Gets or initializes the random number generator seed. If `null` (default), a seed is obtained from the OS and exposed via `Seed`. / 乱数シードを取得または設定します。`null`（既定）の場合は OS からシードを取得し、`Seed` で参照できます。 |
| `int Thin { get; set; }` | Gets or initializes the thinning interval (1 or greater; default is 1). After warmup, one draw is kept every `Thin` transitions. / 間引き間隔を取得または設定します（1 以上、既定値 1）。ウォームアップ後、`Thin` 回の遷移ごとに 1 標本を保持します。 |
| `int Warmup { get; set; }` | Gets or initializes the number of warmup transitions per chain (0 or greater; default is 1000). / チェーンごとのウォームアップ遷移回数を取得または設定します（0 以上、既定値 1000）。 |

### `class SamplingResult`

Represents the post-warmup draws and evaluation statistics produced by an MCMC sampler. / MCMC サンプラーによって生成されたウォームアップ後の標本と評価統計量を表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int ChainCount { get; }` | Gets the number of chains (or walkers in `EnsembleSampler`). / チェーン数（`EnsembleSampler` ではウォーカー数）を取得します。 |
| `int Dimension { get; }` | Gets the parameter dimension. / パラメータの次元を取得します。 |
| `int DrawCount { get; }` | Gets the number of retained draws per chain. / チェーンごとの保持標本数を取得します。 |
| `Tensor<double> Draws { get; }` | Gets the post-warmup draws tensor with shape `[dimension, draw, chain]`. / ウォームアップ後に保持された標本テンソル（形状 `[次元, 反復, チェーン]`）を取得します。 |
| `IReadOnlyList<long> Evaluations { get; }` | Gets the total number of log-density evaluations (including gradient evaluations and warmup) for each chain. / チェーンごとの対数密度評価回数（勾配評価およびウォームアップ期間を含む）を取得します。 |
| `Matrix<double> LogDensities { get; }` | Gets the log-density evaluated at each retained draw, with shape `[draw × chain]`. / 保持された各標本における対数密度行列（形状 `[反復 × チェーン]`）を取得します。 |
| `ulong Seed { get; }` | Gets the random seed used for sampling (either configured or obtained from the OS). / サンプリングに使用した乱数シード（指定値、または省略時に OS から取得した値）を取得します。 |
| `long TotalEvaluations { get; }` | Gets the total number of log-density evaluations summed across all chains. / 全チェーン合計の対数密度評価回数を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `MatrixSpan<double> Chain(int chain)` | Returns a matrix view of all draws for the specified chain. / 指定したチェーンの標本に対する行列ビューを取得します。 |
| `Matrix<double> Covariance()` | Computes the unbiased sample covariance matrix pooled across all chains. / 全チェーンを統合した標本の不偏共分散行列を計算します。 |
| `Vector<double> Mean()` | Computes the sample mean pooled across all chains. / 全チェーンを統合した標本の平均ベクトルを計算します。 |
| `ReadOnlyMatrixSpan<double> Parameter(int index)` | Returns a read-only view of all draws for the specified parameter. / 指定したパラメータの標本に対する読み取り専用ビューを取得します。 |
| `PosteriorSummary Summarize(IReadOnlyList<string> names = null)` | Computes per-parameter posterior summary statistics and convergence diagnostics. / 標本からパラメータごとの事後分布の要約統計量と収束診断を計算します。 |
| `SamplingResult Transform<TBijector>(TBijector bijector)` | Maps all draws into the constrained parameter space using the specified bijector. / 指定した制約変換を用いて、すべての標本を制約付きパラメータ空間へ写像します。 |

### `class SimplexBijector`

Maps `ℝ^(K−1)` onto the `K`-component probability simplex via stick-breaking. / 棒折り変換により `ℝ^(K−1)` を `K` 成分の確率単体へ写像します。

- 継承・実装: `IBijector`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `SimplexBijector(int size)` | Initializes a new instance of `SimplexBijector` for a simplex of the specified size. / 指定した成分数の単体に対する `SimplexBijector` の新しいインスタンスを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int ConstrainedDimension { get; }` | Gets the dimension (number of coordinates) of the constrained space. / 制約付き空間の次元（成分数）を取得します。 |
| `int UnconstrainedDimension { get; }` | Gets the dimension of the unconstrained space. / 制約のない空間の次元を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Forward(ReadOnlyVectorSpan<double> unconstrained, VectorSpan<double> constrained)` | Maps an unconstrained point to the constrained space and returns the log absolute Jacobian determinant. / 制約のない点を制約付き空間へ写像し、ヤコビアン行列式の絶対値の自然対数を返します。 |
| `void Inverse(ReadOnlyVectorSpan<double> constrained, VectorSpan<double> unconstrained)` | Maps a constrained point back to the unconstrained space. / 制約付きの点を制約のない空間へ逆写像します。 |
| `void PullBack(ReadOnlyVectorSpan<double> unconstrained, ReadOnlyVectorSpan<double> constrainedGradient, VectorSpan<double> unconstrainedGradient)` | Pulls a constrained-space log-density gradient back to the unconstrained space, including the Jacobian gradient term. / 制約付き空間の勾配にヤコビアン項の勾配を加えて制約のない空間へ引き戻します。 |

### `class SliceSampler`

Samples from a log-density using orthogonal stepping-out and shrinkage slice sampling. / 直交する方向に沿って区間を拡張・縮小するスライスサンプラーにより、対数密度からサンプリングします。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `SliceSampler(SliceSamplerOptions options = null)` | Initializes a new instance of `SliceSampler` with the specified options. / 指定した設定で `SliceSampler` の新しいインスタンスを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Chains { get; }` | Gets the number of independent chains. / 独立したチェーンの数を取得します。 |
| `ulong Seed { get; }` | Gets the random seed used for sampling (configured value or obtained from the OS). / サンプリングに使用する乱数シード（指定値、または省略時に OS から取得した値）を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `SliceSamplingResult Sample(Func<ReadOnlyVectorSpan<double>, double> logDensity, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)` | Generates posterior draws starting all chains from the specified initial point. / すべてのチェーンを指定した初期点から開始して標本を生成します。 |
| `SliceSamplingResult Sample(Func<ReadOnlyVectorSpan<double>, double> logDensity, ReadOnlyMatrixSpan<double> initialPoints, CancellationToken cancellationToken = default)` | Generates posterior draws starting each chain from its corresponding column of `initialPoints`. / チェーンごとに指定した初期点から標本を生成します。 |
| `SliceSamplingResult Sample<TDensity>(TDensity density, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)` | Generates posterior draws starting all chains from the specified initial point. / すべてのチェーンを指定した初期点から開始して標本を生成します。 |
| `SliceSamplingResult Sample<TDensity>(TDensity density, ReadOnlyMatrixSpan<double> initialPoints, CancellationToken cancellationToken = default)` | Generates posterior draws starting each chain from its corresponding column of `initialPoints`. / チェーンごとに指定した初期点から標本を生成します。 |

### `class SliceSamplerOptions`

Configures `SliceSampler`. / `SliceSampler` の設定オプションを表します。

- 継承・実装: `MultiChainSamplerOptions`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `SliceSamplerOptions()` |  |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `bool AdaptDirections { get; set; }` | Gets or initializes a value indicating whether orthogonal slice update directions are adapted during warmup (default is `true`). / ウォームアップ期間中にスライスの直交更新方向を適応させるかどうかを取得または設定します（既定値 `true`）。 |
| `bool AdaptWidths { get; set; }` | Gets or initializes a value indicating whether slice widths are adapted to `3` times the windowed sample standard deviation along each direction at the end of each warmup window (default is `true`). / ウォームアップの各ウィンドウ終端で、各方向のスライス幅を標本標準偏差の 3 倍へ更新するかどうかを取得または設定します（既定値 `true`）。 |
| `double InitialWidth { get; set; }` | Gets or initializes the initial slice bracket width `w` along each direction (positive finite value; default is 1). Used until the end of the first warmup window when adapting widths. / 各方向の初期スライス幅 `w` を取得または設定します（正の有限値、既定値 1）。幅を適応させる場合も最初のウィンドウ終了まではこの値が使用されます。 |
| `int MaxSteppingOut { get; set; }` | Gets or initializes the maximum number of stepping-out expansions `m` per 1D slice update (1 or greater; default is 100, per Neal 2003 Fig. 3). Bounds the interval to `m·w` so flat densities still terminate. / 1 次元スライス更新あたりのステップアウト拡張回数の上限 `m` を取得または設定します（1 以上、既定値 100、Neal 2003 Fig. 3）。区間幅を `m·w` 以内に制限し、平坦な密度でも停止するようにします。 |
| `WarmupWindowOptions Windows { get; set; }` | Gets or initializes the warmup adaptation window schedule. / ウォームアップ期間の適応ウィンドウ設定を取得または設定します。 |

### `class SliceSamplingResult`

Represents the draws and adapted slice geometry produced by `SliceSampler`. / `SliceSampler` のサンプリング結果と適応後のスライス方向・幅を表します。

- 継承・実装: `SamplingResult`

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `Tensor<double> Directions { get; }` | Gets the post-warmup orthonormal slice update directions for each chain, with shape `[dimension, dimension, chain]` (each column is one unit direction). / チェーンごとのウォームアップ後に固定された正規直交スライス方向（形状 `[次元, 次元, チェーン]`、各列が 1 つの単位方向ベクトル）を取得します。 |
| `Matrix<double> Widths { get; }` | Gets the post-warmup slice bracket widths along each direction for each chain, with shape `[dimension × chain]`. / チェーンごとのウォームアップ後に固定された各方向のスライス幅（形状 `[次元 × チェーン]`）を取得します。 |

### `readonly struct TransformedDifferentiableLogDensity<TDensity, TBijector>`

Wraps a constrained-space differentiable log-density with an `IBijector` to form an unconstrained `IDifferentiableLogDensity`. / 制約付き空間の微分可能な対数密度に `IBijector` を適用し、制約なし空間の `IDifferentiableLogDensity` を構成します。

- 継承・実装: `IDifferentiableLogDensity`, `ILogDensity`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `TransformedDifferentiableLogDensity(TDensity density, TBijector bijector)` | Initializes a new instance of `TransformedDifferentiableLogDensity`. / `TransformedDifferentiableLogDensity` の新しいインスタンスを初期化します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double LogDensity(ReadOnlyVectorSpan<double> theta)` | Evaluates the log-density at the specified point. / 指定した点の対数密度を評価します。 |
| `double LogDensity(ReadOnlyVectorSpan<double> theta, VectorSpan<double> gradient)` | Evaluates the log-density and its gradient at the specified point. / 指定した点の対数密度と勾配を評価します。 |

### `readonly struct TransformedLogDensity<TDensity, TBijector>`

Wraps a constrained-space log-density with an `IBijector` to form the unconstrained log-density `ln p(f(y)) + ln|det J_f(y)|`. / 制約付き空間の対数密度に `IBijector` を適用し、制約なし空間の対数密度 `ln p(f(y)) + ln|det J_f(y)|` を構成します。

- 継承・実装: `ILogDensity`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `TransformedLogDensity(TDensity density, TBijector bijector)` | Initializes a new instance of `TransformedLogDensity`. / `TransformedLogDensity` の新しいインスタンスを初期化します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double LogDensity(ReadOnlyVectorSpan<double> theta)` | Evaluates the log-density at the specified point. / 指定した点の対数密度を評価します。 |

### `class WarmupWindowOptions`

Configures the windowed warmup adaptation schedule used by `NoUTurnSampler`, `HamiltonianMonteCarlo`, `RandomWalkMetropolis`, `SliceSampler`, and `ParallelTempering`. / `NoUTurnSampler`、`HamiltonianMonteCarlo`、`RandomWalkMetropolis`、`SliceSampler`、および `ParallelTempering` で使用するウォームアップ適応ウィンドウの区切りを設定します。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `WarmupWindowOptions()` |  |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int BaseWindow { get; set; }` | Gets or initializes the length of the first adaptation window in transitions (1 or greater; default is 25). Subsequent windows double in length. / 最初の適応ウィンドウの遷移回数を取得または設定します（1 以上、既定値 25）。以降のウィンドウは倍々に長くなります。 |
| `int InitialBuffer { get; set; }` | Gets or initializes the length of the initial buffer in transitions (0 or greater; default is 75). / 初期バッファの遷移回数を取得または設定します（0 以上、既定値 75）。 |
| `int TerminalBuffer { get; set; }` | Gets or initializes the length of the terminal buffer in transitions (0 or greater; default is 50). / 終端バッファの遷移回数を取得または設定します（0 以上、既定値 50）。 |

