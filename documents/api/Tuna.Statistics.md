# Tuna.Statistics API リファレンス

> このファイルは `tools/ApiDocGenerator` によって自動生成されています。手動で編集せず、`dotnet build Tuna.slnx -c Release` の後に再生成してください。
>
> `(+N overloads)` は `Vector` / `VectorSpan` / `ReadOnlyVectorSpan` 等のコンテナ・ビュー組み合わせや `double` / `float` 固定型オーバーロードの省略を表します。

## 公開型一覧

- `Tuna.Statistics.Distributions` (42 型): [`BernoulliDistribution`](#readonly-struct-bernoullidistribution), [`BetaDistribution`](#readonly-struct-betadistribution), [`BinomialDistribution`](#readonly-struct-binomialdistribution), [`CategoricalDistribution`](#class-categoricaldistribution), [`CauchyDistribution`](#readonly-struct-cauchydistribution), [`ChiSquaredDistribution`](#readonly-struct-chisquareddistribution), [`DirichletDistribution`](#class-dirichletdistribution), [`DirichletMultinomialDistribution`](#class-dirichletmultinomialdistribution), [`DiscreteUniformDistribution`](#readonly-struct-discreteuniformdistribution), [`DistributionParameterOutOfRangeException`](#class-distributionparameteroutofrangeexception), [`ExponentialDistribution`](#readonly-struct-exponentialdistribution), [`FDistribution`](#readonly-struct-fdistribution), [`GammaDistribution`](#readonly-struct-gammadistribution), [`GeometricDistribution`](#readonly-struct-geometricdistribution), [`GumbelDistribution`](#readonly-struct-gumbeldistribution), [`IContinuousDistribution`](#interface-icontinuousdistribution), [`IDiscreteDistribution`](#interface-idiscretedistribution), [`IMatrixContinuousDistribution`](#interface-imatrixcontinuousdistribution), [`IMultivariateContinuousDistribution`](#interface-imultivariatecontinuousdistribution), [`IMultivariateDiscreteDistribution`](#interface-imultivariatediscretedistribution), [`IUnivariateDistribution<TValue>`](#interface-iunivariatedistributiontvalue), [`InverseGammaDistribution`](#readonly-struct-inversegammadistribution), [`InverseWishartDistribution`](#class-inversewishartdistribution), [`LaplaceDistribution`](#readonly-struct-laplacedistribution), [`LkjCorrelationDistribution`](#class-lkjcorrelationdistribution), [`LocationScaleDistribution<TDistribution>`](#readonly-struct-locationscaledistributiontdistribution), [`LogNormalDistribution`](#readonly-struct-lognormaldistribution), [`LogisticDistribution`](#readonly-struct-logisticdistribution), [`MatrixNormalDistribution`](#class-matrixnormaldistribution), [`MultinomialDistribution`](#class-multinomialdistribution), [`MultivariateDistributionExtensions`](#static-class-multivariatedistributionextensions), [`MultivariateNormalDistribution`](#class-multivariatenormaldistribution), [`MultivariateStudentTDistribution`](#class-multivariatestudenttdistribution), [`NegativeBinomialDistribution`](#readonly-struct-negativebinomialdistribution), [`NegativeMultinomialDistribution`](#class-negativemultinomialdistribution), [`NormalDistribution`](#readonly-struct-normaldistribution), [`PoissonDistribution`](#readonly-struct-poissondistribution), [`StudentTDistribution`](#readonly-struct-studenttdistribution), [`TruncatedDistribution<TDistribution>`](#readonly-struct-truncateddistributiontdistribution), [`UniformDistribution`](#readonly-struct-uniformdistribution), [`WeibullDistribution`](#readonly-struct-weibulldistribution), [`WishartDistribution`](#class-wishartdistribution)

## Tuna.Statistics.Distributions

### `readonly struct BernoulliDistribution`

Represents the Bernoulli distribution taking value 1 with probability `p` and 0 with probability `1 - p`. / 確率 `p` で 1、確率 `1 - p` で 0 をとるベルヌーイ分布を表します。

- 継承・実装: `IDiscreteDistribution`, `IUnivariateDistribution<long>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `BernoulliDistribution(double probability)` | Initializes a new instance of `BernoulliDistribution` with the specified success probability. / 成功確率を指定して `BernoulliDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `long Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `double Probability { get; }` | Gets the success probability `p`. / 成功確率 `p` を取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `long SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `long SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(long k)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ k)`. / 累積分布関数（CDF）`P(X ≤ k)` を評価します。 |
| `long InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `long InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `long InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `long InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(long k)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ k)`. / 累積分布関数の自然対数 `ln P(X ≤ k)` を評価します。 |
| `static ScalarNode LogCdf(ScalarNode probability, long k)` | Records the Bernoulli log cumulative probability and its derivative with respect to the success probability. / ベルヌーイ分布の対数累積確率と成功確率に関する微分を記録します。 |
| `double LogPmf(long k)` | Evaluates the natural logarithm of the probability mass function at the specified integer. / 指定した整数点で確率質量関数の自然対数を評価します。 |
| `static ScalarNode LogPmf(ScalarNode probability, long k)` | Records the Bernoulli log probability mass and its derivative with respect to the success probability. / ベルヌーイ分布の対数確率質量と成功確率に関する微分を記録します。 |
| `static ScalarNode LogPmf(ScalarNode probability, ReadOnlyVectorSpan<long> observations)` | Records the Bernoulli log likelihood of independent observations as a single operation. / ベルヌーイ分布に従う独立な観測値の対数尤度を 1 回の演算として記録します。 |
| `double LogSf(long k)` | Evaluates the natural logarithm of the survival function, `ln P(X > k)`. / 生存関数の自然対数 `ln P(X > k)` を評価します。 |
| `static ScalarNode LogSf(ScalarNode probability, long k)` | Records the Bernoulli log survival probability and its derivative with respect to the success probability. / ベルヌーイ分布の対数上側確率と成功確率に関する微分を記録します。 |
| `double Pmf(long k)` | Evaluates the probability mass function (PMF) at the specified integer. / 指定した整数点で確率質量関数（PMF）を評価します。 |
| `long Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<long> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(long k)` | Evaluates the survival function, `P(X > k)`. / 生存関数（上側累積確率）`P(X > k)` を評価します。 |

### `readonly struct BetaDistribution`

Represents the beta distribution on `[0, 1]` with shape parameters `α` and `β`. / 形状パラメーター `α`、`β` を持つ `[0, 1]` 上のベータ分布を表します。

- 継承・実装: `IContinuousDistribution`, `IUnivariateDistribution<double>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `BetaDistribution(double alpha, double beta)` | Initializes a new instance of `BetaDistribution` with the specified shape parameters. / 2 つの形状パラメーターを指定して `BetaDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Alpha { get; }` | Gets the first shape parameter `α`. / 第 1 形状パラメーター `α` を取得します。 |
| `double Beta { get; }` | Gets the second shape parameter `β`. / 第 2 形状パラメーター `β` を取得します。 |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `double Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `double SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `double SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(double x)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ x)`. / 累積分布関数（CDF）`P(X ≤ x)` を評価します。 |
| `double InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `double InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `double InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `double InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(double x)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ x)`. / 累積分布関数の自然対数 `ln P(X ≤ x)` を評価します。 |
| `static ScalarNode LogCdf(ScalarNode alpha, ScalarNode beta, ScalarNode x)` | Records the beta log cumulative probability and its derivative with respect to the variate. / ベータ分布の対数累積確率と評価点に関する微分を記録します。 |
| `double LogPdf(double x)` | Evaluates the natural logarithm of the probability density function at the specified point. / 指定した点で確率密度関数の自然対数を評価します。 |
| `static ScalarNode LogPdf(ScalarNode alpha, ScalarNode beta, ScalarNode x)` | Records the beta log density and its derivatives. / ベータ分布の対数密度とその微分を記録します。 |
| `static ScalarNode LogPdf(ScalarNode alpha, ScalarNode beta, ReadOnlyVectorSpan<double> observations)` | Records the beta log likelihood of independent observations as a single operation. / ベータ分布に従う独立な観測値の対数尤度を 1 回の演算として記録します。 |
| `double LogSf(double x)` | Evaluates the natural logarithm of the survival function, `ln P(X > x)`. / 生存関数の自然対数 `ln P(X > x)` を評価します。 |
| `static ScalarNode LogSf(ScalarNode alpha, ScalarNode beta, ScalarNode x)` | Records the beta log survival probability and its derivative with respect to the variate. / ベータ分布の対数生存確率と評価点に関する微分を記録します。 |
| `double Pdf(double x)` | Evaluates the probability density function (PDF) at the specified point. / 指定した点で確率密度関数（PDF）を評価します。 |
| `double Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<double> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(double x)` | Evaluates the survival function (complementary CDF), `P(X > x)`. / 生存関数（上側累積確率）`P(X > x)` を評価します。 |

### `readonly struct BinomialDistribution`

Represents the binomial distribution `B(n, p)` for the number of successes in `n` independent Bernoulli trials. / `n` 回の独立なベルヌーイ試行における成功回数の二項分布 `B(n, p)` を表します。

- 継承・実装: `IDiscreteDistribution`, `IUnivariateDistribution<long>`

**フィールド・定数**

| メンバー | 説明 |
| --- | --- |
| `const long MaxTrials` | The maximum supported number of trials (`2⁵³`, the largest exact integer range in `Double`). / サポートされる試行回数の上限（`2⁵³`、`Double` で正確に表現可能な整数範囲）。 |

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `BinomialDistribution(long trials, double probability)` | Initializes a new instance of `BinomialDistribution` with the specified trial count and success probability. / 試行回数と成功確率を指定して `BinomialDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `long Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `double Probability { get; }` | Gets the success probability `p`. / 成功確率 `p` を取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `long SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `long SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `long Trials { get; }` | Gets the number of trials `n`. / 試行回数 `n` を取得します。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(long k)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ k)`. / 累積分布関数（CDF）`P(X ≤ k)` を評価します。 |
| `long InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `long InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `long InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `long InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(long k)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ k)`. / 累積分布関数の自然対数 `ln P(X ≤ k)` を評価します。 |
| `static ScalarNode LogCdf(long trials, ScalarNode probability, long k)` | Records the binomial log cumulative probability and its derivatives. / 二項分布の対数累積確率とその微分を記録します。 |
| `double LogPmf(long k)` | Evaluates the natural logarithm of the probability mass function at the specified integer. / 指定した整数点で確率質量関数の自然対数を評価します。 |
| `static ScalarNode LogPmf(long trials, ScalarNode probability, long k)` | Records the binomial log mass and its derivatives. / 二項分布の対数質量とその微分を記録します。 |
| `static ScalarNode LogPmf(long trials, ScalarNode probability, ReadOnlyVectorSpan<long> observations)` | Records the binomial log likelihood of independent observations as a single operation. / 二項分布の対数尤度を 1 回の演算として記録します。 |
| `double LogSf(long k)` | Evaluates the natural logarithm of the survival function, `ln P(X > k)`. / 生存関数の自然対数 `ln P(X > k)` を評価します。 |
| `static ScalarNode LogSf(long trials, ScalarNode probability, long k)` | Records the binomial log survival probability and its derivatives. / 二項分布の対数生存確率とその微分を記録します。 |
| `double Pmf(long k)` | Evaluates the probability mass function (PMF) at the specified integer. / 指定した整数点で確率質量関数（PMF）を評価します。 |
| `long Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<long> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(long k)` | Evaluates the survival function, `P(X > k)`. / 生存関数（上側累積確率）`P(X > k)` を評価します。 |

### `class CategoricalDistribution`

Represents the categorical distribution over `0, 1, …, K - 1` with probabilities proportional to the given weights. / 与えられた重みに比例する確率で `0, 1, …, K - 1` の値をとるカテゴリカル分布を表します。

- 継承・実装: `IDiscreteDistribution`, `IUnivariateDistribution<long>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `CategoricalDistribution(ReadOnlySpan<double> weights)` | Initializes a new instance of `CategoricalDistribution` from non-negative category weights. / 非負の重みを指定して `CategoricalDistribution` を初期化します。確率は重みを総和で正規化した値になります。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int CategoryCount { get; }` | Gets the number of categories `K`. / カテゴリ数 `K` を取得します。 |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `long Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `ReadOnlySpan<double> Probabilities { get; }` | Gets a read-only span of the normalized category probabilities. / 正規化された各カテゴリの確率の読み取り専用スパンを取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `long SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `long SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(long k)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ k)`. / 累積分布関数（CDF）`P(X ≤ k)` を評価します。 |
| `static CategoricalDistribution FromLogWeights(ReadOnlySpan<double> logWeights)` | Creates a `CategoricalDistribution` from log-weights without intermediate overflow or underflow. / 重みの自然対数から `CategoricalDistribution` を生成します（対数が極端に大きい場合や小さい場合でもオーバーフローしません）。 |
| `long InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `long InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `long InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `long InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(long k)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ k)`. / 累積分布関数の自然対数 `ln P(X ≤ k)` を評価します。 |
| `static ScalarNode LogCdf(ReadOnlySpan<ScalarNode> weights, long k)` | Records the categorical log cumulative probability and its weight derivatives. / カテゴリカル分布の対数累積確率と重みに関する微分を記録します。 |
| `double LogPmf(long k)` | Evaluates the natural logarithm of the probability mass function at the specified integer. / 指定した整数点で確率質量関数の自然対数を評価します。 |
| `static ScalarNode LogPmf(ReadOnlySpan<ScalarNode> weights, long k)` | Records the categorical log mass and its weight derivatives. / カテゴリカル分布の対数質量と重みに関する微分を記録します。 |
| `static ScalarNode LogPmf(ReadOnlySpan<ScalarNode> weights, ReadOnlyVectorSpan<long> observations)` | Records the categorical log likelihood of independent observations as a single operation. / カテゴリカル分布の独立な観測値の対数尤度を 1 回の演算として記録します。 |
| `double LogSf(long k)` | Evaluates the natural logarithm of the survival function, `ln P(X > k)`. / 生存関数の自然対数 `ln P(X > k)` を評価します。 |
| `static ScalarNode LogSf(ReadOnlySpan<ScalarNode> weights, long k)` | Records the categorical log survival probability and its weight derivatives. / カテゴリカル分布の対数生存確率と重みに関する微分を記録します。 |
| `double Pmf(long k)` | Evaluates the probability mass function (PMF) at the specified integer. / 指定した整数点で確率質量関数（PMF）を評価します。 |
| `long Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<long> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(long k)` | Evaluates the survival function, `P(X > k)`. / 生存関数（上側累積確率）`P(X > k)` を評価します。 |

### `readonly struct CauchyDistribution`

Represents the Cauchy distribution with location `x₀` and scale `γ`. / 位置 `x₀`、尺度 `γ` を持つコーシー分布を表します。

- 継承・実装: `IContinuousDistribution`, `IUnivariateDistribution<double>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `CauchyDistribution(double location, double scale)` | Initializes a new instance of `CauchyDistribution` with the specified location and scale. / 位置および尺度パラメーターを指定して `CauchyDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `double Location { get; }` | Gets the location parameter `x₀`. / 位置パラメーター `x₀` を取得します。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `double Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `double Scale { get; }` | Gets the scale parameter `γ`. / 尺度パラメーター `γ` を取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `double SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `double SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(double x)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ x)`. / 累積分布関数（CDF）`P(X ≤ x)` を評価します。 |
| `double InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `double InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `double InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `double InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(double x)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ x)`. / 累積分布関数の自然対数 `ln P(X ≤ x)` を評価します。 |
| `double LogPdf(double x)` | Evaluates the natural logarithm of the probability density function at the specified point. / 指定した点で確率密度関数の自然対数を評価します。 |
| `static ScalarNode LogPdf(ScalarNode location, ScalarNode scale, ScalarNode x)` | Records the Cauchy log density and its derivatives with respect to the location, scale, and variate. / コーシー分布の対数密度と、位置・尺度・評価点に関する微分を記録します。 |
| `static ScalarNode LogPdf(ScalarNode location, ScalarNode scale, ReadOnlyVectorSpan<double> observations)` | Records the Cauchy log likelihood of independent observations as a single operation. / コーシー分布に従う独立な観測値の対数尤度を 1 回の演算として記録します。 |
| `double LogSf(double x)` | Evaluates the natural logarithm of the survival function, `ln P(X > x)`. / 生存関数の自然対数 `ln P(X > x)` を評価します。 |
| `double Pdf(double x)` | Evaluates the probability density function (PDF) at the specified point. / 指定した点で確率密度関数（PDF）を評価します。 |
| `double Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<double> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(double x)` | Evaluates the survival function (complementary CDF), `P(X > x)`. / 生存関数（上側累積確率）`P(X > x)` を評価します。 |

### `readonly struct ChiSquaredDistribution`

Represents the chi-squared distribution `χ²(ν)` (equivalent to a gamma distribution with shape `ν/2` and scale 2). / カイ二乗分布 `χ²(ν)`（形状 `ν/2`、尺度 2 のガンマ分布と同値）を表します。

- 継承・実装: `IContinuousDistribution`, `IUnivariateDistribution<double>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `ChiSquaredDistribution(double degreesOfFreedom)` | Initializes a new instance of `ChiSquaredDistribution` with the specified degrees of freedom. / 自由度を指定して `ChiSquaredDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double DegreesOfFreedom { get; }` | Gets the degrees of freedom `ν`. / 自由度 `ν` を取得します。 |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `double Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `double SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `double SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(double x)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ x)`. / 累積分布関数（CDF）`P(X ≤ x)` を評価します。 |
| `double InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `double InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `double InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `double InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(double x)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ x)`. / 累積分布関数の自然対数 `ln P(X ≤ x)` を評価します。 |
| `static ScalarNode LogCdf(ScalarNode degreesOfFreedom, ScalarNode x)` | Records the chi-squared log cumulative probability and its derivatives. / カイ二乗分布の対数累積確率とその微分を記録します。 |
| `double LogPdf(double x)` | Evaluates the natural logarithm of the probability density function at the specified point. / 指定した点で確率密度関数の自然対数を評価します。 |
| `static ScalarNode LogPdf(ScalarNode degreesOfFreedom, ScalarNode x)` | Records the chi-squared log density and its derivatives. / カイ二乗分布の対数密度とその微分を記録します。 |
| `static ScalarNode LogPdf(ScalarNode degreesOfFreedom, ReadOnlyVectorSpan<double> observations)` | Records the chi-squared log likelihood of independent observations as a single operation. / カイ二乗分布に従う独立な観測値の対数尤度を 1 回の演算として記録します。 |
| `double LogSf(double x)` | Evaluates the natural logarithm of the survival function, `ln P(X > x)`. / 生存関数の自然対数 `ln P(X > x)` を評価します。 |
| `static ScalarNode LogSf(ScalarNode degreesOfFreedom, ScalarNode x)` | Records the chi-squared log survival probability and its derivatives. / カイ二乗分布の対数生存確率とその微分を記録します。 |
| `double Pdf(double x)` | Evaluates the probability density function (PDF) at the specified point. / 指定した点で確率密度関数（PDF）を評価します。 |
| `double Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<double> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(double x)` | Evaluates the survival function (complementary CDF), `P(X > x)`. / 生存関数（上側累積確率）`P(X > x)` を評価します。 |

### `class DirichletDistribution`

Represents the Dirichlet distribution `Dir(α)` over the probability simplex. / 確率単体上のディリクレ分布 `Dir(α)` を表します。

- 継承・実装: `IMultivariateContinuousDistribution`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `DirichletDistribution(ReadOnlySpan<double> alpha)` | Initializes a new instance of `DirichletDistribution` with the specified concentration parameters. / 集中度パラメーターを指定して `DirichletDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `ReadOnlySpan<double> Alpha { get; }` | Gets a read-only span of the concentration parameters `α`. / 集中度パラメーター `α` の読み取り専用スパンを取得します。 |
| `double Concentration { get; }` | Gets the sum of the concentration parameters `α₀ = Σᵢ αᵢ`. / 集中度パラメーターの総和 `α₀ = Σᵢ αᵢ` を取得します。 |
| `int Dimension { get; }` | Gets the dimension (length of the random vector). / 次元（確率ベクトルの長さ）を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `Matrix<double> GetCovariance()` | Computes the covariance matrix of the distribution. / 分布の共分散行列を計算します。 |
| `Vector<double> GetMean()` | Computes the mean vector of the distribution. / 分布の平均ベクトルを計算します。 |
| `double LogPdf(ReadOnlyVectorSpan<double> x)` | Evaluates the natural logarithm of the probability density function at the specified vector. / 指定したベクトルで確率密度関数の自然対数を評価します。 |
| `void LogPdf(ReadOnlyMatrixSpan<double> x, VectorSpan<double> destination)` | Evaluates the natural logarithm of the probability density function for each column of the input matrix. / 入力行列の各列を評価点として、確率密度関数の自然対数を計算します。 |
| `static ScalarNode LogPdf(ReadOnlySpan<ScalarNode> alpha, ReadOnlySpan<ScalarNode> x)` | Records the Dirichlet log density and its derivatives. / ディリクレ分布の対数密度とその微分を記録します。 |
| `double Pdf(ReadOnlyVectorSpan<double> x)` | Evaluates the probability density function (PDF) at the specified vector. / 指定したベクトルで確率密度関数（PDF）を評価します。 |
| `void Sample(IRandomSource random, VectorSpan<double> destination)` | Draws a single random vector from the distribution and writes it to the destination. / 分布に従って標本ベクトルを 1 個生成し、出力先へ書き込みます。 |
| `void Sample(IRandomSource random, MatrixSpan<double> destination)` | Draws a single random vector from the distribution and writes it to the destination. / 分布に従って標本ベクトルを 1 個生成し、出力先へ書き込みます。 |

### `class DirichletMultinomialDistribution`

Represents the Dirichlet-multinomial compound distribution over non-negative integer vectors summing to `n`. / 確率ベクトルがディリクレ分布に従う複合分布（ディリクレ多項分布）を表します。

- 継承・実装: `IMultivariateDiscreteDistribution`

**フィールド・定数**

| メンバー | 説明 |
| --- | --- |
| `const long MaxTrials` | The maximum supported number of trials (`2⁵³`, matching `MaxTrials`). / サポートされる試行回数の上限（`2⁵³`、`MaxTrials` と同一）。 |

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `DirichletMultinomialDistribution(long trials, ReadOnlySpan<double> alpha)` | Initializes a new instance of `DirichletMultinomialDistribution` with the specified trial count and concentration parameters. / 試行回数と集中度パラメーターを指定して `DirichletMultinomialDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `ReadOnlySpan<double> Alpha { get; }` | Gets a read-only span of the concentration parameters `α`. / 集中度パラメーター `α` の読み取り専用スパンを取得します。 |
| `double Concentration { get; }` | Gets the sum of the concentration parameters `α₀ = Σᵢ αᵢ`. / 集中度パラメーターの総和 `α₀ = Σᵢ αᵢ` を取得します。 |
| `int Dimension { get; }` | Gets the dimension (length of the random vector). / 次元（確率ベクトルの長さ）を取得します。 |
| `long Trials { get; }` | Gets the number of trials `n`. / 試行回数 `n` を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `Matrix<double> GetCovariance()` | Computes the covariance matrix of the distribution. / 分布の共分散行列を計算します。 |
| `Vector<double> GetMean()` | Computes the mean vector of the distribution. / 分布の平均ベクトルを計算します。 |
| `double LogPmf(ReadOnlyVectorSpan<long> x)` | Evaluates the natural logarithm of the probability mass function at the specified integer vector. / 指定した整数ベクトルで確率質量関数の自然対数を評価します。 |
| `void LogPmf(ReadOnlyMatrixSpan<long> x, VectorSpan<double> destination)` | Evaluates the natural logarithm of the probability mass function for each column of the input matrix. / 入力行列の各列を評価点として、確率質量関数の自然対数を計算します。 |
| `double Pmf(ReadOnlyVectorSpan<long> x)` | Evaluates the probability mass function (PMF) at the specified integer vector. / 指定した整数ベクトルで確率質量関数（PMF）を評価します。 |
| `void Sample(IRandomSource random, VectorSpan<long> destination)` | Draws a single random integer vector from the distribution and writes it to the destination. / 分布に従って整数標本ベクトルを 1 個生成し、出力先へ書き込みます。 |
| `void Sample(IRandomSource random, MatrixSpan<long> destination)` | Draws a single random integer vector from the distribution and writes it to the destination. / 分布に従って整数標本ベクトルを 1 個生成し、出力先へ書き込みます。 |

### `readonly struct DiscreteUniformDistribution`

Represents the discrete uniform distribution over the closed integer range `[a, b]`. / 閉区間 `[a, b]` 上の整数を等確率でとる離散一様分布を表します。

- 継承・実装: `IDiscreteDistribution`, `IUnivariateDistribution<long>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `DiscreteUniformDistribution(long lower, long upper)` | Initializes a new instance of `DiscreteUniformDistribution` over `[lower, upper]`. / 閉区間 `[lower, upper]` を指定して `DiscreteUniformDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `long Lower { get; }` | Gets the inclusive lower bound `a`. / 下端 `a`（含む）を取得します。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `long Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `long SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `long SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `long Upper { get; }` | Gets the inclusive upper bound `b`. / 上端 `b`（含む）を取得します。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(long k)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ k)`. / 累積分布関数（CDF）`P(X ≤ k)` を評価します。 |
| `long InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `long InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `long InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `long InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(long k)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ k)`. / 累積分布関数の自然対数 `ln P(X ≤ k)` を評価します。 |
| `double LogPmf(long k)` | Evaluates the natural logarithm of the probability mass function at the specified integer. / 指定した整数点で確率質量関数の自然対数を評価します。 |
| `double LogSf(long k)` | Evaluates the natural logarithm of the survival function, `ln P(X > k)`. / 生存関数の自然対数 `ln P(X > k)` を評価します。 |
| `double Pmf(long k)` | Evaluates the probability mass function (PMF) at the specified integer. / 指定した整数点で確率質量関数（PMF）を評価します。 |
| `long Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<long> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(long k)` | Evaluates the survival function, `P(X > k)`. / 生存関数（上側累積確率）`P(X > k)` を評価します。 |

### `class DistributionParameterOutOfRangeException`

Indicates that a differentiable distribution parameter is outside its domain. / 自動微分に対応する分布のパラメータが定義域外であることを示す例外です。

- 継承・実装: `ArgumentOutOfRangeException`, `ISerializable`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `DistributionParameterOutOfRangeException(string paramName, object actualValue, string message)` | Initializes an exception for an out-of-domain distribution parameter. / 定義域外の分布パラメータを示す例外を初期化します。 |

### `readonly struct ExponentialDistribution`

Represents the exponential distribution parameterized by rate `λ` on `[0, ∞)`. / 率パラメーター `λ` でパラメーター化された `[0, ∞)` 上の指数分布を表します。

- 継承・実装: `IContinuousDistribution`, `IUnivariateDistribution<double>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `ExponentialDistribution(double rate)` | Initializes a new instance of `ExponentialDistribution` with the specified rate parameter. / 率パラメーターを指定して `ExponentialDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `double Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `double Rate { get; }` | Gets the rate parameter `λ`. / 率パラメーター `λ` を取得します。 |
| `double Scale { get; }` | Gets the scale parameter `1/λ`. / 尺度パラメーター `1/λ` を取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `double SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `double SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(double x)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ x)`. / 累積分布関数（CDF）`P(X ≤ x)` を評価します。 |
| `static ExponentialDistribution FromScale(double scale)` | Creates an `ExponentialDistribution` from its scale (mean) parameter `θ = 1/λ`. / 尺度（平均）パラメーター `θ = 1/λ` を指定して `ExponentialDistribution` を生成します。 |
| `double InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `double InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `double InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `double InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(double x)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ x)`. / 累積分布関数の自然対数 `ln P(X ≤ x)` を評価します。 |
| `static ScalarNode LogCdf(ScalarNode rate, ScalarNode x)` | Records the exponential log cumulative probability and its derivatives. / 指数分布の対数累積確率とその微分を記録します。 |
| `double LogPdf(double x)` | Evaluates the natural logarithm of the probability density function at the specified point. / 指定した点で確率密度関数の自然対数を評価します。 |
| `static ScalarNode LogPdf(ScalarNode rate, ScalarNode x)` | Records the exponential log density and its derivatives. / 指数分布の対数密度とその微分を記録します。 |
| `static ScalarNode LogPdf(ScalarNode rate, ReadOnlyVectorSpan<double> observations)` | Records the exponential log likelihood of independent observations as a single operation. / 指数分布の対数尤度を 1 回の演算として記録します。 |
| `double LogSf(double x)` | Evaluates the natural logarithm of the survival function, `ln P(X > x)`. / 生存関数の自然対数 `ln P(X > x)` を評価します。 |
| `static ScalarNode LogSf(ScalarNode rate, ScalarNode x)` | Records the exponential log survival probability and its derivatives. / 指数分布の対数生存確率とその微分を記録します。 |
| `double Pdf(double x)` | Evaluates the probability density function (PDF) at the specified point. / 指定した点で確率密度関数（PDF）を評価します。 |
| `double Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<double> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(double x)` | Evaluates the survival function (complementary CDF), `P(X > x)`. / 生存関数（上側累積確率）`P(X > x)` を評価します。 |

### `readonly struct FDistribution`

Represents the Fisher–Snedecor F-distribution `F(d₁, d₂)`. / F 分布 `F(d₁, d₂)` を表します。

- 継承・実装: `IContinuousDistribution`, `IUnivariateDistribution<double>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `FDistribution(double numeratorDegreesOfFreedom, double denominatorDegreesOfFreedom)` | Initializes a new instance of `FDistribution` with the specified degrees of freedom. / 分子・分母の自由度を指定して `FDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double DenominatorDegreesOfFreedom { get; }` | Gets the denominator degrees of freedom `d₂`. / 分母の自由度 `d₂` を取得します。 |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `double Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `double NumeratorDegreesOfFreedom { get; }` | Gets the numerator degrees of freedom `d₁`. / 分子の自由度 `d₁` を取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `double SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `double SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(double x)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ x)`. / 累積分布関数（CDF）`P(X ≤ x)` を評価します。 |
| `double InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `double InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `double InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `double InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(double x)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ x)`. / 累積分布関数の自然対数 `ln P(X ≤ x)` を評価します。 |
| `static ScalarNode LogCdf(ScalarNode numeratorDegreesOfFreedom, ScalarNode denominatorDegreesOfFreedom, ScalarNode x)` | Records the F log cumulative probability and its derivatives. / F 分布の対数累積確率とその微分を記録します。 |
| `double LogPdf(double x)` | Evaluates the natural logarithm of the probability density function at the specified point. / 指定した点で確率密度関数の自然対数を評価します。 |
| `static ScalarNode LogPdf(ScalarNode numeratorDegreesOfFreedom, ScalarNode denominatorDegreesOfFreedom, ScalarNode x)` | Records the F log density and its derivatives. / F 分布の対数密度とその微分を記録します。 |
| `static ScalarNode LogPdf(ScalarNode numeratorDegreesOfFreedom, ScalarNode denominatorDegreesOfFreedom, ReadOnlyVectorSpan<double> observations)` | Records the F log likelihood of independent observations as a single operation. / F 分布の対数尤度を 1 回の演算として記録します。 |
| `double LogSf(double x)` | Evaluates the natural logarithm of the survival function, `ln P(X > x)`. / 生存関数の自然対数 `ln P(X > x)` を評価します。 |
| `static ScalarNode LogSf(ScalarNode numeratorDegreesOfFreedom, ScalarNode denominatorDegreesOfFreedom, ScalarNode x)` | Records the F log survival probability and its derivatives. / F 分布の対数生存確率とその微分を記録します。 |
| `double Pdf(double x)` | Evaluates the probability density function (PDF) at the specified point. / 指定した点で確率密度関数（PDF）を評価します。 |
| `double Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<double> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(double x)` | Evaluates the survival function (complementary CDF), `P(X > x)`. / 生存関数（上側累積確率）`P(X > x)` を評価します。 |

### `readonly struct GammaDistribution`

Represents the gamma distribution parameterized by shape `k` and scale `θ` on `[0, ∞)`. / 形状 `k` と尺度 `θ` でパラメーター化された `[0, ∞)` 上のガンマ分布を表します。

- 継承・実装: `IContinuousDistribution`, `IUnivariateDistribution<double>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `GammaDistribution(double shape, double scale)` | Initializes a new instance of `GammaDistribution` with the specified shape and scale. / 形状および尺度パラメーターを指定して `GammaDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `double Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `double Rate { get; }` | Gets the rate parameter `1/θ`. / 率パラメーター `1/θ` を取得します。 |
| `double Scale { get; }` | Gets the scale parameter `θ`. / 尺度パラメーター `θ` を取得します。 |
| `double Shape { get; }` | Gets the shape parameter `k`. / 形状パラメーター `k` を取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `double SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `double SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(double x)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ x)`. / 累積分布関数（CDF）`P(X ≤ x)` を評価します。 |
| `static GammaDistribution FromShapeRate(double shape, double rate)` | Creates a `GammaDistribution` from shape `k` and rate `β = 1/θ`. / 形状 `k` と率 `β = 1/θ` を指定して `GammaDistribution` を生成します。 |
| `double InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `double InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `double InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `double InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(double x)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ x)`. / 累積分布関数の自然対数 `ln P(X ≤ x)` を評価します。 |
| `static ScalarNode LogCdf(ScalarNode shape, ScalarNode scale, ScalarNode x)` | Records the gamma log cumulative probability and its derivatives. / ガンマ分布の対数累積確率とその微分を記録します。 |
| `double LogPdf(double x)` | Evaluates the natural logarithm of the probability density function at the specified point. / 指定した点で確率密度関数の自然対数を評価します。 |
| `static ScalarNode LogPdf(ScalarNode shape, ScalarNode scale, ScalarNode x)` | Records the gamma log density and its derivatives. / ガンマ分布の対数密度とその微分を記録します。 |
| `static ScalarNode LogPdf(ScalarNode shape, ScalarNode scale, ReadOnlyVectorSpan<double> observations)` | Records the gamma log likelihood of independent observations as a single operation. / ガンマ分布に従う独立な観測値の対数尤度を 1 回の演算として記録します。 |
| `double LogSf(double x)` | Evaluates the natural logarithm of the survival function, `ln P(X > x)`. / 生存関数の自然対数 `ln P(X > x)` を評価します。 |
| `static ScalarNode LogSf(ScalarNode shape, ScalarNode scale, ScalarNode x)` | Records the gamma log survival probability and its derivatives. / ガンマ分布の対数生存確率とその微分を記録します。 |
| `double Pdf(double x)` | Evaluates the probability density function (PDF) at the specified point. / 指定した点で確率密度関数（PDF）を評価します。 |
| `double Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<double> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(double x)` | Evaluates the survival function (complementary CDF), `P(X > x)`. / 生存関数（上側累積確率）`P(X > x)` を評価します。 |

### `readonly struct GeometricDistribution`

Represents the geometric distribution for the number of failures before the first success. / 初めて成功するまでの失敗回数を表す幾何分布です。

- 継承・実装: `IDiscreteDistribution`, `IUnivariateDistribution<long>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `GeometricDistribution(double probability)` | Initializes a new instance of `GeometricDistribution` with the specified success probability. / 成功確率を指定して `GeometricDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `long Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `double Probability { get; }` | Gets the per-trial success probability `p`. / 各試行の成功確率 `p` を取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `long SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `long SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(long k)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ k)`. / 累積分布関数（CDF）`P(X ≤ k)` を評価します。 |
| `long InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `long InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `long InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `long InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(long k)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ k)`. / 累積分布関数の自然対数 `ln P(X ≤ k)` を評価します。 |
| `static ScalarNode LogCdf(ScalarNode probability, long k)` | Records the geometric log cumulative probability and its derivatives. / 幾何分布の対数累積確率とその微分を記録します。 |
| `double LogPmf(long k)` | Evaluates the natural logarithm of the probability mass function at the specified integer. / 指定した整数点で確率質量関数の自然対数を評価します。 |
| `static ScalarNode LogPmf(ScalarNode probability, long k)` | Records the geometric log mass and its derivatives. / 幾何分布の対数質量とその微分を記録します。 |
| `static ScalarNode LogPmf(ScalarNode probability, ReadOnlyVectorSpan<long> observations)` | Records the geometric log likelihood of independent observations as a single operation. / 幾何分布の対数尤度を 1 回の演算として記録します。 |
| `double LogSf(long k)` | Evaluates the natural logarithm of the survival function, `ln P(X > k)`. / 生存関数の自然対数 `ln P(X > k)` を評価します。 |
| `static ScalarNode LogSf(ScalarNode probability, long k)` | Records the geometric log survival probability and its derivatives. / 幾何分布の対数生存確率とその微分を記録します。 |
| `double Pmf(long k)` | Evaluates the probability mass function (PMF) at the specified integer. / 指定した整数点で確率質量関数（PMF）を評価します。 |
| `long Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<long> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(long k)` | Evaluates the survival function, `P(X > k)`. / 生存関数（上側累積確率）`P(X > k)` を評価します。 |

### `readonly struct GumbelDistribution`

Represents the Gumbel (Type I extreme value for maxima) distribution with location `μ` and scale `β`. / 位置 `μ`、尺度 `β` を持つガンベル分布（最大値の I 型極値分布）を表します。

- 継承・実装: `IContinuousDistribution`, `IUnivariateDistribution<double>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `GumbelDistribution(double location, double scale)` | Initializes a new instance of `GumbelDistribution` with the specified location and scale. / 位置および尺度パラメーターを指定して `GumbelDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `double Location { get; }` | Gets the location parameter `μ`. / 位置パラメーター `μ` を取得します。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `double Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `double Scale { get; }` | Gets the scale parameter `β`. / 尺度パラメーター `β` を取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `double SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `double SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(double x)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ x)`. / 累積分布関数（CDF）`P(X ≤ x)` を評価します。 |
| `double InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `double InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `double InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `double InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(double x)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ x)`. / 累積分布関数の自然対数 `ln P(X ≤ x)` を評価します。 |
| `static ScalarNode LogCdf(ScalarNode location, ScalarNode scale, ScalarNode x)` | Records the Gumbel log cumulative probability and its derivatives. / ガンベル分布の対数累積確率とその微分を記録します。 |
| `double LogPdf(double x)` | Evaluates the natural logarithm of the probability density function at the specified point. / 指定した点で確率密度関数の自然対数を評価します。 |
| `static ScalarNode LogPdf(ScalarNode location, ScalarNode scale, ScalarNode x)` | Records the Gumbel log density and its derivatives. / ガンベル分布の対数密度とその微分を記録します。 |
| `static ScalarNode LogPdf(ScalarNode location, ScalarNode scale, ReadOnlyVectorSpan<double> observations)` | Records the Gumbel log likelihood of independent observations as a single operation. / ガンベル分布の対数尤度を 1 回の演算として記録します。 |
| `double LogSf(double x)` | Evaluates the natural logarithm of the survival function, `ln P(X > x)`. / 生存関数の自然対数 `ln P(X > x)` を評価します。 |
| `static ScalarNode LogSf(ScalarNode location, ScalarNode scale, ScalarNode x)` | Records the Gumbel log survival probability and its derivatives. / ガンベル分布の対数生存確率とその微分を記録します。 |
| `double Pdf(double x)` | Evaluates the probability density function (PDF) at the specified point. / 指定した点で確率密度関数（PDF）を評価します。 |
| `double Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<double> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(double x)` | Evaluates the survival function (complementary CDF), `P(X > x)`. / 生存関数（上側累積確率）`P(X > x)` を評価します。 |

### `interface IContinuousDistribution`

Represents a continuous univariate probability distribution. / 連続値をとる 1 変量の確率分布を表します。

- 継承・実装: `IUnivariateDistribution<double>`

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double LogPdf(double x)` | Evaluates the natural logarithm of the probability density function at the specified point. / 指定した点で確率密度関数の自然対数を評価します。 |
| `double Pdf(double x)` | Evaluates the probability density function (PDF) at the specified point. / 指定した点で確率密度関数（PDF）を評価します。 |

### `interface IDiscreteDistribution`

Represents a discrete univariate probability distribution over integers. / 整数値をとる 1 変量の離散確率分布を表します。

- 継承・実装: `IUnivariateDistribution<long>`

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double LogPmf(long k)` | Evaluates the natural logarithm of the probability mass function at the specified integer. / 指定した整数点で確率質量関数の自然対数を評価します。 |
| `double Pmf(long k)` | Evaluates the probability mass function (PMF) at the specified integer. / 指定した整数点で確率質量関数（PMF）を評価します。 |

### `interface IMatrixContinuousDistribution`

Represents a continuous probability distribution over matrices. / 行列値をとる連続確率分布を表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Columns { get; }` | Gets the number of columns of the random matrix. / 確率行列の列数を取得します。 |
| `int Rows { get; }` | Gets the number of rows of the random matrix. / 確率行列の行数を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `Matrix<double> GetMean()` | Computes the mean matrix of the distribution. / 分布の平均行列を計算します。 |
| `double LogPdf(ReadOnlyMatrixSpan<double> x)` | Evaluates the natural logarithm of the probability density function at the specified matrix. / 指定した行列で確率密度関数の自然対数を評価します。 |
| `double Pdf(ReadOnlyMatrixSpan<double> x)` | Evaluates the probability density function (PDF) at the specified matrix. / 指定した行列で確率密度関数（PDF）を評価します。 |
| `void Sample(IRandomSource random, MatrixSpan<double> destination)` | Draws a single random matrix from the distribution and writes it to the destination. / 分布に従って標本行列を 1 個生成し、出力先へ書き込みます。 |

### `interface IMultivariateContinuousDistribution`

Represents a continuous multivariate probability distribution over real vectors. / 実数ベクトルをとる多変量連続確率分布を表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Dimension { get; }` | Gets the dimension (length of the random vector). / 次元（確率ベクトルの長さ）を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `Matrix<double> GetCovariance()` | Computes the covariance matrix of the distribution. / 分布の共分散行列を計算します。 |
| `Vector<double> GetMean()` | Computes the mean vector of the distribution. / 分布の平均ベクトルを計算します。 |
| `double LogPdf(ReadOnlyVectorSpan<double> x)` | Evaluates the natural logarithm of the probability density function at the specified vector. / 指定したベクトルで確率密度関数の自然対数を評価します。 |
| `void LogPdf(ReadOnlyMatrixSpan<double> x, VectorSpan<double> destination)` | Evaluates the natural logarithm of the probability density function for each column of the input matrix. / 入力行列の各列を評価点として、確率密度関数の自然対数を計算します。 |
| `double Pdf(ReadOnlyVectorSpan<double> x)` | Evaluates the probability density function (PDF) at the specified vector. / 指定したベクトルで確率密度関数（PDF）を評価します。 |
| `void Sample(IRandomSource random, VectorSpan<double> destination)` | Draws a single random vector from the distribution and writes it to the destination. / 分布に従って標本ベクトルを 1 個生成し、出力先へ書き込みます。 |
| `void Sample(IRandomSource random, MatrixSpan<double> destination)` | Draws random vectors from the distribution and writes one sample per column of the destination matrix. / 分布に従って標本ベクトルを生成し、出力先行列の各列に 1 個ずつ書き込みます。 |

### `interface IMultivariateDiscreteDistribution`

Represents a discrete multivariate probability distribution over integer vectors. / 整数ベクトルをとる多変量離散確率分布を表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Dimension { get; }` | Gets the dimension (length of the random vector). / 次元（確率ベクトルの長さ）を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `Matrix<double> GetCovariance()` | Computes the covariance matrix of the distribution. / 分布の共分散行列を計算します。 |
| `Vector<double> GetMean()` | Computes the mean vector of the distribution. / 分布の平均ベクトルを計算します。 |
| `double LogPmf(ReadOnlyVectorSpan<long> x)` | Evaluates the natural logarithm of the probability mass function at the specified integer vector. / 指定した整数ベクトルで確率質量関数の自然対数を評価します。 |
| `void LogPmf(ReadOnlyMatrixSpan<long> x, VectorSpan<double> destination)` | Evaluates the natural logarithm of the probability mass function for each column of the input matrix. / 入力行列の各列を評価点として、確率質量関数の自然対数を計算します。 |
| `double Pmf(ReadOnlyVectorSpan<long> x)` | Evaluates the probability mass function (PMF) at the specified integer vector. / 指定した整数ベクトルで確率質量関数（PMF）を評価します。 |
| `void Sample(IRandomSource random, VectorSpan<long> destination)` | Draws a single random integer vector from the distribution and writes it to the destination. / 分布に従って整数標本ベクトルを 1 個生成し、出力先へ書き込みます。 |
| `void Sample(IRandomSource random, MatrixSpan<long> destination)` | Draws random integer vectors from the distribution and writes one sample per column of the destination matrix. / 分布に従って整数標本ベクトルを生成し、出力先行列の各列に 1 個ずつ書き込みます。 |

### `interface IUnivariateDistribution<TValue>`

Represents a univariate probability distribution. / 1 変量の確率分布を表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `TValue Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `TValue SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `TValue SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(TValue x)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ x)`. / 累積分布関数（CDF）`P(X ≤ x)` を評価します。 |
| `TValue InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `TValue InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `TValue InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `TValue InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(TValue x)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ x)`. / 累積分布関数の自然対数 `ln P(X ≤ x)` を評価します。 |
| `double LogSf(TValue x)` | Evaluates the natural logarithm of the survival function, `ln P(X > x)`. / 生存関数の自然対数 `ln P(X > x)` を評価します。 |
| `TValue Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<TValue> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(TValue x)` | Evaluates the survival function (complementary CDF), `P(X > x)`. / 生存関数（上側累積確率）`P(X > x)` を評価します。 |

### `readonly struct InverseGammaDistribution`

Represents the inverse gamma distribution with shape `α` and scale `β` on `(0, ∞)`, where `1/X` follows a gamma distribution with shape `α` and rate `β`. / 形状 `α`、尺度 `β` を持つ `(0, ∞)` 上の逆ガンマ分布（`1/X` が形状 `α`、率 `β` のガンマ分布に従う分布）を表します。

- 継承・実装: `IContinuousDistribution`, `IUnivariateDistribution<double>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `InverseGammaDistribution(double shape, double scale)` | Initializes a new instance of `InverseGammaDistribution` with the specified shape and scale. / 形状および尺度パラメーターを指定して `InverseGammaDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `double Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `double Scale { get; }` | Gets the scale parameter `β`. / 尺度パラメーター `β` を取得します。 |
| `double Shape { get; }` | Gets the shape parameter `α`. / 形状パラメーター `α` を取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `double SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `double SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(double x)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ x)`. / 累積分布関数（CDF）`P(X ≤ x)` を評価します。 |
| `double InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `double InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `double InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `double InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(double x)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ x)`. / 累積分布関数の自然対数 `ln P(X ≤ x)` を評価します。 |
| `static ScalarNode LogCdf(ScalarNode shape, ScalarNode scale, ScalarNode x)` | Records the inverse gamma log cumulative probability and its derivatives. / 逆ガンマ分布の対数累積確率とその微分を記録します。 |
| `double LogPdf(double x)` | Evaluates the natural logarithm of the probability density function at the specified point. / 指定した点で確率密度関数の自然対数を評価します。 |
| `static ScalarNode LogPdf(ScalarNode shape, ScalarNode scale, ScalarNode x)` | Records the inverse gamma log density and its derivatives. / 逆ガンマ分布の対数密度とその微分を記録します。 |
| `static ScalarNode LogPdf(ScalarNode shape, ScalarNode scale, ReadOnlyVectorSpan<double> observations)` | Records the inverse gamma log likelihood of independent observations as a single operation. / 逆ガンマ分布に従う独立な観測値の対数尤度を 1 回の演算として記録します。 |
| `double LogSf(double x)` | Evaluates the natural logarithm of the survival function, `ln P(X > x)`. / 生存関数の自然対数 `ln P(X > x)` を評価します。 |
| `static ScalarNode LogSf(ScalarNode shape, ScalarNode scale, ScalarNode x)` | Records the inverse gamma log survival probability and its derivatives. / 逆ガンマ分布の対数生存確率とその微分を記録します。 |
| `double Pdf(double x)` | Evaluates the probability density function (PDF) at the specified point. / 指定した点で確率密度関数（PDF）を評価します。 |
| `double Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<double> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(double x)` | Evaluates the survival function (complementary CDF), `P(X > x)`. / 生存関数（上側累積確率）`P(X > x)` を評価します。 |

### `class InverseWishartDistribution`

Represents the inverse Wishart distribution `W⁻¹_p(ν, Ψ)` over symmetric positive definite `p × p` matrices, where `X⁻¹ ~ W_p(ν, Ψ⁻¹)`. / 対称正定値 `p × p` 行列上の逆ウィシャート分布 `W⁻¹_p(ν, Ψ)`（`X⁻¹ ~ W_p(ν, Ψ⁻¹)`）を表します。

- 継承・実装: `IMatrixContinuousDistribution`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `InverseWishartDistribution(double degreesOfFreedom, ReadOnlyMatrixSpan<double> scale)` | Initializes a new instance of `InverseWishartDistribution` with the specified degrees of freedom and scale matrix. / 自由度と尺度行列を指定して `InverseWishartDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double DegreesOfFreedom { get; }` | Gets the degrees of freedom `ν`. / 自由度 `ν` を取得します。 |
| `int Dimension { get; }` | Gets the matrix dimension `p` (samples are `p × p` matrices). / 行列の次元 `p`（確率変数は `p × p` 行列）を取得します。 |
| `ReadOnlyMatrixSpan<double> Scale { get; }` | Gets a read-only view of the symmetric scale matrix `Ψ`. / 対称尺度行列 `Ψ` の読み取り専用ビューを取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `Matrix<double> GetMean()` | Computes the mean matrix of the distribution. / 分布の平均行列を計算します。 |
| `Matrix<double> GetVariance()` | Computes the elementwise variance matrix of the distribution. / 分布の各成分の分散行列を計算します。 |
| `double LogPdf(ReadOnlyMatrixSpan<double> x)` | Evaluates the natural logarithm of the probability density function at the specified matrix. / 指定した行列で確率密度関数の自然対数を評価します。 |
| `double Pdf(ReadOnlyMatrixSpan<double> x)` | Evaluates the probability density function (PDF) at the specified matrix. / 指定した行列で確率密度関数（PDF）を評価します。 |
| `void Sample(IRandomSource random, MatrixSpan<double> destination)` | Draws a single random matrix from the distribution and writes it to the destination. / 分布に従って標本行列を 1 個生成し、出力先へ書き込みます。 |

### `readonly struct LaplaceDistribution`

Represents the Laplace (double exponential) distribution with location `μ` and scale `b`. / 位置 `μ`、尺度 `b` を持つラプラス分布（二重指数分布）を表します。

- 継承・実装: `IContinuousDistribution`, `IUnivariateDistribution<double>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `LaplaceDistribution(double location, double scale)` | Initializes a new instance of `LaplaceDistribution` with the specified location and scale. / 位置および尺度パラメーターを指定して `LaplaceDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `double Location { get; }` | Gets the location parameter `μ`. / 位置パラメーター `μ` を取得します。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `double Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `double Scale { get; }` | Gets the scale parameter `b`. / 尺度パラメーター `b` を取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `double SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `double SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(double x)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ x)`. / 累積分布関数（CDF）`P(X ≤ x)` を評価します。 |
| `double InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `double InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `double InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `double InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(double x)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ x)`. / 累積分布関数の自然対数 `ln P(X ≤ x)` を評価します。 |
| `static ScalarNode LogCdf(ScalarNode location, ScalarNode scale, ScalarNode x)` | Records the Laplace log cumulative probability and its derivatives. / ラプラス分布の対数累積確率とその微分を記録します。 |
| `double LogPdf(double x)` | Evaluates the natural logarithm of the probability density function at the specified point. / 指定した点で確率密度関数の自然対数を評価します。 |
| `static ScalarNode LogPdf(ScalarNode location, ScalarNode scale, ScalarNode x)` | Records the Laplace log density and its derivatives. / ラプラス分布の対数密度とその微分を記録します。 |
| `static ScalarNode LogPdf(ScalarNode location, ScalarNode scale, ReadOnlyVectorSpan<double> observations)` | Records the Laplace log likelihood of independent observations as a single operation. / ラプラス分布の対数尤度を 1 回の演算として記録します。 |
| `double LogSf(double x)` | Evaluates the natural logarithm of the survival function, `ln P(X > x)`. / 生存関数の自然対数 `ln P(X > x)` を評価します。 |
| `static ScalarNode LogSf(ScalarNode location, ScalarNode scale, ScalarNode x)` | Records the Laplace log survival probability and its derivatives. / ラプラス分布の対数生存確率とその微分を記録します。 |
| `double Pdf(double x)` | Evaluates the probability density function (PDF) at the specified point. / 指定した点で確率密度関数（PDF）を評価します。 |
| `double Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<double> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(double x)` | Evaluates the survival function (complementary CDF), `P(X > x)`. / 生存関数（上側累積確率）`P(X > x)` を評価します。 |

### `class LkjCorrelationDistribution`

Represents the Lewandowski–Kurowicka–Joe (LKJ) distribution over `d × d` correlation matrices. / `d × d` 相関行列上の LKJ（Lewandowski–Kurowicka–Joe）分布を表します。

- 継承・実装: `IMatrixContinuousDistribution`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `LkjCorrelationDistribution(int dimension, double shape)` | Initializes a new instance of `LkjCorrelationDistribution` with the specified dimension and shape parameter. / 次元と形状パラメーターを指定して `LkjCorrelationDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Dimension { get; }` | Gets the dimension `d` of the correlation matrix. / 相関行列の次元 `d` を取得します。 |
| `double Shape { get; }` | Gets the shape parameter `η`. / 形状パラメーター `η` を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `Matrix<double> GetMean()` | Computes the mean matrix of the distribution. / 分布の平均行列を計算します。 |
| `double LogPdf(ReadOnlyMatrixSpan<double> x)` | Evaluates the natural logarithm of the probability density function at the specified matrix. / 指定した行列で確率密度関数の自然対数を評価します。 |
| `static ScalarNode LogPdf(int dimension, ScalarNode shape, ReadOnlySpan<ScalarNode> correlations)` | Records the LKJ log density of a correlation matrix and its derivatives. / 相関行列に対する LKJ 分布の対数密度とその微分を記録します。 |
| `static ScalarNode LogPdfCholesky(int dimension, ScalarNode shape, ReadOnlySpan<ScalarNode> lowerFactor)` | Records the LKJ log density in its correlation Cholesky coordinates. / 相関行列の Cholesky 因子を座標とする LKJ 分布の対数密度とその微分を記録します。 |
| `double Pdf(ReadOnlyMatrixSpan<double> x)` | Evaluates the probability density function (PDF) at the specified matrix. / 指定した行列で確率密度関数（PDF）を評価します。 |
| `void Sample(IRandomSource random, MatrixSpan<double> destination)` | Draws a single random matrix from the distribution and writes it to the destination. / 分布に従って標本行列を 1 個生成し、出力先へ書き込みます。 |

### `readonly struct LocationScaleDistribution<TDistribution>`

Represents a location-scale transformation `X = μ + σY` of a continuous distribution. / 連続分布に位置・尺度変換 `X = μ + σY` を適用した分布を表します。

- 継承・実装: `IContinuousDistribution`, `IUnivariateDistribution<double>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `LocationScaleDistribution(TDistribution distribution, double location, double scale)` | Initializes a new instance of `LocationScaleDistribution`. / 元の分布と位置・尺度パラメーターを指定して `LocationScaleDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `TDistribution Distribution { get; }` | Gets the underlying distribution. / 元の連続分布を取得します。 |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `double Location { get; }` | Gets the location parameter `μ`. / 位置パラメーター `μ` を取得します。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `double Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `double Scale { get; }` | Gets the scale parameter `σ`. / 尺度パラメーター `σ` を取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `double SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `double SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(double x)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ x)`. / 累積分布関数（CDF）`P(X ≤ x)` を評価します。 |
| `double InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `double InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `double InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `double InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(double x)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ x)`. / 累積分布関数の自然対数 `ln P(X ≤ x)` を評価します。 |
| `double LogPdf(double x)` | Evaluates the natural logarithm of the probability density function at the specified point. / 指定した点で確率密度関数の自然対数を評価します。 |
| `double LogSf(double x)` | Evaluates the natural logarithm of the survival function, `ln P(X > x)`. / 生存関数の自然対数 `ln P(X > x)` を評価します。 |
| `double Pdf(double x)` | Evaluates the probability density function (PDF) at the specified point. / 指定した点で確率密度関数（PDF）を評価します。 |
| `double Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<double> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(double x)` | Evaluates the survival function (complementary CDF), `P(X > x)`. / 生存関数（上側累積確率）`P(X > x)` を評価します。 |

### `readonly struct LogNormalDistribution`

Represents the log-normal distribution where `ln X ~ N(μ, σ²)`. / `ln X ~ N(μ, σ²)` を満たす対数正規分布を表します。

- 継承・実装: `IContinuousDistribution`, `IUnivariateDistribution<double>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `LogNormalDistribution(double mu, double sigma)` | Initializes a new instance of `LogNormalDistribution` with the mean and standard deviation of the underlying normal distribution. / 対数変換後の正規分布の平均と標準偏差を指定して `LogNormalDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `double Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `double Mu { get; }` | Gets the mean `μ` of `ln X`. / `ln X` の平均 `μ` を取得します。 |
| `double Sigma { get; }` | Gets the standard deviation `σ` of `ln X`. / `ln X` の標準偏差 `σ` を取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `double SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `double SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(double x)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ x)`. / 累積分布関数（CDF）`P(X ≤ x)` を評価します。 |
| `double InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `double InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `double InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `double InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(double x)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ x)`. / 累積分布関数の自然対数 `ln P(X ≤ x)` を評価します。 |
| `static ScalarNode LogCdf(ScalarNode mu, ScalarNode sigma, ScalarNode x)` | Records the lognormal log cumulative probability and its derivatives. / 対数正規分布の対数累積確率とその微分を記録します。 |
| `double LogPdf(double x)` | Evaluates the natural logarithm of the probability density function at the specified point. / 指定した点で確率密度関数の自然対数を評価します。 |
| `static ScalarNode LogPdf(ScalarNode mu, ScalarNode sigma, ScalarNode x)` | Records the lognormal log density and its derivatives. / 対数正規分布の対数密度とその微分を記録します。 |
| `static ScalarNode LogPdf(ScalarNode mu, ScalarNode sigma, ReadOnlyVectorSpan<double> observations)` | Records the lognormal log likelihood of independent observations as a single operation. / 対数正規分布の対数尤度を 1 回の演算として記録します。 |
| `double LogSf(double x)` | Evaluates the natural logarithm of the survival function, `ln P(X > x)`. / 生存関数の自然対数 `ln P(X > x)` を評価します。 |
| `static ScalarNode LogSf(ScalarNode mu, ScalarNode sigma, ScalarNode x)` | Records the lognormal log survival probability and its derivatives. / 対数正規分布の対数生存確率とその微分を記録します。 |
| `double Pdf(double x)` | Evaluates the probability density function (PDF) at the specified point. / 指定した点で確率密度関数（PDF）を評価します。 |
| `double Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<double> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(double x)` | Evaluates the survival function (complementary CDF), `P(X > x)`. / 生存関数（上側累積確率）`P(X > x)` を評価します。 |

### `readonly struct LogisticDistribution`

Represents the logistic distribution with location `μ` and scale `s`. / 位置 `μ`、尺度 `s` を持つロジスティック分布を表します。

- 継承・実装: `IContinuousDistribution`, `IUnivariateDistribution<double>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `LogisticDistribution(double location, double scale)` | Initializes a new instance of `LogisticDistribution` with the specified location and scale. / 位置および尺度パラメーターを指定して `LogisticDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `double Location { get; }` | Gets the location parameter `μ`. / 位置パラメーター `μ` を取得します。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `double Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `double Scale { get; }` | Gets the scale parameter `s`. / 尺度パラメーター `s` を取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `double SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `double SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(double x)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ x)`. / 累積分布関数（CDF）`P(X ≤ x)` を評価します。 |
| `double InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `double InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `double InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `double InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(double x)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ x)`. / 累積分布関数の自然対数 `ln P(X ≤ x)` を評価します。 |
| `static ScalarNode LogCdf(ScalarNode location, ScalarNode scale, ScalarNode x)` | Records the logistic log cumulative probability and its derivatives. / ロジスティック分布の対数累積確率とその微分を記録します。 |
| `double LogPdf(double x)` | Evaluates the natural logarithm of the probability density function at the specified point. / 指定した点で確率密度関数の自然対数を評価します。 |
| `static ScalarNode LogPdf(ScalarNode location, ScalarNode scale, ScalarNode x)` | Records the logistic log density and its derivatives. / ロジスティック分布の対数密度とその微分を記録します。 |
| `static ScalarNode LogPdf(ScalarNode location, ScalarNode scale, ReadOnlyVectorSpan<double> observations)` | Records the logistic log likelihood of independent observations as a single operation. / ロジスティック分布の対数尤度を 1 回の演算として記録します。 |
| `double LogSf(double x)` | Evaluates the natural logarithm of the survival function, `ln P(X > x)`. / 生存関数の自然対数 `ln P(X > x)` を評価します。 |
| `static ScalarNode LogSf(ScalarNode location, ScalarNode scale, ScalarNode x)` | Records the logistic log survival probability and its derivatives. / ロジスティック分布の対数生存確率とその微分を記録します。 |
| `double Pdf(double x)` | Evaluates the probability density function (PDF) at the specified point. / 指定した点で確率密度関数（PDF）を評価します。 |
| `double Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<double> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(double x)` | Evaluates the survival function (complementary CDF), `P(X > x)`. / 生存関数（上側累積確率）`P(X > x)` を評価します。 |

### `class MatrixNormalDistribution`

Represents the matrix normal distribution `MN(M, U, V)` with row covariance `U` and column covariance `V`. / 行共分散 `U` と列共分散 `V` を持つ行列正規分布 `MN(M, U, V)` を表します。

- 継承・実装: `IMatrixContinuousDistribution`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `MatrixNormalDistribution(ReadOnlyMatrixSpan<double> mean, ReadOnlyMatrixSpan<double> rowCovariance, ReadOnlyMatrixSpan<double> columnCovariance)` | Initializes a new instance of `MatrixNormalDistribution` with the specified mean, row covariance, and column covariance. / 平均行列、行共分散行列、および列共分散行列を指定して `MatrixNormalDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `ReadOnlyMatrixSpan<double> ColumnCovariance { get; }` | Gets a read-only view of the symmetric column covariance matrix `V`. / 対称な列共分散行列 `V` の読み取り専用ビューを取得します。 |
| `int Columns { get; }` | Gets the number of columns of the random matrix. / 確率行列の列数を取得します。 |
| `ReadOnlyMatrixSpan<double> Mean { get; }` | Gets a read-only view of the mean matrix `M`. / 平均行列 `M` の読み取り専用ビューを取得します。 |
| `ReadOnlyMatrixSpan<double> RowCovariance { get; }` | Gets a read-only view of the symmetric row covariance matrix `U`. / 対称な行共分散行列 `U` の読み取り専用ビューを取得します。 |
| `int Rows { get; }` | Gets the number of rows of the random matrix. / 確率行列の行数を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `Matrix<double> GetMean()` | Computes the mean matrix of the distribution. / 分布の平均行列を計算します。 |
| `double LogPdf(ReadOnlyMatrixSpan<double> x)` | Evaluates the natural logarithm of the probability density function at the specified matrix. / 指定した行列で確率密度関数の自然対数を評価します。 |
| `double Pdf(ReadOnlyMatrixSpan<double> x)` | Evaluates the probability density function (PDF) at the specified matrix. / 指定した行列で確率密度関数（PDF）を評価します。 |
| `void Sample(IRandomSource random, MatrixSpan<double> destination)` | Draws a single random matrix from the distribution and writes it to the destination. / 分布に従って標本行列を 1 個生成し、出力先へ書き込みます。 |

### `class MultinomialDistribution`

Represents the multinomial distribution `Mult(n, p)` for category counts across `n` independent trials. / `n` 回の独立試行における各結果の出現回数を表す多項分布 `Mult(n, p)` です。

- 継承・実装: `IMultivariateDiscreteDistribution`

**フィールド・定数**

| メンバー | 説明 |
| --- | --- |
| `const long MaxTrials` | The maximum supported number of trials (`2⁵³`, matching `MaxTrials`). / サポートされる試行回数の上限（`2⁵³`、`MaxTrials` と同一）。 |

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `MultinomialDistribution(long trials, ReadOnlySpan<double> probabilities)` | Initializes a new instance of `MultinomialDistribution` with the specified trial count and category probabilities. / 試行回数と各結果の確率を指定して `MultinomialDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Dimension { get; }` | Gets the dimension (length of the random vector). / 次元（確率ベクトルの長さ）を取得します。 |
| `ReadOnlySpan<double> Probabilities { get; }` | Gets a read-only span of the normalized outcome probabilities `p`. / 正規化された各結果の確率 `p` の読み取り専用スパンを取得します。 |
| `long Trials { get; }` | Gets the number of trials `n`. / 試行回数 `n` を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `Matrix<double> GetCovariance()` | Computes the covariance matrix of the distribution. / 分布の共分散行列を計算します。 |
| `Vector<double> GetMean()` | Computes the mean vector of the distribution. / 分布の平均ベクトルを計算します。 |
| `double LogPmf(ReadOnlyVectorSpan<long> x)` | Evaluates the natural logarithm of the probability mass function at the specified integer vector. / 指定した整数ベクトルで確率質量関数の自然対数を評価します。 |
| `void LogPmf(ReadOnlyMatrixSpan<long> x, VectorSpan<double> destination)` | Evaluates the natural logarithm of the probability mass function for each column of the input matrix. / 入力行列の各列を評価点として、確率質量関数の自然対数を計算します。 |
| `double Pmf(ReadOnlyVectorSpan<long> x)` | Evaluates the probability mass function (PMF) at the specified integer vector. / 指定した整数ベクトルで確率質量関数（PMF）を評価します。 |
| `void Sample(IRandomSource random, VectorSpan<long> destination)` | Draws a single random integer vector from the distribution and writes it to the destination. / 分布に従って整数標本ベクトルを 1 個生成し、出力先へ書き込みます。 |
| `void Sample(IRandomSource random, MatrixSpan<long> destination)` | Draws a single random integer vector from the distribution and writes it to the destination. / 分布に従って整数標本ベクトルを 1 個生成し、出力先へ書き込みます。 |

### `static class MultivariateDistributionExtensions`

Provides allocating convenience overloads for multivariate and matrix distributions. / 多変量分布および行列値分布に対して、結果を新規確保して返す拡張メソッドを提供します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static Vector<double> LogPdf(IMultivariateContinuousDistribution distribution, ReadOnlyMatrixSpan<double> x)` | Evaluates the natural logarithm of the probability density function for each column of the input matrix. / 入力行列の各列を評価点として、確率密度関数の自然対数を計算します。 |
| `static Vector<double> LogPmf(IMultivariateDiscreteDistribution distribution, ReadOnlyMatrixSpan<long> x)` | Evaluates the natural logarithm of the probability mass function for each column of the input matrix. / 入力行列の各列を評価点として、確率質量関数の自然対数を計算します。 |
| `static Vector<double> Sample(IMultivariateContinuousDistribution distribution, IRandomSource random)` | Draws a single random vector from the distribution. / 分布に従って標本ベクトルを 1 個生成します。 |
| `static Vector<long> Sample(IMultivariateDiscreteDistribution distribution, IRandomSource random)` | Draws a single random integer vector from the distribution. / 分布に従って整数標本ベクトルを 1 個生成します。 |
| `static Matrix<double> Sample(IMatrixContinuousDistribution distribution, IRandomSource random)` | Draws a single random matrix from the distribution. / 分布に従って標本行列を 1 個生成します。 |
| `static Matrix<double> Sample(IMultivariateContinuousDistribution distribution, IRandomSource random, int count)` | Draws multiple random vectors from the distribution. / 分布に従って複数の標本ベクトルを生成します。 |
| `static Matrix<long> Sample(IMultivariateDiscreteDistribution distribution, IRandomSource random, int count)` | Draws multiple random integer vectors from the distribution. / 分布に従って複数の整数標本ベクトルを生成します。 |

### `class MultivariateNormalDistribution`

Represents the multivariate normal (Gaussian) distribution `N(μ, Σ)`. / 多変量正規分布（ガウス分布）`N(μ, Σ)` を表します。

- 継承・実装: `IMultivariateContinuousDistribution`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `MultivariateNormalDistribution(ReadOnlyVectorSpan<double> mean, ReadOnlyMatrixSpan<double> covariance)` | Initializes a new instance of `MultivariateNormalDistribution` with the specified mean vector and covariance matrix. / 平均ベクトルと共分散行列を指定して `MultivariateNormalDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `ReadOnlyMatrixSpan<double> CholeskyFactor { get; }` | Gets a read-only view of the lower-triangular Cholesky factor `L` (with zeros in the strictly upper triangle). / 共分散行列の下三角コレスキー因子 `L`（狭義上三角は 0）の読み取り専用ビューを取得します。 |
| `ReadOnlyMatrixSpan<double> Covariance { get; }` | Gets a read-only view of the symmetric covariance matrix `Σ`. / 対称共分散行列 `Σ` の読み取り専用ビューを取得します。 |
| `int Dimension { get; }` | Gets the dimension (length of the random vector). / 次元（確率ベクトルの長さ）を取得します。 |
| `double LogDeterminant { get; }` | Gets the natural logarithm of the covariance determinant, `ln det Σ = 2 Σᵢ ln lᵢᵢ`. / 共分散行列の行列式の自然対数 `ln det Σ = 2 Σᵢ ln lᵢᵢ` を取得します。 |
| `ReadOnlyVectorSpan<double> Mean { get; }` | Gets a read-only view of the mean vector `μ`. / 平均ベクトル `μ` の読み取り専用ビューを取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `IntegrationResult Cdf(ReadOnlyVectorSpan<double> upper, IntegrationOptions options = null)` | Computes the multivariate cumulative distribution function `P(X ≤ upper)` via numerical integration. / 指定した点までの多変量累積確率 `P(X ≤ upper)` を数値積分で計算します。 |
| `static MultivariateNormalDistribution FromCholeskyFactor(ReadOnlyVectorSpan<double> mean, ReadOnlyMatrixSpan<double> lowerFactor)` | Creates a `MultivariateNormalDistribution` from a mean vector and lower-triangular Cholesky factor `L` such that `Σ = LLᵀ`. / 平均ベクトルと下三角コレスキー因子 `L`（`Σ = LLᵀ`）から `MultivariateNormalDistribution` を生成します。 |
| `Matrix<double> GetCovariance()` | Computes the covariance matrix of the distribution. / 分布の共分散行列を計算します。 |
| `Vector<double> GetMean()` | Computes the mean vector of the distribution. / 分布の平均ベクトルを計算します。 |
| `double LogPdf(ReadOnlyVectorSpan<double> x)` | Evaluates the natural logarithm of the probability density function at the specified vector. / 指定したベクトルで確率密度関数の自然対数を評価します。 |
| `void LogPdf(ReadOnlyMatrixSpan<double> x, VectorSpan<double> destination)` | Evaluates the natural logarithm of the probability density function for each column of the input matrix. / 入力行列の各列を評価点として、確率密度関数の自然対数を計算します。 |
| `static ScalarNode LogPdfCholesky(ReadOnlySpan<ScalarNode> mean, ReadOnlySpan<ScalarNode> lowerFactor, ReadOnlySpan<ScalarNode> x)` | Records a multivariate normal log density parameterized by the covariance Cholesky factor. / 共分散の Cholesky 因子を指定し、多変量正規分布の対数密度とその微分を記録します。 |
| `double Pdf(ReadOnlyVectorSpan<double> x)` | Evaluates the probability density function (PDF) at the specified vector. / 指定したベクトルで確率密度関数（PDF）を評価します。 |
| `IntegrationResult Probability(ReadOnlyVectorSpan<double> lower, ReadOnlyVectorSpan<double> upper, IntegrationOptions options = null)` | Computes the hyperrectangle probability `P(lower < X ≤ upper)` via numerical integration. / 指定した超直方体に入る確率 `P(lower < X ≤ upper)` を数値積分で計算します。 |
| `void Sample(IRandomSource random, VectorSpan<double> destination)` | Draws a single random vector from the distribution and writes it to the destination. / 分布に従って標本ベクトルを 1 個生成し、出力先へ書き込みます。 |
| `void Sample(IRandomSource random, MatrixSpan<double> destination)` | Draws a single random vector from the distribution and writes it to the destination. / 分布に従って標本ベクトルを 1 個生成し、出力先へ書き込みます。 |

### `class MultivariateStudentTDistribution`

Represents the multivariate Student's t-distribution `X = μ + Lz √(ν/V)` with degrees of freedom `ν`, location `μ`, and scale matrix `S = LLᵀ`. / 自由度 `ν`、位置 `μ`、尺度行列 `S = LLᵀ` を持つ多変量 t 分布（`X = μ + Lz √(ν/V)`）を表します。

- 継承・実装: `IMultivariateContinuousDistribution`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `MultivariateStudentTDistribution(double degreesOfFreedom, ReadOnlyVectorSpan<double> location, ReadOnlyMatrixSpan<double> scale)` | Initializes a new instance of `MultivariateStudentTDistribution` with the specified degrees of freedom, location vector, and scale matrix. / 自由度、位置ベクトル、および尺度行列を指定して `MultivariateStudentTDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `ReadOnlyMatrixSpan<double> CholeskyFactor { get; }` | Gets a read-only view of the lower-triangular Cholesky factor `L` of `Scale` (with zeros in the strictly upper triangle). / `Scale` の下三角コレスキー因子 `L`（狭義上三角は 0）の読み取り専用ビューを取得します。 |
| `double DegreesOfFreedom { get; }` | Gets the degrees of freedom `ν`. / 自由度 `ν` を取得します。 |
| `int Dimension { get; }` | Gets the dimension (length of the random vector). / 次元（確率ベクトルの長さ）を取得します。 |
| `ReadOnlyVectorSpan<double> Location { get; }` | Gets a read-only view of the location vector `μ`. / 位置ベクトル `μ` の読み取り専用ビューを取得します。 |
| `ReadOnlyMatrixSpan<double> Scale { get; }` | Gets a read-only view of the symmetric scale matrix `S` (distinct from the covariance matrix). / 対称な尺度行列 `S`（共分散行列とは異なります）の読み取り専用ビューを取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `IntegrationResult Cdf(ReadOnlyVectorSpan<double> upper, IntegrationOptions options = null)` | Computes the multivariate cumulative distribution function `P(X ≤ upper)` via numerical integration. / 指定した点までの多変量累積確率 `P(X ≤ upper)` を数値積分で計算します。 |
| `static MultivariateStudentTDistribution FromCholeskyFactor(double degreesOfFreedom, ReadOnlyVectorSpan<double> location, ReadOnlyMatrixSpan<double> lowerFactor)` | Creates a `MultivariateStudentTDistribution` from degrees of freedom, location vector, and lower-triangular Cholesky factor `L` such that `S = LLᵀ`. / 自由度、位置ベクトル、および下三角コレスキー因子 `L`（`S = LLᵀ`）から `MultivariateStudentTDistribution` を生成します。 |
| `Matrix<double> GetCovariance()` | Computes the covariance matrix of the distribution. / 分布の共分散行列を計算します。 |
| `Vector<double> GetMean()` | Computes the mean vector of the distribution. / 分布の平均ベクトルを計算します。 |
| `double LogPdf(ReadOnlyVectorSpan<double> x)` | Evaluates the natural logarithm of the probability density function at the specified vector. / 指定したベクトルで確率密度関数の自然対数を評価します。 |
| `void LogPdf(ReadOnlyMatrixSpan<double> x, VectorSpan<double> destination)` | Evaluates the natural logarithm of the probability density function for each column of the input matrix. / 入力行列の各列を評価点として、確率密度関数の自然対数を計算します。 |
| `double Pdf(ReadOnlyVectorSpan<double> x)` | Evaluates the probability density function (PDF) at the specified vector. / 指定したベクトルで確率密度関数（PDF）を評価します。 |
| `IntegrationResult Probability(ReadOnlyVectorSpan<double> lower, ReadOnlyVectorSpan<double> upper, IntegrationOptions options = null)` | Computes the hyperrectangle probability `P(lower < X ≤ upper)` via numerical integration. / 指定した超直方体に入る確率 `P(lower < X ≤ upper)` を数値積分で計算します。 |
| `void Sample(IRandomSource random, VectorSpan<double> destination)` | Draws a single random vector from the distribution and writes it to the destination. / 分布に従って標本ベクトルを 1 個生成し、出力先へ書き込みます。 |
| `void Sample(IRandomSource random, MatrixSpan<double> destination)` | Draws a single random vector from the distribution and writes it to the destination. / 分布に従って標本ベクトルを 1 個生成し、出力先へ書き込みます。 |

### `readonly struct NegativeBinomialDistribution`

Represents the negative binomial distribution for the number of failures before `r` successes. / `r` 回の成功に至るまでの失敗回数を表す負の二項分布です。

- 継承・実装: `IDiscreteDistribution`, `IUnivariateDistribution<long>`

**フィールド・定数**

| メンバー | 説明 |
| --- | --- |
| `const double MaxSuccesses` | The maximum supported value for `Successes` (`2⁵²`, keeping `r + k` representable in `Double`). / `Successes` の上限（`2⁵²`。`r + k` を `Double` で扱うための範囲）。 |

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `NegativeBinomialDistribution(double successes, double probability)` | Initializes a new instance of `NegativeBinomialDistribution` with the specified success count and success probability. / 成功回数と成功確率を指定して `NegativeBinomialDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `long Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `double Probability { get; }` | Gets the per-trial success probability `p`. / 各試行の成功確率 `p` を取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `double Successes { get; }` | Gets the success count (shape) parameter `r`. / 成功回数（形状）パラメーター `r` を取得します。 |
| `long SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `long SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(long k)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ k)`. / 累積分布関数（CDF）`P(X ≤ k)` を評価します。 |
| `static NegativeBinomialDistribution FromMean(double mean, double successes)` | Creates a `NegativeBinomialDistribution` from mean `μ` and dispersion shape `r` (variance `μ + μ²/r`). / 平均 `μ` と形状パラメーター `r`（分散 `μ + μ²/r`）から `NegativeBinomialDistribution` を生成します。 |
| `long InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `long InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `long InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `long InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(long k)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ k)`. / 累積分布関数の自然対数 `ln P(X ≤ k)` を評価します。 |
| `static ScalarNode LogCdf(ScalarNode successes, ScalarNode probability, long k)` | Records the negative binomial log cumulative probability and its derivatives. / 負の二項分布の対数累積確率とその微分を記録します。 |
| `double LogPmf(long k)` | Evaluates the natural logarithm of the probability mass function at the specified integer. / 指定した整数点で確率質量関数の自然対数を評価します。 |
| `static ScalarNode LogPmf(ScalarNode successes, ScalarNode probability, long k)` | Records the negative binomial log mass and its derivatives. / 負の二項分布の対数質量とその微分を記録します。 |
| `static ScalarNode LogPmf(ScalarNode successes, ScalarNode probability, ReadOnlyVectorSpan<long> observations)` | Records the negative binomial log likelihood of independent observations as a single operation. / 負の二項分布の対数尤度を 1 回の演算として記録します。 |
| `double LogSf(long k)` | Evaluates the natural logarithm of the survival function, `ln P(X > k)`. / 生存関数の自然対数 `ln P(X > k)` を評価します。 |
| `static ScalarNode LogSf(ScalarNode successes, ScalarNode probability, long k)` | Records the negative binomial log survival probability and its derivatives. / 負の二項分布の対数生存確率とその微分を記録します。 |
| `double Pmf(long k)` | Evaluates the probability mass function (PMF) at the specified integer. / 指定した整数点で確率質量関数（PMF）を評価します。 |
| `long Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<long> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(long k)` | Evaluates the survival function, `P(X > k)`. / 生存関数（上側累積確率）`P(X > k)` を評価します。 |

### `class NegativeMultinomialDistribution`

Represents the negative multinomial distribution for category failure counts before `r` successes. / `r` 回の成功に至るまでの種類別失敗回数を表す負の多項分布です。

- 継承・実装: `IMultivariateDiscreteDistribution`

**フィールド・定数**

| メンバー | 説明 |
| --- | --- |
| `const double MaxSuccesses` | The maximum supported value for `Successes` (`2⁵²`, matching `MaxSuccesses`). / `Successes` の上限（`2⁵²`、`MaxSuccesses` と同一）。 |

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `NegativeMultinomialDistribution(double successes, double successProbability, ReadOnlySpan<double> failureProbabilities)` | Initializes a new instance of `NegativeMultinomialDistribution` with the specified success count, success probability, and failure probabilities. / 成功回数、成功確率、および種類別の失敗確率を指定して `NegativeMultinomialDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Dimension { get; }` | Gets the dimension (length of the random vector). / 次元（確率ベクトルの長さ）を取得します。 |
| `ReadOnlySpan<double> FailureProbabilities { get; }` | Gets a read-only span of the normalized failure probabilities `p`. / 正規化された各種類の失敗確率 `p` の読み取り専用スパンを取得します。 |
| `double SuccessProbability { get; }` | Gets the normalized success probability `p₀`. / 正規化された成功確率 `p₀` を取得します。 |
| `double Successes { get; }` | Gets the success count (shape) parameter `r`. / 成功回数（形状）パラメーター `r` を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `Matrix<double> GetCovariance()` | Computes the covariance matrix of the distribution. / 分布の共分散行列を計算します。 |
| `Vector<double> GetMean()` | Computes the mean vector of the distribution. / 分布の平均ベクトルを計算します。 |
| `double LogPmf(ReadOnlyVectorSpan<long> x)` | Evaluates the natural logarithm of the probability mass function at the specified integer vector. / 指定した整数ベクトルで確率質量関数の自然対数を評価します。 |
| `void LogPmf(ReadOnlyMatrixSpan<long> x, VectorSpan<double> destination)` | Evaluates the natural logarithm of the probability mass function for each column of the input matrix. / 入力行列の各列を評価点として、確率質量関数の自然対数を計算します。 |
| `double Pmf(ReadOnlyVectorSpan<long> x)` | Evaluates the probability mass function (PMF) at the specified integer vector. / 指定した整数ベクトルで確率質量関数（PMF）を評価します。 |
| `void Sample(IRandomSource random, VectorSpan<long> destination)` | Draws a single random integer vector from the distribution and writes it to the destination. / 分布に従って整数標本ベクトルを 1 個生成し、出力先へ書き込みます。 |
| `void Sample(IRandomSource random, MatrixSpan<long> destination)` | Draws a single random integer vector from the distribution and writes it to the destination. / 分布に従って整数標本ベクトルを 1 個生成し、出力先へ書き込みます。 |

### `readonly struct NormalDistribution`

Represents the normal (Gaussian) distribution `N(μ, σ²)`. / 正規分布（ガウス分布）`N(μ, σ²)` を表します。

- 継承・実装: `IContinuousDistribution`, `IUnivariateDistribution<double>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `NormalDistribution(double mean, double standardDeviation)` | Initializes a new instance of `NormalDistribution` with the specified mean and standard deviation. / 平均と標準偏差を指定して `NormalDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `double Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `static NormalDistribution Standard { get; }` | Gets the standard normal distribution `N(0, 1)`. / 標準正規分布 `N(0, 1)` を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `double SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `double SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(double x)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ x)`. / 累積分布関数（CDF）`P(X ≤ x)` を評価します。 |
| `double InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `double InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `double InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `double InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(double x)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ x)`. / 累積分布関数の自然対数 `ln P(X ≤ x)` を評価します。 |
| `static ScalarNode LogCdf(ScalarNode mean, ScalarNode standardDeviation, ScalarNode x)` | Records the normal log cumulative probability and its derivatives with respect to the mean, standard deviation, and variate. / 正規分布の対数累積確率と、平均・標準偏差・評価点に関する微分を記録します。 |
| `double LogPdf(double x)` | Evaluates the natural logarithm of the probability density function at the specified point. / 指定した点で確率密度関数の自然対数を評価します。 |
| `static ScalarNode LogPdf(ScalarNode mean, ScalarNode standardDeviation, ScalarNode x)` | Records the normal log density and its derivatives with respect to the mean, standard deviation, and variate. / 正規分布の対数密度と、平均・標準偏差・評価点に関する微分を記録します。 |
| `static ScalarNode LogPdf(ScalarNode mean, ScalarNode standardDeviation, ReadOnlyVectorSpan<double> observations)` | Records the normal log likelihood of independent observations as a single operation. / 正規分布に従う独立な観測値の対数尤度を 1 回の演算として記録します。 |
| `double LogSf(double x)` | Evaluates the natural logarithm of the survival function, `ln P(X > x)`. / 生存関数の自然対数 `ln P(X > x)` を評価します。 |
| `static ScalarNode LogSf(ScalarNode mean, ScalarNode standardDeviation, ScalarNode x)` | Records the normal log survival probability and its derivatives with respect to the mean, standard deviation, and variate. / 正規分布の対数上側確率と、平均・標準偏差・評価点に関する微分を記録します。 |
| `double Pdf(double x)` | Evaluates the probability density function (PDF) at the specified point. / 指定した点で確率密度関数（PDF）を評価します。 |
| `double Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<double> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(double x)` | Evaluates the survival function (complementary CDF), `P(X > x)`. / 生存関数（上側累積確率）`P(X > x)` を評価します。 |

### `readonly struct PoissonDistribution`

Represents the Poisson distribution with rate (mean) `λ`. / 平均（発生率）`λ` を持つポアソン分布を表します。

- 継承・実装: `IDiscreteDistribution`, `IUnivariateDistribution<long>`

**フィールド・定数**

| メンバー | 説明 |
| --- | --- |
| `const double MaxRate` | The maximum supported rate (`10¹⁸`, keeping samples within `Int64` range). / サポートされる平均の上限（`10¹⁸`、標本が `Int64` に収まる範囲）。 |

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `PoissonDistribution(double rate)` | Initializes a new instance of `PoissonDistribution` with the specified rate. / 平均（率）パラメーターを指定して `PoissonDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `long Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `double Rate { get; }` | Gets the rate (mean) parameter `λ`. / 平均（率）パラメーター `λ` を取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `long SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `long SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(long k)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ k)`. / 累積分布関数（CDF）`P(X ≤ k)` を評価します。 |
| `long InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `long InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `long InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `long InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(long k)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ k)`. / 累積分布関数の自然対数 `ln P(X ≤ k)` を評価します。 |
| `static ScalarNode LogCdf(ScalarNode rate, long k)` | Records the Poisson log cumulative probability and its derivatives. / ポアソン分布の対数累積確率とその微分を記録します。 |
| `double LogPmf(long k)` | Evaluates the natural logarithm of the probability mass function at the specified integer. / 指定した整数点で確率質量関数の自然対数を評価します。 |
| `static ScalarNode LogPmf(ScalarNode rate, long k)` | Records the Poisson log mass and its derivatives. / ポアソン分布の対数質量とその微分を記録します。 |
| `static ScalarNode LogPmf(ScalarNode rate, ReadOnlyVectorSpan<long> observations)` | Records the Poisson log likelihood of independent observations as a single operation. / ポアソン分布の対数尤度を 1 回の演算として記録します。 |
| `double LogSf(long k)` | Evaluates the natural logarithm of the survival function, `ln P(X > k)`. / 生存関数の自然対数 `ln P(X > k)` を評価します。 |
| `static ScalarNode LogSf(ScalarNode rate, long k)` | Records the Poisson log survival probability and its derivatives. / ポアソン分布の対数生存確率とその微分を記録します。 |
| `double Pmf(long k)` | Evaluates the probability mass function (PMF) at the specified integer. / 指定した整数点で確率質量関数（PMF）を評価します。 |
| `long Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<long> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(long k)` | Evaluates the survival function, `P(X > k)`. / 生存関数（上側累積確率）`P(X > k)` を評価します。 |

### `readonly struct StudentTDistribution`

Represents the location-scale Student's t-distribution `X = μ + σT`, where `T` has `ν` degrees of freedom. / 位置・尺度付きスチューデントの t 分布（`X = μ + σT`、`T` は自由度 `ν` の標準 t 分布）を表します。

- 継承・実装: `IContinuousDistribution`, `IUnivariateDistribution<double>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `StudentTDistribution(double degreesOfFreedom, double location = 0, double scale = 1)` | Initializes a new instance of `StudentTDistribution` with the specified degrees of freedom, location, and scale. / 自由度・位置・尺度を指定して `StudentTDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double DegreesOfFreedom { get; }` | Gets the degrees of freedom `ν`. / 自由度 `ν` を取得します。 |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `double Location { get; }` | Gets the location parameter `μ`. / 位置パラメーター `μ` を取得します。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `double Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `double Scale { get; }` | Gets the scale parameter `σ`. / 尺度パラメーター `σ` を取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `double SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `double SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(double x)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ x)`. / 累積分布関数（CDF）`P(X ≤ x)` を評価します。 |
| `double InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `double InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `double InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `double InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(double x)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ x)`. / 累積分布関数の自然対数 `ln P(X ≤ x)` を評価します。 |
| `double LogPdf(double x)` | Evaluates the natural logarithm of the probability density function at the specified point. / 指定した点で確率密度関数の自然対数を評価します。 |
| `static ScalarNode LogPdf(ScalarNode degreesOfFreedom, ScalarNode location, ScalarNode scale, ScalarNode x)` | Records the Student's t log density and its derivatives with respect to the degrees of freedom, location, scale, and variate. / スチューデントの t 分布の対数密度と、自由度・位置・尺度・評価点に関する微分を記録します。 |
| `static ScalarNode LogPdf(ScalarNode degreesOfFreedom, ScalarNode location, ScalarNode scale, ReadOnlyVectorSpan<double> observations)` | Records the Student's t log likelihood of independent observations as a single operation. / スチューデントの t 分布に従う独立な観測値の対数尤度を 1 回の演算として記録します。 |
| `double LogSf(double x)` | Evaluates the natural logarithm of the survival function, `ln P(X > x)`. / 生存関数の自然対数 `ln P(X > x)` を評価します。 |
| `double Pdf(double x)` | Evaluates the probability density function (PDF) at the specified point. / 指定した点で確率密度関数（PDF）を評価します。 |
| `double Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<double> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(double x)` | Evaluates the survival function (complementary CDF), `P(X > x)`. / 生存関数（上側累積確率）`P(X > x)` を評価します。 |

### `readonly struct TruncatedDistribution<TDistribution>`

Represents a continuous distribution conditioned (truncated) to a specified interval. / 連続分布を指定した区間に制限した切断分布（条件付き分布）を表します。

- 継承・実装: `IContinuousDistribution`, `IUnivariateDistribution<double>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `TruncatedDistribution(TDistribution distribution, double lower, double upper)` | Initializes a new instance of `TruncatedDistribution`. / 元の分布と切断区間を指定して `TruncatedDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `TDistribution Distribution { get; }` | Gets the underlying distribution. / 元の連続分布を取得します。 |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `double LogProbability { get; }` | Gets the natural logarithm of the probability that the underlying distribution falls within the truncation interval. / 元の分布において切断区間内に入る確率の自然対数を取得します。 |
| `double LowerBound { get; }` | Gets the specified lower bound `a` (for the effective support lower bound, see `SupportMinimum`). / 指定された下限 `a` を取得します（実際の台の下限は `SupportMinimum` を参照してください）。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `double Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `double Probability { get; }` | Gets the probability that the underlying distribution falls within the truncation interval. / 元の分布において切断区間内に入る確率を取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `double SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `double SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `double UpperBound { get; }` | Gets the specified upper bound `b` (for the effective support upper bound, see `SupportMaximum`). / 指定された上限 `b` を取得します（実際の台の上限は `SupportMaximum` を参照してください）。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(double x)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ x)`. / 累積分布関数（CDF）`P(X ≤ x)` を評価します。 |
| `double InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `double InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `double InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `double InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(double x)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ x)`. / 累積分布関数の自然対数 `ln P(X ≤ x)` を評価します。 |
| `double LogPdf(double x)` | Evaluates the natural logarithm of the probability density function at the specified point. / 指定した点で確率密度関数の自然対数を評価します。 |
| `double LogSf(double x)` | Evaluates the natural logarithm of the survival function, `ln P(X > x)`. / 生存関数の自然対数 `ln P(X > x)` を評価します。 |
| `double Pdf(double x)` | Evaluates the probability density function (PDF) at the specified point. / 指定した点で確率密度関数（PDF）を評価します。 |
| `double Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<double> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(double x)` | Evaluates the survival function (complementary CDF), `P(X > x)`. / 生存関数（上側累積確率）`P(X > x)` を評価します。 |

### `readonly struct UniformDistribution`

Represents the continuous uniform distribution `U(a, b)` on `[a, b]`. / `[a, b]` 上の連続一様分布 `U(a, b)` を表します。

- 継承・実装: `IContinuousDistribution`, `IUnivariateDistribution<double>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `UniformDistribution(double lower, double upper)` | Initializes a new instance of `UniformDistribution` over the specified interval. / 区間の下端と上端を指定して `UniformDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `double Lower { get; }` | Gets the lower bound `a`. / 下端 `a` を取得します。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `double Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `double SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `double SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `double Upper { get; }` | Gets the upper bound `b`. / 上端 `b` を取得します。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(double x)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ x)`. / 累積分布関数（CDF）`P(X ≤ x)` を評価します。 |
| `double InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `double InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `double InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `double InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(double x)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ x)`. / 累積分布関数の自然対数 `ln P(X ≤ x)` を評価します。 |
| `static ScalarNode LogCdf(ScalarNode lower, ScalarNode upper, ScalarNode x)` | Records the uniform log cumulative probability and its derivatives. / 一様分布の対数累積確率とその微分を記録します。 |
| `double LogPdf(double x)` | Evaluates the natural logarithm of the probability density function at the specified point. / 指定した点で確率密度関数の自然対数を評価します。 |
| `static ScalarNode LogPdf(ScalarNode lower, ScalarNode upper, ScalarNode x)` | Records the uniform log density and its derivatives. / 一様分布の対数密度とその微分を記録します。 |
| `static ScalarNode LogPdf(ScalarNode lower, ScalarNode upper, ReadOnlyVectorSpan<double> observations)` | Records the uniform log likelihood of independent observations as a single operation. / 一様分布の対数尤度を 1 回の演算として記録します。 |
| `double LogSf(double x)` | Evaluates the natural logarithm of the survival function, `ln P(X > x)`. / 生存関数の自然対数 `ln P(X > x)` を評価します。 |
| `static ScalarNode LogSf(ScalarNode lower, ScalarNode upper, ScalarNode x)` | Records the uniform log survival probability and its derivatives. / 一様分布の対数生存確率とその微分を記録します。 |
| `double Pdf(double x)` | Evaluates the probability density function (PDF) at the specified point. / 指定した点で確率密度関数（PDF）を評価します。 |
| `double Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<double> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(double x)` | Evaluates the survival function (complementary CDF), `P(X > x)`. / 生存関数（上側累積確率）`P(X > x)` を評価します。 |

### `readonly struct WeibullDistribution`

Represents the Weibull distribution with shape `k` and scale `λ` on `[0, ∞)`. / 形状 `k`、尺度 `λ` を持つ `[0, ∞)` 上のワイブル分布を表します。

- 継承・実装: `IContinuousDistribution`, `IUnivariateDistribution<double>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `WeibullDistribution(double shape, double scale)` | Initializes a new instance of `WeibullDistribution` with the specified shape and scale. / 形状および尺度パラメーターを指定して `WeibullDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Entropy { get; }` | Gets the entropy of the distribution in nats. / 分布のエントロピー（単位: nat）を取得します。 |
| `double ExcessKurtosis { get; }` | Gets the excess kurtosis of the distribution. / 分布の超過尖度を取得します。 |
| `double Mean { get; }` | Gets the mean (expected value) of the distribution. / 分布の平均（期待値）を取得します。 |
| `double Median { get; }` | Gets the median of the distribution. / 分布の中央値を取得します。 |
| `double Scale { get; }` | Gets the scale parameter `λ`. / 尺度パラメーター `λ` を取得します。 |
| `double Shape { get; }` | Gets the shape parameter `k`. / 形状パラメーター `k` を取得します。 |
| `double Skewness { get; }` | Gets the skewness of the distribution. / 分布の歪度を取得します。 |
| `double StandardDeviation { get; }` | Gets the standard deviation of the distribution. / 分布の標準偏差を取得します。 |
| `double SupportMaximum { get; }` | Gets the upper bound of the distribution's support. / 分布の台の上限を取得します。 |
| `double SupportMinimum { get; }` | Gets the lower bound of the distribution's support. / 分布の台の下限を取得します。 |
| `double Variance { get; }` | Gets the variance of the distribution. / 分布の分散を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Cdf(double x)` | Evaluates the cumulative distribution function (CDF), `P(X ≤ x)`. / 累積分布関数（CDF）`P(X ≤ x)` を評価します。 |
| `double InverseCdf(double p)` | Computes the quantile corresponding to the specified cumulative probability. / 指定した累積確率に対応する分位点を計算します。 |
| `double InverseLogCdf(double logP)` | Computes the quantile corresponding to the specified log cumulative probability. / 指定した対数累積確率に対応する分位点を計算します。 |
| `double InverseLogSf(double logQ)` | Computes the quantile corresponding to the specified log upper-tail probability. / 指定した対数上側確率に対応する分位点を計算します。 |
| `double InverseSf(double q)` | Computes the quantile corresponding to the specified upper-tail probability. / 指定した上側確率に対応する分位点を計算します。 |
| `double LogCdf(double x)` | Evaluates the natural logarithm of the cumulative distribution function, `ln P(X ≤ x)`. / 累積分布関数の自然対数 `ln P(X ≤ x)` を評価します。 |
| `static ScalarNode LogCdf(ScalarNode shape, ScalarNode scale, ScalarNode x)` | Records the Weibull log cumulative probability and its derivatives. / ワイブル分布の対数累積確率とその微分を記録します。 |
| `double LogPdf(double x)` | Evaluates the natural logarithm of the probability density function at the specified point. / 指定した点で確率密度関数の自然対数を評価します。 |
| `static ScalarNode LogPdf(ScalarNode shape, ScalarNode scale, ScalarNode x)` | Records the Weibull log density and its derivatives. / ワイブル分布の対数密度とその微分を記録します。 |
| `static ScalarNode LogPdf(ScalarNode shape, ScalarNode scale, ReadOnlyVectorSpan<double> observations)` | Records the Weibull log likelihood of independent observations as a single operation. / ワイブル分布の対数尤度を 1 回の演算として記録します。 |
| `double LogSf(double x)` | Evaluates the natural logarithm of the survival function, `ln P(X > x)`. / 生存関数の自然対数 `ln P(X > x)` を評価します。 |
| `static ScalarNode LogSf(ScalarNode shape, ScalarNode scale, ScalarNode x)` | Records the Weibull log survival probability and its derivatives. / ワイブル分布の対数生存確率とその微分を記録します。 |
| `double Pdf(double x)` | Evaluates the probability density function (PDF) at the specified point. / 指定した点で確率密度関数（PDF）を評価します。 |
| `double Sample(IRandomSource random)` | Draws a single random sample from the distribution. / 分布に従って標本を 1 個生成します。 |
| `void Sample(IRandomSource random, Span<double> destination)` | Fills a span with random samples drawn from the distribution. / 分布に従って生成した標本をスパンに書き込みます。 |
| `double Sf(double x)` | Evaluates the survival function (complementary CDF), `P(X > x)`. / 生存関数（上側累積確率）`P(X > x)` を評価します。 |

### `class WishartDistribution`

Represents the Wishart distribution `W_p(ν, Σ)` over symmetric positive definite `p × p` matrices. / 対称正定値 `p × p` 行列上のウィシャート分布 `W_p(ν, Σ)` を表します。

- 継承・実装: `IMatrixContinuousDistribution`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `WishartDistribution(double degreesOfFreedom, ReadOnlyMatrixSpan<double> scale)` | Initializes a new instance of `WishartDistribution` with the specified degrees of freedom and scale matrix. / 自由度と尺度行列を指定して `WishartDistribution` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double DegreesOfFreedom { get; }` | Gets the degrees of freedom `ν`. / 自由度 `ν` を取得します。 |
| `int Dimension { get; }` | Gets the matrix dimension `p` (samples are `p × p` matrices). / 行列の次元 `p`（確率変数は `p × p` 行列）を取得します。 |
| `ReadOnlyMatrixSpan<double> Scale { get; }` | Gets a read-only view of the symmetric scale matrix `Σ`. / 対称尺度行列 `Σ` の読み取り専用ビューを取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `Matrix<double> GetMean()` | Computes the mean matrix of the distribution. / 分布の平均行列を計算します。 |
| `Matrix<double> GetVariance()` | Computes the elementwise variance matrix of the distribution. / 分布の各成分の分散行列を計算します。 |
| `double LogPdf(ReadOnlyMatrixSpan<double> x)` | Evaluates the natural logarithm of the probability density function at the specified matrix. / 指定した行列で確率密度関数の自然対数を評価します。 |
| `static ScalarNode LogPdf(int order, ScalarNode degreesOfFreedom, ReadOnlySpan<ScalarNode> scale, ReadOnlySpan<ScalarNode> x)` | Records the Wishart log density and its derivatives. / ウィシャート分布の対数密度とその微分を記録します。 |
| `double Pdf(ReadOnlyMatrixSpan<double> x)` | Evaluates the probability density function (PDF) at the specified matrix. / 指定した行列で確率密度関数（PDF）を評価します。 |
| `void Sample(IRandomSource random, MatrixSpan<double> destination)` | Draws a single random matrix from the distribution and writes it to the destination. / 分布に従って標本行列を 1 個生成し、出力先へ書き込みます。 |

