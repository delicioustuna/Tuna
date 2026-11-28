# 確率分布の自動微分

`Tuna.Statistics.Distributions` の静的メソッドには、分布パラメータを先に、評価点や観測値を後に渡します。`Tape.Leaf` で微分対象となる引数のノードを作成します。定数の引数には数値をそのまま渡せます。単点評価や合計を求めるメソッドは `ScalarNode`、観測ごとの値を求めるメソッドは `VectorNode<double>` を返します。

## 対応メソッド

| 分布 | 単一の観測 | 微分の対象 | 観測ベクトルの対数尤度 |
|---|---|---|---|
| 正規 | `LogPdf(μ, σ, x)`、`LogCdf(μ, σ, x)`、`LogSf(μ, σ, x)` | 平均 `μ`、標準偏差 `σ`、評価点 `x` | `LogPdf(μ, σ, observations)`。観測値は `ReadOnlyVectorSpan<double>` |
| ベルヌーイ | `LogPmf(p, k)`、`LogCdf(p, k)`、`LogSf(p, k)` | 成功確率 `p`。評価点 `k` は `long` | `LogPmf(p, observations)`。観測値は `ReadOnlyVectorSpan<long>` |
| スチューデントの t | `LogPdf(ν, μ, σ, x)` | 自由度 `ν`、位置 `μ`、尺度 `σ`、評価点 `x` | `LogPdf(ν, μ, σ, observations)`。観測値は `ReadOnlyVectorSpan<double>` |
| コーシー | `LogPdf(x₀, γ, x)` | 位置 `x₀`、尺度 `γ`、評価点 `x` | `LogPdf(x₀, γ, observations)`。観測値は `ReadOnlyVectorSpan<double>` |
| ガンマ・逆ガンマ | `LogPdf(a, s, x)`、`LogCdf(a, s, x)`、`LogSf(a, s, x)` | 形状 `a`、尺度 `s`、評価点 `x` | `LogPdf(a, s, observations)` |
| カイ二乗 | `LogPdf(ν, x)`、`LogCdf(ν, x)`、`LogSf(ν, x)` | 自由度 `ν`、評価点 `x` | `LogPdf(ν, observations)` |
| ベータ | `LogPdf(a, b, x)`、`LogCdf(a, b, x)`、`LogSf(a, b, x)` | 対数密度は形状 `a`、`b` と評価点 `x`。対数累積・生存確率は評価点 `x` | `LogPdf(a, b, observations)` |
| 指数 | `LogPdf`、`LogCdf`、`LogSf` | 率、評価点 | `LogPdf` |
| ロジスティック・ラプラス・ガンベル | `LogPdf`、`LogCdf`、`LogSf` | 位置、尺度、評価点 | `LogPdf` |
| ワイブル | `LogPdf`、`LogCdf`、`LogSf` | 形状、尺度、評価点 | `LogPdf` |
| 一様 | `LogPdf`、`LogCdf`、`LogSf` | 下端、上端、評価点 | `LogPdf` |
| 対数正規 | `LogPdf`、`LogCdf`、`LogSf` | 対数座標の平均と標準偏差、評価点 | `LogPdf` |
| F | `LogPdf`、`LogCdf`、`LogSf` | 対数密度は分子と分母の自由度、評価点。対数累積・生存確率は評価点 | `LogPdf` |
| ポアソン | `LogPmf`、`LogCdf`、`LogSf` | 率。観測値は `long` | `LogPmf` |
| 幾何・二項 | `LogPmf`、`LogCdf`、`LogSf` | 成功確率。観測値と二項分布の試行回数は `long` | `LogPmf` |
| 負の二項 | `LogPmf`、`LogCdf`、`LogSf` | 対数質量は成功回数と成功確率。対数累積・生存確率は成功確率 | `LogPmf` |
| カテゴリカル | `LogPmf`、`LogCdf`、`LogSf` | 各カテゴリの重み | `LogPmf`。`ReadOnlySpan<ScalarNode>` または `VectorSetNode<double>` で重みを指定 |

計算値は、同じパラメータを持つ分布インスタンスで評価した値とビット単位で一致します。観測ベクトルと共通のスカラーパラメータを渡す `LogPdf` / `LogPmf` は、独立同一分布（i.i.d.）の対数尤度を入力順に合計し、パラメータごとの偏微分の合計をテープに保存します。

観測ごとに異なる分布パラメータには、観測数と同じ長さの `VectorNode<double>` を渡します。`ScalarSetNode<double>` への暗黙変換により、共通の `ScalarNode` や `double` と組み合わせられます。既知のベクトル値は `Tape.Constant` でノードにできます。`LogPdf` / `LogPmf` は対数尤度の合計、`LogPdfPointwise` / `LogPmfPointwise` は観測順の対数密度または対数確率質量を返します。いずれもバッチ全体を 1 回の演算として扱います。合計はモデルの学習に、観測ごとの値は重み付き尤度や WAIC、LOO-CV に使えます。

ノードは同じテープから作成します。パラメータがすべてスカラー定数のときは、観測ごとの値を返すメソッドに `tape` を指定します。二項分布の試行回数は共通の `long` または観測数と同じ長さの `ReadOnlyVectorSpan<long>` で指定できます。カテゴリカル分布の `VectorSetNode<double>` には、共通の重みを `VectorNode<double>` で、観測ごとの重みを `MatrixNode<double>` で渡せます。重み行列の行はカテゴリ、列は観測に対応します。

観測値が空でも、共通のスカラーパラメータやカテゴリの重みは定義域を検証します。一様分布では上下限がともに共通の場合、その大小関係と幅も検証します。入力が有効なら、合計と勾配は 0 になり、観測ごとの結果は空ベクトルになります。

パラメータの有効範囲はコンストラクタと同じです。境界や特殊な入力では次のように振る舞います。

- ベルヌーイ分布: `p = 0, 1` では片側極限を微分値として用います。0 と 1 以外の観測値では対数尤度が `−∞` となり、その観測値から勾配への寄与は 0 になります。
- 正規分布: `x = ±∞` では対応する裾の極限を用います。`x = NaN` では値と偏微分がいずれも `NaN` です。
- スチューデントの t 分布: 自由度が `+∞` の場合は正規分布の極限を用い、自由度に関する偏微分は 0 です。有限の自由度で `ν/2` が 0 に丸まる場合、対数密度は `NaN` です。

t 分布（有限の自由度）とコーシー分布では、評価点が有限でも標準化がオーバーフローすると対数密度が `−∞` になることがあります。その場合でも、位置などの偏微分は倍精度で表現できる範囲で有限値を返します。

| 分布 | `x = ±∞` における偏微分 |
|---|---|
| t 分布（有限の自由度） | 自由度は `−∞`、位置と評価点は 0、尺度は `ν/σ` |
| コーシー分布 | 位置と評価点は 0、尺度は `1/γ` |

両分布の `x = NaN` では値と偏微分がいずれも `NaN` です（t 分布の自由度が `+∞` の場合は上記の正規分布の極限に従います）。

ガンマ系分布とベータ分布では、評価点が台の端点に等しい場合、台の内側からの片側極限を微分値として用います。台の外側にある有限な評価点の勾配は 0 です。ベータ分布の `LogCdf` と `LogSf` に勾配を必要とする形状ノードを渡すと `NotSupportedException` が発生します。

F 分布の自由度および負の二項分布の成功回数に関する裾確率の微分には不完全ベータ関数の形状微分が必要なため、裾確率の計算ではこれらの引数を定数として渡してください。微分対象のノードを渡すと `NotSupportedException` が発生します。なお、離散一様分布はパラメータ・評価点ともに整数のため、自動微分の対象外です。

## 観測ベクトルの使用例

正規分布に従う独立な観測値について、対数尤度と平均・標準偏差の勾配を求めます。

```csharp
using System;
using Tuna.Differentiation;
using Tuna.LinearAlgebra;
using Tuna.Statistics.Distributions;

var tape = new Tape();
ScalarNode mean = tape.Leaf(0.2);
ScalarNode scale = tape.Leaf(1.5);
var observations = new Vector<double>([0.7, -0.3, 2.0]);
ScalarNode likelihood = NormalDistribution.LogPdf(mean, scale, observations);
Span<double> gradient = stackalloc double[2];
double value = tape.Gradient(likelihood, new VectorSpan<double>(gradient));
// gradient[0] は平均、gradient[1] は標準偏差に関する勾配
```

## 多変量分布

各成分をスカラーノードとして渡すと、対数密度と入力に関する勾配を 1 回の演算として記録できます。行列の下三角成分は行の順に格納します。たとえば、3 × 3 の対称行列では `[a₁₁, a₂₁, a₂₂, a₃₁, a₃₂, a₃₃]` の順です。

| 分布 | メソッド | 微分の対象 |
|---|---|---|
| 多変量正規 | `MultivariateNormalDistribution.LogPdfCholesky(mean, lowerFactor, x)` | 平均、共分散の Cholesky 因子の下三角成分、観測値 |
| ディリクレ | `DirichletDistribution.LogPdf(alpha, x)` | 集中度、観測値。観測値の座標の総和による正規化も微分 |
| LKJ | `LkjCorrelationDistribution.LogPdf(dimension, shape, correlations)` | 形状と相関行列の対角を除く下三角成分 |
| ウィシャート | `WishartDistribution.LogPdf(order, degreesOfFreedom, scale, x)` | 自由度と、尺度・観測値の対称行列の下三角成分 |

相関行列の LKJ 密度には、対角を除く下三角成分 `[r₂₁, r₃₁, r₃₂, …]` を渡します。`MatrixBijectors.CorrelationCholesky` と組み合わせる場合は、`LkjCorrelationDistribution.LogPdfCholesky` に対角を含む因子を渡してください。このメソッドは因子から相関行列への対数ヤコビアン Σᵢ₌₂ᵈ (d − i) log Lᵢᵢ を含みます。

## MCMC への接続

`LogDensities.FromAutomaticDifferentiation` は、パラメータのノードから対数密度を返すモデルを `IDifferentiableLogDensity` として公開します。スカラーノードのスパン、またはテープとベクトルノードを受け取るモデルを渡せます。ベクトルノードの `p[i]` はスカラーノードを返し、単一の評価点を受け取る分布 API にそのまま渡せます。評価ごとに別々のテープを借りるため、並列チェーン間でモデルを共有できます。その場合は、モデルが読み書きする外部データも同時呼び出しに対応させてください。

```csharp
using Tuna.Probabilistic.Inference;
using Tuna.Statistics.Distributions;

var density = LogDensities.FromAutomaticDifferentiation(p =>
    NormalDistribution.LogPdf(0, 5, p[0]) + NormalDistribution.LogPdf(p[0], 2, 1.8));
var result = new NoUTurnSampler(new() { Chains = 1, Warmup = 128, Draws = 64, Seed = 91 }).Sample(density, [0.0]);
```

`MatrixBijectors` は共分散行列、共分散の Cholesky 因子、相関行列の Cholesky 因子の制約変換を提供します。制約付きの座標は、対角を含む下三角成分を行の順に並べます。相関行列の場合、独立な座標は因子の対角を除く下三角成分です。

制約付き対数密度をモデルの一部として合成するには `LogDensities.RecordTransformed` を使います。このメソッドは変換後の対数密度に対数ヤコビアンを加え、`IBijector.PullBack` による勾配とともに 1 回の演算として記録します。

たとえば、下三角成分を受け取るウィシャート分布を、制約のない座標から評価できます。

```csharp
var covarianceDensity = LogDensities.FromAutomaticDifferentiation(
    x => WishartDistribution.LogPdf(2, 5, [1.0, 0.2, 1.3], x));
var covarianceTransform = MatrixBijectors.Covariance(2);
var model = LogDensities.FromAutomaticDifferentiation(
    u => LogDensities.RecordTransformed(covarianceDensity, covarianceTransform, u));
```

自動微分に対応する分布 API は、パラメータの値や重みの総和が定義域外になると、`ArgumentOutOfRangeException` の派生型 `DistributionParameterOutOfRangeException` をスローします。NUTS と HMC は、軌道上の提案でこの例外が発生すると、密度ゼロとして軌道を打ち切り、発散を記録します。初期点には有限の対数密度と勾配が必要です。初期点でこの例外が発生した場合は、`ArgumentException` の `InnerException` から原因となったパラメータを確認できます。添字や次元の誤り、勾配未対応の引数に対する例外は呼び出し元にそのまま送出されます。

対数密度の値だけを使う `RandomWalkMetropolis`、`SliceSampler`、`ParallelTempering` などでは、`DistributionParameterOutOfRangeException` は呼び出し元にそのまま送出されます。これらのサンプラーで定義域外を密度ゼロとして扱う場合は、モデルから `double.NegativeInfinity` を返してください。

ロバスト回帰では、観測ごとの位置を `locations = intercept + slope * features` でベクトルとして求め、`StudentTDistribution.LogPdf(4, locations, scale, response)` で対数尤度を集計できます。尺度にコーシー分布の事前分布を設定する場合は `CauchyDistribution.LogPdf(0, 5, scale)` を加えます。自由度も推定するモデルでは、第 1 引数に自由度のノードを渡します。ロバスト回帰を NUTS で実行する具体例は [AutomaticDifferentiationTests.cs](../tests/Tuna.Probabilistic.Tests/Probabilistic/AutomaticDifferentiationTests.cs) を参照してください。
