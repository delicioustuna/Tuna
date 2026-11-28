# Tuna.Integration API リファレンス

> `(+N overloads)` は `Vector` / `VectorSpan` / `ReadOnlyVectorSpan` 等のコンテナ・ビュー組み合わせや `double` / `float` 固定型オーバーロードの省略を表します。

## 公開型一覧

- `Tuna.Integration` (10 型): [`Cubature`](#static-class-cubature), [`IIntegrand`](#interface-iintegrand), [`IMultivariateIntegrand`](#interface-imultivariateintegrand), [`IntegrationOptions`](#class-integrationoptions), [`IntegrationResult`](#readonly-struct-integrationresult), [`IntegrationTermination`](#enum-integrationtermination), [`Quadrature`](#static-class-quadrature), [`QuasiMonteCarlo`](#static-class-quasimontecarlo), [`QuasiMonteCarloOptions`](#class-quasimontecarlooptions), [`QuasiMonteCarloResult`](#readonly-struct-quasimontecarloresult)

## Tuna.Integration

### `static class Cubature`

Provides adaptive numerical integration over multidimensional hyperrectangles. / 多次元の超直方体上の適応型数値積分を提供します。

**フィールド・定数**

| メンバー | 説明 |
| --- | --- |
| `const int MaxDimension` | The maximum supported number of dimensions. / サポートされる最大次元数。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static IntegrationResult Integrate(Func<ReadOnlyVectorSpan<double>, double> integrand, ReadOnlySpan<double> lower, ReadOnlySpan<double> upper, IntegrationOptions options = null)` | Computes the definite integral over the hyperrectangle `Π [lowerᵢ, upperᵢ]`. / 超直方体 `Π [lowerᵢ, upperᵢ]` 上の定積分を計算します。 |
| `static IntegrationResult Integrate<TIntegrand>(TIntegrand integrand, ReadOnlySpan<double> lower, ReadOnlySpan<double> upper, IntegrationOptions options = null)` | Computes the definite integral over the hyperrectangle `Π [lowerᵢ, upperᵢ]`. / 超直方体 `Π [lowerᵢ, upperᵢ]` 上の定積分を計算します。 |

### `interface IIntegrand`

Represents a one-dimensional integrand function. / 1変数の被積分関数を表します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Evaluate(double x)` | Evaluates the integrand at the specified point. / 指定した点で被積分関数を評価します。 |

### `interface IMultivariateIntegrand`

Represents a multivariate integrand function. / 多変数の被積分関数を表します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Evaluate(ReadOnlyVectorSpan<double> x)` | Evaluates the integrand at the specified point. / 指定した点で被積分関数を評価します。 |

### `class IntegrationOptions`

Specifies options for deterministic numerical integration (`Quadrature` and `Cubature`). / 決定的な数値積分（`Quadrature` および `Cubature`）の設定を表します。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `IntegrationOptions()` |  |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double AbsoluteTolerance { get; set; }` | Gets the absolute error tolerance. / 絶対許容誤差を取得します。 |
| `long MaxEvaluations { get; set; }` | Gets the maximum number of integrand evaluations. / 被積分関数の評価回数の上限を取得します。 |
| `double RelativeTolerance { get; set; }` | Gets the relative error tolerance. / 相対許容誤差を取得します。 |

### `readonly struct IntegrationResult`

Represents the result of deterministic numerical integration (`Quadrature` or `Cubature`). / 決定的な数値積分（`Quadrature` または `Cubature`）の結果を表します。

- 継承・実装: `IEquatable<IntegrationResult>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `IntegrationResult(double Value, double ErrorEstimate, long Evaluations, IntegrationTermination Termination)` | Represents the result of deterministic numerical integration (`Quadrature` or `Cubature`). / 決定的な数値積分（`Quadrature` または `Cubature`）の結果を表します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `bool Converged { get; }` | Gets a value indicating whether integration converged within the requested tolerance. / 積分が許容誤差内に収束したかどうかを取得します。 |
| `double ErrorEstimate { get; set; }` | The estimated absolute error summed over all subregions; not a strict upper bound. / 絶対誤差の見積もり（全小区間の見積もりの和。厳密な上界ではありません）。 |
| `long Evaluations { get; set; }` | The total number of integrand evaluations performed. / 被積分関数を評価した総回数。 |
| `IntegrationTermination Termination { get; set; }` | The reason integration terminated. / 積分の終了理由。 |
| `double Value { get; set; }` | The computed integral value. / 計算された積分値。 |

### `enum IntegrationTermination`

Specifies the reason why numerical integration terminated. / 数値積分の終了理由を表します。

- 継承・実装: `IComparable`, `IConvertible`, `IFormattable`, `ISpanFormattable`

| 値 | 説明 |
| --- | --- |
| `Tolerance` | The error estimate (or standard error for quasi-Monte Carlo integration) met the requested tolerance. / 誤差見積もり（準モンテカルロ積分では標準誤差）が許容誤差以下になりました。 |
| `MaxEvaluations` | The maximum number of integrand evaluations was reached. / 被積分関数の評価回数が上限に達しました。 |
| `RoundoffLimited` | Rounding error prevented the requested tolerance from being reached. / 丸め誤差により許容誤差に到達できませんでした。 |
| `IntervalTooSmall` | The interval or region with the largest error could no longer be subdivided. / 誤差の大きい区間または領域をこれ以上分割できませんでした。 |
| `NonFiniteValue` | The integrand returned a non-finite value or an intermediate sum became non-finite. / 被積分関数の戻り値または積分の途中結果が有限値でなくなりました。 |

### `static class Quadrature`

Provides one-dimensional adaptive numerical integration and trapezoidal rules for sampled data. / 1変数の適応型数値積分および標本値に対する台形則を提供します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static Vector<double> CumulativeTrapezoid(ReadOnlyVectorSpan<double> y, double spacing = 1)` | Computes the cumulative trapezoidal integral of uniformly spaced samples and returns a new vector. / 等間隔の標本値に対する台形則の累積積分を計算し、新しいベクトルとして返します。 |
| `static Vector<double> CumulativeTrapezoid(ReadOnlyVectorSpan<double> y, ReadOnlyVectorSpan<double> x)` | Computes the cumulative trapezoidal integral of samples at arbitrary coordinates and returns a new vector. / 指定した座標の標本値に対する台形則の累積積分を計算し、新しいベクトルとして返します。 |
| `static VectorSpan<double> CumulativeTrapezoid(ReadOnlyVectorSpan<double> y, double spacing, VectorSpan<double> destination)` | Computes the cumulative trapezoidal integral of uniformly spaced samples into `destination`. / 等間隔の標本値に対する台形則の累積積分を計算し、`destination` に格納します。 |
| `static VectorSpan<double> CumulativeTrapezoid(ReadOnlyVectorSpan<double> y, ReadOnlyVectorSpan<double> x, VectorSpan<double> destination)` | Computes the cumulative trapezoidal integral of samples at arbitrary coordinates into `destination`. / 指定した座標の標本値に対する台形則の累積積分を計算し、`destination` に格納します。 |
| `static IntegrationResult Integrate(Func<double, double> integrand, double a, double b, IntegrationOptions options = null)` | Computes the definite integral `∫ₐᵇ f(x) dx`. / 定積分 `∫ₐᵇ f(x) dx` を計算します。 |
| `static IntegrationResult Integrate<TIntegrand>(TIntegrand integrand, double a, double b, IntegrationOptions options = null)` | Computes the definite integral `∫ₐᵇ f(x) dx`. / 定積分 `∫ₐᵇ f(x) dx` を計算します。 |
| `static IntegrationResult Integrate(Func<double, double> integrand, double a, double b, ReadOnlySpan<double> breakpoints, IntegrationOptions options = null)` | Computes the definite integral `∫ₐᵇ f(x) dx` with interior breakpoints. / 区切り点を指定して定積分 `∫ₐᵇ f(x) dx` を計算します。 |
| `static IntegrationResult Integrate<TIntegrand>(TIntegrand integrand, double a, double b, ReadOnlySpan<double> breakpoints, IntegrationOptions options = null)` | Computes the definite integral `∫ₐᵇ f(x) dx` with interior breakpoints. / 区切り点を指定して定積分 `∫ₐᵇ f(x) dx` を計算します。 |
| `static double Trapezoid(ReadOnlyVectorSpan<double> y, double spacing = 1)` | Integrates uniformly spaced samples using the composite trapezoidal rule `h * (y₀/2 + y₁ + ... + y_(n-2) + y_(n-1)/2)`. / 等間隔の標本値に対して合成台形則 `h * (y₀/2 + y₁ + ... + y_(n-2) + y_(n-1)/2)` で積分を計算します。 |
| `static double Trapezoid(ReadOnlyVectorSpan<double> y, ReadOnlyVectorSpan<double> x)` | Integrates samples at arbitrary coordinates using the composite trapezoidal rule `Σ (x_(i+1) - x_i)(y_i + y_(i+1))/2`. / 指定した座標の標本値に対して合成台形則 `Σ (x_(i+1) - x_i)(y_i + y_(i+1))/2` で積分を計算します。 |

### `static class QuasiMonteCarlo`

Provides randomized quasi-Monte Carlo integration over the unit hypercube `(0, 1)^d`. / 単位超立方体 `(0, 1)^d` 上の乱択化準モンテカルロ積分を提供します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static QuasiMonteCarloResult Integrate(Func<ReadOnlyVectorSpan<double>, double> integrand, int dimension, IRandomSource random, QuasiMonteCarloOptions options = null)` | Estimates the integral `∫_(0,1)^d f(u) du` over the unit hypercube. / 単位超立方体上の積分 `∫_(0,1)^d f(u) du` を推定します。 |
| `static QuasiMonteCarloResult Integrate<TIntegrand>(TIntegrand integrand, int dimension, IRandomSource random, QuasiMonteCarloOptions options = null)` | Estimates the integral `∫_(0,1)^d f(u) du` over the unit hypercube. / 単位超立方体上の積分 `∫_(0,1)^d f(u) du` を推定します。 |

### `class QuasiMonteCarloOptions`

Specifies options for randomized quasi-Monte Carlo integration (`QuasiMonteCarlo`). / 乱択化準モンテカルロ積分（`QuasiMonteCarlo`）の設定を表します。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `QuasiMonteCarloOptions()` |  |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double AbsoluteTolerance { get; set; }` | Gets the absolute tolerance for the standard error. / 標準誤差の絶対許容誤差を取得します。 |
| `int InitialPoints { get; set; }` | Gets the initial number of points evaluated per replicate. / 各複製で最初に評価する点数を取得します。 |
| `long MaxEvaluations { get; set; }` | Gets the maximum total number of integrand evaluations. / 被積分関数の総評価回数の上限を取得します。 |
| `double RelativeTolerance { get; set; }` | Gets the relative tolerance for the standard error. / 標準誤差の相対許容誤差を取得します。 |
| `int Replicates { get; set; }` | Gets the number of independently randomized Sobol' sequences. / 独立に乱択化する Sobol' 列の複製数を取得します。 |

### `readonly struct QuasiMonteCarloResult`

Represents the result of randomized quasi-Monte Carlo integration. / 乱択化準モンテカルロ積分の結果を表します。

- 継承・実装: `IEquatable<QuasiMonteCarloResult>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `QuasiMonteCarloResult(double Value, double StandardError, int Replicates, long PointsPerReplicate, long Evaluations, IntegrationTermination Termination)` | Represents the result of randomized quasi-Monte Carlo integration. / 乱択化準モンテカルロ積分の結果を表します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `bool Converged { get; }` | Gets a value indicating whether integration converged within the requested tolerance. / 積分が許容誤差内に収束したかどうかを取得します。 |
| `long Evaluations { get; set; }` | The total number of integrand evaluations performed, typically `Replicates * PointsPerReplicate`. / 被積分関数を評価した総回数（通常は `Replicates * PointsPerReplicate`）。 |
| `long PointsPerReplicate { get; set; }` | The number of points used per replicate (or the target stage count when `Termination` is `NonFiniteValue`). / 各複製で使用した点数（`Termination` が `NonFiniteValue` の場合は評価していた段階の点数）。 |
| `int Replicates { get; set; }` | The number of independent replicates. / 独立な複製数。 |
| `double StandardError { get; set; }` | The standard error of the estimate (sample standard deviation across replicates divided by the square root of `Replicates`). / 推定値の標準誤差（複製間の標本標準偏差を `Replicates` の平方根で割った値）。 |
| `IntegrationTermination Termination { get; set; }` | The reason integration terminated (`Tolerance`, `MaxEvaluations`, or `NonFiniteValue`). / 終了理由（`Tolerance`、`MaxEvaluations`、または `NonFiniteValue`）。 |
| `double Value { get; set; }` | The estimated integral value (mean of replicate estimates). / 積分の推定値（各複製の推定値の平均）。 |

