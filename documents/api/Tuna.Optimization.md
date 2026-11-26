# Tuna.Optimization API リファレンス

> このファイルは `tools/ApiDocGenerator` によって自動生成されています。手動で編集せず、`dotnet build Tuna.slnx -c Release` の後に再生成してください。
>
> `(+N overloads)` は `Vector` / `VectorSpan` / `ReadOnlyVectorSpan` 等のコンテナ・ビュー組み合わせや `double` / `float` 固定型オーバーロードの省略を表します。

## 名前空間と公開型

- [`Tuna.Optimization`](#tunaoptimization) (6 型): [`FiniteDifferenceGradient`](#class-finitedifferencegradient), [`GradientCheck`](#static-class-gradientcheck), [`IDifferentiableObjectiveFunction`](#interface-idifferentiableobjectivefunction), [`IObjectiveFunction`](#interface-iobjectivefunction), [`MinimizationResult`](#class-minimizationresult), [`MinimizationTermination`](#enum-minimizationtermination)
- [`Tuna.Optimization.Evolutionary`](#tunaoptimizationevolutionary) (12 型): [`GeneticAlgorithm`](#class-geneticalgorithm), [`GeneticAlgorithmOptions`](#class-geneticalgorithmoptions), [`GeneticAlgorithmResult`](#class-geneticalgorithmresult), [`GeneticAlgorithmResult<TPoint>`](#class-geneticalgorithmresulttpoint), [`GeneticAlgorithmTermination`](#enum-geneticalgorithmtermination), [`Genome`](#class-genome), [`GenomeBatch`](#readonly-ref-struct-genomebatch), [`GenomeLayout`](#class-genomelayout), [`GenomeView`](#readonly-ref-struct-genomeview), [`IGenomeObjective`](#interface-igenomeobjective), [`RealCodedGeneticAlgorithm`](#class-realcodedgeneticalgorithm), [`RealCodedGeneticAlgorithmOptions`](#class-realcodedgeneticalgorithmoptions)
- [`Tuna.Optimization.Global`](#tunaoptimizationglobal) (10 型): [`BranchAndBound`](#class-branchandbound), [`BranchAndBoundOptions`](#class-branchandboundoptions), [`BranchAndBoundResult`](#class-branchandboundresult), [`BranchAndBoundTermination`](#enum-branchandboundtermination), [`FactorableObjective<TFunction>`](#class-factorableobjectivetfunction), [`GradientDual`](#readonly-struct-gradientdual), [`IFactorableFunction`](#interface-ifactorablefunction), [`IFactorableNumber<TSelf>`](#interface-ifactorablenumbertself), [`Interval`](#readonly-struct-interval), [`McCormick`](#readonly-struct-mccormick)
- [`Tuna.Optimization.Gradient`](#tunaoptimizationgradient) (5 型): [`Lbfgs`](#class-lbfgs), [`LbfgsB`](#class-lbfgsb), [`LbfgsBOptions`](#class-lbfgsboptions), [`LbfgsOptions`](#class-lbfgsoptions), [`LineSearchMinimizer`](#abstract-class-linesearchminimizer)
- [`Tuna.Optimization.Proximal`](#tunaoptimizationproximal) (20 型): [`Admm`](#class-admm), [`AdmmOptions`](#class-admmoptions), [`AdmmResult`](#class-admmresult), [`BoxConstraint`](#class-boxconstraint), [`ElasticNetPenalty`](#class-elasticnetpenalty), [`Fista`](#class-fista), [`FistaOptions`](#class-fistaoptions), [`GroupLassoPenalty`](#class-grouplassopenalty), [`IAdmmSubproblem`](#interface-iadmmsubproblem), [`IProximalOperator`](#interface-iproximaloperator), [`L1BallConstraint`](#class-l1ballconstraint), [`L1Penalty`](#class-l1penalty), [`L2BallConstraint`](#class-l2ballconstraint), [`L2NormPenalty`](#class-l2normpenalty), [`LeastSquaresObjective`](#class-leastsquaresobjective), [`NonNegativeConstraint`](#class-nonnegativeconstraint), [`QuadraticAdmmSubproblem`](#class-quadraticadmmsubproblem), [`QuadraticObjective`](#class-quadraticobjective), [`SimplexConstraint`](#class-simplexconstraint), [`ZeroFunction`](#class-zerofunction)
- [`Tuna.Optimization.Stochastic`](#tunaoptimizationstochastic) (27 型): [`AdaGrad`](#class-adagrad), [`AdaGrad<T>`](#class-adagradt), [`AdaGradOptions`](#class-adagradoptions), [`AdaMax`](#class-adamax), [`AdaMax<T>`](#class-adamaxt), [`AdaMaxOptions`](#class-adamaxoptions), [`Adam`](#class-adam), [`Adam<T>`](#class-adamt), [`AdamOptions`](#class-adamoptions), [`CosineAnnealingSchedule`](#class-cosineannealingschedule), [`ExponentialDecaySchedule`](#class-exponentialdecayschedule), [`ILearningRateSchedule`](#interface-ilearningrateschedule), [`InverseTimeDecaySchedule`](#class-inversetimedecayschedule), [`LinearWarmupSchedule`](#class-linearwarmupschedule), [`NAdam`](#class-nadam), [`NAdam<T>`](#class-nadamt), [`NAdamOptions`](#class-nadamoptions), [`RmsProp`](#class-rmsprop), [`RmsProp<T>`](#class-rmspropt), [`RmsPropOptions`](#class-rmspropoptions), [`StepDecaySchedule`](#class-stepdecayschedule), [`StochasticGradientDescent`](#class-stochasticgradientdescent), [`StochasticGradientDescent<T>`](#class-stochasticgradientdescentt), [`StochasticGradientDescentOptions`](#class-stochasticgradientdescentoptions), [`StochasticGradientOptimizer`](#abstract-class-stochasticgradientoptimizer), [`StochasticGradientOptimizer<T>`](#abstract-class-stochasticgradientoptimizert), [`StochasticGradientOptions`](#abstract-class-stochasticgradientoptions)

## Tuna.Optimization

### `class FiniteDifferenceGradient`

Approximates the gradient of a value-only objective function using finite differences. / 値のみを返す目的関数に対し、有限差分で勾配を近似するラッパーです。

- 継承・実装: `IDifferentiableObjectiveFunction`, `IObjectiveFunction`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `FiniteDifferenceGradient(Func<ReadOnlyVectorSpan<double>, double> function, int dimension, ReadOnlySpan<double> lowerBounds = default, ReadOnlySpan<double> upperBounds = default)` | Initializes a new instance that approximates the gradient of a delegate objective function. / デリゲートで与えた目的関数の勾配を有限差分で近似するインスタンスを初期化します。 |
| `FiniteDifferenceGradient(IObjectiveFunction objective, int dimension, ReadOnlySpan<double> lowerBounds = default, ReadOnlySpan<double> upperBounds = default)` | Initializes a new instance that approximates the gradient of an `IObjectiveFunction`. / `IObjectiveFunction` の勾配を有限差分で近似するインスタンスを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Dimension { get; }` | Gets the number of variables. / 変数の数を取得します。 |
| `long FunctionEvaluations { get; }` | Gets the total number of evaluations performed on the wrapped objective, including value-only evaluations. / ラップした目的関数の評価回数（値のみの評価を含む）を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Evaluate(ReadOnlyVectorSpan<double> x)` | Evaluates the wrapped objective function at the specified point. / 指定した点でラップした目的関数を評価します。 |
| `double Evaluate(ReadOnlyVectorSpan<double> x, VectorSpan<double> gradient)` | Evaluates the wrapped objective function at the specified point and writes its finite-difference gradient approximation. / 指定した点で目的関数を評価し、有限差分で近似した勾配を書き込みます。 |

### `static class GradientCheck`

Provides utilities for verifying analytic gradients against finite-difference approximations. / 解析勾配を有限差分近似と照合して検証するためのユーティリティを提供します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static double MaximumError<TObjective>(TObjective objective, ReadOnlySpan<double> x)` | Compares the analytic gradient of an objective function against a central finite-difference approximation at the specified point. / 指定した点で目的関数の解析勾配と中心差分近似を比較します。 |
| `static double MaximumError(Func<ReadOnlyVectorSpan<double>, VectorSpan<double>, double> objective, ReadOnlySpan<double> x)` | Compares the analytic gradient of a delegate objective function against a central finite-difference approximation at the specified point. / 指定した点でデリゲート目的関数の解析勾配と中心差分近似を比較します。 |

### `interface IDifferentiableObjectiveFunction`

Represents a differentiable objective function to minimize over ℝⁿ. / ℝⁿ 上で最小化する、勾配を計算できる目的関数を表します。

- 継承・実装: `IObjectiveFunction`

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Evaluate(ReadOnlyVectorSpan<double> x, VectorSpan<double> gradient)` | Evaluates the objective function at the specified point and writes its gradient. / 指定した点で目的関数を評価し、勾配を書き込みます。 |

### `interface IObjectiveFunction`

Represents an objective function to minimize over ℝⁿ. / ℝⁿ 上で最小化する目的関数を表します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Evaluate(ReadOnlyVectorSpan<double> x)` | Evaluates the objective function at the specified point. / 指定した点で目的関数を評価します。 |

### `class MinimizationResult`

Represents the result of a minimization run. / 最小化の実行結果を表します。

- 継承・実装: `IEquatable<MinimizationResult>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `MinimizationResult(Vector<double> Point, double Value, double GradientNorm, int Iterations, long Evaluations, MinimizationTermination Termination)` | Represents the result of a minimization run. / 最小化の実行結果を表します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `bool Converged { get; }` | Gets a value indicating whether minimization terminated by satisfying a convergence tolerance. / 収束許容誤差の条件を満たして終了したかどうかを取得します。 |
| `long Evaluations { get; set; }` | The total number of objective evaluations. / 目的関数の評価回数。 |
| `double GradientNorm { get; set; }` | The infinity norm of the gradient (or projected gradient for bound-constrained problems) at `Point`. / `Point` における勾配（範囲制約付き問題では射影勾配）の最大ノルム。 |
| `int Iterations { get; set; }` | The number of completed iterations (accepted steps). / 反復回数（採用したステップ数）。 |
| `Vector<double> Point { get; set; }` | The returned solution point (copied). For line-search methods this is the last accepted iterate (best evaluated); for deterministic stochastic-gradient runs this is the best evaluated point. / 返される解の点（コピー）。直線探索手法では最後に採用した反復点（評価した中で最良）、確率的勾配法の決定的な最小化では評価した中で最小値の点。 |
| `MinimizationTermination Termination { get; set; }` | The reason minimization terminated. / 終了理由。 |
| `double Value { get; set; }` | The objective value at `Point`. / `Point` における目的関数値。 |

### `enum MinimizationTermination`

Specifies the reason a gradient-based or proximal minimization terminated. / 勾配法または近接勾配法の終了理由を表します。

- 継承・実装: `IComparable`, `IConvertible`, `IFormattable`, `ISpanFormattable`

| 値 | 説明 |
| --- | --- |
| `None` | The optimizer has not terminated yet. / まだ終了していません。 |
| `GradientTolerance` | The infinity norm of the gradient (or projected gradient for bound-constrained problems) fell within tolerance. / 勾配（範囲制約付き問題では射影勾配）の最大ノルムが許容誤差以下になりました。 |
| `FunctionTolerance` | The relative decrease in the objective value over one iteration fell within tolerance. / 1反復あたりの目的関数値の相対的な減少量が許容誤差以下になりました。 |
| `StepTolerance` | The step size between consecutive iterates fell within tolerance. / 1反復あたりの点の移動量が許容誤差以下になりました。 |
| `ResidualTolerance` | Both primal and dual residuals of ADMM fell within tolerance. / ADMM の主残差と双対残差がともに許容誤差以下になりました。 |
| `MaxIterations` | The maximum number of iterations was reached. / 反復回数が上限に達しました。 |
| `MaxEvaluations` | The maximum number of objective evaluations was reached. / 目的関数の評価回数が上限に達しました。 |
| `LineSearchFailed` | Line search failed to find a step length satisfying the acceptance conditions (including finite-precision stagnation or gradient inconsistency). / 直線探索が条件を満たすステップ幅を見つけられませんでした（有限精度での停滞や、勾配と関数値変化の不整合を含みます）。 |
| `NonFiniteValue` | An objective value, gradient, or intermediate evaluation was non-finite. / 目的関数値や勾配などの評価値が有限値ではありませんでした。 |

## Tuna.Optimization.Evolutionary

### `class GeneticAlgorithm`

Minimizes an objective function over mixed-type genomes (real, integer, binary, categorical, and permutation genes) using a genetic algorithm. / 実数・整数・二値・カテゴリ・順列から成る混合型ゲノム上で目的関数を最小化する遺伝的アルゴリズムです。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `GeneticAlgorithm(GenomeLayout layout, GeneticAlgorithmOptions options = null)` | Initializes a new genetic algorithm over the specified genome layout. / 指定したゲノム構成上で最小化を行う遺伝的アルゴリズムを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double BestFitness { get; }` | Gets the best objective value evaluated so far. / これまでに評価した候補の最良目的関数値を取得します。 |
| `GenomeView BestGenome { get; }` | Gets a view of the best genome evaluated so far (retained even when `EliteCount` is 0). / これまでに評価した最良ゲノムへのビューを取得します（エリート数 0 の設定でも保持されます）。 |
| `ulong EvaluationSeed { get; }` | Gets a deterministic random seed derived from `Seed` and the current generation for stochastic candidate evaluation. / 確率的な候補評価に使用するため、`Seed` と現在の世代番号から決定的に導出した乱数シードを取得します。 |
| `long Evaluations { get; }` | Gets the total number of candidate evaluations reported to `Tell`. / `Tell` に報告された候補評価回数の累計を取得します。 |
| `int Generations { get; }` | Gets the number of completed generations (including the initial population). / 評価を完了した世代数（初期個体群を含む）を取得します。 |
| `bool IsFinished { get; }` | Gets a value indicating whether the algorithm has terminated. / 最小化が終了したかどうかを取得します。 |
| `GenomeLayout Layout { get; }` | Gets the genome layout. / ゲノム構成を取得します。 |
| `GenomeBatch Population { get; }` | Gets the population from the most recently evaluated generation. / 最後に評価を完了した世代の個体群を取得します。 |
| `ReadOnlySpan<double> PopulationFitness { get; }` | Gets the objective values corresponding to each individual in `Population`. / `Population` の各個体に対応する目的関数値を取得します。 |
| `ulong Seed { get; }` | Gets the random seed used by this instance (pass the same seed in options to reproduce the candidate sequence). / このインスタンスで使用している乱数シードを取得します（同じシードを設定に渡すと候補列を再現できます）。 |
| `GeneticAlgorithmTermination Termination { get; }` | Gets the reason the algorithm terminated. / 終了理由を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `GenomeBatch Ask()` | Returns the batch of candidate genomes to evaluate for the current generation. / 現在の世代で評価すべき候補ゲノムのバッチを取得します。 |
| `GeneticAlgorithmResult<Genome> Run(Func<GenomeView, double> objective, CancellationToken cancellationToken = default)` | Runs the genetic algorithm to termination using the specified delegate objective. / 指定したデリゲート目的関数を用いて終了条件まで遺伝的アルゴリズムを実行します。 |
| `GeneticAlgorithmResult<Genome> Run<TObjective>(TObjective objective, CancellationToken cancellationToken = default)` | Runs the genetic algorithm to termination using the specified genome objective. / 指定したゲノム目的関数を用いて終了条件まで遺伝的アルゴリズムを実行します。 |
| `void Tell(ReadOnlySpan<double> values)` | Reports objective values for the candidates returned by `Ask` and advances to the next generation. / `Ask` が返した候補の目的関数値を順に報告し、世代を1つ進めます。 |

### `class GeneticAlgorithmOptions`

Configures options for `GeneticAlgorithm` and `RealCodedGeneticAlgorithm`. / `GeneticAlgorithm` および `RealCodedGeneticAlgorithm` の設定オプションを表します。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `GeneticAlgorithmOptions()` |  |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double CrossoverDistributionIndex { get; set; }` | Gets the SBX distribution index `η_c` for real and integer genes. / 実数・整数遺伝子の SBX 分布指数 `η_c` を取得します。 |
| `double CrossoverProbability { get; set; }` | Gets the probability of applying crossover to a selected pair of parents. / 選択した親ペアに交叉を適用する確率を取得します。 |
| `int EliteCount { get; set; }` | Gets the number of top individuals carried unchanged into the next generation. / 次世代へそのまま引き継ぐエリート個体数を取得します。 |
| `int MaxGenerations { get; set; }` | Gets the maximum number of generations to evaluate (including the initial population). / 評価する世代数の上限（初期個体群を含む）を取得します。 |
| `double MutationDistributionIndex { get; set; }` | Gets the polynomial mutation distribution index `η_m` for real and integer genes. / 実数・整数遺伝子の多項式突然変異の分布指数 `η_m` を取得します。 |
| `double? MutationProbability { get; set; }` | Gets the optional per-unit mutation probability (per real/integer/binary/categorical gene or per permutation). / 突然変異単位（実数・整数・二値・カテゴリの各遺伝子、または各順列）ごとの突然変異確率を取得します。 |
| `int PopulationSize { get; set; }` | Gets the population size. / 個体数を取得します。 |
| `bool ReevaluateElites { get; set; }` | Gets a value indicating whether elites carried to the next generation are re-evaluated. / 次世代へ引き継ぐエリート個体も再評価するかどうかを取得します。 |
| `ulong? Seed { get; set; }` | Gets the optional random seed. / 乱数シードを取得します。 |
| `int? StagnationGenerations { get; set; }` | Gets the optional number of consecutive generations without strict improvement in best fitness before terminating. / 最良値の厳密な改善がない連続世代数による停滞終了の上限を取得します。 |
| `double? TargetFitness { get; set; }` | Gets the optional target fitness threshold at or below which the algorithm terminates. / この値以下の目的関数値が得られたときに終了する目標値を取得します。 |
| `int TournamentSize { get; set; }` | Gets the tournament size for parent selection. / 親選択に用いるトーナメントサイズを取得します。 |

### `class GeneticAlgorithmResult`

Represents the result of a `RealCodedGeneticAlgorithm` run. / 実数値遺伝的アルゴリズムの実行結果を表します。

- 継承・実装: `IEquatable<GeneticAlgorithmResult>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `GeneticAlgorithmResult(Vector<double> BestPoint, double BestFitness, int Generations, long Evaluations, GeneticAlgorithmTermination Termination)` | Represents the result of a `RealCodedGeneticAlgorithm` run. / 実数値遺伝的アルゴリズムの実行結果を表します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double BestFitness { get; set; }` | The objective value at `BestPoint`, or `NaN` if all evaluations were `NaN`. / `BestPoint` の目的関数値。全評価が `NaN` の場合は `NaN`。 |
| `Vector<double> BestPoint { get; set; }` | A copy of the best point evaluated so far. / これまでに評価した点のうち最良の点（コピー）。 |
| `long Evaluations { get; set; }` | The total number of candidate objective evaluations. / 目的関数の総評価回数。 |
| `int Generations { get; set; }` | The number of evaluated generations (including the initial population). / 評価した世代数（初期個体群を含む）。 |
| `GeneticAlgorithmTermination Termination { get; set; }` | The reason the algorithm terminated. / 終了理由。 |

### `class GeneticAlgorithmResult<TPoint>`

Represents the result of a `GeneticAlgorithm` run. / 遺伝的アルゴリズムの実行結果を表します。

- 継承・実装: `IEquatable<GeneticAlgorithmResult<TPoint>>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `GeneticAlgorithmResult(TPoint BestPoint, double BestFitness, int Generations, long Evaluations, GeneticAlgorithmTermination Termination)` | Represents the result of a `GeneticAlgorithm` run. / 遺伝的アルゴリズムの実行結果を表します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double BestFitness { get; set; }` | The objective value at `BestPoint`, or `NaN` if all evaluations were `NaN`. / `BestPoint` の目的関数値。全評価が `NaN` の場合は `NaN`。 |
| `TPoint BestPoint { get; set; }` | A copy of the best candidate evaluated so far. / これまでに評価した候補のうち最良の解（コピー）。 |
| `long Evaluations { get; set; }` | The total number of candidate objective evaluations. / 目的関数の総評価回数。 |
| `int Generations { get; set; }` | The number of evaluated generations (including the initial population). / 評価した世代数（初期個体群を含む）。 |
| `GeneticAlgorithmTermination Termination { get; set; }` | The reason the algorithm terminated. / 終了理由。 |

### `enum GeneticAlgorithmTermination`

Specifies the reason a genetic algorithm terminated. / 遺伝的アルゴリズムの終了理由を表します。

- 継承・実装: `IComparable`, `IConvertible`, `IFormattable`, `ISpanFormattable`

| 値 | 説明 |
| --- | --- |
| `None` | The algorithm has not terminated yet. / まだ終了していません。 |
| `MaxGenerations` | The evaluated generation count reached `MaxGenerations`. / 評価した世代数が上限（`MaxGenerations`）に達しました。 |
| `TargetReached` | An objective value at or below `TargetFitness` was reached. / 目標値（`TargetFitness`）以下の目的関数値が得られました。 |
| `Stagnation` | Best fitness did not strictly improve for `StagnationGenerations` consecutive generations. / 指定された世代数（`StagnationGenerations`）続けて最良値が改善しませんでした。 |

### `class Genome`

Represents an immutable owned copy of a single individual's genome. / 1個体分のゲノムの不変コピーを表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `ReadOnlySpan<bool> Bits { get; }` | Gets the binary genes (length `BitCount`). / 二値遺伝子（長さ `BitCount`）を取得します。 |
| `ReadOnlySpan<int> Categories { get; }` | Gets the categorical genes (length `CategoryCount`, each in `[0, count − 1]`). / カテゴリ遺伝子（長さ `CategoryCount`、各値は 0 からカテゴリ数 − 1）を取得します。 |
| `ReadOnlySpan<long> Integers { get; }` | Gets the integer-valued genes (length `IntegerCount`). / 整数遺伝子（長さ `IntegerCount`）を取得します。 |
| `GenomeLayout Layout { get; }` | Gets the layout describing this genome. / このゲノムの構成を取得します。 |
| `ReadOnlySpan<int> Permutations { get; }` | Gets all permutations concatenated in order (length `PermutationElementCount`). / すべての順列を順に連結したスパン（長さ `PermutationElementCount`）を取得します。 |
| `ReadOnlySpan<double> Reals { get; }` | Gets the real-valued genes (length `RealCount`). / 実数遺伝子（長さ `RealCount`）を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `GenomeView AsView()` | Returns a `GenomeView` referencing this genome. / このゲノムを参照する `GenomeView` を返します。 |
| `ReadOnlySpan<int> Permutation(int index)` | Gets a view of the specified permutation. / 指定した順列のビューを取得します。 |

### `readonly ref struct GenomeBatch`

Provides a read-only batch view of multiple individuals' genomes stored column-major per section (genes × individuals). / 複数個体のゲノムを区画ごとの列優先形式（遺伝子数 × 個体数、1列 = 1個体）で参照する読み取り専用ビューを表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `GenomeView this[int index] { get; }` | Gets a view of the genome at the specified individual index. / 指定したインデックスの個体のゲノムビューを取得します。 |
| `int Count { get; }` | Gets the number of individuals in this batch. / このバッチに含まれる個体数を取得します。 |
| `GenomeLayout Layout { get; }` | Gets the layout shared by all genomes in this batch. / このバッチ内の全ゲノムに共通する構成を取得します。 |
| `ReadOnlyMatrixSpan<double> Reals { get; }` | Gets the real-valued genes of all individuals as a matrix view of size `RealCount` × `Count`. / 全個体の実数遺伝子を行列ビュー（`RealCount` × `Count`）として取得します。 |

### `class GenomeLayout`

Describes the structure and bounds of a mixed-type genome containing real, integer, binary, categorical, and permutation sections. / 実数・整数・二値・カテゴリ・順列の各区画からなる混合型ゲノムの構成と範囲を定義します。

**フィールド・定数**

| メンバー | 説明 |
| --- | --- |
| `const ulong MaxIntegerWidth` | The maximum allowed width (`upper − lower`) for an integer gene (`2⁵³`), ensuring exact `Double` representation of offsets from the lower bound. / 整数遺伝子の許容最大幅（`upper − lower`）である `2⁵³` を表します。この範囲内では下限からのオフセットを `Double` で正確に表現できます。 |

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `GenomeLayout(ReadOnlySpan<double> realLowerBounds = default, ReadOnlySpan<double> realUpperBounds = default, ReadOnlySpan<long> integerLowerBounds = default, ReadOnlySpan<long> integerUpperBounds = default, int bitCount = 0, ReadOnlySpan<int> categoryCounts = default, ReadOnlySpan<int> permutationLengths = default)` | Initializes a new `GenomeLayout` from the specified section definitions. / 指定した各区画の定義から `GenomeLayout` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int BitCount { get; }` | Gets the number of binary genes. / 二値遺伝子の数を取得します。 |
| `int CategoryCount { get; }` | Gets the number of categorical genes. / カテゴリ遺伝子の数を取得します。 |
| `ReadOnlySpan<int> CategoryCounts { get; }` | Gets the number of categories for each categorical gene. / 各カテゴリ遺伝子のカテゴリ数を取得します。 |
| `int IntegerCount { get; }` | Gets the number of integer-valued genes. / 整数遺伝子の数を取得します。 |
| `ReadOnlySpan<long> IntegerLowerBounds { get; }` | Gets the inclusive lower bounds for the integer-valued genes. / 整数遺伝子の下限を取得します。 |
| `ReadOnlySpan<long> IntegerUpperBounds { get; }` | Gets the inclusive upper bounds for the integer-valued genes. / 整数遺伝子の上限を取得します。 |
| `long MutationUnitCount { get; }` | Gets the total number of mutation units (the sum of real, integer, binary, and categorical gene counts plus the number of permutations). / 突然変異単位の総数（実数・整数・二値・カテゴリの各遺伝子数と順列数の合計）を取得します。既定の突然変異確率はこの逆数になります。 |
| `int PermutationCount { get; }` | Gets the number of permutation genes. / 順列の数を取得します。 |
| `int PermutationElementCount { get; }` | Gets the total number of elements across all permutations. / すべての順列の長さの合計を取得します。 |
| `ReadOnlySpan<int> PermutationLengths { get; }` | Gets the length of each permutation. / 各順列の長さを取得します。 |
| `int RealCount { get; }` | Gets the number of real-valued genes. / 実数遺伝子の数を取得します。 |
| `ReadOnlySpan<double> RealLowerBounds { get; }` | Gets the lower bounds for the real-valued genes. / 実数遺伝子の下限を取得します。 |
| `ReadOnlySpan<double> RealUpperBounds { get; }` | Gets the upper bounds for the real-valued genes. / 実数遺伝子の上限を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `int PermutationOffset(int index)` | Gets the starting offset of the specified permutation within the concatenated permutation buffer. / 連結された順列配列内における指定した順列の開始オフセットを取得します。 |

### `readonly ref struct GenomeView`

Provides a read-only view of a single individual's genome partitioned according to a `GenomeLayout`. / `GenomeLayout` に従って区画分けされた1個体分のゲノムへの読み取り専用ビューを表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `ReadOnlySpan<bool> Bits { get; }` | Gets the binary genes (length `BitCount`). / 二値遺伝子（長さ `BitCount`）を取得します。 |
| `ReadOnlySpan<int> Categories { get; }` | Gets the categorical genes (length `CategoryCount`, each in `[0, count − 1]`). / カテゴリ遺伝子（長さ `CategoryCount`、各値は 0 からカテゴリ数 − 1）を取得します。 |
| `ReadOnlySpan<long> Integers { get; }` | Gets the integer-valued genes (length `IntegerCount`). / 整数遺伝子（長さ `IntegerCount`）を取得します。 |
| `GenomeLayout Layout { get; }` | Gets the layout describing this genome. / このゲノムの構成を取得します。 |
| `ReadOnlySpan<int> Permutations { get; }` | Gets all permutations concatenated in order (length `PermutationElementCount`). / すべての順列を順に連結したスパン（長さ `PermutationElementCount`）を取得します。 |
| `ReadOnlySpan<double> Reals { get; }` | Gets the real-valued genes (length `RealCount`). / 実数遺伝子（長さ `RealCount`）を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `ReadOnlySpan<int> Permutation(int index)` | Gets a view of the specified permutation. / 指定した順列のビューを取得します。 |
| `Genome ToGenome()` | Copies this genome view into an independent immutable `Genome`. / このゲノムビューの内容を独立した不変の `Genome` にコピーします。 |

### `interface IGenomeObjective`

Represents an objective function over genomes to minimize with `GeneticAlgorithm`. / `GeneticAlgorithm` で最小化するゲノム上の目的関数を表します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Evaluate(GenomeView genome)` | Evaluates the objective function for the specified genome. / 指定したゲノムの目的関数値を評価します。 |

### `class RealCodedGeneticAlgorithm`

Minimizes an objective function over a bounded real-valued box using a real-coded genetic algorithm. / 有界な実数ボックス領域上で目的関数を最小化する実数値遺伝的アルゴリズムです。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `RealCodedGeneticAlgorithm(ReadOnlySpan<double> lowerBounds, ReadOnlySpan<double> upperBounds, RealCodedGeneticAlgorithmOptions options = null)` | Initializes a new real-coded genetic algorithm over the box `[lowerBounds[i], upperBounds[i]]`. / ボックス領域 `[lowerBounds[i], upperBounds[i]]` 上で最小化を行う実数値遺伝的アルゴリズムを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double BestFitness { get; }` | Gets the best objective value evaluated so far. / これまでに評価した候補の最良目的関数値を取得します。 |
| `ReadOnlyVectorSpan<double> BestPoint { get; }` | Gets a view of the best point evaluated so far (retained even when `EliteCount` is 0). / これまでに評価した最良点へのビューを取得します（エリート数 0 の設定でも保持されます）。 |
| `int Dimension { get; }` | Gets the number of variables. / 変数の数を取得します。 |
| `ulong EvaluationSeed { get; }` | Gets a deterministic random seed derived from `Seed` and the current generation for stochastic candidate evaluation. / 確率的な候補評価に使用するため、`Seed` と現在の世代番号から決定的に導出した乱数シードを取得します。 |
| `long Evaluations { get; }` | Gets the total number of candidate evaluations reported to `Tell`. / `Tell` に報告された候補評価回数の累計を取得します。 |
| `int Generations { get; }` | Gets the number of completed generations (including the initial population). / 評価を完了した世代数（初期個体群を含む）を取得します。 |
| `bool IsFinished { get; }` | Gets a value indicating whether the algorithm has terminated. / 最小化が終了したかどうかを取得します。 |
| `ReadOnlyMatrixSpan<double> Population { get; }` | Gets the population from the most recently evaluated generation as a `Dimension` × population-size matrix view (one individual per column). / 最後に評価を完了した世代の個体群を `Dimension` × 個体数の行列ビュー（1列 = 1個体）として取得します。 |
| `ReadOnlySpan<double> PopulationFitness { get; }` | Gets the objective values corresponding to each column of `Population`. / `Population` の各列に対応する目的関数値を取得します。 |
| `ulong Seed { get; }` | Gets the random seed used by this instance (pass the same seed in options to reproduce the candidate sequence). / このインスタンスで使用している乱数シードを取得します（同じシードを設定に渡すと候補列を再現できます）。 |
| `GeneticAlgorithmTermination Termination { get; }` | Gets the reason the algorithm terminated. / 終了理由を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `ReadOnlyMatrixSpan<double> Ask()` | Returns the candidate points to evaluate for the current generation as a `Dimension` × candidate-count matrix view. / 現在の世代で評価すべき候補点を `Dimension` × 候補数の行列ビューとして取得します。 |
| `GeneticAlgorithmResult Run(Func<ReadOnlyVectorSpan<double>, double> objective, CancellationToken cancellationToken = default)` | Runs the genetic algorithm to termination using the specified delegate objective. / 指定したデリゲート目的関数を用いて終了条件まで実数値遺伝的アルゴリズムを実行します。 |
| `GeneticAlgorithmResult Run<TObjective>(TObjective objective, CancellationToken cancellationToken = default)` | Runs the genetic algorithm to termination using the specified objective function. / 指定した目的関数を用いて終了条件まで実数値遺伝的アルゴリズムを実行します。 |
| `void Tell(ReadOnlySpan<double> values)` | Reports objective values for the candidate columns returned by `Ask` and advances to the next generation. / `Ask` が返した候補列の目的関数値を順に報告し、世代を1つ進めます。 |

### `class RealCodedGeneticAlgorithmOptions`

Configures options for `RealCodedGeneticAlgorithm`. / `RealCodedGeneticAlgorithm` の設定オプションを表します（設定項目は `GeneticAlgorithmOptions` と同一です）。

- 継承・実装: `GeneticAlgorithmOptions`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `RealCodedGeneticAlgorithmOptions()` |  |

## Tuna.Optimization.Global

### `class BranchAndBound`

Globally minimizes a factorable function over a finite box using spatial branch-and-bound with interval arithmetic and McCormick relaxations. / 区間演算と McCormick 緩和を用いた空間分枝限定法により、有限ボックス上で分解可能な関数を大域的に最小化します。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `BranchAndBound(ReadOnlySpan<double> lowerBounds, ReadOnlySpan<double> upperBounds, BranchAndBoundOptions options = null)` | Initializes a new spatial branch-and-bound solver over the box `[lowerBounds, upperBounds]`. / ボックス `[lowerBounds, upperBounds]` 上の空間分枝限定法ソルバーを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Dimension { get; }` | Gets the number of variables. / 変数の数を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `BranchAndBoundResult Minimize<TFunction>(TFunction function, CancellationToken cancellationToken = default)` | Globally minimizes the specified factorable function over the configured box. / 指定した分解可能な関数を設定ボックス内で大域的に最小化します。 |

### `class BranchAndBoundOptions`

Configures options for `BranchAndBound`. / `BranchAndBound` の設定オプションを表します。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `BranchAndBoundOptions()` |  |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double AbsoluteTolerance { get; set; }` | Gets the absolute optimality gap tolerance between upper and lower bounds. / 上界と下界の差に対する絶対許容誤差を取得します。 |
| `bool LocalSearch { get; set; }` | Gets a value indicating whether `LbfgsB` local search (using `GradientDual`) is run from the root midpoint and from boxes that improve the upper bound. / 初期ボックスの中点および上界を更新したボックスの中点から `LbfgsB` による局所最小化（勾配は `GradientDual`）を実行するかどうかを取得します。 |
| `LbfgsBOptions LocalSearchOptions { get; set; }` | Gets optional `LbfgsBOptions` for local search, or `null` for defaults. / 局所探索に使用する `LbfgsBOptions` を取得します（`null` の場合は既定値）。 |
| `long MaxNodes { get; set; }` | Gets the maximum number of boxes (branch-and-bound nodes) that may be split. / 分割するボックス（ノード）数の上限を取得します。 |
| `double MinimumRelativeWidth { get; set; }` | Gets the minimum relative box width (relative to the initial box) below which a box is no longer split. / 初期ボックス幅に対する相対幅がこの値以下になったボックスを分割対象外とするしきい値を取得します。 |
| `double RelativeTolerance { get; set; }` | Gets the relative optimality gap tolerance scaled by `\|upper\|`. / 上界の絶対値に対する相対許容誤差を取得します。 |
| `bool UseRelaxation { get; set; }` | Gets a value indicating whether McCormick affine underestimators are used alongside interval bounds. / 下界計算に区間演算と併せて McCormick 緩和のアフィン下界を使用するかどうかを取得します。 |

### `class BranchAndBoundResult`

Represents the result of global minimization by `BranchAndBound`. / `BranchAndBound` による大域的最小化の実行結果を表します。

- 継承・実装: `IEquatable<BranchAndBoundResult>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `BranchAndBoundResult(Vector<double> Point, double Value, double LowerBound, long Nodes, int LocalSearches, BranchAndBoundTermination Termination)` | Represents the result of global minimization by `BranchAndBound`. / `BranchAndBound` による大域的最小化の実行結果を表します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `bool Converged { get; }` | Gets a value indicating whether the global minimum was certified within tolerance. / 許容誤差内で大域的最適性が保証されて終了したかどうかを取得します。 |
| `double Gap { get; }` | Gets the optimality gap `Value − LowerBound`. / 上界と下界のギャップ `Value − LowerBound` を取得します。 |
| `int LocalSearches { get; set; }` | The number of local searches performed. / 実行した局所探索の回数。 |
| `double LowerBound { get; set; }` | A certified lower bound on the global minimum over the box (minimum over remaining, set-aside, and pruned boxes). / ボックス上の最小値に対する下界（残存・保留・枝刈りした全ボックスの下界の最小値）。 |
| `long Nodes { get; set; }` | The number of split boxes. / 分割したボックス数。 |
| `Vector<double> Point { get; set; }` | A copy of the best feasible point found (or the initial box midpoint if none was found). / 見つかった最良点のコピー（見つからなかった場合は初期ボックスの中点）。 |
| `BranchAndBoundTermination Termination { get; set; }` | The reason the search terminated. / 終了理由。 |
| `double Value { get; set; }` | The objective value (upper bound) at `Point`, or `NaN` if no finite point was found. / `Point` における目的関数値（上界）。有限点が見つからなかった場合は `NaN`。 |

### `enum BranchAndBoundTermination`

Specifies the reason `BranchAndBound` terminated. / `BranchAndBound` の終了理由を表します。

- 継承・実装: `IComparable`, `IConvertible`, `IFormattable`, `ISpanFormattable`

| 値 | 説明 |
| --- | --- |
| `GapTolerance` | The gap between upper and lower bounds fell within tolerance (global minimum certified within tolerance). / 上界と下界の差が許容誤差以下になりました（許容誤差内で大域的最小値が保証されました）。 |
| `MaxNodes` | The maximum number of split boxes (`MaxNodes`) was reached. / 分割したボックス数が上限（`MaxNodes`）に達しました。 |
| `MinimumWidth` | All remaining boxes became narrower than `MinimumRelativeWidth` before the gap closed. / 残存するすべてのボックスが分割下限幅未満となり、ギャップが許容誤差まで縮まりませんでした。 |
| `NoFeasiblePoint` | The function was undefined everywhere on the box or no finite objective value was found. / ボックス全体で関数が未定義（空区間）であったか、有限値を返す点が見つかりませんでした。 |

### `class FactorableObjective<TFunction>`

Wraps an `IFactorableFunction` as an `IDifferentiableObjectiveFunction` using `GradientDual` for exact automatic differentiation. / `IFactorableFunction` を `GradientDual` による自動微分付き `IDifferentiableObjectiveFunction` としてラップします（`Lbfgs` などに渡せます）。

- 継承・実装: `IDifferentiableObjectiveFunction`, `IObjectiveFunction`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `FactorableObjective(TFunction function, int dimension)` | Initializes a new differentiable objective wrapper for `function` in `dimension` variables. / `dimension` 変数の分解可能関数 `function` に対する微分可能目的関数ラッパーを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Dimension { get; }` | Gets the number of variables. / 変数の数を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Evaluate(ReadOnlyVectorSpan<double> x)` | Evaluates the wrapped function at the specified point without computing gradients. / 勾配を計算せずに、指定した点でラップした関数を評価します。 |
| `double Evaluate(ReadOnlyVectorSpan<double> x, VectorSpan<double> gradient)` | Evaluates the wrapped function at the specified point and writes its exact gradient to `gradient`. / 指定した点でラップした関数を評価し、正確な勾配を `gradient` に書き込みます。 |

### `readonly struct GradientDual`

Represents a forward-mode dual number carrying a scalar value and its gradient for factorable functions. / 分解可能な関数の値と勾配を前進型自動微分で計算するデュアル数型を表します。

- 継承・実装: `IAdditionOperators<GradientDual, GradientDual, GradientDual>`, `IDivisionOperators<GradientDual, GradientDual, GradientDual>`, `IFactorableNumber<GradientDual>`, `IMultiplyOperators<GradientDual, GradientDual, GradientDual>`, `ISubtractionOperators<GradientDual, GradientDual, GradientDual>`, `IUnaryNegationOperators<GradientDual, GradientDual>`

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `ReadOnlySpan<double> Gradient { get; }` | Gets the gradient vector (an empty span represents the zero vector). / 勾配ベクトルを取得します（長さ 0 のスパンは零ベクトルを表します）。 |
| `double Value { get; }` | Gets the primal scalar value. / 主値（スカラー値）を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static GradientDual Abs(GradientDual x)` | Computes `\|x\|` and propagates its gradient (derivative is 0 at 0). / `\|x\|` とその勾配を計算します（0 での勾配は 0）。 |
| `static GradientDual Exp(GradientDual x)` | Computes `eˣ` and propagates its gradient. / `eˣ` とその勾配を計算します。 |
| `static GradientDual Log(GradientDual x)` | Computes `ln x` and propagates its gradient. / `ln x` とその勾配を計算します。 |
| `static GradientDual Max(GradientDual x, GradientDual y)` | Computes `max(x, y)` and propagates the gradient of the larger operand (selecting `x` on ties). / `max(x, y)` を計算し、大きい側の勾配を伝播します（同値時は `x` を選択）。 |
| `static GradientDual Min(GradientDual x, GradientDual y)` | Computes `min(x, y)` and propagates the gradient of the smaller operand (selecting `x` on ties). / `min(x, y)` を計算し、小さい側の勾配を伝播します（同値時は `x` を選択）。 |
| `static GradientDual Pow(GradientDual x, int n)` | Computes the integer power `xⁿ` and propagates its gradient (constant 1 when `n == 0`). / 整数乗 `xⁿ` とその勾配を計算します（`n == 0` の場合は定数 1）。 |
| `static GradientDual Reciprocal(GradientDual x)` | Computes `1 / x` and propagates its gradient. / `1 / x` とその勾配を計算します。 |
| `static GradientDual Sqrt(GradientDual x)` | Computes `√x` and propagates its gradient (local derivative is `PositiveInfinity` at 0). / `√x` とその勾配を計算します（0 における局所微分は `PositiveInfinity`）。 |
| `static GradientDual Square(GradientDual x)` | Computes `x²` and propagates its gradient. / `x²` とその勾配を計算します。 |
| `string ToString()` |  |
| `static GradientDual Variable(double value, int index, int dimension)` | Creates an independent variable at the specified coordinate index. / 指定した座標インデックスの独立変数を作成します。 |

**演算子**

`GradientDual operator +(GradientDual x, GradientDual y)` / `GradientDual operator +(GradientDual x, double c)` / `GradientDual operator +(double c, GradientDual x)` / `GradientDual operator /(GradientDual x, GradientDual y)` / `GradientDual operator /(GradientDual x, double c)` / `GradientDual operator /(double c, GradientDual x)` / `implicit operator GradientDual(double value)` / `GradientDual operator *(GradientDual x, GradientDual y)` / `GradientDual operator *(GradientDual x, double c)` / `GradientDual operator *(double c, GradientDual x)` / `GradientDual operator -(GradientDual x, GradientDual y)` / `GradientDual operator -(GradientDual x, double c)` / `GradientDual operator -(double c, GradientDual x)` / `GradientDual operator -(GradientDual x)`

### `interface IFactorableFunction`

Represents a factorable objective function evaluable over any `IFactorableNumber` (used by `BranchAndBound` and `FactorableObjective`). / 任意の `IFactorableNumber` で評価できる分解可能な目的関数を表します（`BranchAndBound` や `FactorableObjective` に渡します）。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `T Evaluate<T>(ReadOnlySpan<T> x)` | Evaluates the function at the specified point or box using arithmetic of type `T`. / 型 `T` の演算を用いて、指定した点または区間上で関数を評価します。 |

### `interface IFactorableNumber<TSelf>`

Represents a number type supporting arithmetic and univariate elementary operations for factorable functions, implemented by `Interval`, `McCormick`, and `GradientDual`. / 分解可能な関数（四則演算と1変数初等関数の合成）を評価するための数値型インターフェースです。`Interval`（区間）、`McCormick`（McCormick 緩和）、`GradientDual`（値と勾配）が実装します。

- 継承・実装: `IAdditionOperators<TSelf, TSelf, TSelf>`, `IDivisionOperators<TSelf, TSelf, TSelf>`, `IMultiplyOperators<TSelf, TSelf, TSelf>`, `ISubtractionOperators<TSelf, TSelf, TSelf>`, `IUnaryNegationOperators<TSelf, TSelf>`

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static TSelf Abs(TSelf x)` | Computes the absolute value `\|x\|`. / 絶対値 `\|x\|` を計算します。 |
| `static TSelf Exp(TSelf x)` | Computes the exponential `eˣ`. / 指数関数 `eˣ` を計算します。 |
| `static TSelf Log(TSelf x)` | Computes the natural logarithm `ln x` (for `x > 0`). / 自然対数 `ln x`（`x > 0` の範囲）を計算します。 |
| `static TSelf Max(TSelf x, TSelf y)` | Computes the maximum `max(x, y)`. / 最大値 `max(x, y)` を計算します。 |
| `static TSelf Min(TSelf x, TSelf y)` | Computes the minimum `min(x, y)`. / 最小値 `min(x, y)` を計算します。 |
| `static TSelf Pow(TSelf x, int n)` | Computes the integer power `xⁿ` (with `1 / x^\|n\|` for negative `n`). / 整数乗 `xⁿ`（負の `n` では `1 / x^\|n\|`）を計算します。 |
| `static TSelf Reciprocal(TSelf x)` | Computes the reciprocal `1 / x`. / 逆数 `1 / x` を計算します。 |
| `static TSelf Sqrt(TSelf x)` | Computes the square root `√x` (for `x ≥ 0`). / 平方根 `√x`（`x ≥ 0` の範囲）を計算します。 |
| `static TSelf Square(TSelf x)` | Computes the square `x²`. / 2乗 `x²` を計算します。 |

**演算子**

`TSelf operator +(TSelf x, double c)` / `TSelf operator +(double c, TSelf x)` / `TSelf operator /(TSelf x, double c)` / `TSelf operator /(double c, TSelf x)` / `implicit operator TSelf(double value)` / `TSelf operator *(TSelf x, double c)` / `TSelf operator *(double c, TSelf x)` / `TSelf operator -(TSelf x, double c)` / `TSelf operator -(double c, TSelf x)`

### `readonly struct Interval`

Represents a closed real interval with outwardly rounded arithmetic. / 外向き丸めによる区間演算をサポートする実数閉区間を表します。

- 継承・実装: `IAdditionOperators<Interval, Interval, Interval>`, `IDivisionOperators<Interval, Interval, Interval>`, `IEquatable<Interval>`, `IFactorableNumber<Interval>`, `IMultiplyOperators<Interval, Interval, Interval>`, `ISubtractionOperators<Interval, Interval, Interval>`, `IUnaryNegationOperators<Interval, Interval>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `Interval(double value)` | Initializes a degenerate point interval `[value, value]`. / 1点区間 `[value, value]` を初期化します。 |
| `Interval(double lower, double upper)` | Initializes a new interval `[lower, upper]`. / 区間 `[lower, upper]` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `static Interval Empty { get; }` | Gets the empty interval (both endpoints `NaN`). / 空区間（両端が `NaN`）を取得します。 |
| `static Interval Entire { get; }` | Gets the interval representing all real numbers `(−∞, +∞)`. / 実数全体 `(−∞, +∞)` を表す区間を取得します。 |
| `bool IsEmpty { get; }` | Gets a value indicating whether this interval is empty. / 区間が空かどうかを取得します。 |
| `double Lower { get; }` | Gets the lower endpoint of the interval, or `NaN` if empty. / 区間の下端を取得します（空区間の場合は `NaN`）。 |
| `double Midpoint { get; }` | Gets the midpoint of the interval. / 区間の中点を取得します。 |
| `double Upper { get; }` | Gets the upper endpoint of the interval, or `NaN` if empty. / 区間の上端を取得します（空区間の場合は `NaN`）。 |
| `double Width { get; }` | Gets the outwardly rounded width `Upper − Lower`, or `NaN` if empty. / 上向きに丸めた区間の幅 `Upper − Lower` を取得します（空区間の場合は `NaN`）。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static Interval Abs(Interval x)` | Computes the interval absolute value `\|x\|`. / 区間の絶対値 `\|x\|` を計算します。 |
| `bool Contains(double value)` | Determines whether the interval contains the specified value. / 指定した値がこの区間に含まれるかどうかを判定します。 |
| `bool Equals(Interval other)` |  |
| `bool Equals(object obj)` |  |
| `static Interval Exp(Interval x)` | Computes the outwardly rounded exponential `eˣ`. / 外向き丸めによる指数関数 `eˣ` を計算します。 |
| `int GetHashCode()` |  |
| `Interval Hull(Interval other)` | Computes the smallest interval enclosing both this interval and `other`. / この区間と `other` の両方を含む最小の区間（区間包）を計算します。 |
| `Interval Intersect(Interval other)` | Computes the intersection of this interval and `other`. / この区間と `other` の共通部分を計算します。 |
| `static Interval Log(Interval x)` | Computes the outwardly rounded natural logarithm `ln x` over the positive part of `x`. / `x` の正の部分に対する外向き丸めの自然対数 `ln x` を計算します（下端が 0 以下なら下端は −∞）。 |
| `static Interval Max(Interval x, Interval y)` | Computes the interval maximum `max(x, y)`. / 区間の最大値 `max(x, y)` を計算します。 |
| `static Interval Min(Interval x, Interval y)` | Computes the interval minimum `min(x, y)`. / 区間の最小値 `min(x, y)` を計算します。 |
| `static Interval Pow(Interval x, int n)` | Computes the outwardly rounded integer power `xⁿ` (`[1, 1]` for `n == 0`, `1 / x^\|n\|` for negative `n`). / 外向き丸めによる区間の整数乗 `xⁿ`（`n == 0` なら `[1, 1]`、負の `n` なら `1 / x^\|n\|`）を計算します。 |
| `static Interval Reciprocal(Interval x)` | Computes the outwardly rounded reciprocal `1 / x`. / 外向き丸めによる逆数区間 `1 / x` を計算します。 |
| `static Interval Sqrt(Interval x)` | Computes the outwardly rounded square root `√x` over the non-negative part of `x`. / `x` の非負部分に対する外向き丸めの平方根 `√x` を計算します。 |
| `static Interval Square(Interval x)` | Computes the outwardly rounded interval square `x²` (with non-negative lower endpoint). / 外向き丸めによる区間2乗 `x²`（下端は 0 以上）を計算します。 |
| `string ToString()` | Returns a string representation of the interval (`[Lower, Upper]` or `[empty]`). / 区間の文字列表現（`[Lower, Upper]`、空区間では `[empty]`）を返します。 |

**演算子**

`Interval operator +(Interval x, Interval y)` / `Interval operator +(Interval x, double c)` / `Interval operator +(double c, Interval x)` / `Interval operator /(Interval x, Interval y)` / `Interval operator /(Interval x, double c)` / `Interval operator /(double c, Interval x)` / `bool operator ==(Interval left, Interval right)` / `implicit operator Interval(double value)` / `bool operator !=(Interval left, Interval right)` / `Interval operator *(Interval x, Interval y)` / `Interval operator *(Interval x, double c)` / `Interval operator *(double c, Interval x)` / `Interval operator -(Interval x, Interval y)` / `Interval operator -(Interval x, double c)` / `Interval operator -(double c, Interval x)` / `Interval operator -(Interval x)`

### `readonly struct McCormick`

Represents an interval enclosure together with McCormick convex underestimator and concave overestimator relaxations (and their subgradients) for factorable functions. / 分解可能な関数に対する区間包含と、McCormick 凸下界・凹上界緩和（およびそれらの劣勾配・優勾配）を表します。

- 継承・実装: `IAdditionOperators<McCormick, McCormick, McCormick>`, `IDivisionOperators<McCormick, McCormick, McCormick>`, `IFactorableNumber<McCormick>`, `IMultiplyOperators<McCormick, McCormick, McCormick>`, `ISubtractionOperators<McCormick, McCormick, McCormick>`, `IUnaryNegationOperators<McCormick, McCormick>`

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Concave { get; }` | Gets the value of the concave overestimator at the reference point (or `NaN` if empty). / 基準点における凹上界関数の値を取得します（空の場合は `NaN`）。 |
| `ReadOnlySpan<double> ConcaveSubgradient { get; }` | Gets a supergradient of the concave overestimator at the reference point (an empty span represents the zero vector). / 基準点における凹上界関数の優勾配を取得します（長さ 0 のスパンは零ベクトルを表します）。 |
| `double Convex { get; }` | Gets the value of the convex underestimator at the reference point (or `NaN` if empty). / 基準点における凸下界関数の値を取得します（空の場合は `NaN`）。 |
| `ReadOnlySpan<double> ConvexSubgradient { get; }` | Gets a subgradient of the convex underestimator at the reference point (an empty span represents the zero vector). / 基準点における凸下界関数の劣勾配を取得します（長さ 0 のスパンは零ベクトルを表します）。 |
| `Interval Range { get; }` | Gets the outwardly rounded interval enclosure over the box (`Empty` if undefined everywhere). / ボックス上の外向き丸め区間包含を取得します（ボックス全体で未定義の場合は `Empty`）。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static McCormick Abs(McCormick x)` | Computes the McCormick relaxation of `\|x\|`. / `\|x\|` の McCormick 緩和を計算します。 |
| `static McCormick Exp(McCormick x)` | Computes the McCormick relaxation of `eˣ`. / `eˣ` の McCormick 緩和を計算します。 |
| `static McCormick Log(McCormick x)` | Computes the McCormick relaxation of `ln x` over the positive part of `x` (convex bound is `NegativeInfinity` when lower endpoint ≤ 0). / `x` の正の部分に対する `ln x` の McCormick 緩和を計算します（下端が 0 以下の場合、凸下界は −∞）。 |
| `static McCormick Max(McCormick x, McCormick y)` | Computes the McCormick relaxation of `max(x, y) = (x + y + \|x − y\|) / 2`, tightened with the interval maximum. / `max(x, y) = (x + y + \|x − y\|) / 2` の McCormick 緩和を計算し、区間の最大値で切り詰めます。 |
| `static McCormick Min(McCormick x, McCormick y)` | Computes the McCormick relaxation of `min(x, y) = (x + y − \|x − y\|) / 2`, tightened with the interval minimum. / `min(x, y) = (x + y − \|x − y\|) / 2` の McCormick 緩和を計算し、区間の最小値で切り詰めます。 |
| `static McCormick Pow(McCormick x, int n)` | Computes the McCormick relaxation of the integer power `xⁿ` (`1 / x^\|n\|` for negative `n`). / 整数乗 `xⁿ`（負の `n` では `1 / x^\|n\|`）の McCormick 緩和を計算します。 |
| `static McCormick Reciprocal(McCormick x)` | Computes the McCormick relaxation of `1 / x` (falls back to constant interval bounds when 0 lies strictly inside `x`). / `1 / x` の McCormick 緩和を計算します（0 を内部に含む場合は区間端の定数緩和にフォールバックします）。 |
| `static McCormick Sqrt(McCormick x)` | Computes the McCormick relaxation of `√x` over the non-negative part of `x`. / `x` の非負部分に対する `√x` の McCormick 緩和を計算します。 |
| `static McCormick Square(McCormick x)` | Computes the McCormick relaxation of `x²`. / `x²` の McCormick 緩和を計算します。 |
| `string ToString()` |  |
| `static McCormick Variable(Interval range, double point, int index, int dimension)` | Creates an independent variable at the specified coordinate index. / 指定した座標インデックスの独立変数を作成します。 |

**演算子**

`McCormick operator +(McCormick x, McCormick y)` / `McCormick operator +(McCormick x, double c)` / `McCormick operator +(double c, McCormick x)` / `McCormick operator /(McCormick x, McCormick y)` / `McCormick operator /(McCormick x, double c)` / `McCormick operator /(double c, McCormick x)` / `implicit operator McCormick(double value)` / `McCormick operator *(McCormick x, McCormick y)` / `McCormick operator *(McCormick x, double c)` / `McCormick operator *(double c, McCormick x)` / `McCormick operator -(McCormick x, McCormick y)` / `McCormick operator -(McCormick x, double c)` / `McCormick operator -(double c, McCormick x)` / `McCormick operator -(McCormick x)`

## Tuna.Optimization.Gradient

### `class Lbfgs`

Minimizes an unconstrained differentiable objective function using the L-BFGS method. / L-BFGS 法を用いて無制約の微分可能な目的関数を最小化します。

- 継承・実装: `LineSearchMinimizer`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `Lbfgs(int dimension, LbfgsOptions options = null)` | Initializes a new L-BFGS minimizer for the specified problem dimension. / 指定した次元に対する L-BFGS オプティマイザーを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int HistoryCount { get; }` | Gets the number of curvature pairs currently stored in history. / 現在履歴に保持している曲率ペアの数を取得します。 |
| `int HistorySize { get; }` | Gets the maximum number of curvature pairs retained in history. / 履歴に保持する曲率ペアの最大数 `m` を取得します。 |

### `class LbfgsB`

Minimizes a differentiable objective function subject to simple box bounds using the L-BFGS-B method. / L-BFGS-B 法を用いて上下限のボックス制約付き微分可能な目的関数を最小化します。

- 継承・実装: `LineSearchMinimizer`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `LbfgsB(ReadOnlySpan<double> lowerBounds, ReadOnlySpan<double> upperBounds, LbfgsBOptions options = null)` | Initializes a new L-BFGS-B minimizer over the box `[lowerBounds[i], upperBounds[i]]`. / ボックス領域 `[lowerBounds[i], upperBounds[i]]` 上で最小化を行う L-BFGS-B オプティマイザーを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int HistoryCount { get; }` | Gets the number of curvature pairs currently stored in history. / 現在履歴に保持している曲率ペアの数を取得します。 |
| `int HistorySize { get; }` | Gets the maximum number of curvature pairs retained in history. / 履歴に保持する曲率ペアの最大数 `m` を取得します。 |
| `ReadOnlySpan<double> LowerBounds { get; }` | Gets the lower bounds for each variable. / 各変数の下限を取得します。 |
| `ReadOnlySpan<double> UpperBounds { get; }` | Gets the upper bounds for each variable. / 各変数の上限を取得します。 |

### `class LbfgsBOptions`

Configures options for `LbfgsB`. / `LbfgsB` の設定オプションを表します（設定項目は `LbfgsOptions` と同一です）。

- 継承・実装: `LbfgsOptions`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `LbfgsBOptions()` |  |

### `class LbfgsOptions`

Configures options for `Lbfgs` and `LbfgsB`. / `Lbfgs` および `LbfgsB` の設定オプションを表します。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `LbfgsOptions()` |  |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Curvature { get; set; }` | Gets the curvature coefficient η for the strong Wolfe condition `\|φ′(α)\| ≤ η\|φ′(0)\|`. / 直線探索の強 Wolfe 曲率条件 `\|φ′(α)\| ≤ η\|φ′(0)\|` の係数 η を取得します。 |
| `double FunctionTolerance { get; set; }` | Gets the convergence tolerance on the relative decrease in the objective value per iteration. / 1反復あたりの目的関数値の相対減少に対する収束許容誤差を取得します。 |
| `double GradientTolerance { get; set; }` | Gets the convergence tolerance on the infinity norm of the gradient (or projected gradient for `LbfgsB`). / 勾配（`LbfgsB` では射影勾配）の最大ノルムに対する収束許容誤差を取得します。 |
| `int HistorySize { get; set; }` | Gets the maximum number of curvature pairs `(s, y)` retained in history. / 履歴として保持する曲率ペア `(s, y)` の最大数 `m` を取得します。 |
| `long? MaxEvaluations { get; set; }` | Gets the optional maximum number of objective function evaluations. / 目的関数の評価回数の上限を取得します。 |
| `int MaxIterations { get; set; }` | Gets the maximum number of iterations. / 反復回数の上限を取得します。 |
| `int MaxLineSearchEvaluations { get; set; }` | Gets the maximum number of objective evaluations allowed in a single line search. / 1回の直線探索で許容する目的関数評価回数の上限を取得します。 |
| `double? StepTolerance { get; set; }` | Gets the optional convergence tolerance on the relative step size per iteration. / 1反復あたりの相対ステップ幅に対する収束許容誤差を取得します。 |
| `double SufficientDecrease { get; set; }` | Gets the sufficient decrease (Armijo) coefficient μ used by the line search. / 直線探索の十分減少（Armijo）条件の係数 μ を取得します。 |

### `abstract class LineSearchMinimizer`

Provides a base class for gradient-based minimizers using line search. / 勾配と直線探索を用いて目的関数を最小化するオプティマイザーの基底クラスです。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Dimension { get; }` | Gets the number of variables. / 変数の数を取得します。 |
| `long Evaluations { get; }` | Gets the total number of objective evaluations (calls to `Tell`). / 目的関数の評価回数（`Tell` の呼び出し回数）を取得します。 |
| `ReadOnlyVectorSpan<double> Gradient { get; }` | Gets the gradient at the current iterate. / 現在の反復点における勾配を取得します。 |
| `double GradientNorm { get; }` | Gets the infinity norm of the gradient (or projected gradient `x − P(x − ∇f)` for `LbfgsB`) at the current iterate. / 現在の反復点における勾配（`LbfgsB` では射影勾配 `x − P(x − ∇f)`）の最大ノルムを取得します。 |
| `bool IsFinished { get; }` | Gets a value indicating whether minimization has terminated. / 最小化が終了したかどうかを取得します。 |
| `int Iterations { get; }` | Gets the number of completed iterations (accepted steps). / 反復回数（採用したステップ数）を取得します。 |
| `ReadOnlyVectorSpan<double> Point { get; }` | Gets the current iterate. / 現在の反復点を取得します。 |
| `MinimizationTermination Termination { get; }` | Gets the reason minimization terminated. / 最小化の終了理由を取得します。 |
| `double Value { get; }` | Gets the objective value at the current iterate. / 現在の反復点における目的関数値を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `ReadOnlyVectorSpan<double> Ask()` | Returns the next trial point to evaluate. / 次に評価する試行点を取得します。 |
| `MinimizationResult GetResult()` | Gets a snapshot of the current minimization result. / 現在の最小化結果のスナップショットを取得します。 |
| `MinimizationResult Minimize(Func<ReadOnlyVectorSpan<double>, VectorSpan<double>, double> objective, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)` | Minimizes the specified delegate objective function starting from `initialPoint`. / 指定した初期点からデリゲート目的関数を最小化します。 |
| `MinimizationResult Minimize<TObjective>(TObjective objective, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)` | Minimizes the specified differentiable objective function starting from `initialPoint`. / 指定した初期点から微分可能な目的関数を最小化します。 |
| `void Start(ReadOnlySpan<double> initialPoint)` | Starts a new minimization run from the specified initial point. / 指定した初期点から新しい最小化を開始します。 |
| `void Tell(double value, ReadOnlyVectorSpan<double> gradient)` | Reports the objective value and gradient for the trial point returned by `Ask` and advances the solver. / `Ask` が返した試行点での目的関数値と勾配を報告し、探索を1ステップ進めます。 |

## Tuna.Optimization.Proximal

### `class Admm`

Minimizes the sum of convex functions `f(x) + g(z)` subject to `Dx = z` using ADMM. / ADMM を用いて制約 `Dx = z` の下で凸関数の和 `f(x) + g(z)` を最小化します。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `Admm(AdmmOptions options = null)` | Initializes a new ADMM solver with the specified options. / 指定した設定で ADMM ソルバーを初期化します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `AdmmResult Minimize<TF, TG>(TF f, TG g, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)` | Solves the consensus-form problem `min f(x) + g(z)` subject to `x = z`. / 合意形式の最適化問題 `min f(x) + g(z)`（`x = z`）を解きます。 |
| `AdmmResult Minimize<TG>(IAdmmSubproblem f, TG g, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)` | Solves the splitting-form problem `min f(x) + g(z)` subject to `Dx = z`. / 分割形式の最適化問題 `min f(x) + g(z)`（`Dx = z`）を解きます。 |

### `class AdmmOptions`

Configures options for `Admm`. / `Admm` の設定オプションを表します。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `AdmmOptions()` |  |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double AbsoluteTolerance { get; set; }` | Gets the absolute residual stopping tolerance `ε_abs`. / 残差停止条件の絶対許容誤差 `ε_abs` を取得します。 |
| `bool AdaptivePenalty { get; set; }` | Gets a value indicating whether ρ is adaptively adjusted based on the primal-to-dual residual ratio. / 主残差と双対残差の比に応じて罰則係数 ρ を適応的に調整するかどうかを取得します。 |
| `int MaxIterations { get; set; }` | Gets the maximum number of iterations. / 反復回数の上限を取得します。 |
| `int MaxPenaltyUpdates { get; set; }` | Gets the maximum number of adaptive ρ updates allowed. / 罰則係数 ρ の適応更新回数の上限を取得します。 |
| `double OverRelaxation { get; set; }` | Gets the over-relaxation parameter α. / 過緩和係数 α を取得します。 |
| `double Penalty { get; set; }` | Gets the initial augmented Lagrangian penalty parameter ρ. / 拡張ラグランジュ法の罰則係数 ρ の初期値を取得します。 |
| `double PenaltyBalance { get; set; }` | Gets the residual ratio threshold μ triggering a ρ update. / 罰則係数 ρ を更新する残差比のしきい値 μ を取得します。 |
| `double PenaltyFactor { get; set; }` | Gets the multiplicative scaling factor τ applied when adjusting ρ. / 罰則係数 ρ の調整時に掛けたり割ったりする倍率 τ を取得します。 |
| `double RelativeTolerance { get; set; }` | Gets the relative residual stopping tolerance `ε_rel`. / 残差停止条件の相対許容誤差 `ε_rel` を取得します。 |

### `class AdmmResult`

Represents the result of an `Admm` minimization run. / `Admm` による最小化の実行結果を表します。

- 継承・実装: `IEquatable<AdmmResult>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `AdmmResult(Vector<double> X, Vector<double> Z, Vector<double> ScaledDual, double Value, double PrimalResidual, double DualResidual, double Penalty, int Iterations, MinimizationTermination Termination)` | Represents the result of an `Admm` minimization run. / `Admm` による最小化の実行結果を表します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `bool Converged { get; }` | Gets a value indicating whether ADMM terminated by satisfying the residual tolerances. / 残差の許容誤差を満たして収束したかどうかを取得します。 |
| `double DualResidual { get; set; }` | The Euclidean norm of the dual residual `‖ρDᵀ(z − z_old)‖₂`. / 双対残差の L2 ノルム `‖ρDᵀ(z − z_old)‖₂`。 |
| `int Iterations { get; set; }` | The number of completed iterations. / 完了した反復回数。 |
| `double Penalty { get; set; }` | The final penalty parameter ρ. / 最終的な罰則係数 ρ。 |
| `double PrimalResidual { get; set; }` | The Euclidean norm of the primal residual `‖Dx − z‖₂` (or `‖x − z‖₂` in consensus form). / 主残差の L2 ノルム `‖Dx − z‖₂`（合意形式では `‖x − z‖₂`）。 |
| `Vector<double> ScaledDual { get; set; }` | A copy of the final scaled dual variable `u = y / ρ` (Lagrange multiplier is `ρu`). / 最終スケール化双対変数 `u = y / ρ` のコピー（ラグランジュ乗数は `ρu`）。 |
| `MinimizationTermination Termination { get; set; }` | The termination reason (`ResidualTolerance`, `MaxIterations`, or `NonFiniteValue`). / 終了理由（`ResidualTolerance`、`MaxIterations`、または `NonFiniteValue`）。 |
| `double Value { get; set; }` | The composite objective value `f(x) + g(z)`. / 複合目的関数値 `f(x) + g(z)`。 |
| `Vector<double> X { get; set; }` | A copy of the final primal iterate `x`. / 最終主変数 `x` のコピー。 |
| `Vector<double> Z { get; set; }` | A copy of the final split variable `z` (lies in the domain of `g` as the output of its proximal operator). / 最終分割変数 `z` のコピー（`g` の近接写像の出力のため `g` の定義域内に属します）。 |

### `class BoxConstraint`

Represents the indicator function of a box constraint `{x | l ≤ x ≤ u}`. / ボックス制約 `{x | l ≤ x ≤ u}` の指示関数を表します。

- 継承・実装: `IProximalOperator`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `BoxConstraint(ReadOnlySpan<double> lowerBounds, ReadOnlySpan<double> upperBounds)` | Initializes a box constraint from componentwise lower and upper bounds. / 成分ごとの下限および上限からボックス制約を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `ReadOnlySpan<double> LowerBounds { get; }` | Gets the lower bounds for each component. / 各成分の下限を取得します。 |
| `ReadOnlySpan<double> UpperBounds { get; }` | Gets the upper bounds for each component. / 各成分の上限を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)` | Computes `prox_{step·h}(v)` and writes the result to `destination`. / `prox_{step·h}(v)` を計算し、`destination` に書き込みます。 |
| `double Evaluate(ReadOnlyVectorSpan<double> x)` | Evaluates the convex function `h` at the specified point. / 指定した点で凸関数 `h` を評価します。 |

### `class ElasticNetPenalty`

Represents the elastic net penalty `h(x) = λ₁‖x‖₁ + (λ₂/2)‖x‖₂²`. / Elastic Net 罰則 `h(x) = λ₁‖x‖₁ + (λ₂/2)‖x‖₂²` を表します。

- 継承・実装: `IProximalOperator`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `ElasticNetPenalty(double l1, double l2)` | Initializes an elastic net penalty with the specified L1 and squared-L2 coefficients. / 指定した L1 係数および L2 二乗係数で Elastic Net 罰則を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double L1 { get; }` | Gets the L1 coefficient λ₁. / L1 係数 λ₁ を取得します。 |
| `double L2 { get; }` | Gets the squared-L2 coefficient λ₂. / L2 二乗係数 λ₂ を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)` | Computes `prox_{step·h}(v)` and writes the result to `destination`. / `prox_{step·h}(v)` を計算し、`destination` に書き込みます。 |
| `double Evaluate(ReadOnlyVectorSpan<double> x)` | Evaluates the convex function `h` at the specified point. / 指定した点で凸関数 `h` を評価します。 |

### `class Fista`

Minimizes the sum of a smooth convex function and a proximable convex function using FISTA. / FISTA を用いて、滑らかな凸関数と近接写像を持つ凸関数の和を最小化します。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `Fista(FistaOptions options = null)` | Initializes a new FISTA solver with the specified options. / 指定した設定で FISTA ソルバーを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double LastStep { get; }` | Gets the final step size `t` used in the most recent call to `Minimize` (`1/t` estimates the Lipschitz constant). / 直近の `Minimize` 呼び出しで使用した最終ステップ幅 `t`（`1/t` がリプシッツ定数の推定値）を取得します。 |
| `int Restarts { get; }` | Gets the number of adaptive momentum restarts performed in the most recent call to `Minimize`. / 直近の `Minimize` 呼び出しで行われた適応的再始動の回数を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `MinimizationResult Minimize<TSmooth, TProximal>(TSmooth smooth, TProximal proximal, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)` | Minimizes the composite objective `f(x) + h(x)`. / 複合目的関数 `f(x) + h(x)` を最小化します。 |
| `MinimizationResult Minimize<TProximal>(Func<ReadOnlyVectorSpan<double>, VectorSpan<double>, double> smooth, TProximal proximal, ReadOnlySpan<double> initialPoint, CancellationToken cancellationToken = default)` | Minimizes the composite objective `f(x) + h(x)` using a delegate for the smooth term. / 滑らかな項をデリゲートで指定して複合目的関数 `f(x) + h(x)` を最小化します。 |

### `class FistaOptions`

Configures options for `Fista`. / `Fista` の設定オプションを表します。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `FistaOptions()` |  |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `bool AdaptiveRestart { get; set; }` | Gets a value indicating whether gradient-based adaptive momentum restart (O'Donoghue & Candès 2015) is enabled. / 勾配に基づく適応的再始動（O'Donoghue & Candès 2015）を有効にするかどうかを取得します。 |
| `double BacktrackingFactor { get; set; }` | Gets the multiplicative step-size shrinkage factor used during backtracking. / バックトラッキング時にステップ幅へ掛ける縮小倍率を取得します。 |
| `double GradientTolerance { get; set; }` | Gets the convergence tolerance on the infinity norm of the gradient mapping `(y − x⁺) / t`. / 勾配写像 `(y − x⁺) / t` の最大ノルムに対する収束許容誤差を取得します。 |
| `double? InitialStep { get; set; }` | Gets the optional initial proximal gradient step size. / 近接勾配法の初期ステップ幅を取得します。 |
| `long? MaxEvaluations { get; set; }` | Gets the optional maximum number of evaluations of the smooth term. / 滑らかな項の評価回数の上限を取得します。 |
| `int MaxIterations { get; set; }` | Gets the maximum number of iterations. / 反復回数の上限を取得します。 |

### `class GroupLassoPenalty`

Represents the group lasso penalty over contiguous variable groups `h(x) = λ Σ_g √|g| ‖x_g‖₂`. / 連続する変数グループに対するグループ Lasso 罰則 `h(x) = λ Σ_g √|g| ‖x_g‖₂` を表します。

- 継承・実装: `IProximalOperator`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `GroupLassoPenalty(double lambda, ReadOnlySpan<int> groupSizes)` | Initializes a group lasso penalty with contiguous groups of the specified sizes. / 先頭から順に指定したサイズで分割したグループに対するグループ Lasso 罰則を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int GroupCount { get; }` | Gets the number of groups. / グループ数を取得します。 |
| `double Lambda { get; }` | Gets the regularization strength λ. / 正則化係数 λ を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)` | Computes `prox_{step·h}(v)` and writes the result to `destination`. / `prox_{step·h}(v)` を計算し、`destination` に書き込みます。 |
| `double Evaluate(ReadOnlyVectorSpan<double> x)` | Evaluates the convex function `h` at the specified point. / 指定した点で凸関数 `h` を評価します。 |

### `interface IAdmmSubproblem`

Represents the `x`-update subproblem `argmin_x [f(x) + (ρ/2)‖Dx − c‖²]` and linear coupling `D` (`p × n`) for splitting-form ADMM `f(x) + g(Dx)`. / 分割形式 ADMM `f(x) + g(Dx)` における `x` 更新部分問題 `argmin_x [f(x) + (ρ/2)‖Dx − c‖²]` と線形写像 `D`（`p × n`）を表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int ConstraintCount { get; }` | Gets the dimension `p` of `z = Dx` (the number of rows of `D`). / 分割変数 `z = Dx` の次元 `p`（`D` の行数）を取得します。 |
| `int Dimension { get; }` | Gets the dimension `n` of the primal variable `x`. / 主変数 `x` の次元 `n` を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Evaluate(ReadOnlyVectorSpan<double> x)` | Evaluates `f(x)`. / `f(x)` を評価します。 |
| `void Multiply(ReadOnlyVectorSpan<double> x, VectorSpan<double> result)` | Computes `result = Dx`. / `result = Dx` を計算します。 |
| `void MultiplyTransposed(ReadOnlyVectorSpan<double> y, VectorSpan<double> result)` | Computes `result = Dᵀy`. / `result = Dᵀy` を計算します。 |
| `void Solve(ReadOnlyVectorSpan<double> target, double penalty, VectorSpan<double> x)` | Solves `argmin_x [f(x) + (ρ/2)‖Dx − target‖²]` and writes the solution to `x`. / `argmin_x [f(x) + (ρ/2)‖Dx − target‖²]` を解き、解を `x` に書き込みます。 |

### `interface IProximalOperator`

Represents a convex function `h` (including indicator functions of convex sets) with a computable proximal operator. / 近接写像を計算できる凸関数 `h`（凸集合の指示関数を含む）を表します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)` | Computes `prox_{step·h}(v)` and writes the result to `destination`. / `prox_{step·h}(v)` を計算し、`destination` に書き込みます。 |
| `double Evaluate(ReadOnlyVectorSpan<double> x)` | Evaluates the convex function `h` at the specified point. / 指定した点で凸関数 `h` を評価します。 |

### `class L1BallConstraint`

Represents the indicator function of the L1 ball `{x | ‖x‖₁ ≤ r}`. / L1 球 `{x | ‖x‖₁ ≤ r}` の指示関数を表します。

- 継承・実装: `IProximalOperator`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `L1BallConstraint(double radius)` | Initializes an L1 ball constraint with the specified radius. / 指定した半径の L1 球制約を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Radius { get; }` | Gets the radius `r` of the L1 ball. / L1 球の半径 `r` を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)` | Computes `prox_{step·h}(v)` and writes the result to `destination`. / `prox_{step·h}(v)` を計算し、`destination` に書き込みます。 |
| `double Evaluate(ReadOnlyVectorSpan<double> x)` | Evaluates the convex function `h` at the specified point. / 指定した点で凸関数 `h` を評価します。 |

### `class L1Penalty`

Represents the (optionally weighted) L1 norm penalty `h(x) = λ Σ wᵢ|xᵢ|`. / L1 ノルム罰則 `h(x) = λ Σ wᵢ|xᵢ|` を表します。

- 継承・実装: `IProximalOperator`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `L1Penalty(double lambda)` | Initializes an unweighted L1 penalty with uniform weight `lambda`. / すべての成分に共通の係数 `lambda` を適用する L1 罰則を初期化します。 |
| `L1Penalty(double lambda, ReadOnlySpan<double> weights)` | Initializes a weighted L1 penalty with componentwise weights `λwᵢ`. / 成分ごとの重み `λwᵢ` を持つ L1 罰則を初期化します。変数の次元は重みベクトルの長さで固定されます。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Lambda { get; }` | Gets the global regularization strength λ. / 全体の正則化係数 λ を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)` | Computes `prox_{step·h}(v)` and writes the result to `destination`. / `prox_{step·h}(v)` を計算し、`destination` に書き込みます。 |
| `double Evaluate(ReadOnlyVectorSpan<double> x)` | Evaluates the convex function `h` at the specified point. / 指定した点で凸関数 `h` を評価します。 |

### `class L2BallConstraint`

Represents the indicator function of the origin-centered L2 ball `{x | ‖x‖₂ ≤ r}`. / 原点を中心とする半径 `r` の L2 球 `{x | ‖x‖₂ ≤ r}` の指示関数を表します。

- 継承・実装: `IProximalOperator`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `L2BallConstraint(double radius)` | Initializes an L2 ball constraint with the specified radius. / 指定した半径の L2 球制約を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Radius { get; }` | Gets the radius `r` of the L2 ball. / L2 球の半径 `r` を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)` | Computes `prox_{step·h}(v)` and writes the result to `destination`. / `prox_{step·h}(v)` を計算し、`destination` に書き込みます。 |
| `double Evaluate(ReadOnlyVectorSpan<double> x)` | Evaluates the convex function `h` at the specified point. / 指定した点で凸関数 `h` を評価します。 |

### `class L2NormPenalty`

Represents the unsquared L2 norm penalty `h(x) = λ‖x‖₂`. / 二乗しない L2 ノルム罰則 `h(x) = λ‖x‖₂` を表します。

- 継承・実装: `IProximalOperator`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `L2NormPenalty(double lambda)` | Initializes an L2 norm penalty with coefficient `lambda`. / 指定した係数 `lambda` で L2 ノルム罰則を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Lambda { get; }` | Gets the regularization strength λ. / 正則化係数 λ を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)` | Computes `prox_{step·h}(v)` and writes the result to `destination`. / `prox_{step·h}(v)` を計算し、`destination` に書き込みます。 |
| `double Evaluate(ReadOnlyVectorSpan<double> x)` | Evaluates the convex function `h` at the specified point. / 指定した点で凸関数 `h` を評価します。 |

### `class LeastSquaresObjective`

Represents the least-squares objective `f(x) = ½‖Ax − b‖²`, usable as both a differentiable objective (for `Fista`) and a proximal operator (for `Admm`). / 最小二乗目的関数 `f(x) = ½‖Ax − b‖²` を表します。微分可能な目的関数（`Fista` の滑らかな項）としても、近接写像（`Admm` の項）としても利用できます。

- 継承・実装: `IDifferentiableObjectiveFunction`, `IObjectiveFunction`, `IProximalOperator`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `LeastSquaresObjective(Matrix<double> a, ReadOnlyVectorSpan<double> b)` | Initializes a least-squares objective `½‖Ax − b‖²`. / 最小二乗目的関数 `½‖Ax − b‖²` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Dimension { get; }` | Gets the number of variables `n`. / 変数の数 `n` を取得します。 |
| `int Rows { get; }` | Gets the number of rows `m` of the matrix `A`. / 行列 `A` の行数 `m` を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)` | Computes `prox_{step·h}(v)` and writes the result to `destination`. / `prox_{step·h}(v)` を計算し、`destination` に書き込みます。 |
| `double Evaluate(ReadOnlyVectorSpan<double> x)` | Evaluates `½‖Ax − b‖²` at `x`. / `x` における `½‖Ax − b‖²` を評価します。 |
| `double Evaluate(ReadOnlyVectorSpan<double> x, VectorSpan<double> gradient)` | Evaluates `½‖Ax − b‖²` and writes its gradient `Aᵀ(Ax − b)` to `gradient`. / `½‖Ax − b‖²` を評価し、勾配 `Aᵀ(Ax − b)` を `gradient` に書き込みます。 |

### `class NonNegativeConstraint`

Represents the indicator function of the non-negative orthant `{x | x ≥ 0}`. / 非負象限 `{x | x ≥ 0}` の指示関数を表します。

- 継承・実装: `IProximalOperator`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `NonNegativeConstraint()` |  |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)` | Computes `prox_{step·h}(v)` and writes the result to `destination`. / `prox_{step·h}(v)` を計算し、`destination` に書き込みます。 |
| `double Evaluate(ReadOnlyVectorSpan<double> x)` | Evaluates the convex function `h` at the specified point. / 指定した点で凸関数 `h` を評価します。 |

### `class QuadraticAdmmSubproblem`

Represents a splitting-form ADMM subproblem for the quadratic term `f(x) = ½xᵀPx + qᵀx + c`. / 二次形式 `f(x) = ½xᵀPx + qᵀx + c` に対する分割形式 ADMM の部分問題を表します。

- 継承・実装: `IAdmmSubproblem`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `QuadraticAdmmSubproblem(Matrix<double> p, ReadOnlyVectorSpan<double> q, Matrix<double> d, double constant = 0)` | Initializes a quadratic ADMM subproblem for `½xᵀPx + qᵀx + c` with linear operator `D`. / 二次形式 `½xᵀPx + qᵀx + c` と線形写像 `D` から ADMM 部分問題を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int ConstraintCount { get; }` | Gets the dimension `p` of `z = Dx` (the number of rows of `D`). / 分割変数 `z = Dx` の次元 `p`（`D` の行数）を取得します。 |
| `int Dimension { get; }` | Gets the dimension `n` of the primal variable `x`. / 主変数 `x` の次元 `n` を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Evaluate(ReadOnlyVectorSpan<double> x)` | Evaluates `f(x)`. / `f(x)` を評価します。 |
| `static QuadraticAdmmSubproblem LeastSquares(Matrix<double> a, ReadOnlyVectorSpan<double> b, Matrix<double> d)` | Creates a quadratic ADMM subproblem for the least-squares term `½‖Ax − b‖²`. / 最小二乗項 `½‖Ax − b‖²` に対応する ADMM 部分問題（`P = AᵀA`、`q = −Aᵀb`、`c = ½‖b‖²`）を生成します。 |
| `void Multiply(ReadOnlyVectorSpan<double> x, VectorSpan<double> result)` | Computes `result = Dx`. / `result = Dx` を計算します。 |
| `void MultiplyTransposed(ReadOnlyVectorSpan<double> y, VectorSpan<double> result)` | Computes `result = Dᵀy`. / `result = Dᵀy` を計算します。 |
| `void Solve(ReadOnlyVectorSpan<double> target, double penalty, VectorSpan<double> x)` | Solves `argmin_x [f(x) + (ρ/2)‖Dx − target‖²]` and writes the solution to `x`. / `argmin_x [f(x) + (ρ/2)‖Dx − target‖²]` を解き、解を `x` に書き込みます。 |

### `class QuadraticObjective`

Represents the convex quadratic objective `f(x) = ½xᵀPx + qᵀx + c` for symmetric positive semidefinite `P`, usable as both a differentiable objective and a proximal operator. / 対称半正定値行列 `P` に対する二次形式目的関数 `f(x) = ½xᵀPx + qᵀx + c` を表します。微分可能な目的関数としても近接写像としても利用できます。

- 継承・実装: `IDifferentiableObjectiveFunction`, `IObjectiveFunction`, `IProximalOperator`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `QuadraticObjective(Matrix<double> p, ReadOnlyVectorSpan<double> q, double constant = 0)` | Initializes a quadratic objective `½xᵀPx + qᵀx + c`. / 二次形式目的関数 `½xᵀPx + qᵀx + c` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Dimension { get; }` | Gets the number of variables `n`. / 変数の数 `n` を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)` | Computes `prox_{step·h}(v)` and writes the result to `destination`. / `prox_{step·h}(v)` を計算し、`destination` に書き込みます。 |
| `double Evaluate(ReadOnlyVectorSpan<double> x)` | Evaluates `½xᵀPx + qᵀx + c` at `x`. / `x` における `½xᵀPx + qᵀx + c` を評価します。 |
| `double Evaluate(ReadOnlyVectorSpan<double> x, VectorSpan<double> gradient)` | Evaluates `½xᵀPx + qᵀx + c` and writes its gradient `Px + q` to `gradient`. / `½xᵀPx + qᵀx + c` を評価し、勾配 `Px + q` を `gradient` に書き込みます。 |

### `class SimplexConstraint`

Represents the indicator function of the simplex `{x | x ≥ 0, Σx = s}`. / 単体 `{x | x ≥ 0, Σx = s}` の指示関数を表します。

- 継承・実装: `IProximalOperator`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `SimplexConstraint(double total = 1)` | Initializes a simplex constraint with the specified coordinate sum. / 指定した総和の単体制約を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Total { get; }` | Gets the target sum `s` of the simplex coordinates. / 単体の成分総和 `s` を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)` | Computes `prox_{step·h}(v)` and writes the result to `destination`. / `prox_{step·h}(v)` を計算し、`destination` に書き込みます。 |
| `double Evaluate(ReadOnlyVectorSpan<double> x)` | Evaluates the convex function `h` at the specified point. / 指定した点で凸関数 `h` を評価します。 |

### `class ZeroFunction`

Represents the identically zero function, whose proximal operator is the identity mapping. / 恒等的に 0 を返す関数を表します（近接写像は恒等写像です）。ADMM で一方の項が存在しない場合などに使用します。

- 継承・実装: `IProximalOperator`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `ZeroFunction()` |  |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `void Apply(ReadOnlyVectorSpan<double> v, double step, VectorSpan<double> destination)` | Computes `prox_{step·h}(v)` and writes the result to `destination`. / `prox_{step·h}(v)` を計算し、`destination` に書き込みます。 |
| `double Evaluate(ReadOnlyVectorSpan<double> x)` | Evaluates the convex function `h` at the specified point. / 指定した点で凸関数 `h` を評価します。 |

## Tuna.Optimization.Stochastic

### `class AdaGrad`

Implements the diagonal AdaGrad optimizer. / 対角型 AdaGrad オプティマイザーです。

- 継承・実装: `StochasticGradientOptimizer`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `AdaGrad(int dimension, AdaGradOptions options = null)` | Initializes a new `AdaGrad` optimizer for the specified parameter dimension. / 指定したパラメーター次元に対する `AdaGrad` オプティマイザーを初期化します。 |

### `class AdaGrad<T>`

Updates a parameter set with the AdaGrad optimizer. / AdaGrad でパラメーター一式を更新します。

- 継承・実装: `StochasticGradientOptimizer<T>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `AdaGrad(ParameterSet<T> parameters, AdaGradOptions options = null)` | Initializes an optimizer bound to the specified parameter set. / 指定したパラメーター一式を更新するオプティマイザーを初期化します。 |

### `class AdaGradOptions`

Configures options for `AdaGrad` (default learning rate 0.01). / `AdaGrad` の設定オプションを表します（基準学習率の既定値は 0.01）。

- 継承・実装: `StochasticGradientOptions`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `AdaGradOptions()` | Initializes a new instance with default AdaGrad options. / 既定の AdaGrad 設定でインスタンスを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Epsilon { get; set; }` | Gets the numerical stability constant ε added to the denominator. / 分母に加算する数値安定化定数 ε を取得します。 |
| `double InitialAccumulator { get; set; }` | Gets the initial value of the squared-gradient accumulator. / 勾配二乗和アキュムレーターの初期値を取得します。 |

### `class AdaMax`

Implements the AdaMax optimizer, an infinity-norm variant of Adam. / Adam の 2 次モーメントを無限大ノルムに置き換えた AdaMax オプティマイザーです。

- 継承・実装: `StochasticGradientOptimizer`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `AdaMax(int dimension, AdaMaxOptions options = null)` | Initializes a new `AdaMax` optimizer for the specified parameter dimension. / 指定したパラメーター次元に対する `AdaMax` オプティマイザーを初期化します。 |

### `class AdaMax<T>`

Updates a parameter set with the AdaMax optimizer. / AdaMax でパラメーター一式を更新します。

- 継承・実装: `StochasticGradientOptimizer<T>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `AdaMax(ParameterSet<T> parameters, AdaMaxOptions options = null)` | Initializes an optimizer bound to the specified parameter set. / 指定したパラメーター一式を更新するオプティマイザーを初期化します。 |

### `class AdaMaxOptions`

Configures options for `AdaMax` (default learning rate 0.002). / `AdaMax` の設定オプションを表します（基準学習率の既定値は 0.002）。

- 継承・実装: `StochasticGradientOptions`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `AdaMaxOptions()` | Initializes a new instance with default AdaMax options. / 既定の AdaMax 設定でインスタンスを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Beta1 { get; set; }` | Gets the exponential decay rate β₁ for the first-moment estimate. / 1次モーメント推定の指数移動平均係数 β₁ を取得します。 |
| `double Beta2 { get; set; }` | Gets the exponential decay rate β₂ for the infinity-norm estimate. / 無限大ノルム推定の指数減衰係数 β₂ を取得します。 |

### `class Adam`

Implements the Adam optimizer, supporting AMSGrad and decoupled weight decay (AdamW). / Adam オプティマイザーです（AMSGrad および切り離した重み減衰による AdamW に対応）。

- 継承・実装: `StochasticGradientOptimizer`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `Adam(int dimension, AdamOptions options = null)` | Initializes a new `Adam` optimizer for the specified parameter dimension. / 指定したパラメーター次元に対する `Adam` オプティマイザーを初期化します。 |

### `class Adam<T>`

Updates a parameter set with Adam, including optional AMSGrad or decoupled weight decay. / Adam でパラメーター一式を更新します。AMSGrad と切り離し重み減衰も使用できます。

- 継承・実装: `StochasticGradientOptimizer<T>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `Adam(ParameterSet<T> parameters, AdamOptions options = null)` | Initializes an optimizer bound to the specified parameter set. / 指定したパラメーター一式を更新するオプティマイザーを初期化します。 |

### `class AdamOptions`

Configures options for `Adam` (default learning rate 0.001). / `Adam` の設定オプションを表します（基準学習率の既定値は 0.001）。

- 継承・実装: `StochasticGradientOptions`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `AdamOptions()` | Initializes a new instance with default Adam options. / 既定の Adam 設定でインスタンスを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `bool Amsgrad { get; set; }` | Gets a value indicating whether to use the AMSGrad variant (maintaining the running maximum of the second moment). / 2次モーメントのこれまでの最大値を分母に使用する AMSGrad バリアントを有効にするかどうかを取得します。 |
| `double Beta1 { get; set; }` | Gets the exponential decay rate β₁ for the first-moment estimate. / 1次モーメント推定の指数移動平均係数 β₁ を取得します。 |
| `double Beta2 { get; set; }` | Gets the exponential decay rate β₂ for the second-moment estimate. / 2次モーメント推定の指数移動平均係数 β₂ を取得します。 |
| `double Epsilon { get; set; }` | Gets the numerical stability constant ε added to the denominator. / 分母に加算する数値安定化定数 ε を取得します。 |

### `class CosineAnnealingSchedule`

Implements a cosine annealing learning-rate schedule, optionally with periodic restarts. / 余弦曲線に沿って学習率を減衰させるコサインアニーリングスケジュールを表します（周期的リスタートにも対応）。

- 継承・実装: `ILearningRateSchedule`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `CosineAnnealingSchedule(long period, double minimumMultiplier = 0, bool restart = false)` | Initializes a cosine annealing schedule. / コサインアニーリングスケジュールを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double MinimumMultiplier { get; }` | Gets the minimum learning-rate multiplier. / 最小倍率を取得します。 |
| `long Period { get; }` | Gets the decay period in update steps. / 減衰周期のステップ数を取得します。 |
| `bool Restart { get; }` | Gets a value indicating whether the schedule restarts every `Period` steps. / `Period` ステップごとに倍率を 1 に戻すかどうかを取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Multiplier(long step)` | Computes the learning-rate multiplier for the specified update step. / 指定した更新ステップにおける学習率の倍率を計算します。 |

### `class ExponentialDecaySchedule`

Implements a continuous exponential decay learning-rate schedule `r^(t / s)`. / 連続的な指数減衰スケジュール `r^(t / s)` を表します（`s` 回の更新で倍率 `r`）。

- 継承・実装: `ILearningRateSchedule`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `ExponentialDecaySchedule(double rate, long decaySteps = 1)` | Initializes an exponential decay schedule. / 指数減衰スケジュールを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `long DecaySteps { get; }` | Gets the number of updates corresponding to `Rate`. / 倍率 `Rate` に対応する更新回数を取得します。 |
| `double Rate { get; }` | Gets the decay multiplier applied per `DecaySteps` updates. / `DecaySteps` 回の更新ごとに掛かる倍率を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Multiplier(long step)` | Computes the learning-rate multiplier for the specified update step. / 指定した更新ステップにおける学習率の倍率を計算します。 |

### `interface ILearningRateSchedule`

Represents a learning-rate schedule that returns a multiplier for the base learning rate at each update step. / 各更新ステップにおいて基準学習率に掛ける倍率を返す学習率スケジュールを表します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Multiplier(long step)` | Computes the learning-rate multiplier for the specified update step. / 指定した更新ステップにおける学習率の倍率を計算します。 |

### `class InverseTimeDecaySchedule`

Implements an inverse-time decay learning-rate schedule `1 / (1 + λt)`. / 時間に反比例して減衰する学習率スケジュール `1 / (1 + λt)` を表します。

- 継承・実装: `ILearningRateSchedule`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `InverseTimeDecaySchedule(double decay)` | Initializes an inverse-time decay schedule. / 逆時間減衰スケジュールを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Decay { get; }` | Gets the decay coefficient λ. / 減衰係数 λ を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Multiplier(long step)` | Computes the learning-rate multiplier for the specified update step. / 指定した更新ステップにおける学習率の倍率を計算します。 |

### `class LinearWarmupSchedule`

Implements linear warmup `(t + 1) / W` over the first `W` steps, followed by an optional schedule. / 最初の `W` ステップで倍率を `(t + 1) / W` と線形に 1 まで立ち上げ、その後は指定したスケジュールに従うウォームアップスケジュールを表します。

- 継承・実装: `ILearningRateSchedule`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `LinearWarmupSchedule(long warmupSteps, ILearningRateSchedule after = null)` | Initializes a linear warmup schedule. / 線形ウォームアップスケジュールを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `ILearningRateSchedule After { get; }` | Gets the schedule applied after warmup completes, or `null` for constant multiplier 1. / ウォームアップ完了後に適用されるスケジュール（未指定時は `null`）を取得します。 |
| `long WarmupSteps { get; }` | Gets the number of linear warmup steps. / ウォームアップのステップ数を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Multiplier(long step)` | Computes the learning-rate multiplier for the specified update step. / 指定した更新ステップにおける学習率の倍率を計算します。 |

### `class NAdam`

Implements the NAdam optimizer (Adam with Nesterov momentum and constant `β₁`). / Nesterov モーメンタムを組み込んだ NAdam オプティマイザーです（モーメンタム係数 `β₁` は一定）。

- 継承・実装: `StochasticGradientOptimizer`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `NAdam(int dimension, NAdamOptions options = null)` | Initializes a new `NAdam` optimizer for the specified parameter dimension. / 指定したパラメーター次元に対する `NAdam` オプティマイザーを初期化します。 |

### `class NAdam<T>`

Updates a parameter set with the NAdam optimizer. / NAdam でパラメーター一式を更新します。

- 継承・実装: `StochasticGradientOptimizer<T>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `NAdam(ParameterSet<T> parameters, NAdamOptions options = null)` | Initializes an optimizer bound to the specified parameter set. / 指定したパラメーター一式を更新するオプティマイザーを初期化します。 |

### `class NAdamOptions`

Configures options for `NAdam` (default learning rate 0.002). / `NAdam` の設定オプションを表します（基準学習率の既定値は 0.002）。

- 継承・実装: `StochasticGradientOptions`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `NAdamOptions()` | Initializes a new instance with default NAdam options. / 既定の NAdam 設定でインスタンスを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Beta1 { get; set; }` | Gets the exponential decay rate β₁ for the first-moment estimate. / 1次モーメント推定の指数移動平均係数 β₁ を取得します。 |
| `double Beta2 { get; set; }` | Gets the exponential decay rate β₂ for the second-moment estimate. / 2次モーメント推定の指数移動平均係数 β₂ を取得します。 |
| `double Epsilon { get; set; }` | Gets the numerical stability constant ε added to the denominator. / 分母に加算する数値安定化定数 ε を取得します。 |

### `class RmsProp`

Implements the RMSProp optimizer with optional centering and momentum. / 中心化およびモーメンタムに対応した RMSProp オプティマイザーです。

- 継承・実装: `StochasticGradientOptimizer`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `RmsProp(int dimension, RmsPropOptions options = null)` | Initializes a new `RmsProp` optimizer for the specified parameter dimension. / 指定したパラメーター次元に対する `RmsProp` オプティマイザーを初期化します。 |

### `class RmsProp<T>`

Updates a parameter set with RMSProp, with optional centering and momentum. / RMSProp でパラメーター一式を更新します。中心化とモーメンタムも使用できます。

- 継承・実装: `StochasticGradientOptimizer<T>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `RmsProp(ParameterSet<T> parameters, RmsPropOptions options = null)` | Initializes an optimizer bound to the specified parameter set. / 指定したパラメーター一式を更新するオプティマイザーを初期化します。 |

### `class RmsPropOptions`

Configures options for `RmsProp` (default learning rate 0.001). / `RmsProp` の設定オプションを表します（基準学習率の既定値は 0.001）。

- 継承・実装: `StochasticGradientOptions`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `RmsPropOptions()` | Initializes a new instance with default RMSProp options. / 既定の RMSProp 設定でインスタンスを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `bool Centered { get; set; }` | Gets a value indicating whether to use centered RMSProp (normalizing by the estimated variance `s − m²`). / 勾配の移動平均を引いた分散推定値 `s − m²` で正規化する中心化 RMSProp を使用するかどうかを取得します。 |
| `double Decay { get; set; }` | Gets the exponential moving average decay factor ρ for squared gradients. / 勾配二乗の指数移動平均係数 ρ を取得します。 |
| `double Epsilon { get; set; }` | Gets the numerical stability constant ε added to the denominator. / 分母に加算する数値安定化定数 ε を取得します。 |
| `double Momentum { get; set; }` | Gets the momentum coefficient μ. / モーメンタム係数 μ を取得します。 |

### `class StepDecaySchedule`

Implements a step decay learning-rate schedule `γ^⌊t / s⌋`. / 一定間隔ごとに倍率を掛けるステップ減衰スケジュール `γ^⌊t / s⌋` を表します。

- 継承・実装: `ILearningRateSchedule`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `StepDecaySchedule(long stepSize, double factor)` | Initializes a step decay schedule. / ステップ減衰スケジュールを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Factor { get; }` | Gets the decay factor applied at each interval. / 間隔ごとに掛ける倍率を取得します。 |
| `long StepSize { get; }` | Gets the number of steps between multiplier drops. / 倍率を減衰させる間隔を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `double Multiplier(long step)` | Computes the learning-rate multiplier for the specified update step. / 指定した更新ステップにおける学習率の倍率を計算します。 |

### `class StochasticGradientDescent`

Implements stochastic gradient descent (SGD) with optional classical or Nesterov momentum. / モーメンタム（Polyak 型および Nesterov 型）に対応した確率的勾配降下法（SGD）オプティマイザーです。

- 継承・実装: `StochasticGradientOptimizer`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `StochasticGradientDescent(int dimension, StochasticGradientDescentOptions options = null)` | Initializes a new `StochasticGradientDescent` optimizer for the specified parameter dimension. / 指定したパラメーター次元に対する `StochasticGradientDescent` オプティマイザーを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Momentum { get; }` | Gets the momentum coefficient μ. / モーメンタム係数 μ を取得します。 |
| `bool Nesterov { get; }` | Gets a value indicating whether Nesterov momentum is active. / Nesterov モーメンタムが有効かどうかを取得します。 |

### `class StochasticGradientDescent<T>`

Updates a parameter set with stochastic gradient descent, with optional momentum. / 確率的勾配降下法でパラメーター一式を更新します。モーメンタムも使用できます。

- 継承・実装: `StochasticGradientOptimizer<T>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `StochasticGradientDescent(ParameterSet<T> parameters, StochasticGradientDescentOptions options = null)` | Initializes an optimizer bound to the specified parameter set. / 指定したパラメーター一式を更新するオプティマイザーを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Momentum { get; }` | Gets the momentum coefficient μ. / モーメンタム係数 μ を取得します。 |
| `bool Nesterov { get; }` | Gets a value indicating whether Nesterov momentum is active. / Nesterov モーメンタムが有効かどうかを取得します。 |

### `class StochasticGradientDescentOptions`

Configures options for `StochasticGradientDescent` (default learning rate 0.01). / `StochasticGradientDescent` の設定オプションを表します（基準学習率の既定値は 0.01）。

- 継承・実装: `StochasticGradientOptions`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `StochasticGradientDescentOptions()` | Initializes a new instance with default SGD options. / 既定の SGD 設定でインスタンスを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Momentum { get; set; }` | Gets the momentum coefficient μ. / モーメンタム係数 μ を取得します。 |
| `bool Nesterov { get; set; }` | Gets a value indicating whether Nesterov momentum is enabled. / Nesterov モーメンタムを使用するかどうかを取得します。 |

### `abstract class StochasticGradientOptimizer`

Provides a base class for stochastic gradient optimizers that update parameters from supplied gradients. / 与えられた勾配からパラメーターを更新する確率的勾配法オプティマイザーの基底クラスです。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double CurrentLearningRate { get; }` | Gets the effective learning rate that will be used on the next `Step` (`LearningRate` multiplied by the schedule multiplier). / 次の `Step` で使用される実効学習率（`LearningRate` × スケジュール倍率）を取得します。 |
| `int Dimension { get; }` | Gets the number of parameters. / パラメーターの数を取得します。 |
| `double LearningRate { get; set; }` | Gets or sets the base learning rate (may be modified between steps). / 基準学習率を取得または設定します（更新の合間に変更できます）。 |
| `ILearningRateSchedule Schedule { get; }` | Gets the learning-rate schedule, or `null` if constant. / 学習率スケジュールを取得します（一定の場合は `null`）。 |
| `long StepCount { get; }` | Gets the number of updates performed so far (reset to 0 by `Reset`). / これまでに行った更新回数を取得します（`Reset` で 0 に戻ります）。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `MinimizationResult Minimize(Func<ReadOnlyVectorSpan<double>, VectorSpan<double>, double> objective, ReadOnlySpan<double> initialPoint, int maxIterations, double gradientTolerance = 0, CancellationToken cancellationToken = default)` | Minimizes a delegate objective function for up to `maxIterations` updates. / デリゲート目的関数を評価しながら、最大 `maxIterations` 回の更新で最小化します。 |
| `MinimizationResult Minimize<TObjective>(TObjective objective, ReadOnlySpan<double> initialPoint, int maxIterations, double gradientTolerance = 0, CancellationToken cancellationToken = default)` | Minimizes a differentiable objective function for up to `maxIterations` updates. / 微分可能な目的関数を評価しながら、最大 `maxIterations` 回の更新で最小化します。 |
| `void Reset()` | Resets the step count and all internal optimizer state (such as moment estimates). / 更新回数とオプティマイザーの内部状態（モーメントなど）を初期化します。 |
| `void Step(VectorSpan<double> parameters, ReadOnlyVectorSpan<double> gradient)` | Performs a single in-place parameter update using the specified gradient. / 指定した勾配を用いてパラメーターをインプレースで1回更新します。 |

### `abstract class StochasticGradientOptimizer<T>`

Updates a fixed set of learnable parameters using their accumulated gradients. / 逆伝播で加算された勾配を使用して、初期化時に指定した学習パラメーターを更新します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double CurrentLearningRate { get; }` | Gets the learning rate for the next update, including the schedule multiplier. / 学習率スケジュールの倍率を反映した次回の学習率を取得します。 |
| `long Dimension { get; }` | Gets the total number of parameter elements. / パラメーターの総要素数を取得します。 |
| `double LearningRate { get; set; }` | Gets or sets the base learning rate. / 学習率スケジュールを適用する前の基準学習率を取得または設定します。 |
| `ParameterSet<T> Parameters { get; }` | Gets the parameter set bound to this optimizer. / このオプティマイザーが更新するパラメーター一式を取得します。 |
| `ILearningRateSchedule Schedule { get; }` | Gets the learning rate schedule, or `null` for a constant learning rate. / 学習率スケジュールを取得します。固定の学習率の場合は `null` です。 |
| `long StepCount { get; }` | Gets the number of completed updates since initialization or the last reset. / 初期化または直近のリセットからの更新回数を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `void Reset()` | Resets the update count and optimizer state. / 更新回数とモーメントなどのオプティマイザーの状態を初期化します。 |
| `void Step()` | Updates all bound parameter values in place using their accumulated gradients. / 加算された勾配に基づき、パラメーターの値をまとめてインプレースで更新します。 |

### `abstract class StochasticGradientOptions`

Provides common configuration options for stochastic gradient optimizers. / 確率的勾配法オプティマイザーに共通する設定オプションの基底クラスです。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `bool DecoupledWeightDecay { get; set; }` | Gets a value indicating whether weight decay is decoupled from the gradient update (as in AdamW). / 重み減衰を勾配更新から切り離して適用するかどうか（AdamW 方式）を取得します。 |
| `double? GradientClipNorm { get; set; }` | Gets the optional maximum Euclidean norm for gradient clipping. / 勾配クリッピングにおける L2 ノルム上限を取得します。 |
| `double LearningRate { get; set; }` | Gets the base learning rate η. / 基準学習率 η を取得します。 |
| `ILearningRateSchedule Schedule { get; set; }` | Gets the optional learning-rate schedule. / 学習率スケジュールを取得します。 |
| `double WeightDecay { get; set; }` | Gets the weight decay coefficient λ. / 重み減衰係数 λ を取得します。 |

