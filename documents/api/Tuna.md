# Tuna API リファレンス

> `(+N overloads)` は `Vector` / `VectorSpan` / `ReadOnlyVectorSpan` 等のコンテナ・ビュー組み合わせや `double` / `float` 固定型オーバーロードの省略を表します。

## 名前空間と公開型

- [`Tuna`](#tuna) (19 型): [`ElementwiseOperationAttribute`](#class-elementwiseoperationattribute), [`FixedElementTypeOverloadsAttribute`](#class-fixedelementtypeoverloadsattribute), [`IBinaryOperator<T>`](#interface-ibinaryoperatort), [`IIndexOperator<T>`](#interface-iindexoperatort), [`IPairwiseOperator<T>`](#interface-ipairwiseoperatort), [`IStatefulBinaryOperator<T>`](#interface-istatefulbinaryoperatort), [`IStatefulIndexOperator<T>`](#interface-istatefulindexoperatort), [`IStatefulPairwiseOperator<T>`](#interface-istatefulpairwiseoperatort), [`IStatefulTensorIndexOperator<T>`](#interface-istatefultensorindexoperatort), [`IStatefulTernaryOperator<T>`](#interface-istatefulternaryoperatort), [`IStatefulUnaryOperator<T>`](#interface-istatefulunaryoperatort), [`IStatefulVectorIndexOperator<T>`](#interface-istatefulvectorindexoperatort), [`ITensorIndexOperator<T>`](#interface-itensorindexoperatort), [`ITernaryOperator<T>`](#interface-iternaryoperatort), [`IUnaryOperator<T>`](#interface-iunaryoperatort), [`IVectorIndexOperator<T>`](#interface-ivectorindexoperatort), [`InferenceOverloadsAttribute`](#class-inferenceoverloadsattribute), [`PairwiseOperationAttribute`](#class-pairwiseoperationattribute), [`XMath`](#static-class-xmath)
- [`Tuna.LinearAlgebra`](#tunalinearalgebra) (25 型): [`BatchCholeskyDecomposition`](#class-batchcholeskydecomposition), [`Blas`](#static-class-blas), [`BlasDiagonal`](#enum-blasdiagonal), [`BlasTranspose`](#enum-blastranspose), [`BlasTriangle`](#enum-blastriangle), [`CholeskyDecomposition`](#class-choleskydecomposition), [`DenseExtensions`](#static-class-denseextensions), [`EigenDecomposition`](#class-eigendecomposition), [`LinearAlgebraSettings`](#static-class-linearalgebrasettings), [`LuDecomposition`](#class-ludecomposition), [`Matrix<TElement>`](#class-matrixtelement), [`MatrixSpan<TElement>`](#readonly-ref-struct-matrixspantelement), [`QrDecomposition`](#class-qrdecomposition), [`ReadOnlyMatrixSpan<TElement>`](#readonly-ref-struct-readonlymatrixspantelement), [`ReadOnlyTensorMatrices<TElement>`](#readonly-ref-struct-readonlytensormatricestelement), [`ReadOnlyTensorSpan<TElement>`](#readonly-ref-struct-readonlytensorspantelement), [`ReadOnlyVectorSpan<T>`](#readonly-ref-struct-readonlyvectorspant), [`SingularValueDecomposition`](#class-singularvaluedecomposition), [`SymmetricEigenDecomposition`](#class-symmetriceigendecomposition), [`Tensor<TElement>`](#class-tensortelement), [`TensorMatrices<TElement>`](#readonly-ref-struct-tensormatricestelement), [`TensorSpan<TElement>`](#readonly-ref-struct-tensorspantelement), [`TwoStageReduction`](#enum-twostagereduction), [`Vector<T>`](#class-vectort), [`VectorSpan<T>`](#readonly-ref-struct-vectorspant)
- [`Tuna.QuasiRandom`](#tunaquasirandom) (2 型): [`IPointSequence`](#interface-ipointsequence), [`SobolSequence`](#class-sobolsequence)
- [`Tuna.RandomSource`](#tunarandomsource) (9 型): [`BoundedCompositionSampler`](#class-boundedcompositionsampler), [`IRandomSource`](#interface-irandomsource), [`Pcg64DxsmRandom`](#class-pcg64dxsmrandom), [`Pcg64DxsmState`](#readonly-struct-pcg64dxsmstate), [`Philox4x32Block`](#readonly-struct-philox4x32block), [`Philox4x32Random`](#class-philox4x32random), [`Philox4x32State`](#readonly-struct-philox4x32state), [`RandomGenerator`](#abstract-class-randomgenerator), [`RandomSourceExtensions`](#static-class-randomsourceextensions)

## Tuna

### `class ElementwiseOperationAttribute`

Generates elementwise vector and matrix overloads and a corresponding operator struct for a scalar method. / スカラーメソッドに対して、ベクトル・行列向けの要素ごとのオーバーロードと対応する演算子構造体を生成します。

- 継承・実装: `Attribute`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `ElementwiseOperationAttribute()` |  |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `string Derivative { get; set; }` | Gets or sets the derivative method name for a single-input operation. / 単一入力の演算に対応する導関数のメソッド名を取得または設定します。 |
| `string OperatorContainer { get; set; }` | Gets or sets the name of a nested static partial class in which to place the generated operator struct. / 生成する演算子構造体を配置する入れ子の `public static partial` クラス名を取得または設定します。 |
| `string OperatorName { get; set; }` | Gets or sets the name of the generated operator struct. / 生成する演算子構造体の名前を取得または設定します。 |
| `string[] Over { get; set; }` | Gets or sets the names of 1 to 3 parameters to iterate elementwise. / 要素ごとに反復する 1〜3 個の引数名を取得または設定します。 |
| `string[] PartialDerivatives { get; set; }` | Gets or sets the partial derivative method names for two or three inputs in `Over` order. / 2 個または 3 個の入力に対応する偏微分のメソッド名を `Over` の指定順で取得または設定します。 |

### `class FixedElementTypeOverloadsAttribute`

Generates overloads with the first type parameter fixed to each specified element type. / 最初の型引数（要素型）を指定した各型に固定したオーバーロードを生成します。

- 継承・実装: `Attribute`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `FixedElementTypeOverloadsAttribute(params Type[] elementTypes)` | Initializes a new instance of the `FixedElementTypeOverloadsAttribute` class with the element types to fix. / 固定する要素型の一覧を指定して `FixedElementTypeOverloadsAttribute` クラスの新しいインスタンスを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `Type[] ElementTypes { get; }` | Gets the element types for which fixed overloads are generated. / 固定オーバーロードを生成する要素型の一覧を取得します。 |

### `interface IBinaryOperator<T>`

Represents a stateless binary elementwise operator. / 状態を持たない2引数の要素ごと演算を表します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static T Invoke(T x, T y)` | Applies the operator to a pair of elements. / 2つの要素に演算を適用します。 |

### `interface IIndexOperator<T>`

Represents a stateless operator that computes a matrix element from its row and column indices. / 行と列のインデックスから行列の要素を計算する、状態を持たない演算を表します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static T Invoke(int row, int column)` | Computes the element at the specified zero-based row and column indices. / 指定した 0 始まりの行・列インデックスに対応する要素を計算します。 |

### `interface IPairwiseOperator<T>`

Represents a stateless operator that maps two column vectors to a scalar (such as a distance or kernel function). / 2つの列ベクトルから1つのスカラー値を計算する、状態を持たない演算（距離関数やカーネル関数など）を表します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static T Invoke(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)` | Applies the operator to a pair of vectors. / 2つのベクトルに演算を適用します。 |

### `interface IStatefulBinaryOperator<T>`

Represents a stateful binary elementwise operator. / 状態（係数など）を持つ2引数の要素ごと演算を表します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `T Invoke(T x, T y)` | Applies the operator to a pair of elements. / 2つの要素に演算を適用します。 |

### `interface IStatefulIndexOperator<T>`

Represents a stateful operator that computes a matrix element from its row and column indices. / 状態（係数や節点情報など）を持ち、行と列のインデックスから行列の要素を計算する演算を表します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `T Invoke(int row, int column)` | Computes the element at the specified zero-based row and column indices. / 指定した 0 始まりの行・列インデックスに対応する要素を計算します。 |

### `interface IStatefulPairwiseOperator<T>`

Represents a stateful operator that maps two column vectors to a scalar. / 状態（長さ尺度など）を持ち、2つの列ベクトルから1つのスカラー値を計算する演算を表します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `T Invoke(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)` | Applies the operator to a pair of vectors. / 2つのベクトルに演算を適用します。 |

### `interface IStatefulTensorIndexOperator<T>`

Represents a stateful operator that computes a tensor element from its multi-dimensional indices. / 状態を持ち、全軸のインデックスからテンソルの要素を計算する演算を表します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `T Invoke(ReadOnlySpan<int> indices)` | Computes the element at the specified zero-based indices. / 指定した 0 始まりの全軸インデックスに対応する要素を計算します。 |

### `interface IStatefulTernaryOperator<T>`

Represents a stateful ternary elementwise operator. / 状態（係数など）を持つ3引数の要素ごと演算を表します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `T Invoke(T x, T y, T z)` | Applies the operator to three elements. / 3つの要素に演算を適用します。 |

### `interface IStatefulUnaryOperator<T>`

Represents a stateful unary elementwise operator. / 状態（係数など）を持つ1引数の要素ごと演算を表します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `T Invoke(T x)` | Applies the operator to a single element. / 1つの要素に演算を適用します。 |

### `interface IStatefulVectorIndexOperator<T>`

Represents a stateful operator that computes a vector element from its index. / 状態を持ち、インデックスからベクトルの要素を計算する演算を表します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `T Invoke(int index)` | Computes the element at the specified zero-based index. / 指定した 0 始まりのインデックスに対応する要素を計算します。 |

### `interface ITensorIndexOperator<T>`

Represents a stateless operator that computes a tensor element from its multi-dimensional indices. / 全軸のインデックスからテンソルの要素を計算する、状態を持たない演算を表します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static T Invoke(ReadOnlySpan<int> indices)` | Computes the element at the specified zero-based indices. / 指定した 0 始まりの全軸インデックスに対応する要素を計算します。 |

### `interface ITernaryOperator<T>`

Represents a stateless ternary elementwise operator. / 状態を持たない3引数の要素ごと演算を表します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static T Invoke(T x, T y, T z)` | Applies the operator to three elements. / 3つの要素に演算を適用します。 |

### `interface IUnaryOperator<T>`

Represents a stateless unary elementwise operator. / 状態を持たない1引数の要素ごと演算を表します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static T Invoke(T x)` | Applies the operator to a single element. / 1つの要素に演算を適用します。 |

### `interface IVectorIndexOperator<T>`

Represents a stateless operator that computes a vector element from its index. / インデックスからベクトルの要素を計算する、状態を持たない演算を表します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static T Invoke(int index)` | Computes the element at the specified zero-based index. / 指定した 0 始まりのインデックスに対応する要素を計算します。 |

### `class InferenceOverloadsAttribute`

Generates overloads that accept containers and writable views so generic element types can be inferred at call sites. / ビューを受け取るジェネリックメソッドに対して、コンテナや書き込み可能ビューから要素型を推論できるオーバーロードを生成します。

- 継承・実装: `Attribute`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `InferenceOverloadsAttribute()` |  |

### `class PairwiseOperationAttribute`

Generates pairwise matrix overloads and a corresponding operator struct for a two-vector scalar method. / 2つのベクトルからスカラーを返すメソッドに対して、行列の列ペアごとのオーバーロードと対応する演算子構造体を生成します。

- 継承・実装: `Attribute`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `PairwiseOperationAttribute()` |  |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `string OperatorContainer { get; set; }` | Gets or sets the name of a nested static partial class in which to place the generated operator struct. / 生成する演算子構造体を配置する入れ子クラス名を取得または設定します。 |
| `string OperatorName { get; set; }` | Gets or sets the name of the generated operator struct. / 生成する演算子構造体の名前を取得または設定します。 |

### `static class XMath`

Provides scalar elementary functions and numerical operations for vectors, matrices, and tensors. / スカラーの初等関数およびベクトル・行列・テンソル向けの数値演算を提供します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static T Abs<T>(T x)` | Computes the absolute value `\|x\|`. / 絶対値 `\|x\|` を計算します。 |
| `static Vector<T> Abs<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Applies `Abs` elementwise to the input vector and returns a new vector. / `Abs` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Abs<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Applies `Abs` elementwise to the input matrix and returns a new matrix. / `Abs` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static VectorSpan<T> Abs<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Applies `Abs` elementwise to the input vector, writes the result to `destination`, and returns it. / `Abs` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Abs<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Abs` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Abs` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static Tensor<T> Add<T>(ReadOnlyTensorSpan<T> x, ReadOnlyTensorSpan<T> y)  (+2 overloads)` | Computes the elementwise sum `X[k] + Y[k]` of two tensors. / 2つのテンソルの要素ごとの和 `X[k] + Y[k]` を計算します。 |
| `static Vector<T> Add<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)  (+2 overloads)` | Computes the elementwise sum `x + y` of two vectors. / 2つのベクトルの要素ごとの和 `x + y` を計算します。 |
| `static Matrix<T> Add<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y)  (+2 overloads)` | Computes the elementwise sum `x + y` of two matrices. / 2つの行列の要素ごとの和 `x + y` を計算します。 |
| `static Vector<T> Add<T>(ReadOnlyVectorSpan<T> x, T scalar)` | Adds a scalar to every element of a vector (`x + scalar`). / ベクトルの全要素にスカラーを加算します（`x + scalar`）。 |
| `static Matrix<T> Add<T>(ReadOnlyMatrixSpan<T> x, T scalar)` | Adds a scalar to every element of a matrix (`x + scalar`). / 行列の全要素にスカラーを加算します（`x + scalar`）。 |
| `static TensorSpan<T> Add<T>(ReadOnlyTensorSpan<T> x, ReadOnlyTensorSpan<T> y, TensorSpan<T> destination)  (+2 overloads)` | Computes the elementwise sum `X[k] + Y[k]` of two tensors and writes the result to `destination`. / 2つのテンソルの要素ごとの和 `X[k] + Y[k]` を計算し、`destination` に書き込みます。 |
| `static VectorSpan<T> Add<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, VectorSpan<T> destination)  (+2 overloads)` | Computes the elementwise sum `x + y` of two vectors and writes the result to `destination`. / 2つのベクトルの要素ごとの和 `x + y` を計算し、`destination` に書き込みます。 |
| `static MatrixSpan<T> Add<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, MatrixSpan<T> destination)  (+2 overloads)` | Computes the elementwise sum `x + y` of two matrices and writes the result to `destination`. / 2つの行列の要素ごとの和 `x + y` を計算し、`destination` に書き込みます。 |
| `static VectorSpan<T> Add<T>(ReadOnlyVectorSpan<T> x, T scalar, VectorSpan<T> destination)` | Adds a scalar to every element of a vector and writes the result to `destination`. / ベクトルの全要素にスカラーを加算し、`destination` に書き込みます。 |
| `static MatrixSpan<T> Add<T>(ReadOnlyMatrixSpan<T> x, T scalar, MatrixSpan<T> destination)` | Adds a scalar to every element of a matrix and writes the result to `destination`. / 行列の全要素にスカラーを加算し、`destination` に書き込みます。 |
| `static int[] ArgSort<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Returns the zero-based indices that would sort `x` in ascending order. / `x` を昇順に並べ替えるための 0 始まりのインデックス配列を返します。 |
| `static Span<int> ArgSort<T>(ReadOnlyVectorSpan<T> x, Span<int> destination)  (+2 overloads)` | Computes the zero-based indices that would sort `x` in ascending order and writes them to `destination`. / `x` を昇順に並べ替える 0 始まりのインデックスを計算し、`destination` に書き込みます。 |
| `static double CholeskyLogDeterminant(ReadOnlyMatrixSpan<double> a)` | Computes the natural logarithm of the determinant of a symmetric positive definite matrix. / 対称正定値行列の行列式の自然対数を計算します。 |
| `static T Clamp<T>(T x, T min, T max)` | Clamps `x` to the inclusive range `[min, max]`. / `x` を閉区間 `[min, max]` に収めます。 |
| `static Vector<T> Clamp<T>(ReadOnlyVectorSpan<T> x, T min, T max)  (+2 overloads)` | Applies `Clamp` elementwise to the input vector and returns a new vector. / `Clamp` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Clamp<T>(ReadOnlyMatrixSpan<T> x, T min, T max)  (+2 overloads)` | Applies `Clamp` elementwise to the input matrix and returns a new matrix. / `Clamp` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static VectorSpan<T> Clamp<T>(ReadOnlyVectorSpan<T> x, T min, T max, VectorSpan<T> destination)  (+2 overloads)` | Applies `Clamp` elementwise to the input vector, writes the result to `destination`, and returns it. / `Clamp` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Clamp<T>(ReadOnlyMatrixSpan<T> x, T min, T max, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Clamp` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Clamp` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static int[] ColumnIndexOfMax<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Returns the zero-based row index of the first maximum element in each column of a matrix. / 行列の各列について、最大値を持つ最初の要素の行インデックスを返します。 |
| `static Span<int> ColumnIndexOfMax<T>(ReadOnlyMatrixSpan<T> x, Span<int> destination)  (+2 overloads)` | Computes the zero-based row index of the first maximum element in each column of a matrix and writes the result to `destination`. / 行列の各列で最大値を持つ最初の要素の行インデックスを計算し、`destination` に書き込みます。 |
| `static int[] ColumnIndexOfMin<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Returns the zero-based row index of the first minimum element in each column of a matrix. / 行列の各列について、最小値を持つ最初の要素の行インデックスを返します。 |
| `static Span<int> ColumnIndexOfMin<T>(ReadOnlyMatrixSpan<T> x, Span<int> destination)  (+2 overloads)` | Computes the zero-based row index of the first minimum element in each column of a matrix and writes the result to `destination`. / 行列の各列で最小値を持つ最初の要素の行インデックスを計算し、`destination` に書き込みます。 |
| `static Matrix<T> ColumnLogSoftmax<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Computes the log-softmax independently for each column of a matrix. / 行列の各列ごとに対数ソフトマックスを計算します。 |
| `static MatrixSpan<T> ColumnLogSoftmax<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Computes the log-softmax independently for each column of a matrix and writes the result to `destination`. / 行列の各列ごとに対数ソフトマックスを計算し、`destination` に書き込みます。 |
| `static Vector<T> ColumnLogSumExp<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Computes the log-sum-exp `ln Σ exp` of each column in a matrix. / 行列の各列の `ln Σ exp` を計算します。 |
| `static VectorSpan<T> ColumnLogSumExp<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Computes the log-sum-exp `ln Σ exp` of each column in a matrix and writes the result to `destination`. / 行列の各列の `ln Σ exp` を計算し、`destination` に書き込みます。 |
| `static Vector<T> ColumnMax<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Computes the maximum element of each column in a matrix. / 行列の各列の最大値を計算します。 |
| `static VectorSpan<T> ColumnMax<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Computes the maximum element of each column in a matrix and writes the result to `destination`. / 行列の各列の最大値を計算し、`destination` に書き込みます。 |
| `static Vector<T> ColumnMax<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN)  (+2 overloads)` | Computes the maximum element of each column in a matrix, optionally excluding `NaN` values. / 必要に応じて `NaN` を除外して、行列の各列の最大値を計算します。 |
| `static VectorSpan<T> ColumnMax<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN, VectorSpan<T> destination)  (+2 overloads)` | Computes the maximum element of each column in a matrix, optionally excluding `NaN` values, and writes the result to `destination`. / 必要に応じて `NaN` を除外して行列の各列の最大値を計算し、`destination` に書き込みます。 |
| `static Vector<T> ColumnMeans<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Computes the arithmetic mean of each column in a matrix. / 行列の各列の算術平均を計算します。 |
| `static VectorSpan<T> ColumnMeans<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Computes the arithmetic mean of each column in a matrix and writes the result to `destination`. / 行列の各列の算術平均を計算し、`destination` に書き込みます。 |
| `static Vector<T> ColumnMeans<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN)  (+2 overloads)` | Computes the arithmetic mean of each column in a matrix, optionally excluding `NaN` values. / 必要に応じて `NaN` を除外して、行列の各列の算術平均を計算します。 |
| `static VectorSpan<T> ColumnMeans<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN, VectorSpan<T> destination)  (+2 overloads)` | Computes the arithmetic mean of each column in a matrix, optionally excluding `NaN` values, and writes the result to `destination`. / 必要に応じて `NaN` を除外して行列の各列の算術平均を計算し、`destination` に書き込みます。 |
| `static Vector<T> ColumnMedians<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Computes the median of each column in a matrix. / 行列の各列の中央値を計算します。 |
| `static VectorSpan<T> ColumnMedians<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Computes the median of each column in a matrix and writes the result to `destination`. / 行列の各列の中央値を計算し、`destination` に書き込みます。 |
| `static Vector<T> ColumnMedians<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN)  (+2 overloads)` | Computes the median of each column in a matrix, optionally excluding `NaN` values. / 必要に応じて `NaN` を除外して、行列の各列の中央値を計算します。 |
| `static VectorSpan<T> ColumnMedians<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN, VectorSpan<T> destination)  (+2 overloads)` | Computes the median of each column in a matrix, optionally excluding `NaN` values, and writes the result to `destination`. / 必要に応じて `NaN` を除外して行列の各列の中央値を計算し、`destination` に書き込みます。 |
| `static Vector<T> ColumnMin<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Computes the minimum element of each column in a matrix. / 行列の各列の最小値を計算します。 |
| `static VectorSpan<T> ColumnMin<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Computes the minimum element of each column in a matrix and writes the result to `destination`. / 行列の各列の最小値を計算し、`destination` に書き込みます。 |
| `static Vector<T> ColumnMin<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN)  (+2 overloads)` | Computes the minimum element of each column in a matrix, optionally excluding `NaN` values. / 必要に応じて `NaN` を除外して、行列の各列の最小値を計算します。 |
| `static VectorSpan<T> ColumnMin<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN, VectorSpan<T> destination)  (+2 overloads)` | Computes the minimum element of each column in a matrix, optionally excluding `NaN` values, and writes the result to `destination`. / 必要に応じて `NaN` を除外して行列の各列の最小値を計算し、`destination` に書き込みます。 |
| `static Vector<T> ColumnNorms<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Computes the Euclidean (L2) norm of each column in a matrix. / 行列の各列のユークリッドノルムを計算します。 |
| `static VectorSpan<T> ColumnNorms<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Computes the Euclidean (L2) norm of each column in a matrix and writes the result to `destination`. / 行列の各列のユークリッドノルムを計算し、`destination` に書き込みます。 |
| `static Vector<T> ColumnQuantiles<T>(ReadOnlyMatrixSpan<T> x, double probability)  (+2 overloads)` | Computes the quantile of each column in a matrix. / 行列の各列の分位点を計算します。 |
| `static VectorSpan<T> ColumnQuantiles<T>(ReadOnlyMatrixSpan<T> x, double probability, VectorSpan<T> destination)  (+2 overloads)` | Computes the quantile of each column in a matrix and writes the result to `destination`. / 行列の各列の分位点を計算し、`destination` に書き込みます。 |
| `static Vector<T> ColumnQuantiles<T>(ReadOnlyMatrixSpan<T> x, double probability, bool ignoreNaN)  (+2 overloads)` | Computes the quantile of each column in a matrix, optionally excluding `NaN` values. / 必要に応じて `NaN` を除外して、行列の各列の分位点を計算します。 |
| `static VectorSpan<T> ColumnQuantiles<T>(ReadOnlyMatrixSpan<T> x, double probability, bool ignoreNaN, VectorSpan<T> destination)  (+2 overloads)` | Computes the quantile of each column in a matrix, optionally excluding `NaN` values, and writes the result to `destination`. / 必要に応じて `NaN` を除外して行列の各列の分位点を計算し、`destination` に書き込みます。 |
| `static Matrix<T> ColumnSoftmax<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Computes the softmax distribution independently for each column of a matrix. / 行列の各列ごとにソフトマックス分布を計算します。 |
| `static MatrixSpan<T> ColumnSoftmax<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Computes the softmax distribution independently for each column of a matrix and writes the result to `destination`. / 行列の各列ごとにソフトマックス分布を計算し、`destination` に書き込みます。 |
| `static Vector<T> ColumnStandardDeviations<T>(ReadOnlyMatrixSpan<T> x, int correction)  (+2 overloads)` | Computes the standard deviation of each column in a matrix. / 行列の各列の標準偏差を計算します。 |
| `static VectorSpan<T> ColumnStandardDeviations<T>(ReadOnlyMatrixSpan<T> x, int correction, VectorSpan<T> destination)  (+2 overloads)` | Computes the standard deviation of each column in a matrix and writes the result to `destination`. / 行列の各列の標準偏差を計算し、`destination` に書き込みます。 |
| `static Vector<T> ColumnSums<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Computes the sum of each column in a matrix. / 行列の各列の総和を計算します。 |
| `static VectorSpan<T> ColumnSums<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Computes the sum of each column in a matrix and writes the result to `destination`. / 行列の各列の総和を計算し、`destination` に書き込みます。 |
| `static Vector<T> ColumnSums<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN)  (+2 overloads)` | Computes the sum of each column in a matrix, optionally excluding `NaN` values. / 必要に応じて `NaN` を除外して、行列の各列の総和を計算します。 |
| `static VectorSpan<T> ColumnSums<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN, VectorSpan<T> destination)  (+2 overloads)` | Computes the sum of each column in a matrix, optionally excluding `NaN` values, and writes the result to `destination`. / 必要に応じて `NaN` を除外して行列の各列の総和を計算し、`destination` に書き込みます。 |
| `static Vector<T> ColumnSumsOfSquares<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Computes the sum of squares of each column in a matrix. / 行列の各列の平方和を計算します。 |
| `static VectorSpan<T> ColumnSumsOfSquares<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Computes the sum of squares of each column in a matrix and writes the result to `destination`. / 行列の各列の平方和を計算し、`destination` に書き込みます。 |
| `static Vector<T> ColumnVariances<T>(ReadOnlyMatrixSpan<T> x, int correction)  (+2 overloads)` | Computes the variance of each column in a matrix. / 行列の各列の分散を計算します。 |
| `static VectorSpan<T> ColumnVariances<T>(ReadOnlyMatrixSpan<T> x, int correction, VectorSpan<T> destination)  (+2 overloads)` | Computes the variance of each column in a matrix and writes the result to `destination`. / 行列の各列の分散を計算し、`destination` に書き込みます。 |
| `static Vector<T> Cross<T>(ReadOnlyVectorSpan<T> u, ReadOnlyVectorSpan<T> v)  (+2 overloads)` | Computes the cross product `u × v` of two three-dimensional vectors. / 2 つの 3 次元ベクトルの外積 `u × v` を計算します。 |
| `static VectorSpan<T> Cross<T>(ReadOnlyVectorSpan<T> u, ReadOnlyVectorSpan<T> v, VectorSpan<T> destination)  (+2 overloads)` | Writes the cross product `u × v` to `destination`. / 外積 `u × v` を `destination` に書き込みます。 |
| `static T Distance<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)  (+2 overloads)` | Computes the Euclidean distance `√Σᵢ (xᵢ − yᵢ)²` between two vectors without undue intermediate overflow or underflow. / 途中のオーバーフローやアンダーフローを回避して、2点間のユークリッド距離 `√Σᵢ (xᵢ − yᵢ)²` を計算します。 |
| `static Matrix<T> Distance<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y)  (+2 overloads)` | Applies `Distance` to every pair of columns of the input matrices (`result[i, j]` corresponds to column i and column j) and returns a new matrix. / `Distance` を入力行列の列の組ごとに適用し（`result[i, j]` は列 i と列 j の値）、新しい行列を返します。 |
| `static MatrixSpan<T> Distance<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Distance` to every pair of columns of the input matrices (`result[i, j]` corresponds to column i and column j), writes the result to `destination`, and returns it. / `Distance` を入力行列の列の組ごとに適用し（`result[i, j]` は列 i と列 j の値）、結果を `destination` に書き込んで返します。 |
| `static Tensor<T> Divide<T>(ReadOnlyTensorSpan<T> x, ReadOnlyTensorSpan<T> y)  (+2 overloads)` | Computes the elementwise quotient of two tensors. / 2つのテンソルの要素ごとの商を計算します。 |
| `static Tensor<T> Divide<T>(ReadOnlyTensorSpan<T> x, T scalar)  (+2 overloads)` | Divides each element of a tensor by a scalar. / テンソルの各要素をスカラーで割ります。 |
| `static T Divide<T>(T x, T y)` | Computes the quotient `x / y`. / 商 `x / y` を計算します。 |
| `static Vector<T> Divide<T>(ReadOnlyVectorSpan<T> x, T y)  (+2 overloads)` | Applies `Divide` elementwise to the input vector and returns a new vector. / `Divide` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Divide<T>(ReadOnlyMatrixSpan<T> x, T y)  (+2 overloads)` | Applies `Divide` elementwise to the input matrix and returns a new matrix. / `Divide` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static Vector<T> Divide<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)  (+2 overloads)` | Applies `Divide` elementwise to the input vector and returns a new vector. / `Divide` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Divide<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y)  (+2 overloads)` | Applies `Divide` elementwise to the input matrix and returns a new matrix. / `Divide` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static TensorSpan<T> Divide<T>(ReadOnlyTensorSpan<T> x, ReadOnlyTensorSpan<T> y, TensorSpan<T> destination)  (+2 overloads)` | Divides two tensors elementwise and writes the result to `destination`. / 2つのテンソルを要素ごとに除算し、結果を `destination` に書き込みます。 |
| `static TensorSpan<T> Divide<T>(ReadOnlyTensorSpan<T> x, T scalar, TensorSpan<T> destination)  (+2 overloads)` | Divides each element of a tensor by a scalar and writes the result to `destination`. / テンソルの各要素をスカラーで割り、結果を `destination` に書き込みます。 |
| `static VectorSpan<T> Divide<T>(ReadOnlyVectorSpan<T> x, T y, VectorSpan<T> destination)  (+2 overloads)` | Applies `Divide` elementwise to the input vector, writes the result to `destination`, and returns it. / `Divide` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Divide<T>(ReadOnlyMatrixSpan<T> x, T y, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Divide` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Divide` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static VectorSpan<T> Divide<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, VectorSpan<T> destination)  (+2 overloads)` | Applies `Divide` elementwise to the input vector, writes the result to `destination`, and returns it. / `Divide` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Divide<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Divide` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Divide` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static T Dot<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)  (+2 overloads)` | Computes the dot product `Σ x[i]·y[i]` of two vectors. / 2つのベクトルの内積 `Σ x[i]·y[i]` を計算します。 |
| `static T Elu<T>(T x, T alpha)` | Computes the exponential linear unit (`x` if `x > 0`, otherwise `alpha · (eˣ - 1)`). / ELU（`x > 0` のとき `x`、それ以外は `alpha · (eˣ - 1)`）を計算します。 |
| `static Vector<T> Elu<T>(ReadOnlyVectorSpan<T> x, T alpha)  (+2 overloads)` | Applies `Elu` elementwise to the input vector and returns a new vector. / `Elu` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Elu<T>(ReadOnlyMatrixSpan<T> x, T alpha)  (+2 overloads)` | Applies `Elu` elementwise to the input matrix and returns a new matrix. / `Elu` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static VectorSpan<T> Elu<T>(ReadOnlyVectorSpan<T> x, T alpha, VectorSpan<T> destination)  (+2 overloads)` | Applies `Elu` elementwise to the input vector, writes the result to `destination`, and returns it. / `Elu` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Elu<T>(ReadOnlyMatrixSpan<T> x, T alpha, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Elu` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Elu` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static Vector<T> Exp<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Computes the elementwise exponential `exp(x)` of a vector. / ベクトルの要素ごとの指数関数 `exp(x)` を計算します。 |
| `static Matrix<T> Exp<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Computes the elementwise exponential `exp(x)` of a matrix. / 行列の要素ごとの指数関数 `exp(x)` を計算します。 |
| `static Vector<T> Exp<T>(ReadOnlyVectorSpan<T> x, bool approximate)  (+2 overloads)` | Computes the elementwise exponential `exp(x)` of a vector, optionally using a fast approximation for `Single`. / ベクトルの要素ごとの指数関数 `exp(x)` を計算します（`Single` では高速な近似式を選択可能）。 |
| `static Matrix<T> Exp<T>(ReadOnlyMatrixSpan<T> x, bool approximate)  (+2 overloads)` | Computes the elementwise exponential `exp(x)` of a matrix, optionally using a fast approximation for `Single`. / 行列の要素ごとの指数関数 `exp(x)` を計算します（`Single` では高速な近似式を選択可能）。 |
| `static VectorSpan<T> Exp<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Computes the elementwise exponential `exp(x)` of a vector and writes the result to `destination`. / ベクトルの要素ごとの指数関数 `exp(x)` を計算し、`destination` に書き込みます。 |
| `static MatrixSpan<T> Exp<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Computes the elementwise exponential `exp(x)` of a matrix and writes the result to `destination`. / 行列の要素ごとの指数関数 `exp(x)` を計算し、`destination` に書き込みます。 |
| `static VectorSpan<T> Exp<T>(ReadOnlyVectorSpan<T> x, bool approximate, VectorSpan<T> destination)  (+2 overloads)` | Computes the elementwise exponential `exp(x)` of a vector and writes the result to `destination`. / ベクトルの要素ごとの指数関数 `exp(x)` を計算し、`destination` に書き込みます。 |
| `static MatrixSpan<T> Exp<T>(ReadOnlyMatrixSpan<T> x, bool approximate, MatrixSpan<T> destination)  (+2 overloads)` | Computes the elementwise exponential `exp(x)` of a matrix and writes the result to `destination`. / 行列の要素ごとの指数関数 `exp(x)` を計算し、`destination` に書き込みます。 |
| `static T Expm1<T>(T x)` | Computes `eˣ - 1` accurately for values of `x` near zero. / 0 付近でも桁落ちしにくい方法で `eˣ - 1` を計算します。 |
| `static Vector<T> Expm1<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Applies `Expm1` elementwise to the input vector and returns a new vector. / `Expm1` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Expm1<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Applies `Expm1` elementwise to the input matrix and returns a new matrix. / `Expm1` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static VectorSpan<T> Expm1<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Applies `Expm1` elementwise to the input vector, writes the result to `destination`, and returns it. / `Expm1` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Expm1<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Expm1` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Expm1` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static T Gelu<T>(T x)` | Computes the exact Gaussian error linear unit `x · Φ(x) = (x / 2) · erfc(-x / √2)`. / 定義どおりの GELU `x · Φ(x) = (x / 2) · erfc(-x / √2)` を計算します。 |
| `static Vector<T> Gelu<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Applies `Gelu` elementwise to the input vector and returns a new vector. / `Gelu` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Gelu<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Applies `Gelu` elementwise to the input matrix and returns a new matrix. / `Gelu` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static T Gelu<T>(T x, bool approximate)` | Computes the GELU function, optionally using the `tanh`-based approximation. / GELU 関数を計算します（`tanh` による近似式を選択可能）。 |
| `static Vector<T> Gelu<T>(ReadOnlyVectorSpan<T> x, bool approximate)  (+2 overloads)` | Applies the GELU function elementwise to a vector, optionally using the `tanh`-based approximation. / ベクトルの各要素に GELU 関数を適用します（`tanh` による近似式を選択可能）。 |
| `static Matrix<T> Gelu<T>(ReadOnlyMatrixSpan<T> x, bool approximate)  (+2 overloads)` | Applies the GELU function elementwise to a matrix, optionally using the `tanh`-based approximation. / 行列の各要素に GELU 関数を適用します（`tanh` による近似式を選択可能）。 |
| `static VectorSpan<T> Gelu<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Applies `Gelu` elementwise to the input vector, writes the result to `destination`, and returns it. / `Gelu` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Gelu<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Gelu` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Gelu` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static VectorSpan<T> Gelu<T>(ReadOnlyVectorSpan<T> x, bool approximate, VectorSpan<T> destination)  (+2 overloads)` | Applies the GELU function elementwise to a vector and writes the result to `destination`. / ベクトルの各要素に GELU 関数を適用し、`destination` に書き込みます。 |
| `static MatrixSpan<T> Gelu<T>(ReadOnlyMatrixSpan<T> x, bool approximate, MatrixSpan<T> destination)  (+2 overloads)` | Applies the GELU function elementwise to a matrix and writes the result to `destination`. / 行列の各要素に GELU 関数を適用し、`destination` に書き込みます。 |
| `static VectorSpan<Half> Generate<TOperator>(VectorSpan<Half> destination)  (+3 overloads)` | Populates every element of `destination` using a stateless index operator (`destination[i] = TOperator(i)`). / 状態を持たないインデックス演算子でベクトルの全要素を設定します（`destination[i] = TOperator(i)`）。 |
| `static MatrixSpan<Half> Generate<TOperator>(MatrixSpan<Half> destination)  (+3 overloads)` | Populates every element of `destination` using a stateless index operator (`destination[i, j] = TOperator(i, j)`). / 状態を持たないインデックス演算子で行列の全要素を設定します（`destination[i, j] = TOperator(i, j)`）。 |
| `static TensorSpan<Half> Generate<TOperator>(TensorSpan<Half> destination)  (+3 overloads)` | Populates every element of `destination` using a stateless tensor index operator. / 状態を持たないテンソルインデックス演算子でテンソルの全要素を設定します。 |
| `static MatrixSpan<decimal> Generate<TOperator>(MatrixSpan<decimal> destination)` | Populates every element of `destination` using a stateless index operator (`destination[i, j] = TOperator(i, j)`). / 状態を持たないインデックス演算子で行列の全要素を設定します（`destination[i, j] = TOperator(i, j)`）。 |
| `static MatrixSpan<Complex> Generate<TOperator>(MatrixSpan<Complex> destination)` | Populates every element of `destination` using a stateless index operator (`destination[i, j] = TOperator(i, j)`). / 状態を持たないインデックス演算子で行列の全要素を設定します（`destination[i, j] = TOperator(i, j)`）。 |
| `static TensorSpan<decimal> Generate<TOperator>(TensorSpan<decimal> destination)` | Populates every element of `destination` using a stateless tensor index operator. / 状態を持たないテンソルインデックス演算子でテンソルの全要素を設定します。 |
| `static TensorSpan<Complex> Generate<TOperator>(TensorSpan<Complex> destination)` | Populates every element of `destination` using a stateless tensor index operator. / 状態を持たないテンソルインデックス演算子でテンソルの全要素を設定します。 |
| `static VectorSpan<decimal> Generate<TOperator>(VectorSpan<decimal> destination)` | Populates every element of `destination` using a stateless index operator (`destination[i] = TOperator(i)`). / 状態を持たないインデックス演算子でベクトルの全要素を設定します（`destination[i] = TOperator(i)`）。 |
| `static VectorSpan<Complex> Generate<TOperator>(VectorSpan<Complex> destination)` | Populates every element of `destination` using a stateless index operator (`destination[i] = TOperator(i)`). / 状態を持たないインデックス演算子でベクトルの全要素を設定します（`destination[i] = TOperator(i)`）。 |
| `static VectorSpan<T> Generate<T, TOperator>(TOperator op, VectorSpan<T> destination)  (+1 overloads)` | Populates every element of `destination` using a stateful index operator (`destination[i] = op(i)`). / 状態を持つインデックス演算子でベクトルの全要素を設定します（`destination[i] = op(i)`）。 |
| `static VectorSpan<T> Generate<T>(Func<int, T> op, VectorSpan<T> destination)  (+1 overloads)` | Populates every element of `destination` using a delegate (`destination[i] = op(i)`). / デリゲートを用いてベクトルの全要素を設定します（`destination[i] = op(i)`）。 |
| `static MatrixSpan<T> Generate<T, TOperator>(TOperator op, MatrixSpan<T> destination)  (+1 overloads)` | Populates every element of `destination` using a stateful index operator (`destination[i, j] = op(i, j)`). / 状態を持つインデックス演算子で行列の全要素を設定します（`destination[i, j] = op(i, j)`）。 |
| `static MatrixSpan<T> Generate<T>(Func<int, int, T> op, MatrixSpan<T> destination)  (+1 overloads)` | Populates every element of `destination` using a delegate (`destination[i, j] = op(i, j)`). / デリゲートを用いて行列の全要素を設定します（`destination[i, j] = op(i, j)`）。 |
| `static TensorSpan<T> Generate<T, TOperator>(TOperator op, TensorSpan<T> destination)  (+1 overloads)` | Populates every element of `destination` using a stateful tensor index operator. / 状態を持つテンソルインデックス演算子でテンソルの全要素を設定します。 |
| `static TensorSpan<T> Generate<T>(Func<ReadOnlySpan<int>, T> op, TensorSpan<T> destination)  (+1 overloads)` | Populates every element of `destination` using a delegate over multi-dimensional indices. / 全軸のインデックスを受け取るデリゲートを用いてテンソルの全要素を設定します。 |
| `static MatrixSpan<Half> GenerateBand<TOperator>(int lowerBandwidth, int upperBandwidth, MatrixSpan<Half> destination)  (+3 overloads)` | Populates a banded matrix using a stateless index operator for entries with `−lowerBandwidth ≤ j − i ≤ upperBandwidth` and zeros elsewhere. / 状態を持たない演算子で `−lowerBandwidth ≤ j − i ≤ upperBandwidth` の帯内要素を設定し、帯外を 0 で埋めます。 |
| `static MatrixSpan<decimal> GenerateBand<TOperator>(int lowerBandwidth, int upperBandwidth, MatrixSpan<decimal> destination)` | Populates a banded matrix using a stateless index operator for entries with `−lowerBandwidth ≤ j − i ≤ upperBandwidth` and zeros elsewhere. / 状態を持たない演算子で `−lowerBandwidth ≤ j − i ≤ upperBandwidth` の帯内要素を設定し、帯外を 0 で埋めます。 |
| `static MatrixSpan<Complex> GenerateBand<TOperator>(int lowerBandwidth, int upperBandwidth, MatrixSpan<Complex> destination)` | Populates a banded matrix using a stateless index operator for entries with `−lowerBandwidth ≤ j − i ≤ upperBandwidth` and zeros elsewhere. / 状態を持たない演算子で `−lowerBandwidth ≤ j − i ≤ upperBandwidth` の帯内要素を設定し、帯外を 0 で埋めます。 |
| `static MatrixSpan<T> GenerateBand<T, TOperator>(int lowerBandwidth, int upperBandwidth, TOperator op, MatrixSpan<T> destination)  (+1 overloads)` | Populates a banded matrix using a stateful index operator for entries with `−lowerBandwidth ≤ j − i ≤ upperBandwidth` and zeros elsewhere. / 状態を持つ演算子で `−lowerBandwidth ≤ j − i ≤ upperBandwidth` の帯内要素を設定し、帯外を 0 で埋めます。 |
| `static MatrixSpan<T> GenerateBand<T>(int lowerBandwidth, int upperBandwidth, Func<int, int, T> op, MatrixSpan<T> destination)  (+1 overloads)` | Populates a banded matrix using a delegate for entries within the band and zeros elsewhere. / デリゲートを用いて帯内の要素を設定し、帯外を 0 で埋めます。 |
| `static MatrixSpan<Half> GenerateSymmetric<TOperator>(MatrixSpan<Half> destination)  (+3 overloads)` | Populates a symmetric matrix by evaluating a stateless index operator on the lower triangle (`i ≥ j`) and mirroring to the upper triangle. / 状態を持たない演算子で下三角（`i ≥ j`）のみを評価し、上三角へ対称に転記して対称行列を生成します。 |
| `static MatrixSpan<decimal> GenerateSymmetric<TOperator>(MatrixSpan<decimal> destination)` | Populates a symmetric matrix by evaluating a stateless index operator on the lower triangle (`i ≥ j`) and mirroring to the upper triangle. / 状態を持たない演算子で下三角（`i ≥ j`）のみを評価し、上三角へ対称に転記して対称行列を生成します。 |
| `static MatrixSpan<Complex> GenerateSymmetric<TOperator>(MatrixSpan<Complex> destination)` | Populates a symmetric matrix by evaluating a stateless index operator on the lower triangle (`i ≥ j`) and mirroring to the upper triangle. / 状態を持たない演算子で下三角（`i ≥ j`）のみを評価し、上三角へ対称に転記して対称行列を生成します。 |
| `static MatrixSpan<T> GenerateSymmetric<T, TOperator>(TOperator op, MatrixSpan<T> destination)  (+1 overloads)` | Populates a symmetric matrix by evaluating a stateful index operator on the lower triangle (`i ≥ j`) and mirroring to the upper triangle. / 状態を持つ演算子で下三角（`i ≥ j`）のみを評価し、上三角へ対称に転記して対称行列を生成します。 |
| `static MatrixSpan<T> GenerateSymmetric<T>(Func<int, int, T> op, MatrixSpan<T> destination)  (+1 overloads)` | Populates a symmetric matrix by evaluating a delegate on the lower triangle (`i ≥ j`) and mirroring to the upper triangle. / デリゲートを用いて下三角（`i ≥ j`）のみを評価し、上三角へ対称に転記して対称行列を生成します。 |
| `static MatrixSpan<Half> GenerateSymmetric<TOperator>(int bandwidth, MatrixSpan<Half> destination)  (+3 overloads)` | Populates a symmetric banded matrix by evaluating a stateless index operator for `0 ≤ i − j ≤ bandwidth`, mirroring to the upper triangle, and zeroing entries outside the band. / 状態を持たない演算子で `0 ≤ i − j ≤ bandwidth` の下三角帯内のみを評価し、上三角へ対称に転記して帯外を 0 で埋めます。 |
| `static MatrixSpan<decimal> GenerateSymmetric<TOperator>(int bandwidth, MatrixSpan<decimal> destination)` | Populates a symmetric banded matrix by evaluating a stateless index operator for `0 ≤ i − j ≤ bandwidth`, mirroring to the upper triangle, and zeroing entries outside the band. / 状態を持たない演算子で `0 ≤ i − j ≤ bandwidth` の下三角帯内のみを評価し、上三角へ対称に転記して帯外を 0 で埋めます。 |
| `static MatrixSpan<Complex> GenerateSymmetric<TOperator>(int bandwidth, MatrixSpan<Complex> destination)` | Populates a symmetric banded matrix by evaluating a stateless index operator for `0 ≤ i − j ≤ bandwidth`, mirroring to the upper triangle, and zeroing entries outside the band. / 状態を持たない演算子で `0 ≤ i − j ≤ bandwidth` の下三角帯内のみを評価し、上三角へ対称に転記して帯外を 0 で埋めます。 |
| `static MatrixSpan<T> GenerateSymmetric<T, TOperator>(int bandwidth, TOperator op, MatrixSpan<T> destination)  (+1 overloads)` | Populates a symmetric banded matrix by evaluating a stateful index operator for `0 ≤ i − j ≤ bandwidth`, mirroring to the upper triangle, and zeroing entries outside the band. / 状態を持つ演算子で `0 ≤ i − j ≤ bandwidth` の下三角帯内のみを評価し、上三角へ対称に転記して帯外を 0 で埋めます。 |
| `static MatrixSpan<T> GenerateSymmetric<T>(int bandwidth, Func<int, int, T> op, MatrixSpan<T> destination)  (+1 overloads)` | Populates a symmetric banded matrix using a delegate for `0 ≤ i − j ≤ bandwidth`, mirroring to the upper triangle, and zeroing entries outside the band. / デリゲートを用いて `0 ≤ i − j ≤ bandwidth` の下三角帯内のみを評価し、上三角へ対称に転記して帯外を 0 で埋めます。 |
| `static Tensor<T> Hadamard<T>(ReadOnlyTensorSpan<T> x, ReadOnlyTensorSpan<T> y)  (+2 overloads)` | Computes the elementwise (Hadamard) product `X[k] ∘ Y[k]` of two tensors. / 2つのテンソルの要素ごとの積（アダマール積）`X[k] ∘ Y[k]` を計算します。 |
| `static Vector<T> Hadamard<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)  (+2 overloads)` | Computes the elementwise (Hadamard) product `x ∘ y` of two vectors. / 2つのベクトルの要素ごとの積（アダマール積）`x ∘ y` を計算します。 |
| `static Matrix<T> Hadamard<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y)  (+2 overloads)` | Computes the elementwise (Hadamard) product `x ∘ y` of two matrices. / 2つの行列の要素ごとの積（アダマール積）`x ∘ y` を計算します（行列積ではありません）。 |
| `static TensorSpan<T> Hadamard<T>(ReadOnlyTensorSpan<T> x, ReadOnlyTensorSpan<T> y, TensorSpan<T> destination)  (+2 overloads)` | Computes the elementwise (Hadamard) product `X[k] ∘ Y[k]` of two tensors and writes the result to `destination`. / 2つのテンソルの要素ごとの積（アダマール積）`X[k] ∘ Y[k]` を計算し、`destination` に書き込みます。 |
| `static VectorSpan<T> Hadamard<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, VectorSpan<T> destination)  (+2 overloads)` | Computes the elementwise (Hadamard) product `x ∘ y` of two vectors and writes the result to `destination`. / 2つのベクトルの要素ごとの積（アダマール積）`x ∘ y` を計算し、`destination` に書き込みます。 |
| `static MatrixSpan<T> Hadamard<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, MatrixSpan<T> destination)  (+2 overloads)` | Computes the elementwise (Hadamard) product `x ∘ y` of two matrices and writes the result to `destination`. / 2つの行列の要素ごとの積（アダマール積）`x ∘ y` を計算し、`destination` に書き込みます。 |
| `static T Hypot<T>(T x, T y)` | Computes `√(x² + y²)` without undue intermediate overflow or underflow. / 途中計算でのオーバーフローやアンダーフローを回避して `√(x² + y²)` を計算します。 |
| `static Vector<T> Hypot<T>(ReadOnlyVectorSpan<T> x, T y)  (+2 overloads)` | Applies `Hypot` elementwise to the input vector and returns a new vector. / `Hypot` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Hypot<T>(ReadOnlyMatrixSpan<T> x, T y)  (+2 overloads)` | Applies `Hypot` elementwise to the input matrix and returns a new matrix. / `Hypot` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static Vector<T> Hypot<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)  (+2 overloads)` | Applies `Hypot` elementwise to the input vector and returns a new vector. / `Hypot` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Hypot<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y)  (+2 overloads)` | Applies `Hypot` elementwise to the input matrix and returns a new matrix. / `Hypot` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static VectorSpan<T> Hypot<T>(ReadOnlyVectorSpan<T> x, T y, VectorSpan<T> destination)  (+2 overloads)` | Applies `Hypot` elementwise to the input vector, writes the result to `destination`, and returns it. / `Hypot` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Hypot<T>(ReadOnlyMatrixSpan<T> x, T y, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Hypot` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Hypot` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static VectorSpan<T> Hypot<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, VectorSpan<T> destination)  (+2 overloads)` | Applies `Hypot` elementwise to the input vector, writes the result to `destination`, and returns it. / `Hypot` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Hypot<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Hypot` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Hypot` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static Vector<double> Imaginary(ReadOnlyVectorSpan<Complex> x)` | Extracts the imaginary part of each element in a complex vector. / 複素数ベクトルの各要素の虚部を抽出します。 |
| `static Matrix<double> Imaginary(ReadOnlyMatrixSpan<Complex> x)` | Extracts the imaginary part of each element in a complex matrix. / 複素数行列の各要素の虚部を抽出します。 |
| `static VectorSpan<double> Imaginary(ReadOnlyVectorSpan<Complex> x, VectorSpan<double> destination)` | Extracts the imaginary part of each element in a complex vector and writes the result to `destination`. / 複素数ベクトルの各要素の虚部を抽出し、`destination` に書き込みます。 |
| `static MatrixSpan<double> Imaginary(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<double> destination)` | Extracts the imaginary part of each element in a complex matrix and writes the result to `destination`. / 複素数行列の各要素の虚部を抽出し、`destination` に書き込みます。 |
| `static int IndexOfMax<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Returns the zero-based index of the first element with the maximum value in a vector. / ベクトル内で最大値を持つ最初の要素の 0 始まりのインデックスを返します。 |
| `static (int, int) IndexOfMax<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Returns the zero-based row and column indices of the first element with the maximum value in a matrix. / 行列全体で最大値を持つ最初の要素の 0 始まりの行番号と列番号を返します。 |
| `static int IndexOfMin<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Returns the zero-based index of the first element with the minimum value in a vector. / ベクトル内で最小値を持つ最初の要素の 0 始まりのインデックスを返します。 |
| `static (int, int) IndexOfMin<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Returns the zero-based row and column indices of the first element with the minimum value in a matrix. / 行列全体で最小値を持つ最初の要素の 0 始まりの行番号と列番号を返します。 |
| `static Matrix<double> Inverse(ReadOnlyMatrixSpan<double> a)` | Computes the matrix inverse `A⁻¹` using LU decomposition with partial pivoting. / 部分ピボット付き LU 分解により逆行列 `A⁻¹` を計算します。 |
| `static MatrixSpan<double> Inverse(ReadOnlyMatrixSpan<double> a, MatrixSpan<double> destination)` | Computes the matrix inverse `A⁻¹` using LU decomposition with partial pivoting and writes it to `destination`. / 部分ピボット付き LU 分解により逆行列 `A⁻¹` を計算し、`destination` に書き込みます。 |
| `static T LeakyRelu<T>(T x, T slope)` | Computes the leaky rectified linear unit (`x` if `x > 0`, otherwise `slope · x`). / Leaky ReLU（`x > 0` のとき `x`、それ以外は `slope · x`）を計算します。 |
| `static Vector<T> LeakyRelu<T>(ReadOnlyVectorSpan<T> x, T slope)  (+2 overloads)` | Applies `LeakyRelu` elementwise to the input vector and returns a new vector. / `LeakyRelu` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> LeakyRelu<T>(ReadOnlyMatrixSpan<T> x, T slope)  (+2 overloads)` | Applies `LeakyRelu` elementwise to the input matrix and returns a new matrix. / `LeakyRelu` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static VectorSpan<T> LeakyRelu<T>(ReadOnlyVectorSpan<T> x, T slope, VectorSpan<T> destination)  (+2 overloads)` | Applies `LeakyRelu` elementwise to the input vector, writes the result to `destination`, and returns it. / `LeakyRelu` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> LeakyRelu<T>(ReadOnlyMatrixSpan<T> x, T slope, MatrixSpan<T> destination)  (+2 overloads)` | Applies `LeakyRelu` elementwise to the input matrix, writes the result to `destination`, and returns it. / `LeakyRelu` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static T Log<T>(T x)` | Computes the natural logarithm `ln(x)`. / 自然対数 `ln(x)` を計算します。 |
| `static Vector<T> Log<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Applies `Log` elementwise to the input vector and returns a new vector. / `Log` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Log<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Applies `Log` elementwise to the input matrix and returns a new matrix. / `Log` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static VectorSpan<T> Log<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Applies `Log` elementwise to the input vector, writes the result to `destination`, and returns it. / `Log` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Log<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Log` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Log` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static T Log1mExp<T>(T x)` | Computes `ln(1 - eˣ)` for `x ≤ 0` accurately without cancellation. / `x ≤ 0` に対して、桁落ちを回避して `ln(1 - eˣ)` を計算します。 |
| `static Vector<T> Log1mExp<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Applies `Log1mExp` elementwise to the input vector and returns a new vector. / `Log1mExp` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Log1mExp<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Applies `Log1mExp` elementwise to the input matrix and returns a new matrix. / `Log1mExp` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static VectorSpan<T> Log1mExp<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Applies `Log1mExp` elementwise to the input vector, writes the result to `destination`, and returns it. / `Log1mExp` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Log1mExp<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Log1mExp` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Log1mExp` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static T Log1p<T>(T x)` | Computes `ln(1 + x)` accurately for values of `x` near zero. / 0 付近でも桁落ちしにくい方法で `ln(1 + x)` を計算します。 |
| `static Vector<T> Log1p<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Applies `Log1p` elementwise to the input vector and returns a new vector. / `Log1p` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Log1p<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Applies `Log1p` elementwise to the input matrix and returns a new matrix. / `Log1p` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static VectorSpan<T> Log1p<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Applies `Log1p` elementwise to the input vector, writes the result to `destination`, and returns it. / `Log1p` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Log1p<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Log1p` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Log1p` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static T Log1pmx<T>(T x)` | Computes `ln(1 + x) - x` for `x ≥ -1` accurately near zero. / `x ≥ -1` に対して、0 付近での桁落ちを回避して `ln(1 + x) - x` を計算します。 |
| `static Vector<T> Log1pmx<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Applies `Log1pmx` elementwise to the input vector and returns a new vector. / `Log1pmx` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Log1pmx<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Applies `Log1pmx` elementwise to the input matrix and returns a new matrix. / `Log1pmx` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static VectorSpan<T> Log1pmx<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Applies `Log1pmx` elementwise to the input vector, writes the result to `destination`, and returns it. / `Log1pmx` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Log1pmx<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Log1pmx` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Log1pmx` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static T LogAddExp<T>(T x, T y)` | Computes `ln(eˣ + eʸ)` without intermediate overflow or underflow. / 途中計算でのオーバーフローやアンダーフローを回避して `ln(eˣ + eʸ)` を計算します。 |
| `static Vector<T> LogAddExp<T>(ReadOnlyVectorSpan<T> x, T y)  (+2 overloads)` | Applies `LogAddExp` elementwise to the input vector and returns a new vector. / `LogAddExp` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> LogAddExp<T>(ReadOnlyMatrixSpan<T> x, T y)  (+2 overloads)` | Applies `LogAddExp` elementwise to the input matrix and returns a new matrix. / `LogAddExp` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static Vector<T> LogAddExp<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)  (+2 overloads)` | Applies `LogAddExp` elementwise to the input vector and returns a new vector. / `LogAddExp` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> LogAddExp<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y)  (+2 overloads)` | Applies `LogAddExp` elementwise to the input matrix and returns a new matrix. / `LogAddExp` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static VectorSpan<T> LogAddExp<T>(ReadOnlyVectorSpan<T> x, T y, VectorSpan<T> destination)  (+2 overloads)` | Applies `LogAddExp` elementwise to the input vector, writes the result to `destination`, and returns it. / `LogAddExp` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> LogAddExp<T>(ReadOnlyMatrixSpan<T> x, T y, MatrixSpan<T> destination)  (+2 overloads)` | Applies `LogAddExp` elementwise to the input matrix, writes the result to `destination`, and returns it. / `LogAddExp` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static VectorSpan<T> LogAddExp<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, VectorSpan<T> destination)  (+2 overloads)` | Applies `LogAddExp` elementwise to the input vector, writes the result to `destination`, and returns it. / `LogAddExp` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> LogAddExp<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, MatrixSpan<T> destination)  (+2 overloads)` | Applies `LogAddExp` elementwise to the input matrix, writes the result to `destination`, and returns it. / `LogAddExp` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static T LogLogistic<T>(T x)` | Computes the log-logistic function `ln σ(x) = -ln(1 + e⁻ˣ)` without intermediate overflow. / 途中計算でのオーバーフローや桁落ちを回避して、ロジスティック関数の自然対数 `ln σ(x) = -ln(1 + e⁻ˣ)` を計算します。 |
| `static Vector<T> LogLogistic<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Applies `LogLogistic` elementwise to the input vector and returns a new vector. / `LogLogistic` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> LogLogistic<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Applies `LogLogistic` elementwise to the input matrix and returns a new matrix. / `LogLogistic` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static VectorSpan<T> LogLogistic<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Applies `LogLogistic` elementwise to the input vector, writes the result to `destination`, and returns it. / `LogLogistic` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> LogLogistic<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Applies `LogLogistic` elementwise to the input matrix, writes the result to `destination`, and returns it. / `LogLogistic` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static Vector<T> LogSoftmax<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Computes the log-softmax `x - LogSumExp(x)` over a vector. / ベクトル全体に対する対数ソフトマックス `x - LogSumExp(x)` を計算します。 |
| `static VectorSpan<T> LogSoftmax<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Computes the log-softmax `x - LogSumExp(x)` over a vector and writes the result to `destination`. / ベクトル全体に対する対数ソフトマックスを計算し、`destination` に書き込みます。 |
| `static T LogSubExp<T>(T x, T y)` | Computes `ln(eˣ - eʸ)` for `x ≥ y` without intermediate overflow or cancellation. / `x ≥ y` に対して、オーバーフローや桁落ちを回避して `ln(eˣ - eʸ)` を計算します。 |
| `static Vector<T> LogSubExp<T>(ReadOnlyVectorSpan<T> x, T y)  (+2 overloads)` | Applies `LogSubExp` elementwise to the input vector and returns a new vector. / `LogSubExp` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> LogSubExp<T>(ReadOnlyMatrixSpan<T> x, T y)  (+2 overloads)` | Applies `LogSubExp` elementwise to the input matrix and returns a new matrix. / `LogSubExp` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static Vector<T> LogSubExp<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)  (+2 overloads)` | Applies `LogSubExp` elementwise to the input vector and returns a new vector. / `LogSubExp` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> LogSubExp<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y)  (+2 overloads)` | Applies `LogSubExp` elementwise to the input matrix and returns a new matrix. / `LogSubExp` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static VectorSpan<T> LogSubExp<T>(ReadOnlyVectorSpan<T> x, T y, VectorSpan<T> destination)  (+2 overloads)` | Applies `LogSubExp` elementwise to the input vector, writes the result to `destination`, and returns it. / `LogSubExp` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> LogSubExp<T>(ReadOnlyMatrixSpan<T> x, T y, MatrixSpan<T> destination)  (+2 overloads)` | Applies `LogSubExp` elementwise to the input matrix, writes the result to `destination`, and returns it. / `LogSubExp` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static VectorSpan<T> LogSubExp<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, VectorSpan<T> destination)  (+2 overloads)` | Applies `LogSubExp` elementwise to the input vector, writes the result to `destination`, and returns it. / `LogSubExp` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> LogSubExp<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, MatrixSpan<T> destination)  (+2 overloads)` | Applies `LogSubExp` elementwise to the input matrix, writes the result to `destination`, and returns it. / `LogSubExp` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static T LogSumExp<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Computes the log-sum-exp `ln Σ exp(x[i])` of the elements in a vector without intermediate overflow. / 途中のオーバーフローを回避して、ベクトル要素の `ln Σ exp(x[i])` を計算します。 |
| `static T Logistic<T>(T x)` | Computes the logistic sigmoid function `σ(x) = 1 / (1 + e⁻ˣ)`. / ロジスティック（シグモイド）関数 `σ(x) = 1 / (1 + e⁻ˣ)` を計算します。 |
| `static Vector<T> Logistic<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Applies `Logistic` elementwise to the input vector and returns a new vector. / `Logistic` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Logistic<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Applies `Logistic` elementwise to the input matrix and returns a new matrix. / `Logistic` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static T Logistic<T>(T x, bool approximate)` | Computes the logistic sigmoid function `σ(x) = 1 / (1 + e⁻ˣ)`, optionally using a fast approximation for `Single`. / ロジスティック関数 `σ(x) = 1 / (1 + e⁻ˣ)` を計算します（`Single` では高速な近似式を選択可能）。 |
| `static Vector<T> Logistic<T>(ReadOnlyVectorSpan<T> x, bool approximate)  (+2 overloads)` | Applies the logistic sigmoid function elementwise to a vector, optionally using a fast approximation for `Single`. / ベクトルの各要素にロジスティック関数を適用します（`Single` では高速な近似式を選択可能）。 |
| `static Matrix<T> Logistic<T>(ReadOnlyMatrixSpan<T> x, bool approximate)  (+2 overloads)` | Applies the logistic sigmoid function elementwise to a matrix, optionally using a fast approximation for `Single`. / 行列の各要素にロジスティック関数を適用します（`Single` では高速な近似式を選択可能）。 |
| `static VectorSpan<T> Logistic<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Applies `Logistic` elementwise to the input vector, writes the result to `destination`, and returns it. / `Logistic` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Logistic<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Logistic` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Logistic` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static VectorSpan<T> Logistic<T>(ReadOnlyVectorSpan<T> x, bool approximate, VectorSpan<T> destination)  (+2 overloads)` | Applies the logistic sigmoid function elementwise to a vector and writes the result to `destination`. / ベクトルの各要素にロジスティック関数を適用し、`destination` に書き込みます。 |
| `static MatrixSpan<T> Logistic<T>(ReadOnlyMatrixSpan<T> x, bool approximate, MatrixSpan<T> destination)  (+2 overloads)` | Applies the logistic sigmoid function elementwise to a matrix and writes the result to `destination`. / 行列の各要素にロジスティック関数を適用し、`destination` に書き込みます。 |
| `static T Logit<T>(T p)` | Computes the logit function `ln(p / (1 - p))`, the inverse of `Logistic`. / ロジスティック関数の逆関数であるロジット関数 `ln(p / (1 - p))` を計算します。 |
| `static Vector<T> Logit<T>(ReadOnlyVectorSpan<T> p)  (+2 overloads)` | Applies `Logit` elementwise to the input vector and returns a new vector. / `Logit` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Logit<T>(ReadOnlyMatrixSpan<T> p)  (+2 overloads)` | Applies `Logit` elementwise to the input matrix and returns a new matrix. / `Logit` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static VectorSpan<T> Logit<T>(ReadOnlyVectorSpan<T> p, VectorSpan<T> destination)  (+2 overloads)` | Applies `Logit` elementwise to the input vector, writes the result to `destination`, and returns it. / `Logit` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Logit<T>(ReadOnlyMatrixSpan<T> p, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Logit` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Logit` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static Vector<double> Magnitude(ReadOnlyVectorSpan<Complex> x)` | Computes the magnitude `\|z\|` of each element in a complex vector. / 複素数ベクトルの各要素の大きさ `\|z\|` を計算します。 |
| `static Matrix<double> Magnitude(ReadOnlyMatrixSpan<Complex> x)` | Computes the magnitude `\|z\|` of each element in a complex matrix. / 複素数行列の各要素の大きさ `\|z\|` を計算します。 |
| `static VectorSpan<double> Magnitude(ReadOnlyVectorSpan<Complex> x, VectorSpan<double> destination)` | Computes the magnitude `\|z\|` of each element in a complex vector and writes the result to `destination`. / 複素数ベクトルの各要素の大きさ `\|z\|` を計算し、`destination` に書き込みます。 |
| `static MatrixSpan<double> Magnitude(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<double> destination)` | Computes the magnitude `\|z\|` of each element in a complex matrix and writes the result to `destination`. / 複素数行列の各要素の大きさ `\|z\|` を計算し、`destination` に書き込みます。 |
| `static Tensor<Half> Map<TOperator>(ReadOnlyTensorSpan<Half> x)  (+3 overloads)` | Applies a stateless unary operator `TOperator` elementwise to a tensor. / 状態を持たない1引数演算子 `TOperator` をテンソルの各要素に適用します。 |
| `static Vector<Half> Map<TOperator>(ReadOnlyVectorSpan<Half> x)  (+3 overloads)` | Applies a stateless unary operator `TOperator` elementwise to a vector. / 状態を持たない1引数演算子 `TOperator` をベクトルの各要素に適用します。 |
| `static Matrix<Half> Map<TOperator>(ReadOnlyMatrixSpan<Half> x)  (+3 overloads)` | Applies a stateless unary operator `TOperator` elementwise to a matrix. / 状態を持たない1引数演算子 `TOperator` を行列の各要素に適用します。 |
| `static Matrix<decimal> Map<TOperator>(ReadOnlyMatrixSpan<decimal> x)` | Applies a stateless unary operator `TOperator` elementwise to a matrix. / 状態を持たない1引数演算子 `TOperator` を行列の各要素に適用します。 |
| `static Matrix<Complex> Map<TOperator>(ReadOnlyMatrixSpan<Complex> x)` | Applies a stateless unary operator `TOperator` elementwise to a matrix. / 状態を持たない1引数演算子 `TOperator` を行列の各要素に適用します。 |
| `static Tensor<decimal> Map<TOperator>(ReadOnlyTensorSpan<decimal> x)` | Applies a stateless unary operator `TOperator` elementwise to a tensor. / 状態を持たない1引数演算子 `TOperator` をテンソルの各要素に適用します。 |
| `static Tensor<Complex> Map<TOperator>(ReadOnlyTensorSpan<Complex> x)` | Applies a stateless unary operator `TOperator` elementwise to a tensor. / 状態を持たない1引数演算子 `TOperator` をテンソルの各要素に適用します。 |
| `static Vector<decimal> Map<TOperator>(ReadOnlyVectorSpan<decimal> x)` | Applies a stateless unary operator `TOperator` elementwise to a vector. / 状態を持たない1引数演算子 `TOperator` をベクトルの各要素に適用します。 |
| `static Vector<Complex> Map<TOperator>(ReadOnlyVectorSpan<Complex> x)` | Applies a stateless unary operator `TOperator` elementwise to a vector. / 状態を持たない1引数演算子 `TOperator` をベクトルの各要素に適用します。 |
| `static TensorSpan<Half> Map<TOperator>(ReadOnlyTensorSpan<Half> x, TensorSpan<Half> destination)  (+3 overloads)` | Applies a stateless unary operator `TOperator` elementwise to a tensor and writes the result to `destination`. / 状態を持たない1引数演算子 `TOperator` をテンソルの各要素に適用し、`destination` に書き込みます。 |
| `static Tensor<T> Map<T, TOperator>(ReadOnlyTensorSpan<T> x, TOperator op)  (+2 overloads)` | Applies a stateful unary operator `op` elementwise to a tensor. / 状態を持つ1引数演算子 `op` をテンソルの各要素に適用します。 |
| `static VectorSpan<Half> Map<TOperator>(ReadOnlyVectorSpan<Half> x, VectorSpan<Half> destination)  (+3 overloads)` | Applies a stateless unary operator `TOperator` elementwise to a vector and writes the result to `destination`. / 状態を持たない1引数演算子 `TOperator` をベクトルの各要素に適用し、`destination` に書き込みます。 |
| `static MatrixSpan<Half> Map<TOperator>(ReadOnlyMatrixSpan<Half> x, MatrixSpan<Half> destination)  (+3 overloads)` | Applies a stateless unary operator `TOperator` elementwise to a matrix and writes the result to `destination`. / 状態を持たない1引数演算子 `TOperator` を行列の各要素に適用し、`destination` に書き込みます。 |
| `static Vector<T> Map<T, TOperator>(ReadOnlyVectorSpan<T> x, TOperator op)  (+2 overloads)` | Applies a stateful unary operator `op` elementwise to a vector. / 状態を持つ1引数演算子 `op` をベクトルの各要素に適用します。 |
| `static Matrix<T> Map<T, TOperator>(ReadOnlyMatrixSpan<T> x, TOperator op)  (+2 overloads)` | Applies a stateful unary operator `op` elementwise to a matrix. / 状態を持つ1引数演算子 `op` を行列の各要素に適用します。 |
| `static MatrixSpan<decimal> Map<TOperator>(ReadOnlyMatrixSpan<decimal> x, MatrixSpan<decimal> destination)` | Applies a stateless unary operator `TOperator` elementwise to a matrix and writes the result to `destination`. / 状態を持たない1引数演算子 `TOperator` を行列の各要素に適用し、`destination` に書き込みます。 |
| `static MatrixSpan<Complex> Map<TOperator>(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination)` | Applies a stateless unary operator `TOperator` elementwise to a matrix and writes the result to `destination`. / 状態を持たない1引数演算子 `TOperator` を行列の各要素に適用し、`destination` に書き込みます。 |
| `static TensorSpan<decimal> Map<TOperator>(ReadOnlyTensorSpan<decimal> x, TensorSpan<decimal> destination)` | Applies a stateless unary operator `TOperator` elementwise to a tensor and writes the result to `destination`. / 状態を持たない1引数演算子 `TOperator` をテンソルの各要素に適用し、`destination` に書き込みます。 |
| `static TensorSpan<Complex> Map<TOperator>(ReadOnlyTensorSpan<Complex> x, TensorSpan<Complex> destination)` | Applies a stateless unary operator `TOperator` elementwise to a tensor and writes the result to `destination`. / 状態を持たない1引数演算子 `TOperator` をテンソルの各要素に適用し、`destination` に書き込みます。 |
| `static VectorSpan<decimal> Map<TOperator>(ReadOnlyVectorSpan<decimal> x, VectorSpan<decimal> destination)` | Applies a stateless unary operator `TOperator` elementwise to a vector and writes the result to `destination`. / 状態を持たない1引数演算子 `TOperator` をベクトルの各要素に適用し、`destination` に書き込みます。 |
| `static VectorSpan<Complex> Map<TOperator>(ReadOnlyVectorSpan<Complex> x, VectorSpan<Complex> destination)` | Applies a stateless unary operator `TOperator` elementwise to a vector and writes the result to `destination`. / 状態を持たない1引数演算子 `TOperator` をベクトルの各要素に適用し、`destination` に書き込みます。 |
| `static TensorSpan<T> Map<T, TOperator>(ReadOnlyTensorSpan<T> x, TOperator op, TensorSpan<T> destination)  (+2 overloads)` | Applies a stateful unary operator `op` elementwise to a tensor and writes the result to `destination`. / 状態を持つ1引数演算子 `op` をテンソルの各要素に適用し、`destination` に書き込みます。 |
| `static VectorSpan<T> Map<T, TOperator>(ReadOnlyVectorSpan<T> x, TOperator op, VectorSpan<T> destination)  (+2 overloads)` | Applies a stateful unary operator `op` elementwise to a vector and writes the result to `destination`. / 状態を持つ1引数演算子 `op` をベクトルの各要素に適用し、`destination` に書き込みます。 |
| `static MatrixSpan<T> Map<T, TOperator>(ReadOnlyMatrixSpan<T> x, TOperator op, MatrixSpan<T> destination)  (+2 overloads)` | Applies a stateful unary operator `op` elementwise to a matrix and writes the result to `destination`. / 状態を持つ1引数演算子 `op` を行列の各要素に適用し、`destination` に書き込みます。 |
| `static T Max<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Returns the maximum element in a vector. / ベクトル要素の最大値を返します。 |
| `static T Max<T>(T x, T y)` | Returns the larger of two values (`max(x, y)`). / 2つの値のうち大きい方（`max(x, y)`）を返します。 |
| `static T Max<T>(ReadOnlyVectorSpan<T> x, bool ignoreNaN)  (+2 overloads)` | Returns the maximum element in a vector, optionally excluding `NaN` values. / 必要に応じて `NaN` を除外して、ベクトル要素の最大値を返します。 |
| `static Vector<T> Max<T>(ReadOnlyVectorSpan<T> x, T y)  (+2 overloads)` | Applies `Max` elementwise to the input vector and returns a new vector. / `Max` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Max<T>(ReadOnlyMatrixSpan<T> x, T y)  (+2 overloads)` | Applies `Max` elementwise to the input matrix and returns a new matrix. / `Max` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static Vector<T> Max<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)  (+2 overloads)` | Applies `Max` elementwise to the input vector and returns a new vector. / `Max` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Max<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y)  (+2 overloads)` | Applies `Max` elementwise to the input matrix and returns a new matrix. / `Max` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static VectorSpan<T> Max<T>(ReadOnlyVectorSpan<T> x, T y, VectorSpan<T> destination)  (+2 overloads)` | Applies `Max` elementwise to the input vector, writes the result to `destination`, and returns it. / `Max` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Max<T>(ReadOnlyMatrixSpan<T> x, T y, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Max` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Max` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static VectorSpan<T> Max<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, VectorSpan<T> destination)  (+2 overloads)` | Applies `Max` elementwise to the input vector, writes the result to `destination`, and returns it. / `Max` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Max<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Max` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Max` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static T Mean<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Computes the arithmetic mean `Σ x[i] / n` of the elements in a vector. / ベクトル要素の算術平均 `Σ x[i] / n` を計算します。 |
| `static T Mean<T>(ReadOnlyVectorSpan<T> x, bool ignoreNaN)  (+2 overloads)` | Computes the arithmetic mean of the elements in a vector, optionally excluding `NaN` values. / 必要に応じて `NaN` を除外して、ベクトル要素の算術平均を計算します。 |
| `static T Median<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Computes the median of the elements in a vector. / ベクトル要素の中央値を計算します。 |
| `static T Median<T>(ReadOnlyVectorSpan<T> x, bool ignoreNaN)  (+2 overloads)` | Computes the median of the elements in a vector, optionally excluding `NaN` values. / 必要に応じて `NaN` を除外して、ベクトル要素の中央値を計算します。 |
| `static T Min<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Returns the minimum element in a vector. / ベクトル要素の最小値を返します。 |
| `static T Min<T>(T x, T y)` | Returns the smaller of two values (`min(x, y)`). / 2つの値のうち小さい方（`min(x, y)`）を返します。 |
| `static T Min<T>(ReadOnlyVectorSpan<T> x, bool ignoreNaN)  (+2 overloads)` | Returns the minimum element in a vector, optionally excluding `NaN` values. / 必要に応じて `NaN` を除外して、ベクトル要素の最小値を返します。 |
| `static Vector<T> Min<T>(ReadOnlyVectorSpan<T> x, T y)  (+2 overloads)` | Applies `Min` elementwise to the input vector and returns a new vector. / `Min` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Min<T>(ReadOnlyMatrixSpan<T> x, T y)  (+2 overloads)` | Applies `Min` elementwise to the input matrix and returns a new matrix. / `Min` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static Vector<T> Min<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)  (+2 overloads)` | Applies `Min` elementwise to the input vector and returns a new vector. / `Min` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Min<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y)  (+2 overloads)` | Applies `Min` elementwise to the input matrix and returns a new matrix. / `Min` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static VectorSpan<T> Min<T>(ReadOnlyVectorSpan<T> x, T y, VectorSpan<T> destination)  (+2 overloads)` | Applies `Min` elementwise to the input vector, writes the result to `destination`, and returns it. / `Min` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Min<T>(ReadOnlyMatrixSpan<T> x, T y, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Min` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Min` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static VectorSpan<T> Min<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, VectorSpan<T> destination)  (+2 overloads)` | Applies `Min` elementwise to the input vector, writes the result to `destination`, and returns it. / `Min` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Min<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Min` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Min` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static Tensor<T> Multiply<T>(ReadOnlyTensorSpan<T> x, T scalar)` | Multiplies every element of a tensor by a scalar (`X[k] · scalar`). / テンソルの全要素をスカラー倍します（`X[k] · scalar`）。 |
| `static Tensor<T> Multiply<T>(ReadOnlyTensorSpan<T> a, ReadOnlyTensorSpan<T> b)  (+2 overloads)` | Computes the batched matrix product `C[:, :, k] = A[:, :, k] · B[:, :, k]` for each matrix slice `k`. / 各行列スライス `k` についてバッチ行列積 `C[:, :, k] = A[:, :, k] · B[:, :, k]` を計算します。 |
| `static Vector<T> Multiply<T>(ReadOnlyVectorSpan<T> x, T scalar)` | Multiplies every element of a vector by a scalar (`x · scalar`). / ベクトルの全要素をスカラー倍します（`x · scalar`）。 |
| `static Matrix<T> Multiply<T>(ReadOnlyMatrixSpan<T> x, T scalar)` | Multiplies every element of a matrix by a scalar (`x · scalar`). / 行列の全要素をスカラー倍します（`x · scalar`）。 |
| `static Matrix<T> Multiply<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b)  (+2 overloads)` | Computes the matrix product `AB` and returns a new matrix. / 行列積 `AB` を計算し、新しい行列を返します。 |
| `static Vector<T> Multiply<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Computes the matrix-vector product `Ax` and returns a new vector. / 行列ベクトル積 `Ax` を計算し、新しいベクトルを返します。 |
| `static TensorSpan<T> Multiply<T>(ReadOnlyTensorSpan<T> x, T scalar, TensorSpan<T> destination)` | Multiplies every element of a tensor by a scalar and writes the result to `destination`. / テンソルの全要素をスカラー倍し、`destination` に書き込みます。 |
| `static TensorSpan<T> Multiply<T>(ReadOnlyTensorSpan<T> a, ReadOnlyTensorSpan<T> b, TensorSpan<T> destination)  (+2 overloads)` | Computes the batched matrix product `A[:, :, k] · B[:, :, k]` and writes the result to `destination`. / 行列ごとの積 `A[:, :, k] · B[:, :, k]` を計算し、`destination` に書き込みます。 |
| `static VectorSpan<T> Multiply<T>(ReadOnlyVectorSpan<T> x, T scalar, VectorSpan<T> destination)` | Multiplies every element of a vector by a scalar and writes the result to `destination`. / ベクトルの全要素をスカラー倍し、`destination` に書き込みます。 |
| `static MatrixSpan<T> Multiply<T>(ReadOnlyMatrixSpan<T> x, T scalar, MatrixSpan<T> destination)` | Multiplies every element of a matrix by a scalar and writes the result to `destination`. / 行列の全要素をスカラー倍し、`destination` に書き込みます。 |
| `static MatrixSpan<T> Multiply<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b, MatrixSpan<T> destination)  (+2 overloads)` | Computes the matrix product `AB` and writes the result to `destination`. / 行列積 `AB` を計算し、結果を `destination` に書き込みます。 |
| `static VectorSpan<T> Multiply<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Computes the matrix-vector product `Ax` and writes the result to `destination`. / 行列ベクトル積 `Ax` を計算し、結果を `destination` に書き込みます。 |
| `static Tensor<T> Negate<T>(ReadOnlyTensorSpan<T> x)  (+2 overloads)` | Computes the elementwise negation `-X[k]` of a tensor. / テンソルの要素ごとの符号反転 `-X[k]` を計算します。 |
| `static Vector<T> Negate<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Computes the elementwise negation `-x` of a vector. / ベクトルの要素ごとの符号反転 `-x` を計算します。 |
| `static Matrix<T> Negate<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Computes the elementwise negation `-x` of a matrix. / 行列の要素ごとの符号反転 `-x` を計算します。 |
| `static TensorSpan<T> Negate<T>(ReadOnlyTensorSpan<T> x, TensorSpan<T> destination)  (+2 overloads)` | Computes the elementwise negation `-X[k]` of a tensor and writes the result to `destination`. / テンソルの要素ごとの符号反転 `-X[k]` を計算し、`destination` に書き込みます。 |
| `static VectorSpan<T> Negate<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Computes the elementwise negation `-x` of a vector and writes the result to `destination`. / ベクトルの要素ごとの符号反転 `-x` を計算し、`destination` に書き込みます。 |
| `static MatrixSpan<T> Negate<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Computes the elementwise negation `-x` of a matrix and writes the result to `destination`. / 行列の要素ごとの符号反転 `-x` を計算し、`destination` に書き込みます。 |
| `static T Norm<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Computes the Euclidean (L2) norm `‖x‖₂` of a vector without undue intermediate overflow or underflow. / 途中のオーバーフローやアンダーフローを回避して、ベクトルのユークリッドノルム `‖x‖₂` を計算します。 |
| `static T Norm1<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Computes the L1 norm `Σ \|x[i]\|` of a vector. / ベクトルの 1 ノルム `Σ \|x[i]\|` を計算します。 |
| `static T NormInfinity<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Computes the infinity (maximum) norm `max \|x[i]\|` of a vector. / ベクトルの最大値ノルム `max \|x[i]\|` を計算します。 |
| `static Matrix<T> Outer<T>(ReadOnlyVectorSpan<T> u, ReadOnlyVectorSpan<T> v)  (+2 overloads)` | Computes the outer product `u vᵀ` of two vectors. / 2 つのベクトルの外積（直積）`u vᵀ` を計算します。 |
| `static MatrixSpan<T> Outer<T>(ReadOnlyVectorSpan<T> u, ReadOnlyVectorSpan<T> v, MatrixSpan<T> destination)  (+2 overloads)` | Writes the outer product `u vᵀ` of two vectors to `destination`. / 2 つのベクトルの外積（直積）`u vᵀ` を `destination` に書き込みます。 |
| `static Matrix<Half> Pairwise<TOperator>(ReadOnlyMatrixSpan<Half> x, ReadOnlyMatrixSpan<Half> y)  (+3 overloads)` | Applies a stateless pairwise operator to every pair of columns of `x` and `y` (`result[i, j] = TOperator(x.Column(i), y.Column(j))`). / 状態を持たないペア演算子を `x` と `y` のすべての列ペアに適用します（`result[i, j] = TOperator(x.Column(i), y.Column(j))`）。 |
| `static Matrix<decimal> Pairwise<TOperator>(ReadOnlyMatrixSpan<decimal> x, ReadOnlyMatrixSpan<decimal> y)` | Applies a stateless pairwise operator to every pair of columns of `x` and `y` (`result[i, j] = TOperator(x.Column(i), y.Column(j))`). / 状態を持たないペア演算子を `x` と `y` のすべての列ペアに適用します（`result[i, j] = TOperator(x.Column(i), y.Column(j))`）。 |
| `static Matrix<Complex> Pairwise<TOperator>(ReadOnlyMatrixSpan<Complex> x, ReadOnlyMatrixSpan<Complex> y)` | Applies a stateless pairwise operator to every pair of columns of `x` and `y` (`result[i, j] = TOperator(x.Column(i), y.Column(j))`). / 状態を持たないペア演算子を `x` と `y` のすべての列ペアに適用します（`result[i, j] = TOperator(x.Column(i), y.Column(j))`）。 |
| `static MatrixSpan<Half> Pairwise<TOperator>(ReadOnlyMatrixSpan<Half> x, ReadOnlyMatrixSpan<Half> y, MatrixSpan<Half> destination)  (+3 overloads)` | Applies a stateless pairwise operator to every pair of columns of `x` and `y` and writes the result to `destination`. / 状態を持たないペア演算子を `x` と `y` のすべての列ペアに適用し、`destination` に書き込みます。 |
| `static Matrix<T> Pairwise<T, TOperator>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, TOperator op)  (+2 overloads)` | Applies a stateful pairwise operator `op` to every pair of columns of `x` and `y` (`result[i, j] = op(x.Column(i), y.Column(j))`). / 状態を持つペア演算子 `op` を `x` と `y` のすべての列ペアに適用します（`result[i, j] = op(x.Column(i), y.Column(j))`）。 |
| `static MatrixSpan<decimal> Pairwise<TOperator>(ReadOnlyMatrixSpan<decimal> x, ReadOnlyMatrixSpan<decimal> y, MatrixSpan<decimal> destination)` | Applies a stateless pairwise operator to every pair of columns of `x` and `y` and writes the result to `destination`. / 状態を持たないペア演算子を `x` と `y` のすべての列ペアに適用し、`destination` に書き込みます。 |
| `static MatrixSpan<Complex> Pairwise<TOperator>(ReadOnlyMatrixSpan<Complex> x, ReadOnlyMatrixSpan<Complex> y, MatrixSpan<Complex> destination)` | Applies a stateless pairwise operator to every pair of columns of `x` and `y` and writes the result to `destination`. / 状態を持たないペア演算子を `x` と `y` のすべての列ペアに適用し、`destination` に書き込みます。 |
| `static MatrixSpan<T> Pairwise<T, TOperator>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, TOperator op, MatrixSpan<T> destination)  (+2 overloads)` | Applies a stateful pairwise operator `op` to every pair of columns of `x` and `y` and writes the result to `destination`. / 状態を持つペア演算子 `op` を `x` と `y` のすべての列ペアに適用し、`destination` に書き込みます。 |
| `static Matrix<Half> PairwiseSymmetric<TOperator>(ReadOnlyMatrixSpan<Half> x)  (+3 overloads)` | Applies a stateless symmetric pairwise operator to all column pairs of `x`, evaluating only the lower triangle (`i ≥ j`) and mirroring to the upper triangle. / 状態を持たない対称ペア演算子を `x` の下三角（`i ≥ j`）の列ペアのみに適用し、上三角へ対称に転記して対称行列を返します。 |
| `static Matrix<decimal> PairwiseSymmetric<TOperator>(ReadOnlyMatrixSpan<decimal> x)` | Applies a stateless symmetric pairwise operator to all column pairs of `x`, evaluating only the lower triangle (`i ≥ j`) and mirroring to the upper triangle. / 状態を持たない対称ペア演算子を `x` の下三角（`i ≥ j`）の列ペアのみに適用し、上三角へ対称に転記して対称行列を返します。 |
| `static Matrix<Complex> PairwiseSymmetric<TOperator>(ReadOnlyMatrixSpan<Complex> x)` | Applies a stateless symmetric pairwise operator to all column pairs of `x`, evaluating only the lower triangle (`i ≥ j`) and mirroring to the upper triangle. / 状態を持たない対称ペア演算子を `x` の下三角（`i ≥ j`）の列ペアのみに適用し、上三角へ対称に転記して対称行列を返します。 |
| `static MatrixSpan<Half> PairwiseSymmetric<TOperator>(ReadOnlyMatrixSpan<Half> x, MatrixSpan<Half> destination)  (+3 overloads)` | Applies a stateless symmetric pairwise operator to all column pairs of `x` and writes the full symmetric matrix to `destination`. / 状態を持たない対称ペア演算子を `x` の列ペアに適用し、対称行列全体を `destination` に書き込みます。 |
| `static Matrix<T> PairwiseSymmetric<T, TOperator>(ReadOnlyMatrixSpan<T> x, TOperator op)  (+2 overloads)` | Applies a stateful symmetric pairwise operator `op` to all column pairs of `x`, evaluating only the lower triangle (`i ≥ j`) and mirroring to the upper triangle. / 状態を持つ対称ペア演算子 `op` を `x` の下三角（`i ≥ j`）の列ペアのみに適用し、上三角へ対称に転記して対称行列を返します。 |
| `static MatrixSpan<decimal> PairwiseSymmetric<TOperator>(ReadOnlyMatrixSpan<decimal> x, MatrixSpan<decimal> destination)` | Applies a stateless symmetric pairwise operator to all column pairs of `x` and writes the full symmetric matrix to `destination`. / 状態を持たない対称ペア演算子を `x` の列ペアに適用し、対称行列全体を `destination` に書き込みます。 |
| `static MatrixSpan<Complex> PairwiseSymmetric<TOperator>(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination)` | Applies a stateless symmetric pairwise operator to all column pairs of `x` and writes the full symmetric matrix to `destination`. / 状態を持たない対称ペア演算子を `x` の列ペアに適用し、対称行列全体を `destination` に書き込みます。 |
| `static MatrixSpan<T> PairwiseSymmetric<T, TOperator>(ReadOnlyMatrixSpan<T> x, TOperator op, MatrixSpan<T> destination)  (+2 overloads)` | Applies a stateful symmetric pairwise operator `op` to all column pairs of `x` and writes the full symmetric matrix to `destination`. / 状態を持つ対称ペア演算子 `op` を `x` の列ペアに適用し、対称行列全体を `destination` に書き込みます。 |
| `static Vector<double> Phase(ReadOnlyVectorSpan<Complex> x)` | Computes the phase angle `atan2(im, re)` in `[-π, π]` of each element in a complex vector. / 複素数ベクトルの各要素の偏角 `atan2(im, re)`（範囲 `[-π, π]`）を計算します。 |
| `static Matrix<double> Phase(ReadOnlyMatrixSpan<Complex> x)` | Computes the phase angle `atan2(im, re)` in `[-π, π]` of each element in a complex matrix. / 複素数行列の各要素の偏角 `atan2(im, re)`（範囲 `[-π, π]`）を計算します。 |
| `static VectorSpan<double> Phase(ReadOnlyVectorSpan<Complex> x, VectorSpan<double> destination)` | Computes the phase angle `atan2(im, re)` of each element in a complex vector and writes the result to `destination`. / 複素数ベクトルの各要素の偏角 `atan2(im, re)` を計算し、`destination` に書き込みます。 |
| `static MatrixSpan<double> Phase(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<double> destination)` | Computes the phase angle `atan2(im, re)` of each element in a complex matrix and writes the result to `destination`. / 複素数行列の各要素の偏角 `atan2(im, re)` を計算し、`destination` に書き込みます。 |
| `static T Pow<T>(T x, T y)` | Computes `x` raised to the power `y` (`x^y`). / `x` の `y` 乗（`x^y`）を計算します。 |
| `static Vector<T> Pow<T>(ReadOnlyVectorSpan<T> x, T y)  (+2 overloads)` | Applies `Pow` elementwise to the input vector and returns a new vector. / `Pow` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Pow<T>(ReadOnlyMatrixSpan<T> x, T y)  (+2 overloads)` | Applies `Pow` elementwise to the input matrix and returns a new matrix. / `Pow` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static Vector<T> Pow<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)  (+2 overloads)` | Applies `Pow` elementwise to the input vector and returns a new vector. / `Pow` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Pow<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y)  (+2 overloads)` | Applies `Pow` elementwise to the input matrix and returns a new matrix. / `Pow` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static VectorSpan<T> Pow<T>(ReadOnlyVectorSpan<T> x, T y, VectorSpan<T> destination)  (+2 overloads)` | Applies `Pow` elementwise to the input vector, writes the result to `destination`, and returns it. / `Pow` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Pow<T>(ReadOnlyMatrixSpan<T> x, T y, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Pow` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Pow` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static VectorSpan<T> Pow<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, VectorSpan<T> destination)  (+2 overloads)` | Applies `Pow` elementwise to the input vector, writes the result to `destination`, and returns it. / `Pow` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Pow<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Pow` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Pow` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static T Quantile<T>(ReadOnlyVectorSpan<T> x, double probability)  (+2 overloads)` | Computes the quantile `Q(p)` of the elements in a vector. / ベクトル要素の分位点 `Q(p)` を計算します。 |
| `static T Quantile<T>(ReadOnlyVectorSpan<T> x, double probability, bool ignoreNaN)  (+2 overloads)` | Computes the quantile of the elements in a vector, optionally excluding `NaN` values. / 必要に応じて `NaN` を除外して、ベクトル要素の分位点を計算します。 |
| `static Vector<T> Quantiles<T>(ReadOnlyVectorSpan<T> x, ReadOnlySpan<double> probabilities)  (+2 overloads)` | Computes quantiles of the elements in a vector at multiple probabilities. / ベクトル要素に対して複数の確率における分位点を計算します。 |
| `static VectorSpan<T> Quantiles<T>(ReadOnlyVectorSpan<T> x, ReadOnlySpan<double> probabilities, VectorSpan<T> destination)  (+2 overloads)` | Computes quantiles of the elements in a vector at multiple probabilities and writes the result to `destination`. / ベクトル要素に対して複数の確率における分位点を計算し、`destination` に書き込みます。 |
| `static Vector<T> Quantiles<T>(ReadOnlyVectorSpan<T> x, ReadOnlySpan<double> probabilities, bool ignoreNaN)  (+2 overloads)` | Computes quantiles of the elements in a vector at multiple probabilities, optionally excluding `NaN` values. / 必要に応じて `NaN` を除外して、ベクトル要素に対する複数の確率の分位点を計算します。 |
| `static VectorSpan<T> Quantiles<T>(ReadOnlyVectorSpan<T> x, ReadOnlySpan<double> probabilities, bool ignoreNaN, VectorSpan<T> destination)  (+2 overloads)` | Computes quantiles of the elements in a vector at multiple probabilities, optionally excluding `NaN` values, and writes the result to `destination`. / 必要に応じて `NaN` を除外して複数の確率の分位点を計算し、`destination` に書き込みます。 |
| `static Vector<double> Real(ReadOnlyVectorSpan<Complex> x)` | Extracts the real part of each element in a complex vector. / 複素数ベクトルの各要素の実部を抽出します。 |
| `static Matrix<double> Real(ReadOnlyMatrixSpan<Complex> x)` | Extracts the real part of each element in a complex matrix. / 複素数行列の各要素の実部を抽出します。 |
| `static VectorSpan<double> Real(ReadOnlyVectorSpan<Complex> x, VectorSpan<double> destination)` | Extracts the real part of each element in a complex vector and writes the result to `destination`. / 複素数ベクトルの各要素の実部を抽出し、`destination` に書き込みます。 |
| `static MatrixSpan<double> Real(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<double> destination)` | Extracts the real part of each element in a complex matrix and writes the result to `destination`. / 複素数行列の各要素の実部を抽出し、`destination` に書き込みます。 |
| `static T Reciprocal<T>(T x)` | Computes the reciprocal `1 / x`. / 逆数 `1 / x` を計算します。 |
| `static Vector<T> Reciprocal<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Applies `Reciprocal` elementwise to the input vector and returns a new vector. / `Reciprocal` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Reciprocal<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Applies `Reciprocal` elementwise to the input matrix and returns a new matrix. / `Reciprocal` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static VectorSpan<T> Reciprocal<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Applies `Reciprocal` elementwise to the input vector, writes the result to `destination`, and returns it. / `Reciprocal` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Reciprocal<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Reciprocal` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Reciprocal` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static T Relu<T>(T x)` | Computes the rectified linear unit `max(x, 0)`. / ReLU 関数 `max(x, 0)` を計算します。 |
| `static Vector<T> Relu<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Applies `Relu` elementwise to the input vector and returns a new vector. / `Relu` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Relu<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Applies `Relu` elementwise to the input matrix and returns a new matrix. / `Relu` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static VectorSpan<T> Relu<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Applies `Relu` elementwise to the input vector, writes the result to `destination`, and returns it. / `Relu` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Relu<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Relu` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Relu` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static int[] RowIndexOfMax<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Returns the zero-based column index of the first maximum element in each row of a matrix. / 行列の各行について、最大値を持つ最初の要素の列インデックスを返します。 |
| `static Span<int> RowIndexOfMax<T>(ReadOnlyMatrixSpan<T> x, Span<int> destination)  (+2 overloads)` | Computes the zero-based column index of the first maximum element in each row of a matrix and writes the result to `destination`. / 行列の各行で最大値を持つ最初の要素の列インデックスを計算し、`destination` に書き込みます。 |
| `static int[] RowIndexOfMin<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Returns the zero-based column index of the first minimum element in each row of a matrix. / 行列の各行について、最小値を持つ最初の要素の列インデックスを返します。 |
| `static Span<int> RowIndexOfMin<T>(ReadOnlyMatrixSpan<T> x, Span<int> destination)  (+2 overloads)` | Computes the zero-based column index of the first minimum element in each row of a matrix and writes the result to `destination`. / 行列の各行で最小値を持つ最初の要素の列インデックスを計算し、`destination` に書き込みます。 |
| `static Matrix<T> RowLogSoftmax<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Computes the log-softmax independently for each row of a matrix. / 行列の各行ごとに対数ソフトマックスを計算します。 |
| `static MatrixSpan<T> RowLogSoftmax<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Computes the log-softmax independently for each row of a matrix and writes the result to `destination`. / 行列の各行ごとに対数ソフトマックスを計算し、`destination` に書き込みます。 |
| `static Vector<T> RowLogSumExp<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Computes the log-sum-exp `ln Σ exp` of each row in a matrix. / 行列の各行の `ln Σ exp` を計算します。 |
| `static VectorSpan<T> RowLogSumExp<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Computes the log-sum-exp `ln Σ exp` of each row in a matrix and writes the result to `destination`. / 行列の各行の `ln Σ exp` を計算し、`destination` に書き込みます。 |
| `static Vector<T> RowMax<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Computes the maximum element of each row in a matrix. / 行列の各行の最大値を計算します。 |
| `static VectorSpan<T> RowMax<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Computes the maximum element of each row in a matrix and writes the result to `destination`. / 行列の各行の最大値を計算し、`destination` に書き込みます。 |
| `static Vector<T> RowMax<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN)  (+2 overloads)` | Computes the maximum element of each row in a matrix, optionally excluding `NaN` values. / 必要に応じて `NaN` を除外して、行列の各行の最大値を計算します。 |
| `static VectorSpan<T> RowMax<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN, VectorSpan<T> destination)  (+2 overloads)` | Computes the maximum element of each row in a matrix, optionally excluding `NaN` values, and writes the result to `destination`. / 必要に応じて `NaN` を除外して行列の各行の最大値を計算し、`destination` に書き込みます。 |
| `static Vector<T> RowMeans<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Computes the arithmetic mean of each row in a matrix. / 行列の各行の算術平均を計算します。 |
| `static VectorSpan<T> RowMeans<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Computes the arithmetic mean of each row in a matrix and writes the result to `destination`. / 行列の各行の算術平均を計算し、`destination` に書き込みます。 |
| `static Vector<T> RowMeans<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN)  (+2 overloads)` | Computes the arithmetic mean of each row in a matrix, optionally excluding `NaN` values. / 必要に応じて `NaN` を除外して、行列の各行の算術平均を計算します。 |
| `static VectorSpan<T> RowMeans<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN, VectorSpan<T> destination)  (+2 overloads)` | Computes the arithmetic mean of each row in a matrix, optionally excluding `NaN` values, and writes the result to `destination`. / 必要に応じて `NaN` を除外して行列の各行の算術平均を計算し、`destination` に書き込みます。 |
| `static Vector<T> RowMedians<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Computes the median of each row in a matrix. / 行列の各行の中央値を計算します。 |
| `static VectorSpan<T> RowMedians<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Computes the median of each row in a matrix and writes the result to `destination`. / 行列の各行の中央値を計算し、`destination` に書き込みます。 |
| `static Vector<T> RowMedians<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN)  (+2 overloads)` | Computes the median of each row in a matrix, optionally excluding `NaN` values. / 必要に応じて `NaN` を除外して、行列の各行の中央値を計算します。 |
| `static VectorSpan<T> RowMedians<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN, VectorSpan<T> destination)  (+2 overloads)` | Computes the median of each row in a matrix, optionally excluding `NaN` values, and writes the result to `destination`. / 必要に応じて `NaN` を除外して行列の各行の中央値を計算し、`destination` に書き込みます。 |
| `static Vector<T> RowMin<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Computes the minimum element of each row in a matrix. / 行列の各行の最小値を計算します。 |
| `static VectorSpan<T> RowMin<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Computes the minimum element of each row in a matrix and writes the result to `destination`. / 行列の各行の最小値を計算し、`destination` に書き込みます。 |
| `static Vector<T> RowMin<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN)  (+2 overloads)` | Computes the minimum element of each row in a matrix, optionally excluding `NaN` values. / 必要に応じて `NaN` を除外して、行列の各行の最小値を計算します。 |
| `static VectorSpan<T> RowMin<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN, VectorSpan<T> destination)  (+2 overloads)` | Computes the minimum element of each row in a matrix, optionally excluding `NaN` values, and writes the result to `destination`. / 必要に応じて `NaN` を除外して行列の各行の最小値を計算し、`destination` に書き込みます。 |
| `static Vector<T> RowNorms<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Computes the Euclidean (L2) norm of each row in a matrix. / 行列の各行のユークリッドノルムを計算します。 |
| `static VectorSpan<T> RowNorms<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Computes the Euclidean (L2) norm of each row in a matrix and writes the result to `destination`. / 行列の各行のユークリッドノルムを計算し、`destination` に書き込みます。 |
| `static Vector<T> RowQuantiles<T>(ReadOnlyMatrixSpan<T> x, double probability)  (+2 overloads)` | Computes the quantile of each row in a matrix. / 行列の各行の分位点を計算します。 |
| `static VectorSpan<T> RowQuantiles<T>(ReadOnlyMatrixSpan<T> x, double probability, VectorSpan<T> destination)  (+2 overloads)` | Computes the quantile of each row in a matrix and writes the result to `destination`. / 行列の各行の分位点を計算し、`destination` に書き込みます。 |
| `static Vector<T> RowQuantiles<T>(ReadOnlyMatrixSpan<T> x, double probability, bool ignoreNaN)  (+2 overloads)` | Computes the quantile of each row in a matrix, optionally excluding `NaN` values. / 必要に応じて `NaN` を除外して、行列の各行の分位点を計算します。 |
| `static VectorSpan<T> RowQuantiles<T>(ReadOnlyMatrixSpan<T> x, double probability, bool ignoreNaN, VectorSpan<T> destination)  (+2 overloads)` | Computes the quantile of each row in a matrix, optionally excluding `NaN` values, and writes the result to `destination`. / 必要に応じて `NaN` を除外して行列の各行の分位点を計算し、`destination` に書き込みます。 |
| `static Matrix<T> RowSoftmax<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Computes the softmax distribution independently for each row of a matrix. / 行列の各行ごとにソフトマックス分布を計算します。 |
| `static MatrixSpan<T> RowSoftmax<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Computes the softmax distribution independently for each row of a matrix and writes the result to `destination`. / 行列の各行ごとにソフトマックス分布を計算し、`destination` に書き込みます。 |
| `static Vector<T> RowStandardDeviations<T>(ReadOnlyMatrixSpan<T> x, int correction)  (+2 overloads)` | Computes the standard deviation of each row in a matrix. / 行列の各行の標準偏差を計算します。 |
| `static VectorSpan<T> RowStandardDeviations<T>(ReadOnlyMatrixSpan<T> x, int correction, VectorSpan<T> destination)  (+2 overloads)` | Computes the standard deviation of each row in a matrix and writes the result to `destination`. / 行列の各行の標準偏差を計算し、`destination` に書き込みます。 |
| `static Vector<T> RowSums<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Computes the sum of each row in a matrix. / 行列の各行の総和を計算します。 |
| `static VectorSpan<T> RowSums<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Computes the sum of each row in a matrix and writes the result to `destination`. / 行列の各行の総和を計算し、`destination` に書き込みます。 |
| `static Vector<T> RowSums<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN)  (+2 overloads)` | Computes the sum of each row in a matrix, optionally excluding `NaN` values. / 必要に応じて `NaN` を除外して、行列の各行の総和を計算します。 |
| `static VectorSpan<T> RowSums<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN, VectorSpan<T> destination)  (+2 overloads)` | Computes the sum of each row in a matrix, optionally excluding `NaN` values, and writes the result to `destination`. / 必要に応じて `NaN` を除外して行列の各行の総和を計算し、`destination` に書き込みます。 |
| `static Vector<T> RowSumsOfSquares<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Computes the sum of squares of each row in a matrix. / 行列の各行の平方和を計算します。 |
| `static VectorSpan<T> RowSumsOfSquares<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Computes the sum of squares of each row in a matrix and writes the result to `destination`. / 行列の各行の平方和を計算し、`destination` に書き込みます。 |
| `static Vector<T> RowVariances<T>(ReadOnlyMatrixSpan<T> x, int correction)  (+2 overloads)` | Computes the variance of each row in a matrix. / 行列の各行の分散を計算します。 |
| `static VectorSpan<T> RowVariances<T>(ReadOnlyMatrixSpan<T> x, int correction, VectorSpan<T> destination)  (+2 overloads)` | Computes the variance of each row in a matrix and writes the result to `destination`. / 行列の各行の分散を計算し、`destination` に書き込みます。 |
| `static T Selu<T>(T x)` | Computes the scaled exponential linear unit `λ · ELU(x; α)` using the Klambauer et al. (2017) constants. / Klambauer ら (2017) の定数を用いて SELU `λ · ELU(x; α)` を計算します。 |
| `static Vector<T> Selu<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Applies `Selu` elementwise to the input vector and returns a new vector. / `Selu` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Selu<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Applies `Selu` elementwise to the input matrix and returns a new matrix. / `Selu` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static VectorSpan<T> Selu<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Applies `Selu` elementwise to the input vector, writes the result to `destination`, and returns it. / `Selu` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Selu<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Selu` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Selu` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static T Silu<T>(T x)` | Computes the sigmoid linear unit (SiLU / Swish) `x · σ(x) = x / (1 + e⁻ˣ)`. / SiLU（Swish）関数 `x · σ(x) = x / (1 + e⁻ˣ)` を計算します。 |
| `static Vector<T> Silu<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Applies `Silu` elementwise to the input vector and returns a new vector. / `Silu` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Silu<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Applies `Silu` elementwise to the input matrix and returns a new matrix. / `Silu` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static T Silu<T>(T x, bool approximate)` | Computes the SiLU (Swish) function `x · σ(x)`, optionally using a fast approximation for `Single`. / SiLU（Swish）関数 `x · σ(x)` を計算します（`Single` では高速な近似式を選択可能）。 |
| `static Vector<T> Silu<T>(ReadOnlyVectorSpan<T> x, bool approximate)  (+2 overloads)` | Applies the SiLU function elementwise to a vector, optionally using a fast approximation for `Single`. / ベクトルの各要素に SiLU 関数を適用します（`Single` では高速な近似式を選択可能）。 |
| `static Matrix<T> Silu<T>(ReadOnlyMatrixSpan<T> x, bool approximate)  (+2 overloads)` | Applies the SiLU function elementwise to a matrix, optionally using a fast approximation for `Single`. / 行列の各要素に SiLU 関数を適用します（`Single` では高速な近似式を選択可能）。 |
| `static VectorSpan<T> Silu<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Applies `Silu` elementwise to the input vector, writes the result to `destination`, and returns it. / `Silu` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Silu<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Silu` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Silu` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static VectorSpan<T> Silu<T>(ReadOnlyVectorSpan<T> x, bool approximate, VectorSpan<T> destination)  (+2 overloads)` | Applies the SiLU function elementwise to a vector and writes the result to `destination`. / ベクトルの各要素に SiLU 関数を適用し、`destination` に書き込みます。 |
| `static MatrixSpan<T> Silu<T>(ReadOnlyMatrixSpan<T> x, bool approximate, MatrixSpan<T> destination)  (+2 overloads)` | Applies the SiLU function elementwise to a matrix and writes the result to `destination`. / 行列の各要素に SiLU 関数を適用し、`destination` に書き込みます。 |
| `static Vector<T> SoftThreshold<T>(ReadOnlyVectorSpan<T> x, T threshold)` | Applies elementwise soft thresholding `sign(x) · max(\|x\| - threshold, 0)` to a vector. / ベクトルに対して要素ごとの軟判定しきい値 `sign(x) · max(\|x\| - threshold, 0)` を適用します。 |
| `static Matrix<T> SoftThreshold<T>(ReadOnlyMatrixSpan<T> x, T threshold)` | Applies elementwise soft thresholding `sign(x) · max(\|x\| - threshold, 0)` to a matrix. / 行列に対して要素ごとの軟判定しきい値 `sign(x) · max(\|x\| - threshold, 0)` を適用します。 |
| `static VectorSpan<T> SoftThreshold<T>(ReadOnlyVectorSpan<T> x, T threshold, VectorSpan<T> destination)` | Applies elementwise soft thresholding to a vector and writes the result to `destination`. / ベクトルに対して要素ごとの軟判定しきい値を適用し、`destination` に書き込みます。 |
| `static MatrixSpan<T> SoftThreshold<T>(ReadOnlyMatrixSpan<T> x, T threshold, MatrixSpan<T> destination)` | Applies elementwise soft thresholding to a matrix and writes the result to `destination`. / 行列に対して要素ごとの軟判定しきい値を適用し、`destination` に書き込みます。 |
| `static Vector<T> Softmax<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Computes the softmax distribution over a vector. / ベクトル全体に対するソフトマックス分布を計算します。 |
| `static VectorSpan<T> Softmax<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Computes the softmax distribution over a vector and writes the result to `destination`. / ベクトル全体に対するソフトマックス分布を計算し、`destination` に書き込みます。 |
| `static T Softplus<T>(T x)` | Computes the softplus function `ln(1 + eˣ)` without intermediate overflow. / 途中計算でのオーバーフローを回避して Softplus 関数 `ln(1 + eˣ)` を計算します。 |
| `static Vector<T> Softplus<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Applies `Softplus` elementwise to the input vector and returns a new vector. / `Softplus` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Softplus<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Applies `Softplus` elementwise to the input matrix and returns a new matrix. / `Softplus` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static T Softplus<T>(T x, bool approximate)` | Computes the softplus function `ln(1 + eˣ)`, optionally using a fast approximation for `Single`. / Softplus 関数 `ln(1 + eˣ)` を計算します（`Single` では高速な近似式を選択可能）。 |
| `static Vector<T> Softplus<T>(ReadOnlyVectorSpan<T> x, bool approximate)  (+2 overloads)` | Applies the softplus function elementwise to a vector, optionally using a fast approximation for `Single`. / ベクトルの各要素に Softplus 関数を適用します（`Single` では高速な近似式を選択可能）。 |
| `static Matrix<T> Softplus<T>(ReadOnlyMatrixSpan<T> x, bool approximate)  (+2 overloads)` | Applies the softplus function elementwise to a matrix, optionally using a fast approximation for `Single`. / 行列の各要素に Softplus 関数を適用します（`Single` では高速な近似式を選択可能）。 |
| `static VectorSpan<T> Softplus<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Applies `Softplus` elementwise to the input vector, writes the result to `destination`, and returns it. / `Softplus` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Softplus<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Softplus` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Softplus` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static VectorSpan<T> Softplus<T>(ReadOnlyVectorSpan<T> x, bool approximate, VectorSpan<T> destination)  (+2 overloads)` | Applies the softplus function elementwise to a vector and writes the result to `destination`. / ベクトルの各要素に Softplus 関数を適用し、`destination` に書き込みます。 |
| `static MatrixSpan<T> Softplus<T>(ReadOnlyMatrixSpan<T> x, bool approximate, MatrixSpan<T> destination)  (+2 overloads)` | Applies the softplus function elementwise to a matrix and writes the result to `destination`. / 行列の各要素に Softplus 関数を適用し、`destination` に書き込みます。 |
| `static Tensor<double> Solve(ReadOnlyTensorSpan<double> a, ReadOnlyTensorSpan<double> b)` | Solves the batched linear systems `A[k] · X[k] = B[k]` using slice-wise LU decomposition with partial pivoting. / 各行列スライスごとに部分ピボット付き LU 分解を用いて連立一次方程式 `A[k] · X[k] = B[k]` を解きます。 |
| `static Vector<double> Solve(ReadOnlyMatrixSpan<double> a, ReadOnlyVectorSpan<double> b)` | Solves the linear system `Ax = b` using LU decomposition with partial pivoting. / 部分ピボット付き LU 分解により連立一次方程式 `Ax = b` を解き、解ベクトルを返します。 |
| `static Matrix<double> Solve(ReadOnlyMatrixSpan<double> a, ReadOnlyMatrixSpan<double> b)` | Solves the linear system `AX = B` with multiple right-hand sides using LU decomposition with partial pivoting. / 部分ピボット付き LU 分解により複数右辺の連立一次方程式 `AX = B` を解き、解行列を返します。 |
| `static TensorSpan<double> Solve(ReadOnlyTensorSpan<double> a, ReadOnlyTensorSpan<double> b, TensorSpan<double> destination)` | Solves the batched linear systems `A[k] · X[k] = B[k]` using slice-wise LU decomposition and writes the result to `destination`. / 各行列スライスごとに部分ピボット付き LU 分解を用いて連立一次方程式 `A[k] · X[k] = B[k]` を解き、`destination` に書き込みます。 |
| `static VectorSpan<double> Solve(ReadOnlyMatrixSpan<double> a, ReadOnlyVectorSpan<double> b, VectorSpan<double> destination)` | Solves the linear system `Ax = b` using LU decomposition with partial pivoting and writes the solution to `destination`. / 部分ピボット付き LU 分解により連立一次方程式 `Ax = b` を解き、解を `destination` に書き込みます。 |
| `static MatrixSpan<double> Solve(ReadOnlyMatrixSpan<double> a, ReadOnlyMatrixSpan<double> b, MatrixSpan<double> destination)` | Solves the linear system `AX = B` using LU decomposition with partial pivoting and writes the solution to `destination`. / 部分ピボット付き LU 分解により複数右辺の連立一次方程式 `AX = B` を解き、解を `destination` に書き込みます。 |
| `static Tensor<T> SolveLowerTriangular<T>(ReadOnlyTensorSpan<T> a, ReadOnlyTensorSpan<T> b)  (+2 overloads)` | Solves the batched lower-triangular linear systems `L[k] · X[k] = B[k]`. / 各行列スライスについて下三角連立一次方程式 `L[k] · X[k] = B[k]` を解きます。 |
| `static Vector<T> SolveLowerTriangular<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyVectorSpan<T> b)  (+2 overloads)` | Solves the lower-triangular linear system `Lx = b` and returns the solution vector. / 下三角行列を係数とする連立一次方程式 `Lx = b` を解き、解ベクトルを返します。 |
| `static Matrix<T> SolveLowerTriangular<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b)  (+2 overloads)` | Solves the lower-triangular linear system `LX = B` with multiple right-hand sides and returns the solution matrix. / 下三角行列を係数とする複数右辺の連立一次方程式 `LX = B` を解き、解行列を返します。 |
| `static TensorSpan<T> SolveLowerTriangular<T>(ReadOnlyTensorSpan<T> a, ReadOnlyTensorSpan<T> b, TensorSpan<T> destination)  (+2 overloads)` | Solves the batched lower-triangular linear systems `L[k] · X[k] = B[k]` and writes the result to `destination`. / 各行列スライスについて下三角連立一次方程式 `L[k] · X[k] = B[k]` を解き、`destination` に書き込みます。 |
| `static VectorSpan<T> SolveLowerTriangular<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyVectorSpan<T> b, VectorSpan<T> destination)  (+2 overloads)` | Solves the lower-triangular linear system `Lx = b` and writes the solution to `destination`. / 下三角行列を係数とする連立一次方程式 `Lx = b` を解き、解を `destination` に書き込みます。 |
| `static MatrixSpan<T> SolveLowerTriangular<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b, MatrixSpan<T> destination)  (+2 overloads)` | Solves the lower-triangular linear system `LX = B` and writes the solution to `destination`. / 下三角行列を係数とする複数右辺の連立一次方程式 `LX = B` を解き、解を `destination` に書き込みます。 |
| `static Tensor<T> SolveUpperTriangular<T>(ReadOnlyTensorSpan<T> a, ReadOnlyTensorSpan<T> b)  (+2 overloads)` | Solves the batched upper-triangular linear systems `U[k] · X[k] = B[k]`. / 各行列スライスについて上三角連立一次方程式 `U[k] · X[k] = B[k]` を解きます。 |
| `static Vector<T> SolveUpperTriangular<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyVectorSpan<T> b)  (+2 overloads)` | Solves the upper-triangular linear system `Ux = b` and returns the solution vector. / 上三角行列を係数とする連立一次方程式 `Ux = b` を解き、解ベクトルを返します。 |
| `static Matrix<T> SolveUpperTriangular<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b)  (+2 overloads)` | Solves the upper-triangular linear system `UX = B` with multiple right-hand sides and returns the solution matrix. / 上三角行列を係数とする複数右辺の連立一次方程式 `UX = B` を解き、解行列を返します。 |
| `static TensorSpan<T> SolveUpperTriangular<T>(ReadOnlyTensorSpan<T> a, ReadOnlyTensorSpan<T> b, TensorSpan<T> destination)  (+2 overloads)` | Solves the batched upper-triangular linear systems `U[k] · X[k] = B[k]` and writes the result to `destination`. / 各行列スライスについて上三角連立一次方程式 `U[k] · X[k] = B[k]` を解き、`destination` に書き込みます。 |
| `static VectorSpan<T> SolveUpperTriangular<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyVectorSpan<T> b, VectorSpan<T> destination)  (+2 overloads)` | Solves the upper-triangular linear system `Ux = b` and writes the solution to `destination`. / 上三角行列を係数とする連立一次方程式 `Ux = b` を解き、解を `destination` に書き込みます。 |
| `static MatrixSpan<T> SolveUpperTriangular<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b, MatrixSpan<T> destination)  (+2 overloads)` | Solves the upper-triangular linear system `UX = B` and writes the solution to `destination`. / 上三角行列を係数とする複数右辺の連立一次方程式 `UX = B` を解き、解を `destination` に書き込みます。 |
| `static Vector<T> Sort<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Returns a new vector containing the elements of `x` sorted in ascending order. / `x` の要素を昇順に並べ替えた新しいベクトルを返します。 |
| `static VectorSpan<T> Sort<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Sorts the elements of `x` in ascending order into `destination`. / `x` の要素を昇順に並べ替え、`destination` に書き込みます。 |
| `static T Sqrt<T>(T x)` | Computes the square root `√x`. / 平方根 `√x` を計算します。 |
| `static Vector<T> Sqrt<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Applies `Sqrt` elementwise to the input vector and returns a new vector. / `Sqrt` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Sqrt<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Applies `Sqrt` elementwise to the input matrix and returns a new matrix. / `Sqrt` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static VectorSpan<T> Sqrt<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Applies `Sqrt` elementwise to the input vector, writes the result to `destination`, and returns it. / `Sqrt` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Sqrt<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Sqrt` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Sqrt` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static T Square<T>(T x)` | Computes the square `x²`. / 2乗 `x²` を計算します。 |
| `static Vector<T> Square<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Applies `Square` elementwise to the input vector and returns a new vector. / `Square` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Square<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Applies `Square` elementwise to the input matrix and returns a new matrix. / `Square` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static VectorSpan<T> Square<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Applies `Square` elementwise to the input vector, writes the result to `destination`, and returns it. / `Square` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Square<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Square` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Square` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static T SquaredDistance<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)  (+2 overloads)` | Computes the squared Euclidean distance `Σᵢ (xᵢ − yᵢ)²` between two vectors. / 2点間の二乗ユークリッド距離 `Σᵢ (xᵢ − yᵢ)²` を計算します。 |
| `static Matrix<T> SquaredDistance<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y)  (+2 overloads)` | Applies `SquaredDistance` to every pair of columns of the input matrices (`result[i, j]` corresponds to column i and column j) and returns a new matrix. / `SquaredDistance` を入力行列の列の組ごとに適用し（`result[i, j]` は列 i と列 j の値）、新しい行列を返します。 |
| `static MatrixSpan<T> SquaredDistance<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, MatrixSpan<T> destination)  (+2 overloads)` | Applies `SquaredDistance` to every pair of columns of the input matrices (`result[i, j]` corresponds to column i and column j), writes the result to `destination`, and returns it. / `SquaredDistance` を入力行列の列の組ごとに適用し（`result[i, j]` は列 i と列 j の値）、結果を `destination` に書き込んで返します。 |
| `static T StandardDeviation<T>(ReadOnlyVectorSpan<T> x, int correction)  (+2 overloads)` | Computes the standard deviation `√Variance(x, correction)` of the elements in a vector. / ベクトル要素の標準偏差 `√Variance(x, correction)` を計算します。 |
| `static Tensor<T> Subtract<T>(ReadOnlyTensorSpan<T> x, ReadOnlyTensorSpan<T> y)  (+2 overloads)` | Computes the elementwise difference `X[k] - Y[k]` of two tensors. / 2つのテンソルの要素ごとの差 `X[k] - Y[k]` を計算します。 |
| `static Vector<T> Subtract<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)  (+2 overloads)` | Computes the elementwise difference `x - y` of two vectors. / 2つのベクトルの要素ごとの差 `x - y` を計算します。 |
| `static Matrix<T> Subtract<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y)  (+2 overloads)` | Computes the elementwise difference `x - y` of two matrices. / 2つの行列の要素ごとの差 `x - y` を計算します。 |
| `static Vector<T> Subtract<T>(ReadOnlyVectorSpan<T> x, T scalar)` | Subtracts a scalar from every element of a vector (`x - scalar`). / ベクトルの全要素からスカラーを減算します（`x - scalar`）。 |
| `static Matrix<T> Subtract<T>(ReadOnlyMatrixSpan<T> x, T scalar)` | Subtracts a scalar from every element of a matrix (`x - scalar`). / 行列の全要素からスカラーを減算します（`x - scalar`）。 |
| `static TensorSpan<T> Subtract<T>(ReadOnlyTensorSpan<T> x, ReadOnlyTensorSpan<T> y, TensorSpan<T> destination)  (+2 overloads)` | Computes the elementwise difference `X[k] - Y[k]` of two tensors and writes the result to `destination`. / 2つのテンソルの要素ごとの差 `X[k] - Y[k]` を計算し、`destination` に書き込みます。 |
| `static VectorSpan<T> Subtract<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, VectorSpan<T> destination)  (+2 overloads)` | Computes the elementwise difference `x - y` of two vectors and writes the result to `destination`. / 2つのベクトルの要素ごとの差 `x - y` を計算し、`destination` に書き込みます。 |
| `static MatrixSpan<T> Subtract<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, MatrixSpan<T> destination)  (+2 overloads)` | Computes the elementwise difference `x - y` of two matrices and writes the result to `destination`. / 2つの行列の要素ごとの差 `x - y` を計算し、`destination` に書き込みます。 |
| `static VectorSpan<T> Subtract<T>(ReadOnlyVectorSpan<T> x, T scalar, VectorSpan<T> destination)` | Subtracts a scalar from every element of a vector and writes the result to `destination`. / ベクトルの全要素からスカラーを減算し、`destination` に書き込みます。 |
| `static MatrixSpan<T> Subtract<T>(ReadOnlyMatrixSpan<T> x, T scalar, MatrixSpan<T> destination)` | Subtracts a scalar from every element of a matrix and writes the result to `destination`. / 行列の全要素からスカラーを減算し、`destination` に書き込みます。 |
| `static T Sum<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Computes the sum `Σ x[i]` of the elements in a vector. / ベクトル要素の総和 `Σ x[i]` を計算します。 |
| `static T Sum<T>(ReadOnlyVectorSpan<T> x, bool ignoreNaN)  (+2 overloads)` | Computes the sum of the elements in a vector, optionally excluding `NaN` values. / 必要に応じて `NaN` を除外して、ベクトル要素の総和を計算します。 |
| `static Matrix<T> SumMatrices<T>(ReadOnlyTensorSpan<T> x)  (+2 overloads)` | Sums a tensor over its matrix index axis. / テンソルの行列番号の軸に沿って総和を計算します。 |
| `static MatrixSpan<T> SumMatrices<T>(ReadOnlyTensorSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Sums a tensor over its matrix index axis and writes the result to the destination. / テンソルの行列番号の軸に沿う総和を出力先に書き込みます。 |
| `static T SumOfSquares<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Computes the sum of squares `Σ x[i]²` of the elements in a vector. / ベクトル要素の平方和 `Σ x[i]²` を計算します。 |
| `static T Tanh<T>(T x)` | Computes the hyperbolic tangent `tanh(x)`. / 双曲線正接 `tanh(x)` を計算します。 |
| `static Vector<T> Tanh<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Applies `Tanh` elementwise to the input vector and returns a new vector. / `Tanh` をベクトルの要素ごとに適用し、新しいベクトルを返します。 |
| `static Matrix<T> Tanh<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Applies `Tanh` elementwise to the input matrix and returns a new matrix. / `Tanh` を行列の要素ごとに適用し、新しい行列を返します。 |
| `static T Tanh<T>(T x, bool approximate)` | Computes the hyperbolic tangent `tanh(x)`, optionally using a fast approximation for `Single`. / 双曲線正接 `tanh(x)` を計算します（`Single` では高速な近似式を選択可能）。 |
| `static Vector<T> Tanh<T>(ReadOnlyVectorSpan<T> x, bool approximate)  (+2 overloads)` | Computes the hyperbolic tangent `tanh(x)` elementwise for a vector, optionally using a fast approximation for `Single`. / ベクトルの各要素の双曲線正接 `tanh(x)` を計算します（`Single` では高速な近似式を選択可能）。 |
| `static Matrix<T> Tanh<T>(ReadOnlyMatrixSpan<T> x, bool approximate)  (+2 overloads)` | Computes the hyperbolic tangent `tanh(x)` elementwise for a matrix, optionally using a fast approximation for `Single`. / 行列の各要素の双曲線正接 `tanh(x)` を計算します（`Single` では高速な近似式を選択可能）。 |
| `static VectorSpan<T> Tanh<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Applies `Tanh` elementwise to the input vector, writes the result to `destination`, and returns it. / `Tanh` をベクトルの要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static MatrixSpan<T> Tanh<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Applies `Tanh` elementwise to the input matrix, writes the result to `destination`, and returns it. / `Tanh` を行列の要素ごとに適用し、結果を `destination` に書き込んで返します。 |
| `static VectorSpan<T> Tanh<T>(ReadOnlyVectorSpan<T> x, bool approximate, VectorSpan<T> destination)  (+2 overloads)` | Computes the hyperbolic tangent `tanh(x)` elementwise for a vector and writes the results to `destination`, optionally using a fast approximation for `Single`. / ベクトルの各要素の双曲線正接 `tanh(x)` を計算し、`destination` に書き込みます（`Single` では高速な近似式を選択可能）。 |
| `static MatrixSpan<T> Tanh<T>(ReadOnlyMatrixSpan<T> x, bool approximate, MatrixSpan<T> destination)  (+2 overloads)` | Computes the hyperbolic tangent `tanh(x)` elementwise for a matrix and writes the results to `destination`, optionally using a fast approximation for `Single`. / 行列の各要素の双曲線正接 `tanh(x)` を計算し、`destination` に書き込みます（`Single` では高速な近似式を選択可能）。 |
| `static T Trace<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Computes the sum of the main diagonal of a matrix. / 行列の主対角成分の和を計算します。 |
| `static Tensor<T> Transpose<T>(ReadOnlyTensorSpan<T> x)  (+2 overloads)` | Returns a newly allocated tensor with each matrix slice transposed (`B[i, j, k] = A[j, i, k]`). / 各行列スライスを転置した新しいテンソル（`B[i, j, k] = A[j, i, k]`）を返します。 |
| `static Matrix<T> Transpose<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Returns a newly allocated transposed copy `Aᵀ` of the input matrix. / 入力行列を転置した新しいコピー `Aᵀ` を返します。 |
| `static TensorSpan<T> Transpose<T>(ReadOnlyTensorSpan<T> x, TensorSpan<T> destination)  (+2 overloads)` | Writes the slice-wise transpose `A[:, :, k]ᵀ` into `destination`. / 各行列スライスの転置 `A[:, :, k]ᵀ` を `destination` に書き込みます。 |
| `static MatrixSpan<T> Transpose<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Writes the transpose `Aᵀ` of the input matrix into `destination`. / 入力行列の転置 `Aᵀ` を `destination` に書き込みます。 |
| `static T Variance<T>(ReadOnlyVectorSpan<T> x, int correction)  (+2 overloads)` | Computes the variance `Σ(x[i] - mean)² / (n - correction)` of the elements in a vector. / ベクトル要素の分散 `Σ(x[i] - mean)² / (n - correction)` を計算します。 |
| `static Matrix<T> Wedge<T>(ReadOnlyVectorSpan<T> u, ReadOnlyVectorSpan<T> v)  (+2 overloads)` | Computes the wedge product `u vᵀ − v uᵀ` of two vectors of the same length. / 同じ長さの 2 つのベクトルのウェッジ積 `u vᵀ − v uᵀ` を計算します。 |
| `static MatrixSpan<T> Wedge<T>(ReadOnlyVectorSpan<T> u, ReadOnlyVectorSpan<T> v, MatrixSpan<T> destination)  (+2 overloads)` | Writes the wedge product `u vᵀ − v uᵀ` of two vectors to `destination`. / 2 つのベクトルのウェッジ積 `u vᵀ − v uᵀ` を `destination` に書き込みます。 |
| `static Tensor<Half> Zip<TOperator>(ReadOnlyTensorSpan<Half> x, ReadOnlyTensorSpan<Half> y)  (+3 overloads)` | Applies a stateless binary operator `TOperator` elementwise to two tensors. / 状態を持たない2引数演算子 `TOperator` を2つのテンソルの対応する要素ごとに適用します。 |
| `static Vector<Half> Zip<TOperator>(ReadOnlyVectorSpan<Half> x, ReadOnlyVectorSpan<Half> y)  (+3 overloads)` | Applies a stateless binary operator `TOperator` elementwise to two vectors (`result[i] = TOperator(x[i], y[i])`). / 状態を持たない2引数演算子 `TOperator` を2つのベクトルの対応する要素ごとに適用します（`result[i] = TOperator(x[i], y[i])`）。 |
| `static Matrix<Half> Zip<TOperator>(ReadOnlyMatrixSpan<Half> x, ReadOnlyMatrixSpan<Half> y)  (+3 overloads)` | Applies a stateless binary operator `TOperator` elementwise to two matrices (`result[i, j] = TOperator(x[i, j], y[i, j])`). / 状態を持たない2引数演算子 `TOperator` を2つの行列の対応する要素ごとに適用します（`result[i, j] = TOperator(x[i, j], y[i, j])`）。 |
| `static Matrix<decimal> Zip<TOperator>(ReadOnlyMatrixSpan<decimal> x, ReadOnlyMatrixSpan<decimal> y)` | Applies a stateless binary operator `TOperator` elementwise to two matrices (`result[i, j] = TOperator(x[i, j], y[i, j])`). / 状態を持たない2引数演算子 `TOperator` を2つの行列の対応する要素ごとに適用します（`result[i, j] = TOperator(x[i, j], y[i, j])`）。 |
| `static Matrix<Complex> Zip<TOperator>(ReadOnlyMatrixSpan<Complex> x, ReadOnlyMatrixSpan<Complex> y)` | Applies a stateless binary operator `TOperator` elementwise to two matrices (`result[i, j] = TOperator(x[i, j], y[i, j])`). / 状態を持たない2引数演算子 `TOperator` を2つの行列の対応する要素ごとに適用します（`result[i, j] = TOperator(x[i, j], y[i, j])`）。 |
| `static Tensor<decimal> Zip<TOperator>(ReadOnlyTensorSpan<decimal> x, ReadOnlyTensorSpan<decimal> y)` | Applies a stateless binary operator `TOperator` elementwise to two tensors. / 状態を持たない2引数演算子 `TOperator` を2つのテンソルの対応する要素ごとに適用します。 |
| `static Tensor<Complex> Zip<TOperator>(ReadOnlyTensorSpan<Complex> x, ReadOnlyTensorSpan<Complex> y)` | Applies a stateless binary operator `TOperator` elementwise to two tensors. / 状態を持たない2引数演算子 `TOperator` を2つのテンソルの対応する要素ごとに適用します。 |
| `static Vector<decimal> Zip<TOperator>(ReadOnlyVectorSpan<decimal> x, ReadOnlyVectorSpan<decimal> y)` | Applies a stateless binary operator `TOperator` elementwise to two vectors (`result[i] = TOperator(x[i], y[i])`). / 状態を持たない2引数演算子 `TOperator` を2つのベクトルの対応する要素ごとに適用します（`result[i] = TOperator(x[i], y[i])`）。 |
| `static Vector<Complex> Zip<TOperator>(ReadOnlyVectorSpan<Complex> x, ReadOnlyVectorSpan<Complex> y)` | Applies a stateless binary operator `TOperator` elementwise to two vectors (`result[i] = TOperator(x[i], y[i])`). / 状態を持たない2引数演算子 `TOperator` を2つのベクトルの対応する要素ごとに適用します（`result[i] = TOperator(x[i], y[i])`）。 |
| `static TensorSpan<Half> Zip<TOperator>(ReadOnlyTensorSpan<Half> x, ReadOnlyTensorSpan<Half> y, TensorSpan<Half> destination)  (+3 overloads)` | Applies a stateless binary operator `TOperator` elementwise to two tensors and writes the result to `destination`. / 状態を持たない2引数演算子 `TOperator` を2つのテンソルの対応する要素ごとに適用し、`destination` に書き込みます。 |
| `static Tensor<T> Zip<T, TOperator>(ReadOnlyTensorSpan<T> x, ReadOnlyTensorSpan<T> y, TOperator op)  (+2 overloads)` | Applies a stateful binary operator `op` elementwise to two tensors. / 状態を持つ2引数演算子 `op` を2つのテンソルの対応する要素ごとに適用します。 |
| `static VectorSpan<Half> Zip<TOperator>(ReadOnlyVectorSpan<Half> x, ReadOnlyVectorSpan<Half> y, VectorSpan<Half> destination)  (+3 overloads)` | Applies a stateless binary operator `TOperator` elementwise to two vectors and writes the result to `destination`. / 状態を持たない2引数演算子 `TOperator` を2つのベクトルの対応する要素ごとに適用し、`destination` に書き込みます。 |
| `static MatrixSpan<Half> Zip<TOperator>(ReadOnlyMatrixSpan<Half> x, ReadOnlyMatrixSpan<Half> y, MatrixSpan<Half> destination)  (+3 overloads)` | Applies a stateless binary operator `TOperator` elementwise to two matrices and writes the result to `destination`. / 状態を持たない2引数演算子 `TOperator` を2つの行列の対応する要素ごとに適用し、`destination` に書き込みます。 |
| `static Vector<T> Zip<T, TOperator>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, TOperator op)  (+2 overloads)` | Applies a stateful binary operator `op` elementwise to two vectors (`result[i] = op(x[i], y[i])`). / 状態を持つ2引数演算子 `op` を2つのベクトルの対応する要素ごとに適用します（`result[i] = op(x[i], y[i])`）。 |
| `static Matrix<T> Zip<T, TOperator>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, TOperator op)  (+2 overloads)` | Applies a stateful binary operator `op` elementwise to two matrices (`result[i, j] = op(x[i, j], y[i, j])`). / 状態を持つ2引数演算子 `op` を2つの行列の対応する要素ごとに適用します（`result[i, j] = op(x[i, j], y[i, j])`）。 |
| `static Vector<Half> Zip<TOperator>(ReadOnlyVectorSpan<Half> x, ReadOnlyVectorSpan<Half> y, ReadOnlyVectorSpan<Half> z)  (+3 overloads)` | Applies a stateless ternary operator `TOperator` elementwise to three vectors (`result[i] = TOperator(x[i], y[i], z[i])`). / 状態を持たない3引数演算子 `TOperator` を3つのベクトルの対応する要素ごとに適用します（`result[i] = TOperator(x[i], y[i], z[i])`）。 |
| `static Matrix<Half> Zip<TOperator>(ReadOnlyMatrixSpan<Half> x, ReadOnlyMatrixSpan<Half> y, ReadOnlyMatrixSpan<Half> z)  (+3 overloads)` | Applies a stateless ternary operator `TOperator` elementwise to three matrices (`result[i, j] = TOperator(x[i, j], y[i, j], z[i, j])`). / 状態を持たない3引数演算子 `TOperator` を3つの行列の対応する要素ごとに適用します（`result[i, j] = TOperator(x[i, j], y[i, j], z[i, j])`）。 |
| `static MatrixSpan<decimal> Zip<TOperator>(ReadOnlyMatrixSpan<decimal> x, ReadOnlyMatrixSpan<decimal> y, MatrixSpan<decimal> destination)` | Applies a stateless binary operator `TOperator` elementwise to two matrices and writes the result to `destination`. / 状態を持たない2引数演算子 `TOperator` を2つの行列の対応する要素ごとに適用し、`destination` に書き込みます。 |
| `static MatrixSpan<Complex> Zip<TOperator>(ReadOnlyMatrixSpan<Complex> x, ReadOnlyMatrixSpan<Complex> y, MatrixSpan<Complex> destination)` | Applies a stateless binary operator `TOperator` elementwise to two matrices and writes the result to `destination`. / 状態を持たない2引数演算子 `TOperator` を2つの行列の対応する要素ごとに適用し、`destination` に書き込みます。 |
| `static Matrix<decimal> Zip<TOperator>(ReadOnlyMatrixSpan<decimal> x, ReadOnlyMatrixSpan<decimal> y, ReadOnlyMatrixSpan<decimal> z)` | Applies a stateless ternary operator `TOperator` elementwise to three matrices (`result[i, j] = TOperator(x[i, j], y[i, j], z[i, j])`). / 状態を持たない3引数演算子 `TOperator` を3つの行列の対応する要素ごとに適用します（`result[i, j] = TOperator(x[i, j], y[i, j], z[i, j])`）。 |
| `static Matrix<Complex> Zip<TOperator>(ReadOnlyMatrixSpan<Complex> x, ReadOnlyMatrixSpan<Complex> y, ReadOnlyMatrixSpan<Complex> z)` | Applies a stateless ternary operator `TOperator` elementwise to three matrices (`result[i, j] = TOperator(x[i, j], y[i, j], z[i, j])`). / 状態を持たない3引数演算子 `TOperator` を3つの行列の対応する要素ごとに適用します（`result[i, j] = TOperator(x[i, j], y[i, j], z[i, j])`）。 |
| `static TensorSpan<decimal> Zip<TOperator>(ReadOnlyTensorSpan<decimal> x, ReadOnlyTensorSpan<decimal> y, TensorSpan<decimal> destination)` | Applies a stateless binary operator `TOperator` elementwise to two tensors and writes the result to `destination`. / 状態を持たない2引数演算子 `TOperator` を2つのテンソルの対応する要素ごとに適用し、`destination` に書き込みます。 |
| `static TensorSpan<Complex> Zip<TOperator>(ReadOnlyTensorSpan<Complex> x, ReadOnlyTensorSpan<Complex> y, TensorSpan<Complex> destination)` | Applies a stateless binary operator `TOperator` elementwise to two tensors and writes the result to `destination`. / 状態を持たない2引数演算子 `TOperator` を2つのテンソルの対応する要素ごとに適用し、`destination` に書き込みます。 |
| `static Vector<decimal> Zip<TOperator>(ReadOnlyVectorSpan<decimal> x, ReadOnlyVectorSpan<decimal> y, ReadOnlyVectorSpan<decimal> z)` | Applies a stateless ternary operator `TOperator` elementwise to three vectors (`result[i] = TOperator(x[i], y[i], z[i])`). / 状態を持たない3引数演算子 `TOperator` を3つのベクトルの対応する要素ごとに適用します（`result[i] = TOperator(x[i], y[i], z[i])`）。 |
| `static Vector<Complex> Zip<TOperator>(ReadOnlyVectorSpan<Complex> x, ReadOnlyVectorSpan<Complex> y, ReadOnlyVectorSpan<Complex> z)` | Applies a stateless ternary operator `TOperator` elementwise to three vectors (`result[i] = TOperator(x[i], y[i], z[i])`). / 状態を持たない3引数演算子 `TOperator` を3つのベクトルの対応する要素ごとに適用します（`result[i] = TOperator(x[i], y[i], z[i])`）。 |
| `static VectorSpan<decimal> Zip<TOperator>(ReadOnlyVectorSpan<decimal> x, ReadOnlyVectorSpan<decimal> y, VectorSpan<decimal> destination)` | Applies a stateless binary operator `TOperator` elementwise to two vectors and writes the result to `destination`. / 状態を持たない2引数演算子 `TOperator` を2つのベクトルの対応する要素ごとに適用し、`destination` に書き込みます。 |
| `static VectorSpan<Complex> Zip<TOperator>(ReadOnlyVectorSpan<Complex> x, ReadOnlyVectorSpan<Complex> y, VectorSpan<Complex> destination)` | Applies a stateless binary operator `TOperator` elementwise to two vectors and writes the result to `destination`. / 状態を持たない2引数演算子 `TOperator` を2つのベクトルの対応する要素ごとに適用し、`destination` に書き込みます。 |
| `static TensorSpan<T> Zip<T, TOperator>(ReadOnlyTensorSpan<T> x, ReadOnlyTensorSpan<T> y, TOperator op, TensorSpan<T> destination)  (+2 overloads)` | Applies a stateful binary operator `op` elementwise to two tensors and writes the result to `destination`. / 状態を持つ2引数演算子 `op` を2つのテンソルの対応する要素ごとに適用し、`destination` に書き込みます。 |
| `static VectorSpan<T> Zip<T, TOperator>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, TOperator op, VectorSpan<T> destination)  (+2 overloads)` | Applies a stateful binary operator `op` elementwise to two vectors and writes the result to `destination`. / 状態を持つ2引数演算子 `op` を2つのベクトルの対応する要素ごとに適用し、`destination` に書き込みます。 |
| `static MatrixSpan<T> Zip<T, TOperator>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, TOperator op, MatrixSpan<T> destination)  (+2 overloads)` | Applies a stateful binary operator `op` elementwise to two matrices and writes the result to `destination`. / 状態を持つ2引数演算子 `op` を2つの行列の対応する要素ごとに適用し、`destination` に書き込みます。 |
| `static VectorSpan<Half> Zip<TOperator>(ReadOnlyVectorSpan<Half> x, ReadOnlyVectorSpan<Half> y, ReadOnlyVectorSpan<Half> z, VectorSpan<Half> destination)  (+3 overloads)` | Applies a stateless ternary operator `TOperator` elementwise to three vectors and writes the result to `destination`. / 状態を持たない3引数演算子 `TOperator` を3つのベクトルの対応する要素ごとに適用し、`destination` に書き込みます。 |
| `static MatrixSpan<Half> Zip<TOperator>(ReadOnlyMatrixSpan<Half> x, ReadOnlyMatrixSpan<Half> y, ReadOnlyMatrixSpan<Half> z, MatrixSpan<Half> destination)  (+3 overloads)` | Applies a stateless ternary operator `TOperator` elementwise to three matrices and writes the result to `destination`. / 状態を持たない3引数演算子 `TOperator` を3つの行列の対応する要素ごとに適用し、`destination` に書き込みます。 |
| `static Vector<T> Zip<T, TOperator>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, ReadOnlyVectorSpan<T> z, TOperator op)  (+2 overloads)` | Applies a stateful ternary operator `op` elementwise to three vectors (`result[i] = op(x[i], y[i], z[i])`). / 状態を持つ3引数演算子 `op` を3つのベクトルの対応する要素ごとに適用します（`result[i] = op(x[i], y[i], z[i])`）。 |
| `static Matrix<T> Zip<T, TOperator>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, ReadOnlyMatrixSpan<T> z, TOperator op)  (+2 overloads)` | Applies a stateful ternary operator `op` elementwise to three matrices (`result[i, j] = op(x[i, j], y[i, j], z[i, j])`). / 状態を持つ3引数演算子 `op` を3つの行列の対応する要素ごとに適用します（`result[i, j] = op(x[i, j], y[i, j], z[i, j])`）。 |
| `static MatrixSpan<decimal> Zip<TOperator>(ReadOnlyMatrixSpan<decimal> x, ReadOnlyMatrixSpan<decimal> y, ReadOnlyMatrixSpan<decimal> z, MatrixSpan<decimal> destination)` | Applies a stateless ternary operator `TOperator` elementwise to three matrices and writes the result to `destination`. / 状態を持たない3引数演算子 `TOperator` を3つの行列の対応する要素ごとに適用し、`destination` に書き込みます。 |
| `static MatrixSpan<Complex> Zip<TOperator>(ReadOnlyMatrixSpan<Complex> x, ReadOnlyMatrixSpan<Complex> y, ReadOnlyMatrixSpan<Complex> z, MatrixSpan<Complex> destination)` | Applies a stateless ternary operator `TOperator` elementwise to three matrices and writes the result to `destination`. / 状態を持たない3引数演算子 `TOperator` を3つの行列の対応する要素ごとに適用し、`destination` に書き込みます。 |
| `static VectorSpan<decimal> Zip<TOperator>(ReadOnlyVectorSpan<decimal> x, ReadOnlyVectorSpan<decimal> y, ReadOnlyVectorSpan<decimal> z, VectorSpan<decimal> destination)` | Applies a stateless ternary operator `TOperator` elementwise to three vectors and writes the result to `destination`. / 状態を持たない3引数演算子 `TOperator` を3つのベクトルの対応する要素ごとに適用し、`destination` に書き込みます。 |
| `static VectorSpan<Complex> Zip<TOperator>(ReadOnlyVectorSpan<Complex> x, ReadOnlyVectorSpan<Complex> y, ReadOnlyVectorSpan<Complex> z, VectorSpan<Complex> destination)` | Applies a stateless ternary operator `TOperator` elementwise to three vectors and writes the result to `destination`. / 状態を持たない3引数演算子 `TOperator` を3つのベクトルの対応する要素ごとに適用し、`destination` に書き込みます。 |
| `static VectorSpan<T> Zip<T, TOperator>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, ReadOnlyVectorSpan<T> z, TOperator op, VectorSpan<T> destination)  (+2 overloads)` | Applies a stateful ternary operator `op` elementwise to three vectors and writes the result to `destination`. / 状態を持つ3引数演算子 `op` を3つのベクトルの対応する要素ごとに適用し、`destination` に書き込みます。 |
| `static MatrixSpan<T> Zip<T, TOperator>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, ReadOnlyMatrixSpan<T> z, TOperator op, MatrixSpan<T> destination)  (+2 overloads)` | Applies a stateful ternary operator `op` elementwise to three matrices and writes the result to `destination`. / 状態を持つ3引数演算子 `op` を3つの行列の対応する要素ごとに適用し、`destination` に書き込みます。 |

#### `static class XMath.Constants`

Provides precomputed double-precision values of mathematical constants frequently used in numerical computing. / 数値計算でよく使う数学定数の倍精度の値を、事前計算済みの定数として提供します。

**フィールド・定数**

| メンバー | 説明 |
| --- | --- |
| `const double Catalan` | Catalan's constant `G = β(2)`. / カタラン定数 `G = β(2)`。 |
| `const double DegreesToRadians` | The number of radians in one degree, `π/180`. / 1 度に相当するラジアン `π/180`。 |
| `const double E` | The base of the natural logarithm, `e`. / 自然対数の底 `e`。 |
| `const double EulerGamma` | The Euler–Mascheroni constant `γ`. / オイラー・マスケローニ定数 `γ`。 |
| `const double GoldenRatio` | The golden ratio `φ = (1 + √5)/2`. / 黄金比 `φ = (1 + √5)/2`。 |
| `const double Ln10` | The natural logarithm of 10, `ln 10`. / 10 の自然対数 `ln 10`。 |
| `const double Ln2` | The natural logarithm of 2, `ln 2`. / 2 の自然対数 `ln 2`。 |
| `const double LnPi` | The natural logarithm of `π`, `ln π`. / `π` の自然対数 `ln π`。 |
| `const double LnSqrtPi` | The value of `ln √π = (ln π)/2`. / `ln √π = (ln π)/2` の値。 |
| `const double LnSqrtTwoPi` | The value of `ln √(2π) = ln(2π)/2`. / `ln √(2π) = ln(2π)/2` の値。 |
| `const double LnTwoPi` | The value of `ln(2π)`. / `ln(2π)` の値。 |
| `const double Log10E` | The base-10 logarithm of `e`, `log₁₀ e = 1/ln 10`. / `e` の常用対数 `log₁₀ e = 1/ln 10`。 |
| `const double Log10Of2` | The base-10 logarithm of 2, `log₁₀ 2`. / 2 の常用対数 `log₁₀ 2`。 |
| `const double Log2E` | The base-2 logarithm of `e`, `log₂ e = 1/ln 2`. / `e` の 2 を底とする対数 `log₂ e = 1/ln 2`。 |
| `const double Log2Of10` | The base-2 logarithm of 10, `log₂ 10`. / 10 の 2 を底とする対数 `log₂ 10`。 |
| `const double OneOverPi` | The value of `1/π`. / `1/π` の値。 |
| `const double OneOverSqrt2` | The value of `1/√2 = √2/2`. / `1/√2 = √2/2` の値。 |
| `const double OneOverSqrtPi` | The value of `1/√π`. / `1/√π` の値。 |
| `const double OneOverSqrtTwoPi` | The value of `1/√(2π)`. / `1/√(2π)` の値。 |
| `const double OneOverTwoPi` | The value of `1/(2π)`. / `1/(2π)` の値。 |
| `const double Pi` | The ratio of a circle's circumference to its diameter, `π`. / 円周率 `π`。 |
| `const double PiOver2` | The value of `π/2`. / `π/2` の値。 |
| `const double PiOver4` | The value of `π/4`. / `π/4` の値。 |
| `const double PiSquared` | The value of `π²`. / `π²` の値。 |
| `const double PiSquaredOver6` | The value of `π²/6 = ζ(2)`. / `π²/6 = ζ(2)` の値。 |
| `const double RadiansToDegrees` | The number of degrees in one radian, `180/π`. / 1 ラジアンに相当する度数 `180/π`。 |
| `const double Sqrt2` | The square root of 2, `√2`. / 2 の平方根 `√2`。 |
| `const double Sqrt3` | The square root of 3, `√3`. / 3 の平方根 `√3`。 |
| `const double SqrtPi` | The value of `√π`. / `√π` の値。 |
| `const double SqrtPiOver2` | The value of `√(π/2)`. / `√(π/2)` の値。 |
| `const double SqrtTwoOverPi` | The value of `√(2/π)`. / `√(2/π)` の値。 |
| `const double SqrtTwoPi` | The value of `√(2π)`. / `√(2π)` の値。 |
| `const double Tau` | The ratio of a circle's circumference to its radius, `τ = 2π`. / 円周と半径の比 `τ = 2π`。 |
| `const double TwoOverPi` | The value of `2/π`. / `2/π` の値。 |
| `const double TwoOverSqrtPi` | The value of `2/√π`. / `2/√π` の値。 |
| `const double Zeta3` | Apéry's constant `ζ(3)`. / アペリーの定数 `ζ(3)`。 |

#### `static class XMath.Constants<T>`

Provides mathematical constants rounded directly to the floating-point type `T`. / 数学定数を浮動小数点型 `T` へ直接丸めた値を提供します。

**フィールド・定数**

| メンバー | 説明 |
| --- | --- |
| `static T Catalan` | Catalan's constant `G = β(2)`. / カタラン定数 `G = β(2)`。 |
| `static T DegreesToRadians` | The number of radians in one degree, `π/180`. / 1 度に相当するラジアン `π/180`。 |
| `static T E` | The base of the natural logarithm, `e`. / 自然対数の底 `e`。 |
| `static T EulerGamma` | The Euler–Mascheroni constant `γ`. / オイラー・マスケローニ定数 `γ`。 |
| `static T GoldenRatio` | The golden ratio `φ = (1 + √5)/2`. / 黄金比 `φ = (1 + √5)/2`。 |
| `static T Ln10` | The natural logarithm of 10, `ln 10`. / 10 の自然対数 `ln 10`。 |
| `static T Ln2` | The natural logarithm of 2, `ln 2`. / 2 の自然対数 `ln 2`。 |
| `static T LnPi` | The natural logarithm of `π`, `ln π`. / `π` の自然対数 `ln π`。 |
| `static T LnSqrtPi` | The value of `ln √π = (ln π)/2`. / `ln √π = (ln π)/2` の値。 |
| `static T LnSqrtTwoPi` | The value of `ln √(2π) = ln(2π)/2`. / `ln √(2π) = ln(2π)/2` の値。 |
| `static T LnTwoPi` | The value of `ln(2π)`. / `ln(2π)` の値。 |
| `static T Log10E` | The base-10 logarithm of `e`, `log₁₀ e = 1/ln 10`. / `e` の常用対数 `log₁₀ e = 1/ln 10`。 |
| `static T Log10Of2` | The base-10 logarithm of 2, `log₁₀ 2`. / 2 の常用対数 `log₁₀ 2`。 |
| `static T Log2E` | The base-2 logarithm of `e`, `log₂ e = 1/ln 2`. / `e` の 2 を底とする対数 `log₂ e = 1/ln 2`。 |
| `static T Log2Of10` | The base-2 logarithm of 10, `log₂ 10`. / 10 の 2 を底とする対数 `log₂ 10`。 |
| `static T OneOverPi` | The value of `1/π`. / `1/π` の値。 |
| `static T OneOverSqrt2` | The value of `1/√2 = √2/2`. / `1/√2 = √2/2` の値。 |
| `static T OneOverSqrtPi` | The value of `1/√π`. / `1/√π` の値。 |
| `static T OneOverSqrtTwoPi` | The value of `1/√(2π)`. / `1/√(2π)` の値。 |
| `static T OneOverTwoPi` | The value of `1/(2π)`. / `1/(2π)` の値。 |
| `static T Pi` | The ratio of a circle's circumference to its diameter, `π`. / 円周率 `π`。 |
| `static T PiOver2` | The value of `π/2`. / `π/2` の値。 |
| `static T PiOver4` | The value of `π/4`. / `π/4` の値。 |
| `static T PiSquared` | The value of `π²`. / `π²` の値。 |
| `static T PiSquaredOver6` | The value of `π²/6 = ζ(2)`. / `π²/6 = ζ(2)` の値。 |
| `static T RadiansToDegrees` | The number of degrees in one radian, `180/π`. / 1 ラジアンに相当する度数 `180/π`。 |
| `static T Sqrt2` | The square root of 2, `√2`. / 2 の平方根 `√2`。 |
| `static T Sqrt3` | The square root of 3, `√3`. / 3 の平方根 `√3`。 |
| `static T SqrtPi` | The value of `√π`. / `√π` の値。 |
| `static T SqrtPiOver2` | The value of `√(π/2)`. / `√(π/2)` の値。 |
| `static T SqrtTwoOverPi` | The value of `√(2/π)`. / `√(2/π)` の値。 |
| `static T SqrtTwoPi` | The value of `√(2π)`. / `√(2π)` の値。 |
| `static T Tau` | The ratio of a circle's circumference to its radius, `τ = 2π`. / 円周と半径の比 `τ = 2π`。 |
| `static T TwoOverPi` | The value of `2/π`. / `2/π` の値。 |
| `static T TwoOverSqrtPi` | The value of `2/√π`. / `2/√π` の値。 |
| `static T Zeta3` | Apéry's constant `ζ(3)`. / アペリーの定数 `ζ(3)`。 |

#### `static class XMath.InPlace`

Provides in-place operations that overwrite their input buffer. / 入力バッファをインプレースで上書きする演算を提供します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static void Abs<T>(VectorSpan<T> x)  (+1 overloads)` | Applies `Abs` elementwise in place, overwriting `x`. / `Abs` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Abs<T>(MatrixSpan<T> x)  (+1 overloads)` | Applies `Abs` elementwise in place, overwriting `x`. / `Abs` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Add<T>(VectorSpan<T> x, T scalar)` | Adds a scalar to every element of a vector in place (`x ← x + scalar`). / ベクトルの全要素にスカラーをインプレースで加算します（`x ← x + scalar`）。 |
| `static void Add<T>(MatrixSpan<T> x, T scalar)` | Adds a scalar to every element of a matrix in place (`x ← x + scalar`). / 行列の全要素にスカラーをインプレースで加算します（`x ← x + scalar`）。 |
| `static void Clamp<T>(VectorSpan<T> x, T min, T max)  (+1 overloads)` | Applies `Clamp` elementwise in place, overwriting `x`. / `Clamp` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Clamp<T>(MatrixSpan<T> x, T min, T max)  (+1 overloads)` | Applies `Clamp` elementwise in place, overwriting `x`. / `Clamp` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void ColumnLogSoftmax<T>(MatrixSpan<T> x)` | Computes the column-wise log-softmax of a matrix in place. / 行列の各列ごとの対数ソフトマックスをインプレースで計算します。 |
| `static void ColumnLogSoftmax<T>(Matrix<T> x)` | Computes the column-wise log-softmax of a matrix in place. / 行列の各列ごとの対数ソフトマックスをインプレースで計算します。 |
| `static void ColumnSoftmax<T>(MatrixSpan<T> x)` | Computes the column-wise softmax distribution of a matrix in place. / 行列の各列ごとのソフトマックス分布をインプレースで計算します。 |
| `static void ColumnSoftmax<T>(Matrix<T> x)` | Computes the column-wise softmax distribution of a matrix in place. / 行列の各列ごとのソフトマックス分布をインプレースで計算します。 |
| `static void Divide<T>(TensorSpan<T> x, T scalar)  (+1 overloads)` | Divides each element of a tensor by a scalar in place. / テンソルの各要素をインプレースでスカラー除算します。 |
| `static void Divide<T>(VectorSpan<T> x, T y)  (+1 overloads)` | Applies `Divide` elementwise in place, overwriting `x`. / `Divide` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Divide<T>(MatrixSpan<T> x, T y)  (+1 overloads)` | Applies `Divide` elementwise in place, overwriting `x`. / `Divide` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Elu<T>(VectorSpan<T> x, T alpha)  (+1 overloads)` | Applies `Elu` elementwise in place, overwriting `x`. / `Elu` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Elu<T>(MatrixSpan<T> x, T alpha)  (+1 overloads)` | Applies `Elu` elementwise in place, overwriting `x`. / `Elu` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Exp<T>(VectorSpan<T> x)  (+1 overloads)` | Computes the elementwise exponential of a vector in place (`x ← exp(x)`). / ベクトルの要素ごとの指数関数をインプレースで計算します（`x ← exp(x)`）。 |
| `static void Exp<T>(MatrixSpan<T> x)  (+1 overloads)` | Computes the elementwise exponential of a matrix in place (`x ← exp(x)`). / 行列の要素ごとの指数関数をインプレースで計算します（`x ← exp(x)`）。 |
| `static void Exp<T>(VectorSpan<T> x, bool approximate)  (+1 overloads)` | Computes the elementwise exponential `x ← exp(x)` of a vector in place, optionally using a fast approximation for `Single`. / ベクトルの要素ごとの指数関数 `x ← exp(x)` をインプレースで計算します（`Single` では高速な近似式を選択可能）。 |
| `static void Exp<T>(MatrixSpan<T> x, bool approximate)  (+1 overloads)` | Computes the elementwise exponential `x ← exp(x)` of a matrix in place, optionally using a fast approximation for `Single`. / 行列の要素ごとの指数関数 `x ← exp(x)` をインプレースで計算します（`Single` では高速な近似式を選択可能）。 |
| `static void Expm1<T>(VectorSpan<T> x)  (+1 overloads)` | Applies `Expm1` elementwise in place, overwriting `x`. / `Expm1` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Expm1<T>(MatrixSpan<T> x)  (+1 overloads)` | Applies `Expm1` elementwise in place, overwriting `x`. / `Expm1` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Gelu<T>(VectorSpan<T> x)  (+1 overloads)` | Applies `Gelu` elementwise in place, overwriting `x`. / `Gelu` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Gelu<T>(MatrixSpan<T> x)  (+1 overloads)` | Applies `Gelu` elementwise in place, overwriting `x`. / `Gelu` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Gelu<T>(VectorSpan<T> x, bool approximate)  (+1 overloads)` | Applies the GELU function elementwise to a vector in place, optionally using the `tanh`-based approximation. / ベクトルの各要素に GELU 関数をインプレースで適用します（`tanh` による近似式を選択可能）。 |
| `static void Gelu<T>(MatrixSpan<T> x, bool approximate)  (+1 overloads)` | Applies the GELU function elementwise to a matrix in place, optionally using the `tanh`-based approximation. / 行列の各要素に GELU 関数をインプレースで適用します（`tanh` による近似式を選択可能）。 |
| `static void Hypot<T>(VectorSpan<T> x, T y)  (+1 overloads)` | Applies `Hypot` elementwise in place, overwriting `x`. / `Hypot` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Hypot<T>(MatrixSpan<T> x, T y)  (+1 overloads)` | Applies `Hypot` elementwise in place, overwriting `x`. / `Hypot` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void LeakyRelu<T>(VectorSpan<T> x, T slope)  (+1 overloads)` | Applies `LeakyRelu` elementwise in place, overwriting `x`. / `LeakyRelu` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void LeakyRelu<T>(MatrixSpan<T> x, T slope)  (+1 overloads)` | Applies `LeakyRelu` elementwise in place, overwriting `x`. / `LeakyRelu` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Log<T>(VectorSpan<T> x)  (+1 overloads)` | Applies `Log` elementwise in place, overwriting `x`. / `Log` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Log<T>(MatrixSpan<T> x)  (+1 overloads)` | Applies `Log` elementwise in place, overwriting `x`. / `Log` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Log1mExp<T>(VectorSpan<T> x)  (+1 overloads)` | Applies `Log1mExp` elementwise in place, overwriting `x`. / `Log1mExp` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Log1mExp<T>(MatrixSpan<T> x)  (+1 overloads)` | Applies `Log1mExp` elementwise in place, overwriting `x`. / `Log1mExp` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Log1p<T>(VectorSpan<T> x)  (+1 overloads)` | Applies `Log1p` elementwise in place, overwriting `x`. / `Log1p` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Log1p<T>(MatrixSpan<T> x)  (+1 overloads)` | Applies `Log1p` elementwise in place, overwriting `x`. / `Log1p` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Log1pmx<T>(VectorSpan<T> x)  (+1 overloads)` | Applies `Log1pmx` elementwise in place, overwriting `x`. / `Log1pmx` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Log1pmx<T>(MatrixSpan<T> x)  (+1 overloads)` | Applies `Log1pmx` elementwise in place, overwriting `x`. / `Log1pmx` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void LogAddExp<T>(VectorSpan<T> x, T y)  (+1 overloads)` | Applies `LogAddExp` elementwise in place, overwriting `x`. / `LogAddExp` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void LogAddExp<T>(MatrixSpan<T> x, T y)  (+1 overloads)` | Applies `LogAddExp` elementwise in place, overwriting `x`. / `LogAddExp` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void LogLogistic<T>(VectorSpan<T> x)  (+1 overloads)` | Applies `LogLogistic` elementwise in place, overwriting `x`. / `LogLogistic` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void LogLogistic<T>(MatrixSpan<T> x)  (+1 overloads)` | Applies `LogLogistic` elementwise in place, overwriting `x`. / `LogLogistic` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void LogSoftmax<T>(VectorSpan<T> x)` | Computes the log-softmax `x ← x - LogSumExp(x)` over a vector in place. / ベクトル全体に対する対数ソフトマックスをインプレースで計算します（`x ← x - LogSumExp(x)`）。 |
| `static void LogSoftmax<T>(Vector<T> x)` | Computes the log-softmax `x ← x - LogSumExp(x)` over a vector in place. / ベクトル全体に対する対数ソフトマックスをインプレースで計算します（`x ← x - LogSumExp(x)`）。 |
| `static void LogSubExp<T>(VectorSpan<T> x, T y)  (+1 overloads)` | Applies `LogSubExp` elementwise in place, overwriting `x`. / `LogSubExp` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void LogSubExp<T>(MatrixSpan<T> x, T y)  (+1 overloads)` | Applies `LogSubExp` elementwise in place, overwriting `x`. / `LogSubExp` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Logistic<T>(VectorSpan<T> x)  (+1 overloads)` | Applies `Logistic` elementwise in place, overwriting `x`. / `Logistic` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Logistic<T>(MatrixSpan<T> x)  (+1 overloads)` | Applies `Logistic` elementwise in place, overwriting `x`. / `Logistic` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Logistic<T>(VectorSpan<T> x, bool approximate)  (+1 overloads)` | Applies the logistic sigmoid function `x ← σ(x)` elementwise to a vector in place, optionally using a fast approximation for `Single`. / ベクトルの各要素にロジスティック関数 `x ← σ(x)` をインプレースで適用します（`Single` では高速な近似式を選択可能）。 |
| `static void Logistic<T>(MatrixSpan<T> x, bool approximate)  (+1 overloads)` | Applies the logistic sigmoid function `x ← σ(x)` elementwise to a matrix in place, optionally using a fast approximation for `Single`. / 行列の各要素にロジスティック関数 `x ← σ(x)` をインプレースで適用します（`Single` では高速な近似式を選択可能）。 |
| `static void Logit<T>(VectorSpan<T> p)  (+1 overloads)` | Applies `Logit` elementwise in place, overwriting `p`. / `Logit` を要素ごとにインプレースで適用し、`p` を上書きします。 |
| `static void Logit<T>(MatrixSpan<T> p)  (+1 overloads)` | Applies `Logit` elementwise in place, overwriting `p`. / `Logit` を要素ごとにインプレースで適用し、`p` を上書きします。 |
| `static void Map<TOperator>(TensorSpan<Half> x)  (+3 overloads)` | Applies a stateless unary operator elementwise to a tensor in place (`x ← TOperator(x)`). / 状態を持たない1引数演算子をテンソルの各要素にインプレースで適用します（`x ← TOperator(x)`）。 |
| `static void Map<TOperator>(VectorSpan<Half> x)  (+3 overloads)` | Applies a stateless unary operator elementwise to a vector in place (`x ← TOperator(x)`). / 状態を持たない1引数演算子をベクトルの各要素にインプレースで適用します（`x ← TOperator(x)`）。 |
| `static void Map<TOperator>(MatrixSpan<Half> x)  (+3 overloads)` | Applies a stateless unary operator elementwise to a matrix in place (`x ← TOperator(x)`). / 状態を持たない1引数演算子を行列の各要素にインプレースで適用します（`x ← TOperator(x)`）。 |
| `static void Map<TOperator>(MatrixSpan<decimal> x)` | Applies a stateless unary operator elementwise to a matrix in place (`x ← TOperator(x)`). / 状態を持たない1引数演算子を行列の各要素にインプレースで適用します（`x ← TOperator(x)`）。 |
| `static void Map<TOperator>(MatrixSpan<Complex> x)` | Applies a stateless unary operator elementwise to a matrix in place (`x ← TOperator(x)`). / 状態を持たない1引数演算子を行列の各要素にインプレースで適用します（`x ← TOperator(x)`）。 |
| `static void Map<TOperator>(TensorSpan<decimal> x)` | Applies a stateless unary operator elementwise to a tensor in place (`x ← TOperator(x)`). / 状態を持たない1引数演算子をテンソルの各要素にインプレースで適用します（`x ← TOperator(x)`）。 |
| `static void Map<TOperator>(TensorSpan<Complex> x)` | Applies a stateless unary operator elementwise to a tensor in place (`x ← TOperator(x)`). / 状態を持たない1引数演算子をテンソルの各要素にインプレースで適用します（`x ← TOperator(x)`）。 |
| `static void Map<TOperator>(VectorSpan<decimal> x)` | Applies a stateless unary operator elementwise to a vector in place (`x ← TOperator(x)`). / 状態を持たない1引数演算子をベクトルの各要素にインプレースで適用します（`x ← TOperator(x)`）。 |
| `static void Map<TOperator>(VectorSpan<Complex> x)` | Applies a stateless unary operator elementwise to a vector in place (`x ← TOperator(x)`). / 状態を持たない1引数演算子をベクトルの各要素にインプレースで適用します（`x ← TOperator(x)`）。 |
| `static void Map<T, TOperator>(TensorSpan<T> x, TOperator op)  (+1 overloads)` | Applies a stateful unary operator elementwise to a tensor in place (`x ← op(x)`). / 状態を持つ1引数演算子をテンソルの各要素にインプレースで適用します（`x ← op(x)`）。 |
| `static void Map<T, TOperator>(VectorSpan<T> x, TOperator op)  (+1 overloads)` | Applies a stateful unary operator elementwise to a vector in place (`x ← op(x)`). / 状態を持つ1引数演算子をベクトルの各要素にインプレースで適用します（`x ← op(x)`）。 |
| `static void Map<T, TOperator>(MatrixSpan<T> x, TOperator op)  (+1 overloads)` | Applies a stateful unary operator elementwise to a matrix in place (`x ← op(x)`). / 状態を持つ1引数演算子を行列の各要素にインプレースで適用します（`x ← op(x)`）。 |
| `static void Max<T>(VectorSpan<T> x, T y)  (+1 overloads)` | Applies `Max` elementwise in place, overwriting `x`. / `Max` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Max<T>(MatrixSpan<T> x, T y)  (+1 overloads)` | Applies `Max` elementwise in place, overwriting `x`. / `Max` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Min<T>(VectorSpan<T> x, T y)  (+1 overloads)` | Applies `Min` elementwise in place, overwriting `x`. / `Min` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Min<T>(MatrixSpan<T> x, T y)  (+1 overloads)` | Applies `Min` elementwise in place, overwriting `x`. / `Min` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Multiply<T>(TensorSpan<T> x, T scalar)` | Multiplies every element of a tensor by a scalar in place (`X[k] ← X[k] · scalar`). / テンソルの全要素をインプレースでスカラー倍します（`X[k] ← X[k] · scalar`）。 |
| `static void Multiply<T>(VectorSpan<T> x, T scalar)` | Multiplies every element of a vector by a scalar in place (`x ← x · scalar`). / ベクトルの全要素をインプレースでスカラー倍します（`x ← x · scalar`）。 |
| `static void Multiply<T>(MatrixSpan<T> x, T scalar)` | Multiplies every element of a matrix by a scalar in place (`x ← x · scalar`). / 行列の全要素をインプレースでスカラー倍します（`x ← x · scalar`）。 |
| `static void Pow<T>(VectorSpan<T> x, T y)  (+1 overloads)` | Applies `Pow` elementwise in place, overwriting `x`. / `Pow` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Pow<T>(MatrixSpan<T> x, T y)  (+1 overloads)` | Applies `Pow` elementwise in place, overwriting `x`. / `Pow` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Reciprocal<T>(VectorSpan<T> x)  (+1 overloads)` | Applies `Reciprocal` elementwise in place, overwriting `x`. / `Reciprocal` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Reciprocal<T>(MatrixSpan<T> x)  (+1 overloads)` | Applies `Reciprocal` elementwise in place, overwriting `x`. / `Reciprocal` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Relu<T>(VectorSpan<T> x)  (+1 overloads)` | Applies `Relu` elementwise in place, overwriting `x`. / `Relu` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Relu<T>(MatrixSpan<T> x)  (+1 overloads)` | Applies `Relu` elementwise in place, overwriting `x`. / `Relu` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void RowLogSoftmax<T>(MatrixSpan<T> x)` | Computes the row-wise log-softmax of a matrix in place. / 行列の各行ごとの対数ソフトマックスをインプレースで計算します。 |
| `static void RowLogSoftmax<T>(Matrix<T> x)` | Computes the row-wise log-softmax of a matrix in place. / 行列の各行ごとの対数ソフトマックスをインプレースで計算します。 |
| `static void RowSoftmax<T>(MatrixSpan<T> x)` | Computes the row-wise softmax distribution of a matrix in place. / 行列の各行ごとのソフトマックス分布をインプレースで計算します。 |
| `static void RowSoftmax<T>(Matrix<T> x)` | Computes the row-wise softmax distribution of a matrix in place. / 行列の各行ごとのソフトマックス分布をインプレースで計算します。 |
| `static void Selu<T>(VectorSpan<T> x)  (+1 overloads)` | Applies `Selu` elementwise in place, overwriting `x`. / `Selu` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Selu<T>(MatrixSpan<T> x)  (+1 overloads)` | Applies `Selu` elementwise in place, overwriting `x`. / `Selu` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Silu<T>(VectorSpan<T> x)  (+1 overloads)` | Applies `Silu` elementwise in place, overwriting `x`. / `Silu` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Silu<T>(MatrixSpan<T> x)  (+1 overloads)` | Applies `Silu` elementwise in place, overwriting `x`. / `Silu` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Silu<T>(VectorSpan<T> x, bool approximate)  (+1 overloads)` | Applies the SiLU function `x ← x · σ(x)` elementwise to a vector in place, optionally using a fast approximation for `Single`. / ベクトルの各要素に SiLU 関数 `x ← x · σ(x)` をインプレースで適用します（`Single` では高速な近似式を選択可能）。 |
| `static void Silu<T>(MatrixSpan<T> x, bool approximate)  (+1 overloads)` | Applies the SiLU function `x ← x · σ(x)` elementwise to a matrix in place, optionally using a fast approximation for `Single`. / 行列の各要素に SiLU 関数 `x ← x · σ(x)` をインプレースで適用します（`Single` では高速な近似式を選択可能）。 |
| `static void SoftThreshold<T>(VectorSpan<T> x, T threshold)` | Applies elementwise soft thresholding to a vector in place (`x ← sign(x) · max(\|x\| - threshold, 0)`). / ベクトルに対して要素ごとの軟判定しきい値をインプレースで適用します（`x ← sign(x) · max(\|x\| - threshold, 0)`）。 |
| `static void SoftThreshold<T>(MatrixSpan<T> x, T threshold)` | Applies elementwise soft thresholding to a matrix in place (`x ← sign(x) · max(\|x\| - threshold, 0)`). / 行列に対して要素ごとの軟判定しきい値をインプレースで適用します（`x ← sign(x) · max(\|x\| - threshold, 0)`）。 |
| `static void Softmax<T>(VectorSpan<T> x)` | Computes the softmax distribution over a vector in place (`x ← softmax(x)`). / ベクトル全体に対するソフトマックス分布をインプレースで計算します（`x ← softmax(x)`）。 |
| `static void Softmax<T>(Vector<T> x)` | Computes the softmax distribution over a vector in place (`x ← softmax(x)`). / ベクトル全体に対するソフトマックス分布をインプレースで計算します（`x ← softmax(x)`）。 |
| `static void Softplus<T>(VectorSpan<T> x)  (+1 overloads)` | Applies `Softplus` elementwise in place, overwriting `x`. / `Softplus` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Softplus<T>(MatrixSpan<T> x)  (+1 overloads)` | Applies `Softplus` elementwise in place, overwriting `x`. / `Softplus` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Softplus<T>(VectorSpan<T> x, bool approximate)  (+1 overloads)` | Applies the softplus function `x ← ln(1 + eˣ)` elementwise to a vector in place, optionally using a fast approximation for `Single`. / ベクトルの各要素に Softplus 関数 `x ← ln(1 + eˣ)` をインプレースで適用します（`Single` では高速な近似式を選択可能）。 |
| `static void Softplus<T>(MatrixSpan<T> x, bool approximate)  (+1 overloads)` | Applies the softplus function `x ← ln(1 + eˣ)` elementwise to a matrix in place, optionally using a fast approximation for `Single`. / 行列の各要素に Softplus 関数 `x ← ln(1 + eˣ)` をインプレースで適用します（`Single` では高速な近似式を選択可能）。 |
| `static void Sort<T>(VectorSpan<T> x)` | Sorts the elements of `x` in ascending order in place. / `x` の要素をインプレースで昇順に並べ替えます。 |
| `static void Sort<T>(Vector<T> x)` | Sorts the elements of `x` in ascending order in place. / `x` の要素をインプレースで昇順に並べ替えます。 |
| `static void Sqrt<T>(VectorSpan<T> x)  (+1 overloads)` | Applies `Sqrt` elementwise in place, overwriting `x`. / `Sqrt` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Sqrt<T>(MatrixSpan<T> x)  (+1 overloads)` | Applies `Sqrt` elementwise in place, overwriting `x`. / `Sqrt` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Square<T>(VectorSpan<T> x)  (+1 overloads)` | Applies `Square` elementwise in place, overwriting `x`. / `Square` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Square<T>(MatrixSpan<T> x)  (+1 overloads)` | Applies `Square` elementwise in place, overwriting `x`. / `Square` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Subtract<T>(VectorSpan<T> x, T scalar)` | Subtracts a scalar from every element of a vector in place (`x ← x - scalar`). / ベクトルの全要素からスカラーをインプレースで減算します（`x ← x - scalar`）。 |
| `static void Subtract<T>(MatrixSpan<T> x, T scalar)` | Subtracts a scalar from every element of a matrix in place (`x ← x - scalar`). / 行列の全要素からスカラーをインプレースで減算します（`x ← x - scalar`）。 |
| `static void Tanh<T>(VectorSpan<T> x)  (+1 overloads)` | Applies `Tanh` elementwise in place, overwriting `x`. / `Tanh` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Tanh<T>(MatrixSpan<T> x)  (+1 overloads)` | Applies `Tanh` elementwise in place, overwriting `x`. / `Tanh` を要素ごとにインプレースで適用し、`x` を上書きします。 |
| `static void Tanh<T>(VectorSpan<T> x, bool approximate)  (+1 overloads)` | Applies the hyperbolic tangent `x ← tanh(x)` elementwise to a vector in place, optionally using a fast approximation for `Single`. / ベクトルの各要素に双曲線正接 `x ← tanh(x)` をインプレースで適用します（`Single` では高速な近似式を選択可能）。 |
| `static void Tanh<T>(MatrixSpan<T> x, bool approximate)  (+1 overloads)` | Applies the hyperbolic tangent `x ← tanh(x)` elementwise to a matrix in place, optionally using a fast approximation for `Single`. / 行列の各要素に双曲線正接 `x ← tanh(x)` をインプレースで適用します（`Single` では高速な近似式を選択可能）。 |

#### `static class XMath.Operators`

`XMath.Map` / `XMath.Zip` / `XMath.Pairwise` に渡せる生成済み演算子構造体（38 型）:

`XMath.Operators.Abs<T>`, `XMath.Operators.Clamp<T>`, `XMath.Operators.Distance<T>`, `XMath.Operators.Divide<T>`, `XMath.Operators.DivideScalar<T>`, `XMath.Operators.Elu<T>`, `XMath.Operators.Expm1<T>`, `XMath.Operators.Gelu<T>`, `XMath.Operators.Hypot<T>`, `XMath.Operators.HypotScalar<T>`, `XMath.Operators.LeakyRelu<T>`, `XMath.Operators.Log1mExp<T>`, `XMath.Operators.Log1p<T>`, `XMath.Operators.Log1pmx<T>`, `XMath.Operators.Log<T>`, `XMath.Operators.LogAddExp<T>`, `XMath.Operators.LogAddExpScalar<T>`, `XMath.Operators.LogLogistic<T>`, `XMath.Operators.LogSubExp<T>`, `XMath.Operators.LogSubExpScalar<T>`, `XMath.Operators.Logistic<T>`, `XMath.Operators.Logit<T>`, `XMath.Operators.Max<T>`, `XMath.Operators.MaxScalar<T>`, `XMath.Operators.Min<T>`, `XMath.Operators.MinScalar<T>`, `XMath.Operators.Pow<T>`, `XMath.Operators.PowScalar<T>`, `XMath.Operators.Reciprocal<T>`, `XMath.Operators.Relu<T>`, `XMath.Operators.ScaledSquaredDistance<T>`, `XMath.Operators.Selu<T>`, `XMath.Operators.Silu<T>`, `XMath.Operators.Softplus<T>`, `XMath.Operators.Sqrt<T>`, `XMath.Operators.Square<T>`, `XMath.Operators.SquaredDistance<T>`, `XMath.Operators.Tanh<T>`

#### `static class XMath.Special`

Provides real and complex special mathematical functions, including gamma, beta, Bessel, and hypergeometric functions. / ガンマ関数、ベータ関数、ベッセル関数、超幾何関数などの実数および複素数の特殊関数を提供します。

**フィールド・定数**

| メンバー | 説明 |
| --- | --- |
| `const double EulerGamma` | The Euler–Mascheroni constant `γ`. / オイラー・マスケローニ定数 `γ`。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static (double, double, double, double) Airy(double x)` | Simultaneously computes the Airy functions `Ai(x)` and `Bi(x)` together with their derivatives `Ai'(x)` and `Bi'(x)`. / エアリー関数 `Ai(x)`、`Bi(x)` とその導関数 `Ai'(x)`、`Bi'(x)` を同時に計算します。 |
| `static double AiryAi(double x)` | Computes the Airy function of the first kind, `Ai(x)`. / 第 1 種エアリー関数 `Ai(x)` を計算します。 |
| `static double AiryAiPrime(double x)` | Computes the derivative of the Airy function of the first kind, `Ai'(x)`. / 第 1 種エアリー関数の導関数 `Ai'(x)` を計算します。 |
| `static double AiryAiPrimeZero(int k)` | Computes the `k`-th zero `a'ₖ` of the derivative of the Airy function `Ai'(x)`. / エアリー関数の導関数 `Ai'(x)` の第 `k` 零点 `a'ₖ` を計算します。 |
| `static double AiryAiZero(int k)` | Computes the `k`-th zero `aₖ` of the Airy function `Ai(x)`. / エアリー関数 `Ai(x)` の第 `k` 零点 `aₖ` を計算します。 |
| `static double AiryBi(double x)` | Computes the Airy function of the second kind, `Bi(x)`. / 第 2 種エアリー関数 `Bi(x)` を計算します。 |
| `static double AiryBiPrime(double x)` | Computes the derivative of the Airy function of the second kind, `Bi'(x)`. / 第 2 種エアリー関数の導関数 `Bi'(x)` を計算します。 |
| `static double AiryBiPrimeZero(int k)` | Computes the `k`-th negative real zero `b'ₖ` of the derivative of the Airy function of the second kind `Bi'(x)`. / 第 2 種エアリー関数の導関数 `Bi'(x)` の負の実軸上の第 `k` 零点 `b'ₖ` を計算します。 |
| `static double AiryBiZero(int k)` | Computes the `k`-th negative real zero `bₖ` of the Airy function of the second kind `Bi(x)`. / 第 2 種エアリー関数 `Bi(x)` の負の実軸上の第 `k` 零点 `bₖ` を計算します。 |
| `static double ArithmeticGeometricMean(double a, double b)` | Computes the arithmetic-geometric mean `AGM(a, b)`. / 算術幾何平均 `AGM(a, b)` を計算します。 |
| `static double AssociatedLegendreP(double nu, double mu, double x)` | Computes the associated Legendre function of the first kind (Ferrers function) `P_ν^μ(x)` for real degree `ν` and order `μ`. / 実数次数 `ν` および実数位数 `μ` の第 1 種ルジャンドル陪関数（フェラーズ関数）`P_ν^μ(x)` を計算します。 |
| `static double AssociatedLegendreP(int n, int m, double x)` | Computes the associated Legendre function `Pₙᵐ(x)`. / ルジャンドル陪関数 `Pₙᵐ(x)` を計算します。 |
| `static double AssociatedLegendreQ(int n, int m, double x)` | Computes the associated Legendre function of the second kind (Ferrers function) `Qₙᵐ(x)`. / 第 2 種ルジャンドル陪関数（フェラーズ関数）`Qₙᵐ(x)` を計算します。 |
| `static double AssociatedLegendreQ(double nu, double mu, double x)` | Computes the associated Legendre function of the second kind (Ferrers function) `Q_ν^μ(x)` for real degree `ν` and order `μ`. / 実数次数 `ν` および実数位数 `μ` の第 2 種ルジャンドル陪関数（フェラーズ関数）`Q_ν^μ(x)` を計算します。 |
| `static double BarnesG(double z)` | Computes the Barnes G-function `G(z)`, satisfying `G(1) = 1` and `G(z + 1) = Γ(z) G(z)`. / バーンズの G 関数 `G(z)`（`G(1) = 1`、`G(z + 1) = Γ(z) G(z)`）を計算します。 |
| `static double BellNumber(int n)` | Computes the `n`-th Bell number `Bₙ`, the number of partitions of an `n`-element set. / 第 `n` ベル数 `Bₙ`（`n` 個の要素からなる集合の分割数）を計算します。 |
| `static double BernoulliNumber(int n)` | Computes the `n`-th Bernoulli number `Bₙ` using the convention `B₁ = -1/2`. / 第 `n` ベルヌーイ数 `Bₙ`（`B₁ = -1/2` の定義）を計算します。 |
| `static double BernoulliPolynomial(int n, double x)` | Computes the Bernoulli polynomial `Bₙ(x)`. / ベルヌーイ多項式 `Bₙ(x)` を計算します。 |
| `static double BesselI(double nu, double x)` | Computes the modified Bessel function of the first kind, `I_ν(x)`. / 第 1 種変形ベッセル関数 `I_ν(x)` を計算します。 |
| `static double BesselIRatio(double nu, double x)` | Computes the ratio of consecutive modified Bessel functions of the first kind `I_(ν+1)(x) / I_ν(x)`. / 隣接する次数の第 1 種変形ベッセル関数の比 `I_(ν+1)(x) / I_ν(x)` を計算します。 |
| `static double BesselIScaled(double nu, double x)` | Computes the exponentially scaled modified Bessel function of the first kind, `e^(-\|x\|) I_ν(x)`. / 指数スケーリングした第 1 種変形ベッセル関数 `e^(-\|x\|) I_ν(x)` を計算します。 |
| `static double BesselJ(double nu, double x)` | Computes the Bessel function of the first kind, `J_ν(x)`. / 第 1 種ベッセル関数 `J_ν(x)` を計算します。 |
| `static double BesselJPrimeZero(double nu, int k)` | Computes the `k`-th zero `j'_ν,ₖ` of the derivative of the Bessel function of the first kind `J'_ν(x)`. / 第 1 種ベッセル関数の導関数 `J'_ν(x)` の第 `k` 零点 `j'_ν,ₖ` を計算します。 |
| `static double BesselJZero(double nu, int k)` | Computes the `k`-th positive zero `j_ν,ₖ` of the Bessel function of the first kind `J_ν(x)`. / 第 1 種ベッセル関数 `J_ν(x)` の第 `k` 正零点 `j_ν,ₖ` を計算します。 |
| `static double BesselK(double nu, double x)` | Computes the modified Bessel function of the second kind, `K_ν(x)`. / 第 2 種変形ベッセル関数 `K_ν(x)` を計算します。 |
| `static double BesselKScaled(double nu, double x)` | Computes the exponentially scaled modified Bessel function of the second kind, `eˣ K_ν(x)`. / 指数スケーリングした第 2 種変形ベッセル関数 `eˣ K_ν(x)` を計算します。 |
| `static double BesselY(double nu, double x)` | Computes the Bessel function of the second kind (Neumann function), `Y_ν(x)`. / 第 2 種ベッセル関数（ノイマン関数）`Y_ν(x)` を計算します。 |
| `static double BesselYPrimeZero(double nu, int k)` | Computes the `k`-th positive zero `y'_ν,ₖ` of the derivative of the Bessel function of the second kind `Y'_ν(x)`. / 第 2 種ベッセル関数の導関数 `Y'_ν(x)` の第 `k` 正零点 `y'_ν,ₖ` を計算します。 |
| `static double BesselYZero(double nu, int k)` | Computes the `k`-th positive zero `y_ν,ₖ` of the Bessel function of the second kind `Y_ν(x)`. / 第 2 種ベッセル関数 `Y_ν(x)` の第 `k` 正零点 `y_ν,ₖ` を計算します。 |
| `static double Beta(double a, double b)` | Computes the beta function `B(a, b) = Γ(a) Γ(b) / Γ(a + b)`. / ベータ関数 `B(a, b) = Γ(a) Γ(b) / Γ(a + b)` を計算します。 |
| `static double Binomial(double n, double k)` | Computes the generalized binomial coefficient `C(n, k) = Γ(n + 1) / (Γ(k + 1) Γ(n - k + 1))`. / 一般化二項係数 `C(n, k) = Γ(n + 1) / (Γ(k + 1) Γ(n - k + 1))` を計算します。 |
| `static double BivariateNormalCdf(double x, double y, double rho)` | Computes the cumulative distribution function of the bivariate standard normal distribution, `Φ₂(x, y; ρ) = P(X ≤ x, Y ≤ y)`. / 二変量標準正規分布の累積分布関数 `Φ₂(x, y; ρ) = P(X ≤ x, Y ≤ y)` を計算します。 |
| `static double BoseEinstein(double j, double x)` | Computes the complete Bose–Einstein integral `G_j(x) = Li_(j+1)(e^x)`. / 完全 Bose–Einstein 積分 `G_j(x) = Li_(j+1)(eˣ)` を計算します。 |
| `static double CarlsonRC(double x, double y)` | Computes Carlson's degenerate elliptic integral `R_C(x, y) = R_F(x, y, y)`. / カールソンの退化楕円積分 `R_C(x, y) = R_F(x, y, y)` を計算します。 |
| `static double CarlsonRD(double x, double y, double z)` | Computes Carlson's symmetric elliptic integral of the second kind, `R_D(x, y, z) = (3 / 2) ∫₀^∞ dt / ((t + z) √((t + x)(t + y)(t + z)))`. / カールソンの第 2 種対称楕円積分 `R_D(x, y, z) = (3 / 2) ∫₀^∞ dt / ((t + z) √((t + x)(t + y)(t + z)))` を計算します。 |
| `static double CarlsonRF(double x, double y, double z)` | Computes Carlson's symmetric elliptic integral of the first kind, `R_F(x, y, z) = (1 / 2) ∫₀^∞ dt / √((t + x)(t + y)(t + z))`. / カールソンの第 1 種対称楕円積分 `R_F(x, y, z) = (1 / 2) ∫₀^∞ dt / √((t + x)(t + y)(t + z))` を計算します。 |
| `static double CarlsonRG(double x, double y, double z)` | Computes Carlson's completely symmetric elliptic integral of the second kind, `R_G(x, y, z)`. / カールソンの第 2 種完全対称楕円積分 `R_G(x, y, z)` を計算します。 |
| `static double CarlsonRJ(double x, double y, double z, double p)` | Computes Carlson's symmetric elliptic integral of the third kind, `R_J(x, y, z, p) = (3 / 2) ∫₀^∞ dt / ((t + p) √((t + x)(t + y)(t + z)))`. / カールソンの第 3 種対称楕円積分 `R_J(x, y, z, p) = (3 / 2) ∫₀^∞ dt / ((t + p) √((t + x)(t + y)(t + z)))` を計算します。 |
| `static double ChebyshevT(int n, double x)` | Computes the Chebyshev polynomial of the first kind, `Tₙ(x)`. / 第 1 種チェビシェフ多項式 `Tₙ(x)` を計算します。 |
| `static double ChebyshevU(int n, double x)` | Computes the Chebyshev polynomial of the second kind, `Uₙ(x)`. / 第 2 種チェビシェフ多項式 `Uₙ(x)` を計算します。 |
| `static double Clausen(double theta)` | Computes the Clausen function `Cl₂(θ) = Σₖ sin(kθ) / k² = -∫₀^θ ln \|2 sin(t / 2)\| dt`. / クラウゼン関数 `Cl₂(θ) = Σₖ sin(kθ) / k² = -∫₀^θ ln \|2 sin(t / 2)\| dt` を計算します。 |
| `static double ClebschGordan(double j1, double m1, double j2, double m2, double j, double m)` | Computes the Clebsch–Gordan coefficient `⟨j₁ m₁ j₂ m₂ \| j m⟩`. / Clebsch–Gordan 係数 `⟨j₁ m₁ j₂ m₂ \| j m⟩` を計算します。 |
| `static double CosIntegral(double x)` | Computes the cosine integral `Ci(x)`. / 余弦積分 `Ci(x)` を計算します。 |
| `static double CoshIntegral(double x)` | Computes the hyperbolic cosine integral `Chi(x)`. / 双曲線余弦積分 `Chi(x)` を計算します。 |
| `static double CoulombF(double l, double eta, double rho)` | Computes the regular Coulomb wave function `F_ℓ(η, ρ)`. / 正則クーロン波動関数 `F_ℓ(η, ρ)` を計算します。 |
| `static double CoulombG(double l, double eta, double rho)` | Computes the irregular Coulomb wave function `G_ℓ(η, ρ)`. / 非正則クーロン波動関数 `G_ℓ(η, ρ)` を計算します。 |
| `static double Dawson(double x)` | Computes Dawson's integral `F(x) = e^(-x²) ∫₀ˣ e^(t²) dt`. / ドーソン関数 `F(x) = e^(-x²) ∫₀ˣ e^(t²) dt` を計算します。 |
| `static double Debye(int n, double x)` | Computes the Debye function `Dₙ(x) = (n / xⁿ) ∫₀ˣ tⁿ / (eᵗ - 1) dt`. / デバイ関数 `Dₙ(x) = (n / xⁿ) ∫₀ˣ tⁿ / (eᵗ - 1) dt` を計算します。 |
| `static Complex Digamma(Complex z)` | Computes the complex digamma function `ψ(z) = Γ'(z) / Γ(z)`. / 複素ディガンマ関数 `ψ(z) = Γ'(z) / Γ(z)` を計算します。 |
| `static double Digamma(double x)` | Computes the digamma function `ψ(x) = Γ'(x) / Γ(x)`. / ディガンマ関数 `ψ(x) = Γ'(x) / Γ(x)` を計算します。 |
| `static double Dilogarithm(double x)` | Computes the dilogarithm (Spence's function), `Li₂(x) = -∫₀ˣ ln(1 - t) / t dt = Σ xᵏ / k²`. / 二重対数関数（スペンス関数）`Li₂(x) = -∫₀ˣ ln(1 - t) / t dt = Σ xᵏ / k²` を計算します。 |
| `static double DirichletBeta(double s)` | Computes the Dirichlet beta function `β(s) = Σₙ₌₀^∞ (-1)ⁿ (2n + 1)⁻ˢ`. / ディリクレのベータ関数 `β(s) = Σₙ₌₀^∞ (-1)ⁿ (2n + 1)⁻ˢ` を計算します。 |
| `static double DirichletEta(double s)` | Computes the Dirichlet eta function `η(s) = (1 - 2¹⁻ˢ) ζ(s)`. / ディリクレのイータ関数 `η(s) = (1 - 2¹⁻ˢ) ζ(s)` を計算します。 |
| `static double DoubleFactorial(int n)` | Computes the double factorial `n!! = n(n - 2)(n - 4)…`. / 二重階乗 `n!! = n(n - 2)(n - 4)…` を計算します。 |
| `static double EllipticD(double m)` | Computes the complete elliptic integral `D(m) = (K(m) - E(m)) / m = ∫₀^(π/2) sin²θ / √(1 - m sin²θ) dθ`. / 完全楕円積分 `D(m) = (K(m) - E(m)) / m = ∫₀^(π/2) sin²θ / √(1 - m sin²θ) dθ` を計算します。 |
| `static double EllipticD(double phi, double m)` | Computes the incomplete elliptic integral `D(φ\|m) = (F(φ\|m) - E(φ\|m)) / m = ∫₀^φ sin²θ / √(1 - m sin²θ) dθ`. / 不完全楕円積分 `D(φ\|m) = (F(φ\|m) - E(φ\|m)) / m = ∫₀^φ sin²θ / √(1 - m sin²θ) dθ` を計算します。 |
| `static double EllipticE(double m)` | Computes the complete elliptic integral of the second kind, `E(m) = ∫₀^(π/2) √(1 - m sin²θ) dθ`. / 第 2 種完全楕円積分 `E(m) = ∫₀^(π/2) √(1 - m sin²θ) dθ`（パラメータ `m = k²`）を計算します。 |
| `static double EllipticE(double phi, double m)` | Computes the incomplete elliptic integral of the second kind, `E(φ\|m) = ∫₀^φ √(1 - m sin²θ) dθ`. / 第 2 種不完全楕円積分 `E(φ\|m) = ∫₀^φ √(1 - m sin²θ) dθ` を計算します。 |
| `static double EllipticF(double phi, double m)` | Computes the incomplete elliptic integral of the first kind, `F(φ\|m) = ∫₀^φ dθ / √(1 - m sin²θ)`. / 第 1 種不完全楕円積分 `F(φ\|m) = ∫₀^φ dθ / √(1 - m sin²θ)` を計算します。 |
| `static double EllipticK(double m)` | Computes the complete elliptic integral of the first kind, `K(m) = ∫₀^(π/2) dθ / √(1 - m sin²θ)`. / 第 1 種完全楕円積分 `K(m) = ∫₀^(π/2) dθ / √(1 - m sin²θ)`（パラメータ `m = k²`）を計算します。 |
| `static double EllipticNome(double m)` | Computes the elliptic nome `q(m) = exp(-π K(1 - m) / K(m))`. / 楕円関数のネーム `q(m) = exp(-π K(1 - m) / K(m))` を計算します。 |
| `static double EllipticNomeInverse(double q)` | Computes the elliptic parameter `m(q) = (θ₂(0, q) / θ₃(0, q))⁴` from the nome `q`. / ネーム `q` から楕円パラメータ `m(q) = (θ₂(0, q) / θ₃(0, q))⁴` を逆算します。 |
| `static double EllipticPi(double n, double m)` | Computes the complete elliptic integral of the third kind, `Π(n\|m) = ∫₀^(π/2) dθ / ((1 - n sin²θ) √(1 - m sin²θ))`. / 第 3 種完全楕円積分 `Π(n\|m) = ∫₀^(π/2) dθ / ((1 - n sin²θ) √(1 - m sin²θ))` を計算します。 |
| `static double EllipticPi(double n, double phi, double m)` | Computes the incomplete elliptic integral of the third kind, `Π(n; φ\|m) = ∫₀^φ dθ / ((1 - n sin²θ) √(1 - m sin²θ))`. / 第 3 種不完全楕円積分 `Π(n; φ\|m) = ∫₀^φ dθ / ((1 - n sin²θ) √(1 - m sin²θ))` を計算します。 |
| `static double Erf(double x)` | Computes the error function `erf(x)`. / 誤差関数 `erf(x)` を計算します。 |
| `static double ErfDerivative(double x)` | Computes the derivative of the error function, `erf'(x) = (2 / √π) e^(-x²)`. / 誤差関数の導関数 `erf'(x) = (2 / √π) e^(-x²)` を計算します。 |
| `static double Erfc(double x)` | Computes the complementary error function `erfc(x) = 1 - erf(x)`. / 相補誤差関数 `erfc(x) = 1 - erf(x)` を計算します。 |
| `static double Erfcx(double x)` | Computes the scaled complementary error function `erfcx(x) = e^(x²) erfc(x)`. / スケーリングされた相補誤差関数 `erfcx(x) = e^(x²) erfc(x)` を計算します。 |
| `static double Erfi(double x)` | Computes the imaginary error function `erfi(x) = -i erf(ix)`. / 虚誤差関数 `erfi(x) = -i erf(ix)` を計算します。 |
| `static double EulerNumber(int n)` | Computes the `n`-th Euler number `Eₙ`. / 第 `n` オイラー数 `Eₙ` を計算します。 |
| `static double EulerPolynomial(int n, double x)` | Computes the Euler polynomial `Eₙ(x)`. / オイラー多項式 `Eₙ(x)` を計算します。 |
| `static double ExpIntegralE(int n, double x)` | Computes the generalized exponential integral `Eₙ(x)` for integer order `n`. / 整数次数の一般化指数積分 `Eₙ(x)` を計算します。 |
| `static double ExpIntegralE(double p, double x)` | Computes the generalized exponential integral `E_p(x)` for real order `p`. / 実数次数の一般化指数積分 `E_p(x)` を計算します。 |
| `static double ExpIntegralEi(double x)` | Computes the exponential integral `Ei(x)`. / 指数積分 `Ei(x)` を計算します。 |
| `static double Factorial(int n)` | Computes the factorial `n!`. / 階乗 `n!` を計算します。 |
| `static Complex Faddeeva(Complex z)` | Computes the Faddeeva function `w(z) = e^(-z²) erfc(-iz)`. / Faddeeva 関数 `w(z) = e^(-z²) erfc(-iz)` を計算します。 |
| `static double FallingFactorial(double x, double n)` | Computes the falling factorial `x(x - 1)…(x - n + 1) = Γ(x + 1) / Γ(x - n + 1)`. / 下降階乗 `x(x - 1)…(x - n + 1) = Γ(x + 1) / Γ(x - n + 1)` を計算します。 |
| `static double FermiDirac(double j, double x)` | Computes the complete Fermi–Dirac integral `F_j(x) = -Li_(j+1)(-e^x)`. / 完全 Fermi–Dirac 積分 `F_j(x) = -Li_(j+1)(-eˣ)` を計算します。 |
| `static (double, double) Fresnel(double x)` | Simultaneously computes the Fresnel integrals `S(x)` and `C(x)`. / フレネル積分 `S(x)` と `C(x)` を同時に計算します。 |
| `static double FresnelC(double x)` | Computes the Fresnel cosine integral `C(x) = ∫₀ˣ cos(πt² / 2) dt`. / フレネル余弦積分 `C(x) = ∫₀ˣ cos(πt² / 2) dt` を計算します。 |
| `static double FresnelS(double x)` | Computes the Fresnel sine integral `S(x) = ∫₀ˣ sin(πt² / 2) dt`. / フレネル正弦積分 `S(x) = ∫₀ˣ sin(πt² / 2) dt` を計算します。 |
| `static Complex Gamma(Complex z)` | Computes the complex gamma function `Γ(z)`. / 複素ガンマ関数 `Γ(z)` を計算します。 |
| `static double Gamma(double x)` | Computes the gamma function `Γ(x)`. / ガンマ関数 `Γ(x)` を計算します。 |
| `static double GammaRatio(double a, double b)` | Computes the ratio of gamma functions `Γ(a) / Γ(b)`. / ガンマ関数の比 `Γ(a) / Γ(b)` を計算します。 |
| `static double GegenbauerC(int n, double lambda, double x)` | Computes the Gegenbauer (ultraspherical) polynomial `Cₙ^(λ)(x)`. / ゲーゲンバウアー多項式 `Cₙ^(λ)(x)` を計算します。 |
| `static Complex HankelH1(double nu, double x)` | Computes the Hankel function of the first kind, `H⁽¹⁾_ν(x) = J_ν(x) + i Y_ν(x)`. / 第 1 種ハンケル関数 `H⁽¹⁾_ν(x) = J_ν(x) + i Y_ν(x)` を計算します。 |
| `static Complex HankelH2(double nu, double x)` | Computes the Hankel function of the second kind, `H⁽²⁾_ν(x) = J_ν(x) - i Y_ν(x)`. / 第 2 種ハンケル関数 `H⁽²⁾_ν(x) = J_ν(x) - i Y_ν(x)` を計算します。 |
| `static double HarmonicNumber(double x)` | Computes the harmonic number `Hₓ = ψ(x + 1) + γ`. / 調和数 `Hₓ = ψ(x + 1) + γ` を計算します。 |
| `static double HermiteH(int n, double x)` | Computes the physicists' Hermite polynomial `Hₙ(x)`. / エルミート多項式（物理学者の流儀）`Hₙ(x)` を計算します。 |
| `static double HermiteHe(int n, double x)` | Computes the probabilists' Hermite polynomial `Heₙ(x)`. / エルミート多項式（確率論者の流儀）`Heₙ(x)` を計算します。 |
| `static double HeumanLambda(double phi, double m)` | Computes Heuman's lambda function `Λ₀(φ\|m) = F(φ\|1 - m) / K(1 - m) + (2 / π) K(m) Z(φ\|1 - m)`. / ヒューマンのラムダ関数 `Λ₀(φ\|m) = F(φ\|1 - m) / K(1 - m) + (2 / π) K(m) Z(φ\|1 - m)` を計算します。 |
| `static double HurwitzZeta(double s, double a)` | Computes the Hurwitz zeta function `ζ(s, a)`. / フルヴィッツのゼータ関数 `ζ(s, a)` を計算します。 |
| `static double Hyperfactorial(double z)` | Computes the hyperfactorial `H(z) = Γ(z + 1)^z / G(z + 1)`. / ハイパー階乗（超階乗）`H(z) = Γ(z + 1)^z / G(z + 1)` を計算します。 |
| `static double Hypergeometric0F1(double b, double x)` | Computes the confluent hypergeometric limit function `₀F₁(; b; x)`. / 合流型極限超幾何関数 `₀F₁(; b; x)` を計算します。 |
| `static double Hypergeometric1F1(double a, double b, double x)` | Computes Kummer's confluent hypergeometric function `₁F₁(a; b; x)` (also denoted `M(a, b, x)`). / 合流型超幾何関数（Kummer 関数）`₁F₁(a; b; x)`（`M(a, b, x)`）を計算します。 |
| `static double Hypergeometric1F1Regularized(double a, double b, double x)` | Computes the regularized confluent hypergeometric function `₁F̃₁(a; b; x) = ₁F₁(a; b; x) / Γ(b)`. / 正則化された合流型超幾何関数 `₁F̃₁(a; b; x) = ₁F₁(a; b; x) / Γ(b)` を計算します。 |
| `static double Hypergeometric2F1(double a, double b, double c, double x)` | Computes the Gauss hypergeometric function `₂F₁(a, b; c; x)`. / ガウスの超幾何関数 `₂F₁(a, b; c; x)` を計算します。 |
| `static double HypergeometricU(double a, double b, double x)` | Computes the confluent hypergeometric function of the second kind (Tricomi's function) `U(a, b, x)`. / 第 2 種合流型超幾何関数（Tricomi 関数）`U(a, b, x)` を計算します。 |
| `static double IncompleteBeta(double a, double b, double x)` | Computes the incomplete beta function `Bₓ(a, b) = ∫₀ˣ t^(a - 1) (1 - t)^(b - 1) dt`. / 不完全ベータ関数 `Bₓ(a, b) = ∫₀ˣ t^(a - 1) (1 - t)^(b - 1) dt` を計算します。 |
| `static double IntegralBesselI0(double x)` | Computes the integral of the modified Bessel function of the first kind, `∫₀ˣ I₀(t) dt`. / 第 1 種変形ベッセル関数の積分 `∫₀ˣ I₀(t) dt` を計算します。 |
| `static double IntegralBesselJ0(double x)` | Computes the integral of the Bessel function of the first kind, `∫₀ˣ J₀(t) dt`. / 第 1 種ベッセル関数の積分 `∫₀ˣ J₀(t) dt` を計算します。 |
| `static double IntegralBesselK0(double x)` | Computes the integral of the modified Bessel function of the second kind, `∫₀ˣ K₀(t) dt`. / 第 2 種変形ベッセル関数の積分 `∫₀ˣ K₀(t) dt` を計算します。 |
| `static double IntegralBesselY0(double x)` | Computes the integral of the Bessel function of the second kind, `∫₀ˣ Y₀(t) dt`. / 第 2 種ベッセル関数の積分 `∫₀ˣ Y₀(t) dt` を計算します。 |
| `static double InverseDigamma(double y)` | Computes the inverse of the digamma function, returning the unique positive `x` satisfying `ψ(x) = y`. / 逆ディガンマ関数を計算し、`ψ(x) = y` を満たす唯一の正の値 `x` を返します。 |
| `static double InverseErf(double x)` | Computes the inverse error function `erf⁻¹(x)`. / 逆誤差関数 `erf⁻¹(x)` を計算します。 |
| `static double InverseErfc(double q)` | Computes the inverse complementary error function `erfc⁻¹(q)`. / 逆相補誤差関数 `erfc⁻¹(q)` を計算します。 |
| `static double InverseJacobiCN(double x, double m)` | Computes the principal inverse Jacobi elliptic function `arccn(x\|m) = F(arccos x\|m)`. / 逆ヤコビ楕円関数 `arccn(x\|m) = F(arccos x\|m)`（主値 `0 ≤ u ≤ 2K`）を計算します。 |
| `static double InverseJacobiDN(double x, double m)` | Computes the principal inverse Jacobi elliptic function `arcdn(x\|m)`. / 逆ヤコビ楕円関数 `arcdn(x\|m)`（主値 `0 ≤ u ≤ K`）を計算します。 |
| `static double InverseJacobiSN(double x, double m)` | Computes the principal inverse Jacobi elliptic function `arcsn(x\|m) = F(arcsin x\|m)`. / 逆ヤコビ楕円関数 `arcsn(x\|m) = F(arcsin x\|m)`（主値 `\|u\| ≤ K`）を計算します。 |
| `static double InverseKolmogorovCdf(double p)` | Computes the quantile `K⁻¹(p)` of the Kolmogorov distribution. / コルモゴロフ分布の分位点 `K⁻¹(p)` を計算します。 |
| `static double InverseKolmogorovSf(double q)` | Computes the upper-tail quantile of the Kolmogorov distribution satisfying `1 - K(x) = q`. / コルモゴロフ分布の上側分位点（`1 - K(x) = q` を満たす `x`）を計算します。 |
| `static double InverseNormalCdf(double p)` | Computes the standard normal quantile function `Φ⁻¹(p)`. / 標準正規分布の分位点関数 `Φ⁻¹(p)` を計算します。 |
| `static double InverseNormalLogCdf(double logP)` | Computes the standard normal quantile `Φ⁻¹(e^logP)` from a log-probability. / 対数確率から標準正規分布の分位点 `Φ⁻¹(e^logP)` を計算します。 |
| `static double InverseNormalLogSf(double logQ)` | Computes the standard normal upper-tail quantile `Q⁻¹(e^logQ)` from a log-probability. / 対数確率から標準正規分布の上側分位点 `Q⁻¹(e^logQ)` を計算します。 |
| `static double InverseNormalSf(double q)` | Computes the standard normal upper-tail quantile `Q⁻¹(q) = -Φ⁻¹(q)`. / 標準正規分布の上側分位点 `Q⁻¹(q) = -Φ⁻¹(q)` を計算します。 |
| `static double InverseRegularizedGammaP(double a, double p)` | Computes the inverse of the regularized lower incomplete gamma function, finding `x` such that `P(a, x) = p`. / 正則化下側不完全ガンマ関数の逆関数（`P(a, x) = p` を満たす `x`）を計算します。 |
| `static double InverseRegularizedGammaQ(double a, double q)` | Computes the inverse of the regularized upper incomplete gamma function, finding `x` such that `Q(a, x) = q`. / 正則化上側不完全ガンマ関数の逆関数（`Q(a, x) = q` を満たす `x`）を計算します。 |
| `static double InverseRegularizedIncompleteBeta(double a, double b, double p)` | Computes the inverse of the regularized incomplete beta function, finding `x` such that `Iₓ(a, b) = p`. / 正則化不完全ベータ関数の逆関数（`Iₓ(a, b) = p` を満たす `x`）を計算します。 |
| `static double InverseRegularizedIncompleteBetaComplement(double a, double b, double q)` | Computes the inverse of the complemented regularized incomplete beta function, finding `x` such that `1 - Iₓ(a, b) = q`. / 正則化不完全ベータ関数の補関数の逆関数（`1 - Iₓ(a, b) = q` を満たす `x`）を計算します。 |
| `static double JacobiAmplitude(double u, double m)` | Computes the Jacobi amplitude function `am(u \| m)`. / ヤコビの振幅関数 `am(u \| m)` を計算します。 |
| `static double JacobiCN(double u, double m)` | Computes the Jacobi elliptic function `cn(u \| m)`. / ヤコビの楕円関数 `cn(u \| m)` を計算します。 |
| `static double JacobiDN(double u, double m)` | Computes the Jacobi elliptic function `dn(u \| m)`. / ヤコビの楕円関数 `dn(u \| m)` を計算します。 |
| `static (double, double, double) JacobiElliptic(double u, double m)` | Computes the Jacobi elliptic functions `sn(u \| m)`, `cn(u \| m)`, and `dn(u \| m)` simultaneously. / ヤコビの楕円関数 `sn(u \| m)`、`cn(u \| m)`、`dn(u \| m)` を同時に計算します。 |
| `static double JacobiP(int n, double alpha, double beta, double x)` | Computes the Jacobi polynomial `Pₙ^(α,β)(x)`. / ヤコビ多項式 `Pₙ^(α,β)(x)` を計算します。 |
| `static double JacobiSN(double u, double m)` | Computes the Jacobi elliptic function `sn(u \| m)`. / ヤコビの楕円関数 `sn(u \| m)` を計算します。 |
| `static double JacobiTheta1(double z, double q)` | Computes the Jacobi theta function `θ₁(z, q)`. / ヤコビのテータ関数 `θ₁(z, q)` を計算します。 |
| `static double JacobiTheta2(double z, double q)` | Computes the Jacobi theta function `θ₂(z, q)`. / ヤコビのテータ関数 `θ₂(z, q)` を計算します。 |
| `static double JacobiTheta3(double z, double q)` | Computes the Jacobi theta function `θ₃(z, q)`. / ヤコビのテータ関数 `θ₃(z, q)` を計算します。 |
| `static double JacobiTheta4(double z, double q)` | Computes the Jacobi theta function `θ₄(z, q)`. / ヤコビのテータ関数 `θ₄(z, q)` を計算します。 |
| `static double JacobiZeta(double phi, double m)` | Computes Jacobi's zeta function `Z(φ\|m) = E(φ\|m) - (E(m) / K(m)) F(φ\|m)`. / ヤコビのゼータ関数 `Z(φ\|m) = E(φ\|m) - (E(m) / K(m)) F(φ\|m)` を計算します。 |
| `static double KelvinBei(int n, double x)` | Computes the Kelvin function `bei_n(x) = Im J_n(x e^(3πi / 4))`. / ケルビン関数 `bei_n(x) = Im J_n(x e^(3πi / 4))` を計算します。 |
| `static double KelvinBer(int n, double x)` | Computes the Kelvin function `ber_n(x) = Re J_n(x e^(3πi / 4))`. / ケルビン関数 `ber_n(x) = Re J_n(x e^(3πi / 4))` を計算します。 |
| `static double KelvinKei(int n, double x)` | Computes the Kelvin function `kei_n(x) = Im(e^(-nπi / 2) K_n(x e^(πi / 4)))`. / ケルビン関数 `kei_n(x) = Im(e^(-nπi / 2) K_n(x e^(πi / 4)))` を計算します。 |
| `static double KelvinKer(int n, double x)` | Computes the Kelvin function `ker_n(x) = Re(e^(-nπi / 2) K_n(x e^(πi / 4)))`. / ケルビン関数 `ker_n(x) = Re(e^(-nπi / 2) K_n(x e^(πi / 4)))` を計算します。 |
| `static double KolmogorovCdf(double x)` | Computes the cumulative distribution function `K(x)` of the Kolmogorov distribution. / コルモゴロフ分布の累積分布関数 `K(x)` を計算します。 |
| `static double KolmogorovPdf(double x)` | Computes the probability density function `K'(x)` of the Kolmogorov distribution. / コルモゴロフ分布の確率密度関数 `K'(x)` を計算します。 |
| `static double KolmogorovSf(double x)` | Computes the survival function (upper-tail probability) `1 - K(x)` of the Kolmogorov distribution. / コルモゴロフ分布の生存関数（上側確率）`1 - K(x)` を計算します。 |
| `static double LaguerreL(int n, double x)` | Computes the Laguerre polynomial `Lₙ(x)`. / ラゲール多項式 `Lₙ(x)` を計算します。 |
| `static double LaguerreL(int n, double alpha, double x)` | Computes the generalized (associated) Laguerre polynomial `Lₙ^(α)(x)`. / 一般化ラゲール多項式（ラゲール陪多項式）`Lₙ^(α)(x)` を計算します。 |
| `static double LambertW0(double x)` | Computes the principal branch `W₀(x)` of the Lambert W function. / ランベルトの W 関数の主枝 `W₀(x)` を計算します。 |
| `static double LambertWm1(double x)` | Computes the lower real branch `W₋₁(x)` of the Lambert W function. / ランベルトの W 関数の下側の枝 `W₋₁(x)` を計算します。 |
| `static double LegendreP(double nu, double x)` | Computes the Legendre function of the first kind (Ferrers function) `P_ν(x)` for real degree `ν`. / 実数次数 `ν` の第 1 種ルジャンドル関数（フェラーズ関数）`P_ν(x)` を計算します。 |
| `static double LegendreP(int n, double x)` | Computes the Legendre polynomial `Pₙ(x)`. / ルジャンドル多項式 `Pₙ(x)` を計算します。 |
| `static double LegendreQ(int n, double x)` | Computes the Legendre function of the second kind (Ferrers function) `Qₙ(x)`. / 第 2 種ルジャンドル関数（フェラーズ関数）`Qₙ(x)` を計算します。 |
| `static double LegendreQ(double nu, double x)` | Computes the Legendre function of the second kind (Ferrers function) `Q_ν(x)` for real degree `ν`. / 実数次数 `ν` の第 2 種ルジャンドル関数（フェラーズ関数）`Q_ν(x)` を計算します。 |
| `static double LerchPhi(double z, double s, double a)` | Computes the Lerch transcendent `Φ(z, s, a)`. / レルヒの超越関数 `Φ(z, s, a)` を計算します。 |
| `static double LogBarnesG(double z)` | Computes the natural logarithm of the absolute value of the Barnes G-function, `ln \|G(z)\|`. / バーンズの G 関数の絶対値の自然対数 `ln \|G(z)\|` を計算します。 |
| `static double LogBesselI(double nu, double x)` | Computes the natural logarithm of the modified Bessel function of the first kind `ln I_ν(x)`. / 第 1 種変形ベッセル関数の自然対数 `ln I_ν(x)` を計算します。 |
| `static double LogBesselIScaled(double nu, double x)` | Computes the natural logarithm of the exponentially scaled modified Bessel function of the first kind `ln(e⁻ˣ I_ν(x))`. / 指数スケーリングされた第 1 種変形ベッセル関数の自然対数 `ln(e⁻ˣ I_ν(x))` を計算します。 |
| `static double LogBeta(ReadOnlySpan<double> alpha)` | Computes the natural logarithm of the multivariate beta function, `ln B(α) = Σᵢ ln Γ(αᵢ) - ln Γ(Σᵢ αᵢ)`. / 多変量ベータ関数の自然対数 `ln B(α) = Σᵢ ln Γ(αᵢ) - ln Γ(Σᵢ αᵢ)`（ディリクレ分布の正規化定数）を計算します。 |
| `static double LogBeta(double a, double b)` | Computes the natural logarithm of the absolute value of the beta function, `ln \|B(a, b)\|`. / ベータ関数の絶対値の自然対数 `ln \|B(a, b)\|` を計算します。 |
| `static double LogBetaDerivative(double a, double b)` | Computes the partial derivative of the log-beta function with respect to its first parameter, `∂ ln \|B(a, b)\| / ∂a = ψ(a) - ψ(a + b)`. / 対数ベータ関数の第 1 引数に関する偏微分 `∂ ln \|B(a, b)\| / ∂a = ψ(a) - ψ(a + b)` を計算します。 |
| `static double LogBinomial(double n, double k)` | Computes the natural logarithm of the absolute value of the generalized binomial coefficient, `ln \|C(n, k)\|`. / 一般化二項係数の絶対値の自然対数 `ln \|C(n, k)\|` を計算します。 |
| `static double LogFactorial(int n)` | Computes the natural logarithm of the factorial, `ln(n!)`. / 階乗の自然対数 `ln(n!)` を計算します。 |
| `static Complex LogGamma(Complex z)` | Computes the principal branch of the complex log-gamma function, `ln Γ(z)`. / 複素対数ガンマ関数 `ln Γ(z)` の主枝を計算します。 |
| `static double LogGamma(double x)` | Computes the natural logarithm of the absolute value of the gamma function, `ln \|Γ(x)\|`. / ガンマ関数の絶対値の自然対数 `ln \|Γ(x)\|` を計算します。 |
| `static double LogIntegral(double x)` | Computes the logarithmic integral `li(x) = Ei(ln x)`. / 対数積分 `li(x) = Ei(ln x)` を計算します。 |
| `static double LogMultivariateGamma(double a, int dimension)` | Computes the natural logarithm of the multivariate gamma function `ln Γₚ(a)`. / 多変量ガンマ関数の自然対数 `ln Γₚ(a)` を計算します。 |
| `static double LogRegularizedGammaP(double a, double x)` | Computes the natural logarithm of the regularized lower incomplete gamma function `ln P(a, x)`. / 正則化下側不完全ガンマ関数の自然対数 `ln P(a, x)` を計算します。 |
| `static double LogRegularizedGammaPShapeDerivative(double a, double x)` | Computes the shape derivative of the natural logarithm of the regularized lower incomplete gamma function. / 正則化下側不完全ガンマ関数の自然対数を形状パラメータで微分します。 |
| `static double LogRegularizedGammaQ(double a, double x)` | Computes the natural logarithm of the regularized upper incomplete gamma function `ln Q(a, x)`. / 正則化上側不完全ガンマ関数の自然対数 `ln Q(a, x)` を計算します。 |
| `static double LogRegularizedGammaQShapeDerivative(double a, double x)` | Computes the shape derivative of the natural logarithm of the regularized upper incomplete gamma function. / 正則化上側不完全ガンマ関数の自然対数を形状パラメータで微分します。 |
| `static double LogRegularizedIncompleteBeta(double a, double b, double x)` | Computes the natural logarithm of the regularized incomplete beta function `ln Iₓ(a, b)`. / 正則化不完全ベータ関数の自然対数 `ln Iₓ(a, b)` を計算します。 |
| `static double LogRegularizedIncompleteBetaComplement(double a, double b, double x)` | Computes the natural logarithm of the complement of the regularized incomplete beta function `ln(1 - Iₓ(a, b))`. / 正則化不完全ベータ関数の補関数の自然対数 `ln(1 - Iₓ(a, b))` を計算します。 |
| `static double LowerIncompleteGamma(double a, double x)` | Computes the lower incomplete gamma function `γ(a, x) = ∫₀ˣ t^(a - 1) e⁻ᵗ dt`. / 下側不完全ガンマ関数 `γ(a, x) = ∫₀ˣ t^(a - 1) e⁻ᵗ dt` を計算します。 |
| `static double MarcumQ(double mu, double x, double y)` | Computes the generalized Marcum Q-function `Q_μ(x, y)`. / 一般化 Marcum Q 関数 `Q_μ(x, y)` を計算します。 |
| `static double MathieuA(int n, double q)` | Computes the Mathieu characteristic value `aₙ(q)` for even periodic solutions. / Mathieu 方程式の偶関数解に対応する特性値 `aₙ(q)` を計算します。 |
| `static double MathieuB(int n, double q)` | Computes the Mathieu characteristic value `bₙ(q)` for odd periodic solutions. / Mathieu 方程式の奇関数解に対応する特性値 `bₙ(q)` を計算します。 |
| `static double MathieuCe(int n, double q, double x)` | Computes the even Mathieu function `ceₙ(x, q)`. / 偶関数解の Mathieu 関数 `ceₙ(x, q)` を計算します。 |
| `static double MathieuSe(int n, double q, double x)` | Computes the odd Mathieu function `seₙ(x, q)`. / 奇関数解の Mathieu 関数 `seₙ(x, q)` を計算します。 |
| `static double MittagLeffler(double alpha, double x)` | Computes the one-parameter Mittag-Leffler function `E_α(x) = E_(α,1)(x)`. / 1 パラメータ Mittag-Leffler 関数 `E_α(x) = E_(α,1)(x)` を計算します。 |
| `static double MittagLeffler(double alpha, double beta, double x)` | Computes the two-parameter Mittag-Leffler function `E_(α,β)(x) = Σ_(k≥0) xᵏ / Γ(αk + β)`. / 2 パラメータ Mittag-Leffler 関数 `E_(α,β)(x) = Σ_(k≥0) xᵏ / Γ(αk + β)` を計算します。 |
| `static double ModifiedSphericalBesselI(int n, double x)` | Computes the modified spherical Bessel function of the first kind `iₙ(x)`. / 第 1 種変形球ベッセル関数 `iₙ(x)` を計算します。 |
| `static double ModifiedSphericalBesselK(int n, double x)` | Computes the modified spherical Bessel function of the second kind `kₙ(x)`. / 第 2 種変形球ベッセル関数 `kₙ(x)` を計算します。 |
| `static double NoncentralChiSquaredCdf(double k, double lambda, double x)` | Computes the cumulative distribution function `F(x; k, λ)` of the noncentral chi-squared distribution. / 非心 χ² 分布の累積分布関数 `F(x; k, λ)` を計算します。 |
| `static double NoncentralChiSquaredLogPdf(double k, double lambda, double x)` | Computes the natural logarithm of the probability density function `ln f(x; k, λ)` of the noncentral chi-squared distribution. / 非心 χ² 分布の確率密度関数の自然対数 `ln f(x; k, λ)` を計算します。 |
| `static double NoncentralChiSquaredPdf(double k, double lambda, double x)` | Computes the probability density function `f(x; k, λ)` of the noncentral chi-squared distribution. / 非心 χ² 分布の確率密度関数 `f(x; k, λ)` を計算します。 |
| `static double NoncentralChiSquaredSf(double k, double lambda, double x)` | Computes the survival function (upper-tail probability) `1 - F(x; k, λ)` of the noncentral chi-squared distribution. / 非心 χ² 分布の生存関数（上側確率）`1 - F(x; k, λ)` を計算します。 |
| `static double NormalCdf(double x)` | Computes the standard normal cumulative distribution function `Φ(x)`. / 標準正規分布の累積分布関数 `Φ(x)` を計算します。 |
| `static double NormalHazard(double x)` | Computes the hazard function (inverse Mills ratio) of the standard normal distribution, `φ(x) / Q(x)`. / 標準正規分布のハザード関数（逆ミルズ比）`φ(x) / Q(x)` を計算します。 |
| `static double NormalLogCdf(double x)` | Computes the natural logarithm of the standard normal cumulative distribution function `ln Φ(x)`. / 標準正規分布の累積分布関数の自然対数 `ln Φ(x)` を計算します。 |
| `static double NormalLogPdf(double x)` | Computes the natural logarithm of the standard normal probability density function `ln φ(x)`. / 標準正規分布の確率密度関数の自然対数 `ln φ(x)` を計算します。 |
| `static double NormalLogSf(double x)` | Computes the natural logarithm of the standard normal survival function `ln Q(x) = ln(1 - Φ(x))`. / 標準正規分布の生存関数の自然対数 `ln Q(x) = ln(1 - Φ(x))` を計算します。 |
| `static double NormalPdf(double x)` | Computes the standard normal probability density function `φ(x)`. / 標準正規分布の確率密度関数 `φ(x)` を計算します。 |
| `static double NormalPdfDerivative(double x)` | Computes the derivative of the standard normal probability density function, `φ'(x) = -x φ(x)`. / 標準正規分布の確率密度関数の導関数 `φ'(x) = -x φ(x)` を計算します。 |
| `static double NormalReverseHazard(double x)` | Computes the reverse hazard function of the standard normal distribution, `φ(x) / Φ(x)`. / 標準正規分布の逆ハザード関数 `φ(x) / Φ(x)` を計算します。 |
| `static double NormalSf(double x)` | Computes the standard normal survival function (upper-tail probability) `Q(x) = 1 - Φ(x)`. / 標準正規分布の生存関数（上側確率）`Q(x) = 1 - Φ(x)` を計算します。 |
| `static double OwensT(double h, double a)` | Computes Owen's T function `T(h, a)`. / Owen の T 関数 `T(h, a)` を計算します。 |
| `static double ParabolicCylinderD(double nu, double x)` | Computes Whittaker's parabolic cylinder function `D_ν(x) = U(-ν - 1/2, x)`. / Whittaker の放物柱関数 `D_ν(x) = U(-ν - 1/2, x)` を計算します。 |
| `static double ParabolicCylinderU(double a, double x)` | Computes the parabolic cylinder function `U(a, x)`. / 放物柱関数 `U(a, x)` を計算します。 |
| `static double ParabolicCylinderV(double a, double x)` | Computes the parabolic cylinder function `V(a, x)`. / 放物柱関数 `V(a, x)` を計算します。 |
| `static double Pochhammer(double x, double n)` | Computes the Pochhammer symbol (rising factorial) `(x)ₙ = Γ(x + n) / Γ(x)`. / ポッホハマー記号（上昇階乗）`(x)ₙ = Γ(x + n) / Γ(x)` を計算します。 |
| `static double Polygamma(int n, double x)` | Computes the polygamma function `ψ⁽ⁿ⁾(x) = dⁿψ(x) / dxⁿ`. / ポリガンマ関数 `ψ⁽ⁿ⁾(x) = dⁿψ(x) / dxⁿ` を計算します。 |
| `static double Polylogarithm(double s, double x)` | Computes the polylogarithm `Li_s(x)` for real order `s` and real argument `x`. / 実数の次数 `s` と実数引数 `x` に対する多重対数関数 `Li_s(x)` を計算します。 |
| `static Complex ReciprocalGamma(Complex z)` | Computes the reciprocal of the complex gamma function, `1 / Γ(z)`. / 複素ガンマ関数の逆数 `1 / Γ(z)`（整関数）を計算します。 |
| `static double ReciprocalGamma(double x)` | Computes the reciprocal gamma function `1 / Γ(x)`. / ガンマ関数の逆数 `1 / Γ(x)` を計算します。 |
| `static double RegularizedGammaP(double a, double x)` | Computes the regularized lower incomplete gamma function `P(a, x) = γ(a, x) / Γ(a)`. / 正則化下側不完全ガンマ関数 `P(a, x) = γ(a, x) / Γ(a)` を計算します。 |
| `static double RegularizedGammaQ(double a, double x)` | Computes the regularized upper incomplete gamma function `Q(a, x) = Γ(a, x) / Γ(a) = 1 - P(a, x)`. / 正則化上側不完全ガンマ関数 `Q(a, x) = Γ(a, x) / Γ(a) = 1 - P(a, x)` を計算します。 |
| `static double RegularizedIncompleteBeta(double a, double b, double x)` | Computes the regularized incomplete beta function `Iₓ(a, b) = Bₓ(a, b) / B(a, b)`. / 正則化不完全ベータ関数 `Iₓ(a, b) = Bₓ(a, b) / B(a, b)` を計算します。 |
| `static double RegularizedIncompleteBetaComplement(double a, double b, double x)` | Computes the complement of the regularized incomplete beta function, `1 - Iₓ(a, b) = I₁₋ₓ(b, a)`. / 正則化不完全ベータ関数の補関数 `1 - Iₓ(a, b) = I₁₋ₓ(b, a)` を計算します。 |
| `static double ScorerGi(double x)` | Computes Scorer's function `Gi(x)`. / スコーラー関数 `Gi(x)` を計算します。 |
| `static double ScorerHi(double x)` | Computes Scorer's function `Hi(x)`. / スコーラー関数 `Hi(x)` を計算します。 |
| `static double SignGamma(double x)` | Computes the sign of the gamma function, `sgn Γ(x)`. / ガンマ関数の符号 `sgn Γ(x)` を計算します。 |
| `static double SinIntegral(double x)` | Computes the sine integral `Si(x)`. / 正弦積分 `Si(x)` を計算します。 |
| `static double SinhIntegral(double x)` | Computes the hyperbolic sine integral `Shi(x)`. / 双曲線正弦積分 `Shi(x)` を計算します。 |
| `static double SphericalBesselJ(int n, double x)` | Computes the spherical Bessel function of the first kind `jₙ(x)`. / 第 1 種球ベッセル関数 `jₙ(x)` を計算します。 |
| `static double SphericalBesselY(int n, double x)` | Computes the spherical Bessel function of the second kind `yₙ(x)`. / 第 2 種球ベッセル関数 `yₙ(x)` を計算します。 |
| `static Complex SphericalHarmonicY(int l, int m, double theta, double phi)` | Computes the spherical harmonic `Yₗᵐ(θ, φ)`. / 球面調和関数 `Yₗᵐ(θ, φ)` を計算します。 |
| `static double StirlingFirstKind(int n, int k)` | Computes the signed Stirling number of the first kind `s(n, k)`, the coefficient of `xᵏ` in the falling factorial `x(x - 1)…(x - n + 1)`. / 符号付き第 1 種スターリング数 `s(n, k)`（下降階乗 `x(x - 1)…(x - n + 1)` における `xᵏ` の係数）を計算します。 |
| `static double StirlingSecondKind(int n, int k)` | Computes the Stirling number of the second kind `S(n, k)`, the number of partitions of an `n`-element set into `k` non-empty subsets. / 第 2 種スターリング数 `S(n, k)`（`n` 個の要素を `k` 個の空でない部分集合に分ける方法の数）を計算します。 |
| `static double StruveH(double nu, double x)` | Computes the Struve function `H_ν(x)`. / シュトルーベ関数 `H_ν(x)` を計算します。 |
| `static double StruveL(double nu, double x)` | Computes the modified Struve function `L_ν(x)`. / 変形シュトルーベ関数 `L_ν(x)` を計算します。 |
| `static double Superfactorial(double z)` | Computes the superfactorial `sf(z) = G(z + 2)`. / スーパー階乗（超階乗）`sf(z) = G(z + 2)` を計算します。 |
| `static double SynchrotronF(double x)` | Computes the first synchrotron radiation function `F(x) = x ∫ₓ^∞ K_(5/3)(t) dt`. / 第 1 シンクロトロン関数 `F(x) = x ∫ₓ^∞ K_(5/3)(t) dt` を計算します。 |
| `static double SynchrotronG(double x)` | Computes the second synchrotron radiation function `G(x) = x K_(2/3)(x)`. / 第 2 シンクロトロン関数 `G(x) = x K_(2/3)(x)` を計算します。 |
| `static double Trigamma(double x)` | Computes the trigamma function `ψ₁(x) = ψ'(x)`. / トリガンマ関数 `ψ₁(x) = ψ'(x)` を計算します。 |
| `static double UpperIncompleteGamma(double a, double x)` | Computes the upper incomplete gamma function `Γ(a, x) = ∫ₓ^∞ t^(a - 1) e⁻ᵗ dt`. / 上側不完全ガンマ関数 `Γ(a, x) = ∫ₓ^∞ t^(a - 1) e⁻ᵗ dt` を計算します。 |
| `static double VoigtProfile(double x, double sigma, double gamma)` | Computes the Voigt profile `V(x; σ, γ)`. / Voigt プロファイル `V(x; σ, γ)` を計算します。 |
| `static double WhittakerM(double kappa, double mu, double x)` | Computes the Whittaker function of the first kind `M_(κ,μ)(x) = e^(-x/2) x^(μ + 1/2) M(1/2 + μ - κ, 1 + 2μ, x)`. / 第 1 種 Whittaker 関数 `M_(κ,μ)(x) = e^(-x/2) x^(μ + 1/2) M(1/2 + μ - κ, 1 + 2μ, x)` を計算します。 |
| `static double WhittakerW(double kappa, double mu, double x)` | Computes the Whittaker function of the second kind `W_(κ,μ)(x) = e^(-x/2) x^(μ + 1/2) U(1/2 + μ - κ, 1 + 2μ, x)`. / 第 2 種 Whittaker 関数 `W_(κ,μ)(x) = e^(-x/2) x^(μ + 1/2) U(1/2 + μ - κ, 1 + 2μ, x)` を計算します。 |
| `static double Wigner3j(double j1, double j2, double j3, double m1, double m2, double m3)` | Computes the Wigner 3-j symbol `(j₁ j₂ j₃; m₁ m₂ m₃)`. / Wigner の 3j 記号 `(j₁ j₂ j₃; m₁ m₂ m₃)` を計算します。 |
| `static double Wigner6j(double j1, double j2, double j3, double j4, double j5, double j6)` | Computes the Wigner 6-j symbol `{j₁ j₂ j₃; j₄ j₅ j₆}`. / Wigner の 6j 記号 `{j₁ j₂ j₃; j₄ j₅ j₆}` を計算します。 |
| `static double WrightOmega(double x)` | Computes the Wright omega function `ω(x)`. / ライトのオメガ関数 `ω(x)` を計算します。 |
| `static double Zeta(double s)` | Computes the Riemann zeta function `ζ(s)`. / リーマンゼータ関数 `ζ(s)` を計算します。 |
| `static double ZetaMinusOne(double s)` | Computes `ζ(s) - 1` without cancellation for large `s`. / 大きな `s` での桁落ちを避けて `ζ(s) - 1` を計算します。 |

## Tuna.LinearAlgebra

### `class BatchCholeskyDecomposition`

Represents the batched Cholesky decomposition, `A[k] = L[k]L[k]ᵀ`, of a batch of symmetric positive definite matrices of the same order. / 同じ次数の対称正定値行列のバッチに対するコレスキー分解 `A[k] = L[k]L[k]ᵀ` を表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int BatchCount { get; }` | Gets the number of matrices in the batch. / バッチ内の行列の個数を取得します。 |
| `int FailureCount { get; }` | Gets the number of matrices in the batch that failed factorization because they were not positive definite. / 正定値でなく分解に失敗した行列の個数を取得します。 |
| `Tensor<double> L { get; }` | Gets the batch of lower triangular factors `L[k]` with positive diagonal entries. / 対角成分が正の下三角因子 `L[k]` のバッチを取得します。 |
| `int Order { get; }` | Gets the order `n` of each matrix in the batch. / バッチ内の各行列の次数 `n` を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static BatchCholeskyDecomposition Compute(ReadOnlyTensorSpan<double> a)` | Computes the batched Cholesky decomposition of the specified symmetric positive definite matrices. / 指定した対称正定値行列のバッチに対してコレスキー分解を計算します。 |
| `static BatchCholeskyDecomposition Compute(ReadOnlyTensorSpan<double> a, Tensor<double> destination)` | Computes the batched Cholesky decomposition and stores the factor tensor `L` in `destination`. / バッチコレスキー分解を計算し、因子テンソル `L` を `destination` に格納します。 |
| `Vector<double> EstimateReciprocalConditionNumbers()` | Estimates the reciprocal 1-norm condition number, `1 / (‖A[k]‖₁‖A[k]⁻¹‖₁)`, for each matrix in the batch. / バッチ内の各行列の 1-ノルム条件数の逆数 `1 / (‖A[k]‖₁‖A[k]⁻¹‖₁)` を推定します。 |
| `VectorSpan<double> EstimateReciprocalConditionNumbers(VectorSpan<double> destination)` | Estimates the reciprocal 1-norm condition number for each matrix in the batch and writes the results to `destination`. / バッチ内の各行列の 1-ノルム条件数の逆数を推定し、`destination` に書き込みます。 |
| `Tensor<double> Inverse()` | Computes the inverse `A[k]⁻¹` for each matrix in the batch. / バッチ内の各行列の逆行列 `A[k]⁻¹` を計算します。 |
| `TensorSpan<double> Inverse(TensorSpan<double> destination)` | Computes the inverse `A[k]⁻¹` for each matrix in the batch and writes the results to `destination`. / バッチ内の各行列の逆行列 `A[k]⁻¹` を計算し、`destination` に書き込みます。 |
| `bool IsPositiveDefinite(int index)` | Determines whether the matrix at the specified index was numerically positive definite and successfully factored. / 指定したインデックスの行列が数値的に正定値であり、分解に成功したかどうかを返します。 |
| `Vector<double> LogDeterminants()` | Computes the natural logarithm of the determinant, `ln det A[k] = 2 Σ ln lᵢᵢ`, for each matrix in the batch. / バッチ内の各行列の行列式の自然対数 `ln det A[k] = 2 Σ ln lᵢᵢ` を計算します。 |
| `VectorSpan<double> LogDeterminants(VectorSpan<double> destination)` | Computes the natural logarithm of the determinant for each matrix in the batch and writes the results to `destination`. / バッチ内の各行列の行列式の自然対数を計算し、`destination` に書き込みます。 |
| `Matrix<double> Solve(ReadOnlyMatrixSpan<double> b)` | Solves the linear systems `A[k]x[k] = b[k]` for each matrix in the batch with a single right-hand side vector per matrix. / バッチ内の各行列について、単一右辺の連立一次方程式 `A[k]x[k] = b[k]` を解きます。 |
| `Tensor<double> Solve(ReadOnlyTensorSpan<double> b)` | Solves the linear systems `A[k]X[k] = B[k]` for each matrix in the batch with multiple right-hand sides. / バッチ内の各行列について、複数右辺の連立一次方程式 `A[k]X[k] = B[k]` を解きます。 |
| `MatrixSpan<double> Solve(ReadOnlyMatrixSpan<double> b, MatrixSpan<double> destination)` | Solves the linear systems `A[k]x[k] = b[k]` for each matrix in the batch and writes the solutions to `destination`. / バッチ内の各行列について `A[k]x[k] = b[k]` を解き、解を `destination` に書き込みます。 |
| `TensorSpan<double> Solve(ReadOnlyTensorSpan<double> b, TensorSpan<double> destination)` | Solves the linear systems `A[k]X[k] = B[k]` for each matrix in the batch and writes the solutions to `destination`. / バッチ内の各行列について複数右辺の連立一次方程式 `A[k]X[k] = B[k]` を解き、解を `destination` に書き込みます。 |
| `static bool TryCompute(ReadOnlyTensorSpan<double> a, out BatchCholeskyDecomposition result)` | Attempts to compute the batched Cholesky decomposition of the specified symmetric matrices. / 指定した対称行列のバッチに対してコレスキー分解を試みます。 |
| `static bool TryCompute(ReadOnlyTensorSpan<double> a, Tensor<double> destination, out BatchCholeskyDecomposition result)` | Attempts to compute the batched Cholesky decomposition and stores the factor tensor `L` in `destination`. / バッチコレスキー分解を試み、因子テンソル `L` を `destination` に格納します。 |

### `static class Blas`

Provides Basic Linear Algebra Subprograms (BLAS) Level 1, 2, and 3 routines operating directly on `ReadOnlySpan` and `Span` buffers. / `ReadOnlySpan` および `Span` に対して直接動作する BLAS（Level 1 / 2 / 3）基本線形代数ルーチンを提供します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static void Axpy(double alpha, ReadOnlySpan<double> x, Span<double> y)  (+1 overloads)` | Computes `y ← α·x + y` in place (BLAS `daxpy`). / スカラー倍加算 `y ← α·x + y` をインプレースで計算します（BLAS `daxpy`）。 |
| `static void Axpy(Complex alpha, ReadOnlySpan<Complex> x, Span<Complex> y)` | Computes `y ← α·x + y` in place (BLAS `zaxpy`). / スカラー倍加算 `y ← α·x + y` をインプレースで計算します（BLAS `zaxpy`）。 |
| `static void Copy(ReadOnlySpan<double> x, Span<double> y)  (+1 overloads)` | Copies elements from `x` to `y`: `y ← x` (BLAS `dcopy`). / `x` の要素を `y` へコピーします（`y ← x`、BLAS `dcopy`）。 |
| `static void Copy(ReadOnlySpan<Complex> x, Span<Complex> y)` | Copies elements from `x` to `y`: `y ← x` (BLAS `zcopy`). / `x` の要素を `y` へコピーします（`y ← x`、BLAS `zcopy`）。 |
| `static double Dot(ReadOnlySpan<double> x, ReadOnlySpan<double> y)  (+1 overloads)` | Computes the dot product `Σ x[i]·y[i]` (BLAS `ddot`). / 内積 `Σ x[i]·y[i]` を計算します（BLAS `ddot`）。 |
| `static Complex Dot(ReadOnlySpan<Complex> x, ReadOnlySpan<Complex> y)` | Computes the unconjugated complex dot product `Σ x[i]·y[i]` (BLAS `zdotu`). / 共役を取らない複素内積 `Σ x[i]·y[i]` を計算します（BLAS `zdotu`）。 |
| `static Complex DotConjugate(ReadOnlySpan<Complex> x, ReadOnlySpan<Complex> y)` | Computes the conjugate complex dot product `Σ conj(x[i])·y[i]` (BLAS `zdotc`). / 共役内積 `Σ conj(x[i])·y[i]` を計算します（BLAS `zdotc`）。 |
| `static void Gemm(BlasTranspose transA, BlasTranspose transB, int m, int n, int k, float alpha, ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, float beta, Span<float> c, int ldc)` | Computes the general matrix-matrix product `C ← α·op(A)·op(B) + β·C` (BLAS `sgemm`). / 一般行列積 `C ← α·op(A)·op(B) + β·C` を計算します（BLAS `sgemm`）。 |
| `static void Gemm(BlasTranspose transA, BlasTranspose transB, int m, int n, int k, double alpha, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> b, int ldb, double beta, Span<double> c, int ldc)` | Computes the general matrix-matrix product `C ← α·op(A)·op(B) + β·C` (BLAS `dgemm`). / 一般行列積 `C ← α·op(A)·op(B) + β·C` を計算します（BLAS `dgemm`）。 |
| `static void Gemv<T>(BlasTranspose trans, int m, int n, T alpha, ReadOnlySpan<T> a, int lda, ReadOnlySpan<T> x, T beta, Span<T> y)  (+2 overloads)` | Computes the integer matrix-vector product `y ← α·op(A)·x + β·y`. / 整数型の行列ベクトル積 `y ← α·op(A)·x + β·y` を計算します。 |
| `static void Gemv(BlasTranspose trans, int m, int n, Complex alpha, ReadOnlySpan<Complex> a, int lda, ReadOnlySpan<Complex> x, Complex beta, Span<Complex> y)` | Computes the complex matrix-vector product `y ← α·op(A)·x + β·y` (BLAS `zgemv`). / 複素行列ベクトル積 `y ← α·op(A)·x + β·y` を計算します（BLAS `zgemv`）。 |
| `static void Ger<T>(int m, int n, T alpha, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> a, int lda)  (+2 overloads)` | Performs the integer rank-1 update `A ← α·x·yᵀ + A` in place. / 整数型のランク 1 更新 `A ← α·x·yᵀ + A` をインプレースで実行します。 |
| `static void Ger(int m, int n, Complex alpha, ReadOnlySpan<Complex> x, ReadOnlySpan<Complex> y, Span<Complex> a, int lda)` | Performs the unconjugated complex rank-1 update `A ← α·x·yᵀ + A` in place (BLAS `zgeru`). / 共役を取らない複素ランク 1 更新 `A ← α·x·yᵀ + A` をインプレースで実行します（BLAS `zgeru`）。 |
| `static void GerConjugate(int m, int n, Complex alpha, ReadOnlySpan<Complex> x, ReadOnlySpan<Complex> y, Span<Complex> a, int lda)` | Performs the conjugate complex rank-1 update `A ← α·x·yᴴ + A` in place (BLAS `zgerc`). / 共役を取る複素ランク 1 更新 `A ← α·x·yᴴ + A` をインプレースで実行します（BLAS `zgerc`）。 |
| `static void Hemv(BlasTriangle uplo, int n, Complex alpha, ReadOnlySpan<Complex> a, int lda, ReadOnlySpan<Complex> x, Complex beta, Span<Complex> y)` | Computes the Hermitian matrix-vector product `y ← α·A·x + β·y` (BLAS `zhemv`). / エルミート行列ベクトル積 `y ← α·A·x + β·y` を計算します（BLAS `zhemv`）。 |
| `static void Her(BlasTriangle uplo, int n, double alpha, ReadOnlySpan<Complex> x, Span<Complex> a, int lda)` | Performs the Hermitian rank-1 update `A ← α·x·xᴴ + A` in place (BLAS `zher`). / エルミートランク 1 更新 `A ← α·x·xᴴ + A` をインプレースで実行します（BLAS `zher`）。 |
| `static void Her2(BlasTriangle uplo, int n, Complex alpha, ReadOnlySpan<Complex> x, ReadOnlySpan<Complex> y, Span<Complex> a, int lda)` | Performs the Hermitian rank-2 update `A ← α·x·yᴴ + conj(α)·y·xᴴ + A` in place (BLAS `zher2`). / エルミートランク 2 更新 `A ← α·x·yᴴ + conj(α)·y·xᴴ + A` をインプレースで実行します（BLAS `zher2`）。 |
| `static int IndexOfMaxAbs(ReadOnlySpan<double> x)  (+1 overloads)` | Finds the zero-based index of the first element with maximum absolute value (BLAS `idamax`). / 絶対値が最大となる最初の要素の 0 から始まるインデックスを返します（BLAS `idamax`）。 |
| `static int IndexOfMaxAbs(ReadOnlySpan<Complex> x)` | Finds the zero-based index of the first element with maximum `\|Re\| + \|Im\|` value (BLAS `izamax`). / `\|Re\| + \|Im\|` が最大となる最初の要素の 0 から始まるインデックスを返します（BLAS `izamax`）。 |
| `static double Norm2(ReadOnlySpan<double> x)  (+1 overloads)` | Computes the Euclidean norm `‖x‖₂` (BLAS `dnrm2`). / ユークリッドノルム `‖x‖₂` を計算します（BLAS `dnrm2`）。 |
| `static double Norm2(ReadOnlySpan<Complex> x)` | Computes the Euclidean norm `‖x‖₂` of a complex vector (BLAS `dznrm2`). / 複素ベクトルのユークリッドノルム `‖x‖₂` を計算します（BLAS `dznrm2`）。 |
| `static void Rotate(Span<double> x, Span<double> y, double c, double s)  (+1 overloads)` | Applies a plane rotation in place: `(x, y) ← (c·x + s·y, c·y - s·x)` (BLAS `drot`). / 平面回転 `(x, y) ← (c·x + s·y, c·y - s·x)` をインプレースで適用します（BLAS `drot`）。 |
| `static void Rotate(Span<Complex> x, Span<Complex> y, double c, double s)` | Applies a real plane rotation to complex vectors in place: `(x, y) ← (c·x + s·y, c·y - s·x)` (BLAS `zdrot`). / 複素ベクトルに実平面回転 `(x, y) ← (c·x + s·y, c·y - s·x)` をインプレースで適用します（BLAS `zdrot`）。 |
| `static void Scale(double alpha, Span<double> x)  (+1 overloads)` | Scales a vector in place: `x ← α·x` (BLAS `dscal`). / ベクトルをインプレースでスカラー倍します（`x ← α·x`、BLAS `dscal`）。 |
| `static void Scale(Complex alpha, Span<Complex> x)` | Scales a complex vector by a complex scalar in place: `x ← α·x` (BLAS `zscal`). / 複素ベクトルを複素スカラーでインプレースにスカラー倍します（`x ← α·x`、BLAS `zscal`）。 |
| `static void Scale(double alpha, Span<Complex> x)` | Scales a complex vector by a real scalar in place: `x ← α·x` (BLAS `zdscal`). / 複素ベクトルを実数スカラーでインプレースにスカラー倍します（`x ← α·x`、BLAS `zdscal`）。 |
| `static double SumAbs(ReadOnlySpan<double> x)  (+1 overloads)` | Computes the sum of absolute values `Σ \|x[i]\|` (BLAS `dasum`). / 絶対値の和 `Σ \|x[i]\|` を計算します（BLAS `dasum`）。 |
| `static double SumAbs(ReadOnlySpan<Complex> x)` | Computes the sum of real and imaginary magnitudes `Σ (\|Re x[i]\| + \|Im x[i]\|)` (BLAS `dzasum`). / 実部と虚部の絶対値の和 `Σ (\|Re x[i]\| + \|Im x[i]\|)` を計算します（BLAS `dzasum`）。 |
| `static void Swap(Span<double> x, Span<double> y)  (+1 overloads)` | Swaps the elements of `x` and `y` in place (BLAS `dswap`). / `x` と `y` の要素をインプレースで交換します（BLAS `dswap`）。 |
| `static void Swap(Span<Complex> x, Span<Complex> y)` | Swaps the elements of `x` and `y` in place (BLAS `zswap`). / `x` と `y` の要素をインプレースで交換します（BLAS `zswap`）。 |
| `static void Symv<T>(BlasTriangle uplo, int n, T alpha, ReadOnlySpan<T> a, int lda, ReadOnlySpan<T> x, T beta, Span<T> y)  (+2 overloads)` | Computes the integer symmetric matrix-vector product `y ← α·A·x + β·y`. / 整数型の対称行列ベクトル積 `y ← α·A·x + β·y` を計算します。 |
| `static void Symv(BlasTriangle uplo, int n, Complex alpha, ReadOnlySpan<Complex> a, int lda, ReadOnlySpan<Complex> x, Complex beta, Span<Complex> y)` | Computes the complex symmetric matrix-vector product `y ← α·A·x + β·y` (LAPACK `zsymv`). / 複素対称行列ベクトル積 `y ← α·A·x + β·y` を計算します（LAPACK `zsymv`）。 |
| `static void Syr<T>(BlasTriangle uplo, int n, T alpha, ReadOnlySpan<T> x, Span<T> a, int lda)  (+2 overloads)` | Performs the integer symmetric rank-1 update `A ← α·x·xᵀ + A` in place. / 整数型の対称ランク 1 更新 `A ← α·x·xᵀ + A` をインプレースで実行します。 |
| `static void Syr2<T>(BlasTriangle uplo, int n, T alpha, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> a, int lda)  (+2 overloads)` | Performs the integer symmetric rank-2 update `A ← α·x·yᵀ + α·y·xᵀ + A` in place. / 整数型の対称ランク 2 更新 `A ← α·x·yᵀ + α·y·xᵀ + A` をインプレースで実行します。 |
| `static void Trmv<T>(BlasTriangle uplo, BlasTranspose trans, BlasDiagonal diag, int n, ReadOnlySpan<T> a, int lda, Span<T> x)  (+2 overloads)` | Computes the integer triangular matrix-vector product `x ← op(A)·x` in place. / 整数型の三角行列ベクトル積 `x ← op(A)·x` をインプレースで計算します。 |
| `static void Trmv(BlasTriangle uplo, BlasTranspose trans, BlasDiagonal diag, int n, ReadOnlySpan<Complex> a, int lda, Span<Complex> x)` | Computes the complex triangular matrix-vector product `x ← op(A)·x` in place (BLAS `ztrmv`). / 複素三角行列ベクトル積 `x ← op(A)·x` をインプレースで計算します（BLAS `ztrmv`）。 |
| `static void Trsv(BlasTriangle uplo, BlasTranspose trans, BlasDiagonal diag, int n, ReadOnlySpan<double> a, int lda, Span<double> x)  (+1 overloads)` | Solves the triangular linear system `op(A)·x = b` in place (BLAS `dtrsv`). / 三角連立一次方程式 `op(A)·x = b` をインプレースで解きます（BLAS `dtrsv`）。 |
| `static void Trsv(BlasTriangle uplo, BlasTranspose trans, BlasDiagonal diag, int n, ReadOnlySpan<Complex> a, int lda, Span<Complex> x)` | Solves the complex triangular linear system `op(A)·x = b` in place (BLAS `ztrsv`). / 複素三角連立一次方程式 `op(A)·x = b` をインプレースで解きます（BLAS `ztrsv`）。 |

### `enum BlasDiagonal`

Specifies whether a triangular matrix is assumed to have unit diagonal elements (corresponding to the BLAS `DIAG` parameter). / 三角行列の対角要素を 1 とみなすかどうかを指定します（BLAS の `DIAG` 引数に相当）。

- 継承・実装: `IComparable`, `IConvertible`, `IFormattable`, `ISpanFormattable`

| 値 | 説明 |
| --- | --- |
| `NonUnit` | Uses the stored diagonal elements (`'N'`). / 格納されている対角要素の値をそのまま使用します（`'N'`）。 |
| `Unit` | Assumes all diagonal elements are 1 and does not read the stored diagonal entries (`'U'`). / すべての対角要素を 1 とみなし、格納されている対角要素は参照しません（`'U'`）。 |

### `enum BlasTranspose`

Specifies the transpose operation applied to a matrix in BLAS routines (corresponding to the BLAS `TRANS` parameter). / BLAS ルーチンで行列に適用する転置操作を指定します（BLAS の `TRANS` 引数に相当）。

- 継承・実装: `IComparable`, `IConvertible`, `IFormattable`, `ISpanFormattable`

| 値 | 説明 |
| --- | --- |
| `None` | Uses the matrix as-is, `A` (`'N'`). / 行列をそのまま `A` として使用します（`'N'`）。 |
| `Transpose` | Uses the transpose `Aᵀ` (`'T'`). / 転置 `Aᵀ` を使用します（`'T'`）。 |
| `ConjugateTranspose` | Uses the conjugate transpose `Aᴴ` (`'C'`). / 共役転置 `Aᴴ` を使用します（`'C'`）。 |

### `enum BlasTriangle`

Specifies which triangular part of a matrix is referenced in BLAS routines (corresponding to the BLAS `UPLO` parameter). / BLAS ルーチンで参照する行列の三角部分を指定します（BLAS の `UPLO` 引数に相当）。

- 継承・実装: `IComparable`, `IConvertible`, `IFormattable`, `ISpanFormattable`

| 値 | 説明 |
| --- | --- |
| `Upper` | References the upper triangular part (`'U'`). / 上三角部分を参照します（`'U'`）。 |
| `Lower` | References the lower triangular part (`'L'`). / 下三角部分を参照します（`'L'`）。 |

### `class CholeskyDecomposition`

Represents the Cholesky decomposition, `A = LLᵀ`, of a symmetric positive definite matrix. / 対称正定値行列のコレスキー分解 `A = LLᵀ` を表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Determinant { get; }` | Gets the determinant `det A = Π lᵢᵢ²`, or `NaN` if `IsPositiveDefinite` is `false`. / 行列式 `det A = Π lᵢᵢ²` を取得します。`IsPositiveDefinite` が `false` の場合は `NaN` を返します。 |
| `bool IsPositiveDefinite { get; }` | Gets a value indicating whether the decomposition currently holds a valid factorization. / 現在有効な分解を保持しているかどうかを示す値を取得します。 |
| `Matrix<double> L { get; }` | Gets the lower triangular factor `L` with positive diagonal entries. / 対角成分が正の下三角因子 `L` を取得します。 |
| `double LogDeterminant { get; }` | Gets the natural logarithm of the determinant, `ln det A = 2 Σ ln lᵢᵢ`, or `NaN` if `IsPositiveDefinite` is `false`. / 行列式の自然対数 `ln det A = 2 Σ ln lᵢᵢ` を取得します。`IsPositiveDefinite` が `false` の場合は `NaN` を返します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `CholeskyDecomposition Append(ReadOnlyVectorSpan<double> column)` | Returns a new decomposition for the extended matrix `[[A, b], [bᵀ, c]]` obtained by appending one row and column in `O(n²)` time without modifying this instance. / 末尾に 1 行・1 列を追加した拡張行列 `[[A, b], [bᵀ, c]]` のコレスキー分解を `O(n²)` で計算し、新しいインスタンスとして返します（このインスタンスは変更しません）。 |
| `static CholeskyDecomposition Compute(ReadOnlyMatrixSpan<double> a)  (+1 overloads)` | Computes the Cholesky decomposition of the specified symmetric positive definite matrix view. / 指定した対称正定値行列ビューのコレスキー分解を計算します。 |
| `static CholeskyDecomposition Compute(ReadOnlyMatrixSpan<double> a, Matrix<double> destination)  (+1 overloads)` | Computes the Cholesky decomposition of a matrix view and stores the lower triangular factor `L` in `destination`. / 行列ビューのコレスキー分解を計算し、下三角因子 `L` を `destination` に格納します。 |
| `double EstimateReciprocalConditionNumber()` | Estimates the reciprocal 1-norm condition number `1 / (‖A‖₁‖A⁻¹‖₁)` in `O(n²)` time. / 1-ノルム条件数の逆数 `1 / (‖A‖₁‖A⁻¹‖₁)` を `O(n²)` で推定します。 |
| `static CholeskyDecomposition FromFactor(ReadOnlyMatrixSpan<double> factor)` | Creates a `CholeskyDecomposition` by copying the lower triangular entries of an existing Cholesky factor `L`. / 既存の下三角因子 `L`（`A = LLᵀ`）の下三角部分をコピーして `CholeskyDecomposition` を生成します。 |
| `Matrix<double> Inverse()` | Computes the symmetric inverse matrix `A⁻¹ = L⁻ᵀL⁻¹`. / 対称な逆行列 `A⁻¹ = L⁻ᵀL⁻¹` を計算します。 |
| `MatrixSpan<double> Inverse(MatrixSpan<double> destination)` | Computes the symmetric inverse matrix `A⁻¹ = L⁻ᵀL⁻¹` and writes it to `destination`. / 対称な逆行列 `A⁻¹ = L⁻ᵀL⁻¹` を計算し、`destination` に書き込みます。 |
| `void Recompute(ReadOnlyMatrixSpan<double> a)` | Recomputes the Cholesky decomposition for another symmetric matrix `a` of the same order in place, throwing if it is not positive definite. / 同じ次数の別の対称行列 `a` に対してコレスキー分解をインプレースで再計算します。正定値でない場合は例外をスローします。 |
| `double Refine(ReadOnlyMatrixSpan<double> a, ReadOnlyVectorSpan<double> b, VectorSpan<double> x, int maxIterations = 5)` | Improves the solution `x` of `Ax = b` in place using iterative refinement. / 反復改良を用いて連立一次方程式 `Ax = b` の解 `x` をインプレースで高精度化します。 |
| `double Refine(ReadOnlyMatrixSpan<double> a, ReadOnlyMatrixSpan<double> b, MatrixSpan<double> x, int maxIterations = 5)` | Improves the solutions `x` of `AX = B` column by column in place using iterative refinement. / 反復改良を用いて複数右辺の連立一次方程式 `AX = B` の解 `x` を列ごとにインプレースで高精度化します。 |
| `double[] Solve(ReadOnlySpan<double> b)` | Solves the linear system `Ax = b`. / 連立一次方程式 `Ax = b` を解きます。 |
| `Vector<double> Solve(ReadOnlyVectorSpan<double> b)` | Solves the linear system `Ax = b`. / 連立一次方程式 `Ax = b` を解きます。 |
| `Matrix<double> Solve(ReadOnlyMatrixSpan<double> b)` | Solves the linear system `AX = B` with multiple right-hand sides. / 複数右辺の連立一次方程式 `AX = B` を解きます。 |
| `Span<double> Solve(ReadOnlySpan<double> b, Span<double> destination)` | Solves the linear system `Ax = b` and writes the solution to `destination`. / 連立一次方程式 `Ax = b` を解き、解を `destination` に書き込みます。 |
| `VectorSpan<double> Solve(ReadOnlyVectorSpan<double> b, VectorSpan<double> destination)` | Solves the linear system `Ax = b` and writes the solution to `destination`. / 連立一次方程式 `Ax = b` を解き、解を `destination` に書き込みます。 |
| `MatrixSpan<double> Solve(ReadOnlyMatrixSpan<double> b, MatrixSpan<double> destination)` | Solves the linear system `AX = B` with multiple right-hand sides and writes the solutions to `destination`. / 複数右辺の連立一次方程式 `AX = B` を解き、解を `destination` に書き込みます。 |
| `bool TryAppend(ReadOnlyVectorSpan<double> column, out CholeskyDecomposition result)` | Attempts to compute a new decomposition for the extended matrix `[[A, b], [bᵀ, c]]` by appending one row and column in `O(n²)` time. / 末尾に 1 行・1 列を追加した拡張行列 `[[A, b], [bᵀ, c]]` のコレスキー分解を `O(n²)` で試みます。 |
| `static bool TryCompute(ReadOnlyMatrixSpan<double> a, out CholeskyDecomposition result)  (+1 overloads)` | Attempts to compute the Cholesky decomposition of the specified symmetric matrix view. / 指定した対称行列ビューのコレスキー分解を試みます。 |
| `static bool TryCompute(ReadOnlyMatrixSpan<double> a, Matrix<double> destination, out CholeskyDecomposition result)  (+1 overloads)` | Attempts to compute the Cholesky decomposition of a matrix view and stores the lower triangular factor `L` in `destination`. / 行列ビューのコレスキー分解を試み、下三角因子 `L` を `destination` に格納します。 |
| `bool TryDowndate(ReadOnlyVectorSpan<double> x)` | Attempts an in-place rank-1 Cholesky downdate for `A ← A − xxᵀ` in `O(n²)` time. / ランク 1 ダウンデート `A ← A − xxᵀ` に対応する因子へ `L` のインプレース更新を `O(n²)` で試みます。 |
| `bool TryRecompute(ReadOnlyMatrixSpan<double> a)` | Attempts to recompute the Cholesky decomposition for another symmetric matrix `a` of the same order, overwriting `L` in place without allocating. / 同じ次数の別の対称行列 `a` に対してコレスキー分解の再計算を試み、`L` を新規アロケーションなしでインプレース更新します。 |
| `void Update(ReadOnlyVectorSpan<double> x)` | Performs an in-place rank-1 Cholesky update for `A ← A + xxᵀ` in `O(n²)` time. / ランク 1 更新 `A ← A + xxᵀ` に対応する因子へ `L` を `O(n²)` でインプレース更新します。 |

### `static class DenseExtensions`

Provides extension methods for converting arrays and spans to dense containers and invoking BLAS operations and matrix decompositions. / 配列やスパンから密コンテナへの変換、および BLAS 演算や行列分解を呼び出す拡張メソッドを提供します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static CholeskyDecomposition Cholesky(ReadOnlyMatrixSpan<double> matrix)  (+2 overloads)` | Computes the Cholesky decomposition `A = LLᵀ` of the symmetric positive-definite matrix view. / 対称正定値行列ビューのコレスキー分解 `A = LLᵀ` を計算します。 |
| `static double Dot(Vector<double> x, Vector<double> y)  (+1 overloads)` | Computes the dot product of two double-precision vectors. / 2 つの倍精度ベクトルの内積を計算します。 |
| `static Complex Dot(Vector<Complex> x, Vector<Complex> y)` | Computes the unconjugated dot product of two complex vectors. / 2 つの複素ベクトルの共役を取らない内積を計算します。 |
| `static Complex DotConjugate(Vector<Complex> x, Vector<Complex> y)` | Computes the Hermitian dot product `Σ conj(x[i])·y[i]` of two complex vectors. / 2 つの複素ベクトルのエルミート内積 `Σ conj(x[i])·y[i]` を計算します。 |
| `static EigenDecomposition Eigen(ReadOnlyMatrixSpan<double> matrix, bool computeVectors = true)  (+2 overloads)` | Computes the eigendecomposition `AV = VD` of a general real square matrix view. / 一般の実正方行列ビューの固有値分解 `AV = VD` を計算します。 |
| `static double FrobeniusNorm(Matrix<double> matrix)` | Computes the Frobenius norm `‖A‖_F` of the matrix without intermediate overflow or underflow. / 中間のオーバーフローやアンダーフローを回避して、行列のフロベニウスノルム `‖A‖_F` を計算します。 |
| `static LuDecomposition Lu(ReadOnlyMatrixSpan<double> matrix)  (+2 overloads)` | Computes the LU decomposition with partial pivoting `PA = LU` of the square matrix view. / 正方行列ビューの部分ピボット付き LU 分解 `PA = LU` を計算します。 |
| `static double Norm2(Vector<double> x)  (+1 overloads)` | Computes the Euclidean (L2) norm of a double-precision vector. / 倍精度ベクトルのユークリッドノルム（L2 ノルム）を計算します。 |
| `static double Norm2(Vector<Complex> x)` | Computes the Euclidean (L2) norm of a complex vector. / 複素ベクトルのユークリッドノルム（L2 ノルム）を計算します。 |
| `static QrDecomposition Qr(ReadOnlyMatrixSpan<double> matrix)  (+2 overloads)` | Computes the thin QR decomposition `A = QR` of the matrix view. / 行列ビューの薄い QR 分解 `A = QR` を計算します。 |
| `static SingularValueDecomposition Svd(ReadOnlyMatrixSpan<double> matrix, bool computeVectors = true)  (+2 overloads)` | Computes the thin singular value decomposition `A = UΣVᵀ` of the matrix view. / 行列ビューの薄い特異値分解 `A = UΣVᵀ` を計算します。 |
| `static SymmetricEigenDecomposition SymmetricEigen(ReadOnlyMatrixSpan<double> matrix, bool computeVectors = true)  (+2 overloads)` | Computes the eigendecomposition `A = VΛVᵀ` of a real symmetric matrix view. / 実対称行列ビューの固有値分解 `A = VΛVᵀ` を計算します。 |
| `static Matrix<T> ToMatrix<T>(T[,] values)` | Creates a new `Matrix` by copying the elements of the specified two-dimensional array. / 指定した 2 次元配列の要素をコピーして新しい `Matrix` を作成します。 |
| `static Vector<T> ToVector<T>(T[] values)` | Creates a new `Vector` by copying the elements of the specified array. / 指定した配列の要素をコピーして新しい `Vector` を作成します。 |
| `static Vector<T> ToVector<T>(ReadOnlySpan<T> values)` | Creates a new `Vector` by copying the elements of the specified read-only span. / 指定した読み取り専用スパンの要素をコピーして新しい `Vector` を作成します。 |
| `static Vector<T> ToVector<T>(Span<T> values)` | Creates a new `Vector` by copying the elements of the specified span. / 指定したスパンの要素をコピーして新しい `Vector` を作成します。 |

### `class EigenDecomposition`

Represents the eigendecomposition `AV = VD` of a general real square matrix. / 一般の実正方行列の固有値分解 `AV = VD` を表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `Complex[] Values { get; }` | Gets the computed complex eigenvalues (complex conjugate pairs appear consecutively with the positive imaginary part first). / 計算された複素固有値を取得します（複素共役対は虚部が正のものが先に連続して並びます）。 |
| `Matrix<double> Vectors { get; }` | Gets the real-packed eigenvector matrix, or `null` if eigenvectors were not computed. / 実形式で格納された固有ベクトル行列を取得します（固有ベクトルを計算しなかった場合は `null`）。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static EigenDecomposition Compute(ReadOnlyMatrixSpan<double> a, bool computeVectors = true)` | Computes the eigendecomposition of a general real square matrix view. / 一般の実正方行列ビューの固有値分解を計算します。 |
| `static EigenDecomposition Compute(Matrix<double> a, bool computeVectors = true, bool overwriteInput = false)` | Computes the eigendecomposition of a general real square matrix. / 一般の実正方行列の固有値分解を計算します。 |
| `Complex[] GetVector(int index)` | Extracts the complex eigenvector corresponding to the eigenvalue at `index`. / `index` 番目の固有値に対応する複素固有ベクトルを取り出します。 |

### `static class LinearAlgebraSettings`

Provides global execution settings for linear algebra decompositions and kernels. / 線形代数演算および行列分解のグローバルな実行設定を提供します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `static int MaxDegreeOfParallelism { get; set; }` | Gets or sets the maximum number of threads used by a single linear algebra operation. / 1 回の線形代数演算で使用する最大スレッド数を取得または設定します。 |
| `static TwoStageReduction TwoStageReduction { get; set; }` | Gets or sets the policy controlling two-stage reduction (dense to band to tridiagonal or bidiagonal) in symmetric eigendecomposition and SVD. / 対称行列の固有値分解および特異値分解で、二段階縮約（密行列から帯行列を経由して三重対角・二重対角行列へ縮約）を使用する条件を取得または設定します。 |

### `class LuDecomposition`

Represents the LU decomposition with partial pivoting, `PA = LU`, of a square matrix. / 正方行列の部分ピボット付き LU 分解 `PA = LU` を表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double Determinant { get; }` | Gets the determinant `det A = (det P)·Π uᵢᵢ`. / 行列式 `det A = (det P)·Π uᵢᵢ` を取得します。 |
| `int DeterminantSign { get; }` | Gets the sign of the determinant. / 行列式の符号を取得します。 |
| `bool IsSingular { get; }` | Gets a value indicating whether an exact zero pivot was encountered during factorization. / 分解中に厳密に 0 のピボットが発生した（特異である）かどうかを示す値を取得します。 |
| `Matrix<double> L { get; }` | Gets the unit lower triangular factor `L` (`n × n`). / 対角成分が 1 の下三角因子 `L`（`n × n`）を取得します。 |
| `double LogAbsDeterminant { get; }` | Gets the natural logarithm of the absolute determinant, `ln \|det A\| = Σ ln \|uᵢᵢ\|`. / 行列式の絶対値の自然対数 `ln \|det A\| = Σ ln \|uᵢᵢ\|` を取得します。 |
| `int Order { get; }` | Gets the order `n` of the factored matrix. / 分解された行列の次数 `n` を取得します。 |
| `int[] Permutation { get; }` | Gets the row permutation `P` as a new index array. / 行の置換 `P` を新しいインデックス配列として取得します。 |
| `Matrix<double> U { get; }` | Gets the upper triangular factor `U` (`n × n`). / 上三角因子 `U`（`n × n`）を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static LuDecomposition Compute(ReadOnlyMatrixSpan<double> a)` | Computes the LU decomposition with partial pivoting of the specified square matrix view. / 指定した正方行列ビューの部分ピボット付き LU 分解を計算します。 |
| `static LuDecomposition Compute(Matrix<double> a, bool overwriteInput = false)` | Computes the LU decomposition with partial pivoting of the specified square matrix. / 指定した正方行列の部分ピボット付き LU 分解を計算します。 |
| `double EstimateReciprocalConditionNumber()` | Estimates the reciprocal 1-norm condition number `1 / (‖A‖₁‖A⁻¹‖₁)` in `O(n²)` time. / 1-ノルム条件数の逆数 `1 / (‖A‖₁‖A⁻¹‖₁)` を `O(n²)` で推定します。 |
| `Matrix<double> Inverse()` | Computes the inverse matrix `A⁻¹`. / 逆行列 `A⁻¹` を計算します。 |
| `MatrixSpan<double> Inverse(MatrixSpan<double> destination)` | Computes the inverse matrix `A⁻¹` and writes it to `destination`. / 逆行列 `A⁻¹` を計算し、`destination` に書き込みます。 |
| `void Recompute(ReadOnlyMatrixSpan<double> a)` | Recomputes the LU decomposition for another square matrix `a` of the same order, overwriting the existing factor storage in place without allocating. / 同じ次数の別の正方行列 `a` に対して LU 分解を再計算し、既存の因子ストレージを新規アロケーションなしでインプレース更新します。 |
| `double Refine(ReadOnlyMatrixSpan<double> a, ReadOnlyVectorSpan<double> b, VectorSpan<double> x, int maxIterations = 5)` | Improves the solution `x` of `Ax = b` in place using iterative refinement. / 反復改良を用いて連立一次方程式 `Ax = b` の解 `x` をインプレースで高精度化します。 |
| `double Refine(ReadOnlyMatrixSpan<double> a, ReadOnlyMatrixSpan<double> b, MatrixSpan<double> x, int maxIterations = 5)` | Improves the solutions `x` of `AX = B` column by column in place using iterative refinement. / 反復改良を用いて複数右辺の連立一次方程式 `AX = B` の解 `x` を列ごとにインプレースで高精度化します。 |
| `Vector<double> Solve(ReadOnlyVectorSpan<double> b)` | Solves the linear system `Ax = b`. / 連立一次方程式 `Ax = b` を解きます。 |
| `Matrix<double> Solve(ReadOnlyMatrixSpan<double> b)` | Solves the linear system `AX = B` with multiple right-hand sides. / 複数右辺の連立一次方程式 `AX = B` を解きます。 |
| `VectorSpan<double> Solve(ReadOnlyVectorSpan<double> b, VectorSpan<double> destination)` | Solves the linear system `Ax = b` and writes the solution to `destination`. / 連立一次方程式 `Ax = b` を解き、解を `destination` に書き込みます。 |
| `MatrixSpan<double> Solve(ReadOnlyMatrixSpan<double> b, MatrixSpan<double> destination)` | Solves the linear system `AX = B` with multiple right-hand sides and writes the solutions to `destination`. / 複数右辺の連立一次方程式 `AX = B` を解き、解を `destination` に書き込みます。 |
| `Vector<double> SolveTransposed(ReadOnlyVectorSpan<double> b)` | Solves the transposed linear system `Aᵀx = b` using the existing factors. / 既存の因子を用いて転置連立一次方程式 `Aᵀx = b` を解きます。 |
| `Matrix<double> SolveTransposed(ReadOnlyMatrixSpan<double> b)` | Solves the transposed linear system `AᵀX = B` with multiple right-hand sides. / 複数右辺の転置連立一次方程式 `AᵀX = B` を解きます。 |
| `VectorSpan<double> SolveTransposed(ReadOnlyVectorSpan<double> b, VectorSpan<double> destination)` | Solves the transposed linear system `Aᵀx = b` and writes the solution to `destination`. / 転置連立一次方程式 `Aᵀx = b` を解き、解を `destination` に書き込みます。 |
| `MatrixSpan<double> SolveTransposed(ReadOnlyMatrixSpan<double> b, MatrixSpan<double> destination)` | Solves the transposed linear system `AᵀX = B` with multiple right-hand sides and writes the solutions to `destination`. / 複数右辺の転置連立一次方程式 `AᵀX = B` を解き、解を `destination` に書き込みます。 |

### `class Matrix<TElement>`

Represents a dense column-major two-dimensional numerical matrix. / 列優先レイアウトの 2 次元密行列を表します。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `Matrix(TElement[,] values)` | Initializes a new matrix by copying the elements of a two-dimensional array. / 2 次元配列の要素をコピーして新しい行列を初期化します。 |
| `Matrix(int rows, int columns)` | Initializes a new zero-filled matrix with the specified dimensions. / 指定した行数と列数の零行列を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `TElement this[int row, int column] { get; set; }` | Gets or sets the element at the specified row and column. / 指定した行および列の要素を取得または設定します。 |
| `MatrixSpan<TElement> this[Range rows, Range columns] { get; }` | Gets a zero-copy writable view of the specified submatrix. / 指定した範囲の部分行列を参照する、コピーなしの書き込み可能なビューを取得します。 |
| `int Columns { get; }` | Gets the number of columns in the matrix. / 行列の列数を取得します。 |
| `bool IsSquare { get; }` | Gets a value indicating whether the matrix is square (`Rows == Columns`). / 行列が正方行列（`Rows == Columns`）かどうかを示す値を取得します。 |
| `int Length { get; }` | Gets the total number of elements (`Rows * Columns`). / 総要素数（`Rows * Columns`）を取得します。 |
| `int Rows { get; }` | Gets the number of rows in the matrix. / 行列の行数を取得します。 |
| `MatrixSpan<TElement> T { get; }` | Gets a zero-copy writable view of the transpose `Aᵀ` (without complex conjugation). / 転置 `Aᵀ` を参照するコピーなしの書き込み可能なビューを取得します（複素共役は取りません）。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `MatrixSpan<TElement> AsMatrixSpan()` | Returns a writable `MatrixSpan` view over the entire matrix. / 行列全体を参照する書き込み可能な `MatrixSpan` ビューを返します。 |
| `Span<TElement> AsSpan()` | Returns a `Span` sharing the underlying column-major contiguous storage. / 内部の列優先連続格納領域を共有する `Span` を返します。 |
| `Vector<TElement> AsVector()` | Reinterprets this matrix as a vector in column-major order sharing the same underlying storage. / 同じ格納領域を共有する列優先順のベクトルとしてこの行列を再解釈します。 |
| `ReadOnlyTensorSpan<TElement> Broadcast(int count)` | Returns a zero-copy read-only rank-3 tensor view that broadcasts this matrix `count` times along the third axis. / この行列を第 3 軸方向に `count` 個ブロードキャストした、コピーなしの読み取り専用 3 階テンソルビューを返します。 |
| `Matrix<TElement> Clone()` | Creates a deep copy of this matrix with independent storage. / 独立した格納領域を持つこの行列の複製を作成します。 |
| `VectorSpan<TElement> Column(int column)` | Returns a zero-copy contiguous writable view of the specified column. / 指定した列を参照するコピーなしの連続した書き込み可能なビューを返します。 |
| `Matrix<TTo> ConvertChecked<TTo>()` | Converts each element to `TTo` using checked numeric conversion (`CreateChecked`). / チェック付きの数値変換（`CreateChecked`）を用いて各要素を `TTo` に変換します。 |
| `VectorSpan<TElement> Diagonal()` | Returns a zero-copy writable view of the main diagonal (length `min(Rows, Columns)`). / 主対角成分を参照するコピーなしの書き込み可能なビュー（長さ `min(Rows, Columns)`）を返します。 |
| `static Matrix<TElement> FromColumnMajor(int rows, int columns, ReadOnlySpan<TElement> values)` | Creates a new matrix by copying elements stored in column-major order. / 列優先順で格納された要素をコピーして新しい行列を作成します。 |
| `static Matrix<TElement> FromDiagonal(ReadOnlySpan<TElement> values)` | Creates a square diagonal matrix with the specified diagonal entries. / 指定した対角成分を持つ正方対角行列を作成します。 |
| `static Matrix<TElement> FromRowMajor(int rows, int columns, ReadOnlySpan<TElement> values)` | Creates a new column-major matrix by copying and reordering elements stored in row-major order. / 行優先順で格納された要素を列優先順に並べ替えてコピーし、新しい行列を作成します。 |
| `static Matrix<TElement> Generate<TOperator>(int rows, int columns)` | Creates a new matrix whose elements are computed from their indices using a stateless operator: `result[i, j] = TOperator(i, j)`. / 状態を持たない演算子を用いて添字から各要素を計算した新しい行列（`result[i, j] = TOperator(i, j)`）を作成します。 |
| `static Matrix<TElement> Generate<TOperator>(int rows, int columns, TOperator op)` | Creates a new matrix whose elements are computed from their indices using a stateful operator: `result[i, j] = op(i, j)`. / 状態を持つ演算子を用いて添字から各要素を計算した新しい行列（`result[i, j] = op(i, j)`）を作成します。 |
| `static Matrix<TElement> Generate(int rows, int columns, Func<int, int, TElement> op)` | Creates a new matrix whose elements are computed from their indices using a delegate: `result[i, j] = op(i, j)`. / デリゲートを用いて添字から各要素を計算した新しい行列（`result[i, j] = op(i, j)`）を作成します。 |
| `static Matrix<TElement> GenerateBand<TOperator>(int rows, int columns, int lowerBandwidth, int upperBandwidth)` | Creates a dense band matrix by evaluating `TOperator(i, j)` for `-lowerBandwidth <= j - i <= upperBandwidth` and setting elements outside the band to zero. / `-lowerBandwidth <= j - i <= upperBandwidth` の範囲を状態のない演算子 `TOperator(i, j)` で計算し、帯外の要素を 0 とする密行列を作成します。 |
| `static Matrix<TElement> GenerateBand<TOperator>(int rows, int columns, int lowerBandwidth, int upperBandwidth, TOperator op)` | Creates a dense band matrix by evaluating `op(i, j)` for `-lowerBandwidth <= j - i <= upperBandwidth` and setting elements outside the band to zero. / `-lowerBandwidth <= j - i <= upperBandwidth` の範囲を状態を持つ演算子 `op(i, j)` で計算し、帯外の要素を 0 とする密行列を作成します。 |
| `static Matrix<TElement> GenerateBand(int rows, int columns, int lowerBandwidth, int upperBandwidth, Func<int, int, TElement> op)` | Creates a dense band matrix by evaluating `op` inside the band and setting elements outside the band to zero. / 帯内の要素をデリゲート `op` で計算し、帯外の要素を 0 とする密行列を作成します。 |
| `static Matrix<TElement> GenerateSymmetric<TOperator>(int order)` | Creates a symmetric matrix by evaluating a stateless operator `TOperator(i, j)` for the lower triangle (`i >= j`) and mirroring it to the upper triangle. / 下三角部分（`i >= j`）を状態のない演算子 `TOperator(i, j)` で計算し、上三角部分へ対称にコピーした新しい対称行列を作成します。 |
| `static Matrix<TElement> GenerateSymmetric<TOperator>(int order, TOperator op)` | Creates a symmetric matrix by evaluating a stateful operator `op(i, j)` for the lower triangle (`i >= j`) and mirroring it to the upper triangle. / 下三角部分（`i >= j`）を状態を持つ演算子 `op(i, j)` で計算し、上三角部分へ対称にコピーした新しい対称行列を作成します。 |
| `static Matrix<TElement> GenerateSymmetric(int order, Func<int, int, TElement> op)` | Creates a symmetric matrix by evaluating `op` for the lower triangle (`i >= j`) and mirroring it to the upper triangle. / 下三角部分（`i >= j`）をデリゲート `op` で計算し、上三角部分へ対称にコピーした新しい対称行列を作成します。 |
| `static Matrix<TElement> GenerateSymmetric<TOperator>(int order, int bandwidth)` | Creates a symmetric band matrix by evaluating `TOperator(i, j)` for `0 <= i - j <= bandwidth`, mirroring to the upper triangle, and zeroing elements outside the band. / `0 <= i - j <= bandwidth` の範囲を状態のない演算子 `TOperator(i, j)` で計算して上三角へ対称にコピーし、帯外を 0 とする新しい対称帯行列を作成します。 |
| `static Matrix<TElement> GenerateSymmetric<TOperator>(int order, int bandwidth, TOperator op)` | Creates a symmetric band matrix by evaluating `op(i, j)` for `0 <= i - j <= bandwidth`, mirroring to the upper triangle, and zeroing elements outside the band. / `0 <= i - j <= bandwidth` の範囲を状態を持つ演算子 `op(i, j)` で計算して上三角へ対称にコピーし、帯外を 0 とする新しい対称帯行列を作成します。 |
| `static Matrix<TElement> GenerateSymmetric(int order, int bandwidth, Func<int, int, TElement> op)` | Creates a symmetric band matrix by evaluating `op` for the lower half of the band, mirroring to the upper triangle, and zeroing elements outside the band. / 帯の下半分をデリゲート `op` で計算して上三角へ対称にコピーし、帯外を 0 とする新しい対称帯行列を作成します。 |
| `static Matrix<TElement> Identity(int size)` | Creates an identity matrix of the specified order. / 指定した次数の単位行列を作成します。 |
| `VectorSpan<TElement> Row(int row)` | Returns a zero-copy writable view of the specified row (with stride equal to `Rows`). / 指定した行を参照するコピーなしの書き込み可能なビュー（ストライドは `Rows`）を返します。 |
| `TElement[,] ToArray()` | Copies the elements of this matrix to a new two-dimensional array preserving indices. / 添字を維持してこの行列の要素を新しい 2 次元配列へコピーします。 |
| `string ToString()` | Returns a string representation containing the matrix dimensions and up to the leading 8 × 8 elements. / 行列の形状と先頭 8 × 8 までの要素を含む文字列表現を返します。 |
| `Vector<TElement> ToVector()` | Copies the elements of this matrix in column-major order into a new vector. / この行列の要素を列優先順で新しいベクトルへコピーします。 |
| `TensorSpan<TElement> Unsqueeze(int axis)` | Returns a zero-copy writable rank-3 tensor view with a length-1 axis inserted at `axis`. / 位置 `axis` に長さ 1 の軸を挿入した、コピーなしの書き込み可能な 3 階テンソルビューを返します。 |

**演算子**

`Matrix<TElement> operator +(Matrix<TElement> a, Matrix<TElement> b)` / `Matrix<TElement> operator /(Matrix<TElement> a, TElement scalar)` / `explicit operator TElement[,](Matrix<TElement> matrix)` / `explicit operator Matrix<TElement>(TElement[,] values)` / `implicit operator MatrixSpan<TElement>(Matrix<TElement> matrix)` / `implicit operator ReadOnlyMatrixSpan<TElement>(Matrix<TElement> matrix)` / `Matrix<TElement> operator *(Matrix<TElement> a, Matrix<TElement> b)` / `Matrix<TElement> operator *(Matrix<TElement> a, TElement scalar)` / `Matrix<TElement> operator *(TElement scalar, Matrix<TElement> a)` / `Matrix<TElement> operator -(Matrix<TElement> a, Matrix<TElement> b)` / `Matrix<TElement> operator -(Matrix<TElement> a)`

### `readonly ref struct MatrixSpan<TElement>`

Represents a zero-copy, strided writable view over a two-dimensional matrix. / ストライド付きの 2 次元行列を参照する、コピーなしの書き込み可能なビューを表します。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `MatrixSpan(Span<TElement> data, ReadOnlySpan<IntPtr> lengths, ReadOnlySpan<IntPtr> strides)` | Initializes a writable view over the specified span with explicit shape and strides. / 形状とストライドを指定して領域を参照する書き込み可能なビューを初期化します。 |
| `MatrixSpan(Span<TElement> data, int rows, int columns)` | Initializes a writable view over a contiguous column-major span. / 列優先の連続領域を参照する書き込み可能なビューを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `TElement& this[int row, int column] { get; }` | Gets a writable reference to the element at the specified row and column. / 指定した行および列の要素への書き込み可能な参照を取得します。 |
| `MatrixSpan<TElement> this[Range rows, Range columns] { get; }` | Gets a zero-copy writable view of the specified submatrix. / 指定した範囲の部分行列を参照する書き込み可能なビューを取得します。 |
| `int Columns { get; }` | Gets the number of columns in the view. / ビューの列数を取得します。 |
| `Span<TElement> Data { get; }` | Gets a contiguous span from the first element through the last element, including any intervening elements outside the view. / 先頭要素から最終要素までの連続領域を取得します（ストライド間のビュー外要素も含みます）。 |
| `IntPtr FlattenedLength { get; }` | Gets the total number of elements (`Rows * Columns`). / 総要素数（`Rows * Columns`）を取得します。 |
| `bool IsEmpty { get; }` | Gets a value indicating whether the view contains no elements. / ビューが空かどうかを示す値を取得します。 |
| `ReadOnlySpan<IntPtr> Lengths { get; }` | Gets the shape `[Rows, Columns]` of the view. / ビューの形状 `[Rows, Columns]` を取得します。 |
| `int Rank { get; }` | Gets the number of dimensions, which is always 2. / 次元数（常に 2）を取得します。 |
| `int Rows { get; }` | Gets the number of rows in the view. / ビューの行数を取得します。 |
| `ReadOnlySpan<IntPtr> Strides { get; }` | Gets the element strides along each axis (0 for axes of length 0 or 1). / 各軸の要素単位のストライドを取得します（長さが 0 または 1 の軸は 0）。 |
| `MatrixSpan<TElement> T { get; }` | Gets a zero-copy writable view of the transpose `Aᵀ` (swapping strides without copying or conjugating). / 転置 `Aᵀ` の書き込み可能なビューを取得します（ストライドの交換のみでコピーや複素共役は行いません）。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `ReadOnlyMatrixSpan<TElement> AsReadOnly()` | Returns a read-only view over the same elements. / 同じ要素を参照する読み取り専用ビューを返します。 |
| `ReadOnlyTensorSpan<TElement> Broadcast(int count)` | Returns a zero-copy read-only `[Rows, Columns, count]` tensor view repeating this matrix `count` times along the third axis. / この行列を第 3 軸方向に `count` 個繰り返す、コピーなしの読み取り専用 `[Rows, Columns, count]` テンソルビューを返します。 |
| `VectorSpan<TElement> Column(int column)` | Returns a zero-copy writable view of the specified column. / 指定した列を参照する書き込み可能なビューを返します。 |
| `void CopyTo(MatrixSpan<TElement> destination)` | Copies the elements of this view into a destination view of the same shape. / このビューの要素を同じ形状の出力先ビューへコピーします。 |
| `static MatrixSpan<TElement> Create(ref TElement reference, ReadOnlySpan<IntPtr> lengths, ReadOnlySpan<IntPtr> strides)` | Creates a writable matrix view from a reference to the first element, shape, and strides. / 先頭要素への参照・形状・ストライドから書き込み可能な行列ビューを作成します。 |
| `VectorSpan<TElement> Diagonal()` | Returns a zero-copy writable view of the main diagonal (length `min(Rows, Columns)`). / 主対角成分を参照する書き込み可能なビュー（長さ `min(Rows, Columns)`）を返します。 |
| `void Fill(TElement value)` | Fills all elements of the matrix view with the specified value. / 行列ビューのすべての要素を指定した値で埋めます。 |
| `VectorSpan<TElement> Row(int row)` | Returns a zero-copy writable view of the specified row. / 指定した行を参照する書き込み可能なビューを返します。 |
| `TElement[,] ToArray()` | Copies the elements of this view into a new two-dimensional array preserving indices. / 添字を維持してこのビューの要素を新しい 2 次元配列へコピーします。 |
| `Matrix<TElement> ToMatrix()` | Copies the elements of this view into a new `Matrix`. / このビューの要素を新しい `Matrix` へコピーします。 |
| `TensorSpan<TElement> Unsqueeze(int axis)` | Returns a zero-copy writable rank-3 tensor view with a length-1 axis inserted at `axis`. / 位置 `axis` に長さ 1 の軸を挿入した、コピーなしの書き込み可能な 3 階テンソルビューを返します。 |

**演算子**

`Matrix<TElement> operator +(MatrixSpan<TElement> a, MatrixSpan<TElement> b)` / `Matrix<TElement> operator /(MatrixSpan<TElement> a, TElement scalar)` / `implicit operator ReadOnlyMatrixSpan<TElement>(MatrixSpan<TElement> view)` / `Matrix<TElement> operator *(MatrixSpan<TElement> a, MatrixSpan<TElement> b)` / `Matrix<TElement> operator *(MatrixSpan<TElement> a, TElement scalar)` / `Matrix<TElement> operator *(TElement scalar, MatrixSpan<TElement> a)` / `Matrix<TElement> operator -(MatrixSpan<TElement> a, MatrixSpan<TElement> b)` / `Matrix<TElement> operator -(MatrixSpan<TElement> a)`

### `class QrDecomposition`

Represents the thin QR decomposition `A = QR` of a real matrix. / 実行列の薄い QR 分解 `A = QR` を表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Columns { get; }` | Gets the number of columns `n` of the original matrix. / 元の行列の列数 `n` を取得します。 |
| `Matrix<double> Q { get; }` | Gets the `m × min(m, n)` matrix `Q` with orthonormal columns. / 正規直交列を持つ `m × min(m, n)` 行列 `Q` を取得します。 |
| `Matrix<double> R { get; }` | Gets the `min(m, n) × n` upper trapezoidal factor `R`. / `min(m, n) × n` の上台形行列 `R` を取得します。 |
| `int Rows { get; }` | Gets the number of rows `m` of the original matrix. / 元の行列の行数 `m` を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static QrDecomposition Compute(ReadOnlyMatrixSpan<double> a)` | Computes the QR decomposition of the matrix view `a` and immediately materializes `Q` and `R`. / 行列ビュー `a` の QR 分解を計算し、`Q` および `R` を直ちに生成します。 |
| `static QrDecomposition Compute(Matrix<double> a, bool overwriteInput = false)` | Computes the QR decomposition of `a` and immediately materializes `Q` and `R`. / `a` の QR 分解を計算し、`Q` および `R` を直ちに生成します。 |
| `static QrDecomposition ComputeFactors(ReadOnlyMatrixSpan<double> a)` | Computes only the compact Householder factorization of the matrix view `a`, deferring materialization of `Q` and `R` until accessed. / 行列ビュー `a` のコンパクトなハウスホルダー因子のみを計算し、`Q` と `R` の生成を初回アクセス時まで遅延します。 |
| `static QrDecomposition ComputeFactors(Matrix<double> a, bool overwriteInput = false)` | Computes only the compact Householder factorization, deferring materialization of `Q` and `R` until accessed. / コンパクトなハウスホルダー因子のみを計算し、`Q` と `R` の生成を初回アクセス時まで遅延します。 |
| `void Recompute(ReadOnlyMatrixSpan<double> a)` | Recomputes the QR decomposition in place for another matrix `a` of the same dimensions, reusing existing buffers without heap allocation. / 同じ形状の別の行列 `a` を再分解し、既存の領域とオブジェクトを再利用してインプレースで更新します（追加のヒープアロケーションなし）。 |
| `double[] Solve(ReadOnlySpan<double> b)` | Computes the linear least-squares solution `argmin ‖Ax - b‖₂` assuming `Rows >= Columns` and full column rank. / `Rows >= Columns` かつ列フルランクのもとで最小二乗解 `argmin ‖Ax - b‖₂` を計算します。 |
| `Vector<double> Solve(ReadOnlyVectorSpan<double> b)` | Computes the linear least-squares solution `argmin ‖Ax - b‖₂` assuming `Rows >= Columns` and full column rank. / `Rows >= Columns` かつ列フルランクのもとで最小二乗解 `argmin ‖Ax - b‖₂` を計算します。 |
| `Matrix<double> Solve(ReadOnlyMatrixSpan<double> b)` | Computes the linear least-squares solutions `argmin ‖AX - B‖_F` for multiple right-hand sides assuming `Rows >= Columns` and full column rank. / `Rows >= Columns` かつ列フルランクのもとで、複数右辺に対する最小二乗解 `argmin ‖AX - B‖_F` を計算します。 |
| `Span<double> Solve(ReadOnlySpan<double> b, Span<double> destination)` | Computes the linear least-squares solution `argmin ‖Ax - b‖₂` and writes it into `destination`. / 最小二乗解 `argmin ‖Ax - b‖₂` を計算し、`destination` に書き込みます。 |
| `VectorSpan<double> Solve(ReadOnlyVectorSpan<double> b, VectorSpan<double> destination)` | Computes the linear least-squares solution `argmin ‖Ax - b‖₂` and writes it into `destination`. / 最小二乗解 `argmin ‖Ax - b‖₂` を計算し、`destination` に書き込みます。 |
| `MatrixSpan<double> Solve(ReadOnlyMatrixSpan<double> b, MatrixSpan<double> destination)` | Computes the linear least-squares solutions `argmin ‖AX - B‖_F` for multiple right-hand sides and writes them into `destination`. / 複数右辺に対する最小二乗解 `argmin ‖AX - B‖_F` を計算し、`destination` に書き込みます。 |

### `readonly ref struct ReadOnlyMatrixSpan<TElement>`

Represents a zero-copy, strided read-only view over a two-dimensional matrix. / ストライド付きの 2 次元行列を参照する、コピーなしの読み取り専用ビューを表します。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `ReadOnlyMatrixSpan(ReadOnlySpan<TElement> data, ReadOnlySpan<IntPtr> lengths, ReadOnlySpan<IntPtr> strides)` | Initializes a read-only view over the specified span with explicit shape and strides. / 形状とストライドを指定して領域を参照する読み取り専用ビューを初期化します。 |
| `ReadOnlyMatrixSpan(ReadOnlySpan<TElement> data, int rows, int columns)` | Initializes a read-only view over a contiguous column-major span. / 列優先の連続領域を参照する読み取り専用ビューを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `TElement& this[int row, int column] { get; }` | Gets a read-only reference to the element at the specified row and column. / 指定した行および列の要素への読み取り専用参照を取得します。 |
| `ReadOnlyMatrixSpan<TElement> this[Range rows, Range columns] { get; }` | Gets a zero-copy read-only view of the specified submatrix. / 指定した範囲の部分行列を参照する読み取り専用ビューを取得します。 |
| `int Columns { get; }` | Gets the number of columns in the view. / ビューの列数を取得します。 |
| `ReadOnlySpan<TElement> Data { get; }` | Gets a contiguous span from the first element through the last element, including any intervening elements outside the view. / 先頭要素から最終要素までの連続領域を取得します（ストライド間のビュー外要素も含みます）。 |
| `IntPtr FlattenedLength { get; }` | Gets the total number of elements (`Rows * Columns`). / 総要素数（`Rows * Columns`）を取得します。 |
| `bool IsEmpty { get; }` | Gets a value indicating whether the view contains no elements. / ビューが空かどうかを示す値を取得します。 |
| `ReadOnlySpan<IntPtr> Lengths { get; }` | Gets the shape `[Rows, Columns]` of the view. / ビューの形状 `[Rows, Columns]` を取得します。 |
| `int Rank { get; }` | Gets the number of dimensions, which is always 2. / 次元数（常に 2）を取得します。 |
| `int Rows { get; }` | Gets the number of rows in the view. / ビューの行数を取得します。 |
| `ReadOnlySpan<IntPtr> Strides { get; }` | Gets the element strides along each axis (0 for axes of length 0 or 1). / 各軸の要素単位のストライドを取得します（長さが 0 または 1 の軸は 0）。 |
| `ReadOnlyMatrixSpan<TElement> T { get; }` | Gets a zero-copy read-only view of the transpose `Aᵀ` (swapping strides without copying or conjugating). / 転置 `Aᵀ` の読み取り専用ビューを取得します（ストライドの交換のみでコピーや複素共役は行いません）。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `ReadOnlyTensorSpan<TElement> Broadcast(int count)` | Returns a zero-copy read-only `[Rows, Columns, count]` tensor view repeating this matrix `count` times along the third axis. / この行列を第 3 軸方向に `count` 個繰り返す、コピーなしの読み取り専用 `[Rows, Columns, count]` テンソルビューを返します。 |
| `ReadOnlyVectorSpan<TElement> Column(int column)` | Returns a zero-copy read-only view of the specified column. / 指定した列を参照する読み取り専用ビューを返します。 |
| `void CopyTo(MatrixSpan<TElement> destination)` | Copies the elements of this view into a destination view of the same shape. / このビューの要素を同じ形状の出力先ビューへコピーします。 |
| `static ReadOnlyMatrixSpan<TElement> Create(in TElement reference, ReadOnlySpan<IntPtr> lengths, ReadOnlySpan<IntPtr> strides)` | Creates a read-only matrix view from a reference to the first element, shape, and strides. / 先頭要素への参照・形状・ストライドから読み取り専用行列ビューを作成します。 |
| `ReadOnlyVectorSpan<TElement> Diagonal()` | Returns a zero-copy read-only view of the main diagonal (length `min(Rows, Columns)`). / 主対角成分を参照する読み取り専用ビュー（長さ `min(Rows, Columns)`）を返します。 |
| `ReadOnlyVectorSpan<TElement> Row(int row)` | Returns a zero-copy read-only view of the specified row. / 指定した行を参照する読み取り専用ビューを返します。 |
| `TElement[,] ToArray()` | Copies the elements of this view into a new two-dimensional array preserving indices. / 添字を維持してこのビューの要素を新しい 2 次元配列へコピーします。 |
| `Matrix<TElement> ToMatrix()` | Copies the elements of this view into a new `Matrix`. / このビューの要素を新しい `Matrix` へコピーします。 |
| `ReadOnlyTensorSpan<TElement> Unsqueeze(int axis)` | Returns a zero-copy read-only rank-3 tensor view with a length-1 axis inserted at `axis`. / 位置 `axis` に長さ 1 の軸を挿入した、コピーなしの読み取り専用 3 階テンソルビューを返します。 |

**演算子**

`Matrix<TElement> operator +(ReadOnlyMatrixSpan<TElement> a, ReadOnlyMatrixSpan<TElement> b)` / `Matrix<TElement> operator /(ReadOnlyMatrixSpan<TElement> a, TElement scalar)` / `Matrix<TElement> operator *(ReadOnlyMatrixSpan<TElement> a, ReadOnlyMatrixSpan<TElement> b)` / `Matrix<TElement> operator *(ReadOnlyMatrixSpan<TElement> a, TElement scalar)` / `Matrix<TElement> operator *(TElement scalar, ReadOnlyMatrixSpan<TElement> a)` / `Matrix<TElement> operator -(ReadOnlyMatrixSpan<TElement> a, ReadOnlyMatrixSpan<TElement> b)` / `Matrix<TElement> operator -(ReadOnlyMatrixSpan<TElement> a)`

### `readonly ref struct ReadOnlyTensorMatrices<TElement>`

Provides indexed read-only access and allocation-free enumeration over the matrix slices of a tensor. / テンソル内の各行列スライスに対するインデックスアクセスと、アロケーションなしの列挙を提供する読み取り専用ビューです。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `ReadOnlyMatrixSpan<TElement> this[int index] { get; }` | Gets a read-only view of the matrix at the specified index. / 指定したインデックスにある行列の読み取り専用ビューを取得します。 |
| `int Count { get; }` | Gets the number of matrix slices in the tensor. / テンソルに含まれる行列の個数を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `Enumerator<TElement> GetEnumerator()` | Returns an enumerator that iterates through the matrix slices in index order. / 行列スライスをインデックス順に反復処理する列挙子を返します。 |

#### `ref struct ReadOnlyTensorMatrices<TElement>.Enumerator`

Enumerates the read-only matrix slices of a tensor without allocation. / テンソルの読み取り専用行列スライスをアロケーションなしで列挙する列挙子です。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `ReadOnlyMatrixSpan<TElement> Current { get; }` | Gets the matrix slice at the current position of the enumerator. / 列挙子の現在位置にある行列スライスを取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `bool MoveNext()` | Advances the enumerator to the next matrix slice. / 列挙子を次の行列スライスへ進めます。 |

### `readonly ref struct ReadOnlyTensorSpan<TElement>`

Represents a zero-copy, strided read-only view over a rank-3 tensor. / ストライド付きの 3 階テンソルを参照する、コピーなしの読み取り専用ビューを表します。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `ReadOnlyTensorSpan(ReadOnlySpan<TElement> data, ReadOnlySpan<IntPtr> lengths)` | Initializes a read-only view over a contiguous first-index-fastest span. / 先頭軸が最も速く変化する順の連続領域を参照する読み取り専用ビューを初期化します。 |
| `ReadOnlyTensorSpan(ReadOnlySpan<TElement> data, ReadOnlySpan<IntPtr> lengths, ReadOnlySpan<IntPtr> strides)` | Initializes a read-only view over the specified span with explicit shape and strides. / 形状とストライドを指定して領域を参照する読み取り専用ビューを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `TElement& this[int row, int column, int index] { get; }` | Gets a read-only reference to element `(row, column)` of matrix `index`. / 第 `index` 行列の要素 `(row, column)` への読み取り専用参照を取得します。 |
| `ReadOnlyTensorSpan<TElement> this[Range rows, Range columns, Range indices] { get; }` | Gets a zero-copy read-only view over the specified subrange of rows, columns, and matrix indices. / 指定した行・列・行列インデックスの範囲を参照する読み取り専用ビューを取得します。 |
| `ReadOnlySpan<TElement> Data { get; }` | Gets a contiguous span from the first element through the last element, including any intervening elements outside the view. / 先頭要素から最終要素までの連続領域を取得します（ストライド間のビュー外要素も含みます）。 |
| `IntPtr FlattenedLength { get; }` | Gets the total number of elements (the product of `Lengths`). / 総要素数（`Lengths` の積）を取得します。 |
| `bool IsEmpty { get; }` | Gets a value indicating whether the view contains no elements. / ビューが空かどうかを示す値を取得します。 |
| `ReadOnlySpan<IntPtr> Lengths { get; }` | Gets the shape `[rows, columns, matrixCount]` of the view. / ビューの形状 `[行数, 列数, 行列数]` を取得します。 |
| `ReadOnlyTensorMatrices<TElement> Matrices { get; }` | Gets an indexed, allocation-free enumerable view over all matrix slices. / すべての行列スライスにインデックスアクセスおよび列挙できる読み取り専用ビューを取得します。 |
| `int Rank { get; }` | Gets the rank (number of dimensions) of the view, which is currently always 3. / ビューの階数（次元数、現在は常に 3）を取得します。 |
| `ReadOnlySpan<IntPtr> Strides { get; }` | Gets the element strides along each axis (0 for axes of length 0 or 1). / 各軸の要素単位のストライドを取得します（長さが 0 または 1 の軸は 0）。 |
| `ReadOnlyTensorSpan<TElement> T { get; }` | Gets a zero-copy read-only view with each matrix slice transposed: `B[i, j, k] = A[j, i, k]` (swaps axes 0 and 1 without complex conjugation). / 各行列スライスを転置した読み取り専用ビュー（`B[i, j, k] = A[j, i, k]`、第 0・第 1 軸のみ交換し複素共役は取りません）を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `ReadOnlyMatrixSpan<TElement> AsMatrix()` | Returns a zero-copy read-only `Lengths[0] × (Lengths[1] * Lengths[2])` matrix view placing all matrices side by side: `[A₀ A₁ …]`. / 各行列を横に並べた `[A₀ A₁ …]` 形式の、コピーなしの読み取り専用 `Lengths[0] × (Lengths[1] * Lengths[2])` 行列ビューを返します。 |
| `void CopyTo(TensorSpan<TElement> destination)` | Copies the elements of this view into a destination view of the same shape. / このビューの要素を同じ形状の出力先ビューへコピーします。 |
| `static ReadOnlyTensorSpan<TElement> Create(in TElement reference, ReadOnlySpan<IntPtr> lengths, ReadOnlySpan<IntPtr> strides)` | Creates a read-only tensor view from a reference to the first element, shape, and strides. / 先頭要素への参照・形状・ストライドから読み取り専用テンソルビューを作成します。 |
| `ReadOnlyMatrixSpan<TElement> Diagonals()` | Returns a zero-copy read-only `min(Lengths[0], Lengths[1]) × Lengths[2]` matrix view whose column `k` is the main diagonal of matrix `k`. / 第 `k` 列が第 `k` 行列の主対角成分となる、コピーなしの読み取り専用 `min(Lengths[0], Lengths[1]) × Lengths[2]` 行列ビューを返します。 |
| `ReadOnlyMatrixSpan<TElement> GetMatrix(int index)` | Returns a zero-copy read-only view of matrix `index` (`Lengths[0] × Lengths[1]`). / 第 `index` 行列（`Lengths[0] × Lengths[1]`）を参照する読み取り専用ビューを返します。 |
| `ReadOnlyVectorSpan<TElement> GetVector(int column, int index)` | Returns a zero-copy read-only view of column `column` of matrix `index`. / 第 `index` 行列の第 `column` 列を参照する読み取り専用ビューを返します。 |
| `TElement[,,] ToArray()` | Copies the elements of this view into a new three-dimensional array preserving `[row, column, index]` indices. / `[row, column, index]` の添字を維持してこのビューの要素を新しい 3 次元配列へコピーします。 |
| `Tensor<TElement> ToTensor()` | Copies the elements of this view into a new `Tensor`. / このビューの要素を新しい `Tensor` へコピーします。 |

**演算子**

`Tensor<TElement> operator +(ReadOnlyTensorSpan<TElement> a, ReadOnlyTensorSpan<TElement> b)` / `Tensor<TElement> operator /(ReadOnlyTensorSpan<TElement> a, TElement scalar)` / `Tensor<TElement> operator *(ReadOnlyTensorSpan<TElement> a, ReadOnlyTensorSpan<TElement> b)` / `Tensor<TElement> operator *(ReadOnlyTensorSpan<TElement> a, TElement scalar)` / `Tensor<TElement> operator *(TElement scalar, ReadOnlyTensorSpan<TElement> a)` / `Tensor<TElement> operator -(ReadOnlyTensorSpan<TElement> a, ReadOnlyTensorSpan<TElement> b)` / `Tensor<TElement> operator -(ReadOnlyTensorSpan<TElement> a)`

### `readonly ref struct ReadOnlyVectorSpan<T>`

Represents a zero-copy, strided read-only view over a one-dimensional vector. / ストライド付きの 1 次元ベクトルを参照する、コピーなしの読み取り専用ビューを表します。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `ReadOnlyVectorSpan(ReadOnlySpan<T> data)` | Initializes a read-only view over a contiguous span of elements. / 連続した領域を参照する読み取り専用ビューを初期化します。 |
| `ReadOnlyVectorSpan(ReadOnlySpan<T> data, ReadOnlySpan<IntPtr> lengths, ReadOnlySpan<IntPtr> strides)` | Initializes a read-only view over the specified span with explicit length and stride. / 長さとストライドを指定して領域を参照する読み取り専用ビューを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `T& this[int index] { get; }` | Gets a read-only reference to the element at the specified index. / 指定した位置の要素への読み取り専用参照を取得します。 |
| `ReadOnlyVectorSpan<T> this[Range range] { get; }` | Gets a zero-copy read-only view of the specified subrange. / 指定した範囲の部分ベクトルを参照する読み取り専用ビューを取得します。 |
| `ReadOnlySpan<T> Data { get; }` | Gets a contiguous span from the first element through the last element, including any intervening elements outside the view. / 先頭要素から最終要素までの連続領域を取得します（ストライド間のビュー外要素も含みます）。 |
| `IntPtr FlattenedLength { get; }` | Gets the total number of elements. / 総要素数を取得します。 |
| `bool IsEmpty { get; }` | Gets a value indicating whether the view contains no elements. / ビューが空かどうかを示す値を取得します。 |
| `int Length { get; }` | Gets the number of elements in the vector view. / ベクトルビューの要素数を取得します。 |
| `ReadOnlySpan<IntPtr> Lengths { get; }` | Gets the shape `[Length]` of the view. / ビューの形状 `[Length]` を取得します。 |
| `int Rank { get; }` | Gets the number of dimensions, which is always 1. / 次元数（常に 1）を取得します。 |
| `ReadOnlySpan<IntPtr> Strides { get; }` | Gets the element stride of the view (0 when `Length` is 0 or 1). / 要素単位のストライドを取得します（`Length` が 0 または 1 の場合は 0）。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `ReadOnlyMatrixSpan<T> Broadcast(int count)` | Returns a zero-copy read-only `Length × count` matrix view repeating this vector across `count` columns. / このベクトルを `count` 本の列として繰り返す、コピーなしの読み取り専用 `Length × count` 行列ビューを返します。 |
| `ReadOnlyMatrixSpan<T> BroadcastRows(int rows)` | Returns a zero-copy read-only `rows × Length` matrix view repeating this vector across `rows` rows. / このベクトルを `rows` 本の行として繰り返す、コピーなしの読み取り専用 `rows × Length` 行列ビューを返します。 |
| `void CopyTo(VectorSpan<T> destination)` | Copies the elements of this view into a destination view of the same length. / このビューの要素を同じ長さの出力先ビューへコピーします。 |
| `static ReadOnlyVectorSpan<T> Create(in T reference, ReadOnlySpan<IntPtr> lengths, ReadOnlySpan<IntPtr> strides)` | Creates a read-only view from a reference to the first element, length, and stride. / 先頭要素への参照・長さ・ストライドから読み取り専用ビューを作成します。 |
| `T[] ToArray()` | Copies the elements of this view into a new array. / このビューの要素を新しい配列へコピーします。 |
| `Vector<T> ToVector()` | Copies the elements of this view into a new `Vector`. / このビューの要素を新しい `Vector` へコピーします。 |

**演算子**

`Vector<T> operator +(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)` / `Vector<T> operator /(ReadOnlyVectorSpan<T> x, T scalar)` / `implicit operator ReadOnlyVectorSpan<T>(ReadOnlySpan<T> data)` / `Vector<T> operator *(ReadOnlyMatrixSpan<T> a, ReadOnlyVectorSpan<T> x)` / `Vector<T> operator *(ReadOnlyVectorSpan<T> x, T scalar)` / `Vector<T> operator *(T scalar, ReadOnlyVectorSpan<T> x)` / `Vector<T> operator -(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)` / `Vector<T> operator -(ReadOnlyVectorSpan<T> x)`

### `class SingularValueDecomposition`

Represents the thin singular value decomposition `A = UΣVᵀ` of a real matrix. / 実行列の薄い特異値分解 `A = UΣVᵀ` を表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double ConditionNumber { get; }` | Gets the 2-norm condition number `σ_max / σ_min` (`NaN` for empty matrices). / 2-ノルム条件数 `σ_max / σ_min` を取得します（空行列の場合は `NaN`）。 |
| `int Rank { get; }` | Gets the numerical rank, defined as the count of singular values strictly greater than `max(m, n)·ε·σ_max`. / 数値ランク（`σᵢ > max(m, n)·ε·σ_max` を満たす特異値の個数）を取得します。 |
| `Matrix<double> U { get; }` | Gets the `m × min(m, n)` matrix `U` of left singular vectors, or `null` if singular vectors were not computed. / 左特異ベクトルを列とする `m × min(m, n)` 行列 `U` を取得します（特異ベクトルを計算しなかった場合は `null`）。 |
| `Matrix<double> V { get; }` | Gets the `n × min(m, n)` matrix `V` of right singular vectors, or `null` if singular vectors were not computed. / 右特異ベクトルを列とする `n × min(m, n)` 行列 `V` を取得します（特異ベクトルを計算しなかった場合は `null`）。 |
| `double[] Values { get; }` | Gets the non-negative singular values in descending order. / 降順に並んだ非負の特異値を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static SingularValueDecomposition Compute(ReadOnlyMatrixSpan<double> a, bool computeVectors = true)` | Computes the thin singular value decomposition of the matrix view `a`. / 行列ビュー `a` の薄い特異値分解を計算します。 |
| `static SingularValueDecomposition Compute(Matrix<double> a, bool computeVectors = true, bool overwriteInput = false)` | Computes the thin singular value decomposition of `a`. / `a` の薄い特異値分解を計算します。 |
| `Matrix<double> PseudoInverse(double? tolerance = default)` | Returns the Moore–Penrose pseudoinverse computed from the retained singular values in a new matrix. / 保持した特異値からムーア・ペンローズ疑似逆行列を求め、新しい行列として返します。 |
| `MatrixSpan<double> PseudoInverse(MatrixSpan<double> destination, double? tolerance = default)` | Stores the Moore–Penrose pseudoinverse computed from the retained singular values in the specified matrix view. / 保持した特異値からムーア・ペンローズ疑似逆行列を求め、指定した行列ビューに格納します。 |
| `void Recompute(ReadOnlyMatrixSpan<double> a)` | Recomputes the singular value decomposition for another matrix `a` of the same dimensions, overwriting `Values`, `U`, and `V` in place. / 同じ形状の別の行列 `a` を再分解し、`Values`・`U`・`V` をインプレースで更新します。 |
| `Vector<double> Solve(ReadOnlyVectorSpan<double> b, double? tolerance = default)` | Returns the minimum-norm least-squares solution to `Ax = b` in a new vector. / `Ax = b` の最小二乗解のうち、ノルムが最小の解を新しいベクトルとして返します。 |
| `Matrix<double> Solve(ReadOnlyMatrixSpan<double> b, double? tolerance = default)` | Returns the minimum-norm least-squares solutions for the columns of `b` in a new matrix. / `b` の列ごとに最小ノルム最小二乗解を求め、新しい行列として返します。 |
| `VectorSpan<double> Solve(ReadOnlyVectorSpan<double> b, VectorSpan<double> destination, double? tolerance = default)` | Stores the minimum-norm least-squares solution to `Ax = b` in the specified vector view. / `Ax = b` の最小ノルム最小二乗解を指定したベクトルビューに格納します。 |
| `MatrixSpan<double> Solve(ReadOnlyMatrixSpan<double> b, MatrixSpan<double> destination, double? tolerance = default)` | Stores the minimum-norm least-squares solutions for the columns of `b` in the specified matrix view. / `b` の列ごとに最小ノルム最小二乗解を求め、指定した行列ビューに格納します。 |

### `class SymmetricEigenDecomposition`

Represents the eigendecomposition `A = VΛVᵀ` of a real symmetric matrix. / 実対称行列の固有値分解 `A = VΛVᵀ` を表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double[] Values { get; }` | Gets the eigenvalues in ascending order. / 昇順に並んだ固有値を取得します。 |
| `Matrix<double> Vectors { get; }` | Gets the orthogonal matrix `V` whose columns are the eigenvectors, or `null` if eigenvectors were not computed. / 固有ベクトルを列とする直交行列 `V` を取得します（固有ベクトルを計算しなかった場合は `null`）。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static SymmetricEigenDecomposition Compute(ReadOnlyMatrixSpan<double> a, bool computeVectors = true)` | Computes the eigendecomposition of a real symmetric matrix view. / 実対称行列ビューの固有値分解を計算します。 |
| `static SymmetricEigenDecomposition Compute(Matrix<double> a, bool computeVectors = true, bool overwriteInput = false)` | Computes the eigendecomposition of a real symmetric matrix. / 実対称行列の固有値分解を計算します。 |
| `void Recompute(ReadOnlyMatrixSpan<double> a)` | Recomputes the eigendecomposition for another symmetric matrix `a` of the same order, overwriting `Values` and `Vectors` in place. / 同じ次数の別の対称行列 `a` を再分解し、`Values` と `Vectors` をインプレースで更新します。 |

### `class Tensor<TElement>`

Represents a dense rank-3 multidimensional numerical array storing a batch of column-major matrices. / 列優先行列のバッチを格納する 3 階の密多次元配列（テンソル）を表します。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `Tensor(ReadOnlySpan<IntPtr> lengths)` | Initializes a new zero-filled tensor with the specified shape. / 指定した形状の零テンソルを初期化します。 |
| `Tensor(TElement[,,] values)` | Initializes a new rank-3 tensor by copying the elements of a three-dimensional array indexed as `[row, column, index]`. / `[row, column, index]` 順の 3 次元配列の要素をコピーして新しい 3 階テンソルを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `TElement this[int row, int column, int index] { get; set; }` | Gets or sets element `(row, column)` of matrix `index`. / 第 `index` 行列の要素 `(row, column)` を取得または設定します。 |
| `TensorSpan<TElement> this[Range rows, Range columns, Range indices] { get; }` | Gets a zero-copy writable view over the specified subrange of rows, columns, and matrix indices. / 指定した行・列・行列インデックスの範囲を参照する、コピーなしの書き込み可能なビューを取得します。 |
| `IntPtr FlattenedLength { get; }` | Gets the total number of elements (the product of `Lengths`). / 総要素数（`Lengths` の積）を取得します。 |
| `bool IsEmpty { get; }` | Gets a value indicating whether the tensor contains no elements. / テンソルが空かどうかを示す値を取得します。 |
| `ReadOnlySpan<IntPtr> Lengths { get; }` | Gets the shape `[rows, columns, matrixCount]` of the tensor. / テンソルの形状 `[行数, 列数, 行列数]` を取得します。 |
| `TensorMatrices<TElement> Matrices { get; }` | Gets an indexed, allocation-free enumerable view over all matrix slices in this tensor. / このテンソル内のすべての行列スライスにインデックスアクセスおよび列挙できるビューを取得します。 |
| `int Rank { get; }` | Gets the rank (number of dimensions) of the tensor, which is currently always 3. / テンソルの階数（次元数、現在は常に 3）を取得します。 |
| `ReadOnlySpan<IntPtr> Strides { get; }` | Gets the element strides along each axis (0 for axes of length 0 or 1). / 各軸の要素単位のストライドを取得します（長さが 0 または 1 の軸は 0）。 |
| `TensorSpan<TElement> T { get; }` | Gets a zero-copy writable view with each matrix slice transposed: `B[i, j, k] = A[j, i, k]` (swaps axes 0 and 1 without complex conjugation). / 各行列スライスを転置したコピーなしの書き込み可能なビュー（`B[i, j, k] = A[j, i, k]`、第 0・第 1 軸のみ交換し複素共役は取りません）を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `MatrixSpan<TElement> AsMatrix()` | Returns a zero-copy writable `Lengths[0] × (Lengths[1] * Lengths[2])` matrix view placing all matrices side by side: `[A₀ A₁ …]`. / 各行列を横に並べた `[A₀ A₁ …]` 形式の、コピーなしの書き込み可能な `Lengths[0] × (Lengths[1] * Lengths[2])` 行列ビューを返します。 |
| `Span<TElement> AsSpan()` | Returns a `Span` sharing the underlying contiguous first-index-fastest storage. / 先頭軸が最も速く変化する順の内部連続格納領域を共有する `Span` を返します。 |
| `TensorSpan<TElement> AsTensorSpan()` | Returns a writable `TensorSpan` view over the entire tensor. / テンソル全体を参照する書き込み可能な `TensorSpan` ビューを返します。 |
| `Tensor<TElement> Clone()` | Creates a deep copy of this tensor with independent storage. / 独立した格納領域を持つこのテンソルの複製を作成します。 |
| `Tensor<TTo> ConvertChecked<TTo>()` | Converts each element to `TTo` using checked numeric conversion (`CreateChecked`). / チェック付きの数値変換（`CreateChecked`）を用いて各要素を `TTo` に変換します。 |
| `MatrixSpan<TElement> Diagonals()` | Returns a zero-copy writable `min(Lengths[0], Lengths[1]) × Lengths[2]` matrix view whose column `k` is the main diagonal of matrix `k`. / 第 `k` 列が第 `k` 行列の主対角成分となる、コピーなしの書き込み可能な `min(Lengths[0], Lengths[1]) × Lengths[2]` 行列ビューを返します。 |
| `static Tensor<TElement> FromColumnMajor(ReadOnlySpan<IntPtr> lengths, ReadOnlySpan<TElement> values)` | Creates a new tensor by copying elements stored in first-index-fastest (column-major per matrix) order. / 先頭軸が最も速く変化する順（各行列が列優先で番号順に並んだ順）の要素をコピーして新しいテンソルを作成します。 |
| `static Tensor<TElement> FromMatrices(ReadOnlySpan<Matrix<TElement>> matrices)` | Creates a new rank-3 tensor by stacking matrices of identical shape along the third axis. / 同じ形状の行列を第 3 軸に沿って順にコピーし、新しい 3 階テンソルを作成します。 |
| `static Tensor<TElement> Generate<TOperator>(ReadOnlySpan<IntPtr> lengths)` | Creates a new tensor whose elements are computed from their multi-axis indices using a stateless operator. / 状態を持たない演算子を用いて全軸の添字から各要素を計算した新しいテンソルを作成します。 |
| `static Tensor<TElement> Generate<TOperator>(ReadOnlySpan<IntPtr> lengths, TOperator op)` | Creates a new tensor whose elements are computed from their multi-axis indices using a stateful operator. / 状態を持つ演算子を用いて全軸の添字から各要素を計算した新しいテンソルを作成します。 |
| `static Tensor<TElement> Generate(ReadOnlySpan<IntPtr> lengths, Func<ReadOnlySpan<int>, TElement> op)` | Creates a new tensor whose elements are computed from their multi-axis indices using a delegate. / デリゲートを用いて全軸の添字から各要素を計算した新しいテンソルを作成します。 |
| `MatrixSpan<TElement> GetMatrix(int index)` | Returns a zero-copy contiguous writable view of the matrix at `index`. / 第 `index` 行列（`Lengths[0] × Lengths[1]` の連続領域）を参照する書き込み可能なビューを返します。 |
| `VectorSpan<TElement> GetVector(int column, int index)` | Returns a zero-copy contiguous writable view of column `column` of matrix `index`. / 第 `index` 行列の第 `column` 列（長さ `Lengths[0]` の連続領域）を参照する書き込み可能なビューを返します。 |
| `static Tensor<TElement> Identity(int size, int count)` | Creates a rank-3 tensor consisting of `count` identity matrices of order `size`. / `size` 次の単位行列を `count` 個並べた 3 階テンソルを作成します。 |
| `TElement[,,] ToArray()` | Copies the elements of this tensor into a new three-dimensional array preserving `[row, column, index]` indices. / `[row, column, index]` の添字を維持してこのテンソルの要素を新しい 3 次元配列へコピーします。 |
| `string ToString()` | Returns a string representation containing the element type and tensor shape. / 要素型とテンソルの形状を表す文字列を返します。 |

**演算子**

`Tensor<TElement> operator +(Tensor<TElement> a, Tensor<TElement> b)` / `Tensor<TElement> operator /(Tensor<TElement> a, TElement scalar)` / `implicit operator TensorSpan<TElement>(Tensor<TElement> tensor)` / `implicit operator ReadOnlyTensorSpan<TElement>(Tensor<TElement> tensor)` / `Tensor<TElement> operator *(TElement scalar, Tensor<TElement> a)` / `Tensor<TElement> operator *(Tensor<TElement> a, TElement scalar)` / `Tensor<TElement> operator *(Tensor<TElement> a, Tensor<TElement> b)` / `Tensor<TElement> operator -(Tensor<TElement> a, Tensor<TElement> b)` / `Tensor<TElement> operator -(Tensor<TElement> a)`

### `readonly ref struct TensorMatrices<TElement>`

Provides indexed writable access and allocation-free enumeration over the matrix slices of a tensor. / テンソル内の各行列スライスに対する書き込み可能なインデックスアクセスと、アロケーションなしの列挙を提供するビューです。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `MatrixSpan<TElement> this[int index] { get; }` | Gets a writable view of the matrix at the specified index. / 指定したインデックスにある行列の書き込み可能なビューを取得します。 |
| `int Count { get; }` | Gets the number of matrix slices in the tensor. / テンソルに含まれる行列の個数を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `ReadOnlyTensorMatrices<TElement> AsReadOnly()` | Returns a read-only view over the same matrix slices. / 同じ行列スライスを参照する読み取り専用ビューを返します。 |
| `Enumerator<TElement> GetEnumerator()` | Returns an enumerator that iterates through the matrix slices in index order. / 行列スライスをインデックス順に反復処理する列挙子を返します。 |

**演算子**

`implicit operator ReadOnlyTensorMatrices<TElement>(TensorMatrices<TElement> matrices)`

#### `ref struct TensorMatrices<TElement>.Enumerator`

Enumerates the writable matrix slices of a tensor without allocation. / テンソルの書き込み可能な行列スライスをアロケーションなしで列挙する列挙子です。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `MatrixSpan<TElement> Current { get; }` | Gets the matrix slice at the current position of the enumerator. / 列挙子の現在位置にある行列スライスを取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `bool MoveNext()` | Advances the enumerator to the next matrix slice. / 列挙子を次の行列スライスへ進めます。 |

### `readonly ref struct TensorSpan<TElement>`

Represents a zero-copy, strided writable view over a rank-3 tensor. / ストライド付きの 3 階テンソルを参照する、コピーなしの書き込み可能なビューを表します。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `TensorSpan(Span<TElement> data, ReadOnlySpan<IntPtr> lengths)` | Initializes a writable view over a contiguous first-index-fastest span. / 先頭軸が最も速く変化する順の連続領域を参照する書き込み可能なビューを初期化します。 |
| `TensorSpan(Span<TElement> data, ReadOnlySpan<IntPtr> lengths, ReadOnlySpan<IntPtr> strides)` | Initializes a writable view over the specified span with explicit shape and strides. / 形状とストライドを指定して領域を参照する書き込み可能なビューを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `TElement& this[int row, int column, int index] { get; }` | Gets a writable reference to element `(row, column)` of matrix `index`. / 第 `index` 行列の要素 `(row, column)` への書き込み可能な参照を取得します。 |
| `TensorSpan<TElement> this[Range rows, Range columns, Range indices] { get; }` | Gets a zero-copy writable view over the specified subrange of rows, columns, and matrix indices. / 指定した行・列・行列インデックスの範囲を参照する書き込み可能なビューを取得します。 |
| `Span<TElement> Data { get; }` | Gets a contiguous span from the first element through the last element, including any intervening elements outside the view. / 先頭要素から最終要素までの連続領域を取得します（ストライド間のビュー外要素も含みます）。 |
| `IntPtr FlattenedLength { get; }` | Gets the total number of elements (the product of `Lengths`). / 総要素数（`Lengths` の積）を取得します。 |
| `bool IsEmpty { get; }` | Gets a value indicating whether the view contains no elements. / ビューが空かどうかを示す値を取得します。 |
| `ReadOnlySpan<IntPtr> Lengths { get; }` | Gets the shape `[rows, columns, matrixCount]` of the view. / ビューの形状 `[行数, 列数, 行列数]` を取得します。 |
| `TensorMatrices<TElement> Matrices { get; }` | Gets an indexed, allocation-free enumerable view over all matrix slices. / すべての行列スライスにインデックスアクセスおよび列挙できる書き込み可能なビューを取得します。 |
| `int Rank { get; }` | Gets the rank (number of dimensions) of the view, which is currently always 3. / ビューの階数（次元数、現在は常に 3）を取得します。 |
| `ReadOnlySpan<IntPtr> Strides { get; }` | Gets the element strides along each axis (0 for axes of length 0 or 1). / 各軸の要素単位のストライドを取得します（長さが 0 または 1 の軸は 0）。 |
| `TensorSpan<TElement> T { get; }` | Gets a zero-copy writable view with each matrix slice transposed: `B[i, j, k] = A[j, i, k]` (swaps axes 0 and 1 without complex conjugation). / 各行列スライスを転置した書き込み可能なビュー（`B[i, j, k] = A[j, i, k]`、第 0・第 1 軸のみ交換し複素共役は取りません）を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `MatrixSpan<TElement> AsMatrix()` | Returns a zero-copy writable `Lengths[0] × (Lengths[1] * Lengths[2])` matrix view placing all matrices side by side: `[A₀ A₁ …]`. / 各行列を横に並べた `[A₀ A₁ …]` 形式の、コピーなしの書き込み可能な `Lengths[0] × (Lengths[1] * Lengths[2])` 行列ビューを返します。 |
| `ReadOnlyTensorSpan<TElement> AsReadOnly()` | Returns a read-only view over the same elements. / 同じ要素を参照する読み取り専用ビューを返します。 |
| `void CopyTo(TensorSpan<TElement> destination)` | Copies the elements of this view into a destination view of the same shape. / このビューの要素を同じ形状の出力先ビューへコピーします。 |
| `static TensorSpan<TElement> Create(ref TElement reference, ReadOnlySpan<IntPtr> lengths, ReadOnlySpan<IntPtr> strides)` | Creates a writable tensor view from a reference to the first element, shape, and strides. / 先頭要素への参照・形状・ストライドから書き込み可能なテンソルビューを作成します。 |
| `MatrixSpan<TElement> Diagonals()` | Returns a zero-copy writable `min(Lengths[0], Lengths[1]) × Lengths[2]` matrix view whose column `k` is the main diagonal of matrix `k`. / 第 `k` 列が第 `k` 行列の主対角成分となる、コピーなしの書き込み可能な `min(Lengths[0], Lengths[1]) × Lengths[2]` 行列ビューを返します。 |
| `void Fill(TElement value)` | Fills all elements of the tensor view with the specified value. / テンソルビューのすべての要素を指定した値で埋めます。 |
| `MatrixSpan<TElement> GetMatrix(int index)` | Returns a zero-copy writable view of matrix `index` (`Lengths[0] × Lengths[1]`). / 第 `index` 行列（`Lengths[0] × Lengths[1]`）を参照する書き込み可能なビューを返します。 |
| `VectorSpan<TElement> GetVector(int column, int index)` | Returns a zero-copy writable view of column `column` of matrix `index`. / 第 `index` 行列の第 `column` 列を参照する書き込み可能なビューを返します。 |
| `TElement[,,] ToArray()` | Copies the elements of this view into a new three-dimensional array preserving `[row, column, index]` indices. / `[row, column, index]` の添字を維持してこのビューの要素を新しい 3 次元配列へコピーします。 |
| `Tensor<TElement> ToTensor()` | Copies the elements of this view into a new `Tensor`. / このビューの要素を新しい `Tensor` へコピーします。 |

**演算子**

`Tensor<TElement> operator +(TensorSpan<TElement> a, TensorSpan<TElement> b)` / `Tensor<TElement> operator /(TensorSpan<TElement> a, TElement scalar)` / `implicit operator ReadOnlyTensorSpan<TElement>(TensorSpan<TElement> view)` / `Tensor<TElement> operator *(TElement scalar, TensorSpan<TElement> a)` / `Tensor<TElement> operator *(TensorSpan<TElement> a, TElement scalar)` / `Tensor<TElement> operator *(TensorSpan<TElement> a, TensorSpan<TElement> b)` / `Tensor<TElement> operator -(TensorSpan<TElement> a, TensorSpan<TElement> b)` / `Tensor<TElement> operator -(TensorSpan<TElement> a)`

### `enum TwoStageReduction`

Specifies when two-stage reduction is used by `TwoStageReduction`. / `TwoStageReduction` で二段階縮約を使用する条件を指定します。

- 継承・実装: `IComparable`, `IConvertible`, `IFormattable`, `ISpanFormattable`

| 値 | 説明 |
| --- | --- |
| `Auto` | Uses two-stage reduction only when vectors are not computed and the matrix order is at least 2048. / 固有ベクトルまたは特異ベクトルを求めない、次数 2048 以上の行列でのみ使用します。 |
| `Never` | Disables two-stage reduction and always uses single-stage reduction. / 二段階縮約を使用せず、常に一段階縮約を実行します。 |
| `Always` | Always uses two-stage reduction whenever the matrix order exceeds twice the band width, including when vectors are computed. / ベクトルを求める場合も含め、次数が帯幅の 2 倍を超える行列で常に二段階縮約を使用します。 |

### `class Vector<T>`

Represents a dense one-dimensional numerical vector. / 1 次元の密ベクトルを表します。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `Vector(ReadOnlySpan<T> values)` | Initializes a new vector by copying the specified values. / 指定した値をコピーして新しいベクトルを初期化します。 |
| `Vector(int length)` | Initializes a new zero-filled vector of the specified length. / 指定した長さの零ベクトルを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `T this[int index] { get; set; }` | Gets or sets the element at the specified index. / 指定した位置の要素を取得または設定します。 |
| `VectorSpan<T> this[Range range] { get; }` | Gets a zero-copy writable view of the specified subrange of elements. / 指定した範囲の要素を参照する、コピーなしの書き込み可能な部分ベクトルビューを取得します。 |
| `int Length { get; }` | Gets the number of elements in the vector. / ベクトルの要素数を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `Matrix<T> AsMatrix(int rows, int columns)` | Reinterprets this vector as a column-major matrix sharing the same underlying storage. / 同じ格納領域を共有する列優先の行列としてこのベクトルを再解釈します。 |
| `Span<T> AsSpan()` | Returns a `Span` sharing the underlying contiguous storage. / 内部の連続した格納領域を共有する `Span` を返します。 |
| `VectorSpan<T> AsVectorSpan()` | Returns a writable `VectorSpan` view over the entire vector. / ベクトル全体を参照する書き込み可能な `VectorSpan` ビューを返します。 |
| `ReadOnlyMatrixSpan<T> Broadcast(int count)` | Returns a zero-copy read-only matrix view that broadcasts this vector as `count` columns. / このベクトルを `count` 本の列としてブロードキャストした、コピーなしの読み取り専用行列ビューを返します。 |
| `ReadOnlyMatrixSpan<T> BroadcastRows(int rows)` | Returns a zero-copy read-only matrix view that broadcasts this vector as `rows` rows. / このベクトルを `rows` 本の行としてブロードキャストした、コピーなしの読み取り専用行列ビューを返します。 |
| `Vector<T> Clone()` | Creates a deep copy of this vector with independent storage. / 独立した格納領域を持つこのベクトルの複製を作成します。 |
| `Vector<TTo> ConvertChecked<TTo>()` | Converts each element to `TTo` using checked numeric conversion (`CreateChecked`). / チェック付きの数値変換（`CreateChecked`）を用いて各要素を `TTo` に変換します。 |
| `static Vector<T> Generate<TOperator>(int length)` | Creates a new vector whose elements are computed from their indices using a stateless operator: `result[i] = TOperator(i)`. / 状態を持たない演算子を用いて添字から各要素を計算した新しいベクトル（`result[i] = TOperator(i)`）を作成します。 |
| `static Vector<T> Generate<TOperator>(int length, TOperator op)` | Creates a new vector whose elements are computed from their indices using a stateful operator: `result[i] = op(i)`. / 状態を持つ演算子を用いて添字から各要素を計算した新しいベクトル（`result[i] = op(i)`）を作成します。 |
| `static Vector<T> Generate(int length, Func<int, T> op)` | Creates a new vector whose elements are computed from their indices using a delegate: `result[i] = op(i)`. / デリゲートを用いて添字から各要素を計算した新しいベクトル（`result[i] = op(i)`）を作成します。 |
| `T[] ToArray()` | Copies the elements of the vector to a new array. / ベクトルの要素を新しい配列へコピーします。 |
| `Matrix<T> ToMatrix(int rows, int columns)` | Copies the elements of this vector into a new column-major matrix with the specified dimensions. / このベクトルの要素を指定した形状の新しい列優先行列へコピーします。 |

**演算子**

`Vector<T> operator +(Vector<T> x, Vector<T> y)` / `Vector<T> operator /(Vector<T> x, T scalar)` / `explicit operator Vector<T>(T[] values)` / `explicit operator T[](Vector<T> vector)` / `implicit operator VectorSpan<T>(Vector<T> vector)` / `implicit operator ReadOnlyVectorSpan<T>(Vector<T> vector)` / `Vector<T> operator *(ReadOnlyMatrixSpan<T> a, Vector<T> x)` / `Vector<T> operator *(T scalar, Vector<T> x)` / `Vector<T> operator *(Vector<T> x, T scalar)` / `Vector<T> operator -(Vector<T> x, Vector<T> y)` / `Vector<T> operator -(Vector<T> x)`

### `readonly ref struct VectorSpan<T>`

Represents a zero-copy, strided writable view over a one-dimensional vector. / ストライド付きの 1 次元ベクトルを参照する、コピーなしの書き込み可能なビューを表します。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `VectorSpan(Span<T> data)` | Initializes a writable view over a contiguous span of elements. / 連続した領域を参照する書き込み可能なビューを初期化します。 |
| `VectorSpan(Span<T> data, ReadOnlySpan<IntPtr> lengths, ReadOnlySpan<IntPtr> strides)` | Initializes a writable view over the specified span with explicit length and stride. / 長さとストライドを指定して領域を参照する書き込み可能なビューを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `T& this[int index] { get; }` | Gets a writable reference to the element at the specified index. / 指定した位置の要素への書き込み可能な参照を取得します。 |
| `VectorSpan<T> this[Range range] { get; }` | Gets a zero-copy writable view of the specified subrange. / 指定した範囲の部分ベクトルを参照する書き込み可能なビューを取得します。 |
| `Span<T> Data { get; }` | Gets a contiguous span from the first element through the last element, including any intervening elements outside the view. / 先頭要素から最終要素までの連続領域を取得します（ストライド間のビュー外要素も含みます）。 |
| `IntPtr FlattenedLength { get; }` | Gets the total number of elements. / 総要素数を取得します。 |
| `bool IsEmpty { get; }` | Gets a value indicating whether the view contains no elements. / ビューが空かどうかを示す値を取得します。 |
| `int Length { get; }` | Gets the number of elements in the vector view. / ベクトルビューの要素数を取得します。 |
| `ReadOnlySpan<IntPtr> Lengths { get; }` | Gets the shape `[Length]` of the view. / ビューの形状 `[Length]` を取得します。 |
| `int Rank { get; }` | Gets the number of dimensions, which is always 1. / 次元数（常に 1）を取得します。 |
| `ReadOnlySpan<IntPtr> Strides { get; }` | Gets the element stride of the view (0 when `Length` is 0 or 1). / 要素単位のストライドを取得します（`Length` が 0 または 1 の場合は 0）。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `ReadOnlyVectorSpan<T> AsReadOnly()` | Returns a read-only view over the same elements. / 同じ要素を参照する読み取り専用ビューを返します。 |
| `ReadOnlyMatrixSpan<T> Broadcast(int count)` | Returns a zero-copy read-only `Length × count` matrix view repeating this vector across `count` columns. / このベクトルを `count` 本の列として繰り返す、コピーなしの読み取り専用 `Length × count` 行列ビューを返します。 |
| `ReadOnlyMatrixSpan<T> BroadcastRows(int rows)` | Returns a zero-copy read-only `rows × Length` matrix view repeating this vector across `rows` rows. / このベクトルを `rows` 本の行として繰り返す、コピーなしの読み取り専用 `rows × Length` 行列ビューを返します。 |
| `void CopyTo(VectorSpan<T> destination)` | Copies the elements of this view into a destination view of the same length. / このビューの要素を同じ長さの出力先ビューへコピーします。 |
| `static VectorSpan<T> Create(ref T reference, ReadOnlySpan<IntPtr> lengths, ReadOnlySpan<IntPtr> strides)` | Creates a writable view from a reference to the first element, length, and stride. / 先頭要素への参照・長さ・ストライドから書き込み可能なビューを作成します。 |
| `void Fill(T value)` | Fills all elements of the vector view with the specified value. / ベクトルビューのすべての要素を指定した値で埋めます。 |
| `T[] ToArray()` | Copies the elements of this view into a new array. / このビューの要素を新しい配列へコピーします。 |
| `Vector<T> ToVector()` | Copies the elements of this view into a new `Vector`. / このビューの要素を新しい `Vector` へコピーします。 |

**演算子**

`Vector<T> operator +(VectorSpan<T> x, VectorSpan<T> y)` / `Vector<T> operator /(VectorSpan<T> x, T scalar)` / `implicit operator VectorSpan<T>(Span<T> data)` / `implicit operator ReadOnlyVectorSpan<T>(VectorSpan<T> view)` / `Vector<T> operator *(ReadOnlyMatrixSpan<T> a, VectorSpan<T> x)` / `Vector<T> operator *(T scalar, VectorSpan<T> x)` / `Vector<T> operator *(VectorSpan<T> x, T scalar)` / `Vector<T> operator -(VectorSpan<T> x, VectorSpan<T> y)` / `Vector<T> operator -(VectorSpan<T> x)`

## Tuna.QuasiRandom

### `interface IPointSequence`

Represents a deterministic low-discrepancy point sequence in the open unit hypercube `(0, 1)^d` with random-access indexing. / 単位超立方体 `(0, 1)^d` 内の点をインデックスでランダムアクセスできる決定的な点列を表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Dimension { get; }` | Gets the dimensionality of points in the sequence. / 点列の次元数を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `void GetPoint(ulong index, Span<double> destination)` | Writes the point at the specified zero-based index into `destination`. / 指定した 0 始まりのインデックスに対応する点を `destination` に書き込みます。 |
| `MatrixSpan<double> GetPoints(ulong start, MatrixSpan<double> destination)` | Writes consecutive points starting at `start` into the columns of `destination` (one point per column). / `start` から始まる連続した点を、`destination` の各列に 1 列あたり 1 点ずつ書き込みます。 |
| `Matrix<double> GetPoints(ulong start, int count)` | Allocates a new `Dimension`-by-`count` matrix and fills its columns with consecutive points starting at `start`. / `Dimension` 行 `count` 列の新しい行列を確保し、`start` から始まる連続した点を各列に書き込みます。 |

### `class SobolSequence`

Represents a Sobol' low-discrepancy point sequence, either unscrambled or randomized via linear matrix scrambling and digital shifting. / 非スクランブル、または線形行列スクランブルとデジタルシフトによりランダム化された Sobol' 低食い違い量点列を表します。

- 継承・実装: `IPointSequence`

**フィールド・定数**

| メンバー | 説明 |
| --- | --- |
| `const int MaxDimension` | The maximum number of dimensions supported by `SobolSequence`. / `SobolSequence` がサポートする最大次元数。 |

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `SobolSequence(int dimension)` | Initializes a new unscrambled `SobolSequence` with the specified dimensionality. / 指定した次元数の非スクランブル Sobol' 点列を初期化します。 |
| `SobolSequence(int dimension, IRandomSource random)` | Initializes a new randomized `SobolSequence` using linear matrix scrambling and a random digital shift. / 線形行列スクランブルとランダムなデジタルシフトによりランダム化された Sobol' 点列を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Dimension { get; }` | Gets the dimensionality of points in the sequence. / 点列の次元数を取得します。 |
| `bool IsScrambled { get; }` | Gets a value indicating whether this sequence is randomized by scrambling and digital shifting. / この点列がスクランブルおよびデジタルシフトによりランダム化されているかどうかを示す値を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `void GetPoint(ulong index, Span<double> destination)` | Writes the point at the specified zero-based index into `destination`. / 指定した 0 始まりのインデックスに対応する点を `destination` に書き込みます。 |
| `MatrixSpan<double> GetPoints(ulong start, MatrixSpan<double> destination)` | Writes consecutive points starting at `start` into the columns of `destination` (one point per column). / `start` から始まる連続した点を、`destination` の各列に 1 列あたり 1 点ずつ書き込みます。 |
| `Matrix<double> GetPoints(ulong start, int count)` | Writes consecutive points starting at `start` into the columns of `destination` (one point per column). / `start` から始まる連続した点を、`destination` の各列に 1 列あたり 1 点ずつ書き込みます。 |

## Tuna.RandomSource

### `class BoundedCompositionSampler`

Samples integer compositions satisfying per-component lower and upper bounds and a fixed total from an approximately uniform distribution. / 各成分の上下限および総和の制約を満たす整数の組を、近似的な一様分布からサンプリングします。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `BoundedCompositionSampler(ReadOnlySpan<long> lower, ReadOnlySpan<long> upper, long total)` | Initializes a new `BoundedCompositionSampler` and precomputes the counting table for the specified bounds and total. / 指定した上下限および総和に対する計数テーブルを前計算し、`BoundedCompositionSampler` の新しいインスタンスを初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Length { get; }` | Gets the number of components `n`. / 成分の数 `n` を取得します。 |
| `double LogCount { get; }` | Gets the natural logarithm `ln N` of the number of feasible compositions. / 条件を満たす組の総数の自然対数 `ln N` を取得します。 |
| `long Total { get; }` | Gets the required total sum `s`. / 全成分の総和 `s` を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `void Sample(IRandomSource random, Span<long> destination)` | Draws one composition and writes its components to `destination`. / 条件を満たす組を 1 つサンプリングし、`destination` に書き込みます。 |

### `interface IRandomSource`

Represents a thread-safe uniform pseudorandom number generator. / スレッドセーフな一様擬似乱数生成器を表します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `void Fill(Span<uint> destination)` | Fills the specified span with random 32-bit unsigned integers in the same order as repeated calls to `NextUInt32`. / 指定したスパンを、`NextUInt32` の連続呼び出しと同じ順序の 32 ビット符号なし一様乱数で埋めます。 |
| `void Fill(Span<ulong> destination)` | Fills the specified span with random 64-bit unsigned integers in the same order as repeated calls to `NextUInt64`. / 指定したスパンを、`NextUInt64` の連続呼び出しと同じ順序の 64 ビット符号なし一様乱数で埋めます。 |
| `void Fill(Span<double> destination)  (+1 overloads)` | Fills the specified span with random double-precision floating-point numbers in the same order as repeated calls to `NextDouble`. / 指定したスパンを、`NextDouble` の連続呼び出しと同じ順序の倍精度浮動小数点一様乱数で埋めます。 |
| `void Fill(Span<byte> destination)` | Fills the specified span with random bytes generated from 64-bit outputs in little-endian order. / 64 ビット出力をリトルエンディアン順に展開して、指定したスパンを乱数バイトで埋めます。 |
| `double NextDouble()` | Returns a random double-precision floating-point number in the half-open interval `[0, 1)`. / 半開区間 `[0, 1)` の倍精度浮動小数点一様乱数を返します。 |
| `double NextDoubleOpen()` | Returns a random double-precision floating-point number in the open interval `(0, 1)`. / 開区間 `(0, 1)` の倍精度浮動小数点一様乱数を返します。 |
| `int NextInt32()` | Returns a non-negative random 32-bit signed integer. / 非負の 32 ビット符号付き一様乱数を返します。 |
| `int NextInt32(int exclusiveMax)` | Returns a non-negative random 32-bit signed integer that is less than the specified maximum. / 指定した上限未満の非負 32 ビット符号付き一様乱数を返します。 |
| `int NextInt32(int inclusiveMin, int exclusiveMax)` | Returns a random 32-bit signed integer within the specified range. / 指定した範囲内の 32 ビット符号付き一様乱数を返します。 |
| `long NextInt64()` | Returns a non-negative random 64-bit signed integer. / 非負の 64 ビット符号付き一様乱数を返します。 |
| `long NextInt64(long exclusiveMax)` | Returns a non-negative random 64-bit signed integer that is less than the specified maximum. / 指定した上限未満の非負 64 ビット符号付き一様乱数を返します。 |
| `long NextInt64(long inclusiveMin, long exclusiveMax)` | Returns a random 64-bit signed integer within the specified range. / 指定した範囲内の 64 ビット符号付き一様乱数を返します。 |
| `float NextSingle()` | Returns a random single-precision floating-point number in the half-open interval `[0, 1)`. / 半開区間 `[0, 1)` の単精度浮動小数点一様乱数を返します。 |
| `uint NextUInt32()` | Returns a random 32-bit unsigned integer. / 32 ビットの符号なし一様乱数を返します。 |
| `uint NextUInt32(uint exclusiveMax)` | Returns a random 32-bit unsigned integer that is less than the specified maximum. / 指定した上限未満の 32 ビット符号なし一様乱数を返します。 |
| `ulong NextUInt64()` | Returns a random 64-bit unsigned integer. / 64 ビットの符号なし一様乱数を返します。 |
| `ulong NextUInt64(ulong exclusiveMax)` | Returns a random 64-bit unsigned integer that is less than the specified maximum. / 指定した上限未満の 64 ビット符号なし一様乱数を返します。 |

### `class Pcg64DxsmRandom`

A thread-safe PCG64 DXSM (128-bit state, 64-bit output) pseudorandom number generator with deterministic seeding and stream selection. / 決定的なシード初期化とストリーム識別をサポートする、スレッドセーフな PCG64 DXSM（128 ビット状態・64 ビット出力）擬似乱数生成器です。

- 継承・実装: `RandomGenerator`, `IRandomSource`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `Pcg64DxsmRandom()` | Initializes a new instance of `Pcg64DxsmRandom` with a seed obtained from the operating system's cryptographic random number generator. / OS の暗号論的乱数生成器から取得したシードで `Pcg64DxsmRandom` の新しいインスタンスを初期化します。 |
| `Pcg64DxsmRandom(Pcg64DxsmState state)` | Initializes a new instance of `Pcg64DxsmRandom` restored from a previously saved state. / 保存された状態から `Pcg64DxsmRandom` の新しいインスタンスを復元します。 |
| `Pcg64DxsmRandom(ulong seed, ulong streamId = 0)` | Initializes a new instance of `Pcg64DxsmRandom` from the specified seed and logical stream identifier. / 指定したシードと論理的なストリーム識別子から `Pcg64DxsmRandom` の新しいインスタンスを初期化します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `void Advance(UInt128 count)` | Advances the generator state by the specified number of raw 64-bit outputs in `O(log count)` time. / 生の 64 ビット出力を指定個数だけ消費した位置へ `O(log count)` 時間で状態を進めます。 |
| `Pcg64DxsmState GetState()` | Captures a consistent snapshot of the current generator state. / 現在の乱数生成器の状態を一貫したスナップショットとして取得します。 |
| `void RestoreState(Pcg64DxsmState saved)` | Restores the generator to a previously saved state. / 乱数生成器を保存された状態へ復元します。 |

### `readonly struct Pcg64DxsmState`

Represents a serializable snapshot of a `Pcg64DxsmRandom` generator's internal state. / `Pcg64DxsmRandom` の内部状態を保存するためのスナップショットを表します。

- 継承・実装: `IEquatable<Pcg64DxsmState>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `Pcg64DxsmState(UInt128 State, UInt128 Increment, int Version = 1)` | Represents a serializable snapshot of a `Pcg64DxsmRandom` generator's internal state. / `Pcg64DxsmRandom` の内部状態を保存するためのスナップショットを表します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `UInt128 Increment { get; set; }` | The 128-bit LCG increment (must be odd). / 128 ビットの LCG 増分（奇数である必要があります）。 |
| `UInt128 State { get; set; }` | The 128-bit LCG state value. / 128 ビットの LCG 状態値。 |
| `int Version { get; set; }` | The state format version (currently `1`). / 状態フォーマットのバージョン（現在は `1`）。 |

### `readonly struct Philox4x32Block`

Represents a 4-word (128-bit) output block produced by Philox4x32-10. / Philox4x32-10 が生成する 4 語（128 ビット）の出力ブロックを表します。

- 継承・実装: `IEquatable<Philox4x32Block>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `Philox4x32Block(uint Word0, uint Word1, uint Word2, uint Word3)` | Represents a 4-word (128-bit) output block produced by Philox4x32-10. / Philox4x32-10 が生成する 4 語（128 ビット）の出力ブロックを表します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `uint this[int index] { get; }` | Gets the 32-bit output word at the specified index within the block. / ブロック内の指定したインデックスにある 32 ビット出力語を取得します。 |
| `uint Word0 { get; set; }` | The first 32-bit output word (index 0). / 1 番目の 32 ビット出力語（インデックス 0）。 |
| `uint Word1 { get; set; }` | The second 32-bit output word (index 1). / 2 番目の 32 ビット出力語（インデックス 1）。 |
| `uint Word2 { get; set; }` | The third 32-bit output word (index 2). / 3 番目の 32 ビット出力語（インデックス 2）。 |
| `uint Word3 { get; set; }` | The fourth 32-bit output word (index 3). / 4 番目の 32 ビット出力語（インデックス 3）。 |

### `class Philox4x32Random`

A thread-safe Philox4x32-10 counter-based pseudorandom number generator with 64-bit seed and stream identifiers. / 64 ビットのシードとストリーム識別子を持つ、スレッドセーフなカウンタ方式擬似乱数生成器（Philox4x32-10）です。

- 継承・実装: `RandomGenerator`, `IRandomSource`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `Philox4x32Random()` | Initializes a new instance of `Philox4x32Random` with a seed obtained from the operating system's cryptographic random number generator. / OS の暗号論的乱数生成器から取得したシードで `Philox4x32Random` の新しいインスタンスを初期化します。 |
| `Philox4x32Random(Philox4x32State state)` | Initializes a new instance of `Philox4x32Random` restored from a previously saved state. / 保存された状態から `Philox4x32Random` の新しいインスタンスを復元します。 |
| `Philox4x32Random(ulong seed, ulong streamId = 0)` | Initializes a new instance of `Philox4x32Random` starting at block 0, word 0 for the specified seed and stream identifier. / 指定したシードとストリーム識別子で、ブロック 0・語 0 から開始する `Philox4x32Random` の新しいインスタンスを初期化します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static Philox4x32Block GenerateBlock(ulong seed, ulong streamId, ulong blockIndex)` | Computes the output block at the specified counter position without side effects, shared state, or locking. / 副作用・共有状態・ロックなしで、指定したカウンタ位置の出力ブロックを計算します。 |
| `Philox4x32State GetState()` | Captures a consistent snapshot of the current generator state, including the unread word offset. / 未消費の語位置を含む現在の乱数生成器の状態を一貫したスナップショットとして取得します。 |
| `void RestoreState(Philox4x32State state)` | Restores the generator to a previously saved state. / 乱数生成器を保存された状態へ復元します。 |
| `void Seek(ulong blockIndex, int wordIndex = 0)` | Moves the generator to the specified block and word position within the current seed and stream. / 現在のシードとストリームのまま、指定したブロック番号および語位置へ移動します。 |

### `readonly struct Philox4x32State`

Represents a serializable snapshot of a `Philox4x32Random` generator's state. / `Philox4x32Random` の状態を保存するためのスナップショットを表します。

- 継承・実装: `IEquatable<Philox4x32State>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `Philox4x32State(ulong Seed, ulong StreamId, ulong BlockIndex, int WordIndex, int Version = 1)` | Represents a serializable snapshot of a `Philox4x32Random` generator's state. / `Philox4x32Random` の状態を保存するためのスナップショットを表します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `ulong BlockIndex { get; set; }` | The zero-based 64-bit block index (lower 64 bits of the 128-bit counter). / 0 から始まる 64 ビットのブロック番号（128 ビットカウンタの下位 64 ビット）。 |
| `ulong Seed { get; set; }` | The 64-bit seed (Philox key). / 64 ビットのシード（Philox のキー）。 |
| `ulong StreamId { get; set; }` | The 64-bit stream identifier (upper 64 bits of the 128-bit counter). / 64 ビットのストリーム識別子（128 ビットカウンタの上位 64 ビット）。 |
| `int Version { get; set; }` | The state format version (currently `1`). / 状態フォーマットのバージョン（現在は `1`）。 |
| `int WordIndex { get; set; }` | The next unread word index in the current block (0 through 4, where 4 indicates the current block is fully consumed). / 現在のブロック内で次に読み出す語の位置（0 以上 4 以下。4 は現在のブロックを消費済みであることを表します）。 |

### `abstract class RandomGenerator`

Provides a thread-safe base implementation of `IRandomSource` with uniform range conversion and per-call synchronization. / 一様乱数の範囲変換と呼び出し単位の同期を提供する、`IRandomSource` のスレッドセーフな基底クラスです。

- 継承・実装: `IRandomSource`

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `void Fill(Span<uint> destination)` | Fills the specified span with random 32-bit unsigned integers in the same order as repeated calls to `NextUInt32`. / 指定したスパンを、`NextUInt32` の連続呼び出しと同じ順序の 32 ビット符号なし一様乱数で埋めます。 |
| `void Fill(Span<ulong> destination)` | Fills the specified span with random 32-bit unsigned integers in the same order as repeated calls to `NextUInt32`. / 指定したスパンを、`NextUInt32` の連続呼び出しと同じ順序の 32 ビット符号なし一様乱数で埋めます。 |
| `void Fill(Span<double> destination)  (+1 overloads)` | Fills the specified span with random 32-bit unsigned integers in the same order as repeated calls to `NextUInt32`. / 指定したスパンを、`NextUInt32` の連続呼び出しと同じ順序の 32 ビット符号なし一様乱数で埋めます。 |
| `void Fill(Span<byte> destination)` | Fills the specified span with random 32-bit unsigned integers in the same order as repeated calls to `NextUInt32`. / 指定したスパンを、`NextUInt32` の連続呼び出しと同じ順序の 32 ビット符号なし一様乱数で埋めます。 |
| `double NextDouble()` | Returns a random double-precision floating-point number in the half-open interval `[0, 1)`. / 半開区間 `[0, 1)` の倍精度浮動小数点一様乱数を返します。 |
| `double NextDoubleOpen()` | Returns a random double-precision floating-point number in the open interval `(0, 1)`. / 開区間 `(0, 1)` の倍精度浮動小数点一様乱数を返します。 |
| `int NextInt32()` | Returns a non-negative random 32-bit signed integer. / 非負の 32 ビット符号付き一様乱数を返します。 |
| `int NextInt32(int exclusiveMax)` | Returns a non-negative random 32-bit signed integer that is less than the specified maximum. / 指定した上限未満の非負 32 ビット符号付き一様乱数を返します。 |
| `int NextInt32(int inclusiveMin, int exclusiveMax)` | Returns a random 32-bit signed integer within the specified range. / 指定した範囲内の 32 ビット符号付き一様乱数を返します。 |
| `long NextInt64()` | Returns a non-negative random 64-bit signed integer. / 非負の 64 ビット符号付き一様乱数を返します。 |
| `long NextInt64(long exclusiveMax)` | Returns a non-negative random 64-bit signed integer that is less than the specified maximum. / 指定した上限未満の非負 64 ビット符号付き一様乱数を返します。 |
| `long NextInt64(long inclusiveMin, long exclusiveMax)` | Returns a random 64-bit signed integer within the specified range. / 指定した範囲内の 64 ビット符号付き一様乱数を返します。 |
| `float NextSingle()` | Returns a random single-precision floating-point number in the half-open interval `[0, 1)`. / 半開区間 `[0, 1)` の単精度浮動小数点一様乱数を返します。 |
| `uint NextUInt32()` | Returns a random 32-bit unsigned integer. / 32 ビットの符号なし一様乱数を返します。 |
| `uint NextUInt32(uint exclusiveMax)` | Returns a random 32-bit unsigned integer that is less than the specified maximum. / 指定した上限未満の 32 ビット符号なし一様乱数を返します。 |
| `ulong NextUInt64()` | Returns a random 64-bit unsigned integer. / 64 ビットの符号なし一様乱数を返します。 |
| `ulong NextUInt64(ulong exclusiveMax)` | Returns a random 64-bit unsigned integer that is less than the specified maximum. / 指定した上限未満の 64 ビット符号なし一様乱数を返します。 |

### `static class RandomSourceExtensions`

Provides extension methods for uniformly shuffling and generating random permutations with an `IRandomSource`. / `IRandomSource` を用いて有限列を一様にシャッフルおよび順列生成するための拡張メソッドを提供します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static void Permutation(IRandomSource random, Span<int> destination)` | Writes a uniformly random permutation of the integers from `0` to `destination.Length - 1` into the specified span. / `0` から `destination.Length - 1` までの整数の一様な順列を指定したスパンに書き込みます。 |
| `static void Shuffle<T>(IRandomSource random, Span<T> values)` | Randomly shuffles the elements of a span in place into a uniformly distributed permutation. / スパンの要素をインプレースで一様な順序にシャッフルします。 |

