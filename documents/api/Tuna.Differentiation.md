# Tuna.Differentiation API リファレンス

> このファイルは `tools/ApiDocGenerator` によって自動生成されています。手動で編集せず、`dotnet build Tuna.slnx -c Release` の後に再生成してください。
>
> `(+N overloads)` は `Vector` / `VectorSpan` / `ReadOnlyVectorSpan` 等のコンテナ・ビュー組み合わせや `double` / `float` 固定型オーバーロードの省略を表します。

## 公開型一覧

- `Tuna.Differentiation` (16 型): [`CholeskyDecompositionNode`](#readonly-struct-choleskydecompositionnode), [`DifferentiableOperationContext<T>`](#readonly-ref-struct-differentiableoperationcontextt), [`IDifferentiableOperation<T>`](#interface-idifferentiableoperationt), [`LuDecompositionNode`](#readonly-struct-ludecompositionnode), [`MatrixNode<TElement>`](#readonly-struct-matrixnodetelement), [`Parameter<T>`](#class-parametert), [`ParameterSet<T>`](#class-parametersett), [`QrDecompositionNode`](#readonly-struct-qrdecompositionnode), [`ScalarNode`](#readonly-struct-scalarnode), [`SingularValueDecompositionNode`](#readonly-struct-singularvaluedecompositionnode), [`SymmetricEigenDecompositionNode`](#readonly-struct-symmetriceigendecompositionnode), [`Tape`](#class-tape), [`TensorNode<TElement>`](#readonly-struct-tensornodetelement), [`VectorNode<T>`](#readonly-struct-vectornodet), [`XMathExtensions`](#static-class-xmathextensions), [`XMathSpecialExtensions`](#static-class-xmathspecialextensions)

## Tuna.Differentiation

### `readonly struct CholeskyDecompositionNode`

Represents a Cholesky decomposition `A = LLᵀ` of a symmetric positive-definite matrix recorded on a `Tape`. / `Tape` に記録された対称正定値行列のコレスキー分解 `A = LLᵀ` を表すノードです。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `MatrixNode<double> L { get; }` | Gets the lower-triangular Cholesky factor `L` with positive diagonal entries. / 対角成分が正の下三角因子 `L` の行列ノードを取得します。 |
| `ScalarNode LogDeterminant { get; }` | Gets the natural logarithm of the determinant, `ln det A = 2 Σ ln lᵢᵢ`. / 行列式の自然対数 `ln det A = 2 Σ ln lᵢᵢ` を取得します。 |
| `int Order { get; }` | Gets the order `n` of the factored matrix. / 分解対象の行列の次数 `n` を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static CholeskyDecompositionNode Compute(MatrixNode<double> a)` | Computes the Cholesky decomposition of a symmetric positive-definite matrix node. / 対称正定値行列ノードのコレスキー分解を計算します。 |
| `ScalarNode InverseQuadraticForm(VectorNode<double> x)` | Computes the inverse quadratic form `xᵀA⁻¹x = ‖L⁻¹x‖²`. / 二次形式 `xᵀA⁻¹x = ‖L⁻¹x‖²` を計算します。 |
| `VectorNode<double> Solve(VectorNode<double> b)` | Solves the linear system `Ax = b` for a single right-hand side vector. / 単一右辺の連立一次方程式 `Ax = b` を解きます。 |
| `MatrixNode<double> Solve(MatrixNode<double> b)` | Solves the linear system `AX = B` for multiple right-hand sides. / 複数右辺の連立一次方程式 `AX = B` を解きます。 |
| `static bool TryCompute(MatrixNode<double> a, out CholeskyDecompositionNode result)` | Attempts to compute the Cholesky decomposition of a symmetric matrix node. / 対称行列ノードのコレスキー分解を試みます。 |

### `readonly ref struct DifferentiableOperationContext<T>`

Provides values, gradients, and tape-owned storage during a user operation callback. / 利用者定義演算のコールバック中に、値、勾配、およびテープ所有の保存領域を提供します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `ReadOnlyMatrixSpan<T> this[int index] { get; }` | Gets the input value as a matrix view. / 指定した入力値を行列ビューとして取得します。 |
| `int Count { get; }` | Gets the number of inputs. / 入力の個数を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `Span<T> AllocateSaved(int length)` | Allocates a tape-owned buffer for values needed by the pullback. / 逆伝播に必要な値を保存するテープ所有の領域を確保します。 |
| `MatrixSpan<T> GetGradient(int index)` | Gets a writable view for accumulating contributions to the specified input gradient. / 指定した入力の勾配に寄与を加算するための書き込み可能なビューを取得します。 |
| `ReadOnlySpan<T> GetSaved(int index)` | Reads a buffer saved by this operation's forward callback. / この演算の順伝播で保存した領域を読み取ります。 |
| `bool RequiresGradient(int index)` | Determines whether the specified input requires a gradient. / 指定した入力が勾配を必要とするかどうかを判定します。 |

### `interface IDifferentiableOperation<T>`

Defines the forward evaluation and pullback of an array operation. / 配列演算の順伝播と逆伝播を定義します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `void Forward(DifferentiableOperationContext<T> context, MatrixSpan<T> destination)` | Computes all output elements from the input values. / 入力値から出力の全要素を計算します。 |
| `void PullBack(DifferentiableOperationContext<T> context, ReadOnlyMatrixSpan<T> output, ReadOnlyMatrixSpan<T> outputGradient)` | Accumulates contributions to input gradients from the output adjoint. / 出力の随伴から求めた寄与を、勾配が必要な入力に加算します。 |

### `readonly struct LuDecompositionNode`

Represents a partially pivoted LU decomposition `PA = LU` recorded on a `Tape`. / `Tape` に記録された部分ピボット付き LU 分解 `PA = LU` を表すノードです。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int DeterminantSign { get; }` | Gets the sign of the determinant (`1`, `-1`, or `0` if singular; non-differentiable). / 行列式の符号（`1`、`-1`、特異の場合は `0`。微分対象外）を取得します。 |
| `bool IsSingular { get; }` | Gets a value indicating whether an exact zero pivot was encountered during factorization. / 分解の途中でピボットが厳密に 0 になった（特異である）かどうかを取得します。 |
| `ScalarNode LogAbsDeterminant { get; }` | Gets the natural logarithm of the absolute value of the determinant, `ln \|det A\|`. / 行列式の絶対値の自然対数 `ln \|det A\|` を取得します。 |
| `int Order { get; }` | Gets the order `n` of the factored matrix. / 分解対象の行列の次数 `n` を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static LuDecompositionNode Compute(MatrixNode<double> a)` | Computes the LU decomposition of a square matrix node. / 正方行列ノードの LU 分解を計算します。 |
| `VectorNode<double> Solve(VectorNode<double> b)` | Solves the linear system `Ax = b` for a single right-hand side vector. / 単一右辺の連立一次方程式 `Ax = b` を解きます。 |
| `MatrixNode<double> Solve(MatrixNode<double> b)` | Solves the linear system `AX = B` for multiple right-hand sides. / 複数右辺の連立一次方程式 `AX = B` を解きます。 |
| `VectorNode<double> SolveTransposed(VectorNode<double> b)` | Solves the transposed linear system `Aᵀx = b` for a single right-hand side vector. / 転置された単一右辺の連立一次方程式 `Aᵀx = b` を解きます。 |
| `MatrixNode<double> SolveTransposed(MatrixNode<double> b)` | Solves the transposed linear system `AᵀX = B` for multiple right-hand sides. / 転置された複数右辺の連立一次方程式 `AᵀX = B` を解きます。 |

### `readonly struct MatrixNode<TElement>`

Represents a matrix node for reverse-mode automatic differentiation. / 逆モード自動微分の行列ノードを表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `MatrixNode<TElement> this[Range rows, Range columns] { get; }` | Gets a zero-copy submatrix view over the specified row and column ranges. / 指定した行範囲と列範囲の部分行列ビューをコピーなしで取得します。 |
| `int Columns { get; }` | Gets the number of columns in the matrix. / 行列の列数を取得します。 |
| `bool RequiresGradient { get; }` | Gets a value indicating whether this node depends on any differentiable leaf or `Parameter`. / このノードが微分対象の葉または `Parameter` に依存しているかどうかを示す値を取得します。 |
| `int Rows { get; }` | Gets the number of rows in the matrix. / 行列の行数を取得します。 |
| `MatrixNode<TElement> T { get; }` | Gets a zero-copy transposed view `Aᵀ`. / 転置 `Aᵀ` のビューをコピーなしで取得します。 |
| `ReadOnlyMatrixSpan<TElement> Value { get; }` | Gets a read-only view of the node's values. / ノードの値の読み取り専用ビューを取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `TensorNode<TElement> Broadcast(int count)` | Repeats the matrix along the third axis without copying its values. / 行列の値をコピーせず、第 3 軸に沿って繰り返します。 |
| `static MatrixNode<TElement> Record<TOperation>(TOperation operation, ReadOnlySpan<MatrixNode<TElement>> inputs, int rows, int columns)` | Records a user operation on the tape shared by its matrix inputs. / 行列入力が共有するテープに利用者定義演算を記録します。 |

**演算子**

`MatrixNode<TElement> operator +(MatrixNode<TElement> x, MatrixNode<TElement> y)` / `MatrixNode<TElement> operator +(MatrixNode<TElement> x, ScalarNode s)` / `MatrixNode<TElement> operator +(ScalarNode s, MatrixNode<TElement> x)` / `MatrixNode<TElement> operator /(MatrixNode<TElement> x, ScalarNode s)` / `MatrixNode<TElement> operator *(MatrixNode<TElement> a, MatrixNode<TElement> b)` / `MatrixNode<TElement> operator *(MatrixNode<TElement> x, ScalarNode s)` / `MatrixNode<TElement> operator *(ScalarNode s, MatrixNode<TElement> x)` / `MatrixNode<TElement> operator -(MatrixNode<TElement> x, MatrixNode<TElement> y)` / `MatrixNode<TElement> operator -(MatrixNode<TElement> x, ScalarNode s)` / `MatrixNode<TElement> operator -(ScalarNode s, MatrixNode<TElement> x)` / `MatrixNode<TElement> operator -(MatrixNode<TElement> x)`

### `class Parameter<T>`

Represents persistent learnable parameter storage (value and gradient buffers) that outlives a `Tape`. / `Tape` の寿命を超えて保持される学習パラメーター（値および勾配バッファ）を表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Columns { get; }` | Gets the number of columns (1 for a vector). / 列数（ベクトルの場合は 1）を取得します。 |
| `Span<T> Gradient { get; }` | Gets a span over the accumulated gradients in column-major order. / 逆伝播で加算された勾配へのスパン（列優先）を取得します。 |
| `int Length { get; }` | Gets the total number of elements. / 総要素数を取得します。 |
| `int MatrixCount { get; }` | Gets the number of matrices along the third axis, or 1 for vectors and matrices. / 第3軸の行列数を取得します。ベクトルと行列の場合は1です。 |
| `int Rank { get; }` | Gets the rank of the parameter (1 for a vector, 2 for a matrix, 3 for a tensor). / パラメーターの次元数（ベクトルは1、行列は2、テンソルは3）を取得します。 |
| `int Rows { get; }` | Gets the number of rows (equal to the element count for a vector). / 行数（ベクトルの場合は要素数）を取得します。 |
| `Span<T> Value { get; }` | Gets a span over the parameter values in column-major order. / 列優先で格納された値へのスパンを取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `MatrixSpan<T> AsMatrixSpan()` | Returns a zero-copy matrix view over `Value` for a rank-2 parameter. / 2 階（行列）パラメーターの `Value` に対するコピーなしの行列ビューを返します。 |
| `TensorSpan<T> AsTensorSpan()` | Returns a zero-copy tensor view over `Value` for a rank-3 parameter. / 3 階（テンソル）パラメーターの `Value` に対するコピーなしのテンソルビューを返します。 |
| `VectorSpan<T> AsVectorSpan()` | Returns a zero-copy vector view over `Value` for a rank-1 parameter. / 1 階（ベクトル）パラメーターの `Value` に対するコピーなしのベクトルビューを返します。 |
| `static Parameter<T> Matrix(int rows, int columns)` | Creates a matrix parameter initialized to zeros. / 0 で初期化された行列型の学習パラメーターを作成します。 |
| `static Parameter<T> Tensor(int rows, int columns, int matrixCount)` | Creates a three-dimensional parameter initialized to zeros. / 0 で初期化された3階テンソルの学習パラメーターを作成します。 |
| `static Parameter<T> Vector(int length)` | Creates a vector parameter initialized to zeros. / 0 で初期化されたベクトル型の学習パラメーターを作成します。 |
| `void ZeroGradient()` | Clears all gradient entries to zero. / 勾配のすべての要素を 0 にリセットします。 |

### `class ParameterSet<T>`

Represents an ordered collection of distinct `Parameter` instances. / 重複しない複数の `Parameter` を順序付きで保持するコレクションを表します。

- 継承・実装: `IEnumerable`, `IEnumerable<Parameter<T>>`, `IReadOnlyCollection<Parameter<T>>`, `IReadOnlyList<Parameter<T>>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `ParameterSet(ReadOnlySpan<Parameter<T>> parameters)` | Initializes a new `ParameterSet` containing the specified parameters in order. / 指定したパラメーターをその順序で保持する `ParameterSet` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `Parameter<T> this[int index] { get; }` | Gets the parameter at the specified index. / 指定したインデックスにあるパラメーターを取得します。 |
| `int Count { get; }` | Gets the number of parameters in the set. / 含まれるパラメーターの個数を取得します。 |
| `long Length { get; }` | Gets the total number of scalar elements across all parameters. / すべてのパラメーターの総要素数を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `IEnumerator<Parameter<T>> GetEnumerator()` | Returns an enumerator that iterates through the parameters in order. / パラメーターを順に反復処理する列挙子を返します。 |
| `void ZeroGradients()` | Clears the gradients of all parameters in the set to zero. / すべてのパラメーターの勾配を 0 にリセットします。 |

### `readonly struct QrDecompositionNode`

Represents a thin QR decomposition `A = QR` recorded on a `Tape`. / `Tape` に記録された薄い QR 分解 `A = QR` を表すノードです。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Columns { get; }` | Gets the column count `n` of the factored matrix. / 元の行列の列数 `n` を取得します。 |
| `MatrixNode<double> Q { get; }` | Gets the `m × min(m, n)` matrix `Q` with orthonormal columns. / 正規直交列を持つ `m × min(m, n)` 行列 `Q` のノードを取得します。 |
| `MatrixNode<double> R { get; }` | Gets the `min(m, n) × n` upper-trapezoidal matrix `R`. / `min(m, n) × n` の上台形行列 `R` のノードを取得します。 |
| `int Rows { get; }` | Gets the row count `m` of the factored matrix. / 元の行列の行数 `m` を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static QrDecompositionNode Compute(MatrixNode<double> a)` | Computes the thin QR decomposition of a matrix node. / 行列ノードの薄い QR 分解を計算します。 |

### `readonly struct ScalarNode`

Represents a scalar node for reverse-mode automatic differentiation. A node without an associated `Tape` is an unrecorded constant. / 逆モード自動微分のスカラーノードを表します。`Tape` に属さない値は記録されない定数として扱われます。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `bool RequiresGradient { get; }` | Gets a value indicating whether this node depends on any differentiable leaf. / このノードが微分対象の葉に依存しているかどうかを示す値を取得します。 |
| `double Value { get; }` | Gets the scalar value of the node. / ノードのスカラー値を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static ScalarNode Record(double value, ReadOnlySpan<(ScalarNode, double)> inputs)` | Records a custom scalar operation with user-supplied local derivatives. / ユーザー定義のスカラー演算と局所微分をテープに記録します。 |

**演算子**

`ScalarNode operator +(ScalarNode left, ScalarNode right)` / `ScalarNode operator /(ScalarNode left, ScalarNode right)` / `bool operator >(ScalarNode left, ScalarNode right)` / `bool operator >=(ScalarNode left, ScalarNode right)` / `implicit operator ScalarNode(double value)` / `bool operator <(ScalarNode left, ScalarNode right)` / `bool operator <=(ScalarNode left, ScalarNode right)` / `ScalarNode operator *(ScalarNode left, ScalarNode right)` / `ScalarNode operator -(ScalarNode left, ScalarNode right)` / `ScalarNode operator -(ScalarNode operand)`

### `readonly struct SingularValueDecompositionNode`

Represents a thin singular value decomposition `A = UΣVᵀ` recorded on a `Tape`. / `Tape` に記録された薄い特異値分解 `A = UΣVᵀ` を表すノードです。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Columns { get; }` | Gets the column count `n` of the factored matrix. / 元の行列の列数 `n` を取得します。 |
| `int Rows { get; }` | Gets the row count `m` of the factored matrix. / 元の行列の行数 `m` を取得します。 |
| `MatrixNode<double> U { get; }` | Gets the `m × min(m, n)` matrix `U` whose columns are the left singular vectors. / 左特異ベクトルを列とする `m × min(m, n)` 行列 `U` のノードを取得します。 |
| `MatrixNode<double> V { get; }` | Gets the `n × min(m, n)` matrix `V` whose columns are the right singular vectors. / 右特異ベクトルを列とする `n × min(m, n)` 行列 `V` のノードを取得します。 |
| `VectorNode<double> Values { get; }` | Gets the non-negative singular values in descending order. / 降順かつ非負の特異値のベクトルノードを取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static SingularValueDecompositionNode Compute(MatrixNode<double> a)` | Computes the thin singular value decomposition of a matrix node. / 行列ノードの薄い特異値分解を計算します。 |
| `MatrixNode<double> PseudoInverse(double? tolerance = default)` | Returns the Moore–Penrose pseudoinverse computed with the retained singular values. / 保持した特異値を使ってムーア・ペンローズ疑似逆行列を求めます。 |
| `VectorNode<double> Solve(VectorNode<double> b, double? tolerance = default)` | Returns the minimum-norm least-squares solution for a vector right-hand side. / ベクトル右辺に対して、最小二乗解のうちノルムが最小の解を返します。 |
| `MatrixNode<double> Solve(MatrixNode<double> b, double? tolerance = default)` | Returns the minimum-norm least-squares solutions for the columns of a matrix right-hand side. / 右辺行列の列ごとに、最小二乗解のうちノルムが最小の解を返します。 |

### `readonly struct SymmetricEigenDecompositionNode`

Represents an eigendecomposition `A = VΛVᵀ` of a real symmetric matrix recorded on a `Tape`. / `Tape` に記録された実対称行列の固有値分解 `A = VΛVᵀ` を表すノードです。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Order { get; }` | Gets the order `n` of the factored matrix. / 分解対象の行列の次数 `n` を取得します。 |
| `VectorNode<double> Values { get; }` | Gets the eigenvalues in ascending order. / 昇順に並んだ固有値のベクトルノードを取得します。 |
| `MatrixNode<double> Vectors { get; }` | Gets the orthogonal matrix `V` whose columns are the eigenvectors. / 固有ベクトルを列とする直交行列 `V` のノードを取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static SymmetricEigenDecompositionNode Compute(MatrixNode<double> a)` | Computes the eigendecomposition of a real symmetric matrix node. / 実対称行列ノードの固有値分解を計算します。 |

### `class Tape`

Records operations for reverse-mode automatic differentiation. / 逆モード自動微分のために演算を記録するテープを表します。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `Tape()` | Initializes a new empty automatic differentiation tape. / 空の自動微分テープを初期化します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `VectorNode<T> Apply<T, TOperation>(TOperation operation, ReadOnlySpan<VectorNode<T>> inputs, int length)` | Applies a user operation and returns a vector node of the specified length. / 利用者定義演算を適用し、指定した長さのベクトルノードを返します。 |
| `MatrixNode<T> Apply<T, TOperation>(TOperation operation, ReadOnlySpan<MatrixNode<T>> inputs, int rows, int columns)` | Applies a user operation and returns a matrix node of the specified shape. / 利用者定義演算を適用し、指定した形状の行列ノードを返します。 |
| `void Backward(ScalarNode output)` | Runs reverse-mode backpropagation starting from `output` with an initial adjoint of 1. / `output` の随伴を 1 として逆伝播を実行します。 |
| `VectorNode<T> Constant<T>(ReadOnlyVectorSpan<T> values)  (+2 overloads)` | Creates a non-differentiable constant vector node by copying the specified span into the tape. / 指定したスパンをテープにコピーして定数ベクトルノードを作成します。 |
| `MatrixNode<T> Constant<T>(ReadOnlyMatrixSpan<T> values)  (+2 overloads)` | Creates a non-differentiable constant matrix node by copying the specified span into the tape. / 指定したスパンをテープにコピーして定数行列ノードを作成します。 |
| `TensorNode<T> Constant<T>(ReadOnlyTensorSpan<T> values)  (+2 overloads)` | Creates a constant tensor node by copying the view into the tape. / テンソルビューをテープにコピーして定数ノードを作成します。 |
| `MatrixNode<T> ConstantMatrix<T>(Parameter<T> parameter)` | Creates a non-differentiable constant matrix node referencing the values of a `Parameter` without copying. / 行列の学習パラメーター `Parameter` を定数としてコピーせずに参照するノードを作成します。 |
| `TensorNode<T> ConstantTensor<T>(Parameter<T> parameter)` | References the values of a rank-three parameter as a constant node. / 3 階テンソルの学習パラメーターの値を定数ノードとして参照します。 |
| `VectorNode<T> ConstantVector<T>(Parameter<T> parameter)` | Creates a non-differentiable constant vector node referencing the values of a `Parameter` without copying. / ベクトルの学習パラメーター `Parameter` を定数としてコピーせずに参照するノードを作成します。 |
| `ReadOnlyVectorSpan<T> GetGradient<T>(VectorNode<T> node)` | Gets a read-only view of the gradient of the specified vector node after `Backward`. / `Backward` 実行後の指定したベクトルノードの勾配を読み取り専用ビューとして取得します。 |
| `ReadOnlyMatrixSpan<T> GetGradient<T>(MatrixNode<T> node)` | Gets a read-only view of the gradient of the specified matrix node after `Backward`. / `Backward` 実行後の指定した行列ノードの勾配を読み取り専用ビューとして取得します。 |
| `double GetGradient(ScalarNode node)` | Gets the gradient of the specified scalar node computed by `Backward`. / `Backward` 実行後の指定したスカラーノードの勾配を取得します。 |
| `ReadOnlyTensorSpan<T> GetGradient<T>(TensorNode<T> node)` | Gets a read-only gradient view after backpropagation. / 逆伝播後の勾配を読み取り専用ビューとして取得します。 |
| `double Gradient(ScalarNode output, VectorSpan<double> gradient)` | Runs backpropagation from `output` and writes the flattened leaf gradients to `gradient`. / `output` から逆伝播を実行し、すべての葉の勾配を `gradient` に書き込みます。 |
| `double Gradient<T>(ScalarNode output, VectorNode<T> node, VectorSpan<T> gradient)` | Runs backpropagation from `output` and writes the gradient of `node` to `gradient`. / `output` から逆伝播を実行し、`node` の勾配を `gradient` に書き込みます。 |
| `double Gradient<T>(ScalarNode output, MatrixNode<T> node, MatrixSpan<T> gradient)` | Runs backpropagation from `output` and writes the gradient of `node` to `gradient`. / `output` から逆伝播を実行し、`node` の勾配を `gradient` に書き込みます。 |
| `double Gradient<T>(ScalarNode output, TensorNode<T> node, TensorSpan<T> gradient)` | Runs backpropagation and copies the gradient of the tensor node into the destination. / 逆伝播を実行し、テンソルノードの勾配を出力先にコピーします。 |
| `VectorNode<T> Leaf<T>(ReadOnlyVectorSpan<T> values)  (+2 overloads)` | Creates a differentiable vector leaf node by copying the specified values into the tape. / 指定した値をテープにコピーして、微分可能なベクトルの葉ノードを作成します。 |
| `MatrixNode<T> Leaf<T>(ReadOnlyMatrixSpan<T> values)  (+2 overloads)` | Creates a differentiable matrix leaf node by copying the specified values into the tape. / 指定した値をテープにコピーして、微分可能な行列の葉ノードを作成します。 |
| `ScalarNode Leaf(double value)` | Creates a differentiable scalar leaf node with the specified value. / 指定した値を持つ微分可能なスカラーの葉ノードを作成します。 |
| `TensorNode<T> Leaf<T>(ReadOnlyTensorSpan<T> values)  (+2 overloads)` | Creates a differentiable tensor leaf by copying the view into the tape. / テンソルビューをテープにコピーして微分対象の葉を作成します。 |
| `ReadOnlySpan<ScalarNode> Leaves(ReadOnlyVectorSpan<double> values)` | Creates scalar leaf nodes in order for each element of the specified vector. / 指定したベクトルの各要素から、順にスカラーの葉ノードを作成します。 |
| `void Reset()` | Discards the current recording and prepares the tape for reuse. / 現在の記録を破棄し、次の計算で再利用できる状態に戻します。 |
| `MatrixNode<T> UseMatrix<T>(Parameter<T> parameter)` | Attaches a matrix `Parameter` to the current recording as a differentiable node without copying. / 行列の学習パラメーター `Parameter` をコピーせずに現在の記録へ参加させます。 |
| `TensorNode<T> UseTensor<T>(Parameter<T> parameter)` | Attaches a rank-three parameter to the tape without copying its values or gradients. / 値と勾配をコピーせずに、3 階テンソルの学習パラメーターをテープへ参加させます。 |
| `VectorNode<T> UseVector<T>(Parameter<T> parameter)` | Attaches a vector `Parameter` to the current recording as a differentiable node without copying. / ベクトルの学習パラメーター `Parameter` をコピーせずに現在の記録へ参加させます。 |

### `readonly struct TensorNode<TElement>`

Represents a three-dimensional tensor in a reverse-mode recording. / 逆モード自動微分の記録に属する3階テンソルを表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `TensorNode<TElement> this[Range rows, Range columns, Range matrices] { get; }` | Gets a zero-copy view over the specified ranges. / 指定した範囲のビューをコピーなしで取得します。 |
| `int Columns { get; }` | Gets the column count of each matrix. / 各行列の列数を取得します。 |
| `int MatrixCount { get; }` | Gets the number of matrices along the last axis. / 最後の軸の行列数を取得します。 |
| `bool RequiresGradient { get; }` | Gets whether the node depends on a differentiable leaf or parameter. / ノードが微分対象の葉またはパラメーターに依存しているかどうかを取得します。 |
| `int Rows { get; }` | Gets the row count of each matrix. / 各行列の行数を取得します。 |
| `TensorNode<TElement> T { get; }` | Gets a zero-copy view that transposes each matrix. / 各行列を転置したコピーなしのビューを取得します。 |
| `ReadOnlyTensorSpan<TElement> Value { get; }` | Gets a read-only view of the recorded values. / 記録された値の読み取り専用ビューを取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `MatrixNode<TElement> GetMatrix(int index)` | Gets a differentiable view of one matrix without copying. / 指定した行列を微分可能なビューとしてコピーなしで取得します。 |
| `static TensorNode<TElement> Record<TOperation>(TOperation operation, ReadOnlySpan<MatrixNode<TElement>> inputs, int rows, int columns, int matrixCount)` | Records a user operation with a tensor result. / テンソルを返す利用者定義演算を記録します。 |
| `MatrixNode<TElement> ToMatrix()` | Places the matrices side by side in a column-major matrix node. / 各行列を横に並べた列優先の行列ノードを返します。 |

**演算子**

`TensorNode<TElement> operator *(TensorNode<TElement> a, TensorNode<TElement> b)`

### `readonly struct VectorNode<T>`

Represents a vector node for reverse-mode automatic differentiation. / 逆モード自動微分のベクトルノードを表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `VectorNode<T> this[Range range] { get; }` | Gets a zero-copy subvector view over the specified range. / 指定した範囲の部分ベクトルビューをコピーなしで取得します。 |
| `ScalarNode this[int index] { get; }` | Selects a vector element as a scalar node. / ベクトルの要素をスカラーノードとして取得します。 |
| `int Length { get; }` | Gets the number of elements in the vector. / ベクトルの要素数を取得します。 |
| `bool RequiresGradient { get; }` | Gets a value indicating whether this node depends on any differentiable leaf or `Parameter`. / このノードが微分対象の葉または `Parameter` に依存しているかどうかを示す値を取得します。 |
| `ReadOnlyVectorSpan<T> Value { get; }` | Gets a read-only view of the node's values. / ノードの値の読み取り専用ビューを取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `MatrixNode<T> AsColumnMatrix()` | Gets a single-column matrix node sharing this vector's values and gradients. / このベクトルの値と勾配を共有する 1 列の行列ノードを取得します。 |
| `MatrixNode<T> Broadcast(int count)` | Creates a zero-copy matrix view repeating this vector across `count` columns. / このベクトルを `count` 列に繰り返し配置した行列ビューをコピーなしで作成します。 |
| `MatrixNode<T> BroadcastRows(int rows)` | Creates a zero-copy matrix view repeating this vector across `rows` rows. / このベクトルを `rows` 行に繰り返し配置した行列ビューをコピーなしで作成します。 |
| `static VectorNode<T> Record<TOperation>(TOperation operation, ReadOnlySpan<VectorNode<T>> inputs, int length)` | Records a user operation on the tape shared by its vector inputs. / ベクトル入力が共有するテープに利用者定義演算を記録します。 |

**演算子**

`VectorNode<T> operator +(ScalarNode s, VectorNode<T> x)` / `VectorNode<T> operator +(VectorNode<T> x, ScalarNode s)` / `VectorNode<T> operator +(VectorNode<T> x, VectorNode<T> y)` / `VectorNode<T> operator /(VectorNode<T> x, ScalarNode s)` / `VectorNode<T> operator *(MatrixNode<T> a, VectorNode<T> x)` / `VectorNode<T> operator *(ScalarNode s, VectorNode<T> x)` / `VectorNode<T> operator *(VectorNode<T> x, ScalarNode s)` / `VectorNode<T> operator -(ScalarNode s, VectorNode<T> x)` / `VectorNode<T> operator -(VectorNode<T> x, ScalarNode s)` / `VectorNode<T> operator -(VectorNode<T> x, VectorNode<T> y)` / `VectorNode<T> operator -(VectorNode<T> x)`

### `static class XMathExtensions`

Provides `XMath` extension methods for building differentiable scalar, vector, matrix, and tensor expressions. / スカラー・ベクトル・行列・テンソルの微分可能な計算式を組み立てるための `XMath` 拡張メソッドを提供します。

- 呼び出し形式: `using Tuna.Differentiation;` のうえで `XMath.<Method>(...)` として呼び出します（`using static` では解決されません）。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static VectorNode<T> Abs<T>(VectorNode<T> x)` | Computes the absolute value `\|x\|` elementwise. / 各要素の絶対値 `\|x\|` を計算します。 |
| `static MatrixNode<T> Abs<T>(MatrixNode<T> x)` | Computes the absolute value `\|x\|` elementwise. / 各要素の絶対値 `\|x\|` を計算します。 |
| `static ScalarNode Abs(ScalarNode x)` | Computes the absolute value `\|x\|`. / 絶対値 `\|x\|` を計算します。 |
| `static VectorNode<T> Add<T>(VectorNode<T> x, VectorNode<T> y)` | Adds two vector nodes elementwise, `x + y`. / 2 つのベクトルノードの要素ごとの和 `x + y` を計算します。 |
| `static MatrixNode<T> Add<T>(MatrixNode<T> x, MatrixNode<T> y)` | Adds two matrix nodes elementwise, `x + y`. / 2 つの行列ノードの要素ごとの和 `x + y` を計算します。 |
| `static VectorNode<T> Add<T>(VectorNode<T> x, ScalarNode scalar)` | Adds a scalar to each element of a vector, `x + s`. / ベクトルの各要素にスカラーを加算した `x + s` を計算します。 |
| `static MatrixNode<T> Add<T>(MatrixNode<T> x, ScalarNode scalar)` | Adds a scalar to each element of a matrix, `x + s`. / 行列の各要素にスカラーを加算した `x + s` を計算します。 |
| `static ScalarNode Atan(ScalarNode x)` | Computes the inverse tangent `atan(x)`. / 逆正接 `atan(x)` を計算します。 |
| `static ScalarNode CholeskyLogDeterminant(MatrixNode<double> a)` | Computes the natural logarithm of the determinant of a symmetric positive definite matrix node. / 対称正定値行列ノードの行列式の自然対数を計算します。 |
| `static MatrixNode<T> ColumnLogSoftmax<T>(MatrixNode<T> x)` | Computes the log-softmax transformation along each column. / 各列に対して対数ソフトマックス変換を計算します。 |
| `static VectorNode<T> ColumnLogSumExp<T>(MatrixNode<T> x)` | Computes `ln Σ exp` for each column. / 各列の `ln Σ exp` を計算します。 |
| `static VectorNode<T> ColumnMeans<T>(MatrixNode<T> x)` | Computes the arithmetic mean of each column. / 各列の算術平均を計算します。 |
| `static MatrixNode<T> ColumnSoftmax<T>(MatrixNode<T> x)` | Computes the softmax transformation along each column. / 各列に対してソフトマックス変換を計算します。 |
| `static VectorNode<T> ColumnSums<T>(MatrixNode<T> x)` | Computes the sum of each column. / 各列の総和を計算します。 |
| `static VectorNode<T> ColumnSumsOfSquares<T>(MatrixNode<T> x)` | Computes the sum of squares of each column. / 各列の 2 乗和を計算します。 |
| `static ScalarNode Cos(ScalarNode x)` | Computes the cosine `cos(x)`. / 余弦 `cos(x)` を計算します。 |
| `static VectorNode<T> Cross<T>(VectorNode<T> u, VectorNode<T> v)` | Computes the cross product `u × v` of two three-dimensional vector nodes. / 2 つの 3 次元ベクトルノードの外積 `u × v` を計算します。 |
| `static VectorNode<T> Divide<T>(VectorNode<T> x, VectorNode<T> y)` | Divides two vector nodes elementwise, `x / y`. / 2 つのベクトルノードの要素ごとの商 `x / y` を計算します。 |
| `static MatrixNode<T> Divide<T>(MatrixNode<T> x, MatrixNode<T> y)` | Divides two matrix nodes elementwise, `x / y`. / 2 つの行列ノードの要素ごとの商 `x / y` を計算します。 |
| `static VectorNode<T> Divide<T>(VectorNode<T> x, ScalarNode y)` | Divides each element of a vector by a scalar, `x / s`. / ベクトルの各要素をスカラーで割った `x / s` を計算します。 |
| `static MatrixNode<T> Divide<T>(MatrixNode<T> x, ScalarNode y)` | Divides each element of a matrix by a scalar, `x / s`. / 行列の各要素をスカラーで割った `x / s` を計算します。 |
| `static ScalarNode Dot<T>(VectorNode<T> x, VectorNode<T> y)` | Computes the dot product `Σ x[i]·y[i]`. / 内積 `Σ x[i]·y[i]` を計算します。 |
| `static VectorNode<T> Elu<T>(VectorNode<T> x, T alpha)` | Applies the ELU activation function (`x` for `x > 0`, otherwise `alpha·(eˣ - 1)`) elementwise. / 各要素に ELU 活性化関数（`x > 0` で `x`、それ以外で `alpha·(eˣ - 1)`）を適用します。 |
| `static MatrixNode<T> Elu<T>(MatrixNode<T> x, T alpha)` | Applies the ELU activation function (`x` for `x > 0`, otherwise `alpha·(eˣ - 1)`) elementwise. / 各要素に ELU 活性化関数（`x > 0` で `x`、それ以外で `alpha·(eˣ - 1)`）を適用します。 |
| `static MatrixNode<T> Embedding<T>(MatrixNode<T> table, ReadOnlySpan<int> indices)` | Gathers the indexed columns of an embedding table and records their gradients. / 埋め込みテーブルから指定された列を集め、その勾配を記録します。 |
| `static VectorNode<T> Exp<T>(VectorNode<T> x)` | Computes the natural exponential function `eˣ` elementwise. / 各要素の自然指数関数 `eˣ` を計算します。 |
| `static MatrixNode<T> Exp<T>(MatrixNode<T> x)` | Computes the natural exponential function `eˣ` elementwise. / 各要素の自然指数関数 `eˣ` を計算します。 |
| `static ScalarNode Exp(ScalarNode x)` | Computes the natural exponential function `eˣ`. / 自然指数関数 `eˣ` を計算します。 |
| `static VectorNode<T> Exp<T>(VectorNode<T> x, bool approximate)` | Computes the natural exponential function `eˣ` elementwise, optionally using an approximate `float` kernel. / 各要素の自然指数関数 `eˣ` を計算します（`float` では近似カーネルを選択できます）。 |
| `static MatrixNode<T> Exp<T>(MatrixNode<T> x, bool approximate)` | Computes the natural exponential function `eˣ` elementwise, optionally using an approximate `float` kernel. / 各要素の自然指数関数 `eˣ` を計算します（`float` では近似カーネルを選択できます）。 |
| `static VectorNode<T> Expm1<T>(VectorNode<T> x)` | Computes `eˣ - 1` elementwise, preserving accuracy near zero. / 0 の近傍でも精度を保って各要素の `eˣ - 1` を計算します。 |
| `static MatrixNode<T> Expm1<T>(MatrixNode<T> x)` | Computes `eˣ - 1` elementwise, preserving accuracy near zero. / 0 の近傍でも精度を保って各要素の `eˣ - 1` を計算します。 |
| `static ScalarNode Expm1(ScalarNode x)` | Computes `eˣ - 1` accurately near zero. / 0 の近傍でも精度を保って `eˣ - 1` を計算します。 |
| `static VectorNode<T> Gelu<T>(VectorNode<T> x)` | Applies the exact GELU activation function `x·Φ(x)` elementwise. / 各要素に厳密な GELU 活性化関数 `x·Φ(x)` を適用します。 |
| `static MatrixNode<T> Gelu<T>(MatrixNode<T> x)` | Applies the exact GELU activation function `x·Φ(x)` elementwise. / 各要素に厳密な GELU 活性化関数 `x·Φ(x)` を適用します。 |
| `static VectorNode<T> Gelu<T>(VectorNode<T> x, bool approximate)` | Applies the GELU activation function elementwise, optionally using the `tanh` approximation. / 各要素に GELU 活性化関数を適用します（オプションで `tanh` 近似式を使用できます）。 |
| `static MatrixNode<T> Gelu<T>(MatrixNode<T> x, bool approximate)` | Applies the GELU activation function elementwise, optionally using the `tanh` approximation. / 各要素に GELU 活性化関数を適用します（オプションで `tanh` 近似式を使用できます）。 |
| `static VectorNode<T> Hadamard<T>(VectorNode<T> x, VectorNode<T> y)` | Computes the elementwise (Hadamard) product `x ∘ y`. / 要素ごとの積（アダマール積）`x ∘ y` を計算します。 |
| `static MatrixNode<T> Hadamard<T>(MatrixNode<T> x, MatrixNode<T> y)` | Computes the elementwise (Hadamard) product `x ∘ y`. / 要素ごとの積（アダマール積）`x ∘ y` を計算します。 |
| `static ScalarNode Hypot(ScalarNode x, ScalarNode y)` | Computes `√(x² + y²)` and records its partial derivatives. / `√(x² + y²)` を計算し、偏微分を記録します。 |
| `static VectorNode<T> Hypot<T>(VectorNode<T> x, VectorNode<T> y)` | Computes `√(x² + y²)` elementwise for two vectors. / 2 つのベクトルの要素ごとに `√(x² + y²)` を計算します。 |
| `static MatrixNode<T> Hypot<T>(MatrixNode<T> x, MatrixNode<T> y)` | Computes `√(x² + y²)` elementwise for two matrices. / 2 つの行列の要素ごとに `√(x² + y²)` を計算します。 |
| `static VectorNode<T> Hypot<T>(VectorNode<T> x, ScalarNode y)` | Computes `√(x[i]² + y²)` for each element of a vector. / ベクトルの要素ごとに `√(x[i]² + y²)` を計算します。 |
| `static MatrixNode<T> Hypot<T>(MatrixNode<T> x, ScalarNode y)` | Computes `√(x[i,j]² + y²)` for each element of a matrix. / 行列の要素ごとに `√(x[i,j]² + y²)` を計算します。 |
| `static VectorNode<T> Hypot<T>(ScalarNode x, VectorNode<T> y)` | Computes `√(x² + y[i]²)` for each element of a vector. / ベクトルの要素ごとに `√(x² + y[i]²)` を計算します。 |
| `static MatrixNode<T> Hypot<T>(ScalarNode x, MatrixNode<T> y)` | Computes `√(x² + y[i,j]²)` for each element of a matrix. / 行列の要素ごとに `√(x² + y[i,j]²)` を計算します。 |
| `static VectorNode<T> LeakyRelu<T>(VectorNode<T> x, T slope)` | Applies the Leaky ReLU activation function (`x` for `x > 0`, otherwise `slope·x`) elementwise. / 各要素に Leaky ReLU 活性化関数（`x > 0` で `x`、それ以外で `slope·x`）を適用します。 |
| `static MatrixNode<T> LeakyRelu<T>(MatrixNode<T> x, T slope)` | Applies the Leaky ReLU activation function (`x` for `x > 0`, otherwise `slope·x`) elementwise. / 各要素に Leaky ReLU 活性化関数（`x > 0` で `x`、それ以外で `slope·x`）を適用します。 |
| `static VectorNode<T> Log<T>(VectorNode<T> x)` | Computes the natural logarithm `ln x` elementwise. / 各要素の自然対数 `ln x` を計算します。 |
| `static MatrixNode<T> Log<T>(MatrixNode<T> x)` | Computes the natural logarithm `ln x` elementwise. / 各要素の自然対数 `ln x` を計算します。 |
| `static ScalarNode Log(ScalarNode x)` | Computes the natural logarithm `ln x`. / 自然対数 `ln x` を計算します。 |
| `static VectorNode<T> Log1mExp<T>(VectorNode<T> x)` | Computes `ln(1 - eˣ)` elementwise on the non-positive domain. / 非正の定義域で各要素の `ln(1 - eˣ)` を計算します。 |
| `static MatrixNode<T> Log1mExp<T>(MatrixNode<T> x)` | Computes `ln(1 - eˣ)` elementwise on the non-positive domain. / 非正の定義域で各要素の `ln(1 - eˣ)` を計算します。 |
| `static ScalarNode Log1mExp(ScalarNode x)` | Computes `ln(1 - eˣ)` on the non-positive domain. / 非正の定義域で `ln(1 - eˣ)` を計算します。 |
| `static VectorNode<T> Log1p<T>(VectorNode<T> x)` | Computes `ln(1 + x)` elementwise, preserving accuracy near zero. / 0 の近傍でも精度を保って各要素の `ln(1 + x)` を計算します。 |
| `static MatrixNode<T> Log1p<T>(MatrixNode<T> x)` | Computes `ln(1 + x)` elementwise, preserving accuracy near zero. / 0 の近傍でも精度を保って各要素の `ln(1 + x)` を計算します。 |
| `static ScalarNode Log1p(ScalarNode x)` | Computes `ln(1 + x)` accurately near zero. / 0 の近傍でも精度を保って `ln(1 + x)` を計算します。 |
| `static VectorNode<T> LogAddExp<T>(VectorNode<T> x, VectorNode<T> y)` | Computes `ln(eˣ + eʸ)` elementwise. / 要素ごとに `ln(eˣ + eʸ)` を計算します。 |
| `static MatrixNode<T> LogAddExp<T>(MatrixNode<T> x, MatrixNode<T> y)` | Computes `ln(eˣ + eʸ)` elementwise. / 要素ごとに `ln(eˣ + eʸ)` を計算します。 |
| `static VectorNode<T> LogAddExp<T>(VectorNode<T> x, ScalarNode y)` | Computes `ln(eˣ + eˢ)` elementwise with a scalar. / ベクトルの各要素とスカラーの `ln(eˣ + eˢ)` を計算します。 |
| `static MatrixNode<T> LogAddExp<T>(MatrixNode<T> x, ScalarNode y)` | Computes `ln(eˣ + eˢ)` elementwise with a scalar. / 行列の各要素とスカラーの `ln(eˣ + eˢ)` を計算します。 |
| `static ScalarNode LogAddExp(ScalarNode x, ScalarNode y)` | Computes `ln(eˣ + eʸ)` stably. / 指数の和の対数 `ln(eˣ + eʸ)` を安定に計算します。 |
| `static VectorNode<T> LogLogistic<T>(VectorNode<T> x)` | Computes the natural logarithm of the logistic sigmoid function, `ln σ(x)`, elementwise. / 各要素のロジスティック関数の自然対数 `ln σ(x)` を計算します。 |
| `static MatrixNode<T> LogLogistic<T>(MatrixNode<T> x)` | Computes the natural logarithm of the logistic sigmoid function, `ln σ(x)`, elementwise. / 各要素のロジスティック関数の自然対数 `ln σ(x)` を計算します。 |
| `static ScalarNode LogLogistic(ScalarNode x)` | Computes the natural logarithm of the logistic sigmoid function, `ln σ(x) = -ln(1 + e⁻ˣ)`. / ロジスティック関数の自然対数 `ln σ(x) = -ln(1 + e⁻ˣ)` を計算します。 |
| `static VectorNode<T> LogSoftmax<T>(VectorNode<T> x)` | Computes the log-softmax transformation `x - ln Σ exp(x[k])` over the vector. / ベクトル全体の対数ソフトマックス変換 `x - ln Σ exp(x[k])` を計算します。 |
| `static VectorNode<T> LogSubExp<T>(VectorNode<T> x, VectorNode<T> y)` | Computes `ln(eˣ - eʸ)` elementwise. / 要素ごとに `ln(eˣ - eʸ)` を計算します。 |
| `static MatrixNode<T> LogSubExp<T>(MatrixNode<T> x, MatrixNode<T> y)` | Computes `ln(eˣ - eʸ)` elementwise. / 要素ごとに `ln(eˣ - eʸ)` を計算します。 |
| `static VectorNode<T> LogSubExp<T>(VectorNode<T> x, ScalarNode y)` | Computes `ln(eˣ - eˢ)` elementwise with a scalar. / ベクトルの各要素とスカラーの `ln(eˣ - eˢ)` を計算します。 |
| `static MatrixNode<T> LogSubExp<T>(MatrixNode<T> x, ScalarNode y)` | Computes `ln(eˣ - eˢ)` elementwise with a scalar. / 行列の各要素とスカラーの `ln(eˣ - eˢ)` を計算します。 |
| `static ScalarNode LogSubExp(ScalarNode x, ScalarNode y)` | Computes `ln(eˣ - eʸ)` stably for `x ≥ y`. / `x ≥ y` の範囲で指数の差の対数 `ln(eˣ - eʸ)` を安定に計算します。 |
| `static ScalarNode LogSumExp<T>(VectorNode<T> x)` | Computes the log-sum-exp `ln Σ exp(x[i])`. / 指数の総和の対数 `ln Σ exp(x[i])` を計算します。 |
| `static ScalarNode LogSumExp(ReadOnlySpan<ScalarNode> x)` | Computes the log-sum-exp `ln Σ eˣⁱ` over a span of scalar nodes. / スカラーノード列の指数の総和の対数 `ln Σ eˣⁱ` を計算します。 |
| `static VectorNode<T> Logistic<T>(VectorNode<T> x)` | Computes the logistic sigmoid function `σ(x) = 1/(1 + e⁻ˣ)` elementwise. / 各要素のロジスティック関数 `σ(x) = 1/(1 + e⁻ˣ)` を計算します。 |
| `static MatrixNode<T> Logistic<T>(MatrixNode<T> x)` | Computes the logistic sigmoid function `σ(x) = 1/(1 + e⁻ˣ)` elementwise. / 各要素のロジスティック関数 `σ(x) = 1/(1 + e⁻ˣ)` を計算します。 |
| `static ScalarNode Logistic(ScalarNode x)` | Computes the logistic sigmoid function `σ(x) = 1/(1 + e⁻ˣ)`. / ロジスティック関数 `σ(x) = 1/(1 + e⁻ˣ)` を計算します。 |
| `static VectorNode<T> Logistic<T>(VectorNode<T> x, bool approximate)` | Computes the logistic sigmoid function `σ(x) = 1/(1 + e⁻ˣ)` elementwise, optionally using an approximate `float` kernel. / 各要素のロジスティック関数 `σ(x) = 1/(1 + e⁻ˣ)` を計算します（`float` では近似カーネルを選択できます）。 |
| `static MatrixNode<T> Logistic<T>(MatrixNode<T> x, bool approximate)` | Computes the logistic sigmoid function `σ(x) = 1/(1 + e⁻ˣ)` elementwise, optionally using an approximate `float` kernel. / 各要素のロジスティック関数 `σ(x) = 1/(1 + e⁻ˣ)` を計算します（`float` では近似カーネルを選択できます）。 |
| `static ScalarNode LogisticCrossEntropy<T>(VectorNode<T> logits, VectorNode<T> targets)` | Computes the summed logistic (binary) cross-entropy loss `Σᵢ [yᵢ·softplus(-ηᵢ) + (1 - yᵢ)·softplus(ηᵢ)]`. / ロジスティック交差エントロピーの総和 `Σᵢ [yᵢ·softplus(-ηᵢ) + (1 - yᵢ)·softplus(ηᵢ)]` を計算します。 |
| `static ScalarNode LogisticCrossEntropy<T>(MatrixNode<T> logits, MatrixNode<T> targets)` | Computes the summed logistic (binary) cross-entropy loss `Σᵢⱼ [Yᵢⱼ·softplus(-Hᵢⱼ) + (1 - Yᵢⱼ)·softplus(Hᵢⱼ)]` over a matrix of independent binary logits. / 独立な 2 値ロジット行列に対するロジスティック交差エントロピーの総和 `Σᵢⱼ [Yᵢⱼ·softplus(-Hᵢⱼ) + (1 - Yᵢⱼ)·softplus(Hᵢⱼ)]` を計算します。 |
| `static VectorNode<T> Logit<T>(VectorNode<T> x)` | Computes the logit function `ln(p/(1 - p))` elementwise. / 各要素のロジット関数 `ln(p/(1 - p))` を計算します。 |
| `static MatrixNode<T> Logit<T>(MatrixNode<T> x)` | Computes the logit function `ln(p/(1 - p))` elementwise. / 各要素のロジット関数 `ln(p/(1 - p))` を計算します。 |
| `static ScalarNode Logit(ScalarNode p)` | Computes the logit function `ln(p/(1 - p))`. / ロジット関数 `ln(p/(1 - p))` を計算します。 |
| `static VectorNode<T> Max<T>(VectorNode<T> x, VectorNode<T> y)` | Computes the elementwise maximum `max(x, y)`. / 要素ごとの最大値 `max(x, y)` を計算します。 |
| `static MatrixNode<T> Max<T>(MatrixNode<T> x, MatrixNode<T> y)` | Computes the elementwise maximum `max(x, y)`. / 要素ごとの最大値 `max(x, y)` を計算します。 |
| `static VectorNode<T> Max<T>(VectorNode<T> x, ScalarNode y)` | Computes the elementwise maximum with a scalar, `max(x, s)`. / ベクトルの各要素とスカラーの最大値 `max(x, s)` を計算します。 |
| `static MatrixNode<T> Max<T>(MatrixNode<T> x, ScalarNode y)` | Computes the elementwise maximum with a scalar, `max(x, s)`. / 行列の各要素とスカラーの最大値 `max(x, s)` を計算します。 |
| `static ScalarNode Max(ScalarNode left, ScalarNode right)` | Returns the larger of two scalar nodes, `max(left, right)`. / 2 つのスカラーノードの最大値 `max(left, right)` を返します。 |
| `static ScalarNode Mean<T>(VectorNode<T> x)` | Computes the arithmetic mean `Σ x[i] / n`. / 算術平均 `Σ x[i] / n` を計算します。 |
| `static VectorNode<T> Min<T>(VectorNode<T> x, VectorNode<T> y)` | Computes the elementwise minimum `min(x, y)`. / 要素ごとの最小値 `min(x, y)` を計算します。 |
| `static MatrixNode<T> Min<T>(MatrixNode<T> x, MatrixNode<T> y)` | Computes the elementwise minimum `min(x, y)`. / 要素ごとの最小値 `min(x, y)` を計算します。 |
| `static VectorNode<T> Min<T>(VectorNode<T> x, ScalarNode y)` | Computes the elementwise minimum with a scalar, `min(x, s)`. / ベクトルの各要素とスカラーの最小値 `min(x, s)` を計算します。 |
| `static MatrixNode<T> Min<T>(MatrixNode<T> x, ScalarNode y)` | Computes the elementwise minimum with a scalar, `min(x, s)`. / 行列の各要素とスカラーの最小値 `min(x, s)` を計算します。 |
| `static ScalarNode Min(ScalarNode left, ScalarNode right)` | Returns the smaller of two scalar nodes, `min(left, right)`. / 2 つのスカラーノードの最小値 `min(left, right)` を返します。 |
| `static MatrixNode<T> Multiply<T>(MatrixNode<T> a, MatrixNode<T> b)` | Computes the matrix product `AB`. / 行列積 `AB` を計算します。 |
| `static VectorNode<T> Multiply<T>(MatrixNode<T> a, VectorNode<T> x)` | Computes the matrix-vector product `Ax`. / 行列ベクトル積 `Ax` を計算します。 |
| `static VectorNode<T> Multiply<T>(VectorNode<T> x, ScalarNode scalar)` | Multiplies each element of a vector by a scalar, `s·x`. / ベクトルの各要素をスカラー倍した `s·x` を計算します。 |
| `static MatrixNode<T> Multiply<T>(MatrixNode<T> x, ScalarNode scalar)` | Multiplies each element of a matrix by a scalar, `s·x`. / 行列の各要素をスカラー倍した `s·x` を計算します。 |
| `static TensorNode<T> Multiply<T>(TensorNode<T> a, TensorNode<T> b)` | Records the batched product of the matrices at matching indices. / インデックスが一致する行列ごとの積を記録します。 |
| `static VectorNode<T> Negate<T>(VectorNode<T> x)` | Negates each element, `-x`. / 各要素の符号を反転した `-x` を計算します。 |
| `static MatrixNode<T> Negate<T>(MatrixNode<T> x)` | Negates each element, `-x`. / 各要素の符号を反転した `-x` を計算します。 |
| `static MatrixNode<T> Outer<T>(VectorNode<T> u, VectorNode<T> v)` | Computes the outer product `u vᵀ`. / 2 つのベクトルの外積（直積）`u vᵀ` を計算します。 |
| `static VectorNode<T> Pow<T>(VectorNode<T> x, VectorNode<T> y)` | Computes the elementwise power `x^y`. / 要素ごとの冪乗 `x^y` を計算します。 |
| `static MatrixNode<T> Pow<T>(MatrixNode<T> x, MatrixNode<T> y)` | Computes the elementwise power `x^y`. / 要素ごとの冪乗 `x^y` を計算します。 |
| `static VectorNode<T> Pow<T>(VectorNode<T> x, ScalarNode y)` | Raises each element of a vector to a scalar power, `x^s`. / ベクトルの各要素のスカラー冪 `x^s` を計算します。 |
| `static MatrixNode<T> Pow<T>(MatrixNode<T> x, ScalarNode y)` | Raises each element of a matrix to a scalar power, `x^s`. / 行列の各要素のスカラー冪 `x^s` を計算します。 |
| `static ScalarNode Pow(ScalarNode x, ScalarNode y)` | Computes the power `x^y`. / 冪乗 `x^y` を計算します。 |
| `static VectorNode<T> Reciprocal<T>(VectorNode<T> x)` | Computes the reciprocal `1/x` elementwise. / 各要素の逆数 `1/x` を計算します。 |
| `static MatrixNode<T> Reciprocal<T>(MatrixNode<T> x)` | Computes the reciprocal `1/x` elementwise. / 各要素の逆数 `1/x` を計算します。 |
| `static VectorNode<T> Relu<T>(VectorNode<T> x)` | Applies the ReLU activation function `max(x, 0)` elementwise. / 各要素に ReLU 活性化関数 `max(x, 0)` を適用します。 |
| `static MatrixNode<T> Relu<T>(MatrixNode<T> x)` | Applies the ReLU activation function `max(x, 0)` elementwise. / 各要素に ReLU 活性化関数 `max(x, 0)` を適用します。 |
| `static MatrixNode<T> RowLogSoftmax<T>(MatrixNode<T> x)` | Computes the log-softmax transformation along each row. / 各行に対して対数ソフトマックス変換を計算します。 |
| `static VectorNode<T> RowLogSumExp<T>(MatrixNode<T> x)` | Computes `ln Σ exp` for each row. / 各行の `ln Σ exp` を計算します。 |
| `static VectorNode<T> RowMeans<T>(MatrixNode<T> x)` | Computes the arithmetic mean of each row. / 各行の算術平均を計算します。 |
| `static MatrixNode<T> RowSoftmax<T>(MatrixNode<T> x)` | Computes the softmax transformation along each row. / 各行に対してソフトマックス変換を計算します。 |
| `static VectorNode<T> RowSums<T>(MatrixNode<T> x)` | Computes the sum of each row. / 各行の総和を計算します。 |
| `static VectorNode<T> RowSumsOfSquares<T>(MatrixNode<T> x)` | Computes the sum of squares of each row. / 各行の 2 乗和を計算します。 |
| `static VectorNode<T> Selu<T>(VectorNode<T> x)` | Applies the SELU activation function `λ·ELU(x; α)` (`α ≈ 1.6733`, `λ ≈ 1.0507`) elementwise. / 各要素に SELU 活性化関数 `λ·ELU(x; α)`（`α ≈ 1.6733`、`λ ≈ 1.0507`）を適用します。 |
| `static MatrixNode<T> Selu<T>(MatrixNode<T> x)` | Applies the SELU activation function `λ·ELU(x; α)` (`α ≈ 1.6733`, `λ ≈ 1.0507`) elementwise. / 各要素に SELU 活性化関数 `λ·ELU(x; α)`（`α ≈ 1.6733`、`λ ≈ 1.0507`）を適用します。 |
| `static VectorNode<T> Silu<T>(VectorNode<T> x)` | Applies the SiLU (Swish) activation function `x·σ(x)` elementwise. / 各要素に SiLU（Swish）活性化関数 `x·σ(x)` を適用します。 |
| `static MatrixNode<T> Silu<T>(MatrixNode<T> x)` | Applies the SiLU (Swish) activation function `x·σ(x)` elementwise. / 各要素に SiLU（Swish）活性化関数 `x·σ(x)` を適用します。 |
| `static VectorNode<T> Silu<T>(VectorNode<T> x, bool approximate)` | Applies the SiLU (Swish) activation function `x·σ(x)` elementwise, optionally using an approximate `float` kernel. / 各要素に SiLU（Swish）活性化関数 `x·σ(x)` を適用します（`float` では近似カーネルを選択できます）。 |
| `static MatrixNode<T> Silu<T>(MatrixNode<T> x, bool approximate)` | Applies the SiLU (Swish) activation function `x·σ(x)` elementwise, optionally using an approximate `float` kernel. / 各要素に SiLU（Swish）活性化関数 `x·σ(x)` を適用します（`float` では近似カーネルを選択できます）。 |
| `static ScalarNode Sin(ScalarNode x)` | Computes the sine `sin(x)`. / 正弦 `sin(x)` を計算します。 |
| `static VectorNode<T> Softmax<T>(VectorNode<T> x)` | Computes the softmax transformation `exp(x[i]) / Σ exp(x[k])` over the vector. / ベクトル全体のソフトマックス変換 `exp(x[i]) / Σ exp(x[k])` を計算します。 |
| `static ScalarNode SoftmaxCrossEntropy<T>(MatrixNode<T> logits, ReadOnlySpan<int> labels)` | Computes the summed softmax cross-entropy loss `-Σₙ ln softmax(Z[:, n])[t[n]]` for integer class labels. / 整数クラスラベルに対するソフトマックス交差エントロピーの総和 `-Σₙ ln softmax(Z[:, n])[t[n]]` を計算します。 |
| `static ScalarNode SoftmaxCrossEntropy<T>(VectorNode<T> logits, int label)` | Computes the softmax cross-entropy loss `-ln softmax(z)[t]` for a single sample. / 単一標本のソフトマックス交差エントロピー `-ln softmax(z)[t]` を計算します。 |
| `static ScalarNode SoftmaxCrossEntropy<T>(MatrixNode<T> logits, MatrixNode<T> targets)` | Computes the summed softmax cross-entropy loss `-Σₙ Σₖ T[k, n]·ln softmax(Z[:, n])[k]` against target distributions. / 目標分布に対するソフトマックス交差エントロピーの総和 `-Σₙ Σₖ T[k, n]·ln softmax(Z[:, n])[k]` を計算します。 |
| `static ScalarNode SoftmaxCrossEntropy<T>(VectorNode<T> logits, VectorNode<T> targets)` | Computes the softmax cross-entropy loss `-Σₖ t[k]·ln softmax(z)[k]` for a single sample against a target distribution. / 単一標本における目標分布とのソフトマックス交差エントロピー `-Σₖ t[k]·ln softmax(z)[k]` を計算します。 |
| `static VectorNode<T> Softplus<T>(VectorNode<T> x)` | Computes the softplus function `ln(1 + eˣ)` elementwise. / 各要素の Softplus 関数 `ln(1 + eˣ)` を計算します。 |
| `static MatrixNode<T> Softplus<T>(MatrixNode<T> x)` | Computes the softplus function `ln(1 + eˣ)` elementwise. / 各要素の Softplus 関数 `ln(1 + eˣ)` を計算します。 |
| `static ScalarNode Softplus(ScalarNode x)` | Computes the softplus function `ln(1 + eˣ)`. / Softplus 関数 `ln(1 + eˣ)` を計算します。 |
| `static VectorNode<T> Softplus<T>(VectorNode<T> x, bool approximate)` | Computes the softplus function `ln(1 + eˣ)` elementwise, optionally using an approximate `float` kernel. / 各要素の Softplus 関数 `ln(1 + eˣ)` を計算します（`float` では近似カーネルを選択できます）。 |
| `static MatrixNode<T> Softplus<T>(MatrixNode<T> x, bool approximate)` | Computes the softplus function `ln(1 + eˣ)` elementwise, optionally using an approximate `float` kernel. / 各要素の Softplus 関数 `ln(1 + eˣ)` を計算します（`float` では近似カーネルを選択できます）。 |
| `static VectorNode<double> Solve(MatrixNode<double> a, VectorNode<double> b)` | Solves the linear system `Ax = b` using partially pivoted LU decomposition. / 部分ピボット付き LU 分解を用いて連立一次方程式 `Ax = b` を解きます。 |
| `static MatrixNode<double> Solve(MatrixNode<double> a, MatrixNode<double> b)` | Solves the linear system `AX = B` for multiple right-hand sides using partially pivoted LU decomposition. / 部分ピボット付き LU 分解を用いて複数右辺の連立一次方程式 `AX = B` を解きます。 |
| `static VectorNode<double> SolveLowerTriangular(MatrixNode<double> a, VectorNode<double> b)` | Solves the lower-triangular linear system `Lx = b`. / 下三角行列を係数とする連立一次方程式 `Lx = b` を解きます。 |
| `static MatrixNode<double> SolveLowerTriangular(MatrixNode<double> a, MatrixNode<double> b)` | Solves the lower-triangular linear system `LX = B` for multiple right-hand sides. / 下三角行列を係数とする複数右辺の連立一次方程式 `LX = B` を解きます。 |
| `static VectorNode<double> SolveUpperTriangular(MatrixNode<double> a, VectorNode<double> b)` | Solves the upper-triangular linear system `Ux = b`. / 上三角行列を係数とする連立一次方程式 `Ux = b` を解きます。 |
| `static MatrixNode<double> SolveUpperTriangular(MatrixNode<double> a, MatrixNode<double> b)` | Solves the upper-triangular linear system `UX = B` for multiple right-hand sides. / 上三角行列を係数とする複数右辺の連立一次方程式 `UX = B` を解きます。 |
| `static VectorNode<T> Sqrt<T>(VectorNode<T> x)` | Computes the square root `√x` elementwise. / 各要素の平方根 `√x` を計算します。 |
| `static MatrixNode<T> Sqrt<T>(MatrixNode<T> x)` | Computes the square root `√x` elementwise. / 各要素の平方根 `√x` を計算します。 |
| `static ScalarNode Sqrt(ScalarNode x)` | Computes the square root `√x`. / 平方根 `√x` を計算します。 |
| `static VectorNode<T> Square<T>(VectorNode<T> x)` | Computes the square `x²` elementwise. / 各要素の 2 乗 `x²` を計算します。 |
| `static MatrixNode<T> Square<T>(MatrixNode<T> x)` | Computes the square `x²` elementwise. / 各要素の 2 乗 `x²` を計算します。 |
| `static ScalarNode Square(ScalarNode x)` | Computes the square `x²`. / 2 乗 `x²` を計算します。 |
| `static VectorNode<T> Subtract<T>(VectorNode<T> x, VectorNode<T> y)` | Subtracts two vector nodes elementwise, `x - y`. / 2 つのベクトルノードの要素ごとの差 `x - y` を計算します。 |
| `static MatrixNode<T> Subtract<T>(MatrixNode<T> x, MatrixNode<T> y)` | Subtracts two matrix nodes elementwise, `x - y`. / 2 つの行列ノードの要素ごとの差 `x - y` を計算します。 |
| `static VectorNode<T> Subtract<T>(VectorNode<T> x, ScalarNode scalar)` | Subtracts a scalar from each element of a vector, `x - s`. / ベクトルの各要素からスカラーを減算した `x - s` を計算します。 |
| `static MatrixNode<T> Subtract<T>(MatrixNode<T> x, ScalarNode scalar)` | Subtracts a scalar from each element of a matrix, `x - s`. / 行列の各要素からスカラーを減算した `x - s` を計算します。 |
| `static ScalarNode Sum<T>(VectorNode<T> x)` | Computes the sum of all elements, `Σ x[i]`. / 全要素の総和 `Σ x[i]` を計算します。 |
| `static MatrixNode<T> SumMatrices<T>(TensorNode<T> x)` | Records the sum over the matrix index axis of a tensor. / テンソルの行列インデックス軸に沿う総和を記録します。 |
| `static ScalarNode SumOfSquares<T>(VectorNode<T> x)` | Computes the sum of squares `Σ x[i]²`. / 2 乗和 `Σ x[i]²` を計算します。 |
| `static ScalarNode Tan(ScalarNode x)` | Computes the tangent `tan(x)`. / 正接 `tan(x)` を計算します。 |
| `static VectorNode<T> Tanh<T>(VectorNode<T> x)` | Computes the hyperbolic tangent `tanh(x)` elementwise. / 各要素の双曲線正接 `tanh(x)` を計算します。 |
| `static MatrixNode<T> Tanh<T>(MatrixNode<T> x)` | Computes the hyperbolic tangent `tanh(x)` elementwise. / 各要素の双曲線正接 `tanh(x)` を計算します。 |
| `static ScalarNode Tanh(ScalarNode x)` | Computes the hyperbolic tangent `tanh(x)`. / 双曲線正接 `tanh(x)` を計算します。 |
| `static VectorNode<T> Tanh<T>(VectorNode<T> x, bool approximate)` | Computes the hyperbolic tangent `tanh(x)` elementwise, optionally using an approximate `float` kernel. / 各要素の双曲線正接 `tanh(x)` を計算します（`float` では近似カーネルを選択可能）。 |
| `static MatrixNode<T> Tanh<T>(MatrixNode<T> x, bool approximate)` | Computes the hyperbolic tangent `tanh(x)` elementwise, optionally using an approximate `float` kernel. / 各要素の双曲線正接 `tanh(x)` を計算します（`float` では近似カーネルを選択可能）。 |
| `static ScalarNode Trace<T>(MatrixNode<T> x)` | Computes the sum of the main diagonal of a matrix node. / 行列ノードの主対角成分の和を計算します。 |

### `static class XMathSpecialExtensions`

Provides `Special` extension methods for differentiable scalar nodes. / `ScalarNode` に適用できる特殊関数を `Special` の拡張メソッドとして提供します。

- 呼び出し形式: `using Tuna.Differentiation;` のうえで `XMath.Special.<Method>(...)` として呼び出します（`using static` では解決されません）。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static ScalarNode Digamma(ScalarNode x)` | Computes the digamma function `ψ(x)`. / ディガンマ関数 `ψ(x)` を計算します。 |
| `static ScalarNode Erf(ScalarNode x)` | Computes the error function `erf(x)`. / 誤差関数 `erf(x)` を計算します。 |
| `static ScalarNode Erfc(ScalarNode x)` | Computes the complementary error function `erfc(x) = 1 - erf(x)`. / 相補誤差関数 `erfc(x) = 1 - erf(x)` を計算します。 |
| `static ScalarNode LogBeta(ScalarNode a, ScalarNode b)` | Computes the natural logarithm of the absolute value of the beta function, `ln \|B(a, b)\|`. / ベータ関数の絶対値の自然対数 `ln \|B(a, b)\|` を計算します。 |
| `static ScalarNode LogGamma(ScalarNode x)` | Computes the natural logarithm of the absolute value of the gamma function, `ln \|Γ(x)\|`. / ガンマ関数の絶対値の自然対数 `ln \|Γ(x)\|` を計算します。 |
| `static ScalarNode NormalCdf(ScalarNode x)` | Computes the standard normal cumulative distribution function `Φ(x)`. / 標準正規分布の累積分布関数 `Φ(x)` を計算します。 |
| `static ScalarNode NormalLogCdf(ScalarNode x)` | Computes the natural logarithm of the standard normal cumulative distribution function, `ln Φ(x)`. / 標準正規分布の累積分布関数の自然対数 `ln Φ(x)` を計算します。 |
| `static ScalarNode NormalLogPdf(ScalarNode x)` | Computes the standard normal log-density `ln φ(x)`. / 標準正規分布の対数確率密度 `ln φ(x)` を計算します。 |
| `static ScalarNode NormalLogSf(ScalarNode x)` | Computes the natural logarithm of the standard normal survival function, `ln Q(x)`. / 標準正規分布の生存関数の自然対数 `ln Q(x)` を計算します。 |
| `static ScalarNode NormalPdf(ScalarNode x)` | Computes the standard normal probability density function `φ(x) = e^(-x²/2)/√(2π)`. / 標準正規分布の確率密度関数 `φ(x) = e^(-x²/2)/√(2π)` を計算します。 |
| `static ScalarNode NormalSf(ScalarNode x)` | Computes the standard normal survival function `Q(x) = 1 - Φ(x)`. / 標準正規分布の生存関数 `Q(x) = 1 - Φ(x)` を計算します。 |

