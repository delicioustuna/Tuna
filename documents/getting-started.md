# はじめに（コンテナとビューの基本）

Tuna の線形代数データ構造は、ヒープ上にデータを保持する**コンテナ型**（クラス）と、その一部や転置をコピーせずに参照する**ビュー型**（`readonly ref struct`）の 2 層に分かれています。

```csharp
using Tuna;
using Tuna.LinearAlgebra;
// 複素数（FFT や一般固有値分解）を使う場合は using System.Numerics; ではなくエイリアスを使うと
// System.Numerics.Vector<T> との名前衝突（CS0104）を避けられます。
// using Complex = System.Numerics.Complex;
```

> **補足: `using static Tuna.XMath;` の使い分け**
> コア `Tuna` の数式だけを短く書きたい場合は `using static Tuna.XMath;` が便利です。ただし、拡張パッケージ（`Tuna.Differentiation`・`Tuna.SignalProcessing`・`Tuna.NeuralNetwork`）が `XMath` に追加するメンバーは C# の仕様上 `using static` でインポートされないため、拡張パッケージを併用するときは `XMath.Sum(...)`・`XMath.Linear(...)` のように `XMath.` を付けて呼び出してください。

## コンテナ型（`Vector<T>`・`Matrix<T>`・`Tensor<T>`）

要素型 `T` は `unmanaged, INumberBase<T>` を満たす数値型（`double`, `float`, `Half`, `Complex`, `int`, `decimal` など）を指定できます。行列分解など一部のアルゴリズムは `double` 専用です。

### ベクトルと行列の作成

```csharp
// ベクトル
var x = new Vector<double>([1.0, 2.0, 3.0]);  // 値をコピーして作成
var zeros = new Vector<double>(4);            // 長さ 4 の零ベクトル
var fromArray = new[] { 0.5, 1.5 }.ToVector();

// 行列（内部のメモリ配置は列優先 / column-major）
var A = new Matrix<double>(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
var B = Matrix<double>.FromRowMajor(2, 3, [1, 2, 3, 4, 5, 6]);
var C = Matrix<double>.FromColumnMajor(2, 3, [1, 4, 2, 5, 3, 6]);

var I = Matrix<double>.Identity(3);
var D = Matrix<double>.FromDiagonal([2.0, 3.0]);
```

> **メモリ配置とデータの並び**
> - 行列の要素 `(i, j)` は列優先で格納されており、`A.AsSpan()[i + j * A.Rows]` に対応します。
> - Tuna 全体を通して、データ点の集合を行列で表すときは **1 列 = 1 点**（次元数 × 点数）を基本としています。

### 3 階テンソル（行列のバッチ）

`Tensor<T>` は形状 `[Rows, Columns, Count]` の 3 階配列で、同じ形状の行列を第 3 軸方向に並べた構造です。

```csharp
var T = Tensor<double>.FromMatrices(A0, A1, A2);
MatrixSpan<double> second = T.GetMatrix(1); // 第 1 行列へのビュー（コピーなし）
MatrixSpan<double> diags = T.Diagonals();   // 第 k 列が第 k 行列の対角になるビュー
```

## ビュー型（ゼロコピー参照）

ビューは `VectorSpan<T>` / `ReadOnlyVectorSpan<T>`、`MatrixSpan<T>` / `ReadOnlyMatrixSpan<T>`、`TensorSpan<T>` / `ReadOnlyTensorSpan<T>` として提供されます。

```csharp
var A = Matrix<double>.FromRowMajor(3, 4,
[
    1,  2,  3,  4,
    5,  6,  7,  8,
    9, 10, 11, 12,
]);

MatrixSpan<double> transposed = A.T;       // 転置ビュー（4 × 3）
MatrixSpan<double> block = A[1..3, 1..3];  // 部分行列ビュー（2 × 2）
VectorSpan<double> row = A.Row(0);         // 第 0 行（ストライド付き）
VectorSpan<double> col = A.Column(2);      // 第 2 列（連続領域）
VectorSpan<double> diag = A.Diagonal();    // 対角成分

// 書き込み可能なビューを変更すると元の行列も書き換わる
diag.Fill(0);
block[0, 0] = -1; // A[1, 1] = -1 と同じ

// コピーとして独立保持したいときは ToMatrix() / ToVector() を使う
Matrix<double> kept = A[0..2, ..].ToMatrix();
```

ビューは `ref struct` のためフィールドや非同期メソッドをまたいで保持できませんが、そのまま `XMath` の引数や演算子に渡せます。

## 演算子と複合代入

標準の算術演算子は新しいコンテナを返します。ビューをそのまま式に混ぜて書くことができます。

```csharp
Matrix<double> C = A.T * B + 2.0 * A;
Vector<double> y = A * x;
```

- `*` は**行列積**（または行列とベクトルの積、スカラー倍）です。`Tensor<T>` 同士の `*` も各スライスの行列積です。
- 要素ごとの積は `XMath.Hadamard(A, B)` を使います（`Vector * Vector` の `*` 演算子は定義していません）。
- `+=`・`-=`・`*=` は**インプレース**（元のインスタンスを直接書き換え）で動作します。

メモリ割り当てを抑えたい場合の詳しい使い方は、[付録: メモリ割り当ての回避とパフォーマンス](appendix/performance-and-memory.md) を参照してください。
