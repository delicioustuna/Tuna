# 線形代数と行列分解

Tuna の線形代数機能は、`XMath` のワンショット関数、各種行列分解クラス、および低レベル BLAS ルーチン（`Blas`）で構成されています。
行列分解は現在 `double` を対象としています。

```csharp
using Tuna;
using Tuna.LinearAlgebra;
using static Tuna.XMath;
```

## 1 回だけ解く: `XMath.Solve` と三角求解

同じ係数行列を再利用しない場合は `XMath.Solve` を呼び出します。内部で部分ピボット付き LU 分解を実行します。

```csharp
var A = Matrix<double>.FromRowMajor(3, 3, [2, 1, 1, 1, 3, 2, 1, 0, 0]);
var b = new Vector<double>([4.0, 5.0, 6.0]);

Vector<double> x = Solve(A, b);
Matrix<double> Ainv = Inverse(A); // 解くだけなら Solve の方が高速かつ高精度
```

三角行列の連立一次方程式には `SolveLowerTriangular` / `SolveUpperTriangular` を使用できます。

## 行列分解クラス一覧

同じ係数行列に対して複数回解いたり、行列式・固有値・特異値を取り出したりする場合は分解オブジェクトを作成します。拡張メソッド（`A.Lu()` 等）と静的メソッド（`LuDecomposition.Compute(A)` 等）の両方で呼び出せ、いずれも `Matrix<double>` に加えて部分行列や転置のビュー（`A[0..50, ..].Qr()` や `A.T.Svd()` など）もそのまま渡せます。

| 分解クラス | 拡張メソッド | 対象・主な用途 |
| --- | --- | --- |
| `LuDecomposition` | `A.Lu()` | 一般正方行列の連立一次方程式、行列式（`LogAbsDeterminant`）、反復改良（`Refine`） |
| `CholeskyDecomposition` | `A.Cholesky()` | 対称正定値行列（下三角部分のみ参照）、ガウス過程、対数行列式（`LogDeterminant`）、`TryCompute` |
| `QrDecomposition` | `A.Qr()` | 縦長・正方行列の最小二乗法（`Solve`：`Vector<double>`・`Matrix<double>`・ビュー・`ReadOnlySpan<double>` 対応）、直交基底 |
| `SymmetricEigenDecomposition` | `A.SymmetricEigen()` | 対称行列の固有値（昇順）・直交固有ベクトル（下三角部分のみ参照） |
| `EigenDecomposition` | `A.Eigen()` | 一般正方行列の複素固有値・固有ベクトル（`Complex` を使う際は `using Complex = System.Numerics.Complex;` を推奨） |
| `SingularValueDecomposition` | `A.Svd()` | 一般行列の特異値（降順）・数値ランク・条件数・低ランク近似 |
| `BatchCholeskyDecomposition` | `BatchCholeskyDecomposition.Compute(T)` | 3 階 `Tensor<double>` 内の複数行列に対する一括 Cholesky 分解 |

### 使用例: Cholesky 分解と条件数

```csharp
// 正定値でない可能性がある場合は TryCompute を使う
if (CholeskyDecomposition.TryCompute(K, out var chol))
{
    Vector<double> alpha = chol.Solve(y);
    double logDet = chol.LogDeterminant;
    double rcond = chol.EstimateReciprocalConditionNumber();
}
```

### 使用例: 固有値分解・SVD でベクトルを省略する

固有値や特異値だけが必要な場合は `computeVectors: false` を渡すと計算量を抑えられます。

```csharp
SymmetricEigenDecomposition eig = A.SymmetricEigen(computeVectors: false);
double[] eigenvalues = eig.Values; // 昇順

SingularValueDecomposition svd = M.Svd(computeVectors: false);
double[] singularValues = svd.Values; // 降順
```

## 分解オブジェクトの再利用（`Recompute`）

最適化ループや MCMC などで同じサイズ（またはそれ以下）の行列を繰り返し分解する場合、`Recompute` を使うと内部バッファを再利用してヒープ割り当てを抑えられます。

```csharp
var chol = K.Cholesky();
// 次のイテレーションで K の中身を更新したあと、既存オブジェクト上で再分解
bool ok = chol.TryRecompute(K);
```

`LuDecomposition`, `CholeskyDecomposition`, `QrDecomposition`, `SymmetricEigenDecomposition`, `SingularValueDecomposition` の各クラスが `Recompute`（Cholesky は `TryRecompute` も）を提供しています。

## バッチ演算（`Tensor<T>`）

3 階テンソル `Tensor<T>` を使うと、多数の小〜中規模行列を一括処理できます。

```csharp
// C[k] = A[k] * B[k]
Tensor<double> C = A * B;

// すべてのスライスに同じ行列 M を掛ける（コピーなし）
Tensor<double> rotated = Multiply(M.Broadcast(B.Matrices.Count), B);

// 行列番号の軸に沿って総和を計算
Matrix<double> total = XMath.SumMatrices(C);

// バッチ Cholesky 分解（各スライスの結果は単体 Cholesky とビット単位で一致）
var batchChol = BatchCholeskyDecomposition.Compute(KTensor);
Vector<double> logDets = batchChol.LogDeterminants();
Matrix<double> solutions = batchChol.Solve(rhsMatrix); // 第 k 列が第 k 行列の右辺
```

## 並列度とアルゴリズム設定（`LinearAlgebraSettings`）

`LinearAlgebraSettings.MaxDegreeOfParallelism` で線形代数カーネル（BLAS・分解・並列縮約）が使用する最大スレッド数を制御できます（既定は物理コア数ベース）。
また、`LinearAlgebraSettings.TwoStageReduction` で大規模対称行列の二段階三重対角化（固有値のみ計算時）の有効・無効を切り替えられます。
