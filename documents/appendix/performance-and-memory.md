# 付録: メモリ割り当ての回避とパフォーマンス

本文のガイドでは読みやすさを優先して、結果を新規に割り当てる呼び出し（`Exp(x)` や `A * B` など）を中心に紹介しました。
この付録では、反復計算・リアルタイム処理・大規模データ処理で GC 負荷とメモリコピーを最小限に抑えるための実践テクニックをまとめます。

## 1. `destination` オーバーロードと `XMath.InPlace` を使う

ホットループ内で `XMath` の新規割り当てオーバーロードや `+` / `*` 演算子を繰り返すと、イテレーションごとにヒープ割り当てが発生します。事前にバッファを確保して末尾引数 `destination` として渡すか、`XMath.InPlace` / 複合代入を使ってください。

```csharp
var buffer = new Vector<double>(n);
var temp = new Matrix<double>(m, n);

for (int iter = 0; iter < 10_000; iter++)
{
    XMath.Multiply(A, x, buffer);      // 割り当てなし
    XMath.InPlace.Add(buffer, bias);   // インプレース加算
    XMath.InPlace.Relu(buffer);        // インプレース活性化
}
```

- 新規割り当てオーバーロードは内部で `destination` オーバーロードを呼んでいるだけのため、両者の計算結果はビット単位で一致します。
- コンテナの加減算やスカラー乗除は `x += y`, `x -= y`, `x *= alpha`, `x /= alpha` でインプレース実行できます。

## 2. ビューによるゼロコピー（転置・部分行列・ブロードキャスト）

- `A.T`（転置）、`A[r0..r1, c0..c1]`（部分行列）、`A.Column(j)` / `A.Row(i)` / `A.Diagonal()`、`v.Broadcast(cols)`、`v.BroadcastRows(rows)` はすべて `readonly ref struct` のビューを返し、**ヒープ割り当て・データコピーを行いません**。
- `XMath.Multiply(A.T, B, C)` のように転置ビューを渡すと、内部の GEMM カーネルが転置フラグとして直接処理するため、転置行列の実体化は不要です。
- **列優先の局所性**: `Matrix<T>` は列優先（column-major）です。`A.Column(j)` はメモリ上で連続（ストライド 1）になり SIMD 最適化が最大限に効きます。一方 `A.Row(i)` はストライド `Rows` になります。頻繁に走査する軸を列側に配置（1 列 = 1 点）するとキャッシュ効率が上がります。

## 3. 行列分解オブジェクトの再利用（`Recompute`）

同じサイズ（またはそれ以下）の行列に対して LU・Cholesky・QR・固有値・SVD を繰り返す場合は、毎回 `A.Cholesky()` などを新規生成するのではなく、一度作ったインスタンスの `Recompute` / `TryRecompute` を呼び出します。

```csharp
var chol = K.Cholesky();
var rhs = new Vector<double>(n);
var sol = new Vector<double>(n);

for (int step = 0; step < 1000; step++)
{
    UpdateKernelMatrix(K, step);
    if (!chol.TryRecompute(K))
    {
        break;
    }

    chol.Solve(rhs, sol); // 解の出力先も再利用
}
```

分解オブジェクトと `Solve` の `destination` オーバーロードを組み合わせることで、反復中のヒープ割り当てを実質ゼロにできます。

## 4. ユーザー定義演算のゼロ抽象化オーバーヘッド

独自の要素別関数を適用するとき、`Func<double, double>` デリゲートではなく **構造体演算子**（`IUnaryOperator<T>` / `IStatefulUnaryOperator<T>`）または **`[ElementwiseOperation]` 属性**を使ってください。

- `struct` をジェネリック型引数 `TOperator` として渡すことで、JIT が `Invoke` 呼び出しをインライン展開し、仮想呼び出しやクロージャのヒープ割り当てを回避できます。
- `[ElementwiseOperation]` を付けたメソッドには演算子構造体も自動生成されるため、手書きのボイラープレートなしで同じ性能が得られます。

## 5. エイリアシング規則に注意する

Tuna は暗黙の一時バッファ確保を行いません。`destination` が入力と重なる場合のルールは次のとおりです。

- **要素別演算**: 完全一致（同じ先頭要素・同じストライド・同じサイズ）のみインプレースとして許可。部分的な重なりは `ArgumentException`。
- **行列積・転置・縮約**: いかなる重なりも `ArgumentException`。必ず入力とは別の出力バッファを用意してください。

## 6. 並列度と決定性の制御

- **スレッド数の制御**: `LinearAlgebraSettings.MaxDegreeOfParallelism` で内部の並列化スレッド上限を設定できます。すでに呼び出し側で `Parallel.For` などの外側並列を行っている場合は、`MaxDegreeOfParallelism = 1` に設定してスレッドの過剰競合（oversubscription）を防ぎます。
- **再現性（決定性）**: 要素別演算の SIMD カーネルやペアワイズ加算・並列縮約は、FMA によるハードウェア依存の丸め差を避け、固定ブロック構造で集約するよう設計されているため、同一アーキテクチャ上での再現性が高くなっています。
- **大規模対称行列の固有値計算**: 固有値のみが必要な大規模行列では `LinearAlgebraSettings.TwoStageReduction`（既定 `Auto`）により、キャッシュ効率の高い二段階三重対角化が自動的に選ばれます。
