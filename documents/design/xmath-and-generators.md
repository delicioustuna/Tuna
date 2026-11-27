# XMath の API 設計と Source Generator

## 1. 3 つの呼び出し形式とビット一致

`XMath` の配列・行列演算は、以下の 3 形式で一貫して提供します。

1. **結果を新規割り当てするオーバーロード** `F(x, ...)`
2. **`destination` を末尾引数に取るオーバーロード** `F(x, ..., destination)`（`System.Numerics.Tensors` と同様に `destination` を末尾に置き、`destination` 自身を返す）
3. **インプレース更新** `XMath.InPlace.F(x, ...)`

新規割り当てオーバーロードは、領域を確保したうえで `destination` オーバーロードを呼び出すだけにしています。これにより、両者の計算結果が**ビット単位で一致**することを保証しています。

`InPlace` を入れ子の静的クラス `XMath.InPlace` に分離している理由は、`using static Tuna.XMath;` を宣言しても取り込まれるのが型名 `InPlace` だけになるようにするためです。これにより、非破壊の呼び出しとインプレース更新を誤って取り違えるリスクを防ぎます。

## 2. 統一入口と入れ子サブドメイン・拡張コンテナの命名方針

数値演算の入口は `XMath` に一本化し、役割や規模に応じて次の入れ子静的クラスで整理しています。

- **`XMath`**: 初等関数およびベクトル・行列・テンソル向けの数値演算
- **`XMath.InPlace`**: 入力バッファをインプレースで上書きする演算
- **`XMath.Operators`**: `[ElementwiseOperation]` / `[PairwiseOperation]` 等で提供する構造体演算子
- **`XMath.Special`**: 200 種を超えるスカラー特殊関数（配列向けオーバーロードを持つ `XMath` 直下と分離し、IntelliSense の見通しと利用者の期待を明確に保つためネスト配置しています）

各拡張パッケージ（`Tuna.SignalProcessing`・`Tuna.Differentiation`・`Tuna.NeuralNetwork`）から `XMath` / `XMath.InPlace` / `XMath.Special` へ演算を追加する際は、C# 14 の静的拡張メンバー（`extension(XMath)` 等）を用い、コンテナクラスの命名と公開方針を次のルールで統一しています。

- **基本名**: 名前空間ごとに `XMathExtensions`（`extension(XMath)`）、`XMathInPlaceExtensions`（`extension(XMath.InPlace)`）、`XMathSpecialExtensions`（`extension(XMath.Special)`）とします。
- **IntelliSense からの隠蔽**: すべての拡張コンテナクラスに `[EditorBrowsable(EditorBrowsableState.Never)]` を付与し、利用者が常に `XMath.*` / `XMath.InPlace.*` / `XMath.Special.*` から統一的に探索・呼び出しできるようにしています（必要な場合は `using static` での取り込みも可能です）。
- **技術的理由による分離**: 同一名前空間内でクラス分離が必要な場合（Source Generator におけるオーバーロード衝突の回避や、推論専用利用時の不要なアセンブリ読み込みの防止）のみ、役割を明示した別コンテナ（`XMathInPlaceExtensions`, `XMathDifferentiationExtensions`）へ分離します。

## 3. コンテナとビューの型推論（`Tuna.Generators`）

`XMath` の基本実装は、入力を読み取り専用ビュー（`ReadOnlyVectorSpan<T>` / `ReadOnlyMatrixSpan<T>` / `ReadOnlyTensorSpan<T>`）、出力先を書き込み可能ビュー（`VectorSpan<T>` 等）で受け取ります。

しかし、C# のジェネリック型推論はユーザー定義の暗黙変換（`Vector<T>` → `ReadOnlyVectorSpan<T>` など）を考慮しないため、基本形だけでは `XMath.Exp(vector)` の呼び出しで型引数 `T` を推論できません。

これを解決するため、手書きするのは各関数の基本形だけとし、ビルド時 Source Generator（`src/Tuna.Generators`）で以下を自動生成しています。

- **`[InferenceOverloads]`**: コンテナ（`Vector<T>`, `Matrix<T>`, `Tensor<T>`）や書き込み可能ビューを直接受け取り、型推論を成立させるオーバーロード群を生成する。コンテナとビューが混在した呼び出しでも、いずれかの引数から型引数が決まれば受け付ける。
- **`[FixedElementTypeOverloads]`**: 最初の型引数を特定の要素型に固定したオーバーロードを生成する。
- **`[ElementwiseOperation]` / `[PairwiseOperation]`**: スカラー関数または列ペア関数から、ベクトル・行列の各種オーバーロード（新規割り当て、`destination`、`InPlace`）と、構造体演算子（`IUnaryOperator<T>` 等）を生成する。この Generator は NuGet パッケージのアナライザーとして同梱されるため、利用者コードでも同じ属性が動作する。

## 4. エイリアシング規則（入力と出力先の重なり）

Tuna は、出力先と入力の領域が重なっている場合に**暗黙の一時バッファへ退避しません**。

- **要素別演算**: 出力先が入力と完全に同一の要素を同一の添字・ストライドで参照する場合のみ、インプレース更新として許容します。部分的な重なりは `ArgumentException` をスローします。
- **積・転置・縮約**: 出力先と入力が少しでも重なる場合は、すべて `ArgumentException` をスローします（3 成分をすべて読み込んでから書き込む `Cross` は除く）。

## 5. ユーザー定義演算子インターフェース

デリゲート（`Func<T, T>`）経由の呼び出しは間接呼び出しとクロージャ割り当てを伴うため、ゼロオーバーヘッドの拡張点として構造体ベースの演算子インターフェースを用意しています。

- **状態なし**: `IUnaryOperator<T>`, `IBinaryOperator<T>`, `ITernaryOperator<T>`, `IPairwiseOperator<T>`, `IVectorIndexOperator<T>`, `IIndexOperator<T>`, `ITensorIndexOperator<T>`（静的抽象メンバー `static T Invoke(...)` を持ち、`XMath.Map<MyOp>(x)` のように型引数で渡す）
- **状態あり**: `IStatefulUnaryOperator<T>` ほか（インスタンスメソッド `Invoke(...)` を持ち、値で渡す）

## 6. 初等関数の数値安定性と近似オプション（`approximate`）

- `Log1p` などの数値安定関数では、.NET のジェネリック実装（`T.LogP1` 等）が 0 付近で桁落ちするケースを避けるため、`double` 精度で桁落ちしない式を通してから `T` に丸めています。
- `Exp`, `Tanh`, `Logistic`, `Silu`, `Softplus`, `Gelu` には `approximate` オプションがあります。
  - `float` で `approximate: true` の場合、`double` への拡幅やテーブル参照を行わない高速近似式（`DenseElementwise.Approximate.cs`、誤差は `float` の数 ULP）を使用します。スカラー・ベクトル・行列・インプレースのすべてで同じ式を使うため、配置・長さ・命令セットに依存せずビット一致します。
  - `double` / `Half` では精度を落としても大幅な高速化が見込めないため、`Gelu`（`tanh` 近似式を選択）を除き `approximate: false` と同じ値を返します。
