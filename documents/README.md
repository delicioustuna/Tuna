# Tuna ドキュメント

**Tuna**は、C#で作成された数値計算のためのライブラリです。
コアの機能として、**線形代数**・**特殊関数**・**疑似乱数 / 準乱数**を提供しており、目的に応じて統計・信号処理・最適化・数値積分・自動微分・確率的推論・ニューラルネットワークの拡張パッケージを組み合わせて利用することもできます。

このドキュメントでは、Tuna の構成と使い方をなるべく簡単に把握できるよう、要点を絞って解説しています。一般的な数学の定義は省略しており、Tuna 固有の呼び方や仕様を中心とした解説になっているため、個々の関数の定義や用途については個別の文献を参照してください。

## パッケージ構成

コアパッケージを**Tuna**として、以下の表のとおり目的に応じたサブパッケージを提供しています。

| パッケージ | 主な機能 | 主な名前空間 |
| --- | --- | --- |
| **Tuna** | ベクトル・行列・テンソル、`XMath`、行列分解・BLAS、特殊関数、疑似乱数・準乱数 | `Tuna`, `Tuna.LinearAlgebra`, `Tuna.RandomSource`, `Tuna.QuasiRandom` |
| **Tuna.Statistics** | 連続・離散・多変量の確率分布（PDF / CDF / 分位点 / サンプリング） | `Tuna.Statistics.Distributions` |
| **Tuna.SignalProcessing** | FFT・DCT・畳み込み・相関・離散 / 連続ウェーブレット変換 | `Tuna.SignalProcessing` |
| **Tuna.Optimization** | 勾配法（L-BFGS 等）・確率的勾配法・近接勾配法・大域最適化・遺伝的アルゴリズム | `Tuna.Optimization.*` |
| **Tuna.Integration** | 1 次元数値積分・多次元適応積分・準モンテカルロ積分 | `Tuna.Integration` |
| **Tuna.Differentiation** | 逆モード自動微分（スカラー・ベクトル・行列・分解） | `Tuna.Differentiation` |
| **Tuna.Probabilistic** | MCMC（HMC / NUTS / スライス / レプリカ交換）と収束診断 | `Tuna.Probabilistic.Inference` |
| **Tuna.NeuralNetwork** | 2D 畳み込み・Mamba ブロック・正規化・SafeTensors 読み取り | `Tuna.NeuralNetwork`, `Tuna.NeuralNetwork.IO` |

## 基本仕様

- **列優先・1 列 = 1 点**: `Matrix<T>` は列優先（column-major）で、データ点の集合は「次元数 × 点数（1 列 = 1 点）」を基本とします。
- **暗黙のブロードキャストなし**: ベクトルを行列へ適用するときは `v.Broadcast(cols)` / `v.BroadcastRows(rows)` で明示的にゼロコピービューを作ります。
- **拡張パッケージの `XMath` 拡張は `XMath.` を付けて呼ぶ**: `Tuna.Differentiation`・`Tuna.SignalProcessing`・`Tuna.NeuralNetwork` が追加する `XMath` メンバー（`XMath.Linear`, `XMath.Convolve`, `VectorNode<T>` 版 `XMath.Sum` 等）は C# の拡張ブロック仕様上 `using static Tuna.XMath;` では解決されません（`CS0103` や `CS0411` が発生します）。拡張パッケージを併用する場合は `XMath.` 修飾を基本にしてください。
- **`Complex` 併用時の `Vector<T>` 衝突回避**: `System.Numerics.Complex` を使うファイルで `using System.Numerics;` を書くと `System.Numerics.Vector<T>` と衝突します（`CS0104`）。`using Complex = System.Numerics.Complex;` のように型エイリアスを使うと衝突を回避できます。
- **エイリアシング規則**: 要素別演算は完全一致のみインプレース可（部分重複は `ArgumentException`）。行列積・転置・縮約は入力と出力先の重複を一切許可しません。

## 取扱説明書

### 使い方

1. [はじめに（コンテナとビューの基本）](getting-started.md) — `Vector<T>`・`Matrix<T>`・`Tensor<T>` の作成、ゼロコピービュー、演算子
2. [`XMath` の使い方](xmath.md) — 3 つの呼び出し形式、ビューとブロードキャスト、要素別演算・縮約・数学定数・ユーザー定義演算子
3. [線形代数と行列分解](linear-algebra.md) — `XMath.Solve`、LU / Cholesky / QR / 固有値 / SVD、バッチ演算
4. [`XMath.Special` の使い方](special-functions.md) — 特殊関数の呼び方、対数スケール版・補関数版、定義域外の挙動
5. [疑似乱数と準乱数](random.md) — 決定的なシード管理、並列ストリーム、状態保存、Sobol' 列

### 拡張パッケージ・付録・リファレンス

- [拡張パッケージの概要](extensions.md) — 各拡張パッケージの入口と最小コード例
- [確率分布の自動微分](distribution-differentiation.md) — 正規分布・ベルヌーイ分布の対数確率と観測ベクトルの自動微分、MCMC への接続
- [付録: メモリ割り当ての回避とパフォーマンス](appendix/performance-and-memory.md) — `destination` オーバーロード、`InPlace`、ビュー、分解の再利用、決定性
- [設計文書](design/README.md) — API 設計、Source Generator、SIMD・BLAS・特殊関数の設計上の理由（Why）
- [API リファレンス一覧](api/README.md) — ビルド済みアセンブリと XML ドキュメントから機械生成された公開 API 一覧
