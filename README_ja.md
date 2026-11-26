# Tuna

[English](README.md)

**T**iny **U**tilities for **N**umerical **A**nalysis — .NET 10 向けの、純 C#・SIMD 最適化数値計算ライブラリです。

Tuna はネイティブ依存なしで高速な線形代数・特殊関数・決定的な疑似乱数 / 準乱数を提供し、目的に応じて統計・信号処理・最適化・数値積分・自動微分・MCMC・ニューラルネットワークの拡張パッケージを組み合わせて利用できます。

## 主な特徴

- **純 C#・ネイティブ依存なし**: SIMD 最適化したカーネルを純 C# で実装しており、ネイティブの BLAS / LAPACK バイナリを配置せずに .NET 10 環境でそのまま動作します。
- **ゼロコピービュー（`readonly ref struct`）**: 転置（`A.T`）、スライス（`A[1..3, ..]`）、行・列・対角ビュー、および明示的なブロードキャスト（`v.Broadcast(cols)`）をメモリアロケーションなしで扱えます。
- **アロケーションを抑えられる API 設計**: `XMath` の要素別演算や線形代数演算では、結果を新規確保するオーバーロード、`destination` を受け取るオーバーロード、`XMath.InPlace` によるインプレース版を一貫して提供しています。行列分解も内部バッファを再利用した再計算（`Recompute`）に対応します。
- **モジュール化されたパッケージ構成**: コアパッケージ `Tuna` を基盤とし、用途に応じた拡張パッケージ（`Tuna.*`）だけを選択して利用できます。

## パッケージ構成

| パッケージ | 主な機能 |
| --- | --- |
| [`Tuna`](src/Tuna) | コア: `Vector<T>`・`Matrix<T>`・`Tensor<T>`、`XMath`、BLAS・行列分解（LU / Cholesky / QR / 固有値 / SVD）、特殊関数、疑似乱数・準乱数 |
| [`Tuna.Statistics`](src/Tuna.Statistics) | 連続・離散・多変量の確率分布（PDF / CDF / 分位点 / サンプリング） |
| [`Tuna.SignalProcessing`](src/Tuna.SignalProcessing) | 1D/2D FFT、実数 FFT、DCT / DST、畳み込み、相関、離散 / 連続ウェーブレット変換 |
| [`Tuna.Optimization`](src/Tuna.Optimization) | L-BFGS / L-BFGS-B、SGD / Adam / AdamW、近接勾配法（FISTA / ADMM）、大域最適化、遺伝的アルゴリズム |
| [`Tuna.Integration`](src/Tuna.Integration) | 1 次元適応数値積分、多次元適応積分、準モンテカルロ積分 |
| [`Tuna.Differentiation`](src/Tuna.Differentiation) | テープベースの逆モード自動微分（スカラー・ベクトル・行列・テンソル・行列分解） |
| [`Tuna.Probabilistic`](src/Tuna.Probabilistic) | MCMC サンプラー（HMC / NUTS / スライス / アンサンブル / レプリカ交換）と収束診断（$\hat{R}$, ESS, MCSE） |
| [`Tuna.NeuralNetwork`](src/Tuna.NeuralNetwork) | 2D 畳み込み、Mamba ブロック、正規化層、SafeTensors 読み取り |

## クイックスタート

### 線形代数とゼロコピービュー

```csharp
using Tuna;
using Tuna.LinearAlgebra;

var A = Matrix<double>.FromRowMajor(3, 3,
[
    4, 1, 0,
    1, 3, 1,
    0, 1, 2,
]);
var b = new Vector<double>([1.0, 2.0, 3.0]);

// 連立一次方程式 A * x = b を解く
Vector<double> x = XMath.Solve(A, b);

// Cholesky 分解
var chol = A.Cholesky();
double logDet = chol.LogDeterminant;

// ゼロコピーの転置・スライス・ブロードキャスト
Matrix<double> C = A.T * A[0..3, 0..2];
Vector<double> rowMeans = XMath.RowMeans(A);
XMath.Subtract(A, rowMeans.Broadcast(A.Columns), A); // 列ごとに中心化（インプレース）
```

### アロケーション制御（`XMath`）

`XMath` の多くの演算では、用途に応じて 3 つの呼び出し形式を使い分けられます。

```csharp
var v = new Vector<double>([0.0, 1.0, 2.0]);
var buffer = new Vector<double>(v.Length);

Vector<double> y = XMath.Exp(v);     // 1. 結果を新規確保
XMath.Exp(v, buffer);                // 2. 既存の destination バッファへ書き込み
XMath.InPlace.Exp(v);                // 3. インプレース更新
```

### 特殊関数と決定的な疑似乱数

```csharp
using Tuna;
using Tuna.RandomSource;

double logGamma = XMath.Special.LogGamma(200.0);
double tail = XMath.Special.Erfc(6.0);

var rng = new Pcg64DxsmRandom(seed: 42);
double u = rng.NextDouble();
```

### 拡張パッケージの例（逆モード自動微分）

```csharp
using Tuna;
using Tuna.Differentiation;

var tape = new Tape();
var x = tape.Leaf(2.0);
var y = XMath.Sin(x) + x * x;
tape.Backward(y);
double dydx = tape.GetGradient(x); // cos(2) + 4
```

## 設計上の特徴

- **列優先（column-major）と 1 列 = 1 点**: `Matrix<T>` は列優先で要素を格納します。データ点の集合は「次元数 × 点数（1 列 = 1 点）」を基本とします。
- **暗黙のブロードキャストなし**: 意図しない形状変換を防ぐため、暗黙のブロードキャストは行いません。`v.Broadcast(cols)` / `v.BroadcastRows(rows)` で明示的にゼロコピービューを作成します。
- **厳格なエイリアシングルール**: 要素別演算は完全一致のみインプレース更新を許可します（部分重複は `ArgumentException`）。行列積・転置・縮約では入力と出力先の重複を一切許可しません。

## 動作環境

- .NET 10 (`net10.0`)

## ドキュメント

詳しい使い方・設計文書・機械生成された API リファレンスは [`documents/`](documents/README.md) にまとめています。

- [はじめに（コンテナとビューの基本）](documents/getting-started.md)
- [`XMath` の使い方](documents/xmath.md)
- [線形代数と行列分解](documents/linear-algebra.md)
- [`XMath.Special` の使い方](documents/special-functions.md)
- [疑似乱数と準乱数](documents/random.md)
- [拡張パッケージの概要](documents/extensions.md)
- [付録: メモリ割り当ての回避とパフォーマンス](documents/appendix/performance-and-memory.md)
- [設計文書](documents/design/README.md)
- [API リファレンス一覧](documents/api/README.md)

## ライセンス

このライブラリは [MIT License](LICENSE.txt) のもとで公開されています。
