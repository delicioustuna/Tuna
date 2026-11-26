# 疑似乱数と準乱数

コアパッケージ `Tuna` は、再現性と並列計算を重視した疑似乱数生成器（`Tuna.RandomSource`）と、準モンテカルロ法向けの低食い違い量点列（`Tuna.QuasiRandom`）を提供します。
いずれも統計・数値シミュレーション向けであり、**暗号用途には使用しないでください**。
正規分布・ガンマ分布などの確率分布からのサンプリングは拡張パッケージ `Tuna.Statistics` にあります。

## 疑似乱数（`Tuna.RandomSource`）

すべての乱数生成器は `IRandomSource` を実装します。同じシード・`streamId`・API 呼び出し列であれば、CPU のエンディアンや `.NET` の `System.Random` の実装によらず同じ値列を再現します。

| クラス | アルゴリズム | 特徴 |
| --- | --- | --- |
| `Pcg64DxsmRandom` | PCG64 DXSM (128-bit state, 64-bit output) | 汎用の既定推奨。周期 $2^{128}$、`Advance(steps)` で $O(\log n)$ のジャンプが可能 |
| `Philox4x32Random` | Philox 4x32-10（カウンタベース） | `(seed, streamId, blockIndex)` から直接生成でき、`Seek` によるランダムアクセスが $O(1)$ |

### 基本的な使い方と再現性

```csharp
using Tuna.RandomSource;

// シードを明示すると同一メジャーバージョン内で完全に再現可能
var rng = new Pcg64DxsmRandom(seed: 20261021);

double u = rng.NextDouble();         // [0, 1)
double uOpen = rng.NextDoubleOpen(); // (0, 1) — 対数変換や逆 CDF に安全
int dice = rng.NextInt32(1, 7);      // 1..6（偏りのない Lemire 法）

var buf = new double[100];
rng.Fill(buf);                       // NextDouble() を順に呼ぶのと同一の値列
```

- `Fill(Span<uint/ulong/double/float>)` は対応する単発 API の繰り返しと一致し、バッファの分割方法に依存しません。`Fill(Span<byte>)` は 64-bit 出力をリトルエンディアンで格納し末尾の余剰バイトを破棄するため、8 の倍数でない分割では消費量が変わります。
- `Pcg64DxsmRandom(seed, streamId)` は独自のシード初期化形式を用いるため、同じ数値シードでも NumPy の `PCG64DXSM` とは異なる列になります（生の状態・増分を指定する `Pcg64DxsmState` からの復元では NumPy 等と照合できます）。
- `Philox4x32Random` の 1 ストリームは $2^{64}$ ブロック（$2^{66}$ 語）で、使い切ると循環せず `InvalidOperationException` をスローします。

### 状態の保存と復元・ジャンプ

```csharp
Pcg64DxsmState saved = rng.GetState();
ulong expected = rng.NextUInt64();

var restored = new Pcg64DxsmRandom(saved);
// restored.NextUInt64() == expected

// 64-bit 出力を n 回分スキップ
rng.Advance(1_000_000);
```

### 並列計算でスレッド数に依存しない再現性を保つ

1 つの乱数生成器はスレッドセーフですが、複数スレッドで共有するとスケジューリング順によって各スレッドへ割り当てられる値が変わります。並列処理では、同じ `seed` に対してタスク・チェーンなどの論理 ID ごとに異なる `streamId` を割り当ててください。

```csharp
const ulong seed = 42;
Parallel.For(0, 8, taskId =>
{
    var localRng = new Pcg64DxsmRandom(seed, streamId: (ulong)taskId);
    // localRng を使った計算は実行のたびにビット単位で一致する
});
```

`Philox4x32Random` では `Philox4x32Random.GenerateBlock(seed, streamId, blockIndex)` でインスタンスを作らずに任意ブロックの 4 語を直接計算することもできます。

### シャッフル・順列・制約付き整数分割

`IRandomSource` の拡張メソッド（`RandomSourceExtensions`）と `BoundedCompositionSampler` で、有限集合から一様にサンプリングできます。

```csharp
int[] items = [10, 20, 30, 40];
rng.Shuffle(items);                      // Fisher–Yates 法によるインプレースシャッフル

var perm = new int[4];
rng.Permutation(perm);                   // 0..3 のランダムな順列

// l_i <= x_i <= u_i, sum(x_i) == 10 を満たす整数組をほぼ等確率で生成
var sampler = new BoundedCompositionSampler(lower: [0, 0, 0], upper: [5, 5, 5], sum: 10);
var sample = new long[3];
sampler.Sample(rng, sample);
```

## 準乱数（`Tuna.QuasiRandom`）

`SobolSequence`（`IPointSequence` 実装）は、1〜1,111 次元・最大 $2^{64}$ 点の Sobol' 点列を生成し、$(0, 1)^d$ を一様乱数よりも均等に埋めます。構築後は不変でスレッドセーフであり、点の範囲を分割して並列生成しても一括生成とビット単位で一致します。

```csharp
using Tuna.LinearAlgebra;
using Tuna.QuasiRandom;
using Tuna.RandomSource;

// 2 次元の Sobol' 点列（1 列 = 1 点の 2 × 8 行列）
var sobol = new SobolSequence(dimension: 2);
Matrix<double> points = sobol.GetPoints(start: 0, count: 8);

// 1 点だけ取得する場合
var pt = new double[2];
sobol.GetPoint(index: 5, pt);
```

### 使用上のポイント

- **点数は $2^m$ 個、先頭から捨てずに使う**: Sobol' 列の均等性（デジタルネット性）は `start: 0`, `count: 1 << m` で成り立ちます。先頭の点（非スクランブル時の原点）を捨てると均等性が崩れるため、原点を避けたい場合は乱択化を使ってください。
- **重要な変数を前の次元に割り当てる**: 前の次元ほど 2 次元射影の均等性が高いため、結果への寄与が大きい変数を小さい次元番号へ割り当てると収束が速くなります。
- **0 と 1 を返さない**: 各座標は $(k + \tfrac{1}{2}) \cdot 2^{-52} \in [2^{-53}, 1 - 2^{-53}]$ の形にシフトされるため、`XMath.Special.InverseNormalCdf` などの逆関数へそのまま渡しても $\pm\infty$ になりません（棄却法ではなく逆関数法と組み合わせてください）。
- **誤差推定にはスクランブル（乱択化）を使う**: コンストラクタに `IRandomSource` を渡すとスクランブルされた Sobol' 列になります。独立な `streamId` で複数本生成して平均と標準誤差を取ることで、不偏推定と誤差評価ができます。

```csharp
var scrambled = new SobolSequence(dimension: 4, new Pcg64DxsmRandom(seed: 5, streamId: 0));
Matrix<double> P = scrambled.GetPoints(0, 1 << 12);
```
