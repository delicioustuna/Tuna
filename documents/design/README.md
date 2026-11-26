# 設計文書（`documents/design/`）

このディレクトリには、ソースコードのコメントに散在していた**複数ファイルにまたがる設計上の理由（Why）**、API 設計の背景、SIMD・BLAS・並列化・数値安定性の方針をまとめています。

ソースコード内の `//` コメントは、その箇所でしか分からない局所的な不変条件・数式出典・エッジケース回避だけに留め、アーキテクチャレベルの意図はこの設計文書で一元管理します。

## 目次

- [XMath の API 設計と Source Generator](xmath-and-generators.md) — 3 形式のオーバーロード設計、入れ子サブドメインと拡張コンテナの命名方針、型推論とオーバーロード生成、エイリアシング規則、近似演算のポリシー
- [線形代数・SIMD・並列化の設計](linear-algebra-and-simd.md) — 決定的な SIMD カーネル、ペアワイズ加算木、JIT 最適化属性の方針、4M 複素 GEMM、二段階三重対角化、owner-computes 列分担、`Recompute`
- [特殊関数の実装方針](special-functions.md) — `XMath.Special` の配置設計、定義域外と極の扱い、静的初期化順、倍々精度（double-double）と漸近展開の使い分け
- [疑似乱数・準乱数の設計](random-and-quasirandom.md) — PCG64 DXSM / Philox の再現性とストリーム設計、Sobol' 列の Gray code・半 ULP シフト・Matoušek スクランブル
- [自動微分の設計](differentiation.md) — リバースモード Wengert テープと世代検証、`ArrayStore` のプーリングと LRU 保持上限、ビット一致逆伝播、行列分解の解析微分
- [信号処理・統計分布の設計](signal-processing-and-statistics.md) — 決定的な FFT / Stockham / Bluestein・DCT/DST・ウェーブレット設計、統計分布の対数裾・Loader 鞍点展開・DE 積分・Genz 矩形確率・Ziggurat サンプリング
- [数値積分・最適化の設計](integration-and-optimization.md) — Gauss–Kronrod / Genz–Malik / RQMC 適応求積、L-BFGS-B / Moré–Thuente・FISTA / ADMM・McCormick 分枝限定・混合ゲノム GA
- [確率的推論・ニューラルネットワークの設計](probabilistic-and-neural-network.md) — 構造体ジェネリクス MCMC・決定論的ストリーム・Stan 準拠適応スケジュール、`Convolution2D` / `MambaBlock` のレイアウトとビット一致並列化・SafeTensors 検証
