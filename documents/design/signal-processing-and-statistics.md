# 信号処理・統計分布の設計

## 1. 信号処理（`Tuna.SignalProcessing`）

### 畳み込みと相関（`XMathExtensions.Convolution.cs`）
- **API の公開構成**: 畳み込み・相関・自己相関は `[EditorBrowsable(EditorBrowsableState.Never)]` を付与した `XMathExtensions` の `extension(XMath)` として定義し、利用者が `XMath.Convolve` 等の統一入口から呼び出せるようにしています。
- **直接計算と FFT の自動選択**: `ConvolutionMethod.Auto` は `double`・`Complex` に対して `min(M, N) > 32` かつ `M * N > 16384` の場合にのみ FFT を選択します。それ以外は $O(MN)$ の直接和を使い、整数型での厳密性と小振幅の裾での相対精度（FFT 丸め誤差は $\|x\|_2 \|y\|_2$ に比例）を守ります。
- **パディングと相関の定義**: 線形畳み込みは `FourierPlan.NextFastLength(M + N - 1)` に、自己相関は `NextFastLength(n + maxLag)` にパディングして $| \mathrm{FFT}(x) |^2$ から算出します。`Correlate(..., Same)` のオフセットは NumPy に一致させ、入力と出力先の重複は `StridedLayout.Overlaps` で拒否します。

### 複素・実数 FFT コア（`Fourier/`）
- **SIMD 幅に依存しない決定性**: `IComplexLanes<TSelf>`（`Lanes1` / `Lanes2` / `Lanes4`）は FMA を避けて `x * wr + (i * x) * wi` として複素乗算を統一評価し、SIMD 幅とスカラー端数ループ間でビット一致を保証します。
- **8 分円還元による回転因子（`UnitRoots.cs`）**: 整数 8 分円還元 $8k = o \cdot n + r$ で $[0, \pi/4]$ へ写像してから三角関数を評価し、漸化誤差の蓄積を防ぎつつ $n/8$ の倍数での厳密値・共役対称性・`+0.0` 正規化を保証します。
- **決定的な因数分解と Bluestein フォールバック（`FourierPlan.Core.cs`）**: 実行時ベンチマークは行わず、長さのみから決定的にプランを生成し、上限 64 件の `ConcurrentDictionary` にキャッシュします（`TrigonometricPlan` も同様）。2 の冪は基数 8 を優先し、末尾の `8 * 2` を `4 * 4` へ変換します。素因数 31 以下は Stockham ステージで直接実行し、それより大きい素因数を含む長さは `NextFastLength(2n - 1)` の Bluestein 畳み込みにフォールバックします（`1/m` は `kernelSpectrum` に事前折り込み、逆変換は $\overline{\mathrm{DFT}(\bar{x})}$）。
- **Stockham カーネルと並列化（`StockhamKernels.cs`）**: 基数 2, 3, 4, 5, 8 は JIT インライン上限内に収めるため `[MethodImpl(NoInlining)]` の専用カーネルとし、一般奇数基数カーネルは $p^2/2$ 対称性を利用します。単一変換のステージ 0 では 4 レーンパックした `FirstTwiddles` でキャッシュ競合を避けつつ、`j1 = 0` は `-0.0` や `Infinity` の変質を防ぐため乗算なし経路（`WithoutTwiddles`）で処理してバッチ変換とビット一致させます。並列化は `FirstBlock = 4` 境界で分割し、スレッド数によらずビット一致します。
- **実数 FFT パッキング（`RealFourierPlan.Core.cs`）**: 偶数長は隣接実数を $n/2$ 複素数へパック（単一変換時はゼロコピーの `Unsafe.As<double, Complex>`）して複素 FFT を実行し、共役対 $(k, h-k)$ を SIMD・並列展開します。奇数長は長さ $n$ の複素 FFT へ委譲します。
- **ストライド・2D FFT・Chirp-Z・DCT/DST**:
  - `FourierViews` は列優先密配置（直接バッチ）・行連続配置（行並列）・一般ストライド（最大 512 KiB ブロックの gather/scatter）の 3 経路でディスパッチします。
  - `Forward2D` は行 → 列、`Inverse2D` は列 → 行の順で変換します。`InverseReal2D` は入力を変更しないよう `ArrayPool<Complex>` の中間バッファを使用します。
  - `Zoom`（Rabiner–Schafer–Rader 1969）は位相回転数を FMA による倍々精度で $\bmod 1$ 累積し `double.SinCosPi` で評価して長信号でも位相精度を維持します。
  - DCT / DST（`TrigonometricPlan`）は SciPy の非正規化定義（`norm="backward"`）に従い、各タイプを単一 FFT へ帰着します（DCT-II/III は Makhoul 1980 の実数 FFT、DST-II〜IV は符号反転・逆順を前後ループへ統合）。

### ウェーブレット（`Wavelets/`）
- **フィルタ係数と定義**: 離散フィルタ（`db1`–`db20`, `sym2`–`sym20`, `coif1`–`coif5`, `bior`/`rbio`）は定義方程式から 60 桁精度で計算して倍精度へ丸めています。`ContinuousWavelet` は Torrence–Compo (1998, Table 1) の単位エネルギー正規化に従い、指数下限 `-708` で非正規化数による性能低下を避けています。
- **1D / 2D 離散ウェーブレット変換（DWT）**: 1D では拡張信号を偶奇位相へ分けて L1 タイル（`Block = 1024`）の連続 SIMD 積和（FMA 不使用で幅非依存ビット一致）を実行します。2D の行変換は行方向へ SIMD ベクトル化しつつ 1D と同一のタップ加算順を保ち、行ごとの 1D 変換とビット一致させます。
- **定常ウェーブレット変換（SWT / MODWT）**: `Stationary` は直交・双直交の両方および任意長 $n \ge 1$ に対応し、`MaximalOverlap`（MODWT）は Percival–Walden の $1/\sqrt{2}$ スケーリングでエネルギー保存則を満たします。行優先の `ArrayPool` 作業領域上で巡回折り返しを高々 2 つの連続 SIMD スイープに分割します。
- **ウェーブレットパケットと最良基底**: レベル $j$ を $m \times 2^j$ の列優先行列に自然順で格納し、Gray code（`p ^ (p >> 1)`）で周波数順と相互変換します。`BestBasis` はボトムアップ DP で非正規化加法的コストを最小化します（同点時は親ノード優先）。
- **連続ウェーブレット変換（CWT）とノイズ除去**: CWT は Torrence–Compo (1998) を FFT 畳み込みで実装し、信号の FFT を全スケールで再利用して独立スケールを並列化します。`InverseContinuous` は対数スケール台形重みと 64 周波数平均による再構成係数 $K$ の数値較正により、任意の許容ウェーブレットでハードコード表なしに逆変換できます。ノイズ除去は MAD 推定と introselect 中央値選択（最悪 $O(n \log n)$）および `XMath.InPlace.SoftThreshold` を用います。

---

## 2. 統計分布（`Tuna.Statistics`）

### インターフェース階層・値型合成・API 設計
- **1 変量分布**: `IUnivariateDistribution<TValue>` から `IContinuousDistribution`（`double`）と `IDiscreteDistribution`（`long`）へ分岐します。具象型は不変な `readonly struct` とし（テーブル保持の `CategoricalDistribution` のみ `sealed class`）、`LocationScaleDistribution<TDistribution>` や `TruncatedDistribution<TDistribution>` のジェネリック合成でボクシング・ヒープ確保を発生させません。
- **多変量・行列分布**: `IMultivariateContinuousDistribution`・`IMultivariateDiscreteDistribution`・`IMatrixContinuousDistribution` はスパンベースのゼロアロケーション API を中核とし、C# 14 `extension(...)` ブロックで結果新規確保・バッチ版オーバーロードを提供します。バッチ入出力は列優先の `Dimension × N` 行列（1 列 = 1 点）であり、単一点の逐次呼び出しと値・乱数消費順序が一致します。
- **パラメータ化と対称行列・単体のサポート判定**:
  - 連続分布の逆関数は定義域外・NaN で `double.NaN` を返し、離散分布の逆関数は `ArgumentOutOfRangeException` をスローします。
  - 行列分布（`Wishart`, `InverseWishart`, `LKJ`）の密度は独立な下三角要素（LKJ は狭義下三角）のルベーグ測度に対して正規化します。対称性判定は丸め誤差許容幅 $\sqrt{\varepsilon} \approx 1.49 \times 10^{-8}$ を認め、下三角を鏡映して厳密な対称性を保ちます。単体上のサポート判定は $|\sum x_i - 1| \le 16 K \varepsilon$ を許容します。

### 対数空間・深い裾・オーバーフロー回避
- **深い裾の統一閾値（`LogProbabilityLinearLimit = -700`）**: 対数確率が $-700$ 未満では線形確率を経由せず、対数座標や標準化座標上で直接ブラケット付きニュートン法（`DistributionMath.SolveIncreasing`、二分法フォールバック付き）により分位点を求めます。
- **2 の冪スケーリングによる安全アフィン変換**: `Standardize` / `Unstandardize` / `AddLowerProduct` は中間オーバーフローを検知すると 2 の冪での半減・倍増や `ILogB`/`ScaleB` の仮数指数分離（`ScaledRatio`）へ切り替え、`AddScaledLowerProduct` は $L t$ を先に計算して $0 \cdot \infty = \mathrm{NaN}$ を防ぎます。
- **下側・上側相補評価**: $p = 0.5$（$\ln p = -\ln 2$）を境界に `Cdf`/`Sf` と `Log1p(-Sf)`/`Log1mExp` を切り替え、減算による桁落ちを防ぎます。

### Loader (2000) 鞍点展開・離散エントロピー・DE 積分
- **Loader (2000) 剰余項（`DistributionMath`）**: `StirlingRemainder`（`stirlerr`）、シフト版剰余項、および `Bd0` により、`Binomial`, `Poisson`, `NegativeBinomial`, `Multinomial`, `NegativeMultinomial`, `DirichletMultinomial` の `LogPmf` で高精度かつ桁落ちのない評価を実現しています。
- **離散逆関数とハイブリッドエントロピー（`DiscreteMath`）**: 2 次 Cornish–Fisher 初期推定から指数ブラケット＋二分探索を行い、$p > 0.5$ では `Sf` を基準に判定します。エントロピーは歪度・尖度が十分小さい場合に 2 次 Edgeworth 展開、通常はモード中心の Neumaier 補償外向き和（幾何減衰打ち切り）、負の二項分布の広裾領域では PGF を用いた台形積分を切り替えます。
- **二重指数（DE）数値積分と切断分布（`DoubleExponentialQuadrature`, `TruncatedDistribution`）**: 有限区間 tanh-sinh / 半無限区間 exp-sinh 積分をスタック割り当て多成分 Neumaier 補償で評価し、特異端点の丸め誤差をべき乗則補外で補正します。切断分布は `F(v) - F(u)` と `S(u) - S(v)` の被減数が小さい方を選び、3 ビット以上の桁落ち（`CancellationRatio = 0.125`）が起きる場合は中点シフト DE 積分へフォールバックします。

### 多変量正規・t 分布の矩形確率（`GenzProbability`）
- Genz (1992) のコレスキー変換により $P(\text{lower} < X \le \text{upper})$ を超立方体 $(0, 1)^{d-1}$（有限自由度 $t$ 分布では動径 $\chi^2(\nu)$ 分位点を含む $(0, 1)^d$）上の積分へ変換します。条件付き中点確率を最小化する貪欲ピボットで分散を低減し、積分次元 $m \le 2$ は適応積則（`Cubature`）、$m > 2$ は固定シード（`Pcg64DxsmRandom`）の `QuasiMonteCarlo`（既定絶対誤差 $10^{-8}$、最大評価回数 $2^{22}$）で決定的に評価します。

### サンプリングアルゴリズム（`Ziggurat`, `Sampling`）
- **Ziggurat（256 層）**: 1 語の `ulong` から互いに素なビット領域（層・符号・一様乱数）を切り出して Doornik (2005) の層–一様相関を回避しています。テーブルは初期化時に二分法で動的構築し、裾は Marsaglia (1964) 棄却法（正規）および無記憶性（指数）で補完します。
- **各種サンプリング**: `Bernoulli` は 64-bit 2 進展開比較による厳密サンプリング、`StandardGamma` / `LogStandardGamma` は Marsaglia & Tsang (2000)（$a < 1$ は対数空間で実行してアンダーフロー回避）、`Poisson` / `Binomial` は小期待値で逐次逆関数法・中〜大期待値で Hörmann (1993) PTRS/BTRS＋Loader (2000) 鞍点判定（`Poisson` の $\lambda > 10^{18}$ は正規近似＋`long.MaxValue` 飽和）、`Categorical` は Vose (1991) エイリアス法、`Wishart` / `InverseWishart` は Bartlett 分解、`LKJ` は Onion/Cholesky 法を採用しています。
