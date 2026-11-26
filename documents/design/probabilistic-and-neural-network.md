# 確率的推論・ニューラルネットワークの設計

## 1. 確率的推論・MCMC（`Tuna.Probabilistic`）

### 構造体ジェネリクスによるゼロオーバーヘッド抽象化
- 対数密度（`ILogDensity` / `IDifferentiableLogDensity`）と制約変換（`IBijector`）をインターフェースとして定義し、各サンプラーのジェネリック型引数（`TDensity`, `TBijector` 等）で受け取る設計としています。利用者が `struct` で実装して渡すことで、JIT による仮想呼び出し除去とインライン展開が可能になります。
- `IDifferentiableLogDensity` の既定の明示的実装 `ILogDensity.LogDensity` は勾配バッファを都度確保するため、対数密度の値だけを安価に評価できる型は `ILogDensity.LogDensity` を個別実装する前提としています。

### 並列度に依存しない決定論的乱数ストリーム
- 全サンプラーで `Philox4x32Random` のストリーム ID を次のビット配置で固定生成しています（再現性のための固定仕様）。
  - `[63..56]`: 用途（`Transition = 0`, `Initialization = 1`, `Exchange = 2`）
  - `[55..32]`: レプリカ番号（`uint`）
  - `[31..0]`: チェーン／ウォーカー番号（`uint`）
- これにより、同一の `Seed`・設定・初期点・対数密度であれば `MaxDegreeOfParallelism` によらずビット単位で同一の標本列が得られます。

### 共通ウォームアップ適応スケジュールと双対平均法
- ウォームアップ期間は Stan 準拠の 3 段階（初期バッファ 75 ステップ → 基準幅 25 から倍々拡張する適応ウィンドウ → 終端バッファ 50 ステップ）で構成します。合計が `Warmup` を超える場合は 15% / 10% に縮小し、`Warmup < 20` では計量・共分散適応をスキップしてステップ幅のみ適応します。
- 共分散推定は Welford 法でオンライン更新し、各ウィンドウ終端で Stan と同じ縮小推定 $\frac{n}{n+5}\hat{\Sigma} + 10^{-3}\frac{5}{n+5}I$ を適用して正定値性を保証します。
- 双対平均法（Nesterov 2009 / Hoffman & Gelman 2014 §3.2、$t_0 = 10$, $\kappa = 0.75$）のハイパーパラメータは手法特性に応じて切り替えています。
  - **NUTS / HMC**: 大きめの初期ステップ幅から探索するため `centerFactor = 10`、`gamma = 0.05`（論文推奨値）。初期ステップ幅未指定時は 1 リープフロッグ受理確率が 0.5 をまたぐ値を倍増・半減探索します（Hoffman & Gelman 2014 Algorithm 4）。
  - **Adaptive RWM / Parallel Tempering のローカル更新**: 理論最適スケール $2.38 / \sqrt{d}$（Gelman, Roberts & Gilks 1996）および最適目標受理率（1 次元: 0.44、2 次元以上: 0.234; Roberts, Gelman & Gilks 1997; Roberts & Rosenthal 2001）から開始するため `centerFactor = 1`、`gamma = 1` を採用しています。単一遷移の受理確率 $\min(1, \pi'/\pi)$ は分散が大きく、`gamma = 0.05` では最終固定スケールが目標受理率から乖離しやすいためです。

### 制約変換の作業バッファ方針
- `TransformedLogDensity`・`TransformedDifferentiableLogDensity`・`SimplexBijector` の一時作業領域は、`Bijection.StackLimit`（256 要素）以下を `stackalloc`、超過時は `ArrayPool<double>.Shared` から借用して `finally` で返却します。低〜中次元でのヒープアロケーションを避けつつ、スレッドセーフな並列評価を両立しています。

### 各サンプラーおよび収束診断の採用方針
- **NUTS (`NoUTurnSampler`)**: Hoffman & Gelman (2014) と Betancourt (2017) の多項サンプリング版を採用。倍増間のバイアス付き漸進サンプリング、部分木内の一様多項サンプリング、部分木間境界を含む一般化 U ターン判定を組み合わせています。
- **HMC (`HamiltonianMonteCarlo`)**: 固定リープフロッグ回数に加え、目標分布の周期との共振による停滞を防ぐステップ幅ジッター（Neal 2011 §5.4.2.2）を提供します。発散閾値超過時は即座に打ち切り・棄却します。
- **Adaptive RWM (`RandomWalkMetropolis`)**: ウィンドウ単位の適応メトロポリス法（Haario, Saksman & Tamminen 2001）と双対平均法スケール適応を組み合わせ、サンプリング期間中はパラメータを完全固定してエルゴード性を保証します。
- **Slice Sampler (`SliceSampler`)**: Neal (2003, Figs. 3 & 5) のステップアウト・縮小 1 次元スライス更新に、ウィンドウ標本共分散の固有ベクトルを用いる因子スライス法（Tibbits et al. 2014）と標準偏差 3 倍幅適応を組み合わせています。
- **Elliptical Slice Sampler (`EllipticalSliceSampler`)**: Murray, Adams & MacKay (2010, Fig. 2) による正規事前分布＋任意尤度向けサンプラー。
- **Ensemble Sampler (`EnsembleSampler`)**: Goodman & Weare (2010) のアフィン不変ストレッチ移動を、ウォーカー 2 群分割の赤黒並列更新で実装し、詳細釣り合いと並列化を両立しています。
- **Parallel Tempering (`ParallelTempering`)**: Syed, Bouchard-Côté, Deligiannidis & Doucet (2022) の非可逆交換（偶数／奇数ペア交互試行）を採用。累積交換棄却率の区分線形補間（下限 `MinimumRejection = 1e-3` で狭義単調性を保護）により温度ラダーを適応し、往復回数と大域的通信障壁 $\Lambda$ を診断値として提供します。全密度焼きなましでは $\beta_{\min} = 0$ が正規化不能となるため正値を要求します（目安: モード間対数密度差 $\Delta E$ に対し $\beta_{\min}\Delta E \lesssim 1$）。
- **収束診断 (`ConvergenceDiagnostics`)**: Vehtari et al. (2021) 準拠の順位正規化・折り返し分割 $\hat{R}$、FFT 自己相関＋Geyer 初期正単調列によるバルク／裾／分位点 ESS、Beta 分布による分位点 MCSE（§4.3）、および E-BFMI を実装しています。

---

## 2. ニューラルネットワーク・推論演算（`Tuna.NeuralNetwork`）

### テンソル・特徴量レイアウトと対応型
- 2 次元特徴行列は列優先の `[channels × positions]`（行 = チャネル、列 = 時刻・空間位置・系列位置）を標準とします。
- `Convolution2D` の 3 次元特徴マップは `[channels, width, height]` です（PyTorch の行優先 `[C, H, W]` とは軸順が異なります）。
- 計算対応型は `float` と `double` です。カーネル本体は列要素が連続（`RowStride == 1`、列ストライドは任意）であることを前提とし、非連続な入力・出力は必要な場合のみ `ArrayPool<T>` 経由で詰め替えます。

### SIMD 幅・スレッド数に依存しないビット一致再現性
- 並列化はコアの `Parallelism`（`LinearAlgebraSettings.MaxDegreeOfParallelism`）に委譲し、正規化・PReLU・im2col-GEMM の独立列、`SelectiveScan` の独立チャネル、`Step` / `SelectiveScanStep` の独立系列など、結果がスレッド数や分割数に依存せずビット単位で一致するようにタスクを分割しています。
- 要素ごとの SIMD カーネルはレーン独立演算と FMA を `Vector512`・`Vector256`・スカラー経路で統一し、命令セット幅によるビット差が出ないようにしています。
- 正規化（`NormalizeColumn`）の平均・分散縮約は、SIMD 累積器幅を `Vector256` に固定して加算順序を CPU 間で統一し、Chan–Golub–LeVeque (1983) のシフト分散法で桁落ちと平均丸め誤差の偏りを防いでいます。

### 名前空間と Source Generator の構成
- `using Tuna.LinearAlgebra;` は `namespace Tuna.NeuralNetwork;` 内に配置し、`Tuna.LinearAlgebra.Vector<T>` が `System.Numerics.Vector<T>` より優先されるようにしています。
- `XMathInPlaceExtensions`（`extension(XMath.InPlace)`）は `XMathExtensions`（`extension(XMath)`）と別クラスに分離し、`[InferenceOverloads]` Source Generator におけるオーバーロードシグネチャ衝突を防いでいます。
- `XMathExtensions`・`XMathInPlaceExtensions`・`XMathDifferentiationExtensions` にはいずれも `[EditorBrowsable(EditorBrowsableState.Never)]` を付与し、IntelliSense 上の入口を `XMath` / `XMath.InPlace` に統一しています。

### 2 次元畳み込みとストリーミング実行（`Convolution2D`, `Convolution2DStream`）
- 重みは構築時に 1 度だけピン留めされた列優先配列 `[OutChannels × K]`（im2col 行順 `(i, j, c)`）へ詰め替えます。これにより、推論時の im2col 列構築がチャネル長の連続 `Buffer.MemoryCopy` だけで完結します（Chellapilla, Puri & Simard 2006）。
- `Convolution2DStream` は `(KernelHeight - 1) * DilationHeight + 1` 行分のピン留めリングバッファを保持します。現在行が連続かつ出力と非重複なら入力から直接読みつつリングバッファへ並列コピーし、GEMM 重みパッキングの重複を避けるため並列チャンク数をスレッドあたり 1 に抑えています。

### 状態空間モデル（`MambaBlock`, `MambaState`, `XMathExtensions.StateSpace`）
- `in_proj → causal depthwise conv1d → SiLU → x_proj / dt_proj (+ softplus) → selective scan → SiLU gate → out_proj` を実行し、`mamba_ssm` 形式の行優先重みは構築時にピン留め列優先バッファへ 1 回だけ変換します。
- `Forward` では射影・畳み込みを系列位置で並列化し、`SelectiveScan` は偽共有を避けるためスレッドローカル作業領域を用いて `ChannelUnit = 16`（`Vector512<float>` 倍数）単位のチャネルブロックで並列化します。`MambaState` を引き継ぐチャンク実行は全系列一括の `Forward` とビット単位で一致します。
- `Step` は独立した系列ごとに並列化し、各スレッドが担当系列の全パイプラインを実行します。

### 学習用の記録（`XMathDifferentiationExtensions`）
- 学習用 API は推論用の `XMathExtensions` と別の静的クラス `XMathDifferentiationExtensions`（`extension(XMath)`）に配置し、`MatrixNode<T>`・`VectorNode<T>`・`TensorNode<T>` を受け取ります。`Tuna.Differentiation` の型を参照するのは `XMathDifferentiationExtensions` だけであるため、推論のみの呼び出しでは `Tuna.Differentiation.dll` を読み込みません。
- 各層は `IDifferentiableOperation<T>` を実装した非公開の readonly struct で、`MatrixNode<T>.Record` または `TensorNode<T>.Record` により 1 つの演算として記録します。順伝播は推論と共通の内部カーネルを呼び出すため、出力値は推論 API とビット単位で一致します。
- LayerNorm・RMSNorm は、順伝播カーネルが求める列ごとの `(K, δ, r = 1/√(σ² + ε))` だけを `AllocateSaved` に保存します。逆伝播では正規化した値 `x̂ = ((x − K) − δ)·r` を順伝播と同じ式で再計算するため、`x̂` 自体を保存せず（保存量は列数の 3 倍）、順伝播との丸め差も入りません。
- 逆伝播の列ごとのカーネル（`GradientKernels.NormalizeBackwardColumn`・`PreluBackwardColumn`）は、順伝播と同じく縮約の SIMD 幅を `Vector256` に固定し、列の範囲で並列化します。パラメーター勾配は作業領域に置いた列ごとの項を `XMath.RowSums`（共有傾き PReLU は `XMath.Sum`）で集約するため、並列数・命令セットによらずビット単位で一致します。
- 全結合の逆伝播は `Blas.Gemm` を β = 1 で呼び出し、勾配のビューへ直接加算します（転置ビューの場合は転置した積として計算します）。行列積の逆伝播と同じ呼び出しになるため、勾配は基本演算で組み立てた記録とビット単位で一致します。
- 推論の正規化・PReLU の並列処理は値型ワーク（`Split.For<TWork>`）で渡し、呼び出しごとのクロージャ・デリゲートのアロケーションを排除しました。学習記録の 2 回目以降のゼロアロケーションはこれに依存しています。
- 2 次元畳み込みの重み勾配には、推論と同じ展開処理で再計算した入力パッチを使用します。入力勾配は重みの転置と出力勾配の行列積をパッチ内の有効な位置へ加算し、重複する位置は出力位置の順で集約します。
- 選択的走査の逆伝播は、短い系列では保存した状態を直接参照します。系列長が 64 を超えると 32 位置ごとに状態を保存し、順伝播と同じカーネルでチェックポイント間の状態を区間ごとに再計算します。状態の保存と作業領域の大きさは系列長に応じて決まります。

### SafeTensors I/O（`SafeTensorFile`, `SafeTensorHeader`）
- 公式 `safetensors` 仕様の検証ルールに準拠しています（ヘッダ上限 100 MB、JSON ネスト上限 `MaxDepth = 128`（`serde_json` 準拠）、無視される追加フィールドを含む全ヘッダでの厳密な UTF-8・エスケープ検証、`__metadata__: null` を未指定扱い、JSON キー順に依存しないデータ領域の隙間なし非重複被覆検証）。
- 読み取り操作と `Dispose` はインスタンスごとのロックで直列化しています。
