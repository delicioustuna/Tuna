# 拡張パッケージの概要

Tuna はコアパッケージ `Tuna` の上に、領域別の 7 つの拡張パッケージを提供しています。必要なパッケージだけを参照して利用できます。
各パッケージの詳細な API 一覧は [API リファレンス一覧](api/README.md) を参照してください。

## 1. `Tuna.Statistics` — 確率分布

連続分布（正規・ガンマ・ベータ・スチューデント t・指数・ワイブル等）、離散分布（ベルヌーイ・二項・ポアソン・カテゴリカル等）、多変量分布（多変量正規・ディリクレ・ウィシャート・LKJ 相関等）、合成分布（位置尺度変換・切断分布）を提供します。

- 統一インターフェース: `IContinuousDistribution`, `IDiscreteDistribution`, `IMultivariateContinuousDistribution`, `IMultivariateDiscreteDistribution`
- 主な操作: `Pdf`/`Pmf`, `LogPdf`/`LogPmf`, `Cdf`, `Sf`, `LogCdf`, `LogSf`, `InverseCdf`, `InverseSf`, `InverseLogCdf`, `InverseLogSf`, `Sample`

```csharp
using Tuna.RandomSource;
using Tuna.Statistics.Distributions;

var normal = new NormalDistribution(mean: 1.0, standardDeviation: 2.0);
double logP = normal.LogPdf(2.5);
double q95 = normal.InverseCdf(0.95);
double sample = normal.Sample(new Pcg64DxsmRandom(seed: 1));
```

単体分布・多変量分布の `LogPdf` / `LogPmf`（および対応する分布の `LogCdf`・`LogSf`）には、自動微分に対応する静的オーバーロードがあります。観測ベクトルの対数尤度は 1 回の演算としてテープに記録されます。ロバスト回帰でスチューデントの t 分布を尤度に、コーシー分布を尺度の事前分布に用いる例や、MCMC サンプラーとの連携方法については [確率分布の自動微分](distribution-differentiation.md) および [MCMC への接続](distribution-differentiation.md#mcmc-への接続) を参照してください。

API 一覧: [Tuna.Statistics.md](api/Tuna.Statistics.md)

## 2. `Tuna.SignalProcessing` — FFT・畳み込み・ウェーブレット

周波数変換・畳み込み・相関・ウェーブレット変換に特化したパッケージです（窓関数や IIR / FIR デジタルフィルタ設計は含みません）。

- **フーリエ変換**: 1D/2D 複素 FFT、実数 FFT、DCT / DST、ズーム FFT、周波数軸生成（`Fourier`、再利用可能な計画オブジェクト `FourierPlan` / `RealFourierPlan`）
- **畳み込み・相関**: 1D 畳み込み（`Full` / `Same` / `Valid`）、整数厳密畳み込み、相互相関・自己相関（`XMath` 拡張: `XMath.Convolve` / `XMath.Correlate` / `XMath.AutoCorrelate`。`XMath.` を付けて呼び出します）
- **ウェーブレット**: 離散ウェーブレット変換（DWT / 多段 / 2D / MODWT / SWT / パケット / ノイズ除去 `Wavelets.Denoise`）、連続ウェーブレット変換（`Wavelets`, `DiscreteWavelet`, `ContinuousWavelet`）

```csharp
using Tuna.LinearAlgebra;
using Tuna.SignalProcessing;
using Complex = System.Numerics.Complex; // System.Numerics.Vector<T> との名前衝突を回避

var signal = new Vector<double>([1.0, 0.0, -1.0, 0.0]);
Vector<Complex> spectrum = Fourier.ForwardReal(signal);
Vector<double> recovered = Fourier.InverseReal(spectrum, signal.Length);
```

API 一覧: [Tuna.SignalProcessing.md](api/Tuna.SignalProcessing.md)

## 3. `Tuna.Optimization` — 数値最適化

用途に応じて 5 つの名前空間に分かれています。

- `Tuna.Optimization.Gradient`: L-BFGS（`Lbfgs`）、L-BFGS-B（箱制約、`LbfgsB`）
- `Tuna.Optimization.Stochastic`: SGD（`StochasticGradientDescent`）、Adam（`AdamOptions { DecoupledWeightDecay = true }` で AdamW 方式）、NAdam、AdaGrad、AdaMax、RMSProp、学習率スケジュール（`CosineAnnealingSchedule` 等）
- `Tuna.Optimization.Proximal`: FISTA、ADMM、近接写像（L1 / L2 / ElasticNet / GroupLasso / 非負 / 単体 / 箱制約など）
- `Tuna.Optimization.Global`: 区間演算・McCormick 緩和と分枝限定法による大域的最小化
- `Tuna.Optimization.Evolutionary`: 遺伝的アルゴリズム（実数・二値・順列・混合ゲノム、ask/tell API）

自動微分で学習する場合は、`ParameterSet<float>` または `ParameterSet<double>` を `Adam<T>` などのコンストラクタに渡します。逆伝播後の `Step()` はパラメーターをインプレースで更新し、`GradientClipNorm` には全パラメーターの勾配に対する L2 ノルムの上限を指定できます。

API 一覧: [Tuna.Optimization.md](api/Tuna.Optimization.md)

## 4. `Tuna.Integration` — 数値積分

- `Quadrature`: 1 次元の適応的 Gauss-Kronrod 積分・二重指数関数型積分・台形則
- `Cubature`: 多次元の適応的数値積分
- `QuasiMonteCarlo`: `SobolSequence` を用いた準モンテカルロ積分と標準誤差推定

API 一覧: [Tuna.Integration.md](api/Tuna.Integration.md)

## 5. `Tuna.Differentiation` — 自動微分

逆モード自動微分（テープベース）を提供します。スカラー（`ScalarNode`）、ベクトル・行列・3 階テンソル・埋め込み・連立一次方程式・LU / Cholesky / QR / 対称固有値 / SVD 分解（`VectorNode<T>`, `MatrixNode<T>`, `TensorNode<T>`）、および `[ElementwiseOperation]` / `IDifferentiableOperation<T>` によるユーザー定義演算に対応しています。演算は `XMath` / `XMath.Special` の拡張メンバーとして提供されるため、`using static Tuna.XMath;` ではなく `XMath.Sin(x)`・`XMath.Sum(v)` のように修飾して呼び出してください。

```csharp
using Tuna;
using Tuna.Differentiation;

var tape = new Tape();
var x = tape.Leaf(2.0);
var y = XMath.Sin(x) + x * x;
tape.Backward(y);
double dydx = tape.GetGradient(x); // cos(2) + 4
```

API 一覧: [Tuna.Differentiation.md](api/Tuna.Differentiation.md)

## 6. `Tuna.Probabilistic` — 確率的推論（MCMC）

HMC・NUTS・楕円スライスサンプリング・アンサンブルサンプラー・レプリカ交換法、制約空間への変換（`BoxBijector` 等）、および収束診断（$\hat{R}$, ESS, MCSE: `ConvergenceDiagnostics` / `SamplingResult.Summarize`）を提供します。

API 一覧: [Tuna.Probabilistic.md](api/Tuna.Probabilistic.md)

## 7. `Tuna.NeuralNetwork` — ニューラルネットワーク層と SafeTensors

2D 畳み込み（`Convolution2D<T>` / `Convolution2DStream<T>`）、Mamba 状態空間モデルブロック（`MambaBlock<T>`）、正規化・補助演算（`XMath` 拡張: `XMathExtensions` / `XMathInPlaceExtensions`）、および HuggingFace 互換の SafeTensors フォーマット**読み取り**（`SafeTensorFile`、学習済み重みの読み込み向け。書き出しは非対応）を提供します。

学習用には、全結合・LayerNorm・RMSNorm・PReLU・2 次元畳み込み・因果畳み込み・選択的走査を `Tuna.Differentiation` のテープに記録する `XMath` 拡張（`XMathDifferentiationExtensions`）を提供します。`MatrixNode<T>`・`VectorNode<T>`・`TensorNode<T>` を受け取る `XMath.Linear`・`XMath.LayerNorm`・`XMath.RmsNorm`・`XMath.Prelu`・`XMath.Convolve2D`・`XMath.CausalDepthwiseConvolve`・`XMath.SelectiveScan` を利用でき、順伝播の出力値は推論用の `XMath` 演算とビット単位で一致します（いずれも `XMath.` を付けて呼び出します）。学習後の `Parameter<T>` は `AsMatrixSpan()` / `AsVectorSpan()` / `AsTensorSpan()` でコピーせずに推論パスへ渡せます。推論のみで利用する場合、`Tuna.Differentiation.dll` は読み込まれません。

API 一覧: [Tuna.NeuralNetwork.md](api/Tuna.NeuralNetwork.md)
