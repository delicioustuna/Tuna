# Tuna.NeuralNetwork API リファレンス

> このファイルは `tools/ApiDocGenerator` によって自動生成されています。手動で編集せず、`dotnet build Tuna.slnx -c Release` の後に再生成してください。
>
> `(+N overloads)` は `Vector` / `VectorSpan` / `ReadOnlyVectorSpan` 等のコンテナ・ビュー組み合わせや `double` / `float` 固定型オーバーロードの省略を表します。

## 名前空間と公開型

- [`Tuna.NeuralNetwork`](#tunaneuralnetwork) (9 型): [`Convolution2D<T>`](#class-convolution2dt), [`Convolution2DStream<T>`](#class-convolution2dstreamt), [`MambaBlock<T>`](#class-mambablockt), [`MambaDimensions`](#readonly-struct-mambadimensions), [`MambaState<T>`](#class-mambastatet), [`Padding2D`](#readonly-struct-padding2d), [`XMathDifferentiationExtensions`](#static-class-xmathdifferentiationextensions), [`XMathExtensions`](#static-class-xmathextensions), [`XMathInPlaceExtensions`](#static-class-xmathinplaceextensions)
- [`Tuna.NeuralNetwork.IO`](#tunaneuralnetworkio) (3 型): [`SafeTensorDataType`](#enum-safetensordatatype), [`SafeTensorFile`](#class-safetensorfile), [`SafeTensorInfo`](#class-safetensorinfo)

## Tuna.NeuralNetwork

### `class Convolution2D<T>`

Performs 2D convolution (cross-correlation) over feature maps. / 特徴マップに対する 2 次元畳み込み（相互相関）を実行します。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `Convolution2D(ReadOnlySpan<T> weight, ReadOnlySpan<T> bias, int outChannels, int inChannels, int kernelHeight, int kernelWidth, int strideHeight = 1, int strideWidth = 1, int dilationHeight = 1, int dilationWidth = 1)` | Initializes a new `Convolution2D` layer from weights and optional bias. / 重みと任意のバイアスから `Convolution2D` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int DilationHeight { get; }` | Gets the dilation along the height dimension. / 高さ方向のダイレーションを取得します。 |
| `int DilationWidth { get; }` | Gets the dilation along the width dimension. / 幅方向のダイレーションを取得します。 |
| `int InChannels { get; }` | Gets the number of input channels. / 入力チャネル数を取得します。 |
| `int KernelHeight { get; }` | Gets the kernel height. / カーネルの高さを取得します。 |
| `int KernelWidth { get; }` | Gets the kernel width. / カーネルの幅を取得します。 |
| `int OutChannels { get; }` | Gets the number of output channels. / 出力チャネル数を取得します。 |
| `int StrideHeight { get; }` | Gets the stride along the height dimension. / 高さ方向のストライドを取得します。 |
| `int StrideWidth { get; }` | Gets the stride along the width dimension. / 幅方向のストライドを取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `Convolution2DStream<T> CreateStream(int width, int left, int right)` | Creates a causal streaming convolution that processes one height row at a time. / 高さ方向に 1 行ずつ処理する因果的な 2 次元畳み込みストリームを作成します。 |
| `Tensor<T> Forward(ReadOnlyTensorSpan<T> x, Padding2D padding)` | Applies the 2D convolution to `x` and returns a new tensor. / 特徴マップ `x` に 2 次元畳み込みを適用し、新しいテンソルとして返します。 |
| `TensorSpan<T> Forward(ReadOnlyTensorSpan<T> x, Padding2D padding, TensorSpan<T> destination)` | Applies the 2D convolution to `x` and writes the result into `destination`. / 特徴マップ `x` に 2 次元畳み込みを適用し、結果を `destination` に書き込みます。 |
| `int OutputHeight(int height, Padding2D padding)` | Computes the output height for the specified input height and padding. / 指定した入力の高さとパディングに対する出力の高さを計算します。 |
| `int OutputWidth(int width, Padding2D padding)` | Computes the output width for the specified input width and padding. / 指定した入力の幅とパディングに対する出力の幅を計算します。 |

### `class Convolution2DStream<T>`

Performs causal 2D convolution incrementally, consuming one height row at a time. / 高さ方向に 1 行ずつ入力を受け取り、因果的な 2 次元畳み込みを逐次実行します。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `Convolution2DStream(Convolution2D<T> convolution, int width, int left, int right)` | Initializes a new `Convolution2DStream` for the specified convolution and input row width. / 指定した畳み込み層と入力行の幅から `Convolution2DStream` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `Convolution2D<T> Convolution { get; }` | Gets the underlying 2D convolution layer. / 基になる 2 次元畳み込み層を取得します。 |
| `int InputWidth { get; }` | Gets the expected width of each input row. / 入力の各行の幅を取得します。 |
| `int OutputWidth { get; }` | Gets the width of each output row. / 出力の各行の幅を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `Matrix<T> Push(ReadOnlyMatrixSpan<T> row)` | Processes one input row and returns the corresponding output row as a new matrix. / 入力の 1 行を処理し、対応する出力の 1 行を新しい行列として返します。 |
| `MatrixSpan<T> Push(ReadOnlyMatrixSpan<T> row, MatrixSpan<T> destination)` | Processes one input row and writes the corresponding output row into `destination`. / 入力の 1 行を処理し、対応する出力の 1 行を `destination` に書き込みます。 |
| `void Reset()` | Clears the buffered input history to zero, resetting the stream to its initial state. / 保持している過去の入力行を 0 でクリアし、初期状態に戻します。 |

### `class MambaBlock<T>`

Implements a Mamba selective state-space block for inference. / 推論用の選択的状態空間モデル（Mamba）ブロックを実装します。

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `MambaBlock(MambaDimensions dimensions, ReadOnlySpan<T> inProjection, ReadOnlySpan<T> convolutionWeight, ReadOnlySpan<T> convolutionBias, ReadOnlySpan<T> xProjection, ReadOnlySpan<T> deltaProjection, ReadOnlySpan<T> deltaProjectionBias, ReadOnlySpan<T> aLog, ReadOnlySpan<T> d, ReadOnlySpan<T> outProjection, ReadOnlySpan<T> inProjectionBias = default, ReadOnlySpan<T> outProjectionBias = default)` | Initializes a new `MambaBlock` from `mamba_ssm` weights. / `mamba_ssm` の重みから `MambaBlock` を初期化します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `MambaDimensions Dimensions { get; }` | Gets the dimensions of this block. / このブロックの次元構成を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `MambaState<T> CreateState(int sequences)` | Creates a zero-initialized state for the specified number of independent sequences. / 指定した系列数に対応するゼロ初期化済みの状態オブジェクトを作成します。 |
| `Matrix<T> Forward(ReadOnlyMatrixSpan<T> u, bool approximate)` | Processes a single sequence from a zero initial state and returns a new matrix. / 初期状態 0 から 1 本の系列を処理し、出力を新しい行列として返します。 |
| `MatrixSpan<T> Forward(ReadOnlyMatrixSpan<T> u, MambaState<T> state, bool approximate, MatrixSpan<T> destination)` | Processes a single sequence, optionally continuing from and updating `state`, and writes the result to `destination`. / 1 本の系列を処理し（必要に応じて `state` を引き継いで更新）、結果を `destination` に書き込みます。 |
| `Matrix<T> Step(ReadOnlyMatrixSpan<T> u, MambaState<T> state, bool approximate)` | Advances multiple independent sequences by one step and returns a new matrix. / 複数の独立した系列を 1 ステップ進め、出力を新しい行列として返します。 |
| `MatrixSpan<T> Step(ReadOnlyMatrixSpan<T> u, MambaState<T> state, bool approximate, MatrixSpan<T> destination)` | Advances multiple independent sequences by one step, updating `state` in place and writing outputs to `destination`. / 複数の独立した系列を 1 ステップ進め、`state` を更新するとともに結果を `destination` に書き込みます。 |

### `readonly struct MambaDimensions`

Specifies the dimensions of a `MambaBlock`. / `MambaBlock` の各次元を表します。

- 継承・実装: `IEquatable<MambaDimensions>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `MambaDimensions(int Model, int Inner, int State, int ConvolutionKernel, int DeltaRank)` | Specifies the dimensions of a `MambaBlock`. / `MambaBlock` の各次元を表します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int ConvolutionKernel { get; set; }` | The causal 1D convolution kernel length (`d_conv`). / 因果1次元畳み込みのカーネル長（`d_conv`）。 |
| `int DeltaRank { get; set; }` | The low-rank projection dimension for step size `Δ` (`dt_rank`). / 刻み幅 `Δ` の低ランク射影次元（`dt_rank`）。 |
| `int Inner { get; set; }` | The expanded inner channel count (`d_inner`; defaults to `2 * d_model` in `mamba_ssm`). / 内部チャネル数（`d_inner`。`mamba_ssm` の既定値は `2 * d_model`）。 |
| `int Model { get; set; }` | The input and output channel count (`d_model`). / 入出力チャネル数（`d_model`）。 |
| `int State { get; set; }` | The state-space dimension `N` (`d_state`). / 状態空間の次元 `N`（`d_state`）。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static MambaDimensions Default(int model)` | Creates the default `mamba_ssm` dimensions for the specified model width. / 指定したモデル次元に対する `mamba_ssm` の既定の次元構成を作成します。 |

### `class MambaState<T>`

Holds the recurrent inference state (causal convolution history and selective-scan state `h`) for one or more sequences in a `MambaBlock`. / `MambaBlock` における複数系列の推論状態（因果畳み込みの過去入力と選択的走査の状態 `h`）を保持します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `MambaDimensions Dimensions { get; }` | Gets the dimensions of the `MambaBlock` for which this state was created. / この状態を作成した `MambaBlock` の次元構成を取得します。 |
| `int Sequences { get; }` | Gets the number of independent sequences tracked by this state. / この状態が保持する独立な系列数を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `void Reset()` | Resets all sequences to their initial zero state. / すべての系列の状態を初期状態（0）にリセットします。 |

### `readonly struct Padding2D`

Specifies the number of zero-padded elements added to each edge of a 2D convolution input. / 2次元畳み込みの入力の各辺に追加するゼロパディング数を表します。

- 継承・実装: `IEquatable<Padding2D>`

**コンストラクタ**

| シグネチャ | 説明 |
| --- | --- |
| `Padding2D(int height, int width)` | Initializes a symmetric `Padding2D` with `height` on top and bottom and `width` on left and right, matching PyTorch `padding=(height, width)`. / 上下に `height`、左右に `width` を設定する対称な `Padding2D` を初期化します（PyTorch の `padding=(height, width)` に対応）。 |
| `Padding2D(int Top, int Bottom, int Left, int Right)` | Specifies the number of zero-padded elements added to each edge of a 2D convolution input. / 2次元畳み込みの入力の各辺に追加するゼロパディング数を表します。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Bottom { get; set; }` | The padding added to the end (bottom) of the height dimension. / 高さ方向の末尾（下）のパディング数。 |
| `int Left { get; set; }` | The padding added to the start (left) of the width dimension. / 幅方向の先頭（左）のパディング数。 |
| `int Right { get; set; }` | The padding added to the end (right) of the width dimension. / 幅方向の末尾（右）のパディング数。 |
| `int Top { get; set; }` | The padding added to the start (top) of the height dimension. / 高さ方向の先頭（上）のパディング数。 |

### `static class XMathDifferentiationExtensions`

Provides neural network layers on automatic differentiation nodes as `XMath` extensions, so that their gradients can be computed. / 自動微分のノードに対するニューラルネットワークの層を、勾配を計算できる形で記録する `XMath` 拡張として提供します。

- 呼び出し形式: `using Tuna.NeuralNetwork;` のうえで `XMath.<Method>(...)` として呼び出します（`using static` では解決されません）。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static MatrixNode<T> CausalDepthwiseConvolve<T>(MatrixNode<T> x, MatrixNode<T> kernel, VectorNode<T> bias = default)` | Records causal depthwise convolution with optional bias. / 省略可能なバイアスを持つ因果的な深さ方向の畳み込みを記録します。 |
| `static TensorNode<T> Convolve2D<T>(TensorNode<T> x, MatrixNode<T> weights, int kernelHeight, int kernelWidth, Padding2D padding, VectorNode<T> bias = default, int strideHeight = 1, int strideWidth = 1, int dilationHeight = 1, int dilationWidth = 1)` | Records a two-dimensional convolution with optional bias. / 省略可能なバイアスを持つ2次元畳み込みを記録します。 |
| `static MatrixNode<T> LayerNorm<T>(MatrixNode<T> x, T epsilon)` | Records unscaled column-wise LayerNorm, `y = (x - μ) / √(σ² + ε)`. / スケール・シフトなしの列ごとの LayerNorm `y = (x - μ) / √(σ² + ε)` を記録します。 |
| `static MatrixNode<T> LayerNorm<T>(MatrixNode<T> x, VectorNode<T> scale, VectorNode<T> shift, T epsilon)` | Records column-wise LayerNorm, `y = (x - μ) / √(σ² + ε) ⊙ scale + shift`. / 列ごとの LayerNorm `y = (x - μ) / √(σ² + ε) ⊙ scale + shift` を記録します。 |
| `static MatrixNode<T> Linear<T>(MatrixNode<T> weight, MatrixNode<T> x)` | Records the linear transformation `y = W * x` without bias. / バイアスなしの全結合 `y = W * x` を記録します。 |
| `static MatrixNode<T> Linear<T>(MatrixNode<T> weight, MatrixNode<T> x, VectorNode<T> bias)` | Records the linear (affine) transformation `y = W * x + b`. / 全結合（アフィン変換）`y = W * x + b` を記録します。 |
| `static MatrixNode<T> Prelu<T>(MatrixNode<T> x, VectorNode<T> slopes)` | Records parametric ReLU (PReLU), `y = x` for `x ≥ 0` and `aᵢ * x` for `x < 0`. / PReLU（`x ≥ 0` で `x`、`x < 0` で `aᵢ * x`）を記録します。 |
| `static MatrixNode<T> RmsNorm<T>(MatrixNode<T> x, T epsilon)` | Records unscaled column-wise RMSNorm, `y = x / √(mean(x²) + ε)`. / スケールなしの列ごとの RMSNorm `y = x / √(mean(x²) + ε)` を記録します。 |
| `static MatrixNode<T> RmsNorm<T>(MatrixNode<T> x, VectorNode<T> scale, T epsilon)` | Records column-wise RMSNorm, `y = x / √(mean(x²) + ε) ⊙ scale`. / 列ごとの RMSNorm `y = x / √(mean(x²) + ε) ⊙ scale` を記録します。 |
| `static MatrixNode<T> SelectiveScan<T>(MatrixNode<T> x, MatrixNode<T> delta, MatrixNode<T> a, MatrixNode<T> b, MatrixNode<T> c, VectorNode<T> d = default, bool approximate = false)` | Records the selective state-space scan from a zero initial state. / 初期状態がゼロの選択的状態空間モデルの走査を記録します。 |

### `static class XMathExtensions`

Provides neural network inference operations such as normalization, activation, linear projection, and state-space operations as `XMath` extensions. / 正規化、活性化関数、全結合、状態空間モデル演算など、ニューラルネットワーク推論向けの演算を `XMath` 拡張として提供します。

- 呼び出し形式: `using Tuna.NeuralNetwork;` のうえで `XMath.<Method>(...)` として呼び出します（`using static` では解決されません）。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static Matrix<T> CausalDepthwiseConvolve<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> kernel)  (+2 overloads)` | Computes channel-wise causal depthwise 1D convolution without bias and returns a new matrix. / バイアスなしのチャネルごとの因果的な depthwise 1次元畳み込みを計算し、新しい行列として返します。 |
| `static Matrix<T> CausalDepthwiseConvolve<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> kernel, ReadOnlyVectorSpan<T> bias)  (+2 overloads)` | Computes channel-wise causal depthwise 1D convolution, `y(d, t) = bias_d + Σ_{k=0}^{K-1} w(d, k) * x(d, t - K + 1 + k)`, and returns a new matrix. / チャネルごとの因果的な depthwise 1次元畳み込み `y(d, t) = bias_d + Σ_{k=0}^{K-1} w(d, k) * x(d, t - K + 1 + k)` を計算し、新しい行列として返します。 |
| `static MatrixSpan<T> CausalDepthwiseConvolve<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> kernel, ReadOnlyVectorSpan<T> bias, MatrixSpan<T> destination)  (+2 overloads)` | Computes channel-wise causal depthwise 1D convolution, `y(d, t) = bias_d + Σ_{k=0}^{K-1} w(d, k) * x(d, t - K + 1 + k)`, into `destination`. / チャネルごとの因果的な depthwise 1次元畳み込み `y(d, t) = bias_d + Σ_{k=0}^{K-1} w(d, k) * x(d, t - K + 1 + k)` を計算し、`destination` に書き込みます。 |
| `static Matrix<T> LayerNorm<T>(ReadOnlyMatrixSpan<T> x, T epsilon)  (+2 overloads)` | Applies unscaled column-wise LayerNorm, `y = (x - μ) / √(σ² + ε)`, and returns a new matrix. / スケール・シフトなしの列ごとの LayerNorm `y = (x - μ) / √(σ² + ε)` を計算し、新しい行列として返します。 |
| `static Matrix<T> LayerNorm<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyVectorSpan<T> scale, ReadOnlyVectorSpan<T> shift, T epsilon)  (+2 overloads)` | Applies column-wise LayerNorm, `y = (x - μ) / √(σ² + ε) ⊙ scale + shift`, and returns a new matrix. / 列ごとの LayerNorm `y = (x - μ) / √(σ² + ε) ⊙ scale + shift` を計算し、新しい行列として返します。 |
| `static MatrixSpan<T> LayerNorm<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyVectorSpan<T> scale, ReadOnlyVectorSpan<T> shift, T epsilon, MatrixSpan<T> destination)  (+2 overloads)` | Applies column-wise LayerNorm, `y = (x - μ) / √(σ² + ε) ⊙ scale + shift`, into `destination`. / 列ごとの LayerNorm `y = (x - μ) / √(σ² + ε) ⊙ scale + shift` を計算し、`destination` に書き込みます。 |
| `static Matrix<T> Linear<T>(ReadOnlyMatrixSpan<T> weight, ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Computes the linear transformation `y = W * x` without bias and returns a new matrix. / バイアスなしの全結合 `y = W * x` を計算し、新しい行列として返します。 |
| `static Matrix<T> Linear<T>(ReadOnlyMatrixSpan<T> weight, ReadOnlyMatrixSpan<T> x, ReadOnlyVectorSpan<T> bias)  (+2 overloads)` | Computes the linear (affine) transformation `y = W * x + b` and returns a new matrix. / 全結合（アフィン変換）`y = W * x + b` を計算し、新しい行列として返します。 |
| `static MatrixSpan<T> Linear<T>(ReadOnlyMatrixSpan<T> weight, ReadOnlyMatrixSpan<T> x, ReadOnlyVectorSpan<T> bias, MatrixSpan<T> destination)  (+2 overloads)` | Computes the linear (affine) transformation `y = W * x + b` into `destination`. / 全結合（アフィン変換）`y = W * x + b` を計算し、`destination` に書き込みます。 |
| `static Matrix<T> Prelu<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyVectorSpan<T> slopes)  (+2 overloads)` | Applies parametric ReLU (PReLU), `y = x` for `x ≥ 0` and `aᵢ * x` for `x < 0`, and returns a new matrix. / PReLU（`x ≥ 0` で `x`、`x < 0` で `aᵢ * x`）を適用し、新しい行列として返します。 |
| `static MatrixSpan<T> Prelu<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyVectorSpan<T> slopes, MatrixSpan<T> destination)  (+2 overloads)` | Applies parametric ReLU (PReLU), `y = x` for `x ≥ 0` and `aᵢ * x` for `x < 0`, into `destination`. / PReLU（`x ≥ 0` で `x`、`x < 0` で `aᵢ * x`）を適用し、`destination` に書き込みます。 |
| `static Matrix<T> RmsNorm<T>(ReadOnlyMatrixSpan<T> x, T epsilon)  (+2 overloads)` | Applies unscaled column-wise RMSNorm, `y = x / √(mean(x²) + ε)`, and returns a new matrix. / スケールなしの列ごとの RMSNorm `y = x / √(mean(x²) + ε)` を計算し、新しい行列として返します。 |
| `static Matrix<T> RmsNorm<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyVectorSpan<T> scale, T epsilon)  (+2 overloads)` | Applies column-wise RMSNorm, `y = x / √(mean(x²) + ε) ⊙ scale`, and returns a new matrix. / 列ごとの RMSNorm `y = x / √(mean(x²) + ε) ⊙ scale` を計算し、新しい行列として返します。 |
| `static MatrixSpan<T> RmsNorm<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyVectorSpan<T> scale, T epsilon, MatrixSpan<T> destination)  (+2 overloads)` | Applies column-wise RMSNorm, `y = x / √(mean(x²) + ε) ⊙ scale`, into `destination`. / 列ごとの RMSNorm `y = x / √(mean(x²) + ε) ⊙ scale` を計算し、`destination` に書き込みます。 |
| `static Matrix<T> SelectiveScan<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> delta, ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b, ReadOnlyMatrixSpan<T> c, ReadOnlyVectorSpan<T> d, bool approximate)  (+2 overloads)` | Runs the Mamba S6 selective state-space scan from a zero initial state and returns a new matrix. / 初期状態 0 から Mamba の選択的状態空間モデル（S6）を走査し、新しい行列として返します。 |
| `static MatrixSpan<T> SelectiveScan<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> delta, ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b, ReadOnlyMatrixSpan<T> c, ReadOnlyVectorSpan<T> d, MatrixSpan<T> state, bool approximate, MatrixSpan<T> destination)  (+2 overloads)` | Runs the selective state-space scan from `state`, updating `state` in place and writing outputs to `destination`. / 指定した `state` から選択的状態空間モデルの走査を実行し、`state` を更新するとともに結果を `destination` に書き込みます。 |
| `static MatrixSpan<T> SelectiveScanStep<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> delta, ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b, ReadOnlyMatrixSpan<T> c, ReadOnlyVectorSpan<T> d, TensorSpan<T> state, bool approximate, MatrixSpan<T> destination)  (+2 overloads)` | Advances multiple independent sequences by a single step in the selective state-space model. / 選択的状態空間モデルで複数の独立した系列を 1 ステップ進めます。 |

### `static class XMathInPlaceExtensions`

Provides in-place normalization and PReLU operations as `InPlace` extensions. / 特徴量行列をインプレースで更新する正規化および PReLU を `InPlace` 拡張として提供します。

- 呼び出し形式: `using Tuna.NeuralNetwork;` のうえで `XMath.InPlace.<Method>(...)` として呼び出します（`using static` では解決されません）。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static void LayerNorm<T>(MatrixSpan<T> x, ReadOnlyVectorSpan<T> scale, ReadOnlyVectorSpan<T> shift, T epsilon)  (+2 overloads)` | Applies column-wise LayerNorm, `y = (x - μ) / √(σ² + ε) ⊙ scale + shift`, in place. / 列ごとの LayerNorm `y = (x - μ) / √(σ² + ε) ⊙ scale + shift` をインプレースで適用します。 |
| `static void Prelu<T>(MatrixSpan<T> x, ReadOnlyVectorSpan<T> slopes)  (+2 overloads)` | Applies parametric ReLU (PReLU), `y = x` for `x ≥ 0` and `aᵢ * x` for `x < 0`, in place. / PReLU（`x ≥ 0` で `x`、`x < 0` で `aᵢ * x`）をインプレースで適用します。 |
| `static void RmsNorm<T>(MatrixSpan<T> x, ReadOnlyVectorSpan<T> scale, T epsilon)  (+2 overloads)` | Applies column-wise RMSNorm, `y = x / √(mean(x²) + ε) ⊙ scale`, in place. / 列ごとの RMSNorm `y = x / √(mean(x²) + ε) ⊙ scale` をインプレースで適用します。 |

## Tuna.NeuralNetwork.IO

### `enum SafeTensorDataType`

Specifies the stored element data type of a tensor in a safetensors file. / safetensors ファイルに保存されているテンソルの要素型を表します。

- 継承・実装: `IComparable`, `IConvertible`, `IFormattable`, `ISpanFormattable`

| 値 | 説明 |
| --- | --- |
| `Boolean` | 1-byte boolean (`BOOL`). / 1 バイトの真偽値（`BOOL`）。 |
| `Float4` | 4-bit floating-point (`F4`). / 4 ビット浮動小数点数（`F4`）。 |
| `Float6E2M3` | 6-bit floating-point with 2 exponent bits and 3 mantissa bits (`F6_E2M3`). / 6 ビット浮動小数点数（指数 2 ビット、仮数 3 ビット、`F6_E2M3`）。 |
| `Float6E3M2` | 6-bit floating-point with 3 exponent bits and 2 mantissa bits (`F6_E3M2`). / 6 ビット浮動小数点数（指数 3 ビット、仮数 2 ビット、`F6_E3M2`）。 |
| `UInt8` | Unsigned 8-bit integer (`U8`). / 符号なし 8 ビット整数（`U8`）。 |
| `Int8` | Signed 8-bit integer (`I8`). / 符号付き 8 ビット整数（`I8`）。 |
| `Float8E5M2` | 8-bit floating-point in E5M2 format (`F8_E5M2`). / 8 ビット浮動小数点数（E5M2 形式、`F8_E5M2`）。 |
| `Float8E4M3` | 8-bit floating-point in E4M3 format (`F8_E4M3`). / 8 ビット浮動小数点数（E4M3 形式、`F8_E4M3`）。 |
| `Float8E8M0` | 8-bit exponent-only scale in E8M0 format (`F8_E8M0`). / 8 ビットスケール値（E8M0 形式、`F8_E8M0`）。 |
| `Float8E4M3Fnuz` | 8-bit floating-point in E4M3FNUZ format (`F8_E4M3FNUZ`). / 8 ビット浮動小数点数（E4M3FNUZ 形式、`F8_E4M3FNUZ`）。 |
| `Float8E5M2Fnuz` | 8-bit floating-point in E5M2FNUZ format (`F8_E5M2FNUZ`). / 8 ビット浮動小数点数（E5M2FNUZ 形式、`F8_E5M2FNUZ`）。 |
| `Int16` | Signed 16-bit integer (`I16`). / 符号付き 16 ビット整数（`I16`）。 |
| `UInt16` | Unsigned 16-bit integer (`U16`). / 符号なし 16 ビット整数（`U16`）。 |
| `Float16` | IEEE 754 binary16 floating-point (`F16`). / IEEE 754 binary16 浮動小数点数（`F16`）。 |
| `BFloat16` | 16-bit brain floating-point (`BF16`). / bfloat16 浮動小数点数（`BF16`）。 |
| `Int32` | Signed 32-bit integer (`I32`). / 符号付き 32 ビット整数（`I32`）。 |
| `UInt32` | Unsigned 32-bit integer (`U32`). / 符号なし 32 ビット整数（`U32`）。 |
| `Float32` | IEEE 754 binary32 floating-point (`F32`). / IEEE 754 binary32 浮動小数点数（`F32`）。 |
| `Complex64` | 64-bit complex number composed of two IEEE 754 binary32 components (`C64`). / 実部と虚部がそれぞれ IEEE 754 binary32 の複素数（`C64`）。 |
| `Float64` | IEEE 754 binary64 floating-point (`F64`). / IEEE 754 binary64 浮動小数点数（`F64`）。 |
| `Int64` | Signed 64-bit integer (`I64`). / 符号付き 64 ビット整数（`I64`）。 |
| `UInt64` | Unsigned 64-bit integer (`U64`). / 符号なし 64 ビット整数（`U64`）。 |

### `class SafeTensorFile`

Reads tensors and metadata from a safetensors file or stream. / safetensors ファイルまたはストリームからテンソルおよびメタデータを読み取ります。

- 継承・実装: `IDisposable`

**フィールド・定数**

| メンバー | 説明 |
| --- | --- |
| `static int MaxHeaderLength` | The maximum allowed header size in bytes (100,000,000). / 許容されるヘッダーサイズの最大値（100,000,000 バイト）。 |

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `IReadOnlyDictionary<string, string> Metadata { get; }` | Gets the file-level metadata from the `__metadata__` header section. / ヘッダーの `__metadata__` セクションに含まれるメタデータを取得します。 |
| `IReadOnlyDictionary<string, SafeTensorInfo> Tensors { get; }` | Gets the validated tensor entries contained in the file. / ファイルに含まれる検証済みテンソル情報の一覧を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `void CopyTo(string name, Stream destination)` | Copies the raw little-endian bytes of a tensor to the current position of `destination`. / テンソルの生バイト列（リトルエンディアン）を `destination` の現在位置へコピーします。 |
| `void Dispose()` | Releases the underlying stream unless opened with `leaveOpen` set to `true`. / 入力ストリームを解放します（`leaveOpen` に `true` を指定して開いた場合を除く）。 |
| `SafeTensorInfo GetTensorInfo(string name)` | Retrieves the metadata entry for the tensor with the specified name. / 指定した名前のテンソル情報を取得します。 |
| `static SafeTensorFile Open(string path)` | Opens a safetensors file at the specified path and validates its header and data layout. / 指定したパスの safetensors ファイルを開き、ヘッダーとデータ配置を検証します。 |
| `static SafeTensorFile Open(Stream stream, bool leaveOpen = false)` | Opens a safetensors file from a readable, seekable stream and validates its header and data layout. / 読み取りおよびシーク可能なストリームから safetensors ファイルを開き、ヘッダーとデータ配置を検証します。 |
| `T[] Read<T>(string name)` | Reads a tensor into a new row-major array without type conversion. / 型変換を行わずに、保存時の要素型でテンソルを新しい行優先配列として読み取ります。 |
| `byte[] ReadBytes(string name)` | Reads the raw little-endian bytes of a tensor into a new array. / テンソルの生バイト列（リトルエンディアン）を新しい配列として読み取ります。 |
| `double[] ReadDouble(string name)` | Reads a floating-point tensor (`F64`, `F32`, `F16`, or `BF16`) into a new `Double` array. / 浮動小数点テンソル（`F64`、`F32`、`F16`、または `BF16`）を倍精度浮動小数点数の新しい配列として読み取ります。 |
| `T[] ReadFloating<T>(string name)` | Reads a floating-point tensor into a new array of the specified floating-point type, converting from narrower formats as needed. / 必要に応じて低精度形式から変換しながら、テンソルを指定した浮動小数点型の新しい配列として読み取ります。 |
| `float[] ReadSingle(string name)` | Reads a floating-point tensor (`F32`, `F16`, or `BF16`) into a new `Single` array. / 浮動小数点テンソル（`F32`、`F16`、または `BF16`）を単精度浮動小数点数の新しい配列として読み取ります。 |

### `class SafeTensorInfo`

Describes a validated tensor entry in a safetensors file. / safetensors ファイル内の検証済みテンソル情報を表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `long ByteLength { get; }` | Gets the stored byte length of the tensor data. / テンソルデータの保存バイト数を取得します。 |
| `long DataOffset { get; }` | Gets the byte offset from the start of the file's data region. / ファイルのデータ領域先頭からのバイトオフセットを取得します。 |
| `SafeTensorDataType DataType { get; }` | Gets the stored element data type. / 保存されている要素型を取得します。 |
| `ulong ElementCount { get; }` | Gets the total number of elements in the tensor. / テンソルの総要素数を取得します。 |
| `string Name { get; }` | Gets the tensor name. / テンソル名を取得します。 |
| `IReadOnlyList<ulong> Shape { get; }` | Gets the dimension lengths of the tensor. / テンソルの各軸の長さを取得します。 |

