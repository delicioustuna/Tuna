# Tuna.SignalProcessing API リファレンス

> `(+N overloads)` は `Vector` / `VectorSpan` / `ReadOnlyVectorSpan` 等のコンテナ・ビュー組み合わせや `double` / `float` 固定型オーバーロードの省略を表します。

## 公開型一覧

- `Tuna.SignalProcessing` (17 型): [`ContinuousWavelet`](#abstract-class-continuouswavelet), [`ConvolutionMethod`](#enum-convolutionmethod), [`ConvolutionMode`](#enum-convolutionmode), [`DiscreteWavelet`](#class-discretewavelet), [`Fourier`](#static-class-fourier), [`FourierNormalization`](#enum-fouriernormalization), [`FourierPlan`](#class-fourierplan), [`RealFourierPlan`](#class-realfourierplan), [`WaveletBoundary`](#enum-waveletboundary), [`WaveletDecomposition`](#class-waveletdecomposition), [`WaveletDecomposition2D`](#class-waveletdecomposition2d), [`WaveletPacketTree`](#class-waveletpackettree), [`WaveletPadding`](#enum-waveletpadding), [`WaveletShrinkage`](#enum-waveletshrinkage), [`WaveletThresholdRule`](#enum-waveletthresholdrule), [`Wavelets`](#static-class-wavelets), [`XMathExtensions`](#static-class-xmathextensions)

## Tuna.SignalProcessing

### `abstract class ContinuousWavelet`

Represents a mother wavelet basis (`ψ₀` and its Fourier transform `ψ̂₀`) for continuous wavelet transforms. / 連続ウェーブレット変換の基底（母ウェーブレット `ψ₀` とそのフーリエ変換 `ψ̂₀`）を表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `double EFoldingTime { get; }` | Gets the e-folding time ratio `τ_s / s` at which edge effects decay by a factor of `e^(-2)`. / 境界効果のパワーが `e^(−2)` に減衰する e-folding 時間とスケールの比 `τ_s / s` を取得します。 |
| `double FourierFactor { get; }` | Gets the scale-to-Fourier-period conversion factor (`period = FourierFactor · s`). / スケールからフーリエ周期への換算係数（`周期 = FourierFactor · s`）を取得します。 |
| `static ContinuousWavelet MexicanHat { get; }` | Gets the Mexican hat (Ricker) wavelet, equivalent to `DerivativeOfGaussian` of order 2. / Mexican hat（Ricker）ウェーブレット（2 次 `DerivativeOfGaussian` と同一）を取得します。 |
| `string Name { get; }` | Gets the wavelet name (for example, `Morlet(6)`, `Paul(4)`, or `DOG(2)`). / ウェーブレット名（`Morlet(6)`、`Paul(4)`、`DOG(2)` など）を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static ContinuousWavelet DerivativeOfGaussian(int order)` | Returns the real-valued derivative-of-Gaussian (DOG) wavelet of the specified order. / 指定した階数のガウス関数微分（DOG）実数ウェーブレットを返します。 |
| `Complex Evaluate(double time)` | Evaluates the mother wavelet `ψ₀(η)` at the dimensionless time `time`. / 無次元時刻 `time` における母ウェーブレット `ψ₀(η)` を評価します。 |
| `Complex FourierTransform(double angularFrequency)` | Evaluates the Fourier transform `ψ̂₀(ω)` at the dimensionless scaled angular frequency `angularFrequency`. / 無次元角周波数 `angularFrequency` におけるフーリエ変換 `ψ̂₀(ω)` を評価します。 |
| `static ContinuousWavelet Morlet()` | Returns the default complex Morlet wavelet with central angular frequency `ω₀ = 6`. / 既定の中心角周波数 `ω₀ = 6` の複素 Morlet ウェーブレットを返します。 |
| `static ContinuousWavelet Morlet(double centralFrequency)` | Returns the complex Morlet wavelet with the specified central angular frequency. / 指定した中心角周波数の複素 Morlet ウェーブレットを返します。 |
| `static ContinuousWavelet Paul()` | Returns the default 4th-order complex Paul wavelet. / 既定の 4 次複素 Paul ウェーブレットを返します。 |
| `static ContinuousWavelet Paul(int order)` | Returns the complex Paul wavelet of the specified order. / 指定した次数の複素 Paul ウェーブレットを返します。 |
| `string ToString()` |  |

### `enum ConvolutionMethod`

Specifies the algorithm used for convolution, cross-correlation, and autocorrelation. / 畳み込み・相互相関・自己相関の計算方式を指定します。

- 継承・実装: `IComparable`, `IConvertible`, `IFormattable`, `ISpanFormattable`

| 値 | 説明 |
| --- | --- |
| `Auto` | Deterministically selects FFT for long `Double` and `Complex` inputs, and direct summation otherwise. / 入力長に応じて方式を自動的に選択します。`Double` と `Complex` の長い入力では FFT、それ以外では直接計算を使用します。 |
| `Direct` | Computes the sum directly in O(M·N) time, preserving exact arithmetic for integer types and relative accuracy on small tails. / 定義どおりに直接和を計算します（O(M·N)）。整数型では厳密な結果が得られ、裾の小さな値でも相対精度を保ちます。 |
| `Fourier` | Computes via zero-padded FFT in O((M + N) log(M + N)) time. Supported only for `Double` and `Complex`. / ゼロパディングした FFT で計算します（O((M + N) log(M + N))）。`Double` と `Complex` のみ対応します。 |

### `enum ConvolutionMode`

Specifies the output length mode for convolution and cross-correlation. / 畳み込みおよび相互相関の出力範囲を指定します。

- 継承・実装: `IComparable`, `IConvertible`, `IFormattable`, `ISpanFormattable`

| 値 | 説明 |
| --- | --- |
| `Full` | All overlapping positions, producing length `M + N - 1`. / 重なりのあるすべての位置を出力します（長さ `M + N − 1`）。 |
| `Same` | The central portion of `Full` with length `max(M, N)`. / `Full` の中央部分を長い方の入力と同じ長さ `max(M, N)` で出力します。 |
| `Valid` | Only positions where the shorter input completely overlaps the longer input, producing length `max(M, N) - min(M, N) + 1`. / 短い方の入力が長い方に完全に重なる位置だけを出力します（長さ `max(M, N) − min(M, N) + 1`）。 |

### `class DiscreteWavelet`

Represents a discrete wavelet filter bank consisting of decomposition and reconstruction low-pass and high-pass filters. / 分解および再構成の低域・高域フィルタ組から成る離散ウェーブレット基底を表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `ReadOnlySpan<double> DecompositionHigh { get; }` | Gets the decomposition high-pass filter coefficients. / 分解用の高域フィルタ係数を取得します。 |
| `ReadOnlySpan<double> DecompositionLow { get; }` | Gets the decomposition low-pass filter coefficients. / 分解用の低域フィルタ係数を取得します。 |
| `int FilterLength { get; }` | Gets the filter length shared by all four filters. / 4 つのフィルタに共通のフィルタ長を取得します。 |
| `static DiscreteWavelet Haar { get; }` | Gets the Haar wavelet (identical filter coefficients to `db1`). / Haar ウェーブレット（`db1` と同一のフィルタ係数）を取得します。 |
| `bool IsOrthogonal { get; }` | Gets a value indicating whether this wavelet is orthogonal (`true` for Haar, Daubechies, Symlet, and Coiflet; `false` for biorthogonal families). / 直交ウェーブレットかどうかを取得します（Haar・Daubechies・Symlet・Coiflet は `true`、双直交系は `false`）。 |
| `string Name { get; }` | Gets the canonical wavelet name (such as `haar`, `db4`, `sym8`, `coif3`, `bior4.4`, or `rbio2.2`). / ウェーブレットの標準名（`haar`、`db4`、`sym8`、`coif3`、`bior4.4`、`rbio2.2` など）を取得します。 |
| `ReadOnlySpan<double> ReconstructionHigh { get; }` | Gets the reconstruction high-pass filter coefficients. / 再構成用の高域フィルタ係数を取得します。 |
| `ReadOnlySpan<double> ReconstructionLow { get; }` | Gets the reconstruction low-pass filter coefficients. / 再構成用の低域フィルタ係数を取得します。 |
| `int VanishingMoments { get; }` | Gets the number of vanishing moments of the decomposition wavelet. / 分解側ウェーブレットの消失モーメント数を取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static DiscreteWavelet Biorthogonal(int reconstructionOrder, int decompositionOrder)` | Returns the Cohen–Daubechies–Feauveau biorthogonal wavelet for the specified order pair. / 指定した次数の組に対応する双直交ウェーブレット（Cohen–Daubechies–Feauveau 系）を返します。 |
| `static DiscreteWavelet Coiflet(int order)` | Returns the Coiflet wavelet of the specified order. / 指定した次数の Coiflet ウェーブレットを返します。 |
| `static DiscreteWavelet Daubechies(int order)` | Returns the Daubechies extremal-phase wavelet of the specified order. / 指定した次数の Daubechies ウェーブレットを返します。 |
| `static DiscreteWavelet FromName(string name)` | Looks up a built-in discrete wavelet by name (case-insensitive, ignoring surrounding whitespace). / 名前から組み込みの離散ウェーブレットを取得します（前後の空白および大文字・小文字を区別しません）。 |
| `static DiscreteWavelet ReverseBiorthogonal(int reconstructionOrder, int decompositionOrder)` | Returns the reverse biorthogonal wavelet obtained by swapping the decomposition and reconstruction filters of `Biorthogonal`. / `Biorthogonal` の分解フィルタと再構成フィルタを入れ替えた逆双直交ウェーブレットを返します。 |
| `static DiscreteWavelet Symlet(int order)` | Returns the Symlet (least-asymmetric Daubechies) wavelet of the specified order. / 指定した次数の Symlet ウェーブレットを返します。 |
| `string ToString()` |  |

### `static class Fourier`

Provides discrete Fourier transforms, discrete cosine and sine transforms, chirp-z transforms, and spectral utilities. / 離散フーリエ変換、離散コサイン・サイン変換、チャープ z 変換、およびスペクトル関連ユーティリティを提供します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static Vector<Complex> ChirpZ(ReadOnlyVectorSpan<Complex> x, int count, Complex ratio, Complex start)` | Evaluates the chirp-z transform of a complex signal along a logarithmic spiral contour `z_k = start · ratio^(-k)`. / 複素平面上の対数らせん点列 `z_k = start · ratio^(−k)` に沿ってチャープ z 変換を評価します。 |
| `static VectorSpan<Complex> ChirpZ(ReadOnlyVectorSpan<Complex> x, Complex ratio, Complex start, VectorSpan<Complex> destination)` | Evaluates the chirp-z transform along `z_k = start · ratio^(-k)` and writes the result to `destination`. / 複素平面上の点列 `z_k = start · ratio^(−k)` に沿ってチャープ z 変換を評価し、`destination` に書き込みます。 |
| `static Vector<double> Cosine(ReadOnlyVectorSpan<double> x, int type)` | Computes the discrete cosine transform (DCT-I through DCT-IV) of a real vector using `Backward`. / 実数ベクトルの離散コサイン変換（DCT-I〜IV）を `Backward` で計算します。 |
| `static Vector<double> Cosine(ReadOnlyVectorSpan<double> x, int type, FourierNormalization normalization)` | Computes the discrete cosine transform (DCT-I through DCT-IV) of a real vector with the specified normalization. / 指定した正規化方式で実数ベクトルの離散コサイン変換（DCT-I〜IV）を計算します。 |
| `static VectorSpan<double> Cosine(ReadOnlyVectorSpan<double> x, int type, VectorSpan<double> destination)` | Computes the discrete cosine transform using `Backward` and writes the result to `destination`. / 実数ベクトルの離散コサイン変換を `Backward` で計算し、`destination` に書き込みます。 |
| `static VectorSpan<double> Cosine(ReadOnlyVectorSpan<double> x, int type, FourierNormalization normalization, VectorSpan<double> destination)` | Computes the discrete cosine transform with the specified normalization and writes the result to `destination`. / 指定した正規化方式で実数ベクトルの離散コサイン変換を計算し、`destination` に書き込みます。 |
| `static Matrix<double> CosineColumns(ReadOnlyMatrixSpan<double> x, int type)` | Computes the discrete cosine transform independently for each column of a matrix using `Backward`. / 実数行列の各列ごとに離散コサイン変換を `Backward` で計算します。 |
| `static Matrix<double> CosineColumns(ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization)` | Computes the discrete cosine transform independently for each column of a matrix with the specified normalization. / 指定した正規化方式で実数行列の各列ごとに離散コサイン変換を計算します。 |
| `static MatrixSpan<double> CosineColumns(ReadOnlyMatrixSpan<double> x, int type, MatrixSpan<double> destination)` | Computes the discrete cosine transform independently for each column using `Backward` and writes the result to `destination`. / 実数行列の各列ごとに離散コサイン変換を `Backward` で計算し、`destination` に書き込みます。 |
| `static MatrixSpan<double> CosineColumns(ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization, MatrixSpan<double> destination)` | Computes the discrete cosine transform independently for each column with the specified normalization and writes the result to `destination`. / 指定した正規化方式で実数行列の各列ごとに離散コサイン変換を計算し、`destination` に書き込みます。 |
| `static Matrix<double> CosineRows(ReadOnlyMatrixSpan<double> x, int type)` | Computes the discrete cosine transform independently for each row of a matrix using `Backward`. / 実数行列の各行ごとに離散コサイン変換を `Backward` で計算します。 |
| `static Matrix<double> CosineRows(ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization)` | Computes the discrete cosine transform independently for each row of a matrix with the specified normalization. / 指定した正規化方式で実数行列の各行ごとに離散コサイン変換を計算します。 |
| `static MatrixSpan<double> CosineRows(ReadOnlyMatrixSpan<double> x, int type, MatrixSpan<double> destination)` | Computes the discrete cosine transform independently for each row using `Backward` and writes the result to `destination`. / 実数行列の各行ごとに離散コサイン変換を `Backward` で計算し、`destination` に書き込みます。 |
| `static MatrixSpan<double> CosineRows(ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization, MatrixSpan<double> destination)` | Computes the discrete cosine transform independently for each row with the specified normalization and writes the result to `destination`. / 指定した正規化方式で実数行列の各行ごとに離散コサイン変換を計算し、`destination` に書き込みます。 |
| `static Vector<Complex> Forward(ReadOnlyVectorSpan<Complex> x)` | Computes the forward discrete Fourier transform of a complex vector using `Backward`. / 複素数ベクトルの順方向離散フーリエ変換を `Backward` で計算します。 |
| `static Vector<Complex> Forward(ReadOnlyVectorSpan<Complex> x, FourierNormalization normalization)` | Computes the forward discrete Fourier transform of a complex vector with the specified normalization. / 指定した正規化方式で複素数ベクトルの順方向離散フーリエ変換を計算します。 |
| `static VectorSpan<Complex> Forward(ReadOnlyVectorSpan<Complex> x, VectorSpan<Complex> destination)` | Computes the forward discrete Fourier transform of a complex vector using `Backward` and writes the result to `destination`. / 複素数ベクトルの順方向離散フーリエ変換を `Backward` で計算し、`destination` に書き込みます。 |
| `static VectorSpan<Complex> Forward(ReadOnlyVectorSpan<Complex> x, FourierNormalization normalization, VectorSpan<Complex> destination)` | Computes the forward discrete Fourier transform of a complex vector with the specified normalization and writes the result to `destination`. / 指定した正規化方式で複素数ベクトルの順方向離散フーリエ変換を計算し、`destination` に書き込みます。 |
| `static Matrix<Complex> Forward2D(ReadOnlyMatrixSpan<Complex> x)` | Computes the two-dimensional forward discrete Fourier transform of a complex matrix using `Backward`. / 複素数行列の 2 次元順方向離散フーリエ変換を `Backward` で計算します。 |
| `static Matrix<Complex> Forward2D(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization)` | Computes the two-dimensional forward discrete Fourier transform of a complex matrix with the specified normalization. / 指定した正規化方式で複素数行列の 2 次元順方向離散フーリエ変換を計算します。 |
| `static MatrixSpan<Complex> Forward2D(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination)` | Computes the two-dimensional forward discrete Fourier transform using `Backward` and writes the result to `destination`. / 複素数行列の 2 次元順方向離散フーリエ変換を `Backward` で計算し、`destination` に書き込みます。 |
| `static MatrixSpan<Complex> Forward2D(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<Complex> destination)` | Computes the two-dimensional forward discrete Fourier transform with the specified normalization and writes the result to `destination`. / 指定した正規化方式で複素数行列の 2 次元順方向離散フーリエ変換を計算し、`destination` に書き込みます。 |
| `static Matrix<Complex> ForwardColumns(ReadOnlyMatrixSpan<Complex> x)` | Computes the forward discrete Fourier transform independently for each column of a complex matrix using `Backward`. / 複素数行列の各列ごとに順方向離散フーリエ変換を `Backward` で計算します。 |
| `static Matrix<Complex> ForwardColumns(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization)` | Computes the forward discrete Fourier transform independently for each column of a complex matrix with the specified normalization. / 指定した正規化方式で複素数行列の各列ごとに順方向離散フーリエ変換を計算します。 |
| `static MatrixSpan<Complex> ForwardColumns(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination)` | Computes the forward discrete Fourier transform independently for each column of a complex matrix using `Backward` and writes the result to `destination`. / 複素数行列の各列ごとに順方向離散フーリエ変換を `Backward` で計算し、`destination` に書き込みます。 |
| `static MatrixSpan<Complex> ForwardColumns(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<Complex> destination)` | Computes the forward discrete Fourier transform independently for each column of a complex matrix with the specified normalization and writes the result to `destination`. / 指定した正規化方式で複素数行列の各列ごとに順方向離散フーリエ変換を計算し、`destination` に書き込みます。 |
| `static Vector<Complex> ForwardReal(ReadOnlyVectorSpan<double> x)` | Computes the one-sided forward real discrete Fourier transform of a real vector using `Backward`. / 実数ベクトルの片側スペクトルを `Backward` で計算します。 |
| `static Vector<Complex> ForwardReal(ReadOnlyVectorSpan<double> x, FourierNormalization normalization)` | Computes the one-sided forward real discrete Fourier transform of a real vector with the specified normalization. / 指定した正規化方式で実数ベクトルの片側スペクトルを計算します。 |
| `static VectorSpan<Complex> ForwardReal(ReadOnlyVectorSpan<double> x, VectorSpan<Complex> destination)` | Computes the one-sided forward real discrete Fourier transform using `Backward` and writes the result to `destination`. / 実数ベクトルの片側スペクトルを `Backward` で計算し、`destination` に書き込みます。 |
| `static VectorSpan<Complex> ForwardReal(ReadOnlyVectorSpan<double> x, FourierNormalization normalization, VectorSpan<Complex> destination)` | Computes the one-sided forward real discrete Fourier transform with the specified normalization and writes the result to `destination`. / 指定した正規化方式で実数ベクトルの片側スペクトルを計算し、`destination` に書き込みます。 |
| `static Matrix<Complex> ForwardReal2D(ReadOnlyMatrixSpan<double> x)` | Computes the two-dimensional forward real discrete Fourier transform using `Backward`, returning the one-sided spectrum along the column axis. / 実数行列の 2 次元順方向実数 FFT を `Backward` で計算し、列方向に片側スペクトルを返します。 |
| `static Matrix<Complex> ForwardReal2D(ReadOnlyMatrixSpan<double> x, FourierNormalization normalization)` | Computes the two-dimensional forward real discrete Fourier transform with the specified normalization, returning the one-sided spectrum along the column axis. / 指定した正規化方式で実数行列の 2 次元順方向実数 FFT を計算し、列方向に片側スペクトルを返します。 |
| `static MatrixSpan<Complex> ForwardReal2D(ReadOnlyMatrixSpan<double> x, MatrixSpan<Complex> destination)` | Computes the two-dimensional forward real discrete Fourier transform using `Backward` and writes the result to `destination`. / 実数行列の 2 次元順方向実数 FFT を `Backward` で計算し、`destination` に書き込みます。 |
| `static MatrixSpan<Complex> ForwardReal2D(ReadOnlyMatrixSpan<double> x, FourierNormalization normalization, MatrixSpan<Complex> destination)` | Computes the two-dimensional forward real discrete Fourier transform with the specified normalization and writes the result to `destination`. / 指定した正規化方式で実数行列の 2 次元順方向実数 FFT を計算し、`destination` に書き込みます。 |
| `static Matrix<Complex> ForwardRealColumns(ReadOnlyMatrixSpan<double> x)` | Computes the one-sided forward real discrete Fourier transform independently for each column of a matrix using `Backward`. / 実数行列の各列ごとに片側スペクトルを `Backward` で計算します。 |
| `static Matrix<Complex> ForwardRealColumns(ReadOnlyMatrixSpan<double> x, FourierNormalization normalization)` | Computes the one-sided forward real discrete Fourier transform independently for each column of a matrix with the specified normalization. / 指定した正規化方式で実数行列の各列ごとに片側スペクトルを計算します。 |
| `static MatrixSpan<Complex> ForwardRealColumns(ReadOnlyMatrixSpan<double> x, MatrixSpan<Complex> destination)` | Computes the one-sided forward real discrete Fourier transform independently for each column using `Backward` and writes the result to `destination`. / 実数行列の各列ごとに片側スペクトルを `Backward` で計算し、`destination` に書き込みます。 |
| `static MatrixSpan<Complex> ForwardRealColumns(ReadOnlyMatrixSpan<double> x, FourierNormalization normalization, MatrixSpan<Complex> destination)` | Computes the one-sided forward real discrete Fourier transform independently for each column with the specified normalization and writes the result to `destination`. / 指定した正規化方式で実数行列の各列ごとに片側スペクトルを計算し、`destination` に書き込みます。 |
| `static Matrix<Complex> ForwardRealRows(ReadOnlyMatrixSpan<double> x)` | Computes the one-sided forward real discrete Fourier transform independently for each row of a matrix using `Backward`. / 実数行列の各行ごとに片側スペクトルを `Backward` で計算します。 |
| `static Matrix<Complex> ForwardRealRows(ReadOnlyMatrixSpan<double> x, FourierNormalization normalization)` | Computes the one-sided forward real discrete Fourier transform independently for each row of a matrix with the specified normalization. / 指定した正規化方式で実数行列の各行ごとに片側スペクトルを計算します。 |
| `static MatrixSpan<Complex> ForwardRealRows(ReadOnlyMatrixSpan<double> x, MatrixSpan<Complex> destination)` | Computes the one-sided forward real discrete Fourier transform independently for each row using `Backward` and writes the result to `destination`. / 実数行列の各行ごとに片側スペクトルを `Backward` で計算し、`destination` に書き込みます。 |
| `static MatrixSpan<Complex> ForwardRealRows(ReadOnlyMatrixSpan<double> x, FourierNormalization normalization, MatrixSpan<Complex> destination)` | Computes the one-sided forward real discrete Fourier transform independently for each row with the specified normalization and writes the result to `destination`. / 指定した正規化方式で実数行列の各行ごとに片側スペクトルを計算し、`destination` に書き込みます。 |
| `static Matrix<Complex> ForwardRows(ReadOnlyMatrixSpan<Complex> x)` | Computes the forward discrete Fourier transform independently for each row of a complex matrix using `Backward`. / 複素数行列の各行ごとに順方向離散フーリエ変換を `Backward` で計算します。 |
| `static Matrix<Complex> ForwardRows(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization)` | Computes the forward discrete Fourier transform independently for each row of a complex matrix with the specified normalization. / 指定した正規化方式で複素数行列の各行ごとに順方向離散フーリエ変換を計算します。 |
| `static MatrixSpan<Complex> ForwardRows(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination)` | Computes the forward discrete Fourier transform independently for each row of a complex matrix using `Backward` and writes the result to `destination`. / 複素数行列の各行ごとに順方向離散フーリエ変換を `Backward` で計算し、`destination` に書き込みます。 |
| `static MatrixSpan<Complex> ForwardRows(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<Complex> destination)` | Computes the forward discrete Fourier transform independently for each row of a complex matrix with the specified normalization and writes the result to `destination`. / 指定した正規化方式で複素数行列の各行ごとに順方向離散フーリエ変換を計算し、`destination` に書き込みます。 |
| `static Vector<double> Frequencies(int length, double sampleSpacing = 1)` | Returns the discrete Fourier transform sample frequencies for a complex transform of length `length`. / 長さ `length` の複素 FFT の各スペクトル係数に対応する周波数を返します。 |
| `static Vector<Complex> Inverse(ReadOnlyVectorSpan<Complex> x)` | Computes the inverse discrete Fourier transform of a complex vector using `Backward`. / 複素数ベクトルの逆離散フーリエ変換を `Backward` で計算します。 |
| `static Vector<Complex> Inverse(ReadOnlyVectorSpan<Complex> x, FourierNormalization normalization)` | Computes the inverse discrete Fourier transform of a complex vector with the specified normalization. / 指定した正規化方式で複素数ベクトルの逆離散フーリエ変換を計算します。 |
| `static VectorSpan<Complex> Inverse(ReadOnlyVectorSpan<Complex> x, VectorSpan<Complex> destination)` | Computes the inverse discrete Fourier transform of a complex vector using `Backward` and writes the result to `destination`. / 複素数ベクトルの逆離散フーリエ変換を `Backward` で計算し、`destination` に書き込みます。 |
| `static VectorSpan<Complex> Inverse(ReadOnlyVectorSpan<Complex> x, FourierNormalization normalization, VectorSpan<Complex> destination)` | Computes the inverse discrete Fourier transform of a complex vector with the specified normalization and writes the result to `destination`. / 指定した正規化方式で複素数ベクトルの逆離散フーリエ変換を計算し、`destination` に書き込みます。 |
| `static Matrix<Complex> Inverse2D(ReadOnlyMatrixSpan<Complex> x)` | Computes the two-dimensional inverse discrete Fourier transform of a complex matrix using `Backward`. / 複素数行列の 2 次元逆離散フーリエ変換を `Backward` で計算します。 |
| `static Matrix<Complex> Inverse2D(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization)` | Computes the two-dimensional inverse discrete Fourier transform of a complex matrix with the specified normalization. / 指定した正規化方式で複素数行列の 2 次元逆離散フーリエ変換を計算します。 |
| `static MatrixSpan<Complex> Inverse2D(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination)` | Computes the two-dimensional inverse discrete Fourier transform using `Backward` and writes the result to `destination`. / 複素数行列の 2 次元逆離散フーリエ変換を `Backward` で計算し、`destination` に書き込みます。 |
| `static MatrixSpan<Complex> Inverse2D(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<Complex> destination)` | Computes the two-dimensional inverse discrete Fourier transform with the specified normalization and writes the result to `destination`. / 指定した正規化方式で複素数行列の 2 次元逆離散フーリエ変換を計算し、`destination` に書き込みます。 |
| `static Matrix<Complex> InverseColumns(ReadOnlyMatrixSpan<Complex> x)` | Computes the inverse discrete Fourier transform independently for each column of a complex matrix using `Backward`. / 複素数行列の各列ごとに逆離散フーリエ変換を `Backward` で計算します。 |
| `static Matrix<Complex> InverseColumns(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization)` | Computes the inverse discrete Fourier transform independently for each column of a complex matrix with the specified normalization. / 指定した正規化方式で複素数行列の各列ごとに逆離散フーリエ変換を計算します。 |
| `static MatrixSpan<Complex> InverseColumns(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination)` | Computes the inverse discrete Fourier transform independently for each column of a complex matrix using `Backward` and writes the result to `destination`. / 複素数行列の各列ごとに逆離散フーリエ変換を `Backward` で計算し、`destination` に書き込みます。 |
| `static MatrixSpan<Complex> InverseColumns(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<Complex> destination)` | Computes the inverse discrete Fourier transform independently for each column of a complex matrix with the specified normalization and writes the result to `destination`. / 指定した正規化方式で複素数行列の各列ごとに逆離散フーリエ変換を計算し、`destination` に書き込みます。 |
| `static Vector<double> InverseCosine(ReadOnlyVectorSpan<double> x, int type)` | Computes the inverse discrete cosine transform (IDCT-I through IDCT-IV) of a real vector using `Backward`. / 実数ベクトルの逆離散コサイン変換（IDCT-I〜IV）を `Backward` で計算します。 |
| `static Vector<double> InverseCosine(ReadOnlyVectorSpan<double> x, int type, FourierNormalization normalization)` | Computes the inverse discrete cosine transform (IDCT-I through IDCT-IV) of a real vector with the specified normalization. / 指定した正規化方式で実数ベクトルの逆離散コサイン変換（IDCT-I〜IV）を計算します。 |
| `static VectorSpan<double> InverseCosine(ReadOnlyVectorSpan<double> x, int type, VectorSpan<double> destination)` | Computes the inverse discrete cosine transform using `Backward` and writes the result to `destination`. / 実数ベクトルの逆離散コサイン変換を `Backward` で計算し、`destination` に書き込みます。 |
| `static VectorSpan<double> InverseCosine(ReadOnlyVectorSpan<double> x, int type, FourierNormalization normalization, VectorSpan<double> destination)` | Computes the inverse discrete cosine transform with the specified normalization and writes the result to `destination`. / 指定した正規化方式で実数ベクトルの逆離散コサイン変換を計算し、`destination` に書き込みます。 |
| `static Matrix<double> InverseCosineColumns(ReadOnlyMatrixSpan<double> x, int type)` | Computes the inverse discrete cosine transform independently for each column of a matrix using `Backward`. / 実数行列の各列ごとに逆離散コサイン変換を `Backward` で計算します。 |
| `static Matrix<double> InverseCosineColumns(ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization)` | Computes the inverse discrete cosine transform independently for each column of a matrix with the specified normalization. / 指定した正規化方式で実数行列の各列ごとに逆離散コサイン変換を計算します。 |
| `static MatrixSpan<double> InverseCosineColumns(ReadOnlyMatrixSpan<double> x, int type, MatrixSpan<double> destination)` | Computes the inverse discrete cosine transform independently for each column using `Backward` and writes the result to `destination`. / 実数行列の各列ごとに逆離散コサイン変換を `Backward` で計算し、`destination` に書き込みます。 |
| `static MatrixSpan<double> InverseCosineColumns(ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization, MatrixSpan<double> destination)` | Computes the inverse discrete cosine transform independently for each column with the specified normalization and writes the result to `destination`. / 指定した正規化方式で実数行列の各列ごとに逆離散コサイン変換を計算し、`destination` に書き込みます。 |
| `static Matrix<double> InverseCosineRows(ReadOnlyMatrixSpan<double> x, int type)` | Computes the inverse discrete cosine transform independently for each row of a matrix using `Backward`. / 実数行列の各行ごとに逆離散コサイン変換を `Backward` で計算します。 |
| `static Matrix<double> InverseCosineRows(ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization)` | Computes the inverse discrete cosine transform independently for each row of a matrix with the specified normalization. / 指定した正規化方式で実数行列の各行ごとに逆離散コサイン変換を計算します。 |
| `static MatrixSpan<double> InverseCosineRows(ReadOnlyMatrixSpan<double> x, int type, MatrixSpan<double> destination)` | Computes the inverse discrete cosine transform independently for each row using `Backward` and writes the result to `destination`. / 実数行列の各行ごとに逆離散コサイン変換を `Backward` で計算し、`destination` に書き込みます。 |
| `static MatrixSpan<double> InverseCosineRows(ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization, MatrixSpan<double> destination)` | Computes the inverse discrete cosine transform independently for each row with the specified normalization and writes the result to `destination`. / 指定した正規化方式で実数行列の各行ごとに逆離散コサイン変換を計算し、`destination` に書き込みます。 |
| `static Vector<double> InverseReal(ReadOnlyVectorSpan<Complex> x, int length)` | Reconstructs a real vector of length `length` from its one-sided spectrum using `Backward`. / 片側スペクトルから長さ `length` の実数ベクトルを `Backward` で逆変換します。 |
| `static VectorSpan<double> InverseReal(ReadOnlyVectorSpan<Complex> x, VectorSpan<double> destination)` | Reconstructs a real vector from its one-sided spectrum using `Backward` and writes the result to `destination`. / 片側スペクトルから実数ベクトルを `Backward` で逆変換し、`destination` に書き込みます。 |
| `static Vector<double> InverseReal(ReadOnlyVectorSpan<Complex> x, int length, FourierNormalization normalization)` | Reconstructs a real vector of length `length` from its one-sided spectrum with the specified normalization. / 指定した正規化方式で片側スペクトルから長さ `length` の実数ベクトルを逆変換します。 |
| `static VectorSpan<double> InverseReal(ReadOnlyVectorSpan<Complex> x, FourierNormalization normalization, VectorSpan<double> destination)` | Reconstructs a real vector from its one-sided spectrum with the specified normalization and writes the result to `destination`. / 指定した正規化方式で片側スペクトルから実数ベクトルを逆変換し、`destination` に書き込みます。 |
| `static Matrix<double> InverseReal2D(ReadOnlyMatrixSpan<Complex> x, int columns)` | Reconstructs a real matrix with `columns` columns from its two-dimensional one-sided spectrum using `Backward`. / 2 次元の片側スペクトルから列数 `columns` の実数行列を `Backward` で逆変換します。 |
| `static MatrixSpan<double> InverseReal2D(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<double> destination)` | Reconstructs a real matrix from its two-dimensional one-sided spectrum using `Backward` and writes the result to `destination`. / 2 次元の片側スペクトルから実数行列を `Backward` で逆変換し、`destination` に書き込みます。 |
| `static Matrix<double> InverseReal2D(ReadOnlyMatrixSpan<Complex> x, int columns, FourierNormalization normalization)` | Reconstructs a real matrix with `columns` columns from its two-dimensional one-sided spectrum with the specified normalization. / 指定した正規化方式で 2 次元の片側スペクトルから列数 `columns` の実数行列を逆変換します。 |
| `static MatrixSpan<double> InverseReal2D(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<double> destination)` | Reconstructs a real matrix from its two-dimensional one-sided spectrum with the specified normalization and writes the result to `destination`. / 指定した正規化方式で 2 次元の片側スペクトルから実数行列を逆変換し、`destination` に書き込みます。 |
| `static Matrix<double> InverseRealColumns(ReadOnlyMatrixSpan<Complex> x, int length)` | Reconstructs real columns of length `length` from their one-sided spectra using `Backward`. / 行列の各列の片側スペクトルから長さ `length` の実数列を `Backward` で逆変換します。 |
| `static MatrixSpan<double> InverseRealColumns(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<double> destination)` | Reconstructs real columns from their one-sided spectra using `Backward` and writes the result to `destination`. / 行列の各列の片側スペクトルから実数列を `Backward` で逆変換し、`destination` に書き込みます。 |
| `static Matrix<double> InverseRealColumns(ReadOnlyMatrixSpan<Complex> x, int length, FourierNormalization normalization)` | Reconstructs real columns of length `length` from their one-sided spectra with the specified normalization. / 指定した正規化方式で行列の各列の片側スペクトルから長さ `length` の実数列を逆変換します。 |
| `static MatrixSpan<double> InverseRealColumns(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<double> destination)` | Reconstructs real columns from their one-sided spectra with the specified normalization and writes the result to `destination`. / 指定した正規化方式で行列の各列の片側スペクトルから実数列を逆変換し、`destination` に書き込みます。 |
| `static Matrix<double> InverseRealRows(ReadOnlyMatrixSpan<Complex> x, int length)` | Reconstructs real rows of length `length` from their one-sided spectra using `Backward`. / 行列の各行の片側スペクトルから長さ `length` の実数列を `Backward` で逆変換します。 |
| `static MatrixSpan<double> InverseRealRows(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<double> destination)` | Reconstructs real rows from their one-sided spectra using `Backward` and writes the result to `destination`. / 行列の各行の片側スペクトルから実数列を `Backward` で逆変換し、`destination` に書き込みます。 |
| `static Matrix<double> InverseRealRows(ReadOnlyMatrixSpan<Complex> x, int length, FourierNormalization normalization)` | Reconstructs real rows of length `length` from their one-sided spectra with the specified normalization. / 指定した正規化方式で行列の各行の片側スペクトルから長さ `length` の実数列を逆変換します。 |
| `static MatrixSpan<double> InverseRealRows(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<double> destination)` | Reconstructs real rows from their one-sided spectra with the specified normalization and writes the result to `destination`. / 指定した正規化方式で行列の各行の片側スペクトルから実数列を逆変換し、`destination` に書き込みます。 |
| `static Matrix<Complex> InverseRows(ReadOnlyMatrixSpan<Complex> x)` | Computes the inverse discrete Fourier transform independently for each row of a complex matrix using `Backward`. / 複素数行列の各行ごとに逆離散フーリエ変換を `Backward` で計算します。 |
| `static Matrix<Complex> InverseRows(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization)` | Computes the inverse discrete Fourier transform independently for each row of a complex matrix with the specified normalization. / 指定した正規化方式で複素数行列の各行ごとに逆離散フーリエ変換を計算します。 |
| `static MatrixSpan<Complex> InverseRows(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination)` | Computes the inverse discrete Fourier transform independently for each row of a complex matrix using `Backward` and writes the result to `destination`. / 複素数行列の各行ごとに逆離散フーリエ変換を `Backward` で計算し、`destination` に書き込みます。 |
| `static MatrixSpan<Complex> InverseRows(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<Complex> destination)` | Computes the inverse discrete Fourier transform independently for each row of a complex matrix with the specified normalization and writes the result to `destination`. / 指定した正規化方式で複素数行列の各行ごとに逆離散フーリエ変換を計算し、`destination` に書き込みます。 |
| `static Vector<T> InverseShift<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Reverses `Shift`, moving the zero-frequency component back to index 0. / `Shift` の逆操作を行い、周波数 0 成分を先頭へ戻します。 |
| `static VectorSpan<T> InverseShift<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Reverses `Shift` and writes the result to `destination`. / 周波数 0 成分を先頭へ戻し、`destination` に書き込みます。 |
| `static Matrix<T> InverseShift2D<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Reverses `Shift2D`, moving the zero-frequency component back to `(0, 0)`. / `Shift2D` の逆操作を行い、周波数 0 成分を先頭 `(0, 0)` へ戻します。 |
| `static MatrixSpan<T> InverseShift2D<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Reverses `Shift2D` and writes the result to `destination`. / 2 次元スペクトルの周波数 0 成分を先頭 `(0, 0)` へ戻し、`destination` に書き込みます。 |
| `static Matrix<T> InverseShiftColumns<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Reverses `ShiftColumns`, moving the zero-frequency component of each column back to row 0. / 行列の各列について周波数 0 成分を先頭行へ戻します。 |
| `static MatrixSpan<T> InverseShiftColumns<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Reverses `ShiftColumns` and writes the result to `destination`. / 行列の各列について周波数 0 成分を先頭行へ戻し、`destination` に書き込みます。 |
| `static Matrix<T> InverseShiftRows<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Reverses `ShiftRows`, moving the zero-frequency component of each row back to column 0. / 行列の各行について周波数 0 成分を先頭列へ戻します。 |
| `static MatrixSpan<T> InverseShiftRows<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Reverses `ShiftRows` and writes the result to `destination`. / 行列の各行について周波数 0 成分を先頭列へ戻し、`destination` に書き込みます。 |
| `static Vector<double> InverseSine(ReadOnlyVectorSpan<double> x, int type)` | Computes the inverse discrete sine transform (IDST-I through IDST-IV) of a real vector using `Backward`. / 実数ベクトルの逆離散サイン変換（IDST-I〜IV）を `Backward` で計算します。 |
| `static Vector<double> InverseSine(ReadOnlyVectorSpan<double> x, int type, FourierNormalization normalization)` | Computes the inverse discrete sine transform (IDST-I through IDST-IV) of a real vector with the specified normalization. / 指定した正規化方式で実数ベクトルの逆離散サイン変換（IDST-I〜IV）を計算します。 |
| `static VectorSpan<double> InverseSine(ReadOnlyVectorSpan<double> x, int type, VectorSpan<double> destination)` | Computes the inverse discrete sine transform using `Backward` and writes the result to `destination`. / 実数ベクトルの逆離散サイン変換を `Backward` で計算し、`destination` に書き込みます。 |
| `static VectorSpan<double> InverseSine(ReadOnlyVectorSpan<double> x, int type, FourierNormalization normalization, VectorSpan<double> destination)` | Computes the inverse discrete sine transform with the specified normalization and writes the result to `destination`. / 指定した正規化方式で実数ベクトルの逆離散サイン変換を計算し、`destination` に書き込みます。 |
| `static Matrix<double> InverseSineColumns(ReadOnlyMatrixSpan<double> x, int type)` | Computes the inverse discrete sine transform independently for each column of a matrix using `Backward`. / 実数行列の各列ごとに逆離散サイン変換を `Backward` で計算します。 |
| `static Matrix<double> InverseSineColumns(ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization)` | Computes the inverse discrete sine transform independently for each column of a matrix with the specified normalization. / 指定した正規化方式で実数行列の各列ごとに逆離散サイン変換を計算します。 |
| `static MatrixSpan<double> InverseSineColumns(ReadOnlyMatrixSpan<double> x, int type, MatrixSpan<double> destination)` | Computes the inverse discrete sine transform independently for each column using `Backward` and writes the result to `destination`. / 実数行列の各列ごとに逆離散サイン変換を `Backward` で計算し、`destination` に書き込みます。 |
| `static MatrixSpan<double> InverseSineColumns(ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization, MatrixSpan<double> destination)` | Computes the inverse discrete sine transform independently for each column with the specified normalization and writes the result to `destination`. / 指定した正規化方式で実数行列の各列ごとに逆離散サイン変換を計算し、`destination` に書き込みます。 |
| `static Matrix<double> InverseSineRows(ReadOnlyMatrixSpan<double> x, int type)` | Computes the inverse discrete sine transform independently for each row of a matrix using `Backward`. / 実数行列の各行ごとに逆離散サイン変換を `Backward` で計算します。 |
| `static Matrix<double> InverseSineRows(ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization)` | Computes the inverse discrete sine transform independently for each row of a matrix with the specified normalization. / 指定した正規化方式で実数行列の各行ごとに逆離散サイン変換を計算します。 |
| `static MatrixSpan<double> InverseSineRows(ReadOnlyMatrixSpan<double> x, int type, MatrixSpan<double> destination)` | Computes the inverse discrete sine transform independently for each row using `Backward` and writes the result to `destination`. / 実数行列の各行ごとに逆離散サイン変換を `Backward` で計算し、`destination` に書き込みます。 |
| `static MatrixSpan<double> InverseSineRows(ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization, MatrixSpan<double> destination)` | Computes the inverse discrete sine transform independently for each row with the specified normalization and writes the result to `destination`. / 指定した正規化方式で実数行列の各行ごとに逆離散サイン変換を計算し、`destination` に書き込みます。 |
| `static int NextFastLength(int length)` | Returns the smallest FFT-friendly length greater than or equal to `length` whose prime factors are limited to 2, 3, 5, and 7. / `length` 以上で素因数が 2・3・5・7 のみから成る最小の FFT 向け長さを返します。 |
| `static Vector<double> RealFrequencies(int length, double sampleSpacing = 1)` | Returns the discrete Fourier transform sample frequencies for the one-sided spectrum of a real transform of length `length`. / 長さ `length` の実数 FFT の片側スペクトルに対応する周波数を返します。 |
| `static Vector<T> Shift<T>(ReadOnlyVectorSpan<T> x)  (+2 overloads)` | Shifts the zero-frequency component of a vector to the center of the spectrum. / ベクトルの周波数 0 成分をスペクトルの中央へ移動します。 |
| `static VectorSpan<T> Shift<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)  (+2 overloads)` | Shifts the zero-frequency component of a vector to the center of the spectrum and writes the result to `destination`. / ベクトルの周波数 0 成分を中央へ移動し、`destination` に書き込みます。 |
| `static Matrix<T> Shift2D<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Shifts the zero-frequency component of a 2D spectrum to the center along both axes. / 2 次元スペクトルの周波数 0 成分を両軸の中央へ移動します。 |
| `static MatrixSpan<T> Shift2D<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Shifts the zero-frequency component of a 2D spectrum to the center along both axes and writes the result to `destination`. / 2 次元スペクトルの周波数 0 成分を両軸の中央へ移動し、`destination` に書き込みます。 |
| `static Matrix<T> ShiftColumns<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Shifts the zero-frequency component of each column to the vertical center. / 行列の各列について周波数 0 成分を中央へ移動します。 |
| `static MatrixSpan<T> ShiftColumns<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Shifts the zero-frequency component of each column to the vertical center and writes the result to `destination`. / 行列の各列について周波数 0 成分を中央へ移動し、`destination` に書き込みます。 |
| `static Matrix<T> ShiftRows<T>(ReadOnlyMatrixSpan<T> x)  (+2 overloads)` | Shifts the zero-frequency component of each row to the horizontal center. / 行列の各行について周波数 0 成分を中央へ移動します。 |
| `static MatrixSpan<T> ShiftRows<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)  (+2 overloads)` | Shifts the zero-frequency component of each row to the horizontal center and writes the result to `destination`. / 行列の各行について周波数 0 成分を中央へ移動し、`destination` に書き込みます。 |
| `static Vector<double> Sine(ReadOnlyVectorSpan<double> x, int type)` | Computes the discrete sine transform (DST-I through DST-IV) of a real vector using `Backward`. / 実数ベクトルの離散サイン変換（DST-I〜IV）を `Backward` で計算します。 |
| `static Vector<double> Sine(ReadOnlyVectorSpan<double> x, int type, FourierNormalization normalization)` | Computes the discrete sine transform (DST-I through DST-IV) of a real vector with the specified normalization. / 指定した正規化方式で実数ベクトルの離散サイン変換（DST-I〜IV）を計算します。 |
| `static VectorSpan<double> Sine(ReadOnlyVectorSpan<double> x, int type, VectorSpan<double> destination)` | Computes the discrete sine transform using `Backward` and writes the result to `destination`. / 実数ベクトルの離散サイン変換を `Backward` で計算し、`destination` に書き込みます。 |
| `static VectorSpan<double> Sine(ReadOnlyVectorSpan<double> x, int type, FourierNormalization normalization, VectorSpan<double> destination)` | Computes the discrete sine transform with the specified normalization and writes the result to `destination`. / 指定した正規化方式で実数ベクトルの離散サイン変換を計算し、`destination` に書き込みます。 |
| `static Matrix<double> SineColumns(ReadOnlyMatrixSpan<double> x, int type)` | Computes the discrete sine transform independently for each column of a matrix using `Backward`. / 実数行列の各列ごとに離散サイン変換を `Backward` で計算します。 |
| `static Matrix<double> SineColumns(ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization)` | Computes the discrete sine transform independently for each column of a matrix with the specified normalization. / 指定した正規化方式で実数行列の各列ごとに離散サイン変換を計算します。 |
| `static MatrixSpan<double> SineColumns(ReadOnlyMatrixSpan<double> x, int type, MatrixSpan<double> destination)` | Computes the discrete sine transform independently for each column using `Backward` and writes the result to `destination`. / 実数行列の各列ごとに離散サイン変換を `Backward` で計算し、`destination` に書き込みます。 |
| `static MatrixSpan<double> SineColumns(ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization, MatrixSpan<double> destination)` | Computes the discrete sine transform independently for each column with the specified normalization and writes the result to `destination`. / 指定した正規化方式で実数行列の各列ごとに離散サイン変換を計算し、`destination` に書き込みます。 |
| `static Matrix<double> SineRows(ReadOnlyMatrixSpan<double> x, int type)` | Computes the discrete sine transform independently for each row of a matrix using `Backward`. / 実数行列の各行ごとに離散サイン変換を `Backward` で計算します。 |
| `static Matrix<double> SineRows(ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization)` | Computes the discrete sine transform independently for each row of a matrix with the specified normalization. / 指定した正規化方式で実数行列の各行ごとに離散サイン変換を計算します。 |
| `static MatrixSpan<double> SineRows(ReadOnlyMatrixSpan<double> x, int type, MatrixSpan<double> destination)` | Computes the discrete sine transform independently for each row using `Backward` and writes the result to `destination`. / 実数行列の各行ごとに離散サイン変換を `Backward` で計算し、`destination` に書き込みます。 |
| `static MatrixSpan<double> SineRows(ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization, MatrixSpan<double> destination)` | Computes the discrete sine transform independently for each row with the specified normalization and writes the result to `destination`. / 指定した正規化方式で実数行列の各行ごとに離散サイン変換を計算し、`destination` に書き込みます。 |
| `static Vector<Complex> Zoom(ReadOnlyVectorSpan<Complex> x, double firstFrequency, double frequencyStep, int count, double sampleSpacing = 1)` | Evaluates the discrete Fourier transform of a complex signal on an arbitrary uniformly spaced frequency grid (zoom FFT). / 任意の等間隔周波数グリッド上で複素数信号の離散フーリエ変換を評価します（ズーム FFT）。 |
| `static Vector<Complex> Zoom(ReadOnlyVectorSpan<double> x, double firstFrequency, double frequencyStep, int count, double sampleSpacing = 1)` | Evaluates the discrete Fourier transform of a real signal on an arbitrary uniformly spaced frequency grid (zoom FFT). / 任意の等間隔周波数グリッド上で実数信号の離散フーリエ変換を評価します（ズーム FFT）。 |
| `static VectorSpan<Complex> Zoom(ReadOnlyVectorSpan<Complex> x, double firstFrequency, double frequencyStep, double sampleSpacing, VectorSpan<Complex> destination)` | Evaluates the discrete Fourier transform of a complex signal on a uniformly spaced frequency grid and writes the result to `destination`. / 等間隔周波数グリッド上で複素数信号の離散フーリエ変換を評価し、`destination` に書き込みます。 |
| `static VectorSpan<Complex> Zoom(ReadOnlyVectorSpan<double> x, double firstFrequency, double frequencyStep, double sampleSpacing, VectorSpan<Complex> destination)` | Evaluates the discrete Fourier transform of a real signal on a uniformly spaced frequency grid and writes the result to `destination`. / 等間隔周波数グリッド上で実数信号の離散フーリエ変換を評価し、`destination` に書き込みます。 |

#### `static class Fourier.InPlace`

Provides in-place Fourier and trigonometric transforms that overwrite the input buffer (`x ← F(x)`). / 入力バッファを直接書き換えるインプレースのフーリエ変換および三角変換を提供します（`x ← F(x)`）。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static void Forward(VectorSpan<Complex> x)` | Computes the forward discrete Fourier transform of a complex vector in place using `Backward`. / 複素数ベクトルの順方向離散フーリエ変換を `Backward` でインプレース計算します。 |
| `static void Forward(VectorSpan<Complex> x, FourierNormalization normalization)` | Computes the forward discrete Fourier transform of a complex vector in place with the specified normalization. / 指定した正規化方式で複素数ベクトルの順方向離散フーリエ変換をインプレース計算します。 |
| `static void Forward2D(MatrixSpan<Complex> x)` | Computes the two-dimensional forward discrete Fourier transform of a complex matrix in place using `Backward`. / 複素数行列の 2 次元順方向離散フーリエ変換を `Backward` でインプレース計算します。 |
| `static void Forward2D(MatrixSpan<Complex> x, FourierNormalization normalization)` | Computes the two-dimensional forward discrete Fourier transform of a complex matrix in place with the specified normalization. / 指定した正規化方式で複素数行列の 2 次元順方向離散フーリエ変換をインプレース計算します。 |
| `static void ForwardColumns(MatrixSpan<Complex> x)` | Computes the forward discrete Fourier transform independently for each column of a complex matrix in place using `Backward`. / 複素数行列の各列ごとに順方向離散フーリエ変換を `Backward` でインプレース計算します。 |
| `static void ForwardColumns(MatrixSpan<Complex> x, FourierNormalization normalization)` | Computes the forward discrete Fourier transform independently for each column of a complex matrix in place with the specified normalization. / 指定した正規化方式で複素数行列の各列ごとに順方向離散フーリエ変換をインプレース計算します。 |
| `static void ForwardRows(MatrixSpan<Complex> x)` | Computes the forward discrete Fourier transform independently for each row of a complex matrix in place using `Backward`. / 複素数行列の各行ごとに順方向離散フーリエ変換を `Backward` でインプレース計算します。 |
| `static void ForwardRows(MatrixSpan<Complex> x, FourierNormalization normalization)` | Computes the forward discrete Fourier transform independently for each row of a complex matrix in place with the specified normalization. / 指定した正規化方式で複素数行列の各行ごとに順方向離散フーリエ変換をインプレース計算します。 |
| `static void Inverse(VectorSpan<Complex> x)` | Computes the inverse discrete Fourier transform of a complex vector in place using `Backward`. / 複素数ベクトルの逆離散フーリエ変換を `Backward` でインプレース計算します。 |
| `static void Inverse(VectorSpan<Complex> x, FourierNormalization normalization)` | Computes the inverse discrete Fourier transform of a complex vector in place with the specified normalization. / 指定した正規化方式で複素数ベクトルの逆離散フーリエ変換をインプレース計算します。 |
| `static void Inverse2D(MatrixSpan<Complex> x)` | Computes the two-dimensional inverse discrete Fourier transform of a complex matrix in place using `Backward`. / 複素数行列の 2 次元逆離散フーリエ変換を `Backward` でインプレース計算します。 |
| `static void Inverse2D(MatrixSpan<Complex> x, FourierNormalization normalization)` | Computes the two-dimensional inverse discrete Fourier transform of a complex matrix in place with the specified normalization. / 指定した正規化方式で複素数行列の 2 次元逆離散フーリエ変換をインプレース計算します。 |
| `static void InverseColumns(MatrixSpan<Complex> x)` | Computes the inverse discrete Fourier transform independently for each column of a complex matrix in place using `Backward`. / 複素数行列の各列ごとに逆離散フーリエ変換を `Backward` でインプレース計算します。 |
| `static void InverseColumns(MatrixSpan<Complex> x, FourierNormalization normalization)` | Computes the inverse discrete Fourier transform independently for each column of a complex matrix in place with the specified normalization. / 指定した正規化方式で複素数行列の各列ごとに逆離散フーリエ変換をインプレース計算します。 |
| `static void InverseRows(MatrixSpan<Complex> x)` | Computes the inverse discrete Fourier transform independently for each row of a complex matrix in place using `Backward`. / 複素数行列の各行ごとに逆離散フーリエ変換を `Backward` でインプレース計算します。 |
| `static void InverseRows(MatrixSpan<Complex> x, FourierNormalization normalization)` | Computes the inverse discrete Fourier transform independently for each row of a complex matrix in place with the specified normalization. / 指定した正規化方式で複素数行列の各行ごとに逆離散フーリエ変換をインプレース計算します。 |

### `enum FourierNormalization`

Specifies the scaling convention applied to discrete Fourier transforms. / 離散フーリエ変換に適用する正規化（スケーリング）方式を指定します。

- 継承・実装: `IComparable`, `IConvertible`, `IFormattable`, `ISpanFormattable`

| 値 | 説明 |
| --- | --- |
| `Backward` | No scaling on the forward transform and `1/n` on the inverse transform (default). / 順変換ではスケーリングを行わず、逆変換で `1/n` を掛けます（既定）。 |
| `Orthonormal` | Scales both forward and inverse transforms by `1/√n`, making the transform unitary. / 順変換・逆変換の両方に `1/√n` を掛け、ユニタリ変換にします。 |
| `Forward` | Scales the forward transform by `1/n` and applies no scaling on the inverse transform. / 順変換で `1/n` を掛け、逆変換ではスケーリングを行いません。 |

### `class FourierPlan`

Represents a reusable plan for computing discrete Fourier transforms of complex sequences of a fixed length. / 固定長の複素数列に対する離散フーリエ変換を繰り返し計算するための実行計画を表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Length { get; }` | Gets the transform length. / 変換の長さを取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static FourierPlan Create(int length)` | Gets or creates a complex Fourier transform plan for the specified length. / 指定した長さの複素フーリエ変換計画を取得または作成します。 |
| `VectorSpan<Complex> Forward(ReadOnlyVectorSpan<Complex> x, VectorSpan<Complex> destination)` | Computes the forward discrete Fourier transform of a complex vector using `Backward` and writes the result to `destination`. / 複素数ベクトルの順方向離散フーリエ変換を `Backward` で計算し、`destination` に書き込みます。 |
| `VectorSpan<Complex> Forward(ReadOnlyVectorSpan<Complex> x, FourierNormalization normalization, VectorSpan<Complex> destination)` | Computes the forward discrete Fourier transform of a complex vector with the specified normalization and writes the result to `destination`. / 指定した正規化方式で複素数ベクトルの順方向離散フーリエ変換を計算し、`destination` に書き込みます。 |
| `MatrixSpan<Complex> ForwardColumns(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination)` | Computes the forward discrete Fourier transform independently for each column of a matrix using `Backward`. / 行列の各列ごとに順方向離散フーリエ変換を `Backward` で計算し、`destination` に書き込みます。 |
| `MatrixSpan<Complex> ForwardColumns(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<Complex> destination)` | Computes the forward discrete Fourier transform independently for each column of a matrix with the specified normalization. / 指定した正規化方式で行列の各列ごとに順方向離散フーリエ変換を計算し、`destination` に書き込みます。 |
| `MatrixSpan<Complex> ForwardRows(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination)` | Computes the forward discrete Fourier transform independently for each row of a matrix using `Backward`. / 行列の各行ごとに順方向離散フーリエ変換を `Backward` で計算し、`destination` に書き込みます。 |
| `MatrixSpan<Complex> ForwardRows(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<Complex> destination)` | Computes the forward discrete Fourier transform independently for each row of a matrix with the specified normalization. / 指定した正規化方式で行列の各行ごとに順方向離散フーリエ変換を計算し、`destination` に書き込みます。 |
| `VectorSpan<Complex> Inverse(ReadOnlyVectorSpan<Complex> x, VectorSpan<Complex> destination)` | Computes the inverse discrete Fourier transform of a complex vector using `Backward` and writes the result to `destination`. / 複素数ベクトルの逆離散フーリエ変換を `Backward` で計算し、`destination` に書き込みます。 |
| `VectorSpan<Complex> Inverse(ReadOnlyVectorSpan<Complex> x, FourierNormalization normalization, VectorSpan<Complex> destination)` | Computes the inverse discrete Fourier transform of a complex vector with the specified normalization and writes the result to `destination`. / 指定した正規化方式で複素数ベクトルの逆離散フーリエ変換を計算し、`destination` に書き込みます。 |
| `MatrixSpan<Complex> InverseColumns(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination)` | Computes the inverse discrete Fourier transform independently for each column of a matrix using `Backward`. / 行列の各列ごとに逆離散フーリエ変換を `Backward` で計算し、`destination` に書き込みます。 |
| `MatrixSpan<Complex> InverseColumns(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<Complex> destination)` | Computes the inverse discrete Fourier transform independently for each column of a matrix with the specified normalization. / 指定した正規化方式で行列の各列ごとに逆離散フーリエ変換を計算し、`destination` に書き込みます。 |
| `MatrixSpan<Complex> InverseRows(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination)` | Computes the inverse discrete Fourier transform independently for each row of a matrix using `Backward`. / 行列の各行ごとに逆離散フーリエ変換を `Backward` で計算し、`destination` に書き込みます。 |
| `MatrixSpan<Complex> InverseRows(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<Complex> destination)` | Computes the inverse discrete Fourier transform independently for each row of a matrix with the specified normalization. / 指定した正規化方式で行列の各行ごとに逆離散フーリエ変換を計算し、`destination` に書き込みます。 |

### `class RealFourierPlan`

Represents a reusable plan for computing discrete Fourier transforms of real-valued sequences of a fixed length. / 固定長の実数列に対する離散フーリエ変換を繰り返し計算するための実行計画を表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `int Length { get; }` | Gets the length of the real-valued sequence. / 実数列の長さ `n` を取得します。 |
| `int SpectrumLength { get; }` | Gets the length of the one-sided Hermitian-symmetric spectrum. / 片側スペクトルの長さを取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static RealFourierPlan Create(int length)` | Gets or creates a real Fourier transform plan for the specified length. / 指定した長さの実数フーリエ変換計画を取得または作成します。 |
| `VectorSpan<Complex> Forward(ReadOnlyVectorSpan<double> x, VectorSpan<Complex> destination)` | Computes the one-sided forward real discrete Fourier transform using `Backward` and writes the result to `destination`. / 実数ベクトルの片側スペクトルを `Backward` で計算し、`destination` に書き込みます。 |
| `VectorSpan<Complex> Forward(ReadOnlyVectorSpan<double> x, FourierNormalization normalization, VectorSpan<Complex> destination)` | Computes the one-sided forward real discrete Fourier transform with the specified normalization and writes the result to `destination`. / 指定した正規化方式で実数ベクトルの片側スペクトルを計算し、`destination` に書き込みます。 |
| `MatrixSpan<Complex> ForwardColumns(ReadOnlyMatrixSpan<double> x, MatrixSpan<Complex> destination)` | Computes the one-sided forward real discrete Fourier transform independently for each column of a matrix using `Backward`. / 行列の各列ごとに実数 FFT の片側スペクトルを `Backward` で計算し、`destination` に書き込みます。 |
| `MatrixSpan<Complex> ForwardColumns(ReadOnlyMatrixSpan<double> x, FourierNormalization normalization, MatrixSpan<Complex> destination)` | Computes the one-sided forward real discrete Fourier transform independently for each column of a matrix with the specified normalization. / 指定した正規化方式で行列の各列ごとに実数 FFT の片側スペクトルを計算し、`destination` に書き込みます。 |
| `MatrixSpan<Complex> ForwardRows(ReadOnlyMatrixSpan<double> x, MatrixSpan<Complex> destination)` | Computes the one-sided forward real discrete Fourier transform independently for each row of a matrix using `Backward`. / 行列の各行ごとに実数 FFT の片側スペクトルを `Backward` で計算し、`destination` に書き込みます。 |
| `MatrixSpan<Complex> ForwardRows(ReadOnlyMatrixSpan<double> x, FourierNormalization normalization, MatrixSpan<Complex> destination)` | Computes the one-sided forward real discrete Fourier transform independently for each row of a matrix with the specified normalization. / 指定した正規化方式で行列の各行ごとに実数 FFT の片側スペクトルを計算し、`destination` に書き込みます。 |
| `VectorSpan<double> Inverse(ReadOnlyVectorSpan<Complex> x, VectorSpan<double> destination)` | Reconstructs a real vector from its one-sided spectrum using `Backward` and writes the result to `destination`. / 片側スペクトルから実数ベクトルを `Backward` で逆変換し、`destination` に書き込みます。 |
| `VectorSpan<double> Inverse(ReadOnlyVectorSpan<Complex> x, FourierNormalization normalization, VectorSpan<double> destination)` | Reconstructs a real vector from its one-sided spectrum with the specified normalization and writes the result to `destination`. / 指定した正規化方式で片側スペクトルから実数ベクトルを逆変換し、`destination` に書き込みます。 |
| `MatrixSpan<double> InverseColumns(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<double> destination)` | Reconstructs real columns from their one-sided spectra using `Backward` and writes the result to `destination`. / 行列の各列の片側スペクトルから実数列を `Backward` で逆変換し、`destination` に書き込みます。 |
| `MatrixSpan<double> InverseColumns(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<double> destination)` | Reconstructs real columns from their one-sided spectra with the specified normalization and writes the result to `destination`. / 指定した正規化方式で行列の各列の片側スペクトルから実数列を逆変換し、`destination` に書き込みます。 |
| `MatrixSpan<double> InverseRows(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<double> destination)` | Reconstructs real rows from their one-sided spectra using `Backward` and writes the result to `destination`. / 行列の各行の片側スペクトルから実数列を `Backward` で逆変換し、`destination` に書き込みます。 |
| `MatrixSpan<double> InverseRows(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<double> destination)` | Reconstructs real rows from their one-sided spectra with the specified normalization and writes the result to `destination`. / 指定した正規化方式で行列の各行の片側スペクトルから実数列を逆変換し、`destination` に書き込みます。 |

### `enum WaveletBoundary`

Specifies the signal boundary extension mode used by discrete wavelet transforms. / 離散ウェーブレット変換における信号境界の拡張方式を指定します。

- 継承・実装: `IComparable`, `IConvertible`, `IFormattable`, `ISpanFormattable`

| 値 | 説明 |
| --- | --- |
| `Symmetric` | Half-point symmetric extension: `… x₂ x₁ \| x₁ … xₙ \| xₙ xₙ₋₁ …` (default). / 半標本対称拡張 `… x₂ x₁ \| x₁ … xₙ \| xₙ xₙ₋₁ …`（既定）。 |
| `Periodization` | Periodization producing minimal coefficient length `⌈n / 2⌉` (duplicates the last sample when `n` is odd). Yields an orthogonal transform for orthogonal wavelets. / 周期化。係数長が最小の `⌈n / 2⌉` となり（`n` が奇数のときは末尾を 1 点複製）、直交ウェーブレットでは変換が直交行列になります。 |
| `Zero` | Zero padding: `… 0 0 \| x₁ … xₙ \| 0 0 …`. / ゼロパディング `… 0 0 \| x₁ … xₙ \| 0 0 …`。 |
| `Constant` | Constant edge replication: `… x₁ x₁ \| x₁ … xₙ \| xₙ xₙ …`. / 端点の定数拡張 `… x₁ x₁ \| x₁ … xₙ \| xₙ xₙ …`。 |
| `Reflect` | Whole-point symmetric extension without repeating edge values: `… x₃ x₂ \| x₁ … xₙ \| xₙ₋₁ xₙ₋₂ …`. Requires `n ≥ 2`. / 全標本対称拡張（端点を繰り返さない）`… x₃ x₂ \| x₁ … xₙ \| xₙ₋₁ xₙ₋₂ …`。`n ≥ 2` が必要です。 |
| `Periodic` | Periodic extension: `… xₙ₋₁ xₙ \| x₁ … xₙ \| x₁ x₂ …`, producing coefficient length `⌊(n + F - 1) / 2⌋`. / 周期拡張 `… xₙ₋₁ xₙ \| x₁ … xₙ \| x₁ x₂ …`（`Periodization` とは異なり係数長は `⌊(n + F − 1) / 2⌋` です）。 |

### `class WaveletDecomposition`

Holds the result of a multilevel 1D discrete wavelet decomposition from `Decompose`. / `Decompose` による多段 1 次元離散ウェーブレット分解の結果を保持します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `Vector<double> Approximation { get; }` | Gets the coarsest-level approximation coefficients (`cA_n`). / 最も粗い段の近似係数（`cA_n`）を取得します。 |
| `WaveletBoundary Boundary { get; }` | Gets the boundary extension mode used for the decomposition. / 分解に使用した境界拡張方式を取得します。 |
| `IReadOnlyList<Vector<double>> Details { get; }` | Gets the detail coefficient vectors ordered from coarsest (`cD_n` at index 0) to finest (`cD_1` at the end). / 粗い段（先頭が `cD_n`）から細かい段（末尾が `cD_1`）の順に並んだ詳細係数の一覧を取得します。 |
| `int Length { get; }` | Gets the length of the original input signal. / 元の入力信号の長さを取得します。 |
| `int Levels { get; }` | Gets the number of decomposition levels. / 分解段数を取得します。 |
| `DiscreteWavelet Wavelet { get; }` | Gets the discrete wavelet used for the decomposition. / 分解に使用した離散ウェーブレットを取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `Vector<double> Detail(int level)` | Returns the detail coefficients at the specified 1-based level (`1` is the finest level `cD_1`; `Levels` is the coarsest). / 指定した段（1 が最も細かい `cD_1`、`Levels` が最も粗い段）の詳細係数を返します。 |

### `class WaveletDecomposition2D`

Holds the result of a multilevel 2D discrete wavelet decomposition from `Decompose2D`. / `Decompose2D` による多段 2 次元離散ウェーブレット分解の結果を保持します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `Matrix<double> Approximation { get; }` | Gets the coarsest-level approximation coefficients (`cA_n`). / 最も粗い段の近似係数（`cA_n`）を取得します。 |
| `WaveletBoundary Boundary { get; }` | Gets the boundary extension mode used for the decomposition. / 分解に使用した境界拡張方式を取得します。 |
| `int Columns { get; }` | Gets the column count of the original input matrix. / 元の入力行列の列数を取得します。 |
| `IReadOnlyList<(Matrix<double>, Matrix<double>, Matrix<double>)> Details { get; }` | Gets the detail subband triples (Horizontal, Vertical, Diagonal) ordered from coarsest to finest level. / 粗い段から細かい段へ並べた詳細係数（水平・垂直・対角）の組の一覧を取得します。 |
| `int Levels { get; }` | Gets the number of decomposition levels. / 分解段数を取得します。 |
| `int Rows { get; }` | Gets the row count of the original input matrix. / 元の入力行列の行数を取得します。 |
| `DiscreteWavelet Wavelet { get; }` | Gets the discrete wavelet used for the decomposition. / 分解に使用した離散ウェーブレットを取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `(Matrix<double>, Matrix<double>, Matrix<double>) Detail(int level)` | Returns the detail subband triple at the specified 1-based level (`1` is the finest level; `Levels` is the coarsest). / 指定した段（1 が最も細かい段、`Levels` が最も粗い段）の詳細係数の組を返します。 |

### `class WaveletPacketTree`

Represents the result of a wavelet packet decomposition returned by `DecomposePackets`. / `DecomposePackets` が返すウェーブレットパケット分解の結果を表します。

**プロパティ**

| メンバー | 説明 |
| --- | --- |
| `WaveletBoundary Boundary { get; }` | Gets the boundary extension mode used for decomposition. / 分解に使用した境界拡張モードを取得します。 |
| `int Length { get; }` | Gets the length of the original input signal. / 元の入力信号の長さを取得します。 |
| `int Levels { get; }` | Gets the number of decomposition levels (the deepest level index). / 分解段数（最も深い段の番号）を取得します。 |
| `DiscreteWavelet Wavelet { get; }` | Gets the discrete wavelet used for decomposition. / 分解に使用した離散ウェーブレットを取得します。 |

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `IReadOnlyList<(int, int)> BestBasis()` | Finds the wavelet packet basis that minimizes total `ShannonEntropy` across the tree. / `ShannonEntropy` の合計を最小にするウェーブレットパケット基底を求めます。 |
| `IReadOnlyList<(int, int)> BestBasis(Func<ReadOnlyVectorSpan<double>, double> cost)` | Finds the wavelet packet basis that minimizes the sum of an additive node cost function across the tree. / 加法的なノードコスト関数の合計を最小にするウェーブレットパケット基底を求めます。 |
| `static int FrequencyIndex(int naturalIndex)` | Converts a zero-based natural-order node index to its corresponding ascending frequency-order index (the inverse of `NaturalIndex`). / 0 始まりの自然順ノード番号を、周波数の昇順の番号（`NaturalIndex` の逆変換）に変換します。 |
| `MatrixSpan<double> Level(int level)` | Returns a writable view of all wavelet packet coefficients at the specified level in natural order. / 指定した段のすべてのウェーブレットパケット係数を自然順で参照する書き込み可能なビューを返します。 |
| `Matrix<double> LevelByFrequency(int level)` | Returns a newly allocated matrix containing all wavelet packet coefficients at the specified level ordered by ascending frequency. / 指定した段のウェーブレットパケット係数を周波数の昇順に並べた新しい行列を返します。 |
| `static int NaturalIndex(int frequencyIndex)` | Converts a zero-based frequency-ordered node index to its corresponding natural-order index using the Gray code `p ^ (p >> 1)`. / 0 始まりの周波数順ノード番号を、グレイ符号 `p ^ (p >> 1)` によって自然順のノード番号に変換します。 |
| `VectorSpan<double> Node(int level, int index)` | Returns a writable contiguous view of the coefficients for the specified node. / 指定したノードの係数を参照する書き込み可能な連続ビューを返します。 |
| `static double ShannonEntropy(ReadOnlyVectorSpan<double> coefficients)` | Computes the unnormalized Shannon entropy of the specified wavelet coefficients. / 指定したウェーブレット係数の非正規化シャノンエントロピーを計算します。 |

### `enum WaveletPadding`

Specifies the boundary padding mode used by the continuous wavelet transform. / 連続ウェーブレット変換における信号境界のパディング方式を指定します。

- 継承・実装: `IComparable`, `IConvertible`, `IFormattable`, `ISpanFormattable`

| 値 | 説明 |
| --- | --- |
| `Zero` | Zero-pads the signal to `Fourier.NextFastLength(2n - 1)` so edge values do not wrap around (default). / 長さ `Fourier.NextFastLength(2n − 1)` までゼロパディングして FFT を実行し、境界の回り込みを防ぎます（既定）。 |
| `Periodic` | Treats the signal as periodic with period `n` and transforms it directly without padding. / 信号を周期 `n` の周期信号とみなし、パディングなしで長さ `n` の FFT を実行します。 |

### `enum WaveletShrinkage`

Specifies how wavelet detail coefficients are shrunk by a threshold `T`. / しきい値 `T` によるウェーブレット係数の収縮方式を指定します。

- 継承・実装: `IComparable`, `IConvertible`, `IFormattable`, `ISpanFormattable`

| 値 | 説明 |
| --- | --- |
| `Soft` | Soft thresholding: `sign(x) · max(\|x\| - T, 0)` (default, equivalent to `SoftThreshold`). / 軟判定しきい値処理 `sign(x) · max(\|x\| − T, 0)`（既定。`SoftThreshold` と同一）。 |
| `Hard` | Hard thresholding: sets coefficients with `\|x\| < T` to zero and leaves all others unchanged. / 硬判定しきい値処理。`\|x\| < T` の係数を 0 にし、それ以外はそのまま残します。 |

### `enum WaveletThresholdRule`

Specifies the threshold selection rule used for wavelet denoising. / ウェーブレットノイズ除去におけるしきい値決定規則を指定します。

- 継承・実装: `IComparable`, `IConvertible`, `IFormattable`, `ISpanFormattable`

| 値 | 説明 |
| --- | --- |
| `Bayes` | Selects a subband-adaptive BayesShrink threshold from detail coefficient variance (default). / 詳細係数の分散からサブバンドごとに BayesShrink しきい値を決定します（既定）。 |
| `Universal` | Applies the VisuShrink universal threshold `σ · √(2 ln N)` (Donoho–Johnstone 1994) to all detail subbands, where `N` is the number of signal samples. / VisuShrink（Donoho–Johnstone 1994 の普遍しきい値）を適用し、すべての詳細係数に `σ · √(2 ln N)`（`N` は信号標本数）を使用します。 |

### `static class Wavelets`

Provides discrete, stationary, packet, and continuous wavelet transforms, signal reconstruction, and wavelet denoising. / 離散・定常・パケット・連続ウェーブレット変換、信号の再構成、およびウェーブレットノイズ除去を提供します。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static double BayesThreshold(ReadOnlyVectorSpan<double> detail, double noiseLevel)` | Computes the BayesShrink threshold `sigma^2 / sqrt(max(mean(d_i^2) - sigma^2, eps))` for a 1D detail vector, where `eps = 2^-52`. / 1 次元詳細係数ベクトルに対する BayesShrink のしきい値 `sigma^2 / sqrt(max(mean(d_i^2) - sigma^2, eps))`（`eps = 2^-52`）を計算します。 |
| `static double BayesThreshold(ReadOnlyMatrixSpan<double> detail, double noiseLevel)` | Computes the BayesShrink threshold for a 2D detail coefficient matrix, accumulating squared values in column-major order. / 2 次元詳細係数行列に対する BayesShrink のしきい値を計算します（二乗和は列優先の順に集計します）。 |
| `static int CoefficientLength(int length, DiscreteWavelet wavelet, WaveletBoundary boundary)` | Computes the length of the approximation or detail coefficient vector produced by a single level of the discrete wavelet transform. / 離散ウェーブレット変換の 1 段で生成される近似係数または詳細係数の長さを計算します。 |
| `static Vector<double> ConeOfInfluence(int length, ContinuousWavelet wavelet, double sampleSpacing)` | Computes the cone of influence (COI) scale boundary at each sample index for a continuous wavelet transform. / 連続ウェーブレット変換の各時刻において境界の影響を受けるスケールの目安（Cone of Influence）を計算します。 |
| `static Matrix<Complex> Continuous(ReadOnlyVectorSpan<double> x, ReadOnlyVectorSpan<double> scales, ContinuousWavelet wavelet, double sampleSpacing)` | Computes the continuous wavelet transform (CWT) of a signal at the specified scales using zero padding. / ゼロパディングを用いて、指定したスケールにおける入力信号の連続ウェーブレット変換（CWT）を計算します。 |
| `static Matrix<Complex> Continuous(ReadOnlyVectorSpan<double> x, ReadOnlyVectorSpan<double> scales, ContinuousWavelet wavelet, double sampleSpacing, WaveletPadding padding)` | Computes the continuous wavelet transform (CWT) of a signal at the specified scales and padding mode. / 指定したスケールとパディング方式で、入力信号の連続ウェーブレット変換（CWT）を計算します。 |
| `static MatrixSpan<Complex> Continuous(ReadOnlyVectorSpan<double> x, ReadOnlyVectorSpan<double> scales, ContinuousWavelet wavelet, double sampleSpacing, WaveletPadding padding, MatrixSpan<Complex> destination)` | Computes the continuous wavelet transform (CWT) of a signal and writes the complex coefficients into `destination`. / 連続ウェーブレット変換（CWT）を計算し、複素係数を `destination` へ書き込みます。 |
| `static WaveletDecomposition Decompose(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet)` | Computes a multi-level discrete wavelet decomposition using the maximum useful level and symmetric boundary extension. / 最大有効段数と対称境界拡張を用いて、入力信号の多段離散ウェーブレット分解を計算します。 |
| `static WaveletDecomposition Decompose(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, WaveletBoundary boundary)` | Computes a multi-level discrete wavelet decomposition using the maximum useful level and the specified boundary extension. / 最大有効段数と指定した境界拡張モードを用いて、入力信号の多段離散ウェーブレット分解を計算します。 |
| `static WaveletDecomposition Decompose(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, int levels)` | Computes a multi-level discrete wavelet decomposition to the specified number of levels using symmetric boundary extension. / 指定した段数と対称境界拡張を用いて、入力信号の多段離散ウェーブレット分解を計算します。 |
| `static WaveletDecomposition Decompose(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, int levels, WaveletBoundary boundary)` | Computes a multi-level discrete wavelet decomposition to the specified number of levels and boundary extension. / 指定した段数および境界拡張モードで、入力信号の多段離散ウェーブレット分解を計算します。 |
| `static WaveletDecomposition2D Decompose2D(ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet)` | Computes a multi-level 2D discrete wavelet decomposition using the maximum useful level and symmetric boundary extension. / 最大有効段数と対称境界拡張を用いて、行列の多段 2 次元離散ウェーブレット分解を計算します。 |
| `static WaveletDecomposition2D Decompose2D(ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet, WaveletBoundary boundary)` | Computes a multi-level 2D discrete wavelet decomposition using the maximum useful level and the specified boundary extension. / 最大有効段数と指定した境界拡張モードを用いて、行列の多段 2 次元離散ウェーブレット分解を計算します。 |
| `static WaveletDecomposition2D Decompose2D(ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet, int levels)` | Computes a multi-level 2D discrete wavelet decomposition to the specified number of levels using symmetric boundary extension. / 指定した段数と対称境界拡張を用いて、行列の多段 2 次元離散ウェーブレット分解を計算します。 |
| `static WaveletDecomposition2D Decompose2D(ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet, int levels, WaveletBoundary boundary)` | Computes a multi-level 2D discrete wavelet decomposition to the specified number of levels and boundary extension. / 指定した段数および境界拡張モードで、行列の多段 2 次元離散ウェーブレット分解を計算します。 |
| `static WaveletPacketTree DecomposePackets(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet)` | Computes the wavelet packet decomposition of a signal using the maximum useful level and symmetric boundary extension. / 最大有効段数と対称境界拡張を用いて、入力信号のウェーブレットパケット分解を計算します。 |
| `static WaveletPacketTree DecomposePackets(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, WaveletBoundary boundary)` | Computes the wavelet packet decomposition of a signal using the maximum useful level and the specified boundary extension. / 最大有効段数と指定した境界拡張モードを用いて、入力信号のウェーブレットパケット分解を計算します。 |
| `static WaveletPacketTree DecomposePackets(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, int levels)` | Computes the wavelet packet decomposition of a signal to the specified number of levels using symmetric boundary extension. / 指定した段数と対称境界拡張を用いて、入力信号のウェーブレットパケット分解を計算します。 |
| `static WaveletPacketTree DecomposePackets(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, int levels, WaveletBoundary boundary)` | Computes the wavelet packet decomposition of a signal to the specified number of levels and boundary extension. / 指定した段数および境界拡張モードで、入力信号のウェーブレットパケット分解を計算します。 |
| `static Vector<double> Denoise(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet)` | Denoises a 1D signal by wavelet thresholding using BayesShrink and soft thresholding. / BayesShrink と軟判定しきい値処理により、1 次元信号のノイズを除去します。 |
| `static Vector<double> Denoise(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, WaveletThresholdRule rule, WaveletShrinkage shrinkage)` | Denoises a 1D signal by wavelet thresholding using the specified threshold rule and shrinkage function. / 指定したしきい値ルールと縮小方式を用いて、1 次元信号のノイズを除去します。 |
| `static Vector<double> Denoise(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, WaveletThresholdRule rule, WaveletShrinkage shrinkage, int levels)` | Denoises a 1D signal by wavelet thresholding using the specified threshold rule, shrinkage function, and decomposition depth. / 指定したしきい値ルール、縮小方式、および分解段数を用いて、1 次元信号のノイズを除去します。 |
| `static Vector<double> Denoise(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, WaveletThresholdRule rule, WaveletShrinkage shrinkage, int levels, double noiseLevel)` | Denoises a 1D signal by wavelet thresholding using the specified parameters and a known noise standard deviation. / 既知のノイズ標準偏差と指定した設定を用いて、1 次元信号のノイズを除去します。 |
| `static Matrix<double> Denoise2D(ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet)` | Denoises a 2D signal matrix by wavelet thresholding using BayesShrink and soft thresholding. / BayesShrink と軟判定しきい値処理により、2 次元信号行列のノイズを除去します。 |
| `static Matrix<double> Denoise2D(ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet, WaveletThresholdRule rule, WaveletShrinkage shrinkage)` | Denoises a 2D signal matrix by wavelet thresholding using the specified threshold rule and shrinkage function. / 指定したしきい値ルールと縮小方式を用いて、2 次元信号行列のノイズを除去します。 |
| `static Matrix<double> Denoise2D(ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet, WaveletThresholdRule rule, WaveletShrinkage shrinkage, int levels)` | Denoises a 2D signal matrix by wavelet thresholding using the specified threshold rule, shrinkage function, and decomposition depth. / 指定したしきい値ルール、縮小方式、および分解段数を用いて、2 次元信号行列のノイズを除去します。 |
| `static Matrix<double> Denoise2D(ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet, WaveletThresholdRule rule, WaveletShrinkage shrinkage, int levels, double noiseLevel)` | Denoises a 2D signal matrix by wavelet thresholding using the specified parameters and a known noise standard deviation. / 既知のノイズ標準偏差と指定した設定を用いて、2 次元信号行列のノイズを除去します。 |
| `static double EstimateNoise(ReadOnlyVectorSpan<double> detail)` | Estimates the noise standard deviation from a 1D detail coefficient vector using the median absolute deviation (MAD). / 中央絶対偏差（MAD）を用いて、1 次元詳細係数ベクトルからノイズの標準偏差を推定します。 |
| `static double EstimateNoise(ReadOnlyMatrixSpan<double> detail)` | Estimates the noise standard deviation from a 2D detail coefficient matrix using the median absolute deviation (MAD). / 中央絶対偏差（MAD）を用いて、2 次元詳細係数行列からノイズの標準偏差を推定します。 |
| `static (Vector<double>, Vector<double>) Forward(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet)` | Computes a single-level discrete wavelet transform using symmetric boundary extension. / 対称境界拡張を用いて 1 段の離散ウェーブレット変換を計算します。 |
| `static (Vector<double>, Vector<double>) Forward(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, WaveletBoundary boundary)` | Computes a single-level discrete wavelet transform using the specified boundary extension. / 指定した境界拡張モードを用いて 1 段の離散ウェーブレット変換を計算します。 |
| `static void Forward(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, WaveletBoundary boundary, VectorSpan<double> approximation, VectorSpan<double> detail)` | Computes a single-level discrete wavelet transform and writes the coefficients into the specified destination spans. / 1 段の離散ウェーブレット変換を計算し、指定した出力先スパンへ係数を書き込みます。 |
| `static (Matrix<double>, Matrix<double>, Matrix<double>, Matrix<double>) Forward2D(ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet)` | Computes a single-level 2D discrete wavelet transform along both axes of a matrix using symmetric boundary extension. / 対称境界拡張を用いて、行列の両軸に対して 1 段の 2 次元離散ウェーブレット変換を計算します。 |
| `static (Matrix<double>, Matrix<double>, Matrix<double>, Matrix<double>) Forward2D(ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet, WaveletBoundary boundary)` | Computes a single-level 2D discrete wavelet transform along both axes of a matrix using the specified boundary extension. / 指定した境界拡張モードを用いて、行列の両軸に対して 1 段の 2 次元離散ウェーブレット変換を計算します。 |
| `static (Matrix<double>, Matrix<double>) ForwardColumns(ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet)` | Computes a single-level discrete wavelet transform along each column of a matrix using symmetric boundary extension. / 対称境界拡張を用いて、行列の各列に対して 1 段の離散ウェーブレット変換を計算します。 |
| `static (Matrix<double>, Matrix<double>) ForwardColumns(ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet, WaveletBoundary boundary)` | Computes a single-level discrete wavelet transform along each column of a matrix using the specified boundary extension. / 指定した境界拡張モードを用いて、行列の各列に対して 1 段の離散ウェーブレット変換を計算します。 |
| `static (Matrix<double>, Matrix<double>) ForwardRows(ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet)` | Computes a single-level discrete wavelet transform along each row of a matrix using symmetric boundary extension. / 対称境界拡張を用いて、行列の各行に対して 1 段の離散ウェーブレット変換を計算します。 |
| `static (Matrix<double>, Matrix<double>) ForwardRows(ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet, WaveletBoundary boundary)` | Computes a single-level discrete wavelet transform along each row of a matrix using the specified boundary extension. / 指定した境界拡張モードを用いて、行列の各行に対して 1 段の離散ウェーブレット変換を計算します。 |
| `static Vector<double> GeometricScales(double smallest, double octaveFraction, int count)` | Generates a geometric progression of scales `s_j = s0 * 2^(j * delta_j)` for `j = 0 .. count - 1`. / 等比数列のスケール `s_j = s0 * 2^(j * delta_j)`（`j = 0 .. count - 1`）を生成します。 |
| `static Vector<double> Inverse(ReadOnlyVectorSpan<double> approximation, ReadOnlyVectorSpan<double> detail, DiscreteWavelet wavelet)` | Computes a single-level inverse discrete wavelet transform using symmetric boundary extension. / 対称境界拡張を用いて 1 段の離散ウェーブレット逆変換を計算します。 |
| `static Vector<double> Inverse(ReadOnlyVectorSpan<double> approximation, ReadOnlyVectorSpan<double> detail, DiscreteWavelet wavelet, WaveletBoundary boundary)` | Computes a single-level inverse discrete wavelet transform using the specified boundary extension. / 指定した境界拡張モードを用いて 1 段の離散ウェーブレット逆変換を計算します。 |
| `static VectorSpan<double> Inverse(ReadOnlyVectorSpan<double> approximation, ReadOnlyVectorSpan<double> detail, DiscreteWavelet wavelet, WaveletBoundary boundary, VectorSpan<double> destination)` | Computes a single-level inverse discrete wavelet transform and writes the reconstructed signal into `destination`. / 1 段の離散ウェーブレット逆変換を計算し、再構成した信号を `destination` へ書き込みます。 |
| `static Matrix<double> Inverse2D(ReadOnlyMatrixSpan<double> approximation, ReadOnlyMatrixSpan<double> horizontal, ReadOnlyMatrixSpan<double> vertical, ReadOnlyMatrixSpan<double> diagonal, DiscreteWavelet wavelet)` | Computes a single-level 2D inverse discrete wavelet transform using symmetric boundary extension. / 対称境界拡張を用いて、1 段の 2 次元離散ウェーブレット逆変換を計算します。 |
| `static Matrix<double> Inverse2D(ReadOnlyMatrixSpan<double> approximation, ReadOnlyMatrixSpan<double> horizontal, ReadOnlyMatrixSpan<double> vertical, ReadOnlyMatrixSpan<double> diagonal, DiscreteWavelet wavelet, WaveletBoundary boundary)` | Computes a single-level 2D inverse discrete wavelet transform using the specified boundary extension. / 指定した境界拡張モードを用いて、1 段の 2 次元離散ウェーブレット逆変換を計算します。 |
| `static Matrix<double> InverseColumns(ReadOnlyMatrixSpan<double> approximation, ReadOnlyMatrixSpan<double> detail, DiscreteWavelet wavelet)` | Computes a single-level inverse discrete wavelet transform along each column of a matrix using symmetric boundary extension. / 対称境界拡張を用いて、行列の各列に対して 1 段の離散ウェーブレット逆変換を計算します。 |
| `static Matrix<double> InverseColumns(ReadOnlyMatrixSpan<double> approximation, ReadOnlyMatrixSpan<double> detail, DiscreteWavelet wavelet, WaveletBoundary boundary)` | Computes a single-level inverse discrete wavelet transform along each column of a matrix using the specified boundary extension. / 指定した境界拡張モードを用いて、行列の各列に対して 1 段の離散ウェーブレット逆変換を計算します。 |
| `static Vector<double> InverseContinuous(ReadOnlyMatrixSpan<Complex> coefficients, ReadOnlyVectorSpan<double> scales, ContinuousWavelet wavelet, double sampleSpacing)` | Approximately reconstructs a real-valued signal from its continuous wavelet transform coefficients. / 連続ウェーブレット変換の係数から実数信号を近似的に再構成します。 |
| `static VectorSpan<double> InverseContinuous(ReadOnlyMatrixSpan<Complex> coefficients, ReadOnlyVectorSpan<double> scales, ContinuousWavelet wavelet, double sampleSpacing, VectorSpan<double> destination)` | Approximately reconstructs a real-valued signal from its continuous wavelet transform coefficients into `destination`. / 連続ウェーブレット変換の係数から実数信号を近似的に再構成し、`destination` へ書き込みます。 |
| `static int InverseLength(int coefficientLength, DiscreteWavelet wavelet, WaveletBoundary boundary)` | Computes the signal length produced by a single level of the inverse discrete wavelet transform. / 離散ウェーブレット逆変換の 1 段で生成される信号の長さを計算します。 |
| `static Vector<double> InverseMaximalOverlap(ReadOnlyMatrixSpan<double> coefficients, DiscreteWavelet wavelet)` | Reconstructs a signal from its maximal overlap discrete wavelet transform (MODWT) coefficients. / 最大重複離散ウェーブレット変換（MODWT）の係数行列から信号を再構成します。 |
| `static VectorSpan<double> InverseMaximalOverlap(ReadOnlyMatrixSpan<double> coefficients, DiscreteWavelet wavelet, VectorSpan<double> destination)` | Reconstructs a signal from its maximal overlap discrete wavelet transform (MODWT) coefficients into `destination`. / 最大重複離散ウェーブレット変換（MODWT）の係数行列から信号を再構成し、`destination` へ書き込みます。 |
| `static Matrix<double> InverseRows(ReadOnlyMatrixSpan<double> approximation, ReadOnlyMatrixSpan<double> detail, DiscreteWavelet wavelet)` | Computes a single-level inverse discrete wavelet transform along each row of a matrix using symmetric boundary extension. / 対称境界拡張を用いて、行列の各行に対して 1 段の離散ウェーブレット逆変換を計算します。 |
| `static Matrix<double> InverseRows(ReadOnlyMatrixSpan<double> approximation, ReadOnlyMatrixSpan<double> detail, DiscreteWavelet wavelet, WaveletBoundary boundary)` | Computes a single-level inverse discrete wavelet transform along each row of a matrix using the specified boundary extension. / 指定した境界拡張モードを用いて、行列の各行に対して 1 段の離散ウェーブレット逆変換を計算します。 |
| `static Vector<double> InverseStationary(ReadOnlyMatrixSpan<double> coefficients, DiscreteWavelet wavelet)` | Reconstructs a signal from its stationary wavelet transform (SWT) coefficients. / 定常ウェーブレット変換（SWT）の係数行列から信号を再構成します。 |
| `static VectorSpan<double> InverseStationary(ReadOnlyMatrixSpan<double> coefficients, DiscreteWavelet wavelet, VectorSpan<double> destination)` | Reconstructs a signal from its stationary wavelet transform (SWT) coefficients into `destination`. / 定常ウェーブレット変換（SWT）の係数行列から信号を再構成し、`destination` へ書き込みます。 |
| `static int MaxLevel(int length, DiscreteWavelet wavelet)` | Computes the maximum useful decomposition level at which at least one coefficient remains free of boundary effects. / 境界の影響を受けない係数が残る最大分解段数を計算します。 |
| `static int MaxLevel(int rows, int columns, DiscreteWavelet wavelet)` | Computes the maximum useful 2D decomposition level at which at least one coefficient remains free of boundary effects along both axes. / 両軸について境界の影響を受けない係数が残る最大分解段数を計算します。 |
| `static Matrix<double> MaximalOverlap(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, int levels)` | Computes the maximal overlap discrete wavelet transform (MODWT) of a signal. / 入力信号の最大重複離散ウェーブレット変換（MODWT）を計算します。 |
| `static MatrixSpan<double> MaximalOverlap(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, MatrixSpan<double> destination)` | Computes the maximal overlap discrete wavelet transform (MODWT) and writes the coefficients into `destination`. / 最大重複離散ウェーブレット変換（MODWT）を計算し、係数を `destination` へ書き込みます。 |
| `static Matrix<double> MaximalOverlapMultiresolution(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, int levels)` | Computes the MODWT multiresolution analysis (MRA) components of a signal. / MODWT による多重解像度解析（MRA）の成分を計算します。 |
| `static Vector<double> Reconstruct(WaveletDecomposition decomposition)` | Reconstructs a signal from its multi-level discrete wavelet decomposition. / 多段離散ウェーブレット分解から元の信号を再構成します。 |
| `static Matrix<double> Reconstruct(WaveletDecomposition2D decomposition)` | Reconstructs a matrix from its multi-level 2D discrete wavelet decomposition. / 多段 2 次元離散ウェーブレット分解から元の行列を再構成します。 |
| `static Vector<double> Reconstruct(WaveletPacketTree tree, IEnumerable<(int, int)> basis)` | Reconstructs a signal from the specified wavelet packet basis. / 指定したウェーブレットパケット基底から信号を再構成します。 |
| `static Vector<double> Reconstruct(WaveletPacketTree tree, int level)` | Reconstructs a signal from all wavelet packet nodes at the specified level. / 指定した段の全ノードから信号を再構成します。 |
| `static void Shrink(VectorSpan<double> coefficients, double threshold, WaveletShrinkage shrinkage)` | Applies thresholding to a vector of wavelet coefficients in place. / ウェーブレット係数のベクトルに対してインプレースでしきい値処理を適用します。 |
| `static void Shrink(MatrixSpan<double> coefficients, double threshold, WaveletShrinkage shrinkage)` | Applies thresholding to a matrix of wavelet coefficients in place. / ウェーブレット係数の行列に対してインプレースでしきい値処理を適用します。 |
| `static Matrix<double> Stationary(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, int levels)` | Computes the undecimated stationary wavelet transform (SWT) of a signal. / 入力信号の定常ウェーブレット変換（SWT）を計算します。 |
| `static MatrixSpan<double> Stationary(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, MatrixSpan<double> destination)` | Computes the undecimated stationary wavelet transform (SWT) and writes the coefficients into `destination`. / 定常ウェーブレット変換（SWT）を計算し、係数を `destination` へ書き込みます。 |
| `static double UniversalThreshold(double noiseLevel, long count)` | Computes the universal (VisuShrink) threshold `sigma * sqrt(2 * ln(N))`. / 普遍しきい値（VisuShrink）`sigma * sqrt(2 * ln(N))` を計算します。 |

### `static class XMathExtensions`

Provides signal processing extension members for `XMath`, including convolution, cross-correlation, and autocorrelation. / 畳み込み・相互相関・自己相関など、`XMath` に対する信号処理の拡張メンバーを提供します。

- 呼び出し形式: `using Tuna.SignalProcessing;` のうえで `XMath.<Method>(...)` として呼び出します（`using static` では解決されません）。

**メソッド**

| メソッド | 説明 |
| --- | --- |
| `static Vector<T> AutoCorrelate<T>(ReadOnlyVectorSpan<T> x, int maxLag)  (+2 overloads)` | Computes the unnormalized autocorrelation of a vector for lags `0` through `maxLag`. / ベクトルの非正規化自己相関をラグ `0` から `maxLag` まで計算します。 |
| `static Vector<T> AutoCorrelate<T>(ReadOnlyVectorSpan<T> x, int maxLag, ConvolutionMethod method)  (+2 overloads)` | Computes the unnormalized autocorrelation of a vector for lags `0` through `maxLag` using the specified method. / 指定した計算方式でベクトルの非正規化自己相関をラグ `0` から `maxLag` まで計算します。 |
| `static VectorSpan<T> AutoCorrelate<T>(ReadOnlyVectorSpan<T> x, int maxLag, VectorSpan<T> destination)  (+2 overloads)` | Computes the unnormalized autocorrelation of a vector and writes the result to `destination`. / ベクトルの非正規化自己相関を計算し、`destination` に書き込みます。 |
| `static VectorSpan<T> AutoCorrelate<T>(ReadOnlyVectorSpan<T> x, int maxLag, ConvolutionMethod method, VectorSpan<T> destination)  (+2 overloads)` | Computes the unnormalized autocorrelation of a vector using the specified method and writes the result to `destination`. / 指定した計算方式でベクトルの非正規化自己相関を計算し、`destination` に書き込みます。 |
| `static Matrix<T> ColumnAutoCorrelations<T>(ReadOnlyMatrixSpan<T> x, int maxLag)  (+2 overloads)` | Computes the unnormalized autocorrelation independently for each column of a matrix. / 行列の各列ごとに非正規化自己相関を計算します。 |
| `static Matrix<T> ColumnAutoCorrelations<T>(ReadOnlyMatrixSpan<T> x, int maxLag, ConvolutionMethod method)  (+2 overloads)` | Computes the unnormalized autocorrelation independently for each column of a matrix using the specified method. / 指定した計算方式で行列の各列ごとに非正規化自己相関を計算します。 |
| `static MatrixSpan<T> ColumnAutoCorrelations<T>(ReadOnlyMatrixSpan<T> x, int maxLag, MatrixSpan<T> destination)  (+2 overloads)` | Computes the unnormalized autocorrelation independently for each column of a matrix and writes the result to `destination`. / 行列の各列ごとに非正規化自己相関を計算し、`destination` に書き込みます。 |
| `static MatrixSpan<T> ColumnAutoCorrelations<T>(ReadOnlyMatrixSpan<T> x, int maxLag, ConvolutionMethod method, MatrixSpan<T> destination)  (+2 overloads)` | Computes the unnormalized autocorrelation independently for each column of a matrix using the specified method and writes the result to `destination`. / 指定した計算方式で行列の各列ごとに非正規化自己相関を計算し、`destination` に書き込みます。 |
| `static Vector<T> Convolve<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)  (+2 overloads)` | Computes the linear convolution of two vectors using `Full` and `Auto`. / 2 つのベクトルの線形畳み込みを `Full` および `Auto` で計算します。 |
| `static Vector<T> Convolve<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, ConvolutionMode mode)  (+2 overloads)` | Computes the linear convolution of two vectors with the specified output mode. / 指定した出力範囲で 2 つのベクトルの線形畳み込みを計算します。 |
| `static VectorSpan<T> Convolve<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, VectorSpan<T> destination)  (+2 overloads)` | Computes the linear convolution of two vectors using `Full` and writes the result to `destination`. / 2 つのベクトルの線形畳み込みを `Full` で計算し、`destination` に書き込みます。 |
| `static Vector<T> Convolve<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, ConvolutionMode mode, ConvolutionMethod method)  (+2 overloads)` | Computes the linear convolution of two vectors with the specified output mode and computation method. / 指定した出力範囲と計算方式で 2 つのベクトルの線形畳み込みを計算します。 |
| `static VectorSpan<T> Convolve<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, ConvolutionMode mode, VectorSpan<T> destination)  (+2 overloads)` | Computes the linear convolution of two vectors with the specified output mode and writes the result to `destination`. / 指定した出力範囲で 2 つのベクトルの線形畳み込みを計算し、`destination` に書き込みます。 |
| `static VectorSpan<T> Convolve<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, ConvolutionMode mode, ConvolutionMethod method, VectorSpan<T> destination)  (+2 overloads)` | Computes the linear convolution of two vectors with the specified mode and method and writes the result to `destination`. / 指定した出力範囲と計算方式で 2 つのベクトルの線形畳み込みを計算し、`destination` に書き込みます。 |
| `static Vector<T> Correlate<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)  (+2 overloads)` | Computes the cross-correlation of two vectors using `Full` and `Auto`. / 2 つのベクトルの相互相関を `Full` および `Auto` で計算します。 |
| `static Vector<T> Correlate<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, ConvolutionMode mode)  (+2 overloads)` | Computes the cross-correlation of two vectors with the specified output mode. / 指定した出力範囲で 2 つのベクトルの相互相関を計算します。 |
| `static VectorSpan<T> Correlate<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, VectorSpan<T> destination)  (+2 overloads)` | Computes the cross-correlation of two vectors using `Full` and writes the result to `destination`. / 2 つのベクトルの相互相関を `Full` で計算し、`destination` に書き込みます。 |
| `static Vector<T> Correlate<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, ConvolutionMode mode, ConvolutionMethod method)  (+2 overloads)` | Computes the cross-correlation of two vectors with the specified output mode and computation method. / 指定した出力範囲と計算方式で 2 つのベクトルの相互相関を計算します。 |
| `static VectorSpan<T> Correlate<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, ConvolutionMode mode, VectorSpan<T> destination)  (+2 overloads)` | Computes the cross-correlation of two vectors with the specified output mode and writes the result to `destination`. / 指定した出力範囲で 2 つのベクトルの相互相関を計算し、`destination` に書き込みます。 |
| `static VectorSpan<T> Correlate<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, ConvolutionMode mode, ConvolutionMethod method, VectorSpan<T> destination)  (+2 overloads)` | Computes the cross-correlation of two vectors with the specified mode and method and writes the result to `destination`. / 指定した出力範囲と計算方式で 2 つのベクトルの相互相関を計算し、`destination` に書き込みます。 |
| `static Matrix<T> RowAutoCorrelations<T>(ReadOnlyMatrixSpan<T> x, int maxLag)  (+2 overloads)` | Computes the unnormalized autocorrelation independently for each row of a matrix. / 行列の各行ごとに非正規化自己相関を計算します。 |
| `static Matrix<T> RowAutoCorrelations<T>(ReadOnlyMatrixSpan<T> x, int maxLag, ConvolutionMethod method)  (+2 overloads)` | Computes the unnormalized autocorrelation independently for each row of a matrix using the specified method. / 指定した計算方式で行列の各行ごとに非正規化自己相関を計算します。 |
| `static MatrixSpan<T> RowAutoCorrelations<T>(ReadOnlyMatrixSpan<T> x, int maxLag, MatrixSpan<T> destination)  (+2 overloads)` | Computes the unnormalized autocorrelation independently for each row of a matrix and writes the result to `destination`. / 行列の各行ごとに非正規化自己相関を計算し、`destination` に書き込みます。 |
| `static MatrixSpan<T> RowAutoCorrelations<T>(ReadOnlyMatrixSpan<T> x, int maxLag, ConvolutionMethod method, MatrixSpan<T> destination)  (+2 overloads)` | Computes the unnormalized autocorrelation independently for each row of a matrix using the specified method and writes the result to `destination`. / 指定した計算方式で行列の各行ごとに非正規化自己相関を計算し、`destination` に書き込みます。 |

