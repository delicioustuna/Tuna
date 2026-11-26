using System.Buffers;
using System.ComponentModel;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Tuna.SignalProcessing;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

/// <summary>
/// Specifies the output length mode for convolution and cross-correlation.
/// <para>畳み込みおよび相互相関の出力範囲を指定します。</para>
/// </summary>
public enum ConvolutionMode
{
    /// <summary>
    /// All overlapping positions, producing length <c>M + N - 1</c>.
    /// <para>重なりのあるすべての位置を出力します（長さ <c>M + N − 1</c>）。</para>
    /// </summary>
    Full,

    /// <summary>
    /// The central portion of <see cref="Full"/> with length <c>max(M, N)</c>.
    /// <para><see cref="Full"/> の中央部分を長い方の入力と同じ長さ <c>max(M, N)</c> で出力します。</para>
    /// </summary>
    Same,

    /// <summary>
    /// Only positions where the shorter input completely overlaps the longer input, producing length <c>max(M, N) - min(M, N) + 1</c>.
    /// <para>短い方の入力が長い方に完全に重なる位置だけを出力します（長さ <c>max(M, N) − min(M, N) + 1</c>）。</para>
    /// </summary>
    Valid,
}

/// <summary>
/// Specifies the algorithm used for convolution, cross-correlation, and autocorrelation.
/// <para>畳み込み・相互相関・自己相関の計算方式を指定します。</para>
/// </summary>
public enum ConvolutionMethod
{
    /// <summary>
    /// Deterministically selects FFT for long <see cref="double"/> and <see cref="Complex"/> inputs, and direct summation otherwise.
    /// <para>入力長に応じて方式を自動的に選択します。<see cref="double"/> と <see cref="Complex"/> の長い入力では FFT、それ以外では直接計算を使用します。</para>
    /// </summary>
    Auto,

    /// <summary>
    /// Computes the sum directly in O(M·N) time, preserving exact arithmetic for integer types and relative accuracy on small tails.
    /// <para>定義どおりに直接和を計算します（O(M·N)）。整数型では厳密な結果が得られ、裾の小さな値でも相対精度を保ちます。</para>
    /// </summary>
    Direct,

    /// <summary>
    /// Computes via zero-padded FFT in O((M + N) log(M + N)) time. Supported only for <see cref="double"/> and <see cref="Complex"/>.
    /// <para>ゼロパディングした FFT で計算します（O((M + N) log(M + N))）。<see cref="double"/> と <see cref="Complex"/> のみ対応します。</para>
    /// </summary>
    Fourier,
}

/// <summary>
/// Provides signal processing extension members for <see cref="XMath"/>, including convolution, cross-correlation, and autocorrelation.
/// <para>畳み込み・相互相関・自己相関など、<see cref="XMath"/> に対する信号処理の拡張メンバーを提供します。</para>
/// </summary>
[EditorBrowsable(EditorBrowsableState.Never)]
public static partial class XMathExtensions
{
    private const int DirectConvolutionMaximumShortLength = 32;

    private const long DirectConvolutionMaximumWork = 16384;

    extension(XMath)
    {
        /// <summary>
        /// Computes the linear convolution of two vectors using <see cref="ConvolutionMode.Full"/> and <see cref="ConvolutionMethod.Auto"/>.
        /// <para>2 つのベクトルの線形畳み込みを <see cref="ConvolutionMode.Full"/> および <see cref="ConvolutionMethod.Auto"/> で計算します。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The first input signal. / 第 1 の入力信号。</param>
        /// <param name="y">The second input signal. / 第 2 の入力信号。</param>
        /// <returns>A new vector of length <c>M + N - 1</c> containing <c>(x * y)[k] = Σ_j x[j] · y[k - j]</c>. / <c>(x ∗ y)[k] = Σ_j x[j] · y[k − j]</c> を格納した長さ <c>M + N − 1</c> の新しいベクトル。</returns>
        /// <exception cref="ArgumentException"><paramref name="x"/> or <paramref name="y"/> is empty. / <paramref name="x"/> または <paramref name="y"/> が空の場合。</exception>
        [InferenceOverloads]
        public static Vector<T> Convolve<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y) where T : unmanaged, INumberBase<T> =>
            Convolve(x, y, ConvolutionMode.Full, ConvolutionMethod.Auto);

        /// <summary>
        /// Computes the linear convolution of two vectors with the specified output mode.
        /// <para>指定した出力範囲で 2 つのベクトルの線形畳み込みを計算します。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The first input signal. / 第 1 の入力信号。</param>
        /// <param name="y">The second input signal. / 第 2 の入力信号。</param>
        /// <param name="mode">The output length mode. / 出力範囲を指定するモード。</param>
        /// <returns>A new vector containing the convolution with the length determined by <paramref name="mode"/>. / <paramref name="mode"/> に応じた長さで畳み込みを格納した新しいベクトル。</returns>
        /// <exception cref="ArgumentException"><paramref name="x"/> or <paramref name="y"/> is empty. / <paramref name="x"/> または <paramref name="y"/> が空の場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> is not a valid <see cref="ConvolutionMode"/> value. / <paramref name="mode"/> が無効な <see cref="ConvolutionMode"/> 値の場合。</exception>
        [InferenceOverloads]
        public static Vector<T> Convolve<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, ConvolutionMode mode) where T : unmanaged, INumberBase<T> =>
            Convolve(x, y, mode, ConvolutionMethod.Auto);

        /// <summary>
        /// Computes the linear convolution of two vectors with the specified output mode and computation method.
        /// <para>指定した出力範囲と計算方式で 2 つのベクトルの線形畳み込みを計算します。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The first input signal. / 第 1 の入力信号。</param>
        /// <param name="y">The second input signal. / 第 2 の入力信号。</param>
        /// <param name="mode">The output length mode. / 出力範囲を指定するモード。</param>
        /// <param name="method">The computation method. <see cref="ConvolutionMethod.Fourier"/> supports <see cref="double"/> and <see cref="Complex"/>. / 計算方式。<see cref="ConvolutionMethod.Fourier"/> は <see cref="double"/> と <see cref="Complex"/> に対応します。</param>
        /// <returns>A new vector containing the convolution with the length determined by <paramref name="mode"/>. / <paramref name="mode"/> に応じた長さで畳み込みを格納した新しいベクトル。</returns>
        /// <remarks>
        /// <para>FFT-based convolution introduces rounding error proportional to <c>‖x‖₂ · ‖y‖₂</c>; use <see cref="ConvolutionMethod.Direct"/> when relative accuracy on tiny tail entries is required.</para>
        /// <para>FFT による計算では誤差が <c>‖x‖₂ · ‖y‖₂</c> に比例するため、極端に小さな裾の要素に相対精度が必要な場合は <see cref="ConvolutionMethod.Direct"/> を使用してください。</para>
        /// </remarks>
        /// <exception cref="ArgumentException"><paramref name="x"/> or <paramref name="y"/> is empty, or <paramref name="method"/> is <see cref="ConvolutionMethod.Fourier"/> for an unsupported type <typeparamref name="T"/>. / <paramref name="x"/> または <paramref name="y"/> が空の場合、あるいは未対応の型に対して <see cref="ConvolutionMethod.Fourier"/> を指定した場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> or <paramref name="method"/> is invalid. / <paramref name="mode"/> または <paramref name="method"/> が無効な場合。</exception>
        [InferenceOverloads]
        public static Vector<T> Convolve<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, ConvolutionMode mode, ConvolutionMethod method)
            where T : unmanaged, INumberBase<T>
        {
            var result = Vector<T>.Uninitialized(ConvolutionLength(x.Length, y.Length, mode));
            Convolve(x, y, mode, method, result);
            return result;
        }

        /// <summary>
        /// Computes the linear convolution of two vectors using <see cref="ConvolutionMode.Full"/> and writes the result to <paramref name="destination"/>.
        /// <para>2 つのベクトルの線形畳み込みを <see cref="ConvolutionMode.Full"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The first input signal. / 第 1 の入力信号。</param>
        /// <param name="y">The second input signal. / 第 2 の入力信号。</param>
        /// <param name="destination">The non-overlapping destination span of length <c>M + N - 1</c>. / 入力と重ならない長さ <c>M + N − 1</c> の出力先。</param>
        /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
        /// <exception cref="ArgumentException"><paramref name="x"/> or <paramref name="y"/> is empty, <paramref name="destination"/> has an invalid length, or <paramref name="destination"/> overlaps an input. / 入力が空の場合、出力先の長さが合わない場合、または出力先が入力と重なる場合。</exception>
        [InferenceOverloads]
        public static VectorSpan<T> Convolve<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, VectorSpan<T> destination)
            where T : unmanaged, INumberBase<T> =>
            Convolve(x, y, ConvolutionMode.Full, ConvolutionMethod.Auto, destination);

        /// <summary>
        /// Computes the linear convolution of two vectors with the specified output mode and writes the result to <paramref name="destination"/>.
        /// <para>指定した出力範囲で 2 つのベクトルの線形畳み込みを計算し、<paramref name="destination"/> に書き込みます。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The first input signal. / 第 1 の入力信号。</param>
        /// <param name="y">The second input signal. / 第 2 の入力信号。</param>
        /// <param name="mode">The output length mode. / 出力範囲を指定するモード。</param>
        /// <param name="destination">The non-overlapping destination span sized for <paramref name="mode"/>. / 入力と重ならない、<paramref name="mode"/> に対応する長さの出力先。</param>
        /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
        /// <exception cref="ArgumentException"><paramref name="x"/> or <paramref name="y"/> is empty, <paramref name="destination"/> has an invalid length, or <paramref name="destination"/> overlaps an input. / 入力が空の場合、出力先の長さが合わない場合、または出力先が入力と重なる場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> is not a valid <see cref="ConvolutionMode"/> value. / <paramref name="mode"/> が無効な <see cref="ConvolutionMode"/> 値の場合。</exception>
        [InferenceOverloads]
        public static VectorSpan<T> Convolve<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, ConvolutionMode mode, VectorSpan<T> destination)
            where T : unmanaged, INumberBase<T> =>
            Convolve(x, y, mode, ConvolutionMethod.Auto, destination);

        /// <summary>
        /// Computes the linear convolution of two vectors with the specified mode and method and writes the result to <paramref name="destination"/>.
        /// <para>指定した出力範囲と計算方式で 2 つのベクトルの線形畳み込みを計算し、<paramref name="destination"/> に書き込みます。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The first input signal. / 第 1 の入力信号。</param>
        /// <param name="y">The second input signal. / 第 2 の入力信号。</param>
        /// <param name="mode">The output length mode. / 出力範囲を指定するモード。</param>
        /// <param name="method">The computation method. <see cref="ConvolutionMethod.Fourier"/> supports <see cref="double"/> and <see cref="Complex"/>. / 計算方式。<see cref="ConvolutionMethod.Fourier"/> は <see cref="double"/> と <see cref="Complex"/> に対応します。</param>
        /// <param name="destination">The non-overlapping destination span sized for <paramref name="mode"/>. / 入力と重ならない、<paramref name="mode"/> に対応する長さの出力先。</param>
        /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
        /// <exception cref="ArgumentException"><paramref name="x"/> or <paramref name="y"/> is empty, <paramref name="destination"/> has an invalid length or overlaps an input, or <paramref name="method"/> is <see cref="ConvolutionMethod.Fourier"/> for an unsupported type <typeparamref name="T"/>. / 入力が空の場合、出力先の長さが合わないか入力と重なる場合、または未対応の型に対して <see cref="ConvolutionMethod.Fourier"/> を指定した場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> or <paramref name="method"/> is invalid. / <paramref name="mode"/> または <paramref name="method"/> が無効な場合。</exception>
        [InferenceOverloads]
        public static VectorSpan<T> Convolve<T>(
            ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, ConvolutionMode mode, ConvolutionMethod method, VectorSpan<T> destination)
            where T : unmanaged, INumberBase<T>
        {
            RequireLength(ConvolutionLength(x.Length, y.Length, mode), destination.Length);
            int start = mode switch
            {
                ConvolutionMode.Full => 0,
                ConvolutionMode.Same => (Math.Min(x.Length, y.Length) - 1) / 2,
                ConvolutionMode.Valid => Math.Min(x.Length, y.Length) - 1,
                _ => throw new ArgumentOutOfRangeException(nameof(mode)),
            };
            ConvolveCore(x, y, correlate: false, start, method, destination);
            return destination;
        }

        /// <summary>
        /// Computes the cross-correlation of two vectors using <see cref="ConvolutionMode.Full"/> and <see cref="ConvolutionMethod.Auto"/>.
        /// <para>2 つのベクトルの相互相関を <see cref="ConvolutionMode.Full"/> および <see cref="ConvolutionMethod.Auto"/> で計算します。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The first input signal. / 第 1 の入力信号。</param>
        /// <param name="y">The second input signal (conjugated when <typeparamref name="T"/> is <see cref="Complex"/>). / 第 2 の入力信号（<typeparamref name="T"/> が <see cref="Complex"/> の場合は共役を取ります）。</param>
        /// <returns>A new vector of length <c>M + N - 1</c> containing <c>c[k] = Σ_t x[t + k] · conj(y[t])</c>. / <c>c[k] = Σ_t x[t + k] · conj(y[t])</c> を格納した長さ <c>M + N − 1</c> の新しいベクトル。</returns>
        /// <exception cref="ArgumentException"><paramref name="x"/> or <paramref name="y"/> is empty. / <paramref name="x"/> または <paramref name="y"/> が空の場合。</exception>
        [InferenceOverloads]
        public static Vector<T> Correlate<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y) where T : unmanaged, INumberBase<T> =>
            Correlate(x, y, ConvolutionMode.Full, ConvolutionMethod.Auto);

        /// <summary>
        /// Computes the cross-correlation of two vectors with the specified output mode.
        /// <para>指定した出力範囲で 2 つのベクトルの相互相関を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The first input signal. / 第 1 の入力信号。</param>
        /// <param name="y">The second input signal. / 第 2 の入力信号。</param>
        /// <param name="mode">The output length mode. / 出力範囲を指定するモード。</param>
        /// <returns>A new vector containing the cross-correlation with the length determined by <paramref name="mode"/>. / <paramref name="mode"/> に応じた長さで相互相関を格納した新しいベクトル。</returns>
        /// <exception cref="ArgumentException"><paramref name="x"/> or <paramref name="y"/> is empty. / <paramref name="x"/> または <paramref name="y"/> が空の場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> is not a valid <see cref="ConvolutionMode"/> value. / <paramref name="mode"/> が無効な <see cref="ConvolutionMode"/> 値の場合。</exception>
        [InferenceOverloads]
        public static Vector<T> Correlate<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, ConvolutionMode mode) where T : unmanaged, INumberBase<T> =>
            Correlate(x, y, mode, ConvolutionMethod.Auto);

        /// <summary>
        /// Computes the cross-correlation of two vectors with the specified output mode and computation method.
        /// <para>指定した出力範囲と計算方式で 2 つのベクトルの相互相関を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The first input signal. / 第 1 の入力信号。</param>
        /// <param name="y">The second input signal. / 第 2 の入力信号。</param>
        /// <param name="mode">The output length mode. / 出力範囲を指定するモード。</param>
        /// <param name="method">The computation method. <see cref="ConvolutionMethod.Fourier"/> supports <see cref="double"/> and <see cref="Complex"/>. / 計算方式。<see cref="ConvolutionMethod.Fourier"/> は <see cref="double"/> と <see cref="Complex"/> に対応します。</param>
        /// <returns>A new vector containing the cross-correlation with the length determined by <paramref name="mode"/>. / <paramref name="mode"/> に応じた長さで相互相関を格納した新しいベクトル。</returns>
        /// <remarks>
        /// <para>In <see cref="ConvolutionMode.Full"/>, element <c>i</c> corresponds to lag <c>k = i - (N - 1)</c>. In <see cref="ConvolutionMode.Same"/>, when <c>x.Length &lt; y.Length</c> and <c>x.Length</c> is even, the slice starts at index <c>x.Length / 2</c> (matching NumPy's <c>correlate</c>, shifted by one from <c>Convolve(x, conj(reverse(y)), Same)</c>).</para>
        /// <para><see cref="ConvolutionMode.Full"/> の第 <c>i</c> 要素はラグ <c>k = i − (N − 1)</c> の値です。<see cref="ConvolutionMode.Same"/> では <c>x.Length &lt; y.Length</c> かつ <c>x.Length</c> が偶数のとき開始位置を <c>x.Length / 2</c> とします（NumPy の <c>correlate</c> と同じ規則で、<c>Convolve(x, conj(reverse(y)), Same)</c> より 1 要素後ろから切り出します）。</para>
        /// </remarks>
        /// <exception cref="ArgumentException"><paramref name="x"/> or <paramref name="y"/> is empty, or <paramref name="method"/> is <see cref="ConvolutionMethod.Fourier"/> for an unsupported type <typeparamref name="T"/>. / <paramref name="x"/> または <paramref name="y"/> が空の場合、あるいは未対応の型に対して <see cref="ConvolutionMethod.Fourier"/> を指定した場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> or <paramref name="method"/> is invalid. / <paramref name="mode"/> または <paramref name="method"/> が無効な場合。</exception>
        [InferenceOverloads]
        public static Vector<T> Correlate<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, ConvolutionMode mode, ConvolutionMethod method)
            where T : unmanaged, INumberBase<T>
        {
            var result = Vector<T>.Uninitialized(ConvolutionLength(x.Length, y.Length, mode));
            Correlate(x, y, mode, method, result);
            return result;
        }

        /// <summary>
        /// Computes the cross-correlation of two vectors using <see cref="ConvolutionMode.Full"/> and writes the result to <paramref name="destination"/>.
        /// <para>2 つのベクトルの相互相関を <see cref="ConvolutionMode.Full"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The first input signal. / 第 1 の入力信号。</param>
        /// <param name="y">The second input signal. / 第 2 の入力信号。</param>
        /// <param name="destination">The non-overlapping destination span of length <c>M + N - 1</c>. / 入力と重ならない長さ <c>M + N − 1</c> の出力先。</param>
        /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
        /// <exception cref="ArgumentException"><paramref name="x"/> or <paramref name="y"/> is empty, <paramref name="destination"/> has an invalid length, or <paramref name="destination"/> overlaps an input. / 入力が空の場合、出力先の長さが合わない場合、または出力先が入力と重なる場合。</exception>
        [InferenceOverloads]
        public static VectorSpan<T> Correlate<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, VectorSpan<T> destination)
            where T : unmanaged, INumberBase<T> =>
            Correlate(x, y, ConvolutionMode.Full, ConvolutionMethod.Auto, destination);

        /// <summary>
        /// Computes the cross-correlation of two vectors with the specified output mode and writes the result to <paramref name="destination"/>.
        /// <para>指定した出力範囲で 2 つのベクトルの相互相関を計算し、<paramref name="destination"/> に書き込みます。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The first input signal. / 第 1 の入力信号。</param>
        /// <param name="y">The second input signal. / 第 2 の入力信号。</param>
        /// <param name="mode">The output length mode. / 出力範囲を指定するモード。</param>
        /// <param name="destination">The non-overlapping destination span sized for <paramref name="mode"/>. / 入力と重ならない、<paramref name="mode"/> に対応する長さの出力先。</param>
        /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
        /// <exception cref="ArgumentException"><paramref name="x"/> or <paramref name="y"/> is empty, <paramref name="destination"/> has an invalid length, or <paramref name="destination"/> overlaps an input. / 入力が空の場合、出力先の長さが合わない場合、または出力先が入力と重なる場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> is not a valid <see cref="ConvolutionMode"/> value. / <paramref name="mode"/> が無効な <see cref="ConvolutionMode"/> 値の場合。</exception>
        [InferenceOverloads]
        public static VectorSpan<T> Correlate<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, ConvolutionMode mode, VectorSpan<T> destination)
            where T : unmanaged, INumberBase<T> =>
            Correlate(x, y, mode, ConvolutionMethod.Auto, destination);

        /// <summary>
        /// Computes the cross-correlation of two vectors with the specified mode and method and writes the result to <paramref name="destination"/>.
        /// <para>指定した出力範囲と計算方式で 2 つのベクトルの相互相関を計算し、<paramref name="destination"/> に書き込みます。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The first input signal. / 第 1 の入力信号。</param>
        /// <param name="y">The second input signal. / 第 2 の入力信号。</param>
        /// <param name="mode">The output length mode. / 出力範囲を指定するモード。</param>
        /// <param name="method">The computation method. <see cref="ConvolutionMethod.Fourier"/> supports <see cref="double"/> and <see cref="Complex"/>. / 計算方式。<see cref="ConvolutionMethod.Fourier"/> は <see cref="double"/> と <see cref="Complex"/> に対応します。</param>
        /// <param name="destination">The non-overlapping destination span sized for <paramref name="mode"/>. / 入力と重ならない、<paramref name="mode"/> に対応する長さの出力先。</param>
        /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
        /// <exception cref="ArgumentException"><paramref name="x"/> or <paramref name="y"/> is empty, <paramref name="destination"/> has an invalid length or overlaps an input, or <paramref name="method"/> is <see cref="ConvolutionMethod.Fourier"/> for an unsupported type <typeparamref name="T"/>. / 入力が空の場合、出力先の長さが合わないか入力と重なる場合、または未対応の型に対して <see cref="ConvolutionMethod.Fourier"/> を指定した場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="mode"/> or <paramref name="method"/> is invalid. / <paramref name="mode"/> または <paramref name="method"/> が無効な場合。</exception>
        [InferenceOverloads]
        public static VectorSpan<T> Correlate<T>(
            ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, ConvolutionMode mode, ConvolutionMethod method, VectorSpan<T> destination)
            where T : unmanaged, INumberBase<T>
        {
            int m = x.Length, n = y.Length;
            RequireLength(ConvolutionLength(m, n, mode), destination.Length);
            int start = mode switch
            {
                ConvolutionMode.Full => 0,
                ConvolutionMode.Same => m >= n ? (n - 1) / 2 : m / 2,
                ConvolutionMode.Valid => Math.Min(m, n) - 1,
                _ => throw new ArgumentOutOfRangeException(nameof(mode)),
            };
            ConvolveCore(x, y, correlate: true, start, method, destination);
            return destination;
        }

        /// <summary>
        /// Computes the unnormalized autocorrelation of a vector for lags <c>0</c> through <paramref name="maxLag"/>.
        /// <para>ベクトルの非正規化自己相関をラグ <c>0</c> から <paramref name="maxLag"/> まで計算します。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The input signal. / 入力信号。</param>
        /// <param name="maxLag">The maximum lag, in the range <c>[0, x.Length - 1]</c>. / 最大ラグ（<c>0</c> 以上、入力信号の長さ未満）。</param>
        /// <returns>A new vector of length <c>maxLag + 1</c> containing <c>r[k] = Σ_t x[t + k] · conj(x[t])</c>. / <c>r[k] = Σ_t x[t + k] · conj(x[t])</c> を格納した長さ <c>maxLag + 1</c> の新しいベクトル。</returns>
        /// <remarks>
        /// <para>Computes the raw sum without subtracting the mean or dividing by the signal length.</para>
        /// <para>平均の差し引きや信号長による除算は行わず、定義どおりの和を返します。</para>
        /// </remarks>
        /// <exception cref="ArgumentException"><paramref name="x"/> is empty. / <paramref name="x"/> が空の場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxLag"/> is negative or greater than or equal to <paramref name="x"/>.<c>Length</c>. / <paramref name="maxLag"/> が負または入力信号の長さ以上の場合。</exception>
        [InferenceOverloads]
        public static Vector<T> AutoCorrelate<T>(ReadOnlyVectorSpan<T> x, int maxLag) where T : unmanaged, INumberBase<T> =>
            AutoCorrelate(x, maxLag, ConvolutionMethod.Auto);

        /// <summary>
        /// Computes the unnormalized autocorrelation of a vector for lags <c>0</c> through <paramref name="maxLag"/> using the specified method.
        /// <para>指定した計算方式でベクトルの非正規化自己相関をラグ <c>0</c> から <paramref name="maxLag"/> まで計算します。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The input signal. / 入力信号。</param>
        /// <param name="maxLag">The maximum lag, in the range <c>[0, x.Length - 1]</c>. / 最大ラグ（<c>0</c> 以上、入力信号の長さ未満）。</param>
        /// <param name="method">The computation method. <see cref="ConvolutionMethod.Fourier"/> supports <see cref="double"/> and <see cref="Complex"/>. / 計算方式。<see cref="ConvolutionMethod.Fourier"/> は <see cref="double"/> と <see cref="Complex"/> に対応します。</param>
        /// <returns>A new vector of length <c>maxLag + 1</c> containing <c>r[k] = Σ_t x[t + k] · conj(x[t])</c>. / <c>r[k] = Σ_t x[t + k] · conj(x[t])</c> を格納した長さ <c>maxLag + 1</c> の新しいベクトル。</returns>
        /// <exception cref="ArgumentException"><paramref name="x"/> is empty, or <paramref name="method"/> is <see cref="ConvolutionMethod.Fourier"/> for an unsupported type <typeparamref name="T"/>. / <paramref name="x"/> が空の場合、または未対応の型に対して <see cref="ConvolutionMethod.Fourier"/> を指定した場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxLag"/> is out of range or <paramref name="method"/> is invalid. / <paramref name="maxLag"/> が範囲外の場合、または <paramref name="method"/> が無効な場合。</exception>
        [InferenceOverloads]
        public static Vector<T> AutoCorrelate<T>(ReadOnlyVectorSpan<T> x, int maxLag, ConvolutionMethod method) where T : unmanaged, INumberBase<T>
        {
            RequireMaxLag(x.Length, maxLag);
            var result = Vector<T>.Uninitialized(maxLag + 1);
            AutoCorrelate(x, maxLag, method, result);
            return result;
        }

        /// <summary>
        /// Computes the unnormalized autocorrelation of a vector and writes the result to <paramref name="destination"/>.
        /// <para>ベクトルの非正規化自己相関を計算し、<paramref name="destination"/> に書き込みます。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The input signal. / 入力信号。</param>
        /// <param name="maxLag">The maximum lag, in the range <c>[0, x.Length - 1]</c>. / 最大ラグ（<c>0</c> 以上、入力信号の長さ未満）。</param>
        /// <param name="destination">The non-overlapping destination span of length <c>maxLag + 1</c>. / 入力と重ならない長さ <c>maxLag + 1</c> の出力先。</param>
        /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
        /// <exception cref="ArgumentException"><paramref name="x"/> is empty, <paramref name="destination"/> has an invalid length, or <paramref name="destination"/> overlaps <paramref name="x"/>. / 入力が空の場合、出力先の長さが合わない場合、または出力先が入力と重なる場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxLag"/> is out of range. / <paramref name="maxLag"/> が範囲外の場合。</exception>
        [InferenceOverloads]
        public static VectorSpan<T> AutoCorrelate<T>(ReadOnlyVectorSpan<T> x, int maxLag, VectorSpan<T> destination) where T : unmanaged, INumberBase<T> =>
            AutoCorrelate(x, maxLag, ConvolutionMethod.Auto, destination);

        /// <summary>
        /// Computes the unnormalized autocorrelation of a vector using the specified method and writes the result to <paramref name="destination"/>.
        /// <para>指定した計算方式でベクトルの非正規化自己相関を計算し、<paramref name="destination"/> に書き込みます。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The input signal. / 入力信号。</param>
        /// <param name="maxLag">The maximum lag, in the range <c>[0, x.Length - 1]</c>. / 最大ラグ（<c>0</c> 以上、入力信号の長さ未満）。</param>
        /// <param name="method">The computation method. <see cref="ConvolutionMethod.Fourier"/> supports <see cref="double"/> and <see cref="Complex"/>. / 計算方式。<see cref="ConvolutionMethod.Fourier"/> は <see cref="double"/> と <see cref="Complex"/> に対応します。</param>
        /// <param name="destination">The non-overlapping destination span of length <c>maxLag + 1</c>. / 入力と重ならない長さ <c>maxLag + 1</c> の出力先。</param>
        /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
        /// <exception cref="ArgumentException"><paramref name="x"/> is empty, <paramref name="destination"/> has an invalid length or overlaps <paramref name="x"/>, or <paramref name="method"/> is <see cref="ConvolutionMethod.Fourier"/> for an unsupported type <typeparamref name="T"/>. / 入力が空の場合、出力先の長さが合わないか入力と重なる場合、または未対応の型に対して <see cref="ConvolutionMethod.Fourier"/> を指定した場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxLag"/> is out of range or <paramref name="method"/> is invalid. / <paramref name="maxLag"/> が範囲外の場合、または <paramref name="method"/> が無効な場合。</exception>
        [InferenceOverloads]
        public static VectorSpan<T> AutoCorrelate<T>(ReadOnlyVectorSpan<T> x, int maxLag, ConvolutionMethod method, VectorSpan<T> destination)
            where T : unmanaged, INumberBase<T>
        {
            RequireMaxLag(x.Length, maxLag);
            RequireLength(maxLag + 1, destination.Length);
            RequireDisjoint(x, destination);
            int n = x.Length;
            if (UseFourier<T>(method, (long)n * (maxLag + 1), Math.Min(n, maxLag + 1)))
            {
                AutoCorrelateFourier(x, maxLag, destination);
            }
            else
            {
                for (int k = 0; k <= maxLag; k++)
                {
                    T sum = T.Zero;
                    for (int t = 0; t + k < n; t++)
                    {
                        sum += x[t + k] * Conjugate(x[t]);
                    }

                    destination[k] = sum;
                }
            }

            return destination;
        }

        /// <summary>
        /// Computes the unnormalized autocorrelation independently for each column of a matrix.
        /// <para>行列の各列ごとに非正規化自己相関を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The input matrix whose columns are signals. / 各列を入力信号とする行列。</param>
        /// <param name="maxLag">The maximum lag, in the range <c>[0, x.Rows - 1]</c>. / 最大ラグ（<c>0</c> 以上、行数未満）。</param>
        /// <returns>A new <c>(maxLag + 1) × x.Columns</c> matrix of column autocorrelations. / 各列の自己相関を格納した <c>(maxLag + 1) × x.Columns</c> の新しい行列。</returns>
        /// <exception cref="ArgumentException"><paramref name="x"/> has zero rows. / <paramref name="x"/> の行数が 0 の場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxLag"/> is out of range. / <paramref name="maxLag"/> が範囲外の場合。</exception>
        [InferenceOverloads]
        public static Matrix<T> ColumnAutoCorrelations<T>(ReadOnlyMatrixSpan<T> x, int maxLag) where T : unmanaged, INumberBase<T> =>
            ColumnAutoCorrelations(x, maxLag, ConvolutionMethod.Auto);

        /// <summary>
        /// Computes the unnormalized autocorrelation independently for each column of a matrix using the specified method.
        /// <para>指定した計算方式で行列の各列ごとに非正規化自己相関を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The input matrix whose columns are signals. / 各列を入力信号とする行列。</param>
        /// <param name="maxLag">The maximum lag, in the range <c>[0, x.Rows - 1]</c>. / 最大ラグ（<c>0</c> 以上、行数未満）。</param>
        /// <param name="method">The computation method. <see cref="ConvolutionMethod.Fourier"/> supports <see cref="double"/> and <see cref="Complex"/>. / 計算方式。<see cref="ConvolutionMethod.Fourier"/> は <see cref="double"/> と <see cref="Complex"/> に対応します。</param>
        /// <returns>A new <c>(maxLag + 1) × x.Columns</c> matrix of column autocorrelations. / 各列の自己相関を格納した <c>(maxLag + 1) × x.Columns</c> の新しい行列。</returns>
        /// <exception cref="ArgumentException"><paramref name="x"/> has zero rows, or <paramref name="method"/> is <see cref="ConvolutionMethod.Fourier"/> for an unsupported type <typeparamref name="T"/>. / <paramref name="x"/> の行数が 0 の場合、または未対応の型に対して <see cref="ConvolutionMethod.Fourier"/> を指定した場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxLag"/> is out of range or <paramref name="method"/> is invalid. / <paramref name="maxLag"/> が範囲外の場合、または <paramref name="method"/> が無効な場合。</exception>
        [InferenceOverloads]
        public static Matrix<T> ColumnAutoCorrelations<T>(ReadOnlyMatrixSpan<T> x, int maxLag, ConvolutionMethod method) where T : unmanaged, INumberBase<T>
        {
            RequireMaxLag(x.Rows, maxLag);
            var result = Matrix<T>.Uninitialized(maxLag + 1, x.Columns);
            ColumnAutoCorrelations(x, maxLag, method, result);
            return result;
        }

        /// <summary>
        /// Computes the unnormalized autocorrelation independently for each column of a matrix and writes the result to <paramref name="destination"/>.
        /// <para>行列の各列ごとに非正規化自己相関を計算し、<paramref name="destination"/> に書き込みます。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The input matrix whose columns are signals. / 各列を入力信号とする行列。</param>
        /// <param name="maxLag">The maximum lag, in the range <c>[0, x.Rows - 1]</c>. / 最大ラグ（<c>0</c> 以上、行数未満）。</param>
        /// <param name="destination">The non-overlapping destination matrix of shape <c>(maxLag + 1) × x.Columns</c>. / 入力と重ならない <c>(maxLag + 1) × x.Columns</c> の出力先。</param>
        /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
        /// <exception cref="ArgumentException"><paramref name="x"/> has zero rows, or <paramref name="destination"/> has an invalid shape or overlaps <paramref name="x"/>. / 入力の行数が 0 の場合、または出力先の形状が合わないか入力と重なる場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxLag"/> is out of range. / <paramref name="maxLag"/> が範囲外の場合。</exception>
        [InferenceOverloads]
        public static MatrixSpan<T> ColumnAutoCorrelations<T>(ReadOnlyMatrixSpan<T> x, int maxLag, MatrixSpan<T> destination)
            where T : unmanaged, INumberBase<T> =>
            ColumnAutoCorrelations(x, maxLag, ConvolutionMethod.Auto, destination);

        /// <summary>
        /// Computes the unnormalized autocorrelation independently for each column of a matrix using the specified method and writes the result to <paramref name="destination"/>.
        /// <para>指定した計算方式で行列の各列ごとに非正規化自己相関を計算し、<paramref name="destination"/> に書き込みます。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The input matrix whose columns are signals. / 各列を入力信号とする行列。</param>
        /// <param name="maxLag">The maximum lag, in the range <c>[0, x.Rows - 1]</c>. / 最大ラグ（<c>0</c> 以上、行数未満）。</param>
        /// <param name="method">The computation method. <see cref="ConvolutionMethod.Fourier"/> supports <see cref="double"/> and <see cref="Complex"/>. / 計算方式。<see cref="ConvolutionMethod.Fourier"/> は <see cref="double"/> と <see cref="Complex"/> に対応します。</param>
        /// <param name="destination">The non-overlapping destination matrix of shape <c>(maxLag + 1) × x.Columns</c>. / 入力と重ならない <c>(maxLag + 1) × x.Columns</c> の出力先。</param>
        /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
        /// <exception cref="ArgumentException"><paramref name="x"/> has zero rows, <paramref name="destination"/> has an invalid shape or overlaps <paramref name="x"/>, or <paramref name="method"/> is <see cref="ConvolutionMethod.Fourier"/> for an unsupported type <typeparamref name="T"/>. / 入力の行数が 0 の場合、出力先の形状が合わないか入力と重なる場合、または未対応の型に対して <see cref="ConvolutionMethod.Fourier"/> を指定した場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxLag"/> is out of range or <paramref name="method"/> is invalid. / <paramref name="maxLag"/> が範囲外の場合、または <paramref name="method"/> が無効な場合。</exception>
        [InferenceOverloads]
        public static MatrixSpan<T> ColumnAutoCorrelations<T>(ReadOnlyMatrixSpan<T> x, int maxLag, ConvolutionMethod method, MatrixSpan<T> destination)
            where T : unmanaged, INumberBase<T>
        {
            RequireMaxLag(x.Rows, maxLag);
            if (destination.Rows != maxLag + 1 || destination.Columns != x.Columns)
            {
                throw new ArgumentException($"The destination must be {maxLag + 1} x {x.Columns}.", nameof(destination));
            }

            var output = destination.AsReadOnly();
            if (StridedLayout.Overlaps(
                    in x.Reference, x.Rows, x.Columns, x.RowStride, x.ColumnStride,
                    in output.Reference, output.Rows, output.Columns, output.RowStride, output.ColumnStride, out _))
            {
                throw new ArgumentException("The destination must not overlap the inputs.", nameof(destination));
            }

            for (int j = 0; j < x.Columns; j++)
            {
                AutoCorrelate(x.Column(j), maxLag, method, destination.Column(j));
            }

            return destination;
        }

        /// <summary>
        /// Computes the unnormalized autocorrelation independently for each row of a matrix.
        /// <para>行列の各行ごとに非正規化自己相関を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The input matrix whose rows are signals. / 各行を入力信号とする行列。</param>
        /// <param name="maxLag">The maximum lag, in the range <c>[0, x.Columns - 1]</c>. / 最大ラグ（<c>0</c> 以上、列数未満）。</param>
        /// <returns>A new <c>x.Rows × (maxLag + 1)</c> matrix of row autocorrelations. / 各行の自己相関を格納した <c>x.Rows × (maxLag + 1)</c> の新しい行列。</returns>
        /// <exception cref="ArgumentException"><paramref name="x"/> has zero columns. / <paramref name="x"/> の列数が 0 の場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxLag"/> is out of range. / <paramref name="maxLag"/> が範囲外の場合。</exception>
        [InferenceOverloads]
        public static Matrix<T> RowAutoCorrelations<T>(ReadOnlyMatrixSpan<T> x, int maxLag) where T : unmanaged, INumberBase<T> =>
            RowAutoCorrelations(x, maxLag, ConvolutionMethod.Auto);

        /// <summary>
        /// Computes the unnormalized autocorrelation independently for each row of a matrix using the specified method.
        /// <para>指定した計算方式で行列の各行ごとに非正規化自己相関を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The input matrix whose rows are signals. / 各行を入力信号とする行列。</param>
        /// <param name="maxLag">The maximum lag, in the range <c>[0, x.Columns - 1]</c>. / 最大ラグ（<c>0</c> 以上、列数未満）。</param>
        /// <param name="method">The computation method. <see cref="ConvolutionMethod.Fourier"/> supports <see cref="double"/> and <see cref="Complex"/>. / 計算方式。<see cref="ConvolutionMethod.Fourier"/> は <see cref="double"/> と <see cref="Complex"/> に対応します。</param>
        /// <returns>A new <c>x.Rows × (maxLag + 1)</c> matrix of row autocorrelations. / 各行の自己相関を格納した <c>x.Rows × (maxLag + 1)</c> の新しい行列。</returns>
        /// <exception cref="ArgumentException"><paramref name="x"/> has zero columns, or <paramref name="method"/> is <see cref="ConvolutionMethod.Fourier"/> for an unsupported type <typeparamref name="T"/>. / <paramref name="x"/> の列数が 0 の場合、または未対応の型に対して <see cref="ConvolutionMethod.Fourier"/> を指定した場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxLag"/> is out of range or <paramref name="method"/> is invalid. / <paramref name="maxLag"/> が範囲外の場合、または <paramref name="method"/> が無効な場合。</exception>
        [InferenceOverloads]
        public static Matrix<T> RowAutoCorrelations<T>(ReadOnlyMatrixSpan<T> x, int maxLag, ConvolutionMethod method) where T : unmanaged, INumberBase<T>
        {
            RequireMaxLag(x.Columns, maxLag);
            var result = Matrix<T>.Uninitialized(x.Rows, maxLag + 1);
            RowAutoCorrelations(x, maxLag, method, result);
            return result;
        }

        /// <summary>
        /// Computes the unnormalized autocorrelation independently for each row of a matrix and writes the result to <paramref name="destination"/>.
        /// <para>行列の各行ごとに非正規化自己相関を計算し、<paramref name="destination"/> に書き込みます。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The input matrix whose rows are signals. / 各行を入力信号とする行列。</param>
        /// <param name="maxLag">The maximum lag, in the range <c>[0, x.Columns - 1]</c>. / 最大ラグ（<c>0</c> 以上、列数未満）。</param>
        /// <param name="destination">The non-overlapping destination matrix of shape <c>x.Rows × (maxLag + 1)</c>. / 入力と重ならない <c>x.Rows × (maxLag + 1)</c> の出力先。</param>
        /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
        /// <exception cref="ArgumentException"><paramref name="x"/> has zero columns, or <paramref name="destination"/> has an invalid shape or overlaps <paramref name="x"/>. / 入力の列数が 0 の場合、または出力先の形状が合わないか入力と重なる場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxLag"/> is out of range. / <paramref name="maxLag"/> が範囲外の場合。</exception>
        [InferenceOverloads]
        public static MatrixSpan<T> RowAutoCorrelations<T>(ReadOnlyMatrixSpan<T> x, int maxLag, MatrixSpan<T> destination)
            where T : unmanaged, INumberBase<T> =>
            RowAutoCorrelations(x, maxLag, ConvolutionMethod.Auto, destination);

        /// <summary>
        /// Computes the unnormalized autocorrelation independently for each row of a matrix using the specified method and writes the result to <paramref name="destination"/>.
        /// <para>指定した計算方式で行列の各行ごとに非正規化自己相関を計算し、<paramref name="destination"/> に書き込みます。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The input matrix whose rows are signals. / 各行を入力信号とする行列。</param>
        /// <param name="maxLag">The maximum lag, in the range <c>[0, x.Columns - 1]</c>. / 最大ラグ（<c>0</c> 以上、列数未満）。</param>
        /// <param name="method">The computation method. <see cref="ConvolutionMethod.Fourier"/> supports <see cref="double"/> and <see cref="Complex"/>. / 計算方式。<see cref="ConvolutionMethod.Fourier"/> は <see cref="double"/> と <see cref="Complex"/> に対応します。</param>
        /// <param name="destination">The non-overlapping destination matrix of shape <c>x.Rows × (maxLag + 1)</c>. / 入力と重ならない <c>x.Rows × (maxLag + 1)</c> の出力先。</param>
        /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
        /// <exception cref="ArgumentException"><paramref name="x"/> has zero columns, <paramref name="destination"/> has an invalid shape or overlaps <paramref name="x"/>, or <paramref name="method"/> is <see cref="ConvolutionMethod.Fourier"/> for an unsupported type <typeparamref name="T"/>. / 入力の列数が 0 の場合、出力先の形状が合わないか入力と重なる場合、または未対応の型に対して <see cref="ConvolutionMethod.Fourier"/> を指定した場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxLag"/> is out of range or <paramref name="method"/> is invalid. / <paramref name="maxLag"/> が範囲外の場合、または <paramref name="method"/> が無効な場合。</exception>
        [InferenceOverloads]
        public static MatrixSpan<T> RowAutoCorrelations<T>(ReadOnlyMatrixSpan<T> x, int maxLag, ConvolutionMethod method, MatrixSpan<T> destination)
            where T : unmanaged, INumberBase<T>
        {
            ColumnAutoCorrelations(x.T, maxLag, method, destination.T);
            return destination;
        }
    }

    private static int ConvolutionLength(int m, int n, ConvolutionMode mode)
    {
        if (m == 0 || n == 0)
        {
            throw new ArgumentException("The inputs of a convolution must not be empty.");
        }

        return mode switch
        {
            ConvolutionMode.Full => m + n - 1,
            ConvolutionMode.Same => Math.Max(m, n),
            ConvolutionMode.Valid => Math.Max(m, n) - Math.Min(m, n) + 1,
            _ => throw new ArgumentOutOfRangeException(nameof(mode)),
        };
    }

    private static void ConvolveCore<T>(
        ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, bool correlate, int start, ConvolutionMethod method, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        int m = x.Length, n = y.Length;
        int count = destination.Length;
        RequireDisjoint(x, destination);
        RequireDisjoint(y, destination);
        if (UseFourier<T>(method, (long)m * n, Math.Min(m, n)))
        {
            ConvolveFourier(x, y, correlate, start, destination);
            return;
        }

        // 相関は y を共役反転して畳み込む。j の昇順で加算する。
        for (int k = 0; k < count; k++)
        {
            int i = start + k;
            T sum = T.Zero;
            for (int j = Math.Max(0, i - n + 1), last = Math.Min(i, m - 1); j <= last; j++)
            {
                T other = correlate ? Conjugate(y[n - 1 - (i - j)]) : y[i - j];
                sum += x[j] * other;
            }

            destination[k] = sum;
        }
    }

    private static bool UseFourier<T>(ConvolutionMethod method, long work, int shortLength)
    {
        bool supported = typeof(T) == typeof(double) || typeof(T) == typeof(Complex);
        return method switch
        {
            ConvolutionMethod.Direct => false,
            ConvolutionMethod.Fourier => supported
                ? true
                : throw new ArgumentException($"The Fourier method supports only double and Complex, not {typeof(T).Name}.", nameof(method)),
            ConvolutionMethod.Auto => supported && shortLength > DirectConvolutionMaximumShortLength && work > DirectConvolutionMaximumWork,
            _ => throw new ArgumentOutOfRangeException(nameof(method)),
        };
    }

    private static void ConvolveFourier<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, bool correlate, int start, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        int m = x.Length, n = y.Length;
        int length = Fourier.NextFastLength(m + n - 1);
        if (typeof(T) == typeof(double))
        {
            var plan = RealFourierPlan.Create(length);
            int spectrumLength = plan.SpectrumLength;
            double[] px = ArrayPool<double>.Shared.Rent(length);
            double[] py = ArrayPool<double>.Shared.Rent(length);
            Complex[] sx = ArrayPool<Complex>.Shared.Rent(spectrumLength);
            Complex[] sy = ArrayPool<Complex>.Shared.Rent(spectrumLength);
            Pad(x, px.AsSpan(0, length), reverse: false);
            Pad(y, py.AsSpan(0, length), reverse: correlate);
            var spectrumX = new VectorSpan<Complex>(sx.AsSpan(0, spectrumLength));
            var spectrumY = new VectorSpan<Complex>(sy.AsSpan(0, spectrumLength));
            plan.Forward(new ReadOnlyVectorSpan<double>(px.AsSpan(0, length)), spectrumX);
            plan.Forward(new ReadOnlyVectorSpan<double>(py.AsSpan(0, length)), spectrumY);
            for (int k = 0; k < spectrumLength; k++)
            {
                sx[k] *= sy[k];
            }

            plan.Inverse(spectrumX, new VectorSpan<double>(px.AsSpan(0, length)));
            for (int k = 0; k < destination.Length; k++)
            {
                destination[k] = Unsafe.As<double, T>(ref px[start + k]);
            }

            ArrayPool<double>.Shared.Return(px);
            ArrayPool<double>.Shared.Return(py);
            ArrayPool<Complex>.Shared.Return(sx);
            ArrayPool<Complex>.Shared.Return(sy);
        }
        else
        {
            var plan = FourierPlan.Create(length);
            Complex[] px = ArrayPool<Complex>.Shared.Rent(length);
            Complex[] py = ArrayPool<Complex>.Shared.Rent(length);
            Pad(x, px.AsSpan(0, length), reverse: false);
            Pad(y, py.AsSpan(0, length), reverse: correlate);
            if (correlate)
            {
                for (int t = 0; t < n; t++)
                {
                    py[t] = Complex.Conjugate(py[t]);
                }
            }

            var fx = new VectorSpan<Complex>(px.AsSpan(0, length));
            var fy = new VectorSpan<Complex>(py.AsSpan(0, length));
            plan.Forward(fx, fx);
            plan.Forward(fy, fy);
            for (int k = 0; k < length; k++)
            {
                px[k] *= py[k];
            }

            plan.Inverse(fx, fx);
            for (int k = 0; k < destination.Length; k++)
            {
                destination[k] = Unsafe.As<Complex, T>(ref px[start + k]);
            }

            ArrayPool<Complex>.Shared.Return(px);
            ArrayPool<Complex>.Shared.Return(py);
        }
    }

    // L ≥ n + maxLag にゼロパディングすれば、巡回相関の 0..maxLag が線形自己相関と一致する。
    private static void AutoCorrelateFourier<T>(ReadOnlyVectorSpan<T> x, int maxLag, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        int n = x.Length;
        int length = Fourier.NextFastLength(n + maxLag);
        if (typeof(T) == typeof(double))
        {
            var plan = RealFourierPlan.Create(length);
            int spectrumLength = plan.SpectrumLength;
            double[] padded = ArrayPool<double>.Shared.Rent(length);
            Complex[] spectrum = ArrayPool<Complex>.Shared.Rent(spectrumLength);
            Pad(x, padded.AsSpan(0, length), reverse: false);
            var s = new VectorSpan<Complex>(spectrum.AsSpan(0, spectrumLength));
            plan.Forward(new ReadOnlyVectorSpan<double>(padded.AsSpan(0, length)), s);
            for (int k = 0; k < spectrumLength; k++)
            {
                Complex v = spectrum[k];
                spectrum[k] = new Complex(v.Real * v.Real + v.Imaginary * v.Imaginary, 0);
            }

            plan.Inverse(s, new VectorSpan<double>(padded.AsSpan(0, length)));
            for (int k = 0; k <= maxLag; k++)
            {
                destination[k] = Unsafe.As<double, T>(ref padded[k]);
            }

            ArrayPool<double>.Shared.Return(padded);
            ArrayPool<Complex>.Shared.Return(spectrum);
        }
        else
        {
            var plan = FourierPlan.Create(length);
            Complex[] padded = ArrayPool<Complex>.Shared.Rent(length);
            Pad(x, padded.AsSpan(0, length), reverse: false);
            var s = new VectorSpan<Complex>(padded.AsSpan(0, length));
            plan.Forward(s, s);
            for (int k = 0; k < length; k++)
            {
                Complex v = padded[k];
                padded[k] = new Complex(v.Real * v.Real + v.Imaginary * v.Imaginary, 0);
            }

            plan.Inverse(s, s);
            for (int k = 0; k <= maxLag; k++)
            {
                destination[k] = Unsafe.As<Complex, T>(ref padded[k]);
            }

            ArrayPool<Complex>.Shared.Return(padded);
        }
    }

    private static void Pad<T, TPadded>(ReadOnlyVectorSpan<T> source, Span<TPadded> padded, bool reverse)
        where T : unmanaged, INumberBase<T>
        where TPadded : unmanaged
    {
        int n = source.Length;
        for (int t = 0; t < n; t++)
        {
            T value = source[reverse ? n - 1 - t : t];
            padded[t] = Unsafe.As<T, TPadded>(ref value);
        }

        padded[n..].Clear();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T Conjugate<T>(T value) where T : unmanaged, INumberBase<T>
    {
        if (typeof(T) == typeof(Complex))
        {
            var conjugate = Complex.Conjugate(Unsafe.As<T, Complex>(ref value));
            return Unsafe.As<Complex, T>(ref conjugate);
        }

        return value;
    }

    private static void RequireMaxLag(int length, int maxLag)
    {
        if (length == 0)
        {
            throw new ArgumentException("The input of an autocorrelation must not be empty.");
        }

        ArgumentOutOfRangeException.ThrowIfNegative(maxLag);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(maxLag, length);
    }

    private static void RequireLength(int expected, int actual)
    {
        if (expected != actual)
        {
            throw new ArgumentException($"The destination must have {expected} elements, but it has {actual}.", "destination");
        }
    }

    private static void RequireDisjoint<T>(ReadOnlyVectorSpan<T> input, VectorSpan<T> destination) where T : unmanaged, INumberBase<T>
    {
        var output = destination.AsReadOnly();
        if (StridedLayout.Overlaps(
                in input.Reference, input.Length, 1, input.Stride, 0,
                in output.Reference, output.Length, 1, output.Stride, 0, out _))
        {
            throw new ArgumentException("The destination must not overlap the inputs.", nameof(destination));
        }
    }
}
