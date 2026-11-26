using System.ComponentModel;
using System.Numerics;

namespace Tuna.NeuralNetwork;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

/// <summary>
/// Provides neural network inference operations such as normalization, activation, linear projection, and state-space operations as <see cref="XMath"/> extensions.
/// <para>正規化、活性化関数、全結合、状態空間モデル演算など、ニューラルネットワーク推論向けの演算を <see cref="XMath"/> 拡張として提供します。</para>
/// </summary>
/// <remarks>
/// <para>Feature matrices use rows for channels and columns for positions (such as time steps or spatial locations). Supported element types are <see cref="float"/> and <see cref="double"/>.</para>
/// <para>特徴量行列は行をチャネル、列を位置（時刻や空間位置など）として扱います。対応する要素型は <see cref="float"/> および <see cref="double"/> です。</para>
/// </remarks>
[EditorBrowsable(EditorBrowsableState.Never)]
public static partial class XMathExtensions
{
    extension(XMath)
    {
        /// <summary>
        /// Applies column-wise LayerNorm, <c>y = (x - μ) / √(σ² + ε) ⊙ scale + shift</c>, and returns a new matrix.
        /// <para>列ごとの LayerNorm <c>y = (x - μ) / √(σ² + ε) ⊙ scale + shift</c> を計算し、新しい行列として返します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">The input matrix of shape <c>[channels × positions]</c>. / 入力行列 <c>[チャネル × 位置]</c>。</param>
        /// <param name="scale">Per-channel scale factors of length <c>x.Rows</c>, or empty for all ones. / チャネルごとのスケール（長さは入力の行数。空の場合は全チャネルで 1）。</param>
        /// <param name="shift">Per-channel shift values of length <c>x.Rows</c>, or empty for all zeros. / チャネルごとのシフト（長さは入力の行数。空の場合は全チャネルで 0）。</param>
        /// <param name="epsilon">The non-negative constant added to the variance. / 分散に加える 0 以上の定数。</param>
        /// <returns>A new matrix with the same shape as <paramref name="x"/>. / <paramref name="x"/> と同じ形状の新しい行列。</returns>
        /// <remarks>
        /// <para><c>μ</c> is the column mean and <c>σ²</c> is the population variance (sum of squared deviations divided by <c>x.Rows</c>).</para>
        /// <para><c>μ</c> は列の平均、<c>σ²</c> は母分散（偏差平方和を行数で割った値）です。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">The length of <paramref name="scale"/> or <paramref name="shift"/> is neither <c>x.Rows</c> nor 0. / <paramref name="scale"/> または <paramref name="shift"/> の長さが入力の行数でも 0 でもない場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="epsilon"/> is negative or <c>NaN</c>. / <paramref name="epsilon"/> が負または <c>NaN</c> の場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        /// <seealso href="https://arxiv.org/abs/1607.06450">Layer Normalization (2016)</seealso>
        [InferenceOverloads]
        public static Matrix<T> LayerNorm<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyVectorSpan<T> scale, ReadOnlyVectorSpan<T> shift, T epsilon)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
            Normalize(x, scale, shift, epsilon, center: true, result);
            return result;
        }

        /// <summary>
        /// Applies column-wise LayerNorm, <c>y = (x - μ) / √(σ² + ε) ⊙ scale + shift</c>, into <paramref name="destination"/>.
        /// <para>列ごとの LayerNorm <c>y = (x - μ) / √(σ² + ε) ⊙ scale + shift</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">The input matrix of shape <c>[channels × positions]</c>. / 入力行列 <c>[チャネル × 位置]</c>。</param>
        /// <param name="scale">Per-channel scale factors of length <c>x.Rows</c>, or empty for all ones. / チャネルごとのスケール（長さは入力の行数。空の場合は全チャネルで 1）。</param>
        /// <param name="shift">Per-channel shift values of length <c>x.Rows</c>, or empty for all zeros. / チャネルごとのシフト（長さは入力の行数。空の場合は全チャネルで 0）。</param>
        /// <param name="epsilon">The non-negative constant added to the variance. / 分散に加える 0 以上の定数。</param>
        /// <param name="destination">The output matrix span with the same shape as <paramref name="x"/>. / <paramref name="x"/> と同じ形状の出力先。</param>
        /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
        /// <remarks>
        /// <para><c>μ</c> is the column mean and <c>σ²</c> is the population variance. <paramref name="destination"/> must either alias <paramref name="x"/> identically or not overlap <paramref name="x"/>, and must not overlap <paramref name="scale"/> or <paramref name="shift"/>.</para>
        /// <para><c>μ</c> は列の平均、<c>σ²</c> は母分散です。<paramref name="destination"/> は <paramref name="x"/> と完全一致するか重複しないビューである必要があり、<paramref name="scale"/> や <paramref name="shift"/> と重複してはなりません。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">The parameter lengths, shapes, or memory overlap rules are violated. / パラメータの長さ、形状、またはメモリ重複の条件を満たさない場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="epsilon"/> is negative or <c>NaN</c>. / <paramref name="epsilon"/> が負または <c>NaN</c> の場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        /// <seealso href="https://arxiv.org/abs/1607.06450">Layer Normalization (2016)</seealso>
        [InferenceOverloads]
        public static MatrixSpan<T> LayerNorm<T>(
            ReadOnlyMatrixSpan<T> x, ReadOnlyVectorSpan<T> scale, ReadOnlyVectorSpan<T> shift, T epsilon, MatrixSpan<T> destination)
            where T : unmanaged, IFloatingPointIeee754<T> =>
            Normalize(x, scale, shift, epsilon, center: true, destination);

        /// <summary>
        /// Applies column-wise RMSNorm, <c>y = x / √(mean(x²) + ε) ⊙ scale</c>, and returns a new matrix.
        /// <para>列ごとの RMSNorm <c>y = x / √(mean(x²) + ε) ⊙ scale</c> を計算し、新しい行列として返します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">The input matrix of shape <c>[channels × positions]</c>. / 入力行列 <c>[チャネル × 位置]</c>。</param>
        /// <param name="scale">Per-channel scale factors of length <c>x.Rows</c>, or empty for all ones. / チャネルごとのスケール（長さは入力の行数。空の場合は全チャネルで 1）。</param>
        /// <param name="epsilon">The non-negative constant added to the mean square. / 二乗平均に加える 0 以上の定数。</param>
        /// <returns>A new matrix with the same shape as <paramref name="x"/>. / <paramref name="x"/> と同じ形状の新しい行列。</returns>
        /// <exception cref="ArgumentException">The length of <paramref name="scale"/> is neither <c>x.Rows</c> nor 0. / <paramref name="scale"/> の長さが入力の行数でも 0 でもない場合。</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="epsilon"/> is negative or <c>NaN</c>. / <paramref name="epsilon"/> が負または <c>NaN</c> の場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        /// <seealso href="https://arxiv.org/abs/1910.07467">Root Mean Square Layer Normalization (2019)</seealso>
        [InferenceOverloads]
        public static Matrix<T> RmsNorm<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyVectorSpan<T> scale, T epsilon)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
            Normalize(x, scale, default, epsilon, center: false, result);
            return result;
        }

        /// <summary>
        /// Applies column-wise RMSNorm, <c>y = x / √(mean(x²) + ε) ⊙ scale</c>, into <paramref name="destination"/>.
        /// <para>列ごとの RMSNorm <c>y = x / √(mean(x²) + ε) ⊙ scale</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">The input matrix of shape <c>[channels × positions]</c>. / 入力行列 <c>[チャネル × 位置]</c>。</param>
        /// <param name="scale">Per-channel scale factors of length <c>x.Rows</c>, or empty for all ones. / チャネルごとのスケール（長さは入力の行数。空の場合は全チャネルで 1）。</param>
        /// <param name="epsilon">The non-negative constant added to the mean square. / 二乗平均に加える 0 以上の定数。</param>
        /// <param name="destination">The output matrix span with the same shape as <paramref name="x"/>. / <paramref name="x"/> と同じ形状の出力先。</param>
        /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
        /// <remarks>
        /// <para><paramref name="destination"/> must either alias <paramref name="x"/> identically or not overlap <paramref name="x"/>, and must not overlap <paramref name="scale"/>.</para>
        /// <para><paramref name="destination"/> は <paramref name="x"/> と完全一致するか重複しないビューである必要があり、<paramref name="scale"/> と重複してはなりません。</para>
        /// </remarks>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="epsilon"/> is negative or <c>NaN</c>. / <paramref name="epsilon"/> が負または <c>NaN</c> の場合。</exception>
        /// <exception cref="ArgumentException">The parameter length, shape, or memory overlap rules are violated. / パラメータの長さ、形状、またはメモリ重複の条件を満たさない場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        /// <seealso href="https://arxiv.org/abs/1910.07467">Root Mean Square Layer Normalization (2019)</seealso>
        [InferenceOverloads]
        public static MatrixSpan<T> RmsNorm<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyVectorSpan<T> scale, T epsilon, MatrixSpan<T> destination)
            where T : unmanaged, IFloatingPointIeee754<T> =>
            Normalize(x, scale, default, epsilon, center: false, destination);

        /// <summary>
        /// Applies parametric ReLU (PReLU), <c>y = x</c> for <c>x ≥ 0</c> and <c>aᵢ * x</c> for <c>x &lt; 0</c>, and returns a new matrix.
        /// <para>PReLU（<c>x ≥ 0</c> で <c>x</c>、<c>x &lt; 0</c> で <c>aᵢ * x</c>）を適用し、新しい行列として返します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">The input matrix of shape <c>[channels × positions]</c>. / 入力行列 <c>[チャネル × 位置]</c>。</param>
        /// <param name="slopes">Negative-region slopes of length <c>x.Rows</c> (per channel) or 1 (shared across all channels). / 負の領域の傾き（長さは入力の行数または全チャネル共通の 1）。</param>
        /// <returns>A new matrix with the same shape as <paramref name="x"/>. <c>NaN</c> inputs are propagated unchanged. / <paramref name="x"/> と同じ形状の新しい行列。<c>NaN</c> の要素はそのまま保持されます。</returns>
        /// <exception cref="ArgumentException">The length of <paramref name="slopes"/> is neither <c>x.Rows</c> nor 1. / <paramref name="slopes"/> の長さが入力の行数でも 1 でもない場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        /// <seealso href="https://arxiv.org/abs/1502.01852">Delving Deep into Rectifiers (2015)</seealso>
        [InferenceOverloads]
        public static Matrix<T> Prelu<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyVectorSpan<T> slopes)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
            PreluCore(x, slopes, result);
            return result;
        }

        /// <summary>
        /// Applies parametric ReLU (PReLU), <c>y = x</c> for <c>x ≥ 0</c> and <c>aᵢ * x</c> for <c>x &lt; 0</c>, into <paramref name="destination"/>.
        /// <para>PReLU（<c>x ≥ 0</c> で <c>x</c>、<c>x &lt; 0</c> で <c>aᵢ * x</c>）を適用し、<paramref name="destination"/> に書き込みます。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">The input matrix of shape <c>[channels × positions]</c>. / 入力行列 <c>[チャネル × 位置]</c>。</param>
        /// <param name="slopes">Negative-region slopes of length <c>x.Rows</c> (per channel) or 1 (shared across all channels). / 負の領域の傾き（長さは入力の行数または全チャネル共通の 1）。</param>
        /// <param name="destination">The output matrix span with the same shape as <paramref name="x"/>. / <paramref name="x"/> と同じ形状の出力先。</param>
        /// <returns><paramref name="destination"/>. <c>NaN</c> inputs are propagated unchanged. / <paramref name="destination"/>。<c>NaN</c> の要素はそのまま保持されます。</returns>
        /// <remarks>
        /// <para><paramref name="destination"/> must either alias <paramref name="x"/> identically or not overlap <paramref name="x"/>, and must not overlap <paramref name="slopes"/>.</para>
        /// <para><paramref name="destination"/> は <paramref name="x"/> と完全一致するか重複しないビューである必要があり、<paramref name="slopes"/> と重複してはなりません。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">The slope length, shape, or memory overlap rules are violated. / 傾きの長さ、形状、またはメモリ重複の条件を満たさない場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        /// <seealso href="https://arxiv.org/abs/1502.01852">Delving Deep into Rectifiers (2015)</seealso>
        [InferenceOverloads]
        public static MatrixSpan<T> Prelu<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyVectorSpan<T> slopes, MatrixSpan<T> destination)
            where T : unmanaged, IFloatingPointIeee754<T> =>
            PreluCore(x, slopes, destination);

        /// <summary>
        /// Computes the linear (affine) transformation <c>y = W * x + b</c> and returns a new matrix.
        /// <para>全結合（アフィン変換）<c>y = W * x + b</c> を計算し、新しい行列として返します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="weight">The weight matrix of shape <c>[outChannels × inChannels]</c>. / 重み行列 <c>[出力チャネル × 入力チャネル]</c>。</param>
        /// <param name="x">The input matrix of shape <c>[inChannels × positions]</c>. / 入力行列 <c>[入力チャネル × 位置]</c>。</param>
        /// <param name="bias">Per-output-channel bias of length <c>weight.Rows</c>, or empty for zero bias. / 出力チャネルごとのバイアス（長さは重みの行数。空の場合は 0）。</param>
        /// <returns>A new matrix of shape <c>[outChannels × positions]</c>. / <c>[出力チャネル × 位置]</c> の新しい行列。</returns>
        /// <remarks>
        /// <para>Row-major PyTorch <c>nn.Linear.weight</c> data can be wrapped without copying via <c>new ReadOnlyMatrixSpan&lt;float&gt;(w, inChannels, outChannels).T</c>.</para>
        /// <para>行優先の PyTorch <c>nn.Linear.weight</c> は <c>new ReadOnlyMatrixSpan&lt;float&gt;(w, inChannels, outChannels).T</c> でコピーせずに参照できます。</para>
        /// </remarks>
        /// <exception cref="ArgumentException"><c>x.Rows</c> does not match <c>weight.Columns</c>, or <paramref name="bias"/> length is neither <c>weight.Rows</c> nor 0. / 入力の行数が重みの列数と一致しない場合、または <paramref name="bias"/> の長さが重みの行数でも 0 でもない場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        [InferenceOverloads]
        public static Matrix<T> Linear<T>(ReadOnlyMatrixSpan<T> weight, ReadOnlyMatrixSpan<T> x, ReadOnlyVectorSpan<T> bias)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            var result = Matrix<T>.Uninitialized(weight.Rows, x.Columns);
            LinearCore(weight, x, bias, result);
            return result;
        }

        /// <summary>
        /// Computes the linear (affine) transformation <c>y = W * x + b</c> into <paramref name="destination"/>.
        /// <para>全結合（アフィン変換）<c>y = W * x + b</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="weight">The weight matrix of shape <c>[outChannels × inChannels]</c>. / 重み行列 <c>[出力チャネル × 入力チャネル]</c>。</param>
        /// <param name="x">The input matrix of shape <c>[inChannels × positions]</c>. / 入力行列 <c>[入力チャネル × 位置]</c>。</param>
        /// <param name="bias">Per-output-channel bias of length <c>weight.Rows</c>, or empty for zero bias. / 出力チャネルごとのバイアス（長さは重みの行数。空の場合は 0）。</param>
        /// <param name="destination">The output matrix span of shape <c>[outChannels × positions]</c>. / <c>[出力チャネル × 位置]</c> の出力先。</param>
        /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
        /// <remarks>
        /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>, <paramref name="weight"/>, or <paramref name="bias"/>.</para>
        /// <para><paramref name="destination"/> は <paramref name="x"/>、<paramref name="weight"/>、<paramref name="bias"/> のいずれとも重複してはなりません。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">The matrix dimensions, bias length, or memory overlap rules are violated. / 行列形状、バイアス長、またはメモリ重複の条件を満たさない場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        [InferenceOverloads]
        public static MatrixSpan<T> Linear<T>(ReadOnlyMatrixSpan<T> weight, ReadOnlyMatrixSpan<T> x, ReadOnlyVectorSpan<T> bias, MatrixSpan<T> destination)
            where T : unmanaged, IFloatingPointIeee754<T> =>
            LinearCore(weight, x, bias, destination);

        /// <summary>
        /// Applies unscaled column-wise LayerNorm, <c>y = (x - μ) / √(σ² + ε)</c>, and returns a new matrix.
        /// <para>スケール・シフトなしの列ごとの LayerNorm <c>y = (x - μ) / √(σ² + ε)</c> を計算し、新しい行列として返します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">The input matrix of shape <c>[channels × positions]</c>. / 入力行列 <c>[チャネル × 位置]</c>。</param>
        /// <param name="epsilon">The non-negative constant added to the variance. / 分散に加える 0 以上の定数。</param>
        /// <returns>A new matrix with the same shape as <paramref name="x"/>. / <paramref name="x"/> と同じ形状の新しい行列。</returns>
        /// <remarks>
        /// <para><c>μ</c> is the column mean and <c>σ²</c> is the population variance.</para>
        /// <para><c>μ</c> は列の平均、<c>σ²</c> は母分散です。</para>
        /// </remarks>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="epsilon"/> is negative or <c>NaN</c>. / <paramref name="epsilon"/> が負または <c>NaN</c> の場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        /// <seealso href="https://arxiv.org/abs/1607.06450">Layer Normalization (2016)</seealso>
        [InferenceOverloads]
        public static Matrix<T> LayerNorm<T>(ReadOnlyMatrixSpan<T> x, T epsilon)
            where T : unmanaged, IFloatingPointIeee754<T> =>
            XMath.LayerNorm(x, default(ReadOnlyVectorSpan<T>), default, epsilon);

        /// <summary>
        /// Applies unscaled column-wise RMSNorm, <c>y = x / √(mean(x²) + ε)</c>, and returns a new matrix.
        /// <para>スケールなしの列ごとの RMSNorm <c>y = x / √(mean(x²) + ε)</c> を計算し、新しい行列として返します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">The input matrix of shape <c>[channels × positions]</c>. / 入力行列 <c>[チャネル × 位置]</c>。</param>
        /// <param name="epsilon">The non-negative constant added to the mean square. / 二乗平均に加える 0 以上の定数。</param>
        /// <returns>A new matrix with the same shape as <paramref name="x"/>. / <paramref name="x"/> と同じ形状の新しい行列。</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="epsilon"/> is negative or <c>NaN</c>. / <paramref name="epsilon"/> が負または <c>NaN</c> の場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        /// <seealso href="https://arxiv.org/abs/1910.07467">Root Mean Square Layer Normalization (2019)</seealso>
        [InferenceOverloads]
        public static Matrix<T> RmsNorm<T>(ReadOnlyMatrixSpan<T> x, T epsilon)
            where T : unmanaged, IFloatingPointIeee754<T> =>
            XMath.RmsNorm(x, default(ReadOnlyVectorSpan<T>), epsilon);

        /// <summary>
        /// Computes the linear transformation <c>y = W * x</c> without bias and returns a new matrix.
        /// <para>バイアスなしの全結合 <c>y = W * x</c> を計算し、新しい行列として返します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="weight">The weight matrix of shape <c>[outChannels × inChannels]</c>. / 重み行列 <c>[出力チャネル × 入力チャネル]</c>。</param>
        /// <param name="x">The input matrix of shape <c>[inChannels × positions]</c>. / 入力行列 <c>[入力チャネル × 位置]</c>。</param>
        /// <returns>A new matrix of shape <c>[outChannels × positions]</c>. / <c>[出力チャネル × 位置]</c> の新しい行列。</returns>
        /// <exception cref="ArgumentException"><c>x.Rows</c> does not match <c>weight.Columns</c>. / 入力の行数が重みの列数と一致しない場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        [InferenceOverloads]
        public static Matrix<T> Linear<T>(ReadOnlyMatrixSpan<T> weight, ReadOnlyMatrixSpan<T> x)
            where T : unmanaged, IFloatingPointIeee754<T> =>
            XMath.Linear(weight, x, default(ReadOnlyVectorSpan<T>));
    }

    internal static MatrixSpan<T> Normalize<T>(
        ReadOnlyMatrixSpan<T> x, ReadOnlyVectorSpan<T> scale, ReadOnlyVectorSpan<T> shift, T epsilon, bool center, MatrixSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        Normalize(x, scale, shift, epsilon, center, destination, default);

    // stats が空でなければ、列ごとに (K, δ, 1/√(σ² + ε)) の 3 個を書き込む（Kernels.NormalizeColumn）。
    internal static unsafe MatrixSpan<T> Normalize<T>(
        ReadOnlyMatrixSpan<T> x, ReadOnlyVectorSpan<T> scale, ReadOnlyVectorSpan<T> shift, T epsilon, bool center, MatrixSpan<T> destination,
        Span<T> stats)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        Guard.RequireElementType<T>();
        Guard.RequireShape(x.Rows, x.Columns, destination.Rows, destination.Columns, nameof(destination));
        Guard.RequireOptionalLength(x.Rows, scale.Length, nameof(scale));
        Guard.RequireOptionalLength(x.Rows, shift.Length, nameof(shift));
        if (!(epsilon >= T.Zero))
        {
            throw new ArgumentOutOfRangeException(nameof(epsilon), "The epsilon must be non-negative.");
        }

        Guard.RequireNoOverlap(x, destination, allowExact: true, nameof(destination));
        Guard.RequireNoOverlap(scale, destination, nameof(destination));
        Guard.RequireNoOverlap(shift, destination, nameof(destination));
        if (x.IsEmpty)
        {
            return destination;
        }

        int rows = x.Rows, columns = x.Columns;
        T[]? rx = null, rs = null, rb = null, ry = null;
        try
        {
            var xd = Dense.Columns(x, ref rx);
            var sd = Dense.Contiguous(scale, ref rs);
            var bd = Dense.Contiguous(shift, ref rb);
            var target = destination;
            bool direct = rows <= 1 || destination.AsReadOnly().RowStride == 1;
            if (!direct)
            {
                ry = System.Buffers.ArrayPool<T>.Shared.Rent(rows * columns);
                target = new MatrixSpan<T>(ry.AsSpan(0, rows * columns), rows, columns);
            }

            int threads = Split.Threads(4.0 * rows * columns);
            fixed (T* px = &Dense.Ref(xd), ps = &Dense.Ref(sd), pb = &Dense.Ref(bd), py = &Dense.Ref(target), pt = stats)
            {
                var work = new NormalizeWork<T>
                {
                    X = px,
                    Y = py,
                    Scale = scale.Length == 0 ? null : ps,
                    Shift = shift.Length == 0 ? null : pb,
                    Stats = stats.IsEmpty ? null : pt,
                    Rows = rows,
                    Ldx = xd.ColumnStride,
                    Ldy = target.AsReadOnly().ColumnStride,
                    Epsilon = epsilon,
                    Center = center,
                };
                Split.For(columns, Split.Chunks(columns, threads, 16), threads, ref work);
            }

            if (!direct)
            {
                target.CopyTo(destination);
            }
        }
        finally
        {
            Dense.Return(rx);
            Dense.Return(rs);
            Dense.Return(rb);
            Dense.Return(ry);
        }

        return destination;
    }

    internal static unsafe MatrixSpan<T> PreluCore<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyVectorSpan<T> slopes, MatrixSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        Guard.RequireElementType<T>();
        Guard.RequireShape(x.Rows, x.Columns, destination.Rows, destination.Columns, nameof(destination));
        if (slopes.Length != x.Rows && slopes.Length != 1)
        {
            throw new ArgumentException($"The slopes must have length {x.Rows} (one per row) or 1 but got {slopes.Length}.", nameof(slopes));
        }

        Guard.RequireNoOverlap(x, destination, allowExact: true, nameof(destination));
        Guard.RequireNoOverlap(slopes, destination, nameof(destination));
        if (x.IsEmpty)
        {
            return destination;
        }

        int rows = x.Rows, columns = x.Columns;
        T[]? rx = null, rs = null, ry = null;
        try
        {
            var xd = Dense.Columns(x, ref rx);
            var sd = Dense.Contiguous(slopes, ref rs);
            var target = destination;
            bool direct = rows <= 1 || destination.AsReadOnly().RowStride == 1;
            if (!direct)
            {
                ry = System.Buffers.ArrayPool<T>.Shared.Rent(rows * columns);
                target = new MatrixSpan<T>(ry.AsSpan(0, rows * columns), rows, columns);
            }

            int threads = Split.Threads(1.0 * rows * columns);
            bool perRow = slopes.Length == rows && rows > 1;
            fixed (T* px = &Dense.Ref(xd), ps = &Dense.Ref(sd), py = &Dense.Ref(target))
            {
                var work = new PreluWork<T>
                {
                    X = px,
                    Y = py,
                    Slopes = perRow ? ps : null,
                    Slope = sd[0],
                    Rows = rows,
                    Ldx = xd.ColumnStride,
                    Ldy = target.AsReadOnly().ColumnStride,
                };
                Split.For(columns, Split.Chunks(columns, threads, 16), threads, ref work);
            }

            if (!direct)
            {
                target.CopyTo(destination);
            }
        }
        finally
        {
            Dense.Return(rx);
            Dense.Return(rs);
            Dense.Return(ry);
        }

        return destination;
    }

    private unsafe struct NormalizeWork<T> : Split.IRangeWork where T : unmanaged, IFloatingPointIeee754<T>
    {
        public T* X, Y, Scale, Shift, Stats;
        public int Rows, Ldx, Ldy;
        public T Epsilon;
        public bool Center;

        public readonly void Invoke(int start, int end)
        {
            for (int j = start; j < end; j++)
            {
                Kernels.NormalizeColumn(X + (nint)j * Ldx, Y + (nint)j * Ldy, Rows, Scale, Shift, Epsilon, Center,
                    Stats == null ? null : Stats + (nint)3 * j);
            }
        }
    }

    private unsafe struct PreluWork<T> : Split.IRangeWork where T : unmanaged, IFloatingPointIeee754<T>
    {
        public T* X, Y, Slopes;
        public T Slope;
        public int Rows, Ldx, Ldy;

        public readonly void Invoke(int start, int end)
        {
            for (int j = start; j < end; j++)
            {
                Kernels.PreluColumn(X + (nint)j * Ldx, Y + (nint)j * Ldy, Rows, Slopes, Slope);
            }
        }
    }

    internal static MatrixSpan<T> LinearCore<T>(
        ReadOnlyMatrixSpan<T> weight, ReadOnlyMatrixSpan<T> x, ReadOnlyVectorSpan<T> bias, MatrixSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        Guard.RequireElementType<T>();
        if (weight.Columns != x.Rows)
        {
            throw new ArgumentException($"The weight has {weight.Columns} columns but the input has {x.Rows} rows.", nameof(x));
        }

        Guard.RequireShape(weight.Rows, x.Columns, destination.Rows, destination.Columns, nameof(destination));
        Guard.RequireOptionalLength(weight.Rows, bias.Length, nameof(bias));
        Guard.RequireNoOverlap(bias, destination, nameof(destination));
        XMath.Multiply(weight, x, destination);
        if (bias.Length == 0 || destination.IsEmpty)
        {
            return destination;
        }

        AddBias(destination, bias);
        return destination;
    }

    internal static unsafe void AddBias<T>(MatrixSpan<T> destination, ReadOnlyVectorSpan<T> bias) where T : unmanaged, IFloatingPointIeee754<T>
    {
        int rows = destination.Rows, columns = destination.Columns;
        if (rows > 1 && destination.AsReadOnly().RowStride != 1)
        {
            for (int j = 0; j < columns; j++)
            {
                var column = destination.Column(j);
                XMath.Add(column, bias, column);
            }

            return;
        }

        T[]? rb = null;
        try
        {
            var bd = Dense.Contiguous(bias, ref rb);
            int ld = destination.AsReadOnly().ColumnStride;
            fixed (T* py = &Dense.Ref(destination), pb = &Dense.Ref(bd))
            {
                for (int j = 0; j < columns; j++)
                {
                    Kernels.AddColumn(py + (nint)j * ld, pb, rows);
                }
            }
        }
        finally
        {
            Dense.Return(rb);
        }
    }
}
