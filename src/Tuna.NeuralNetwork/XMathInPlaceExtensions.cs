using System.ComponentModel;
using System.Numerics;

namespace Tuna.NeuralNetwork;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

// XMathExtensions の extension(XMath) と同一クラスに配置するとソースジェネレータの生成オーバーロードが衝突するため分離している。
/// <summary>
/// Provides in-place normalization and PReLU operations as <see cref="XMath.InPlace"/> extensions.
/// <para>特徴量行列をインプレースで更新する正規化および PReLU を <see cref="XMath.InPlace"/> 拡張として提供します。</para>
/// </summary>
[EditorBrowsable(EditorBrowsableState.Never)]
public static partial class XMathInPlaceExtensions
{
    extension(XMath.InPlace)
    {
        /// <summary>
        /// Applies column-wise LayerNorm, <c>y = (x - μ) / √(σ² + ε) ⊙ scale + shift</c>, in place.
        /// <para>列ごとの LayerNorm <c>y = (x - μ) / √(σ² + ε) ⊙ scale + shift</c> をインプレースで適用します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">The input matrix of shape <c>[channels × positions]</c>, overwritten in place. / インプレースで更新する入力行列 <c>[チャネル × 位置]</c>。</param>
        /// <param name="scale">Per-channel scale factors of length <c>x.Rows</c>, or empty for all ones. Must not overlap <paramref name="x"/>. / チャネルごとのスケール（長さは入力の行数。空の場合は全チャネルで 1）。<paramref name="x"/> と重複してはなりません。</param>
        /// <param name="shift">Per-channel shift values of length <c>x.Rows</c>, or empty for all zeros. Must not overlap <paramref name="x"/>. / チャネルごとのシフト（長さは入力の行数。空の場合は全チャネルで 0）。<paramref name="x"/> と重複してはなりません。</param>
        /// <param name="epsilon">The non-negative constant added to the variance. / 分散に加える 0 以上の定数。</param>
        /// <remarks>
        /// <para><c>μ</c> is the column mean and <c>σ²</c> is the population variance.</para>
        /// <para><c>μ</c> は列の平均、<c>σ²</c> は母分散です。</para>
        /// </remarks>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="epsilon"/> is negative or <c>NaN</c>. / <paramref name="epsilon"/> が負または <c>NaN</c> の場合。</exception>
        /// <exception cref="ArgumentException">The length of <paramref name="scale"/> or <paramref name="shift"/> is invalid, or either overlaps <paramref name="x"/>. / <paramref name="scale"/> または <paramref name="shift"/> の長さが不正、あるいは <paramref name="x"/> と重複する場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        /// <seealso href="https://arxiv.org/abs/1607.06450">Layer Normalization (2016)</seealso>
        [InferenceOverloads]
        public static void LayerNorm<T>(MatrixSpan<T> x, ReadOnlyVectorSpan<T> scale, ReadOnlyVectorSpan<T> shift, T epsilon)
            where T : unmanaged, IFloatingPointIeee754<T> =>
            XMathExtensions.Normalize(x, scale, shift, epsilon, center: true, x);

        /// <summary>
        /// Applies column-wise RMSNorm, <c>y = x / √(mean(x²) + ε) ⊙ scale</c>, in place.
        /// <para>列ごとの RMSNorm <c>y = x / √(mean(x²) + ε) ⊙ scale</c> をインプレースで適用します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">The input matrix of shape <c>[channels × positions]</c>, overwritten in place. / インプレースで更新する入力行列 <c>[チャネル × 位置]</c>。</param>
        /// <param name="scale">Per-channel scale factors of length <c>x.Rows</c>, or empty for all ones. Must not overlap <paramref name="x"/>. / チャネルごとのスケール（長さは入力の行数。空の場合は全チャネルで 1）。<paramref name="x"/> と重複してはなりません。</param>
        /// <param name="epsilon">The non-negative constant added to the mean square. / 二乗平均に加える 0 以上の定数。</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="epsilon"/> is negative or <c>NaN</c>. / <paramref name="epsilon"/> が負または <c>NaN</c> の場合。</exception>
        /// <exception cref="ArgumentException">The length of <paramref name="scale"/> is invalid, or <paramref name="scale"/> overlaps <paramref name="x"/>. / <paramref name="scale"/> の長さが不正、または <paramref name="x"/> と重複する場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        /// <seealso href="https://arxiv.org/abs/1910.07467">Root Mean Square Layer Normalization (2019)</seealso>
        [InferenceOverloads]
        public static void RmsNorm<T>(MatrixSpan<T> x, ReadOnlyVectorSpan<T> scale, T epsilon)
            where T : unmanaged, IFloatingPointIeee754<T> =>
            XMathExtensions.Normalize(x, scale, default, epsilon, center: false, x);

        /// <summary>
        /// Applies parametric ReLU (PReLU), <c>y = x</c> for <c>x ≥ 0</c> and <c>aᵢ * x</c> for <c>x &lt; 0</c>, in place.
        /// <para>PReLU（<c>x ≥ 0</c> で <c>x</c>、<c>x &lt; 0</c> で <c>aᵢ * x</c>）をインプレースで適用します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see cref="float"/> or <see cref="double"/>). / 浮動小数点の要素型（<see cref="float"/> または <see cref="double"/>）。</typeparam>
        /// <param name="x">The input matrix of shape <c>[channels × positions]</c>, overwritten in place. / インプレースで更新する入力行列 <c>[チャネル × 位置]</c>。</param>
        /// <param name="slopes">Negative-region slopes of length <c>x.Rows</c> (per channel) or 1 (shared across all channels). Must not overlap <paramref name="x"/>. / 負の領域の傾き（長さは入力の行数または全チャネル共通の 1）。<paramref name="x"/> と重複してはなりません。</param>
        /// <exception cref="ArgumentException">The length of <paramref name="slopes"/> is neither <c>x.Rows</c> nor 1, or <paramref name="slopes"/> overlaps <paramref name="x"/>. / <paramref name="slopes"/> の長さが入力の行数でも 1 でもない場合、または <paramref name="x"/> と重複する場合。</exception>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see cref="float"/> nor <see cref="double"/>. / <typeparamref name="T"/> が <see cref="float"/> でも <see cref="double"/> でもない場合。</exception>
        /// <seealso href="https://arxiv.org/abs/1502.01852">Delving Deep into Rectifiers (2015)</seealso>
        [InferenceOverloads]
        public static void Prelu<T>(MatrixSpan<T> x, ReadOnlyVectorSpan<T> slopes)
            where T : unmanaged, IFloatingPointIeee754<T> =>
            XMathExtensions.PreluCore(x, slopes, x);
    }
}
