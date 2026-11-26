using System.Numerics;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

public static partial class XMathExtensions
{
    extension(XMath)
    {
        /// <summary>
        /// Applies the ReLU activation function <c>max(x, 0)</c> elementwise.
        /// <para>各要素に ReLU 活性化関数 <c>max(x, 0)</c> を適用します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <returns>A vector node holding the activated elements. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is 1 for <c>x &gt; 0</c>, 0 for <c>x ≤ 0</c> (including <c>±0</c>), and <c>NaN</c> for <c>NaN</c>.</para>
        /// <para>局所微分は <c>x &gt; 0</c> で 1、<c>x ≤ 0</c>（<c>±0</c> を含む）で 0、<c>NaN</c> で <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Relu<T>(VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => ActivationCore(x, ArrayOperation.Relu, 0, approximate: false);

        /// <summary>
        /// Applies the ReLU activation function <c>max(x, 0)</c> elementwise.
        /// <para>各要素に ReLU 活性化関数 <c>max(x, 0)</c> を適用します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A matrix node holding the activated elements. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative is 1 for <c>x &gt; 0</c>, 0 for <c>x ≤ 0</c> (including <c>±0</c>), and <c>NaN</c> for <c>NaN</c>.</para>
        /// <para>局所微分は <c>x &gt; 0</c> で 1、<c>x ≤ 0</c>（<c>±0</c> を含む）で 0、<c>NaN</c> で <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Relu<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => ActivationCore(x, ArrayOperation.Relu, 0, approximate: false);

        /// <summary>
        /// Applies the Leaky ReLU activation function (<c>x</c> for <c>x &gt; 0</c>, otherwise <c>slope·x</c>) elementwise.
        /// <para>各要素に Leaky ReLU 活性化関数（<c>x &gt; 0</c> で <c>x</c>、それ以外で <c>slope·x</c>）を適用します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <param name="slope">The constant negative slope (non-differentiable). / 負の領域に適用する定数の傾き（微分対象外）。</param>
        /// <returns>A vector node holding the activated elements. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is 1 for <c>x &gt; 0</c>, <paramref name="slope"/> for <c>x ≤ 0</c> (including <c>±0</c>), and <c>NaN</c> for <c>NaN</c>.</para>
        /// <para>局所微分は <c>x &gt; 0</c> で 1、<c>x ≤ 0</c>（<c>±0</c> を含む）で <paramref name="slope"/>、<c>NaN</c> で <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> LeakyRelu<T>(VectorNode<T> x, T slope) where T : unmanaged, IFloatingPointIeee754<T>
            => ActivationCore(x, ArrayOperation.LeakyRelu, double.CreateTruncating(slope), approximate: false);

        /// <summary>
        /// Applies the Leaky ReLU activation function (<c>x</c> for <c>x &gt; 0</c>, otherwise <c>slope·x</c>) elementwise.
        /// <para>各要素に Leaky ReLU 活性化関数（<c>x &gt; 0</c> で <c>x</c>、それ以外で <c>slope·x</c>）を適用します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <param name="slope">The constant negative slope (non-differentiable). / 負の領域に適用する定数の傾き（微分対象外）。</param>
        /// <returns>A matrix node holding the activated elements. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative is 1 for <c>x &gt; 0</c>, <paramref name="slope"/> for <c>x ≤ 0</c> (including <c>±0</c>), and <c>NaN</c> for <c>NaN</c>.</para>
        /// <para>局所微分は <c>x &gt; 0</c> で 1、<c>x ≤ 0</c>（<c>±0</c> を含む）で <paramref name="slope"/>、<c>NaN</c> で <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> LeakyRelu<T>(MatrixNode<T> x, T slope) where T : unmanaged, IFloatingPointIeee754<T>
            => ActivationCore(x, ArrayOperation.LeakyRelu, double.CreateTruncating(slope), approximate: false);

        /// <summary>
        /// Applies the ELU activation function (<c>x</c> for <c>x &gt; 0</c>, otherwise <c>alpha·(eˣ - 1)</c>) elementwise.
        /// <para>各要素に ELU 活性化関数（<c>x &gt; 0</c> で <c>x</c>、それ以外で <c>alpha·(eˣ - 1)</c>）を適用します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <param name="alpha">The constant scale factor for negative inputs (non-differentiable). / 負の領域に適用する定数係数（微分対象外）。</param>
        /// <returns>A vector node holding the activated elements. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is 1 for <c>x &gt; 0</c> and <c>alpha·eˣ</c> for <c>x ≤ 0</c> (including <c>±0</c>).</para>
        /// <para>局所微分は <c>x &gt; 0</c> で 1、<c>x ≤ 0</c>（<c>±0</c> を含む）で <c>alpha·eˣ</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Elu<T>(VectorNode<T> x, T alpha) where T : unmanaged, IFloatingPointIeee754<T>
            => ActivationCore(x, ArrayOperation.Elu, double.CreateTruncating(alpha), approximate: false);

        /// <summary>
        /// Applies the ELU activation function (<c>x</c> for <c>x &gt; 0</c>, otherwise <c>alpha·(eˣ - 1)</c>) elementwise.
        /// <para>各要素に ELU 活性化関数（<c>x &gt; 0</c> で <c>x</c>、それ以外で <c>alpha·(eˣ - 1)</c>）を適用します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <param name="alpha">The constant scale factor for negative inputs (non-differentiable). / 負の領域に適用する定数係数（微分対象外）。</param>
        /// <returns>A matrix node holding the activated elements. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative is 1 for <c>x &gt; 0</c> and <c>alpha·eˣ</c> for <c>x ≤ 0</c> (including <c>±0</c>).</para>
        /// <para>局所微分は <c>x &gt; 0</c> で 1、<c>x ≤ 0</c>（<c>±0</c> を含む）で <c>alpha·eˣ</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Elu<T>(MatrixNode<T> x, T alpha) where T : unmanaged, IFloatingPointIeee754<T>
            => ActivationCore(x, ArrayOperation.Elu, double.CreateTruncating(alpha), approximate: false);

        /// <summary>
        /// Applies the SELU activation function <c>λ·ELU(x; α)</c> (<c>α ≈ 1.6733</c>, <c>λ ≈ 1.0507</c>) elementwise.
        /// <para>各要素に SELU 活性化関数 <c>λ·ELU(x; α)</c>（<c>α ≈ 1.6733</c>、<c>λ ≈ 1.0507</c>）を適用します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <returns>A vector node holding the activated elements. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>λ</c> for <c>x &gt; 0</c> and <c>λα·eˣ</c> for <c>x ≤ 0</c> (including <c>±0</c>).</para>
        /// <para>局所微分は <c>x &gt; 0</c> で <c>λ</c>、<c>x ≤ 0</c>（<c>±0</c> を含む）で <c>λα·eˣ</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Selu<T>(VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => ActivationCore(x, ArrayOperation.Selu, 0, approximate: false);

        /// <summary>
        /// Applies the SELU activation function <c>λ·ELU(x; α)</c> (<c>α ≈ 1.6733</c>, <c>λ ≈ 1.0507</c>) elementwise.
        /// <para>各要素に SELU 活性化関数 <c>λ·ELU(x; α)</c>（<c>α ≈ 1.6733</c>、<c>λ ≈ 1.0507</c>）を適用します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A matrix node holding the activated elements. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>λ</c> for <c>x &gt; 0</c> and <c>λα·eˣ</c> for <c>x ≤ 0</c> (including <c>±0</c>).</para>
        /// <para>局所微分は <c>x &gt; 0</c> で <c>λ</c>、<c>x ≤ 0</c>（<c>±0</c> を含む）で <c>λα·eˣ</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Selu<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => ActivationCore(x, ArrayOperation.Selu, 0, approximate: false);

        /// <summary>
        /// Applies the exact GELU activation function <c>x·Φ(x)</c> elementwise.
        /// <para>各要素に厳密な GELU 活性化関数 <c>x·Φ(x)</c> を適用します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <returns>A vector node holding the activated elements. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>Φ(x) + x·φ(x)</c>.</para>
        /// <para>局所微分は <c>Φ(x) + x·φ(x)</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Gelu<T>(VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => ActivationCore(x, ArrayOperation.Gelu, 0, approximate: false);

        /// <summary>
        /// Applies the GELU activation function elementwise, optionally using the <c>tanh</c> approximation.
        /// <para>各要素に GELU 活性化関数を適用します（オプションで <c>tanh</c> 近似式を使用できます）。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <param name="approximate"><see langword="true"/> to use the approximation <c>(x/2)·(1 + tanh(√(2/π)·(x + 0.044715·x³)))</c>; <see langword="false"/> for exact <c>x·Φ(x)</c>. / <c>tanh</c> 近似式 <c>(x/2)·(1 + tanh(√(2/π)·(x + 0.044715·x³)))</c> を使用する場合は <see langword="true"/>、厳密式 <c>x·Φ(x)</c> を使用する場合は <see langword="false"/>。</param>
        /// <returns>A vector node holding the activated elements. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>Φ(x) + x·φ(x)</c> when <paramref name="approximate"/> is <see langword="false"/>, or the analytic derivative of the <c>tanh</c> approximation when <see langword="true"/>.</para>
        /// <para>局所微分は <paramref name="approximate"/> が <see langword="false"/> のとき <c>Φ(x) + x·φ(x)</c>、<see langword="true"/> のときは <c>tanh</c> 近似式の導関数とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Gelu<T>(VectorNode<T> x, bool approximate) where T : unmanaged, IFloatingPointIeee754<T>
            => ActivationCore(x, approximate ? ArrayOperation.GeluTanh : ArrayOperation.Gelu, 0, approximate);

        /// <summary>
        /// Applies the exact GELU activation function <c>x·Φ(x)</c> elementwise.
        /// <para>各要素に厳密な GELU 活性化関数 <c>x·Φ(x)</c> を適用します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A matrix node holding the activated elements. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>Φ(x) + x·φ(x)</c>.</para>
        /// <para>局所微分は <c>Φ(x) + x·φ(x)</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Gelu<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => ActivationCore(x, ArrayOperation.Gelu, 0, approximate: false);

        /// <summary>
        /// Applies the GELU activation function elementwise, optionally using the <c>tanh</c> approximation.
        /// <para>各要素に GELU 活性化関数を適用します（オプションで <c>tanh</c> 近似式を使用できます）。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <param name="approximate"><see langword="true"/> to use the approximation <c>(x/2)·(1 + tanh(√(2/π)·(x + 0.044715·x³)))</c>; <see langword="false"/> for exact <c>x·Φ(x)</c>. / <c>tanh</c> 近似式 <c>(x/2)·(1 + tanh(√(2/π)·(x + 0.044715·x³)))</c> を使用する場合は <see langword="true"/>、厳密式 <c>x·Φ(x)</c> を使用する場合は <see langword="false"/>。</param>
        /// <returns>A matrix node holding the activated elements. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>Φ(x) + x·φ(x)</c> when <paramref name="approximate"/> is <see langword="false"/>, or the analytic derivative of the <c>tanh</c> approximation when <see langword="true"/>.</para>
        /// <para>局所微分は <paramref name="approximate"/> が <see langword="false"/> のとき <c>Φ(x) + x·φ(x)</c>、<see langword="true"/> のときは <c>tanh</c> 近似式の導関数とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Gelu<T>(MatrixNode<T> x, bool approximate) where T : unmanaged, IFloatingPointIeee754<T>
            => ActivationCore(x, approximate ? ArrayOperation.GeluTanh : ArrayOperation.Gelu, 0, approximate);

        /// <summary>
        /// Applies the SiLU (Swish) activation function <c>x·σ(x)</c> elementwise.
        /// <para>各要素に SiLU（Swish）活性化関数 <c>x·σ(x)</c> を適用します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <returns>A vector node holding the activated elements. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>σ(x)·(1 + x·σ(-x))</c>.</para>
        /// <para>局所微分は <c>σ(x)·(1 + x·σ(-x))</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Silu<T>(VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => ActivationCore(x, ArrayOperation.Silu, 0, approximate: false);

        /// <summary>
        /// Applies the SiLU (Swish) activation function <c>x·σ(x)</c> elementwise, optionally using an approximate <see langword="float"/> kernel.
        /// <para>各要素に SiLU（Swish）活性化関数 <c>x·σ(x)</c> を適用します（<see langword="float"/> では近似カーネルを選択できます）。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <param name="approximate"><see langword="true"/> to use the approximate kernel for <see langword="float"/> arrays (ignored for <see langword="double"/>). / <see langword="float"/> 配列で近似式を使用する場合は <see langword="true"/>（<see langword="double"/> では結果に影響しません）。</param>
        /// <returns>A vector node holding the activated elements. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>σ(x)·(1 + x·σ(-x))</c> regardless of <paramref name="approximate"/>.</para>
        /// <para>局所微分は <paramref name="approximate"/> の指定にかかわらず <c>σ(x)·(1 + x·σ(-x))</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Silu<T>(VectorNode<T> x, bool approximate) where T : unmanaged, IFloatingPointIeee754<T>
            => ActivationCore(x, ArrayOperation.Silu, 0, approximate);

        /// <summary>
        /// Applies the SiLU (Swish) activation function <c>x·σ(x)</c> elementwise.
        /// <para>各要素に SiLU（Swish）活性化関数 <c>x·σ(x)</c> を適用します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A matrix node holding the activated elements. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>σ(x)·(1 + x·σ(-x))</c>.</para>
        /// <para>局所微分は <c>σ(x)·(1 + x·σ(-x))</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Silu<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => ActivationCore(x, ArrayOperation.Silu, 0, approximate: false);

        /// <summary>
        /// Applies the SiLU (Swish) activation function <c>x·σ(x)</c> elementwise, optionally using an approximate <see langword="float"/> kernel.
        /// <para>各要素に SiLU（Swish）活性化関数 <c>x·σ(x)</c> を適用します（<see langword="float"/> では近似カーネルを選択できます）。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <param name="approximate"><see langword="true"/> to use the approximate kernel for <see langword="float"/> arrays (ignored for <see langword="double"/>). / <see langword="float"/> 配列で近似式を使用する場合は <see langword="true"/>（<see langword="double"/> では結果に影響しません）。</param>
        /// <returns>A matrix node holding the activated elements. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>σ(x)·(1 + x·σ(-x))</c> regardless of <paramref name="approximate"/>.</para>
        /// <para>局所微分は <paramref name="approximate"/> の指定にかかわらず <c>σ(x)·(1 + x·σ(-x))</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Silu<T>(MatrixNode<T> x, bool approximate) where T : unmanaged, IFloatingPointIeee754<T>
            => ActivationCore(x, ArrayOperation.Silu, 0, approximate);
    }

    private static VectorNode<T> ActivationCore<T>(VectorNode<T> x, ArrayOperation operation, double parameter, bool approximate)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        Tape tape = x.Owner();
        var store = tape.Store<T>();
        bool record = store.RequiresGradient(x.View.Slot);
        tape.EnsureCanApply(record);

        int slot = store.AddPooled(x.Length, 1, record);
        ActivationForward(operation, ArrayStore<T>.FromDouble(parameter), approximate, store.VectorView(x.View), store.WritableVector(slot));
        if (record)
        {
            tape.AppendArrayRecord<T>(operation, slot, x.View, default, parameter, -1);
        }

        return new(tape, ArrayView.Whole(slot, x.Length, 1));
    }

    private static MatrixNode<T> ActivationCore<T>(MatrixNode<T> x, ArrayOperation operation, double parameter, bool approximate)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        Tape tape = x.Owner();
        var store = tape.Store<T>();
        bool record = store.RequiresGradient(x.View.Slot);
        tape.EnsureCanApply(record);

        int slot = store.AddPooled(x.Rows, x.Columns, record);
        ActivationForward(operation, ArrayStore<T>.FromDouble(parameter), approximate, store.MatrixView(x.View), store.WritableMatrix(slot));
        if (record)
        {
            tape.AppendArrayRecord<T>(operation, slot, x.View, default, parameter, -1);
        }

        return new(tape, ArrayView.Whole(slot, x.Rows, x.Columns));
    }

    private static void ActivationForward<T>(ArrayOperation operation, T parameter, bool approximate, ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        switch (operation)
        {
            case ArrayOperation.Relu:
                XMath.Relu(x, destination);
                break;
            case ArrayOperation.LeakyRelu:
                XMath.LeakyRelu(x, parameter, destination);
                break;
            case ArrayOperation.Elu:
                XMath.Elu(x, parameter, destination);
                break;
            case ArrayOperation.Selu:
                XMath.Selu(x, destination);
                break;
            case ArrayOperation.Gelu:
                XMath.Gelu(x, destination);
                break;
            case ArrayOperation.GeluTanh:
                XMath.Gelu(x, true, destination);
                break;
            case ArrayOperation.Silu:
                XMath.Silu(x, approximate, destination);
                break;
        }
    }

    private static void ActivationForward<T>(ArrayOperation operation, T parameter, bool approximate, ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        switch (operation)
        {
            case ArrayOperation.Relu:
                XMath.Relu(x, destination);
                break;
            case ArrayOperation.LeakyRelu:
                XMath.LeakyRelu(x, parameter, destination);
                break;
            case ArrayOperation.Elu:
                XMath.Elu(x, parameter, destination);
                break;
            case ArrayOperation.Selu:
                XMath.Selu(x, destination);
                break;
            case ArrayOperation.Gelu:
                XMath.Gelu(x, destination);
                break;
            case ArrayOperation.GeluTanh:
                XMath.Gelu(x, true, destination);
                break;
            case ArrayOperation.Silu:
                XMath.Silu(x, approximate, destination);
                break;
        }
    }
}
