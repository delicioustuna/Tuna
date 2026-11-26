using System.Numerics;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

public static partial class XMathExtensions
{
    extension(XMath)
    {
        /// <summary>
        /// Negates each element, <c>-x</c>.
        /// <para>各要素の符号を反転した <c>-x</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <returns>A vector node holding <c>-x</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>-1</c>.</para>
        /// <para>局所微分は <c>-1</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Negate<T>(VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Negate, approximate: false);

        /// <summary>
        /// Negates each element, <c>-x</c>.
        /// <para>各要素の符号を反転した <c>-x</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A matrix node holding <c>-x</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>-1</c>.</para>
        /// <para>局所微分は <c>-1</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Negate<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Negate, approximate: false);

        /// <summary>
        /// Computes the natural exponential function <c>eˣ</c> elementwise.
        /// <para>各要素の自然指数関数 <c>eˣ</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <returns>A vector node holding <c>eˣ</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is the output value <c>eˣ</c>.</para>
        /// <para>局所微分は出力の値 <c>eˣ</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Exp<T>(VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Exp, approximate: false);

        /// <summary>
        /// Computes the natural exponential function <c>eˣ</c> elementwise, optionally using an approximate <see langword="float"/> kernel.
        /// <para>各要素の自然指数関数 <c>eˣ</c> を計算します（<see langword="float"/> では近似カーネルを選択できます）。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <param name="approximate"><see langword="true"/> to use the approximate kernel for <see langword="float"/> arrays (ignored for <see langword="double"/>). / <see langword="float"/> 配列で近似式を使用する場合は <see langword="true"/>（<see langword="double"/> では結果に影響しません）。</param>
        /// <returns>A vector node holding <c>eˣ</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is the computed output value (including when <paramref name="approximate"/> is <see langword="true"/>).</para>
        /// <para>局所微分は順伝播の出力値（近似カーネル使用時はその近似値）とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Exp<T>(VectorNode<T> x, bool approximate) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Exp, approximate);

        /// <summary>
        /// Computes the natural exponential function <c>eˣ</c> elementwise.
        /// <para>各要素の自然指数関数 <c>eˣ</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A matrix node holding <c>eˣ</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative is the output value <c>eˣ</c>.</para>
        /// <para>局所微分は出力の値 <c>eˣ</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Exp<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Exp, approximate: false);

        /// <summary>
        /// Computes the natural exponential function <c>eˣ</c> elementwise, optionally using an approximate <see langword="float"/> kernel.
        /// <para>各要素の自然指数関数 <c>eˣ</c> を計算します（<see langword="float"/> では近似カーネルを選択できます）。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <param name="approximate"><see langword="true"/> to use the approximate kernel for <see langword="float"/> arrays (ignored for <see langword="double"/>). / <see langword="float"/> 配列で近似式を使用する場合は <see langword="true"/>（<see langword="double"/> では結果に影響しません）。</param>
        /// <returns>A matrix node holding <c>eˣ</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative is the computed output value (including when <paramref name="approximate"/> is <see langword="true"/>).</para>
        /// <para>局所微分は順伝播の出力値（近似カーネル使用時はその近似値）とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Exp<T>(MatrixNode<T> x, bool approximate) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Exp, approximate);

        /// <summary>
        /// Computes the natural logarithm <c>ln x</c> elementwise.
        /// <para>各要素の自然対数 <c>ln x</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <returns>A vector node holding <c>ln x</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>1/x</c>, with the right-hand limit <c>+∞</c> at <c>±0</c> and <c>NaN</c> for negative inputs.</para>
        /// <para>局所微分は <c>1/x</c> とします。<c>±0</c> では右側極限の <c>+∞</c>、負の入力では <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Log<T>(VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Log, approximate: false);

        /// <summary>
        /// Computes the natural logarithm <c>ln x</c> elementwise.
        /// <para>各要素の自然対数 <c>ln x</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A matrix node holding <c>ln x</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>1/x</c>, with the right-hand limit <c>+∞</c> at <c>±0</c> and <c>NaN</c> for negative inputs.</para>
        /// <para>局所微分は <c>1/x</c> とします。<c>±0</c> では右側極限の <c>+∞</c>、負の入力では <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Log<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Log, approximate: false);

        /// <summary>
        /// Computes the square root <c>√x</c> elementwise.
        /// <para>各要素の平方根 <c>√x</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <returns>A vector node holding <c>√x</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>1/(2√x)</c>, with the right-hand limit <c>+∞</c> at <c>±0</c> and <c>NaN</c> for negative inputs.</para>
        /// <para>局所微分は <c>1/(2√x)</c> とします。<c>±0</c> では右側極限の <c>+∞</c>、負の入力では <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Sqrt<T>(VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Sqrt, approximate: false);

        /// <summary>
        /// Computes the square root <c>√x</c> elementwise.
        /// <para>各要素の平方根 <c>√x</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A matrix node holding <c>√x</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>1/(2√x)</c>, with the right-hand limit <c>+∞</c> at <c>±0</c> and <c>NaN</c> for negative inputs.</para>
        /// <para>局所微分は <c>1/(2√x)</c> とします。<c>±0</c> では右側極限の <c>+∞</c>、負の入力では <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Sqrt<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Sqrt, approximate: false);

        /// <summary>
        /// Computes the square <c>x²</c> elementwise.
        /// <para>各要素の 2 乗 <c>x²</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <returns>A vector node holding <c>x²</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>2x</c>.</para>
        /// <para>局所微分は <c>2x</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Square<T>(VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Square, approximate: false);

        /// <summary>
        /// Computes the square <c>x²</c> elementwise.
        /// <para>各要素の 2 乗 <c>x²</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A matrix node holding <c>x²</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>2x</c>.</para>
        /// <para>局所微分は <c>2x</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Square<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Square, approximate: false);

        /// <summary>
        /// Computes the absolute value <c>|x|</c> elementwise.
        /// <para>各要素の絶対値 <c>|x|</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <returns>A vector node holding <c>|x|</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>-1</c> for negative inputs, <c>+1</c> for positive inputs and <c>±0</c> (right-hand derivative), and <c>NaN</c> for <c>NaN</c>.</para>
        /// <para>局所微分は負の入力で <c>-1</c>、正の入力および <c>±0</c>（右側微分）で <c>+1</c>、<c>NaN</c> では <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Abs<T>(VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Abs, approximate: false);

        /// <summary>
        /// Computes the absolute value <c>|x|</c> elementwise.
        /// <para>各要素の絶対値 <c>|x|</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A matrix node holding <c>|x|</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>-1</c> for negative inputs, <c>+1</c> for positive inputs and <c>±0</c> (right-hand derivative), and <c>NaN</c> for <c>NaN</c>.</para>
        /// <para>局所微分は負の入力で <c>-1</c>、正の入力および <c>±0</c>（右側微分）で <c>+1</c>、<c>NaN</c> では <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Abs<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Abs, approximate: false);

        /// <summary>
        /// Computes <c>eˣ - 1</c> elementwise, preserving accuracy near zero.
        /// <para>0 の近傍でも精度を保って各要素の <c>eˣ - 1</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <returns>A vector node holding <c>eˣ - 1</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>eˣ</c>.</para>
        /// <para>局所微分は <c>eˣ</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Expm1<T>(VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Expm1, approximate: false);

        /// <summary>
        /// Computes <c>eˣ - 1</c> elementwise, preserving accuracy near zero.
        /// <para>0 の近傍でも精度を保って各要素の <c>eˣ - 1</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A matrix node holding <c>eˣ - 1</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>eˣ</c>.</para>
        /// <para>局所微分は <c>eˣ</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Expm1<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Expm1, approximate: false);

        /// <summary>
        /// Computes <c>ln(1 + x)</c> elementwise, preserving accuracy near zero.
        /// <para>0 の近傍でも精度を保って各要素の <c>ln(1 + x)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <returns>A vector node holding <c>ln(1 + x)</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>1/(1 + x)</c>, with the right-hand limit <c>+∞</c> at <c>-1</c> and <c>NaN</c> for inputs less than <c>-1</c>.</para>
        /// <para>局所微分は <c>1/(1 + x)</c> とします。<c>-1</c> では右側極限の <c>+∞</c>、<c>-1</c> 未満の入力では <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Log1p<T>(VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Log1p, approximate: false);

        /// <summary>
        /// Computes <c>ln(1 + x)</c> elementwise, preserving accuracy near zero.
        /// <para>0 の近傍でも精度を保って各要素の <c>ln(1 + x)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A matrix node holding <c>ln(1 + x)</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>1/(1 + x)</c>, with the right-hand limit <c>+∞</c> at <c>-1</c> and <c>NaN</c> for inputs less than <c>-1</c>.</para>
        /// <para>局所微分は <c>1/(1 + x)</c> とします。<c>-1</c> では右側極限の <c>+∞</c>、<c>-1</c> 未満の入力では <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Log1p<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Log1p, approximate: false);

        /// <summary>
        /// Computes the logistic sigmoid function <c>σ(x) = 1/(1 + e⁻ˣ)</c> elementwise.
        /// <para>各要素のロジスティック関数 <c>σ(x) = 1/(1 + e⁻ˣ)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <returns>A vector node holding <c>σ(x)</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is computed as <c>t/(1 + t)²</c> with <c>t = e^-|x|</c>, avoiding premature underflow in the positive tail.</para>
        /// <para>局所微分は <c>t = e^-|x|</c> を用いて <c>t/(1 + t)²</c> として計算し、値が 1 に丸まる正の裾でも精度を保ちます。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Logistic<T>(VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Logistic, approximate: false);

        /// <summary>
        /// Computes the logistic sigmoid function <c>σ(x) = 1/(1 + e⁻ˣ)</c> elementwise, optionally using an approximate <see langword="float"/> kernel.
        /// <para>各要素のロジスティック関数 <c>σ(x) = 1/(1 + e⁻ˣ)</c> を計算します（<see langword="float"/> では近似カーネルを選択できます）。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <param name="approximate"><see langword="true"/> to use the approximate kernel for <see langword="float"/> arrays (ignored for <see langword="double"/>). / <see langword="float"/> 配列で近似式を使用する場合は <see langword="true"/>（<see langword="double"/> では結果に影響しません）。</param>
        /// <returns>A vector node holding <c>σ(x)</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative uses the exact analytic formula regardless of <paramref name="approximate"/>.</para>
        /// <para>局所微分は <paramref name="approximate"/> の指定にかかわらず厳密式で計算します。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Logistic<T>(VectorNode<T> x, bool approximate) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Logistic, approximate);

        /// <summary>
        /// Computes the logistic sigmoid function <c>σ(x) = 1/(1 + e⁻ˣ)</c> elementwise.
        /// <para>各要素のロジスティック関数 <c>σ(x) = 1/(1 + e⁻ˣ)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A matrix node holding <c>σ(x)</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative is computed as <c>t/(1 + t)²</c> with <c>t = e^-|x|</c>, avoiding premature underflow in the positive tail.</para>
        /// <para>局所微分は <c>t = e^-|x|</c> を用いて <c>t/(1 + t)²</c> として計算し、値が 1 に丸まる正の裾でも精度を保ちます。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Logistic<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Logistic, approximate: false);

        /// <summary>
        /// Computes the logistic sigmoid function <c>σ(x) = 1/(1 + e⁻ˣ)</c> elementwise, optionally using an approximate <see langword="float"/> kernel.
        /// <para>各要素のロジスティック関数 <c>σ(x) = 1/(1 + e⁻ˣ)</c> を計算します（<see langword="float"/> では近似カーネルを選択できます）。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <param name="approximate"><see langword="true"/> to use the approximate kernel for <see langword="float"/> arrays (ignored for <see langword="double"/>). / <see langword="float"/> 配列で近似式を使用する場合は <see langword="true"/>（<see langword="double"/> では結果に影響しません）。</param>
        /// <returns>A matrix node holding <c>σ(x)</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative uses the exact analytic formula regardless of <paramref name="approximate"/>.</para>
        /// <para>局所微分は <paramref name="approximate"/> の指定にかかわらず厳密式で計算します。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Logistic<T>(MatrixNode<T> x, bool approximate) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Logistic, approximate);

        /// <summary>
        /// Computes the natural logarithm of the logistic sigmoid function, <c>ln σ(x)</c>, elementwise.
        /// <para>各要素のロジスティック関数の自然対数 <c>ln σ(x)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <returns>A vector node holding <c>ln σ(x)</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>σ(-x)</c>.</para>
        /// <para>局所微分は <c>σ(-x)</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> LogLogistic<T>(VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.LogLogistic, approximate: false);

        /// <summary>
        /// Computes the natural logarithm of the logistic sigmoid function, <c>ln σ(x)</c>, elementwise.
        /// <para>各要素のロジスティック関数の自然対数 <c>ln σ(x)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A matrix node holding <c>ln σ(x)</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>σ(-x)</c>.</para>
        /// <para>局所微分は <c>σ(-x)</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> LogLogistic<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.LogLogistic, approximate: false);

        /// <summary>
        /// Computes the logit function <c>ln(p/(1 - p))</c> elementwise.
        /// <para>各要素のロジット関数 <c>ln(p/(1 - p))</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input probability vector node. / 確率を格納した入力ベクトルノード。</param>
        /// <returns>A vector node holding <c>ln(p/(1 - p))</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>1/(p(1 - p))</c>, with <c>+∞</c> at <c>0</c> and <c>1</c>, and <c>NaN</c> outside <c>[0, 1]</c> or for <c>NaN</c>.</para>
        /// <para>局所微分は <c>1/(p(1 - p))</c> とします。区間 <c>[0, 1]</c> の端点では <c>+∞</c>、区間外または <c>NaN</c> では <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Logit<T>(VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Logit, approximate: false);

        /// <summary>
        /// Computes the logit function <c>ln(p/(1 - p))</c> elementwise.
        /// <para>各要素のロジット関数 <c>ln(p/(1 - p))</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input probability matrix node. / 確率を格納した入力行列ノード。</param>
        /// <returns>A matrix node holding <c>ln(p/(1 - p))</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>1/(p(1 - p))</c>, with <c>+∞</c> at <c>0</c> and <c>1</c>, and <c>NaN</c> outside <c>[0, 1]</c> or for <c>NaN</c>.</para>
        /// <para>局所微分は <c>1/(p(1 - p))</c> とします。区間 <c>[0, 1]</c> の端点では <c>+∞</c>、区間外または <c>NaN</c> では <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Logit<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Logit, approximate: false);

        /// <summary>
        /// Computes the softplus function <c>ln(1 + eˣ)</c> elementwise.
        /// <para>各要素の Softplus 関数 <c>ln(1 + eˣ)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <returns>A vector node holding <c>ln(1 + eˣ)</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>σ(x)</c>.</para>
        /// <para>局所微分は <c>σ(x)</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Softplus<T>(VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Softplus, approximate: false);

        /// <summary>
        /// Computes the softplus function <c>ln(1 + eˣ)</c> elementwise, optionally using an approximate <see langword="float"/> kernel.
        /// <para>各要素の Softplus 関数 <c>ln(1 + eˣ)</c> を計算します（<see langword="float"/> では近似カーネルを選択できます）。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <param name="approximate"><see langword="true"/> to use the approximate kernel for <see langword="float"/> arrays (ignored for <see langword="double"/>). / <see langword="float"/> 配列で近似式を使用する場合は <see langword="true"/>（<see langword="double"/> では結果に影響しません）。</param>
        /// <returns>A vector node holding <c>ln(1 + eˣ)</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative uses the exact <c>σ(x)</c> regardless of <paramref name="approximate"/>.</para>
        /// <para>局所微分は <paramref name="approximate"/> の指定にかかわらず厳密な <c>σ(x)</c> を使用します。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Softplus<T>(VectorNode<T> x, bool approximate) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Softplus, approximate);

        /// <summary>
        /// Computes the softplus function <c>ln(1 + eˣ)</c> elementwise.
        /// <para>各要素の Softplus 関数 <c>ln(1 + eˣ)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A matrix node holding <c>ln(1 + eˣ)</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>σ(x)</c>.</para>
        /// <para>局所微分は <c>σ(x)</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Softplus<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Softplus, approximate: false);

        /// <summary>
        /// Computes the softplus function <c>ln(1 + eˣ)</c> elementwise, optionally using an approximate <see langword="float"/> kernel.
        /// <para>各要素の Softplus 関数 <c>ln(1 + eˣ)</c> を計算します（<see langword="float"/> では近似カーネルを選択できます）。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <param name="approximate"><see langword="true"/> to use the approximate kernel for <see langword="float"/> arrays (ignored for <see langword="double"/>). / <see langword="float"/> 配列で近似式を使用する場合は <see langword="true"/>（<see langword="double"/> では結果に影響しません）。</param>
        /// <returns>A matrix node holding <c>ln(1 + eˣ)</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative uses the exact <c>σ(x)</c> regardless of <paramref name="approximate"/>.</para>
        /// <para>局所微分は <paramref name="approximate"/> の指定にかかわらず厳密な <c>σ(x)</c> を使用します。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Softplus<T>(MatrixNode<T> x, bool approximate) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Softplus, approximate);

        /// <summary>
        /// Computes <c>ln(1 - eˣ)</c> elementwise on the non-positive domain.
        /// <para>非正の定義域で各要素の <c>ln(1 - eˣ)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <returns>A vector node holding <c>ln(1 - eˣ)</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>-eˣ/(1 - eˣ)</c> (evaluated stably across <c>-ln 2</c>), with the left-hand limit <c>-∞</c> at <c>±0</c> and <c>NaN</c> for positive inputs or <c>NaN</c>.</para>
        /// <para>局所微分は <c>-ln 2</c> を境に安定な式を切り替えて <c>-eˣ/(1 - eˣ)</c> を求めます。<c>±0</c> では左側極限の <c>-∞</c>、正の入力または <c>NaN</c> では <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Log1mExp<T>(VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Log1mExp, approximate: false);

        /// <summary>
        /// Computes <c>ln(1 - eˣ)</c> elementwise on the non-positive domain.
        /// <para>非正の定義域で各要素の <c>ln(1 - eˣ)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A matrix node holding <c>ln(1 - eˣ)</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>-eˣ/(1 - eˣ)</c> (evaluated stably across <c>-ln 2</c>), with the left-hand limit <c>-∞</c> at <c>±0</c> and <c>NaN</c> for positive inputs or <c>NaN</c>.</para>
        /// <para>局所微分は <c>-ln 2</c> を境に安定な式を切り替えて <c>-eˣ/(1 - eˣ)</c> を求めます。<c>±0</c> では左側極限の <c>-∞</c>、正の入力または <c>NaN</c> では <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Log1mExp<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Log1mExp, approximate: false);

        /// <summary>
        /// Computes the reciprocal <c>1/x</c> elementwise.
        /// <para>各要素の逆数 <c>1/x</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <returns>A vector node holding <c>1/x</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivative is computed from the output <c>y</c> as <c>-y²</c> (<c>-∞</c> at <c>±0</c>).</para>
        /// <para>局所微分は出力値 <c>y</c> から <c>-y²</c> として求めます（<c>±0</c> では <c>-∞</c>）。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Reciprocal<T>(VectorNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Reciprocal, approximate: false);

        /// <summary>
        /// Computes the reciprocal <c>1/x</c> elementwise.
        /// <para>各要素の逆数 <c>1/x</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <returns>A matrix node holding <c>1/x</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivative is computed from the output <c>y</c> as <c>-y²</c> (<c>-∞</c> at <c>±0</c>).</para>
        /// <para>局所微分は出力値 <c>y</c> から <c>-y²</c> として求めます（<c>±0</c> では <c>-∞</c>）。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the node is uninitialized, belongs to a stale generation, or the tape's backward pass has completed. / ノードが未初期化、世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Reciprocal<T>(MatrixNode<T> x) where T : unmanaged, IFloatingPointIeee754<T>
            => UnaryCore(x, ArrayOperation.Reciprocal, approximate: false);

        /// <summary>
        /// Adds two vector nodes elementwise, <c>x + y</c>.
        /// <para>2 つのベクトルノードの要素ごとの和 <c>x + y</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The first input vector node. / 第 1 入力ベクトルノード。</param>
        /// <param name="y">The second input vector node. / 第 2 入力ベクトルノード。</param>
        /// <returns>A vector node holding <c>x + y</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>Both local derivatives are 1.</para>
        /// <para>両入力の局所微分はいずれも 1 とします。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if the vectors do not have the same length. / ベクトルの要素数が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Add<T>(VectorNode<T> x, VectorNode<T> y) where T : unmanaged, IFloatingPointIeee754<T>
            => BinaryCore(x, y, ArrayOperation.Add);

        /// <summary>
        /// Adds two matrix nodes elementwise, <c>x + y</c>.
        /// <para>2 つの行列ノードの要素ごとの和 <c>x + y</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The first input matrix node. / 第 1 入力行列ノード。</param>
        /// <param name="y">The second input matrix node. / 第 2 入力行列ノード。</param>
        /// <returns>A matrix node holding <c>x + y</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>Both local derivatives are 1.</para>
        /// <para>両入力の局所微分はいずれも 1 とします。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if the matrices do not have the same shape. / 行列の形状が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Add<T>(MatrixNode<T> x, MatrixNode<T> y) where T : unmanaged, IFloatingPointIeee754<T>
            => BinaryCore(x, y, ArrayOperation.Add);

        /// <summary>
        /// Subtracts two vector nodes elementwise, <c>x - y</c>.
        /// <para>2 つのベクトルノードの要素ごとの差 <c>x - y</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The minuend vector node. / 被減数ベクトルノード。</param>
        /// <param name="y">The subtrahend vector node. / 減数ベクトルノード。</param>
        /// <returns>A vector node holding <c>x - y</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivatives are <c>1</c> and <c>-1</c> respectively.</para>
        /// <para>局所微分は順に <c>1</c>、<c>-1</c> とします。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if the vectors do not have the same length. / ベクトルの要素数が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Subtract<T>(VectorNode<T> x, VectorNode<T> y) where T : unmanaged, IFloatingPointIeee754<T>
            => BinaryCore(x, y, ArrayOperation.Subtract);

        /// <summary>
        /// Subtracts two matrix nodes elementwise, <c>x - y</c>.
        /// <para>2 つの行列ノードの要素ごとの差 <c>x - y</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The minuend matrix node. / 被減数行列ノード。</param>
        /// <param name="y">The subtrahend matrix node. / 減数行列ノード。</param>
        /// <returns>A matrix node holding <c>x - y</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivatives are <c>1</c> and <c>-1</c> respectively.</para>
        /// <para>局所微分は順に <c>1</c>、<c>-1</c> とします。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if the matrices do not have the same shape. / 行列の形状が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Subtract<T>(MatrixNode<T> x, MatrixNode<T> y) where T : unmanaged, IFloatingPointIeee754<T>
            => BinaryCore(x, y, ArrayOperation.Subtract);

        /// <summary>
        /// Computes the elementwise (Hadamard) product <c>x ∘ y</c>.
        /// <para>要素ごとの積（アダマール積）<c>x ∘ y</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The first input vector node. / 第 1 入力ベクトルノード。</param>
        /// <param name="y">The second input vector node. / 第 2 入力ベクトルノード。</param>
        /// <returns>A vector node holding <c>x ∘ y</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivatives are <c>y</c> and <c>x</c> respectively.</para>
        /// <para>局所微分は順に <c>y</c>、<c>x</c> とします。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if the vectors do not have the same length. / ベクトルの要素数が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Hadamard<T>(VectorNode<T> x, VectorNode<T> y) where T : unmanaged, IFloatingPointIeee754<T>
            => BinaryCore(x, y, ArrayOperation.Hadamard);

        /// <summary>
        /// Computes the elementwise (Hadamard) product <c>x ∘ y</c>.
        /// <para>要素ごとの積（アダマール積）<c>x ∘ y</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The first input matrix node. / 第 1 入力行列ノード。</param>
        /// <param name="y">The second input matrix node. / 第 2 入力行列ノード。</param>
        /// <returns>A matrix node holding <c>x ∘ y</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivatives are <c>y</c> and <c>x</c> respectively.</para>
        /// <para>局所微分は順に <c>y</c>、<c>x</c> とします。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if the matrices do not have the same shape. / 行列の形状が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Hadamard<T>(MatrixNode<T> x, MatrixNode<T> y) where T : unmanaged, IFloatingPointIeee754<T>
            => BinaryCore(x, y, ArrayOperation.Hadamard);

        /// <summary>
        /// Divides two vector nodes elementwise, <c>x / y</c>.
        /// <para>2 つのベクトルノードの要素ごとの商 <c>x / y</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The numerator vector node. / 分子ベクトルノード。</param>
        /// <param name="y">The denominator vector node. / 分母ベクトルノード。</param>
        /// <returns>A vector node holding <c>x / y</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The local derivatives are <c>1/y</c> and <c>-x/y²</c> (evaluated as <c>-(x/y)/y</c>, using exponent-separated scaling where <c>x/y</c> is subnormal).</para>
        /// <para>局所微分は <c>1/y</c> と <c>-x/y²</c> とします（後者は <c>-(x/y)/y</c> で求め、<c>x/y</c> が非正規化数になる要素では指数を分離して計算します）。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if the vectors do not have the same length. / ベクトルの要素数が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Divide<T>(VectorNode<T> x, VectorNode<T> y) where T : unmanaged, IFloatingPointIeee754<T>
            => BinaryCore(x, y, ArrayOperation.Divide);

        /// <summary>
        /// Divides two matrix nodes elementwise, <c>x / y</c>.
        /// <para>2 つの行列ノードの要素ごとの商 <c>x / y</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The numerator matrix node. / 分子行列ノード。</param>
        /// <param name="y">The denominator matrix node. / 分母行列ノード。</param>
        /// <returns>A matrix node holding <c>x / y</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The local derivatives are <c>1/y</c> and <c>-x/y²</c> (evaluated as <c>-(x/y)/y</c>, using exponent-separated scaling where <c>x/y</c> is subnormal).</para>
        /// <para>局所微分は <c>1/y</c> と <c>-x/y²</c> とします（後者は <c>-(x/y)/y</c> で求め、<c>x/y</c> が非正規化数になる要素では指数を分離して計算します）。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if the matrices do not have the same shape. / 行列の形状が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Divide<T>(MatrixNode<T> x, MatrixNode<T> y) where T : unmanaged, IFloatingPointIeee754<T>
            => BinaryCore(x, y, ArrayOperation.Divide);

        /// <summary>
        /// Computes the elementwise minimum <c>min(x, y)</c>.
        /// <para>要素ごとの最小値 <c>min(x, y)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The first input vector node. / 第 1 入力ベクトルノード。</param>
        /// <param name="y">The second input vector node. / 第 2 入力ベクトルノード。</param>
        /// <returns>A vector node holding <c>min(x, y)</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The selected input receives local derivative 1 and the other receives 0. For equal values (including signed zeros), <paramref name="x"/> receives 1 and <paramref name="y"/> receives 0. If either input is <c>NaN</c>, both local derivatives are <c>NaN</c>.</para>
        /// <para>選択された入力の局所微分を 1、もう一方を 0 とします。同値（符号付きゼロを含む）の場合は <paramref name="x"/> を 1、<paramref name="y"/> を 0 とします。いずれかが <c>NaN</c> の場合は両方の局所微分を <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if the vectors do not have the same length. / ベクトルの要素数が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Min<T>(VectorNode<T> x, VectorNode<T> y) where T : unmanaged, IFloatingPointIeee754<T>
            => BinaryCore(x, y, ArrayOperation.Min);

        /// <summary>
        /// Computes the elementwise minimum <c>min(x, y)</c>.
        /// <para>要素ごとの最小値 <c>min(x, y)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The first input matrix node. / 第 1 入力行列ノード。</param>
        /// <param name="y">The second input matrix node. / 第 2 入力行列ノード。</param>
        /// <returns>A matrix node holding <c>min(x, y)</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The selected input receives local derivative 1 and the other receives 0. For equal values (including signed zeros), <paramref name="x"/> receives 1 and <paramref name="y"/> receives 0. If either input is <c>NaN</c>, both local derivatives are <c>NaN</c>.</para>
        /// <para>選択された入力の局所微分を 1、もう一方を 0 とします。同値（符号付きゼロを含む）の場合は <paramref name="x"/> を 1、<paramref name="y"/> を 0 とします。いずれかが <c>NaN</c> の場合は両方の局所微分を <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if the matrices do not have the same shape. / 行列の形状が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Min<T>(MatrixNode<T> x, MatrixNode<T> y) where T : unmanaged, IFloatingPointIeee754<T>
            => BinaryCore(x, y, ArrayOperation.Min);

        /// <summary>
        /// Computes the elementwise maximum <c>max(x, y)</c>.
        /// <para>要素ごとの最大値 <c>max(x, y)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The first input vector node. / 第 1 入力ベクトルノード。</param>
        /// <param name="y">The second input vector node. / 第 2 入力ベクトルノード。</param>
        /// <returns>A vector node holding <c>max(x, y)</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The selected input receives local derivative 1 and the other receives 0. For equal values (including signed zeros), <paramref name="x"/> receives 1 and <paramref name="y"/> receives 0. If either input is <c>NaN</c>, both local derivatives are <c>NaN</c>.</para>
        /// <para>選択された入力の局所微分を 1、もう一方を 0 とします。同値（符号付きゼロを含む）の場合は <paramref name="x"/> を 1、<paramref name="y"/> を 0 とします。いずれかが <c>NaN</c> の場合は両方の局所微分を <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if the vectors do not have the same length. / ベクトルの要素数が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Max<T>(VectorNode<T> x, VectorNode<T> y) where T : unmanaged, IFloatingPointIeee754<T>
            => BinaryCore(x, y, ArrayOperation.Max);

        /// <summary>
        /// Computes the elementwise maximum <c>max(x, y)</c>.
        /// <para>要素ごとの最大値 <c>max(x, y)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The first input matrix node. / 第 1 入力行列ノード。</param>
        /// <param name="y">The second input matrix node. / 第 2 入力行列ノード。</param>
        /// <returns>A matrix node holding <c>max(x, y)</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The selected input receives local derivative 1 and the other receives 0. For equal values (including signed zeros), <paramref name="x"/> receives 1 and <paramref name="y"/> receives 0. If either input is <c>NaN</c>, both local derivatives are <c>NaN</c>.</para>
        /// <para>選択された入力の局所微分を 1、もう一方を 0 とします。同値（符号付きゼロを含む）の場合は <paramref name="x"/> を 1、<paramref name="y"/> を 0 とします。いずれかが <c>NaN</c> の場合は両方の局所微分を <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if the matrices do not have the same shape. / 行列の形状が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Max<T>(MatrixNode<T> x, MatrixNode<T> y) where T : unmanaged, IFloatingPointIeee754<T>
            => BinaryCore(x, y, ArrayOperation.Max);

        /// <summary>
        /// Computes the elementwise power <c>x^y</c>.
        /// <para>要素ごとの冪乗 <c>x^y</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The base vector node. / 底となるベクトルノード。</param>
        /// <param name="y">The exponent vector node. / 指数となるベクトルノード。</param>
        /// <returns>A vector node holding <c>x^y</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>Local derivatives follow the scalar <c>Pow(ScalarNode, ScalarNode)</c> rules, evaluated per element in <see langword="double"/> precision and converted back to <typeparamref name="T"/>.</para>
        /// <para>局所微分はスカラー版 <c>Pow(ScalarNode, ScalarNode)</c> と同じ規則に従い、要素ごとに <see langword="double"/> 精度で求めて <typeparamref name="T"/> へ変換します。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if the vectors do not have the same length. / ベクトルの要素数が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Pow<T>(VectorNode<T> x, VectorNode<T> y) where T : unmanaged, IFloatingPointIeee754<T>
            => BinaryCore(x, y, ArrayOperation.Pow);

        /// <summary>
        /// Computes the elementwise power <c>x^y</c>.
        /// <para>要素ごとの冪乗 <c>x^y</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The base matrix node. / 底となる行列ノード。</param>
        /// <param name="y">The exponent matrix node. / 指数となる行列ノード。</param>
        /// <returns>A matrix node holding <c>x^y</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>Local derivatives follow the scalar <c>Pow(ScalarNode, ScalarNode)</c> rules, evaluated per element in <see langword="double"/> precision and converted back to <typeparamref name="T"/>.</para>
        /// <para>局所微分はスカラー版 <c>Pow(ScalarNode, ScalarNode)</c> と同じ規則に従い、要素ごとに <see langword="double"/> 精度で求めて <typeparamref name="T"/> へ変換します。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if the matrices do not have the same shape. / 行列の形状が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Pow<T>(MatrixNode<T> x, MatrixNode<T> y) where T : unmanaged, IFloatingPointIeee754<T>
            => BinaryCore(x, y, ArrayOperation.Pow);

        /// <summary>
        /// Computes <c>ln(eˣ + eʸ)</c> elementwise.
        /// <para>要素ごとに <c>ln(eˣ + eʸ)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The first input vector node. / 第 1 入力ベクトルノード。</param>
        /// <param name="y">The second input vector node. / 第 2 入力ベクトルノード。</param>
        /// <returns>A vector node holding <c>ln(eˣ + eʸ)</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>Local derivatives follow the scalar <c>LogAddExp(ScalarNode, ScalarNode)</c> rules, evaluated per element in <see langword="double"/> precision and converted back to <typeparamref name="T"/>.</para>
        /// <para>局所微分はスカラー版 <c>LogAddExp(ScalarNode, ScalarNode)</c> と同じ規則に従い、要素ごとに <see langword="double"/> 精度で求めて <typeparamref name="T"/> へ変換します。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if the vectors do not have the same length. / ベクトルの要素数が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> LogAddExp<T>(VectorNode<T> x, VectorNode<T> y) where T : unmanaged, IFloatingPointIeee754<T>
            => BinaryCore(x, y, ArrayOperation.LogAddExp);

        /// <summary>
        /// Computes <c>ln(eˣ + eʸ)</c> elementwise.
        /// <para>要素ごとに <c>ln(eˣ + eʸ)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The first input matrix node. / 第 1 入力行列ノード。</param>
        /// <param name="y">The second input matrix node. / 第 2 入力行列ノード。</param>
        /// <returns>A matrix node holding <c>ln(eˣ + eʸ)</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>Local derivatives follow the scalar <c>LogAddExp(ScalarNode, ScalarNode)</c> rules, evaluated per element in <see langword="double"/> precision and converted back to <typeparamref name="T"/>.</para>
        /// <para>局所微分はスカラー版 <c>LogAddExp(ScalarNode, ScalarNode)</c> と同じ規則に従い、要素ごとに <see langword="double"/> 精度で求めて <typeparamref name="T"/> へ変換します。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if the matrices do not have the same shape. / 行列の形状が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> LogAddExp<T>(MatrixNode<T> x, MatrixNode<T> y) where T : unmanaged, IFloatingPointIeee754<T>
            => BinaryCore(x, y, ArrayOperation.LogAddExp);

        /// <summary>
        /// Computes <c>ln(eˣ - eʸ)</c> elementwise.
        /// <para>要素ごとに <c>ln(eˣ - eʸ)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The minuend vector node. / 被減数ベクトルノード。</param>
        /// <param name="y">The subtrahend vector node. / 減数ベクトルノード。</param>
        /// <returns>A vector node holding <c>ln(eˣ - eʸ)</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>Local derivatives follow the scalar <c>LogSubExp(ScalarNode, ScalarNode)</c> rules, evaluated per element in <see langword="double"/> precision and converted back to <typeparamref name="T"/>.</para>
        /// <para>局所微分はスカラー版 <c>LogSubExp(ScalarNode, ScalarNode)</c> と同じ規則に従い、要素ごとに <see langword="double"/> 精度で求めて <typeparamref name="T"/> へ変換します。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if the vectors do not have the same length. / ベクトルの要素数が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> LogSubExp<T>(VectorNode<T> x, VectorNode<T> y) where T : unmanaged, IFloatingPointIeee754<T>
            => BinaryCore(x, y, ArrayOperation.LogSubExp);

        /// <summary>
        /// Computes <c>ln(eˣ - eʸ)</c> elementwise.
        /// <para>要素ごとに <c>ln(eˣ - eʸ)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The minuend matrix node. / 被減数行列ノード。</param>
        /// <param name="y">The subtrahend matrix node. / 減数行列ノード。</param>
        /// <returns>A matrix node holding <c>ln(eˣ - eʸ)</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>Local derivatives follow the scalar <c>LogSubExp(ScalarNode, ScalarNode)</c> rules, evaluated per element in <see langword="double"/> precision and converted back to <typeparamref name="T"/>.</para>
        /// <para>局所微分はスカラー版 <c>LogSubExp(ScalarNode, ScalarNode)</c> と同じ規則に従い、要素ごとに <see langword="double"/> 精度で求めて <typeparamref name="T"/> へ変換します。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">Thrown if the matrices do not have the same shape. / 行列の形状が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> LogSubExp<T>(MatrixNode<T> x, MatrixNode<T> y) where T : unmanaged, IFloatingPointIeee754<T>
            => BinaryCore(x, y, ArrayOperation.LogSubExp);

        /// <summary>
        /// Adds a scalar to each element of a vector, <c>x + s</c>.
        /// <para>ベクトルの各要素にスカラーを加算した <c>x + s</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <param name="scalar">The scalar node or constant <c>s</c> to add. / 各要素に加算するスカラー <c>s</c>（ノードまたは定数）。</param>
        /// <returns>A vector node holding <c>x + s</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>For <see langword="float"/> arrays, <paramref name="scalar"/> is rounded to <see langword="float"/>. Both local derivatives are 1, and the gradient of <paramref name="scalar"/> sums the elementwise contributions (computed in <typeparamref name="T"/> and widened to <see langword="double"/>).</para>
        /// <para><see langword="float"/> 配列では <paramref name="scalar"/> を <see langword="float"/> に丸めて適用します。局所微分はいずれも 1 であり、<paramref name="scalar"/> の勾配は要素ごとの寄与の和（<typeparamref name="T"/> で計算して <see langword="double"/> へ拡幅）となります。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Add<T>(VectorNode<T> x, ScalarNode scalar) where T : unmanaged, IFloatingPointIeee754<T>
            => ScalarCore(x, scalar, ArrayOperation.AddScalar);

        /// <summary>
        /// Adds a scalar to each element of a matrix, <c>x + s</c>.
        /// <para>行列の各要素にスカラーを加算した <c>x + s</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <param name="scalar">The scalar node or constant <c>s</c> to add. / 各要素に加算するスカラー <c>s</c>（ノードまたは定数）。</param>
        /// <returns>A matrix node holding <c>x + s</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>For <see langword="float"/> arrays, <paramref name="scalar"/> is rounded to <see langword="float"/>. Both local derivatives are 1, and the gradient of <paramref name="scalar"/> sums the elementwise contributions (computed in <typeparamref name="T"/> and widened to <see langword="double"/>).</para>
        /// <para><see langword="float"/> 配列では <paramref name="scalar"/> を <see langword="float"/> に丸めて適用します。局所微分はいずれも 1 であり、<paramref name="scalar"/> の勾配は要素ごとの寄与の和（<typeparamref name="T"/> で計算して <see langword="double"/> へ拡幅）となります。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Add<T>(MatrixNode<T> x, ScalarNode scalar) where T : unmanaged, IFloatingPointIeee754<T>
            => ScalarCore(x, scalar, ArrayOperation.AddScalar);

        /// <summary>
        /// Subtracts a scalar from each element of a vector, <c>x - s</c>.
        /// <para>ベクトルの各要素からスカラーを減算した <c>x - s</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <param name="scalar">The scalar node or constant <c>s</c> to subtract. / 各要素から減算するスカラー <c>s</c>（ノードまたは定数）。</param>
        /// <returns>A vector node holding <c>x - s</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>For <see langword="float"/> arrays, <paramref name="scalar"/> is rounded to <see langword="float"/>. The local derivatives are <c>1</c> and <c>-1</c>, and the gradient of <paramref name="scalar"/> sums the elementwise contributions (computed in <typeparamref name="T"/> and widened to <see langword="double"/>).</para>
        /// <para><see langword="float"/> 配列では <paramref name="scalar"/> を <see langword="float"/> に丸めて適用します。局所微分は順に <c>1</c>、<c>-1</c> であり、<paramref name="scalar"/> の勾配は要素ごとの寄与の和（<typeparamref name="T"/> で計算して <see langword="double"/> へ拡幅）となります。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Subtract<T>(VectorNode<T> x, ScalarNode scalar) where T : unmanaged, IFloatingPointIeee754<T>
            => ScalarCore(x, scalar, ArrayOperation.SubtractScalar);

        /// <summary>
        /// Subtracts a scalar from each element of a matrix, <c>x - s</c>.
        /// <para>行列の各要素からスカラーを減算した <c>x - s</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <param name="scalar">The scalar node or constant <c>s</c> to subtract. / 各要素から減算するスカラー <c>s</c>（ノードまたは定数）。</param>
        /// <returns>A matrix node holding <c>x - s</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>For <see langword="float"/> arrays, <paramref name="scalar"/> is rounded to <see langword="float"/>. The local derivatives are <c>1</c> and <c>-1</c>, and the gradient of <paramref name="scalar"/> sums the elementwise contributions (computed in <typeparamref name="T"/> and widened to <see langword="double"/>).</para>
        /// <para><see langword="float"/> 配列では <paramref name="scalar"/> を <see langword="float"/> に丸めて適用します。局所微分は順に <c>1</c>、<c>-1</c> であり、<paramref name="scalar"/> の勾配は要素ごとの寄与の和（<typeparamref name="T"/> で計算して <see langword="double"/> へ拡幅）となります。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Subtract<T>(MatrixNode<T> x, ScalarNode scalar) where T : unmanaged, IFloatingPointIeee754<T>
            => ScalarCore(x, scalar, ArrayOperation.SubtractScalar);

        /// <summary>
        /// Multiplies each element of a vector by a scalar, <c>s·x</c>.
        /// <para>ベクトルの各要素をスカラー倍した <c>s·x</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <param name="scalar">The scalar node or constant <c>s</c>. / 各要素に掛けるスカラー <c>s</c>（ノードまたは定数）。</param>
        /// <returns>A vector node holding <c>s·x</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>For <see langword="float"/> arrays, <paramref name="scalar"/> is rounded to <see langword="float"/>. The local derivatives are <c>s</c> and <c>x</c>, and the gradient of <paramref name="scalar"/> sums the elementwise contributions (computed in <typeparamref name="T"/> and widened to <see langword="double"/>).</para>
        /// <para><see langword="float"/> 配列では <paramref name="scalar"/> を <see langword="float"/> に丸めて適用します。局所微分は順に <c>s</c>、<c>x</c> であり、<paramref name="scalar"/> の勾配は要素ごとの寄与の和（<typeparamref name="T"/> で計算して <see langword="double"/> へ拡幅）となります。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Multiply<T>(VectorNode<T> x, ScalarNode scalar) where T : unmanaged, IFloatingPointIeee754<T>
            => ScalarCore(x, scalar, ArrayOperation.MultiplyScalar);

        /// <summary>
        /// Multiplies each element of a matrix by a scalar, <c>s·x</c>.
        /// <para>行列の各要素をスカラー倍した <c>s·x</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <param name="scalar">The scalar node or constant <c>s</c>. / 各要素に掛けるスカラー <c>s</c>（ノードまたは定数）。</param>
        /// <returns>A matrix node holding <c>s·x</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>For <see langword="float"/> arrays, <paramref name="scalar"/> is rounded to <see langword="float"/>. The local derivatives are <c>s</c> and <c>x</c>, and the gradient of <paramref name="scalar"/> sums the elementwise contributions (computed in <typeparamref name="T"/> and widened to <see langword="double"/>).</para>
        /// <para><see langword="float"/> 配列では <paramref name="scalar"/> を <see langword="float"/> に丸めて適用します。局所微分は順に <c>s</c>、<c>x</c> であり、<paramref name="scalar"/> の勾配は要素ごとの寄与の和（<typeparamref name="T"/> で計算して <see langword="double"/> へ拡幅）となります。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Multiply<T>(MatrixNode<T> x, ScalarNode scalar) where T : unmanaged, IFloatingPointIeee754<T>
            => ScalarCore(x, scalar, ArrayOperation.MultiplyScalar);

        /// <summary>
        /// Divides each element of a vector by a scalar, <c>x / s</c>.
        /// <para>ベクトルの各要素をスカラーで割った <c>x / s</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The numerator vector node. / 分子ベクトルノード。</param>
        /// <param name="y">The divisor scalar node or constant <c>s</c>. / 除数となるスカラー <c>s</c>（ノードまたは定数）。</param>
        /// <returns>A vector node holding <c>x / s</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>For <see langword="float"/> arrays, <paramref name="y"/> is rounded to <see langword="float"/>. The local derivatives are <c>1/s</c> and <c>-x/s²</c> (evaluated as in elementwise division), and the gradient of <paramref name="y"/> sums the elementwise contributions (computed in <typeparamref name="T"/> and widened to <see langword="double"/>).</para>
        /// <para><see langword="float"/> 配列では <paramref name="y"/> を <see langword="float"/> に丸めて適用します。局所微分は <c>1/s</c> と <c>-x/s²</c>（要素ごとの商と同じ算出法）であり、<paramref name="y"/> の勾配は要素ごとの寄与の和（<typeparamref name="T"/> で計算して <see langword="double"/> へ拡幅）となります。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Divide<T>(VectorNode<T> x, ScalarNode y) where T : unmanaged, IFloatingPointIeee754<T>
            => ScalarCore(x, y, ArrayOperation.DivideScalar);

        /// <summary>
        /// Divides each element of a matrix by a scalar, <c>x / s</c>.
        /// <para>行列の各要素をスカラーで割った <c>x / s</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The numerator matrix node. / 分子行列ノード。</param>
        /// <param name="y">The divisor scalar node or constant <c>s</c>. / 除数となるスカラー <c>s</c>（ノードまたは定数）。</param>
        /// <returns>A matrix node holding <c>x / s</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>For <see langword="float"/> arrays, <paramref name="y"/> is rounded to <see langword="float"/>. The local derivatives are <c>1/s</c> and <c>-x/s²</c> (evaluated as in elementwise division), and the gradient of <paramref name="y"/> sums the elementwise contributions (computed in <typeparamref name="T"/> and widened to <see langword="double"/>).</para>
        /// <para><see langword="float"/> 配列では <paramref name="y"/> を <see langword="float"/> に丸めて適用します。局所微分は <c>1/s</c> と <c>-x/s²</c>（要素ごとの商と同じ算出法）であり、<paramref name="y"/> の勾配は要素ごとの寄与の和（<typeparamref name="T"/> で計算して <see langword="double"/> へ拡幅）となります。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Divide<T>(MatrixNode<T> x, ScalarNode y) where T : unmanaged, IFloatingPointIeee754<T>
            => ScalarCore(x, y, ArrayOperation.DivideScalar);

        /// <summary>
        /// Computes the elementwise minimum with a scalar, <c>min(x, s)</c>.
        /// <para>ベクトルの各要素とスカラーの最小値 <c>min(x, s)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <param name="y">The scalar node or constant <c>s</c>. / 比較するスカラー <c>s</c>（ノードまたは定数）。</param>
        /// <returns>A vector node holding <c>min(x, s)</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>For <see langword="float"/> arrays, <paramref name="y"/> is rounded to <see langword="float"/>. Local derivatives follow the elementwise <c>Min</c> rules, and the gradient of <paramref name="y"/> sums the elementwise contributions (computed in <typeparamref name="T"/> and widened to <see langword="double"/>).</para>
        /// <para><see langword="float"/> 配列では <paramref name="y"/> を <see langword="float"/> に丸めて適用します。局所微分は要素ごとの <c>Min</c> と同じ規則に従い、<paramref name="y"/> の勾配は要素ごとの寄与の和（<typeparamref name="T"/> で計算して <see langword="double"/> へ拡幅）となります。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Min<T>(VectorNode<T> x, ScalarNode y) where T : unmanaged, IFloatingPointIeee754<T>
            => ScalarCore(x, y, ArrayOperation.MinScalar);

        /// <summary>
        /// Computes the elementwise minimum with a scalar, <c>min(x, s)</c>.
        /// <para>行列の各要素とスカラーの最小値 <c>min(x, s)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <param name="y">The scalar node or constant <c>s</c>. / 比較するスカラー <c>s</c>（ノードまたは定数）。</param>
        /// <returns>A matrix node holding <c>min(x, s)</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>For <see langword="float"/> arrays, <paramref name="y"/> is rounded to <see langword="float"/>. Local derivatives follow the elementwise <c>Min</c> rules, and the gradient of <paramref name="y"/> sums the elementwise contributions (computed in <typeparamref name="T"/> and widened to <see langword="double"/>).</para>
        /// <para><see langword="float"/> 配列では <paramref name="y"/> を <see langword="float"/> に丸めて適用します。局所微分は要素ごとの <c>Min</c> と同じ規則に従い、<paramref name="y"/> の勾配は要素ごとの寄与の和（<typeparamref name="T"/> で計算して <see langword="double"/> へ拡幅）となります。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Min<T>(MatrixNode<T> x, ScalarNode y) where T : unmanaged, IFloatingPointIeee754<T>
            => ScalarCore(x, y, ArrayOperation.MinScalar);

        /// <summary>
        /// Computes the elementwise maximum with a scalar, <c>max(x, s)</c>.
        /// <para>ベクトルの各要素とスカラーの最大値 <c>max(x, s)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <param name="y">The scalar node or constant <c>s</c>. / 比較するスカラー <c>s</c>（ノードまたは定数）。</param>
        /// <returns>A vector node holding <c>max(x, s)</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>For <see langword="float"/> arrays, <paramref name="y"/> is rounded to <see langword="float"/>. Local derivatives follow the elementwise <c>Max</c> rules, and the gradient of <paramref name="y"/> sums the elementwise contributions (computed in <typeparamref name="T"/> and widened to <see langword="double"/>).</para>
        /// <para><see langword="float"/> 配列では <paramref name="y"/> を <see langword="float"/> に丸めて適用します。局所微分は要素ごとの <c>Max</c> と同じ規則に従い、<paramref name="y"/> の勾配は要素ごとの寄与の和（<typeparamref name="T"/> で計算して <see langword="double"/> へ拡幅）となります。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Max<T>(VectorNode<T> x, ScalarNode y) where T : unmanaged, IFloatingPointIeee754<T>
            => ScalarCore(x, y, ArrayOperation.MaxScalar);

        /// <summary>
        /// Computes the elementwise maximum with a scalar, <c>max(x, s)</c>.
        /// <para>行列の各要素とスカラーの最大値 <c>max(x, s)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <param name="y">The scalar node or constant <c>s</c>. / 比較するスカラー <c>s</c>（ノードまたは定数）。</param>
        /// <returns>A matrix node holding <c>max(x, s)</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>For <see langword="float"/> arrays, <paramref name="y"/> is rounded to <see langword="float"/>. Local derivatives follow the elementwise <c>Max</c> rules, and the gradient of <paramref name="y"/> sums the elementwise contributions (computed in <typeparamref name="T"/> and widened to <see langword="double"/>).</para>
        /// <para><see langword="float"/> 配列では <paramref name="y"/> を <see langword="float"/> に丸めて適用します。局所微分は要素ごとの <c>Max</c> と同じ規則に従い、<paramref name="y"/> の勾配は要素ごとの寄与の和（<typeparamref name="T"/> で計算して <see langword="double"/> へ拡幅）となります。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Max<T>(MatrixNode<T> x, ScalarNode y) where T : unmanaged, IFloatingPointIeee754<T>
            => ScalarCore(x, y, ArrayOperation.MaxScalar);

        /// <summary>
        /// Raises each element of a vector to a scalar power, <c>x^s</c>.
        /// <para>ベクトルの各要素のスカラー冪 <c>x^s</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The base vector node. / 底となるベクトルノード。</param>
        /// <param name="y">The exponent scalar node or constant <c>s</c>. / 指数となるスカラー <c>s</c>（ノードまたは定数）。</param>
        /// <returns>A vector node holding <c>x^s</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>For <see langword="float"/> arrays, <paramref name="y"/> is rounded to <see langword="float"/>. Local derivatives follow elementwise <c>Pow</c>, and the gradient of <paramref name="y"/> sums the elementwise contributions (computed in <typeparamref name="T"/> and widened to <see langword="double"/>).</para>
        /// <para><see langword="float"/> 配列では <paramref name="y"/> を <see langword="float"/> に丸めて適用します。局所微分は要素ごとの <c>Pow</c> と同じ算出法に従い、<paramref name="y"/> の勾配は要素ごとの寄与の和（<typeparamref name="T"/> で計算して <see langword="double"/> へ拡幅）となります。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Pow<T>(VectorNode<T> x, ScalarNode y) where T : unmanaged, IFloatingPointIeee754<T>
            => ScalarCore(x, y, ArrayOperation.PowScalar);

        /// <summary>
        /// Raises each element of a matrix to a scalar power, <c>x^s</c>.
        /// <para>行列の各要素のスカラー冪 <c>x^s</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The base matrix node. / 底となる行列ノード。</param>
        /// <param name="y">The exponent scalar node or constant <c>s</c>. / 指数となるスカラー <c>s</c>（ノードまたは定数）。</param>
        /// <returns>A matrix node holding <c>x^s</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>For <see langword="float"/> arrays, <paramref name="y"/> is rounded to <see langword="float"/>. Local derivatives follow elementwise <c>Pow</c>, and the gradient of <paramref name="y"/> sums the elementwise contributions (computed in <typeparamref name="T"/> and widened to <see langword="double"/>).</para>
        /// <para><see langword="float"/> 配列では <paramref name="y"/> を <see langword="float"/> に丸めて適用します。局所微分は要素ごとの <c>Pow</c> と同じ算出法に従い、<paramref name="y"/> の勾配は要素ごとの寄与の和（<typeparamref name="T"/> で計算して <see langword="double"/> へ拡幅）となります。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Pow<T>(MatrixNode<T> x, ScalarNode y) where T : unmanaged, IFloatingPointIeee754<T>
            => ScalarCore(x, y, ArrayOperation.PowScalar);

        /// <summary>
        /// Computes <c>ln(eˣ + eˢ)</c> elementwise with a scalar.
        /// <para>ベクトルの各要素とスカラーの <c>ln(eˣ + eˢ)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
        /// <param name="y">The scalar node or constant <c>s</c>. / スカラー <c>s</c>（ノードまたは定数）。</param>
        /// <returns>A vector node holding <c>ln(eˣ + eˢ)</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>For <see langword="float"/> arrays, <paramref name="y"/> is rounded to <see langword="float"/>. Local derivatives follow elementwise <c>LogAddExp</c>, and the gradient of <paramref name="y"/> sums the elementwise contributions (computed in <typeparamref name="T"/> and widened to <see langword="double"/>).</para>
        /// <para><see langword="float"/> 配列では <paramref name="y"/> を <see langword="float"/> に丸めて適用します。局所微分は要素ごとの <c>LogAddExp</c> と同じ算出法に従い、<paramref name="y"/> の勾配は要素ごとの寄与の和（<typeparamref name="T"/> で計算して <see langword="double"/> へ拡幅）となります。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> LogAddExp<T>(VectorNode<T> x, ScalarNode y) where T : unmanaged, IFloatingPointIeee754<T>
            => ScalarCore(x, y, ArrayOperation.LogAddExpScalar);

        /// <summary>
        /// Computes <c>ln(eˣ + eˢ)</c> elementwise with a scalar.
        /// <para>行列の各要素とスカラーの <c>ln(eˣ + eˢ)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The input matrix node. / 入力行列ノード。</param>
        /// <param name="y">The scalar node or constant <c>s</c>. / スカラー <c>s</c>（ノードまたは定数）。</param>
        /// <returns>A matrix node holding <c>ln(eˣ + eˢ)</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>For <see langword="float"/> arrays, <paramref name="y"/> is rounded to <see langword="float"/>. Local derivatives follow elementwise <c>LogAddExp</c>, and the gradient of <paramref name="y"/> sums the elementwise contributions (computed in <typeparamref name="T"/> and widened to <see langword="double"/>).</para>
        /// <para><see langword="float"/> 配列では <paramref name="y"/> を <see langword="float"/> に丸めて適用します。局所微分は要素ごとの <c>LogAddExp</c> と同じ算出法に従い、<paramref name="y"/> の勾配は要素ごとの寄与の和（<typeparamref name="T"/> で計算して <see langword="double"/> へ拡幅）となります。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> LogAddExp<T>(MatrixNode<T> x, ScalarNode y) where T : unmanaged, IFloatingPointIeee754<T>
            => ScalarCore(x, y, ArrayOperation.LogAddExpScalar);

        /// <summary>
        /// Computes <c>ln(eˣ - eˢ)</c> elementwise with a scalar.
        /// <para>ベクトルの各要素とスカラーの <c>ln(eˣ - eˢ)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The minuend vector node. / 被減数ベクトルノード。</param>
        /// <param name="y">The subtrahend scalar node or constant <c>s</c>. / 減数となるスカラー <c>s</c>（ノードまたは定数）。</param>
        /// <returns>A vector node holding <c>ln(eˣ - eˢ)</c>. / 各要素の演算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>For <see langword="float"/> arrays, <paramref name="y"/> is rounded to <see langword="float"/>. Local derivatives follow elementwise <c>LogSubExp</c>, and the gradient of <paramref name="y"/> sums the elementwise contributions (computed in <typeparamref name="T"/> and widened to <see langword="double"/>).</para>
        /// <para><see langword="float"/> 配列では <paramref name="y"/> を <see langword="float"/> に丸めて適用します。局所微分は要素ごとの <c>LogSubExp</c> と同じ算出法に従い、<paramref name="y"/> の勾配は要素ごとの寄与の和（<typeparamref name="T"/> で計算して <see langword="double"/> へ拡幅）となります。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> LogSubExp<T>(VectorNode<T> x, ScalarNode y) where T : unmanaged, IFloatingPointIeee754<T>
            => ScalarCore(x, y, ArrayOperation.LogSubExpScalar);

        /// <summary>
        /// Computes <c>ln(eˣ - eˢ)</c> elementwise with a scalar.
        /// <para>行列の各要素とスカラーの <c>ln(eˣ - eˢ)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The minuend matrix node. / 被減数行列ノード。</param>
        /// <param name="y">The subtrahend scalar node or constant <c>s</c>. / 減数となるスカラー <c>s</c>（ノードまたは定数）。</param>
        /// <returns>A matrix node holding <c>ln(eˣ - eˢ)</c>. / 各要素の演算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>For <see langword="float"/> arrays, <paramref name="y"/> is rounded to <see langword="float"/>. Local derivatives follow elementwise <c>LogSubExp</c>, and the gradient of <paramref name="y"/> sums the elementwise contributions (computed in <typeparamref name="T"/> and widened to <see langword="double"/>).</para>
        /// <para><see langword="float"/> 配列では <paramref name="y"/> を <see langword="float"/> に丸めて適用します。局所微分は要素ごとの <c>LogSubExp</c> と同じ算出法に従い、<paramref name="y"/> の勾配は要素ごとの寄与の和（<typeparamref name="T"/> で計算して <see langword="double"/> へ拡幅）となります。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> LogSubExp<T>(MatrixNode<T> x, ScalarNode y) where T : unmanaged, IFloatingPointIeee754<T>
            => ScalarCore(x, y, ArrayOperation.LogSubExpScalar);
    }

    internal static VectorNode<T> UnaryCore<T>(VectorNode<T> x, ArrayOperation operation, bool approximate)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        Tape tape = x.Owner();
        var store = tape.Store<T>();
        bool record = store.RequiresGradient(x.View.Slot);
        tape.EnsureCanApply(record);

        int slot = store.AddPooled(x.Length, 1, record);
        Forward(operation, approximate, store.VectorView(x.View), store.WritableVector(slot));
        if (record)
        {
            tape.AppendArrayRecord<T>(operation, slot, x.View, default);
        }

        return new(tape, ArrayView.Whole(slot, x.Length, 1));
    }

    internal static MatrixNode<T> UnaryCore<T>(MatrixNode<T> x, ArrayOperation operation, bool approximate)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        Tape tape = x.Owner();
        var store = tape.Store<T>();
        bool record = store.RequiresGradient(x.View.Slot);
        tape.EnsureCanApply(record);

        int slot = store.AddPooled(x.Rows, x.Columns, record);
        Forward(operation, approximate, store.MatrixView(x.View), store.WritableMatrix(slot));
        if (record)
        {
            tape.AppendArrayRecord<T>(operation, slot, x.View, default);
        }

        return new(tape, ArrayView.Whole(slot, x.Rows, x.Columns));
    }

    internal static VectorNode<T> BinaryCore<T>(VectorNode<T> x, VectorNode<T> y, ArrayOperation operation)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        Tape tape = Owner(x.Tape, x.Generation, y.Tape, y.Generation);
        if (x.Length != y.Length)
        {
            throw new ArgumentException("The vectors must have the same length.", nameof(y));
        }

        var store = tape.Store<T>();
        bool record = store.RequiresGradient(x.View.Slot) || store.RequiresGradient(y.View.Slot);
        tape.EnsureCanApply(record);

        int slot = store.AddPooled(x.Length, 1, record);
        Forward(operation, store.VectorView(x.View), store.VectorView(y.View), store.WritableVector(slot));
        if (record)
        {
            tape.AppendArrayRecord<T>(operation, slot, x.View, y.View);
        }

        return new(tape, ArrayView.Whole(slot, x.Length, 1));
    }

    internal static MatrixNode<T> BinaryCore<T>(MatrixNode<T> x, MatrixNode<T> y, ArrayOperation operation)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        Tape tape = Owner(x.Tape, x.Generation, y.Tape, y.Generation);
        if (x.Rows != y.Rows || x.Columns != y.Columns)
        {
            throw new ArgumentException("The matrices must have the same shape.", nameof(y));
        }

        var store = tape.Store<T>();
        bool record = store.RequiresGradient(x.View.Slot) || store.RequiresGradient(y.View.Slot);
        tape.EnsureCanApply(record);

        int slot = store.AddPooled(x.Rows, x.Columns, record);
        Forward(operation, store.MatrixView(x.View), store.MatrixView(y.View), store.WritableMatrix(slot));
        if (record)
        {
            tape.AppendArrayRecord<T>(operation, slot, x.View, y.View);
        }

        return new(tape, ArrayView.Whole(slot, x.Rows, x.Columns));
    }

    internal static VectorNode<T> ScalarCore<T>(VectorNode<T> x, ScalarNode s, ArrayOperation operation)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        Tape tape = x.Owner();
        int scalarInput = ScalarInput(tape, s);
        double value = s.Value;
        var store = tape.Store<T>();
        bool record = scalarInput >= 0 || store.RequiresGradient(x.View.Slot);
        tape.EnsureCanApply(record);

        int slot = store.AddPooled(x.Length, 1, record);
        Forward(operation, store.VectorView(x.View), ArrayStore<T>.FromDouble(value), store.WritableVector(slot));
        if (record)
        {
            tape.AppendArrayRecord<T>(operation, slot, x.View, default, value, scalarInput);
        }

        return new(tape, ArrayView.Whole(slot, x.Length, 1));
    }

    internal static MatrixNode<T> ScalarCore<T>(MatrixNode<T> x, ScalarNode s, ArrayOperation operation)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        Tape tape = x.Owner();
        int scalarInput = ScalarInput(tape, s);
        double value = s.Value;
        var store = tape.Store<T>();
        bool record = scalarInput >= 0 || store.RequiresGradient(x.View.Slot);
        tape.EnsureCanApply(record);

        int slot = store.AddPooled(x.Rows, x.Columns, record);
        Forward(operation, store.MatrixView(x.View), ArrayStore<T>.FromDouble(value), store.WritableMatrix(slot));
        if (record)
        {
            tape.AppendArrayRecord<T>(operation, slot, x.View, default, value, scalarInput);
        }

        return new(tape, ArrayView.Whole(slot, x.Rows, x.Columns));
    }

    private static int ScalarInput(Tape tape, ScalarNode s)
    {
        Tape? scalarTape = s.Tape;
        if (scalarTape is null)
        {
            return -1;
        }

        scalarTape.ValidateGeneration(s.Generation);
        if (scalarTape != tape)
        {
            ThrowDifferentTapes();
        }

        return s.Index;
    }

    private static void Forward<T>(ArrayOperation operation, bool approximate, ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        switch (operation)
        {
            case ArrayOperation.Negate:
                XMath.Negate(x, destination);
                break;
            case ArrayOperation.Exp:
                _ = approximate ? XMath.Exp(x, true, destination) : XMath.Exp(x, destination);
                break;
            case ArrayOperation.Log:
                XMath.Log(x, destination);
                break;
            case ArrayOperation.Sqrt:
                XMath.Sqrt(x, destination);
                break;
            case ArrayOperation.Square:
                XMath.Square(x, destination);
                break;
            case ArrayOperation.Abs:
                XMath.Abs(x, destination);
                break;
            case ArrayOperation.Expm1:
                XMath.Expm1(x, destination);
                break;
            case ArrayOperation.Log1p:
                XMath.Log1p(x, destination);
                break;
            case ArrayOperation.Logistic:
                _ = approximate ? XMath.Logistic(x, true, destination) : XMath.Logistic(x, destination);
                break;
            case ArrayOperation.LogLogistic:
                XMath.LogLogistic(x, destination);
                break;
            case ArrayOperation.Logit:
                XMath.Logit(x, destination);
                break;
            case ArrayOperation.Softplus:
                _ = approximate ? XMath.Softplus(x, true, destination) : XMath.Softplus(x, destination);
                break;
            case ArrayOperation.Log1mExp:
                XMath.Log1mExp(x, destination);
                break;
            case ArrayOperation.Reciprocal:
                XMath.Reciprocal(x, destination);
                break;
        }
    }

    private static void Forward<T>(ArrayOperation operation, bool approximate, ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        switch (operation)
        {
            case ArrayOperation.Negate:
                XMath.Negate(x, destination);
                break;
            case ArrayOperation.Exp:
                _ = approximate ? XMath.Exp(x, true, destination) : XMath.Exp(x, destination);
                break;
            case ArrayOperation.Log:
                XMath.Log(x, destination);
                break;
            case ArrayOperation.Sqrt:
                XMath.Sqrt(x, destination);
                break;
            case ArrayOperation.Square:
                XMath.Square(x, destination);
                break;
            case ArrayOperation.Abs:
                XMath.Abs(x, destination);
                break;
            case ArrayOperation.Expm1:
                XMath.Expm1(x, destination);
                break;
            case ArrayOperation.Log1p:
                XMath.Log1p(x, destination);
                break;
            case ArrayOperation.Logistic:
                _ = approximate ? XMath.Logistic(x, true, destination) : XMath.Logistic(x, destination);
                break;
            case ArrayOperation.LogLogistic:
                XMath.LogLogistic(x, destination);
                break;
            case ArrayOperation.Logit:
                XMath.Logit(x, destination);
                break;
            case ArrayOperation.Softplus:
                _ = approximate ? XMath.Softplus(x, true, destination) : XMath.Softplus(x, destination);
                break;
            case ArrayOperation.Log1mExp:
                XMath.Log1mExp(x, destination);
                break;
            case ArrayOperation.Reciprocal:
                XMath.Reciprocal(x, destination);
                break;
        }
    }

    private static void Forward<T>(ArrayOperation operation, ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        switch (operation)
        {
            case ArrayOperation.Add:
                XMath.Add(x, y, destination);
                break;
            case ArrayOperation.Subtract:
                XMath.Subtract(x, y, destination);
                break;
            case ArrayOperation.Hadamard:
                XMath.Hadamard(x, y, destination);
                break;
            case ArrayOperation.Divide:
                XMath.Divide(x, y, destination);
                break;
            case ArrayOperation.Min:
                XMath.Min(x, y, destination);
                break;
            case ArrayOperation.Max:
                XMath.Max(x, y, destination);
                break;
            case ArrayOperation.Pow:
                XMath.Pow(x, y, destination);
                break;
            case ArrayOperation.LogAddExp:
                XMath.LogAddExp(x, y, destination);
                break;
            case ArrayOperation.LogSubExp:
                XMath.LogSubExp(x, y, destination);
                break;
        }
    }

    private static void Forward<T>(ArrayOperation operation, ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, MatrixSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        switch (operation)
        {
            case ArrayOperation.Add:
                XMath.Add(x, y, destination);
                break;
            case ArrayOperation.Subtract:
                XMath.Subtract(x, y, destination);
                break;
            case ArrayOperation.Hadamard:
                XMath.Hadamard(x, y, destination);
                break;
            case ArrayOperation.Divide:
                XMath.Divide(x, y, destination);
                break;
            case ArrayOperation.Min:
                XMath.Min(x, y, destination);
                break;
            case ArrayOperation.Max:
                XMath.Max(x, y, destination);
                break;
            case ArrayOperation.Pow:
                XMath.Pow(x, y, destination);
                break;
            case ArrayOperation.LogAddExp:
                XMath.LogAddExp(x, y, destination);
                break;
            case ArrayOperation.LogSubExp:
                XMath.LogSubExp(x, y, destination);
                break;
        }
    }

    // XMath has no scalar-minus-array overload, so ScalarSubtract uses Negate followed by Add (bit-identical to s - x except NaN payloads).
    private static void Forward<T>(ArrayOperation operation, ReadOnlyVectorSpan<T> x, T s, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        switch (operation)
        {
            case ArrayOperation.AddScalar:
                XMath.Add(x, s, destination);
                break;
            case ArrayOperation.SubtractScalar:
                XMath.Subtract(x, s, destination);
                break;
            case ArrayOperation.ScalarSubtract:
                XMath.Negate(x, destination);
                XMath.Add(destination.AsReadOnly(), s, destination);
                break;
            case ArrayOperation.MultiplyScalar:
                XMath.Multiply(x, s, destination);
                break;
            case ArrayOperation.DivideScalar:
                XMath.Divide(x, s, destination);
                break;
            case ArrayOperation.MinScalar:
                XMath.Min(x, s, destination);
                break;
            case ArrayOperation.MaxScalar:
                XMath.Max(x, s, destination);
                break;
            case ArrayOperation.PowScalar:
                XMath.Pow(x, s, destination);
                break;
            case ArrayOperation.LogAddExpScalar:
                XMath.LogAddExp(x, s, destination);
                break;
            case ArrayOperation.LogSubExpScalar:
                XMath.LogSubExp(x, s, destination);
                break;
        }
    }

    private static void Forward<T>(ArrayOperation operation, ReadOnlyMatrixSpan<T> x, T s, MatrixSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        switch (operation)
        {
            case ArrayOperation.AddScalar:
                XMath.Add(x, s, destination);
                break;
            case ArrayOperation.SubtractScalar:
                XMath.Subtract(x, s, destination);
                break;
            case ArrayOperation.ScalarSubtract:
                XMath.Negate(x, destination);
                XMath.Add(destination.AsReadOnly(), s, destination);
                break;
            case ArrayOperation.MultiplyScalar:
                XMath.Multiply(x, s, destination);
                break;
            case ArrayOperation.DivideScalar:
                XMath.Divide(x, s, destination);
                break;
            case ArrayOperation.MinScalar:
                XMath.Min(x, s, destination);
                break;
            case ArrayOperation.MaxScalar:
                XMath.Max(x, s, destination);
                break;
            case ArrayOperation.PowScalar:
                XMath.Pow(x, s, destination);
                break;
            case ArrayOperation.LogAddExpScalar:
                XMath.LogAddExp(x, s, destination);
                break;
            case ArrayOperation.LogSubExpScalar:
                XMath.LogSubExp(x, s, destination);
                break;
        }
    }
}
