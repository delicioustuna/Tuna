using System.Numerics;
using System.Runtime.CompilerServices;

namespace Tuna.Differentiation;

public static partial class XMathExtensions
{
    extension(XMath)
    {
        /// <summary>
        /// Computes <c>√(x² + y²)</c> and records its partial derivatives.
        /// <para><c>√(x² + y²)</c> を計算し、偏微分を記録します。</para>
        /// </summary>
        /// <param name="x">The first scalar node. / 第1スカラーノード。</param>
        /// <param name="y">The second scalar node. / 第2スカラーノード。</param>
        /// <returns>A scalar node holding <c>√(x² + y²)</c>. / <c>√(x² + y²)</c> を格納したスカラーノード。</returns>
        /// <remarks>
        /// <para>
        /// For finite nonzero pairs, the partial derivatives are the inputs divided by the Euclidean norm, with rescaling when the norm overflows or is subnormal.
        /// Both partials are zero at the origin. If exactly one input is infinite, its partial is its sign and the other partial is zero (including when the other input is NaN).
        /// Two infinities or a NaN paired with a finite input yield two NaN partials.
        /// </para>
        /// <para>
        /// 有限かつ非ゼロの入力では、各入力をユークリッドノルムで割って偏微分を求め、ノルムがオーバーフローする場合や非正規化数の場合はスケールを正規化します。
        /// 原点では両方の偏微分を 0 とします。一方だけが無限大の場合、その側の偏微分は符号、もう一方は NaN の場合も含めて 0 とします。
        /// 両方が無限大、または有限値と NaN の組み合わせでは、両方の偏微分を NaN とします。
        /// </para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node belongs to a stale generation, the nodes belong to different tapes, or the tape's backward pass has completed. / 世代が古い、異なるテープに属している、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode Hypot(ScalarNode x, ScalarNode y)
        {
            double a = x.Value, b = y.Value;
            double value = XMath.Hypot(a, b);
            if (x.Tape is null && y.Tape is null)
            {
                return new(value);
            }

            double dx = x.Tape is null ? 0 : HypotDerivative<double>.Invoke(a, b, value);
            double dy = y.Tape is null ? 0 : HypotDerivative<double>.Invoke(b, a, value);
            return ScalarNode.Binary(value, x, dx, y, dy);
        }

        /// <summary>
        /// Computes <c>√(x² + y²)</c> elementwise for two vectors.
        /// <para>2 つのベクトルの要素ごとに <c>√(x² + y²)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The first vector node. / 第1ベクトルノード。</param>
        /// <param name="y">The second vector node. / 第2ベクトルノード。</param>
        /// <returns>A vector node holding the elementwise results. / 各要素の計算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>Each element follows the derivative rules of <c>Hypot(ScalarNode, ScalarNode)</c>.</para>
        /// <para>各要素の偏微分は <c>Hypot(ScalarNode, ScalarNode)</c> と同じ定義に従います。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">The input lengths differ. / 入力の長さが一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Hypot<T>(VectorNode<T> x, VectorNode<T> y) where T : unmanaged, IFloatingPointIeee754<T>
            => BinaryCore(x, y, ArrayOperation.Hypot);

        /// <summary>
        /// Computes <c>√(x² + y²)</c> elementwise for two matrices.
        /// <para>2 つの行列の要素ごとに <c>√(x² + y²)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The first matrix node. / 第1行列ノード。</param>
        /// <param name="y">The second matrix node. / 第2行列ノード。</param>
        /// <returns>A matrix node holding the elementwise results. / 各要素の計算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>Each element follows the derivative rules of <c>Hypot(ScalarNode, ScalarNode)</c>.</para>
        /// <para>各要素の偏微分は <c>Hypot(ScalarNode, ScalarNode)</c> と同じ定義に従います。</para>
        /// </remarks>
        /// <exception cref="ArgumentException">The input shapes differ. / 入力の形状が一致しない場合。</exception>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Hypot<T>(MatrixNode<T> x, MatrixNode<T> y) where T : unmanaged, IFloatingPointIeee754<T>
            => BinaryCore(x, y, ArrayOperation.Hypot);

        /// <summary>
        /// Computes <c>√(x[i]² + y²)</c> for each element of a vector.
        /// <para>ベクトルの要素ごとに <c>√(x[i]² + y²)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The vector node. / ベクトルノード。</param>
        /// <param name="y">The shared scalar node. / 各要素に共通のスカラーノード。</param>
        /// <returns>A vector node holding the elementwise results. / 各要素の計算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>The scalar is rounded to <typeparamref name="T"/> for evaluation. Each element follows the derivative rules of <c>Hypot(ScalarNode, ScalarNode)</c>.</para>
        /// <para>スカラーは <typeparamref name="T"/> へ丸めて評価します。各要素の偏微分は <c>Hypot(ScalarNode, ScalarNode)</c> と同じ定義に従います。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Hypot<T>(VectorNode<T> x, ScalarNode y) where T : unmanaged, IFloatingPointIeee754<T>
            => ScalarCore(x, y, ArrayOperation.HypotScalar);

        /// <summary>
        /// Computes <c>√(x[i,j]² + y²)</c> for each element of a matrix.
        /// <para>行列の要素ごとに <c>√(x[i,j]² + y²)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The matrix node. / 行列ノード。</param>
        /// <param name="y">The shared scalar node. / 各要素に共通のスカラーノード。</param>
        /// <returns>A matrix node holding the elementwise results. / 各要素の計算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>The scalar is rounded to <typeparamref name="T"/> for evaluation. Each element follows the derivative rules of <c>Hypot(ScalarNode, ScalarNode)</c>.</para>
        /// <para>スカラーは <typeparamref name="T"/> へ丸めて評価します。各要素の偏微分は <c>Hypot(ScalarNode, ScalarNode)</c> と同じ定義に従います。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Hypot<T>(MatrixNode<T> x, ScalarNode y) where T : unmanaged, IFloatingPointIeee754<T>
            => ScalarCore(x, y, ArrayOperation.HypotScalar);

        /// <summary>
        /// Computes <c>√(x² + y[i]²)</c> for each element of a vector.
        /// <para>ベクトルの要素ごとに <c>√(x² + y[i]²)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The shared scalar node. / 各要素に共通のスカラーノード。</param>
        /// <param name="y">The vector node. / ベクトルノード。</param>
        /// <returns>A vector node holding the elementwise results. / 各要素の計算結果を格納したベクトルノード。</returns>
        /// <remarks>
        /// <para>Equivalent to <c>Hypot(y, x)</c>.</para>
        /// <para><c>Hypot(y, x)</c> と同じ計算を行います。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static VectorNode<T> Hypot<T>(ScalarNode x, VectorNode<T> y) where T : unmanaged, IFloatingPointIeee754<T>
            => ScalarCore(y, x, ArrayOperation.HypotScalar);

        /// <summary>
        /// Computes <c>√(x² + y[i,j]²)</c> for each element of a matrix.
        /// <para>行列の要素ごとに <c>√(x² + y[i,j]²)</c> を計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
        /// <param name="x">The shared scalar node. / 各要素に共通のスカラーノード。</param>
        /// <param name="y">The matrix node. / 行列ノード。</param>
        /// <returns>A matrix node holding the elementwise results. / 各要素の計算結果を格納した行列ノード。</returns>
        /// <remarks>
        /// <para>Equivalent to <c>Hypot(y, x)</c>.</para>
        /// <para><c>Hypot(y, x)</c> と同じ計算を行います。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node is uninitialized or stale, the nodes belong to different tapes, or the tape's backward pass has completed. / ノードが未初期化または世代が古い、異なるテープに属している、あるいは逆伝播を完了したテープへ記録しようとした場合。</exception>
        public static MatrixNode<T> Hypot<T>(ScalarNode x, MatrixNode<T> y) where T : unmanaged, IFloatingPointIeee754<T>
            => ScalarCore(y, x, ArrayOperation.HypotScalar);
    }
}
