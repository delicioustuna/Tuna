using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace Tuna.Differentiation;

/// <summary>
/// Provides <see cref="XMath.Special"/> extension methods for differentiable scalar nodes.
/// <para><see cref="ScalarNode"/> に適用できる特殊関数を <see cref="XMath.Special"/> の拡張メソッドとして提供します。</para>
/// </summary>
[EditorBrowsable(EditorBrowsableState.Never)]
public static partial class XMathSpecialExtensions
{
    extension(XMath.Special)
    {
        /// <summary>
        /// Computes the natural logarithm of the absolute value of the gamma function, <c>ln |Γ(x)|</c>.
        /// <para>ガンマ関数の絶対値の自然対数 <c>ln |Γ(x)|</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input scalar node. / 入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>ln |Γ(x)|</c>. / <c>ln |Γ(x)|</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>ψ(x)</c>, with right-hand limit <c>-∞</c> at <c>+0</c>, left-hand limit <c>+∞</c> at <c>-0</c>, and <c>NaN</c> at negative integers and <c>-∞</c>.</para>
        /// <para>局所微分は <c>ψ(x)</c> とします。<c>+0</c> では右側極限の <c>-∞</c>、<c>-0</c> では左側極限の <c>+∞</c>、負の整数および <c>-∞</c> では <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode LogGamma(ScalarNode x)
        {
            double raw = x.Value;
            double value = XMath.Special.LogGamma(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            return ScalarNode.Unary(value, x, XMath.Special.Digamma(raw));
        }

        /// <summary>
        /// Computes the digamma function <c>ψ(x)</c>.
        /// <para>ディガンマ関数 <c>ψ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input scalar node. / 入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>ψ(x)</c>. / <c>ψ(x)</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>ψ₁(x)</c>, with <c>+∞</c> at <c>±0</c>, <c>NaN</c> at negative integers and <c>-∞</c>, and 0 at <c>+∞</c>.</para>
        /// <para>局所微分は <c>ψ₁(x)</c> とします。<c>±0</c> では <c>+∞</c>、負の整数および <c>-∞</c> では <c>NaN</c>、<c>+∞</c> では 0 とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode Digamma(ScalarNode x)
        {
            double raw = x.Value;
            double value = XMath.Special.Digamma(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            // Trigamma returns +∞ at negative integers; propagate NaN wherever Digamma itself is NaN.
            double derivative = double.IsNaN(value) ? double.NaN : XMath.Special.Trigamma(raw);
            return ScalarNode.Unary(value, x, derivative);
        }

        /// <summary>
        /// Computes the natural logarithm of the absolute value of the beta function, <c>ln |B(a, b)|</c>.
        /// <para>ベータ関数の絶対値の自然対数 <c>ln |B(a, b)|</c> を計算します。</para>
        /// </summary>
        /// <param name="a">The first parameter node. / 第 1 引数のスカラーノード。</param>
        /// <param name="b">The second parameter node. / 第 2 引数のスカラーノード。</param>
        /// <returns>A scalar node holding <c>ln |B(a, b)|</c>. / <c>ln |B(a, b)|</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivatives are <c>ψ(a) - ψ(a + b)</c> and <c>ψ(b) - ψ(a + b)</c>. Pole and infinity conventions follow <see cref="XMath.Special.LogBetaDerivative(double, double)"/>.</para>
        /// <para>局所微分は <c>ψ(a) - ψ(a + b)</c> および <c>ψ(b) - ψ(a + b)</c> とします。極および無限大の扱いは <see cref="XMath.Special.LogBetaDerivative(double, double)"/> に従います。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node belongs to a stale generation, the nodes belong to different tapes, or the tape's backward pass has completed. / 世代が古い、異なるテープに属している、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode LogBeta(ScalarNode a, ScalarNode b)
        {
            double rawA = a.Value;
            double rawB = b.Value;
            double value = XMath.Special.LogBeta(rawA, rawB);
            if (a.Tape is null && b.Tape is null)
            {
                return new(value);
            }

            double da = a.Tape is null ? 0 : XMath.Special.LogBetaDerivative(rawA, rawB);
            double db = b.Tape is null ? 0 : XMath.Special.LogBetaDerivative(rawB, rawA);
            return ScalarNode.Binary(value, a, da, b, db);
        }
    }
}
