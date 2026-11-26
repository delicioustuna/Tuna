using System.Runtime.CompilerServices;

namespace Tuna.Differentiation;

public static partial class XMathExtensions
{
    extension(XMath)
    {
        /// <summary>
        /// Computes the natural exponential function <c>eˣ</c>.
        /// <para>自然指数関数 <c>eˣ</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input scalar node. / 入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>eˣ</c>. / <c>eˣ</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>eˣ</c>.</para>
        /// <para>局所微分は <c>eˣ</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode Exp(ScalarNode x)
        {
            double raw = x.Value;
            double value = Math.Exp(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            return ScalarNode.Unary(value, x, value);
        }

        /// <summary>
        /// Computes the natural logarithm <c>ln x</c>.
        /// <para>自然対数 <c>ln x</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input scalar node. / 入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>ln x</c>. / <c>ln x</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>1/x</c>, with the right-hand limit <c>+∞</c> at <c>±0</c> and <c>NaN</c> for negative inputs.</para>
        /// <para>局所微分は <c>1/x</c> とします。<c>±0</c> では右側極限の <c>+∞</c>、負の入力では <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode Log(ScalarNode x)
        {
            double raw = x.Value;
            double value = XMath.Log<double>(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            double derivative = raw == 0 ? double.PositiveInfinity : raw < 0 ? double.NaN : 1 / raw;
            return ScalarNode.Unary(value, x, derivative);
        }

        /// <summary>
        /// Computes the square root <c>√x</c>.
        /// <para>平方根 <c>√x</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input scalar node. / 入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>√x</c>. / <c>√x</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>1/(2√x)</c>, with the right-hand limit <c>+∞</c> at <c>±0</c> and <c>NaN</c> for negative inputs.</para>
        /// <para>局所微分は <c>1/(2√x)</c> とします。<c>±0</c> では右側極限の <c>+∞</c>、負の入力では <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode Sqrt(ScalarNode x)
        {
            double raw = x.Value;
            double value = XMath.Sqrt<double>(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            double derivative = raw == 0 ? double.PositiveInfinity : 0.5 / value;
            return ScalarNode.Unary(value, x, derivative);
        }

        /// <summary>
        /// Computes the square <c>x²</c>.
        /// <para>2 乗 <c>x²</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input scalar node. / 入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>x²</c>. / <c>x²</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>2x</c>.</para>
        /// <para>局所微分は <c>2x</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode Square(ScalarNode x)
        {
            double raw = x.Value;
            double value = XMath.Square<double>(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            return ScalarNode.Unary(value, x, 2 * raw);
        }

        /// <summary>
        /// Computes the absolute value <c>|x|</c>.
        /// <para>絶対値 <c>|x|</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input scalar node. / 入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>|x|</c>. / <c>|x|</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>-1</c> for negative inputs, <c>+1</c> for positive inputs and <c>±0</c> (right-hand derivative), and <c>NaN</c> for <c>NaN</c>.</para>
        /// <para>局所微分は負の入力で <c>-1</c>、正の入力および <c>±0</c>（右側微分）で <c>+1</c>、<c>NaN</c> では <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode Abs(ScalarNode x)
        {
            double raw = x.Value;
            double value = XMath.Abs<double>(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            double derivative = double.IsNaN(raw) ? double.NaN : raw < 0 ? -1 : 1;
            return ScalarNode.Unary(value, x, derivative);
        }

        /// <summary>
        /// Returns the smaller of two scalar nodes, <c>min(left, right)</c>.
        /// <para>2 つのスカラーノードの最小値 <c>min(left, right)</c> を返します。</para>
        /// </summary>
        /// <param name="left">The first scalar node. / 左側のスカラーノード。</param>
        /// <param name="right">The second scalar node. / 右側のスカラーノード。</param>
        /// <returns>A scalar node holding <c>min(left, right)</c>. / <c>min(left, right)</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The selected input receives local derivative 1 and the other receives 0. For equal values (including signed zeros), <paramref name="left"/> receives 1 and <paramref name="right"/> receives 0. If either input is <c>NaN</c>, both local derivatives are <c>NaN</c>.</para>
        /// <para>選択された入力の局所微分を 1、もう一方を 0 とします。同値（符号付きゼロを含む）の場合は <paramref name="left"/> を 1、<paramref name="right"/> を 0 とします。いずれかが <c>NaN</c> の場合は両方の局所微分を <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node belongs to a stale generation, the nodes belong to different tapes, or the tape's backward pass has completed. / 世代が古い、異なるテープに属している、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode Min(ScalarNode left, ScalarNode right)
        {
            double a = left.Value;
            double b = right.Value;
            double value = XMath.Min<double>(a, b);
            if (left.Tape is null && right.Tape is null)
            {
                return new(value);
            }

            bool nan = double.IsNaN(a) || double.IsNaN(b);
            double dl = left.Tape is null ? 0 : nan ? double.NaN : a <= b ? 1 : 0;
            double dr = right.Tape is null ? 0 : nan ? double.NaN : a <= b ? 0 : 1;
            return ScalarNode.Binary(value, left, dl, right, dr);
        }

        /// <summary>
        /// Returns the larger of two scalar nodes, <c>max(left, right)</c>.
        /// <para>2 つのスカラーノードの最大値 <c>max(left, right)</c> を返します。</para>
        /// </summary>
        /// <param name="left">The first scalar node. / 左側のスカラーノード。</param>
        /// <param name="right">The second scalar node. / 右側のスカラーノード。</param>
        /// <returns>A scalar node holding <c>max(left, right)</c>. / <c>max(left, right)</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The selected input receives local derivative 1 and the other receives 0. For equal values (including signed zeros), <paramref name="left"/> receives 1 and <paramref name="right"/> receives 0. If either input is <c>NaN</c>, both local derivatives are <c>NaN</c>.</para>
        /// <para>選択された入力の局所微分を 1、もう一方を 0 とします。同値（符号付きゼロを含む）の場合は <paramref name="left"/> を 1、<paramref name="right"/> を 0 とします。いずれかが <c>NaN</c> の場合は両方の局所微分を <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if either node belongs to a stale generation, the nodes belong to different tapes, or the tape's backward pass has completed. / 世代が古い、異なるテープに属している、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode Max(ScalarNode left, ScalarNode right)
        {
            double a = left.Value;
            double b = right.Value;
            double value = XMath.Max<double>(a, b);
            if (left.Tape is null && right.Tape is null)
            {
                return new(value);
            }

            bool nan = double.IsNaN(a) || double.IsNaN(b);
            double dl = left.Tape is null ? 0 : nan ? double.NaN : a >= b ? 1 : 0;
            double dr = right.Tape is null ? 0 : nan ? double.NaN : a >= b ? 0 : 1;
            return ScalarNode.Binary(value, left, dl, right, dr);
        }

        /// <summary>
        /// Computes <c>eˣ - 1</c> accurately near zero.
        /// <para>0 の近傍でも精度を保って <c>eˣ - 1</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input scalar node. / 入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>eˣ - 1</c>. / <c>eˣ - 1</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>eˣ</c>.</para>
        /// <para>局所微分は <c>eˣ</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode Expm1(ScalarNode x)
        {
            double raw = x.Value;
            double value = XMath.Expm1<double>(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            return ScalarNode.Unary(value, x, Math.Exp(raw));
        }

        /// <summary>
        /// Computes <c>ln(1 + x)</c> accurately near zero.
        /// <para>0 の近傍でも精度を保って <c>ln(1 + x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input scalar node. / 入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>ln(1 + x)</c>. / <c>ln(1 + x)</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>1/(1 + x)</c>, with the right-hand limit <c>+∞</c> at <c>-1</c> and <c>NaN</c> for inputs less than <c>-1</c>.</para>
        /// <para>局所微分は <c>1/(1 + x)</c> とします。<c>-1</c> では右側極限の <c>+∞</c>、<c>-1</c> 未満の入力では <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode Log1p(ScalarNode x)
        {
            double raw = x.Value;
            double value = XMath.Log1p<double>(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            double derivative = raw < -1 ? double.NaN : 1 / (1 + raw);
            return ScalarNode.Unary(value, x, derivative);
        }
    }
}
