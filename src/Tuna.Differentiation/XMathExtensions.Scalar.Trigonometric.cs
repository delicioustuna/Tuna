using System.Runtime.CompilerServices;

namespace Tuna.Differentiation;

public static partial class XMathExtensions
{
    extension(XMath)
    {
        /// <summary>
        /// Computes the sine <c>sin(x)</c>.
        /// <para>正弦 <c>sin(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The angle in radians. / ラジアン単位の角度。</param>
        /// <returns>A scalar node holding <c>sin(x)</c>. / <c>sin(x)</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>cos(x)</c>. For <c>±∞</c> and <c>NaN</c>, both the value and local derivative are <c>NaN</c>.</para>
        /// <para>局所微分は <c>cos(x)</c> とします。<c>±∞</c> および <c>NaN</c> では値と局所微分の両方を <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode Sin(ScalarNode x)
        {
            double raw = x.Value;
            double value = Math.Sin(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            return ScalarNode.Unary(value, x, Math.Cos(raw));
        }

        /// <summary>
        /// Computes the cosine <c>cos(x)</c>.
        /// <para>余弦 <c>cos(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The angle in radians. / ラジアン単位の角度。</param>
        /// <returns>A scalar node holding <c>cos(x)</c>. / <c>cos(x)</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>-sin(x)</c>. For <c>±∞</c> and <c>NaN</c>, both the value and local derivative are <c>NaN</c>.</para>
        /// <para>局所微分は <c>-sin(x)</c> とします。<c>±∞</c> および <c>NaN</c> では値と局所微分の両方を <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode Cos(ScalarNode x)
        {
            double raw = x.Value;
            double value = Math.Cos(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            return ScalarNode.Unary(value, x, -Math.Sin(raw));
        }

        /// <summary>
        /// Computes the tangent <c>tan(x)</c>.
        /// <para>正接 <c>tan(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The angle in radians. / ラジアン単位の角度。</param>
        /// <returns>A scalar node holding <c>tan(x)</c>. / <c>tan(x)</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>1 + tan²(x)</c>. For <c>±∞</c> and <c>NaN</c>, both the value and local derivative are <c>NaN</c>.</para>
        /// <para>局所微分は <c>1 + tan²(x)</c> とします。<c>±∞</c> および <c>NaN</c> では値と局所微分の両方を <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode Tan(ScalarNode x)
        {
            double raw = x.Value;
            double value = Math.Tan(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            return ScalarNode.Unary(value, x, 1 + value * value);
        }

        /// <summary>
        /// Computes the hyperbolic tangent <c>tanh(x)</c>.
        /// <para>双曲線正接 <c>tanh(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input scalar node. / 入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>tanh(x)</c>. / <c>tanh(x)</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>1/cosh²(x)</c>, with 0 at <c>±∞</c> and <c>NaN</c> for <c>NaN</c>. Representable tail derivatives remain accurate even where <c>tanh(x)</c> rounds to <c>±1</c>.</para>
        /// <para>局所微分は <c>1/cosh²(x)</c> とし、<c>±∞</c> では 0、<c>NaN</c> では <c>NaN</c> とします。値が <c>±1</c> に丸まる裾でも、表現可能な局所微分の精度を保ちます。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode Tanh(ScalarNode x)
        {
            double raw = x.Value;
            double value = XMath.Tanh<double>(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            double t = Math.Exp(-Math.Abs(raw));
            return ScalarNode.Unary(value, x, TanhDerivativeFromExp<double>.Invoke(t));
        }

        /// <summary>
        /// Computes the inverse tangent <c>atan(x)</c>.
        /// <para>逆正接 <c>atan(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input scalar node. / 入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>atan(x)</c>. / <c>atan(x)</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>1/(1 + x²)</c>, with 0 at <c>±∞</c> and <c>NaN</c> for <c>NaN</c>. Overflow of <c>x²</c> is avoided for large finite inputs.</para>
        /// <para>局所微分は <c>1/(1 + x²)</c> とし、<c>±∞</c> では 0、<c>NaN</c> では <c>NaN</c> とします。大きな有限値でも <c>x²</c> のオーバーフローを回避します。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode Atan(ScalarNode x)
        {
            double raw = x.Value;
            double value = Math.Atan(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            double derivative;
            if (Math.Abs(raw) <= 1)
            {
                derivative = 1 / (1 + raw * raw);
            }
            else
            {
                double reciprocal = 1 / raw;
                double square = reciprocal * reciprocal;
                derivative = square / (1 + square);
            }

            return ScalarNode.Unary(value, x, derivative);
        }
    }
}
