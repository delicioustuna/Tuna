using System.Runtime.CompilerServices;

namespace Tuna.Differentiation;

public static partial class XMathSpecialExtensions
{
    extension(XMath.Special)
    {
        /// <summary>
        /// Computes the error function <c>erf(x)</c>.
        /// <para>誤差関数 <c>erf(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input scalar node. / 入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>erf(x)</c>. / <c>erf(x)</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>2e^(-x²)/√π</c>, with 0 at <c>±∞</c> and <c>NaN</c> for <c>NaN</c>. Representable tail derivatives remain accurate.</para>
        /// <para>局所微分は <c>2e^(-x²)/√π</c> とし、<c>±∞</c> では 0、<c>NaN</c> では <c>NaN</c> とします。表現可能な裾の局所微分の精度を保ちます。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode Erf(ScalarNode x)
        {
            double raw = x.Value;
            double value = XMath.Special.Erf(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            return ScalarNode.Unary(value, x, XMath.Special.ErfDerivative(raw));
        }

        /// <summary>
        /// Computes the complementary error function <c>erfc(x) = 1 - erf(x)</c>.
        /// <para>相補誤差関数 <c>erfc(x) = 1 - erf(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input scalar node. / 入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>erfc(x)</c>. / <c>erfc(x)</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>-2e^(-x²)/√π</c>, with 0 at <c>±∞</c> and <c>NaN</c> for <c>NaN</c>. Representable tail derivatives remain accurate.</para>
        /// <para>局所微分は <c>-2e^(-x²)/√π</c> とし、<c>±∞</c> では 0、<c>NaN</c> では <c>NaN</c> とします。表現可能な裾の局所微分の精度を保ちます。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode Erfc(ScalarNode x)
        {
            double raw = x.Value;
            double value = XMath.Special.Erfc(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            return ScalarNode.Unary(value, x, -XMath.Special.ErfDerivative(raw));
        }

        /// <summary>
        /// Computes the standard normal probability density function <c>φ(x) = e^(-x²/2)/√(2π)</c>.
        /// <para>標準正規分布の確率密度関数 <c>φ(x) = e^(-x²/2)/√(2π)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input scalar node. / 入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>φ(x)</c>. / <c>φ(x)</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>-x·φ(x)</c>, with 0 at <c>±∞</c> and <c>NaN</c> for <c>NaN</c>. Representable tail derivatives remain accurate even where the density underflows to 0.</para>
        /// <para>局所微分は <c>-x·φ(x)</c> とし、<c>±∞</c> では 0、<c>NaN</c> では <c>NaN</c> とします。密度が 0 にアンダーフローする裾でも、表現可能な局所微分の精度を保ちます。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode NormalPdf(ScalarNode x)
        {
            double raw = x.Value;
            double value = XMath.Special.NormalPdf(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            return ScalarNode.Unary(value, x, XMath.Special.NormalPdfDerivative(raw));
        }

        /// <summary>
        /// Computes the standard normal log-density <c>ln φ(x)</c>.
        /// <para>標準正規分布の対数確率密度 <c>ln φ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input scalar node. / 入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>ln φ(x)</c>. / <c>ln φ(x)</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>-x</c>, with <c>+∞</c> at <c>-∞</c>, <c>-∞</c> at <c>+∞</c>, and <c>NaN</c> for <c>NaN</c>.</para>
        /// <para>局所微分は <c>-x</c> とし、<c>-∞</c> では <c>+∞</c>、<c>+∞</c> では <c>-∞</c>、<c>NaN</c> では <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode NormalLogPdf(ScalarNode x)
        {
            double raw = x.Value;
            double value = XMath.Special.NormalLogPdf(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            return ScalarNode.Unary(value, x, -raw);
        }

        /// <summary>
        /// Computes the standard normal cumulative distribution function <c>Φ(x)</c>.
        /// <para>標準正規分布の累積分布関数 <c>Φ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input scalar node. / 入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>Φ(x)</c>. / <c>Φ(x)</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>φ(x)</c>, with 0 at <c>±∞</c> and <c>NaN</c> for <c>NaN</c>.</para>
        /// <para>局所微分は <c>φ(x)</c> とし、<c>±∞</c> では 0、<c>NaN</c> では <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode NormalCdf(ScalarNode x)
        {
            double raw = x.Value;
            double value = XMath.Special.NormalCdf(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            return ScalarNode.Unary(value, x, XMath.Special.NormalPdf(raw));
        }

        /// <summary>
        /// Computes the standard normal survival function <c>Q(x) = 1 - Φ(x)</c>.
        /// <para>標準正規分布の生存関数 <c>Q(x) = 1 - Φ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input scalar node. / 入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>Q(x)</c>. / <c>Q(x)</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>-φ(x)</c>, with 0 at <c>±∞</c> and <c>NaN</c> for <c>NaN</c>.</para>
        /// <para>局所微分は <c>-φ(x)</c> とし、<c>±∞</c> では 0、<c>NaN</c> では <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode NormalSf(ScalarNode x)
        {
            double raw = x.Value;
            double value = XMath.Special.NormalSf(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            return ScalarNode.Unary(value, x, -XMath.Special.NormalPdf(raw));
        }

        /// <summary>
        /// Computes the natural logarithm of the standard normal cumulative distribution function, <c>ln Φ(x)</c>.
        /// <para>標準正規分布の累積分布関数の自然対数 <c>ln Φ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input scalar node. / 入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>ln Φ(x)</c>. / <c>ln Φ(x)</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is the inverse Mills ratio <c>φ(x)/Φ(x)</c>, with <c>+∞</c> at <c>-∞</c>, 0 at <c>+∞</c>, and <c>NaN</c> for <c>NaN</c>. Accuracy is preserved in the negative tail even where <c>Φ(x)</c> underflows to 0.</para>
        /// <para>局所微分は <c>φ(x)/Φ(x)</c> とし、<c>-∞</c> では <c>+∞</c>、<c>+∞</c> では 0、<c>NaN</c> では <c>NaN</c> とします。<c>Φ(x)</c> が 0 にアンダーフローする負の裾でも精度を保ちます。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode NormalLogCdf(ScalarNode x)
        {
            double raw = x.Value;
            double value = XMath.Special.NormalLogCdf(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            return ScalarNode.Unary(value, x, XMath.Special.NormalReverseHazard(raw));
        }

        /// <summary>
        /// Computes the natural logarithm of the standard normal survival function, <c>ln Q(x)</c>.
        /// <para>標準正規分布の生存関数の自然対数 <c>ln Q(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input scalar node. / 入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>ln Q(x)</c>. / <c>ln Q(x)</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>-φ(x)/Q(x)</c>, with 0 at <c>-∞</c>, <c>-∞</c> at <c>+∞</c>, and <c>NaN</c> for <c>NaN</c>. Accuracy is preserved in the positive tail even where <c>Q(x)</c> underflows to 0.</para>
        /// <para>局所微分は <c>-φ(x)/Q(x)</c> とし、<c>-∞</c> では 0、<c>+∞</c> では <c>-∞</c>、<c>NaN</c> では <c>NaN</c> とします。<c>Q(x)</c> が 0 にアンダーフローする正の裾でも精度を保ちます。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode NormalLogSf(ScalarNode x)
        {
            double raw = x.Value;
            double value = XMath.Special.NormalLogSf(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            return ScalarNode.Unary(value, x, -XMath.Special.NormalHazard(raw));
        }
    }
}
