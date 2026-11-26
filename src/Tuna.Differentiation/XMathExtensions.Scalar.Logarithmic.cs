using System.Runtime.CompilerServices;

namespace Tuna.Differentiation;

public static partial class XMathExtensions
{
    private const double Ln2 = 0.6931471805599453;

    extension(XMath)
    {
        /// <summary>
        /// Computes the logistic sigmoid function <c>σ(x) = 1/(1 + e⁻ˣ)</c>.
        /// <para>ロジスティック関数 <c>σ(x) = 1/(1 + e⁻ˣ)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input scalar node. / 入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>σ(x)</c>. / <c>σ(x)</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>σ(x)σ(-x)</c>, preserving accuracy in the positive tail even where the value rounds to 1.</para>
        /// <para>局所微分は <c>σ(x)σ(-x)</c> とします。値が 1 に丸まる正の裾でも、表現可能な局所微分の精度を保ちます。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode Logistic(ScalarNode x)
        {
            double raw = x.Value;
            double value = XMath.Logistic<double>(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            // Avoid value * (1 - value), which underflows to 0 in the positive tail where value rounds to 1.
            double t = Math.Exp(-Math.Abs(raw));
            double denominator = 1 + t;
            return ScalarNode.Unary(value, x, t / (denominator * denominator));
        }

        /// <summary>
        /// Computes the natural logarithm of the logistic sigmoid function, <c>ln σ(x) = -ln(1 + e⁻ˣ)</c>.
        /// <para>ロジスティック関数の自然対数 <c>ln σ(x) = -ln(1 + e⁻ˣ)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input scalar node. / 入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>ln σ(x)</c>. / <c>ln σ(x)</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>σ(-x)</c>.</para>
        /// <para>局所微分は <c>σ(-x)</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode LogLogistic(ScalarNode x)
        {
            double raw = x.Value;
            double value = XMath.LogLogistic<double>(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            return ScalarNode.Unary(value, x, XMath.Logistic<double>(-raw));
        }

        /// <summary>
        /// Computes the logit function <c>ln(p/(1 - p))</c>.
        /// <para>ロジット関数 <c>ln(p/(1 - p))</c> を計算します。</para>
        /// </summary>
        /// <param name="p">The input probability node. / 確率を表す入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>ln(p/(1 - p))</c>. / <c>ln(p/(1 - p))</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>1/(p(1 - p))</c>, with <c>+∞</c> at the endpoints <c>0</c> and <c>1</c>, and <c>NaN</c> outside <c>[0, 1]</c> or for <c>NaN</c>.</para>
        /// <para>局所微分は <c>1/(p(1 - p))</c> とします。区間 <c>[0, 1]</c> の端点では <c>+∞</c>、区間外または <c>NaN</c> では <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode Logit(ScalarNode p)
        {
            double raw = p.Value;
            double value = XMath.Logit<double>(raw);
            if (p.Tape is null)
            {
                return new(value);
            }

            double derivative;
            if (raw is not (>= 0 and <= 1))
            {
                derivative = double.NaN;
            }
            else if (raw == 0 || raw == 1)
            {
                derivative = double.PositiveInfinity;
            }
            else
            {
                derivative = 1 / (raw * (1 - raw));
            }

            return ScalarNode.Unary(value, p, derivative);
        }

        /// <summary>
        /// Computes the softplus function <c>ln(1 + eˣ)</c>.
        /// <para>Softplus 関数 <c>ln(1 + eˣ)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input scalar node. / 入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>ln(1 + eˣ)</c>. / <c>ln(1 + eˣ)</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is the logistic sigmoid <c>σ(x)</c>.</para>
        /// <para>局所微分はロジスティック関数 <c>σ(x)</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode Softplus(ScalarNode x)
        {
            double raw = x.Value;
            double value = XMath.Softplus<double>(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            return ScalarNode.Unary(value, x, XMath.Logistic<double>(raw));
        }

        /// <summary>
        /// Computes <c>ln(1 - eˣ)</c> on the non-positive domain.
        /// <para>非正の定義域で <c>ln(1 - eˣ)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input scalar node. / 入力スカラーノード。</param>
        /// <returns>A scalar node holding <c>ln(1 - eˣ)</c>. / <c>ln(1 - eˣ)</c> を値とするスカラーノード。</returns>
        /// <remarks>
        /// <para>The local derivative is <c>-eˣ/(1 - eˣ)</c>, preserving accuracy near zero and in the negative tail. It is <c>-∞</c> (left-hand limit) at <c>±0</c>, and <c>NaN</c> for positive inputs or <c>NaN</c>.</para>
        /// <para>局所微分は <c>-eˣ/(1 - eˣ)</c> とし、0 付近と負の裾でも精度を保ちます。<c>±0</c> では左側極限の <c>-∞</c>、正の入力または <c>NaN</c> では <c>NaN</c> とします。</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if recording to a stale generation or a tape whose backward pass has completed. / 世代が古い、または逆伝播を完了したテープへ記録しようとした場合。</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ScalarNode Log1mExp(ScalarNode x)
        {
            double raw = x.Value;
            double value = XMath.Log1mExp<double>(raw);
            if (x.Tape is null)
            {
                return new(value);
            }

            double derivative;
            if (double.IsNaN(raw) || raw > 0)
            {
                derivative = double.NaN;
            }
            else if (raw == 0)
            {
                derivative = double.NegativeInfinity;
            }
            else if (raw < -Ln2)
            {
                // For x < -ln 2, eˣ < 1/2 so 1 - eˣ has no cancellation, and this preserves subnormal tails down to x ≈ -745
                // (whereas -1/expm1(-x) overflows its denominator for x < -709.78).
                double t = Math.Exp(raw);
                derivative = -t / (1 - t);
            }
            else
            {
                derivative = -1 / XMath.Expm1<double>(-raw);
            }

            return ScalarNode.Unary(value, x, derivative);
        }
    }
}
