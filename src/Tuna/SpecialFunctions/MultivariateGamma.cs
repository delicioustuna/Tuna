using static Tuna.XMath.Constants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        /// <summary>
        /// Computes the natural logarithm of the multivariate gamma function <c>ln Γₚ(a)</c>.
        /// <para>多変量ガンマ関数の自然対数 <c>ln Γₚ(a)</c> を計算します。</para>
        /// </summary>
        /// <param name="a">The argument, which must satisfy <c>a &gt; (dimension - 1) / 2</c>. / 引数（<c>a &gt; (dimension - 1) / 2</c>）。</param>
        /// <param name="dimension">The dimension <c>p</c>, which must be at least 1. / 次元 <c>p</c>（1 以上）。</param>
        /// <returns>The value of <c>ln Γₚ(a)</c>, or <see cref="double.NaN"/> if <paramref name="dimension"/> &lt; 1, <c>a ≤ (dimension - 1) / 2</c>, or <paramref name="a"/> is <see cref="double.NaN"/>. / <c>ln Γₚ(a)</c> の値。<paramref name="dimension"/> &lt; 1、<c>a ≤ (dimension - 1) / 2</c>、または <paramref name="a"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <see cref="double.PositiveInfinity"/> when <paramref name="a"/> is <see cref="double.PositiveInfinity"/> or the result overflows. For <paramref name="dimension"/> = 1, this equals <see cref="LogGamma(double)"/>.</para>
        /// <para><paramref name="a"/> が <see cref="double.PositiveInfinity"/> の場合、または結果がオーバーフローする場合は <see cref="double.PositiveInfinity"/> を返します。<paramref name="dimension"/> = 1 では <see cref="LogGamma(double)"/> と一致します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/35.3.E4"/>
        public static double LogMultivariateGamma(double a, int dimension)
        {
            if (dimension < 1 || !(a > 0.5 * (dimension - 1)))
            {
                return double.NaN;
            }

            if (double.IsPositiveInfinity(a))
            {
                return double.PositiveInfinity;
            }

            double sum = 0.25 * dimension * (dimension - 1.0) * LnPi;
            double compensation = 0;
            for (int j = 0; j < dimension; j++)
            {
                double term = LogGamma(a - 0.5 * j);
                double t = sum + term;
                if (double.IsPositiveInfinity(t))
                {
                    // 大きな項はすべて正なので、和がオーバーフローすれば値も double の範囲を超える（補償の ∞ - ∞ を避けて返す）
                    return double.PositiveInfinity;
                }

                compensation += Math.Abs(sum) >= Math.Abs(term) ? sum - t + term : term - t + sum;
                sum = t;
            }

            return sum + compensation;
        }
    }
}
