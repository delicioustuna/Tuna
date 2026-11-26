namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        /// <summary>
        /// Computes the Wright omega function <c>ω(x)</c>.
        /// <para>ライトのオメガ関数 <c>ω(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The real argument. / 実数の引数。</param>
        /// <returns>The value of <c>ω(x)</c>, satisfying <c>ω + ln ω = x</c> (equivalently <c>W₀(eˣ)</c>). / <c>ω + ln ω = x</c>（<c>W₀(eˣ)</c>）を満たす <c>ω(x)</c> の値。</returns>
        /// <remarks>
        /// <para>Returns <c>0</c> for <c>-∞</c>, <see cref="double.PositiveInfinity"/> for <c>+∞</c>, and <see cref="double.NaN"/> for <see cref="double.NaN"/>.</para>
        /// <para><c>x = -∞</c> では <c>0</c>、<c>x = +∞</c> では <see cref="double.PositiveInfinity"/>、<see cref="double.NaN"/> では <see cref="double.NaN"/> を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/4.13.E6"/>
        public static double WrightOmega(double x)
        {
            if (double.IsNaN(x) || double.IsPositiveInfinity(x))
            {
                return x;
            }

            if (x <= -40)
            {
                return Math.Exp(x);
            }

            if (x <= 1)
            {
                return LambertW0(Math.Exp(x));
            }

            double lnx = Math.Log(x);
            return SolveLogForm(x - lnx + lnx / x, x, negative: false);
        }
    }
}
