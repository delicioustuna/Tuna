namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        /// <summary>
        /// Computes the spherical Bessel function of the first kind <c>jₙ(x)</c>.
        /// <para>第 1 種球ベッセル関数 <c>jₙ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The non-negative integer order (<c>n ≥ 0</c>). / 非負の整数次数（<c>n ≥ 0</c>）。</param>
        /// <param name="x">The real argument. / 実数の引数。</param>
        /// <returns>The value of <c>jₙ(x)</c>, or <see cref="double.NaN"/> if <paramref name="n"/> &lt; 0 or <paramref name="x"/> is <see cref="double.NaN"/>. / <c>jₙ(x)</c> の値。<paramref name="n"/> &lt; 0 または <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Bessel_function#Spherical_Bessel_functions"/>
        public static double SphericalBesselJ(int n, double x)
        {
            if (double.IsNaN(x) || n < 0)
            {
                return double.NaN;
            }

            if (x < 0)
            {
                double value = SphericalBesselJ(n, -x);
                return n % 2 == 1 ? -value : value;
            }

            if (x == 0)
            {
                return n == 0 ? 1 : 0;
            }

            if (double.IsPositiveInfinity(x))
            {
                return 0;
            }

            if (n == 0)
            {
                return Math.Sin(x) / x;
            }

            if (x <= BesselSmallArgument)
            {
                return SphericalBesselJSmall(n, x);
            }

            return Math.Sqrt(Math.PI / 2 / x) * BesselJ(n + 0.5, x);
        }

        // 原点近傍（0 < x ≤ BesselSmallArgument）の級数先頭 2 項: jₙ(x) = xⁿ/(2n+1)!! · (1 - x²/(2(2n+3)))。
        // √(π/(2x)) と J_(n+1/2) の積は非正規数の範囲で精度が落ちるため、xⁿ を仮数と 2 進指数に分けて最後に 1 回だけ丸める
        private static double SphericalBesselJSmall(int n, double x)
        {
            int e = Math.ILogB(x);
            if ((long)n * (e + 1) < -1100)
            {
                return 0;
            }

            double m = Math.ScaleB(x, -e);
            double value = Math.Pow(m, n) / DoubleFactorial(2 * n + 1) * (1 - x * x / (2 * (2.0 * n + 3)));
            return Math.ScaleB(value, n * e);
        }

        /// <summary>
        /// Computes the spherical Bessel function of the second kind <c>yₙ(x)</c>.
        /// <para>第 2 種球ベッセル関数 <c>yₙ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The non-negative integer order (<c>n ≥ 0</c>). / 非負の整数次数（<c>n ≥ 0</c>）。</param>
        /// <param name="x">The real argument. / 実数の引数。</param>
        /// <returns>The value of <c>yₙ(x)</c>, or <see cref="double.NaN"/> if <paramref name="n"/> &lt; 0 or <paramref name="x"/> is <see cref="double.NaN"/>. / <c>yₙ(x)</c> の値。<paramref name="n"/> &lt; 0 または <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <see cref="double.NegativeInfinity"/> at <c>x = 0</c> and <c>0</c> as <c>|x| → ∞</c>.</para>
        /// <para><c>x = 0</c> では <see cref="double.NegativeInfinity"/>、<c>|x| → ∞</c> では <c>0</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Bessel_function#Spherical_Bessel_functions"/>
        public static double SphericalBesselY(int n, double x)
        {
            if (double.IsNaN(x) || n < 0)
            {
                return double.NaN;
            }

            if (x < 0)
            {
                // yₙ(-x) = (-1)ⁿ⁺¹ yₙ(x)
                double value = SphericalBesselY(n, -x);
                return n % 2 == 0 ? -value : value;
            }

            if (x == 0)
            {
                return double.NegativeInfinity;
            }

            if (double.IsPositiveInfinity(x))
            {
                return 0;
            }

            if (n == 0)
            {
                return -Math.Cos(x) / x;
            }

            return Math.Sqrt(Math.PI / 2 / x) * BesselY(n + 0.5, x);
        }
    }
}
