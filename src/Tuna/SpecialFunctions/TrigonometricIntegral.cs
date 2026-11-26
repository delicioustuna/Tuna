using System.Numerics;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        /// <summary>
        /// Computes the sine integral <c>Si(x)</c>.
        /// <para>正弦積分 <c>Si(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The real argument. / 実数の引数。</param>
        /// <returns>The value of <c>Si(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is <see cref="double.NaN"/>. / <c>Si(x)</c> の値。<paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <c>±π / 2</c> for <c>x = ±∞</c>.</para>
        /// <para><c>x = ±∞</c> では <c>±π / 2</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Trigonometric_integral#Sine_integral"/>
        public static double SinIntegral(double x)
        {
            if (double.IsNaN(x) || x == 0)
            {
                return x;
            }

            if (double.IsInfinity(x))
            {
                return x > 0 ? Math.PI / 2 : -Math.PI / 2;
            }

            SinCosIntegral(Math.Abs(x), out double si, out _);
            return x < 0 ? -si : si;
        }

        /// <summary>
        /// Computes the cosine integral <c>Ci(x)</c>.
        /// <para>余弦積分 <c>Ci(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The non-negative argument (<c>x ≥ 0</c>). / 非負の引数（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>Ci(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> &lt; 0 or <see cref="double.NaN"/>. / <c>Ci(x)</c> の値。<paramref name="x"/> &lt; 0 または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <see cref="double.NegativeInfinity"/> at <c>x = 0</c> and <c>0</c> at <c>x = +∞</c>.</para>
        /// <para><c>x = 0</c> では <see cref="double.NegativeInfinity"/>、<c>x = +∞</c> では <c>0</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Trigonometric_integral#Cosine_integral"/>
        public static double CosIntegral(double x)
        {
            if (double.IsNaN(x) || x < 0)
            {
                return double.NaN;
            }

            if (x == 0)
            {
                return double.NegativeInfinity;
            }

            if (double.IsPositiveInfinity(x))
            {
                return 0;
            }

            SinCosIntegral(x, out _, out double ci);
            return ci;
        }

        /// <summary>
        /// Computes the hyperbolic sine integral <c>Shi(x)</c>.
        /// <para>双曲線正弦積分 <c>Shi(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The real argument. / 実数の引数。</param>
        /// <returns>The value of <c>Shi(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is <see cref="double.NaN"/>. / <c>Shi(x)</c> の値。<paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Trigonometric_integral#Hyperbolic_sine_integral"/>
        public static double SinhIntegral(double x)
        {
            if (double.IsNaN(x) || x == 0 || double.IsInfinity(x))
            {
                return x;
            }

            double ax = Math.Abs(x);
            double value;
            if (ax <= 50)
            {
                // Shi(x) = Σ x^(2k+1) / ((2k+1)(2k+1)!)
                double x2 = ax * ax;
                double term = ax;
                value = ax;
                for (int k = 1; k < 1000; k++)
                {
                    term *= x2 / ((2.0 * k) * (2 * k + 1));
                    double t = term / (2 * k + 1);
                    value += t;
                    if (t <= Epsilon * value * 0.5)
                    {
                        break;
                    }
                }
            }
            else
            {
                value = HalfExpIntegralEiLarge(ax);
            }

            return x < 0 ? -value : value;
        }

        // x > 50 では E₁(x)/Ei(x) ≈ e⁻²ˣ が倍精度で無視できるため Shi(x) = Chi(x) = Ei(x)/2。
        // Ei(x) が Ei(x)/2 より先にオーバーフローするのを避けるため、eˣ を e^(x/2) の 2 回に分けて掛ける
        private static double HalfExpIntegralEiLarge(double x)
        {
            double half = Math.Exp(0.5 * x);
            return half * (half * (0.5 * ExpIntegralEiAsymptoticSeries(x) / x));
        }

        /// <summary>
        /// Computes the hyperbolic cosine integral <c>Chi(x)</c>.
        /// <para>双曲線余弦積分 <c>Chi(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The non-negative argument (<c>x ≥ 0</c>). / 非負の引数（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>Chi(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> &lt; 0 or <see cref="double.NaN"/>. / <c>Chi(x)</c> の値。<paramref name="x"/> &lt; 0 または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <see cref="double.NegativeInfinity"/> at <c>x = 0</c> and <see cref="double.PositiveInfinity"/> at <c>x = +∞</c>.</para>
        /// <para><c>x = 0</c> では <see cref="double.NegativeInfinity"/>、<c>x = +∞</c> では <see cref="double.PositiveInfinity"/> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Trigonometric_integral#Hyperbolic_cosine_integral"/>
        public static double CoshIntegral(double x)
        {
            if (double.IsNaN(x) || x < 0)
            {
                return double.NaN;
            }

            if (x == 0)
            {
                return double.NegativeInfinity;
            }

            if (double.IsPositiveInfinity(x))
            {
                return x;
            }

            if (x <= 50)
            {
                // Chi(x) = γ + ln x + Σ x^(2k) / (2k (2k)!)
                double x2 = x * x;
                double term = 1;
                double sum = 0;
                for (int k = 1; k < 1000; k++)
                {
                    term *= x2 / ((2.0 * k - 1) * (2 * k));
                    double t = term / (2 * k);
                    sum += t;
                    if (t <= Epsilon * sum * 0.5)
                    {
                        break;
                    }
                }

                return (EulerGamma + Math.Log(x)) + sum;
            }

            return HalfExpIntegralEiLarge(x);
        }

        // Si(t) と Ci(t) を同時に計算する（t > 0）
        private static void SinCosIntegral(double t, out double si, out double ci)
        {
            if (t > 2)
            {
                // E₁(it) = -Ci(t) + i(Si(t) - π/2) の連分数（Lentz法）
                var b = new Complex(1, t);
                var c = new Complex(1 / Tiny, 0);
                Complex d = 1 / b;
                Complex h = d;
                for (int i = 2; i < MaxIterations; i++)
                {
                    double a = -(double)(i - 1) * (i - 1);
                    b += 2;
                    d = 1 / (a * d + b);
                    c = b + a / c;
                    Complex delta = c * d;
                    h *= delta;
                    if (Math.Abs(delta.Real - 1) + Math.Abs(delta.Imaginary) <= Epsilon)
                    {
                        break;
                    }
                }

                h *= new Complex(Math.Cos(t), -Math.Sin(t));

                // Ci + i Si = -conj(E₁(it)) + iπ/2
                ci = -h.Real;
                si = h.Imaginary + Math.PI / 2;
                return;
            }

            // Si(t) = Σ_{k 奇数} (-1)^((k-1)/2) tᵏ/(k k!)、Ci(t) = γ + ln t + Σ_{k 偶数} (-1)^(k/2) tᵏ/(k k!)
            si = 0;
            double ciSum = 0;
            double factor = 1;
            for (int k = 1; k < 200; k++)
            {
                factor *= t / k;
                double term = factor / k;
                if (k % 2 == 1)
                {
                    si += ((k - 1) / 2) % 2 == 0 ? term : -term;
                }
                else
                {
                    ciSum += (k / 2) % 2 == 0 ? term : -term;
                }

                if (k >= 2 && term <= Epsilon * si * 0.5)
                {
                    break;
                }
            }

            ci = (EulerGamma + Math.Log(t)) + ciSum;
        }
    }
}
