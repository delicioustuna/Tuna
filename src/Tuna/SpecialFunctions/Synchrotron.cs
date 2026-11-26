namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // 正しく丸めた定数（1/Γ(1/3), 1/Γ(4/3), 1/Γ(5/3), π/√3）
        private const double SynchrotronReciprocalGammaOneThird = 0.3732821739073952;
        private const double SynchrotronReciprocalGammaFourThirds = 1.1198465217221858;
        private const double SynchrotronReciprocalGammaFiveThirds = 1.1077321674324725;
        private const double SynchrotronPiOverSqrt3 = 1.8137993642342178;

        // これ以下の x では冪級数、それ以上では台形則を使う
        private const double SynchrotronFSeriesLimit = 0.25;
        private const double SynchrotronGSeriesLimit = 0.5;

        /// <summary>
        /// Computes the first synchrotron radiation function <c>F(x) = x ∫ₓ^∞ K_(5/3)(t) dt</c>.
        /// <para>第 1 シンクロトロン関数 <c>F(x) = x ∫ₓ^∞ K_(5/3)(t) dt</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The non-negative argument (<c>x ≥ 0</c>). / 非負の引数（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>F(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> &lt; 0 or <see cref="double.NaN"/>. / <c>F(x)</c> の値。<paramref name="x"/> &lt; 0 または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <c>0</c> at <c>x = 0</c> and <c>x = +∞</c>.</para>
        /// <para><c>x = 0</c> および <c>x = +∞</c> では <c>0</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Synchrotron_radiation"/>
        public static double SynchrotronF(double x)
        {
            if (double.IsNaN(x) || x < 0)
            {
                return double.NaN;
            }

            if (x == 0 || double.IsPositiveInfinity(x))
            {
                return 0;
            }

            if (x <= SynchrotronFSeriesLimit)
            {
                double w = 0.5 * x;
                double w2 = w * w;
                double c = Math.Cbrt(w);

                // 3 つの級数はいずれも正項で、項の比は w²/(k(k + O(1))) ≤ 1/64 程度
                double p1 = 1;
                double t1 = 1;
                double p2 = 0;
                double t2 = 1;
                double p3 = 0;
                double t3 = 1;
                for (int k = 0; k < 40; k++)
                {
                    if (k > 0)
                    {
                        t1 *= w2 / (k * (k - 2.0 / 3));
                        p1 += t1;
                        t2 *= w2 / (k * (k + 2.0 / 3));
                        t3 *= w2 / (k * (k + 1.0 / 3));
                    }

                    double a2 = k * t2 / (2 * k + 2.0 / 3);
                    double a3 = t3 / (2 * k + 4.0 / 3);
                    p2 += a2;
                    p3 += a3;
                    if (k > 0 && t1 <= 1e-17 * p1)
                    {
                        break;
                    }
                }

                double value = 4 * c * SynchrotronReciprocalGammaOneThird * p1 - 2 * w
                    - 4 * w * c * c * SynchrotronReciprocalGammaFiveThirds * p2
                    - 4 * w2 * c * SynchrotronReciprocalGammaFourThirds * p3;
                return SynchrotronPiOverSqrt3 * value;
            }

            // ∫ₓ^∞ K_(5/3) = e^(-x) × （台形則の和）
            double halfDecay = Math.Exp(-0.5 * x);
            return x * BesselIntegralKTailScaled(5.0 / 3, x) * halfDecay * halfDecay;
        }

        /// <summary>
        /// Computes the second synchrotron radiation function <c>G(x) = x K_(2/3)(x)</c>.
        /// <para>第 2 シンクロトロン関数 <c>G(x) = x K_(2/3)(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The non-negative argument (<c>x ≥ 0</c>). / 非負の引数（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>G(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> &lt; 0 or <see cref="double.NaN"/>. / <c>G(x)</c> の値。<paramref name="x"/> &lt; 0 または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <c>0</c> at <c>x = 0</c> and <c>x = +∞</c>.</para>
        /// <para><c>x = 0</c> および <c>x = +∞</c> では <c>0</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Synchrotron_radiation"/>
        public static double SynchrotronG(double x)
        {
            if (double.IsNaN(x) || x < 0)
            {
                return double.NaN;
            }

            if (x == 0 || double.IsPositiveInfinity(x))
            {
                return 0;
            }

            if (x <= SynchrotronGSeriesLimit)
            {
                double w = 0.5 * x;
                double w2 = w * w;
                double c = Math.Cbrt(w);
                double p1 = 1;
                double t1 = 1;
                double q = 1;
                double t2 = 1;
                for (int k = 1; k < 40; k++)
                {
                    t1 *= w2 / (k * (k - 2.0 / 3));
                    t2 *= w2 / (k * (k + 2.0 / 3));
                    p1 += t1;
                    q += t2;
                    if (t1 <= 1e-17 * p1)
                    {
                        break;
                    }
                }

                double value = c * SynchrotronReciprocalGammaOneThird * p1 - w * c * c * SynchrotronReciprocalGammaFiveThirds * q;
                return 2 * SynchrotronPiOverSqrt3 * value;
            }

            double h = Math.Min(0.15, 0.6 / Math.Sqrt(x));
            double sum = 0.5;
            for (int k = 1; k < 100000; k++)
            {
                double u = k * h;
                double sinhHalf = Math.Sinh(0.5 * u);
                double exponent = 2 * x * sinhHalf * sinhHalf;
                sum += Math.Exp(-exponent) * Math.Cosh(2.0 / 3 * u);
                if (exponent - 2.0 / 3 * u > 42)
                {
                    break;
                }
            }

            double halfDecay = Math.Exp(-0.5 * x);
            return x * (h * sum) * halfDecay * halfDecay;
        }
    }
}
