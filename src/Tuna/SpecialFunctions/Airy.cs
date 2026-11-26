namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // Ai(0) = 1 / (3^(2/3) Γ(2/3)), -Ai'(0) = 1 / (3^(1/3) Γ(1/3))
        private const double AiryAiAtZero = 0.3550280538878172;
        private const double MinusAiryAiPrimeAtZero = 0.2588194037928068;
        private const double OneOverSqrt3 = 0.5773502691896258;
        private const double Sqrt3 = 1.7320508075688772;

        /// <summary>
        /// Computes the Airy function of the first kind, <c>Ai(x)</c>.
        /// <para>第 1 種エアリー関数 <c>Ai(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>Ai(x)</c>. / <c>Ai(x)</c> の値。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Airy_function#Definitions"/>
        public static double AiryAi(double x) => AiryCore(x, derivatives: false).Ai;

        /// <summary>
        /// Computes the derivative of the Airy function of the first kind, <c>Ai'(x)</c>.
        /// <para>第 1 種エアリー関数の導関数 <c>Ai'(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>Ai'(x)</c>. / <c>Ai'(x)</c> の値。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Airy_function#Definitions"/>
        public static double AiryAiPrime(double x) => Airy(x).AiPrime;

        /// <summary>
        /// Computes the Airy function of the second kind, <c>Bi(x)</c>.
        /// <para>第 2 種エアリー関数 <c>Bi(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>Bi(x)</c>. / <c>Bi(x)</c> の値。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Airy_function#Definitions"/>
        public static double AiryBi(double x) => AiryCore(x, derivatives: false).Bi;

        /// <summary>
        /// Computes the derivative of the Airy function of the second kind, <c>Bi'(x)</c>.
        /// <para>第 2 種エアリー関数の導関数 <c>Bi'(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>Bi'(x)</c>. / <c>Bi'(x)</c> の値。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Airy_function#Definitions"/>
        public static double AiryBiPrime(double x) => Airy(x).BiPrime;

        /// <summary>
        /// Simultaneously computes the Airy functions <c>Ai(x)</c> and <c>Bi(x)</c> together with their derivatives <c>Ai'(x)</c> and <c>Bi'(x)</c>.
        /// <para>エアリー関数 <c>Ai(x)</c>、<c>Bi(x)</c> とその導関数 <c>Ai'(x)</c>、<c>Bi'(x)</c> を同時に計算します。</para>
        /// </summary>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>A tuple containing <c>(Ai(x), Ai'(x), Bi(x), Bi'(x))</c>. / <c>(Ai(x), Ai'(x), Bi(x), Bi'(x))</c> のタプル。</returns>
        /// <remarks>
        /// <para><see cref="AiryAi"/> and <see cref="AiryBi"/> skip computing the derivatives. Use this method when both function values and derivatives are needed.</para>
        /// <para><see cref="AiryAi"/> および <see cref="AiryBi"/> は導関数を計算しません。関数値と導関数の両方が必要な場合はこのメソッドを使用してください。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Airy_function#Definitions"/>
        public static (double Ai, double AiPrime, double Bi, double BiPrime) Airy(double x) => AiryCore(x, derivatives: true);

        // derivatives が false の場合は導関数用の次数 2/3 のベッセル関数を省略する
        private static (double Ai, double AiPrime, double Bi, double BiPrime) AiryCore(double x, bool derivatives)
        {
            if (double.IsNaN(x))
            {
                return (x, x, x, x);
            }

            if (double.IsPositiveInfinity(x))
            {
                return (0, 0, double.PositiveInfinity, double.PositiveInfinity);
            }

            if (double.IsNegativeInfinity(x))
            {
                return (0, double.NaN, 0, double.NaN);
            }

            if (Math.Abs(x) <= 1)
            {
                return AiryMaclaurin(x);
            }

            double absx = Math.Abs(x);
            double root = Math.Sqrt(absx);
            if (x > 0)
            {
                double zeta = 2.0 / 3 * absx * root;
                // Ai(x) = (1/π)√(x/3) K_{1/3}(ζ)、Bi(x) = √x (2/√3 I_{1/3}(ζ) + K_{1/3}(ζ)/π)
                BesselIKCore(1.0 / 3, zeta, out double i13, out double k13);
                double i23 = 0;
                double k23 = 0;
                if (derivatives)
                {
                    BesselIKCore(2.0 / 3, zeta, out i23, out k23);
                }

                double decay = Math.Exp(-zeta);
                double growth = Math.Exp(zeta);
                double ai = root * OneOverSqrt3 * k13 * decay / Math.PI;
                double aiPrime = -x * OneOverSqrt3 * k23 * decay / Math.PI;
                if (double.IsPositiveInfinity(growth))
                {
                    // e^ζ がオーバーフローしても Bi は有限のことがある (x ≈ 104.4 まで) ため、指数を半分ずつ掛けて範囲を最後の積で判定する。
                    // ζ 自体がオーバーフローするとスケーリング済み I が 0 になり 0·∞ で NaN になるため、直接 +∞ を返す
                    double halfGrowth = Math.Exp(0.5 * zeta);
                    if (double.IsPositiveInfinity(halfGrowth))
                    {
                        return (ai, aiPrime, double.PositiveInfinity, double.PositiveInfinity);
                    }

                    double largeBi = root * (2 * OneOverSqrt3 * i13 * halfGrowth) * halfGrowth;
                    double largeBiPrime = x * (2 * OneOverSqrt3 * i23 * halfGrowth) * halfGrowth;
                    return (ai, aiPrime, largeBi, largeBiPrime);
                }

                double bi = root * (2 * OneOverSqrt3 * i13 * growth + k13 * decay / Math.PI);
                double biPrime = x * (2 * OneOverSqrt3 * i23 * growth + k23 * decay / Math.PI);
                return (ai, aiPrime, bi, biPrime);
            }
            else
            {
                // Ai(-z) = (√z/2)(J_{1/3}(ζ) - Y_{1/3}(ζ)/√3), Bi(-z) = -(√z/2)(J_{1/3}(ζ)/√3 + Y_{1/3}(ζ))。
                // ζ の丸め誤差は位相誤差になるため、倍々精度 ζ_hi + ζ_lo で求め、ζ_hi で計算後に J + iY = M e^(iθ) の位相を ζ_lo だけ回転する。
                // 2(z √z) がオーバーフローする範囲 (|x| ≳ 2.0e205) は ζ を倍々精度で表せないため NaN を返す
                if (absx * root > double.MaxValue / 2)
                {
                    return (double.NaN, double.NaN, double.NaN, double.NaN);
                }

                double zetaHi = AiryOscillatoryZeta(absx, root, out double zetaLo);
                BesselJYCore(1.0 / 3, zetaHi, out double j13, out double y13);
                double j23 = 0;
                double y23 = 0;
                if (derivatives)
                {
                    BesselJYCore(2.0 / 3, zetaHi, out j23, out y23);
                }

                (double sin, double cos) = Math.SinCos(zetaLo);
                (j13, y13) = (j13 * cos - y13 * sin, j13 * sin + y13 * cos);
                (j23, y23) = (j23 * cos - y23 * sin, j23 * sin + y23 * cos);
                double ai = 0.5 * root * (j13 - OneOverSqrt3 * y13);
                double bi = -0.5 * root * (OneOverSqrt3 * j13 + y13);
                double aiPrime = 0.5 * absx * (OneOverSqrt3 * y23 + j23);
                double biPrime = 0.5 * absx * (OneOverSqrt3 * j23 - y23);
                return (ai, aiPrime, bi, biPrime);
            }
        }

        // ζ = (2/3) z^(3/2) (z > 0, root ≈ √z) を倍々精度 ζ_hi + ζ_lo で計算し、ζ_hi を返す
        private static double AiryOscillatoryZeta(double z, double root, out double zetaLo)
        {
            // √z = root + rootLo, z √z = product + productLo, ζ = 2(z √z)/3 (2(z √z) がオーバーフローしない範囲で呼ぶ)
            double product = z * root;
            double rootLo = -Math.FusedMultiplyAdd(root, root, -z) / (2 * root);
            double productLo = Math.FusedMultiplyAdd(z, root, -product) + z * rootLo;
            double zetaHi = 2 * product / 3;
            zetaLo = (Math.FusedMultiplyAdd(-zetaHi, 3, 2 * product) + 2 * productLo) / 3;
            return zetaHi;
        }

        // マクローリン展開によるエアリー関数 (|x| ≤ 1)
        private static (double Ai, double AiPrime, double Bi, double BiPrime) AiryMaclaurin(double x)
        {
            // f(x) = Σ 3ᵏ (1/3)ₖ x³ᵏ/(3k)!、g(x) = Σ 3ᵏ (2/3)ₖ x³ᵏ⁺¹/(3k+1)!
            // Ai = c₁f - c₂g、Bi = √3(c₁f + c₂g)
            double x2 = x * x;
            double x3 = x2 * x;
            double fTerm = 1;
            double f = 1;
            double fPrime = 0;
            double gTerm = x;
            double g = x;
            double gPrime = 1;
            for (int k = 1; k < 100; k++)
            {
                // fₖ' = fₖ₋₁ x²/(3k-1)、gₖ' = gₖ₋₁ x²/(3k)
                fPrime += fTerm * x2 / (3 * k - 1);
                gPrime += gTerm * x2 / (3 * k);
                fTerm *= x3 / ((3.0 * k - 1) * (3 * k));
                gTerm *= x3 / ((3.0 * k) * (3 * k + 1));
                f += fTerm;
                g += gTerm;
                if (Math.Abs(fTerm) + Math.Abs(gTerm) <= Epsilon * 0.25 * (Math.Abs(f) + Math.Abs(g)))
                {
                    break;
                }
            }

            double c1 = AiryAiAtZero;
            double c2 = MinusAiryAiPrimeAtZero;
            return (
                c1 * f - c2 * g,
                c1 * fPrime - c2 * gPrime,
                Sqrt3 * (c1 * f + c2 * g),
                Sqrt3 * (c1 * fPrime + c2 * gPrime));
        }
    }
}
