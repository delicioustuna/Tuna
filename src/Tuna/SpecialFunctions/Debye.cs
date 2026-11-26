namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        /// <summary>
        /// Computes the Debye function <c>Dₙ(x) = (n / xⁿ) ∫₀ˣ tⁿ / (eᵗ - 1) dt</c>.
        /// <para>デバイ関数 <c>Dₙ(x) = (n / xⁿ) ∫₀ˣ tⁿ / (eᵗ - 1) dt</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The positive integer order (<c>n ≥ 1</c>). / 正の整数次数（<c>n ≥ 1</c>）。</param>
        /// <param name="x">The real input value. / 実数の入力値。</param>
        /// <returns>The value of <c>Dₙ(x)</c> (<c>1</c> at <c>0</c>, <c>0</c> at <c>+∞</c>, <see cref="double.PositiveInfinity"/> at <c>-∞</c>), or <see cref="double.NaN"/> if <paramref name="n"/> is less than 1 or <paramref name="x"/> is <see cref="double.NaN"/>. / <c>Dₙ(x)</c> の値（<c>0</c> で <c>1</c>、<c>+∞</c> で <c>0</c>、<c>-∞</c> で <see cref="double.PositiveInfinity"/>）。<paramref name="n"/> が 1 未満または <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Debye_function"/>
        public static double Debye(int n, double x)
        {
            if (n < 1 || double.IsNaN(x))
            {
                return double.NaN;
            }

            if (x == 0)
            {
                return 1;
            }

            if (double.IsPositiveInfinity(x))
            {
                return 0;
            }

            if (double.IsNegativeInfinity(x))
            {
                return double.PositiveInfinity;
            }

            if (x < 0)
            {
                // 被積分関数の差が -tⁿ なので Dₙ(-x) = Dₙ(x) + nx/(n+1) (両項とも正で桁落ちなし)
                double absX = -x;
                return DebyeCore(n, absX) + n * absX / (n + 1.0);
            }

            return DebyeCore(n, x);
        }

        // Dₙ(x) (0 < x < ∞)
        private static double DebyeCore(int n, double x)
        {
            if (x <= 2)
            {
                return DebyeSeries(n, x);
            }

            if (x >= n + 1.0)
            {
                return DebyeLarge(n, x);
            }

            return DebyeExponentialSum(n, x);
        }

        // Dₙ(x) = 1 - nx/(2(n+1)) + 2n Σₖ (-1)ᵏ⁺¹ ζ(2k) r^k/(2k+n) (r = (x/2π)² ≤ 0.102; 20 項で r²⁰ < 10⁻¹⁹)
        private static double DebyeSeries(int n, double x)
        {
            double r = x * InvTwoPi;
            r *= r;
            double power = 1;
            double sum = 0;
            for (int k = 1; 2 * k < ZetaAtInteger.Length; k++)
            {
                power *= r;
                double term = ZetaAtInteger[2 * k] * power / (2 * k + n);
                sum += (k & 1) == 1 ? term : -term;
                if (term <= 0.0625 * Epsilon * Math.Abs(sum))
                {
                    break;
                }
            }

            return 1 - n * x / (2 * (n + 1.0)) + 2 * n * sum;
        }

        // Dₙ(x) = n (n!/xⁿ) [P(n+1, x) + (ζ(n+1) - 1) - Σ_(m≥2) Q(n+1, mx)/mⁿ⁺¹] (x ≥ n + 1; n!/xⁿ は仮数と指数に分離)
        private static double DebyeLarge(int n, double x)
        {
            double a = n + 1.0;
            double bracket = RegularizedGammaP(a, x) + ZetaMinusOne(a);
            for (int m = 2; m < 100000; m++)
            {
                double q = RegularizedGammaQ(a, m * x);
                if (q == 0)
                {
                    break;
                }

                double term = q * Math.Pow(m, -a);
                bracket -= term;
                if (term <= 0.0625 * Epsilon * bracket)
                {
                    break;
                }
            }

            double factorial = ScaledFactorial(n, out long factorialExponent);
            double power = ScaledPower(x, n, out long powerExponent);
            return ScaleB(n * (factorial / power) * bracket, factorialExponent - powerExponent);
        }

        // 2 < x < n + 1 での指数和展開: m₁ = ⌊(n+1)/x⌋ + 1 として m < m₁ は Jₙ(mx) を直接加算し、m ≥ m₁ はフルヴィッツゼータ関数と Q でまとめる
        private static double DebyeExponentialSum(int n, double x)
        {
            double a = n + 1.0;
            double m1 = Math.Floor(a / x) + 1;
            double decay = 1 / -Expm1(-0.5 * x);
            double sum = 0;
            for (int m = 1; m < m1; m++)
            {
                double j = DebyeJ(n, m * x);
                sum += j;

                // 残りの m < m₁ の和は j·decay 以下、m ≥ m₁ の和は j(1 + m₁/n) 以下
                if (j * (decay + 1 + m1 / n) <= 0.0625 * Epsilon * sum)
                {
                    return n * x * sum;
                }
            }

            // m ≥ m₁: Jₙ(mx) = n!(1 - Q(n+1, mx))/(mx)ⁿ⁺¹。ゼータ部分はフルヴィッツゼータ m₁ⁿ⁺¹ ζ(n+1, m₁)
            double hurwitz = HurwitzZetaCore(a, m1, scaled: true);
            double block = hurwitz;
            for (double m = m1; m < m1 + 100000; m++)
            {
                double q = RegularizedGammaQ(a, m * x);
                if (q == 0)
                {
                    break;
                }

                double term = q * Math.Exp(-a * Math.Log(m / m1));
                block -= term;
                if (term <= 0.0625 * Epsilon * block)
                {
                    break;
                }
            }

            // n!/(m₁x)ⁿ⁺¹ = n!/(m₁ⁿ⁺¹ xⁿ⁺¹) (m₁ は整数なので積 m₁x の丸めを避ける)
            double factorial = ScaledFactorial(n, out long factorialExponent);
            double powerM = ScaledPower(m1, n + 1, out long exponentM);
            double powerX = ScaledPower(x, n + 1, out long exponentX);
            double prefactor = ScaleB(factorial / (powerM * powerX), factorialExponent - exponentM - exponentX);
            return n * x * (sum + prefactor * block);
        }

        // Jₙ(z) = ∫₀¹ uⁿ e⁻ᶻᵘ du = e⁻ᶻ Σₖ zᵏ/((n+1)(n+2)…(n+1+k)) (0 < z ≤ n + 1)
        private static double DebyeJ(int n, double z)
        {
            double term = 1.0 / (n + 1);
            double sum = term;
            for (int k = 1; k < 1000000; k++)
            {
                term *= z / (n + 1 + k);
                sum += term;
                if (term <= 0.0625 * Epsilon * sum)
                {
                    break;
                }
            }

            return Math.Exp(-z) * sum;
        }
    }
}
