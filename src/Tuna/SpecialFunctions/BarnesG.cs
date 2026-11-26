namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // ζ'(-1) = 1/12 - ln A (A はグレイシャー・キンケリン定数)
        private const double ZetaDerivativeAtMinusOne = -0.16542114370045092;

        // |z| ≤ 40 では漸化式 G(z+1) = Γ(z) G(z) で [1/2, 3/2] へ寄せ、それを超えると漸近展開と反転公式を用いる
        private const double BarnesGRecurrenceLimit = 40;

        /// <summary>
        /// Computes the Barnes G-function <c>G(z)</c>, satisfying <c>G(1) = 1</c> and <c>G(z + 1) = Γ(z) G(z)</c>.
        /// <para>バーンズの G 関数 <c>G(z)</c>（<c>G(1) = 1</c>、<c>G(z + 1) = Γ(z) G(z)</c>）を計算します。</para>
        /// </summary>
        /// <param name="z">The real input value. / 実数の入力値。</param>
        /// <returns>The value of <c>G(z)</c>, <c>0</c> at non-positive integers, <see cref="double.PositiveInfinity"/> for <c>z ≳ 28.5</c>, or <see cref="double.NaN"/> for <see cref="double.NaN"/> and <see cref="double.NegativeInfinity"/>. / <c>G(z)</c> の値。0 以下の整数では <c>0</c>、<c>z ≳ 28.5</c> では <see cref="double.PositiveInfinity"/>、<see cref="double.NaN"/> や <see cref="double.NegativeInfinity"/> では <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/5.17"/>
        public static double BarnesG(double z)
        {
            if (double.IsNaN(z) || double.IsNegativeInfinity(z))
            {
                return double.NaN;
            }

            if (z <= 0 && IsInteger(z))
            {
                return 0;
            }

            if (z > BarnesGRecurrenceLimit)
            {
                return double.PositiveInfinity;
            }

            if (z < -BarnesGRecurrenceLimit)
            {
                return BarnesGSign(z) * Math.Exp(LogBarnesG(z));
            }

            double mantissa = ScaledBarnesG(z, out long exponent);
            return ScaleB(mantissa, exponent);
        }

        /// <summary>
        /// Computes the natural logarithm of the absolute value of the Barnes G-function, <c>ln |G(z)|</c>.
        /// <para>バーンズの G 関数の絶対値の自然対数 <c>ln |G(z)|</c> を計算します。</para>
        /// </summary>
        /// <param name="z">The real input value. / 実数の入力値。</param>
        /// <returns>The value of <c>ln |G(z)|</c>, <see cref="double.NegativeInfinity"/> at non-positive integers, or <see cref="double.NaN"/> for <see cref="double.NaN"/> and <see cref="double.NegativeInfinity"/>. / <c>ln |G(z)|</c> の値。0 以下の整数では <see cref="double.NegativeInfinity"/>、<see cref="double.NaN"/> や <see cref="double.NegativeInfinity"/> では <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/5.17"/>
        public static double LogBarnesG(double z)
        {
            if (double.IsNaN(z) || double.IsNegativeInfinity(z))
            {
                return double.NaN;
            }

            if (double.IsPositiveInfinity(z))
            {
                return z;
            }

            if (z <= 0 && IsInteger(z))
            {
                return double.NegativeInfinity;
            }

            if (z > BarnesGRecurrenceLimit)
            {
                return LogBarnesGAsymptotic(z - 1);
            }

            if (z < -BarnesGRecurrenceLimit)
            {
                // u = 1 - z。sin πu = sin πz、Cl₂(2πu) = -Cl₂(2πz) なので丸めた u は漸近展開と u ln 2 の部分にしか使わない
                double u = 1 - z;
                return LogBarnesGAsymptotic(u) - u * LnTwoPi + u * Math.Log(Math.Abs(2 * double.SinPi(z))) - ClausenTurns(z) / (2 * Math.PI);
            }

            double n = Math.Round(z - 1);
            double t = z - 1 - n;
            double sum = LogBarnesGTaylor(t);
            if (n > 0)
            {
                for (double j = 1; j <= n; j++)
                {
                    sum += LogGamma(t + j);
                }
            }
            else
            {
                for (double j = 0; j < -n; j++)
                {
                    sum -= Math.Log(Math.Abs(Gamma(z + j)));
                }
            }

            return sum;
        }

        /// <summary>
        /// Computes the hyperfactorial <c>H(z) = Γ(z + 1)^z / G(z + 1)</c>.
        /// <para>ハイパー階乗（超階乗）<c>H(z) = Γ(z + 1)^z / G(z + 1)</c> を計算します。</para>
        /// </summary>
        /// <param name="z">The real input value (<c>z ≥ -1</c>). / 実数の入力値（<c>z ≥ -1</c>）。</param>
        /// <returns>The value of <c>H(z)</c> (with <c>H(-1) = H(0) = 1</c>), <see cref="double.PositiveInfinity"/> for <c>z ≳ 23</c>, or <see cref="double.NaN"/> if <paramref name="z"/> is less than -1 or <see cref="double.NaN"/>. / <c>H(z)</c> の値（<c>H(-1) = H(0) = 1</c>）。<c>z ≳ 23</c> では <see cref="double.PositiveInfinity"/>、<paramref name="z"/> が -1 未満または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>For non-negative integers, <c>H(n) = 1¹ · 2² · … · nⁿ</c>.</para>
        /// <para>非負整数では <c>H(n) = 1¹ · 2² · … · nⁿ</c> に一致します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Hyperfactorial"/>
        public static double Hyperfactorial(double z)
        {
            if (double.IsNaN(z) || z < -1)
            {
                return double.NaN;
            }

            if (z == -1 || z == 0)
            {
                return 1;
            }

            if (z > BarnesGRecurrenceLimit)
            {
                return double.PositiveInfinity;
            }

            double a = z + 1;
            double gamma = Gamma(a);
            int e = Math.ILogB(gamma);
            double m = Math.ScaleB(gamma, -e);

            // 2^(ez) = 2^(⌊p⌋) 2^(p - ⌊p⌋) (1 + r ln 2)、p + r = ez
            double p = e * z;
            double r = Math.FusedMultiplyAdd(e, z, -p);
            double whole = Math.Floor(p);
            double power = Math.Pow(m, z) * Math.Pow(2, p - whole) * (1 + r * Ln2);

            double g = ScaledBarnesG(a, out long gExponent);
            return ScaleB(power / g, (long)whole - gExponent);
        }

        /// <summary>
        /// Computes the superfactorial <c>sf(z) = G(z + 2)</c>.
        /// <para>スーパー階乗（超階乗）<c>sf(z) = G(z + 2)</c> を計算します。</para>
        /// </summary>
        /// <param name="z">The real input value. / 実数の入力値。</param>
        /// <returns>The value of <c>G(z + 2)</c>, <c>0</c> at integers less than or equal to -2, or <see cref="double.PositiveInfinity"/> for <c>z ≳ 26.5</c>. / <c>G(z + 2)</c> の値。-2 以下の整数では <c>0</c>、<c>z ≳ 26.5</c> では <see cref="double.PositiveInfinity"/>。</returns>
        /// <remarks>
        /// <para>For non-negative integers, <c>sf(n) = 1! · 2! · … · n!</c>.</para>
        /// <para>非負整数では <c>sf(n) = 1! · 2! · … · n!</c> に一致します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/5.17"/>
        public static double Superfactorial(double z) => BarnesG(z + 2);

        // G(z) = mantissa · 2^exponent (-40 ≤ z ≤ 40、0 以下の整数を除く)
        private static double ScaledBarnesG(double z, out long exponent)
        {
            double n = Math.Round(z - 1);
            double t = z - 1 - n;
            double mantissa = Math.Exp(LogBarnesGTaylor(t));
            exponent = 0;
            if (n > 0)
            {
                // G(1 + t + n) = G(1+t) Π_(j=1)^n Γ(t + j)
                for (double j = 1; j <= n; j++)
                {
                    mantissa *= Gamma(t + j);
                    int shift = Math.ILogB(mantissa);
                    mantissa = Math.ScaleB(mantissa, -shift);
                    exponent += shift;
                }
            }
            else
            {
                // G(z) = G(1+t)/Π_(j=0)^(-n-1) Γ(z + j)
                for (double j = 0; j < -n; j++)
                {
                    mantissa /= Gamma(z + j);
                    int shift = Math.ILogB(mantissa);
                    mantissa = Math.ScaleB(mantissa, -shift);
                    exponent += shift;
                }
            }

            return mantissa;
        }

        // 非整数 z < 0 に対する G(z) の符号 (-1)^((m+1)(m+2)/2) (z ∈ (-m-1, -m))
        private static double BarnesGSign(double z)
        {
            double m = Math.Floor(-z);
            double e = Math.IEEERemainder((m + 1) * (m + 2) / 2, 2);
            return e == 0 ? 1 : -1;
        }

        // ln G(1 + t) (|t| ≤ 1/2) のテイラー級数
        private static double LogBarnesGTaylor(double t)
        {
            if (t == 0)
            {
                return 0;
            }

            // Σₖ₌₂ (-1)ᵏ ζ(k) t^(k+1)/(k+1)。|t| ≤ 1/2 で 2⁻⁵⁶ 程度まで加算する
            double sum = 0;
            for (int k = 60; k >= 2; k--)
            {
                double zeta = k < ZetaAtInteger.Length ? ZetaAtInteger[k] : 1 + Math.ScaleB(1.0, -k) + Math.Pow(3, -k);
                sum = sum * -t + zeta / (k + 1);
            }

            // k = 2 の項の符号は正
            sum *= t * t * t;
            return 0.5 * t * (LnTwoPi - 1) - 0.5 * (1 + EulerGamma) * t * t + sum;
        }

        // ln G(1 + u) の漸近展開 (u > 39)
        private static double LogBarnesGAsymptotic(double u)
        {
            double lnu = Math.Log(u);
            double inverse = 1 / (u * u);
            double power = inverse;
            double sum = 0;
            for (int k = 1; k < 12; k++)
            {
                double term = BernoulliEven[k + 1] / (4.0 * k * (k + 1)) * power;
                sum += term;
                if (Math.Abs(term) <= 0.25 * Epsilon * Math.Abs(sum))
                {
                    break;
                }

                power *= inverse;
            }

            return 0.5 * u * u * (lnu - 1.5) + 0.5 * u * LnTwoPi - lnu / 12 + ZetaDerivativeAtMinusOne + sum;
        }
    }
}
