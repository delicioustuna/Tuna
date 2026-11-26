namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // ζ(s) = 1/(s - 1) + Σ cₙ (s - 1)ⁿ の係数 cₙ = (-1)ⁿ γₙ / n!（γₙ はスティルチェス定数）
        private static readonly double[] ZetaStieltjesCoefficients =
        [
            0.5772156649015329, 0.07281584548367673, -0.00484518159643616, -0.00034230573671722433,
            9.689041939447084e-05, -6.6110318108421895e-06, -3.316240908752772e-07, 1.0462094584479188e-07,
            -8.733218100273798e-09, 9.47827778276236e-11, 5.658421927608708e-11, -6.768689863513697e-12,
            3.4921159366720317e-13, 4.4104247417577536e-15, -2.3997862217709992e-15, 2.1677312200726828e-16,
            -9.544466076366965e-18, -7.387676660538637e-20, 4.800850782488065e-20, -4.139956737713306e-21,
            1.9168201593991233e-22, -2.0441543122262165e-24, -4.818498501107353e-25, 4.8118570515125666e-26,
            -2.560263310318815e-27, 6.927840895304667e-29, 1.628607550485587e-30, -3.1939375611532554e-31,
            2.0991515893634255e-32, -8.33674529544144e-34,
        ];

        // オイラー・マクローリン展開の係数 B₂ⱼ / (2j)!（j = 0, 1, …, 15。静的コンストラクターで初期化）
        private static readonly double[] EulerMaclaurinCoefficients;

        private static double[] CreateEulerMaclaurinCoefficients()
        {
            var coefficients = new double[16];
            for (int j = 0; j < coefficients.Length; j++)
            {
                coefficients[j] = BernoulliEven[j] / FactorialTable[2 * j];
            }

            return coefficients;
        }

        /// <summary>
        /// Computes the Riemann zeta function <c>ζ(s)</c>.
        /// <para>リーマンゼータ関数 <c>ζ(s)</c> を計算します。</para>
        /// </summary>
        /// <param name="s">The real argument. / 実数の引数。</param>
        /// <returns>The value of <c>ζ(s)</c>, or <see cref="double.NaN"/> if <paramref name="s"/> is <see cref="double.NegativeInfinity"/> or <see cref="double.NaN"/>. / <c>ζ(s)</c> の値。<paramref name="s"/> が <see cref="double.NegativeInfinity"/> または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <see cref="double.PositiveInfinity"/> at the pole <c>s = 1</c> and <c>1</c> at <c>s = +∞</c>.</para>
        /// <para>極である <c>s = 1</c> では <see cref="double.PositiveInfinity"/>、<c>s = +∞</c> では <c>1</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Riemann_zeta_function"/>
        public static double Zeta(double s)
        {
            if (double.IsNaN(s))
            {
                return s;
            }

            if (s == 1)
            {
                return double.PositiveInfinity;
            }

            if (double.IsInfinity(s))
            {
                return s > 0 ? 1 : double.NaN;
            }

            if (IsInteger(s) && s >= 0 && s < ZetaAtInteger.Length)
            {
                return ZetaAtInteger[(int)s];
            }

            if (s > -1 && s < 2)
            {
                double t = s - 1;
                double sum = 0;
                for (int n = ZetaStieltjesCoefficients.Length - 1; n >= 0; n--)
                {
                    sum = sum * t + ZetaStieltjesCoefficients[n];
                }

                return 1 / t + sum;
            }

            if (s >= 2)
            {
                return 1 + ZetaMinusOneCore(s);
            }

            // 関数等式 ζ(s) = 2ˢ πˢ⁻¹ sin(πs/2) Γ(1 - s) ζ(1 - s)（s ≤ -1）
            if (IsInteger(s) && !IsOddInteger(s))
            {
                return 0;
            }

            // 1 - s は 2 進指数が繰り上がると丸められ、Γ(1 - s) に ψ(1 - s)·ulp 程度の相対誤差を生むので、
            // 丸めの残り δ を double-double で求めて Γ(x + δ) ≈ Γ(x)(1 + ψ(x)δ) と補正する
            var (oneMinusS, delta) = DoubleDoubleTwoSum(1, -s);
            double zeta = Zeta(oneMinusS);
            double sinTerm = double.SinPi(s / 2);

            // (2π)ˢ を、2π の丸め誤差を補正して計算する
            double twoPiPow = Math.Pow(2 * Math.PI, s) * Math.Exp(s * (TwoPiLo / (2 * Math.PI)));
            if (oneMinusS < 171)
            {
                double gamma = Gamma(oneMinusS);
                if (delta != 0)
                {
                    gamma *= 1 + Digamma(oneMinusS) * delta;
                }

                return twoPiPow / Math.PI * sinTerm * gamma * zeta;
            }

            if (oneMinusS >= 342)
            {
                // |ζ(s)| > 1e400 で必ずオーバーフローする
                return Math.Sign(sinTerm) * double.PositiveInfinity;
            }

            // Γ(x) はオーバーフローするので、倍数公式 Γ(x) = 2ˣ⁻¹ Γ(x/2) Γ((x+1)/2)/√π でオーバーフローしない 2 つのガンマ関数に分け、
            // (2π)ˢ を先に掛けてから 2ˣ⁻¹ の整数部分を最後に 2 の冪として掛ける
            double half = 0.5 * oneMinusS;
            var (onePlus, onePlusDelta) = DoubleDoubleTwoSum(oneMinusS, 1);
            double halfPlus = 0.5 * onePlus;
            // 補正量が 0（1 - s と x + 1 が丸められていない、ほとんどの入力）なら Digamma を呼ばない
            double halfDelta = 0.5 * delta;
            double halfPlusDelta = 0.5 * (delta + onePlusDelta);
            double g1 = halfDelta == 0 ? Gamma(half) : Gamma(half) * (1 + Digamma(half) * halfDelta);
            double g2 = halfPlusDelta == 0 ? Gamma(halfPlus) : Gamma(halfPlus) * (1 + Digamma(halfPlus) * halfPlusDelta);
            double exponent = oneMinusS - 1;
            double whole = Math.Floor(exponent);
            double pow2 = Math.Pow(2, exponent - whole) * (1 + Ln2 * delta);
            double value = twoPiPow * g1 * g2 * (pow2 / (SqrtPi * Math.PI)) * sinTerm * zeta;
            return Math.ScaleB(value, (int)whole);
        }

        /// <summary>
        /// Computes <c>ζ(s) - 1</c> without cancellation for large <paramref name="s"/>.
        /// <para>大きな <paramref name="s"/> での桁落ちを避けて <c>ζ(s) - 1</c> を計算します。</para>
        /// </summary>
        /// <param name="s">The real argument. / 実数の引数。</param>
        /// <returns>The value of <c>ζ(s) - 1</c>, or <see cref="double.NaN"/> if <paramref name="s"/> is <see cref="double.NegativeInfinity"/> or <see cref="double.NaN"/>. / <c>ζ(s) - 1</c> の値。<paramref name="s"/> が <see cref="double.NegativeInfinity"/> または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <see cref="double.PositiveInfinity"/> at <c>s = 1</c> and <c>0</c> at <c>s = +∞</c>.</para>
        /// <para><c>s = 1</c> では <see cref="double.PositiveInfinity"/>、<c>s = +∞</c> では <c>0</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Riemann_zeta_function"/>
        public static double ZetaMinusOne(double s)
        {
            if (double.IsNaN(s))
            {
                return s;
            }

            if (s == 1)
            {
                return double.PositiveInfinity;
            }

            if (double.IsPositiveInfinity(s))
            {
                return 0;
            }

            if (s >= 2)
            {
                return ZetaMinusOneCore(s);
            }

            if (s > 1)
            {
                return HurwitzZetaCore(s, 2);
            }

            return Zeta(s) - 1;
        }

        // ζ(s) - 1 = ζ(s, 2)（s ≥ 2）
        private static double ZetaMinusOneCore(double s)
        {
            if (s > 64)
            {
                return Math.Pow(2, -s) + Math.Pow(3, -s);
            }

            return HurwitzZetaCore(s, 2);
        }

        /// <summary>
        /// Computes the Dirichlet eta function <c>η(s) = (1 - 2¹⁻ˢ) ζ(s)</c>.
        /// <para>ディリクレのイータ関数 <c>η(s) = (1 - 2¹⁻ˢ) ζ(s)</c> を計算します。</para>
        /// </summary>
        /// <param name="s">The real argument. / 実数の引数。</param>
        /// <returns>The value of <c>η(s)</c>, or <see cref="double.NaN"/> if <paramref name="s"/> is <see cref="double.NegativeInfinity"/> or <see cref="double.NaN"/>. / <c>η(s)</c> の値。<paramref name="s"/> が <see cref="double.NegativeInfinity"/> または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <c>ln 2</c> at <c>s = 1</c> and <c>1</c> at <c>s = +∞</c>.</para>
        /// <para><c>s = 1</c> では <c>ln 2</c>、<c>s = +∞</c> では <c>1</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Dirichlet_eta_function"/>
        public static double DirichletEta(double s)
        {
            if (double.IsNaN(s))
            {
                return s;
            }

            if (s == 1)
            {
                return Ln2;
            }

            if (double.IsPositiveInfinity(s))
            {
                return 1;
            }

            if (s > 64)
            {
                return 1 - Math.Pow(2, -s);
            }

            if (s < 0 && IsInteger(s) && !IsOddInteger(s))
            {
                // 負の偶数は ζ の自明零点。係数 1 - 2¹⁻ˢ はオーバーフローしうるので、0 × ∞ を作らずに返す
                return 0;
            }

            if (s < 0)
            {
                // 2¹⁻ˢ ≥ 2 なので 1 - 2¹⁻ˢ に桁落ちはない。(1 - s) ln 2 を exp に渡すと引数の丸めが |1 - s| 倍に拡大するので、
                // 1 - s を double-double で求め、2 の冪の整数部分と小数部分に分けて計算する
                var (oneMinusS, delta) = DoubleDoubleTwoSum(1, -s);
                double zeta = Zeta(s);
                if (double.IsNaN(zeta))
                {
                    return zeta;
                }

                if (oneMinusS >= 1024 || double.IsInfinity(zeta))
                {
                    return zeta == 0 ? 0 : -Math.Sign(zeta) * double.PositiveInfinity;
                }

                double whole = Math.Floor(oneMinusS);
                double pow2 = Math.ScaleB(Math.Pow(2, oneMinusS - whole) * (1 + Ln2 * delta), (int)whole);
                return (1 - pow2) * zeta;
            }

            return -Expm1((1 - s) * Ln2) * Zeta(s);
        }

        /// <summary>
        /// Computes the Hurwitz zeta function <c>ζ(s, a)</c>.
        /// <para>フルヴィッツのゼータ関数 <c>ζ(s, a)</c> を計算します。</para>
        /// </summary>
        /// <param name="s">The exponent (<c>s ≥ 0</c>). / 指数（<c>s ≥ 0</c>）。</param>
        /// <param name="a">The shift parameter (<c>a &gt; 0</c>). / シフトパラメータ（<c>a &gt; 0</c>）。</param>
        /// <returns>The value of <c>ζ(s, a)</c>, or <see cref="double.NaN"/> if <paramref name="s"/> &lt; 0, <paramref name="a"/> ≤ 0, or either argument is <see cref="double.NaN"/>. / <c>ζ(s, a)</c> の値。<paramref name="s"/> &lt; 0、<paramref name="a"/> ≤ 0、またはいずれかの引数が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <see cref="double.PositiveInfinity"/> at the pole <c>s = 1</c> and <c>1/2 - a</c> at <c>s = 0</c>.</para>
        /// <para>極である <c>s = 1</c> では <see cref="double.PositiveInfinity"/>、<c>s = 0</c> では <c>1/2 - a</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Hurwitz_zeta_function"/>
        public static double HurwitzZeta(double s, double a)
        {
            if (double.IsNaN(s) || double.IsNaN(a))
            {
                return double.NaN;
            }

            if (a <= 0 || s < 0)
            {
                return double.NaN;
            }

            if (s == 1)
            {
                return double.PositiveInfinity;
            }

            if (s == 0)
            {
                return 0.5 - a;
            }

            if (double.IsPositiveInfinity(s))
            {
                return a < 1 ? double.PositiveInfinity : a == 1 ? 1 : 0;
            }

            if (double.IsPositiveInfinity(a))
            {
                return s > 1 ? 0 : double.NegativeInfinity;
            }

            if (a == 1)
            {
                return Zeta(s);
            }

            if (a == 2 && s > 1)
            {
                return ZetaMinusOne(s);
            }

            return HurwitzZetaCore(s, a);
        }

        // オイラー・マクローリンの和公式による ζ(s, a)（s > 0、s ≠ 1、a > 0）。
        // scaled が真なら aˢ ζ(s, a) を返し、ζ(s, a) 自体のアンダーフローを避ける
        private static double HurwitzZetaCore(double s, double a, bool scaled = false)
        {
            double Power(double k) => scaled ? Math.Exp(-s * Log1p(k / a)) : Math.Pow(a + k, -s);

            double sum;
            if (s > 30)
            {
                // 項が幾何級数的に減衰するなら直接和をとる
                sum = 0;
                for (int k = 0; k < 64; k++)
                {
                    double term = Power(k);
                    sum += term;
                    if (term <= Epsilon * sum * 0.25)
                    {
                        return sum;
                    }
                }
            }

            int count = (int)Math.Max(0, Math.Ceiling(Math.Max(10, s) - a));
            sum = 0;
            for (int k = count - 1; k >= 0; k--)
            {
                sum += Power(k);
            }

            double w = a + count;
            double wPow = Power(count);
            double tail = w * wPow / (s - 1) + 0.5 * wPow;

            // Σ B₂ⱼ/(2j)! s(s+1)…(s+2j-2) w^(-s-2j+1)
            double factor = s * wPow / w;
            double w2 = w * w;
            double correction = 0;
            for (int j = 1; j < EulerMaclaurinCoefficients.Length; j++)
            {
                double term = EulerMaclaurinCoefficients[j] * factor;
                correction += term;
                if (Math.Abs(term) <= Epsilon * Math.Abs(tail + sum) * 0.125)
                {
                    break;
                }

                factor *= (s + 2 * j - 1) * (s + 2 * j) / w2;
            }

            return sum + (tail + correction);
        }
    }
}
