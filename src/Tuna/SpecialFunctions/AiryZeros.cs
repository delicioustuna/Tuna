namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // 零点の漸近展開 T(t) = t^(2/3) Σ cₙ t⁻²ⁿ の係数 (DLMF 9.9.E18)。
        // c₀, …, c₅ は有理数、c₆ は高精度で求めた零点への当てはめ値
        private static readonly double[] AiryZeroTCoefficients =
        [
            1.0, 0.10416666666666667, -0.1388888888888889, 0.9298442322530864, -15.509155201673648,
            485.5290969259571, -24505.300371559155,
        ];

        // 導関数の零点の漸近展開 U(t) = t^(2/3) Σ cₙ t⁻²ⁿ の係数 (DLMF 9.9.E19)。
        // c₀, …, c₄ は有理数、c₅・c₆ は AiryZeroTCoefficients と同様に当てはめた値
        private static readonly double[] AiryZeroUCoefficients =
        [
            1.0, -0.14583333333333334, 0.12152777777777778, -0.8739535108024692, 15.016855549125514,
            -476.9464414881744, 24241.309444872288,
        ];

        // 次の項 c₇ t⁻¹⁴ (|c₇| ≈ 1.8×10⁶) は t ≥ 60 (k ≥ 16 相当) で 2.4×10⁻¹⁹ 以下になるため、このしきい値以上では漸近展開を直接使用する
        private const double AiryZeroAsymptoticMin = 60;

        /// <summary>
        /// Computes the <paramref name="k"/>-th zero <c>aₖ</c> of the Airy function <c>Ai(x)</c>.
        /// <para>エアリー関数 <c>Ai(x)</c> の第 <paramref name="k"/> 零点 <c>aₖ</c> を計算します。</para>
        /// </summary>
        /// <param name="k">The 1-based zero index ordered by proximity to the origin (<c>k ≥ 1</c>). / 原点に近い順の 1 始まり零点番号（<c>k ≥ 1</c>）。</param>
        /// <returns>The negative zero <c>aₖ</c> (for example, <c>a₁ ≈ -2.33811</c>), or <see cref="double.NaN"/> if <paramref name="k"/> is less than or equal to 0. / 負の零点 <c>aₖ</c>（例: <c>a₁ ≈ -2.33811</c>）。<paramref name="k"/> が 0 以下の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/9.9"/>
        public static double AiryAiZero(int k) => AiryZero(k, AiryZeroKind.Ai);

        /// <summary>
        /// Computes the <paramref name="k"/>-th zero <c>a'ₖ</c> of the derivative of the Airy function <c>Ai'(x)</c>.
        /// <para>エアリー関数の導関数 <c>Ai'(x)</c> の第 <paramref name="k"/> 零点 <c>a'ₖ</c> を計算します。</para>
        /// </summary>
        /// <param name="k">The 1-based zero index ordered by proximity to the origin (<c>k ≥ 1</c>). / 原点に近い順の 1 始まり零点番号（<c>k ≥ 1</c>）。</param>
        /// <returns>The negative zero <c>a'ₖ</c> (for example, <c>a'₁ ≈ -1.01879</c>), or <see cref="double.NaN"/> if <paramref name="k"/> is less than or equal to 0. / 負の零点 <c>a'ₖ</c>（例: <c>a'₁ ≈ -1.01879</c>）。<paramref name="k"/> が 0 以下の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/9.9"/>
        public static double AiryAiPrimeZero(int k) => AiryZero(k, AiryZeroKind.AiPrime);

        /// <summary>
        /// Computes the <paramref name="k"/>-th negative real zero <c>bₖ</c> of the Airy function of the second kind <c>Bi(x)</c>.
        /// <para>第 2 種エアリー関数 <c>Bi(x)</c> の負の実軸上の第 <paramref name="k"/> 零点 <c>bₖ</c> を計算します。</para>
        /// </summary>
        /// <param name="k">The 1-based zero index ordered by proximity to the origin (<c>k ≥ 1</c>). / 原点に近い順の 1 始まり零点番号（<c>k ≥ 1</c>）。</param>
        /// <returns>The negative zero <c>bₖ</c> (for example, <c>b₁ ≈ -1.17371</c>), or <see cref="double.NaN"/> if <paramref name="k"/> is less than or equal to 0. / 負の零点 <c>bₖ</c>（例: <c>b₁ ≈ -1.17371</c>）。<paramref name="k"/> が 0 以下の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/9.9"/>
        public static double AiryBiZero(int k) => AiryZero(k, AiryZeroKind.Bi);

        /// <summary>
        /// Computes the <paramref name="k"/>-th negative real zero <c>b'ₖ</c> of the derivative of the Airy function of the second kind <c>Bi'(x)</c>.
        /// <para>第 2 種エアリー関数の導関数 <c>Bi'(x)</c> の負の実軸上の第 <paramref name="k"/> 零点 <c>b'ₖ</c> を計算します。</para>
        /// </summary>
        /// <param name="k">The 1-based zero index ordered by proximity to the origin (<c>k ≥ 1</c>). / 原点に近い順の 1 始まり零点番号（<c>k ≥ 1</c>）。</param>
        /// <returns>The negative zero <c>b'ₖ</c> (for example, <c>b'₁ ≈ -2.29444</c>), or <see cref="double.NaN"/> if <paramref name="k"/> is less than or equal to 0. / 負の零点 <c>b'ₖ</c>（例: <c>b'₁ ≈ -2.29444</c>）。<paramref name="k"/> が 0 以下の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/9.9"/>
        public static double AiryBiPrimeZero(int k) => AiryZero(k, AiryZeroKind.BiPrime);

        private enum AiryZeroKind
        {
            Ai,
            AiPrime,
            Bi,
            BiPrime,
        }

        private static double AiryZero(int k, AiryZeroKind kind)
        {
            if (k <= 0)
            {
                return double.NaN;
            }

            bool derivative = kind is AiryZeroKind.AiPrime or AiryZeroKind.BiPrime;

            // aₖ、b'ₖ は 4k - 1、a'ₖ、bₖ は 4k - 3（k は int なので 4k を double で作る）
            bool minusOne = kind is AiryZeroKind.Ai or AiryZeroKind.BiPrime;
            double t = 3 * Math.PI / 8 * (4.0 * k - (minusOne ? 1 : 3));
            double x = -AiryZeroAsymptotic(t, derivative ? AiryZeroUCoefficients : AiryZeroTCoefficients);
            if (t >= AiryZeroAsymptoticMin)
            {
                return x;
            }

            // ハレー法。f = Ai（または Bi）なら f' = Ai'、f'' = x Ai。f = Ai' なら f' = x Ai、f'' = Ai + x Ai'。
            // 初期値の誤差は k = 1 でも 10⁻² 程度なので、3次収束で数回で収束する
            for (int iteration = 0; iteration < 20; iteration++)
            {
                var (ai, aiPrime, bi, biPrime) = Airy(x);
                bool isAi = kind is AiryZeroKind.Ai or AiryZeroKind.AiPrime;
                double w = isAi ? ai : bi;
                double wPrime = isAi ? aiPrime : biPrime;
                double f;
                double f1;
                double f2;
                if (derivative)
                {
                    f = wPrime;
                    f1 = x * w;
                    f2 = w + x * wPrime;
                }
                else
                {
                    f = w;
                    f1 = wPrime;
                    f2 = x * w;
                }

                double newton = f / f1;
                double step = newton / (1 - 0.5 * newton * f2 / f1);
                x -= step;
                if (Math.Abs(step) <= 2 * Epsilon * Math.Abs(x))
                {
                    break;
                }
            }

            return x;
        }

        // 漸近展開 t^(2/3) Σ cₙ t⁻²ⁿ。t が小さい (k = 1, 2) ときは発散的なため、項の絶対値が増え始めたら最小項の手前で打ち切る (ハレー法の初期値として十分)
        private static double AiryZeroAsymptotic(double t, double[] coefficients)
        {
            double inv2 = 1 / (t * t);
            double power = 1;
            double sum = coefficients[0];
            double previous = double.PositiveInfinity;
            for (int n = 1; n < coefficients.Length; n++)
            {
                power *= inv2;
                double term = coefficients[n] * power;
                if (Math.Abs(term) >= previous)
                {
                    break;
                }

                sum += term;
                previous = Math.Abs(term);
            }

            // t^(2/3) = (t^(1/3))²。Math.Pow(t, 2.0/3) は指数 2/3 の丸めが ln t 倍されて効く
            double cbrt = Math.Cbrt(t);
            return cbrt * cbrt * sum;
        }
    }
}
