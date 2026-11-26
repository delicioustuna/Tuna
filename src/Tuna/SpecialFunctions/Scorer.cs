namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // これ以上の |x| では Gi(x)（x > 0）と Hi(x)（x < 0）を漸近展開で求める（|x| = 20 では第 10 項で 10⁻¹⁷ を下回る）
        private const double ScorerAsymptoticMin = 20;

        // これ以上の z では Ai(-z), Bi(-z) を位相 ζ = (2/3)z^(3/2) の double-double 漸近展開で求める（z = 12 で最小項 ≈ 10⁻²⁴）
        private const double ScorerAiryNegativeAsymptoticMin = 12;

        // 積分打ち切りの指数部 s³/3 + cs の上限（e⁻⁴¹ ≈ 1.6×10⁻¹⁸）
        private const double ScorerTruncationExponent = 41;

        // ScorerAiryNegativeAsymptotic で位相 ζ を扱える上限（2⁷⁷ ≈ 1.5×10²³、z ≈ 3.2×10¹⁵。超過時は NaN）
        private const double ScorerAiryPhaseMaxZeta = 1.5111572745182865e23;

        // これより大きな ζ = (2/3)x^(3/2) では Bi(x) が確実にオーバーフローする（実境界は x ≈ 104.42、ζ ≈ 711.4）
        private const double ScorerBiOverflowZeta = 720;

        // ln 2 の倍精度表現との残差（ln 2 - Ln2）
        private const double ScorerLn2Lo = 2.3190468138462996e-17;

        // 40 点ガウス・ルジャンドル則（正の節点 20 個）。[0, S] で指数部が 0 から 41 まで変わる整関数の積分に使用する
        private static readonly (double[] Nodes, double[] Weights) ScorerRule = CreateGaussLegendreRule(40);

        /// <summary>
        /// Computes Scorer's function <c>Gi(x)</c>.
        /// <para>スコーラー関数 <c>Gi(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The real argument. / 実数の引数。</param>
        /// <returns>The value of <c>Gi(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is <see cref="double.NaN"/> or phase precision is exhausted for extremely large negative <paramref name="x"/>. / <c>Gi(x)</c> の値。<paramref name="x"/> が <see cref="double.NaN"/> の場合、または極端に大きな負の <paramref name="x"/> で位相精度を保てない場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <c>0</c> at <c>x = ±∞</c>.</para>
        /// <para><c>x = ±∞</c> では <c>0</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/9.12"/>
        public static double ScorerGi(double x)
        {
            if (double.IsNaN(x))
            {
                return x;
            }

            if (double.IsInfinity(x))
            {
                return 0;
            }

            if (x >= ScorerAsymptoticMin)
            {
                return ScorerAsymptotic(x, alternating: false);
            }

            if (x >= 0)
            {
                return ScorerIntegral(x, oscillating: true);
            }

            // Gi + Hi = Bi。x < 0 では Hi(x) > 0 が 1/(π|x|) で減衰し、Bi(x) が振動の大部分を担う
            double bi = ScorerAiryBiNegative(-x);
            return bi - ScorerHiNegative(x);
        }

        /// <summary>
        /// Computes Scorer's function <c>Hi(x)</c>.
        /// <para>スコーラー関数 <c>Hi(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The real argument. / 実数の引数。</param>
        /// <returns>The value of <c>Hi(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is <see cref="double.NaN"/>. / <c>Hi(x)</c> の値。<paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <c>0</c> at <c>x = -∞</c> and overflows to <see cref="double.PositiveInfinity"/> for <c>x ≳ 104.42</c>.</para>
        /// <para><c>x = -∞</c> では <c>0</c> を返し、<c>x ≳ 104.42</c> では <see cref="double.PositiveInfinity"/> にオーバーフローします。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/9.12"/>
        public static double ScorerHi(double x)
        {
            if (double.IsNaN(x))
            {
                return x;
            }

            if (double.IsPositiveInfinity(x))
            {
                return double.PositiveInfinity;
            }

            if (x <= 0)
            {
                return ScorerHiNegative(x);
            }

            double bi = x <= 1 ? AiryBi(x) : ScorerAiryBiPositive(x);
            if (double.IsPositiveInfinity(bi))
            {
                return bi;
            }

            double gi = x >= ScorerAsymptoticMin ? ScorerAsymptotic(x, alternating: false) : ScorerIntegral(x, oscillating: true);
            return bi - gi;
        }

        // Hi(x)（x ≤ 0）
        private static double ScorerHiNegative(double x)
        {
            if (double.IsNegativeInfinity(x))
            {
                return 0;
            }

            if (x <= -ScorerAsymptoticMin)
            {
                return ScorerAsymptotic(-x, alternating: true);
            }

            return ScorerIntegral(-x, oscillating: false);
        }

        // c ≥ 0 に対し、oscillating が真なら Gi(c) = (1/π) ∫₀^∞ e^(-s³/3 - cs/2) sin(√3cs/2 + π/6) ds、
        // 偽なら Hi(-c) = (1/π) ∫₀^∞ e^(-s³/3 - cs) ds を計算する。
        // 指数部 s³/3 + bs が ScorerTruncationExponent に達する点 S で打ち切り、[0, S] を ScorerRule で積分する
        private static double ScorerIntegral(double c, bool oscillating)
        {
            // c = -0 で 41/b が -∞ にならないよう絶対値をとる
            double b = Math.Abs(oscillating ? 0.5 * c : c);
            double limit = ScorerTruncationPoint(b);
            double half = 0.5 * limit;
            double frequency = 0.5 * Sqrt3 * c;
            double[] nodes = ScorerRule.Nodes;
            double[] weights = ScorerRule.Weights;
            double sum = 0;
            for (int i = 0; i < nodes.Length; i++)
            {
                // 節点 ±tᵢ を [0, S] の s = (1 ± tᵢ)S/2 に写す
                double sMinus = half * (1 - nodes[i]);
                double sPlus = half * (1 + nodes[i]);
                double fMinus = Math.Exp(-sMinus * (sMinus * sMinus / 3 + b));
                double fPlus = Math.Exp(-sPlus * (sPlus * sPlus / 3 + b));
                if (oscillating)
                {
                    fMinus *= Math.Sin(frequency * sMinus + Math.PI / 6);
                    fPlus *= Math.Sin(frequency * sPlus + Math.PI / 6);
                }

                sum += weights[i] * (fMinus + fPlus);
            }

            return sum * half / Math.PI;
        }

        // s³/3 + bs = 41（b ≥ 0）の正の解。min(41/b, 123^(1/3)) は解以上なので、凸な増加関数へのニュートン法は右側から単調に収束する
        private static double ScorerTruncationPoint(double b)
        {
            double s = Math.Min(ScorerTruncationExponent / b, Math.Cbrt(3 * ScorerTruncationExponent));
            for (int i = 0; i < 30; i++)
            {
                double step = (s * (s * s / 3 + b) - ScorerTruncationExponent) / (s * s + b);
                s -= step;
                if (step <= 1e-6 * s)
                {
                    break;
                }
            }

            return s;
        }

        // 漸近展開 (1/(πz)) Σ (±1)ᵏ (3k)!/(k! (3z³)ᵏ)（z ≥ 20）。
        // alternating = false で Gi(z)（正項）、true で Hi(-z)（交代級数）
        private static double ScorerAsymptotic(double z, bool alternating)
        {
            // z³ のオーバーフローを避けるため 1/z を 3 回掛ける（z が大きければ補正項は単に 0 になる）
            double inv = 1 / z;
            double inv3 = inv * inv * inv;
            double term = 1;
            double sum = 1;
            for (int k = 1; k < 60; k++)
            {
                term *= (3.0 * k - 2) * (3 * k - 1) * inv3;
                sum += alternating && (k & 1) == 1 ? -term : term;
                if (term <= 0.25 * Epsilon * sum)
                {
                    break;
                }
            }

            // π z は z ≈ 10³⁰⁸ でオーバーフローするため、1/π を先に掛ける
            return sum * (1 / Math.PI) / z;
        }

        // ζ = (2/3) z^(3/2) を double-double（hi + lo）で求める（z > 0）。
        // ζ は e^ζ の指数や cos ζ の位相に入るため、倍精度丸めの ε ζ の誤差を防ぐ
        private static (double Hi, double Lo) ScorerAiryZeta(double z)
        {
            // √z = s + e（e = (z - s²)/(2s)、z - s² は FMA で正確に求まる）
            double s = Math.Sqrt(z);
            double e = Math.FusedMultiplyAdd(-s, s, z) / (2 * s);

            // z√z = p + pe
            double p = z * s;
            double pe = Math.FusedMultiplyAdd(z, s, -p) + z * e;

            // (2/3)(p + pe)：2p は正確で、2p/3 の剰余も FMA で正確に求まる
            double q = 2 * p / 3;
            double r = Math.FusedMultiplyAdd(-q, 3, 2 * p);
            return DoubleDoubleTwoSum(q, (r + 2 * pe) / 3);
        }

        // Bi(x) = √x ((2/√3) I₁/₃(ζ) + K₁/₃(ζ)/π)（x > 1）。
        // e^ζ は double-double の ζ から 2ⁿ e^r（r = ζ - n ln 2）としてスケーリングする
        private static double ScorerAiryBiPositive(double x)
        {
            var (zeta, zetaLo) = ScorerAiryZeta(x);
            if (!(zeta <= ScorerBiOverflowZeta))
            {
                // Bi(x) ≈ e^ζ/(√π x^(1/4)) は x ≈ 104.42（ζ ≈ 711.4）でオーバーフローする（z√z がオーバーフローすると ζ は NaN になる）
                return double.PositiveInfinity;
            }

            BesselIKCore(1.0 / 3, zeta, out double iScaled, out double kScaled);
            double root = Math.Sqrt(x);

            // e^ζ = 2ⁿ e^r。r = ζ - n·ln2 は ln2 の上位を FMA で、下位と ζ の下位を後から足す（|r| ≤ ln2/2）
            double n = Math.Round(zeta / Ln2);
            double reduced = Math.FusedMultiplyAdd(-n, Ln2, zeta) - n * ScorerLn2Lo + zetaLo;
            double growth = Math.Exp(reduced);

            // K の項は I の項に対して e^(-2ζ) 程度。n が大きければ単に 0 になる
            double decay = ScaleB(Math.Exp(-reduced), (long)-n);
            double mantissa = root * 2 * OneOverSqrt3 * iScaled * growth;
            return ScaleB(mantissa, (long)n) + root * kScaled * decay / Math.PI;
        }

        // Bi(-z)（z ≥ 0）
        private static double ScorerAiryBiNegative(double z)
        {
            if (z < ScorerAiryNegativeAsymptoticMin)
            {
                return AiryBi(-z);
            }

            ScorerAiryNegativeAsymptotic(z, out _, out double bi);
            return bi;
        }

        // DLMF 9.7.9, 9.7.11: Ai(-z), Bi(-z) の漸近展開（z ≥ 12）。
        // 位相は double-double の ζ = hi + lo から加法定理で組み立て、ζ > ScorerAiryPhaseMaxZeta では NaN を返す
        private static void ScorerAiryNegativeAsymptotic(double z, out double ai, out double bi)
        {
            var (zeta, zetaLo) = ScorerAiryZeta(z);
            if (!(zeta <= ScorerAiryPhaseMaxZeta))
            {
                ai = double.NaN;
                bi = double.NaN;
                return;
            }

            // P、Q の和。項は k ≈ 2ζ まで減少するので、ε 未満になったら打ち切る（ζ ≥ 27.7 では 20 項程度）
            double inv = 1 / zeta;
            double u = 1;
            double power = 1;
            double pSum = 1;
            double qSum = 0;
            for (int k = 1; k < 200; k++)
            {
                u *= (6.0 * k - 5) * (6 * k - 3) * (6 * k - 1) / (216.0 * k * (2 * k - 1));
                power *= inv;
                double term = u * power;

                // k = 1, 2, 3, 4, … の符号は Q:+、P:-、Q:-、P:+、…
                int sign = (k & 3) is 0 or 1 ? 1 : -1;
                if ((k & 1) == 0)
                {
                    pSum += sign * term;
                }
                else
                {
                    qSum += sign * term;
                }

                if (term <= 0.25 * Epsilon)
                {
                    break;
                }
            }

            // cos(ζ - π/4) = (cos ζ + sin ζ)/√2、sin(ζ - π/4) = (sin ζ - cos ζ)/√2 に、ζ の下位 lo を加法定理で加える
            // （ζ ≈ 10¹⁵ では lo が 0.1 程度になるので、1次の近似では足りない）
            double cosZeta = Math.Cos(zeta);
            double sinZeta = Math.Sin(zeta);
            double c = InvSqrt2 * (cosZeta + sinZeta);
            double s = InvSqrt2 * (sinZeta - cosZeta);
            double cosLo = Math.Cos(zetaLo);
            double sinLo = Math.Sin(zetaLo);
            double cosPhase = c * cosLo - s * sinLo;
            double sinPhase = s * cosLo + c * sinLo;
            double amplitude = OneOverSqrtPi / Math.Sqrt(Math.Sqrt(z));
            ai = amplitude * (cosPhase * pSum + sinPhase * qSum);
            bi = amplitude * (cosPhase * qSum - sinPhase * pSum);
        }
    }
}
