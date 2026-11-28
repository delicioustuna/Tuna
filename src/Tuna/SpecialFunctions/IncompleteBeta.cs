using static Tuna.XMath.Constants;
using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        /// <summary>
        /// Computes the regularized incomplete beta function <c>Iₓ(a, b) = Bₓ(a, b) / B(a, b)</c>.
        /// <para>正則化不完全ベータ関数 <c>Iₓ(a, b) = Bₓ(a, b) / B(a, b)</c> を計算します。</para>
        /// </summary>
        /// <param name="a">The first shape parameter (<c>a &gt; 0</c>). / 第 1 形状パラメータ（<c>a &gt; 0</c>）。</param>
        /// <param name="b">The second shape parameter (<c>b &gt; 0</c>). / 第 2 形状パラメータ（<c>b &gt; 0</c>）。</param>
        /// <param name="x">The upper integration limit (<c>0 ≤ x ≤ 1</c>). / 積分上限（<c>0 ≤ x ≤ 1</c>）。</param>
        /// <returns>The value of <c>Iₓ(a, b)</c>, or <see cref="double.NaN"/> if any argument is outside its domain or <see cref="double.NaN"/>. / <c>Iₓ(a, b)</c> の値。引数が定義域外または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/8.17.E2"/>
        public static double RegularizedIncompleteBeta(double a, double b, double x)
        {
            if (!TryIncompleteBetaSpecialCase(a, b, x, out double i, out _))
            {
                IncompleteBetaCore(a, b, x, 1 - x, out i, out _);
            }

            return i;
        }

        /// <summary>
        /// Computes the complement of the regularized incomplete beta function, <c>1 - Iₓ(a, b) = I₁₋ₓ(b, a)</c>.
        /// <para>正則化不完全ベータ関数の補関数 <c>1 - Iₓ(a, b) = I₁₋ₓ(b, a)</c> を計算します。</para>
        /// </summary>
        /// <param name="a">The first shape parameter (<c>a &gt; 0</c>). / 第 1 形状パラメータ（<c>a &gt; 0</c>）。</param>
        /// <param name="b">The second shape parameter (<c>b &gt; 0</c>). / 第 2 形状パラメータ（<c>b &gt; 0</c>）。</param>
        /// <param name="x">The lower integration limit (<c>0 ≤ x ≤ 1</c>). / 積分下限（<c>0 ≤ x ≤ 1</c>）。</param>
        /// <returns>The value of <c>1 - Iₓ(a, b)</c>, or <see cref="double.NaN"/> if any argument is outside its domain or <see cref="double.NaN"/>. / <c>1 - Iₓ(a, b)</c> の値。引数が定義域外または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Avoids catastrophic cancellation when <c>Iₓ(a, b)</c> is close to <c>1</c>.</para>
        /// <para><c>Iₓ(a, b)</c> が <c>1</c> に近い場合でも桁落ちを防いで計算します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/8.17.E4"/>
        public static double RegularizedIncompleteBetaComplement(double a, double b, double x)
        {
            if (!TryIncompleteBetaSpecialCase(a, b, x, out _, out double complement))
            {
                IncompleteBetaCore(a, b, x, 1 - x, out _, out complement);
            }

            return complement;
        }

        /// <summary>
        /// Computes the incomplete beta function <c>Bₓ(a, b) = ∫₀ˣ t^(a - 1) (1 - t)^(b - 1) dt</c>.
        /// <para>不完全ベータ関数 <c>Bₓ(a, b) = ∫₀ˣ t^(a - 1) (1 - t)^(b - 1) dt</c> を計算します。</para>
        /// </summary>
        /// <param name="a">The first shape parameter (<c>a &gt; 0</c>). / 第 1 形状パラメータ（<c>a &gt; 0</c>）。</param>
        /// <param name="b">The second shape parameter (<c>b &gt; 0</c>). / 第 2 形状パラメータ（<c>b &gt; 0</c>）。</param>
        /// <param name="x">The upper integration limit (<c>0 ≤ x ≤ 1</c>). / 積分上限（<c>0 ≤ x ≤ 1</c>）。</param>
        /// <returns>The value of <c>Bₓ(a, b)</c>, or <see cref="double.NaN"/> if any argument is outside its domain or <see cref="double.NaN"/>. / <c>Bₓ(a, b)</c> の値。引数が定義域外または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/8.17.E1"/>
        public static double IncompleteBeta(double a, double b, double x)
        {
            if (b < TinyIncompleteBetaParameter && b > 0 && x < 1)
            {
                // 極小の b では Iₓ(a, b) は b に比例して非正規化数や 0 になり、B(a, b) ≈ 1/b はオーバーフローするため正則化経由では精度を失う。
                // 一方 x < 1 では |∂ ln Bₓ(a, b)/∂b| ≤ |ln(1 - x)| ≤ 53 ln 2 なので、b を 2⁻⁷⁰ に置き換えても Bₓ は相対 3e-20 未満しか変わらない
                b = TinyIncompleteBetaParameter;
            }

            double regularized = RegularizedIncompleteBeta(a, b, x);
            if (double.IsNaN(regularized) || x == 0)
            {
                return regularized;
            }

            // 通常の大きさなら、正則化値に B(a, b) を掛け戻す
            double beta = Beta(a, b);
            bool betaNormal = double.IsFinite(beta) && beta >= MinNormal;
            double value = regularized * beta;
            if (betaNormal && regularized >= MinNormal && double.IsFinite(value) && value > 1e-290)
            {
                return value;
            }

            // 極端に小さい値では正則化値が対数経由の誤差を持つか非正規化数や 0 に丸まっているため、連分数の値に xᵃ yᵇ を直接掛ける
            if (TryUnregularizedIncompleteBeta(a, b, x, out double unregularized))
            {
                return unregularized;
            }

            // Iₓ が正規化数なら積は最後の 1 回丸めるのみ。Iₓ が非正規化数でも B(a, b) ≤ 1 なら量子化誤差は B(a, b) 倍に縮む
            if (betaNormal && (regularized >= MinNormal || (regularized > 0 && beta <= 1)) && double.IsFinite(value))
            {
                return value;
            }

            // 最後の手段として対数から戻す（正則化値が非正規化数や 0 なら対数版で再計算する）
            double logRegularized = regularized >= MinNormal ? Math.Log(regularized) : LogRegularizedIncompleteBeta(a, b, x);
            return Math.Exp(logRegularized + LogBeta(a, b));
        }

        // Bₓ(a, b) = xᵃ yᵇ · f を B(a, b) や前因子の対数を経由せずに計算する（連分数で下側を求める経路に限る）。
        // xᵃ = (x^(a/2))² を仮数・指数分離して掛け、yᵇ は Sterbenz の補題で正確な丸め誤差 y_lo = (1 - y) - x で 1 次補正する
        private static bool TryUnregularizedIncompleteBeta(double a, double b, double x, out double value)
        {
            value = 0;
            // 一様漸近展開を実際に使う（IncompleteBetaCore と同じ判定）ときは連分数の値がないので除く。
            // min(a, b) が下限以上でも、中心から遠い裾では連分数の経路になる
            if (!(x > 0 && x < 1) || b >= MaxUnregularizedBetaParameter
                || (Math.Min(a, b) >= IncompleteBetaExpansionMin && TryIncompleteBetaAsymptoticTailFactor(a, b, x, out _, out _, out _)))
            {
                return false;
            }

            double y = 1 - x;
            double fraction = IncompleteBetaTailFraction(a, b, x, y, BetaPrefix(a, b, x, y), log: false, out bool upper, out bool direct);
            double yb = Math.Pow(y, b);
            if (x < 0.5)
            {
                // y^b = (y + y_lo)^b ≈ y^b (1 + b·y_lo/y)。b·|y_lo/y| ≤ 2²⁶·2⁻⁵³ なので 2 次の項は無視できる
                yb += yb * (b * (((1 - y) - x) / y));
            }

            double half = Math.Pow(x, 0.5 * a);
            if (upper || direct || !(fraction > 0) || !double.IsFinite(fraction) || !(yb >= MinNormal) || !(half >= MinNormal))
            {
                return false;
            }

            int exponent = Math.ILogB(half);
            double mantissa = Math.ScaleB(half, -exponent);
            value = Math.ScaleB(mantissa * mantissa * yb * fraction, 2 * exponent);
            return true;
        }

        // TryUnregularizedIncompleteBeta で yᵇ の 1 次補正が十分な b の上限（2²⁶）
        private const double MaxUnregularizedBetaParameter = 67108864;

        // IncompleteBeta で極小の第 2 パラメータを置き換える閾値（2⁻⁷⁰）
        private const double TinyIncompleteBetaParameter = 8.470329472543003e-22;

        /// <summary>
        /// Computes the inverse of the regularized incomplete beta function, finding <c>x</c> such that <c>Iₓ(a, b) = p</c>.
        /// <para>正則化不完全ベータ関数の逆関数（<c>Iₓ(a, b) = p</c> を満たす <c>x</c>）を計算します。</para>
        /// </summary>
        /// <param name="a">The first shape parameter (<c>a &gt; 0</c>). / 第 1 形状パラメータ（<c>a &gt; 0</c>）。</param>
        /// <param name="b">The second shape parameter (<c>b &gt; 0</c>). / 第 2 形状パラメータ（<c>b &gt; 0</c>）。</param>
        /// <param name="p">The probability (<c>0 ≤ p ≤ 1</c>). / 確率（<c>0 ≤ p ≤ 1</c>）。</param>
        /// <returns>The value <c>x</c> in <c>[0, 1]</c>, or <see cref="double.NaN"/> if any argument is outside its domain or <see cref="double.NaN"/>. / 範囲 <c>[0, 1]</c> の <c>x</c>。引数が定義域外または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/8.17.E2"/>
        public static double InverseRegularizedIncompleteBeta(double a, double b, double p) =>
            InverseIncompleteBetaCore(a, b, p, 1 - p);

        /// <summary>
        /// Computes the inverse of the complemented regularized incomplete beta function, finding <c>x</c> such that <c>1 - Iₓ(a, b) = q</c>.
        /// <para>正則化不完全ベータ関数の補関数の逆関数（<c>1 - Iₓ(a, b) = q</c> を満たす <c>x</c>）を計算します。</para>
        /// </summary>
        /// <param name="a">The first shape parameter (<c>a &gt; 0</c>). / 第 1 形状パラメータ（<c>a &gt; 0</c>）。</param>
        /// <param name="b">The second shape parameter (<c>b &gt; 0</c>). / 第 2 形状パラメータ（<c>b &gt; 0</c>）。</param>
        /// <param name="q">The complementary probability (<c>0 ≤ q ≤ 1</c>). / 上側確率（<c>0 ≤ q ≤ 1</c>）。</param>
        /// <returns>The value <c>x</c> in <c>[0, 1]</c>, or <see cref="double.NaN"/> if any argument is outside its domain or <see cref="double.NaN"/>. / 範囲 <c>[0, 1]</c> の <c>x</c>。引数が定義域外または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/8.17.E4"/>
        public static double InverseRegularizedIncompleteBetaComplement(double a, double b, double q) =>
            InverseIncompleteBetaCore(a, b, 1 - q, q);

        private static bool TryIncompleteBetaSpecialCase(double a, double b, double x, out double i, out double complement)
        {
            if (double.IsNaN(a) || double.IsNaN(b) || double.IsNaN(x) || a <= 0 || b <= 0
                || double.IsInfinity(a) || double.IsInfinity(b) || x < 0 || x > 1)
            {
                i = complement = double.NaN;
                return true;
            }

            if (x == 0)
            {
                i = 0;
                complement = 1;
                return true;
            }

            if (x == 1)
            {
                i = 1;
                complement = 0;
                return true;
            }

            i = complement = 0;
            return false;
        }

        // Iₓ(a, b) と 1 - Iₓ(a, b) を小さい方から直接計算する（y = 1 - x）
        private static void IncompleteBetaCore(double a, double b, double x, double y, out double i, out double complement)
        {
            IncompleteBetaCore(a, b, x, y, out i, out complement, out _);
        }

        // 逆関数の導関数用に前因子 prefix も返す（一様漸近展開経路では NaN）
        private static void IncompleteBetaCore(double a, double b, double x, double y, out double i, out double complement, out double prefix)
        {
            prefix = double.NaN;
            if (Math.Min(a, b) >= IncompleteBetaExpansionMin && TryIncompleteBetaAsymptotic(a, b, x, out i, out complement))
            {
                return;
            }

            // 前因子を掛ける側では、前因子がアンダーフローしたら裾は 0（連分数の値によらない）
            prefix = BetaPrefix(a, b, x, y);
            double fraction = IncompleteBetaTailFraction(a, b, x, y, prefix, log: false, out bool upper, out bool direct);
            double tail = direct ? fraction : prefix == 0 ? 0 : prefix * fraction;
            if (upper)
            {
                complement = tail;
                i = 1 - complement;
            }
            else
            {
                i = tail;
                complement = 1 - i;
            }
        }

        // min(a, b) ≤ 1 で下側 Iₓ(a, b) と上側 I₁₋ₓ(b, a) のどちらを直接計算するか判定する
        private static bool UseIncompleteBetaLowerFraction(double a, double b, double x) => x < (a + 1) / (a + b + 2);

        // 前因子 xᵃ yᵇ / B(a, b) に掛けて直接計算側の裾を得る連分数の値を返す（upper = true で補関数側）。
        // direct = true のときは前因子を掛けない裾そのもの（log = true なら裾の対数）を返す。
        // min(a, b) > 1 は BFRAC、min(a, b) ≤ 1 は先頭パラメータに応じて BetaFraction / BFRAC / 小パラメータ級数を使い分ける
        private static double IncompleteBetaTailFraction(double a, double b, double x, double y, double prefix, bool log,
            out bool upper, out bool direct)
        {
            direct = false;
            if (Math.Min(a, b) > 1)
            {
                double lambda = -IncompleteBetaCenterDeviation(a, b, x, y);
                upper = lambda < 0;
                return upper ? DiDonatoMorrisBetaFraction(b, a, y, x, -lambda) : DiDonatoMorrisBetaFraction(a, b, x, y, lambda);
            }

            // UseIncompleteBetaLowerFraction の各側で、λ は BFRAC の条件（先頭のパラメータの側で非負）を満たす：
            // 下側で a > 1 ≥ b なら λ > (a - b)/(a + b + 2) > 0、上側で b > 1 ≥ a なら λ < 0
            upper = !UseIncompleteBetaLowerFraction(a, b, x);
            if (!upper)
            {
                if (a > 1)
                {
                    return DiDonatoMorrisBetaFraction(a, b, x, y, -IncompleteBetaCenterDeviation(a, b, x, y));
                }

                double lowerContinuedFraction = BetaFraction(a, b, x);
                double lowerFraction = lowerContinuedFraction / a;
                double lowerTail = prefix * lowerFraction;
                bool lowerScaled = !(prefix >= MinNormal);
                double logLowerScaledPrefix = 0;
                if (lowerScaled)
                {
                    lowerTail = SmallParameterBetaScaledPrefix(a, b, x, Math.Log(x), x < 0.5 ? Log1p(-x) : Math.Log(y),
                        out logLowerScaledPrefix) * lowerContinuedFraction;
                }

                if (lowerTail > 0.5)
                {
                    // 補関数を、級数が使えるなら小さな a のべき級数から、そうでなく Iₓ > 0.9 なら BFRAC で直接求める
                    if (b <= 1 ? x <= 0.7 : b * x <= 1)
                    {
                        double complementValue = log
                            ? SmallParameterBetaLogComplement(a, b, x, Math.Log(x))
                            : SmallParameterBetaComplement(a, b, x, Math.Log(x));
                        if (!double.IsNaN(complementValue))
                        {
                            upper = true;
                            direct = true;
                            return complementValue;
                        }
                    }

                    if (lowerTail > 0.9 && b > 1)
                    {
                        double deviation = IncompleteBetaCenterDeviation(a, b, x, y);
                        if (deviation > 0)
                        {
                            upper = true;
                            return DiDonatoMorrisBetaFraction(b, a, y, x, deviation);
                        }
                    }
                }

                if (lowerScaled)
                {
                    direct = true;
                    return log ? logLowerScaledPrefix + Math.Log(lowerContinuedFraction) : lowerTail;
                }

                return lowerFraction;
            }

            if (b > 1)
            {
                return DiDonatoMorrisBetaFraction(b, a, y, x, IncompleteBetaCenterDeviation(a, b, x, y));
            }

            // x < 1/2 の y は丸められているので、ln y は x から求める
            double upperContinuedFraction = BetaFraction(b, a, y);
            double upperFraction = upperContinuedFraction / b;
            double upperTail = prefix * upperFraction;
            bool upperScaled = !(prefix >= MinNormal);
            double logUpperScaledPrefix = 0;
            if (upperScaled)
            {
                upperTail = SmallParameterBetaScaledPrefix(b, a, y, x < 0.5 ? Log1p(-x) : Math.Log(y), Math.Log(x),
                    out logUpperScaledPrefix) * upperContinuedFraction;
            }

            if (upperTail > 0.5)
            {
                // Iₓ を、級数が使えるなら小さな b のべき級数から、そうでなく補関数 > 0.9 なら BFRAC で直接求める
                if (a <= 1 ? y <= 0.7 : a * y <= 1)
                {
                    double lny = x < 0.5 ? Log1p(-x) : Math.Log(y);
                    double value = log
                        ? SmallParameterBetaLogComplement(b, a, y, lny)
                        : SmallParameterBetaComplement(b, a, y, lny);
                    if (!double.IsNaN(value))
                    {
                        upper = false;
                        direct = true;
                        return value;
                    }
                }

                if (upperTail > 0.9 && a > 1)
                {
                    double lambda = -IncompleteBetaCenterDeviation(a, b, x, y);
                    if (lambda > 0)
                    {
                        upper = false;
                        return DiDonatoMorrisBetaFraction(a, b, x, y, lambda);
                    }
                }
            }

            if (upperScaled)
            {
                direct = true;
                return log ? logUpperScaledPrefix + Math.Log(upperContinuedFraction) : upperTail;
            }

            return upperFraction;
        }

        // p ≤ 1 で I_z(p, q) が 1 に近いときの補関数 1 - I_z(p, q) を、べき級数（DLMF 8.17.8）の対数から -expm1(ln I) として求める。
        // 各項の打ち消しが大きい場合は NaN を返し、呼び出し側で別手法に切り替える
        private static double SmallParameterBetaComplement(double p, double q, double z, double lnz)
        {
            // ln I = pσ が極小なら、非正規化数の範囲で丸め誤差が積み重なった ln I を使わず、p と σ の積として 1 回だけ丸める
            // （このとき |ln I| ≪ ε で -expm1(ln I) = -ln I）
            double logValue = SmallParameterBetaLogValue(p, q, z, lnz, out double scaledLogValue);
            return double.IsNaN(scaledLogValue) ? -Expm1(logValue) : -(p * scaledLogValue);
        }

        // SmallParameterBetaComplement の補関数の対数 ln(1 - I_z(p, q))（適用不可なら NaN）
        private static double SmallParameterBetaLogComplement(double p, double q, double z, double lnz)
        {
            double logValue = SmallParameterBetaLogValue(p, q, z, lnz, out double scaledLogValue);
            return double.IsNaN(scaledLogValue) ? Log1mExp(logValue) : Math.Log(p) + Math.Log(-scaledLogValue);
        }

        // べき級数による ln I_z(p, q)。|ln I| < SmallParameterBetaScaledLogMax なら極限式 σ = ln I / p を scaledLogValue に返す
        private static double SmallParameterBetaLogValue(double p, double q, double z, double lnz, out double scaledLogValue)
        {
            scaledLogValue = double.NaN;

            // 打ち消しの判定はコストの低い項から順に行い、級数の和は最後に求める
            double logPower = p * lnz;
            if (!(Math.Abs(logPower) <= 2))
            {
                return double.NaN;
            }

            double logPB = LogGammaShift(1, p) - LogGammaShift(q, p);
            if (!(Math.Abs(logPower) + Math.Abs(logPB) <= 2))
            {
                return double.NaN;
            }

            double c = 1;
            double t = 0;
            for (int j = 1; j < 1000; j++)
            {
                c *= (j - q) * z / j;
                double term = c / (p + j);
                t += term;
                if (!(Math.Abs(term) > 0.5 * MachineEpsilon * Math.Abs(t)))
                {
                    break;
                }
            }

            double logSeries = Log1p(p * t);
            if (!(Math.Abs(logPower) + Math.Abs(logSeries) + Math.Abs(logPB) <= 2))
            {
                return double.NaN;
            }

            double logValue = logPower + logSeries - logPB;
            if (Math.Abs(logValue) < SmallParameterBetaScaledLogMax)
            {
                scaledLogValue = lnz + t + (EulerGamma + Digamma(q));
            }

            return logValue;
        }

        // SmallParameterBetaLogValue で ln I を σ = ln I/p から再計算する上限（2⁻⁹⁷⁰ ≈ 1e-292）
        private const double SmallParameterBetaScaledLogMax = MinNormal / MachineEpsilon;

        // 前因子を先頭パラメータ p ≤ 1 で割った zᵖ wᑫ/(p B(p, q)) とその対数 logValue を、前因子が非正規化数でも精度を保って計算する
        private static double SmallParameterBetaScaledPrefix(double p, double q, double z, double lnz, double lnw, out double logValue)
        {
            double power;
            double logGamma;
            double ratio = 1;
            double logRatio = 0;
            if (q < 1)
            {
                power = Math.Pow(z, p);
                logGamma = LogGammaShift(1 + q, p) - LogGammaShift(1, p);
                ratio = q / (p + q);

                // ln(q/(p + q)) = -ln(1 + p/q)。p/q ≥ 2⁵² では ln(p/q) との差は ε 未満で、Log1p の途中の積がオーバーフローするのも避ける（p/q がオーバーフローするなら対数の差）
                double pq = p / q;
                logRatio = -(pq < 1 / MachineEpsilon ? Log1p(pq) : pq <= double.MaxValue ? Math.Log(pq) : Math.Log(p) - Math.Log(q));
            }
            else if (q < 10)
            {
                power = Math.Pow(z, p);
                logGamma = LogGammaShift(q, p) - LogGammaShift(1, p);
            }
            else
            {
                // zq が非正規化数なら丸め誤差が累乗で拡大するので、zᵖ と qᵖ を別に求める
                double zq = z * q;
                power = zq >= MinNormal ? Math.Pow(zq, p) : Math.Pow(z, p) * Math.Pow(q, p);
                logGamma = LogScaledGammaShiftRatio(q, p) - LogGammaShift(1, p);
                logRatio = p * Math.Log(q);
            }

            double exponent = q * lnw + logGamma;
            logValue = p * lnz + exponent + logRatio;
            return power * Math.Exp(exponent) * ratio;
        }

        // 高次級数の範囲外で一様漸近展開の先頭 2 項 c₀ のみを使う min(a, b) の下限
        private const double IncompleteBetaAsymptoticMin = 1e9;

        // 中心付近で高次項 c₁, …, c₅ 付きの一様漸近展開を使う min(a, b) の下限（BFRAC の O(min(a,b)^(1/3)) 反復を回避）
        private const double IncompleteBetaExpansionMin = 300;

        // 係数 c₀, …, c₅ を τ の級数で評価する範囲（|τ| ≤ 0.1, |rτ| ≤ 0.1）
        private const double IncompleteBetaSeriesRadius = 0.1;

        // 高次展開を用いる中心付近の範囲 t = |η|√(μ/2) ≤ 3（約 ±4σ）
        private const double IncompleteBetaExpansionMaxT = 3;

        // 高次係数の r 多項式（最大 r²⁹）を直接評価する |r| の上限
        private const double IncompleteBetaSeriesMaxR = 1e8;

        // 一様漸近展開の係数 c_k(τ) = Σⱼ P_kj(r) τʲ（k = 0, …, 5）。
        // [k][j] は P_kj を j 偶数で r·Q(r²)、j 奇数で Q(r²) としたときの Q の係数（r² の昇冪、Temme の部分積分から有理数厳密計算）
        private static readonly double[][][] IncompleteBetaTemmeCoefficients =
        [
            [
                [0.3333333333333333],
                [-0.25, -0.08333333333333333],
                [0.15, 0.04259259259259259],
                [-0.07291666666666667, -0.12083333333333333, -0.027237654320987653],
                [0.1181547619047619, 0.10522486772486772, 0.01947751322751323],
                [-0.0390625, -0.15521329365079364, -0.09511078042328042, -0.01489620076425632],
                [0.10270337301587301, 0.18783344356261023, 0.0878720238095238, 0.011915478640015678],
                [-0.025618489583333334, -0.18666294642857142, -0.21756675760582012, -0.08236033032039977, -0.009842230520037232],
                [0.09311657309704184, 0.2886212196368446, 0.24523598459796375, 0.07798145559430282, 0.008328093512180232],
                [-0.0186279296875, -0.21602271955578317, -0.407067668235048, -0.2713378830696022, -0.07439285172732697, -0.007180348385069859],
                [0.08640992308472777, 0.4063752985760519, 0.5409230237871421, 0.29619847676979166, 0.07138135025685567, 0.006284419272101214],
                [-0.014422526687541336, -0.24381870857962515, -0.6752418017364598, -0.6893687393711497, -0.3200447827702161, -0.06880645076142598, -0.005568252617885114],
                [0.08136930483223614, 0.540209145632032, 1.0329960266821994, 0.8517572272969399, 0.34304207959408256, 0.06657130144773722, 0.004984445684415658],
                [-0.011648337550894923, -0.2703895014982705, -1.0331242368240703, -1.489485734829546, -1.0275597078861047, -0.3653149686544622, -0.06460668939591682, -0.004500636357257656],
                [0.07739538058774366, 0.6894362988575196, 1.78779909020276, 2.0541425116669156, 1.216333696977203, 0.3869600622474113, 0.06286166624067879, 0.0040940356694601585],
                [-0.0096978022644339, -0.29596589381939176, -1.4912939876075446, -2.876784641325838, -2.736058019811077, -1.4177016176946073, -0.4080540238256637, -0.06129778684731249, -0.0037481681888972334],
                [0.07415355640746502, 0.8535032212270799, 2.8796264031531122, 4.379649816838718, 3.5440393109394983, 1.6313361163077147, 0.4286588774936086, 0.05988542319602994, 0.003450850376970949],
                [-0.008260684890091104, -0.32071312081359726, -2.0599630986051456, -5.114737962913345, -6.382738105339748, -4.486650379372437, -1.8569496224690476, -0.4488256366536482, -0.058601324502998264, -0.003192893611806143],
                [0.07144029956106275, 1.0319501028832172, 4.390454983979384, 8.51910706507707, 8.978963566347211, 5.572244285344303, 2.094286707398352, 0.4685968570810906, 0.05742695455432526, 0.0029672473773571044],
            ],
            [
                [0.13333333333333333, 0.02962962962962963],
                [-0.03125, -0.020833333333333332, -0.003472222222222222],
                [-0.008630952380952382, -0.0036375661375661374, -0.0002535273368606702],
                [0.005208333333333333, 0.022767857142857142, 0.010179673721340389, 0.0011592445620223397],
                [-0.02217261904761905, -0.03381283068783069, -0.012593694885361552, -0.0013650058788947678],
                [0.006217447916666666, 0.043587549603174604, 0.042875124007936506, 0.013540885875955321, 0.0013559081361454047],
                [-0.0249412371482684, -0.0691545539722623, -0.050690776398241676, -0.013862001017672314, -0.0012779089894841352],
                [0.0052734375, 0.05856812303519893, 0.09838111128289699, 0.05764859568617009, 0.013887674358697838, 0.0011814267551433495],
                [-0.025205577842296593, -0.10950622800232175, -0.13090588857515942, -0.06397612873451994, -0.013766433063924275, -0.001084934951934909],
                [0.0043465912026703045, 0.07073990511054545, 0.17993337041981675, 0.16645827596153892, 0.06981811958385072, 0.013571188251621602, 0.0009950515588534022],
                [-0.024712885652216708, -0.15445438098427078, -0.27184237262462824, -0.20482763537590803, -0.0752726139502461, -0.013339983051087984, -0.0009137289777399125],
                [0.0036137737294353505, 0.08121737851632581, 0.29042581182344424, 0.3870815654974497, 0.24584454487614568, 0.08040930974212661, 0.01309349997140734, 0.0008410545171479478],
                [-0.02398239171199759, -0.20363202124593388, -0.4941137005874472, -0.5273838079628068, -0.2893691111275775, -0.08527967308898922, -0.0128432631153215, -0.0007764027982060095],
                [0.0030492698074017885, 0.09054237024457563, 0.4324999309408109, 0.7822670756257748, 0.6943873304827003, 0.33528336694894784, 0.08992288982220424, 0.01259574743185783, 0.0007189235092122453],
                [-0.023196614528543533, -0.25673636189951354, -0.8201718290465319, -1.1728488283712706, -0.8896511129318162, -0.3834861189828163, -0.09436955644260027, -0.012354549841562189, -0.0006677471628016385],
            ],
            [
                [-0.0380952380952381, -0.021164021164021163, -0.0028218694885361554],
                [0.0390625, 0.0640625, 0.024131944444444445, 0.0026813271604938273],
                [-0.05111607142857143, -0.05521660052910053, -0.017920524691358023, -0.0018343988830099942],
                [0.01318359375, 0.055648561507936505, 0.04496355544532628, 0.012864904835390947, 0.0012249419765334116],
                [-0.026177370806277057, -0.05598841841029341, -0.03618314218574635, -0.009236407561060339, -0.0008202830019155893],
                [0.004606119791666667, 0.036727472830988456, 0.053023379253848, 0.028705166120530704, 0.006621590442843337, 0.000550810429778301],
                [-0.011873656942016317, -0.043136190222778766, -0.04741886481252801, -0.022241628639921462, -0.004696939731487807, -0.00036811913975354904],
                [0.0014046805245535715, 0.018670726401264905, 0.043906005131556194, 0.03958266891276254, 0.01656147581827688, 0.0032482226288612585, 0.00024181000267343918],
                [-0.003455020050479035, -0.020875498109384827, -0.037776395837837894, -0.029794679716663223, -0.011495218320530512, -0.0021353857395999616, -0.00015290875635676917],
                [5.0067397021742725e-05, 0.002464369307899872, 0.013453768168989272, 0.02363904641697872, 0.01825948503984271, 0.006919191004326127, 0.0012653060406225493, 8.937814600236314e-05],
                [0.0017919770983825585, 0.008850153382518513, 0.009413280256974306, -0.0004956639190885073, -0.005133526652264041, -0.002741872894432045, -0.0005745906088619407, -4.341298959433666e-05],
            ],
            [
                [-0.0761904761904762, -0.06772486772486773, -0.019753086419753086, -0.0018812463256907702],
                [0.01025390625, 0.019921875, 0.011697048611111111, 0.00275366512345679, 0.00022947209362139917],
                [0.031214065882034632, 0.04215206379268879, 0.019650767346079845, 0.003887840875688098, 0.000279489246583511],
                [-0.013671875, -0.0786265275523088, -0.08316296647025814, -0.034945083690743416, -0.006531212700194182, -0.0004540467964518462],
                [0.05188272014443889, 0.13709426062941688, 0.11498316353785104, 0.04291781858188108, 0.0074552732700398595, 0.0004931117251344017],
                [-0.011671956380208333, -0.11561876030870172, -0.2019907311203698, -0.14043508195363663, -0.04696171977516921, -0.007583634727708507, -0.00047610323210985056],
                [0.04942269441170808, 0.20859156254745448, 0.2717000450931929, 0.16126809901748415, 0.04876272149157905, 0.007340030011092549, 0.0004374506713521628],
                [-0.008381434849330358, -0.1272232156077894, -0.33218708683313525, -0.3448101291219878, -0.1785409189762017, -0.04922518546254084, -0.00692965441915654, -0.00039247649681966803],
                [0.04237317182425035, 0.26044529561526913, 0.4871114712931715, 0.42018096818846834, 0.1929489331769547, 0.04886415546045875, 0.00645505144297377, 0.0003479430747485604],
            ],
            [
                [0.06926406926406926, 0.09758537758537758, 0.0470995670995671, 0.00961145850034739, 0.0007119598889146215],
                [-0.0487060546875, -0.18753836495535714, -0.17229740203373015, -0.06662421117173721, -0.011760588325801, -0.0007840392217200666],
                [0.10254903787618631, 0.22476831057397464, 0.16853913845124782, 0.058149084109370565, 0.009516769935626834, 0.000600532699787083],
                [-0.016596476236979168, -0.1319916879182016, -0.20854828496246952, -0.13525069794411157, -0.04287561940518303, -0.00663220891229983, -0.00040181897844927296],
                [0.029433956139888758, 0.12164570089740988, 0.1566359015559455, 0.09181770238546497, 0.027401940305463064, 0.0040717788079110175, 0.00023971832182486806],
            ],
            [
                [0.17262737262737263, 0.3245110445110445, 0.2222133422133422, 0.071882027437583, 0.011193147654053003, 0.0006783725850941215],
                [-0.0132598876953125, -0.05428205217633929, -0.0625133792937748, -0.03306377538718033, -0.009060266350101962, -0.0012551064765058545, -6.972813758365857e-05],
            ],
        ];

        // 大きな a, b での一様漸近展開（Temme、DLMF 8.18.9–8.18.12）による Iₓ(a, b) と補関数（中心付近では c₁, …, c₅ も加算）
        private static bool TryIncompleteBetaAsymptotic(double a, double b, double x, out double i, out double complement)
        {
            if (!TryIncompleteBetaAsymptoticTailFactor(a, b, x, out double factor, out double exponent, out bool lower))
            {
                i = complement = 0;
                return false;
            }

            double exponential = Math.Exp(exponent);
            if (lower)
            {
                i = exponential * factor;
                complement = 1 - i;
            }
            else
            {
                complement = exponential * factor;
                i = 1 - complement;
            }

            return true;
        }

        // 一様漸近展開の小さい方の裾を e^exponent · factor（exponent = -μη²/2）に分離して計算する（対数版でも再利用）。
        // lower = true なら小さい方の裾が Iₓ、false なら補関数。適用条件外なら false を返す
        private static bool TryIncompleteBetaAsymptoticTailFactor(double a, double b, double x, out double factor, out double exponent,
            out bool lower)
        {
            // x₀ = a/(a+b)、1 - x₀ = b/(a+b)。a + b はオーバーフローしうるので比で求める
            double x0 = 1 / (1 + b / a);
            double y0 = 1 / (1 + a / b);

            // d = x - x₀ を丸めた x₀ から引くと、x₀ の丸め誤差（ε x₀ 程度）が d に入る。t = |η|√(μ/2) ≈ |d|/σ
            // （σ = √(x₀(1-x₀)/μ) は分布の幅）なので、t の誤差は ε x₀/σ に拡大する。
            // そこで x(a+b) - a = xb - ya を丸め誤差込みで求め、u = d/x₀ = (xb - ya)/a、v = -d/(1-x₀) = -(xb - ya)/b とする
            double deviation = IncompleteBetaCenterDeviation(a, b, x, 1 - x);
            double u = deviation / a;
            double v = -deviation / b;
            double d = u * x0;
            double s = Math.Sqrt(x0) * Math.Sqrt(y0);

            // τ = (x - x₀)/√(x₀(1-x₀))、r = (1 - 2x₀)/√(x₀(1-x₀)) とすると、
            // 1/η = 1/τ + r/3 - (r² + 3)τ/12 + r(23r² + 81)τ²/540 + O(τ³) なので、x ≈ x₀ では打ち消しを避けて級数で求める。
            // パラメータの比が大きいと r ≈ √(max/min) は 1e140 にもなり r³ がオーバーフローするので、w = rτ（|w| < 1e-5）でまとめる
            double tau = d / s;
            double r = (y0 - x0) / s;
            double w = r * tau;

            // η²/2 = -(x₀ (ln(1 + u) - u) + (1-x₀)(ln(1 + v) - v))（1 次の項は打ち消し合う）
            double halfEta2 = -(x0 * Log1pmx(u) + y0 * Log1pmx(v));
            double eta = Math.CopySign(Math.Sqrt(2 * halfEta2), d);

            // erfc(t)/2 = e^(-t²) erfcx(t)/2（t = |η|√(μ/2)、t² = μη²/2）として、小さい方の裾を e^(-μη²/2) でくくる
            double halfMu = 0.5 * a + 0.5 * b;
            double t = Math.Abs(eta) * Math.Sqrt(halfMu);
            bool series = Math.Abs(tau) <= IncompleteBetaSeriesRadius && Math.Abs(w) <= IncompleteBetaSeriesRadius
                && Math.Abs(r) <= IncompleteBetaSeriesMaxR && t <= IncompleteBetaExpansionMaxT;
            if (!series && Math.Min(a, b) < IncompleteBetaAsymptoticMin)
            {
                factor = exponent = 0;
                lower = false;
                return false;
            }

            double c0 = (1 + Math.Abs(r)) * Math.Abs(tau) < 1e-5
                ? r * (1.0 / 3 - w / 12 + 23 * w * w / 540) - tau / 4 + 0.15 * w * tau
                : 1 / eta - 1 / tau;

            exponent = -(a * halfEta2 + b * halfEta2);
            double rho = Math.Exp(StirlingCorrection(a + b) - StirlingCorrection(a) - StirlingCorrection(b));
            double sum = series ? IncompleteBetaTemmeSeries(tau, r, 0.5 / halfMu) : c0;
            double remainder = rho * sum / Math.Sqrt(4 * Math.PI) / Math.Sqrt(halfMu);
            lower = eta <= 0;
            factor = lower ? 0.5 * Erfcx(t) + remainder : 0.5 * Erfcx(t) - remainder;
            return true;
        }

        // 一様漸近展開の係数和 Σₖ c_k(τ) μ^(-k)（k = 0, …, 5、invMu = 1/μ）
        private static double IncompleteBetaTemmeSeries(double tau, double r, double invMu)
        {
            double r2 = r * r;
            double total = 0;
            for (int k = IncompleteBetaTemmeCoefficients.Length - 1; k >= 0; k--)
            {
                double[][] rows = IncompleteBetaTemmeCoefficients[k];
                double ck = 0;
                for (int j = rows.Length - 1; j >= 0; j--)
                {
                    double[] q = rows[j];
                    double pj = 0;
                    for (int m = q.Length - 1; m >= 0; m--)
                    {
                        pj = pj * r2 + q[m];
                    }

                    ck = ck * tau + (j % 2 == 0 ? r * pj : pj);
                }

                total = total * invMu + ck;
            }

            return total;
        }

        // Iₓ(a, b) = p の x → 0 側主要項 x = (p a B(a, b))^(1/a)
        private static double InverseIncompleteBetaTail(double a, double b, double p)
        {
            double scaled = p * a * Beta(a, b);
            return scaled >= MinNormal && double.IsFinite(scaled)
                ? Math.Pow(scaled, 1 / a)
                : Math.Exp((Math.Log(p) + Math.Log(a) + LogBeta(a, b)) / a);
        }

        // ln(z(1 + r)) を |ln z| と ln(1 + r) の打ち消しを避けて計算する
        private static double LogScaledCoordinate(double z, double lnz, double r)
        {
            double scaled = z * (1 + r);
            return scaled >= MinNormal && scaled <= double.MaxValue ? Math.Log(scaled) : lnz + Log1p(r);
        }

        // 前因子 xᵃ yᵇ / B(a, b) をオーバーフロー・桁落ちを避けて計算する（y = 1 - x。x < 1/2 では y の丸め誤差拡大を避け x から対数を求める）
        internal static double BetaPrefix(double a, double b, double x, double y)
        {
            double lny = x < 0.5 ? Log1p(-x) : Math.Log(y);
            if (a >= 10 && b >= 10)
            {
                // x₀ = a/(a+b) からのずれで表すと、指数部の大きな項が解析的に打ち消し合う。
                // a + b や ab はオーバーフローしうるので、u = x(a+b)/a - 1、v = y(a+b)/b - 1 と √(ab/(a+b)) を比で求める
                double deviation = IncompleteBetaCenterDeviation(a, b, x, y);
                double u = deviation / a;
                double v = -deviation / b;
                double lu = Math.Abs(u) < 0.5 ? Log1pmx(u) : LogScaledCoordinate(x, Math.Log(x), b / a) - u;
                double lv = Math.Abs(v) < 0.5 ? Log1pmx(v) : (x < 0.5 ? lny + Log1p(a / b) : LogScaledCoordinate(y, lny, a / b)) - v;
                double exponent = a * lu + b * lv
                    + StirlingCorrection(a + b) - StirlingCorrection(a) - StirlingCorrection(b);
                return Math.Sqrt(a / (1 + a / b) / (2 * Math.PI)) * Math.Exp(exponent);
            }

            // b < 10 では y の丸め誤差の拡大は 5ε 以下、x ≥ 1/2 では y は正確なので、累乗をそのまま使う
            double xa = Math.Pow(x, a);
            double yb = b >= 10 && x < 0.5 ? Math.Exp(b * lny) : Math.Pow(y, b);
            double beta = Beta(a, b);
            double direct = xa * yb / beta;
            if (xa > 1e-290 && yb > 1e-290 && beta > 1e-290 && double.IsFinite(beta)
                && double.IsFinite(direct) && direct > 1e-290)
            {
                return direct;
            }

            // 片方だけ大きいなら、xᵃ yᵇ / B(a, b) = (xb)ᵃ yᵇ / Γ(a) · Γ(b + a)/(Γ(b) bᵃ)（a ↔ b、x ↔ y も同様）と並べ替えると、
            // 裾で xᵃ や B(a, b) が範囲外でも各因子は正規化数に収まることが多い（I ≈ 1e-300 の裾）。
            // 対数を経由すると指数部 |L| の大きさに比例した誤差 |L|ε が入るので、先にこちらを試す
            if (a < 10 && b >= 10)
            {
                double scaledPower = Math.Pow(x * b, a);
                double scaled = scaledPower * yb * ReciprocalGamma(a) * Math.Exp(LogScaledGammaShiftRatio(b, a));
                if (scaledPower >= MinNormal && yb >= MinNormal && double.IsFinite(scaled) && scaled >= MinNormal)
                {
                    return scaled;
                }
            }
            else if (b < 10 && a >= 10)
            {
                double scaledPower = Math.Pow(y * a, b);
                double scaled = scaledPower * xa * ReciprocalGamma(b) * Math.Exp(LogScaledGammaShiftRatio(a, b));
                if (scaledPower >= MinNormal && xa >= MinNormal && double.IsFinite(scaled) && scaled >= MinNormal)
                {
                    return scaled;
                }
            }

            // それでも範囲外なら対数で求める（a ln x と ln B(a, b) の大きな項の打ち消しは LogBetaPrefix が解析的に除く）
            return Math.Exp(LogBetaPrefix(a, b, x, y));
        }

        // 中心 x₀ = a/(a+b) からのずれ x(a+b) - a = xb - ya を FMA と Sterbenz の補題（yLo = (1 - y) - x）で丸め誤差込みに計算する
        private static double IncompleteBetaCenterDeviation(double a, double b, double x, double y)
        {
            double yLo = (1 - y) - x;
            double xb = x * b;
            double xbLo = Math.FusedMultiplyAdd(x, b, -xb);
            double ya = y * a;
            double yaLo = Math.FusedMultiplyAdd(y, a, -ya) + yLo * a;
            return (xb - ya) + (xbLo - yaLo);
        }

        // 不完全ベータ関数の連分数（DLMF 8.17.22、Lentz 法）。Iₓ(a, b) = xᵃ yᵇ/(a B(a, b)) · 戻り値（a ≤ 1, x < (a+1)/(a+b+2) で使用）
        private static double BetaFraction(double a, double b, double x)
        {
            double qab = a + b;
            double qap = a + 1;
            double qam = a - 1;
            double c = 1;
            double d = 1 - qab * x / qap;
            if (Math.Abs(d) < Tiny)
            {
                d = Tiny;
            }

            d = 1 / d;
            double h = d;
            for (int iteration = 1; iteration < MaxIterations; iteration++)
            {
                double m = iteration;
                double m2 = 2 * m;

                // 偶数番目の項
                double aa = m / (qam + m2) * ((b - m) * x) / (a + m2);
                d = 1 + aa * d;
                if (Math.Abs(d) < Tiny)
                {
                    d = Tiny;
                }

                c = 1 + aa / c;
                if (Math.Abs(c) < Tiny)
                {
                    c = Tiny;
                }

                d = 1 / d;
                h *= d * c;

                // 奇数番目の項
                aa = -(a + m) / (a + m2) * ((qab + m) * x) / (qap + m2);
                d = 1 + aa * d;
                if (Math.Abs(d) < Tiny)
                {
                    d = Tiny;
                }

                c = 1 + aa / c;
                if (Math.Abs(c) < Tiny)
                {
                    c = Tiny;
                }

                d = 1 / d;
                double delta = d * c;
                h *= delta;
                if (Math.Abs(delta - 1) <= MachineEpsilon)
                {
                    break;
                }
            }

            return h;
        }

        // DiDonato–Morris の連分数（ACM TOMS Algorithm 708 の BFRAC）。
        // Iₓ(a, b) = xᵃ yᵇ/B(a, b) · 戻り値（λ = a - (a+b)x ≥ 0, max(a, b) > 1）
        private static double DiDonatoMorrisBetaFraction(double a, double b, double x, double y, double lambda)
        {
            double c = 1 + lambda;
            double c1 = 1 + 1 / a;
            double yp1 = y + 1;
            double p = 1;
            double s = a + 1;

            // 初期値 (aₙ, bₙ, aₙ₊₁, bₙ₊₁) = (0, 1, 1, c/c₁) を bₙ₊₁ で割っておく（λ が大きいと c/c₁ と β の積がオーバーフローする）
            double r = c1 / c;
            double an = 0;
            double bn = r;
            double anp1 = r;
            double bnp1 = 1;
            for (int iteration = 1; iteration < MaxIterations; iteration++)
            {
                double n = iteration;
                double t = n / a;
                double e = a / s;

                // α = p(p + b/a)e² · n(b - n)x²（e = a/s）。b が大きいと n(b - n) や (p + b/a)(w x) が途中でオーバーフローするので、
                // λ ≥ 0（(a+b)x ≤ a）で有界な (ap + b)x/s と (b - n)x の積として求める
                double bx = (b - n) * x;
                double alpha = p * e * n * ((a * p + b) * x / s) * bx;
                e = (1 + t) / (c1 + t + t);
                double beta = n + n * bx / s + e * (c + n * yp1);
                p = 1 + t;
                s += 2;

                t = alpha * an + beta * anp1;
                an = anp1;
                anp1 = t;
                t = alpha * bn + beta * bnp1;
                bn = bnp1;
                bnp1 = t;
                double previous = r;
                r = anp1 / bnp1;
                if (!(Math.Abs(r - previous) > MachineEpsilon * r))
                {
                    break;
                }

                // 次の反復のために正規化する
                an /= bnp1;
                bn /= bnp1;
                anp1 = r;
                bnp1 = 1;
            }

            return r;
        }

        // Iₓ(a, b) = p（1 - Iₓ(a, b) = q）の根を初期値推定 + ハレー法で求める
        private static double InverseIncompleteBetaCore(double a, double b, double p, double q)
        {
            if (double.IsNaN(a) || double.IsNaN(b) || double.IsNaN(p) || double.IsNaN(q)
                || a <= 0 || b <= 0 || double.IsInfinity(a) || double.IsInfinity(b) || p < 0 || q < 0)
            {
                return double.NaN;
            }

            if (p == 0)
            {
                return 0;
            }

            if (q == 0)
            {
                return 1;
            }

            bool useQ = q < p;

            // 初期値（Abramowitz & Stegun 26.5.22、Numerical Recipes）
            double x;
            bool skewed = b >= InverseBetaSkewRatio * Math.Max(a, 1) || a >= InverseBetaSkewRatio * Math.Max(b, 1);
            if (skewed)
            {
                x = InverseIncompleteBetaSkewedInitial(a, b, p, q, useQ);

                // a ≫ b で 1 - x が 2⁻⁵⁴ 未満なら、根に最も近い倍精度数は 1（区間 (0, 1) の中の点では関数値がほぼ一定で、残差では判別できない）
                if (x == 1)
                {
                    return 1;
                }
            }
            else if (a >= 1 && b >= 1)
            {
                // 正規分布の上側 p 分位点
                double z = useQ ? -Sqrt2 * InverseErfc(2 * q) : Sqrt2 * InverseErfc(2 * p);
                double al = (z * z - 3) / 6;
                double h = 2 / (1 / (2 * a - 1) + 1 / (2 * b - 1));
                double w = z * Math.Sqrt(al + h) / h
                    - (1 / (2 * b - 1) - 1 / (2 * a - 1)) * (al + 5.0 / 6 - 2 / (3 * h));
                x = a / (a + b * Math.Exp(2 * w));
            }
            else
            {
                double lna = Math.Log(a / (a + b));
                double lnb = Math.Log(b / (a + b));
                double t = Math.Exp(a * lna) / a;
                double u = Math.Exp(b * lnb) / b;
                double w = t + u;
                x = p < t / w ? Math.Pow(a * w * p, 1 / a) : 1 - Math.Pow(b * w * q, 1 / b);
            }

            // 裾では上の近似が大きく外れ、反復の上限までに収束しないことがある。
            // 端点の近くでは Iₓ(a, b) = xᵃ/(a B(a, b)) (1 + O(bx)) なので、その主要項を初期値にする（上側は a, b を入れ替える）
            if (!useQ)
            {
                double tail = InverseIncompleteBetaTail(a, b, p);
                double skewedInitial = x;
                if (tail * (1 + b) < 0.01 || (!(x >= MinNormal) && !skewed))
                {
                    x = tail;
                }

                // 非正規化数以下では反復で改善する余地がないため、近似のうち誤差が倍精度で無視できる方をそのまま返す。
                // 主要項の相対誤差は O(bx) で、bx ≤ ε でなければ b ≳ ε/MinNormal ≈ 1e292 なので、ガンマ分布による初期値の誤差 O(1/b) が無視できる
                if (!(x >= MinNormal))
                {
                    return skewed && tail * (1 + b) > MachineEpsilon ? skewedInitial : x;
                }
            }
            else
            {
                double tail = InverseIncompleteBetaTail(b, a, q);
                if (tail * (1 + a) < 0.01)
                {
                    x = 1 - tail;

                    // 1 - x が 2⁻⁵⁴ 未満なら、最も近い倍精度数は 1
                    if (x == 1)
                    {
                        return 1;
                    }
                }
            }

            return InverseIncompleteBetaRefine(a, b, p, q, useQ, x);
        }

        // 初期値をガンマ分布の分位点から求めるパラメータ比の閾値（b ≫ a や a ≫ b の非対称分布用）
        private const double InverseBetaSkewRatio = 10;

        // b ≫ a（または a ≫ b）での初期値。-(b + (a-1)/2) ln(1 - X) ≈ Γ(a)（DiDonato–Morris の BGRAT 変数）の分位点から変換する
        private static double InverseIncompleteBetaSkewedInitial(double a, double b, double p, double q, bool useQ)
        {
            if (b >= a)
            {
                double u = useQ ? InverseRegularizedGammaQ(a, q) : InverseRegularizedGammaP(a, p);
                return -Expm1(-u / (b + 0.5 * (a - 1)));
            }

            double v = useQ ? InverseRegularizedGammaP(b, q) : InverseRegularizedGammaQ(b, p);
            return 1 + Expm1(-v / (a + 0.5 * (b - 1)));
        }

        // 区間を保つハレー法（収束不十分・区間外時は二分法へフォールバック）で Iₓ(a, b) = p（useQ なら 1 - Iₓ(a, b) = q）を解く
        private static double InverseIncompleteBetaRefine(double a, double b, double p, double q, bool useQ, double x)
        {
            double lower = 0;
            double upper = 1;
            double best = double.NaN;
            double bestError = double.PositiveInfinity;
            double previousStep = 1;
            double olderStep = 1;
            bool finalCheck = false;
            for (int iteration = 0; iteration < 400; iteration++)
            {
                if (!(x > lower && x < upper))
                {
                    x = InverseIncompleteBetaBisect(lower, upper);
                    if (!(x > lower && x < upper))
                    {
                        // 区間の両端が隣接する倍精度数になった
                        break;
                    }
                }

                double y = 1 - x;
                IncompleteBetaCore(a, b, x, y, out double ix, out double complement, out double prefix);
                double error = useQ ? q - complement : ix - p;
                if (Math.Abs(error) < bestError)
                {
                    best = x;
                    bestError = Math.Abs(error);
                }

                if (error == 0 || finalCheck)
                {
                    break;
                }

                if (error > 0)
                {
                    upper = x;
                }
                else
                {
                    lower = x;
                }

                // dI/dx = x^(a-1) y^(b-1) / B(a, b)、(d²I/dx²)/(dI/dx) = (a-1)/x - (b-1)/y
                double derivative = (double.IsNaN(prefix) ? BetaPrefix(a, b, x, y) : prefix) / (x * y);
                double next = double.NaN;
                double newton = double.NaN;
                if (derivative > 0 && double.IsFinite(derivative))
                {
                    newton = error / derivative;
                    next = x - newton / (1 - 0.5 * Math.Min(1, newton * ((a - 1) / x - (b - 1) / y)));
                }

                // 収束の判定は区間の判定より先に行う（x は直前に区間の端になっているので、収束すると next が端に一致する）。
                // 最後の更新値が区間内なら評価して残差の小さい方を返し、区間外（評価済みの端より外）なら評価済みの点を返す
                if (Math.Abs(next - x) <= 4 * MachineEpsilon * x)
                {
                    if (!(next > lower && next < upper))
                    {
                        break;
                    }

                    finalCheck = true;
                    x = next;
                    continue;
                }

                if (!(next > lower && next < upper) || !(Math.Abs(newton) <= 0.5 * olderStep))
                {
                    next = InverseIncompleteBetaBisect(lower, upper);
                }

                olderStep = previousStep;
                previousStep = Math.Abs(next - x);
                x = next;
            }

            return best;
        }

        // 区間 [lower, upper] の二分点（複数桁にまたがる場合は 0 または 1 に近い側を対数スケールで二分する）
        private static double InverseIncompleteBetaBisect(double lower, double upper)
        {
            double mid = 0.5 * (lower + upper);
            if (upper <= 0.5)
            {
                double low = lower > 0 ? lower : Math.ScaleB(upper, -64);
                if (upper > 16 * low)
                {
                    mid = Math.Sqrt(low) * Math.Sqrt(upper);
                }
            }
            else if (lower >= 0.5)
            {
                double yHigh = 1 - lower;
                double yLow = upper < 1 ? 1 - upper : Math.ScaleB(yHigh, -64);
                if (yHigh > 16 * yLow)
                {
                    mid = 1 - Math.Sqrt(yLow) * Math.Sqrt(yHigh);
                }
            }

            return mid > lower && mid < upper ? mid : 0.5 * (lower + upper);
        }
    }
}
