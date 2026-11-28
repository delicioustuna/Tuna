using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        /// <summary>
        /// Computes the natural logarithm of the regularized incomplete beta function <c>ln Iₓ(a, b)</c>.
        /// <para>正則化不完全ベータ関数の自然対数 <c>ln Iₓ(a, b)</c> を計算します。</para>
        /// </summary>
        /// <param name="a">The first shape parameter (<c>a &gt; 0</c>). / 第 1 形状パラメータ（<c>a &gt; 0</c>）。</param>
        /// <param name="b">The second shape parameter (<c>b &gt; 0</c>). / 第 2 形状パラメータ（<c>b &gt; 0</c>）。</param>
        /// <param name="x">The upper integration limit (<c>0 ≤ x ≤ 1</c>). / 積分区間の上端（<c>0 ≤ x ≤ 1</c>）。</param>
        /// <returns>The value of <c>ln Iₓ(a, b)</c>, or <see cref="double.NaN"/> if any argument is out of domain or <see cref="double.NaN"/>. / <c>ln Iₓ(a, b)</c> の値。いずれかの引数が定義域外または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <see cref="double.NegativeInfinity"/> at <c>x = 0</c> and <c>0</c> at <c>x = 1</c>. Remains finite even in tail regions where <c>Iₓ(a, b)</c> underflows to zero.</para>
        /// <para><c>x = 0</c> では <see cref="double.NegativeInfinity"/>、<c>x = 1</c> では <c>0</c> を返します。<c>Iₓ(a, b)</c> が 0 にアンダーフローする裾の領域でも有限の値を保ちます。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/8.17.E2"/>
        public static double LogRegularizedIncompleteBeta(double a, double b, double x)
        {
            if (TryIncompleteBetaSpecialCase(a, b, x, out double i, out _))
            {
                return Math.Log(i);
            }

            LogIncompleteBetaCore(a, b, x, 1 - x, out double logI, out _);
            return logI;
        }

        /// <summary>
        /// Computes the natural logarithm of the complement of the regularized incomplete beta function <c>ln(1 - Iₓ(a, b))</c>.
        /// <para>正則化不完全ベータ関数の補関数の自然対数 <c>ln(1 - Iₓ(a, b))</c> を計算します。</para>
        /// </summary>
        /// <param name="a">The first shape parameter (<c>a &gt; 0</c>). / 第 1 形状パラメータ（<c>a &gt; 0</c>）。</param>
        /// <param name="b">The second shape parameter (<c>b &gt; 0</c>). / 第 2 形状パラメータ（<c>b &gt; 0</c>）。</param>
        /// <param name="x">The integration limit (<c>0 ≤ x ≤ 1</c>). / 積分区間の端点（<c>0 ≤ x ≤ 1</c>）。</param>
        /// <returns>The value of <c>ln(1 - Iₓ(a, b)) = ln I₁₋ₓ(b, a)</c>, or <see cref="double.NaN"/> if any argument is out of domain or <see cref="double.NaN"/>. / <c>ln(1 - Iₓ(a, b)) = ln I₁₋ₓ(b, a)</c> の値。いずれかの引数が定義域外または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <c>0</c> at <c>x = 0</c> and <see cref="double.NegativeInfinity"/> at <c>x = 1</c>. Remains finite even in tail regions where <c>1 - Iₓ(a, b)</c> underflows to zero.</para>
        /// <para><c>x = 0</c> では <c>0</c>、<c>x = 1</c> では <see cref="double.NegativeInfinity"/> を返します。<c>1 - Iₓ(a, b)</c> が 0 にアンダーフローする裾の領域でも有限の値を保ちます。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/8.17.E4"/>
        public static double LogRegularizedIncompleteBetaComplement(double a, double b, double x)
        {
            if (TryIncompleteBetaSpecialCase(a, b, x, out _, out double complement))
            {
                return Math.Log(complement);
            }

            LogIncompleteBetaCore(a, b, x, 1 - x, out _, out double logComplement);
            return logComplement;
        }

        // ln Iₓ(a, b) と ln(1 - Iₓ(a, b))（y = 1 - x）。
        // IncompleteBetaCore と同じ分岐で直接計算する方を対数で求め、もう一方は ln(1 - e^(直接計算した方)) とする
        private static void LogIncompleteBetaCore(double a, double b, double x, double y, out double logI, out double logComplement)
        {
            if (Math.Min(a, b) >= IncompleteBetaExpansionMin
                && TryIncompleteBetaAsymptoticTailFactor(a, b, x, out double factor, out double exponent, out bool lower))
            {
                AssignLogTails(exponent + Math.Log(factor), upper: !lower, out logI, out logComplement);
                return;
            }

            double logPrefix = LogBetaPrefix(a, b, x, y);
            double fraction = IncompleteBetaTailFraction(a, b, x, y, Math.Exp(logPrefix), log: true, out bool upper, out bool direct);
            AssignLogTails(direct ? fraction : logPrefix + Math.Log(fraction), upper, out logI, out logComplement);
        }

        internal static double LogBetaTailPrefixRatio(double a, double b, double x, double logTail, bool upper)
        {
            double y = 1 - x;
            double logPrefix = LogBetaPrefix(a, b, x, y);
            if (logTail < -16)
            {
                // 遠い裾では対数の差から共通の指数項を除く。
                if (Math.Min(a, b) >= IncompleteBetaExpansionMin
                    && TryIncompleteBetaAsymptoticTailFactor(a, b, x, out double factor, out _, out bool lower))
                {
                    if (upper != lower)
                    {
                        return 0.5 * Math.Log(a / (1 + a / b) / (2 * Math.PI))
                            + StirlingCorrection(a + b) - StirlingCorrection(a) - StirlingCorrection(b) - Math.Log(factor);
                    }
                }
                else
                {
                    double fraction = IncompleteBetaTailFraction(a, b, x, y, Math.Exp(logPrefix), log: true,
                        out bool directUpper, out bool direct);
                    if (upper == directUpper && !direct)
                    {
                        return -Math.Log(fraction);
                    }
                }
            }

            return logPrefix - logTail;
        }

        internal static (double Mantissa, long Exponent) ScaledBetaTailPrefixRatio(
            double a, double b, double x, double xError, double logTail, bool upper)
        {
            double y = 1 - x;
            double logPrefix = LogBetaPrefix(a, b, x, y);
            double fraction;
            bool directUpper;
            if (Math.Min(a, b) >= IncompleteBetaExpansionMin
                && TryIncompleteBetaAsymptoticTailFactor(a, b, x, out double factor, out _, out bool lower))
            {
                double normalization = Math.Sqrt(a / (1 + a / b) / (2 * Math.PI)) *
                    Math.Exp(StirlingCorrection(a + b) - StirlingCorrection(a) - StirlingCorrection(b));
                directUpper = !lower;
                fraction = factor / normalization;
            }
            else
            {
                fraction = IncompleteBetaTailFraction(a, b, x, y, Math.Exp(logPrefix), log: true, out directUpper, out bool direct);
                if (direct)
                {
                    fraction = double.NaN;
                }
            }

            if (upper == directUpper && fraction > 0 && double.IsFinite(fraction))
            {
                return ScaledReciprocal(fraction);
            }

            var prefix = (Hi: logPrefix, Lo: 0.0);
            if (Math.Max(a, b) <= 1e12)
            {
                var logX = DoubleDoubleLog((x, xError));
                var complement = DoubleDoubleAdd(DoubleDoubleTwoSum(1, -x), (-xError, 0));
                prefix = PreciseBetaLogPrefix(a, b, logX, DoubleDoubleLog(complement));
            }

            return ScaledTailPrefixRatio(prefix, logTail, upper == directUpper ? double.NaN : fraction);
        }

        internal static (double Mantissa, long Exponent) ScaledBetaTailPrefixRatioFromLog(
            double a, double b, (double Hi, double Lo) logX, double logTail)
        {
            var prefix = Math.Max(a, b) <= 1e12
                ? PreciseBetaLogPrefix(a, b, logX, (0, 0))
                : (Math.FusedMultiplyAdd(a, logX.Hi, -LogBeta(a, b)), a * logX.Lo);
            return ScaledTailPrefixRatio(prefix, logTail, double.NaN);
        }

        private static (double Hi, double Lo) PreciseBetaLogPrefix(
            double a, double b, (double Hi, double Lo) logX, (double Hi, double Lo) logY)
        {
            var total = PrecisePositiveLogGamma(DoubleDoubleTwoSum(a, b));
            var first = PrecisePositiveLogGamma((a, 0));
            var second = PrecisePositiveLogGamma((b, 0));
            var normalization = DoubleDoubleAdd(DoubleDoubleAdd(total, (-first.Hi, -first.Lo)), (-second.Hi, -second.Lo));
            var power = DoubleDoubleAdd(DoubleDoubleMultiply((a, 0), logX), DoubleDoubleMultiply((b, 0), logY));
            return DoubleDoubleAdd(normalization, power);
        }

        // ln(xᵃ yᵇ / B(a, b))（y は 1 - x を倍精度に丸めた値）。
        // BetaPrefix と同様に大きな項を解析的に打ち消して対数のまま評価し、x < 1/2 での y の丸め誤差を x から補正する
        internal static double LogBetaPrefix(double a, double b, double x, double y)
        {
            // y = 1 - x の丸め誤差 yLo = (1 - x) - y = (1 - y) - x（Sterbenz の補題により 2 回の減算は正確）。x ≥ 1/2 では 0
            double yLo = (1 - y) - x;
            double lnx = Math.Log(x);
            double lny = x < 0.5 ? Log1p(-x) : Math.Log(y);

            if (a >= 10 && b >= 10)
            {
                double deviation = IncompleteBetaCenterDeviation(a, b, x, y);
                double u = deviation / a;
                double v = -deviation / b;
                double lu = Math.Abs(u) < 0.5 ? Log1pmx(u) : LogScaledCoordinate(x, lnx, b / a) - u;
                double lv = Math.Abs(v) < 0.5 ? Log1pmx(v) : (x < 0.5 ? lny + Log1p(a / b) : LogScaledCoordinate(y, lny, a / b)) - v;
                return 0.5 * Math.Log(a / (1 + a / b) / (2 * Math.PI)) + a * lu + b * lv
                    + StirlingCorrection(a + b) - StirlingCorrection(a) - StirlingCorrection(b);
            }

            if (b >= 10)
            {
                // xᵃ yᵇ / B(a, b) = (xb)ᵃ yᵇ / Γ(a) · Γ(b + a)/(Γ(b) bᵃ)。最頻値の近くでは xb ≈ a で、各項が O(a ln a) に収まる
                double xb = x * b;
                double lnxb = xb >= MinNormal ? Math.Log(xb) : lnx + Math.Log(b);
                return a * lnxb + b * lny - LogGamma(a) + LogScaledGammaShiftRatio(b, a);
            }

            if (a >= 10)
            {
                // a と b、x と y の役割を入れ替えた式。y の丸め誤差は ln(ya) に yLo/y として加える
                double ya = y * a;
                double lnya = ya >= MinNormal ? Math.Log(ya) + yLo / y : lny + Math.Log(a);
                return b * lnya + a * lnx - LogGamma(b) + LogScaledGammaShiftRatio(a, b);
            }

            return a * lnx + b * lny - LogBeta(a, b);
        }

        // ln(Γ(x + d)/(Γ(x) x^d))（x ≥ 10、0 < d < 10）。
        // スターリングの公式で d ln x の大きな項を解析的に打ち消す
        private static double LogScaledGammaShiftRatio(double x, double d)
        {
            double log1pRatio = Log1p(d / x);
            return (x - 0.5) * log1pRatio + d * (log1pRatio - 1) + StirlingCorrection(x + d) - StirlingCorrection(x);
        }
    }
}
