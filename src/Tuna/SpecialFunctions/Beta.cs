using static Tuna.XMath.Constants;
using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        /// <summary>
        /// Computes the beta function <c>B(a, b) = Γ(a) Γ(b) / Γ(a + b)</c>.
        /// <para>ベータ関数 <c>B(a, b) = Γ(a) Γ(b) / Γ(a + b)</c> を計算します。</para>
        /// </summary>
        /// <param name="a">The first parameter. / 第 1 引数。</param>
        /// <param name="b">The second parameter. / 第 2 引数。</param>
        /// <returns>The value of <c>B(a, b)</c>, or <see cref="double.NaN"/> if <paramref name="a"/> or <paramref name="b"/> is a non-positive integer or <see cref="double.NaN"/>. / <c>B(a, b)</c> の値。<paramref name="a"/> または <paramref name="b"/> が 0 以下の整数または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/5.12.E1"/>
        public static double Beta(double a, double b)
        {
            if (double.IsNaN(a) || double.IsNaN(b))
            {
                return double.NaN;
            }

            if (IsNonPositiveInteger(a) || IsNonPositiveInteger(b))
            {
                return double.NaN;
            }

            if (a > 0 && b > 0)
            {
                double large = Math.Max(a, b);
                double small = Math.Min(a, b);
                if (double.IsPositiveInfinity(large))
                {
                    return 0;
                }

                if (small < 170)
                {
                    // Γ(large)/Γ(large + small)。a + b の丸めで small の寄与が失われるのを防ぐため差を保持する。
                    // 比が非正規化数へアンダーフローした場合は精度低下を避けて対数から計算する
                    double ratio = GammaShiftRatio(large, small, true);
                    if (ratio >= MinNormal)
                    {
                        return ratio * Gamma(small);
                    }
                }

                return Math.Exp(LogBeta(a, b));
            }

            double sum = a + b;
            if (IsNonPositiveInteger(sum))
            {
                return 0;
            }

            if (sum > 0)
            {
                // 一方のみ負 (非整数) で和が正なら B(a, b) = Γ(negative) · Γ(positive)/Γ(a + b)。
                // positive が大きいと a + b に丸められて negative の寄与が失われるため差を保持して計算する
                double negative = Math.Min(a, b);
                double positive = Math.Max(a, b);
                if (double.IsPositiveInfinity(positive))
                {
                    // B(a, b) ~ Γ(a) b⁻ᵃ (b → ∞) は a < 0 で発散する
                    return SignGamma(negative) * double.PositiveInfinity;
                }

                double gammaNegative = Gamma(negative);
                double shifted = gammaNegative * GammaShiftRatio(positive, negative, true);
                if (Math.Abs(gammaNegative) >= MinNormal && double.IsFinite(shifted) && Math.Abs(shifted) >= MinNormal)
                {
                    return shifted;
                }

                return SignGamma(negative) * Math.Exp(LogBeta(a, b));
            }

            double value = Gamma(a) * Gamma(b) * ReciprocalGamma(sum);
            if (double.IsFinite(value) && value != 0)
            {
                return value;
            }

            return SignGamma(a) * SignGamma(b) * SignGamma(sum) * Math.Exp(LogGamma(a) + LogGamma(b) - LogGamma(sum));
        }

        /// <summary>
        /// Computes the natural logarithm of the absolute value of the beta function, <c>ln |B(a, b)|</c>.
        /// <para>ベータ関数の絶対値の自然対数 <c>ln |B(a, b)|</c> を計算します。</para>
        /// </summary>
        /// <param name="a">The first parameter. / 第 1 引数。</param>
        /// <param name="b">The second parameter. / 第 2 引数。</param>
        /// <returns>The value of <c>ln |B(a, b)|</c>, or <see cref="double.NaN"/> if <paramref name="a"/> or <paramref name="b"/> is a non-positive integer or <see cref="double.NaN"/>. / <c>ln |B(a, b)|</c> の値。<paramref name="a"/> または <paramref name="b"/> が 0 以下の整数または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/5.12.E1"/>
        public static double LogBeta(double a, double b)
        {
            if (double.IsNaN(a) || double.IsNaN(b))
            {
                return double.NaN;
            }

            if (IsNonPositiveInteger(a) || IsNonPositiveInteger(b))
            {
                return double.NaN;
            }

            if (a > 0 && b > 0)
            {
                double large = Math.Max(a, b);
                double small = Math.Min(a, b);
                if (double.IsPositiveInfinity(large))
                {
                    return double.NegativeInfinity;
                }

                if (small >= 10)
                {
                    // スターリング展開で ln Γ の主要項を解析的に打ち消す。
                    // large + small のオーバーフローを避けるため ln(small/(large + small)) = -ln(1 + large/small) とする
                    double sum = large + small;
                    return LnSqrtTwoPi - 0.5 * Math.Log(small) - small * Log1p(large / small)
                        - (large - 0.5) * Log1p(small / large)
                        + StirlingCorrection(large) + StirlingCorrection(small) - StirlingCorrection(sum);
                }

                if (large >= 10)
                {
                    // ln Γ(large) - ln Γ(large + small) をスターリング展開で計算する
                    double sum = large + small;
                    return LogGamma(small) - (large - 0.5) * Log1p(small / large) - small * Math.Log(sum) + small
                        + StirlingCorrection(large) - StirlingCorrection(sum);
                }

                return LogGamma(a) + LogGamma(b) - LogGamma(a + b);
            }

            double s = a + b;
            if (IsNonPositiveInteger(s))
            {
                return double.NegativeInfinity;
            }

            if (s > 0)
            {
                // 一方のみ負 (非整数) で和が正なら ln|Γ(negative)| - ln(Γ(a + b)/Γ(positive))。
                // positive が大きいと巨大項に negative の寄与が吸収されるため差を保持して計算する
                double negative = Math.Min(a, b);
                double positive = Math.Max(a, b);
                return double.IsPositiveInfinity(positive)
                    ? double.PositiveInfinity
                    : LogGamma(negative) - LogGammaShiftRatio(positive, negative);
            }

            return LogGamma(a) + LogGamma(b) - LogGamma(s);
        }

        /// <summary>
        /// Computes the natural logarithm of the multivariate beta function, <c>ln B(α) = Σᵢ ln Γ(αᵢ) - ln Γ(Σᵢ αᵢ)</c>.
        /// <para>多変量ベータ関数の自然対数 <c>ln B(α) = Σᵢ ln Γ(αᵢ) - ln Γ(Σᵢ αᵢ)</c>（ディリクレ分布の正規化定数）を計算します。</para>
        /// </summary>
        /// <param name="alpha">The positive concentration parameters (may include <see cref="double.PositiveInfinity"/>). / 正のパラメータ列（<see cref="double.PositiveInfinity"/> も許容）。</param>
        /// <returns>The value of <c>ln B(α)</c>, <c>0</c> for a single element, <see cref="double.NegativeInfinity"/> if any element is <see cref="double.PositiveInfinity"/> with at least two elements, or <see cref="double.NaN"/> if <paramref name="alpha"/> is empty or contains a non-positive or <see cref="double.NaN"/> element. / <c>ln B(α)</c> の値。要素が 1 個なら <c>0</c>、2 個以上で <see cref="double.PositiveInfinity"/> を含む場合は <see cref="double.NegativeInfinity"/>。空または 0 以下・<see cref="double.NaN"/> の要素を含む場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Beta_function#Multivariate_beta_function"/>
        public static double LogBeta(ReadOnlySpan<double> alpha)
        {
            if (alpha.IsEmpty)
            {
                return double.NaN;
            }

            foreach (double value in alpha)
            {
                if (!(value > 0))
                {
                    return double.NaN;
                }
            }

            if (alpha.Length == 1)
            {
                return 0;
            }

            Span<double> sorted = alpha.Length <= 64 ? stackalloc double[alpha.Length] : new double[alpha.Length];
            alpha.CopyTo(sorted);
            sorted.Sort();

            double partial = sorted[0];
            double sum = 0;
            double compensation = 0;
            for (int k = 1; k < sorted.Length; k++)
            {
                if (double.IsPositiveInfinity(partial))
                {
                    return double.NegativeInfinity;
                }

                double term = LogBeta(partial, sorted[k]);
                if (double.IsNegativeInfinity(term))
                {
                    return double.NegativeInfinity;
                }

                double t = sum + term;
                if (double.IsNegativeInfinity(t))
                {
                    // 正の項は高々 -ln(最小成分) 程度なので、和が -∞ へオーバーフローすれば結果も -∞ (補償計算での ∞ - ∞ を回避)
                    return double.NegativeInfinity;
                }

                compensation += Math.Abs(sum) >= Math.Abs(term) ? sum - t + term : term - t + sum;
                sum = t;
                partial += sorted[k];
            }

            return sum + compensation;
        }
    }
}
