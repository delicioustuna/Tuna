namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // これ未満の a では、UpperGammaSmallA の線形計算で非正規数の下位ビットが失われるため、x < 1.1 の上側を Q(a, x) ≈ a E₁(x) で近似する
        private const double LogGammaTinyA = 1e-20;

        /// <summary>
        /// Computes the natural logarithm of the regularized lower incomplete gamma function <c>ln P(a, x)</c>.
        /// <para>正則化下側不完全ガンマ関数の自然対数 <c>ln P(a, x)</c> を計算します。</para>
        /// </summary>
        /// <param name="a">The shape parameter (<c>a &gt; 0</c>). / 形状パラメータ（<c>a &gt; 0</c>）。</param>
        /// <param name="x">The upper integration limit (<c>x ≥ 0</c>). / 積分区間の上端（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>ln P(a, x)</c>, or <see cref="double.NaN"/> if <paramref name="a"/> ≤ 0, <paramref name="x"/> &lt; 0, or either argument is <see cref="double.NaN"/>. / <c>ln P(a, x)</c> の値。<paramref name="a"/> ≤ 0、<paramref name="x"/> &lt; 0、またはいずれかの引数が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <see cref="double.NegativeInfinity"/> at <c>x = 0</c> and <c>0</c> at <c>x = +∞</c>. Remains finite even where <c>P(a, x)</c> underflows to zero.</para>
        /// <para><c>x = 0</c> では <see cref="double.NegativeInfinity"/>、<c>x = +∞</c> では <c>0</c> を返します。<c>P(a, x)</c> が 0 にアンダーフローする領域でも有限の値を保ちます。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/8.2.E4"/>
        public static double LogRegularizedGammaP(double a, double x)
        {
            if (TryIncompleteGammaSpecialCase(a, x, out double p, out _))
            {
                return Math.Log(p);
            }

            LogIncompleteGammaCore(a, x, out double logP, out _);
            return logP;
        }

        /// <summary>
        /// Computes the natural logarithm of the regularized upper incomplete gamma function <c>ln Q(a, x)</c>.
        /// <para>正則化上側不完全ガンマ関数の自然対数 <c>ln Q(a, x)</c> を計算します。</para>
        /// </summary>
        /// <param name="a">The shape parameter (<c>a &gt; 0</c>). / 形状パラメータ（<c>a &gt; 0</c>）。</param>
        /// <param name="x">The lower integration limit (<c>x ≥ 0</c>). / 積分区間の下端（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>ln Q(a, x)</c>, or <see cref="double.NaN"/> if <paramref name="a"/> ≤ 0, <paramref name="x"/> &lt; 0, or either argument is <see cref="double.NaN"/>. / <c>ln Q(a, x)</c> の値。<paramref name="a"/> ≤ 0、<paramref name="x"/> &lt; 0、またはいずれかの引数が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <c>0</c> at <c>x = 0</c> and <see cref="double.NegativeInfinity"/> at <c>x = +∞</c>. Remains finite even where <c>Q(a, x)</c> underflows to zero.</para>
        /// <para><c>x = 0</c> では <c>0</c>、<c>x = +∞</c> では <see cref="double.NegativeInfinity"/> を返します。<c>Q(a, x)</c> が 0 にアンダーフローする領域でも有限の値を保ちます。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/8.2.E4"/>
        public static double LogRegularizedGammaQ(double a, double x)
        {
            if (TryIncompleteGammaSpecialCase(a, x, out _, out double q))
            {
                return Math.Log(q);
            }

            LogIncompleteGammaCore(a, x, out _, out double logQ);
            return logQ;
        }

        // ln P(a, x) と ln Q(a, x)（a > 0、0 < x < ∞）。
        // IncompleteGammaCore と同じ分岐で直接計算する小さい方を対数で求め、もう一方は ln(1 - e^(小さい方)) とする
        private static void LogIncompleteGammaCore(double a, double x, out double logP, out double logQ)
        {
            if (InTemmeGammaRegion(a, x))
            {
                double factor = TemmeGammaTailFactor(a, x, out double halfEta2, out bool upper);
                AssignLogTails(Math.Log(factor) - a * halfEta2, upper, out logP, out logQ);
                return;
            }

            if (x < 1.1)
            {
                if (UseLowerGammaSeriesForSmallX(a, x))
                {
                    AssignLogTails(Math.Log(LowerGammaSum(a, x)) + LogGammaPrefix(a, x), upper: false, out logP, out logQ);
                }
                else
                {
                    AssignLogTails(LogUpperGammaSmallA(a, x), upper: true, out logP, out logQ);
                }

                return;
            }

            if (x < a + 1)
            {
                AssignLogTails(Math.Log(LowerGammaSum(a, x)) + LogGammaPrefix(a, x), upper: false, out logP, out logQ);
            }
            else
            {
                AssignLogTails(Math.Log(UpperGammaFraction(a, x)) + LogGammaPrefix(a, x), upper: true, out logP, out logQ);
            }
        }

        // 直接計算した裾の対数から ln P と ln Q を割り当てる（upper が真なら直接計算したのは Q）
        private static void AssignLogTails(double logTail, bool upper, out double logP, out double logQ)
        {
            // 丸めで正になった値は 1 とみなす（もう一方は -∞ になる）
            logTail = Math.Min(logTail, 0);
            double logOther = Log1mExp(logTail);
            logP = upper ? logOther : logTail;
            logQ = upper ? logTail : logOther;
        }

        // ln(xᵃ e⁻ˣ / Γ(a))（a > 0、x > 0）。
        // GammaPrefix と同じ式を対数のまま評価し、アンダーフローやオーバーフローを起こす範囲でも有限の値を返す
        internal static double LogGammaPrefix(double a, double x)
        {
            if (a >= 10)
            {
                // ln √(a/2π) - a φ(x/a) - δ(a)、φ(t) = t - 1 - ln t。x/a は x が小さいとアンダーフローしうるので、そのときは対数の差にする
                double u = (x - a) / a;
                double phi;
                if (Math.Abs(u) < 0.5)
                {
                    phi = -Log1pmx(u);
                }
                else
                {
                    double ratio = x / a;
                    phi = u - (ratio >= MinNormal ? Math.Log(ratio) : Math.Log(x) - Math.Log(a));
                }

                return 0.5 * Math.Log(a / (2 * Math.PI)) - a * phi - StirlingCorrection(a);
            }

            if (x < 700)
            {
                // 正規化数で表せる範囲では、各因子を相対 ε で求めてから対数をとる方が、大きな項の和より誤差が小さい
                double direct = Math.Pow(x, a) * Math.Exp(-x) * ReciprocalGamma(a);
                if (direct > 1e-290 && double.IsFinite(direct))
                {
                    return Math.Log(direct);
                }
            }

            return a * Math.Log(x) - x - LogGamma(a);
        }

        // ln Q(a, x)（x < 1.1 で UseLowerGammaSeriesForSmallX が偽の範囲）
        private static double LogUpperGammaSmallA(double a, double x)
        {
            if (a < LogGammaTinyA)
            {
                return Math.Log(a) + Math.Log(ExpIntegralE(1, x));
            }

            return Math.Log(UpperGammaSmallA(a, x));
        }
    }
}
