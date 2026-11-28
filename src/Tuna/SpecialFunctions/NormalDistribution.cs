using static Tuna.XMath.Constants;
using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        private const double LogQuarter = -1.3862943611198906;
        private const double LogThreeQuarters = -0.2876820724517809;
        private const double LogMinNormal = -708.3964185322641;

        // |x| < NormalCentralLimit（Q(x) ≥ 0.25）では Φ(x) - 1/2 = x N(x²) の多項式で評価する
        private const double NormalCentralLimit = 0.66;

        // 上側分位点で初期値がこれ未満（Q(x) ≥ 0.105）なら中央多項式で残差を計算し、それ以上は erfcx を使う
        private const double InverseNormalCentralRefineLimit = 1.25;

        // これより大きい |x| では e^(-x²/2) と Q(x) がともに 0 にアンダーフローする（実境界は |x| ≈ 38.6）
        private const double NormalTailUnderflow = 40;

        /// <summary>
        /// Computes the standard normal probability density function <c>φ(x)</c>.
        /// <para>標準正規分布の確率密度関数 <c>φ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The evaluation point. / 評価点。</param>
        /// <returns>The density <c>φ(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is <see cref="double.NaN"/>. / 確率密度 <c>φ(x)</c>。<paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Normal_distribution#Standard_normal_distribution"/>
        public static double NormalPdf(double x)
        {
            if (double.IsNaN(x))
            {
                return x;
            }

            return ScaledExpMinusHalfSquare(x, OneOverSqrtTwoPi);
        }

        /// <summary>
        /// Computes the natural logarithm of the standard normal probability density function <c>ln φ(x)</c>.
        /// <para>標準正規分布の確率密度関数の自然対数 <c>ln φ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The evaluation point. / 評価点。</param>
        /// <returns>The log-density <c>ln φ(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is <see cref="double.NaN"/>. / 対数密度 <c>ln φ(x)</c>。<paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <see cref="double.NegativeInfinity"/> at <c>x = ±∞</c>. Remains finite even where <c>φ(x)</c> underflows to zero.</para>
        /// <para><c>x = ±∞</c> では <see cref="double.NegativeInfinity"/> を返します。<c>φ(x)</c> が 0 にアンダーフローする領域でも有限の値を保ちます。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Normal_distribution#Standard_normal_distribution"/>
        public static double NormalLogPdf(double x)
        {
            if (double.IsNaN(x))
            {
                return x;
            }

            // x² は x²/2 より先にオーバーフローするので、x²/2 を (x/2)·x として丸め誤差 lo とともに求める
            double half = 0.5 * x;
            double hi = half * x;
            if (double.IsInfinity(hi))
            {
                return double.NegativeInfinity;
            }

            double lo = Math.FusedMultiplyAdd(half, x, -hi);
            return (-LnSqrtTwoPi - lo) - hi;
        }

        /// <summary>
        /// Computes the standard normal cumulative distribution function <c>Φ(x)</c>.
        /// <para>標準正規分布の累積分布関数 <c>Φ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The evaluation point. / 評価点。</param>
        /// <returns>The cumulative probability <c>Φ(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is <see cref="double.NaN"/>. / 累積確率 <c>Φ(x)</c>。<paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Preserves relative accuracy in the negative tail.</para>
        /// <para>負の裾の領域でも相対精度を保ちます。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function"/>
        public static double NormalCdf(double x)
        {
            if (Math.Abs(x) < NormalCentralLimit)
            {
                return Math.FusedMultiplyAdd(x, NormalCentralPolynomial(x), 0.5);
            }

            if (double.IsNaN(x))
            {
                return x;
            }

            return x < 0 ? NormalUpperTail(-x) : 1 - NormalUpperTail(x);
        }

        /// <summary>
        /// Computes the standard normal survival function (upper-tail probability) <c>Q(x) = 1 - Φ(x)</c>.
        /// <para>標準正規分布の生存関数（上側確率）<c>Q(x) = 1 - Φ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The evaluation point. / 評価点。</param>
        /// <returns>The upper-tail probability <c>Q(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is <see cref="double.NaN"/>. / 上側確率 <c>Q(x)</c>。<paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Evaluated directly without subtracting <c>Φ(x)</c> from 1, preserving relative accuracy in the positive tail.</para>
        /// <para><c>1 - Φ(x)</c> の減算を経由せず直接計算するため、正の裾でも相対精度を保ちます。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Q-function"/>
        public static double NormalSf(double x)
        {
            if (Math.Abs(x) < NormalCentralLimit)
            {
                return Math.FusedMultiplyAdd(-x, NormalCentralPolynomial(x), 0.5);
            }

            if (double.IsNaN(x))
            {
                return x;
            }

            return x > 0 ? NormalUpperTail(x) : 1 - NormalUpperTail(-x);
        }

        /// <summary>
        /// Computes the natural logarithm of the standard normal cumulative distribution function <c>ln Φ(x)</c>.
        /// <para>標準正規分布の累積分布関数の自然対数 <c>ln Φ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The evaluation point. / 評価点。</param>
        /// <returns>The value of <c>ln Φ(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is <see cref="double.NaN"/>. / <c>ln Φ(x)</c> の値。<paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <see cref="double.NegativeInfinity"/> at <c>x = -∞</c> and <c>0</c> at <c>x = +∞</c>. Remains finite even in the negative tail where <c>Φ(x)</c> underflows to zero.</para>
        /// <para><c>x = -∞</c> では <see cref="double.NegativeInfinity"/>、<c>x = +∞</c> では <c>0</c> を返します。<c>Φ(x)</c> が 0 にアンダーフローする負の裾でも有限の値を保ちます。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function"/>
        public static double NormalLogCdf(double x)
        {
            if (double.IsNaN(x))
            {
                return x;
            }

            return x <= 0 ? LogNormalUpperTail(-x) : LogOneMinusNormalUpperTail(x);
        }

        /// <summary>
        /// Computes the natural logarithm of the standard normal survival function <c>ln Q(x) = ln(1 - Φ(x))</c>.
        /// <para>標準正規分布の生存関数の自然対数 <c>ln Q(x) = ln(1 - Φ(x))</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The evaluation point. / 評価点。</param>
        /// <returns>The value of <c>ln Q(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is <see cref="double.NaN"/>. / <c>ln Q(x)</c> の値。<paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <c>0</c> at <c>x = -∞</c> and <see cref="double.NegativeInfinity"/> at <c>x = +∞</c>. Remains finite even in the positive tail where <c>Q(x)</c> underflows to zero.</para>
        /// <para><c>x = -∞</c> では <c>0</c>、<c>x = +∞</c> では <see cref="double.NegativeInfinity"/> を返します。<c>Q(x)</c> が 0 にアンダーフローする正の裾でも有限の値を保ちます。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Q-function"/>
        public static double NormalLogSf(double x)
        {
            if (double.IsNaN(x))
            {
                return x;
            }

            return x >= 0 ? LogNormalUpperTail(x) : LogOneMinusNormalUpperTail(-x);
        }

        /// <summary>
        /// Computes the standard normal quantile function <c>Φ⁻¹(p)</c>.
        /// <para>標準正規分布の分位点関数 <c>Φ⁻¹(p)</c> を計算します。</para>
        /// </summary>
        /// <param name="p">The cumulative probability (<c>0 ≤ p ≤ 1</c>). / 累積確率（<c>0 ≤ p ≤ 1</c>）。</param>
        /// <returns>The value <c>x</c> satisfying <c>Φ(x) = p</c>, or <see cref="double.NaN"/> if <paramref name="p"/> is outside <c>[0, 1]</c> or <see cref="double.NaN"/>. / <c>Φ(x) = p</c> を満たす <c>x</c>。<paramref name="p"/> が <c>[0, 1]</c> の範囲外または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <see cref="double.NegativeInfinity"/> at <c>p = 0</c> and <see cref="double.PositiveInfinity"/> at <c>p = 1</c>.</para>
        /// <para><c>p = 0</c> では <see cref="double.NegativeInfinity"/>、<c>p = 1</c> では <see cref="double.PositiveInfinity"/> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Normal_distribution#Quantile_function"/>
        public static double InverseNormalCdf(double p)
        {
            if (!(p >= 0 && p <= 1))
            {
                return double.NaN;
            }

            // p - 1/2 と 1 - p は丸め誤差なく求まる
            if (p < 0.25)
            {
                return -InverseNormalSfSmall(p);
            }

            return p <= 0.75 ? InverseNormalCentral(p - 0.5) : InverseNormalSfSmall(1 - p);
        }

        /// <summary>
        /// Computes the standard normal upper-tail quantile <c>Q⁻¹(q) = -Φ⁻¹(q)</c>.
        /// <para>標準正規分布の上側分位点 <c>Q⁻¹(q) = -Φ⁻¹(q)</c> を計算します。</para>
        /// </summary>
        /// <param name="q">The upper-tail probability (<c>0 ≤ q ≤ 1</c>). / 上側確率（<c>0 ≤ q ≤ 1</c>）。</param>
        /// <returns>The value <c>x</c> satisfying <c>Q(x) = q</c>, or <see cref="double.NaN"/> if <paramref name="q"/> is outside <c>[0, 1]</c> or <see cref="double.NaN"/>. / <c>Q(x) = q</c> を満たす <c>x</c>。<paramref name="q"/> が <c>[0, 1]</c> の範囲外または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Evaluated directly from <paramref name="q"/> without forming <c>1 - q</c>, preserving accuracy for small <paramref name="q"/>. Returns <see cref="double.PositiveInfinity"/> at <c>q = 0</c> and <see cref="double.NegativeInfinity"/> at <c>q = 1</c>.</para>
        /// <para><c>1 - q</c> を経由せず直接計算するため、小さな <paramref name="q"/> でも精度を保ちます。<c>q = 0</c> では <see cref="double.PositiveInfinity"/>、<c>q = 1</c> では <see cref="double.NegativeInfinity"/> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Q-function"/>
        public static double InverseNormalSf(double q)
        {
            if (!(q >= 0 && q <= 1))
            {
                return double.NaN;
            }

            if (q < 0.25)
            {
                return InverseNormalSfSmall(q);
            }

            return q <= 0.75 ? InverseNormalCentral(0.5 - q) : -InverseNormalSfSmall(1 - q);
        }

        /// <summary>
        /// Computes the standard normal quantile <c>Φ⁻¹(e^logP)</c> from a log-probability.
        /// <para>対数確率から標準正規分布の分位点 <c>Φ⁻¹(e^logP)</c> を計算します。</para>
        /// </summary>
        /// <param name="logP">The natural logarithm of the cumulative probability (<c>logP ≤ 0</c>). / 累積確率の自然対数（<c>logP ≤ 0</c>）。</param>
        /// <returns>The value <c>x</c> satisfying <c>ln Φ(x) = logP</c>, or <see cref="double.NaN"/> if <paramref name="logP"/> &gt; 0 or <see cref="double.NaN"/>. / <c>ln Φ(x) = logP</c> を満たす <c>x</c>。<paramref name="logP"/> &gt; 0 または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Remains finite even where <c>e^logP</c> underflows to zero. Returns <see cref="double.NegativeInfinity"/> at <c>logP = -∞</c> and <see cref="double.PositiveInfinity"/> at <c>logP = 0</c>.</para>
        /// <para><c>e^logP</c> が 0 にアンダーフローする極端な裾でも有限の値を返します。<c>logP = -∞</c> では <see cref="double.NegativeInfinity"/>、<c>logP = 0</c> では <see cref="double.PositiveInfinity"/> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Normal_distribution#Quantile_function"/>
        public static double InverseNormalLogCdf(double logP)
        {
            // Φ(x) = p ⇔ Q(-x) = p
            return -InverseNormalLogSf(logP);
        }

        /// <summary>
        /// Computes the standard normal upper-tail quantile <c>Q⁻¹(e^logQ)</c> from a log-probability.
        /// <para>対数確率から標準正規分布の上側分位点 <c>Q⁻¹(e^logQ)</c> を計算します。</para>
        /// </summary>
        /// <param name="logQ">The natural logarithm of the upper-tail probability (<c>logQ ≤ 0</c>). / 上側確率の自然対数（<c>logQ ≤ 0</c>）。</param>
        /// <returns>The value <c>x</c> satisfying <c>ln Q(x) = logQ</c>, or <see cref="double.NaN"/> if <paramref name="logQ"/> &gt; 0 or <see cref="double.NaN"/>. / <c>ln Q(x) = logQ</c> を満たす <c>x</c>。<paramref name="logQ"/> &gt; 0 または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Remains finite even where <c>e^logQ</c> underflows to zero. Returns <see cref="double.PositiveInfinity"/> at <c>logQ = -∞</c> and <see cref="double.NegativeInfinity"/> at <c>logQ = 0</c>.</para>
        /// <para><c>e^logQ</c> が 0 にアンダーフローする極端な裾でも有限の値を返します。<c>logQ = -∞</c> では <see cref="double.PositiveInfinity"/>、<c>logQ = 0</c> では <see cref="double.NegativeInfinity"/> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Q-function"/>
        public static double InverseNormalLogSf(double logQ)
        {
            if (double.IsNaN(logQ) || logQ > 0)
            {
                return double.NaN;
            }

            if (logQ == 0)
            {
                return double.NegativeInfinity;
            }

            if (double.IsNegativeInfinity(logQ))
            {
                return double.PositiveInfinity;
            }

            if (logQ > LogQuarter && logQ < LogThreeQuarters)
            {
                // 1/4 < q < 3/4 では Q(x) = q ⇔ Φ(x) - 1/2 = 1/2 - q = -expm1(ln q + ln 2)/2。
                // q を経由すると q の丸め誤差が x の相対誤差に十数倍に拡大されるので、1/2 - q を対数から直接求める。
                // ln q + ln 2 は 2 数の和の誤差と ln 2 の下位を 1 次で補正する
                double sum = logQ + Ln2;
                double sumLo = (logQ - (sum - Ln2)) + Ln2Lo;
                double m = Expm1(sum);
                return InverseNormalCentral(-0.5 * Math.FusedMultiplyAdd(sumLo, 1 + m, m));
            }

            if (logQ > -Ln2)
            {
                // q > 1/2 では Q(x) = q ⇔ Q(-x) = 1 - q。1 - q = -expm1(ln q) は桁落ちしない
                return -InverseNormalSf(-Expm1(logQ));
            }

            if (logQ >= LogMinNormal)
            {
                // e^logQ の相対誤差は数 ε で、x の相対誤差には ε/(x²) 程度にしか伝わらない
                return InverseNormalSf(Math.Exp(logQ));
            }

            return InverseNormalLogSfTail(logQ);
        }

        // 上側確率 Q(x) = erfc(x/√2)/2（x ≥ 0）
        private static double NormalUpperTail(double x)
        {
            if (x < NormalCentralLimit)
            {
                return Math.FusedMultiplyAdd(-x, NormalCentralPolynomial(x), 0.5);
            }

            if (!(x <= NormalTailUnderflow))
            {
                return double.IsNaN(x) ? x : 0;
            }

            // Q(x) = e^(-x²/2) erfcx(x/√2)/2。x² = hi + lo、x/√2 = t + tLo として、e^(-x²/2) = e^(-hi/2)(1 - lo/2) の補正と
            // t の丸め誤差の補正は erfcx の最後の丸めの前に入れる
            double hi = x * x;
            double lo = Math.FusedMultiplyAdd(x, x, -hi);
            double t = x * OneOverSqrt2;
            double tLo = Math.FusedMultiplyAdd(x, OneOverSqrt2, -t) + x * OneOverSqrt2Lo;
            return ScaleByExpMinusHalf(0.5 * ErfcxPositive(t, tLo, -0.5 * lo), hi);
        }

        // ln Q(x)（x ≥ 0。Q(x) がアンダーフローする範囲でも有限の値を返す）
        private static double LogNormalUpperTail(double x)
        {
            if (x < NormalCentralLimit)
            {
                return Math.Log(NormalUpperTail(x));
            }

            // ln Q(x) = -x²/2 + ln(erfcx(x/√2)/2)。x²/2 は (x/2)·x として丸め誤差 lo とともに求める
            double half = 0.5 * x;
            double hi = half * x;
            if (double.IsInfinity(hi))
            {
                return double.NegativeInfinity;
            }

            double lo = Math.FusedMultiplyAdd(half, x, -hi);
            return (Math.Log(0.5 * NormalErfcx(x)) - lo) - hi;
        }

        // |x| < √2 での N(x²) = (Φ(x) - 1/2)/x
        private static double NormalCentralPolynomial(double x) =>
            PolynomialSplitConstant(NormalCentralCoefficients, x * x);

        // erfcx(x/√2)（x ≥ 0, 有限）。t = x/√2 の丸め誤差 tLo（積の丸めと 1/√2 の残差）を 1 次補正する
        private static double NormalErfcx(double x)
        {
            double t = x * OneOverSqrt2;
            double tLo = Math.FusedMultiplyAdd(x, OneOverSqrt2, -t) + x * OneOverSqrt2Lo;
            return ErfcxPositive(t, tLo, 0);
        }

        // |d| ≤ 1/4 での Φ⁻¹(1/2 + d)。初期値 √2 erf⁻¹(2d)（相対 2⁻³⁴ 程度）を Halley 法 1 回で仕上げる
        private static double InverseNormalCentral(double d)
        {
            double w = 2 * d;
            return RefineInverseNormalCentral(Sqrt2 * (w * Polynomial(InverseErfCentralCoefficients, w * w)), d, 0);
        }

        // Φ(x) - 1/2 = d + dLo（|x| < 1.3）を初期値 x から Halley 法 1 回で仕上げる。
        // 残差 x N(x²) - d は定数項の上位積を誤差なし分割して桁落ちなく求める
        private static double RefineInverseNormalCentral(double x, double d, double dLo)
        {
            ReadOnlySpan<double> c = NormalCentralCoefficients;
            double u = x * x;
            double product = x * c[0];
            double productLo = Math.FusedMultiplyAdd(x, c[0], -product);
            double rest = Math.FusedMultiplyAdd(Polynomial(c[1..^1], u), u, c[^1]);
            double residual = (product - d) + (Math.FusedMultiplyAdd(x, rest, productLo) - dLo);

            // f(x) = Φ(x) - 1/2 - d、f'(x) = φ(x)、f''/f' = -x
            double step = residual / (OneOverSqrtTwoPi * Math.Exp(-0.5 * u));
            return x - step / (1 + 0.5 * x * step);
        }

        // 0 ≤ q < 1/4 での Q⁻¹(q)。初期値 √2 erfc⁻¹(2q)（相対 2⁻³⁴ 程度）を Halley 法 1 回で仕上げる
        private static double InverseNormalSfSmall(double q)
        {
            if (q == 0)
            {
                return double.PositiveInfinity;
            }

            if (q < MinNormal)
            {
                // q/φ(x) の e^(x²/2) がオーバーフローする裾は対数で解く
                return InverseNormalLogSfTail(Math.Log(q));
            }

            double x = Sqrt2 * InverseErfcInitialGuess(Math.Sqrt(-Math.Log(2 * q)));
            if (x < InverseNormalCentralRefineLimit)
            {
                // Q(x) - q = (1/2 - q) - (Φ(x) - 1/2)。1/2 - q を上位と下位に分けると、残差は中央の形で桁落ちなく求まる
                double d = 0.5 - q;
                return RefineInverseNormalCentral(x, d, (0.5 - d) - q);
            }

            // g(x) = Q(x) - q、g'(x) = -φ(x)、g''/g' = -x。g/g' = -(M(x) - q/φ(x))（ミルズ比 M(x) = √(π/2) erfcx(x/√2)）
            double step = -SqrtPiOver2 * (NormalErfcx(x) - 2 * q * ExpHalfSquare(x));
            return x - step / (1 + 0.5 * x * step);
        }

        // ln(1 - Q(x))（x ≥ 0, Q(x) ≤ 1/2 のため桁落ちしない）
        private static double LogOneMinusNormalUpperTail(double x)
        {
            double q = NormalUpperTail(x);
            return q == 0 ? 0 : Log1p(-q);
        }

        // ln Q(x) = logQ（logQ < LogMinNormal の深い裾）をニュートン法で解く。
        // ln Q は凹関数のため、1 回目以降の反復は解の右側から単調に収束する
        private static double InverseNormalLogSfTail(double logQ)
        {
            double l = -logQ;
            double x;
            if (l < 778)
            {
                // r = √(-ln 2q) = √(l - ln 2) < 27.9
                x = Sqrt2 * InverseErfcInitialGuess(Math.Sqrt(l - Ln2));
            }
            else
            {
                // x² = 2(l - ln(x√(2π)))。2l はオーバーフローしうるので √2 √l の形で求める
                x = Sqrt2 * Math.Sqrt(l);
                for (int i = 0; i < 3; i++)
                {
                    x = Sqrt2 * Math.Sqrt(l - (Math.Log(x) + LnSqrtTwoPi));
                }
            }

            for (int i = 0; i < 50; i++)
            {
                // d ln Q/dx = -φ(x)/Q(x) = -1/M(x)。ミルズ比 M(x) = √(π/2) erfcx(x/√2)。
                // 2 次の項から、ステップ後の誤差は step²/(2x) 程度。|step| ≤ 1e-9 x なら倍精度の丸めより十分小さい
                double residual = LogNormalUpperTail(x) - logQ;
                double step = residual * SqrtPiOver2 * Erfcx(x * OneOverSqrt2);
                x += step;
                if (!(Math.Abs(step) > 1e-9 * x))
                {
                    break;
                }
            }

            return x;
        }

        // e^(x²/2) を x²/2 の丸め誤差を補正して計算する
        private static double ExpHalfSquare(double x)
        {
            double half = 0.5 * x;
            double hi = half * x;
            double lo = Math.FusedMultiplyAdd(half, x, -hi);
            double e = Math.Exp(hi);
            return e + e * lo;
        }

        // factor · e^(-x²/2)（factor は正規化数を想定）。x² の丸め誤差を補正し、非正規数域でも丸めを最後の 1 回にする
        private static double ScaledExpMinusHalfSquare(double x, double factor)
        {
            double ax = Math.Abs(x);
            if (ax > NormalTailUnderflow)
            {
                return 0;
            }

            // e^(-(hi + lo)/2) = e^(-hi/2) (1 - lo/2 + O(lo²))
            double hi = ax * ax;
            double lo = Math.FusedMultiplyAdd(ax, ax, -hi);
            return ScaleByExpMinusHalf(factor - factor * (0.5 * lo), hi);
        }

        // factor · e^(-hi/2)（非正規数域では e^(-hi/4) を 2 回に分けて掛ける）
        private static double ScaleByExpMinusHalf(double factor, double hi)
        {
            if (hi < 1400)
            {
                return factor * Math.Exp(-0.5 * hi);
            }

            double quarter = Math.Exp(-0.25 * hi);
            return factor * quarter * quarter;
        }
    }
}
