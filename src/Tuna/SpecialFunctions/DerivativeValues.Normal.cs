using static Tuna.XMath.Constants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // |x| > 28 では 2e^(-x²)/√π が最小非正規化数の半分を下回る
        private const double ErfDerivativeUnderflowLimit = 28;
        // x ≥ 2²⁷ では φ(x)/Q(x) = x + 1/x + O(x⁻³) の補正が x の半 ULP 未満になる
        private const double NormalHazardRoundedAsymptoteLimit = 134217728;

        /// <summary>
        /// Computes the derivative of the error function, <c>erf'(x) = (2 / √π) e^(-x²)</c>.
        /// <para>誤差関数の導関数 <c>erf'(x) = (2 / √π) e^(-x²)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>erf'(x)</c>, <c>0</c> at <c>±∞</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is <see cref="double.NaN"/>. / <c>erf'(x)</c> の値。<c>±∞</c> では <c>0</c>、<paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/7.10"/>
        public static double ErfDerivative(double x)
        {
            if (double.IsNaN(x))
            {
                return x;
            }

            double absolute = Math.Abs(x);
            if (absolute > ErfDerivativeUnderflowLimit)
            {
                return 0;
            }

            // 2 乗の丸め誤差を係数へ反映し、exp に渡す引数の誤差を補償する
            double square = absolute * absolute;
            double error = Math.FusedMultiplyAdd(absolute, absolute, -square);
            double factor = TwoOverSqrtPi - TwoOverSqrtPi * error;
            if (square < 700)
            {
                return factor * Math.Exp(-square);
            }

            // e^(-x²) を先に丸めると、係数を掛けた結果が表現可能でも 0 になりうるため半分ずつ掛ける
            double half = Math.Exp(-0.5 * square);
            return factor * half * half;
        }

        /// <summary>
        /// Computes the derivative of the standard normal probability density function, <c>φ'(x) = -x φ(x)</c>.
        /// <para>標準正規分布の確率密度関数の導関数 <c>φ'(x) = -x φ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The standardized variate. / 標準化された変量。</param>
        /// <returns>The value of <c>φ'(x)</c> (preserving sign-inverted zero at <c>±0</c>, <c>0</c> at <c>±∞</c>), or <see cref="double.NaN"/> if <paramref name="x"/> is <see cref="double.NaN"/>. / <c>φ'(x)</c> の値（<c>±0</c> では符号を反転した 0、<c>±∞</c> では <c>0</c>）。<paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Normal_distribution"/>
        public static double NormalPdfDerivative(double x)
        {
            if (double.IsNaN(x))
            {
                return x;
            }

            if (double.IsInfinity(x))
            {
                return 0;
            }

            if (x == 0)
            {
                return -x;
            }

            // 密度を先に丸めず、-x を掛けた導関数が表現可能な範囲まで指数を保持する
            return ScaledExpMinusHalfSquare(x, -x * OneOverSqrtTwoPi);
        }

        /// <summary>
        /// Computes the hazard function (inverse Mills ratio) of the standard normal distribution, <c>φ(x) / Q(x)</c>.
        /// <para>標準正規分布のハザード関数（逆ミルズ比）<c>φ(x) / Q(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The standardized variate. / 標準化された変量。</param>
        /// <returns>The value of <c>φ(x) / Q(x)</c> (<c>0</c> at <c>-∞</c>, <see cref="double.PositiveInfinity"/> at <c>+∞</c>), or <see cref="double.NaN"/> if <paramref name="x"/> is <see cref="double.NaN"/>. / <c>φ(x) / Q(x)</c> の値（<c>-∞</c> では <c>0</c>、<c>+∞</c> では <see cref="double.PositiveInfinity"/>）。<paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Remains finite and accurate even in the far right tail where <c>Q(x)</c> underflows to zero.</para>
        /// <para>上側確率 <c>Q(x)</c> がアンダーフローして 0 になる正の裾でも、有限かつ高精度な比を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Mills_ratio"/>
        public static double NormalHazard(double x)
        {
            if (double.IsNaN(x))
            {
                return x;
            }

            // φ(x)/Q(x) = x + 1/x + O(x⁻³)。補正が半 ULP 未満となる領域では x を直接返し、erfcx の非正規化丸めによる誤差も避ける
            if (x >= NormalHazardRoundedAsymptoteLimit)
            {
                return x;
            }

            return x >= 0 ? (2 * OneOverSqrtTwoPi) / NormalErfcx(x) : NormalPdf(x) / NormalSf(x);
        }

        /// <summary>
        /// Computes the reverse hazard function of the standard normal distribution, <c>φ(x) / Φ(x)</c>.
        /// <para>標準正規分布の逆ハザード関数 <c>φ(x) / Φ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The standardized variate. / 標準化された変量。</param>
        /// <returns>The value of <c>φ(x) / Φ(x)</c> (<see cref="double.PositiveInfinity"/> at <c>-∞</c>, <c>0</c> at <c>+∞</c>), or <see cref="double.NaN"/> if <paramref name="x"/> is <see cref="double.NaN"/>. / <c>φ(x) / Φ(x)</c> の値（<c>-∞</c> では <see cref="double.PositiveInfinity"/>、<c>+∞</c> では <c>0</c>）。<paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Remains finite and accurate even in the far left tail where <c>Φ(x)</c> underflows to zero.</para>
        /// <para>下側確率 <c>Φ(x)</c> がアンダーフローして 0 になる負の裾でも、有限かつ高精度な比を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Mills_ratio"/>
        public static double NormalReverseHazard(double x) => NormalHazard(-x);
    }
}
