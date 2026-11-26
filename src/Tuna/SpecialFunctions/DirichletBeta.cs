namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // Cohen–Villegas–Zagier 交代級数加速の項数 (誤差は和の 2/(3 + √8)ⁿ 倍以下; n = 24 で 7e-19)
        private const int AlternatingSeriesTerms = 24;

        /// <summary>
        /// Computes the Dirichlet beta function <c>β(s) = Σₙ₌₀^∞ (-1)ⁿ (2n + 1)⁻ˢ</c>.
        /// <para>ディリクレのベータ関数 <c>β(s) = Σₙ₌₀^∞ (-1)ⁿ (2n + 1)⁻ˢ</c> を計算します。</para>
        /// </summary>
        /// <param name="s">The real input value. / 実数の入力値。</param>
        /// <returns>The value of <c>β(s)</c> (with <c>β(1) = π / 4</c>, <c>β(2)</c> equal to Catalan's constant, and <c>0</c> at negative odd integers), or <see cref="double.NaN"/> at <see cref="double.NaN"/> and <see cref="double.NegativeInfinity"/>. / <c>β(s)</c> の値（<c>β(1) = π / 4</c>、<c>β(2)</c> はカタラン定数、負の奇数では <c>0</c>）。<see cref="double.NaN"/> および <see cref="double.NegativeInfinity"/> では <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/25.15"/>
        public static double DirichletBeta(double s)
        {
            if (double.IsNaN(s))
            {
                return s;
            }

            if (double.IsPositiveInfinity(s))
            {
                return 1;
            }

            if (s >= 0.5)
            {
                return DirichletBetaAlternating(s);
            }

            if (double.IsNegativeInfinity(s))
            {
                return double.NaN;
            }

            var (u, delta) = DoubleDoubleTwoSum(1, -s);
            double cos = double.CosPi(0.5 * s);
            if (cos == 0)
            {
                return 0;
            }

            double beta = DirichletBetaAlternating(u);
            // (2/π)ᵘ は 2/π の丸め相対誤差が u 倍に拡大されるため 1 次補正する
            const double twoOverPiRelativeError = -6.182238608e-17;
            const double logTwoOverPi = -0.45158270528945486;
            double power = Math.Pow(2 / Math.PI, u) * (1 + u * twoOverPiRelativeError);

            // Γ(u) は u > 171 でオーバーフローするため仮数と指数に分離して掛ける
            double mantissa = ScaledGammaSigned(u, out long exponent);
            if (delta != 0)
            {
                mantissa *= 1 + (Digamma(u) + logTwoOverPi) * delta;
            }

            return ScaleB(power * mantissa * cos * beta, exponent);
        }

        // s > 0 での β(s) を Cohen–Villegas–Zagier 加速で計算する
        private static double DirichletBetaAlternating(double s)
        {
            if (s > 60)
            {
                // β(s) = 1 - 3⁻ˢ + 5⁻ˢ - …。3⁻ˢ < 2⁻⁹⁵ なら第3項以降は不要
                return 1 - Math.Pow(3, -s);
            }

            // d = ((3+√8)ⁿ + (3+√8)⁻ⁿ)/2、b と c はチェビシェフ多項式の係数の漸化式
            const int n = AlternatingSeriesTerms;
            double d = Math.Pow(3 + Math.Sqrt(8), n);
            d = 0.5 * (d + 1 / d);
            double b = -1;
            double c = -d;
            double sum = 0;
            for (int k = 0; k < n; k++)
            {
                c = b - c;
                sum += c * Math.Pow(2 * k + 1, -s);
                b = (k + n) * (k - n) * b / ((k + 0.5) * (k + 1));
            }

            return sum / d;
        }
    }
}
