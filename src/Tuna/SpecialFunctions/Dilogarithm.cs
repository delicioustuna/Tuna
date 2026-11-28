using static Tuna.XMath.Constants;
using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        /// <summary>
        /// Computes the dilogarithm (Spence's function), <c>Li₂(x) = -∫₀ˣ ln(1 - t) / t dt = Σ xᵏ / k²</c>.
        /// <para>二重対数関数（スペンス関数）<c>Li₂(x) = -∫₀ˣ ln(1 - t) / t dt = Σ xᵏ / k²</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The real input value (<c>x ≤ 1</c>). / 実数の入力値（<c>x ≤ 1</c>）。</param>
        /// <returns>The real value of <c>Li₂(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is greater than 1 or <see cref="double.NaN"/>. / <c>Li₂(x)</c> の実数値。<paramref name="x"/> が 1 より大きい場合または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Dilogarithm"/>
        public static double Dilogarithm(double x)
        {
            if (double.IsNaN(x) || x > 1)
            {
                return double.NaN;
            }

            if (x == 1)
            {
                return PiSquaredOver6;
            }

            if (x == 0)
            {
                return x;
            }

            if (double.IsNegativeInfinity(x))
            {
                return x;
            }

            if (x < -1)
            {
                // Li₂(x) = -π²/6 - ln²(-x)/2 - Li₂(1/x)
                double ln = Math.Log(-x);
                return -PiSquaredOver6 - 0.5 * ln * ln - DilogarithmCore(1 / x);
            }

            if (x > 0.5)
            {
                // Li₂(x) = π²/6 - ln x ln(1-x) - Li₂(1-x)
                double y = 1 - x;
                return PiSquaredOver6 - Math.Log(x) * Math.Log(y) - DilogarithmCore(y);
            }

            return DilogarithmCore(x);
        }

        // -1 ≤ x ≤ 1/2 での Li₂(x) = Σ Bₙ uⁿ⁺¹/(n+1)! (u = -ln(1-x), |u| ≤ ln 2)
        private static double DilogarithmCore(double x)
        {
            double u = -Log1p(-x);
            double u2 = u * u;

            // u - u²/4 + Σ B₂ₖ u^(2k+1)/(2k+1)!
            double power = u;
            double sum = 0;
            for (int k = 1; k < BernoulliEven.Length && 2 * k + 1 < FactorialTable.Length; k++)
            {
                power *= u2;
                double term = BernoulliEven[k] * power / FactorialTable[2 * k + 1];
                sum += term;
                if (Math.Abs(term) <= MachineEpsilon * Math.Abs(u) * 0.25)
                {
                    break;
                }
            }

            return u - 0.25 * u2 + sum;
        }
    }
}
