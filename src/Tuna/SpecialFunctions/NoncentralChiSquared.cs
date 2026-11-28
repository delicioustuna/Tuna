using static Tuna.XMath.Constants;
using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        /// <summary>
        /// Computes the probability density function <c>f(x; k, λ)</c> of the noncentral chi-squared distribution.
        /// <para>非心 χ² 分布の確率密度関数 <c>f(x; k, λ)</c> を計算します。</para>
        /// </summary>
        /// <param name="k">The degrees of freedom (<c>k &gt; 0</c>, finite). / 自由度（有限な <c>k &gt; 0</c>）。</param>
        /// <param name="lambda">The noncentrality parameter (<c>λ ≥ 0</c>, finite). / 非心度（有限な <c>λ ≥ 0</c>）。</param>
        /// <param name="x">The evaluation point. / 評価点。</param>
        /// <returns>The density <c>f(x; k, λ)</c>, or <see cref="double.NaN"/> if parameters are out of domain or any argument is <see cref="double.NaN"/>. / 確率密度 <c>f(x; k, λ)</c>。パラメータが定義域外、またはいずれかの引数が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Reduces to the central chi-squared density at <c>λ = 0</c>. Returns <c>0</c> for <c>x &lt; 0</c> or <c>x = +∞</c>; at <c>x = 0</c>, returns <see cref="double.PositiveInfinity"/> for <c>k &lt; 2</c>, <c>e^(-λ/2) / 2</c> for <c>k = 2</c>, and <c>0</c> for <c>k &gt; 2</c>.</para>
        /// <para><c>λ = 0</c> では中心 χ² 分布の密度に一致します。<c>x &lt; 0</c> および <c>x = +∞</c> では <c>0</c> を返し、<c>x = 0</c> では <c>k &lt; 2</c> で <see cref="double.PositiveInfinity"/>、<c>k = 2</c> で <c>e^(-λ/2) / 2</c>、<c>k &gt; 2</c> で <c>0</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution#Density"/>
        public static double NoncentralChiSquaredPdf(double k, double lambda, double x) =>
            Math.Exp(NoncentralChiSquaredLogPdf(k, lambda, x));

        /// <summary>
        /// Computes the natural logarithm of the probability density function <c>ln f(x; k, λ)</c> of the noncentral chi-squared distribution.
        /// <para>非心 χ² 分布の確率密度関数の自然対数 <c>ln f(x; k, λ)</c> を計算します。</para>
        /// </summary>
        /// <param name="k">The degrees of freedom (<c>k &gt; 0</c>, finite). / 自由度（有限な <c>k &gt; 0</c>）。</param>
        /// <param name="lambda">The noncentrality parameter (<c>λ ≥ 0</c>, finite). / 非心度（有限な <c>λ ≥ 0</c>）。</param>
        /// <param name="x">The evaluation point. / 評価点。</param>
        /// <returns>The log-density <c>ln f(x; k, λ)</c>, or <see cref="double.NaN"/> if parameters are out of domain or any argument is <see cref="double.NaN"/>. / 対数密度 <c>ln f(x; k, λ)</c>。パラメータが定義域外、またはいずれかの引数が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <see cref="double.NegativeInfinity"/> for <c>x &lt; 0</c> or <c>x = +∞</c>, and the logarithm of the limiting value at <c>x = 0</c>. Remains finite even where the density overflows or underflows.</para>
        /// <para><c>x &lt; 0</c> および <c>x = +∞</c> では <see cref="double.NegativeInfinity"/>、<c>x = 0</c> では極限値の対数を返します。密度がオーバーフローまたはアンダーフローする領域でも有限の値を保ちます。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution#Density"/>
        public static double NoncentralChiSquaredLogPdf(double k, double lambda, double x)
        {
            if (double.IsNaN(k) || double.IsNaN(lambda) || double.IsNaN(x) || !(k > 0) || !(lambda >= 0)
                || double.IsInfinity(k) || double.IsInfinity(lambda))
            {
                return double.NaN;
            }

            if (x < 0 || double.IsPositiveInfinity(x))
            {
                return double.NegativeInfinity;
            }

            // a = k/2 は対応するガンマ分布の形状パラメータ、ν = a - 1 はベッセル関数の次数（ν > -1）
            double a = 0.5 * k;
            if (x == 0)
            {
                // f(x) ~ e^(-λ/2) (x/2)^(a-1)/(2Γ(a))（x → 0）
                return k < 2 ? double.PositiveInfinity : k > 2 ? double.NegativeInfinity : -Ln2 - 0.5 * lambda;
            }

            if (lambda == 0)
            {
                return LogCentralChiSquaredPdf(a, x);
            }

            // ポアソン混合の形 f = e^(-λ/2) f_central(x; k) Σ tʲ/(j! (a)ⱼ)（t = λx/4）は全項が正で桁落ちせず、
            // ベッセル関数の形で λ が小さいと (x/λ)^(ν/2) と I_ν(√(λx)) の中の ln λ が打ち消し合うのも避けられる。
            // t ≤ max(1, a) なら項の比が 1/j 以下になり、20 項程度で収束する
            double t = 0.25 * lambda * x;
            if (t <= Math.Max(1, a))
            {
                if (a < t)
                {
                    // 小さい a では第 1 項 t/a が大きく、Γ(a) と (a)ⱼ の a が打ち消し合う（a が 0 に丸まると ∞ - ∞）。
                    // Σ = 1 + (t/a) S₁、S₁ = Σ tᵐ/((m+1)! (a+1)ₘ) と分け、f_central(x; k)·t/a = f_central(x; k+2)·λ/2 を使って
                    // f = e^(-λ/2) f_central(x; k+2) (λ/2) (a/t + S₁) とする（a/t < 1 なのでオーバーフローしない）
                    return LogCentralChiSquaredPdf(a + 1, x) + Math.Log(0.5 * lambda) - 0.5 * lambda
                        + Math.Log(a / t + NoncentralChiSquaredShiftedSeriesSum(a, t));
                }

                return LogCentralChiSquaredPdf(a, x) - 0.5 * lambda + Math.Log(NoncentralChiSquaredSeriesSum(a, t));
            }

            // ln f = -ln 2 + (ν/2) ln(x/λ) - (√x - √λ)²/2 + ln(e^(-z) I_ν(z))（z = √(λx)）。
            // -(x + λ)/2 + z = -(√x - √λ)²/2 とし、√x - √λ = (x - λ)/(√x + √λ) で近い平方根の減算を避ける
            double nu = a - 1;
            double sqrtX = Math.Sqrt(x);
            double sqrtLambda = Math.Sqrt(lambda);
            double z = sqrtX * sqrtLambda;
            double d = (x - lambda) / (sqrtX + sqrtLambda);
            double ratio = x / lambda;
            double logRatio = ratio >= MinNormal && double.IsFinite(ratio) ? Math.Log(ratio) : Math.Log(x) - Math.Log(lambda);
            return -Ln2 + 0.5 * nu * logRatio - 0.5 * d * d + LogBesselICore(nu, z, scaled: true);
        }

        // 中心 χ² 分布の対数密度 ln((x/2)^(a-1) e^(-x/2)/(2Γ(a)))（a = k/2 > 0、0 < x < ∞）。
        // ガンマ分布の前因子 yᵃ e⁻ʸ/Γ(a)（y = x/2）を用いて ln(yᵃ e⁻ʸ/Γ(a)) - ln x として計算する
        private static double LogCentralChiSquaredPdf(double a, double x)
        {
            if (x < 2 * MinNormal)
            {
                // x/2 は非正規数の x で下位ビットを失うため、対数の差で求める
                double logHalf = Math.Log(x) - Ln2;
                return (a - 1) * logHalf - 0.5 * x - Ln2 - LogGamma(a);
            }

            return LogGammaPrefix(a, 0.5 * x) - Math.Log(x);
        }

        // ポアソン混合の和 Σ tʲ/(j! (a)ⱼ)（a > 0、0 ≤ t ≤ max(1, a)。値は 1 以上）
        private static double NoncentralChiSquaredSeriesSum(double a, double t)
        {
            double term = 1;
            double sum = 1;
            for (int j = 1; j < 1000; j++)
            {
                // a + (j - 1) の順にする（(a + j) - 1 は小さい a の下位ビットを失う）
                term *= t / (j * (a + (j - 1)));
                sum += term;
                if (term <= sum * MachineEpsilon * 0.5)
                {
                    break;
                }
            }

            return sum;
        }

        // 第 1 項を除いて t/a で割ったポアソン混合の和 S₁ = Σₘ tᵐ/((m+1)! (a+1)ₘ)（a ≥ 0、0 < t ≤ 1。a = 0 でも有限）
        private static double NoncentralChiSquaredShiftedSeriesSum(double a, double t)
        {
            double term = 1;
            double sum = 1;
            for (int m = 1; m < 1000; m++)
            {
                term *= t / ((m + 1) * (a + m));
                sum += term;
                if (term <= sum * MachineEpsilon * 0.5)
                {
                    break;
                }
            }

            return sum;
        }
    }
}
