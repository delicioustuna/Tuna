using static Tuna.XMath.Constants;
using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        private const double PiSquaredOver8 = 1.2337005501361697;

        // これ未満では CDF をテータ変換級数で、以上では SF を交代級数で直接計算する。
        // 中央値（≈ 0.8276）付近で切り替えるため、補確率を 1 から減算しても桁落ちしない
        private const double KolmogorovSwitch = 0.82;

        // u = π²/(8x²) がこれ以上なら、前因子を掛けても K(x) と密度は 0 にアンダーフローする
        private const double KolmogorovUnderflowExponent = 3000;

        /// <summary>
        /// Computes the cumulative distribution function <c>K(x)</c> of the Kolmogorov distribution.
        /// <para>コルモゴロフ分布の累積分布関数 <c>K(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The evaluation point. / 評価点。</param>
        /// <returns>The value of <c>K(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is <see cref="double.NaN"/>. / <c>K(x)</c> の値。<paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <c>0</c> for <c>x ≤ 0</c> and <c>1</c> at <c>x = +∞</c>.</para>
        /// <para><c>x ≤ 0</c> では <c>0</c>、<c>x = +∞</c> では <c>1</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Kolmogorov_distribution"/>
        public static double KolmogorovCdf(double x)
        {
            if (double.IsNaN(x))
            {
                return x;
            }

            if (x <= 0)
            {
                return 0;
            }

            if (double.IsPositiveInfinity(x))
            {
                return 1;
            }

            KolmogorovCore(x, out double cdf, out _, out _);
            return cdf;
        }

        /// <summary>
        /// Computes the survival function (upper-tail probability) <c>1 - K(x)</c> of the Kolmogorov distribution.
        /// <para>コルモゴロフ分布の生存関数（上側確率）<c>1 - K(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The evaluation point. / 評価点。</param>
        /// <returns>The value of <c>1 - K(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is <see cref="double.NaN"/>. / <c>1 - K(x)</c> の値。<paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Evaluated directly without subtracting <c>K(x)</c> from 1, preserving relative accuracy for large <paramref name="x"/>. Returns <c>1</c> for <c>x ≤ 0</c> and <c>0</c> at <c>x = +∞</c>.</para>
        /// <para><c>1 - K(x)</c> の減算を経由せず直接計算するため、大きな <paramref name="x"/> でも相対精度を保ちます。<c>x ≤ 0</c> では <c>1</c>、<c>x = +∞</c> では <c>0</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Kolmogorov_distribution"/>
        public static double KolmogorovSf(double x)
        {
            if (double.IsNaN(x))
            {
                return x;
            }

            if (x <= 0)
            {
                return 1;
            }

            if (double.IsPositiveInfinity(x))
            {
                return 0;
            }

            KolmogorovCore(x, out _, out double sf, out _);
            return sf;
        }

        /// <summary>
        /// Computes the probability density function <c>K'(x)</c> of the Kolmogorov distribution.
        /// <para>コルモゴロフ分布の確率密度関数 <c>K'(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The evaluation point. / 評価点。</param>
        /// <returns>The value of <c>K'(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is <see cref="double.NaN"/>. / <c>K'(x)</c> の値。<paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <c>0</c> for <c>x ≤ 0</c> and <c>x = +∞</c>.</para>
        /// <para><c>x ≤ 0</c> および <c>x = +∞</c> では <c>0</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Kolmogorov_distribution"/>
        public static double KolmogorovPdf(double x)
        {
            if (double.IsNaN(x))
            {
                return x;
            }

            if (x <= 0 || double.IsPositiveInfinity(x))
            {
                return 0;
            }

            KolmogorovCore(x, out _, out _, out double pdf);
            return pdf;
        }

        /// <summary>
        /// Computes the quantile <c>K⁻¹(p)</c> of the Kolmogorov distribution.
        /// <para>コルモゴロフ分布の分位点 <c>K⁻¹(p)</c> を計算します。</para>
        /// </summary>
        /// <param name="p">The cumulative probability (<c>0 ≤ p ≤ 1</c>). / 累積確率（<c>0 ≤ p ≤ 1</c>）。</param>
        /// <returns>The value <c>x</c> satisfying <c>K(x) = p</c>, or <see cref="double.NaN"/> if <paramref name="p"/> is outside <c>[0, 1]</c> or <see cref="double.NaN"/>. / <c>K(x) = p</c> を満たす <c>x</c>。<paramref name="p"/> が <c>[0, 1]</c> の範囲外または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <c>0</c> at <c>p = 0</c> and <see cref="double.PositiveInfinity"/> at <c>p = 1</c>.</para>
        /// <para><c>p = 0</c> では <c>0</c>、<c>p = 1</c> では <see cref="double.PositiveInfinity"/> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Kolmogorov_distribution"/>
        public static double InverseKolmogorovCdf(double p) => InverseKolmogorov(p, 1 - p);

        /// <summary>
        /// Computes the upper-tail quantile of the Kolmogorov distribution satisfying <c>1 - K(x) = q</c>.
        /// <para>コルモゴロフ分布の上側分位点（<c>1 - K(x) = q</c> を満たす <c>x</c>）を計算します。</para>
        /// </summary>
        /// <param name="q">The upper-tail probability (<c>0 ≤ q ≤ 1</c>). / 上側確率（<c>0 ≤ q ≤ 1</c>）。</param>
        /// <returns>The value <c>x</c> satisfying <c>1 - K(x) = q</c>, or <see cref="double.NaN"/> if <paramref name="q"/> is outside <c>[0, 1]</c> or <see cref="double.NaN"/>. / <c>1 - K(x) = q</c> を満たす <c>x</c>。<paramref name="q"/> が <c>[0, 1]</c> の範囲外または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Evaluated directly from <paramref name="q"/> without forming <c>1 - q</c>, preserving accuracy for small <paramref name="q"/>. Returns <see cref="double.PositiveInfinity"/> at <c>q = 0</c> and <c>0</c> at <c>q = 1</c>.</para>
        /// <para><c>1 - q</c> を経由せず直接計算するため、小さな <paramref name="q"/> でも精度を保ちます。<c>q = 0</c> では <see cref="double.PositiveInfinity"/>、<c>q = 1</c> では <c>0</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Kolmogorov_distribution"/>
        public static double InverseKolmogorovSf(double q) => InverseKolmogorov(1 - q, q);

        // K(x), 1 - K(x), K'(x)（0 < x < ∞）。
        // KolmogorovSwitch 未満では u = π²/(8x²) の正項級数、以上では e^(-2x²) 基準の交代級数で計算する
        private static void KolmogorovCore(double x, out double cdf, out double sf, out double pdf)
        {
            if (x < KolmogorovSwitch)
            {
                double u = PiSquaredOver8 / (x * x);
                if (!(u < KolmogorovUnderflowExponent))
                {
                    // x² のアンダーフロー（u = ∞）を含め、前因子 √(2π)/x を掛けても 0 に丸められる範囲
                    cdf = 0;
                    pdf = 0;
                    sf = 1;
                    return;
                }

                // j = 1 の項は e⁰ = 1。u が大きくても 0·u を作らないよう、和の初期値として扱う
                double sum = 1;
                double derivativeSum = 2 * u - 1;
                for (int j = 2; j < 100; j++)
                {
                    double m = (2.0 * j - 1) * (2.0 * j - 1);
                    double e = Math.Exp(-(m - 1) * u);
                    sum += e;
                    derivativeSum += e * (2 * m * u - 1);
                    if (e <= MachineEpsilon * 0.5 * sum)
                    {
                        break;
                    }
                }

                // e⁻ᵘ が非正規数になる範囲では、前因子との積を 1 つの指数にまとめて途中のアンダーフローを避ける
                double leading = u < 700 ? SqrtTwoPi / x * Math.Exp(-u) : Math.Exp(Math.Log(SqrtTwoPi / x) - u);
                cdf = leading * sum;
                pdf = leading / x * derivativeSum;
                sf = 1 - cdf;
                return;
            }

            if (2 * x > NormalTailUnderflow)
            {
                // e^(-2x²) = e^(-(2x)²/2) が 0 に丸められる範囲（x² のオーバーフローも含む）
                cdf = 1;
                pdf = 0;
                sf = 0;
                return;
            }

            // j = 1 の項は e⁰ = 1 なので、和の初期値として扱う
            double x2 = x * x;
            double alternating = 1;
            double derivativeAlternating = 1;
            for (int j = 2; j < 100; j++)
            {
                double j2 = (double)j * j;
                double e = Math.Exp(-2 * (j2 - 1) * x2);
                double sign = j % 2 == 1 ? 1 : -1;
                alternating += sign * e;
                derivativeAlternating += sign * j2 * e;
                if (j2 * e <= MachineEpsilon * 0.5 * Math.Abs(derivativeAlternating))
                {
                    break;
                }
            }

            // e^(-2x²) = e^(-(2x)²/2)（2x は正確）は (2x)² の丸め誤差を補正し、非正規数になる裾でも丸めを最後の 1 回にする
            sf = ScaledExpMinusHalfSquare(2 * x, 2 * alternating);
            pdf = ScaledExpMinusHalfSquare(2 * x, 8 * x * derivativeAlternating);
            cdf = 1 - sf;
        }

        // K(x) = p（1 - K(x) = q）を満たす x を、小さい方の裾の対数に対するニュートン法（区間保護付き）で求める。
        // 初期値は上側 1 - K ≈ 2e^(-2x²)、下側 K ≈ (√(2π)/x) e^(-π²/(8x²)) の主要項から求める
        private static double InverseKolmogorov(double p, double q)
        {
            if (double.IsNaN(p) || double.IsNaN(q) || p < 0 || q < 0)
            {
                return double.NaN;
            }

            if (p == 0)
            {
                return 0;
            }

            if (q == 0)
            {
                return double.PositiveInfinity;
            }

            bool useUpper = q <= p;
            double target = Math.Log(useUpper ? q : p);
            double x;
            if (useUpper)
            {
                x = Math.Sqrt(-0.5 * (target - Ln2));
            }
            else
            {
                // π²/(8x²) = ln(√(2π)/x) - ln p を不動点反復で解く
                x = 0.5;
                for (int i = 0; i < 4; i++)
                {
                    x = Math.PI / Math.Sqrt(8 * (Math.Log(SqrtTwoPi / x) - target));
                }
            }

            double lower = 0;
            double upper = double.PositiveInfinity;
            for (int iteration = 0; iteration < 100; iteration++)
            {
                KolmogorovCore(x, out double cdf, out double sf, out double pdf);
                double tail = useUpper ? sf : cdf;
                double residual = Math.Log(tail) - target;
                if (residual == 0)
                {
                    return x;
                }

                // K は増加関数なので、上側の裾が大きすぎる（下側の裾が小さすぎる）なら x は解より小さい
                if (useUpper == (residual > 0))
                {
                    lower = x;
                }
                else
                {
                    upper = x;
                }

                // d ln(1 - K)/dx = -K'/(1 - K)、d ln K/dx = K'/K
                double slope = useUpper ? -pdf / sf : pdf / cdf;
                double next = x - residual / slope;

                // 収束の判定は区間の判定より先に行う（x は直前に区間の端になっているので、収束すると next が端に一致する）
                if (Math.Abs(next - x) <= 4 * MachineEpsilon * x)
                {
                    return next;
                }

                if (!(next > lower && next < upper))
                {
                    next = double.IsPositiveInfinity(upper) ? 2 * x
                        : lower > 0 ? Math.Sqrt(lower) * Math.Sqrt(upper)
                        : 0.5 * upper;
                }

                x = next;
            }

            return x;
        }
    }
}
