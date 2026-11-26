namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        /// <summary>
        /// Computes the generalized exponential integral <c>E_p(x)</c> for real order <paramref name="p"/>.
        /// <para>実数次数の一般化指数積分 <c>E_p(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="p">The real order. / 実数次数。</param>
        /// <param name="x">The non-negative input value (<c>x ≥ 0</c>). / 非負の入力値（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>E_p(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> &lt; 0, <paramref name="p"/> is non-finite, or <paramref name="x"/> is <see cref="double.NaN"/>. / <c>E_p(x)</c> の値。<paramref name="x"/> &lt; 0、<paramref name="p"/> が有限でない場合、または <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>At <c>x = 0</c>, returns <c>1 / (p - 1)</c> for <c>p &gt; 1</c> and <see cref="double.PositiveInfinity"/> otherwise.</para>
        /// <para><c>x = 0</c> では <c>p &gt; 1</c> のとき <c>1 / (p - 1)</c>、それ以外のとき <see cref="double.PositiveInfinity"/> を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/8.19.E1"/>
        public static double ExpIntegralE(double p, double x)
        {
            if (double.IsNaN(p) || double.IsNaN(x) || x < 0 || double.IsInfinity(p))
            {
                return double.NaN;
            }

            if (x == 0)
            {
                return p > 1 ? 1 / (p - 1) : double.PositiveInfinity;
            }

            if (double.IsPositiveInfinity(x))
            {
                return 0;
            }

            if (IsInteger(p) && p >= 0 && p < int.MaxValue)
            {
                return ExpIntegralE((int)p, x);
            }

            if (x > 1 && (p >= 0 || x + p > 1))
            {
                return GeneralizedExpIntegralFraction(p, x);
            }

            if (p >= HugeExpIntegralOrder)
            {
                // 大きな次数の漸近形 E_p(x) = e^(-x)/(x + p)·(1 + O(1/p))（DLMF 8.20.3、λ = x/p）。p ≥ 2⁶⁰ では省略項は倍精度で無視できる。
                // 下の級数表示では a = 1 - p が負の整数（2⁵³ 以上の double はすべて整数）になり、打ち消すはずの Γ(a) の極を個別に評価して NaN になる
                return Math.Exp(-x) / (x + p);
            }

            double a = 1 - p;
            if (a > 0)
            {
                // p < 1。a が大きいと Γ(a, x) と x^(p-1) の一方がオーバーフローまたはアンダーフローしうるため、その場合はスケール分離で掛ける
                double power = Math.Pow(x, p - 1);
                double upper = UpperIncompleteGamma(a, x);
                double product = power * upper;
                if (double.IsFinite(power) && double.IsFinite(upper) && power != 0 && upper != 0)
                {
                    return product;
                }

                // x^(p-1) Γ(a) Q(a, x) の x^(p-1) と Γ(a) を仮数と指数に分けて掛ける（対数で掛けると |ln Γ(a)|·ε の誤差が乗る）。
                // この分岐は x ≤ a 程度なので Q(a, x) はアンダーフローしない
                double e = p - 1;
                double n = Math.Round(e);
                double mantissa = ScaledGammaSigned(a, out long exponent) * ScaledPower(x, (long)n, out long powerExponent) * Math.Pow(x, e - n);
                return ScaleB(mantissa * RegularizedGammaQ(a, x), exponent + powerExponent);
            }

            // p > 1、x ≤ 1。E_p(x) = x^(p-1) Γ(a) - Σ (-x)ⁿ/(n! (a+n))
            double nearest = Math.Round(-a);
            double epsilon = a + nearest;
            int pole = nearest < FactorialTable.Length && Math.Abs(epsilon) <= 0.25 ? (int)nearest : -1;
            double sum = 0;
            double term = 1;
            for (int n = 0; n < MaxIterations; n++)
            {
                if (n != pole)
                {
                    double t = term / (a + n);
                    sum += t;
                    if (n > pole && n > 0 && Math.Abs(t) <= Epsilon * Math.Abs(sum))
                    {
                        break;
                    }
                }

                term *= -x / (n + 1);
            }

            // 極の項：pole = m なら x^(p-1) [Γ(a) - (-1)ᵐ x^(a+m)/(m!(a+m))]、x^(p-1) x^(a+m) = xᵐ でオーバーフローしない
            double poleTerms = pole < 0 ? ReciprocalGammaSafeProduct(a, x, p - 1) : Math.Pow(x, p - 1) * UpperGammaPoleTerms(pole, epsilon, x);
            return poleTerms - sum;
        }

        // これ以上の次数（2⁶⁰）では連分数を使わない側を漸近形で評価する
        private const double HugeExpIntegralOrder = 1.152921504606846976e18;

        // x^e Γ(a)（a < 0, 0 < x ≤ 1, e > 0）を途中オーバーフローなしで計算する
        private static double ReciprocalGammaSafeProduct(double a, double x, double e)
        {
            double gamma = Gamma(a);
            if (double.IsFinite(gamma) && gamma != 0)
            {
                return Math.Pow(x, e) * gamma;
            }

            return SignGamma(a) * Math.Exp(e * Math.Log(x) + LogGamma(a));
        }

        // E_p(x) の連分数（x > 1、ExpIntegralE(int, double) の実数次数版）
        private static double GeneralizedExpIntegralFraction(double p, double x)
        {
            double pm1 = p - 1;
            double b = x + p;
            double c = 1 / Tiny;
            double d = 1 / b;
            double h = d;
            for (int i = 1; i < MaxIterations; i++)
            {
                double an = -i * (pm1 + i);
                b += 2;
                d = an * d + b;
                if (d == 0)
                {
                    d = Tiny;
                }

                c = b + an / c;
                if (c == 0)
                {
                    c = Tiny;
                }

                d = 1 / d;
                double delta = c * d;
                h *= delta;
                if (Math.Abs(delta - 1) <= Epsilon)
                {
                    break;
                }
            }

            return h * Math.Exp(-x);
        }
    }
}
