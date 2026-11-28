using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        /// <summary>
        /// Computes the Lerch transcendent <c>Φ(z, s, a)</c>.
        /// <para>レルヒの超越関数 <c>Φ(z, s, a)</c> を計算します。</para>
        /// </summary>
        /// <param name="z">The first argument (<c>z ≤ 1</c>; values <c>z &lt; -1</c> require <c>s ≥ 0</c>). / 第 1 引数（<c>z ≤ 1</c>。<c>z &lt; -1</c> では <c>s ≥ 0</c> のみ定義されます）。</param>
        /// <param name="s">The order (real number). / 次数（実数）。</param>
        /// <param name="a">The parameter (<c>a &gt; 0</c>). / パラメータ（<c>a &gt; 0</c>）。</param>
        /// <returns>The value of <c>Φ(z, s, a)</c>, or <see cref="double.NaN"/> if arguments are out of domain or <see cref="double.NaN"/>. / <c>Φ(z, s, a)</c> の値。引数が定義域外または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>For <c>z = 1</c>, reduces to the Hurwitz zeta function <c>ζ(s, a)</c> (returning <see cref="double.PositiveInfinity"/> at <c>s = 1</c> and analytic continuation for <c>s &lt; 0</c>). For <c>z &lt; -1</c> and <c>s &gt; 0</c>, evaluated via the real-valued integral continuation (<see href="https://dlmf.nist.gov/25.14.E5">DLMF 25.14.5</see>). Returns <see cref="double.NaN"/> on the branch cut <c>z &gt; 1</c>, for <c>a ≤ 0</c>, or for <c>z &lt; -1</c> with <c>s &lt; 0</c>.</para>
        /// <para><c>z = 1</c> ではフルヴィッツのゼータ関数 <c>ζ(s, a)</c>（<c>s = 1</c> で <see cref="double.PositiveInfinity"/>、<c>s &lt; 0</c> は解析接続）に一致します。<c>z &lt; -1</c> かつ <c>s &gt; 0</c> では積分表示（<see href="https://dlmf.nist.gov/25.14.E5">DLMF 25.14.5</see>）による実数値の解析接続を返します。分岐切断 <c>z &gt; 1</c>、<c>a ≤ 0</c>、および <c>z &lt; -1</c> かつ <c>s &lt; 0</c> では <see cref="double.NaN"/> を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/25.14.E1"/>
        public static double LerchPhi(double z, double s, double a)
        {
            if (double.IsNaN(z) || double.IsNaN(s) || double.IsNaN(a) || !(a > 0) || double.IsNegativeInfinity(s) || z > 1)
            {
                return double.NaN;
            }

            if (double.IsPositiveInfinity(s))
            {
                // n = 0 の項 a⁻ˢ だけが残りうる（n ≥ 1 では n + a > 1）
                return a > 1 ? 0 : a == 1 ? 1 : double.PositiveInfinity;
            }

            if (double.IsPositiveInfinity(a))
            {
                return s > 0 ? 0 : double.NaN;
            }

            if (z == 0)
            {
                return Math.Pow(a, -s);
            }

            if (s == 0)
            {
                return z == 1 ? 0.5 - a : double.IsNegativeInfinity(z) ? 0 : 1 / (1 - z);
            }

            if (z < -1)
            {
                if (s < 0)
                {
                    return double.NaN;
                }

                return double.IsNegativeInfinity(z) ? 0 : LerchIntegral(-z, s, a);
            }

            if (z > 0)
            {
                return LerchPositive(z, s, a);
            }

            if (z >= -0.5)
            {
                return LerchSeries(z, s, a);
            }

            if (s > 0)
            {
                return LerchAlternating(-z, s, a);
            }

            if (z == -1 && IsInteger(s) && s > -int.MaxValue)
            {
                // Φ(-1, -n, a) = Eₙ(a)/2（オイラー多項式。Eₙ(x) = 2 Σ (-1)ᵏ (k + x)ⁿ をアーベル和とみた解析接続）
                return 0.5 * EulerPolynomial((int)-s, a);
            }

            // s < 0：偶数項と奇数項に分ける Φ(z, s, a) = 2⁻ˢ (Φ(z², s, a/2) + z Φ(z², s, (a+1)/2))
            double z2 = z * z;
            return Math.Pow(2, -s) * (LerchPositive(z2, s, 0.5 * a) + z * LerchPositive(z2, s, 0.5 * (a + 1)));
        }

        // 0 < z ≤ 1 での Φ(z, s, a)（s ≠ 0）
        private static double LerchPositive(double z, double s, double a)
        {
            if (z == 1)
            {
                if (s > 0)
                {
                    return HurwitzZeta(s, a);
                }

                if (IsInteger(s) && s > -int.MaxValue)
                {
                    // ζ(-n, a) = -B_(n+1)(a)/(n+1)（DLMF 25.11.14）
                    int n = (int)-s;
                    return -BernoulliPolynomial(n + 1, a) / (n + 1);
                }
            }

            if (z <= 0.5)
            {
                return LerchSeries(z, s, a);
            }

            // z - 1 は丸められない（z ≥ 1/2）
            double lambda = -Log1p(z - 1);
            if (s > 0 && lambda * 200 + s * Log1p(200 / a) >= 40)
            {
                // (n + a)⁻ˢ と zⁿ の減衰で 200 項程度以内に収束する
                return LerchSeries(z, s, a);
            }

            return LerchEulerMaclaurin(z, lambda, s, a);
        }

        // 級数 a⁻ˢ Σ zⁿ (1 + n/a)⁻ˢ（|z| < 1）
        private static double LerchSeries(double z, double s, double a)
        {
            double sum = 1;
            double power = 1;
            for (int n = 1; n < MaxIterations; n++)
            {
                power *= z;
                double term = power * Math.Exp(-s * Log1p(n / a));
                sum += term;

                // s < 0 では項がいったん増えるので、項の比 |z| (1 + 1/(n+a))^(-s) が 1 未満になってから打ち切る
                if (Math.Abs(term) <= 0.25 * MachineEpsilon * Math.Abs(sum) && (s >= 0 || Math.Abs(z) * Math.Exp(-s * Log1p(1 / (n + a))) < 1))
                {
                    break;
                }

                if (power == 0)
                {
                    break;
                }
            }

            return Math.Pow(a, -s) * sum;
        }

        // 交代級数 Σ (-1)ⁿ yⁿ (n + a)⁻ˢ（1/2 < y ≤ 1、s > 0）を Cohen–Villegas–Zagier のアルゴリズム 1 で加速する。
        // bₙ = yⁿ (n + a)⁻ˢ = ∫ tⁿ dμ(t)（正の測度）のため、誤差は (3 + √8)⁻ⁿ b₀ 程度
        private static double LerchAlternating(double y, double s, double a)
        {
            const int Terms = 24;
            double d = Math.Pow(3 + Math.Sqrt(8), Terms);
            d = 0.5 * (d + 1 / d);
            double b = -1;
            double c = -d;
            double sum = 0;
            double power = 1;
            for (int k = 0; k < Terms; k++)
            {
                c = b - c;
                sum += c * power * Math.Exp(-s * Log1p(k / a));
                b = (double)(k + Terms) * (k - Terms) * b / ((k + 0.5) * (k + 1));
                power *= y;
            }

            return Math.Pow(a, -s) * sum / d;
        }

        // オイラー・マクローリンの和公式による Φ(z, s, a)（1/2 < z ≤ 1、λ = -ln z。λ = 0 では s ≠ 1）。
        // Σₙ₌₀^(N-1) f(n) + ∫_N^∞ f + f(N)/2 + Σⱼ B₂ⱼ/(2j)! f(N) g₂ⱼ₋₁、gₖ = Σᵢ C(k,i) λ^(k-i) (s)ᵢ w⁻ⁱ、w = N + a
        private static double LerchEulerMaclaurin(double z, double lambda, double s, double a)
        {
            const int MaxOrder = 30;

            // w ≥ |s| + 30 なら (s)ᵢ w⁻ⁱ（i ≤ 2·MaxOrder）が抑えられ、補正項は (λ + 1)/(2π) 程度の比で減る。
            // ただし z = 1（λ = 0）、s < 0 では先頭の和と積分 -w^(1-s)/(1-s) が打ち消し合って ζ(s, a) になる（相対誤差 ≈ ε w^(1-s)/|ζ(s, a)|）ので、
            // w は漸近展開の最小項 e^(-2πw) w^(1-s) が十分小さく、補正項の比 ((|s| + 2j)/(2πw))² も抑えられる w ≥ max(7, |s|/3) に留める
            double count = lambda == 0 && s < 0
                ? Math.Max(0, Math.Ceiling(Math.Max(7, -s / 3) - a))
                : Math.Max(0, Math.Ceiling(Math.Abs(s) + 30 - a));
            if (count > int.MaxValue)
            {
                return double.NaN;
            }

            double sum = 0;
            double power = 1;
            for (int n = 0; n < count; n++)
            {
                sum += power * Math.Pow(n + a, -s);
                power *= z;
            }

            double w = count + a;
            double endpoint = power * Math.Pow(w, -s);
            double integral;
            if (lambda == 0)
            {
                integral = w * endpoint / (s - 1);
            }
            else
            {
                integral = w * endpoint * LerchScaledExpIntegral(s, lambda * w);
            }

            // hᵢ = (s)ᵢ w⁻ⁱ と λ のべき
            Span<double> h = stackalloc double[2 * MaxOrder];
            Span<double> lambdaPower = stackalloc double[2 * MaxOrder];
            h[0] = 1;
            lambdaPower[0] = 1;
            for (int i = 1; i < h.Length; i++)
            {
                h[i] = h[i - 1] * (s + i - 1) / w;
                lambdaPower[i] = lambdaPower[i - 1] * lambda;
            }

            double total = sum + integral + 0.5 * endpoint;
            double correction = 0;
            for (int j = 1; j < MaxOrder; j++)
            {
                int k = 2 * j - 1;
                double g = 0;
                for (int i = 0; i <= k; i++)
                {
                    g += FactorialTable[k] / (FactorialTable[i] * FactorialTable[k - i]) * lambdaPower[k - i] * h[i];
                }

                double term = BernoulliEven[j] / FactorialTable[2 * j] * endpoint * g;
                correction += term;
                if (Math.Abs(term) <= 0.125 * MachineEpsilon * Math.Abs(total + correction))
                {
                    break;
                }
            }

            return total + correction;
        }

        // DLMF 8.19.17: eˣ E_p(x) = ∫₀^∞ e^(-xu) (1 + u)^(-p) du（x > 0）。
        // x ≥ 1（p < 0 では x + p > 1 も）では連分数を Lentz 法で評価し、eˣ と E_p(x) の独立したオーバーフローやアンダーフローを避ける
        private static double LerchScaledExpIntegral(double p, double x)
        {
            if (x < 1 || (p < 0 && x + p <= 1))
            {
                return Math.Exp(x) * ExpIntegralE(p, x);
            }

            double b = x + p;
            double c = 1 / Tiny;
            double d = 1 / b;
            double f = d;
            for (int k = 1; k < MaxIterations; k++)
            {
                double an = -k * (p + k - 1.0);
                b += 2;
                d = 1 / (an * d + b);
                c = b + an / c;
                double delta = c * d;
                f *= delta;
                if (Math.Abs(delta - 1) <= MachineEpsilon)
                {
                    break;
                }
            }

            return f;
        }

        // z = -y < -1、s > 0 での Φ = a⁻ˢ/Γ(s) ∫₀^∞ u^(s-1) e⁻ᵘ/(1 + y e^(-u/a)) du。
        // u = exp((π/2) sinh τ) の二重指数型台形則で、刻み h を半減させながら新規格子点のみ加算して収束させる
        private static double LerchIntegral(double y, double s, double a)
        {
            double logGamma = LogGamma(s);
            double previous = double.NaN;
            double total = 0;
            double h = 0.5;
            for (int level = 0; level < 8; level++)
            {
                // level 0 はすべての格子点、以降は新しい（奇数番目の）点だけを足す
                int start = level == 0 ? 0 : 1;
                int step = level == 0 ? 1 : 2;
                double partial = level == 0 ? LerchIntegrand(0, y, s, a, logGamma) : 0;
                for (int sign = -1; sign <= 1; sign += 2)
                {
                    for (int j = Math.Max(start, 1); j < 100000; j += step)
                    {
                        double term = LerchIntegrand(sign * j * h, y, s, a, logGamma);
                        partial += term;
                        if (term <= 1e-20 * partial && j * h > 1)
                        {
                            break;
                        }
                    }
                }

                total = level == 0 ? partial * h : 0.5 * total + partial * h;
                if (level > 0 && Math.Abs(total - previous) <= 1e-10 * Math.Abs(total))
                {
                    // 二重指数型台形則は刻みを半分にすると誤差がおよそ 2 乗になるため、差が 1e-10 で十分収束している
                    break;
                }

                previous = total;
                h *= 0.5;
            }

            return Math.Pow(a, -s) * total;
        }

        // 二重指数型台形則の被積分関数 uˢ e⁻ᵘ (π/2) cosh τ / (Γ(s)(1 + y e^(-u/a)))（u = exp((π/2) sinh τ)）。
        // uˢ e⁻ᵘ/Γ(s) はオーバーフローを避けて対数で組み立てる（s が小さいと左の裾 uˢ の減衰が遅く、τ ≈ -9 まで寄与する）
        private static double LerchIntegrand(double tau, double y, double s, double a, double logGamma)
        {
            double logU = 0.5 * Math.PI * Math.Sinh(tau);
            double u = Math.Exp(logU);
            if (double.IsInfinity(u))
            {
                return 0;
            }

            // u がアンダーフローしても uˢ = exp(s ln u) は s が小さいと無視できないため、ln u から組み立てる
            double log = s * logU - u - logGamma + Math.Log(0.5 * Math.PI * Math.Cosh(tau)) - Log1p(y * Math.Exp(-u / a));
            return Math.Exp(log);
        }
    }
}
