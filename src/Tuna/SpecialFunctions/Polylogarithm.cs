using System.Numerics;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // 反転公式による Li_s(-eˡ) の漸近展開（非整数 s）を使う ℓ の下限（ℓ = 40 で打ち切り誤差は相対 6×10⁻¹⁸ 程度）
        private const double PolylogAsymptoticThreshold = 38;

        // 冪級数 Σ xᵏ/kˢ を直接加算する際の最大項数
        private const int PolylogDirectMaxTerms = 64;

        // Cohen–Villegas–Zagier 交代級数加速の項数（誤差は初項の 4·(3+√8)⁻ⁿ ≈ 2×10⁻¹⁸ 倍以下）
        private const int PolylogAlternatingTerms = 24;

        /// <summary>
        /// Computes the polylogarithm <c>Li_s(x)</c> for real order <paramref name="s"/> and real argument <paramref name="x"/>.
        /// <para>実数の次数 <paramref name="s"/> と実数引数 <paramref name="x"/> に対する多重対数関数 <c>Li_s(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="s">The order. / 次数。</param>
        /// <param name="x">The argument (<c>x &lt;= 1</c>). / 引数（<c>x ≤ 1</c>）。</param>
        /// <returns>The value of <c>Li_s(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is outside the real-valued domain. / <c>Li_s(x)</c> の値。実数値の定義域外では <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>
        /// For <c>x = 1</c>, returns <c>ζ(s)</c> when <c>s &gt; 1</c>, and returns the left-hand limit
        /// <see cref="double.PositiveInfinity"/> when <c>s &lt;= 1</c> rather than the analytic continuation through <c>x = 1</c>.
        /// Returns <see cref="double.NaN"/> for <c>x &gt; 1</c> (where the principal value is complex) or <c>s = -∞</c>.
        /// At <c>x = -∞</c>, returns <see cref="double.NegativeInfinity"/> for <c>s &gt; 0</c>, <c>-1</c> for <c>s = 0</c>, and <c>0</c> for <c>s &lt; 0</c>.
        /// </para>
        /// <para>
        /// <c>x = 1</c> では <c>s &gt; 1</c> のとき <c>ζ(s)</c> を返し、<c>s ≤ 1</c> のときは <c>x = 1</c> における解析接続ではなく左極限 <c>x → 1⁻</c> の <see cref="double.PositiveInfinity"/> を返します。
        /// <c>x &gt; 1</c>（主値が複素数になる領域）および <c>s = -∞</c> では <see cref="double.NaN"/> を返します。
        /// <c>x = -∞</c> では <c>s &gt; 0</c> で <see cref="double.NegativeInfinity"/>、<c>s = 0</c> で <c>-1</c>、<c>s &lt; 0</c> で <c>0</c> を返します。
        /// </para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/25.12#ii"/>
        public static double Polylogarithm(double s, double x)
        {
            if (double.IsNaN(s) || double.IsNaN(x) || x > 1 || double.IsNegativeInfinity(s))
            {
                return double.NaN;
            }

            if (x == 0 || double.IsPositiveInfinity(s))
            {
                return x;
            }

            if (x == 1)
            {
                return s > 1 ? Zeta(s) : double.PositiveInfinity;
            }

            if (double.IsNegativeInfinity(x))
            {
                return s > 0 ? double.NegativeInfinity : s == 0 ? -1 : 0;
            }

            if (s == 1)
            {
                return -Log1p(-x);
            }

            if (Math.Abs(s) < 1e-17)
            {
                return x / (1 - x);
            }

            if (s == 2)
            {
                return Dilogarithm(x);
            }

            return x > 0 ? PolylogPositive(s, x, Math.Log(x)) : PolylogNegative(s, -x, Math.Log(-x));
        }

        /// <summary>
        /// Computes the complete Fermi–Dirac integral <c>F_j(x) = -Li_(j+1)(-e^x)</c>.
        /// <para>完全 Fermi–Dirac 積分 <c>F_j(x) = -Li_(j+1)(-eˣ)</c> を計算します。</para>
        /// </summary>
        /// <param name="j">The order. / 次数。</param>
        /// <param name="x">The argument. / 引数。</param>
        /// <returns>The value of <c>F_j(x)</c>, or <see cref="double.NaN"/> if either argument is <see cref="double.NaN"/> or <paramref name="j"/> is <see cref="double.NegativeInfinity"/>. / <c>F_j(x)</c> の値。いずれかの引数が <see cref="double.NaN"/> または <paramref name="j"/> が <see cref="double.NegativeInfinity"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>
        /// Uses the normalized convention <c>F_j(x) = (1 / Γ(j + 1)) ∫_0^∞ t^j / (e^(t - x) + 1) dt</c> for <c>j &gt; -1</c>,
        /// extended to <c>j &lt;= -1</c> by analytic continuation via <c>-Li_(j+1)(-e^x)</c>.
        /// Returns <c>0</c> at <c>x = -∞</c>; at <c>x = +∞</c>, returns <see cref="double.PositiveInfinity"/> for <c>j &gt; -1</c>, <c>1</c> for <c>j = -1</c>, and <c>0</c> for <c>j &lt; -1</c>.
        /// </para>
        /// <para>
        /// <c>j &gt; -1</c> では正規化定義 <c>F_j(x) = (1/Γ(j+1)) ∫₀^∞ tʲ/(eᵗ⁻ˣ + 1) dt</c> を採用し、<c>j ≤ -1</c> は <c>-Li_(j+1)(-eˣ)</c> による解析接続で定義します。
        /// <c>x = -∞</c> では <c>0</c> を返し、<c>x = +∞</c> では <c>j &gt; -1</c> で <see cref="double.PositiveInfinity"/>、<c>j = -1</c> で <c>1</c>、<c>j &lt; -1</c> で <c>0</c> を返します。
        /// </para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/25.12#iii"/>
        public static double FermiDirac(double j, double x)
        {
            if (double.IsNaN(j) || double.IsNaN(x) || double.IsNegativeInfinity(j))
            {
                return double.NaN;
            }

            double s = j + 1;
            if (double.IsNegativeInfinity(x))
            {
                return 0;
            }

            if (double.IsPositiveInfinity(x))
            {
                return s > 0 ? double.PositiveInfinity : s == 0 ? 1 : 0;
            }

            if (double.IsPositiveInfinity(s))
            {
                // Li_∞(-eˣ) = -eˣ
                return Math.Exp(x);
            }

            // 0 は +0 で返す（-0 を作らない）
            return 0.0 - PolylogNegative(s, Math.Exp(x), x);
        }

        /// <summary>
        /// Computes the complete Bose–Einstein integral <c>G_j(x) = Li_(j+1)(e^x)</c>.
        /// <para>完全 Bose–Einstein 積分 <c>G_j(x) = Li_(j+1)(eˣ)</c> を計算します。</para>
        /// </summary>
        /// <param name="j">The order. / 次数。</param>
        /// <param name="x">The argument (<c>x &lt;= 0</c>). / 引数（<c>x ≤ 0</c>）。</param>
        /// <returns>The value of <c>G_j(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is greater than 0, <paramref name="j"/> is <see cref="double.NegativeInfinity"/>, or either argument is <see cref="double.NaN"/>. / <c>G_j(x)</c> の値。<paramref name="x"/> が 0 より大きい場合、<paramref name="j"/> が <see cref="double.NegativeInfinity"/> の場合、またはいずれかの引数が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>
        /// Uses the normalized convention <c>G_j(x) = (1 / Γ(j + 1)) ∫_0^∞ t^j / (e^(t - x) - 1) dt</c> for <c>j &gt; -1</c> and <c>x &lt;= 0</c>,
        /// extended to <c>j &lt;= -1</c> by analytic continuation via <c>Li_(j+1)(e^x)</c>.
        /// At <c>x = 0</c>, returns <c>ζ(j + 1)</c> for <c>j &gt; 0</c> and <see cref="double.PositiveInfinity"/> for <c>j &lt;= 0</c>. Returns <c>0</c> at <c>x = -∞</c>.
        /// </para>
        /// <para>
        /// <c>j &gt; -1</c>、<c>x ≤ 0</c> では正規化定義 <c>G_j(x) = (1/Γ(j+1)) ∫₀^∞ tʲ/(eᵗ⁻ˣ - 1) dt</c> を採用し、<c>j ≤ -1</c> は <c>Li_(j+1)(eˣ)</c> による解析接続で定義します。
        /// <c>x = 0</c> では <c>j &gt; 0</c> で <c>ζ(j+1)</c>、<c>j ≤ 0</c> で <see cref="double.PositiveInfinity"/> を返します。<c>x = -∞</c> では <c>0</c> を返します。
        /// </para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/25.12#iii"/>
        public static double BoseEinstein(double j, double x)
        {
            if (double.IsNaN(j) || double.IsNaN(x) || x > 0 || double.IsNegativeInfinity(j))
            {
                return double.NaN;
            }

            double s = j + 1;
            if (x == 0)
            {
                return s > 1 ? Zeta(s) : double.PositiveInfinity;
            }

            if (double.IsNegativeInfinity(x))
            {
                return 0;
            }

            if (double.IsPositiveInfinity(s))
            {
                return Math.Exp(x);
            }

            return PolylogPositive(s, Math.Exp(x), x);
        }

        // Li_s(x)（0 ≤ x < 1、μ = ln x）
        private static double PolylogPositive(double s, double x, double mu)
        {
            // |s| < 10⁻¹⁷ では Li_s と Li₀ の差 s ∂Li/∂s が丸め以下（積分表示の tˢ⁻¹ の山も表せなくなる）
            if (Math.Abs(s) < 1e-17)
            {
                // x/(1 - x) = 1/(e⁻ᵘ - 1)
                return 1 / Expm1(-mu);
            }

            if (s == 1)
            {
                return -Log1mExp(mu);
            }

            if (PolylogDirectIsCheap(s, x, mu))
            {
                return PolylogDirect(s, x);
            }

            if (s < 0)
            {
                return PolylogPoleSum(s, -mu, false);
            }

            return PolylogLogSeries(s, mu);
        }

        // Li_s(-y)（y = eˡ > 0。y はオーバーフローまたはアンダーフローしていてもよい）
        private static double PolylogNegative(double s, double y, double ell)
        {
            if (Math.Abs(s) < 1e-17)
            {
                // -y/(1 + y)
                return ell > 0 ? -1 / (1 + Math.Exp(-ell)) : -y / (1 + y);
            }

            if (s == 1)
            {
                // -ln(1 + y)
                return ell > 0 ? -(ell + Log1p(Math.Exp(-ell))) : -Log1p(y);
            }

            if (ell == 0)
            {
                // Li_s(-1) = -η(s)
                return -DirichletEta(s);
            }

            if (ell <= 0)
            {
                if (PolylogDirectIsCheap(s, -y, ell))
                {
                    return PolylogDirect(s, -y);
                }

                return s > 0 ? PolylogAlternating(s, y) : PolylogPoleSum(s, -ell, true);
            }

            if (IsInteger(s) || (ell >= PolylogAsymptoticThreshold && PolylogInversionConverges(s, ell)))
            {
                return PolylogInversion(s, ell);
            }

            if (s < 0)
            {
                return PolylogPoleSum(s, -ell, true);
            }

            // s が ℓ に比べて十分大きければ、発散級数 Σ (-y)ᵏ/kˢ を項が増え始める前に打ち切った値が
            // 積分表示のうち t < ℓ の部分（大きさ ℓˢ/Γ(s+1) 程度）を除いて一致する
            if (s * Math.Log(ell) - LogGamma(s + 1) < ell - 40)
            {
                double direct = PolylogDirectDivergent(s, ell);
                if (!double.IsNaN(direct))
                {
                    return direct;
                }
            }

            return -PolylogFermiDiracTrapezoid(s, ell);
        }

        // 冪級数 Σ xᵏ/kˢ が PolylogDirectMaxTerms 項以内に収束し、桁落ちも起きないか判定する（|x| ≤ 1、logAbsX = ln|x|）
        private static bool PolylogDirectIsCheap(double s, double x, double logAbsX)
        {
            if (x == 0)
            {
                return true;
            }

            double n = PolylogDirectMaxTerms;

            // 最後の項と初項の比 |x|ⁿ⁻¹ n⁻ˢ
            if ((n - 1) * logAbsX - s * Math.Log(n) > -39)
            {
                return false;
            }

            if (s >= 0 || x > 0)
            {
                return true;
            }

            // s < 0 の交代級数は、項 |x|ᵏ kᵃ（a = -s）の最大値が初項より大きいと打ち消し合いで桁落ちする
            double peak = s / logAbsX;
            if (peak <= 1)
            {
                return true;
            }

            double logMax = -s * Math.Log(peak) + peak * logAbsX;
            return logMax - logAbsX <= 1.4;
        }

        // 冪級数 Σₖ xᵏ/kˢ を直接加算する（PolylogDirectIsCheap が真のとき）
        private static double PolylogDirect(double s, double x)
        {
            double sum = x;
            double power = x;
            double absX = Math.Abs(x);
            double peak = s < 0 && absX < 1 ? s / Math.Log(absX) : 0;
            for (int k = 2; k <= PolylogDirectMaxTerms; k++)
            {
                power *= x;
                double term = power * Math.Pow(k, -s);
                sum += term;
                if (k > peak && Math.Abs(term) <= 0.0625 * Epsilon * Math.Abs(sum))
                {
                    break;
                }
            }

            return sum;
        }

        // Li_s(-y) = -Σₖ (-1)ᵏ yᵏ⁺¹/(k+1)ˢ（s > 0、0 < y ≤ 1）。
        // aₖ = yᵏ⁺¹/(k+1)ˢ = ∫₀¹ tᵏ dμ(t)（μ ≥ 0）なので Cohen–Villegas–Zagier のアルゴリズム 1 で加速する。
        // 誤差は 2a₀/(3+√8)ⁿ 以下、和は a₀/2 以上なので相対誤差は 2×10⁻¹⁸ 以下。
        private static double PolylogAlternating(double s, double y)
        {
            const int n = PolylogAlternatingTerms;
            double d = Math.Pow(3 + Math.Sqrt(8), n);
            d = 0.5 * (d + 1 / d);
            double b = -1;
            double c = -d;
            double sum = 0;
            double power = 1;
            for (int k = 0; k < n; k++)
            {
                power *= y;
                c = b - c;
                sum += c * power * Math.Pow(k + 1, -s);
                b *= (k + n) * (double)(k - n) / ((k + 0.5) * (k + 1));
            }

            return -sum / d;
        }

        // s が大きいときの Li_s(-eˡ)（ℓ > 0）を発散級数 Σ (-eˡ)ᵏ/kˢ の加算で求める。
        // 項が減少しなくなる前に収束しなければ double.NaN を返す。
        private static double PolylogDirectDivergent(double s, double ell)
        {
            double sum = -Math.Exp(ell);
            double previous = Math.Abs(sum);
            for (int k = 2; k <= PolylogDirectMaxTerms; k++)
            {
                // yᵏ はオーバーフローしうるので指数でまとめる（この項は初項より十分小さいので指数の丸めは効かない）
                double magnitude = Math.Exp(k * ell - s * Math.Log(k));
                if (magnitude >= previous)
                {
                    return double.NaN;
                }

                sum += (k & 1) == 0 ? magnitude : -magnitude;
                if (magnitude <= 0.0625 * Epsilon * Math.Abs(sum))
                {
                    return sum;
                }

                previous = magnitude;
            }

            return double.NaN;
        }

        // ζ(a)（a ≥ 2）の上界 1 + 2⁻ᵃ + 2^(1-a)/(a-1)
        private static double PolylogZetaBound(double a) => 1 + Math.Pow(2, -a) * (1 + 2 / (a - 1));

        // 整数 k ≥ 2 に対する ζ(k)（表にない k > 40 は 1 + 2⁻ᵏ + 3⁻ᵏ）
        private static double PolylogZetaInteger(int k) =>
            k < ZetaAtInteger.Length ? ZetaAtInteger[k] : 1 + Math.ScaleB(1.0, -k) + Math.Pow(3, -k);

        // Li_s(eᵘ) を μ の級数 Γ(1-s)(-μ)ˢ⁻¹ + Σ ζ(s-k)μᵏ/k! で求める（s > 0、-2π < μ < 0、実用上 |μ| ≤ ln 2）。
        // s ≥ 1/2 では最も近い整数 m = s - 1 の項 ζ(s-m)μᵐ/m! を極の項 Γ(1-s)(-μ)ˢ⁻¹ とまとめて桁落ちを避ける。
        private static double PolylogLogSeries(double s, double mu)
        {
            double l = -mu;
            int m = s >= 0.5 ? (int)Math.Round(s - 1) : -1;

            // 正則な項 Σ_(k≠m) ζ(s-k) μᵏ/k!。|ζ(s-k)| は k が大きいと 2Γ(k-s+1)/(2π)^(k-s+1) 程度なので項の比は |μ|/2π 程度
            double sum = 0;
            double factor = 1;
            double factorM = 0;
            for (int k = 0; k < 80; k++)
            {
                if (k == m)
                {
                    factorM = factor;
                }
                else
                {
                    double term = Zeta(s - k) * factor;
                    sum += term;

                    // ζ(s-k) は負の偶数の近くで 0 になりうるので、打ち切りは包絡線 |ζ(s-k)| ≤ 2Γ(k+1-s)/(2π)^(k+1-s)·ζ(k+1-s) で判定する
                    double a = k + 1 - s;
                    if (k > m && a >= 2)
                    {
                        double envelope = 2 * Math.Exp(LogGamma(a) - a * LnTwoPi) * PolylogZetaBound(a) * Math.Abs(factor);
                        if (envelope <= 0.0625 * Epsilon * Math.Abs(sum))
                        {
                            break;
                        }
                    }
                }

                factor *= mu / (k + 1);
            }

            if (m < 0)
            {
                return Gamma(1 - s) * Math.Pow(l, s - 1) + sum;
            }

            double eps = (s - 1) - m;

            // ζ(1+ε) - 1/ε = Σ cₙ εⁿ
            double stieltjes = 0;
            for (int i = ZetaStieltjesCoefficients.Length - 1; i >= 0; i--)
            {
                stieltjes = stieltjes * eps + ZetaStieltjesCoefficients[i];
            }

            // g = f/ε = ln(-μ) + ln Γ(1-ε)/ε - Σᵢ ln(1 + ε/i)/ε。ln Γ(1-ε)/ε = γ + Σ_(j≥2) ζ(j) εʲ⁻¹/j（|ε| ≤ 1/2）
            double logGammaOverEps = 0;
            for (int j = 60; j >= 2; j--)
            {
                logGammaOverEps = logGammaOverEps * eps + PolylogZetaInteger(j) / j;
            }

            logGammaOverEps = EulerGamma + logGammaOverEps * eps;
            double harmonic = 0;
            for (int i = 1; i <= m; i++)
            {
                harmonic += eps == 0 ? 1.0 / i : Log1p(eps / i) / eps;
            }

            double g = Math.Log(l) + logGammaOverEps - harmonic;
            double expm1OverEps = eps == 0 ? g : Expm1(eps * g) / eps;
            return factorM * (stieltjes - expm1OverEps) + sum;
        }

        // 極の和 Li_s(eʷ) = Γ(1-s) Σⱼ (2πij - w)ˢ⁻¹（s < 0）。
        // x > 0（negative が偽）は w = -L（L = -ln x > 0）で括弧は Lˢ⁻¹ + 2 Σ_(m≥1) Re zₘˢ⁻¹、
        // x < 0 は w = -L + πi（L = -ln(-x)）で 2 Σ_(m≥0) Re zₘˢ⁻¹（zₘ = L + (2m + c)πi、c は 0 または 1）。
        // |zₘ| ≥ 40 + 2|s| 以降はオイラー・マクローリンの和公式で裾を加算する（項の比は |s-2k|²/|z_N|² 程度、最小項は e^(-|z_N|) 程度）。
        private static double PolylogPoleSum(double s, double l, bool negative)
        {
            double p = s - 1;
            int c = negative ? 1 : 0;

            // すべての項を最小の |zₘ| の p 乗で割って持つ
            double rRef = negative ? double.Hypot(l, Math.PI) : l;
            double bracket = negative ? 0 : 1;
            double scale = 1;
            double tailRadius = 40 + 2 * Math.Abs(s);
            for (int m = negative ? 0 : 1; ; m++)
            {
                double im = Math.PI * (2 * m + c);
                double r = double.Hypot(l, im);
                double theta = Math.Atan2(im, l);
                double magnitude = Math.Pow(r / rRef, p);
                if (r >= tailRadius)
                {
                    bracket += 2 * PolylogPoleSumTail(s, r, theta, magnitude, scale);
                    break;
                }

                bracket += 2 * magnitude * Math.Cos(p * theta);
                scale = Math.Max(scale, Math.Abs(bracket));

                // 残りの和 Σ_(m'>m) |zₘ'|ᵖ は |zₘ|ˢ/(2π|s|) 程度以下
                if (magnitude * Math.Max(1, r / (2 * Math.PI * Math.Abs(s))) <= 0.03125 * Epsilon * scale)
                {
                    break;
                }
            }

            // Γ(1-s) rRefᵖ。オーバーフローまたはアンダーフローするときは対数でまとめる
            double gamma = Gamma(1 - s);
            double power = Math.Pow(rRef, p);
            double prefactor = gamma * power;
            if (!double.IsFinite(prefactor) || prefactor < MinNormal || !double.IsFinite(gamma) || power == 0 || double.IsInfinity(power))
            {
                prefactor = Math.Exp(LogGamma(1 - s) + p * Math.Log(rRef));
            }

            return prefactor * bracket;
        }

        // PolylogPoleSum のオイラー・マクローリンの裾 Re Σ_(m≥N) zₘˢ⁻¹ を rRefˢ⁻¹ で割った値（z_N = r e^(iθ)、magnitude = (r/rRef)ˢ⁻¹）
        private static double PolylogPoleSumTail(double s, double r, double theta, double magnitude, double scale)
        {
            double p = s - 1;

            // Re[-zˢ/(2πis)] = -|z|ˢ sin(sθ)/(2πs)。s → 0 でも sin(sθ)/s は桁落ちしない
            double st = s * theta;
            double sinOverS = Math.Abs(st) < 1e-4 ? theta * (1 - st * st / 6) : Math.Sin(st) / s;
            double tail = -magnitude * r * sinOverS * InvTwoPi + 0.5 * magnitude * Math.Cos(p * theta);

            // Re[-B₂ₖ/(2k)! (s-1)↓(2k-1) (2πi)²ᵏ⁻¹ zˢ⁻²ᵏ] = (-1)ᵏ⁺¹ B₂ₖ/(2k)! (s-1)↓(2k-1) (2π)²ᵏ⁻¹ |z|ˢ⁻²ᵏ sin((s-2k)θ)
            double falling = p;
            double twoPiPower = 2 * Math.PI;
            double radiusPower = magnitude / r;
            double previous = double.PositiveInfinity;
            for (int k = 1; k < BernoulliEven.Length; k++)
            {
                double coefficient = BernoulliEven[k] / FactorialTable[2 * k] * falling * twoPiPower * radiusPower;
                double bound = Math.Abs(coefficient);
                if (bound > previous)
                {
                    break;
                }

                double term = ((k & 1) == 1 ? coefficient : -coefficient) * Math.Sin((s - 2 * k) * theta);
                tail += term;
                if (bound <= 0.03125 * Epsilon * Math.Max(scale, Math.Abs(tail)))
                {
                    break;
                }

                previous = bound;
                falling *= (s - 2 * k) * (s - 2 * k - 1);
                twoPiPower *= 4 * Math.PI * Math.PI;
                radiusPower /= r * r;
            }

            return tail;
        }

        // η(2k) = (1 - 2¹⁻²ᵏ) ζ(2k)（η(0) = 1/2）
        private static double PolylogEtaEven(int k)
        {
            if (k == 0)
            {
                return 0.5;
            }

            if (2 * k < ZetaAtInteger.Length)
            {
                return (1 - Math.ScaleB(1.0, 1 - 2 * k)) * ZetaAtInteger[2 * k];
            }

            return 1 - Math.ScaleB(1.0, -2 * k) + Math.Pow(3, -2 * k);
        }

        // PolylogInversion の漸近級数（非整数 s、ℓ ≥ 38）が倍精度まで収束するか判定する。
        // s < 0 では項の比 |(s-2k)(s-2k-1)|/ℓ² が初めから大きいことがあり、最小項と初項の比はおよそ Γ(ℓ)/(Γ(a) ℓ^(ℓ-a))（a = -s）なので、その対数 a ln ℓ - ln Γ(a) - ℓ で判定する。
        private static bool PolylogInversionConverges(double s, double ell)
        {
            if (s >= 0)
            {
                return true;
            }

            double a = -s;
            return a * Math.Log(ell) - LogGamma(a) - ell < -37;
        }

        // 反転公式 Li_s(-eˡ) = -Σₖ 2η(2k) ℓˢ⁻²ᵏ/Γ(s+1-2k) - cos(πs) Li_s(-e⁻ˡ)（ℓ > 0）。
        // 整数 s では k ≤ s/2 で終わる恒等式、非整数 s では ℓ → ∞ の漸近級数（最小項で打ち切り）。
        // p = s - 2k ∈ (-1, 1] の項 ℓᵖ/Γ(p+1) から漸化式 ℓᵖ⁺²/Γ(p+3) = ℓᵖ/Γ(p+1) · ℓ²/((p+1)(p+2)) で両方向に計算し、ℓˢ 単独のオーバーフローを避ける。
        private static double PolylogInversion(double s, double ell)
        {
            int k0 = s > 1 ? (int)Math.Ceiling((s - 1) / 2) : 0;
            double p0 = s - 2.0 * k0;
            double baseTerm = Math.Pow(ell, p0) * ReciprocalGamma(p0 + 1);
            if (p0 < -1 && (baseTerm == 0 || !double.IsFinite(baseTerm)) && !IsNonPositiveInteger(p0 + 1))
            {
                // s が大きな負の数で ℓˢ と 1/Γ(s+1) が別々にアンダーフローまたはオーバーフローするとき
                baseTerm = SignGamma(p0 + 1) * Math.Exp(p0 * Math.Log(ell) - LogGamma(p0 + 1));
            }

            // k < k0（p > p0）の項。ℓᵖ/Γ(p+1) は p < ℓ で単調増加するため小さい方から加算する
            double sum = 2 * PolylogEtaEven(k0) * baseTerm;
            double term = baseTerm;
            double p = p0;
            for (int k = k0 - 1; k >= 0; k--)
            {
                p += 2;
                term *= ell / p * (ell / (p - 1));
                sum += 2 * PolylogEtaEven(k) * term;
            }

            // k > k0（p < p0）の項。整数 s では係数 p(p-1) が 0 になって終了する
            term = baseTerm;
            p = p0;
            double tail = 0;
            double previous = double.PositiveInfinity;
            for (int k = k0 + 1; k < 100000; k++)
            {
                term *= p / ell * ((p - 1) / ell);
                p -= 2;
                double value = 2 * PolylogEtaEven(k) * term;
                double magnitude = Math.Abs(value);
                if (magnitude >= previous)
                {
                    break;
                }

                tail += value;
                if (magnitude <= 0.0625 * Epsilon * Math.Abs(sum + tail))
                {
                    break;
                }

                previous = magnitude;
            }

            double result = -(sum + tail);
            double cosine = double.CosPi(s);
            if (cosine != 0)
            {
                result -= cosine * PolylogNegative(s, Math.Exp(-ell), -ell);
            }

            return result;
        }

        // 完全 Fermi–Dirac 積分 -Li_s(-eˡ) = (1/Γ(s)) ∫₀^∞ tˢ⁻¹/(eᵗ⁻ˡ + 1) dt（s > 0、0 < ℓ < 38）を台形則で計算する。
        // t = exp(v - e⁻ᵛ) により被積分関数 f(v) = tˢ(1 + e⁻ᵛ)/(1 + eᵗ⁻ˡ) は両側で二重指数的に減衰し、実軸近傍の特異点は t = ℓ ± (2k+1)πi の単純極（留数 ∓tₖˢ⁻¹）のみとなる。
        // 刻み h（s が大きいと山の幅が 1/√s 程度になるため h ∝ 1/√s）、節点 v₀ + nh の台形則の極誤差 2πi Res q/(1 - q)（q = e^(2πi(vₖ - v₀)/h)）を共役対の実部として引く（最小項で打ち切り）。
        private static double PolylogFermiDiracTrapezoid(double s, double ell)
        {
            // 節点は v = v₀ + u（v₀ = ln T、T = max(ℓ, s, 1)）。ln t - ln T = u - e⁻ᵛ を直接作り、s 倍で拡大する丸めを山の近くで小さく保つ
            double h = Math.Min(0.15, 0.4 / Math.Sqrt(s));
            double reference = Math.Max(Math.Max(ell, s), 1);
            double v0 = Math.Log(reference);
            double logS = Math.Log(s);

            // 右側（u ≥ 0）：t > T + 1 を過ぎて最大値より e⁻⁴⁰ 小さくなったら止める
            double sum = 0;
            double logMax = double.NegativeInfinity;
            for (int n = 0; n < 4000; n++)
            {
                double logValue = PolylogFermiDiracLogIntegrand(s, ell, reference, v0, n * h, out double t);
                logMax = Math.Max(logMax, logValue);
                sum += Math.Exp(logValue);
                if (logValue < logMax - 40 && t > reference + 1)
                {
                    break;
                }
            }

            // 左側：s < 1 では v = ln s 付近にも山があるので、それより左で小さくなったら止める
            for (int n = 1; n < 4000; n++)
            {
                double u = -n * h;
                double logValue = PolylogFermiDiracLogIntegrand(s, ell, reference, v0, u, out _);
                logMax = Math.Max(logMax, logValue);
                sum += Math.Exp(logValue);
                if (logValue < logMax - 40 && v0 + u < Math.Min(v0, logS) - 1)
                {
                    break;
                }
            }

            double trapezoid = h * sum;

            // 極 vₖ = v₀ + wₖ（v - e⁻ᵛ = ln tₖ、tₖ = ℓ + (2k+1)πi）による誤差を引く。
            // w - e^(-v₀) e⁻ʷ = ln(tₖ/T) をニュートン法で解き、留数 -tₖˢ⁻¹ は Tˢ で割った (tₖ/T)ˢ⁻¹/T として持つ
            double inverseReference = 1 / reference;
            double correction = 0;
            double previous = double.PositiveInfinity;
            for (int k = 0; k < 1000; k++)
            {
                Complex logRatio = Complex.Log(new Complex(ell * inverseReference, Math.PI * (2 * k + 1) * inverseReference));
                Complex w = logRatio + inverseReference * Complex.Exp(-logRatio);
                for (int iteration = 0; iteration < 50; iteration++)
                {
                    Complex e = inverseReference * Complex.Exp(-w);
                    Complex dw = (w - e - logRatio) / (1 + e);
                    w -= dw;
                    if (Complex.Abs(dw) <= 1e-16 * Math.Max(1, Complex.Abs(w)))
                    {
                        break;
                    }
                }

                Complex residue = -inverseReference * Complex.Exp((s - 1) * logRatio);
                Complex q = Complex.Exp(new Complex(0, 2 * Math.PI / h) * w);
                Complex term = new Complex(0, 2 * Math.PI) * residue * q / (1 - q);
                double magnitude = Complex.Abs(term);
                if (magnitude > previous)
                {
                    break;
                }

                correction += 2 * term.Real;
                if (magnitude <= 0.0625 * Epsilon * Math.Abs(trapezoid))
                {
                    break;
                }

                previous = magnitude;
            }

            // Tˢ/Γ(s)（この経路では s ≲ 110 なのでどちらもオーバーフローしない）
            return (trapezoid - correction) * (Math.Pow(reference, s) / Gamma(s));
        }

        // PolylogFermiDiracTrapezoid の被積分関数 (t/T)ˢ(1 + e⁻ᵛ)/(1 + eᵗ⁻ˡ) の対数（v = v₀ + u、ln(t/T) = u - e⁻ᵛ）
        private static double PolylogFermiDiracLogIntegrand(double s, double ell, double reference, double v0, double u, out double t)
        {
            double e = Math.Exp(-(v0 + u));
            if (e > 1e300)
            {
                // tˢ が二重指数的に小さい左の裾
                t = 0;
                return double.NegativeInfinity;
            }

            double logRatio = u - e;
            t = reference * Math.Exp(logRatio);
            double a = t - ell;
            double softplus = a > 0 ? a + Log1p(Math.Exp(-a)) : Log1p(Math.Exp(a));
            return s * logRatio + Log1p(e) - softplus;
        }
    }
}
