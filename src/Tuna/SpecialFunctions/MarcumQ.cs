using System.Numerics;
using static Tuna.XMath.Constants;
using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // X = x²/2 < 30 または ξ = 2√(XY) ≤ 40 ではポアソン重み級数（Gil–Segura–Temme の境界）、それ以外は複素積分を台形則で評価する
        private const double MarcumSeriesMaxX = 30;
        private const double MarcumSeriesMaxXi = 40;

        // 積分路と実軸の交点を極 u = 1 から鞍点幅のこの倍数以上離す（桁落ちは 1 桁未満に収まる）
        private const double MarcumPoleClearance = 2;

        /// <summary>
        /// Computes the generalized Marcum Q-function <c>Q_μ(x, y)</c>.
        /// <para>一般化 Marcum Q 関数 <c>Q_μ(x, y)</c> を計算します。</para>
        /// </summary>
        /// <param name="mu">The positive order (<c>μ &gt; 0</c>, finite; need not be an integer). / 正の有限次数（<c>μ &gt; 0</c>、非整数も指定可能です）。</param>
        /// <param name="x">The non-negative first argument (<c>x ≥ 0</c>). / 非負の第 1 引数（<c>x ≥ 0</c>）。</param>
        /// <param name="y">The non-negative lower integration limit (<c>y ≥ 0</c>). / 非負の下端（<c>y ≥ 0</c>）。</param>
        /// <returns>The value of <c>Q_μ(x, y)</c> in <c>[0, 1]</c>, or <see cref="double.NaN"/> if parameters are out of domain or both <paramref name="x"/> and <paramref name="y"/> are <see cref="double.PositiveInfinity"/>. / <c>[0, 1]</c> の範囲の <c>Q_μ(x, y)</c> の値。引数が定義域外、または <paramref name="x"/> と <paramref name="y"/> がともに <see cref="double.PositiveInfinity"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <c>1</c> when <c>y = 0</c> or <c>x = +∞</c>, <c>0</c> when <c>y = +∞</c>, and <c>Q(μ, y² / 2)</c> when <c>x = 0</c>. Satisfies <c>Q_(k/2)(√λ, √x) =</c> <see cref="NoncentralChiSquaredSf"/>.</para>
        /// <para><c>y = 0</c> または <c>x = +∞</c> では <c>1</c>、<c>y = +∞</c> では <c>0</c>、<c>x = 0</c> では <c>Q(μ, y² / 2)</c> を返します。<c>Q_(k/2)(√λ, √x)</c> は <see cref="NoncentralChiSquaredSf"/> と一致します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/10.45"/>
        public static double MarcumQ(double mu, double x, double y)
        {
            if (double.IsNaN(mu) || double.IsNaN(x) || double.IsNaN(y) || !(mu > 0) || double.IsPositiveInfinity(mu)
                || x < 0 || y < 0 || (double.IsPositiveInfinity(x) && double.IsPositiveInfinity(y)))
            {
                return double.NaN;
            }

            if (y == 0 || double.IsPositiveInfinity(x))
            {
                return 1;
            }

            if (double.IsPositiveInfinity(y))
            {
                return 0;
            }

            if (x > 1e150 || y > 1e150)
            {
                // X = x²/2 がオーバーフローしうる範囲。非心ガンマ分布は平均 X + μ、分散 2X + μ の正規分布に近づき、
                // (Y - X - μ)/√(2X + μ) = (y - x) + O(μ/x)（x ≈ y の範囲。それ以外は 0 か 1）
                return mu < 1e100 ? 0.5 * Erfc((y - x) * OneOverSqrt2) : double.NaN;
            }

            // X - Y = (x - y)(x + y)/2 は x ≈ y でも丸め誤差が小さい
            MarcumCore(mu, 0.5 * x * x, 0.5 * y * y, 0.5 * (x - y) * (x + y), out _, out double q);
            return q;
        }

        /// <summary>
        /// Computes the cumulative distribution function <c>F(x; k, λ)</c> of the noncentral chi-squared distribution.
        /// <para>非心 χ² 分布の累積分布関数 <c>F(x; k, λ)</c> を計算します。</para>
        /// </summary>
        /// <param name="k">The degrees of freedom (<c>k &gt; 0</c>, finite). / 自由度（有限な <c>k &gt; 0</c>）。</param>
        /// <param name="lambda">The noncentrality parameter (<c>λ ≥ 0</c>, finite). / 非心度（有限な <c>λ ≥ 0</c>）。</param>
        /// <param name="x">The evaluation point. / 評価点。</param>
        /// <returns>The cumulative probability <c>1 - Q_(k/2)(√λ, √x)</c>, or <see cref="double.NaN"/> if parameters are out of domain or any argument is <see cref="double.NaN"/>. / 累積確率 <c>1 - Q_(k/2)(√λ, √x)</c>。引数が定義域外または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <c>0</c> for <c>x ≤ 0</c>, <c>1</c> at <c>x = +∞</c>, and <c>P(k/2, x/2)</c> at <c>λ = 0</c>. Parameter order matches <see cref="NoncentralChiSquaredPdf"/>.</para>
        /// <para><c>x ≤ 0</c> では <c>0</c>、<c>x = +∞</c> では <c>1</c>、<c>λ = 0</c> では中心 χ² 分布の <c>P(k/2, x/2)</c> を返します。引数の順序は <see cref="NoncentralChiSquaredPdf"/> と同じです。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution"/>
        public static double NoncentralChiSquaredCdf(double k, double lambda, double x)
        {
            if (!NoncentralChiSquaredTails(k, lambda, x, out double p, out _))
            {
                return double.NaN;
            }

            return p;
        }

        /// <summary>
        /// Computes the survival function (upper-tail probability) <c>1 - F(x; k, λ)</c> of the noncentral chi-squared distribution.
        /// <para>非心 χ² 分布の生存関数（上側確率）<c>1 - F(x; k, λ)</c> を計算します。</para>
        /// </summary>
        /// <param name="k">The degrees of freedom (<c>k &gt; 0</c>, finite). / 自由度（有限な <c>k &gt; 0</c>）。</param>
        /// <param name="lambda">The noncentrality parameter (<c>λ ≥ 0</c>, finite). / 非心度（有限な <c>λ ≥ 0</c>）。</param>
        /// <param name="x">The evaluation point. / 評価点。</param>
        /// <returns>The upper-tail probability <c>Q_(k/2)(√λ, √x)</c>, or <see cref="double.NaN"/> if parameters are out of domain or any argument is <see cref="double.NaN"/>. / 上側確率 <c>Q_(k/2)(√λ, √x)</c>。引数が定義域外または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <c>1</c> for <c>x ≤ 0</c>, <c>0</c> at <c>x = +∞</c>, and <c>Q(k/2, x/2)</c> at <c>λ = 0</c>.</para>
        /// <para><c>x ≤ 0</c> では <c>1</c>、<c>x = +∞</c> では <c>0</c>、<c>λ = 0</c> では <c>Q(k/2, x/2)</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution"/>
        public static double NoncentralChiSquaredSf(double k, double lambda, double x)
        {
            if (!NoncentralChiSquaredTails(k, lambda, x, out _, out double q))
            {
                return double.NaN;
            }

            return q;
        }

        private static bool NoncentralChiSquaredTails(double k, double lambda, double x, out double p, out double q)
        {
            p = q = double.NaN;
            if (double.IsNaN(k) || double.IsNaN(lambda) || double.IsNaN(x) || !(k > 0) || !(lambda >= 0)
                || double.IsInfinity(k) || double.IsInfinity(lambda))
            {
                return false;
            }

            if (x <= 0)
            {
                p = 0;
                q = 1;
                return true;
            }

            if (double.IsPositiveInfinity(x))
            {
                p = 1;
                q = 0;
                return true;
            }

            // λ/2、x/2 は（非正規数を除いて）正確で、X - Y = (λ - x)/2 も丸め 1 回
            MarcumCore(0.5 * k, 0.5 * lambda, 0.5 * x, 0.5 * (lambda - x), out p, out q);
            return true;
        }

        // 非心ガンマ分布の下側・上側確率 P = Σ e^(-X)Xᵏ/k! P(μ+k, Y), Q = 1 - P（xMinusY は桁落ちなく求めた X - Y）
        private static void MarcumCore(double mu, double x, double y, double xMinusY, out double p, out double q)
        {
            if (y == 0)
            {
                p = 0;
                q = 1;
                return;
            }

            if (x == 0)
            {
                IncompleteGammaCore(mu, y, out p, out q);
                return;
            }

            bool upper = xMinusY + mu < 0;
            double xi = 2 * Math.Sqrt(x) * Math.Sqrt(y);

            // 鞍点 u₀ と、そこでの指数 φ₀ = ln(裾の大きさ)。u₀ - 1 = (X - Y + μ)/(Y(1 + 2X/(D₀ + μ)))（D₀ = √(μ² + 4XY)）は桁落ちしない形
            double d0 = MarcumHypot(mu, xi);
            double uMinus1 = (xMinusY + mu) / (y * (1 + 2 * x / (d0 + mu)));
            double t = uMinus1 / (1 + uMinus1);

            // ln(1 - t) + t：t が 1 に近い（u₀ が大きい）と 1 - t が丸めで潰れるので、ln(1 - t) = -ln u₀ として求める
            double logTerm = Math.Abs(t) < 0.5 ? Log1pmx(-t) : t - Log1p(uMinus1);
            double phi0 = -x * t * t + mu * logTerm;
            if (phi0 < -800)
            {
                // 小さい方の裾は e^φ₀ に高々 O(1) の因子を掛けた大きさで、倍精度ではアンダーフローする
                p = upper ? 1 : 0;
                q = upper ? 0 : 1;
                return;
            }

            if (x < MarcumSeriesMaxX || xi <= MarcumSeriesMaxXi)
            {
                if (upper)
                {
                    q = MarcumUpperSeries(mu, x, y);
                    p = 1 - q;
                }
                else
                {
                    p = MarcumLowerSeries(mu, x, y);
                    q = 1 - p;
                }

                return;
            }

            double tail = MarcumIntegral(mu, x, y, d0, uMinus1, phi0, out bool tailIsUpper);
            if (tailIsUpper)
            {
                q = tail;
                p = 1 - q;
            }
            else
            {
                p = tail;
                q = 1 - p;
            }
        }

        // Q = Σₖ wₖ Q(μ+k, Y)（wₖ = e^(-X)Xᵏ/k!、X < 30 を想定）。
        // Qₖ₊₁ = Qₖ + dₖ は正項加算で前進安定。深い裾では共通倍率でスケーリングして保持する
        private static double MarcumUpperSeries(double mu, double x, double y)
        {
            // Q₀ が正規化数で表せるなら直接（対数を経由すると指数の丸めで相対誤差が ε|ln Q₀| になる）、そうでなければ対数で持つ
            double q0 = RegularizedGammaQ(mu, y);
            bool direct = q0 > 1e-280;
            double scale = direct ? 0 : LogRegularizedGammaQ(mu, y);
            double qk = 1;
            double d0 = GammaPrefix(mu, y) / mu;
            double dk = direct && d0 > 1e-290 ? d0 / q0 : Math.Exp(LogGammaPrefix(mu, y) - Math.Log(mu) - (direct ? Math.Log(q0) : scale));
            double wk = Math.Exp(-x);
            double sum = 0;
            long binaryExponent = 0;
            for (int k = 0; k < 100000; k++)
            {
                double term = wk * qk;
                sum += term;

                // 次の項との比 X/(k+1)·Qₖ₊₁/Qₖ は k とともに減る。比が 1/2 未満で項が十分小さければ残りの和も無視できる
                double ratio = x / (k + 1) * (1 + dk / qk);
                if (k + 1 > x && ratio < 0.5 && term <= 0.25 * MachineEpsilon * sum)
                {
                    break;
                }

                qk += dk;
                dk *= y / (mu + k + 1);
                wk *= x / (k + 1);
                if (qk > 1e280)
                {
                    // 2 のべきで縮めるので丸めは生じない
                    qk = Math.ScaleB(qk, -930);
                    dk = Math.ScaleB(dk, -930);
                    sum = Math.ScaleB(sum, -930);
                    binaryExponent += 930;
                }
            }

            return direct ? ScaleB(q0 * sum, binaryExponent) : Math.Exp(scale + binaryExponent * Ln2 + Math.Log(sum));
        }

        // P = Σₖ wₖ P(μ+k, Y) = w₀d₀ Σₖ rₖ Sₖ。Sₖ₋₁ = 1 + Y/(μ+k)·Sₖ は後退安定のため、打ち切り位置 K を決めてから戻る
        private static double MarcumLowerSeries(double mu, double x, double y)
        {
            double xy = x * y;
            var ratios = new List<double>(64);
            double r = 1;
            double rSum = 0;

            // ポアソン重みの比 vₖ = wₖ/w₀ = Xᵏ/k! とその和（大きくなったら 2 のべきで縮める）
            double v = 1;
            double vSum = 0;
            int kMax = 0;
            for (int k = 0; k < 1000000; k++)
            {
                ratios.Add(r);
                rSum += r;
                vSum += v;
                kMax = k;

                // 項の比 t_(j+1)/tⱼ = ρⱼ S_(j+1)/Sⱼ ≤ ρⱼ = XY/((j+1)(μ+j+1))（Sⱼ は j について減少）で、ρⱼ も減少する。
                // ρₖ ≤ 1/2 なら残りの和は 2ρₖ tₖ 以下、tₖ ∝ rₖ Sₖ ≤ rₖ/(1 - Y/(μ+k+1))。これを和の下界 Σ rⱼ（Sⱼ ≥ 1）と比べる
                double a = mu + k + 1;
                double rho = xy / ((k + 1) * a);
                if (a > y && rho <= 0.5)
                {
                    double bound = 2 * rho * r / (1 - y / a);
                    if (bound <= 0.25 * MachineEpsilon * rSum)
                    {
                        break;
                    }
                }

                // Pⱼ は j について減少するので、残りの和と和の比はポアソン重みだけの比 Σ_(j>k) wⱼ/Σ_(j≤k) wⱼ 以下。
                // μ が Y に比べて極端に大きく（μ + k + 1 が丸めで Y を超えない）上の判定が使えないときに効く
                double next = x / (k + 2);
                if (next < 0.5 && 2 * v * x / (k + 1) <= 0.25 * MachineEpsilon * vSum)
                {
                    break;
                }

                r *= rho;
                v *= x / (k + 1);
                if (v > 1e280)
                {
                    v = Math.ScaleB(v, -930);
                    vSum = Math.ScaleB(vSum, -930);
                }
            }

            // S_K：Y/(μ+K+1) が小さければ級数 Σ Yʲ/((μ+K+1)…(μ+K+j))、そうでなければ P(μ+K, Y) と d_K から
            double aK = mu + kMax;
            double s;
            if (y < 0.5 * (aK + 1))
            {
                s = 1;
                double term = 1;
                for (int j = 1; j < 200; j++)
                {
                    term *= y / (aK + j);
                    s += term;
                    if (term <= 0.5 * MachineEpsilon * s)
                    {
                        break;
                    }
                }
            }
            else
            {
                s = Math.Exp(LogRegularizedGammaP(aK, y) - (LogGammaPrefix(aK, y) - Math.Log(aK)));
            }

            double sum = 0;
            for (int k = kMax; k >= 0; k--)
            {
                sum += ratios[k] * s;
                s = 1 + y / (mu + k) * s;
            }

            // 前因子 w₀d₀ = e^(-X) Y^μ e^(-Y)/Γ(μ+1)。正規化数なら直接掛け、アンダーフローするときは（和を掛けると表せることがあるので）対数で合わせる
            double prefix = Math.Exp(-x) * (GammaPrefix(mu, y) / mu);
            if (prefix > 1e-290)
            {
                return prefix * sum;
            }

            return Math.Exp(-x + LogGammaPrefix(mu, y) - Math.Log(mu) + Math.Log(sum));
        }

        // 小さい方の裾（isUpper が真なら Q、偽なら P）を最急降下路（極 u = 1 近傍のみシフト）上の台形則で求める
        private static double MarcumIntegral(double mu, double x, double y, double d0, double uMinus1, double phi0, out bool isUpper)
        {
            double u0 = 1 + uMinus1;
            double logU0 = Log1p(uMinus1);
            double xOverU0 = x / u0;

            // 鞍点での θ についての 2 階微分 β = 2X/u₀ + μ（被積分関数 ≈ e^(-βθ²/2)）と、その幅 1/√β
            double beta = 2 * xOverU0 + mu;
            double width = 1 / Math.Sqrt(beta);
            isUpper = uMinus1 < 0;
            double logCrossing = logU0;
            if (Math.Abs(logU0) < MarcumPoleClearance * width)
            {
                logCrossing = isUpper ? -MarcumPoleClearance * width : MarcumPoleClearance * width;
            }

            double kappaMinus1 = Expm1(logCrossing - logU0);

            // 台形則の刻み：ガウス型の幅に対して誤差 e^(-2π²(幅/h)²)、θ 平面で距離 |ln u_c| にある極に対して誤差 e^(-2π|ln u_c|/h) が
            // どちらも 10⁻¹⁸ 未満になるよう、余裕をもって選ぶ
            double h = Math.Min(0.35 * width, Math.Abs(logCrossing) / 10);
            double total = MarcumIntegrand(mu, x, y, d0, u0, uMinus1, xOverU0, kappaMinus1, 0).Real;
            for (int j = 1; ; j++)
            {
                double theta = j * h;
                if (theta >= Math.PI)
                {
                    break;
                }

                Complex f = MarcumIntegrand(mu, x, y, d0, u0, uMinus1, xOverU0, kappaMinus1, theta);

                // 被積分関数は θ について偶関数（実部）なので、正の側を 2 倍する
                total += 2 * f.Real;
                if (theta > 4 * width && Complex.Abs(f) < 1e-18 * Math.Abs(total))
                {
                    break;
                }
            }

            double integral = total * h * OneOverTwoPi;
            if (!isUpper)
            {
                integral = -integral;
            }

            return Math.Exp(phi0) * integral;
        }

        // 積分路上の被積分関数 e^(φ-φ₀) u₀ v'(θ)/(i(1 - u))（u = u₀ v, v = κ ρ(θ) e^(iθ)）。
        // w = v - 1 を桁落ちなく構成し、指数を (X/u₀)w²/(1+w) - μ(ln(1+w) - w) とする
        private static Complex MarcumIntegrand(double mu, double x, double y, double d0, double u0, double uMinus1, double xOverU0,
            double kappaMinus1, double theta)
        {
            // s - 1 = θ/sin θ - 1 と s' = (sin θ - θ cos θ)/sin²θ（小さな θ では級数で桁落ちを避ける）
            MarcumSineRatio(theta, out double sMinus1, out double sPrime);
            double s = 1 + sMinus1;
            double xi = 2 * Math.Sqrt(x) * Math.Sqrt(y);
            double dTheta = MarcumHypot(mu * s, xi);

            // D_θ - D₀ = μ²(s² - 1)/(D_θ + D₀)、ρ - 1 = (μ(s - 1) + D_θ - D₀)/(μ + D₀)
            double dDiff = mu * mu * sMinus1 * (s + 1) / (dTheta + d0);
            double rhoMinus1 = (mu * sMinus1 + dDiff) / (mu + d0);
            double rho = 1 + rhoMinus1;

            // r Y - X/r = μ s を微分して r' = μ s'/(Y + X/r²)、ρ' = r'/u₀ = μ s'/(μ + (X/u₀)(1 + 1/ρ²))（u₀Y = μ + X/u₀ を使う）
            double rhoPrime = mu * sPrime / (mu + xOverU0 * (1 + 1 / (rho * rho)));

            double sinTheta = Math.Sin(theta);
            double sinHalf = Math.Sin(0.5 * theta);
            double cosTheta = Math.Cos(theta);
            var expIThetaMinus1 = new Complex(-2 * sinHalf * sinHalf, sinTheta);
            double scaleMinus1 = kappaMinus1 + rhoMinus1 + kappaMinus1 * rhoMinus1;
            Complex w = scaleMinus1 + (1 + scaleMinus1) * expIThetaMinus1;

            Complex exponent = xOverU0 * w * w / (1 + w) - mu * MarcumLog1pmx(w);
            var vPrime = (1 + kappaMinus1) * new Complex(rhoPrime, rho) * new Complex(cosTheta, sinTheta);

            // 1 - u = (1 - u₀) - u₀ w
            Complex denominator = Complex.ImaginaryOne * (-uMinus1 - u0 * w);
            return Complex.Exp(exponent) * (u0 * vPrime / denominator);
        }

        // s - 1 = θ/sin θ - 1 と s' = (sin θ - θ cos θ)/sin²θ（|θ| < π）
        private static void MarcumSineRatio(double theta, out double sMinus1, out double sPrime)
        {
            if (theta == 0)
            {
                sMinus1 = 0;
                sPrime = 0;
                return;
            }

            double sin = Math.Sin(theta);
            if (Math.Abs(theta) >= 0.5)
            {
                sMinus1 = (theta - sin) / sin;
                sPrime = (sin - theta * Math.Cos(theta)) / (sin * sin);
                return;
            }

            // θ - sin θ = Σ_(k≥1) (-1)^(k+1) θ^(2k+1)/(2k+1)!、sin θ - θ cos θ = Σ_(k≥1) (-1)^(k+1) 2k θ^(2k+1)/(2k+1)!
            double t2 = theta * theta;
            double power = theta * t2;
            double factorial = 6;
            double a = 0;
            double b = 0;
            for (int k = 1; k < 30; k++)
            {
                double term = power / factorial;
                if ((k & 1) == 0)
                {
                    term = -term;
                }

                a += term;
                b += 2 * k * term;
                if (Math.Abs(term) <= 1e-18 * Math.Abs(a))
                {
                    break;
                }

                power *= t2;
                factorial *= (2 * k + 2) * (2 * k + 3);
            }

            sMinus1 = a / sin;
            sPrime = b / (sin * sin);
        }

        // 複素数の ln(1 + w) - w を桁落ちなく求める
        private static Complex MarcumLog1pmx(Complex w)
        {
            if (w == Complex.Zero)
            {
                return Complex.Zero;
            }

            if (Complex.Abs(w) >= 0.5)
            {
                return Complex.Log(1 + w) - w;
            }

            // ln(1+w) = 2 artanh(z)（z = w/(2+w)、|z| ≤ 1/3）を展開すると、ln(1+w) - w = -w z + 2(z³/3 + z⁵/5 + …)
            Complex z = w / (2 + w);
            Complex z2 = z * z;
            Complex power = z * z2;
            Complex sum = Complex.Zero;
            for (int k = 3; k < 200; k += 2)
            {
                Complex term = power / k;
                sum += term;
                if (Complex.Abs(term) <= 0.5 * MachineEpsilon * Complex.Abs(sum))
                {
                    break;
                }

                power *= z2;
            }

            return -w * z + 2 * sum;
        }

        // 途中オーバーフローを避けて √(a² + b²)（a, b ≥ 0）を求める
        private static double MarcumHypot(double a, double b)
        {
            double big = Math.Max(a, b);
            if (big == 0)
            {
                return 0;
            }

            if (big < 1e150)
            {
                return Math.Sqrt(a * a + b * b);
            }

            double ra = a / big;
            double rb = b / big;
            return big * Math.Sqrt(ra * ra + rb * rb);
        }
    }
}
