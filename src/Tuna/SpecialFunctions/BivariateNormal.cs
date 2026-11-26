namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // Genz (2004) の切り替え点: |ρ| < 0.925 は Drezner–Wesolowsky 積分、以上は ρ = ±1 からの差表示を用いる
        private const double BivariateNormalHighCorrelation = 0.925;

        // Genz (2004) のガウス・ルジャンドル求積則 (|ρ| < 0.3 で 6 点、0.3 ≤ |ρ| < 0.75 で 12 点、|ρ| ≥ 0.75 および差積分で 20 点)
        private static readonly (double[] Nodes, double[] Weights) BivariateNormalRule6 = CreateGaussLegendreRule(6);
        private static readonly (double[] Nodes, double[] Weights) BivariateNormalRule12 = CreateGaussLegendreRule(12);
        private static readonly (double[] Nodes, double[] Weights) BivariateNormalRule20 = CreateGaussLegendreRule(20);

        /// <summary>
        /// Computes the cumulative distribution function of the bivariate standard normal distribution, <c>Φ₂(x, y; ρ) = P(X ≤ x, Y ≤ y)</c>.
        /// <para>二変量標準正規分布の累積分布関数 <c>Φ₂(x, y; ρ) = P(X ≤ x, Y ≤ y)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The upper integration limit for the first variable. / 第 1 変数の上限。</param>
        /// <param name="y">The upper integration limit for the second variable. / 第 2 変数の上限。</param>
        /// <param name="rho">The correlation coefficient (<c>-1 ≤ ρ ≤ 1</c>). / 相関係数（<c>-1 ≤ ρ ≤ 1</c>）。</param>
        /// <returns>The value of <c>Φ₂(x, y; ρ)</c>, or <see cref="double.NaN"/> if <c>|ρ| &gt; 1</c> or any argument is <see cref="double.NaN"/>. / <c>Φ₂(x, y; ρ)</c> の値。<c>|ρ| &gt; 1</c> またはいずれかの引数が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>The upper tail probability <c>P(X &gt; x, Y &gt; y)</c> can be obtained as <c>Φ₂(-x, -y; ρ)</c>.</para>
        /// <para>上側確率 <c>P(X &gt; x, Y &gt; y)</c> は <c>Φ₂(-x, -y; ρ)</c> として求められます。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case"/>
        public static double BivariateNormalCdf(double x, double y, double rho)
        {
            if (double.IsNaN(x) || double.IsNaN(y) || double.IsNaN(rho) || rho < -1 || rho > 1)
            {
                return double.NaN;
            }

            // 周辺分布の裾が倍精度で 0 になる範囲 (|x| > 40) や ±∞ では 1 変量分布に帰着する
            if (x <= -NormalTailUnderflow || y <= -NormalTailUnderflow)
            {
                return 0;
            }

            if (x >= NormalTailUnderflow)
            {
                return NormalCdf(y);
            }

            if (y >= NormalTailUnderflow)
            {
                return NormalCdf(x);
            }

            double value;
            if (rho == 1)
            {
                value = NormalCdf(Math.Min(x, y));
            }
            else if (rho == -1)
            {
                // P(X ≤ x, -X ≤ y) = P(-y ≤ X ≤ x)
                value = x > -y ? NormalInterval(-y, x) : 0;
            }
            else if (Math.Abs(rho) < BivariateNormalHighCorrelation)
            {
                value = BivariateNormalModerate(x, y, rho);
            }
            else if (rho > 0)
            {
                value = NormalCdf(Math.Min(x, y)) - BivariateNormalHighDeficit(x, y, rho);
            }
            else
            {
                // Φ₂(x, y; ρ) = Φ(x) - Φ₂(x, -y; -ρ) = (Φ(x) - Φ(min(x, -y))) + deficit(x, -y, -ρ)
                double interval = x > -y ? NormalInterval(-y, x) : 0;
                value = interval + BivariateNormalHighDeficit(x, -y, -rho);
            }

            return Math.Clamp(value, 0, 1);
        }

        // |ρ| < 0.925 向け Drezner–Wesolowsky (1990) 積分表示:
        // Φ₂(h, k; ρ) = Φ(h)Φ(k) + (1/2π) ∫₀^(asin ρ) exp((hk sin θ - (h² + k²)/2)/cos² θ) dθ
        private static double BivariateNormalModerate(double h, double k, double rho)
        {
            double absRho = Math.Abs(rho);
            (double[] nodes, double[] weights) = absRho < 0.3 ? BivariateNormalRule6
                : absRho < 0.75 ? BivariateNormalRule12
                : BivariateNormalRule20;

            double hk = h * k;
            double hs = 0.5 * (h * h + k * k);
            double asr = Math.Asin(rho);
            double sum = 0;
            for (int i = 0; i < nodes.Length; i++)
            {
                // θ = (asin ρ / 2)(1 ± xᵢ) の 2 点を同じ重みで加算する (hs ≥ |hk| より指数は 0 以下)
                for (int sign = -1; sign <= 1; sign += 2)
                {
                    double sn = Math.Sin(0.5 * asr * (1 + sign * nodes[i]));
                    sum += weights[i] * Math.Exp((hk * sn - hs) / ((1 - sn) * (1 + sn)));
                }
            }

            return NormalCdf(h) * NormalCdf(k) + 0.5 * asr * sum * InvTwoPi;
        }

        // 0.925 ≤ ρ < 1 向け Genz (2004) の差表示:
        // Φ(min(h, k)) - Φ₂(h, k; ρ) = (1/2π) ∫_ρ¹ exp(-(h² - 2rhk + k²)/(2(1 - r²)))/√(1 - r²) dr。
        // s = √(1 - r²) と置換し、s⁴ までの主要部を解析的に積分、残余を 20 点ガウス・ルジャンドル則で積分する
        private static double BivariateNormalHighDeficit(double h, double k, double rho)
        {
            // ρ ≥ 1/2 より 1 - ρ は丸め誤差なし
            double as2 = (1 - rho) * (1 + rho);
            double a = Math.Sqrt(as2);
            double hk = h * k;
            double bs = (h - k) * (h - k);
            double b = Math.Abs(h - k);
            double c = (4 - hk) / 8;
            double d = (12 - hk) / 16;

            // 主要部 e^(-hk/2) ∫₀ᵃ e^(-b²/(2s²)) (1 + c s² + c d s⁴) ds。
            // オーバーフロー・アンダーフロー回避のためミルズ比 M(t) = Q(t)/φ(t) を用いて e^(-(b²/a² + hk)/2) M(b/a)/√(2π) とする (b² ≥ -4hk より指数は 0 以下)
            double exponential = Math.Exp(-0.5 * (bs / as2 + hk));
            double p1 = 1 - c * (bs - as2) * (1 - d * bs / 5) / 3 + c * d * as2 * as2 / 5;
            double p2 = 1 - c * bs * (1 - d * bs / 5) / 3;
            double mills = SqrtHalfPi * Erfcx(b / a * InvSqrt2);
            double analytic = exponential * (a * p1 - b * mills * p2);

            // 残余積分。第 1 項の指数は -b²/(2s²) - hk/(1 + r) ≤ 0 にまとめることでオーバーフローを防ぐ
            double[] nodes = BivariateNormalRule20.Nodes;
            double[] weights = BivariateNormalRule20.Weights;
            double halfA = 0.5 * a;
            double remainder = 0;
            for (int i = 0; i < nodes.Length; i++)
            {
                for (int sign = -1; sign <= 1; sign += 2)
                {
                    double s = halfA * (1 + sign * nodes[i]);
                    double s2 = s * s;
                    double r = Math.Sqrt((1 - s) * (1 + s));
                    double exact = Math.Exp(-0.5 * bs / s2 - hk / (1 + r)) / r;
                    double approximation = Math.Exp(-0.5 * (bs / s2 + hk)) * (1 + c * s2 * (1 + d * s2));
                    remainder += weights[i] * (exact - approximation);
                }
            }

            return (analytic + halfA * remainder) * InvTwoPi;
        }

        // Φ(upper) - Φ(lower) (lower < upper) を裾確率の差として計算する
        private static double NormalInterval(double lower, double upper)
        {
            if (lower >= 0)
            {
                return NormalUpperTail(lower) - NormalUpperTail(upper);
            }

            if (upper <= 0)
            {
                return NormalUpperTail(-upper) - NormalUpperTail(-lower);
            }

            return 1 - NormalUpperTail(upper) - NormalUpperTail(-lower);
        }
    }
}
