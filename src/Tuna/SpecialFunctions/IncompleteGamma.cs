namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // これ以上の a で |x/a - 1| ≤ TemmeGammaMaxDeviation なら Temme の一様漸近展開を使う（中心付近の O(√a) 反復を回避）
        private const double TemmeGammaMinA = 1e4;

        // Temme の一様漸近展開を用いる |x/a - 1| の上限（|η| ≤ 0.34 で係数級数が速く収束）
        private const double TemmeGammaMaxDeviation = 0.3;

        // これ未満の q では Q(a, x) = q を ln Q・ln x 空間で解く（線形反復でのアンダーフローを回避）
        private const double InverseGammaLogTailQ = 1e-200;

        // 上側不完全ガンマ関数で Γ(a, x) を Γ(0, x) = E₁(x) とみなす |a| の上限（2⁻⁷⁰）
        private const double TinyGammaShape = 8.470329472543003e-22;

        // 一様漸近展開の係数 c_k(η) = Σ d_{k,n} ηⁿ（k = 0, …, 4、DLMF 8.12.12–8.12.13 から有理数で厳密算出）
        private static readonly double[][] TemmeGammaCoefficients =
        [
            [
                -0.3333333333333333, 0.08333333333333333, -0.014814814814814815, 0.0011574074074074073,
                0.0003527336860670194, -0.0001787551440329218, 3.919263178522438e-05, -2.185448510679992e-06,
                -1.85406221071516e-06, 8.296711340953087e-07, -1.7665952736826078e-07, 6.707853543401498e-09,
                1.0261809784240309e-08, -4.382036018453353e-09, 9.14769958223679e-10, -2.5514193994946248e-11,
                -5.830772132550426e-11, 2.4361948020667415e-11, -5.0276692801141755e-12, 1.1004392031956135e-13,
                3.371763262400985e-13, -1.392388722418162e-13, 2.8534893807047445e-14, -5.139111834242572e-16,
                -1.9752288294349442e-15, 8.099521156704561e-16,
            ],
            [
                -0.001851851851851852, -0.003472222222222222, 0.0026455026455026454, -0.0009902263374485596,
                0.00020576131687242798, -4.018775720164609e-07, -1.8098550334489977e-05, 7.64916091608111e-06,
                -1.6120900894563446e-06, 4.647127802807434e-09, 1.378633446915721e-07, -5.752545603517705e-08,
                1.1951628599778148e-08, -1.7543241719747647e-11, -1.0091543710600413e-09, 4.162792991842583e-10,
                -8.56390702649298e-11, 6.067215101604758e-14, 7.1624989648114856e-12, -2.933186643771437e-12,
            ],
            [
                0.004133597883597883, -0.0026813271604938273, 0.0007716049382716049, 2.0093878600823047e-06,
                -0.0001073665322636516, 5.2923448829120125e-05, -1.2760635188618728e-05, 3.423578734096138e-08,
                1.3721957309062934e-06, -6.298992138380055e-07, 1.4280614206064242e-07, -2.0477098421990866e-10,
                -1.409252991086752e-08, 6.228974084922022e-09,
            ],
            [
                0.0006494341563786008, 0.00022947209362139917, -0.0004691894943952557, 0.00026772063206283885,
                -7.561801671883977e-05, -2.396505113867297e-07, 1.1082654115347302e-05, -5.6749528269915965e-06,
                1.4230900732435883e-06, -2.7861080291528143e-11,
            ],
            [
                -0.0008618882909167117, 0.0007840392217200666, -0.0002990724803031902, -1.4638452578843418e-06,
                6.641498215465122e-05, -3.968365047179435e-05,
            ],
        ];

        /// <summary>
        /// Computes the regularized lower incomplete gamma function <c>P(a, x) = γ(a, x) / Γ(a)</c>.
        /// <para>正則化下側不完全ガンマ関数 <c>P(a, x) = γ(a, x) / Γ(a)</c> を計算します。</para>
        /// </summary>
        /// <param name="a">The shape parameter (<c>a &gt; 0</c>). / 形状パラメータ（<c>a &gt; 0</c>）。</param>
        /// <param name="x">The upper integration limit (<c>x ≥ 0</c>). / 積分上限（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>P(a, x)</c>, or <see cref="double.NaN"/> if any argument is outside its domain or <see cref="double.NaN"/>. / <c>P(a, x)</c> の値。引数が定義域外または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/8.2.E4"/>
        public static double RegularizedGammaP(double a, double x)
        {
            if (!TryIncompleteGammaSpecialCase(a, x, out double p, out _))
            {
                IncompleteGammaCore(a, x, out p, out _);
            }

            return p;
        }

        /// <summary>
        /// Computes the regularized upper incomplete gamma function <c>Q(a, x) = Γ(a, x) / Γ(a) = 1 - P(a, x)</c>.
        /// <para>正則化上側不完全ガンマ関数 <c>Q(a, x) = Γ(a, x) / Γ(a) = 1 - P(a, x)</c> を計算します。</para>
        /// </summary>
        /// <param name="a">The shape parameter (<c>a &gt; 0</c>). / 形状パラメータ（<c>a &gt; 0</c>）。</param>
        /// <param name="x">The lower integration limit (<c>x ≥ 0</c>). / 積分下限（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>Q(a, x)</c>, or <see cref="double.NaN"/> if any argument is outside its domain or <see cref="double.NaN"/>. / <c>Q(a, x)</c> の値。引数が定義域外または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/8.2.E4"/>
        public static double RegularizedGammaQ(double a, double x)
        {
            if (!TryIncompleteGammaSpecialCase(a, x, out _, out double q))
            {
                IncompleteGammaCore(a, x, out _, out q);
            }

            return q;
        }

        /// <summary>
        /// Computes the lower incomplete gamma function <c>γ(a, x) = ∫₀ˣ t^(a - 1) e⁻ᵗ dt</c>.
        /// <para>下側不完全ガンマ関数 <c>γ(a, x) = ∫₀ˣ t^(a - 1) e⁻ᵗ dt</c> を計算します。</para>
        /// </summary>
        /// <param name="a">The shape parameter (<c>a &gt; 0</c>). / 形状パラメータ（<c>a &gt; 0</c>）。</param>
        /// <param name="x">The upper integration limit (<c>x ≥ 0</c>). / 積分上限（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>γ(a, x)</c>, or <see cref="double.NaN"/> if any argument is outside its domain or <see cref="double.NaN"/>. / <c>γ(a, x)</c> の値。引数が定義域外または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/8.2.E1"/>
        public static double LowerIncompleteGamma(double a, double x)
        {
            if (double.IsNaN(a) || double.IsNaN(x) || a <= 0 || x < 0)
            {
                return double.NaN;
            }

            if (x == 0)
            {
                return 0;
            }

            if (double.IsPositiveInfinity(x))
            {
                return Gamma(a);
            }

            if (double.IsPositiveInfinity(a))
            {
                return double.NaN;
            }

            // 中心付近（Temme の一様漸近展開の範囲）では級数の反復が O(√a) 回必要になり、反復上限で打ち切られうるので、正則化した値から求める
            if ((x < a + 1 || x < 1.1) && !InTemmeGammaRegion(a, x))
            {
                // γ(a, x) = xᵃ e⁻ˣ Σ xⁿ / (a(a+1)…(a+n))
                return PowerTimesExp(a, x, LowerGammaSum(a, x));
            }

            if (a < 171)
            {
                IncompleteGammaCore(a, x, out _, out double q);
                return Gamma(a) * (1 - q);
            }

            // Γ(a) はオーバーフローするので対数で求める。中心付近の下側（Temme の範囲）では P がアンダーフローして 1 - Q = 0 になりうるが、ln P は有限に求まる
            LogIncompleteGammaCore(a, x, out double logP, out _);
            return Math.Exp(LogGamma(a) + logP);
        }

        /// <summary>
        /// Computes the upper incomplete gamma function <c>Γ(a, x) = ∫ₓ^∞ t^(a - 1) e⁻ᵗ dt</c>.
        /// <para>上側不完全ガンマ関数 <c>Γ(a, x) = ∫ₓ^∞ t^(a - 1) e⁻ᵗ dt</c> を計算します。</para>
        /// </summary>
        /// <param name="a">The shape parameter (any finite real value, including <c>a ≤ 0</c>). / 形状パラメータ（<c>a ≤ 0</c> を含む任意の有限実数）。</param>
        /// <param name="x">The lower integration limit (<c>x ≥ 0</c>). / 積分下限（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>Γ(a, x)</c>, <see cref="double.PositiveInfinity"/> when <c>x = 0</c> and <c>a ≤ 0</c>, or <see cref="double.NaN"/> if <paramref name="x"/> &lt; 0, <paramref name="a"/> is infinite, or either argument is <see cref="double.NaN"/>. / <c>Γ(a, x)</c> の値。<c>x = 0</c> かつ <c>a ≤ 0</c> では <see cref="double.PositiveInfinity"/>、<paramref name="x"/> &lt; 0、<paramref name="a"/> が無限大、またはいずれかの引数が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/8.2.E2"/>
        public static double UpperIncompleteGamma(double a, double x)
        {
            if (double.IsNaN(a) || double.IsNaN(x) || x < 0 || double.IsInfinity(a))
            {
                return double.NaN;
            }

            if (x == 0)
            {
                return a > 0 ? Gamma(a) : double.PositiveInfinity;
            }

            if (double.IsPositiveInfinity(x))
            {
                return 0;
            }

            if (a == 0 || Math.Abs(a) < TinyGammaShape)
            {
                // |∂ ln Γ(a, x)/∂a| は x ≥ 2⁻¹⁰⁷⁴ で約 745 以下なので、|a| < 2⁻⁷⁰ の Γ(a, x) は Γ(0, x) = E₁(x) と相対 1e-18 未満しか違わない。
                // 正の a で Γ(a)・Q(a, x) と分けると Γ(a) がオーバーフローし、Q が非正規化数や 0 になって積が ∞ や NaN になる
                return ExpIntegralE(1, x);
            }

            if (a > 0)
            {
                // 中心付近（Temme の一様漸近展開の範囲）では連分数の反復が O(√a) 回必要になるので、正則化した値から求める
                // （a ≥ 2⁵³ では a + 1 = a で中心 x = a も連分数の側になり、反復上限で打ち切られた値は NaN になる）
                if (x >= a + 1 && x >= 1.1 && !InTemmeGammaRegion(a, x))
                {
                    return PowerTimesExp(a, x, UpperGammaFraction(a, x));
                }

                if (a < 171)
                {
                    IncompleteGammaCore(a, x, out _, out double q);
                    return Gamma(a) * q;
                }

                // 下側不完全ガンマ関数と同じく、中心付近の上側で Q がアンダーフローしても ln Q は有限に求まる
                LogIncompleteGammaCore(a, x, out _, out double logQ);
                return Math.Exp(LogGamma(a) + logQ);
            }

            if (IsInteger(a) && 1 - a < int.MaxValue)
            {
                // Γ(-n, x) = x⁻ⁿ Eₙ₊₁(x)
                return PowerTimes(x, a, ExpIntegralE((int)(1 - a), x));
            }

            if (x >= 1)
            {
                return PowerTimesExp(a, x, UpperGammaFraction(a, x));
            }

            // Γ(a, x) = Γ(a) - Σ (-1)ⁿ x^(a+n) / (n! (a+n))
            // a が負の整数 -m に近いと Γ(a) と n = m の項がともに極に近づいて桁落ちするので、
            // その 2 つは pole で除いて UpperGammaPoleTerms でまとめて計算する
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

            double poleTerms = pole < 0 ? Gamma(a) : UpperGammaPoleTerms(pole, epsilon, x);
            return poleTerms - PowerTimes(x, a, sum);
        }

        // xᵃ·v を xᵃ のみのオーバーフローを避けて計算する（x^(a/2)·v·x^(a/2)）
        private static double PowerTimes(double x, double a, double v)
        {
            double power = Math.Pow(x, a);
            if (!double.IsInfinity(power))
            {
                return power * v;
            }

            double half = Math.Pow(x, 0.5 * a);
            return half * v * half;
        }

        // a = -m + ε（|ε| ≤ 1/4）での Γ(a) - (-1)ᵐ x^(a+m) / (m! (a+m))（0 < x < 1）
        private static double UpperGammaPoleTerms(int m, double epsilon, double x)
        {
            // Γ(ε - m) = (-1)ᵐ Γ(1+ε) / (m! ε Π(1 - ε/j)) より、
            // 2 つの極をまとめると (-1)ᵐ (Γ(1+ε)/Π(1 - ε/j) - x^ε) / (m! ε) となり、括弧内は ε に比例する
            double logRatio = LogGamma1pSeries(epsilon);
            for (int j = 1; j <= m; j++)
            {
                logRatio -= Log1p(-epsilon / j);
            }

            double value = (Expm1(logRatio) - Expm1(epsilon * Math.Log(x))) / epsilon / FactorialTable[m];
            return m % 2 == 0 ? value : -value;
        }

        /// <summary>
        /// Computes the inverse of the regularized lower incomplete gamma function, finding <c>x</c> such that <c>P(a, x) = p</c>.
        /// <para>正則化下側不完全ガンマ関数の逆関数（<c>P(a, x) = p</c> を満たす <c>x</c>）を計算します。</para>
        /// </summary>
        /// <param name="a">The shape parameter (<c>a &gt; 0</c>). / 形状パラメータ（<c>a &gt; 0</c>）。</param>
        /// <param name="p">The lower probability (<c>0 ≤ p ≤ 1</c>). / 下側確率（<c>0 ≤ p ≤ 1</c>）。</param>
        /// <returns>The value <c>x ≥ 0</c>, <see cref="double.PositiveInfinity"/> when <c>p = 1</c>, or <see cref="double.NaN"/> if any argument is outside its domain or <see cref="double.NaN"/>. / <c>x ≥ 0</c> の値。<c>p = 1</c> では <see cref="double.PositiveInfinity"/>、引数が定義域外または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/8.2.E4"/>
        public static double InverseRegularizedGammaP(double a, double p) =>
            InverseIncompleteGamma(a, p, 1 - p);

        /// <summary>
        /// Computes the inverse of the regularized upper incomplete gamma function, finding <c>x</c> such that <c>Q(a, x) = q</c>.
        /// <para>正則化上側不完全ガンマ関数の逆関数（<c>Q(a, x) = q</c> を満たす <c>x</c>）を計算します。</para>
        /// </summary>
        /// <param name="a">The shape parameter (<c>a &gt; 0</c>). / 形状パラメータ（<c>a &gt; 0</c>）。</param>
        /// <param name="q">The upper probability (<c>0 ≤ q ≤ 1</c>). / 上側確率（<c>0 ≤ q ≤ 1</c>）。</param>
        /// <returns>The value <c>x ≥ 0</c>, <see cref="double.PositiveInfinity"/> when <c>q = 0</c>, or <see cref="double.NaN"/> if any argument is outside its domain or <see cref="double.NaN"/>. / <c>x ≥ 0</c> の値。<c>q = 0</c> では <see cref="double.PositiveInfinity"/>、引数が定義域外または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/8.2.E4"/>
        public static double InverseRegularizedGammaQ(double a, double q) =>
            InverseIncompleteGamma(a, 1 - q, q);

        private static bool TryIncompleteGammaSpecialCase(double a, double x, out double p, out double q)
        {
            if (double.IsNaN(a) || double.IsNaN(x) || a <= 0 || x < 0
                || (double.IsPositiveInfinity(a) && double.IsPositiveInfinity(x)))
            {
                p = q = double.NaN;
                return true;
            }

            if (x == 0 || double.IsPositiveInfinity(a))
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

            p = q = 0;
            return false;
        }

        // P(a, x) と Q(a, x) を小さい方から直接計算する（a > 0, 0 < x < ∞）
        private static void IncompleteGammaCore(double a, double x, out double p, out double q)
        {
            IncompleteGammaCore(a, x, out p, out q, out _);
        }

        // 逆関数の導関数用に前因子 prefix も返す（前因子を用いない経路では NaN）
        private static void IncompleteGammaCore(double a, double x, out double p, out double q, out double prefix)
        {
            prefix = double.NaN;
            if (InTemmeGammaRegion(a, x))
            {
                IncompleteGammaTemme(a, x, out p, out q);
                return;
            }

            if (x < 1.1)
            {
                if (UseLowerGammaSeriesForSmallX(a, x))
                {
                    double sum = LowerGammaSum(a, x);
                    prefix = GammaPrefix(a, x);
                    p = sum * prefix;
                    q = 1 - p;
                }
                else
                {
                    q = UpperGammaSmallA(a, x);
                    p = 1 - q;
                }

                return;
            }

            if (x < a + 1)
            {
                double sum = LowerGammaSum(a, x);
                prefix = GammaPrefix(a, x);
                p = sum * prefix;
                q = 1 - p;
            }
            else
            {
                double fraction = UpperGammaFraction(a, x);
                prefix = GammaPrefix(a, x);
                q = fraction * prefix;
                p = 1 - q;
            }
        }

        // x < 1.1 で P(a, x) の級数を用いるか判定する（false なら UpperGammaSmallA で Q(a, x) を計算）
        private static bool UseLowerGammaSeriesForSmallX(double a, double x) =>
            x < 0.5 ? -0.4 / Math.Log(x) < a : 0.75 * x < a;

        // Temme の一様漸近展開を用いる範囲（大きな a で x ≈ a）かどうか
        private static bool InTemmeGammaRegion(double a, double x) =>
            a >= TemmeGammaMinA && Math.Abs((x - a) / a) <= TemmeGammaMaxDeviation;

        // Temme の一様漸近展開（DLMF 8.12.3–8.12.8）による P(a, x) と Q(a, x)
        private static void IncompleteGammaTemme(double a, double x, out double p, out double q)
        {
            double factor = TemmeGammaTailFactor(a, x, out double halfEta2, out bool upper);
            double exponential = Math.Exp(-a * halfEta2);
            if (upper)
            {
                q = exponential * factor;
                p = 1 - q;
            }
            else
            {
                p = exponential * factor;
                q = 1 - p;
            }
        }

        // Temme の一様漸近展開で小さい方の裾を e^(-aη²/2) · factor に分離し factor を返す（upper = true で Q 側、対数版でも再利用）
        private static double TemmeGammaTailFactor(double a, double x, out double halfEta2, out bool upper)
        {
            // λ - 1 = (x - a)/a、η²/2 = λ - 1 - ln λ（η は λ - 1 と同符号）
            double mu = (x - a) / a;
            halfEta2 = -Log1pmx(mu);
            double eta = Math.CopySign(Math.Sqrt(2 * halfEta2), mu);

            double sum = 0;
            double power = 1;
            foreach (double[] row in TemmeGammaCoefficients)
            {
                double c = 0;
                for (int n = row.Length - 1; n >= 0; n--)
                {
                    c = c * eta + row[n];
                }

                sum += c * power;
                power /= a;
            }

            // erfc(t)/2 = e^(-t²) erfcx(t)/2（t = |η|√(a/2)、t² = aη²/2）として、小さい方の裾を e^(-aη²/2) でくくる
            double t = Math.Abs(eta) * Math.Sqrt(0.5 * a);
            double remainder = sum / Math.Sqrt(2 * Math.PI) / Math.Sqrt(a);
            upper = eta >= 0;
            return upper ? 0.5 * Erfcx(t) + remainder : 0.5 * Erfcx(t) - remainder;
        }

        // xᵃ e⁻ˣ / Γ(a) をオーバーフロー・桁落ちを避けて計算する（a > 0, x > 0）
        internal static double GammaPrefix(double a, double x)
        {
            if (a >= 10)
            {
                // xᵃ e⁻ˣ / Γ(a) = √(a/2π) exp(-a φ(x/a) - δ(a))、φ(t) = t - 1 - ln t
                double u = (x - a) / a;
                double phi = Math.Abs(u) < 0.5 ? -Log1pmx(u) : u - Math.Log(x / a);
                return Math.Sqrt(a / (2 * Math.PI)) * Math.Exp(-a * phi - StirlingCorrection(a));
            }

            if (x < 700)
            {
                double direct = Math.Pow(x, a) * Math.Exp(-x) * ReciprocalGamma(a);
                if (direct > 1e-290 && double.IsFinite(direct))
                {
                    return direct;
                }
            }

            return Math.Exp(a * Math.Log(x) - x - LogGamma(a));
        }

        // xᵃ e⁻ˣ · factor をオーバーフロー・アンダーフローを避けて計算する
        private static double PowerTimesExp(double a, double x, double factor)
        {
            double direct = Math.Pow(x, a) * Math.Exp(-x) * factor;
            if (double.IsFinite(direct) && Math.Abs(direct) > 1e-290)
            {
                return direct;
            }

            // Math.Sign は NaN で例外を送出するので、符号は CopySign で付ける（NaN はそのまま返る）
            return Math.CopySign(Math.Exp(a * Math.Log(x) - x + Math.Log(Math.Abs(factor))), factor);
        }

        // 級数 Σ xⁿ / (a(a+1)…(a+n))（P(a, x) の級数表示の和の部分）
        private static double LowerGammaSum(double a, double x, bool boundRemainder = false)
        {
            double term = 1 / a;
            double sum = term;
            double ap = a;
            for (int n = 0; n < MaxIterations; n++)
            {
                ap += 1;
                term *= x / ap;
                sum += term;
                double tolerance = sum * Epsilon;
                // 残りの項の比は x/(ap+1) 以下なので、等比級数で残差を抑える。
                if (term <= tolerance && (!boundRemainder || term <= tolerance * (1 - x / (ap + 1))))
                {
                    break;
                }
            }

            return sum;
        }

        // 連分数 Γ(a, x) = xᵃ e⁻ˣ / (x + 1 - a - 1(1-a)/(x + 3 - a - …)) の連分数部分
        private static double UpperGammaFraction(double a, double x)
        {
            double b = x + 1 - a;
            double c = 1 / Tiny;
            double d = 1 / b;
            double h = d;
            for (int i = 1; i < MaxIterations; i++)
            {
                double an = -i * (i - a);
                b += 2;
                d = an * d + b;
                if (Math.Abs(d) < Tiny)
                {
                    d = Tiny;
                }

                c = b + an / c;
                if (Math.Abs(c) < Tiny)
                {
                    c = Tiny;
                }

                d = 1 / d;
                double delta = d * c;
                h *= delta;
                if (Math.Abs(delta - 1) <= Epsilon)
                {
                    break;
                }
            }

            return h;
        }

        // a が小さく x < 1.1 のときの Q(a, x)
        private static double UpperGammaSmallA(double a, double x)
        {
            // a Γ(a, x) = (Γ(1+a) - 1) - (xᵃ - 1) - a xᵃ Σ (-x)ⁿ / (n! (a+n))（n ≥ 1）
            double gamma1pm1 = Gamma1pm1(a);
            double powerMinusOne = Expm1(a * Math.Log(x));
            double sum = 0;
            double term = 1;
            for (int n = 1; n < MaxIterations; n++)
            {
                term *= -x / n;
                double t = term / (a + n);
                sum += t;
                if (Math.Abs(t) <= Epsilon * Math.Abs(sum))
                {
                    break;
                }
            }

            double scaledUpper = gamma1pm1 - powerMinusOne - a * (powerMinusOne + 1) * sum;
            return scaledUpper / (1 + gamma1pm1);
        }

        // P(a, x) = p（Q(a, x) = q）を満たす x をハレー法で求める
        private static double InverseIncompleteGamma(double a, double p, double q)
        {
            if (double.IsNaN(a) || double.IsNaN(p) || double.IsNaN(q) || a <= 0 || double.IsInfinity(a)
                || p < 0 || q < 0)
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

            bool useQ = q < p;

            // 初期値（DiDonato & Morris、Numerical Recipes）
            double x;
            if (a > 1)
            {
                // Wilson–Hilferty 近似。z は正規分布の p 分位点
                double z = useQ ? Math.Sqrt(2) * InverseErfc(2 * q) : -Math.Sqrt(2) * InverseErfc(2 * p);
                double t = 1 - 1 / (9 * a) + z / (3 * Math.Sqrt(a));
                x = a * t * t * t;
                if (!(x > 0) || !double.IsFinite(x))
                {
                    // P(a, x) ≈ xᵃ / Γ(a + 1)
                    x = Math.Exp((Math.Log(p) + LogGamma(a + 1)) / a);
                }
            }
            else
            {
                double t = 1 - a * (0.253 + a * 0.12);
                x = p < t ? Math.Exp(Math.Log(p / t) / a) : 1 - Math.Log(q / (1 - t));
                if (!(x > 0) || !double.IsFinite(x))
                {
                    // 微小な a では t と p = 1 - q が 1 に丸まり、上の式は 1 - t = 0 や負の x を与える。
                    // 1 - t を直接求め、下側の近似 (p/t)^(1/a) を対数の差 ln p - ln t で求める
                    double oneMinusT = a * (0.253 + a * 0.12);
                    double logRatio = (useQ ? Log1p(-q) : Math.Log(p)) - Log1p(-oneMinusT);
                    x = logRatio < 0 ? Math.Exp(logRatio / a) : 1 - Math.Log(q / oneMinusT);
                }
            }

            // a ≤ 1 では解が x > 100 にある場合に限る（d ln Q/d ln x ≈ -x が小さいと、ln Q の絶対誤差が x の相対誤差に拡大する）
            if (useQ && q < InverseGammaLogTailQ && (a > 1 || LogRegularizedGammaQ(a, 100) > Math.Log(q)))
            {
                return InverseGammaUpperTailFromLog(a, Math.Log(q), x > 0 && double.IsFinite(x) ? x : -Math.Log(q));
            }

            if (x == 0)
            {
                return 0;
            }

            double lower = 0;
            double upper = double.PositiveInfinity;
            for (int iteration = 0; iteration < 200; iteration++)
            {
                IncompleteGammaCore(a, x, out double px, out double qx, out double prefix);
                double error = useQ ? q - qx : px - p;
                if (error == 0)
                {
                    return x;
                }

                if (error > 0)
                {
                    upper = x;
                }
                else
                {
                    lower = x;
                }

                // dP/dx = x^(a-1) e⁻ˣ / Γ(a)、(d²P/dx²)/(dP/dx) = (a-1)/x - 1
                double derivative = (double.IsNaN(prefix) ? GammaPrefix(a, x) : prefix) / x;
                double next;
                if (derivative > 0 && double.IsFinite(derivative))
                {
                    double u = error / derivative;
                    next = x - u / (1 - 0.5 * Math.Min(1, u * ((a - 1) / x - 1)));
                }
                else
                {
                    next = double.NaN;
                }

                // 収束の判定は区間の判定より先に行う（x は直前に区間の端になっているので、収束すると next が端に一致する）
                if (Math.Abs(next - x) <= 4 * Epsilon * x)
                {
                    return next;
                }

                if (!(next > lower && next < upper))
                {
                    next = double.IsPositiveInfinity(upper) ? 2 * x : 0.5 * (lower + upper);
                }

                x = next;
            }

            return x;
        }

        // 上側の裾（Q がアンダーフローしうる領域）で ln Q(a, x) = logQ を t = ln x のニュートン法で解く
        private static double InverseGammaUpperTailFromLog(double a, double logQ, double x)
        {
            double lower = 0;
            double upper = double.PositiveInfinity;
            for (int iteration = 0; iteration < 200; iteration++)
            {
                double logQx = LogRegularizedGammaQ(a, x);
                double error = logQx - logQ;
                if (error == 0)
                {
                    return x;
                }

                if (error > 0)
                {
                    lower = x;
                }
                else
                {
                    upper = x;
                }

                double slope = -Math.Exp(LogGammaPrefix(a, x) - logQx);
                double step = -error / slope;
                double next = x * Math.Exp(step);
                if (Math.Abs(step) <= 4 * Epsilon)
                {
                    return next;
                }

                if (!(next > lower && next < upper))
                {
                    next = double.IsPositiveInfinity(upper) ? 2 * x : lower == 0 ? 0.5 * x : Math.Sqrt(lower) * Math.Sqrt(upper);
                }

                x = next;
            }

            return x;
        }
    }
}
