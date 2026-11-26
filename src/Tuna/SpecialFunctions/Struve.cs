namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // StruveKQuadrature の exp-sinh 則の刻み幅（x ≥ 10 では離散化誤差が 1e-18 未満）
        private const double StruveQuadratureStep = 1.0 / 32;

        // exp-sinh 則 τ = exp((π/2) sinh s) の節点と重み（s = jh、j = -160..96）。
        // s = -5 で τ ≈ 1e-51、s = 3 で τ ≈ 7e6 となり両端の打ち切り誤差は無視できる
        private static readonly (double[] Nodes, double[] Weights) StruveExpSinhRule = CreateStruveExpSinhRule();

        /// <summary>
        /// Computes the Struve function <c>H_ν(x)</c>.
        /// <para>シュトルーベ関数 <c>H_ν(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="nu">The finite real order <c>ν</c>. / 有限の実数次数 <c>ν</c>。</param>
        /// <param name="x">The argument (<c>x ≥ 0</c>, or any real value when <paramref name="nu"/> is an integer). / 引数（<c>x ≥ 0</c>。<paramref name="nu"/> が整数の場合は負の値も指定可能です）。</param>
        /// <returns>The value of <c>H_ν(x)</c>, or <see cref="double.NaN"/> if parameters are out of domain or <paramref name="nu"/> is non-finite. / <c>H_ν(x)</c> の値。引数が定義域外、または <paramref name="nu"/> が有限でない場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>For integer orders, satisfies <c>H_n(-x) = (-1)^(n+1) H_n(x)</c>. At <c>x = 0</c>, returns <c>0</c> for <c>ν &gt; -1</c> or <c>ν = -n - 1/2</c>, <c>2 / π</c> for <c>ν = -1</c>, and signed infinity otherwise. At <c>x = +∞</c>, returns <c>0</c> for <c>ν &lt; 1</c>, <c>2 / π</c> for <c>ν = 1</c>, and <see cref="double.PositiveInfinity"/> for <c>ν &gt; 1</c>.</para>
        /// <para>整数次数では <c>H_n(-x) = (-1)^(n+1) H_n(x)</c> を満たします。<c>x = 0</c> では <c>ν &gt; -1</c> または <c>ν = -n - 1/2</c> で <c>0</c>、<c>ν = -1</c> で <c>2 / π</c>、それ以外では符号付き無限大を返します。<c>x = +∞</c> では <c>ν &lt; 1</c> で <c>0</c>、<c>ν = 1</c> で <c>2 / π</c>、<c>ν &gt; 1</c> で <see cref="double.PositiveInfinity"/> を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/11.2.E1"/>
        public static double StruveH(double nu, double x) => StruveCore(nu, x, false);

        /// <summary>
        /// Computes the modified Struve function <c>L_ν(x)</c>.
        /// <para>変形シュトルーベ関数 <c>L_ν(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="nu">The finite real order <c>ν</c>. / 有限の実数次数 <c>ν</c>。</param>
        /// <param name="x">The argument (<c>x ≥ 0</c>, or any real value when <paramref name="nu"/> is an integer). / 引数（<c>x ≥ 0</c>。<paramref name="nu"/> が整数の場合は負の値も指定可能です）。</param>
        /// <returns>The value of <c>L_ν(x)</c>, or <see cref="double.NaN"/> if parameters are out of domain or <paramref name="nu"/> is non-finite. / <c>L_ν(x)</c> の値。引数が定義域外、または <paramref name="nu"/> が有限でない場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>For integer orders, satisfies <c>L_n(-x) = (-1)^(n+1) L_n(x)</c>. At <c>x = 0</c>, follows the same limits as <see cref="StruveH"/>. Overflows to <see cref="double.PositiveInfinity"/> for <c>x ≳ 713</c>.</para>
        /// <para>整数次数では <c>L_n(-x) = (-1)^(n+1) L_n(x)</c> を満たします。<c>x = 0</c> での極限は <see cref="StruveH"/> と同じです。<c>x ≳ 713</c> では <see cref="double.PositiveInfinity"/> にオーバーフローします。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/11.2.E2"/>
        public static double StruveL(double nu, double x) => StruveCore(nu, x, true);

        // StruveH（modified = false）および StruveL（modified = true）の本体
        private static double StruveCore(double nu, double x, bool modified)
        {
            if (double.IsNaN(nu) || double.IsNaN(x) || double.IsInfinity(nu))
            {
                return double.NaN;
            }

            if (x < 0)
            {
                if (!IsInteger(nu))
                {
                    return double.NaN;
                }

                // 級数は (x/2)^(n+1) × (x² の級数) なので、整数次数では (-1)^(n+1) の偶奇性を持つ
                double value = StruveCore(nu, -x, modified);
                return IsOddInteger(nu) ? value : -value;
            }

            // ν = -n - 1/2 では Γ(k+ν+3/2) の極で級数の初めの n 項が消え、ベッセル関数に帰着する（DLMF 11.4.4, 11.4.5）
            if (nu < 0 && IsInteger(nu + 0.5))
            {
                double order = -nu;
                if (modified)
                {
                    return BesselI(order, x);
                }

                double j = BesselJ(order, x);
                return IsOddInteger(order - 0.5) ? -j : j;
            }

            if (x == 0)
            {
                if (nu > -1)
                {
                    return 0;
                }

                if (nu == -1)
                {
                    return 2 / Math.PI;
                }

                // 先頭項 (x/2)^(ν+1)/(Γ(3/2) Γ(ν+3/2)) が発散する（ν + 3/2 は極ではない）
                return SignGamma(nu + 1.5) * double.PositiveInfinity;
            }

            if (double.IsPositiveInfinity(x))
            {
                if (modified)
                {
                    return double.PositiveInfinity;
                }

                // H_ν = Y_ν + K_ν、K_ν ~ (x/2)^(ν-1)/(√π Γ(ν+1/2))
                return nu < 1 ? 0 : nu == 1 ? 2 / Math.PI : double.PositiveInfinity;
            }

            double result;
            if (modified)
            {
                if (x >= 30 && StruveLAsymptotic(nu, x, out result))
                {
                    return result;
                }

                // ν > -3/2 では項がすべて正。ν ≤ -3/2 で桁落ちが大きい場合も、漸近展開が使えないのでこの値を返す
                StruveSeries(nu, x, true, out result);
                return result;
            }

            // x が大きいと冪級数は e^x 程度の桁落ちを起こすので、先に漸近展開を試す
            if (x > 45 && StruveHAsymptotic(nu, x, out result))
            {
                return result;
            }

            // x > 30 で ν < x/2 なら級数の桁落ちはほぼ確実に 10¹³ を超えるので、級数を試さない
            if (!(x > 30 && nu < 0.5 * x) && StruveSeries(nu, x, false, out result))
            {
                return result;
            }

            if (x >= 12 && x <= 45 && StruveHAsymptotic(nu, x, out result))
            {
                return result;
            }

            return BesselY(nu, x) + StruveKQuadrature(nu, x);
        }

        // 冪級数 (x/2)^(ν+1) Σ (±1)ᵏ (x/2)^(2k)/(Γ(k+3/2) Γ(k+ν+3/2))（modified が真なら全項正符号）。
        // double-double で和を取り、絶対値和との比が 10¹³ 以下で収束した場合に true を返す
        private static bool StruveSeries(double nu, double x, bool modified, out double value)
        {
            double halfX = 0.5 * x;
            double qHi = halfX * halfX;
            (double Hi, double Lo) q = (qHi, Math.FusedMultiplyAdd(halfX, halfX, -qHi));
            if (!modified)
            {
                q = (-q.Hi, -q.Lo);
            }

            (double Hi, double Lo) sum = (1.0, 0.0);
            (double Hi, double Lo) term = (1.0, 0.0);
            double absSum = 1;

            // k + ν + 3/2 < 0 の間は 1/Γ(k+ν+3/2) の符号が交互に変わり、項の比が 1 未満でも打ち切れない
            double lastSignChange = -(nu + 1.5);
            bool converged = false;
            for (int k = 0; k < 4000; k++)
            {
                double a = k + 1.5;
                var c = DoubleDoubleTwoSum(nu, a);
                var denominator = DoubleDoubleMultiply(c, (a, 0.0));
                term = DoubleDoubleDivide(DoubleDoubleMultiply(term, q), denominator);
                sum = DoubleDoubleAdd(sum, term);
                absSum += Math.Abs(term.Hi);
                if (k + 1 > lastSignChange && Math.Abs(denominator.Hi) > Math.Abs(qHi)
                    && Math.Abs(term.Hi) <= DoubleDoubleEpsilon * Math.Abs(sum.Hi))
                {
                    converged = true;
                    break;
                }
            }

            // (x/2)^(ν+1)/(Γ(3/2) Γ(ν+3/2))、1/Γ(3/2) = 2/√π
            double mantissa = StruvePowerOverGamma(halfX, nu, 1, 1.5, out long exponent) * TwoOverSqrtPi;
            value = StruveScaleProduct(mantissa, exponent, sum.Hi);
            return converged && absSum <= 1e13 * Math.Abs(sum.Hi);
        }

        // DLMF 11.6.1: 大きい x での H_ν = Y_ν + K_ν の漸近展開（最小項が ε/4 以下になる前に発散し始めたら false）
        private static bool StruveHAsymptotic(double nu, double x, out double value)
        {
            if (!StruveAsymptoticSum(nu, x, false, out double sum))
            {
                value = double.NaN;
                return false;
            }

            double mantissa = StruvePowerOverGamma(0.5 * x, nu, -1, 0.5, out long exponent) * OneOverSqrtPi;
            value = BesselY(nu, x) + StruveScaleProduct(mantissa, exponent, sum);
            return true;
        }

        // DLMF 11.6.2: 大きい x での L_ν = I_ν + M_ν の漸近展開。
        // I_ν e^(-x) に M_ν e^(-x) を加算してから e^x を掛け、打ち切り誤差が和の ε/4 以下にならなければ false を返す
        private static bool StruveLAsymptotic(double nu, double x, out double value)
        {
            value = double.NaN;
            double scaledI = BesselIScaled(nu, x);
            double mantissa = -StruvePowerOverGamma(0.5 * x, nu, -1, 0.5, out long exponent) * OneOverSqrtPi;

            // M_ν e^(-x) の前因子。e^(-x) は x > 700 で 2 回に分けて掛け、途中のアンダーフローを避ける
            double halfDecay = Math.Exp(-0.5 * x);
            double scaledPrefactor = ScaleB(mantissa * halfDecay, exponent) * halfDecay;

            // 漸近展開の各項は I_ν との和に対して ε/4 以下まで加算する
            double sum = 1;
            double term = 1;
            double previous = double.PositiveInfinity;
            double r = 4 / (x * x);
            for (int k = 0; k < 500; k++)
            {
                double magnitude = Math.Abs(scaledPrefactor * term);
                if (magnitude <= 0.25 * Epsilon * Math.Abs(scaledI + scaledPrefactor * sum) || term == 0)
                {
                    double total = scaledI + scaledPrefactor * sum;
                    double halfGrowth = Math.Exp(0.5 * x);
                    value = total * halfGrowth * halfGrowth;
                    return double.IsFinite(scaledPrefactor);
                }

                if (Math.Abs(term) > previous)
                {
                    return false;
                }

                previous = Math.Abs(term);
                term *= -(k + 0.5) * (nu - 0.5 - k) * r;
                sum += term;
            }

            return false;
        }

        // 漸近展開の和 Σ (±1)ᵏ sₖ（s₀ = 1, sₖ₊₁ = sₖ (k+1/2)(ν-1/2-k)(2/x)²）。
        // ν + 1/2 が正の整数なら有限和で終了する
        private static bool StruveAsymptoticSum(double nu, double x, bool alternating, out double sum)
        {
            sum = 1;
            double term = 1;
            double r = 4 / (x * x);
            for (int k = 0; k < 500; k++)
            {
                double next = term * (k + 0.5) * (nu - 0.5 - k) * r;
                if (alternating)
                {
                    next = -next;
                }

                if (Math.Abs(next) > Math.Abs(term))
                {
                    return false;
                }

                sum += next;
                term = next;
                if (Math.Abs(term) <= 0.25 * Epsilon * Math.Abs(sum))
                {
                    return true;
                }
            }

            return false;
        }

        // DLMF 11.5.2: K_ν(x) = H_ν(x) - Y_ν(x) を t = τ/x の変数変換と exp-sinh 則で積分評価する。
        // 被積分関数は正のため相対精度で得られる（Γ(ν+1/2) の極では 0）
        private static double StruveKQuadrature(double nu, double x)
        {
            double mantissa = StruvePowerOverGamma(0.5 * x, nu, 0, 0.5, out long exponent);
            if (mantissa == 0)
            {
                return 0;
            }

            double[] nodes = StruveExpSinhRule.Nodes;
            double[] weights = StruveExpSinhRule.Weights;
            double power = nu - 0.5;
            double invX = 1 / x;
            double sum = 0;
            for (int i = 0; i < nodes.Length; i++)
            {
                double tau = nodes[i];
                if (tau > 800)
                {
                    break;
                }

                double u = tau * invX;
                sum += weights[i] * Math.Exp(power * Log1p(u * u) - tau);
            }

            return StruveScaleProduct(mantissa * TwoOverSqrtPi, exponent, sum * invX);
        }

        private static (double[] Nodes, double[] Weights) CreateStruveExpSinhRule()
        {
            const int first = -160;
            const int last = 96;
            var nodes = new double[last - first + 1];
            var weights = new double[nodes.Length];
            for (int j = first; j <= last; j++)
            {
                double s = j * StruveQuadratureStep;
                double tau = Math.Exp(0.5 * Math.PI * Math.Sinh(s));
                nodes[j - first] = tau;
                weights[j - first] = StruveQuadratureStep * tau * 0.5 * Math.PI * Math.Cosh(s);
            }

            return (nodes, weights);
        }

        // (x/2)^(ν+m)/Γ(ν+a) = mantissa · 2^exponent の仮数を返す（ν + a が非正整数なら 0）。
        // べきとガンマ関数を仮数・指数で分離し、ν + a の丸め残りを 1 - δ ψ(g) で補正する（|ν| > 10⁵ は対数評価）
        private static double StruvePowerOverGamma(double halfX, double nu, int m, double a, out long exponent)
        {
            exponent = 0;
            var g = DoubleDoubleTwoSum(nu, a);
            if (g.Lo == 0 && IsNonPositiveInteger(g.Hi))
            {
                return 0;
            }

            if (Math.Abs(nu) > 1e5)
            {
                double log = (nu + m) * Math.Log(halfX) - LogGamma(g.Hi);
                if (double.IsNaN(log))
                {
                    return double.NaN;
                }

                double e = Math.Floor(log / Ln2);
                exponent = (long)Math.Clamp(e, -1e17, 1e17);
                return SignGamma(g.Hi) * Math.Exp(log - e * Ln2);
            }

            double n = Math.Floor(nu);
            double power = ScaledPower(halfX, (long)n + m, out long powerExponent) * Math.Pow(halfX, nu - n);
            double gamma = ScaledGammaSigned(g.Hi, out long gammaExponent);
            exponent = powerExponent - gammaExponent;
            double correction = g.Lo == 0 ? 1 : 1 - g.Lo * Digamma(g.Hi);
            return power / gamma * correction;
        }

        // mantissa · 2^exponent · value（value の指数を分離して途中のオーバーフローやアンダーフローを防ぐ）
        private static double StruveScaleProduct(double mantissa, long exponent, double value)
        {
            if (value == 0 || !double.IsFinite(value))
            {
                return mantissa * value;
            }

            int shift = Math.ILogB(value);
            return ScaleB(mantissa * Math.ScaleB(value, -shift), exponent + shift);
        }
    }
}
