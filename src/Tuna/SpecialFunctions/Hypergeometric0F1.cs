namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        /// <summary>
        /// Computes the confluent hypergeometric limit function <c>₀F₁(; b; x)</c>.
        /// <para>合流型極限超幾何関数 <c>₀F₁(; b; x)</c> を計算します。</para>
        /// </summary>
        /// <param name="b">The denominator parameter (real value other than a non-positive integer). / 分母パラメータ（0 以下の整数を除く実数）。</param>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>₀F₁(; b; x)</c>, or <see cref="double.NaN"/> if <paramref name="b"/> is a non-positive integer or non-finite, or if <paramref name="x"/> is <see cref="double.NaN"/>. / <c>₀F₁(; b; x)</c> の値。<paramref name="b"/> が 0 以下の整数または有限でない場合、または <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/16.2.E1"/>
        public static double Hypergeometric0F1(double b, double x)
        {
            if (double.IsNaN(b) || double.IsNaN(x) || IsNonPositiveInteger(b) || double.IsInfinity(b))
            {
                return double.NaN;
            }

            if (x == 0)
            {
                return 1;
            }

            double absX = Math.Abs(x);
            bool series = x > 0
                ? absX <= Math.Max(16, 4 * Math.Abs(b)) || (b > 0 && absX <= 1e4)
                : absX <= 2 || (b > 0 ? absX <= b : absX <= Math.Max(16, 4 * -b));
            if (double.IsFinite(x) && series)
            {
                return Hypergeometric0F1Series(b, x);
            }

            if (double.IsPositiveInfinity(x))
            {
                return b > 0 ? double.PositiveInfinity : double.NaN;
            }

            if (double.IsNegativeInfinity(x))
            {
                return double.NaN;
            }

            // √|x| = root + rootLo（rootLo は丸め誤差）。z = 2 root のずれは関数の対数微分を掛けて補正する
            double root = Math.Sqrt(absX);
            double rootLo = Math.FusedMultiplyAdd(-root, root, absX) / (2 * root);
            double z = 2 * root;
            double dz = 2 * rootLo;
            double nu = b - 1;

            // Γ(b) |x|^(-ν/2) を仮数と指数に分けて求める（対数で掛けると |ln Γ(b)|·ε の誤差が乗る）
            double mantissa = ScaledGammaSigned(b, out long exponent);
            double half = 0.5 * nu;
            double n = Math.Round(half);
            if (Math.Abs(n) > long.MaxValue / 2)
            {
                return double.NaN;
            }

            mantissa *= ScaledPower(absX, -(long)n, out long powerExponent) * Math.Pow(absX, n - half);
            exponent += powerExponent;

            double value;
            if (x > 0)
            {
                // I_ν(z) = e^z Iˢ_ν(z)。d ln I_ν/dz ≈ √(1 + (ν/z)²)（誤差 O(1/z)、dz ~ zε なので十分）
                double scaled = BesselIScaled(nu, z);
                if (Math.Abs(scaled) < UnderflowGuard && b > 0)
                {
                    // ν ≫ z で Iˢ_ν がアンダーフローする場合は項がすべて正の級数（項数はおよそ x/b）で足す
                    return Hypergeometric0F1Series(b, x);
                }

                if (z > 1400)
                {
                    double log = Math.Log(Math.Abs(mantissa * scaled)) + exponent * Ln2 + z;
                    return Math.Sign(mantissa * scaled) * Math.Exp(log);
                }

                double halfExp = Math.Exp(0.5 * z);
                value = mantissa * scaled * halfExp * (1 + dz * Math.Sqrt(1 + nu / z * (nu / z)));
                value = ScaleB(value, exponent) * halfExp;
                if (!double.IsFinite(value) && b < 0)
                {
                    // b ≪ 0 では I_(b-1) がオーバーフローし、Γ(b) がアンダーフローしても値は有限なことがある。級数に戻す（打ち消しは級数の項の最大値で決まる）
                    return Hypergeometric0F1Series(b, x);
                }

                return value;
            }

            double j = BesselJ(nu, z);
            if (Math.Abs(j) < UnderflowGuard && nu > z + 1)
            {
                // ν ≫ z で J_ν がアンダーフローする場合は J_ν/J_μ（μ = ν - m ∈ [z, z+1)）を後退漸化式で仮数と指数に分けて求める（μ は J_μ がアンダーフローしない次数）
                double ratio = ScaledBesselJRatio(nu, z, out double mu, out long ratioExponent);
                // d ln J_ν/dz ≈ √(ν² - z²)/z（デバイの主要項。ν ≫ z なので十分）で z の丸めを補正する
                j = BesselJ(mu, z) * ratio * (1 + dz * Math.Sqrt((nu - z) * (nu + z)) / z);
                exponent += ratioExponent;
                dz = 0;
            }

            if (dz != 0)
            {
                // J_ν(z + dz) ≈ J_ν(z) + dz J'_ν(z)、J'_ν = (ν/z) J_ν - J_(ν+1)
                double derivative = nu / z * j - BesselJ(nu + 1, z);
                value = mantissa * (j + dz * derivative);
            }
            else
            {
                value = mantissa * j;
            }

            return ScaleB(value, exponent);
        }

        // これより小さいベッセル関数の値は非正規化数で精度が落ちているとみなす
        private const double UnderflowGuard = 1e-280;

        // J_ν(z)/J_μ(z) = ratio · 2^exponent を返す（ν > z + 1、μ = ν - m ≥ z は J_μ(z) がアンダーフローしない次数）。
        // J_ν/J_(ν-1) を連分数 z/(2ν - z²/(2(ν+1) - …))（DLMF 10.33）で求め、J_(k-1) = (2k/z) J_k - J_(k+1) を μ まで下げる（k > z では安定）
        private static double ScaledBesselJRatio(double nu, double z, out double mu, out long exponent)
        {
            // 連分数（修正 Lentz 法）
            double b0 = 2 * nu / z;
            double f = b0;
            double c = b0;
            double d = 0;
            for (int i = 1; i < MaxIterations; i++)
            {
                double bi = 2 * (nu + i) / z;
                d = bi - d;
                d = d == 0 ? Tiny : 1 / d;
                c = bi - 1 / c;
                if (c == 0)
                {
                    c = Tiny;
                }

                double delta = c * d;
                f *= delta;
                if (Math.Abs(delta - 1) <= Epsilon)
                {
                    break;
                }
            }

            // f = J_(ν-1)/J_ν。J_ν = 1、J_(ν-1) = f から J_μ がアンダーフローしなくなる次数まで下げる（丸めは 1 段ごとに相対 ε 程度たまる）。
            // ln J_ν(z) はデバイの主要項 -ν(α - tanh α) - ln(2πν tanh α)/2（cosh α = ν/z）で見積もる
            double tanh = Math.Sqrt((1 - z / nu) * (1 + z / nu));
            double logNu = -nu * (Math.Acosh(nu / z) - tanh) - 0.5 * Math.Log(2 * Math.PI * nu * tanh);
            double lowest = nu - Math.Floor(nu - z);
            mu = nu - 1;
            double upper = 1;
            double current = f;
            exponent = 0;
            while (mu > lowest && logNu + Math.Log(current) + exponent * Ln2 < -600)
            {
                double next = 2 * mu / z * current - upper;
                upper = current;
                current = next;
                mu--;
                int shift = Math.ILogB(current);
                if (shift > 500)
                {
                    upper = Math.ScaleB(upper, -shift);
                    current = Math.ScaleB(current, -shift);
                    exponent += shift;
                }
            }

            // ratio = J_ν/J_μ = 2^(-exponent) / current
            double inverse = 1 / current;
            int s = Math.ILogB(inverse);
            exponent = s - exponent;
            return Math.ScaleB(inverse, -s);
        }

        // Γ(b) = mantissa · 2^exponent の仮数（1 ≤ |mantissa| < 2、符号は Γ(b) と一致）を返す。
        // b > 170 ではスターリングの公式の各因子を仮数と指数に分けて掛け、b < -170 では反射公式で b > 170 に帰着させる
        private static double ScaledGammaSigned(double b, out long exponent)
        {
            double mantissa;
            if (b > 170)
            {
                // b^(b-1/2) = bⁿ b^f（b - 1/2 は丸められない）、e^(-b) = e^(-(b - 512c)) (e^(-512))ᶜ
                double h = b - 0.5;
                double n = Math.Floor(h);
                mantissa = ScaledPower(b, (long)n, out exponent) * Math.Pow(b, h - n) * Math.Sqrt(2 * Math.PI) * Math.Exp(StirlingCorrection(b));
                double chunks = Math.Floor(b / 512);
                mantissa *= Math.Exp(-(b - 512 * chunks));
                if (chunks > 0)
                {
                    mantissa *= ScaledPower(Math.Exp(-512), (long)chunks, out long chunkExponent);
                    exponent += chunkExponent;
                }

                int normalize = Math.ILogB(mantissa);
                exponent += normalize;
                return Math.ScaleB(mantissa, -normalize);
            }

            if (b < -170)
            {
                // 反射公式 Γ(b) = π/(sin(πb) Γ(1-b))。1 - b の丸め誤差 δ は Γ(1-b) に相対 ψ(1-b)δ の誤差を生むため補正する
                var (u, delta) = DoubleDoubleTwoSum(1, -b);
                double reflected = ScaledGammaSigned(u, out long reflectedExponent);
                if (delta != 0)
                {
                    reflected *= 1 + Digamma(u) * delta;
                }

                mantissa = Math.PI / (double.SinPi(b) * reflected);
                int shift = Math.ILogB(mantissa);
                exponent = shift - reflectedExponent;
                return Math.ScaleB(mantissa, -shift);
            }

            mantissa = Gamma(b);
            exponent = Math.ILogB(mantissa);
            return Math.ScaleB(mantissa, (int)-exponent);
        }

        // ₀F₁(; b; x) の級数評価
        private static double Hypergeometric0F1Series(double b, double x)
        {
            double sum = 1;
            double term = 1;
            for (int k = 0; k < MaxIterations; k++)
            {
                // 巨大な b では分母の積 (b + k)(k + 1) だけがオーバーフローして項が 0 になり、未計算の項で収束と判定してしまうので、順に割る
                double denominator = (b + k) * (k + 1);
                term *= double.IsInfinity(denominator) ? x / (b + k) / (k + 1) : x / denominator;
                sum += term;
                if (Math.Abs(term) <= 0.5 * Epsilon * Math.Abs(sum) && k + 1 > -b)
                {
                    break;
                }

                if (double.IsInfinity(sum))
                {
                    break;
                }
            }

            return sum;
        }
    }
}
