using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // ν ≤ 1e-17 では j'_ν,₁ = √(4ν(ν+1)/(ν+2)) が倍精度で正確 (この範囲では Y'ᵥ が巨大になり位相計算がオーバーフローしやすい)
        private const double BesselZeroTinyOrder = 1e-17;

        private enum BesselZeroKind
        {
            J,
            Y,
            JPrime,
            YPrime,
        }

        /// <summary>
        /// Computes the <paramref name="k"/>-th positive zero <c>j_ν,ₖ</c> of the Bessel function of the first kind <c>J_ν(x)</c>.
        /// <para>第 1 種ベッセル関数 <c>J_ν(x)</c> の第 <paramref name="k"/> 正零点 <c>j_ν,ₖ</c> を計算します。</para>
        /// </summary>
        /// <param name="nu">The non-negative order <c>ν</c> (<c>ν ≥ 0</c>). / 非負の次数 <c>ν</c>（<c>ν ≥ 0</c>）。</param>
        /// <param name="k">The 1-based zero index ordered by proximity to the origin (<c>k ≥ 1</c>). / 原点に近い順の 1 始まり零点番号（<c>k ≥ 1</c>）。</param>
        /// <returns>The positive zero <c>j_ν,ₖ</c> (for example, <c>j₀,₁ ≈ 2.40483</c>), or <see cref="double.NaN"/> if <paramref name="nu"/> is negative, non-finite, or <paramref name="k"/> is less than or equal to 0. / 正零点 <c>j_ν,ₖ</c>（例: <c>j₀,₁ ≈ 2.40483</c>）。<paramref name="nu"/> が負・非有限、または <paramref name="k"/> が 0 以下の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/10.21"/>
        public static double BesselJZero(double nu, int k) => BesselZero(nu, k, BesselZeroKind.J);

        /// <summary>
        /// Computes the <paramref name="k"/>-th positive zero <c>y_ν,ₖ</c> of the Bessel function of the second kind <c>Y_ν(x)</c>.
        /// <para>第 2 種ベッセル関数 <c>Y_ν(x)</c> の第 <paramref name="k"/> 正零点 <c>y_ν,ₖ</c> を計算します。</para>
        /// </summary>
        /// <param name="nu">The non-negative order <c>ν</c> (<c>ν ≥ 0</c>). / 非負の次数 <c>ν</c>（<c>ν ≥ 0</c>）。</param>
        /// <param name="k">The 1-based zero index ordered by proximity to the origin (<c>k ≥ 1</c>). / 原点に近い順の 1 始まり零点番号（<c>k ≥ 1</c>）。</param>
        /// <returns>The positive zero <c>y_ν,ₖ</c> (for example, <c>y₀,₁ ≈ 0.89358</c>), or <see cref="double.NaN"/> if <paramref name="nu"/> is negative, non-finite, or <paramref name="k"/> is less than or equal to 0. / 正零点 <c>y_ν,ₖ</c>（例: <c>y₀,₁ ≈ 0.89358</c>）。<paramref name="nu"/> が負・非有限、または <paramref name="k"/> が 0 以下の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/10.21"/>
        public static double BesselYZero(double nu, int k) => BesselZero(nu, k, BesselZeroKind.Y);

        /// <summary>
        /// Computes the <paramref name="k"/>-th zero <c>j'_ν,ₖ</c> of the derivative of the Bessel function of the first kind <c>J'_ν(x)</c>.
        /// <para>第 1 種ベッセル関数の導関数 <c>J'_ν(x)</c> の第 <paramref name="k"/> 零点 <c>j'_ν,ₖ</c> を計算します。</para>
        /// </summary>
        /// <param name="nu">The non-negative order <c>ν</c> (<c>ν ≥ 0</c>). / 非負の次数 <c>ν</c>（<c>ν ≥ 0</c>）。</param>
        /// <param name="k">The 1-based zero index ordered by proximity to the origin (<c>k ≥ 1</c>). / 原点に近い順の 1 始まり零点番号（<c>k ≥ 1</c>）。</param>
        /// <returns>The zero <c>j'_ν,ₖ</c>, or <see cref="double.NaN"/> if <paramref name="nu"/> is negative, non-finite, or <paramref name="k"/> is less than or equal to 0. / 零点 <c>j'_ν,ₖ</c>。<paramref name="nu"/> が負・非有限、または <paramref name="k"/> が 0 以下の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Following the DLMF convention, <c>j'₀,₁ = 0</c> and <c>j'₀,ₖ = j₁,ₖ₋₁</c> for <c>k ≥ 2</c>; for <c>ν &gt; 0</c>, <c>j'_ν,₁ ≥ ν</c>.</para>
        /// <para>DLMF の定義に従い、<c>j'₀,₁ = 0</c>、<c>k ≥ 2</c> で <c>j'₀,ₖ = j₁,ₖ₋₁</c> とします。<c>ν &gt; 0</c> では <c>j'_ν,₁ ≥ ν</c> です。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/10.21"/>
        public static double BesselJPrimeZero(double nu, int k) => BesselZero(nu, k, BesselZeroKind.JPrime);

        /// <summary>
        /// Computes the <paramref name="k"/>-th positive zero <c>y'_ν,ₖ</c> of the derivative of the Bessel function of the second kind <c>Y'_ν(x)</c>.
        /// <para>第 2 種ベッセル関数の導関数 <c>Y'_ν(x)</c> の第 <paramref name="k"/> 正零点 <c>y'_ν,ₖ</c> を計算します。</para>
        /// </summary>
        /// <param name="nu">The non-negative order <c>ν</c> (<c>ν ≥ 0</c>). / 非負の次数 <c>ν</c>（<c>ν ≥ 0</c>）。</param>
        /// <param name="k">The 1-based zero index ordered by proximity to the origin (<c>k ≥ 1</c>). / 原点に近い順の 1 始まり零点番号（<c>k ≥ 1</c>）。</param>
        /// <returns>The positive zero <c>y'_ν,ₖ</c> (for example, <c>y'₀,₁ ≈ 2.19714</c>), or <see cref="double.NaN"/> if <paramref name="nu"/> is negative, non-finite, or <paramref name="k"/> is less than or equal to 0. / 正零点 <c>y'_ν,ₖ</c>（例: <c>y'₀,₁ ≈ 2.19714</c>）。<paramref name="nu"/> が負・非有限、または <paramref name="k"/> が 0 以下の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/10.21"/>
        public static double BesselYPrimeZero(double nu, int k) => BesselZero(nu, k, BesselZeroKind.YPrime);

        private static double BesselZero(double nu, int k, BesselZeroKind kind)
        {
            if (!(nu >= 0) || double.IsPositiveInfinity(nu) || k <= 0)
            {
                return double.NaN;
            }

            bool derivative = kind is BesselZeroKind.JPrime or BesselZeroKind.YPrime;
            if (kind == BesselZeroKind.JPrime && nu == 0 && k == 1)
            {
                // J₀'(0) = 0 を第1零点に数える（DLMF 10.21(i) の慣習）
                return 0;
            }

            // 初期値は零点の下限（J、Y では 0、導関数では ν）より大きくしておく（位相の式は x > ν の導関数でしか単調でない）
            double lower = derivative ? nu : 0;
            if (kind == BesselZeroKind.JPrime && k == 1 && nu <= BesselZeroTinyOrder)
            {
                // x² = 4ν(ν+1)/(ν+2) の相対誤差は O(x²) = O(ν) なので、ν ≤ 10⁻¹⁷ ではそのまま倍精度の値になる
                // （このとき Y'ᵥ は 1/x² 程度に大きく、位相の計算がオーバーフローしやすい）
                return BesselZeroSmallOrderPrime(nu);
            }

            double x = BesselZeroInitialGuess(nu, k, kind);
            if (!(x > lower))
            {
                x = lower + Math.Max(1, lower) * 1e-3;
            }

            // 位相へのニュートン法。δ は零点までの位相差（atan の主値なので ±π/2 の範囲に収まる）、slope は位相の導関数
            double previousChange = double.PositiveInfinity;
            for (int iteration = 0; iteration < 100; iteration++)
            {
                BesselJYCore(nu, x, out double j, out double y);
                double delta;
                double slope;
                if (derivative)
                {
                    // J'ᵥ = (ν/x)Jᵥ - Jᵥ₊₁、Y'ᵥ = (ν/x)Yᵥ - Yᵥ₊₁
                    BesselJYCore(nu + 1, x, out double j1, out double y1);
                    double jPrime = nu / x * j - j1;
                    double yPrime = nu / x * y - y1;
                    // φ' = 2(x² - ν²)/(πx³N²)。小さな x では Y' が大きく N² や x³ がオーバーフローするので、N の大きい方でくくって組み立てる
                    double scale = Math.Max(Math.Abs(jPrime), Math.Abs(yPrime));
                    double jr = jPrime / scale;
                    double yr = yPrime / scale;
                    slope = 2 / (Math.PI * x) * ((x - nu) / x * ((x + nu) / x)) / scale / (scale * (jr * jr + yr * yr));
                    delta = kind == BesselZeroKind.JPrime ? Math.Atan(jPrime / yPrime) : -Math.Atan(yPrime / jPrime);
                }
                else
                {
                    double m2 = j * j + y * y;
                    slope = 2 / (Math.PI * x * m2);
                    delta = kind == BesselZeroKind.J ? Math.Atan(j / y) : -Math.Atan(y / j);
                }

                double step = delta / slope;
                double next = x + step;
                if (double.IsNaN(next))
                {
                    return double.NaN;
                }

                // 導関数の零点は x > ν にあり、x ≤ ν では位相が逆向きに動くので、ν との中点に戻す。
                // J、Y の零点でも x > 0 を保つ
                if (!(next > lower))
                {
                    next = 0.5 * (x + lower);
                }

                // 2次収束なので、刻みが ε 程度になるか、十分小さくなってからも半分以下に減らなくなったら
                // （J'ᵥ の打ち消しなどによる値の雑音に達したら）収束とみなす
                double change = Math.Abs(next - x);
                if (change <= 2 * MachineEpsilon * x || (change <= 1e-10 * x && change >= 0.5 * previousChange))
                {
                    return next;
                }

                previousChange = change;
                x = next;
            }

            return x;
        }

        // 零点の初期推定値。ν < 1 または β ≥ 5ν ではマクマホン展開、それ以外はエアリー零点による一様漸近展開を用いる
        private static double BesselZeroInitialGuess(double nu, int k, BesselZeroKind kind)
        {
            bool derivative = kind is BesselZeroKind.JPrime or BesselZeroKind.YPrime;
            if (kind == BesselZeroKind.JPrime && k == 1 && nu < 1)
            {
                // 原点級数の導関数を 2 項で打ち切った零点 x² = 4ν(ν+1)/(ν+2) (ν → 0 でマクマホン展開は使えない)
                return BesselZeroSmallOrderPrime(nu);
            }

            // β は J, Y' で k + ν/2 - 1/4、Y, J' で k + ν/2 - 3/4
            double shift = kind is BesselZeroKind.J or BesselZeroKind.YPrime ? 0.25 : 0.75;
            double beta = (k + 0.5 * nu - shift) * Math.PI;
            if (nu < 1 || beta >= 5 * nu)
            {
                return BesselZeroMcMahon(nu, beta, derivative);
            }

            // 一様漸近展開: ζ = ν^(-2/3) × (Ai, Bi, Ai', Bi' の零点)
            double airyZero = kind switch
            {
                BesselZeroKind.J => AiryZero(k, AiryZeroKind.Ai),
                BesselZeroKind.Y => AiryZero(k, AiryZeroKind.Bi),
                BesselZeroKind.JPrime => AiryZero(k, AiryZeroKind.AiPrime),
                _ => AiryZero(k, AiryZeroKind.BiPrime),
            };
            double cbrtNu = Math.Cbrt(nu);
            double zeta = airyZero / (cbrtNu * cbrtNu);
            double w = BesselZeroUniformW(zeta);
            double z = Math.Sqrt(1 + w * w);
            double x = nu * z;
            if (!derivative)
            {
                // f₁(ζ) = (1/2) z h² b₀(ζ), h² = √(4(-ζ)/(z² - 1)), b₀(ζ) = -5/(48ζ²) + (-ζ)^(-1/2)(5/(24w³) + 1/(8w))
                double minusZeta = -zeta;
                double h2 = Math.Sqrt(4 * minusZeta) / w;
                double b0 = -5 / (48 * zeta * zeta) + (5 / (24 * w * w * w) + 1 / (8 * w)) / Math.Sqrt(minusZeta);
                x += 0.5 * z * h2 * b0 / nu;
            }

            return x;
        }

        // マクマホン展開 (DLMF 10.21.E19, 10.21.E20; μ = 4ν², e = 8β)。
        // β が小さいと発散的なため、項の絶対値が増え始めたら手前で打ち切る
        private static double BesselZeroMcMahon(double nu, double beta, bool derivative)
        {
            double mu = 4 * nu * nu;
            double e = 8 * beta;
            double e2 = e * e;
            double e3 = e2 * e;
            double e5 = e3 * e2;
            double e7 = e5 * e2;
            Span<double> terms = stackalloc double[4];
            if (!derivative)
            {
                double m1 = mu - 1;
                terms[0] = m1 / e;
                terms[1] = 4 * m1 * (7 * mu - 31) / (3 * e3);
                terms[2] = 32 * m1 * ((83 * mu - 982) * mu + 3779) / (15 * e5);
                terms[3] = 64 * m1 * (((6949 * mu - 153855) * mu + 1585743) * mu - 6277237) / (105 * e7);
            }
            else
            {
                terms[0] = (mu + 3) / e;
                terms[1] = 4 * ((7 * mu + 82) * mu - 9) / (3 * e3);
                terms[2] = 32 * (((83 * mu + 2075) * mu - 3039) * mu + 3537) / (15 * e5);
                terms[3] = 64 * ((((6949 * mu + 296492) * mu - 1248002) * mu + 7414380) * mu - 5853627) / (105 * e7);
            }

            double x = beta;
            double previous = double.PositiveInfinity;
            foreach (double term in terms)
            {
                if (Math.Abs(term) >= previous)
                {
                    break;
                }

                x -= term;
                previous = Math.Abs(term);
            }

            return x;
        }

        // 一様漸近展開の z(ζ) (ζ < 0, z > 1) を w = √(z² - 1) として (2/3)(-ζ)^(3/2) = w - atan w (DLMF 10.20.E3) からニュートン法で求める。
        // 左辺 s は w → 0 で w³/3、w → ∞ で w - π/2 に漸近するためそれぞれの逆関数を初期値とする
        private static double BesselZeroUniformW(double zeta)
        {
            double s = 2.0 / 3 * Math.Pow(-zeta, 1.5);
            double w = s < 1 ? Math.Cbrt(3 * s) : s + 0.5 * Math.PI;
            for (int i = 0; i < 100; i++)
            {
                // g(w) = w - atan w - s, g'(w) = w²/(1 + w²)
                double w2 = w * w;
                double step = (w - Math.Atan(w) - s) * (1 + w2) / w2;
                w -= step;
                // 小さい w では w - atan w の桁落ちで ε/w² 程度の誤差が残るため、初期値として十分な精度で打ち切る
                if (Math.Abs(step) <= 1e-12 * w)
                {
                    break;
                }
            }

            return w;
        }

        // j'_ν,₁ ≈ √(4ν(ν+1)/(ν+2)) (0 ≤ ν < 1; 相対誤差 O(ν))
        private static double BesselZeroSmallOrderPrime(double nu) => Math.Sqrt(4 * nu * (nu + 1) / (nu + 2));
    }
}
