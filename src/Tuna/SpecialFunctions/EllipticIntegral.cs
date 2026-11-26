namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // カールソン倍加公式の収束判定相対偏差 (打ち切り誤差は概ねこの 6 乗)
        private const double CarlsonTolerance = 0.0008;

        private const double PiHi = 3.141592653589793;
        private const double PiLo = 1.2246467991473532e-16;

        /// <summary>
        /// Computes Carlson's symmetric elliptic integral of the first kind, <c>R_F(x, y, z) = (1 / 2) ∫₀^∞ dt / √((t + x)(t + y)(t + z))</c>.
        /// <para>カールソンの第 1 種対称楕円積分 <c>R_F(x, y, z) = (1 / 2) ∫₀^∞ dt / √((t + x)(t + y)(t + z))</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The first non-negative argument (<c>x ≥ 0</c>). / 非負の第 1 引数（<c>x ≥ 0</c>）。</param>
        /// <param name="y">The second non-negative argument (<c>y ≥ 0</c>). / 非負の第 2 引数（<c>y ≥ 0</c>）。</param>
        /// <param name="z">The third non-negative argument (<c>z ≥ 0</c>). / 非負の第 3 引数（<c>z ≥ 0</c>）。</param>
        /// <returns>The value of <c>R_F(x, y, z)</c>, <see cref="double.PositiveInfinity"/> if at least two arguments are zero, or <see cref="double.NaN"/> if any argument is negative or <see cref="double.NaN"/>. / <c>R_F(x, y, z)</c> の値。2 つ以上の引数が 0 の場合は <see cref="double.PositiveInfinity"/>、負または <see cref="double.NaN"/> の引数がある場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/19.16.E1"/>
        public static double CarlsonRF(double x, double y, double z)
        {
            if (double.IsNaN(x) || double.IsNaN(y) || double.IsNaN(z) || x < 0 || y < 0 || z < 0)
            {
                return double.NaN;
            }

            if (x + y == 0 || y + z == 0 || x + z == 0)
            {
                return double.PositiveInfinity;
            }

            if (double.IsInfinity(x) || double.IsInfinity(y) || double.IsInfinity(z))
            {
                return 0;
            }

            // 倍加公式の xt + λ は最大引数の 4 倍以下なので、オーバーフロー直前のみ縮小する
            int k = CarlsonScaleExponent(Math.Max(x, Math.Max(y, z)), 1020);
            if (CarlsonScaleLosesPrecision(MinPositive(x, y, z), k))
            {
                // R_F(x, y, z) = R_F((x+λ)/4, (y+λ)/4, (z+λ)/4)
                double quarterLambda = CarlsonQuarterLambda(Math.Sqrt(x), Math.Sqrt(y), Math.Sqrt(z));
                return CarlsonRF(0.25 * x + quarterLambda, 0.25 * y + quarterLambda, 0.25 * z + quarterLambda);
            }

            if (k != 0)
            {
                // R_F(4ᵏx, 4ᵏy, 4ᵏz) = 2⁻ᵏ R_F(x, y, z)
                return Math.ScaleB(CarlsonRF(Math.ScaleB(x, 2 * k), Math.ScaleB(y, 2 * k), Math.ScaleB(z, 2 * k)), k);
            }

            double xt = x;
            double yt = y;
            double zt = z;
            double average;
            double dx;
            double dy;
            double dz;
            do
            {
                double sx = Math.Sqrt(xt);
                double sy = Math.Sqrt(yt);
                double sz = Math.Sqrt(zt);
                double lambda = sx * (sy + sz) + sy * sz;
                xt = 0.25 * (xt + lambda);
                yt = 0.25 * (yt + lambda);
                zt = 0.25 * (zt + lambda);
                average = (xt + yt + zt) / 3;
                dx = (average - xt) / average;
                dy = (average - yt) / average;
                dz = (average - zt) / average;
            }
            while (Math.Max(Math.Abs(dx), Math.Max(Math.Abs(dy), Math.Abs(dz))) > CarlsonTolerance);

            double e2 = dx * dy - dz * dz;
            double e3 = dx * dy * dz;
            return (1 + (e2 / 24 - 0.1 - 3.0 / 44 * e3) * e2 + e3 / 14) / Math.Sqrt(average);
        }

        /// <summary>
        /// Computes Carlson's symmetric elliptic integral of the second kind, <c>R_D(x, y, z) = (3 / 2) ∫₀^∞ dt / ((t + z) √((t + x)(t + y)(t + z)))</c>.
        /// <para>カールソンの第 2 種対称楕円積分 <c>R_D(x, y, z) = (3 / 2) ∫₀^∞ dt / ((t + z) √((t + x)(t + y)(t + z)))</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The first non-negative argument (<c>x ≥ 0</c>). / 非負の第 1 引数（<c>x ≥ 0</c>）。</param>
        /// <param name="y">The second non-negative argument (<c>y ≥ 0</c>, <c>x + y &gt; 0</c>). / 非負の第 2 引数（<c>y ≥ 0</c>、<c>x + y &gt; 0</c>）。</param>
        /// <param name="z">The third positive argument (<c>z &gt; 0</c>). / 正の第 3 引数（<c>z &gt; 0</c>）。</param>
        /// <returns>The value of <c>R_D(x, y, z)</c>, <see cref="double.PositiveInfinity"/> if <c>z = 0</c> or <c>x + y = 0</c>, or <see cref="double.NaN"/> if any argument is negative or <see cref="double.NaN"/>. / <c>R_D(x, y, z)</c> の値。<c>z = 0</c> または <c>x + y = 0</c> では <see cref="double.PositiveInfinity"/>、負または <see cref="double.NaN"/> の引数がある場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/19.16.E5"/>
        public static double CarlsonRD(double x, double y, double z)
        {
            if (double.IsNaN(x) || double.IsNaN(y) || double.IsNaN(z) || x < 0 || y < 0 || z < 0)
            {
                return double.NaN;
            }

            if (z == 0 || x + y == 0)
            {
                return double.PositiveInfinity;
            }

            if (double.IsInfinity(x) || double.IsInfinity(y) || double.IsInfinity(z))
            {
                return 0;
            }

            // 倍加公式の √z (z + λ) と結果の分母は引数の 3/2 次なので 2⁶⁸² 未満に収める
            int k = CarlsonScaleExponent(Math.Max(x, Math.Max(y, z)), 600);
            if (CarlsonScaleLosesPrecision(MinPositive(x, y, z), k))
            {
                // R_D(x, y, z) = 3/(√z (z+λ)) + R_D((x+λ)/4, (y+λ)/4, (z+λ)/4)/4
                double sz = Math.Sqrt(z);
                double quarterLambda = CarlsonQuarterLambda(Math.Sqrt(x), Math.Sqrt(y), sz);
                double zq = 0.25 * z + quarterLambda;
                return 0.75 / sz / zq + 0.25 * CarlsonRD(0.25 * x + quarterLambda, 0.25 * y + quarterLambda, zq);
            }

            if (k != 0)
            {
                // R_D(4ᵏx, 4ᵏy, 4ᵏz) = 2⁻³ᵏ R_D(x, y, z)
                return Math.ScaleB(CarlsonRD(Math.ScaleB(x, 2 * k), Math.ScaleB(y, 2 * k), Math.ScaleB(z, 2 * k)), 3 * k);
            }

            double xt = x;
            double yt = y;
            double zt = z;
            double sum = 0;
            double factor = 1;
            double average;
            double dx;
            double dy;
            double dz;
            do
            {
                double sx = Math.Sqrt(xt);
                double sy = Math.Sqrt(yt);
                double sz = Math.Sqrt(zt);
                double lambda = sx * (sy + sz) + sy * sz;
                sum += factor / (sz * (zt + lambda));
                factor *= 0.25;
                xt = 0.25 * (xt + lambda);
                yt = 0.25 * (yt + lambda);
                zt = 0.25 * (zt + lambda);
                average = 0.2 * (xt + yt + 3 * zt);
                dx = (average - xt) / average;
                dy = (average - yt) / average;
                dz = (average - zt) / average;
            }
            while (Math.Max(Math.Abs(dx), Math.Max(Math.Abs(dy), Math.Abs(dz))) > CarlsonTolerance);

            const double c1 = 3.0 / 14;
            const double c2 = 1.0 / 6;
            const double c3 = 9.0 / 22;
            const double c4 = 3.0 / 26;
            const double c5 = 0.25 * c3;
            const double c6 = 1.5 * c4;
            double ea = dx * dy;
            double eb = dz * dz;
            double ec = ea - eb;
            double ed = ea - 6 * eb;
            double ee = ed + ec + ec;
            return 3 * sum + factor * (1 + ed * (-c1 + c5 * ed - c6 * dz * ee)
                + dz * (c2 * ee + dz * (-c3 * ec + dz * c4 * ea))) / (average * Math.Sqrt(average));
        }

        /// <summary>
        /// Computes Carlson's degenerate elliptic integral <c>R_C(x, y) = R_F(x, y, y)</c>.
        /// <para>カールソンの退化楕円積分 <c>R_C(x, y) = R_F(x, y, y)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The first non-negative argument (<c>x ≥ 0</c>). / 非負の第 1 引数（<c>x ≥ 0</c>）。</param>
        /// <param name="y">The second non-zero argument (Cauchy principal value for <c>y &lt; 0</c>). / 0 以外の第 2 引数（<c>y &lt; 0</c> ではコーシーの主値）。</param>
        /// <returns>The value of <c>R_C(x, y)</c>, <see cref="double.PositiveInfinity"/> at <c>y = 0</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is negative or either argument is <see cref="double.NaN"/>. / <c>R_C(x, y)</c> の値。<c>y = 0</c> では <see cref="double.PositiveInfinity"/>、<paramref name="x"/> が負または引数が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/19.16.E6"/>
        public static double CarlsonRC(double x, double y)
        {
            if (double.IsNaN(x) || double.IsNaN(y) || x < 0)
            {
                return double.NaN;
            }

            if (y == 0)
            {
                return double.PositiveInfinity;
            }

            if (double.IsInfinity(x) || double.IsInfinity(y))
            {
                return 0;
            }

            if (y < 0)
            {
                // 主値: R_C(x, y) = √(x/(x-y)) R_C(x-y, -y)。x - y がオーバーフローする場合は 1/4 に縮小する
                double shifted = x - y;
                if (double.IsPositiveInfinity(shifted))
                {
                    return 0.5 * CarlsonRC(0.25 * x, 0.25 * y);
                }

                return Math.Sqrt(x) / Math.Sqrt(shifted) * CarlsonRC(shifted, -y);
            }

            // 倍加公式の xt + λ は最大引数の 4 倍以下なので、オーバーフロー直前のみ縮小する
            int k = CarlsonScaleExponent(Math.Max(x, y), 1020);
            if (CarlsonScaleLosesPrecision(MinPositive(x, y), k))
            {
                // R_C(x, y) = R_C((x+λ)/4, (y+λ)/4) (λ = 2√x√y + y)
                double quarterLambda = 0.5 * Math.Sqrt(x) * Math.Sqrt(y) + 0.25 * y;
                return CarlsonRC(0.25 * x + quarterLambda, 0.25 * y + quarterLambda);
            }

            if (k != 0)
            {
                // R_C(4ᵏx, 4ᵏy) = 2⁻ᵏ R_C(x, y)
                return Math.ScaleB(CarlsonRC(Math.ScaleB(x, 2 * k), Math.ScaleB(y, 2 * k)), k);
            }

            double xt = x;
            double yt = y;
            double average;
            double s;
            do
            {
                double lambda = 2 * Math.Sqrt(xt) * Math.Sqrt(yt) + yt;
                xt = 0.25 * (xt + lambda);
                yt = 0.25 * (yt + lambda);
                average = (xt + yt + yt) / 3;
                s = (yt - average) / average;
            }
            while (Math.Abs(s) > CarlsonTolerance);

            return (1 + s * s * (0.3 + s * (1.0 / 7 + s * (0.375 + s * 9.0 / 22)))) / Math.Sqrt(average);
        }

        /// <summary>
        /// Computes Carlson's symmetric elliptic integral of the third kind, <c>R_J(x, y, z, p) = (3 / 2) ∫₀^∞ dt / ((t + p) √((t + x)(t + y)(t + z)))</c>.
        /// <para>カールソンの第 3 種対称楕円積分 <c>R_J(x, y, z, p) = (3 / 2) ∫₀^∞ dt / ((t + p) √((t + x)(t + y)(t + z)))</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The first non-negative argument (<c>x ≥ 0</c>). / 非負の第 1 引数（<c>x ≥ 0</c>）。</param>
        /// <param name="y">The second non-negative argument (<c>y ≥ 0</c>). / 非負の第 2 引数（<c>y ≥ 0</c>）。</param>
        /// <param name="z">The third non-negative argument (<c>z ≥ 0</c>). / 非負の第 3 引数（<c>z ≥ 0</c>）。</param>
        /// <param name="p">The fourth non-zero argument (Cauchy principal value for <c>p &lt; 0</c>). / 0 以外の第 4 引数（<c>p &lt; 0</c> ではコーシーの主値）。</param>
        /// <returns>The value of <c>R_J(x, y, z, p)</c>, or <see cref="double.NaN"/> if any argument is out of domain. / <c>R_J(x, y, z, p)</c> の値。定義域外では <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/19.16.E2"/>
        public static double CarlsonRJ(double x, double y, double z, double p)
        {
            if (double.IsNaN(x) || double.IsNaN(y) || double.IsNaN(z) || double.IsNaN(p) || x < 0 || y < 0 || z < 0)
            {
                return double.NaN;
            }

            if (p == 0 || x + y == 0 || y + z == 0 || x + z == 0)
            {
                return double.PositiveInfinity;
            }

            if (double.IsInfinity(x) || double.IsInfinity(y) || double.IsInfinity(z) || double.IsInfinity(p))
            {
                return 0;
            }

            // 結果の分母は引数の 3/2 次なので R_D と同じ範囲に正規化する (α, β は CarlsonRJTerm で比として扱う)
            int k = CarlsonScaleExponent(Math.Max(Math.Max(x, y), Math.Max(z, Math.Abs(p))), 600);
            bool losesPrecision = CarlsonScaleLosesPrecision(MinPositive(x, y, z, Math.Abs(p)), k);
            if (k != 0 && !losesPrecision)
            {
                // R_J(4ᵏx, 4ᵏy, 4ᵏz, 4ᵏp) = 2⁻³ᵏ R_J(x, y, z, p)
                return Math.ScaleB(CarlsonRJ(Math.ScaleB(x, 2 * k), Math.ScaleB(y, 2 * k), Math.ScaleB(z, 2 * k),
                    Math.ScaleB(p, 2 * k)), 3 * k);
            }

            if (p < 0)
            {
                return CarlsonRJPrincipalValue(x, y, z, p);
            }

            if (losesPrecision)
            {
                return CarlsonRJDuplicated(x, y, z, p);
            }

            double xt = x;
            double yt = y;
            double zt = z;
            double pt = p;
            double sum = 0;
            double factor = 1;
            double average;
            double dx;
            double dy;
            double dz;
            double dp;
            do
            {
                double sx = Math.Sqrt(xt);
                double sy = Math.Sqrt(yt);
                double sz = Math.Sqrt(zt);
                double sp = Math.Sqrt(pt);
                double quarterLambda = CarlsonQuarterLambda(sx, sy, sz);
                xt = 0.25 * xt + quarterLambda;
                yt = 0.25 * yt + quarterLambda;
                zt = 0.25 * zt + quarterLambda;
                pt = 0.25 * pt + quarterLambda;
                sum += factor * CarlsonRJTerm(sx, sy, sz, sp, pt);
                factor *= 0.25;
                average = 0.2 * (xt + yt + zt + pt + pt);
                dx = (average - xt) / average;
                dy = (average - yt) / average;
                dz = (average - zt) / average;
                dp = (average - pt) / average;
            }
            while (Math.Max(Math.Max(Math.Abs(dx), Math.Abs(dy)), Math.Max(Math.Abs(dz), Math.Abs(dp))) > CarlsonTolerance);

            const double c1 = 3.0 / 14;
            const double c2 = 1.0 / 3;
            const double c3 = 3.0 / 22;
            const double c4 = 3.0 / 26;
            const double c5 = 0.75 * c3;
            const double c6 = 1.5 * c4;
            const double c7 = 0.5 * c2;
            const double c8 = c3 + c3;
            double ea = dx * (dy + dz) + dy * dz;
            double eb = dx * dy * dz;
            double ec = dp * dp;
            double ed = ea - 3 * ec;
            double ee = eb + 2 * dp * (ea - ec);
            return 3 * sum + factor * (1 + ed * (-c1 + c5 * ed - c6 * ee) + eb * (c7 + dp * (-c8 + dp * c4))
                + dp * ea * (c2 - dp * c3) - c2 * dp * ec) / (average * Math.Sqrt(average));
        }

        // p < 0 での R_J(x, y, z, p) の主値を正の p の積分へ変換して計算する
        private static double CarlsonRJPrincipalValue(double x, double y, double z, double p)
        {
            // x ≤ y ≤ z に並べ替え、(y - p) R_J(x, y, z, p) = (q - y) R_J(x, y, z, q) - 3 R_F(x, y, z) + 3 R_C(xz/y, pq/y)
            // (q = y + (z - y)(y - x)/(y - p))。各積はオーバーフローしない順に計算する
            double xt = Math.Min(Math.Min(x, y), z);
            double zt = Math.Max(Math.Max(x, y), z);
            double yt = Math.Max(Math.Min(x, y), Math.Min(Math.Max(x, y), z));
            double shifted = yt - p;
            double a = double.IsPositiveInfinity(shifted) ? 0.5 / (0.5 * yt - 0.5 * p) : 1 / shifted;
            double b = (zt - yt) * ((yt - xt) * a);
            double pt = yt + b;
            double rho = xt / yt * zt;
            double tau = p * (pt / yt);
            return a * (b * CarlsonRJ(xt, yt, zt, pt) + 3 * (CarlsonRC(rho, tau) - CarlsonRF(xt, yt, zt)));
        }

        // 倍加公式の 1 段適用: R_J(x, y, z, p) = 3 R_C(α, β) + R_J((x+λ)/4, (y+λ)/4, (z+λ)/4, (p+λ)/4)/4 (p > 0)
        private static double CarlsonRJDuplicated(double x, double y, double z, double p)
        {
            double sx = Math.Sqrt(x);
            double sy = Math.Sqrt(y);
            double sz = Math.Sqrt(z);
            double quarterLambda = CarlsonQuarterLambda(sx, sy, sz);
            double pq = 0.25 * p + quarterLambda;
            return 3 * CarlsonRJTerm(sx, sy, sz, Math.Sqrt(p), pq)
                + 0.25 * CarlsonRJ(0.25 * x + quarterLambda, 0.25 * y + quarterLambda, 0.25 * z + quarterLambda, pq);
        }

        // CarlsonRJ の倍加公式の 1 段項 R_C(α, β) (α = (p(√x+√y+√z) + √x√y√z)², β = p(p+λ)², pq = (p+λ)/4)
        private static double CarlsonRJTerm(double sx, double sy, double sz, double sp, double pq)
        {
            // α = A², β = B² (B = 4√p pq) は引数の 3 次で単独ではオーバーフロー・アンダーフローするため、
            // R_C(α, β) = R_C(r², 1)/B (r = A/B) として比 r と 1/B に分離する
            double sMax = Math.Max(Math.Max(sx, sy), sz);
            double sMin = Math.Min(Math.Min(sx, sy), sz);
            double sMid = Math.Max(Math.Min(sx, sy), Math.Min(Math.Max(sx, sy), sz));
            double r = sp / pq * (0.25 * (sx + sy + sz)) + sMax * sMid / pq * (0.25 * sMin / sp);
            return 0.25 * CarlsonRCSquare(r) / sp / pq;
        }

        // R_C(r², 1) (r ≥ 0; r² がオーバーフローする範囲では漸近形を使用)
        private static double CarlsonRCSquare(double r)
        {
            if (r <= 1e150)
            {
                return CarlsonRC(r * r, 1);
            }

            // R_C(r², 1) = arcosh(r)/√(r² - 1) = (ln(2r) + O(r⁻²))/r
            return double.IsPositiveInfinity(r) ? 0 : (Math.Log(r) + Ln2) / r;
        }

        /// <summary>
        /// Computes Carlson's completely symmetric elliptic integral of the second kind, <c>R_G(x, y, z)</c>.
        /// <para>カールソンの第 2 種完全対称楕円積分 <c>R_G(x, y, z)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The first non-negative argument (<c>x ≥ 0</c>). / 非負の第 1 引数（<c>x ≥ 0</c>）。</param>
        /// <param name="y">The second non-negative argument (<c>y ≥ 0</c>). / 非負の第 2 引数（<c>y ≥ 0</c>）。</param>
        /// <param name="z">The third non-negative argument (<c>z ≥ 0</c>). / 非負の第 3 引数（<c>z ≥ 0</c>）。</param>
        /// <returns>The value of <c>R_G(x, y, z)</c>, or <see cref="double.NaN"/> if any argument is negative or <see cref="double.NaN"/>. / <c>R_G(x, y, z)</c> の値。負または <see cref="double.NaN"/> の引数がある場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/19.16.E2_5"/>
        public static double CarlsonRG(double x, double y, double z)
        {
            if (double.IsNaN(x) || double.IsNaN(y) || double.IsNaN(z) || x < 0 || y < 0 || z < 0)
            {
                return double.NaN;
            }

            // 対称関数なので x ≤ z ≤ y に並べ替える
            double lo = Math.Min(Math.Min(x, y), z);
            double hi = Math.Max(Math.Max(x, y), z);
            double mid = Math.Max(Math.Min(x, y), Math.Min(Math.Max(x, y), z));
            if (hi == 0)
            {
                return 0;
            }

            if (double.IsInfinity(hi))
            {
                return double.PositiveInfinity;
            }

            if (mid == 0)
            {
                // R_G(0, 0, z) = √z / 2
                return 0.5 * Math.Sqrt(hi);
            }

            // (z-x)(y-z) は引数の2次なので、オーバーフロー・アンダーフローしない範囲に正規化する
            int k = CarlsonScaleExponent(hi, 500);
            if (k != 0)
            {
                // R_G(4ᵏx, 4ᵏy, 4ᵏz) = 2ᵏ R_G(x, y, z)
                return Math.ScaleB(CarlsonRG(Math.ScaleB(lo, 2 * k), Math.ScaleB(hi, 2 * k), Math.ScaleB(mid, 2 * k)), -k);
            }

            // 2 R_G = z R_F - (x-z)(y-z) R_D / 3 + √(xy/z)。z を中央の値にすると (x-z)(y-z) ≤ 0 となり、
            // 3項がすべて非負になって桁落ちしない（DLMF 19.21.10）
            return 0.5 * (mid * CarlsonRF(lo, hi, mid) + (mid - lo) * (hi - mid) * CarlsonRD(lo, hi, mid) / 3
                + Math.Sqrt(lo / mid * hi));
        }

        /// <summary>
        /// Computes the complete elliptic integral of the first kind, <c>K(m) = ∫₀^(π/2) dθ / √(1 - m sin²θ)</c>.
        /// <para>第 1 種完全楕円積分 <c>K(m) = ∫₀^(π/2) dθ / √(1 - m sin²θ)</c>（パラメータ <c>m = k²</c>）を計算します。</para>
        /// </summary>
        /// <param name="m">The parameter <c>m = k²</c> (<c>m ≤ 1</c>). / パラメータ <c>m = k²</c>（<c>m ≤ 1</c>）。</param>
        /// <returns>The value of <c>K(m)</c>, <see cref="double.PositiveInfinity"/> at <c>m = 1</c>, or <see cref="double.NaN"/> if <paramref name="m"/> is greater than 1 or <see cref="double.NaN"/>. / <c>K(m)</c> の値。<c>m = 1</c> では <see cref="double.PositiveInfinity"/>、<paramref name="m"/> が 1 より大きい場合または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://fungrim.org/entry/0455b3/"/>
        public static double EllipticK(double m)
        {
            if (double.IsNaN(m) || m > 1)
            {
                return double.NaN;
            }

            if (m == 1)
            {
                return double.PositiveInfinity;
            }

            if (double.IsNegativeInfinity(m))
            {
                return 0;
            }

            // 完全積分では K(m) = π/(2 AGM(1, √(1-m))) (DLMF 19.8.5)。R_F の倍加公式より高速に 2 次収束する。
            // 初期値の最大は √double.MaxValue、以後の平均はその半分程度以下のため a * b はオーバーフローしない
            double a = 1;
            double b = Math.Sqrt(1 - m);
            for (int iteration = 0; iteration < 32; iteration++)
            {
                double nextA = 0.5 * a + 0.5 * b;
                double nextB = Math.Sqrt(a * b);
                a = nextA;
                b = nextB;
                if (Math.Abs(a - b) <= Epsilon * a)
                {
                    break;
                }
            }

            return (Math.PI / 2) / a;
        }

        /// <summary>
        /// Computes the complete elliptic integral of the second kind, <c>E(m) = ∫₀^(π/2) √(1 - m sin²θ) dθ</c>.
        /// <para>第 2 種完全楕円積分 <c>E(m) = ∫₀^(π/2) √(1 - m sin²θ) dθ</c>（パラメータ <c>m = k²</c>）を計算します。</para>
        /// </summary>
        /// <param name="m">The parameter <c>m = k²</c> (<c>m ≤ 1</c>). / パラメータ <c>m = k²</c>（<c>m ≤ 1</c>）。</param>
        /// <returns>The value of <c>E(m)</c>, or <see cref="double.NaN"/> if <paramref name="m"/> is greater than 1 or <see cref="double.NaN"/>. / <c>E(m)</c> の値。<paramref name="m"/> が 1 より大きい場合または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://fungrim.org/entry/190843/"/>
        public static double EllipticE(double m)
        {
            if (double.IsNaN(m) || m > 1)
            {
                return double.NaN;
            }

            if (m == 1)
            {
                return 1;
            }

            if (double.IsNegativeInfinity(m))
            {
                return double.PositiveInfinity;
            }

            double y = 1 - m;
            return m < 0 ? CarlsonRF(0, y, 1) - m / 3 * CarlsonRD(0, y, 1) : CompleteEllipticE(m, y);
        }

        // 0 ≤ m < 1 での E(m) = 2 R_G(0, 1-m, 1) = y R_F(0, 1, y) + (ym/3) R_D(0, 1, y) (DLMF 19.21.10; 両項非負で R_F/R_D の倍加反復を共有)
        private static double CompleteEllipticE(double m, double y)
        {
            // 引数は (0, 1, y), 2⁻⁵³ ≤ y ≤ 1 のためスケーリング不要
            double xt = 0;
            double yt = 1;
            double zt = y;
            double sum = 0;
            double factor = 1;
            double averageF;
            double averageD;
            double fx;
            double fy;
            double fz;
            double dx;
            double dy;
            double dz;
            do
            {
                double sx = Math.Sqrt(xt);
                double sy = Math.Sqrt(yt);
                double sz = Math.Sqrt(zt);
                double lambda = sx * (sy + sz) + sy * sz;
                sum += factor / (sz * (zt + lambda));
                factor *= 0.25;
                xt = 0.25 * (xt + lambda);
                yt = 0.25 * (yt + lambda);
                zt = 0.25 * (zt + lambda);
                averageF = (xt + yt + zt) / 3;
                averageD = 0.2 * (xt + yt + 3 * zt);
                fx = (averageF - xt) / averageF;
                fy = (averageF - yt) / averageF;
                fz = (averageF - zt) / averageF;
                dx = (averageD - xt) / averageD;
                dy = (averageD - yt) / averageD;
                dz = (averageD - zt) / averageD;
            }
            while (Math.Max(Math.Abs(fx), Math.Max(Math.Abs(fy), Math.Abs(fz))) > CarlsonTolerance
                || Math.Max(Math.Abs(dx), Math.Max(Math.Abs(dy), Math.Abs(dz))) > CarlsonTolerance);

            // CarlsonRF / CarlsonRD と同一の補正多項式・許容差で評価する
            double e2 = fx * fy - fz * fz;
            double e3 = fx * fy * fz;
            double rf = (1 + (e2 / 24 - 0.1 - 3.0 / 44 * e3) * e2 + e3 / 14) / Math.Sqrt(averageF);
            const double c1 = 3.0 / 14;
            const double c2 = 1.0 / 6;
            const double c3 = 9.0 / 22;
            const double c4 = 3.0 / 26;
            const double c5 = 0.25 * c3;
            const double c6 = 1.5 * c4;
            double ea = dx * dy;
            double eb = dz * dz;
            double ec = ea - eb;
            double ed = ea - 6 * eb;
            double ee = ed + ec + ec;
            double rd = 3 * sum + factor * (1 + ed * (-c1 + c5 * ed - c6 * dz * ee)
                + dz * (c2 * ee + dz * (-c3 * ec + dz * c4 * ea))) / (averageD * Math.Sqrt(averageD));
            return y * rf + y * (m / 3) * rd;
        }

        /// <summary>
        /// Computes the complete elliptic integral of the third kind, <c>Π(n|m) = ∫₀^(π/2) dθ / ((1 - n sin²θ) √(1 - m sin²θ))</c>.
        /// <para>第 3 種完全楕円積分 <c>Π(n|m) = ∫₀^(π/2) dθ / ((1 - n sin²θ) √(1 - m sin²θ))</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The elliptic characteristic (<c>n ≠ 1</c>; Cauchy principal value for <c>n &gt; 1</c>). / 特性（<c>n ≠ 1</c>。<c>n &gt; 1</c> ではコーシーの主値）。</param>
        /// <param name="m">The parameter <c>m = k²</c> (<c>m &lt; 1</c>). / パラメータ <c>m = k²</c>（<c>m &lt; 1</c>）。</param>
        /// <returns>The value of <c>Π(n|m)</c>, or <see cref="double.NaN"/> if out of domain. / <c>Π(n|m)</c> の値。定義域外では <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://fungrim.org/entry/83a535/"/>
        public static double EllipticPi(double n, double m)
        {
            if (double.IsNaN(n) || double.IsNaN(m) || m > 1)
            {
                return double.NaN;
            }

            if (n == 1 || m == 1)
            {
                // n ≤ 1 なら被積分関数は正のまま θ = π/2 で発散する
                return n <= 1 ? double.PositiveInfinity : double.NaN;
            }

            double y = 1 - m;
            if (n < -1 && n < m)
            {
                return EllipticPiLargeNegative(n, y);
            }

            return CarlsonRF(0, y, 1) + n / 3 * CarlsonRJ(0, y, 1, 1 - n);
        }

        // n < -1 かつ n < m での Π(n|m) (y = 1 - m)。
        // R_F + (n/3) R_J は n → -∞ で桁落ちするため、R_J の変換式で正項主体の表現へ書き換える
        private static double EllipticPiLargeNegative(double n, double y)
        {
            double b = Math.Min(y, 1);
            double c = Math.Max(y, 1);
            double p = 1 - n;
            double d = p - b;

            // q = b (p - c)/d > 0, R_C(0, pq/b) = (π/2)√(d/(p(p - c)))
            double q = b * ((p - c) / d);
            double transformed = -(c - b) * b / d / 3 * CarlsonRJ(0, b, c, q)
                + Math.PI / 2 * Math.Sqrt(d / p) / Math.Sqrt(p - c);

            // Π = R_F (1 - b)/d + (-n/d) × transformed (m ≥ 0 では 1 - b = m)
            return (1 - b) / d * CarlsonRF(0, b, c) + -n / d * transformed;
        }

        /// <summary>
        /// Computes the incomplete elliptic integral of the first kind, <c>F(φ|m) = ∫₀^φ dθ / √(1 - m sin²θ)</c>.
        /// <para>第 1 種不完全楕円積分 <c>F(φ|m) = ∫₀^φ dθ / √(1 - m sin²θ)</c> を計算します。</para>
        /// </summary>
        /// <param name="phi">The amplitude <c>φ</c> in radians. / 振幅 <c>φ</c>（ラジアン）。</param>
        /// <param name="m">The parameter <c>m = k²</c> (<c>m sin²φ ≤ 1</c>). / パラメータ <c>m = k²</c>（<c>m sin²φ ≤ 1</c>）。</param>
        /// <returns>The real value of <c>F(φ|m)</c>, or <see cref="double.NaN"/> if the result is complex or either argument is invalid. / <c>F(φ|m)</c> の実数値。値が複素数になる場合や引数が無効な場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://fungrim.org/entry/81fb10/"/>
        public static double EllipticF(double phi, double m)
        {
            if (!TryReduceAmplitude(phi, m, out double periods, out double s, out double c2, out double delta))
            {
                return double.NaN;
            }

            double value = s * CarlsonRF(c2, delta, 1);
            return periods == 0 ? value : value + 2 * periods * EllipticK(m);
        }

        /// <summary>
        /// Computes the incomplete elliptic integral of the second kind, <c>E(φ|m) = ∫₀^φ √(1 - m sin²θ) dθ</c>.
        /// <para>第 2 種不完全楕円積分 <c>E(φ|m) = ∫₀^φ √(1 - m sin²θ) dθ</c> を計算します。</para>
        /// </summary>
        /// <param name="phi">The amplitude <c>φ</c> in radians. / 振幅 <c>φ</c>（ラジアン）。</param>
        /// <param name="m">The parameter <c>m = k²</c> (<c>m sin²φ ≤ 1</c>). / パラメータ <c>m = k²</c>（<c>m sin²φ ≤ 1</c>）。</param>
        /// <returns>The real value of <c>E(φ|m)</c>, or <see cref="double.NaN"/> if the result is complex or either argument is invalid. / <c>E(φ|m)</c> の実数値。値が複素数になる場合や引数が無効な場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://fungrim.org/entry/2ff7e7/"/>
        public static double EllipticE(double phi, double m)
        {
            if (!TryReduceAmplitude(phi, m, out double periods, out double s, out double c2, out double delta))
            {
                return double.NaN;
            }

            double value = s * CarlsonRF(c2, delta, 1) - m / 3 * s * s * s * CarlsonRD(c2, delta, 1);
            return periods == 0 ? value : value + 2 * periods * EllipticE(m);
        }

        /// <summary>
        /// Computes the incomplete elliptic integral of the third kind, <c>Π(n; φ|m) = ∫₀^φ dθ / ((1 - n sin²θ) √(1 - m sin²θ))</c>.
        /// <para>第 3 種不完全楕円積分 <c>Π(n; φ|m) = ∫₀^φ dθ / ((1 - n sin²θ) √(1 - m sin²θ))</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The elliptic characteristic (Cauchy principal value for <c>n sin²φ &gt; 1</c>). / 特性（<c>n sin²φ &gt; 1</c> ではコーシーの主値）。</param>
        /// <param name="phi">The amplitude <c>φ</c> in radians. / 振幅 <c>φ</c>（ラジアン）。</param>
        /// <param name="m">The parameter <c>m = k²</c> (<c>m sin²φ ≤ 1</c>). / パラメータ <c>m = k²</c>（<c>m sin²φ ≤ 1</c>）。</param>
        /// <returns>The real value of <c>Π(n; φ|m)</c>, or <see cref="double.NaN"/> if the result is complex or any argument is invalid. / <c>Π(n; φ|m)</c> の実数値。値が複素数になる場合や引数が無効な場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://fungrim.org/entry/60f858/"/>
        public static double EllipticPi(double n, double phi, double m)
        {
            if (double.IsNaN(n) || !TryReduceAmplitude(phi, m, out double periods, out double s, out double c2, out double delta))
            {
                return double.NaN;
            }

            double s2 = s * s;
            double p = 1 - n * s2;
            if (p == 0)
            {
                return Math.Sign(s) * double.PositiveInfinity;
            }

            double value;
            double ratio = m / n * s2;
            if (n < 0 && ratio <= 0.5)
            {
                value = EllipticPiNegativeCharacteristic(n, m, s, c2, delta, ratio);
            }
            else
            {
                value = s * CarlsonRF(c2, delta, 1) + n / 3 * s * s2 * CarlsonRJ(c2, delta, 1, p);
            }

            return periods == 0 ? value : value + 2 * periods * EllipticPi(n, m);
        }

        // n < 0 での Π(n; φ|m) (ratio = (m/n) sin²φ ≤ 1/2; DLMF 19.21.E12)。
        // n → -∞ での O(√(-n)) 桁落ちを避けるため R_F を消去し Π = s R_C(c²Δ, pq) - (m s³/(3n)) R_J(c², Δ, 1, q) とする
        private static double EllipticPiNegativeCharacteristic(double n, double m, double s, double c2, double delta, double ratio)
        {
            double q = 1 - ratio;
            double s2 = s * s;
            double rc;
            if (n < -1)
            {
                // R_C は -1/2 次同次。pq のオーバーフロー回避のため -n で割った引数で計算する (p/(-n) = s² + 1/(-n))
                double scale = -n;
                rc = CarlsonRC(c2 * delta / scale, (s2 + 1 / scale) * q) / Math.Sqrt(scale);
            }
            else
            {
                rc = CarlsonRC(c2 * delta, (1 - n * s2) * q);
            }

            double value = s * rc;
            return m == 0 ? value : value - m / (3 * n) * s * s2 * CarlsonRJ(c2, delta, 1, q);
        }

        // カールソン積分の引数を 4ᵏ 倍して最大引数の 2 進指数を [-500, limit] に収める k (範囲内なら 0)。
        // 4 の冪のため平方根を含めて丸め誤差なく変換でき、同次性から 2 の冪で復元できる
        private static int CarlsonScaleExponent(double max, int limit)
        {
            int e = Math.ILogB(max);
            if (e > limit)
            {
                return -((e - limit + 1) / 2);
            }

            return e < -500 ? -e / 2 : 0;
        }

        // 4ᵏ 倍 (k < 0) で正の最小引数が非正規化数になり下位ビットを失うかどうか (true の場合は先に倍加公式を 1 段適用する)
        private static bool CarlsonScaleLosesPrecision(double minPositive, int k) =>
            k < 0 && Math.ILogB(minPositive) + 2 * k < -1022;

        private static double MinPositive(double a, double b, double c = double.PositiveInfinity, double d = double.PositiveInfinity)
        {
            static double PositiveOrInfinity(double v) => v > 0 ? v : double.PositiveInfinity;
            return Math.Min(Math.Min(PositiveOrInfinity(a), PositiveOrInfinity(b)), Math.Min(PositiveOrInfinity(c), PositiveOrInfinity(d)));
        }

        // 倍加公式の λ/4 = (√x√y + √x√z + √y√z)/4 (最大引数付近でもオーバーフローしない順序で計算)
        private static double CarlsonQuarterLambda(double sx, double sy, double sz) =>
            sx * (0.25 * (sy + sz)) + 0.25 * sy * sz;

        // 振幅 φ を φ = kπ + φ' (|φ'| ≤ π/2) に分解し sin φ', cos²φ', 1 - m sin²φ' を返す
        private static bool TryReduceAmplitude(double phi, double m, out double periods, out double s, out double c2, out double delta)
        {
            periods = 0;
            s = c2 = delta = double.NaN;
            if (double.IsNaN(phi) || double.IsNaN(m) || double.IsInfinity(phi) || double.IsInfinity(m))
            {
                return false;
            }

            periods = Math.Round(phi / Math.PI);
            double reduced = periods == 0 ? phi : (phi - periods * PiHi) - periods * PiLo;
            if (periods != 0 && m > 1)
            {
                return false;
            }

            s = Math.Sin(reduced);
            double c = Math.Cos(reduced);
            c2 = c * c;
            delta = 1 - m * s * s;
            return delta >= 0;
        }
    }
}
