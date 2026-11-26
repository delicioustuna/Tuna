namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        /// <summary>
        /// Computes the Legendre function of the second kind (Ferrers function) <c>Qₙ(x)</c>.
        /// <para>第 2 種ルジャンドル関数（フェラーズ関数）<c>Qₙ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The non-negative integer degree (<c>n ≥ 0</c>). / 非負の整数次数（<c>n ≥ 0</c>）。</param>
        /// <param name="x">The argument on the cut (<c>-1 &lt; x &lt; 1</c>). / 引数（<c>-1 &lt; x &lt; 1</c>）。</param>
        /// <returns>The value of <c>Qₙ(x)</c>, or <see cref="double.NaN"/> if <paramref name="n"/> &lt; 0, <c>|x| ≥ 1</c>, or <paramref name="x"/> is <see cref="double.NaN"/>. / <c>Qₙ(x)</c> の値。<paramref name="n"/> &lt; 0、<c>|x| ≥ 1</c>、または <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Equivalent to <see cref="AssociatedLegendreQ(int, int, double)"/> with <c>m = 0</c> (for example, <c>Q₀(x) = artanh x</c>).</para>
        /// <para><see cref="AssociatedLegendreQ(int, int, double)"/> の <c>m = 0</c> の場合に相当します（例: <c>Q₀(x) = artanh x</c>）。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/14.5.E21"/>
        public static double LegendreQ(int n, double x) => AssociatedLegendreQ(n, 0, x);

        /// <summary>
        /// Computes the associated Legendre function of the second kind (Ferrers function) <c>Qₙᵐ(x)</c>.
        /// <para>第 2 種ルジャンドル陪関数（フェラーズ関数）<c>Qₙᵐ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The non-negative integer degree (<c>n ≥ 0</c>). / 非負の整数次数（<c>n ≥ 0</c>）。</param>
        /// <param name="m">The integer order (<c>n + m ≥ 0</c>; <c>m &gt; n</c> is allowed). / 整数位数（<c>n + m ≥ 0</c>。<c>m &gt; n</c> も指定可能です）。</param>
        /// <param name="x">The argument on the cut (<c>-1 &lt; x &lt; 1</c>). / 引数（<c>-1 &lt; x &lt; 1</c>）。</param>
        /// <returns>The value of <c>Qₙᵐ(x)</c>, or <see cref="double.NaN"/> if <paramref name="n"/> &lt; 0, <c>n + m &lt; 0</c>, <c>|x| ≥ 1</c>, or <paramref name="x"/> is <see cref="double.NaN"/>. / <c>Qₙᵐ(x)</c> の値。<paramref name="n"/> &lt; 0、<c>n + m &lt; 0</c>、<c>|x| ≥ 1</c>、または <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Uses the Ferrers function convention on <c>(-1, 1)</c> including the Condon–Shortley phase <c>(-1)ᵐ</c>, consistent with <see cref="AssociatedLegendreP(int, int, double)"/>.</para>
        /// <para>区間 <c>(-1, 1)</c> 上のフェラーズ関数の定義に従い、<see cref="AssociatedLegendreP(int, int, double)"/> と同じコンドン・ショートレー位相 <c>(-1)ᵐ</c> を含みます。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/14.6.E1"/>
        public static double AssociatedLegendreQ(int n, int m, double x)
        {
            if (n < 0 || double.IsNaN(x) || x <= -1 || x >= 1)
            {
                return double.NaN;
            }

            if (m >= 0)
            {
                double value = FerrersQInteger(n, m, x, out long exponent);
                return ScaleB(value, exponent);
            }

            if ((long)n + m < 0)
            {
                return double.NaN;
            }

            // Qₙ⁻ᵏ = (-1)ᵏ (n-k)!/(n+k)! Qₙᵏ。階乗比と Qₙᵏ はそれぞれオーバーフローしうるので仮数と指数に分けて掛ける
            int k = -m;
            double q = FerrersQInteger(n, k, x, out long qExponent);
            double ratio = ScaledFactorial(n - k, out long e1) / ScaledFactorial(n + k, out long e2);
            double result = ScaleB(q * ratio, qExponent + e1 - e2);
            return k % 2 == 0 ? result : -result;
        }

        // 整数次数・位数の Qₙᵐ(x) = mantissa · 2^exponent（n, m ≥ 0、|x| < 1）の仮数を返す
        private static double FerrersQInteger(int n, int m, double x, out long exponent)
        {
            exponent = 0;
            double q0 = Math.Atanh(x);
            double degreeZero;
            double previous = q0;
            double current = x * q0 - 1;
            if (n == 0)
            {
                degreeZero = q0;
            }
            else
            {
                // (k+1) Qₖ₊₁ = (2k+1) x Qₖ - k Qₖ₋₁（|x| < 1 では 2 つの解がともに振動し、前進方向に安定）
                for (int k = 1; k < n; k++)
                {
                    double next = ((2.0 * k + 1) * x * current - k * previous) / (k + 1.0);
                    previous = current;
                    current = next;
                }

                degreeZero = current;
            }

            if (m == 0)
            {
                return degreeZero;
            }

            double oneMinusX2 = (1 - x) * (1 + x);
            double root = Math.Sqrt(oneMinusX2);

            // Q₀¹ = -(1-x²)^(-1/2)、Q₁¹ = -((1-x²)^(1/2) artanh x + x (1-x²)^(-1/2))、k Qₖ₊₁¹ = (2k+1) x Qₖ¹ - (k+1) Qₖ₋₁¹
            previous = -1 / root;
            current = -(root * q0 + x / root);
            double degreeOne;
            if (n == 0)
            {
                degreeOne = previous;
            }
            else
            {
                for (int k = 1; k < n; k++)
                {
                    double next = ((2.0 * k + 1) * x * current - (k + 1.0) * previous) / k;
                    previous = current;
                    current = next;
                }

                degreeOne = current;
            }

            if (m == 1)
            {
                return degreeOne;
            }

            // 位数についての漸化式 Qʲ⁺² = -2(j+1) x/√(1-x²) Qʲ⁺¹ - (n-j)(n+j+1) Qʲ（m > n では Pₙᵐ = 0 なので Q が優勢解）
            double ratio = x / root;
            previous = degreeZero;
            current = degreeOne;
            for (int j = 0; j <= m - 2; j++)
            {
                double next = -2.0 * (j + 1) * ratio * current - ((double)n - j) * ((double)n + j + 1) * previous;
                previous = current;
                current = next;
                RenormalizePair(ref current, ref previous, ref exponent);
            }

            return current;
        }

        /// <summary>
        /// Computes the Legendre function of the first kind (Ferrers function) <c>P_ν(x)</c> for real degree <c>ν</c>.
        /// <para>実数次数 <c>ν</c> の第 1 種ルジャンドル関数（フェラーズ関数）<c>P_ν(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="nu">The real degree <c>ν</c>. / 実数の次数 <c>ν</c>。</param>
        /// <param name="x">The argument (<c>-1 &lt; x ≤ 1</c>; <c>x = -1</c> is supported for integer degrees). / 引数（<c>-1 &lt; x ≤ 1</c>。<c>x = -1</c> は整数次数のみ）。</param>
        /// <returns>The value of <c>P_ν(x)</c>, equivalent to <see cref="AssociatedLegendreP(double, double, double)"/> with <c>μ = 0</c>. / <c>P_ν(x)</c> の値（<see cref="AssociatedLegendreP(double, double, double)"/> の <c>μ = 0</c>）。</returns>
        /// <seealso href="https://dlmf.nist.gov/14.3.E1"/>
        public static double LegendreP(double nu, double x) => AssociatedLegendreP(nu, 0, x);

        /// <summary>
        /// Computes the Legendre function of the second kind (Ferrers function) <c>Q_ν(x)</c> for real degree <c>ν</c>.
        /// <para>実数次数 <c>ν</c> の第 2 種ルジャンドル関数（フェラーズ関数）<c>Q_ν(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="nu">The real degree <c>ν</c> (excluding negative integers). / 実数の次数 <c>ν</c>（負の整数を除く）。</param>
        /// <param name="x">The argument (<c>-1 &lt; x &lt; 1</c>). / 引数（<c>-1 &lt; x &lt; 1</c>）。</param>
        /// <returns>The value of <c>Q_ν(x)</c>, equivalent to <see cref="AssociatedLegendreQ(double, double, double)"/> with <c>μ = 0</c>. / <c>Q_ν(x)</c> の値（<see cref="AssociatedLegendreQ(double, double, double)"/> の <c>μ = 0</c>）。</returns>
        /// <seealso href="https://dlmf.nist.gov/14.3.E2"/>
        public static double LegendreQ(double nu, double x) => AssociatedLegendreQ(nu, 0, x);

        /// <summary>
        /// Computes the associated Legendre function of the first kind (Ferrers function) <c>P_ν^μ(x)</c> for real degree <c>ν</c> and order <c>μ</c>.
        /// <para>実数次数 <c>ν</c> および実数位数 <c>μ</c> の第 1 種ルジャンドル陪関数（フェラーズ関数）<c>P_ν^μ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="nu">The real degree <c>ν</c>. / 実数の次数 <c>ν</c>。</param>
        /// <param name="mu">The real order <c>μ</c>. / 実数の位数 <c>μ</c>。</param>
        /// <param name="x">The argument (<c>-1 &lt; x ≤ 1</c>; <c>x = -1</c> is supported only for integer <c>ν ≥ 0</c> with <c>|μ| ≤ ν</c>). / 引数（<c>-1 &lt; x ≤ 1</c>。<c>x = -1</c> は整数 <c>ν ≥ 0</c> かつ <c>|μ| ≤ ν</c> の場合のみ）。</param>
        /// <returns>The value of <c>P_ν^μ(x)</c>, or <see cref="double.NaN"/> if arguments are out of domain or non-finite. / <c>P_ν^μ(x)</c> の値。引数が定義域外または有限でない場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Uses the Ferrers function definition on the cut <c>(-1, 1)</c> (DLMF 14.3.1), matching the Condon–Shortley phase convention of <see cref="AssociatedLegendreP(int, int, double)"/> at integer orders. At <c>x = 1</c>, returns <c>1</c> for <c>μ = 0</c>, <c>0</c> for <c>μ &lt; 0</c> or positive integers, and signed infinity otherwise.</para>
        /// <para>区間 <c>(-1, 1)</c> 上のフェラーズ関数の定義（DLMF 14.3.1）に従い、整数位数では <see cref="AssociatedLegendreP(int, int, double)"/> と同じコンドン・ショートレー位相に一致します。<c>x = 1</c> では <c>μ = 0</c> で <c>1</c>、<c>μ &lt; 0</c> または正の整数で <c>0</c>、それ以外では符号付き無限大を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/14.3.E1"/>
        public static double AssociatedLegendreP(double nu, double mu, double x)
        {
            if (double.IsNaN(nu) || double.IsNaN(mu) || double.IsNaN(x) || double.IsInfinity(nu) || double.IsInfinity(mu) || x < -1 || x > 1)
            {
                return double.NaN;
            }

            if (nu < -0.5)
            {
                // P_(-ν-1)^μ = P_ν^μ（DLMF 14.9.5）
                nu = -nu - 1;
            }

            bool integerOrder = IsInteger(mu);
            if (IsInteger(nu) && integerOrder && nu <= int.MaxValue && Math.Abs(mu) <= nu)
            {
                return AssociatedLegendreP((int)nu, (int)mu, x);
            }

            if (x == 1)
            {
                if (mu == 0)
                {
                    return 1;
                }

                // ((1+x)/(1-x))^(μ/2)/Γ(1-μ) の振る舞い
                return mu < 0 || integerOrder ? 0 : SignGamma(1 - mu) * double.PositiveInfinity;
            }

            if (x == -1)
            {
                return double.NaN;
            }

            if (integerOrder && mu > 0)
            {
                // P_ν^m = (-1)ᵐ Γ(ν+m+1)/Γ(ν-m+1) P_ν^(-m)（1-μ が極になるのを避ける）
                double negative = FerrersPRecurrence(nu, -mu, x);
                double factor = GammaRatio(nu + mu + 1, nu - mu + 1);
                double value = factor == 0 ? 0 : factor * negative;
                return IsOddInteger(mu) ? -value : value;
            }

            return FerrersPRecurrence(nu, mu, x);
        }

        /// <summary>
        /// Computes the associated Legendre function of the second kind (Ferrers function) <c>Q_ν^μ(x)</c> for real degree <c>ν</c> and order <c>μ</c>.
        /// <para>実数次数 <c>ν</c> および実数位数 <c>μ</c> の第 2 種ルジャンドル陪関数（フェラーズ関数）<c>Q_ν^μ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="nu">The real degree <c>ν</c>. / 実数の次数 <c>ν</c>。</param>
        /// <param name="mu">The real order <c>μ</c> (evaluated by limit at integer orders). / 実数の位数 <c>μ</c>（整数位数では極限値）。</param>
        /// <param name="x">The argument (<c>-1 &lt; x &lt; 1</c>). / 引数（<c>-1 &lt; x &lt; 1</c>）。</param>
        /// <returns>The value of <c>Q_ν^μ(x)</c>, or <see cref="double.NaN"/> if <c>ν + μ</c> is a negative integer, <c>|x| ≥ 1</c>, or any parameter is non-finite. / <c>Q_ν^μ(x)</c> の値。<c>ν + μ</c> が負の整数、<c>|x| ≥ 1</c>、または引数が有限でない場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Uses the Ferrers function definition on the cut <c>(-1, 1)</c> (DLMF 14.3.2), including the Condon–Shortley phase.</para>
        /// <para>区間 <c>(-1, 1)</c> 上のフェラーズ関数の定義（DLMF 14.3.2）に従い、コンドン・ショートレー位相を含みます。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/14.3.E2"/>
        public static double AssociatedLegendreQ(double nu, double mu, double x)
        {
            if (double.IsNaN(nu) || double.IsNaN(mu) || double.IsNaN(x) || double.IsInfinity(nu) || double.IsInfinity(mu) || x <= -1 || x >= 1)
            {
                return double.NaN;
            }

            double sum = nu + mu;
            if (IsInteger(sum) && sum < 0)
            {
                return double.NaN;
            }

            if (IsInteger(nu) && IsInteger(mu) && nu >= 0 && nu <= int.MaxValue && Math.Abs(mu) <= int.MaxValue)
            {
                return AssociatedLegendreQ((int)nu, (int)mu, x);
            }

            // 出発点 t₀ = max(-1/2, |μ| - 1)。ν ≥ -1/2 は ν について、ν < -1/2 は t = -ν-1 について前進させる
            bool reflected = nu < -0.5;
            double target = reflected ? -nu - 1 : nu;
            double lowest = Math.Max(Math.Max(-0.5, Math.Abs(mu) - 1), FerrersTurningDegree(mu, x));
            if (target <= lowest + 1)
            {
                return FerrersQBase(nu, mu, x);
            }

            double steps = Math.Ceiling(target - lowest) - 1;
            double start = target - steps;
            double first = FerrersQBase(reflected ? -start - 1 : start, mu, x);
            double second = FerrersQBase(reflected ? -start - 2 : start + 1, mu, x);
            return FerrersRecurrence(start, mu, x, first, second, (long)steps - 1);
        }

        // P_ν^μ(x)（ν ≥ -1/2、μ は正の整数でない、|x| < 1）。ν₀ の直上の次数から前進漸化式で ν まで上げる
        private static double FerrersPRecurrence(double nu, double mu, double x)
        {
            double lowest = Math.Max(Math.Max(-0.5, mu - 1), FerrersTurningDegree(mu, x));
            if (nu <= lowest + 1)
            {
                return FerrersPBase(nu, mu, x);
            }

            double steps = Math.Ceiling(nu - lowest) - 1;
            double start = nu - steps;
            return FerrersRecurrence(start, mu, x, FerrersPBase(start, mu, x), FerrersPBase(start + 1, mu, x), (long)steps - 1);
        }

        // 次数についての漸化式が振動域に入る次数の目安 |μ|/√(1-x²) - 1/2（(ν+1/2) sin θ = |μ|、x = cos θ）。
        // これより低い非振動域では前進漸化式が不安定になりうるため、出発値はこれより高い次数でとる
        private static double FerrersTurningDegree(double mu, double x) =>
            Math.Abs(mu) / Math.Sqrt((1 - x) * (1 + x)) - 0.5;

        // 次数漸化式 (k-μ+1) yₖ₊₁ = (2k+1) x yₖ - (k+μ) yₖ₋₁ を k = start + 1 から count 回前進させる（first, second は次数 start, start + 1 の値）
        private static double FerrersRecurrence(double start, double mu, double x, double first, double second, long count)
        {
            double previous = first;
            double current = second;
            long exponent = 0;
            double k = start + 1;
            for (long i = 0; i < count; i++, k++)
            {
                double next = ((2 * k + 1) * x * current - (k + mu) * previous) / (k - mu + 1);
                previous = current;
                current = next;
                RenormalizePair(ref current, ref previous, ref exponent);
            }

            return ScaleB(current, exponent);
        }

        // P_ν^μ(x) = ((1+x)/(1-x))^(μ/2) F(ν+1, -ν; 1-μ; (1-x)/2)/Γ(1-μ)（μ は正の整数でない）。1 - (1-x)/2 = (1+x)/2 を丸めずに渡す
        private static double FerrersPBase(double nu, double mu, double x)
        {
            double z = 0.5 * (1 - x);
            double w = 0.5 * (1 + x);
            double f = Hypergeometric2F1Core(nu + 1, -nu, 1 - mu, z, w);
            double scale = mu == 0 ? 1 : Math.Pow((1 + x) / (1 - x), 0.5 * mu);
            double reciprocal = ReciprocalGamma(1 - mu);
            return reciprocal == 0 || f == 0 ? 0 : f * scale * reciprocal;
        }

        // x² の超幾何関数による Q_ν^μ(x)（|x| < 1、ν + μ は負の整数でない）
        private static double FerrersQBase(double nu, double mu, double x)
        {
            double x2 = x * x;
            double w = (1 - x) * (1 + x);
            double s = 0.5 * (nu + mu + 1);
            double t = 0.5 * (nu + mu) + 1;

            // cos(πs) Γ(s)/Γ(·) は分母が極なら0（GammaRatio が0を返す）、cos(πs) = 0 なら第1項は0
            double cosS = double.CosPi(s);
            double even = cosS == 0 ? 0 : cosS * GammaRatio(s, 0.5 * (nu - mu) + 1);
            if (even != 0)
            {
                even *= Hypergeometric2F1Core(-0.5 * (nu + mu), 0.5 * (nu - mu + 1), 0.5, x2, w);
            }

            double cosT = double.CosPi(t);
            double odd = cosT == 0 || x == 0 ? 0 : cosT * GammaRatio(t, 0.5 * (nu - mu + 1));
            if (odd != 0)
            {
                odd *= 2 * x * Hypergeometric2F1Core(0.5 * (1 - nu - mu), 0.5 * (nu - mu) + 1, 1.5, x2, w);
            }

            double prefactor = Math.Pow(2, mu - 1) * SqrtPi * (mu == 0 ? 1 : Math.Pow(w, -0.5 * mu));
            return prefactor * (even - odd);
        }
    }
}
