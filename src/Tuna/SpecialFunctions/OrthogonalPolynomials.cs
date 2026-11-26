using System.Numerics;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        /// <summary>
        /// Computes the Legendre polynomial <c>Pₙ(x)</c>.
        /// <para>ルジャンドル多項式 <c>Pₙ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The non-negative degree (<c>n ≥ 0</c>). / 非負の次数（<c>n ≥ 0</c>）。</param>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>Pₙ(x)</c>, or <see cref="double.NaN"/> if <paramref name="n"/> &lt; 0 or <paramref name="x"/> is <see cref="double.NaN"/>. / <c>Pₙ(x)</c> の値。<paramref name="n"/> &lt; 0 または <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Legendre_polynomials"/>
        public static double LegendreP(int n, double x)
        {
            if (n < 0 || double.IsNaN(x))
            {
                return double.NaN;
            }

            if (n == 0)
            {
                return 1;
            }

            // (k+1) Pₖ₊₁ = (2k+1) x Pₖ - k Pₖ₋₁
            double previous = 1;
            double current = x;
            for (int k = 1; k < n; k++)
            {
                double next = ((2 * k + 1) * x * current - k * previous) / (k + 1);
                previous = current;
                current = next;
            }

            return current;
        }

        /// <summary>
        /// Computes the associated Legendre function <c>Pₙᵐ(x)</c>.
        /// <para>ルジャンドル陪関数 <c>Pₙᵐ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The non-negative degree (<c>n ≥ 0</c>). / 非負の次数（<c>n ≥ 0</c>）。</param>
        /// <param name="m">The integer order (negative orders are supported; returns <c>0</c> when <c>|m| &gt; n</c>). / 整数位数（負の位数にも対応。<c>|m| &gt; n</c> の場合は <c>0</c>）。</param>
        /// <param name="x">The input value in <c>[-1, 1]</c>. / 範囲 <c>[-1, 1]</c> の入力値。</param>
        /// <returns>The value of <c>Pₙᵐ(x)</c>, <c>0</c> if <c>|m| &gt; n</c>, or <see cref="double.NaN"/> if <paramref name="n"/> &lt; 0, <paramref name="x"/> is outside <c>[-1, 1]</c>, or <paramref name="x"/> is <see cref="double.NaN"/>. / <c>Pₙᵐ(x)</c> の値。<c>|m| &gt; n</c> の場合は <c>0</c>、<paramref name="n"/> &lt; 0、<paramref name="x"/> が <c>[-1, 1]</c> 外、または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Includes the Condon–Shortley phase factor <c>(-1)ᵐ</c>.</para>
        /// <para>コンドン・ショートレーの位相因子 <c>(-1)ᵐ</c> を含みます。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/14.7.E8"/>
        public static double AssociatedLegendreP(int n, int m, double x)
        {
            if (n < 0 || double.IsNaN(x) || x < -1 || x > 1)
            {
                return double.NaN;
            }

            // Math.Abs(int.MinValue) はオーバーフローするので、符号付きのまま範囲を判定する
            if (m > n || m < -n)
            {
                return 0;
            }

            if (m < 0)
            {
                return AssociatedLegendrePNegativeOrder(n, -m, x);
            }

            // Pₘᵐ = (-1)ᵐ (2m-1)!! (1-x²)^(m/2)
            double pmm = 1;
            if (m > 0)
            {
                double sqrtOneMinusX2 = Math.Sqrt((1 - x) * (1 + x));
                double factor = 1;
                for (int i = 1; i <= m; i++)
                {
                    pmm *= -factor * sqrtOneMinusX2;
                    factor += 2;
                }
            }

            if (n == m)
            {
                return pmm;
            }

            double pmmp1 = x * (2 * m + 1) * pmm;
            if (n == m + 1)
            {
                return pmmp1;
            }

            // (l-m) Pₗᵐ = (2l-1) x Pₗ₋₁ᵐ - (l+m-1) Pₗ₋₂ᵐ
            double pll = 0;
            for (int l = m + 2; l <= n; l++)
            {
                pll = (x * (2 * l - 1) * pmmp1 - (l + m - 1) * pmm) / (l - m);
                pmm = pmmp1;
                pmmp1 = pll;
            }

            return pll;
        }

        // 負の位数 Pₙ⁻ᵐ(x)（0 < m ≤ n, -1 ≤ x ≤ 1）を階乗比アンダーフローを避けて漸化式（DLMF 14.10.3）で直接評価する
        private static double AssociatedLegendrePNegativeOrder(int n, int m, double x)
        {
            double oneMinusX2 = (1 - x) * (1 + x);
            if (oneMinusX2 == 0)
            {
                return 0;
            }

            double previous = ScaledPower(Math.Sqrt(oneMinusX2), m, out long powerExponent)
                / ScaledFactorial(m, out long factorialExponent);
            long exponent = powerExponent - factorialExponent - m;
            if (n == m)
            {
                return ScaleB(previous, exponent);
            }

            double current = x * previous;
            for (long l = m + 2L; l <= n; l++)
            {
                double next = ((2.0 * l - 1) * x * current - (l - m - 1.0) * previous) / ((double)l + m);
                previous = current;
                current = next;
                RenormalizePair(ref current, ref previous, ref exponent);
            }

            return ScaleB(current, exponent);
        }

        // 線形漸化式の隣接 2 項を同じ 2 の冪でスケーリングして共通指数に加える
        private static void RenormalizePair(ref double current, ref double previous, ref long exponent)
        {
            if (current == 0 || !double.IsFinite(current))
            {
                return;
            }

            int shift = Math.ILogB(current);
            if (Math.Abs(shift) > 500)
            {
                current = Math.ScaleB(current, -shift);
                previous = Math.ScaleB(previous, -shift);
                exponent += shift;
            }
        }

        /// <summary>
        /// Computes the spherical harmonic <c>Yₗᵐ(θ, φ)</c>.
        /// <para>球面調和関数 <c>Yₗᵐ(θ, φ)</c> を計算します。</para>
        /// </summary>
        /// <param name="l">The non-negative degree (<c>l ≥ 0</c>). / 非負の次数（<c>l ≥ 0</c>）。</param>
        /// <param name="m">The integer order (returns <see cref="Complex.Zero"/> when <c>|m| &gt; l</c>). / 整数位数（<c>|m| &gt; l</c> の場合は <see cref="Complex.Zero"/>）。</param>
        /// <param name="theta">The polar (colatitude) angle <c>θ</c> in radians. / 天頂角 <c>θ</c>（ラジアン）。</param>
        /// <param name="phi">The azimuthal angle <c>φ</c> in radians. / 方位角 <c>φ</c>（ラジアン）。</param>
        /// <returns>The value of <c>Yₗᵐ(θ, φ)</c>, <see cref="Complex.Zero"/> if <c>|m| &gt; l</c>, or a <see cref="Complex"/> with <see cref="double.NaN"/> components if <paramref name="l"/> &lt; 0 or either angle is <see cref="double.NaN"/>. / <c>Yₗᵐ(θ, φ)</c> の値。<c>|m| &gt; l</c> の場合は <see cref="Complex.Zero"/>、<paramref name="l"/> &lt; 0 またはいずれかの角度が <see cref="double.NaN"/> の場合は成分が <see cref="double.NaN"/> の <see cref="Complex"/>。</returns>
        /// <remarks>
        /// <para>Normalized over the unit sphere and includes the Condon–Shortley phase factor <c>(-1)ᵐ</c>.</para>
        /// <para>単位球面上で正規化されており、コンドン・ショートレーの位相因子 <c>(-1)ᵐ</c> を含みます。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/14.30.E1"/>
        public static Complex SphericalHarmonicY(int l, int m, double theta, double phi)
        {
            if (l < 0 || double.IsNaN(theta) || double.IsNaN(phi))
            {
                return new Complex(double.NaN, double.NaN);
            }

            // Math.Abs(int.MinValue) はオーバーフローするので、符号付きのまま範囲を判定する
            if (m > l || m < -l)
            {
                return Complex.Zero;
            }

            int order = Math.Abs(m);
            double x = Math.Cos(theta);
            double sinTheta = Math.Abs(Math.Sin(theta));

            // 正規化したルジャンドル陪関数を、オーバーフローしない漸化式で計算する
            // P̄ₘᵐ = (-1)ᵐ √((2m+1)/(4π) (2m-1)!!/(2m)!!) sinᵐθ。
            // sin²ᵐθ を作ってから平方根を取ると先にアンダーフローするので、正規化係数と sinᵐθ を分け、
            // sinᵐθ と漸化式の値は仮数と共通の 2 進指数に分けて保持する
            double norm = 1;
            for (int i = 1; i <= order; i++)
            {
                norm *= (2.0 * i - 1) / (2 * i);
            }

            long exponent = 0;
            double sinPower = order == 0 ? 1 : sinTheta == 0 ? 0 : ScaledPower(sinTheta, order, out exponent);
            double pmm = Math.Sqrt((2 * order + 1) * norm / (4 * Math.PI)) * sinPower;
            if (order % 2 == 1)
            {
                pmm = -pmm;
            }

            double value;
            if (l == order)
            {
                value = pmm;
            }
            else
            {
                double pmmp1 = x * Math.Sqrt(2 * order + 3) * pmm;
                if (l == order + 1)
                {
                    value = pmmp1;
                }
                else
                {
                    double oldFactor = Math.Sqrt(2 * order + 3);
                    double pll = 0;
                    for (int ll = order + 2; ll <= l; ll++)
                    {
                        double f = Math.Sqrt((4.0 * ll * ll - 1) / ((double)ll * ll - (double)order * order));
                        pll = (x * pmmp1 - pmm / oldFactor) * f;
                        oldFactor = f;
                        pmm = pmmp1;
                        pmmp1 = pll;
                        RenormalizePair(ref pmmp1, ref pmm, ref exponent);
                    }

                    value = pmmp1;
                }
            }

            value = ScaleB(value, exponent);

            // Yₗ⁻ᵐ = (-1)ᵐ conj(Yₗᵐ)
            double angle = order * phi;
            var result = new Complex(value * Math.Cos(angle), value * Math.Sin(angle));
            if (m < 0)
            {
                result = Complex.Conjugate(result);
                if (order % 2 == 1)
                {
                    result = -result;
                }
            }

            return result;
        }

        /// <summary>
        /// Computes the physicists' Hermite polynomial <c>Hₙ(x)</c>.
        /// <para>エルミート多項式（物理学者の流儀）<c>Hₙ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The non-negative degree (<c>n ≥ 0</c>). / 非負の次数（<c>n ≥ 0</c>）。</param>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>Hₙ(x)</c>, or <see cref="double.NaN"/> if <paramref name="n"/> &lt; 0 or <paramref name="x"/> is <see cref="double.NaN"/>. / <c>Hₙ(x)</c> の値。<paramref name="n"/> &lt; 0 または <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Hermite_polynomials#Definition"/>
        public static double HermiteH(int n, double x)
        {
            if (n < 0 || double.IsNaN(x))
            {
                return double.NaN;
            }

            if (n == 0)
            {
                return 1;
            }

            if (x == 0)
            {
                // 原点では奇数次は厳密に 0、偶数次は H₂ₘ(0) = -2(2m-1) H₂ₘ₋₂(0)。漸化式のままだと、
                // 偶数次がオーバーフローした後の 0·∞ が NaN になって奇数次の 0 と次の偶数次の符号付き無限大を失う
                return n % 2 == 1 ? OddOriginValue(n, x) : EvenOriginValue(n, 2);
            }

            // Hₖ₊₁ = 2x Hₖ - 2k Hₖ₋₁
            double previous = 1;
            double current = 2 * x;
            for (int k = 1; k < n; k++)
            {
                double next = 2 * x * current - 2.0 * k * previous;
                previous = current;
                current = next;
            }

            return current;
        }

        /// <summary>
        /// Computes the probabilists' Hermite polynomial <c>Heₙ(x)</c>.
        /// <para>エルミート多項式（確率論者の流儀）<c>Heₙ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The non-negative degree (<c>n ≥ 0</c>). / 非負の次数（<c>n ≥ 0</c>）。</param>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>Heₙ(x)</c>, or <see cref="double.NaN"/> if <paramref name="n"/> &lt; 0 or <paramref name="x"/> is <see cref="double.NaN"/>. / <c>Heₙ(x)</c> の値。<paramref name="n"/> &lt; 0 または <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Hermite_polynomials#Definition"/>
        public static double HermiteHe(int n, double x)
        {
            if (n < 0 || double.IsNaN(x))
            {
                return double.NaN;
            }

            if (n == 0)
            {
                return 1;
            }

            if (x == 0)
            {
                // HermiteH と同じく、原点は奇偶性から求める（He₂ₘ(0) = -(2m-1) He₂ₘ₋₂(0)）
                return n % 2 == 1 ? OddOriginValue(n, x) : EvenOriginValue(n, 1);
            }

            // Heₖ₊₁ = x Heₖ - k Heₖ₋₁
            double previous = 1;
            double current = x;
            for (int k = 1; k < n; k++)
            {
                double next = x * current - k * previous;
                previous = current;
                current = next;
            }

            return current;
        }

        // エルミート多項式の原点 x = ±0 での奇数次の値（傾きの符号 (-1)^((n-1)/2) に応じた符号付き 0）
        private static double OddOriginValue(int n, double x) => (n / 2) % 2 == 0 ? x : -x;

        // エルミート多項式の原点での偶数次の値 Πₖ (-scale·k)（k = 1, 3, …, n-1）
        private static double EvenOriginValue(int n, double scale)
        {
            double value = 1;
            for (int k = 1; k < n; k += 2)
            {
                value *= -scale * k;
            }

            return value;
        }

        /// <summary>
        /// Computes the Laguerre polynomial <c>Lₙ(x)</c>.
        /// <para>ラゲール多項式 <c>Lₙ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The non-negative degree (<c>n ≥ 0</c>). / 非負の次数（<c>n ≥ 0</c>）。</param>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>Lₙ(x)</c>, or <see cref="double.NaN"/> if <paramref name="n"/> &lt; 0 or <paramref name="x"/> is <see cref="double.NaN"/>. / <c>Lₙ(x)</c> の値。<paramref name="n"/> &lt; 0 または <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Laguerre_polynomials"/>
        public static double LaguerreL(int n, double x) => LaguerreL(n, 0, x);

        /// <summary>
        /// Computes the generalized (associated) Laguerre polynomial <c>Lₙ^(α)(x)</c>.
        /// <para>一般化ラゲール多項式（ラゲール陪多項式）<c>Lₙ^(α)(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The non-negative degree (<c>n ≥ 0</c>). / 非負の次数（<c>n ≥ 0</c>）。</param>
        /// <param name="alpha">The parameter <c>α</c>. / パラメータ <c>α</c>。</param>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>Lₙ^(α)(x)</c>, or <see cref="double.NaN"/> if <paramref name="n"/> &lt; 0, or if <paramref name="alpha"/> or <paramref name="x"/> is <see cref="double.NaN"/>. / <c>Lₙ^(α)(x)</c> の値。<paramref name="n"/> &lt; 0、または <paramref name="alpha"/> もしくは <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Laguerre_polynomials#Generalized_Laguerre_polynomials"/>
        public static double LaguerreL(int n, double alpha, double x)
        {
            if (n < 0 || double.IsNaN(alpha) || double.IsNaN(x))
            {
                return double.NaN;
            }

            if (n == 0)
            {
                return 1;
            }

            // (k+1) Lₖ₊₁ = (2k+1+α-x) Lₖ - (k+α) Lₖ₋₁
            double previous = 1;
            double current = 1 + alpha - x;
            for (int k = 1; k < n; k++)
            {
                double next = ((2 * k + 1 + alpha - x) * current - (k + alpha) * previous) / (k + 1);
                previous = current;
                current = next;
            }

            return current;
        }

        /// <summary>
        /// Computes the Chebyshev polynomial of the first kind, <c>Tₙ(x)</c>.
        /// <para>第 1 種チェビシェフ多項式 <c>Tₙ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The non-negative degree (<c>n ≥ 0</c>). / 非負の次数（<c>n ≥ 0</c>）。</param>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>Tₙ(x)</c>, or <see cref="double.NaN"/> if <paramref name="n"/> &lt; 0 or <paramref name="x"/> is <see cref="double.NaN"/>. / <c>Tₙ(x)</c> の値。<paramref name="n"/> &lt; 0 または <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Chebyshev_polynomials#Definitions"/>
        public static double ChebyshevT(int n, double x)
        {
            if (n < 0 || double.IsNaN(x))
            {
                return double.NaN;
            }

            if (n == 0)
            {
                return 1;
            }

            // Tₖ₊₁ = 2x Tₖ - Tₖ₋₁
            double previous = 1;
            double current = x;
            for (int k = 1; k < n; k++)
            {
                double next = 2 * x * current - previous;
                previous = current;
                current = next;
            }

            return current;
        }

        /// <summary>
        /// Computes the Chebyshev polynomial of the second kind, <c>Uₙ(x)</c>.
        /// <para>第 2 種チェビシェフ多項式 <c>Uₙ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The non-negative degree (<c>n ≥ 0</c>). / 非負の次数（<c>n ≥ 0</c>）。</param>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>Uₙ(x)</c>, or <see cref="double.NaN"/> if <paramref name="n"/> &lt; 0 or <paramref name="x"/> is <see cref="double.NaN"/>. / <c>Uₙ(x)</c> の値。<paramref name="n"/> &lt; 0 または <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Chebyshev_polynomials#Definitions"/>
        public static double ChebyshevU(int n, double x)
        {
            if (n < 0 || double.IsNaN(x))
            {
                return double.NaN;
            }

            if (n == 0)
            {
                return 1;
            }

            // Uₖ₊₁ = 2x Uₖ - Uₖ₋₁
            double previous = 1;
            double current = 2 * x;
            for (int k = 1; k < n; k++)
            {
                double next = 2 * x * current - previous;
                previous = current;
                current = next;
            }

            return current;
        }

        /// <summary>
        /// Computes the Gegenbauer (ultraspherical) polynomial <c>Cₙ^(λ)(x)</c>.
        /// <para>ゲーゲンバウアー多項式 <c>Cₙ^(λ)(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The non-negative degree (<c>n ≥ 0</c>). / 非負の次数（<c>n ≥ 0</c>）。</param>
        /// <param name="lambda">The parameter <c>λ</c>. / パラメータ <c>λ</c>。</param>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>Cₙ^(λ)(x)</c>, or <see cref="double.NaN"/> if <paramref name="n"/> &lt; 0, or if <paramref name="lambda"/> or <paramref name="x"/> is <see cref="double.NaN"/>. / <c>Cₙ^(λ)(x)</c> の値。<paramref name="n"/> &lt; 0、または <paramref name="lambda"/> もしくは <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/18.3.T1"/>
        public static double GegenbauerC(int n, double lambda, double x)
        {
            if (n < 0 || double.IsNaN(lambda) || double.IsNaN(x))
            {
                return double.NaN;
            }

            if (n == 0)
            {
                return 1;
            }

            // (k+1) Cₖ₊₁ = 2x(k+λ) Cₖ - (k+2λ-1) Cₖ₋₁
            double previous = 1;
            double current = 2 * lambda * x;
            for (int k = 1; k < n; k++)
            {
                double next = (2 * x * (k + lambda) * current - (k + 2 * lambda - 1) * previous) / (k + 1);
                previous = current;
                current = next;
            }

            return current;
        }

        /// <summary>
        /// Computes the Jacobi polynomial <c>Pₙ^(α,β)(x)</c>.
        /// <para>ヤコビ多項式 <c>Pₙ^(α,β)(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The non-negative degree (<c>n ≥ 0</c>). / 非負の次数（<c>n ≥ 0</c>）。</param>
        /// <param name="alpha">The first parameter <c>α</c>. / 第 1 パラメータ <c>α</c>。</param>
        /// <param name="beta">The second parameter <c>β</c>. / 第 2 パラメータ <c>β</c>。</param>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>Pₙ^(α,β)(x)</c>, or <see cref="double.NaN"/> if <paramref name="n"/> &lt; 0 or any floating-point argument is <see cref="double.NaN"/>. / <c>Pₙ^(α,β)(x)</c> の値。<paramref name="n"/> &lt; 0 またはいずれかの浮動小数点引数が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/18.3.T1"/>
        public static double JacobiP(int n, double alpha, double beta, double x)
        {
            if (n < 0 || double.IsNaN(alpha) || double.IsNaN(beta) || double.IsNaN(x))
            {
                return double.NaN;
            }

            if (n == 0)
            {
                return 1;
            }

            double ab = alpha + beta;
            double previous = 1;
            double current = (alpha + 1) + (ab + 2) * (x - 1) / 2;

            // 2(k+1)(k+α+β+1)(2k+α+β) Pₖ₊₁
            //   = (2k+α+β+1)[(2k+α+β+2)(2k+α+β)x + α² - β²] Pₖ - 2(k+α)(k+β)(2k+α+β+2) Pₖ₋₁
            for (int k = 1; k < n; k++)
            {
                double s = 2 * k + ab;
                double next;
                if (Math.Abs(k + ab + 1) < 0.25 || Math.Abs(s) < 0.25)
                {
                    // 左辺の係数が 0（またはその近く）になると右辺も 0 に近づいて桁落ちするので、有限和で直接求める
                    next = JacobiPSum(k + 1, alpha, beta, x);
                }
                else
                {
                    double a1 = 2 * (k + 1) * (k + ab + 1) * s;
                    double a2 = (s + 1) * (alpha * alpha - beta * beta);
                    double a3 = s * (s + 1) * (s + 2);
                    double a4 = 2 * (k + alpha) * (k + beta) * (s + 2);
                    next = ((a2 + a3 * x) * current - a4 * previous) / a1;
                }

                previous = current;
                current = next;
            }

            return current;
        }

        // 有限和 Pₙ^(α,β)(x) = Σ (n+α+β+1)ₗ (α+l+1)ₙ₋ₗ / (l! (n-l)!) ((x-1)/2)ˡ によるヤコビ多項式の直接評価
        private static double JacobiPSum(int n, double alpha, double beta, double x)
        {
            // (α+l+1)ₙ₋ₗ / (n-l)! を l の降順に求める（α が負の整数でも0で割らない）
            var risingAlpha = new double[n + 1];
            risingAlpha[n] = 1;
            for (int l = n - 1; l >= 0; l--)
            {
                risingAlpha[l] = risingAlpha[l + 1] * (alpha + l + 1) / (n - l);
            }

            double c = n + alpha + beta + 1;
            double y = (x - 1) / 2;
            double risingSum = 1;
            double power = 1;
            double sum = 0;
            for (int l = 0; l <= n; l++)
            {
                sum += risingSum * risingAlpha[l] * power;
                risingSum *= (c + l) / (l + 1);
                power *= y;
            }

            return sum;
        }
    }
}
