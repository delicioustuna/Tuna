namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // ModifiedSphericalBesselK の有限和をリスケーリングする閾値（2^600）とシフト量
        private const double SphericalBesselKSumLimit = 4.149515568880993e180;
        private const int SphericalBesselKSumShift = 600;

        /// <summary>
        /// Computes the modified spherical Bessel function of the first kind <c>iₙ(x)</c>.
        /// <para>第 1 種変形球ベッセル関数 <c>iₙ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The non-negative integer order (<c>n ≥ 0</c>). / 非負の整数次数（<c>n ≥ 0</c>）。</param>
        /// <param name="x">The real argument. / 実数の引数。</param>
        /// <returns>The value of <c>iₙ(x)</c>, or <see cref="double.NaN"/> if <paramref name="n"/> &lt; 0 or <paramref name="x"/> is <see cref="double.NaN"/>. / <c>iₙ(x)</c> の値。<paramref name="n"/> &lt; 0 または <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Satisfies <c>iₙ(-x) = (-1)ⁿ iₙ(x)</c>, with <c>i₀(0) = 1</c> and <c>iₙ(0) = 0</c> for <c>n &gt; 0</c>.</para>
        /// <para><c>iₙ(-x) = (-1)ⁿ iₙ(x)</c> を満たし、<c>i₀(0) = 1</c>、<c>n &gt; 0</c> では <c>iₙ(0) = 0</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/10.47.E7"/>
        public static double ModifiedSphericalBesselI(int n, double x)
        {
            if (double.IsNaN(x) || n < 0)
            {
                return double.NaN;
            }

            if (x == 0)
            {
                return n == 0 ? 1 : 0;
            }

            double a = Math.Abs(x);
            double value;
            if (double.IsPositiveInfinity(a))
            {
                value = double.PositiveInfinity;
            }
            else if (n == 0)
            {
                if (a > 700)
                {
                    // sinh は x ≈ 710 でオーバーフローするが、1/x を掛けた結果はまだ有限なことがある。
                    // e^x を仮数と 2 の冪に分け、1/(2x) に掛けてから最後に指数を戻す（e^(-x) の項は無視できる）
                    double exp = ConfluentScaledExp(a, 0, out long exponent);
                    value = ScaleB(0.5 / a * exp, exponent);
                }
                else
                {
                    // i₀(x) = sinh x / x
                    value = a < 1e-8 ? 1 + a * a / 6 : Math.Sinh(a) / a;
                }
            }
            else if (a <= 1)
            {
                // 小さい x では I_(n+1/2) がアンダーフローし、非正規数の x では √(π/(2x)) がオーバーフローしても、iₙ は範囲内のことがある
                value = ModifiedSphericalBesselISeries(n, a);
            }
            else
            {
                // √(π/(2x)) I_(n+1/2)(x)。大きい n では e⁻ˣ I や I が、x ≈ 710 では eˣ が倍精度の範囲を外れても iₙ は範囲内のことがある。
                // e⁻ˣ I と eˣ を仮数と 2 の冪のまま掛け、最後に 1 回だけ丸める
                double scaled = BesselIScaledPositive(n + 0.5, a, out long exponent);
                double exp = ConfluentScaledExp(a, 0, out long expExponent);
                value = scaled == 0 ? 0 : ScaleB(Math.Sqrt(Math.PI / (2 * a)) * scaled * exp, exponent + expExponent);
            }

            return x < 0 && n % 2 == 1 ? -value : value;
        }

        // DLMF 10.53.E3: iₙ(x) = xⁿ/(2n+1)!! Σₖ (x²/2)ᵏ/(k! (2n+3)(2n+5)…(2n+2k+1))（n ≥ 1、0 < x ≤ 1。全項が正）。
        // 前因子 xⁿ/(2n+1)!! は仮数と 2 の冪で保持する
        private static double ModifiedSphericalBesselISeries(int n, double x)
        {
            double mantissa = ScaledPower(x, n, out long exponent);
            // (2n+1)!! は整数のまま正確に掛け、2 の冪を指数に移す（2⁵³ を超えるまでは丸めない）。
            // x ≤ 1 では指数が -1100 を下回ると結果は 0 なので打ち切る（1 因子で 1 以上下がるので、反復は高々約 1100 回）
            double doubleFactorial = 1;
            for (int j = 1; j <= n; j++)
            {
                doubleFactorial *= 2 * j + 1;
                int shift = Math.ILogB(doubleFactorial);
                doubleFactorial = Math.ScaleB(doubleFactorial, -shift);
                exponent -= shift;
                if (exponent < -1100)
                {
                    return 0;
                }
            }

            double t = 0.5 * x * x;
            double term = 1;
            double sum = 1;
            for (int k = 0; term > sum * Epsilon * 0.5; k++)
            {
                term *= t / ((k + 1) * (2.0 * n + 2 * k + 3));
                sum += term;
            }

            return ScaleB(mantissa / doubleFactorial * sum, exponent);
        }

        /// <summary>
        /// Computes the modified spherical Bessel function of the second kind <c>kₙ(x)</c>.
        /// <para>第 2 種変形球ベッセル関数 <c>kₙ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The integer order (negative orders are mapped via <c>k₋ₙ₋₁ = kₙ</c>). / 整数次数（負の次数は <c>k₋ₙ₋₁ = kₙ</c> により対称に扱われます）。</param>
        /// <param name="x">The non-negative argument (<c>x ≥ 0</c>). / 非負の引数（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>kₙ(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> &lt; 0 or <see cref="double.NaN"/>. / <c>kₙ(x)</c> の値。<paramref name="x"/> &lt; 0 または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Follows the DLMF normalization <c>kₙ(x) = √(π / (2x)) K_(n+1/2)(x)</c> (matching SciPy's <c>spherical_kn</c>), which differs by a factor of <c>π / 2</c> from the <c>√(2 / (πx)) K_(n+1/2)(x)</c> convention. Returns <see cref="double.PositiveInfinity"/> at <c>x = 0</c> and <c>0</c> at <c>x = +∞</c>.</para>
        /// <para>正規化は DLMF（SciPy の <c>spherical_kn</c> と同じ <c>kₙ(x) = √(π / (2x)) K_(n+1/2)(x)</c>）に従います。<c>√(2 / (πx)) K_(n+1/2)(x)</c> の定義とは <c>π / 2</c> 倍異なります。<c>x = 0</c> では <see cref="double.PositiveInfinity"/>、<c>x = +∞</c> では <c>0</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/10.47.E9"/>
        public static double ModifiedSphericalBesselK(int n, double x)
        {
            if (double.IsNaN(x) || x < 0)
            {
                return double.NaN;
            }

            if (n < 0)
            {
                n = -n - 1;
            }

            if (x == 0)
            {
                return double.PositiveInfinity;
            }

            if (double.IsPositiveInfinity(x))
            {
                return 0;
            }

            // 和は Σ cₖ (2x)^(-k)、cₖ₊₁/cₖ = (n+k+1)(n-k)/(k+1)。
            // 和だけがオーバーフローしても小さい e^(-x)/x を掛けると有限になりうるので、和は sum · 2^exponent で持つ
            double sum = 1;
            double term = 1;
            long exponent = 0;
            double inv2x = 0.5 / x;
            for (int k = 0; k < n; k++)
            {
                term *= (double)(n + k + 1) * (n - k) / (k + 1) * inv2x;
                sum += term;
                if (sum > SphericalBesselKSumLimit)
                {
                    sum = Math.ScaleB(sum, -SphericalBesselKSumShift);
                    term = Math.ScaleB(term, -SphericalBesselKSumShift);
                    exponent += SphericalBesselKSumShift;
                }

                if (double.IsPositiveInfinity(sum))
                {
                    // 1段で 2^424 倍以上に増えるのは (2x)^(-1) が巨大な x ≪ 1 のときだけで、そのとき e^(-x)/x ≥ 1 なので結果もオーバーフローする
                    return sum;
                }
            }

            double prefactor = Math.PI / 2 * Math.Exp(-x) / x;
            if (exponent == 0 && prefactor >= MinNormal)
            {
                return prefactor * sum;
            }

            // e^(-x)/x がアンダーフローする場合や和を縮めた場合は、e^(-x) を仮数と2の冪に分けて掛け、最後に指数を戻す
            double exp = ConfluentScaledExp(-x, 0, out long expExponent);
            return ScaleB(Math.PI / 2 / x * sum * exp, exponent + expExponent);
        }
    }
}
