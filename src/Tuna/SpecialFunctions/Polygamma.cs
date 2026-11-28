using static Tuna.XMath.Constants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // ディガンマ関数の正の零点 x₀ = 1.46163…（double-double）と零点まわりのテイラー係数 ψ⁽ⁿ⁾(x₀)/n!（n = 1..30）
        private const double DigammaRootHi = 1.4616321449683622;
        private const double DigammaRootLo = 9.549995429965697e-17;

        private static readonly double[] DigammaRootTaylor =
        [
            0.9676722454476212, -0.4427631689835921, 0.258499760955651, -0.16394270544240652,
            0.10782405069126237, -0.07219956125645471, 0.04880428816414311, -0.03316112647484736,
            0.022597648232218104, -0.01542476590494896, 0.010538791616612175, -0.007204534386356869,
            0.004926781395729853, -0.003369801655439328, 0.002305126326734928, -0.0015769367714301972,
            0.0010788252019162967, -0.0007380709389960052, 0.000504953265834602, -0.0003454680251063077,
            0.00023635601564027053, -0.00016170622091974803, 0.0001106337276874741, -7.569179582195066e-05,
            5.178575795222081e-05, -3.5430070947659604e-05, 2.424006611860132e-05, -1.6584242271854135e-05,
            1.134638458466385e-05, -7.762817668462094e-06,
        ];

        /// <summary>
        /// Computes the digamma function <c>ψ(x) = Γ'(x) / Γ(x)</c>.
        /// <para>ディガンマ関数 <c>ψ(x) = Γ'(x) / Γ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The real argument. / 実数の引数。</param>
        /// <returns>The value of <c>ψ(x)</c>, or <see cref="double.NaN"/> at negative integers, <see cref="double.NegativeInfinity"/>, or <see cref="double.NaN"/>. / <c>ψ(x)</c> の値。負の整数、<see cref="double.NegativeInfinity"/>、または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <see cref="double.NegativeInfinity"/> for <c>+0</c> and <see cref="double.PositiveInfinity"/> for <c>-0</c>.</para>
        /// <para><c>x = +0</c> では <see cref="double.NegativeInfinity"/>、<c>x = -0</c> では <see cref="double.PositiveInfinity"/> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Digamma_function"/>
        public static double Digamma(double x)
        {
            if (double.IsNaN(x) || double.IsPositiveInfinity(x))
            {
                return x;
            }

            if (x <= 0)
            {
                if (x == 0)
                {
                    return double.IsNegative(x) ? double.PositiveInfinity : double.NegativeInfinity;
                }

                if (IsInteger(x) || double.IsInfinity(x))
                {
                    return double.NaN;
                }

                // ψ(1 - x) - ψ(x) = π cot(πx)。反射前の項に比べて差が小さい場合だけ、
                // 拡張精度で引数の丸め残りと打ち消しを補い、負の零点付近の値と符号を保つ。
                double reflected = Digamma(1 - x);
                double value = reflected - Math.PI / double.TanPi(x);
                if (double.IsFinite(value) && Math.Abs(value) < GammaReflectionCancellationRatio * Math.Max(1, Math.Abs(reflected)))
                {
                    var precise = DoubleDoubleDigamma((x, 0));
                    return precise.Hi + precise.Lo;
                }

                return value;
            }

            double d = (x - DigammaRootHi) - DigammaRootLo;
            if (Math.Abs(d) < 0.2)
            {
                double sum = 0;
                for (int n = DigammaRootTaylor.Length - 1; n >= 0; n--)
                {
                    sum = sum * d + DigammaRootTaylor[n];
                }

                return sum * d;
            }

            double result = 0;
            while (x < 10)
            {
                result -= 1 / x;
                x += 1;
            }

            // ψ(x) ~ ln x - 1/(2x) - Σ B₂ₖ / (2k x²ᵏ)
            double z2 = 1 / (x * x);
            double tail = z2 * (1.0 / 12 + z2 * (-1.0 / 120 + z2 * (1.0 / 252 + z2 * (-1.0 / 240 + z2 * (1.0 / 132
                + z2 * (-691.0 / 32760 + z2 * (1.0 / 12 + z2 * (-3617.0 / 8160))))))));
            return result + (Math.Log(x) - 0.5 / x - tail);
        }

        /// <summary>
        /// Computes the trigamma function <c>ψ₁(x) = ψ'(x)</c>.
        /// <para>トリガンマ関数 <c>ψ₁(x) = ψ'(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The real argument. / 実数の引数。</param>
        /// <returns>The value of <c>ψ₁(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is <see cref="double.NegativeInfinity"/> or <see cref="double.NaN"/>. / <c>ψ₁(x)</c> の値。<paramref name="x"/> が <see cref="double.NegativeInfinity"/> または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <see cref="double.PositiveInfinity"/> at non-positive integer poles and <c>0</c> at <c>x = +∞</c>.</para>
        /// <para>0 以下の整数（2 位の極）では <see cref="double.PositiveInfinity"/>、<c>x = +∞</c> では <c>0</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Trigamma_function"/>
        public static double Trigamma(double x)
        {
            if (double.IsNaN(x))
            {
                return x;
            }

            if (double.IsPositiveInfinity(x))
            {
                return 0;
            }

            if (x <= 0)
            {
                if (double.IsNegativeInfinity(x))
                {
                    return double.NaN;
                }

                if (IsInteger(x))
                {
                    return double.PositiveInfinity;
                }

                // ψ₁(1 - x) + ψ₁(x) = π² / sin²(πx)
                double s = double.SinPi(x);
                return Math.PI * Math.PI / (s * s) - Trigamma(1 - x);
            }

            double result = 0;
            while (x < 10)
            {
                result += 1 / (x * x);
                x += 1;
            }

            // ψ₁(x) ~ 1/x + 1/(2x²) + Σ B₂ₖ / x^(2k+1)
            double z = 1 / x;
            double z2 = z * z;
            double tail = z * (1 + z * (0.5 + z * (1.0 / 6 + z2 * (-1.0 / 30 + z2 * (1.0 / 42 + z2 * (-1.0 / 30
                + z2 * (5.0 / 66 + z2 * (-691.0 / 2730 + z2 * (7.0 / 6)))))))));
            return result + tail;
        }

        /// <summary>
        /// Computes the polygamma function <c>ψ⁽ⁿ⁾(x) = dⁿψ(x) / dxⁿ</c>.
        /// <para>ポリガンマ関数 <c>ψ⁽ⁿ⁾(x) = dⁿψ(x) / dxⁿ</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The non-negative derivative order (<c>n ≥ 0</c>). / 非負の微分階数（<c>n ≥ 0</c>）。</param>
        /// <param name="x">The real argument. / 実数の引数。</param>
        /// <returns>The value of <c>ψ⁽ⁿ⁾(x)</c>, or <see cref="double.NaN"/> if <paramref name="n"/> &lt; 0, <paramref name="x"/> is <see cref="double.NegativeInfinity"/> or <see cref="double.NaN"/>, or at non-positive integers for positive even <paramref name="n"/>. / <c>ψ⁽ⁿ⁾(x)</c> の値。<paramref name="n"/> &lt; 0、<paramref name="x"/> が <see cref="double.NegativeInfinity"/> または <see cref="double.NaN"/>、あるいは正の偶数 <paramref name="n"/> で <paramref name="x"/> が 0 以下の整数の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>For <c>n = 0</c> and <c>n = 1</c>, delegates to <see cref="Digamma(double)"/> and <see cref="Trigamma(double)"/> respectively. At non-positive integers, returns <see cref="double.PositiveInfinity"/> for odd <paramref name="n"/> and <see cref="double.NaN"/> for positive even <paramref name="n"/>.</para>
        /// <para><c>n = 0</c> では <see cref="Digamma(double)"/>、<c>n = 1</c> では <see cref="Trigamma(double)"/> に委譲します。0 以下の整数では、奇数 <paramref name="n"/> で <see cref="double.PositiveInfinity"/>、正の偶数 <paramref name="n"/> で <see cref="double.NaN"/> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Polygamma_function"/>
        public static double Polygamma(int n, double x)
        {
            if (n < 0)
            {
                return double.NaN;
            }

            if (n == 0)
            {
                return Digamma(x);
            }

            if (n == 1)
            {
                return Trigamma(x);
            }

            if (double.IsNaN(x))
            {
                return x;
            }

            if (double.IsPositiveInfinity(x))
            {
                return 0;
            }

            if (x <= 0 && (IsInteger(x) || double.IsInfinity(x)))
            {
                return n % 2 == 1 && !double.IsInfinity(x) ? double.PositiveInfinity : double.NaN;
            }

            if (x < 0)
            {
                if (n <= CotDerivativeMaxOrder)
                {
                    // 相反公式 ψ(1 - x) - ψ(x) = π cot(πx) を n 回微分すると
                    // ψ⁽ⁿ⁾(x) = (-1)ⁿ ψ⁽ⁿ⁾(1 - x) - πⁿ⁺¹ cot⁽ⁿ⁾(πx)
                    double reflected = PolygammaPositive(n, 1 - x);
                    double cotDerivative = CotDerivative(n, 1 / double.TanPi(x));
                    return (n % 2 == 0 ? reflected : -reflected) - Math.Pow(Math.PI, n + 1) * cotDerivative;
                }

                return PolygammaNegative(n, x);
            }

            return PolygammaPositive(n, x);
        }

        // x > 0 での ψ⁽ⁿ⁾(x) = (-1)ⁿ⁺¹ n! ζ(n + 1, x)（n ≥ 1）
        private static double PolygammaPositive(int n, double x)
        {
            // n!、x^(-n-1)、ζ(n + 1, x) はそれぞれ単独でオーバーフローやアンダーフローを起こしうるため、
            // ζ は x^(n+1) ζ(n + 1, x)（1 以上）として求め、3 つを仮数と 2 進指数に分けて掛け合わせる
            double ratio = HurwitzZetaCore(n + 1.0, x, scaled: true);
            int ratioExponent = Math.ILogB(ratio);
            double mantissa = ScaledFactorial(n, out long factorialExponent)
                * ScaledPower(x, -(n + 1L), out long powerExponent)
                * Math.ScaleB(ratio, -ratioExponent);
            long exponent = factorialExponent + powerExponent + ratioExponent;
            double value = ScaleB(mantissa, exponent);
            return n % 2 == 1 ? value : -value;
        }

        // 負の x で相反公式を使う最大の階数（cot の高階導関数は O(n²) となるため、これ以上は PolygammaNegative の方が速い）
        private const int CotDerivativeMaxOrder = 8;

        // cot の n 階導関数を c = cot t の多項式 Pₙ(c)（P₀ = c, Pₙ₊₁ = -(1 + c²) Pₙ'）として評価する
        private static double CotDerivative(int n, double c)
        {
            var coefficients = new double[n + 2];
            coefficients[1] = 1;
            var derivative = new double[n + 2];
            for (int order = 0; order < n; order++)
            {
                Array.Clear(derivative);
                for (int j = 0; j + 1 < coefficients.Length; j++)
                {
                    derivative[j] = (j + 1) * coefficients[j + 1];
                }

                for (int j = 0; j < coefficients.Length; j++)
                {
                    coefficients[j] = -(derivative[j] + (j >= 2 ? derivative[j - 2] : 0));
                }
            }

            double sum = 0;
            for (int j = coefficients.Length - 1; j >= 0; j--)
            {
                sum = sum * c + coefficients[j];
            }

            return sum;
        }

        // 0 以下の整数でない x < 0 での ψ⁽ⁿ⁾(x)（n ≥ 2）。
        // y = x + K ∈ (0, 1)（K = ⌈-x⌉）として極の両側のフルヴィッツゼータに分け、仮数・指数表現で途中オーバーフローを避ける
        private static double PolygammaNegative(int n, double x)
        {
            double s = n + 1.0;
            double k = Math.Ceiling(-x);
            // y = x + K と w = 1 - x は丸められることがある（-1/2 < x < 0 の y や、2 進指数が繰り上がる w）。
            // 丸めの残りは重み (d/p)ˢ に s 倍されて効くので、double-double で求めて補正する。1 - y = -x - (K - 1) は丸めなしに求まる
            var (y, yLo) = DoubleDoubleTwoSum(x, k);
            double yc = -x - (k - 1);
            var (w, wLo) = DoubleDoubleTwoSum(1, -x);
            double d = Math.Min(y, yc);

            // ζ(s, p)·(d/p)ˢ（p ≥ d）を仮数と 2 進指数の組で返す。半整数の x では near と reflected が打ち消し合い、
            // far の相対誤差がそのまま結果の誤差になるので、d/p を丸めずに dˢ と p⁻ˢ を仮数と 2 進指数に分けて求める。
            // far の重み (d/(1 - x))ˢ は n が大きいとアンダーフローするので、指数は最後まで別に持つ
            double dPower = ScaledPower(d, n + 1L, out long dExponent);
            (double Mantissa, long Exponent) Weight(double p)
            {
                if (p == d)
                {
                    return (1, 0);
                }

                double mantissa = dPower * ScaledPower(p, -(n + 1L), out long exponent);
                return (mantissa, dExponent + exponent);
            }

            // pLo は p の丸めの残り。(p/(p + pLo))ˢ ≈ 1 - s·pLo/p で重みを補正する（s·|pLo/p| ≤ s·2⁻⁵³ なので 1 次で足りる）
            (double Mantissa, long Exponent) Term(double p, double pLo = 0)
            {
                var weight = Weight(p);
                double correction = pLo == 0 ? 1 : 1 - s * (pLo / p);
                return (HurwitzZetaCore(s, p, scaled: true) * weight.Mantissa * correction, weight.Exponent);
            }

            double sign = n % 2 == 0 ? -1 : 1;
            (double Mantissa, long Exponent) pair;
            double delta = (yc - y) - yLo;
            if (sign < 0 && delta == 0)
            {
                // 半整数の x では near と reflected が厳密に打ち消し合う
                pair = (0, 0);
            }
            else if (sign < 0 && s * Math.Abs(delta) < d)
            {
                // 偶数階で x が半整数に近いと、near - reflected は相対 s|δ|/d 程度まで打ち消し合う（δ = (1 - y) - y）。
                // ζ(s, y) - ζ(s, 1 - y) の先頭の項は (j + y)⁻ˢ - (j + 1 - y)⁻ˢ = (j + 1 - y)⁻ˢ expm1(s log1p(δ/(j + y))) と
                // 打ち消しなしに求め、残りは (d/(J + y))ˢ ≤ 2⁻⁵³ となる J 項目以降の ζ の差とする。
                // 残りの差の丸め誤差は先頭の項の ε² 倍程度なので無視できる（δ は y の丸めの残りも含めて正確に求める）
                int terms = (int)Math.Clamp(Math.Ceiling(d * (Math.Pow(2, 53 / s) - 1)), 1, 64);
                pair = (0, 0);
                for (int j = 0; j < terms; j++)
                {
                    double p = j + y;
                    double q = j + yc;
                    var weight = Weight(q);
                    pair = AddScaled(pair, (weight.Mantissa * Expm1(s * Log1p(delta / p)), weight.Exponent));
                }

                var tailNear = Term(terms + y);
                var tailReflected = Term(terms + yc);
                pair = AddScaled(pair, AddScaled(tailNear, (-tailReflected.Mantissa, tailReflected.Exponent)));
            }
            else
            {
                var near = Term(y, yLo);
                var reflected = Term(yc);
                pair = AddScaled(near, (sign * reflected.Mantissa, reflected.Exponent));
            }

            var far = Term(w, wLo);
            var (sum, sumExponent) = AddScaled(pair, (-sign * far.Mantissa, far.Exponent));
            if (sum == 0)
            {
                return 0;
            }

            double resultSign = (n % 2 == 1 ? 1 : -1) * Math.Sign(sum);
            // 結果の 2 進指数の概算 log₂ n! - s log₂ d + sumExponent。n > 170 では n! の仮数の計算が O(n) なので、
            // 明らかにオーバーフローやアンダーフローする場合はその前に返す
            double magnitude = LogGamma(n + 1.0) / Ln2 - s * Math.Log2(d) + sumExponent;
            if (magnitude > 1100)
            {
                return resultSign * double.PositiveInfinity;
            }

            if (magnitude < -1200)
            {
                return resultSign * 0.0;
            }

            double mantissa = ScaledFactorial(n, out long factorialExponent)
                * ScaledPower(d, -(long)s, out long powerExponent)
                * sum;
            double value = ScaleB(mantissa, factorialExponent + powerExponent + sumExponent);
            return n % 2 == 1 ? value : -value;
        }

        // 仮数と 2 進指数の組で表した 2 数の和 a + b（仮数は [1, 2) に正規化。和が 0 なら (0, 0)）
        private static (double Mantissa, long Exponent) AddScaled((double Mantissa, long Exponent) a, (double Mantissa, long Exponent) b)
        {
            if (a.Mantissa == 0)
            {
                return Normalize(b);
            }

            if (b.Mantissa == 0)
            {
                return Normalize(a);
            }

            a = Normalize(a);
            b = Normalize(b);
            long exponent = Math.Max(a.Exponent, b.Exponent);
            double sum = ScaleB(a.Mantissa, a.Exponent - exponent) + ScaleB(b.Mantissa, b.Exponent - exponent);
            return Normalize((sum, exponent));

            static (double Mantissa, long Exponent) Normalize((double Mantissa, long Exponent) x)
            {
                if (x.Mantissa == 0)
                {
                    return (0, 0);
                }

                int shift = Math.ILogB(x.Mantissa);
                return (Math.ScaleB(x.Mantissa, -shift), x.Exponent + shift);
            }
        }
    }
}
