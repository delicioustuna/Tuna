namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // Lanczos 近似（g = 6.024680040776729583740234375, N = 13）の係数（分子・分母の多項式比）
        private const double LanczosG = 6.024680040776729583740234375;
        private const double LanczosGMinusHalf = 5.524680040776729583740234375;

        private static readonly double[] LanczosNumerator =
        [
            23531376880.410759688572007674451636754734846804940,
            42919803642.649098768957899047001988850926355848959,
            35711959237.355668049440185451547166705960488635843,
            17921034426.037209699919755754458931112671403265390,
            6039542586.3520280050642916443072979210699388420708,
            1439720407.3117216736632230727949123939715485786772,
            248874557.86205415651146038641322942321632125127801,
            31426415.585400194380614231628318205362874684987640,
            2876370.6289353724412254090516208496135991145378768,
            186056.26539522349504029498971604569928220784236328,
            8071.6720023658162106380029022722506138218516325024,
            210.82427775157934587250973392071336271166969580291,
            2.5066282746310002701649081771338373386264310793408,
        ];

        private static readonly double[] LanczosDenominator =
        [
            0.0, 39916800.0, 120543840.0, 150917976.0, 105258076.0, 45995730.0, 13339535.0,
            2637558.0, 357423.0, 32670.0, 1925.0, 66.0, 1.0,
        ];

        // ln Γ(1 + z) = -γz + Σ (-1)ᵏ ζ(k)/k zᵏ の係数（添字 k、静的コンストラクターで初期化）
        private static readonly double[] LogGamma1pCoefficients;

        private static double[] CreateLogGamma1pCoefficients()
        {
            var coefficients = new double[ZetaAtInteger.Length];
            for (int k = 2; k < coefficients.Length; k++)
            {
                coefficients[k] = (k % 2 == 0 ? 1 : -1) * ZetaAtInteger[k] / k;
            }

            return coefficients;
        }

        // Lanczos 近似の有理関数部分（x > 0）
        private static double LanczosSum(double x)
        {
            double numerator = 0;
            double denominator = 0;
            if (x < 5)
            {
                for (int i = LanczosNumerator.Length - 1; i >= 0; i--)
                {
                    numerator = numerator * x + LanczosNumerator[i];
                    denominator = denominator * x + LanczosDenominator[i];
                }
            }
            else
            {
                for (int i = 0; i < LanczosNumerator.Length; i++)
                {
                    numerator = numerator / x + LanczosNumerator[i];
                    denominator = denominator / x + LanczosDenominator[i];
                }
            }

            return numerator / denominator;
        }

        // ln Γ(1 + z) のべき級数（|z| ≤ 1/4 を想定）
        private static double LogGamma1pSeries(double z)
        {
            double sum = 0;
            for (int k = LogGamma1pCoefficients.Length - 1; k >= 2; k--)
            {
                sum = sum * z + LogGamma1pCoefficients[k];
            }

            return sum * z * z - EulerGamma * z;
        }

        // ln Γ(x + d) - ln Γ(x)（x > 0, 0 < d ≤ 1）を d が小さくても相対精度を保って計算する。
        // x < 10 は ln(1 + d/x) を引いて x + 1 へ移し、x ≥ 10 ではスターリングの公式の差を
        // u = d/x として x(ln(1+u) - u) - ln(1+u)/2 + d ln(x + d) + Σ cₖ x^(1-2k) ((1+u)^(1-2k) - 1) で評価する
        private static double LogGammaShift(double x, double d)
        {
            // x = 1 は ln Γ(1 + d)。d ≤ 1/4 では既存のべき級数が速く正確で、d > 1/4 では値が小さくないので直接求めてよい
            if (x == 1)
            {
                return d <= 0.25 ? LogGamma1pSeries(d) : LogGamma(1 + d);
            }

            double sum = 0;
            while (x < 10)
            {
                sum -= Log1p(d / x);
                x += 1;
            }

            double u = d / x;
            double log1pU = Log1p(u);

            // 補正項の差 δ(x+d) - δ(x) は値全体の約 1/(12x² ln x) 倍以下なので、u ≥ 1e-3 なら 2 つの δ の差でも相対誤差は ε 程度。
            // それより小さい u では差を項ごとに (1+u)^(1-2k) - 1 から求める
            double correction;
            if (u >= 1e-3)
            {
                correction = StirlingCorrection(x + d) - StirlingCorrection(x);
            }
            else
            {
                correction = 0;
                double power = 1 / x;
                double inverseSquare = power * power;
                for (int k = 1; k <= StirlingSeriesCoefficients.Length; k++)
                {
                    correction += StirlingSeriesCoefficients[k - 1] * power * Expm1((1 - 2 * k) * log1pU);
                    power *= inverseSquare;
                }
            }

            return sum + (x * Log1pmx(u) - 0.5 * log1pU + d * Math.Log(x + d) + correction);
        }

        // ln Γ(1 + a)（a > -1）。|a| ≤ 1/4 ではべき級数を使うため 1 + a の丸めで精度が落ちない
        internal static double LogGamma1p(double a) => Math.Abs(a) <= 0.25 ? LogGamma1pSeries(a) : LogGamma(1 + a);

        // Γ(1 + a) - 1 を桁落ちなく計算する
        private static double Gamma1pm1(double a)
        {
            if (Math.Abs(a) <= 0.25)
            {
                return Expm1(LogGamma1pSeries(a));
            }

            if (Math.Abs(a - 1) <= 0.25)
            {
                double z = a - 1;
                return Expm1(Log1p(z) + LogGamma1pSeries(z));
            }

            return Gamma(1 + a) - 1;
        }

        // 負の非整数でのガンマ関数の符号
        private static double SignGammaOfNegative(double x) => IsOddInteger(Math.Floor(x)) ? -1.0 : 1.0;

        /// <summary>
        /// Computes the gamma function <c>Γ(x)</c>.
        /// <para>ガンマ関数 <c>Γ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>Γ(x)</c>, <c>±∞</c> at <c>±0</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is a negative integer, <see cref="double.NegativeInfinity"/>, or <see cref="double.NaN"/>. / <c>Γ(x)</c> の値。<c>±0</c> では <c>±∞</c>、<paramref name="x"/> が負の整数、<see cref="double.NegativeInfinity"/>、または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Gamma_function#Main_definition"/>
        public static double Gamma(double x)
        {
            if (double.IsNaN(x))
            {
                return x;
            }

            if (double.IsInfinity(x))
            {
                return x > 0 ? x : double.NaN;
            }

            if (x == 0)
            {
                return 1 / x;
            }

            if (IsInteger(x))
            {
                if (x < 0)
                {
                    return double.NaN;
                }

                return x <= FactorialTable.Length ? FactorialTable[(int)x - 1] : double.PositiveInfinity;
            }

            double absx = Math.Abs(x);
            if (absx < 1e-20)
            {
                return 1 / x;
            }

            if (absx > 200)
            {
                return x > 0 ? double.PositiveInfinity : SignGammaOfNegative(x) * 0.0;
            }

            // y = absx + g - 1/2 の丸め誤差 z を求め、y^(absx-1/2) e^(-y) への影響を 1 次の補正項で打ち消す
            double y = absx + LanczosGMinusHalf;
            double z;
            if (absx > LanczosGMinusHalf)
            {
                double q = y - absx;
                z = q - LanczosGMinusHalf;
            }
            else
            {
                double q = y - LanczosGMinusHalf;
                z = q - absx;
            }

            z = z * LanczosG / y;

            double r;
            if (x < 0)
            {
                r = -Math.PI / double.SinPi(absx) / absx * Math.Exp(y) / LanczosSum(absx);
                r -= z * r;
                if (absx < 140)
                {
                    r /= Math.Pow(y, absx - 0.5);
                }
                else
                {
                    double sqrtPow = Math.Pow(y, absx / 2 - 0.25);
                    r /= sqrtPow;
                    r /= sqrtPow;
                }
            }
            else
            {
                r = LanczosSum(absx) / Math.Exp(y);
                r += z * r;
                if (absx < 140)
                {
                    r *= Math.Pow(y, absx - 0.5);
                }
                else
                {
                    double sqrtPow = Math.Pow(y, absx / 2 - 0.25);
                    r *= sqrtPow;
                    r *= sqrtPow;
                }
            }

            return r;
        }

        /// <summary>
        /// Computes the natural logarithm of the absolute value of the gamma function, <c>ln |Γ(x)|</c>.
        /// <para>ガンマ関数の絶対値の自然対数 <c>ln |Γ(x)|</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>ln |Γ(x)|</c>, or <see cref="double.PositiveInfinity"/> at non-positive integers and infinities. / <c>ln |Γ(x)|</c> の値。0 以下の整数および無限大では <see cref="double.PositiveInfinity"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Gamma_function#Log-gamma_function"/>
        public static double LogGamma(double x)
        {
            if (double.IsNaN(x))
            {
                return x;
            }

            if (double.IsInfinity(x))
            {
                return double.PositiveInfinity;
            }

            if (IsNonPositiveInteger(x))
            {
                return double.PositiveInfinity;
            }

            if (Math.Abs(x) < 0.25)
            {
                // ln |Γ(x)| = ln Γ(1 + x) - ln |x|。|x| が非正規化数だと反射公式の sin(π|x|) の精度が落ちるので、負の x でも反射公式を使わない
                return LogGamma1pSeries(x) - Math.Log(Math.Abs(x));
            }

            if (x < 0)
            {
                // Γ(x)Γ(-x) = -π / (x sin(πx))
                double absx = -x;
                double logGammaOfAbs = LogGamma(absx);
                double value = LnPi - Math.Log(Math.Abs(double.SinPi(absx))) - Math.Log(absx) - logGammaOfAbs;
                // 対数にした反射項と ln Γ(|x|) が打ち消す場合は、Γ(x)Γ(1 - x) = π / sin(πx) で
                // 丸め誤差を保って再評価し、ln |Γ| の負の零点付近の値と符号を保つ
                return double.IsFinite(value) && Math.Abs(value) < GammaReflectionCancellationRatio * Math.Max(1, Math.Abs(logGammaOfAbs))
                    ? LogGammaReflectedNearZeros(x)
                    : value;
            }

            if (Math.Abs(x - 1) <= 0.25)
            {
                return LogGamma1pSeries(x - 1);
            }

            if (Math.Abs(x - 2) <= 0.25)
            {
                double z = x - 2;
                return Log1p(z) + LogGamma1pSeries(z);
            }

            if (x < 10)
            {
                return Math.Log(Gamma(x));
            }

            double lnx = Math.Log(x);
            return x * (lnx - 1) - 0.5 * lnx + LnSqrt2Pi + StirlingCorrection(x);
        }

        /// <summary>
        /// Computes the sign of the gamma function, <c>sgn Γ(x)</c>.
        /// <para>ガンマ関数の符号 <c>sgn Γ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns><c>1</c> or <c>-1</c>, or <see cref="double.NaN"/> at negative integers, <see cref="double.NegativeInfinity"/>, and <see cref="double.NaN"/>. / <c>1</c> または <c>-1</c>。負の整数、<see cref="double.NegativeInfinity"/>、および <see cref="double.NaN"/> では <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <c>1</c> at <c>+0</c> and <c>-1</c> at <c>-0</c>, matching the sign of <see cref="Gamma(double)"/>.</para>
        /// <para><see cref="Gamma(double)"/> の符号に合わせて、<c>+0</c> では <c>1</c>、<c>-0</c> では <c>-1</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Gamma_function#Main_definition"/>
        public static double SignGamma(double x)
        {
            if (double.IsNaN(x))
            {
                return x;
            }

            if (x == 0)
            {
                return double.IsNegative(x) ? -1 : 1;
            }

            if (x > 0)
            {
                return 1;
            }

            return IsInteger(x) || double.IsInfinity(x) ? double.NaN : SignGammaOfNegative(x);
        }

        /// <summary>
        /// Computes the reciprocal gamma function <c>1 / Γ(x)</c>.
        /// <para>ガンマ関数の逆数 <c>1 / Γ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>1 / Γ(x)</c>, <c>0</c> at non-positive integers and <see cref="double.PositiveInfinity"/>, or <see cref="double.NaN"/> at <see cref="double.NegativeInfinity"/> and <see cref="double.NaN"/>. / <c>1 / Γ(x)</c> の値。0 以下の整数および <see cref="double.PositiveInfinity"/> では <c>0</c>、<see cref="double.NegativeInfinity"/> および <see cref="double.NaN"/> では <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Reciprocal_gamma_function"/>
        public static double ReciprocalGamma(double x)
        {
            if (double.IsNaN(x))
            {
                return x;
            }

            if (double.IsInfinity(x))
            {
                return x > 0 ? 0 : double.NaN;
            }

            if (IsNonPositiveInteger(x))
            {
                return 0;
            }

            // 1/Γ(x) = x + γx² + O(x³)。|x| < 1e-20 では γx² が x の半 ulp より小さく、正しく丸めた値は x そのもの
            if (Math.Abs(x) < 1e-20)
            {
                return x;
            }

            if (x > 0)
            {
                return x < 171 ? 1 / Gamma(x) : Math.Exp(-LogGamma(x));
            }

            // 1/Γ(x) = Γ(1-x) sin(πx) / π。1 - x は 2 の冪をまたぐと丸まるので、丸めた値 u の誤差 e を Γ(u + e) ≈ Γ(u)(1 + ψ(u)e) で補う
            var (oneMinusX, error) = DoubleDoubleTwoSum(1, -x);
            double sinPi = double.SinPi(x);
            double shift = error == 0 ? 0 : Digamma(oneMinusX) * error;
            if (oneMinusX < 171)
            {
                return Gamma(oneMinusX) * (1 + shift) * sinPi / Math.PI;
            }

            return Math.Sign(sinPi) * Math.Exp(LogGamma(oneMinusX) + shift + Math.Log(Math.Abs(sinPi)) - LnPi);
        }

        /// <summary>
        /// Computes the ratio of gamma functions <c>Γ(a) / Γ(b)</c>.
        /// <para>ガンマ関数の比 <c>Γ(a) / Γ(b)</c> を計算します。</para>
        /// </summary>
        /// <param name="a">The numerator parameter. / 分子の引数。</param>
        /// <param name="b">The denominator parameter. / 分母の引数。</param>
        /// <returns>The value of <c>Γ(a) / Γ(b)</c>, or <see cref="double.NaN"/> if <paramref name="a"/> is a non-positive integer while <paramref name="b"/> is not, or if either argument is <see cref="double.NaN"/>. / <c>Γ(a) / Γ(b)</c> の値。<paramref name="a"/> のみが 0 以下の整数の場合、またはいずれかの引数が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>When both <paramref name="a"/> and <paramref name="b"/> are non-positive integers (<c>a = -m</c>, <c>b = -k</c>), returns the limiting value <c>(-1)^(m - k) k! / m!</c>.</para>
        /// <para><paramref name="a"/> と <paramref name="b"/> がともに 0 以下の整数（<c>a = -m</c>, <c>b = -k</c>）のときは極限値 <c>(-1)^(m - k) k! / m!</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Gamma_function#Main_definition"/>
        public static double GammaRatio(double a, double b)
        {
            if (double.IsNaN(a) || double.IsNaN(b))
            {
                return double.NaN;
            }

            if (a == b)
            {
                return 1;
            }

            bool aPole = IsNonPositiveInteger(a);
            bool bPole = IsNonPositiveInteger(b);
            if (aPole || bPole)
            {
                if (aPole && bPole)
                {
                    double m = -a;
                    double k = -b;
                    return (IsOddInteger(m - k) ? -1 : 1) * GammaRatio(k + 1, m + 1);
                }

                return bPole ? 0 : double.NaN;
            }

            double d = a - b;
            if (IsInteger(d) && Math.Abs(d) <= 20 && b + d == a)
            {
                // d > 0: b(b+1)…(a-1)、d < 0: 1/(a(a+1)…(b-1))
                return d > 0 ? ScaledShiftProduct(b, 0, (int)d, 1, false) : ScaledShiftProduct(a, 0, (int)-d, 1, true);
            }

            if (a > 0 && b > 0)
            {
                if (double.IsPositiveInfinity(a))
                {
                    return double.IsPositiveInfinity(b) ? double.NaN : double.PositiveInfinity;
                }

                if (double.IsPositiveInfinity(b))
                {
                    return 0;
                }

                if (a >= 10 && b >= 10)
                {
                    // ln Γ(a) - ln Γ(b) = (a - 1/2) ln(a/b) + (a - b)(ln b - 1) + δ(a) - δ(b)
                    return Math.Exp((a - 0.5) * Log1p(d / b) + d * (Math.Log(b) - 1)
                        + StirlingCorrection(a) - StirlingCorrection(b));
                }

                if (a < 171 && b < 171)
                {
                    return Gamma(a) / Gamma(b);
                }

                return Math.Exp(LogGamma(a) - LogGamma(b));
            }

            double ga = Gamma(a);
            double rgb = ReciprocalGamma(b);
            double ratio = ga * rgb;
            if (double.IsFinite(ga) && ga != 0 && double.IsFinite(ratio) && ratio != 0)
            {
                return ratio;
            }

            return SignGamma(a) * SignGamma(b) * Math.Exp(LogGamma(a) - LogGamma(b));
        }

        /// <summary>
        /// Computes the Pochhammer symbol (rising factorial) <c>(x)ₙ = Γ(x + n) / Γ(x)</c>.
        /// <para>ポッホハマー記号（上昇階乗）<c>(x)ₙ = Γ(x + n) / Γ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The base argument. / 引数。</param>
        /// <param name="n">The real order. / 実数の次数。</param>
        /// <returns>The value of <c>(x)ₙ</c>. / <c>(x)ₙ</c> の値。</returns>
        /// <remarks>
        /// <para>For integer <paramref name="n"/>, equals the product <c>x(x + 1)…(x + n - 1)</c> for <c>n &gt; 0</c>, <c>1</c> for <c>n = 0</c>, and <c>1 / ((x - 1)…(x + n))</c> for <c>n &lt; 0</c>.</para>
        /// <para>整数 <paramref name="n"/> に対しては、<c>n &gt; 0</c> で <c>x(x + 1)…(x + n - 1)</c>、<c>n = 0</c> で <c>1</c>、<c>n &lt; 0</c> で <c>1 / ((x - 1)…(x + n))</c> と一致します。</para>
        /// </remarks>
        /// <seealso href="https://mathworld.wolfram.com/PochhammerSymbol.html"/>
        public static double Pochhammer(double x, double n)
        {
            if (double.IsNaN(x) || double.IsNaN(n))
            {
                return double.NaN;
            }

            if (n == 0)
            {
                return 1;
            }

            if (IsInteger(n) && Math.Abs(n) <= 20)
            {
                // (x)ₙ = x(x+1)…(x+n-1)、(x)₋ₘ = 1 / ((x-1)(x-2)…(x-m))
                return n > 0 ? ScaledShiftProduct(x, 0, (int)n, 1, false) : ScaledShiftProduct(x, -1, (int)-n, -1, true);
            }

            if (x > 0 && x + n > 0)
            {
                return GammaShiftRatio(x, n, false);
            }

            return GammaRatio(x + n, x);
        }

        /// <summary>
        /// Computes the falling factorial <c>x(x - 1)…(x - n + 1) = Γ(x + 1) / Γ(x - n + 1)</c>.
        /// <para>下降階乗 <c>x(x - 1)…(x - n + 1) = Γ(x + 1) / Γ(x - n + 1)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The base argument. / 引数。</param>
        /// <param name="n">The real order. / 実数の次数。</param>
        /// <returns>The value of the falling factorial. / 下降階乗の値。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Falling_and_rising_factorials"/>
        public static double FallingFactorial(double x, double n)
        {
            if (double.IsNaN(x) || double.IsNaN(n))
            {
                return double.NaN;
            }

            if (n == 0)
            {
                return 1;
            }

            if (IsInteger(n) && n > 0 && n <= 20)
            {
                return ScaledShiftProduct(x, 0, (int)n, -1, false);
            }

            // Γ(x + 1)/Γ(x - n + 1) = (x - n + 1)ₙ。x - n + 1 の丸めは比への影響が小さいが、差 n は別に保持する
            double lower = x - n + 1;
            if (x + 1 > 0 && lower > 0)
            {
                return GammaShiftRatio(lower, n, false);
            }

            return GammaRatio(x + 1, lower);
        }

        // Γ(x + d)/Γ(x)（x > 0, x + d > 0; invert = true なら逆数）。x + d の丸めで差 d が失われないよう d を保持して計算する
        private static double GammaShiftRatio(double x, double d, bool invert)
        {
            if (d == 0)
            {
                return 1;
            }

            if (IsInteger(d) && Math.Abs(d) <= 20)
            {
                // d > 0: x(x+1)…(x+d-1)、d < 0: 1/((x-1)(x-2)…(x+d))
                return d > 0 ? ScaledShiftProduct(x, 0, (int)d, 1, invert) : ScaledShiftProduct(x, -1, (int)-d, -1, !invert);
            }

            double y = x + d;
            if (x >= 10 && y >= 10 && double.IsFinite(x))
            {
                double log = StirlingLogGammaShift(x, d, y);
                return Math.Exp(invert ? -log : log);
            }

            return invert ? GammaRatio(x, y) : GammaRatio(y, x);
        }

        // ln(Γ(x + d)/Γ(x))（x > 0, x + d > 0, 有限）。差 d を保持して x + d が x に丸まる領域でも d の寄与を残す
        private static double LogGammaShiftRatio(double x, double d)
        {
            double y = x + d;
            return x >= 10 && y >= 10 ? StirlingLogGammaShift(x, d, y) : LogGamma(y) - LogGamma(x);
        }

        // スターリング展開による ln Γ(x + d) - ln Γ(x)（x ≥ 10, y = x + d ≥ 10）
        private static double StirlingLogGammaShift(double x, double d, double y)
        {
            double log1pRatio = Log1p(d / x);
            return (x - 0.5) * log1pRatio + d * (Math.Log(x) + log1pRatio - 1)
                + StirlingCorrection(y) - StirlingCorrection(x);
        }

        // 積 (x + start)(x + start + step)…（reciprocal = true なら逆数）を仮数と 2 進指数に分離して計算する
        private static double ScaledShiftProduct(double x, int start, int count, int step, bool reciprocal)
        {
            double mantissa = 1;
            long exponent = 0;
            for (int i = 0, offset = start; i < count; i++, offset += step)
            {
                mantissa *= x + offset;
                if (mantissa != 0 && double.IsFinite(mantissa))
                {
                    int shift = Math.ILogB(mantissa);
                    mantissa = Math.ScaleB(mantissa, -shift);
                    exponent += shift;
                }
            }

            int scale = (int)Math.Clamp(exponent, -4096, 4096);
            return reciprocal ? Math.ScaleB(1 / mantissa, -scale) : Math.ScaleB(mantissa, scale);
        }
    }
}
