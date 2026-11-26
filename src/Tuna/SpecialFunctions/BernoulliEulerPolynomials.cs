namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // 低次では零点中心のテイラー展開を用い、この次数以上では [0, 1] 上でフーリエ級数を用いる (k⁻ⁿ の減衰が速く 30 項程度で収束する)
        private const int AppellFourierDegree = 12;

        /// <summary>
        /// Computes the Bernoulli polynomial <c>Bₙ(x)</c>.
        /// <para>ベルヌーイ多項式 <c>Bₙ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The polynomial degree (non-negative integer). / 多項式の次数（非負整数）。</param>
        /// <param name="x">The real input value. / 実数の入力値。</param>
        /// <returns>The value of <c>Bₙ(x)</c> (with <c>Bₙ(0) = Bₙ</c>), or <see cref="double.NaN"/> if <paramref name="n"/> is negative or <paramref name="x"/> is <see cref="double.NaN"/>. / <c>Bₙ(x)</c> の値（<c>Bₙ(0) = Bₙ</c>）。<paramref name="n"/> が負または <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/24.2.E3"/>
        public static double BernoulliPolynomial(int n, double x) => AppellPolynomial(n, x, euler: false);

        /// <summary>
        /// Computes the Euler polynomial <c>Eₙ(x)</c>.
        /// <para>オイラー多項式 <c>Eₙ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The polynomial degree (non-negative integer). / 多項式の次数（非負整数）。</param>
        /// <param name="x">The real input value. / 実数の入力値。</param>
        /// <returns>The value of <c>Eₙ(x)</c> (where <c>2ⁿ Eₙ(1/2)</c> equals <see cref="EulerNumber"/>), or <see cref="double.NaN"/> if <paramref name="n"/> is negative or <paramref name="x"/> is <see cref="double.NaN"/>. / <c>Eₙ(x)</c> の値（<c>2ⁿ Eₙ(1/2)</c> は <see cref="EulerNumber"/> に一致）。<paramref name="n"/> が負または <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/24.2.E8"/>
        public static double EulerPolynomial(int n, double x) => AppellPolynomial(n, x, euler: true);

        private static double AppellPolynomial(int n, double x, bool euler)
        {
            if (n < 0 || double.IsNaN(x))
            {
                return double.NaN;
            }

            if (n == 0)
            {
                return 1;
            }

            if (n == 1)
            {
                return x - 0.5;
            }

            if (x >= -0.25 && x <= 1)
            {
                return ScaleB(AppellPolynomialNearUnit(n, x, euler, out long nearExponent), nearExponent);
            }

            double sign = 1;
            if (x < 0)
            {
                // Pₙ(1 - x) = (-1)ⁿ Pₙ(x)（1 - x の丸めは相対 |x Pₙ'/Pₙ|·ε 程度）
                x = 1 - x;
                sign = n % 2 == 0 ? 1 : -1;
            }

            if (double.IsPositiveInfinity(x))
            {
                return sign * double.PositiveInfinity;
            }

            double y = x - 0.5;
            if (y >= 0.5 * n)
            {
                return sign * AppellPolynomialCentral(n, y, euler);
            }

            // Pₙ(f + m) へずらす。f + j は丸められない。値がオーバーフローしても符号を失わないよう、各項を仮数と指数に分けて最大の項の指数にそろえる
            double m = Math.Floor(x);
            double f = x - m;
            int power = euler ? n : n - 1;
            ScaledPower(x, power, out long top);
            double baseValue = AppellPolynomialNearUnit(n, f, euler, out long baseExponent);
            double sum = 0;
            for (double j = m - 1; j >= 0 && f + j > 0; j--)
            {
                double term = ScaledPower(f + j, power, out long exponent);
                term = ScaleB(term, exponent - top);
                if (term == 0)
                {
                    break;
                }

                // オイラー：符号 (-1)^(m-1-j)
                sum += euler && (m - 1 - j) % 2 == 1 ? -term : term;
            }

            double shifted = euler ? 2 * sum : n * sum;
            double parity = euler && m % 2 == 1 ? -1 : 1;
            double value = parity * ScaleB(baseValue, baseExponent - top) + shifted;
            return sign * ScaleB(value, top);
        }

        // -1/4 ≤ x ≤ 1 での Pₙ(x) = mantissa · 2^exponent (n ≥ 2)。
        // x < 0 のフーリエ級数は周期的な延長 Bₙ(x + 1), -Eₙ(x + 1) を与えるため、Bₙ(x) = Bₙ(x + 1) - n x^(n-1), Eₙ(x) = 2xⁿ - Eₙ(x + 1) で補正する
        private static double AppellPolynomialNearUnit(int n, double x, bool euler, out long exponent)
        {
            double sign = 1;
            if (x > 0.5)
            {
                x = 1 - x;
                sign = n % 2 == 0 ? 1 : -1;
            }

            // 中心 c ∈ {0, 1/2} とずれ t = x - c（丸め誤差なし）
            double center = Math.Abs(x) <= 0.25 ? 0 : 0.5;
            double t = x - center;
            if (n < AppellFourierDegree)
            {
                exponent = 0;
                return sign * AppellPolynomialTaylor(n, center, t, euler);
            }

            double value = AppellPolynomialFourier(n, center, t, euler, out exponent);
            if (x < 0)
            {
                double correction = euler ? 2 * Math.Pow(x, n) : -n * Math.Pow(x, n - 1);
                value += ScaleB(correction, -exponent);
            }

            return sign * value;
        }

        // Pₙ(c + t) = Σⱼ C(n, j) P_(n-j)(c) tʲ (c ∈ {0, 1/2}, n < 12)
        private static double AppellPolynomialTaylor(int n, double center, double t, bool euler)
        {
            double sum = 0;
            for (int j = n; j >= 0; j--)
            {
                sum = sum * t + Binomial(n, j) * AppellValueAtCenter(n - j, center, euler);
            }

            return sum;
        }

        private static double AppellValueAtCenter(int m, double center, bool euler)
        {
            if (euler)
            {
                if (center == 0.5)
                {
                    // Eₘ(1/2) = Eₘ/2ᵐ
                    return Math.ScaleB(EulerNumber(m), -m);
                }

                // Eₘ(0) = -2 (2^(m+1) - 1) B_(m+1)/(m+1) (m ≥ 1)
                return m == 0 ? 1 : -2 * (Math.ScaleB(1.0, m + 1) - 1) * BernoulliNumber(m + 1) / (m + 1);
            }

            // Bₘ(1/2) = (2^(1-m) - 1) Bₘ
            double b = BernoulliNumber(m);
            return center == 0.5 ? (Math.ScaleB(1.0, 1 - m) - 1) * b : b;
        }

        // フーリエ級数展開。位相の中心 c の分を 1/4 周期の倍数として分離し、残り t のみ三角関数に渡す
        private static double AppellPolynomialFourier(int n, double center, double t, bool euler, out long exponent)
        {
            // 振幅 2n!/(2π)ⁿ または 4n!/π^(n+1)。丸めた π のべきの誤差 (1 + δ)ⁿ は 1 次で補正する
            double factorial = ScaledFactorial(n, out long factorialExponent);
            double amplitude;
            if (euler)
            {
                double power = ScaledPower(Math.PI, n + 1, out long powerExponent);
                amplitude = 4 * factorial / power * (1 - (n + 1) * (PiLo / Math.PI));
                exponent = factorialExponent - powerExponent;
            }
            else
            {
                double power = ScaledPower(2 * Math.PI, n, out long powerExponent);
                amplitude = -2 * factorial / power * (1 - n * (TwoPiLo / (2 * Math.PI)));
                exponent = factorialExponent - powerExponent;
            }

            // ベルヌーイ：cos(2πk(c + t) - nπ/2) = sin(2πkt + q π/2)、q = 4kc - n + 1
            // オイラー：sin((2k+1)π(c + t) - nπ/2) = sin((2k+1)πt + q π/2)、q = 2(2k+1)c - n
            double sum = 0;
            for (int k = euler ? 0 : 1; k < MaxIterations; k++)
            {
                double frequency = euler ? 2 * k + 1 : 2 * k;
                double weight = Math.Pow(euler ? 2 * k + 1 : k, -(euler ? n + 1 : n));
                int quarter = (int)(((long)(2 * frequency * center) - n + (euler ? 0 : 1)) & 3);
                double a = frequency * t;
                double s = quarter switch
                {
                    0 => double.SinPi(a),
                    1 => double.CosPi(a),
                    2 => -double.SinPi(a),
                    _ => -double.CosPi(a),
                };

                sum += weight * s;
                if (weight < 8.673617379884035e-19)
                {
                    break;
                }
            }

            return amplitude * sum;
        }

        // y ≥ n/2 での Pₙ(1/2 + y) = yⁿ Σₖ C(n, k) Pₖ(1/2) y⁻ᵏ (偶数 k のみ残る。項比はおよそ n²/(π²y²) ≤ 4/π²)
        private static double AppellPolynomialCentral(int n, double y, bool euler)
        {
            double sum = 1;
            double binomial = 1;
            double inverse = 1 / (y * y);
            double scale = 1;
            int limit = euler ? 2 * (CombinatorialTables.EulerEven.Length - 1) : int.MaxValue;
            for (int k = 2; k <= n && k <= limit; k += 2)
            {
                binomial *= (double)(n - k + 2) * (n - k + 1) / ((k - 1) * k);
                scale *= inverse;
                double term = binomial * scale * AppellValueAtCenter(k, 0.5, euler);
                sum += term;
                if (Math.Abs(term) <= 0.25 * Epsilon * Math.Abs(sum))
                {
                    break;
                }
            }

            return Math.Pow(y, n) * sum;
        }
    }
}
