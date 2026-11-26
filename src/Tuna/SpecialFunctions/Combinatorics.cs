using System.Globalization;
using System.Numerics;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // 正しく丸めた二重階乗 n!! のテーブル (n = 0..301; これ以降は +∞)
        private static readonly double[] DoubleFactorialTable = CreateDoubleFactorialTable();

        private static double[] CreateDoubleFactorialTable()
        {
            var table = new double[302];
            BigInteger even = BigInteger.One;
            BigInteger odd = BigInteger.One;
            for (int n = 0; n < table.Length; n++)
            {
                BigInteger value;
                if (n % 2 == 0)
                {
                    if (n > 0)
                    {
                        even *= n;
                    }

                    value = even;
                }
                else
                {
                    odd *= n;
                    value = odd;
                }

                table[n] = double.Parse(value.ToString(CultureInfo.InvariantCulture), CultureInfo.InvariantCulture);
            }

            return table;
        }

        /// <summary>
        /// Computes the factorial <c>n!</c>.
        /// <para>階乗 <c>n!</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The non-negative integer. / 非負整数。</param>
        /// <returns>The value of <c>n!</c>, <see cref="double.PositiveInfinity"/> for <c>n &gt; 170</c>, or <see cref="double.NaN"/> if <paramref name="n"/> is negative. / <c>n!</c> の値。<c>n &gt; 170</c> では <see cref="double.PositiveInfinity"/>、<paramref name="n"/> が負の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Factorial"/>
        public static double Factorial(int n)
        {
            if (n < 0)
            {
                return double.NaN;
            }

            return n < FactorialTable.Length ? FactorialTable[n] : double.PositiveInfinity;
        }

        /// <summary>
        /// Computes the natural logarithm of the factorial, <c>ln(n!)</c>.
        /// <para>階乗の自然対数 <c>ln(n!)</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The non-negative integer. / 非負整数。</param>
        /// <returns>The value of <c>ln(n!)</c>, or <see cref="double.NaN"/> if <paramref name="n"/> is negative. / <c>ln(n!)</c> の値。<paramref name="n"/> が負の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Factorial"/>
        public static double LogFactorial(int n)
        {
            if (n < 0)
            {
                return double.NaN;
            }

            return n < FactorialTable.Length ? Math.Log(FactorialTable[n]) : LogGamma(n + 1.0);
        }

        /// <summary>
        /// Computes the double factorial <c>n!! = n(n - 2)(n - 4)…</c>.
        /// <para>二重階乗 <c>n!! = n(n - 2)(n - 4)…</c> を計算します。</para>
        /// </summary>
        /// <param name="n">An integer greater than or equal to -1. / -1 以上の整数。</param>
        /// <returns>The value of <c>n!!</c> (with <c>(-1)!! = 0!! = 1</c>), <see cref="double.PositiveInfinity"/> on overflow, or <see cref="double.NaN"/> if <paramref name="n"/> is less than -1. / <c>n!!</c> の値（<c>(-1)!! = 0!! = 1</c>）。オーバーフロー時は <see cref="double.PositiveInfinity"/>、<paramref name="n"/> が -1 未満の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Double_factorial"/>
        public static double DoubleFactorial(int n)
        {
            if (n < -1)
            {
                return double.NaN;
            }

            if (n == -1)
            {
                return 1;
            }

            return n < DoubleFactorialTable.Length ? DoubleFactorialTable[n] : double.PositiveInfinity;
        }

        /// <summary>
        /// Computes the generalized binomial coefficient <c>C(n, k) = Γ(n + 1) / (Γ(k + 1) Γ(n - k + 1))</c>.
        /// <para>一般化二項係数 <c>C(n, k) = Γ(n + 1) / (Γ(k + 1) Γ(n - k + 1))</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The upper real argument. / 上側の実数引数。</param>
        /// <param name="k">The lower real argument. / 下側の実数引数。</param>
        /// <returns>The value of <c>C(n, k)</c>. / <c>C(n, k)</c> の値。</returns>
        /// <remarks>
        /// <para>For non-negative integer <paramref name="k"/> and any real <paramref name="n"/>, this equals <c>n(n - 1)…(n - k + 1) / k!</c>. For non-negative integer <paramref name="n"/> and integer <paramref name="k"/> outside <c>[0, n]</c>, the result is <c>0</c>.</para>
        /// <para><paramref name="k"/> が非負整数のときは任意の実数 <paramref name="n"/> に対して <c>n(n - 1)…(n - k + 1) / k!</c> となります。非負整数 <paramref name="n"/> に対して整数 <paramref name="k"/> が <c>[0, n]</c> の外にある場合は <c>0</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://fungrim.org/entry/e87c43/"/>
        public static double Binomial(double n, double k)
        {
            if (double.IsNaN(n) || double.IsNaN(k))
            {
                return double.NaN;
            }

            k = SmallIntegralComplement(n, k);
            if (IsInteger(k))
            {
                if (IsInteger(n) && n >= 0)
                {
                    if (k < 0 || k > n)
                    {
                        return 0;
                    }

                    if (n < 9.2e18 && TryExactBinomial((ulong)n, (ulong)k, out double exact))
                    {
                        return exact;
                    }

                    k = Math.Min(k, n - k);
                    if (n < FactorialTable.Length)
                    {
                        double value = FactorialTable[(int)n] / (FactorialTable[(int)k] * FactorialTable[(int)(n - k)]);
                        return value < 9007199254740992.0 ? Math.Round(value) : value;
                    }
                }
                else if (IsInteger(n))
                {
                    // 負の整数 n では Γ の比で極を打ち消せない (k > 20 で NaN) ため非負整数の二項係数へ移す
                    if (k < 0 && k > n)
                    {
                        return 0;
                    }

                    var (upper, lower) = NegatedUpperIndex(n, k);
                    double positive = Binomial(upper, lower);
                    return NegatedUpperIndexIsOdd(n, k) ? -positive : positive;
                }
                else if (k < 0)
                {
                    return 0;
                }

                if (k <= 20)
                {
                    // 中間積が非正規化数・∞ になっても精度を保つため指数を分離して掛け、最後に 1 回だけ戻す
                    double product = ScaledBinomialProduct(n, (int)k, out int exponent);
                    return Math.ScaleB(product, exponent);
                }
            }

            return RealBinomial(n, k);
        }

        // 負の整数 n に対する変換 C(n, k) = (-1)^p C(p + m, p) (m = -n - 1 ≥ 0; k ≥ 0 で p = k, k ≤ n で p = n - k)。
        // 小さい m が大きい k の丸めに吸収されないよう m の加算を 1 回にし、下側引数を min(p, m) とする
        private static (double Upper, double Lower) NegatedUpperIndex(double n, double k)
        {
            double m = -n - 1;
            double p = k >= 0 ? k : n - k;
            return (p + m, Math.Min(p, m));
        }

        // NegatedUpperIndex の符号 (-1)^p の偶奇 (p = n - k は丸めた差ではなく n と k の偶奇から判定)
        private static bool NegatedUpperIndexIsOdd(double n, double k) => k >= 0 ? IsOddInteger(k) : IsOddInteger(n) != IsOddInteger(k);

        // 実数引数の二項係数 (n, k 有限)。Γ 引数が小さければ直接積、それ以外は BinomialAsBeta または対数から計算する
        private static double RealBinomial(double n, double k)
        {
            if (!double.IsFinite(n) || !double.IsFinite(k))
            {
                return double.NaN;
            }

            var c = ShiftedDifference(n, k);
            if (IsNonPositiveInteger(c.Hi) && IsInteger(c.Lo))
            {
                // 1/Γ(n - k + 1) = 0
                return 0;
            }

            if (IsNonPositiveInteger(n + 1))
            {
                return double.NaN;
            }

            // n + 1, k + 1 も 2 の冪をまたぐと丸められるため誤差を保持する
            var a = DoubleDoubleTwoSum(n, 1);
            var b = DoubleDoubleTwoSum(k, 1);
            if (Math.Max(Math.Abs(a.Hi), Math.Max(Math.Abs(b.Hi), Math.Abs(c.Hi))) < 170)
            {
                // 対数を経由せず Γ を計算できる範囲は直接掛ける (ベータ形式では ln B に比例する誤差になりうる)
                double partial = GammaOfSum(a.Hi, a.Lo) * ReciprocalGammaOfSum(b.Hi, b.Lo);
                double direct = partial * ReciprocalGammaOfSum(c.Hi, c.Lo);
                if (double.IsNormal(partial) && double.IsNormal(direct))
                {
                    return direct;
                }
            }

            // ベータ値が非正規化数や 0 になる場合は精度低下や 1/0 = ∞ を避けて対数経路へフォールバックする
            var (factor, scale, p, q, reciprocal) = BinomialAsBeta(n, k, c.Hi);
            double beta = Beta(p, q);
            if (beta >= MinNormal)
            {
                double value = reciprocal ? factor / (scale * beta) : factor * beta;
                if (double.IsFinite(value) && Math.Abs(value) >= MinNormal)
                {
                    return value;
                }
            }

            // 係数も含めて対数で合成し最後に 1 回だけ指数へ戻す (途中のオーバーフロー回避)
            double log = Math.Log(Math.Abs(factor)) - Math.Log(scale) + (reciprocal ? -LogBeta(p, q) : LogBeta(p, q));
            return Math.Sign(factor) * Math.Exp(log);
        }

        // 実数引数の二項係数 Γ(a)/(Γ(b)Γ(c)) (a = n + 1, b = k + 1, c = n - k + 1) を
        // factor · B(p, q) または factor / (scale · B(p, q)) (Reciprocal = true; p, q, scale > 0) に分解する。
        // 負の Γ 引数は反射公式で正側へ移し、ベータ関数の差保持によって大引数での桁落ちを抑える
        private static (double Factor, double Scale, double P, double Q, bool Reciprocal) BinomialAsBeta(double n, double k, double c)
        {
            double a = n + 1;
            double b = k + 1;
            var (d, e) = DoubleDoubleTwoSum(n, -k);
            if (b > 0 && c > 0)
            {
                // b + c = a + 1 なので C = 1/(a B(c, b))（a < 0 なら b, c < 1）
                return (Math.Sign(a), Math.Abs(a), c, b, true);
            }

            if (a > 0)
            {
                // b, c の負の方 m を反射し、C = sin(πm)/π · B(a, 1 - m)（a + 1 - m が正の方）。sin(πb) = -sin(πk)、sin(πc) = -sin(π(n - k))
                return b < 0
                    ? (-double.SinPi(k) / Math.PI, 1, a, -k, false)
                    : (-SinPiOfSum(d, e) / Math.PI, 1, a, -d, false);
            }

            double sinPiN = double.SinPi(n);
            if (b > 0)
            {
                // a, c < 0：C = sin(π(n - k))/sin(πn) / ((k - n) B(k + 1, -n))（整数の k では係数は (-1)^k）
                double sign = IsInteger(k) ? (IsOddInteger(k) ? -1 : 1) : SinPiOfSum(d, e) / sinPiN;
                return (sign, -d, b, -n, true);
            }

            if (c > 0)
            {
                // a, b < 0：C = sin(πk)/sin(πn) / ((-k) B(-n, c))
                return (double.SinPi(k) / sinPiN, -k, -n, c, true);
            }

            // a, b, c < 0：C = -sin(πk) sin(π(n - k))/(π sin(πn)) · B(-k, k - n)
            return (-double.SinPi(k) * SinPiOfSum(d, e) / (Math.PI * sinPiN), 1, -k, -d, false);
        }

        // n - k + 1 を double-double で表す (丸めた差が 0 以下の整数になって 1/Γ = 0 と誤判定するのを防ぐ)
        private static (double Hi, double Lo) ShiftedDifference(double n, double k)
        {
            var (d, e) = DoubleDoubleTwoSum(n, -k);
            var (s, t) = DoubleDoubleTwoSum(d, 1);
            return DoubleDoubleTwoSum(s, t + e);
        }

        // sin(π(hi + lo)) (|lo| ≤ ulp(hi)/2; 整数近傍でも lo の寄与を保持)
        private static double SinPiOfSum(double hi, double lo) =>
            lo == 0 ? double.SinPi(hi) : double.SinPi(hi) * double.CosPi(lo) + double.CosPi(hi) * double.SinPi(lo);

        // Γ(hi + lo) ≈ Γ(hi)(1 + ψ(hi)lo) (|hi| < 170, |lo| ≤ ulp(hi)/2; hi < 0 なら lo = 0)
        private static double GammaOfSum(double hi, double lo) => lo == 0 ? Gamma(hi) : Gamma(hi) * (1 + Digamma(hi) * lo);

        // 1/Γ(hi + lo) (|hi| < 170, |lo| ≤ ulp(hi)/2; 負側では反射公式と 1 次補正を使用)
        private static double ReciprocalGammaOfSum(double hi, double lo)
        {
            if (lo == 0)
            {
                return ReciprocalGamma(hi);
            }

            if (hi > 0)
            {
                return ReciprocalGamma(hi) * (1 - Digamma(hi) * lo);
            }

            var (u, v) = DoubleDoubleTwoSum(1, -hi);
            double w = v - lo;
            return SinPiOfSum(hi, lo) * (Gamma(u) * (1 + Digamma(u) * w)) / Math.PI;
        }

        // 非負整数の二項係数を UInt128 で厳密に計算し最後に 1 回だけ丸める (階乗比の丸め誤差で 2⁵³ 以下でも隣の整数にずれるのを防ぐ)
        private static bool TryExactBinomial(ulong n, ulong k, out double value)
        {
            k = Math.Min(k, n - k);
            UInt128 c = 1;
            for (ulong i = 1; i <= k; i++)
            {
                // C(m, i) = C(m-1, i-1) m / i (m = n - k + i) は各段で割り切れる
                UInt128 m = n - k + i;
                if (c > UInt128.MaxValue / m)
                {
                    value = 0;
                    return false;
                }

                c = c * m / i;
            }

            value = (double)c;
            return true;
        }

        // 非整数の k で n - k が厳密に 0..20 の整数なら対称性 C(n, k) = C(n, n - k) で下側引数を差し替える (丸めで整数化しただけの差は除外)
        private static double SmallIntegralComplement(double n, double k)
        {
            var (complement, error) = DoubleDoubleTwoSum(n, -k);
            return !IsInteger(k) && error == 0 && IsInteger(complement) && complement >= 0 && complement <= 20 ? complement : k;
        }

        // k 項の積 n(n-1)…(n-k+1)/k! = product · 2^exponent (各因子の指数を分離して非正規化数・∞ でも相対精度を維持)
        private static double ScaledBinomialProduct(double n, int k, out int exponent)
        {
            double product = 1;
            exponent = 0;
            for (int i = 0; i < k; i++)
            {
                double factor = (n - i) / (i + 1);
                if (factor == 0)
                {
                    return 0;
                }

                int e = Math.ILogB(factor);
                product *= Math.ScaleB(factor, -e);
                exponent += e;
            }

            return product;
        }

        // k 項の積 n(n-1)…(n-k+1)/k! の絶対値の自然対数
        private static double LogBinomialProduct(double n, int k)
        {
            double product = ScaledBinomialProduct(n, k, out int exponent);
            return product == 0 ? double.NegativeInfinity : Math.Log(Math.Abs(product)) + exponent * Ln2;
        }

        /// <summary>
        /// Computes the natural logarithm of the absolute value of the generalized binomial coefficient, <c>ln |C(n, k)|</c>.
        /// <para>一般化二項係数の絶対値の自然対数 <c>ln |C(n, k)|</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The upper real argument. / 上側の実数引数。</param>
        /// <param name="k">The lower real argument. / 下側の実数引数。</param>
        /// <returns>The value of <c>ln |C(n, k)|</c>. / <c>ln |C(n, k)|</c> の値。</returns>
        /// <seealso href="https://fungrim.org/entry/e87c43/"/>
        public static double LogBinomial(double n, double k)
        {
            if (double.IsNaN(n) || double.IsNaN(k))
            {
                return double.NaN;
            }

            k = SmallIntegralComplement(n, k);
            if (IsInteger(k) && IsInteger(n) && n < 0)
            {
                // 負の整数 n では Binomial と同様に非負整数の二項係数へ移す (k < 0 は k ≤ n でなければ 0)
                if (k < 0 && k > n)
                {
                    return double.NegativeInfinity;
                }

                var (upper, lower) = NegatedUpperIndex(n, k);
                return LogBinomial(upper, lower);
            }

            if (IsInteger(k) && (k <= 20 || (IsInteger(n) && n >= 0)))
            {
                // 整数 k では Binomial の結果の対数を取る (k 項積が非正規化数・0・∞ になる場合は対数積で計算)
                double value = Binomial(n, k);
                if (k >= 0 && k <= 20 && !(double.IsFinite(value) && Math.Abs(value) >= MinNormal))
                {
                    return LogBinomialProduct(n, (int)k);
                }

                if (double.IsFinite(value))
                {
                    return Math.Log(Math.Abs(value));
                }
            }

            if (!double.IsFinite(n) || !double.IsFinite(k))
            {
                return double.IsFinite(n) && n + 1 > 0 && double.IsInfinity(k) ? double.NegativeInfinity : double.NaN;
            }

            var c = ShiftedDifference(n, k);
            if (IsNonPositiveInteger(c.Hi) && IsInteger(c.Lo))
            {
                return double.NegativeInfinity;
            }

            if (IsNonPositiveInteger(n + 1))
            {
                return double.NaN;
            }

            if (n + 1 < 0 || k + 1 < 0 || c.Hi < 0)
            {
                // 負の Γ 引数がある場合、Binomial が正規化数なら直接積の精度を生かすためその対数を取る
                double binomial = RealBinomial(n, k);
                if (double.IsFinite(binomial) && Math.Abs(binomial) >= MinNormal)
                {
                    return Math.Log(Math.Abs(binomial));
                }
            }

            // 正引数またはオーバーフロー・非正規化数領域はベータ関数形式の対数で直接計算する
            var (factor, scale, p, q, reciprocal) = BinomialAsBeta(n, k, c.Hi);
            return Math.Log(Math.Abs(factor)) - Math.Log(scale) + (reciprocal ? -LogBeta(p, q) : LogBeta(p, q));
        }

        /// <summary>
        /// Computes the <paramref name="n"/>-th Bernoulli number <c>Bₙ</c> using the convention <c>B₁ = -1/2</c>.
        /// <para>第 <paramref name="n"/> ベルヌーイ数 <c>Bₙ</c>（<c>B₁ = -1/2</c> の定義）を計算します。</para>
        /// </summary>
        /// <param name="n">The non-negative integer index. / 非負整数の番号。</param>
        /// <returns>The value of <c>Bₙ</c>, or <see cref="double.NaN"/> if <paramref name="n"/> is negative. / <c>Bₙ</c> の値。<paramref name="n"/> が負の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/24.2.E1"/>
        public static double BernoulliNumber(int n)
        {
            if (n < 0)
            {
                return double.NaN;
            }

            if (n == 1)
            {
                return -0.5;
            }

            if (n % 2 == 1)
            {
                return 0;
            }

            int k = n / 2;
            if (k < BernoulliEven.Length)
            {
                return BernoulliEven[k];
            }

            // B₂ₖ = (-1)^(k+1) 2 (2k)! ζ(2k) / (2π)^(2k)
            double sign = k % 2 == 1 ? 1 : -1;
            return sign * Math.Exp(Ln2 + LogGamma(n + 1.0) + Math.Log(Zeta(n)) - n * LnTwoPi);
        }

        /// <summary>
        /// Computes the harmonic number <c>Hₓ = ψ(x + 1) + γ</c>.
        /// <para>調和数 <c>Hₓ = ψ(x + 1) + γ</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The real input value. / 実数の入力値。</param>
        /// <returns>The value of <c>Hₓ</c>. / <c>Hₓ</c> の値。</returns>
        /// <remarks>
        /// <para>For positive integers <c>n</c>, this equals <c>1 + 1/2 + … + 1/n</c>.</para>
        /// <para>正の整数 <c>n</c> では <c>1 + 1/2 + … + 1/n</c> に一致します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Harmonic_number#Harmonic_numbers_for_real_and_complex_values"/>
        public static double HarmonicNumber(double x)
        {
            if (double.IsNaN(x))
            {
                return x;
            }

            if (IsInteger(x) && x >= 0 && x <= 64)
            {
                double sum = 0;
                for (int k = (int)x; k >= 1; k--)
                {
                    sum += 1.0 / k;
                }

                return sum;
            }

            if (Math.Abs(x) < 0.25)
            {
                // Hₓ = Σ (-1)ᵏ ζ(k) x^(k-1)（k ≥ 2）
                double sum = 0;
                for (int k = ZetaAtInteger.Length - 1; k >= 2; k--)
                {
                    sum = sum * -x + ZetaAtInteger[k];
                }

                return sum * x;
            }

            return Digamma(x + 1) + EulerGamma;
        }
    }
}
