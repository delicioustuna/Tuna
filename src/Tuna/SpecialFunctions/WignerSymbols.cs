using System.Numerics;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // Wigner 記号で受け付ける角運動量の 2 倍（2j）の上限（計算量は O(j²) ビット演算）。
        // 6j 記号の項比の分子（4 因子の積、≤ (2·10⁴ + 2)⁴）が long に収まる範囲でもある。
        private const int WignerMaxTwiceJ = 20000;

        // あらかじめ篩っておく素数の上限（これを超える階乗が現れた場合のみ呼び出しごとに篩う）
        private const int WignerSmallPrimeLimit = 4096;

        // WignerSmallPrimeLimit 以下の素数（他の静的フィールドに依存しない）
        private static readonly int[] WignerSmallPrimes = WignerSieve(WignerSmallPrimeLimit);

        /// <summary>
        /// Computes the Wigner 3-j symbol <c>(j₁ j₂ j₃; m₁ m₂ m₃)</c>.
        /// <para>Wigner の 3j 記号 <c>(j₁ j₂ j₃; m₁ m₂ m₃)</c> を計算します。</para>
        /// </summary>
        /// <param name="j1">The first angular momentum (<c>j₁ &gt;= 0</c>, integer or half-integer). / 第 1 の角運動量（<c>j₁ ≥ 0</c>、整数または半整数）。</param>
        /// <param name="j2">The second angular momentum (<c>j₂ &gt;= 0</c>, integer or half-integer). / 第 2 の角運動量（<c>j₂ ≥ 0</c>、整数または半整数）。</param>
        /// <param name="j3">The third angular momentum (<c>j₃ &gt;= 0</c>, integer or half-integer). / 第 3 の角運動量（<c>j₃ ≥ 0</c>、整数または半整数）。</param>
        /// <param name="m1">The first magnetic quantum number (integer or half-integer). / 第 1 の磁気量子数（整数または半整数）。</param>
        /// <param name="m2">The second magnetic quantum number (integer or half-integer). / 第 2 の磁気量子数（整数または半整数）。</param>
        /// <param name="m3">The third magnetic quantum number (integer or half-integer). / 第 3 の磁気量子数（整数または半整数）。</param>
        /// <returns>The value of the Wigner 3-j symbol, <c>0</c> if the selection rules are not satisfied, or <see cref="double.NaN"/> if any argument is invalid. / Wigner の 3j 記号の値。選択則を満たさない場合は <c>0</c>、引数が無効な場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>
        /// Returns <c>0</c> unless all selection rules hold (<c>m₁ + m₂ + m₃ = 0</c>, <c>|mᵢ| &lt;= jᵢ</c>, <c>jᵢ - mᵢ</c> integer,
        /// triangle condition <c>|j₁ - j₂| &lt;= j₃ &lt;= j₁ + j₂</c>, and <c>j₁ + j₂ + j₃</c> integer).
        /// Returns <see cref="double.NaN"/> if any argument is <see cref="double.NaN"/>, infinite, or neither an integer nor a half-integer,
        /// if any <c>jᵢ &lt; 0</c>, or if <c>2|jᵢ|</c> or <c>2|mᵢ|</c> exceeds 20000.
        /// </para>
        /// <para>
        /// 選択則（<c>m₁ + m₂ + m₃ = 0</c>、<c>|mᵢ| ≤ jᵢ</c>、<c>jᵢ - mᵢ</c> が整数、三角条件 <c>|j₁ - j₂| ≤ j₃ ≤ j₁ + j₂</c>、<c>j₁ + j₂ + j₃</c> が整数）を満たさない場合は <c>0</c> を返します。
        /// いずれかの引数が <see cref="double.NaN"/>、無限大、または整数でも半整数でもない値の場合、<c>jᵢ &lt; 0</c> の場合、あるいは <c>2|jᵢ|</c> もしくは <c>2|mᵢ|</c> が 20000 を超える場合は <see cref="double.NaN"/> を返します。
        /// </para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/34.2"/>
        public static double Wigner3j(double j1, double j2, double j3, double m1, double m2, double m3)
        {
            if (!WignerTryTwice(j1, out int tj1) || !WignerTryTwice(j2, out int tj2) || !WignerTryTwice(j3, out int tj3)
                || !WignerTryTwice(m1, out int tm1) || !WignerTryTwice(m2, out int tm2) || !WignerTryTwice(m3, out int tm3)
                || tj1 < 0 || tj2 < 0 || tj3 < 0)
            {
                return double.NaN;
            }

            if (tm1 + tm2 + tm3 != 0)
            {
                return 0;
            }

            return Wigner3jCore(tj1, tj2, tj3, tm1, tm2, tm3, 1);
        }

        /// <summary>
        /// Computes the Clebsch–Gordan coefficient <c>⟨j₁ m₁ j₂ m₂ | j m⟩</c>.
        /// <para>Clebsch–Gordan 係数 <c>⟨j₁ m₁ j₂ m₂ | j m⟩</c> を計算します。</para>
        /// </summary>
        /// <param name="j1">The first angular momentum (<c>j₁ &gt;= 0</c>, integer or half-integer). / 第 1 の角運動量（<c>j₁ ≥ 0</c>、整数または半整数）。</param>
        /// <param name="m1">The first magnetic quantum number (integer or half-integer). / 第 1 の磁気量子数（整数または半整数）。</param>
        /// <param name="j2">The second angular momentum (<c>j₂ &gt;= 0</c>, integer or half-integer). / 第 2 の角運動量（<c>j₂ ≥ 0</c>、整数または半整数）。</param>
        /// <param name="m2">The second magnetic quantum number (integer or half-integer). / 第 2 の磁気量子数（整数または半整数）。</param>
        /// <param name="j">The coupled angular momentum (<c>j &gt;= 0</c>, integer or half-integer). / 合成角運動量（<c>j ≥ 0</c>、整数または半整数）。</param>
        /// <param name="m">The coupled magnetic quantum number (integer or half-integer). / 合成磁気量子数（整数または半整数）。</param>
        /// <returns>The Clebsch–Gordan coefficient, <c>0</c> if the selection rules are not satisfied, or <see cref="double.NaN"/> if any argument is invalid. / Clebsch–Gordan 係数の値。選択則を満たさない場合は <c>0</c>、引数が無効な場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>
        /// Follows the Condon–Shortley phase convention: <c>⟨j₁ m₁ j₂ m₂ | j m⟩ = (-1)^(j₁ - j₂ + m) √(2j + 1) (j₁ j₂ j; m₁ m₂ -m)</c>.
        /// Returns <c>0</c> when <c>m₁ + m₂ != m</c> or any selection rule of <see cref="Wigner3j"/> fails.
        /// Returns <see cref="double.NaN"/> if any argument is <see cref="double.NaN"/>, infinite, or neither an integer nor a half-integer,
        /// if any angular momentum is negative, or if twice any argument's magnitude exceeds 20000.
        /// </para>
        /// <para>
        /// Condon–Shortley の位相定義 <c>⟨j₁ m₁ j₂ m₂ | j m⟩ = (-1)^(j₁-j₂+m) √(2j+1) (j₁ j₂ j; m₁ m₂ -m)</c> に従います。
        /// <c>m₁ + m₂ ≠ m</c> または <see cref="Wigner3j"/> の選択則を満たさない場合は <c>0</c> を返します。
        /// いずれかの引数が <see cref="double.NaN"/>、無限大、または整数でも半整数でもない値の場合、角運動量が負の場合、あるいは引数の絶対値の 2 倍が 20000 を超える場合は <see cref="double.NaN"/> を返します。
        /// </para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/34.1"/>
        public static double ClebschGordan(double j1, double m1, double j2, double m2, double j, double m)
        {
            if (!WignerTryTwice(j1, out int tj1) || !WignerTryTwice(j2, out int tj2) || !WignerTryTwice(j, out int tj)
                || !WignerTryTwice(m1, out int tm1) || !WignerTryTwice(m2, out int tm2) || !WignerTryTwice(m, out int tm)
                || tj1 < 0 || tj2 < 0 || tj < 0)
            {
                return double.NaN;
            }

            if (tm1 + tm2 != tm)
            {
                return 0;
            }

            return Wigner3jCore(tj1, tj2, tj, tm1, tm2, -tm, tj + 1);
        }

        /// <summary>
        /// Computes the Wigner 6-j symbol <c>{j₁ j₂ j₃; j₄ j₅ j₆}</c>.
        /// <para>Wigner の 6j 記号 <c>{j₁ j₂ j₃; j₄ j₅ j₆}</c> を計算します。</para>
        /// </summary>
        /// <param name="j1">The angular momentum <c>j₁ &gt;= 0</c> (integer or half-integer). / 角運動量 <c>j₁ ≥ 0</c>（整数または半整数）。</param>
        /// <param name="j2">The angular momentum <c>j₂ &gt;= 0</c> (integer or half-integer). / 角運動量 <c>j₂ ≥ 0</c>（整数または半整数）。</param>
        /// <param name="j3">The angular momentum <c>j₃ &gt;= 0</c> (integer or half-integer). / 角運動量 <c>j₃ ≥ 0</c>（整数または半整数）。</param>
        /// <param name="j4">The angular momentum <c>j₄ &gt;= 0</c> (integer or half-integer). / 角運動量 <c>j₄ ≥ 0</c>（整数または半整数）。</param>
        /// <param name="j5">The angular momentum <c>j₅ &gt;= 0</c> (integer or half-integer). / 角運動量 <c>j₅ ≥ 0</c>（整数または半整数）。</param>
        /// <param name="j6">The angular momentum <c>j₆ &gt;= 0</c> (integer or half-integer). / 角運動量 <c>j₆ ≥ 0</c>（整数または半整数）。</param>
        /// <returns>The value of the Wigner 6-j symbol, <c>0</c> if any triangle condition fails, or <see cref="double.NaN"/> if any argument is invalid. / Wigner の 6j 記号の値。三角条件を満たさない場合は <c>0</c>、引数が無効な場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>
        /// Returns <c>0</c> unless all four triads <c>(j₁ j₂ j₃)</c>, <c>(j₁ j₅ j₆)</c>, <c>(j₄ j₂ j₆)</c>, and <c>(j₄ j₅ j₃)</c>
        /// satisfy the triangle condition (integer sum and <c>|a - b| &lt;= c &lt;= a + b</c>).
        /// Returns <see cref="double.NaN"/> if any argument is <see cref="double.NaN"/>, infinite, negative,
        /// neither an integer nor a half-integer, or if <c>2 jᵢ</c> exceeds 20000.
        /// </para>
        /// <para>
        /// 4 つの三つ組 <c>(j₁ j₂ j₃)</c>、<c>(j₁ j₅ j₆)</c>、<c>(j₄ j₂ j₆)</c>、<c>(j₄ j₅ j₃)</c> のいずれかが三角条件（和が整数かつ <c>|a - b| ≤ c ≤ a + b</c>）を満たさない場合は <c>0</c> を返します。
        /// いずれかの引数が <see cref="double.NaN"/>、無限大、負、または整数でも半整数でもない値の場合、あるいは <c>2 jᵢ</c> が 20000 を超える場合は <see cref="double.NaN"/> を返します。
        /// </para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/34.4"/>
        public static double Wigner6j(double j1, double j2, double j3, double j4, double j5, double j6)
        {
            if (!WignerTryTwice(j1, out int t1) || !WignerTryTwice(j2, out int t2) || !WignerTryTwice(j3, out int t3)
                || !WignerTryTwice(j4, out int t4) || !WignerTryTwice(j5, out int t5) || !WignerTryTwice(j6, out int t6)
                || t1 < 0 || t2 < 0 || t3 < 0 || t4 < 0 || t5 < 0 || t6 < 0)
            {
                return double.NaN;
            }

            if (!WignerTriangle(t1, t2, t3) || !WignerTriangle(t1, t5, t6) || !WignerTriangle(t4, t2, t6) || !WignerTriangle(t4, t5, t3))
            {
                return 0;
            }

            // αᵢ（三つ組の和）と βⱼ（向かい合う2列の和）。三角条件から全て整数で、すべての βⱼ - αᵢ は非負
            Span<long> alpha = [(t1 + t2 + t3) / 2, (t1 + t5 + t6) / 2, (t4 + t2 + t6) / 2, (t4 + t5 + t3) / 2];
            long b1 = (t1 + t2 + t4 + t5) / 2;
            long b2 = (t2 + t3 + t5 + t6) / 2;
            long b3 = (t3 + t1 + t6 + t4) / 2;

            // 最大の α を α₁ に置く（C(t+1, α₁+1) = C(t+1, t-α₁) の t - α₁ が最小になり、項の桁数が減る）。
            // 残りの α と β の組み合わせは任意でよい（どの βⱼ - αᵢ も三角条件のどれかの量で非負）
            int maxIndex = 0;
            for (int i = 1; i < 4; i++)
            {
                if (alpha[i] > alpha[maxIndex])
                {
                    maxIndex = i;
                }
            }

            (alpha[0], alpha[maxIndex]) = (alpha[maxIndex], alpha[0]);
            long a1 = alpha[0];
            long a2 = alpha[1];
            long a3 = alpha[2];
            long a4 = alpha[3];
            long tMin = a1;
            long tMax = Math.Min(b1, Math.Min(b2, b3));
            if (tMin > tMax)
            {
                return 0;
            }

            // 前因子の平方 R = Π Δ² ((α₁+1)!)²/((β₁-α₂)!(β₂-α₃)!(β₃-α₄)!)²
            Span<int> numerator = stackalloc int[14];
            Span<int> denominator = stackalloc int[10];
            int nn = 0;
            int nd = 0;
            WignerAddTriangleFactorials(t1, t2, t3, numerator, ref nn, denominator, ref nd);
            WignerAddTriangleFactorials(t1, t5, t6, numerator, ref nn, denominator, ref nd);
            WignerAddTriangleFactorials(t4, t2, t6, numerator, ref nn, denominator, ref nd);
            WignerAddTriangleFactorials(t4, t5, t3, numerator, ref nn, denominator, ref nd);
            numerator[nn++] = (int)(a1 + 1);
            numerator[nn++] = (int)(a1 + 1);
            denominator[nd++] = (int)(b1 - a2);
            denominator[nd++] = (int)(b1 - a2);
            denominator[nd++] = (int)(b2 - a3);
            denominator[nd++] = (int)(b2 - a3);
            denominator[nd++] = (int)(b3 - a4);
            denominator[nd++] = (int)(b3 - a4);

            // 項 T_t = C(t+1, α₁+1) C(β₁-α₂, t-α₂) C(β₂-α₃, t-α₃) C(β₃-α₄, t-α₄) の大きさの上界（ビット数）と、
            // 項比の分子 (t+2)(β₁-t)(β₂-t)(β₃-t) を掛けたときの余裕
            double bits = Math.Min(tMax + 1, (tMax - a1) * Math.Log2(tMax + 1)) + (b1 - a2) + (b2 - a3) + (b3 - a4)
                + 4 * Math.Log2(tMax + 2) + Math.Log2(tMax - tMin + 1) + 2;
            double sumHi;
            double sumLo;
            long sumExponent = 0;
            if (bits <= 124)
            {
                Int128 term = WignerBinomial128(tMin + 1, a1 + 1) * WignerBinomial128(b1 - a2, tMin - a2)
                    * WignerBinomial128(b2 - a3, tMin - a3) * WignerBinomial128(b3 - a4, tMin - a4);
                Int128 sum = 0;
                for (long t = tMin; ; t++)
                {
                    sum += (t & 1) == 0 ? term : -term;
                    if (t == tMax)
                    {
                        break;
                    }

                    // T_(t+1)/T_t は有理数だが T_(t+1) は整数なので、先に掛けてから割れば割り切れる
                    term = term * ((t + 2) * (b1 - t) * (b2 - t) * (b3 - t))
                        / ((t + 1 - a1) * (t + 1 - a2) * (t + 1 - a3) * (t + 1 - a4));
                }

                (sumHi, sumLo) = WignerToDoubleDouble(sum);
            }
            else
            {
                var big = new WignerBigSum((int)Math.Ceiling(bits));
                big.MultiplyBinomial(tMin + 1, a1 + 1);
                big.MultiplyBinomial(b1 - a2, tMin - a2);
                big.MultiplyBinomial(b2 - a3, tMin - a3);
                big.MultiplyBinomial(b3 - a4, tMin - a4);
                for (long t = tMin; ; t++)
                {
                    big.Accumulate((t & 1) != 0);
                    if (t == tMax)
                    {
                        break;
                    }

                    big.Multiply((ulong)((t + 2) * (b1 - t) * (b2 - t) * (b3 - t)));
                    big.DivideExact((ulong)((t + 1 - a1) * (t + 1 - a2) * (t + 1 - a3) * (t + 1 - a4)));
                }

                (sumHi, sumLo) = big.Result(out sumExponent);
            }

            return WignerCombine(sumHi, sumLo, sumExponent, numerator[..nn], denominator[..nd], 1, (int)(a1 + 1));
        }

        // 3j 記号（引数は 2 倍の整数で、m₁ + m₂ + m₃ = 0 は確認済み）。
        // 3j 記号の位相 (-1)^(j₁-j₂-m₃) は extraFactor = 1 のときだけ掛ける（CG 係数では Condon–Shortley の位相と打ち消し合う）。
        // extraFactor = 2j₃+1 のときは Clebsch–Gordan 係数（√(2j₃+1) 倍、符号なし位相）を計算する。
        private static double Wigner3jCore(int tj1, int tj2, int tj3, int tm1, int tm2, int tm3, int extraFactor)
        {
            if (Math.Abs(tm1) > tj1 || Math.Abs(tm2) > tj2 || Math.Abs(tm3) > tj3
                || ((tj1 - tm1) & 1) != 0 || ((tj2 - tm2) & 1) != 0 || ((tj3 - tm3) & 1) != 0
                || !WignerTriangle(tj1, tj2, tj3))
            {
                return 0;
            }

            long n1 = (tj1 + tj2 - tj3) / 2;
            long n2 = (tj1 - tj2 + tj3) / 2;
            long n3 = (-tj1 + tj2 + tj3) / 2;
            long bigJ = n1 + n2 + n3;
            long alpha = (tj1 - tm1) / 2;
            long beta = (tj2 + tm2) / 2;
            long kMin = Math.Max(0, Math.Max(alpha - n2, beta - n3));
            long kMax = Math.Min(n1, Math.Min(alpha, beta));
            if (kMin > kMax)
            {
                return 0;
            }

            // R = Π(jᵢ ± mᵢ)!/(n₁! n₂! n₃! (J+1)!)
            Span<int> numerator =
            [
                (tj1 + tm1) / 2, (tj1 - tm1) / 2, (tj2 + tm2) / 2, (tj2 - tm2) / 2, (tj3 + tm3) / 2, (tj3 - tm3) / 2,
            ];
            Span<int> denominator = [(int)n1, (int)n2, (int)n3, (int)(bigJ + 1)];

            // 項 T_k = C(n₁, k) C(n₂, α-k) C(n₃, β-k) ≤ 2ᴶ と、項比の分子 (n₁-k)(α-k)(β-k) を掛けたときの余裕
            double bits = bigJ + 3 * Math.Log2(bigJ + 2) + 2;
            double sumHi;
            double sumLo;
            long sumExponent = 0;
            if (bits <= 124)
            {
                Int128 term = WignerBinomial128(n1, kMin) * WignerBinomial128(n2, alpha - kMin) * WignerBinomial128(n3, beta - kMin);
                Int128 sum = 0;
                for (long k = kMin; ; k++)
                {
                    sum += (k & 1) == 0 ? term : -term;
                    if (k == kMax)
                    {
                        break;
                    }

                    // T_(k+1) は整数なので、項比の分子を先に掛ければ分母で割り切れる
                    term = term * ((n1 - k) * (alpha - k) * (beta - k)) / ((k + 1) * (n2 - alpha + k + 1) * (n3 - beta + k + 1));
                }

                (sumHi, sumLo) = WignerToDoubleDouble(sum);
            }
            else
            {
                var big = new WignerBigSum((int)Math.Ceiling(bits));
                big.MultiplyBinomial(n1, kMin);
                big.MultiplyBinomial(n2, alpha - kMin);
                big.MultiplyBinomial(n3, beta - kMin);
                for (long k = kMin; ; k++)
                {
                    big.Accumulate((k & 1) != 0);
                    if (k == kMax)
                    {
                        break;
                    }

                    big.Multiply((ulong)((n1 - k) * (alpha - k) * (beta - k)));
                    big.DivideExact((ulong)((k + 1) * (n2 - alpha + k + 1) * (n3 - beta + k + 1)));
                }

                (sumHi, sumLo) = big.Result(out sumExponent);
            }

            if (extraFactor == 1 && (((tj1 - tj2 - tm3) / 2) & 1) != 0)
            {
                sumHi = -sumHi;
                sumLo = -sumLo;
            }

            return WignerCombine(sumHi, sumLo, sumExponent, numerator, denominator, extraFactor, (int)(bigJ + 1));
        }

        // sum · 2^sumExponent · √(extraFactor · Π numerator! / Π denominator!) を倍精度に丸める。
        // 階乗の比は素因数ごとの指数に直して約分し、(A/B)² C/D（C, D は平方因子なし）として A/B · √(C/D) を double-double で求める。
        private static double WignerCombine(double sumHi, double sumLo, long sumExponent, ReadOnlySpan<int> numerator,
            ReadOnlySpan<int> denominator, int extraFactor, int maxArgument)
        {
            if (sumHi == 0)
            {
                return 0;
            }

            int[] primes = maxArgument <= WignerSmallPrimeLimit ? WignerSmallPrimes : WignerSieve(maxArgument);
            int primeCount = 0;
            while (primeCount < primes.Length && primes[primeCount] <= maxArgument)
            {
                primeCount++;
            }

            int[] exponentsArray = primeCount <= 1024 ? [] : new int[primeCount];
            Span<int> exponents = primeCount <= 1024 ? stackalloc int[primeCount] : exponentsArray;
            exponents.Clear();
            foreach (int n in numerator)
            {
                WignerAddLegendre(exponents, primes, n, 1);
            }

            foreach (int n in denominator)
            {
                WignerAddLegendre(exponents, primes, n, -1);
            }

            // extraFactor（2j+1）の素因数分解
            int rest = extraFactor;
            for (int i = 0; i < primeCount && rest > 1; i++)
            {
                while (rest % primes[i] == 0)
                {
                    exponents[i]++;
                    rest /= primes[i];
                }
            }

            // A = Π_(e>0) p^⌊e/2⌋、C = Π_(e>0 奇数) p、B と D は負の指数について同様。各因子は double-double の仮数と 2 の指数で持つ
            double aHi = 1, aLo = 0, bHi = 1, bLo = 0, cHi = 1, cLo = 0, dHi = 1, dLo = 0;
            long aExp = 0, bExp = 0, cExp = 0, dExp = 0;
            for (int i = 0; i < primeCount; i++)
            {
                int e = exponents[i];
                if (e > 0)
                {
                    WignerMultiplyPower(ref aHi, ref aLo, ref aExp, primes[i], e / 2);
                    WignerMultiplyPower(ref cHi, ref cLo, ref cExp, primes[i], e & 1);
                }
                else if (e < 0)
                {
                    WignerMultiplyPower(ref bHi, ref bLo, ref bExp, primes[i], -e / 2);
                    WignerMultiplyPower(ref dHi, ref dLo, ref dExp, primes[i], -e & 1);
                }
            }

            var ratio = DoubleDoubleDivide(DoubleDoubleMultiply((sumHi, sumLo), (aHi, aLo)), (bHi, bLo));
            long exponent = sumExponent + aExp - bExp;

            // √(C/D)：2 の指数を偶数にそろえてから平方根をとる
            var quotient = DoubleDoubleDivide((cHi, cLo), (dHi, dLo));
            long rootExponent = cExp - dExp;
            if ((rootExponent & 1) != 0)
            {
                quotient = (2 * quotient.Hi, 2 * quotient.Lo);
                rootExponent--;
            }

            var root = DoubleDoubleSqrt(quotient);
            var value = DoubleDoubleMultiply(ratio, root);
            return ScaleB(value.Hi + value.Lo, exponent + rootExponent / 2);
        }

        // double-double の仮数 (hi, lo)·2^exponent に素数べき p^count を掛ける
        private static void WignerMultiplyPower(ref double hi, ref double lo, ref long exponent, int p, int count)
        {
            while (count > 0)
            {
                // 2⁵³ 以下の整数に p をまとめてから掛ける（まとめた値は正確に表せる）
                double chunk = p;
                count--;
                while (count > 0 && chunk * p <= 9007199254740992.0)
                {
                    chunk *= p;
                    count--;
                }

                (hi, lo) = DoubleDoubleMultiply((hi, lo), (chunk, 0));
                int shift = Math.ILogB(hi);
                if (shift > 512)
                {
                    // 2 のべきでのスケーリングは誤差を生じない
                    hi = Math.ScaleB(hi, -shift);
                    lo = Math.ScaleB(lo, -shift);
                    exponent += shift;
                }
            }
        }

        // ルジャンドルの公式 Σ⌊n/pⁱ⌋ により n! の素因数 p の指数を sign 倍して加える
        private static void WignerAddLegendre(Span<int> exponents, int[] primes, int n, int sign)
        {
            for (int i = 0; i < exponents.Length; i++)
            {
                int p = primes[i];
                if (p > n)
                {
                    break;
                }

                int e = 0;
                for (long q = n / p; q > 0; q /= p)
                {
                    e += (int)q;
                }

                exponents[i] += sign * e;
            }
        }

        // 三つ組 (a b c)（2 倍の値）の Δ² の階乗 (a+b-c)!(a-b+c)!(-a+b+c)!/(a+b+c+1)! を加える
        private static void WignerAddTriangleFactorials(int ta, int tb, int tc, Span<int> numerator, ref int nn,
            Span<int> denominator, ref int nd)
        {
            numerator[nn++] = (ta + tb - tc) / 2;
            numerator[nn++] = (ta - tb + tc) / 2;
            numerator[nn++] = (-ta + tb + tc) / 2;
            denominator[nd++] = (ta + tb + tc) / 2 + 1;
        }

        // 三角条件（2 倍の値で、和が偶数かつ |a - b| ≤ c ≤ a + b）
        private static bool WignerTriangle(int ta, int tb, int tc) =>
            ((ta + tb + tc) & 1) == 0 && tc >= Math.Abs(ta - tb) && tc <= ta + tb;

        // 引数の 2 倍が絶対値 WignerMaxTwiceJ 以下の整数なら、その値を返す
        private static bool WignerTryTwice(double x, out int twice)
        {
            double t = 2 * x;
            if (!(Math.Abs(t) <= WignerMaxTwiceJ) || t != Math.Floor(t))
            {
                twice = 0;
                return false;
            }

            twice = (int)t;
            return true;
        }

        // 二項係数 C(n, k)（Int128 に収まることは呼び出し側で確認済み）
        private static Int128 WignerBinomial128(long n, long k)
        {
            if (k < 0 || k > n)
            {
                return 0;
            }

            k = Math.Min(k, n - k);
            Int128 result = 1;
            for (long i = 1; i <= k; i++)
            {
                // result = C(n-k+i-1, i-1) なので、(n-k+i) を掛けると i で割り切れる
                result = result * (n - k + i) / i;
            }

            return result;
        }

        // Int128 を double-double に正確に変換する
        private static (double Hi, double Lo) WignerToDoubleDouble(Int128 value)
        {
            double hi = (double)value;
            double lo = (double)(value - (Int128)hi);
            return DoubleDoubleNormalize(hi, lo);
        }

        // エラトステネスの篩で limit 以下の素数を列挙する
        private static int[] WignerSieve(int limit)
        {
            var composite = new bool[limit + 1];
            var primes = new List<int>();
            for (int i = 2; i <= limit; i++)
            {
                if (composite[i])
                {
                    continue;
                }

                primes.Add(i);
                for (long j = (long)i * i; j <= limit; j += i)
                {
                    composite[j] = true;
                }
            }

            return [.. primes];
        }

        // Racah の交代和を厳密に計算する多倍長整数（64 ビット語の配列、下位語順）。
        // 項の更新は 64 ビット以下の整数乗算と割り切れる 64 ビット以下の整数除算のみのため、
        // 除算は除数の奇数部の mod 2⁶⁴ 逆数を掛ける Hensel の厳密除算（下位語から 1 パス、ハードウェア除算なし）と右シフトで行う。
        // 正の項と負の項は別々に累積し、最後に 1 回だけ減算する。
        private sealed class WignerBigSum
        {
            private readonly ulong[] term;
            private readonly ulong[] positive;
            private readonly ulong[] negative;
            private int termLength = 1;
            private int positiveLength;
            private int negativeLength;

            // bits は項、和、および項比の分子を掛けた中間値のビット数の上界
            public WignerBigSum(int bits)
            {
                int capacity = bits / 64 + 3;
                term = new ulong[capacity];
                positive = new ulong[capacity];
                negative = new ulong[capacity];
                term[0] = 1;
            }

            // 項に二項係数 C(n, k) を掛ける（(n-k+i) を掛けて i で割るたびに割り切れる）
            public void MultiplyBinomial(long n, long k)
            {
                k = Math.Min(k, n - k);
                for (long i = 1; i <= k; i++)
                {
                    Multiply((ulong)(n - k + i));
                    DivideExact((ulong)i);
                }
            }

            public void Multiply(ulong m)
            {
                // フィールドをループ内で読み直さないよう、局所変数（Span）に取り出してから回す
                Span<ulong> x = term.AsSpan(0, termLength);
                ulong carry = 0;
                for (int i = 0; i < x.Length; i++)
                {
                    ulong hi = Math.BigMul(x[i], m, out ulong lo);
                    lo += carry;
                    if (lo < carry)
                    {
                        hi++;
                    }

                    x[i] = lo;
                    carry = hi;
                }

                if (carry != 0)
                {
                    term[termLength++] = carry;
                }
            }

            // 割り切れることが分かっている d ≥ 1 で割る
            public void DivideExact(ulong d)
            {
                Span<ulong> x = term.AsSpan(0, termLength);
                int shift = BitOperations.TrailingZeroCount(d);
                d >>= shift;
                if (d != 1)
                {
                    // d⁻¹ mod 2⁶⁴ をニュートン法で求める（d·d ≡ 1 mod 8 から始め、1 回ごとに正しいビット数が倍になる）
                    ulong inverse = d;
                    for (int i = 0; i < 5; i++)
                    {
                        inverse *= 2 - d * inverse;
                    }

                    ulong borrow = 0;
                    for (int i = 0; i < x.Length; i++)
                    {
                        ulong xi = x[i];
                        ulong s = xi - borrow;
                        ulong carry = xi < borrow ? 1UL : 0UL;
                        ulong q = s * inverse;
                        x[i] = q;
                        borrow = Math.BigMul(q, d, out _) + carry;
                    }
                }

                if (shift > 0)
                {
                    for (int i = 0; i < x.Length - 1; i++)
                    {
                        x[i] = (x[i] >> shift) | (x[i + 1] << (64 - shift));
                    }

                    x[^1] >>= shift;
                }

                int length = x.Length;
                while (length > 1 && x[length - 1] == 0)
                {
                    length--;
                }

                termLength = length;
            }

            public void Accumulate(bool isNegative)
            {
                if (isNegative)
                {
                    negativeLength = Add(negative, negativeLength, term, termLength);
                }
                else
                {
                    positiveLength = Add(positive, positiveLength, term, termLength);
                }
            }

            // 和（正の和 - 負の和）を (hi, lo)·2^exponent に変換する（上位 3 語を double-double に加算するため相対誤差は 2⁻¹⁰⁴ 程度）
            public (double Hi, double Lo) Result(out long exponent)
            {
                int cmp = Compare(positive, positiveLength, negative, negativeLength);
                exponent = 0;
                if (cmp == 0)
                {
                    return (0, 0);
                }

                ulong[] large = cmp > 0 ? positive : negative;
                int length = Subtract(large, cmp > 0 ? positiveLength : negativeLength, cmp > 0 ? negative : positive,
                    cmp > 0 ? negativeLength : positiveLength);
                (double Hi, double Lo) value = (0, 0);
                int start = Math.Max(0, length - 3);
                for (int i = length - 1; i >= start; i--)
                {
                    // 64 ビット語は 32 ビットずつ正確に double に変換して加算する（2⁶⁴ 倍は正確）
                    value = (Math.ScaleB(value.Hi, 64), Math.ScaleB(value.Lo, 64));
                    value = DoubleDoubleAdd(value, (Math.ScaleB((double)(large[i] >> 32), 32), 0));
                    value = DoubleDoubleAdd(value, ((double)(large[i] & 0xFFFFFFFFUL), 0));
                }

                exponent = 64L * start;
                return cmp > 0 ? value : (-value.Hi, -value.Lo);
            }

            private static int Add(ulong[] accumulator, int length, ulong[] addend, int addendLength)
            {
                // 項と和のどちらが長いかは決まらないので、長い方に合わせて加算する
                int n = Math.Max(length, addendLength);
                Span<ulong> a = accumulator.AsSpan(0, n);
                ReadOnlySpan<ulong> b = addend.AsSpan(0, addendLength);
                if (length < n)
                {
                    a[length..].Clear();
                }

                ulong carry = 0;
                for (int i = 0; i < n; i++)
                {
                    ulong ai = a[i];
                    ulong s = ai + (i < b.Length ? b[i] : 0);
                    ulong c1 = s < ai ? 1UL : 0UL;
                    ulong s2 = s + carry;
                    ulong c2 = s2 < s ? 1UL : 0UL;
                    a[i] = s2;
                    carry = c1 + c2;
                }

                if (carry != 0)
                {
                    accumulator[n++] = carry;
                }

                return n;
            }

            // x -= y（x ≥ y）を計算し、結果の語数を返す
            private static int Subtract(ulong[] x, int xLength, ulong[] y, int yLength)
            {
                ulong borrow = 0;
                for (int i = 0; i < xLength; i++)
                {
                    ulong b = i < yLength ? y[i] : 0;
                    ulong d = x[i] - b;
                    ulong b1 = x[i] < b ? 1UL : 0UL;
                    ulong d2 = d - borrow;
                    ulong b2 = d < borrow ? 1UL : 0UL;
                    x[i] = d2;
                    borrow = b1 + b2;
                }

                while (xLength > 0 && x[xLength - 1] == 0)
                {
                    xLength--;
                }

                return xLength;
            }

            private static int Compare(ulong[] x, int xLength, ulong[] y, int yLength)
            {
                while (xLength > 0 && x[xLength - 1] == 0)
                {
                    xLength--;
                }

                while (yLength > 0 && y[yLength - 1] == 0)
                {
                    yLength--;
                }

                if (xLength != yLength)
                {
                    return xLength > yLength ? 1 : -1;
                }

                for (int i = xLength - 1; i >= 0; i--)
                {
                    if (x[i] != y[i])
                    {
                        return x[i] > y[i] ? 1 : -1;
                    }
                }

                return 0;
            }
        }
    }
}
