namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        /// <summary>
        /// Computes the Gauss hypergeometric function <c>₂F₁(a, b; c; x)</c>.
        /// <para>ガウスの超幾何関数 <c>₂F₁(a, b; c; x)</c> を計算します。</para>
        /// </summary>
        /// <param name="a">The first numerator parameter. / 第 1 分子パラメータ。</param>
        /// <param name="b">The second numerator parameter. / 第 2 分子パラメータ。</param>
        /// <param name="c">The denominator parameter. / 分母パラメータ。</param>
        /// <param name="x">The input value (<c>x ≤ 1</c>, or any finite real value when the series terminates as a polynomial). / 入力値（<c>x ≤ 1</c>、多項式となる場合は任意の有限実数）。</param>
        /// <returns>The value of <c>₂F₁(a, b; c; x)</c>, or <see cref="double.NaN"/> on the branch cut (<c>x &gt; 1</c> for non-polynomials), at poles, or for invalid arguments. / <c>₂F₁(a, b; c; x)</c> の値。分岐切断上（多項式を除く <c>x &gt; 1</c>）、極、または不正な引数の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>When <paramref name="a"/> or <paramref name="b"/> is a non-positive integer <c>-n</c>, the series terminates as a polynomial of degree <c>n</c> (defined even when <paramref name="c"/> is a non-positive integer provided <c>-c ≥ n</c>). At <c>x = 1</c>, returns Gauss's summation formula when <c>c - a - b &gt; 0</c> and a signed infinity otherwise.</para>
        /// <para><paramref name="a"/> または <paramref name="b"/> が 0 以下の整数 <c>-n</c> のときは <c>n</c> 次多項式となります（<paramref name="c"/> が 0 以下の整数でも <c>-c ≥ n</c> であれば定義されます）。<c>x = 1</c> では <c>c - a - b &gt; 0</c> のときガウスの和公式の値、それ以外は符号付き無限大を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/15.2.E1"/>
        public static double Hypergeometric2F1(double a, double b, double c, double x) =>
            Hypergeometric2F1Core(a, b, c, x, 1 - x);

        // ₂F₁(a, b; c; x) の本体（w には 1 - x を丸めずに渡せる）
        private static double Hypergeometric2F1Core(double a, double b, double c, double x, double w)
        {
            if (double.IsNaN(a) || double.IsNaN(b) || double.IsNaN(c) || double.IsNaN(x)
                || double.IsInfinity(a) || double.IsInfinity(b) || double.IsInfinity(c))
            {
                return double.NaN;
            }

            if (a == 0 || b == 0 || x == 0)
            {
                return 1;
            }

            // 多項式（a または b が 0 以下の整数）。c が極でも、その手前で打ち切られれば値が定まる
            bool aPolynomial = IsNonPositiveInteger(a);
            bool bPolynomial = IsNonPositiveInteger(b);
            if (aPolynomial || bPolynomial)
            {
                double degree = aPolynomial && bPolynomial ? Math.Min(-a, -b) : aPolynomial ? -a : -b;
                double other = aPolynomial && (!bPolynomial || -a <= -b) ? b : a;
                if ((IsNonPositiveInteger(c) && -c < degree) || degree > int.MaxValue || double.IsInfinity(x))
                {
                    return double.NaN;
                }

                return Hypergeometric2F1Polynomial((int)degree, other, c, x, w);
            }

            if (IsNonPositiveInteger(c) || x > 1 || double.IsPositiveInfinity(x))
            {
                return double.NaN;
            }

            if (double.IsNegativeInfinity(x))
            {
                // x → -∞ では ₂F₁ ~ C₁(-x)^(-a) + C₂(-x)^(-b)
                return a > 0 && b > 0 ? 0 : double.NaN;
            }

            // s = c - a - b を double-double で求める（ε = s - m の精度は接続公式の w^ε などに効く）
            var s = DoubleDoubleAdd(DoubleDoubleTwoSum(c, -a), (-b, 0));
            if (w == 0)
            {
                return Hypergeometric2F1AtOne(a, b, c, s.Hi + s.Lo);
            }

            // c - a または c - b が 0 以下の整数なら、オイラー変換で (1-x)^(c-a-b) × 多項式
            bool caPolynomial = IsNonPositiveInteger(c - a);
            bool cbPolynomial = IsNonPositiveInteger(c - b);
            if (caPolynomial || cbPolynomial)
            {
                bool useA = caPolynomial && (!cbPolynomial || a - c <= b - c);
                double degree = useA ? a - c : b - c;
                if (degree > int.MaxValue)
                {
                    return double.NaN;
                }

                double polynomial = Hypergeometric2F1Polynomial((int)degree, useA ? c - b : c - a, c, x, w);
                return polynomial == 0 ? 0 : Hypergeometric2F1Power(w, s) * polynomial;
            }

            if (a == c)
            {
                return Math.Pow(w, -b);
            }

            if (b == c)
            {
                return Math.Pow(w, -a);
            }

            if (x > 0.5)
            {
                return Hypergeometric2F1UpperHalf(a, b, c, x, w, s, out _);
            }

            if (x >= 0)
            {
                // 級数。桁落ちが大きければオイラー変換した級数と、x ≥ 1/4 なら接続公式とも比べる
                double best = Hypergeometric2F1Series(a, b, c, x, out double bestCondition);
                if (bestCondition <= 16)
                {
                    return best;
                }

                double euler = Hypergeometric2F1Series(c - a, c - b, c, x, out double condition);
                Hypergeometric2F1Choose(ref best, ref bestCondition, condition, euler, w, s);
                if (bestCondition > 16 && x >= 0.25)
                {
                    double near = Hypergeometric2F1NearOne(a, b, c, w, s, out condition);
                    Hypergeometric2F1Choose(ref best, ref bestCondition, condition, near);
                }

                return best;
            }

            // x < 0：パフ変換 ₂F₁(a, b; c; x) = (1-x)^(-a) ₂F₁(a, c-b; c; z) = (1-x)^(-b) ₂F₁(c-a, b; c; z)、z = x/(x-1)、1 - z = 1/(1-x)。
            // 変換後の c - a - (c - b) = b - a
            double zz = -x / w;
            var ba = DoubleDoubleTwoSum(b, -a);
            var ab = DoubleDoubleTwoSum(a, -b);
            if (x >= -1)
            {
                // z ∈ (0, 1/2] の級数の 2 通り。桁落ちが大きければ、|x| ≤ 1/2 なら元の級数と、x ≤ -1/3（1 - z ≤ 3/4）なら z の接続公式とも比べる
                double best = Math.Pow(w, -a) * Hypergeometric2F1Series(a, c - b, c, zz, out double bestCondition);
                if (bestCondition <= 16)
                {
                    return best;
                }

                double second = Hypergeometric2F1Series(c - a, b, c, zz, out double condition);
                Hypergeometric2F1Choose(ref best, ref bestCondition, condition, Math.Pow(w, -b) * second);
                if (bestCondition > 16 && x >= -0.5)
                {
                    double direct = Hypergeometric2F1Series(a, b, c, x, out condition);
                    Hypergeometric2F1Choose(ref best, ref bestCondition, condition, direct);
                }

                if (bestCondition > 16 && x <= -1.0 / 3)
                {
                    double near = Hypergeometric2F1NearOne(a, c - b, c, 1 / w, ba, out condition);
                    Hypergeometric2F1Choose(ref best, ref bestCondition, condition, Math.Pow(w, -a) * near);
                    if (bestCondition > 16)
                    {
                        near = Hypergeometric2F1NearOne(b, c - a, c, 1 / w, ab, out condition);
                        Hypergeometric2F1Choose(ref best, ref bestCondition, condition, Math.Pow(w, -b) * near);
                    }
                }

                return best;
            }

            // x < -1：z ∈ (1/2, 1)。桁落ちが大きければ a と b を入れ替えた変換とも比べる
            double first = Math.Pow(w, -a) * Hypergeometric2F1UpperHalf(a, c - b, c, zz, 1 / w, ba, out double firstCondition);
            if (firstCondition <= 16)
            {
                return first;
            }

            double swapped = Math.Pow(w, -b) * Hypergeometric2F1UpperHalf(b, c - a, c, zz, 1 / w, ab, out double swappedCondition);
            return swappedCondition < firstCondition ? swapped : first;
        }

        private static void Hypergeometric2F1Choose(ref double best, ref double bestCondition, double condition, double value)
        {
            if (condition < bestCondition && double.IsFinite(value))
            {
                best = value;
                bestCondition = condition;
            }
        }

        private static void Hypergeometric2F1Choose(ref double best, ref double bestCondition, double condition, double euler, double w, (double Hi, double Lo) s)
        {
            if (condition < bestCondition)
            {
                Hypergeometric2F1Choose(ref best, ref bestCondition, condition, Hypergeometric2F1Power(w, s) * euler);
            }
        }

        // 1/2 < x < 1（w = 1 - x, s = c - a - b）での評価。接続公式で打ち消しが大きく x ≤ 0.9 なら級数・オイラー変換とも比較する
        private static double Hypergeometric2F1UpperHalf(double a, double b, double c, double x, double w, (double Hi, double Lo) s, out double condition)
        {
            double near = Hypergeometric2F1NearOne(a, b, c, w, s, out condition);
            if (condition <= 16 || x > 0.9)
            {
                return near;
            }

            double direct = Hypergeometric2F1Series(a, b, c, x, out double directCondition);
            double euler = Hypergeometric2F1Series(c - a, c - b, c, x, out double eulerCondition);
            if (eulerCondition < directCondition)
            {
                direct = Hypergeometric2F1Power(w, s) * euler;
                directCondition = eulerCondition;
            }

            if (directCondition < condition)
            {
                condition = directCondition;
                return direct;
            }

            return near;
        }

        // double-double の指数 s に対する w^s（下位は 1 + s_lo ln w で補正）
        private static double Hypergeometric2F1Power(double w, (double Hi, double Lo) s)
        {
            double power = Math.Pow(w, s.Hi);
            return s.Lo == 0 ? power : power * (1 + s.Lo * Math.Log(w));
        }

        // x = 1 でのガウスの和公式（DLMF 15.4.20）。c - a - b ≤ 0 では発散の主要項の符号付き無限大を返す
        private static double Hypergeometric2F1AtOne(double a, double b, double c, double s)
        {
            if (s <= 0)
            {
                double sign = SignGamma(c) * SignGamma(a) * SignGamma(b) * (s < 0 ? SignGamma(-s) : 1);
                return double.IsNaN(sign) ? double.NaN : sign * double.PositiveInfinity;
            }

            if (IsNonPositiveInteger(c - a) || IsNonPositiveInteger(c - b))
            {
                return 0;
            }

            // Γ(c)Γ(s)/(Γ(c-a)Γ(c-b)) を仮数と指数に分けて掛ける（個々のガンマ関数はオーバーフローすることがある）
            double mantissa = ScaledGammaSigned(c, out long e1) * ScaledGammaSigned(s, out long e2)
                / (ScaledGammaSigned(c - a, out long e3) * ScaledGammaSigned(c - b, out long e4));
            return ScaleB(mantissa, e1 + e2 - e3 - e4);
        }

        // 級数 Σ (a)ₖ(b)ₖ/((c)ₖ k!) xᵏ（|x| ≤ 1/2 を想定）。condition は項の絶対値の和と和の比
        private static double Hypergeometric2F1Series(double a, double b, double c, double x, out double condition)
        {
            double sum = 1;
            double absoluteSum = 1;
            double term = 1;
            // 項の比がいったん 1 を超えて増える範囲（k が -a, -b, -c 付近より手前）では打ち切らない
            double hump = Math.Max(Math.Max(-a, -b), -c);
            for (int k = 0; k < MaxIterations; k++)
            {
                double ratio = (a + k) * (b + k) / ((c + k) * (k + 1)) * x;
                term *= ratio;
                sum += term;
                absoluteSum += Math.Abs(term);
                if (Math.Abs(term) <= 0.25 * Epsilon * Math.Abs(sum) && k > hump && Math.Abs(ratio) < 1)
                {
                    break;
                }

                if (term == 0 || !double.IsFinite(sum))
                {
                    break;
                }
            }

            condition = sum == 0 || !double.IsFinite(absoluteSum) ? double.PositiveInfinity : absoluteSum / Math.Abs(sum);
            return sum;
        }

        // 多項式 ₂F₁(-n, b; c; x)（w = 1 - x）。直接和・パフ変換・1 - x 変換・隣接関係式から桁落ちの小さいものを採用する
        private static double Hypergeometric2F1Polynomial(int n, double b, double c, double x, double w)
        {
            double best = Hypergeometric2F1PolynomialSum(n, b, c, x, out double bestCondition);
            if (bestCondition <= 4)
            {
                return best;
            }

            if (x < 0)
            {
                double sum = Hypergeometric2F1PolynomialSum(n, c - b, c, -x / w, out double condition);
                Hypergeometric2F1Choose(ref best, ref bestCondition, condition, Math.Pow(w, n) * sum);
            }

            double d = b - c - n + 1;
            if (x > 0.5 && bestCondition > 4 && (!IsNonPositiveInteger(d) || -d >= n))
            {
                double sum = Hypergeometric2F1PolynomialSum(n, b, d, w, out double condition);
                if (condition < bestCondition)
                {
                    // (c-b)ₙ/(c)ₙ を 1 因子ずつ掛ける（c は極にならない：c ≤ 0 の整数なら -c ≥ n）
                    double ratio = 1;
                    for (int j = 0; j < n && ratio != 0; j++)
                    {
                        ratio *= (c - b + j) / (c + j);
                    }

                    Hypergeometric2F1Choose(ref best, ref bestCondition, condition, ratio * sum);
                }
            }

            if (bestCondition > 16)
            {
                // 項が大きく打ち消し合う（ヤコビ多項式の振動域など）ときは、a についての隣接関係式（DLMF 15.5.11）による漸化式とも比べる。
                // 漸化式の誤差増幅率は、各段に相対 2⁻²⁰ の擬似乱数摂動を与えた結果との差で見積もる
                double recurrence = Hypergeometric2F1PolynomialRecurrence(n, b, c, x, w, 0);
                double perturbed = Hypergeometric2F1PolynomialRecurrence(n, b, c, x, w, 9.5367431640625e-7);
                double amplification = Math.Abs(perturbed - recurrence) / (Math.Abs(recurrence) * 9.5367431640625e-7) + 1;
                Hypergeometric2F1Choose(ref best, ref bestCondition, amplification, recurrence);
            }

            return best;
        }

        // a についての隣接関係式（DLMF 15.5.11）による多項式 ₂F₁(-n, b; c; x) の前進漸化式
        private static double Hypergeometric2F1PolynomialRecurrence(int n, double b, double c, double x, double w, double perturbation)
        {
            double previous = 1;
            double current = (1 - b * x / c) * (1 + perturbation);
            for (int k = 1; k < n; k++)
            {
                // ずらす向きは段ごとに擬似乱数で決める（同じ向きだと求める解そのものの変化と区別できない）
                double sign = ((uint)k * 2654435761u >> 31) == 0 ? 1 : -1;
                double next = ((2 * k + c - (b + k) * x) * current - k * w * previous) / (c + k) * (1 + sign * perturbation);
                previous = current;
                current = next;
            }

            return current;
        }

        // 有限和 Σₖ₌₀ⁿ (-n)ₖ(b)ₖ/((c)ₖ k!) xᵏ（condition は項の絶対値の和と和の比）
        private static double Hypergeometric2F1PolynomialSum(int n, double b, double c, double x, out double condition)
        {
            double sum = 1;
            double absoluteSum = 1;
            double term = 1;
            for (int k = 0; k < n; k++)
            {
                term *= (k - n) * (b + k) / ((c + k) * (k + 1.0)) * x;
                sum += term;
                absoluteSum += Math.Abs(term);
                if (term == 0)
                {
                    break;
                }
            }

            condition = sum == 0 || !double.IsFinite(absoluteSum) ? double.PositiveInfinity : absoluteSum / Math.Abs(sum);
            return sum;
        }

        // w = 1 - x ∈ (0, 1/2] への接続公式（s = c - a - b = m + ε, m ≥ 0 整数, |ε| ≤ 1/2）。
        // 整数付近での極の打ち消しを Eₙ = m!(fₙ - gₙ)/ε の形で解析的に処理して評価する
        private static double Hypergeometric2F1NearOne(double a, double b, double c, double w, (double Hi, double Lo) s, out double condition)
        {
            double m = Math.Round(s.Hi);
            if (m < 0)
            {
                // オイラー変換 ₂F₁(a, b; c; x) = w^s ₂F₁(c-a, c-b; c; x) で c - a - b の符号を反転する
                double value = Hypergeometric2F1NearOne(c - a, c - b, c, w, (-s.Hi, -s.Lo), out condition);
                return value == 0 ? 0 : Hypergeometric2F1Power(w, s) * value;
            }

            condition = double.PositiveInfinity;
            if (m > int.MaxValue / 2)
            {
                return double.NaN;
            }

            // s.Hi - m は丸められない（|s.Hi - m| ≤ 1/2 で m は s.Hi に最も近い整数）
            double eps = s.Hi - m + s.Lo;
            double lnW = Math.Log(w);
            double alpha = a + m;
            double beta = b + m;

            // Γ(x)/Γ(x+ε) = 1 + ε Q(x, ε) の Q を使うと、E₀ は ε で約した形に書ける：
            // m! f₀ = (1 + εQα)(1 + εQβ)(1 + εQ₁)、Q₁ = (1/Γ(1-ε) - 1)/ε、m! g₀ = (1 + εℓ)(1 + εQₘ)、ℓ = (w^ε - 1)/ε、Qₘ = Q(m+1, ε)
            double qa = Hypergeometric2F1GammaQuotient(alpha, eps);
            double qb = Hypergeometric2F1GammaQuotient(beta, eps);
            double qm = Hypergeometric2F1GammaQuotient(m + 1, eps);
            double q1 = -Hypergeometric2F1GammaQuotient(1, -eps);
            double ell = eps == 0 ? lnW : Expm1(eps * lnW) / eps;
            double e = qa + qb + q1 - ell - qm + eps * (qa * qb + qa * q1 + qb * q1 - ell * qm) + eps * eps * qa * qb * q1;
            double g = (1 + eps * ell) * (1 + eps * qm);

            // Eₙ₊₁ = Eₙ r₁ + gₙ (r₁ - r₂)/ε、gₙ₊₁ = gₙ r₂。r₁ = AB/((p-ε)q)、r₂ = (A+ε)(B+ε)/((q+ε)p)（A = α+n、B = β+n、p = n+1、q = m+n+1）。
            // r₁ - r₂ の分子は ε(AB(p+q) - (A+B)pq + εq(A+B-p) + qε²) と ε で約せる
            double tail = 0;
            double absoluteTail = 0;
            double power = 1;
            double hump = Math.Max(Math.Max(-alpha, -beta), 0);
            int small = 0;
            for (int n = 0; n < MaxIterations; n++)
            {
                double term = power * e;
                tail += term;
                absoluteTail += Math.Abs(term);
                if (Math.Abs(term) <= 0.25 * Epsilon * Math.Abs(tail) && n > hump)
                {
                    // Eₙ が偶然 0 に近いだけの場合を除くため、2 項続けて小さいことを確かめる
                    if (++small >= 2)
                    {
                        break;
                    }
                }
                else
                {
                    small = 0;
                }

                double aa = alpha + n;
                double bb = beta + n;
                double p = n + 1;
                double q = m + n + 1;
                double r1 = aa * bb / ((p - eps) * q);
                double r2 = (aa + eps) * (bb + eps) / ((q + eps) * p);
                double d = (aa * bb * (p + q) - (aa + bb) * p * q + eps * q * (aa + bb - p) + q * eps * eps) / ((p - eps) * q * (q + eps) * p);
                e = e * r1 + g * d;
                g *= r2;
                power *= w;
                if (power == 0 || !double.IsFinite(e))
                {
                    break;
                }
            }

            // (-1)ᵐ Γ(c) wᵐ/(Γ(a)Γ(b) m!) (πε/sin πε) を仮数と指数に分けて掛ける
            double sinc = eps == 0 ? 1 : Math.PI * eps / double.SinPi(eps);
            double mantissa = ScaledGammaSigned(c, out long ec) / (ScaledGammaSigned(a, out long ea) * ScaledGammaSigned(b, out long eb))
                * ScaledPower(w, (long)m, out long ew) / ScaledFactorial((int)m, out long ef) * sinc;
            if (m % 2 == 1)
            {
                mantissa = -mantissa;
            }

            double value2 = ScaleB(mantissa * tail, ec - ea - eb + ew - ef);
            double absolute = ScaleB(Math.Abs(mantissa) * absoluteTail, ec - ea - eb + ew - ef);
            if (m == 0)
            {
                condition = value2 == 0 ? double.PositiveInfinity : absolute / Math.Abs(value2);
                return value2;
            }

            // 有限和の部分。(1-m-ε)ₖ は k < m で 0 にならない（|j - m - ε| ≥ 1/2）
            double sum = 1;
            double absoluteSum = 1;
            double t = 1;
            for (int k = 0; k < m - 1; k++)
            {
                t *= (a + k) * (b + k) / ((1 - m - eps + k) * (k + 1)) * w;
                sum += t;
                absoluteSum += Math.Abs(t);
            }

            double prefactor = ScaledGammaSigned(c, out long fc) * ScaledGammaSigned(m + eps, out long fm)
                / (ScaledGammaSigned(c - a, out long fa) * ScaledGammaSigned(c - b, out long fb));
            double result = ScaleB(prefactor * sum, fc + fm - fa - fb) + value2;
            absolute += ScaleB(Math.Abs(prefactor) * absoluteSum, fc + fm - fa - fb);
            condition = result == 0 ? double.PositiveInfinity : absolute / Math.Abs(result);
            return result;
        }

        // Q(x, ε) = (Γ(x)/Γ(x+ε) - 1)/ε（|ε| ≤ 1/2、ε = 0 では極限 -ψ(x)）を桁落ちなく計算する。
        // x ≥ 10 はスターリング展開の差分、x < -20 は反転公式、それ以外は x + N ≥ 10 へのシフト積を用いる
        private static double Hypergeometric2F1GammaQuotient(double x, double eps)
        {
            if (x >= 10)
            {
                // D = (ln Γ(x+ε) - ln Γ(x))/ε = (x - 1/2) log1p(u)/ε + ln(x+ε) - 1 + (δ(x+ε) - δ(x))/ε、u = ε/x
                double u = eps / x;
                double l = Log1p(u);
                double lnRatio = u == 0 ? 1 : l / u;
                double dd = (x - 0.5) * lnRatio / x + Math.Log(x + eps) - 1;

                // δ(y) = Σ cₖ y^(1-2k)。((x+ε)^p - x^p)/ε = p x^(p-1) · (expm1(p l)/(p l)) · (l/u)（p = 1 - 2k）
                double inverse = 1 / x;
                double inverse2 = inverse * inverse;
                double power = inverse2;
                for (int k = 1; k <= StirlingSeriesCoefficients.Length; k++)
                {
                    int p = 1 - 2 * k;
                    double pl = p * l;
                    double expRatio = pl == 0 ? 1 : Expm1(pl) / pl;
                    double term = StirlingSeriesCoefficients[k - 1] * p * power * expRatio * lnRatio;
                    dd += term;
                    if (Math.Abs(term) <= 0.25 * Epsilon * Math.Abs(dd))
                    {
                        break;
                    }

                    power *= inverse2;
                }

                double v = -eps * dd;
                return v == 0 ? -dd : -dd * (Expm1(v) / v);
            }

            if (x < -20)
            {
                // Γ(1-x-ε)/Γ(1-x) = 1/(1 - ε Q(1-x, -ε))、sin π(x+ε)/sin πx = cos πε + cot πx sin πε
                double qr = Hypergeometric2F1GammaQuotient(1 - x, -eps);
                double reflected = qr / (1 - eps * qr);
                double sinRatio = eps == 0 ? Math.PI : double.SinPi(eps) / eps;
                double half = double.SinPi(0.5 * eps);
                double cosTerm = eps == 0 ? 0 : -2 * half * half / eps;
                double cot = double.CosPi(x) / double.SinPi(x);
                double sineQuotient = cosTerm + cot * sinRatio;
                // (X S - 1)/ε = (X - 1)/ε · S + (S - 1)/ε
                return reflected * (1 + eps * sineQuotient) + sineQuotient;
            }

            int count = (int)Math.Ceiling(10 - x);
            double product = 0;
            for (int j = 0; j < count; j++)
            {
                // (Πⱼ₊₁ - 1)/ε = (Πⱼ - 1)/ε · (1 + ε/(x+j)) + 1/(x+j)
                double inv = 1 / (x + j);
                product = product * (1 + eps * inv) + inv;
            }

            double qy = Hypergeometric2F1GammaQuotient(x + count, eps);
            return product + qy + eps * product * qy;
        }
    }
}
