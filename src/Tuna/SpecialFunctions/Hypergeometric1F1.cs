using System.Numerics;
using static Tuna.XMath.Constants;
using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // 級数・漸化式の相対誤差の見積もり許容値（超える場合は別手法を試す）
        private const double Hypergeometric1F1Tolerance = 4 * MachineEpsilon;

        // 漸近展開を試す最小の x（これ未満では項が減少する前に発散する）
        private const double Hypergeometric1F1AsymptoticMin = 8;

        // 多倍長級数の最大精度（ビット数）。これを超える打ち消しが必要な場合は double.NaN を返す
        private const int Hypergeometric1F1MaxPrecisionBits = 8192;

        // 倍精度仮数を再正規化する閾値（2^±500）
        private const double ConfluentRescaleLimit = 3.273390607896142e150;

        /// <summary>
        /// Computes Kummer's confluent hypergeometric function <c>₁F₁(a; b; x)</c> (also denoted <c>M(a, b, x)</c>).
        /// <para>合流型超幾何関数（Kummer 関数）<c>₁F₁(a; b; x)</c>（<c>M(a, b, x)</c>）を計算します。</para>
        /// </summary>
        /// <param name="a">The numerator parameter. / 分子パラメータ。</param>
        /// <param name="b">The denominator parameter. / 分母パラメータ。</param>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>₁F₁(a; b; x)</c>, or <see cref="double.NaN"/> at poles or when the value cannot be evaluated. / <c>₁F₁(a; b; x)</c> の値。極または計算不能な場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>When <paramref name="a"/> is a non-positive integer <c>-n</c>, the function reduces to a polynomial of degree <c>n</c>; when <paramref name="b"/> is a non-positive integer <c>-m</c>, the result is the polynomial value if <c>a = -n</c> with <c>n ≤ m</c> and <see cref="double.NaN"/> otherwise (use <see cref="Hypergeometric1F1Regularized"/> for finite values at non-positive integer <paramref name="b"/>). Evaluation may be slow or return <see cref="double.NaN"/> when <paramref name="a"/>, <paramref name="b"/>, and <paramref name="x"/> are all large (10⁴ or more of comparable magnitude).</para>
        /// <para><paramref name="a"/> が 0 以下の整数 <c>-n</c> のときは <c>n</c> 次多項式となります。<paramref name="b"/> が 0 以下の整数 <c>-m</c> の場合、<c>a = -n</c>（<c>n ≤ m</c>）であれば多項式の値を返し、それ以外は極のため <see cref="double.NaN"/> を返します（0 以下の整数 <paramref name="b"/> で有限値が必要な場合は <see cref="Hypergeometric1F1Regularized"/> を使用してください）。<paramref name="a"/>、<paramref name="b"/>、<paramref name="x"/> がいずれも大きい場合（10⁴ 以上で同程度）は計算に時間がかかるか <see cref="double.NaN"/> を返すことがあります。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/13.2.E2"/>
        public static double Hypergeometric1F1(double a, double b, double x)
        {
            if (double.IsNaN(a) || double.IsNaN(b) || double.IsNaN(x) || double.IsInfinity(a) || double.IsInfinity(b))
            {
                return double.NaN;
            }

            if (x == 0 || a == 0)
            {
                return 1;
            }

            if (IsNonPositiveInteger(b) && !(IsNonPositiveInteger(a) && a >= b))
            {
                return double.NaN;
            }

            if (double.IsInfinity(x))
            {
                return Hypergeometric1F1AtInfinity(a, b, x);
            }

            double mantissa = Hypergeometric1F1Scaled(a, b, x, out long exponent);
            return ScaleB(mantissa, exponent);
        }

        /// <summary>
        /// Computes the regularized confluent hypergeometric function <c>₁F̃₁(a; b; x) = ₁F₁(a; b; x) / Γ(b)</c>.
        /// <para>正則化された合流型超幾何関数 <c>₁F̃₁(a; b; x) = ₁F₁(a; b; x) / Γ(b)</c> を計算します。</para>
        /// </summary>
        /// <param name="a">The numerator parameter. / 分子パラメータ。</param>
        /// <param name="b">The denominator parameter (including non-positive integers). / 分母パラメータ（0 以下の整数も指定可能）。</param>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>₁F₁(a; b; x) / Γ(b)</c>. / <c>₁F₁(a; b; x) / Γ(b)</c> の値。</returns>
        /// <remarks>
        /// <para>Entire in <paramref name="a"/> and <paramref name="b"/>, avoiding poles at non-positive integers <c>b = -m</c> and overflow of <c>Γ(b)</c> for large <paramref name="b"/>.</para>
        /// <para><paramref name="a"/> および <paramref name="b"/> に関する整関数であり、0 以下の整数 <c>b = -m</c> における極や大きな <paramref name="b"/> での <c>Γ(b)</c> のオーバーフローを回避して有限値を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/13.2.E3"/>
        public static double Hypergeometric1F1Regularized(double a, double b, double x)
        {
            if (double.IsNaN(a) || double.IsNaN(b) || double.IsNaN(x) || double.IsInfinity(a) || double.IsInfinity(b))
            {
                return double.NaN;
            }

            if (IsNonPositiveInteger(b))
            {
                if (b < -int.MaxValue + 2)
                {
                    return double.NaN;
                }

                // b = -m: (a)ₘ₊₁ x^(m+1)/(m+1)! · M(a + m + 1, m + 2, x)（DLMF 13.2.5）。a = -n（n ≤ m）なら (a)ₘ₊₁ = 0
                int m = (int)-b;
                if (IsNonPositiveInteger(a) && a >= b)
                {
                    return 0;
                }

                if (x == 0)
                {
                    return 0;
                }

                double coefficient = ConfluentScaledPochhammerPowerOverFactorial(a, x, m + 1, out long coefficientExponent);
                if (double.IsInfinity(x))
                {
                    return coefficient * Hypergeometric1F1(a + m + 1, m + 2, x) * double.PositiveInfinity;
                }

                double rest = Hypergeometric1F1Scaled(a + m + 1, m + 2, x, out long restExponent);
                return ScaleB(coefficient * rest, coefficientExponent + restExponent);
            }

            if (x == 0 || a == 0)
            {
                return ReciprocalGamma(b);
            }

            if (double.IsInfinity(x))
            {
                return Hypergeometric1F1AtInfinity(a, b, x) * SignGamma(b);
            }

            double mantissa = Hypergeometric1F1Scaled(a, b, x, out long exponent);
            double gamma = ScaledGammaSigned(b, out long gammaExponent);
            return ScaleB(mantissa / gamma, exponent - gammaExponent);
        }

        // x = ±∞ での M(a, b, x)（a ≠ 0、b は極でないか a が b 以上の 0 以下の整数）
        private static double Hypergeometric1F1AtInfinity(double a, double b, double x)
        {
            bool polynomial = IsNonPositiveInteger(a);
            if (x > 0)
            {
                if (polynomial)
                {
                    // 最高次の項 (-n)ₙ xⁿ/((b)ₙ n!) = (-1)ⁿ xⁿ/(b)ₙ
                    return (IsOddInteger(-a) ? -1 : 1) * ConfluentSignPochhammer(b, -a) * double.PositiveInfinity;
                }

                // M ~ Γ(b)/Γ(a) eˣ x^(a-b)
                return SignGamma(b) * SignGamma(a) * double.PositiveInfinity;
            }

            // M(a, b, x) = eˣ M(b - a, b, -x) ~ Γ(b)/Γ(b - a) |x|^(-a)（多項式では (-1)ⁿxⁿ/(b)ₙ → 1/(b)ₙ の符号）
            if (polynomial)
            {
                return ConfluentSignPochhammer(b, -a) * double.PositiveInfinity;
            }

            if (IsNonPositiveInteger(b - a) || a > 0)
            {
                return 0;
            }

            return SignGamma(b) * SignGamma(b - a) * double.PositiveInfinity;
        }

        // ポッホハマー記号 (b)ₙ の符号（負の因子の個数の偶奇、因子に 0 があれば 0）
        private static double ConfluentSignPochhammer(double b, double n)
        {
            if (b > 0)
            {
                return 1;
            }

            if (IsInteger(b) && -b < n)
            {
                return 0;
            }

            double negatives = Math.Min(n, Math.Ceiling(-b));
            return IsOddInteger(negatives) ? -1 : 1;
        }

        // 有限の x に対する M(a, b, x) = mantissa · 2^exponent の仮数を返す
        private static double Hypergeometric1F1Scaled(double a, double b, double x, out long exponent)
        {
            exponent = 0;
            if (x == 0 || a == 0)
            {
                return 1;
            }

            if (IsNonPositiveInteger(a))
            {
                // a = -n（0 以下の整数）は n 次で打ち切る多項式として評価する（a = b = -m も m 次の有限和として扱う）
                return Hypergeometric1F1Polynomial(a, b, x, out exponent);
            }

            if (a == b)
            {
                return ConfluentScaledExp(x, 0, out exponent);
            }

            double kummerA = b - a;
            if (IsNonPositiveInteger(kummerA))
            {
                // Kummer 変換で多項式になる
                double polynomial = Hypergeometric1F1Polynomial(kummerA, b, -x, out long polynomialExponent);
                double exp = ConfluentScaledExp(x, 0, out long expExponent);
                exponent = polynomialExponent + expExponent;
                return polynomial * exp;
            }

            if (b < 0)
            {
                return Hypergeometric1F1NegativeB(a, 0, b, x, out exponent);
            }

            if (x > 0)
            {
                return Hypergeometric1F1PositiveArgument(a, 0, b, x, out exponent, out _);
            }

            // x < 0、b > 0: |x| が小さく直接の級数の打ち消し（およそ e^(2|x| max(a, 0)/b)）が小さければ直接足す
            if (-x * (Math.Abs(a) + 1) <= 4 * (b + 1))
            {
                double direct = Hypergeometric1F1Series(a, 0, b, x, out exponent, out double error);
                if (error <= Hypergeometric1F1Tolerance)
                {
                    return direct;
                }
            }

            // b - a は丸めずに上位と下位に分けて渡す（丸めると M の a についての感度だけ誤差が増える）
            var kummer = DoubleDoubleTwoSum(b, -a);
            if (x < -1000 && b <= 50 && Hypergeometric1F1Underflows(kummer.Hi, b, -x))
            {
                return 0;
            }

            double transformed = Hypergeometric1F1PositiveArgument(kummer.Hi, kummer.Lo, b, -x, out long transformedExponent, out _);
            double factor = ConfluentScaledExp(x, 0, out long factorExponent);
            exponent = transformedExponent + factorExponent;
            return transformed * factor;
        }

        // x > 0, b > 0（a は 0 以下の整数でない）での M(a, b, x) を漸近展開・振動域の漸化式・級数の順に評価する
        private static double Hypergeometric1F1PositiveArgument(double a, double aLo, double b, double x, out long exponent, out double error)
        {
            if (x >= Hypergeometric1F1AsymptoticMin && Hypergeometric1F1TryAsymptotic(a, aLo, b, x, out double asymptotic, out exponent))
            {
                error = MachineEpsilon;
                return asymptotic;
            }

            if (a < 0)
            {
                // 振動域（κ = b/2 - a が転回点 κₜ = (x² + (b-1)²)/(4x) より大きい）では、転回点の近くから下向きの漸化式で下げる
                double turning = 0.5 * b - (x * x + (b - 1) * (b - 1)) / (4 * x);
                if (a < turning - 2)
                {
                    double steps = Math.Floor(turning - a);
                    if (steps > MaxIterations)
                    {
                        exponent = 0;
                        error = double.PositiveInfinity;
                        return double.NaN;
                    }

                    return Hypergeometric1F1RecurDown(a, aLo, b, x, (int)steps, out exponent, out error);
                }
            }

            var sum = Hypergeometric1F1SeriesAccurate(a, aLo, b, x, out exponent, out error);
            return sum.Hi + sum.Lo;
        }

        // Kummer 変換後の eˣ M(a, b, -x)（x < 0, b > 0）が振動域で包絡線（DLMF 13.8.9 主要項）に余裕 e⁶⁰ を掛けても
        // 非正規化数の範囲を下回るかを判定し、不要な多倍長計算を省く
        private static bool Hypergeometric1F1Underflows(double a, double b, double absX)
        {
            double turning = 0.5 * b - (absX * absX + (b - 1) * (b - 1)) / (4 * absX);
            if (!(a < turning - 2))
            {
                return false;
            }

            double kappa = 0.5 * b - a;
            double logEnvelope = LogGamma(b) + 0.5 * absX + ((1 - b) * 0.5 - 0.25) * Math.Log(kappa * absX);
            return -absX + logEnvelope + 60 < -745;
        }

        // b < 0（極でない、a および b - a は 0 以下の整数でない）での M(a, b, x)。
        // 漸近展開が使えなければ直接級数と Kummer 変換後級数の誤差見積もりが小さい方を採用する
        private static double Hypergeometric1F1NegativeB(double a, double aLo, double b, double x, out long exponent)
        {
            var kummer = DoubleDoubleAdd(DoubleDoubleTwoSum(b, -a), (-aLo, 0.0));
            double absX = Math.Abs(x);
            if (absX >= Hypergeometric1F1AsymptoticMin)
            {
                if (x > 0 && Hypergeometric1F1TryAsymptotic(a, aLo, b, x, out double value, out exponent))
                {
                    return value;
                }

                if (x < 0 && Hypergeometric1F1TryAsymptotic(kummer.Hi, kummer.Lo, b, -x, out value, out long valueExponent))
                {
                    double factor = ConfluentScaledExp(x, 0, out long factorExponent);
                    exponent = valueExponent + factorExponent;
                    return value * factor;
                }
            }

            double direct = Hypergeometric1F1Series(a, aLo, b, x, out exponent, out double directError);
            if (directError <= Hypergeometric1F1Tolerance)
            {
                return direct;
            }

            double transformed = Hypergeometric1F1Series(kummer.Hi, kummer.Lo, b, -x, out long transformedExponent, out double transformedError);
            if (transformedError <= Hypergeometric1F1Tolerance)
            {
                double factor = ConfluentScaledExp(x, 0, out long factorExponent);
                exponent = transformedExponent + factorExponent;
                return transformed * factor;
            }

            // どちらも打ち消しが大きいため、打ち消しの小さい方を多倍長の級数で再評価する
            if (transformedError < directError)
            {
                var exact = Hypergeometric1F1SeriesMultiprecision(kummer.Hi, kummer.Lo, b, -x, transformedError, out long exactExponent);
                double factor = ConfluentScaledExp(x, 0, out long factorExponent);
                exponent = exactExponent + factorExponent;
                return (exact.Hi + exact.Lo) * factor;
            }

            var directExact = Hypergeometric1F1SeriesMultiprecision(a, aLo, b, x, directError, out exponent);
            return directExact.Hi + directExact.Lo;
        }

        // 多項式 M(-n, b, x)。打ち消しが大きく b, x > 0 ならラゲール型の 3 項漸化式で M(0), M(-1) から下げる
        private static double Hypergeometric1F1Polynomial(double a, double b, double x, out long exponent)
        {
            if (-a > MaxIterations)
            {
                exponent = 0;
                return double.NaN;
            }

            double series = Hypergeometric1F1Series(a, 0, b, x, out exponent, out double error);
            if (error <= Hypergeometric1F1Tolerance)
            {
                return series;
            }

            if (!(b > 0 && x > 0))
            {
                var exact = Hypergeometric1F1SeriesMultiprecision(a, 0, b, x, error, out exponent);
                return exact.Hi + exact.Lo;
            }

            // M(0) = 1、M(-1) = 1 - x/b から a = -n まで下げる（ラゲール多項式の漸化式。下向きに安定）
            int n = (int)-a;
            var quotient = DoubleDoubleDivide((x, 0.0), b);
            var current = DoubleDoubleAdd((1.0, 0.0), (-quotient.Hi, -quotient.Lo));
            exponent = 0;
            return Hypergeometric1F1RecurrenceDown(a, 0, b, x, n - 1, current, (1.0, 0.0), ref exponent);
        }

        // 振動域の M(a, b, x)（x > 0, b > 0, a < 0）。a₁ = a + steps と a₁ + 1 から下向きの 3 項漸化式で下げる
        private static double Hypergeometric1F1RecurDown(double a, double aLo, double b, double x, int steps, out long exponent, out double error)
        {
            var a1 = DoubleDoubleAdd(DoubleDoubleTwoSum(a, steps), (aLo, 0.0));
            var a2 = DoubleDoubleAdd(DoubleDoubleTwoSum(a, steps + 1.0), (aLo, 0.0));
            var current = Hypergeometric1F1SeriesAccurate(a1.Hi, a1.Lo, b, x, out long currentExponent, out double currentError);
            var next = Hypergeometric1F1SeriesAccurate(a2.Hi, a2.Lo, b, x, out long nextExponent, out double nextError);
            if (current.Hi == 0 && next.Hi == 0)
            {
                exponent = 0;
                error = double.PositiveInfinity;
                return 0;
            }

            // 2 つの値を共通の指数にそろえる
            exponent = Math.Max(currentExponent, nextExponent);
            current = ConfluentScaleDoubleDouble(current, currentExponent - exponent);
            next = ConfluentScaleDoubleDouble(next, nextExponent - exponent);

            // 振動域の漸化式では誤差はほぼ増えない
            error = Math.Max(currentError, nextError);
            return Hypergeometric1F1RecurrenceDown(a, aLo, b, x, steps, current, next, ref exponent);
        }

        // s = a + steps での M(s)（current）と M(s+1)（next）から (b - s)M(s-1) = s M(s+1) - (2s - b + x) M(s) を
        // double-double で進めて M(a) を求める（数百段以上で倍精度の丸め誤差が蓄積するのを防ぐ）
        private static double Hypergeometric1F1RecurrenceDown(double a, double aLo, double b, double x, int steps,
            (double Hi, double Lo) current, (double Hi, double Lo) next, ref long exponent)
        {
            var negativeB = (-b, 0.0);
            for (int k = 0; k < steps; k++)
            {
                var s = DoubleDoubleAdd(DoubleDoubleTwoSum(a, steps - k), (aLo, 0.0));
                var middle = DoubleDoubleAdd(DoubleDoubleAdd((2 * s.Hi, 2 * s.Lo), negativeB), (x, 0.0));
                var numerator = DoubleDoubleAdd(DoubleDoubleMultiply(s, next), DoubleDoubleMultiply((-middle.Hi, -middle.Lo), current));
                var previous = DoubleDoubleDivide(numerator, DoubleDoubleAdd((b, 0.0), (-s.Hi, -s.Lo)));
                next = current;
                current = previous;
                double size = Math.Max(Math.Abs(current.Hi), Math.Abs(next.Hi));
                if (size > ConfluentRescaleLimit || (size < 1 / ConfluentRescaleLimit && size > 0))
                {
                    int shift = Math.ILogB(size);
                    current = ConfluentScaleDoubleDouble(current, -shift);
                    next = ConfluentScaleDoubleDouble(next, -shift);
                    exponent += shift;
                }
            }

            return current.Hi + current.Lo;
        }

        private static (double Hi, double Lo) ConfluentScaleDoubleDouble((double Hi, double Lo) value, long shift)
        {
            int s = (int)Math.Clamp(shift, -4096, 4096);
            return (Math.ScaleB(value.Hi, s), Math.ScaleB(value.Lo, s));
        }

        // current と next を同じ 2 の冪でスケーリングして指数に加える
        private static void ConfluentRenormalizePair(ref double current, ref double next, ref long exponent)
        {
            double size = Math.Max(Math.Abs(current), Math.Abs(next));
            if (size > ConfluentRescaleLimit || (size < 1 / ConfluentRescaleLimit && size > 0))
            {
                int shift = Math.ILogB(size);
                current = Math.ScaleB(current, -shift);
                next = Math.ScaleB(next, -shift);
                exponent += shift;
            }
        }

        // 級数 M(a, b, x) = Σ (a)ₖ xᵏ/((b)ₖ k!) を double-double で計算し仮数と指数で返す
        private static double Hypergeometric1F1Series(double a, double aLo, double b, double x, out long exponent, out double error)
        {
            var sum = Hypergeometric1F1SeriesDoubleDouble(a, aLo, b, x, out exponent, out error);
            return sum.Hi + sum.Lo;
        }

        // 級数 M(a, b, x) を double-double のまま返す（漸化式の初期値にも使用）。
        // a + k, b + k は TwoSum で正確に表すため、誤差見積もり error は打ち消しと項数から求める
        private static (double Hi, double Lo) Hypergeometric1F1SeriesDoubleDouble(double a, double aLo, double b, double x, out long exponent, out double error)
        {
            (double Hi, double Lo) sum = (1.0, 0.0);
            (double Hi, double Lo) term = (1.0, 0.0);
            double maxTerm = 1;
            exponent = 0;
            int k = 0;
            for (; k < MaxIterations; k++)
            {
                var numerator = DoubleDoubleTwoSum(a, k);
                if (aLo != 0)
                {
                    numerator = DoubleDoubleAdd(numerator, (aLo, 0.0));
                }

                if (numerator.Hi == 0)
                {
                    // a が 0 以下の整数なら有限和で終わる
                    break;
                }

                var denominator = DoubleDoubleMultiply(DoubleDoubleTwoSum(b, k), (k + 1.0, 0.0));
                term = DoubleDoubleDivide(DoubleDoubleMultiply(DoubleDoubleMultiply(term, numerator), (x, 0.0)), denominator);
                sum = DoubleDoubleAdd(sum, term);
                double absTerm = Math.Abs(term.Hi);
                if (absTerm > maxTerm)
                {
                    maxTerm = absTerm;
                }

                // 次の項の比 |(a + k + 1) x/((b + k + 1)(k + 2))| が 1/2 未満で、符号の変わる区間（k < -a, k < -b）を過ぎていれば、
                // 残りは今の項の大きさで抑えられる
                double ratio = Math.Abs((a + k + 1) * x / ((b + k + 1) * (k + 2)));
                if (ratio < 0.5 && k + 1 > -a && k + 1 > -b && absTerm <= 1.3877787807814457e-17 * Math.Abs(sum.Hi))
                {
                    break;
                }

                if (Math.Abs(sum.Hi) > ConfluentRescaleLimit || absTerm > ConfluentRescaleLimit)
                {
                    // 和と項を同じ 2 の冪で縮める（double-double の下位も正確に縮む）
                    int shift = Math.ILogB(Math.Max(Math.Abs(sum.Hi), absTerm));
                    sum = (Math.ScaleB(sum.Hi, -shift), Math.ScaleB(sum.Lo, -shift));
                    term = (Math.ScaleB(term.Hi, -shift), Math.ScaleB(term.Lo, -shift));
                    maxTerm = Math.ScaleB(maxTerm, -shift);
                    exponent += shift;
                }

                if (!double.IsFinite(sum.Hi))
                {
                    break;
                }
            }

            double result = sum.Hi + sum.Lo;
            error = k >= MaxIterations || !double.IsFinite(result) || result == 0
                ? double.PositiveInfinity
                : (maxTerm / Math.Abs(result)) * (k + 2) * 9.860761315262648e-32 + 0.5 * MachineEpsilon;
            if (k >= MaxIterations)
            {
                return (double.NaN, 0);
            }

            return sum;
        }

        // 級数を double-double で計算し、打ち消しが大きく誤差見積もりが許容値を超える場合は多倍長で再評価する
        private static (double Hi, double Lo) Hypergeometric1F1SeriesAccurate(double a, double aLo, double b, double x, out long exponent, out double error)
        {
            var sum = Hypergeometric1F1SeriesDoubleDouble(a, aLo, b, x, out exponent, out error);
            if (error <= Hypergeometric1F1Tolerance || double.IsNaN(sum.Hi))
            {
                return sum;
            }

            var exact = Hypergeometric1F1SeriesMultiprecision(a, aLo, b, x, error, out long exactExponent);
            if (double.IsNaN(exact.Hi))
            {
                return sum;
            }

            exponent = exactExponent;
            error = MachineEpsilon;
            return exact;
        }

        // 打ち消しの大きい領域向けの多倍長浮動小数点（BigInteger 仮数 + 2 の指数）による級数評価。
        // 初期精度は倍精度の誤差見積もりから決め、有効桁数が不足すれば精度を上げて再評価する
        private static (double Hi, double Lo) Hypergeometric1F1SeriesMultiprecision(double a, double aLo, double b, double x, double estimatedError, out long exponent)
        {
            // 倍精度の見積もりは error ≈ (最大の項/和)·k·2⁻¹⁰³ なので、失われるビット数はおよそ log₂(error) + 103
            double lostEstimate = double.IsFinite(estimatedError) ? Math.Log2(estimatedError) + 103 : 256;
            int precision = (int)Math.Max(128, lostEstimate + 128);
            while (precision <= Hypergeometric1F1MaxPrecisionBits)
            {
                if (Hypergeometric1F1TryMultiprecision(a, aLo, b, x, precision, out var result, out exponent, out long lostBits))
                {
                    return result;
                }

                precision = (int)Math.Min(int.MaxValue / 2, Math.Max(2L * precision, lostBits + 128));
            }

            exponent = 0;
            return (double.NaN, 0);
        }

        // 精度 precision ビットで級数を評価し、打ち消しによる桁落ち（lostBits）が精度内に収まれば true を返す
        private static bool Hypergeometric1F1TryMultiprecision(double a, double aLo, double b, double x, int precision,
            out (double Hi, double Lo) result, out long exponent, out long lostBits)
        {
            BigInteger numeratorA = ConfluentDyadic(a, out int exponentA);
            if (aLo != 0)
            {
                // a + aLo を共通の指数にそろえて正確に足す
                BigInteger numeratorLo = ConfluentDyadic(aLo, out int exponentLo);
                int common = Math.Min(exponentA, exponentLo);
                numeratorA = (numeratorA << (exponentA - common)) + (numeratorLo << (exponentLo - common));
                exponentA = common;
            }

            BigInteger numeratorB = ConfluentDyadic(b, out int exponentB);
            BigInteger numeratorX = ConfluentDyadic(x, out int exponentX);
            BigInteger unitA = BigInteger.One << -exponentA;
            BigInteger unitB = BigInteger.One << -exponentB;

            BigInteger term = BigInteger.One;
            long termExponent = 0;
            BigInteger sum = BigInteger.One;
            long sumExponent = 0;
            long maxTop = 1;
            int k = 0;
            for (; k < MaxIterations; k++)
            {
                BigInteger factorA = numeratorA + k * unitA;
                if (factorA.IsZero)
                {
                    break;
                }

                BigInteger divisor = (numeratorB + k * unitB) * (k + 1);

                // term · (a + k) x / ((b + k)(k + 1))。割る前に左シフトして商が precision ビット以上になるようにする
                term *= factorA * numeratorX;
                termExponent += exponentA + exponentX - exponentB;
                long shift = precision + ConfluentBitLength(divisor) - ConfluentBitLength(term) + 2;
                if (shift > 0)
                {
                    term <<= (int)shift;
                    termExponent -= shift;
                }

                term /= divisor;
                ConfluentTruncate(ref term, ref termExponent, precision);

                long termTop = termExponent + ConfluentBitLength(term);
                maxTop = Math.Max(maxTop, termTop);
                ConfluentAccumulate(ref sum, ref sumExponent, term, termExponent, precision + 64);

                double ratio = Math.Abs((a + k + 1) * x / ((b + k + 1) * (k + 2)));
                long sumTop = sumExponent + ConfluentBitLength(sum);
                if (ratio < 0.5 && k + 1 > -a && k + 1 > -b && termTop < sumTop - 64)
                {
                    break;
                }
            }

            result = (double.NaN, 0);
            exponent = 0;
            if (k >= MaxIterations)
            {
                lostBits = long.MaxValue / 4;
                return false;
            }

            if (sum.IsZero)
            {
                lostBits = 2L * precision;
                return false;
            }

            long length = ConfluentBitLength(sum);
            lostBits = maxTop - (sumExponent + length);
            if (lostBits + 72 > precision)
            {
                return false;
            }

            // 上位 110 ビットを double-double に変換する
            BigInteger top = length > 110 ? sum >> (int)(length - 110) : sum << (int)(110 - length);
            double hi = (double)top;
            double lo = (double)(top - new BigInteger(hi));
            result = (hi, lo);
            exponent = sumExponent + length - 110;
            return true;
        }

        // v = numerator · 2^exponent（exponent ≤ 0）に分解する
        private static BigInteger ConfluentDyadic(double v, out int exponent)
        {
            if (v == 0)
            {
                exponent = 0;
                return BigInteger.Zero;
            }

            long bits = BitConverter.DoubleToInt64Bits(v);
            int biased = (int)((bits >> 52) & 0x7FF);
            long mantissa = bits & 0xFFFFFFFFFFFFFL;
            if (biased == 0)
            {
                biased = 1;
            }
            else
            {
                mantissa |= 1L << 52;
            }

            exponent = biased - 1075;
            BigInteger numerator = mantissa;
            if (exponent > 0)
            {
                numerator <<= exponent;
                exponent = 0;
            }

            return v < 0 ? -numerator : numerator;
        }

        // 2 の補数の最小ビット数（負の数の場合、絶対値のビット数と高々 1 異なる）
        private static long ConfluentBitLength(BigInteger v) => (long)v.GetBitLength();

        // 仮数を precision ビットに切り詰め、切り捨てた分だけ指数を上げる
        private static void ConfluentTruncate(ref BigInteger mantissa, ref long exponent, int precision)
        {
            long excess = ConfluentBitLength(mantissa) - precision;
            if (excess > 0)
            {
                // 負の数の右シフトは -∞ 向きの丸めになるが、誤差は最下位の 1 単位
                mantissa >>= (int)excess;
                exponent += excess;
            }
        }

        // sum · 2^sumExponent に term · 2^termExponent を加算し、precision ビットに切り詰める
        private static void ConfluentAccumulate(ref BigInteger sum, ref long sumExponent, BigInteger term, long termExponent, int precision)
        {
            long sumTop = sumExponent + ConfluentBitLength(sum);
            long termTop = termExponent + ConfluentBitLength(term);
            if (term.IsZero || (!sum.IsZero && sumTop - termTop > precision))
            {
                return;
            }

            if (sum.IsZero || termTop - sumTop > precision)
            {
                sum = term;
                sumExponent = termExponent;
                return;
            }

            if (sumExponent > termExponent)
            {
                sum = (sum << (int)(sumExponent - termExponent)) + term;
                sumExponent = termExponent;
            }
            else
            {
                sum += term << (int)(termExponent - sumExponent);
            }

            ConfluentTruncate(ref sum, ref sumExponent, precision);
        }

        // 大きな x > 0 での漸近展開 M(a, b, x) ~ Γ(b)/Γ(a) eˣ x^(a-b) Σ (1-a)ₛ(b-a)ₛ/(s! xˢ)（DLMF 13.7.2）。
        // 劣勢項との比が ε/16 を超えるか倍精度まで収束しない場合は false を返す
        private static bool Hypergeometric1F1TryAsymptotic(double a, double aLo, double b, double x, out double mantissa, out long exponent)
        {
            mantissa = 0;
            exponent = 0;
            if (IsNonPositiveInteger(a))
            {
                return false;
            }

            double c = (1 - a) - aLo;
            double d = (b - a) - aLo;

            // 最初の項の比が 1 を超えるなら収束しない
            if (Math.Abs(c * d) >= x)
            {
                return false;
            }

            // 劣勢な項との比 |Γ(a)/Γ(b-a)| x^(b-2a) e^(-x)
            double logRatio = LogGamma(a) - LogGamma(d) + (b - 2 * a) * Math.Log(x) - x;
            if (!(logRatio < Math.Log(MachineEpsilon / 16)))
            {
                return false;
            }

            double sum = 1;
            double term = 1;
            bool converged = false;
            for (int s = 0; s < 1000; s++)
            {
                double next = term * (c + s) * (d + s) / ((s + 1) * x);
                if (next == 0)
                {
                    // c または d が 0 以下の整数なら有限和で終わる
                    converged = true;
                    break;
                }

                if (Math.Abs(next) > Math.Abs(term))
                {
                    break;
                }

                term = next;
                sum += term;
                if (Math.Abs(term) <= 0.125 * MachineEpsilon * Math.Abs(sum))
                {
                    converged = true;
                    break;
                }
            }

            if (!converged)
            {
                return false;
            }

            double gammaB = ScaledGammaSigned(b, out long gammaBExponent);
            double gammaA = ScaledGammaSigned(a, out long gammaAExponent);
            double exp = ConfluentScaledExp(x, 0, out long expExponent);
            double power = ConfluentScaledPower(x, a - b, out long powerExponent);
            mantissa = gammaB / gammaA * exp * power * sum;
            if (aLo != 0)
            {
                // a の下位 aLo の分: 1/Γ(a + aLo) x^(a+aLo-b) ≈ 1/Γ(a) x^(a-b) (1 + aLo(ln x - ψ(a)))
                mantissa *= 1 + aLo * (Math.Log(x) - Digamma(a));
            }

            exponent = gammaBExponent - gammaAExponent + expExponent + powerExponent;
            return true;
        }

        // e^(hi + lo) = mantissa · 2^exponent（|lo| ≪ |hi|）。n = round(hi/ln 2) として ln 2 を上位・下位に分けて引くことで精度を保つ
        internal static double ConfluentScaledExp(double hi, double lo, out long exponent)
        {
            if (double.IsInfinity(hi) || Math.Abs(hi) > 1e18)
            {
                exponent = 0;
                return hi > 0 ? double.PositiveInfinity : 0;
            }

            double n = Math.Round(hi / Ln2);
            double r = Math.FusedMultiplyAdd(-n, Ln2, hi) - n * Ln2Lo + lo;
            exponent = (long)n;
            return Math.Exp(r);
        }

        // x^p = mantissa · 2^exponent（x > 0、実数 p）。整数部分は ScaledPower を使う
        private static double ConfluentScaledPower(double x, double p, out long exponent)
        {
            double n = Math.Round(p);
            if (Math.Abs(n) > 1e17)
            {
                // 指数が大きすぎる場合は対数で計算する（結果はいずれにせよオーバーフローまたは 0 になる）
                double log = p * Math.Log(x);
                return ConfluentScaledExp(log, 0, out exponent);
            }

            double mantissa = ScaledPower(x, (long)n, out exponent);
            return mantissa * Math.Pow(x, p - n);
        }

        // (a)ₘ xᵐ/m! = mantissa · 2^exponent の仮数を 1 因子ごとに正規化しながら計算する
        private static double ConfluentScaledPochhammerPowerOverFactorial(double a, double x, int m, out long exponent)
        {
            double mantissa = 1;
            exponent = 0;
            for (int j = 0; j < m; j++)
            {
                mantissa *= (a + j) * x / (j + 1);
                if (mantissa == 0 || !double.IsFinite(mantissa))
                {
                    return mantissa;
                }

                int shift = Math.ILogB(mantissa);
                mantissa = Math.ScaleB(mantissa, -shift);
                exponent += shift;
            }

            return mantissa;
        }
    }
}
