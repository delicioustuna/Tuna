using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // HypergeometricUTemme の級数を使う x の上限（max(2, b/2) の定数部。x が大きいと正則部と特異部が打ち消し合う）
        private const double HypergeometricUTemmeMax = 2;

        // 漸化式の誤差見積もりが大きいときに領域外でも HypergeometricUTemme を試す x の上限
        private const double HypergeometricUTemmeFallbackMax = 1000;

        // 漸近展開を試す最小の x
        private const double HypergeometricUAsymptoticMin = 4;

        /// <summary>
        /// Computes the confluent hypergeometric function of the second kind (Tricomi's function) <c>U(a, b, x)</c>.
        /// <para>第 2 種合流型超幾何関数（Tricomi 関数）<c>U(a, b, x)</c> を計算します。</para>
        /// </summary>
        /// <param name="a">The first parameter. / 第 1 パラメータ。</param>
        /// <param name="b">The second parameter. / 第 2 パラメータ。</param>
        /// <param name="x">The non-negative input value (<c>x ≥ 0</c>). / 非負の入力値（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>U(a, b, x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> &lt; 0, if any parameter is non-finite or <see cref="double.NaN"/>, or if the value cannot be evaluated. / <c>U(a, b, x)</c> の値。<paramref name="x"/> &lt; 0、パラメータが有限でないか <see cref="double.NaN"/>、または計算不能な場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Reduces to a polynomial of degree <c>n</c> when <c>a = -n</c> is a non-positive integer, and to <c>x^(1 - b)</c> times a polynomial when <c>a - b + 1 = -n</c>. Evaluation may be slow when <paramref name="a"/> is large and <paramref name="x"/> is small, or when <c>a ≪ 0</c> and <paramref name="b"/> is large; may return <see cref="double.NaN"/> when <c>|a|</c> or the recurrence depth exceeds <c>10⁶</c>.</para>
        /// <para><c>a = -n</c>（0 以下の整数）のときは <c>n</c> 次多項式、<c>a - b + 1 = -n</c> のときは <c>x^(1 - b)</c> と多項式の積になります。<paramref name="a"/> が大きく <paramref name="x"/> が小さい場合や、<c>a ≪ 0</c> で <paramref name="b"/> が大きい場合は計算に時間がかかり、<c>|a|</c> や漸化式の段数が <c>10⁶</c> を超えると <see cref="double.NaN"/> を返すことがあります。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/13.2.E6"/>
        /// <seealso href="https://dlmf.nist.gov/13.2.E42"/>
        public static double HypergeometricU(double a, double b, double x)
        {
            if (double.IsNaN(a) || double.IsNaN(b) || double.IsNaN(x) || double.IsInfinity(a) || double.IsInfinity(b) || x < 0)
            {
                return double.NaN;
            }

            if (a == 0)
            {
                return 1;
            }

            if (double.IsPositiveInfinity(x))
            {
                // U ~ x^(-a)（多項式も最高次の係数は +1）
                return a > 0 ? 0 : double.PositiveInfinity;
            }

            if (x == 0)
            {
                return HypergeometricUAtZero(a, b);
            }

            double mantissa = HypergeometricUScaled(a, b, x, out long exponent);
            return ScaleB(mantissa, exponent);
        }

        /// <summary>
        /// Computes the Whittaker function of the first kind <c>M_(κ,μ)(x) = e^(-x/2) x^(μ + 1/2) M(1/2 + μ - κ, 1 + 2μ, x)</c>.
        /// <para>第 1 種 Whittaker 関数 <c>M_(κ,μ)(x) = e^(-x/2) x^(μ + 1/2) M(1/2 + μ - κ, 1 + 2μ, x)</c> を計算します。</para>
        /// </summary>
        /// <param name="kappa">The parameter <c>κ</c>. / パラメータ <c>κ</c>。</param>
        /// <param name="mu">The parameter <c>μ</c>. / パラメータ <c>μ</c>。</param>
        /// <param name="x">The non-negative input value (<c>x ≥ 0</c>). / 非負の入力値（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>M_(κ,μ)(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> &lt; 0, if <c>1 + 2μ</c> is a pole, or if any argument is invalid. / <c>M_(κ,μ)(x)</c> の値。<paramref name="x"/> &lt; 0、<c>1 + 2μ</c> が極となる場合、または不正な引数の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns finite values even where <see cref="Hypergeometric1F1"/> alone would overflow, as long as the scaled product with <c>e^(-x/2) x^(μ + 1/2)</c> is representable.</para>
        /// <para><see cref="Hypergeometric1F1"/> 単体ではオーバーフローする範囲でも、<c>e^(-x/2) x^(μ + 1/2)</c> との積が表現可能であれば有限値を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/13.14.E2"/>
        public static double WhittakerM(double kappa, double mu, double x)
        {
            if (double.IsNaN(kappa) || double.IsNaN(mu) || double.IsNaN(x) || double.IsInfinity(kappa) || double.IsInfinity(mu) || x < 0)
            {
                return double.NaN;
            }

            double b = 1 + 2 * mu;
            double a = 0.5 + mu - kappa;
            if (IsNonPositiveInteger(b) && !(IsNonPositiveInteger(a) && a >= b))
            {
                return double.NaN;
            }

            if (x == 0)
            {
                double power = mu + 0.5;
                return power > 0 ? 0 : power == 0 ? 1 : double.PositiveInfinity;
            }

            if (double.IsPositiveInfinity(x))
            {
                // M_(κ,μ) ~ Γ(1+2μ)/Γ(1/2+μ-κ) e^(x/2) x^(-κ)
                return IsNonPositiveInteger(a) ? double.NaN : SignGamma(b) * SignGamma(a) * double.PositiveInfinity;
            }

            double m = Hypergeometric1F1Scaled(a, b, x, out long exponent);
            double prefactor = WhittakerPrefactor(mu, x, out long prefactorExponent);
            return ScaleB(m * prefactor, exponent + prefactorExponent);
        }

        /// <summary>
        /// Computes the Whittaker function of the second kind <c>W_(κ,μ)(x) = e^(-x/2) x^(μ + 1/2) U(1/2 + μ - κ, 1 + 2μ, x)</c>.
        /// <para>第 2 種 Whittaker 関数 <c>W_(κ,μ)(x) = e^(-x/2) x^(μ + 1/2) U(1/2 + μ - κ, 1 + 2μ, x)</c> を計算します。</para>
        /// </summary>
        /// <param name="kappa">The parameter <c>κ</c>. / パラメータ <c>κ</c>。</param>
        /// <param name="mu">The parameter <c>μ</c> (symmetric: <c>W_(κ,μ) = W_(κ,-μ)</c>). / パラメータ <c>μ</c>（偶対称：<c>W_(κ,μ) = W_(κ,-μ)</c>）。</param>
        /// <param name="x">The non-negative input value (<c>x ≥ 0</c>). / 非負の入力値（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>W_(κ,μ)(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> &lt; 0 or if any argument is invalid. / <c>W_(κ,μ)(x)</c> の値。<paramref name="x"/> &lt; 0 または不正な引数の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns finite values even where <see cref="HypergeometricU"/> alone would overflow, as long as the scaled product with <c>e^(-x/2) x^(μ + 1/2)</c> is representable.</para>
        /// <para><see cref="HypergeometricU"/> 単体ではオーバーフローする範囲でも、<c>e^(-x/2) x^(μ + 1/2)</c> との積が表現可能であれば有限値を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/13.14.E3"/>
        public static double WhittakerW(double kappa, double mu, double x)
        {
            if (double.IsNaN(kappa) || double.IsNaN(mu) || double.IsNaN(x) || double.IsInfinity(kappa) || double.IsInfinity(mu) || x < 0)
            {
                return double.NaN;
            }

            mu = Math.Abs(mu);
            double a = 0.5 + mu - kappa;
            double b = 1 + 2 * mu;
            if (double.IsPositiveInfinity(x))
            {
                return 0;
            }

            if (x == 0)
            {
                if (IsNonPositiveInteger(a))
                {
                    // 多項式: U(-n, b, 0) = (-1)ⁿ (b)ₙ は有限なので x^(μ+1/2) → 0
                    return 0;
                }

                if (mu < 0.5)
                {
                    return 0;
                }

                if (mu == 0.5)
                {
                    return ReciprocalGamma(1 - kappa);
                }

                return SignGamma(a) * double.PositiveInfinity;
            }

            double u = HypergeometricUScaled(a, b, x, out long exponent);
            double prefactor = WhittakerPrefactor(mu, x, out long prefactorExponent);
            return ScaleB(u * prefactor, exponent + prefactorExponent);
        }

        // e^(-x/2) x^(μ+1/2) = mantissa · 2^exponent（x > 0）の仮数を返す
        private static double WhittakerPrefactor(double mu, double x, out long exponent)
        {
            double exp = ConfluentScaledExp(-0.5 * x, 0, out long expExponent);
            // μ + 1/2 の丸めを避けるため x^μ と √x に分ける
            double power = ConfluentScaledPower(x, mu, out long powerExponent) * Math.Sqrt(x);
            exponent = expExponent + powerExponent;
            return exp * power;
        }

        // U(a, b, 0)（a ≠ 0）
        private static double HypergeometricUAtZero(double a, double b)
        {
            if (IsNonPositiveInteger(a))
            {
                // U(-n, b, 0) = (-1)ⁿ (b)ₙ
                double n = -a;
                double pochhammer = Pochhammer(b, n);
                return IsOddInteger(n) ? -pochhammer : pochhammer;
            }

            if (b < 1)
            {
                // a - b + 1 が 0 以下の整数なら U = x^(1-b)·多項式 → 0
                return IsNonPositiveInteger(a - b + 1) ? 0 : GammaRatio(1 - b, a - b + 1);
            }

            if (IsNonPositiveInteger(a - b + 1))
            {
                // U = x^(1-b) U(a-b+1, 2-b, x) で、多項式の定数項 (-1)ⁿ(2-b)ₙ の符号の無限大（b = 1 では有限）
                double n = -(a - b + 1);
                double constant = Pochhammer(2 - b, n) * (IsOddInteger(n) ? -1 : 1);
                return b == 1 ? constant : Math.Sign(constant) * double.PositiveInfinity;
            }

            // b > 1: Γ(b-1)/Γ(a) x^(1-b)、b = 1: -ln x/Γ(a)
            return SignGamma(a) * double.PositiveInfinity;
        }

        // U(a, b, x) = mantissa · 2^exponent（0 < x < ∞）の仮数を返す
        private static double HypergeometricUScaled(double a, double b, double x, out long exponent)
        {
            exponent = 0;
            if (a == 0)
            {
                return 1;
            }

            if (IsNonPositiveInteger(a))
            {
                return HypergeometricUPolynomial(a, b, x, out exponent);
            }

            double c = a - b + 1;
            if (IsNonPositiveInteger(c))
            {
                // Kummer 変換 U(a, b, x) = x^(1-b) U(a-b+1, 2-b, x) で多項式になる
                double polynomial = HypergeometricUPolynomial(c, 2 - b, x, out long polynomialExponent);
                double power = ConfluentScaledPower(x, 1 - b, out long powerExponent);
                exponent = polynomialExponent + powerExponent;
                return polynomial * power;
            }

            if (x >= HypergeometricUAsymptoticMin && HypergeometricUTryAsymptotic(a, b, x, out double asymptotic, out exponent))
            {
                return asymptotic;
            }

            double temme = double.NaN;
            long temmeExponent = 0;
            double temmeError = double.PositiveInfinity;
            if (HypergeometricUInTemmeRegion(b, x))
            {
                temme = HypergeometricUTemmeAny(a, b, x, out temmeExponent, out temmeError);
                if (temmeError <= 64 * MachineEpsilon)
                {
                    exponent = temmeExponent;
                    return temme;
                }
            }

            if (double.IsNaN(temme) && b > x)
            {
                // b > x では a の漸化式で U が劣勢解にならない範囲があり（b - x が大きいほど広い）、Miller 法や下向きの漸化式の精度が落ちる。
                // Temme 型の級数は x/b がそれほど小さくなくても打ち消しが小さいことが多いので、見積もり誤差が十分小さければ採用する
                temme = HypergeometricUTemmeAny(a, b, x, out temmeExponent, out temmeError);
                if (temmeError <= 64 * MachineEpsilon)
                {
                    exponent = temmeExponent;
                    return temme;
                }
            }

            // Miller 法（a ≥ 2 では主な方法。a < 2 でも U(a₀) の誤差を拡大しないので多くの場合に安定）
            double value = HypergeometricUMiller(a, b, x, out exponent, out double error);
            if (error <= 64 * MachineEpsilon)
            {
                return value;
            }

            if (a < 2)
            {
                // a₀ での 2 つの値からの下向きの漸化式（Miller 法の連分数の項数が多すぎる場合など）
                double fromBase = HypergeometricUFromBase(a, b, x, out long fromBaseExponent, out double fromBaseError);
                if (!double.IsNaN(fromBase) && (double.IsNaN(value) || fromBaseError < error))
                {
                    value = fromBase;
                    exponent = fromBaseExponent;
                    error = fromBaseError;
                }
            }

            if (double.IsNaN(temme) && !(error <= 64 * MachineEpsilon) && x <= HypergeometricUTemmeFallbackMax)
            {
                // 漸化式が不安定なら、領域外でも Temme 型の級数を試す（見積もり誤差で比べる）
                temme = HypergeometricUTemmeAny(a, b, x, out temmeExponent, out temmeError);
            }

            if (!double.IsNaN(temme) && (double.IsNaN(value) || temmeError < error))
            {
                value = temme;
                exponent = temmeExponent;
                error = temmeError;
            }

            if (!(error <= 64 * MachineEpsilon) && !IsInteger(a) && !IsInteger(b))
            {
                // どれも打ち消しが大きい。2 つの M を直接求める接続公式を試す
                double connection = HypergeometricUConnection(a, b, x, out long connectionExponent, out double connectionError);
                if (!double.IsNaN(connection) && (double.IsNaN(value) || connectionError < error))
                {
                    exponent = connectionExponent;
                    return connection;
                }
            }

            return value;
        }

        // 接続公式 U = Γ(1-b)/Γ(a-b+1) M(a, b, x) + Γ(b-1)/Γ(a) x^(1-b) M(a-b+1, 2-b, x)（DLMF 13.2.42、a, b は非整数）。
        // a - b + 1, 1 - b, b - 1 の丸めは double-double で保持して 1 次補正する
        private static double HypergeometricUConnection(double a, double b, double x, out long exponent, out double error)
        {
            exponent = 0;
            error = double.PositiveInfinity;
            double bPrime = 2 - b;
            var c = DoubleDoubleAdd(DoubleDoubleTwoSum(a, -b), (1.0, 0.0));
            if (2 - bPrime != b || (c.Lo != 0 && IsInteger(c.Hi)))
            {
                // 2 - b が丸められる場合（第 2 項のパラメータがずれる）や、c の上位がちょうど整数になる場合は使わない
                return double.NaN;
            }

            // 第 1 項: Γ(1-b)/Γ(a-b+1) M(a, b, x)
            double m1 = Hypergeometric1F1Scaled(a, b, x, out long m1Exponent);
            double gammaOneMinusB = HypergeometricUScaledGamma(DoubleDoubleTwoSum(1, -b), out long gammaOneMinusBExponent);
            double gammaC = HypergeometricUScaledGamma(c, out long gammaCExponent);
            double first = m1 * gammaOneMinusB / gammaC;
            long firstExponent = m1Exponent + gammaOneMinusBExponent - gammaCExponent;

            // 第 2 項: Γ(b-1)/Γ(a) x^(1-b) M(a-b+1, 2-b, x)。c は上位と下位に分けて渡す
            double m2;
            long m2Exponent;
            if (c.Lo == 0)
            {
                m2 = Hypergeometric1F1Scaled(c.Hi, bPrime, x, out m2Exponent);
            }
            else if (bPrime < 0)
            {
                m2 = Hypergeometric1F1NegativeB(c.Hi, c.Lo, bPrime, x, out m2Exponent);
            }
            else
            {
                m2 = Hypergeometric1F1PositiveArgument(c.Hi, c.Lo, bPrime, x, out m2Exponent, out _);
            }

            double gammaBMinusOne = HypergeometricUScaledGamma(DoubleDoubleTwoSum(b, -1), out long gammaBMinusOneExponent);
            double gammaA = ScaledGammaSigned(a, out long gammaAExponent);
            var oneMinusB = DoubleDoubleTwoSum(1, -b);
            double power = ConfluentScaledPower(x, oneMinusB.Hi, out long powerExponent);
            if (oneMinusB.Lo != 0)
            {
                power *= 1 + oneMinusB.Lo * Math.Log(x);
            }

            double second = m2 * gammaBMinusOne / gammaA * power;
            long secondExponent = m2Exponent + gammaBMinusOneExponent - gammaAExponent + powerExponent;

            double result = ConfluentAlignPair(first, firstExponent, ref second, secondExponent, out exponent) + second;
            double magnitude = Math.Abs(result - second) + Math.Abs(second);
            if (result == 0 || !double.IsFinite(result) || !double.IsFinite(magnitude))
            {
                return double.NaN;
            }

            // 各項の相対誤差はおよそ 8ε（M と 3 つのガンマ関数、冪）
            error = 8 * MachineEpsilon * magnitude / Math.Abs(result) + 2 * MachineEpsilon;
            return result;
        }

        // double-double 引数の Γ(hi + lo) = mantissa · 2^exponent（lo は ψ(hi) で 1 次補正）
        private static double HypergeometricUScaledGamma((double Hi, double Lo) argument, out long exponent)
        {
            double gamma = ScaledGammaSigned(argument.Hi, out exponent);
            if (argument.Lo != 0)
            {
                gamma *= 1 + argument.Lo * Digamma(argument.Hi);
            }

            return gamma;
        }

        private static bool HypergeometricUInTemmeRegion(double b, double x)
        {
            double effectiveB = b >= 0.5 ? b : 2 - b;
            return x <= Math.Max(HypergeometricUTemmeMax, 0.5 * effectiveB);
        }

        // Miller 法による U(a, b, x)。U は a → +∞ で劣勢解のため、a₀ = a - ⌊a⌋ + 1 ∈ [1, 2) として
        // max(a, a₀) + extra から double-double の下向き漸化式で min(a, a₀) まで下げ、U(a₀) で正規化する。
        // 直接計算した U(a₀+1)/U(a₀) と比が一致しない場合は double.NaN を返す
        private static double HypergeometricUMiller(double a, double b, double x, out long exponent, out double error)
        {
            double a0 = a - Math.Floor(a) + 1;
            double steps = Math.Abs(Math.Floor(a) - 1);
            double upper = Math.Max(a, a0);
            exponent = 0;
            error = double.PositiveInfinity;
            if (steps > MaxIterations)
            {
                return double.NaN;
            }

            int iterations = HypergeometricURatioIterations(upper, b, x);
            if (iterations < 0 || steps + 2.0 * iterations + 50 > MaxIterations)
            {
                return double.NaN;
            }

            // 連分数の収束判定はまれに早すぎる（b > x で係数の符号が途中で変わる場合など）ので、出発点を遠ざけて
            // upper での比 U(upper+1)/U(upper) が double-double の精度で変わらなくなるまで繰り返す。
            // b > x では a ≲ b で見かけ上誤った比へ収束しうるため、出発点は少なくとも a ≈ b + x より上にとる
            double settled = Math.Max(b, 0) + x - upper;
            if (steps + 2.0 * iterations + 50 + Math.Max(settled, 0) > MaxIterations)
            {
                return double.NaN;
            }

            int extra = 2 * iterations + 50 + (int)Math.Ceiling(Math.Max(settled, 0));
            (double Hi, double Lo) current = (1.0, 0.0);
            (double Hi, double Lo) next = (0.0, 0.0);
            long currentExponent = 0;
            HypergeometricURecurrenceDownPair(upper, extra, b, x, ref current, ref next, ref currentExponent);
            double startError = double.PositiveInfinity;
            for (int attempt = 0; attempt < 4 && steps + 2.0 * extra <= MaxIterations; attempt++)
            {
                // 1 回目は少しだけ遠ざける（ほとんどの場合はこれで収束が確かめられる）。収束していなければ倍々に広げる
                extra = attempt == 0 ? extra + extra / 4 + 20 : 2 * extra;
                (double Hi, double Lo) trialCurrent = (1.0, 0.0);
                (double Hi, double Lo) trialNext = (0.0, 0.0);
                long trialExponent = 0;
                HypergeometricURecurrenceDownPair(upper, extra, b, x, ref trialCurrent, ref trialNext, ref trialExponent);
                var previousRatio = DoubleDoubleDivide(next, current);
                var trialRatio = DoubleDoubleDivide(trialNext, trialCurrent);
                var change = DoubleDoubleAdd(trialRatio, (-previousRatio.Hi, -previousRatio.Lo));
                double lastError = startError;
                startError = Math.Abs((change.Hi + change.Lo) / trialRatio.Hi);
                current = trialCurrent;
                next = trialNext;
                currentExponent = trialExponent;
                if (startError <= 1e-29 || startError > 0.25 * lastError)
                {
                    // 収束したか、丸めの水準で頭打ちになった
                    break;
                }
            }

            // 下側の区間を double-double と倍精度の両方で進めて比 w(a)/w(a₀) を比べる。
            // 倍精度のずれ δ に精度比 2⁻⁵³ を掛けて double-double の誤差目安とする（ずれが大きすぎれば線形範囲外のため棄却）
            double ratio = HypergeometricUMillerLowerRatio(a >= a0, upper, (int)steps, b, x, current, next, currentExponent, out exponent, out double millerBaseRatio);
            double check = HypergeometricUMillerLowerRatioDouble(a >= a0, upper, (int)steps, b, x, current.Hi, next.Hi, out long checkExponent);
            double baseValue = HypergeometricUBasePair(a0, b, x, out double baseNext, out long baseExponent, out double baseError);
            if (!double.IsFinite(ratio) || ratio == 0)
            {
                return double.NaN;
            }

            // Miller 法の比 w(a₀+1)/w(a₀) を、直接求めた U(a₀+1)/U(a₀) と照合する。b ≫ x で a₀ と b の間に U が下向きに劣勢になる区間があると、
            // 出発点をどこに置いても別の解に引き込まれて誤った比に収束するため、ここで検出する
            double directRatio = baseNext / baseValue;
            if (!(Math.Abs(millerBaseRatio - directRatio) <= 1e-10 * Math.Abs(directRatio)))
            {
                return double.NaN;
            }

            double deviation = Math.Abs(ratio - ScaleB(check, Math.Clamp(checkExponent - exponent, -4000, 4000))) / Math.Abs(ratio);
            double recurrenceError = deviation > 1e-3 || double.IsNaN(deviation) ? double.PositiveInfinity : 8 * MachineEpsilon * deviation;
            error = baseError + recurrenceError + 4 * startError + 2 * MachineEpsilon;
            exponent += baseExponent;
            return baseValue * ratio;
        }

        // HypergeometricUMiller の下側区間。upper から steps 段下げて w(a)/w(a₀) を仮数と指数で返す
        private static double HypergeometricUMillerLowerRatio(bool aAboveBase, double upper, int steps, double b, double x,
            (double Hi, double Lo) current, (double Hi, double Lo) next, long currentExponent, out long exponent, out double baseRatio)
        {
            var atUpper = current;
            long atUpperExponent = currentExponent;
            double upperRatio = (next.Hi + next.Lo) / (current.Hi + current.Lo);
            HypergeometricURecurrenceDownPair(upper - steps, steps, b, x, ref current, ref next, ref currentExponent);
            if (aAboveBase)
            {
                baseRatio = (next.Hi + next.Lo) / (current.Hi + current.Lo);
                exponent = atUpperExponent - currentExponent;
                return (atUpper.Hi + atUpper.Lo) / (current.Hi + current.Lo);
            }

            baseRatio = upperRatio;
            exponent = currentExponent - atUpperExponent;
            return (current.Hi + current.Lo) / (atUpper.Hi + atUpper.Lo);
        }

        // HypergeometricUMillerLowerRatio の倍精度版（誤差見積もり用）
        private static double HypergeometricUMillerLowerRatioDouble(bool aAboveBase, double upper, int steps, double b, double x,
            double current, double next, out long exponent)
        {
            double atUpper = current;
            long currentExponent = 0;
            double bottom = upper - steps;
            for (int k = 0; k < steps; k++)
            {
                double s = bottom + (steps - k);
                double previous = (2 * s - b + x) * current - s * (s - b + 1) * next;
                next = current;
                current = previous;
                double size = Math.Max(Math.Abs(current), Math.Abs(next));
                if (size > ConfluentRescaleLimit || (size < 1 / ConfluentRescaleLimit && size > 0))
                {
                    int scale = Math.ILogB(size);
                    current = Math.ScaleB(current, -scale);
                    next = Math.ScaleB(next, -scale);
                    currentExponent += scale;
                }
            }

            if (aAboveBase)
            {
                exponent = -currentExponent;
                return atUpper / current;
            }

            exponent = currentExponent;
            return current / atUpper;
        }

        // a < 2 の U(a, b, x)。a₀ ∈ [1, 2) での U(a₀), U(a₀+1) から独立な 2 解 g₁, g₂ を下げて合成し、打ち消しから誤差を見積もる
        private static double HypergeometricUFromBase(double a, double b, double x, out long exponent, out double error)
        {
            double steps = 1 - Math.Floor(a);
            double a0 = a + steps;
            if (steps > MaxIterations)
            {
                exponent = 0;
                error = double.PositiveInfinity;
                return double.NaN;
            }

            double current = HypergeometricUBasePair(a0, b, x, out double next, out long baseExponent, out double baseError);
            if (steps == 0)
            {
                exponent = baseExponent;
                error = baseError;
                return current;
            }

            (double Hi, double Lo) u1 = (1.0, 0.0);
            (double Hi, double Lo) v1 = (0.0, 0.0);
            long exponent1 = 0;
            HypergeometricURecurrenceDownPair(a, (int)steps, b, x, ref u1, ref v1, ref exponent1);
            (double Hi, double Lo) u2 = (0.0, 0.0);
            (double Hi, double Lo) v2 = (1.0, 0.0);
            long exponent2 = 0;
            HypergeometricURecurrenceDownPair(a, (int)steps, b, x, ref u2, ref v2, ref exponent2);

            // 2 つの寄与を共通の指数にそろえて double-double で足す
            var term1 = DoubleDoubleMultiply(u1, (current, 0.0));
            var term2 = DoubleDoubleMultiply(u2, (next, 0.0));
            long common = Math.Max(exponent1, exponent2);
            term1 = ConfluentScaleDoubleDouble(term1, exponent1 - common);
            term2 = ConfluentScaleDoubleDouble(term2, exponent2 - common);
            var sum = DoubleDoubleAdd(term1, term2);
            double value = sum.Hi + sum.Lo;
            double magnitude = Math.Abs(term1.Hi) + Math.Abs(term2.Hi);
            error = value == 0 ? double.PositiveInfinity : (baseError + 4 * MachineEpsilon) * magnitude / Math.Abs(value) + 2 * MachineEpsilon;
            exponent = baseExponent + common;
            return value;
        }

        // s = a + steps での U(s)（u）と U(s+1)（v）から U(s-1) = (2s - b + x) U(s) - s(s - b + 1) U(s+1)（DLMF 13.3.7）を
        // double-double で進めて U(a), U(a+1) を求める
        private static void HypergeometricURecurrenceDownPair(double a, int steps, double b, double x,
            ref (double Hi, double Lo) u, ref (double Hi, double Lo) v, ref long exponent)
        {
            var shift = DoubleDoubleTwoSum(x, -b);
            for (int k = 0; k < steps; k++)
            {
                var s = DoubleDoubleTwoSum(a, steps - k);
                var linear = DoubleDoubleAdd((2 * s.Hi, 2 * s.Lo), shift);
                var quadratic = DoubleDoubleMultiply(s, DoubleDoubleAdd(s, DoubleDoubleTwoSum(1, -b)));
                var previous = DoubleDoubleAdd(DoubleDoubleMultiply(linear, u), DoubleDoubleMultiply((-quadratic.Hi, -quadratic.Lo), v));
                v = u;
                u = previous;
                double size = Math.Max(Math.Abs(u.Hi), Math.Abs(v.Hi));
                if (size > ConfluentRescaleLimit || (size < 1 / ConfluentRescaleLimit && size > 0))
                {
                    int scale = Math.ILogB(size);
                    u = ConfluentScaleDoubleDouble(u, -scale);
                    v = ConfluentScaleDoubleDouble(v, -scale);
                    exponent += scale;
                }
            }
        }

        // 1 ≤ a₀ < 2 での U(a₀, b, x)（戻り値）と U(a₀+1, b, x)（next）を共通の指数で返す（漸近展開・Temme 級数・数値積分の順）
        private static double HypergeometricUBasePair(double a0, double b, double x, out double next, out long exponent, out double error)
        {
            if (x >= HypergeometricUAsymptoticMin
                && HypergeometricUTryAsymptotic(a0, b, x, out double current, out long currentExponent)
                && HypergeometricUTryAsymptotic(a0 + 1, b, x, out next, out long nextExponent))
            {
                error = 4 * MachineEpsilon;
                return ConfluentAlignPair(current, currentExponent, ref next, nextExponent, out exponent);
            }

            if (HypergeometricUInTemmeRegion(b, x))
            {
                current = HypergeometricUTemmeAny(a0, b, x, out currentExponent, out double currentError);
                next = HypergeometricUTemmeAny(a0 + 1, b, x, out nextExponent, out double nextError);
                if (Math.Max(currentError, nextError) <= 64 * MachineEpsilon)
                {
                    error = Math.Max(currentError, nextError);
                    return ConfluentAlignPair(current, currentExponent, ref next, nextExponent, out exponent);
                }
            }

            // 台形則の誤差は数 ε（積分値の大きさに対する丸め）
            error = 8 * MachineEpsilon;
            return HypergeometricUQuadrature(a0, b, x, out next, out exponent);
        }

        // 2 つの仮数・指数の組を大きい方の指数にそろえ、1 つ目の仮数を返す
        private static double ConfluentAlignPair(double first, long firstExponent, ref double second, long secondExponent, out long exponent)
        {
            exponent = Math.Max(firstExponent, secondExponent);
            second = ScaleB(second, secondExponent - exponent);
            return ScaleB(first, firstExponent - exponent);
        }

        // U(a+1, b, x)/U(a, b, x) の 3 項漸化式連分数（修正 Lentz 法）が倍精度で収束するまでの項数（収束しなければ -1）
        private static int HypergeometricURatioIterations(double a, double b, double x)
        {
            double f = 2 * (a + 1) - b + x;
            if (f == 0)
            {
                f = Tiny;
            }

            double c = f;
            double d = 0;
            for (int j = 1; j < MaxIterations; j++)
            {
                double an = -(a + j) * (a + j + 1 - b);
                double bn = 2 * (a + j + 1) - b + x;
                d = bn + an * d;
                d = d == 0 ? Tiny : 1 / d;
                c = bn + an / c;
                if (c == 0)
                {
                    c = Tiny;
                }

                double delta = c * d;
                f *= delta;
                if (Math.Abs(delta - 1) <= MachineEpsilon)
                {
                    return j;
                }
            }

            return -1;
        }

        // 大きな x での漸近展開 U ~ x^(-a) Σ (a)ₛ(a-b+1)ₛ/(s! (-x)ˢ)（DLMF 13.7.3）。倍精度まで収束しなければ false を返す
        private static bool HypergeometricUTryAsymptotic(double a, double b, double x, out double mantissa, out long exponent)
        {
            mantissa = 0;
            exponent = 0;
            double c = a - b + 1;
            if (Math.Abs(a * c) >= x)
            {
                return false;
            }

            double sum = 1;
            double term = 1;
            bool converged = false;
            for (int s = 0; s < 1000; s++)
            {
                double next = -term * (a + s) * (c + s) / ((s + 1) * x);
                if (next == 0)
                {
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

            mantissa = ConfluentScaledPower(x, -a, out exponent) * sum;
            return true;
        }

        // 多項式 U(-n, b, x) = (-1)ⁿ (b)ₙ M(-n, b, x)（DLMF 13.2.7）。
        // b = -m（m < n）では (b)ₙ = 0 と M の極が打ち消し合うため Kummer 変換 x^(1+m) U(m+1-n, m+2, x) を用いる
        private static double HypergeometricUPolynomial(double a, double b, double x, out long exponent)
        {
            double n = -a;
            if (IsNonPositiveInteger(b) && -b < n)
            {
                double inner = HypergeometricUPolynomial(a - b + 1, 2 - b, x, out long innerExponent);
                double power = ConfluentScaledPower(x, 1 - b, out long powerExponent);
                exponent = innerExponent + powerExponent;
                return inner * power;
            }

            if (n > MaxIterations)
            {
                exponent = 0;
                return double.NaN;
            }

            double m = Hypergeometric1F1Polynomial(a, b, x, out exponent);
            double pochhammer = ConfluentScaledPochhammerPowerOverFactorial(b, 1, (int)n, out long pochhammerExponent);

            // ConfluentScaledPochhammerPowerOverFactorial は (b)ₙ/n! を返すので n! を掛け戻す
            double factorial = ScaledFactorial((int)n, out long factorialExponent);
            exponent += pochhammerExponent + factorialExponent;
            double sign = IsOddInteger(n) ? -1 : 1;
            return sign * m * pochhammer * factorial;
        }

        // HypergeometricUTemme（b ≥ 1/2）を b < 1/2 では Kummer 変換 U(a, b, x) = x^(1-b) U(a-b+1, 2-b, x) 経由で呼び出す
        private static double HypergeometricUTemmeAny(double a, double b, double x, out long exponent, out double error)
        {
            if (b >= 0.5)
            {
                return HypergeometricUTemme(a, 0, b, x, out exponent, out error);
            }

            // b' = 2 - b は丸められることがあり、内側の x^(1-b') と外側の x^(1-b) がずれると小さな x で |ln x| 倍の誤差になる。
            // 外側も丸めた b' で x^(b'-1) とし（b' - 1 は正確）、a' = a + b' - 1 は上位と下位に分けて渡す（b を b̃ = 2 - b' に置き換えた U を求めることになる）
            double bPrime = 2 - b;
            var shifted = DoubleDoubleTwoSum(a, bPrime - 1);
            double inner = HypergeometricUTemme(shifted.Hi, shifted.Lo, bPrime, x, out long innerExponent, out error);
            double power = ConfluentScaledPower(x, bPrime - 1, out long powerExponent);
            exponent = innerExponent + powerExponent;
            return inner * power;
        }

        // 小さな x での Temme 級数による U(a, b, x)（b ≥ 1/2、a および a - b + 1 は 0 以下の整数でない）。
        // b = n + 1 + ε（n ≥ 0 整数, |ε| ≤ 1/2）として接続公式の 2 級数の xᵏ 項を dₖ = (hₖ - gₖ)/ε にまとめ、ε → 0 でも連続に評価する
        private static double HypergeometricUTemme(double a, double aLo, double b, double x, out long exponent, out double error)
        {
            // b = 1/2 は n = 0、ε = -1/2 とする（偶数丸めだと n = -1 になる）
            double m = Math.Round(b, MidpointRounding.AwayFromZero);
            double eps = b - m;
            exponent = 0;
            error = double.PositiveInfinity;
            if (m - 1 > MaxIterations)
            {
                return double.NaN;
            }

            int n = (int)m - 1;
            double logX = Math.Log(x);

            // (1/Γ(1+ε) - 1/Γ(1-ε))/ε = -2 gam1、B = 1/Γ(1-ε)
            TemmeGammas(eps, out double gam1, out _, out _, out double reciprocalGammaMinus);
            double gammaDifference = -2 * gam1;

            // P = n!/(1+ε)ₙ = exp(-Σ ln(1 + ε/j)) と (P - 1)/ε
            double logSum = 0;
            double logSumOverEps = 0;
            for (int j = 1; j <= n; j++)
            {
                double t = eps / j;
                logSum += Log1p(t);
                logSumOverEps += ConfluentLogRatio(t) / j;
            }

            double p = Math.Exp(-logSum);
            double pMinusOne = -ConfluentExpRatio(-logSum) * logSumOverEps;

            // G = Γ(a-ε)/Γ(a) と (1 - G)/ε
            HypergeometricUGammaShift(a, eps, out double g, out double oneMinusG);
            if (aLo != 0)
            {
                // a の下位 aLo の分の 1 次の補正: ∂G/∂a = G (ψ(a-ε) - ψ(a))。a が負で大きいと ψ の差が大きく、無視すると 1e-13 程度の誤差になる
                double digammaDifferenceOverEps = Math.Abs(eps) > 1e-4 ? (Digamma(a - eps) - Digamma(a)) / eps : -Trigamma(a);
                g += aLo * g * eps * digammaDifferenceOverEps;
                oneMinusG -= aLo * g * digammaDifferenceOverEps;
            }

            // Z = x^(-ε) と (1 - Z)/ε。|ε ln x| が大きいと exp(-ε ln x) は ε ln x の丸めの分だけ誤差が増えるので、べきを直接求める
            double epsLogX = eps * logX;
            double z;
            double oneMinusZ;
            if (Math.Abs(epsLogX) <= 0.5)
            {
                z = Math.Exp(-epsLogX);
                oneMinusZ = logX * ConfluentExpRatio(-epsLogX);
            }
            else
            {
                z = ScaleB(ConfluentScaledPower(x, -eps, out long zExponent), zExponent);
                oneMinusZ = (1 - z) / eps;
            }

            // d₀ n! = (A - B)/ε P + B (P - 1)/ε + B (1 - G)/ε + B G (1 - Z)/ε、g₀ n! = G B Z。
            // d₀ の各項は ε → 0 で 2γ、-Hₙ、ψ(a)、ln x になり（DLMF 13.2.9 の対数項の係数）、打ち消し合うことがあるので大きさも記録する
            double d0 = gammaDifference * p + reciprocalGammaMinus * (pMinusOne + oneMinusG + g * oneMinusZ);
            double d0Size = Math.Abs(gammaDifference * p) + Math.Abs(reciprocalGammaMinus) * (Math.Abs(pMinusOne) + Math.Abs(oneMinusG) + Math.Abs(g * oneMinusZ));
            double g0 = g * reciprocalGammaMinus * z;

            // Σ xᵏ dₖ = d₀ Σ Aₖ + g₀ Σ Bₖ。A は (d, g) = (1, 0)、B は (0, 1) から始めた列で、係数はすべて double-double で正確に作る:
            // r_h = (a+k)/((b+k)(k+1))、r_g = (a-b+m+k)/((k+1-b+m)(m+k))、
            // (r_h - r_g)/ε = ((k+1)(b-a) - (a+k)(m+k))/((b+k)(k+1)(k+1-b+m)(m+k))
            var aMinusB = DoubleDoubleAdd(DoubleDoubleTwoSum(a, -b), (aLo, 0.0));
            var bMinusA = (-aMinusB.Hi, -aMinusB.Lo);
            (double Hi, double Lo) termA = (1.0, 0.0);
            (double Hi, double Lo) termB = (0.0, 0.0);
            (double Hi, double Lo) gB = (1.0, 0.0);
            (double Hi, double Lo) sumA = (1.0, 0.0);
            (double Hi, double Lo) sumB = (0.0, 0.0);
            long sumExponent = 0;
            int k = 0;
            for (; k < MaxIterations; k++)
            {
                var pk = DoubleDoubleAdd(DoubleDoubleTwoSum(a, k), (aLo, 0.0));
                double qk = k + 1.0;
                double rk = m + k;
                var bk = DoubleDoubleTwoSum(b, k);
                var qMinusEps = DoubleDoubleTwoSum(qk + m, -b);
                var gNumerator = DoubleDoubleAdd(aMinusB, (rk, 0.0));
                var hDenominator = DoubleDoubleMultiply(bk, (qk, 0.0));
                var gDenominator = DoubleDoubleMultiply(qMinusEps, (rk, 0.0));
                var rh = DoubleDoubleDivide(DoubleDoubleMultiply(pk, (x, 0.0)), hDenominator);
                var rg = DoubleDoubleDivide(DoubleDoubleMultiply(gNumerator, (x, 0.0)), gDenominator);
                var diffNumerator = DoubleDoubleAdd(DoubleDoubleMultiply(bMinusA, (qk, 0.0)), DoubleDoubleMultiply((-pk.Hi, -pk.Lo), (rk, 0.0)));
                var difference = DoubleDoubleDivide(DoubleDoubleMultiply(diffNumerator, (x, 0.0)), DoubleDoubleMultiply(hDenominator, gDenominator));
                termA = DoubleDoubleMultiply(rh, termA);
                termB = DoubleDoubleAdd(DoubleDoubleMultiply(rh, termB), DoubleDoubleMultiply(difference, gB));
                gB = DoubleDoubleMultiply(rg, gB);
                sumA = DoubleDoubleAdd(sumA, termA);
                sumB = DoubleDoubleAdd(sumB, termB);
                double size = Math.Max(Math.Abs(sumA.Hi) + Math.Abs(sumB.Hi), Math.Abs(termA.Hi) + Math.Abs(termB.Hi) + Math.Abs(gB.Hi));
                if (k + 1 > -a && Math.Abs(rh.Hi) < 0.5
                    && Math.Abs(termA.Hi) * Math.Abs(d0) + (Math.Abs(termB.Hi) + Math.Abs(gB.Hi)) * Math.Abs(g0)
                        <= 0.0625 * MachineEpsilon * Math.Abs(d0 * sumA.Hi + g0 * sumB.Hi))
                {
                    break;
                }

                if (size > ConfluentRescaleLimit)
                {
                    int shift = Math.ILogB(size);
                    termA = ConfluentScaleDoubleDouble(termA, -shift);
                    termB = ConfluentScaleDoubleDouble(termB, -shift);
                    gB = ConfluentScaleDoubleDouble(gB, -shift);
                    sumA = ConfluentScaleDoubleDouble(sumA, -shift);
                    sumB = ConfluentScaleDoubleDouble(sumB, -shift);
                    sumExponent += shift;
                }
            }

            if (k >= MaxIterations)
            {
                return double.NaN;
            }

            double partA = d0 * (sumA.Hi + sumA.Lo);
            double partB = g0 * (sumB.Hi + sumB.Lo);
            double sum = partA + partB;
            double sumError = d0Size * Math.Abs(sumA.Hi) + Math.Abs(partB);

            // U_reg = (-1)^(n+1) (πε/sin πε) Σ / (n! Γ(a-b+1))
            double piEps = eps == 0 ? 1 : Math.PI * eps / double.SinPi(eps);
            double factorial = ScaledFactorial(n, out long factorialExponent);
            // Γ(a-b+1) の引数の丸め（極の近くでは ψ 倍に拡大される）は 1 次の補正で取り除く
            var c = DoubleDoubleAdd(aMinusB, (1.0, 0.0));
            double gammaC = ScaledGammaSigned(c.Hi, out long gammaCExponent);
            if (c.Lo != 0)
            {
                gammaC *= 1 + c.Lo * Digamma(c.Hi);
            }

            double scale = ((n & 1) == 0 ? -1 : 1) * piEps / (factorial * gammaC);
            double regular = scale * sum;
            double regularError = Math.Abs(scale) * sumError;
            long regularExponent = sumExponent - factorialExponent - gammaCExponent;

            // U_sing = x^(1-b)/Γ(a) Σ_(j<n) tⱼ、t₀ = Γ(b-1)、tⱼ₊₁ = -tⱼ (a-b+1+j) x/((b-2-j)(j+1))（double-double で足す）
            double singular = 0;
            long singularExponent = 0;
            if (n > 0)
            {
                double t0 = ScaledGammaSigned(b - 1, out long termExponent);
                (double Hi, double Lo) term = (t0, 0.0);
                (double Hi, double Lo) singularSum = term;
                for (int j = 0; j < n - 1; j++)
                {
                    var numerator = DoubleDoubleAdd(aMinusB, (1.0 + j, 0.0));
                    var denominator = DoubleDoubleMultiply(DoubleDoubleTwoSum(b, -(2.0 + j)), (j + 1.0, 0.0));
                    term = DoubleDoubleDivide(DoubleDoubleMultiply(term, DoubleDoubleMultiply(numerator, (-x, 0.0))), denominator);
                    singularSum = DoubleDoubleAdd(singularSum, term);
                    double size = Math.Max(Math.Abs(singularSum.Hi), Math.Abs(term.Hi));
                    if (size > ConfluentRescaleLimit || size < 1 / ConfluentRescaleLimit)
                    {
                        int shift = Math.ILogB(size);
                        term = ConfluentScaleDoubleDouble(term, -shift);
                        singularSum = ConfluentScaleDoubleDouble(singularSum, -shift);
                        termExponent += shift;
                    }
                }

                double power = ConfluentScaledPower(x, 1 - b, out long powerExponent);
                double gammaA = ScaledGammaSigned(a, out long gammaAExponent);
                if (aLo != 0)
                {
                    gammaA *= 1 + aLo * Digamma(a);
                }

                singular = (singularSum.Hi + singularSum.Lo) * power / gammaA;
                singularExponent = termExponent + powerExponent - gammaAExponent;
            }

            // 2 つの部分を共通の指数にそろえて足す
            double result = ConfluentAlignPair(regular, regularExponent, ref singular, singularExponent, out exponent) + singular;
            double alignedRegularError = ScaleB(regularError, regularExponent - exponent);
            if (result == 0 || !double.IsFinite(result))
            {
                error = double.PositiveInfinity;
                return result;
            }

            error = 4 * MachineEpsilon * (alignedRegularError + Math.Abs(singular)) / Math.Abs(result) + 2 * MachineEpsilon;
            return result;
        }

        // G = Γ(a-ε)/Γ(a) と (1 - G)/ε（|ε| ≤ 1/2）を桁落ちなく計算する。
        // L/ε を y = a + m ≥ 10 へのシフト積とスターリング差分で求め、(1 - G)/ε = -(expm1(L)/L)(L/ε) とする（極近傍因子は別途 double-double で評価）
        private static void HypergeometricUGammaShift(double a, double eps, out double g, out double oneMinusGOverEps)
        {
            int shift = a < 10 ? (int)Math.Ceiling(10 - a) : 0;
            if (shift > MaxIterations)
            {
                g = GammaRatio(a - eps, a);
                oneMinusGOverEps = (1 - g) / eps;
                return;
            }

            double logOverEps = 0;
            double nearPoleLog = 0;
            double nearPoleSign = 1;
            bool nearPole = false;
            for (int i = 0; i < shift; i++)
            {
                // (a+i)/(a+i-ε) = 1/(1+t)、t = -ε/(a+i)。ln の ε による商は log1p(t)/t /(a+i)
                double ai = a + i;
                double t = -eps / ai;
                if (Math.Abs(t) > 0.5)
                {
                    // a + i が極（0）の近くで因子が 1 から大きく離れる（符号が変わることもある）。
                    // a - ε を倍精度で丸めると ψ(a - ε) 倍（極の近くで大きい）の誤差になるので、a + i（正確）から ε を double-double で引いて比を作り、
                    // ε で割らない対数として別に足す（このとき G は 1 から離れているので (1 - G)/ε は直接求めてよい）
                    var denominator = DoubleDoubleTwoSum(ai, -eps);
                    double ratio = ai / denominator.Hi / (1 + denominator.Lo / denominator.Hi);
                    nearPole = true;
                    nearPoleLog += Math.Log(Math.Abs(ratio));
                    if (ratio < 0)
                    {
                        nearPoleSign = -nearPoleSign;
                    }

                    continue;
                }

                logOverEps += ConfluentLogRatio(t) / ai;
            }

            // ln Γ(y-ε) - ln Γ(y) = (y-1/2) log1p(-ε/y) - ε ln(y-ε) + ε + δ(y-ε) - δ(y)
            double y = a + shift;
            double u = -eps / y;
            double w = Log1p(u);
            double wOverEps = -ConfluentLogRatio(u) / y;
            logOverEps += -(y - 0.5) / y * ConfluentLogRatio(u) - Math.Log(y - eps) + 1;

            // δ(y-ε) - δ(y) = Σ cₖ y^(1-2k) expm1((1-2k) w)
            double inverse = 1 / y;
            double inverseSquared = inverse * inverse;
            double powerOfY = inverse;
            for (int k = 0; k < StirlingSeriesCoefficients.Length; k++)
            {
                double p = 1 - 2 * (k + 1);
                double contribution = StirlingSeriesCoefficients[k] * powerOfY * ConfluentExpRatio(p * w) * p * wOverEps;
                logOverEps += contribution;
                if (Math.Abs(contribution) <= 0.25 * MachineEpsilon * Math.Abs(logOverEps))
                {
                    break;
                }

                powerOfY *= inverseSquared;
            }

            double l = eps * logOverEps;
            if (nearPole)
            {
                g = nearPoleSign * Math.Exp(l + nearPoleLog);
                oneMinusGOverEps = (1 - g) / eps;
                return;
            }

            g = Math.Exp(l);
            oneMinusGOverEps = -ConfluentExpRatio(l) * logOverEps;
        }

        // log1p(t)/t（t = 0 では 1）
        private static double ConfluentLogRatio(double t)
        {
            if (Math.Abs(t) < 1e-4)
            {
                return 1 - t * (0.5 - t * (1.0 / 3 - 0.25 * t));
            }

            return Log1p(t) / t;
        }

        // expm1(u)/u（u = 0 では 1）
        private static double ConfluentExpRatio(double u) => u == 0 ? 1 : Expm1(u) / u;

        // 積分表示 U(a, b, x) = 1/Γ(a) ∫₀^∞ e^(-xt) t^(a-1) (1+t)^(b-a-1) dt（a₀ ≥ 1, x > 0）を
        // 二重指数型公式 t = exp((π/2) sinh u) の台形則で評価し、U(a₀, b, x)（戻り値）と U(a₀+1, b, x)（next）を同時に求める
        private static double HypergeometricUQuadrature(double a0, double b, double x, out double next, out long exponent)
        {
            var nodes = HypergeometricUQuadratureNodes.Instance;
            int half = HypergeometricUQuadratureNodes.HalfCount;
            double reference = double.NegativeInfinity;
            double sum0 = 0;
            double sum1 = 0;
            double previous0 = double.NaN;
            double previous1 = double.NaN;
            double integral0 = 0;
            double integral1 = 0;
            double c = b - a0 - 1;
            int lowest = -half;
            int highest = half;
            for (int level = 0; level < HypergeometricUQuadratureNodes.Levels; level++)
            {
                int stride = HypergeometricUQuadratureNodes.BaseStride >> level;
                int step = level == 0 ? stride : 2 * stride;
                int newLowest = int.MaxValue;
                int newHighest = int.MinValue;
                for (int sign = -1; sign <= 1; sign += 2)
                {
                    // 0 段目は u = 0 と ±stride の倍数、それ以降は前の段の節点の中点（stride の奇数倍）
                    int first = level == 0 ? (sign < 0 ? stride : 0) : stride;
                    for (int i = first; i <= half; i += step)
                    {
                        int index = sign * i;
                        if (sign * index > sign * (sign < 0 ? lowest : highest))
                        {
                            // 外側の境界を越えたら打ち切る
                            break;
                        }

                        if (index < lowest || index > highest)
                        {
                            // 内側の境界までは飛ばす
                            continue;
                        }

                        double t = nodes.T[index + half];
                        double log = -x * t + a0 * nodes.LogT[index + half] + c * Log1p(t) + nodes.LogWeight[index + half];
                        if (log > reference)
                        {
                            // 基準を更新して既存の和を縮める
                            double scale = double.IsNegativeInfinity(reference) ? 0 : Math.Exp(reference - log);
                            sum0 *= scale;
                            sum1 *= scale;
                            reference = log;
                        }

                        if (log >= reference - 50)
                        {
                            newLowest = Math.Min(newLowest, index);
                            newHighest = Math.Max(newHighest, index);
                        }

                        double value = Math.Exp(log - reference);
                        sum0 += value;
                        sum1 += value * t / (1 + t);
                    }
                }

                // 寄与が e⁻⁵⁰ 以下の外側は次の段の節点を省く（粗い刻み 2 つ分の余裕を残す）
                if (newLowest <= newHighest)
                {
                    lowest = Math.Max(lowest, newLowest - 2 * stride);
                    highest = Math.Min(highest, newHighest + 2 * stride);
                }

                double h = HypergeometricUQuadratureNodes.FinestStep * stride;
                integral0 = h * sum0;
                integral1 = h * sum1;
                if (level >= 3 && Math.Abs(integral0 - previous0) <= 1e-9 * Math.Abs(integral0)
                    && Math.Abs(integral1 - previous1) <= 1e-9 * Math.Abs(integral1))
                {
                    break;
                }

                previous0 = integral0;
                previous1 = integral1;
            }

            double scaleFactor = ConfluentScaledExp(reference, 0, out exponent);
            double gamma = Gamma(a0);
            next = integral1 * scaleFactor / (a0 * gamma);
            return integral0 * scaleFactor / gamma;
        }

        // HypergeometricUQuadrature の節点表（u = jh, h = 1/128, |u| ≤ 4.5）
        private sealed class HypergeometricUQuadratureNodes
        {
            public const int Levels = 6;
            public const int BaseStride = 32;
            public const double FinestStep = 1.0 / 128;
            public const int HalfCount = 576;

            public static readonly HypergeometricUQuadratureNodes Instance = new();

            public readonly double[] T = new double[2 * HalfCount + 1];
            public readonly double[] LogT = new double[2 * HalfCount + 1];
            public readonly double[] LogWeight = new double[2 * HalfCount + 1];

            private HypergeometricUQuadratureNodes()
            {
                for (int i = -HalfCount; i <= HalfCount; i++)
                {
                    double u = i * FinestStep;
                    double s = 0.5 * Math.PI * Math.Sinh(u);
                    T[i + HalfCount] = Math.Exp(s);
                    LogT[i + HalfCount] = s;
                    LogWeight[i + HalfCount] = Math.Log(0.5 * Math.PI * Math.Cosh(u));
                }
            }
        }
    }
}
