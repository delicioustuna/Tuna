namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        /// <summary>
        /// Computes the shape derivative of the natural logarithm of the regularized lower incomplete gamma function.
        /// <para>正則化下側不完全ガンマ関数の自然対数を形状パラメータで微分します。</para>
        /// </summary>
        /// <param name="a">The shape parameter, greater than zero. / 正の形状パラメータ。</param>
        /// <param name="x">The nonnegative argument. / 非負の引数。</param>
        /// <returns>The derivative <c>∂ ln P(a, x) / ∂a</c>, or <see cref="double.NaN"/> for arguments outside the domain. / 形状微分 <c>∂ ln P(a, x) / ∂a</c>。定義域外では <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns negative infinity when <paramref name="x"/> is zero or only <paramref name="a"/> is positive infinity, and negative zero when only <paramref name="x"/> is positive infinity.</para>
        /// <para><paramref name="x"/> が 0 の場合、または <paramref name="a"/> だけが正の無限大の場合は <see cref="double.NegativeInfinity"/> を返します。<paramref name="x"/> だけが正の無限大の場合は -0.0 を返します。</para>
        /// </remarks>
        /// <seealso cref="LogRegularizedGammaP"/>
        public static double LogRegularizedGammaPShapeDerivative(double a, double x) =>
            LogRegularizedGammaShapeDerivative(a, x, upper: false);

        /// <summary>
        /// Computes the shape derivative of the natural logarithm of the regularized upper incomplete gamma function.
        /// <para>正則化上側不完全ガンマ関数の自然対数を形状パラメータで微分します。</para>
        /// </summary>
        /// <param name="a">The shape parameter, greater than zero. / 正の形状パラメータ。</param>
        /// <param name="x">The nonnegative argument. / 非負の引数。</param>
        /// <returns>The derivative <c>∂ ln Q(a, x) / ∂a</c>, or <see cref="double.NaN"/> for arguments outside the domain. / 形状微分 <c>∂ ln Q(a, x) / ∂a</c>。定義域外では <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns zero when <paramref name="x"/> is zero or only <paramref name="a"/> is positive infinity, and positive infinity when only <paramref name="x"/> is positive infinity.</para>
        /// <para><paramref name="x"/> が 0 の場合、または <paramref name="a"/> だけが正の無限大の場合は 0 を返します。<paramref name="x"/> だけが正の無限大の場合は <see cref="double.PositiveInfinity"/> を返します。</para>
        /// </remarks>
        /// <seealso cref="LogRegularizedGammaQ"/>
        public static double LogRegularizedGammaQShapeDerivative(double a, double x) =>
            LogRegularizedGammaShapeDerivative(a, x, upper: true);

        private static double LogRegularizedGammaShapeDerivative(double a, double x, bool upper)
        {
            if (TryIncompleteGammaSpecialCase(a, x, out double p, out _))
            {
                if (double.IsNaN(p))
                {
                    return double.NaN;
                }

                return p == 0 ? (upper ? 0 : double.NegativeInfinity)
                              : (upper ? double.PositiveInfinity : -0.0);
            }

            // a → 0 で Q = a E₁(x) + O(a²)、∂P/∂a = -E₁(x) + O(a)。
            // 1/a がオーバーフローする場合も下側の導関数は有限なので、別々に返す。
            if (a < LogGammaTinyA)
            {
                if (upper)
                {
                    return 1 / a;
                }

                return x <= 700 ? -ExpIntegralE(1, x)
                    : ScaleGammaShapeExponential((-x, 0), -UpperGammaFraction(0, x));
            }

            if (InTemmeGammaRegion(a, x))
            {
                return TemmeGammaShapeDerivative(a, x, upper);
            }

            if (x < 1.1 && !UseLowerGammaSeriesForSmallX(a, x))
            {
                SmallShapeGammaUpperDerivative(a, x, out double smallShapeLogQ, out double derivative);
                return upper ? derivative : ComplementGammaLogDerivative(smallShapeLogQ, derivative);
            }

            if (x - a < 1)
            {
                double sum = LowerGammaShapeSeries(a, x, out double derivative);
                if (!upper)
                {
                    return derivative;
                }

                double logP = LogGammaPrefix(a, x) - Math.Log(a) + Math.Log(sum);
                double result = ComplementGammaLogDerivative(logP, derivative);
                return UsePreciseGammaShapeSubnormal(logP, derivative, result)
                    ? ScaleGammaShapeExponential(GammaShapePreciseLogPrefix(a, x, lower: true), -sum * derivative)
                    : result;
            }

            double fraction = UpperGammaShapeFraction(a, x, out double fractionDerivative);
            double upperDerivative = GammaShapePrefixDerivative(a, x, lower: false) + fractionDerivative;
            if (upper)
            {
                return upperDerivative;
            }

            double logQ = LogGammaPrefix(a, x) + Math.Log(fraction);
            double lowerResult = ComplementGammaLogDerivative(logQ, upperDerivative);
            return UsePreciseGammaShapeSubnormal(logQ, upperDerivative, lowerResult)
                ? ScaleGammaShapeExponential(GammaShapePreciseLogPrefix(a, x, lower: false), -fraction * upperDerivative)
                : lowerResult;
        }

        // 補確率が 1 に丸められる範囲では、非正規化数の丸め精度を保つために最後に 2 の冪でスケーリングする。
        private static bool UsePreciseGammaShapeSubnormal(double logTail, double derivative, double result) =>
            logTail < -40 && Math.Abs(result) < 2 * MinNormal
            && (result != 0 || logTail + Math.Log(Math.Abs(derivative)) > -746);

        private static double ScaleGammaShapeExponential((double Hi, double Lo) exponent, double factor)
        {
            double mantissa = ConfluentScaledExp(exponent.Hi, exponent.Lo, out long scale);
            // この呼び出しで int の範囲を超える指数は、極端に遠い裾の負の指数だけ。
            return scale < int.MinValue ? Math.CopySign(0, factor) : Math.ScaleB(mantissa * factor, (int)scale);
        }

        private static (double Hi, double Lo) GammaShapePreciseLogPrefix(double a, double x, bool lower)
        {
            var gammaArgument = lower ? DoubleDoubleTwoSum(a, 1) : (Hi: a, Lo: 0.0);
            var gamma = PrecisePositiveLogGamma(gammaArgument);
            var power = DoubleDoubleMultiply((a, 0), DoubleDoubleLog((x, 0)));
            return DoubleDoubleAdd(DoubleDoubleAdd(power, (-x, 0)), (-gamma.Hi, -gamma.Lo));
        }

        private static (double Hi, double Lo) PrecisePositiveLogGamma((double Hi, double Lo) argument)
        {
            if (argument.Hi >= 1.25)
            {
                return DoubleDoublePositiveLogGamma(argument);
            }

            // 小さい a の Γ は漸化式で求め、a が丸めで失われないよう対数を分ける。
            var shifted = DoubleDoublePositiveLogGamma(DoubleDoubleAdd(argument, (1, 0)));
            var logarithm = DoubleDoubleLog(argument);
            return DoubleDoubleAdd(shifted, (-logarithm.Hi, -logarithm.Lo));
        }

        // 対数の導関数の比を使う。分子の確率が 0 に丸まっても、導関数との積は表現できることがある。
        private static double ComplementGammaLogDerivative(double logTail, double derivative)
        {
            double logRatio = logTail - Log1mExp(logTail);
            if (logRatio > -700)
            {
                return -Math.Exp(logRatio) * derivative;
            }

            return Math.CopySign(Math.Exp(logRatio + Math.Log(Math.Abs(derivative))), -derivative);
        }

        // 下側級数は 1/a を前因子に移す。このため対数微分の前因子に ψ(a+1) が現れる。
        private static double LowerGammaShapeSeries(double a, double x, out double derivative)
        {
            double term = 1;
            double sum = 1;
            double reciprocals = 0;
            double weighted = 0;
            for (int n = 1; n < MaxIterations; n++)
            {
                double inverse = 1 / (a + n);
                term *= x * inverse;
                sum += term;
                reciprocals += inverse;
                double weight = term * reciprocals;
                weighted += weight;
                if (term <= sum * Epsilon && weight <= weighted * Epsilon)
                {
                    break;
                }
            }

            derivative = GammaShapePrefixDerivative(a, x, lower: true) - weighted / sum;
            return sum;
        }

        // 修正 Lentz 法の c と d の対数微分を同時に計算する。
        private static double UpperGammaShapeFraction(double a, double x, out double derivative)
        {
            double difference = x - a;
            double b = difference + 1;
            double d = 1 / b;
            derivative = d;

            // これより先の相対補正は O(a/(x-a)²)。a が大きいときは Temme の範囲外なので無視できる。
            if (difference >= 1e150)
            {
                return d;
            }

            double c = 1 / Tiny;
            double cDerivative = 0;
            double dDerivative = d;
            double fraction = d;
            for (int i = 1; i < MaxIterations; i++)
            {
                double numerator = i * (a - i);
                b += 2;
                double dTerm = numerator * d;
                double dNumerator = -1 + i * d + dTerm * dDerivative;
                double cTerm = numerator / c;
                double cNumerator = -1 + i / c - cTerm * cDerivative;
                d = 1 / (b + dTerm);
                c = b + cTerm;
                dDerivative = -dNumerator * d;
                cDerivative = cNumerator / c;
                double delta = c * d;
                double deltaDerivative = cDerivative + dDerivative;
                fraction *= delta;
                derivative += deltaDerivative;

                // a が整数では、値の係数が 0 になる箇所でも形状微分は続く。
                if (Math.Abs(delta - 1) <= Epsilon && Math.Abs(deltaDerivative) <= Epsilon * Math.Abs(derivative))
                {
                    break;
                }
            }

            return fraction;
        }

        // 小さい形状と小さい x では、上側の値と導関数を同じ展開から求める。
        private static void SmallShapeGammaUpperDerivative(double a, double x, out double logQ, out double derivative)
        {
            double logX = Math.Log(x);
            double gammaMinusOne = Gamma1pm1(a);
            double powerMinusOne = Expm1(a * logX);
            double power = Math.Exp(a * logX);
            double sum = 0;
            double sumDerivative = 0;
            double term = 1;
            for (int n = 1; n < MaxIterations; n++)
            {
                term *= -x / n;
                double denominator = a + n;
                double delta = term / denominator;
                sum += delta;
                double deltaDerivative = -delta / denominator;
                sumDerivative += deltaDerivative;
                if (Math.Abs(delta) <= Epsilon * Math.Abs(sum)
                    && Math.Abs(deltaDerivative) <= Epsilon * Math.Abs(sumDerivative))
                {
                    break;
                }
            }

            double gamma = 1 + gammaMinusOne;
            double digamma = Digamma(1 + a);
            double scaledQ = gammaMinusOne - powerMinusOne - a * power * sum;
            double scaledDerivative = gamma * digamma - power * (logX + (1 + a * logX) * sum + a * sumDerivative);
            logQ = Math.Log(scaledQ) - Log1p(gammaMinusOne);
            derivative = scaledDerivative / scaledQ - digamma;
        }

        // ln(x) - ψ(a) または ln(x) - ψ(a+1)。中心付近では ln(x) と ψ の桁落ちを避ける。
        internal static double GammaShapePrefixDerivative(double a, double x, bool lower)
        {
            if (a < 16)
            {
                return Math.Log(x) - Digamma(lower ? a + 1 : a);
            }

            double offset = (x - a) / a;
            double ratio = x / a;
            double logRatio = Math.Abs(offset) < 0.5 ? Log1p(offset)
                : ratio >= MinNormal ? Math.Log(ratio) : Math.Log(x) - Math.Log(a);

            // ψ の漸近展開（DLMF 5.11.2）。a ≥ 16 で、次の項の絶対値は 1.2e-18 未満。
            double inverse = 1 / a;
            double square = inverse * inverse;
            double correction = square * (1.0 / 12 + square * (-1.0 / 120 + square * (1.0 / 252
                + square * (-1.0 / 240 + square * (1.0 / 132 + square * (-691.0 / 32760))))));
            return logRatio + ((lower ? -0.5 : 0.5) * inverse + correction);
        }

        internal static double LogGammaTailPrefixRatio(double a, double x, double logTail, bool upper)
        {
            // 遠い裾では共通の指数項が差の中で桁落ちするため、級数または連分数の係数だけを使う。
            if (logTail < -16)
            {
                if (InTemmeGammaRegion(a, x))
                {
                    double factor = TemmeGammaTailFactor(a, x, out _, out bool directUpper);
                    if (upper == directUpper)
                    {
                        return 0.5 * Math.Log(a / (2 * Math.PI)) - StirlingCorrection(a) - Math.Log(factor);
                    }
                }
                else if (upper && x >= 1.1 && x - a >= 1)
                {
                    return -Math.Log(UpperGammaFraction(a, x));
                }
                else if (!upper && (x < 1.1 ? UseLowerGammaSeriesForSmallX(a, x) : x - a < 1))
                {
                    return -Math.Log(LowerGammaSum(a, x));
                }
            }

            return LogGammaPrefix(a, x) - logTail;
        }

        internal static (double Mantissa, long Exponent) ScaledGammaTailPrefixRatio(double a, double x, double logTail, bool upper)
        {
            double fraction;
            bool directUpper;
            if (InTemmeGammaRegion(a, x))
            {
                double factor = TemmeGammaTailFactor(a, x, out _, out directUpper);
                double normalization = Math.Sqrt(a / (2 * Math.PI)) * Math.Exp(-StirlingCorrection(a));
                fraction = factor / normalization;
            }
            else
            {
                directUpper = x >= 1.1 && x - a >= 1;
                fraction = directUpper ? UpperGammaFraction(a, x)
                    : x >= 1.1 || UseLowerGammaSeriesForSmallX(a, x) ? LowerGammaSum(a, x, boundRemainder: true) : double.NaN;
            }

            // 直接計算する裾との比では、分子と分母に共通の指数項が消える。
            if (upper == directUpper && fraction > 0 && double.IsFinite(fraction))
            {
                return ScaledReciprocal(fraction);
            }

            var prefix = a <= 1e12 ? GammaShapePreciseLogPrefix(a, x, lower: false) : (LogGammaPrefix(a, x), 0);
            return ScaledTailPrefixRatio(prefix, logTail, upper == directUpper ? double.NaN : fraction);
        }

        // Temme の補正項を含めて微分し、共通因子 exp(-aη²/2) を分子・分母から除く。
        private static double TemmeGammaShapeDerivative(double a, double x, bool upper)
        {
            double offset = (x - a) / a;
            double halfEtaSquared = -Log1pmx(offset);
            double eta = Math.CopySign(Math.Sqrt(2 * halfEtaSquared), offset);
            double logRatio = Log1p(offset);
            double inverse = 1 / a;
            double sqrtA = Math.Sqrt(a);
            double absoluteEta = Math.Abs(eta);
            // t² = aη²/2 の微分は -ln(x/a)。x = a では上側の極限を使う。
            double etaDerivative = eta == 0 ? -inverse : -(offset / eta) * inverse;
            double t = absoluteEta * (sqrtA * InvSqrt2);
            double tDerivative = eta == 0 ? -InvSqrt2 / sqrtA : -(logRatio / absoluteEta) * InvSqrt2 / sqrtA;

            double sum = 0;
            double sumDerivative = 0;
            double power = 1;
            for (int k = 0; k < TemmeGammaCoefficients.Length; k++)
            {
                double[] row = TemmeGammaCoefficients[k];
                double c = row[^1];
                double cDerivative = 0;
                for (int n = row.Length - 2; n >= 0; n--)
                {
                    cDerivative = cDerivative * eta + c;
                    c = c * eta + row[n];
                }

                sum += c * power;
                sumDerivative += (cDerivative * etaDerivative - (k + 0.5) * c * inverse) * power;
                power *= inverse;
            }

            bool directUpper = offset >= 0;
            double sign = directUpper ? 1 : -1;
            double scale = InvSqrt2Pi / sqrtA;
            double remainder = sum * scale;
            double factor = 0.5 * Erfcx(t) + sign * remainder;
            double scaledDerivative = -tDerivative * OneOverSqrtPi + sign * (sumDerivative * scale + logRatio * remainder);
            double derivative = scaledDerivative / factor;
            if (directUpper == upper)
            {
                return derivative;
            }

            double logTail = Math.Log(factor) - a * halfEtaSquared;
            double result = ComplementGammaLogDerivative(logTail, derivative);
            return UsePreciseGammaShapeSubnormal(logTail, derivative, result)
                ? ScaleGammaShapeExponential(TemmeGammaShapePreciseExponent(a, x), -scaledDerivative)
                : result;
        }

        private static (double Hi, double Lo) TemmeGammaShapePreciseExponent(double a, double x)
        {
            // -a(μ - ln(1+μ)) を μ² から始めて、中心付近での桁落ちを避ける。
            var offset = DoubleDoubleDivide(DoubleDoubleTwoSum(x, -a), a);
            var factor = (-offset.Hi, -offset.Lo);
            var power = (Hi: 1.0, Lo: 0.0);
            var sum = (Hi: 0.5, Lo: 0.0);
            for (int n = 3; n <= 64; n++)
            {
                power = DoubleDoubleMultiply(power, factor);
                var term = DoubleDoubleDivide(power, n);
                sum = DoubleDoubleAdd(sum, term);
                if (Math.Abs(term.Hi) <= DoubleDoubleEpsilon * Math.Abs(sum.Hi))
                {
                    break;
                }
            }

            var square = DoubleDoubleMultiply(offset, offset);
            return DoubleDoubleMultiply(DoubleDoubleMultiply((-a, 0), square), sum);
        }
    }
}
