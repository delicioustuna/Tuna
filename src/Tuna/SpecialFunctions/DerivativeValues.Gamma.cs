namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        /// <summary>
        /// Computes the partial derivative of the log-beta function with respect to its first parameter, <c>∂ ln |B(a, b)| / ∂a = ψ(a) - ψ(a + b)</c>.
        /// <para>対数ベータ関数の第 1 引数に関する偏微分 <c>∂ ln |B(a, b)| / ∂a = ψ(a) - ψ(a + b)</c> を計算します。</para>
        /// </summary>
        /// <param name="a">The first parameter. / 第 1 引数。</param>
        /// <param name="b">The second parameter. / 第 2 引数。</param>
        /// <returns>The value of <c>ψ(a) - ψ(a + b)</c>, or <see cref="double.NaN"/> when either argument or their exact sum is a non-positive integer, <see cref="double.NaN"/>, or <see cref="double.NegativeInfinity"/>, or when both are <see cref="double.PositiveInfinity"/>. / <c>ψ(a) - ψ(a + b)</c> の値。引数または厳密な和が 0 以下の整数・<see cref="double.NaN"/>・<see cref="double.NegativeInfinity"/> の場合、あるいは両方が <see cref="double.PositiveInfinity"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>The partial derivative with respect to the second parameter can be obtained by calling <c>LogBetaDerivative(b, a)</c>. When only <paramref name="a"/> is <see cref="double.PositiveInfinity"/>, zero with the sign of <c>-b</c> is returned; when only <paramref name="b"/> is <see cref="double.PositiveInfinity"/>, <see cref="double.NegativeInfinity"/> is returned.</para>
        /// <para>第 2 引数に関する偏微分は引数を入れ替えた <c>LogBetaDerivative(b, a)</c> で求められます。<paramref name="a"/> のみが <see cref="double.PositiveInfinity"/> の場合は <c>-b</c> と同符号の 0、<paramref name="b"/> のみが <see cref="double.PositiveInfinity"/> の場合は <see cref="double.NegativeInfinity"/> を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/5.12"/>
        public static double LogBetaDerivative(double a, double b)
        {
            if (double.IsNaN(a) || double.IsNaN(b) || double.IsNegativeInfinity(a) || double.IsNegativeInfinity(b)
                || IsNonPositiveInteger(a) || IsNonPositiveInteger(b))
            {
                return double.NaN;
            }

            if (double.IsPositiveInfinity(a))
            {
                return double.IsPositiveInfinity(b) ? double.NaN : Math.CopySign(0, -b);
            }

            if (double.IsPositiveInfinity(b))
            {
                return double.NegativeInfinity;
            }

            // a + b が double の範囲を超えるのは、ここでは正の引数どうしの場合だけ。
            // この範囲の逆数の補正は主項より約 10⁻³⁰⁸ 倍小さいので、対数の比だけで丸めが決まる。
            double roundedSum = a + b;
            if (double.IsPositiveInfinity(roundedSum))
            {
                return -Log1p(b / a);
            }

            var sum = DoubleDoubleTwoSum(a, b);
            if (sum.Lo == 0 && IsNonPositiveInteger(sum.Hi))
            {
                return double.NaN;
            }

            // 和や変換後の引数から差を求め直すと、小さい b が丸め残りの中で失われるため、入口の差を保つ。
            double value = LogBetaDigammaDifference((a, 0), sum, (b, 0));
            // 1/8 は微小な増分に ψ 同士の引き算を使わないための切替目安。小さな増分では上の差の式を保つ。
            // 離れた引数で、差が個々の ψ の大きさ（下限1）の半分未満なら、反射項の打ち消しを拡張精度で補う。
            const double minimumSeparatedIncrement = 0.125;
            const double cancellationRatio = 0.5;
            if ((a < 0 || sum.Hi < 0) && Math.Abs(b) > minimumSeparatedIncrement)
            {
                double scale = Math.Max(1, Math.Max(Math.Abs(Digamma(a)), Math.Abs(Digamma(sum.Hi))));
                if (Math.Abs(value) < cancellationRatio * scale)
                {
                    // 小さな増分は上の差の式を使い続ける。離れた引数で反射項が打ち消すときだけ、
                    // ψ の値を拡張精度で求めてから引き、結果に残る小さな値を保つ。
                    // 誤差は打ち消す前の大きさに対する約 2⁻¹⁰⁴ の絶対誤差なので、極端な打ち消しでは相対精度が落ちる。
                    var first = DoubleDoubleDigamma((a, 0));
                    var second = DoubleDoubleDigamma(sum);
                    var precise = DoubleDoubleAdd(first, (-second.Hi, -second.Lo));
                    return precise.Hi + precise.Lo;
                }
            }

            return value;
        }

        private static double LogBetaDigammaDifference((double Hi, double Lo) x, (double Hi, double Lo) y, (double Hi, double Lo) difference)
        {
            if (difference.Hi == 0 && difference.Lo == 0)
            {
                return 0;
            }

            if (x.Hi > 0 && y.Hi > 0)
            {
                return LogBetaPositiveDigammaDifference(x, y, difference);
            }

            // 0 の左側では sin(πx) を使うと非正規化数の丸めが効くため、漸化式で正の側へ移す。
            bool smallX = x.Hi < 0 && x.Hi > -0.25;
            bool smallY = y.Hi < 0 && y.Hi > -0.25;
            if (smallX && smallY)
            {
                return LogBetaPositiveDigammaDifference(DoubleDoubleAdd(x, (1, 0)), DoubleDoubleAdd(y, (1, 0)), difference)
                    + LogBetaReciprocalDifference(x, y, difference);
            }

            if (smallX)
            {
                return LogBetaDigammaDifference(DoubleDoubleAdd(x, (1, 0)), y, DoubleDoubleAdd(difference, (-1, 0))) - 1 / x.Hi;
            }

            if (smallY)
            {
                return LogBetaDigammaDifference(x, DoubleDoubleAdd(y, (1, 0)), DoubleDoubleAdd(difference, (1, 0))) + 1 / y.Hi;
            }

            var reflectedX = DoubleDoubleAdd((1, 0), (-x.Hi, -x.Lo));
            var reflectedY = DoubleDoubleAdd((1, 0), (-y.Hi, -y.Lo));
            if (x.Hi < 0 && y.Hi < 0)
            {
                // cot(πy) - cot(πx) = -sin(π(y-x))/(sin(πx)sin(πy))。
                // 2つの cot を直接引かないため、整数の差や半整数付近でも小さな寄与を保つ。
                double reflection = LogBetaCotangentDifference(x, y, difference);
                return LogBetaPositiveDigammaDifference(reflectedX, reflectedY, (-difference.Hi, -difference.Lo)) + reflection;
            }

            if (x.Hi < 0)
            {
                return LogBetaPositiveDigammaDifference(reflectedX, y, DoubleDoubleAdd(y, (-reflectedX.Hi, -reflectedX.Lo))) - Math.PI * LogBetaCotPi(x);
            }

            return LogBetaPositiveDigammaDifference(x, reflectedY, DoubleDoubleAdd(reflectedY, (-x.Hi, -x.Lo))) + Math.PI * LogBetaCotPi(y);
        }

        private static double LogBetaCotPi((double Hi, double Lo) x)
        {
            double cosine = double.CosPi(x.Hi) * double.CosPi(x.Lo) - double.SinPi(x.Hi) * double.SinPi(x.Lo);
            return cosine / SinPiOfSum(x.Hi, x.Lo);
        }

        private static double LogBetaCotangentDifference((double Hi, double Lo) x, (double Hi, double Lo) y, (double Hi, double Lo) difference)
        {
            double sx = SinPiOfSum(x.Hi, x.Lo);
            double sy = SinPiOfSum(y.Hi, y.Lo);
            if (Math.Abs(difference.Hi) < 1e-9)
            {
                // sin(πd) ≈ πd の相対誤差は 2×10⁻¹⁸ 未満。d が非正規化数のときも、
                // πd を先に丸めず、極に近い分母で割った最終結果まで d の指数を保つ。
                int exponent = Math.ILogB(Math.Abs(difference.Hi));
                double mantissa = Math.ScaleB(difference.Hi, -exponent);
                double value = -(Math.PI * Math.PI) * mantissa / sx / sy;
                return Math.ScaleB(value * (1 + difference.Lo / difference.Hi), exponent);
            }

            return -Math.PI * SinPiOfSum(difference.Hi, difference.Lo) / sx / sy;
        }

        // difference は y - x。1 を足したり反射した後の引数では微小な差が失われるため、呼び出し側から渡す。
        private static double LogBetaPositiveDigammaDifference((double Hi, double Lo) x, (double Hi, double Lo) y, (double Hi, double Lo) difference)
        {
            var result = (Hi: 0.0, Lo: 0.0);
            while (Math.Min(x.Hi, y.Hi) < 16)
            {
                double term = LogBetaReciprocalDifference(x, y, difference);
                if (double.IsInfinity(term))
                {
                    return term;
                }

                result = DoubleDoubleAdd(result, (term, 0));
                x = DoubleDoubleAdd(x, (1, 0));
                y = DoubleDoubleAdd(y, (1, 0));
            }

            bool increasing = difference.Hi >= 0;
            var denominator = increasing ? x : y;
            var numerator = increasing ? difference : (-difference.Hi, -difference.Lo);
            var ratio = DoubleDoubleDivide(numerator, denominator.Hi);
            ratio = DoubleDoubleAdd(ratio, (-ratio.Hi * (denominator.Lo / denominator.Hi), 0));
            double logarithm = Log1p(ratio.Hi);
            double logError = ratio.Lo / (1 + ratio.Hi);
            double direction = increasing ? -1 : 1;
            result = DoubleDoubleAdd(result, (direction * logarithm, direction * logError));
            result = DoubleDoubleAdd(result, (0.5 * LogBetaReciprocalDifference(x, y, difference), 0));

            // ψ(x) = ln x - 1/(2x) - Σ B₂ₖ/(2k x²ᵏ)。差のべき乗は expm1 で求める。
            // 小さいほうを16以上へ移したので、10項の後の打ち切り誤差は double の丸めより小さい。
            double inverse = 1 / Math.Min(x.Hi, y.Hi);
            double square = inverse * inverse;
            double power = square;
            double logMagnitude = logarithm + logError;
            for (int k = 1; k <= 10; k++)
            {
                double powerDifference = -Expm1(-2 * k * logMagnitude);
                double term = direction * BernoulliEven[k] / (2 * k) * power * powerDifference;
                result = DoubleDoubleAdd(result, (term, 0));
                power *= square;
            }

            return result.Hi + result.Lo;
        }

        private static double LogBetaReciprocalDifference((double Hi, double Lo) x, (double Hi, double Lo) y, (double Hi, double Lo) difference)
        {
            // 1/y - 1/x = -(y-x)/(xy)。x・y の積や単独の逆数が範囲外でも、最終結果まで指数を分ける。
            if (difference.Hi == 0)
            {
                return 0;
            }

            int dx = Math.ILogB(Math.Abs(x.Hi));
            int dy = Math.ILogB(Math.Abs(y.Hi));
            int dd = Math.ILogB(Math.Abs(difference.Hi));
            double numerator = Math.ScaleB(difference.Hi, -dd);
            double denominatorX = Math.ScaleB(x.Hi, -dx);
            double denominatorY = Math.ScaleB(y.Hi, -dy);
            double quotient = -numerator / denominatorX / denominatorY;
            double correction = difference.Lo / difference.Hi - x.Lo / x.Hi - y.Lo / y.Hi;
            return Math.ScaleB(quotient * (1 + correction), dd - dx - dy);
        }
    }
}
