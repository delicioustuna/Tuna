namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // 零点の直近だけでなく、その前後も再評価するため、差が代表尺度の半分未満なら切り替える。
        // 代表尺度の下限を1にして、反射する項自身が小さい場合にも零点付近を含める。
        private const double GammaReflectionCancellationRatio = 0.5;

        // B₂ₖ の厳密な分子・分母。整数は double で表せるので、除算の残りまで保持する。
        private static readonly (double Numerator, double Denominator)[] BernoulliEvenFractions =
        [
            (1, 6), (-1, 30), (1, 42), (-1, 30), (5, 66), (-691, 2730), (7, 6), (-3617, 510),
            (43867, 798), (-174611, 330), (854513, 138), (-236364091, 2730), (8553103, 6),
            (-23749461029, 870), (8615841276005, 14322), (-7709321041217, 510),
        ];

        // partial class の静的コンストラクターで、厳密な分数を一度だけ double-double へ変換する。
        private static readonly (double Hi, double Lo)[] DoubleDoubleDigammaStirlingCoefficients;
        private static readonly (double Hi, double Lo)[] DoubleDoubleLogGammaStirlingCoefficients;

        private static ((double Hi, double Lo)[] Digamma, (double Hi, double Lo)[] LogGamma) CreateDoubleDoubleGammaStirlingCoefficients()
        {
            var digamma = new (double Hi, double Lo)[BernoulliEvenFractions.Length];
            var logGamma = new (double Hi, double Lo)[BernoulliEvenFractions.Length];
            for (int k = 1; k <= BernoulliEvenFractions.Length; k++)
            {
                var fraction = BernoulliEvenFractions[k - 1];
                digamma[k - 1] = DoubleDoubleDivide((fraction.Numerator, 0), fraction.Denominator * (2 * k));
                logGamma[k - 1] = DoubleDoubleDivide((fraction.Numerator, 0), fraction.Denominator * (2 * k) * (2 * k - 1));
            }

            return (digamma, logGamma);
        }

        // 反射公式の2項が打ち消す場合の ψ。入力の丸め残りも反射後の引数へ渡す。
        private static (double Hi, double Lo) DoubleDoubleDigamma((double Hi, double Lo) x)
        {
            if (x.Hi > 0)
            {
                return DoubleDoublePositiveDigamma(x);
            }

            var reflected = DoubleDoubleAdd((1, 0), (-x.Hi, -x.Lo));
            var first = DoubleDoublePositiveDigamma(reflected);
            var (sine, cosine) = DoubleDoubleSinCosPi(x);
            var cotangent = DoubleDoubleDivide(cosine, sine);
            var second = DoubleDoubleMultiply((Math.PI, 0.5 * TwoPiLo), cotangent);
            return DoubleDoubleAdd(first, (-second.Hi, -second.Lo));
        }

        private static (double Hi, double Lo) DoubleDoublePositiveDigamma((double Hi, double Lo) x)
        {
            var result = (Hi: 0.0, Lo: 0.0);
            while (x.Hi < 32)
            {
                var reciprocal = DoubleDoubleDivide((1.0, 0.0), x);
                result = DoubleDoubleAdd(result, (-reciprocal.Hi, -reciprocal.Lo));
                x = DoubleDoubleAdd(x, (1, 0));
            }

            result = DoubleDoubleAdd(result, DoubleDoubleLog(x));
            var inverse = DoubleDoubleDivide((1.0, 0.0), x);
            result = DoubleDoubleAdd(result, (-0.5 * inverse.Hi, -0.5 * inverse.Lo));
            var square = DoubleDoubleMultiply(inverse, inverse);
            var power = square;
            // x ≥ 32 の16項の後の誤差は約 10⁻⁴² 以下で、拡張精度の丸めより小さい。
            foreach (var coefficient in DoubleDoubleDigammaStirlingCoefficients)
            {
                var term = DoubleDoubleMultiply(coefficient, power);
                result = DoubleDoubleAdd(result, (-term.Hi, -term.Lo));
                power = DoubleDoubleMultiply(power, square);
            }

            return result;
        }

        // 整数を除いた部分の sin(πx)・cos(πx)。整数部分による共通の符号は、比と絶対値には影響しない。
        private static ((double Hi, double Lo) Sine, (double Hi, double Lo) Cosine) DoubleDoubleSinCosPi((double Hi, double Lo) x)
        {
            // |angle| ≤ π/4 へ還元する。半整数付近では補角を使い、cos の小さな値も保つ。
            var reduced = DoubleDoubleAdd((x.Hi - Math.Round(x.Hi), 0), (x.Lo, 0));
            bool complement = Math.Abs(reduced.Hi) > 0.25;
            var fraction = complement
                ? DoubleDoubleAdd((0.5, 0), reduced.Hi < 0 ? reduced : (-reduced.Hi, -reduced.Lo))
                : reduced;
            var angle = DoubleDoubleMultiply((Math.PI, 0.5 * TwoPiLo), fraction);
            var square = DoubleDoubleMultiply(angle, angle);
            var negativeSquare = (-square.Hi, -square.Lo);
            var sineTerm = angle;
            var cosineTerm = (Hi: 1.0, Lo: 0.0);
            var sine = sineTerm;
            var cosine = cosineTerm;
            // 17回の更新後の最初の省略項は sin の37次・cos の36次で、最大でも 5×10⁻⁴⁶ 未満。
            // 打ち切り誤差を double-double の丸めより十分小さく保つため、保守的な余裕を取る。
            for (int k = 1; k <= 17; k++)
            {
                sineTerm = DoubleDoubleDivide(DoubleDoubleMultiply(sineTerm, negativeSquare), (2 * k) * (2 * k + 1));
                cosineTerm = DoubleDoubleDivide(DoubleDoubleMultiply(cosineTerm, negativeSquare), (2 * k - 1) * (2 * k));
                sine = DoubleDoubleAdd(sine, sineTerm);
                cosine = DoubleDoubleAdd(cosine, cosineTerm);
            }

            return complement
                ? (reduced.Hi < 0 ? (-cosine.Hi, -cosine.Lo) : cosine, sine)
                : (sine, cosine);
        }

        private static double LogGammaReflectedNearZeros(double x)
        {
            foreach (var expansion in LogGammaNegativeZeroExpansions)
            {
                double difference = (x - expansion.RootHi) - expansion.RootLo;
                if (Math.Abs(difference) <= expansion.Radius)
                {
                    // ln |Γ| の零点そのものから展開し、反射前の通常の大きさの項を引かない。
                    // 近くの極までの距離の 1/32 以内なので、16項の打ち切り誤差は double の丸めより小さい。
                    double sum = 0;
                    for (int i = expansion.Coefficients.Length - 1; i >= 0; i--)
                    {
                        sum = Math.FusedMultiplyAdd(sum, difference, expansion.Coefficients[i]);
                    }

                    return sum * difference;
                }
            }

            var (sine, _) = DoubleDoubleSinCosPi((x, 0));
            if (sine.Hi < 0)
            {
                sine = (-sine.Hi, -sine.Lo);
            }

            var logarithm = DoubleDoubleLog(sine);
            var reflected = DoubleDoubleTwoSum(1, -x);
            var positive = DoubleDoublePositiveLogGamma(reflected);
            var result = DoubleDoubleAdd((LnPi, LnPiLo), (-logarithm.Hi, -logarithm.Lo));
            result = DoubleDoubleAdd(result, (-positive.Hi, -positive.Lo));
            return result.Hi + result.Lo;
        }

        private static (double Hi, double Lo) DoubleDoublePositiveLogGamma((double Hi, double Lo) x)
        {
            // 呼び出し時は x ≥ 1.25。32まで移す最大31因子の積は 32³¹ 未満で、オーバーフローしない。
            // ln Γ(x) = ln Γ(x + n) - ln(x(x + 1)…(x + n - 1)) とし、対数を一度だけ求める。
            var product = (Hi: 1.0, Lo: 0.0);
            while (x.Hi < 32)
            {
                product = DoubleDoubleMultiply(product, x);
                x = DoubleDoubleAdd(x, (1, 0));
            }

            var recurrence = DoubleDoubleLog(product);
            var result = DoubleDoubleMultiply(DoubleDoubleAdd(x, (-0.5, 0)), DoubleDoubleLog(x));
            result = DoubleDoubleAdd(result, (-x.Hi, -x.Lo));
            result = DoubleDoubleAdd(result, (LnSqrt2Pi, LnSqrt2PiLo));
            var inverse = DoubleDoubleDivide((1.0, 0.0), x);
            var square = DoubleDoubleMultiply(inverse, inverse);
            var power = inverse;
            foreach (var coefficient in DoubleDoubleLogGammaStirlingCoefficients)
            {
                result = DoubleDoubleAdd(result, DoubleDoubleMultiply(coefficient, power));
                power = DoubleDoubleMultiply(power, square);
            }

            return DoubleDoubleAdd(result, (-recurrence.Hi, -recurrence.Lo));
        }
    }
}
