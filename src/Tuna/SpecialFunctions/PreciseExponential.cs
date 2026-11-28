using static Tuna.XMath.Constants;
using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        internal static (double Hi, double Lo) PreciseLogPowerRatio(double numerator, double denominator, double power)
        {
            var (hi, lo) = DoubleDoubleLog((denominator, 0));
            var logRatio = DoubleDoubleAdd(DoubleDoubleLog((numerator, 0)), (-hi, -lo));
            return DoubleDoubleMultiply(logRatio, (power, 0));
        }

        internal static (double Hi, double Lo) PreciseLogComplementPower(double x, double power) =>
            DoubleDoubleMultiply(DoubleDoubleLog(DoubleDoubleTwoSum(1, -x)), (power, 0));

        internal static (double Hi, double Lo) PreciseLogProductRatio(double first, double second, double denominator)
        {
            var ratio = PreciseLogPowerRatio(first, denominator, 1);
            return DoubleDoubleAdd(ratio, DoubleDoubleLog((second, 0)));
        }

        // 指数関数の結果を次の指数に使うときは、最初の指数関数の丸め誤差も保存する。
        internal static (double Hi, double Lo) PreciseExp((double Hi, double Lo) exponent)
        {
            int scale = (int)Math.Round(exponent.Hi / Ln2);
            var reduction = DoubleDoubleMultiply((scale, 0), (Ln2, Ln2Lo));
            var reduced = DoubleDoubleAdd(exponent, (-reduction.Hi, -reduction.Lo));
            var sum = (Hi: 1.0, Lo: 0.0);
            var term = sum;
            for (int n = 1; n <= 22; n++)
            {
                term = DoubleDoubleDivide(DoubleDoubleMultiply(term, reduced), n);
                sum = DoubleDoubleAdd(sum, term);
            }

            return (Math.ScaleB(sum.Hi, scale), Math.ScaleB(sum.Lo, scale));
        }

        private static (double Mantissa, long Exponent) ScaledReciprocal(double value)
        {
            int exponent = Math.ILogB(value);
            return (1 / Math.ScaleB(value, -exponent), -exponent);
        }

        private static (double Mantissa, long Exponent) ScaledTailPrefixRatio(
            (double Hi, double Lo) prefix, double logTail, double complementFactor)
        {
            if (complementFactor > 0 && double.IsFinite(complementFactor))
            {
                double mantissa = ConfluentScaledExp(prefix.Hi, prefix.Lo, out long scale);
                int exponent = Math.ILogB(complementFactor);
                double complement = Math.ScaleB(mantissa * Math.ScaleB(complementFactor, -exponent),
                    (int)Math.Clamp(scale + exponent, int.MinValue, int.MaxValue));
                if (complement <= 0.75)
                {
                    return (mantissa / (1 - complement), scale);
                }
            }

            if (!double.IsFinite(prefix.Hi) || !double.IsFinite(logTail))
            {
                return (Math.Exp(prefix.Hi - logTail), 0);
            }

            var logRatio = DoubleDoubleAdd(prefix, (-logTail, 0));
            double result = ConfluentScaledExp(logRatio.Hi, logRatio.Lo, out long resultScale);
            return (result, resultScale);
        }
    }
}
