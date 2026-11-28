using System.Numerics;
using static Tuna.XMath.Constants;
using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // |z| ≥ 2⁴⁰ では π の上位・下位分割による還元の誤差が周期数に比例して増大するため、ReduceByPiExact で精密還元する
        private const double ExactPiReductionThreshold = 1099511627776.0;

        // z = kπ + r（|r| ≤ π/2、k は z/π に最も近い整数）の r を double-double で求め、k mod 4（0〜3）を返す。
        // Payne–Hanek の還元と同様に、z = M·2ᵉ（M は 53 ビット整数）に 1/π の長い 2 進展開（PiReductionTable）を掛け、
        // z/π の整数部の下位 2 ビットと小数部の 128 ビットだけを取り出す
        private static int ReduceByPiExact(double z, out double hi, out double lo)
        {
            const int fractionBits = PiReductionTable.FractionBits;
            long bits = BitConverter.DoubleToInt64Bits(z) & long.MaxValue;
            int biased = (int)(bits >> 52);
            long mantissa = bits & 0xFFFFFFFFFFFFFL;
            if (biased == 0)
            {
                biased = 1;
            }
            else
            {
                mantissa |= 1L << 52;
            }

            // |z|/π·2^F = M·(2^B/π)·2^(e - B + F)
            int exponent = biased - 1075;
            BigInteger product = mantissa * PiReductionTable.InversePi;
            int shift = PiReductionTable.InversePiBits - fractionBits - exponent;
            BigInteger scaled = shift >= 0 ? product >> shift : product << -shift;

            // 整数部の下位 2 ビットと小数部だけを残し、最も近い整数 k との差 f = |z|/π - k（|f| ≤ 1/2）を作る
            BigInteger window = scaled & PiReductionTable.WindowMask;
            BigInteger nearest = (window + PiReductionTable.Half) >> fractionBits;
            BigInteger fraction = window - (nearest << fractionBits);
            int quadrant = (int)(nearest & 3);

            int sign = fraction.Sign;
            BigInteger magnitude = BigInteger.Abs(fraction);
            double fHi = 0;
            double fLo = 0;
            if (!magnitude.IsZero)
            {
                // 上位 53 ビットとその次の 53 ビットに分ける（どちらも double で正確に表せる）
                int length = (int)magnitude.GetBitLength();
                int highShift = Math.Max(0, length - 53);
                BigInteger top = magnitude >> highShift;
                BigInteger rest = magnitude - (top << highShift);
                int lowShift = Math.Max(0, highShift - 53);
                fHi = Math.ScaleB((double)(ulong)top, highShift - fractionBits);
                fLo = Math.ScaleB((double)(ulong)(rest >> lowShift), lowShift - fractionBits);
            }

            // r = f·π を double-double で計算する（π = Pi + PiLo）
            double p = fHi * Pi;
            double q = Math.FusedMultiplyAdd(fHi, Pi, -p) + (fHi * PiLo + fLo * Pi);
            hi = p + q;
            lo = q - (hi - p);
            if (sign < 0)
            {
                (hi, lo) = (-hi, -lo);
            }

            if (z < 0)
            {
                (hi, lo) = (-hi, -lo);
                quadrant = (4 - quadrant) & 3;
            }

            return quadrant;
        }

        // ReduceByPiExact が使用する 1/π の 2 進展開（初回使用時に遅延初期化）
        private static class PiReductionTable
        {
            // InversePi の小数部ビット数 B
            public const int InversePiBits = 1200;

            // 取り出す z/π の小数部ビット数 F
            public const int FractionBits = 128;

            // ⌊2^B / π⌋
            public static readonly BigInteger InversePi = CreateInversePi();

            // 整数部の下位 2 ビットと小数部 F ビットのマスク
            public static readonly BigInteger WindowMask = (BigInteger.One << (FractionBits + 2)) - 1;

            // 最も近い整数への丸めに加える 1/2（2^(F-1)）
            public static readonly BigInteger Half = BigInteger.One << (FractionBits - 1);

            // マチンの公式 π = 16 arctan(1/5) - 4 arctan(1/239) を固定小数点で計算し、⌊2^B / π⌋ を求める
            private static BigInteger CreateInversePi()
            {
                // 各項の切り捨て誤差（数百単位）を吸収するガードビットを加えた精度 W で π·2^W を求める
                const int guard = 64;
                int width = InversePiBits + guard;
                BigInteger pi = 16 * ArctanOfReciprocal(5, width) - 4 * ArctanOfReciprocal(239, width);
                return (BigInteger.One << (InversePiBits + width)) / pi;
            }

            // arctan(1/x)·2^W = Σ (-1)ᵏ 2^W / ((2k+1) x^(2k+1))（各項は切り捨て）
            private static BigInteger ArctanOfReciprocal(int x, int width)
            {
                BigInteger power = (BigInteger.One << width) / x;
                BigInteger square = (BigInteger)x * x;
                BigInteger sum = power;
                for (int k = 1; !power.IsZero; k++)
                {
                    power /= square;
                    BigInteger term = power / (2 * k + 1);
                    sum = k % 2 == 1 ? sum - term : sum + term;
                }

                return sum;
            }
        }
    }
}
