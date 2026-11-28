using static Tuna.XMath.Constants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // 2^(1/4), 2^(3/4)
        private const double ParabolicCylinderTwoToQuarter = 1.189207115002721;
        private const double ParabolicCylinderTwoToThreeQuarters = 1.681792830507429;

        // これより小さい |x| では 1 次式 U(a, 0) + U'(a, 0) x を使う（2 次以降の項は相対 1e-200 未満）
        private const double ParabolicCylinderTinyArgument = 1e-100;

        // これより大きい |x| では e^(∓x²/4) が x^(∓a) に必ず勝つため、x → ±∞ の極限値を返す
        private const double ParabolicCylinderHugeArgument = 1e4;

        // 原点近傍の偶奇展開で許容する打ち消しの拡大率（相対誤差がおよそこの倍数の ε になる）
        private const double ParabolicCylinderCancellationLimit = 16;

        /// <summary>
        /// Computes the parabolic cylinder function <c>U(a, x)</c>.
        /// <para>放物柱関数 <c>U(a, x)</c> を計算します。</para>
        /// </summary>
        /// <param name="a">The finite real parameter. / 有限の実数パラメータ。</param>
        /// <param name="x">The real argument. / 実数の引数。</param>
        /// <returns>The value of <c>U(a, x)</c>, or <see cref="double.NaN"/> if <paramref name="a"/> is non-finite or <paramref name="x"/> is <see cref="double.NaN"/>. / <c>U(a, x)</c> の値。<paramref name="a"/> が有限でない場合、または <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>The standard solution of Weber's equation <c>w'' = (x²/4 + a) w</c> that decays as <c>x → +∞</c>. For <c>a = -n - 1/2</c> (integer <c>n ≥ 0</c>), reduces to <c>e^(-x²/4) Heₙ(x)</c>. Returns <c>0</c> at <c>x = +∞</c>; at <c>x = -∞</c>, returns <c>0</c> for Hermite cases and signed infinity with the sign of <c>π / Γ(1/2 + a)</c> otherwise.</para>
        /// <para>Weber 方程式 <c>w'' = (x²/4 + a) w</c> の <c>x → +∞</c> で減衰する標準解です。<c>a = -n - 1/2</c>（整数 <c>n ≥ 0</c>）では <c>e^(-x²/4) Heₙ(x)</c> に一致します。<c>x = +∞</c> では <c>0</c>、<c>x = -∞</c> ではエルミート関数の場合に <c>0</c>、それ以外では <c>π / Γ(1/2 + a)</c> の符号を持つ無限大を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/12.2"/>
        /// <seealso href="https://dlmf.nist.gov/12.7.E14"/>
        public static double ParabolicCylinderU(double a, double x)
        {
            if (double.IsNaN(a) || double.IsInfinity(a) || double.IsNaN(x))
            {
                return double.NaN;
            }

            if (x > ParabolicCylinderHugeArgument)
            {
                return 0;
            }

            if (x < -ParabolicCylinderHugeArgument)
            {
                // U(a, -x) ~ π/Γ(1/2+a) V(a, x) は増大する。エルミート関数（1/Γ(1/2+a) = 0）では減衰する
                double reciprocal = ParabolicCylinderReciprocalGamma(0.5 + a, 0.5 - a, out _);
                return reciprocal == 0 ? 0 : Math.Sign(reciprocal) * double.PositiveInfinity;
            }

            double mantissa = ParabolicCylinderUScaled(a, x, out long exponent);
            return ScaleB(mantissa, exponent);
        }

        /// <summary>
        /// Computes the parabolic cylinder function <c>V(a, x)</c>.
        /// <para>放物柱関数 <c>V(a, x)</c> を計算します。</para>
        /// </summary>
        /// <param name="a">The finite real parameter. / 有限の実数パラメータ。</param>
        /// <param name="x">The real argument. / 実数の引数。</param>
        /// <returns>The value of <c>V(a, x)</c>, or <see cref="double.NaN"/> if <paramref name="a"/> is non-finite or <paramref name="x"/> is <see cref="double.NaN"/>. / <c>V(a, x)</c> の値。<paramref name="a"/> が有限でない場合、または <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>The standard solution of Weber's equation <c>w'' = (x²/4 + a) w</c> that grows as <c>x → +∞</c> (overflowing to <see cref="double.PositiveInfinity"/> for <c>x ≳ 54</c>). At <c>x = -∞</c>, returns <c>0</c> for integer <paramref name="a"/> and signed infinity with the sign of <c>sin(πa)</c> otherwise.</para>
        /// <para>Weber 方程式 <c>w'' = (x²/4 + a) w</c> の <c>x → +∞</c> で増大する標準解です（<c>x ≳ 54</c> で <see cref="double.PositiveInfinity"/> にオーバーフローします）。<c>x = -∞</c> では <paramref name="a"/> が整数のとき <c>0</c>、それ以外では <c>sin(πa)</c> の符号を持つ無限大を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/12.2"/>
        /// <seealso href="https://dlmf.nist.gov/12.7.E13"/>
        public static double ParabolicCylinderV(double a, double x)
        {
            if (double.IsNaN(a) || double.IsInfinity(a) || double.IsNaN(x))
            {
                return double.NaN;
            }

            if (x > ParabolicCylinderHugeArgument)
            {
                return double.PositiveInfinity;
            }

            if (x < -ParabolicCylinderHugeArgument)
            {
                // V(a, -x) ~ sin(πa) V(a, x)。a が整数なら U の項だけが残って減衰する
                double sine = double.SinPi(a);
                return sine == 0 ? 0 : Math.Sign(sine) * double.PositiveInfinity;
            }

            double mantissa = ParabolicCylinderVScaled(a, x, out long exponent);
            return ScaleB(mantissa, exponent);
        }

        /// <summary>
        /// Computes Whittaker's parabolic cylinder function <c>D_ν(x) = U(-ν - 1/2, x)</c>.
        /// <para>Whittaker の放物柱関数 <c>D_ν(x) = U(-ν - 1/2, x)</c> を計算します。</para>
        /// </summary>
        /// <param name="nu">The finite real order <c>ν</c>. / 有限の実数次数 <c>ν</c>。</param>
        /// <param name="x">The real argument. / 実数の引数。</param>
        /// <returns>The value of <c>D_ν(x)</c>, or <see cref="double.NaN"/> if <paramref name="nu"/> is non-finite or <paramref name="x"/> is <see cref="double.NaN"/>. / <c>D_ν(x)</c> の値。<paramref name="nu"/> が有限でない場合、または <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>For non-negative integer <c>ν = n</c>, equals <c>e^(-x²/4) Heₙ(x)</c>. Follows the same limit and <see cref="double.NaN"/> conventions as <see cref="ParabolicCylinderU"/>.</para>
        /// <para>非負整数 <c>ν = n</c> では <c>e^(-x²/4) Heₙ(x)</c> に一致します。極限値および <see cref="double.NaN"/> の扱いは <see cref="ParabolicCylinderU"/> に準じます。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/12.1"/>
        public static double ParabolicCylinderD(double nu, double x)
        {
            if (double.IsNaN(nu) || double.IsInfinity(nu))
            {
                return double.NaN;
            }

            return ParabolicCylinderU(-nu - 0.5, x);
        }

        // U(a, x) = mantissa · 2^exponent（x は有限）の仮数を返す
        private static double ParabolicCylinderUScaled(double a, double x, out long exponent)
        {
            if (Math.Abs(x) < ParabolicCylinderTinyArgument)
            {
                // U(a, x) ≈ U(a, 0) + U'(a, 0) x（U(a, 0) が 0 になる a でも傾きの項が残る）
                double value0 = ParabolicCylinderValueAtZeroU(a, out long valueExponent);
                double slope0 = ParabolicCylinderSlopeAtZeroU(a, out long slopeExponent);
                double slopeTerm = slope0 * x;
                return ConfluentAlignPair(value0, valueExponent, ref slopeTerm, slopeExponent, out exponent) + slopeTerm;
            }

            if (a > -0.5 && x < 0)
            {
                // U = U(a,0) u₁ + U'(a,0) u₂。x < 0 では u₂ < 0、U'(a,0) < 0 なので 2 項とも正（打ち消しがない）
                return ParabolicCylinderEvenOddCombination(a, x, false, out exponent, out _);
            }

            if (ParabolicCylinderNearOrigin(a, x))
            {
                // 原点の近く（a < 0 の振動域の内側、または |a| x² が小さい）では偶関数・奇関数の組が速い。
                // 増大する部分どうしの打ち消し（転回点の外側で起こる）が小さいことを確かめてから使う
                double combination = ParabolicCylinderEvenOddCombination(a, x, false, out exponent, out double cancellation);
                if (cancellation <= ParabolicCylinderCancellationLimit)
                {
                    return combination;
                }
            }

            if (x > 0)
            {
                // U(a, x) = 2^(-1/4-a/2) e^(-x²/4) U(a/2 + 1/4, 1/2, x²/2)
                double z = 0.5 * x * x;
                double tricomi = HypergeometricUScaled(0.5 * a + 0.25, 0.5, z, out long tricomiExponent);
                double power = ParabolicCylinderPowerOfTwo(-0.5 * a, 1 / ParabolicCylinderTwoToQuarter, out long powerExponent);
                double gaussian = ParabolicCylinderGaussian(x, -1, out long gaussianExponent);
                exponent = tricomiExponent + powerExponent + gaussianExponent;
                return tricomi * power * gaussian;
            }

            // a ≤ -1/2: U(a, -|x|) = -sin(πa) U(a, |x|) + π/Γ(1/2+a) V(a, |x|)、π/Γ(1/2+a) = Γ(1/2-a) cos(πa)
            double absX = -x;
            double u = ParabolicCylinderUScaled(a, absX, out long uExponent);
            double v = ParabolicCylinderVScaled(a, absX, out long vExponent);
            double gamma = ScaledGammaSigned(0.5 - a, out long gammaExponent);
            double uTerm = -double.SinPi(a) * u;
            double vTerm = gamma * double.CosPi(a) * v;
            return ConfluentAlignPair(uTerm, uExponent, ref vTerm, vExponent + gammaExponent, out exponent) + vTerm;
        }

        // 原点近傍の偶奇展開を先に試す領域（a < 0 の転回点 x² = -4a の内側、または |a| x² ≤ 2, x² ≤ 1）
        private static bool ParabolicCylinderNearOrigin(double a, double x)
        {
            double square = x * x;
            return square <= 1 || Math.Abs(a) * square <= 2 || (a < 0 && square <= -2 * a);
        }

        // c₀ u₁ + c₁ u₂ を返す（isV が真なら V、偽なら U の原点値・傾きを使用）。cancellation は打ち消しによる誤差の拡大率
        private static double ParabolicCylinderEvenOddCombination(double a, double x, bool isV, out long exponent, out double cancellation)
        {
            ParabolicCylinderEvenOdd(a, x, out double even, out long evenExponent, out double odd, out long oddExponent);
            long valueExponent;
            long slopeExponent;
            double value0 = isV ? ParabolicCylinderValueAtZeroV(a, out valueExponent) : ParabolicCylinderValueAtZeroU(a, out valueExponent);
            double slope0 = isV ? ParabolicCylinderSlopeAtZeroV(a, out slopeExponent) : ParabolicCylinderSlopeAtZeroU(a, out slopeExponent);
            double first = value0 * even;
            double second = slope0 * odd;
            first = ConfluentAlignPair(first, valueExponent + evenExponent, ref second, slopeExponent + oddExponent, out exponent);
            double result = first + second;
            cancellation = result == 0 ? double.PositiveInfinity : (Math.Abs(first) + Math.Abs(second)) / Math.Abs(result);
            return result;
        }

        // V(a, x) = mantissa · 2^exponent（x は有限）の仮数を返す
        private static double ParabolicCylinderVScaled(double a, double x, out long exponent)
        {
            if (Math.Abs(x) < ParabolicCylinderTinyArgument)
            {
                double value0 = ParabolicCylinderValueAtZeroV(a, out long valueExponent);
                double slope0 = ParabolicCylinderSlopeAtZeroV(a, out long slopeExponent);
                double slopeTerm = slope0 * x;
                return ConfluentAlignPair(value0, valueExponent, ref slopeTerm, slopeExponent, out exponent) + slopeTerm;
            }

            if (x > 0)
            {
                // V = V(a,0) u₁ + V'(a,0) u₂（x > 0 では増大する部分が同符号で足し合わさる）
                return ParabolicCylinderEvenOddCombination(a, x, true, out exponent, out _);
            }

            if (ParabolicCylinderNearOrigin(a, x))
            {
                // x < 0 でも原点の近くでは同じ組を試す（V(a, -x) が劣勢になる a ≈ 整数で遠くまで行くと打ち消すので確かめる）
                double combination = ParabolicCylinderEvenOddCombination(a, x, true, out exponent, out double cancellation);
                if (cancellation <= ParabolicCylinderCancellationLimit)
                {
                    return combination;
                }
            }

            // V(a, -|x|) = cos(πa)/Γ(1/2-a) U(a, |x|) + sin(πa) V(a, |x|)
            double absX = -x;
            double u = ParabolicCylinderUScaled(a, absX, out long uExponent);
            double v = ParabolicCylinderVScaled(a, absX, out long vExponent);
            double reciprocal = ParabolicCylinderReciprocalGamma(0.5 - a, 0.5 + a, out long reciprocalExponent);
            double uTerm = double.CosPi(a) * reciprocal * u;
            double vTerm = double.SinPi(a) * v;
            return ConfluentAlignPair(uTerm, uExponent + reciprocalExponent, ref vTerm, vExponent, out exponent) + vTerm;
        }

        // 偶関数 u₁ = e^(-x²/4) M(a/2+1/4, 1/2, x²/2) と奇関数 u₂ = x e^(-x²/4) M(a/2+3/4, 3/2, x²/2) を仮数と指数で返す。
        // e^(-x²/4) M(z) = e^(x²/4) (e^(-z) M(z))（z = x²/2）と分け、e^(x²/4) は double-double の x² から計算する
        private static void ParabolicCylinderEvenOdd(double a, double x, out double even, out long evenExponent, out double odd, out long oddExponent)
        {
            double z = 0.5 * x * x;
            double gaussian = ParabolicCylinderGaussian(x, 1, out long gaussianExponent);
            double decay = ConfluentScaledExp(-z, 0, out long decayExponent);
            double m1 = Hypergeometric1F1Scaled(0.5 * a + 0.25, 0.5, z, out long m1Exponent);
            double m2 = Hypergeometric1F1Scaled(0.5 * a + 0.75, 1.5, z, out long m2Exponent);
            even = m1 * decay * gaussian;
            evenExponent = m1Exponent + decayExponent + gaussianExponent;
            odd = m2 * decay * gaussian * x;
            oddExponent = m2Exponent + decayExponent + gaussianExponent;
        }

        // e^(sign·x²/4) = mantissa · 2^exponent。x² を FMA で double-double にして指数の丸め誤差拡大を防ぐ
        private static double ParabolicCylinderGaussian(double x, int sign, out long exponent)
        {
            double square = x * x;
            double squareLo = Math.FusedMultiplyAdd(x, x, -square);
            return ConfluentScaledExp(sign * 0.25 * square, sign * 0.25 * squareLo, out exponent);
        }

        // 2^t · factor = mantissa · 2^exponent（t を整数部と小数部に分ける）
        private static double ParabolicCylinderPowerOfTwo(double t, double factor, out long exponent)
        {
            double n = Math.Floor(t);
            if (Math.Abs(n) > 1e18)
            {
                exponent = 0;
                return t > 0 ? double.PositiveInfinity : 0;
            }

            exponent = (long)n;
            return Math.Pow(2, t - n) * factor;
        }

        // 1/Γ(z) = mantissa · 2^exponent（z < 1/2 では相反公式 1/Γ(z) = Γ(1-z) sin(πz)/π を使用）
        private static double ParabolicCylinderReciprocalGamma(double z, double oneMinusZ, out long exponent)
        {
            if (z >= 0.5)
            {
                double gamma = ScaledGammaSigned(z, out long gammaExponent);
                exponent = -gammaExponent;
                return 1 / gamma;
            }

            double sine = double.SinPi(z);
            if (sine == 0)
            {
                exponent = 0;
                return 0;
            }

            double reflected = ScaledGammaSigned(oneMinusZ, out exponent);
            return reflected * sine / Math.PI;
        }

        // DLMF 12.2.6: U(a, 0) = √π / (2^(a/2+1/4) Γ(3/4 + a/2))
        private static double ParabolicCylinderValueAtZeroU(double a, out long exponent)
        {
            double reciprocal = ParabolicCylinderReciprocalGamma(0.75 + 0.5 * a, 0.25 - 0.5 * a, out long reciprocalExponent);
            double power = ParabolicCylinderPowerOfTwo(-0.5 * a, 1 / ParabolicCylinderTwoToQuarter, out long powerExponent);
            exponent = reciprocalExponent + powerExponent;
            return SqrtPi * power * reciprocal;
        }

        // DLMF 12.2.7: U'(a, 0) = -√π / (2^(a/2-1/4) Γ(1/4 + a/2))
        private static double ParabolicCylinderSlopeAtZeroU(double a, out long exponent)
        {
            double reciprocal = ParabolicCylinderReciprocalGamma(0.25 + 0.5 * a, 0.75 - 0.5 * a, out long reciprocalExponent);
            double power = ParabolicCylinderPowerOfTwo(-0.5 * a, ParabolicCylinderTwoToQuarter, out long powerExponent);
            exponent = reciprocalExponent + powerExponent;
            return -SqrtPi * power * reciprocal;
        }

        // DLMF 12.2.21: V(a, 0) = 2^(a/2+1/4) sin(π(3/4 - a/2)) / Γ(3/4 - a/2)
        private static double ParabolicCylinderValueAtZeroV(double a, out long exponent)
        {
            double z = 0.75 - 0.5 * a;
            double reciprocal = ParabolicCylinderReciprocalGamma(z, 0.25 + 0.5 * a, out long reciprocalExponent);
            double power = ParabolicCylinderPowerOfTwo(0.5 * a, ParabolicCylinderTwoToQuarter, out long powerExponent);
            exponent = reciprocalExponent + powerExponent;
            return power * double.SinPi(z) * reciprocal;
        }

        // DLMF 12.2.22: V'(a, 0) = 2^(a/2+3/4) sin(π(1/4 - a/2)) / Γ(1/4 - a/2)
        private static double ParabolicCylinderSlopeAtZeroV(double a, out long exponent)
        {
            double z = 0.25 - 0.5 * a;
            double reciprocal = ParabolicCylinderReciprocalGamma(z, 0.75 + 0.5 * a, out long reciprocalExponent);
            double power = ParabolicCylinderPowerOfTwo(0.5 * a, ParabolicCylinderTwoToThreeQuarters, out long powerExponent);
            exponent = reciprocalExponent + powerExponent;
            return power * double.SinPi(z) * reciprocal;
        }
    }
}
