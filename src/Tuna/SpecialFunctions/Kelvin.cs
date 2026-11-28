using System.Numerics;
using static Tuna.XMath.Constants;
using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // これ以下の x では ber, bei に冪級数を使う（それより大きい x は漸近展開）
        private const double KelvinSeriesLimit = 30;

        // これより大きい x では漸近展開が使えないとき（n² ≳ x）に冪級数で代用しない（項がオーバーフローするため）
        private const double KelvinSeriesMaximum = 700;

        // これ以下の x では ker₀, ker₁ などに冪級数を使う（x = 10 で打ち消し倍率 ~3e7、double-double で十分）
        private const double KelvinKSeriesLimit = 10;

        // これより大きい x では K_n(x e^(iπ/4))（n = 0, 1）に漸近展開を使う（それ以下は台形則）
        private const double KelvinKAsymptoticLimit = 30;

        /// <summary>
        /// Computes the Kelvin function <c>ber_n(x) = Re J_n(x e^(3πi / 4))</c>.
        /// <para>ケルビン関数 <c>ber_n(x) = Re J_n(x e^(3πi / 4))</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The integer order. / 整数次数。</param>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>ber_n(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is non-finite, <paramref name="n"/> is <see cref="int.MinValue"/>, or the value cannot be evaluated. / <c>ber_n(x)</c> の値。<paramref name="x"/> が有限でない場合、<paramref name="n"/> が <see cref="int.MinValue"/> の場合、または計算不能な場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Satisfies <c>ber_(-n)(x) = (-1)ⁿ ber_n(x)</c> and <c>ber_n(-x) = (-1)ⁿ ber_n(x)</c>. Returns <see cref="double.NaN"/> when <c>|x| &gt; 700</c> and <c>n² ≳ |x|</c> where neither the asymptotic expansion nor the power series applies.</para>
        /// <para><c>ber_(-n)(x) = (-1)ⁿ ber_n(x)</c> および <c>ber_n(-x) = (-1)ⁿ ber_n(x)</c> を満たします。<c>|x| &gt; 700</c> かつ <c>n² ≳ |x|</c> で漸近展開も冪級数も適用できない場合は <see cref="double.NaN"/> を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/10.61.E1"/>
        public static double KelvinBer(int n, double x) => KelvinBerBei(n, x).Real;

        /// <summary>
        /// Computes the Kelvin function <c>bei_n(x) = Im J_n(x e^(3πi / 4))</c>.
        /// <para>ケルビン関数 <c>bei_n(x) = Im J_n(x e^(3πi / 4))</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The integer order. / 整数次数。</param>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>bei_n(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is non-finite, <paramref name="n"/> is <see cref="int.MinValue"/>, or the value cannot be evaluated. / <c>bei_n(x)</c> の値。<paramref name="x"/> が有限でない場合、<paramref name="n"/> が <see cref="int.MinValue"/> の場合、または計算不能な場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Shares the same symmetry relations and domain behavior as <see cref="KelvinBer"/>.</para>
        /// <para>対称性および <see cref="double.NaN"/> を返す範囲は <see cref="KelvinBer"/> と同じです。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/10.61.E1"/>
        public static double KelvinBei(int n, double x) => KelvinBerBei(n, x).Imaginary;

        /// <summary>
        /// Computes the Kelvin function <c>ker_n(x) = Re(e^(-nπi / 2) K_n(x e^(πi / 4)))</c>.
        /// <para>ケルビン関数 <c>ker_n(x) = Re(e^(-nπi / 2) K_n(x e^(πi / 4)))</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The integer order. / 整数次数。</param>
        /// <param name="x">The non-negative input value (<c>x ≥ 0</c>). / 非負の入力値（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>ker_n(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> &lt; 0, <paramref name="x"/> is <see cref="double.NaN"/>, or <paramref name="n"/> is <see cref="int.MinValue"/>. / <c>ker_n(x)</c> の値。<paramref name="x"/> &lt; 0、<paramref name="x"/> が <see cref="double.NaN"/>、または <paramref name="n"/> が <see cref="int.MinValue"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Satisfies <c>ker_(-n)(x) = (-1)ⁿ ker_n(x)</c>. At <c>x = 0</c>, returns <c>1 / 2</c> for <c>|n| = 2</c> and a signed infinity for all other orders; returns <c>0</c> at <see cref="double.PositiveInfinity"/>.</para>
        /// <para><c>ker_(-n)(x) = (-1)ⁿ ker_n(x)</c> を満たします。<c>x = 0</c> では <c>|n| = 2</c> で <c>1 / 2</c>、その他の次数では符号付き無限大となり、<see cref="double.PositiveInfinity"/> では <c>0</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/10.61.E2"/>
        public static double KelvinKer(int n, double x) => KelvinKerKei(n, x).Real;

        /// <summary>
        /// Computes the Kelvin function <c>kei_n(x) = Im(e^(-nπi / 2) K_n(x e^(πi / 4)))</c>.
        /// <para>ケルビン関数 <c>kei_n(x) = Im(e^(-nπi / 2) K_n(x e^(πi / 4)))</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The integer order. / 整数次数。</param>
        /// <param name="x">The non-negative input value (<c>x ≥ 0</c>). / 非負の入力値（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>kei_n(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> &lt; 0, <paramref name="x"/> is <see cref="double.NaN"/>, or <paramref name="n"/> is <see cref="int.MinValue"/>. / <c>kei_n(x)</c> の値。<paramref name="x"/> &lt; 0、<paramref name="x"/> が <see cref="double.NaN"/>、または <paramref name="n"/> が <see cref="int.MinValue"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Satisfies <c>kei_(-n)(x) = (-1)ⁿ kei_n(x)</c>. At <c>x = 0</c>, returns <c>-π / 4</c> for <c>n = 0</c> and a signed infinity for all other orders; returns <c>0</c> at <see cref="double.PositiveInfinity"/>.</para>
        /// <para><c>kei_(-n)(x) = (-1)ⁿ kei_n(x)</c> を満たします。<c>x = 0</c> では <c>n = 0</c> で <c>-π / 4</c>、その他の次数では符号付き無限大となり、<see cref="double.PositiveInfinity"/> では <c>0</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/10.61.E2"/>
        public static double KelvinKei(int n, double x) => KelvinKerKei(n, x).Imaginary;

        // ber_n(x) + i bei_n(x)
        private static Complex KelvinBerBei(int n, double x)
        {
            if (double.IsNaN(x) || double.IsInfinity(x) || n == int.MinValue)
            {
                return new Complex(double.NaN, double.NaN);
            }

            // ber_(-n) = (-1)ⁿ ber_n、ber_n(-x) = (-1)ⁿ ber_n(x)（bei も同じ。DLMF 10.61.6, 10.61.7）
            int order = Math.Abs(n);
            double sign = (n < 0 && (order & 1) == 1) ? -1 : 1;
            if (x < 0 && (order & 1) == 1)
            {
                sign = -sign;
            }

            x = Math.Abs(x);
            if (x == 0)
            {
                return new Complex(order == 0 ? 1 : 0, 0);
            }

            Complex value;
            if (x > KelvinSeriesLimit && KelvinBerBeiAsymptotic(order, x, out value))
            {
                return KelvinNegateIf(sign < 0, value);
            }

            if (x > KelvinSeriesMaximum)
            {
                return new Complex(double.NaN, double.NaN);
            }

            return KelvinNegateIf(sign < 0, KelvinBerBeiSeries(order, x));
        }

        // 冪級数による ber_n + i bei_n（n ≥ 0, x > 0）。x ≤ KelvinDoubleSeriesLimit では倍精度で足す
        private static Complex KelvinBerBeiSeries(int n, double x)
        {
            if (x <= KelvinDoubleSeriesLimit)
            {
                return KelvinBerBeiDoubleSeries(n, x);
            }

            KelvinPowerSeries(n, x, false, out var a, out var b, out _, out _);
            return KelvinApplyPrefactor(n, x, a, b);
        }

        // 倍精度冪級数を使う上限（x ≤ 2 では打ち消し倍率が高々 4/3 倍のため倍精度で十分）
        private const double KelvinDoubleSeriesLimit = 2;

        // KelvinBerBeiSeries の倍精度版（0 < x ≤ KelvinDoubleSeriesLimit）
        private static Complex KelvinBerBeiDoubleSeries(int n, double x)
        {
            double halfX = 0.5 * x;
            double y = halfX * halfX;
            double term = 1;
            double re = 1;
            double im = 0;
            for (int k = 1; k < 1000; k++)
            {
                term = term * y / ((double)k * (n + k));

                // iᵏ = 1, i, -1, -i
                switch (k & 3)
                {
                    case 0:
                        re += term;
                        break;
                    case 1:
                        im += term;
                        break;
                    case 2:
                        re -= term;
                        break;
                    default:
                        im -= term;
                        break;
                }

                if (term <= 0.5 * MachineEpsilon * Math.Max(Math.Abs(re), Math.Abs(im)))
                {
                    break;
                }
            }

            return KelvinApplyPrefactor(n, x, (re, 0.0), (im, 0.0));
        }

        // Σ iᵏ tₖ（tₖ = (x²/4)ᵏ n!/(k! (n+k)!)）の実部・虚部を double-double で計算する。
        // withHarmonic = true のときは各項に Hₖ + H_(n+k) を掛けた和も同時に返す
        private static void KelvinPowerSeries(
            int n, double x, bool withHarmonic,
            out (double Hi, double Lo) re, out (double Hi, double Lo) im,
            out (double Hi, double Lo) harmonicRe, out (double Hi, double Lo) harmonicIm)
        {
            double halfX = 0.5 * x;
            double yHi = halfX * halfX;
            (double Hi, double Lo) y = (yHi, Math.FusedMultiplyAdd(halfX, halfX, -yHi));
            (double Hi, double Lo) term = (1.0, 0.0);
            re = (1.0, 0.0);
            im = (0.0, 0.0);
            harmonicRe = (0.0, 0.0);
            harmonicIm = (0.0, 0.0);

            // H_k と H_(n+k)（k = 0 では H_0 = 0 と H_n）
            (double Hi, double Lo) hk = (0.0, 0.0);
            (double Hi, double Lo) hnk = (0.0, 0.0);
            if (withHarmonic)
            {
                for (int j = 1; j <= n; j++)
                {
                    hnk = DoubleDoubleAdd(hnk, DoubleDoubleDivide((1.0, 0.0), j));
                }

                harmonicRe = hnk;
            }

            for (int k = 1; k < 100000; k++)
            {
                term = DoubleDoubleDivide(DoubleDoubleMultiply(term, y), (double)k * (n + k));
                var weighted = term;
                if (withHarmonic)
                {
                    hk = DoubleDoubleAdd(hk, DoubleDoubleDivide((1.0, 0.0), k));
                    hnk = DoubleDoubleAdd(hnk, DoubleDoubleDivide((1.0, 0.0), n + k));
                    weighted = DoubleDoubleMultiply(term, DoubleDoubleAdd(hk, hnk));
                }

                // iᵏ = 1, i, -1, -i
                switch (k & 3)
                {
                    case 0:
                        re = DoubleDoubleAdd(re, term);
                        harmonicRe = withHarmonic ? DoubleDoubleAdd(harmonicRe, weighted) : harmonicRe;
                        break;
                    case 1:
                        im = DoubleDoubleAdd(im, term);
                        harmonicIm = withHarmonic ? DoubleDoubleAdd(harmonicIm, weighted) : harmonicIm;
                        break;
                    case 2:
                        re = DoubleDoubleAdd(re, (-term.Hi, -term.Lo));
                        harmonicRe = withHarmonic ? DoubleDoubleAdd(harmonicRe, (-weighted.Hi, -weighted.Lo)) : harmonicRe;
                        break;
                    default:
                        im = DoubleDoubleAdd(im, (-term.Hi, -term.Lo));
                        harmonicIm = withHarmonic ? DoubleDoubleAdd(harmonicIm, (-weighted.Hi, -weighted.Lo)) : harmonicIm;
                        break;
                }

                if ((double)k * (n + k) > yHi && Math.Abs(weighted.Hi) <= DoubleDoubleEpsilon * Math.Max(Math.Abs(re.Hi), Math.Abs(im.Hi)))
                {
                    break;
                }
            }
        }

        // ((x/2)ⁿ/n!) e^(3nπi/4) (A + iB) を double-double の回転とスケール分離で計算する
        private static Complex KelvinApplyPrefactor(int n, double x, (double Hi, double Lo) a, (double Hi, double Lo) b)
        {
            var real = a;
            var imaginary = b;
            KelvinRotate(3 * (long)n, ref real, ref imaginary);

            // (x/2)ⁿ/n! を仮数と指数に分けて掛ける
            double power = ScaledPower(0.5 * x, n, out long powerExponent);
            double factorial = ScaledFactorial(n, out long factorialExponent);
            double mantissa = power / factorial;
            long exponent = powerExponent - factorialExponent;
            return new Complex(
                StruveScaleProduct(mantissa, exponent, real.Hi + real.Lo),
                StruveScaleProduct(mantissa, exponent, imaginary.Hi + imaginary.Lo));
        }

        // e^(mπi/4) の成分の符号（diagonal = true のときは 1/√2 を掛ける）
        private static void KelvinEighthRoot(long m, out int c, out int s, out bool diagonal)
        {
            switch ((int)(((m % 8) + 8) % 8))
            {
                case 0:
                    (c, s, diagonal) = (1, 0, false);
                    break;
                case 1:
                    (c, s, diagonal) = (1, 1, true);
                    break;
                case 2:
                    (c, s, diagonal) = (0, 1, false);
                    break;
                case 3:
                    (c, s, diagonal) = (-1, 1, true);
                    break;
                case 4:
                    (c, s, diagonal) = (-1, 0, false);
                    break;
                case 5:
                    (c, s, diagonal) = (-1, -1, true);
                    break;
                case 6:
                    (c, s, diagonal) = (0, -1, false);
                    break;
                default:
                    (c, s, diagonal) = (1, -1, true);
                    break;
            }
        }

        private static (double Hi, double Lo) KelvinScale((double Hi, double Lo) a, int sign) =>
            sign == 0 ? (0.0, 0.0) : sign > 0 ? a : (-a.Hi, -a.Lo);

        // 漸近展開による ber_n + i bei_n = iⁿ I_n(x e^(iπ/4))（x > 30）。収束しなければ false を返す
        private static bool KelvinBerBeiAsymptotic(int n, double x, out Complex value)
        {
            value = Complex.Zero;
            if (!KelvinHankelSum(n, x, true, out Complex plus) || !KelvinHankelSum(n, x, false, out Complex minus))
            {
                return false;
            }

            // x/√2 を double-double で求め、e^(±z)/√(2πz) = e^(±x/√2)/√(2πx) · e^(i(±x/√2 - π/8)) の位相を正確に還元する
            var g = KelvinXOverSqrt2(x);
            DoubleDoubleCosSin(DoubleDoubleAdd(g, (-0.125 * Math.PI, -0.125 * PiLo)), out double cosPlus, out double sinPlus);
            DoubleDoubleCosSin(DoubleDoubleAdd((-g.Hi, -g.Lo), (-0.125 * Math.PI, -0.125 * PiLo)), out double cosMinus, out double sinMinus);
            double amplitude = 1 / Math.Sqrt(2 * Math.PI * x);

            // e^(x/√2) は x ≳ 1004 でオーバーフローするので、半分ずつ 2 回掛ける（e^(g.Hi + g.Lo) ≈ e^(g.Hi)(1 + g.Lo)）
            double halfGrowth = Math.Exp(0.5 * g.Hi);
            double decay = Math.Exp(-g.Hi) * (1 - g.Lo);
            Complex dominant = new Complex(cosPlus, sinPlus) * plus * (amplitude * (1 + g.Lo));
            Complex recessive = new Complex(cosMinus, sinMinus) * minus * (amplitude * decay);

            // I_n(z) = dominant + i(-1)ⁿ recessive、ber + i bei = iⁿ I_n(z)
            // 大きさがオーバーフローしても NaN にならないよう、iⁿ と e^(x/√2) は成分ごとに掛ける
            Complex recessiveTerm = Complex.ImaginaryOne * ((n & 1) == 1 ? -recessive : recessive);
            Complex sum = new Complex(dominant.Real * halfGrowth, dominant.Imaginary * halfGrowth) + recessiveTerm / halfGrowth;
            sum = new Complex(sum.Real * halfGrowth, sum.Imaginary * halfGrowth);
            value = (n & 3) switch
            {
                0 => sum,
                1 => new Complex(-sum.Imaginary, sum.Real),
                2 => new Complex(-sum.Real, -sum.Imaginary),
                _ => new Complex(sum.Imaginary, -sum.Real),
            };
            return true;
        }

        // ハンケルの漸近展開の和 Σ (∓1)ᵏ aₖ(n)/zᵏ（DLMF 10.17.1、z = x e^(iπ/4)、alternating = true で (-1)ᵏ 付き）
        private static bool KelvinHankelSum(int n, double x, bool alternating, out Complex sum)
        {
            double re = 1;
            double im = 0;
            double magnitude = 1;
            double fourNSquared = 4.0 * n * n;
            for (int k = 1; k < 500; k++)
            {
                double odd = 2 * k - 1;
                double next = magnitude * (fourNSquared - odd * odd) / (8 * k * x);
                if (alternating)
                {
                    next = -next;
                }

                if (Math.Abs(next) > Math.Abs(magnitude) && k > 1)
                {
                    sum = Complex.Zero;
                    return false;
                }

                magnitude = next;

                // e^(-ikπ/4)
                KelvinEighthRoot(-k, out int c, out int s, out bool diagonal);
                double scale = diagonal ? OneOverSqrt2 * magnitude : magnitude;
                re += c * scale;
                im += s * scale;
                if (Math.Abs(magnitude) <= 0.25 * MachineEpsilon * Math.Sqrt(re * re + im * im) || magnitude == 0)
                {
                    sum = new Complex(re, im);
                    return true;
                }
            }

            sum = Complex.Zero;
            return false;
        }

        // 成分ごとの符号反転（double * Complex は無限大成分で 0·∞ = NaN になるため避ける）
        private static Complex KelvinNegateIf(bool negate, Complex value) =>
            negate ? new Complex(-value.Real, -value.Imaginary) : value;

        private static Complex KelvinIPower(long n) => (((n % 4) + 4) % 4) switch
        {
            0 => Complex.One,
            1 => Complex.ImaginaryOne,
            2 => -Complex.One,
            _ => -Complex.ImaginaryOne,
        };

        // x/√2 を double-double で返す
        private static (double Hi, double Lo) KelvinXOverSqrt2(double x)
        {
            double hi = x * OneOverSqrt2;
            double lo = Math.FusedMultiplyAdd(x, OneOverSqrt2, -hi) + x * OneOverSqrt2Lo;
            return DoubleDoubleNormalize(hi, lo);
        }

        // ker_n(x) + i kei_n(x)
        private static Complex KelvinKerKei(int n, double x)
        {
            if (double.IsNaN(x) || x < 0 || n == int.MinValue)
            {
                return new Complex(double.NaN, double.NaN);
            }

            // ker_(-n) = (-1)ⁿ ker_n、kei_(-n) = (-1)ⁿ kei_n（DLMF 10.61.6）
            int order = Math.Abs(n);
            double sign = (n < 0 && (order & 1) == 1) ? -1 : 1;
            if (double.IsPositiveInfinity(x))
            {
                return Complex.Zero;
            }

            if (x == 0)
            {
                return KelvinNegateIf(sign < 0, KelvinKerKeiAtZero(order));
            }

            if (x < 1e-140 && order > 0)
            {
                return KelvinNegateIf(sign < 0, KelvinKerKeiTiny(order, x));
            }

            if (x <= KelvinKSeriesLimit)
            {
                return KelvinNegateIf(sign < 0, KelvinKerKeiSeries(order, x));
            }

            // e^z K₀(z)、e^z K₁(z) を求め、前進漸化式で e^z K_n(z) にする（大きさは仮数と指数に分けて持つ）
            KelvinKBase(x, out Complex k0, out Complex k1);
            Complex current = order == 0 ? k0 : k1;
            long exponent = 0;
            if (order >= 2)
            {
                // 2/z = (2/x) e^(-iπ/4)
                Complex twoOverZ = new Complex(OneOverSqrt2, -OneOverSqrt2) * (2 / x);
                Complex previous = k0;
                for (int j = 1; j < order; j++)
                {
                    Complex next = previous + j * twoOverZ * current;
                    previous = current;
                    current = next;

                    // K_n は n とともに急に大きくなるので、オーバーフローする前に仮数と指数に分ける
                    double size = Math.Max(Math.Abs(current.Real), Math.Abs(current.Imaginary));
                    if (size > 1e280)
                    {
                        int shift = Math.ILogB(size);
                        current = new Complex(Math.ScaleB(current.Real, -shift), Math.ScaleB(current.Imaginary, -shift));
                        previous = new Complex(Math.ScaleB(previous.Real, -shift), Math.ScaleB(previous.Imaginary, -shift));
                        exponent += shift;
                    }
                }
            }

            // k_n = (-i)ⁿ K_n(z)、K_n(z) = e^(-z) × current。e^(-z) の位相 x/√2 は double-double で還元する
            Complex rotated = KelvinIPower(-(long)order) * current;
            var g = KelvinXOverSqrt2(x);
            DoubleDoubleCosSin((-g.Hi, -g.Lo), out double cos, out double sin);
            rotated *= new Complex(cos, sin) * (1 - g.Lo);
            double halfDecay = Math.Exp(-0.5 * g.Hi);
            return KelvinNegateIf(sign < 0, new Complex(
                ScaleB(rotated.Real * halfDecay, exponent) * halfDecay,
                ScaleB(rotated.Imaginary * halfDecay, exponent) * halfDecay));
        }

        // e^z K₀(z), e^z K₁(z)（z = x e^(iπ/4), x > 10; x ≤ 30 は台形則、x > 30 は漸近展開）
        private static void KelvinKBase(double x, out Complex k0, out Complex k1)
        {
            if (x > KelvinKAsymptoticLimit && KelvinHankelSum(0, x, false, out Complex s0) && KelvinHankelSum(1, x, false, out Complex s1))
            {
                // √(π/(2z)) = √(π/(2x)) e^(-iπ/8)
                Complex factor = new Complex(Math.Cos(Math.PI / 8), -Math.Sin(Math.PI / 8)) * Math.Sqrt(Math.PI / (2 * x));
                k0 = factor * s0;
                k1 = factor * s1;
                return;
            }

            k0 = KelvinKTrapezoid(0, x);
            k1 = KelvinKTrapezoid(1, x);
        }

        // e^z K_n(z) = ∫₀^∞ e^(-2z sinh²(t/2)) cosh(nt) dt（DLMF 10.32.9、z = x e^(iπ/4)）の台形則
        private static Complex KelvinKTrapezoid(int n, double x)
        {
            double h = Math.Min(0.08, 0.4 / Math.Sqrt(x));
            double re = 0.5;
            double im = 0;
            double c = Sqrt2 * x;
            for (int k = 1; k < 100000; k++)
            {
                double t = k * h;
                double sinhHalf = Math.Sinh(0.5 * t);
                double s2 = sinhHalf * sinhHalf;

                // 2z sinh²(t/2) = √2 x sinh²(t/2) (1 + i)
                double a = c * s2;
                double magnitude = Math.Exp(-a) * (n == 0 ? 1 : Math.Cosh(n * t));
                re += magnitude * Math.Cos(a);
                im -= magnitude * Math.Sin(a);
                if (a - n * t > 42)
                {
                    break;
                }
            }

            return new Complex(h * re, h * im);
        }

        // 冪級数による ker_n + i kei_n（n ≥ 0, 0 < x ≤ 10）。
        // 無限和と有限和を前因子比込みの double-double で合成し、実部・虚部それぞれの打ち消しに耐える
        private static Complex KelvinKerKeiSeries(int n, double x)
        {
            KelvinPowerSeries(n, x, true, out var a, out var b, out var ha, out var hb);
            var log = BesselIntegralLogHalfPlusEulerGamma(x);
            (double Hi, double Lo) quarterPi = (0.25 * Math.PI, 0.25 * PiLo);

            // -(L + iπ/4)(A + iB) + ½(HA + iHB) = (-LA + (π/4)B + ½HA) + i(-LB - (π/4)A + ½HB)
            var real = DoubleDoubleAdd(
                DoubleDoubleAdd(DoubleDoubleMultiply(log, (-a.Hi, -a.Lo)), DoubleDoubleMultiply(quarterPi, b)),
                (0.5 * ha.Hi, 0.5 * ha.Lo));
            var imaginary = DoubleDoubleAdd(
                DoubleDoubleAdd(DoubleDoubleMultiply(log, (-b.Hi, -b.Lo)), DoubleDoubleMultiply(quarterPi, (-a.Hi, -a.Lo))),
                (0.5 * hb.Hi, 0.5 * hb.Lo));
            KelvinRotate(3 * (long)n, ref real, ref imaginary);
            if (n == 0)
            {
                return new Complex(real.Hi + real.Lo, imaginary.Hi + imaginary.Lo);
            }

            // 有限和 F = Σ_(k<n) cₖ (-i y)ᵏ、c₀ = 1、cₖ = cₖ₋₁/(k(n-k))、y = x²/4（(-i)ᵏ = 1, -i, -1, i）
            double halfX = 0.5 * x;
            double yHi = halfX * halfX;
            (double Hi, double Lo) y = (yHi, Math.FusedMultiplyAdd(halfX, halfX, -yHi));
            (double Hi, double Lo) term = (1.0, 0.0);
            (double Hi, double Lo) finiteRe = (1.0, 0.0);
            (double Hi, double Lo) finiteIm = (0.0, 0.0);
            for (int k = 1; k < n; k++)
            {
                term = DoubleDoubleDivide(DoubleDoubleMultiply(term, y), (double)k * (n - k));
                switch (k & 3)
                {
                    case 0:
                        finiteRe = DoubleDoubleAdd(finiteRe, term);
                        break;
                    case 1:
                        finiteIm = DoubleDoubleAdd(finiteIm, (-term.Hi, -term.Lo));
                        break;
                    case 2:
                        finiteRe = DoubleDoubleAdd(finiteRe, (-term.Hi, -term.Lo));
                        break;
                    default:
                        finiteIm = DoubleDoubleAdd(finiteIm, term);
                        break;
                }
            }

            KelvinRotate(-3 * (long)n, ref finiteRe, ref finiteIm);

            // 有限和の前因子 P = ½(x/2)⁻ⁿ (n-1)! と級数の前因子 (x/2)ⁿ/n! の比 R = 2yⁿ/(n! (n-1)!) を double-double で求める。
            // 2 つの部分は打ち消し合って e^(-x/√2) 程度の値になるため、比を double-double で掛けて和を取り P は最後に 1 回だけ掛ける
            (double Hi, double Lo) ratio = (2.0, 0.0);
            for (int j = 1; j <= n; j++)
            {
                ratio = DoubleDoubleDivide(DoubleDoubleMultiply(ratio, y), j < n ? (double)j * j : j);
                if (ratio.Hi == 0)
                {
                    break;
                }
            }

            var re = DoubleDoubleAdd(finiteRe, DoubleDoubleMultiply(ratio, real));
            var im = DoubleDoubleAdd(finiteIm, DoubleDoubleMultiply(ratio, imaginary));
            double mantissa = 0.5 * ScaledPower(halfX, -n, out long powerExponent) * ScaledFactorial(n - 1, out long factorialExponent);
            long exponent = powerExponent + factorialExponent;
            return new Complex(
                StruveScaleProduct(mantissa, exponent, re.Hi + re.Lo),
                StruveScaleProduct(mantissa, exponent, im.Hi + im.Lo));
        }

        // double-double の複素数 re + i im に e^(mπi/4) を掛ける
        private static void KelvinRotate(long m, ref (double Hi, double Lo) re, ref (double Hi, double Lo) im)
        {
            KelvinEighthRoot(m, out int c, out int s, out bool diagonal);
            var real = DoubleDoubleAdd(KelvinScale(re, c), KelvinScale(im, -s));
            var imaginary = DoubleDoubleAdd(KelvinScale(re, s), KelvinScale(im, c));
            if (diagonal)
            {
                real = DoubleDoubleMultiply(real, (OneOverSqrt2, OneOverSqrt2Lo));
                imaginary = DoubleDoubleMultiply(imaginary, (OneOverSqrt2, OneOverSqrt2Lo));
            }

            re = real;
            im = imaginary;
        }

        // x < 10⁻¹⁴⁰, n ≥ 1 での ker_n + i kei_n（有限和の先頭項と、成分が 0 のときは次項のみで評価）
        private static Complex KelvinKerKeiTiny(int n, double x)
        {
            KelvinEighthRoot(-3 * (long)n, out int c, out int s, out bool diagonal);
            double factor = diagonal ? OneOverSqrt2 : 1;
            double leading = 0.5 * ScaledPower(0.5 * x, -n, out long e1) * ScaledFactorial(n - 1, out long f1);
            double re;
            double im;
            if (c != 0 && s != 0)
            {
                re = ScaleB(c * factor * leading, e1 + f1);
                im = ScaleB(s * factor * leading, e1 + f1);
                return new Complex(re, im);
            }

            // 次の項の係数 e^(-3nπi/4)(-i) = (s, -c)（ここでは n は偶数で factor = 1）
            double next = 0.5 * ScaledPower(0.5 * x, 2 - n, out long e2) * ScaledFactorial(n - 2, out long f2);
            re = c != 0 ? ScaleB(c * leading, e1 + f1) : ScaleB(s * next, e2 + f2);
            im = s != 0 ? ScaleB(s * leading, e1 + f1) : ScaleB(-c * next, e2 + f2);
            return new Complex(re, im);
        }

        // x = 0 での ker_n + i kei_n（n ≥ 0; n = 0 は (+∞, -π/4)、n = 2 の実部は 1/2、その他は ±∞）
        private static Complex KelvinKerKeiAtZero(int n)
        {
            if (n == 0)
            {
                return new Complex(double.PositiveInfinity, -0.25 * Math.PI);
            }

            // e^(-3nπi/4) の成分の符号
            KelvinEighthRoot(-3 * (long)n, out int c, out int s, out _);

            // 次の項の係数 e^(-3nπi/4)(-i) = (s, -c) の符号
            double re = c != 0 ? c * double.PositiveInfinity : (n == 2 ? 0.5 * s : s * double.PositiveInfinity);
            double im = s != 0 ? s * double.PositiveInfinity : -c * double.PositiveInfinity;
            return new Complex(re, im);
        }
    }
}
