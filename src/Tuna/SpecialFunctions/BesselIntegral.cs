using static Tuna.XMath.Constants;
using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // |x| ≤ 35 では double-double 冪級数を用いる (これより大きい x では級数の丸め誤差が倍精度を超える)
        private const double BesselIntegralSeriesLimit = 35;

        // |x| > 50 では補助量 A, B に漸近展開を用いる (最小項 πx e^(-x) は x = 50 で 3e-20)。
        // IntegralBesselI0 の正項級数は桁落ちしないため 50 まで使用し、J0/Y0 の (35, 50] は exp-sinh 積分を用いる
        private const double BesselIntegralAsymptoticThreshold = 50;

        // x < 1.5 では IntegralBesselK0 に冪級数を用い、それ以上では π/2 - ∫ₓ^∞ K₀ を使う
        private const double BesselIntegralK0SeriesLimit = 1.5;

        /// <summary>
        /// Computes the integral of the Bessel function of the first kind, <c>∫₀ˣ J₀(t) dt</c>.
        /// <para>第 1 種ベッセル関数の積分 <c>∫₀ˣ J₀(t) dt</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The upper integration limit. / 積分区間の上端。</param>
        /// <returns>The value of <c>∫₀ˣ J₀(t) dt</c> (an odd function approaching <c>±1</c> as <c>x → ±∞</c>). / <c>∫₀ˣ J₀(t) dt</c> の値（奇関数であり、<c>x → ±∞</c> で <c>±1</c> に収束）。</returns>
        /// <seealso href="https://dlmf.nist.gov/10.22.E2"/>
        public static double IntegralBesselJ0(double x)
        {
            if (double.IsNaN(x))
            {
                return double.NaN;
            }

            if (x < 0)
            {
                return -IntegralBesselJ0(-x);
            }

            if (double.IsPositiveInfinity(x))
            {
                return 1;
            }

            if (x <= BesselIntegralSeriesLimit)
            {
                BesselIntegralOscillatorySeries(x, false, out var sum, out _);
                return x * (sum.Hi + sum.Lo);
            }

            BesselIntegralAuxiliary(x, out double a, out double b);
            return 1 + (BesselJ(1, x) * a - BesselJ(0, x) * b);
        }

        /// <summary>
        /// Computes the integral of the Bessel function of the second kind, <c>∫₀ˣ Y₀(t) dt</c>.
        /// <para>第 2 種ベッセル関数の積分 <c>∫₀ˣ Y₀(t) dt</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The non-negative upper integration limit (<c>x ≥ 0</c>). / 非負の積分区間上端（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>∫₀ˣ Y₀(t) dt</c> (<c>0</c> at <c>x = 0</c> and as <c>x → +∞</c>), or <see cref="double.NaN"/> if <paramref name="x"/> is negative or <see cref="double.NaN"/>. / <c>∫₀ˣ Y₀(t) dt</c> の値（<c>x = 0</c> および <c>x → +∞</c> で <c>0</c>）。<paramref name="x"/> が負または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/10.22.E2"/>
        public static double IntegralBesselY0(double x)
        {
            if (double.IsNaN(x) || x < 0)
            {
                return double.NaN;
            }

            if (x == 0 || double.IsPositiveInfinity(x))
            {
                return 0;
            }

            if (x <= BesselIntegralSeriesLimit)
            {
                BesselIntegralOscillatorySeries(x, true, out var sumJ, out var sumY);
                var log = BesselIntegralLogHalfPlusEulerGamma(x);
                var value = DoubleDoubleAdd(DoubleDoubleMultiply(log, sumJ), (-sumY.Hi, -sumY.Lo));
                return 2 / Math.PI * (x * value.Hi + x * value.Lo);
            }

            BesselIntegralAuxiliary(x, out double a, out double b);
            return BesselY(1, x) * a - BesselY(0, x) * b;
        }

        /// <summary>
        /// Computes the integral of the modified Bessel function of the first kind, <c>∫₀ˣ I₀(t) dt</c>.
        /// <para>第 1 種変形ベッセル関数の積分 <c>∫₀ˣ I₀(t) dt</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The upper integration limit. / 積分区間の上端。</param>
        /// <returns>The value of <c>∫₀ˣ I₀(t) dt</c> (an odd function that overflows to <c>±∞</c> for <c>|x| ≳ 713</c>). / <c>∫₀ˣ I₀(t) dt</c> の値（奇関数であり、<c>|x| ≳ 713</c> ではオーバーフローして <c>±∞</c>）。</returns>
        /// <seealso href="https://dlmf.nist.gov/10.43.E2"/>
        public static double IntegralBesselI0(double x)
        {
            if (double.IsNaN(x))
            {
                return double.NaN;
            }

            if (x < 0)
            {
                return -IntegralBesselI0(-x);
            }

            if (double.IsPositiveInfinity(x))
            {
                return double.PositiveInfinity;
            }

            if (x <= BesselIntegralAsymptoticThreshold)
            {
                var sum = BesselIntegralModifiedSeries(x, false, out _);
                return x * (sum.Hi + sum.Lo);
            }

            BesselIntegralAsymptotic(x, true, out double c, out double d);
            double scaled = BesselIScaled(0, x) * d + BesselIScaled(1, x) * c;
            double halfGrowth = Math.Exp(0.5 * x);
            return scaled * halfGrowth * halfGrowth;
        }

        /// <summary>
        /// Computes the integral of the modified Bessel function of the second kind, <c>∫₀ˣ K₀(t) dt</c>.
        /// <para>第 2 種変形ベッセル関数の積分 <c>∫₀ˣ K₀(t) dt</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The non-negative upper integration limit (<c>x ≥ 0</c>). / 非負の積分区間上端（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>∫₀ˣ K₀(t) dt</c> (<c>0</c> at <c>x = 0</c> and <c>π/2</c> as <c>x → +∞</c>), or <see cref="double.NaN"/> if <paramref name="x"/> is negative or <see cref="double.NaN"/>. / <c>∫₀ˣ K₀(t) dt</c> の値（<c>x = 0</c> で <c>0</c>、<c>x → +∞</c> で <c>π/2</c>）。<paramref name="x"/> が負または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/10.43.E2"/>
        public static double IntegralBesselK0(double x)
        {
            if (double.IsNaN(x) || x < 0)
            {
                return double.NaN;
            }

            if (x == 0)
            {
                return 0;
            }

            if (x < BesselIntegralK0SeriesLimit)
            {
                var sumI = BesselIntegralModifiedSeries(x, true, out var sumK);
                var log = BesselIntegralLogHalfPlusEulerGamma(x);
                var value = DoubleDoubleAdd(sumK, DoubleDoubleMultiply(log, (-sumI.Hi, -sumI.Lo)));
                return x * value.Hi + x * value.Lo;
            }

            if (double.IsPositiveInfinity(x))
            {
                return 0.5 * Math.PI;
            }

            // π/2 の下位部分も考慮して裾との差を取る
            double tail = BesselIntegralKTailScaled(0, x) * Math.Exp(-x);
            return 0.5 * Math.PI - (tail - 0.5 * PiLo);
        }

        // ∫₀ˣ J₀ と ∫₀ˣ Y₀ の double-double 冪級数:
        // uₖ = (-1)ᵏ (x/2)^(2k)/(k!)² として sumJ = Σ uₖ/(2k+1), sumY = Σ uₖ (Hₖ + 1/(2k+1))/(2k+1) (withY が true のみ)
        private static void BesselIntegralOscillatorySeries(
            double x, bool withY, out (double Hi, double Lo) sumJ, out (double Hi, double Lo) sumY)
        {
            double halfX = 0.5 * x;
            double qHi = halfX * halfX;
            (double Hi, double Lo) q = (-qHi, -Math.FusedMultiplyAdd(halfX, halfX, -qHi));
            (double Hi, double Lo) term = (1.0, 0.0);
            (double Hi, double Lo) harmonic = (0.0, 0.0);
            sumJ = (1.0, 0.0);
            sumY = (1.0, 0.0);
            for (int k = 1; k < 1000; k++)
            {
                term = DoubleDoubleDivide(DoubleDoubleMultiply(term, q), (double)k * k);
                int odd = 2 * k + 1;
                var contribution = DoubleDoubleDivide(term, odd);
                sumJ = DoubleDoubleAdd(sumJ, contribution);
                if (withY)
                {
                    harmonic = DoubleDoubleAdd(harmonic, DoubleDoubleDivide((1.0, 0.0), k));
                    var weight = DoubleDoubleAdd(harmonic, DoubleDoubleDivide((1.0, 0.0), odd));
                    sumY = DoubleDoubleAdd(sumY, DoubleDoubleMultiply(contribution, weight));
                }

                // 項の比 (x/2)²/k² が1未満になった後で、項が和の 2⁻¹¹⁰ 以下になれば打ち切る（Y の側は (Hₖ + 1/(2k+1)) の分だけ大きい）
                if (k * k > qHi && Math.Abs(contribution.Hi) * (1 + Math.Log(k + 1)) <= DoubleDoubleEpsilon * Math.Abs(sumJ.Hi))
                {
                    break;
                }
            }
        }

        // ∫₀ˣ I₀ と ∫₀ˣ K₀ の double-double 冪級数 (全項正):
        // aₖ = (x/2)^(2k)/((k!)² (2k+1)) として 戻り値 = Σ aₖ, sumK = Σ aₖ (Hₖ + 1/(2k+1)) (withK が true のみ)
        private static (double Hi, double Lo) BesselIntegralModifiedSeries(double x, bool withK, out (double Hi, double Lo) sumK)
        {
            double halfX = 0.5 * x;
            double qHi = halfX * halfX;
            (double Hi, double Lo) q = (qHi, Math.FusedMultiplyAdd(halfX, halfX, -qHi));
            (double Hi, double Lo) term = (1.0, 0.0);
            (double Hi, double Lo) harmonic = (0.0, 0.0);
            (double Hi, double Lo) sum = (1.0, 0.0);
            sumK = (1.0, 0.0);
            for (int k = 1; k < 1000; k++)
            {
                term = DoubleDoubleDivide(DoubleDoubleMultiply(term, q), (double)k * k);
                int odd = 2 * k + 1;
                var contribution = DoubleDoubleDivide(term, odd);
                sum = DoubleDoubleAdd(sum, contribution);
                if (withK)
                {
                    harmonic = DoubleDoubleAdd(harmonic, DoubleDoubleDivide((1.0, 0.0), k));
                    var weight = DoubleDoubleAdd(harmonic, DoubleDoubleDivide((1.0, 0.0), odd));
                    sumK = DoubleDoubleAdd(sumK, DoubleDoubleMultiply(contribution, weight));
                }

                if (k * k > qHi && contribution.Hi * (1 + Math.Log(k + 1)) <= DoubleDoubleEpsilon * sum.Hi)
                {
                    break;
                }
            }

            return sum;
        }

        // IntegralBesselJ0 / Y0 の補助量 A = (πx/2)K₀(x), B = (πx/2)(K₁(x) - 2/π) (K_ν はシュトルーベの K 関数, x > 35)。
        // x > 50 では漸近展開、それ以下では正値積分表示を StruveExpSinhRule で計算する (特異点 τ = ±ix は十分遠い)
        private static void BesselIntegralAuxiliary(double x, out double a, out double b)
        {
            if (x > BesselIntegralAsymptoticThreshold)
            {
                BesselIntegralAsymptotic(x, false, out a, out b);
                return;
            }

            double[] nodes = StruveExpSinhRule.Nodes;
            double[] weights = StruveExpSinhRule.Weights;
            double sumA = 0;
            double sumB = 0;
            for (int i = 0; i < nodes.Length; i++)
            {
                double tau = nodes[i];
                if (tau > 800)
                {
                    break;
                }

                double u = tau / x;
                double root = Math.Sqrt(1 + u * u);
                double w = weights[i] * Math.Exp(-tau);
                sumA += w / root;
                sumB += w * u * u / (root + 1);
            }

            a = sumA;
            b = x * sumB;
        }

        // 大引数向け補助量 (シュトルーベ K_ν, M_ν の漸近展開; DLMF 11.6)。
        // modified = false なら A = Σ sₖ, B = x Σ pₖ、true なら C = Σ |sₖ|, D = x Σ |pₖ| を返す。
        // 最小項は k ≈ x/2 でおよそ πx e^(-x) となり、x > 50 では倍精度以下で打ち切れる
        private static void BesselIntegralAsymptotic(double x, bool modified, out double first, out double second)
        {
            double r = 4 / (x * x);
            double s = 1;
            first = 1;
            for (int k = 0; k < 200; k++)
            {
                double next = s * (k + 0.5) * (k + 0.5) * r;
                next = modified ? next : -next;
                if (Math.Abs(next) > Math.Abs(s) || Math.Abs(next) <= 0.125 * MachineEpsilon)
                {
                    break;
                }

                first += next;
                s = next;
            }

            double p = 1 / (x * x);
            double sum = p;
            for (int k = 1; k < 200; k++)
            {
                double next = p * (k * (double)k - 0.25) * r;
                next = modified ? next : -next;
                if (Math.Abs(next) > Math.Abs(p) || x * Math.Abs(next) <= 0.125 * MachineEpsilon * Math.Abs(x * sum))
                {
                    break;
                }

                sum += next;
                p = next;
            }

            second = x * sum;
        }

        // e^x ∫ₓ^∞ K_ν(t) dt = ∫₀^∞ e^(-2x sinh²(u/2)) cosh(νu)/cosh u du (x > 0; DLMF 10.32.E9 を t 積分)。
        // x cosh u - x = 2x sinh²(u/2) で桁落ちを避け、刻み幅 h = min(0.15, 0.6/√x) の台形則で求める
        private static double BesselIntegralKTailScaled(double nu, double x)
        {
            double h = Math.Min(0.15, 0.6 / Math.Sqrt(x));
            double sum = 0.5;
            double slope = Math.Abs(nu) - 1;
            for (int k = 1; k < 100000; k++)
            {
                double u = k * h;
                double sinhHalf = Math.Sinh(0.5 * u);
                double exponent = 2 * x * sinhHalf * sinhHalf;
                sum += Math.Exp(-exponent) * (nu == 0 ? 1 : Math.Cosh(nu * u)) / Math.Cosh(u);

                // 被積分関数の対数は -2x sinh²(u/2) + (|ν| - 1)u 程度で、u = 0 の値 1 から e^(-42) 以下になれば打ち切る
                if (exponent - slope * u > 42)
                {
                    break;
                }
            }

            return h * sum;
        }

        // ln(x/2) + γ を double-double で計算する (x > 0)。
        // x/2 = m 2ᵉ (√½ ≤ m < √2) として ln m = 2 artanh((m-1)/(m+1)) を級数展開し、e ln 2 と γ の上位・下位成分を加算する (ケルビン関数 ker/kei 等で対数項と級数の桁落ちを避ける用途でも共用)
        private static (double Hi, double Lo) BesselIntegralLogHalfPlusEulerGamma(double x)
        {
            int e = Math.ILogB(x) - 1;
            double m = Math.ScaleB(x, -Math.ILogB(x));
            if (m >= Sqrt2)
            {
                m *= 0.5;
                e++;
            }

            // m - 1 はスターベンツの補題で正確、m + 1 は TwoSum で正確に表す
            var numerator = (m - 1, 0.0);
            var denominator = DoubleDoubleTwoSum(m, 1);
            var s = DoubleDoubleDivide(numerator, denominator);
            var s2 = DoubleDoubleMultiply(s, s);
            var power = s;
            var atanh = s;
            for (int j = 1; j < 40; j++)
            {
                power = DoubleDoubleMultiply(power, s2);
                var term = DoubleDoubleDivide(power, 2 * j + 1);
                atanh = DoubleDoubleAdd(atanh, term);
                if (Math.Abs(term.Hi) <= DoubleDoubleEpsilon * Math.Abs(atanh.Hi) || term.Hi == 0)
                {
                    break;
                }
            }

            double productHi = e * Ln2;
            double productLo = Math.FusedMultiplyAdd(e, Ln2, -productHi) + e * Ln2Lo;
            var log = DoubleDoubleAdd((2 * atanh.Hi, 2 * atanh.Lo), DoubleDoubleNormalize(productHi, productLo));
            return DoubleDoubleAdd(log, DoubleDoubleNormalize(EulerGamma, EulerGammaLo));
        }
    }
}
