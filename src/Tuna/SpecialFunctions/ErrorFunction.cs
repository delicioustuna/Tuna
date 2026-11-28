using static Tuna.XMath.Constants;
using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // erf(x) < 1/2 < erfc(x) の範囲（erfc(x) = 1 - erf(x) の上限）。これ以上は erfc(x) = e^(-x²) erfcx(x) を使う
        private const double ErfcCentralLimit = 0.47;

        // |x| < 1 は ErfCoefficients の多項式、これ以上は erf(x) = 1 - erfc(x)（erfc(x) ≤ 0.16 のため桁落ちしない）
        private const double ErfPolynomialLimit = 1;

        // これより大きい x では erfc(x) < 2⁻⁵⁴ となり erf(x) は 1 に丸められる
        private const double ErfRoundsToOne = 6;

        // これより大きい x では erfc(x) が最小の非正規化数の半分を下回り 0 に丸められる
        private const double ErfcUnderflow = 27.3;

        // erfcx(x) を漸近形 H(1/x²)/x（ErfcxAsymptoticCoefficients）で計算する下限
        private const double ErfcxAsymptoticThreshold = 4;

        // ErfcxCoefficients の 1 区間の係数数（12 次多項式 + 定数項の下位）
        private const int ErfcxPieceLength = 14;

        // InverseErfcTailCoefficients の 1 区間の係数数（11 次多項式）
        private const int InverseErfcTailPieceLength = 12;

        // ドーソン関数の Rybicki 法の刻み幅と係数 exp(-((2i+1)h)²)
        private const double DawsonStep = 0.2;

        private static readonly double[] DawsonCoefficients = CreateDawsonCoefficients();

        private static double[] CreateDawsonCoefficients()
        {
            var coefficients = new double[17];
            for (int i = 0; i < coefficients.Length; i++)
            {
                double t = (2 * i + 1) * DawsonStep;
                coefficients[i] = Math.Exp(-t * t);
            }

            return coefficients;
        }

        /// <summary>
        /// Computes the error function <c>erf(x)</c>.
        /// <para>誤差関数 <c>erf(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>erf(x)</c>. / <c>erf(x)</c> の値。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Error_function"/>
        public static double Erf(double x)
        {
            double ax = Math.Abs(x);
            if (ax < ErfPolynomialLimit)
            {
                // ±0 は x + x R(0) = ±0 で符号が保たれる
                return ErfSmall(x);
            }

            if (double.IsNaN(x))
            {
                return x;
            }

            double value = ax < ErfRoundsToOne ? 1 - ErfcPositive(ax) : 1;
            return x < 0 ? -value : value;
        }

        /// <summary>
        /// Computes the complementary error function <c>erfc(x) = 1 - erf(x)</c>.
        /// <para>相補誤差関数 <c>erfc(x) = 1 - erf(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>erfc(x)</c>. / <c>erfc(x)</c> の値。</returns>
        /// <remarks>
        /// <para>Preserves relative accuracy for large positive <paramref name="x"/> where <c>1 - Erf(x)</c> would suffer from catastrophic cancellation.</para>
        /// <para><c>1 - Erf(x)</c> では桁落ちが生じる大きな正の <paramref name="x"/> に対しても相対精度を保ちます。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Error_function#Complementary_error_function"/>
        public static double Erfc(double x)
        {
            if (x >= ErfcCentralLimit)
            {
                return ErfcPositive(x);
            }

            if (x > -ErfcCentralLimit)
            {
                return 1 - ErfSmall(x);
            }

            if (double.IsNaN(x))
            {
                return x;
            }

            return 2 - ErfcPositive(-x);
        }

        /// <summary>
        /// Computes the scaled complementary error function <c>erfcx(x) = e^(x²) erfc(x)</c>.
        /// <para>スケーリングされた相補誤差関数 <c>erfcx(x) = e^(x²) erfc(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>erfcx(x)</c>. / <c>erfcx(x)</c> の値。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Error_function#Complementary_error_function"/>
        public static double Erfcx(double x)
        {
            if (x >= 0)
            {
                return ErfcxPositive(x);
            }

            if (double.IsNaN(x))
            {
                return x;
            }

            if (x < -26.7)
            {
                return double.PositiveInfinity;
            }

            // erfcx(x) = 2e^(x²) - erfcx(-x)。2e^(x²) ≥ 2 > 1 ≥ erfcx(-x) なので桁落ちしない
            return 2 * ExpSquare(x) - ErfcxPositive(-x);
        }

        /// <summary>
        /// Computes the imaginary error function <c>erfi(x) = -i erf(ix)</c>.
        /// <para>虚誤差関数 <c>erfi(x) = -i erf(ix)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>erfi(x)</c>. / <c>erfi(x)</c> の値。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Error_function#Imaginary_error_function"/>
        public static double Erfi(double x)
        {
            if (double.IsNaN(x) || double.IsInfinity(x))
            {
                return x;
            }

            double ax = Math.Abs(x);
            if (ax < 3)
            {
                // erfi(x) = (2/√π) Σ x^(2n+1) / (n! (2n+1))（すべて正の項）
                double x2 = x * x;
                double term = x;
                double sum = x;
                for (int n = 1; n < 200; n++)
                {
                    term *= x2 / n;
                    double t = term / (2 * n + 1);
                    sum += t;
                    if (Math.Abs(t) <= MachineEpsilon * Math.Abs(sum) * 0.5)
                    {
                        break;
                    }
                }

                return TwoOverSqrtPi * sum;
            }

            if (ax > 27)
            {
                return x > 0 ? double.PositiveInfinity : double.NegativeInfinity;
            }

            // erfi(x) = (2/√π) e^(x²) F(x)。e^(x²) は最終値より先にオーバーフローしうるので、
            // e^(x²/2) を2回に分けて掛け、x² の丸め誤差は最後に補正する
            double hi = x * x;
            double lo = Math.FusedMultiplyAdd(x, x, -hi);
            double half = Math.Exp(0.5 * hi);
            double value = TwoOverSqrtPi * Dawson(x) * half * half;
            return double.IsInfinity(value) ? value : value + value * lo;
        }

        /// <summary>
        /// Computes Dawson's integral <c>F(x) = e^(-x²) ∫₀ˣ e^(t²) dt</c>.
        /// <para>ドーソン関数 <c>F(x) = e^(-x²) ∫₀ˣ e^(t²) dt</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>F(x)</c>. / <c>F(x)</c> の値。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Dawson_function"/>
        public static double Dawson(double x)
        {
            if (double.IsNaN(x))
            {
                return x;
            }

            double ax = Math.Abs(x);
            if (double.IsInfinity(ax))
            {
                return 0.0 * Math.Sign(x);
            }

            double value;
            if (ax < 0.2)
            {
                // F(x) = Σ (-1)ⁿ 2ⁿ x^(2n+1) / (2n+1)!!
                double twoX2 = 2 * ax * ax;
                double term = ax;
                value = ax;
                for (int n = 1; n < 50; n++)
                {
                    term *= -twoX2 / (2 * n + 1);
                    value += term;
                    if (Math.Abs(term) <= MachineEpsilon * value * 0.5)
                    {
                        break;
                    }
                }
            }
            else if (ax < 10)
            {
                // Rybicki 法: F(x) = lim_{h→0} (1/√π) Σ_{n 奇数} e^(-(x-nh)²)/n
                int n0 = 2 * (int)Math.Round(0.5 * ax / DawsonStep);
                double xp = ax - n0 * DawsonStep;
                double e1 = Math.Exp(2 * xp * DawsonStep);
                double e2 = e1 * e1;
                double d1 = n0 + 1;
                double d2 = d1 - 2;
                double sum = 0;
                for (int i = 0; i < DawsonCoefficients.Length; i++)
                {
                    sum += DawsonCoefficients[i] * (e1 / d1 + 1 / (d2 * e1));
                    d1 += 2;
                    d2 -= 2;
                    e1 *= e2;
                }

                value = OneOverSqrtPi * Math.Exp(-xp * xp) * sum;
            }
            else
            {
                // F(x) ~ (1/(2x)) Σ (2n-1)!! / (2x²)ⁿ
                double inverse2X2 = 1 / (2 * ax * ax);
                double term = 1;
                double sum = 1;
                for (int n = 1; n < 1000; n++)
                {
                    double next = term * (2 * n - 1) * inverse2X2;
                    if (next >= term)
                    {
                        break;
                    }

                    term = next;
                    sum += term;
                    if (term <= MachineEpsilon * sum * 0.5)
                    {
                        break;
                    }
                }

                // 2x はオーバーフローしうるので先に 1/2 を掛ける
                value = 0.5 * sum / ax;
            }

            return x < 0 ? -value : value;
        }

        /// <summary>
        /// Computes the inverse error function <c>erf⁻¹(x)</c>.
        /// <para>逆誤差関数 <c>erf⁻¹(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input value in the range <c>[-1, 1]</c>. / 範囲 <c>[-1, 1]</c> の入力値。</param>
        /// <returns>The value <c>y</c> such that <c>erf(y) = x</c>, <see cref="double.PositiveInfinity"/> at <c>1</c>, <see cref="double.NegativeInfinity"/> at <c>-1</c>, or <see cref="double.NaN"/> outside <c>[-1, 1]</c>. / <c>erf(y) = x</c> を満たす <c>y</c>。<c>1</c> では <see cref="double.PositiveInfinity"/>、<c>-1</c> では <see cref="double.NegativeInfinity"/>、<c>[-1, 1]</c> 外では <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Error_function#Inverse_functions"/>
        public static double InverseErf(double x)
        {
            double ax = Math.Abs(x);
            if (!(ax <= 1))
            {
                return double.NaN;
            }

            if (ax <= 0.5)
            {
                return InverseErfCentral(x);
            }

            // 1 - |x| は丸め誤差なく求まる
            double value = InverseErfcSmall(1 - ax);
            return x < 0 ? -value : value;
        }

        /// <summary>
        /// Computes the inverse complementary error function <c>erfc⁻¹(q)</c>.
        /// <para>逆相補誤差関数 <c>erfc⁻¹(q)</c> を計算します。</para>
        /// </summary>
        /// <param name="q">The input value in the range <c>[0, 2]</c>. / 範囲 <c>[0, 2]</c> の入力値。</param>
        /// <returns>The value <c>y</c> such that <c>erfc(y) = q</c>, <see cref="double.PositiveInfinity"/> at <c>0</c>, <see cref="double.NegativeInfinity"/> at <c>2</c>, or <see cref="double.NaN"/> outside <c>[0, 2]</c>. / <c>erfc(y) = q</c> を満たす <c>y</c>。<c>0</c> では <see cref="double.PositiveInfinity"/>、<c>2</c> では <see cref="double.NegativeInfinity"/>、<c>[0, 2]</c> 外では <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Error_function#Inverse_functions"/>
        public static double InverseErfc(double q)
        {
            if (!(q >= 0 && q <= 2))
            {
                return double.NaN;
            }

            if (q < 0.5)
            {
                return InverseErfcSmall(q);
            }

            // 1 - q と 2 - q は丸め誤差なく求まる
            return q <= 1.5 ? InverseErfCentral(1 - q) : -InverseErfcSmall(2 - q);
        }

        // |x| < 1 での erf(x) = x + x R(x²)（R の丸め誤差は小さい項にしか入らない）
        private static double ErfSmall(double x) =>
            Math.FusedMultiplyAdd(x, Polynomial(ErfCoefficients, x * x), x);

        // x ≥ ErfcCentralLimit での erfc(x) = e^(-x²) erfcx(x)
        private static double ErfcPositive(double x)
        {
            if (!(x <= ErfcUnderflow))
            {
                return double.IsNaN(x) ? x : 0;
            }

            // x² = hi + lo として e^(-x²) = e^(-hi)(1 - lo)。(1 - lo) は erfcx の最後の丸めの前に掛ける。
            // 結果が非正規化数になる範囲でも、e^(-hi) を先に非正規化数へ丸めた誤差は erfcx(x) ≤ 0.02 倍に縮むため、最後の丸めが支配的
            double hi = x * x;
            double lo = Math.FusedMultiplyAdd(x, x, -hi);
            return Math.Exp(-hi) * ErfcxPositive(x, 0, -lo);
        }

        private static double ErfcxPositive(double x) => ErfcxPositive(x, 0, 0);

        // x ≥ 0 での erfcx(x + dx)(1 + scale)。
        // dx と scale は丸め誤差程度の補正で、1 次の項だけを最後の丸めの前に足す（補正のために丸めを増やさない）。
        // [0, 4) は区分多項式（ErfcxCoefficients）、x ≥ 4 は H(1/x²)/x（ErfcxAsymptoticCoefficients）
        private static double ErfcxPositive(double x, double dx, double scale)
        {
            ReadOnlySpan<double> c;
            double s;
            if (x < 0.25)
            {
                c = ErfcxCoefficients[..ErfcxPieceLength];
                s = x;
            }
            else if (x < ErfcxAsymptoticThreshold)
            {
                // 2 倍ごとの区間 [2^e, 2^(e+1)) を 4 等分した区間の番号は、指数と仮数の上位 2 ビットで決まる（1/4 の指数部は 1021 で、区間 1）。
                // 区間の始点 start は仮数の下位 50 ビットを 0 にした値、半幅の逆数は 8·2^-e で、
                // x - start と 2 の冪の積は丸め誤差なく求まるので、区間内の位置 s ∈ [-1, 1) も正確
                long bits = BitConverter.DoubleToInt64Bits(x);
                int piece = (int)(bits >> 50) - ((1021 << 2) - 1);
                double start = BitConverter.Int64BitsToDouble(bits & ~((1L << 50) - 1));
                double inverseHalfWidth = BitConverter.Int64BitsToDouble((2049 - (bits >> 52)) << 52);
                c = ErfcxCoefficients.Slice(piece * ErfcxPieceLength, ErfcxPieceLength);
                s = (x - start) * inverseHalfWidth - 1;
            }
            else
            {
                if (double.IsPositiveInfinity(x))
                {
                    return 0;
                }

                // erfcx(x) = H/x。1/x = u + uLo（uLo は u の丸め誤差）とし、H の定数項の上位と u の積を誤差なしの 2 数に分けて、
                // 最後の丸めを 1 回にする。d ln erfcx/dx = -1/x (1 + O(1/x²)) なので、dx の補正は -dx/x とする（補正の相対誤差 1/16 以下は効かない）
                double u = 1 / x;
                double uLo = Math.FusedMultiplyAdd(-u, x, 1) * u;
                double v = u * u;
                ReadOnlySpan<double> a = ErfcxAsymptoticCoefficients;
                double lowH = Math.FusedMultiplyAdd(PolynomialTail(a, v), v, a[^1]);
                lowH += (a[0] + lowH) * (scale - dx * u);
                double product = a[0] * u;
                double productLo = Math.FusedMultiplyAdd(a[0], u, -product);
                return product + (productLo + Math.FusedMultiplyAdd(a[0], uLo, lowH * u));
            }

            double low = Math.FusedMultiplyAdd(PolynomialTail(c, s), s, c[^1]);
            double e = c[0] + low;
            // erfcx(x + dx)(1 + scale) ≈ erfcx(x)(1 + scale) + dx erfcx'(x)、erfcx'(x) = 2x erfcx(x) - 2/√π
            return c[0] + (low + Math.FusedMultiplyAdd(e, scale + 2 * x * dx, -TwoOverSqrtPi * dx));
        }

        // |w| ≤ 1/2 での erf⁻¹(w)。初期値 w P(w²)（相対 2⁻³⁴ 程度）を erf(y) = w の Halley 法 1 回で仕上げる
        private static double InverseErfCentral(double w) =>
            RefineInverseErf(w * Polynomial(InverseErfCentralCoefficients, w * w), w, 0);

        // erf(y) = w + wLo（|y| < 1）を初期値 y から Halley 法 1 回で解く。
        // 残差 erf(y) - w = (y - w) + y R(y²) の y - w は丸め誤差なく求まる（erf(y)/y ∈ [0.84, 1.13]）ため、残差の丸め誤差は R の項のみに入る
        private static double RefineInverseErf(double y, double w, double wLo)
        {
            // f(y) = erf(y) - w、f'(y) = (2/√π) e^(-y²)、f''/f' = -2y
            double z = y * y;
            double residual = Math.FusedMultiplyAdd(y, Polynomial(ErfCoefficients, z), y - w) - wLo;
            double step = residual * (0.5 * SqrtPi) * Math.Exp(z);
            return y - step / (1 + y * step);
        }

        // 0 ≤ q < 1/2 での erfc⁻¹(q)。r = √(-ln q) の区分多項式（相対 2⁻³⁴ 程度）を初期値とし、Halley 法 1 回で仕上げる
        private static double InverseErfcSmall(double q)
        {
            if (q == 0)
            {
                return double.PositiveInfinity;
            }

            if (q < MinNormal)
            {
                // e^(y²) がオーバーフローする裾。erfc⁻¹(q) = Q⁻¹(q/2)/√2 を対数で解く
                return InverseNormalLogSfTail(Math.Log(q) - Ln2) * OneOverSqrt2;
            }

            double y = InverseErfcInitialGuess(Math.Sqrt(-Math.Log(q)));
            if (y < 1)
            {
                // erfc(y) - q = (1 - q) - erf(y)。1 - q を上位と下位に分けると、残差は erf の形で桁落ちなく求まる
                double w = 1 - q;
                return RefineInverseErf(y, w, (1 - w) - q);
            }

            // g(y) = erfc(y) - q、g'(y) = -(2/√π) e^(-y²)、g''/g' = -2y。g/g' = -(√π/2)(erfcx(y) - q e^(y²))
            double step = -0.5 * SqrtPi * (ErfcxPositive(y) - q * ExpSquare(y));
            return y - step / (1 + y * step);
        }

        // r = √(-ln q) > 0.83（q < 1/2）から erfc⁻¹(q) の初期値（相対 2⁻³⁴ 程度）を求める
        private static double InverseErfcInitialGuess(double r)
        {
            // 区間 [0.83, 1)、[1, 2)、…、[16, 28] の番号は r の指数 + 1
            int piece = (int)(BitConverter.DoubleToInt64Bits(r) >> 52) - 1022;
            double s = (r - InverseErfcTailCenters[piece]) * InverseErfcTailScales[piece];
            return Polynomial(InverseErfcTailCoefficients.Slice(piece * InverseErfcTailPieceLength, InverseErfcTailPieceLength), s);
        }

        // ホーナー法による c[0] + c[1] s + … + c[n-1] s^(n-1) の評価
        private static double Polynomial(ReadOnlySpan<double> c, double s)
        {
            double p = c[^1];
            for (int i = c.Length - 2; i >= 0; i--)
            {
                p = Math.FusedMultiplyAdd(p, s, c[i]);
            }

            return p;
        }

        // 定数項を上位 c[0] と下位 c[^1] に分けて持つ多項式（定数項の丸め誤差を省き最後の丸めを 1 回にする）
        private static double PolynomialSplitConstant(ReadOnlySpan<double> c, double s) =>
            c[0] + Math.FusedMultiplyAdd(PolynomialTail(c, s), s, c[^1]);

        // PolynomialSplitConstant の 1 次以上の部分を s で割った c[1] + c[2] s + … + c[^2] s^(n-3)
        private static double PolynomialTail(ReadOnlySpan<double> c, double s)
        {
            double p = c[^2];
            for (int i = c.Length - 3; i >= 1; i--)
            {
                p = Math.FusedMultiplyAdd(p, s, c[i]);
            }

            return p;
        }
    }
}
