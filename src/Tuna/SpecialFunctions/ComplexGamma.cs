using System.Numerics;
using static Tuna.XMath.Constants;
using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // 複素スターリング級数・ディガンマ漸近展開を直接使う |z| の下限 (|z| = 6 では k = 15 までに項が 2×10⁻¹⁷ 未満)
        private const double ComplexGammaAsymptoticRadius = 6;

        // 1 - γ (γ はオイラー・マスケローニ定数)
        private const double ComplexGammaOneMinusEuler = 0.42278433509846713;

        // ln Γ(2 + w) = (1 - γ) w + Σ_(k≥2) (-1)ᵏ (ζ(k) - 1) wᵏ/k (|w| < 2; DLMF 5.7.3) の係数 (-1)ᵏ (ζ(k) - 1)/k (k = 2..50)。
        // 倍精度の ζ(k) - 1 は大きな k で桁落ちするため高精度値を事前計算している
        private static readonly double[] ComplexGammaShiftedZetaCoefficients =
        [
            0.3224670334241132, -0.0673523010531981, 0.020580808427784546, -0.007385551028673986, 0.0028905103307415234,
            -0.001192753911703261, 0.0005096695247430425, -0.00022315475845357939, 9.945751278180853e-05, -4.492623673813314e-05,
            2.050721277567069e-05, -9.439488275268397e-06, 4.374866789907488e-06, -2.039215753801366e-06, 9.55141213040742e-07,
            -4.492469198764566e-07, 2.1207184805554665e-07, -1.0043224823968099e-07, 4.7698101693639804e-08, -2.2711094608943164e-08,
            1.0838659214896955e-08, -5.183475041970047e-09, 2.4836745438024785e-09, -1.1921401405860912e-09, 5.731367241678862e-10,
            -2.7595228851242334e-10, 1.330476437424449e-10, -6.4229645638381e-11, 3.1044247747322276e-11, -1.5021384080754142e-11,
            7.275974480239079e-12, -3.527742476575915e-12, 1.711991790559618e-12, -8.315385841420285e-13, 4.04220052528944e-13,
            -1.9664756310966165e-13, 9.573630387838556e-14, -4.6640760264283744e-14, 2.2737369600659724e-14, -1.1091399470834522e-14,
            5.413659156725363e-15, -2.643880017860995e-15, 1.2918959062789966e-15, -6.315935504198448e-16, 3.089316266963393e-16,
            -1.5117930628108198e-16, 7.40148685695232e-17, -3.625218048120654e-17, 1.7763568421861633e-17,
        ];

        /// <summary>
        /// Computes the complex gamma function <c>Γ(z)</c>.
        /// <para>複素ガンマ関数 <c>Γ(z)</c> を計算します。</para>
        /// </summary>
        /// <param name="z">The complex input value. / 複素数の入力値。</param>
        /// <returns>The value of <c>Γ(z)</c>. / <c>Γ(z)</c> の値。</returns>
        /// <remarks>
        /// <para>On the real axis, this matches <see cref="Gamma(double)"/>. Non-positive integer poles, <see cref="double.NaN"/> components, and infinite inputs (except real <c>+∞</c>) yield <see cref="double.NaN"/>.</para>
        /// <para>実軸上では <see cref="Gamma(double)"/> と一致します。0 以下の整数（極）、<see cref="double.NaN"/> を含む引数、および（実軸上の <c>+∞</c> を除く）無限大の引数では <see cref="double.NaN"/> を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/5.2#E1"/>
        public static Complex Gamma(Complex z)
        {
            double x = z.Real;
            double y = z.Imaginary;
            if (double.IsNaN(x) || double.IsNaN(y) || double.IsInfinity(y) || (double.IsInfinity(x) && y != 0))
            {
                return new Complex(double.NaN, double.NaN);
            }

            if (y == 0)
            {
                return new Complex(Gamma(x), y);
            }

            return ComplexGammaExp(LogGamma(z));
        }

        /// <summary>
        /// Computes the reciprocal of the complex gamma function, <c>1 / Γ(z)</c>.
        /// <para>複素ガンマ関数の逆数 <c>1 / Γ(z)</c>（整関数）を計算します。</para>
        /// </summary>
        /// <param name="z">The complex input value. / 複素数の入力値。</param>
        /// <returns>The value of <c>1 / Γ(z)</c>, which is <c>0</c> at non-positive integers. / <c>1 / Γ(z)</c> の値（0 以下の整数では <c>0</c>）。</returns>
        /// <remarks>
        /// <para>On the real axis, this matches <see cref="ReciprocalGamma(double)"/>. Inputs containing <see cref="double.NaN"/> or infinite components (except real <c>+∞</c>) yield <see cref="double.NaN"/>.</para>
        /// <para>実軸上では <see cref="ReciprocalGamma(double)"/> と一致します。<see cref="double.NaN"/> を含む引数や（実軸上の <c>+∞</c> を除く）無限大の引数では <see cref="double.NaN"/> を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/5.2#E1"/>
        public static Complex ReciprocalGamma(Complex z)
        {
            double x = z.Real;
            double y = z.Imaginary;
            if (double.IsNaN(x) || double.IsNaN(y) || double.IsInfinity(y) || (double.IsInfinity(x) && y != 0))
            {
                return new Complex(double.NaN, double.NaN);
            }

            if (y == 0)
            {
                return new Complex(ReciprocalGamma(x), y);
            }

            return ComplexGammaExp(-LogGamma(z));
        }

        /// <summary>
        /// Computes the principal branch of the complex log-gamma function, <c>ln Γ(z)</c>.
        /// <para>複素対数ガンマ関数 <c>ln Γ(z)</c> の主枝を計算します。</para>
        /// </summary>
        /// <param name="z">The complex input value. / 複素数の入力値。</param>
        /// <returns>The principal branch value of <c>ln Γ(z)</c>. / <c>ln Γ(z)</c> の主枝の値。</returns>
        /// <remarks>
        /// <para>This branch is real on the positive real axis and analytic everywhere except along the branch cut on the non-positive real axis; its imaginary part may differ from <c>ln(Γ(z))</c> by an integer multiple of <c>2π</c>. Along the negative real axis, signed zero imaginary parts <c>+0</c> and <c>-0</c> select the limit from the upper and lower half-plane, respectively. Non-positive integer poles yield <c>+∞</c> with zero imaginary part.</para>
        /// <para>正の実軸上で実数となり、負の実軸に沿った分岐切断以外で解析的な主枝を返します（虚部は <c>ln(Γ(z))</c> と <c>2π</c> の整数倍だけ異なりえます）。負の実軸上では虚部の符号 <c>+0</c>・<c>-0</c> によりそれぞれ上側・下側からの極限を取ります。0 以下の整数（極）では実部 <c>+∞</c>、虚部 <c>0</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/5.4#E1"/>
        public static Complex LogGamma(Complex z)
        {
            double x = z.Real;
            double y = z.Imaginary;
            if (double.IsNaN(x) || double.IsNaN(y) || double.IsInfinity(y))
            {
                return new Complex(double.NaN, double.NaN);
            }

            if (double.IsInfinity(x))
            {
                return x > 0 && y == 0 ? new Complex(double.PositiveInfinity, y) : new Complex(double.NaN, double.NaN);
            }

            if (double.IsNegative(y))
            {
                return Complex.Conjugate(ComplexLogGammaUpper(x, -y));
            }

            return ComplexLogGammaUpper(x, y);
        }

        /// <summary>
        /// Computes the complex digamma function <c>ψ(z) = Γ'(z) / Γ(z)</c>.
        /// <para>複素ディガンマ関数 <c>ψ(z) = Γ'(z) / Γ(z)</c> を計算します。</para>
        /// </summary>
        /// <param name="z">The complex input value. / 複素数の入力値。</param>
        /// <returns>The value of <c>ψ(z)</c>. / <c>ψ(z)</c> の値。</returns>
        /// <remarks>
        /// <para>On the real axis, this matches <see cref="Digamma(double)"/>. Inputs containing <see cref="double.NaN"/> or infinite components (except real <c>+∞</c>) yield <see cref="double.NaN"/>.</para>
        /// <para>実軸上では <see cref="Digamma(double)"/> と一致します。<see cref="double.NaN"/> を含む引数や（実軸上の <c>+∞</c> を除く）無限大の引数では <see cref="double.NaN"/> を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/5.2#E2"/>
        public static Complex Digamma(Complex z)
        {
            double x = z.Real;
            double y = z.Imaginary;
            if (double.IsNaN(x) || double.IsNaN(y) || double.IsInfinity(y))
            {
                return new Complex(double.NaN, double.NaN);
            }

            if (y == 0)
            {
                return new Complex(Digamma(x), y);
            }

            if (double.IsInfinity(x))
            {
                return new Complex(double.NaN, double.NaN);
            }

            if (y < 0)
            {
                return Complex.Conjugate(ComplexDigammaUpper(x, -y));
            }

            return ComplexDigammaUpper(x, y);
        }

        // exp(w) を絶対値 e^(Re w) と偏角 Im w に分離して計算する (オーバーフロー成分は符号付き ∞、アンダーフローは 0)
        private static Complex ComplexGammaExp(Complex w)
        {
            double modulus = Math.Exp(w.Real);
            (double sin, double cos) = Math.SinCos(w.Imaginary);
            if (!(modulus >= MinNormal && modulus <= double.MaxValue))
            {
                // e^(Re w) がオーバーフローまたは非正規化数になっても各成分が表現可能な場合があるため、指数を半分ずつ掛けて最終積で範囲判定する
                double half = Math.Exp(0.5 * w.Real);
                if (double.IsFinite(half))
                {
                    return new Complex(half * cos * half, half * sin * half);
                }
            }

            if (double.IsPositiveInfinity(modulus))
            {
                return new Complex(cos == 0 ? 0 : cos * modulus, sin == 0 ? 0 : sin * modulus);
            }

            return new Complex(modulus * cos, modulus * sin);
        }

        // ln Γ(x + iy) (y ≥ 0, 有限)
        private static Complex ComplexLogGammaUpper(double x, double y)
        {
            if (y == 0)
            {
                if (x > 0)
                {
                    return new Complex(LogGamma(x), y);
                }

                if (IsInteger(x))
                {
                    return new Complex(double.PositiveInfinity, y);
                }
            }

            var z = new Complex(x, y);
            if (Complex.Abs(z) < 0.3)
            {
                // ln Γ(z) = ln Γ(1 + z) - ln z。z + 1 の丸めで小 z で桁落ちしないよう直接級数へ渡す
                return ComplexLogGamma1pSeries(z) - Complex.Log(z);
            }

            if (x <= 0)
            {
                return ComplexLogGammaReflection(x, y);
            }

            if (x < 0.5)
            {
                // 0 < x < 1/2 では反転公式だと実軸近傍で虚部が O(1) 項の桁落ちになるため、漸化式 ln Γ(z) = ln Γ(z + 1) - ln z を用いる
                return ComplexLogGammaUpper(x + 1, y) - Complex.Log(z);
            }

            double distance1 = Complex.Abs(z - 1);
            if (distance1 < 0.3)
            {
                return ComplexLogGamma1pSeries(z - 1);
            }

            if (Complex.Abs(z - 2) < 0.8)
            {
                return ComplexLogGamma2pSeries(z - 2);
            }

            if (x < 1.2 && distance1 < 0.8)
            {
                // ln Γ(z) = ln Γ(z + 1) - ln z (z + 1 は |w| < 0.8 に入り、Re z ≥ 1/2 より主値の ln z で整合する)
                return ComplexLogGamma2pSeries(z - 1) - Complex.Log(z);
            }

            double modulus = Complex.Abs(z);
            if (modulus >= ComplexGammaAsymptoticRadius)
            {
                return ComplexLogGammaStirling(z);
            }

            // ln Γ(z) = ln Γ(z + n) - Σ_(k<n) ln(z + k)。実部は積の絶対値の対数、虚部は偏角の和で求める (n ≤ 6 でオーバーフローしない)
            Complex product = Complex.One;
            double argumentSum = 0;
            Complex shifted = z;
            while (Complex.Abs(shifted) < ComplexGammaAsymptoticRadius)
            {
                product *= shifted;
                argumentSum += Math.Atan2(y, shifted.Real);
                shifted += 1;
            }

            Complex stirling = ComplexLogGammaStirling(shifted);
            return new Complex(stirling.Real - Math.Log(Complex.Abs(product)), stirling.Imaginary - argumentSum);
        }

        // 反転公式 ln Γ(z) = ln(2π) - iπ/2 + iπz - ln(1 - e^(2πiz)) - ln Γ(1 - z) (x < 1/2, y ≥ 0)
        private static Complex ComplexLogGammaReflection(double x, double y)
        {
            // 1 - q (q = e^(2πiz) = g e^(2πix), g = e^(-2πy)) の実部・虚部。1 - g は expm1、1 - cos(2πx) は 2 sin²(πx) で極近傍の桁落ちを防ぐ
            double g = Math.Exp(-2 * Math.PI * y);
            double oneMinusG = -Expm1(-2 * Math.PI * y);
            double sinPiX = double.SinPi(x);
            double re1 = oneMinusG + 2 * g * sinPiX * sinPiX;
            double im1 = -g * double.SinPi(2 * x);

            // ln|1 - q|。y が大きい (g < 0.5) ときは log1p(g(g - 2 cos 2πx))/2 で桁落ちを避ける
            double logModulus;
            if (g < 0.5)
            {
                logModulus = 0.5 * Log1p(g * (g - 2 * double.CosPi(2 * x)));
            }
            else
            {
                double squares = oneMinusG * oneMinusG + 4 * g * sinPiX * sinPiX;

                // 極近傍の極小 y では 2 乗和が非正規化数・0 になるため hypot を使う
                logModulus = squares >= MinNormal
                    ? 0.5 * Math.Log(squares)
                    : Math.Log(double.Hypot(oneMinusG, 2 * Math.Sqrt(g) * sinPiX));
            }

            double logArgument = Math.Atan2(im1, re1);

            // 1 - z = (1 - x) - iy は下半平面なので共役で上半平面の値から得る
            Complex reflected = Complex.Conjugate(ComplexLogGammaUpper(1 - x, y));

            // iπz = -πy + iπx
            double re = LnTau - Math.PI * y - logModulus - reflected.Real;
            double im = Math.PI * x - 0.5 * Math.PI - logArgument - reflected.Imaginary;
            return new Complex(re, im);
        }

        // ln Γ(1 + w) = -γw + Σ (-1)ᵏ ζ(k) wᵏ/k (|w| < 0.3)
        private static Complex ComplexLogGamma1pSeries(Complex w)
        {
            Complex sum = Complex.Zero;
            for (int k = LogGamma1pCoefficients.Length - 1; k >= 2; k--)
            {
                sum = sum * w + LogGamma1pCoefficients[k];
            }

            return sum * w * w - EulerGamma * w;
        }

        // ln Γ(2 + w) = (1 - γ)w + Σ (-1)ᵏ (ζ(k) - 1) wᵏ/k (|w| < 0.8; 項比 ≤ |w|/2)
        private static Complex ComplexLogGamma2pSeries(Complex w)
        {
            Complex sum = Complex.Zero;
            for (int k = ComplexGammaShiftedZetaCoefficients.Length - 1; k >= 0; k--)
            {
                sum = sum * w + ComplexGammaShiftedZetaCoefficients[k];
            }

            return sum * w * w + ComplexGammaOneMinusEuler * w;
        }

        // スターリング級数 ln Γ(z) = (z - 1/2) ln z - z + ln√(2π) + Σ B₂ₖ/(2k(2k-1) z^(2k-1)) (Re z ≥ 1/2, |z| ≥ 6)
        private static Complex ComplexLogGammaStirling(Complex z)
        {
            double modulus = Complex.Abs(z);
            double logModulus = Math.Log(modulus);
            double argument = Math.Atan2(z.Imaginary, z.Real);

            // (z - 1/2) ln z - z の実部・虚部
            double xm = z.Real - 0.5;
            double re = xm * logModulus - z.Imaginary * argument - z.Real + LnSqrtTwoPi;
            double im = z.Imaginary * logModulus + xm * argument - z.Imaginary;

            Complex inverse = 1 / z;
            Complex inverseSquare = inverse * inverse;
            Complex power = inverse;
            double scale = Math.Max(1, Math.Abs(re) + Math.Abs(im));
            double inverseModulusSquare = 1 / (modulus * modulus);
            double powerModulus = 1 / modulus;
            Complex sum = Complex.Zero;
            double previousModulus = double.PositiveInfinity;
            for (int k = 1; k < BernoulliEven.Length; k++)
            {
                // 漸近級数なので項が増え始めたら (|z| = 6 で k ≈ 19) 打ち切る
                double coefficient = BernoulliEven[k] / (2 * k * (2 * k - 1));
                double termModulus = Math.Abs(coefficient) * powerModulus;
                if (termModulus > previousModulus)
                {
                    break;
                }

                sum += coefficient * power;
                if (termModulus <= 1e-17 * scale)
                {
                    break;
                }

                previousModulus = termModulus;

                power *= inverseSquare;
                powerModulus *= inverseModulusSquare;
            }

            return new Complex(re + sum.Real, im + sum.Imaginary);
        }

        // ψ(x + iy) (y > 0, 有限)
        private static Complex ComplexDigammaUpper(double x, double y)
        {
            var z = new Complex(x, y);
            if (x < 0.5)
            {
                // ψ(z) = ψ(1 - z) - π cot(πz)。1 - z は下半平面なので共役で求める
                Complex reflected = Complex.Conjugate(ComplexDigammaUpper(1 - x, y));
                return reflected - Math.PI * ComplexDigammaCotPi(x, y);
            }

            Complex d = new Complex((x - DigammaRootHi) - DigammaRootLo, y);
            if (Complex.Abs(d) < 0.2)
            {
                Complex taylor = Complex.Zero;
                for (int n = DigammaRootTaylor.Length - 1; n >= 0; n--)
                {
                    taylor = taylor * d + DigammaRootTaylor[n];
                }

                return taylor * d;
            }

            Complex result = Complex.Zero;
            while (Complex.Abs(z) < ComplexGammaAsymptoticRadius)
            {
                result -= 1 / z;
                z += 1;
            }

            // ψ(z) ~ ln z - 1/(2z) - Σ B₂ₖ/(2k z^(2k))
            Complex inverse = 1 / z;
            Complex inverseSquare = inverse * inverse;
            Complex power = inverseSquare;
            double modulus = Complex.Abs(z);
            double inverseModulusSquare = 1 / (modulus * modulus);
            double powerModulus = inverseModulusSquare;
            Complex tail = Complex.Zero;
            double previousModulus = double.PositiveInfinity;
            for (int k = 1; k < BernoulliEven.Length; k++)
            {
                // 漸近級数なので項が増え始めたら打ち切る (|z| = 6 で最小項は 10⁻¹⁷ 程度)
                double coefficient = BernoulliEven[k] / (2 * k);
                double termModulus = Math.Abs(coefficient) * powerModulus;
                if (termModulus > previousModulus)
                {
                    break;
                }

                tail += coefficient * power;
                if (termModulus <= 1e-17)
                {
                    break;
                }

                previousModulus = termModulus;

                power *= inverseSquare;
                powerModulus *= inverseModulusSquare;
            }

            return result + (Complex.Log(z) - 0.5 * inverse - tail);
        }

        // cot(πz) (y > 0)
        private static Complex ComplexDigammaCotPi(double x, double y)
        {
            if (y < 1)
            {
                // cot(πz) = (sin πx cos πx - i sinh πy cosh πy)/(sin²πx + sinh²πy)
                double s = double.SinPi(x);
                double c = double.CosPi(x);
                double sh = Math.Sinh(Math.PI * y);
                double ch = Math.Cosh(Math.PI * y);
                double denominator = s * s + sh * sh;
                if (denominator < MinNormal)
                {
                    // 極近傍 (x が整数で y が極小) のアンダーフロー・0/0 回避のため max(|sin πx|, sinh πy) でスケーリングする
                    double scale = Math.Max(Math.Abs(s), sh);
                    double ss = s / scale;
                    double shs = sh / scale;
                    double scaledDenominator = (ss * ss + shs * shs) * scale;
                    return new Complex(ss * c / scaledDenominator, -shs * ch / scaledDenominator);
                }

                return new Complex(s * c / denominator, -sh * ch / denominator);
            }

            // cot(πz) = -i (1 + q)/(1 - q) = -i (1 + 2q/(1 - q)), q = e^(2πiz) (|q| ≤ e^(-2π))
            double g = Math.Exp(-2 * Math.PI * y);
            var q = new Complex(g * double.CosPi(2 * x), g * double.SinPi(2 * x));
            Complex ratio = 2 * q / (1 - q);
            return new Complex(ratio.Imaginary, -(1 + ratio.Real));
        }
    }
}
