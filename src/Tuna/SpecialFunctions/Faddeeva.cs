using System.Numerics;
using static Tuna.XMath.Constants;
using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // Zaghloul–Ali 型の台形則の刻み a = 13/64（a ≈ 0.518 では虚軸付近で虚部がずれるため小さくし、an と (an)² を誤差なく表現する）
        private const double FaddeevaStep = 0.203125;

        // c = 2a/π = 13/(32π)
        private const double FaddeevaStepOverHalfPi = 0.12931339126216496;

        // これ以上の y では台形則の極の補正項を無視できる（虚軸付近での虚部の桁落ちを避けるため早めに切り替える）
        private const double FaddeevaNoPoleY = 1.4;

        // これ以上の x または y では連分数を使う（実軸の近くでは実部に e^(y²-x²) cos 2xy を加える）
        private const double FaddeevaContinuedFractionX = 7;
        private const double FaddeevaContinuedFractionY = 7;

        // 漸近形で分母の成分がこれ以上（2¹⁰²⁰）なら複素除算を避ける
        private const double FaddeevaLargeDenominator = 1.1235582092889474e307;

        // これ未満（2⁻²⁷）の丸め誤差 δ では e^δ = 1 + δ として扱う（δ²/2 は丸め単位未満）
        private const double FaddeevaFirstOrderLimit = 7.450580596923828e-9;

        // Voigt プロファイルで σ√2 や π·γ のオーバーフローと非正規化数での多重丸めを避けるため 2 の冪で縮める閾値（2¹⁰¹⁸）
        private const double VoigtLargeWidth = 2.8088955232223686e306;

        /// <summary>
        /// Computes the Faddeeva function <c>w(z) = e^(-z²) erfc(-iz)</c>.
        /// <para>Faddeeva 関数 <c>w(z) = e^(-z²) erfc(-iz)</c> を計算します。</para>
        /// </summary>
        /// <param name="z">The complex input value. / 複素数の入力値。</param>
        /// <returns>The value of <c>w(z)</c>, or a <see cref="Complex"/> with <see cref="double.NaN"/> components if either component of <paramref name="z"/> is <see cref="double.NaN"/>. / <c>w(z)</c> の値。<paramref name="z"/> の実部または虚部が <see cref="double.NaN"/> の場合は成分が <see cref="double.NaN"/> の <see cref="Complex"/>。</returns>
        /// <remarks>
        /// <para>On the real axis, <c>w(x) = e^(-x²) + (2i/√π) F(x)</c> where <c>F</c> is Dawson's integral; on the imaginary axis, <c>w(iy) = erfcx(y)</c>. In the lower half-plane where <c>e^(-z²)</c> overflows, components become infinite.</para>
        /// <para>実軸上では <c>w(x) = e^(-x²) + (2i/√π) F(x)</c>（<c>F</c> はドーソン関数）、虚軸上では <c>w(iy) = erfcx(y)</c> となります。下半平面で <c>e^(-z²)</c> がオーバーフローする領域では無限大の成分を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/7.2#E3"/>
        public static Complex Faddeeva(Complex z)
        {
            double x = z.Real;
            double y = z.Imaginary;
            if (double.IsNaN(x) || double.IsNaN(y))
            {
                return new Complex(double.NaN, double.NaN);
            }

            if (y < 0)
            {
                if (double.IsInfinity(x) || double.IsInfinity(y))
                {
                    return new Complex(double.NaN, double.NaN);
                }

                // w(z) = 2e^(-z²) - w(-z)。e^(-z²) = e^(y²-x²) e^(-2ixy) の指数と位相は数百になりうるので、
                // y² - x² と 2xy の丸め誤差を FMA で求めて 1 次の補正をかける（そのままだと相対 ε·|指数| の誤差になる）
                Complex reflected = FaddeevaUpperHalfPlane(-x, -y);
                double x2 = x * x;
                double x2Error = Math.FusedMultiplyAdd(x, x, -x2);
                double y2 = y * y;
                double y2Error = Math.FusedMultiplyAdd(y, y, -y2);
                double exponent = y2 - x2;
                double exponentError = ((y2 - exponent) - x2) + (y2Error - x2Error);
                double phase = -2 * x * y;
                double phaseError = -Math.FusedMultiplyAdd(2 * x, y, phase);
                // 丸め誤差は |z| が大きいと 1 を超える（x = 1e9 で最大 64）ため、回転・指数として掛ける。
                // 誤差が 2⁻²⁷ 未満なら cos = 1、sin = 誤差、e^誤差 ≈ 1 + 誤差となり、1 次の補正と一致する
                (double sinPhase, double cosPhase) = Math.SinCos(phase);
                (double sinError, double cosError) = Math.SinCos(phaseError);
                double cosCorrected = cosPhase * cosError - sinPhase * sinError;
                double sinCorrected = sinPhase * cosError + cosPhase * sinError;
                double exponentFactor = Math.Abs(exponentError) < FaddeevaFirstOrderLimit ? 1 + exponentError : Math.Exp(exponentError);

                // 2e^E を e^(E/2) の 2 乗に分けて、結果が表現可能であるにもかかわらず途中でオーバーフローするのを避ける
                double half = Math.Exp(0.5 * exponent);
                double scale = 2 * half * exponentFactor;
                double re = scale * cosCorrected * half;
                double im = scale * sinCorrected * half;
                if (double.IsInfinity(re) || double.IsInfinity(im))
                {
                    return new Complex(re, im);
                }

                return new Complex(re - reflected.Real, im - reflected.Imaginary);
            }

            return FaddeevaUpperHalfPlane(x, y);
        }

        /// <summary>
        /// Computes the Voigt profile <c>V(x; σ, γ)</c>.
        /// <para>Voigt プロファイル <c>V(x; σ, γ)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The displacement from the center. / 中心からの距離。</param>
        /// <param name="sigma">The standard deviation <c>σ ≥ 0</c> of the Gaussian component. / ガウス成分の標準偏差 <c>σ ≥ 0</c>。</param>
        /// <param name="gamma">The half-width at half-maximum <c>γ ≥ 0</c> of the Lorentzian component. / ローレンツ成分の半値半幅 <c>γ ≥ 0</c>。</param>
        /// <returns>The value of <c>V(x; σ, γ)</c>, or <see cref="double.NaN"/> if <paramref name="sigma"/> or <paramref name="gamma"/> is negative or non-finite, if both are zero, or if any argument is <see cref="double.NaN"/>. / <c>V(x; σ, γ)</c> の値。<paramref name="sigma"/> または <paramref name="gamma"/> が負または有限でない場合、両方が 0 の場合、またはいずれかの引数が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Reduces to the Lorentzian density <c>γ / (π (x² + γ²))</c> when <c>σ = 0</c> and to the normal density when <c>γ = 0</c>. Returns <c>0</c> when <paramref name="x"/> is infinite.</para>
        /// <para><c>σ = 0</c> ではローレンツ分布の密度 <c>γ / (π (x² + γ²))</c>、<c>γ = 0</c> では正規分布の密度に帰着します。<paramref name="x"/> が無限大の場合は <c>0</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/7.19"/>
        public static double VoigtProfile(double x, double sigma, double gamma)
        {
            if (double.IsNaN(x) || double.IsNaN(sigma) || double.IsNaN(gamma) || sigma < 0 || gamma < 0
                || double.IsInfinity(sigma) || double.IsInfinity(gamma) || (sigma == 0 && gamma == 0))
            {
                return double.NaN;
            }

            if (double.IsInfinity(x))
            {
                return 0;
            }

            if (sigma == 0)
            {
                return VoigtLorentzian(x, gamma);
            }

            if (gamma == 0)
            {
                // 正規分布の密度 e^(-u)/(σ√(2π))、u = (x/σ)²/2。x/σ と その2乗を double-double で持ち、
                // 裾（u が数百）でも u の丸め誤差（相対 ε·u）が値の誤差にならないようにする
                double q = x / sigma;
                if (double.IsInfinity(q))
                {
                    return 0;
                }

                double qLo = Math.FusedMultiplyAdd(-q, sigma, x) / sigma;
                double q2 = q * q;
                double q2Lo = Math.FusedMultiplyAdd(q, q, -q2) + 2 * q * qLo;
                double gaussian = Math.Exp(-0.5 * q2);
                if (gaussian >= MinNormal)
                {
                    return gaussian * (1 - 0.5 * q2Lo) * (OneOverSqrtPi * OneOverSqrt2) / sigma;
                }

                // 標準密度 e^(-u) が非正規化数や 0 になっても、小さい σ で割った密度は表現できる場合がある。
                // σ = m·2ᵏ（1 ≤ m < 2）とし、e^(-u)·2⁻ᵏ = e^(-u - k ln 2) の指数を double-double で求めて 1 回の指数にまとめる
                int k = Math.ILogB(sigma);
                double mantissa = Math.ScaleB(sigma, -k);
                double shiftHi = k * Ln2;
                double shiftLo = Math.FusedMultiplyAdd(k, Ln2, -shiftHi) + k * Ln2Lo;
                double u = 0.5 * q2;
                double exponent = -u - shiftHi;
                double exponentLo = ((-u - exponent) - shiftHi) - shiftLo;
                return Math.Exp(exponent) * (1 + exponentLo) * (1 - 0.5 * q2Lo) * (OneOverSqrtPi * OneOverSqrt2) / mantissa;
            }

            if (sigma > VoigtLargeWidth)
            {
                // σ√2 がオーバーフローしうる。z = (x + iγ)/(σ√2) は比として作り、密度は最後に σ で割る（非正規化数への丸めは 1 回）。
                // このとき |z| ≤ √2·2^(1024-1018) なので漸近形の分岐には入らない
                Complex largeW = FaddeevaUpperHalfPlane(x / sigma * OneOverSqrt2, gamma / sigma * OneOverSqrt2);
                return largeW.Real * (OneOverSqrtPi * OneOverSqrt2) / sigma;
            }

            double scale = sigma * Sqrt2;
            double modulus = double.Hypot(x, gamma);
            if (modulus > 1e7 * scale)
            {
                // |z| > 10⁷：w(z) = (i/√π)/z (1 + 1/(2z²) + O(z⁻⁴)) を ζ = x + iγ で書き直すと
                // V = γ/(π|ζ|²) · (1 + σ²(3x² - γ²)/|ζ|⁴ + O(σ⁴/|ζ|⁴))。σ が極端に小さく z がオーバーフローする場合もここで扱う
                double u = x / modulus;
                double v = gamma / modulus;
                double t = sigma / modulus;
                return VoigtLorentzian(x, gamma) * (1 + t * t * (3 * u * u - v * v));
            }

            double re = x / scale;
            double im = gamma / scale;
            Complex w = FaddeevaUpperHalfPlane(re, im);
            return w.Real * OneOverSqrtPi / scale;
        }

        // ローレンツ分布の密度 γ/(π(x² + γ²))（γ > 0）をオーバーフローせずに計算する
        private static double VoigtLorentzian(double x, double gamma)
        {
            double ax = Math.Abs(x);
            double big = Math.Max(ax, gamma);
            double small = Math.Min(ax, gamma);
            double ratio = small / big;
            if (big > VoigtLargeWidth)
            {
                // π·big がオーバーフローし値が非正規化数になるため、big を 2 の冪で縮めて計算し最後に一度だけ戻す
                int k = Math.ILogB(big);
                double scaled = 1 / (Math.PI * Math.ScaleB(big, -k) * (1 + ratio * ratio));
                return Math.ScaleB(ax > gamma ? scaled * (gamma / ax) : scaled, -k);
            }

            double value = 1 / (Math.PI * big * (1 + ratio * ratio));
            return ax > gamma ? value * (gamma / ax) : value;
        }

        // 上半平面（y ≥ 0）での w(x + iy)
        private static Complex FaddeevaUpperHalfPlane(double x, double y)
        {
            double ax = Math.Abs(x);
            if (ax < 1e-100 && ax > 0 && y > 0)
            {
                // 虚部は x に比例し、実部は x² の補正しか受けないので、x を 10⁻¹⁰⁰ に置き換えて虚部だけ比例で戻す
                // （小さすぎる x のままだと途中の積が非正規化数になって精度が落ちる）
                Complex scaled = FaddeevaUpperHalfPlane(1e-100, y);
                return new Complex(scaled.Real, scaled.Imaginary * (x * 1e100));
            }

            Complex value;
            if (y == 0)
            {
                // 実軸上：w(x) = e^(-x²) + (2i/√π) F(x)
                value = new Complex(ExpMinusSquare(ax), TwoOverSqrtPi * Dawson(ax));
            }
            else if (ax >= FaddeevaContinuedFractionX || y >= FaddeevaContinuedFractionY)
            {
                value = FaddeevaContinuedFraction(ax, y);
                if (y < 0.5 && ax < 27)
                {
                    // 実軸の近くでは連分数が e^(-z²) の寄与を含まないので、実部にだけ加える（虚部への寄与は相対的に無視できる）
                    value = new Complex(value.Real + ExpMinusSquare(ax) * Math.Exp(y * y) * Math.Cos(2 * ax * y), value.Imaginary);
                }
            }
            else
            {
                value = FaddeevaTrapezoid(ax, y);
            }

            return x < 0 ? Complex.Conjugate(value) : value;
        }

        // Laplace の連分数による w(x + iy)（x ≥ 0, y ≥ 0, |z| が大きい領域）。
        // Gautschi (1970) の h = 0 の後退漸化式を実数演算で行い、実部が y に比例して小さいときも相対精度を保つ
        private static Complex FaddeevaContinuedFraction(double x, double y)
        {
            double modulus = Math.Max(x, y);
            if (double.IsInfinity(modulus))
            {
                // w(z) ~ i/(√π z) → 0
                return Complex.Zero;
            }

            if (modulus > 1e7)
            {
                // w(z) = (i/√π)/(z - 1/(2z) - …)。省いた項は相対 |z|⁻⁴ ≤ 10⁻²⁸
                var zc = new Complex(x, y);
                Complex denominator = zc - 0.5 / zc;
                double a = denominator.Real;
                double b = denominator.Imaginary;
                double largest = Math.Max(Math.Abs(a), Math.Abs(b));
                if (largest < FaddeevaLargeDenominator)
                {
                    return new Complex(0, OneOverSqrtPi) / denominator;
                }

                // 最大有限値付近の z では複素除算（Smith の方法）の分母 c + d(d/c) がオーバーフローするため、
                // (i/√π)/(a + ib) = (b + ia)/(√π(a² + b²)) を 2ᵏ で縮めて計算し最後に一度だけ戻す
                int k = Math.ILogB(largest);
                double an = Math.ScaleB(a, -k);
                double bn = Math.ScaleB(b, -k);
                double norm = an * an + bn * bn;
                return new Complex(Math.ScaleB(bn / norm * OneOverSqrtPi, -k), Math.ScaleB(an / norm * OneOverSqrtPi, -k));
            }

            double xq = x / 6.3;
            double yq = y / 4.4;
            double rho = Math.Sqrt(xq * xq + yq * yq);
            int nu = (int)(1.3 * (3 + 1442 / (26 * rho + 77)));
            double rx = 0;
            double ry = 0;
            for (int n = nu; n >= 0; n--)
            {
                // 1/(z - (n+1)/2 · …) を (i/2)/((y + (n+1)rx) + i(x - (n+1)ry))… の形で実数のまま後退計算する
                double np1 = n + 1;
                double tx = y + np1 * rx;
                double ty = x - np1 * ry;
                double c = 0.5 / (tx * tx + ty * ty);
                rx = c * tx;
                ry = c * ty;
            }

            return new Complex(TwoOverSqrtPi * rx, TwoOverSqrtPi * ry);
        }

        // Zaghloul–Ali 型の台形則による w(x + iy)（0 ≤ x < 7, 0 < y < 7）。
        // c = 2a/π、S₁ = Σ e^(-a²n²)/(a²n² + y²)、S₂₃ = Σ (e^(-(an-x)²) + e^(-(an+x)²))/(a²n² + y²)、
        // S₅ = Σ an (e^(-(an-x)²) - e^(-(an+x)²))/(a²n² + y²)（n ≥ 1）として実部・虚部を評価する。
        // y ≥ 1.4 では極の補正項が無視でき、正の項のみの和 Re w = c e^(-x²)/(2y) + (c/2) y S₂₃、Im w = (c/2) S₅ を使う
        private static Complex FaddeevaTrapezoid(double x, double y)
        {
            const double a = FaddeevaStep;
            const double c = FaddeevaStepOverHalfPi;
            double expX2 = ExpMinusSquare(x);
            double y2 = y * y;

            // e^(-(an ∓ x)²) は an ≥ x + 6.3 で 10⁻¹⁷ 未満になる
            int terms = (int)((x + 6.3) / a) + 1;
            double sum1 = 0;
            double sum23 = 0;
            double sum5 = 0;
            for (int n = 1; n <= terms; n++)
            {
                double an = a * n;
                double an2 = an * an;
                double denominator = an2 + y2;

                // e^(-t²) は |t| ≥ 6.3 で 10⁻¹⁷ 未満になり、和に寄与しないので計算を省く
                double gauss = an < 6.3 ? Math.Exp(-an2) : 0;
                sum1 += gauss / denominator;
                if (x < 1)
                {
                    // e^(-(an-x)²) - e^(-(an+x)²) = 2e^(-a²n²-x²) sinh(2anx) は x が小さいと桁落ちするので sinh で求める
                    double common = 2 * gauss * expX2 / denominator;
                    double t = 2 * an * x;
                    sum23 += common * Math.Cosh(t);
                    sum5 += common * an * Math.Sinh(t);
                }
                else
                {
                    // x ≥ 1 では e^(-(an+x)²)/e^(-(an-x)²) = e^(-4anx) ≤ e^(-1) なので差の桁落ちは小さい
                    double d = an - x;
                    double minus = Math.Abs(d) < 6.3 ? ExpMinusSquare(d) / denominator : 0;
                    double plus = an + x < 6.3 ? ExpMinusSquare(an + x) / denominator : 0;
                    sum23 += minus + plus;
                    sum5 += an * (minus - plus);
                }
            }

            if (y >= FaddeevaNoPoleY)
            {
                return new Complex(c * expX2 / (2 * y) + 0.5 * c * y * sum23, 0.5 * c * sum5);
            }

            double xy = x * y;
            double sinXy = Math.Sin(xy);
            (double sin2Xy, double cos2Xy) = Math.SinCos(2 * xy);
            double sincXy = xy == 0 ? 1 : sinXy / xy;
            double sinc2Xy = xy == 0 ? 1 : sin2Xy / (2 * xy);
            double coefficient1 = expX2 * (Erfcx(y) - c * y * sum1);
            double coefficient2 = c * x * expX2;
            double re = coefficient1 * cos2Xy + coefficient2 * sinXy * sincXy + 0.5 * c * y * sum23;
            double im = -coefficient1 * sin2Xy + coefficient2 * sinc2Xy + 0.5 * c * sum5;
            return new Complex(re, im);
        }
    }
}
