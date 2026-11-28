using System.Numerics;
using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        /// <summary>
        /// Computes the Fresnel sine integral <c>S(x) = ∫₀ˣ sin(πt² / 2) dt</c>.
        /// <para>フレネル正弦積分 <c>S(x) = ∫₀ˣ sin(πt² / 2) dt</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>S(x)</c>. / <c>S(x)</c> の値。</returns>
        /// <seealso href="https://dlmf.nist.gov/7.2.E8"/>
        public static double FresnelS(double x) => Fresnel(x).S;

        /// <summary>
        /// Computes the Fresnel cosine integral <c>C(x) = ∫₀ˣ cos(πt² / 2) dt</c>.
        /// <para>フレネル余弦積分 <c>C(x) = ∫₀ˣ cos(πt² / 2) dt</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>C(x)</c>. / <c>C(x)</c> の値。</returns>
        /// <seealso href="https://dlmf.nist.gov/7.2.E7"/>
        public static double FresnelC(double x) => Fresnel(x).C;

        /// <summary>
        /// Simultaneously computes the Fresnel integrals <c>S(x)</c> and <c>C(x)</c>.
        /// <para>フレネル積分 <c>S(x)</c> と <c>C(x)</c> を同時に計算します。</para>
        /// </summary>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>A tuple containing <c>(S(x), C(x))</c>. / <c>(S(x), C(x))</c> のタプル。</returns>
        /// <remarks>
        /// <para>Uses the normalized convention with argument <c>πt² / 2</c>. Calling this method avoids redundant work compared with calling <see cref="FresnelS"/> and <see cref="FresnelC"/> separately.</para>
        /// <para>引数を <c>πt² / 2</c> とする正規化定義を採用しています。両方の値が必要な場合は、<see cref="FresnelS"/> と <see cref="FresnelC"/> を個別に呼び出すよりこのメソッドを使う方が重複計算を避けられます。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/7.2.iv"/>
        public static (double S, double C) Fresnel(double x)
        {
            if (double.IsNaN(x))
            {
                return (x, x);
            }

            if (double.IsInfinity(x))
            {
                return x > 0 ? (0.5, 0.5) : (-0.5, -0.5);
            }

            double ax = Math.Abs(x);
            double s;
            double c;
            if (ax < 1.5)
            {
                // t = πx²/2、Tₖ = x tᵏ/k! とすると
                // C(x) = Σ_{k 偶数} (-1)^(k/2) Tₖ/(2k+1)、S(x) = Σ_{k 奇数} (-1)^((k-1)/2) Tₖ/(2k+1)
                double t = 0.5 * Math.PI * ax * ax;
                double term = ax;
                s = 0;
                c = 0;
                for (int k = 0; k < 200; k++)
                {
                    double contribution = term / (2 * k + 1);
                    if (k % 2 == 0)
                    {
                        c += (k / 2) % 2 == 0 ? contribution : -contribution;
                    }
                    else
                    {
                        s += ((k - 1) / 2) % 2 == 0 ? contribution : -contribution;
                    }

                    if (k >= 2 && contribution <= MachineEpsilon * Math.Abs(s) * 0.5)
                    {
                        break;
                    }

                    term *= t / (k + 1);
                }
            }
            else if (ax >= 9007199254740992.0)
            {
                // x ≥ 2⁵³ では x²/2 が偶数となり、sin(πx²/2) = 0、cos(πx²/2) = 1
                c = 0.5;
                s = 0.5 - 1 / (Math.PI * ax);
            }
            else
            {
                // C + iS = ((1+i)/2) erf(z)、z = (√π/2)(1-i)x。erfc(z) の連分数（Lentz法）
                double pix2 = Math.PI * ax * ax;
                var b = new Complex(1, -pix2);
                var cc = new Complex(1 / Tiny, 0);
                Complex d = 1 / b;
                Complex h = d;
                double n = -1;
                for (int k = 2; k < MaxIterations; k++)
                {
                    n += 2;
                    double an = -n * (n + 1);
                    b += 4;
                    d = 1 / (an * d + b);
                    cc = b + an / cc;
                    Complex delta = cc * d;
                    h *= delta;
                    if (Math.Abs(delta.Real - 1) + Math.Abs(delta.Imaginary) <= MachineEpsilon)
                    {
                        break;
                    }
                }

                h *= new Complex(ax, -ax);

                // 位相 πx²/2 を x² = hi + lo（誤差なし）に分けて正確に計算する
                double hi = ax * ax;
                double lo = Math.FusedMultiplyAdd(ax, ax, -hi);
                double cosHi = double.CosPi(hi / 2);
                double sinHi = double.SinPi(hi / 2);
                double cosLo = double.CosPi(lo / 2);
                double sinLo = double.SinPi(lo / 2);
                var phase = new Complex(cosHi * cosLo - sinHi * sinLo, sinHi * cosLo + cosHi * sinLo);
                Complex cs = new Complex(0.5, 0.5) * (1 - phase * h);
                c = cs.Real;
                s = cs.Imaginary;
            }

            return x < 0 ? (-s, -c) : (s, c);
        }
    }
}
