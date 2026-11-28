using static Tuna.XMath.Constants;
using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        /// <summary>
        /// Computes the complete elliptic integral <c>D(m) = (K(m) - E(m)) / m = ∫₀^(π/2) sin²θ / √(1 - m sin²θ) dθ</c>.
        /// <para>完全楕円積分 <c>D(m) = (K(m) - E(m)) / m = ∫₀^(π/2) sin²θ / √(1 - m sin²θ) dθ</c> を計算します。</para>
        /// </summary>
        /// <param name="m">The parameter <c>m = k²</c> (<c>m ≤ 1</c>). / パラメータ <c>m = k²</c>（<c>m ≤ 1</c>）。</param>
        /// <returns>The value of <c>D(m)</c>, <see cref="double.PositiveInfinity"/> at <c>m = 1</c>, or <see cref="double.NaN"/> if <paramref name="m"/> is greater than 1 or <see cref="double.NaN"/>. / <c>D(m)</c> の値。<c>m = 1</c> では <see cref="double.PositiveInfinity"/>、<paramref name="m"/> が 1 より大きい場合または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/19.2.E6"/>
        public static double EllipticD(double m)
        {
            if (double.IsNaN(m) || m > 1)
            {
                return double.NaN;
            }

            if (m == 1)
            {
                return double.PositiveInfinity;
            }

            if (double.IsNegativeInfinity(m))
            {
                return 0;
            }

            return CarlsonRD(0, 1 - m, 1) / 3;
        }

        /// <summary>
        /// Computes the incomplete elliptic integral <c>D(φ|m) = (F(φ|m) - E(φ|m)) / m = ∫₀^φ sin²θ / √(1 - m sin²θ) dθ</c>.
        /// <para>不完全楕円積分 <c>D(φ|m) = (F(φ|m) - E(φ|m)) / m = ∫₀^φ sin²θ / √(1 - m sin²θ) dθ</c> を計算します。</para>
        /// </summary>
        /// <param name="phi">The amplitude <c>φ</c> in radians. / 振幅 <c>φ</c>（ラジアン）。</param>
        /// <param name="m">The parameter <c>m = k²</c> (<c>m sin²φ ≤ 1</c>). / パラメータ <c>m = k²</c>（<c>m sin²φ ≤ 1</c>）。</param>
        /// <returns>The real value of <c>D(φ|m)</c>, or <see cref="double.NaN"/> if the result is complex or either argument is invalid. / <c>D(φ|m)</c> の実数値。値が複素数になる場合や引数が無効な場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/19.2.E6"/>
        public static double EllipticD(double phi, double m)
        {
            if (!TryReduceAmplitude(phi, m, out double periods, out double s, out double c2, out double delta))
            {
                return double.NaN;
            }

            double value = s * s * s / 3 * CarlsonRD(c2, AmplitudeDelta(m, s, c2, delta), 1);
            return periods == 0 ? value : value + 2 * periods * EllipticD(m);
        }

        /// <summary>
        /// Computes Jacobi's zeta function <c>Z(φ|m) = E(φ|m) - (E(m) / K(m)) F(φ|m)</c>.
        /// <para>ヤコビのゼータ関数 <c>Z(φ|m) = E(φ|m) - (E(m) / K(m)) F(φ|m)</c> を計算します。</para>
        /// </summary>
        /// <param name="phi">The amplitude <c>φ</c> in radians. / 振幅 <c>φ</c>（ラジアン）。</param>
        /// <param name="m">The parameter <c>m = k²</c> (<c>m ≤ 1</c>). / パラメータ <c>m = k²</c>（<c>m ≤ 1</c>）。</param>
        /// <returns>The value of <c>Z(φ|m)</c> (an odd function with period <c>π</c>; at <c>m = 1</c>, the limit <c>sin φ · sgn(cos φ)</c>), or <see cref="double.NaN"/> if out of domain. / <c>Z(φ|m)</c> の値（周期 <c>π</c> の奇関数。<c>m = 1</c> では極限 <c>sin φ · sgn(cos φ)</c>）。定義域外では <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/22.16.E32"/>
        public static double JacobiZeta(double phi, double m)
        {
            if (double.IsNaN(phi) || double.IsNaN(m) || double.IsInfinity(phi) || m > 1)
            {
                return double.NaN;
            }

            if (m == 0 || double.IsNegativeInfinity(m))
            {
                return double.IsNegativeInfinity(m) ? double.NaN : 0;
            }

            // Z は周期 π の奇関数なので |r| ≤ π/2 に還元する (大きな |φ| では正確な π 還元を使用)
            double r;
            if (Math.Abs(phi) >= ExactPiReductionThreshold)
            {
                ReduceByPiExact(phi, out r, out _);
            }
            else
            {
                double periods = Math.Round(phi / Math.PI);
                r = periods == 0 ? phi : Math.FusedMultiplyAdd(-periods, Pi, phi) - periods * PiLo;
            }

            double s = Math.Sin(r);
            double c = Math.Cos(r);
            if (m == 1)
            {
                return s;
            }

            double delta = AmplitudeDelta(m, s, c * c, 1 - m * s * s);
            double k = CarlsonRF(0, 1 - m, 1);
            return m / 3 * s * c * Math.Sqrt(delta) * CarlsonRJ(0, 1 - m, 1, delta) / k;
        }

        /// <summary>
        /// Computes Heuman's lambda function <c>Λ₀(φ|m) = F(φ|1 - m) / K(1 - m) + (2 / π) K(m) Z(φ|1 - m)</c>.
        /// <para>ヒューマンのラムダ関数 <c>Λ₀(φ|m) = F(φ|1 - m) / K(1 - m) + (2 / π) K(m) Z(φ|1 - m)</c> を計算します。</para>
        /// </summary>
        /// <param name="phi">The amplitude <c>φ</c> in radians. / 振幅 <c>φ</c>（ラジアン）。</param>
        /// <param name="m">The parameter <c>m = k²</c> (<c>0 ≤ m ≤ 1</c>). / パラメータ <c>m = k²</c>（<c>0 ≤ m ≤ 1</c>）。</param>
        /// <returns>The value of <c>Λ₀(φ|m)</c> (an odd function satisfying <c>Λ₀(φ + π|m) = Λ₀(φ|m) + 2</c>), or <see cref="double.NaN"/> if out of domain. / <c>Λ₀(φ|m)</c> の値（<c>Λ₀(φ + π|m) = Λ₀(φ|m) + 2</c> を満たす奇関数）。定義域外では <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/19.2.E5"/>
        public static double HeumanLambda(double phi, double m)
        {
            if (double.IsNaN(phi) || double.IsNaN(m) || double.IsInfinity(phi) || m < 0 || m > 1)
            {
                return double.NaN;
            }

            if (m == 1)
            {
                return phi / (Math.PI / 2);
            }

            // Λ₀(φ + jπ) = Λ₀(φ) + 2j, |r| ≤ π/2
            double periods = Math.Round(phi / Math.PI);
            double r = periods == 0 ? phi : Math.FusedMultiplyAdd(-periods, Pi, phi) - periods * PiLo;
            double s = Math.Sin(r);
            double value;
            if (m == 0)
            {
                value = s;
            }
            else
            {
                double c = Math.Cos(r);
                double complement = 1 - m;
                double delta = 1 - complement * s * s;
                double kComplement = CarlsonRF(0, m, 1);
                double first = s * CarlsonRF(c * c, delta, 1);
                double second = 2 / (3 * Math.PI) * CarlsonRF(0, complement, 1) * complement * s * c * Math.Sqrt(delta)
                    * CarlsonRJ(0, m, 1, delta);
                value = (first + second) / kComplement;
            }

            return periods == 0 ? value : value + 2 * periods;
        }

        /// <summary>
        /// Computes the principal inverse Jacobi elliptic function <c>arcsn(x|m) = F(arcsin x|m)</c>.
        /// <para>逆ヤコビ楕円関数 <c>arcsn(x|m) = F(arcsin x|m)</c>（主値 <c>|u| ≤ K</c>）を計算します。</para>
        /// </summary>
        /// <param name="x">The input value (<c>|x| ≤ 1</c> and <c>m x² ≤ 1</c>). / 入力値（<c>|x| ≤ 1</c> かつ <c>m x² ≤ 1</c>）。</param>
        /// <param name="m">The parameter <c>m = k²</c>. / パラメータ <c>m = k²</c>。</param>
        /// <returns>The principal value <c>u</c> satisfying <c>sn(u|m) = x</c>, or <see cref="double.NaN"/> if out of domain. / <c>sn(u|m) = x</c> を満たす主値 <c>u</c>。定義域外では <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/22.15"/>
        public static double InverseJacobiSN(double x, double m)
        {
            if (double.IsNaN(x) || double.IsNaN(m) || Math.Abs(x) > 1)
            {
                return double.NaN;
            }

            double x2 = x * x;
            double delta = 1 - m * x2;
            if (delta < 0)
            {
                return double.NaN;
            }

            // 1 - x² = (1 - x)(1 + x) は |x| ≈ 1 で桁落ちしない
            double c2 = (1 - x) * (1 + x);
            if (c2 == 0 && delta == 0)
            {
                return Math.Sign(x) * double.PositiveInfinity;
            }

            return x * CarlsonRF(c2, delta, 1);
        }

        /// <summary>
        /// Computes the principal inverse Jacobi elliptic function <c>arccn(x|m) = F(arccos x|m)</c>.
        /// <para>逆ヤコビ楕円関数 <c>arccn(x|m) = F(arccos x|m)</c>（主値 <c>0 ≤ u ≤ 2K</c>）を計算します。</para>
        /// </summary>
        /// <param name="x">The input value (<c>|x| ≤ 1</c>). / 入力値（<c>|x| ≤ 1</c>）。</param>
        /// <param name="m">The parameter <c>m = k²</c> (<c>m ≤ 1</c>). / パラメータ <c>m = k²</c>（<c>m ≤ 1</c>）。</param>
        /// <returns>The principal value <c>u</c> satisfying <c>cn(u|m) = x</c>, or <see cref="double.NaN"/> if out of domain. / <c>cn(u|m) = x</c> を満たす主値 <c>u</c>。定義域外では <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/22.15"/>
        public static double InverseJacobiCN(double x, double m)
        {
            if (double.IsNaN(x) || double.IsNaN(m) || Math.Abs(x) > 1 || m > 1)
            {
                return double.NaN;
            }

            double a = Math.Abs(x);
            if (m == 1)
            {
                // cn(u|1) = sech u
                return x < 0 ? double.NaN : InverseSech(a);
            }

            double s = Math.Sqrt((1 - a) * (1 + a));
            double a2 = a * a;
            double value = s * CarlsonRF(a2, 1 - m + m * a2, 1);
            return x < 0 ? 2 * EllipticK(m) - value : value;
        }

        /// <summary>
        /// Computes the principal inverse Jacobi elliptic function <c>arcdn(x|m)</c>.
        /// <para>逆ヤコビ楕円関数 <c>arcdn(x|m)</c>（主値 <c>0 ≤ u ≤ K</c>）を計算します。</para>
        /// </summary>
        /// <param name="x">The input value (<c>√(1 - m) ≤ x ≤ 1</c>). / 入力値（<c>√(1 - m) ≤ x ≤ 1</c>）。</param>
        /// <param name="m">The parameter <c>m = k²</c> (<c>0 &lt; m ≤ 1</c>). / パラメータ <c>m = k²</c>（<c>0 &lt; m ≤ 1</c>）。</param>
        /// <returns>The principal value <c>u</c> satisfying <c>dn(u|m) = x</c> (equals <c>F(φ|m)</c> with <c>sin²φ = (1 - x²) / m</c>), or <see cref="double.NaN"/> if out of domain. / <c>dn(u|m) = x</c> を満たす主値 <c>u</c>（<c>sin²φ = (1 - x²) / m</c> のとき <c>F(φ|m)</c>）。定義域外では <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/22.15"/>
        public static double InverseJacobiDN(double x, double m)
        {
            if (double.IsNaN(x) || double.IsNaN(m) || x > 1 || x < 0 || m <= 0 || m > 1)
            {
                return double.NaN;
            }

            if (m == 1)
            {
                // dn(u|1) = sech u。小 x では R_F の引数 x² が非正規化数・0 になり精度低下するため直接計算する
                return InverseSech(x);
            }

            // x² - (1 - m)。m ≥ 1/2 では 1 - m が丸め誤差なしのため FMA を使い、それ以外は 1 - x² = (1 - x)(1 + x) で桁落ちを避ける
            double oneMinusX2 = (1 - x) * (1 + x);
            double c2 = (m >= 0.5 ? Math.FusedMultiplyAdd(x, x, -(1 - m)) : m - oneMinusX2) / m;
            if (c2 < 0)
            {
                return double.NaN;
            }

            if (c2 == 0 && x == 0)
            {
                return double.PositiveInfinity;
            }

            return Math.Sqrt(oneMinusX2 / m) * CarlsonRF(c2, x * x, 1);
        }

        // arsech x = ln(1 + √(1 - x²)) - ln x (0 ≤ x ≤ 1; 両項非負で桁落ち・小 x オーバーフローなし)
        private static double InverseSech(double x) =>
            x == 0 ? double.PositiveInfinity : Log1p(Math.Sqrt((1 - x) * (1 + x))) - Math.Log(x);

        // Δ = 1 - m sin²φ。m ≥ 1/2 では 1 - m に丸め誤差がないため cos²φ + (1 - m) sin²φ で桁落ちを防ぐ
        private static double AmplitudeDelta(double m, double s, double c2, double delta) =>
            m >= 0.5 && m <= 1 ? c2 + (1 - m) * (s * s) : delta;

        /// <summary>
        /// Computes the arithmetic-geometric mean <c>AGM(a, b)</c>.
        /// <para>算術幾何平均 <c>AGM(a, b)</c> を計算します。</para>
        /// </summary>
        /// <param name="a">The first non-negative value (<c>a ≥ 0</c>). / 非負の第 1 引数（<c>a ≥ 0</c>）。</param>
        /// <param name="b">The second non-negative value (<c>b ≥ 0</c>). / 非負の第 2 引数（<c>b ≥ 0</c>）。</param>
        /// <returns>The value of <c>AGM(a, b)</c>, or <see cref="double.NaN"/> if either argument is negative or <see cref="double.NaN"/>. / <c>AGM(a, b)</c> の値。いずれかの引数が負または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/19.8.E1"/>
        public static double ArithmeticGeometricMean(double a, double b)
        {
            if (double.IsNaN(a) || double.IsNaN(b) || a < 0 || b < 0)
            {
                return double.NaN;
            }

            if (a == 0 || b == 0)
            {
                return 0;
            }

            if (double.IsPositiveInfinity(a) || double.IsPositiveInfinity(b))
            {
                return double.PositiveInfinity;
            }

            for (int i = 0; i < 64; i++)
            {
                if (Math.Abs(a - b) <= 2 * MachineEpsilon * a)
                {
                    break;
                }

                double next = 0.5 * a + 0.5 * b;
                b = Math.Sqrt(a) * Math.Sqrt(b);
                a = next;
            }

            return 0.5 * a + 0.5 * b;
        }

        /// <summary>
        /// Computes the elliptic nome <c>q(m) = exp(-π K(1 - m) / K(m))</c>.
        /// <para>楕円関数のネーム <c>q(m) = exp(-π K(1 - m) / K(m))</c> を計算します。</para>
        /// </summary>
        /// <param name="m">The parameter <c>m = k²</c> (<c>0 ≤ m ≤ 1</c>). / パラメータ <c>m = k²</c>（<c>0 ≤ m ≤ 1</c>）。</param>
        /// <returns>The value of <c>q(m)</c> (<c>0</c> at <c>0</c>, <c>1</c> at <c>1</c>), or <see cref="double.NaN"/> if out of domain. / <c>q(m)</c> の値（<c>0</c> で <c>0</c>、<c>1</c> で <c>1</c>）。定義域外では <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/22.2.E1"/>
        public static double EllipticNome(double m)
        {
            if (double.IsNaN(m) || m < 0 || m > 1)
            {
                return double.NaN;
            }

            if (m == 0 || m == 1)
            {
                return m;
            }

            if (m <= 0.5)
            {
                // q = ε + 2ε⁵ + 15ε⁹ + … (DLMF 19.5.5 の逆展開; ε ≤ 0.044)。m → 0 で指数 -πK'/K の丸め誤差を避ける
                double root = Math.Pow(1 - m, 0.25);
                double epsilon = -Expm1(0.25 * Log1p(-m)) / (2 * (1 + root));
                double e4 = epsilon * epsilon * epsilon * epsilon;
                double sum = 0;
                for (int k = EllipticNomeSeries.Length - 1; k >= 0; k--)
                {
                    sum = sum * e4 + EllipticNomeSeries[k];
                }

                return epsilon * sum;
            }

            return Math.Exp(-Math.PI * CarlsonRF(0, m, 1) / CarlsonRF(0, 1 - m, 1));
        }

        // q = Σ cₖ ε^(4k+1) の係数 (EllipticNome)
        private static readonly double[] EllipticNomeSeries = [1, 2, 15, 150, 1707, 20910, 268616, 3567400, 48555069];

        /// <summary>
        /// Computes the elliptic parameter <c>m(q) = (θ₂(0, q) / θ₃(0, q))⁴</c> from the nome <paramref name="q"/>.
        /// <para>ネーム <paramref name="q"/> から楕円パラメータ <c>m(q) = (θ₂(0, q) / θ₃(0, q))⁴</c> を逆算します。</para>
        /// </summary>
        /// <param name="q">The elliptic nome (<c>0 ≤ q ≤ 1</c>). / ネーム（<c>0 ≤ q ≤ 1</c>）。</param>
        /// <returns>The parameter <c>m(q)</c>, or <see cref="double.NaN"/> if out of domain. / パラメータ <c>m(q)</c>。定義域外では <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/20.9.E1"/>
        public static double EllipticNomeInverse(double q)
        {
            if (double.IsNaN(q) || q < 0 || q > 1)
            {
                return double.NaN;
            }

            if (q == 0 || q == 1)
            {
                return q;
            }

            double theta3 = JacobiTheta3(0, q);
            double ratio2 = JacobiTheta2(0, q) / theta3;
            double m = ratio2 * ratio2 * (ratio2 * ratio2);
            if (m <= 0.5)
            {
                return m;
            }

            double ratio4 = JacobiTheta4(0, q) / theta3;
            return 1 - ratio4 * ratio4 * (ratio4 * ratio4);
        }
    }
}
