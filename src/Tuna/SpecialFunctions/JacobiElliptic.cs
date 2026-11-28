using static Tuna.XMath.Constants;
using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        /// <summary>
        /// Computes the Jacobi elliptic function <c>sn(u | m)</c>.
        /// <para>ヤコビの楕円関数 <c>sn(u | m)</c> を計算します。</para>
        /// </summary>
        /// <param name="u">The real argument. / 実数の引数。</param>
        /// <param name="m">The parameter <c>m = k²</c> (any finite real number). / パラメータ <c>m = k²</c>（任意の有限実数）。</param>
        /// <returns>The value of <c>sn(u | m)</c>, or <see cref="double.NaN"/> if either argument is non-finite. / <c>sn(u | m)</c> の値。いずれかの引数が有限でない場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Jacobi_elliptic_functions#Definition_in_terms_of_inverses_of_elliptic_integrals"/>
        public static double JacobiSN(double u, double m) => JacobiElliptic(u, m).Sn;

        /// <summary>
        /// Computes the Jacobi elliptic function <c>cn(u | m)</c>.
        /// <para>ヤコビの楕円関数 <c>cn(u | m)</c> を計算します。</para>
        /// </summary>
        /// <param name="u">The real argument. / 実数の引数。</param>
        /// <param name="m">The parameter <c>m = k²</c> (any finite real number). / パラメータ <c>m = k²</c>（任意の有限実数）。</param>
        /// <returns>The value of <c>cn(u | m)</c>, or <see cref="double.NaN"/> if either argument is non-finite. / <c>cn(u | m)</c> の値。いずれかの引数が有限でない場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Jacobi_elliptic_functions#Definition_in_terms_of_inverses_of_elliptic_integrals"/>
        public static double JacobiCN(double u, double m) => JacobiElliptic(u, m).Cn;

        /// <summary>
        /// Computes the Jacobi elliptic function <c>dn(u | m)</c>.
        /// <para>ヤコビの楕円関数 <c>dn(u | m)</c> を計算します。</para>
        /// </summary>
        /// <param name="u">The real argument. / 実数の引数。</param>
        /// <param name="m">The parameter <c>m = k²</c> (any finite real number). / パラメータ <c>m = k²</c>（任意の有限実数）。</param>
        /// <returns>The value of <c>dn(u | m)</c>, or <see cref="double.NaN"/> if either argument is non-finite. / <c>dn(u | m)</c> の値。いずれかの引数が有限でない場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Jacobi_elliptic_functions#Definition_in_terms_of_inverses_of_elliptic_integrals"/>
        public static double JacobiDN(double u, double m) => JacobiElliptic(u, m).Dn;

        /// <summary>
        /// Computes the Jacobi elliptic functions <c>sn(u | m)</c>, <c>cn(u | m)</c>, and <c>dn(u | m)</c> simultaneously.
        /// <para>ヤコビの楕円関数 <c>sn(u | m)</c>、<c>cn(u | m)</c>、<c>dn(u | m)</c> を同時に計算します。</para>
        /// </summary>
        /// <param name="u">The real argument. / 実数の引数。</param>
        /// <param name="m">The parameter <c>m = k²</c> (any finite real number). / パラメータ <c>m = k²</c>（任意の有限実数）。</param>
        /// <returns>A tuple <c>(Sn, Cn, Dn)</c>, or all <see cref="double.NaN"/> if either argument is non-finite. / タプル <c>(Sn, Cn, Dn)</c>。いずれかの引数が有限でない場合はすべて <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Takes the parameter <c>m = k²</c> (not the modulus <c>k</c>). Use this method when more than one of <c>sn</c>, <c>cn</c>, and <c>dn</c> is needed, as <see cref="JacobiSN"/>, <see cref="JacobiCN"/>, and <see cref="JacobiDN"/> evaluate the same core.</para>
        /// <para>第 2 引数は母数 <c>k</c> ではなくパラメータ <c>m = k²</c> です。<see cref="JacobiSN"/>、<see cref="JacobiCN"/>、<see cref="JacobiDN"/> はいずれも同じ本体を計算するため、複数が必要な場合はこのメソッドを使用してください。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Jacobi_elliptic_functions#Definition_in_terms_of_inverses_of_elliptic_integrals"/>
        public static (double Sn, double Cn, double Dn) JacobiElliptic(double u, double m)
        {
            if (double.IsNaN(u) || double.IsNaN(m) || double.IsInfinity(u) || double.IsInfinity(m))
            {
                return (double.NaN, double.NaN, double.NaN);
            }

            if (m < 0)
            {
                // 虚数モジュラス変換: μ = -m/(1-m)、v = u√(1-m)。
                // |m| が大きいと μ は 1 に丸められるので、1 - μ = 1/(1-m) を別に渡して周期を保つ。
                // v と 1 - μ の丸め誤差は |v| に比例して位相をずらすので、どちらも double-double で求めて還元する
                var oneMinusM = DoubleDoubleTwoSum(1, -m);
                var scale = DoubleDoubleSqrt(oneMinusM);
                var m1 = DoubleDoubleDivide((1.0, 0.0), oneMinusM);
                double quarter = double.NaN;
                var v = ReduceJacobiArgument(u, scale, m1, ref quarter, out _);
                var (sn, cn, dn) = JacobiEllipticCore(v, -m / (1 - m), m1, quarter);
                return (sn / (scale.Hi * dn), cn / dn, 1 / dn);
            }

            if (m > 1)
            {
                // 逆数モジュラス変換: sn(u|m) = sn(u√m | 1/m)/√m
                var k = DoubleDoubleSqrt((m, 0.0));
                var m1 = DoubleDoubleDivide(DoubleDoubleTwoSum(m, -1), m);
                double quarter = double.NaN;
                var v = ReduceJacobiArgument(u, k, m1, ref quarter, out _);
                var (sn, cn, dn) = JacobiEllipticCore(v, 1 / m, m1, quarter);
                return (sn / k.Hi, dn, cn);
            }

            if (m == 0)
            {
                return (Math.Sin(u), Math.Cos(u), 1);
            }

            var complement = DoubleDoubleTwoSum(1, -m);
            if (complement.Hi == 0)
            {
                return JacobiEllipticCore((u, 0.0), m, complement, double.NaN);
            }

            double k0 = double.NaN;
            var reduced = ReduceJacobiArgument(u, (1.0, 0.0), complement, ref k0, out _);
            return JacobiEllipticCore(reduced, m, complement, k0);
        }

        // モジュラス変換後の引数 v = u·scale を周期 4K(m1) で還元し、|r| ≤ 2K の r（double-double）を返す。
        // u·scale がオーバーフローする場合は、先に u を 4K/scale で還元してから掛ける
        private static (double Hi, double Lo) ReduceJacobiArgument(
            double u, (double Hi, double Lo) scale, (double Hi, double Lo) m1, ref double quarter, out double periods)
        {
            periods = 0;
            double v = u * scale.Hi;
            (double Hi, double Lo) x = double.IsFinite(v)
                ? DoubleDoubleNormalize(v, Math.FusedMultiplyAdd(u, scale.Hi, -v) + u * scale.Lo)
                : (v, 0.0);

            // 2K ≥ π なので、|v| ≤ π なら K を求めずに済む
            if (double.IsFinite(v) && (Math.Abs(v) <= Math.PI || Math.Abs(v) <= 2 * JacobiQuarterPeriodEstimate(m1, ref quarter)))
            {
                return x;
            }

            var period = JacobiPeriod(m1);
            if (!double.IsFinite(v))
            {
                var reduced = ReduceDoubleDouble((u, 0.0), DoubleDoubleDivide(period, scale), out periods);
                x = DoubleDoubleMultiply(reduced, scale);
            }

            var r = ReduceDoubleDouble(x, period, out double extra);
            periods += extra;
            return r;
        }

        // 周期 4K(m) = 2π/AGM(1, √(1-m)) を double-double で求める（m1 = 1 - m > 0）
        private static (double Hi, double Lo) JacobiPeriod((double Hi, double Lo) m1)
        {
            // 同じ m で続けて呼ばれることが多いので、スレッドごとに直前の結果を再利用する
            if (m1.Hi == jacobiPeriodKeyHi && m1.Lo == jacobiPeriodKeyLo && m1.Hi != 0)
            {
                return jacobiPeriodValue;
            }

            (double Hi, double Lo) a = (1.0, 0.0);
            var b = DoubleDoubleSqrt(m1);
            // 2 次収束なので、相対差が 1e-17 を下回ってから 1 回反復すれば double-double の精度に達する
            bool last = false;
            for (int i = 0; i < 64; i++)
            {
                var next = DoubleDoubleMultiply(DoubleDoubleAdd(a, b), (0.5, 0.0));
                if (last)
                {
                    a = next;
                    break;
                }

                last = Math.Abs(a.Hi - b.Hi) <= 1e-17 * a.Hi;
                b = DoubleDoubleSqrt(DoubleDoubleMultiply(a, b));
                a = next;
            }

            var period = DoubleDoubleDivide((2 * Pi, 2 * PiLo), a);
            (jacobiPeriodKeyHi, jacobiPeriodKeyLo, jacobiPeriodValue) = (m1.Hi, m1.Lo, period);
            return period;
        }

        [ThreadStatic]
        private static double jacobiPeriodKeyHi;

        [ThreadStatic]
        private static double jacobiPeriodKeyLo;

        [ThreadStatic]
        private static (double Hi, double Lo) jacobiPeriodValue;

        // double-double の x を周期 period で還元し、|r| ≤ period/2 の r を返す
        private static (double Hi, double Lo) ReduceDoubleDouble((double Hi, double Lo) x, (double Hi, double Lo) period, out double periods)
        {
            periods = 0;
            for (int i = 0; i < 64; i++)
            {
                double q = Math.Round(x.Hi / period.Hi);
                if (q == 0)
                {
                    break;
                }

                if (double.IsInfinity(q))
                {
                    // 周期との比がオーバーフローするほど大きい x では位相の情報は残らないので、有限の引数を返すことだけを目的に上位部分を還元する
                    x = (Math.IEEERemainder(x.Hi, period.Hi), 0.0);
                    continue;
                }

                // q·period を 2 つの正確な積（上位・下位）に分けて引く
                double p1 = q * period.Hi;
                double e1 = Math.FusedMultiplyAdd(q, period.Hi, -p1);
                double p2 = q * period.Lo;
                double e2 = Math.FusedMultiplyAdd(q, period.Lo, -p2);
                x = DoubleDoubleAdd(DoubleDoubleAdd(x, (-p1, -e1)), (-p2, -e2));
                periods += q;
            }

            return x;
        }

        // double-double どうしの商
        private static (double Hi, double Lo) DoubleDoubleDivide((double Hi, double Lo) a, (double Hi, double Lo) b)
        {
            double q1 = a.Hi / b.Hi;
            var remainder = DoubleDoubleAdd(a, DoubleDoubleMultiply(b, (-q1, 0.0)));
            return DoubleDoubleNormalize(q1, remainder.Hi / b.Hi);
        }

        // 0 ≤ m ≤ 1 での sn, cn, dn（u は還元済み double-double、m1 = 1 - m）。
        // cn の零点 u = ±K 近傍では DLMF 22.4.T3 のシフト公式 sn(K - w) = cn w/dn w 等を使う
        private static (double Sn, double Cn, double Dn) JacobiEllipticCore(
            (double Hi, double Lo) u, double m, (double Hi, double Lo) m1, double quarterEstimate)
        {
            if (double.IsInfinity(u.Hi))
            {
                return (double.NaN, double.NaN, double.NaN);
            }

            if (m == 0)
            {
                return (Math.Sin(u.Hi), Math.Cos(u.Hi), 1);
            }

            if (m1.Hi == 0)
            {
                // sech u = 2t / (1 + t²)（t = e^(-|u|)）。cosh u のオーバーフローを避ける。
                // t は結果より先にアンダーフローしうるので、h = e^(-|u|/2) として 2t = (2h)h を 1 回の丸めで求める
                double h = Math.Exp(-0.5 * Math.Abs(u.Hi));
                double t = h * h;
                double sech = 2 * h * h / (1 + t * t);
                return (Math.Tanh(u.Hi), sech, sech);
            }

            // K ≥ π/2 なので、|u| ≤ π/4 なら K を求めずに済む
            if (Math.Abs(u.Hi) <= Math.PI / 4)
            {
                return JacobiEllipticFromAmplitude(u.Hi + u.Lo, m, m1.Hi);
            }

            // sn は奇関数、cn と dn は偶関数。sn(2K - a) = sn a、cn(2K - a) = -cn a、dn(2K - a) = dn a。
            // 判定には倍精度の K を使い、2K - a や K - a が K/4 未満で K の丸め誤差が相対的に効く場合だけ double-double の K を求める
            double sign = Math.Sign(u.Hi);
            (double Hi, double Lo) a = u.Hi < 0 ? (-u.Hi, -u.Lo) : u;
            double quarter = JacobiQuarterPeriodEstimate(m1, ref quarterEstimate);
            double distance = a.Hi - quarter;
            if (Math.Abs(distance) <= 0.5 * quarter)
            {
                // K の近く: w = |a - K|。cn の符号は a - K の符号で決まるので、double-double の K を使う場合は
                // 倍精度の K との差 distance ではなく、正確な差の符号を使う（a が 2 つの K の間にあると符号が食い違う）
                double offset;
                if (Math.Abs(distance) >= 0.25 * quarter)
                {
                    offset = distance + a.Lo;
                }
                else
                {
                    var exact = DoubleDoubleAdd(a, Negate(JacobiQuarterPeriod(m1)));
                    offset = exact.Hi + exact.Lo;
                }

                double cnSign = offset > 0 ? -1 : 1;
                double w = Math.Abs(offset);
                var (snW, cnW, dnW) = JacobiEllipticFromAmplitude(w, m, m1.Hi);
                double k1 = Math.Sqrt(m1.Hi);
                return (sign * (cnW / dnW), cnSign * (k1 * snW / dnW), k1 / dnW);
            }

            if (distance < 0)
            {
                var (sn, cn, dn) = JacobiEllipticFromAmplitude(a.Hi + a.Lo, m, m1.Hi);
                return (sign * sn, cn, dn);
            }

            // 2K の近く: a' = 2K - a
            double reflected = (2 * quarter - a.Hi) - a.Lo;
            if (reflected < 0.25 * quarter)
            {
                var exact = DoubleDoubleAdd(DoubleDoubleMultiply(JacobiQuarterPeriod(m1), (2.0, 0.0)), Negate(a));
                reflected = exact.Hi + exact.Lo;
            }

            var (snR, cnR, dnR) = JacobiEllipticFromAmplitude(reflected, m, m1.Hi);
            return (sign * snR, -cnR, dnR);
        }

        private static (double Hi, double Lo) Negate((double Hi, double Lo) x) => (-x.Hi, -x.Lo);

        // 倍精度の K(m)（quarter が NaN なら計算してキャッシュする）
        private static double JacobiQuarterPeriodEstimate((double Hi, double Lo) m1, ref double quarter)
        {
            if (double.IsNaN(quarter))
            {
                // K = π/(2 AGM(1, √m₁))（倍精度で数 ulp。分岐判定および丸め誤差が相対的に効かない引数の計算に使用）
                double a = 1;
                double b = Math.Sqrt(m1.Hi);
                while (Math.Abs(a - b) > MachineEpsilon * a)
                {
                    double next = 0.5 * (a + b);
                    b = Math.Sqrt(a * b);
                    a = next;
                }

                quarter = Math.PI / (2 * a);
            }

            return quarter;
        }

        // 1/4 周期 K(m) の double-double
        private static (double Hi, double Lo) JacobiQuarterPeriod((double Hi, double Lo) m1) =>
            DoubleDoubleMultiply(JacobiPeriod(m1), (0.25, 0.0));

        // 0 < m < 1、|u| ≤ K での sn, cn, dn（振幅 am(u|m) から計算）
        private static (double Sn, double Cn, double Dn) JacobiEllipticFromAmplitude(double u, double m, double m1)
        {
            double phi = JacobiAmplitudeCore(u, m, m1);
            double snValue = Math.Sin(phi);
            double cnValue = Math.Cos(phi);

            // dn² = 1 - m sn² = (1 - m) + m cn²（桁落ちしない形）
            double dnValue = Math.Sqrt(m1 + m * cnValue * cnValue);
            return (snValue, cnValue, dnValue);
        }

        /// <summary>
        /// Computes the Jacobi amplitude function <c>am(u | m)</c>.
        /// <para>ヤコビの振幅関数 <c>am(u | m)</c> を計算します。</para>
        /// </summary>
        /// <param name="u">The real argument. / 実数の引数。</param>
        /// <param name="m">The parameter <c>m = k²</c> (<c>0 ≤ m ≤ 1</c>). / パラメータ <c>m = k²</c>（<c>0 ≤ m ≤ 1</c>）。</param>
        /// <returns>The amplitude <c>am(u | m)</c>, or <see cref="double.NaN"/> if <paramref name="m"/> is outside <c>[0, 1]</c> or either argument is <see cref="double.NaN"/>. / 振幅 <c>am(u | m)</c>。<paramref name="m"/> が <c>[0, 1]</c> の範囲外、またはいずれかが <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>At <c>m = 0</c>, equals <paramref name="u"/>; at <c>m = 1</c>, equals the Gudermannian function <c>arctan(sinh u)</c> (with limits <c>±π / 2</c> at <c>u = ±∞</c>).</para>
        /// <para><c>m = 0</c> では <paramref name="u"/> に一致し、<c>m = 1</c> ではグーデルマン関数 <c>arctan(sinh u)</c>（<c>u = ±∞</c> で <c>±π / 2</c>）になります。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Jacobi_elliptic_functions#Definition_in_terms_of_inverses_of_elliptic_integrals"/>
        public static double JacobiAmplitude(double u, double m)
        {
            if (double.IsNaN(u) || double.IsNaN(m) || m < 0 || m > 1)
            {
                return double.NaN;
            }

            if (m == 1)
            {
                // グーデルマン関数（u = ±∞ でも有限の極限 ±π/2 をとる）
                return Math.Atan(Math.Sinh(u));
            }

            if (m == 0 || double.IsInfinity(u))
            {
                return u;
            }

            // am(u + 4Kq) = am(u) + 2πq
            var m1 = DoubleDoubleTwoSum(1, -m);
            double quarter = double.NaN;
            var r = ReduceJacobiArgument(u, (1.0, 0.0), m1, ref quarter, out double periods);
            double phi = JacobiAmplitudeCore(r.Hi + r.Lo, m, m1.Hi);
            return periods == 0 ? phi : phi + 2 * Math.PI * periods;
        }

        // 算術幾何平均（降下ランデン変換）による振幅 am(u|m)（0 < m < 1、m1 = 1 - m）。
        // 作業領域は書き込んだ要素（0..n）しか読まないためゼロ初期化を省略する
        [System.Runtime.CompilerServices.SkipLocalsInit]
        private static double JacobiAmplitudeCore(double u, double m, double m1)
        {
            Span<double> a = stackalloc double[32];
            Span<double> c = stackalloc double[32];
            Span<double> b = stackalloc double[32];
            a[0] = 1;
            c[0] = Math.Sqrt(m);
            b[0] = Math.Sqrt(m1);
            int n = 0;
            while (n < a.Length - 1 && Math.Abs(c[n]) > MachineEpsilon * a[n])
            {
                a[n + 1] = 0.5 * (a[n] + b[n]);
                c[n + 1] = 0.5 * (a[n] - b[n]);
                b[n + 1] = Math.Sqrt(a[n] * b[n]);
                n++;
            }

            // φₙ = 2ⁿ aₙ u、φₖ₋₁ = (φₖ + arcsin z)/2（z = (cₖ/aₖ) sin φₖ）。
            // m ≈ 1 では cₖ/aₖ ≈ 1 なので |z| ≈ 1 となり、arcsin の傾きで z の丸め誤差が √ε 程度に拡大する。
            // aₖ - cₖ = bₖ₋₁ から 1 - |z| = cos²φₖ/(1 + |sin φₖ|) + (bₖ₋₁/aₖ)|sin φₖ| を打ち消しなしに求め、
            // arcsin z = atan2(z, √((1 - |z|)(1 + |z|))) とする
            double phi = Math.ScaleB(a[n] * u, n);
            for (int k = n; k >= 1; k--)
            {
                double sin = Math.Sin(phi);
                double ratio = c[k] / a[k];
                double z = ratio * sin;
                if (ratio < 0.5)
                {
                    // |z| ≤ 1/2 では arcsin の誤差は拡大しない
                    phi = 0.5 * (phi + Math.Asin(z));
                    continue;
                }

                double cos = Math.Cos(phi);
                double absSin = Math.Abs(sin);
                double oneMinus = cos * cos / (1 + absSin) + b[k - 1] / a[k] * absSin;
                double root = Math.Sqrt(oneMinus * (1 + Math.Abs(z)));
                phi = 0.5 * (phi + Math.Atan2(z, root));
            }

            return phi;
        }
    }
}
