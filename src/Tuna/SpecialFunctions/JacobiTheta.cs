using static Tuna.XMath.Constants;
using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // これより大きいノーム q > e^(-π)（t = -ln q/π < 1）では、ヤコビの虚数変換で q' = e^(-π/t) < e^(-π) の和に変換する
        private const double JacobiThetaTransformNome = 0.043213918263772250;

        /// <summary>
        /// Computes the Jacobi theta function <c>θ₁(z, q)</c>.
        /// <para>ヤコビのテータ関数 <c>θ₁(z, q)</c> を計算します。</para>
        /// </summary>
        /// <param name="z">The real argument. / 実数の引数。</param>
        /// <param name="q">The nome (<c>0 ≤ q &lt; 1</c>). / ノーム（<c>0 ≤ q &lt; 1</c>）。</param>
        /// <returns>The value of <c>θ₁(z, q)</c>, or <see cref="double.NaN"/> if <paramref name="z"/> is non-finite, <paramref name="q"/> is outside <c>[0, 1)</c>, or either argument is <see cref="double.NaN"/>. / <c>θ₁(z, q)</c> の値。<paramref name="z"/> が有限でない場合、<paramref name="q"/> が <c>[0, 1)</c> の範囲外の場合、またはいずれかが <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/20.2.E1"/>
        public static double JacobiTheta1(double z, double q)
        {
            if (!TryReduceTheta(z, q, out double r, out double sign))
            {
                return double.NaN;
            }

            return sign * JacobiTheta1Core(r, q);
        }

        /// <summary>
        /// Computes the Jacobi theta function <c>θ₂(z, q)</c>.
        /// <para>ヤコビのテータ関数 <c>θ₂(z, q)</c> を計算します。</para>
        /// </summary>
        /// <param name="z">The real argument. / 実数の引数。</param>
        /// <param name="q">The nome (<c>0 ≤ q &lt; 1</c>). / ノーム（<c>0 ≤ q &lt; 1</c>）。</param>
        /// <returns>The value of <c>θ₂(z, q)</c>, or <see cref="double.NaN"/> if <paramref name="z"/> is non-finite, <paramref name="q"/> is outside <c>[0, 1)</c>, or either argument is <see cref="double.NaN"/>. / <c>θ₂(z, q)</c> の値。<paramref name="z"/> が有限でない場合、<paramref name="q"/> が <c>[0, 1)</c> の範囲外の場合、またはいずれかが <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/20.2.E2"/>
        public static double JacobiTheta2(double z, double q)
        {
            if (!TryReduceTheta(z, q, out double r, out double rLo, out double sign))
            {
                return double.NaN;
            }

            if (Math.Abs(r) > Math.PI / 4)
            {
                // 零点 z = (k + 1/2)π の近く。θ₂(z) = θ₁(z + π/2) とし、z + π/2 を π の上位・下位で丸めなく還元する
                // （還元済みの r から π/2 - |r| を作ると、r の丸め誤差が零点からの距離の相対誤差になる）
                if (Math.Abs(z) >= ExactPiReductionThreshold)
                {
                    // z = kπ + r（double-double）から、r ∓ π/2 を作る（π/4 < |r| ≤ π/2 なので上位どうしの差は丸められない）。
                    // r > 0 なら z + π/2 = (k+1)π + (r - π/2) で符号が反転する
                    double shifted = r > 0
                        ? (r - Pi / 2) + (rLo - PiLo / 2)
                        : (r + Pi / 2) + (rLo + PiLo / 2);
                    return (r > 0 ? -sign : sign) * JacobiTheta1Core(shifted, q);
                }

                double periods = Math.Round(z / Math.PI + 0.5);
                double w = (Math.FusedMultiplyAdd(-periods, Pi, z) + Pi / 2) + (PiLo / 2 - periods * PiLo);
                double shiftSign = Math.IEEERemainder(periods, 2) == 0 ? 1 : -1;
                return shiftSign * JacobiTheta1Core(w, q);
            }

            if (q == 0)
            {
                return 0;
            }

            if (q <= JacobiThetaTransformNome)
            {
                double q2 = q * q;
                double sum = 0;
                double power = 1;
                double ratio = q2;
                for (int n = 0; n < 32; n++)
                {
                    double term = power * Math.Cos((2 * n + 1) * r);
                    sum += term;
                    if (power <= MachineEpsilon * MachineEpsilon * Math.Abs(sum))
                    {
                        break;
                    }

                    // q^(n(n+1)) → q^((n+1)(n+2))：比は q^(2n+2)
                    power *= ratio;
                    ratio *= q2;
                }

                return sign * 2 * Math.Sqrt(Math.Sqrt(q)) * sum;
            }

            // θ₂(z, q) = t^(-1/2) Σ (-1)ⁿ exp(-(z + nπ)²/(πt))
            double t = -ThetaLogNome(q) / Math.PI;
            double total = Math.Exp(-r * r / (Math.PI * t));
            for (int n = 1; n < 64; n++)
            {
                double plus = r + n * Math.PI;
                double minus = r - n * Math.PI;
                double term = Math.Exp(-plus * plus / (Math.PI * t)) + Math.Exp(-minus * minus / (Math.PI * t));
                total += n % 2 == 1 ? -term : term;
                if (term <= MachineEpsilon * MachineEpsilon * total)
                {
                    break;
                }
            }

            return sign * total / Math.Sqrt(t);
        }

        /// <summary>
        /// Computes the Jacobi theta function <c>θ₃(z, q)</c>.
        /// <para>ヤコビのテータ関数 <c>θ₃(z, q)</c> を計算します。</para>
        /// </summary>
        /// <param name="z">The real argument. / 実数の引数。</param>
        /// <param name="q">The nome (<c>-1 &lt; q &lt; 1</c>; satisfies <c>θ₃(z, -q) = θ₄(z, q)</c>). / ノーム（<c>-1 &lt; q &lt; 1</c>。<c>θ₃(z, -q) = θ₄(z, q)</c>）。</param>
        /// <returns>The value of <c>θ₃(z, q)</c>, or <see cref="double.NaN"/> if <paramref name="z"/> is non-finite, <c>|q| ≥ 1</c>, or either argument is <see cref="double.NaN"/>. / <c>θ₃(z, q)</c> の値。<paramref name="z"/> が有限でない場合、<c>|q| ≥ 1</c> の場合、またはいずれかが <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/20.2.E3"/>
        public static double JacobiTheta3(double z, double q)
        {
            if (q < 0)
            {
                return JacobiTheta4(z, -q);
            }

            if (!TryReduceTheta(z, q, out double r, out _))
            {
                return double.NaN;
            }

            return JacobiTheta3Core(r, q, 0);
        }

        /// <summary>
        /// Computes the Jacobi theta function <c>θ₄(z, q)</c>.
        /// <para>ヤコビのテータ関数 <c>θ₄(z, q)</c> を計算します。</para>
        /// </summary>
        /// <param name="z">The real argument. / 実数の引数。</param>
        /// <param name="q">The nome (<c>-1 &lt; q &lt; 1</c>; satisfies <c>θ₄(z, -q) = θ₃(z, q)</c>). / ノーム（<c>-1 &lt; q &lt; 1</c>。<c>θ₄(z, -q) = θ₃(z, q)</c>）。</param>
        /// <returns>The value of <c>θ₄(z, q)</c>, or <see cref="double.NaN"/> if <paramref name="z"/> is non-finite, <c>|q| ≥ 1</c>, or either argument is <see cref="double.NaN"/>. / <c>θ₄(z, q)</c> の値。<paramref name="z"/> が有限でない場合、<c>|q| ≥ 1</c> の場合、またはいずれかが <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/20.2.E4"/>
        public static double JacobiTheta4(double z, double q)
        {
            if (q < 0)
            {
                return JacobiTheta3(z, -q);
            }

            if (!TryReduceTheta(z, q, out double r, out _))
            {
                return double.NaN;
            }

            return JacobiTheta3Core(r, q, 1);
        }

        // θ₁(r, q)（|r| ≤ π/2）。q ≤ e^(-π) では級数、それ以外は虚数変換したポアソン和を使う
        private static double JacobiTheta1Core(double r, double q)
        {
            if (q == 0 || r == 0)
            {
                return q == 0 ? 0 : r;
            }

            if (q <= JacobiThetaTransformNome)
            {
                double q2 = q * q;
                double sum = 0;
                double power = 1;
                double ratio = q2;
                for (int n = 0; n < 32; n++)
                {
                    double term = power * Math.Sin((2 * n + 1) * r);
                    sum += n % 2 == 0 ? term : -term;
                    if (power <= MachineEpsilon * MachineEpsilon * Math.Abs(sum))
                    {
                        break;
                    }

                    power *= ratio;
                    ratio *= q2;
                }

                return 2 * Math.Sqrt(Math.Sqrt(q)) * sum;
            }

            // 項 n と 1 - n を組にして exp(-(|r| - a)²/(πt)) (1 - exp(-4a|r|/(πt)))（a = (n - 1/2)π）
            double t = -ThetaLogNome(q) / Math.PI;
            double x = Math.Abs(r);
            double total = 0;
            for (int n = 1; n < 64; n++)
            {
                double a = (n - 0.5) * Math.PI;
                double d = x - a;
                double term = Math.Exp(-d * d / (Math.PI * t)) * -Expm1(-4 * a * x / (Math.PI * t));
                total += n % 2 == 1 ? term : -term;
                if (Math.Abs(term) <= MachineEpsilon * MachineEpsilon * Math.Abs(total))
                {
                    break;
                }
            }

            return Math.Sign(r) * total / Math.Sqrt(t);
        }

        // θ₃(r + shift·π/2, q)（|r| ≤ π/2、shift = 1 で θ₄(r, q)）
        private static double JacobiTheta3Core(double r, double q, int shift)
        {
            if (q == 0)
            {
                return 1;
            }

            if (q <= JacobiThetaTransformNome)
            {
                double sum = 0;
                for (int n = 1; n < 32; n++)
                {
                    double power = Math.Pow(q, (double)n * n);
                    double term = power * Math.Cos(2 * n * r);
                    sum += shift == 1 && n % 2 == 1 ? -term : term;
                    if (power <= MachineEpsilon * MachineEpsilon)
                    {
                        break;
                    }
                }

                return 1 + 2 * sum;
            }

            // θ₃(z, q) = t^(-1/2) Σ exp(-(z + nπ)²/(πt))。θ₄ は z = r + π/2 とし、最も近い格子点からのずれ center を中心に加算する
            double t = -ThetaLogNome(q) / Math.PI;
            double center = shift == 0 ? r : (r >= 0 ? r - Math.PI / 2 : r + Math.PI / 2);
            double total = Math.Exp(-center * center / (Math.PI * t));
            for (int n = 1; n < 64; n++)
            {
                double plus = center + n * Math.PI;
                double minus = center - n * Math.PI;
                double term = Math.Exp(-plus * plus / (Math.PI * t)) + Math.Exp(-minus * minus / (Math.PI * t));
                total += term;
                if (term <= MachineEpsilon * MachineEpsilon * total)
                {
                    break;
                }
            }

            return total / Math.Sqrt(t);
        }

        // ln q（q ≥ 1/2 では q - 1 が正確に求まるため Log1p を使う）
        private static double ThetaLogNome(double q) => q >= 0.5 ? Log1p(q - 1) : Math.Log(q);

        // 引数を周期 π で |r| ≤ π/2 に還元する（sign は θ₁, θ₂ の反周期性による符号 (-1)^k）
        private static bool TryReduceTheta(double z, double q, out double r, out double sign) =>
            TryReduceTheta(z, q, out r, out _, out sign);

        // 還元の剰余を double-double（r + rLo）で返すオーバーロード。大きな |z| では ReduceByPiExact を使い、それ以外では rLo = 0
        private static bool TryReduceTheta(double z, double q, out double r, out double rLo, out double sign)
        {
            r = rLo = sign = double.NaN;
            if (double.IsNaN(z) || double.IsNaN(q) || double.IsInfinity(z) || q < 0 || q >= 1)
            {
                return false;
            }

            if (Math.Abs(z) >= ExactPiReductionThreshold)
            {
                int quadrant = ReduceByPiExact(z, out r, out rLo);
                sign = (quadrant & 1) == 0 ? 1 : -1;
                return true;
            }

            rLo = 0;
            double periods = Math.Round(z / Math.PI);
            r = periods == 0 ? z : Math.FusedMultiplyAdd(-periods, Pi, z) - periods * PiLo;
            sign = Math.IEEERemainder(periods, 2) == 0 ? 1 : -1;
            return true;
        }
    }
}
