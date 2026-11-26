namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        /// <summary>
        /// Computes the principal branch <c>W₀(x)</c> of the Lambert W function.
        /// <para>ランベルトの W 関数の主枝 <c>W₀(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The real argument (<c>x ≥ -1/e</c>). / 実数の引数（<c>x ≥ -1/e</c>）。</param>
        /// <returns>The value <c>w ≥ -1</c> satisfying <c>w eʷ = x</c>, or <see cref="double.NaN"/> if <c>x &lt; -1/e</c> or <paramref name="x"/> is <see cref="double.NaN"/>. / <c>w eʷ = x</c> を満たす <c>w ≥ -1</c> の値。<c>x &lt; -1/e</c> または <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://en.wikipedia.org/wiki/Lambert_W_function"/>
        public static double LambertW0(double x)
        {
            if (double.IsNaN(x) || x == 0 || double.IsPositiveInfinity(x))
            {
                return x;
            }

            if (!TryBranchPointDistance(x, out double q))
            {
                return double.NaN;
            }

            if (q == 0)
            {
                return -1;
            }

            if (q < 0.5)
            {
                // 分岐点 -1/e の近傍: p = √(2(ex+1)) の級数を初期値とし、W = t - 1 として解く
                double p = Math.Sqrt(2 * q);
                double t = p * (1 + p * (-1.0 / 3 + p * (11.0 / 72 + p * (-43.0 / 540 + p * (769.0 / 17280)))));
                return SolveNearBranchPoint(t, q) - 1;
            }

            if (x < Math.E)
            {
                // Winitzki の近似を初期値とし、w eʷ = x をハレー法で解く
                double l = Log1p(x);
                double w = l * (1 - Log1p(l) / (2 + l));
                for (int i = 0; i < 50; i++)
                {
                    double ew = Math.Exp(w);
                    double f = w * ew - x;
                    double wp1 = w + 1;
                    double step = f / (ew * wp1 - (w + 2) * f / (2 * wp1));
                    w -= step;
                    if (Math.Abs(step) <= 2 * Epsilon * Math.Abs(w))
                    {
                        break;
                    }
                }

                return w;
            }

            // w + ln w = ln x をハレー法で解く（オーバーフローを避ける）
            double lnx = Math.Log(x);
            double l2 = Math.Log(lnx);
            return SolveLogForm(lnx - l2 + l2 / lnx, lnx, negative: false);
        }

        /// <summary>
        /// Computes the lower real branch <c>W₋₁(x)</c> of the Lambert W function.
        /// <para>ランベルトの W 関数の下側の枝 <c>W₋₁(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The real argument (<c>-1/e ≤ x ≤ 0</c>). / 実数の引数（<c>-1/e ≤ x ≤ 0</c>）。</param>
        /// <returns>The value <c>w ≤ -1</c> satisfying <c>w eʷ = x</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is out of domain or <see cref="double.NaN"/>. / <c>w eʷ = x</c> を満たす <c>w ≤ -1</c> の値。<paramref name="x"/> が定義域外または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <see cref="double.NegativeInfinity"/> at <c>x = 0</c>.</para>
        /// <para><c>x = 0</c> では <see cref="double.NegativeInfinity"/> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Lambert_W_function"/>
        public static double LambertWm1(double x)
        {
            if (double.IsNaN(x) || x > 0)
            {
                return double.NaN;
            }

            if (x == 0)
            {
                return double.NegativeInfinity;
            }

            if (!TryBranchPointDistance(x, out double q))
            {
                return double.NaN;
            }

            if (q == 0)
            {
                return -1;
            }

            if (q < 0.5)
            {
                double p = -Math.Sqrt(2 * q);
                double t = p * (1 + p * (-1.0 / 3 + p * (11.0 / 72 + p * (-43.0 / 540 + p * (769.0 / 17280)))));
                return SolveNearBranchPoint(t, q) - 1;
            }

            // w + ln(-w) = ln(-x)
            double l1 = Math.Log(-x);
            double l2 = Math.Log(-l1);
            return SolveLogForm(l1 - l2 + l2 / l1, l1, negative: true);
        }

        // 分岐点からの距離 q = ex + 1 を e の丸め誤差を補正して求める
        private static bool TryBranchPointDistance(double x, out double q)
        {
            q = Math.FusedMultiplyAdd(Math.E, x, 1) + ELo * x;
            if (q < 0)
            {
                // 入力の丸め誤差程度のずれは分岐点とみなす
                if (q > -4 * Epsilon)
                {
                    q = 0;
                    return true;
                }

                return false;
            }

            return true;
        }

        // W = t - 1 として G(t) = (t - 1)eᵗ + 1 = q（= e(x + 1/e)）をハレー法で解く
        private static double SolveNearBranchPoint(double t, double q)
        {
            for (int i = 0; i < 50; i++)
            {
                double et = Math.Exp(t);
                double g;
                if (Math.Abs(t) < 1)
                {
                    // G(t) = Σ (k-1) tᵏ / k!（k ≥ 2）を桁落ちなく計算する
                    double power = t;
                    g = 0;
                    for (int k = 2; k < 60; k++)
                    {
                        power *= t / k;
                        double term = (k - 1) * power;
                        g += term;
                        if (Math.Abs(term) <= Epsilon * Math.Abs(g) * 0.5)
                        {
                            break;
                        }
                    }
                }
                else
                {
                    g = (t - 1) * et + 1;
                }

                double f = g - q;
                double f1 = t * et;
                double f2 = (t + 1) * et;
                if (f1 == 0)
                {
                    break;
                }

                double step = f / f1 / (1 - f * f2 / (2 * f1 * f1));
                t -= step;
                if (Math.Abs(step) <= 2 * Epsilon * Math.Abs(t))
                {
                    break;
                }
            }

            return t;
        }

        // w + ln(±w) = lnX をハレー法で解く（negative が真なら w < 0）
        private static double SolveLogForm(double w, double lnX, bool negative)
        {
            for (int i = 0; i < 50; i++)
            {
                double f = w + Math.Log(negative ? -w : w) - lnX;
                double f1 = 1 + 1 / w;
                double f2 = -1 / (w * w);
                double step = f / f1 / (1 - f * f2 / (2 * f1 * f1));
                w -= step;
                if (Math.Abs(step) <= 2 * Epsilon * Math.Abs(w))
                {
                    break;
                }
            }

            return w;
        }
    }
}
