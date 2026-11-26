namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // これより大きい ah（h ≥ 0、0 < a ≤ 1）では T(h, a) = Q(h)/2 とする（省略する積分項は相対 3×10⁻¹⁹ 未満）
        private const double OwensTLargeAh = 9;

        // 64 点ガウス・ルジャンドル則（正の節点 32 個）。ah ≤ 9、a ≤ 1 での誤差の上界は 10⁻³⁰ 程度
        private static readonly (double[] Nodes, double[] Weights) OwensTRule = CreateGaussLegendreRule(64);

        /// <summary>
        /// Computes Owen's T function <c>T(h, a)</c>.
        /// <para>Owen の T 関数 <c>T(h, a)</c> を計算します。</para>
        /// </summary>
        /// <param name="h">The first argument. / 第 1 引数。</param>
        /// <param name="a">The second argument. / 第 2 引数。</param>
        /// <returns>The value of <c>T(h, a)</c>, or <see cref="double.NaN"/> if either argument is <see cref="double.NaN"/>. / <c>T(h, a)</c> の値。いずれかの引数が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Satisfies <c>T(-h, a) = T(h, a)</c> and <c>T(h, -a) = -T(h, a)</c>, with limits <c>T(h, ±∞) = ±Q(|h|) / 2</c> and <c>T(±∞, a) = 0</c>.</para>
        /// <para><c>T(-h, a) = T(h, a)</c> および <c>T(h, -a) = -T(h, a)</c> を満たし、極限では <c>T(h, ±∞) = ±Q(|h|) / 2</c>、<c>T(±∞, a) = 0</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Owen%27s_T_function"/>
        public static double OwensT(double h, double a)
        {
            if (double.IsNaN(h) || double.IsNaN(a))
            {
                return double.NaN;
            }

            double absH = Math.Abs(h);
            double absA = Math.Abs(a);
            double value;
            if (absA == 0 || double.IsPositiveInfinity(absH))
            {
                value = 0;
            }
            else if (double.IsPositiveInfinity(absA))
            {
                value = 0.5 * NormalUpperTail(absH);
            }
            else if (absA <= 1)
            {
                value = OwensTCore(absH, absA);
            }
            else
            {
                // a > 1 は T(h, a) + T(ah, 1/a) = Q(h)/2 + Q(ah)/2 - Q(h)Q(ah)（h ≥ 0、a > 0）で 1/a < 1 に帰着させる。
                // 右辺は T(h, a) ≥ T(h, 1) = Φ(h)Q(h)/2 の 4 倍以下なので、減算による桁落ちは 2 ビット程度
                double ah = absA * absH;
                double reflected = OwensTCore(ah, 1 / absA);
                if (ah <= 0.67)
                {
                    // h と ah が小さいと右辺の 3 項はほぼ打ち消し合うので、Q = 1/2 - n（n = Φ - 1/2 = erf(x/√2)/2）を代入して 1/4 - n_h n_ah とする
                    double nh = 0.5 * Erf(absH * InvSqrt2);
                    double nah = 0.5 * Erf(ah * InvSqrt2);
                    value = 0.25 - nh * nah - reflected;
                }
                else
                {
                    double qh = NormalUpperTail(absH);
                    double qah = NormalUpperTail(ah);
                    value = 0.5 * (qh + qah) - qh * qah - reflected;
                }
            }

            return a < 0 ? -value : value;
        }

        // T(h, a)（h ≥ 0、0 < a ≤ 1）を t = as で変数変換した積分で評価する。
        // ah ≤ 9 ではガウス・ルジャンドル則、ah > 9 では T = Q(h)/2 とする
        private static double OwensTCore(double h, double a)
        {
            double ha = h * a;
            if (!(ha <= OwensTLargeAh))
            {
                return 0.5 * NormalUpperTail(h);
            }

            // 被積分関数は s について偶関数なので、∫₀¹ = (1/2)∫₋₁¹ を [-1, 1] の規則の正の節点だけで評価する
            double c = 0.5 * ha * ha;
            double a2 = a * a;
            double sum = 0;
            double[] nodes = OwensTRule.Nodes;
            double[] weights = OwensTRule.Weights;
            for (int i = 0; i < nodes.Length; i++)
            {
                double s2 = nodes[i] * nodes[i];
                sum += weights[i] * Math.Exp(-c * s2) / (1 + a2 * s2);
            }

            // e^(-h²/2) は h² の丸め誤差を補正して掛け、値が非正規数になる裾でも丸めを最後の1回にする
            return ScaledExpMinusHalfSquare(h, a * sum * InvTwoPi);
        }
    }
}
