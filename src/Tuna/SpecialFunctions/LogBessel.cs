namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        /// <summary>
        /// Computes the natural logarithm of the modified Bessel function of the first kind <c>ln I_ν(x)</c>.
        /// <para>第 1 種変形ベッセル関数の自然対数 <c>ln I_ν(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="nu">The order (<c>ν &gt; -1</c>, or any finite integer, using <c>I₋ₙ = Iₙ</c>). / 次数（<c>ν &gt; -1</c> または有限の整数。<c>I₋ₙ = Iₙ</c> を用います）。</param>
        /// <param name="x">The non-negative argument (<c>x ≥ 0</c>). / 非負の引数（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>ln I_ν(x)</c>, or <see cref="double.NaN"/> if parameters are out of domain or <see cref="double.NaN"/>. / <c>ln I_ν(x)</c> の値。引数が定義域外または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>At <c>x = 0</c>, returns <c>0</c> for <c>ν = 0</c>, <see cref="double.NegativeInfinity"/> for <c>ν &gt; 0</c>, and <see cref="double.PositiveInfinity"/> for <c>-1 &lt; ν &lt; 0</c>. Returns <see cref="double.PositiveInfinity"/> at <c>x = +∞</c>, and <see cref="double.NaN"/> for non-integer <c>ν &lt; -1</c> (where <c>I_ν</c> may be negative). Remains finite even where <c>I_ν(x)</c> overflows or underflows.</para>
        /// <para><c>x = 0</c> では <c>ν = 0</c> で <c>0</c>、<c>ν &gt; 0</c> で <see cref="double.NegativeInfinity"/>、<c>-1 &lt; ν &lt; 0</c> で <see cref="double.PositiveInfinity"/> を返します。<c>x = +∞</c> では <see cref="double.PositiveInfinity"/>、<c>I_ν</c> が負になりうる非整数 <c>ν &lt; -1</c> では <see cref="double.NaN"/> を返します。<c>I_ν(x)</c> がオーバーフローまたはアンダーフローする領域でも有限の値を保ちます。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/10.25.E2"/>
        public static double LogBesselI(double nu, double x) => LogBesselIChecked(nu, x, scaled: false);

        /// <summary>
        /// Computes the natural logarithm of the exponentially scaled modified Bessel function of the first kind <c>ln(e⁻ˣ I_ν(x))</c>.
        /// <para>指数スケーリングされた第 1 種変形ベッセル関数の自然対数 <c>ln(e⁻ˣ I_ν(x))</c> を計算します。</para>
        /// </summary>
        /// <param name="nu">The order (<c>ν &gt; -1</c>, or any finite integer, using <c>I₋ₙ = Iₙ</c>). / 次数（<c>ν &gt; -1</c> または有限の整数。<c>I₋ₙ = Iₙ</c> を用います）。</param>
        /// <param name="x">The non-negative argument (<c>x ≥ 0</c>). / 非負の引数（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>ln(e⁻ˣ I_ν(x)) = ln I_ν(x) - x</c>, or <see cref="double.NaN"/> if parameters are out of domain or <see cref="double.NaN"/>. / <c>ln(e⁻ˣ I_ν(x)) = ln I_ν(x) - x</c> の値。引数が定義域外または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Avoids catastrophic cancellation between <c>ln I_ν(x)</c> and <c>x</c> for large <paramref name="x"/>. Matches <see cref="LogBesselI"/> at <c>x = 0</c> and returns <see cref="double.NegativeInfinity"/> at <c>x = +∞</c>.</para>
        /// <para>大きな <paramref name="x"/> で <c>ln I_ν(x) - x</c> を直接計算する際の桁落ちを防ぎます。<c>x = 0</c> では <see cref="LogBesselI"/> と一致し、<c>x = +∞</c> では <see cref="double.NegativeInfinity"/> を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/10.25.E2"/>
        public static double LogBesselIScaled(double nu, double x) => LogBesselIChecked(nu, x, scaled: true);

        private static double LogBesselIChecked(double nu, double x, bool scaled)
        {
            if (double.IsNaN(nu) || double.IsNaN(x) || double.IsInfinity(nu) || x < 0)
            {
                return double.NaN;
            }

            if (nu < 0)
            {
                if (IsInteger(nu))
                {
                    nu = -nu;
                }
                else if (nu < -1)
                {
                    return double.NaN;
                }
            }

            if (x == 0)
            {
                return nu == 0 ? 0 : nu > 0 ? double.NegativeInfinity : double.PositiveInfinity;
            }

            if (double.IsPositiveInfinity(x))
            {
                return scaled ? double.NegativeInfinity : double.PositiveInfinity;
            }

            return LogBesselICore(nu, x, scaled);
        }

        /// <summary>
        /// Computes the ratio of consecutive modified Bessel functions of the first kind <c>I_(ν+1)(x) / I_ν(x)</c>.
        /// <para>隣接する次数の第 1 種変形ベッセル関数の比 <c>I_(ν+1)(x) / I_ν(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="nu">The order (<c>ν &gt; -1</c>, or any finite integer). / 次数（<c>ν &gt; -1</c> または有限の整数）。</param>
        /// <param name="x">The argument (<c>x ≥ 0</c> for non-integer <paramref name="nu"/>; any real value for integer <paramref name="nu"/>). / 引数（非整数次数では <c>x ≥ 0</c>、整数次数では任意の実数）。</param>
        /// <returns>The ratio <c>I_(ν+1)(x) / I_ν(x)</c>, or <see cref="double.NaN"/> if parameters are out of domain or <see cref="double.NaN"/>. / 比 <c>I_(ν+1)(x) / I_ν(x)</c>。引数が定義域外または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Remains finite even where <c>I_ν(x)</c> overflows or underflows. For <c>ν &gt; -1</c>, equals <c>0</c> at <c>x = 0</c> and approaches <c>1</c> as <c>x → +∞</c>. For integer <paramref name="nu"/>, extends to negative <paramref name="x"/> as an odd function; for integer <c>ν ≤ -1</c>, returns <see cref="double.PositiveInfinity"/> at <c>x = 0</c>.</para>
        /// <para>個々の <c>I_ν(x)</c> がオーバーフローまたはアンダーフローする領域でも有限の値を保ちます。<c>ν &gt; -1</c> では <c>x = 0</c> で <c>0</c>、<c>x → +∞</c> で <c>1</c> に収束します。整数次数では奇関数として負の <paramref name="x"/> も受け付け、整数 <c>ν ≤ -1</c> の <c>x = 0</c> では <see cref="double.PositiveInfinity"/> を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/10.25.E2"/>
        public static double BesselIRatio(double nu, double x)
        {
            if (double.IsNaN(nu) || double.IsNaN(x) || double.IsInfinity(nu))
            {
                return double.NaN;
            }

            bool integer = IsInteger(nu);
            if (nu < -1 && !integer)
            {
                return double.NaN;
            }

            if (x < 0)
            {
                // I_ν(-x) = (-1)^ν I_ν(x) より、比は x について奇関数
                return integer ? -BesselIRatio(nu, -x) : double.NaN;
            }

            if (nu <= -1)
            {
                // ν = -n（n ≥ 1）: I_(1-n)/I_(-n) = I_(n-1)/I_n
                return 1 / BesselIRatioPositive(-nu - 1, x);
            }

            return BesselIRatioPositive(nu, x);
        }

        // ln I_ν(x) または ln(e⁻ˣ I_ν(x))（ν > -1、0 < x < ∞）
        private static double LogBesselICore(double nu, double x, bool scaled)
        {
            if (nu < DebyeMinOrder && x < 2)
            {
                // ln I_ν(x) = ν ln(x/2) - ln Γ(ν + 1) + ln(1 + Σₖ₌₁ (x²/4)ᵏ/(k! (ν+1)ₖ))。x/2 は非正規数の x で丸め誤差が出るため対数の差にする。
                // ln Γ(1 + ν) は 1 + ν を丸めると小さい ν で -γν が桁落ちするため LogGamma1p で求める。
                // 小さい x では和が 1 に近く 1 + … の丸めで精度が落ちるため、定数項を除いた和を Log1p に渡す。
                // スケールしない値は x を引いて足し戻すと打ち消し合うため、ここで直接求める
                double log = nu * (Math.Log(x) - Ln2) - LogGamma1p(nu) + Log1p(BesselISeriesTail(nu, x));
                return scaled ? log - x : log;
            }

            double logScaled = nu >= DebyeMinOrder
                ? LogBesselIScaledDebye(nu, x)
                // ν < 50、x ≥ 2 では e⁻ˣ I_ν(x) ≥ e⁻² I₅₀(2) ≈ 4×10⁻⁶⁶ なので、正規化数の範囲で直接求まる
                : Math.Log(BesselIScaledPositive(nu, x));
            return scaled || !double.IsFinite(logScaled) ? logScaled : logScaled + x;
        }

        // 原点での級数から定数項を除いた和 Σₖ₌₁ (x²/4)ᵏ/(k! (ν+1)ₖ)（ν > -1、0 < x < 2。全項が正）
        private static double BesselISeriesTail(double nu, double x)
        {
            double t = 0.25 * x * x;
            double term = t / (nu + 1);
            double sum = term;
            for (int k = 2; k < 1000; k++)
            {
                term *= t / (k * (nu + k));
                sum += term;
                if (term <= sum * Epsilon * 0.5)
                {
                    break;
                }
            }

            return sum;
        }

        // I_(ν+1)(x)/I_ν(x)（ν > -1、x ≥ 0）
        private static double BesselIRatioPositive(double nu, double x)
        {
            if (x == 0)
            {
                return 0;
            }

            if (double.IsPositiveInfinity(x))
            {
                return 1;
            }

            if (nu >= DebyeMinOrder)
            {
                // x ≤ ν では連分数が速く収束する（各段の係数 2(ν+k)/x が 2 以上）
                return x <= nu ? BesselIRatioFraction(nu, x) : BesselIRatioDebye(nu, x);
            }

            double nu1 = nu + 1;
            if (x >= Math.Max(25, nu1 * nu1))
            {
                // 両方の次数がハンケルの漸近展開の範囲（BesselIKCore と同じ条件）にある。e^x/√(2πx) は比で打ち消し合う
                BesselAsymptoticSeries(nu1, x, alternating: false, out double numerator, out _);
                BesselAsymptoticSeries(nu, x, alternating: false, out double denominator, out _);
                return numerator / denominator;
            }

            // 連分数の反復は x > ν で O(x) 回になるが、この範囲では x < max(25, (ν+1)²) ≤ 2601
            return BesselIRatioFraction(nu, x);
        }

        // ガウスの連分数 I_(ν+1)/I_ν = x/(2(ν+1) + x²/(2(ν+2) + …))（ν > -1、x > 0）を Lentz 法で評価する。
        // 漸化式 I_(ν-1) - I_(ν+1) = (2ν/x) I_ν から得られ、全項が正のため桁落ちしない
        private static double BesselIRatioFraction(double nu, double x)
        {
            // x ≥ 1 では各段を x で割った形 1/(2(ν+1)/x + 1/(2(ν+2)/x + …)) とし、x² のオーバーフローを避ける。
            // x < 1 では 2(ν+k)/x が非正規数の x でオーバーフローするので、x² の形のまま評価する
            // 2(ν+k)/x は (ν+k)·(2/x) とする（2倍は丸め誤差を生まないので同じ値）。最大の有限値に近い ν では 2(ν+k) がオーバーフローする
            bool scaled = x >= 1;
            double a = scaled ? 1 : x * x;
            double twiceInverse = scaled ? 2 / x : 2;
            double f = (nu + 1) * twiceInverse;
            double c = f;
            double d = 0;
            for (int k = 2; k <= BesselMaxIterations; k++)
            {
                double b = (nu + k) * twiceInverse;
                d = 1 / (b + a * d);
                c = b + a / c;
                double delta = c * d;
                f *= delta;
                if (Math.Abs(delta - 1) <= Epsilon)
                {
                    break;
                }
            }

            return scaled ? 1 / f : x / f;
        }
    }
}
