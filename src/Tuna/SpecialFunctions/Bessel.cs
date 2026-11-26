using System.Numerics;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // 連分数・漸化式でのゼロ除算回避用の微小値
        private const double BesselTiny = 1e-290;

        // BesselIKCore の漸化式で値が 1e250 を超えるたびに縮小する 2 の冪指数 (2^830 ≈ 7e249。2 の冪のため丸め誤差なし)
        private const int BesselRescaleShift = 830;

        private const int BesselMaxIterations = 10_000_000;

        // 原点付近の級数を使う引数上限。1/x は非正規化数でオーバーフローするため連分数・漸化式を避け、先頭 2 項で倍精度に収める ((x/2)⁴ ≤ 2⁻¹⁰⁸)
        private const double BesselSmallArgument = 1.4901161193847656e-8;

        // 1/Γ(1 + μ) = Σ cₖ μᵏ (k = 0, …, 26) の係数
        private static readonly double[] ReciprocalGammaTaylor =
        [
            1.0, 0.5772156649015329, -0.6558780715202539, -0.04200263503409524, 0.16653861138229148,
            -0.04219773455554433, -0.009621971527876973, 0.0072189432466631, -0.0011651675918590652,
            -0.00021524167411495098, 0.0001280502823881162, -2.013485478078824e-05, -1.2504934821426706e-06,
            1.133027231981696e-06, -2.056338416977607e-07, 6.116095104481416e-09, 5.002007644469223e-09,
            -1.18127457048702e-09, 1.0434267116911005e-10, 7.782263439905071e-12, -3.696805618642206e-12,
            5.100370287454476e-13, -2.0583260535665066e-14, -5.348122539423018e-15, 1.2267786282382608e-15,
            -1.1812593016974588e-16, 1.1866922547516004e-18,
        ];

        /// <summary>
        /// Computes the Bessel function of the first kind, <c>J_ν(x)</c>.
        /// <para>第 1 種ベッセル関数 <c>J_ν(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="nu">The real order <c>ν</c>. / 実数の次数 <c>ν</c>。</param>
        /// <param name="x">The input value (<c>x ≥ 0</c> when <paramref name="nu"/> is not an integer). / 入力値（<paramref name="nu"/> が整数でない場合は <c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>J_ν(x)</c>, or <see cref="double.NaN"/> when the result is not real or either argument is invalid. / <c>J_ν(x)</c> の値。値が実数にならない場合や引数が無効な場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/10.2.E2"/>
        public static double BesselJ(double nu, double x)
        {
            if (double.IsNaN(nu) || double.IsNaN(x) || double.IsInfinity(nu))
            {
                return double.NaN;
            }

            if (x < 0)
            {
                if (!IsInteger(nu))
                {
                    return double.NaN;
                }

                double value = BesselJ(nu, -x);
                return IsOddInteger(nu) ? -value : value;
            }

            if (double.IsPositiveInfinity(x))
            {
                return 0;
            }

            if (x == 0)
            {
                if (nu == 0)
                {
                    return 1;
                }

                if (nu > 0 || IsInteger(nu))
                {
                    return 0;
                }

                return Math.Sign(ReciprocalGamma(nu + 1)) * double.PositiveInfinity;
            }

            // 次数 0 の小引数では原点級数のみで計算でき、Temme の Y の計算は不要
            if (nu == 0 && x < 2)
            {
                return x <= BesselSmallArgument ? BesselSmallSeries(0, x, -1) : BesselJSeries(0, x);
            }

            double order = Math.Abs(nu);
            BesselJYCore(order, x, out double j, out double y);
            if (nu >= 0)
            {
                return j;
            }

            // J₋ᵥ = cos(νπ) Jᵥ - sin(νπ) Yᵥ
            return CombineTerms(double.CosPi(order), j, -double.SinPi(order), y);
        }

        /// <summary>
        /// Computes the Bessel function of the second kind (Neumann function), <c>Y_ν(x)</c>.
        /// <para>第 2 種ベッセル関数（ノイマン関数）<c>Y_ν(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="nu">The real order <c>ν</c>. / 実数の次数 <c>ν</c>。</param>
        /// <param name="x">The non-negative input value (<c>x ≥ 0</c>). / 非負の入力値（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>Y_ν(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is negative or either argument is invalid. / <c>Y_ν(x)</c> の値。<paramref name="x"/> が負または引数が無効な場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/10.2.E3"/>
        public static double BesselY(double nu, double x)
        {
            if (double.IsNaN(nu) || double.IsNaN(x) || double.IsInfinity(nu) || x < 0)
            {
                return double.NaN;
            }

            if (double.IsPositiveInfinity(x))
            {
                return 0;
            }

            double order = Math.Abs(nu);
            if (x == 0)
            {
                if (nu >= 0)
                {
                    return double.NegativeInfinity;
                }

                // Y₋ᵥ = sin(νπ) Jᵥ + cos(νπ) Yᵥ, Jᵥ(0) = 0, Yᵥ(0) = -∞
                double cos = double.CosPi(order);
                return cos == 0 ? 0 : -Math.Sign(cos) * double.PositiveInfinity;
            }

            if (nu == 0 && x < 2)
            {
                TemmeSeriesY(0, x, out double sum, out _);
                return -sum;
            }

            BesselJYCore(order, x, out double j, out double y);
            if (nu >= 0)
            {
                return y;
            }

            return CombineTerms(double.SinPi(order), j, double.CosPi(order), y);
        }

        /// <summary>
        /// Computes the modified Bessel function of the first kind, <c>I_ν(x)</c>.
        /// <para>第 1 種変形ベッセル関数 <c>I_ν(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="nu">The real order <c>ν</c>. / 実数の次数 <c>ν</c>。</param>
        /// <param name="x">The input value (<c>x ≥ 0</c> when <paramref name="nu"/> is not an integer). / 入力値（<paramref name="nu"/> が整数でない場合は <c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>I_ν(x)</c>, or <see cref="double.NaN"/> when the result is not real or either argument is invalid. / <c>I_ν(x)</c> の値。値が実数にならない場合や引数が無効な場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/10.25.E2"/>
        public static double BesselI(double nu, double x)
        {
            if (double.IsNaN(nu) || double.IsNaN(x) || double.IsInfinity(nu))
            {
                return double.NaN;
            }

            if (x < 0)
            {
                if (!IsInteger(nu))
                {
                    return double.NaN;
                }

                double value = BesselI(nu, -x);
                return IsOddInteger(nu) ? -value : value;
            }

            if (double.IsPositiveInfinity(x))
            {
                return x;
            }

            if (x == 0)
            {
                if (nu == 0)
                {
                    return 1;
                }

                if (nu > 0 || IsInteger(nu))
                {
                    return 0;
                }

                // I₋ᵥ(0) = Iᵥ(0) + (2/π) sin(νπ) Kᵥ(0)
                return Math.Sign(double.SinPi(-nu)) * double.PositiveInfinity;
            }

            double scaled = BesselIScaledPositive(nu, x, out long exponent);
            if (exponent == 0 && x <= 700 && Math.Abs(scaled) >= MinNormal)
            {
                return scaled * Math.Exp(x);
            }

            // 高次では e⁻ˣ I が倍精度の範囲を外れても I 自体は有限になりうる (x > 709 では eˣ もオーバーフローする)。
            // eˣ を仮数と 2 の冪に分離して掛け、最後に 1 回だけ丸める (仮数が 0 や ∞ なのは指数部が 1e18 を超えた場合)
            if (scaled == 0 || !double.IsFinite(scaled))
            {
                return scaled;
            }

            double exp = ConfluentScaledExp(x, 0, out long expExponent);
            return ScaleB(scaled * exp, exponent + expExponent);
        }

        /// <summary>
        /// Computes the modified Bessel function of the second kind, <c>K_ν(x)</c>.
        /// <para>第 2 種変形ベッセル関数 <c>K_ν(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="nu">The real order <c>ν</c>. / 実数の次数 <c>ν</c>。</param>
        /// <param name="x">The non-negative input value (<c>x ≥ 0</c>). / 非負の入力値（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>K_ν(x)</c>, <see cref="double.PositiveInfinity"/> at <c>x = 0</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is negative or either argument is invalid. / <c>K_ν(x)</c> の値。<c>x = 0</c> では <see cref="double.PositiveInfinity"/>、<paramref name="x"/> が負または引数が無効な場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/10.27.E4"/>
        public static double BesselK(double nu, double x)
        {
            if (double.IsNaN(nu) || double.IsNaN(x) || double.IsInfinity(nu) || x < 0)
            {
                return double.NaN;
            }

            if (x == 0)
            {
                return double.PositiveInfinity;
            }

            if (double.IsPositiveInfinity(x))
            {
                return 0;
            }

            // 次数 0 では K_μ からの漸化式や I の CF1・正規化が不要
            if (nu == 0 && x > BesselSmallArgument && x < 25)
            {
                BesselKTemmeScaled(0, x, out double k0, out _);
                return k0 * Math.Exp(-x);
            }

            BesselIKCore(Math.Abs(nu), x, out _, out _, out double kScaled, out long exponent);
            if (exponent == 0 && x <= 700)
            {
                return kScaled * Math.Exp(-x);
            }

            // 高次では eˣ K が倍精度の範囲を外れても K 自体は有限になりうる (x > 708 では e⁻ˣ が非正規化数または 0 になる)。
            // e⁻ˣ を仮数と 2 の冪に分離して掛け、最後に 1 回だけ丸める
            if (kScaled == 0 || !double.IsFinite(kScaled))
            {
                return kScaled;
            }

            double exp = ConfluentScaledExp(-x, 0, out long expExponent);
            return ScaleB(kScaled * exp, exponent + expExponent);
        }

        /// <summary>
        /// Computes the exponentially scaled modified Bessel function of the first kind, <c>e^(-|x|) I_ν(x)</c>.
        /// <para>指数スケーリングした第 1 種変形ベッセル関数 <c>e^(-|x|) I_ν(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="nu">The real order <c>ν</c>. / 実数の次数 <c>ν</c>。</param>
        /// <param name="x">The input value (<c>x ≥ 0</c> when <paramref name="nu"/> is not an integer). / 入力値（<paramref name="nu"/> が整数でない場合は <c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>e^(-|x|) I_ν(x)</c>, or <see cref="double.NaN"/> when the result is not real or either argument is invalid. / <c>e^(-|x|) I_ν(x)</c> の値。値が実数にならない場合や引数が無効な場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/10.25.E2"/>
        public static double BesselIScaled(double nu, double x)
        {
            if (double.IsNaN(nu) || double.IsNaN(x) || double.IsInfinity(nu))
            {
                return double.NaN;
            }

            if (x < 0)
            {
                if (!IsInteger(nu))
                {
                    return double.NaN;
                }

                double value = BesselIScaled(nu, -x);
                return IsOddInteger(nu) ? -value : value;
            }

            if (double.IsPositiveInfinity(x))
            {
                return 0;
            }

            if (x == 0)
            {
                return BesselI(nu, x);
            }

            return BesselIScaledPositive(nu, x);
        }

        /// <summary>
        /// Computes the exponentially scaled modified Bessel function of the second kind, <c>eˣ K_ν(x)</c>.
        /// <para>指数スケーリングした第 2 種変形ベッセル関数 <c>eˣ K_ν(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="nu">The real order <c>ν</c>. / 実数の次数 <c>ν</c>。</param>
        /// <param name="x">The non-negative input value (<c>x ≥ 0</c>). / 非負の入力値（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>eˣ K_ν(x)</c>, <see cref="double.PositiveInfinity"/> at <c>x = 0</c>, or <see cref="double.NaN"/> if <paramref name="x"/> is negative or either argument is invalid. / <c>eˣ K_ν(x)</c> の値。<c>x = 0</c> では <see cref="double.PositiveInfinity"/>、<paramref name="x"/> が負または引数が無効な場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/10.27.E4"/>
        public static double BesselKScaled(double nu, double x)
        {
            if (double.IsNaN(nu) || double.IsNaN(x) || double.IsInfinity(nu) || x < 0)
            {
                return double.NaN;
            }

            if (x == 0)
            {
                return double.PositiveInfinity;
            }

            if (double.IsPositiveInfinity(x))
            {
                return 0;
            }

            if (nu == 0 && x > BesselSmallArgument && x < 25)
            {
                BesselKTemmeScaled(0, x, out double k0, out _);
                return k0;
            }

            BesselIKCore(Math.Abs(nu), x, out _, out _, out double kScaled, out long exponent);
            return ScaleB(kScaled, exponent);
        }

        /// <summary>
        /// Computes the Hankel function of the first kind, <c>H⁽¹⁾_ν(x) = J_ν(x) + i Y_ν(x)</c>.
        /// <para>第 1 種ハンケル関数 <c>H⁽¹⁾_ν(x) = J_ν(x) + i Y_ν(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="nu">The real order <c>ν</c>. / 実数の次数 <c>ν</c>。</param>
        /// <param name="x">The positive input value (<c>x &gt; 0</c>). / 正の入力値（<c>x &gt; 0</c>）。</param>
        /// <returns>The complex value of <c>H⁽¹⁾_ν(x)</c>. / <c>H⁽¹⁾_ν(x)</c> の複素数値。</returns>
        /// <seealso href="https://dlmf.nist.gov/10.4.E3"/>
        public static Complex HankelH1(double nu, double x) => new(BesselJ(nu, x), BesselY(nu, x));

        /// <summary>
        /// Computes the Hankel function of the second kind, <c>H⁽²⁾_ν(x) = J_ν(x) - i Y_ν(x)</c>.
        /// <para>第 2 種ハンケル関数 <c>H⁽²⁾_ν(x) = J_ν(x) - i Y_ν(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="nu">The real order <c>ν</c>. / 実数の次数 <c>ν</c>。</param>
        /// <param name="x">The positive input value (<c>x &gt; 0</c>). / 正の入力値（<c>x &gt; 0</c>）。</param>
        /// <returns>The complex value of <c>H⁽²⁾_ν(x)</c>. / <c>H⁽²⁾_ν(x)</c> の複素数値。</returns>
        /// <seealso href="https://dlmf.nist.gov/10.4.E3"/>
        public static Complex HankelH2(double nu, double x) => new(BesselJ(nu, x), -BesselY(nu, x));

        // e⁻ˣ I_ν(x) (x > 0, 任意の実数 ν)
        private static double BesselIScaledPositive(double nu, double x) => ScaleB(BesselIScaledPositive(nu, x, out long exponent), exponent);

        // e⁻ˣ I_ν(x) = 戻り値 · 2^exponent (x > 0, 任意の実数 ν)
        private static double BesselIScaledPositive(double nu, double x, out long exponent)
        {
            double order = Math.Abs(nu);
            BesselIKCore(order, x, out double iScaled, out long iExponent, out double kScaled, out long kExponent);
            if (nu >= 0)
            {
                exponent = iExponent;
                return iScaled;
            }

            // I₋ᵥ = Iᵥ + (2/π) sin(νπ) Kᵥ
            double sin = double.SinPi(order);
            if (iExponent == 0 && kExponent == 0)
            {
                exponent = 0;
                return CombineTerms(1, iScaled, 2 / Math.PI * sin, kScaled * Math.Exp(-2 * x));
            }

            // 2 項の指数が離れている場合は e^(-2x) も仮数と 2 の冪に分離し、大きい方の指数に揃えて加算する
            double exp = ConfluentScaledExp(-2 * x, 0, out long expExponent);
            kExponent += expExponent;
            exponent = Math.Max(iExponent, kExponent);
            return CombineTerms(1, ScaleB(iScaled, iExponent - exponent), 2 / Math.PI * sin, ScaleB(kScaled * exp, kExponent - exponent));
        }

        // |μ| ≤ 1/2 に対する Temme の補助量:
        // gam1 = (1/Γ(1-μ) - 1/Γ(1+μ)) / (2μ), gam2 = (1/Γ(1-μ) + 1/Γ(1+μ)) / 2, gammaPlus = 1/Γ(1+μ), gammaMinus = 1/Γ(1-μ)
        private static void TemmeGammas(double mu, out double gam1, out double gam2, out double gammaPlus, out double gammaMinus)
        {
            double mu2 = mu * mu;
            double even = 0;
            double odd = 0;
            int last = ReciprocalGammaTaylor.Length - 1;
            for (int k = last - last % 2; k >= 0; k -= 2)
            {
                even = even * mu2 + ReciprocalGammaTaylor[k];
            }

            for (int k = last - (1 - last % 2); k >= 1; k -= 2)
            {
                odd = odd * mu2 + ReciprocalGammaTaylor[k];
            }

            gam1 = -odd;
            gam2 = even;
            gammaPlus = even + mu * odd;
            gammaMinus = even - mu * odd;
        }

        // J_ν(x) と Y_ν(x) (ν ≥ 0, x > 0)。
        // Steed 法・Temme 法 (Numerical Recipes の bessjy) に大引数のハンケル漸近展開を組み合わせる (x < 2 の J_ν は原点級数で直接計算)
        private static void BesselJYCore(double nu, double x, out double j, out double y)
        {
            if (x >= Math.Max(25, nu * nu))
            {
                BesselJYAsymptotic(nu, x, out j, out y);
                return;
            }

            if ((nu >= BesselLargeOrderMin || (nu >= DebyeMinOrder && x > nu))
                && TryBesselJYLargeOrder(nu, x, out j, out y))
            {
                return;
            }

            if (x <= BesselSmallArgument)
            {
                j = BesselSmallSeries(nu, x, -1);
                y = BesselSmallSecondKind(nu, x, modified: false);
                return;
            }

            int nl = x < 2 ? (int)(nu + 0.5) : Math.Max(0, (int)(nu - x + 1.5));
            double xmu = nu - nl;
            double xi = 1 / x;
            double xi2 = 2 * xi;
            double rymu;
            double ry1;
            if (x < 2)
            {
                // ロンスキアン Jᵤ Y'ᵤ - J'ᵤ Yᵤ = 2/(πx) から Jᵤ を求めると、μ < 0 では Y'ᵤ - (J'ᵤ/Jᵤ) Yᵤ の2項が打ち消し合い、
                // 入力の条件数（約 ν）を大きく超える相対誤差になる。
                // x < 2 には零点がなく（j_ν,₁ ≥ j₀,₁ ≈ 2.405）、原点の級数の打ち消しは小さいので、級数で直接求める
                j = BesselJSeries(nu, x);
                TemmeSeriesY(xmu, x, out double sum, out double sum1);
                rymu = -sum;
                ry1 = -sum1 * xi2;
            }
            else
            {
                BesselJYSteed(nu, x, nl, xmu, out j, out rymu, out ry1);
            }

            // Yᵤ から Yᵥ まで漸化式を上る。x より大きな次数で |Y| は単調に増えるので、無限大になったらその符号で打ち切る
            // （続けると次の段が -∞ - (-∞) = NaN になる）
            for (int i = 1; i <= nl; i++)
            {
                if (double.IsInfinity(ry1))
                {
                    y = ry1;
                    return;
                }

                double rytemp = (xmu + i) * xi2 * ry1 - rymu;
                rymu = ry1;
                ry1 = rytemp;
            }

            y = rymu;
        }

        // x ≥ 2 での J_ν(x), Y_μ(x), Y_(μ+1)(x) (μ = ν - nl, nl = max(0, ⌊ν - x + 3/2⌋))。
        // CF1 で J'ᵥ/Jᵥ を求めて μ まで漸化式を下り、CF2 (Steed 法) の (J'ᵤ + iY'ᵤ)/(Jᵤ + iYᵤ) とロンスキアンで正規化する。
        // 次数を μ ≲ x まで下げるだけのため、ν < x - 1/2 では μ = ν となり |μ| ≤ 1/2 とは限らない
        private static void BesselJYSteed(double nu, double x, int nl, double xmu, out double j, out double rymu, out double ry1)
        {
            double xmu2 = xmu * xmu;
            double xi = 1 / x;
            double xi2 = 2 * xi;
            double w = xi2 / Math.PI;

            // CF1: f = J'ᵥ/Jᵥ
            int isign = 1;
            double h = nu * xi;
            if (h < BesselTiny)
            {
                h = BesselTiny;
            }

            // 係数は反復ごとに直接計算する (b += 2/x のような累積加算では数千回の反復で丸め誤差が蓄積する)
            double d = 0;
            double c = h;
            for (int i = 1; i <= BesselMaxIterations; i++)
            {
                double b = 2 * (nu + i) / x;
                d = b - d;
                if (Math.Abs(d) < BesselTiny)
                {
                    d = BesselTiny;
                }

                c = b - 1 / c;
                if (Math.Abs(c) < BesselTiny)
                {
                    c = BesselTiny;
                }

                d = 1 / d;
                double delta = c * d;
                h = delta * h;
                if (d < 0)
                {
                    isign = -isign;
                }

                if (Math.Abs(delta - 1) <= Epsilon)
                {
                    break;
                }
            }

            // 次数 ν から μ = ν - nl まで漸化式を下る (オーバーフロー回避のため適宜スケーリング)。
            // J_{μ+l-1} = ((μ+l)/x) J_{μ+l} + J'_{μ+l}, J'_{μ+l-1} = ((μ+l-1)/x) J_{μ+l-1} - J_{μ+l}
            double rjl = isign * BesselTiny;
            double rjpl = h * rjl;
            double rjl1 = rjl;
            for (int l = nl; l >= 1; l--)
            {
                double rjtemp = (xmu + l) / x * rjl + rjpl;
                rjpl = (xmu + l - 1) / x * rjtemp - rjl;
                rjl = rjtemp;
                if (Math.Abs(rjl) > 1e250)
                {
                    rjl *= 1e-250;
                    rjpl *= 1e-250;
                    rjl1 *= 1e-250;
                }
            }

            if (rjl == 0)
            {
                rjl = Epsilon;
            }

            double f = rjpl / rjl;

            // CF2 (Steed 法): p + iq = (J'ᵤ + iY'ᵤ)/(Jᵤ + iYᵤ)。係数 aᵢ = 1/4 - μ² + i(i-1) も直接計算する
            double a1 = 0.25 - xmu2;
            double a = a1;
            double p = -0.5 * xi;
            double q = 1;
            double br = 2 * x;
            double bi = 2;
            double fact = a * xi / (p * p + q * q);
            double cr = br + q * fact;
            double ci = bi + p * fact;
            double den = br * br + bi * bi;
            double dr = br / den;
            double di = -bi / den;
            double dlr = cr * dr - ci * di;
            double dli = cr * di + ci * dr;
            double temp = p * dlr - q * dli;
            q = p * dli + q * dlr;
            p = temp;
            for (int i = 2; i <= BesselMaxIterations; i++)
            {
                a = a1 + (double)i * (i - 1);
                bi += 2;
                dr = a * dr + br;
                di = a * di + bi;
                if (Math.Abs(dr) + Math.Abs(di) < BesselTiny)
                {
                    dr = BesselTiny;
                }

                fact = a / (cr * cr + ci * ci);
                cr = br + cr * fact;
                ci = bi - ci * fact;
                if (Math.Abs(cr) + Math.Abs(ci) < BesselTiny)
                {
                    cr = BesselTiny;
                }

                den = dr * dr + di * di;
                dr /= den;
                di /= -den;
                dlr = cr * dr - ci * di;
                dli = cr * di + ci * dr;
                temp = p * dlr - q * dli;
                q = p * dli + q * dlr;
                p = temp;
                if (Math.Abs(dlr - 1) + Math.Abs(dli) <= Epsilon)
                {
                    break;
                }
            }

            double gam = (p - f) / q;
            double rjmu = Math.Sqrt(w / ((p - f) * gam + q));
            rjmu = Math.CopySign(rjmu, rjl);
            rymu = rjmu * gam;
            double rymup = rymu * (p + q / gam);
            ry1 = xmu * xi * rymu - rymup;
            j = rjl1 * (rjmu / rjl);
        }

        // 原点級数の先頭 2 項 (x/2)^ν/Γ(ν+1) · (1 + sign (x/2)²/(ν+1))。
        // sign = -1 で J_ν, +1 で I_ν, 0 で先頭係数のみ (ν ≥ 0, sign ≠ 0 では 0 < x ≤ BesselSmallArgument)
        private static double BesselSmallSeries(double nu, double x, int sign)
        {
            double t = 0.25 * x * x;
            double correction = 1 + sign * t / (nu + 1);
            if (nu == 0)
            {
                return correction;
            }

            // x/2 は非正規化数で下位ビットを失うため、その範囲では (x/2)^ν = x^ν 2^(-ν) とする
            double power = x >= 4.450147717014403e-308 ? Math.Pow(0.5 * x, nu) : Math.Pow(x, nu) * Math.Pow(2, -nu);
            return power * ReciprocalGamma(nu + 1) * correction;
        }

        // 原点級数 I_ν(x) = (x/2)^ν Σ (x²/4)ᵏ/(k! Γ(ν+k+1)) (ν ≥ 0, 0 < x < 2; 全項正で桁落ちなし)
        private static double BesselISeries(double nu, double x) => BesselSmallSeries(nu, x, 0) * BesselSeriesSum(nu, x, 1);

        // 原点級数 J_ν(x) = (x/2)^ν Σ (-x²/4)ᵏ/(k! Γ(ν+k+1)) (ν ≥ 0, 0 < x < 2; 条件数は最大 I₀(2)/J₀(2) ≈ 10.2)
        private static double BesselJSeries(double nu, double x) => BesselSmallSeries(nu, x, 0) * BesselSeriesSum(nu, x, -1);

        // 原点級数の和 Σ (sign·x²/4)ᵏ/(k! (ν+1)ₖ) (0 < x < 2)。
        // sign = 1 (I_ν, ν > -1) では全項正、sign = -1 (J_ν, ν ≥ 0) では絶対値単調減少の交代級数
        private static double BesselSeriesSum(double nu, double x, int sign)
        {
            double t = sign * 0.25 * x * x;
            double term = 1;
            double sum = 1;
            for (int k = 1; k < 1000; k++)
            {
                term *= t / (k * (nu + k));
                sum += term;
                if (Math.Abs(term) <= sum * Epsilon * 0.5)
                {
                    break;
                }
            }

            return sum;
        }

        // 小引数 (ν ≥ 0, 0 < x ≤ BesselSmallArgument) での Y_ν(x) (modified = true なら K_ν(x))。
        // |μ| ≤ 1/2 を Temme 級数で求め、漸化式で次数を上げる
        private static double BesselSmallSecondKind(double nu, double x, bool modified)
        {
            int nl = (int)(nu + 0.5);
            double xmu = nu - nl;
            double current;
            double next;
            if (modified)
            {
                // K_μ = Σf、K_μ₊₁ = (2/x) Σh
                TemmeSeriesK(xmu, x, out double sum, out double sum1);
                current = sum;
                next = 2 * sum1 / x;
            }
            else
            {
                // Y_μ = -Σf、Y_μ₊₁ = -(2/x) Σh
                TemmeSeriesY(xmu, x, out double sum, out double sum1);
                current = -sum;
                next = -2 * sum1 / x;
            }

            // Y_{k+1} = (2(μ+k)/x) Y_k - Y_{k-1}, K_{k+1} = (2(μ+k)/x) K_k + K_{k-1}。無限大になったらその符号で打ち切る
            for (int i = 1; i <= nl; i++)
            {
                if (double.IsInfinity(next))
                {
                    return next;
                }

                double factor = 2 * (xmu + i) / x;
                double following = modified ? factor * next + current : factor * next - current;
                current = next;
                next = following;
            }

            return current;
        }

        // Temme 級数による Y_μ(x) = -sum, Y_(μ+1)(x) = -(2/x) sum1 (|μ| ≤ 1/2, 0 < x < 2; 1/x を含まないため非正規化数でもオーバーフローしない)
        private static void TemmeSeriesY(double xmu, double x, out double sum, out double sum1)
        {
            double pimu = Math.PI * xmu;
            double fact = Math.Abs(pimu) < Epsilon ? 1 : pimu / Math.Sin(pimu);

            // d = -ln(x/2)。x/2 を作らないことで非正規化数での丸めを避ける
            double d = Ln2 - Math.Log(x);
            double e = xmu * d;
            double fact2 = Math.Abs(e) < Epsilon ? 1 : Math.Sinh(e) / e;
            TemmeGammas(xmu, out double gam1, out double gam2, out double gampl, out double gammi);
            double ff = 2 / Math.PI * fact * (gam1 * Math.Cosh(e) + gam2 * fact2 * d);
            e = Math.Exp(e);
            double p = e / (gampl * Math.PI);
            double q = 1 / (e * Math.PI * gammi);
            double pimu2 = 0.5 * pimu;
            double fact3 = Math.Abs(pimu2) < Epsilon ? 1 : Math.Sin(pimu2) / pimu2;
            double r = Math.PI * pimu2 * fact3 * fact3;
            double x2 = 0.5 * x;
            double c = 1;
            double xmu2 = xmu * xmu;
            sum = ff + r * q;
            sum1 = p;
            for (int i = 1; i <= BesselMaxIterations; i++)
            {
                ff = (i * ff + p + q) / ((double)i * i - xmu2);
                c *= -x2 * x2 / i;
                p /= i - xmu;
                q /= i + xmu;
                double delta = c * (ff + r * q);
                sum += delta;
                double delta1 = c * p - i * delta;
                sum1 += delta1;

                // |μ| = 1/2 付近では sum1 の収束が sum より遅いため両方を判定する
                if (Math.Abs(delta) <= (1 + Math.Abs(sum)) * Epsilon && Math.Abs(delta1) <= Math.Abs(sum1) * Epsilon)
                {
                    break;
                }
            }
        }

        // Temme 級数による K_μ(x) = sum, K_(μ+1)(x) = (2/x) sum1 (|μ| ≤ 1/2, 0 < x < 2; 1/x を含まないため非正規化数でもオーバーフローしない)
        private static void TemmeSeriesK(double xmu, double x, out double sum, out double sum1)
        {
            double pimu = Math.PI * xmu;
            double fact = Math.Abs(pimu) < Epsilon ? 1 : pimu / Math.Sin(pimu);
            double d = Ln2 - Math.Log(x);
            double e = xmu * d;
            double fact2 = Math.Abs(e) < Epsilon ? 1 : Math.Sinh(e) / e;
            TemmeGammas(xmu, out double gam1, out double gam2, out double gampl, out double gammi);
            double ff = fact * (gam1 * Math.Cosh(e) + gam2 * fact2 * d);
            e = Math.Exp(e);
            double p = 0.5 * e / gampl;
            double q = 0.5 / (e * gammi);
            double x2 = 0.5 * x;
            double c = 1;
            double xmu2 = xmu * xmu;
            sum = ff;
            sum1 = p;
            for (int i = 1; i <= BesselMaxIterations; i++)
            {
                ff = (i * ff + p + q) / ((double)i * i - xmu2);
                c *= x2 * x2 / i;
                p /= i - xmu;
                q /= i + xmu;
                double delta = c * ff;
                sum += delta;
                double delta1 = c * (p - i * ff);
                sum1 += delta1;

                // |μ| = 1/2 付近では sum1 の項の相対的な大きさが O(x) で、sum より遅く収束するので両方を判定する
                if (Math.Abs(delta) <= Math.Abs(sum) * Epsilon && Math.Abs(delta1) <= Math.Abs(sum1) * Epsilon)
                {
                    break;
                }
            }
        }

        // ハンケルの漸近展開による J_ν(x), Y_ν(x) (x ≥ max(25, ν²))
        private static void BesselJYAsymptotic(double nu, double x, out double j, out double y)
        {
            BesselAsymptoticSeries(nu, x, alternating: true, out double p, out double q);

            // χ = x - (ν/2 + 1/4)π の三角関数を、x の引数還元を保ったまま加法定理で求める
            BesselHankelPhase(nu, x, out double cosChi, out double sinChi);

            // πx は x が大きいとオーバーフローするため、√(2/π) と √x に分離する
            double amplitude = Math.Sqrt(2 / Math.PI) / Math.Sqrt(x);
            j = amplitude * (p * cosChi - q * sinChi);
            y = amplitude * (p * sinChi + q * cosChi);
        }

        // 漸近展開の係数 aₖ(ν)/xᵏ = (4ν²-1²)(4ν²-3²)…(4ν²-(2k-1)²) / (k! (8x)ᵏ) の和。
        // alternating = true なら P = Σ (-1)ᵏ a₂ₖ/x²ᵏ, Q = Σ (-1)ᵏ a₂ₖ₊₁/x²ᵏ⁺¹、false なら P = Σ (-1)ᵏ aₖ/xᵏ, Q = Σ aₖ/xᵏ
        private static void BesselAsymptoticSeries(double nu, double x, bool alternating, out double p, out double q)
        {
            double mu = 4 * nu * nu;
            double eightX = 8 * x;
            double term = 1;
            double previous = double.MaxValue;
            p = 1;
            q = alternating ? 0 : 1;
            for (int k = 1; k < 1000; k++)
            {
                double odd = 2.0 * k - 1;
                term *= (mu - odd * odd) / (k * eightX);
                double magnitude = Math.Abs(term);
                if (magnitude > previous)
                {
                    break;
                }

                if (alternating)
                {
                    if (k % 2 == 1)
                    {
                        q += ((k - 1) / 2) % 2 == 0 ? term : -term;
                    }
                    else
                    {
                        p += (k / 2) % 2 == 0 ? term : -term;
                    }
                }
                else
                {
                    p += k % 2 == 0 ? term : -term;
                    q += term;
                }

                if (magnitude <= Epsilon * 0.25 || term == 0)
                {
                    break;
                }

                previous = magnitude;
            }
        }

        // e⁻ˣ I_ν(x) と eˣ K_ν(x) (ν ≥ 0, x > 0)。Temme 法・Steed 法 (Numerical Recipes の bessik) に大引数の漸近展開を組み合わせる
        private static void BesselIKCore(double nu, double x, out double iScaled, out double kScaled)
        {
            BesselIKCore(nu, x, out iScaled, out long iExponent, out kScaled, out long kExponent);
            iScaled = ScaleB(iScaled, iExponent);
            kScaled = ScaleB(kScaled, kExponent);
        }

        // e⁻ˣ I_ν(x) = iScaled · 2^iExponent, eˣ K_ν(x) = kScaled · 2^kExponent (ν ≥ 0, x > 0)。
        // 高次ではスケーリング値が倍精度を外れても I, K 自体は有限になりうるため、スケーリング分を 2 の冪指数として分離する
        private static void BesselIKCore(double nu, double x, out double iScaled, out long iExponent, out double kScaled, out long kExponent)
        {
            iExponent = 0;
            kExponent = 0;
            if (x >= Math.Max(25, nu * nu))
            {
                // 2πx は x が大きいとオーバーフローするため、定数と √x に分離する
                BesselAsymptoticSeries(nu, x, alternating: false, out double alternatingSum, out double sum);
                double rootX = Math.Sqrt(x);
                iScaled = alternatingSum / Math.Sqrt(2 * Math.PI) / rootX;
                kScaled = Math.Sqrt(Math.PI / 2) / rootX * sum;
                return;
            }

            if (nu >= BesselLargeOrderMin)
            {
                BesselIKDebye(nu, x, out iScaled, out iExponent, out kScaled, out kExponent);
                BesselFoldExponent(ref iScaled, ref iExponent);
                BesselFoldExponent(ref kScaled, ref kExponent);
                return;
            }

            if (x <= BesselSmallArgument)
            {
                iScaled = BesselSmallSeries(nu, x, 1) * Math.Exp(-x);
                kScaled = BesselSmallSecondKind(nu, x, modified: true) * Math.Exp(x);
                return;
            }

            int nl = (int)(nu + 0.5);
            double xmu = nu - nl;
            double xi = 1 / x;
            double xi2 = 2 * xi;

            // CF1: f = I'ᵥ/Iᵥ
            double h = nu * xi;
            if (h < BesselTiny)
            {
                h = BesselTiny;
            }

            // 係数は反復ごとに直接計算する (累積加算による丸め誤差回避; BesselJYSteed と同様)
            double d = 0;
            double c = h;
            for (int i = 1; i <= BesselMaxIterations; i++)
            {
                double b = 2 * (nu + i) / x;
                d = 1 / (b + d);
                c = b + 1 / c;
                double delta = c * d;
                h = delta * h;
                if (Math.Abs(delta - 1) <= Epsilon)
                {
                    break;
                }
            }

            // 次数 ν から μ まで漸化式を下る (オーバーフロー回避のため 2 の冪で縮小し、縮小分を iShift に記録する)
            double ril = BesselTiny;
            double ripl = h * ril;
            long iShift = 0;
            for (int l = nl; l >= 1; l--)
            {
                double ritemp = (xmu + l) / x * ril + ripl;
                ripl = (xmu + l - 1) / x * ritemp + ril;
                ril = ritemp;
                if (ril > 1e250)
                {
                    ril = Math.ScaleB(ril, -BesselRescaleShift);
                    ripl = Math.ScaleB(ripl, -BesselRescaleShift);
                    iShift += BesselRescaleShift;
                }
            }

            double f = ripl / ril;
            BesselKTemmeScaled(xmu, x, out double rkmu, out double rk1);

            if (x < 2)
            {
                // ロンスキアンでは μ/x の項同士が桁落ちし |μ| = 1/2 付近で相対誤差が O(ε/x) になるため、全項正の原点級数で直接計算する
                iScaled = BesselISeries(nu, x) * Math.Exp(-x);
            }
            else
            {
                // ロンスキアン Iᵤ K'ᵤ - I'ᵤ Kᵤ = -1/x から Iᵤ を求め、Iᵥ/Iᵤ = BesselTiny/(ril 2^iShift) を掛ける。
                // アンダーフローを避けるため BesselTiny と ril の 2 の冪を指数へ移す
                double rkmup = xmu * xi * rkmu - rk1;
                double rimu = xi / (f * rkmu - rkmup);
                int tinyExponent = Math.ILogB(BesselTiny);
                int rilExponent = Math.ILogB(ril);
                iScaled = rimu * Math.ScaleB(BesselTiny, -tinyExponent) / Math.ScaleB(ril, -rilExponent);
                iExponent = tinyExponent - rilExponent - iShift;
            }

            // Kᵤ から Kᵥ まで漸化式を上る (オーバーフロー回避のため 2 の冪で縮小し、縮小分を kExponent に記録する)
            for (int i = 1; i <= nl; i++)
            {
                double rktemp = (xmu + i) * xi2 * rk1 + rkmu;
                rkmu = rk1;
                rk1 = rktemp;
                if (rk1 > 1e250)
                {
                    rkmu = Math.ScaleB(rkmu, -BesselRescaleShift);
                    rk1 = Math.ScaleB(rk1, -BesselRescaleShift);
                    kExponent += BesselRescaleShift;
                }
            }

            kScaled = rkmu;
            BesselFoldExponent(ref iScaled, ref iExponent);
            BesselFoldExponent(ref kScaled, ref kExponent);
        }

        // Temme 級数または CF2 で eˣK_μ(x), eˣK_(μ+1)(x) を計算する (|μ| ≤ 1/2, x > 0; I の対数微分に依存しないため K₀ 単独計算でも共用)
        private static void BesselKTemmeScaled(double xmu, double x, out double rkmu, out double rk1)
        {
            if (x < 2)
            {
                TemmeSeriesK(xmu, x, out double sum, out double sum1);
                double expX = Math.Exp(x);
                rkmu = sum * expX;
                rk1 = sum1 * (2 * (1 / x)) * expX;
            }
            else
            {
                // CF2 (Steed 法、Temme による正規化)。係数 aᵢ = -(1/4 - μ²) - i(i-1), bᵢ = 2(x + i) は直接計算する
                double d = 1 / (2 * (1 + x));
                double h = d;
                double delh = d;
                double q1 = 0;
                double q2 = 1;
                double a1 = 0.25 - xmu * xmu;
                double q = a1;
                double c = a1;
                double s = 1 + q * delh;
                for (int i = 2; i <= BesselMaxIterations; i++)
                {
                    double a = -a1 - (double)i * (i - 1);
                    c = -a * c / i;
                    double qnew = (q1 - 2 * (x + (i - 1)) * q2) / a;
                    q1 = q2;
                    q2 = qnew;
                    q += c * qnew;
                    double b = 2 * (x + i);
                    d = 1 / (b + a * d);
                    delh = (b * d - 1) * delh;
                    h += delh;
                    double dels = q * delh;
                    s += dels;
                    if (Math.Abs(dels / s) <= Epsilon)
                    {
                        break;
                    }
                }

                h = a1 * h;
                rkmu = Math.Sqrt(Math.PI / (2 * x)) / s;
                rk1 = rkmu * (xmu + x + 0.5 - h) * (1 / x);
            }
        }

        // mantissa · 2^exponent が正規化数に収まるなら指数を仮数へ戻す (指数が 0 ならそのまま倍精度で e^(±x) を掛けられる)
        private static void BesselFoldExponent(ref double mantissa, ref long exponent)
        {
            if (exponent == 0)
            {
                return;
            }

            double value = ScaleB(mantissa, exponent);
            if (Math.Abs(value) >= MinNormal && double.IsFinite(value))
            {
                mantissa = value;
                exponent = 0;
            }
        }
    }
}
