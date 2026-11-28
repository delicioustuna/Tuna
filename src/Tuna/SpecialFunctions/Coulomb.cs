using System.Numerics;
using static Tuna.XMath.Constants;
using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // ρ ≥ 30 (かつ matching 点以上) では先に H⁺ = G + iF の漸近展開を試し、未収束ならスティード法へ戻す
        private const double CoulombAsymptoticMin = 30;

        // 引力 (η < 0) で小 ρ のときの CF2 の収束劣化・桁落ちを避けるためのスティード法下限係数 (ρ_m ≥ c√|η|)
        private const double CoulombSmallRhoFactor = 0.3;

        // 原点級数をそのまま採用する条件数 (Σ|項|/|Σ項|) の上限
        private const double CoulombSeriesMaxCondition = 4;

        private const int CoulombMaxIterations = 10_000_000;

        // CoulombScaledNormalization で漸化式の積を使う ⌊ℓ⌋ の上限 (超えると対数公式を使用)
        private const int CoulombNormalizationMaxProduct = 10_000;

        // 障壁内の内向き積分で G がこの 2 の指数を超えたらオーバーフロー確定として打ち切る
        private const long CoulombOverflowExponent = 1100;

        private const int CoulombMaxIterationsCf2 = 100_000;

        /// <summary>
        /// Computes the regular Coulomb wave function <c>F_ℓ(η, ρ)</c>.
        /// <para>正則クーロン波動関数 <c>F_ℓ(η, ρ)</c> を計算します。</para>
        /// </summary>
        /// <param name="l">The real angular momentum order <c>ℓ</c> (<c>ℓ ≥ 0</c>). / 実数の角運動量次数 <c>ℓ</c>（<c>ℓ ≥ 0</c>）。</param>
        /// <param name="eta">The real Sommerfeld parameter <c>η</c> (<c>η &gt; 0</c> repulsive, <c>η &lt; 0</c> attractive). / 実数のゾンマーフェルト・パラメータ <c>η</c>（<c>η &gt; 0</c> で斥力、<c>η &lt; 0</c> で引力）。</param>
        /// <param name="rho">The non-negative radial argument <c>ρ</c> (<c>ρ ≥ 0</c>). / 非負の動径変数 <c>ρ</c>（<c>ρ ≥ 0</c>）。</param>
        /// <returns>The value of <c>F_ℓ(η, ρ)</c>, <c>0</c> at <c>ρ = 0</c>, or <see cref="double.NaN"/> if any argument is out of domain or non-finite. / <c>F_ℓ(η, ρ)</c> の値。<c>ρ = 0</c> では <c>0</c>、定義域外または非有限の引数では <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Deep inside the barrier (<c>ρ ≪ ρ_tp</c>), the result may underflow to <c>0</c>. For very large arguments around <c>ρ &gt; 10⁷</c> with <c>η² + ℓ² ≳ ρ</c>, evaluation may fail to converge and return <see cref="double.NaN"/>.</para>
        /// <para>障壁の奥（<c>ρ ≪ ρ_tp</c>）ではアンダーフローして <c>0</c> になる場合があります。<c>ρ &gt; 10⁷</c> かつ <c>η² + ℓ² ≳ ρ</c> のような極端な領域では収束せず <see cref="double.NaN"/> を返すことがあります。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/33.2"/>
        public static double CoulombF(double l, double eta, double rho)
        {
            CoulombWave(l, eta, rho, needF: true, needG: false, out double f, out _);
            return f;
        }

        /// <summary>
        /// Computes the irregular Coulomb wave function <c>G_ℓ(η, ρ)</c>.
        /// <para>非正則クーロン波動関数 <c>G_ℓ(η, ρ)</c> を計算します。</para>
        /// </summary>
        /// <param name="l">The real angular momentum order <c>ℓ</c> (<c>ℓ ≥ 0</c>). / 実数の角運動量次数 <c>ℓ</c>（<c>ℓ ≥ 0</c>）。</param>
        /// <param name="eta">The real Sommerfeld parameter <c>η</c>. / 実数のゾンマーフェルト・パラメータ <c>η</c>。</param>
        /// <param name="rho">The non-negative radial argument <c>ρ</c> (<c>ρ ≥ 0</c>). / 非負の動径変数 <c>ρ</c>（<c>ρ ≥ 0</c>）。</param>
        /// <returns>The value of <c>G_ℓ(η, ρ)</c>, <c>1 / C₀(η)</c> at <c>ℓ = 0, ρ = 0</c>, <see cref="double.PositiveInfinity"/> at <c>ℓ &gt; 0, ρ = 0</c>, or <see cref="double.NaN"/> if any argument is out of domain or non-finite. / <c>G_ℓ(η, ρ)</c> の値。<c>ℓ = 0, ρ = 0</c> では <c>1 / C₀(η)</c>、<c>ℓ &gt; 0, ρ = 0</c> では <see cref="double.PositiveInfinity"/>、定義域外または非有限の引数では <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Deep inside the barrier, the result may overflow to <see cref="double.PositiveInfinity"/>. The domain and convergence limits match <see cref="CoulombF"/>.</para>
        /// <para>障壁の奥ではオーバーフローして <see cref="double.PositiveInfinity"/> になる場合があります。定義域および収束限界は <see cref="CoulombF"/> と同じです。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/33.2"/>
        public static double CoulombG(double l, double eta, double rho)
        {
            CoulombWave(l, eta, rho, needF: false, needG: true, out _, out double g);
            return g;
        }

        // F_ℓ(η, ρ) と G_ℓ(η, ρ) (needF, needG が false の出力は不定)
        private static void CoulombWave(double l, double eta, double rho, bool needF, bool needG, out double f, out double g)
        {
            f = double.NaN;
            g = double.NaN;
            if (double.IsNaN(l) || double.IsNaN(eta) || double.IsNaN(rho) || !(l >= 0) || double.IsInfinity(l)
                || double.IsInfinity(eta) || !(rho >= 0) || double.IsPositiveInfinity(rho))
            {
                return;
            }

            if (rho == 0)
            {
                // F ~ C ρ^(ℓ+1), G ~ ρ^(-ℓ)/((2ℓ+1) C)
                f = 0;
                if (l == 0)
                {
                    double c0 = CoulombScaledNormalization(l, eta, out long c0Exponent);
                    g = ScaleB(1 / c0, -c0Exponent);
                }
                else
                {
                    g = double.PositiveInfinity;
                }

                return;
            }

            if (l == 0 && eta == 0)
            {
                // F₀(0, ρ) = sin ρ, G₀(0, ρ) = cos ρ (CF2 の分子 a₁ = (1 + iη)(iη) が 0 になるため直接返す)
                f = Math.Sin(rho);
                g = Math.Cos(rho);
                return;
            }

            double lambda = l * (l + 1);
            double turning = CoulombTurningPoint(eta, lambda);
            double matching = CoulombMatchingPoint(eta, turning);
            if (!double.IsFinite(matching))
            {
                return;
            }

            if (rho >= matching)
            {
                if (rho >= CoulombAsymptoticMin && CoulombAsymptotic(l, eta, rho, out f, out g))
                {
                    return;
                }

                if (!CoulombSteed(l, eta, rho, out f, out _, out g, out _))
                {
                    f = double.NaN;
                    g = double.NaN;
                }

                return;
            }

            // ρ < ρ_m。F は原点級数の条件数が良好なら直接使用する
            bool seriesUsed = false;
            if (needF)
            {
                double condition = CoulombSeries(l, eta, rho, out double sum, out long sumExponent);
                if (condition <= CoulombSeriesMaxCondition)
                {
                    f = CoulombSeriesValue(l, eta, rho, sum, sumExponent);
                    seriesUsed = true;
                }
            }

            if (!needG && seriesUsed)
            {
                return;
            }

            // G は ρ_m でのスティード法の値から内向きに積分する。障壁内では単調増加するため 2¹¹⁰⁰ 超で打ち切る
            if (!CoulombSteed(l, eta, matching, out _, out _, out double gm, out double gpm))
            {
                return;
            }

            long exponent = 0;
            long stop = rho < turning ? CoulombOverflowExponent : long.MaxValue;
            CoulombIntegrate(lambda, eta, matching, rho, ref gm, ref gpm, ref exponent, stop);
            if (needG)
            {
                g = ScaleB(gm, exponent);
            }

            if (needF && !seriesUsed)
            {
                // ロンスキアン F'G - FG' = 1 と CF1 (f = F'/F) から F = 1/(fG - G')
                if (CoulombContinuedFraction1(l, eta, rho, out double ratio, out _))
                {
                    f = ScaleB(1 / (ratio * gm - gpm), -exponent);
                }
            }
        }

        // 転回点 ρ_tp = η + √(η² + λ) (λ = ℓ(ℓ+1) ≥ 0; η < 0 では桁落ち回避のため有理化形を使う)
        private static double CoulombTurningPoint(double eta, double lambda)
        {
            double root = Math.Sqrt(eta * eta + lambda);
            if (eta >= 0)
            {
                return eta + root;
            }

            return lambda == 0 ? 0 : lambda / (root - eta);
        }

        // スティード法・漸近展開の適用下限 ρ_m = max(ρ_tp, 0.3√|η|, 1) (小 ρ での CF2 の収束劣化を回避)
        private static double CoulombMatchingPoint(double eta, double turning) =>
            Math.Max(Math.Max(turning, CoulombSmallRhoFactor * Math.Sqrt(Math.Abs(eta))), 1);

        // 原点級数の和 T = sum·2^sumExponent から F = C_ℓ(η) ρ^(ℓ+1) T を組み立てる
        private static double CoulombSeriesValue(double l, double eta, double rho, double sum, long sumExponent)
        {
            double floor = Math.Floor(l);
            double mantissa = CoulombScaledNormalization(l, eta, out long normalizationExponent);
            mantissa *= ScaledPower(rho, (long)floor + 1, out long powerExponent);
            if (l != floor)
            {
                mantissa *= Math.Pow(rho, l - floor);
            }

            return ScaleB(mantissa * sum, sumExponent + normalizationExponent + powerExponent);
        }

        // 原点級数 T = Σₘ tₘ (DLMF 33.6.E1; tₘ = (2ηρ tₘ₋₁ - ρ² tₘ₋₂)/(m(m + 2ℓ + 1)), t₀ = 1)。
        // sum·2^exponent で値を返し、戻り値として条件数 Σ|tₘ|/|T| を返す
        private static double CoulombSeries(double l, double eta, double rho, out double sum, out long exponent)
        {
            double twoEtaRho = 2 * eta * rho;
            double rho2 = rho * rho;
            double previous = 1;
            double current = eta * rho / (l + 1);
            sum = previous + current;
            double absoluteSum = Math.Abs(previous) + Math.Abs(current);
            exponent = 0;
            for (int m = 2; m < CoulombMaxIterations; m++)
            {
                double next = (twoEtaRho * current - rho2 * previous) / (m * (m + 2 * l + 1));
                previous = current;
                current = next;
                sum += next;
                absoluteSum += Math.Abs(next);
                if (absoluteSum > 1e250)
                {
                    // 2⁻⁸⁰⁰ 倍しても項比は不変 (指数へ持ち越す)
                    previous = Math.ScaleB(previous, -800);
                    current = Math.ScaleB(current, -800);
                    sum = Math.ScaleB(sum, -800);
                    absoluteSum = Math.ScaleB(absoluteSum, -800);
                    exponent += 800;
                }

                // 項は m ≈ √(2|η|ρ) + ρ を過ぎると急減する。2 項連続で ε/4 未満で打ち切る
                double tolerance = 0.25 * MachineEpsilon * absoluteSum;
                if (Math.Abs(next) <= tolerance && Math.Abs(previous) <= tolerance && m > 2)
                {
                    return absoluteSum / Math.Abs(sum);
                }
            }

            return double.PositiveInfinity;
        }

        // 微分方程式 u'' = (λ/ρ² + 2η/ρ - 1) u を from から to へテイラー級数で積分する (u·2^exponent, u'·2^exponent)。
        // 収束半径 ρ₀ に対し |h| ≤ ρ₀/2 かつ局所波数 κ|h| ≤ 2 とし 1 ステップ約 30 項で収束させる
        private static void CoulombIntegrate(
            double lambda, double eta, double from, double to, ref double u, ref double up, ref long exponent, long stopExponent = long.MaxValue)
        {
            double x = from;
            double limitFactor = 1;
            while (x != to)
            {
                if (exponent >= stopExponent)
                {
                    // 呼び出し側が「これ以上は増える一方」と分かっている場合の打ち切り（ScaleB で ±∞ になる）
                    exponent = long.MaxValue / 2;
                    return;
                }

                double a = lambda + (2 * eta - x) * x;
                double kappa = Math.Sqrt(Math.Abs(a)) / x;
                double hmax = limitFactor * Math.Min(0.5 * x, kappa > 0 ? 2 / kappa : double.PositiveInfinity);
                double remaining = to - x;
                bool last = Math.Abs(remaining) <= hmax;
                double h = last ? remaining : Math.CopySign(hmax, remaining);
                double r = h / x;
                double r2 = r * r;
                double b = 2 * (eta - x);

                // dm2 = dₙ₋₂、dm1 = dₙ₋₁、d0 = dₙ、d1 = dₙ₊₁ から dₙ₊₂ を作る
                double dm2 = 0;
                double dm1 = 0;
                double d0 = u;
                double d1 = up * h;
                double sum = d0 + d1;
                double derivativeSum = d1;
                double absoluteSum = Math.Abs(d0) + Math.Abs(d1);
                bool converged = false;
                for (int n = 0; n < 300; n++)
                {
                    double d2 = ((a - n * (n - 1.0)) * r2 * d0 + b * h * r2 * dm1 - h * h * r2 * dm2 - 2 * r * (n + 1.0) * n * d1)
                        / ((n + 2.0) * (n + 1));
                    dm2 = dm1;
                    dm1 = d0;
                    d0 = d1;
                    d1 = d2;
                    sum += d2;
                    derivativeSum += (n + 2) * d2;
                    absoluteSum += Math.Abs(d2);

                    // 3項続けて小さくなったら打ち切る（係数の漸化式は3項前まで参照するので、1項だけ小さい偶然を除く）
                    if (n >= 4 && Math.Abs(d2) + Math.Abs(d0) + Math.Abs(dm1) <= 0.125 * MachineEpsilon * absoluteSum)
                    {
                        converged = true;
                        break;
                    }
                }

                if (!converged)
                {
                    // 項が多すぎる（κ の変化が大きい）ので刻みを半分にしてやり直す
                    limitFactor *= 0.5;
                    continue;
                }

                u = sum;
                up = derivativeSum / h;
                x = last ? to : x + h;
                limitFactor = Math.Min(1, 2 * limitFactor);

                // 障壁の下では値が指数的に増減するので、2の冪で正規化して指数を持ち越す
                if (u != 0 && (Math.Abs(u) > 1e150 || Math.Abs(u) < 1e-150))
                {
                    int shift = Math.ILogB(u);
                    u = Math.ScaleB(u, -shift);
                    up = Math.ScaleB(up, -shift);
                    exponent += shift;
                }
            }
        }

        // C_ℓ(η) = mantissa · 2^exponent (DLMF 33.2.E5, 33.2.E6)。
        // ℓ₀ = ℓ - ⌊ℓ⌋ から漸化式 C_L = C_(L-1) √(L² + η²)/(L(2L+1)) で昇順計算し毎回正規化する (⌊ℓ⌋ > CoulombNormalizationMaxProduct は対数公式)
        private static double CoulombScaledNormalization(double l, double eta, out long exponent)
        {
            double floor = Math.Floor(l);
            double fraction = l - floor;
            double logBase;
            if (floor > CoulombNormalizationMaxProduct)
            {
                logBase = CoulombLogNormalization(l, eta);
                floor = 0;
            }
            else if (fraction != 0)
            {
                logBase = CoulombLogNormalization(fraction, eta);
            }
            else if (eta == 0)
            {
                logBase = 0;
            }
            else
            {
                // ln C₀ = (ln|t| - ln|e^t - 1|)/2 (t = 2πη)。小 |t| は expm1、大 t は t + ln(1 - e^(-t))
                double t = 2 * Math.PI * eta;
                double logDenominator = t > 30 ? t + Log1p(-Math.Exp(-t)) : Math.Log(Math.Abs(Expm1(t)));
                logBase = 0.5 * (Math.Log(Math.Abs(t)) - logDenominator);
            }

            double n = Math.Round(logBase / Ln2);
            double mantissa = Math.Exp(Math.FusedMultiplyAdd(-n, Ln2, logBase) - n * Ln2Lo);
            exponent = (long)n;
            for (int j = 1; j <= floor; j++)
            {
                // √(L² + η²)/(L(2L+1)) = √(1 + (η/L)²)/(2L+1) (η² のオーバーフロー回避)
                double order = fraction + j;
                double ratio = eta / order;
                mantissa *= Math.Sqrt(1 + ratio * ratio) / (2 * order + 1);
                int shift = Math.ILogB(mantissa);
                mantissa = Math.ScaleB(mantissa, -shift);
                exponent += shift;
            }

            return mantissa;
        }

        // ln C_ℓ(η) (DLMF 33.2.E5)。引力側での e^(π|η|/2) と |Γ| の打ち消しを避けるため CoulombLogGamma で π|η|/2 を解析的に除いた量を使う
        private static double CoulombLogNormalization(double l, double eta)
        {
            CoulombLogGamma(l + 1, eta, out double modulusPlus, out _);
            double repulsive = eta > 0 ? -Math.PI * eta : 0;
            return l * Ln2 + modulusPlus + repulsive - LogGamma(2 * l + 2);
        }

        // ln Γ(x + iy) (x ≥ 1) の実部 + π|y|/2 (modulusPlus) と連続な偏角 arg Γ(x + iy) (phase)。
        // |z| ≥ 10 で 12 項スターリング展開 (誤差 < 10⁻¹⁹) を用い、-y arg z = -π|y|/2 + |y| atan(x/|y|) により π|y|/2 を解析的に除く
        private static void CoulombLogGamma(double x, double y, out double modulusPlus, out double phase)
        {
            double ay = Math.Abs(y);
            double shiftModulus = 0;
            double shiftPhase = 0;
            double zx = x;
            if (ay < 10)
            {
                while (zx < 10)
                {
                    shiftModulus += 0.5 * Math.Log(zx * zx + y * y);
                    shiftPhase += Math.Atan2(y, zx);
                    zx += 1;
                }
            }

            // ln|z| は zx² + y² のオーバーフローを避けて大きい方でくくる
            double big = Math.Max(zx, ay);
            double small = Math.Min(zx, ay);
            double ratio = small / big;
            double logAbs = Math.Log(big) + 0.5 * Log1p(ratio * ratio);
            double arg = Math.Atan2(y, zx);
            double modulus = (zx - 0.5) * logAbs + (ay == 0 ? 0 : ay * Math.Atan(zx / ay)) - zx + LnSqrtTwoPi;
            double imaginary = (zx - 0.5) * arg + y * (logAbs - 1);

            Complex inverse = Complex.Reciprocal(new Complex(zx, y));
            Complex inverse2 = inverse * inverse;
            Complex power = inverse;
            Complex series = Complex.Zero;
            for (int k = 1; k <= 12; k++)
            {
                series += BernoulliEven[k] / (2.0 * k * (2 * k - 1)) * power;
                power *= inverse2;
            }

            modulusPlus = modulus + series.Real - shiftModulus;
            phase = imaginary + series.Imaginary - shiftPhase;
        }

        // CF1 による対数微分 f = F'/F と符号 sign (DLMF 33.8.E1; Barnett の COULFG と同様に修正 Lentz 法で分母 D の符号反転回数から sign を決定)。
        // O(ρ) 回の反復を要し、未収束なら false を返す
        private static bool CoulombContinuedFraction1(double l, double eta, double rho, out double ratio, out int sign)
        {
            double first = l + 1;
            double h = first / rho + eta / first;
            if (Math.Abs(h) < Tiny)
            {
                h = Tiny;
            }

            double c = h;
            double d = 0;
            sign = 1;
            for (int n = 1; n <= CoulombMaxIterations; n++)
            {
                // L = ℓ + n。係数は反復ごとに直接計算する (累積加算による丸め誤差蓄積を回避)
                double order = l + n;
                double quotient = eta / order;
                double a = -(1 + quotient * quotient);
                double b = order / rho + quotient + (order + 1) / rho + eta / (order + 1);
                d = b + a * d;
                if (Math.Abs(d) < Tiny)
                {
                    d = Tiny;
                }

                c = b + a / c;
                if (Math.Abs(c) < Tiny)
                {
                    c = Tiny;
                }

                d = 1 / d;
                double delta = c * d;
                h *= delta;
                if (d < 0)
                {
                    sign = -sign;
                }

                if (Math.Abs(delta - 1) <= MachineEpsilon)
                {
                    ratio = h;
                    return true;
                }
            }

            ratio = double.NaN;
            return false;
        }

        // スティード法 (DLMF 33.8.E4; ρ > 0)。CF1 の f = F'/F と CF2 の p + iq = (G' + iF')/(G + iF) およびロンスキアン q(F² + G²) = 1 から求める
        private static bool CoulombSteed(double l, double eta, double rho, out double f, out double fp, out double g, out double gp)
        {
            f = double.NaN;
            fp = double.NaN;
            g = double.NaN;
            gp = double.NaN;

            if (!CoulombContinuedFraction1(l, eta, rho, out double h, out int sign))
            {
                return false;
            }

            // CF2: K = a₁/(b₁ + a₂/(b₂ + …)) の分母 b₁ + … を修正 Lentz 法で計算する (初期値 Tiny だと a₁/Tiny がオーバーフローする)
            var alpha = new Complex(1 + l, eta);
            var beta = new Complex(-l, eta);
            double twoRhoMinusEta = 2 * (rho - eta);
            var denominator = new Complex(twoRhoMinusEta, 2);
            Complex cc = denominator;
            Complex dd = Complex.Zero;
            bool converged = false;
            for (int n = 2; n <= CoulombMaxIterationsCf2; n++)
            {
                Complex an = (alpha + (n - 1)) * (beta + (n - 1));
                var bn = new Complex(twoRhoMinusEta, 2.0 * n);
                dd = bn + an * dd;
                if (dd == Complex.Zero)
                {
                    dd = Tiny;
                }

                cc = bn + an / cc;
                if (cc == Complex.Zero)
                {
                    cc = Tiny;
                }

                dd = Complex.Reciprocal(dd);
                Complex delta = cc * dd;
                denominator *= delta;
                if (Math.Abs(delta.Real - 1) + Math.Abs(delta.Imaginary) <= MachineEpsilon)
                {
                    converged = true;
                    break;
                }
            }

            if (!converged)
            {
                return false;
            }

            Complex k = alpha * beta / denominator;
            Complex pq = new Complex(0, 1 - eta / rho) + new Complex(0, 1 / rho) * k;
            double p = pq.Real;
            double q = pq.Imaginary;
            double gamma = (h - p) / q;

            // √(1 + γ²) は γ² のオーバーフローを避けて計算する
            double absGamma = Math.Abs(gamma);
            double norm = absGamma > 1 ? absGamma * Math.Sqrt(1 + 1 / (absGamma * absGamma)) : Math.Sqrt(1 + gamma * gamma);
            f = sign / (Math.Sqrt(q) * norm);
            g = gamma * f;
            fp = h * f;
            gp = p * g - q * f;
            return true;
        }

        // 漸近展開 H⁺ = G + iF ~ e^(iθ) Σ (a)ₖ(b)ₖ/(k!(2iρ)ᵏ) (a = 1 + ℓ + iη, b = -ℓ + iη, θ = ρ - η ln 2ρ - ℓπ/2 + σ_ℓ(η))。
        // 項が増え始める前に倍精度収束しなければ false を返す (e^(iθ) = e^(iρ) e^(iφ) に分離して ρ の正確な引数還元を維持)
        private static bool CoulombAsymptotic(double l, double eta, double rho, out double f, out double g)
        {
            f = double.NaN;
            g = double.NaN;
            var alpha = new Complex(1 + l, eta);
            var beta = new Complex(-l, eta);
            var inverseTwoIRho = new Complex(0, -0.5 / rho);
            Complex term = Complex.One;
            Complex sum = Complex.One;
            double previous = double.PositiveInfinity;
            bool converged = false;
            for (int k = 0; k < 200; k++)
            {
                term *= (alpha + k) * (beta + k) * inverseTwoIRho / (k + 1);
                double magnitude = Complex.Abs(term);
                if (magnitude > previous)
                {
                    return false;
                }

                sum += term;
                previous = magnitude;
                if (magnitude <= 0.25 * MachineEpsilon * Complex.Abs(sum))
                {
                    converged = true;
                    break;
                }
            }

            if (!converged)
            {
                return false;
            }

            CoulombLogGamma(1 + l, eta, out _, out double sigma);

            // ℓπ/2 は ℓ を4で割った剰余（正確に求まる）に π/2 を掛ける
            double phi = sigma - eta * Math.Log(2 * rho) - 0.5 * Math.PI * Math.IEEERemainder(l, 4);
            double cosRho = Math.Cos(rho);
            double sinRho = Math.Sin(rho);
            double cosPhi = Math.Cos(phi);
            double sinPhi = Math.Sin(phi);
            double cosTheta = cosRho * cosPhi - sinRho * sinPhi;
            double sinTheta = sinRho * cosPhi + cosRho * sinPhi;
            g = cosTheta * sum.Real - sinTheta * sum.Imaginary;
            f = sinTheta * sum.Real + cosTheta * sum.Imaginary;
            return true;
        }
    }
}
