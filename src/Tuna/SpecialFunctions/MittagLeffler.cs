using System.Numerics;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // ラプラス逆変換の台形則で目標とする相対誤差の対数 -ln(tol)（tol ≈ 3×10⁻¹⁷）
        private const double MittagLefflerLogTolerance = 38;

        // ラプラス逆変換の台形則の分点数上限
        private const int MittagLefflerMaxNodes = 400;

        // べき級数の項数上限（未収束時はラプラス逆変換へ切り替える）
        private const int MittagLefflerMaxSeriesTerms = 4000;

        // べき級数を採用する打ち消し倍率の上限 Σ|項| / |和|
        private const double MittagLefflerMaxSeriesLoss = 8;

        /// <summary>
        /// Computes the two-parameter Mittag-Leffler function <c>E_(α,β)(x) = Σ_(k≥0) xᵏ / Γ(αk + β)</c>.
        /// <para>2 パラメータ Mittag-Leffler 関数 <c>E_(α,β)(x) = Σ_(k≥0) xᵏ / Γ(αk + β)</c> を計算します。</para>
        /// </summary>
        /// <param name="alpha">The first parameter (<c>α &gt; 0</c>). / 第 1 パラメータ（<c>α &gt; 0</c>）。</param>
        /// <param name="beta">The second parameter. / 第 2 パラメータ。</param>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>E_(α,β)(x)</c>, or <see cref="double.NaN"/> if <paramref name="alpha"/> ≤ 0, if <paramref name="alpha"/> or <paramref name="beta"/> is non-finite, or if any argument is <see cref="double.NaN"/>. / <c>E_(α,β)(x)</c> の値。<paramref name="alpha"/> ≤ 0、<paramref name="alpha"/> または <paramref name="beta"/> が有限でない場合、またはいずれかの引数が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <c>1 / Γ(β)</c> at <c>x = 0</c>, <see cref="double.PositiveInfinity"/> at <see cref="double.PositiveInfinity"/>, and at <see cref="double.NegativeInfinity"/> returns <c>0</c> when <c>α &lt; 2</c> and <see cref="double.NaN"/> otherwise.</para>
        /// <para><c>x = 0</c> では <c>1 / Γ(β)</c>、<see cref="double.PositiveInfinity"/> では <see cref="double.PositiveInfinity"/>、<see cref="double.NegativeInfinity"/> では <c>α &lt; 2</c> のとき <c>0</c>、それ以外のとき <see cref="double.NaN"/> を返します。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/10.46#E3"/>
        public static double MittagLeffler(double alpha, double beta, double x)
        {
            if (double.IsNaN(alpha) || double.IsNaN(beta) || double.IsNaN(x) || alpha <= 0 || double.IsInfinity(alpha) || double.IsInfinity(beta))
            {
                return double.NaN;
            }

            if (x == 0)
            {
                return ReciprocalGamma(beta);
            }

            if (double.IsInfinity(x))
            {
                if (x > 0)
                {
                    return double.PositiveInfinity;
                }

                return alpha < 2 ? 0 : double.NaN;
            }

            if (alpha == 1 && IsInteger(beta) && Math.Abs(beta) < 1e6)
            {
                double closed = MittagLefflerExponential((int)beta, x);
                if (!double.IsNaN(closed))
                {
                    return closed;
                }
            }

            // r = |x|^(1/α)。1/α の丸めで ln r に ε·ln r の誤差が入り、e^r や位相 r sin θ では ε·r·ln r に拡大するので、
            // rᵅ（Math.Pow は正しく丸めに近い）との比で 1 回ニュートン補正する
            double absX = Math.Abs(x);
            double r = Math.Pow(absX, 1 / alpha);
            if (r > 0 && double.IsFinite(r))
            {
                double power = Math.Pow(r, alpha);
                if (power > 0 && double.IsFinite(power))
                {
                    r += r * ((absX - power) / power) / alpha;
                }
            }

            if (r == 0)
            {
                // |x|^(1/α) がアンダーフローするほど小さい：E = 1/Γ(β) + x/Γ(α+β) + …
                return ReciprocalGamma(beta) + x * ReciprocalGamma(alpha + beta);
            }

            double[] angles = MittagLefflerPoleAngles(alpha, x);
            double value;
            if (x > 0)
            {
                if (r >= 19 && MittagLefflerAsymptotic(alpha, beta, x, r, angles, out value))
                {
                    return value;
                }

                if (MittagLefflerSeries(alpha, beta, x, out value))
                {
                    return value;
                }

                return MittagLefflerLaplace(alpha, beta, x, r, angles);
            }

            // x < 0：べき級数の打ち消しは最大の項 ~e^r と、極の寄与 ~e^(r cos θ) の比で決まる
            double growth = 0;
            foreach (double turn in angles)
            {
                growth = Math.Max(growth, r * double.CosPi(turn));
            }

            // β が大きく負のときは項が初めから減るので、打ち消しの見積もり（級数の中で確かめる）に関わらず級数を先に試す
            if ((r - growth <= 3.5 || beta < -r) && MittagLefflerSeries(alpha, beta, x, out value))
            {
                return value;
            }

            if (r >= 20 && MittagLefflerAsymptotic(alpha, beta, x, r, angles, out value))
            {
                return value;
            }

            return MittagLefflerLaplace(alpha, beta, x, r, angles);
        }

        /// <summary>
        /// Computes the one-parameter Mittag-Leffler function <c>E_α(x) = E_(α,1)(x)</c>.
        /// <para>1 パラメータ Mittag-Leffler 関数 <c>E_α(x) = E_(α,1)(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="alpha">The parameter (<c>α &gt; 0</c>). / パラメータ（<c>α &gt; 0</c>）。</param>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>E_α(x)</c>. / <c>E_α(x)</c> の値。</returns>
        /// <seealso href="https://dlmf.nist.gov/10.46#E3"/>
        public static double MittagLeffler(double alpha, double x) => MittagLeffler(alpha, 1, x);

        // α = 1、整数 β = m の閉じた式（m ≤ 1: x^(1-m) eˣ、m ≥ 2 かつ |x| > 1: (eˣ - Σ_(k<m-1) xᵏ/k!)/x^(m-1)、適用不可は NaN）
        private static double MittagLefflerExponential(int m, double x)
        {
            if (m <= 1)
            {
                // E_(1,m)(x) = x E_(1,m+1)(x) + 1/Γ(m) で 1/Γ(m) = 0（m ≤ 0）から
                double power = Math.Pow(x, 1 - m);
                double exponential = Math.Exp(x);
                double product = power * exponential;
                if (double.IsFinite(product) && double.IsFinite(power) && exponential != 0)
                {
                    return product;
                }

                // xⁿ がオーバーフローして eˣ がアンダーフローする（またはその逆）場合は対数で合わせる
                double sign = x < 0 && (1 - m) % 2 != 0 ? -1 : 1;
                return sign * Math.Exp((1 - m) * Math.Log(Math.Abs(x)) + x);
            }

            if (Math.Abs(x) <= 1 || m > 170)
            {
                return double.NaN;
            }

            // x > 0 では eˣ と部分和の差は x が m に比べて大きくないと打ち消すので、eˣ が部分和より十分大きいときだけ使う
            double sum = 0;
            double term = 1;
            double absSum = 0;
            for (int k = 0; k < m - 1; k++)
            {
                sum += term;
                absSum += Math.Abs(term);
                term *= x / (k + 1);
            }

            double exp = Math.Exp(x);
            double difference = exp - sum;
            if (!(Math.Abs(difference) * MittagLefflerMaxSeriesLoss >= exp + absSum))
            {
                return double.NaN;
            }

            return difference / Math.Pow(x, m - 1);
        }

        // s^α = x の主リーマン面上の根の偏角 θⱼ/π（|θⱼ| ≤ π）を返す（double.CosPi 用）
        private static double[] MittagLefflerPoleAngles(double alpha, double x)
        {
            double phase = x > 0 ? 0 : 1;
            long jMin = (long)Math.Ceiling((-alpha - phase) / 2);
            long jMax = (long)Math.Floor((alpha - phase) / 2);
            var turns = new List<double>();
            for (long j = jMin; j <= jMax && turns.Count < 100_000; j++)
            {
                double turn = (phase + 2 * j) / alpha;
                if (Math.Abs(turn) <= 1)
                {
                    turns.Add(turn);
                }
            }

            return [.. turns];
        }

        // 極 r e^(iθ) の留数 (1/α) s^(1-β) e^s の実部和（r cos²(θ/2) ≤ includeLimit の極は除外、切断上の極は重み 1/2）
        private static double MittagLefflerResidues(double alpha, double beta, double r, double[] angles, double includeLimit)
        {
            double logR = Math.Log(r);
            double sum = 0;
            foreach (double turn in angles)
            {
                double halfCos = double.CosPi(0.5 * turn);
                bool onCut = Math.Abs(turn) >= 1;
                if (!onCut && r * halfCos * halfCos <= includeLimit)
                {
                    continue;
                }

                if (onCut && includeLimit > 0)
                {
                    // 切断上の極は積分路の左にあり、積分の側で扱われる
                    continue;
                }

                double weight = onCut ? 0.5 : 1;
                // 絶対値 e^(r cos θ) · r^(1-β)/α。指数を 1 つの和にすると r が大きいとき和の丸め誤差 ε·r が値に効くので、
                // オーバーフローしない限り e^(r cos θ) を別に計算して掛ける
                double growth = r * double.CosPi(turn);
                double logPrefactor = (1 - beta) * logR - Math.Log(alpha);
                // 位相 r sin θ + (1-β)θ は大きくなりうるので、和を作らずに加法定理で cos を求める（和の丸め誤差 ε·r を避ける）
                (double sinA, double cosA) = Math.SinCos(r * double.SinPi(turn));
                (double sinB, double cosB) = Math.SinCos((1 - beta) * Math.PI * turn);
                double magnitude = Math.Exp(growth) * Math.Exp(logPrefactor);
                if (!double.IsFinite(magnitude) || magnitude == 0)
                {
                    magnitude = Math.Exp(growth + logPrefactor);
                }

                double contribution = (cosA * cosB - sinA * sinB) * magnitude;
                sum += weight * (double.IsNaN(contribution) ? 0 : contribution);
            }

            return sum;
        }

        // べき級数 Σ xᵏ/Γ(αk + β)（スケール分離で評価し、打ち消し倍率 Σ|項|/|和| ≤ MittagLefflerMaxSeriesLoss なら採用）
        private static bool MittagLefflerSeries(double alpha, double beta, double x, out double value)
        {
            value = double.NaN;
            double sum = 0;
            double absSum = 0;
            double previous = double.PositiveInfinity;
            double powerMantissa = 1;
            long powerExponent = 0;
            bool scaled = false;
            long reference = 0;
            for (int k = 0; k < MittagLefflerMaxSeriesTerms; k++)
            {
                if (k > 0)
                {
                    powerMantissa *= x;
                    int shift = Math.ILogB(powerMantissa);
                    if (Math.Abs(shift) > 500)
                    {
                        powerMantissa = Math.ScaleB(powerMantissa, -shift);
                        powerExponent += shift;
                    }
                }

                double a = alpha * k + beta;
                double term;
                if (IsNonPositiveInteger(a))
                {
                    term = 0;
                }
                else
                {
                    // 項は仮数と指数で作り、最初の 0 でない項の指数にそろえて足す（値がオーバーフローする場合も ∞ - ∞ にならず ±∞ を返す）
                    double termMantissa;
                    long termExponent;
                    if (Math.Abs(a) <= 100)
                    {
                        termMantissa = powerMantissa * ReciprocalGamma(a);
                        termExponent = powerExponent;
                    }
                    else
                    {
                        double gammaMantissa = ScaledGammaSigned(a, out long gammaExponent);
                        termMantissa = powerMantissa / gammaMantissa;
                        termExponent = powerExponent - gammaExponent;
                    }

                    if (!scaled && termMantissa != 0)
                    {
                        scaled = true;
                        reference = termExponent + Math.ILogB(termMantissa);
                    }

                    term = ScaleB(termMantissa, termExponent - reference);
                }

                sum += term;
                absSum += Math.Abs(term);
                double magnitude = Math.Abs(term);
                if (a > 1 && magnitude <= previous && magnitude <= 1e-17 * absSum)
                {
                    if (double.IsInfinity(absSum) || absSum > MittagLefflerMaxSeriesLoss * Math.Abs(sum))
                    {
                        return false;
                    }

                    value = ScaleB(sum, reference);
                    return true;
                }

                if (term != 0)
                {
                    previous = magnitude;
                }
            }

            return false;
        }

        // 漸近展開 E ≈ Σ 留数 - Σ_(k≥1) x^(-k)/Γ(β - αk)（最小項が値の 2⁻⁵⁵ 倍以下なら採用）
        private static bool MittagLefflerAsymptotic(double alpha, double beta, double x, double r, double[] angles, out double value)
        {
            double residues = MittagLefflerResidues(alpha, beta, r, angles, 0);
            double sum = 0;
            double inversePower = 1;
            double previous = double.PositiveInfinity;
            double smallest = double.PositiveInfinity;
            bool converged = false;
            for (int k = 1; k <= 1000; k++)
            {
                inversePower /= x;
                if (inversePower == 0)
                {
                    converged = true;
                    break;
                }

                double term = -inversePower * ReciprocalGamma(beta - alpha * k);
                double magnitude = Math.Abs(term);
                if (term == 0)
                {
                    // 1/Γ が 0 になる項（β - αk が 0 以下の整数）は最小項の判定に使わない
                    continue;
                }

                if (magnitude > previous)
                {
                    break;
                }

                sum += term;
                previous = magnitude;
                smallest = magnitude;
                if (magnitude <= 1e-17 * Math.Abs(sum + residues))
                {
                    converged = true;
                    break;
                }
            }

            value = residues + sum;
            if (double.IsInfinity(residues))
            {
                return true;
            }

            if (converged || double.IsPositiveInfinity(smallest))
            {
                // 全ての項が 0（例：α = 1, β = 1）か十分小さく収束した
                return !double.IsNaN(value);
            }

            return smallest <= 2.8e-17 * Math.Abs(value);
        }

        // 放物線積分路 s(u) = μ(1 + iu)² 上のラプラス逆変換 E = Σ 留数 + (1/2πi) ∫ e^s s^(α-β)/(s^α - x) ds
        private static double MittagLefflerLaplace(double alpha, double beta, double x, double r, double[] angles)
        {
            // 極が積分路 s(u) = μ(1 + iu)² 上に来る μ は wⱼ = r cos²(θⱼ/2)。各 μ の候補について帯の幅・刻み・分点数を見積もる
            var weights = new List<double>();
            foreach (double turn in angles)
            {
                if (Math.Abs(turn) < 1)
                {
                    double c = double.CosPi(0.5 * turn);
                    weights.Add(r * c * c);
                }
            }

            var candidates = new List<double>();
            for (int k = 0; k < 30; k++)
            {
                candidates.Add(0.05 * Math.Pow(1.3, k));
            }

            if (beta - alpha > 0.05)
            {
                // e^s s^(α-β) の実軸上の鞍点 s = β - α（β が大きいと小さな μ では被積分関数が非常に大きい）
                candidates.Add(beta - alpha);
            }

            foreach (double w in weights)
            {
                candidates.Add(0.5 * w);
                candidates.Add(0.7 * w);
                candidates.Add(1.4 * w);
                candidates.Add(2 * w);
            }

            double bestScore = double.PositiveInfinity;
            double bestMu = double.NaN;
            double bestStep = 0;
            int bestNodes = 0;
            foreach (double mu in candidates)
            {
                if (!MittagLefflerContourParameters(alpha, beta, mu, weights, out double step, out int nodes, out double uMax))
                {
                    continue;
                }

                // 丸め誤差の目安：積分路上の |F(s)(1+iu)| の最大値（u = 0, U/4, U/2 で見積もる）に μU を掛けたもの
                double logMagnitude = double.NegativeInfinity;
                for (int i = 0; i < 3; i++)
                {
                    double u = i * 0.25 * uMax;
                    double u2 = 1 + u * u;
                    double logS = Math.Log(mu * u2);
                    double logDenominator = Math.Max(alpha * logS, Math.Log(Math.Abs(x))) - Ln2;
                    double g = mu * (1 - u * u) + (alpha - beta) * logS + 0.5 * Math.Log(u2) - logDenominator;
                    logMagnitude = Math.Max(logMagnitude, g);
                }

                double score = logMagnitude + Math.Log(mu * uMax) + 0.3 * Math.Log(nodes);
                if (score < bestScore)
                {
                    bestScore = score;
                    bestMu = mu;
                    bestStep = step;
                    bestNodes = nodes;
                }
            }

            if (double.IsNaN(bestMu))
            {
                return double.NaN;
            }

            double total = MittagLefflerIntegrand(alpha, beta, x, bestMu, 0).Real;
            for (int k = 1; k <= bestNodes; k++)
            {
                total += 2 * MittagLefflerIntegrand(alpha, beta, x, bestMu, k * bestStep).Real;
            }

            double integral = bestStep * bestMu / Math.PI * total;
            return integral + MittagLefflerResidues(alpha, beta, r, angles, bestMu);
        }

        // 被積分関数 F(s(u))(1 + iu)（s(u) = μ(1 + iu)²）
        private static Complex MittagLefflerIntegrand(double alpha, double beta, double x, double mu, double u)
        {
            var logS = new Complex(Math.Log(mu) + Log1p(u * u), 2 * Math.Atan(u));
            var s = new Complex(mu * (1 - u) * (1 + u), 2 * mu * u);
            Complex numerator = Complex.Exp(s + (alpha - beta) * logS);
            Complex denominator = Complex.Exp(alpha * logS) - x;
            return numerator / denominator * new Complex(1, u);
        }

        // 積分路 μ に対する台形則の刻み・分点数を Weideman–Trefethen 型の誤差評価で決定する
        private static bool MittagLefflerContourParameters(double alpha, double beta, double mu, List<double> weights,
            out double step, out int nodes, out double uMax)
        {
            step = 0;
            nodes = 0;
            uMax = 0;
            const double logTolerance = MittagLefflerLogTolerance;
            double dMinus = double.PositiveInfinity;
            double dPlus = 1;
            foreach (double w in weights)
            {
                double c = Math.Sqrt(w / mu);
                if (c > 1)
                {
                    dMinus = Math.Min(dMinus, c - 1);
                }
                else
                {
                    dPlus = Math.Min(dPlus, 1 - c);
                }
            }

            if (dMinus <= 0.02 || dPlus <= 0.02)
            {
                return false;
            }

            double stepPlus = 0;
            for (int i = 1; i <= 20; i++)
            {
                double d = 0.95 * dPlus * i / 20;
                double singular = Math.Max(0, 2 * (beta - alpha) * -Math.Log(1 - d));
                double h = 2 * Math.PI * d / (logTolerance + mu * (1 - d) * (1 - d) - mu + singular);
                stepPlus = Math.Max(stepPlus, h);
            }

            double dm = Math.Min(0.9 * dMinus, Math.Sqrt(logTolerance / mu));
            double stepMinus = 2 * Math.PI * dm / (logTolerance + mu * (2 * dm + dm * dm));
            step = Math.Min(stepPlus, stepMinus);

            // 打ち切り：|F(s(u))(1+iu)| ≲ e^(μ(1-u²)) (μ(1+u²))^(α-β) √(1+u²) を e^μ の tol 倍以下にする
            double u = Math.Sqrt(logTolerance / mu);
            for (int i = 0; i < 3; i++)
            {
                u = Math.Sqrt((logTolerance + Math.Max(0, alpha - beta) * Math.Log(1 + u * u) + Math.Log(1 + u)) / mu);
            }

            uMax = u;
            double n = Math.Ceiling(u / step);
            if (!(n <= MittagLefflerMaxNodes))
            {
                return false;
            }

            nodes = (int)n;
            return true;
        }
    }
}
