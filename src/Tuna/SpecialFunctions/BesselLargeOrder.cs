namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // Debye の一様漸近展開 (DLMF 10.41.3) を使う次数の下限。
        // U₀〜U₁₃ で ν ≥ 50 なら打ち切り誤差は max|U₁₄(p)|/ν¹⁴ ≈ 4×10⁻²² 以下 (0 ≤ p ≤ 1)
        private const double DebyeMinOrder = 50;

        // Debye の多項式 U_k(p) (k = 0, …, 13) の係数 (DebyeCoefficients[k][n] が pⁿ の係数; DLMF 10.41.9 の漸化式で生成)
        private static readonly double[][] DebyeCoefficients = CreateDebyeCoefficients();

        // |U_k(p)| ≤ Σₙ |DebyeCoefficients[k][n]| (0 ≤ p ≤ 1) の上界。U_k は零点を持ちうるため項値ではなく上界で打ち切る
        private static readonly double[] DebyeBounds = CreateDebyeBounds(DebyeCoefficients);

        private static double[][] CreateDebyeCoefficients()
        {
            var u = new double[14][];
            u[0] = [1.0];
            for (int k = 0; k + 1 < u.Length; k++)
            {
                double[] c = u[k];
                var next = new double[c.Length + 3];
                for (int n = 0; n < c.Length; n++)
                {
                    // cₙpⁿ からの寄与: p²(1 - p²)·n cₙ pⁿ⁻¹/2 + (1/8)∫₀ᵖ (1 - 5t²) cₙ tⁿ dt
                    next[n + 1] += 0.5 * n * c[n] + c[n] / (8.0 * (n + 1));
                    next[n + 3] -= 0.5 * n * c[n] + 5 * c[n] / (8.0 * (n + 3));
                }

                u[k + 1] = next;
            }

            return u;
        }

        private static double[] CreateDebyeBounds(double[][] coefficients)
        {
            var bounds = new double[coefficients.Length];
            for (int k = 0; k < coefficients.Length; k++)
            {
                foreach (double c in coefficients[k])
                {
                    bounds[k] += Math.Abs(c);
                }
            }

            return bounds;
        }

        // Debye 展開の先頭 1 を除いた和 Σ_{k≥1} U_k(p)/νᵏ (0 < p ≤ 1, ν ≥ DebyeMinOrder)
        private static double DebyeSeriesTail(double p, double nu)
        {
            double sum = 0;
            double scale = 1;
            for (int k = 1; k < DebyeCoefficients.Length; k++)
            {
                scale /= nu;
                if (DebyeBounds[k] * scale <= Epsilon * 0.01)
                {
                    break;
                }

                double[] c = DebyeCoefficients[k];
                double value = 0;
                for (int n = c.Length - 1; n >= 0; n--)
                {
                    value = value * p + c[n];
                }

                sum += value * scale;
            }

            return sum;
        }

        // 一様漸近展開へ切り替える次数しきい値 (漸化式が約 1 万段になる点。BesselJYCore の転回点付近を除く範囲と BesselIKCore 全域)
        private const double BesselLargeOrderMin = 1e4;

        // J_ν, Y_ν の転回点付近で Airy 型一様漸近展開 (DLMF 10.20.4–10.20.5) を使う次数しきい値。
        // A₀ = 1, A₁, B₀ までで省いた A₂/ν⁴, B₁/ν^(10/3) が 1e-19 以下、|ζ| ≤ 20/ν^(2/3) ≤ 0.0093 でテイラー打ち切りも 1e-17 以下
        private const double BesselAiryTypeMinOrder = 1e5;

        // Debye 展開の適用下限: |ν^(2/3) ζ| = W = 20 での (2/3)W^(3/2)。W = 20 では k = 13 の項が 8e-20 で 14 項が倍精度に収まる
        private const double BesselDebyeMinExponent = 59.62847939999439;

        // Airy 型一様漸近展開の A₁(ζ) (DLMF 10.20.10) の ζ = 0 テイラー係数 (ζ⁰..ζ²; A₁(0) = -1/225)
        private static readonly double[] AiryTypeA1Taylor = [-0.0044444444444444444, -0.0014637074635031450, 0.00070641727241968957];

        // Airy 型一様漸近展開の B₀(ζ) (DLMF 10.20.11) の ζ = 0 テイラー係数 (ζ⁰..ζ⁵; B₀(0) = 2^(1/3)/70, B₀'(0) = 2/225)
        private static readonly double[] AiryTypeB0Taylor =
        [
            0.017998872141355331, 0.0088888888888888889, 0.0016256871626835735,
            -0.00036428486521990960, -0.00030206044899922451, -0.000058443572545668709,
        ];

        // 大次数 (ν ≥ BesselLargeOrderMin、または ν ≥ DebyeMinOrder かつ x > ν) の J_ν(x), Y_ν(x) (0 < x < max(25, ν²)) を一様漸近展開で計算する。
        // 非振動域は DLMF 10.19.3、振動域は DLMF 10.19.6、ν ≥ BesselAiryTypeMinOrder の転回点付近は DLMF 10.20.4–10.20.5 を用い、それ以外の転回点付近は false を返して漸化式に委ねる
        private static bool TryBesselJYLargeOrder(double nu, double x, out double j, out double y)
        {
            j = y = 0;
            if (x <= nu)
            {
                // t = tanh α = √(1 - z²)。ν - x は x ≥ ν/2 で正確に求まる
                double z = x / nu;
                double t = Math.Sqrt((nu - x) / nu) * Math.Sqrt(1 + z);

                // g = α - tanh α = artanh t - t。小さい t では級数 t³ Σ t²ᵏ/(2k+3) で桁落ちを避ける
                double series;
                double g;
                if (t < 0.5)
                {
                    series = BesselTurningSeries(t * t);
                    g = series * t * t * t;
                }
                else
                {
                    // artanh t - t = [(ln(1+t) - t) - (ln(1-t) + t)]/2, 1 - t = z²/(1 + t)。2 項の打ち消しは 3 倍以下
                    double lnz = z >= MinNormal ? Math.Log(z) : Math.Log(x) - Math.Log(nu);
                    g = 0.5 * (Log1pmx(t) - (2 * lnz - Log1p(t) + t));
                    series = g / (t * t * t);
                }

                // ν(α - tanh α) は x が小さいとオーバーフローしうるが、そのとき J = 0, Y = -∞ が正しい
                double exponent = nu * g;
                if (exponent > BesselDebyeMinExponent)
                {
                    // 指数部の絶対誤差は値の相対誤差に直結するため、倍精度で表せる範囲 (t ≤ 0.6 程度) は double-double で再計算する。
                    // t ≥ 3/4 では ν ≥ 1e4 で指数部が 2000 を超え J = 0, Y = -∞ になる
                    (double Hi, double Lo) exponentDd = t < BesselPhaseSeriesMaxTau ? BesselExponentNearTurningPoint(nu, x) : (exponent, 0);

                    // J = e^(-ν(α - tanh α))/√(2πν tanh α) Σ U_k(coth α)/νᵏ, Y = -e^(ν(α - tanh α))/√(πν tanh α/2) Σ (-1)ᵏ U_k(coth α)/νᵏ
                    BesselDebyeSums(nu, 1 / t, imaginary: false, out double even, out double odd);
                    j = BesselScaledExp((-exponentDd.Hi, -exponentDd.Lo), 2 * Math.PI, nu, t) * (even + odd);
                    y = -BesselScaledExp(exponentDd, Math.PI / 2, nu, t) * (even - odd);
                    return true;
                }

                if (nu < BesselAiryTypeMinOrder)
                {
                    return false;
                }

                // Airy 関数の引数 ν^(2/3) ζ = (3ν g/2)^(2/3)
                double cube = Math.Cbrt(1.5 * exponent);
                BesselAiryType(nu, cube * cube, series, out j, out y);
                return true;
            }

            // τ = tan β = √(z² - 1)。x - ν は x ≤ 2ν で正確に求まる
            double ratio = x / nu;
            double tau;
            if (ratio <= 2)
            {
                double d = (x - nu) / nu;
                tau = Math.Sqrt(d) * Math.Sqrt(2 + d);
            }
            else
            {
                double inverse = 1 / ratio;
                tau = ratio * Math.Sqrt((1 - inverse) * (1 + inverse));
            }

            // 転回点付近かは θ = ν(tan β - β) で判定。小さい τ では級数 ν τ³ Σ (-τ²)ᵏ/(2k+3) で桁落ちを避ける
            double phaseSeries = tau < 0.5 ? BesselTurningSeries(-tau * tau) : (tau - Math.Atan(tau)) / (tau * tau * tau);
            double theta = nu * phaseSeries * tau * tau * tau;
            if (theta <= BesselDebyeMinExponent)
            {
                if (nu < BesselAiryTypeMinOrder)
                {
                    return false;
                }

                double cube = Math.Cbrt(1.5 * theta);
                BesselAiryType(nu, -cube * cube, phaseSeries, out j, out y);
                return true;
            }

            // 位相 ψ = θ - π/4 は θ (最大 ν 程度) の絶対誤差がそのまま値の誤差になるため double-double で求めて 2π で還元する
            double cosPsi;
            double sinPsi;
            if (tau < BesselPhaseSeriesMaxTau)
            {
                DoubleDoubleCosSin(BesselPhaseNearTurningPoint(nu, x), out double cosTheta, out double sinTheta);
                cosPsi = (cosTheta + sinTheta) * Sqrt1Over2;
                sinPsi = (sinTheta - cosTheta) * Sqrt1Over2;
            }
            else
            {
                // ψ = x - (ν/2 + 1/4)π + c。x と (ν/2 + 1/4)π は正確に還元できるため、残りの c のみ double-double で求める
                DoubleDoubleCosSin(BesselPhaseCorrection(nu, x), out double cosC, out double sinC);
                BesselHankelPhase(nu, x, out double cosChi, out double sinChi);
                cosPsi = cosChi * cosC - sinChi * sinC;
                sinPsi = sinChi * cosC + cosChi * sinC;
            }

            // J = √(2/(πν tan β)) (cos ψ Σ U₂ₖ(i cot β)/ν²ᵏ - i sin ψ Σ U₂ₖ₊₁(i cot β)/ν²ᵏ⁺¹)、Y は cos と sin を入れ替え
            BesselDebyeSums(nu, 1 / tau, imaginary: true, out double evenSum, out double oddSum);
            double amplitude = Math.Sqrt(2 / Math.PI) / (Math.Sqrt(nu) * Math.Sqrt(tau));
            j = amplitude * (cosPsi * evenSum + sinPsi * oddSum);
            y = amplitude * (sinPsi * evenSum - cosPsi * oddSum);
            return true;
        }

        private const double Sqrt1Over2 = 0.7071067811865476;

        // (artanh t - t)/t³ = Σ t²ᵏ/(2k+3) (square = t²) および (τ - arctan τ)/τ³ = Σ (-τ²)ᵏ/(2k+3) (square = -τ²) (|square| < 1/4)
        private static double BesselTurningSeries(double square)
        {
            double sum = 0;
            double power = 1;
            for (int k = 0; k < 100; k++)
            {
                double term = power / (2 * k + 3);
                sum += term;
                if (Math.Abs(term) <= 0.5 * Epsilon * Math.Abs(sum))
                {
                    break;
                }

                power *= square;
            }

            return sum;
        }

        // 転回点付近の Airy 型一様漸近展開 (DLMF 10.20.4–10.20.5; ν ≥ BesselAiryTypeMinOrder, w = ν^(2/3) ζ)。
        // φ = (4ζ/(1 - z²))^(1/4) = √2 (3 series / 2)^(1/6) により z = 1 での 0/0 を回避する
        private static void BesselAiryType(double nu, double w, double series, out double j, out double y)
        {
            double phi = Math.Sqrt(2) * Math.Sqrt(Math.Cbrt(1.5 * series));
            var (ai, aiPrime, bi, biPrime) = Airy(w);
            double inverseCbrt = 1 / Math.Cbrt(nu);
            double zeta = w * inverseCbrt * inverseCbrt;
            double a1 = 0;
            for (int n = AiryTypeA1Taylor.Length - 1; n >= 0; n--)
            {
                a1 = a1 * zeta + AiryTypeA1Taylor[n];
            }

            double b0 = 0;
            for (int n = AiryTypeB0Taylor.Length - 1; n >= 0; n--)
            {
                b0 = b0 * zeta + AiryTypeB0Taylor[n];
            }

            double leading = (1 + a1 / nu / nu) * inverseCbrt;
            double correction = b0 * Math.Pow(inverseCbrt, 5);
            j = phi * (ai * leading + aiPrime * correction);
            y = -phi * (bi * leading + biPrime * correction);
        }

        // Debye 展開の項 T_k = U_k(p)/νᵏ (k = 0, …, 13) の偶数次和 even と奇数次和 odd。
        // |p| ≥ 1 では T_k = (p³/ν)ᵏ Σₘ c_{3k-2m} p^(-2m) でオーバーフローを回避する。
        // imaginary = true では p → ip とし、偶数次に U_k(ip)、奇数次に U_k(ip)/i を加算する
        private static void BesselDebyeSums(double nu, double p, bool imaginary, out double even, out double odd)
        {
            even = 0;
            odd = 0;
            double square = imaginary ? -p * p : p * p;
            if (p >= 1)
            {
                double r = p * p * (p / nu);
                double q = 1 / square;
                double rk = 1;
                for (int k = 0; k < DebyeCoefficients.Length; k++)
                {
                    double[] c = DebyeCoefficients[k];
                    double sum = 0;
                    for (int m = k; m >= 0; m--)
                    {
                        sum = sum * q + c[3 * k - 2 * m];
                    }

                    // p³ᵏ⁻ᵖᵃʳ = (p²)^((3k - k mod 2)/2) なので、p² < 0 ではその指数が奇数のとき符号が反転する
                    double term = rk * sum;
                    if (imaginary && (3 * k - (k & 1)) / 2 % 2 == 1)
                    {
                        term = -term;
                    }

                    if ((k & 1) == 0)
                    {
                        even += term;
                    }
                    else
                    {
                        odd += term;
                    }

                    rk *= r;
                }

                return;
            }

            double scale = 1;
            for (int k = 0; k < DebyeCoefficients.Length; k++)
            {
                double[] c = DebyeCoefficients[k];
                int parity = k & 1;
                double sum = 0;
                for (int n = 3 * k; n >= parity; n -= 2)
                {
                    sum = sum * square + c[n];
                }

                double term = (parity == 1 ? p * sum : sum) * scale;
                if (parity == 0)
                {
                    even += term;
                }
                else
                {
                    odd += term;
                }

                scale /= nu;
            }
        }

        // Debye の一様漸近展開 (DLMF 10.41.3–10.41.4) による e⁻ˣ I_ν(x) = iScaled · 2^iExponent, eˣ K_ν(x) = kScaled · 2^kExponent (ν ≥ BesselLargeOrderMin, x > 0)。
        // 指数部 νη - x の絶対誤差がそのまま相対誤差になるため、y = ν/x ≤ 0.8 では BesselPhaseCorrection (alternating: true)、y > 0.8 では double-double で計算する
        private static void BesselIKDebye(double nu, double x, out double iScaled, out long iExponent, out double kScaled, out long kExponent)
        {
            double z = x / nu;
            double root = double.Hypot(1, z);
            (double Hi, double Lo) exponent;
            if (nu / x <= 0.8)
            {
                var series = BesselPhaseCorrection(nu, x, alternating: true);
                exponent = (-series.Hi, -series.Lo);
            }
            else if (z >= MinNormal)
            {
                // νη - x = ν(1/(z + √(1 + z²)) - arsinh(1/z))。2 項が O(ν) で打ち消し合うため double-double で求める
                var zz = DoubleDoubleDivide((x, 0.0), nu);
                var r = DoubleDoubleSqrt(DoubleDoubleAdd((1.0, 0.0), DoubleDoubleMultiply(zz, zz)));
                var first = DoubleDoubleDivide((1.0, 0.0), DoubleDoubleAdd(zz, r));
                var arsinh = DoubleDoubleLog(DoubleDoubleDivide(DoubleDoubleAdd((1.0, 0.0), r), zz));
                exponent = DoubleDoubleMultiply(DoubleDoubleAdd(first, (-arsinh.Hi, -arsinh.Lo)), (nu, 0.0));
            }
            else
            {
                // z が非正規化数では x と ν の対数から求める (I = 0, K = +∞)
                double lnz = Math.Log(x) - Math.Log(nu);
                exponent = (nu * (1 / (z + root) - (Log1p(root) - lnz)), 0);
            }

            BesselDebyeSums(nu, 1 / root, imaginary: false, out double even, out double odd);
            iScaled = BesselScaledExp(exponent, 2 * Math.PI, nu, root, out iExponent) * (even + odd);
            kScaled = BesselScaledExp((-exponent.Hi, -exponent.Lo), 2 / Math.PI, nu, root, out kExponent) * (even - odd);
        }

        // double-double の自然対数 (a > 0): a = m 2ᵉ (√½ ≤ m < √2) として ln m = 2 artanh((m-1)/(m+1)) を級数展開し e ln 2 を加算
        private static (double Hi, double Lo) DoubleDoubleLog((double Hi, double Lo) a)
        {
            int e = Math.ILogB(a.Hi);
            double mHi = Math.ScaleB(a.Hi, -e);
            double mLo = Math.ScaleB(a.Lo, -e);
            if (mHi >= Math.Sqrt(2))
            {
                mHi *= 0.5;
                mLo *= 0.5;
                e++;
            }

            // mHi - 1 はスターベンツの補題で正確
            var s = DoubleDoubleDivide(DoubleDoubleTwoSum(mHi - 1, mLo), DoubleDoubleAdd(DoubleDoubleTwoSum(mHi, 1), (mLo, 0.0)));
            var s2 = DoubleDoubleMultiply(s, s);
            var power = s;
            var atanh = s;
            for (int j = 1; j < 40; j++)
            {
                power = DoubleDoubleMultiply(power, s2);
                var term = DoubleDoubleDivide(power, 2 * j + 1);
                atanh = DoubleDoubleAdd(atanh, term);
                if (Math.Abs(term.Hi) <= DoubleDoubleEpsilon * Math.Abs(atanh.Hi) || term.Hi == 0)
                {
                    break;
                }
            }

            double productHi = e * Ln2;
            double productLo = Math.FusedMultiplyAdd(e, Ln2, -productHi) + e * Ln2Lo;
            return DoubleDoubleAdd((2 * atanh.Hi, 2 * atanh.Lo), DoubleDoubleNormalize(productHi, productLo));
        }

        // ln 2 - Ln2 (double-double 下位成分)
        private const double Ln2Lo = 2.3190468138462996e-17;

        // e^e / √(c ν f)。e = k ln 2 + r (|r| ≤ ln 2 / 2) に分離し、オーバーフローや非正規化数領域でも 2ᵏ 倍の 1 回だけ丸める (ν f のオーバーフロー回避のため平方根も分離)
        private static double BesselScaledExp((double Hi, double Lo) e, double c, double nu, double f)
        {
            // √(c ν f) は 1e155 以下なので、|e| > 2000 では商もオーバーフローするか 0 になる
            if (Math.Abs(e.Hi) > 2000)
            {
                return e.Hi > 0 ? double.PositiveInfinity : 0;
            }

            double value = BesselScaledExp(e, c, nu, f, out long k);
            return Math.ScaleB(value, (int)k);
        }

        // e^e / √(c ν f) = 戻り値 · 2^exponent (|e| > 1e18 では 2ᵏ が long に収まらないため直接 +∞ または 0 を返す)
        private static double BesselScaledExp((double Hi, double Lo) e, double c, double nu, double f, out long exponent)
        {
            if (Math.Abs(e.Hi) > 1e18)
            {
                exponent = 0;
                return e.Hi > 0 ? double.PositiveInfinity : 0;
            }

            // e.Hi と k ln 2 は ln 2 / 2 以内で近いため引き算は正確 (k ≠ 0 ではスターベンツの補題)
            double k = Math.Round(e.Hi / Ln2);
            double product = k * Ln2;
            double productError = Math.FusedMultiplyAdd(k, Ln2, -product);
            double r = e.Hi - product;
            double rLow = (e.Lo - productError) - k * Ln2Lo;
            exponent = (long)k;
            return Math.Exp(r) * (1 + rLow) / (Math.Sqrt(c) * Math.Sqrt(nu) * Math.Sqrt(f));
        }

        // ハンケル漸近展開の位相 χ = x - (ν/2 + 1/4)π の cos χ, sin χ。
        // x の引数還元を保ちつつ、ν/2 + 1/4 が丸められる場合は ν/2 と 1/4 にも加法定理を適用して位相ずれを防ぐ
        private static void BesselHankelPhase(double nu, double x, out double cosChi, out double sinChi)
        {
            double sinX = Math.Sin(x);
            double cosX = Math.Cos(x);
            double sinPhase;
            double cosPhase;
            double phase = nu / 2 + 0.25;

            // 和が 1 未満 (ν < 3/2) なら丸め誤差は 2⁻⁵⁴ 以下。1 以上では phase - ν/2 == 1/4 で丸めの有無を判定する
            if (phase < 1 || phase - nu / 2 == 0.25)
            {
                sinPhase = double.SinPi(phase);
                cosPhase = double.CosPi(phase);
            }
            else
            {
                double half = nu / 2;
                double sinHalf = double.SinPi(half);
                double cosHalf = double.CosPi(half);
                sinPhase = (sinHalf + cosHalf) * Sqrt1Over2;
                cosPhase = (cosHalf - sinHalf) * Sqrt1Over2;
            }

            cosChi = cosX * cosPhase + sinX * sinPhase;
            sinChi = sinX * cosPhase - cosX * sinPhase;
        }

        // 転回点側の級数を使う τ = tan β の上限 (τ² ≤ 0.5625 で約 130 項。これ以上は BesselPhaseCorrection の v = ν/x ≤ 0.8 を使う)
        private const double BesselPhaseSeriesMaxTau = 0.75;

        // 振動域の位相 θ = ν(τ - arctan τ) (τ < BesselPhaseSeriesMaxTau)
        private static (double Hi, double Lo) BesselPhaseNearTurningPoint(double nu, double x) =>
            BesselTurningPointSeries(nu, x - nu, oscillatory: true);

        // 非振動域の指数部 ν(α - tanh α) (t < BesselPhaseSeriesMaxTau)
        private static (double Hi, double Lo) BesselExponentNearTurningPoint(double nu, double x) =>
            BesselTurningPointSeries(nu, nu - x, oscillatory: false);

        // 転回点側の double-double 級数 ν s³ Σ (±s²)ᵏ/(2k+3)。
        // u = |x - ν|/ν として τ² = u(2 + u), t² = u(2 - u) とし、転回点近傍でも相対精度を保つ (s < 3/4 で約 130 項)
        private static (double Hi, double Lo) BesselTurningPointSeries(double nu, double difference, bool oscillatory)
        {
            var u = DoubleDoubleDivide((difference, 0), nu);
            var square = DoubleDoubleMultiply(u, DoubleDoubleAdd((2, 0), oscillatory ? u : (-u.Hi, -u.Lo)));
            var root = DoubleDoubleSqrt(square);
            var ratio = oscillatory ? (-square.Hi, -square.Lo) : square;
            (double Hi, double Lo) sum = (0, 0);
            (double Hi, double Lo) power = (1, 0);
            for (int k = 0; k < 400; k++)
            {
                var term = DoubleDoubleDivide(power, 2 * k + 3);
                sum = DoubleDoubleAdd(sum, term);
                if (Math.Abs(term.Hi) <= DoubleDoubleEpsilon * Math.Abs(sum.Hi))
                {
                    break;
                }

                power = DoubleDoubleMultiply(power, ratio);
            }

            return DoubleDoubleMultiply(DoubleDoubleMultiply(DoubleDoubleMultiply(square, root), sum), (nu, 0));
        }

        // 振動域の位相補正 c = ν(arcsin v - v/(1 + √(1 - v²))) = ν Σ aₖ v^(2k+1) (v = ν/x ≤ 0.8) を double-double で計算する。
        // alternating = true のときは v² → -v² とした変形ベッセル関数の指数部 x - νη = ν(arsinh v - v/(1 + √(1 + v²))) を返す
        private static (double Hi, double Lo) BesselPhaseCorrection(double nu, double x, bool alternating = false)
        {
            var v = DoubleDoubleDivide((nu, 0), x);
            var v2 = DoubleDoubleMultiply(v, v);
            var ratio = alternating ? (-v2.Hi, -v2.Lo) : v2;
            (double Hi, double Lo) sum = (0, 0);
            var power = v;
            foreach (var coefficient in BesselPhaseCoefficients)
            {
                var term = DoubleDoubleMultiply(power, coefficient);
                sum = DoubleDoubleAdd(sum, term);
                if (Math.Abs(term.Hi) <= DoubleDoubleEpsilon * Math.Abs(sum.Hi))
                {
                    break;
                }

                power = DoubleDoubleMultiply(power, ratio);
            }

            return DoubleDoubleMultiply(sum, (nu, 0));
        }

        // BesselPhaseCorrection の係数 aₖ = βₖ/((2k+1)(2k+2)) (βₖ = C(2k, k)/4ᵏ, k = 0..199; v ≤ 0.8 で約 170 項で収束)
        private static readonly (double Hi, double Lo)[] BesselPhaseCoefficients = CreateBesselPhaseCoefficients();

        private static (double Hi, double Lo)[] CreateBesselPhaseCoefficients()
        {
            var coefficients = new (double Hi, double Lo)[200];
            (double Hi, double Lo) beta = (1, 0);
            for (int k = 0; k < coefficients.Length; k++)
            {
                coefficients[k] = DoubleDoubleDivide(beta, (2.0 * k + 1) * (2.0 * k + 2));
                beta = DoubleDoubleDivide(DoubleDoubleMultiply(beta, (2.0 * k + 1, 0)), 2.0 * k + 2);
            }

            return coefficients;
        }

        // double-double 級数の打ち切り基準 (2⁻¹¹⁰)
        private const double DoubleDoubleEpsilon = 7.703719777548943e-34;

        // double-double 角度の cos, sin。2π の上位・下位成分 (TwoPiLo) で還元し、下位部分は 1 次補正で加える (角度 1e15 程度まで倍精度維持)
        private static void DoubleDoubleCosSin((double Hi, double Lo) angle, out double cos, out double sin)
        {
            // 角度が 2⁷⁰ ≈ 1e21 を超えると double-double 精度でも位相の誤差が 1 を超えるため、上位部分のみを使い |cos|, |sin| ≤ 1 を保つ
            if (!(Math.Abs(angle.Hi) < 1180591620717411303424.0))
            {
                cos = Math.Cos(angle.Hi);
                sin = Math.Sin(angle.Hi);
                return;
            }

            const double twoPiHi = 2 * Math.PI;
            double k = Math.Round(angle.Hi / twoPiHi);
            double product = k * twoPiHi;
            double productError = Math.FusedMultiplyAdd(k, twoPiHi, -product);

            // angle.Hi と k·2π は π 以内で近いため引き算は正確 (k ≠ 0 ではスターベンツの補題)
            var reduced = DoubleDoubleTwoSum(angle.Hi - product, (angle.Lo - productError) - k * TwoPiLo);
            double cosHi = Math.Cos(reduced.Hi);
            double sinHi = Math.Sin(reduced.Hi);
            cos = cosHi - sinHi * reduced.Lo;
            sin = sinHi + cosHi * reduced.Lo;
        }

        // Knuth の TwoSum
        private static (double Hi, double Lo) DoubleDoubleTwoSum(double a, double b)
        {
            double s = a + b;
            double bb = s - a;
            return (s, (a - (s - bb)) + (b - bb));
        }

        // Dekker の Fast2Sum (|hi| ≥ |lo|)
        private static (double Hi, double Lo) DoubleDoubleNormalize(double hi, double lo)
        {
            double s = hi + lo;
            return (s, lo - (s - hi));
        }

        private static (double Hi, double Lo) DoubleDoubleAdd((double Hi, double Lo) a, (double Hi, double Lo) b)
        {
            var (s, e) = DoubleDoubleTwoSum(a.Hi, b.Hi);
            return DoubleDoubleNormalize(s, e + a.Lo + b.Lo);
        }

        private static (double Hi, double Lo) DoubleDoubleMultiply((double Hi, double Lo) a, (double Hi, double Lo) b)
        {
            double p = a.Hi * b.Hi;
            double e = Math.FusedMultiplyAdd(a.Hi, b.Hi, -p) + (a.Hi * b.Lo + a.Lo * b.Hi);
            return DoubleDoubleNormalize(p, e);
        }

        private static (double Hi, double Lo) DoubleDoubleDivide((double Hi, double Lo) a, double b)
        {
            double q = a.Hi / b;
            double remainder = Math.FusedMultiplyAdd(-q, b, a.Hi) + a.Lo;
            return DoubleDoubleNormalize(q, remainder / b);
        }

        private static (double Hi, double Lo) DoubleDoubleSqrt((double Hi, double Lo) a)
        {
            double s = Math.Sqrt(a.Hi);
            double square = s * s;
            double squareError = Math.FusedMultiplyAdd(s, s, -square);
            return DoubleDoubleNormalize(s, ((a.Hi - square) - squareError + a.Lo) / (2 * s));
        }
    }
}
