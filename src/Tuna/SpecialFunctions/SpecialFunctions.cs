using System.Globalization;
using System.Numerics;

namespace Tuna;

public static partial class XMath
{
    /// <summary>
    /// Provides real and complex special mathematical functions, including gamma, beta, Bessel, and hypergeometric functions.
    /// <para>ガンマ関数、ベータ関数、ベッセル関数、超幾何関数などの実数および複素数の特殊関数を提供します。</para>
    /// </summary>
    /// <remarks>
    /// <para>For out-of-domain inputs and poles, real-valued methods return <see cref="double.NaN"/> or signed infinity according to each function's convention rather than throwing exceptions.</para>
    /// <para>実数向けメソッドでは、定義域外の入力や極に対して例外を投げず、各関数の仕様に応じて <see cref="double.NaN"/> または符号付き無限大を返します。</para>
    /// </remarks>
    public static partial class Special
    {
        /// <summary>
        /// The Euler–Mascheroni constant <c>γ</c>.
        /// <para>オイラー・マスケローニ定数 <c>γ</c>。</para>
        /// </summary>
        public const double EulerGamma = 0.5772156649015329;

        internal const double Epsilon = 2.220446049250313e-16;
        private const double MinNormal = 2.2250738585072014e-308;

        // 連分数の Lentz 法で 0 除算を避けるための下限値
        private const double Tiny = 1e-300;
        private const int MaxIterations = 1_000_000;

        private const double LnSqrt2Pi = 0.9189385332046728;

        // ln √(2π) の倍精度表現との残差（ln √(2π) - LnSqrt2Pi）
        private const double LnSqrt2PiLo = -3.8782941580672414e-17;

        private const double LnPi = 1.1447298858494002;

        // ln π の倍精度表現との残差（ln π - LnPi）
        private const double LnPiLo = 1.0265951162707826e-17;

        private const double LnTwoPi = 1.8378770664093453;
        private const double SqrtPi = 1.772453850905516;
        private const double TwoOverSqrtPi = 1.1283791670955126;
        private const double OneOverSqrtPi = 0.5641895835477563;
        internal const double Ln2 = 0.6931471805599453;
        private const double PiSquaredOver6 = 1.6449340668482264;
        private const double InvTwoPi = 0.15915494309189535;

        // 2π の倍精度表現との残差（2π - 6.283185307179586）
        private const double TwoPiLo = 2.4492935982947064e-16;

        // e の倍精度表現との残差（e - 2.718281828459045）
        private const double ELo = 1.4456468917292502e-16;

        // ベルヌーイ数 B_{2k}（k = 0, 1, …, 30）
        private static readonly double[] BernoulliEven =
        [
            1.0, 0.16666666666666666, -0.03333333333333333, 0.023809523809523808, -0.03333333333333333,
            0.07575757575757576, -0.2531135531135531, 1.1666666666666667, -7.092156862745098,
            54.971177944862156, -529.1242424242424, 6192.123188405797, -86580.25311355312,
            1425517.1666666667, -27298231.067816094, 601580873.9006424, -15116315767.092157,
            429614643061.1667, -13711655205088.332, 488332318973593.2, -1.9296579341940068e+16,
            8.416930475736826e+17, -4.0338071854059454e+19, 2.1150748638081993e+21,
            -1.2086626522296526e+23, 7.500866746076964e+24, -5.038778101481069e+26,
            3.6528776484818122e+28, -2.849876930245088e+30, 2.3865427499683627e+32,
            -2.1399949257225335e+34,
        ];

        // リーマンゼータ関数の整数点での値 ζ(k)（k = 0, 1, …, 40）
        private static readonly double[] ZetaAtInteger =
        [
            -0.5, double.PositiveInfinity,
            1.6449340668482264, 1.2020569031595942, 1.0823232337111381, 1.03692775514337,
            1.0173430619844492, 1.008349277381923, 1.0040773561979444, 1.0020083928260821,
            1.000994575127818, 1.0004941886041194, 1.000246086553308, 1.0001227133475785,
            1.0000612481350588, 1.000030588236307, 1.0000152822594086, 1.0000076371976379,
            1.000003817293265, 1.0000019082127165, 1.0000009539620338, 1.0000004769329869,
            1.0000002384505027, 1.000000119219926, 1.000000059608189, 1.0000000298035034,
            1.0000000149015549, 1.0000000074507118, 1.000000003725334, 1.0000000018626598,
            1.0000000009313275, 1.0000000004656628, 1.000000000232831, 1.0000000001164155,
            1.0000000000582077, 1.0000000000291038, 1.000000000014552, 1.000000000007276,
            1.000000000003638, 1.000000000001819, 1.0000000000009095,
        ];

        // 正しく丸めた n!（n = 0, 1, …, 170）
        private static readonly double[] FactorialTable = CreateFactorialTable();

        private static double[] CreateFactorialTable()
        {
            var table = new double[171];
            BigInteger factorial = BigInteger.One;
            for (int n = 0; n < table.Length; n++)
            {
                if (n > 0)
                {
                    factorial *= n;
                }

                table[n] = double.Parse(factorial.ToString(CultureInfo.InvariantCulture), CultureInfo.InvariantCulture);
            }

            return table;
        }

        // partial class のファイル間ではフィールド初期化子の実行順が規定されないため、
        // 他の係数表に依存する表は、すべてのフィールド初期化子の後に実行される静的コンストラクターで生成する。
        static Special()
        {
            LogGamma1pCoefficients = CreateLogGamma1pCoefficients();
            EulerMaclaurinCoefficients = CreateEulerMaclaurinCoefficients();
            (DoubleDoubleDigammaStirlingCoefficients, DoubleDoubleLogGammaStirlingCoefficients) = CreateDoubleDoubleGammaStirlingCoefficients();
        }

        private static bool IsInteger(double x) => double.IsFinite(x) && x == Math.Floor(x);

        private static bool IsNonPositiveInteger(double x) => x <= 0 && IsInteger(x);

        // 2⁵³ 以上はすべて偶数
        private static bool IsOddInteger(double x) => Math.Abs(Math.IEEERemainder(x, 2)) == 1;

        // ln(1 + x) を桁落ちなく計算する
        internal static double Log1p(double x)
        {
            double u = 1 + x;
            if (u == 1)
            {
                return x;
            }

            // u ≥ 2⁵³ では u の丸め誤差（相対 ε/2 以下）が ln u に与える誤差は絶対 ε/2 以下で、ln u ≥ 36 に比べて無視できるので補正しない。
            // 下の補正式の途中の積 ln(u)·x は x ≳ 2.5e305 でオーバーフローする（x = +∞ もここで返る）
            if (u >= 2 / Epsilon)
            {
                return Math.Log(u);
            }

            return Math.Log(u) * x / (u - 1);
        }

        // eˣ - 1 を桁落ちなく計算する
        internal static double Expm1(double x)
        {
            if (Math.Abs(x) < 1e-5)
            {
                return x + x * x * (0.5 + x / 6);
            }

            double u = Math.Exp(x);
            if (u == 1)
            {
                return x;
            }

            double um1 = u - 1;
            if (um1 == -1)
            {
                return -1;
            }

            // u ≥ 2⁵² では u - 1 の相対誤差は u の丸め誤差と同程度で、補正 x/ln u と 1 との差は ε/(2x) 未満なので補正しない。
            // 下の補正式の途中の積 (u - 1)x は x ≳ 703 でオーバーフローする（eˣ がオーバーフローした +∞ もここで返る）
            if (u >= 1 / Epsilon)
            {
                return um1;
            }

            return um1 * x / Math.Log(u);
        }

        // ln(1 - eˣ)（x ≤ 0）を桁落ちなく計算する
        internal static double Log1mExp(double x)
        {
            // x が 0 に近いと 1 - eˣ が、大きな負の数だと ln(1 - eˣ) ≈ -eˣ が桁落ちするので、境界 -ln 2 で式を切り替える
            return x > -Ln2 ? Math.Log(-Expm1(x)) : Log1p(-Math.Exp(x));
        }

        // ln(1 + u) - u を桁落ちなく計算する
        internal static double Log1pmx(double u)
        {
            if (Math.Abs(u) >= 0.5)
            {
                return Log1p(u) - u;
            }

            // ln(1+u) = 2 artanh(y)（y = u/(2+u)）を展開すると、ln(1+u) - u = -u y + 2(y³/3 + y⁵/5 + …)
            double y = u / (2 + u);
            double y2 = y * y;
            double power = y * y2;
            double sum = 0;
            for (int k = 3; k < 200; k += 2)
            {
                double term = power / k;
                sum += term;
                if (Math.Abs(term) <= Epsilon * Math.Abs(sum))
                {
                    break;
                }

                power *= y2;
            }

            return -u * y + 2 * sum;
        }

        // exp(-x²) を x² の丸め誤差を補正して計算する
        private static double ExpMinusSquare(double x)
        {
            double hi = x * x;
            double lo = Math.FusedMultiplyAdd(x, x, -hi);
            double e = Math.Exp(-hi);
            return e - e * lo;
        }

        // exp(x²) を x² の丸め誤差を補正して計算する
        private static double ExpSquare(double x)
        {
            double hi = x * x;
            double lo = Math.FusedMultiplyAdd(x, x, -hi);
            double e = Math.Exp(hi);
            // オーバーフローした e = ∞ に負の補正を加算すると ∞ - ∞ = NaN になるため、そのまま返す
            return double.IsInfinity(e) ? e : e + e * lo;
        }

        // スターリングの公式の補正項 δ(x) = ln Γ(x) - ((x - 1/2) ln x - x + ln√(2π))（x ≥ 10 を想定）
        internal static double StirlingCorrection(double x)
        {
            double z = 1 / x;
            double z2 = z * z;
            double sum = StirlingSeriesCoefficients[^1];
            for (int k = StirlingSeriesCoefficients.Length - 2; k >= 0; k--)
            {
                sum = StirlingSeriesCoefficients[k] + z2 * sum;
            }

            return z * sum;
        }

        // δ(x) = Σ c_k x^(1-2k) の係数 B_{2k}/(2k(2k-1))（k = 1, …, 10）
        private static readonly double[] StirlingSeriesCoefficients =
        [
            1.0 / 12, -1.0 / 360, 1.0 / 1260, -1.0 / 1680, 1.0 / 1188,
            -691.0 / 360360, 1.0 / 156, -3617.0 / 122400, 43867.0 / 244188, -174611.0 / 125400,
        ];

        // mantissa · 2^exponent（指数が int の範囲を超えても飽和させる）
        private static double ScaleB(double mantissa, long exponent) =>
            Math.ScaleB(mantissa, (int)Math.Clamp(exponent, -4096, 4096));

        // n! = mantissa · 2^exponent（1 ≤ mantissa < 2）の仮数を返す。
        // n ≤ 170 は正しく丸めた表の値、それより大きい n は 1 ずつ掛けて毎回正規化する
        private static double ScaledFactorial(int n, out long exponent)
        {
            double mantissa = FactorialTable[Math.Min(n, FactorialTable.Length - 1)];
            int shift = Math.ILogB(mantissa);
            mantissa = Math.ScaleB(mantissa, -shift);
            exponent = shift;
            for (int j = FactorialTable.Length; j <= n; j++)
            {
                mantissa *= j;
                shift = Math.ILogB(mantissa);
                mantissa = Math.ScaleB(mantissa, -shift);
                exponent += shift;
            }

            return mantissa;
        }

        // xⁿ = mantissa · 2^exponent（x > 0、整数 n、1 ≤ mantissa < 2）の仮数を返す
        private static double ScaledPower(double x, long n, out long exponent)
        {
            // x = m 2ᵉ（1 ≤ m < 2）とし、mⁿ はオーバーフローやアンダーフローを起こさない次数ごとに分割して累乗する
            int e = Math.ILogB(x);
            double m = Math.ScaleB(x, -e);
            double mantissa = 1;
            exponent = n * e;
            for (long remaining = n; remaining != 0;)
            {
                long chunk = Math.Clamp(remaining, -512, 512);
                mantissa *= Math.Pow(m, chunk);
                int shift = Math.ILogB(mantissa);
                mantissa = Math.ScaleB(mantissa, -shift);
                exponent += shift;
                remaining -= chunk;
            }

            return mantissa;
        }

        // 2 つの項の和（係数が 0 の項は、値が無限大でも 0 · ∞ = NaN にせず 0 として扱う）
        private static double CombineTerms(double c1, double v1, double c2, double v2) =>
            (c1 == 0 ? 0 : c1 * v1) + (c2 == 0 ? 0 : c2 * v2);
    }
}
