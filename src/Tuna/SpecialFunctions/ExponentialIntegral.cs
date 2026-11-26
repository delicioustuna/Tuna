namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // Ei の正の零点 x₀ = 0.37250… の上位・下位部分と、x₀ まわりのテイラー係数（n = 1, …, 30）
        private const double EiRootHi = 0.3725074107813666;
        private const double EiRootLo = 1.3140183414386028e-17;

        private static readonly double[] EiRootTaylor =
        [
            3.8962157339071672, -3.2816078663985615, 6.522376145438925, -12.969697383536516,
            27.88629796294205, -62.37880152891542, 143.5349488096751, -337.1558271787469,
            804.5318399821382, -1943.7966457234988, 4743.765650402431, -11673.463991167164,
            28926.95530543545, -72107.9458683716, 180669.55858939196, -454696.21885446657,
            1148834.5468177444, -2912721.6638508374, 7407692.958000587, -18891727.000381533,
            48300034.930860244, -123768219.00249171, 317811105.66636217, -817618569.3184928,
            2107109291.8643637, -5438996831.077285, 14060263909.95585, -36396891001.49205,
            94338595092.19165, -244811170506.643,
        ];

        /// <summary>
        /// Computes the generalized exponential integral <c>Eₙ(x)</c> for integer order <paramref name="n"/>.
        /// <para>整数次数の一般化指数積分 <c>Eₙ(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The non-negative integer order (<c>n ≥ 0</c>). / 非負の整数次数（<c>n ≥ 0</c>）。</param>
        /// <param name="x">The non-negative input value (<c>x ≥ 0</c>). / 非負の入力値（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>Eₙ(x)</c>, or <see cref="double.NaN"/> if <paramref name="n"/> &lt; 0, <paramref name="x"/> &lt; 0, or <paramref name="x"/> is <see cref="double.NaN"/>. / <c>Eₙ(x)</c> の値。<paramref name="n"/> &lt; 0、<paramref name="x"/> &lt; 0、または <paramref name="x"/> が <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>At <c>x = 0</c>, returns <see cref="double.PositiveInfinity"/> for <c>n ≤ 1</c> and <c>1 / (n - 1)</c> for <c>n ≥ 2</c>.</para>
        /// <para><c>x = 0</c> では <c>n ≤ 1</c> のとき <see cref="double.PositiveInfinity"/>、<c>n ≥ 2</c> のとき <c>1 / (n - 1)</c> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Exponential_integral#Generalization"/>
        public static double ExpIntegralE(int n, double x)
        {
            if (double.IsNaN(x) || n < 0 || x < 0)
            {
                return double.NaN;
            }

            if (x == 0)
            {
                return n <= 1 ? double.PositiveInfinity : 1.0 / (n - 1);
            }

            if (double.IsPositiveInfinity(x))
            {
                return 0;
            }

            if (n == 0)
            {
                return Math.Exp(-x) / x;
            }

            int nm1 = n - 1;
            if (x > 1)
            {
                // 連分数（Lentz 法）
                double b = x + n;
                double c = 1 / Tiny;
                double d = 1 / b;
                double h = d;
                for (int i = 1; i < MaxIterations; i++)
                {
                    double an = -(double)i * (nm1 + i);
                    b += 2;
                    d = 1 / (an * d + b);
                    c = b + an / c;
                    double delta = c * d;
                    h *= delta;
                    if (Math.Abs(delta - 1) <= Epsilon)
                    {
                        break;
                    }
                }

                return h * Math.Exp(-x);
            }

            // 級数（i = n - 1 の項だけディガンマ関数を含む）
            double sum = nm1 != 0 ? 1.0 / nm1 : -Math.Log(x) - EulerGamma;
            double factor = 1;
            for (int i = 1; i < MaxIterations; i++)
            {
                factor *= -x / i;
                double delta;
                if (i != nm1)
                {
                    delta = -factor / (i - nm1);
                }
                else
                {
                    double psi = -EulerGamma;
                    for (int k = 1; k <= nm1; k++)
                    {
                        psi += 1.0 / k;
                    }

                    delta = factor * (-Math.Log(x) + psi);
                }

                sum += delta;
                if (Math.Abs(delta) <= Math.Abs(sum) * Epsilon * 0.5)
                {
                    break;
                }
            }

            return sum;
        }

        /// <summary>
        /// Computes the exponential integral <c>Ei(x)</c>.
        /// <para>指数積分 <c>Ei(x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The input value. / 入力値。</param>
        /// <returns>The value of <c>Ei(x)</c>. / <c>Ei(x)</c> の値。</returns>
        /// <remarks>
        /// <para>Returns <c>-E₁(-x)</c> for <c>x &lt; 0</c> and <see cref="double.NegativeInfinity"/> at <c>x = 0</c>.</para>
        /// <para><c>x &lt; 0</c> では <c>-E₁(-x)</c>、<c>x = 0</c> では <see cref="double.NegativeInfinity"/> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Exponential_integral#Definitions"/>
        public static double ExpIntegralEi(double x)
        {
            if (double.IsNaN(x))
            {
                return x;
            }

            if (x == 0)
            {
                return double.NegativeInfinity;
            }

            if (x < 0)
            {
                return -ExpIntegralE(1, -x);
            }

            if (double.IsPositiveInfinity(x))
            {
                return x;
            }

            double d = (x - EiRootHi) - EiRootLo;
            if (Math.Abs(d) < 0.08)
            {
                double taylor = 0;
                for (int k = EiRootTaylor.Length - 1; k >= 0; k--)
                {
                    taylor = taylor * d + EiRootTaylor[k];
                }

                return taylor * d;
            }

            if (x <= 50)
            {
                // Ei(x) = γ + ln x + Σ xᵏ / (k k!)
                double sum = 0;
                double factor = 1;
                for (int k = 1; k < 1000; k++)
                {
                    factor *= x / k;
                    double term = factor / k;
                    sum += term;
                    if (term <= Epsilon * sum * 0.5)
                    {
                        break;
                    }
                }

                return (EulerGamma + Math.Log(x)) + sum;
            }

            double series = ExpIntegralEiAsymptoticSeries(x);
            if (x > 700)
            {
                double half = Math.Exp(x / 2);
                return half * (half / x) * series;
            }

            return Math.Exp(x) / x * series;
        }

        // 漸近展開 Ei(x) ~ (eˣ/x) Σ k!/xᵏ の和の部分（x > 50）
        private static double ExpIntegralEiAsymptoticSeries(double x)
        {
            double series = 1;
            double t = 1;
            for (int k = 1; k < 1000; k++)
            {
                double next = t * k / x;
                if (next >= t)
                {
                    break;
                }

                t = next;
                series += t;
                if (t <= Epsilon * series * 0.5)
                {
                    break;
                }
            }

            return series;
        }

        /// <summary>
        /// Computes the logarithmic integral <c>li(x) = Ei(ln x)</c>.
        /// <para>対数積分 <c>li(x) = Ei(ln x)</c> を計算します。</para>
        /// </summary>
        /// <param name="x">The non-negative input value (<c>x ≥ 0</c>). / 非負の入力値（<c>x ≥ 0</c>）。</param>
        /// <returns>The value of <c>li(x)</c>, or <see cref="double.NaN"/> if <paramref name="x"/> &lt; 0 or <see cref="double.NaN"/>. / <c>li(x)</c> の値。<paramref name="x"/> &lt; 0 または <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Returns <c>0</c> at <c>x = 0</c> and <see cref="double.NegativeInfinity"/> at <c>x = 1</c>.</para>
        /// <para><c>x = 0</c> では <c>0</c>、<c>x = 1</c> では <see cref="double.NegativeInfinity"/> を返します。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Logarithmic_integral_function"/>
        public static double LogIntegral(double x)
        {
            if (double.IsNaN(x) || x < 0)
            {
                return double.NaN;
            }

            if (x == 0)
            {
                return 0;
            }

            if (x == 1)
            {
                return double.NegativeInfinity;
            }

            return ExpIntegralEi(Math.Log(x));
        }
    }
}
