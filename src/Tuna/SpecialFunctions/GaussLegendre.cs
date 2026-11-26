namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // 三項漸化式と、cos(π(i + 3/4)/(n + 1/2)) を初期値とするニュートン法で求める。
        /// <summary>
        /// <c>n</c> 点ガウス・ルジャンドル則（区間 <c>[-1, 1]</c>、<c>n</c> は偶数）の正の節点と重み（負の節点は符号を反転したもので、重みは同じ）
        /// </summary>
        private static (double[] Nodes, double[] Weights) CreateGaussLegendreRule(int n)
        {
            int half = n / 2;
            var nodes = new double[half];
            var weights = new double[half];
            for (int i = 0; i < half; i++)
            {
                double x = Math.Cos(Math.PI * (i + 0.75) / (n + 0.5));
                for (int iteration = 0; iteration < 100; iteration++)
                {
                    LegendreWithDerivative(n, x, out double p, out double derivative);
                    double step = p / derivative;
                    x -= step;
                    if (Math.Abs(step) <= Epsilon * 0.5)
                    {
                        break;
                    }
                }

                // 重み 2/((1 - x²) Pₙ'(x)²) は、収束した節点で導関数を求め直して計算する
                LegendreWithDerivative(n, x, out _, out double finalDerivative);
                nodes[i] = x;
                weights[i] = 2 / ((1 - x * x) * finalDerivative * finalDerivative);
            }

            return (nodes, weights);
        }

        /// <summary>
        /// ルジャンドル多項式 <c>Pₙ(x)</c> とその導関数 <c>Pₙ'(x) = n(x Pₙ - Pₙ₋₁)/(x² - 1)</c>（<c>|x| &lt; 1</c>）
        /// </summary>
        private static void LegendreWithDerivative(int n, double x, out double p, out double derivative)
        {
            double previous = 1;
            p = x;
            for (int j = 2; j <= n; j++)
            {
                double next = ((2 * j - 1) * x * p - (j - 1) * previous) / j;
                previous = p;
                p = next;
            }

            derivative = n * (x * p - previous) / (x * x - 1);
        }
    }
}
