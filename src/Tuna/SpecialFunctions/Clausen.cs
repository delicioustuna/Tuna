namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // クラウゼン関数の級数係数 cₖ = |B₂ₖ| / (2k (2k+1)!) (k = 1..28)。
        // Cl₂(θ) = θ - θ ln θ + Σ cₖ θ^(2k+1), Cl₂(π - x) = x ln 2 - Σ cₖ (2^(2k) - 1) x^(2k+1)。
        // 他の表に依存するため遅延初期化用に入れ子クラスへ分離している
        private static class ClausenSeries
        {
            public static readonly double[] Coefficients = Create();

            private static double[] Create()
            {
                var c = new double[29];
                for (int k = 1; k < c.Length; k++)
                {
                    c[k] = Math.Abs(BernoulliEven[k]) / (2 * k * FactorialTable[2 * k + 1]);
                }

                return c;
            }
        }

        /// <summary>
        /// Computes the Clausen function <c>Cl₂(θ) = Σₖ sin(kθ) / k² = -∫₀^θ ln |2 sin(t / 2)| dt</c>.
        /// <para>クラウゼン関数 <c>Cl₂(θ) = Σₖ sin(kθ) / k² = -∫₀^θ ln |2 sin(t / 2)| dt</c> を計算します。</para>
        /// </summary>
        /// <param name="theta">The angle in radians. / 角度（ラジアン）。</param>
        /// <returns>The value of <c>Cl₂(θ)</c> (an odd function with period <c>2π</c>), or <see cref="double.NaN"/> if <paramref name="theta"/> is <see cref="double.NaN"/> or infinite. / <c>Cl₂(θ)</c> の値（周期 <c>2π</c> の奇関数）。<paramref name="theta"/> が <see cref="double.NaN"/> または無限大の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/4.26"/>
        public static double Clausen(double theta)
        {
            if (double.IsNaN(theta) || double.IsInfinity(theta))
            {
                return double.NaN;
            }

            if (Math.Abs(theta) >= ExactPiReductionThreshold)
            {
                // 大引数では θ = kπ + r (|r| ≤ π/2) を正確に還元する。k が奇数なら Cl₂(π + r) = -Cl₂(π - r)
                int quadrant = ReduceByPiExact(theta, out double rHi, out double rLo);
                double rSign = 1;
                if (rHi < 0)
                {
                    (rHi, rLo, rSign) = (-rHi, -rLo, -1);
                }

                return (quadrant & 1) == 0 ? rSign * ClausenNearZero(rHi + rLo) : -rSign * ClausenNearPi(rHi + rLo);
            }

            // r = θ - 2πp を hi + lo で保持する (p ≠ 0 なら θ - 2π_hi·p は丸め誤差なし)
            double periods = Math.Round(theta / (2 * Math.PI));
            double hi = theta;
            double lo = 0;
            if (periods != 0)
            {
                double product = periods * (2 * PiHi);
                double productError = Math.FusedMultiplyAdd(periods, 2 * PiHi, -product);
                (hi, lo) = DoubleDoubleTwoSum(theta - product, -productError - periods * TwoPiLo);
            }

            double sign = 1;
            if (hi < 0)
            {
                (hi, lo, sign) = (-hi, -lo, -1);
            }

            if (hi <= 0.5 * Math.PI)
            {
                return sign * ClausenNearZero(hi + lo);
            }

            // x = π - |r| (PiHi - hi は丸め誤差なし)
            double x = (PiHi - hi) + (PiLo - lo);
            return sign * ClausenNearPi(x);
        }

        // Cl₂(2πt) を周回数 t で受け取る (t の整数部の除去に丸め誤差なし)
        private static double ClausenTurns(double t)
        {
            double r = t - Math.Round(t);
            double sign = 1;
            if (r < 0)
            {
                (r, sign) = (-r, -1);
            }

            return sign * (r <= 0.25 ? ClausenNearZero(2 * Math.PI * r) : ClausenNearPi(2 * Math.PI * (0.5 - r)));
        }

        // Cl₂(θ) (0 ≤ θ ≤ π/2)
        private static double ClausenNearZero(double theta)
        {
            if (theta == 0)
            {
                return 0;
            }

            double[] c = ClausenSeries.Coefficients;
            double square = theta * theta;
            double sum = 0;
            for (int k = c.Length - 1; k >= 1; k--)
            {
                sum = (sum + c[k]) * square;
            }

            return theta * (1 - Math.Log(theta) + sum);
        }

        // Cl₂(π - x) (0 ≤ x ≤ π/2)
        private static double ClausenNearPi(double x)
        {
            double[] c = ClausenSeries.Coefficients;
            double square = x * x;
            double sum = 0;
            for (int k = c.Length - 1; k >= 1; k--)
            {
                sum = (sum + c[k] * (Math.ScaleB(1.0, 2 * k) - 1)) * square;
            }

            return x * (Ln2 - sum);
        }
    }
}
