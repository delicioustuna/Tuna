using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // 初期値の切り替え点（T. Minka, "Estimating a Dirichlet distribution", 2000）。
        // これ以上の y では ψ(x) = ln(x - 1/2) + O(x⁻²) から x ≈ eʸ + 1/2、未満では ψ(x) = -1/x - γ + O(x) から x ≈ -1/(y + γ) とする
        private const double InverseDigammaSwitch = -2.22;

        /// <summary>
        /// Computes the inverse of the digamma function, returning the unique positive <c>x</c> satisfying <c>ψ(x) = y</c>.
        /// <para>逆ディガンマ関数を計算し、<c>ψ(x) = y</c> を満たす唯一の正の値 <c>x</c> を返します。</para>
        /// </summary>
        /// <param name="y">The target digamma value. / ディガンマ関数の値。</param>
        /// <returns>The positive solution <c>x &gt; 0</c> to <c>ψ(x) = y</c>. / <c>ψ(x) = y</c> の正の解 <c>x &gt; 0</c>。</returns>
        /// <remarks>
        /// <para>Returns <c>0</c> for <c>-∞</c>, <see cref="double.PositiveInfinity"/> when <paramref name="y"/> is <c>+∞</c> or the solution overflows (<c>y ≳ 709.78</c>), and <see cref="double.NaN"/> for <see cref="double.NaN"/>. Negative solutions are not returned.</para>
        /// <para><c>y = -∞</c> では <c>0</c>、<c>y = +∞</c> または解がオーバーフローする領域（<c>y ≳ 709.78</c>）では <see cref="double.PositiveInfinity"/>、<see cref="double.NaN"/> では <see cref="double.NaN"/> を返します。負の解は返しません。</para>
        /// </remarks>
        /// <seealso href="https://en.wikipedia.org/wiki/Digamma_function"/>
        public static double InverseDigamma(double y)
        {
            if (double.IsNaN(y) || double.IsPositiveInfinity(y))
            {
                return y;
            }

            if (double.IsNegativeInfinity(y))
            {
                return 0;
            }

            double x;
            if (y >= InverseDigammaSwitch)
            {
                x = Math.Exp(y) + 0.5;
                if (double.IsPositiveInfinity(x))
                {
                    return x;
                }
            }
            else
            {
                x = -1 / (y + EulerGamma);
            }

            // ψ は (0, ∞) で凹かつ単調増加なので、ニュートン法の反復は1回目以降、解の左側から単調に収束する。
            // 1回目に x ≤ 0 へ行き過ぎる場合に備えて、解を挟む区間を保つ
            double lower = 0;
            double upper = double.PositiveInfinity;
            for (int iteration = 0; iteration < 100; iteration++)
            {
                double residual = Digamma(x) - y;
                if (residual == 0)
                {
                    return x;
                }

                if (residual > 0)
                {
                    upper = x;
                }
                else
                {
                    lower = x;
                }

                // x が非常に小さいと ψ₁(x) ≈ 1/x² はオーバーフローし、ステップは0になる（このとき初期値 -1/(y + γ) が既に倍精度で正しい）
                double next = x - residual / Trigamma(x);

                // 収束の判定は区間の判定より先に行う（x は直前に区間の端になっているので、収束すると next が端に一致する）
                if (Math.Abs(next - x) <= 2 * MachineEpsilon * x)
                {
                    return next;
                }

                if (!(next > lower && next < upper))
                {
                    // 区間の外に出たら、区間を対数尺度で二分する（解は何桁にもわたりうる）
                    next = double.IsPositiveInfinity(upper) ? 2 * x
                        : lower > 0 ? Math.Sqrt(lower) * Math.Sqrt(upper)
                        : 0.125 * upper;
                }

                x = next;
            }

            return x;
        }
    }
}
