using System.Globalization;
using System.Numerics;
using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // BigInteger で厳密に計算して正しく丸めた整数列テーブル (遅延初期化のため入れ子クラスへ分離)
        private static class CombinatorialTables
        {
            // オイラー数 E₂ₖ (k = 0, 1, …; これ以降は倍精度でオーバーフロー)
            public static readonly double[] EulerEven = CreateEulerEven();

            // ベル数 Bₙ (n = 0, 1, …; これ以降は倍精度でオーバーフロー)
            public static readonly double[] Bell = CreateBell();

            private static double ToDouble(BigInteger value) =>
                double.Parse(value.ToString(CultureInfo.InvariantCulture), CultureInfo.InvariantCulture);

            private static double[] CreateEulerEven()
            {
                // ザイデルの牛耕式三角形でジグザグ数 Aₙ を求め、E₂ₖ = (-1)ᵏ A₂ₖ
                var values = new List<double>();
                var row = new List<BigInteger> { BigInteger.One };
                for (int n = 1; ; n++)
                {
                    var next = new List<BigInteger>(n + 1) { BigInteger.Zero };
                    for (int k = 1; k <= n; k++)
                    {
                        next.Add(next[k - 1] + row[n - k]);
                    }

                    row = next;
                    if (n % 2 == 0)
                    {
                        double value = ToDouble(row[n]);
                        if (double.IsInfinity(value))
                        {
                            break;
                        }

                        if (values.Count == 0)
                        {
                            values.Add(1);
                        }

                        values.Add(n / 2 % 2 == 0 ? value : -value);
                    }
                }

                return [.. values];
            }

            private static double[] CreateBell()
            {
                // ベルの三角形: 各行は前行の末尾から始め、左隣と左上の和を並べる。先頭要素がベル数
                var values = new List<double> { 1 };
                var row = new List<BigInteger> { BigInteger.One };
                while (true)
                {
                    var next = new List<BigInteger>(row.Count + 1) { row[^1] };
                    for (int k = 0; k < row.Count; k++)
                    {
                        next.Add(next[k] + row[k]);
                    }

                    row = next;
                    double value = ToDouble(row[0]);
                    if (double.IsInfinity(value))
                    {
                        break;
                    }

                    values.Add(value);
                }

                return [.. values];
            }
        }

        /// <summary>
        /// Computes the <paramref name="n"/>-th Euler number <c>Eₙ</c>.
        /// <para>第 <paramref name="n"/> オイラー数 <c>Eₙ</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The non-negative index. / 非負の番号。</param>
        /// <returns>The correctly rounded value of <c>Eₙ</c> (<c>E₀ = 1</c>, <c>E₂ = -1</c>, <c>E₄ = 5</c>, <c>0</c> for odd <paramref name="n"/>), <c>±∞</c> for even <c>n ≥ 188</c>, or <see cref="double.NaN"/> if <paramref name="n"/> is negative. / 正しく丸めた <c>Eₙ</c>（<c>E₀ = 1</c>、<c>E₂ = -1</c>、<c>E₄ = 5</c>、奇数番目は <c>0</c>）。偶数かつ <c>n ≥ 188</c> では <c>±∞</c>、<paramref name="n"/> が負の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/24.2.E6"/>
        public static double EulerNumber(int n)
        {
            if (n < 0)
            {
                return double.NaN;
            }

            if (n % 2 == 1)
            {
                return 0;
            }

            double[] table = CombinatorialTables.EulerEven;
            int k = n / 2;
            if (k < table.Length)
            {
                return table[k];
            }

            return k % 2 == 0 ? double.PositiveInfinity : double.NegativeInfinity;
        }

        /// <summary>
        /// Computes the <paramref name="n"/>-th Bell number <c>Bₙ</c>, the number of partitions of an <paramref name="n"/>-element set.
        /// <para>第 <paramref name="n"/> ベル数 <c>Bₙ</c>（<paramref name="n"/> 個の要素からなる集合の分割数）を計算します。</para>
        /// </summary>
        /// <param name="n">The non-negative index. / 非負の番号。</param>
        /// <returns>The correctly rounded value of <c>Bₙ</c>, <see cref="double.PositiveInfinity"/> for <c>n ≥ 219</c>, or <see cref="double.NaN"/> if <paramref name="n"/> is negative. / 正しく丸めた <c>Bₙ</c>。<c>n ≥ 219</c> では <see cref="double.PositiveInfinity"/>、<paramref name="n"/> が負の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/26.7.E1"/>
        public static double BellNumber(int n)
        {
            if (n < 0)
            {
                return double.NaN;
            }

            double[] table = CombinatorialTables.Bell;
            return n < table.Length ? table[n] : double.PositiveInfinity;
        }

        /// <summary>
        /// Computes the Stirling number of the second kind <c>S(n, k)</c>, the number of partitions of an <paramref name="n"/>-element set into <paramref name="k"/> non-empty subsets.
        /// <para>第 2 種スターリング数 <c>S(n, k)</c>（<paramref name="n"/> 個の要素を <paramref name="k"/> 個の空でない部分集合に分ける方法の数）を計算します。</para>
        /// </summary>
        /// <param name="n">The total number of elements (non-negative integer). / 要素数（非負整数）。</param>
        /// <param name="k">The number of non-empty subsets (non-negative integer). / 部分集合の個数（非負整数）。</param>
        /// <returns>The value of <c>S(n, k)</c>, <c>0</c> if <c>k &gt; n</c>, <see cref="double.PositiveInfinity"/> on overflow, or <see cref="double.NaN"/> if either argument is negative. / <c>S(n, k)</c> の値。<c>k &gt; n</c> では <c>0</c>、オーバーフロー時は <see cref="double.PositiveInfinity"/>、引数が負の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/26.8.E6"/>
        public static double StirlingSecondKind(int n, int k)
        {
            if (n < 0 || k < 0)
            {
                return double.NaN;
            }

            if (k > n)
            {
                return 0;
            }

            if (k == n)
            {
                return 1;
            }

            if (k == 0)
            {
                return 0;
            }

            if (k == 1)
            {
                return 1;
            }

            int d = n - k;
            if ((double)d * Math.Log(k) > 710 || LogBinomial(n, 2.0 * d) > 710)
            {
                return double.PositiveInfinity;
            }

            if (n >= k * Math.Log(2.0 * k))
            {
                return StirlingSecondKindAlternating(n, k);
            }

            double value = d <= k ? StirlingByEulerian(n, d, secondKind: true) : StirlingRecurrence(n, k, secondKind: true);
            return double.IsNaN(value) ? double.PositiveInfinity : value;
        }

        /// <summary>
        /// Computes the signed Stirling number of the first kind <c>s(n, k)</c>, the coefficient of <c>xᵏ</c> in the falling factorial <c>x(x - 1)…(x - n + 1)</c>.
        /// <para>符号付き第 1 種スターリング数 <c>s(n, k)</c>（下降階乗 <c>x(x - 1)…(x - n + 1)</c> における <c>xᵏ</c> の係数）を計算します。</para>
        /// </summary>
        /// <param name="n">The degree of the falling factorial (non-negative integer). / 下降階乗の次数（非負整数）。</param>
        /// <param name="k">The power of <c>x</c> (non-negative integer). / <c>x</c> の次数（非負整数）。</param>
        /// <returns>The value of <c>s(n, k)</c> with sign <c>(-1)^(n-k)</c>, <c>0</c> if <c>k &gt; n</c>, <c>±∞</c> on overflow, or <see cref="double.NaN"/> if either argument is negative. / 符号 <c>(-1)^(n-k)</c> を持つ <c>s(n, k)</c> の値。<c>k &gt; n</c> では <c>0</c>、オーバーフロー時は <c>±∞</c>、引数が負の場合は <see cref="double.NaN"/>。</returns>
        /// <seealso href="https://dlmf.nist.gov/26.8.E1"/>
        public static double StirlingFirstKind(int n, int k)
        {
            if (n < 0 || k < 0)
            {
                return double.NaN;
            }

            if (k > n)
            {
                return 0;
            }

            if (k == n)
            {
                return 1;
            }

            if (k == 0)
            {
                return 0;
            }

            int d = n - k;
            double sign = d % 2 == 0 ? 1 : -1;
            if (k == 1)
            {
                return sign * Factorial(n - 1);
            }

            if ((double)d * Math.Log(k) > 710 || LogBinomial(n, 2.0 * d) > 710)
            {
                return sign * double.PositiveInfinity;
            }

            double value = d <= k ? StirlingByEulerian(n, d, secondKind: false) : StirlingRecurrence(n, k, secondKind: false);
            return sign * (double.IsNaN(value) ? double.PositiveInfinity : value);
        }

        // S(n, k) = (kⁿ/k!) Σⱼ (-1)ʲ C(k, j) ((k-j)/k)ⁿ (n ≥ k ln(2k))
        private static double StirlingSecondKindAlternating(int n, int k)
        {
            // ((k-j)/k)ⁿ = (k-j)ⁿ/kⁿ。商を丸めてから累乗すると相対誤差が nε になるため仮数と指数に分離する
            double top = ScaledPower(k, n, out long topExponent);
            double sum = 1;
            double binomial = 1;
            for (int j = 1; j < k; j++)
            {
                binomial *= (double)(k - j + 1) / j;
                double power = ScaledPower(k - j, n, out long exponent);
                double term = binomial * ScaleB(power / top, exponent - topExponent);
                sum += (j % 2 == 0 ? term : -term);
                if (term < 0.25 * MachineEpsilon * sum)
                {
                    break;
                }
            }

            double factorial = ScaledFactorial(k, out long factorialExponent);
            return ScaleB(top / factorial * sum, topExponent - factorialExponent);
        }

        // 2 階オイラー数による和: secondKind = true なら S(n, n-d) = Σⱼ ⟨⟨d, j⟩⟩ C(n+d-1-j, 2d)、false なら |s(n, n-d)| = Σⱼ ⟨⟨d, j⟩⟩ C(n+j, 2d)
        private static double StirlingByEulerian(int n, int d, bool secondKind)
        {
            // ⟨⟨i, j⟩⟩ = (j+1)⟨⟨i-1, j⟩⟩ + (2i-1-j)⟨⟨i-1, j-1⟩⟩。和は (2d-1)!! でオーバーフローは呼び出し元で判定済み
            var eulerian = new (double Hi, double Lo)[d];
            eulerian[0] = (1, 0);
            for (int i = 2; i <= d; i++)
            {
                for (int j = i - 1; j >= 0; j--)
                {
                    var value = DoubleDoubleMultiply(eulerian[j], (j + 1, 0));
                    if (j > 0)
                    {
                        value = DoubleDoubleAdd(value, DoubleDoubleMultiply(eulerian[j - 1], (2 * i - 1 - j, 0)));
                    }

                    eulerian[j] = value;
                }
            }

            // C(N, 2d) (N = n, …, n + d - 1) を昇順に計算する (N < 2d では 0)
            int width = 2 * d;
            long first = Math.Max(n, width);
            (double Hi, double Lo) binomial = (1, 0);
            for (int i = 1; i <= width; i++)
            {
                binomial = DoubleDoubleDivide(DoubleDoubleMultiply(binomial, (first - width + i, 0)), i);
            }

            (double Hi, double Lo) sum = (0, 0);
            for (long N = first; N <= (long)n + d - 1; N++)
            {
                if (N > first)
                {
                    binomial = DoubleDoubleDivide(DoubleDoubleMultiply(binomial, (N, 0)), N - width);
                }

                int j = (int)(secondKind ? n + d - 1 - N : N - n);
                sum = DoubleDoubleAdd(sum, DoubleDoubleMultiply(eulerian[j], binomial));
            }

            return sum.Hi + sum.Lo;
        }

        // 漸化式 S(i, j) = j S(i-1, j) + S(i-1, j-1) (第 2 種) または |s(i, j)| = (i-1)|s(i-1, j)| + |s(i-1, j-1)| (第 1 種) を double-double で計算する。
        // (n, k) に寄与する k - (n - i) ≤ j ≤ k のみ更新するため計算量は O((n - k)k)
        private static double StirlingRecurrence(int n, int k, bool secondKind)
        {
            var row = new (double Hi, double Lo)[k + 1];
            row[0] = (1, 0);
            for (int i = 1; i <= n; i++)
            {
                int low = Math.Max(1, k - (n - i));
                for (int j = Math.Min(i, k); j >= low; j--)
                {
                    double factor = secondKind ? j : i - 1;
                    row[j] = DoubleDoubleAdd(DoubleDoubleMultiply(row[j], (factor, 0)), row[j - 1]);
                }

                row[0] = (0, 0);
            }

            return row[k].Hi + row[k].Lo;
        }
    }
}
