using static Tuna.XMath.Constants;
using static Tuna.NumericConstants;

namespace Tuna;

public static partial class XMath
{
    public static partial class Special
    {
        // 受け付ける次数 n の上限（打ち切り三重対角行列の大きさはおよそ n/2）と |q| の上限（行列サイズは 8|q|^(1/4) 程度）
        private const int MathieuMaxOrder = 100000;
        private const double MathieuMaxQ = 1e12;

        // フーリエ係数の打ち切り許容比（末尾係数 / 最大係数）
        private const double MathieuTailTolerance = 1e-20;

        /// <summary>
        /// Computes the Mathieu characteristic value <c>aₙ(q)</c> for even periodic solutions.
        /// <para>Mathieu 方程式の偶関数解に対応する特性値 <c>aₙ(q)</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The integer order (<c>0 ≤ n ≤ 100000</c>). / 整数次数（<c>0 ≤ n ≤ 100000</c>）。</param>
        /// <param name="q">The real parameter (<c>|q| ≤ 10¹²</c>). / 実数パラメータ（<c>|q| ≤ 10¹²</c>）。</param>
        /// <returns>The characteristic value <c>aₙ(q)</c>, or <see cref="double.NaN"/> if <paramref name="n"/> or <paramref name="q"/> is out of range or <see cref="double.NaN"/>. / 特性値 <c>aₙ(q)</c>。<paramref name="n"/> または <paramref name="q"/> が範囲外あるいは <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Satisfies <c>aₙ(0) = n²</c>, <c>a₂ₙ(-q) = a₂ₙ(q)</c>, and <c>a₂ₙ₊₁(-q) = b₂ₙ₊₁(q)</c>.</para>
        /// <para><c>aₙ(0) = n²</c>、<c>a₂ₙ(-q) = a₂ₙ(q)</c>、<c>a₂ₙ₊₁(-q) = b₂ₙ₊₁(q)</c> を満たします。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/28.2#v"/>
        public static double MathieuA(int n, double q)
        {
            if (n < 0 || n > MathieuMaxOrder || !(Math.Abs(q) <= MathieuMaxQ))
            {
                return double.NaN;
            }

            return MathieuCharacteristic(n % 2 == 0 ? 0 : 1, n / 2, q, false, out _);
        }

        /// <summary>
        /// Computes the Mathieu characteristic value <c>bₙ(q)</c> for odd periodic solutions.
        /// <para>Mathieu 方程式の奇関数解に対応する特性値 <c>bₙ(q)</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The positive integer order (<c>1 ≤ n ≤ 100000</c>). / 正の整数次数（<c>1 ≤ n ≤ 100000</c>）。</param>
        /// <param name="q">The real parameter (<c>|q| ≤ 10¹²</c>). / 実数パラメータ（<c>|q| ≤ 10¹²</c>）。</param>
        /// <returns>The characteristic value <c>bₙ(q)</c>, or <see cref="double.NaN"/> if <paramref name="n"/> or <paramref name="q"/> is out of range or <see cref="double.NaN"/>. / 特性値 <c>bₙ(q)</c>。<paramref name="n"/> または <paramref name="q"/> が範囲外あるいは <see cref="double.NaN"/> の場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Satisfies <c>bₙ(0) = n²</c>, <c>b₂ₙ₊₂(-q) = b₂ₙ₊₂(q)</c>, and <c>b₂ₙ₊₁(-q) = a₂ₙ₊₁(q)</c>.</para>
        /// <para><c>bₙ(0) = n²</c>、<c>b₂ₙ₊₂(-q) = b₂ₙ₊₂(q)</c>、<c>b₂ₙ₊₁(-q) = a₂ₙ₊₁(q)</c> を満たします。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/28.2#v"/>
        public static double MathieuB(int n, double q)
        {
            if (n < 1 || n > MathieuMaxOrder || !(Math.Abs(q) <= MathieuMaxQ))
            {
                return double.NaN;
            }

            return n % 2 == 1 ? MathieuCharacteristic(2, (n - 1) / 2, q, false, out _) : MathieuCharacteristic(3, (n - 2) / 2, q, false, out _);
        }

        /// <summary>
        /// Computes the even Mathieu function <c>ceₙ(x, q)</c>.
        /// <para>偶関数解の Mathieu 関数 <c>ceₙ(x, q)</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The integer order (<c>0 ≤ n ≤ 100000</c>). / 整数次数（<c>0 ≤ n ≤ 100000</c>）。</param>
        /// <param name="q">The real parameter (<c>|q| ≤ 10¹²</c>). / 実数パラメータ（<c>|q| ≤ 10¹²</c>）。</param>
        /// <param name="x">The finite argument in radians. / 有限の引数（ラジアン）。</param>
        /// <returns>The value of <c>ceₙ(x, q)</c>, or <see cref="double.NaN"/> if arguments are out of range or non-finite. / <c>ceₙ(x, q)</c> の値。引数が範囲外または有限でない場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Normalized per DLMF 28.2(vi) such that <c>∫₀^(2π) ceₙ² dx = π</c>, with sign continuous in <paramref name="q"/> satisfying <c>ce₀(x, 0) = 1 / √2</c>, <c>ceₙ(x, 0) = cos(nx)</c>, and <c>ceₙ(0, q) &gt; 0</c>.</para>
        /// <para>DLMF 28.2(vi) の正規化 <c>∫₀^(2π) ceₙ² dx = π</c> に従い、符号は <paramref name="q"/> について連続で <c>ce₀(x, 0) = 1 / √2</c>、<c>ceₙ(x, 0) = cos(nx)</c>、<c>ceₙ(0, q) &gt; 0</c> を満たします。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/28.2#vi"/>
        public static double MathieuCe(int n, double q, double x)
        {
            if (n < 0 || n > MathieuMaxOrder || !(Math.Abs(q) <= MathieuMaxQ) || !double.IsFinite(x))
            {
                return double.NaN;
            }

            return MathieuFunction(n % 2 == 0 ? 0 : 1, n / 2, q, x);
        }

        /// <summary>
        /// Computes the odd Mathieu function <c>seₙ(x, q)</c>.
        /// <para>奇関数解の Mathieu 関数 <c>seₙ(x, q)</c> を計算します。</para>
        /// </summary>
        /// <param name="n">The positive integer order (<c>1 ≤ n ≤ 100000</c>). / 正の整数次数（<c>1 ≤ n ≤ 100000</c>）。</param>
        /// <param name="q">The real parameter (<c>|q| ≤ 10¹²</c>). / 実数パラメータ（<c>|q| ≤ 10¹²</c>）。</param>
        /// <param name="x">The finite argument in radians. / 有限の引数（ラジアン）。</param>
        /// <returns>The value of <c>seₙ(x, q)</c>, or <see cref="double.NaN"/> if arguments are out of range or non-finite. / <c>seₙ(x, q)</c> の値。引数が範囲外または有限でない場合は <see cref="double.NaN"/>。</returns>
        /// <remarks>
        /// <para>Normalized per DLMF 28.2(vi) such that <c>∫₀^(2π) seₙ² dx = π</c>, with sign continuous in <paramref name="q"/> satisfying <c>seₙ(x, 0) = sin(nx)</c> and <c>se'ₙ(0, q) &gt; 0</c>.</para>
        /// <para>DLMF 28.2(vi) の正規化 <c>∫₀^(2π) seₙ² dx = π</c> に従い、符号は <paramref name="q"/> について連続で <c>seₙ(x, 0) = sin(nx)</c>、<c>se'ₙ(0, q) &gt; 0</c> を満たします。</para>
        /// </remarks>
        /// <seealso href="https://dlmf.nist.gov/28.2#vi"/>
        public static double MathieuSe(int n, double q, double x)
        {
            if (n < 1 || n > MathieuMaxOrder || !(Math.Abs(q) <= MathieuMaxQ) || !double.IsFinite(x))
            {
                return double.NaN;
            }

            return n % 2 == 1 ? MathieuFunction(2, (n - 1) / 2, q, x) : MathieuFunction(3, (n - 2) / 2, q, x);
        }

        // 種類 kind（0: ce₂ᵣ, 1: ce₂ᵣ₊₁, 2: se₂ᵣ₊₁, 3: se₂ᵣ₊₂）の index 番目の関数の値
        private static double MathieuFunction(int kind, int index, double q, double x)
        {
            bool cosine = kind <= 1;
            if (Math.Abs(x) > 1e5)
            {
                // mx の double-double 還元は k = mx/2π が大きいと 2π の表現誤差が効くので、x を先に (-π, π] へ還元する。
                // Math.Sin, Math.Cos は大きな引数でも正しく還元するため、atan2 で得る角は x mod 2π を絶対誤差 ε 程度で表す
                x = Math.Atan2(Math.Sin(x), Math.Cos(x));
            }

            if (q == 0)
            {
                int m0 = MathieuOrder(kind, index);
                return m0 == 0 ? OneOverSqrt2 : MathieuHarmonic(m0, x, cosine);
            }

            MathieuCharacteristic(kind, index, q, true, out double[]? coefficients);
            double[] c = coefficients!;
            double maxAbs = 0;
            foreach (double v in c)
            {
                maxAbs = Math.Max(maxAbs, Math.Abs(v));
            }

            // 係数は高次側で急速に小さくなるため、小さい項から先に加算するよう高次側から走査する
            double sum = 0;
            for (int i = c.Length - 1; i >= 0; i--)
            {
                if (Math.Abs(c[i]) < 1e-22 * maxAbs)
                {
                    continue;
                }

                sum += c[i] * MathieuHarmonic(MathieuOrder(kind, i), x, cosine);
            }

            return sum;
        }

        // cos(mx) または sin(mx)。積 mx を double-double で作って還元することで ε|mx| の位相誤差を防ぐ
        private static double MathieuHarmonic(int m, double x, bool cosine)
        {
            double hi = m * x;
            double lo = Math.FusedMultiplyAdd(m, x, -hi);
            DoubleDoubleCosSin((hi, lo), out double cos, out double sin);
            return cosine ? cos : sin;
        }

        // 種類ごとの i 番目の係数の次数 m（0: 2i, 1・2: 2i+1, 3: 2i+2）
        private static int MathieuOrder(int kind, int i) => kind == 0 ? 2 * i : kind == 3 ? 2 * i + 2 : 2 * i + 1;

        // 三重対角行列の対角成分 m²（ce₂ᵣ₊₁, se₂ᵣ₊₁ の先頭成分のみ 1 ± q）
        private static double MathieuDiagonal(int kind, int i, double q)
        {
            double m = MathieuOrder(kind, i);
            double d = m * m;
            if (i == 0)
            {
                if (kind == 1)
                {
                    d += q;
                }
                else if (kind == 2)
                {
                    d -= q;
                }
            }

            return d;
        }

        // 種類 kind の index 番目（0 始まり）の特性値と、必要に応じ DLMF 符号に正規化したフーリエ係数を求める
        private static double MathieuCharacteristic(int kind, int index, double q, bool normalize, out double[]? coefficients)
        {
            coefficients = null;
            if (q == 0)
            {
                double m = MathieuOrder(kind, index);
                return m * m;
            }

            // 係数の広がりは大きな |q| で |q|^(1/4) に比例する（ce は z = π/2（q > 0）の近くに幅 |q|^(-1/4) で局在する）
            int size = index + 10 + (int)Math.Ceiling(8 * Math.Sqrt(Math.Sqrt(Math.Abs(q))));
            while (true)
            {
                double eigenvalue = MathieuBisection(kind, index, q, size);
                double[] vector = MathieuEigenvector(kind, q, size, eigenvalue, out double tail);
                if (tail <= MathieuTailTolerance || size > 4 * MathieuMaxOrder)
                {
                    if (normalize)
                    {
                        MathieuNormalize(kind, index, q, vector);
                        coefficients = vector;
                    }

                    return eigenvalue;
                }

                size = size + size / 2 + 10;
            }
        }

        // size 次の打ち切り行列の index 番目の固有値をスツルム列の計数による二分法で求める。
        // 初期区間は Weyl の不等式から index 番目に小さい対角成分 ± 2.5|q|
        private static double MathieuBisection(int kind, int index, double q, int size)
        {
            // index 番目に小さい対角成分：i ≥ 1 の対角成分は単調増加なので、最初の成分（1 ± q になりうる）の位置だけ考えればよい
            double first = MathieuDiagonal(kind, 0, q);
            int position = 0;
            while (position + 1 < size && MathieuDiagonal(kind, position + 1, q) < first)
            {
                position++;
            }

            double center = index < position ? MathieuDiagonal(kind, index + 1, q) : index == position ? first : MathieuDiagonal(kind, index, q);
            double radius = 2.5 * Math.Abs(q);
            double lo = center - radius;
            double hi = center + radius;
            for (int iteration = 0; iteration < 200; iteration++)
            {
                double mid = lo + 0.5 * (hi - lo);
                if (mid <= lo || mid >= hi)
                {
                    break;
                }

                if (MathieuSturmCount(kind, q, size, mid) > index)
                {
                    hi = mid;
                }
                else
                {
                    lo = mid;
                }
            }

            return lo + 0.5 * (hi - lo);
        }

        // 打ち切り行列の固有値のうち x より小さいものの個数（LDLᵀ 分解の負のピボットの数）
        private static int MathieuSturmCount(int kind, double q, int size, double x)
        {
            double q2 = q * q;
            // ピボットが 0 になったときの置き換え（LAPACK の pivmin と同じ考え方。オフ対角の 2 乗に比べて無視できる大きさ）
            double pivmin = MinNormal * Math.Max(1, q2);
            double t = MathieuDiagonal(kind, 0, q) - x;
            if (Math.Abs(t) < pivmin)
            {
                t = -pivmin;
            }

            int count = t < 0 ? 1 : 0;
            double e2 = kind == 0 ? 2 * q2 : q2;
            for (int i = 1; i < size; i++)
            {
                double m = MathieuOrder(kind, i);
                t = (m * m - x) - e2 / t;
                if (Math.Abs(t) < pivmin)
                {
                    t = -pivmin;
                }

                if (t < 0)
                {
                    count++;
                }

                e2 = q2;
            }

            return count;
        }

        // 固有値に対する単位固有ベクトルをねじれ分解（twisted factorization）で求める（tail は末尾成分と最大成分の比）
        private static double[] MathieuEigenvector(int kind, double q, int size, double eigenvalue, out double tail)
        {
            double sqrt2Q = Sqrt2 * q;
            double OffDiagonal(int i) => kind == 0 && i == 0 ? sqrt2Q : q;

            var plus = new double[size];
            var minus = new double[size];
            double guard = MinNormal * Math.Max(1, Math.Abs(q));

            // 前進：D⁺ᵢ = cᵢ - eᵢ₋₁²/D⁺ᵢ₋₁、後退：D⁻ᵢ = cᵢ - eᵢ²/D⁻ᵢ₊₁（cᵢ = dᵢ - λ）
            plus[0] = MathieuDiagonal(kind, 0, q) - eigenvalue;
            for (int i = 1; i < size; i++)
            {
                if (Math.Abs(plus[i - 1]) < guard)
                {
                    plus[i - 1] = guard;
                }

                double e = OffDiagonal(i - 1);
                plus[i] = (MathieuDiagonal(kind, i, q) - eigenvalue) - e * e / plus[i - 1];
            }

            minus[size - 1] = MathieuDiagonal(kind, size - 1, q) - eigenvalue;
            for (int i = size - 2; i >= 0; i--)
            {
                if (Math.Abs(minus[i + 1]) < guard)
                {
                    minus[i + 1] = guard;
                }

                double e = OffDiagonal(i);
                minus[i] = (MathieuDiagonal(kind, i, q) - eigenvalue) - e * e / minus[i + 1];
            }

            // γₖ = D⁺ₖ + D⁻ₖ - cₖ が最小の位置から両側へ広げる（γₖ は (T - λ)⁻¹ の対角成分の逆数で、最小の位置は固有ベクトルの大きい成分）
            int k = 0;
            double best = double.PositiveInfinity;
            for (int i = 0; i < size; i++)
            {
                double gamma = Math.Abs(plus[i] + minus[i] - (MathieuDiagonal(kind, i, q) - eigenvalue));
                if (gamma < best)
                {
                    best = gamma;
                    k = i;
                }
            }

            var x = new double[size];
            x[k] = 1;
            for (int i = k - 1; i >= 0; i--)
            {
                x[i] = -OffDiagonal(i) * x[i + 1] / plus[i];
            }

            for (int i = k + 1; i < size; i++)
            {
                x[i] = -OffDiagonal(i - 1) * x[i - 1] / minus[i];
            }

            // 成分は k から離れると急速に小さくなるため、2 乗和はアンダーフローを避けて最大成分で割ってから求める
            double maxAbs = 0;
            foreach (double v in x)
            {
                maxAbs = Math.Max(maxAbs, Math.Abs(v));
            }

            double sumSquares = 0;
            for (int i = size - 1; i >= 0; i--)
            {
                x[i] /= maxAbs;
                sumSquares += x[i] * x[i];
            }

            tail = Math.Abs(x[size - 1]);
            double norm = Math.Sqrt(sumSquares);
            for (int i = 0; i < size; i++)
            {
                x[i] /= norm;
            }

            return x;
        }

        // 単位固有ベクトルを DLMF の符号にそろえ、ce₂ᵣ の先頭成分（√2 A₀）を A₀ に戻す
        private static void MathieuNormalize(int kind, int index, double q, double[] x)
        {
            if (kind == 0)
            {
                x[0] *= OneOverSqrt2;
            }

            double s = 0;
            for (int i = x.Length - 1; i >= 0; i--)
            {
                double weight = kind == 1 || kind == 3 ? MathieuOrder(kind, i) : 1;
                if (q < 0)
                {
                    // ceₙ(0) = Σ Aₘ、se'ₙ(0) = Σ m Bₘ
                    s += (kind <= 1 ? 1 : MathieuOrder(kind, i)) * x[i];
                }
                else
                {
                    s += ((i & 1) == 0 ? weight : -weight) * x[i];
                }
            }

            bool flip = q < 0 ? s < 0 : (s < 0) == ((index & 1) == 0);
            if (flip)
            {
                for (int i = 0; i < x.Length; i++)
                {
                    x[i] = -x[i];
                }
            }
        }
    }
}
