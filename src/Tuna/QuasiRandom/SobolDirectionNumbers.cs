namespace Tuna.QuasiRandom;

// Sobol' 列の方向数テーブル（SobolDirectionNumbers.Data.cs）から 64 ビット方向数を展開する。
// 第 j 次元は GF(2) 上の原始多項式 p_j(x) = x^s + a_1 x^(s-1) + … + a_(s-1) x + 1 と初期値 m_1, …, m_s で定義され、
// k > s では m_k = 2 a_1 m_(k-1) ⊕ … ⊕ 2^(s-1) a_(s-1) m_(k-s+1) ⊕ 2^s m_(k-s) ⊕ m_(k-s)。第 0 次元は van der Corput 列（m_k = 1）。
internal static partial class SobolDirectionNumbers
{
    internal const int Digits = 64;

    // 原始多項式の係数を 2 進表現で格納した整数（例: x^2 + x + 1 は 7）。
    internal static int Polynomial(int dimension) => Polynomials[dimension];

    // 次元 0..dimension-1 の方向数を destination[k * dimension + j] = v_(k+1)（上位ビット詰めの m_(k+1) << (63 - k)）の配置で書き込む。
    internal static void Fill(int dimension, Span<ulong> destination)
    {
        if ((uint)dimension > (uint)Count)
        {
            throw new ArgumentOutOfRangeException(nameof(dimension));
        }

        if (destination.Length != dimension * Digits)
        {
            throw new ArgumentException("The destination length must be dimension × 64.", nameof(destination));
        }

        Span<ulong> m = stackalloc ulong[Digits];
        var polynomials = Polynomials;
        var initial = InitialValues;
        int offset = 0;
        for (int j = 0; j < dimension; j++)
        {
            int polynomial = polynomials[j];
            int degree = 31 - int.LeadingZeroCount(polynomial);
            if (degree == 0)
            {
                m.Fill(1);
            }
            else
            {
                for (int k = 0; k < degree; k++)
                {
                    m[k] = initial[offset + k];
                }

                offset += degree;
                for (int k = degree; k < Digits; k++)
                {
                    // 0 始まり添字で m[k] = m[k-s] ⊕ (m[k-s] << s) ⊕ Σ a_i (m[k-i] << i)。
                    ulong value = m[k - degree] ^ (m[k - degree] << degree);
                    for (int i = 1; i < degree; i++)
                    {
                        if (((polynomial >> (degree - i)) & 1) != 0)
                        {
                            value ^= m[k - i] << i;
                        }
                    }

                    m[k] = value;
                }
            }

            for (int k = 0; k < Digits; k++)
            {
                destination[k * dimension + j] = m[k] << (Digits - 1 - k);
            }
        }
    }
}
