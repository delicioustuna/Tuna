
namespace Tuna.LinearAlgebra;

// 未収束部分が大きい間は LAPACK dlaqr0 と同じく積極的早期収縮（dlaqr3）を行い、
// その窓の固有値をシフトにして複数のバルジを連ねた掃引（dlaqr5 の考え方）で反復する。
// 小さい部分は通常の二重シフト（例外シフトを含む）で処理する。
/// <summary>
/// ヘッセンベルグ行列の Francis 二重シフト QR 法と、実シューア形からの固有ベクトルの後退代入（EISPACK hqr2 相当）
/// </summary>
internal static partial class HessenbergQr
{
    private const double Epsilon = 2.220446049250313e-16;

    // vectors が真なら z（ヘッセンベルグ化の直交行列）に固有ベクトルの実形式を求める。
    // backSubstitute が偽なら後退代入をせず、H に実シューア形、z にシューアベクトルを残す。
    /// <summary>
    /// ヘッセンベルグ行列 <c>H</c>（<c>n × n</c>、上書きされる）の固有値を <paramref name="wr"/>, <paramref name="wi"/> に求める
    /// </summary>
    /// <returns>
    /// 収束したら真。
    /// </returns>
    internal static bool Solve(int nn, int low, int high, Span<double> h, Span<double> wr, Span<double> wi, Span<double> z, bool vectors, bool backSubstitute = true)
    {
        int ld = nn;
        double eps = Epsilon;
        double exshift = 0.0;
        double p = 0, q = 0, r = 0, s = 0, zz = 0, w, x, y;

        // 平衡化で孤立した固有値と、行列ノルム
        double norm = 0.0;
        for (int i = 0; i < nn; i++)
        {
            if (i < low || i > high)
            {
                wr[i] = h[i + i * ld];
                wi[i] = 0.0;
            }

            for (int j = Math.Max(i - 1, 0); j < nn; j++)
            {
                norm += Math.Abs(h[i + j * ld]);
            }
        }

        int n = high;
        int iter = 0;
        double[]? shifts = null;
        int totalIterations = 0;
        int maxIterations = 30 * Math.Max(10, high - low + 1);
        while (n >= low)
        {
            // 小さい副対角成分を探す
            int l = n;
            while (l > low)
            {
                s = Math.Abs(h[(l - 1) + (l - 1) * ld]) + Math.Abs(h[l + l * ld]);
                if (s == 0.0)
                {
                    s = norm;
                }

                if (Math.Abs(h[l + (l - 1) * ld]) < eps * s)
                {
                    break;
                }

                l--;
            }

            // 無視できる副対角成分は0にして、ブロックの境界をはっきりさせる（LAPACK dlahqr と同じ）
            if (l > low)
            {
                h[l + (l - 1) * ld] = 0.0;
            }

            if (l == n)
            {
                // 実固有値が1つ求まった
                h[n + n * ld] += exshift;
                wr[n] = h[n + n * ld];
                wi[n] = 0.0;
                n--;
                iter = 0;
            }
            else if (l == n - 1)
            {
                // 2×2 ブロック
                w = h[n + (n - 1) * ld] * h[(n - 1) + n * ld];
                p = (h[(n - 1) + (n - 1) * ld] - h[n + n * ld]) / 2.0;
                q = p * p + w;
                zz = Math.Sqrt(Math.Abs(q));
                h[n + n * ld] += exshift;
                h[(n - 1) + (n - 1) * ld] += exshift;
                x = h[n + n * ld];

                if (q >= 0)
                {
                    // 実固有値の組
                    zz = p >= 0 ? p + zz : p - zz;
                    wr[n - 1] = x + zz;
                    wr[n] = wr[n - 1];
                    if (zz != 0.0)
                    {
                        wr[n] = x - w / zz;
                    }

                    wi[n - 1] = 0.0;
                    wi[n] = 0.0;
                    x = h[n + (n - 1) * ld];
                    s = Math.Abs(x) + Math.Abs(zz);
                    p = x / s;
                    q = zz / s;
                    r = Math.Sqrt(p * p + q * q);
                    p /= r;
                    q /= r;

                    // 行の変換
                    int rowEnd = vectors ? nn : n + 1;
                    for (int j = n - 1; j < rowEnd; j++)
                    {
                        zz = h[(n - 1) + j * ld];
                        h[(n - 1) + j * ld] = q * zz + p * h[n + j * ld];
                        h[n + j * ld] = q * h[n + j * ld] - p * zz;
                    }

                    // 列の変換
                    int colStart = vectors ? 0 : l;
                    for (int i = colStart; i <= n; i++)
                    {
                        zz = h[i + (n - 1) * ld];
                        h[i + (n - 1) * ld] = q * zz + p * h[i + n * ld];
                        h[i + n * ld] = q * h[i + n * ld] - p * zz;
                    }

                    if (vectors)
                    {
                        // 変換の蓄積
                        Level1Kernels.Rotate(z.Slice(low + (n - 1) * ld, high - low + 1), z.Slice(low + n * ld, high - low + 1), q, p);
                    }
                }
                else
                {
                    // 複素共役な固有値の組
                    wr[n - 1] = x + p;
                    wr[n] = x + p;
                    wi[n - 1] = zz;
                    wi[n] = -zz;
                }

                n -= 2;
                iter = 0;
            }
            else
            {
                if (++totalIterations > maxIterations)
                {
                    return false;
                }

                // 大きな未収束部分では積極的早期収縮（AED）と、その窓の固有値をシフトにした複数回の掃引を行う。
                // 例外シフトの回（iter = 10, 20, ...）は下の通常の二重シフトに任せる
                if (n - l + 1 >= AedMinimum && (iter == 0 || iter % 10 != 0))
                {
                    int nh = high - low + 1;
                    int nsWanted = ShiftCount(nh);
                    int nw = Math.Min(WindowSize(nh), n - l + 1);
                    shifts ??= new double[2 * Math.Max(WindowSize(nh), nsWanted)];
                    int nd = AggressiveEarlyDeflation(
                        nn, low, high, l, n, nw, h, wr, wi, z, vectors, exshift, shifts, out int undeflated);
                    iter++;
                    if (nd > 0)
                    {
                        n -= nd;
                        iter = 0;
                        if (100 * nd > nw * Nibble || n - l + 1 < AedMinimum)
                        {
                            continue;
                        }
                    }

                    if (n - l + 1 < 3)
                    {
                        continue;
                    }

                    if (MultishiftSweeps(h, ld, l, n, shifts, undeflated, nsWanted, vectors ? nn : n + 1, vectors ? 0 : l, z, low, high, vectors))
                    {
                        continue;
                    }
                }

                // シフトの決定
                x = h[n + n * ld];
                y = 0.0;
                w = 0.0;
                if (l < n)
                {
                    y = h[(n - 1) + (n - 1) * ld];
                    w = h[n + (n - 1) * ld] * h[(n - 1) + n * ld];
                }

                // Wilkinson の例外シフト
                if (iter == 10)
                {
                    exshift += x;
                    for (int i = low; i <= n; i++)
                    {
                        h[i + i * ld] -= x;
                    }

                    s = Math.Abs(h[n + (n - 1) * ld]) + Math.Abs(h[(n - 1) + (n - 2) * ld]);
                    x = y = 0.75 * s;
                    w = -0.4375 * s * s;
                }

                // MATLAB の例外シフト
                if (iter == 30)
                {
                    s = (y - x) / 2.0;
                    s = s * s + w;
                    if (s > 0)
                    {
                        s = Math.Sqrt(s);
                        if (y < x)
                        {
                            s = -s;
                        }

                        s = x - w / ((y - x) / 2.0 + s);
                        for (int i = low; i <= n; i++)
                        {
                            h[i + i * ld] -= s;
                        }

                        exshift += s;
                        x = y = w = 0.964;
                    }
                }

                iter++;

                // 連続する2つの小さい副対角成分を探す
                int m = n - 2;
                while (m >= l)
                {
                    zz = h[m + m * ld];
                    r = x - zz;
                    s = y - zz;
                    p = (r * s - w) / h[(m + 1) + m * ld] + h[m + (m + 1) * ld];
                    q = h[(m + 1) + (m + 1) * ld] - zz - r - s;
                    r = h[(m + 2) + (m + 1) * ld];
                    s = Math.Abs(p) + Math.Abs(q) + Math.Abs(r);
                    p /= s;
                    q /= s;
                    r /= s;
                    if (m == l)
                    {
                        break;
                    }

                    if (Math.Abs(h[m + (m - 1) * ld]) * (Math.Abs(q) + Math.Abs(r)) <
                        eps * (Math.Abs(p) * (Math.Abs(h[(m - 1) + (m - 1) * ld]) + Math.Abs(zz) + Math.Abs(h[(m + 1) + (m + 1) * ld]))))
                    {
                        break;
                    }

                    m--;
                }

                for (int i = m + 2; i <= n; i++)
                {
                    h[i + (i - 2) * ld] = 0.0;
                    if (i > m + 2)
                    {
                        h[i + (i - 3) * ld] = 0.0;
                    }
                }

                // 行 l:n、列 m:n に対する二重シフト QR ステップ
                Sweep(h, ld, l, m, n, p, q, r, vectors ? nn : n + 1, vectors ? 0 : l, z, low, high, vectors);
            }
        }

        if (vectors && backSubstitute)
        {
            BackSubstitute(nn, low, high, h, wr, wi, z, norm);
        }

        return true;
    }

    // LAPACK の NMIN と同じ 75
    /// <summary>
    /// 積極的早期収縮を使う未収束部分の最小の大きさ
    /// </summary>
    private const int AedMinimum = 75;

    // LAPACK の NIBBLE と同じ 14
    /// <summary>
    /// 収縮した固有値の割合（%）がこれを超えたら掃引を省略して次の収縮を試す
    /// </summary>
    private const int Nibble = 14;

    // LAPACK iparmq の NS
    /// <summary>
    /// 1回の反復で使うシフトの数
    /// </summary>
    private static int ShiftCount(int nh)
    {
        int ns = nh < 30 ? 2 : nh < 60 ? 4 : nh < 150 ? 10 : nh < 590 ? Math.Max(10, nh / (int)Math.Round(Math.Log2(nh))) : nh < 3000 ? 64 : 128;
        return Math.Max(2, ns - ns % 2);
    }

    // LAPACK iparmq の NW
    /// <summary>
    /// 積極的早期収縮の窓の大きさ
    /// </summary>
    private static int WindowSize(int nh)
    {
        int ns = ShiftCount(nh);
        return nh <= 500 ? ns : 3 * ns / 2;
    }
}
