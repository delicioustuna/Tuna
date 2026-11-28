namespace Tuna.LinearAlgebra;

/// <summary>
/// 対称三重対角行列の固有値問題
/// </summary>
internal static class TridiagonalEigen
{
    // d を固有値（昇順）で、 z（n × n、null 可）を固有ベクトルを掛けたもので上書きする。
    // e は長さ n で、e[i] が (i+1, i) 成分（e[n-1] は作業用）。
    /// <summary>
    /// 陰的シフト付き QL 法（EISPACK tql2）
    /// </summary>
    /// <returns>
    /// 収束したら真。
    /// </returns>
    internal static bool QlImplicit(int n, Span<double> d, Span<double> e, Span<double> z, int ldz, bool vectors)
    {
        if (n == 0)
        {
            return true;
        }

        e[n - 1] = 0;
        double[]? rotations = vectors ? WorkArrayPool<double>.Rent(checked(2 * n)) : null;
        var rc = vectors ? rotations.AsSpan(0, n) : default;
        var rs = vectors ? rotations.AsSpan(n, n) : default;
        try
        {
            return QlLoop(n, d, e, z, ldz, vectors, rc, rs);
        }
        finally
        {
            if (rotations is not null)
            {
                WorkArrayPool<double>.Return(rotations);
            }
        }
    }

    private static bool QlLoop(int n, Span<double> d, Span<double> e, Span<double> z, int ldz, bool vectors, Span<double> rc, Span<double> rs)
    {
        double f = 0;
        double tst1 = 0;
        int maxIterations = 30 * n;
        int iterations = 0;
        for (int l = 0; l < n; l++)
        {
            tst1 = Math.Max(tst1, Math.Abs(d[l]) + Math.Abs(e[l]));
            int m = l;
            while (m < n - 1)
            {
                if (Math.Abs(e[m]) <= NumericConstants.MachineEpsilon * tst1)
                {
                    break;
                }

                m++;
            }

            if (m > l)
            {
                do
                {
                    if (++iterations > maxIterations)
                    {
                        return false;
                    }

                    // シフトの計算
                    double g = d[l];
                    double p = (d[l + 1] - g) / (2.0 * e[l]);
                    double r = Householder.Hypot(p, 1.0);
                    if (p < 0)
                    {
                        r = -r;
                    }

                    d[l] = e[l] / (p + r);
                    d[l + 1] = e[l] * (p + r);
                    double dl1 = d[l + 1];
                    double h = g - d[l];
                    for (int i = l + 2; i < n; i++)
                    {
                        d[i] -= h;
                    }

                    f += h;

                    // 陰的 QL 変換
                    p = d[m];
                    double c = 1.0;
                    double c2 = c;
                    double c3 = c;
                    double el1 = e[l + 1];
                    double s = 0.0;
                    double s2 = 0.0;
                    for (int i = m - 1; i >= l; i--)
                    {
                        c3 = c2;
                        c2 = c;
                        s2 = s;
                        g = c * e[i];
                        h = c * p;
                        r = Householder.Hypot(p, e[i]);
                        e[i + 1] = s * r;
                        s = e[i] / r;
                        c = p / r;
                        p = c * d[i] - s * g;
                        d[i + 1] = h + s * (c * g + s * d[i]);
                        if (vectors)
                        {
                            // 列 (i, i+1) への回転 (x, y) ← (c·x - s·y, c·y + s·x) を後でまとめて作用させる
                            rc[i] = c;
                            rs[i] = -s;
                        }
                    }

                    if (vectors)
                    {
                        Level1Kernels.RotateSequence(z, ldz, n, l, rc[l..m], rs[l..m], forward: false);
                    }

                    p = -s * s2 * c3 * el1 * e[l] / dl1;
                    e[l] = s * p;
                    d[l] = c * p;
                }
                while (Math.Abs(e[l]) > NumericConstants.MachineEpsilon * tst1);
            }

            d[l] += f;
            e[l] = 0.0;
        }

        SortAscending(n, d, z, ldz, vectors);
        return true;
    }

    /// <summary>
    /// 固有値を昇順に並べ替え、固有ベクトルの列も対応させる（選択ソート、列の交換は高々 n-1 回）
    /// </summary>
    internal static void SortAscending(int n, Span<double> d, Span<double> z, int ldz, bool vectors)
    {
        for (int i = 0; i < n - 1; i++)
        {
            int k = i;
            double p = d[i];
            for (int j = i + 1; j < n; j++)
            {
                if (d[j] < p)
                {
                    k = j;
                    p = d[j];
                }
            }

            if (k != i)
            {
                d[k] = d[i];
                d[i] = p;
                if (vectors)
                {
                    Level1Kernels.Swap(z.Slice(i * ldz, n), z.Slice(k * ldz, n));
                }
            }
        }
    }
}
