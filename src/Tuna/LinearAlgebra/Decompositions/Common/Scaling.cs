namespace Tuna.LinearAlgebra;

// LAPACK dsyev / dgesvd 相当のスケーリング（2 のべき乗でスケーリングし丸め誤差なしにオーバーフロー・アンダーフローを回避）。
internal static class Scaling
{
    private const double SafeMin = 2.2250738585072014e-308;
    private const double Epsilon = 2.220446049250313e-16;

    // 非スケーリング範囲の下限 √(safmin/ε)。
    private static readonly double LowerBound = Math.Sqrt(SafeMin / Epsilon);

    // 非スケーリング範囲の上限 √(ε/safmin)。
    private static readonly double UpperBound = Math.Sqrt(Epsilon / SafeMin);

    // 下三角部分の最大絶対値が範囲外なら 2 のべき乗でスケーリングし、掛けた倍率（未適用なら 1）を返す。
    internal static double ScaleLowerSymmetric(int n, Span<double> a, int lda)
    {
        double max = 0;
        for (int j = 0; j < n; j++)
        {
            foreach (double v in a.Slice(j + j * lda, n - j))
            {
                max = Math.Max(max, Math.Abs(v));
            }
        }

        double factor = Factor(max);
        if (factor != 1)
        {
            for (int j = 0; j < n; j++)
            {
                Level1Kernels.Scale(factor, a.Slice(j + j * lda, n - j));
            }
        }

        return factor;
    }

    // 全要素の最大絶対値が範囲外なら 2 のべき乗でスケーリングし、掛けた倍率（未適用なら 1）を返す。
    internal static double ScaleGeneral(int m, int n, Span<double> a, int lda)
    {
        double max = 0;
        for (int j = 0; j < n; j++)
        {
            foreach (double v in a.Slice(j * lda, m))
            {
                max = Math.Max(max, Math.Abs(v));
            }
        }

        double factor = Factor(max);
        if (factor != 1)
        {
            for (int j = 0; j < n; j++)
            {
                Level1Kernels.Scale(factor, a.Slice(j * lda, m));
            }
        }

        return factor;
    }

    // LAPACK dstedc / dbdsdc 相当。分割統治法の減次閾値のため、最大絶対値を [1, 2) に移す 2 のべき乗倍率を返す（0 や非有限時は 1）。
    internal static double UnitFactor(ReadOnlySpan<double> d, ReadOnlySpan<double> e)
    {
        double max = 0;
        foreach (double v in d)
        {
            max = Math.Max(max, Math.Abs(v));
        }

        foreach (double v in e)
        {
            max = Math.Max(max, Math.Abs(v));
        }

        if (max == 0 || !double.IsFinite(max))
        {
            return 1;
        }

        return Math.ScaleB(1.0, Math.Min(-Math.ILogB(max), 1023));
    }

    private static double Factor(double max)
    {
        if (max == 0 || !double.IsFinite(max))
        {
            return 1;
        }

        if (max < LowerBound)
        {
            return Math.ScaleB(1.0, Math.ILogB(LowerBound) - Math.ILogB(max));
        }

        if (max > UpperBound)
        {
            return Math.ScaleB(1.0, Math.ILogB(UpperBound) - Math.ILogB(max));
        }

        return 1;
    }
}
