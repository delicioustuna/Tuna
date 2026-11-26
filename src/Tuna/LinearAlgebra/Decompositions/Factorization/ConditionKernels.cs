using System.Numerics;
using System.Runtime.CompilerServices;

namespace Tuna.LinearAlgebra;

// どちらも分解済みの係数での求解（O(n²)）を数回使うだけで、分解（O(n³)）をやり直さない。計算は逐次で、結果はスレッド数によらない。
// 倍精度
// LAPACK dgecon・dpocon と dlacn2 に相当
/// <summary>
/// 条件数の逆数の推定と反復改良（dgerfs・dporfs に相当）の計算本体
/// </summary>
internal static class ConditionKernels
{
    // LAPACK dgerfs の ITMAX と同じ
    /// <summary>
    /// 改良の既定の最大回数
    /// </summary>
    internal const int DefaultRefinementIterations = 5;

    // LAPACK dlacn2 の ITMAX と同じ
    /// <summary>
    /// 1-ノルムの推定の反復の最大回数
    /// </summary>
    private const int MaxEstimateIterations = 5;

    // LAPACK dlamch('E')
    /// <summary>
    /// 倍精度の丸めの単位 2⁻⁵³
    /// </summary>
    private const double Epsilon = 1.1102230246251565e-16;

    // transpose が真なら Bᵀ·x。
    /// <summary>
    /// <paramref name="x"/> を <c>B·x</c>で上書きする
    /// </summary>
    internal delegate void Apply(Span<double> x, bool transpose);

    // NaN があれば NaN。
    /// <summary>
    /// 1-ノルム（列の絶対値の和の最大）<c>‖A‖₁</c>
    /// </summary>
    internal static double Norm1(int n, ReadOnlySpan<double> a, int lda)
    {
        double norm = 0;
        for (int j = 0; j < n; j++)
        {
            double sum = 0;
            foreach (double v in a.Slice(j * lda, n))
            {
                sum += Math.Abs(v);
            }

            norm = Math.Max(norm, sum);
        }

        return norm;
    }

    // NaN があれば NaN。
    /// <summary>
    /// 下三角部分で与えた対称行列の 1-ノルム（上三角部分は読まない）
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static double SymmetricNorm1(int n, ReadOnlySpan<double> a, int lda)
    {
        // 第 j 列の和は、対角より下の第 j 列と、対角より左の第 j 行（= 対角より下の各列の第 j 成分）の和
        // （作業領域は借りる。バッチでは行列ごとに呼ばれる）
        double[] rented = WorkArrayPool<double>.Rent(n);
        var sums = rented.AsSpan(0, n);
        sums.Clear();
        for (int j = 0; j < n; j++)
        {
            var column = a.Slice(j * lda, n);
            double sum = sums[j] + Math.Abs(column[j]);
            for (int i = j + 1; i < n; i++)
            {
                double v = Math.Abs(column[i]);
                sum += v;
                sums[i] += v;
            }

            sums[j] = sum;
        }

        double norm = 0;
        foreach (double sum in sums)
        {
            norm = Math.Max(norm, sum);
        }

        WorkArrayPool<double>.Return(rented);
        return norm;
    }

    // ‖A⁻¹‖₁ は下からの推定なので、条件数の逆数は真の値以上になる（通常は 3 倍以内）。次数 0 なら 1、A が NaN を含めば NaN。
    // 推定の途中でオーバーフローした（数値的に特異な）場合と A が ±∞ を含む場合は 0。
    // LAPACK dgecon・dpocon
    /// <summary>
    /// <c>‖A‖₁</c> と <c>A⁻¹</c> の掛け算 <paramref name="applyInverse"/> から、1-ノルムの条件数の逆数 <c>1 / (‖A‖₁‖A⁻¹‖₁)</c> を推定する
    /// </summary>
    internal static double ReciprocalCondition(int n, double norm, Apply applyInverse)
    {
        if (n == 0)
        {
            return 1;
        }

        if (double.IsNaN(norm))
        {
            return double.NaN;
        }

        if (norm == 0 || double.IsInfinity(norm))
        {
            return 0;
        }

        double inverseNorm = EstimateNorm1(n, applyInverse);
        if (!(inverseNorm > 0 && inverseNorm < double.PositiveInfinity))
        {
            return 0;
        }

        // 良条件でも ‖A‖₁ が最大の有限値に近いと ‖A⁻¹‖₁ は非正規数になり、その逆数だけがオーバーフローする。
        // その場合は積 ‖A‖₁‖A⁻¹‖₁（≳ 1）を先に作る（オーバーフローすれば条件数の逆数は 0 とみなせる）
        double reciprocal = 1 / inverseNorm;
        return double.IsFinite(reciprocal) ? reciprocal / norm : 1 / (norm * inverseNorm);
    }

    // N. J. Higham, "FORTRAN codes for estimating the one-norm of a real or complex matrix, with applications to condition estimation",
    // ACM Trans. Math. Softw. 14 (1988) 381–396。掛け算は高々 11 回。
    // Hager の方法を Higham が改良したもの
    // LAPACK dlacn2 と同じ手順
    /// <summary>
    /// <c>B·x</c> と <c>Bᵀ·x</c> だけから <c>‖B‖₁</c> を下から推定する
    /// </summary>
    internal static double EstimateNorm1(int n, Apply apply)
    {
        var x = new double[n];
        var signs = new sbyte[n];
        x.AsSpan().Fill(1.0 / n);
        apply(x, false);
        if (n == 1)
        {
            return Math.Abs(x[0]);
        }

        double estimate = SumAbs(x);
        SetSigns(x, signs);
        apply(x, true);
        int j = ArgMaxAbs(x);
        for (int iteration = 2; ; iteration++)
        {
            // 単位ベクトル e_j で B の第 j 列を取り出す
            Array.Clear(x);
            x[j] = 1;
            apply(x, false);
            double previous = estimate;
            estimate = SumAbs(x);

            // 符号のベクトルが繰り返した（収束）か、推定が増えなければ終える
            if (SameSigns(x, signs) || estimate <= previous)
            {
                break;
            }

            SetSigns(x, signs);
            apply(x, true);
            int last = j;
            j = ArgMaxAbs(x);
            if (x[last] == Math.Abs(x[j]) || iteration >= MaxEstimateIterations)
            {
                break;
            }
        }

        // 交代する符号のベクトル xᵢ = (-1)ⁱ(1 + i/(n-1)) で推定を補う（打ち消しで過小になる行列への対策）
        for (int i = 0; i < n; i++)
        {
            x[i] = (i % 2 == 0 ? 1 : -1) * (1 + (double)i / (n - 1));
        }

        apply(x, false);
        double alternative = 2 * SumAbs(x) / (3.0 * n);
        return alternative > estimate ? alternative : estimate;
    }

    // 残差 r = b - Ax を倍精度の約2倍の精度で計算し（Residual）、分解済みの係数で A·dx = r を解いて x += dx とする。
    // 補正 ‖dx‖∞ が ε‖x‖∞ 以下になるか、前回の半分以下に縮まらなくなるか（その補正は加えない）、maxIterations 回で終える。
    // LAPACK dgerfs・dporfs の手順
    /// <summary>
    /// 反復改良で <c>Ax = b</c> の解 <paramref name="x"/> を改良し（その場）、改良後の成分ごとの後退誤差 <c>maxᵢ |b - Ax|ᵢ / (|A||x| + |b|)ᵢ</c> を返す
    /// </summary>
    /// <param name="n">行列の次数。</param>
    /// <param name="a">係数行列。</param>
    /// <param name="lda">係数行列の列の間隔。</param>
    /// <param name="symmetricLower">真なら <c>A</c> は下三角部分で与えた対称行列（上三角部分は読まない）。</param>
    /// <param name="b">右辺の値。</param>
    /// <param name="x">解の出力先。</param>
    /// <param name="maxIterations">反復回数の上限。</param>
    /// <param name="solve">分解済みの係数で <c>A·y = r</c> を解き、<c>r</c> を <c>y</c> で上書きする。</param>
    internal static double Refine(
        int n, ReadOnlySpan<double> a, int lda, bool symmetricLower, ReadOnlySpan<double> b, Span<double> x, int maxIterations, Apply solve)
    {
        if (n == 0)
        {
            return 0;
        }

        var r = new double[n];
        var scale = new double[n];
        var tail = new double[n];
        double previous = double.PositiveInfinity;
        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            Residual(n, a, lda, symmetricLower, b, x, r, scale, tail);
            solve(r, false);
            double correction = MaxAbs(r);
            if (!(correction <= previous / 2))
            {
                break;
            }

            for (int i = 0; i < n; i++)
            {
                x[i] += r[i];
            }

            previous = correction;
            if (correction <= Epsilon * MaxAbs(x))
            {
                break;
            }
        }

        Residual(n, a, lda, symmetricLower, b, x, r, scale, tail);
        double error = 0;
        for (int i = 0; i < n; i++)
        {
            double ratio = r[i] == 0 ? 0
                : double.IsPositiveInfinity(scale[i]) && double.IsFinite(r[i]) ? ScaledRatio(n, a, lda, symmetricLower, b, x, i, r[i])
                : Math.Abs(r[i]) / scale[i];
            if (!(ratio <= error))
            {
                error = ratio;
            }
        }

        return error;
    }

    // 各成分は倍精度の約2倍の精度で計算して丸めた値になる（Ogita・Rump・Oishi の Dot2。T. Ogita, S. M. Rump, S. Oishi,
    // "Accurate sum and dot product", SIAM J. Sci. Comput. 26 (2005) 1955–1988）。
    // 解がほぼ正しいとき残差は大きな打ち消しで求まるので、倍精度のままでは改良の効果が限られる。
    // scale には |b| + |A||x| を書く（tail は作業領域）。
    // 残差 r = b - Ax を、積と和の丸め誤差を誤差のない変換（TwoProduct・TwoSum）で拾って計算し、倍精度に丸めて r に書く
    /// <summary>
    /// 残差 <c>r = b - Ax</c> を出力先に書き込む
    /// </summary>
    private static void Residual(
        int n, ReadOnlySpan<double> a, int lda, bool symmetricLower, ReadOnlySpan<double> b, ReadOnlySpan<double> x, Span<double> r, Span<double> scale, Span<double> tail)
    {
        tail.Clear();
        for (int i = 0; i < n; i++)
        {
            r[i] = b[i];
            scale[i] = Math.Abs(b[i]);
        }

        for (int j = 0; j < n; j++)
        {
            var column = a.Slice(j * lda, n);
            double xj = x[j];
            double magnitude = Math.Abs(xj);
            for (int i = symmetricLower ? j : 0; i < n; i++)
            {
                Subtract(ref r[i], ref tail[i], column[i], xj);
                scale[i] += Math.Abs(column[i]) * magnitude;
            }

            if (symmetricLower)
            {
                // 対角より下の aᵢⱼ は、対称な aⱼᵢ として第 j 行にも寄与する
                for (int i = j + 1; i < n; i++)
                {
                    Subtract(ref r[j], ref tail[j], column[i], x[i]);
                    scale[j] += Math.Abs(column[i]) * Math.Abs(x[i]);
                }
            }
        }

        for (int i = 0; i < n; i++)
        {
            r[i] += tail[i];
        }
    }

    // 分子と分母の各項を同じ2の冪 2⁻ˢ（2ˢ ≥ n + 1）で縮めて足し直すので、比は尺度によらない。
    /// <summary>
    /// 分母 <c>(|b| + |A||x|)ᵢ</c> の和だけがオーバーフローした行の後退誤差 <c>|rᵢ| / (|b| + |A||x|)ᵢ</c>
    /// </summary>
    private static double ScaledRatio(
        int n, ReadOnlySpan<double> a, int lda, bool symmetricLower, ReadOnlySpan<double> b, ReadOnlySpan<double> x, int i, double residual)
    {
        double factor = Math.ScaleB(1.0, -(BitOperations.Log2((uint)n) + 2));
        double denominator = Math.Abs(b[i]) * factor;
        for (int j = 0; j < n; j++)
        {
            // 対称行列の上三角部分 aᵢⱼ (i < j) は下三角部分の aⱼᵢ を読む
            double aij = symmetricLower && j > i ? a[j + i * lda] : a[i + j * lda];
            denominator += Math.Abs(aij) * factor * Math.Abs(x[j]);
        }

        return Math.Abs(residual) * factor / denominator;
    }

    // sum の丸め誤差と積の丸め誤差を tail に足す。
    /// <summary>
    /// <c>sum + tail</c> から <c>a·x</c> を引く
    /// </summary>
    private static void Subtract(ref double sum, ref double tail, double a, double x)
    {
        double p = a * x;
        double e = Math.FusedMultiplyAdd(a, x, -p);
        double s = sum - p;
        double z = s - sum;
        double t = (sum - (s - z)) + (-p - z);
        sum = s;
        tail += t - e;
    }

    private static double SumAbs(ReadOnlySpan<double> x)
    {
        double sum = 0;
        foreach (double v in x)
        {
            sum += Math.Abs(v);
        }

        return sum;
    }

    private static double MaxAbs(ReadOnlySpan<double> x)
    {
        double max = 0;
        foreach (double v in x)
        {
            max = Math.Max(max, Math.Abs(v));
        }

        return max;
    }

    /// <summary>
    /// 絶対値が最大の成分の番号（同じ値なら小さい番号。BLAS idamax）
    /// </summary>
    private static int ArgMaxAbs(ReadOnlySpan<double> x)
    {
        int index = 0;
        double max = Math.Abs(x[0]);
        for (int i = 1; i < x.Length; i++)
        {
            if (Math.Abs(x[i]) > max)
            {
                index = i;
                max = Math.Abs(x[i]);
            }
        }

        return index;
    }

    /// <summary>
    /// <paramref name="x"/> を符号のベクトル（0 以上なら 1、負なら -1）で置き換え、<paramref name="signs"/> に記録する
    /// </summary>
    private static void SetSigns(Span<double> x, Span<sbyte> signs)
    {
        for (int i = 0; i < x.Length; i++)
        {
            sbyte s = x[i] >= 0 ? (sbyte)1 : (sbyte)-1;
            x[i] = s;
            signs[i] = s;
        }
    }

    private static bool SameSigns(ReadOnlySpan<double> x, ReadOnlySpan<sbyte> signs)
    {
        for (int i = 0; i < x.Length; i++)
        {
            if ((x[i] >= 0 ? 1 : -1) != signs[i])
            {
                return false;
            }
        }

        return true;
    }
}
