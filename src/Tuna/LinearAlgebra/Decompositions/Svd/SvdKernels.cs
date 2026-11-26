namespace Tuna.LinearAlgebra;

/// <summary>
/// 特異値分解の計算本体（<c>m ≥ n</c>）
/// </summary>
internal static class SvdKernels
{
    // LAPACK dgesvd の mnthr と同じ 1.6
    /// <summary>
    /// 先にQR分解する縦横比の閾値
    /// </summary>
    private const double QrCrossover = 1.6;

    /// <summary>
    /// 計算結果（<c>U</c> は <c>m × n</c>、<c>V</c> は <c>n × n</c>、列優先）
    /// </summary>
    internal readonly record struct Result(double[] Values, double[]? UData, double[]? VData);

    /// <summary>
    /// <c>m × n</c>（<c>m ≥ n</c>）の <paramref name="a"/>（上書きされる）の特異値分解
    /// </summary>
    internal static Result Compute(int m, int n, double[] a, bool vectors)
    {
        var values = new double[n];
        if (n == 0)
        {
            return new Result(values, vectors ? new double[0] : null, vectors ? new double[0] : null);
        }

        double[]? u = vectors ? new double[m * n] : null;
        double[]? v = vectors ? new double[n * n] : null;
        Compute(m, n, a.AsSpan(0, m * n), values, u, v, vectors);
        return new Result(values, u, v);
    }

    // 作業領域は配列プールから借りる（二段階の縮約を除く）。
    /// <summary>
    /// <c>m × n</c>（<c>m ≥ n &gt; 0</c>）の <paramref name="a"/>（上書きされる）の特異値分解を、0 で始める領域 <paramref name="values"/>（長さ n）・
    /// <paramref name="u"/>（<c>m × n</c>）・<paramref name="v"/>（<c>n × n</c>）に書く
    /// </summary>
    internal static void Compute(int m, int n, Span<double> a, Span<double> values, Span<double> u, Span<double> v, bool vectors)
    {
        double scale = Scaling.ScaleGeneral(m, n, a, m);
        if (m >= QrCrossover * n && n > 1)
        {
            ComputeViaQr(m, n, a, vectors, values, u, v);
        }
        else
        {
            ComputeDirect(m, n, a, vectors, values, u, v);
        }

        if (scale != 1)
        {
            Level1Kernels.Scale(1.0 / scale, values);
        }
    }

    private static void ComputeDirect(int m, int n, Span<double> a, bool vectors, Span<double> values, Span<double> u, Span<double> v)
    {
        if (LinearAlgebraSettings.UseTwoStage(n, TwoStageBidiagonal.Bandwidth, vectors))
        {
            ComputeTwoStage(m, n, a, vectors, values, u, v);
            return;
        }

        int eLength = Math.Max(n - 1, 1);
        double[] rented = WorkArrayPool<double>.Rent(checked(eLength + 2 * n));
        try
        {
            var work = rented.AsSpan(0, eLength + 2 * n);
            work.Clear();
            var e = work[..eLength];
            var tauq = work.Slice(eLength, n);
            var taup = work.Slice(eLength + n, n);
            BidiagonalReduction.Reduce(m, n, a, m, values, e, tauq, taup);

            bool ok;
            if (vectors && n > BidiagonalDivideConquer.LeafSize)
            {
                // 二重対角行列の特異ベクトルを分割統治法で求め、反射を直接掛けて U = Q·[U_B; 0]、V = P·V_B に戻す
                ok = BidiagonalDivideConquer.Solve(n, values, e, u, m, v, n);
                QrKernels.ApplyQ(m, n, n, a, m, tauq, u, m);
                BidiagonalReduction.ApplyP(n, n, a, m, taup, v, n);
            }
            else
            {
                if (vectors)
                {
                    BidiagonalReduction.FormQ(m, n, a, m, tauq, u, m);
                    BidiagonalReduction.FormP(n, a, m, taup, v, n);
                }

                ok = BidiagonalSvd.Solve(n, values, e, u, vectors ? m : 0, m, v, vectors ? n : 0, n);
            }

            if (!ok)
            {
                Fail(values, u, v);
            }
        }
        finally
        {
            WorkArrayPool<double>.Return(rented);
        }
    }

    /// <summary>
    /// 二段階の縮約（<see cref="TwoStageBidiagonal"/>）による分解
    /// </summary>
    private static void ComputeTwoStage(int m, int n, Span<double> a, bool vectors, Span<double> values, Span<double> u, Span<double> v)
    {
        int b = TwoStageBidiagonal.Bandwidth;
        var e = new double[Math.Max(n - 1, 1)];
        var tauq = new double[n];
        int pn = Math.Max(n - b, 1);
        var p = vectors ? new double[pn * pn] : [];
        var taup = new double[pn];
        var left = vectors ? new BandReflectors(n, b, TwoStageBidiagonal.SweepCount(n)) : null;
        var right = vectors ? new BandReflectors(n, b, TwoStageBidiagonal.SweepCount(n)) : null;
        TwoStageBidiagonal.Reduce(m, n, b, a, m, values, e, tauq, p, taup, left, right);

        bool ok;
        if (vectors)
        {
            // U = Q₁Q₂·[U_B; 0]、V = P₁P₂·V_B
            ok = BidiagonalDivideConquer.Solve(n, values, e, u, m, v, n);
            left!.Apply(n, u, m);
            QrKernels.ApplyQ(m, n, n, a, m, tauq, u, m);
            right!.Apply(n, v, n);
            TwoStageBidiagonal.ApplyP1(n, b, n, p, taup, v, n);
        }
        else
        {
            ok = BidiagonalSvd.Solve(n, values, e, null, 0, m, null, 0, n);
        }

        if (!ok)
        {
            Fail(values, u, v);
        }
    }

    private static void ComputeViaQr(int m, int n, Span<double> a, bool vectors, Span<double> values, Span<double> u, Span<double> v)
    {
        // 作業領域: R（n × n）、R の左特異ベクトル（n × n、ベクトルを求めるときだけ）、τ（n）。
        // 配列プールは長さを2の冪に切り上げた配列を返すので、τ は別に借り、行列の領域を n² の倍数にする（n が2の冪のとき、τ を足すと2倍の長さになる）。
        int length = checked((vectors ? 2 : 1) * n * n);
        double[] rented = WorkArrayPool<double>.Rent(length);
        double[] rentedTau = WorkArrayPool<double>.Rent(n);
        try
        {
            var work = rented.AsSpan(0, length);
            work.Clear();
            var tau = rentedTau.AsSpan(0, n);
            tau.Clear();
            var r = work[..(n * n)];
            var innerU = vectors ? work.Slice(n * n, n * n) : default;
            QrKernels.Factor(m, n, a, m, tau);
            for (int j = 0; j < n; j++)
            {
                a.Slice(j * m, j + 1).CopyTo(r[(j * n)..]);
            }

            ComputeDirect(n, n, r, vectors, values, innerU, v);
            if (!vectors)
            {
                return;
            }

            // U = Q·[U_R; 0]
            Level1Kernels.Copy(n, n, innerU, n, u, m);
            QrKernels.ApplyQ(m, n, n, a, m, tau, u, m);
        }
        finally
        {
            WorkArrayPool<double>.Return(rentedTau);
            WorkArrayPool<double>.Return(rented);
        }
    }

    /// <summary>
    /// 収束しなかった結果を NaN にする
    /// </summary>
    private static void Fail(Span<double> values, Span<double> u, Span<double> v)
    {
        values.Fill(double.NaN);
        u.Fill(double.NaN);
        v.Fill(double.NaN);
    }
}
