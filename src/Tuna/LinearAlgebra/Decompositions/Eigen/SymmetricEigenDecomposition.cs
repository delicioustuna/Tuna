namespace Tuna.LinearAlgebra;

/// <summary>
/// Represents the eigendecomposition <c>A = VΛVᵀ</c> of a real symmetric matrix.
/// <para>実対称行列の固有値分解 <c>A = VΛVᵀ</c> を表します。</para>
/// </summary>
/// <remarks>
/// <para>Only the lower triangular part of the input matrix is read. Eigenvalues in <see cref="Values"/> are sorted in ascending order, and the corresponding orthonormal eigenvectors form the columns of <see cref="Vectors"/>.</para>
/// <para>入力行列の下三角部分のみを参照します。<see cref="Values"/> の固有値は昇順に並び、対応する正規直交固有ベクトルが <see cref="Vectors"/> の列に格納されます。</para>
/// </remarks>
/// <seealso href="https://www.netlib.org/lapack/explore-html/d8/d3c/group__heev.html"/>
public sealed class SymmetricEigenDecomposition
{
    private SymmetricEigenDecomposition(double[] values, Matrix<double>? vectors)
    {
        Values = values;
        Vectors = vectors;
    }

    /// <summary>
    /// Gets the eigenvalues in ascending order.
    /// <para>昇順に並んだ固有値を取得します。</para>
    /// </summary>
    /// <remarks>
    /// <para><see cref="Recompute"/> overwrites this same array instance in place.</para>
    /// <para><see cref="Recompute"/> は同じ配列インスタンスをインプレースで上書きします。</para>
    /// </remarks>
    public double[] Values { get; }

    /// <summary>
    /// Gets the orthogonal matrix <c>V</c> whose columns are the eigenvectors, or <see langword="null"/> if eigenvectors were not computed.
    /// <para>固有ベクトルを列とする直交行列 <c>V</c> を取得します（固有ベクトルを計算しなかった場合は <see langword="null"/>）。</para>
    /// </summary>
    /// <remarks>
    /// <para><see cref="Recompute"/> overwrites this same matrix instance in place.</para>
    /// <para><see cref="Recompute"/> は同じ行列インスタンスをインプレースで上書きします。</para>
    /// </remarks>
    public Matrix<double>? Vectors { get; }

    /// <summary>
    /// Computes the eigendecomposition of a real symmetric matrix.
    /// <para>実対称行列の固有値分解を計算します。</para>
    /// </summary>
    /// <param name="a">The input symmetric matrix (only the lower triangle is read; unmodified when <paramref name="overwriteInput"/> is <see langword="false"/>). / 入力対称行列（下三角部分のみ参照。<paramref name="overwriteInput"/> が <see langword="false"/> の場合は変更されません）。</param>
    /// <param name="computeVectors"><see langword="true"/> to also compute eigenvectors; <see langword="false"/> to compute eigenvalues only. / 固有ベクトルも計算する場合は <see langword="true"/>、固有値のみの場合は <see langword="false"/>。</param>
    /// <param name="overwriteInput"><see langword="true"/> to reuse <paramref name="a"/>'s backing array as workspace (destroying its contents); <see langword="false"/> to copy <paramref name="a"/>. / <paramref name="a"/> の格納領域をワークスペースとして再利用する（元の内容は破壊されます）場合は <see langword="true"/>。</param>
    /// <returns>The computed symmetric eigendecomposition. / 対称固有値分解結果。</returns>
    /// <exception cref="ArgumentException"><paramref name="a"/> is not square. / <paramref name="a"/> が正方行列でない場合。</exception>
    /// <exception cref="OverflowException">The workspace element count required for computing eigenvectors on a very large matrix (e.g., <c>2n² + 8n</c> for divide-and-conquer, overflowing at <c>n &gt;= 32767</c>) exceeds <see cref="int.MaxValue"/>. / 固有ベクトルを求める大規模行列で、ワークスペース要素数（分割統治法では <c>2n² + 8n</c>、<c>n &gt;= 32767</c> で超過）が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static SymmetricEigenDecomposition Compute(Matrix<double> a, bool computeVectors = true, bool overwriteInput = false)
    {
        if (!a.IsSquare)
        {
            throw new ArgumentException("The matrix must be square.", nameof(a));
        }

        int n = a.Rows;
        var values = new double[n];
        if (n == 0)
        {
            return new SymmetricEigenDecomposition(values, computeVectors ? new Matrix<double>(0, 0) : null);
        }

        var work = overwriteInput ? a.Data : (double[])a.Data.Clone();
        var result = new SymmetricEigenDecomposition(values, computeVectors ? Matrix<double>.Wrap(n, n, new double[n * n]) : null);
        result.Decompose(work.AsSpan(0, n * n), new double[n], new double[Math.Max(n - 1, 1)]);
        return result;
    }

    /// <summary>
    /// Computes the eigendecomposition of a real symmetric matrix view.
    /// <para>実対称行列ビューの固有値分解を計算します。</para>
    /// </summary>
    /// <param name="a">The input symmetric matrix view (only the lower triangle is read; not modified). / 入力対称行列ビュー（下三角部分のみ参照し、変更しません）。</param>
    /// <param name="computeVectors"><see langword="true"/> to also compute eigenvectors; <see langword="false"/> to compute eigenvalues only. / 固有ベクトルも計算する場合は <see langword="true"/>、固有値のみの場合は <see langword="false"/>。</param>
    /// <returns>The computed symmetric eigendecomposition. / 対称固有値分解結果。</returns>
    /// <exception cref="ArgumentException"><paramref name="a"/> is not square. / <paramref name="a"/> が正方行列でない場合。</exception>
    /// <exception cref="OverflowException">The workspace element count required for computing eigenvectors on a very large matrix exceeds <see cref="int.MaxValue"/>. / 固有ベクトルを求める大規模行列でワークスペース要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static SymmetricEigenDecomposition Compute(ReadOnlyMatrixSpan<double> a, bool computeVectors = true)
    {
        if (a.Rows != a.Columns)
        {
            throw new ArgumentException("The matrix must be square.", nameof(a));
        }

        int n = a.Rows;
        var values = new double[n];
        if (n == 0)
        {
            return new SymmetricEigenDecomposition(values, computeVectors ? new Matrix<double>(0, 0) : null);
        }

        var work = GC.AllocateUninitializedArray<double>(checked(n * n));
        StridedKernels.CopyIntoFactorStorage(a, work);
        var result = new SymmetricEigenDecomposition(values, computeVectors ? Matrix<double>.Wrap(n, n, new double[n * n]) : null);
        result.Decompose(work.AsSpan(0, n * n), new double[n], new double[Math.Max(n - 1, 1)]);
        return result;
    }

    /// <summary>
    /// Recomputes the eigendecomposition for another symmetric matrix <paramref name="a"/> of the same order, overwriting <see cref="Values"/> and <see cref="Vectors"/> in place.
    /// <para>同じ次数の別の対称行列 <paramref name="a"/> を再分解し、<see cref="Values"/> と <see cref="Vectors"/> をインプレースで更新します。</para>
    /// </summary>
    /// <param name="a">A symmetric matrix or view of the same order (only the lower triangle is read). May alias <see cref="Vectors"/> (copied to workspace before factorization); otherwise unmodified. / 同じ次数の対称行列またはビュー（下三角部分のみ参照）。<see cref="Vectors"/> と領域を共有していても構いません（分解前にワークスペースへコピーされます）。</param>
    /// <remarks>
    /// <para>Produces bit-for-bit identical results to <see cref="Compute(Matrix{double}, bool, bool)"/> with the same eigenvector setting. Workspace buffers are reused from an internal pool, except when two-stage reduction is active (which allocates temporary buffers as in <see cref="Compute(Matrix{double}, bool, bool)"/>). Do not call concurrently with other operations on the same instance.</para>
    /// <para><see cref="Compute(Matrix{double}, bool, bool)"/> と同じ固有ベクトル設定でビット単位に同一の結果を返します。ワークスペースは内部プールから再利用します（二段階縮約が有効な場合のみ <see cref="Compute(Matrix{double}, bool, bool)"/> と同様に作業領域を確保します）。同一インスタンスに対する他の操作と同時に呼び出さないでください。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="a"/> is not square or does not match the decomposition order. / <paramref name="a"/> が正方行列でない、または次数が一致しない場合。</exception>
    public void Recompute(ReadOnlyMatrixSpan<double> a)
    {
        int n = Values.Length;
        if (a.Rows != n || a.Columns != n)
        {
            throw new ArgumentException("The matrix must be square with the same order as the decomposition.", nameof(a));
        }

        if (n == 0)
        {
            return;
        }

        int tauLength = Math.Max(n - 1, 1);
        double[] work = WorkArrayPool<double>.Rent(checked(n * n));
        double[] e = WorkArrayPool<double>.Rent(n);
        double[] tau = WorkArrayPool<double>.Rent(tauLength);
        try
        {
            // Copy input before writing results so a may alias Vectors.
            StridedKernels.CopyIntoFactorStorage(a, work.AsSpan(0, n * n));
            e.AsSpan(0, n).Clear();
            tau.AsSpan(0, tauLength).Clear();
            Vectors?.Data.AsSpan().Clear();
            Decompose(work.AsSpan(0, n * n), e.AsSpan(0, n), tau.AsSpan(0, tauLength));
        }
        finally
        {
            WorkArrayPool<double>.Return(work);
            WorkArrayPool<double>.Return(e);
            WorkArrayPool<double>.Return(tau);
        }
    }

    // Expects e (length n), tau (length max(n - 1, 1)), and Vectors storage to be zero-initialized.
    private void Decompose(Span<double> work, Span<double> e, Span<double> tau)
    {
        int n = Values.Length;
        var values = Values.AsSpan();
        bool computeVectors = Vectors is not null;
        Span<double> z = computeVectors ? Vectors!.Data : default;
        double scale = Scaling.ScaleLowerSymmetric(n, work, n);
        bool converged;
        if (LinearAlgebraSettings.UseTwoStage(n, TwoStageTridiagonal.Bandwidth, computeVectors))
        {
            int b = TwoStageTridiagonal.Bandwidth;
            var bandTau = tau[..Math.Max(n - b, 1)];
            var band = computeVectors ? new BandReflectors(n, b, TwoStageTridiagonal.SweepCount(n)) : null;
            TwoStageTridiagonal.Reduce(n, b, work, n, values, e, bandTau, band);
            if (computeVectors)
            {
                converged = TridiagonalDivideConquer.Solve(n, values, e, z, n);
                band!.Apply(n, z, n);
                TwoStageTridiagonal.ApplyQ1(n, b, n, work, n, bandTau, z, n);
            }
            else
            {
                converged = TridiagonalEigen.QlImplicit(n, values, e, z, n, false);
            }
        }
        else
        {
            TridiagonalReduction.Reduce(n, work, n, values, e, tau);
            if (computeVectors && n > TridiagonalDivideConquer.LeafSize)
            {
                converged = TridiagonalDivideConquer.Solve(n, values, e, z, n);
                TridiagonalReduction.ApplyQ(n, n, work, n, tau, z, n);
            }
            else
            {
                if (computeVectors)
                {
                    TridiagonalReduction.FormQ(n, work, n, tau, z, n);
                }

                converged = TridiagonalEigen.QlImplicit(n, values, e, z, n, computeVectors);
            }
        }

        if (!converged)
        {
            values.Fill(double.NaN);
            z.Fill(double.NaN);
        }

        if (scale != 1)
        {
            Level1Kernels.Scale(1.0 / scale, values);
        }
    }
}
