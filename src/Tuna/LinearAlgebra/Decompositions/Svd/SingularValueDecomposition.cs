namespace Tuna.LinearAlgebra;

/// <summary>
/// Represents the thin singular value decomposition <c>A = UΣVᵀ</c> of a real matrix.
/// <para>実行列の薄い特異値分解 <c>A = UΣVᵀ</c> を表します。</para>
/// </summary>
/// <remarks>
/// <para>For an <c>m × n</c> matrix <c>A</c> with <c>k = min(m, n)</c>, <see cref="U"/> is an <c>m × k</c> matrix and <see cref="V"/> is an <c>n × k</c> matrix, each with orthonormal columns.</para>
/// <para><c>m × n</c> 行列 <c>A</c> に対し <c>k = min(m, n)</c> として、<see cref="U"/> は <c>m × k</c>、<see cref="V"/> は <c>n × k</c> の正規直交列を持つ行列になります（薄い特異値分解）。</para>
/// </remarks>
/// <seealso href="https://www.netlib.org/lapack/explore-html/d3/da8/group__gesvd.html"/>
public sealed partial class SingularValueDecomposition
{
    // Retained so Rank can compute max(m, n) * eps * sigma_max even when U and V are omitted.
    private readonly int _maxDimension;

    private readonly int rows;
    private readonly int columns;

    private SingularValueDecomposition(double[] values, Matrix<double>? u, Matrix<double>? v, int rows, int columns)
    {
        Values = values;
        U = u;
        V = v;
        this.rows = rows;
        this.columns = columns;
        _maxDimension = Math.Max(rows, columns);
    }

    /// <summary>
    /// Gets the non-negative singular values in descending order.
    /// <para>降順に並んだ非負の特異値を取得します。</para>
    /// </summary>
    /// <remarks>
    /// <para><see cref="Recompute"/> overwrites this same array instance in place.</para>
    /// <para><see cref="Recompute"/> は同じ配列インスタンスをインプレースで上書きします。</para>
    /// </remarks>
    public double[] Values { get; }

    /// <summary>
    /// Gets the <c>m × min(m, n)</c> matrix <c>U</c> of left singular vectors, or <see langword="null"/> if singular vectors were not computed.
    /// <para>左特異ベクトルを列とする <c>m × min(m, n)</c> 行列 <c>U</c> を取得します（特異ベクトルを計算しなかった場合は <see langword="null"/>）。</para>
    /// </summary>
    /// <remarks>
    /// <para><see cref="Recompute"/> overwrites this same matrix instance in place.</para>
    /// <para><see cref="Recompute"/> は同じ行列インスタンスをインプレースで上書きします。</para>
    /// </remarks>
    public Matrix<double>? U { get; }

    /// <summary>
    /// Gets the <c>n × min(m, n)</c> matrix <c>V</c> of right singular vectors, or <see langword="null"/> if singular vectors were not computed.
    /// <para>右特異ベクトルを列とする <c>n × min(m, n)</c> 行列 <c>V</c> を取得します（特異ベクトルを計算しなかった場合は <see langword="null"/>）。</para>
    /// </summary>
    /// <remarks>
    /// <para><see cref="Recompute"/> overwrites this same matrix instance in place.</para>
    /// <para><see cref="Recompute"/> は同じ行列インスタンスをインプレースで上書きします。</para>
    /// </remarks>
    public Matrix<double>? V { get; }

    /// <summary>
    /// Gets the 2-norm condition number <c>σ_max / σ_min</c> (<see cref="double.NaN"/> for empty matrices).
    /// <para>2-ノルム条件数 <c>σ_max / σ_min</c> を取得します（空行列の場合は <see cref="double.NaN"/>）。</para>
    /// </summary>
    public double ConditionNumber => Values.Length == 0 ? double.NaN : Values[0] / Values[^1];

    /// <summary>
    /// Gets the numerical rank, defined as the count of singular values strictly greater than <c>max(m, n)·ε·σ_max</c>.
    /// <para>数値ランク（<c>σᵢ &gt; max(m, n)·ε·σ_max</c> を満たす特異値の個数）を取得します。</para>
    /// </summary>
    public int Rank => CountAboveCutoff(_maxDimension * NumericConstants.MachineEpsilon);

    /// <summary>
    /// Computes the thin singular value decomposition of <paramref name="a"/>.
    /// <para><paramref name="a"/> の薄い特異値分解を計算します。</para>
    /// </summary>
    /// <param name="a">The input matrix (unmodified when <paramref name="overwriteInput"/> is <see langword="false"/>). / 入力行列（<paramref name="overwriteInput"/> が <see langword="false"/> の場合は変更されません）。</param>
    /// <param name="computeVectors"><see langword="true"/> to also compute <see cref="U"/> and <see cref="V"/>; <see langword="false"/> to compute singular values only. / <see cref="U"/> および <see cref="V"/> も計算する場合は <see langword="true"/>、特異値のみの場合は <see langword="false"/>。</param>
    /// <param name="overwriteInput"><see langword="true"/> to reuse <paramref name="a"/>'s backing array as workspace when <c>Rows &gt;= Columns</c> (destroying its contents); when <c>Rows &lt; Columns</c>, a transposed workspace is always allocated. / <c>Rows &gt;= Columns</c> のときに <paramref name="a"/> の格納領域をワークスペースとして再利用する（元の内容は破壊されます）場合は <see langword="true"/>。<c>Rows &lt; Columns</c> の場合は転置用領域を常に確保します。</param>
    /// <returns>The computed singular value decomposition. / 特異値分解結果。</returns>
    /// <exception cref="OverflowException">The workspace element count required for computing singular vectors on a very large matrix exceeds <see cref="int.MaxValue"/>. / 特異ベクトルを求める大規模行列で、必要なワークスペース要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static SingularValueDecomposition Compute(Matrix<double> a, bool computeVectors = true, bool overwriteInput = false)
    {
        int m = a.Rows;
        int n = a.Columns;
        if (m < n)
        {
            // Factor A^T = V Sigma U^T and swap factors.
            var t = SvdKernels.Compute(n, m, XMath.Transpose(a).Data, computeVectors);
            return new SingularValueDecomposition(t.Values, Wrap(m, t.VData), Wrap(n, t.UData), m, n);
        }

        var r = SvdKernels.Compute(m, n, overwriteInput ? a.Data : (double[])a.Data.Clone(), computeVectors);
        return new SingularValueDecomposition(r.Values, Wrap(m, r.UData), Wrap(n, r.VData), m, n);
    }

    /// <summary>
    /// Computes the thin singular value decomposition of the matrix view <paramref name="a"/>.
    /// <para>行列ビュー <paramref name="a"/> の薄い特異値分解を計算します。</para>
    /// </summary>
    /// <param name="a">The input matrix view (not modified). / 入力行列ビュー（変更されません）。</param>
    /// <param name="computeVectors"><see langword="true"/> to also compute <see cref="U"/> and <see cref="V"/>; <see langword="false"/> to compute singular values only. / <see cref="U"/> および <see cref="V"/> も計算する場合は <see langword="true"/>、特異値のみの場合は <see langword="false"/>。</param>
    /// <returns>The computed singular value decomposition. / 特異値分解結果。</returns>
    /// <exception cref="OverflowException">The workspace element count required for computing singular vectors on a very large matrix exceeds <see cref="int.MaxValue"/>. / 特異ベクトルを求める大規模行列で、必要なワークスペース要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static SingularValueDecomposition Compute(ReadOnlyMatrixSpan<double> a, bool computeVectors = true)
    {
        int m = a.Rows;
        int n = a.Columns;
        var factors = GC.AllocateUninitializedArray<double>(checked(m * n));
        if (m < n)
        {
            // Factor A^T = V Sigma U^T and swap factors.
            StridedKernels.CopyIntoFactorStorage(a.T, factors);
            var t = SvdKernels.Compute(n, m, factors, computeVectors);
            return new SingularValueDecomposition(t.Values, Wrap(m, t.VData), Wrap(n, t.UData), m, n);
        }

        StridedKernels.CopyIntoFactorStorage(a, factors);
        var r = SvdKernels.Compute(m, n, factors, computeVectors);
        return new SingularValueDecomposition(r.Values, Wrap(m, r.UData), Wrap(n, r.VData), m, n);
    }

    /// <summary>
    /// Recomputes the singular value decomposition for another matrix <paramref name="a"/> of the same dimensions, overwriting <see cref="Values"/>, <see cref="U"/>, and <see cref="V"/> in place.
    /// <para>同じ形状の別の行列 <paramref name="a"/> を再分解し、<see cref="Values"/>・<see cref="U"/>・<see cref="V"/> をインプレースで更新します。</para>
    /// </summary>
    /// <param name="a">A matrix or view with the same dimensions as the original matrix. May alias <see cref="U"/> or <see cref="V"/> (copied to workspace before factorization); otherwise unmodified. / 元の行列と同じ形状の行列またはビュー。<see cref="U"/>・<see cref="V"/> と領域を共有していても構いません（分解前にワークスペースへコピーされます）。</param>
    /// <remarks>
    /// <para>Produces bit-for-bit identical results to <see cref="Compute(Matrix{double}, bool, bool)"/> with the same singular-vector setting. Workspace buffers (including the transpose buffer when <c>Rows &lt; Columns</c>) are reused from an internal pool, except when two-stage reduction is active. Do not call concurrently with other operations on the same instance.</para>
    /// <para><see cref="Compute(Matrix{double}, bool, bool)"/> と同じ特異ベクトル設定でビット単位に同一の結果を返します。ワークスペース（<c>Rows &lt; Columns</c> 時の転置用を含む）は内部プールから再利用します（二段階縮約が有効な場合を除く）。同一インスタンスに対する他の操作と同時に呼び出さないでください。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="a"/> does not match the original matrix dimensions. / <paramref name="a"/> の行数または列数が一致しない場合。</exception>
    public void Recompute(ReadOnlyMatrixSpan<double> a)
    {
        if (a.Rows != rows || a.Columns != columns)
        {
            throw new ArgumentException("The matrix must have the same numbers of rows and columns as the decomposition.", nameof(a));
        }

        int n = Values.Length;
        if (n == 0)
        {
            return;
        }

        // For wide matrices (rows < columns), factor A^T = V Sigma U^T as in Compute.
        bool transposed = rows < columns;
        int m = transposed ? columns : rows;
        var u = transposed ? V : U;
        var v = transposed ? U : V;
        double[] work = WorkArrayPool<double>.Rent(checked(m * n));
        try
        {
            // Copy input before writing results so a may alias U or V.
            var factors = work.AsSpan(0, m * n);
            StridedKernels.CopyIntoFactorStorage(transposed ? a.T : a, factors);
            Values.AsSpan().Clear();
            u?.Data.AsSpan().Clear();
            v?.Data.AsSpan().Clear();
            SvdKernels.Compute(m, n, factors, Values, u?.Data, v?.Data, U is not null);
        }
        finally
        {
            WorkArrayPool<double>.Return(work);
        }
    }

    private static Matrix<double>? Wrap(int rows, double[]? data) =>
        data is null ? null : Matrix<double>.Wrap(rows, rows == 0 ? 0 : data.Length / rows, data);
}
