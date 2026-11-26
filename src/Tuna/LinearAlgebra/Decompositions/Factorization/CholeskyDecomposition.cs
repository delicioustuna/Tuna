namespace Tuna.LinearAlgebra;

/// <summary>
/// Represents the Cholesky decomposition, <c>A = LLᵀ</c>, of a symmetric positive definite matrix.
/// <para>対称正定値行列のコレスキー分解 <c>A = LLᵀ</c> を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Only the lower triangular part of the input matrix is read; the upper triangular part is assumed to be symmetric and ignored.
/// Even for a positive definite matrix, ill-conditioning reduces solution accuracy. Use <see cref="EstimateReciprocalConditionNumber"/> to check conditioning (for example, to decide whether to increase diagonal jitter on a Gaussian process kernel matrix).
/// </para>
/// <para>
/// 入力行列の下三角部分のみを参照し、上三角部分は対称とみなして読み取りません。
/// 正定値であっても悪条件の場合は解の精度が低下します。条件の良し悪しは <see cref="EstimateReciprocalConditionNumber"/> で確認してください（ガウス過程では、値が小さい場合にカーネル行列の対角に加える jitter を増やす判断などに利用できます）。
/// </para>
/// </remarks>
/// <seealso href="https://www.netlib.org/lapack/explore-html/d3/d61/group__potrf.html"/>
public sealed class CholeskyDecomposition
{
    // Cached 1-norm of the original matrix; computed before factorization when A is available,
    // or lazily estimated from L for decompositions created via FromFactor, Update, TryDowndate, or Append.
    private double? norm;

    private CholeskyDecomposition(Matrix<double> factor, double? norm)
    {
        L = factor;
        this.norm = norm;
    }

    /// <summary>
    /// Gets the lower triangular factor <c>L</c> with positive diagonal entries.
    /// <para>対角成分が正の下三角因子 <c>L</c> を取得します。</para>
    /// </summary>
    /// <remarks>
    /// <para>
    /// Returns the underlying factor matrix directly without copying.
    /// Mutating operations (<see cref="Update"/>, <see cref="TryDowndate"/>, <see cref="TryRecompute"/>, and <see cref="Recompute"/>) overwrite this storage in place, so previously obtained references reflect the updated factor.
    /// While <see cref="IsPositiveDefinite"/> is <see langword="false"/>, the contents of <see cref="L"/> are undefined.
    /// </para>
    /// <para>
    /// 内部の因子行列をコピーせずそのまま返します。
    /// <see cref="Update"/>、<see cref="TryDowndate"/>、<see cref="TryRecompute"/>、<see cref="Recompute"/> はこの領域をインプレースで更新するため、取得済みの参照からも新しい因子が参照されます。
    /// <see cref="IsPositiveDefinite"/> が <see langword="false"/> の間は内容が不定です。
    /// </para>
    /// </remarks>
    public Matrix<double> L { get; }

    /// <summary>
    /// Gets a value indicating whether the decomposition currently holds a valid factorization.
    /// <para>現在有効な分解を保持しているかどうかを示す値を取得します。</para>
    /// </summary>
    /// <remarks>
    /// <para>
    /// Becomes <see langword="false"/> only after <see cref="TryRecompute"/> or <see cref="Recompute"/> fails because the input matrix is not numerically positive definite.
    /// Until a subsequent recomputation succeeds, solve, inverse, condition-number, refinement, and update methods throw <see cref="InvalidOperationException"/>, and determinant properties return <see cref="double.NaN"/>.
    /// </para>
    /// <para>
    /// <see cref="TryRecompute"/> または <see cref="Recompute"/> が失敗（入力行列が数値的に正定値でない）した場合にのみ <see langword="false"/> になります。
    /// 次の再計算が成功するまで、求解、逆行列、条件数推定、反復改良、および更新メソッドは <see cref="InvalidOperationException"/> をスローし、行列式プロパティは <see cref="double.NaN"/> を返します。
    /// </para>
    /// </remarks>
    public bool IsPositiveDefinite { get; private set; } = true;

    /// <summary>
    /// Gets the determinant <c>det A = Π lᵢᵢ²</c>, or <see cref="double.NaN"/> if <see cref="IsPositiveDefinite"/> is <see langword="false"/>.
    /// <para>行列式 <c>det A = Π lᵢᵢ²</c> を取得します。<see cref="IsPositiveDefinite"/> が <see langword="false"/> の場合は <see cref="double.NaN"/> を返します。</para>
    /// </summary>
    public double Determinant
    {
        get
        {
            if (!IsPositiveDefinite)
            {
                return double.NaN;
            }

            double det = 1;
            for (int i = 0; i < L.Rows; i++)
            {
                double d = L.Data[i + i * L.Rows];
                det *= d * d;
            }

            return det;
        }
    }

    /// <summary>
    /// Gets the natural logarithm of the determinant, <c>ln det A = 2 Σ ln lᵢᵢ</c>, or <see cref="double.NaN"/> if <see cref="IsPositiveDefinite"/> is <see langword="false"/>.
    /// <para>行列式の自然対数 <c>ln det A = 2 Σ ln lᵢᵢ</c> を取得します。<see cref="IsPositiveDefinite"/> が <see langword="false"/> の場合は <see cref="double.NaN"/> を返します。</para>
    /// </summary>
    public double LogDeterminant
    {
        get
        {
            if (!IsPositiveDefinite)
            {
                return double.NaN;
            }

            double sum = 0;
            for (int i = 0; i < L.Rows; i++)
            {
                sum += Math.Log(L.Data[i + i * L.Rows]);
            }

            return 2 * sum;
        }
    }

    /// <summary>
    /// Computes the Cholesky decomposition of the specified symmetric positive definite matrix.
    /// <para>指定した対称正定値行列のコレスキー分解を計算します。</para>
    /// </summary>
    /// <param name="a">The symmetric positive definite matrix (only the lower triangular part is read; not modified). / 対称正定値行列（下三角部分のみを参照し、変更しません）。</param>
    /// <returns>The Cholesky decomposition. / コレスキー分解の結果。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="a"/> is <see langword="null"/>. / <paramref name="a"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="a"/> is not square or is not numerically positive definite. / <paramref name="a"/> が正方行列でない、または数値的に正定値でない場合。</exception>
    public static CholeskyDecomposition Compute(Matrix<double> a)
    {
        if (!TryCompute(a, out var result))
        {
            throw new ArgumentException("The matrix is not positive definite.", nameof(a));
        }

        return result;
    }

    /// <summary>
    /// Computes the Cholesky decomposition of the specified symmetric positive definite matrix view.
    /// <para>指定した対称正定値行列ビューのコレスキー分解を計算します。</para>
    /// </summary>
    /// <param name="a">The symmetric positive definite matrix view (only the lower triangular part is read; not modified). / 対称正定値行列ビュー（下三角部分のみを参照し、変更しません）。</param>
    /// <returns>The Cholesky decomposition. / コレスキー分解の結果。</returns>
    /// <exception cref="ArgumentException"><paramref name="a"/> is not square or is not numerically positive definite. / <paramref name="a"/> が正方行列でない、または数値的に正定値でない場合。</exception>
    public static CholeskyDecomposition Compute(ReadOnlyMatrixSpan<double> a)
    {
        if (!TryCompute(a, out var result))
        {
            throw new ArgumentException("The matrix is not positive definite.", nameof(a));
        }

        return result;
    }

    /// <summary>
    /// Computes the Cholesky decomposition and stores the lower triangular factor <c>L</c> in <paramref name="destination"/>.
    /// <para>コレスキー分解を計算し、下三角因子 <c>L</c> を <paramref name="destination"/> に格納します。</para>
    /// </summary>
    /// <param name="a">The symmetric positive definite matrix (only the lower triangular part is read). / 対称正定値行列（下三角部分のみを参照します）。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="a"/>. Passing <paramref name="a"/> itself overwrites the input storage in place without allocating another matrix. The returned <see cref="L"/> shares this instance. / <paramref name="a"/> と同じ形状の出力先行列。<paramref name="a"/> 自身を渡すと新規確保せずインプレースで上書きします。結果の <see cref="L"/> はこのインスタンスを共有します。</param>
    /// <returns>The Cholesky decomposition. / コレスキー分解の結果。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="destination"/> is <see langword="null"/>. / <paramref name="a"/> または <paramref name="destination"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="a"/> is not square, the shapes do not match, or <paramref name="a"/> is not numerically positive definite (in which case the contents of <paramref name="destination"/> are undefined). / <paramref name="a"/> が正方行列でない、形状が一致しない、または数値的に正定値でない場合（失敗時の <paramref name="destination"/> の内容は不定です）。</exception>
    public static CholeskyDecomposition Compute(Matrix<double> a, Matrix<double> destination)
    {
        if (!TryCompute(a, destination, out var result))
        {
            throw new ArgumentException("The matrix is not positive definite.", nameof(a));
        }

        return result;
    }

    /// <summary>
    /// Computes the Cholesky decomposition of a matrix view and stores the lower triangular factor <c>L</c> in <paramref name="destination"/>.
    /// <para>行列ビューのコレスキー分解を計算し、下三角因子 <c>L</c> を <paramref name="destination"/> に格納します。</para>
    /// </summary>
    /// <param name="a">The symmetric positive definite matrix view (only the lower triangular part is read). / 対称正定値行列ビュー（下三角部分のみを参照します）。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="a"/>. May identically alias <paramref name="a"/>; partial overlap throws an exception. The returned <see cref="L"/> shares this instance. / <paramref name="a"/> と同じ形状の出力先行列。<paramref name="a"/> と完全一致する領域ならインプレースで上書きしますが、部分的な重なりは例外になります。結果の <see cref="L"/> はこのインスタンスを共有します。</param>
    /// <returns>The Cholesky decomposition. / コレスキー分解の結果。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="destination"/> is <see langword="null"/>. / <paramref name="destination"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="a"/> is not square, the shapes do not match, <paramref name="a"/> partially overlaps <paramref name="destination"/>, or <paramref name="a"/> is not numerically positive definite. / <paramref name="a"/> が正方行列でない、形状が一致しない、部分的に重なっている、または数値的に正定値でない場合。</exception>
    public static CholeskyDecomposition Compute(ReadOnlyMatrixSpan<double> a, Matrix<double> destination)
    {
        if (!TryCompute(a, destination, out var result))
        {
            throw new ArgumentException("The matrix is not positive definite.", nameof(a));
        }

        return result;
    }

    /// <summary>
    /// Attempts to compute the Cholesky decomposition of the specified symmetric matrix.
    /// <para>指定した対称行列のコレスキー分解を試みます。</para>
    /// </summary>
    /// <param name="a">The symmetric matrix (only the lower triangular part is read; not modified). / 対称行列（下三角部分のみを参照し、変更しません）。</param>
    /// <param name="result">When this method returns <see langword="true"/>, contains the Cholesky decomposition; otherwise, <see langword="null"/>. / 成功した場合はコレスキー分解の結果、失敗した場合は <see langword="null"/>。</param>
    /// <returns><see langword="true"/> if <paramref name="a"/> is numerically positive definite and factorization succeeded; otherwise, <see langword="false"/>. / 数値的に正定値で分解できた場合は <see langword="true"/>、それ以外は <see langword="false"/>。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="a"/> is <see langword="null"/>. / <paramref name="a"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="a"/> is not square. / <paramref name="a"/> が正方行列でない場合。</exception>
    public static bool TryCompute(Matrix<double> a, out CholeskyDecomposition result)
    {
        ArgumentNullException.ThrowIfNull(a);
        return TryCompute(a, Matrix<double>.Wrap(a.Rows, a.Columns, GC.AllocateUninitializedArray<double>(a.Data.Length)), out result);
    }

    /// <summary>
    /// Attempts to compute the Cholesky decomposition of the specified symmetric matrix view.
    /// <para>指定した対称行列ビューのコレスキー分解を試みます。</para>
    /// </summary>
    /// <param name="a">The symmetric matrix view (only the lower triangular part is read; not modified). / 対称行列ビュー（下三角部分のみを参照し、変更しません）。</param>
    /// <param name="result">When this method returns <see langword="true"/>, contains the Cholesky decomposition; otherwise, <see langword="null"/>. / 成功した場合はコレスキー分解の結果、失敗した場合は <see langword="null"/>。</param>
    /// <returns><see langword="true"/> if <paramref name="a"/> is numerically positive definite and factorization succeeded; otherwise, <see langword="false"/>. / 数値的に正定値で分解できた場合は <see langword="true"/>、それ以外は <see langword="false"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="a"/> is not square. / <paramref name="a"/> が正方行列でない場合。</exception>
    public static bool TryCompute(ReadOnlyMatrixSpan<double> a, out CholeskyDecomposition result)
    {
        if (a.Rows != a.Columns)
        {
            throw new ArgumentException("The matrix must be square.", nameof(a));
        }

        return TryCompute(a, Matrix<double>.Wrap(a.Rows, a.Columns, GC.AllocateUninitializedArray<double>(checked(a.Rows * a.Columns))), out result);
    }

    /// <summary>
    /// Attempts to compute the Cholesky decomposition and stores the lower triangular factor <c>L</c> in <paramref name="destination"/>.
    /// <para>コレスキー分解を試み、下三角因子 <c>L</c> を <paramref name="destination"/> に格納します。</para>
    /// </summary>
    /// <param name="a">The symmetric matrix (only the lower triangular part is read). / 対称行列（下三角部分のみを参照します）。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="a"/>; may be <paramref name="a"/> itself (contents are undefined on failure). / <paramref name="a"/> と同じ形状の出力先行列。<paramref name="a"/> 自身も指定できます（失敗時の内容は不定です）。</param>
    /// <param name="result">When this method returns <see langword="true"/>, contains the Cholesky decomposition; otherwise, <see langword="null"/>. / 成功した場合はコレスキー分解の結果、失敗した場合は <see langword="null"/>。</param>
    /// <returns><see langword="true"/> if <paramref name="a"/> is numerically positive definite and factorization succeeded; otherwise, <see langword="false"/>. / 数値的に正定値で分解できた場合は <see langword="true"/>、それ以外は <see langword="false"/>。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="destination"/> is <see langword="null"/>. / <paramref name="a"/> または <paramref name="destination"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="a"/> is not square, or <paramref name="destination"/> does not have the same shape as <paramref name="a"/>. / <paramref name="a"/> が正方行列でない、または <paramref name="destination"/> の形状が <paramref name="a"/> と一致しない場合。</exception>
    public static bool TryCompute(Matrix<double> a, Matrix<double> destination, out CholeskyDecomposition result)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(destination);
        if (!a.IsSquare)
        {
            throw new ArgumentException("The matrix must be square.", nameof(a));
        }

        if (destination.Rows != a.Rows || destination.Columns != a.Columns)
        {
            throw new ArgumentException("Destination shape must agree.", nameof(destination));
        }

        var l = destination.Data;
        if (!ReferenceEquals(l, a.Data))
        {
            a.Data.CopyTo(l, 0);
        }

        if (!TryFactorInPlace(a.Rows, l, out double norm))
        {
            result = null!;
            return false;
        }

        result = new CholeskyDecomposition(destination, norm);
        return true;
    }

    /// <summary>
    /// Attempts to compute the Cholesky decomposition of a matrix view and stores the lower triangular factor <c>L</c> in <paramref name="destination"/>.
    /// <para>行列ビューのコレスキー分解を試み、下三角因子 <c>L</c> を <paramref name="destination"/> に格納します。</para>
    /// </summary>
    /// <param name="a">The symmetric matrix view (only the lower triangular part is read). / 対称行列ビュー（下三角部分のみを参照します）。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="a"/>; may identically alias <paramref name="a"/> (contents are undefined on failure). / <paramref name="a"/> と同じ形状の出力先行列。<paramref name="a"/> と完全一致する領域ならインプレースで上書きします（失敗時の内容は不定です）。</param>
    /// <param name="result">When this method returns <see langword="true"/>, contains the Cholesky decomposition; otherwise, <see langword="null"/>. / 成功した場合はコレスキー分解の結果、失敗した場合は <see langword="null"/>。</param>
    /// <returns><see langword="true"/> if <paramref name="a"/> is numerically positive definite and factorization succeeded; otherwise, <see langword="false"/>. / 数値的に正定値で分解できた場合は <see langword="true"/>、それ以外は <see langword="false"/>。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="destination"/> is <see langword="null"/>. / <paramref name="destination"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="a"/> is not square, <paramref name="destination"/> does not have the same shape as <paramref name="a"/>, or <paramref name="a"/> partially overlaps <paramref name="destination"/>. / <paramref name="a"/> が正方行列でない、形状が一致しない、または部分的に重なっている場合。</exception>
    public static bool TryCompute(ReadOnlyMatrixSpan<double> a, Matrix<double> destination, out CholeskyDecomposition result)
    {
        ArgumentNullException.ThrowIfNull(destination);
        if (a.Rows != a.Columns)
        {
            throw new ArgumentException("The matrix must be square.", nameof(a));
        }

        if (destination.Rows != a.Rows || destination.Columns != a.Columns)
        {
            throw new ArgumentException("Destination shape must agree.", nameof(destination));
        }

        var l = destination.Data;
        StridedKernels.CopyIntoFactorStorage(a, l);
        if (!TryFactorInPlace(a.Rows, l, out double norm))
        {
            result = null!;
            return false;
        }

        result = new CholeskyDecomposition(destination, norm);
        return true;
    }

    /// <summary>
    /// Attempts to recompute the Cholesky decomposition for another symmetric matrix <paramref name="a"/> of the same order, overwriting <see cref="L"/> in place without allocating.
    /// <para>同じ次数の別の対称行列 <paramref name="a"/> に対してコレスキー分解の再計算を試み、<see cref="L"/> を新規アロケーションなしでインプレース更新します。</para>
    /// </summary>
    /// <param name="a">A symmetric matrix or matrix view of the same order (only the lower triangular part is read; not modified unless it identically aliases <see cref="L"/>, in which case it is factored in place without copying). / 同じ次数の対称行列または行列ビュー（下三角部分のみを参照）。<see cref="L"/> 自身と同じ領域・配置を指す場合はコピーせずインプレースで分解します。</param>
    /// <returns><see langword="true"/> if <paramref name="a"/> is numerically positive definite and factorization succeeded; otherwise, <see langword="false"/>. / 数値的に正定値で分解できた場合は <see langword="true"/>、それ以外は <see langword="false"/>。</returns>
    /// <remarks>
    /// <para>
    /// Reuses the decomposition instance and the <c>n × n</c> factor storage across repeated factorizations of the same order (such as optimization iterations, MCMC proposal adaptation, or tape replay).
    /// Because <see cref="L"/> is overwritten in place, previously obtained references reflect the new factor (including any <c>destination</c> matrix passed to <see cref="Compute(Matrix{double}, Matrix{double})"/>).
    /// On failure, returns <see langword="false"/>, sets <see cref="IsPositiveDefinite"/> to <see langword="false"/>, and leaves <see cref="L"/> undefined; the prior factorization is not preserved. Keep two instances and swap on success if fallback to the previous factor is needed.
    /// Must not be called concurrently with any other operation on the same decomposition.
    /// </para>
    /// <para>
    /// 最適化の反復、MCMC の提案分布の適応、自動微分の再実行など、同じ次数の行列を繰り返し分解する用途で分解オブジェクトと <c>n × n</c> の因子領域を再利用します。
    /// <see cref="L"/> の領域を直接書き換えるため、取得済みの参照や <see cref="Compute(Matrix{double}, Matrix{double})"/> に渡した出力先行列にも新しい因子が反映されます。
    /// 失敗した場合は <see langword="false"/> を返し、<see cref="IsPositiveDefinite"/> を <see langword="false"/> に設定します（<see cref="L"/> の内容は不定となり、前の分解は保持されません）。失敗時に前の分解を使い続けたい場合は、2 つのインスタンスを用意して成功側へ切り替えてください。
    /// 同じ分解インスタンスに対する他の呼び出しと同時に実行しないでください。
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="a"/> is not square, has a different order, or partially overlaps <see cref="L"/>. / <paramref name="a"/> が正方行列でない、次数が異なる、または <see cref="L"/> と部分的に重なる場合。</exception>
    public bool TryRecompute(ReadOnlyMatrixSpan<double> a)
    {
        int n = L.Rows;
        if (a.Rows != n || a.Columns != n)
        {
            throw new ArgumentException("The matrix must be square with the same order as the decomposition.", nameof(a));
        }

        var l = L.Data;
        StridedKernels.CopyIntoFactorStorage(a, l);
        if (!TryFactorInPlace(n, l, out double anorm))
        {
            IsPositiveDefinite = false;
            norm = null;
            return false;
        }

        IsPositiveDefinite = true;
        norm = anorm;
        return true;
    }

    /// <summary>
    /// Recomputes the Cholesky decomposition for another symmetric matrix <paramref name="a"/> of the same order in place, throwing if it is not positive definite.
    /// <para>同じ次数の別の対称行列 <paramref name="a"/> に対してコレスキー分解をインプレースで再計算します。正定値でない場合は例外をスローします。</para>
    /// </summary>
    /// <param name="a">A symmetric matrix or matrix view of the same order (only the lower triangular part is read). / 同じ次数の対称行列または行列ビュー（下三角部分のみを参照します）。</param>
    /// <exception cref="ArgumentException"><paramref name="a"/> is not square, has a different order, partially overlaps <see cref="L"/>, or is not numerically positive definite (in the last case, <see cref="IsPositiveDefinite"/> becomes <see langword="false"/>). / <paramref name="a"/> が正方行列でない、次数が異なる、<see cref="L"/> と部分的に重なる、または数値的に正定値でない場合（正定値でない場合、<see cref="IsPositiveDefinite"/> は <see langword="false"/> になります）。</exception>
    public void Recompute(ReadOnlyMatrixSpan<double> a)
    {
        if (!TryRecompute(a))
        {
            throw new ArgumentException("The matrix is not positive definite.", nameof(a));
        }
    }

    // Factors the lower triangle of column-major n-by-n buffer l in place and zeroes the strict upper triangle.
    // Contents of l are undefined on failure.
    private static bool TryFactorInPlace(int n, double[] l, out double norm)
    {
        norm = ConditionKernels.SymmetricNorm1(n, l, n);
        if (CholeskyKernels.Factor(n, l, n) != 0)
        {
            return false;
        }

        // Clear the strict upper triangle, including any scratch entries written during blocked factorization.
        for (int j = 1; j < n; j++)
        {
            l.AsSpan(j * n, j).Clear();
        }

        return true;
    }

    // LAPACK dpocon equivalent (at most 11 factored solves, O(n²)).
    /// <summary>
    /// Estimates the reciprocal 1-norm condition number <c>1 / (‖A‖₁‖A⁻¹‖₁)</c> in <c>O(n²)</c> time.
    /// <para>1-ノルム条件数の逆数 <c>1 / (‖A‖₁‖A⁻¹‖₁)</c> を <c>O(n²)</c> で推定します。</para>
    /// </summary>
    /// <returns>The estimated reciprocal 1-norm condition number. / 1-ノルム条件数の逆数の推定値。</returns>
    /// <remarks>
    /// <para>
    /// Values closer to 0 indicate worse conditioning; relative solution error can grow up to roughly <c>ε / rcond</c> (<c>ε ≈ 1.1×10⁻¹⁶</c>).
    /// Because <c>‖A⁻¹‖₁</c> is estimated from below using the Hager–Higham estimator, the returned value is an upper bound on the true reciprocal condition number and is typically within a factor of 3.
    /// For decompositions created or modified via <see cref="FromFactor"/>, <see cref="Update"/>, <see cref="TryDowndate"/>, or <see cref="Append"/>, <c>‖A‖₁</c> is also lazily estimated once in <c>O(n²)</c> time. Returns <c>1</c> when the order is 0.
    /// </para>
    /// <para>
    /// 0 に近いほど悪条件であり、解の相対誤差はおおよそ <c>ε / rcond</c>（<c>ε ≈ 1.1×10⁻¹⁶</c>）程度まで拡大し得ます。
    /// Hager–Higham の推定法により <c>‖A⁻¹‖₁</c> を下から推定するため、戻り値は真の条件数逆数以上となり、通常は真値の 3 倍以内に収まります。
    /// <see cref="FromFactor"/>、<see cref="Update"/>、<see cref="TryDowndate"/>、<see cref="Append"/> による分解では <c>‖A‖₁</c> も初回呼び出し時に 1 回だけ <c>O(n²)</c> で推定します。次数 0 の場合は <c>1</c> を返します。
    /// </para>
    /// </remarks>
    public double EstimateReciprocalConditionNumber()
    {
        RequirePositiveDefinite();
        int n = L.Rows;
        var l = L.Data;
        double anorm = norm ??= n == 0 ? 0 : ConditionKernels.EstimateNorm1(n, (x, _) => CholeskyKernels.MultiplyFactorProduct(n, l, n, x));
        return ConditionKernels.ReciprocalCondition(n, anorm, (x, _) => CholeskyKernels.Solve(n, l, n, 1, x, n, vector: true));
    }

    // LAPACK dporfs equivalent; computes residual b - Ax in extended precision.
    /// <summary>
    /// Improves the solution <paramref name="x"/> of <c>Ax = b</c> in place using iterative refinement.
    /// <para>反復改良を用いて連立一次方程式 <c>Ax = b</c> の解 <paramref name="x"/> をインプレースで高精度化します。</para>
    /// </summary>
    /// <param name="a">The original symmetric matrix (<c>n × n</c>, lower triangular part read). / 分解前の元の対称行列（<c>n × n</c>、下三角部分のみ参照）。</param>
    /// <param name="b">The right-hand side vector. / 右辺ベクトル。</param>
    /// <param name="x">The initial solution vector, overwritten in place with the refined solution; must not overlap <paramref name="a"/> or <paramref name="b"/>. / 初期解ベクトル。改良後の解でインプレース上書きされます（<paramref name="a"/>・<paramref name="b"/> と重なってはなりません）。</param>
    /// <param name="maxIterations">The maximum number of refinement iterations (0 computes only the backward error without refining). / 反復改良の最大回数（0 の場合は改良を行わず後退誤差のみを返します）。</param>
    /// <returns>The componentwise backward error <c>maxᵢ |b - Ax|ᵢ / (|A||x| + |b|)ᵢ</c>. / 成分ごとの後退誤差 <c>maxᵢ |b - Ax|ᵢ / (|A||x| + |b|)ᵢ</c>。</returns>
    /// <remarks>
    /// <para>Intended for ill-conditioned matrices when extra accuracy beyond <see cref="Solve(ReadOnlyVectorSpan{double})"/> is needed (<c>O(n²)</c> per iteration). Requires preserving the original matrix <paramref name="a"/>.</para>
    /// <para>悪条件の行列で <see cref="Solve(ReadOnlyVectorSpan{double})"/> 以上の精度が必要な場合に使用します（1 回あたり <c>O(n²)</c>）。元の行列 <paramref name="a"/> が必要です。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The shapes do not match, or <paramref name="x"/> overlaps <paramref name="a"/>, <paramref name="b"/>, or <see cref="L"/>. / 形状が一致しない、または <paramref name="x"/> が <paramref name="a"/>・<paramref name="b"/>・<see cref="L"/> と重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxIterations"/> is negative. / <paramref name="maxIterations"/> が負の場合。</exception>
    public double Refine(
        ReadOnlyMatrixSpan<double> a, ReadOnlyVectorSpan<double> b, VectorSpan<double> x, int maxIterations = ConditionKernels.DefaultRefinementIterations) =>
        Refine(a, b.AsMatrix(), x.AsMatrix(), maxIterations);

    /// <summary>
    /// Improves the solutions <paramref name="x"/> of <c>AX = B</c> column by column in place using iterative refinement.
    /// <para>反復改良を用いて複数右辺の連立一次方程式 <c>AX = B</c> の解 <paramref name="x"/> を列ごとにインプレースで高精度化します。</para>
    /// </summary>
    /// <param name="a">The original symmetric matrix (<c>n × n</c>, lower triangular part read). / 分解前の元の対称行列（<c>n × n</c>、下三角部分のみ参照）。</param>
    /// <param name="b">The right-hand side matrix. / 右辺行列。</param>
    /// <param name="x">The initial solution matrix, overwritten in place with the refined solutions; must not overlap <paramref name="a"/> or <paramref name="b"/>. / 初期解行列。改良後の解でインプレース上書きされます（<paramref name="a"/>・<paramref name="b"/> と重なってはなりません）。</param>
    /// <param name="maxIterations">The maximum number of refinement iterations per column (0 computes only the backward error). / 各列の反復改良の最大回数（0 の場合は後退誤差のみを返します）。</param>
    /// <returns>The maximum componentwise backward error across all columns. / 全列における成分ごとの後退誤差の最大値。</returns>
    /// <exception cref="ArgumentException">The shapes do not match, or <paramref name="x"/> overlaps <paramref name="a"/>, <paramref name="b"/>, or <see cref="L"/>. / 形状が一致しない、または <paramref name="x"/> が <paramref name="a"/>・<paramref name="b"/>・<see cref="L"/> と重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxIterations"/> is negative. / <paramref name="maxIterations"/> が負の場合。</exception>
    public double Refine(
        ReadOnlyMatrixSpan<double> a, ReadOnlyMatrixSpan<double> b, MatrixSpan<double> x, int maxIterations = ConditionKernels.DefaultRefinementIterations)
    {
        RequirePositiveDefinite();
        int n = L.Rows;
        StridedKernels.RequireRefinementArguments(n, a, b, x, maxIterations, L);
        var l = L.Data;
        return StridedKernels.Refine(a, symmetricLower: true, b, x, maxIterations, (r, _) => CholeskyKernels.Solve(n, l, n, 1, r, n, vector: true));
    }

    /// <summary>
    /// Solves the linear system <c>Ax = b</c>.
    /// <para>連立一次方程式 <c>Ax = b</c> を解きます。</para>
    /// </summary>
    /// <param name="b">The right-hand side vector of length <c>n</c>. / 長さ <c>n</c> の右辺ベクトル。</param>
    /// <returns>A newly allocated array containing the solution <c>x</c>. / 解 <c>x</c> を格納した新しい配列。</returns>
    public double[] Solve(ReadOnlySpan<double> b)
    {
        var x = new double[L.Rows];
        Solve(b, x);
        return x;
    }

    /// <summary>
    /// Solves the linear system <c>Ax = b</c> and writes the solution to <paramref name="destination"/>.
    /// <para>連立一次方程式 <c>Ax = b</c> を解き、解を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="b">The right-hand side vector of length <c>n</c>. / 長さ <c>n</c> の右辺ベクトル。</param>
    /// <param name="destination">The destination span of length <c>n</c>. May identically alias <paramref name="b"/> for in-place solving; partial overlap throws an exception. / 長さ <c>n</c> の出力先スパン。<paramref name="b"/> と完全に一致する領域ならインプレースで解けますが、部分的な重なりは例外になります。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    public Span<double> Solve(ReadOnlySpan<double> b, Span<double> destination)
    {
        RequirePositiveDefinite();
        int n = L.Rows;
        if (b.Length != n)
        {
            throw new ArgumentException("The length must equal the order of the matrix.", nameof(b));
        }

        if (destination.Length != n)
        {
            throw new ArgumentException("The length must equal the order of the matrix.", nameof(destination));
        }

        if (b.Overlaps(destination, out int offset) && offset != 0)
        {
            throw new ArgumentException("The destination partially overlaps the right-hand side.", nameof(destination));
        }

        b.CopyTo(destination);
        CholeskyKernels.Solve(n, L.Data, n, 1, destination, n, vector: true);
        return destination;
    }

    /// <summary>
    /// Solves the linear system <c>Ax = b</c>.
    /// <para>連立一次方程式 <c>Ax = b</c> を解きます。</para>
    /// </summary>
    /// <param name="b">The right-hand side vector. / 右辺ベクトル。</param>
    /// <returns>A newly allocated vector containing the solution <c>x</c>. / 解 <c>x</c> を格納した新しいベクトル。</returns>
    /// <exception cref="ArgumentException">The length of <paramref name="b"/> does not equal the matrix order. / <paramref name="b"/> の長さが次数と一致しない場合。</exception>
    public Vector<double> Solve(ReadOnlyVectorSpan<double> b)
    {
        var x = Vector<double>.Uninitialized(b.Length);
        Solve(b, x);
        return x;
    }

    /// <summary>
    /// Solves the linear system <c>Ax = b</c> and writes the solution to <paramref name="destination"/>.
    /// <para>連立一次方程式 <c>Ax = b</c> を解き、解を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="b">The right-hand side vector. / 右辺ベクトル。</param>
    /// <param name="destination">The destination vector. May identically alias <paramref name="b"/> for in-place solving, but must not partially overlap <paramref name="b"/> or overlap <see cref="L"/>. / 出力先ベクトル。<paramref name="b"/> と完全に一致する領域ならインプレースで解けますが、部分的な重なりや <see cref="L"/> との重なりは不可です。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">The length does not match the matrix order, or <paramref name="destination"/> overlaps invalidly. / 長さが次数と一致しない、または出力先が不正に重なる場合。</exception>
    public VectorSpan<double> Solve(ReadOnlyVectorSpan<double> b, VectorSpan<double> destination)
    {
        Solve(b.AsMatrix(), destination.AsMatrix(), vector: true);
        return destination;
    }

    /// <summary>
    /// Solves the linear system <c>AX = B</c> with multiple right-hand sides.
    /// <para>複数右辺の連立一次方程式 <c>AX = B</c> を解きます。</para>
    /// </summary>
    /// <param name="b">The right-hand side matrix. / 右辺行列。</param>
    /// <returns>A newly allocated matrix containing the solutions <c>X</c>. / 解 <c>X</c> を格納した新しい行列。</returns>
    /// <exception cref="ArgumentException">The number of rows in <paramref name="b"/> does not equal the matrix order. / <paramref name="b"/> の行数が次数と一致しない場合。</exception>
    public Matrix<double> Solve(ReadOnlyMatrixSpan<double> b)
    {
        var x = Matrix<double>.Uninitialized(b.Rows, b.Columns);
        Solve(b, x);
        return x;
    }

    /// <summary>
    /// Solves the linear system <c>AX = B</c> with multiple right-hand sides and writes the solutions to <paramref name="destination"/>.
    /// <para>複数右辺の連立一次方程式 <c>AX = B</c> を解き、解を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="b">The right-hand side matrix. / 右辺行列。</param>
    /// <param name="destination">The destination matrix. May identically alias <paramref name="b"/> for in-place solving, but must not partially overlap <paramref name="b"/> or overlap <see cref="L"/>. / 出力先行列。<paramref name="b"/> と完全に一致する領域ならインプレースで解けますが、部分的な重なりや <see cref="L"/> との重なりは不可です。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">The shapes do not match, or <paramref name="destination"/> overlaps invalidly. / 形状が一致しない、または出力先が不正に重なる場合。</exception>
    public MatrixSpan<double> Solve(ReadOnlyMatrixSpan<double> b, MatrixSpan<double> destination)
    {
        Solve(b, destination, vector: false);
        return destination;
    }

    // LAPACK dpotri equivalent.
    /// <summary>
    /// Computes the symmetric inverse matrix <c>A⁻¹ = L⁻ᵀL⁻¹</c>.
    /// <para>対称な逆行列 <c>A⁻¹ = L⁻ᵀL⁻¹</c> を計算します。</para>
    /// </summary>
    /// <returns>A newly allocated symmetric matrix containing <c>A⁻¹</c>. / <c>A⁻¹</c> を格納した新しい対称行列。</returns>
    /// <remarks>
    /// <para>When solving linear systems, prefer <see cref="Solve(ReadOnlyMatrixSpan{double})"/> over forming the inverse explicitly. The returned matrix is strictly symmetric.</para>
    /// <para>連立一次方程式を解くだけであれば、明示的に逆行列を求めるよりも <see cref="Solve(ReadOnlyMatrixSpan{double})"/> の方が高速かつ高精度です。返される行列は厳密に対称です。</para>
    /// </remarks>
    public Matrix<double> Inverse()
    {
        RequirePositiveDefinite();
        int n = L.Rows;
        var result = Matrix<double>.Uninitialized(n, n);
        CholeskyKernels.Inverse(n, L.Data, n, result.Data, n);
        return result;
    }

    /// <summary>
    /// Computes the symmetric inverse matrix <c>A⁻¹ = L⁻ᵀL⁻¹</c> and writes it to <paramref name="destination"/>.
    /// <para>対称な逆行列 <c>A⁻¹ = L⁻ᵀL⁻¹</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="destination">The <c>n × n</c> destination matrix, which must not overlap <see cref="L"/>. / <c>n × n</c> の出力先行列（<see cref="L"/> と重なってはなりません）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>Writes directly without temporary storage when <paramref name="destination"/> is column-major contiguous (or its transpose view).</para>
    /// <para><paramref name="destination"/> が列優先の連続領域（またはその転置ビュー）であれば作業領域を確保せず直接書き込みます。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The shape does not match, or <paramref name="destination"/> overlaps <see cref="L"/>. / 形状が一致しない、または <paramref name="destination"/> が <see cref="L"/> と重なる場合。</exception>
    public MatrixSpan<double> Inverse(MatrixSpan<double> destination)
    {
        RequirePositiveDefinite();
        int n = L.Rows;
        if (destination.Rows != n || destination.Columns != n)
        {
            throw new ArgumentException("Destination shape must agree.", nameof(destination));
        }

        StridedKernels.RequireDisjoint(L, destination);
        StridedKernels.InverseCholesky(n, L.Data, destination);
        return destination;
    }

    /// <summary>
    /// Creates a <see cref="CholeskyDecomposition"/> by copying the lower triangular entries of an existing Cholesky factor <c>L</c>.
    /// <para>既存の下三角因子 <c>L</c>（<c>A = LLᵀ</c>）の下三角部分をコピーして <see cref="CholeskyDecomposition"/> を生成します。</para>
    /// </summary>
    /// <param name="factor">A square lower triangular matrix with positive finite diagonal entries (only the lower triangle is read). / 対角成分が正かつ有限の下三角正方行列（下三角部分のみを参照します）。</param>
    /// <returns>The created decomposition. / 生成されたコレスキー分解。</returns>
    /// <remarks>
    /// <para>Upper triangular entries of <paramref name="factor"/> are ignored and set to 0 in <see cref="L"/>.</para>
    /// <para><paramref name="factor"/> の上三角部分は読み取らず、生成される <see cref="L"/> では 0 で埋められます。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="factor"/> is not square, contains non-finite values in the lower triangle, or has a non-positive diagonal entry. / <paramref name="factor"/> が正方行列でない、下三角部分に有限でない値を含む、または正でない対角成分がある場合。</exception>
    public static CholeskyDecomposition FromFactor(ReadOnlyMatrixSpan<double> factor)
    {
        if (factor.Rows != factor.Columns)
        {
            throw new ArgumentException("The factor must be square.", nameof(factor));
        }

        int n = factor.Rows;
        var l = new Matrix<double>(n, n);
        var data = l.Data;
        for (int j = 0; j < n; j++)
        {
            if (!(factor[j, j] > 0 && double.IsFinite(factor[j, j])))
            {
                throw new ArgumentException("The diagonal of the factor must be positive and finite.", nameof(factor));
            }

            for (int i = j; i < n; i++)
            {
                double v = factor[i, j];
                if (!double.IsFinite(v))
                {
                    throw new ArgumentException("The factor must be finite.", nameof(factor));
                }

                data[i + j * n] = v;
            }
        }

        return new CholeskyDecomposition(l, null);
    }

    /// <summary>
    /// Performs an in-place rank-1 Cholesky update for <c>A ← A + xxᵀ</c> in <c>O(n²)</c> time.
    /// <para>ランク 1 更新 <c>A ← A + xxᵀ</c> に対応する因子へ <see cref="L"/> を <c>O(n²)</c> でインプレース更新します。</para>
    /// </summary>
    /// <param name="x">The update vector of length <c>n</c> (copied before updating, so it may overlap <see cref="L"/>). For a weighted update <c>A + wxxᵀ</c> (<c>w &gt; 0</c>), pass <c>√w·x</c>. / 長さ <c>n</c> の更新ベクトル（更新前にコピーするため <see cref="L"/> と重なっても構いません）。重み付き更新 <c>A + wxxᵀ</c>（<c>w &gt; 0</c>）には <c>√w·x</c> を渡します。</param>
    /// <exception cref="ArgumentException">The length of <paramref name="x"/> does not equal the matrix order, or <paramref name="x"/> contains non-finite values. / <paramref name="x"/> の長さが次数と一致しない、または有限でない値を含む場合。</exception>
    public void Update(ReadOnlyVectorSpan<double> x)
    {
        RequirePositiveDefinite();
        int n = L.Rows;
        RequireUpdateVector(x, n);
        double[] rented = WorkArrayPool<double>.Rent(n);
        try
        {
            var w = rented.AsSpan(0, n);
            x.CopyTo(w);
            CholeskyKernels.UpdateRankOne(n, L.Data, n, w);
        }
        finally
        {
            WorkArrayPool<double>.Return(rented);
        }

        norm = null;
    }

    /// <summary>
    /// Attempts an in-place rank-1 Cholesky downdate for <c>A ← A − xxᵀ</c> in <c>O(n²)</c> time.
    /// <para>ランク 1 ダウンデート <c>A ← A − xxᵀ</c> に対応する因子へ <see cref="L"/> のインプレース更新を <c>O(n²)</c> で試みます。</para>
    /// </summary>
    /// <param name="x">The downdate vector of length <c>n</c> (copied before updating, so it may overlap <see cref="L"/>). / 長さ <c>n</c> のベクトル（更新前にコピーするため <see cref="L"/> と重なっても構いません）。</param>
    /// <returns><see langword="true"/> if <c>A − xxᵀ</c> remains numerically positive definite and <see cref="L"/> was updated; otherwise, <see langword="false"/> (leaving <see cref="L"/> unchanged). / <c>A − xxᵀ</c> が数値的に正定値を保ち <see cref="L"/> が更新された場合は <see langword="true"/>、それ以外は <see langword="false"/>（<see cref="L"/> は変更されません）。</returns>
    /// <remarks>
    /// <para>Positive definiteness requires <c>‖L⁻¹x‖₂ &lt; 1</c>; accuracy degrades as <c>‖L⁻¹x‖₂</c> approaches 1.</para>
    /// <para>正定値であるための条件は <c>‖L⁻¹x‖₂ &lt; 1</c> であり、<c>‖L⁻¹x‖₂</c> が 1 に近いほど精度が低下します。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The length of <paramref name="x"/> does not equal the matrix order, or <paramref name="x"/> contains non-finite values. / <paramref name="x"/> の長さが次数と一致しない、または有限でない値を含む場合。</exception>
    public bool TryDowndate(ReadOnlyVectorSpan<double> x)
    {
        RequirePositiveDefinite();
        int n = L.Rows;
        RequireUpdateVector(x, n);
        double[] rented = WorkArrayPool<double>.Rent(3 * n);
        try
        {
            var work = rented.AsSpan(0, 3 * n);
            x.CopyTo(work[..n]);
            if (!CholeskyKernels.TryDowndateRankOne(n, L.Data, n, work))
            {
                return false;
            }
        }
        finally
        {
            WorkArrayPool<double>.Return(rented);
        }

        norm = null;
        return true;
    }

    /// <summary>
    /// Returns a new decomposition for the extended matrix <c>[[A, b], [bᵀ, c]]</c> obtained by appending one row and column in <c>O(n²)</c> time without modifying this instance.
    /// <para>末尾に 1 行・1 列を追加した拡張行列 <c>[[A, b], [bᵀ, c]]</c> のコレスキー分解を <c>O(n²)</c> で計算し、新しいインスタンスとして返します（このインスタンスは変更しません）。</para>
    /// </summary>
    /// <param name="column">The appended column <c>[b; c]</c> of length <c>n + 1</c>, where <c>c</c> is the new diagonal entry. / 長さ <c>n + 1</c> の追加列 <c>[b; c]</c>（<c>c</c> が新しい対角成分）。</param>
    /// <returns>The decomposition of order <c>n + 1</c>. / 次数 <c>n + 1</c> の新しい分解。</returns>
    /// <remarks>
    /// <para>Useful for incrementally adding observations to a Gaussian process without a full <c>O(n³)</c> refactorization.</para>
    /// <para>ガウス過程に観測点を 1 点追加する際などに、<c>O(n³)</c> の再分解を避けて更新できます。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The length of <paramref name="column"/> is not <c>n + 1</c>, <paramref name="column"/> contains non-finite values, or the extended matrix is not numerically positive definite. / <paramref name="column"/> の長さが <c>n + 1</c> でない、有限でない値を含む、または拡張行列が数値的に正定値でない場合。</exception>
    public CholeskyDecomposition Append(ReadOnlyVectorSpan<double> column)
    {
        if (!TryAppend(column, out var result))
        {
            throw new ArgumentException("The extended matrix is not positive definite.", nameof(column));
        }

        return result;
    }

    /// <summary>
    /// Attempts to compute a new decomposition for the extended matrix <c>[[A, b], [bᵀ, c]]</c> by appending one row and column in <c>O(n²)</c> time.
    /// <para>末尾に 1 行・1 列を追加した拡張行列 <c>[[A, b], [bᵀ, c]]</c> のコレスキー分解を <c>O(n²)</c> で試みます。</para>
    /// </summary>
    /// <param name="column">The appended column <c>[b; c]</c> of length <c>n + 1</c>, where <c>c</c> is the new diagonal entry. / 長さ <c>n + 1</c> の追加列 <c>[b; c]</c>（<c>c</c> が新しい対角成分）。</param>
    /// <param name="result">When this method returns <see langword="true"/>, contains the decomposition of order <c>n + 1</c>; otherwise, <see langword="null"/>. / 成功した場合は次数 <c>n + 1</c> の分解、失敗した場合は <see langword="null"/>。</param>
    /// <returns><see langword="true"/> if the extended matrix is numerically positive definite; otherwise, <see langword="false"/>. / 拡張行列が数値的に正定値であれば <see langword="true"/>、それ以外は <see langword="false"/>。</returns>
    /// <exception cref="ArgumentException">The length of <paramref name="column"/> is not <c>n + 1</c>, or <paramref name="column"/> contains non-finite values. / <paramref name="column"/> の長さが <c>n + 1</c> でない、または有限でない値を含む場合。</exception>
    public bool TryAppend(ReadOnlyVectorSpan<double> column, out CholeskyDecomposition result)
    {
        RequirePositiveDefinite();
        int n = L.Rows;
        RequireUpdateVector(column, n + 1, nameof(column));
        var extended = new Matrix<double>(n + 1, n + 1);
        var data = extended.Data;
        var l = L.Data;
        int m = n + 1;

        // 新しい行 lᵀ を最後の行に置く: L l = b（前進代入）
        double[] rented = WorkArrayPool<double>.Rent(n);
        try
        {
            var row = rented.AsSpan(0, n);
            column[..n].CopyTo(row);
            TriangularKernels.SolveLeft(lower: true, transposeA: false, unitDiagonal: false, n, 1, l, n, row, n);
            double d2 = column[n] - Level1Kernels.Dot(row, row);
            double d = Math.Sqrt(d2);
            if (!(d > 0 && double.IsFinite(d)))
            {
                result = null!;
                return false;
            }

            for (int j = 0; j < n; j++)
            {
                l.AsSpan(j + j * n, n - j).CopyTo(data.AsSpan(j + j * m, n - j));
                data[n + j * m] = row[j];
            }

            data[n + n * m] = d;
        }
        finally
        {
            WorkArrayPool<double>.Return(rented);
        }

        result = new CholeskyDecomposition(extended, null);
        return true;
    }

    private void RequirePositiveDefinite()
    {
        if (!IsPositiveDefinite)
        {
            throw new InvalidOperationException("The last recomputation failed because the matrix was not positive definite.");
        }
    }

    private static void RequireUpdateVector(ReadOnlyVectorSpan<double> x, int length, string name = "x")
    {
        if (x.Length != length)
        {
            throw new ArgumentException(name == "x"
                ? "The length must equal the order of the matrix."
                : "The length must be the order of the matrix plus one.", name);
        }

        for (int i = 0; i < x.Length; i++)
        {
            if (!double.IsFinite(x[i]))
            {
                throw new ArgumentException("The vector must be finite.", name);
            }
        }
    }

    private void Solve(ReadOnlyMatrixSpan<double> b, MatrixSpan<double> x, bool vector)
    {
        RequirePositiveDefinite();
        int n = L.Rows;
        if (b.Rows != n)
        {
            throw new ArgumentException("The right-hand side must have as many rows as the order of the matrix.", nameof(b));
        }

        if (x.Rows != n || x.Columns != b.Columns)
        {
            throw new ArgumentException("Destination shape must agree.", "destination");
        }

        StridedKernels.RequireDisjoint(L, x);
        StridedKernels.CopyRightHandSide(b, x);
        StridedKernels.SolveCholeskyInPlace(n, L.Data, x, vector);
    }
}
