using System.Runtime.CompilerServices;

namespace Tuna.LinearAlgebra;

/// <summary>
/// Represents the batched Cholesky decomposition, <c>A[k] = L[k]L[k]ᵀ</c>, of a batch of symmetric positive definite matrices of the same order.
/// <para>同じ次数の対称正定値行列のバッチに対するコレスキー分解 <c>A[k] = L[k]L[k]ᵀ</c> を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// Intended for factoring and solving many matrices of the same order together, such as independent Gaussian processes or per-chain MCMC covariances.
/// Only the lower triangular part of each input matrix is read.
/// If any matrix in the batch is not positive definite, the remaining matrices are still factored, and the failed matrix's factor is filled with <see cref="double.NaN"/> (<see cref="IsPositiveDefinite"/> returns <see langword="false"/>, and its log-determinant, solutions, and inverse are <see cref="double.NaN"/>).
/// </para>
/// <para>
/// 独立なガウス過程（出力ごと、またはハイパーパラメーター候補ごと）や MCMC のチェーンごとの共分散など、同じ次数の多数の行列をまとめて分解・求解する用途に適しています。
/// 入力の各行列では下三角部分のみを参照します。
/// 正定値でない行列が含まれていても他の行列の分解は続行し、失敗した行列の因子は <see cref="double.NaN"/> で埋めます（<see cref="IsPositiveDefinite"/> は <see langword="false"/> を返し、対数行列式、解、逆行列も <see cref="double.NaN"/> になります）。
/// </para>
/// </remarks>
public sealed class BatchCholeskyDecomposition
{
    private readonly bool[] positiveDefinite;
    private readonly double[] norms;

    private BatchCholeskyDecomposition(Tensor<double> factor, bool[] positiveDefinite, int failures, double[] norms)
    {
        L = factor;
        this.positiveDefinite = positiveDefinite;
        this.norms = norms;
        FailureCount = failures;
    }

    /// <summary>
    /// Gets the batch of lower triangular factors <c>L[k]</c> with positive diagonal entries.
    /// <para>対角成分が正の下三角因子 <c>L[k]</c> のバッチを取得します。</para>
    /// </summary>
    /// <value>A 3D tensor whose slices hold <c>L[k]</c>; slices for matrices that were not positive definite are filled with <see cref="double.NaN"/>. / 各スライスに <c>L[k]</c> を格納した 3 次元テンソル。正定値でなかった行列のスライスは <see cref="double.NaN"/> で埋められます。</value>
    public Tensor<double> L { get; }

    /// <summary>
    /// Gets the order <c>n</c> of each matrix in the batch.
    /// <para>バッチ内の各行列の次数 <c>n</c> を取得します。</para>
    /// </summary>
    public int Order => L.Rows;

    /// <summary>
    /// Gets the number of matrices in the batch.
    /// <para>バッチ内の行列の個数を取得します。</para>
    /// </summary>
    public int BatchCount => L.MatrixCount;

    /// <summary>
    /// Gets the number of matrices in the batch that failed factorization because they were not positive definite.
    /// <para>正定値でなく分解に失敗した行列の個数を取得します。</para>
    /// </summary>
    public int FailureCount { get; }

    /// <summary>
    /// Determines whether the matrix at the specified index was numerically positive definite and successfully factored.
    /// <para>指定したインデックスの行列が数値的に正定値であり、分解に成功したかどうかを返します。</para>
    /// </summary>
    /// <param name="index">The zero-based matrix index in the batch. / 0 から始まるバッチ内の行列インデックス。</param>
    /// <returns><see langword="true"/> if the Cholesky decomposition of the matrix at <paramref name="index"/> succeeded; otherwise, <see langword="false"/>. / インデックス <paramref name="index"/> の行列のコレスキー分解が成功した場合は <see langword="true"/>、それ以外は <see langword="false"/>。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is negative or greater than or equal to <see cref="BatchCount"/>. / <paramref name="index"/> が負、または <see cref="BatchCount"/> 以上の場合。</exception>
    public bool IsPositiveDefinite(int index)
    {
        DenseStorage.CheckIndex(index, BatchCount, nameof(index));
        return positiveDefinite[index];
    }

    /// <summary>
    /// Computes the batched Cholesky decomposition of the specified symmetric positive definite matrices.
    /// <para>指定した対称正定値行列のバッチに対してコレスキー分解を計算します。</para>
    /// </summary>
    /// <param name="a">The batch of symmetric positive definite matrices (only the lower triangular part of each matrix is read; not modified). / 対称正定値行列のバッチ（各行列の下三角部分のみを参照し、変更しません）。</param>
    /// <returns>The batched Cholesky decomposition. / バッチコレスキー分解の結果。</returns>
    /// <exception cref="ArgumentException">The matrices are not square, or any matrix in the batch is not numerically positive definite (the error message reports the first failing index). / 行列が正方でない、または数値的に正定値でない行列が含まれる場合（最初に失敗したインデックスを示します）。</exception>
    /// <exception cref="OverflowException">The total number of elements in the result exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の総要素数が <see cref="int.MaxValue"/> を超える場合（大きなブロードキャストビューを渡した場合など）。</exception>
    public static BatchCholeskyDecomposition Compute(ReadOnlyTensorSpan<double> a) =>
        Compute(a, Tensor<double>.Uninitialized(a.Rows, a.Columns, a.MatrixCount));

    /// <summary>
    /// Computes the batched Cholesky decomposition and stores the factor tensor <c>L</c> in <paramref name="destination"/>.
    /// <para>バッチコレスキー分解を計算し、因子テンソル <c>L</c> を <paramref name="destination"/> に格納します。</para>
    /// </summary>
    /// <param name="a">The batch of symmetric positive definite matrices (only the lower triangular part of each matrix is read). / 対称正定値行列のバッチ（各行列の下三角部分のみを参照します）。</param>
    /// <param name="destination">The destination tensor with the same shape as <paramref name="a"/>. If <paramref name="a"/> is a view of the entire <paramref name="destination"/> tensor, factorization overwrites it in place without allocating another tensor. The returned <see cref="L"/> shares this instance. / <paramref name="a"/> と同じ形状の出力先テンソル。<paramref name="a"/> が <paramref name="destination"/> 全体のビューであれば新規確保せずインプレースで上書きします。結果の <see cref="L"/> はこのインスタンスを共有します。</param>
    /// <returns>The batched Cholesky decomposition. / バッチコレスキー分解の結果。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="destination"/> is <see langword="null"/>. / <paramref name="destination"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException">The matrices are not square, the shapes do not match, <paramref name="destination"/> partially overlaps <paramref name="a"/>, or any matrix is not numerically positive definite. / 行列が正方でない、形状が一致しない、<paramref name="destination"/> が <paramref name="a"/> と部分的に重なる、または数値的に正定値でない行列が含まれる場合。</exception>
    public static BatchCholeskyDecomposition Compute(ReadOnlyTensorSpan<double> a, Tensor<double> destination)
    {
        var result = Factor(a, destination);
        if (result.FailureCount != 0)
        {
            int index = Array.IndexOf(result.positiveDefinite, false);
            throw new ArgumentException($"The matrix at index {index} is not positive definite.", nameof(a));
        }

        return result;
    }

    /// <summary>
    /// Attempts to compute the batched Cholesky decomposition of the specified symmetric matrices.
    /// <para>指定した対称行列のバッチに対してコレスキー分解を試みます。</para>
    /// </summary>
    /// <param name="a">The batch of symmetric matrices (only the lower triangular part of each matrix is read; not modified). / 対称行列のバッチ（各行列の下三角部分のみを参照し、変更しません）。</param>
    /// <param name="result">When this method returns, contains the decomposition result even if some matrices failed; inspect per-matrix status with <see cref="IsPositiveDefinite"/>. / このメソッドから制御が戻るとき、一部の行列が失敗した場合でも分解結果を格納します。行列ごとの成否は <see cref="IsPositiveDefinite"/> で確認できます。</param>
    /// <returns><see langword="true"/> if every matrix in the batch was numerically positive definite and factored; otherwise, <see langword="false"/>. / すべての行列が数値的に正定値で分解できた場合は <see langword="true"/>、それ以外は <see langword="false"/>。</returns>
    /// <exception cref="ArgumentException">The matrices are not square. / 行列が正方でない場合。</exception>
    /// <exception cref="OverflowException">The total number of elements in the result exceeds <see cref="int.MaxValue"/>. / 結果の総要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static bool TryCompute(ReadOnlyTensorSpan<double> a, out BatchCholeskyDecomposition result)
    {
        result = Factor(a, Tensor<double>.Uninitialized(a.Rows, a.Columns, a.MatrixCount));
        return result.FailureCount == 0;
    }

    /// <summary>
    /// Attempts to compute the batched Cholesky decomposition and stores the factor tensor <c>L</c> in <paramref name="destination"/>.
    /// <para>バッチコレスキー分解を試み、因子テンソル <c>L</c> を <paramref name="destination"/> に格納します。</para>
    /// </summary>
    /// <param name="a">The batch of symmetric matrices (only the lower triangular part of each matrix is read). / 対称行列のバッチ（各行列の下三角部分のみを参照します）。</param>
    /// <param name="destination">The destination tensor with the same shape as <paramref name="a"/>; may be aliased identically with <paramref name="a"/>. / <paramref name="a"/> と同じ形状の出力先テンソル。<paramref name="a"/> がこのテンソル全体のビューであっても構いません。</param>
    /// <param name="result">When this method returns, contains the decomposition result even if some matrices failed. / このメソッドから制御が戻るとき、一部の行列が失敗した場合でも分解結果を格納します。</param>
    /// <returns><see langword="true"/> if every matrix in the batch was numerically positive definite and factored; otherwise, <see langword="false"/>. / すべての行列が数値的に正定値で分解できた場合は <see langword="true"/>、それ以外は <see langword="false"/>。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="destination"/> is <see langword="null"/>. / <paramref name="destination"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException">The matrices are not square, the shapes do not match, or <paramref name="destination"/> partially overlaps <paramref name="a"/>. / 行列が正方でない、形状が一致しない、または <paramref name="destination"/> が <paramref name="a"/> と部分的に重なる場合。</exception>
    public static bool TryCompute(ReadOnlyTensorSpan<double> a, Tensor<double> destination, out BatchCholeskyDecomposition result)
    {
        result = Factor(a, destination);
        return result.FailureCount == 0;
    }

    /// <summary>
    /// Computes the natural logarithm of the determinant, <c>ln det A[k] = 2 Σ ln lᵢᵢ</c>, for each matrix in the batch.
    /// <para>バッチ内の各行列の行列式の自然対数 <c>ln det A[k] = 2 Σ ln lᵢᵢ</c> を計算します。</para>
    /// </summary>
    /// <returns>A newly allocated vector of length <see cref="BatchCount"/>. / 長さ <see cref="BatchCount"/> の新しいベクトル。</returns>
    /// <remarks>
    /// <para>Useful for Gaussian process log marginal likelihoods and multivariate normal log-density normalization terms. Entries for matrices that were not positive definite are <see cref="double.NaN"/>.</para>
    /// <para>ガウス過程の対数周辺尤度や多変量正規分布の対数密度の正規化項の計算に利用できます。正定値でなかった行列に対応する要素は <see cref="double.NaN"/> になります。</para>
    /// </remarks>
    public Vector<double> LogDeterminants()
    {
        var result = Vector<double>.Uninitialized(BatchCount);
        LogDeterminants(result);
        return result;
    }

    /// <summary>
    /// Computes the natural logarithm of the determinant for each matrix in the batch and writes the results to <paramref name="destination"/>.
    /// <para>バッチ内の各行列の行列式の自然対数を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="destination">The destination vector of length <see cref="BatchCount"/>. / 長さ <see cref="BatchCount"/> の出力先ベクトル。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <see cref="BatchCount"/>. / <paramref name="destination"/> の長さが <see cref="BatchCount"/> と一致しない場合。</exception>
    public VectorSpan<double> LogDeterminants(VectorSpan<double> destination)
    {
        if (destination.Length != BatchCount)
        {
            throw new ArgumentException("The length must equal the number of matrices.", nameof(destination));
        }

        int n = Order;
        var data = L.Data;
        for (int k = 0; k < BatchCount; k++)
        {
            int offset = k * n * n;
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += Math.Log(data[offset + i + i * n]);
            }

            destination[k] = 2 * sum;
        }

        return destination;
    }

    /// <summary>
    /// Estimates the reciprocal 1-norm condition number, <c>1 / (‖A[k]‖₁‖A[k]⁻¹‖₁)</c>, for each matrix in the batch.
    /// <para>バッチ内の各行列の 1-ノルム条件数の逆数 <c>1 / (‖A[k]‖₁‖A[k]⁻¹‖₁)</c> を推定します。</para>
    /// </summary>
    /// <returns>A newly allocated vector of length <see cref="BatchCount"/> containing the estimates. / 推定値を格納した長さ <see cref="BatchCount"/> の新しいベクトル。</returns>
    /// <remarks>
    /// <para>Each entry uses the same calculation as <see cref="CholeskyDecomposition.EstimateReciprocalConditionNumber"/>. Entries for matrices that were not positive definite are <see cref="double.NaN"/>.</para>
    /// <para>各行列は <see cref="CholeskyDecomposition.EstimateReciprocalConditionNumber"/> と同じ方法で推定されます。正定値でなかった行列に対応する要素は <see cref="double.NaN"/> になります。</para>
    /// </remarks>
    public Vector<double> EstimateReciprocalConditionNumbers()
    {
        var result = Vector<double>.Uninitialized(BatchCount);
        EstimateReciprocalConditionNumbers(result);
        return result;
    }

    /// <summary>
    /// Estimates the reciprocal 1-norm condition number for each matrix in the batch and writes the results to <paramref name="destination"/>.
    /// <para>バッチ内の各行列の 1-ノルム条件数の逆数を推定し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="destination">The destination vector of length <see cref="BatchCount"/>. / 長さ <see cref="BatchCount"/> の出力先ベクトル。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <see cref="BatchCount"/>. / <paramref name="destination"/> の長さが <see cref="BatchCount"/> と一致しない場合。</exception>
    public VectorSpan<double> EstimateReciprocalConditionNumbers(VectorSpan<double> destination)
    {
        if (destination.Length != BatchCount)
        {
            throw new ArgumentException("The length must equal the number of matrices.", nameof(destination));
        }

        int n = Order;
        var factor = L.Data;
        var values = new double[BatchCount];
        StridedKernels.ForEachMatrix(BatchCount, 22.0 * n * n, (start, end) =>
        {
            for (int k = start; k < end; k++)
            {
                int offset = k * n * n;
                values[k] = positiveDefinite[k]
                    ? ConditionKernels.ReciprocalCondition(n, norms[k], (x, _) => CholeskyKernels.Solve(n, factor.AsSpan(offset, n * n), n, 1, x, n, vector: true))
                    : double.NaN;
            }
        });

        for (int k = 0; k < BatchCount; k++)
        {
            destination[k] = values[k];
        }

        return destination;
    }

    /// <summary>
    /// Solves the linear systems <c>A[k]x[k] = b[k]</c> for each matrix in the batch with a single right-hand side vector per matrix.
    /// <para>バッチ内の各行列について、単一右辺の連立一次方程式 <c>A[k]x[k] = b[k]</c> を解きます。</para>
    /// </summary>
    /// <param name="b">The <c>n × BatchCount</c> matrix whose <c>k</c>-th column is the right-hand side for matrix <c>k</c>. / 第 <c>k</c> 列が第 <c>k</c> 行列の右辺となる <c>n × BatchCount</c> 行列。</param>
    /// <returns>A newly allocated matrix containing the solutions. / 解を格納した新しい行列。</returns>
    /// <remarks>
    /// <para>Equivalent to <c>Solve(b.Unsqueeze(1))</c>; each column matches <see cref="CholeskyDecomposition.Solve(ReadOnlyVectorSpan{double})"/>.</para>
    /// <para><c>Solve(b.Unsqueeze(1))</c> と同じ計算を行い、各列の解は <see cref="CholeskyDecomposition.Solve(ReadOnlyVectorSpan{double})"/> と一致します。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The dimensions of <paramref name="b"/> do not match <c>Order × BatchCount</c>. / <paramref name="b"/> の形状が <c>Order × BatchCount</c> と一致しない場合。</exception>
    public Matrix<double> Solve(ReadOnlyMatrixSpan<double> b)
    {
        var x = Matrix<double>.Uninitialized(b.Rows, b.Columns);
        Solve(b, x);
        return x;
    }

    /// <summary>
    /// Solves the linear systems <c>A[k]x[k] = b[k]</c> for each matrix in the batch and writes the solutions to <paramref name="destination"/>.
    /// <para>バッチ内の各行列について <c>A[k]x[k] = b[k]</c> を解き、解を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="b">The <c>n × BatchCount</c> right-hand side matrix. / <c>n × BatchCount</c> の右辺行列。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="b"/>. May identically alias <paramref name="b"/> for in-place solving, but must not partially overlap <paramref name="b"/> or overlap <see cref="L"/>. / <paramref name="b"/> と同じ形状の出力先行列。<paramref name="b"/> と完全に一致する領域ならインプレースで解けますが、部分的な重なりや <see cref="L"/> との重なりは不可です。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">The shapes do not match, or <paramref name="destination"/> overlaps invalidly. / 形状が一致しない、または出力先が不正に重なる場合。</exception>
    public MatrixSpan<double> Solve(ReadOnlyMatrixSpan<double> b, MatrixSpan<double> destination)
    {
        if (b.Columns != BatchCount)
        {
            throw new ArgumentException("The right-hand side must have as many columns as the number of matrices.", nameof(b));
        }

        if (destination.Rows != b.Rows || destination.Columns != b.Columns)
        {
            throw new ArgumentException("Destination shape must agree.", nameof(destination));
        }

        Solve(b.Unsqueeze(1), destination.Unsqueeze(1));
        return destination;
    }

    /// <summary>
    /// Solves the linear systems <c>A[k]X[k] = B[k]</c> for each matrix in the batch with multiple right-hand sides.
    /// <para>バッチ内の各行列について、複数右辺の連立一次方程式 <c>A[k]X[k] = B[k]</c> を解きます。</para>
    /// </summary>
    /// <param name="b">The right-hand side tensor with shape <c>Order × rhsColumns × BatchCount</c>. / 形状 <c>Order × rhsColumns × BatchCount</c> の右辺テンソル。</param>
    /// <returns>A newly allocated tensor containing the solutions. / 解を格納した新しいテンソル。</returns>
    /// <remarks>
    /// <para>Each slice uses the same computation as <see cref="CholeskyDecomposition.Solve(ReadOnlyMatrixSpan{double})"/> (or the vector overload when there is a single right-hand side column).</para>
    /// <para>各スライスは <see cref="CholeskyDecomposition.Solve(ReadOnlyMatrixSpan{double})"/>（右辺が 1 列の場合はベクトル向けオーバーロード）と同じ計算を行います。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The dimensions of <paramref name="b"/> do not match <see cref="Order"/> and <see cref="BatchCount"/>. / <paramref name="b"/> の形状が <see cref="Order"/> および <see cref="BatchCount"/> と一致しない場合。</exception>
    public Tensor<double> Solve(ReadOnlyTensorSpan<double> b)
    {
        var x = Tensor<double>.Uninitialized(b.Rows, b.Columns, b.MatrixCount);
        Solve(b, x);
        return x;
    }

    /// <summary>
    /// Solves the linear systems <c>A[k]X[k] = B[k]</c> for each matrix in the batch and writes the solutions to <paramref name="destination"/>.
    /// <para>バッチ内の各行列について複数右辺の連立一次方程式 <c>A[k]X[k] = B[k]</c> を解き、解を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="b">The right-hand side tensor. / 右辺テンソル。</param>
    /// <param name="destination">The destination tensor with the same shape as <paramref name="b"/>. May identically alias <paramref name="b"/> for in-place solving, but must not partially overlap <paramref name="b"/> or overlap <see cref="L"/>. / <paramref name="b"/> と同じ形状の出力先テンソル。<paramref name="b"/> と完全に一致する領域ならインプレースで解けますが、部分的な重なりや <see cref="L"/> との重なりは不可です。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">The shapes do not match, or <paramref name="destination"/> overlaps invalidly. / 形状が一致しない、または出力先が不正に重なる場合。</exception>
    public unsafe TensorSpan<double> Solve(ReadOnlyTensorSpan<double> b, TensorSpan<double> destination)
    {
        int n = Order;
        if (b.Rows != n || b.MatrixCount != BatchCount)
        {
            throw new ArgumentException("The right-hand sides must have as many rows as the order and as many matrices as the batch.", nameof(b));
        }

        if (destination.Rows != b.Rows || destination.Columns != b.Columns || destination.MatrixCount != b.MatrixCount)
        {
            throw new ArgumentException("Destination shape must agree.", nameof(destination));
        }

        StridedKernels.RequireDisjoint(L.AsTensorSpan().AsReadOnly(), destination);
        StridedKernels.CopyRightHandSide(b, destination);
        if (destination.IsEmpty)
        {
            return destination;
        }

        var factor = L.Data;
        var view = destination.AsReadOnly();
        var (rows, columns, rs, cs, ms) = (view.Rows, view.Columns, view.RowStride, view.ColumnStride, view.MatrixStride);
        bool vector = StridedKernels.VectorPath(columns);
        fixed (double* px = &Unsafe.AsRef(in view.Reference))
        {
            nint origin = (nint)px;
            StridedKernels.ForEachMatrix(BatchCount, (double)n * n * columns, (start, end) =>
            {
                for (int k = start; k < end; k++)
                {
                    var x = new MatrixSpan<double>(ref Unsafe.AsRef<double>((double*)origin + (nint)k * ms), rows, columns, rs, cs);
                    StridedKernels.SolveCholeskyInPlace(n, factor.AsSpan(k * n * n, n * n), x, vector);
                }
            });
        }

        return destination;
    }

    /// <summary>
    /// Computes the inverse <c>A[k]⁻¹</c> for each matrix in the batch.
    /// <para>バッチ内の各行列の逆行列 <c>A[k]⁻¹</c> を計算します。</para>
    /// </summary>
    /// <returns>A newly allocated tensor containing the inverses (slices for matrices that were not positive definite are <see cref="double.NaN"/>). / 逆行列を格納した新しいテンソル（正定値でなかった行列のスライスは <see cref="double.NaN"/> になります）。</returns>
    public Tensor<double> Inverse()
    {
        int n = Order;
        var result = Tensor<double>.Uninitialized(n, n, BatchCount);
        var factor = L.Data;
        var data = result.Data;
        StridedKernels.ForEachMatrix(BatchCount, 2.0 / 3 * n * n * n, (start, end) =>
        {
            for (int k = start; k < end; k++)
            {
                CholeskyKernels.Inverse(n, factor.AsSpan(k * n * n, n * n), n, data.AsSpan(k * n * n, n * n), n);
            }
        });

        return result;
    }

    /// <summary>
    /// Computes the inverse <c>A[k]⁻¹</c> for each matrix in the batch and writes the results to <paramref name="destination"/>.
    /// <para>バッチ内の各行列の逆行列 <c>A[k]⁻¹</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="destination">The <c>n × n × BatchCount</c> destination tensor, which must not overlap <see cref="L"/>. / <c>n × n × BatchCount</c> の出力先テンソル（<see cref="L"/> と重なってはなりません）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>Slices for matrices where <see cref="IsPositiveDefinite"/> is <see langword="false"/> are filled with <see cref="double.NaN"/>.</para>
    /// <para><see cref="IsPositiveDefinite"/> が <see langword="false"/> の行列に対応するスライスは <see cref="double.NaN"/> で埋められます。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The shape does not match, or <paramref name="destination"/> overlaps <see cref="L"/>. / 形状が一致しない、または <paramref name="destination"/> が <see cref="L"/> と重なる場合。</exception>
    public unsafe TensorSpan<double> Inverse(TensorSpan<double> destination)
    {
        int n = Order;
        if (destination.Rows != n || destination.Columns != n || destination.MatrixCount != BatchCount)
        {
            throw new ArgumentException("Destination shape must agree.", nameof(destination));
        }

        StridedKernels.RequireDisjoint(L.AsTensorSpan().AsReadOnly(), destination);
        if (destination.IsEmpty)
        {
            return destination;
        }

        var factor = L.Data;
        var view = destination.AsReadOnly();
        var (rs, cs, ms) = (view.RowStride, view.ColumnStride, view.MatrixStride);
        fixed (double* px = &Unsafe.AsRef(in view.Reference))
        {
            nint origin = (nint)px;
            StridedKernels.ForEachMatrix(BatchCount, 2.0 / 3 * n * n * n, (start, end) =>
            {
                for (int k = start; k < end; k++)
                {
                    var x = new MatrixSpan<double>(ref Unsafe.AsRef<double>((double*)origin + (nint)k * ms), n, n, rs, cs);
                    StridedKernels.InverseCholesky(n, factor.AsSpan(k * n * n, n * n), x);
                }
            });
        }

        return destination;
    }

    private static BatchCholeskyDecomposition Factor(ReadOnlyTensorSpan<double> a, Tensor<double> destination)
    {
        ArgumentNullException.ThrowIfNull(destination);
        if (a.Rows != a.Columns)
        {
            throw new ArgumentException("The matrices must be square.", nameof(a));
        }

        if (destination.Rows != a.Rows || destination.Columns != a.Columns || destination.MatrixCount != a.MatrixCount)
        {
            throw new ArgumentException("Destination shape must agree.", nameof(destination));
        }

        // 出力先が入力と完全に同じ要素なら写さない（部分的な重なりは例外）
        StridedKernels.CopyInput(a, destination.AsTensorSpan());

        int n = a.Rows;
        int count = a.MatrixCount;
        var data = destination.Data;
        var ok = new bool[count];
        var norms = new double[count];
        StridedKernels.ForEachMatrix(count, (double)n * n * n / 3, (start, end) =>
        {
            for (int k = start; k < end; k++)
            {
                var l = data.AsSpan(k * n * n, n * n);
                norms[k] = ConditionKernels.SymmetricNorm1(n, l, n);
                if (CholeskyKernels.Factor(n, l, n) != 0)
                {
                    l.Fill(double.NaN);
                    continue;
                }

                // 上三角部分（作業中に書き換わったものを含む）を0にする
                for (int j = 1; j < n; j++)
                {
                    l.Slice(j * n, j).Clear();
                }

                ok[k] = true;
            }
        });

        int failures = 0;
        foreach (bool b in ok)
        {
            if (!b)
            {
                failures++;
            }
        }

        return new BatchCholeskyDecomposition(destination, ok, failures, norms);
    }
}
