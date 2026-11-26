using System.Numerics;

namespace Tuna.LinearAlgebra;

/// <summary>
/// Provides extension methods for converting arrays and spans to dense containers and invoking BLAS operations and matrix decompositions.
/// <para>配列やスパンから密コンテナへの変換、および BLAS 演算や行列分解を呼び出す拡張メソッドを提供します。</para>
/// </summary>
public static class DenseExtensions
{
    /// <summary>
    /// Creates a new <see cref="Vector{T}"/> by copying the elements of the specified array.
    /// <para>指定した配列の要素をコピーして新しい <see cref="Vector{T}"/> を作成します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="values">The source array. / コピー元の配列。</param>
    /// <returns>A new vector containing the copied elements. / 要素をコピーした新しいベクトル。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="values"/> is <see langword="null"/>. / <paramref name="values"/> が <see langword="null"/> の場合。</exception>
    public static Vector<T> ToVector<T>(this T[] values) where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(values);
        return new(values);
    }

    /// <summary>
    /// Creates a new <see cref="Vector{T}"/> by copying the elements of the specified read-only span.
    /// <para>指定した読み取り専用スパンの要素をコピーして新しい <see cref="Vector{T}"/> を作成します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="values">The source span. / コピー元のスパン。</param>
    /// <returns>A new vector containing the copied elements. / 要素をコピーした新しいベクトル。</returns>
    public static Vector<T> ToVector<T>(this ReadOnlySpan<T> values) where T : unmanaged, INumberBase<T> => new(values);

    /// <summary>
    /// Creates a new <see cref="Vector{T}"/> by copying the elements of the specified span.
    /// <para>指定したスパンの要素をコピーして新しい <see cref="Vector{T}"/> を作成します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="values">The source span. / コピー元のスパン。</param>
    /// <returns>A new vector containing the copied elements. / 要素をコピーした新しいベクトル。</returns>
    public static Vector<T> ToVector<T>(this Span<T> values) where T : unmanaged, INumberBase<T> => new(values);

    /// <summary>
    /// Creates a new <see cref="Matrix{TElement}"/> by copying the elements of the specified two-dimensional array.
    /// <para>指定した 2 次元配列の要素をコピーして新しい <see cref="Matrix{TElement}"/> を作成します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="values">The source two-dimensional array. / コピー元の 2 次元配列。</param>
    /// <returns>A new matrix containing the copied elements. / 要素をコピーした新しい行列。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="values"/> is <see langword="null"/>. / <paramref name="values"/> が <see langword="null"/> の場合。</exception>
    public static Matrix<T> ToMatrix<T>(this T[,] values) where T : unmanaged, INumberBase<T> => new(values);

    /// <summary>
    /// Computes the Frobenius norm <c>‖A‖_F</c> of the matrix without intermediate overflow or underflow.
    /// <para>中間のオーバーフローやアンダーフローを回避して、行列のフロベニウスノルム <c>‖A‖_F</c> を計算します。</para>
    /// </summary>
    /// <param name="matrix">The input matrix. / 入力行列。</param>
    /// <returns>The Frobenius norm of <paramref name="matrix"/>. / <paramref name="matrix"/> のフロベニウスノルム。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="matrix"/> is <see langword="null"/>. / <paramref name="matrix"/> が <see langword="null"/> の場合。</exception>
    public static double FrobeniusNorm(this Matrix<double> matrix)
    {
        ArgumentNullException.ThrowIfNull(matrix);
        return Level1Kernels.Norm2(matrix.Data);
    }

    /// <summary>
    /// Computes the Cholesky decomposition <c>A = LLᵀ</c> of the symmetric positive-definite matrix.
    /// <para>対称正定値行列のコレスキー分解 <c>A = LLᵀ</c> を計算します。</para>
    /// </summary>
    /// <param name="matrix">The input matrix (only the lower triangle is referenced). / 入力行列（下三角部分のみ参照）。</param>
    /// <returns>The Cholesky decomposition of <paramref name="matrix"/>. / <paramref name="matrix"/> のコレスキー分解結果。</returns>
    public static CholeskyDecomposition Cholesky(this Matrix<double> matrix) => CholeskyDecomposition.Compute(matrix);

    /// <summary>
    /// Computes the Cholesky decomposition <c>A = LLᵀ</c> of the symmetric positive-definite matrix view.
    /// <para>対称正定値行列ビューのコレスキー分解 <c>A = LLᵀ</c> を計算します。</para>
    /// </summary>
    /// <param name="matrix">The input matrix view (only the lower triangle is referenced). / 入力行列ビュー（下三角部分のみ参照）。</param>
    /// <returns>The Cholesky decomposition of <paramref name="matrix"/>. / <paramref name="matrix"/> のコレスキー分解結果。</returns>
    public static CholeskyDecomposition Cholesky(this ReadOnlyMatrixSpan<double> matrix) => CholeskyDecomposition.Compute(matrix);

    /// <summary>
    /// Computes the Cholesky decomposition <c>A = LLᵀ</c> of the symmetric positive-definite matrix view.
    /// <para>対称正定値行列ビューのコレスキー分解 <c>A = LLᵀ</c> を計算します。</para>
    /// </summary>
    /// <param name="matrix">The input matrix view (only the lower triangle is referenced). / 入力行列ビュー（下三角部分のみ参照）。</param>
    /// <returns>The Cholesky decomposition of <paramref name="matrix"/>. / <paramref name="matrix"/> のコレスキー分解結果。</returns>
    public static CholeskyDecomposition Cholesky(this MatrixSpan<double> matrix) => CholeskyDecomposition.Compute(matrix.AsReadOnly());

    /// <summary>
    /// Computes the LU decomposition with partial pivoting <c>PA = LU</c> of the square matrix.
    /// <para>正方行列の部分ピボット付き LU 分解 <c>PA = LU</c> を計算します。</para>
    /// </summary>
    /// <param name="matrix">The input square matrix. / 入力正方行列。</param>
    /// <returns>The LU decomposition of <paramref name="matrix"/>. / <paramref name="matrix"/> の LU 分解結果。</returns>
    public static LuDecomposition Lu(this Matrix<double> matrix) => LuDecomposition.Compute(matrix);

    /// <summary>
    /// Computes the LU decomposition with partial pivoting <c>PA = LU</c> of the square matrix view.
    /// <para>正方行列ビューの部分ピボット付き LU 分解 <c>PA = LU</c> を計算します。</para>
    /// </summary>
    /// <param name="matrix">The input square matrix view. / 入力正方行列ビュー。</param>
    /// <returns>The LU decomposition of <paramref name="matrix"/>. / <paramref name="matrix"/> の LU 分解結果。</returns>
    public static LuDecomposition Lu(this ReadOnlyMatrixSpan<double> matrix) => LuDecomposition.Compute(matrix);

    /// <summary>
    /// Computes the LU decomposition with partial pivoting <c>PA = LU</c> of the square matrix view.
    /// <para>正方行列ビューの部分ピボット付き LU 分解 <c>PA = LU</c> を計算します。</para>
    /// </summary>
    /// <param name="matrix">The input square matrix view. / 入力正方行列ビュー。</param>
    /// <returns>The LU decomposition of <paramref name="matrix"/>. / <paramref name="matrix"/> の LU 分解結果。</returns>
    public static LuDecomposition Lu(this MatrixSpan<double> matrix) => LuDecomposition.Compute(matrix.AsReadOnly());

    /// <summary>
    /// Computes the thin QR decomposition <c>A = QR</c> of the matrix.
    /// <para>行列の薄い QR 分解 <c>A = QR</c> を計算します。</para>
    /// </summary>
    /// <param name="matrix">The input matrix. / 入力行列。</param>
    /// <returns>The QR decomposition of <paramref name="matrix"/>. / <paramref name="matrix"/> の QR 分解結果。</returns>
    public static QrDecomposition Qr(this Matrix<double> matrix) => QrDecomposition.Compute(matrix);

    /// <summary>
    /// Computes the thin QR decomposition <c>A = QR</c> of the matrix view.
    /// <para>行列ビューの薄い QR 分解 <c>A = QR</c> を計算します。</para>
    /// </summary>
    /// <param name="matrix">The input matrix view. / 入力行列ビュー。</param>
    /// <returns>The QR decomposition of <paramref name="matrix"/>. / <paramref name="matrix"/> の QR 分解結果。</returns>
    public static QrDecomposition Qr(this ReadOnlyMatrixSpan<double> matrix) => QrDecomposition.Compute(matrix);

    /// <summary>
    /// Computes the thin QR decomposition <c>A = QR</c> of the matrix view.
    /// <para>行列ビューの薄い QR 分解 <c>A = QR</c> を計算します。</para>
    /// </summary>
    /// <param name="matrix">The input matrix view. / 入力行列ビュー。</param>
    /// <returns>The QR decomposition of <paramref name="matrix"/>. / <paramref name="matrix"/> の QR 分解結果。</returns>
    public static QrDecomposition Qr(this MatrixSpan<double> matrix) => QrDecomposition.Compute(matrix.AsReadOnly());

    /// <summary>
    /// Computes the eigendecomposition <c>A = VΛVᵀ</c> of a real symmetric matrix.
    /// <para>実対称行列の固有値分解 <c>A = VΛVᵀ</c> を計算します。</para>
    /// </summary>
    /// <param name="matrix">The input symmetric matrix (only the lower triangle is referenced). / 入力対称行列（下三角部分のみ参照）。</param>
    /// <param name="computeVectors"><see langword="true"/> to also compute eigenvectors; <see langword="false"/> to compute eigenvalues only. / 固有ベクトルも計算する場合は <see langword="true"/>、固有値のみの場合は <see langword="false"/>。</param>
    /// <returns>The symmetric eigendecomposition of <paramref name="matrix"/>. / <paramref name="matrix"/> の対称固有値分解結果。</returns>
    public static SymmetricEigenDecomposition SymmetricEigen(this Matrix<double> matrix, bool computeVectors = true) =>
        SymmetricEigenDecomposition.Compute(matrix, computeVectors);

    /// <summary>
    /// Computes the eigendecomposition <c>A = VΛVᵀ</c> of a real symmetric matrix view.
    /// <para>実対称行列ビューの固有値分解 <c>A = VΛVᵀ</c> を計算します。</para>
    /// </summary>
    /// <param name="matrix">The input symmetric matrix view (only the lower triangle is referenced). / 入力対称行列ビュー（下三角部分のみ参照）。</param>
    /// <param name="computeVectors"><see langword="true"/> to also compute eigenvectors; <see langword="false"/> to compute eigenvalues only. / 固有ベクトルも計算する場合は <see langword="true"/>、固有値のみの場合は <see langword="false"/>。</param>
    /// <returns>The symmetric eigendecomposition of <paramref name="matrix"/>. / <paramref name="matrix"/> の対称固有値分解結果。</returns>
    public static SymmetricEigenDecomposition SymmetricEigen(this ReadOnlyMatrixSpan<double> matrix, bool computeVectors = true) =>
        SymmetricEigenDecomposition.Compute(matrix, computeVectors);

    /// <summary>
    /// Computes the eigendecomposition <c>A = VΛVᵀ</c> of a real symmetric matrix view.
    /// <para>実対称行列ビューの固有値分解 <c>A = VΛVᵀ</c> を計算します。</para>
    /// </summary>
    /// <param name="matrix">The input symmetric matrix view (only the lower triangle is referenced). / 入力対称行列ビュー（下三角部分のみ参照）。</param>
    /// <param name="computeVectors"><see langword="true"/> to also compute eigenvectors; <see langword="false"/> to compute eigenvalues only. / 固有ベクトルも計算する場合は <see langword="true"/>、固有値のみの場合は <see langword="false"/>。</param>
    /// <returns>The symmetric eigendecomposition of <paramref name="matrix"/>. / <paramref name="matrix"/> の対称固有値分解結果。</returns>
    public static SymmetricEigenDecomposition SymmetricEigen(this MatrixSpan<double> matrix, bool computeVectors = true) =>
        SymmetricEigenDecomposition.Compute(matrix.AsReadOnly(), computeVectors);

    /// <summary>
    /// Computes the eigendecomposition <c>AV = VD</c> of a general real square matrix.
    /// <para>一般の実正方行列の固有値分解 <c>AV = VD</c> を計算します。</para>
    /// </summary>
    /// <param name="matrix">The input square matrix. / 入力正方行列。</param>
    /// <param name="computeVectors"><see langword="true"/> to also compute eigenvectors; <see langword="false"/> to compute eigenvalues only. / 固有ベクトルも計算する場合は <see langword="true"/>、固有値のみの場合は <see langword="false"/>。</param>
    /// <returns>The eigendecomposition of <paramref name="matrix"/>. / <paramref name="matrix"/> の固有値分解結果。</returns>
    public static EigenDecomposition Eigen(this Matrix<double> matrix, bool computeVectors = true) =>
        EigenDecomposition.Compute(matrix, computeVectors);

    /// <summary>
    /// Computes the eigendecomposition <c>AV = VD</c> of a general real square matrix view.
    /// <para>一般の実正方行列ビューの固有値分解 <c>AV = VD</c> を計算します。</para>
    /// </summary>
    /// <param name="matrix">The input square matrix view. / 入力正方行列ビュー。</param>
    /// <param name="computeVectors"><see langword="true"/> to also compute eigenvectors; <see langword="false"/> to compute eigenvalues only. / 固有ベクトルも計算する場合は <see langword="true"/>、固有値のみの場合は <see langword="false"/>。</param>
    /// <returns>The eigendecomposition of <paramref name="matrix"/>. / <paramref name="matrix"/> の固有値分解結果。</returns>
    public static EigenDecomposition Eigen(this ReadOnlyMatrixSpan<double> matrix, bool computeVectors = true) =>
        EigenDecomposition.Compute(matrix, computeVectors);

    /// <summary>
    /// Computes the eigendecomposition <c>AV = VD</c> of a general real square matrix view.
    /// <para>一般の実正方行列ビューの固有値分解 <c>AV = VD</c> を計算します。</para>
    /// </summary>
    /// <param name="matrix">The input square matrix view. / 入力正方行列ビュー。</param>
    /// <param name="computeVectors"><see langword="true"/> to also compute eigenvectors; <see langword="false"/> to compute eigenvalues only. / 固有ベクトルも計算する場合は <see langword="true"/>、固有値のみの場合は <see langword="false"/>。</param>
    /// <returns>The eigendecomposition of <paramref name="matrix"/>. / <paramref name="matrix"/> の固有値分解結果。</returns>
    public static EigenDecomposition Eigen(this MatrixSpan<double> matrix, bool computeVectors = true) =>
        EigenDecomposition.Compute(matrix.AsReadOnly(), computeVectors);

    /// <summary>
    /// Computes the thin singular value decomposition <c>A = UΣVᵀ</c> of the matrix.
    /// <para>行列の薄い特異値分解 <c>A = UΣVᵀ</c> を計算します。</para>
    /// </summary>
    /// <param name="matrix">The input matrix. / 入力行列。</param>
    /// <param name="computeVectors"><see langword="true"/> to also compute singular vectors; <see langword="false"/> to compute singular values only. / 特異ベクトルも計算する場合は <see langword="true"/>、特異値のみの場合は <see langword="false"/>。</param>
    /// <returns>The singular value decomposition of <paramref name="matrix"/>. / <paramref name="matrix"/> の特異値分解結果。</returns>
    public static SingularValueDecomposition Svd(this Matrix<double> matrix, bool computeVectors = true) =>
        SingularValueDecomposition.Compute(matrix, computeVectors);

    /// <summary>
    /// Computes the thin singular value decomposition <c>A = UΣVᵀ</c> of the matrix view.
    /// <para>行列ビューの薄い特異値分解 <c>A = UΣVᵀ</c> を計算します。</para>
    /// </summary>
    /// <param name="matrix">The input matrix view. / 入力行列ビュー。</param>
    /// <param name="computeVectors"><see langword="true"/> to also compute singular vectors; <see langword="false"/> to compute singular values only. / 特異ベクトルも計算する場合は <see langword="true"/>、特異値のみの場合は <see langword="false"/>。</param>
    /// <returns>The singular value decomposition of <paramref name="matrix"/>. / <paramref name="matrix"/> の特異値分解結果。</returns>
    public static SingularValueDecomposition Svd(this ReadOnlyMatrixSpan<double> matrix, bool computeVectors = true) =>
        SingularValueDecomposition.Compute(matrix, computeVectors);

    /// <summary>
    /// Computes the thin singular value decomposition <c>A = UΣVᵀ</c> of the matrix view.
    /// <para>行列ビューの薄い特異値分解 <c>A = UΣVᵀ</c> を計算します。</para>
    /// </summary>
    /// <param name="matrix">The input matrix view. / 入力行列ビュー。</param>
    /// <param name="computeVectors"><see langword="true"/> to also compute singular vectors; <see langword="false"/> to compute singular values only. / 特異ベクトルも計算する場合は <see langword="true"/>、特異値のみの場合は <see langword="false"/>。</param>
    /// <returns>The singular value decomposition of <paramref name="matrix"/>. / <paramref name="matrix"/> の特異値分解結果。</returns>
    public static SingularValueDecomposition Svd(this MatrixSpan<double> matrix, bool computeVectors = true) =>
        SingularValueDecomposition.Compute(matrix.AsReadOnly(), computeVectors);

    /// <summary>
    /// Computes the dot product of two double-precision vectors.
    /// <para>2 つの倍精度ベクトルの内積を計算します。</para>
    /// </summary>
    /// <param name="x">The first input vector. / 第 1 入力ベクトル。</param>
    /// <param name="y">The second input vector. / 第 2 入力ベクトル。</param>
    /// <returns>The dot product <c>Σ x[i]·y[i]</c>. / 内積 <c>Σ x[i]·y[i]</c>。</returns>
    public static double Dot(this Vector<double> x, Vector<double> y) => Blas.Dot(x.AsSpan(), y.AsSpan());

    /// <summary>
    /// Computes the dot product of two single-precision vectors.
    /// <para>2 つの単精度ベクトルの内積を計算します。</para>
    /// </summary>
    /// <param name="x">The first input vector. / 第 1 入力ベクトル。</param>
    /// <param name="y">The second input vector. / 第 2 入力ベクトル。</param>
    /// <returns>The dot product <c>Σ x[i]·y[i]</c>. / 内積 <c>Σ x[i]·y[i]</c>。</returns>
    public static float Dot(this Vector<float> x, Vector<float> y) => Blas.Dot(x.AsSpan(), y.AsSpan());

    /// <summary>
    /// Computes the unconjugated dot product of two complex vectors.
    /// <para>2 つの複素ベクトルの共役を取らない内積を計算します。</para>
    /// </summary>
    /// <param name="x">The first input vector. / 第 1 入力ベクトル。</param>
    /// <param name="y">The second input vector. / 第 2 入力ベクトル。</param>
    /// <returns>The unconjugated dot product <c>Σ x[i]·y[i]</c>. / 共役を取らない内積 <c>Σ x[i]·y[i]</c>。</returns>
    public static Complex Dot(this Vector<Complex> x, Vector<Complex> y) => Blas.Dot(x.AsSpan(), y.AsSpan());

    /// <summary>
    /// Computes the Hermitian dot product <c>Σ conj(x[i])·y[i]</c> of two complex vectors.
    /// <para>2 つの複素ベクトルのエルミート内積 <c>Σ conj(x[i])·y[i]</c> を計算します。</para>
    /// </summary>
    /// <param name="x">The first input vector (conjugated). / 第 1 入力ベクトル（共役を取ります）。</param>
    /// <param name="y">The second input vector. / 第 2 入力ベクトル。</param>
    /// <returns>The conjugate dot product <c>Σ conj(x[i])·y[i]</c>. / 共役内積 <c>Σ conj(x[i])·y[i]</c>。</returns>
    public static Complex DotConjugate(this Vector<Complex> x, Vector<Complex> y) => Blas.DotConjugate(x.AsSpan(), y.AsSpan());

    /// <summary>
    /// Computes the Euclidean (L2) norm of a double-precision vector.
    /// <para>倍精度ベクトルのユークリッドノルム（L2 ノルム）を計算します。</para>
    /// </summary>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>The Euclidean norm <c>‖x‖₂</c>. / ユークリッドノルム <c>‖x‖₂</c>。</returns>
    public static double Norm2(this Vector<double> x) => Blas.Norm2(x.AsSpan());

    /// <summary>
    /// Computes the Euclidean (L2) norm of a single-precision vector.
    /// <para>単精度ベクトルのユークリッドノルム（L2 ノルム）を計算します。</para>
    /// </summary>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>The Euclidean norm <c>‖x‖₂</c>. / ユークリッドノルム <c>‖x‖₂</c>。</returns>
    public static float Norm2(this Vector<float> x) => Blas.Norm2(x.AsSpan());

    /// <summary>
    /// Computes the Euclidean (L2) norm of a complex vector.
    /// <para>複素ベクトルのユークリッドノルム（L2 ノルム）を計算します。</para>
    /// </summary>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>The real-valued Euclidean norm <c>‖x‖₂</c>. / 実数値のユークリッドノルム <c>‖x‖₂</c>。</returns>
    public static double Norm2(this Vector<Complex> x) => Blas.Norm2(x.AsSpan());
}
