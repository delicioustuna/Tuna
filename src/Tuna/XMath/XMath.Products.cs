using System.Numerics;

namespace Tuna;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

public static partial class XMath
{
    /// <summary>
    /// Computes the matrix product <c>AB</c> and returns a new matrix.
    /// <para>行列積 <c>AB</c> を計算し、新しい行列を返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="a">The left input matrix. / 左側の入力行列。</param>
    /// <param name="b">The right input matrix. / 右側の入力行列。</param>
    /// <returns>A new matrix containing <c>AB</c>. / 行列積 <c>AB</c> を格納した新しい行列。</returns>
    /// <remarks>
    /// <para>Transposed views (such as <c>A.T</c>) are passed directly as GEMM transpose flags without copying.</para>
    /// <para>転置ビュー（<c>A.T</c> など）はコピーせず GEMM の転置フラグとして渡されます。</para>
    /// </remarks>
    /// <exception cref="OverflowException">The number of elements <c>a.Rows · b.Columns</c> exceeds <see cref="int.MaxValue"/>. / 結果の要素数 <c>a.Rows · b.Columns</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    [InferenceOverloads]
    public static Matrix<T> Multiply<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b)
        where T : unmanaged, INumberBase<T>
    {
        var result = Matrix<T>.Uninitialized(a.Rows, b.Columns);
        Multiply(a, b, result);
        return result;
    }

    /// <summary>
    /// Computes the matrix product <c>AB</c> and writes the result to <paramref name="destination"/>.
    /// <para>行列積 <c>AB</c> を計算し、結果を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="a">The left input matrix. / 左側の入力行列。</param>
    /// <param name="b">The right input matrix. / 右側の入力行列。</param>
    /// <param name="destination">The destination span to receive the product. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap either input.</para>
    /// <para><paramref name="destination"/> はいずれの入力とも領域が重なってはなりません。</para>
    /// </remarks>
    [InferenceOverloads]
    public static MatrixSpan<T> Multiply<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Multiply(a, b, destination);
        return destination;
    }

    /// <summary>
    /// Computes the matrix-vector product <c>Ax</c> and returns a new vector.
    /// <para>行列ベクトル積 <c>Ax</c> を計算し、新しいベクトルを返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="a">The input matrix. / 入力行列。</param>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>A new vector containing <c>Ax</c>. / 行列ベクトル積 <c>Ax</c> を格納した新しいベクトル。</returns>
    /// <remarks>
    /// <para>Transposed views are passed directly as GEMV transpose flags without copying.</para>
    /// <para>転置ビューはコピーせず GEMV の転置フラグとして渡されます。</para>
    /// </remarks>
    [InferenceOverloads]
    public static Vector<T> Multiply<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyVectorSpan<T> x)
        where T : unmanaged, INumberBase<T>
    {
        var result = Vector<T>.Uninitialized(a.Rows);
        Multiply(a, x, result);
        return result;
    }

    /// <summary>
    /// Computes the matrix-vector product <c>Ax</c> and writes the result to <paramref name="destination"/>.
    /// <para>行列ベクトル積 <c>Ax</c> を計算し、結果を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="a">The input matrix. / 入力行列。</param>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="destination">The destination span to receive the product. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap either input.</para>
    /// <para><paramref name="destination"/> はいずれの入力とも領域が重なってはなりません。</para>
    /// </remarks>
    [InferenceOverloads]
    public static VectorSpan<T> Multiply<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Multiply(a, x, destination);
        return destination;
    }

    /// <summary>
    /// Returns a newly allocated transposed copy <c>Aᵀ</c> of the input matrix.
    /// <para>入力行列を転置した新しいコピー <c>Aᵀ</c> を返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new matrix containing the transpose of <paramref name="x"/>. / <paramref name="x"/> を転置した新しい行列。</returns>
    /// <remarks>
    /// <para>For a zero-copy transposed view, use <c>x.T</c>.</para>
    /// <para>コピーを伴わない転置ビューが必要な場合は <c>x.T</c> を使用してください。</para>
    /// </remarks>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Matrix<T> Transpose<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumberBase<T>
    {
        var result = Matrix<T>.Uninitialized(x.Columns, x.Rows);
        Transpose(x, result);
        return result;
    }

    /// <summary>
    /// Writes the transpose <c>Aᵀ</c> of the input matrix into <paramref name="destination"/>.
    /// <para>入力行列の転置 <c>Aᵀ</c> を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span to receive the transposed matrix. / 転置結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/> (in-place transposition is not supported).</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません（インプレース転置には対応していません）。</para>
    /// </remarks>
    [InferenceOverloads]
    public static MatrixSpan<T> Transpose<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Copy(x.T, destination);
        return destination;
    }
}
