namespace Tuna.LinearAlgebra;

/// <summary>
/// Specifies the transpose operation applied to a matrix in BLAS routines (corresponding to the BLAS <c>TRANS</c> parameter).
/// <para>BLAS ルーチンで行列に適用する転置操作を指定します（BLAS の <c>TRANS</c> 引数に相当）。</para>
/// </summary>
public enum BlasTranspose
{
    /// <summary>
    /// Uses the matrix as-is, <c>A</c> (<c>'N'</c>).
    /// <para>行列をそのまま <c>A</c> として使用します（<c>'N'</c>）。</para>
    /// </summary>
    None,

    /// <summary>
    /// Uses the transpose <c>Aᵀ</c> (<c>'T'</c>).
    /// <para>転置 <c>Aᵀ</c> を使用します（<c>'T'</c>）。</para>
    /// </summary>
    Transpose,

    /// <summary>
    /// Uses the conjugate transpose <c>Aᴴ</c> (<c>'C'</c>).
    /// <para>共役転置 <c>Aᴴ</c> を使用します（<c>'C'</c>）。</para>
    /// </summary>
    /// <remarks>
    /// <para>For real element types, this is equivalent to <see cref="Transpose"/>.</para>
    /// <para>実数型では <see cref="Transpose"/> と同じ動作になります。</para>
    /// </remarks>
    ConjugateTranspose,
}

/// <summary>
/// Specifies which triangular part of a matrix is referenced in BLAS routines (corresponding to the BLAS <c>UPLO</c> parameter).
/// <para>BLAS ルーチンで参照する行列の三角部分を指定します（BLAS の <c>UPLO</c> 引数に相当）。</para>
/// </summary>
/// <remarks>
/// <para>For symmetric and Hermitian matrices, elements in the opposite triangle are not read.</para>
/// <para>対称行列およびエルミート行列では、反対側の三角部分の要素は参照されません。</para>
/// </remarks>
public enum BlasTriangle
{
    /// <summary>
    /// References the upper triangular part (<c>'U'</c>).
    /// <para>上三角部分を参照します（<c>'U'</c>）。</para>
    /// </summary>
    Upper,

    /// <summary>
    /// References the lower triangular part (<c>'L'</c>).
    /// <para>下三角部分を参照します（<c>'L'</c>）。</para>
    /// </summary>
    Lower,
}

/// <summary>
/// Specifies whether a triangular matrix is assumed to have unit diagonal elements (corresponding to the BLAS <c>DIAG</c> parameter).
/// <para>三角行列の対角要素を 1 とみなすかどうかを指定します（BLAS の <c>DIAG</c> 引数に相当）。</para>
/// </summary>
public enum BlasDiagonal
{
    /// <summary>
    /// Uses the stored diagonal elements (<c>'N'</c>).
    /// <para>格納されている対角要素の値をそのまま使用します（<c>'N'</c>）。</para>
    /// </summary>
    NonUnit,

    /// <summary>
    /// Assumes all diagonal elements are 1 and does not read the stored diagonal entries (<c>'U'</c>).
    /// <para>すべての対角要素を 1 とみなし、格納されている対角要素は参照しません（<c>'U'</c>）。</para>
    /// </summary>
    Unit,
}
