using System.Numerics;

namespace Tuna;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

public static partial class XMath
{
    /// <summary>
    /// Solves the lower-triangular linear system <c>Lx = b</c> and returns the solution vector.
    /// <para>下三角行列を係数とする連立一次方程式 <c>Lx = b</c> を解き、解ベクトルを返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="a">The coefficient matrix (only the lower triangle, including the diagonal, is read). / 係数行列（対角を含む下三角部分のみ参照）。</param>
    /// <param name="b">The right-hand side vector. / 右辺ベクトル。</param>
    /// <returns>A new vector containing the solution <c>x</c>. / 解 <c>x</c> を格納した新しいベクトル。</returns>
    /// <remarks>
    /// <para>To solve <c>Lᵀx = b</c>, call <c>SolveUpperTriangular(L.T, b)</c>. Zero diagonal entries are not checked and may produce <c>±∞</c> or <c>NaN</c>.</para>
    /// <para><c>Lᵀx = b</c> は <c>SolveUpperTriangular(L.T, b)</c> で解けます（転置ビューはコピーされません）。対角の 0 検査は行わないため、<c>±∞</c> や <c>NaN</c> が返る場合があります。</para>
    /// </remarks>
    [InferenceOverloads]
    public static Vector<T> SolveLowerTriangular<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyVectorSpan<T> b)
        where T : unmanaged, INumberBase<T>
    {
        var result = Vector<T>.Uninitialized(b.Length);
        SolveLowerTriangular(a, b, result);
        return result;
    }

    /// <summary>
    /// Solves the lower-triangular linear system <c>Lx = b</c> and writes the solution to <paramref name="destination"/>.
    /// <para>下三角行列を係数とする連立一次方程式 <c>Lx = b</c> を解き、解を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="a">The coefficient matrix (lower triangle read). / 係数行列（下三角部分のみ参照）。</param>
    /// <param name="b">The right-hand side vector. / 右辺ベクトル。</param>
    /// <param name="destination">The destination span to receive the solution. / 解を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> may identically alias <paramref name="b"/> (in-place solve). Overlap with <paramref name="a"/> or partial overlap with <paramref name="b"/> throws.</para>
    /// <para><paramref name="destination"/> は <paramref name="b"/> と完全に同一の領域（インプレース求解）を指定できます。<paramref name="a"/> との重なりや <paramref name="b"/> との部分的な重なりは例外となります。</para>
    /// </remarks>
    [InferenceOverloads]
    public static VectorSpan<T> SolveLowerTriangular<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyVectorSpan<T> b, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.SolveTriangular(a, b.AsMatrix(), destination.AsMatrix(), lower: true, vector: true);
        return destination;
    }

    /// <summary>
    /// Solves the lower-triangular linear system <c>LX = B</c> with multiple right-hand sides and returns the solution matrix.
    /// <para>下三角行列を係数とする複数右辺の連立一次方程式 <c>LX = B</c> を解き、解行列を返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="a">The coefficient matrix (lower triangle read). / 係数行列（下三角部分のみ参照）。</param>
    /// <param name="b">The right-hand side matrix. / 右辺行列。</param>
    /// <returns>A new matrix containing the solution <c>X</c>. / 解 <c>X</c> を格納した新しい行列。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Matrix<T> SolveLowerTriangular<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b)
        where T : unmanaged, INumberBase<T>
    {
        var result = Matrix<T>.Uninitialized(b.Rows, b.Columns);
        SolveLowerTriangular(a, b, result);
        return result;
    }

    /// <summary>
    /// Solves the lower-triangular linear system <c>LX = B</c> and writes the solution to <paramref name="destination"/>.
    /// <para>下三角行列を係数とする複数右辺の連立一次方程式 <c>LX = B</c> を解き、解を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="a">The coefficient matrix (lower triangle read). / 係数行列（下三角部分のみ参照）。</param>
    /// <param name="b">The right-hand side matrix. / 右辺行列。</param>
    /// <param name="destination">The destination span to receive the solution. / 解を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> may identically alias <paramref name="b"/> (in-place solve). Overlap with <paramref name="a"/> or partial overlap with <paramref name="b"/> throws.</para>
    /// <para><paramref name="destination"/> は <paramref name="b"/> と完全に同一の領域（インプレース求解）を指定できます。<paramref name="a"/> との重なりや <paramref name="b"/> との部分的な重なりは例外となります。</para>
    /// </remarks>
    [InferenceOverloads]
    public static MatrixSpan<T> SolveLowerTriangular<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.SolveTriangular(a, b, destination, lower: true, vector: false);
        return destination;
    }

    /// <summary>
    /// Solves the upper-triangular linear system <c>Ux = b</c> and returns the solution vector.
    /// <para>上三角行列を係数とする連立一次方程式 <c>Ux = b</c> を解き、解ベクトルを返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="a">The coefficient matrix (only the upper triangle, including the diagonal, is read). / 係数行列（対角を含む上三角部分のみ参照）。</param>
    /// <param name="b">The right-hand side vector. / 右辺ベクトル。</param>
    /// <returns>A new vector containing the solution <c>x</c>. / 解 <c>x</c> を格納した新しいベクトル。</returns>
    /// <remarks>
    /// <para>To solve with the transpose of a Cholesky factor <c>L</c>, pass <c>L.T</c>.</para>
    /// <para>コレスキー因子 <c>L</c> の転置で解く場合は <c>L.T</c> を渡してください。</para>
    /// </remarks>
    [InferenceOverloads]
    public static Vector<T> SolveUpperTriangular<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyVectorSpan<T> b)
        where T : unmanaged, INumberBase<T>
    {
        var result = Vector<T>.Uninitialized(b.Length);
        SolveUpperTriangular(a, b, result);
        return result;
    }

    /// <summary>
    /// Solves the upper-triangular linear system <c>Ux = b</c> and writes the solution to <paramref name="destination"/>.
    /// <para>上三角行列を係数とする連立一次方程式 <c>Ux = b</c> を解き、解を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="a">The coefficient matrix (upper triangle read). / 係数行列（上三角部分のみ参照）。</param>
    /// <param name="b">The right-hand side vector. / 右辺ベクトル。</param>
    /// <param name="destination">The destination span to receive the solution. / 解を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> may identically alias <paramref name="b"/> (in-place solve). Overlap with <paramref name="a"/> or partial overlap with <paramref name="b"/> throws.</para>
    /// <para><paramref name="destination"/> は <paramref name="b"/> と完全に同一の領域（インプレース求解）を指定できます。<paramref name="a"/> との重なりや <paramref name="b"/> との部分的な重なりは例外となります。</para>
    /// </remarks>
    [InferenceOverloads]
    public static VectorSpan<T> SolveUpperTriangular<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyVectorSpan<T> b, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.SolveTriangular(a, b.AsMatrix(), destination.AsMatrix(), lower: false, vector: true);
        return destination;
    }

    /// <summary>
    /// Solves the upper-triangular linear system <c>UX = B</c> with multiple right-hand sides and returns the solution matrix.
    /// <para>上三角行列を係数とする複数右辺の連立一次方程式 <c>UX = B</c> を解き、解行列を返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="a">The coefficient matrix (upper triangle read). / 係数行列（上三角部分のみ参照）。</param>
    /// <param name="b">The right-hand side matrix. / 右辺行列。</param>
    /// <returns>A new matrix containing the solution <c>X</c>. / 解 <c>X</c> を格納した新しい行列。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Matrix<T> SolveUpperTriangular<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b)
        where T : unmanaged, INumberBase<T>
    {
        var result = Matrix<T>.Uninitialized(b.Rows, b.Columns);
        SolveUpperTriangular(a, b, result);
        return result;
    }

    /// <summary>
    /// Solves the upper-triangular linear system <c>UX = B</c> and writes the solution to <paramref name="destination"/>.
    /// <para>上三角行列を係数とする複数右辺の連立一次方程式 <c>UX = B</c> を解き、解を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="a">The coefficient matrix (upper triangle read). / 係数行列（上三角部分のみ参照）。</param>
    /// <param name="b">The right-hand side matrix. / 右辺行列。</param>
    /// <param name="destination">The destination span to receive the solution. / 解を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> may identically alias <paramref name="b"/> (in-place solve). Overlap with <paramref name="a"/> or partial overlap with <paramref name="b"/> throws.</para>
    /// <para><paramref name="destination"/> は <paramref name="b"/> と完全に同一の領域（インプレース求解）を指定できます。<paramref name="a"/> との重なりや <paramref name="b"/> との部分的な重なりは例外となります。</para>
    /// </remarks>
    [InferenceOverloads]
    public static MatrixSpan<T> SolveUpperTriangular<T>(ReadOnlyMatrixSpan<T> a, ReadOnlyMatrixSpan<T> b, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.SolveTriangular(a, b, destination, lower: false, vector: false);
        return destination;
    }

    /// <summary>
    /// Solves the linear system <c>Ax = b</c> using LU decomposition with partial pivoting.
    /// <para>部分ピボット付き LU 分解により連立一次方程式 <c>Ax = b</c> を解き、解ベクトルを返します。</para>
    /// </summary>
    /// <param name="a">The square coefficient matrix. / 正方の係数行列。</param>
    /// <param name="b">The right-hand side vector. / 右辺ベクトル。</param>
    /// <returns>A new vector containing the solution <c>x</c>. / 解 <c>x</c> を格納した新しいベクトル。</returns>
    /// <remarks>
    /// <para>Use <see cref="LuDecomposition"/> to reuse the factorization across multiple solves, or <see cref="CholeskyDecomposition"/> when <paramref name="a"/> is symmetric positive definite.</para>
    /// <para>同じ係数行列で繰り返し解く場合は <see cref="LuDecomposition"/>、対称正定値行列の場合は <see cref="CholeskyDecomposition"/> を使用してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="a"/> is not square, dimensions do not match, or <paramref name="a"/> is singular. / <paramref name="a"/> が正方でない、形状が一致しない、または特異（ピボットが 0）の場合。</exception>
    /// <exception cref="OverflowException">The workspace size <c>n²</c> exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 作業領域の要素数 <c>n²</c> が <see cref="int.MaxValue"/> を超える場合（ブロードキャストビューを渡した場合など）。</exception>
    public static Vector<double> Solve(ReadOnlyMatrixSpan<double> a, ReadOnlyVectorSpan<double> b)
    {
        var result = Vector<double>.Uninitialized(b.Length);
        Solve(a, b, result);
        return result;
    }

    /// <summary>
    /// Solves the linear system <c>Ax = b</c> using LU decomposition with partial pivoting and writes the solution to <paramref name="destination"/>.
    /// <para>部分ピボット付き LU 分解により連立一次方程式 <c>Ax = b</c> を解き、解を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="a">The square coefficient matrix. / 正方の係数行列。</param>
    /// <param name="b">The right-hand side vector. / 右辺ベクトル。</param>
    /// <param name="destination">The destination span to receive the solution. / 解を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> may identically alias <paramref name="b"/>. Overlap with <paramref name="a"/> or partial overlap with <paramref name="b"/> throws.</para>
    /// <para><paramref name="destination"/> は <paramref name="b"/> と完全に同一の領域を指定できます。<paramref name="a"/> との重なりや <paramref name="b"/> との部分的な重なりは例外となります。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="a"/> is not square, dimensions do not match, or <paramref name="a"/> is singular. / <paramref name="a"/> が正方でない、形状が一致しない、または特異の場合。</exception>
    /// <exception cref="OverflowException">The workspace size <c>n²</c> exceeds <see cref="int.MaxValue"/>. / 作業領域の要素数 <c>n²</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static VectorSpan<double> Solve(ReadOnlyMatrixSpan<double> a, ReadOnlyVectorSpan<double> b, VectorSpan<double> destination)
    {
        StridedKernels.Solve(a, b.AsMatrix(), destination.AsMatrix(), vector: true);
        return destination;
    }

    /// <summary>
    /// Solves the linear system <c>AX = B</c> with multiple right-hand sides using LU decomposition with partial pivoting.
    /// <para>部分ピボット付き LU 分解により複数右辺の連立一次方程式 <c>AX = B</c> を解き、解行列を返します。</para>
    /// </summary>
    /// <param name="a">The square coefficient matrix. / 正方の係数行列。</param>
    /// <param name="b">The right-hand side matrix. / 右辺行列。</param>
    /// <returns>A new matrix containing the solution <c>X</c>. / 解 <c>X</c> を格納した新しい行列。</returns>
    /// <exception cref="ArgumentException"><paramref name="a"/> is not square, dimensions do not match, or <paramref name="a"/> is singular. / <paramref name="a"/> が正方でない、形状が一致しない、または特異の場合。</exception>
    /// <exception cref="OverflowException">The workspace size <c>n²</c> or result element count exceeds <see cref="int.MaxValue"/>. / 作業領域の要素数 <c>n²</c> または結果の要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static Matrix<double> Solve(ReadOnlyMatrixSpan<double> a, ReadOnlyMatrixSpan<double> b)
    {
        var result = Matrix<double>.Uninitialized(b.Rows, b.Columns);
        Solve(a, b, result);
        return result;
    }

    /// <summary>
    /// Solves the linear system <c>AX = B</c> using LU decomposition with partial pivoting and writes the solution to <paramref name="destination"/>.
    /// <para>部分ピボット付き LU 分解により複数右辺の連立一次方程式 <c>AX = B</c> を解き、解を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="a">The square coefficient matrix. / 正方の係数行列。</param>
    /// <param name="b">The right-hand side matrix. / 右辺行列。</param>
    /// <param name="destination">The destination span to receive the solution. / 解を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> may identically alias <paramref name="b"/>. Overlap with <paramref name="a"/> or partial overlap with <paramref name="b"/> throws.</para>
    /// <para><paramref name="destination"/> は <paramref name="b"/> と完全に同一の領域を指定できます。<paramref name="a"/> との重なりや <paramref name="b"/> との部分的な重なりは例外となります。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="a"/> is not square, dimensions do not match, or <paramref name="a"/> is singular. / <paramref name="a"/> が正方でない、形状が一致しない、または特異の場合。</exception>
    /// <exception cref="OverflowException">The workspace size <c>n²</c> exceeds <see cref="int.MaxValue"/>. / 作業領域の要素数 <c>n²</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static MatrixSpan<double> Solve(ReadOnlyMatrixSpan<double> a, ReadOnlyMatrixSpan<double> b, MatrixSpan<double> destination)
    {
        StridedKernels.Solve(a, b, destination, vector: false);
        return destination;
    }

    /// <summary>
    /// Computes the matrix inverse <c>A⁻¹</c> using LU decomposition with partial pivoting.
    /// <para>部分ピボット付き LU 分解により逆行列 <c>A⁻¹</c> を計算します。</para>
    /// </summary>
    /// <param name="a">The square matrix to invert. / 逆行列を求める正方行列。</param>
    /// <returns>A new matrix containing <c>A⁻¹</c>. / 逆行列 <c>A⁻¹</c> を格納した新しい行列。</returns>
    /// <remarks>
    /// <para>Prefer <see cref="Solve(ReadOnlyMatrixSpan{double}, ReadOnlyMatrixSpan{double})"/> when solving linear systems, or <see cref="CholeskyDecomposition.Inverse()"/> for symmetric positive definite matrices.</para>
    /// <para>連立一次方程式を解く目的では <see cref="Solve(ReadOnlyMatrixSpan{double}, ReadOnlyMatrixSpan{double})"/> の方が高速かつ高精度です。対称正定値行列には <see cref="CholeskyDecomposition.Inverse()"/> を使用してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="a"/> is not square or is singular. / <paramref name="a"/> が正方でない、または特異の場合。</exception>
    /// <exception cref="OverflowException">The result element count <c>n²</c> exceeds <see cref="int.MaxValue"/>. / 結果の要素数 <c>n²</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static Matrix<double> Inverse(ReadOnlyMatrixSpan<double> a)
    {
        var result = Matrix<double>.Uninitialized(a.Rows, a.Columns);
        Inverse(a, result);
        return result;
    }

    /// <summary>
    /// Computes the matrix inverse <c>A⁻¹</c> using LU decomposition with partial pivoting and writes it to <paramref name="destination"/>.
    /// <para>部分ピボット付き LU 分解により逆行列 <c>A⁻¹</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="a">The square matrix to invert. / 逆行列を求める正方行列。</param>
    /// <param name="destination">The destination span to receive the inverse. / 逆行列を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not share memory with <paramref name="a"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="a"/> と記憶域を共有できません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="a"/> is not square, shapes do not match, <paramref name="destination"/> overlaps <paramref name="a"/>, or <paramref name="a"/> is singular. / <paramref name="a"/> が正方でない、形状が一致しない、領域が重なる、または特異の場合。</exception>
    public static MatrixSpan<double> Inverse(ReadOnlyMatrixSpan<double> a, MatrixSpan<double> destination)
    {
        if (a.Rows != a.Columns)
        {
            throw new ArgumentException("The matrix must be square.", nameof(a));
        }

        if (destination.Rows != a.Rows || destination.Columns != a.Columns)
        {
            throw new ArgumentException("Destination shape must agree.", nameof(destination));
        }

        StridedKernels.Inverse(a, destination);
        return destination;
    }
}
