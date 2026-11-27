using System.Numerics;

namespace Tuna;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

public static partial class XMath
{
    /// <summary>
    /// Computes the sum <c>Σ x[i]</c> of the elements in a vector.
    /// <para>ベクトル要素の総和 <c>Σ x[i]</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>The sum of the elements, or zero if <paramref name="x"/> is empty. / 要素の総和。<paramref name="x"/> が空の場合は 0。</returns>
    /// <remarks>
    /// <para>Propagates <c>NaN</c> values. To exclude <c>NaN</c> elements, use <see cref="Sum{T}(ReadOnlyVectorSpan{T}, bool)"/>.</para>
    /// <para><c>NaN</c> は伝播します。<c>NaN</c> を除外する場合は <see cref="Sum{T}(ReadOnlyVectorSpan{T}, bool)"/> を使用してください。</para>
    /// </remarks>
    [InferenceOverloads]
    public static T Sum<T>(ReadOnlyVectorSpan<T> x)
        where T : unmanaged, INumberBase<T> =>
        StridedKernels.Sum(x);

    /// <summary>
    /// Computes the sum of the main diagonal of a matrix.
    /// <para>行列の主対角成分の和を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>The sum of the first <c>min(x.Rows, x.Columns)</c> diagonal elements, or zero if the matrix is empty. / 主対角上の <c>min(x.Rows, x.Columns)</c> 個の成分の和。空行列の場合は 0。</returns>
    /// <remarks>
    /// <para>Uses the same accumulation and non-finite value handling as <see cref="Sum{T}(ReadOnlyVectorSpan{T})"/>.</para>
    /// <para>加算方法と非有限値の扱いは <see cref="Sum{T}(ReadOnlyVectorSpan{T})"/> と同じです。</para>
    /// </remarks>
    [InferenceOverloads]
    public static T Trace<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumberBase<T> =>
        Sum(x.Diagonal());

    /// <summary>
    /// Computes the sum of each column in a matrix.
    /// <para>行列の各列の総和を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new vector of length <c>x.Columns</c> containing the column sums. / 各列の総和を格納した長さ <c>x.Columns</c> の新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> ColumnSums<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumberBase<T>
    {
        var result = Vector<T>.Uninitialized(x.Columns);
        ColumnSums(x, result);
        return result;
    }

    /// <summary>
    /// Computes the sum of each column in a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各列の総和を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span of length <c>x.Columns</c> to receive the column sums. / 各列の総和を書き込む長さ <c>x.Columns</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Columns</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Columns</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> ColumnSums<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.ColumnSums(x, destination);
        return destination;
    }

    /// <summary>
    /// Computes the sum of each row in a matrix.
    /// <para>行列の各行の総和を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new vector of length <c>x.Rows</c> containing the row sums. / 各行の総和を格納した長さ <c>x.Rows</c> の新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> RowSums<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumberBase<T>
    {
        var result = Vector<T>.Uninitialized(x.Rows);
        RowSums(x, result);
        return result;
    }

    /// <summary>
    /// Computes the sum of each row in a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各行の総和を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span of length <c>x.Rows</c> to receive the row sums. / 各行の総和を書き込む長さ <c>x.Rows</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Rows</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Rows</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> RowSums<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.RowSums(x, destination);
        return destination;
    }

    /// <summary>
    /// Computes the sum of squares <c>Σ x[i]²</c> of the elements in a vector.
    /// <para>ベクトル要素の平方和 <c>Σ x[i]²</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>The sum of squares <c>Σ x[i]²</c>. / 平方和 <c>Σ x[i]²</c>。</returns>
    /// <remarks>
    /// <para>Accumulates in the same order as <see cref="Sum{T}(ReadOnlyVectorSpan{T})"/>. Does not conjugate complex elements, matching <c>Dot(x, x)</c>.</para>
    /// <para><see cref="Sum{T}(ReadOnlyVectorSpan{T})"/> と同じ順序で加算します。複素数でも共役をとらず、<c>Dot(x, x)</c> と同じ値になります。</para>
    /// </remarks>
    [InferenceOverloads]
    public static T SumOfSquares<T>(ReadOnlyVectorSpan<T> x)
        where T : unmanaged, INumberBase<T> =>
        StridedKernels.SumOfSquares(x);

    /// <summary>
    /// Computes the sum of squares of each column in a matrix.
    /// <para>行列の各列の平方和を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new vector of length <c>x.Columns</c> containing the column sums of squares. / 各列の平方和を格納した長さ <c>x.Columns</c> の新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> ColumnSumsOfSquares<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumberBase<T>
    {
        var result = Vector<T>.Uninitialized(x.Columns);
        ColumnSumsOfSquares(x, result);
        return result;
    }

    /// <summary>
    /// Computes the sum of squares of each column in a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各列の平方和を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span of length <c>x.Columns</c> to receive the column sums of squares. / 各列の平方和を書き込む長さ <c>x.Columns</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Columns</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Columns</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> ColumnSumsOfSquares<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.ColumnSumsOfSquares(x, destination);
        return destination;
    }

    /// <summary>
    /// Computes the sum of squares of each row in a matrix.
    /// <para>行列の各行の平方和を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new vector of length <c>x.Rows</c> containing the row sums of squares. / 各行の平方和を格納した長さ <c>x.Rows</c> の新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> RowSumsOfSquares<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumberBase<T>
    {
        var result = Vector<T>.Uninitialized(x.Rows);
        RowSumsOfSquares(x, result);
        return result;
    }

    /// <summary>
    /// Computes the sum of squares of each row in a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各行の平方和を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span of length <c>x.Rows</c> to receive the row sums of squares. / 各行の平方和を書き込む長さ <c>x.Rows</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Rows</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Rows</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> RowSumsOfSquares<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.RowSumsOfSquares(x, destination);
        return destination;
    }

    /// <summary>
    /// Computes the arithmetic mean <c>Σ x[i] / n</c> of the elements in a vector.
    /// <para>ベクトル要素の算術平均 <c>Σ x[i] / n</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>The arithmetic mean, or <c>NaN</c> for an empty floating-point vector. / 算術平均。浮動小数点型で空の場合は <c>NaN</c>。</returns>
    /// <remarks>
    /// <para>Propagates <c>NaN</c> values; to exclude <c>NaN</c>, use <see cref="Mean{T}(ReadOnlyVectorSpan{T}, bool)"/>. For <see cref="double"/> and <see cref="float"/>, finite inputs whose intermediate sum overflows still yield a finite mean when representable. Integer types use the division operator of <typeparamref name="T"/>.</para>
    /// <para><c>NaN</c> は伝播します。<c>NaN</c> を除外する場合は <see cref="Mean{T}(ReadOnlyVectorSpan{T}, bool)"/> を使用してください。<see cref="double"/> および <see cref="float"/> では、途中の総和がオーバーフローしても有限な入力の平均が表現可能な限り有限値を返します。整数型では <typeparamref name="T"/> の除算を使用します。</para>
    /// </remarks>
    [InferenceOverloads]
    public static T Mean<T>(ReadOnlyVectorSpan<T> x)
        where T : unmanaged, INumberBase<T> =>
        StridedKernels.Mean(x);

    /// <summary>
    /// Computes the arithmetic mean of each column in a matrix.
    /// <para>行列の各列の算術平均を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new vector of length <c>x.Columns</c> containing the column means. / 各列の算術平均を格納した長さ <c>x.Columns</c> の新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> ColumnMeans<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumberBase<T>
    {
        var result = Vector<T>.Uninitialized(x.Columns);
        ColumnMeans(x, result);
        return result;
    }

    /// <summary>
    /// Computes the arithmetic mean of each column in a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各列の算術平均を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span of length <c>x.Columns</c> to receive the column means. / 各列の算術平均を書き込む長さ <c>x.Columns</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Columns</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Columns</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> ColumnMeans<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.ColumnMeans(x, destination);
        return destination;
    }

    /// <summary>
    /// Computes the arithmetic mean of each row in a matrix.
    /// <para>行列の各行の算術平均を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new vector of length <c>x.Rows</c> containing the row means. / 各行の算術平均を格納した長さ <c>x.Rows</c> の新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> RowMeans<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumberBase<T>
    {
        var result = Vector<T>.Uninitialized(x.Rows);
        RowMeans(x, result);
        return result;
    }

    /// <summary>
    /// Computes the arithmetic mean of each row in a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各行の算術平均を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span of length <c>x.Rows</c> to receive the row means. / 各行の算術平均を書き込む長さ <c>x.Rows</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Rows</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Rows</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> RowMeans<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.RowMeans(x, destination);
        return destination;
    }

    /// <summary>
    /// Computes the variance <c>Σ(x[i] - mean)² / (n - correction)</c> of the elements in a vector.
    /// <para>ベクトル要素の分散 <c>Σ(x[i] - mean)² / (n - correction)</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="correction">The degrees-of-freedom correction (0 for population variance, 1 for unbiased sample variance). / 自由度の補正（母分散は 0、不偏分散は 1）。</param>
    /// <returns>The variance, or <c>NaN</c> if the element count is less than or equal to <paramref name="correction"/> or any element is <c>NaN</c>. / 分散。要素数が <paramref name="correction"/> 以下の場合や <c>NaN</c> を含む場合は <c>NaN</c>。</returns>
    /// <remarks>
    /// <para>Scales internally to avoid intermediate overflow and loss of accuracy for subnormal results whenever the variance is representable.</para>
    /// <para>分散が表現可能な範囲では、途中のオーバーフローや非正規化数での精度低下を避けるようスケーリングして計算します。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="correction"/> is negative. / <paramref name="correction"/> が負の場合。</exception>
    [InferenceOverloads]
    public static T Variance<T>(ReadOnlyVectorSpan<T> x, int correction)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        StridedKernels.Variance(x, correction);

    /// <summary>
    /// Computes the variance of each column in a matrix.
    /// <para>行列の各列の分散を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="correction">The degrees-of-freedom correction (0 for population variance, 1 for unbiased sample variance). / 自由度の補正（母分散は 0、不偏分散は 1）。</param>
    /// <returns>A new vector of length <c>x.Columns</c> containing the column variances. / 各列の分散を格納した長さ <c>x.Columns</c> の新しいベクトル。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="correction"/> is negative. / <paramref name="correction"/> が負の場合。</exception>
    [InferenceOverloads]
    public static Vector<T> ColumnVariances<T>(ReadOnlyMatrixSpan<T> x, int correction)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = Vector<T>.Uninitialized(x.Columns);
        ColumnVariances(x, correction, result);
        return result;
    }

    /// <summary>
    /// Computes the variance of each column in a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各列の分散を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="correction">The degrees-of-freedom correction (0 for population variance, 1 for unbiased sample variance). / 自由度の補正（母分散は 0、不偏分散は 1）。</param>
    /// <param name="destination">The destination span of length <c>x.Columns</c> to receive the column variances. / 各列の分散を書き込む長さ <c>x.Columns</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="correction"/> is negative. / <paramref name="correction"/> が負の場合。</exception>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Columns</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Columns</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> ColumnVariances<T>(ReadOnlyMatrixSpan<T> x, int correction, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        StridedKernels.ColumnVariances(x, correction, destination);
        return destination;
    }

    /// <summary>
    /// Computes the variance of each row in a matrix.
    /// <para>行列の各行の分散を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="correction">The degrees-of-freedom correction (0 for population variance, 1 for unbiased sample variance). / 自由度の補正（母分散は 0、不偏分散は 1）。</param>
    /// <returns>A new vector of length <c>x.Rows</c> containing the row variances. / 各行の分散を格納した長さ <c>x.Rows</c> の新しいベクトル。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="correction"/> is negative. / <paramref name="correction"/> が負の場合。</exception>
    [InferenceOverloads]
    public static Vector<T> RowVariances<T>(ReadOnlyMatrixSpan<T> x, int correction)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = Vector<T>.Uninitialized(x.Rows);
        RowVariances(x, correction, result);
        return result;
    }

    /// <summary>
    /// Computes the variance of each row in a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各行の分散を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="correction">The degrees-of-freedom correction (0 for population variance, 1 for unbiased sample variance). / 自由度の補正（母分散は 0、不偏分散は 1）。</param>
    /// <param name="destination">The destination span of length <c>x.Rows</c> to receive the row variances. / 各行の分散を書き込む長さ <c>x.Rows</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="correction"/> is negative. / <paramref name="correction"/> が負の場合。</exception>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Rows</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Rows</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> RowVariances<T>(ReadOnlyMatrixSpan<T> x, int correction, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        StridedKernels.RowVariances(x, correction, destination);
        return destination;
    }

    /// <summary>
    /// Computes the standard deviation <c>√Variance(x, correction)</c> of the elements in a vector.
    /// <para>ベクトル要素の標準偏差 <c>√Variance(x, correction)</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="correction">The degrees-of-freedom correction (0 for population, 1 for sample). / 自由度の補正（母標準偏差は 0、標本標準偏差は 1）。</param>
    /// <returns>The standard deviation, or <c>NaN</c> if the element count is less than or equal to <paramref name="correction"/> or any element is <c>NaN</c>. / 標準偏差。要素数が <paramref name="correction"/> 以下の場合や <c>NaN</c> を含む場合は <c>NaN</c>。</returns>
    /// <remarks>
    /// <para>Scales internally so the result remains finite and accurate even when the variance itself would overflow or underflow into subnormals.</para>
    /// <para>分散そのものがオーバーフローまたは非正規化数になる規模の値でも、標準偏差が表現可能であれば有限かつ高精度に計算します。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="correction"/> is negative. / <paramref name="correction"/> が負の場合。</exception>
    [InferenceOverloads]
    public static T StandardDeviation<T>(ReadOnlyVectorSpan<T> x, int correction)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        StridedKernels.StandardDeviation(x, correction);

    /// <summary>
    /// Computes the standard deviation of each column in a matrix.
    /// <para>行列の各列の標準偏差を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="correction">The degrees-of-freedom correction (0 for population, 1 for sample). / 自由度の補正（母標準偏差は 0、標本標準偏差は 1）。</param>
    /// <returns>A new vector of length <c>x.Columns</c> containing the column standard deviations. / 各列の標準偏差を格納した長さ <c>x.Columns</c> の新しいベクトル。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="correction"/> is negative. / <paramref name="correction"/> が負の場合。</exception>
    [InferenceOverloads]
    public static Vector<T> ColumnStandardDeviations<T>(ReadOnlyMatrixSpan<T> x, int correction)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = Vector<T>.Uninitialized(x.Columns);
        ColumnStandardDeviations(x, correction, result);
        return result;
    }

    /// <summary>
    /// Computes the standard deviation of each column in a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各列の標準偏差を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="correction">The degrees-of-freedom correction (0 for population, 1 for sample). / 自由度の補正（母標準偏差は 0、標本標準偏差は 1）。</param>
    /// <param name="destination">The destination span of length <c>x.Columns</c> to receive the column standard deviations. / 各列の標準偏差を書き込む長さ <c>x.Columns</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="correction"/> is negative. / <paramref name="correction"/> が負の場合。</exception>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Columns</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Columns</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> ColumnStandardDeviations<T>(ReadOnlyMatrixSpan<T> x, int correction, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        StridedKernels.ColumnStandardDeviations(x, correction, destination);
        return destination;
    }

    /// <summary>
    /// Computes the standard deviation of each row in a matrix.
    /// <para>行列の各行の標準偏差を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="correction">The degrees-of-freedom correction (0 for population, 1 for sample). / 自由度の補正（母標準偏差は 0、標本標準偏差は 1）。</param>
    /// <returns>A new vector of length <c>x.Rows</c> containing the row standard deviations. / 各行の標準偏差を格納した長さ <c>x.Rows</c> の新しいベクトル。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="correction"/> is negative. / <paramref name="correction"/> が負の場合。</exception>
    [InferenceOverloads]
    public static Vector<T> RowStandardDeviations<T>(ReadOnlyMatrixSpan<T> x, int correction)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = Vector<T>.Uninitialized(x.Rows);
        RowStandardDeviations(x, correction, result);
        return result;
    }

    /// <summary>
    /// Computes the standard deviation of each row in a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各行の標準偏差を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="correction">The degrees-of-freedom correction (0 for population, 1 for sample). / 自由度の補正（母標準偏差は 0、標本標準偏差は 1）。</param>
    /// <param name="destination">The destination span of length <c>x.Rows</c> to receive the row standard deviations. / 各行の標準偏差を書き込む長さ <c>x.Rows</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="correction"/> is negative. / <paramref name="correction"/> が負の場合。</exception>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Rows</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Rows</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> RowStandardDeviations<T>(ReadOnlyMatrixSpan<T> x, int correction, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        StridedKernels.RowStandardDeviations(x, correction, destination);
        return destination;
    }

    /// <summary>
    /// Computes the median of the elements in a vector.
    /// <para>ベクトル要素の中央値を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>The median, or <c>NaN</c> if <paramref name="x"/> is empty or contains <c>NaN</c>. / 中央値。<paramref name="x"/> が空または <c>NaN</c> を含む場合は <c>NaN</c>。</returns>
    /// <remarks>
    /// <para>Orders elements by total order (<c>-0 &lt; +0</c>) without modifying <paramref name="x"/>. For an even number of elements, returns the overflow-safe midpoint of the two middle values (<c>-∞</c> and <c>+∞</c> yield <c>NaN</c>). To exclude <c>NaN</c> elements, use <see cref="Median{T}(ReadOnlyVectorSpan{T}, bool)"/>.</para>
    /// <para><paramref name="x"/> を変更せず全順序（<c>-0 &lt; +0</c>）で中央値を求めます。要素数が偶数の場合は中央の2値の中点をオーバーフローを回避して計算します（<c>-∞</c> と <c>+∞</c> の場合は <c>NaN</c>）。<c>NaN</c> を除外する場合は <see cref="Median{T}(ReadOnlyVectorSpan{T}, bool)"/> を使用してください。</para>
    /// </remarks>
    [InferenceOverloads]
    public static T Median<T>(ReadOnlyVectorSpan<T> x)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        StridedKernels.Median(x, ignoreNaN: false);

    /// <summary>
    /// Computes the median of the elements in a vector, optionally excluding <c>NaN</c> values.
    /// <para>必要に応じて <c>NaN</c> を除外して、ベクトル要素の中央値を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <returns>The median, or <c>NaN</c> if no valid elements remain (or if <paramref name="ignoreNaN"/> is <see langword="false"/> and any element is <c>NaN</c>). / 中央値。有効な要素がない場合、または <paramref name="ignoreNaN"/> が <see langword="false"/> で <c>NaN</c> を含む場合は <c>NaN</c>。</returns>
    /// <remarks>
    /// <para>When <paramref name="ignoreNaN"/> is <see langword="false"/>, behaves identically to <see cref="Median{T}(ReadOnlyVectorSpan{T})"/>.</para>
    /// <para><paramref name="ignoreNaN"/> が <see langword="false"/> の場合は <see cref="Median{T}(ReadOnlyVectorSpan{T})"/> と同じ動作になります。</para>
    /// </remarks>
    [InferenceOverloads]
    public static T Median<T>(ReadOnlyVectorSpan<T> x, bool ignoreNaN)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        StridedKernels.Median(x, ignoreNaN);

    /// <summary>
    /// Computes the median of each column in a matrix.
    /// <para>行列の各列の中央値を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new vector of length <c>x.Columns</c> containing the column medians. / 各列の中央値を格納した長さ <c>x.Columns</c> の新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> ColumnMedians<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        ColumnMedians(x, ignoreNaN: false);

    /// <summary>
    /// Computes the median of each column in a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各列の中央値を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span of length <c>x.Columns</c> to receive the column medians. / 各列の中央値を書き込む長さ <c>x.Columns</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Columns</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Columns</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> ColumnMedians<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        ColumnMedians(x, ignoreNaN: false, destination);

    /// <summary>
    /// Computes the median of each column in a matrix, optionally excluding <c>NaN</c> values.
    /// <para>必要に応じて <c>NaN</c> を除外して、行列の各列の中央値を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <returns>A new vector of length <c>x.Columns</c> containing the column medians. / 各列の中央値を格納した長さ <c>x.Columns</c> の新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> ColumnMedians<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = Vector<T>.Uninitialized(x.Columns);
        ColumnMedians(x, ignoreNaN, result);
        return result;
    }

    /// <summary>
    /// Computes the median of each column in a matrix, optionally excluding <c>NaN</c> values, and writes the result to <paramref name="destination"/>.
    /// <para>必要に応じて <c>NaN</c> を除外して行列の各列の中央値を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <param name="destination">The destination span of length <c>x.Columns</c> to receive the column medians. / 各列の中央値を書き込む長さ <c>x.Columns</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Columns</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Columns</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> ColumnMedians<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        StridedKernels.ColumnMedians(x, ignoreNaN, destination);
        return destination;
    }

    /// <summary>
    /// Computes the median of each row in a matrix.
    /// <para>行列の各行の中央値を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new vector of length <c>x.Rows</c> containing the row medians. / 各行の中央値を格納した長さ <c>x.Rows</c> の新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> RowMedians<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        RowMedians(x, ignoreNaN: false);

    /// <summary>
    /// Computes the median of each row in a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各行の中央値を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span of length <c>x.Rows</c> to receive the row medians. / 各行の中央値を書き込む長さ <c>x.Rows</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Rows</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Rows</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> RowMedians<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        RowMedians(x, ignoreNaN: false, destination);

    /// <summary>
    /// Computes the median of each row in a matrix, optionally excluding <c>NaN</c> values.
    /// <para>必要に応じて <c>NaN</c> を除外して、行列の各行の中央値を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <returns>A new vector of length <c>x.Rows</c> containing the row medians. / 各行の中央値を格納した長さ <c>x.Rows</c> の新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> RowMedians<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = Vector<T>.Uninitialized(x.Rows);
        RowMedians(x, ignoreNaN, result);
        return result;
    }

    /// <summary>
    /// Computes the median of each row in a matrix, optionally excluding <c>NaN</c> values, and writes the result to <paramref name="destination"/>.
    /// <para>必要に応じて <c>NaN</c> を除外して行列の各行の中央値を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <param name="destination">The destination span of length <c>x.Rows</c> to receive the row medians. / 各行の中央値を書き込む長さ <c>x.Rows</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Rows</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Rows</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> RowMedians<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        StridedKernels.RowMedians(x, ignoreNaN, destination);
        return destination;
    }

    /// <summary>
    /// Computes the quantile <c>Q(p)</c> of the elements in a vector.
    /// <para>ベクトル要素の分位点 <c>Q(p)</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="probability">The probability <c>p</c> in the interval <c>[0, 1]</c>. / 確率 <c>p</c>（<c>0 ≤ p ≤ 1</c>）。</param>
    /// <returns>The quantile <c>Q(p)</c>, or <c>NaN</c> if <paramref name="x"/> is empty or contains <c>NaN</c>. / 分位点 <c>Q(p)</c>。<paramref name="x"/> が空または <c>NaN</c> を含む場合は <c>NaN</c>。</returns>
    /// <remarks>
    /// <para>Uses Hyndman &amp; Fan (1996) Definition 7: for sorted values <c>x₀ ≤ … ≤ xₙ₋₁</c> and <c>h = (n - 1)p</c>, <c>j = ⌊h⌋</c>, <c>g = h - j</c>, linearly interpolates <c>xⱼ + g(xⱼ₊₁ - xⱼ)</c> without modifying <paramref name="x"/>. <c>Q(0)</c> is the minimum, <c>Q(1)</c> is the maximum, and <c>Q(0.5)</c> matches <see cref="Median{T}(ReadOnlyVectorSpan{T})"/>. To exclude <c>NaN</c> elements, use <see cref="Quantile{T}(ReadOnlyVectorSpan{T}, double, bool)"/>.</para>
    /// <para>Hyndman &amp; Fan (1996) の定義 7（昇順の値 <c>x₀ ≤ … ≤ xₙ₋₁</c> と <c>h = (n - 1)p</c>、<c>j = ⌊h⌋</c>、<c>g = h - j</c> に対する線形補間 <c>xⱼ + g(xⱼ₊₁ - xⱼ)</c>）を使用し、<paramref name="x"/> を変更せずに計算します。<c>Q(0)</c> は最小値、<c>Q(1)</c> は最大値、<c>Q(0.5)</c> は <see cref="Median{T}(ReadOnlyVectorSpan{T})"/> と一致します。<c>NaN</c> を除外する場合は <see cref="Quantile{T}(ReadOnlyVectorSpan{T}, double, bool)"/> を使用してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="probability"/> is outside <c>[0, 1]</c> or is <see cref="double.NaN"/>. / <paramref name="probability"/> が <c>[0, 1]</c> の範囲外、または <see cref="double.NaN"/> の場合。</exception>
    [InferenceOverloads]
    public static T Quantile<T>(ReadOnlyVectorSpan<T> x, double probability)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        StridedKernels.Quantile(x, probability, ignoreNaN: false);

    /// <summary>
    /// Computes the quantile of the elements in a vector, optionally excluding <c>NaN</c> values.
    /// <para>必要に応じて <c>NaN</c> を除外して、ベクトル要素の分位点を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="probability">The probability in the interval <c>[0, 1]</c>. / 確率（0 以上 1 以下）。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <returns>The quantile, or <c>NaN</c> if no valid elements remain (or if <paramref name="ignoreNaN"/> is <see langword="false"/> and any element is <c>NaN</c>). / 分位点。有効な要素がない場合、または <paramref name="ignoreNaN"/> が <see langword="false"/> で <c>NaN</c> を含む場合は <c>NaN</c>。</returns>
    /// <remarks>
    /// <para>Uses Hyndman &amp; Fan (1996) Definition 7. When <paramref name="ignoreNaN"/> is <see langword="false"/>, behaves identically to <see cref="Quantile{T}(ReadOnlyVectorSpan{T}, double)"/>.</para>
    /// <para>Hyndman &amp; Fan (1996) の定義 7 を使用します。<paramref name="ignoreNaN"/> が <see langword="false"/> の場合は <see cref="Quantile{T}(ReadOnlyVectorSpan{T}, double)"/> と同じ動作になります。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="probability"/> is outside <c>[0, 1]</c> or is <see cref="double.NaN"/>. / <paramref name="probability"/> が <c>[0, 1]</c> の範囲外、または <see cref="double.NaN"/> の場合。</exception>
    [InferenceOverloads]
    public static T Quantile<T>(ReadOnlyVectorSpan<T> x, double probability, bool ignoreNaN)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        StridedKernels.Quantile(x, probability, ignoreNaN);

    /// <summary>
    /// Computes quantiles of the elements in a vector at multiple probabilities.
    /// <para>ベクトル要素に対して複数の確率における分位点を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="probabilities">The probabilities in <c>[0, 1]</c> at which to evaluate quantiles. / 分位点を求める確率の列（各値は 0 以上 1 以下）。</param>
    /// <returns>A new vector of length <c>probabilities.Length</c> containing the quantiles corresponding to <paramref name="probabilities"/>. / 各確率の分位点を格納した長さ <c>probabilities.Length</c> の新しいベクトル。</returns>
    /// <remarks>
    /// <para>Each element matches <see cref="Quantile{T}(ReadOnlyVectorSpan{T}, double)"/> (Definition 7), sorting the data only once. For example, quartiles use <c>[0.25, 0.5, 0.75]</c>.</para>
    /// <para>各要素は <see cref="Quantile{T}(ReadOnlyVectorSpan{T}, double)"/>（定義 7）と一致し、データのソートは 1 回のみ行います。たとえば四分位点は <c>[0.25, 0.5, 0.75]</c> を指定します。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Any element of <paramref name="probabilities"/> is outside <c>[0, 1]</c> or is <see cref="double.NaN"/>. / <paramref name="probabilities"/> に <c>[0, 1]</c> の範囲外、または <see cref="double.NaN"/> の値が含まれる場合。</exception>
    [InferenceOverloads]
    public static Vector<T> Quantiles<T>(ReadOnlyVectorSpan<T> x, ReadOnlySpan<double> probabilities)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        Quantiles(x, probabilities, ignoreNaN: false);

    /// <summary>
    /// Computes quantiles of the elements in a vector at multiple probabilities and writes the result to <paramref name="destination"/>.
    /// <para>ベクトル要素に対して複数の確率における分位点を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="probabilities">The probabilities in <c>[0, 1]</c> at which to evaluate quantiles. / 分位点を求める確率の列（各値は 0 以上 1 以下）。</param>
    /// <param name="destination">The destination span of length <c>probabilities.Length</c> to receive the quantiles. / 分位点を書き込む長さ <c>probabilities.Length</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/> or <paramref name="probabilities"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> および <paramref name="probabilities"/> と記憶域を共有できません。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Any element of <paramref name="probabilities"/> is outside <c>[0, 1]</c> or is <see cref="double.NaN"/>. / <paramref name="probabilities"/> に <c>[0, 1]</c> の範囲外、または <see cref="double.NaN"/> の値が含まれる場合。</exception>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>probabilities.Length</c>, or <paramref name="destination"/> overlaps <paramref name="x"/> or <paramref name="probabilities"/>. / <paramref name="destination"/> の長さが <c>probabilities.Length</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> もしくは <paramref name="probabilities"/> と記憶域を共有する場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> Quantiles<T>(ReadOnlyVectorSpan<T> x, ReadOnlySpan<double> probabilities, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        Quantiles(x, probabilities, ignoreNaN: false, destination);

    /// <summary>
    /// Computes quantiles of the elements in a vector at multiple probabilities, optionally excluding <c>NaN</c> values.
    /// <para>必要に応じて <c>NaN</c> を除外して、ベクトル要素に対する複数の確率の分位点を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="probabilities">The probabilities in <c>[0, 1]</c> at which to evaluate quantiles. / 分位点を求める確率の列（各値は 0 以上 1 以下）。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <returns>A new vector of length <c>probabilities.Length</c> containing the quantiles corresponding to <paramref name="probabilities"/>. / 各確率の分位点を格納した長さ <c>probabilities.Length</c> の新しいベクトル。</returns>
    /// <exception cref="ArgumentOutOfRangeException">Any element of <paramref name="probabilities"/> is outside <c>[0, 1]</c> or is <see cref="double.NaN"/>. / <paramref name="probabilities"/> に <c>[0, 1]</c> の範囲外、または <see cref="double.NaN"/> の値が含まれる場合。</exception>
    [InferenceOverloads]
    public static Vector<T> Quantiles<T>(ReadOnlyVectorSpan<T> x, ReadOnlySpan<double> probabilities, bool ignoreNaN)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = Vector<T>.Uninitialized(probabilities.Length);
        Quantiles(x, probabilities, ignoreNaN, result);
        return result;
    }

    /// <summary>
    /// Computes quantiles of the elements in a vector at multiple probabilities, optionally excluding <c>NaN</c> values, and writes the result to <paramref name="destination"/>.
    /// <para>必要に応じて <c>NaN</c> を除外して複数の確率の分位点を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="probabilities">The probabilities in <c>[0, 1]</c> at which to evaluate quantiles. / 分位点を求める確率の列（各値は 0 以上 1 以下）。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <param name="destination">The destination span of length <c>probabilities.Length</c> to receive the quantiles. / 分位点を書き込む長さ <c>probabilities.Length</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/> or <paramref name="probabilities"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> および <paramref name="probabilities"/> と記憶域を共有できません。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Any element of <paramref name="probabilities"/> is outside <c>[0, 1]</c> or is <see cref="double.NaN"/>. / <paramref name="probabilities"/> に <c>[0, 1]</c> の範囲外、または <see cref="double.NaN"/> の値が含まれる場合。</exception>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>probabilities.Length</c>, or <paramref name="destination"/> overlaps <paramref name="x"/> or <paramref name="probabilities"/>. / <paramref name="destination"/> の長さが <c>probabilities.Length</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> もしくは <paramref name="probabilities"/> と記憶域を共有する場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> Quantiles<T>(ReadOnlyVectorSpan<T> x, ReadOnlySpan<double> probabilities, bool ignoreNaN, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        StridedKernels.Quantiles(x, probabilities, ignoreNaN, destination);
        return destination;
    }

    /// <summary>
    /// Computes the quantile of each column in a matrix.
    /// <para>行列の各列の分位点を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="probability">The probability in the interval <c>[0, 1]</c>. / 確率（0 以上 1 以下）。</param>
    /// <returns>A new vector of length <c>x.Columns</c> containing the column quantiles. / 各列の分位点を格納した長さ <c>x.Columns</c> の新しいベクトル。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="probability"/> is outside <c>[0, 1]</c> or is <see cref="double.NaN"/>. / <paramref name="probability"/> が <c>[0, 1]</c> の範囲外、または <see cref="double.NaN"/> の場合。</exception>
    [InferenceOverloads]
    public static Vector<T> ColumnQuantiles<T>(ReadOnlyMatrixSpan<T> x, double probability)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        ColumnQuantiles(x, probability, ignoreNaN: false);

    /// <summary>
    /// Computes the quantile of each column in a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各列の分位点を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="probability">The probability in the interval <c>[0, 1]</c>. / 確率（0 以上 1 以下）。</param>
    /// <param name="destination">The destination span of length <c>x.Columns</c> to receive the column quantiles. / 各列の分位点を書き込む長さ <c>x.Columns</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="probability"/> is outside <c>[0, 1]</c> or is <see cref="double.NaN"/>. / <paramref name="probability"/> が <c>[0, 1]</c> の範囲外、または <see cref="double.NaN"/> の場合。</exception>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Columns</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Columns</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> ColumnQuantiles<T>(ReadOnlyMatrixSpan<T> x, double probability, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        ColumnQuantiles(x, probability, ignoreNaN: false, destination);

    /// <summary>
    /// Computes the quantile of each column in a matrix, optionally excluding <c>NaN</c> values.
    /// <para>必要に応じて <c>NaN</c> を除外して、行列の各列の分位点を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="probability">The probability in the interval <c>[0, 1]</c>. / 確率（0 以上 1 以下）。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <returns>A new vector of length <c>x.Columns</c> containing the column quantiles. / 各列の分位点を格納した長さ <c>x.Columns</c> の新しいベクトル。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="probability"/> is outside <c>[0, 1]</c> or is <see cref="double.NaN"/>. / <paramref name="probability"/> が <c>[0, 1]</c> の範囲外、または <see cref="double.NaN"/> の場合。</exception>
    [InferenceOverloads]
    public static Vector<T> ColumnQuantiles<T>(ReadOnlyMatrixSpan<T> x, double probability, bool ignoreNaN)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = Vector<T>.Uninitialized(x.Columns);
        ColumnQuantiles(x, probability, ignoreNaN, result);
        return result;
    }

    /// <summary>
    /// Computes the quantile of each column in a matrix, optionally excluding <c>NaN</c> values, and writes the result to <paramref name="destination"/>.
    /// <para>必要に応じて <c>NaN</c> を除外して行列の各列の分位点を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="probability">The probability in the interval <c>[0, 1]</c>. / 確率（0 以上 1 以下）。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <param name="destination">The destination span of length <c>x.Columns</c> to receive the column quantiles. / 各列の分位点を書き込む長さ <c>x.Columns</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="probability"/> is outside <c>[0, 1]</c> or is <see cref="double.NaN"/>. / <paramref name="probability"/> が <c>[0, 1]</c> の範囲外、または <see cref="double.NaN"/> の場合。</exception>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Columns</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Columns</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> ColumnQuantiles<T>(ReadOnlyMatrixSpan<T> x, double probability, bool ignoreNaN, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        StridedKernels.ColumnQuantiles(x, probability, ignoreNaN, destination);
        return destination;
    }

    /// <summary>
    /// Computes the quantile of each row in a matrix.
    /// <para>行列の各行の分位点を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="probability">The probability in the interval <c>[0, 1]</c>. / 確率（0 以上 1 以下）。</param>
    /// <returns>A new vector of length <c>x.Rows</c> containing the row quantiles. / 各行の分位点を格納した長さ <c>x.Rows</c> の新しいベクトル。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="probability"/> is outside <c>[0, 1]</c> or is <see cref="double.NaN"/>. / <paramref name="probability"/> が <c>[0, 1]</c> の範囲外、または <see cref="double.NaN"/> の場合。</exception>
    [InferenceOverloads]
    public static Vector<T> RowQuantiles<T>(ReadOnlyMatrixSpan<T> x, double probability)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        RowQuantiles(x, probability, ignoreNaN: false);

    /// <summary>
    /// Computes the quantile of each row in a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各行の分位点を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="probability">The probability in the interval <c>[0, 1]</c>. / 確率（0 以上 1 以下）。</param>
    /// <param name="destination">The destination span of length <c>x.Rows</c> to receive the row quantiles. / 各行の分位点を書き込む長さ <c>x.Rows</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="probability"/> is outside <c>[0, 1]</c> or is <see cref="double.NaN"/>. / <paramref name="probability"/> が <c>[0, 1]</c> の範囲外、または <see cref="double.NaN"/> の場合。</exception>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Rows</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Rows</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> RowQuantiles<T>(ReadOnlyMatrixSpan<T> x, double probability, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        RowQuantiles(x, probability, ignoreNaN: false, destination);

    /// <summary>
    /// Computes the quantile of each row in a matrix, optionally excluding <c>NaN</c> values.
    /// <para>必要に応じて <c>NaN</c> を除外して、行列の各行の分位点を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="probability">The probability in the interval <c>[0, 1]</c>. / 確率（0 以上 1 以下）。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <returns>A new vector of length <c>x.Rows</c> containing the row quantiles. / 各行の分位点を格納した長さ <c>x.Rows</c> の新しいベクトル。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="probability"/> is outside <c>[0, 1]</c> or is <see cref="double.NaN"/>. / <paramref name="probability"/> が <c>[0, 1]</c> の範囲外、または <see cref="double.NaN"/> の場合。</exception>
    [InferenceOverloads]
    public static Vector<T> RowQuantiles<T>(ReadOnlyMatrixSpan<T> x, double probability, bool ignoreNaN)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = Vector<T>.Uninitialized(x.Rows);
        RowQuantiles(x, probability, ignoreNaN, result);
        return result;
    }

    /// <summary>
    /// Computes the quantile of each row in a matrix, optionally excluding <c>NaN</c> values, and writes the result to <paramref name="destination"/>.
    /// <para>必要に応じて <c>NaN</c> を除外して行列の各行の分位点を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="probability">The probability in the interval <c>[0, 1]</c>. / 確率（0 以上 1 以下）。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <param name="destination">The destination span of length <c>x.Rows</c> to receive the row quantiles. / 各行の分位点を書き込む長さ <c>x.Rows</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="probability"/> is outside <c>[0, 1]</c> or is <see cref="double.NaN"/>. / <paramref name="probability"/> が <c>[0, 1]</c> の範囲外、または <see cref="double.NaN"/> の場合。</exception>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Rows</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Rows</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> RowQuantiles<T>(ReadOnlyMatrixSpan<T> x, double probability, bool ignoreNaN, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        StridedKernels.RowQuantiles(x, probability, ignoreNaN, destination);
        return destination;
    }

    /// <summary>
    /// Returns the maximum element in a vector.
    /// <para>ベクトル要素の最大値を返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>The maximum element, or <c>NaN</c> if any element is <c>NaN</c>. / 最大値。<c>NaN</c> を含む場合は <c>NaN</c>。</returns>
    /// <remarks>
    /// <para>Propagates <c>NaN</c> values. To exclude <c>NaN</c> elements, use <see cref="Max{T}(ReadOnlyVectorSpan{T}, bool)"/>.</para>
    /// <para><c>NaN</c> は伝播します。<c>NaN</c> を除外する場合は <see cref="Max{T}(ReadOnlyVectorSpan{T}, bool)"/> を使用してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="x"/> is empty. / <paramref name="x"/> が空の場合。</exception>
    [InferenceOverloads]
    public static T Max<T>(ReadOnlyVectorSpan<T> x)
        where T : unmanaged, INumber<T> =>
        StridedKernels.Max(x);

    /// <summary>
    /// Returns the minimum element in a vector.
    /// <para>ベクトル要素の最小値を返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>The minimum element, or <c>NaN</c> if any element is <c>NaN</c>. / 最小値。<c>NaN</c> を含む場合は <c>NaN</c>。</returns>
    /// <remarks>
    /// <para>Propagates <c>NaN</c> values. To exclude <c>NaN</c> elements, use <see cref="Min{T}(ReadOnlyVectorSpan{T}, bool)"/>.</para>
    /// <para><c>NaN</c> は伝播します。<c>NaN</c> を除外する場合は <see cref="Min{T}(ReadOnlyVectorSpan{T}, bool)"/> を使用してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="x"/> is empty. / <paramref name="x"/> が空の場合。</exception>
    [InferenceOverloads]
    public static T Min<T>(ReadOnlyVectorSpan<T> x)
        where T : unmanaged, INumber<T> =>
        StridedKernels.Min(x);

    /// <summary>
    /// Computes the maximum element of each column in a matrix.
    /// <para>行列の各列の最大値を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new vector of length <c>x.Columns</c> containing the column maxima. / 各列の最大値を格納した長さ <c>x.Columns</c> の新しいベクトル。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> has columns but zero rows. / <paramref name="x"/> の列数が 1 以上で行数が 0 の場合。</exception>
    [InferenceOverloads]
    public static Vector<T> ColumnMax<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumber<T>
    {
        var result = Vector<T>.Uninitialized(x.Columns);
        ColumnMax(x, result);
        return result;
    }

    /// <summary>
    /// Computes the maximum element of each column in a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各列の最大値を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span of length <c>x.Columns</c> to receive the column maxima. / 各列の最大値を書き込む長さ <c>x.Columns</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="x"/> has columns but zero rows, the length of <paramref name="destination"/> does not equal <c>x.Columns</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="x"/> の列数が 1 以上で行数が 0 の場合、<paramref name="destination"/> の長さが <c>x.Columns</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> ColumnMax<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, INumber<T>
    {
        StridedKernels.ColumnMax(x, destination);
        return destination;
    }

    /// <summary>
    /// Computes the maximum element of each row in a matrix.
    /// <para>行列の各行の最大値を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new vector of length <c>x.Rows</c> containing the row maxima. / 各行の最大値を格納した長さ <c>x.Rows</c> の新しいベクトル。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> has rows but zero columns. / <paramref name="x"/> の行数が 1 以上で列数が 0 の場合。</exception>
    [InferenceOverloads]
    public static Vector<T> RowMax<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumber<T>
    {
        var result = Vector<T>.Uninitialized(x.Rows);
        RowMax(x, result);
        return result;
    }

    /// <summary>
    /// Computes the maximum element of each row in a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各行の最大値を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span of length <c>x.Rows</c> to receive the row maxima. / 各行の最大値を書き込む長さ <c>x.Rows</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="x"/> has rows but zero columns, the length of <paramref name="destination"/> does not equal <c>x.Rows</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="x"/> の行数が 1 以上で列数が 0 の場合、<paramref name="destination"/> の長さが <c>x.Rows</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> RowMax<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, INumber<T>
    {
        StridedKernels.RowMax(x, destination);
        return destination;
    }

    /// <summary>
    /// Computes the minimum element of each column in a matrix.
    /// <para>行列の各列の最小値を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new vector of length <c>x.Columns</c> containing the column minima. / 各列の最小値を格納した長さ <c>x.Columns</c> の新しいベクトル。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> has columns but zero rows. / <paramref name="x"/> の列数が 1 以上で行数が 0 の場合。</exception>
    [InferenceOverloads]
    public static Vector<T> ColumnMin<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumber<T>
    {
        var result = Vector<T>.Uninitialized(x.Columns);
        ColumnMin(x, result);
        return result;
    }

    /// <summary>
    /// Computes the minimum element of each column in a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各列の最小値を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span of length <c>x.Columns</c> to receive the column minima. / 各列の最小値を書き込む長さ <c>x.Columns</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="x"/> has columns but zero rows, the length of <paramref name="destination"/> does not equal <c>x.Columns</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="x"/> の列数が 1 以上で行数が 0 の場合、<paramref name="destination"/> の長さが <c>x.Columns</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> ColumnMin<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, INumber<T>
    {
        StridedKernels.ColumnMin(x, destination);
        return destination;
    }

    /// <summary>
    /// Computes the minimum element of each row in a matrix.
    /// <para>行列の各行の最小値を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new vector of length <c>x.Rows</c> containing the row minima. / 各行の最小値を格納した長さ <c>x.Rows</c> の新しいベクトル。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> has rows but zero columns. / <paramref name="x"/> の行数が 1 以上で列数が 0 の場合。</exception>
    [InferenceOverloads]
    public static Vector<T> RowMin<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumber<T>
    {
        var result = Vector<T>.Uninitialized(x.Rows);
        RowMin(x, result);
        return result;
    }

    /// <summary>
    /// Computes the minimum element of each row in a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各行の最小値を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span of length <c>x.Rows</c> to receive the row minima. / 各行の最小値を書き込む長さ <c>x.Rows</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="x"/> has rows but zero columns, the length of <paramref name="destination"/> does not equal <c>x.Rows</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="x"/> の行数が 1 以上で列数が 0 の場合、<paramref name="destination"/> の長さが <c>x.Rows</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> RowMin<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, INumber<T>
    {
        StridedKernels.RowMin(x, destination);
        return destination;
    }

    /// <summary>
    /// Returns the zero-based index of the first element with the maximum value in a vector.
    /// <para>ベクトル内で最大値を持つ最初の要素の 0 始まりのインデックスを返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The vector to search. / 検索対象のベクトル。</param>
    /// <returns>The zero-based index of the first maximum element, or of the first <c>NaN</c> if any element is <c>NaN</c>. / 最大値を持つ最初の要素の 0 始まりのインデックス。<c>NaN</c> を含む場合は最初の <c>NaN</c> のインデックス。</returns>
    /// <remarks>
    /// <para>Treats <c>+0</c> as greater than <c>-0</c>.</para>
    /// <para><c>+0</c> は <c>-0</c> より大きい値として扱います。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="x"/> is empty. / <paramref name="x"/> が空の場合。</exception>
    [InferenceOverloads]
    public static int IndexOfMax<T>(ReadOnlyVectorSpan<T> x)
        where T : unmanaged, INumber<T> =>
        StridedKernels.IndexOfMax(x);

    /// <summary>
    /// Returns the zero-based row and column indices of the first element with the maximum value in a matrix.
    /// <para>行列全体で最大値を持つ最初の要素の 0 始まりの行番号と列番号を返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The matrix to search. / 検索対象の行列。</param>
    /// <returns>The zero-based <c>(Row, Column)</c> position of the first maximum element, or of the first <c>NaN</c> if any element is <c>NaN</c>. / 最初に見つかった最大値（<c>NaN</c> を含む場合は最初の <c>NaN</c>）の 0 始まりの <c>(Row, Column)</c>。</returns>
    /// <remarks>
    /// <para>Searches in column-major order (columns first, then rows within each column) and treats <c>+0</c> as greater than <c>-0</c>.</para>
    /// <para>列を先頭から順に、各列内は行番号順に検索します。<c>+0</c> は <c>-0</c> より大きい値として扱います。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="x"/> is empty. / <paramref name="x"/> が空の場合。</exception>
    [InferenceOverloads]
    public static (int Row, int Column) IndexOfMax<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumber<T> =>
        StridedKernels.IndexOfMax(x);

    /// <summary>
    /// Returns the zero-based row index of the first maximum element in each column of a matrix.
    /// <para>行列の各列について、最大値を持つ最初の要素の行インデックスを返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new array of length <c>x.Columns</c> containing the row index of the maximum in each column. / 各列の最大値の行インデックスを格納した長さ <c>x.Columns</c> の新しい配列。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> has columns but zero rows. / <paramref name="x"/> の列数が 1 以上で行数が 0 の場合。</exception>
    [InferenceOverloads]
    public static int[] ColumnIndexOfMax<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumber<T>
    {
        var result = new int[x.Columns];
        ColumnIndexOfMax(x, result);
        return result;
    }

    /// <summary>
    /// Computes the zero-based row index of the first maximum element in each column of a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各列で最大値を持つ最初の要素の行インデックスを計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span of length <c>x.Columns</c> to receive the indices. / インデックスを書き込む長さ <c>x.Columns</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> has columns but zero rows, or the length of <paramref name="destination"/> does not equal <c>x.Columns</c>. / <paramref name="x"/> の列数が 1 以上で行数が 0 の場合、または <paramref name="destination"/> の長さが <c>x.Columns</c> と一致しない場合。</exception>
    [InferenceOverloads]
    public static Span<int> ColumnIndexOfMax<T>(ReadOnlyMatrixSpan<T> x, Span<int> destination)
        where T : unmanaged, INumber<T>
    {
        StridedKernels.ColumnIndexOfMax(x, destination);
        return destination;
    }

    /// <summary>
    /// Returns the zero-based column index of the first maximum element in each row of a matrix.
    /// <para>行列の各行について、最大値を持つ最初の要素の列インデックスを返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new array of length <c>x.Rows</c> containing the column index of the maximum in each row. / 各行の最大値の列インデックスを格納した長さ <c>x.Rows</c> の新しい配列。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> has rows but zero columns. / <paramref name="x"/> の行数が 1 以上で列数が 0 の場合。</exception>
    [InferenceOverloads]
    public static int[] RowIndexOfMax<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumber<T>
    {
        var result = new int[x.Rows];
        RowIndexOfMax(x, result);
        return result;
    }

    /// <summary>
    /// Computes the zero-based column index of the first maximum element in each row of a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各行で最大値を持つ最初の要素の列インデックスを計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span of length <c>x.Rows</c> to receive the indices. / インデックスを書き込む長さ <c>x.Rows</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> has rows but zero columns, or the length of <paramref name="destination"/> does not equal <c>x.Rows</c>. / <paramref name="x"/> の行数が 1 以上で列数が 0 の場合、または <paramref name="destination"/> の長さが <c>x.Rows</c> と一致しない場合。</exception>
    [InferenceOverloads]
    public static Span<int> RowIndexOfMax<T>(ReadOnlyMatrixSpan<T> x, Span<int> destination)
        where T : unmanaged, INumber<T>
    {
        StridedKernels.RowIndexOfMax(x, destination);
        return destination;
    }

    /// <summary>
    /// Returns the zero-based index of the first element with the minimum value in a vector.
    /// <para>ベクトル内で最小値を持つ最初の要素の 0 始まりのインデックスを返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The vector to search. / 検索対象のベクトル。</param>
    /// <returns>The zero-based index of the first minimum element, or of the first <c>NaN</c> if any element is <c>NaN</c>. / 最小値を持つ最初の要素の 0 始まりのインデックス。<c>NaN</c> を含む場合は最初の <c>NaN</c> のインデックス。</returns>
    /// <remarks>
    /// <para>Treats <c>-0</c> as less than <c>+0</c>.</para>
    /// <para><c>-0</c> は <c>+0</c> より小さい値として扱います。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="x"/> is empty. / <paramref name="x"/> が空の場合。</exception>
    [InferenceOverloads]
    public static int IndexOfMin<T>(ReadOnlyVectorSpan<T> x)
        where T : unmanaged, INumber<T> =>
        StridedKernels.IndexOfMin(x);

    /// <summary>
    /// Returns the zero-based row and column indices of the first element with the minimum value in a matrix.
    /// <para>行列全体で最小値を持つ最初の要素の 0 始まりの行番号と列番号を返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The matrix to search. / 検索対象の行列。</param>
    /// <returns>The zero-based <c>(Row, Column)</c> position of the first minimum element, or of the first <c>NaN</c> if any element is <c>NaN</c>. / 最初に見つかった最小値（<c>NaN</c> を含む場合は最初の <c>NaN</c>）の 0 始まりの <c>(Row, Column)</c>。</returns>
    /// <remarks>
    /// <para>Searches in column-major order (columns first, then rows within each column) and treats <c>-0</c> as less than <c>+0</c>.</para>
    /// <para>列を先頭から順に、各列内は行番号順に検索します。<c>-0</c> は <c>+0</c> より小さい値として扱います。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="x"/> is empty. / <paramref name="x"/> が空の場合。</exception>
    [InferenceOverloads]
    public static (int Row, int Column) IndexOfMin<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumber<T> =>
        StridedKernels.IndexOfMin(x);

    /// <summary>
    /// Returns the zero-based row index of the first minimum element in each column of a matrix.
    /// <para>行列の各列について、最小値を持つ最初の要素の行インデックスを返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new array of length <c>x.Columns</c> containing the row index of the minimum in each column. / 各列の最小値の行インデックスを格納した長さ <c>x.Columns</c> の新しい配列。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> has columns but zero rows. / <paramref name="x"/> の列数が 1 以上で行数が 0 の場合。</exception>
    [InferenceOverloads]
    public static int[] ColumnIndexOfMin<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumber<T>
    {
        var result = new int[x.Columns];
        ColumnIndexOfMin(x, result);
        return result;
    }

    /// <summary>
    /// Computes the zero-based row index of the first minimum element in each column of a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各列で最小値を持つ最初の要素の行インデックスを計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span of length <c>x.Columns</c> to receive the indices. / インデックスを書き込む長さ <c>x.Columns</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> has columns but zero rows, or the length of <paramref name="destination"/> does not equal <c>x.Columns</c>. / <paramref name="x"/> の列数が 1 以上で行数が 0 の場合、または <paramref name="destination"/> の長さが <c>x.Columns</c> と一致しない場合。</exception>
    [InferenceOverloads]
    public static Span<int> ColumnIndexOfMin<T>(ReadOnlyMatrixSpan<T> x, Span<int> destination)
        where T : unmanaged, INumber<T>
    {
        StridedKernels.ColumnIndexOfMin(x, destination);
        return destination;
    }

    /// <summary>
    /// Returns the zero-based column index of the first minimum element in each row of a matrix.
    /// <para>行列の各行について、最小値を持つ最初の要素の列インデックスを返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new array of length <c>x.Rows</c> containing the column index of the minimum in each row. / 各行の最小値の列インデックスを格納した長さ <c>x.Rows</c> の新しい配列。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> has rows but zero columns. / <paramref name="x"/> の行数が 1 以上で列数が 0 の場合。</exception>
    [InferenceOverloads]
    public static int[] RowIndexOfMin<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumber<T>
    {
        var result = new int[x.Rows];
        RowIndexOfMin(x, result);
        return result;
    }

    /// <summary>
    /// Computes the zero-based column index of the first minimum element in each row of a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各行で最小値を持つ最初の要素の列インデックスを計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span of length <c>x.Rows</c> to receive the indices. / インデックスを書き込む長さ <c>x.Rows</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> has rows but zero columns, or the length of <paramref name="destination"/> does not equal <c>x.Rows</c>. / <paramref name="x"/> の行数が 1 以上で列数が 0 の場合、または <paramref name="destination"/> の長さが <c>x.Rows</c> と一致しない場合。</exception>
    [InferenceOverloads]
    public static Span<int> RowIndexOfMin<T>(ReadOnlyMatrixSpan<T> x, Span<int> destination)
        where T : unmanaged, INumber<T>
    {
        StridedKernels.RowIndexOfMin(x, destination);
        return destination;
    }

    /// <summary>
    /// Computes the sum of the elements in a vector, optionally excluding <c>NaN</c> values.
    /// <para>必要に応じて <c>NaN</c> を除外して、ベクトル要素の総和を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <returns>The sum of the elements. When <paramref name="ignoreNaN"/> is <see langword="true"/>, returns <c>+0</c> if <paramref name="x"/> is empty or contains only <c>NaN</c> values. / 要素の総和。<paramref name="ignoreNaN"/> が <see langword="true"/> の場合、空またはすべて <c>NaN</c> のときは <c>+0</c>。</returns>
    /// <remarks>
    /// <para>When <paramref name="ignoreNaN"/> is <see langword="false"/>, behaves identically to <see cref="Sum{T}(ReadOnlyVectorSpan{T})"/>.</para>
    /// <para><paramref name="ignoreNaN"/> が <see langword="false"/> の場合は <see cref="Sum{T}(ReadOnlyVectorSpan{T})"/> と同じ動作になります。</para>
    /// </remarks>
    [InferenceOverloads]
    public static T Sum<T>(ReadOnlyVectorSpan<T> x, bool ignoreNaN)
        where T : unmanaged, INumberBase<T> =>
        ignoreNaN ? StridedKernels.SumIgnoringNaN(x) : StridedKernels.Sum(x);

    /// <summary>
    /// Computes the sum of each column in a matrix, optionally excluding <c>NaN</c> values.
    /// <para>必要に応じて <c>NaN</c> を除外して、行列の各列の総和を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <returns>A new vector of length <c>x.Columns</c> containing the column sums. / 各列の総和を格納した長さ <c>x.Columns</c> の新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> ColumnSums<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN)
        where T : unmanaged, INumberBase<T>
    {
        var result = Vector<T>.Uninitialized(x.Columns);
        ColumnSums(x, ignoreNaN, result);
        return result;
    }

    /// <summary>
    /// Computes the sum of each column in a matrix, optionally excluding <c>NaN</c> values, and writes the result to <paramref name="destination"/>.
    /// <para>必要に応じて <c>NaN</c> を除外して行列の各列の総和を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <param name="destination">The destination span of length <c>x.Columns</c> to receive the column sums. / 各列の総和を書き込む長さ <c>x.Columns</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Columns</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Columns</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> ColumnSums<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        if (ignoreNaN)
        {
            StridedKernels.ColumnSumsIgnoringNaN(x, destination);
        }
        else
        {
            StridedKernels.ColumnSums(x, destination);
        }

        return destination;
    }

    /// <summary>
    /// Computes the sum of each row in a matrix, optionally excluding <c>NaN</c> values.
    /// <para>必要に応じて <c>NaN</c> を除外して、行列の各行の総和を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <returns>A new vector of length <c>x.Rows</c> containing the row sums. / 各行の総和を格納した長さ <c>x.Rows</c> の新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> RowSums<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN)
        where T : unmanaged, INumberBase<T>
    {
        var result = Vector<T>.Uninitialized(x.Rows);
        RowSums(x, ignoreNaN, result);
        return result;
    }

    /// <summary>
    /// Computes the sum of each row in a matrix, optionally excluding <c>NaN</c> values, and writes the result to <paramref name="destination"/>.
    /// <para>必要に応じて <c>NaN</c> を除外して行列の各行の総和を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <param name="destination">The destination span of length <c>x.Rows</c> to receive the row sums. / 各行の総和を書き込む長さ <c>x.Rows</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Rows</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Rows</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> RowSums<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        if (ignoreNaN)
        {
            StridedKernels.RowSumsIgnoringNaN(x, destination);
        }
        else
        {
            StridedKernels.RowSums(x, destination);
        }

        return destination;
    }

    /// <summary>
    /// Computes the arithmetic mean of the elements in a vector, optionally excluding <c>NaN</c> values.
    /// <para>必要に応じて <c>NaN</c> を除外して、ベクトル要素の算術平均を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <returns>The arithmetic mean, or <c>NaN</c> if no non-<c>NaN</c> elements remain for floating-point types. / 算術平均。浮動小数点型で有効な要素がない場合は <c>NaN</c>。</returns>
    /// <remarks>
    /// <para>Handles intermediate sum overflow in the same way as <see cref="Mean{T}(ReadOnlyVectorSpan{T})"/>. When <paramref name="ignoreNaN"/> is <see langword="false"/>, behaves identically to <see cref="Mean{T}(ReadOnlyVectorSpan{T})"/>.</para>
    /// <para>途中の総和がオーバーフローした場合の扱いは <see cref="Mean{T}(ReadOnlyVectorSpan{T})"/> と同じです。<paramref name="ignoreNaN"/> が <see langword="false"/> の場合は <see cref="Mean{T}(ReadOnlyVectorSpan{T})"/> と同じ動作になります。</para>
    /// </remarks>
    [InferenceOverloads]
    public static T Mean<T>(ReadOnlyVectorSpan<T> x, bool ignoreNaN)
        where T : unmanaged, INumberBase<T> =>
        ignoreNaN ? StridedKernels.MeanIgnoringNaN(x) : StridedKernels.Mean(x);

    /// <summary>
    /// Computes the arithmetic mean of each column in a matrix, optionally excluding <c>NaN</c> values.
    /// <para>必要に応じて <c>NaN</c> を除外して、行列の各列の算術平均を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <returns>A new vector of length <c>x.Columns</c> containing the column means. / 各列の算術平均を格納した長さ <c>x.Columns</c> の新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> ColumnMeans<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN)
        where T : unmanaged, INumberBase<T>
    {
        var result = Vector<T>.Uninitialized(x.Columns);
        ColumnMeans(x, ignoreNaN, result);
        return result;
    }

    /// <summary>
    /// Computes the arithmetic mean of each column in a matrix, optionally excluding <c>NaN</c> values, and writes the result to <paramref name="destination"/>.
    /// <para>必要に応じて <c>NaN</c> を除外して行列の各列の算術平均を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <param name="destination">The destination span of length <c>x.Columns</c> to receive the column means. / 各列の算術平均を書き込む長さ <c>x.Columns</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Columns</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Columns</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> ColumnMeans<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        if (ignoreNaN)
        {
            StridedKernels.ColumnMeansIgnoringNaN(x, destination);
        }
        else
        {
            StridedKernels.ColumnMeans(x, destination);
        }

        return destination;
    }

    /// <summary>
    /// Computes the arithmetic mean of each row in a matrix, optionally excluding <c>NaN</c> values.
    /// <para>必要に応じて <c>NaN</c> を除外して、行列の各行の算術平均を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <returns>A new vector of length <c>x.Rows</c> containing the row means. / 各行の算術平均を格納した長さ <c>x.Rows</c> の新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> RowMeans<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN)
        where T : unmanaged, INumberBase<T>
    {
        var result = Vector<T>.Uninitialized(x.Rows);
        RowMeans(x, ignoreNaN, result);
        return result;
    }

    /// <summary>
    /// Computes the arithmetic mean of each row in a matrix, optionally excluding <c>NaN</c> values, and writes the result to <paramref name="destination"/>.
    /// <para>必要に応じて <c>NaN</c> を除外して行列の各行の算術平均を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <param name="destination">The destination span of length <c>x.Rows</c> to receive the row means. / 各行の算術平均を書き込む長さ <c>x.Rows</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Rows</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Rows</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> RowMeans<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        if (ignoreNaN)
        {
            StridedKernels.RowMeansIgnoringNaN(x, destination);
        }
        else
        {
            StridedKernels.RowMeans(x, destination);
        }

        return destination;
    }

    /// <summary>
    /// Returns the maximum element in a vector, optionally excluding <c>NaN</c> values.
    /// <para>必要に応じて <c>NaN</c> を除外して、ベクトル要素の最大値を返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <returns>The maximum element, or <c>NaN</c> if all elements are <c>NaN</c> (or if <paramref name="ignoreNaN"/> is <see langword="false"/> and any element is <c>NaN</c>). / 最大値。すべて <c>NaN</c> の場合、または <paramref name="ignoreNaN"/> が <see langword="false"/> で <c>NaN</c> を含む場合は <c>NaN</c>。</returns>
    /// <remarks>
    /// <para>When <paramref name="ignoreNaN"/> is <see langword="true"/>, folds with <c>T.MaxNumber</c>. When <see langword="false"/>, behaves identically to <see cref="Max{T}(ReadOnlyVectorSpan{T})"/>.</para>
    /// <para><paramref name="ignoreNaN"/> が <see langword="true"/> の場合は <c>T.MaxNumber</c> で畳み込みます。<see langword="false"/> の場合は <see cref="Max{T}(ReadOnlyVectorSpan{T})"/> と同じ動作になります。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="x"/> is empty. / <paramref name="x"/> が空の場合。</exception>
    [InferenceOverloads]
    public static T Max<T>(ReadOnlyVectorSpan<T> x, bool ignoreNaN)
        where T : unmanaged, INumber<T> =>
        ignoreNaN ? StridedKernels.MaxIgnoringNaN(x) : StridedKernels.Max(x);

    /// <summary>
    /// Returns the minimum element in a vector, optionally excluding <c>NaN</c> values.
    /// <para>必要に応じて <c>NaN</c> を除外して、ベクトル要素の最小値を返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <returns>The minimum element, or <c>NaN</c> if all elements are <c>NaN</c> (or if <paramref name="ignoreNaN"/> is <see langword="false"/> and any element is <c>NaN</c>). / 最小値。すべて <c>NaN</c> の場合、または <paramref name="ignoreNaN"/> が <see langword="false"/> で <c>NaN</c> を含む場合は <c>NaN</c>。</returns>
    /// <remarks>
    /// <para>When <paramref name="ignoreNaN"/> is <see langword="true"/>, folds with <c>T.MinNumber</c>. When <see langword="false"/>, behaves identically to <see cref="Min{T}(ReadOnlyVectorSpan{T})"/>.</para>
    /// <para><paramref name="ignoreNaN"/> が <see langword="true"/> の場合は <c>T.MinNumber</c> で畳み込みます。<see langword="false"/> の場合は <see cref="Min{T}(ReadOnlyVectorSpan{T})"/> と同じ動作になります。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="x"/> is empty. / <paramref name="x"/> が空の場合。</exception>
    [InferenceOverloads]
    public static T Min<T>(ReadOnlyVectorSpan<T> x, bool ignoreNaN)
        where T : unmanaged, INumber<T> =>
        ignoreNaN ? StridedKernels.MinIgnoringNaN(x) : StridedKernels.Min(x);

    /// <summary>
    /// Computes the maximum element of each column in a matrix, optionally excluding <c>NaN</c> values.
    /// <para>必要に応じて <c>NaN</c> を除外して、行列の各列の最大値を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <returns>A new vector of length <c>x.Columns</c> containing the column maxima. / 各列の最大値を格納した長さ <c>x.Columns</c> の新しいベクトル。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> has columns but zero rows. / <paramref name="x"/> の列数が 1 以上で行数が 0 の場合。</exception>
    [InferenceOverloads]
    public static Vector<T> ColumnMax<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN)
        where T : unmanaged, INumber<T>
    {
        var result = Vector<T>.Uninitialized(x.Columns);
        ColumnMax(x, ignoreNaN, result);
        return result;
    }

    /// <summary>
    /// Computes the maximum element of each column in a matrix, optionally excluding <c>NaN</c> values, and writes the result to <paramref name="destination"/>.
    /// <para>必要に応じて <c>NaN</c> を除外して行列の各列の最大値を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <param name="destination">The destination span of length <c>x.Columns</c> to receive the column maxima. / 各列の最大値を書き込む長さ <c>x.Columns</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="x"/> has columns but zero rows, the length of <paramref name="destination"/> does not equal <c>x.Columns</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="x"/> の列数が 1 以上で行数が 0 の場合、<paramref name="destination"/> の長さが <c>x.Columns</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> ColumnMax<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN, VectorSpan<T> destination)
        where T : unmanaged, INumber<T>
    {
        if (ignoreNaN)
        {
            StridedKernels.ColumnMaxIgnoringNaN(x, destination);
        }
        else
        {
            StridedKernels.ColumnMax(x, destination);
        }

        return destination;
    }

    /// <summary>
    /// Computes the maximum element of each row in a matrix, optionally excluding <c>NaN</c> values.
    /// <para>必要に応じて <c>NaN</c> を除外して、行列の各行の最大値を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <returns>A new vector of length <c>x.Rows</c> containing the row maxima. / 各行の最大値を格納した長さ <c>x.Rows</c> の新しいベクトル。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> has rows but zero columns. / <paramref name="x"/> の行数が 1 以上で列数が 0 の場合。</exception>
    [InferenceOverloads]
    public static Vector<T> RowMax<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN)
        where T : unmanaged, INumber<T>
    {
        var result = Vector<T>.Uninitialized(x.Rows);
        RowMax(x, ignoreNaN, result);
        return result;
    }

    /// <summary>
    /// Computes the maximum element of each row in a matrix, optionally excluding <c>NaN</c> values, and writes the result to <paramref name="destination"/>.
    /// <para>必要に応じて <c>NaN</c> を除外して行列の各行の最大値を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <param name="destination">The destination span of length <c>x.Rows</c> to receive the row maxima. / 各行の最大値を書き込む長さ <c>x.Rows</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="x"/> has rows but zero columns, the length of <paramref name="destination"/> does not equal <c>x.Rows</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="x"/> の行数が 1 以上で列数が 0 の場合、<paramref name="destination"/> の長さが <c>x.Rows</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> RowMax<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN, VectorSpan<T> destination)
        where T : unmanaged, INumber<T>
    {
        if (ignoreNaN)
        {
            StridedKernels.RowMaxIgnoringNaN(x, destination);
        }
        else
        {
            StridedKernels.RowMax(x, destination);
        }

        return destination;
    }

    /// <summary>
    /// Computes the minimum element of each column in a matrix, optionally excluding <c>NaN</c> values.
    /// <para>必要に応じて <c>NaN</c> を除外して、行列の各列の最小値を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <returns>A new vector of length <c>x.Columns</c> containing the column minima. / 各列の最小値を格納した長さ <c>x.Columns</c> の新しいベクトル。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> has columns but zero rows. / <paramref name="x"/> の列数が 1 以上で行数が 0 の場合。</exception>
    [InferenceOverloads]
    public static Vector<T> ColumnMin<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN)
        where T : unmanaged, INumber<T>
    {
        var result = Vector<T>.Uninitialized(x.Columns);
        ColumnMin(x, ignoreNaN, result);
        return result;
    }

    /// <summary>
    /// Computes the minimum element of each column in a matrix, optionally excluding <c>NaN</c> values, and writes the result to <paramref name="destination"/>.
    /// <para>必要に応じて <c>NaN</c> を除外して行列の各列の最小値を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <param name="destination">The destination span of length <c>x.Columns</c> to receive the column minima. / 各列の最小値を書き込む長さ <c>x.Columns</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="x"/> has columns but zero rows, the length of <paramref name="destination"/> does not equal <c>x.Columns</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="x"/> の列数が 1 以上で行数が 0 の場合、<paramref name="destination"/> の長さが <c>x.Columns</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> ColumnMin<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN, VectorSpan<T> destination)
        where T : unmanaged, INumber<T>
    {
        if (ignoreNaN)
        {
            StridedKernels.ColumnMinIgnoringNaN(x, destination);
        }
        else
        {
            StridedKernels.ColumnMin(x, destination);
        }

        return destination;
    }

    /// <summary>
    /// Computes the minimum element of each row in a matrix, optionally excluding <c>NaN</c> values.
    /// <para>必要に応じて <c>NaN</c> を除外して、行列の各行の最小値を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <returns>A new vector of length <c>x.Rows</c> containing the row minima. / 各行の最小値を格納した長さ <c>x.Rows</c> の新しいベクトル。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> has rows but zero columns. / <paramref name="x"/> の行数が 1 以上で列数が 0 の場合。</exception>
    [InferenceOverloads]
    public static Vector<T> RowMin<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN)
        where T : unmanaged, INumber<T>
    {
        var result = Vector<T>.Uninitialized(x.Rows);
        RowMin(x, ignoreNaN, result);
        return result;
    }

    /// <summary>
    /// Computes the minimum element of each row in a matrix, optionally excluding <c>NaN</c> values, and writes the result to <paramref name="destination"/>.
    /// <para>必要に応じて <c>NaN</c> を除外して行列の各行の最小値を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="ignoreNaN"><see langword="true"/> to exclude <c>NaN</c> elements; <see langword="false"/> to propagate <c>NaN</c>. / <c>NaN</c> の要素を除外する場合は <see langword="true"/>、伝播させる場合は <see langword="false"/>。</param>
    /// <param name="destination">The destination span of length <c>x.Rows</c> to receive the row minima. / 各行の最小値を書き込む長さ <c>x.Rows</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="x"/> has rows but zero columns, the length of <paramref name="destination"/> does not equal <c>x.Rows</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="x"/> の行数が 1 以上で列数が 0 の場合、<paramref name="destination"/> の長さが <c>x.Rows</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> RowMin<T>(ReadOnlyMatrixSpan<T> x, bool ignoreNaN, VectorSpan<T> destination)
        where T : unmanaged, INumber<T>
    {
        if (ignoreNaN)
        {
            StridedKernels.RowMinIgnoringNaN(x, destination);
        }
        else
        {
            StridedKernels.RowMin(x, destination);
        }

        return destination;
    }

    /// <summary>
    /// Computes the dot product <c>Σ x[i]·y[i]</c> of two vectors.
    /// <para>2つのベクトルの内積 <c>Σ x[i]·y[i]</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The first input vector. / 第1入力ベクトル。</param>
    /// <param name="y">The second input vector. / 第2入力ベクトル。</param>
    /// <returns>The dot product <c>Σ x[i]·y[i]</c>. / 内積 <c>Σ x[i]·y[i]</c>。</returns>
    /// <remarks>
    /// <para>Does not conjugate complex elements.</para>
    /// <para>複素数でも共役はとりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="x"/> and <paramref name="y"/> have different lengths. / <paramref name="x"/> と <paramref name="y"/> の長さが異なる場合。</exception>
    [InferenceOverloads]
    public static T Dot<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)
        where T : unmanaged, INumberBase<T> =>
        StridedKernels.Dot(x, y);

    /// <summary>
    /// Computes the Euclidean (L2) norm <c>‖x‖₂</c> of a vector without undue intermediate overflow or underflow.
    /// <para>途中のオーバーフローやアンダーフローを回避して、ベクトルのユークリッドノルム <c>‖x‖₂</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>The Euclidean norm <c>‖x‖₂</c>, or <c>NaN</c> if any element is <c>NaN</c>. / ユークリッドノルム <c>‖x‖₂</c>。<c>NaN</c> を含む場合は <c>NaN</c>。</returns>
    [InferenceOverloads]
    public static T Norm<T>(ReadOnlyVectorSpan<T> x)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        StridedKernels.Norm(x);

    /// <summary>
    /// Computes the L1 norm <c>Σ |x[i]|</c> of a vector.
    /// <para>ベクトルの 1 ノルム <c>Σ |x[i]|</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>The L1 norm <c>Σ |x[i]|</c>. / 1 ノルム <c>Σ |x[i]|</c>。</returns>
    [InferenceOverloads]
    public static T Norm1<T>(ReadOnlyVectorSpan<T> x)
        where T : unmanaged, INumberBase<T> =>
        StridedKernels.Norm1(x);

    /// <summary>
    /// Computes the infinity (maximum) norm <c>max |x[i]|</c> of a vector.
    /// <para>ベクトルの最大値ノルム <c>max |x[i]|</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>The infinity norm <c>max |x[i]|</c>, zero if <paramref name="x"/> is empty, or <c>NaN</c> if any element is <c>NaN</c>. / 最大値ノルム <c>max |x[i]|</c>。<paramref name="x"/> が空の場合は 0、<c>NaN</c> を含む場合は <c>NaN</c>。</returns>
    [InferenceOverloads]
    public static T NormInfinity<T>(ReadOnlyVectorSpan<T> x)
        where T : unmanaged, INumber<T> =>
        StridedKernels.NormInfinity(x);

    /// <summary>
    /// Computes the Euclidean (L2) norm of each column in a matrix.
    /// <para>行列の各列のユークリッドノルムを計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new vector of length <c>x.Columns</c> containing the column norms. / 各列のユークリッドノルムを格納した長さ <c>x.Columns</c> の新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> ColumnNorms<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = Vector<T>.Uninitialized(x.Columns);
        ColumnNorms(x, result);
        return result;
    }

    /// <summary>
    /// Computes the Euclidean (L2) norm of each column in a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各列のユークリッドノルムを計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span of length <c>x.Columns</c> to receive the column norms. / 各列のユークリッドノルムを書き込む長さ <c>x.Columns</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Columns</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Columns</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> ColumnNorms<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        StridedKernels.ColumnNorms(x, destination);
        return destination;
    }

    /// <summary>
    /// Computes the Euclidean (L2) norm of each row in a matrix.
    /// <para>行列の各行のユークリッドノルムを計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new vector of length <c>x.Rows</c> containing the row norms. / 各行のユークリッドノルムを格納した長さ <c>x.Rows</c> の新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> RowNorms<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = Vector<T>.Uninitialized(x.Rows);
        RowNorms(x, result);
        return result;
    }

    /// <summary>
    /// Computes the Euclidean (L2) norm of each row in a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各行のユークリッドノルムを計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span of length <c>x.Rows</c> to receive the row norms. / 各行のユークリッドノルムを書き込む長さ <c>x.Rows</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Rows</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Rows</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> RowNorms<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        StridedKernels.RowNorms(x, destination);
        return destination;
    }

    /// <summary>
    /// Computes the log-sum-exp <c>ln Σ exp(x[i])</c> of the elements in a vector without intermediate overflow.
    /// <para>途中のオーバーフローを回避して、ベクトル要素の <c>ln Σ exp(x[i])</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns><c>ln Σ exp(x[i])</c>, <c>-∞</c> if <paramref name="x"/> is empty, <c>+∞</c> if any element is <c>+∞</c> (without <c>NaN</c>), or <c>NaN</c> if any element is <c>NaN</c>. / <c>ln Σ exp(x[i])</c>。空の場合は <c>-∞</c>、<c>+∞</c> を含む場合（<c>NaN</c> がないとき）は <c>+∞</c>、<c>NaN</c> を含む場合は <c>NaN</c>。</returns>
    [InferenceOverloads]
    public static T LogSumExp<T>(ReadOnlyVectorSpan<T> x)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        StridedKernels.LogSumExp(x);

    /// <summary>
    /// Computes the log-sum-exp <c>ln Σ exp</c> of each column in a matrix.
    /// <para>行列の各列の <c>ln Σ exp</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new vector of length <c>x.Columns</c> containing the column log-sum-exp values. / 各列の <c>ln Σ exp</c> を格納した長さ <c>x.Columns</c> の新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> ColumnLogSumExp<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = Vector<T>.Uninitialized(x.Columns);
        ColumnLogSumExp(x, result);
        return result;
    }

    /// <summary>
    /// Computes the log-sum-exp <c>ln Σ exp</c> of each column in a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各列の <c>ln Σ exp</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span of length <c>x.Columns</c> to receive the column log-sum-exp values. / 各列の <c>ln Σ exp</c> を書き込む長さ <c>x.Columns</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Columns</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Columns</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> ColumnLogSumExp<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        StridedKernels.ColumnLogSumExp(x, destination);
        return destination;
    }

    /// <summary>
    /// Computes the log-sum-exp <c>ln Σ exp</c> of each row in a matrix.
    /// <para>行列の各行の <c>ln Σ exp</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new vector of length <c>x.Rows</c> containing the row log-sum-exp values. / 各行の <c>ln Σ exp</c> を格納した長さ <c>x.Rows</c> の新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> RowLogSumExp<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = Vector<T>.Uninitialized(x.Rows);
        RowLogSumExp(x, result);
        return result;
    }

    /// <summary>
    /// Computes the log-sum-exp <c>ln Σ exp</c> of each row in a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各行の <c>ln Σ exp</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type. / 浮動小数点要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span of length <c>x.Rows</c> to receive the row log-sum-exp values. / 各行の <c>ln Σ exp</c> を書き込む長さ <c>x.Rows</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <c>x.Rows</c>, or <paramref name="destination"/> overlaps <paramref name="x"/>. / <paramref name="destination"/> の長さが <c>x.Rows</c> と一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> RowLogSumExp<T>(ReadOnlyMatrixSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        StridedKernels.RowLogSumExp(x, destination);
        return destination;
    }
}
