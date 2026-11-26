using System.Numerics;

namespace Tuna;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

public static partial class XMath
{
    /// <summary>
    /// Computes the softmax distribution over a vector.
    /// <para>ベクトル全体に対するソフトマックス分布を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The numeric type. / 数値型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>A new vector whose non-negative elements sum to 1. / 非負で総和が 1 となる新しいベクトル。</returns>
    /// <remarks>
    /// <para>Shifts inputs by the maximum element to avoid overflow. Any <c>NaN</c> or a maximum of <c>±∞</c> produces all <c>NaN</c> values.</para>
    /// <para>オーバーフローを避けるため最大値を引いて計算します。<c>NaN</c> を含む場合や最大値が <c>±∞</c> の場合は全要素が <c>NaN</c> になります。</para>
    /// </remarks>
    [InferenceOverloads]
    public static Vector<T> Softmax<T>(ReadOnlyVectorSpan<T> x)
        where T : unmanaged, INumber<T>, IExponentialFunctions<T>
    {
        var result = Vector<T>.Uninitialized(x.Length);
        Softmax(x, result);
        return result;
    }

    /// <summary>
    /// Computes the softmax distribution over a vector and writes the result to <paramref name="destination"/>.
    /// <para>ベクトル全体に対するソフトマックス分布を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The numeric type. / 数値型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static VectorSpan<T> Softmax<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, INumber<T>, IExponentialFunctions<T>
    {
        StridedKernels.ColumnSoftmax(x.AsMatrix(), destination.AsMatrix());
        return destination;
    }

    /// <summary>
    /// Computes the softmax distribution independently for each column of a matrix.
    /// <para>行列の各列ごとにソフトマックス分布を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The numeric type. / 数値型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new matrix where each column sums to 1. / 各列の和が 1 となる新しい行列。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Matrix<T> ColumnSoftmax<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumber<T>, IExponentialFunctions<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        ColumnSoftmax(x, result);
        return result;
    }

    /// <summary>
    /// Computes the softmax distribution independently for each column of a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各列ごとにソフトマックス分布を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The numeric type. / 数値型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static MatrixSpan<T> ColumnSoftmax<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)
        where T : unmanaged, INumber<T>, IExponentialFunctions<T>
    {
        StridedKernels.ColumnSoftmax(x, destination);
        return destination;
    }

    /// <summary>
    /// Computes the softmax distribution independently for each row of a matrix.
    /// <para>行列の各行ごとにソフトマックス分布を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The numeric type. / 数値型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new matrix where each row sums to 1. / 各行の和が 1 となる新しい行列。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Matrix<T> RowSoftmax<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumber<T>, IExponentialFunctions<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        RowSoftmax(x, result);
        return result;
    }

    /// <summary>
    /// Computes the softmax distribution independently for each row of a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各行ごとにソフトマックス分布を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The numeric type. / 数値型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static MatrixSpan<T> RowSoftmax<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)
        where T : unmanaged, INumber<T>, IExponentialFunctions<T>
    {
        StridedKernels.RowSoftmax(x, destination);
        return destination;
    }

    /// <summary>
    /// Computes the log-softmax <c>x - LogSumExp(x)</c> over a vector.
    /// <para>ベクトル全体に対する対数ソフトマックス <c>x - LogSumExp(x)</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>A new vector containing the log-softmax values. / 対数ソフトマックスを格納した新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> LogSoftmax<T>(ReadOnlyVectorSpan<T> x)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = Vector<T>.Uninitialized(x.Length);
        LogSoftmax(x, result);
        return result;
    }

    /// <summary>
    /// Computes the log-softmax <c>x - LogSumExp(x)</c> over a vector and writes the result to <paramref name="destination"/>.
    /// <para>ベクトル全体に対する対数ソフトマックスを計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static VectorSpan<T> LogSoftmax<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        StridedKernels.ColumnLogSoftmax(x.AsMatrix(), destination.AsMatrix());
        return destination;
    }

    /// <summary>
    /// Computes the log-softmax independently for each column of a matrix.
    /// <para>行列の各列ごとに対数ソフトマックスを計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new matrix containing the column-wise log-softmax values. / 各列の対数ソフトマックスを格納した新しい行列。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Matrix<T> ColumnLogSoftmax<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        ColumnLogSoftmax(x, result);
        return result;
    }

    /// <summary>
    /// Computes the log-softmax independently for each column of a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各列ごとに対数ソフトマックスを計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static MatrixSpan<T> ColumnLogSoftmax<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        StridedKernels.ColumnLogSoftmax(x, destination);
        return destination;
    }

    /// <summary>
    /// Computes the log-softmax independently for each row of a matrix.
    /// <para>行列の各行ごとに対数ソフトマックスを計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new matrix containing the row-wise log-softmax values. / 各行の対数ソフトマックスを格納した新しい行列。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Matrix<T> RowLogSoftmax<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        RowLogSoftmax(x, result);
        return result;
    }

    /// <summary>
    /// Computes the log-softmax independently for each row of a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各行ごとに対数ソフトマックスを計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static MatrixSpan<T> RowLogSoftmax<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        StridedKernels.RowLogSoftmax(x, destination);
        return destination;
    }

    public static partial class InPlace
    {
        /// <summary>
        /// Computes the softmax distribution over a vector in place (<c>x ← softmax(x)</c>).
        /// <para>ベクトル全体に対するソフトマックス分布をインプレースで計算します（<c>x ← softmax(x)</c>）。</para>
        /// </summary>
        /// <typeparam name="T">The numeric type. / 数値型。</typeparam>
        /// <param name="x">The vector to modify in place. / インプレースで更新するベクトル。</param>
        [InferenceOverloads]
        public static void Softmax<T>(VectorSpan<T> x)
            where T : unmanaged, INumber<T>, IExponentialFunctions<T> =>
            StridedKernels.ColumnSoftmax(x.AsMatrix(), x.AsMatrix());

        /// <summary>
        /// Computes the column-wise softmax distribution of a matrix in place.
        /// <para>行列の各列ごとのソフトマックス分布をインプレースで計算します。</para>
        /// </summary>
        /// <typeparam name="T">The numeric type. / 数値型。</typeparam>
        /// <param name="x">The matrix to modify in place. / インプレースで更新する行列。</param>
        [InferenceOverloads]
        public static void ColumnSoftmax<T>(MatrixSpan<T> x)
            where T : unmanaged, INumber<T>, IExponentialFunctions<T> =>
            StridedKernels.ColumnSoftmax(x, x);

        /// <summary>
        /// Computes the row-wise softmax distribution of a matrix in place.
        /// <para>行列の各行ごとのソフトマックス分布をインプレースで計算します。</para>
        /// </summary>
        /// <typeparam name="T">The numeric type. / 数値型。</typeparam>
        /// <param name="x">The matrix to modify in place. / インプレースで更新する行列。</param>
        [InferenceOverloads]
        public static void RowSoftmax<T>(MatrixSpan<T> x)
            where T : unmanaged, INumber<T>, IExponentialFunctions<T> =>
            StridedKernels.RowSoftmax(x, x);

        /// <summary>
        /// Computes the log-softmax <c>x ← x - LogSumExp(x)</c> over a vector in place.
        /// <para>ベクトル全体に対する対数ソフトマックスをインプレースで計算します（<c>x ← x - LogSumExp(x)</c>）。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
        /// <param name="x">The vector to modify in place. / インプレースで更新するベクトル。</param>
        [InferenceOverloads]
        public static void LogSoftmax<T>(VectorSpan<T> x)
            where T : unmanaged, IFloatingPointIeee754<T> =>
            StridedKernels.ColumnLogSoftmax(x.AsMatrix(), x.AsMatrix());

        /// <summary>
        /// Computes the column-wise log-softmax of a matrix in place.
        /// <para>行列の各列ごとの対数ソフトマックスをインプレースで計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
        /// <param name="x">The matrix to modify in place. / インプレースで更新する行列。</param>
        [InferenceOverloads]
        public static void ColumnLogSoftmax<T>(MatrixSpan<T> x)
            where T : unmanaged, IFloatingPointIeee754<T> =>
            StridedKernels.ColumnLogSoftmax(x, x);

        /// <summary>
        /// Computes the row-wise log-softmax of a matrix in place.
        /// <para>行列の各行ごとの対数ソフトマックスをインプレースで計算します。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
        /// <param name="x">The matrix to modify in place. / インプレースで更新する行列。</param>
        [InferenceOverloads]
        public static void RowLogSoftmax<T>(MatrixSpan<T> x)
            where T : unmanaged, IFloatingPointIeee754<T> =>
            StridedKernels.RowLogSoftmax(x, x);
    }
}
