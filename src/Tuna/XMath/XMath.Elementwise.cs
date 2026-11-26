using System.Numerics;

namespace Tuna;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

public static partial class XMath
{
    /// <summary>
    /// Computes the elementwise sum <c>x + y</c> of two vectors.
    /// <para>2つのベクトルの要素ごとの和 <c>x + y</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The first input vector. / 第1入力ベクトル。</param>
    /// <param name="y">The second input vector. / 第2入力ベクトル。</param>
    /// <returns>A new vector containing <c>x + y</c>. / 要素ごとの和を格納した新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> Add<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)
        where T : unmanaged, INumberBase<T>
    {
        var result = Vector<T>.Uninitialized(x.Length);
        Add(x, y, result);
        return result;
    }

    /// <summary>
    /// Computes the elementwise sum <c>x + y</c> of two vectors and writes the result to <paramref name="destination"/>.
    /// <para>2つのベクトルの要素ごとの和 <c>x + y</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The first input vector. / 第1入力ベクトル。</param>
    /// <param name="y">The second input vector. / 第2入力ベクトル。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static VectorSpan<T> Add<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Zip(x.AsMatrix(), y.AsMatrix(), destination.AsMatrix(), default(AddOperator<T>));
        return destination;
    }

    /// <summary>
    /// Computes the elementwise sum <c>x + y</c> of two matrices.
    /// <para>2つの行列の要素ごとの和 <c>x + y</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The first input matrix. / 第1入力行列。</param>
    /// <param name="y">The second input matrix. / 第2入力行列。</param>
    /// <returns>A new matrix containing <c>x + y</c>. / 要素ごとの和を格納した新しい行列。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Matrix<T> Add<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y)
        where T : unmanaged, INumberBase<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        Add(x, y, result);
        return result;
    }

    /// <summary>
    /// Computes the elementwise sum <c>x + y</c> of two matrices and writes the result to <paramref name="destination"/>.
    /// <para>2つの行列の要素ごとの和 <c>x + y</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The first input matrix. / 第1入力行列。</param>
    /// <param name="y">The second input matrix. / 第2入力行列。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static MatrixSpan<T> Add<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Zip(x, y, destination, default(AddOperator<T>));
        return destination;
    }

    /// <summary>
    /// Computes the elementwise difference <c>x - y</c> of two vectors.
    /// <para>2つのベクトルの要素ごとの差 <c>x - y</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The minuend vector. / 被減数ベクトル。</param>
    /// <param name="y">The subtrahend vector. / 減数ベクトル。</param>
    /// <returns>A new vector containing <c>x - y</c>. / 要素ごとの差を格納した新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> Subtract<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)
        where T : unmanaged, INumberBase<T>
    {
        var result = Vector<T>.Uninitialized(x.Length);
        Subtract(x, y, result);
        return result;
    }

    /// <summary>
    /// Computes the elementwise difference <c>x - y</c> of two vectors and writes the result to <paramref name="destination"/>.
    /// <para>2つのベクトルの要素ごとの差 <c>x - y</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The minuend vector. / 被減数ベクトル。</param>
    /// <param name="y">The subtrahend vector. / 減数ベクトル。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static VectorSpan<T> Subtract<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Zip(x.AsMatrix(), y.AsMatrix(), destination.AsMatrix(), default(SubtractOperator<T>));
        return destination;
    }

    /// <summary>
    /// Computes the elementwise difference <c>x - y</c> of two matrices.
    /// <para>2つの行列の要素ごとの差 <c>x - y</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The minuend matrix. / 被減数行列。</param>
    /// <param name="y">The subtrahend matrix. / 減数行列。</param>
    /// <returns>A new matrix containing <c>x - y</c>. / 要素ごとの差を格納した新しい行列。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Matrix<T> Subtract<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y)
        where T : unmanaged, INumberBase<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        Subtract(x, y, result);
        return result;
    }

    /// <summary>
    /// Computes the elementwise difference <c>x - y</c> of two matrices and writes the result to <paramref name="destination"/>.
    /// <para>2つの行列の要素ごとの差 <c>x - y</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The minuend matrix. / 被減数行列。</param>
    /// <param name="y">The subtrahend matrix. / 減数行列。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static MatrixSpan<T> Subtract<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Zip(x, y, destination, default(SubtractOperator<T>));
        return destination;
    }

    /// <summary>
    /// Computes the elementwise (Hadamard) product <c>x ∘ y</c> of two vectors.
    /// <para>2つのベクトルの要素ごとの積（アダマール積）<c>x ∘ y</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The first input vector. / 第1入力ベクトル。</param>
    /// <param name="y">The second input vector. / 第2入力ベクトル。</param>
    /// <returns>A new vector containing the elementwise product. / アダマール積を格納した新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> Hadamard<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)
        where T : unmanaged, INumberBase<T>
    {
        var result = Vector<T>.Uninitialized(x.Length);
        Hadamard(x, y, result);
        return result;
    }

    /// <summary>
    /// Computes the elementwise (Hadamard) product <c>x ∘ y</c> of two vectors and writes the result to <paramref name="destination"/>.
    /// <para>2つのベクトルの要素ごとの積（アダマール積）<c>x ∘ y</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The first input vector. / 第1入力ベクトル。</param>
    /// <param name="y">The second input vector. / 第2入力ベクトル。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static VectorSpan<T> Hadamard<T>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Zip(x.AsMatrix(), y.AsMatrix(), destination.AsMatrix(), default(MultiplyOperator<T>));
        return destination;
    }

    /// <summary>
    /// Computes the elementwise (Hadamard) product <c>x ∘ y</c> of two matrices.
    /// <para>2つの行列の要素ごとの積（アダマール積）<c>x ∘ y</c> を計算します（行列積ではありません）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The first input matrix. / 第1入力行列。</param>
    /// <param name="y">The second input matrix. / 第2入力行列。</param>
    /// <returns>A new matrix containing the elementwise product. / アダマール積を格納した新しい行列。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Matrix<T> Hadamard<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y)
        where T : unmanaged, INumberBase<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        Hadamard(x, y, result);
        return result;
    }

    /// <summary>
    /// Computes the elementwise (Hadamard) product <c>x ∘ y</c> of two matrices and writes the result to <paramref name="destination"/>.
    /// <para>2つの行列の要素ごとの積（アダマール積）<c>x ∘ y</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The first input matrix. / 第1入力行列。</param>
    /// <param name="y">The second input matrix. / 第2入力行列。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static MatrixSpan<T> Hadamard<T>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Zip(x, y, destination, default(MultiplyOperator<T>));
        return destination;
    }

    /// <summary>
    /// Computes the elementwise negation <c>-x</c> of a vector.
    /// <para>ベクトルの要素ごとの符号反転 <c>-x</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>A new vector containing <c>-x</c>. / 符号反転した新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> Negate<T>(ReadOnlyVectorSpan<T> x)
        where T : unmanaged, INumberBase<T>
    {
        var result = Vector<T>.Uninitialized(x.Length);
        Negate(x, result);
        return result;
    }

    /// <summary>
    /// Computes the elementwise negation <c>-x</c> of a vector and writes the result to <paramref name="destination"/>.
    /// <para>ベクトルの要素ごとの符号反転 <c>-x</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static VectorSpan<T> Negate<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Map(x.AsMatrix(), destination.AsMatrix(), default(NegateOperator<T>));
        return destination;
    }

    /// <summary>
    /// Computes the elementwise negation <c>-x</c> of a matrix.
    /// <para>行列の要素ごとの符号反転 <c>-x</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new matrix containing <c>-x</c>. / 符号反転した新しい行列。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Matrix<T> Negate<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumberBase<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        Negate(x, result);
        return result;
    }

    /// <summary>
    /// Computes the elementwise negation <c>-x</c> of a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の要素ごとの符号反転 <c>-x</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static MatrixSpan<T> Negate<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Map(x, destination, default(NegateOperator<T>));
        return destination;
    }

    /// <summary>
    /// Computes the elementwise exponential <c>exp(x)</c> of a vector.
    /// <para>ベクトルの要素ごとの指数関数 <c>exp(x)</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>A new vector containing <c>exp(x)</c>. / 要素ごとの指数関数を格納した新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> Exp<T>(ReadOnlyVectorSpan<T> x)
        where T : unmanaged, INumberBase<T>, IExponentialFunctions<T>
    {
        var result = Vector<T>.Uninitialized(x.Length);
        Exp(x, result);
        return result;
    }

    /// <summary>
    /// Computes the elementwise exponential <c>exp(x)</c> of a vector and writes the result to <paramref name="destination"/>.
    /// <para>ベクトルの要素ごとの指数関数 <c>exp(x)</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static VectorSpan<T> Exp<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>, IExponentialFunctions<T>
    {
        StridedKernels.Map(x.AsMatrix(), destination.AsMatrix(), default(ExpOperator<T>));
        return destination;
    }

    /// <summary>
    /// Computes the elementwise exponential <c>exp(x)</c> of a matrix.
    /// <para>行列の要素ごとの指数関数 <c>exp(x)</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new matrix containing <c>exp(x)</c>. / 要素ごとの指数関数を格納した新しい行列。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Matrix<T> Exp<T>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumberBase<T>, IExponentialFunctions<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        Exp(x, result);
        return result;
    }

    /// <summary>
    /// Computes the elementwise exponential <c>exp(x)</c> of a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の要素ごとの指数関数 <c>exp(x)</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static MatrixSpan<T> Exp<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>, IExponentialFunctions<T>
    {
        StridedKernels.Map(x, destination, default(ExpOperator<T>));
        return destination;
    }

    /// <summary>
    /// Adds a scalar to every element of a vector (<c>x + scalar</c>).
    /// <para>ベクトルの全要素にスカラーを加算します（<c>x + scalar</c>）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="scalar">The scalar value to add. / 加算するスカラー値。</param>
    /// <returns>A new vector containing <c>x + scalar</c>. / 計算結果を格納した新しいベクトル。</returns>
    public static Vector<T> Add<T>(ReadOnlyVectorSpan<T> x, T scalar)
        where T : unmanaged, INumberBase<T>
    {
        var result = Vector<T>.Uninitialized(x.Length);
        Add(x, scalar, result);
        return result;
    }

    /// <summary>
    /// Adds a scalar to every element of a vector and writes the result to <paramref name="destination"/>.
    /// <para>ベクトルの全要素にスカラーを加算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="scalar">The scalar value to add. / 加算するスカラー値。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    public static VectorSpan<T> Add<T>(ReadOnlyVectorSpan<T> x, T scalar, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Map(x.AsMatrix(), destination.AsMatrix(), new AddScalarOperator<T>(scalar));
        return destination;
    }

    /// <summary>
    /// Adds a scalar to every element of a matrix (<c>x + scalar</c>).
    /// <para>行列の全要素にスカラーを加算します（<c>x + scalar</c>）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="scalar">The scalar value to add. / 加算するスカラー値。</param>
    /// <returns>A new matrix containing <c>x + scalar</c>. / 計算結果を格納した新しい行列。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    public static Matrix<T> Add<T>(ReadOnlyMatrixSpan<T> x, T scalar)
        where T : unmanaged, INumberBase<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        Add(x, scalar, result);
        return result;
    }

    /// <summary>
    /// Adds a scalar to every element of a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の全要素にスカラーを加算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="scalar">The scalar value to add. / 加算するスカラー値。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    public static MatrixSpan<T> Add<T>(ReadOnlyMatrixSpan<T> x, T scalar, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Map(x, destination, new AddScalarOperator<T>(scalar));
        return destination;
    }

    /// <summary>
    /// Subtracts a scalar from every element of a vector (<c>x - scalar</c>).
    /// <para>ベクトルの全要素からスカラーを減算します（<c>x - scalar</c>）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="scalar">The scalar value to subtract. / 減算するスカラー値。</param>
    /// <returns>A new vector containing <c>x - scalar</c>. / 計算結果を格納した新しいベクトル。</returns>
    public static Vector<T> Subtract<T>(ReadOnlyVectorSpan<T> x, T scalar)
        where T : unmanaged, INumberBase<T>
    {
        var result = Vector<T>.Uninitialized(x.Length);
        Subtract(x, scalar, result);
        return result;
    }

    /// <summary>
    /// Subtracts a scalar from every element of a vector and writes the result to <paramref name="destination"/>.
    /// <para>ベクトルの全要素からスカラーを減算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="scalar">The scalar value to subtract. / 減算するスカラー値。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    public static VectorSpan<T> Subtract<T>(ReadOnlyVectorSpan<T> x, T scalar, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Map(x.AsMatrix(), destination.AsMatrix(), new SubtractScalarOperator<T>(scalar));
        return destination;
    }

    /// <summary>
    /// Subtracts a scalar from every element of a matrix (<c>x - scalar</c>).
    /// <para>行列の全要素からスカラーを減算します（<c>x - scalar</c>）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="scalar">The scalar value to subtract. / 減算するスカラー値。</param>
    /// <returns>A new matrix containing <c>x - scalar</c>. / 計算結果を格納した新しい行列。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    public static Matrix<T> Subtract<T>(ReadOnlyMatrixSpan<T> x, T scalar)
        where T : unmanaged, INumberBase<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        Subtract(x, scalar, result);
        return result;
    }

    /// <summary>
    /// Subtracts a scalar from every element of a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の全要素からスカラーを減算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="scalar">The scalar value to subtract. / 減算するスカラー値。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    public static MatrixSpan<T> Subtract<T>(ReadOnlyMatrixSpan<T> x, T scalar, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Map(x, destination, new SubtractScalarOperator<T>(scalar));
        return destination;
    }

    /// <summary>
    /// Applies elementwise soft thresholding <c>sign(x) · max(|x| - threshold, 0)</c> to a vector.
    /// <para>ベクトルに対して要素ごとの軟判定しきい値 <c>sign(x) · max(|x| - threshold, 0)</c> を適用します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="threshold">The non-negative threshold value. / 0 以上の軟判定しきい値。</param>
    /// <returns>A new vector containing the soft-thresholded values. / 軟判定しきい値適用後の新しいベクトル。</returns>
    /// <remarks>
    /// <para>Proximal operator of the L1 norm: shrinks elements toward zero by <paramref name="threshold"/>, setting entries with <c>|x| ≤ threshold</c> to zero and propagating <c>NaN</c>.</para>
    /// <para>L1 正則化の近接写像です。<c>|x| ≤ threshold</c> の要素を 0 とし、それ以外の要素を 0 方向へ <paramref name="threshold"/> だけ縮小します（<c>NaN</c> はそのまま伝播します）。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="threshold"/> is negative or <c>NaN</c>. / <paramref name="threshold"/> が負または <c>NaN</c> の場合。</exception>
    public static Vector<T> SoftThreshold<T>(ReadOnlyVectorSpan<T> x, T threshold)
        where T : unmanaged, INumber<T>
    {
        var result = Vector<T>.Uninitialized(x.Length);
        SoftThreshold(x, threshold, result);
        return result;
    }

    /// <summary>
    /// Applies elementwise soft thresholding to a vector and writes the result to <paramref name="destination"/>.
    /// <para>ベクトルに対して要素ごとの軟判定しきい値を適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="threshold">The non-negative threshold value. / 0 以上の軟判定しきい値。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="threshold"/> is negative or <c>NaN</c>. / <paramref name="threshold"/> が負または <c>NaN</c> の場合。</exception>
    public static VectorSpan<T> SoftThreshold<T>(ReadOnlyVectorSpan<T> x, T threshold, VectorSpan<T> destination)
        where T : unmanaged, INumber<T>
    {
        StridedKernels.SoftThreshold(x.AsMatrix(), RequireThreshold(threshold), destination.AsMatrix());
        return destination;
    }

    /// <summary>
    /// Applies elementwise soft thresholding <c>sign(x) · max(|x| - threshold, 0)</c> to a matrix.
    /// <para>行列に対して要素ごとの軟判定しきい値 <c>sign(x) · max(|x| - threshold, 0)</c> を適用します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="threshold">The non-negative threshold value. / 0 以上の軟判定しきい値。</param>
    /// <returns>A new matrix containing the soft-thresholded values. / 軟判定しきい値適用後の新しい行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="threshold"/> is negative or <c>NaN</c>. / <paramref name="threshold"/> が負または <c>NaN</c> の場合。</exception>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    public static Matrix<T> SoftThreshold<T>(ReadOnlyMatrixSpan<T> x, T threshold)
        where T : unmanaged, INumber<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        SoftThreshold(x, threshold, result);
        return result;
    }

    /// <summary>
    /// Applies elementwise soft thresholding to a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列に対して要素ごとの軟判定しきい値を適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="threshold">The non-negative threshold value. / 0 以上の軟判定しきい値。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="threshold"/> is negative or <c>NaN</c>. / <paramref name="threshold"/> が負または <c>NaN</c> の場合。</exception>
    public static MatrixSpan<T> SoftThreshold<T>(ReadOnlyMatrixSpan<T> x, T threshold, MatrixSpan<T> destination)
        where T : unmanaged, INumber<T>
    {
        StridedKernels.SoftThreshold(x, RequireThreshold(threshold), destination);
        return destination;
    }

    // -0 は許容し、NaN は比較が偽になるため例外とする。
    private static T RequireThreshold<T>(T threshold) where T : INumber<T> =>
        threshold >= T.Zero ? threshold
        : throw new ArgumentOutOfRangeException(nameof(threshold), threshold, "The threshold must be zero or positive.");

    /// <summary>
    /// Multiplies every element of a vector by a scalar (<c>x · scalar</c>).
    /// <para>ベクトルの全要素をスカラー倍します（<c>x · scalar</c>）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="scalar">The scalar multiplier. / 乗算するスカラー値。</param>
    /// <returns>A new vector containing <c>x · scalar</c>. / 計算結果を格納した新しいベクトル。</returns>
    public static Vector<T> Multiply<T>(ReadOnlyVectorSpan<T> x, T scalar)
        where T : unmanaged, INumberBase<T>
    {
        var result = Vector<T>.Uninitialized(x.Length);
        Multiply(x, scalar, result);
        return result;
    }

    /// <summary>
    /// Multiplies every element of a vector by a scalar and writes the result to <paramref name="destination"/>.
    /// <para>ベクトルの全要素をスカラー倍し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="scalar">The scalar multiplier. / 乗算するスカラー値。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    public static VectorSpan<T> Multiply<T>(ReadOnlyVectorSpan<T> x, T scalar, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Scale(x.AsMatrix(), scalar, destination.AsMatrix());
        return destination;
    }

    /// <summary>
    /// Multiplies every element of a matrix by a scalar (<c>x · scalar</c>).
    /// <para>行列の全要素をスカラー倍します（<c>x · scalar</c>）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="scalar">The scalar multiplier. / 乗算するスカラー値。</param>
    /// <returns>A new matrix containing <c>x · scalar</c>. / 計算結果を格納した新しい行列。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    public static Matrix<T> Multiply<T>(ReadOnlyMatrixSpan<T> x, T scalar)
        where T : unmanaged, INumberBase<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        Multiply(x, scalar, result);
        return result;
    }

    /// <summary>
    /// Multiplies every element of a matrix by a scalar and writes the result to <paramref name="destination"/>.
    /// <para>行列の全要素をスカラー倍し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="scalar">The scalar multiplier. / 乗算するスカラー値。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    public static MatrixSpan<T> Multiply<T>(ReadOnlyMatrixSpan<T> x, T scalar, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Scale(x, scalar, destination);
        return destination;
    }
}
