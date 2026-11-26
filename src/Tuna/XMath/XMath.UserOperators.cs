using System.Numerics;

namespace Tuna;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

public static partial class XMath
{
    /// <summary>
    /// Applies a stateless unary operator <typeparamref name="TOperator"/> elementwise to a vector.
    /// <para>状態を持たない1引数演算子 <typeparamref name="TOperator"/> をベクトルの各要素に適用します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless unary operator type. / 状態を持たない1引数演算子の型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>A new vector containing the transformed elements. / 演算結果を格納した新しいベクトル。</returns>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static Vector<T> Map<T, TOperator>(ReadOnlyVectorSpan<T> x)
        where T : unmanaged, INumberBase<T>
        where TOperator : IUnaryOperator<T>
    {
        var result = Vector<T>.Uninitialized(x.Length);
        Map<T, TOperator>(x, result);
        return result;
    }

    /// <summary>
    /// Applies a stateless unary operator <typeparamref name="TOperator"/> elementwise to a vector and writes the result to <paramref name="destination"/>.
    /// <para>状態を持たない1引数演算子 <typeparamref name="TOperator"/> をベクトルの各要素に適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless unary operator type. / 状態を持たない1引数演算子の型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static VectorSpan<T> Map<T, TOperator>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IUnaryOperator<T>
    {
        StridedKernels.Map(x.AsMatrix(), destination.AsMatrix(), default(StaticUnaryOperator<T, TOperator>));
        return destination;
    }

    /// <summary>
    /// Applies a stateless unary operator <typeparamref name="TOperator"/> elementwise to a matrix.
    /// <para>状態を持たない1引数演算子 <typeparamref name="TOperator"/> を行列の各要素に適用します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless unary operator type. / 状態を持たない1引数演算子の型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new matrix containing the transformed elements. / 演算結果を格納した新しい行列。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static Matrix<T> Map<T, TOperator>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumberBase<T>
        where TOperator : IUnaryOperator<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        Map<T, TOperator>(x, result);
        return result;
    }

    /// <summary>
    /// Applies a stateless unary operator <typeparamref name="TOperator"/> elementwise to a matrix and writes the result to <paramref name="destination"/>.
    /// <para>状態を持たない1引数演算子 <typeparamref name="TOperator"/> を行列の各要素に適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless unary operator type. / 状態を持たない1引数演算子の型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static MatrixSpan<T> Map<T, TOperator>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IUnaryOperator<T>
    {
        StridedKernels.Map(x, destination, default(StaticUnaryOperator<T, TOperator>));
        return destination;
    }

    /// <summary>
    /// Applies a stateful unary operator <paramref name="op"/> elementwise to a vector.
    /// <para>状態を持つ1引数演算子 <paramref name="op"/> をベクトルの各要素に適用します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful unary operator type. / 状態を持つ1引数演算子の型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="op">The operator to apply. / 適用する演算子。</param>
    /// <returns>A new vector containing the transformed elements. / 演算結果を格納した新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> Map<T, TOperator>(ReadOnlyVectorSpan<T> x, TOperator op)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulUnaryOperator<T>
    {
        var result = Vector<T>.Uninitialized(x.Length);
        Map(x, op, result);
        return result;
    }

    /// <summary>
    /// Applies a stateful unary operator <paramref name="op"/> elementwise to a vector and writes the result to <paramref name="destination"/>.
    /// <para>状態を持つ1引数演算子 <paramref name="op"/> をベクトルの各要素に適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful unary operator type. / 状態を持つ1引数演算子の型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="op">The operator to apply. / 適用する演算子。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static VectorSpan<T> Map<T, TOperator>(ReadOnlyVectorSpan<T> x, TOperator op, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulUnaryOperator<T>
    {
        StridedKernels.Map(x.AsMatrix(), destination.AsMatrix(), op);
        return destination;
    }

    /// <summary>
    /// Applies a stateful unary operator <paramref name="op"/> elementwise to a matrix.
    /// <para>状態を持つ1引数演算子 <paramref name="op"/> を行列の各要素に適用します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful unary operator type. / 状態を持つ1引数演算子の型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="op">The operator to apply. / 適用する演算子。</param>
    /// <returns>A new matrix containing the transformed elements. / 演算結果を格納した新しい行列。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Matrix<T> Map<T, TOperator>(ReadOnlyMatrixSpan<T> x, TOperator op)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulUnaryOperator<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        Map(x, op, result);
        return result;
    }

    /// <summary>
    /// Applies a stateful unary operator <paramref name="op"/> elementwise to a matrix and writes the result to <paramref name="destination"/>.
    /// <para>状態を持つ1引数演算子 <paramref name="op"/> を行列の各要素に適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful unary operator type. / 状態を持つ1引数演算子の型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="op">The operator to apply. / 適用する演算子。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static MatrixSpan<T> Map<T, TOperator>(ReadOnlyMatrixSpan<T> x, TOperator op, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulUnaryOperator<T>
    {
        StridedKernels.Map(x, destination, op);
        return destination;
    }

    /// <summary>
    /// Applies a stateless binary operator <typeparamref name="TOperator"/> elementwise to two vectors (<c>result[i] = TOperator(x[i], y[i])</c>).
    /// <para>状態を持たない2引数演算子 <typeparamref name="TOperator"/> を2つのベクトルの対応する要素ごとに適用します（<c>result[i] = TOperator(x[i], y[i])</c>）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless binary operator type. / 状態を持たない2引数演算子の型。</typeparam>
    /// <param name="x">The first input vector. / 第1入力ベクトル。</param>
    /// <param name="y">The second input vector. / 第2入力ベクトル。</param>
    /// <returns>A new vector containing the computed elements. / 演算結果を格納した新しいベクトル。</returns>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static Vector<T> Zip<T, TOperator>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y)
        where T : unmanaged, INumberBase<T>
        where TOperator : IBinaryOperator<T>
    {
        var result = Vector<T>.Uninitialized(x.Length);
        Zip<T, TOperator>(x, y, result);
        return result;
    }

    /// <summary>
    /// Applies a stateless binary operator <typeparamref name="TOperator"/> elementwise to two vectors and writes the result to <paramref name="destination"/>.
    /// <para>状態を持たない2引数演算子 <typeparamref name="TOperator"/> を2つのベクトルの対応する要素ごとに適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless binary operator type. / 状態を持たない2引数演算子の型。</typeparam>
    /// <param name="x">The first input vector. / 第1入力ベクトル。</param>
    /// <param name="y">The second input vector. / 第2入力ベクトル。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> may identically alias an input (in-place); partial overlap throws.</para>
    /// <para><paramref name="destination"/> は入力と完全に同一の領域（インプレース）を指定できます。部分的な重なりは例外となります。</para>
    /// </remarks>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static VectorSpan<T> Zip<T, TOperator>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IBinaryOperator<T>
    {
        StridedKernels.Zip(x.AsMatrix(), y.AsMatrix(), destination.AsMatrix(), default(StaticBinaryOperator<T, TOperator>));
        return destination;
    }

    /// <summary>
    /// Applies a stateless binary operator <typeparamref name="TOperator"/> elementwise to two matrices (<c>result[i, j] = TOperator(x[i, j], y[i, j])</c>).
    /// <para>状態を持たない2引数演算子 <typeparamref name="TOperator"/> を2つの行列の対応する要素ごとに適用します（<c>result[i, j] = TOperator(x[i, j], y[i, j])</c>）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless binary operator type. / 状態を持たない2引数演算子の型。</typeparam>
    /// <param name="x">The first input matrix. / 第1入力行列。</param>
    /// <param name="y">The second input matrix. / 第2入力行列。</param>
    /// <returns>A new matrix containing the computed elements. / 演算結果を格納した新しい行列。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static Matrix<T> Zip<T, TOperator>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y)
        where T : unmanaged, INumberBase<T>
        where TOperator : IBinaryOperator<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        Zip<T, TOperator>(x, y, result);
        return result;
    }

    /// <summary>
    /// Applies a stateless binary operator <typeparamref name="TOperator"/> elementwise to two matrices and writes the result to <paramref name="destination"/>.
    /// <para>状態を持たない2引数演算子 <typeparamref name="TOperator"/> を2つの行列の対応する要素ごとに適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless binary operator type. / 状態を持たない2引数演算子の型。</typeparam>
    /// <param name="x">The first input matrix. / 第1入力行列。</param>
    /// <param name="y">The second input matrix. / 第2入力行列。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> may identically alias an input (in-place); partial overlap throws.</para>
    /// <para><paramref name="destination"/> は入力と完全に同一の領域（インプレース）を指定できます。部分的な重なりは例外となります。</para>
    /// </remarks>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static MatrixSpan<T> Zip<T, TOperator>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IBinaryOperator<T>
    {
        StridedKernels.Zip(x, y, destination, default(StaticBinaryOperator<T, TOperator>));
        return destination;
    }

    /// <summary>
    /// Applies a stateful binary operator <paramref name="op"/> elementwise to two vectors (<c>result[i] = op(x[i], y[i])</c>).
    /// <para>状態を持つ2引数演算子 <paramref name="op"/> を2つのベクトルの対応する要素ごとに適用します（<c>result[i] = op(x[i], y[i])</c>）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful binary operator type. / 状態を持つ2引数演算子の型。</typeparam>
    /// <param name="x">The first input vector. / 第1入力ベクトル。</param>
    /// <param name="y">The second input vector. / 第2入力ベクトル。</param>
    /// <param name="op">The operator to apply. / 適用する演算子。</param>
    /// <returns>A new vector containing the computed elements. / 演算結果を格納した新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> Zip<T, TOperator>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, TOperator op)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulBinaryOperator<T>
    {
        var result = Vector<T>.Uninitialized(x.Length);
        Zip(x, y, op, result);
        return result;
    }

    /// <summary>
    /// Applies a stateful binary operator <paramref name="op"/> elementwise to two vectors and writes the result to <paramref name="destination"/>.
    /// <para>状態を持つ2引数演算子 <paramref name="op"/> を2つのベクトルの対応する要素ごとに適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful binary operator type. / 状態を持つ2引数演算子の型。</typeparam>
    /// <param name="x">The first input vector. / 第1入力ベクトル。</param>
    /// <param name="y">The second input vector. / 第2入力ベクトル。</param>
    /// <param name="op">The operator to apply. / 適用する演算子。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> may identically alias an input (in-place); partial overlap throws.</para>
    /// <para><paramref name="destination"/> は入力と完全に同一の領域（インプレース）を指定できます。部分的な重なりは例外となります。</para>
    /// </remarks>
    [InferenceOverloads]
    public static VectorSpan<T> Zip<T, TOperator>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, TOperator op, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulBinaryOperator<T>
    {
        StridedKernels.Zip(x.AsMatrix(), y.AsMatrix(), destination.AsMatrix(), op);
        return destination;
    }

    /// <summary>
    /// Applies a stateful binary operator <paramref name="op"/> elementwise to two matrices (<c>result[i, j] = op(x[i, j], y[i, j])</c>).
    /// <para>状態を持つ2引数演算子 <paramref name="op"/> を2つの行列の対応する要素ごとに適用します（<c>result[i, j] = op(x[i, j], y[i, j])</c>）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful binary operator type. / 状態を持つ2引数演算子の型。</typeparam>
    /// <param name="x">The first input matrix. / 第1入力行列。</param>
    /// <param name="y">The second input matrix. / 第2入力行列。</param>
    /// <param name="op">The operator to apply. / 適用する演算子。</param>
    /// <returns>A new matrix containing the computed elements. / 演算結果を格納した新しい行列。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Matrix<T> Zip<T, TOperator>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, TOperator op)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulBinaryOperator<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        Zip(x, y, op, result);
        return result;
    }

    /// <summary>
    /// Applies a stateful binary operator <paramref name="op"/> elementwise to two matrices and writes the result to <paramref name="destination"/>.
    /// <para>状態を持つ2引数演算子 <paramref name="op"/> を2つの行列の対応する要素ごとに適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful binary operator type. / 状態を持つ2引数演算子の型。</typeparam>
    /// <param name="x">The first input matrix. / 第1入力行列。</param>
    /// <param name="y">The second input matrix. / 第2入力行列。</param>
    /// <param name="op">The operator to apply. / 適用する演算子。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> may identically alias an input (in-place); partial overlap throws.</para>
    /// <para><paramref name="destination"/> は入力と完全に同一の領域（インプレース）を指定できます。部分的な重なりは例外となります。</para>
    /// </remarks>
    [InferenceOverloads]
    public static MatrixSpan<T> Zip<T, TOperator>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, TOperator op, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulBinaryOperator<T>
    {
        StridedKernels.Zip(x, y, destination, op);
        return destination;
    }

    /// <summary>
    /// Applies a stateless ternary operator <typeparamref name="TOperator"/> elementwise to three vectors (<c>result[i] = TOperator(x[i], y[i], z[i])</c>).
    /// <para>状態を持たない3引数演算子 <typeparamref name="TOperator"/> を3つのベクトルの対応する要素ごとに適用します（<c>result[i] = TOperator(x[i], y[i], z[i])</c>）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless ternary operator type. / 状態を持たない3引数演算子の型。</typeparam>
    /// <param name="x">The first input vector. / 第1入力ベクトル。</param>
    /// <param name="y">The second input vector. / 第2入力ベクトル。</param>
    /// <param name="z">The third input vector. / 第3入力ベクトル。</param>
    /// <returns>A new vector containing the computed elements. / 演算結果を格納した新しいベクトル。</returns>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static Vector<T> Zip<T, TOperator>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, ReadOnlyVectorSpan<T> z)
        where T : unmanaged, INumberBase<T>
        where TOperator : ITernaryOperator<T>
    {
        var result = Vector<T>.Uninitialized(x.Length);
        Zip<T, TOperator>(x, y, z, result);
        return result;
    }

    /// <summary>
    /// Applies a stateless ternary operator <typeparamref name="TOperator"/> elementwise to three vectors and writes the result to <paramref name="destination"/>.
    /// <para>状態を持たない3引数演算子 <typeparamref name="TOperator"/> を3つのベクトルの対応する要素ごとに適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless ternary operator type. / 状態を持たない3引数演算子の型。</typeparam>
    /// <param name="x">The first input vector. / 第1入力ベクトル。</param>
    /// <param name="y">The second input vector. / 第2入力ベクトル。</param>
    /// <param name="z">The third input vector. / 第3入力ベクトル。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> may identically alias an input (in-place); partial overlap throws.</para>
    /// <para><paramref name="destination"/> は入力と完全に同一の領域（インプレース）を指定できます。部分的な重なりは例外となります。</para>
    /// </remarks>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static VectorSpan<T> Zip<T, TOperator>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, ReadOnlyVectorSpan<T> z, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : ITernaryOperator<T>
    {
        StridedKernels.Zip(x.AsMatrix(), y.AsMatrix(), z.AsMatrix(), destination.AsMatrix(), default(StaticTernaryOperator<T, TOperator>));
        return destination;
    }

    /// <summary>
    /// Applies a stateless ternary operator <typeparamref name="TOperator"/> elementwise to three matrices (<c>result[i, j] = TOperator(x[i, j], y[i, j], z[i, j])</c>).
    /// <para>状態を持たない3引数演算子 <typeparamref name="TOperator"/> を3つの行列の対応する要素ごとに適用します（<c>result[i, j] = TOperator(x[i, j], y[i, j], z[i, j])</c>）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless ternary operator type. / 状態を持たない3引数演算子の型。</typeparam>
    /// <param name="x">The first input matrix. / 第1入力行列。</param>
    /// <param name="y">The second input matrix. / 第2入力行列。</param>
    /// <param name="z">The third input matrix. / 第3入力行列。</param>
    /// <returns>A new matrix containing the computed elements. / 演算結果を格納した新しい行列。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static Matrix<T> Zip<T, TOperator>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, ReadOnlyMatrixSpan<T> z)
        where T : unmanaged, INumberBase<T>
        where TOperator : ITernaryOperator<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        Zip<T, TOperator>(x, y, z, result);
        return result;
    }

    /// <summary>
    /// Applies a stateless ternary operator <typeparamref name="TOperator"/> elementwise to three matrices and writes the result to <paramref name="destination"/>.
    /// <para>状態を持たない3引数演算子 <typeparamref name="TOperator"/> を3つの行列の対応する要素ごとに適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless ternary operator type. / 状態を持たない3引数演算子の型。</typeparam>
    /// <param name="x">The first input matrix. / 第1入力行列。</param>
    /// <param name="y">The second input matrix. / 第2入力行列。</param>
    /// <param name="z">The third input matrix. / 第3入力行列。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> may identically alias an input (in-place); partial overlap throws.</para>
    /// <para><paramref name="destination"/> は入力と完全に同一の領域（インプレース）を指定できます。部分的な重なりは例外となります。</para>
    /// </remarks>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static MatrixSpan<T> Zip<T, TOperator>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, ReadOnlyMatrixSpan<T> z, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : ITernaryOperator<T>
    {
        StridedKernels.Zip(x, y, z, destination, default(StaticTernaryOperator<T, TOperator>));
        return destination;
    }

    /// <summary>
    /// Applies a stateful ternary operator <paramref name="op"/> elementwise to three vectors (<c>result[i] = op(x[i], y[i], z[i])</c>).
    /// <para>状態を持つ3引数演算子 <paramref name="op"/> を3つのベクトルの対応する要素ごとに適用します（<c>result[i] = op(x[i], y[i], z[i])</c>）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful ternary operator type. / 状態を持つ3引数演算子の型。</typeparam>
    /// <param name="x">The first input vector. / 第1入力ベクトル。</param>
    /// <param name="y">The second input vector. / 第2入力ベクトル。</param>
    /// <param name="z">The third input vector. / 第3入力ベクトル。</param>
    /// <param name="op">The operator to apply. / 適用する演算子。</param>
    /// <returns>A new vector containing the computed elements. / 演算結果を格納した新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> Zip<T, TOperator>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, ReadOnlyVectorSpan<T> z, TOperator op)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulTernaryOperator<T>
    {
        var result = Vector<T>.Uninitialized(x.Length);
        Zip(x, y, z, op, result);
        return result;
    }

    /// <summary>
    /// Applies a stateful ternary operator <paramref name="op"/> elementwise to three vectors and writes the result to <paramref name="destination"/>.
    /// <para>状態を持つ3引数演算子 <paramref name="op"/> を3つのベクトルの対応する要素ごとに適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful ternary operator type. / 状態を持つ3引数演算子の型。</typeparam>
    /// <param name="x">The first input vector. / 第1入力ベクトル。</param>
    /// <param name="y">The second input vector. / 第2入力ベクトル。</param>
    /// <param name="z">The third input vector. / 第3入力ベクトル。</param>
    /// <param name="op">The operator to apply. / 適用する演算子。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> may identically alias an input (in-place); partial overlap throws.</para>
    /// <para><paramref name="destination"/> は入力と完全に同一の領域（インプレース）を指定できます。部分的な重なりは例外となります。</para>
    /// </remarks>
    [InferenceOverloads]
    public static VectorSpan<T> Zip<T, TOperator>(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y, ReadOnlyVectorSpan<T> z, TOperator op, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulTernaryOperator<T>
    {
        StridedKernels.Zip(x.AsMatrix(), y.AsMatrix(), z.AsMatrix(), destination.AsMatrix(), op);
        return destination;
    }

    /// <summary>
    /// Applies a stateful ternary operator <paramref name="op"/> elementwise to three matrices (<c>result[i, j] = op(x[i, j], y[i, j], z[i, j])</c>).
    /// <para>状態を持つ3引数演算子 <paramref name="op"/> を3つの行列の対応する要素ごとに適用します（<c>result[i, j] = op(x[i, j], y[i, j], z[i, j])</c>）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful ternary operator type. / 状態を持つ3引数演算子の型。</typeparam>
    /// <param name="x">The first input matrix. / 第1入力行列。</param>
    /// <param name="y">The second input matrix. / 第2入力行列。</param>
    /// <param name="z">The third input matrix. / 第3入力行列。</param>
    /// <param name="op">The operator to apply. / 適用する演算子。</param>
    /// <returns>A new matrix containing the computed elements. / 演算結果を格納した新しい行列。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Matrix<T> Zip<T, TOperator>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, ReadOnlyMatrixSpan<T> z, TOperator op)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulTernaryOperator<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        Zip(x, y, z, op, result);
        return result;
    }

    /// <summary>
    /// Applies a stateful ternary operator <paramref name="op"/> elementwise to three matrices and writes the result to <paramref name="destination"/>.
    /// <para>状態を持つ3引数演算子 <paramref name="op"/> を3つの行列の対応する要素ごとに適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful ternary operator type. / 状態を持つ3引数演算子の型。</typeparam>
    /// <param name="x">The first input matrix. / 第1入力行列。</param>
    /// <param name="y">The second input matrix. / 第2入力行列。</param>
    /// <param name="z">The third input matrix. / 第3入力行列。</param>
    /// <param name="op">The operator to apply. / 適用する演算子。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> may identically alias an input (in-place); partial overlap throws.</para>
    /// <para><paramref name="destination"/> は入力と完全に同一の領域（インプレース）を指定できます。部分的な重なりは例外となります。</para>
    /// </remarks>
    [InferenceOverloads]
    public static MatrixSpan<T> Zip<T, TOperator>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, ReadOnlyMatrixSpan<T> z, TOperator op, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulTernaryOperator<T>
    {
        StridedKernels.Zip(x, y, z, destination, op);
        return destination;
    }

    /// <summary>
    /// Applies a stateless pairwise operator to every pair of columns of <paramref name="x"/> and <paramref name="y"/> (<c>result[i, j] = TOperator(x.Column(i), y.Column(j))</c>).
    /// <para>状態を持たないペア演算子を <paramref name="x"/> と <paramref name="y"/> のすべての列ペアに適用します（<c>result[i, j] = TOperator(x.Column(i), y.Column(j))</c>）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless pairwise operator type. / 状態を持たないペア演算子の型。</typeparam>
    /// <param name="x">The first input matrix (each column is treated as a point). / 第1入力行列（各列が1点）。</param>
    /// <param name="y">The second input matrix (each column is treated as a point). / 第2入力行列（各列が1点）。</param>
    /// <returns>A new <c>x.Columns × y.Columns</c> matrix. / <c>x.Columns × y.Columns</c> の新しい行列。</returns>
    /// <remarks>
    /// <para>If points are stored in rows, pass transposed views (such as <c>x.T</c>). For symmetric self-pairs, prefer <see cref="PairwiseSymmetric{T, TOperator}(ReadOnlyMatrixSpan{T})"/>.</para>
    /// <para>点が行ごとに並んでいる場合は転置ビュー（<c>x.T</c> など）を渡してください。同一集合に対する対称演算には <see cref="PairwiseSymmetric{T, TOperator}(ReadOnlyMatrixSpan{T})"/> を使用できます。</para>
    /// </remarks>
    /// <exception cref="OverflowException">The element count <c>x.Columns · y.Columns</c> exceeds <see cref="int.MaxValue"/>. / 結果の要素数 <c>x.Columns · y.Columns</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static Matrix<T> Pairwise<T, TOperator>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y)
        where T : unmanaged, INumberBase<T>
        where TOperator : IPairwiseOperator<T>
    {
        var result = Matrix<T>.Uninitialized(x.Columns, y.Columns);
        Pairwise<T, TOperator>(x, y, result);
        return result;
    }

    /// <summary>
    /// Applies a stateless pairwise operator to every pair of columns of <paramref name="x"/> and <paramref name="y"/> and writes the result to <paramref name="destination"/>.
    /// <para>状態を持たないペア演算子を <paramref name="x"/> と <paramref name="y"/> のすべての列ペアに適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless pairwise operator type. / 状態を持たないペア演算子の型。</typeparam>
    /// <param name="x">The first input matrix (each column is a point). / 第1入力行列（各列が1点）。</param>
    /// <param name="y">The second input matrix (each column is a point). / 第2入力行列（各列が1点）。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static MatrixSpan<T> Pairwise<T, TOperator>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IPairwiseOperator<T>
    {
        StridedKernels.Pairwise(x, y, destination, default(StaticPairwiseOperator<T, TOperator>));
        return destination;
    }

    /// <summary>
    /// Applies a stateful pairwise operator <paramref name="op"/> to every pair of columns of <paramref name="x"/> and <paramref name="y"/> (<c>result[i, j] = op(x.Column(i), y.Column(j))</c>).
    /// <para>状態を持つペア演算子 <paramref name="op"/> を <paramref name="x"/> と <paramref name="y"/> のすべての列ペアに適用します（<c>result[i, j] = op(x.Column(i), y.Column(j))</c>）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful pairwise operator type. / 状態を持つペア演算子の型。</typeparam>
    /// <param name="x">The first input matrix (each column is a point). / 第1入力行列（各列が1点）。</param>
    /// <param name="y">The second input matrix (each column is a point). / 第2入力行列（各列が1点）。</param>
    /// <param name="op">The operator to apply. / 適用する演算子。</param>
    /// <returns>A new <c>x.Columns × y.Columns</c> matrix. / <c>x.Columns × y.Columns</c> の新しい行列。</returns>
    /// <remarks>
    /// <para>If points are stored in rows, pass transposed views (such as <c>x.T</c>).</para>
    /// <para>点が行ごとに並んでいる場合は転置ビュー（<c>x.T</c> など）を渡してください。</para>
    /// </remarks>
    /// <exception cref="OverflowException">The element count <c>x.Columns · y.Columns</c> exceeds <see cref="int.MaxValue"/>. / 結果の要素数 <c>x.Columns · y.Columns</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    [InferenceOverloads]
    public static Matrix<T> Pairwise<T, TOperator>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, TOperator op)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulPairwiseOperator<T>
    {
        var result = Matrix<T>.Uninitialized(x.Columns, y.Columns);
        Pairwise(x, y, op, result);
        return result;
    }

    /// <summary>
    /// Applies a stateful pairwise operator <paramref name="op"/> to every pair of columns of <paramref name="x"/> and <paramref name="y"/> and writes the result to <paramref name="destination"/>.
    /// <para>状態を持つペア演算子 <paramref name="op"/> を <paramref name="x"/> と <paramref name="y"/> のすべての列ペアに適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful pairwise operator type. / 状態を持つペア演算子の型。</typeparam>
    /// <param name="x">The first input matrix (each column is a point). / 第1入力行列（各列が1点）。</param>
    /// <param name="y">The second input matrix (each column is a point). / 第2入力行列（各列が1点）。</param>
    /// <param name="op">The operator to apply. / 適用する演算子。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static MatrixSpan<T> Pairwise<T, TOperator>(ReadOnlyMatrixSpan<T> x, ReadOnlyMatrixSpan<T> y, TOperator op, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulPairwiseOperator<T>
    {
        StridedKernels.Pairwise(x, y, destination, op);
        return destination;
    }

    /// <summary>
    /// Applies a stateless symmetric pairwise operator to all column pairs of <paramref name="x"/>, evaluating only the lower triangle (<c>i ≥ j</c>) and mirroring to the upper triangle.
    /// <para>状態を持たない対称ペア演算子を <paramref name="x"/> の下三角（<c>i ≥ j</c>）の列ペアのみに適用し、上三角へ対称に転記して対称行列を返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless pairwise operator type. / 状態を持たないペア演算子の型。</typeparam>
    /// <param name="x">The input matrix (each column is a point). / 入力行列（各列が1点）。</param>
    /// <returns>A new symmetric <c>x.Columns × x.Columns</c> matrix. / <c>x.Columns × x.Columns</c> の新しい対称行列。</returns>
    /// <remarks>
    /// <para>Intended for symmetric operators (<c>op(a, b) = op(b, a)</c>) such as distance or Gram matrices; the output is always symmetric because the lower triangle is mirrored.</para>
    /// <para>距離行列やグラム行列などの対称演算（<c>op(a, b) = op(b, a)</c>）向けです。下三角を上三角へ転記するため、結果は常に対称になります。</para>
    /// </remarks>
    /// <exception cref="OverflowException">The element count <c>x.Columns²</c> exceeds <see cref="int.MaxValue"/>. / 結果の要素数 <c>x.Columns²</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static Matrix<T> PairwiseSymmetric<T, TOperator>(ReadOnlyMatrixSpan<T> x)
        where T : unmanaged, INumberBase<T>
        where TOperator : IPairwiseOperator<T>
    {
        var result = Matrix<T>.Uninitialized(x.Columns, x.Columns);
        PairwiseSymmetric<T, TOperator>(x, result);
        return result;
    }

    /// <summary>
    /// Applies a stateless symmetric pairwise operator to all column pairs of <paramref name="x"/> and writes the full symmetric matrix to <paramref name="destination"/>.
    /// <para>状態を持たない対称ペア演算子を <paramref name="x"/> の列ペアに適用し、対称行列全体を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless pairwise operator type. / 状態を持たないペア演算子の型。</typeparam>
    /// <param name="x">The input matrix (each column is a point). / 入力行列（各列が1点）。</param>
    /// <param name="destination">The destination span to receive the symmetric matrix. / 対称行列を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>Writes all entries of <paramref name="destination"/> by mirroring the lower triangle to the upper triangle. <paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para>下三角を上三角へ転記して全要素を書き込みます。<paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static MatrixSpan<T> PairwiseSymmetric<T, TOperator>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IPairwiseOperator<T>
    {
        StridedKernels.Pairwise(x, x, destination, default(StaticPairwiseOperator<T, TOperator>), symmetric: true);
        return destination;
    }

    /// <summary>
    /// Applies a stateful symmetric pairwise operator <paramref name="op"/> to all column pairs of <paramref name="x"/>, evaluating only the lower triangle (<c>i ≥ j</c>) and mirroring to the upper triangle.
    /// <para>状態を持つ対称ペア演算子 <paramref name="op"/> を <paramref name="x"/> の下三角（<c>i ≥ j</c>）の列ペアのみに適用し、上三角へ対称に転記して対称行列を返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful pairwise operator type. / 状態を持つペア演算子の型。</typeparam>
    /// <param name="x">The input matrix (each column is a point). / 入力行列（各列が1点）。</param>
    /// <param name="op">The operator to apply. / 適用する演算子。</param>
    /// <returns>A new symmetric <c>x.Columns × x.Columns</c> matrix. / <c>x.Columns × x.Columns</c> の新しい対称行列。</returns>
    /// <exception cref="OverflowException">The element count <c>x.Columns²</c> exceeds <see cref="int.MaxValue"/>. / 結果の要素数 <c>x.Columns²</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    [InferenceOverloads]
    public static Matrix<T> PairwiseSymmetric<T, TOperator>(ReadOnlyMatrixSpan<T> x, TOperator op)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulPairwiseOperator<T>
    {
        var result = Matrix<T>.Uninitialized(x.Columns, x.Columns);
        PairwiseSymmetric(x, op, result);
        return result;
    }

    /// <summary>
    /// Applies a stateful symmetric pairwise operator <paramref name="op"/> to all column pairs of <paramref name="x"/> and writes the full symmetric matrix to <paramref name="destination"/>.
    /// <para>状態を持つ対称ペア演算子 <paramref name="op"/> を <paramref name="x"/> の列ペアに適用し、対称行列全体を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful pairwise operator type. / 状態を持つペア演算子の型。</typeparam>
    /// <param name="x">The input matrix (each column is a point). / 入力行列（各列が1点）。</param>
    /// <param name="op">The operator to apply. / 適用する演算子。</param>
    /// <param name="destination">The destination span to receive the symmetric matrix. / 対称行列を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>Writes all entries of <paramref name="destination"/> by mirroring the lower triangle to the upper triangle. <paramref name="destination"/> must not overlap <paramref name="x"/>.</para>
    /// <para>下三角を上三角へ転記して全要素を書き込みます。<paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません。</para>
    /// </remarks>
    [InferenceOverloads]
    public static MatrixSpan<T> PairwiseSymmetric<T, TOperator>(ReadOnlyMatrixSpan<T> x, TOperator op, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulPairwiseOperator<T>
    {
        StridedKernels.Pairwise(x, x, destination, op, symmetric: true);
        return destination;
    }
}
