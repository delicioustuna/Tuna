using System.Numerics;

namespace Tuna;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

public static partial class XMath
{
    /// <summary>
    /// Sums a tensor over its matrix index axis.
    /// <para>テンソルの行列番号の軸に沿って総和を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input tensor. / 入力テンソル。</param>
    /// <returns>A new matrix containing <c>Σ_k x[i, j, k]</c> at each position, or zeros for an empty batch. / 各位置に <c>Σ_k x[i, j, k]</c> を格納した新しい行列。バッチが空の場合は全要素が 0。</returns>
    [InferenceOverloads]
    public static Matrix<T> SumMatrices<T>(ReadOnlyTensorSpan<T> x)
        where T : unmanaged, INumberBase<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        SumMatrices(x, result);
        return result;
    }

    /// <summary>
    /// Sums a tensor over its matrix index axis and writes the result to the destination.
    /// <para>テンソルの行列番号の軸に沿う総和を出力先に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input tensor. / 入力テンソル。</param>
    /// <param name="destination">The destination with the same shape as each input matrix. / 入力の各行列と同じ形状の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>Uses the same summation order as <see cref="Sum{T}(ReadOnlyVectorSpan{T})"/> and writes zero at each position for an empty batch.</para>
    /// <para><see cref="Sum{T}(ReadOnlyVectorSpan{T})"/> と同じ順序で加算します。バッチが空の場合は出力先の全要素に 0 を書き込みます。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The destination has a different matrix shape or overlaps the input. / 出力先の形状が入力の行列と異なるか、出力先が入力と重なる場合。</exception>
    [InferenceOverloads]
    public static MatrixSpan<T> SumMatrices<T>(ReadOnlyTensorSpan<T> x, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.SumMatrices(x, destination);
        return destination;
    }

    /// <summary>
    /// Computes the elementwise sum <c>X[k] + Y[k]</c> of two tensors.
    /// <para>2つのテンソルの要素ごとの和 <c>X[k] + Y[k]</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The first input tensor. / 第1入力テンソル。</param>
    /// <param name="y">The second input tensor. / 第2入力テンソル。</param>
    /// <returns>A new tensor containing the elementwise sum. / 要素ごとの和を格納した新しいテンソル。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Tensor<T> Add<T>(ReadOnlyTensorSpan<T> x, ReadOnlyTensorSpan<T> y)
        where T : unmanaged, INumberBase<T>
    {
        var result = Tensor<T>.Uninitialized(x.Rows, x.Columns, x.MatrixCount);
        Add(x, y, result);
        return result;
    }

    /// <summary>
    /// Computes the elementwise sum <c>X[k] + Y[k]</c> of two tensors and writes the result to <paramref name="destination"/>.
    /// <para>2つのテンソルの要素ごとの和 <c>X[k] + Y[k]</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The first input tensor. / 第1入力テンソル。</param>
    /// <param name="y">The second input tensor. / 第2入力テンソル。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> may identically alias an input (in-place); partial overlap throws.</para>
    /// <para><paramref name="destination"/> は入力と完全に同一の領域（インプレース）を指定できます。部分的な重なりは例外となります。</para>
    /// </remarks>
    [InferenceOverloads]
    public static TensorSpan<T> Add<T>(ReadOnlyTensorSpan<T> x, ReadOnlyTensorSpan<T> y, TensorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Zip(x, y, destination, default(AddOperator<T>));
        return destination;
    }

    /// <summary>
    /// Computes the elementwise difference <c>X[k] - Y[k]</c> of two tensors.
    /// <para>2つのテンソルの要素ごとの差 <c>X[k] - Y[k]</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The minuend tensor. / 被減数テンソル。</param>
    /// <param name="y">The subtrahend tensor. / 減数テンソル。</param>
    /// <returns>A new tensor containing the elementwise difference. / 要素ごとの差を格納した新しいテンソル。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Tensor<T> Subtract<T>(ReadOnlyTensorSpan<T> x, ReadOnlyTensorSpan<T> y)
        where T : unmanaged, INumberBase<T>
    {
        var result = Tensor<T>.Uninitialized(x.Rows, x.Columns, x.MatrixCount);
        Subtract(x, y, result);
        return result;
    }

    /// <summary>
    /// Computes the elementwise difference <c>X[k] - Y[k]</c> of two tensors and writes the result to <paramref name="destination"/>.
    /// <para>2つのテンソルの要素ごとの差 <c>X[k] - Y[k]</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The minuend tensor. / 被減数テンソル。</param>
    /// <param name="y">The subtrahend tensor. / 減数テンソル。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> may identically alias an input (in-place); partial overlap throws.</para>
    /// <para><paramref name="destination"/> は入力と完全に同一の領域（インプレース）を指定できます。部分的な重なりは例外となります。</para>
    /// </remarks>
    [InferenceOverloads]
    public static TensorSpan<T> Subtract<T>(ReadOnlyTensorSpan<T> x, ReadOnlyTensorSpan<T> y, TensorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Zip(x, y, destination, default(SubtractOperator<T>));
        return destination;
    }

    /// <summary>
    /// Computes the elementwise (Hadamard) product <c>X[k] ∘ Y[k]</c> of two tensors.
    /// <para>2つのテンソルの要素ごとの積（アダマール積）<c>X[k] ∘ Y[k]</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The first input tensor. / 第1入力テンソル。</param>
    /// <param name="y">The second input tensor. / 第2入力テンソル。</param>
    /// <returns>A new tensor containing the elementwise product. / アダマール積を格納した新しいテンソル。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Tensor<T> Hadamard<T>(ReadOnlyTensorSpan<T> x, ReadOnlyTensorSpan<T> y)
        where T : unmanaged, INumberBase<T>
    {
        var result = Tensor<T>.Uninitialized(x.Rows, x.Columns, x.MatrixCount);
        Hadamard(x, y, result);
        return result;
    }

    /// <summary>
    /// Computes the elementwise (Hadamard) product <c>X[k] ∘ Y[k]</c> of two tensors and writes the result to <paramref name="destination"/>.
    /// <para>2つのテンソルの要素ごとの積（アダマール積）<c>X[k] ∘ Y[k]</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The first input tensor. / 第1入力テンソル。</param>
    /// <param name="y">The second input tensor. / 第2入力テンソル。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> may identically alias an input (in-place); partial overlap throws.</para>
    /// <para><paramref name="destination"/> は入力と完全に同一の領域（インプレース）を指定できます。部分的な重なりは例外となります。</para>
    /// </remarks>
    [InferenceOverloads]
    public static TensorSpan<T> Hadamard<T>(ReadOnlyTensorSpan<T> x, ReadOnlyTensorSpan<T> y, TensorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Zip(x, y, destination, default(MultiplyOperator<T>));
        return destination;
    }

    /// <summary>
    /// Computes the elementwise negation <c>-X[k]</c> of a tensor.
    /// <para>テンソルの要素ごとの符号反転 <c>-X[k]</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input tensor. / 入力テンソル。</param>
    /// <returns>A new tensor containing <c>-x</c>. / 符号反転した新しいテンソル。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Tensor<T> Negate<T>(ReadOnlyTensorSpan<T> x)
        where T : unmanaged, INumberBase<T>
    {
        var result = Tensor<T>.Uninitialized(x.Rows, x.Columns, x.MatrixCount);
        Negate(x, result);
        return result;
    }

    /// <summary>
    /// Computes the elementwise negation <c>-X[k]</c> of a tensor and writes the result to <paramref name="destination"/>.
    /// <para>テンソルの要素ごとの符号反転 <c>-X[k]</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input tensor. / 入力テンソル。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static TensorSpan<T> Negate<T>(ReadOnlyTensorSpan<T> x, TensorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Map(x, destination, default(NegateOperator<T>));
        return destination;
    }

    /// <summary>
    /// Multiplies every element of a tensor by a scalar (<c>X[k] · scalar</c>).
    /// <para>テンソルの全要素をスカラー倍します（<c>X[k] · scalar</c>）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input tensor. / 入力テンソル。</param>
    /// <param name="scalar">The scalar multiplier. / 乗算するスカラー値。</param>
    /// <returns>A new tensor containing <c>x · scalar</c>. / 計算結果を格納した新しいテンソル。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    public static Tensor<T> Multiply<T>(ReadOnlyTensorSpan<T> x, T scalar)
        where T : unmanaged, INumberBase<T>
    {
        var result = Tensor<T>.Uninitialized(x.Rows, x.Columns, x.MatrixCount);
        Multiply(x, scalar, result);
        return result;
    }

    /// <summary>
    /// Multiplies every element of a tensor by a scalar and writes the result to <paramref name="destination"/>.
    /// <para>テンソルの全要素をスカラー倍し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input tensor. / 入力テンソル。</param>
    /// <param name="scalar">The scalar multiplier. / 乗算するスカラー値。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    public static TensorSpan<T> Multiply<T>(ReadOnlyTensorSpan<T> x, T scalar, TensorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Scale(x, scalar, destination);
        return destination;
    }

    /// <summary>
    /// Applies a stateless unary operator <typeparamref name="TOperator"/> elementwise to a tensor.
    /// <para>状態を持たない1引数演算子 <typeparamref name="TOperator"/> をテンソルの各要素に適用します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless unary operator type. / 状態を持たない1引数演算子の型。</typeparam>
    /// <param name="x">The input tensor. / 入力テンソル。</param>
    /// <returns>A new tensor containing the transformed elements. / 演算結果を格納した新しいテンソル。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static Tensor<T> Map<T, TOperator>(ReadOnlyTensorSpan<T> x)
        where T : unmanaged, INumberBase<T>
        where TOperator : IUnaryOperator<T>
    {
        var result = Tensor<T>.Uninitialized(x.Rows, x.Columns, x.MatrixCount);
        Map<T, TOperator>(x, result);
        return result;
    }

    /// <summary>
    /// Applies a stateless unary operator <typeparamref name="TOperator"/> elementwise to a tensor and writes the result to <paramref name="destination"/>.
    /// <para>状態を持たない1引数演算子 <typeparamref name="TOperator"/> をテンソルの各要素に適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless unary operator type. / 状態を持たない1引数演算子の型。</typeparam>
    /// <param name="x">The input tensor. / 入力テンソル。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static TensorSpan<T> Map<T, TOperator>(ReadOnlyTensorSpan<T> x, TensorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IUnaryOperator<T>
    {
        StridedKernels.Map(x, destination, default(StaticUnaryOperator<T, TOperator>));
        return destination;
    }

    /// <summary>
    /// Applies a stateful unary operator <paramref name="op"/> elementwise to a tensor.
    /// <para>状態を持つ1引数演算子 <paramref name="op"/> をテンソルの各要素に適用します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful unary operator type. / 状態を持つ1引数演算子の型。</typeparam>
    /// <param name="x">The input tensor. / 入力テンソル。</param>
    /// <param name="op">The operator to apply. / 適用する演算子。</param>
    /// <returns>A new tensor containing the transformed elements. / 演算結果を格納した新しいテンソル。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Tensor<T> Map<T, TOperator>(ReadOnlyTensorSpan<T> x, TOperator op)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulUnaryOperator<T>
    {
        var result = Tensor<T>.Uninitialized(x.Rows, x.Columns, x.MatrixCount);
        Map(x, op, result);
        return result;
    }

    /// <summary>
    /// Applies a stateful unary operator <paramref name="op"/> elementwise to a tensor and writes the result to <paramref name="destination"/>.
    /// <para>状態を持つ1引数演算子 <paramref name="op"/> をテンソルの各要素に適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful unary operator type. / 状態を持つ1引数演算子の型。</typeparam>
    /// <param name="x">The input tensor. / 入力テンソル。</param>
    /// <param name="op">The operator to apply. / 適用する演算子。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static TensorSpan<T> Map<T, TOperator>(ReadOnlyTensorSpan<T> x, TOperator op, TensorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulUnaryOperator<T>
    {
        StridedKernels.Map(x, destination, op);
        return destination;
    }

    /// <summary>
    /// Applies a stateless binary operator <typeparamref name="TOperator"/> elementwise to two tensors.
    /// <para>状態を持たない2引数演算子 <typeparamref name="TOperator"/> を2つのテンソルの対応する要素ごとに適用します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless binary operator type. / 状態を持たない2引数演算子の型。</typeparam>
    /// <param name="x">The first input tensor. / 第1入力テンソル。</param>
    /// <param name="y">The second input tensor. / 第2入力テンソル。</param>
    /// <returns>A new tensor containing the computed elements. / 演算結果を格納した新しいテンソル。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static Tensor<T> Zip<T, TOperator>(ReadOnlyTensorSpan<T> x, ReadOnlyTensorSpan<T> y)
        where T : unmanaged, INumberBase<T>
        where TOperator : IBinaryOperator<T>
    {
        var result = Tensor<T>.Uninitialized(x.Rows, x.Columns, x.MatrixCount);
        Zip<T, TOperator>(x, y, result);
        return result;
    }

    /// <summary>
    /// Applies a stateless binary operator <typeparamref name="TOperator"/> elementwise to two tensors and writes the result to <paramref name="destination"/>.
    /// <para>状態を持たない2引数演算子 <typeparamref name="TOperator"/> を2つのテンソルの対応する要素ごとに適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless binary operator type. / 状態を持たない2引数演算子の型。</typeparam>
    /// <param name="x">The first input tensor. / 第1入力テンソル。</param>
    /// <param name="y">The second input tensor. / 第2入力テンソル。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> may identically alias an input (in-place); partial overlap throws.</para>
    /// <para><paramref name="destination"/> は入力と完全に同一の領域（インプレース）を指定できます。部分的な重なりは例外となります。</para>
    /// </remarks>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static TensorSpan<T> Zip<T, TOperator>(ReadOnlyTensorSpan<T> x, ReadOnlyTensorSpan<T> y, TensorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IBinaryOperator<T>
    {
        StridedKernels.Zip(x, y, destination, default(StaticBinaryOperator<T, TOperator>));
        return destination;
    }

    /// <summary>
    /// Applies a stateful binary operator <paramref name="op"/> elementwise to two tensors.
    /// <para>状態を持つ2引数演算子 <paramref name="op"/> を2つのテンソルの対応する要素ごとに適用します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful binary operator type. / 状態を持つ2引数演算子の型。</typeparam>
    /// <param name="x">The first input tensor. / 第1入力テンソル。</param>
    /// <param name="y">The second input tensor. / 第2入力テンソル。</param>
    /// <param name="op">The operator to apply. / 適用する演算子。</param>
    /// <returns>A new tensor containing the computed elements. / 演算結果を格納した新しいテンソル。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Tensor<T> Zip<T, TOperator>(ReadOnlyTensorSpan<T> x, ReadOnlyTensorSpan<T> y, TOperator op)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulBinaryOperator<T>
    {
        var result = Tensor<T>.Uninitialized(x.Rows, x.Columns, x.MatrixCount);
        Zip(x, y, op, result);
        return result;
    }

    /// <summary>
    /// Applies a stateful binary operator <paramref name="op"/> elementwise to two tensors and writes the result to <paramref name="destination"/>.
    /// <para>状態を持つ2引数演算子 <paramref name="op"/> を2つのテンソルの対応する要素ごとに適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful binary operator type. / 状態を持つ2引数演算子の型。</typeparam>
    /// <param name="x">The first input tensor. / 第1入力テンソル。</param>
    /// <param name="y">The second input tensor. / 第2入力テンソル。</param>
    /// <param name="op">The operator to apply. / 適用する演算子。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> may identically alias an input (in-place); partial overlap throws.</para>
    /// <para><paramref name="destination"/> は入力と完全に同一の領域（インプレース）を指定できます。部分的な重なりは例外となります。</para>
    /// </remarks>
    [InferenceOverloads]
    public static TensorSpan<T> Zip<T, TOperator>(ReadOnlyTensorSpan<T> x, ReadOnlyTensorSpan<T> y, TOperator op, TensorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulBinaryOperator<T>
    {
        StridedKernels.Zip(x, y, destination, op);
        return destination;
    }

    /// <summary>
    /// Computes the batched matrix product <c>C[:, :, k] = A[:, :, k] · B[:, :, k]</c> for each matrix slice <c>k</c>.
    /// <para>各行列スライス <c>k</c> についてバッチ行列積 <c>C[:, :, k] = A[:, :, k] · B[:, :, k]</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="a">The left input tensor. / 左側の入力テンソル。</param>
    /// <param name="b">The right input tensor. / 右側の入力テンソル。</param>
    /// <returns>A new tensor containing the batched products (no summation over <c>k</c>). / 行列ごとの積を格納した新しいテンソル（<c>k</c> については総和しません）。</returns>
    /// <remarks>
    /// <para>
    /// Transposed views (<c>A.T</c>) are passed as GEMM transpose flags without copying.
    /// To multiply every slice by the same matrix <c>W</c>, pass <c>W.Broadcast(B.Matrices.Count)</c>; for batched matrix-vector products <c>A[k] · x[:, k]</c>, use <c>Multiply(A, X.Unsqueeze(1)).AsMatrix()</c>.
    /// </para>
    /// <para>
    /// 転置ビュー（<c>A.T</c>）はコピーせず GEMM の転置フラグとして渡されます。
    /// すべてのスライスに同じ行列 <c>W</c> を掛ける場合は <c>W.Broadcast(B.Matrices.Count)</c> を、点ごとの行列ベクトル積には <c>Multiply(A, X.Unsqueeze(1)).AsMatrix()</c> を使用してください。
    /// </para>
    /// </remarks>
    /// <exception cref="OverflowException">The result element count exceeds <see cref="int.MaxValue"/>. / 結果の要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    [InferenceOverloads]
    public static Tensor<T> Multiply<T>(ReadOnlyTensorSpan<T> a, ReadOnlyTensorSpan<T> b)
        where T : unmanaged, INumberBase<T>
    {
        var result = Tensor<T>.Uninitialized(a.Rows, b.Columns, a.MatrixCount);
        Multiply(a, b, result);
        return result;
    }

    /// <summary>
    /// Computes the batched matrix product <c>A[:, :, k] · B[:, :, k]</c> and writes the result to <paramref name="destination"/>.
    /// <para>行列ごとの積 <c>A[:, :, k] · B[:, :, k]</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="a">The left input tensor. / 左側の入力テンソル。</param>
    /// <param name="b">The right input tensor. / 右側の入力テンソル。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap either input.</para>
    /// <para><paramref name="destination"/> はいずれの入力とも領域が重なってはなりません。</para>
    /// </remarks>
    [InferenceOverloads]
    public static TensorSpan<T> Multiply<T>(ReadOnlyTensorSpan<T> a, ReadOnlyTensorSpan<T> b, TensorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.Multiply(a, b, destination);
        return destination;
    }

    /// <summary>
    /// Returns a newly allocated tensor with each matrix slice transposed (<c>B[i, j, k] = A[j, i, k]</c>).
    /// <para>各行列スライスを転置した新しいテンソル（<c>B[i, j, k] = A[j, i, k]</c>）を返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input tensor. / 入力テンソル。</param>
    /// <returns>A new tensor with the first two axes swapped. / 第1軸と第2軸を入れ替えた新しいテンソル。</returns>
    /// <remarks>
    /// <para>For a zero-copy transposed view, use <c>x.T</c>.</para>
    /// <para>コピーを伴わない転置ビューが必要な場合は <c>x.T</c> を使用してください。</para>
    /// </remarks>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Tensor<T> Transpose<T>(ReadOnlyTensorSpan<T> x)
        where T : unmanaged, INumberBase<T>
    {
        var result = Tensor<T>.Uninitialized(x.Columns, x.Rows, x.MatrixCount);
        Transpose(x, result);
        return result;
    }

    /// <summary>
    /// Writes the slice-wise transpose <c>A[:, :, k]ᵀ</c> into <paramref name="destination"/>.
    /// <para>各行列スライスの転置 <c>A[:, :, k]ᵀ</c> を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input tensor. / 入力テンソル。</param>
    /// <param name="destination">The destination span to receive the transposed tensor. / 転置結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> must not overlap <paramref name="x"/> (in-place transposition is not supported).</para>
    /// <para><paramref name="destination"/> は <paramref name="x"/> と領域が重なってはなりません（インプレース転置には対応していません）。</para>
    /// </remarks>
    [InferenceOverloads]
    public static TensorSpan<T> Transpose<T>(ReadOnlyTensorSpan<T> x, TensorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        if (destination.Rows != x.Columns || destination.Columns != x.Rows || destination.MatrixCount != x.MatrixCount)
        {
            throw new ArgumentException("Destination shape must agree.", nameof(destination));
        }

        var source = x.T;
        var output = destination.AsReadOnly();
        if (StridedBatchLayout.Overlaps(
            in source.Reference, source.Rows, source.Columns, source.MatrixCount, source.RowStride, source.ColumnStride, source.MatrixStride,
            in output.Reference, output.Rows, output.Columns, output.MatrixCount, output.RowStride, output.ColumnStride, output.MatrixStride, out _))
        {
            throw new ArgumentException("The destination must not overlap the input.", nameof(destination));
        }

        StridedKernels.Copy(source, destination);
        return destination;
    }

    /// <summary>
    /// Solves the batched lower-triangular linear systems <c>L[k] · X[k] = B[k]</c>.
    /// <para>各行列スライスについて下三角連立一次方程式 <c>L[k] · X[k] = B[k]</c> を解きます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="a">The batch of coefficient matrices (lower triangles read). / 係数行列バッチ（下三角部分のみ参照）。</param>
    /// <param name="b">The batch of right-hand side matrices. / 右辺テンソル。</param>
    /// <returns>A new tensor containing the solutions <c>X[k]</c>. / 解 <c>X[k]</c> を格納した新しいテンソル。</returns>
    /// <remarks>
    /// <para>To solve a single right-hand side vector per slice, pass <c>B.Unsqueeze(1)</c>.</para>
    /// <para>スライスごとに1本の右辺ベクトルを解く場合は <c>B.Unsqueeze(1)</c> を渡してください。</para>
    /// </remarks>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Tensor<T> SolveLowerTriangular<T>(ReadOnlyTensorSpan<T> a, ReadOnlyTensorSpan<T> b)
        where T : unmanaged, INumberBase<T>
    {
        var result = Tensor<T>.Uninitialized(b.Rows, b.Columns, b.MatrixCount);
        SolveLowerTriangular(a, b, result);
        return result;
    }

    /// <summary>
    /// Solves the batched lower-triangular linear systems <c>L[k] · X[k] = B[k]</c> and writes the result to <paramref name="destination"/>.
    /// <para>各行列スライスについて下三角連立一次方程式 <c>L[k] · X[k] = B[k]</c> を解き、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="a">The batch of coefficient matrices (lower triangles read). / 係数行列バッチ（下三角部分のみ参照）。</param>
    /// <param name="b">The batch of right-hand side matrices. / 右辺テンソル。</param>
    /// <param name="destination">The destination span to receive the solutions. / 解を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> may identically alias <paramref name="b"/> (in-place solve). Overlap with <paramref name="a"/> or partial overlap with <paramref name="b"/> throws.</para>
    /// <para><paramref name="destination"/> は <paramref name="b"/> と完全に同一の領域（インプレース求解）を指定できます。<paramref name="a"/> との重なりや <paramref name="b"/> との部分的な重なりは例外となります。</para>
    /// </remarks>
    [InferenceOverloads]
    public static TensorSpan<T> SolveLowerTriangular<T>(ReadOnlyTensorSpan<T> a, ReadOnlyTensorSpan<T> b, TensorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.SolveTriangular(a, b, destination, lower: true);
        return destination;
    }

    /// <summary>
    /// Solves the batched upper-triangular linear systems <c>U[k] · X[k] = B[k]</c>.
    /// <para>各行列スライスについて上三角連立一次方程式 <c>U[k] · X[k] = B[k]</c> を解きます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="a">The batch of coefficient matrices (upper triangles read). / 係数行列バッチ（上三角部分のみ参照）。</param>
    /// <param name="b">The batch of right-hand side matrices. / 右辺テンソル。</param>
    /// <returns>A new tensor containing the solutions <c>X[k]</c>. / 解 <c>X[k]</c> を格納した新しいテンソル。</returns>
    /// <remarks>
    /// <para>To solve with transposed Cholesky factors, pass <c>L.T</c>.</para>
    /// <para>コレスキー因子の転置で解く場合は <c>L.T</c> を渡してください。</para>
    /// </remarks>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Tensor<T> SolveUpperTriangular<T>(ReadOnlyTensorSpan<T> a, ReadOnlyTensorSpan<T> b)
        where T : unmanaged, INumberBase<T>
    {
        var result = Tensor<T>.Uninitialized(b.Rows, b.Columns, b.MatrixCount);
        SolveUpperTriangular(a, b, result);
        return result;
    }

    /// <summary>
    /// Solves the batched upper-triangular linear systems <c>U[k] · X[k] = B[k]</c> and writes the result to <paramref name="destination"/>.
    /// <para>各行列スライスについて上三角連立一次方程式 <c>U[k] · X[k] = B[k]</c> を解き、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="a">The batch of coefficient matrices (upper triangles read). / 係数行列バッチ（上三角部分のみ参照）。</param>
    /// <param name="b">The batch of right-hand side matrices. / 右辺テンソル。</param>
    /// <param name="destination">The destination span to receive the solutions. / 解を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> may identically alias <paramref name="b"/> (in-place solve). Overlap with <paramref name="a"/> or partial overlap with <paramref name="b"/> throws.</para>
    /// <para><paramref name="destination"/> は <paramref name="b"/> と完全に同一の領域（インプレース求解）を指定できます。<paramref name="a"/> との重なりや <paramref name="b"/> との部分的な重なりは例外となります。</para>
    /// </remarks>
    [InferenceOverloads]
    public static TensorSpan<T> SolveUpperTriangular<T>(ReadOnlyTensorSpan<T> a, ReadOnlyTensorSpan<T> b, TensorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
    {
        StridedKernels.SolveTriangular(a, b, destination, lower: false);
        return destination;
    }

    /// <summary>
    /// Solves the batched linear systems <c>A[k] · X[k] = B[k]</c> using slice-wise LU decomposition with partial pivoting.
    /// <para>各行列スライスごとに部分ピボット付き LU 分解を用いて連立一次方程式 <c>A[k] · X[k] = B[k]</c> を解きます。</para>
    /// </summary>
    /// <param name="a">The batch of square coefficient matrices. / 正方の係数行列バッチ。</param>
    /// <param name="b">The batch of right-hand side matrices. / 右辺テンソル。</param>
    /// <returns>A new tensor containing the solutions <c>X[k]</c>. / 解 <c>X[k]</c> を格納した新しいテンソル。</returns>
    /// <remarks>
    /// <para>For symmetric positive definite batches, <see cref="BatchCholeskyDecomposition"/> is faster and also provides log-determinants.</para>
    /// <para>対称正定値行列のバッチには、より高速で対数行列式も計算できる <see cref="BatchCholeskyDecomposition"/> を使用してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Shapes do not match, matrices are not square, or any slice is singular. / 形状が一致しない、正方でない、またはいずれかのスライスが特異の場合。</exception>
    /// <exception cref="OverflowException">The per-slice workspace size <c>n²</c> or result element count exceeds <see cref="int.MaxValue"/>. / 各スライスの作業領域 <c>n²</c> または結果の要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static Tensor<double> Solve(ReadOnlyTensorSpan<double> a, ReadOnlyTensorSpan<double> b)
    {
        var result = Tensor<double>.Uninitialized(b.Rows, b.Columns, b.MatrixCount);
        Solve(a, b, result);
        return result;
    }

    /// <summary>
    /// Solves the batched linear systems <c>A[k] · X[k] = B[k]</c> using slice-wise LU decomposition and writes the result to <paramref name="destination"/>.
    /// <para>各行列スライスごとに部分ピボット付き LU 分解を用いて連立一次方程式 <c>A[k] · X[k] = B[k]</c> を解き、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="a">The batch of square coefficient matrices. / 正方の係数行列バッチ。</param>
    /// <param name="b">The batch of right-hand side matrices. / 右辺テンソル。</param>
    /// <param name="destination">The destination span to receive the solutions. / 解を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para><paramref name="destination"/> may identically alias <paramref name="b"/>. Overlap with <paramref name="a"/> or partial overlap with <paramref name="b"/> throws.</para>
    /// <para><paramref name="destination"/> は <paramref name="b"/> と完全に同一の領域を指定できます。<paramref name="a"/> との重なりや <paramref name="b"/> との部分的な重なりは例外となります。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Shapes do not match, matrices are not square, or any slice is singular (contents of <paramref name="destination"/> are unspecified on failure). / 形状が一致しない、正方でない、またはいずれかのスライスが特異の場合（失敗時の <paramref name="destination"/> の内容は不定）。</exception>
    /// <exception cref="OverflowException">The per-slice workspace size <c>n²</c> exceeds <see cref="int.MaxValue"/>. / 各スライスの作業領域 <c>n²</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static TensorSpan<double> Solve(ReadOnlyTensorSpan<double> a, ReadOnlyTensorSpan<double> b, TensorSpan<double> destination)
    {
        StridedKernels.Solve(a, b, destination);
        return destination;
    }

    public static partial class InPlace
    {
        /// <summary>
        /// Multiplies every element of a tensor by a scalar in place (<c>X[k] ← X[k] · scalar</c>).
        /// <para>テンソルの全要素をインプレースでスカラー倍します（<c>X[k] ← X[k] · scalar</c>）。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The tensor to modify in place. / インプレースで更新するテンソル。</param>
        /// <param name="scalar">The scalar multiplier. / 乗算するスカラー値。</param>
        public static void Multiply<T>(TensorSpan<T> x, T scalar)
            where T : unmanaged, INumberBase<T> =>
            StridedKernels.Scale(x, scalar, x);

        /// <summary>
        /// Applies a stateless unary operator elementwise to a tensor in place (<c>x ← TOperator(x)</c>).
        /// <para>状態を持たない1引数演算子をテンソルの各要素にインプレースで適用します（<c>x ← TOperator(x)</c>）。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <typeparam name="TOperator">The stateless unary operator type. / 状態を持たない1引数演算子の型。</typeparam>
        /// <param name="x">The tensor to modify in place. / インプレースで更新するテンソル。</param>
        [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
        public static void Map<T, TOperator>(TensorSpan<T> x)
            where T : unmanaged, INumberBase<T>
            where TOperator : IUnaryOperator<T> =>
            StridedKernels.Map(x, x, default(StaticUnaryOperator<T, TOperator>));

        /// <summary>
        /// Applies a stateful unary operator elementwise to a tensor in place (<c>x ← op(x)</c>).
        /// <para>状態を持つ1引数演算子をテンソルの各要素にインプレースで適用します（<c>x ← op(x)</c>）。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <typeparam name="TOperator">The stateful unary operator type. / 状態を持つ1引数演算子の型。</typeparam>
        /// <param name="x">The tensor to modify in place. / インプレースで更新するテンソル。</param>
        /// <param name="op">The operator to apply. / 適用する演算子。</param>
        [InferenceOverloads]
        public static void Map<T, TOperator>(TensorSpan<T> x, TOperator op)
            where T : unmanaged, INumberBase<T>
            where TOperator : IStatefulUnaryOperator<T> =>
            StridedKernels.Map(x, x, op);
    }
}
