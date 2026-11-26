using System.Numerics;
using System.Runtime.CompilerServices;

namespace Tuna;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

public static partial class XMath
{
    // tanh 近似 GELU の定数: 2·√(2/π) と 3次項係数。
    internal const double GeluTanhScale = 1.5957691216057307117597842397375;
    internal const double GeluTanhCubic = 0.044715;

    [ElementwiseOperation(OperatorName = "ApproximateExp", OperatorContainer = "Operators")]
    internal static T ApproximateExp<T>(T x)
        where T : unmanaged, INumberBase<T>, IExponentialFunctions<T>
    {
        if (typeof(T) == typeof(float))
        {
            float y = DenseElementwise.ApproximateExp(Unsafe.As<T, float>(ref x));
            return Unsafe.As<float, T>(ref y);
        }

        return DenseElementwise.Exp(x);
    }

    [ElementwiseOperation(OperatorName = "ApproximateLogistic", OperatorContainer = "Operators")]
    internal static T ApproximateLogistic<T>(T x)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (typeof(T) == typeof(float))
        {
            float y = DenseElementwise.ApproximateLogistic(Unsafe.As<T, float>(ref x));
            return Unsafe.As<float, T>(ref y);
        }

        return Logistic(x);
    }

    [ElementwiseOperation(OperatorName = "ApproximateSilu", OperatorContainer = "Operators")]
    internal static T ApproximateSilu<T>(T x)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (typeof(T) == typeof(float))
        {
            float y = DenseElementwise.ApproximateSilu(Unsafe.As<T, float>(ref x));
            return Unsafe.As<float, T>(ref y);
        }

        return Silu(x);
    }

    [ElementwiseOperation(OperatorName = "ApproximateSoftplus", OperatorContainer = "Operators")]
    internal static T ApproximateSoftplus<T>(T x)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (typeof(T) == typeof(float))
        {
            float y = DenseElementwise.ApproximateSoftplus(Unsafe.As<T, float>(ref x));
            return Unsafe.As<float, T>(ref y);
        }

        return Softplus(x);
    }

    // (1 + tanh u) / 2 = σ(2u) の恒等式により、負の大きな x での 1 + tanh の桁落ちを回避する。
    [ElementwiseOperation(OperatorName = "ApproximateGelu", OperatorContainer = "Operators")]
    internal static T ApproximateGelu<T>(T x)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (typeof(T) == typeof(float))
        {
            float y = DenseElementwise.ApproximateGelu(Unsafe.As<T, float>(ref x));
            return Unsafe.As<float, T>(ref y);
        }

        double v = double.CreateTruncating(x);
        double z = GeluTanhScale * (v + GeluTanhCubic * v * v * v);
        // x = -∞ では -∞ · 0 が NaN になるため極限値 -0 を返す。
        return T.CreateTruncating(double.IsNegativeInfinity(v) ? -0.0 : v * LogisticOf(z));
    }

    /// <summary>
    /// Computes the elementwise exponential <c>exp(x)</c> of a vector, optionally using a fast approximation for <see cref="float"/>.
    /// <para>ベクトルの要素ごとの指数関数 <c>exp(x)</c> を計算します（<see cref="float"/> では高速な近似式を選択可能）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="approximate"><see langword="true"/> to use a fast approximation for <see cref="float"/> (within a few ULPs); <see langword="false"/> for exact evaluation. / <see cref="float"/> で高速な近似式（誤差数 ULP）を使用する場合は <see langword="true"/>、正確な計算を行う場合は <see langword="false"/>。</param>
    /// <returns>A new vector containing <c>exp(x)</c>. / 計算結果を格納した新しいベクトル。</returns>
    /// <remarks>
    /// <para>For element types other than <see cref="float"/>, <paramref name="approximate"/> has no effect and produces the same result as <see cref="Exp{T}(ReadOnlyVectorSpan{T})"/>.</para>
    /// <para><see cref="float"/> 以外の要素型では <paramref name="approximate"/> の値にかかわらず <see cref="Exp{T}(ReadOnlyVectorSpan{T})"/> と同じ結果を返します。</para>
    /// </remarks>
    [InferenceOverloads]
    public static Vector<T> Exp<T>(ReadOnlyVectorSpan<T> x, bool approximate)
        where T : unmanaged, INumberBase<T>, IExponentialFunctions<T> =>
        approximate ? ApproximateExp(x) : Exp(x);

    /// <summary>
    /// Computes the elementwise exponential <c>exp(x)</c> of a vector and writes the result to <paramref name="destination"/>.
    /// <para>ベクトルの要素ごとの指数関数 <c>exp(x)</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="approximate"><see langword="true"/> to use the fast <see cref="float"/> approximation; otherwise, <see langword="false"/>. / <see cref="float"/> で高速な近似式を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static VectorSpan<T> Exp<T>(ReadOnlyVectorSpan<T> x, bool approximate, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>, IExponentialFunctions<T> =>
        approximate ? ApproximateExp(x, destination) : Exp(x, destination);

    /// <summary>
    /// Computes the elementwise exponential <c>exp(x)</c> of a matrix, optionally using a fast approximation for <see cref="float"/>.
    /// <para>行列の要素ごとの指数関数 <c>exp(x)</c> を計算します（<see cref="float"/> では高速な近似式を選択可能）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="approximate"><see langword="true"/> to use the fast <see cref="float"/> approximation; otherwise, <see langword="false"/>. / <see cref="float"/> で高速な近似式を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>。</param>
    /// <returns>A new matrix containing <c>exp(x)</c>. / 計算結果を格納した新しい行列。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    [InferenceOverloads]
    public static Matrix<T> Exp<T>(ReadOnlyMatrixSpan<T> x, bool approximate)
        where T : unmanaged, INumberBase<T>, IExponentialFunctions<T> =>
        approximate ? ApproximateExp(x) : Exp(x);

    /// <summary>
    /// Computes the elementwise exponential <c>exp(x)</c> of a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の要素ごとの指数関数 <c>exp(x)</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="approximate"><see langword="true"/> to use the fast <see cref="float"/> approximation; otherwise, <see langword="false"/>. / <see cref="float"/> で高速な近似式を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static MatrixSpan<T> Exp<T>(ReadOnlyMatrixSpan<T> x, bool approximate, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>, IExponentialFunctions<T> =>
        approximate ? ApproximateExp(x, destination) : Exp(x, destination);

    /// <summary>
    /// Computes the logistic sigmoid function <c>σ(x) = 1 / (1 + e⁻ˣ)</c>, optionally using a fast approximation for <see cref="float"/>.
    /// <para>ロジスティック関数 <c>σ(x) = 1 / (1 + e⁻ˣ)</c> を計算します（<see cref="float"/> では高速な近似式を選択可能）。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input value. / 入力値。</param>
    /// <param name="approximate"><see langword="true"/> to use a fast approximation for <see cref="float"/> (within a few ULPs); <see langword="false"/> for exact evaluation. / <see cref="float"/> で高速な近似式（誤差数 ULP）を使用する場合は <see langword="true"/>、正確な計算を行う場合は <see langword="false"/>。</param>
    /// <returns><c>σ(x)</c>. / <c>σ(x)</c>。</returns>
    /// <remarks>
    /// <para>For element types other than <see cref="float"/>, produces the same result as <see cref="Logistic{T}(T)"/>.</para>
    /// <para><see cref="float"/> 以外の型では <see cref="Logistic{T}(T)"/> と同じ結果を返します。</para>
    /// </remarks>
    public static T Logistic<T>(T x, bool approximate)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        approximate ? ApproximateLogistic(x) : Logistic(x);

    /// <summary>
    /// Applies the logistic sigmoid function elementwise to a vector, optionally using a fast approximation for <see cref="float"/>.
    /// <para>ベクトルの各要素にロジスティック関数を適用します（<see cref="float"/> では高速な近似式を選択可能）。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="approximate"><see langword="true"/> to use the fast <see cref="float"/> approximation; otherwise, <see langword="false"/>. / <see cref="float"/> で高速な近似式を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>。</param>
    /// <returns>A new vector containing the results. / 計算結果を格納した新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> Logistic<T>(ReadOnlyVectorSpan<T> x, bool approximate)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        approximate ? ApproximateLogistic(x) : Logistic(x);

    /// <summary>
    /// Applies the logistic sigmoid function elementwise to a vector and writes the result to <paramref name="destination"/>.
    /// <para>ベクトルの各要素にロジスティック関数を適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="approximate"><see langword="true"/> to use the fast <see cref="float"/> approximation; otherwise, <see langword="false"/>. / <see cref="float"/> で高速な近似式を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static VectorSpan<T> Logistic<T>(ReadOnlyVectorSpan<T> x, bool approximate, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        approximate ? ApproximateLogistic(x, destination) : Logistic(x, destination);

    /// <summary>
    /// Applies the logistic sigmoid function elementwise to a matrix, optionally using a fast approximation for <see cref="float"/>.
    /// <para>行列の各要素にロジスティック関数を適用します（<see cref="float"/> では高速な近似式を選択可能）。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="approximate"><see langword="true"/> to use the fast <see cref="float"/> approximation; otherwise, <see langword="false"/>. / <see cref="float"/> で高速な近似式を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>。</param>
    /// <returns>A new matrix containing the results. / 計算結果を格納した新しい行列。</returns>
    [InferenceOverloads]
    public static Matrix<T> Logistic<T>(ReadOnlyMatrixSpan<T> x, bool approximate)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        approximate ? ApproximateLogistic(x) : Logistic(x);

    /// <summary>
    /// Applies the logistic sigmoid function elementwise to a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各要素にロジスティック関数を適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="approximate"><see langword="true"/> to use the fast <see cref="float"/> approximation; otherwise, <see langword="false"/>. / <see cref="float"/> で高速な近似式を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static MatrixSpan<T> Logistic<T>(ReadOnlyMatrixSpan<T> x, bool approximate, MatrixSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        approximate ? ApproximateLogistic(x, destination) : Logistic(x, destination);

    /// <summary>
    /// Computes the SiLU (Swish) function <c>x · σ(x)</c>, optionally using a fast approximation for <see cref="float"/>.
    /// <para>SiLU（Swish）関数 <c>x · σ(x)</c> を計算します（<see cref="float"/> では高速な近似式を選択可能）。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input value. / 入力値。</param>
    /// <param name="approximate"><see langword="true"/> to use a fast approximation for <see cref="float"/> (within a few ULPs); <see langword="false"/> for exact evaluation. / <see cref="float"/> で高速な近似式（誤差数 ULP）を使用する場合は <see langword="true"/>、正確な計算を行う場合は <see langword="false"/>。</param>
    /// <returns><c>x · σ(x)</c>. / <c>x · σ(x)</c>。</returns>
    /// <remarks>
    /// <para>For element types other than <see cref="float"/>, produces the same result as <see cref="Silu{T}(T)"/>.</para>
    /// <para><see cref="float"/> 以外の型では <see cref="Silu{T}(T)"/> と同じ結果を返します。</para>
    /// </remarks>
    public static T Silu<T>(T x, bool approximate)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        approximate ? ApproximateSilu(x) : Silu(x);

    /// <summary>
    /// Applies the SiLU function elementwise to a vector, optionally using a fast approximation for <see cref="float"/>.
    /// <para>ベクトルの各要素に SiLU 関数を適用します（<see cref="float"/> では高速な近似式を選択可能）。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="approximate"><see langword="true"/> to use the fast <see cref="float"/> approximation; otherwise, <see langword="false"/>. / <see cref="float"/> で高速な近似式を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>。</param>
    /// <returns>A new vector containing the results. / 計算結果を格納した新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> Silu<T>(ReadOnlyVectorSpan<T> x, bool approximate)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        approximate ? ApproximateSilu(x) : Silu(x);

    /// <summary>
    /// Applies the SiLU function elementwise to a vector and writes the result to <paramref name="destination"/>.
    /// <para>ベクトルの各要素に SiLU 関数を適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="approximate"><see langword="true"/> to use the fast <see cref="float"/> approximation; otherwise, <see langword="false"/>. / <see cref="float"/> で高速な近似式を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static VectorSpan<T> Silu<T>(ReadOnlyVectorSpan<T> x, bool approximate, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        approximate ? ApproximateSilu(x, destination) : Silu(x, destination);

    /// <summary>
    /// Applies the SiLU function elementwise to a matrix, optionally using a fast approximation for <see cref="float"/>.
    /// <para>行列の各要素に SiLU 関数を適用します（<see cref="float"/> では高速な近似式を選択可能）。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="approximate"><see langword="true"/> to use the fast <see cref="float"/> approximation; otherwise, <see langword="false"/>. / <see cref="float"/> で高速な近似式を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>。</param>
    /// <returns>A new matrix containing the results. / 計算結果を格納した新しい行列。</returns>
    [InferenceOverloads]
    public static Matrix<T> Silu<T>(ReadOnlyMatrixSpan<T> x, bool approximate)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        approximate ? ApproximateSilu(x) : Silu(x);

    /// <summary>
    /// Applies the SiLU function elementwise to a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各要素に SiLU 関数を適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="approximate"><see langword="true"/> to use the fast <see langword="float"/> approximation; otherwise, <see langword="false"/>. / <see cref="float"/> で高速な近似式を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static MatrixSpan<T> Silu<T>(ReadOnlyMatrixSpan<T> x, bool approximate, MatrixSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        approximate ? ApproximateSilu(x, destination) : Silu(x, destination);

    /// <summary>
    /// Computes the softplus function <c>ln(1 + eˣ)</c>, optionally using a fast approximation for <see cref="float"/>.
    /// <para>Softplus 関数 <c>ln(1 + eˣ)</c> を計算します（<see cref="float"/> では高速な近似式を選択可能）。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input value. / 入力値。</param>
    /// <param name="approximate"><see langword="true"/> to use a fast approximation for <see cref="float"/> (within a few ULPs); <see langword="false"/> for exact evaluation. / <see cref="float"/> で高速な近似式（誤差数 ULP）を使用する場合は <see langword="true"/>、正確な計算を行う場合は <see langword="false"/>。</param>
    /// <returns><c>ln(1 + eˣ)</c>. / <c>ln(1 + eˣ)</c>。</returns>
    /// <remarks>
    /// <para>For element types other than <see cref="float"/>, produces the same result as <see cref="Softplus{T}(T)"/>.</para>
    /// <para><see cref="float"/> 以外の型では <see cref="Softplus{T}(T)"/> と同じ結果を返します。</para>
    /// </remarks>
    public static T Softplus<T>(T x, bool approximate)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        approximate ? ApproximateSoftplus(x) : Softplus(x);

    /// <summary>
    /// Applies the softplus function elementwise to a vector, optionally using a fast approximation for <see cref="float"/>.
    /// <para>ベクトルの各要素に Softplus 関数を適用します（<see cref="float"/> では高速な近似式を選択可能）。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="approximate"><see langword="true"/> to use the fast <see cref="float"/> approximation; otherwise, <see langword="false"/>. / <see cref="float"/> で高速な近似式を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>。</param>
    /// <returns>A new vector containing the results. / 計算結果を格納した新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> Softplus<T>(ReadOnlyVectorSpan<T> x, bool approximate)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        approximate ? ApproximateSoftplus(x) : Softplus(x);

    /// <summary>
    /// Applies the softplus function elementwise to a vector and writes the result to <paramref name="destination"/>.
    /// <para>ベクトルの各要素に Softplus 関数を適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="approximate"><see langword="true"/> to use the fast <see cref="float"/> approximation; otherwise, <see langword="false"/>. / <see cref="float"/> で高速な近似式を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static VectorSpan<T> Softplus<T>(ReadOnlyVectorSpan<T> x, bool approximate, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        approximate ? ApproximateSoftplus(x, destination) : Softplus(x, destination);

    /// <summary>
    /// Applies the softplus function elementwise to a matrix, optionally using a fast approximation for <see cref="float"/>.
    /// <para>行列の各要素に Softplus 関数を適用します（<see cref="float"/> では高速な近似式を選択可能）。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="approximate"><see langword="true"/> to use the fast <see cref="float"/> approximation; otherwise, <see langword="false"/>. / <see cref="float"/> で高速な近似式を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>。</param>
    /// <returns>A new matrix containing the results. / 計算結果を格納した新しい行列。</returns>
    [InferenceOverloads]
    public static Matrix<T> Softplus<T>(ReadOnlyMatrixSpan<T> x, bool approximate)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        approximate ? ApproximateSoftplus(x) : Softplus(x);

    /// <summary>
    /// Applies the softplus function elementwise to a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各要素に Softplus 関数を適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="approximate"><see langword="true"/> to use the fast <see cref="float"/> approximation; otherwise, <see langword="false"/>. / <see cref="float"/> で高速な近似式を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static MatrixSpan<T> Softplus<T>(ReadOnlyMatrixSpan<T> x, bool approximate, MatrixSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        approximate ? ApproximateSoftplus(x, destination) : Softplus(x, destination);

    /// <summary>
    /// Computes the GELU function, optionally using the <c>tanh</c>-based approximation.
    /// <para>GELU 関数を計算します（<c>tanh</c> による近似式を選択可能）。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input value. / 入力値。</param>
    /// <param name="approximate"><see langword="true"/> to evaluate the <c>tanh</c> approximation; <see langword="false"/> for the exact definition <see cref="Gelu{T}(T)"/>. / <c>tanh</c> による近似式を使う場合は <see langword="true"/>、定義どおりの <see cref="Gelu{T}(T)"/> を計算する場合は <see langword="false"/>。</param>
    /// <returns>The GELU value. / GELU の値。</returns>
    /// <remarks>
    /// <para>
    /// When <paramref name="approximate"/> is <see langword="true"/>, evaluates <c>(x / 2) · (1 + tanh(√(2 / π) · (x + 0.044715 · x³)))</c> (Hendrycks &amp; Gimpel 2016; maximum deviation from exact GELU is about <c>4.7 × 10⁻⁴</c> near <c>x ≈ ±2.7</c>).
    /// For <see cref="float"/>, this uses the fast approximate logistic kernel.
    /// </para>
    /// <para>
    /// <paramref name="approximate"/> が <see langword="true"/> のときは Hendrycks &amp; Gimpel (2016) の近似式 <c>(x / 2) · (1 + tanh(√(2 / π) · (x + 0.044715 · x³)))</c> を評価します（定義との最大差は <c>x ≈ ±2.7</c> 付近で約 <c>4.7 × 10⁻⁴</c>）。
    /// <see cref="float"/> では高速な近似ロジスティックカーネルを使用します。
    /// </para>
    /// </remarks>
    public static T Gelu<T>(T x, bool approximate)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        approximate ? ApproximateGelu(x) : Gelu(x);

    /// <summary>
    /// Applies the GELU function elementwise to a vector, optionally using the <c>tanh</c>-based approximation.
    /// <para>ベクトルの各要素に GELU 関数を適用します（<c>tanh</c> による近似式を選択可能）。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="approximate"><see langword="true"/> to use the <c>tanh</c> approximation; <see langword="false"/> for exact GELU. / <c>tanh</c> 近似式を使用する場合は <see langword="true"/>、正確な GELU を計算する場合は <see langword="false"/>。</param>
    /// <returns>A new vector containing the results. / 計算結果を格納した新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> Gelu<T>(ReadOnlyVectorSpan<T> x, bool approximate)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        approximate ? ApproximateGelu(x) : Gelu(x);

    /// <summary>
    /// Applies the GELU function elementwise to a vector and writes the result to <paramref name="destination"/>.
    /// <para>ベクトルの各要素に GELU 関数を適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="approximate"><see langword="true"/> to use the <c>tanh</c> approximation; <see langword="false"/> for exact GELU. / <c>tanh</c> 近似式を使用する場合は <see langword="true"/>、正確な GELU を計算する場合は <see langword="false"/>。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static VectorSpan<T> Gelu<T>(ReadOnlyVectorSpan<T> x, bool approximate, VectorSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        approximate ? ApproximateGelu(x, destination) : Gelu(x, destination);

    /// <summary>
    /// Applies the GELU function elementwise to a matrix, optionally using the <c>tanh</c>-based approximation.
    /// <para>行列の各要素に GELU 関数を適用します（<c>tanh</c> による近似式を選択可能）。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="approximate"><see langword="true"/> to use the <c>tanh</c> approximation; <see langword="false"/> for exact GELU. / <c>tanh</c> 近似式を使用する場合は <see langword="true"/>、正確な GELU を計算する場合は <see langword="false"/>。</param>
    /// <returns>A new matrix containing the results. / 計算結果を格納した新しい行列。</returns>
    [InferenceOverloads]
    public static Matrix<T> Gelu<T>(ReadOnlyMatrixSpan<T> x, bool approximate)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        approximate ? ApproximateGelu(x) : Gelu(x);

    /// <summary>
    /// Applies the GELU function elementwise to a matrix and writes the result to <paramref name="destination"/>.
    /// <para>行列の各要素に GELU 関数を適用し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="approximate"><see langword="true"/> to use the <c>tanh</c> approximation; <see langword="false"/> for exact GELU. / <c>tanh</c> 近似式を使用する場合は <see langword="true"/>、正確な GELU を計算する場合は <see langword="false"/>。</param>
    /// <param name="destination">The destination span to receive the result. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static MatrixSpan<T> Gelu<T>(ReadOnlyMatrixSpan<T> x, bool approximate, MatrixSpan<T> destination)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        approximate ? ApproximateGelu(x, destination) : Gelu(x, destination);

    public static partial class InPlace
    {
        /// <summary>
        /// Computes the elementwise exponential <c>x ← exp(x)</c> of a vector in place, optionally using a fast approximation for <see cref="float"/>.
        /// <para>ベクトルの要素ごとの指数関数 <c>x ← exp(x)</c> をインプレースで計算します（<see cref="float"/> では高速な近似式を選択可能）。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The vector to modify in place. / インプレースで更新するベクトル。</param>
        /// <param name="approximate"><see langword="true"/> to use the fast <see cref="float"/> approximation; otherwise, <see langword="false"/>. / <see cref="float"/> で高速な近似式を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>。</param>
        [InferenceOverloads]
        public static void Exp<T>(VectorSpan<T> x, bool approximate)
            where T : unmanaged, INumberBase<T>, IExponentialFunctions<T>
        {
            if (approximate)
            {
                ApproximateExp(x);
            }
            else
            {
                Exp(x);
            }
        }

        /// <summary>
        /// Computes the elementwise exponential <c>x ← exp(x)</c> of a matrix in place, optionally using a fast approximation for <see cref="float"/>.
        /// <para>行列の要素ごとの指数関数 <c>x ← exp(x)</c> をインプレースで計算します（<see cref="float"/> では高速な近似式を選択可能）。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The matrix to modify in place. / インプレースで更新する行列。</param>
        /// <param name="approximate"><see langword="true"/> to use the fast <see cref="float"/> approximation; otherwise, <see langword="false"/>. / <see cref="float"/> で高速な近似式を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>。</param>
        [InferenceOverloads]
        public static void Exp<T>(MatrixSpan<T> x, bool approximate)
            where T : unmanaged, INumberBase<T>, IExponentialFunctions<T>
        {
            if (approximate)
            {
                ApproximateExp(x);
            }
            else
            {
                Exp(x);
            }
        }

        /// <summary>
        /// Applies the logistic sigmoid function <c>x ← σ(x)</c> elementwise to a vector in place, optionally using a fast approximation for <see cref="float"/>.
        /// <para>ベクトルの各要素にロジスティック関数 <c>x ← σ(x)</c> をインプレースで適用します（<see cref="float"/> では高速な近似式を選択可能）。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
        /// <param name="x">The vector to modify in place. / インプレースで更新するベクトル。</param>
        /// <param name="approximate"><see langword="true"/> to use the fast <see cref="float"/> approximation; otherwise, <see langword="false"/>. / <see cref="float"/> で高速な近似式を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>。</param>
        [InferenceOverloads]
        public static void Logistic<T>(VectorSpan<T> x, bool approximate)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            if (approximate)
            {
                ApproximateLogistic(x);
            }
            else
            {
                Logistic(x);
            }
        }

        /// <summary>
        /// Applies the logistic sigmoid function <c>x ← σ(x)</c> elementwise to a matrix in place, optionally using a fast approximation for <see cref="float"/>.
        /// <para>行列の各要素にロジスティック関数 <c>x ← σ(x)</c> をインプレースで適用します（<see cref="float"/> では高速な近似式を選択可能）。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
        /// <param name="x">The matrix to modify in place. / インプレースで更新する行列。</param>
        /// <param name="approximate"><see langword="true"/> to use the fast <see cref="float"/> approximation; otherwise, <see langword="false"/>. / <see cref="float"/> で高速な近似式を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>。</param>
        [InferenceOverloads]
        public static void Logistic<T>(MatrixSpan<T> x, bool approximate)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            if (approximate)
            {
                ApproximateLogistic(x);
            }
            else
            {
                Logistic(x);
            }
        }

        /// <summary>
        /// Applies the SiLU function <c>x ← x · σ(x)</c> elementwise to a vector in place, optionally using a fast approximation for <see cref="float"/>.
        /// <para>ベクトルの各要素に SiLU 関数 <c>x ← x · σ(x)</c> をインプレースで適用します（<see cref="float"/> では高速な近似式を選択可能）。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
        /// <param name="x">The vector to modify in place. / インプレースで更新するベクトル。</param>
        /// <param name="approximate"><see langword="true"/> to use the fast <see cref="float"/> approximation; otherwise, <see langword="false"/>. / <see cref="float"/> で高速な近似式を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>。</param>
        [InferenceOverloads]
        public static void Silu<T>(VectorSpan<T> x, bool approximate)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            if (approximate)
            {
                ApproximateSilu(x);
            }
            else
            {
                Silu(x);
            }
        }

        /// <summary>
        /// Applies the SiLU function <c>x ← x · σ(x)</c> elementwise to a matrix in place, optionally using a fast approximation for <see cref="float"/>.
        /// <para>行列の各要素に SiLU 関数 <c>x ← x · σ(x)</c> をインプレースで適用します（<see cref="float"/> では高速な近似式を選択可能）。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
        /// <param name="x">The matrix to modify in place. / インプレースで更新する行列。</param>
        /// <param name="approximate"><see langword="true"/> to use the fast <see cref="float"/> approximation; otherwise, <see langword="false"/>. / <see cref="float"/> で高速な近似式を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>。</param>
        [InferenceOverloads]
        public static void Silu<T>(MatrixSpan<T> x, bool approximate)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            if (approximate)
            {
                ApproximateSilu(x);
            }
            else
            {
                Silu(x);
            }
        }

        /// <summary>
        /// Applies the softplus function <c>x ← ln(1 + eˣ)</c> elementwise to a vector in place, optionally using a fast approximation for <see cref="float"/>.
        /// <para>ベクトルの各要素に Softplus 関数 <c>x ← ln(1 + eˣ)</c> をインプレースで適用します（<see cref="float"/> では高速な近似式を選択可能）。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
        /// <param name="x">The vector to modify in place. / インプレースで更新するベクトル。</param>
        /// <param name="approximate"><see langword="true"/> to use the fast <see cref="float"/> approximation; otherwise, <see langword="false"/>. / <see cref="float"/> で高速な近似式を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>。</param>
        [InferenceOverloads]
        public static void Softplus<T>(VectorSpan<T> x, bool approximate)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            if (approximate)
            {
                ApproximateSoftplus(x);
            }
            else
            {
                Softplus(x);
            }
        }

        /// <summary>
        /// Applies the softplus function <c>x ← ln(1 + eˣ)</c> elementwise to a matrix in place, optionally using a fast approximation for <see cref="float"/>.
        /// <para>行列の各要素に Softplus 関数 <c>x ← ln(1 + eˣ)</c> をインプレースで適用します（<see cref="float"/> では高速な近似式を選択可能）。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
        /// <param name="x">The matrix to modify in place. / インプレースで更新する行列。</param>
        /// <param name="approximate"><see langword="true"/> to use the fast <see cref="float"/> approximation; otherwise, <see langword="false"/>. / <see cref="float"/> で高速な近似式を使用する場合は <see langword="true"/>、それ以外は <see langword="false"/>。</param>
        [InferenceOverloads]
        public static void Softplus<T>(MatrixSpan<T> x, bool approximate)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            if (approximate)
            {
                ApproximateSoftplus(x);
            }
            else
            {
                Softplus(x);
            }
        }

        /// <summary>
        /// Applies the GELU function elementwise to a vector in place, optionally using the <c>tanh</c>-based approximation.
        /// <para>ベクトルの各要素に GELU 関数をインプレースで適用します（<c>tanh</c> による近似式を選択可能）。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
        /// <param name="x">The vector to modify in place. / インプレースで更新するベクトル。</param>
        /// <param name="approximate"><see langword="true"/> to use the <c>tanh</c> approximation; <see langword="false"/> for exact GELU. / <c>tanh</c> 近似式を使用する場合は <see langword="true"/>、正確な GELU を計算する場合は <see langword="false"/>。</param>
        [InferenceOverloads]
        public static void Gelu<T>(VectorSpan<T> x, bool approximate)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            if (approximate)
            {
                ApproximateGelu(x);
            }
            else
            {
                Gelu(x);
            }
        }

        /// <summary>
        /// Applies the GELU function elementwise to a matrix in place, optionally using the <c>tanh</c>-based approximation.
        /// <para>行列の各要素に GELU 関数をインプレースで適用します（<c>tanh</c> による近似式を選択可能）。</para>
        /// </summary>
        /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
        /// <param name="x">The matrix to modify in place. / インプレースで更新する行列。</param>
        /// <param name="approximate"><see langword="true"/> to use the <c>tanh</c> approximation; <see langword="false"/> for exact GELU. / <c>tanh</c> 近似式を使用する場合は <see langword="true"/>、正確な GELU を計算する場合は <see langword="false"/>。</param>
        [InferenceOverloads]
        public static void Gelu<T>(MatrixSpan<T> x, bool approximate)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            if (approximate)
            {
                ApproximateGelu(x);
            }
            else
            {
                Gelu(x);
            }
        }
    }
}
