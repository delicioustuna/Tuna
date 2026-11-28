using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Tuna.SignalProcessing;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

public static partial class Fourier
{
    /// <summary>
    /// Computes the discrete cosine transform (DCT-I through DCT-IV) of a real vector using <see cref="FourierNormalization.Backward"/>.
    /// <para>実数ベクトルの離散コサイン変換（DCT-I〜IV）を <see cref="FourierNormalization.Backward"/> で計算します。</para>
    /// </summary>
    /// <param name="x">The input real vector. / 入力の実数ベクトル。</param>
    /// <param name="type">The DCT type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <returns>A new vector containing the DCT coefficients. / 離散コサイン変換の結果を格納した新しいベクトル。</returns>
    /// <remarks>
    /// <para>Unscaled definitions follow the SciPy convention (<c>N = n - 1</c> for type 1, <c>N = n</c> otherwise): <see cref="FourierNormalization.Backward"/> applies no scaling on forward and <c>1/(2N)</c> on inverse; <see cref="FourierNormalization.Forward"/> swaps those factors; <see cref="FourierNormalization.Orthonormal"/> scales by <c>1/√(2N)</c> and orthogonalizes boundary entries by <c>√2</c>.</para>
    /// <para>非正規化時の計算は SciPy の定義に従います（<c>N</c> は type 1 で <c>n − 1</c>、それ以外で <c>n</c>）。<see cref="FourierNormalization.Backward"/> は順変換が等倍・逆変換が <c>1/(2N)</c>、<see cref="FourierNormalization.Forward"/> はその逆、<see cref="FourierNormalization.Orthonormal"/> は <c>1/√(2N)</c> 倍に加えて変換行列が直交行列になるよう端点項を <c>√2</c> で補正します。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="type"/> is 1 and <paramref name="x"/> has length 1. / type 1 の変換で長さが 1 の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4. / <paramref name="type"/> が 1〜4 の範囲外の場合。</exception>
    public static Vector<double> Cosine(ReadOnlyVectorSpan<double> x, int type) => Cosine(x, type, FourierNormalization.Backward);

    /// <summary>
    /// Computes the discrete cosine transform (DCT-I through DCT-IV) of a real vector with the specified normalization.
    /// <para>指定した正規化方式で実数ベクトルの離散コサイン変換（DCT-I〜IV）を計算します。</para>
    /// </summary>
    /// <param name="x">The input real vector. / 入力の実数ベクトル。</param>
    /// <param name="type">The DCT type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <returns>A new vector containing the DCT coefficients. / 離散コサイン変換の結果を格納した新しいベクトル。</returns>
    /// <exception cref="ArgumentException"><paramref name="type"/> is 1 and <paramref name="x"/> has length 1. / type 1 の変換で長さが 1 の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4, or <paramref name="normalization"/> is invalid. / <paramref name="type"/> が 1〜4 の範囲外の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    public static Vector<double> Cosine(ReadOnlyVectorSpan<double> x, int type, FourierNormalization normalization)
    {
        var result = Vector<double>.Uninitialized(x.Length);
        Cosine(x, type, normalization, result);
        return result;
    }

    /// <summary>
    /// Computes the discrete cosine transform using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>実数ベクトルの離散コサイン変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real vector. / 入力の実数ベクトル。</param>
    /// <param name="type">The DCT type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="destination">The destination span with the same length as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ長さの出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different length or partially overlaps <paramref name="x"/>, or <paramref name="type"/> is 1 with length 1. / 出力先の長さが異なるか部分的に重なる場合、または type 1 で長さが 1 の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4. / <paramref name="type"/> が 1〜4 の範囲外の場合。</exception>
    public static VectorSpan<double> Cosine(ReadOnlyVectorSpan<double> x, int type, VectorSpan<double> destination) =>
        Cosine(x, type, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the discrete cosine transform with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で実数ベクトルの離散コサイン変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real vector. / 入力の実数ベクトル。</param>
    /// <param name="type">The DCT type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The destination span with the same length as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ長さの出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different length or partially overlaps <paramref name="x"/>, or <paramref name="type"/> is 1 with length 1. / 出力先の長さが異なるか部分的に重なる場合、または type 1 で長さが 1 の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4, or <paramref name="normalization"/> is invalid. / <paramref name="type"/> が 1〜4 の範囲外の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    public static VectorSpan<double> Cosine(
        ReadOnlyVectorSpan<double> x, int type, FourierNormalization normalization, VectorSpan<double> destination)
    {
        TrigonometricRows(FourierViews.AsRow(x), FourierViews.AsRow(destination), sine: false, type, inverse: false, normalization);
        return destination;
    }

    /// <summary>
    /// Computes the inverse discrete cosine transform (IDCT-I through IDCT-IV) of a real vector using <see cref="FourierNormalization.Backward"/>.
    /// <para>実数ベクトルの逆離散コサイン変換（IDCT-I〜IV）を <see cref="FourierNormalization.Backward"/> で計算します。</para>
    /// </summary>
    /// <param name="x">The input DCT coefficient vector. / 入力の DCT 係数ベクトル。</param>
    /// <param name="type">The DCT type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <returns>A new vector containing the inverse transform. / 逆離散コサイン変換の結果を格納した新しいベクトル。</returns>
    /// <remarks>
    /// <para>Inverts <see cref="Cosine(ReadOnlyVectorSpan{double}, int)"/> with the same <paramref name="type"/> and normalization (types 1 and 4 are self-inverse up to scaling; types 2 and 3 are mutual inverses).</para>
    /// <para>同じ <paramref name="type"/> と正規化方式の <see cref="Cosine(ReadOnlyVectorSpan{double}, int)"/> の逆変換です（type 1・4 は自身、type 2・3 は互いに逆変換の関係にあります）。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="type"/> is 1 and <paramref name="x"/> has length 1. / type 1 の変換で長さが 1 の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4. / <paramref name="type"/> が 1〜4 の範囲外の場合。</exception>
    public static Vector<double> InverseCosine(ReadOnlyVectorSpan<double> x, int type) => InverseCosine(x, type, FourierNormalization.Backward);

    /// <summary>
    /// Computes the inverse discrete cosine transform (IDCT-I through IDCT-IV) of a real vector with the specified normalization.
    /// <para>指定した正規化方式で実数ベクトルの逆離散コサイン変換（IDCT-I〜IV）を計算します。</para>
    /// </summary>
    /// <param name="x">The input DCT coefficient vector. / 入力の DCT 係数ベクトル。</param>
    /// <param name="type">The DCT type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <returns>A new vector containing the inverse transform. / 逆離散コサイン変換の結果を格納した新しいベクトル。</returns>
    /// <exception cref="ArgumentException"><paramref name="type"/> is 1 and <paramref name="x"/> has length 1. / type 1 の変換で長さが 1 の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4, or <paramref name="normalization"/> is invalid. / <paramref name="type"/> が 1〜4 の範囲外の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    public static Vector<double> InverseCosine(ReadOnlyVectorSpan<double> x, int type, FourierNormalization normalization)
    {
        var result = Vector<double>.Uninitialized(x.Length);
        InverseCosine(x, type, normalization, result);
        return result;
    }

    /// <summary>
    /// Computes the inverse discrete cosine transform using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>実数ベクトルの逆離散コサイン変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input DCT coefficient vector. / 入力の DCT 係数ベクトル。</param>
    /// <param name="type">The DCT type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="destination">The destination span with the same length as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ長さの出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different length or partially overlaps <paramref name="x"/>, or <paramref name="type"/> is 1 with length 1. / 出力先の長さが異なるか部分的に重なる場合、または type 1 で長さが 1 の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4. / <paramref name="type"/> が 1〜4 の範囲外の場合。</exception>
    public static VectorSpan<double> InverseCosine(ReadOnlyVectorSpan<double> x, int type, VectorSpan<double> destination) =>
        InverseCosine(x, type, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the inverse discrete cosine transform with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で実数ベクトルの逆離散コサイン変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input DCT coefficient vector. / 入力の DCT 係数ベクトル。</param>
    /// <param name="type">The DCT type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The destination span with the same length as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ長さの出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different length or partially overlaps <paramref name="x"/>, or <paramref name="type"/> is 1 with length 1. / 出力先の長さが異なるか部分的に重なる場合、または type 1 で長さが 1 の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4, or <paramref name="normalization"/> is invalid. / <paramref name="type"/> が 1〜4 の範囲外の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    public static VectorSpan<double> InverseCosine(
        ReadOnlyVectorSpan<double> x, int type, FourierNormalization normalization, VectorSpan<double> destination)
    {
        TrigonometricRows(FourierViews.AsRow(x), FourierViews.AsRow(destination), sine: false, type, inverse: true, normalization);
        return destination;
    }

    /// <summary>
    /// Computes the discrete sine transform (DST-I through DST-IV) of a real vector using <see cref="FourierNormalization.Backward"/>.
    /// <para>実数ベクトルの離散サイン変換（DST-I〜IV）を <see cref="FourierNormalization.Backward"/> で計算します。</para>
    /// </summary>
    /// <param name="x">The input real vector. / 入力の実数ベクトル。</param>
    /// <param name="type">The DST type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <returns>A new vector containing the DST coefficients. / 離散サイン変換の結果を格納した新しいベクトル。</returns>
    /// <remarks>
    /// <para>Unscaled definitions and scaling modes follow the same SciPy convention as <see cref="Cosine(ReadOnlyVectorSpan{double}, int)"/> with <c>N = n + 1</c> for type 1 and <c>N = n</c> otherwise. Under <see cref="FourierNormalization.Orthonormal"/>, type 2 divides <c>y[n - 1]</c> by <c>√2</c> and type 3 multiplies <c>x[n - 1]</c> by <c>√2</c>.</para>
    /// <para>非正規化時の定義と正規化方式は <see cref="Cosine(ReadOnlyVectorSpan{double}, int)"/> と同様の SciPy の定義に従います（<c>N</c> は type 1 で <c>n + 1</c>、それ以外で <c>n</c>）。<see cref="FourierNormalization.Orthonormal"/> では type 2 で <c>y[n − 1]</c> を <c>√2</c> で割り、type 3 で <c>x[n − 1]</c> に <c>√2</c> を掛けます。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4. / <paramref name="type"/> が 1〜4 の範囲外の場合。</exception>
    public static Vector<double> Sine(ReadOnlyVectorSpan<double> x, int type) => Sine(x, type, FourierNormalization.Backward);

    /// <summary>
    /// Computes the discrete sine transform (DST-I through DST-IV) of a real vector with the specified normalization.
    /// <para>指定した正規化方式で実数ベクトルの離散サイン変換（DST-I〜IV）を計算します。</para>
    /// </summary>
    /// <param name="x">The input real vector. / 入力の実数ベクトル。</param>
    /// <param name="type">The DST type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <returns>A new vector containing the DST coefficients. / 離散サイン変換の結果を格納した新しいベクトル。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4, or <paramref name="normalization"/> is invalid. / <paramref name="type"/> が 1〜4 の範囲外の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    public static Vector<double> Sine(ReadOnlyVectorSpan<double> x, int type, FourierNormalization normalization)
    {
        var result = Vector<double>.Uninitialized(x.Length);
        Sine(x, type, normalization, result);
        return result;
    }

    /// <summary>
    /// Computes the discrete sine transform using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>実数ベクトルの離散サイン変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real vector. / 入力の実数ベクトル。</param>
    /// <param name="type">The DST type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="destination">The destination span with the same length as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ長さの出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different length or partially overlaps <paramref name="x"/>. / 出力先の長さが異なるか、入力と部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4. / <paramref name="type"/> が 1〜4 の範囲外の場合。</exception>
    public static VectorSpan<double> Sine(ReadOnlyVectorSpan<double> x, int type, VectorSpan<double> destination) =>
        Sine(x, type, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the discrete sine transform with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で実数ベクトルの離散サイン変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real vector. / 入力の実数ベクトル。</param>
    /// <param name="type">The DST type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The destination span with the same length as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ長さの出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different length or partially overlaps <paramref name="x"/>. / 出力先の長さが異なるか、入力と部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4, or <paramref name="normalization"/> is invalid. / <paramref name="type"/> が 1〜4 の範囲外の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    public static VectorSpan<double> Sine(
        ReadOnlyVectorSpan<double> x, int type, FourierNormalization normalization, VectorSpan<double> destination)
    {
        TrigonometricRows(FourierViews.AsRow(x), FourierViews.AsRow(destination), sine: true, type, inverse: false, normalization);
        return destination;
    }

    /// <summary>
    /// Computes the inverse discrete sine transform (IDST-I through IDST-IV) of a real vector using <see cref="FourierNormalization.Backward"/>.
    /// <para>実数ベクトルの逆離散サイン変換（IDST-I〜IV）を <see cref="FourierNormalization.Backward"/> で計算します。</para>
    /// </summary>
    /// <param name="x">The input DST coefficient vector. / 入力の DST 係数ベクトル。</param>
    /// <param name="type">The DST type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <returns>A new vector containing the inverse transform. / 逆離散サイン変換の結果を格納した新しいベクトル。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4. / <paramref name="type"/> が 1〜4 の範囲外の場合。</exception>
    public static Vector<double> InverseSine(ReadOnlyVectorSpan<double> x, int type) => InverseSine(x, type, FourierNormalization.Backward);

    /// <summary>
    /// Computes the inverse discrete sine transform (IDST-I through IDST-IV) of a real vector with the specified normalization.
    /// <para>指定した正規化方式で実数ベクトルの逆離散サイン変換（IDST-I〜IV）を計算します。</para>
    /// </summary>
    /// <param name="x">The input DST coefficient vector. / 入力の DST 係数ベクトル。</param>
    /// <param name="type">The DST type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <returns>A new vector containing the inverse transform. / 逆離散サイン変換の結果を格納した新しいベクトル。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4, or <paramref name="normalization"/> is invalid. / <paramref name="type"/> が 1〜4 の範囲外の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    public static Vector<double> InverseSine(ReadOnlyVectorSpan<double> x, int type, FourierNormalization normalization)
    {
        var result = Vector<double>.Uninitialized(x.Length);
        InverseSine(x, type, normalization, result);
        return result;
    }

    /// <summary>
    /// Computes the inverse discrete sine transform using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>実数ベクトルの逆離散サイン変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input DST coefficient vector. / 入力の DST 係数ベクトル。</param>
    /// <param name="type">The DST type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="destination">The destination span with the same length as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ長さの出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different length or partially overlaps <paramref name="x"/>. / 出力先の長さが異なるか、入力と部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4. / <paramref name="type"/> が 1〜4 の範囲外の場合。</exception>
    public static VectorSpan<double> InverseSine(ReadOnlyVectorSpan<double> x, int type, VectorSpan<double> destination) =>
        InverseSine(x, type, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the inverse discrete sine transform with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で実数ベクトルの逆離散サイン変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input DST coefficient vector. / 入力の DST 係数ベクトル。</param>
    /// <param name="type">The DST type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The destination span with the same length as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ長さの出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different length or partially overlaps <paramref name="x"/>. / 出力先の長さが異なるか、入力と部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4, or <paramref name="normalization"/> is invalid. / <paramref name="type"/> が 1〜4 の範囲外の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    public static VectorSpan<double> InverseSine(
        ReadOnlyVectorSpan<double> x, int type, FourierNormalization normalization, VectorSpan<double> destination)
    {
        TrigonometricRows(FourierViews.AsRow(x), FourierViews.AsRow(destination), sine: true, type, inverse: true, normalization);
        return destination;
    }

    /// <summary>
    /// Computes the discrete cosine transform independently for each row of a matrix using <see cref="FourierNormalization.Backward"/>.
    /// <para>実数行列の各行ごとに離散コサイン変換を <see cref="FourierNormalization.Backward"/> で計算します。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="type">The DCT type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <returns>A new matrix with each row transformed. / 各行を変換した新しい行列。</returns>
    /// <remarks>
    /// <para>A 2D DCT can be computed by chaining <c>CosineColumns(CosineRows(x, type), type)</c>.</para>
    /// <para>2 次元 DCT は <c>CosineColumns(CosineRows(x, type), type)</c> の連続適用で計算できます。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="type"/> is 1 and <paramref name="x"/> has 1 column. / type 1 の変換で列数が 1 の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4. / <paramref name="type"/> が 1〜4 の範囲外の場合。</exception>
    public static Matrix<double> CosineRows(ReadOnlyMatrixSpan<double> x, int type) => CosineRows(x, type, FourierNormalization.Backward);

    /// <summary>
    /// Computes the discrete cosine transform independently for each row of a matrix with the specified normalization.
    /// <para>指定した正規化方式で実数行列の各行ごとに離散コサイン変換を計算します。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="type">The DCT type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <returns>A new matrix with each row transformed. / 各行を変換した新しい行列。</returns>
    /// <exception cref="ArgumentException"><paramref name="type"/> is 1 and <paramref name="x"/> has 1 column. / type 1 の変換で列数が 1 の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4, or <paramref name="normalization"/> is invalid. / <paramref name="type"/> が 1〜4 の範囲外の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    public static Matrix<double> CosineRows(ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization)
    {
        var result = Matrix<double>.Uninitialized(x.Rows, x.Columns);
        CosineRows(x, type, normalization, result);
        return result;
    }

    /// <summary>
    /// Computes the discrete cosine transform independently for each row using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>実数行列の各行ごとに離散コサイン変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="type">The DCT type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>, or <paramref name="type"/> is 1 with 1 column. / 出力先の形状が異なるか部分的に重なる場合、または type 1 で列数が 1 の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4. / <paramref name="type"/> が 1〜4 の範囲外の場合。</exception>
    public static MatrixSpan<double> CosineRows(ReadOnlyMatrixSpan<double> x, int type, MatrixSpan<double> destination) =>
        CosineRows(x, type, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the discrete cosine transform independently for each row with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で実数行列の各行ごとに離散コサイン変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="type">The DCT type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>, or <paramref name="type"/> is 1 with 1 column. / 出力先の形状が異なるか部分的に重なる場合、または type 1 で列数が 1 の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4, or <paramref name="normalization"/> is invalid. / <paramref name="type"/> が 1〜4 の範囲外の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    public static MatrixSpan<double> CosineRows(
        ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization, MatrixSpan<double> destination)
    {
        TrigonometricRows(x, destination, sine: false, type, inverse: false, normalization);
        return destination;
    }

    /// <summary>
    /// Computes the discrete cosine transform independently for each column of a matrix using <see cref="FourierNormalization.Backward"/>.
    /// <para>実数行列の各列ごとに離散コサイン変換を <see cref="FourierNormalization.Backward"/> で計算します。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="type">The DCT type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <returns>A new matrix with each column transformed. / 各列を変換した新しい行列。</returns>
    /// <exception cref="ArgumentException"><paramref name="type"/> is 1 and <paramref name="x"/> has 1 row. / type 1 の変換で行数が 1 の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4. / <paramref name="type"/> が 1〜4 の範囲外の場合。</exception>
    public static Matrix<double> CosineColumns(ReadOnlyMatrixSpan<double> x, int type) => CosineColumns(x, type, FourierNormalization.Backward);

    /// <summary>
    /// Computes the discrete cosine transform independently for each column of a matrix with the specified normalization.
    /// <para>指定した正規化方式で実数行列の各列ごとに離散コサイン変換を計算します。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="type">The DCT type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <returns>A new matrix with each column transformed. / 各列を変換した新しい行列。</returns>
    /// <exception cref="ArgumentException"><paramref name="type"/> is 1 and <paramref name="x"/> has 1 row. / type 1 の変換で行数が 1 の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4, or <paramref name="normalization"/> is invalid. / <paramref name="type"/> が 1〜4 の範囲外の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    public static Matrix<double> CosineColumns(ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization)
    {
        var result = Matrix<double>.Uninitialized(x.Rows, x.Columns);
        CosineColumns(x, type, normalization, result);
        return result;
    }

    /// <summary>
    /// Computes the discrete cosine transform independently for each column using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>実数行列の各列ごとに離散コサイン変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="type">The DCT type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>, or <paramref name="type"/> is 1 with 1 row. / 出力先の形状が異なるか部分的に重なる場合、または type 1 で行数が 1 の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4. / <paramref name="type"/> が 1〜4 の範囲外の場合。</exception>
    public static MatrixSpan<double> CosineColumns(ReadOnlyMatrixSpan<double> x, int type, MatrixSpan<double> destination) =>
        CosineColumns(x, type, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the discrete cosine transform independently for each column with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で実数行列の各列ごとに離散コサイン変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="type">The DCT type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>, or <paramref name="type"/> is 1 with 1 row. / 出力先の形状が異なるか部分的に重なる場合、または type 1 で行数が 1 の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4, or <paramref name="normalization"/> is invalid. / <paramref name="type"/> が 1〜4 の範囲外の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    public static MatrixSpan<double> CosineColumns(
        ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization, MatrixSpan<double> destination)
    {
        TrigonometricRows(x.T, destination.T, sine: false, type, inverse: false, normalization);
        return destination;
    }

    /// <summary>
    /// Computes the inverse discrete cosine transform independently for each row of a matrix using <see cref="FourierNormalization.Backward"/>.
    /// <para>実数行列の各行ごとに逆離散コサイン変換を <see cref="FourierNormalization.Backward"/> で計算します。</para>
    /// </summary>
    /// <param name="x">The input matrix of row-wise DCT coefficients. / 行ごとの DCT 係数行列。</param>
    /// <param name="type">The DCT type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <returns>A new matrix with each row inverse-transformed. / 各行を逆変換した新しい行列。</returns>
    /// <exception cref="ArgumentException"><paramref name="type"/> is 1 and <paramref name="x"/> has 1 column. / type 1 の変換で列数が 1 の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4. / <paramref name="type"/> が 1〜4 の範囲外の場合。</exception>
    public static Matrix<double> InverseCosineRows(ReadOnlyMatrixSpan<double> x, int type) =>
        InverseCosineRows(x, type, FourierNormalization.Backward);

    /// <summary>
    /// Computes the inverse discrete cosine transform independently for each row of a matrix with the specified normalization.
    /// <para>指定した正規化方式で実数行列の各行ごとに逆離散コサイン変換を計算します。</para>
    /// </summary>
    /// <param name="x">The input matrix of row-wise DCT coefficients. / 行ごとの DCT 係数行列。</param>
    /// <param name="type">The DCT type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <returns>A new matrix with each row inverse-transformed. / 各行を逆変換した新しい行列。</returns>
    /// <exception cref="ArgumentException"><paramref name="type"/> is 1 and <paramref name="x"/> has 1 column. / type 1 の変換で列数が 1 の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4, or <paramref name="normalization"/> is invalid. / <paramref name="type"/> が 1〜4 の範囲外の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    public static Matrix<double> InverseCosineRows(ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization)
    {
        var result = Matrix<double>.Uninitialized(x.Rows, x.Columns);
        InverseCosineRows(x, type, normalization, result);
        return result;
    }

    /// <summary>
    /// Computes the inverse discrete cosine transform independently for each row using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>実数行列の各行ごとに逆離散コサイン変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input matrix of row-wise DCT coefficients. / 行ごとの DCT 係数行列。</param>
    /// <param name="type">The DCT type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>, or <paramref name="type"/> is 1 with 1 column. / 出力先の形状が異なるか部分的に重なる場合、または type 1 で列数が 1 の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4. / <paramref name="type"/> が 1〜4 の範囲外の場合。</exception>
    public static MatrixSpan<double> InverseCosineRows(ReadOnlyMatrixSpan<double> x, int type, MatrixSpan<double> destination) =>
        InverseCosineRows(x, type, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the inverse discrete cosine transform independently for each row with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で実数行列の各行ごとに逆離散コサイン変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input matrix of row-wise DCT coefficients. / 行ごとの DCT 係数行列。</param>
    /// <param name="type">The DCT type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>, or <paramref name="type"/> is 1 with 1 column. / 出力先の形状が異なるか部分的に重なる場合、または type 1 で列数が 1 の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4, or <paramref name="normalization"/> is invalid. / <paramref name="type"/> が 1〜4 の範囲外の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    public static MatrixSpan<double> InverseCosineRows(
        ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization, MatrixSpan<double> destination)
    {
        TrigonometricRows(x, destination, sine: false, type, inverse: true, normalization);
        return destination;
    }

    /// <summary>
    /// Computes the inverse discrete cosine transform independently for each column of a matrix using <see cref="FourierNormalization.Backward"/>.
    /// <para>実数行列の各列ごとに逆離散コサイン変換を <see cref="FourierNormalization.Backward"/> で計算します。</para>
    /// </summary>
    /// <param name="x">The input matrix of column-wise DCT coefficients. / 列ごとの DCT 係数行列。</param>
    /// <param name="type">The DCT type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <returns>A new matrix with each column inverse-transformed. / 各列を逆変換した新しい行列。</returns>
    /// <exception cref="ArgumentException"><paramref name="type"/> is 1 and <paramref name="x"/> has 1 row. / type 1 の変換で行数が 1 の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4. / <paramref name="type"/> が 1〜4 の範囲外の場合。</exception>
    public static Matrix<double> InverseCosineColumns(ReadOnlyMatrixSpan<double> x, int type) =>
        InverseCosineColumns(x, type, FourierNormalization.Backward);

    /// <summary>
    /// Computes the inverse discrete cosine transform independently for each column of a matrix with the specified normalization.
    /// <para>指定した正規化方式で実数行列の各列ごとに逆離散コサイン変換を計算します。</para>
    /// </summary>
    /// <param name="x">The input matrix of column-wise DCT coefficients. / 列ごとの DCT 係数行列。</param>
    /// <param name="type">The DCT type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <returns>A new matrix with each column inverse-transformed. / 各列を逆変換した新しい行列。</returns>
    /// <exception cref="ArgumentException"><paramref name="type"/> is 1 and <paramref name="x"/> has 1 row. / type 1 の変換で行数が 1 の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4, or <paramref name="normalization"/> is invalid. / <paramref name="type"/> が 1〜4 の範囲外の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    public static Matrix<double> InverseCosineColumns(ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization)
    {
        var result = Matrix<double>.Uninitialized(x.Rows, x.Columns);
        InverseCosineColumns(x, type, normalization, result);
        return result;
    }

    /// <summary>
    /// Computes the inverse discrete cosine transform independently for each column using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>実数行列の各列ごとに逆離散コサイン変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input matrix of column-wise DCT coefficients. / 列ごとの DCT 係数行列。</param>
    /// <param name="type">The DCT type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>, or <paramref name="type"/> is 1 with 1 row. / 出力先の形状が異なるか部分的に重なる場合、または type 1 で行数が 1 の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4. / <paramref name="type"/> が 1〜4 の範囲外の場合。</exception>
    public static MatrixSpan<double> InverseCosineColumns(ReadOnlyMatrixSpan<double> x, int type, MatrixSpan<double> destination) =>
        InverseCosineColumns(x, type, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the inverse discrete cosine transform independently for each column with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で実数行列の各列ごとに逆離散コサイン変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input matrix of column-wise DCT coefficients. / 列ごとの DCT 係数行列。</param>
    /// <param name="type">The DCT type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>, or <paramref name="type"/> is 1 with 1 row. / 出力先の形状が異なるか部分的に重なる場合、または type 1 で行数が 1 の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4, or <paramref name="normalization"/> is invalid. / <paramref name="type"/> が 1〜4 の範囲外の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    public static MatrixSpan<double> InverseCosineColumns(
        ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization, MatrixSpan<double> destination)
    {
        TrigonometricRows(x.T, destination.T, sine: false, type, inverse: true, normalization);
        return destination;
    }

    /// <summary>
    /// Computes the discrete sine transform independently for each row of a matrix using <see cref="FourierNormalization.Backward"/>.
    /// <para>実数行列の各行ごとに離散サイン変換を <see cref="FourierNormalization.Backward"/> で計算します。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="type">The DST type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <returns>A new matrix with each row transformed. / 各行を変換した新しい行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4. / <paramref name="type"/> が 1〜4 の範囲外の場合。</exception>
    public static Matrix<double> SineRows(ReadOnlyMatrixSpan<double> x, int type) => SineRows(x, type, FourierNormalization.Backward);

    /// <summary>
    /// Computes the discrete sine transform independently for each row of a matrix with the specified normalization.
    /// <para>指定した正規化方式で実数行列の各行ごとに離散サイン変換を計算します。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="type">The DST type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <returns>A new matrix with each row transformed. / 各行を変換した新しい行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4, or <paramref name="normalization"/> is invalid. / <paramref name="type"/> が 1〜4 の範囲外の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    public static Matrix<double> SineRows(ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization)
    {
        var result = Matrix<double>.Uninitialized(x.Rows, x.Columns);
        SineRows(x, type, normalization, result);
        return result;
    }

    /// <summary>
    /// Computes the discrete sine transform independently for each row using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>実数行列の各行ごとに離散サイン変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="type">The DST type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4. / <paramref name="type"/> が 1〜4 の範囲外の場合。</exception>
    public static MatrixSpan<double> SineRows(ReadOnlyMatrixSpan<double> x, int type, MatrixSpan<double> destination) =>
        SineRows(x, type, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the discrete sine transform independently for each row with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で実数行列の各行ごとに離散サイン変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="type">The DST type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4, or <paramref name="normalization"/> is invalid. / <paramref name="type"/> が 1〜4 の範囲外の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    public static MatrixSpan<double> SineRows(
        ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization, MatrixSpan<double> destination)
    {
        TrigonometricRows(x, destination, sine: true, type, inverse: false, normalization);
        return destination;
    }

    /// <summary>
    /// Computes the discrete sine transform independently for each column of a matrix using <see cref="FourierNormalization.Backward"/>.
    /// <para>実数行列の各列ごとに離散サイン変換を <see cref="FourierNormalization.Backward"/> で計算します。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="type">The DST type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <returns>A new matrix with each column transformed. / 各列を変換した新しい行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4. / <paramref name="type"/> が 1〜4 の範囲外の場合。</exception>
    public static Matrix<double> SineColumns(ReadOnlyMatrixSpan<double> x, int type) => SineColumns(x, type, FourierNormalization.Backward);

    /// <summary>
    /// Computes the discrete sine transform independently for each column of a matrix with the specified normalization.
    /// <para>指定した正規化方式で実数行列の各列ごとに離散サイン変換を計算します。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="type">The DST type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <returns>A new matrix with each column transformed. / 各列を変換した新しい行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4, or <paramref name="normalization"/> is invalid. / <paramref name="type"/> が 1〜4 の範囲外の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    public static Matrix<double> SineColumns(ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization)
    {
        var result = Matrix<double>.Uninitialized(x.Rows, x.Columns);
        SineColumns(x, type, normalization, result);
        return result;
    }

    /// <summary>
    /// Computes the discrete sine transform independently for each column using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>実数行列の各列ごとに離散サイン変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="type">The DST type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4. / <paramref name="type"/> が 1〜4 の範囲外の場合。</exception>
    public static MatrixSpan<double> SineColumns(ReadOnlyMatrixSpan<double> x, int type, MatrixSpan<double> destination) =>
        SineColumns(x, type, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the discrete sine transform independently for each column with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で実数行列の各列ごとに離散サイン変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="type">The DST type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4, or <paramref name="normalization"/> is invalid. / <paramref name="type"/> が 1〜4 の範囲外の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    public static MatrixSpan<double> SineColumns(
        ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization, MatrixSpan<double> destination)
    {
        TrigonometricRows(x.T, destination.T, sine: true, type, inverse: false, normalization);
        return destination;
    }

    /// <summary>
    /// Computes the inverse discrete sine transform independently for each row of a matrix using <see cref="FourierNormalization.Backward"/>.
    /// <para>実数行列の各行ごとに逆離散サイン変換を <see cref="FourierNormalization.Backward"/> で計算します。</para>
    /// </summary>
    /// <param name="x">The input matrix of row-wise DST coefficients. / 行ごとの DST 係数行列。</param>
    /// <param name="type">The DST type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <returns>A new matrix with each row inverse-transformed. / 各行を逆変換した新しい行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4. / <paramref name="type"/> が 1〜4 の範囲外の場合。</exception>
    public static Matrix<double> InverseSineRows(ReadOnlyMatrixSpan<double> x, int type) =>
        InverseSineRows(x, type, FourierNormalization.Backward);

    /// <summary>
    /// Computes the inverse discrete sine transform independently for each row of a matrix with the specified normalization.
    /// <para>指定した正規化方式で実数行列の各行ごとに逆離散サイン変換を計算します。</para>
    /// </summary>
    /// <param name="x">The input matrix of row-wise DST coefficients. / 行ごとの DST 係数行列。</param>
    /// <param name="type">The DST type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <returns>A new matrix with each row inverse-transformed. / 各行を逆変換した新しい行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4, or <paramref name="normalization"/> is invalid. / <paramref name="type"/> が 1〜4 の範囲外の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    public static Matrix<double> InverseSineRows(ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization)
    {
        var result = Matrix<double>.Uninitialized(x.Rows, x.Columns);
        InverseSineRows(x, type, normalization, result);
        return result;
    }

    /// <summary>
    /// Computes the inverse discrete sine transform independently for each row using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>実数行列の各行ごとに逆離散サイン変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input matrix of row-wise DST coefficients. / 行ごとの DST 係数行列。</param>
    /// <param name="type">The DST type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4. / <paramref name="type"/> が 1〜4 の範囲外の場合。</exception>
    public static MatrixSpan<double> InverseSineRows(ReadOnlyMatrixSpan<double> x, int type, MatrixSpan<double> destination) =>
        InverseSineRows(x, type, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the inverse discrete sine transform independently for each row with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で実数行列の各行ごとに逆離散サイン変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input matrix of row-wise DST coefficients. / 行ごとの DST 係数行列。</param>
    /// <param name="type">The DST type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4, or <paramref name="normalization"/> is invalid. / <paramref name="type"/> が 1〜4 の範囲外の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    public static MatrixSpan<double> InverseSineRows(
        ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization, MatrixSpan<double> destination)
    {
        TrigonometricRows(x, destination, sine: true, type, inverse: true, normalization);
        return destination;
    }

    /// <summary>
    /// Computes the inverse discrete sine transform independently for each column of a matrix using <see cref="FourierNormalization.Backward"/>.
    /// <para>実数行列の各列ごとに逆離散サイン変換を <see cref="FourierNormalization.Backward"/> で計算します。</para>
    /// </summary>
    /// <param name="x">The input matrix of column-wise DST coefficients. / 列ごとの DST 係数行列。</param>
    /// <param name="type">The DST type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <returns>A new matrix with each column inverse-transformed. / 各列を逆変換した新しい行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4. / <paramref name="type"/> が 1〜4 の範囲外の場合。</exception>
    public static Matrix<double> InverseSineColumns(ReadOnlyMatrixSpan<double> x, int type) =>
        InverseSineColumns(x, type, FourierNormalization.Backward);

    /// <summary>
    /// Computes the inverse discrete sine transform independently for each column of a matrix with the specified normalization.
    /// <para>指定した正規化方式で実数行列の各列ごとに逆離散サイン変換を計算します。</para>
    /// </summary>
    /// <param name="x">The input matrix of column-wise DST coefficients. / 列ごとの DST 係数行列。</param>
    /// <param name="type">The DST type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <returns>A new matrix with each column inverse-transformed. / 各列を逆変換した新しい行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4, or <paramref name="normalization"/> is invalid. / <paramref name="type"/> が 1〜4 の範囲外の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    public static Matrix<double> InverseSineColumns(ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization)
    {
        var result = Matrix<double>.Uninitialized(x.Rows, x.Columns);
        InverseSineColumns(x, type, normalization, result);
        return result;
    }

    /// <summary>
    /// Computes the inverse discrete sine transform independently for each column using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>実数行列の各列ごとに逆離散サイン変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input matrix of column-wise DST coefficients. / 列ごとの DST 係数行列。</param>
    /// <param name="type">The DST type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4. / <paramref name="type"/> が 1〜4 の範囲外の場合。</exception>
    public static MatrixSpan<double> InverseSineColumns(ReadOnlyMatrixSpan<double> x, int type, MatrixSpan<double> destination) =>
        InverseSineColumns(x, type, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the inverse discrete sine transform independently for each column with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で実数行列の各列ごとに逆離散サイン変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input matrix of column-wise DST coefficients. / 列ごとの DST 係数行列。</param>
    /// <param name="type">The DST type, from 1 to 4. / 変換の型（1〜4）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="type"/> is not between 1 and 4, or <paramref name="normalization"/> is invalid. / <paramref name="type"/> が 1〜4 の範囲外の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    public static MatrixSpan<double> InverseSineColumns(
        ReadOnlyMatrixSpan<double> x, int type, FourierNormalization normalization, MatrixSpan<double> destination)
    {
        TrigonometricRows(x.T, destination.T, sine: true, type, inverse: true, normalization);
        return destination;
    }

    private static void TrigonometricRows(
        ReadOnlyMatrixSpan<double> x, MatrixSpan<double> destination, bool sine, int type, bool inverse, FourierNormalization normalization)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(type, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(type, 4);
        if (normalization is not (FourierNormalization.Backward or FourierNormalization.Orthonormal or FourierNormalization.Forward))
        {
            throw new ArgumentOutOfRangeException(nameof(normalization), normalization, "Unknown normalization.");
        }

        if (x.Rows != destination.Rows || x.Columns != destination.Columns)
        {
            throw new ArgumentException(
                $"The destination must be {x.Rows} x {x.Columns}, but it is {destination.Rows} x {destination.Columns}.", nameof(destination));
        }

        var output = destination.AsReadOnly();
        if (StridedLayout.Overlaps(
                in x.Reference, x.Rows, x.Columns, x.RowStride, x.ColumnStride,
                in output.Reference, output.Rows, output.Columns, output.RowStride, output.ColumnStride, out bool exact) && !exact)
        {
            throw new ArgumentException("The destination partially overlaps the input. Only an exact alias is allowed.", nameof(destination));
        }

        int n = x.Columns;
        if (!sine && type == 1 && n == 1)
        {
            throw new ArgumentException("The type 1 cosine transform needs at least two points.", nameof(x));
        }

        if (x.Rows == 0 || n == 0)
        {
            return;
        }

        int actual = inverse && type is 2 or 3 ? 5 - type : type;
        double size = 2.0 * (type != 1 ? n : sine ? n + 1 : n - 1);
        bool orthonormal = normalization == FourierNormalization.Orthonormal;
        double scale = normalization switch
        {
            FourierNormalization.Orthonormal => 1 / Math.Sqrt(size),
            FourierNormalization.Backward => inverse ? 1 / size : 1,
            _ => inverse ? 1 : 1 / size,
        };

        var kind = sine && actual == 1 ? TrigonometricKind.Sine1 : (TrigonometricKind)(actual - 1);
        var plan = TrigonometricPlan.Create(kind, n);
        var options = new TrigonometricRowOptions(sine, actual, scale, orthonormal);
        int threads = FourierViews.RowThreads(x.Rows, n, 1);
        if (threads > 1)
        {
            TrigonometricRowsParallel(plan, x, destination, options, threads);
            return;
        }

        TrigonometricRowRange(plan, x, destination, 0, x.Rows, options);
    }

    private readonly record struct TrigonometricRowOptions(bool Sine, int Actual, double Scale, bool Orthonormal);

    // 逐次経路でクロージャーの割り当てを避けるため、並列実行を別メソッドに分離している。
    private static void TrigonometricRowsParallel(
        TrigonometricPlan plan, ReadOnlyMatrixSpan<double> x, MatrixSpan<double> destination, TrigonometricRowOptions options, int threads) =>
        FourierViews.ParallelRows(x, destination, threads, 1, (xv, dv, first, count) => TrigonometricRowRange(plan, xv, dv, first, count, options));

    private static void TrigonometricRowRange(
        TrigonometricPlan plan, ReadOnlyMatrixSpan<double> x, MatrixSpan<double> destination, int first, int count, TrigonometricRowOptions options)
    {
        var output = destination.AsReadOnly();
        int n = x.Columns;
        bool sine = options.Sine;
        int actual = options.Actual;
        double scale = options.Scale;
        double[] buffer = ArrayPool<double>.Shared.Rent(2 * n);
        Complex[] work = ArrayPool<Complex>.Shared.Rent(Math.Max(plan.WorkLength, 1));
        var input = buffer.AsSpan(0, n);
        var result = buffer.AsSpan(n, n);
        double root2 = XMath.Constants.Sqrt2;

        // 直交化の端点補正（SciPy norm="ortho" の定義）: 入力側は DCT-I/III・DST-III、出力側は DCT-I/II・DST-II の端点を √2 倍/除算する。
        bool correctInput = options.Orthonormal && (actual == 3 || (actual == 1 && !sine));
        bool correctOutput = options.Orthonormal && (actual == 2 || (actual == 1 && !sine));
        int edge = sine ? n - 1 : 0;
        nint inputStride = x.ColumnStride, outputStride = output.ColumnStride;
        bool directInput = inputStride == 1 && !correctInput;
        bool directOutput = outputStride == 1;
        for (int i = first; i < first + count; i++)
        {
            ref double source = ref Unsafe.AsRef(in x.ElementAt(i, 0));
            ReadOnlySpan<double> transformInput = input;
            if (directInput)
            {
                transformInput = MemoryMarshal.CreateReadOnlySpan(ref source, n);
            }
            else
            {
                for (int j = 0; j < n; j++)
                {
                    input[j] = Unsafe.Add(ref source, j * inputStride);
                }

                if (correctInput)
                {
                    input[edge] *= root2;
                    if (actual == 1)
                    {
                        input[n - 1] *= root2;
                    }
                }
            }

            ref double target = ref Unsafe.AsRef(in output.ElementAt(i, 0));
            var transformOutput = directOutput ? MemoryMarshal.CreateSpan(ref target, n) : result;
            plan.Execute(transformInput, transformOutput, work, sine);
            if (correctOutput)
            {
                transformOutput[edge] /= root2;
                if (actual == 1)
                {
                    transformOutput[n - 1] /= root2;
                }
            }

            if (directOutput)
            {
                if (scale != 1)
                {
                    for (int k = 0; k < n; k++)
                    {
                        transformOutput[k] *= scale;
                    }
                }
            }
            else
            {
                for (int k = 0; k < n; k++)
                {
                    Unsafe.Add(ref target, k * outputStride) = result[k] * scale;
                }
            }
        }

        ArrayPool<Complex>.Shared.Return(work);
        ArrayPool<double>.Shared.Return(buffer);
    }
}
