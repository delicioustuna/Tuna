using System.Numerics;

namespace Tuna;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

public static partial class XMath
{
    /// <summary>
    /// Extracts the real part of each element in a complex vector.
    /// <para>複素数ベクトルの各要素の実部を抽出します。</para>
    /// </summary>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>A new vector containing the real parts. / 各要素の実部を格納した新しいベクトル。</returns>
    public static Vector<double> Real(ReadOnlyVectorSpan<Complex> x)
    {
        var result = Vector<double>.Uninitialized(x.Length);
        Real(x, result);
        return result;
    }

    /// <summary>
    /// Extracts the real part of each element in a complex vector and writes the result to <paramref name="destination"/>.
    /// <para>複素数ベクトルの各要素の実部を抽出し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="destination">The destination span to receive the real parts. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">The lengths differ, or <paramref name="destination"/> shares memory with <paramref name="x"/>. / 長さが一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と記憶域を共有する場合。</exception>
    public static VectorSpan<double> Real(ReadOnlyVectorSpan<Complex> x, VectorSpan<double> destination)
    {
        StridedKernels.MapTo(x.AsMatrix(), destination.AsMatrix(), default(RealPartOperator));
        return destination;
    }

    /// <summary>
    /// Extracts the real part of each element in a complex matrix.
    /// <para>複素数行列の各要素の実部を抽出します。</para>
    /// </summary>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new matrix containing the real parts. / 各要素の実部を格納した新しい行列。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    public static Matrix<double> Real(ReadOnlyMatrixSpan<Complex> x)
    {
        var result = Matrix<double>.Uninitialized(x.Rows, x.Columns);
        Real(x, result);
        return result;
    }

    /// <summary>
    /// Extracts the real part of each element in a complex matrix and writes the result to <paramref name="destination"/>.
    /// <para>複素数行列の各要素の実部を抽出し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span to receive the real parts. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">The shapes differ, or <paramref name="destination"/> shares memory with <paramref name="x"/>. / 形状が一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と記憶域を共有する場合。</exception>
    public static MatrixSpan<double> Real(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<double> destination)
    {
        StridedKernels.MapTo(x, destination, default(RealPartOperator));
        return destination;
    }

    /// <summary>
    /// Extracts the imaginary part of each element in a complex vector.
    /// <para>複素数ベクトルの各要素の虚部を抽出します。</para>
    /// </summary>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>A new vector containing the imaginary parts. / 各要素の虚部を格納した新しいベクトル。</returns>
    public static Vector<double> Imaginary(ReadOnlyVectorSpan<Complex> x)
    {
        var result = Vector<double>.Uninitialized(x.Length);
        Imaginary(x, result);
        return result;
    }

    /// <summary>
    /// Extracts the imaginary part of each element in a complex vector and writes the result to <paramref name="destination"/>.
    /// <para>複素数ベクトルの各要素の虚部を抽出し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="destination">The destination span to receive the imaginary parts. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">The lengths differ, or <paramref name="destination"/> shares memory with <paramref name="x"/>. / 長さが一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と記憶域を共有する場合。</exception>
    public static VectorSpan<double> Imaginary(ReadOnlyVectorSpan<Complex> x, VectorSpan<double> destination)
    {
        StridedKernels.MapTo(x.AsMatrix(), destination.AsMatrix(), default(ImaginaryPartOperator));
        return destination;
    }

    /// <summary>
    /// Extracts the imaginary part of each element in a complex matrix.
    /// <para>複素数行列の各要素の虚部を抽出します。</para>
    /// </summary>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new matrix containing the imaginary parts. / 各要素の虚部を格納した新しい行列。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    public static Matrix<double> Imaginary(ReadOnlyMatrixSpan<Complex> x)
    {
        var result = Matrix<double>.Uninitialized(x.Rows, x.Columns);
        Imaginary(x, result);
        return result;
    }

    /// <summary>
    /// Extracts the imaginary part of each element in a complex matrix and writes the result to <paramref name="destination"/>.
    /// <para>複素数行列の各要素の虚部を抽出し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span to receive the imaginary parts. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">The shapes differ, or <paramref name="destination"/> shares memory with <paramref name="x"/>. / 形状が一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と記憶域を共有する場合。</exception>
    public static MatrixSpan<double> Imaginary(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<double> destination)
    {
        StridedKernels.MapTo(x, destination, default(ImaginaryPartOperator));
        return destination;
    }

    /// <summary>
    /// Computes the magnitude <c>|z|</c> of each element in a complex vector.
    /// <para>複素数ベクトルの各要素の大きさ <c>|z|</c> を計算します。</para>
    /// </summary>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>A new vector containing the element magnitudes. / 各要素の大きさを格納した新しいベクトル。</returns>
    /// <remarks>
    /// <para>Matches <see cref="Complex.Magnitude"/> without intermediate overflow or underflow: returns <c>+∞</c> if either component is <c>±∞</c> (even if the other is <c>NaN</c>), and <c>NaN</c> otherwise if either component is <c>NaN</c>.</para>
    /// <para><see cref="Complex.Magnitude"/> と同じ値を返します。途中計算でオーバーフローやアンダーフローを起こさず、いずれかの成分が <c>±∞</c> なら（他方が <c>NaN</c> でも）<c>+∞</c>、それ以外で <c>NaN</c> を含めば <c>NaN</c> を返します。</para>
    /// </remarks>
    public static Vector<double> Magnitude(ReadOnlyVectorSpan<Complex> x)
    {
        var result = Vector<double>.Uninitialized(x.Length);
        Magnitude(x, result);
        return result;
    }

    /// <summary>
    /// Computes the magnitude <c>|z|</c> of each element in a complex vector and writes the result to <paramref name="destination"/>.
    /// <para>複素数ベクトルの各要素の大きさ <c>|z|</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="destination">The destination span to receive the magnitudes. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">The lengths differ, or <paramref name="destination"/> shares memory with <paramref name="x"/>. / 長さが一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と記憶域を共有する場合。</exception>
    public static VectorSpan<double> Magnitude(ReadOnlyVectorSpan<Complex> x, VectorSpan<double> destination)
    {
        StridedKernels.MapTo(x.AsMatrix(), destination.AsMatrix(), default(MagnitudeOperator));
        return destination;
    }

    /// <summary>
    /// Computes the magnitude <c>|z|</c> of each element in a complex matrix.
    /// <para>複素数行列の各要素の大きさ <c>|z|</c> を計算します。</para>
    /// </summary>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new matrix containing the element magnitudes. / 各要素の大きさを格納した新しい行列。</returns>
    /// <remarks>
    /// <para>Matches <see cref="Complex.Magnitude"/> with the same edge-case semantics as <see cref="Magnitude(ReadOnlyVectorSpan{Complex})"/>.</para>
    /// <para><see cref="Complex.Magnitude"/> と同じ値を返します。境界条件の挙動は <see cref="Magnitude(ReadOnlyVectorSpan{Complex})"/> と同様です。</para>
    /// </remarks>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    public static Matrix<double> Magnitude(ReadOnlyMatrixSpan<Complex> x)
    {
        var result = Matrix<double>.Uninitialized(x.Rows, x.Columns);
        Magnitude(x, result);
        return result;
    }

    /// <summary>
    /// Computes the magnitude <c>|z|</c> of each element in a complex matrix and writes the result to <paramref name="destination"/>.
    /// <para>複素数行列の各要素の大きさ <c>|z|</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span to receive the magnitudes. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">The shapes differ, or <paramref name="destination"/> shares memory with <paramref name="x"/>. / 形状が一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と記憶域を共有する場合。</exception>
    public static MatrixSpan<double> Magnitude(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<double> destination)
    {
        StridedKernels.MapTo(x, destination, default(MagnitudeOperator));
        return destination;
    }

    /// <summary>
    /// Computes the phase angle <c>atan2(im, re)</c> in <c>[-π, π]</c> of each element in a complex vector.
    /// <para>複素数ベクトルの各要素の偏角 <c>atan2(im, re)</c>（範囲 <c>[-π, π]</c>）を計算します。</para>
    /// </summary>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>A new vector containing the phase angles in radians. / 各要素の偏角（ラジアン）を格納した新しいベクトル。</returns>
    /// <remarks>
    /// <para>Matches <see cref="Complex.Phase"/> and distinguishes signed zero (<c>-1 + 0i</c> yields <c>π</c>, while <c>-1 - 0i</c> yields <c>-π</c>).</para>
    /// <para><see cref="Complex.Phase"/> と同じ値を返します。符号付きゼロを区別し、<c>-1 + 0i</c> は <c>π</c>、<c>-1 - 0i</c> は <c>-π</c> となります。</para>
    /// </remarks>
    public static Vector<double> Phase(ReadOnlyVectorSpan<Complex> x)
    {
        var result = Vector<double>.Uninitialized(x.Length);
        Phase(x, result);
        return result;
    }

    /// <summary>
    /// Computes the phase angle <c>atan2(im, re)</c> of each element in a complex vector and writes the result to <paramref name="destination"/>.
    /// <para>複素数ベクトルの各要素の偏角 <c>atan2(im, re)</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="destination">The destination span to receive the phase angles. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">The lengths differ, or <paramref name="destination"/> shares memory with <paramref name="x"/>. / 長さが一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と記憶域を共有する場合。</exception>
    public static VectorSpan<double> Phase(ReadOnlyVectorSpan<Complex> x, VectorSpan<double> destination)
    {
        StridedKernels.MapTo(x.AsMatrix(), destination.AsMatrix(), default(PhaseOperator));
        return destination;
    }

    /// <summary>
    /// Computes the phase angle <c>atan2(im, re)</c> in <c>[-π, π]</c> of each element in a complex matrix.
    /// <para>複素数行列の各要素の偏角 <c>atan2(im, re)</c>（範囲 <c>[-π, π]</c>）を計算します。</para>
    /// </summary>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <returns>A new matrix containing the phase angles in radians. / 各要素の偏角（ラジアン）を格納した新しい行列。</returns>
    /// <exception cref="OverflowException">The number of elements exceeds <see cref="int.MaxValue"/> (for example, when passing a large broadcast view). / 結果の要素数が <see cref="int.MaxValue"/> を超える場合（要素数が <c>int</c> に収まらないブロードキャストビューを渡した場合など）。</exception>
    public static Matrix<double> Phase(ReadOnlyMatrixSpan<Complex> x)
    {
        var result = Matrix<double>.Uninitialized(x.Rows, x.Columns);
        Phase(x, result);
        return result;
    }

    /// <summary>
    /// Computes the phase angle <c>atan2(im, re)</c> of each element in a complex matrix and writes the result to <paramref name="destination"/>.
    /// <para>複素数行列の各要素の偏角 <c>atan2(im, re)</c> を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input matrix. / 入力行列。</param>
    /// <param name="destination">The destination span to receive the phase angles. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">The shapes differ, or <paramref name="destination"/> shares memory with <paramref name="x"/>. / 形状が一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と記憶域を共有する場合。</exception>
    public static MatrixSpan<double> Phase(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<double> destination)
    {
        StridedKernels.MapTo(x, destination, default(PhaseOperator));
        return destination;
    }
}
