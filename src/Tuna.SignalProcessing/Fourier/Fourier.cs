using System.Numerics;

namespace Tuna.SignalProcessing;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

/// <summary>
/// Provides discrete Fourier transforms, discrete cosine and sine transforms, chirp-z transforms, and spectral utilities.
/// <para>離散フーリエ変換、離散コサイン・サイン変換、チャープ z 変換、およびスペクトル関連ユーティリティを提供します。</para>
/// </summary>
/// <remarks>
/// <para>The forward DFT uses the negative exponent <c>exp(-2πi·jk/n)</c> and defaults to <see cref="FourierNormalization.Backward"/> scaling (<c>1</c> forward, <c>1/n</c> inverse).</para>
/// <para>順方向 DFT は負の指数 <c>exp(−2πi·jk/n)</c> を用い、既定の正規化は <see cref="FourierNormalization.Backward"/>（順変換 <c>1</c>、逆変換 <c>1/n</c>）です。</para>
/// </remarks>
public static partial class Fourier
{
    /// <summary>
    /// Computes the forward discrete Fourier transform of a complex vector using <see cref="FourierNormalization.Backward"/>.
    /// <para>複素数ベクトルの順方向離散フーリエ変換を <see cref="FourierNormalization.Backward"/> で計算します。</para>
    /// </summary>
    /// <param name="x">The input complex vector. / 入力の複素数ベクトル。</param>
    /// <returns>A new vector containing the transform of <paramref name="x"/>. / 変換結果を格納した新しいベクトル。</returns>
    public static Vector<Complex> Forward(ReadOnlyVectorSpan<Complex> x) => Forward(x, FourierNormalization.Backward);

    /// <summary>
    /// Computes the forward discrete Fourier transform of a complex vector with the specified normalization.
    /// <para>指定した正規化方式で複素数ベクトルの順方向離散フーリエ変換を計算します。</para>
    /// </summary>
    /// <param name="x">The input complex vector. / 入力の複素数ベクトル。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <returns>A new vector containing the transform of <paramref name="x"/>. / 変換結果を格納した新しいベクトル。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static Vector<Complex> Forward(ReadOnlyVectorSpan<Complex> x, FourierNormalization normalization)
    {
        var result = Vector<Complex>.Uninitialized(x.Length);
        Forward(x, normalization, result);
        return result;
    }

    /// <summary>
    /// Computes the forward discrete Fourier transform of a complex vector using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>複素数ベクトルの順方向離散フーリエ変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex vector. / 入力の複素数ベクトル。</param>
    /// <param name="destination">The destination span with the same length as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ長さの出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different length or partially overlaps <paramref name="x"/>. / 出力先の長さが異なるか、入力と部分的に重なる場合。</exception>
    public static VectorSpan<Complex> Forward(ReadOnlyVectorSpan<Complex> x, VectorSpan<Complex> destination) =>
        Forward(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the forward discrete Fourier transform of a complex vector with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で複素数ベクトルの順方向離散フーリエ変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex vector. / 入力の複素数ベクトル。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The destination span with the same length as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ長さの出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different length or partially overlaps <paramref name="x"/>. / 出力先の長さが異なるか、入力と部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static VectorSpan<Complex> Forward(ReadOnlyVectorSpan<Complex> x, FourierNormalization normalization, VectorSpan<Complex> destination) =>
        FourierPlan.Create(x.Length).Forward(x, normalization, destination);

    /// <summary>
    /// Computes the inverse discrete Fourier transform of a complex vector using <see cref="FourierNormalization.Backward"/>.
    /// <para>複素数ベクトルの逆離散フーリエ変換を <see cref="FourierNormalization.Backward"/> で計算します。</para>
    /// </summary>
    /// <param name="x">The input complex spectrum. / 入力の複素数スペクトル。</param>
    /// <returns>A new vector containing the inverse transform of <paramref name="x"/>. / 逆変換結果を格納した新しいベクトル。</returns>
    public static Vector<Complex> Inverse(ReadOnlyVectorSpan<Complex> x) => Inverse(x, FourierNormalization.Backward);

    /// <summary>
    /// Computes the inverse discrete Fourier transform of a complex vector with the specified normalization.
    /// <para>指定した正規化方式で複素数ベクトルの逆離散フーリエ変換を計算します。</para>
    /// </summary>
    /// <param name="x">The input complex spectrum. / 入力の複素数スペクトル。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <returns>A new vector containing the inverse transform of <paramref name="x"/>. / 逆変換結果を格納した新しいベクトル。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static Vector<Complex> Inverse(ReadOnlyVectorSpan<Complex> x, FourierNormalization normalization)
    {
        var result = Vector<Complex>.Uninitialized(x.Length);
        Inverse(x, normalization, result);
        return result;
    }

    /// <summary>
    /// Computes the inverse discrete Fourier transform of a complex vector using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>複素数ベクトルの逆離散フーリエ変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex spectrum. / 入力の複素数スペクトル。</param>
    /// <param name="destination">The destination span with the same length as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ長さの出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different length or partially overlaps <paramref name="x"/>. / 出力先の長さが異なるか、入力と部分的に重なる場合。</exception>
    public static VectorSpan<Complex> Inverse(ReadOnlyVectorSpan<Complex> x, VectorSpan<Complex> destination) =>
        Inverse(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the inverse discrete Fourier transform of a complex vector with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で複素数ベクトルの逆離散フーリエ変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex spectrum. / 入力の複素数スペクトル。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The destination span with the same length as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ長さの出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different length or partially overlaps <paramref name="x"/>. / 出力先の長さが異なるか、入力と部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static VectorSpan<Complex> Inverse(ReadOnlyVectorSpan<Complex> x, FourierNormalization normalization, VectorSpan<Complex> destination) =>
        FourierPlan.Create(x.Length).Inverse(x, normalization, destination);

    /// <summary>
    /// Computes the forward discrete Fourier transform independently for each row of a complex matrix using <see cref="FourierNormalization.Backward"/>.
    /// <para>複素数行列の各行ごとに順方向離散フーリエ変換を <see cref="FourierNormalization.Backward"/> で計算します。</para>
    /// </summary>
    /// <param name="x">The input complex matrix. / 入力の複素数行列。</param>
    /// <returns>A new matrix with each row transformed. / 各行を変換した新しい行列。</returns>
    public static Matrix<Complex> ForwardRows(ReadOnlyMatrixSpan<Complex> x) => ForwardRows(x, FourierNormalization.Backward);

    /// <summary>
    /// Computes the forward discrete Fourier transform independently for each row of a complex matrix with the specified normalization.
    /// <para>指定した正規化方式で複素数行列の各行ごとに順方向離散フーリエ変換を計算します。</para>
    /// </summary>
    /// <param name="x">The input complex matrix. / 入力の複素数行列。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <returns>A new matrix with each row transformed. / 各行を変換した新しい行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static Matrix<Complex> ForwardRows(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization)
    {
        var result = Matrix<Complex>.Uninitialized(x.Rows, x.Columns);
        ForwardRows(x, normalization, result);
        return result;
    }

    /// <summary>
    /// Computes the forward discrete Fourier transform independently for each row of a complex matrix using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>複素数行列の各行ごとに順方向離散フーリエ変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex matrix. / 入力の複素数行列。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と部分的に重なる場合。</exception>
    public static MatrixSpan<Complex> ForwardRows(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination) =>
        ForwardRows(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the forward discrete Fourier transform independently for each row of a complex matrix with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で複素数行列の各行ごとに順方向離散フーリエ変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex matrix. / 入力の複素数行列。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static MatrixSpan<Complex> ForwardRows(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<Complex> destination) =>
        FourierPlan.Create(x.Columns).ForwardRows(x, normalization, destination);

    /// <summary>
    /// Computes the forward discrete Fourier transform independently for each column of a complex matrix using <see cref="FourierNormalization.Backward"/>.
    /// <para>複素数行列の各列ごとに順方向離散フーリエ変換を <see cref="FourierNormalization.Backward"/> で計算します。</para>
    /// </summary>
    /// <param name="x">The input complex matrix. / 入力の複素数行列。</param>
    /// <returns>A new matrix with each column transformed. / 各列を変換した新しい行列。</returns>
    public static Matrix<Complex> ForwardColumns(ReadOnlyMatrixSpan<Complex> x) => ForwardColumns(x, FourierNormalization.Backward);

    /// <summary>
    /// Computes the forward discrete Fourier transform independently for each column of a complex matrix with the specified normalization.
    /// <para>指定した正規化方式で複素数行列の各列ごとに順方向離散フーリエ変換を計算します。</para>
    /// </summary>
    /// <param name="x">The input complex matrix. / 入力の複素数行列。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <returns>A new matrix with each column transformed. / 各列を変換した新しい行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static Matrix<Complex> ForwardColumns(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization)
    {
        var result = Matrix<Complex>.Uninitialized(x.Rows, x.Columns);
        ForwardColumns(x, normalization, result);
        return result;
    }

    /// <summary>
    /// Computes the forward discrete Fourier transform independently for each column of a complex matrix using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>複素数行列の各列ごとに順方向離散フーリエ変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex matrix. / 入力の複素数行列。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と部分的に重なる場合。</exception>
    public static MatrixSpan<Complex> ForwardColumns(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination) =>
        ForwardColumns(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the forward discrete Fourier transform independently for each column of a complex matrix with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で複素数行列の各列ごとに順方向離散フーリエ変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex matrix. / 入力の複素数行列。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static MatrixSpan<Complex> ForwardColumns(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<Complex> destination) =>
        FourierPlan.Create(x.Rows).ForwardColumns(x, normalization, destination);

    /// <summary>
    /// Computes the inverse discrete Fourier transform independently for each row of a complex matrix using <see cref="FourierNormalization.Backward"/>.
    /// <para>複素数行列の各行ごとに逆離散フーリエ変換を <see cref="FourierNormalization.Backward"/> で計算します。</para>
    /// </summary>
    /// <param name="x">The input complex matrix. / 入力の複素数行列。</param>
    /// <returns>A new matrix with each row inverse-transformed. / 各行を逆変換した新しい行列。</returns>
    public static Matrix<Complex> InverseRows(ReadOnlyMatrixSpan<Complex> x) => InverseRows(x, FourierNormalization.Backward);

    /// <summary>
    /// Computes the inverse discrete Fourier transform independently for each row of a complex matrix with the specified normalization.
    /// <para>指定した正規化方式で複素数行列の各行ごとに逆離散フーリエ変換を計算します。</para>
    /// </summary>
    /// <param name="x">The input complex matrix. / 入力の複素数行列。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <returns>A new matrix with each row inverse-transformed. / 各行を逆変換した新しい行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static Matrix<Complex> InverseRows(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization)
    {
        var result = Matrix<Complex>.Uninitialized(x.Rows, x.Columns);
        InverseRows(x, normalization, result);
        return result;
    }

    /// <summary>
    /// Computes the inverse discrete Fourier transform independently for each row of a complex matrix using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>複素数行列の各行ごとに逆離散フーリエ変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex matrix. / 入力の複素数行列。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と部分的に重なる場合。</exception>
    public static MatrixSpan<Complex> InverseRows(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination) =>
        InverseRows(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the inverse discrete Fourier transform independently for each row of a complex matrix with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で複素数行列の各行ごとに逆離散フーリエ変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex matrix. / 入力の複素数行列。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static MatrixSpan<Complex> InverseRows(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<Complex> destination) =>
        FourierPlan.Create(x.Columns).InverseRows(x, normalization, destination);

    /// <summary>
    /// Computes the inverse discrete Fourier transform independently for each column of a complex matrix using <see cref="FourierNormalization.Backward"/>.
    /// <para>複素数行列の各列ごとに逆離散フーリエ変換を <see cref="FourierNormalization.Backward"/> で計算します。</para>
    /// </summary>
    /// <param name="x">The input complex matrix. / 入力の複素数行列。</param>
    /// <returns>A new matrix with each column inverse-transformed. / 各列を逆変換した新しい行列。</returns>
    public static Matrix<Complex> InverseColumns(ReadOnlyMatrixSpan<Complex> x) => InverseColumns(x, FourierNormalization.Backward);

    /// <summary>
    /// Computes the inverse discrete Fourier transform independently for each column of a complex matrix with the specified normalization.
    /// <para>指定した正規化方式で複素数行列の各列ごとに逆離散フーリエ変換を計算します。</para>
    /// </summary>
    /// <param name="x">The input complex matrix. / 入力の複素数行列。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <returns>A new matrix with each column inverse-transformed. / 各列を逆変換した新しい行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static Matrix<Complex> InverseColumns(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization)
    {
        var result = Matrix<Complex>.Uninitialized(x.Rows, x.Columns);
        InverseColumns(x, normalization, result);
        return result;
    }

    /// <summary>
    /// Computes the inverse discrete Fourier transform independently for each column of a complex matrix using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>複素数行列の各列ごとに逆離散フーリエ変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex matrix. / 入力の複素数行列。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と部分的に重なる場合。</exception>
    public static MatrixSpan<Complex> InverseColumns(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination) =>
        InverseColumns(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the inverse discrete Fourier transform independently for each column of a complex matrix with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で複素数行列の各列ごとに逆離散フーリエ変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex matrix. / 入力の複素数行列。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static MatrixSpan<Complex> InverseColumns(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<Complex> destination) =>
        FourierPlan.Create(x.Rows).InverseColumns(x, normalization, destination);

    /// <summary>
    /// Computes the one-sided forward real discrete Fourier transform of a real vector using <see cref="FourierNormalization.Backward"/>.
    /// <para>実数ベクトルの片側スペクトルを <see cref="FourierNormalization.Backward"/> で計算します。</para>
    /// </summary>
    /// <param name="x">The input real vector of length <c>n</c>. / 長さ <c>n</c> の入力実数ベクトル。</param>
    /// <returns>A new vector of length <c>0</c> (when <c>n = 0</c>) or <c>⌊n/2⌋ + 1</c> containing the one-sided spectrum. / 片側スペクトルを格納した新しいベクトル（<c>n = 0</c> のときは長さ 0、それ以外は <c>⌊n/2⌋ + 1</c>）。</returns>
    public static Vector<Complex> ForwardReal(ReadOnlyVectorSpan<double> x) => ForwardReal(x, FourierNormalization.Backward);

    /// <summary>
    /// Computes the one-sided forward real discrete Fourier transform of a real vector with the specified normalization.
    /// <para>指定した正規化方式で実数ベクトルの片側スペクトルを計算します。</para>
    /// </summary>
    /// <param name="x">The input real vector of length <c>n</c>. / 長さ <c>n</c> の入力実数ベクトル。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <returns>A new vector of length <c>0</c> (when <c>n = 0</c>) or <c>⌊n/2⌋ + 1</c> containing the one-sided spectrum. / 片側スペクトルを格納した新しいベクトル（<c>n = 0</c> のときは長さ 0、それ以外は <c>⌊n/2⌋ + 1</c>）。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static Vector<Complex> ForwardReal(ReadOnlyVectorSpan<double> x, FourierNormalization normalization)
    {
        var plan = RealFourierPlan.Create(x.Length);
        var result = Vector<Complex>.Uninitialized(plan.SpectrumLength);
        plan.Forward(x, normalization, result);
        return result;
    }

    /// <summary>
    /// Computes the one-sided forward real discrete Fourier transform using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>実数ベクトルの片側スペクトルを <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real vector of length <c>n</c>. / 長さ <c>n</c> の入力実数ベクトル。</param>
    /// <param name="destination">The non-overlapping destination span of length <c>0</c> (when <c>n = 0</c>) or <c>⌊n/2⌋ + 1</c>. / 入力と重ならない片側スペクトル長（<c>n = 0</c> なら 0、それ以外は <c>⌊n/2⌋ + 1</c>）の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has an invalid length or overlaps <paramref name="x"/>. / 出力先の長さが合わないか、入力と重なる場合。</exception>
    public static VectorSpan<Complex> ForwardReal(ReadOnlyVectorSpan<double> x, VectorSpan<Complex> destination) =>
        ForwardReal(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the one-sided forward real discrete Fourier transform with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で実数ベクトルの片側スペクトルを計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real vector of length <c>n</c>. / 長さ <c>n</c> の入力実数ベクトル。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The non-overlapping destination span of length <c>0</c> (when <c>n = 0</c>) or <c>⌊n/2⌋ + 1</c>. / 入力と重ならない片側スペクトル長（<c>n = 0</c> なら 0、それ以外は <c>⌊n/2⌋ + 1</c>）の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has an invalid length or overlaps <paramref name="x"/>. / 出力先の長さが合わないか、入力と重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static VectorSpan<Complex> ForwardReal(ReadOnlyVectorSpan<double> x, FourierNormalization normalization, VectorSpan<Complex> destination) =>
        RealFourierPlan.Create(x.Length).Forward(x, normalization, destination);

    /// <summary>
    /// Reconstructs a real vector of length <paramref name="length"/> from its one-sided spectrum using <see cref="FourierNormalization.Backward"/>.
    /// <para>片側スペクトルから長さ <paramref name="length"/> の実数ベクトルを <see cref="FourierNormalization.Backward"/> で逆変換します。</para>
    /// </summary>
    /// <param name="x">The input one-sided spectrum of length <c>0</c> (when <paramref name="length"/> is 0) or <c>⌊length/2⌋ + 1</c>. / 片側スペクトル（<paramref name="length"/> が 0 なら長さ 0、それ以外は <c>⌊length/2⌋ + 1</c>）。</param>
    /// <param name="length">The target real signal length (non-negative). / 復元する実数列の長さ（0 以上）。</param>
    /// <returns>A new real vector of length <paramref name="length"/>. / 復元した長さ <paramref name="length"/> の実数ベクトル。</returns>
    /// <remarks>
    /// <para>The imaginary part of bin 0 (and of the last bin when <paramref name="length"/> is even) is ignored.</para>
    /// <para>第 0 係数、および <paramref name="length"/> が偶数の場合の末尾係数の虚部は無視されます。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is negative. / <paramref name="length"/> が負の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="x"/> length does not match <paramref name="length"/>. / スペクトル長が <paramref name="length"/> と整合しない場合。</exception>
    public static Vector<double> InverseReal(ReadOnlyVectorSpan<Complex> x, int length) =>
        InverseReal(x, length, FourierNormalization.Backward);

    /// <summary>
    /// Reconstructs a real vector of length <paramref name="length"/> from its one-sided spectrum with the specified normalization.
    /// <para>指定した正規化方式で片側スペクトルから長さ <paramref name="length"/> の実数ベクトルを逆変換します。</para>
    /// </summary>
    /// <param name="x">The input one-sided spectrum of length <c>0</c> (when <paramref name="length"/> is 0) or <c>⌊length/2⌋ + 1</c>. / 片側スペクトル（<paramref name="length"/> が 0 なら長さ 0、それ以外は <c>⌊length/2⌋ + 1</c>）。</param>
    /// <param name="length">The target real signal length (non-negative). / 復元する実数列の長さ（0 以上）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <returns>A new real vector of length <paramref name="length"/>. / 復元した長さ <paramref name="length"/> の実数ベクトル。</returns>
    /// <remarks>
    /// <para>The imaginary part of bin 0 (and of the last bin when <paramref name="length"/> is even) is ignored.</para>
    /// <para>第 0 係数、および <paramref name="length"/> が偶数の場合の末尾係数の虚部は無視されます。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is negative or <paramref name="normalization"/> is invalid. / <paramref name="length"/> が負の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="x"/> length does not match <paramref name="length"/>. / スペクトル長が <paramref name="length"/> と整合しない場合。</exception>
    public static Vector<double> InverseReal(ReadOnlyVectorSpan<Complex> x, int length, FourierNormalization normalization)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        var result = Vector<double>.Uninitialized(length);
        InverseReal(x, normalization, result);
        return result;
    }

    /// <summary>
    /// Reconstructs a real vector from its one-sided spectrum using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>片側スペクトルから実数ベクトルを <see cref="FourierNormalization.Backward"/> で逆変換し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input one-sided spectrum matching <paramref name="destination"/>.<c>Length</c>. / <paramref name="destination"/> の長さに対応する片側スペクトル。</param>
    /// <param name="destination">The non-overlapping destination span whose length determines the reconstructed signal length. / 入力と重ならない出力先（この長さが復元長になります）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>The imaginary part of bin 0 (and of the last bin when <paramref name="destination"/>.<c>Length</c> is even) is ignored.</para>
    /// <para>第 0 係数、および出力長が偶数の場合の末尾係数の虚部は無視されます。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="x"/> length does not match <paramref name="destination"/>.<c>Length</c>, or they overlap. / スペクトル長が出力長と整合しない場合、または入出力が重なる場合。</exception>
    public static VectorSpan<double> InverseReal(ReadOnlyVectorSpan<Complex> x, VectorSpan<double> destination) =>
        InverseReal(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Reconstructs a real vector from its one-sided spectrum with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で片側スペクトルから実数ベクトルを逆変換し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input one-sided spectrum matching <paramref name="destination"/>.<c>Length</c>. / <paramref name="destination"/> の長さに対応する片側スペクトル。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The non-overlapping destination span whose length determines the reconstructed signal length. / 入力と重ならない出力先（この長さが復元長になります）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>The imaginary part of bin 0 (and of the last bin when <paramref name="destination"/>.<c>Length</c> is even) is ignored.</para>
    /// <para>第 0 係数、および出力長が偶数の場合の末尾係数の虚部は無視されます。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="x"/> length does not match <paramref name="destination"/>.<c>Length</c>, or they overlap. / スペクトル長が出力長と整合しない場合、または入出力が重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static VectorSpan<double> InverseReal(ReadOnlyVectorSpan<Complex> x, FourierNormalization normalization, VectorSpan<double> destination) =>
        RealFourierPlan.Create(destination.Length).Inverse(x, normalization, destination);

    /// <summary>
    /// Computes the one-sided forward real discrete Fourier transform independently for each row of a matrix using <see cref="FourierNormalization.Backward"/>.
    /// <para>実数行列の各行ごとに片側スペクトルを <see cref="FourierNormalization.Backward"/> で計算します。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <returns>A new matrix of row spectra with <c>0</c> columns (when <c>x.Columns = 0</c>) or <c>⌊x.Columns/2⌋ + 1</c> columns. / 各行の片側スペクトルを格納した新しい行列。</returns>
    public static Matrix<Complex> ForwardRealRows(ReadOnlyMatrixSpan<double> x) => ForwardRealRows(x, FourierNormalization.Backward);

    /// <summary>
    /// Computes the one-sided forward real discrete Fourier transform independently for each row of a matrix with the specified normalization.
    /// <para>指定した正規化方式で実数行列の各行ごとに片側スペクトルを計算します。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <returns>A new matrix of row spectra with <c>0</c> columns (when <c>x.Columns = 0</c>) or <c>⌊x.Columns/2⌋ + 1</c> columns. / 各行の片側スペクトルを格納した新しい行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static Matrix<Complex> ForwardRealRows(ReadOnlyMatrixSpan<double> x, FourierNormalization normalization)
    {
        var plan = RealFourierPlan.Create(x.Columns);
        var result = Matrix<Complex>.Uninitialized(x.Rows, plan.SpectrumLength);
        plan.ForwardRows(x, normalization, result);
        return result;
    }

    /// <summary>
    /// Computes the one-sided forward real discrete Fourier transform independently for each row using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>実数行列の各行ごとに片側スペクトルを <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="destination">The non-overlapping destination matrix with the same row count and spectrum column count. / 入力と重ならない片側スペクトル形状の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has an invalid shape or overlaps <paramref name="x"/>. / 出力先の形状が合わないか、入力と重なる場合。</exception>
    public static MatrixSpan<Complex> ForwardRealRows(ReadOnlyMatrixSpan<double> x, MatrixSpan<Complex> destination) =>
        ForwardRealRows(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the one-sided forward real discrete Fourier transform independently for each row with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で実数行列の各行ごとに片側スペクトルを計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The non-overlapping destination matrix with the same row count and spectrum column count. / 入力と重ならない片側スペクトル形状の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has an invalid shape or overlaps <paramref name="x"/>. / 出力先の形状が合わないか、入力と重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static MatrixSpan<Complex> ForwardRealRows(ReadOnlyMatrixSpan<double> x, FourierNormalization normalization, MatrixSpan<Complex> destination) =>
        RealFourierPlan.Create(x.Columns).ForwardRows(x, normalization, destination);

    /// <summary>
    /// Computes the one-sided forward real discrete Fourier transform independently for each column of a matrix using <see cref="FourierNormalization.Backward"/>.
    /// <para>実数行列の各列ごとに片側スペクトルを <see cref="FourierNormalization.Backward"/> で計算します。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <returns>A new matrix of column spectra with <c>0</c> rows (when <c>x.Rows = 0</c>) or <c>⌊x.Rows/2⌋ + 1</c> rows. / 各列の片側スペクトルを格納した新しい行列。</returns>
    public static Matrix<Complex> ForwardRealColumns(ReadOnlyMatrixSpan<double> x) => ForwardRealColumns(x, FourierNormalization.Backward);

    /// <summary>
    /// Computes the one-sided forward real discrete Fourier transform independently for each column of a matrix with the specified normalization.
    /// <para>指定した正規化方式で実数行列の各列ごとに片側スペクトルを計算します。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <returns>A new matrix of column spectra with <c>0</c> rows (when <c>x.Rows = 0</c>) or <c>⌊x.Rows/2⌋ + 1</c> rows. / 各列の片側スペクトルを格納した新しい行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static Matrix<Complex> ForwardRealColumns(ReadOnlyMatrixSpan<double> x, FourierNormalization normalization)
    {
        var plan = RealFourierPlan.Create(x.Rows);
        var result = Matrix<Complex>.Uninitialized(plan.SpectrumLength, x.Columns);
        plan.ForwardColumns(x, normalization, result);
        return result;
    }

    /// <summary>
    /// Computes the one-sided forward real discrete Fourier transform independently for each column using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>実数行列の各列ごとに片側スペクトルを <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="destination">The non-overlapping destination matrix with spectrum row count and the same column count. / 入力と重ならない片側スペクトル形状の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has an invalid shape or overlaps <paramref name="x"/>. / 出力先の形状が合わないか、入力と重なる場合。</exception>
    public static MatrixSpan<Complex> ForwardRealColumns(ReadOnlyMatrixSpan<double> x, MatrixSpan<Complex> destination) =>
        ForwardRealColumns(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the one-sided forward real discrete Fourier transform independently for each column with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で実数行列の各列ごとに片側スペクトルを計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The non-overlapping destination matrix with spectrum row count and the same column count. / 入力と重ならない片側スペクトル形状の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has an invalid shape or overlaps <paramref name="x"/>. / 出力先の形状が合わないか、入力と重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static MatrixSpan<Complex> ForwardRealColumns(ReadOnlyMatrixSpan<double> x, FourierNormalization normalization, MatrixSpan<Complex> destination) =>
        RealFourierPlan.Create(x.Rows).ForwardColumns(x, normalization, destination);

    /// <summary>
    /// Reconstructs real rows of length <paramref name="length"/> from their one-sided spectra using <see cref="FourierNormalization.Backward"/>.
    /// <para>行列の各行の片側スペクトルから長さ <paramref name="length"/> の実数列を <see cref="FourierNormalization.Backward"/> で逆変換します。</para>
    /// </summary>
    /// <param name="x">The input row-wise one-sided spectra. / 行ごとの片側スペクトル。</param>
    /// <param name="length">The target row length (non-negative). / 復元する各行の長さ（0 以上）。</param>
    /// <returns>A new <c>x.Rows × length</c> real matrix. / 復元した <c>x.Rows × length</c> の実数行列。</returns>
    /// <remarks>
    /// <para>The imaginary part of bin 0 (and of the last bin when <paramref name="length"/> is even) is ignored.</para>
    /// <para>第 0 係数、および <paramref name="length"/> が偶数の場合の末尾係数の虚部は無視されます。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is negative. / <paramref name="length"/> が負の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="x"/> column count does not match <paramref name="length"/>. / スペクトル列数が <paramref name="length"/> と整合しない場合。</exception>
    public static Matrix<double> InverseRealRows(ReadOnlyMatrixSpan<Complex> x, int length) =>
        InverseRealRows(x, length, FourierNormalization.Backward);

    /// <summary>
    /// Reconstructs real rows of length <paramref name="length"/> from their one-sided spectra with the specified normalization.
    /// <para>指定した正規化方式で行列の各行の片側スペクトルから長さ <paramref name="length"/> の実数列を逆変換します。</para>
    /// </summary>
    /// <param name="x">The input row-wise one-sided spectra. / 行ごとの片側スペクトル。</param>
    /// <param name="length">The target row length (non-negative). / 復元する各行の長さ（0 以上）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <returns>A new <c>x.Rows × length</c> real matrix. / 復元した <c>x.Rows × length</c> の実数行列。</returns>
    /// <remarks>
    /// <para>The imaginary part of bin 0 (and of the last bin when <paramref name="length"/> is even) is ignored.</para>
    /// <para>第 0 係数、および <paramref name="length"/> が偶数の場合の末尾係数の虚部は無視されます。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is negative or <paramref name="normalization"/> is invalid. / <paramref name="length"/> が負の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="x"/> column count does not match <paramref name="length"/>. / スペクトル列数が <paramref name="length"/> と整合しない場合。</exception>
    public static Matrix<double> InverseRealRows(ReadOnlyMatrixSpan<Complex> x, int length, FourierNormalization normalization)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        var result = Matrix<double>.Uninitialized(x.Rows, length);
        InverseRealRows(x, normalization, result);
        return result;
    }

    /// <summary>
    /// Reconstructs real rows from their one-sided spectra using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>行列の各行の片側スペクトルから実数列を <see cref="FourierNormalization.Backward"/> で逆変換し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input row-wise one-sided spectra. / 行ごとの片側スペクトル。</param>
    /// <param name="destination">The non-overlapping destination matrix whose column count determines the reconstructed row length. / 入力と重ならない出力先（列数が復元長になります）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>The imaginary part of bin 0 (and of the last bin when <paramref name="destination"/>.<c>Columns</c> is even) is ignored.</para>
    /// <para>第 0 係数、および出力列数が偶数の場合の末尾係数の虚部は無視されます。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Matrix shapes are inconsistent, or <paramref name="destination"/> overlaps <paramref name="x"/>. / 行列の形状が整合しない場合、または入出力が重なる場合。</exception>
    public static MatrixSpan<double> InverseRealRows(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<double> destination) =>
        InverseRealRows(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Reconstructs real rows from their one-sided spectra with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で行列の各行の片側スペクトルから実数列を逆変換し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input row-wise one-sided spectra. / 行ごとの片側スペクトル。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The non-overlapping destination matrix whose column count determines the reconstructed row length. / 入力と重ならない出力先（列数が復元長になります）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>The imaginary part of bin 0 (and of the last bin when <paramref name="destination"/>.<c>Columns</c> is even) is ignored.</para>
    /// <para>第 0 係数、および出力列数が偶数の場合の末尾係数の虚部は無視されます。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Matrix shapes are inconsistent, or <paramref name="destination"/> overlaps <paramref name="x"/>. / 行列の形状が整合しない場合、または入出力が重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static MatrixSpan<double> InverseRealRows(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<double> destination) =>
        RealFourierPlan.Create(destination.Columns).InverseRows(x, normalization, destination);

    /// <summary>
    /// Reconstructs real columns of length <paramref name="length"/> from their one-sided spectra using <see cref="FourierNormalization.Backward"/>.
    /// <para>行列の各列の片側スペクトルから長さ <paramref name="length"/> の実数列を <see cref="FourierNormalization.Backward"/> で逆変換します。</para>
    /// </summary>
    /// <param name="x">The input column-wise one-sided spectra. / 列ごとの片側スペクトル。</param>
    /// <param name="length">The target column length (non-negative). / 復元する各列の長さ（0 以上）。</param>
    /// <returns>A new <c>length × x.Columns</c> real matrix. / 復元した <c>length × x.Columns</c> の実数行列。</returns>
    /// <remarks>
    /// <para>The imaginary part of bin 0 (and of the last bin when <paramref name="length"/> is even) is ignored.</para>
    /// <para>第 0 係数、および <paramref name="length"/> が偶数の場合の末尾係数の虚部は無視されます。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is negative. / <paramref name="length"/> が負の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="x"/> row count does not match <paramref name="length"/>. / スペクトル行数が <paramref name="length"/> と整合しない場合。</exception>
    public static Matrix<double> InverseRealColumns(ReadOnlyMatrixSpan<Complex> x, int length) =>
        InverseRealColumns(x, length, FourierNormalization.Backward);

    /// <summary>
    /// Reconstructs real columns of length <paramref name="length"/> from their one-sided spectra with the specified normalization.
    /// <para>指定した正規化方式で行列の各列の片側スペクトルから長さ <paramref name="length"/> の実数列を逆変換します。</para>
    /// </summary>
    /// <param name="x">The input column-wise one-sided spectra. / 列ごとの片側スペクトル。</param>
    /// <param name="length">The target column length (non-negative). / 復元する各列の長さ（0 以上）。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <returns>A new <c>length × x.Columns</c> real matrix. / 復元した <c>length × x.Columns</c> の実数行列。</returns>
    /// <remarks>
    /// <para>The imaginary part of bin 0 (and of the last bin when <paramref name="length"/> is even) is ignored.</para>
    /// <para>第 0 係数、および <paramref name="length"/> が偶数の場合の末尾係数の虚部は無視されます。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is negative or <paramref name="normalization"/> is invalid. / <paramref name="length"/> が負の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="x"/> row count does not match <paramref name="length"/>. / スペクトル行数が <paramref name="length"/> と整合しない場合。</exception>
    public static Matrix<double> InverseRealColumns(ReadOnlyMatrixSpan<Complex> x, int length, FourierNormalization normalization)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        var result = Matrix<double>.Uninitialized(length, x.Columns);
        InverseRealColumns(x, normalization, result);
        return result;
    }

    /// <summary>
    /// Reconstructs real columns from their one-sided spectra using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>行列の各列の片側スペクトルから実数列を <see cref="FourierNormalization.Backward"/> で逆変換し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input column-wise one-sided spectra. / 列ごとの片側スペクトル。</param>
    /// <param name="destination">The non-overlapping destination matrix whose row count determines the reconstructed column length. / 入力と重ならない出力先（行数が復元長になります）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>The imaginary part of bin 0 (and of the last bin when <paramref name="destination"/>.<c>Rows</c> is even) is ignored.</para>
    /// <para>第 0 係数、および出力行数が偶数の場合の末尾係数の虚部は無視されます。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Matrix shapes are inconsistent, or <paramref name="destination"/> overlaps <paramref name="x"/>. / 行列の形状が整合しない場合、または入出力が重なる場合。</exception>
    public static MatrixSpan<double> InverseRealColumns(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<double> destination) =>
        InverseRealColumns(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Reconstructs real columns from their one-sided spectra with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で行列の各列の片側スペクトルから実数列を逆変換し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input column-wise one-sided spectra. / 列ごとの片側スペクトル。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The non-overlapping destination matrix whose row count determines the reconstructed column length. / 入力と重ならない出力先（行数が復元長になります）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>The imaginary part of bin 0 (and of the last bin when <paramref name="destination"/>.<c>Rows</c> is even) is ignored.</para>
    /// <para>第 0 係数、および出力行数が偶数の場合の末尾係数の虚部は無視されます。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Matrix shapes are inconsistent, or <paramref name="destination"/> overlaps <paramref name="x"/>. / 行列の形状が整合しない場合、または入出力が重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static MatrixSpan<double> InverseRealColumns(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<double> destination) =>
        RealFourierPlan.Create(destination.Rows).InverseColumns(x, normalization, destination);

    /// <summary>
    /// Provides in-place Fourier and trigonometric transforms that overwrite the input buffer (<c>x ← F(x)</c>).
    /// <para>入力バッファを直接書き換えるインプレースのフーリエ変換および三角変換を提供します（<c>x ← F(x)</c>）。</para>
    /// </summary>
    public static partial class InPlace
    {
        /// <summary>
        /// Computes the forward discrete Fourier transform of a complex vector in place using <see cref="FourierNormalization.Backward"/>.
        /// <para>複素数ベクトルの順方向離散フーリエ変換を <see cref="FourierNormalization.Backward"/> でインプレース計算します。</para>
        /// </summary>
        /// <param name="x">The complex vector to transform in place. / インプレースで変換する複素数ベクトル。</param>
        public static void Forward(VectorSpan<Complex> x) => Fourier.Forward(x, FourierNormalization.Backward, x);

        /// <summary>
        /// Computes the forward discrete Fourier transform of a complex vector in place with the specified normalization.
        /// <para>指定した正規化方式で複素数ベクトルの順方向離散フーリエ変換をインプレース計算します。</para>
        /// </summary>
        /// <param name="x">The complex vector to transform in place. / インプレースで変換する複素数ベクトル。</param>
        /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
        public static void Forward(VectorSpan<Complex> x, FourierNormalization normalization) => Fourier.Forward(x, normalization, x);

        /// <summary>
        /// Computes the inverse discrete Fourier transform of a complex vector in place using <see cref="FourierNormalization.Backward"/>.
        /// <para>複素数ベクトルの逆離散フーリエ変換を <see cref="FourierNormalization.Backward"/> でインプレース計算します。</para>
        /// </summary>
        /// <param name="x">The complex vector to inverse-transform in place. / インプレースで逆変換する複素数ベクトル。</param>
        public static void Inverse(VectorSpan<Complex> x) => Fourier.Inverse(x, FourierNormalization.Backward, x);

        /// <summary>
        /// Computes the inverse discrete Fourier transform of a complex vector in place with the specified normalization.
        /// <para>指定した正規化方式で複素数ベクトルの逆離散フーリエ変換をインプレース計算します。</para>
        /// </summary>
        /// <param name="x">The complex vector to inverse-transform in place. / インプレースで逆変換する複素数ベクトル。</param>
        /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
        public static void Inverse(VectorSpan<Complex> x, FourierNormalization normalization) => Fourier.Inverse(x, normalization, x);

        /// <summary>
        /// Computes the forward discrete Fourier transform independently for each row of a complex matrix in place using <see cref="FourierNormalization.Backward"/>.
        /// <para>複素数行列の各行ごとに順方向離散フーリエ変換を <see cref="FourierNormalization.Backward"/> でインプレース計算します。</para>
        /// </summary>
        /// <param name="x">The complex matrix to transform in place. / インプレースで変換する複素数行列。</param>
        public static void ForwardRows(MatrixSpan<Complex> x) => Fourier.ForwardRows(x, FourierNormalization.Backward, x);

        /// <summary>
        /// Computes the forward discrete Fourier transform independently for each row of a complex matrix in place with the specified normalization.
        /// <para>指定した正規化方式で複素数行列の各行ごとに順方向離散フーリエ変換をインプレース計算します。</para>
        /// </summary>
        /// <param name="x">The complex matrix to transform in place. / インプレースで変換する複素数行列。</param>
        /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
        public static void ForwardRows(MatrixSpan<Complex> x, FourierNormalization normalization) => Fourier.ForwardRows(x, normalization, x);

        /// <summary>
        /// Computes the forward discrete Fourier transform independently for each column of a complex matrix in place using <see cref="FourierNormalization.Backward"/>.
        /// <para>複素数行列の各列ごとに順方向離散フーリエ変換を <see cref="FourierNormalization.Backward"/> でインプレース計算します。</para>
        /// </summary>
        /// <param name="x">The complex matrix to transform in place. / インプレースで変換する複素数行列。</param>
        public static void ForwardColumns(MatrixSpan<Complex> x) => Fourier.ForwardColumns(x, FourierNormalization.Backward, x);

        /// <summary>
        /// Computes the forward discrete Fourier transform independently for each column of a complex matrix in place with the specified normalization.
        /// <para>指定した正規化方式で複素数行列の各列ごとに順方向離散フーリエ変換をインプレース計算します。</para>
        /// </summary>
        /// <param name="x">The complex matrix to transform in place. / インプレースで変換する複素数行列。</param>
        /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
        public static void ForwardColumns(MatrixSpan<Complex> x, FourierNormalization normalization) => Fourier.ForwardColumns(x, normalization, x);

        /// <summary>
        /// Computes the inverse discrete Fourier transform independently for each row of a complex matrix in place using <see cref="FourierNormalization.Backward"/>.
        /// <para>複素数行列の各行ごとに逆離散フーリエ変換を <see cref="FourierNormalization.Backward"/> でインプレース計算します。</para>
        /// </summary>
        /// <param name="x">The complex matrix to inverse-transform in place. / インプレースで逆変換する複素数行列。</param>
        public static void InverseRows(MatrixSpan<Complex> x) => Fourier.InverseRows(x, FourierNormalization.Backward, x);

        /// <summary>
        /// Computes the inverse discrete Fourier transform independently for each row of a complex matrix in place with the specified normalization.
        /// <para>指定した正規化方式で複素数行列の各行ごとに逆離散フーリエ変換をインプレース計算します。</para>
        /// </summary>
        /// <param name="x">The complex matrix to inverse-transform in place. / インプレースで逆変換する複素数行列。</param>
        /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
        public static void InverseRows(MatrixSpan<Complex> x, FourierNormalization normalization) => Fourier.InverseRows(x, normalization, x);

        /// <summary>
        /// Computes the inverse discrete Fourier transform independently for each column of a complex matrix in place using <see cref="FourierNormalization.Backward"/>.
        /// <para>複素数行列の各列ごとに逆離散フーリエ変換を <see cref="FourierNormalization.Backward"/> でインプレース計算します。</para>
        /// </summary>
        /// <param name="x">The complex matrix to inverse-transform in place. / インプレースで逆変換する複素数行列。</param>
        public static void InverseColumns(MatrixSpan<Complex> x) => Fourier.InverseColumns(x, FourierNormalization.Backward, x);

        /// <summary>
        /// Computes the inverse discrete Fourier transform independently for each column of a complex matrix in place with the specified normalization.
        /// <para>指定した正規化方式で複素数行列の各列ごとに逆離散フーリエ変換をインプレース計算します。</para>
        /// </summary>
        /// <param name="x">The complex matrix to inverse-transform in place. / インプレースで逆変換する複素数行列。</param>
        /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
        public static void InverseColumns(MatrixSpan<Complex> x, FourierNormalization normalization) => Fourier.InverseColumns(x, normalization, x);
    }
}
