using System.Buffers;
using System.Numerics;

namespace Tuna.SignalProcessing;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

public static partial class Fourier
{
    /// <summary>
    /// Computes the two-dimensional forward discrete Fourier transform of a complex matrix using <see cref="FourierNormalization.Backward"/>.
    /// <para>複素数行列の 2 次元順方向離散フーリエ変換を <see cref="FourierNormalization.Backward"/> で計算します。</para>
    /// </summary>
    /// <param name="x">The input complex matrix. / 入力の複素数行列。</param>
    /// <returns>A new matrix containing the 2D transform of <paramref name="x"/>. / 2 次元変換結果を格納した新しい行列。</returns>
    /// <remarks>
    /// <para>Transforms rows first and then columns, applying the normalization factor along each axis.</para>
    /// <para>各行を変換してから各列を変換し、正規化係数は変換軸ごとに適用します。</para>
    /// </remarks>
    public static Matrix<Complex> Forward2D(ReadOnlyMatrixSpan<Complex> x) => Forward2D(x, FourierNormalization.Backward);

    /// <summary>
    /// Computes the two-dimensional forward discrete Fourier transform of a complex matrix with the specified normalization.
    /// <para>指定した正規化方式で複素数行列の 2 次元順方向離散フーリエ変換を計算します。</para>
    /// </summary>
    /// <param name="x">The input complex matrix. / 入力の複素数行列。</param>
    /// <param name="normalization">The scaling convention applied to each axis. / 各軸に適用する正規化方式。</param>
    /// <returns>A new matrix containing the 2D transform of <paramref name="x"/>. / 2 次元変換結果を格納した新しい行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static Matrix<Complex> Forward2D(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization)
    {
        var result = Matrix<Complex>.Uninitialized(x.Rows, x.Columns);
        Forward2D(x, normalization, result);
        return result;
    }

    /// <summary>
    /// Computes the two-dimensional forward discrete Fourier transform using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>複素数行列の 2 次元順方向離散フーリエ変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex matrix. / 入力の複素数行列。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と部分的に重なる場合。</exception>
    public static MatrixSpan<Complex> Forward2D(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination) =>
        Forward2D(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the two-dimensional forward discrete Fourier transform with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で複素数行列の 2 次元順方向離散フーリエ変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex matrix. / 入力の複素数行列。</param>
    /// <param name="normalization">The scaling convention applied to each axis. / 各軸に適用する正規化方式。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static MatrixSpan<Complex> Forward2D(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<Complex> destination)
    {
        ForwardRows(x, normalization, destination);
        ForwardColumns(destination, normalization, destination);
        return destination;
    }

    /// <summary>
    /// Computes the two-dimensional inverse discrete Fourier transform of a complex matrix using <see cref="FourierNormalization.Backward"/>.
    /// <para>複素数行列の 2 次元逆離散フーリエ変換を <see cref="FourierNormalization.Backward"/> で計算します。</para>
    /// </summary>
    /// <param name="x">The input 2D complex spectrum. / 入力の 2 次元複素数スペクトル。</param>
    /// <returns>A new matrix containing the 2D inverse transform of <paramref name="x"/>. / 2 次元逆変換結果を格納した新しい行列。</returns>
    public static Matrix<Complex> Inverse2D(ReadOnlyMatrixSpan<Complex> x) => Inverse2D(x, FourierNormalization.Backward);

    /// <summary>
    /// Computes the two-dimensional inverse discrete Fourier transform of a complex matrix with the specified normalization.
    /// <para>指定した正規化方式で複素数行列の 2 次元逆離散フーリエ変換を計算します。</para>
    /// </summary>
    /// <param name="x">The input 2D complex spectrum. / 入力の 2 次元複素数スペクトル。</param>
    /// <param name="normalization">The scaling convention applied to each axis. / 各軸に適用する正規化方式。</param>
    /// <returns>A new matrix containing the 2D inverse transform of <paramref name="x"/>. / 2 次元逆変換結果を格納した新しい行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static Matrix<Complex> Inverse2D(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization)
    {
        var result = Matrix<Complex>.Uninitialized(x.Rows, x.Columns);
        Inverse2D(x, normalization, result);
        return result;
    }

    /// <summary>
    /// Computes the two-dimensional inverse discrete Fourier transform using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>複素数行列の 2 次元逆離散フーリエ変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input 2D complex spectrum. / 入力の 2 次元複素数スペクトル。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と部分的に重なる場合。</exception>
    public static MatrixSpan<Complex> Inverse2D(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination) =>
        Inverse2D(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the two-dimensional inverse discrete Fourier transform with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で複素数行列の 2 次元逆離散フーリエ変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input 2D complex spectrum. / 入力の 2 次元複素数スペクトル。</param>
    /// <param name="normalization">The scaling convention applied to each axis. / 各軸に適用する正規化方式。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or partially overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static MatrixSpan<Complex> Inverse2D(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<Complex> destination)
    {
        InverseColumns(x, normalization, destination);
        InverseRows(destination, normalization, destination);
        return destination;
    }

    /// <summary>
    /// Computes the two-dimensional forward real discrete Fourier transform using <see cref="FourierNormalization.Backward"/>, returning the one-sided spectrum along the column axis.
    /// <para>実数行列の 2 次元順方向実数 FFT を <see cref="FourierNormalization.Backward"/> で計算し、列方向に片側スペクトルを返します。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <returns>A new matrix of shape <c>x.Rows × (x.Columns == 0 ? 0 : x.Columns / 2 + 1)</c>. / 列方向に片側スペクトルを格納した新しい行列。</returns>
    public static Matrix<Complex> ForwardReal2D(ReadOnlyMatrixSpan<double> x) => ForwardReal2D(x, FourierNormalization.Backward);

    /// <summary>
    /// Computes the two-dimensional forward real discrete Fourier transform with the specified normalization, returning the one-sided spectrum along the column axis.
    /// <para>指定した正規化方式で実数行列の 2 次元順方向実数 FFT を計算し、列方向に片側スペクトルを返します。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="normalization">The scaling convention applied to each axis. / 各軸に適用する正規化方式。</param>
    /// <returns>A new matrix of shape <c>x.Rows × (x.Columns == 0 ? 0 : x.Columns / 2 + 1)</c>. / 列方向に片側スペクトルを格納した新しい行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static Matrix<Complex> ForwardReal2D(ReadOnlyMatrixSpan<double> x, FourierNormalization normalization)
    {
        var result = Matrix<Complex>.Uninitialized(x.Rows, x.Columns == 0 ? 0 : x.Columns / 2 + 1);
        ForwardReal2D(x, normalization, result);
        return result;
    }

    /// <summary>
    /// Computes the two-dimensional forward real discrete Fourier transform using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>実数行列の 2 次元順方向実数 FFT を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="destination">The non-overlapping destination matrix with the same row count and one-sided spectrum column count. / 入力と重ならない片側スペクトル形状の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has an invalid shape or overlaps <paramref name="x"/>. / 出力先の形状が合わないか、入力と重なる場合。</exception>
    public static MatrixSpan<Complex> ForwardReal2D(ReadOnlyMatrixSpan<double> x, MatrixSpan<Complex> destination) =>
        ForwardReal2D(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the two-dimensional forward real discrete Fourier transform with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で実数行列の 2 次元順方向実数 FFT を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real matrix. / 入力の実数行列。</param>
    /// <param name="normalization">The scaling convention applied to each axis. / 各軸に適用する正規化方式。</param>
    /// <param name="destination">The non-overlapping destination matrix with the same row count and one-sided spectrum column count. / 入力と重ならない片側スペクトル形状の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has an invalid shape or overlaps <paramref name="x"/>. / 出力先の形状が合わないか、入力と重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static MatrixSpan<Complex> ForwardReal2D(ReadOnlyMatrixSpan<double> x, FourierNormalization normalization, MatrixSpan<Complex> destination)
    {
        ForwardRealRows(x, normalization, destination);
        ForwardColumns(destination, normalization, destination);
        return destination;
    }

    /// <summary>
    /// Reconstructs a real matrix with <paramref name="columns"/> columns from its two-dimensional one-sided spectrum using <see cref="FourierNormalization.Backward"/>.
    /// <para>2 次元の片側スペクトルから列数 <paramref name="columns"/> の実数行列を <see cref="FourierNormalization.Backward"/> で逆変換します。</para>
    /// </summary>
    /// <param name="x">The input 2D one-sided spectrum. / 入力の 2 次元片側スペクトル。</param>
    /// <param name="columns">The target column count (non-negative). / 復元する実数行列の列数（0 以上）。</param>
    /// <returns>A new <c>x.Rows × columns</c> real matrix. / 復元した <c>x.Rows × columns</c> の実数行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="columns"/> is negative. / <paramref name="columns"/> が負の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="x"/> column count does not match <paramref name="columns"/>. / スペクトル列数が <paramref name="columns"/> と整合しない場合。</exception>
    public static Matrix<double> InverseReal2D(ReadOnlyMatrixSpan<Complex> x, int columns) =>
        InverseReal2D(x, columns, FourierNormalization.Backward);

    /// <summary>
    /// Reconstructs a real matrix with <paramref name="columns"/> columns from its two-dimensional one-sided spectrum with the specified normalization.
    /// <para>指定した正規化方式で 2 次元の片側スペクトルから列数 <paramref name="columns"/> の実数行列を逆変換します。</para>
    /// </summary>
    /// <param name="x">The input 2D one-sided spectrum. / 入力の 2 次元片側スペクトル。</param>
    /// <param name="columns">The target column count (non-negative). / 復元する実数行列の列数（0 以上）。</param>
    /// <param name="normalization">The scaling convention applied to each axis. / 各軸に適用する正規化方式。</param>
    /// <returns>A new <c>x.Rows × columns</c> real matrix. / 復元した <c>x.Rows × columns</c> の実数行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="columns"/> is negative or <paramref name="normalization"/> is invalid. / <paramref name="columns"/> が負の場合、または <paramref name="normalization"/> が無効な場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="x"/> column count does not match <paramref name="columns"/>. / スペクトル列数が <paramref name="columns"/> と整合しない場合。</exception>
    public static Matrix<double> InverseReal2D(ReadOnlyMatrixSpan<Complex> x, int columns, FourierNormalization normalization)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(columns);
        var result = Matrix<double>.Uninitialized(x.Rows, columns);
        InverseReal2D(x, normalization, result);
        return result;
    }

    /// <summary>
    /// Reconstructs a real matrix from its two-dimensional one-sided spectrum using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>2 次元の片側スペクトルから実数行列を <see cref="FourierNormalization.Backward"/> で逆変換し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input 2D one-sided spectrum. / 入力の 2 次元片側スペクトル。</param>
    /// <param name="destination">The destination matrix whose column count determines the reconstructed width. May overlap <paramref name="x"/>. / 出力先（列数が復元幅になります。<paramref name="x"/> と重なっても構いません）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">Matrix shapes are inconsistent. / 行列の形状が整合しない場合。</exception>
    public static MatrixSpan<double> InverseReal2D(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<double> destination) =>
        InverseReal2D(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Reconstructs a real matrix from its two-dimensional one-sided spectrum with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で 2 次元の片側スペクトルから実数行列を逆変換し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input 2D one-sided spectrum. / 入力の 2 次元片側スペクトル。</param>
    /// <param name="normalization">The scaling convention applied to each axis. / 各軸に適用する正規化方式。</param>
    /// <param name="destination">The destination matrix whose column count determines the reconstructed width. May overlap <paramref name="x"/>. / 出力先（列数が復元幅になります。<paramref name="x"/> と重なっても構いません）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">Matrix shapes are inconsistent. / 行列の形状が整合しない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public static MatrixSpan<double> InverseReal2D(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<double> destination)
    {
        // 列逆変換を中間バッファへ書くことで、入力を変更せず destination の重なりも安全に許容する。
        int rows = x.Rows, spectrum = x.Columns;
        if (destination.Rows != rows)
        {
            throw new ArgumentException($"The destination must have {rows} rows, but it has {destination.Rows}.", nameof(destination));
        }

        Complex[] buffer = ArrayPool<Complex>.Shared.Rent(Math.Max(checked(rows * spectrum), 1));
        var work = new MatrixSpan<Complex>(buffer.AsSpan(0, rows * spectrum), rows, spectrum);
        InverseColumns(x, normalization, work);
        InverseRealRows(work, normalization, destination);
        ArrayPool<Complex>.Shared.Return(buffer);
        return destination;
    }

    /// <summary>
    /// Shifts the zero-frequency component of a 2D spectrum to the center along both axes.
    /// <para>2 次元スペクトルの周波数 0 成分を両軸の中央へ移動します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input 2D spectrum matrix. / 変換係数の行列。</param>
    /// <returns>A new matrix cyclically shifted by <c>⌊R/2⌋</c> rows and <c>⌊C/2⌋</c> columns. / 行を <c>⌊R/2⌋</c>、列を <c>⌊C/2⌋</c> だけ巡回シフトした新しい行列。</returns>
    /// <remarks>
    /// <para>Equivalent to applying <see cref="ShiftColumns{T}(ReadOnlyMatrixSpan{T})"/> followed by <see cref="ShiftRows{T}(ReadOnlyMatrixSpan{T})"/>. Reversed by <see cref="InverseShift2D{T}(ReadOnlyMatrixSpan{T})"/>.</para>
    /// <para><see cref="ShiftColumns{T}(ReadOnlyMatrixSpan{T})"/> と <see cref="ShiftRows{T}(ReadOnlyMatrixSpan{T})"/> を連続適用した結果と一致し、<see cref="InverseShift2D{T}(ReadOnlyMatrixSpan{T})"/> で元に戻ります。</para>
    /// </remarks>
    [InferenceOverloads]
    public static Matrix<T> Shift2D<T>(ReadOnlyMatrixSpan<T> x) where T : unmanaged, INumberBase<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        Shift2D(x, result);
        return result;
    }

    /// <summary>
    /// Shifts the zero-frequency component of a 2D spectrum to the center along both axes and writes the result to <paramref name="destination"/>.
    /// <para>2 次元スペクトルの周波数 0 成分を両軸の中央へ移動し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input 2D spectrum matrix. / 変換係数の行列。</param>
    /// <param name="destination">The non-overlapping destination matrix with the same shape as <paramref name="x"/>. / 入力と同じ形状で重ならない出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と重なる場合。</exception>
    [InferenceOverloads]
    public static MatrixSpan<T> Shift2D<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination) where T : unmanaged, INumberBase<T>
    {
        Rotate2D(x, destination, (x.Rows + 1) / 2, (x.Columns + 1) / 2);
        return destination;
    }

    /// <summary>
    /// Reverses <see cref="Shift2D{T}(ReadOnlyMatrixSpan{T})"/>, moving the zero-frequency component back to <c>(0, 0)</c>.
    /// <para><see cref="Shift2D{T}(ReadOnlyMatrixSpan{T})"/> の逆操作を行い、周波数 0 成分を先頭 <c>(0, 0)</c> へ戻します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The shifted 2D spectrum matrix. / シフトされた 2 次元スペクトル行列。</param>
    /// <returns>A new matrix with the zero-frequency component restored to <c>(0, 0)</c>. / 周波数 0 成分を先頭へ戻した新しい行列。</returns>
    [InferenceOverloads]
    public static Matrix<T> InverseShift2D<T>(ReadOnlyMatrixSpan<T> x) where T : unmanaged, INumberBase<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        InverseShift2D(x, result);
        return result;
    }

    /// <summary>
    /// Reverses <see cref="Shift2D{T}(ReadOnlyMatrixSpan{T}, MatrixSpan{T})"/> and writes the result to <paramref name="destination"/>.
    /// <para>2 次元スペクトルの周波数 0 成分を先頭 <c>(0, 0)</c> へ戻し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The shifted 2D spectrum matrix. / シフトされた 2 次元スペクトル行列。</param>
    /// <param name="destination">The non-overlapping destination matrix with the same shape as <paramref name="x"/>. / 入力と同じ形状で重ならない出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と重なる場合。</exception>
    [InferenceOverloads]
    public static MatrixSpan<T> InverseShift2D<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination) where T : unmanaged, INumberBase<T>
    {
        Rotate2D(x, destination, x.Rows / 2, x.Columns / 2);
        return destination;
    }

    private static void Rotate2D<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination, int rowSplit, int columnSplit)
        where T : unmanaged, INumberBase<T>
    {
        if (x.Rows != destination.Rows || x.Columns != destination.Columns)
        {
            throw new ArgumentException("Matrix dimensions must agree.", nameof(destination));
        }

        var output = destination.AsReadOnly();
        if (StridedLayout.Overlaps(
                in x.Reference, x.Rows, x.Columns, x.RowStride, x.ColumnStride,
                in output.Reference, output.Rows, output.Columns, output.RowStride, output.ColumnStride, out _))
        {
            throw new ArgumentException("The destination must not overlap the input.", nameof(destination));
        }

        int r = x.Rows - rowSplit, c = x.Columns - columnSplit;
        x[rowSplit.., columnSplit..].CopyTo(destination[..r, ..c]);
        x[rowSplit.., ..columnSplit].CopyTo(destination[..r, c..]);
        x[..rowSplit, columnSplit..].CopyTo(destination[r.., ..c]);
        x[..rowSplit, ..columnSplit].CopyTo(destination[r.., c..]);
    }

    public static partial class InPlace
    {
        /// <summary>
        /// Computes the two-dimensional forward discrete Fourier transform of a complex matrix in place using <see cref="FourierNormalization.Backward"/>.
        /// <para>複素数行列の 2 次元順方向離散フーリエ変換を <see cref="FourierNormalization.Backward"/> でインプレース計算します。</para>
        /// </summary>
        /// <param name="x">The complex matrix to transform in place. / インプレースで変換する複素数行列。</param>
        public static void Forward2D(MatrixSpan<Complex> x) => Fourier.Forward2D(x, FourierNormalization.Backward, x);

        /// <summary>
        /// Computes the two-dimensional forward discrete Fourier transform of a complex matrix in place with the specified normalization.
        /// <para>指定した正規化方式で複素数行列の 2 次元順方向離散フーリエ変換をインプレース計算します。</para>
        /// </summary>
        /// <param name="x">The complex matrix to transform in place. / インプレースで変換する複素数行列。</param>
        /// <param name="normalization">The scaling convention applied to each axis. / 各軸に適用する正規化方式。</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
        public static void Forward2D(MatrixSpan<Complex> x, FourierNormalization normalization) => Fourier.Forward2D(x, normalization, x);

        /// <summary>
        /// Computes the two-dimensional inverse discrete Fourier transform of a complex matrix in place using <see cref="FourierNormalization.Backward"/>.
        /// <para>複素数行列の 2 次元逆離散フーリエ変換を <see cref="FourierNormalization.Backward"/> でインプレース計算します。</para>
        /// </summary>
        /// <param name="x">The complex matrix to inverse-transform in place. / インプレースで逆変換する複素数行列。</param>
        public static void Inverse2D(MatrixSpan<Complex> x) => Fourier.Inverse2D(x, FourierNormalization.Backward, x);

        /// <summary>
        /// Computes the two-dimensional inverse discrete Fourier transform of a complex matrix in place with the specified normalization.
        /// <para>指定した正規化方式で複素数行列の 2 次元逆離散フーリエ変換をインプレース計算します。</para>
        /// </summary>
        /// <param name="x">The complex matrix to inverse-transform in place. / インプレースで逆変換する複素数行列。</param>
        /// <param name="normalization">The scaling convention applied to each axis. / 各軸に適用する正規化方式。</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
        public static void Inverse2D(MatrixSpan<Complex> x, FourierNormalization normalization) => Fourier.Inverse2D(x, normalization, x);
    }
}
