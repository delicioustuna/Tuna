using System.Numerics;

namespace Tuna.SignalProcessing;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

public static partial class Fourier
{
    /// <summary>
    /// Returns the discrete Fourier transform sample frequencies for a complex transform of length <paramref name="length"/>.
    /// <para>長さ <paramref name="length"/> の複素 FFT の各スペクトル係数に対応する周波数を返します。</para>
    /// </summary>
    /// <param name="length">The signal length <c>n</c> (non-negative). / 変換する信号の長さ <c>n</c>（0 以上）。</param>
    /// <param name="sampleSpacing">The sample spacing <c>d</c>. Frequencies are returned in units of <c>1/d</c>. / 標本間隔 <c>d</c>。周波数の単位は <c>1/d</c> です。</param>
    /// <returns>A vector of frequencies <c>[0, 1, …, ⌈n/2⌉ - 1, -⌊n/2⌋, …, -1] / (d·n)</c>, or an empty vector when <paramref name="length"/> is 0. / 周波数ベクトル <c>[0, 1, …, ⌈n/2⌉ − 1, −⌊n/2⌋, …, −1] / (d·n)</c>（長さ 0 の場合は空ベクトル）。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is negative. / <paramref name="length"/> が負の場合。</exception>
    public static Vector<double> Frequencies(int length, double sampleSpacing = 1)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        var result = Vector<double>.Uninitialized(length);
        var span = result.AsSpan();
        double unit = 1.0 / (length * sampleSpacing);
        int positive = (length - 1) / 2 + 1;
        for (int k = 0; k < length; k++)
        {
            span[k] = (k < positive ? k : k - length) * unit;
        }

        return result;
    }

    /// <summary>
    /// Returns the discrete Fourier transform sample frequencies for the one-sided spectrum of a real transform of length <paramref name="length"/>.
    /// <para>長さ <paramref name="length"/> の実数 FFT の片側スペクトルに対応する周波数を返します。</para>
    /// </summary>
    /// <param name="length">The real signal length <c>n</c> (non-negative). / 変換する実数信号の長さ <c>n</c>（0 以上）。</param>
    /// <param name="sampleSpacing">The sample spacing <c>d</c>. Frequencies are returned in units of <c>1/d</c>. / 標本間隔 <c>d</c>。周波数の単位は <c>1/d</c> です。</param>
    /// <returns>A vector of frequencies <c>[0, 1, …, ⌊n/2⌋] / (d·n)</c>, or an empty vector when <paramref name="length"/> is 0. / 周波数ベクトル <c>[0, 1, …, ⌊n/2⌋] / (d·n)</c>（長さ 0 の場合は空ベクトル）。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is negative. / <paramref name="length"/> が負の場合。</exception>
    public static Vector<double> RealFrequencies(int length, double sampleSpacing = 1)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        int count = length == 0 ? 0 : length / 2 + 1;
        var result = Vector<double>.Uninitialized(count);
        var span = result.AsSpan();
        double unit = 1.0 / (length * sampleSpacing);
        for (int k = 0; k < count; k++)
        {
            span[k] = k * unit;
        }

        return result;
    }

    /// <summary>
    /// Returns the smallest FFT-friendly length greater than or equal to <paramref name="length"/> whose prime factors are limited to 2, 3, 5, and 7.
    /// <para><paramref name="length"/> 以上で素因数が 2・3・5・7 のみから成る最小の FFT 向け長さを返します。</para>
    /// </summary>
    /// <param name="length">The minimum required length (non-negative). / 必要な長さの下限（0 以上）。</param>
    /// <returns>The smallest 7-smooth integer greater than or equal to <paramref name="length"/>, or <c>1</c> when <paramref name="length"/> is 0 or 1. / <paramref name="length"/> 以上で素因数が 2・3・5・7 のみの最小の整数（0 または 1 の場合は 1）。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is negative, or the result exceeds <see cref="int.MaxValue"/>. / <paramref name="length"/> が負の場合、または結果が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static int NextFastLength(int length) => FourierPlan.NextFastLength(length);

    /// <summary>
    /// Shifts the zero-frequency component of a vector to the center of the spectrum.
    /// <para>ベクトルの周波数 0 成分をスペクトルの中央へ移動します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input spectrum vector. / 変換係数のベクトル。</param>
    /// <returns>A new vector with elements cyclically shifted right by <c>⌊n/2⌋</c>. / 要素を後ろへ <c>⌊n/2⌋</c> だけ巡回シフトした新しいベクトル。</returns>
    /// <remarks>
    /// <para>Use <see cref="InverseShift{T}(ReadOnlyVectorSpan{T})"/> to reverse this permutation.</para>
    /// <para>元の順序に戻すには <see cref="InverseShift{T}(ReadOnlyVectorSpan{T})"/> を使用します。</para>
    /// </remarks>
    [InferenceOverloads]
    public static Vector<T> Shift<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, INumberBase<T>
    {
        var result = Vector<T>.Uninitialized(x.Length);
        Shift(x, result);
        return result;
    }

    /// <summary>
    /// Shifts the zero-frequency component of a vector to the center of the spectrum and writes the result to <paramref name="destination"/>.
    /// <para>ベクトルの周波数 0 成分を中央へ移動し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input spectrum vector. / 変換係数のベクトル。</param>
    /// <param name="destination">The non-overlapping destination span with the same length as <paramref name="x"/>. / 入力と同じ長さで重ならない出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different length or overlaps <paramref name="x"/>. / 出力先の長さが異なるか、入力と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> Shift<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination) where T : unmanaged, INumberBase<T>
    {
        RotateColumns(FourierViews.AsRow(x), FourierViews.AsRow(destination), (x.Length + 1) / 2);
        return destination;
    }

    /// <summary>
    /// Reverses <see cref="Shift{T}(ReadOnlyVectorSpan{T})"/>, moving the zero-frequency component back to index 0.
    /// <para><see cref="Shift{T}(ReadOnlyVectorSpan{T})"/> の逆操作を行い、周波数 0 成分を先頭へ戻します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The shifted spectrum vector. / 中央へシフトされた変換係数のベクトル。</param>
    /// <returns>A new vector with the zero-frequency component at index 0. / 周波数 0 成分を先頭に戻した新しいベクトル。</returns>
    [InferenceOverloads]
    public static Vector<T> InverseShift<T>(ReadOnlyVectorSpan<T> x) where T : unmanaged, INumberBase<T>
    {
        var result = Vector<T>.Uninitialized(x.Length);
        InverseShift(x, result);
        return result;
    }

    /// <summary>
    /// Reverses <see cref="Shift{T}(ReadOnlyVectorSpan{T}, VectorSpan{T})"/> and writes the result to <paramref name="destination"/>.
    /// <para>周波数 0 成分を先頭へ戻し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The shifted spectrum vector. / 中央へシフトされた変換係数のベクトル。</param>
    /// <param name="destination">The non-overlapping destination span with the same length as <paramref name="x"/>. / 入力と同じ長さで重ならない出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different length or overlaps <paramref name="x"/>. / 出力先の長さが異なるか、入力と重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> InverseShift<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination) where T : unmanaged, INumberBase<T>
    {
        RotateColumns(FourierViews.AsRow(x), FourierViews.AsRow(destination), x.Length / 2);
        return destination;
    }

    /// <summary>
    /// Shifts the zero-frequency component of each row to the horizontal center.
    /// <para>行列の各行について周波数 0 成分を中央へ移動します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input spectrum matrix. / 変換係数の行列。</param>
    /// <returns>A new matrix with each row shifted. / 各行をシフトした新しい行列。</returns>
    [InferenceOverloads]
    public static Matrix<T> ShiftRows<T>(ReadOnlyMatrixSpan<T> x) where T : unmanaged, INumberBase<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        ShiftRows(x, result);
        return result;
    }

    /// <summary>
    /// Shifts the zero-frequency component of each row to the horizontal center and writes the result to <paramref name="destination"/>.
    /// <para>行列の各行について周波数 0 成分を中央へ移動し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input spectrum matrix. / 変換係数の行列。</param>
    /// <param name="destination">The non-overlapping destination matrix with the same shape as <paramref name="x"/>. / 入力と同じ形状で重ならない出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と重なる場合。</exception>
    [InferenceOverloads]
    public static MatrixSpan<T> ShiftRows<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination) where T : unmanaged, INumberBase<T>
    {
        RotateColumns(x, destination, (x.Columns + 1) / 2);
        return destination;
    }

    /// <summary>
    /// Shifts the zero-frequency component of each column to the vertical center.
    /// <para>行列の各列について周波数 0 成分を中央へ移動します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input spectrum matrix. / 変換係数の行列。</param>
    /// <returns>A new matrix with each column shifted. / 各列をシフトした新しい行列。</returns>
    [InferenceOverloads]
    public static Matrix<T> ShiftColumns<T>(ReadOnlyMatrixSpan<T> x) where T : unmanaged, INumberBase<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        ShiftColumns(x, result);
        return result;
    }

    /// <summary>
    /// Shifts the zero-frequency component of each column to the vertical center and writes the result to <paramref name="destination"/>.
    /// <para>行列の各列について周波数 0 成分を中央へ移動し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input spectrum matrix. / 変換係数の行列。</param>
    /// <param name="destination">The non-overlapping destination matrix with the same shape as <paramref name="x"/>. / 入力と同じ形状で重ならない出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と重なる場合。</exception>
    [InferenceOverloads]
    public static MatrixSpan<T> ShiftColumns<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination) where T : unmanaged, INumberBase<T>
    {
        RotateColumns(x.T, destination.T, (x.Rows + 1) / 2);
        return destination;
    }

    /// <summary>
    /// Reverses <see cref="ShiftRows{T}(ReadOnlyMatrixSpan{T})"/>, moving the zero-frequency component of each row back to column 0.
    /// <para>行列の各行について周波数 0 成分を先頭列へ戻します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The shifted spectrum matrix. / シフトされた変換係数の行列。</param>
    /// <returns>A new matrix with each row restored. / 各行を元に戻した新しい行列。</returns>
    [InferenceOverloads]
    public static Matrix<T> InverseShiftRows<T>(ReadOnlyMatrixSpan<T> x) where T : unmanaged, INumberBase<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        InverseShiftRows(x, result);
        return result;
    }

    /// <summary>
    /// Reverses <see cref="ShiftRows{T}(ReadOnlyMatrixSpan{T}, MatrixSpan{T})"/> and writes the result to <paramref name="destination"/>.
    /// <para>行列の各行について周波数 0 成分を先頭列へ戻し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The shifted spectrum matrix. / シフトされた変換係数の行列。</param>
    /// <param name="destination">The non-overlapping destination matrix with the same shape as <paramref name="x"/>. / 入力と同じ形状で重ならない出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と重なる場合。</exception>
    [InferenceOverloads]
    public static MatrixSpan<T> InverseShiftRows<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination) where T : unmanaged, INumberBase<T>
    {
        RotateColumns(x, destination, x.Columns / 2);
        return destination;
    }

    /// <summary>
    /// Reverses <see cref="ShiftColumns{T}(ReadOnlyMatrixSpan{T})"/>, moving the zero-frequency component of each column back to row 0.
    /// <para>行列の各列について周波数 0 成分を先頭行へ戻します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The shifted spectrum matrix. / シフトされた変換係数の行列。</param>
    /// <returns>A new matrix with each column restored. / 各列を元に戻した新しい行列。</returns>
    [InferenceOverloads]
    public static Matrix<T> InverseShiftColumns<T>(ReadOnlyMatrixSpan<T> x) where T : unmanaged, INumberBase<T>
    {
        var result = Matrix<T>.Uninitialized(x.Rows, x.Columns);
        InverseShiftColumns(x, result);
        return result;
    }

    /// <summary>
    /// Reverses <see cref="ShiftColumns{T}(ReadOnlyMatrixSpan{T}, MatrixSpan{T})"/> and writes the result to <paramref name="destination"/>.
    /// <para>行列の各列について周波数 0 成分を先頭行へ戻し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The shifted spectrum matrix. / シフトされた変換係数の行列。</param>
    /// <param name="destination">The non-overlapping destination matrix with the same shape as <paramref name="x"/>. / 入力と同じ形状で重ならない出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has a different shape or overlaps <paramref name="x"/>. / 出力先の形状が異なるか、入力と重なる場合。</exception>
    [InferenceOverloads]
    public static MatrixSpan<T> InverseShiftColumns<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination) where T : unmanaged, INumberBase<T>
    {
        RotateColumns(x.T, destination.T, x.Rows / 2);
        return destination;
    }

    private static void RotateColumns<T>(ReadOnlyMatrixSpan<T> x, MatrixSpan<T> destination, int split) where T : unmanaged, INumberBase<T>
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

        int n = x.Columns;
        x[.., split..].CopyTo(destination[.., ..(n - split)]);
        x[.., ..split].CopyTo(destination[.., (n - split)..]);
    }
}
