using System.Numerics;

namespace Tuna.SignalProcessing;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

/// <summary>
/// Represents a reusable plan for computing discrete Fourier transforms of complex sequences of a fixed length.
/// <para>固定長の複素数列に対する離散フーリエ変換を繰り返し計算するための実行計画を表します。</para>
/// </summary>
/// <remarks>
/// <para>Instances are immutable after construction and safe to use concurrently from multiple threads.</para>
/// <para>構築後は不変であり、複数スレッドから安全に同時利用できます。</para>
/// </remarks>
public sealed partial class FourierPlan
{
    /// <summary>
    /// Gets or creates a complex Fourier transform plan for the specified length.
    /// <para>指定した長さの複素フーリエ変換計画を取得または作成します。</para>
    /// </summary>
    /// <param name="length">The transform length (non-negative). / 変換長（0 以上）。</param>
    /// <returns>A reusable plan for <paramref name="length"/>; cached instances may be shared. / 指定した長さの変換計画（キャッシュされたインスタンスが共有される場合があります）。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is negative. / <paramref name="length"/> が負の場合。</exception>
    public static FourierPlan Create(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        return FourierPlanCache.GetComplex(length, static n => new FourierPlan(n));
    }

    /// <summary>
    /// Gets the transform length.
    /// <para>変換の長さを取得します。</para>
    /// </summary>
    public int Length { get; }

    /// <summary>
    /// Computes the forward discrete Fourier transform of a complex vector using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>複素数ベクトルの順方向離散フーリエ変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex vector of length <see cref="Length"/>. / 長さ <see cref="Length"/> の入力複素数ベクトル。</param>
    /// <param name="destination">The destination span of length <see cref="Length"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 長さ <see cref="Length"/> の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> or <paramref name="destination"/> does not match <see cref="Length"/>, or they partially overlap. / 長さが <see cref="Length"/> と一致しない場合、または入出力が部分的に重なる場合。</exception>
    public VectorSpan<Complex> Forward(ReadOnlyVectorSpan<Complex> x, VectorSpan<Complex> destination) =>
        Forward(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the forward discrete Fourier transform of a complex vector with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で複素数ベクトルの順方向離散フーリエ変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex vector of length <see cref="Length"/>. / 長さ <see cref="Length"/> の入力複素数ベクトル。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The destination span of length <see cref="Length"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 長さ <see cref="Length"/> の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> or <paramref name="destination"/> does not match <see cref="Length"/>, or they partially overlap. / 長さが <see cref="Length"/> と一致しない場合、または入出力が部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public VectorSpan<Complex> Forward(ReadOnlyVectorSpan<Complex> x, FourierNormalization normalization, VectorSpan<Complex> destination)
    {
        FourierViews.ComplexRows<ForwardDirection>(
            this, FourierViews.AsRow(x), FourierViews.AsRow(destination), FourierScaling.Forward(normalization, Length));
        return destination;
    }

    /// <summary>
    /// Computes the inverse discrete Fourier transform of a complex vector using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>複素数ベクトルの逆離散フーリエ変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex vector of length <see cref="Length"/>. / 長さ <see cref="Length"/> の入力複素数ベクトル。</param>
    /// <param name="destination">The destination span of length <see cref="Length"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 長さ <see cref="Length"/> の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> or <paramref name="destination"/> does not match <see cref="Length"/>, or they partially overlap. / 長さが <see cref="Length"/> と一致しない場合、または入出力が部分的に重なる場合。</exception>
    public VectorSpan<Complex> Inverse(ReadOnlyVectorSpan<Complex> x, VectorSpan<Complex> destination) =>
        Inverse(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the inverse discrete Fourier transform of a complex vector with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で複素数ベクトルの逆離散フーリエ変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex vector of length <see cref="Length"/>. / 長さ <see cref="Length"/> の入力複素数ベクトル。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The destination span of length <see cref="Length"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 長さ <see cref="Length"/> の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="x"/> or <paramref name="destination"/> does not match <see cref="Length"/>, or they partially overlap. / 長さが <see cref="Length"/> と一致しない場合、または入出力が部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public VectorSpan<Complex> Inverse(ReadOnlyVectorSpan<Complex> x, FourierNormalization normalization, VectorSpan<Complex> destination)
    {
        FourierViews.ComplexRows<InverseDirection>(
            this, FourierViews.AsRow(x), FourierViews.AsRow(destination), FourierScaling.Inverse(normalization, Length));
        return destination;
    }

    /// <summary>
    /// Computes the forward discrete Fourier transform independently for each row of a matrix using <see cref="FourierNormalization.Backward"/>.
    /// <para>行列の各行ごとに順方向離散フーリエ変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex matrix with <see cref="Length"/> columns. / 列数が <see cref="Length"/> の入力複素数行列。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">Matrix shapes disagree, <c>x.Columns != Length</c>, or <paramref name="destination"/> partially overlaps <paramref name="x"/>. / 形状が一致しない場合、列数が <see cref="Length"/> と異なる場合、または入出力が部分的に重なる場合。</exception>
    public MatrixSpan<Complex> ForwardRows(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination) =>
        ForwardRows(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the forward discrete Fourier transform independently for each row of a matrix with the specified normalization.
    /// <para>指定した正規化方式で行列の各行ごとに順方向離散フーリエ変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex matrix with <see cref="Length"/> columns. / 列数が <see cref="Length"/> の入力複素数行列。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">Matrix shapes disagree, <c>x.Columns != Length</c>, or <paramref name="destination"/> partially overlaps <paramref name="x"/>. / 形状が一致しない場合、列数が <see cref="Length"/> と異なる場合、または入出力が部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public MatrixSpan<Complex> ForwardRows(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<Complex> destination)
    {
        FourierViews.ComplexRows<ForwardDirection>(this, x, destination, FourierScaling.Forward(normalization, Length));
        return destination;
    }

    /// <summary>
    /// Computes the forward discrete Fourier transform independently for each column of a matrix using <see cref="FourierNormalization.Backward"/>.
    /// <para>行列の各列ごとに順方向離散フーリエ変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex matrix with <see cref="Length"/> rows. / 行数が <see cref="Length"/> の入力複素数行列。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">Matrix shapes disagree, <c>x.Rows != Length</c>, or <paramref name="destination"/> partially overlaps <paramref name="x"/>. / 形状が一致しない場合、行数が <see cref="Length"/> と異なる場合、または入出力が部分的に重なる場合。</exception>
    public MatrixSpan<Complex> ForwardColumns(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination) =>
        ForwardColumns(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the forward discrete Fourier transform independently for each column of a matrix with the specified normalization.
    /// <para>指定した正規化方式で行列の各列ごとに順方向離散フーリエ変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex matrix with <see cref="Length"/> rows. / 行数が <see cref="Length"/> の入力複素数行列。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">Matrix shapes disagree, <c>x.Rows != Length</c>, or <paramref name="destination"/> partially overlaps <paramref name="x"/>. / 形状が一致しない場合、行数が <see cref="Length"/> と異なる場合、または入出力が部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public MatrixSpan<Complex> ForwardColumns(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<Complex> destination)
    {
        FourierViews.ComplexRows<ForwardDirection>(this, x.T, destination.T, FourierScaling.Forward(normalization, Length));
        return destination;
    }

    /// <summary>
    /// Computes the inverse discrete Fourier transform independently for each row of a matrix using <see cref="FourierNormalization.Backward"/>.
    /// <para>行列の各行ごとに逆離散フーリエ変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex matrix with <see cref="Length"/> columns. / 列数が <see cref="Length"/> の入力複素数行列。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">Matrix shapes disagree, <c>x.Columns != Length</c>, or <paramref name="destination"/> partially overlaps <paramref name="x"/>. / 形状が一致しない場合、列数が <see cref="Length"/> と異なる場合、または入出力が部分的に重なる場合。</exception>
    public MatrixSpan<Complex> InverseRows(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination) =>
        InverseRows(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the inverse discrete Fourier transform independently for each row of a matrix with the specified normalization.
    /// <para>指定した正規化方式で行列の各行ごとに逆離散フーリエ変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex matrix with <see cref="Length"/> columns. / 列数が <see cref="Length"/> の入力複素数行列。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">Matrix shapes disagree, <c>x.Columns != Length</c>, or <paramref name="destination"/> partially overlaps <paramref name="x"/>. / 形状が一致しない場合、列数が <see cref="Length"/> と異なる場合、または入出力が部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public MatrixSpan<Complex> InverseRows(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<Complex> destination)
    {
        FourierViews.ComplexRows<InverseDirection>(this, x, destination, FourierScaling.Inverse(normalization, Length));
        return destination;
    }

    /// <summary>
    /// Computes the inverse discrete Fourier transform independently for each column of a matrix using <see cref="FourierNormalization.Backward"/>.
    /// <para>行列の各列ごとに逆離散フーリエ変換を <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex matrix with <see cref="Length"/> rows. / 行数が <see cref="Length"/> の入力複素数行列。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">Matrix shapes disagree, <c>x.Rows != Length</c>, or <paramref name="destination"/> partially overlaps <paramref name="x"/>. / 形状が一致しない場合、行数が <see cref="Length"/> と異なる場合、または入出力が部分的に重なる場合。</exception>
    public MatrixSpan<Complex> InverseColumns(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<Complex> destination) =>
        InverseColumns(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the inverse discrete Fourier transform independently for each column of a matrix with the specified normalization.
    /// <para>指定した正規化方式で行列の各列ごとに逆離散フーリエ変換を計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input complex matrix with <see cref="Length"/> rows. / 行数が <see cref="Length"/> の入力複素数行列。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The destination matrix with the same shape as <paramref name="x"/>, either identical to <paramref name="x"/> (in-place) or non-overlapping. / 入力と同じ形状の出力先（入力と同一ビューまたは重ならない領域）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">Matrix shapes disagree, <c>x.Rows != Length</c>, or <paramref name="destination"/> partially overlaps <paramref name="x"/>. / 形状が一致しない場合、行数が <see cref="Length"/> と異なる場合、または入出力が部分的に重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public MatrixSpan<Complex> InverseColumns(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<Complex> destination)
    {
        FourierViews.ComplexRows<InverseDirection>(this, x.T, destination.T, FourierScaling.Inverse(normalization, Length));
        return destination;
    }
}
