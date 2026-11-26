using System.Numerics;

namespace Tuna.SignalProcessing;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

/// <summary>
/// Represents a reusable plan for computing discrete Fourier transforms of real-valued sequences of a fixed length.
/// <para>固定長の実数列に対する離散フーリエ変換を繰り返し計算するための実行計画を表します。</para>
/// </summary>
/// <remarks>
/// <para>Instances are immutable after construction and safe to use concurrently from multiple threads.</para>
/// <para>構築後は不変であり、複数スレッドから安全に同時利用できます。</para>
/// </remarks>
public sealed partial class RealFourierPlan
{
    /// <summary>
    /// Gets or creates a real Fourier transform plan for the specified length.
    /// <para>指定した長さの実数フーリエ変換計画を取得または作成します。</para>
    /// </summary>
    /// <param name="length">The real signal length (non-negative). / 実数信号の長さ（0 以上）。</param>
    /// <returns>A reusable plan for <paramref name="length"/>; cached instances may be shared. / 指定した長さの変換計画（キャッシュされたインスタンスが共有される場合があります）。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is negative. / <paramref name="length"/> が負の場合。</exception>
    public static RealFourierPlan Create(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        return FourierPlanCache.GetReal(length, static n => new RealFourierPlan(n));
    }

    /// <summary>
    /// Gets the length of the real-valued sequence.
    /// <para>実数列の長さ <c>n</c> を取得します。</para>
    /// </summary>
    public int Length { get; }

    /// <summary>
    /// Gets the length of the one-sided Hermitian-symmetric spectrum.
    /// <para>片側スペクトルの長さを取得します。</para>
    /// </summary>
    /// <value><c>0</c> when <see cref="Length"/> is 0; otherwise <c>⌊n/2⌋ + 1</c>. / <see cref="Length"/> が 0 の場合は 0、それ以外は <c>⌊n/2⌋ + 1</c>。</value>
    public int SpectrumLength => Length == 0 ? 0 : Length / 2 + 1;

    /// <summary>
    /// Computes the one-sided forward real discrete Fourier transform using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>実数ベクトルの片側スペクトルを <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real vector of length <see cref="Length"/>. / 長さ <see cref="Length"/> の入力実数ベクトル。</param>
    /// <param name="destination">The non-overlapping destination span of length <see cref="SpectrumLength"/>. / 入力と重ならない長さ <see cref="SpectrumLength"/> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">The input or destination length does not match the plan, or they overlap. / 入力または出力先の長さが計画と一致しない場合、または入出力が重なる場合。</exception>
    public VectorSpan<Complex> Forward(ReadOnlyVectorSpan<double> x, VectorSpan<Complex> destination) =>
        Forward(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the one-sided forward real discrete Fourier transform with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で実数ベクトルの片側スペクトルを計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real vector of length <see cref="Length"/>. / 長さ <see cref="Length"/> の入力実数ベクトル。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The non-overlapping destination span of length <see cref="SpectrumLength"/>. / 入力と重ならない長さ <see cref="SpectrumLength"/> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">The input or destination length does not match the plan, or they overlap. / 入力または出力先の長さが計画と一致しない場合、または入出力が重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public VectorSpan<Complex> Forward(ReadOnlyVectorSpan<double> x, FourierNormalization normalization, VectorSpan<Complex> destination)
    {
        FourierViews.RealForwardRows(
            this, FourierViews.AsRow(x), FourierViews.AsRow(destination), FourierScaling.Forward(normalization, Length));
        return destination;
    }

    /// <summary>
    /// Reconstructs a real vector from its one-sided spectrum using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>片側スペクトルから実数ベクトルを <see cref="FourierNormalization.Backward"/> で逆変換し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input one-sided spectrum of length <see cref="SpectrumLength"/>. / 長さ <see cref="SpectrumLength"/> の片側スペクトル。</param>
    /// <param name="destination">The non-overlapping destination span of length <see cref="Length"/>. / 入力と重ならない長さ <see cref="Length"/> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>The imaginary part of bin 0 (and of the last bin when <see cref="Length"/> is even) is ignored.</para>
    /// <para>第 0 係数、および <see cref="Length"/> が偶数の場合の末尾係数の虚部は無視されます。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The input or destination length does not match the plan, or they overlap. / 入力または出力先の長さが計画と一致しない場合、または入出力が重なる場合。</exception>
    public VectorSpan<double> Inverse(ReadOnlyVectorSpan<Complex> x, VectorSpan<double> destination) =>
        Inverse(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Reconstructs a real vector from its one-sided spectrum with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で片側スペクトルから実数ベクトルを逆変換し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input one-sided spectrum of length <see cref="SpectrumLength"/>. / 長さ <see cref="SpectrumLength"/> の片側スペクトル。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The non-overlapping destination span of length <see cref="Length"/>. / 入力と重ならない長さ <see cref="Length"/> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>The imaginary part of bin 0 (and of the last bin when <see cref="Length"/> is even) is ignored.</para>
    /// <para>第 0 係数、および <see cref="Length"/> が偶数の場合の末尾係数の虚部は無視されます。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The input or destination length does not match the plan, or they overlap. / 入力または出力先の長さが計画と一致しない場合、または入出力が重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public VectorSpan<double> Inverse(ReadOnlyVectorSpan<Complex> x, FourierNormalization normalization, VectorSpan<double> destination)
    {
        FourierViews.RealInverseRows(
            this, FourierViews.AsRow(x), FourierViews.AsRow(destination), FourierScaling.Inverse(normalization, Length));
        return destination;
    }

    /// <summary>
    /// Computes the one-sided forward real discrete Fourier transform independently for each row of a matrix using <see cref="FourierNormalization.Backward"/>.
    /// <para>行列の各行ごとに実数 FFT の片側スペクトルを <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real matrix with <see cref="Length"/> columns. / 列数が <see cref="Length"/> の入力実数行列。</param>
    /// <param name="destination">The non-overlapping destination matrix of shape <c>x.Rows × SpectrumLength</c>. / 入力と重ならない <c>x.Rows × SpectrumLength</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">Matrix shapes do not match the plan, or <paramref name="destination"/> overlaps <paramref name="x"/>. / 行列の形状が計画と一致しない場合、または入出力が重なる場合。</exception>
    public MatrixSpan<Complex> ForwardRows(ReadOnlyMatrixSpan<double> x, MatrixSpan<Complex> destination) =>
        ForwardRows(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the one-sided forward real discrete Fourier transform independently for each row of a matrix with the specified normalization.
    /// <para>指定した正規化方式で行列の各行ごとに実数 FFT の片側スペクトルを計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real matrix with <see cref="Length"/> columns. / 列数が <see cref="Length"/> の入力実数行列。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The non-overlapping destination matrix of shape <c>x.Rows × SpectrumLength</c>. / 入力と重ならない <c>x.Rows × SpectrumLength</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">Matrix shapes do not match the plan, or <paramref name="destination"/> overlaps <paramref name="x"/>. / 行列の形状が計画と一致しない場合、または入出力が重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public MatrixSpan<Complex> ForwardRows(ReadOnlyMatrixSpan<double> x, FourierNormalization normalization, MatrixSpan<Complex> destination)
    {
        FourierViews.RealForwardRows(this, x, destination, FourierScaling.Forward(normalization, Length));
        return destination;
    }

    /// <summary>
    /// Computes the one-sided forward real discrete Fourier transform independently for each column of a matrix using <see cref="FourierNormalization.Backward"/>.
    /// <para>行列の各列ごとに実数 FFT の片側スペクトルを <see cref="FourierNormalization.Backward"/> で計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real matrix with <see cref="Length"/> rows. / 行数が <see cref="Length"/> の入力実数行列。</param>
    /// <param name="destination">The non-overlapping destination matrix of shape <c>SpectrumLength × x.Columns</c>. / 入力と重ならない <c>SpectrumLength × x.Columns</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">Matrix shapes do not match the plan, or <paramref name="destination"/> overlaps <paramref name="x"/>. / 行列の形状が計画と一致しない場合、または入出力が重なる場合。</exception>
    public MatrixSpan<Complex> ForwardColumns(ReadOnlyMatrixSpan<double> x, MatrixSpan<Complex> destination) =>
        ForwardColumns(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Computes the one-sided forward real discrete Fourier transform independently for each column of a matrix with the specified normalization.
    /// <para>指定した正規化方式で行列の各列ごとに実数 FFT の片側スペクトルを計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input real matrix with <see cref="Length"/> rows. / 行数が <see cref="Length"/> の入力実数行列。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The non-overlapping destination matrix of shape <c>SpectrumLength × x.Columns</c>. / 入力と重ならない <c>SpectrumLength × x.Columns</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException">Matrix shapes do not match the plan, or <paramref name="destination"/> overlaps <paramref name="x"/>. / 行列の形状が計画と一致しない場合、または入出力が重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public MatrixSpan<Complex> ForwardColumns(ReadOnlyMatrixSpan<double> x, FourierNormalization normalization, MatrixSpan<Complex> destination)
    {
        FourierViews.RealForwardRows(this, x.T, destination.T, FourierScaling.Forward(normalization, Length));
        return destination;
    }

    /// <summary>
    /// Reconstructs real rows from their one-sided spectra using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>行列の各行の片側スペクトルから実数列を <see cref="FourierNormalization.Backward"/> で逆変換し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input spectrum matrix with <see cref="SpectrumLength"/> columns. / 列数が <see cref="SpectrumLength"/> の片側スペクトル行列。</param>
    /// <param name="destination">The non-overlapping destination matrix of shape <c>x.Rows × Length</c>. / 入力と重ならない <c>x.Rows × Length</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>The imaginary part of bin 0 (and of the last bin when <see cref="Length"/> is even) is ignored.</para>
    /// <para>第 0 係数、および <see cref="Length"/> が偶数の場合の末尾係数の虚部は無視されます。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Matrix shapes do not match the plan, or <paramref name="destination"/> overlaps <paramref name="x"/>. / 行列の形状が計画と一致しない場合、または入出力が重なる場合。</exception>
    public MatrixSpan<double> InverseRows(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<double> destination) =>
        InverseRows(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Reconstructs real rows from their one-sided spectra with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で行列の各行の片側スペクトルから実数列を逆変換し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input spectrum matrix with <see cref="SpectrumLength"/> columns. / 列数が <see cref="SpectrumLength"/> の片側スペクトル行列。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The non-overlapping destination matrix of shape <c>x.Rows × Length</c>. / 入力と重ならない <c>x.Rows × Length</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>The imaginary part of bin 0 (and of the last bin when <see cref="Length"/> is even) is ignored.</para>
    /// <para>第 0 係数、および <see cref="Length"/> が偶数の場合の末尾係数の虚部は無視されます。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Matrix shapes do not match the plan, or <paramref name="destination"/> overlaps <paramref name="x"/>. / 行列の形状が計画と一致しない場合、または入出力が重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public MatrixSpan<double> InverseRows(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<double> destination)
    {
        FourierViews.RealInverseRows(this, x, destination, FourierScaling.Inverse(normalization, Length));
        return destination;
    }

    /// <summary>
    /// Reconstructs real columns from their one-sided spectra using <see cref="FourierNormalization.Backward"/> and writes the result to <paramref name="destination"/>.
    /// <para>行列の各列の片側スペクトルから実数列を <see cref="FourierNormalization.Backward"/> で逆変換し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input spectrum matrix with <see cref="SpectrumLength"/> rows. / 行数が <see cref="SpectrumLength"/> の片側スペクトル行列。</param>
    /// <param name="destination">The non-overlapping destination matrix of shape <c>Length × x.Columns</c>. / 入力と重ならない <c>Length × x.Columns</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>The imaginary part of bin 0 (and of the last bin when <see cref="Length"/> is even) is ignored.</para>
    /// <para>第 0 係数、および <see cref="Length"/> が偶数の場合の末尾係数の虚部は無視されます。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Matrix shapes do not match the plan, or <paramref name="destination"/> overlaps <paramref name="x"/>. / 行列の形状が計画と一致しない場合、または入出力が重なる場合。</exception>
    public MatrixSpan<double> InverseColumns(ReadOnlyMatrixSpan<Complex> x, MatrixSpan<double> destination) =>
        InverseColumns(x, FourierNormalization.Backward, destination);

    /// <summary>
    /// Reconstructs real columns from their one-sided spectra with the specified normalization and writes the result to <paramref name="destination"/>.
    /// <para>指定した正規化方式で行列の各列の片側スペクトルから実数列を逆変換し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The input spectrum matrix with <see cref="SpectrumLength"/> rows. / 行数が <see cref="SpectrumLength"/> の片側スペクトル行列。</param>
    /// <param name="normalization">The scaling convention to apply. / 適用する正規化方式。</param>
    /// <param name="destination">The non-overlapping destination matrix of shape <c>Length × x.Columns</c>. / 入力と重ならない <c>Length × x.Columns</c> の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>The imaginary part of bin 0 (and of the last bin when <see cref="Length"/> is even) is ignored.</para>
    /// <para>第 0 係数、および <see cref="Length"/> が偶数の場合の末尾係数の虚部は無視されます。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">Matrix shapes do not match the plan, or <paramref name="destination"/> overlaps <paramref name="x"/>. / 行列の形状が計画と一致しない場合、または入出力が重なる場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="normalization"/> is not a valid <see cref="FourierNormalization"/> value. / <paramref name="normalization"/> が無効な値の場合。</exception>
    public MatrixSpan<double> InverseColumns(ReadOnlyMatrixSpan<Complex> x, FourierNormalization normalization, MatrixSpan<double> destination)
    {
        FourierViews.RealInverseRows(this, x.T, destination.T, FourierScaling.Inverse(normalization, Length));
        return destination;
    }
}
