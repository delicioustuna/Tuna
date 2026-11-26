using System.Buffers;

namespace Tuna.SignalProcessing;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

public static partial class Wavelets
{
    /// <summary>
    /// Computes a single-level discrete wavelet transform along each row of a matrix using symmetric boundary extension.
    /// <para>対称境界拡張を用いて、行列の各行に対して 1 段の離散ウェーブレット変換を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal matrix, where each row represents a channel or 1D signal. / 各行をチャンネルまたは 1 次元信号とする入力行列。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <returns>
    /// A tuple of newly allocated approximation and detail matrices whose column count is <see cref="CoefficientLength"/>. /
    /// 列数が <see cref="CoefficientLength"/> である近似係数行列と詳細係数行列のタプル。
    /// </returns>
    /// <remarks>
    /// <para>
    /// Row i of the output is bit-identical to applying <see cref="Forward(ReadOnlyVectorSpan{double}, DiscreteWavelet, WaveletBoundary)"/> to row i of <paramref name="x"/>.
    /// Uses <see cref="WaveletBoundary.Symmetric"/> at the edges.
    /// </para>
    /// <para>
    /// 結果の行 i は <paramref name="x"/> の行 i へ <see cref="Forward(ReadOnlyVectorSpan{double}, DiscreteWavelet, WaveletBoundary)"/> を適用した結果とビット単位で一致します。
    /// 境界条件には <see cref="WaveletBoundary.Symmetric"/> を使用します。
    /// </para>
    /// </remarks>
    public static (Matrix<double> Approximation, Matrix<double> Detail) ForwardRows(ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet) =>
        ForwardRows(x, wavelet, WaveletBoundary.Symmetric);

    /// <summary>
    /// Computes a single-level discrete wavelet transform along each row of a matrix using the specified boundary extension.
    /// <para>指定した境界拡張モードを用いて、行列の各行に対して 1 段の離散ウェーブレット変換を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal matrix. / 入力信号行列。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="boundary">The boundary extension mode applied at the row edges. / 行端に適用する境界拡張モード。</param>
    /// <returns>
    /// A tuple of newly allocated approximation and detail matrices whose column count is <see cref="CoefficientLength"/>. /
    /// 列数が <see cref="CoefficientLength"/> である近似係数行列と詳細係数行列のタプル。
    /// </returns>
    public static (Matrix<double> Approximation, Matrix<double> Detail) ForwardRows(
        ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet, WaveletBoundary boundary)
    {
        RequireMatrixSignal(x.Rows, x.Columns, boundary, nameof(x));
        int m = CoefficientLength(x.Columns, wavelet, boundary);
        var approximation = Matrix<double>.Uninitialized(x.Rows, m);
        var detail = Matrix<double>.Uninitialized(x.Rows, m);
        ForwardRowsCore(x, wavelet, boundary, approximation, detail);
        return (approximation, detail);
    }

    /// <summary>
    /// Computes a single-level inverse discrete wavelet transform along each row of a matrix using symmetric boundary extension.
    /// <para>対称境界拡張を用いて、行列の各行に対して 1 段の離散ウェーブレット逆変換を計算します。</para>
    /// </summary>
    /// <param name="approximation">The approximation coefficient matrix. / 近似係数行列。</param>
    /// <param name="detail">The detail coefficient matrix with the same shape as <paramref name="approximation"/>. / <paramref name="approximation"/> と同じ形状の詳細係数行列。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <returns>A newly allocated matrix containing the reconstructed rows, with column count <see cref="InverseLength"/>. / 列数 <see cref="InverseLength"/> の再構成行列。</returns>
    /// <remarks>
    /// <para>
    /// Uses <see cref="WaveletBoundary.Symmetric"/> at the row edges.
    /// </para>
    /// <para>
    /// 境界条件には <see cref="WaveletBoundary.Symmetric"/> を使用します。
    /// </para>
    /// </remarks>
    public static Matrix<double> InverseRows(ReadOnlyMatrixSpan<double> approximation, ReadOnlyMatrixSpan<double> detail, DiscreteWavelet wavelet) =>
        InverseRows(approximation, detail, wavelet, WaveletBoundary.Symmetric);

    /// <summary>
    /// Computes a single-level inverse discrete wavelet transform along each row of a matrix using the specified boundary extension.
    /// <para>指定した境界拡張モードを用いて、行列の各行に対して 1 段の離散ウェーブレット逆変換を計算します。</para>
    /// </summary>
    /// <param name="approximation">The approximation coefficient matrix. / 近似係数行列。</param>
    /// <param name="detail">The detail coefficient matrix with the same shape as <paramref name="approximation"/>. / <paramref name="approximation"/> と同じ形状の詳細係数行列。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="boundary">The boundary extension mode applied at the row edges. / 行端に適用する境界拡張モード。</param>
    /// <returns>A newly allocated matrix containing the reconstructed rows, with column count <see cref="InverseLength"/>. / 列数 <see cref="InverseLength"/> の再構成行列。</returns>
    public static Matrix<double> InverseRows(
        ReadOnlyMatrixSpan<double> approximation, ReadOnlyMatrixSpan<double> detail, DiscreteWavelet wavelet, WaveletBoundary boundary)
    {
        RequireMatrixCoefficients(approximation, detail);
        var result = Matrix<double>.Uninitialized(approximation.Rows, InverseLength(approximation.Columns, wavelet, boundary));
        InverseRowsCore(approximation, detail, wavelet, boundary, result);
        return result;
    }

    /// <summary>
    /// Computes a single-level discrete wavelet transform along each column of a matrix using symmetric boundary extension.
    /// <para>対称境界拡張を用いて、行列の各列に対して 1 段の離散ウェーブレット変換を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal matrix. / 入力信号行列。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <returns>
    /// A tuple of newly allocated approximation and detail matrices whose row count is <see cref="CoefficientLength"/>. /
    /// 行数が <see cref="CoefficientLength"/> である近似係数行列と詳細係数行列のタプル。
    /// </returns>
    /// <remarks>
    /// <para>
    /// Column j of the output is bit-identical to applying <see cref="Forward(ReadOnlyVectorSpan{double}, DiscreteWavelet, WaveletBoundary)"/> to column j of <paramref name="x"/>.
    /// Uses <see cref="WaveletBoundary.Symmetric"/> at the column edges.
    /// </para>
    /// <para>
    /// 結果の列 j は <paramref name="x"/> の列 j へ <see cref="Forward(ReadOnlyVectorSpan{double}, DiscreteWavelet, WaveletBoundary)"/> を適用した結果とビット単位で一致します。
    /// 境界条件には <see cref="WaveletBoundary.Symmetric"/> を使用します。
    /// </para>
    /// </remarks>
    public static (Matrix<double> Approximation, Matrix<double> Detail) ForwardColumns(ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet) =>
        ForwardColumns(x, wavelet, WaveletBoundary.Symmetric);

    /// <summary>
    /// Computes a single-level discrete wavelet transform along each column of a matrix using the specified boundary extension.
    /// <para>指定した境界拡張モードを用いて、行列の各列に対して 1 段の離散ウェーブレット変換を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal matrix. / 入力信号行列。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="boundary">The boundary extension mode applied at the column edges. / 列端に適用する境界拡張モード。</param>
    /// <returns>
    /// A tuple of newly allocated approximation and detail matrices whose row count is <see cref="CoefficientLength"/>. /
    /// 行数が <see cref="CoefficientLength"/> である近似係数行列と詳細係数行列のタプル。
    /// </returns>
    public static (Matrix<double> Approximation, Matrix<double> Detail) ForwardColumns(
        ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet, WaveletBoundary boundary)
    {
        RequireMatrixSignal(x.Columns, x.Rows, boundary, nameof(x));
        int m = CoefficientLength(x.Rows, wavelet, boundary);
        var approximation = Matrix<double>.Uninitialized(m, x.Columns);
        var detail = Matrix<double>.Uninitialized(m, x.Columns);
        ForwardColumnsCore(x, wavelet, boundary, approximation, detail);
        return (approximation, detail);
    }

    /// <summary>
    /// Computes a single-level inverse discrete wavelet transform along each column of a matrix using symmetric boundary extension.
    /// <para>対称境界拡張を用いて、行列の各列に対して 1 段の離散ウェーブレット逆変換を計算します。</para>
    /// </summary>
    /// <param name="approximation">The approximation coefficient matrix. / 近似係数行列。</param>
    /// <param name="detail">The detail coefficient matrix with the same shape as <paramref name="approximation"/>. / <paramref name="approximation"/> と同じ形状の詳細係数行列。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <returns>A newly allocated matrix containing the reconstructed columns, with row count <see cref="InverseLength"/>. / 行数 <see cref="InverseLength"/> の再構成行列。</returns>
    /// <remarks>
    /// <para>
    /// Uses <see cref="WaveletBoundary.Symmetric"/> at the column edges.
    /// </para>
    /// <para>
    /// 境界条件には <see cref="WaveletBoundary.Symmetric"/> を使用します。
    /// </para>
    /// </remarks>
    public static Matrix<double> InverseColumns(ReadOnlyMatrixSpan<double> approximation, ReadOnlyMatrixSpan<double> detail, DiscreteWavelet wavelet) =>
        InverseColumns(approximation, detail, wavelet, WaveletBoundary.Symmetric);

    /// <summary>
    /// Computes a single-level inverse discrete wavelet transform along each column of a matrix using the specified boundary extension.
    /// <para>指定した境界拡張モードを用いて、行列の各列に対して 1 段の離散ウェーブレット逆変換を計算します。</para>
    /// </summary>
    /// <param name="approximation">The approximation coefficient matrix. / 近似係数行列。</param>
    /// <param name="detail">The detail coefficient matrix with the same shape as <paramref name="approximation"/>. / <paramref name="approximation"/> と同じ形状の詳細係数行列。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="boundary">The boundary extension mode applied at the column edges. / 列端に適用する境界拡張モード。</param>
    /// <returns>A newly allocated matrix containing the reconstructed columns, with row count <see cref="InverseLength"/>. / 行数 <see cref="InverseLength"/> の再構成行列。</returns>
    public static Matrix<double> InverseColumns(
        ReadOnlyMatrixSpan<double> approximation, ReadOnlyMatrixSpan<double> detail, DiscreteWavelet wavelet, WaveletBoundary boundary)
    {
        RequireMatrixCoefficients(approximation, detail);
        var result = Matrix<double>.Uninitialized(InverseLength(approximation.Rows, wavelet, boundary), approximation.Columns);
        InverseColumnsCore(approximation, detail, wavelet, boundary, result);
        return result;
    }

    /// <summary>
    /// Computes a single-level 2D discrete wavelet transform along both axes of a matrix using symmetric boundary extension.
    /// <para>対称境界拡張を用いて、行列の両軸に対して 1 段の 2 次元離散ウェーブレット変換を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal matrix. / 入力信号行列。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <returns>
    /// A tuple containing the approximation, horizontal detail, vertical detail, and diagonal detail subband matrices, each sized <see cref="CoefficientLength"/> along both axes. /
    /// 各軸の長さが <see cref="CoefficientLength"/> である近似係数および水平・垂直・対角の詳細係数行列のタプル。
    /// </returns>
    /// <remarks>
    /// <para>
    /// Uses <see cref="WaveletBoundary.Symmetric"/> at the edges.
    /// </para>
    /// <para>
    /// 境界条件には <see cref="WaveletBoundary.Symmetric"/> を使用します。
    /// </para>
    /// </remarks>
    public static (Matrix<double> Approximation, Matrix<double> Horizontal, Matrix<double> Vertical, Matrix<double> Diagonal) Forward2D(
        ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet) =>
        Forward2D(x, wavelet, WaveletBoundary.Symmetric);

    /// <summary>
    /// Computes a single-level 2D discrete wavelet transform along both axes of a matrix using the specified boundary extension.
    /// <para>指定した境界拡張モードを用いて、行列の両軸に対して 1 段の 2 次元離散ウェーブレット変換を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal matrix. / 入力信号行列。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="boundary">The boundary extension mode applied at the matrix edges. / 行列端に適用する境界拡張モード。</param>
    /// <returns>
    /// A tuple containing the approximation, horizontal detail, vertical detail, and diagonal detail subband matrices. /
    /// 近似係数および水平・垂直・対角の詳細係数行列のタプル。
    /// </returns>
    /// <remarks>
    /// <para>
    /// Transforms columns (axis 0) first and then rows (axis 1), producing four subbands of shape <c>CoefficientLength(Rows) x CoefficientLength(Columns)</c>:
    /// <c>Approximation</c> (low-pass columns and rows), <c>Horizontal</c> (high-pass columns, low-pass rows; responds to horizontal edges),
    /// <c>Vertical</c> (low-pass columns, high-pass rows), and <c>Diagonal</c> (high-pass columns and rows).
    /// </para>
    /// <para>
    /// 各列（第 0 軸）を変換してから各行（第 1 軸）を変換し、形状 <c>CoefficientLength(Rows) x CoefficientLength(Columns)</c> の 4 成分を返します。
    /// <c>Approximation</c>（列・行ともに低域）、<c>Horizontal</c>（列が高域・行が低域で水平エッジに反応）、<c>Vertical</c>（列が低域・行が高域）、<c>Diagonal</c>（列・行ともに高域）に対応します。
    /// </para>
    /// </remarks>
    public static (Matrix<double> Approximation, Matrix<double> Horizontal, Matrix<double> Vertical, Matrix<double> Diagonal) Forward2D(
        ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet, WaveletBoundary boundary)
    {
        var (low, high) = ForwardColumns(x, wavelet, boundary);
        var (approximation, vertical) = ForwardRows(low, wavelet, boundary);
        var (horizontal, diagonal) = ForwardRows(high, wavelet, boundary);
        return (approximation, horizontal, vertical, diagonal);
    }

    /// <summary>
    /// Computes a single-level 2D inverse discrete wavelet transform using symmetric boundary extension.
    /// <para>対称境界拡張を用いて、1 段の 2 次元離散ウェーブレット逆変換を計算します。</para>
    /// </summary>
    /// <param name="approximation">The approximation subband matrix. / 近似係数行列。</param>
    /// <param name="horizontal">The horizontal detail subband matrix with the same shape as <paramref name="approximation"/>. / <paramref name="approximation"/> と同じ形状の水平詳細係数行列。</param>
    /// <param name="vertical">The vertical detail subband matrix with the same shape as <paramref name="approximation"/>. / <paramref name="approximation"/> と同じ形状の垂直詳細係数行列。</param>
    /// <param name="diagonal">The diagonal detail subband matrix with the same shape as <paramref name="approximation"/>. / <paramref name="approximation"/> と同じ形状の対角詳細係数行列。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <returns>A newly allocated matrix containing the reconstructed signal, sized <see cref="InverseLength"/> along both axes. / 各軸の長さが <see cref="InverseLength"/> である再構成行列。</returns>
    /// <remarks>
    /// <para>
    /// Uses <see cref="WaveletBoundary.Symmetric"/> at the edges.
    /// </para>
    /// <para>
    /// 境界条件には <see cref="WaveletBoundary.Symmetric"/> を使用します。
    /// </para>
    /// </remarks>
    public static Matrix<double> Inverse2D(
        ReadOnlyMatrixSpan<double> approximation, ReadOnlyMatrixSpan<double> horizontal, ReadOnlyMatrixSpan<double> vertical,
        ReadOnlyMatrixSpan<double> diagonal, DiscreteWavelet wavelet) =>
        Inverse2D(approximation, horizontal, vertical, diagonal, wavelet, WaveletBoundary.Symmetric);

    /// <summary>
    /// Computes a single-level 2D inverse discrete wavelet transform using the specified boundary extension.
    /// <para>指定した境界拡張モードを用いて、1 段の 2 次元離散ウェーブレット逆変換を計算します。</para>
    /// </summary>
    /// <param name="approximation">The approximation subband matrix. / 近似係数行列。</param>
    /// <param name="horizontal">The horizontal detail subband matrix with the same shape as <paramref name="approximation"/>. / <paramref name="approximation"/> と同じ形状の水平詳細係数行列。</param>
    /// <param name="vertical">The vertical detail subband matrix with the same shape as <paramref name="approximation"/>. / <paramref name="approximation"/> と同じ形状の垂直詳細係数行列。</param>
    /// <param name="diagonal">The diagonal detail subband matrix with the same shape as <paramref name="approximation"/>. / <paramref name="approximation"/> と同じ形状の対角詳細係数行列。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="boundary">The boundary extension mode applied at the matrix edges. / 行列端に適用する境界拡張モード。</param>
    /// <returns>A newly allocated matrix containing the reconstructed signal, sized <see cref="InverseLength"/> along both axes. / 各軸の長さが <see cref="InverseLength"/> である再構成行列。</returns>
    public static Matrix<double> Inverse2D(
        ReadOnlyMatrixSpan<double> approximation, ReadOnlyMatrixSpan<double> horizontal, ReadOnlyMatrixSpan<double> vertical,
        ReadOnlyMatrixSpan<double> diagonal, DiscreteWavelet wavelet, WaveletBoundary boundary)
    {
        RequireMatrixCoefficients(approximation, horizontal);
        RequireMatrixCoefficients(approximation, vertical);
        RequireMatrixCoefficients(approximation, diagonal);
        var low = InverseRows(approximation, vertical, wavelet, boundary);
        var high = InverseRows(horizontal, diagonal, wavelet, boundary);
        return InverseColumns(low, high, wavelet, boundary);
    }

    /// <summary>
    /// Computes the maximum useful 2D decomposition level at which at least one coefficient remains free of boundary effects along both axes.
    /// <para>両軸について境界の影響を受けない係数が残る最大分解段数を計算します。</para>
    /// </summary>
    /// <param name="rows">The number of rows in the input matrix, which must be non-negative. / 0 以上の入力行列の行数。</param>
    /// <param name="columns">The number of columns in the input matrix, which must be non-negative. / 0 以上の入力行列の列数。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <returns>The minimum of <see cref="MaxLevel(int, DiscreteWavelet)"/> for <paramref name="rows"/> and <paramref name="columns"/>. / <paramref name="rows"/> と <paramref name="columns"/> それぞれに対する <see cref="MaxLevel(int, DiscreteWavelet)"/> のうち小さい方。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="wavelet"/> is <see langword="null"/>. / <paramref name="wavelet"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rows"/> or <paramref name="columns"/> is negative. / <paramref name="rows"/> または <paramref name="columns"/> が負の場合。</exception>
    public static int MaxLevel(int rows, int columns, DiscreteWavelet wavelet) =>
        Math.Min(MaxLevel(rows, wavelet), MaxLevel(columns, wavelet));

    /// <summary>
    /// Computes a multi-level 2D discrete wavelet decomposition using the maximum useful level and symmetric boundary extension.
    /// <para>最大有効段数と対称境界拡張を用いて、行列の多段 2 次元離散ウェーブレット分解を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal matrix to decompose. / 分解対象の入力行列。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <returns>A 2D decomposition containing the coarsest approximation and detail subbands ordered from coarsest to finest level. / 最も粗い近似係数と、粗い段から細かい段へ並べた詳細係数を保持する 2 次元分解結果。</returns>
    /// <remarks>
    /// <para>
    /// The decomposition depth is determined by <see cref="MaxLevel(int, int, DiscreteWavelet)"/> and uses <see cref="WaveletBoundary.Symmetric"/>.
    /// </para>
    /// <para>
    /// 分解段数は <see cref="MaxLevel(int, int, DiscreteWavelet)"/> で決定され、境界条件には <see cref="WaveletBoundary.Symmetric"/> を使用します。
    /// </para>
    /// </remarks>
    public static WaveletDecomposition2D Decompose2D(ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet) =>
        Decompose2D(x, wavelet, WaveletBoundary.Symmetric);

    /// <summary>
    /// Computes a multi-level 2D discrete wavelet decomposition using the maximum useful level and the specified boundary extension.
    /// <para>最大有効段数と指定した境界拡張モードを用いて、行列の多段 2 次元離散ウェーブレット分解を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal matrix to decompose. / 分解対象の入力行列。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="boundary">The boundary extension mode applied at the matrix edges. / 行列端に適用する境界拡張モード。</param>
    /// <returns>A 2D decomposition containing the coarsest approximation and detail subbands ordered from coarsest to finest level. / 最も粗い近似係数と、粗い段から細かい段へ並べた詳細係数を保持する 2 次元分解結果。</returns>
    /// <remarks>
    /// <para>
    /// The decomposition depth is determined by <see cref="MaxLevel(int, int, DiscreteWavelet)"/>.
    /// </para>
    /// <para>
    /// 分解段数は <see cref="MaxLevel(int, int, DiscreteWavelet)"/> で決定されます。
    /// </para>
    /// </remarks>
    public static WaveletDecomposition2D Decompose2D(ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet, WaveletBoundary boundary)
    {
        ArgumentNullException.ThrowIfNull(wavelet);
        return Decompose2D(x, wavelet, MaxLevel(x.Rows, x.Columns, wavelet), boundary);
    }

    /// <summary>
    /// Computes a multi-level 2D discrete wavelet decomposition to the specified number of levels using symmetric boundary extension.
    /// <para>指定した段数と対称境界拡張を用いて、行列の多段 2 次元離散ウェーブレット分解を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal matrix to decompose. / 分解対象の入力行列。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="levels">The number of decomposition levels, which must be non-negative. / 0 以上の分解段数。</param>
    /// <returns>A 2D decomposition containing the coarsest approximation and detail subbands ordered from coarsest to finest level. / 最も粗い近似係数と、粗い段から細かい段へ並べた詳細係数を保持する 2 次元分解結果。</returns>
    /// <remarks>
    /// <para>
    /// Uses <see cref="WaveletBoundary.Symmetric"/> at the edges.
    /// </para>
    /// <para>
    /// 境界条件には <see cref="WaveletBoundary.Symmetric"/> を使用します。
    /// </para>
    /// </remarks>
    public static WaveletDecomposition2D Decompose2D(ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet, int levels) =>
        Decompose2D(x, wavelet, levels, WaveletBoundary.Symmetric);

    /// <summary>
    /// Computes a multi-level 2D discrete wavelet decomposition to the specified number of levels and boundary extension.
    /// <para>指定した段数および境界拡張モードで、行列の多段 2 次元離散ウェーブレット分解を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal matrix to decompose. / 分解対象の入力行列。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="levels">The number of decomposition levels, which must be non-negative. / 0 以上の分解段数。</param>
    /// <param name="boundary">The boundary extension mode applied at the matrix edges. / 行列端に適用する境界拡張モード。</param>
    /// <returns>A 2D decomposition containing the coarsest approximation and detail subbands ordered from coarsest to finest level. / 最も粗い近似係数と、粗い段から細かい段へ並べた詳細係数を保持する 2 次元分解結果。</returns>
    /// <remarks>
    /// <para>
    /// Repeatedly applies <see cref="Forward2D(ReadOnlyMatrixSpan{double}, DiscreteWavelet, WaveletBoundary)"/> to the approximation subband.
    /// </para>
    /// <para>
    /// 近似係数に対して <see cref="Forward2D(ReadOnlyMatrixSpan{double}, DiscreteWavelet, WaveletBoundary)"/> を繰り返し適用します。
    /// </para>
    /// </remarks>
    public static WaveletDecomposition2D Decompose2D(ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet, int levels, WaveletBoundary boundary)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(levels);
        RequireMatrixSignal(x.Rows, x.Columns, boundary, nameof(x));
        RequireMatrixSignal(x.Columns, x.Rows, boundary, nameof(x));
        ArgumentNullException.ThrowIfNull(wavelet);
        var shapes = new (int Rows, int Columns)[levels + 1];
        shapes[0] = (x.Rows, x.Columns);
        var details = new (Matrix<double> Horizontal, Matrix<double> Vertical, Matrix<double> Diagonal)[levels];
        var approximation = x.ToMatrix();
        for (int level = 1; level <= levels; level++)
        {
            if (boundary == WaveletBoundary.Reflect && (approximation.Rows < 2 || approximation.Columns < 2))
            {
                throw new ArgumentOutOfRangeException(
                    nameof(levels), $"Level {level} has a single approximation coefficient along an axis, which the reflect boundary cannot extend.");
            }

            var (a, h, v, d) = Forward2D(approximation, wavelet, boundary);
            shapes[level] = (a.Rows, a.Columns);
            details[levels - level] = (h, v, d);
            approximation = a;
        }

        return new WaveletDecomposition2D(approximation, details, shapes, wavelet, boundary);
    }

    /// <summary>
    /// Reconstructs a matrix from its multi-level 2D discrete wavelet decomposition.
    /// <para>多段 2 次元離散ウェーブレット分解から元の行列を再構成します。</para>
    /// </summary>
    /// <param name="decomposition">The 2D wavelet decomposition containing approximation and detail subbands. / 近似係数と詳細係数を保持する 2 次元ウェーブレット分解。</param>
    /// <returns>A newly allocated matrix containing the reconstructed signal with the original matrix shape. / 元の行列と同じ形状の再構成信号を格納した新しい行列。</returns>
    /// <remarks>
    /// <para>
    /// When the original matrix or any intermediate approximation had an odd row or column count, the trailing extra row or column produced by single-level inversion is trimmed at each level so the exact original shape is restored.
    /// </para>
    /// <para>
    /// 元の行列や途中の近似係数の行数・列数が奇数だった場合も、各段の逆変換で生じる末尾の余剰な行・列を切り詰めて元の形状に正確に復元します。
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="decomposition"/> is <see langword="null"/>. / <paramref name="decomposition"/> が <see langword="null"/> の場合。</exception>
    public static Matrix<double> Reconstruct(WaveletDecomposition2D decomposition)
    {
        ArgumentNullException.ThrowIfNull(decomposition);
        var wavelet = decomposition.Wavelet;
        var boundary = decomposition.Boundary;
        int levels = decomposition.Levels;
        Matrix<double> approximation = decomposition.Approximation.Clone();
        for (int i = 0; i < levels; i++)
        {
            var (h, v, d) = decomposition.Details[i];
            var (rows, columns) = decomposition.ApproximationShape(levels - 1 - i);
            var full = Inverse2D(approximation, h, v, d, wavelet, boundary);
            approximation = full.Rows == rows && full.Columns == columns ? full : full[..rows, ..columns].ToMatrix();
        }

        return approximation;
    }

    // Threshold below which per-row 1D transforms are used; both paths are bit-identical.
    private static int RowsPerSimd => 2 * System.Numerics.Vector<double>.Count;

    private static void ForwardRowsCore(
        ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet, WaveletBoundary boundary, Matrix<double> approximation, Matrix<double> detail)
    {
        int rows = x.Rows;
        int n = x.Columns;
        int m = approximation.Columns;
        if (rows < RowsPerSimd)
        {
            double[] row = ArrayPool<double>.Shared.Rent(2 * m);
            for (int i = 0; i < rows; i++)
            {
                ForwardCore(x.Row(i), wavelet, boundary, row.AsSpan(0, m), row.AsSpan(m, m));
                Copy(row.AsSpan(0, m), approximation.Row(i));
                Copy(row.AsSpan(m, m), detail.Row(i));
            }

            ArrayPool<double>.Shared.Return(row);
            return;
        }

        // Column n serves as a sentinel zero column for zero-padded out-of-bounds taps.
        int f = wavelet.FilterLength;
        var low = wavelet.DecompositionLow;
        var high = wavelet.DecompositionHigh;
        double[] source = ArrayPool<double>.Shared.Rent((n + 1) * rows);
        x.CopyTo(new MatrixSpan<double>(source.AsSpan(0, n * rows), rows, n));
        source.AsSpan(n * rows, rows).Clear();
        var a = approximation.AsSpan();
        var d = detail.AsSpan();
        a.Clear();
        d.Clear();
        int offset = boundary == WaveletBoundary.Periodization ? f / 2 - 1 : 0;
        int first = 2 - f + offset;
        for (int i0 = 0; i0 < rows; i0 += Block)
        {
            int count = Math.Min(Block, rows - i0);
            for (int k = 0; k < m; k++)
            {
                var ak = a.Slice(k * rows + i0, count);
                var dk = d.Slice(k * rows + i0, count);
                for (int j = 0; j < f; j++)
                {
                    int column = ExtendedColumn(first + 2 * k + f - 1 - j, n, boundary);
                    MultiplyAdd(source.AsSpan(column * rows + i0, count), low[j], high[j], ak, dk);
                }
            }
        }

        ArrayPool<double>.Shared.Return(source);
    }

    private static void InverseRowsCore(
        ReadOnlyMatrixSpan<double> approximation, ReadOnlyMatrixSpan<double> detail, DiscreteWavelet wavelet, WaveletBoundary boundary,
        Matrix<double> destination)
    {
        int rows = approximation.Rows;
        int m = approximation.Columns;
        int length = destination.Columns;
        if (rows < RowsPerSimd)
        {
            double[] row = ArrayPool<double>.Shared.Rent(length);
            for (int i = 0; i < rows; i++)
            {
                InverseCore(approximation.Row(i), detail.Row(i), wavelet, boundary, row.AsSpan(0, length));
                Copy(row.AsSpan(0, length), destination.Row(i));
            }

            ArrayPool<double>.Shared.Return(row);
            return;
        }

        int f = wavelet.FilterLength;
        var low = wavelet.ReconstructionLow;
        var high = wavelet.ReconstructionHigh;
        bool periodization = boundary == WaveletBoundary.Periodization;
        double[] buffer = ArrayPool<double>.Shared.Rent(2 * m * rows);
        var a = buffer.AsSpan(0, m * rows);
        var d = buffer.AsSpan(m * rows, m * rows);
        approximation.CopyTo(new MatrixSpan<double>(a, rows, m));
        detail.CopyTo(new MatrixSpan<double>(d, rows, m));
        var output = destination.AsSpan();
        output.Clear();
        for (int i0 = 0; i0 < rows; i0 += Block)
        {
            int count = Math.Min(Block, rows - i0);
            for (int t = 0; t < length; t++)
            {
                int r = t & 1;
                int i = t >> 1;
                var target = output.Slice(t * rows + i0, count);
                for (int q = 0; q < f / 2; q++)
                {
                    int j, k;
                    if (periodization)
                    {
                        int parity = (r + f / 2 - 1) & 1;
                        j = parity + 2 * q;
                        k = Modulo((r + f / 2 - 1 - parity) / 2 - q + i, m);
                    }
                    else
                    {
                        j = f - 2 + r - 2 * q;
                        k = i + q;
                    }

                    AddProducts(target, a.Slice(k * rows + i0, count), low[j], d.Slice(k * rows + i0, count), high[j]);
                }
            }
        }

        ArrayPool<double>.Shared.Return(buffer);
    }

    private static void ForwardColumnsCore(
        ReadOnlyMatrixSpan<double> x, DiscreteWavelet wavelet, WaveletBoundary boundary, Matrix<double> approximation, Matrix<double> detail)
    {
        int m = approximation.Rows;
        var a = approximation.AsSpan();
        var d = detail.AsSpan();
        for (int column = 0; column < x.Columns; column++)
        {
            ForwardCore(x.Column(column), wavelet, boundary, a.Slice(column * m, m), d.Slice(column * m, m));
        }
    }

    private static void InverseColumnsCore(
        ReadOnlyMatrixSpan<double> approximation, ReadOnlyMatrixSpan<double> detail, DiscreteWavelet wavelet, WaveletBoundary boundary,
        Matrix<double> destination)
    {
        int length = destination.Rows;
        var output = destination.AsSpan();
        for (int column = 0; column < approximation.Columns; column++)
        {
            InverseCore(approximation.Column(column), detail.Column(column), wavelet, boundary, output.Slice(column * length, length));
        }
    }

    // Maps zero-padded out-of-bounds positions to the sentinel zero column n.
    private static int ExtendedColumn(int position, int n, WaveletBoundary boundary)
    {
        if ((uint)position < (uint)n)
        {
            return position;
        }

        int index = boundary == WaveletBoundary.Periodization ? PeriodizedIndex(position, n) : ExtendedIndex(position, n, boundary);
        return index < 0 ? n : index;
    }

    private static void RequireMatrixSignal(int count, int length, WaveletBoundary boundary, string name)
    {
        RequireBoundary(boundary);
        if (count == 0 || length == 0)
        {
            throw new ArgumentException("The matrix must not be empty.", name);
        }

        if (boundary == WaveletBoundary.Reflect && length < 2)
        {
            throw new ArgumentException("The reflect boundary needs at least 2 samples along the transformed axis.", name);
        }
    }

    private static void RequireMatrixCoefficients(ReadOnlyMatrixSpan<double> approximation, ReadOnlyMatrixSpan<double> other)
    {
        if (approximation.Rows != other.Rows || approximation.Columns != other.Columns)
        {
            throw new ArgumentException(
                $"The coefficient matrices must have the same shape, but they are {approximation.Rows} x {approximation.Columns} and {other.Rows} x {other.Columns}.",
                nameof(other));
        }

        if (approximation.Rows == 0 || approximation.Columns == 0)
        {
            throw new ArgumentException("The coefficients must not be empty.", nameof(approximation));
        }
    }
}
