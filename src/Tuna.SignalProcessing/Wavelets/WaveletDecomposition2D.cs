namespace Tuna.SignalProcessing;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

/// <summary>
/// Holds the result of a multilevel 2D discrete wavelet decomposition from <see cref="Wavelets.Decompose2D(ReadOnlyMatrixSpan{double}, DiscreteWavelet, int, WaveletBoundary)"/>.
/// <para><see cref="Wavelets.Decompose2D(ReadOnlyMatrixSpan{double}, DiscreteWavelet, int, WaveletBoundary)"/> による多段 2 次元離散ウェーブレット分解の結果を保持します。</para>
/// </summary>
/// <remarks>
/// <para>Stores the coarsest approximation matrix and tuples of horizontal, vertical, and diagonal detail matrices ordered from coarsest to finest level. Coefficient matrices are mutable; <see cref="Wavelets.Reconstruct(WaveletDecomposition2D)"/> reconstructs from their current values.</para>
/// <para>最も粗い段の近似係数行列と、粗い段から細かい段へ並べた水平・垂直・対角の詳細係数行列の組を保持します。各行列は書き換え可能で、<see cref="Wavelets.Reconstruct(WaveletDecomposition2D)"/> は現在の値を使用して再構成します。</para>
/// </remarks>
public sealed class WaveletDecomposition2D
{
    private readonly (Matrix<double> Horizontal, Matrix<double> Vertical, Matrix<double> Diagonal)[] details;
    private readonly (int Rows, int Columns)[] approximationShapes;

    internal WaveletDecomposition2D(
        Matrix<double> approximation, (Matrix<double> Horizontal, Matrix<double> Vertical, Matrix<double> Diagonal)[] details,
        (int Rows, int Columns)[] approximationShapes, DiscreteWavelet wavelet, WaveletBoundary boundary)
    {
        Approximation = approximation;
        this.details = details;
        this.approximationShapes = approximationShapes;
        Wavelet = wavelet;
        Boundary = boundary;
    }

    /// <summary>
    /// Gets the coarsest-level approximation coefficients (<c>cA_n</c>).
    /// <para>最も粗い段の近似係数（<c>cA_n</c>）を取得します。</para>
    /// </summary>
    public Matrix<double> Approximation { get; }

    /// <summary>
    /// Gets the detail subband triples (Horizontal, Vertical, Diagonal) ordered from coarsest to finest level.
    /// <para>粗い段から細かい段へ並べた詳細係数（水平・垂直・対角）の組の一覧を取得します。</para>
    /// </summary>
    public IReadOnlyList<(Matrix<double> Horizontal, Matrix<double> Vertical, Matrix<double> Diagonal)> Details => details;

    /// <summary>
    /// Gets the number of decomposition levels.
    /// <para>分解段数を取得します。</para>
    /// </summary>
    public int Levels => details.Length;

    /// <summary>
    /// Gets the row count of the original input matrix.
    /// <para>元の入力行列の行数を取得します。</para>
    /// </summary>
    public int Rows => approximationShapes[0].Rows;

    /// <summary>
    /// Gets the column count of the original input matrix.
    /// <para>元の入力行列の列数を取得します。</para>
    /// </summary>
    public int Columns => approximationShapes[0].Columns;

    /// <summary>
    /// Gets the discrete wavelet used for the decomposition.
    /// <para>分解に使用した離散ウェーブレットを取得します。</para>
    /// </summary>
    public DiscreteWavelet Wavelet { get; }

    /// <summary>
    /// Gets the boundary extension mode used for the decomposition.
    /// <para>分解に使用した境界拡張方式を取得します。</para>
    /// </summary>
    public WaveletBoundary Boundary { get; }

    /// <summary>
    /// Returns the detail subband triple at the specified 1-based level (<c>1</c> is the finest level; <see cref="Levels"/> is the coarsest).
    /// <para>指定した段（1 が最も細かい段、<see cref="Levels"/> が最も粗い段）の詳細係数の組を返します。</para>
    /// </summary>
    /// <param name="level">The 1-based decomposition level, from 1 to <see cref="Levels"/>. / 1 始まりの分解段（1〜<see cref="Levels"/>）。</param>
    /// <returns>The tuple of horizontal, vertical, and diagonal detail matrices for <paramref name="level"/>. / 第 <paramref name="level"/> 段の水平・垂直・対角詳細係数行列の組。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="level"/> is less than 1 or greater than <see cref="Levels"/>. / <paramref name="level"/> が 1 未満または <see cref="Levels"/> を超える場合。</exception>
    public (Matrix<double> Horizontal, Matrix<double> Vertical, Matrix<double> Diagonal) Detail(int level)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(level, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(level, Levels);
        return details[Levels - level];
    }

    internal (int Rows, int Columns) ApproximationShape(int level) => approximationShapes[level];
}
