namespace Tuna.SignalProcessing;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

/// <summary>
/// Holds the result of a multilevel 1D discrete wavelet decomposition from <see cref="Wavelets.Decompose(ReadOnlyVectorSpan{double}, DiscreteWavelet, int, WaveletBoundary)"/>.
/// <para><see cref="Wavelets.Decompose(ReadOnlyVectorSpan{double}, DiscreteWavelet, int, WaveletBoundary)"/> による多段 1 次元離散ウェーブレット分解の結果を保持します。</para>
/// </summary>
/// <remarks>
/// <para>Stores the coarsest approximation coefficients together with detail coefficients ordered from coarsest to finest level. Coefficient vectors are mutable; <see cref="Wavelets.Reconstruct(WaveletDecomposition)"/> reconstructs from their current values.</para>
/// <para>最も粗い段の近似係数と、粗い段から細かい段へ並べた詳細係数を保持します。各係数ベクトルは書き換え可能で、<see cref="Wavelets.Reconstruct(WaveletDecomposition)"/> は現在の値を使用して再構成します。</para>
/// </remarks>
public sealed class WaveletDecomposition
{
    private readonly Vector<double>[] details;
    private readonly int[] approximationLengths;

    internal WaveletDecomposition(
        Vector<double> approximation, Vector<double>[] details, int[] approximationLengths, DiscreteWavelet wavelet, WaveletBoundary boundary)
    {
        Approximation = approximation;
        this.details = details;
        this.approximationLengths = approximationLengths;
        Wavelet = wavelet;
        Boundary = boundary;
    }

    /// <summary>
    /// Gets the coarsest-level approximation coefficients (<c>cA_n</c>).
    /// <para>最も粗い段の近似係数（<c>cA_n</c>）を取得します。</para>
    /// </summary>
    public Vector<double> Approximation { get; }

    /// <summary>
    /// Gets the detail coefficient vectors ordered from coarsest (<c>cD_n</c> at index 0) to finest (<c>cD_1</c> at the end).
    /// <para>粗い段（先頭が <c>cD_n</c>）から細かい段（末尾が <c>cD_1</c>）の順に並んだ詳細係数の一覧を取得します。</para>
    /// </summary>
    public IReadOnlyList<Vector<double>> Details => details;

    /// <summary>
    /// Gets the number of decomposition levels.
    /// <para>分解段数を取得します。</para>
    /// </summary>
    public int Levels => details.Length;

    /// <summary>
    /// Gets the length of the original input signal.
    /// <para>元の入力信号の長さを取得します。</para>
    /// </summary>
    public int Length => approximationLengths[0];

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
    /// Returns the detail coefficients at the specified 1-based level (<c>1</c> is the finest level <c>cD_1</c>; <see cref="Levels"/> is the coarsest).
    /// <para>指定した段（1 が最も細かい <c>cD_1</c>、<see cref="Levels"/> が最も粗い段）の詳細係数を返します。</para>
    /// </summary>
    /// <param name="level">The 1-based decomposition level, from 1 to <see cref="Levels"/>. / 1 始まりの分解段（1〜<see cref="Levels"/>）。</param>
    /// <returns>The detail coefficient vector for <paramref name="level"/>. / 第 <paramref name="level"/> 段の詳細係数ベクトル。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="level"/> is less than 1 or greater than <see cref="Levels"/>. / <paramref name="level"/> が 1 未満または <see cref="Levels"/> を超える場合。</exception>
    public Vector<double> Detail(int level)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(level, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(level, Levels);
        return details[Levels - level];
    }

    internal int ApproximationLength(int level) => approximationLengths[level];
}
