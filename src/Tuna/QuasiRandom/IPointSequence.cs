using Tuna.LinearAlgebra;

namespace Tuna.QuasiRandom;

/// <summary>
/// Represents a deterministic low-discrepancy point sequence in the open unit hypercube <c>(0, 1)^d</c> with random-access indexing.
/// <para>単位超立方体 <c>(0, 1)^d</c> 内の点をインデックスでランダムアクセスできる決定的な点列を表します。</para>
/// </summary>
/// <remarks>
/// <para>Points are written into matrices with one point per column (<see cref="Dimension"/> rows by point count columns), matching the convention used by <see cref="XMath"/> pairwise operations.</para>
/// <para>Splitting a range across multiple calls or threads produces results that are bit-for-bit identical to generating the range in a single call. Implementations are immutable after construction and thread-safe across concurrent calls.</para>
/// <para>点は「1 列 = 1 点」のレイアウトで行列に書き込まれます（行数が <see cref="Dimension"/>、列数が点の個数）。この形式は <see cref="XMath"/> のペアワイズ演算などと共通です。</para>
/// <para>範囲を分割して並列に書き込んでも、一括で書き込んだ場合とビット単位で同一の結果になります。実装は構築後に不変であり、各メソッドはスレッドセーフです。</para>
/// </remarks>
public interface IPointSequence
{
    /// <summary>
    /// Gets the dimensionality of points in the sequence.
    /// <para>点列の次元数を取得します。</para>
    /// </summary>
    int Dimension { get; }

    /// <summary>
    /// Writes the point at the specified zero-based index into <paramref name="destination"/>.
    /// <para>指定した 0 始まりのインデックスに対応する点を <paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <param name="index">The zero-based index of the point. / 0 から始まる点のインデックス。</param>
    /// <param name="destination">The destination span of length <see cref="Dimension"/>. / 長さ <see cref="Dimension"/> の出力先スパン。</param>
    /// <exception cref="ArgumentException">The length of <paramref name="destination"/> does not equal <see cref="Dimension"/>. / <paramref name="destination"/> の長さが <see cref="Dimension"/> と等しくない場合。</exception>
    void GetPoint(ulong index, Span<double> destination);

    /// <summary>
    /// Writes consecutive points starting at <paramref name="start"/> into the columns of <paramref name="destination"/> (one point per column).
    /// <para><paramref name="start"/> から始まる連続した点を、<paramref name="destination"/> の各列に 1 列あたり 1 点ずつ書き込みます。</para>
    /// </summary>
    /// <param name="start">The zero-based index of the first point. / 最初の点の 0 始まりインデックス。</param>
    /// <param name="destination">The destination matrix view with <see cref="Dimension"/> rows. / 行数が <see cref="Dimension"/> の出力先行列ビュー。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentException"><paramref name="destination"/> does not have <see cref="Dimension"/> rows. / <paramref name="destination"/> の行数が <see cref="Dimension"/> と等しくない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">The requested range exceeds the indexable range of the sequence. / 要求された範囲が点列のインデックス上限を超える場合。</exception>
    MatrixSpan<double> GetPoints(ulong start, MatrixSpan<double> destination);

    /// <summary>
    /// Allocates a new <see cref="Dimension"/>-by-<paramref name="count"/> matrix and fills its columns with consecutive points starting at <paramref name="start"/>.
    /// <para><see cref="Dimension"/> 行 <paramref name="count"/> 列の新しい行列を確保し、<paramref name="start"/> から始まる連続した点を各列に書き込みます。</para>
    /// </summary>
    /// <param name="start">The zero-based index of the first point. / 最初の点の 0 始まりインデックス。</param>
    /// <param name="count">The number of points to generate. / 生成する点の個数。</param>
    /// <returns>A new <see cref="Matrix{T}"/> containing the generated points (one point per column). / 生成された点を格納した新しい <see cref="Matrix{T}"/>（1 列あたり 1 点）。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative, or the requested range exceeds the indexable range of the sequence. / <paramref name="count"/> が負の場合、または要求された範囲が点列のインデックス上限を超える場合。</exception>
    Matrix<double> GetPoints(ulong start, int count);
}
