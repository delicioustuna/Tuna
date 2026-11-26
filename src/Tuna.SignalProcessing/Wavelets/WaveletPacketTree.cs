namespace Tuna.SignalProcessing;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

/// <summary>
/// Represents the result of a wavelet packet decomposition returned by <see cref="Wavelets.DecomposePackets(ReadOnlyVectorSpan{double}, DiscreteWavelet, int, WaveletBoundary)"/>.
/// <para><see cref="Wavelets.DecomposePackets(ReadOnlyVectorSpan{double}, DiscreteWavelet, int, WaveletBoundary)"/> が返すウェーブレットパケット分解の結果を表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// A wavelet packet tree is a full binary tree in which both approximation and detail coefficients are decomposed at every level, yielding 2^j nodes (frequency bands) at level j.
/// Nodes are indexed in natural tree order: decomposing node (j, i) produces approximation child (j + 1, 2i) and detail child (j + 1, 2i + 1).
/// Use <see cref="NaturalIndex"/> and <see cref="FrequencyIndex"/> to convert between natural order and ascending frequency order.
/// </para>
/// <para>
/// Coefficients may be modified in place through the views returned by <see cref="Level"/> and <see cref="Node"/> (for example, for thresholding);
/// <see cref="Wavelets.Reconstruct(WaveletPacketTree, IEnumerable{ValueTuple{int, int}})"/> reconstructs the signal from the modified values.
/// </para>
/// <para>
/// ウェーブレットパケット木は近似係数と詳細係数の両方を各段で分解した完全二分木であり、第 j 段に 2^j 個のノード（周波数帯）を持ちます。
/// ノード番号は自然順（木構造順）で、ノード (j, i) を 1 段分解すると近似係数が (j + 1, 2i)、詳細係数が (j + 1, 2i + 1) になります。
/// 自然順と周波数の昇順との相互変換には <see cref="NaturalIndex"/> および <see cref="FrequencyIndex"/> を使用します。
/// </para>
/// <para>
/// 係数は <see cref="Level"/> や <see cref="Node"/> が返すビューを通じて直接書き換えることができ（しきい値処理など）、
/// <see cref="Wavelets.Reconstruct(WaveletPacketTree, IEnumerable{ValueTuple{int, int}})"/> は書き換え後の値から信号を再構成します。
/// </para>
/// </remarks>
public sealed class WaveletPacketTree
{
    // Level j is stored as an m x 2^j column-major matrix so each node (j, i) is contiguous in column i.
    private readonly Matrix<double>[] levels;

    internal WaveletPacketTree(Matrix<double>[] levels, DiscreteWavelet wavelet, WaveletBoundary boundary)
    {
        this.levels = levels;
        Wavelet = wavelet;
        Boundary = boundary;
    }

    /// <summary>
    /// Gets the number of decomposition levels (the deepest level index).
    /// <para>分解段数（最も深い段の番号）を取得します。</para>
    /// </summary>
    public int Levels => levels.Length - 1;

    /// <summary>
    /// Gets the length of the original input signal.
    /// <para>元の入力信号の長さを取得します。</para>
    /// </summary>
    public int Length => levels[0].Rows;

    /// <summary>
    /// Gets the discrete wavelet used for decomposition.
    /// <para>分解に使用した離散ウェーブレットを取得します。</para>
    /// </summary>
    public DiscreteWavelet Wavelet { get; }

    /// <summary>
    /// Gets the boundary extension mode used for decomposition.
    /// <para>分解に使用した境界拡張モードを取得します。</para>
    /// </summary>
    public WaveletBoundary Boundary { get; }

    /// <summary>
    /// Returns a writable view of all wavelet packet coefficients at the specified level in natural order.
    /// <para>指定した段のすべてのウェーブレットパケット係数を自然順で参照する書き込み可能なビューを返します。</para>
    /// </summary>
    /// <param name="level">The level index, from 0 to <see cref="Levels"/>. / 0 以上 <see cref="Levels"/> 以下の段番号。</param>
    /// <returns>
    /// A <c>2^level × m</c> matrix view whose rows contain the nodes in natural order; edits are reflected in subsequent reconstructions. Level 0 returns a single-row view of the copied input signal. /
    /// 自然順の各ノードを行に持つ <c>2^level × m</c> の行列ビュー。書き換えは再構成に反映されます。第 0 段はコピーされた元の信号 1 行へのビューを返します。
    /// </returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="level"/> is negative or greater than <see cref="Levels"/>. / <paramref name="level"/> が負または <see cref="Levels"/> より大きい場合。</exception>
    public MatrixSpan<double> Level(int level)
    {
        RequireLevel(level);
        return levels[level].T;
    }

    /// <summary>
    /// Returns a newly allocated matrix containing all wavelet packet coefficients at the specified level ordered by ascending frequency.
    /// <para>指定した段のウェーブレットパケット係数を周波数の昇順に並べた新しい行列を返します。</para>
    /// </summary>
    /// <param name="level">The level index, from 0 to <see cref="Levels"/>. / 0 以上 <see cref="Levels"/> 以下の段番号。</param>
    /// <returns>A matrix whose rows contain the nodes ordered from lowest to highest frequency band. / 周波数の低い帯域から順に各ノードを行に並べた行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="level"/> is negative or greater than <see cref="Levels"/>. / <paramref name="level"/> が負または <see cref="Levels"/> より大きい場合。</exception>
    public Matrix<double> LevelByFrequency(int level)
    {
        RequireLevel(level);
        var source = levels[level];
        var result = new Matrix<double>(source.Columns, source.Rows);
        for (int p = 0; p < source.Columns; p++)
        {
            source.Column(NaturalIndex(p)).CopyTo(result.Row(p));
        }

        return result;
    }

    /// <summary>
    /// Returns a writable contiguous view of the coefficients for the specified node.
    /// <para>指定したノードの係数を参照する書き込み可能な連続ビューを返します。</para>
    /// </summary>
    /// <param name="level">The level index, from 0 to <see cref="Levels"/>. / 0 以上 <see cref="Levels"/> 以下の段番号。</param>
    /// <param name="index">The zero-based node index in natural order at <paramref name="level"/>. / <paramref name="level"/> における 0 始まりの自然順ノード番号。</param>
    /// <returns>A writable contiguous vector view of the node coefficients; edits are reflected in subsequent reconstructions. / ノード係数への書き込み可能な連続ベクトルビュー。書き換えは再構成に反映されます。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="level"/> or <paramref name="index"/> is out of range. / <paramref name="level"/> または <paramref name="index"/> が範囲外の場合。</exception>
    public VectorSpan<double> Node(int level, int index)
    {
        RequireNode(level, index);
        return levels[level].Column(index);
    }

    /// <summary>
    /// Converts a zero-based frequency-ordered node index to its corresponding natural-order index using the Gray code <c>p ^ (p &gt;&gt; 1)</c>.
    /// <para>0 始まりの周波数順ノード番号を、グレイ符号 <c>p ^ (p &gt;&gt; 1)</c> によって自然順のノード番号に変換します。</para>
    /// </summary>
    /// <param name="frequencyIndex">The zero-based index in ascending frequency order. / 周波数の昇順における 0 始まりの番号。</param>
    /// <returns>The corresponding natural-order node index at the same level. / 同じ段における対応する自然順のノード番号。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="frequencyIndex"/> is negative. / <paramref name="frequencyIndex"/> が負の場合。</exception>
    public static int NaturalIndex(int frequencyIndex)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(frequencyIndex);
        return frequencyIndex ^ (frequencyIndex >> 1);
    }

    /// <summary>
    /// Converts a zero-based natural-order node index to its corresponding ascending frequency-order index (the inverse of <see cref="NaturalIndex"/>).
    /// <para>0 始まりの自然順ノード番号を、周波数の昇順の番号（<see cref="NaturalIndex"/> の逆変換）に変換します。</para>
    /// </summary>
    /// <param name="naturalIndex">The zero-based node index in natural order. / 自然順における 0 始まりのノード番号。</param>
    /// <returns>The corresponding zero-based index in ascending frequency order at the same level. / 同じ段における周波数の昇順での 0 始まりの番号。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="naturalIndex"/> is negative. / <paramref name="naturalIndex"/> が負の場合。</exception>
    public static int FrequencyIndex(int naturalIndex)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(naturalIndex);
        int p = naturalIndex;
        for (int shift = 1; shift < 32; shift <<= 1)
        {
            p ^= p >> shift;
        }

        return p;
    }

    /// <summary>
    /// Computes the unnormalized Shannon entropy of the specified wavelet coefficients.
    /// <para>指定したウェーブレット係数の非正規化シャノンエントロピーを計算します。</para>
    /// </summary>
    /// <param name="coefficients">The node coefficients to evaluate. / 評価対象のノード係数。</param>
    /// <returns>The unnormalized Shannon entropy <c>-Sum(x_i^2 * ln(x_i^2))</c>, with zero terms contributing 0. / 非正規化シャノンエントロピー <c>-Sum(x_i^2 * ln(x_i^2))</c>（係数が 0 の項は 0 とみなします）。</returns>
    /// <remarks>
    /// <para>
    /// This is the default additive cost used by <see cref="BestBasis()"/> and matches MATLAB's <c>wentropy(x, 'shannon')</c>.
    /// </para>
    /// <para>
    /// <see cref="BestBasis()"/> の既定の加法的コスト関数であり、MATLAB の <c>wentropy(x, 'shannon')</c> と同じ定義です。
    /// </para>
    /// </remarks>
    public static double ShannonEntropy(ReadOnlyVectorSpan<double> coefficients)
    {
        double sum = 0.0;
        for (int i = 0; i < coefficients.Length; i++)
        {
            double square = coefficients[i] * coefficients[i];
            if (square > 0.0)
            {
                sum -= square * Math.Log(square);
            }
        }

        return sum;
    }

    /// <summary>
    /// Finds the wavelet packet basis that minimizes total <see cref="ShannonEntropy"/> across the tree.
    /// <para><see cref="ShannonEntropy"/> の合計を最小にするウェーブレットパケット基底を求めます。</para>
    /// </summary>
    /// <returns>
    /// A list of <c>(Level, Index)</c> pairs partitioning the tree without overlap or gaps, ordered depth-first from the approximation side. /
    /// 木を重複なく完全に分割する <c>(Level, Index)</c> の一覧。近似係数側から深さ優先の順に並びます。
    /// </returns>
    public IReadOnlyList<(int Level, int Index)> BestBasis() => BestBasis(static c => ShannonEntropy(c));

    /// <summary>
    /// Finds the wavelet packet basis that minimizes the sum of an additive node cost function across the tree.
    /// <para>加法的なノードコスト関数の合計を最小にするウェーブレットパケット基底を求めます。</para>
    /// </summary>
    /// <param name="cost">The cost function evaluated on each node's coefficients. / 各ノードの係数からコストを計算する関数。</param>
    /// <returns>
    /// A list of <c>(Level, Index)</c> pairs partitioning the tree without overlap or gaps, ordered depth-first from the approximation side. /
    /// 木を重複なく完全に分割する <c>(Level, Index)</c> の一覧。近似係数側から深さ優先の順に並びます。
    /// </returns>
    /// <remarks>
    /// <para>
    /// Level 0 (the unsplit signal) is included as a candidate. When a parent node and its two children have equal cost, the parent node is preferred.
    /// </para>
    /// <para>
    /// 第 0 段（未分解の信号）も候補に含まれます。親ノードと 2 つの子ノードの合計コストが等しい場合は親ノードを選択します。
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="cost"/> is <see langword="null"/>. / <paramref name="cost"/> が <see langword="null"/> の場合。</exception>
    public IReadOnlyList<(int Level, int Index)> BestBasis(Func<ReadOnlyVectorSpan<double>, double> cost)
    {
        ArgumentNullException.ThrowIfNull(cost);
        int depth = Levels;
        var keep = new bool[depth + 1][];
        double[] best = new double[1 << depth];
        keep[depth] = new bool[1 << depth];
        for (int i = 0; i < best.Length; i++)
        {
            best[i] = cost(levels[depth].Column(i));
            keep[depth][i] = true;
        }

        for (int level = depth - 1; level >= 0; level--)
        {
            int count = 1 << level;
            keep[level] = new bool[count];
            for (int i = 0; i < count; i++)
            {
                double own = cost(levels[level].Column(i));
                double children = best[2 * i] + best[2 * i + 1];
                keep[level][i] = own <= children;
                best[i] = keep[level][i] ? own : children;
            }
        }

        var basis = new List<(int Level, int Index)>();
        Collect(keep, 0, 0, basis);
        return basis;
    }

    private static void Collect(bool[][] keep, int level, int index, List<(int Level, int Index)> basis)
    {
        if (keep[level][index])
        {
            basis.Add((level, index));
            return;
        }

        Collect(keep, level + 1, 2 * index, basis);
        Collect(keep, level + 1, 2 * index + 1, basis);
    }

    internal Matrix<double> LevelMatrix(int level) => levels[level];

    private void RequireLevel(int level)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(level);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(level, Levels);
    }

    internal void RequireNode(int level, int index)
    {
        RequireLevel(level);
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, 1 << level);
    }
}
