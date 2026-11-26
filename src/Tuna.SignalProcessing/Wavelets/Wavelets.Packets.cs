using System.Buffers;

namespace Tuna.SignalProcessing;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

public static partial class Wavelets
{
    /// <summary>
    /// Computes the wavelet packet decomposition of a signal using the maximum useful level and symmetric boundary extension.
    /// <para>最大有効段数と対称境界拡張を用いて、入力信号のウェーブレットパケット分解を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal to decompose. / 分解対象の入力信号。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <returns>A wavelet packet binary tree containing coefficients for all levels. / 各段の係数を保持するウェーブレットパケット二分木。</returns>
    /// <remarks>
    /// <para>
    /// The decomposition depth is determined by <see cref="MaxLevel(int, DiscreteWavelet)"/> and uses <see cref="WaveletBoundary.Symmetric"/>.
    /// </para>
    /// <para>
    /// 分解段数は <see cref="MaxLevel(int, DiscreteWavelet)"/> で決定され、境界条件には <see cref="WaveletBoundary.Symmetric"/> を使用します。
    /// </para>
    /// </remarks>
    public static WaveletPacketTree DecomposePackets(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet) =>
        DecomposePackets(x, wavelet, WaveletBoundary.Symmetric);

    /// <summary>
    /// Computes the wavelet packet decomposition of a signal using the maximum useful level and the specified boundary extension.
    /// <para>最大有効段数と指定した境界拡張モードを用いて、入力信号のウェーブレットパケット分解を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal to decompose. / 分解対象の入力信号。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="boundary">The boundary extension mode applied at the signal edges. / 信号端に適用する境界拡張モード。</param>
    /// <returns>A wavelet packet binary tree containing coefficients for all levels. / 各段の係数を保持するウェーブレットパケット二分木。</returns>
    /// <remarks>
    /// <para>
    /// The decomposition depth is determined by <see cref="MaxLevel(int, DiscreteWavelet)"/>.
    /// </para>
    /// <para>
    /// 分解段数は <see cref="MaxLevel(int, DiscreteWavelet)"/> で決定されます。
    /// </para>
    /// </remarks>
    public static WaveletPacketTree DecomposePackets(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, WaveletBoundary boundary)
    {
        ArgumentNullException.ThrowIfNull(wavelet);
        return DecomposePackets(x, wavelet, MaxLevel(x.Length, wavelet), boundary);
    }

    /// <summary>
    /// Computes the wavelet packet decomposition of a signal to the specified number of levels using symmetric boundary extension.
    /// <para>指定した段数と対称境界拡張を用いて、入力信号のウェーブレットパケット分解を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal to decompose. / 分解対象の入力信号。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="levels">The number of decomposition levels, from 0 to 30. / 0 以上 30 以下の分解段数。</param>
    /// <returns>A wavelet packet binary tree containing coefficients for all levels. / 各段の係数を保持するウェーブレットパケット二分木。</returns>
    /// <remarks>
    /// <para>
    /// Uses <see cref="WaveletBoundary.Symmetric"/> at the signal edges.
    /// </para>
    /// <para>
    /// 境界条件には <see cref="WaveletBoundary.Symmetric"/> を使用します。
    /// </para>
    /// </remarks>
    public static WaveletPacketTree DecomposePackets(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, int levels) =>
        DecomposePackets(x, wavelet, levels, WaveletBoundary.Symmetric);

    /// <summary>
    /// Computes the wavelet packet decomposition of a signal to the specified number of levels and boundary extension.
    /// <para>指定した段数および境界拡張モードで、入力信号のウェーブレットパケット分解を計算します。</para>
    /// </summary>
    /// <param name="x">The input signal to decompose. / 分解対象の入力信号。</param>
    /// <param name="wavelet">The discrete wavelet to use. / 使用する離散ウェーブレット。</param>
    /// <param name="levels">The number of decomposition levels, from 0 to 30. / 0 以上 30 以下の分解段数。</param>
    /// <param name="boundary">The boundary extension mode applied at the signal edges. / 信号端に適用する境界拡張モード。</param>
    /// <returns>A wavelet packet binary tree containing coefficients for all levels. / 各段の係数を保持するウェーブレットパケット二分木。</returns>
    /// <remarks>
    /// <para>
    /// Each node at level j is decomposed into two child nodes at level j + 1 using a single-level <see cref="Forward(ReadOnlyVectorSpan{double}, DiscreteWavelet, WaveletBoundary)"/> step.
    /// Each level holds approximately <c>x.Length</c> coefficients across its 2^j nodes, so total memory scales as roughly <c>x.Length * (levels + 1)</c>.
    /// </para>
    /// <para>
    /// 第 j 段の各ノードを <see cref="Forward(ReadOnlyVectorSpan{double}, DiscreteWavelet, WaveletBoundary)"/> による 1 段分解で第 j + 1 段の 2 つの子ノードへ展開します。
    /// 各段は 2^j 個のノードを合わせて約 <c>x.Length</c> 個の係数を持つため、全体のメモリ使用量は概ね <c>x.Length * (levels + 1)</c> です。
    /// </para>
    /// </remarks>
    public static WaveletPacketTree DecomposePackets(ReadOnlyVectorSpan<double> x, DiscreteWavelet wavelet, int levels, WaveletBoundary boundary)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(levels);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(levels, 30);
        int length = RequireSignal(x, boundary);
        ArgumentNullException.ThrowIfNull(wavelet);
        var tree = new Matrix<double>[levels + 1];
        tree[0] = Matrix<double>.Uninitialized(length, 1);
        x.CopyTo(tree[0].Column(0));
        for (int level = 1; level <= levels; level++)
        {
            var parent = tree[level - 1];
            int m = parent.Rows;
            if (boundary == WaveletBoundary.Reflect && m < 2)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(levels), $"Level {level - 1} has a single coefficient per node, which the reflect boundary cannot extend.");
            }

            int half = CoefficientLength(m, wavelet, boundary);
            var child = Matrix<double>.Uninitialized(half, 2 * parent.Columns);
            var output = child.AsSpan();
            for (int i = 0; i < parent.Columns; i++)
            {
                ForwardCore(parent.Column(i), wavelet, boundary, output.Slice(2 * i * half, half), output.Slice((2 * i + 1) * half, half));
            }

            tree[level] = child;
        }

        return new WaveletPacketTree(tree, wavelet, boundary);
    }

    /// <summary>
    /// Reconstructs a signal from the specified wavelet packet basis.
    /// <para>指定したウェーブレットパケット基底から信号を再構成します。</para>
    /// </summary>
    /// <param name="tree">The wavelet packet tree containing the coefficients. / 係数を保持するウェーブレットパケット木。</param>
    /// <param name="basis">The sequence of <c>(Level, Index)</c> nodes forming a complete, non-overlapping partition of the tree's leaves. / 木の全葉を重複なく覆う <c>(Level, Index)</c> ノードの列。</param>
    /// <returns>A newly allocated vector containing the reconstructed signal with the original signal length. / 元の信号と同じ長さの再構成信号を格納した新しいベクトル。</returns>
    /// <remarks>
    /// <para>
    /// Suitable bases include the result of <see cref="WaveletPacketTree.BestBasis()"/> or all nodes at a single level.
    /// </para>
    /// <para>
    /// <see cref="WaveletPacketTree.BestBasis()"/> の結果や、同じ段の全ノードを基底として指定できます。
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="tree"/> or <paramref name="basis"/> is <see langword="null"/>. / <paramref name="tree"/> または <paramref name="basis"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">A node level or index is out of range. / 段番号またはノード番号が範囲外の場合。</exception>
    /// <exception cref="ArgumentException">Nodes are duplicated, overlap as ancestor and descendant, or do not cover all leaves. / ノードが重複している場合、祖先と子孫の関係で重なっている場合、またはすべての葉を覆っていない場合。</exception>
    public static Vector<double> Reconstruct(WaveletPacketTree tree, IEnumerable<(int Level, int Index)> basis)
    {
        ArgumentNullException.ThrowIfNull(tree);
        ArgumentNullException.ThrowIfNull(basis);
        var nodes = new HashSet<(int Level, int Index)>();
        var intervals = new List<(long Start, long End)>();
        int depth = tree.Levels;
        foreach (var node in basis)
        {
            tree.RequireNode(node.Level, node.Index);
            if (!nodes.Add(node))
            {
                throw new ArgumentException($"The node ({node.Level}, {node.Index}) appears more than once.", nameof(basis));
            }

            int shift = depth - node.Level;
            intervals.Add(((long)node.Index << shift, ((long)node.Index + 1) << shift));
        }

        // Verify that the node intervals partition [0, 2^depth) with no overlaps or gaps.
        intervals.Sort();
        long covered = 0;
        foreach (var (start, end) in intervals)
        {
            if (start != covered)
            {
                throw new ArgumentException(
                    start < covered ? "The nodes overlap: one is an ancestor of another." : "The nodes do not cover the whole tree.", nameof(basis));
            }

            covered = end;
        }

        if (covered != 1L << depth)
        {
            throw new ArgumentException("The nodes do not cover the whole tree.", nameof(basis));
        }

        var result = Vector<double>.Uninitialized(tree.Length);
        ReconstructNode(tree, nodes, 0, 0).CopyTo(result.AsSpan());
        return result;
    }

    /// <summary>
    /// Reconstructs a signal from all wavelet packet nodes at the specified level.
    /// <para>指定した段の全ノードから信号を再構成します。</para>
    /// </summary>
    /// <param name="tree">The wavelet packet tree containing the coefficients. / 係数を保持するウェーブレットパケット木。</param>
    /// <param name="level">The level index to reconstruct from, from 0 to <see cref="WaveletPacketTree.Levels"/>. / 再構成に使用する 0 以上 <see cref="WaveletPacketTree.Levels"/> 以下の段番号。</param>
    /// <returns>A newly allocated vector containing the reconstructed signal with the original signal length. / 元の信号と同じ長さの再構成信号を格納した新しいベクトル。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="tree"/> is <see langword="null"/>. / <paramref name="tree"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="level"/> is negative or greater than <see cref="WaveletPacketTree.Levels"/>. / <paramref name="level"/> が負または <see cref="WaveletPacketTree.Levels"/> より大きい場合。</exception>
    public static Vector<double> Reconstruct(WaveletPacketTree tree, int level)
    {
        ArgumentNullException.ThrowIfNull(tree);
        ArgumentOutOfRangeException.ThrowIfNegative(level);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(level, tree.Levels);
        var basis = new (int Level, int Index)[1 << level];
        for (int i = 0; i < basis.Length; i++)
        {
            basis[i] = (level, i);
        }

        return Reconstruct(tree, basis);
    }

    private static double[] ReconstructNode(WaveletPacketTree tree, HashSet<(int Level, int Index)> basis, int level, int index)
    {
        var matrix = tree.LevelMatrix(level);
        int length = matrix.Rows;
        var result = new double[length];
        if (basis.Contains((level, index)))
        {
            matrix.AsSpan().Slice(index * length, length).CopyTo(result);
            return result;
        }

        var approximation = ReconstructNode(tree, basis, level + 1, 2 * index);
        var detail = ReconstructNode(tree, basis, level + 1, 2 * index + 1);
        int full = InverseLength(approximation.Length, tree.Wavelet, tree.Boundary);
        double[] buffer = ArrayPool<double>.Shared.Rent(full);
        InverseCore((ReadOnlySpan<double>)approximation, (ReadOnlySpan<double>)detail, tree.Wavelet, tree.Boundary, buffer.AsSpan(0, full));
        buffer.AsSpan(0, length).CopyTo(result);
        ArrayPool<double>.Shared.Return(buffer);
        return result;
    }
}
