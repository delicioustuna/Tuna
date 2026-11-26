using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

/// <summary>
/// Records operations for reverse-mode automatic differentiation.
/// <para>逆モード自動微分のために演算を記録するテープを表します。</para>
/// </summary>
/// <remarks>
/// <para>
/// A tape is not thread-safe and must not be accessed concurrently from multiple threads.
/// Each recording supports a single <see cref="Backward"/> pass; call <see cref="Reset"/> before starting the next computation.
/// </para>
/// <para>
/// 同一のテープを複数スレッドから同時に操作することはできません。
/// 1 回の記録につき <see cref="Backward"/> は 1 回だけ実行でき、次の計算を始める前に <see cref="Reset"/> を呼び出します。
/// </para>
/// </remarks>
public sealed partial class Tape
{
    // Edges for scalar node i occupy [edgeEnds[i - 1], edgeEnds[i]) (starting at 0 for i = 0).
    // Leaves have no edges; operations with all-constant inputs are not recorded.
    private int[] edgeEnds = [];
    private double[] adjoints = [];
    private int[] inputIndices = [];
    private double[] derivatives = [];
    private int[] leafIndices = [];
    private ScalarNode[] leaves = [];
    private int nodeCount;
    private int edgeCount;
    private int leafCount;
    private int generation;
    private RecordingState state;

    private enum RecordingState { Ready, Forward, Backward, Complete, Failed }

    internal int Generation
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => generation;
    }

    /// <summary>
    /// Initializes a new empty automatic differentiation tape.
    /// <para>空の自動微分テープを初期化します。</para>
    /// </summary>
    public Tape() { }

    /// <summary>
    /// Discards the current recording and prepares the tape for reuse.
    /// <para>現在の記録を破棄し、次の計算で再利用できる状態に戻します。</para>
    /// </summary>
    /// <remarks>
    /// <para>
    /// All previously created nodes and borrowed views referencing tape storage become invalid after reset.
    /// Backing buffers are retained for reuse. Pooled array buffers are tracked per element type (<see langword="float"/> and <see langword="double"/>), and least-recently-used idle buffers exceeding three times the peak element count of a single recording are released.
    /// Backward scratch buffers are excluded from this limit and retain their peak size. Discard the tape instance to release all retained memory.
    /// </para>
    /// <para>
    /// 以前に作成したノードおよびテープ領域を借用するビューはリセット後に無効になります。
    /// 内部バッファは次の計算のために保持されます。配列の値・勾配用プールは要素型（<see langword="float"/>・<see langword="double"/>）ごとに管理され、1 回の記録で使用した最大要素数の 3 倍を超える未使用バッファは古いものから解放されます。
    /// 逆伝播用の一時バッファはこの上限に含まれず、最大サイズを保持します。保持領域をすべて解放するには、テープのインスタンス自体を破棄してください。
    /// </para>
    /// </remarks>
    public void Reset()
    {
        EnsureNotApplyingOperation();
        // 32-bit generation wraps around after 2^32 resets.
        generation = unchecked(generation + 1);
        // Do not clear leaves so stale spans returned by Leaves() still hit generation checks instead of reading default constant 0.
        nodeCount = 0;
        edgeCount = 0;
        leafCount = 0;
        state = RecordingState.Ready;
        ResetArrays();
    }

    /// <summary>
    /// Creates a differentiable scalar leaf node with the specified value.
    /// <para>指定した値を持つ微分可能なスカラーの葉ノードを作成します。</para>
    /// </summary>
    /// <param name="value">The input value. / 入力値。</param>
    /// <returns>A scalar leaf node representing the differentiable input. / 微分対象の入力を表すスカラーの葉ノード。</returns>
    /// <exception cref="InvalidOperationException">Thrown if <see cref="Reset"/> has not been called since the previous <see cref="Backward"/> pass. / 逆伝播の完了後に <see cref="Reset"/> を呼ばずに入力を追加した場合。</exception>
    public ScalarNode Leaf(double value)
    {
        EnsureRecording();
        EnsureLeafCapacity(checked(leafCount + 1));
        EnsureNodeCapacity(checked(nodeCount + 1));
        ScalarNode leaf = new(value, this, AppendNode(), generation);
        leafIndices[leafCount] = leaf.Index;
        leaves[leafCount++] = leaf;
        return leaf;
    }

    /// <summary>
    /// Creates scalar leaf nodes in order for each element of the specified vector.
    /// <para>指定したベクトルの各要素から、順にスカラーの葉ノードを作成します。</para>
    /// </summary>
    /// <param name="values">The input values to copy into scalar leaves. / 各要素からスカラーの葉ノードを作成する入力値（値はコピーされます）。</param>
    /// <returns>A read-only span of scalar leaves in element order, borrowing tape storage that becomes invalid after <see cref="Reset"/>. / 要素順に並んだスカラーの葉ノードの読み取り専用スパン。テープの領域を借用するため、<see cref="Reset"/> 後は参照できません。</returns>
    /// <exception cref="InvalidOperationException">Thrown if <see cref="Reset"/> has not been called since the previous <see cref="Backward"/> pass. / 逆伝播の完了後に <see cref="Reset"/> を呼ばずに入力を追加した場合。</exception>
    public ReadOnlySpan<ScalarNode> Leaves(ReadOnlyVectorSpan<double> values)
    {
        EnsureRecording();
        int start = leafCount;
        EnsureLeafCapacity(checked(leafCount + values.Length));
        for (int i = 0; i < values.Length; i++)
        {
            Leaf(values[i]);
        }

        return leaves.AsSpan(start, values.Length);
    }

    /// <summary>
    /// Runs reverse-mode backpropagation starting from <paramref name="output"/> with an initial adjoint of 1.
    /// <para><paramref name="output"/> の随伴を 1 として逆伝播を実行します。</para>
    /// </summary>
    /// <remarks>
    /// <para>If backpropagation fails, call <see cref="Reset"/> before reusing the tape. Contributions already added to parameter gradients remain. Clear them with <c>ParameterSet&lt;T&gt;.ZeroGradients()</c> or <c>Parameter&lt;T&gt;.ZeroGradient()</c> before evaluating again.</para>
    /// <para>逆伝播が失敗した場合は、テープを再利用する前に <see cref="Reset"/> を呼び出します。パラメータに加算済みの勾配は残るため、再評価の前に <c>ParameterSet&lt;T&gt;.ZeroGradients()</c> または <c>Parameter&lt;T&gt;.ZeroGradient()</c> でクリアしてください。</para>
    /// </remarks>
    /// <param name="output">The output scalar node to differentiate; if it is a constant, all gradients are 0. / 勾配計算の起点となる出力ノード。定数の場合、すべての勾配は 0 になります。</param>
    /// <exception cref="InvalidOperationException">Thrown if <paramref name="output"/> belongs to an earlier generation or another tape, or if <see cref="Backward"/> has already been run without <see cref="Reset"/>. / ノードの世代が古い、別のテープに属している、または <see cref="Reset"/> せずに 2 回目の逆伝播を実行した場合。</exception>
    public void Backward(ScalarNode output)
    {
        EnsureRecording();
        ValidateOwned(output);
        Array.Clear(adjoints, 0, nodeCount);
        if (output.Tape is not null)
        {
            adjoints[output.Index] = 1;
        }

        if (arrayRecordCount != 0)
        {
            state = RecordingState.Backward;
            try
            {
                BackwardWithArrays();
                state = RecordingState.Complete;
            }
            catch
            {
                state = RecordingState.Failed;
                throw;
            }

            return;
        }

        for (int i = nodeCount - 1; i >= 0; i--)
        {
            double adjoint = adjoints[i];
            if (adjoint == 0)
            {
                continue;
            }

            int start = i == 0 ? 0 : edgeEnds[i - 1];
            for (int j = start; j < edgeEnds[i]; j++)
            {
                double derivative = derivatives[j];
                if (derivative != 0)
                {
                    adjoints[inputIndices[j]] += adjoint * derivative;
                }
            }
        }

        state = RecordingState.Complete;
    }

    /// <summary>
    /// Runs backpropagation from <paramref name="output"/> and writes the flattened leaf gradients to <paramref name="gradient"/>.
    /// <para><paramref name="output"/> から逆伝播を実行し、すべての葉の勾配を <paramref name="gradient"/> に書き込みます。</para>
    /// </summary>
    /// <param name="output">The output scalar node to differentiate. / 勾配計算の起点となる出力ノード。</param>
    /// <param name="gradient">The destination span receiving leaf gradients in creation order (column-major for matrix leaves). / 葉を作成した順に勾配を書き込む出力先（行列の葉は列優先）。</param>
    /// <returns>The value of <paramref name="output"/>. / <paramref name="output"/> の値。</returns>
    /// <remarks>
    /// <para>
    /// Array leaves are flattened in element order (column-major for matrices) at the position where each leaf was created.
    /// Gradients of <see langword="float"/> leaves are widened to <see langword="double"/>.
    /// </para>
    /// <para>
    /// 配列の葉は作成した位置に要素順（行列は列優先）で展開されます。
    /// <see langword="float"/> の葉の勾配は <see langword="double"/> に拡張されます。
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the length of <paramref name="gradient"/> does not match the total number of leaf elements. / 出力先の長さが葉の総要素数と一致しない場合。</exception>
    /// <exception cref="InvalidOperationException">Thrown if <paramref name="output"/> belongs to an earlier generation or another tape, or if <see cref="Backward"/> has already been run without <see cref="Reset"/>. / ノードの世代が古い、別のテープに属している、または <see cref="Reset"/> せずに 2 回目の逆伝播を実行した場合。</exception>
    public double Gradient(ScalarNode output, VectorSpan<double> gradient)
    {
        if (gradient.Length != (arrayLeafCount == 0 ? leafCount : FlatLeafLength()))
        {
            throw new ArgumentException("The gradient length must equal the number of leaves.", nameof(gradient));
        }

        Backward(output);
        if (arrayLeafCount != 0)
        {
            WriteFlatGradient(gradient);
            return output.Value;
        }

        for (int i = 0; i < leafCount; i++)
        {
            gradient[i] = adjoints[leafIndices[i]];
        }

        return output.Value;
    }

    /// <summary>
    /// Gets the gradient of the specified scalar node computed by <see cref="Backward"/>.
    /// <para><see cref="Backward"/> 実行後の指定したスカラーノードの勾配を取得します。</para>
    /// </summary>
    /// <param name="node">The scalar node whose gradient to retrieve. / 勾配を取得するスカラーノード。</param>
    /// <returns>The derivative of the output with respect to <paramref name="node"/>, or 0 for constants and unreached nodes. / 出力の <paramref name="node"/> に対する偏微分値。定数や逆伝播が到達しなかったノードの場合は 0。</returns>
    /// <exception cref="InvalidOperationException">Thrown if <see cref="Backward"/> has not been called, or if <paramref name="node"/> belongs to an earlier generation or another tape. / 逆伝播の実行前であるか、ノードの世代が古い、または別のテープに属している場合。</exception>
    public double GetGradient(ScalarNode node)
    {
        ValidateOwned(node);
        EnsureBackwardComplete();

        return node.Tape is null ? 0 : adjoints[node.Index];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal void ValidateGeneration(int nodeGeneration)
    {
        if (nodeGeneration != generation)
        {
            ThrowOldGeneration();
        }
    }

    // Callers validate generation and ownership up front; constant inputs are encoded as index -1.
    [MethodImpl(MethodImplOptions.NoInlining)]
    internal int Append(int firstIndex, double firstDerivative, int secondIndex, double secondDerivative)
    {
        EnsureRecording();
        int node = nodeCount, edge = edgeCount;
        // Paired arrays always grow together, so checking one length suffices.
        if ((uint)node >= (uint)edgeEnds.Length)
        {
            EnsureNodeCapacity(checked(node + 1));
        }

        if (inputIndices.Length - edge < 2)
        {
            EnsureEdgeCapacity(checked(edge + 2));
        }

        int[] indices = inputIndices;
        double[] values = derivatives;
        if (firstIndex >= 0)
        {
            indices[edge] = firstIndex;
            values[edge++] = firstDerivative;
        }

        if (secondIndex >= 0)
        {
            indices[edge] = secondIndex;
            values[edge++] = secondDerivative;
        }

        edgeCount = edge;
        edgeEnds[node] = edge;
        nodeCount = node + 1;
        return node;
    }

    internal ScalarNode Record(double value, ReadOnlySpan<(ScalarNode Input, double Derivative)> inputs)
    {
        EnsureRecording();
        int count = 0;
        foreach (var (input, _) in inputs)
        {
            if (input.Tape is not null)
            {
                count++;
            }
        }

        EnsureRecordCapacity(count);
        foreach (var (input, derivative) in inputs)
        {
            if (input.Tape is not null)
            {
                inputIndices[edgeCount] = input.Index;
                derivatives[edgeCount++] = derivative;
            }
        }

        return new(value, this, AppendNode(), generation);
    }

    internal ScalarNode Record(double value, ReadOnlySpan<ScalarNode> inputs, ReadOnlySpan<double> localDerivatives)
    {
        EnsureRecording();
        int count = 0;
        foreach (var input in inputs)
        {
            if (input.Tape is not null)
            {
                count++;
            }
        }

        EnsureRecordCapacity(count);
        for (int i = 0; i < inputs.Length; i++)
        {
            if (inputs[i].Tape is not null)
            {
                inputIndices[edgeCount] = inputs[i].Index;
                derivatives[edgeCount++] = localDerivatives[i];
            }
        }

        return new(value, this, AppendNode(), generation);
    }

    private void EnsureRecordCapacity(int inputCount)
    {
        // Reserve both buffers before writing edges so OOM cannot leave a partial record.
        EnsureEdgeCapacity(checked(edgeCount + inputCount));
        EnsureNodeCapacity(checked(nodeCount + 1));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int AppendNode()
    {
        int index = nodeCount++;
        edgeEnds[index] = edgeCount;
        return index;
    }

    private void ValidateOwned(ScalarNode node)
    {
        node.Validate();
        if (node.Tape is not null && node.Tape != this)
        {
            throw new InvalidOperationException("The node belongs to another tape.");
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void EnsureRecording()
    {
        if (state != RecordingState.Ready)
        {
            ThrowCannotRecord();
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal void EnsureCanApply(bool requiresGradient)
    {
        if (state != RecordingState.Ready)
        {
            EnsureCanApplyAfterRecording(requiresGradient);
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private void EnsureCanApplyAfterRecording(bool requiresGradient)
    {
        if (requiresGradient || state != RecordingState.Complete)
        {
            ThrowCannotRecord();
        }
    }

    [DoesNotReturn]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private void ThrowCannotRecord()
    {
        EnsureNotApplyingOperation();
        if (state == RecordingState.Failed)
        {
            ThrowBackwardFailed();
        }

        ThrowBackwardComplete();
    }

    private void EnsureNodeCapacity(int required)
    {
        if (edgeEnds.Length < required)
        {
            // Resize both paired arrays before assigning so OOM keeps their lengths synchronized.
            int[] newEnds = edgeEnds;
            double[] newAdjoints = adjoints;
            EnsureCapacity(ref newEnds, required);
            EnsureCapacity(ref newAdjoints, required);
            edgeEnds = newEnds;
            adjoints = newAdjoints;
        }
    }

    private void EnsureEdgeCapacity(int required)
    {
        if (inputIndices.Length < required)
        {
            int[] newIndices = inputIndices;
            double[] newDerivatives = derivatives;
            EnsureCapacity(ref newIndices, required);
            EnsureCapacity(ref newDerivatives, required);
            inputIndices = newIndices;
            derivatives = newDerivatives;
        }
    }

    private void EnsureLeafCapacity(int required)
    {
        EnsureCapacity(ref leafIndices, required);
        EnsureCapacity(ref leaves, required);
    }

    private static void EnsureCapacity<T>(ref T[] storage, int required)
    {
        if (storage.Length < required)
        {
            int capacity = (int)Math.Min(Array.MaxLength, Math.Max(16L, Math.Max(required, 2L * storage.Length)));
            Array.Resize(ref storage, capacity);
        }
    }

    [DoesNotReturn]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ThrowOldGeneration() => throw new InvalidOperationException("The node belongs to an earlier tape generation.");

    [DoesNotReturn]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ThrowBackwardComplete() => throw new InvalidOperationException("Call Reset before recording or running Backward again.");

    [DoesNotReturn]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ThrowBackwardFailed() => throw new InvalidOperationException("Backpropagation failed. Call Reset before reusing the tape.");
}
