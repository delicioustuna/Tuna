using System.Numerics;

namespace Tuna.Differentiation;

public sealed partial class Tape
{
    /// <summary>
    /// Applies a user operation and returns a vector node of the specified length.
    /// <para>利用者定義演算を適用し、指定した長さのベクトルノードを返します。</para>
    /// </summary>
    /// <remarks>
    /// <para>Inputs and outputs are passed as single-column matrices. When all inputs are constant, only forward evaluation runs.</para>
    /// <para>入力と出力は 1 列の行列として渡されます。入力がすべて定数の場合は順伝播だけを実行します。</para>
    /// </remarks>
    /// <typeparam name="T">The element type, either float or double. / 要素型。float または double。</typeparam>
    /// <typeparam name="TOperation">The differentiable operation type. / 微分可能な演算の型。</typeparam>
    /// <param name="operation">The user operation. / 利用者定義演算。</param>
    /// <param name="inputs">The input nodes owned by this tape. / このテープが所有する入力ノード。</param>
    /// <param name="length">The output length. / 出力の長さ。</param>
    /// <returns>The output vector node. / 出力ベクトルノード。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="operation"/> is <see langword="null"/>. / <paramref name="operation"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is negative. / <paramref name="length"/> が負の場合。</exception>
    /// <exception cref="InvalidOperationException">An input is invalid or belongs to a different tape, a callback is active, or recording requires a reset. / 入力が無効または別のテープに属する、コールバック中、または記録にリセットが必要な場合。</exception>
    public VectorNode<T> Apply<T, TOperation>(TOperation operation, ReadOnlySpan<VectorNode<T>> inputs, int length)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TOperation : IDifferentiableOperation<T>
    {
        EnsureCanApply(requiresGradient: false);
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        var store = Store<T>();
        Span<ArrayView> views = store.ReserveOperationInputs(inputs.Length);
        for (int i = 0; i < inputs.Length; i++)
        {
            ValidateOwned(inputs[i].Owner());
            views[i] = inputs[i].View;
        }

        return new(this, ApplyOperation(store, operation, views, length, 1));
    }

    /// <summary>
    /// Applies a user operation and returns a matrix node of the specified shape.
    /// <para>利用者定義演算を適用し、指定した形状の行列ノードを返します。</para>
    /// </summary>
    /// <remarks>
    /// <para>Use <see cref="VectorNode{T}.AsColumnMatrix"/> to pass a vector input. When all inputs are constant, only forward evaluation runs.</para>
    /// <para>ベクトル入力は <see cref="VectorNode{T}.AsColumnMatrix"/> で渡せます。入力がすべて定数の場合は順伝播だけを実行します。</para>
    /// </remarks>
    /// <typeparam name="T">The element type, either float or double. / 要素型。float または double。</typeparam>
    /// <typeparam name="TOperation">The differentiable operation type. / 微分可能な演算の型。</typeparam>
    /// <param name="operation">The user operation. / 利用者定義演算。</param>
    /// <param name="inputs">The input nodes owned by this tape. / このテープが所有する入力ノード。</param>
    /// <param name="rows">The output row count. / 出力の行数。</param>
    /// <param name="columns">The output column count. / 出力の列数。</param>
    /// <returns>The output matrix node. / 出力行列ノード。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="operation"/> is <see langword="null"/>. / <paramref name="operation"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">An output dimension is negative. / 出力の行数または列数が負の場合。</exception>
    /// <exception cref="InvalidOperationException">An input is invalid or belongs to a different tape, a callback is active, or recording requires a reset. / 入力が無効または別のテープに属する、コールバック中、または記録にリセットが必要な場合。</exception>
    public MatrixNode<T> Apply<T, TOperation>(TOperation operation, ReadOnlySpan<MatrixNode<T>> inputs, int rows, int columns)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TOperation : IDifferentiableOperation<T>
    {
        EnsureCanApply(requiresGradient: false);
        ArgumentOutOfRangeException.ThrowIfNegative(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(columns);
        var store = Store<T>();
        Span<ArrayView> views = store.ReserveOperationInputs(inputs.Length);
        for (int i = 0; i < inputs.Length; i++)
        {
            ValidateOwned(inputs[i].Owner());
            views[i] = inputs[i].View;
        }

        return new(this, ApplyOperation(store, operation, views, rows, columns));
    }

    private ArrayView ApplyOperation<T, TOperation>(
        ArrayStore<T> store, TOperation operation, ReadOnlySpan<ArrayView> inputs, int rows, int columns)
        where T : unmanaged, IFloatingPointIeee754<T>
        where TOperation : IDifferentiableOperation<T>
    {
        if (operation is null)
        {
            throw new ArgumentNullException(nameof(operation));
        }

        bool requiresGradient = false;
        foreach (var input in inputs)
        {
            requiresGradient |= store.RequiresGradient(input.Slot);
        }

        if (requiresGradient)
        {
            EnsureRecording();
        }

        int slot = store.AddPooled(rows, columns, requiresGradient);
        RecordingState previousState = state;
        state = RecordingState.Forward;
        try
        {
            operation.Forward(new(store, inputs, slot, backward: false), store.WritableMatrix(slot));
        }
        catch
        {
            store.DiscardOperationSlots(slot);
            throw;
        }
        finally
        {
            state = previousState;
        }

        if (requiresGradient)
        {
            int index = store.RecordOperation(operation, inputs.Length);
            AppendArrayRecord<T>(ArrayOperation.UserOperation, slot, default, default, 0, index);
        }
        else
        {
            store.DiscardOperationSavedSlots(slot);
        }

        return ArrayView.Whole(slot, rows, columns);
    }

    private void EnsureNotApplyingOperation()
    {
        if (state is RecordingState.Forward or RecordingState.Backward)
        {
            throw new InvalidOperationException("A user operation cannot modify its tape during Forward or PullBack.");
        }
    }
}
