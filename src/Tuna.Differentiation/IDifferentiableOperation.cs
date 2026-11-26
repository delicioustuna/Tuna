using System.Numerics;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

/// <summary>
/// Defines the forward evaluation and pullback of an array operation.
/// <para>配列演算の順伝播と逆伝播を定義します。</para>
/// </summary>
/// <remarks>
/// <para>Vectors are passed as single-column matrices. Use the supplied views during callbacks. Keep any referenced operation state valid until backpropagation finishes.</para>
/// <para>ベクトルは 1 列の行列として渡されます。コールバック内では渡されたビューを使用します。演算が参照する状態は逆伝播が終わるまで保持します。</para>
/// </remarks>
/// <typeparam name="T">The element type, either float or double. / 要素型。float または double。</typeparam>
public interface IDifferentiableOperation<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    /// <summary>
    /// Computes all output elements from the input values.
    /// <para>入力値から出力の全要素を計算します。</para>
    /// </summary>
    /// <param name="context">Input values and tape-owned storage for saved values. / 入力値と、保存値用のテープ所有領域。</param>
    /// <param name="destination">The destination to fill with the operation's output. / 演算結果の全要素を書き込む出力先。</param>
    void Forward(DifferentiableOperationContext<T> context, MatrixSpan<T> destination);

    /// <summary>
    /// Accumulates contributions to input gradients from the output adjoint.
    /// <para>出力の随伴から求めた寄与を、勾配が必要な入力に加算します。</para>
    /// </summary>
    /// <remarks>
    /// <para>Called for operations reached during backpropagation. Input gradient views may share storage.</para>
    /// <para>逆伝播が到達した演算について呼び出されます。入力の勾配ビューが領域を共有することがあります。</para>
    /// </remarks>
    /// <param name="context">Input values, saved values, and writable input gradients. / 入力値、保存値、および書き込み可能な入力勾配。</param>
    /// <param name="output">The operation's forward result. / 演算の順伝播の結果。</param>
    /// <param name="outputGradient">The adjoint of the operation's output. / 演算結果の随伴。</param>
    void PullBack(DifferentiableOperationContext<T> context, ReadOnlyMatrixSpan<T> output, ReadOnlyMatrixSpan<T> outputGradient);
}

/// <summary>
/// Provides values, gradients, and tape-owned storage during a user operation callback.
/// <para>利用者定義演算のコールバック中に、値、勾配、およびテープ所有の保存領域を提供します。</para>
/// </summary>
/// <typeparam name="T">The element type, either float or double. / 要素型。float または double。</typeparam>
public readonly ref struct DifferentiableOperationContext<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    private readonly ArrayStore<T> store;
    private readonly ReadOnlySpan<ArrayView> inputs;
    private readonly int output;
    private readonly bool backward;

    internal DifferentiableOperationContext(ArrayStore<T> store, ReadOnlySpan<ArrayView> inputs, int output, bool backward)
    {
        this.store = store;
        this.inputs = inputs;
        this.output = output;
        this.backward = backward;
    }

    /// <summary>
    /// Gets the number of inputs.
    /// <para>入力の個数を取得します。</para>
    /// </summary>
    public int Count => inputs.Length;

    /// <summary>
    /// Gets the input value as a matrix view.
    /// <para>指定した入力値を行列ビューとして取得します。</para>
    /// </summary>
    /// <param name="index">The zero-based input index. / 0 から始まる入力のインデックス。</param>
    /// <value>A read-only view of the input value. / 入力値の読み取り専用ビュー。</value>
    /// <exception cref="IndexOutOfRangeException">The index is outside the input range. / インデックスが入力の範囲外の場合。</exception>
    public ReadOnlyMatrixSpan<T> this[int index] => store.MatrixView(inputs[index]);

    /// <summary>
    /// Determines whether the specified input requires a gradient.
    /// <para>指定した入力が勾配を必要とするかどうかを判定します。</para>
    /// </summary>
    /// <param name="index">The zero-based input index. / 0 から始まる入力のインデックス。</param>
    /// <returns><see langword="true"/> if backpropagation can propagate to this input. / この入力に勾配を伝播できる場合は <see langword="true"/>。</returns>
    /// <exception cref="IndexOutOfRangeException">The index is outside the input range. / インデックスが入力の範囲外の場合。</exception>
    public bool RequiresGradient(int index) => store.RequiresGradient(inputs[index].Slot);

    /// <summary>
    /// Gets a writable view for accumulating contributions to the specified input gradient.
    /// <para>指定した入力の勾配に寄与を加算するための書き込み可能なビューを取得します。</para>
    /// </summary>
    /// <remarks>
    /// <para>For a broadcast input, contributions are reduced into the source gradient after the callback returns.</para>
    /// <para>ブロードキャストされた入力への寄与は、コールバック終了後に集約されて元の勾配に加算されます。</para>
    /// </remarks>
    /// <param name="index">The zero-based input index. / 0 から始まる入力のインデックス。</param>
    /// <returns>The writable gradient view, valid during this callback. / このコールバック中に有効な書き込み可能な勾配ビュー。</returns>
    /// <exception cref="IndexOutOfRangeException">The index is outside the input range. / インデックスが入力の範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">The operation is in its forward pass or the input is constant. / 順伝播の実行中、または指定した入力が定数の場合。</exception>
    public MatrixSpan<T> GetGradient(int index) => GetGradient(index, allowOverwrite: false, out _);

    internal ArrayStore<T> Store => store;

    internal MatrixSpan<T> GetGradient(int index, bool allowOverwrite, out bool overwrite)
    {
        ArrayView input = inputs[index];
        if (!backward || !store.RequiresGradient(input.Slot))
        {
            throw new InvalidOperationException("Only inputs requiring gradients have a writable gradient during PullBack.");
        }

        return store.OperationGradient(index, input, allowOverwrite, out overwrite);
    }

    /// <summary>
    /// Allocates a tape-owned buffer for values needed by the pullback.
    /// <para>逆伝播に必要な値を保存するテープ所有の領域を確保します。</para>
    /// </summary>
    /// <remarks>
    /// <para>Fill the uninitialized buffer during Forward. Buffers are indexed in allocation order within this operation, starting at zero, and released when the tape is reset. If all inputs are constant, buffers are released when Forward returns.</para>
    /// <para>初期化されていない領域には順伝播中に書き込みます。この演算内での確保順に 0 からインデックスが付き、テープをリセットすると返却されます。入力がすべて定数の場合は、順伝播の終了時に返却されます。</para>
    /// </remarks>
    /// <param name="length">The number of elements. / 要素数。</param>
    /// <returns>A writable span valid during this forward callback. / この順伝播のコールバック中に有効な書き込み可能スパン。</returns>
    /// <exception cref="ArgumentOutOfRangeException">The length is negative. / 長さが負の場合。</exception>
    /// <exception cref="InvalidOperationException">This method was called during PullBack. / 逆伝播中に呼び出した場合。</exception>
    public Span<T> AllocateSaved(int length)
    {
        if (backward)
        {
            throw new InvalidOperationException("Save values during Forward.");
        }

        ArgumentOutOfRangeException.ThrowIfNegative(length);
        return store.AllocateOperationSaved(output, length);
    }

    /// <summary>
    /// Reads a buffer saved by this operation's forward callback.
    /// <para>この演算の順伝播で保存した領域を読み取ります。</para>
    /// </summary>
    /// <param name="index">The zero-based allocation order. / 0 から始まる確保順のインデックス。</param>
    /// <returns>A read-only span valid during this pullback. / この逆伝播のコールバック中に有効な読み取り専用スパン。</returns>
    /// <exception cref="ArgumentOutOfRangeException">No buffer was allocated at the specified index. / 指定のインデックスに確保済みの領域がない場合。</exception>
    /// <exception cref="InvalidOperationException">This method was called during Forward. / 順伝播中に呼び出した場合。</exception>
    public ReadOnlySpan<T> GetSaved(int index)
    {
        if (!backward)
        {
            throw new InvalidOperationException("Read saved values during PullBack.");
        }

        return store.GetOperationSaved(output, index);
    }
}
