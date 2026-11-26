using System.Numerics;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

public sealed partial class Tape
{
    /// <summary>
    /// Creates a constant node referencing a tensor without copying.
    /// <para>テンソルをコピーせずに参照する定数ノードを作成します。</para>
    /// </summary>
    /// <typeparam name="T">The element type, either float or double. / 要素型。float または double。</typeparam>
    /// <param name="values">The tensor to reference. / 参照するテンソル。</param>
    /// <returns>A tensor node that does not accumulate gradients. / 勾配を加算しないテンソルノード。</returns>
    /// <remarks>
    /// <para>Keep the tensor values unchanged until backpropagation finishes.</para>
    /// <para>逆伝播が終わるまでテンソルの値を変更しないでください。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="values"/> is <see langword="null"/>. / <paramref name="values"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither float nor double. / 要素型が float でも double でもない場合。</exception>
    public TensorNode<T> Constant<T>(Tensor<T> values) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(values);
        EnsureCanApply(requiresGradient: false);
        var shape = values.Lengths;
        var view = TensorView.Whole((int)shape[0], (int)shape[1], (int)shape[2]);
        int slot = Store<T>().AddConstant(values, view.StorageRows, view.FlatColumns);
        return TensorFromSlot<T>(slot, view);
    }

    /// <summary>
    /// Creates a constant tensor node by copying the view into the tape.
    /// <para>テンソルビューをテープにコピーして定数ノードを作成します。</para>
    /// </summary>
    /// <typeparam name="T">The element type, either float or double. / 要素型。float または double。</typeparam>
    /// <param name="values">The input tensor view. / テンソルビュー。</param>
    /// <returns>A constant tensor node. / テンソルの定数ノード。</returns>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither float nor double. / 要素型が float でも double でもない場合。</exception>
    public TensorNode<T> Constant<T>(ReadOnlyTensorSpan<T> values) where T : unmanaged, IFloatingPointIeee754<T>
        => CopyTensor(values, requiresGradient: false);

    /// <inheritdoc cref="Constant{T}(ReadOnlyTensorSpan{T})"/>
    public TensorNode<T> Constant<T>(TensorSpan<T> values) where T : unmanaged, IFloatingPointIeee754<T>
        => Constant(values.AsReadOnly());

    /// <summary>
    /// Creates a differentiable tensor leaf by copying the view into the tape.
    /// <para>テンソルビューをテープにコピーして微分対象の葉を作成します。</para>
    /// </summary>
    /// <typeparam name="T">The element type, either float or double. / 要素型。float または double。</typeparam>
    /// <param name="values">The input tensor view. / テンソルビュー。</param>
    /// <returns>A differentiable tensor leaf. / 微分対象のテンソルの葉。</returns>
    /// <remarks>
    /// <para>Flattened gradients visit matrices in order and use logical column order within each matrix.</para>
    /// <para>フラットな勾配は行列順に出力され、各行列の中では論理上の列優先順になります。</para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Recording requires a reset. / 記録にリセットが必要な場合。</exception>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither float nor double. / 要素型が float でも double でもない場合。</exception>
    public TensorNode<T> Leaf<T>(ReadOnlyTensorSpan<T> values) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var node = CopyTensor(values, requiresGradient: true);
        var storage = node.Storage.View;
        AddArrayLeaf<T>(node.View.Transposed ? storage.Transpose() : storage, node.View.StorageColumns);
        return node;
    }

    /// <inheritdoc cref="Leaf{T}(ReadOnlyTensorSpan{T})"/>
    public TensorNode<T> Leaf<T>(TensorSpan<T> values) where T : unmanaged, IFloatingPointIeee754<T>
        => Leaf(values.AsReadOnly());

    /// <inheritdoc cref="Leaf{T}(ReadOnlyTensorSpan{T})"/>
    /// <exception cref="ArgumentNullException"><paramref name="values"/> is <see langword="null"/>. / <paramref name="values"/> が <see langword="null"/> の場合。</exception>
    public TensorNode<T> Leaf<T>(Tensor<T> values) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(values);
        return Leaf((ReadOnlyTensorSpan<T>)values);
    }

    /// <summary>
    /// Attaches a rank-three parameter to the tape without copying its values or gradients.
    /// <para>値と勾配をコピーせずに、3 階テンソルの学習パラメーターをテープへ参加させます。</para>
    /// </summary>
    /// <typeparam name="T">The parameter element type. / パラメーターの要素型。</typeparam>
    /// <param name="parameter">The tensor parameter. / テンソルの学習パラメーター。</param>
    /// <returns>A node that accumulates contributions into the parameter's gradient. / パラメーターの勾配に寄与を加算するノード。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="parameter"/> is <see langword="null"/>. / <paramref name="parameter"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="parameter"/> does not have rank three. / <paramref name="parameter"/> が 3 階テンソルでない場合。</exception>
    /// <exception cref="InvalidOperationException">Recording requires a reset. / 記録にリセットが必要な場合。</exception>
    public TensorNode<T> UseTensor<T>(Parameter<T> parameter) where T : unmanaged, IFloatingPointIeee754<T>
        => AttachTensor(parameter, requiresGradient: true);

    /// <summary>
    /// References the values of a rank-three parameter as a constant node.
    /// <para>3 階テンソルの学習パラメーターの値を定数ノードとして参照します。</para>
    /// </summary>
    /// <typeparam name="T">The parameter element type. / パラメーターの要素型。</typeparam>
    /// <param name="parameter">The tensor parameter. / テンソルの学習パラメーター。</param>
    /// <returns>A constant node referencing the parameter's values. / パラメーターの値を参照する定数ノード。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="parameter"/> is <see langword="null"/>. / <paramref name="parameter"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="parameter"/> does not have rank three. / <paramref name="parameter"/> が 3 階テンソルでない場合。</exception>
    public TensorNode<T> ConstantTensor<T>(Parameter<T> parameter) where T : unmanaged, IFloatingPointIeee754<T>
        => AttachTensor(parameter, requiresGradient: false);

    /// <summary>
    /// Gets a read-only gradient view after backpropagation.
    /// <para>逆伝播後の勾配を読み取り専用ビューとして取得します。</para>
    /// </summary>
    /// <typeparam name="T">The node element type. / ノードの要素型。</typeparam>
    /// <param name="node">The tensor node. / テンソルノード。</param>
    /// <returns>The accumulated gradient, or a zero view for unreached nodes. / 加算済みの勾配。逆伝播が到達しなかったノードではゼロのビュー。</returns>
    /// <remarks>
    /// <para>Broadcast tensor views repeat the source matrix's accumulated gradient. The view is valid until <see cref="Reset"/>.</para>
    /// <para>ブロードキャストビューは元の行列に加算された勾配を繰り返します。ビューは <see cref="Reset"/> まで有効です。</para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Backpropagation has not finished or the node is invalid for this tape. / 逆伝播が未完了、またはノードがこのテープで無効な場合。</exception>
    public ReadOnlyTensorSpan<T> GetGradient<T>(TensorNode<T> node) where T : unmanaged, IFloatingPointIeee754<T>
        => node.View.Read(GetGradient(node.Storage));

    /// <summary>
    /// Runs backpropagation and copies the gradient of the tensor node into the destination.
    /// <para>逆伝播を実行し、テンソルノードの勾配を出力先にコピーします。</para>
    /// </summary>
    /// <typeparam name="T">The node element type. / ノードの要素型。</typeparam>
    /// <param name="output">The scalar to differentiate. / 微分するスカラー。</param>
    /// <param name="node">The tensor node whose gradient is requested. / 勾配を取得するテンソルノード。</param>
    /// <param name="gradient">A destination that matches the tensor shape. / テンソルと同じ形状の出力先。</param>
    /// <returns>The scalar value. / スカラーの値。</returns>
    /// <exception cref="ArgumentException">The destination shape differs from the node shape. / 出力先とノードの形状が異なる場合。</exception>
    /// <exception cref="InvalidOperationException">A node is invalid for this tape, or backpropagation requires a reset. / ノードがこのテープで無効、または逆伝播にリセットが必要な場合。</exception>
    public double Gradient<T>(ScalarNode output, TensorNode<T> node, TensorSpan<T> gradient)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        ValidateOwned(node.Storage.Owner());
        var shape = gradient.Lengths;
        if (shape[0] != node.Rows || shape[1] != node.Columns || shape[2] != node.MatrixCount)
        {
            throw new ArgumentException("The gradient destination must have the same shape as the node.", nameof(gradient));
        }

        Backward(output);
        GetGradient(node).CopyTo(gradient);
        return output.Value;
    }

    private TensorNode<T> AttachTensor<T>(Parameter<T> parameter, bool requiresGradient) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(parameter);
        if (parameter.Rank != 3)
        {
            throw new ArgumentException("The parameter is not a tensor.", nameof(parameter));
        }

        EnsureCanApply(requiresGradient);
        var view = TensorView.Whole(parameter.Rows, parameter.Columns, parameter.MatrixCount);
        var store = Store<T>();
        int slot = requiresGradient
            ? store.AddParameter(parameter.ValueArray, parameter.GradientArray, view.StorageRows, view.FlatColumns)
            : store.AddConstant(parameter.ValueArray, view.StorageRows, view.FlatColumns);
        return TensorFromSlot<T>(slot, view);
    }

    private TensorNode<T> CopyTensor<T>(ReadOnlyTensorSpan<T> values, bool requiresGradient) where T : unmanaged, IFloatingPointIeee754<T>
    {
        EnsureCanApply(requiresGradient);
        var shape = values.Lengths;
        bool transposed = shape[2] != 0 && IsTransposedLayout(values.GetMatrix(0));
        var view = TensorView.Whole((int)shape[transposed ? 1 : 0], (int)shape[transposed ? 0 : 1], (int)shape[2]);
        var store = Store<T>();
        int slot = store.AddPooled(view.StorageRows, view.FlatColumns, requiresGradient);
        (transposed ? values.T : values).CopyTo(view.Write(store.WritableMatrix(slot)));
        return TensorFromSlot<T>(slot, view with { Transposed = transposed });
    }

    private TensorNode<T> TensorFromSlot<T>(int slot, TensorView view) where T : unmanaged, IFloatingPointIeee754<T>
        => new(new MatrixNode<T>(this, ArrayView.Whole(slot, view.StorageRows, view.FlatColumns)), view);
}
