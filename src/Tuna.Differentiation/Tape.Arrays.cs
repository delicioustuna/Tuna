using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

public sealed partial class Tape
{
    // Array operations are ordered against scalar nodes by recording the scalar nodeCount (Position).
    // In leafIndices, array leaves are encoded as ~arrayLeafIndex.
    private ArrayStore<double>? doubles;
    private ArrayStore<float>? singles;
    private ArrayRecord[] arrayRecords = [];
    private int arrayRecordCount;
    private ArrayLeaf[] arrayLeaves = [];
    private int arrayLeafCount;

    /// <summary>
    /// Creates a non-differentiable constant vector node referencing the specified vector without copying.
    /// <para>指定したベクトルをコピーせずに参照する定数ベクトルノードを作成します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
    /// <param name="values">The vector container to reference without copying. / コピーせずに参照するベクトルコンテナ。</param>
    /// <returns>A constant vector node that does not accumulate gradients. / 勾配を計算しない定数ベクトルノード。</returns>
    /// <remarks>
    /// <para>
    /// Do not mutate <paramref name="values"/> until backpropagation completes.
    /// </para>
    /// <para>
    /// 逆伝播が完了するまで <paramref name="values"/> の内容を変更しないでください。
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="values"/> is <see langword="null"/>. / <paramref name="values"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see langword="float"/> nor <see langword="double"/>. / 要素型が <see langword="float"/> でも <see langword="double"/> でもない場合。</exception>
    public VectorNode<T> Constant<T>(LinearAlgebra.Vector<T> values) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(values);
        EnsureCanApply(requiresGradient: false);
        int slot = Store<T>().AddConstant(values, values.Length, 1);
        return new(this, ArrayView.Whole(slot, values.Length, 1));
    }

    /// <summary>
    /// Creates a non-differentiable constant matrix node referencing the specified matrix without copying.
    /// <para>指定した行列をコピーせずに参照する定数行列ノードを作成します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
    /// <param name="values">The matrix container to reference without copying. / コピーせずに参照する行列コンテナ。</param>
    /// <returns>A constant matrix node that does not accumulate gradients. / 勾配を計算しない定数行列ノード。</returns>
    /// <remarks>
    /// <para>
    /// Do not mutate <paramref name="values"/> until backpropagation completes.
    /// Transposes and submatrices can be obtained without copying via <see cref="MatrixNode{TElement}.T"/> and range indexing.
    /// </para>
    /// <para>
    /// 逆伝播が完了するまで <paramref name="values"/> の内容を変更しないでください。
    /// 転置や部分行列は <see cref="MatrixNode{TElement}.T"/> や範囲インデクサーでコピーなしに取得できます。
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="values"/> is <see langword="null"/>. / <paramref name="values"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see langword="float"/> nor <see langword="double"/>. / 要素型が <see langword="float"/> でも <see langword="double"/> でもない場合。</exception>
    public MatrixNode<T> Constant<T>(Matrix<T> values) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(values);
        EnsureCanApply(requiresGradient: false);
        int slot = Store<T>().AddConstant(values, values.Rows, values.Columns);
        return new(this, ArrayView.Whole(slot, values.Rows, values.Columns));
    }

    /// <summary>
    /// Creates a non-differentiable constant vector node by copying the specified span into the tape.
    /// <para>指定したスパンをテープにコピーして定数ベクトルノードを作成します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
    /// <param name="values">The input vector view whose values are copied into the tape. / テープにコピーされる入力ベクトルビュー。</param>
    /// <returns>A constant vector node that does not accumulate gradients. / 勾配を計算しない定数ベクトルノード。</returns>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see langword="float"/> nor <see langword="double"/>. / 要素型が <see langword="float"/> でも <see langword="double"/> でもない場合。</exception>
    public VectorNode<T> Constant<T>(ReadOnlyVectorSpan<T> values) where T : unmanaged, IFloatingPointIeee754<T>
        => CopyVector(values, requiresGradient: false);

    /// <inheritdoc cref="Constant{T}(ReadOnlyVectorSpan{T})"/>
    public VectorNode<T> Constant<T>(VectorSpan<T> values) where T : unmanaged, IFloatingPointIeee754<T>
        => CopyVector(values.AsReadOnly(), requiresGradient: false);

    /// <summary>
    /// Creates a non-differentiable constant matrix node by copying the specified span into the tape.
    /// <para>指定したスパンをテープにコピーして定数行列ノードを作成します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
    /// <param name="values">The input matrix view whose values are copied into the tape. / テープにコピーされる入力行列ビュー。</param>
    /// <returns>A constant matrix node that does not accumulate gradients. / 勾配を計算しない定数行列ノード。</returns>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see langword="float"/> nor <see langword="double"/>. / 要素型が <see langword="float"/> でも <see langword="double"/> でもない場合。</exception>
    public MatrixNode<T> Constant<T>(ReadOnlyMatrixSpan<T> values) where T : unmanaged, IFloatingPointIeee754<T>
        => CopyMatrix(values, requiresGradient: false);

    /// <inheritdoc cref="Constant{T}(ReadOnlyMatrixSpan{T})"/>
    public MatrixNode<T> Constant<T>(MatrixSpan<T> values) where T : unmanaged, IFloatingPointIeee754<T>
        => CopyMatrix(values.AsReadOnly(), requiresGradient: false);

    /// <summary>
    /// Creates a differentiable vector leaf node by copying the specified values into the tape.
    /// <para>指定した値をテープにコピーして、微分可能なベクトルの葉ノードを作成します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
    /// <param name="values">The input values to copy into the tape. / テープにコピーされる入力値。</param>
    /// <returns>A vector leaf node representing the differentiable input. / 微分対象の入力を表すベクトルの葉ノード。</returns>
    /// <remarks>
    /// <para>
    /// In <see cref="Gradient(ScalarNode, VectorSpan{double})"/>, the elements of this leaf appear in order at the position where the leaf was created.
    /// </para>
    /// <para>
    /// <see cref="Gradient(ScalarNode, VectorSpan{double})"/> では、この葉を作成した位置に要素順で勾配が出力されます。
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown if <see cref="Reset"/> has not been called since the previous <see cref="Backward"/> pass. / 逆伝播の完了後に <see cref="Reset"/> を呼ばずに入力を追加した場合。</exception>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see langword="float"/> nor <see langword="double"/>. / 要素型が <see langword="float"/> でも <see langword="double"/> でもない場合。</exception>
    public VectorNode<T> Leaf<T>(ReadOnlyVectorSpan<T> values) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var node = CopyVector(values, requiresGradient: true);
        AddArrayLeaf<T>(node.View);
        return node;
    }

    /// <inheritdoc cref="Leaf{T}(ReadOnlyVectorSpan{T})"/>
    public VectorNode<T> Leaf<T>(VectorSpan<T> values) where T : unmanaged, IFloatingPointIeee754<T> => Leaf(values.AsReadOnly());

    /// <exception cref="ArgumentNullException"><paramref name="values"/> is <see langword="null"/>. / <paramref name="values"/> が <see langword="null"/> の場合。</exception>
    /// <inheritdoc cref="Leaf{T}(ReadOnlyVectorSpan{T})"/>
    public VectorNode<T> Leaf<T>(LinearAlgebra.Vector<T> values) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(values);
        return Leaf((ReadOnlyVectorSpan<T>)values);
    }

    /// <summary>
    /// Creates a differentiable matrix leaf node by copying the specified values into the tape.
    /// <para>指定した値をテープにコピーして、微分可能な行列の葉ノードを作成します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
    /// <param name="values">The input values to copy into the tape. / テープにコピーされる入力値。</param>
    /// <returns>A matrix leaf node representing the differentiable input. / 微分対象の入力を表す行列の葉ノード。</returns>
    /// <remarks>
    /// <para>
    /// Transposed column-major views (such as <c>X.T</c>) are copied in their underlying storage orientation so downstream operations produce the same bits as passing the view directly to <see cref="XMath"/>.
    /// Regardless of internal orientation, <see cref="Gradient(ScalarNode, VectorSpan{double})"/> writes the gradient in column-major order of the logical matrix at the leaf's creation position.
    /// </para>
    /// <para>
    /// 列優先行列の転置ビュー（<c>X.T</c> など）は格納時の向きのままコピーされ、後続の演算結果が元のビューを直接 <see cref="XMath"/> に渡した場合とビット単位で一致します。
    /// 内部の向きにかかわらず、<see cref="Gradient(ScalarNode, VectorSpan{double})"/> では作成した位置に論理行列の列優先順で勾配が出力されます。
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown if <see cref="Reset"/> has not been called since the previous <see cref="Backward"/> pass. / 逆伝播の完了後に <see cref="Reset"/> を呼ばずに入力を追加した場合。</exception>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see langword="float"/> nor <see langword="double"/>. / 要素型が <see langword="float"/> でも <see langword="double"/> でもない場合。</exception>
    public MatrixNode<T> Leaf<T>(ReadOnlyMatrixSpan<T> values) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var node = CopyMatrix(values, requiresGradient: true);
        AddArrayLeaf<T>(node.View);
        return node;
    }

    /// <inheritdoc cref="Leaf{T}(ReadOnlyMatrixSpan{T})"/>
    public MatrixNode<T> Leaf<T>(MatrixSpan<T> values) where T : unmanaged, IFloatingPointIeee754<T> => Leaf(values.AsReadOnly());

    /// <exception cref="ArgumentNullException"><paramref name="values"/> is <see langword="null"/>. / <paramref name="values"/> が <see langword="null"/> の場合。</exception>
    /// <inheritdoc cref="Leaf{T}(ReadOnlyMatrixSpan{T})"/>
    public MatrixNode<T> Leaf<T>(Matrix<T> values) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(values);
        return Leaf((ReadOnlyMatrixSpan<T>)values);
    }

    /// <summary>
    /// Attaches a vector <see cref="Parameter{T}"/> to the current recording as a differentiable node without copying.
    /// <para>ベクトルの学習パラメーター <see cref="Parameter{T}"/> をコピーせずに現在の記録へ参加させます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
    /// <param name="parameter">The vector parameter to reference without copying. / コピーせずに参照するベクトル型の学習パラメーター。</param>
    /// <returns>A vector node whose backward contributions accumulate into <see cref="Parameter{T}.Gradient"/>. / 逆伝播の寄与を <see cref="Parameter{T}.Gradient"/> に加算するベクトルノード。</returns>
    /// <remarks>
    /// <para>
    /// The same parameter may be attached multiple times in a recording; all contributions are summed into <see cref="Parameter{T}.Gradient"/>.
    /// Parameters are not included in the flat output of <see cref="Gradient(ScalarNode, VectorSpan{double})"/>.
    /// </para>
    /// <para>
    /// 同じパラメーターを複数回参加させることもでき、すべての寄与が <see cref="Parameter{T}.Gradient"/> に加算されます。
    /// パラメーターは <see cref="Gradient(ScalarNode, VectorSpan{double})"/> の出力対象には含まれません。
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="parameter"/> is <see langword="null"/>. / <paramref name="parameter"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="parameter"/> does not have rank 1. / パラメーターがベクトル（1 次元）でない場合。</exception>
    /// <exception cref="InvalidOperationException">Thrown if <see cref="Reset"/> has not been called since the previous <see cref="Backward"/> pass. / 逆伝播の完了後に <see cref="Reset"/> を呼ばずに参加させた場合。</exception>
    public VectorNode<T> UseVector<T>(Parameter<T> parameter) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(parameter);
        if (parameter.Rank != 1)
        {
            throw new ArgumentException("The parameter is not a vector.", nameof(parameter));
        }

        EnsureRecording();
        int slot = Store<T>().AddParameter(parameter.ValueArray, parameter.GradientArray, parameter.Rows, 1);
        return new(this, ArrayView.Whole(slot, parameter.Rows, 1));
    }

    /// <summary>
    /// Attaches a matrix <see cref="Parameter{T}"/> to the current recording as a differentiable node without copying.
    /// <para>行列の学習パラメーター <see cref="Parameter{T}"/> をコピーせずに現在の記録へ参加させます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
    /// <param name="parameter">The matrix parameter to reference without copying. / コピーせずに参照する行列型の学習パラメーター。</param>
    /// <returns>A matrix node whose backward contributions accumulate into <see cref="Parameter{T}.Gradient"/>. / 逆伝播の寄与を <see cref="Parameter{T}.Gradient"/> に加算する行列ノード。</returns>
    /// <remarks>
    /// <para>
    /// The same parameter may be attached multiple times in a recording; all contributions are summed into <see cref="Parameter{T}.Gradient"/>.
    /// Parameters are not included in the flat output of <see cref="Gradient(ScalarNode, VectorSpan{double})"/>.
    /// </para>
    /// <para>
    /// 同じパラメーターを複数回参加させることもでき、すべての寄与が <see cref="Parameter{T}.Gradient"/> に加算されます。
    /// パラメーターは <see cref="Gradient(ScalarNode, VectorSpan{double})"/> の出力対象には含まれません。
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="parameter"/> is <see langword="null"/>. / <paramref name="parameter"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="parameter"/> does not have rank 2. / パラメーターが行列（2 次元）でない場合。</exception>
    /// <exception cref="InvalidOperationException">Thrown if <see cref="Reset"/> has not been called since the previous <see cref="Backward"/> pass. / 逆伝播の完了後に <see cref="Reset"/> を呼ばずに参加させた場合。</exception>
    public MatrixNode<T> UseMatrix<T>(Parameter<T> parameter) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(parameter);
        if (parameter.Rank != 2)
        {
            throw new ArgumentException("The parameter is not a matrix.", nameof(parameter));
        }

        EnsureRecording();
        int slot = Store<T>().AddParameter(parameter.ValueArray, parameter.GradientArray, parameter.Rows, parameter.Columns);
        return new(this, ArrayView.Whole(slot, parameter.Rows, parameter.Columns));
    }

    /// <summary>
    /// Creates a non-differentiable constant vector node referencing the values of a <see cref="Parameter{T}"/> without copying.
    /// <para>ベクトルの学習パラメーター <see cref="Parameter{T}"/> を定数としてコピーせずに参照するノードを作成します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
    /// <param name="parameter">The vector parameter to reference as a constant. / 定数として参照するベクトル型の学習パラメーター。</param>
    /// <returns>A constant vector node referencing <paramref name="parameter"/>. / <paramref name="parameter"/> を参照する定数ベクトルノード。</returns>
    /// <remarks>
    /// <para>
    /// Use this when differentiating with respect to inputs while keeping parameters fixed.
    /// <see cref="Parameter{T}.Gradient"/> is not modified. Do not mutate <paramref name="parameter"/> until backpropagation completes.
    /// </para>
    /// <para>
    /// パラメーターを固定して入力について微分する場合に使用します。
    /// <see cref="Parameter{T}.Gradient"/> は更新されません。逆伝播が完了するまでパラメーターの値を変更しないでください。
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="parameter"/> is <see langword="null"/>. / <paramref name="parameter"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="parameter"/> does not have rank 1. / パラメーターがベクトル（1 次元）でない場合。</exception>
    public VectorNode<T> ConstantVector<T>(Parameter<T> parameter) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(parameter);
        if (parameter.Rank != 1)
        {
            throw new ArgumentException("The parameter is not a vector.", nameof(parameter));
        }

        EnsureCanApply(requiresGradient: false);
        int slot = Store<T>().AddConstant(parameter.ValueArray, parameter.Rows, 1);
        return new(this, ArrayView.Whole(slot, parameter.Rows, 1));
    }

    /// <summary>
    /// Creates a non-differentiable constant matrix node referencing the values of a <see cref="Parameter{T}"/> without copying.
    /// <para>行列の学習パラメーター <see cref="Parameter{T}"/> を定数としてコピーせずに参照するノードを作成します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
    /// <param name="parameter">The matrix parameter to reference as a constant. / 定数として参照する行列型の学習パラメーター。</param>
    /// <returns>A constant matrix node referencing <paramref name="parameter"/>. / <paramref name="parameter"/> を参照する定数行列ノード。</returns>
    /// <remarks>
    /// <para>
    /// Use this when differentiating with respect to inputs while keeping parameters fixed.
    /// <see cref="Parameter{T}.Gradient"/> is not modified. Do not mutate <paramref name="parameter"/> until backpropagation completes.
    /// </para>
    /// <para>
    /// パラメーターを固定して入力について微分する場合に使用します。
    /// <see cref="Parameter{T}.Gradient"/> は更新されません。逆伝播が完了するまでパラメーターの値を変更しないでください。
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="parameter"/> is <see langword="null"/>. / <paramref name="parameter"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="parameter"/> does not have rank 2. / パラメーターが行列（2 次元）でない場合。</exception>
    public MatrixNode<T> ConstantMatrix<T>(Parameter<T> parameter) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(parameter);
        if (parameter.Rank != 2)
        {
            throw new ArgumentException("The parameter is not a matrix.", nameof(parameter));
        }

        EnsureCanApply(requiresGradient: false);
        int slot = Store<T>().AddConstant(parameter.ValueArray, parameter.Rows, parameter.Columns);
        return new(this, ArrayView.Whole(slot, parameter.Rows, parameter.Columns));
    }

    /// <summary>
    /// Runs backpropagation from <paramref name="output"/> and writes the gradient of <paramref name="node"/> to <paramref name="gradient"/>.
    /// <para><paramref name="output"/> から逆伝播を実行し、<paramref name="node"/> の勾配を <paramref name="gradient"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
    /// <param name="output">The output scalar node to differentiate. / 勾配計算の起点となる出力ノード。</param>
    /// <param name="node">The vector node whose gradient to write. / 勾配を取得するベクトルノード。</param>
    /// <param name="gradient">The destination span receiving the gradient of <paramref name="node"/>. / <paramref name="node"/> の勾配を書き込む出力先。</param>
    /// <returns>The value of <paramref name="output"/>. / <paramref name="output"/> の値。</returns>
    /// <remarks>
    /// <para>
    /// Non-leaf and constant nodes are accepted (constant nodes yield a zero gradient). Gradients of other leaves are not written to <paramref name="gradient"/>.
    /// </para>
    /// <para>
    /// 中間ノードや定数ノードも指定できます（定数の勾配は 0 になります）。他の葉の勾配は <paramref name="gradient"/> には書き込まれません。
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the length of <paramref name="gradient"/> does not match <see cref="VectorNode{T}.Length"/>. / 出力先の長さがノードの要素数と一致しない場合。</exception>
    /// <exception cref="InvalidOperationException">Thrown if any node belongs to an earlier generation or another tape, or if <see cref="Backward"/> has already been run without <see cref="Reset"/>. / ノードの世代が古い、別のテープに属している、または <see cref="Reset"/> せずに 2 回目の逆伝播を実行した場合。</exception>
    public double Gradient<T>(ScalarNode output, VectorNode<T> node, VectorSpan<T> gradient) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ValidateOwned(node.Owner());
        if (gradient.Length != node.Length)
        {
            throw new ArgumentException("The gradient length must equal the length of the node.", nameof(gradient));
        }

        Backward(output);
        Store<T>().GradientVector(node.View).CopyTo(gradient);
        return output.Value;
    }

    /// <summary>
    /// Runs backpropagation from <paramref name="output"/> and writes the gradient of <paramref name="node"/> to <paramref name="gradient"/>.
    /// <para><paramref name="output"/> から逆伝播を実行し、<paramref name="node"/> の勾配を <paramref name="gradient"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
    /// <param name="output">The output scalar node to differentiate. / 勾配計算の起点となる出力ノード。</param>
    /// <param name="node">The matrix node whose gradient to write. / 勾配を取得する行列ノード。</param>
    /// <param name="gradient">The destination span receiving the gradient of <paramref name="node"/>. / <paramref name="node"/> の勾配を書き込む出力先。</param>
    /// <returns>The value of <paramref name="output"/>. / <paramref name="output"/> の値。</returns>
    /// <remarks>
    /// <para>
    /// Non-leaf and constant nodes are accepted (constant nodes yield a zero gradient). Gradients of other leaves are not written to <paramref name="gradient"/>.
    /// For broadcast views (<see cref="VectorNode{T}.Broadcast(int)"/> and <see cref="VectorNode{T}.BroadcastRows(int)"/>), the source vector's gradient is repeated to match the view's shape.
    /// </para>
    /// <para>
    /// 中間ノードや定数ノードも指定できます（定数の勾配は 0 になります）。他の葉の勾配は <paramref name="gradient"/> には書き込まれません。
    /// ブロードキャストビュー（<see cref="VectorNode{T}.Broadcast(int)"/>・<see cref="VectorNode{T}.BroadcastRows(int)"/>）の場合は、元のベクトルの勾配を同じ形状に繰り返して書き込みます。
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the shape of <paramref name="gradient"/> does not match <paramref name="node"/>. / 出力先の形状がノードと一致しない場合。</exception>
    /// <exception cref="InvalidOperationException">Thrown if any node belongs to an earlier generation or another tape, or if <see cref="Backward"/> has already been run without <see cref="Reset"/>. / ノードの世代が古い、別のテープに属している、または <see cref="Reset"/> せずに 2 回目の逆伝播を実行した場合。</exception>
    public double Gradient<T>(ScalarNode output, MatrixNode<T> node, MatrixSpan<T> gradient) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ValidateOwned(node.Owner());
        if (gradient.Rows != node.Rows || gradient.Columns != node.Columns)
        {
            throw new ArgumentException("The gradient shape must equal the shape of the node.", nameof(gradient));
        }

        Backward(output);
        Store<T>().GradientMatrix(node.View).CopyTo(gradient);
        return output.Value;
    }

    /// <summary>
    /// Gets a read-only view of the gradient of the specified vector node after <see cref="Backward"/>.
    /// <para><see cref="Backward"/> 実行後の指定したベクトルノードの勾配を読み取り専用ビューとして取得します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
    /// <param name="node">The vector node whose gradient to retrieve. / 勾配を取得するベクトルノード。</param>
    /// <returns>A read-only view of the gradient (0 for constants and unreached nodes). / 勾配の読み取り専用ビュー（定数や逆伝播が到達しなかったノードは 0）。</returns>
    /// <remarks>
    /// <para>
    /// The returned view borrows tape storage and must not be accessed after <see cref="Reset"/>.
    /// For nodes attached via <see cref="UseVector{T}(Parameter{T})"/>, returns the accumulated <see cref="Parameter{T}.Gradient"/>.
    /// </para>
    /// <para>
    /// 返されるビューはテープの領域を借用するため、<see cref="Reset"/> 後は参照しないでください。
    /// <see cref="UseVector{T}(Parameter{T})"/> で参加させたノードでは、加算後の <see cref="Parameter{T}.Gradient"/> を返します。
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown if <see cref="Backward"/> has not been called, or if <paramref name="node"/> belongs to an earlier generation or another tape. / 逆伝播の実行前であるか、ノードの世代が古い、または別のテープに属している場合。</exception>
    public ReadOnlyVectorSpan<T> GetGradient<T>(VectorNode<T> node) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ValidateOwned(node.Owner());
        EnsureBackwardComplete();
        return Store<T>().GradientVector(node.View);
    }

    /// <summary>
    /// Gets a read-only view of the gradient of the specified matrix node after <see cref="Backward"/>.
    /// <para><see cref="Backward"/> 実行後の指定した行列ノードの勾配を読み取り専用ビューとして取得します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
    /// <param name="node">The matrix node whose gradient to retrieve. / 勾配を取得する行列ノード。</param>
    /// <returns>A read-only view of the gradient (0 for constants and unreached nodes). / 勾配の読み取り専用ビュー（定数や逆伝播が到達しなかったノードは 0）。</returns>
    /// <remarks>
    /// <para>
    /// The returned view borrows tape storage and must not be accessed after <see cref="Reset"/>.
    /// For nodes attached via <see cref="UseMatrix{T}(Parameter{T})"/>, returns the accumulated <see cref="Parameter{T}.Gradient"/>.
    /// For broadcast views (<see cref="VectorNode{T}.Broadcast(int)"/> and <see cref="VectorNode{T}.BroadcastRows(int)"/>), the source vector's gradient is repeated to match the view's shape.
    /// </para>
    /// <para>
    /// 返されるビューはテープの領域を借用するため、<see cref="Reset"/> 後は参照しないでください。
    /// <see cref="UseMatrix{T}(Parameter{T})"/> で参加させたノードでは、加算後の <see cref="Parameter{T}.Gradient"/> を返します。
    /// ブロードキャストビュー（<see cref="VectorNode{T}.Broadcast(int)"/>・<see cref="VectorNode{T}.BroadcastRows(int)"/>）の場合は、元のベクトルの勾配を同じ形状に繰り返したビューを返します。
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown if <see cref="Backward"/> has not been called, or if <paramref name="node"/> belongs to an earlier generation or another tape. / 逆伝播の実行前であるか、ノードの世代が古い、または別のテープに属している場合。</exception>
    public ReadOnlyMatrixSpan<T> GetGradient<T>(MatrixNode<T> node) where T : unmanaged, IFloatingPointIeee754<T>
    {
        ValidateOwned(node.Owner());
        EnsureBackwardComplete();
        return Store<T>().GradientMatrix(node.View);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal ArrayStore<T> Store<T>() where T : unmanaged, IFloatingPointIeee754<T>
    {
        if (typeof(T) == typeof(double))
        {
            return Unsafe.As<ArrayStore<T>>(doubles ??= new());
        }

        if (typeof(T) == typeof(float))
        {
            return Unsafe.As<ArrayStore<T>>(singles ??= new());
        }

        TapeElementTypes.Require<T>();
        throw new UnreachableException();
    }

    internal void AppendArrayRecord<T>(ArrayOperation operation, int output, in ArrayView left, in ArrayView right)
        where T : unmanaged, IFloatingPointIeee754<T>
        => AppendArrayRecord<T>(operation, output, left, right, 0, -1);

    internal void AppendArrayRecord<T>(ArrayOperation operation, int output, in ArrayView left, in ArrayView right, double scalar, int scalarInput)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        EnsureRecording();
        int index = arrayRecordCount;
        if (index == arrayRecords.Length)
        {
            EnsureCapacity(ref arrayRecords, checked(index + 1));
        }

        ref ArrayRecord record = ref arrayRecords[index];
        record.Operation = operation;
        record.Single = typeof(T) == typeof(float);
        record.Position = nodeCount;
        record.Output = output;
        record.Left = left;
        record.Right = right;
        record.Scalar = scalar;
        record.ScalarInput = scalarInput;
        arrayRecordCount = index + 1;
    }

    // Allocates an edgeless scalar node whose adjoint is consumed by the subsequent array record.
    internal int AppendScalarOutput()
    {
        EnsureRecording();
        if ((uint)nodeCount >= (uint)edgeEnds.Length)
        {
            EnsureNodeCapacity(checked(nodeCount + 1));
        }

        return AppendNode();
    }

    private VectorNode<T> CopyVector<T>(ReadOnlyVectorSpan<T> values, bool requiresGradient) where T : unmanaged, IFloatingPointIeee754<T>
    {
        EnsureCanApply(requiresGradient);
        var store = Store<T>();
        int slot = store.AddPooled(values.Length, 1, requiresGradient);
        values.CopyTo(store.WritableVector(slot));
        return new(this, ArrayView.Whole(slot, values.Length, 1));
    }

    // Preserve transposed column-major layout on copy so downstream BLAS calls follow the exact same
    // transpose flag path and match XMath bit-for-bit; other layouts are packed column-major.
    private MatrixNode<T> CopyMatrix<T>(ReadOnlyMatrixSpan<T> values, bool requiresGradient) where T : unmanaged, IFloatingPointIeee754<T>
    {
        EnsureCanApply(requiresGradient);
        var store = Store<T>();
        int rows = values.Rows, columns = values.Columns;
        if (IsTransposedLayout(values))
        {
            int transposedSlot = store.AddPooled(columns, rows, requiresGradient);
            values.T.CopyTo(store.WritableMatrix(transposedSlot));
            return new(this, ArrayView.Whole(transposedSlot, columns, rows).Transpose());
        }

        int slot = store.AddPooled(rows, columns, requiresGradient);
        values.CopyTo(store.WritableMatrix(slot));
        return new(this, ArrayView.Whole(slot, rows, columns));
    }

    private static bool IsTransposedLayout<T>(ReadOnlyMatrixSpan<T> values) where T : unmanaged, IFloatingPointIeee754<T>
    {
        int rows = values.Rows, columns = values.Columns;
        nint rowStride = values.Strides[0], columnStride = values.Strides[1];
        bool columnMajor = (rows <= 1 || rowStride == 1) && (columns <= 1 || columnStride >= Math.Max(1, rows));
        return !columnMajor && (columns <= 1 || columnStride == 1) && (rows <= 1 || rowStride >= Math.Max(1, columns));
    }

    private void AddArrayLeaf<T>(in ArrayView view, int columnsPerMatrix = 0)
    {
        EnsureLeafCapacity(checked(leafCount + 1));
        if (arrayLeafCount == arrayLeaves.Length)
        {
            EnsureCapacity(ref arrayLeaves, checked(arrayLeafCount + 1));
        }

        arrayLeaves[arrayLeafCount] = new(typeof(T) == typeof(float), view.Slot, view.Transposed, columnsPerMatrix);
        leafIndices[leafCount] = ~arrayLeafCount;
        // Write a stale-generation node so borrowed spans from Leaves() throw instead of seeing default constant 0.
        leaves[leafCount++] = new(0, this, 0, unchecked(generation - 1));
        arrayLeafCount++;
    }

    private void ResetArrays()
    {
        arrayRecordCount = 0;
        arrayLeafCount = 0;
        doubles?.Reset();
        singles?.Reset();
    }

    private int FlatLeafLength()
    {
        long length = leafCount - arrayLeafCount;
        for (int i = 0; i < arrayLeafCount; i++)
        {
            ArrayLeaf leaf = arrayLeaves[i];
            length += leaf.Single ? singles!.Length(leaf.Slot) : doubles!.Length(leaf.Slot);
        }

        return length > int.MaxValue ? -1 : (int)length;
    }

    private void WriteFlatGradient(VectorSpan<double> gradient)
    {
        int position = 0;
        for (int i = 0; i < leafCount; i++)
        {
            int entry = leafIndices[i];
            if (entry >= 0)
            {
                gradient[position++] = adjoints[entry];
                continue;
            }

            ArrayLeaf leaf = arrayLeaves[~entry];
            position += leaf.Single
                ? singles!.CopyGradient(leaf.Slot, leaf.Transposed, gradient, position, leaf.ColumnsPerMatrix)
                : doubles!.CopyGradient(leaf.Slot, leaf.Transposed, gradient, position, leaf.ColumnsPerMatrix);
        }
    }

    // Interleaves reverse array records with scalar nodes at each record's recorded scalar Position.
    private void BackwardWithArrays()
    {
        double[] adjointValues = adjoints;
        ArrayRecord[] records = arrayRecords;
        int r = arrayRecordCount - 1;
        for (int i = nodeCount - 1; i >= 0; i--)
        {
            while (r >= 0 && records[r].Position > i)
            {
                BackwardArray(records[r--], adjointValues);
            }

            double adjoint = adjointValues[i];
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
                    adjointValues[inputIndices[j]] += adjoint * derivative;
                }
            }
        }

        while (r >= 0)
        {
            BackwardArray(records[r--], adjointValues);
        }
    }

    private void BackwardArray(in ArrayRecord record, double[] adjointValues)
    {
        if (record.Single)
        {
            singles!.Backward(record, adjointValues);
        }
        else
        {
            doubles!.Backward(record, adjointValues);
        }
    }

    private void ValidateOwned(Tape owner)
    {
        if (owner != this)
        {
            throw new InvalidOperationException("The node belongs to another tape.");
        }
    }

    private void EnsureBackwardComplete()
    {
        if (state == RecordingState.Failed)
        {
            ThrowBackwardFailed();
        }

        if (state != RecordingState.Complete)
        {
            throw new InvalidOperationException("Run Backward before reading gradients.");
        }
    }
}
