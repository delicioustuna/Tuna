using System.Numerics;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

/// <summary>
/// Represents a three-dimensional tensor in a reverse-mode recording.
/// <para>逆モード自動微分の記録に属する3階テンソルを表します。</para>
/// </summary>
/// <typeparam name="TElement">The element type, either <see langword="float"/> or <see langword="double"/>. / 要素型。<see langword="float"/> または <see langword="double"/>。</typeparam>
/// <remarks>
/// <para>The first two axes are the row and column axes. The last axis indexes the matrices. Transposes and slices share storage with the source node.</para>
/// <para>最初の2軸は行と列です。最後の軸は行列を指定します。転置やスライスは元のノードと格納領域を共有します。</para>
/// </remarks>
public readonly struct TensorNode<TElement> where TElement : unmanaged, IFloatingPointIeee754<TElement>
{
    internal readonly MatrixNode<TElement> Storage;
    internal readonly TensorView View;

    internal TensorNode(MatrixNode<TElement> storage, TensorView view)
    {
        Storage = storage;
        View = view;
    }

    /// <summary>
    /// Gets the row count of each matrix.
    /// <para>各行列の行数を取得します。</para>
    /// </summary>
    public int Rows => View.Rows;

    /// <summary>
    /// Gets the column count of each matrix.
    /// <para>各行列の列数を取得します。</para>
    /// </summary>
    public int Columns => View.Columns;

    /// <summary>
    /// Gets the number of matrices along the last axis.
    /// <para>最後の軸の行列数を取得します。</para>
    /// </summary>
    public int MatrixCount => View.Count;

    /// <summary>
    /// Gets a read-only view of the recorded values.
    /// <para>記録された値の読み取り専用ビューを取得します。</para>
    /// </summary>
    /// <value>A view valid until <see cref="Tape.Reset"/>. / <see cref="Tape.Reset"/> まで有効なビュー。</value>
    /// <exception cref="InvalidOperationException">The node is uninitialized or stale. / ノードが未初期化または古い世代の場合。</exception>
    public ReadOnlyTensorSpan<TElement> Value => View.Read(Storage.Value);

    /// <summary>
    /// Gets whether the node depends on a differentiable leaf or parameter.
    /// <para>ノードが微分対象の葉またはパラメーターに依存しているかどうかを取得します。</para>
    /// </summary>
    /// <exception cref="InvalidOperationException">The node is uninitialized or stale. / ノードが未初期化または古い世代の場合。</exception>
    public bool RequiresGradient => Storage.RequiresGradient;

    /// <summary>
    /// Gets a zero-copy view that transposes each matrix.
    /// <para>各行列を転置したコピーなしのビューを取得します。</para>
    /// </summary>
    /// <exception cref="InvalidOperationException">The node is uninitialized or stale. / ノードが未初期化または古い世代の場合。</exception>
    public TensorNode<TElement> T
    {
        get
        {
            Storage.Owner();
            return new(Storage, View with { Transposed = !View.Transposed });
        }
    }

    /// <summary>
    /// Gets a zero-copy view over the specified ranges.
    /// <para>指定した範囲のビューをコピーなしで取得します。</para>
    /// </summary>
    /// <param name="rows">The row range. / 行の範囲。</param>
    /// <param name="columns">The column range. / 列の範囲。</param>
    /// <param name="matrices">The matrix index range. / 行列インデックスの範囲。</param>
    /// <exception cref="InvalidOperationException">The node is uninitialized or stale. / ノードが未初期化または古い世代の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">A range is outside the corresponding axis. / 範囲が対応する軸の外側にある場合。</exception>
    public TensorNode<TElement> this[Range rows, Range columns, Range matrices]
    {
        get
        {
            Storage.Owner();
            return new(Storage, View.Slice(rows, columns, matrices));
        }
    }

    /// <summary>
    /// Gets a differentiable view of one matrix without copying.
    /// <para>指定した行列を微分可能なビューとしてコピーなしで取得します。</para>
    /// </summary>
    /// <param name="index">The zero-based matrix index. / 0 始まりの行列インデックス。</param>
    /// <returns>A matrix node that shares the tensor's values and gradients. / テンソルの値と勾配を共有する行列ノード。</returns>
    /// <exception cref="InvalidOperationException">The node is uninitialized or stale. / ノードが未初期化または古い世代の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is outside the matrix axis. / <paramref name="index"/> が行列軸の範囲外の場合。</exception>
    public MatrixNode<TElement> GetMatrix(int index)
    {
        var tape = Storage.Owner();
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, MatrixCount);
        if (View.Repeated)
        {
            var matrix = Storage[View.FirstRow..(View.FirstRow + View.RowCount), View.FirstColumn..(View.FirstColumn + View.ColumnCount)];
            return View.Transposed ? matrix.T : matrix;
        }

        int start = View.IsEmpty ? 0 : (View.FirstMatrix + index) * View.StorageColumns + View.FirstColumn;
        return new(tape, new ArrayView(Storage.View.Slot, View.FirstRow, start, View.RowCount, View.ColumnCount, View.Transposed));
    }

    /// <summary>
    /// Places the matrices side by side in a column-major matrix node.
    /// <para>各行列を横に並べた列優先の行列ノードを返します。</para>
    /// </summary>
    /// <returns>A differentiable matrix with <c>Rows</c> rows and <c>Columns * MatrixCount</c> columns. / 形状が <c>Rows × (Columns * MatrixCount)</c> の微分可能な行列。</returns>
    /// <remarks>
    /// <para>Uses a view when the physical storage already has the requested order. Other layouts are copied into tape storage.</para>
    /// <para>格納順が一致する場合はビューを使用します。そのほかの配置はテープの領域にコピーします。</para>
    /// </remarks>
    /// <exception cref="OverflowException">The resulting column count exceeds <see cref="int.MaxValue"/>. / 結果の列数が <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="InvalidOperationException">The node is invalid or the tape cannot record a required copy. / ノードが無効、または必要なコピーをテープへ記録できない場合。</exception>
    public MatrixNode<TElement> ToMatrix()
    {
        _ = Storage.Owner();
        int columns = checked(Columns * MatrixCount);
        if (!View.Repeated && !View.Transposed && View.RowCount == View.StorageRows && View.ColumnCount == View.StorageColumns)
        {
            int first = View.IsEmpty ? 0 : View.FirstMatrix * View.StorageColumns;
            return Storage[.., first..(first + columns)];
        }

        return XMathExtensions.PackMatrices(this, columns);
    }

    /// <summary>
    /// Records a user operation with a tensor result.
    /// <para>テンソルを返す利用者定義演算を記録します。</para>
    /// </summary>
    /// <typeparam name="TOperation">The differentiable operation type. / 微分可能な演算の型。</typeparam>
    /// <param name="operation">The operation whose output is presented as side-by-side matrices. / 横並びの行列として出力を受け取る演算。</param>
    /// <param name="inputs">The matrix inputs on the same tape. / 同じテープに属する行列入力。</param>
    /// <param name="rows">The row count of each output matrix. / 各出力行列の行数。</param>
    /// <param name="columns">The column count of each output matrix. / 各出力行列の列数。</param>
    /// <param name="matrixCount">The number of output matrices. / 出力の行列数。</param>
    /// <returns>The result tensor node. / 結果のテンソルノード。</returns>
    /// <remarks>
    /// <para>The operation receives its output and its output adjoint as matrices of shape <c>rows × (columns * matrixCount)</c>.</para>
    /// <para>演算には、出力とその随伴を形状 <c>rows × (columns * matrixCount)</c> の行列として渡します。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="operation"/> is <see langword="null"/>. / <paramref name="operation"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">An output dimension is negative. / 出力の次元が負の場合。</exception>
    /// <exception cref="OverflowException">The flattened column count or element count exceeds <see cref="int.MaxValue"/>. / フラットな列数または要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="inputs"/> is empty. / <paramref name="inputs"/> が空の場合。</exception>
    /// <exception cref="InvalidOperationException">An input is invalid or recording requires a reset. / 入力が無効、または記録にリセットが必要な場合。</exception>
    public static TensorNode<TElement> Record<TOperation>(TOperation operation, ReadOnlySpan<MatrixNode<TElement>> inputs,
        int rows, int columns, int matrixCount)
        where TOperation : IDifferentiableOperation<TElement>
    {
        ArgumentOutOfRangeException.ThrowIfNegative(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(columns);
        ArgumentOutOfRangeException.ThrowIfNegative(matrixCount);
        int flatColumns = checked(columns * matrixCount);
        return new(MatrixNode<TElement>.Record(operation, inputs, rows, flatColumns), TensorView.Whole(rows, columns, matrixCount));
    }

    /// <summary>
    /// Multiplies each pair of matrices at the same index.
    /// <para>同じインデックスにある行列の組ごとに行列積を計算します。</para>
    /// </summary>
    /// <param name="a">The left operand. / 左辺のテンソル。</param>
    /// <param name="b">The right operand. / 右辺のテンソル。</param>
    /// <returns>The differentiable batch of matrix products. / 行列積のバッチを表す微分可能なノード。</returns>
    /// <exception cref="ArgumentException">The inner dimensions or matrix counts do not match. / 積の内側の次元または行列数が一致しない場合。</exception>
    /// <exception cref="OverflowException">The output element count or flattened column count exceeds <see cref="int.MaxValue"/>. / 出力の要素数またはフラットな列数が <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="InvalidOperationException">An input is stale, inputs belong to different tapes, or recording requires a reset. / 入力が古い世代、別のテープに属する、または記録にリセットが必要な場合。</exception>
    public static TensorNode<TElement> operator *(TensorNode<TElement> a, TensorNode<TElement> b) => XMath.Multiply(a, b);
}
