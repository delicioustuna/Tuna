using System.Numerics;
using System.Runtime.CompilerServices;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

/// <summary>
/// Represents a matrix node for reverse-mode automatic differentiation.
/// <para>逆モード自動微分の行列ノードを表します。</para>
/// </summary>
/// <typeparam name="TElement">The floating-point element type (<see langword="float"/> or <see langword="double"/>; named <c>TElement</c> to avoid colliding with the transpose property <see cref="T"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>。転置プロパティ <see cref="T"/> との名前衝突を避けるため <c>TElement</c> としています）。</typeparam>
/// <remarks>
/// <para>
/// Created via <see cref="Tape.Leaf{T}(ReadOnlyMatrixSpan{T})"/>, <see cref="Tape.Constant{T}(Matrix{T})"/>, <see cref="Tape.UseMatrix{T}(Parameter{T})"/>, or differentiable operations.
/// <see cref="T"/> and range slicing return zero-copy views whose gradients accumulate into the corresponding elements of the source node.
/// Broadcast matrix views created via <see cref="VectorNode{T}.Broadcast(int)"/> and <see cref="VectorNode{T}.BroadcastRows(int)"/> also support transposition and slicing.
/// Supports <see langword="float"/> and <see langword="double"/>. Nodes created before <see cref="Tape.Reset"/> become invalid and cannot be used.
/// </para>
/// <para>
/// <see cref="Tape.Leaf{T}(ReadOnlyMatrixSpan{T})"/>、<see cref="Tape.Constant{T}(Matrix{T})"/>、<see cref="Tape.UseMatrix{T}(Parameter{T})"/>、および各種演算から作成します。
/// <see cref="T"/> および範囲インデクサーはコピーなしのビューを返し、勾配は元のノードの対応する要素に加算されます。
/// <see cref="VectorNode{T}.Broadcast(int)"/> や <see cref="VectorNode{T}.BroadcastRows(int)"/> で作成したブロードキャストビューに対しても転置や部分行列の取得が可能です。
/// 要素型は <see langword="float"/> と <see langword="double"/> に対応します。<see cref="Tape.Reset"/> より前に作成したノードは無効になり使用できません。
/// </para>
/// </remarks>
public readonly struct MatrixNode<TElement> where TElement : unmanaged, IFloatingPointIeee754<TElement>
{
    internal readonly Tape? Tape;
    internal readonly int Generation;
    internal readonly ArrayView View;

    internal MatrixNode(Tape tape, ArrayView view)
    {
        Tape = tape;
        Generation = tape.Generation;
        View = view;
    }

    /// <summary>
    /// Gets the number of rows in the matrix.
    /// <para>行列の行数を取得します。</para>
    /// </summary>
    public int Rows => View.Rows;

    /// <summary>
    /// Gets the number of columns in the matrix.
    /// <para>行列の列数を取得します。</para>
    /// </summary>
    public int Columns => View.Columns;

    /// <summary>
    /// Gets a read-only view of the node's values.
    /// <para>ノードの値の読み取り専用ビューを取得します。</para>
    /// </summary>
    /// <value>A read-only view borrowing tape storage; must not be accessed after <see cref="Tape.Reset"/>. / テープの領域を借用する読み取り専用ビュー。<see cref="Tape.Reset"/> 後は参照できません。</value>
    /// <exception cref="InvalidOperationException">Thrown if the node belongs to an earlier generation or is uninitialized. / 世代が古い、またはテープから作成されていない未初期化ノードの場合。</exception>
    public ReadOnlyMatrixSpan<TElement> Value => Owner().Store<TElement>().MatrixView(View);

    /// <summary>
    /// Gets a value indicating whether this node depends on any differentiable leaf or <see cref="Parameter{T}"/>.
    /// <para>このノードが微分対象の葉または <see cref="Parameter{T}"/> に依存しているかどうかを示す値を取得します。</para>
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if the node belongs to an earlier generation or is uninitialized. / 世代が古い、またはテープから作成されていない未初期化ノードの場合。</exception>
    public bool RequiresGradient => Owner().Store<TElement>().RequiresGradient(View.Slot);

    /// <summary>
    /// Records a user operation on the tape shared by its matrix inputs.
    /// <para>行列入力が共有するテープに利用者定義演算を記録します。</para>
    /// </summary>
    /// <typeparam name="TOperation">The differentiable operation type. / 微分可能な演算の型。</typeparam>
    /// <param name="operation">The user operation. / 利用者定義演算。</param>
    /// <param name="inputs">One or more input nodes on the same tape. / 同じテープに属する 1 個以上の入力ノード。</param>
    /// <param name="rows">The output row count. / 出力の行数。</param>
    /// <param name="columns">The output column count. / 出力の列数。</param>
    /// <returns>The output matrix node. / 出力行列ノード。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="operation"/> is <see langword="null"/>. / <paramref name="operation"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException">The input span is empty. / 入力スパンが空の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">A dimension is negative. / 行数または列数が負の場合。</exception>
    /// <exception cref="InvalidOperationException">An input is uninitialized or stale, the inputs belong to different tapes, or the tape cannot record the operation. / 入力が未初期化、古い世代、別のテープに属する、またはテープが演算を記録できない場合。</exception>
    public static MatrixNode<TElement> Record<TOperation>(TOperation operation, ReadOnlySpan<MatrixNode<TElement>> inputs, int rows, int columns)
        where TOperation : IDifferentiableOperation<TElement>
    {
        if (inputs.IsEmpty)
        {
            throw new ArgumentException("At least one input is required to identify the tape.", nameof(inputs));
        }

        return inputs[0].Owner().Apply(operation, inputs, rows, columns);
    }

    /// <summary>
    /// Gets a zero-copy transposed view <c>Aᵀ</c>.
    /// <para>転置 <c>Aᵀ</c> のビューをコピーなしで取得します。</para>
    /// </summary>
    /// <value>A transposed view sharing storage with the source node; gradients accumulate into the corresponding elements of the source node. / 元のノードと領域を共有する転置ビュー。勾配は元のノードの対応する要素に加算されます。</value>
    /// <exception cref="InvalidOperationException">Thrown if the node belongs to an earlier generation or is uninitialized. / 世代が古い、またはテープから作成されていない未初期化ノードの場合。</exception>
    public MatrixNode<TElement> T => new(Owner(), View.Transpose());

    /// <summary>
    /// Repeats the matrix along the third axis without copying its values.
    /// <para>行列の値をコピーせず、第 3 軸に沿って繰り返します。</para>
    /// </summary>
    /// <param name="count">The number of matrices in the result. / 結果の行列数。</param>
    /// <returns>A tensor view whose gradient contributions are summed along the third axis. / 勾配の寄与を第 3 軸に沿って合計するテンソルビュー。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative. / <paramref name="count"/> が負の場合。</exception>
    /// <exception cref="InvalidOperationException">The node is uninitialized or stale. / ノードが未初期化または古い世代の場合。</exception>
    public TensorNode<TElement> Broadcast(int count)
    {
        Owner();
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        return new(this, TensorView.Whole(Rows, Columns, count, repeated: true));
    }

    /// <summary>
    /// Gets a zero-copy submatrix view over the specified row and column ranges.
    /// <para>指定した行範囲と列範囲の部分行列ビューをコピーなしで取得します。</para>
    /// </summary>
    /// <param name="rows">The row range to slice. / 切り出す行範囲。</param>
    /// <param name="columns">The column range to slice. / 切り出す列範囲。</param>
    /// <value>A submatrix view sharing storage with the source node; gradients accumulate into the corresponding elements of the source node. / 元のノードと領域を共有する部分行列ビュー。勾配は元のノードの対応する要素に加算されます。</value>
    /// <exception cref="InvalidOperationException">Thrown if the node belongs to an earlier generation or is uninitialized. / 世代が古い、またはテープから作成されていない未初期化ノードの場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rows"/> or <paramref name="columns"/> is out of bounds. / 行範囲または列範囲が次元を超える場合。</exception>
    public MatrixNode<TElement> this[Range rows, Range columns]
    {
        get
        {
            Tape tape = Owner();
            return new(tape, View.Slice(rows, columns));
        }
    }

    /// <summary>
    /// Computes the matrix product <c>AB</c> (equivalent to <c>XMath.Multiply(a, b)</c>).
    /// <para>行列積 <c>AB</c> を計算します（<c>XMath.Multiply(a, b)</c> と同等です）。</para>
    /// </summary>
    /// <param name="a">The left matrix operand. / 左辺の行列。</param>
    /// <param name="b">The right matrix operand. / 右辺の行列。</param>
    /// <returns>The matrix product <c>AB</c>. / 行列積 <c>AB</c>。</returns>
    /// <exception cref="ArgumentException">The column count of <paramref name="a"/> does not match the row count of <paramref name="b"/>. / <paramref name="a"/> の列数と <paramref name="b"/> の行数が一致しない場合。</exception>
    /// <exception cref="InvalidOperationException">Thrown if any node is stale, belongs to a different tape, or backpropagation has already completed. / 世代が古い、テープが異なる、または逆伝播完了後のテープへ記録しようとした場合。</exception>
    public static MatrixNode<TElement> operator *(MatrixNode<TElement> a, MatrixNode<TElement> b) => XMathExtensions.MultiplyCore(a, b);

    /// <summary>
    /// Computes the elementwise sum <c>x + y</c>.
    /// <para>要素ごとの和 <c>x + y</c> を計算します。</para>
    /// </summary>
    /// <param name="x">The left operand. / 左辺。</param>
    /// <param name="y">The right operand. / 右辺。</param>
    /// <returns>The elementwise sum. / 要素ごとの和。</returns>
    /// <exception cref="ArgumentException">The shapes of <paramref name="x"/> and <paramref name="y"/> do not match. / 行列の形状が一致しない場合。</exception>
    /// <exception cref="InvalidOperationException">Thrown if any node is stale, belongs to a different tape, or backpropagation has already completed. / 世代が古い、テープが異なる、または逆伝播完了後のテープへ記録しようとした場合。</exception>
    public static MatrixNode<TElement> operator +(MatrixNode<TElement> x, MatrixNode<TElement> y) => XMathExtensions.BinaryCore(x, y, ArrayOperation.Add);

    /// <summary>
    /// Computes the elementwise difference <c>x - y</c>.
    /// <para>要素ごとの差 <c>x - y</c> を計算します。</para>
    /// </summary>
    /// <param name="x">The left operand. / 左辺。</param>
    /// <param name="y">The right operand. / 右辺。</param>
    /// <returns>The elementwise difference. / 要素ごとの差。</returns>
    /// <exception cref="ArgumentException">The shapes of <paramref name="x"/> and <paramref name="y"/> do not match. / 行列の形状が一致しない場合。</exception>
    /// <exception cref="InvalidOperationException">Thrown if any node is stale, belongs to a different tape, or backpropagation has already completed. / 世代が古い、テープが異なる、または逆伝播完了後のテープへ記録しようとした場合。</exception>
    public static MatrixNode<TElement> operator -(MatrixNode<TElement> x, MatrixNode<TElement> y) => XMathExtensions.BinaryCore(x, y, ArrayOperation.Subtract);

    /// <summary>
    /// Negates each element of <paramref name="x"/>.
    /// <para><paramref name="x"/> の各要素の符号を反転します。</para>
    /// </summary>
    /// <param name="x">The input matrix node. / 入力行列ノード。</param>
    /// <returns>The negated matrix <c>-x</c>. / 符号反転した行列 <c>-x</c>。</returns>
    /// <exception cref="InvalidOperationException">Thrown if the node is stale or backpropagation has already completed. / 世代が古い、または逆伝播完了後のテープへ記録しようとした場合。</exception>
    public static MatrixNode<TElement> operator -(MatrixNode<TElement> x) => XMathExtensions.UnaryCore(x, ArrayOperation.Negate, approximate: false);

    /// <summary>
    /// Adds a scalar to each element of a matrix (<c>x + s</c>).
    /// <para>行列の各要素にスカラーを加算します（<c>x + s</c>）。</para>
    /// </summary>
    /// <param name="x">The matrix operand. / 行列の被演算子。</param>
    /// <param name="s">The scalar operand. / スカラーの被演算子。</param>
    /// <returns>The shifted matrix <c>x + s</c>. / 各要素に <paramref name="s"/> を加算した行列。</returns>
    /// <exception cref="InvalidOperationException">Thrown if any node is stale, belongs to a different tape, or backpropagation has already completed. / 世代が古い、テープが異なる、または逆伝播完了後のテープへ記録しようとした場合。</exception>
    public static MatrixNode<TElement> operator +(MatrixNode<TElement> x, ScalarNode s) => XMathExtensions.ScalarCore(x, s, ArrayOperation.AddScalar);

    /// <summary>
    /// Adds a scalar to each element of a matrix (<c>s + x</c>).
    /// <para>スカラーを行列の各要素に加算します（<c>s + x</c>）。</para>
    /// </summary>
    /// <param name="s">The scalar operand. / スカラーの被演算子。</param>
    /// <param name="x">The matrix operand. / 行列の被演算子。</param>
    /// <returns>The shifted matrix <c>s + x</c>. / 各要素に <paramref name="s"/> を加算した行列。</returns>
    /// <exception cref="InvalidOperationException">Thrown if any node is stale, belongs to a different tape, or backpropagation has already completed. / 世代が古い、テープが異なる、または逆伝播完了後のテープへ記録しようとした場合。</exception>
    public static MatrixNode<TElement> operator +(ScalarNode s, MatrixNode<TElement> x) => XMathExtensions.ScalarCore(x, s, ArrayOperation.AddScalar);

    /// <summary>
    /// Subtracts a scalar from each element of a matrix (<c>x - s</c>).
    /// <para>行列の各要素からスカラーを減算します（<c>x - s</c>）。</para>
    /// </summary>
    /// <param name="x">The matrix operand. / 行列の被演算子。</param>
    /// <param name="s">The scalar operand. / スカラーの被演算子。</param>
    /// <returns>The shifted matrix <c>x - s</c>. / 各要素から <paramref name="s"/> を減算した行列。</returns>
    /// <exception cref="InvalidOperationException">Thrown if any node is stale, belongs to a different tape, or backpropagation has already completed. / 世代が古い、テープが異なる、または逆伝播完了後のテープへ記録しようとした場合。</exception>
    public static MatrixNode<TElement> operator -(MatrixNode<TElement> x, ScalarNode s) => XMathExtensions.ScalarCore(x, s, ArrayOperation.SubtractScalar);

    /// <summary>
    /// Subtracts each element of a matrix from a scalar (<c>s - x</c>).
    /// <para>スカラーから行列の各要素を減算します（<c>s - x</c>）。</para>
    /// </summary>
    /// <param name="s">The scalar minuend. / 被減数のスカラー。</param>
    /// <param name="x">The matrix subtrahend. / 減数の行列。</param>
    /// <returns>The matrix <c>s - x</c>. / <c>s - x</c> の結果行列。</returns>
    /// <remarks>
    /// <para>
    /// Evaluated as <c>XMath.Negate(x)</c> followed by <c>XMath.Add(..., s)</c> (rounding <paramref name="s"/> to <see langword="float"/> for <see langword="float"/> arrays), which matches IEEE 754 <c>s - x</c> for non-NaN values.
    /// </para>
    /// <para>
    /// 値は <c>XMath.Negate(x)</c> に続けて <c>XMath.Add(..., s)</c> を適用した結果とビット単位で一致します（<see langword="float"/> 配列では <paramref name="s"/> を <see langword="float"/> に丸めます）。NaN 以外では IEEE 754 の <c>s - x</c> と同一です。
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown if any node is stale, belongs to a different tape, or backpropagation has already completed. / 世代が古い、テープが異なる、または逆伝播完了後のテープへ記録しようとした場合。</exception>
    public static MatrixNode<TElement> operator -(ScalarNode s, MatrixNode<TElement> x) => XMathExtensions.ScalarCore(x, s, ArrayOperation.ScalarSubtract);

    /// <summary>
    /// Multiplies each element of a matrix by a scalar (<c>x * s</c>).
    /// <para>行列の各要素にスカラーを掛けます（<c>x * s</c>）。</para>
    /// </summary>
    /// <param name="x">The matrix operand. / 行列の被演算子。</param>
    /// <param name="s">The scalar operand. / スカラーの被演算子。</param>
    /// <returns>The scaled matrix <c>x * s</c>. / スカラー倍した行列 <c>x * s</c>。</returns>
    /// <exception cref="InvalidOperationException">Thrown if any node is stale, belongs to a different tape, or backpropagation has already completed. / 世代が古い、テープが異なる、または逆伝播完了後のテープへ記録しようとした場合。</exception>
    public static MatrixNode<TElement> operator *(MatrixNode<TElement> x, ScalarNode s) => XMathExtensions.ScalarCore(x, s, ArrayOperation.MultiplyScalar);

    /// <summary>
    /// Multiplies a scalar by each element of a matrix (<c>s * x</c>).
    /// <para>スカラーを行列の各要素に掛けます（<c>s * x</c>）。</para>
    /// </summary>
    /// <param name="s">The scalar operand. / スカラーの被演算子。</param>
    /// <param name="x">The matrix operand. / 行列の被演算子。</param>
    /// <returns>The scaled matrix <c>s * x</c>. / スカラー倍した行列 <c>s * x</c>。</returns>
    /// <exception cref="InvalidOperationException">Thrown if any node is stale, belongs to a different tape, or backpropagation has already completed. / 世代が古い、テープが異なる、または逆伝播完了後のテープへ記録しようとした場合。</exception>
    public static MatrixNode<TElement> operator *(ScalarNode s, MatrixNode<TElement> x) => XMathExtensions.ScalarCore(x, s, ArrayOperation.MultiplyScalar);

    /// <summary>
    /// Divides each element of a matrix by a scalar (<c>x / s</c>).
    /// <para>行列の各要素をスカラーで割ります（<c>x / s</c>）。</para>
    /// </summary>
    /// <param name="x">The matrix dividend. / 被除数の行列。</param>
    /// <param name="s">The scalar divisor. / 除数のスカラー。</param>
    /// <returns>The scaled matrix <c>x / s</c>. / スカラーで割った行列 <c>x / s</c>。</returns>
    /// <exception cref="InvalidOperationException">Thrown if any node is stale, belongs to a different tape, or backpropagation has already completed. / 世代が古い、テープが異なる、または逆伝播完了後のテープへ記録しようとした場合。</exception>
    public static MatrixNode<TElement> operator /(MatrixNode<TElement> x, ScalarNode s) => XMathExtensions.ScalarCore(x, s, ArrayOperation.DivideScalar);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal Tape Owner()
    {
        Tape? tape = Tape;
        if (tape is null)
        {
            VectorNode<TElement>.ThrowUninitialized();
        }

        tape.ValidateGeneration(Generation);
        return tape;
    }
}
