using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

/// <summary>
/// Represents a vector node for reverse-mode automatic differentiation.
/// <para>逆モード自動微分のベクトルノードを表します。</para>
/// </summary>
/// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
/// <remarks>
/// <para>
/// Created via <see cref="Tape.Leaf{T}(ReadOnlyVectorSpan{T})"/>, <see cref="Tape.Constant{T}(LinearAlgebra.Vector{T})"/>, <see cref="Tape.UseVector{T}(Parameter{T})"/>, or differentiable operations.
/// Supports <see langword="float"/> and <see langword="double"/>. Nodes created before <see cref="Tape.Reset"/> become invalid and cannot be used.
/// </para>
/// <para>
/// <see cref="Tape.Leaf{T}(ReadOnlyVectorSpan{T})"/>、<see cref="Tape.Constant{T}(LinearAlgebra.Vector{T})"/>、<see cref="Tape.UseVector{T}(Parameter{T})"/>、および各種演算から作成します。
/// 要素型は <see langword="float"/> と <see langword="double"/> に対応します。<see cref="Tape.Reset"/> より前に作成したノードは無効になり使用できません。
/// </para>
/// </remarks>
public readonly struct VectorNode<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    internal readonly Tape? Tape;
    internal readonly int Generation;
    internal readonly ArrayView View;

    internal VectorNode(Tape tape, ArrayView view)
    {
        Tape = tape;
        Generation = tape.Generation;
        View = view;
    }

    /// <summary>
    /// Gets the number of elements in the vector.
    /// <para>ベクトルの要素数を取得します。</para>
    /// </summary>
    public int Length => View.RowCount;

    /// <summary>
    /// Gets a read-only view of the node's values.
    /// <para>ノードの値の読み取り専用ビューを取得します。</para>
    /// </summary>
    /// <value>A read-only view borrowing tape storage; must not be accessed after <see cref="Tape.Reset"/>. / テープの領域を借用する読み取り専用ビュー。<see cref="Tape.Reset"/> 後は参照できません。</value>
    /// <exception cref="InvalidOperationException">Thrown if the node belongs to an earlier generation or is uninitialized. / 世代が古い、またはテープから作成されていない未初期化ノードの場合。</exception>
    public ReadOnlyVectorSpan<T> Value => Owner().Store<T>().VectorView(View);

    /// <summary>
    /// Gets a value indicating whether this node depends on any differentiable leaf or <see cref="Parameter{T}"/>.
    /// <para>このノードが微分対象の葉または <see cref="Parameter{T}"/> に依存しているかどうかを示す値を取得します。</para>
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if the node belongs to an earlier generation or is uninitialized. / 世代が古い、またはテープから作成されていない未初期化ノードの場合。</exception>
    public bool RequiresGradient => Owner().Store<T>().RequiresGradient(View.Slot);

    /// <summary>
    /// Records a user operation on the tape shared by its vector inputs.
    /// <para>ベクトル入力が共有するテープに利用者定義演算を記録します。</para>
    /// </summary>
    /// <typeparam name="TOperation">The differentiable operation type. / 微分可能な演算の型。</typeparam>
    /// <param name="operation">The user operation. / 利用者定義演算。</param>
    /// <param name="inputs">One or more input nodes on the same tape. / 同じテープに属する 1 個以上の入力ノード。</param>
    /// <param name="length">The output vector length. / 出力ベクトルの長さ。</param>
    /// <returns>The output vector node. / 出力ベクトルノード。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="operation"/> is <see langword="null"/>. / <paramref name="operation"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException">The input span is empty. / 入力スパンが空の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">The length is negative. / 長さが負の場合。</exception>
    /// <exception cref="InvalidOperationException">An input is uninitialized or stale, the inputs belong to different tapes, or the tape cannot record the operation. / 入力が未初期化、古い世代、別のテープに属する、またはテープが演算を記録できない場合。</exception>
    public static VectorNode<T> Record<TOperation>(TOperation operation, ReadOnlySpan<VectorNode<T>> inputs, int length)
        where TOperation : IDifferentiableOperation<T>
    {
        if (inputs.IsEmpty)
        {
            throw new ArgumentException("At least one input is required to identify the tape.", nameof(inputs));
        }

        return inputs[0].Owner().Apply(operation, inputs, length);
    }

    /// <summary>
    /// Gets a single-column matrix node sharing this vector's values and gradients.
    /// <para>このベクトルの値と勾配を共有する 1 列の行列ノードを取得します。</para>
    /// </summary>
    /// <exception cref="InvalidOperationException">The node is uninitialized or belongs to an earlier generation. / ノードが未初期化、または古い世代に属する場合。</exception>
    public MatrixNode<T> AsColumnMatrix() => new(Owner(), View);

    /// <summary>
    /// Gets a zero-copy subvector view over the specified range.
    /// <para>指定した範囲の部分ベクトルビューをコピーなしで取得します。</para>
    /// </summary>
    /// <param name="range">The element range to slice. / 切り出す要素範囲。</param>
    /// <value>A view sharing storage with the source node; gradients accumulate into the corresponding elements of the source node. / 元のノードと領域を共有するビュー。勾配は元のノードの対応する要素に加算されます。</value>
    /// <exception cref="InvalidOperationException">Thrown if the node belongs to an earlier generation or is uninitialized. / 世代が古い、またはテープから作成されていない未初期化ノードの場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="range"/> is out of bounds for this vector. / 範囲が要素数を超える場合。</exception>
    public VectorNode<T> this[Range range]
    {
        get
        {
            Tape tape = Owner();
            return new(tape, View.Slice(range));
        }
    }

    /// <summary>
    /// Selects a vector element as a scalar node.
    /// <para>ベクトルの要素をスカラーノードとして取得します。</para>
    /// </summary>
    /// <param name="index">The zero-based element index. / 0 始まりの要素インデックス。</param>
    /// <returns>The selected scalar node. / 指定した要素のスカラーノード。</returns>
    /// <exception cref="ArgumentOutOfRangeException">The index is outside the vector. / インデックスが範囲外の場合。</exception>
    /// <exception cref="InvalidOperationException">The node is stale or its tape has run backward. / ノードが失効しているか、テープが逆伝播済みの場合。</exception>
    public ScalarNode this[int index] => XMathExtensions.ElementAt(this, index);

    /// <summary>
    /// Creates a zero-copy matrix view repeating this vector across <paramref name="count"/> columns.
    /// <para>このベクトルを <paramref name="count"/> 列に繰り返し配置した行列ビューをコピーなしで作成します。</para>
    /// </summary>
    /// <param name="count">The number of columns to repeat. / 繰り返す列数。</param>
    /// <returns>A <c>Length × count</c> matrix view referencing the source vector without copying. / 元のベクトルをコピーせずに参照する <c>Length × count</c> の行列ビュー。</returns>
    /// <remarks>
    /// <para>
    /// Backpropagation sums contributions across the repeated columns into the source vector.
    /// <see cref="Tape.GetGradient{T}(MatrixNode{T})"/> returns the source vector's gradient repeated to the same shape.
    /// </para>
    /// <para>
    /// 逆伝播では繰り返した列方向の寄与を合算して元のベクトルに加算します。
    /// <see cref="Tape.GetGradient{T}(MatrixNode{T})"/> は元のベクトルの勾配を同じ形状に繰り返したビューを返します。
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown if the node belongs to an earlier generation or is uninitialized. / 世代が古い、またはテープから作成されていない未初期化ノードの場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative, or <c>Length * count</c> exceeds <see cref="int.MaxValue"/>. / <paramref name="count"/> が負、または総要素数 <c>Length * count</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    public MatrixNode<T> Broadcast(int count)
    {
        Tape tape = Owner();
        RequireBroadcastCount(count);
        return new(tape, View.Repeat(count, asRows: false));
    }

    /// <summary>
    /// Creates a zero-copy matrix view repeating this vector across <paramref name="rows"/> rows.
    /// <para>このベクトルを <paramref name="rows"/> 行に繰り返し配置した行列ビューをコピーなしで作成します。</para>
    /// </summary>
    /// <param name="rows">The number of rows to repeat. / 繰り返す行数。</param>
    /// <returns>A <c>rows × Length</c> matrix view referencing the source vector without copying. / 元のベクトルをコピーせずに参照する <c>rows × Length</c> の行列ビュー。</returns>
    /// <remarks>
    /// <para>
    /// Backpropagation sums contributions across the repeated rows into the source vector.
    /// <see cref="Tape.GetGradient{T}(MatrixNode{T})"/> returns the source vector's gradient repeated to the same shape.
    /// </para>
    /// <para>
    /// 逆伝播では繰り返した行方向の寄与を合算して元のベクトルに加算します。
    /// <see cref="Tape.GetGradient{T}(MatrixNode{T})"/> は元のベクトルの勾配を同じ形状に繰り返したビューを返します。
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown if the node belongs to an earlier generation or is uninitialized. / 世代が古い、またはテープから作成されていない未初期化ノードの場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rows"/> is negative, or <c>Length * rows</c> exceeds <see cref="int.MaxValue"/>. / <paramref name="rows"/> が負、または総要素数 <c>Length * rows</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    public MatrixNode<T> BroadcastRows(int rows)
    {
        Tape tape = Owner();
        RequireBroadcastCount(rows);
        return new(tape, View.Repeat(rows, asRows: true));
    }

    // Keep logical element count within int.MaxValue so backward scratch buffers can be int-indexed.
    private void RequireBroadcastCount(int count, [CallerArgumentExpression(nameof(count))] string? name = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count, name);
        if (count > 0 && Length > int.MaxValue / count)
        {
            throw new ArgumentOutOfRangeException(name, count, "The number of elements of the broadcast view must not exceed int.MaxValue.");
        }
    }

    /// <summary>
    /// Computes the matrix-vector product <c>Ax</c> (equivalent to <c>XMath.Multiply(a, x)</c>).
    /// <para>行列ベクトル積 <c>Ax</c> を計算します（<c>XMath.Multiply(a, x)</c> と同等です）。</para>
    /// </summary>
    /// <param name="a">The matrix operand. / 行列の被演算子。</param>
    /// <param name="x">The vector operand. / ベクトルの被演算子。</param>
    /// <returns>The product vector <c>Ax</c>. / 行列ベクトル積 <c>Ax</c>。</returns>
    /// <exception cref="ArgumentException">The column count of <paramref name="a"/> does not match the length of <paramref name="x"/>. / <paramref name="a"/> の列数と <paramref name="x"/> の要素数が一致しない場合。</exception>
    /// <exception cref="InvalidOperationException">Thrown if any node is stale, belongs to a different tape, or backpropagation has already completed. / 世代が古い、テープが異なる、または逆伝播完了後のテープへ記録しようとした場合。</exception>
    public static VectorNode<T> operator *(MatrixNode<T> a, VectorNode<T> x) => XMathExtensions.MultiplyCore(a, x);

    /// <summary>
    /// Computes the elementwise sum <c>x + y</c>.
    /// <para>要素ごとの和 <c>x + y</c> を計算します。</para>
    /// </summary>
    /// <param name="x">The left operand. / 左辺。</param>
    /// <param name="y">The right operand. / 右辺。</param>
    /// <returns>The elementwise sum. / 要素ごとの和。</returns>
    /// <exception cref="ArgumentException">The lengths of <paramref name="x"/> and <paramref name="y"/> do not match. / 要素数が一致しない場合。</exception>
    /// <exception cref="InvalidOperationException">Thrown if any node is stale, belongs to a different tape, or backpropagation has already completed. / 世代が古い、テープが異なる、または逆伝播完了後のテープへ記録しようとした場合。</exception>
    public static VectorNode<T> operator +(VectorNode<T> x, VectorNode<T> y) => XMathExtensions.BinaryCore(x, y, ArrayOperation.Add);

    /// <summary>
    /// Computes the elementwise difference <c>x - y</c>.
    /// <para>要素ごとの差 <c>x - y</c> を計算します。</para>
    /// </summary>
    /// <param name="x">The left operand. / 左辺。</param>
    /// <param name="y">The right operand. / 右辺。</param>
    /// <returns>The elementwise difference. / 要素ごとの差。</returns>
    /// <exception cref="ArgumentException">The lengths of <paramref name="x"/> and <paramref name="y"/> do not match. / 要素数が一致しない場合。</exception>
    /// <exception cref="InvalidOperationException">Thrown if any node is stale, belongs to a different tape, or backpropagation has already completed. / 世代が古い、テープが異なる、または逆伝播完了後のテープへ記録しようとした場合。</exception>
    public static VectorNode<T> operator -(VectorNode<T> x, VectorNode<T> y) => XMathExtensions.BinaryCore(x, y, ArrayOperation.Subtract);

    /// <summary>
    /// Negates each element of <paramref name="x"/>.
    /// <para><paramref name="x"/> の各要素の符号を反転します。</para>
    /// </summary>
    /// <param name="x">The input vector node. / 入力ベクトルノード。</param>
    /// <returns>The negated vector <c>-x</c>. / 符号反転したベクトル <c>-x</c>。</returns>
    /// <exception cref="InvalidOperationException">Thrown if the node is stale or backpropagation has already completed. / 世代が古い、または逆伝播完了後のテープへ記録しようとした場合。</exception>
    public static VectorNode<T> operator -(VectorNode<T> x) => XMathExtensions.UnaryCore(x, ArrayOperation.Negate, approximate: false);

    /// <summary>
    /// Adds a scalar to each element of a vector (<c>x + s</c>).
    /// <para>ベクトルの各要素にスカラーを加算します（<c>x + s</c>）。</para>
    /// </summary>
    /// <param name="x">The vector operand. / ベクトルの被演算子。</param>
    /// <param name="s">The scalar operand. / スカラーの被演算子。</param>
    /// <returns>The shifted vector <c>x + s</c>. / 各要素に <paramref name="s"/> を加算したベクトル。</returns>
    /// <exception cref="InvalidOperationException">Thrown if any node is stale, belongs to a different tape, or backpropagation has already completed. / 世代が古い、テープが異なる、または逆伝播完了後のテープへ記録しようとした場合。</exception>
    public static VectorNode<T> operator +(VectorNode<T> x, ScalarNode s) => XMathExtensions.ScalarCore(x, s, ArrayOperation.AddScalar);

    /// <summary>
    /// Adds a scalar to each element of a vector (<c>s + x</c>).
    /// <para>スカラーをベクトルの各要素に加算します（<c>s + x</c>）。</para>
    /// </summary>
    /// <param name="s">The scalar operand. / スカラーの被演算子。</param>
    /// <param name="x">The vector operand. / ベクトルの被演算子。</param>
    /// <returns>The shifted vector <c>s + x</c>. / 各要素に <paramref name="s"/> を加算したベクトル。</returns>
    /// <exception cref="InvalidOperationException">Thrown if any node is stale, belongs to a different tape, or backpropagation has already completed. / 世代が古い、テープが異なる、または逆伝播完了後のテープへ記録しようとした場合。</exception>
    public static VectorNode<T> operator +(ScalarNode s, VectorNode<T> x) => XMathExtensions.ScalarCore(x, s, ArrayOperation.AddScalar);

    /// <summary>
    /// Subtracts a scalar from each element of a vector (<c>x - s</c>).
    /// <para>ベクトルの各要素からスカラーを減算します（<c>x - s</c>）。</para>
    /// </summary>
    /// <param name="x">The vector operand. / ベクトルの被演算子。</param>
    /// <param name="s">The scalar operand. / スカラーの被演算子。</param>
    /// <returns>The shifted vector <c>x - s</c>. / 各要素から <paramref name="s"/> を減算したベクトル。</returns>
    /// <exception cref="InvalidOperationException">Thrown if any node is stale, belongs to a different tape, or backpropagation has already completed. / 世代が古い、テープが異なる、または逆伝播完了後のテープへ記録しようとした場合。</exception>
    public static VectorNode<T> operator -(VectorNode<T> x, ScalarNode s) => XMathExtensions.ScalarCore(x, s, ArrayOperation.SubtractScalar);

    /// <summary>
    /// Subtracts each element of a vector from a scalar (<c>s - x</c>).
    /// <para>スカラーからベクトルの各要素を減算します（<c>s - x</c>）。</para>
    /// </summary>
    /// <param name="s">The scalar minuend. / 被減数のスカラー。</param>
    /// <param name="x">The vector subtrahend. / 減数のベクトル。</param>
    /// <returns>The vector <c>s - x</c>. / <c>s - x</c> の結果ベクトル。</returns>
    /// <remarks>
    /// <para>
    /// Evaluated as <c>XMath.Negate(x)</c> followed by <c>XMath.Add(..., s)</c> (rounding <paramref name="s"/> to <see langword="float"/> for <see langword="float"/> arrays), which matches IEEE 754 <c>s - x</c> for non-NaN values.
    /// </para>
    /// <para>
    /// 値は <c>XMath.Negate(x)</c> に続けて <c>XMath.Add(..., s)</c> を適用した結果とビット単位で一致します（<see langword="float"/> 配列では <paramref name="s"/> を <see langword="float"/> に丸めます）。NaN 以外では IEEE 754 の <c>s - x</c> と同一です。
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown if any node is stale, belongs to a different tape, or backpropagation has already completed. / 世代が古い、テープが異なる、または逆伝播完了後のテープへ記録しようとした場合。</exception>
    public static VectorNode<T> operator -(ScalarNode s, VectorNode<T> x) => XMathExtensions.ScalarCore(x, s, ArrayOperation.ScalarSubtract);

    /// <summary>
    /// Multiplies each element of a vector by a scalar (<c>x * s</c>).
    /// <para>ベクトルの各要素にスカラーを掛けます（<c>x * s</c>）。</para>
    /// </summary>
    /// <param name="x">The vector operand. / ベクトルの被演算子。</param>
    /// <param name="s">The scalar operand. / スカラーの被演算子。</param>
    /// <returns>The scaled vector <c>x * s</c>. / スカラー倍したベクトル <c>x * s</c>。</returns>
    /// <exception cref="InvalidOperationException">Thrown if any node is stale, belongs to a different tape, or backpropagation has already completed. / 世代が古い、テープが異なる、または逆伝播完了後のテープへ記録しようとした場合。</exception>
    public static VectorNode<T> operator *(VectorNode<T> x, ScalarNode s) => XMathExtensions.ScalarCore(x, s, ArrayOperation.MultiplyScalar);

    /// <summary>
    /// Multiplies a scalar by each element of a vector (<c>s * x</c>).
    /// <para>スカラーをベクトルの各要素に掛けます（<c>s * x</c>）。</para>
    /// </summary>
    /// <param name="s">The scalar operand. / スカラーの被演算子。</param>
    /// <param name="x">The vector operand. / ベクトルの被演算子。</param>
    /// <returns>The scaled vector <c>s * x</c>. / スカラー倍したベクトル <c>s * x</c>。</returns>
    /// <exception cref="InvalidOperationException">Thrown if any node is stale, belongs to a different tape, or backpropagation has already completed. / 世代が古い、テープが異なる、または逆伝播完了後のテープへ記録しようとした場合。</exception>
    public static VectorNode<T> operator *(ScalarNode s, VectorNode<T> x) => XMathExtensions.ScalarCore(x, s, ArrayOperation.MultiplyScalar);

    /// <summary>
    /// Divides each element of a vector by a scalar (<c>x / s</c>).
    /// <para>ベクトルの各要素をスカラーで割ります（<c>x / s</c>）。</para>
    /// </summary>
    /// <param name="x">The vector dividend. / 被除数のベクトル。</param>
    /// <param name="s">The scalar divisor. / 除数のスカラー。</param>
    /// <returns>The scaled vector <c>x / s</c>. / スカラーで割ったベクトル <c>x / s</c>。</returns>
    /// <exception cref="InvalidOperationException">Thrown if any node is stale, belongs to a different tape, or backpropagation has already completed. / 世代が古い、テープが異なる、または逆伝播完了後のテープへ記録しようとした場合。</exception>
    public static VectorNode<T> operator /(VectorNode<T> x, ScalarNode s) => XMathExtensions.ScalarCore(x, s, ArrayOperation.DivideScalar);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal Tape Owner()
    {
        Tape? tape = Tape;
        if (tape is null)
        {
            ThrowUninitialized();
        }

        tape.ValidateGeneration(Generation);
        return tape;
    }

    [DoesNotReturn]
    [MethodImpl(MethodImplOptions.NoInlining)]
    internal static void ThrowUninitialized() => throw new InvalidOperationException("The array node is not initialized. Create it from a Tape.");
}
