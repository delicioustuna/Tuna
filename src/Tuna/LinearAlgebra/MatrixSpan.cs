using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Tuna.LinearAlgebra;

/// <summary>
/// Represents a zero-copy, strided read-only view over a two-dimensional matrix.
/// <para>ストライド付きの 2 次元行列を参照する、コピーなしの読み取り専用ビューを表します。</para>
/// </summary>
/// <typeparam name="TElement">The element type of the matrix. / 行列の要素型。</typeparam>
/// <remarks>
/// <para>Element <c>(i, j)</c> is located at offset <c>i * Strides[0] + j * Strides[1]</c>. Transposition, slicing, and diagonal extraction adjust strides without copying data.</para>
/// <para><see cref="Data"/>, <see cref="Lengths"/>, and <see cref="Strides"/> can be passed directly to the <c>System.Numerics.Tensors.ReadOnlyTensorSpan&lt;T&gt;</c> constructor; use <see cref="Create"/> for the reverse conversion.</para>
/// <para>要素 <c>(i, j)</c> は先頭参照から <c>i * Strides[0] + j * Strides[1]</c> 要素目の位置にあります。転置・スライス・対角取得はいずれもストライド調整のみでコピーを行いません。</para>
/// <para><see cref="Data"/>、<see cref="Lengths"/>、<see cref="Strides"/> は <c>System.Numerics.Tensors.ReadOnlyTensorSpan&lt;T&gt;</c> のコンストラクターにそのまま渡せます。逆方向の変換には <see cref="Create"/> を使用してください。</para>
/// </remarks>
public readonly ref struct ReadOnlyMatrixSpan<TElement> where TElement : unmanaged, INumberBase<TElement>
{
    private readonly ref readonly TElement reference;
    private readonly NintPair lengths;
    private readonly NintPair strides;

    internal ReadOnlyMatrixSpan(ref readonly TElement reference, int rows, int columns, int rowStride, int columnStride)
    {
        this.reference = ref reference;
        lengths = NintPair.Create(rows, columns);
        strides = NintPair.Create(StridedLayout.Normalize(rows, rowStride), StridedLayout.Normalize(columns, columnStride));
    }

    /// <summary>
    /// Initializes a read-only view over a contiguous column-major span.
    /// <para>列優先の連続領域を参照する読み取り専用ビューを初期化します。</para>
    /// </summary>
    /// <param name="data">The contiguous column-major span of length <c>rows * columns</c>. / 長さ <c>rows * columns</c> の列優先連続領域。</param>
    /// <param name="rows">The number of rows. / 行数。</param>
    /// <param name="columns">The number of columns. / 列数。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rows"/> or <paramref name="columns"/> is negative. / <paramref name="rows"/> または <paramref name="columns"/> が負の場合。</exception>
    /// <exception cref="OverflowException"><c>rows * columns</c> exceeds <see cref="int.MaxValue"/>. / 要素数 <c>rows * columns</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="data"/> does not have length <c>rows * columns</c>. / <paramref name="data"/> の長さが <c>rows * columns</c> と一致しない場合。</exception>
    public ReadOnlyMatrixSpan(ReadOnlySpan<TElement> data, int rows, int columns)
    {
        DenseStorage.RequireLength(DenseStorage.Length(rows, columns), data.Length);
        this = new(in MemoryMarshal.GetReference(data), rows, columns, 1, rows);
    }

    /// <summary>
    /// Initializes a read-only view over the specified span with explicit shape and strides.
    /// <para>形状とストライドを指定して領域を参照する読み取り専用ビューを初期化します。</para>
    /// </summary>
    /// <param name="data">The backing span starting at the first element and covering through the last element. / 先頭要素から最終要素までを含む格納領域。</param>
    /// <param name="lengths">A length-2 span specifying <c>[rows, columns]</c>. / <c>[行数, 列数]</c> を指定する長さ 2 の形状。</param>
    /// <param name="strides">A length-2 span specifying non-negative element strides (ignored for axes of length 0 or 1). / 非負の要素ストライドを指定する長さ 2 のスパン（長さが 0 または 1 の軸では無視されます）。</param>
    /// <exception cref="ArgumentException"><paramref name="lengths"/> or <paramref name="strides"/> does not have length 2, or <paramref name="data"/> is too short. / <paramref name="lengths"/> または <paramref name="strides"/> の長さが 2 でない、または <paramref name="data"/> の長さが不足している場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">A length or stride is negative or exceeds <see cref="int.MaxValue"/>. / 長さまたはストライドが負、または <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="OverflowException">The spanned extent exceeds <see cref="int.MaxValue"/>. / 参照範囲の要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public ReadOnlyMatrixSpan(ReadOnlySpan<TElement> data, ReadOnlySpan<nint> lengths, ReadOnlySpan<nint> strides)
    {
        var (rows, columns, rowStride, columnStride) = StridedLayout.Validate(lengths, strides, 2, writable: false);
        if (data.Length < StridedLayout.Extent(rows, columns, rowStride, columnStride))
        {
            throw new ArgumentException("The data is too short for the lengths and strides.", nameof(data));
        }

        this = new(in MemoryMarshal.GetReference(data), rows, columns, rowStride, columnStride);
    }

    /// <summary>
    /// Creates a read-only matrix view from a reference to the first element, shape, and strides.
    /// <para>先頭要素への参照・形状・ストライドから読み取り専用行列ビューを作成します。</para>
    /// </summary>
    /// <param name="reference">A reference to the first element. / 先頭要素への参照。</param>
    /// <param name="lengths">A length-2 span specifying <c>[rows, columns]</c>. / <c>[行数, 列数]</c> を指定する長さ 2 の形状。</param>
    /// <param name="strides">A length-2 span specifying non-negative element strides. / 非負の要素ストライドを指定する長さ 2 のスパン。</param>
    /// <returns>A read-only matrix view referencing the specified memory layout. / 指定したレイアウトを参照する読み取り専用行列ビュー。</returns>
    /// <remarks>
    /// <para>Compatible with <c>System.Numerics.Tensors</c> via <c>Create(in t.GetPinnableReference(), t.Lengths, t.Strides)</c>. Bounds of the backing memory are not checked, matching <see cref="MemoryMarshal.CreateReadOnlySpan"/>.</para>
    /// <para><c>System.Numerics.Tensors</c> からは <c>Create(in t.GetPinnableReference(), t.Lengths, t.Strides)</c> で変換できます。<see cref="MemoryMarshal.CreateReadOnlySpan"/> と同様に、メモリ範囲の境界検査は行いません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="lengths"/> or <paramref name="strides"/> does not have length 2. / <paramref name="lengths"/> または <paramref name="strides"/> の長さが 2 でない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">A length or stride is negative or exceeds <see cref="int.MaxValue"/>. / 長さまたはストライドが負、または <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="OverflowException">The spanned extent exceeds <see cref="int.MaxValue"/>. / 参照範囲の要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static ReadOnlyMatrixSpan<TElement> Create(ref readonly TElement reference, ReadOnlySpan<nint> lengths, ReadOnlySpan<nint> strides)
    {
        var (rows, columns, rowStride, columnStride) = StridedLayout.Validate(lengths, strides, 2, writable: false);
        return new(in reference, rows, columns, rowStride, columnStride);
    }

    /// <summary>
    /// Gets the number of rows in the view.
    /// <para>ビューの行数を取得します。</para>
    /// </summary>
    public int Rows => (int)lengths[0];

    /// <summary>
    /// Gets the number of columns in the view.
    /// <para>ビューの列数を取得します。</para>
    /// </summary>
    public int Columns => (int)lengths[1];

    /// <summary>
    /// Gets the number of dimensions, which is always 2.
    /// <para>次元数（常に 2）を取得します。</para>
    /// </summary>
    public int Rank => 2;

    /// <summary>
    /// Gets the total number of elements (<c>Rows * Columns</c>).
    /// <para>総要素数（<c>Rows * Columns</c>）を取得します。</para>
    /// </summary>
    public nint FlattenedLength => (nint)Rows * Columns;

    /// <summary>
    /// Gets a value indicating whether the view contains no elements.
    /// <para>ビューが空かどうかを示す値を取得します。</para>
    /// </summary>
    public bool IsEmpty => Rows == 0 || Columns == 0;

    /// <summary>
    /// Gets the shape <c>[Rows, Columns]</c> of the view.
    /// <para>ビューの形状 <c>[Rows, Columns]</c> を取得します。</para>
    /// </summary>
    [UnscopedRef] public ReadOnlySpan<nint> Lengths => lengths;

    /// <summary>
    /// Gets the element strides along each axis (0 for axes of length 0 or 1).
    /// <para>各軸の要素単位のストライドを取得します（長さが 0 または 1 の軸は 0）。</para>
    /// </summary>
    [UnscopedRef] public ReadOnlySpan<nint> Strides => strides;

    /// <summary>
    /// Gets a contiguous span from the first element through the last element, including any intervening elements outside the view.
    /// <para>先頭要素から最終要素までの連続領域を取得します（ストライド間のビュー外要素も含みます）。</para>
    /// </summary>
    public ReadOnlySpan<TElement> Data => MemoryMarshal.CreateReadOnlySpan(in reference, Extent);

    internal int RowStride => (int)strides[0];
    internal int ColumnStride => (int)strides[1];
    internal int Extent => StridedLayout.Extent(Rows, Columns, RowStride, ColumnStride);
    internal ref readonly TElement Reference => ref reference;

    /// <summary>
    /// Gets a read-only reference to the element at the specified row and column.
    /// <para>指定した行および列の要素への読み取り専用参照を取得します。</para>
    /// </summary>
    /// <param name="row">The zero-based row index. / 0 から始まる行インデックス。</param>
    /// <param name="column">The zero-based column index. / 0 から始まる列インデックス。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="row"/> or <paramref name="column"/> is out of range. / <paramref name="row"/> または <paramref name="column"/> が範囲外の場合。</exception>
    public ref readonly TElement this[int row, int column]
    {
        get
        {
            DenseStorage.CheckIndex(row, Rows, nameof(row));
            DenseStorage.CheckIndex(column, Columns, nameof(column));
            return ref ElementAt(row, column);
        }
    }

    /// <summary>
    /// Gets a zero-copy read-only view of the specified submatrix.
    /// <para>指定した範囲の部分行列を参照する読み取り専用ビューを取得します。</para>
    /// </summary>
    /// <param name="rows">The range of rows to slice. / 切り出す行の範囲。</param>
    /// <param name="columns">The range of columns to slice. / 切り出す列の範囲。</param>
    public ReadOnlyMatrixSpan<TElement> this[Range rows, Range columns]
    {
        get
        {
            var (row, rowCount) = rows.GetOffsetAndLength(Rows);
            var (column, columnCount) = columns.GetOffsetAndLength(Columns);
            ref readonly TElement start = ref rowCount == 0 || columnCount == 0 ? ref reference : ref ElementAt(row, column);
            return new(in start, rowCount, columnCount, RowStride, ColumnStride);
        }
    }

    /// <summary>
    /// Gets a zero-copy read-only view of the transpose <c>Aᵀ</c> (swapping strides without copying or conjugating).
    /// <para>転置 <c>Aᵀ</c> の読み取り専用ビューを取得します（ストライドの交換のみでコピーや複素共役は行いません）。</para>
    /// </summary>
    public ReadOnlyMatrixSpan<TElement> T => new(in reference, Columns, Rows, ColumnStride, RowStride);

    /// <summary>
    /// Returns a zero-copy read-only view of the main diagonal (length <c>min(Rows, Columns)</c>).
    /// <para>主対角成分を参照する読み取り専用ビュー（長さ <c>min(Rows, Columns)</c>）を返します。</para>
    /// </summary>
    /// <returns>A read-only vector view over the main diagonal. / 主対角成分を参照する読み取り専用ベクトルビュー。</returns>
    public ReadOnlyVectorSpan<TElement> Diagonal() => new(in reference, Math.Min(Rows, Columns), RowStride + ColumnStride);

    /// <summary>
    /// Returns a zero-copy read-only view of the specified row.
    /// <para>指定した行を参照する読み取り専用ビューを返します。</para>
    /// </summary>
    /// <param name="row">The zero-based row index. / 0 から始まる行インデックス。</param>
    /// <returns>A read-only vector view over row <paramref name="row"/>. / 第 <paramref name="row"/> 行を参照する読み取り専用ベクトルビュー。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="row"/> is out of range. / <paramref name="row"/> が範囲外の場合。</exception>
    public ReadOnlyVectorSpan<TElement> Row(int row)
    {
        DenseStorage.CheckIndex(row, Rows, nameof(row));
        return new(in Columns == 0 ? ref reference : ref ElementAt(row, 0), Columns, ColumnStride);
    }

    /// <summary>
    /// Returns a zero-copy read-only view of the specified column.
    /// <para>指定した列を参照する読み取り専用ビューを返します。</para>
    /// </summary>
    /// <param name="column">The zero-based column index. / 0 から始まる列インデックス。</param>
    /// <returns>A read-only vector view over column <paramref name="column"/>. / 第 <paramref name="column"/> 列を参照する読み取り専用ベクトルビュー。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="column"/> is out of range. / <paramref name="column"/> が範囲外の場合。</exception>
    public ReadOnlyVectorSpan<TElement> Column(int column)
    {
        DenseStorage.CheckIndex(column, Columns, nameof(column));
        return new(in Rows == 0 ? ref reference : ref ElementAt(0, column), Rows, RowStride);
    }

    /// <summary>
    /// Returns a zero-copy read-only <c>[Rows, Columns, count]</c> tensor view repeating this matrix <paramref name="count"/> times along the third axis.
    /// <para>この行列を第 3 軸方向に <paramref name="count"/> 個繰り返す、コピーなしの読み取り専用 <c>[Rows, Columns, count]</c> テンソルビューを返します。</para>
    /// </summary>
    /// <param name="count">The number of repeated matrices along the third axis. / 第 3 軸方向に並べる行列の個数。</param>
    /// <returns>A read-only rank-3 tensor view with zero matrix stride. / 行列ストライドが 0 の読み取り専用 3 階テンソルビュー。</returns>
    /// <remarks>
    /// <para>Enables applying a single matrix across all slices of a batch tensor without copying, e.g. <c>XMath.Multiply(W.Broadcast(X.Matrices.Count), X)</c>. Because distinct indices alias the same elements, the result is read-only.</para>
    /// <para><c>XMath.Multiply(W.Broadcast(X.Matrices.Count), X)</c> のように、テンソル内のすべての行列へ同じ行列をコピーなしで適用できます。異なる添字が同じ要素を参照するため読み取り専用になります。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative. / <paramref name="count"/> が負の場合。</exception>
    public ReadOnlyTensorSpan<TElement> Broadcast(int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        return new(in reference, Rows, Columns, count, RowStride, ColumnStride, 0);
    }

    /// <summary>
    /// Returns a zero-copy read-only rank-3 tensor view with a length-1 axis inserted at <paramref name="axis"/>.
    /// <para>位置 <paramref name="axis"/> に長さ 1 の軸を挿入した、コピーなしの読み取り専用 3 階テンソルビューを返します。</para>
    /// </summary>
    /// <param name="axis">The zero-based axis position (0, 1, or 2). / 0 から始まる挿入位置（0、1、または 2）。</param>
    /// <returns>A read-only rank-3 tensor view sharing the underlying storage. / 元の格納領域を共有する読み取り専用 3 階テンソルビュー。</returns>
    /// <remarks>
    /// <para><c>Unsqueeze(0)</c> produces <c>[1, Rows, Columns]</c>; <c>Unsqueeze(1)</c> produces <c>[Rows, 1, Columns]</c> (treating each column as a <c>Rows × 1</c> column matrix for batched operations such as <c>XMath.Multiply(A, X.Unsqueeze(1)).AsMatrix()</c>); <c>Unsqueeze(2)</c> produces <c>[Rows, Columns, 1]</c>.</para>
    /// <para><c>Unsqueeze(0)</c> は <c>[1, Rows, Columns]</c>、<c>Unsqueeze(1)</c> は <c>[Rows, 1, Columns]</c>（各列を <c>Rows × 1</c> の行列とみなし、<c>XMath.Multiply(A, X.Unsqueeze(1)).AsMatrix()</c> のようなバッチ演算に渡せます）、<c>Unsqueeze(2)</c> は <c>[Rows, Columns, 1]</c> になります。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="axis"/> is not 0, 1, or 2. / <paramref name="axis"/> が 0、1、2 のいずれでもない場合。</exception>
    public ReadOnlyTensorSpan<TElement> Unsqueeze(int axis) => axis switch
    {
        0 => new(in reference, 1, Rows, Columns, 0, RowStride, ColumnStride),
        1 => new(in reference, Rows, 1, Columns, RowStride, 0, ColumnStride),
        2 => new(in reference, Rows, Columns, 1, RowStride, ColumnStride, 0),
        _ => throw new ArgumentOutOfRangeException(nameof(axis), axis, "The axis must be 0, 1 or 2."),
    };

    /// <summary>
    /// Copies the elements of this view into a destination view of the same shape.
    /// <para>このビューの要素を同じ形状の出力先ビューへコピーします。</para>
    /// </summary>
    /// <param name="destination">The destination matrix view. / コピー先の行列ビュー。</param>
    /// <remarks>
    /// <para>Throws an exception if <paramref name="destination"/> partially overlaps this view.</para>
    /// <para><paramref name="destination"/> がこのビューと部分的に重なっている場合は例外をスローします。</para>
    /// </remarks>
    public void CopyTo(MatrixSpan<TElement> destination) => StridedKernels.Copy(this, destination);

    /// <summary>
    /// Copies the elements of this view into a new <see cref="Matrix{TElement}"/>.
    /// <para>このビューの要素を新しい <see cref="Matrix{TElement}"/> へコピーします。</para>
    /// </summary>
    /// <returns>A new column-major matrix containing the copied elements. / 要素をコピーした新しい列優先行列。</returns>
    /// <exception cref="OverflowException"><c>Rows * Columns</c> exceeds <see cref="int.MaxValue"/> (for example, on a broadcasted view). / ブロードキャストされたビューなどで要素数 <c>Rows * Columns</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    public Matrix<TElement> ToMatrix()
    {
        var result = Matrix<TElement>.Uninitialized(Rows, Columns);
        CopyTo(result);
        return result;
    }

    /// <summary>
    /// Copies the elements of this view into a new two-dimensional array preserving indices.
    /// <para>添字を維持してこのビューの要素を新しい 2 次元配列へコピーします。</para>
    /// </summary>
    /// <returns>A new two-dimensional array containing the copied elements. / 要素をコピーした新しい 2 次元配列。</returns>
    public TElement[,] ToArray()
    {
        var result = new TElement[Rows, Columns];
        for (int j = 0; j < Columns; j++)
        {
            for (int i = 0; i < Rows; i++)
            {
                result[i, j] = ElementAt(i, j);
            }
        }

        return result;
    }

    internal ref readonly TElement ElementAt(int row, int column) =>
        ref Unsafe.Add(ref Unsafe.AsRef(in reference), (nint)row * RowStride + (nint)column * ColumnStride);

    internal bool TryGetDense(out ReadOnlySpan<TElement> span)
    {
        bool dense = (Rows <= 1 || RowStride == 1) && (Columns <= 1 || ColumnStride == Rows);
        span = dense ? MemoryMarshal.CreateReadOnlySpan(in reference, Rows * Columns) : default;
        return dense;
    }

    /// <summary>
    /// Computes the elementwise sum of two matrix views.
    /// <para>2 つの行列ビューの要素ごとの和を計算します。</para>
    /// </summary>
    /// <param name="a">The left-hand matrix. / 左辺の行列。</param>
    /// <param name="b">The right-hand matrix. / 右辺の行列。</param>
    /// <returns>A new matrix containing <c>a + b</c>. / <c>a + b</c> を格納した新しい行列。</returns>
    public static Matrix<TElement> operator +(ReadOnlyMatrixSpan<TElement> a, ReadOnlyMatrixSpan<TElement> b) => XMath.Add(a, b);

    /// <summary>
    /// Computes the elementwise difference of two matrix views.
    /// <para>2 つの行列ビューの要素ごとの差を計算します。</para>
    /// </summary>
    /// <param name="a">The left-hand matrix. / 左辺の行列。</param>
    /// <param name="b">The right-hand matrix. / 右辺の行列。</param>
    /// <returns>A new matrix containing <c>a - b</c>. / <c>a - b</c> を格納した新しい行列。</returns>
    public static Matrix<TElement> operator -(ReadOnlyMatrixSpan<TElement> a, ReadOnlyMatrixSpan<TElement> b) => XMath.Subtract(a, b);

    /// <summary>
    /// Negates each element of the matrix view.
    /// <para>行列ビューの各要素の符号を反転します。</para>
    /// </summary>
    /// <param name="a">The input matrix. / 入力行列。</param>
    /// <returns>A new matrix containing <c>-a</c>. / <c>-a</c> を格納した新しい行列。</returns>
    public static Matrix<TElement> operator -(ReadOnlyMatrixSpan<TElement> a) => XMath.Negate(a);

    /// <summary>
    /// Multiplies a matrix view by a scalar.
    /// <para>行列ビューをスカラー倍します。</para>
    /// </summary>
    /// <param name="scalar">The scalar multiplier. / 掛けるスカラー値。</param>
    /// <param name="a">The input matrix. / 入力行列。</param>
    /// <returns>A new matrix containing <c>scalar * a</c>. / <c>scalar * a</c> を格納した新しい行列。</returns>
    public static Matrix<TElement> operator *(TElement scalar, ReadOnlyMatrixSpan<TElement> a) => XMath.Multiply(a, scalar);

    /// <summary>
    /// Multiplies a matrix view by a scalar.
    /// <para>行列ビューをスカラー倍します。</para>
    /// </summary>
    /// <param name="a">The input matrix. / 入力行列。</param>
    /// <param name="scalar">The scalar multiplier. / 掛けるスカラー値。</param>
    /// <returns>A new matrix containing <c>a * scalar</c>. / <c>a * scalar</c> を格納した新しい行列。</returns>
    public static Matrix<TElement> operator *(ReadOnlyMatrixSpan<TElement> a, TElement scalar) => XMath.Multiply(a, scalar);

    /// <summary>
    /// Computes the matrix product <c>a * b</c>.
    /// <para>行列積 <c>a * b</c> を計算します。</para>
    /// </summary>
    /// <param name="a">The left-hand matrix. / 左辺の行列。</param>
    /// <param name="b">The right-hand matrix. / 右辺の行列。</param>
    /// <returns>A new matrix containing <c>a * b</c>. / <c>a * b</c> を格納した新しい行列。</returns>
    public static Matrix<TElement> operator *(ReadOnlyMatrixSpan<TElement> a, ReadOnlyMatrixSpan<TElement> b) => XMath.Multiply(a, b);
}

/// <summary>
/// Represents a zero-copy, strided writable view over a two-dimensional matrix.
/// <para>ストライド付きの 2 次元行列を参照する、コピーなしの書き込み可能なビューを表します。</para>
/// </summary>
/// <typeparam name="TElement">The element type of the matrix. / 行列の要素型。</typeparam>
/// <remarks>
/// <para>Every distinct <c>(row, column)</c> index pair maps to a distinct memory location (zero-stride broadcasting is only permitted on <see cref="ReadOnlyMatrixSpan{TElement}"/>).</para>
/// <para><see cref="Data"/>, <see cref="Lengths"/>, and <see cref="Strides"/> can be passed directly to the <c>System.Numerics.Tensors.TensorSpan&lt;T&gt;</c> constructor.</para>
/// <para>異なる <c>(row, column)</c> 添字ペアは必ず異なるメモリ位置を指します（ストライド 0 によるブロードキャストは <see cref="ReadOnlyMatrixSpan{TElement}"/> のみで許可されます）。</para>
/// <para><see cref="Data"/>、<see cref="Lengths"/>、<see cref="Strides"/> は <c>System.Numerics.Tensors.TensorSpan&lt;T&gt;</c> のコンストラクターにそのまま渡せます。</para>
/// </remarks>
public readonly ref struct MatrixSpan<TElement> where TElement : unmanaged, INumberBase<TElement>
{
    private readonly ref TElement reference;
    private readonly NintPair lengths;
    private readonly NintPair strides;

    internal MatrixSpan(ref TElement reference, int rows, int columns, int rowStride, int columnStride)
    {
        this.reference = ref reference;
        lengths = NintPair.Create(rows, columns);
        strides = NintPair.Create(StridedLayout.Normalize(rows, rowStride), StridedLayout.Normalize(columns, columnStride));
    }

    /// <summary>
    /// Initializes a writable view over a contiguous column-major span.
    /// <para>列優先の連続領域を参照する書き込み可能なビューを初期化します。</para>
    /// </summary>
    /// <param name="data">The contiguous column-major span of length <c>rows * columns</c>. / 長さ <c>rows * columns</c> の列優先連続領域。</param>
    /// <param name="rows">The number of rows. / 行数。</param>
    /// <param name="columns">The number of columns. / 列数。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rows"/> or <paramref name="columns"/> is negative. / <paramref name="rows"/> または <paramref name="columns"/> が負の場合。</exception>
    /// <exception cref="OverflowException"><c>rows * columns</c> exceeds <see cref="int.MaxValue"/>. / 要素数 <c>rows * columns</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="data"/> does not have length <c>rows * columns</c>. / <paramref name="data"/> の長さが <c>rows * columns</c> と一致しない場合。</exception>
    public MatrixSpan(Span<TElement> data, int rows, int columns)
    {
        DenseStorage.RequireLength(DenseStorage.Length(rows, columns), data.Length);
        this = new(ref MemoryMarshal.GetReference(data), rows, columns, 1, rows);
    }

    /// <summary>
    /// Initializes a writable view over the specified span with explicit shape and strides.
    /// <para>形状とストライドを指定して領域を参照する書き込み可能なビューを初期化します。</para>
    /// </summary>
    /// <param name="data">The backing span starting at the first element and covering through the last element. / 先頭要素から最終要素までを含む格納領域。</param>
    /// <param name="lengths">A length-2 span specifying <c>[rows, columns]</c>. / <c>[行数, 列数]</c> を指定する長さ 2 の形状。</param>
    /// <param name="strides">A length-2 span specifying non-negative element strides (ignored for axes of length 0 or 1). / 非負の要素ストライドを指定する長さ 2 のスパン（長さが 0 または 1 の軸では無視されます）。</param>
    /// <exception cref="ArgumentException"><paramref name="lengths"/> or <paramref name="strides"/> does not have length 2, <paramref name="data"/> is too short, or distinct indices alias the same element. / <paramref name="lengths"/> または <paramref name="strides"/> の長さが 2 でない、<paramref name="data"/> の長さが不足している、または異なる添字が同じ要素を参照する場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">A length or stride is negative or exceeds <see cref="int.MaxValue"/>. / 長さまたはストライドが負、または <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="OverflowException">The spanned extent exceeds <see cref="int.MaxValue"/>. / 参照範囲の要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public MatrixSpan(Span<TElement> data, ReadOnlySpan<nint> lengths, ReadOnlySpan<nint> strides)
    {
        var (rows, columns, rowStride, columnStride) = StridedLayout.Validate(lengths, strides, 2, writable: true);
        int extent = StridedLayout.Extent(rows, columns, rowStride, columnStride);
        if (data.Length < extent)
        {
            throw new ArgumentException("The data is too short for the lengths and strides.", nameof(data));
        }

        this = new(ref MemoryMarshal.GetReference(data), rows, columns, rowStride, columnStride);
    }

    /// <summary>
    /// Creates a writable matrix view from a reference to the first element, shape, and strides.
    /// <para>先頭要素への参照・形状・ストライドから書き込み可能な行列ビューを作成します。</para>
    /// </summary>
    /// <param name="reference">A reference to the first element. / 先頭要素への参照。</param>
    /// <param name="lengths">A length-2 span specifying <c>[rows, columns]</c>. / <c>[行数, 列数]</c> を指定する長さ 2 の形状。</param>
    /// <param name="strides">A length-2 span specifying non-negative element strides. / 非負の要素ストライドを指定する長さ 2 のスパン。</param>
    /// <returns>A writable matrix view referencing the specified memory layout. / 指定したレイアウトを参照する書き込み可能な行列ビュー。</returns>
    /// <remarks>
    /// <para>Compatible with <c>System.Numerics.Tensors</c> via <c>Create(ref t.GetPinnableReference(), t.Lengths, t.Strides)</c>. Bounds of the backing memory are not checked, matching <see cref="MemoryMarshal.CreateSpan"/>.</para>
    /// <para><c>System.Numerics.Tensors</c> からは <c>Create(ref t.GetPinnableReference(), t.Lengths, t.Strides)</c> で変換できます。<see cref="MemoryMarshal.CreateSpan"/> と同様に、メモリ範囲の境界検査は行いません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="lengths"/> or <paramref name="strides"/> does not have length 2, or distinct indices alias the same element. / <paramref name="lengths"/> または <paramref name="strides"/> の長さが 2 でない、または異なる添字が同じ要素を参照する場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">A length or stride is negative or exceeds <see cref="int.MaxValue"/>. / 長さまたはストライドが負、または <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="OverflowException">The spanned extent exceeds <see cref="int.MaxValue"/>. / 参照範囲の要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static MatrixSpan<TElement> Create(ref TElement reference, ReadOnlySpan<nint> lengths, ReadOnlySpan<nint> strides)
    {
        var (rows, columns, rowStride, columnStride) = StridedLayout.Validate(lengths, strides, 2, writable: true);
        return new(ref reference, rows, columns, rowStride, columnStride);
    }

    /// <summary>
    /// Gets the number of rows in the view.
    /// <para>ビューの行数を取得します。</para>
    /// </summary>
    public int Rows => (int)lengths[0];

    /// <summary>
    /// Gets the number of columns in the view.
    /// <para>ビューの列数を取得します。</para>
    /// </summary>
    public int Columns => (int)lengths[1];

    /// <summary>
    /// Gets the number of dimensions, which is always 2.
    /// <para>次元数（常に 2）を取得します。</para>
    /// </summary>
    public int Rank => 2;

    /// <summary>
    /// Gets the total number of elements (<c>Rows * Columns</c>).
    /// <para>総要素数（<c>Rows * Columns</c>）を取得します。</para>
    /// </summary>
    public nint FlattenedLength => (nint)Rows * Columns;

    /// <summary>
    /// Gets a value indicating whether the view contains no elements.
    /// <para>ビューが空かどうかを示す値を取得します。</para>
    /// </summary>
    public bool IsEmpty => Rows == 0 || Columns == 0;

    /// <summary>
    /// Gets the shape <c>[Rows, Columns]</c> of the view.
    /// <para>ビューの形状 <c>[Rows, Columns]</c> を取得します。</para>
    /// </summary>
    [UnscopedRef] public ReadOnlySpan<nint> Lengths => lengths;

    /// <summary>
    /// Gets the element strides along each axis (0 for axes of length 0 or 1).
    /// <para>各軸の要素単位のストライドを取得します（長さが 0 または 1 の軸は 0）。</para>
    /// </summary>
    [UnscopedRef] public ReadOnlySpan<nint> Strides => strides;

    /// <summary>
    /// Gets a contiguous span from the first element through the last element, including any intervening elements outside the view.
    /// <para>先頭要素から最終要素までの連続領域を取得します（ストライド間のビュー外要素も含みます）。</para>
    /// </summary>
    public Span<TElement> Data => MemoryMarshal.CreateSpan(ref reference, AsReadOnly().Extent);

    /// <summary>
    /// Gets a writable reference to the element at the specified row and column.
    /// <para>指定した行および列の要素への書き込み可能な参照を取得します。</para>
    /// </summary>
    /// <param name="row">The zero-based row index. / 0 から始まる行インデックス。</param>
    /// <param name="column">The zero-based column index. / 0 から始まる列インデックス。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="row"/> or <paramref name="column"/> is out of range. / <paramref name="row"/> または <paramref name="column"/> が範囲外の場合。</exception>
    public ref TElement this[int row, int column] => ref Unsafe.AsRef(in AsReadOnly()[row, column]);

    /// <summary>
    /// Gets a zero-copy writable view of the specified submatrix.
    /// <para>指定した範囲の部分行列を参照する書き込み可能なビューを取得します。</para>
    /// </summary>
    /// <param name="rows">The range of rows to slice. / 切り出す行の範囲。</param>
    /// <param name="columns">The range of columns to slice. / 切り出す列の範囲。</param>
    public MatrixSpan<TElement> this[Range rows, Range columns] => Writable(AsReadOnly()[rows, columns]);

    /// <summary>
    /// Gets a zero-copy writable view of the transpose <c>Aᵀ</c> (swapping strides without copying or conjugating).
    /// <para>転置 <c>Aᵀ</c> の書き込み可能なビューを取得します（ストライドの交換のみでコピーや複素共役は行いません）。</para>
    /// </summary>
    public MatrixSpan<TElement> T => Writable(AsReadOnly().T);

    /// <summary>
    /// Returns a zero-copy writable view of the main diagonal (length <c>min(Rows, Columns)</c>).
    /// <para>主対角成分を参照する書き込み可能なビュー（長さ <c>min(Rows, Columns)</c>）を返します。</para>
    /// </summary>
    /// <returns>A writable vector view over the main diagonal. / 主対角成分を参照する書き込み可能なベクトルビュー。</returns>
    public VectorSpan<TElement> Diagonal() => VectorSpan<TElement>.Writable(AsReadOnly().Diagonal());

    /// <summary>
    /// Returns a zero-copy writable view of the specified row.
    /// <para>指定した行を参照する書き込み可能なビューを返します。</para>
    /// </summary>
    /// <param name="row">The zero-based row index. / 0 から始まる行インデックス。</param>
    /// <returns>A writable vector view over row <paramref name="row"/>. / 第 <paramref name="row"/> 行を参照する書き込み可能なベクトルビュー。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="row"/> is out of range. / <paramref name="row"/> が範囲外の場合。</exception>
    public VectorSpan<TElement> Row(int row) => VectorSpan<TElement>.Writable(AsReadOnly().Row(row));

    /// <summary>
    /// Returns a zero-copy writable view of the specified column.
    /// <para>指定した列を参照する書き込み可能なビューを返します。</para>
    /// </summary>
    /// <param name="column">The zero-based column index. / 0 から始まる列インデックス。</param>
    /// <returns>A writable vector view over column <paramref name="column"/>. / 第 <paramref name="column"/> 列を参照する書き込み可能なベクトルビュー。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="column"/> is out of range. / <paramref name="column"/> が範囲外の場合。</exception>
    public VectorSpan<TElement> Column(int column) => VectorSpan<TElement>.Writable(AsReadOnly().Column(column));

    /// <summary>
    /// Returns a zero-copy read-only <c>[Rows, Columns, count]</c> tensor view repeating this matrix <paramref name="count"/> times along the third axis.
    /// <para>この行列を第 3 軸方向に <paramref name="count"/> 個繰り返す、コピーなしの読み取り専用 <c>[Rows, Columns, count]</c> テンソルビューを返します。</para>
    /// </summary>
    /// <param name="count">The number of repeated matrices along the third axis. / 第 3 軸方向に並べる行列の個数。</param>
    /// <returns>A read-only rank-3 tensor view with zero matrix stride. / 行列ストライドが 0 の読み取り専用 3 階テンソルビュー。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative. / <paramref name="count"/> が負の場合。</exception>
    public ReadOnlyTensorSpan<TElement> Broadcast(int count) => AsReadOnly().Broadcast(count);

    /// <summary>
    /// Returns a zero-copy writable rank-3 tensor view with a length-1 axis inserted at <paramref name="axis"/>.
    /// <para>位置 <paramref name="axis"/> に長さ 1 の軸を挿入した、コピーなしの書き込み可能な 3 階テンソルビューを返します。</para>
    /// </summary>
    /// <param name="axis">The zero-based axis position (0, 1, or 2). / 0 から始まる挿入位置（0、1、または 2）。</param>
    /// <returns>A writable rank-3 tensor view sharing the underlying storage. / 元の格納領域を共有する書き込み可能な 3 階テンソルビュー。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="axis"/> is not 0, 1, or 2. / <paramref name="axis"/> が 0、1、2 のいずれでもない場合。</exception>
    public TensorSpan<TElement> Unsqueeze(int axis) => TensorSpan<TElement>.Writable(AsReadOnly().Unsqueeze(axis));

    /// <summary>
    /// Returns a read-only view over the same elements.
    /// <para>同じ要素を参照する読み取り専用ビューを返します。</para>
    /// </summary>
    /// <returns>A read-only matrix view. / 読み取り専用の行列ビュー。</returns>
    public ReadOnlyMatrixSpan<TElement> AsReadOnly() =>
        new(in reference, Rows, Columns, (int)strides[0], (int)strides[1]);

    /// <summary>
    /// Fills all elements of the matrix view with the specified value.
    /// <para>行列ビューのすべての要素を指定した値で埋めます。</para>
    /// </summary>
    /// <param name="value">The value to assign to each element. / 各要素に設定する値。</param>
    public void Fill(TElement value)
    {
        for (int j = 0; j < Columns; j++)
        {
            for (int i = 0; i < Rows; i++)
            {
                Unsafe.AsRef(in AsReadOnly().ElementAt(i, j)) = value;
            }
        }
    }

    /// <summary>
    /// Copies the elements of this view into a destination view of the same shape.
    /// <para>このビューの要素を同じ形状の出力先ビューへコピーします。</para>
    /// </summary>
    /// <param name="destination">The destination matrix view. / コピー先の行列ビュー。</param>
    /// <remarks>
    /// <para>Throws an exception if <paramref name="destination"/> partially overlaps this view.</para>
    /// <para><paramref name="destination"/> がこのビューと部分的に重なっている場合は例外をスローします。</para>
    /// </remarks>
    public void CopyTo(MatrixSpan<TElement> destination) => AsReadOnly().CopyTo(destination);

    /// <summary>
    /// Copies the elements of this view into a new <see cref="Matrix{TElement}"/>.
    /// <para>このビューの要素を新しい <see cref="Matrix{TElement}"/> へコピーします。</para>
    /// </summary>
    /// <returns>A new column-major matrix containing the copied elements. / 要素をコピーした新しい列優先行列。</returns>
    public Matrix<TElement> ToMatrix() => AsReadOnly().ToMatrix();

    /// <summary>
    /// Copies the elements of this view into a new two-dimensional array preserving indices.
    /// <para>添字を維持してこのビューの要素を新しい 2 次元配列へコピーします。</para>
    /// </summary>
    /// <returns>A new two-dimensional array containing the copied elements. / 要素をコピーした新しい 2 次元配列。</returns>
    public TElement[,] ToArray() => AsReadOnly().ToArray();

    internal bool TryGetDense(out Span<TElement> span)
    {
        bool dense = AsReadOnly().TryGetDense(out var readOnly);
        span = dense ? MemoryMarshal.CreateSpan(ref MemoryMarshal.GetReference(readOnly), readOnly.Length) : default;
        return dense;
    }

    internal static MatrixSpan<TElement> Writable(ReadOnlyMatrixSpan<TElement> view) =>
        new(ref Unsafe.AsRef(in view.Reference), view.Rows, view.Columns, view.RowStride, view.ColumnStride);

    /// <summary>
    /// Implicitly converts a writable <see cref="MatrixSpan{TElement}"/> to a <see cref="ReadOnlyMatrixSpan{TElement}"/>.
    /// <para>書き込み可能な <see cref="MatrixSpan{TElement}"/> を <see cref="ReadOnlyMatrixSpan{TElement}"/> へ暗黙的に変換します。</para>
    /// </summary>
    /// <param name="view">The writable view to convert. / 変換元の書き込み可能なビュー。</param>
    /// <returns>A read-only view referencing the same elements. / 同じ要素を参照する読み取り専用ビュー。</returns>
    public static implicit operator ReadOnlyMatrixSpan<TElement>(MatrixSpan<TElement> view) => view.AsReadOnly();

    /// <summary>
    /// Computes the elementwise sum of two matrix views.
    /// <para>2 つの行列ビューの要素ごとの和を計算します。</para>
    /// </summary>
    /// <param name="a">The left-hand matrix. / 左辺の行列。</param>
    /// <param name="b">The right-hand matrix. / 右辺の行列。</param>
    /// <returns>A new matrix containing <c>a + b</c>. / <c>a + b</c> を格納した新しい行列。</returns>
    public static Matrix<TElement> operator +(MatrixSpan<TElement> a, MatrixSpan<TElement> b) => XMath.Add(a, b);

    /// <summary>
    /// Computes the elementwise difference of two matrix views.
    /// <para>2 つの行列ビューの要素ごとの差を計算します。</para>
    /// </summary>
    /// <param name="a">The left-hand matrix. / 左辺の行列。</param>
    /// <param name="b">The right-hand matrix. / 右辺の行列。</param>
    /// <returns>A new matrix containing <c>a - b</c>. / <c>a - b</c> を格納した新しい行列。</returns>
    public static Matrix<TElement> operator -(MatrixSpan<TElement> a, MatrixSpan<TElement> b) => XMath.Subtract(a, b);

    /// <summary>
    /// Negates each element of the matrix view.
    /// <para>行列ビューの各要素の符号を反転します。</para>
    /// </summary>
    /// <param name="a">The input matrix. / 入力行列。</param>
    /// <returns>A new matrix containing <c>-a</c>. / <c>-a</c> を格納した新しい行列。</returns>
    public static Matrix<TElement> operator -(MatrixSpan<TElement> a) => XMath.Negate(a);

    /// <summary>
    /// Multiplies a matrix view by a scalar.
    /// <para>行列ビューをスカラー倍します。</para>
    /// </summary>
    /// <param name="scalar">The scalar multiplier. / 掛けるスカラー値。</param>
    /// <param name="a">The input matrix. / 入力行列。</param>
    /// <returns>A new matrix containing <c>scalar * a</c>. / <c>scalar * a</c> を格納した新しい行列。</returns>
    public static Matrix<TElement> operator *(TElement scalar, MatrixSpan<TElement> a) => XMath.Multiply(a, scalar);

    /// <summary>
    /// Multiplies a matrix view by a scalar.
    /// <para>行列ビューをスカラー倍します。</para>
    /// </summary>
    /// <param name="a">The input matrix. / 入力行列。</param>
    /// <param name="scalar">The scalar multiplier. / 掛けるスカラー値。</param>
    /// <returns>A new matrix containing <c>a * scalar</c>. / <c>a * scalar</c> を格納した新しい行列。</returns>
    public static Matrix<TElement> operator *(MatrixSpan<TElement> a, TElement scalar) => XMath.Multiply(a, scalar);

    /// <summary>
    /// Computes the matrix product <c>a * b</c>.
    /// <para>行列積 <c>a * b</c> を計算します。</para>
    /// </summary>
    /// <param name="a">The left-hand matrix. / 左辺の行列。</param>
    /// <param name="b">The right-hand matrix. / 右辺の行列。</param>
    /// <returns>A new matrix containing <c>a * b</c>. / <c>a * b</c> を格納した新しい行列。</returns>
    public static Matrix<TElement> operator *(MatrixSpan<TElement> a, MatrixSpan<TElement> b) => XMath.Multiply(a, b);
}
