using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Tuna.LinearAlgebra;

/// <summary>
/// Represents a zero-copy, strided read-only view over a rank-3 tensor.
/// <para>ストライド付きの 3 階テンソルを参照する、コピーなしの読み取り専用ビューを表します。</para>
/// </summary>
/// <typeparam name="TElement">The element type of the tensor. / テンソルの要素型。</typeparam>
/// <remarks>
/// <para>Element <c>(i, j, k)</c> (entry <c>(i, j)</c> of matrix <c>k</c>) is located at offset <c>i * Strides[0] + j * Strides[1] + k * Strides[2]</c>. Non-rank-3 shapes throw <see cref="NotSupportedException"/>.</para>
/// <para><see cref="Data"/>, <see cref="Lengths"/>, and <see cref="Strides"/> can be passed directly to the <c>System.Numerics.Tensors.ReadOnlyTensorSpan&lt;T&gt;</c> constructor; use <see cref="Create"/> for the reverse conversion.</para>
/// <para>要素 <c>(i, j, k)</c>（第 <c>k</c> 行列の <c>(i, j)</c> 成分）は先頭参照から <c>i * Strides[0] + j * Strides[1] + k * Strides[2]</c> 要素目の位置にあります。3 階以外の形状は <see cref="NotSupportedException"/> をスローします。</para>
/// <para><see cref="Data"/>、<see cref="Lengths"/>、<see cref="Strides"/> は <c>System.Numerics.Tensors.ReadOnlyTensorSpan&lt;T&gt;</c> のコンストラクターにそのまま渡せます。逆方向の変換には <see cref="Create"/> を使用してください。</para>
/// </remarks>
public readonly ref struct ReadOnlyTensorSpan<TElement> where TElement : unmanaged, INumberBase<TElement>
{
    private readonly ref readonly TElement reference;
    private readonly NintTriple lengths;
    private readonly NintTriple strides;

    internal ReadOnlyTensorSpan(ref readonly TElement reference, int rows, int columns, int count, int rowStride, int columnStride, int matrixStride)
    {
        this.reference = ref reference;
        lengths = NintTriple.Create(rows, columns, count);
        strides = NintTriple.Create(
            StridedLayout.Normalize(rows, rowStride), StridedLayout.Normalize(columns, columnStride), StridedLayout.Normalize(count, matrixStride));
    }

    /// <summary>
    /// Initializes a read-only view over a contiguous first-index-fastest span.
    /// <para>先頭軸が最も速く変化する順の連続領域を参照する読み取り専用ビューを初期化します。</para>
    /// </summary>
    /// <param name="data">The contiguous span whose length equals the product of <paramref name="lengths"/>. / 長さが <paramref name="lengths"/> の積と一致する連続領域。</param>
    /// <param name="lengths">The shape <c>[rows, columns, matrixCount]</c>. / 形状 <c>[行数, 列数, 行列数]</c>。</param>
    /// <exception cref="NotSupportedException">The rank is not 3. / 階数が 3 でない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">Any dimension length is negative or exceeds <see cref="int.MaxValue"/>. / いずれかの軸長が負、または <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="OverflowException">The total number of elements exceeds <see cref="int.MaxValue"/>. / 総要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="data"/> does not match the total element count. / <paramref name="data"/> の長さが総要素数と一致しない場合。</exception>
    public ReadOnlyTensorSpan(ReadOnlySpan<TElement> data, params ReadOnlySpan<nint> lengths)
    {
        var (rows, columns, count) = StridedBatchLayout.ValidateLengths(lengths);
        this = new(data, rows, columns, count);
    }

    internal ReadOnlyTensorSpan(ReadOnlySpan<TElement> data, int rows, int columns, int count)
    {
        DenseStorage.RequireLength(DenseStorage.Length(rows, columns, count), data.Length);
        this = new(in MemoryMarshal.GetReference(data), rows, columns, count, 1, rows, DenseStorage.MatrixStride(rows, columns, count));
    }

    /// <summary>
    /// Initializes a read-only view over the specified span with explicit shape and strides.
    /// <para>形状とストライドを指定して領域を参照する読み取り専用ビューを初期化します。</para>
    /// </summary>
    /// <param name="data">The backing span starting at the first element and covering through the last element. / 先頭要素から最終要素までを含む格納領域。</param>
    /// <param name="lengths">The shape <c>[rows, columns, matrixCount]</c>. / 形状 <c>[行数, 列数, 行列数]</c>。</param>
    /// <param name="strides">Non-negative element strides matching the rank of <paramref name="lengths"/> (ignored for axes of length 0 or 1). / <paramref name="lengths"/> と同じ長さの非負ストライド（長さが 0 または 1 の軸では無視されます）。</param>
    /// <exception cref="ArgumentException"><paramref name="lengths"/> and <paramref name="strides"/> have different lengths, or <paramref name="data"/> is too short. / <paramref name="lengths"/> と <paramref name="strides"/> の長さが異なる、または <paramref name="data"/> の長さが不足している場合。</exception>
    /// <exception cref="NotSupportedException">The rank is not 3. / 階数が 3 でない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">A length or stride is negative or exceeds <see cref="int.MaxValue"/>. / 長さまたはストライドが負、または <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="OverflowException">The spanned extent exceeds <see cref="int.MaxValue"/>. / 参照範囲の要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public ReadOnlyTensorSpan(ReadOnlySpan<TElement> data, ReadOnlySpan<nint> lengths, ReadOnlySpan<nint> strides)
    {
        var (rows, columns, count, rowStride, columnStride, matrixStride) = StridedBatchLayout.Validate(lengths, strides, writable: false);
        if (data.Length < StridedBatchLayout.Extent(rows, columns, count, rowStride, columnStride, matrixStride))
        {
            throw new ArgumentException("The data is too short for the lengths and strides.", nameof(data));
        }

        this = new(in MemoryMarshal.GetReference(data), rows, columns, count, rowStride, columnStride, matrixStride);
    }

    /// <summary>
    /// Creates a read-only tensor view from a reference to the first element, shape, and strides.
    /// <para>先頭要素への参照・形状・ストライドから読み取り専用テンソルビューを作成します。</para>
    /// </summary>
    /// <param name="reference">A reference to the first element. / 先頭要素への参照。</param>
    /// <param name="lengths">The shape <c>[rows, columns, matrixCount]</c>. / 形状 <c>[行数, 列数, 行列数]</c>。</param>
    /// <param name="strides">Non-negative element strides for each axis. / 各軸の非負ストライド。</param>
    /// <returns>A read-only tensor view referencing the specified memory layout. / 指定したレイアウトを参照する読み取り専用テンソルビュー。</returns>
    /// <remarks>
    /// <para>Compatible with <c>System.Numerics.Tensors</c> via <c>Create(in t.GetPinnableReference(), t.Lengths, t.Strides)</c>. Bounds of the backing memory are not checked, matching <see cref="MemoryMarshal.CreateReadOnlySpan"/>.</para>
    /// <para><c>System.Numerics.Tensors</c> からは <c>Create(in t.GetPinnableReference(), t.Lengths, t.Strides)</c> で変換できます。<see cref="MemoryMarshal.CreateReadOnlySpan"/> と同様に、メモリ範囲の境界検査は行いません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="lengths"/> and <paramref name="strides"/> have different lengths. / <paramref name="lengths"/> と <paramref name="strides"/> の長さが異なる場合。</exception>
    /// <exception cref="NotSupportedException">The rank is not 3. / 階数が 3 でない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">A length or stride is negative or exceeds <see cref="int.MaxValue"/>. / 長さまたはストライドが負、または <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="OverflowException">The spanned extent exceeds <see cref="int.MaxValue"/>. / 参照範囲の要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static ReadOnlyTensorSpan<TElement> Create(ref readonly TElement reference, ReadOnlySpan<nint> lengths, ReadOnlySpan<nint> strides)
    {
        var (rows, columns, count, rowStride, columnStride, matrixStride) = StridedBatchLayout.Validate(lengths, strides, writable: false);
        return new(in reference, rows, columns, count, rowStride, columnStride, matrixStride);
    }

    /// <summary>
    /// Gets the rank (number of dimensions) of the view, which is currently always 3.
    /// <para>ビューの階数（次元数、現在は常に 3）を取得します。</para>
    /// </summary>
    public int Rank => StridedBatchLayout.SupportedRank;

    /// <summary>
    /// Gets the total number of elements (the product of <see cref="Lengths"/>).
    /// <para>総要素数（<see cref="Lengths"/> の積）を取得します。</para>
    /// </summary>
    public nint FlattenedLength => (nint)Rows * Columns * MatrixCount;

    /// <summary>
    /// Gets a value indicating whether the view contains no elements.
    /// <para>ビューが空かどうかを示す値を取得します。</para>
    /// </summary>
    public bool IsEmpty => Rows == 0 || Columns == 0 || MatrixCount == 0;

    /// <summary>
    /// Gets the shape <c>[rows, columns, matrixCount]</c> of the view.
    /// <para>ビューの形状 <c>[行数, 列数, 行列数]</c> を取得します。</para>
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

    internal int Rows => (int)lengths[0];
    internal int Columns => (int)lengths[1];
    internal int MatrixCount => (int)lengths[2];
    internal int RowStride => (int)strides[0];
    internal int ColumnStride => (int)strides[1];
    internal int MatrixStride => (int)strides[2];
    internal int Extent => StridedBatchLayout.Extent(Rows, Columns, MatrixCount, RowStride, ColumnStride, MatrixStride);
    internal ref readonly TElement Reference => ref reference;

    /// <summary>
    /// Gets a read-only reference to element <c>(row, column)</c> of matrix <paramref name="index"/>.
    /// <para>第 <paramref name="index"/> 行列の要素 <c>(row, column)</c> への読み取り専用参照を取得します。</para>
    /// </summary>
    /// <param name="row">The zero-based row index. / 0 から始まる行インデックス。</param>
    /// <param name="column">The zero-based column index. / 0 から始まる列インデックス。</param>
    /// <param name="index">The zero-based matrix index. / 0 から始まる行列インデックス。</param>
    /// <exception cref="ArgumentOutOfRangeException">Any index is out of range. / いずれかのインデックスが範囲外の場合。</exception>
    public ref readonly TElement this[int row, int column, int index]
    {
        get
        {
            DenseStorage.CheckIndex(row, Rows, nameof(row));
            DenseStorage.CheckIndex(column, Columns, nameof(column));
            DenseStorage.CheckIndex(index, MatrixCount, nameof(index));
            return ref ElementAt(row, column, index);
        }
    }

    /// <summary>
    /// Returns a zero-copy read-only view of matrix <paramref name="index"/> (<c>Lengths[0] × Lengths[1]</c>).
    /// <para>第 <paramref name="index"/> 行列（<c>Lengths[0] × Lengths[1]</c>）を参照する読み取り専用ビューを返します。</para>
    /// </summary>
    /// <param name="index">The zero-based matrix index. / 0 から始まる行列インデックス。</param>
    /// <returns>A read-only matrix view sharing the underlying storage. / 元の格納領域を共有する読み取り専用行列ビュー。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is out of range. / <paramref name="index"/> が範囲外の場合。</exception>
    public ReadOnlyMatrixSpan<TElement> GetMatrix(int index)
    {
        DenseStorage.CheckIndex(index, MatrixCount, nameof(index));
        return Matrix(index);
    }

    /// <summary>
    /// Returns a zero-copy read-only view of column <paramref name="column"/> of matrix <paramref name="index"/>.
    /// <para>第 <paramref name="index"/> 行列の第 <paramref name="column"/> 列を参照する読み取り専用ビューを返します。</para>
    /// </summary>
    /// <param name="column">The zero-based column index. / 0 から始まる列インデックス。</param>
    /// <param name="index">The zero-based matrix index. / 0 から始まる行列インデックス。</param>
    /// <returns>A read-only vector view equivalent to <c>GetMatrix(index).Column(column)</c>. / <c>GetMatrix(index).Column(column)</c> と同等の読み取り専用ベクトルビュー。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="column"/> or <paramref name="index"/> is out of range. / <paramref name="column"/> または <paramref name="index"/> が範囲外の場合。</exception>
    public ReadOnlyVectorSpan<TElement> GetVector(int column, int index)
    {
        DenseStorage.CheckIndex(index, MatrixCount, nameof(index));
        return Matrix(index).Column(column);
    }

    /// <summary>
    /// Gets an indexed, allocation-free enumerable view over all matrix slices.
    /// <para>すべての行列スライスにインデックスアクセスおよび列挙できる読み取り専用ビューを取得します。</para>
    /// </summary>
    public ReadOnlyTensorMatrices<TElement> Matrices => new(this);

    /// <summary>
    /// Gets a zero-copy read-only view over the specified subrange of rows, columns, and matrix indices.
    /// <para>指定した行・列・行列インデックスの範囲を参照する読み取り専用ビューを取得します。</para>
    /// </summary>
    /// <param name="rows">The range of rows to slice. / 切り出す行の範囲。</param>
    /// <param name="columns">The range of columns to slice. / 切り出す列の範囲。</param>
    /// <param name="indices">The range of matrix indices to slice. / 切り出す行列インデックスの範囲。</param>
    public ReadOnlyTensorSpan<TElement> this[Range rows, Range columns, Range indices]
    {
        get
        {
            var (row, rowCount) = rows.GetOffsetAndLength(Rows);
            var (column, columnCount) = columns.GetOffsetAndLength(Columns);
            var (index, count) = indices.GetOffsetAndLength(MatrixCount);
            ref readonly TElement start = ref rowCount == 0 || columnCount == 0 || count == 0 ? ref reference : ref ElementAt(row, column, index);
            return new(in start, rowCount, columnCount, count, RowStride, ColumnStride, MatrixStride);
        }
    }

    /// <summary>
    /// Gets a zero-copy read-only view with each matrix slice transposed: <c>B[i, j, k] = A[j, i, k]</c> (swaps axes 0 and 1 without complex conjugation).
    /// <para>各行列スライスを転置した読み取り専用ビュー（<c>B[i, j, k] = A[j, i, k]</c>、第 0・第 1 軸のみ交換し複素共役は取りません）を取得します。</para>
    /// </summary>
    public ReadOnlyTensorSpan<TElement> T => new(in reference, Columns, Rows, MatrixCount, ColumnStride, RowStride, MatrixStride);

    /// <summary>
    /// Returns a zero-copy read-only <c>min(Lengths[0], Lengths[1]) × Lengths[2]</c> matrix view whose column <c>k</c> is the main diagonal of matrix <c>k</c>.
    /// <para>第 <c>k</c> 列が第 <c>k</c> 行列の主対角成分となる、コピーなしの読み取り専用 <c>min(Lengths[0], Lengths[1]) × Lengths[2]</c> 行列ビューを返します。</para>
    /// </summary>
    /// <returns>A read-only matrix view over the diagonals of all matrix slices. / 各行列の対角成分を列として並べた読み取り専用行列ビュー。</returns>
    /// <remarks>
    /// <para>Useful for batch diagonal operations such as computing all traces via <c>XMath.ColumnSums(A.Diagonals())</c>.</para>
    /// <para><c>XMath.ColumnSums(A.Diagonals())</c> で各行列のトレースを一括計算するなど、対角成分へのバッチ操作に利用できます。</para>
    /// </remarks>
    public ReadOnlyMatrixSpan<TElement> Diagonals() =>
        new(in reference, Math.Min(Rows, Columns), MatrixCount, RowStride + ColumnStride, MatrixStride);

    /// <summary>
    /// Returns a zero-copy read-only <c>Lengths[0] × (Lengths[1] * Lengths[2])</c> matrix view placing all matrices side by side: <c>[A₀ A₁ …]</c>.
    /// <para>各行列を横に並べた <c>[A₀ A₁ …]</c> 形式の、コピーなしの読み取り専用 <c>Lengths[0] × (Lengths[1] * Lengths[2])</c> 行列ビューを返します。</para>
    /// </summary>
    /// <returns>A read-only side-by-side matrix view sharing the underlying storage. / 元の格納領域を共有する横並びの読み取り専用行列ビュー。</returns>
    /// <remarks>
    /// <para>Succeeds only when matrices are laid out contiguously at the column stride (for example, a full <see cref="Tensor{TElement}"/>, a matrix-index slice of one, or <see cref="ReadOnlyMatrixSpan{TElement}.Unsqueeze(int)"/> with axis 1). Enables computing <c>W * [X₀ X₁ …]</c> in a single GEMM call.</para>
    /// <para>行列同士が列ストライド間隔で連続配置されている場合（<see cref="Tensor{TElement}"/> 全体やその行列範囲スライス、<see cref="ReadOnlyMatrixSpan{TElement}.Unsqueeze(int)"/> の軸 1 など）にのみ成功します。<c>W * [X₀ X₁ …]</c> のような計算を 1 回の行列積呼び出しで実行できます。</para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Matrices are not laid out side by side at the column stride (e.g., transposed or broadcasted views), or the combined column count exceeds <see cref="int.MaxValue"/>. / 転置やブロードキャストされたビューなどで行列同士が列ストライド間隔で連続していない、または結合後の列数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public ReadOnlyMatrixSpan<TElement> AsMatrix()
    {
        if (!TryAsMatrix(out var matrix))
        {
            throw new InvalidOperationException((long)Columns * MatrixCount > int.MaxValue
                ? "The side-by-side matrix would have more than Int32.MaxValue columns."
                : "The matrices are not laid out side by side at the column stride.");
        }

        return matrix;
    }

    /// <summary>
    /// Copies the elements of this view into a destination view of the same shape.
    /// <para>このビューの要素を同じ形状の出力先ビューへコピーします。</para>
    /// </summary>
    /// <param name="destination">The destination tensor view. / コピー先のテンソルビュー。</param>
    /// <remarks>
    /// <para>Throws an exception if <paramref name="destination"/> partially overlaps this view.</para>
    /// <para><paramref name="destination"/> がこのビューと部分的に重なっている場合は例外をスローします。</para>
    /// </remarks>
    public void CopyTo(TensorSpan<TElement> destination) => StridedKernels.Copy(this, destination);

    /// <summary>
    /// Copies the elements of this view into a new <see cref="Tensor{TElement}"/>.
    /// <para>このビューの要素を新しい <see cref="Tensor{TElement}"/> へコピーします。</para>
    /// </summary>
    /// <returns>A new tensor containing the copied elements. / 要素をコピーした新しいテンソル。</returns>
    /// <exception cref="OverflowException">The total element count exceeds <see cref="int.MaxValue"/> (for example, on a broadcasted view). / ブロードキャストされたビューなどで総要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public Tensor<TElement> ToTensor()
    {
        var result = Tensor<TElement>.Uninitialized(Rows, Columns, MatrixCount);
        CopyTo(result);
        return result;
    }

    /// <summary>
    /// Copies the elements of this view into a new three-dimensional array preserving <c>[row, column, index]</c> indices.
    /// <para><c>[row, column, index]</c> の添字を維持してこのビューの要素を新しい 3 次元配列へコピーします。</para>
    /// </summary>
    /// <returns>A new three-dimensional array containing the copied elements. / 要素をコピーした新しい 3 次元配列。</returns>
    public TElement[,,] ToArray()
    {
        var result = new TElement[Rows, Columns, MatrixCount];
        for (int k = 0; k < MatrixCount; k++)
        {
            for (int j = 0; j < Columns; j++)
            {
                for (int i = 0; i < Rows; i++)
                {
                    result[i, j, k] = ElementAt(i, j, k);
                }
            }
        }

        return result;
    }

    internal ref readonly TElement ElementAt(int row, int column, int index) =>
        ref Unsafe.Add(ref Unsafe.AsRef(in reference), (nint)row * RowStride + (nint)column * ColumnStride + (nint)index * MatrixStride);

    internal ReadOnlyMatrixSpan<TElement> Matrix(int index) =>
        new(in Rows == 0 || Columns == 0 ? ref reference : ref ElementAt(0, 0, index), Rows, Columns, RowStride, ColumnStride);

    internal bool TryAsMatrix(out ReadOnlyMatrixSpan<TElement> matrix)
    {
        // Reject combined column counts that overflow Int32 (can occur on empty or broadcasted views).
        if ((long)Columns * MatrixCount > int.MaxValue || (Columns > 1 && MatrixCount > 1 && (long)ColumnStride * Columns != MatrixStride))
        {
            matrix = default;
            return false;
        }

        int columnStride = Columns > 1 ? ColumnStride : MatrixStride;
        matrix = new(in reference, Rows, Columns * MatrixCount, RowStride, columnStride);
        return true;
    }

    internal bool TryGetDense(out ReadOnlySpan<TElement> span)
    {
        span = default;
        return TryAsMatrix(out var matrix) && matrix.TryGetDense(out span);
    }

    /// <summary>
    /// Computes the elementwise sum of two tensor views.
    /// <para>2 つのテンソルビューの要素ごとの和を計算します。</para>
    /// </summary>
    /// <param name="a">The left-hand tensor. / 左辺のテンソル。</param>
    /// <param name="b">The right-hand tensor. / 右辺のテンソル。</param>
    /// <returns>A new tensor containing <c>a + b</c>. / <c>a + b</c> を格納した新しいテンソル。</returns>
    public static Tensor<TElement> operator +(ReadOnlyTensorSpan<TElement> a, ReadOnlyTensorSpan<TElement> b) => XMath.Add(a, b);

    /// <summary>
    /// Computes the elementwise difference of two tensor views.
    /// <para>2 つのテンソルビューの要素ごとの差を計算します。</para>
    /// </summary>
    /// <param name="a">The left-hand tensor. / 左辺のテンソル。</param>
    /// <param name="b">The right-hand tensor. / 右辺のテンソル。</param>
    /// <returns>A new tensor containing <c>a - b</c>. / <c>a - b</c> を格納した新しいテンソル。</returns>
    public static Tensor<TElement> operator -(ReadOnlyTensorSpan<TElement> a, ReadOnlyTensorSpan<TElement> b) => XMath.Subtract(a, b);

    /// <summary>
    /// Negates each element of the tensor view.
    /// <para>テンソルビューの各要素の符号を反転します。</para>
    /// </summary>
    /// <param name="a">The input tensor. / 入力テンソル。</param>
    /// <returns>A new tensor containing <c>-a</c>. / <c>-a</c> を格納した新しいテンソル。</returns>
    public static Tensor<TElement> operator -(ReadOnlyTensorSpan<TElement> a) => XMath.Negate(a);

    /// <summary>
    /// Multiplies a tensor view by a scalar.
    /// <para>テンソルビューをスカラー倍します。</para>
    /// </summary>
    /// <param name="scalar">The scalar multiplier. / 掛けるスカラー値。</param>
    /// <param name="a">The input tensor. / 入力テンソル。</param>
    /// <returns>A new tensor containing <c>scalar * a</c>. / <c>scalar * a</c> を格納した新しいテンソル。</returns>
    public static Tensor<TElement> operator *(TElement scalar, ReadOnlyTensorSpan<TElement> a) => XMath.Multiply(a, scalar);

    /// <summary>
    /// Multiplies a tensor view by a scalar.
    /// <para>テンソルビューをスカラー倍します。</para>
    /// </summary>
    /// <param name="a">The input tensor. / 入力テンソル。</param>
    /// <param name="scalar">The scalar multiplier. / 掛けるスカラー値。</param>
    /// <returns>A new tensor containing <c>a * scalar</c>. / <c>a * scalar</c> を格納した新しいテンソル。</returns>
    public static Tensor<TElement> operator *(ReadOnlyTensorSpan<TElement> a, TElement scalar) => XMath.Multiply(a, scalar);

    /// <summary>
    /// Computes the batched matrix product <c>C[i, j, k] = Σ_l A[i, l, k]·B[l, j, k]</c> (without summing over <c>k</c>).
    /// <para>行列ごとのバッチ積 <c>C[i, j, k] = Σ_l A[i, l, k]·B[l, j, k]</c>（<c>k</c> についての和は取りません）を計算します。</para>
    /// </summary>
    /// <param name="a">The left-hand tensor. / 左辺のテンソル。</param>
    /// <param name="b">The right-hand tensor. / 右辺のテンソル。</param>
    /// <returns>A new tensor containing the per-matrix products. / 各行列の積を格納した新しいテンソル。</returns>
    public static Tensor<TElement> operator *(ReadOnlyTensorSpan<TElement> a, ReadOnlyTensorSpan<TElement> b) => XMath.Multiply(a, b);
}

/// <summary>
/// Represents a zero-copy, strided writable view over a rank-3 tensor.
/// <para>ストライド付きの 3 階テンソルを参照する、コピーなしの書き込み可能なビューを表します。</para>
/// </summary>
/// <typeparam name="TElement">The element type of the tensor. / テンソルの要素型。</typeparam>
/// <remarks>
/// <para>Every distinct index tuple maps to a distinct memory location (zero-stride broadcasting is only permitted on <see cref="ReadOnlyTensorSpan{TElement}"/>).</para>
/// <para><see cref="Data"/>, <see cref="Lengths"/>, and <see cref="Strides"/> can be passed directly to the <c>System.Numerics.Tensors.TensorSpan&lt;T&gt;</c> constructor.</para>
/// <para>異なる添字タプルは必ず異なるメモリ位置を指します（ストライド 0 によるブロードキャストは <see cref="ReadOnlyTensorSpan{TElement}"/> のみで許可されます）。</para>
/// <para><see cref="Data"/>、<see cref="Lengths"/>、<see cref="Strides"/> は <c>System.Numerics.Tensors.TensorSpan&lt;T&gt;</c> のコンストラクターにそのまま渡せます。</para>
/// </remarks>
public readonly ref struct TensorSpan<TElement> where TElement : unmanaged, INumberBase<TElement>
{
    private readonly ref TElement reference;
    private readonly NintTriple lengths;
    private readonly NintTriple strides;

    internal TensorSpan(ref TElement reference, int rows, int columns, int count, int rowStride, int columnStride, int matrixStride)
    {
        this.reference = ref reference;
        lengths = NintTriple.Create(rows, columns, count);
        strides = NintTriple.Create(
            StridedLayout.Normalize(rows, rowStride), StridedLayout.Normalize(columns, columnStride), StridedLayout.Normalize(count, matrixStride));
    }

    /// <summary>
    /// Initializes a writable view over a contiguous first-index-fastest span.
    /// <para>先頭軸が最も速く変化する順の連続領域を参照する書き込み可能なビューを初期化します。</para>
    /// </summary>
    /// <param name="data">The contiguous span whose length equals the product of <paramref name="lengths"/>. / 長さが <paramref name="lengths"/> の積と一致する連続領域。</param>
    /// <param name="lengths">The shape <c>[rows, columns, matrixCount]</c>. / 形状 <c>[行数, 列数, 行列数]</c>。</param>
    /// <exception cref="NotSupportedException">The rank is not 3. / 階数が 3 でない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">Any dimension length is negative or exceeds <see cref="int.MaxValue"/>. / いずれかの軸長が負、または <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="OverflowException">The total number of elements exceeds <see cref="int.MaxValue"/>. / 総要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="data"/> does not match the total element count. / <paramref name="data"/> の長さが総要素数と一致しない場合。</exception>
    public TensorSpan(Span<TElement> data, params ReadOnlySpan<nint> lengths)
    {
        var (rows, columns, count) = StridedBatchLayout.ValidateLengths(lengths);
        this = new(data, rows, columns, count);
    }

    internal TensorSpan(Span<TElement> data, int rows, int columns, int count)
    {
        DenseStorage.RequireLength(DenseStorage.Length(rows, columns, count), data.Length);
        this = new(ref MemoryMarshal.GetReference(data), rows, columns, count, 1, rows, DenseStorage.MatrixStride(rows, columns, count));
    }

    /// <summary>
    /// Initializes a writable view over the specified span with explicit shape and strides.
    /// <para>形状とストライドを指定して領域を参照する書き込み可能なビューを初期化します。</para>
    /// </summary>
    /// <param name="data">The backing span starting at the first element and covering through the last element. / 先頭要素から最終要素までを含む格納領域。</param>
    /// <param name="lengths">The shape <c>[rows, columns, matrixCount]</c>. / 形状 <c>[行数, 列数, 行列数]</c>。</param>
    /// <param name="strides">Non-negative element strides forming a non-aliasing nested layout (when sorted by stride, each axis's span is strictly smaller than the next axis's stride). / 添字の重複が生じない入れ子配置をなす非負ストライド（ストライド昇順に並べたとき、各軸の広がりが次の軸のストライド未満であること）。</param>
    /// <exception cref="ArgumentException"><paramref name="lengths"/> and <paramref name="strides"/> have different lengths, <paramref name="data"/> is too short, or the strides do not form a non-aliasing nested layout. / <paramref name="lengths"/> と <paramref name="strides"/> の長さが異なる、<paramref name="data"/> の長さが不足している、またはストライドが入れ子配置でない場合。</exception>
    /// <exception cref="NotSupportedException">The rank is not 3. / 階数が 3 でない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">A length or stride is negative or exceeds <see cref="int.MaxValue"/>. / 長さまたはストライドが負、または <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="OverflowException">The spanned extent exceeds <see cref="int.MaxValue"/>. / 参照範囲の要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public TensorSpan(Span<TElement> data, ReadOnlySpan<nint> lengths, ReadOnlySpan<nint> strides)
    {
        var (rows, columns, count, rowStride, columnStride, matrixStride) = StridedBatchLayout.Validate(lengths, strides, writable: true);
        if (data.Length < StridedBatchLayout.Extent(rows, columns, count, rowStride, columnStride, matrixStride))
        {
            throw new ArgumentException("The data is too short for the lengths and strides.", nameof(data));
        }

        this = new(ref MemoryMarshal.GetReference(data), rows, columns, count, rowStride, columnStride, matrixStride);
    }

    /// <summary>
    /// Creates a writable tensor view from a reference to the first element, shape, and strides.
    /// <para>先頭要素への参照・形状・ストライドから書き込み可能なテンソルビューを作成します。</para>
    /// </summary>
    /// <param name="reference">A reference to the first element. / 先頭要素への参照。</param>
    /// <param name="lengths">The shape <c>[rows, columns, matrixCount]</c>. / 形状 <c>[行数, 列数, 行列数]</c>。</param>
    /// <param name="strides">Non-negative element strides forming a non-aliasing nested layout. / 入れ子配置をなす各軸の非負ストライド。</param>
    /// <returns>A writable tensor view referencing the specified memory layout. / 指定したレイアウトを参照する書き込み可能なテンソルビュー。</returns>
    /// <remarks>
    /// <para>Compatible with <c>System.Numerics.Tensors</c> via <c>Create(ref t.GetPinnableReference(), t.Lengths, t.Strides)</c>. Bounds of the backing memory are not checked, matching <see cref="MemoryMarshal.CreateSpan"/>.</para>
    /// <para><c>System.Numerics.Tensors</c> からは <c>Create(ref t.GetPinnableReference(), t.Lengths, t.Strides)</c> で変換できます。<see cref="MemoryMarshal.CreateSpan"/> と同様に、メモリ範囲の境界検査は行いません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="lengths"/> and <paramref name="strides"/> have different lengths, or the strides do not form a non-aliasing nested layout. / <paramref name="lengths"/> と <paramref name="strides"/> の長さが異なる、またはストライドが入れ子配置でない場合。</exception>
    /// <exception cref="NotSupportedException">The rank is not 3. / 階数が 3 でない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">A length or stride is negative or exceeds <see cref="int.MaxValue"/>. / 長さまたはストライドが負、または <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="OverflowException">The spanned extent exceeds <see cref="int.MaxValue"/>. / 参照範囲の要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static TensorSpan<TElement> Create(ref TElement reference, ReadOnlySpan<nint> lengths, ReadOnlySpan<nint> strides)
    {
        var (rows, columns, count, rowStride, columnStride, matrixStride) = StridedBatchLayout.Validate(lengths, strides, writable: true);
        return new(ref reference, rows, columns, count, rowStride, columnStride, matrixStride);
    }

    /// <summary>
    /// Gets the rank (number of dimensions) of the view, which is currently always 3.
    /// <para>ビューの階数（次元数、現在は常に 3）を取得します。</para>
    /// </summary>
    public int Rank => StridedBatchLayout.SupportedRank;

    /// <summary>
    /// Gets the total number of elements (the product of <see cref="Lengths"/>).
    /// <para>総要素数（<see cref="Lengths"/> の積）を取得します。</para>
    /// </summary>
    public nint FlattenedLength => (nint)Rows * Columns * MatrixCount;

    /// <summary>
    /// Gets a value indicating whether the view contains no elements.
    /// <para>ビューが空かどうかを示す値を取得します。</para>
    /// </summary>
    public bool IsEmpty => Rows == 0 || Columns == 0 || MatrixCount == 0;

    /// <summary>
    /// Gets the shape <c>[rows, columns, matrixCount]</c> of the view.
    /// <para>ビューの形状 <c>[行数, 列数, 行列数]</c> を取得します。</para>
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

    internal int Rows => (int)lengths[0];
    internal int Columns => (int)lengths[1];
    internal int MatrixCount => (int)lengths[2];

    /// <summary>
    /// Gets a writable reference to element <c>(row, column)</c> of matrix <paramref name="index"/>.
    /// <para>第 <paramref name="index"/> 行列の要素 <c>(row, column)</c> への書き込み可能な参照を取得します。</para>
    /// </summary>
    /// <param name="row">The zero-based row index. / 0 から始まる行インデックス。</param>
    /// <param name="column">The zero-based column index. / 0 から始まる列インデックス。</param>
    /// <param name="index">The zero-based matrix index. / 0 から始まる行列インデックス。</param>
    /// <exception cref="ArgumentOutOfRangeException">Any index is out of range. / いずれかのインデックスが範囲外の場合。</exception>
    public ref TElement this[int row, int column, int index] => ref Unsafe.AsRef(in AsReadOnly()[row, column, index]);

    /// <summary>
    /// Returns a zero-copy writable view of matrix <paramref name="index"/> (<c>Lengths[0] × Lengths[1]</c>).
    /// <para>第 <paramref name="index"/> 行列（<c>Lengths[0] × Lengths[1]</c>）を参照する書き込み可能なビューを返します。</para>
    /// </summary>
    /// <param name="index">The zero-based matrix index. / 0 から始まる行列インデックス。</param>
    /// <returns>A writable matrix view sharing the underlying storage. / 元の格納領域を共有する書き込み可能な行列ビュー。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is out of range. / <paramref name="index"/> が範囲外の場合。</exception>
    public MatrixSpan<TElement> GetMatrix(int index) => MatrixSpan<TElement>.Writable(AsReadOnly().GetMatrix(index));

    /// <summary>
    /// Returns a zero-copy writable view of column <paramref name="column"/> of matrix <paramref name="index"/>.
    /// <para>第 <paramref name="index"/> 行列の第 <paramref name="column"/> 列を参照する書き込み可能なビューを返します。</para>
    /// </summary>
    /// <param name="column">The zero-based column index. / 0 から始まる列インデックス。</param>
    /// <param name="index">The zero-based matrix index. / 0 から始まる行列インデックス。</param>
    /// <returns>A writable vector view equivalent to <c>GetMatrix(index).Column(column)</c>. / <c>GetMatrix(index).Column(column)</c> と同等の書き込み可能なベクトルビュー。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="column"/> or <paramref name="index"/> is out of range. / <paramref name="column"/> または <paramref name="index"/> が範囲外の場合。</exception>
    public VectorSpan<TElement> GetVector(int column, int index) => VectorSpan<TElement>.Writable(AsReadOnly().GetVector(column, index));

    /// <summary>
    /// Gets an indexed, allocation-free enumerable view over all matrix slices.
    /// <para>すべての行列スライスにインデックスアクセスおよび列挙できる書き込み可能なビューを取得します。</para>
    /// </summary>
    public TensorMatrices<TElement> Matrices => new(this);

    /// <summary>
    /// Gets a zero-copy writable view over the specified subrange of rows, columns, and matrix indices.
    /// <para>指定した行・列・行列インデックスの範囲を参照する書き込み可能なビューを取得します。</para>
    /// </summary>
    /// <param name="rows">The range of rows to slice. / 切り出す行の範囲。</param>
    /// <param name="columns">The range of columns to slice. / 切り出す列の範囲。</param>
    /// <param name="indices">The range of matrix indices to slice. / 切り出す行列インデックスの範囲。</param>
    public TensorSpan<TElement> this[Range rows, Range columns, Range indices] => Writable(AsReadOnly()[rows, columns, indices]);

    /// <summary>
    /// Gets a zero-copy writable view with each matrix slice transposed: <c>B[i, j, k] = A[j, i, k]</c> (swaps axes 0 and 1 without complex conjugation).
    /// <para>各行列スライスを転置した書き込み可能なビュー（<c>B[i, j, k] = A[j, i, k]</c>、第 0・第 1 軸のみ交換し複素共役は取りません）を取得します。</para>
    /// </summary>
    public TensorSpan<TElement> T => Writable(AsReadOnly().T);

    /// <summary>
    /// Returns a zero-copy writable <c>min(Lengths[0], Lengths[1]) × Lengths[2]</c> matrix view whose column <c>k</c> is the main diagonal of matrix <c>k</c>.
    /// <para>第 <c>k</c> 列が第 <c>k</c> 行列の主対角成分となる、コピーなしの書き込み可能な <c>min(Lengths[0], Lengths[1]) × Lengths[2]</c> 行列ビューを返します。</para>
    /// </summary>
    /// <returns>A writable matrix view over the diagonals of all matrix slices. / 各行列の対角成分を列として並べた書き込み可能な行列ビュー。</returns>
    /// <exception cref="InvalidOperationException">The resulting diagonal strides would not form a non-aliasing writable layout. / 対角成分のストライド構成が一意にアドレス付けできる書き込み可能なレイアウトにならない場合。</exception>
    public MatrixSpan<TElement> Diagonals()
    {
        var view = AsReadOnly().Diagonals();
        if (!StridedBatchLayout.IsNested(view.Rows, view.Columns, 1, view.RowStride, view.ColumnStride, 0))
        {
            throw new InvalidOperationException("The diagonals of this layout cannot form a writable view.");
        }

        return MatrixSpan<TElement>.Writable(view);
    }

    /// <summary>
    /// Returns a zero-copy writable <c>Lengths[0] × (Lengths[1] * Lengths[2])</c> matrix view placing all matrices side by side: <c>[A₀ A₁ …]</c>.
    /// <para>各行列を横に並べた <c>[A₀ A₁ …]</c> 形式の、コピーなしの書き込み可能な <c>Lengths[0] × (Lengths[1] * Lengths[2])</c> 行列ビューを返します。</para>
    /// </summary>
    /// <returns>A writable side-by-side matrix view sharing the underlying storage. / 元の格納領域を共有する横並びの書き込み可能な行列ビュー。</returns>
    /// <exception cref="InvalidOperationException">Matrices are not laid out side by side at the column stride, or the combined column count exceeds <see cref="int.MaxValue"/>. / 行列同士が列ストライド間隔で連続していない、または結合後の列数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public MatrixSpan<TElement> AsMatrix() => MatrixSpan<TElement>.Writable(AsReadOnly().AsMatrix());

    /// <summary>
    /// Returns a read-only view over the same elements.
    /// <para>同じ要素を参照する読み取り専用ビューを返します。</para>
    /// </summary>
    /// <returns>A read-only tensor view. / 読み取り専用のテンソルビュー。</returns>
    public ReadOnlyTensorSpan<TElement> AsReadOnly() =>
        new(in reference, Rows, Columns, MatrixCount, (int)strides[0], (int)strides[1], (int)strides[2]);

    /// <summary>
    /// Fills all elements of the tensor view with the specified value.
    /// <para>テンソルビューのすべての要素を指定した値で埋めます。</para>
    /// </summary>
    /// <param name="value">The value to assign to each element. / 各要素に設定する値。</param>
    public void Fill(TElement value)
    {
        var view = AsReadOnly();
        for (int k = 0; k < MatrixCount; k++)
        {
            MatrixSpan<TElement>.Writable(view.Matrix(k)).Fill(value);
        }
    }

    /// <summary>
    /// Copies the elements of this view into a destination view of the same shape.
    /// <para>このビューの要素を同じ形状の出力先ビューへコピーします。</para>
    /// </summary>
    /// <param name="destination">The destination tensor view. / コピー先のテンソルビュー。</param>
    /// <remarks>
    /// <para>Throws an exception if <paramref name="destination"/> partially overlaps this view.</para>
    /// <para><paramref name="destination"/> がこのビューと部分的に重なっている場合は例外をスローします。</para>
    /// </remarks>
    public void CopyTo(TensorSpan<TElement> destination) => AsReadOnly().CopyTo(destination);

    /// <summary>
    /// Copies the elements of this view into a new <see cref="Tensor{TElement}"/>.
    /// <para>このビューの要素を新しい <see cref="Tensor{TElement}"/> へコピーします。</para>
    /// </summary>
    /// <returns>A new tensor containing the copied elements. / 要素をコピーした新しいテンソル。</returns>
    public Tensor<TElement> ToTensor() => AsReadOnly().ToTensor();

    /// <summary>
    /// Copies the elements of this view into a new three-dimensional array preserving <c>[row, column, index]</c> indices.
    /// <para><c>[row, column, index]</c> の添字を維持してこのビューの要素を新しい 3 次元配列へコピーします。</para>
    /// </summary>
    /// <returns>A new three-dimensional array containing the copied elements. / 要素をコピーした新しい 3 次元配列。</returns>
    public TElement[,,] ToArray() => AsReadOnly().ToArray();

    internal static TensorSpan<TElement> Writable(ReadOnlyTensorSpan<TElement> view) =>
        new(ref Unsafe.AsRef(in view.Reference), view.Rows, view.Columns, view.MatrixCount, view.RowStride, view.ColumnStride, view.MatrixStride);

    /// <summary>
    /// Implicitly converts a writable <see cref="TensorSpan{TElement}"/> to a <see cref="ReadOnlyTensorSpan{TElement}"/>.
    /// <para>書き込み可能な <see cref="TensorSpan{TElement}"/> を <see cref="ReadOnlyTensorSpan{TElement}"/> へ暗黙的に変換します。</para>
    /// </summary>
    /// <param name="view">The writable view to convert. / 変換元の書き込み可能なビュー。</param>
    /// <returns>A read-only view referencing the same elements. / 同じ要素を参照する読み取り専用ビュー。</returns>
    public static implicit operator ReadOnlyTensorSpan<TElement>(TensorSpan<TElement> view) => view.AsReadOnly();

    /// <summary>
    /// Computes the elementwise sum of two tensor views.
    /// <para>2 つのテンソルビューの要素ごとの和を計算します。</para>
    /// </summary>
    /// <param name="a">The left-hand tensor. / 左辺のテンソル。</param>
    /// <param name="b">The right-hand tensor. / 右辺のテンソル。</param>
    /// <returns>A new tensor containing <c>a + b</c>. / <c>a + b</c> を格納した新しいテンソル。</returns>
    public static Tensor<TElement> operator +(TensorSpan<TElement> a, TensorSpan<TElement> b) => XMath.Add(a, b);

    /// <summary>
    /// Computes the elementwise difference of two tensor views.
    /// <para>2 つのテンソルビューの要素ごとの差を計算します。</para>
    /// </summary>
    /// <param name="a">The left-hand tensor. / 左辺のテンソル。</param>
    /// <param name="b">The right-hand tensor. / 右辺のテンソル。</param>
    /// <returns>A new tensor containing <c>a - b</c>. / <c>a - b</c> を格納した新しいテンソル。</returns>
    public static Tensor<TElement> operator -(TensorSpan<TElement> a, TensorSpan<TElement> b) => XMath.Subtract(a, b);

    /// <summary>
    /// Negates each element of the tensor view.
    /// <para>テンソルビューの各要素の符号を反転します。</para>
    /// </summary>
    /// <param name="a">The input tensor. / 入力テンソル。</param>
    /// <returns>A new tensor containing <c>-a</c>. / <c>-a</c> を格納した新しいテンソル。</returns>
    public static Tensor<TElement> operator -(TensorSpan<TElement> a) => XMath.Negate(a);

    /// <summary>
    /// Multiplies a tensor view by a scalar.
    /// <para>テンソルビューをスカラー倍します。</para>
    /// </summary>
    /// <param name="scalar">The scalar multiplier. / 掛けるスカラー値。</param>
    /// <param name="a">The input tensor. / 入力テンソル。</param>
    /// <returns>A new tensor containing <c>scalar * a</c>. / <c>scalar * a</c> を格納した新しいテンソル。</returns>
    public static Tensor<TElement> operator *(TElement scalar, TensorSpan<TElement> a) => XMath.Multiply(a, scalar);

    /// <summary>
    /// Multiplies a tensor view by a scalar.
    /// <para>テンソルビューをスカラー倍します。</para>
    /// </summary>
    /// <param name="a">The input tensor. / 入力テンソル。</param>
    /// <param name="scalar">The scalar multiplier. / 掛けるスカラー値。</param>
    /// <returns>A new tensor containing <c>a * scalar</c>. / <c>a * scalar</c> を格納した新しいテンソル。</returns>
    public static Tensor<TElement> operator *(TensorSpan<TElement> a, TElement scalar) => XMath.Multiply(a, scalar);

    /// <summary>
    /// Computes the batched matrix product <c>C[i, j, k] = Σ_l A[i, l, k]·B[l, j, k]</c> (without summing over <c>k</c>).
    /// <para>行列ごとのバッチ積 <c>C[i, j, k] = Σ_l A[i, l, k]·B[l, j, k]</c>（<c>k</c> についての和は取りません）を計算します。</para>
    /// </summary>
    /// <param name="a">The left-hand tensor. / 左辺のテンソル。</param>
    /// <param name="b">The right-hand tensor. / 右辺のテンソル。</param>
    /// <returns>A new tensor containing the per-matrix products. / 各行列の積を格納した新しいテンソル。</returns>
    public static Tensor<TElement> operator *(TensorSpan<TElement> a, TensorSpan<TElement> b) => XMath.Multiply(a, b);
}
