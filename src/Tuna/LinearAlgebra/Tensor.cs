using System.Globalization;
using System.Numerics;

namespace Tuna.LinearAlgebra;

/// <summary>
/// Represents a dense rank-3 multidimensional numerical array storing a batch of column-major matrices.
/// <para>列優先行列のバッチを格納する 3 階の密多次元配列（テンソル）を表します。</para>
/// </summary>
/// <typeparam name="TElement">The element type of the tensor. / テンソルの要素型。</typeparam>
/// <remarks>
/// <para>Currently only rank 3 (<c>[rows, columns, matrixCount]</c>) is supported; passing other ranks throws <see cref="NotSupportedException"/> (use <see cref="Vector{T}"/> for rank 1 and <see cref="Matrix{TElement}"/> for rank 2).</para>
/// <para>Elements are stored in first-index-fastest order at <c>i + j * Lengths[0] + k * Lengths[0] * Lengths[1]</c>, so each slice from <see cref="GetMatrix(int)"/> has the same contiguous column-major layout as <see cref="Matrix{TElement}"/>. The <c>*</c> operator computes batched matrix multiplication <c>C[k] = A[k] * B[k]</c>, and compound assignment operators mutate this tensor in place.</para>
/// <para>現在は 3 階（<c>[行数, 列数, 行列数]</c>）のみサポートしており、それ以外の階数は <see cref="NotSupportedException"/> をスローします（1 階には <see cref="Vector{T}"/>、2 階には <see cref="Matrix{TElement}"/> を使用してください）。</para>
/// <para>要素は先頭軸が最も速く変化する順（<c>i + j * Lengths[0] + k * Lengths[0] * Lengths[1]</c>）で格納されるため、<see cref="GetMatrix(int)"/> で得る各スライスは <see cref="Matrix{TElement}"/> と同じ列優先連続レイアウトになります。<c>*</c> 演算子は行列ごとのバッチ積 <c>C[k] = A[k] * B[k]</c> を計算し、複合代入演算子はこのテンソルをインプレースで更新します。</para>
/// </remarks>
public sealed class Tensor<TElement> where TElement : unmanaged, INumberBase<TElement>
{
    private readonly TElement[] data;
    private readonly NintTriple lengths;
    private readonly NintTriple strides;

    /// <summary>
    /// Initializes a new zero-filled tensor with the specified shape.
    /// <para>指定した形状の零テンソルを初期化します。</para>
    /// </summary>
    /// <param name="lengths">The shape of the tensor (<c>[rows, columns, matrixCount]</c> for rank 3). / テンソルの形状（3 階の場合は <c>[行数, 列数, 行列数]</c>）。</param>
    /// <exception cref="NotSupportedException">The rank is not 3. / 階数が 3 でない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">Any dimension length is negative or exceeds <see cref="int.MaxValue"/>. / いずれかの軸長が負、または <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="OverflowException">The total number of elements exceeds <see cref="int.MaxValue"/>. / 総要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public Tensor(params ReadOnlySpan<nint> lengths) : this(StridedBatchLayout.ValidateLengths(lengths))
    {
    }

    private Tensor((int Rows, int Columns, int MatrixCount) shape) : this(
        shape.Rows, shape.Columns, shape.MatrixCount, DenseStorage.Allocate<TElement>(DenseStorage.Length(shape.Rows, shape.Columns, shape.MatrixCount)))
    {
    }

    /// <summary>
    /// Initializes a new rank-3 tensor by copying the elements of a three-dimensional array indexed as <c>[row, column, index]</c>.
    /// <para><c>[row, column, index]</c> 順の 3 次元配列の要素をコピーして新しい 3 階テンソルを初期化します。</para>
    /// </summary>
    /// <param name="values">The source three-dimensional array. / コピー元の 3 次元配列。</param>
    /// <exception cref="ArgumentNullException"><paramref name="values"/> is <see langword="null"/>. / <paramref name="values"/> が <see langword="null"/> の場合。</exception>
    public Tensor(TElement[,,] values) : this(
        (values ?? throw new ArgumentNullException(nameof(values))).GetLength(0), values.GetLength(1), values.GetLength(2),
        DenseStorage.Allocate<TElement>(DenseStorage.Length(values.GetLength(0), values.GetLength(1), values.GetLength(2))))
    {
        for (int k = 0; k < MatrixCount; k++)
        {
            for (int j = 0; j < Columns; j++)
            {
                for (int i = 0; i < Rows; i++)
                {
                    data[Offset(i, j, k)] = values[i, j, k];
                }
            }
        }
    }

    private Tensor(int rows, int columns, int count, TElement[] storage)
    {
        DenseStorage.RequireLength(DenseStorage.Length(rows, columns, count), storage.Length);
        lengths = NintTriple.Create(rows, columns, count);
        strides = NintTriple.Create(
            StridedLayout.Normalize(rows, 1),
            StridedLayout.Normalize(columns, rows),
            DenseStorage.MatrixStride(rows, columns, count));
        data = storage;
    }

    /// <summary>
    /// Gets the rank (number of dimensions) of the tensor, which is currently always 3.
    /// <para>テンソルの階数（次元数、現在は常に 3）を取得します。</para>
    /// </summary>
    public int Rank => StridedBatchLayout.SupportedRank;

    /// <summary>
    /// Gets the shape <c>[rows, columns, matrixCount]</c> of the tensor.
    /// <para>テンソルの形状 <c>[行数, 列数, 行列数]</c> を取得します。</para>
    /// </summary>
    public ReadOnlySpan<nint> Lengths => lengths;

    /// <summary>
    /// Gets the element strides along each axis (0 for axes of length 0 or 1).
    /// <para>各軸の要素単位のストライドを取得します（長さが 0 または 1 の軸は 0）。</para>
    /// </summary>
    public ReadOnlySpan<nint> Strides => strides;

    /// <summary>
    /// Gets the total number of elements (the product of <see cref="Lengths"/>).
    /// <para>総要素数（<see cref="Lengths"/> の積）を取得します。</para>
    /// </summary>
    public nint FlattenedLength => data.Length;

    /// <summary>
    /// Gets a value indicating whether the tensor contains no elements.
    /// <para>テンソルが空かどうかを示す値を取得します。</para>
    /// </summary>
    public bool IsEmpty => data.Length == 0;

    internal int Rows => (int)lengths[0];
    internal int Columns => (int)lengths[1];
    internal int MatrixCount => (int)lengths[2];

    /// <summary>
    /// Gets or sets element <c>(row, column)</c> of matrix <paramref name="index"/>.
    /// <para>第 <paramref name="index"/> 行列の要素 <c>(row, column)</c> を取得または設定します。</para>
    /// </summary>
    /// <param name="row">The zero-based row index. / 0 から始まる行インデックス。</param>
    /// <param name="column">The zero-based column index. / 0 から始まる列インデックス。</param>
    /// <param name="index">The zero-based matrix index. / 0 から始まる行列インデックス。</param>
    /// <exception cref="ArgumentOutOfRangeException">Any index is out of range. / いずれかのインデックスが範囲外の場合。</exception>
    public TElement this[int row, int column, int index]
    {
        get
        {
            CheckIndex(row, column, index);
            return data[Offset(row, column, index)];
        }
        set
        {
            CheckIndex(row, column, index);
            data[Offset(row, column, index)] = value;
        }
    }

    /// <summary>
    /// Gets a zero-copy writable view over the specified subrange of rows, columns, and matrix indices.
    /// <para>指定した行・列・行列インデックスの範囲を参照する、コピーなしの書き込み可能なビューを取得します。</para>
    /// </summary>
    /// <param name="rows">The range of rows to slice. / 切り出す行の範囲。</param>
    /// <param name="columns">The range of columns to slice. / 切り出す列の範囲。</param>
    /// <param name="indices">The range of matrix indices to slice. / 切り出す行列インデックスの範囲。</param>
    public TensorSpan<TElement> this[Range rows, Range columns, Range indices] => AsTensorSpan()[rows, columns, indices];

    /// <summary>
    /// Returns a zero-copy contiguous writable view of the matrix at <paramref name="index"/>.
    /// <para>第 <paramref name="index"/> 行列（<c>Lengths[0] × Lengths[1]</c> の連続領域）を参照する書き込み可能なビューを返します。</para>
    /// </summary>
    /// <param name="index">The zero-based matrix index. / 0 から始まる行列インデックス。</param>
    /// <returns>A writable matrix view sharing this tensor's storage. / このテンソルと格納領域を共有する書き込み可能な行列ビュー。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is out of range. / <paramref name="index"/> が範囲外の場合。</exception>
    public MatrixSpan<TElement> GetMatrix(int index) => AsTensorSpan().GetMatrix(index);

    /// <summary>
    /// Returns a zero-copy contiguous writable view of column <paramref name="column"/> of matrix <paramref name="index"/>.
    /// <para>第 <paramref name="index"/> 行列の第 <paramref name="column"/> 列（長さ <c>Lengths[0]</c> の連続領域）を参照する書き込み可能なビューを返します。</para>
    /// </summary>
    /// <param name="column">The zero-based column index. / 0 から始まる列インデックス。</param>
    /// <param name="index">The zero-based matrix index. / 0 から始まる行列インデックス。</param>
    /// <returns>A writable vector view equivalent to <c>GetMatrix(index).Column(column)</c>. / <c>GetMatrix(index).Column(column)</c> と同等の書き込み可能なベクトルビュー。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="column"/> or <paramref name="index"/> is out of range. / <paramref name="column"/> または <paramref name="index"/> が範囲外の場合。</exception>
    public VectorSpan<TElement> GetVector(int column, int index) => AsTensorSpan().GetVector(column, index);

    /// <summary>
    /// Gets an indexed, allocation-free enumerable view over all matrix slices in this tensor.
    /// <para>このテンソル内のすべての行列スライスにインデックスアクセスおよび列挙できるビューを取得します。</para>
    /// </summary>
    public TensorMatrices<TElement> Matrices => AsTensorSpan().Matrices;

    /// <summary>
    /// Creates a new tensor by copying elements stored in first-index-fastest (column-major per matrix) order.
    /// <para>先頭軸が最も速く変化する順（各行列が列優先で番号順に並んだ順）の要素をコピーして新しいテンソルを作成します。</para>
    /// </summary>
    /// <param name="lengths">The shape <c>[rows, columns, matrixCount]</c>. / 形状 <c>[行数, 列数, 行列数]</c>。</param>
    /// <param name="values">The source elements whose length equals the product of <paramref name="lengths"/>. / 長さが <paramref name="lengths"/> の積と一致する入力要素。</param>
    /// <returns>A new tensor containing the copied elements. / 要素をコピーした新しいテンソル。</returns>
    /// <exception cref="NotSupportedException">The rank is not 3. / 階数が 3 でない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">Any dimension length is negative or exceeds <see cref="int.MaxValue"/>. / いずれかの軸長が負、または <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="OverflowException">The total number of elements exceeds <see cref="int.MaxValue"/>. / 総要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="values"/> does not match the total element count. / <paramref name="values"/> の長さが総要素数と一致しない場合。</exception>
    public static Tensor<TElement> FromColumnMajor(ReadOnlySpan<nint> lengths, ReadOnlySpan<TElement> values)
    {
        var (rows, columns, count) = StridedBatchLayout.ValidateLengths(lengths);
        DenseStorage.RequireLength(DenseStorage.Length(rows, columns, count), values.Length);
        return new(rows, columns, count, values.ToArray());
    }

    /// <summary>
    /// Creates a new rank-3 tensor by stacking matrices of identical shape along the third axis.
    /// <para>同じ形状の行列を第 3 軸に沿って順にコピーし、新しい 3 階テンソルを作成します。</para>
    /// </summary>
    /// <param name="matrices">One or more matrices with identical dimensions. / 同じ形状を持つ 1 個以上の行列。</param>
    /// <returns>A new tensor where matrix <c>k</c> is a copy of <c>matrices[k]</c>. / 第 <c>k</c> 行列が <c>matrices[k]</c> のコピーである新しいテンソル。</returns>
    /// <exception cref="ArgumentNullException">Any element in <paramref name="matrices"/> is <see langword="null"/>. / <paramref name="matrices"/> に <see langword="null"/> が含まれる場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="matrices"/> is empty or the matrices do not all have the same shape. / <paramref name="matrices"/> が空、または行列の形状が一致しない場合。</exception>
    /// <exception cref="OverflowException">The total number of elements exceeds <see cref="int.MaxValue"/>. / 総要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static Tensor<TElement> FromMatrices(params ReadOnlySpan<Matrix<TElement>> matrices)
    {
        if (matrices.IsEmpty)
        {
            throw new ArgumentException("At least one matrix is required to determine the shape.", nameof(matrices));
        }

        ArgumentNullException.ThrowIfNull(matrices[0], nameof(matrices));
        int rows = matrices[0].Rows;
        int columns = matrices[0].Columns;
        var result = Uninitialized(rows, columns, matrices.Length);
        for (int k = 0; k < matrices.Length; k++)
        {
            var m = matrices[k] ?? throw new ArgumentNullException(nameof(matrices));
            if (m.Rows != rows || m.Columns != columns)
            {
                throw new ArgumentException("All matrices must have the same shape.", nameof(matrices));
            }

            m.Data.CopyTo(result.data, k * rows * columns);
        }

        return result;
    }

    /// <summary>
    /// Creates a rank-3 tensor consisting of <paramref name="count"/> identity matrices of order <paramref name="size"/>.
    /// <para><paramref name="size"/> 次の単位行列を <paramref name="count"/> 個並べた 3 階テンソルを作成します。</para>
    /// </summary>
    /// <param name="size">The order of each identity matrix. / 各単位行列の次数。</param>
    /// <param name="count">The number of identity matrices. / 単位行列の個数。</param>
    /// <returns>A new <c>[size, size, count]</c> batch identity tensor. / <c>[size, size, count]</c> の新しいバッチ単位テンソル。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="size"/> or <paramref name="count"/> is negative. / <paramref name="size"/> または <paramref name="count"/> が負の場合。</exception>
    /// <exception cref="OverflowException"><c>size * size * count</c> exceeds <see cref="int.MaxValue"/>. / 要素数 <c>size² * count</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static Tensor<TElement> Identity(int size, int count)
    {
        var result = new Tensor<TElement>(size, size, count);
        result.Diagonals().Fill(TElement.One);
        return result;
    }

    /// <summary>
    /// Creates a new tensor whose elements are computed from their multi-axis indices using a stateless operator.
    /// <para>状態を持たない演算子を用いて全軸の添字から各要素を計算した新しいテンソルを作成します。</para>
    /// </summary>
    /// <typeparam name="TOperator">The stateless tensor index operator type. / 状態を持たないテンソル添字演算子の型。</typeparam>
    /// <param name="lengths">The shape of the tensor (currently rank 3). / テンソルの形状（現在は 3 階）。</param>
    /// <returns>A new tensor populated by <typeparamref name="TOperator"/>. / <typeparamref name="TOperator"/> で生成された新しいテンソル。</returns>
    /// <exception cref="NotSupportedException">The rank is not 3. / 階数が 3 でない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">Any dimension length is negative or exceeds <see cref="int.MaxValue"/>. / いずれかの軸長が負、または <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="OverflowException">The total number of elements exceeds <see cref="int.MaxValue"/>. / 総要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static Tensor<TElement> Generate<TOperator>(ReadOnlySpan<nint> lengths)
        where TOperator : ITensorIndexOperator<TElement> =>
        Generate(lengths, default(StaticTensorIndexOperator<TElement, TOperator>));

    /// <summary>
    /// Creates a new tensor whose elements are computed from their multi-axis indices using a stateful operator.
    /// <para>状態を持つ演算子を用いて全軸の添字から各要素を計算した新しいテンソルを作成します。</para>
    /// </summary>
    /// <typeparam name="TOperator">The stateful tensor index operator type. / 状態を持つテンソル添字演算子の型。</typeparam>
    /// <param name="lengths">The shape of the tensor (currently rank 3). / テンソルの形状（現在は 3 階）。</param>
    /// <param name="op">The operator evaluated for each index tuple (the index span is valid only during the call). / 各添字タプルに対して評価する演算子（添字スパンは呼び出し中のみ有効）。</param>
    /// <returns>A new tensor populated by <paramref name="op"/>. / <paramref name="op"/> で生成された新しいテンソル。</returns>
    /// <exception cref="NotSupportedException">The rank is not 3. / 階数が 3 でない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">Any dimension length is negative or exceeds <see cref="int.MaxValue"/>. / いずれかの軸長が負、または <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="OverflowException">The total number of elements exceeds <see cref="int.MaxValue"/>. / 総要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static Tensor<TElement> Generate<TOperator>(ReadOnlySpan<nint> lengths, TOperator op)
        where TOperator : IStatefulTensorIndexOperator<TElement>
    {
        var (rows, columns, count) = StridedBatchLayout.ValidateLengths(lengths);
        var result = Uninitialized(rows, columns, count);
        XMath.Generate(op, result.AsTensorSpan());
        return result;
    }

    /// <summary>
    /// Creates a new tensor whose elements are computed from their multi-axis indices using a delegate.
    /// <para>デリゲートを用いて全軸の添字から各要素を計算した新しいテンソルを作成します。</para>
    /// </summary>
    /// <param name="lengths">The shape of the tensor (currently rank 3). / テンソルの形状（現在は 3 階）。</param>
    /// <param name="op">The delegate evaluated for each index tuple (the index span is valid only during the call). / 各添字タプルに対して評価するデリゲート（添字スパンは呼び出し中のみ有効）。</param>
    /// <returns>A new tensor populated by <paramref name="op"/>. / <paramref name="op"/> で生成された新しいテンソル。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="op"/> is <see langword="null"/>. / <paramref name="op"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="NotSupportedException">The rank is not 3. / 階数が 3 でない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">Any dimension length is negative or exceeds <see cref="int.MaxValue"/>. / いずれかの軸長が負、または <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="OverflowException">The total number of elements exceeds <see cref="int.MaxValue"/>. / 総要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static Tensor<TElement> Generate(ReadOnlySpan<nint> lengths, Func<ReadOnlySpan<int>, TElement> op) =>
        Generate(lengths, new DelegateTensorIndexOperator<TElement>(op));

    /// <summary>
    /// Returns a <see cref="Span{T}"/> sharing the underlying contiguous first-index-fastest storage.
    /// <para>先頭軸が最も速く変化する順の内部連続格納領域を共有する <see cref="Span{T}"/> を返します。</para>
    /// </summary>
    /// <returns>A span over the tensor's backing storage. / テンソルの格納領域を参照するスパン。</returns>
    public Span<TElement> AsSpan() => data;

    /// <summary>
    /// Returns a writable <see cref="TensorSpan{TElement}"/> view over the entire tensor.
    /// <para>テンソル全体を参照する書き込み可能な <see cref="TensorSpan{TElement}"/> ビューを返します。</para>
    /// </summary>
    /// <returns>A writable view sharing this tensor's storage. / 格納領域を共有する書き込み可能なテンソルビュー。</returns>
    public TensorSpan<TElement> AsTensorSpan() => new(data, Rows, Columns, MatrixCount);

    /// <summary>
    /// Gets a zero-copy writable view with each matrix slice transposed: <c>B[i, j, k] = A[j, i, k]</c> (swaps axes 0 and 1 without complex conjugation).
    /// <para>各行列スライスを転置したコピーなしの書き込み可能なビュー（<c>B[i, j, k] = A[j, i, k]</c>、第 0・第 1 軸のみ交換し複素共役は取りません）を取得します。</para>
    /// </summary>
    /// <remarks>
    /// <para>To materialize a transposed copy, use <see cref="XMath.Transpose{T}(Tensor{T})"/>.</para>
    /// <para>転置した新しいコピーを作成する場合は <see cref="XMath.Transpose{T}(Tensor{T})"/> を使用してください。</para>
    /// </remarks>
    public TensorSpan<TElement> T => AsTensorSpan().T;

    /// <summary>
    /// Returns a zero-copy writable <c>min(Lengths[0], Lengths[1]) × Lengths[2]</c> matrix view whose column <c>k</c> is the main diagonal of matrix <c>k</c>.
    /// <para>第 <c>k</c> 列が第 <c>k</c> 行列の主対角成分となる、コピーなしの書き込み可能な <c>min(Lengths[0], Lengths[1]) × Lengths[2]</c> 行列ビューを返します。</para>
    /// </summary>
    /// <returns>A writable matrix view over the diagonals of all matrix slices. / 各行列の対角成分を列として並べた書き込み可能な行列ビュー。</returns>
    public MatrixSpan<TElement> Diagonals() => AsTensorSpan().Diagonals();

    /// <summary>
    /// Returns a zero-copy writable <c>Lengths[0] × (Lengths[1] * Lengths[2])</c> matrix view placing all matrices side by side: <c>[A₀ A₁ …]</c>.
    /// <para>各行列を横に並べた <c>[A₀ A₁ …]</c> 形式の、コピーなしの書き込み可能な <c>Lengths[0] × (Lengths[1] * Lengths[2])</c> 行列ビューを返します。</para>
    /// </summary>
    /// <returns>A writable side-by-side matrix view sharing this tensor's storage. / このテンソルと格納領域を共有する横並びの行列ビュー。</returns>
    public MatrixSpan<TElement> AsMatrix() => AsTensorSpan().AsMatrix();

    /// <summary>
    /// Copies the elements of this tensor into a new three-dimensional array preserving <c>[row, column, index]</c> indices.
    /// <para><c>[row, column, index]</c> の添字を維持してこのテンソルの要素を新しい 3 次元配列へコピーします。</para>
    /// </summary>
    /// <returns>A new three-dimensional array containing the copied elements. / 要素をコピーした新しい 3 次元配列。</returns>
    public TElement[,,] ToArray() => AsTensorSpan().ToArray();

    /// <summary>
    /// Creates a deep copy of this tensor with independent storage.
    /// <para>独立した格納領域を持つこのテンソルの複製を作成します。</para>
    /// </summary>
    /// <returns>A new tensor containing the same elements. / 同じ要素を持つ新しいテンソル。</returns>
    public Tensor<TElement> Clone() => new(Rows, Columns, MatrixCount, (TElement[])data.Clone());

    /// <summary>
    /// Converts each element to <typeparamref name="TTo"/> using checked numeric conversion (<see cref="INumberBase{TSelf}.CreateChecked{TOther}(TOther)"/>).
    /// <para>チェック付きの数値変換（<see cref="INumberBase{TSelf}.CreateChecked{TOther}(TOther)"/>）を用いて各要素を <typeparamref name="TTo"/> に変換します。</para>
    /// </summary>
    /// <typeparam name="TTo">The target element type. / 変換先の要素型。</typeparam>
    /// <returns>A new tensor of the same shape containing the converted elements. / 変換後の要素を格納した同じ形状の新しいテンソル。</returns>
    /// <exception cref="NotSupportedException">Conversion between <typeparamref name="TElement"/> and <typeparamref name="TTo"/> is not supported. / <typeparamref name="TElement"/> から <typeparamref name="TTo"/> への変換に対応していない場合。</exception>
    /// <exception cref="OverflowException">An element cannot be represented in <typeparamref name="TTo"/>. / いずれかの要素が <typeparamref name="TTo"/> で表現できない場合。</exception>
    public Tensor<TTo> ConvertChecked<TTo>() where TTo : unmanaged, INumberBase<TTo> =>
        Tensor<TTo>.Wrap(Rows, Columns, MatrixCount, DenseStorage.ConvertChecked<TElement, TTo>(data));

    /// <summary>
    /// Implicitly converts a <see cref="Tensor{TElement}"/> to a writable <see cref="TensorSpan{TElement}"/> view.
    /// <para><see cref="Tensor{TElement}"/> を書き込み可能な <see cref="TensorSpan{TElement}"/> ビューへ暗黙的に変換します。</para>
    /// </summary>
    /// <param name="tensor">The tensor to view. / 対象のテンソル。</param>
    /// <returns>A writable view over <paramref name="tensor"/>. / <paramref name="tensor"/> を参照する書き込み可能なビュー。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="tensor"/> is <see langword="null"/>. / <paramref name="tensor"/> が <see langword="null"/> の場合。</exception>
    public static implicit operator TensorSpan<TElement>(Tensor<TElement> tensor) =>
        (tensor ?? throw new ArgumentNullException(nameof(tensor))).AsTensorSpan();

    /// <summary>
    /// Implicitly converts a <see cref="Tensor{TElement}"/> to a read-only <see cref="ReadOnlyTensorSpan{TElement}"/> view.
    /// <para><see cref="Tensor{TElement}"/> を読み取り専用の <see cref="ReadOnlyTensorSpan{TElement}"/> ビューへ暗黙的に変換します。</para>
    /// </summary>
    /// <param name="tensor">The tensor to view. / 対象のテンソル。</param>
    /// <returns>A read-only view over <paramref name="tensor"/>. / <paramref name="tensor"/> を参照する読み取り専用ビュー。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="tensor"/> is <see langword="null"/>. / <paramref name="tensor"/> が <see langword="null"/> の場合。</exception>
    public static implicit operator ReadOnlyTensorSpan<TElement>(Tensor<TElement> tensor) =>
        (tensor ?? throw new ArgumentNullException(nameof(tensor))).AsTensorSpan();

    /// <summary>
    /// Computes the elementwise sum of two tensors.
    /// <para>2 つのテンソルの要素ごとの和を計算します。</para>
    /// </summary>
    /// <param name="a">The left-hand tensor. / 左辺のテンソル。</param>
    /// <param name="b">The right-hand tensor. / 右辺のテンソル。</param>
    /// <returns>A new tensor containing <c>a + b</c>. / <c>a + b</c> を格納した新しいテンソル。</returns>
    public static Tensor<TElement> operator +(Tensor<TElement> a, Tensor<TElement> b) => XMath.Add(a, b);

    /// <summary>
    /// Computes the elementwise difference of two tensors.
    /// <para>2 つのテンソルの要素ごとの差を計算します。</para>
    /// </summary>
    /// <param name="a">The left-hand tensor. / 左辺のテンソル。</param>
    /// <param name="b">The right-hand tensor. / 右辺のテンソル。</param>
    /// <returns>A new tensor containing <c>a - b</c>. / <c>a - b</c> を格納した新しいテンソル。</returns>
    public static Tensor<TElement> operator -(Tensor<TElement> a, Tensor<TElement> b) => XMath.Subtract(a, b);

    /// <summary>
    /// Negates each element of the tensor.
    /// <para>テンソルの各要素の符号を反転します。</para>
    /// </summary>
    /// <param name="a">The input tensor. / 入力テンソル。</param>
    /// <returns>A new tensor containing <c>-a</c>. / <c>-a</c> を格納した新しいテンソル。</returns>
    public static Tensor<TElement> operator -(Tensor<TElement> a) => XMath.Negate(a);

    /// <summary>
    /// Multiplies a tensor by a scalar.
    /// <para>テンソルをスカラー倍します。</para>
    /// </summary>
    /// <param name="scalar">The scalar multiplier. / 掛けるスカラー値。</param>
    /// <param name="a">The input tensor. / 入力テンソル。</param>
    /// <returns>A new tensor containing <c>scalar * a</c>. / <c>scalar * a</c> を格納した新しいテンソル。</returns>
    public static Tensor<TElement> operator *(TElement scalar, Tensor<TElement> a) => XMath.Multiply(a, scalar);

    /// <summary>
    /// Multiplies a tensor by a scalar.
    /// <para>テンソルをスカラー倍します。</para>
    /// </summary>
    /// <param name="a">The input tensor. / 入力テンソル。</param>
    /// <param name="scalar">The scalar multiplier. / 掛けるスカラー値。</param>
    /// <returns>A new tensor containing <c>a * scalar</c>. / <c>a * scalar</c> を格納した新しいテンソル。</returns>
    public static Tensor<TElement> operator *(Tensor<TElement> a, TElement scalar) => XMath.Multiply(a, scalar);

    /// <summary>
    /// Divides each element of a tensor by a scalar.
    /// <para>テンソルの各要素をスカラーで割ります。</para>
    /// </summary>
    /// <param name="a">The input tensor. / 入力テンソル。</param>
    /// <param name="scalar">The scalar divisor. / 除数。</param>
    /// <returns>A new tensor containing the quotients. / 各要素の商を格納した新しいテンソル。</returns>
    public static Tensor<TElement> operator /(Tensor<TElement> a, TElement scalar) => XMath.Divide(a, scalar);

    /// <summary>
    /// Computes the batched matrix product <c>C[i, j, k] = Σ_l A[i, l, k]·B[l, j, k]</c> (without summing over <c>k</c>).
    /// <para>行列ごとのバッチ積 <c>C[i, j, k] = Σ_l A[i, l, k]·B[l, j, k]</c>（<c>k</c> についての和は取りません）を計算します。</para>
    /// </summary>
    /// <param name="a">The left-hand tensor. / 左辺のテンソル。</param>
    /// <param name="b">The right-hand tensor. / 右辺のテンソル。</param>
    /// <returns>A new tensor containing the per-matrix products. / 各行列の積を格納した新しいテンソル。</returns>
    public static Tensor<TElement> operator *(Tensor<TElement> a, Tensor<TElement> b) => XMath.Multiply(a, b);

    /// <summary>
    /// Adds the elements of <paramref name="other"/> to this tensor in place.
    /// <para>このテンソルに <paramref name="other"/> をインプレースで加算します。</para>
    /// </summary>
    /// <param name="other">The tensor to add. / 加算するテンソル。</param>
    public void operator +=(ReadOnlyTensorSpan<TElement> other) => XMath.Add(this, other, this);

    /// <summary>
    /// Subtracts the elements of <paramref name="other"/> from this tensor in place.
    /// <para>このテンソルから <paramref name="other"/> をインプレースで減算します。</para>
    /// </summary>
    /// <param name="other">The tensor to subtract. / 減算するテンソル。</param>
    public void operator -=(ReadOnlyTensorSpan<TElement> other) => XMath.Subtract(this, other, this);

    /// <summary>
    /// Scales this tensor by <paramref name="scalar"/> in place.
    /// <para>このテンソルを <paramref name="scalar"/> でインプレースにスカラー倍します。</para>
    /// </summary>
    /// <param name="scalar">The scalar multiplier. / 掛けるスカラー値。</param>
    public void operator *=(TElement scalar) => XMath.InPlace.Multiply(this, scalar);

    /// <summary>
    /// Divides each element of this tensor by a scalar in place.
    /// <para>このテンソルの各要素をインプレースでスカラー除算します。</para>
    /// </summary>
    /// <param name="scalar">The scalar divisor. / 除数。</param>
    public void operator /=(TElement scalar) => XMath.InPlace.Divide(this, scalar);

    /// <summary>
    /// Returns a string representation containing the element type and tensor shape.
    /// <para>要素型とテンソルの形状を表す文字列を返します。</para>
    /// </summary>
    /// <returns>A string describing the tensor's type and shape. / テンソルの型と形状を表す文字列。</returns>
    public override string ToString() =>
        string.Create(CultureInfo.InvariantCulture, $"Tensor<{typeof(TElement).Name}> [{Rows}, {Columns}, {MatrixCount}]");

    internal static Tensor<TElement> Wrap(int rows, int columns, int count, TElement[] data) => new(rows, columns, count, data);

    // Allocates uninitialized storage for kernels that overwrite every element (not zero-filled).
    internal static Tensor<TElement> Uninitialized(int rows, int columns, int count) =>
        new(rows, columns, count, GC.AllocateUninitializedArray<TElement>(DenseStorage.Length(rows, columns, count)));
    internal TElement[] Data => data;
    internal bool IsSquare => Rows == Columns;

    private int Offset(int row, int column, int index) => row + column * Rows + index * Rows * Columns;

    private void CheckIndex(int row, int column, int index)
    {
        DenseStorage.CheckIndex(row, Rows, nameof(row));
        DenseStorage.CheckIndex(column, Columns, nameof(column));
        DenseStorage.CheckIndex(index, MatrixCount, nameof(index));
    }
}
