using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Tuna.LinearAlgebra;

/// <summary>
/// Represents a zero-copy, strided read-only view over a one-dimensional vector.
/// <para>ストライド付きの 1 次元ベクトルを参照する、コピーなしの読み取り専用ビューを表します。</para>
/// </summary>
/// <typeparam name="T">The element type of the vector. / ベクトルの要素型。</typeparam>
/// <remarks>
/// <para>Element <c>i</c> is located at offset <c>i * Strides[0]</c> from the start reference. Matrix rows, columns, and diagonals are represented using this view.</para>
/// <para><see cref="Data"/>, <see cref="Lengths"/>, and <see cref="Strides"/> can be passed directly to the <c>System.Numerics.Tensors.ReadOnlyTensorSpan&lt;T&gt;</c> constructor.</para>
/// <para>要素 <c>i</c> は先頭参照から <c>i * Strides[0]</c> 要素目の位置にあります。行列の行・列・対角成分はこのビューで表現されます。</para>
/// <para><see cref="Data"/>、<see cref="Lengths"/>、<see cref="Strides"/> は <c>System.Numerics.Tensors.ReadOnlyTensorSpan&lt;T&gt;</c> のコンストラクターにそのまま渡せます。</para>
/// </remarks>
public readonly ref struct ReadOnlyVectorSpan<T> where T : unmanaged, INumberBase<T>
{
    private readonly ref readonly T reference;
    // [0] is length, [1] is stride.
    private readonly NintPair shape;

    internal ReadOnlyVectorSpan(ref readonly T reference, int length, int stride)
    {
        this.reference = ref reference;
        shape = NintPair.Create(length, StridedLayout.Normalize(length, stride));
    }

    /// <summary>
    /// Initializes a read-only view over a contiguous span of elements.
    /// <para>連続した領域を参照する読み取り専用ビューを初期化します。</para>
    /// </summary>
    /// <param name="data">The contiguous span to view. / 参照する連続領域。</param>
    public ReadOnlyVectorSpan(ReadOnlySpan<T> data) : this(in MemoryMarshal.GetReference(data), data.Length, 1) { }

    /// <summary>
    /// Initializes a read-only view over the specified span with explicit length and stride.
    /// <para>長さとストライドを指定して領域を参照する読み取り専用ビューを初期化します。</para>
    /// </summary>
    /// <param name="data">The backing span starting at the first element and covering through the last element. / 先頭要素から最後の要素までを含む格納領域。</param>
    /// <param name="lengths">A length-1 span containing the vector length. / ベクトル長を含む長さ 1 の形状。</param>
    /// <param name="strides">A length-1 span containing the non-negative element stride. / 非負の要素ストライドを含む長さ 1 のスパン。</param>
    /// <exception cref="ArgumentException"><paramref name="lengths"/> or <paramref name="strides"/> does not have length 1, or <paramref name="data"/> is too short. / <paramref name="lengths"/> または <paramref name="strides"/> の長さが 1 でない、または <paramref name="data"/> の長さが不足している場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">A length or stride is negative or exceeds <see cref="int.MaxValue"/>. / 長さまたはストライドが負、または <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="OverflowException">The spanned extent from the first to the last element exceeds <see cref="int.MaxValue"/>. / 先頭から最終要素までの範囲が <see cref="int.MaxValue"/> を超える場合。</exception>
    public ReadOnlyVectorSpan(ReadOnlySpan<T> data, ReadOnlySpan<nint> lengths, ReadOnlySpan<nint> strides)
    {
        var (length, _, stride, _) = StridedLayout.Validate(lengths, strides, 1, writable: false);
        if (data.Length < StridedLayout.Extent(length, 1, stride, 0))
        {
            throw new ArgumentException("The data is too short for the lengths and strides.", nameof(data));
        }

        this = new(in MemoryMarshal.GetReference(data), length, stride);
    }

    /// <summary>
    /// Creates a read-only view from a reference to the first element, length, and stride.
    /// <para>先頭要素への参照・長さ・ストライドから読み取り専用ビューを作成します。</para>
    /// </summary>
    /// <param name="reference">A reference to the first element. / 先頭要素への参照。</param>
    /// <param name="lengths">A length-1 span containing the vector length. / ベクトル長を含む長さ 1 の形状。</param>
    /// <param name="strides">A length-1 span containing the non-negative element stride. / 非負の要素ストライドを含む長さ 1 のスパン。</param>
    /// <returns>A read-only view referencing the specified memory layout. / 指定したレイアウトを参照する読み取り専用ビュー。</returns>
    /// <remarks>
    /// <para>Bounds of the backing memory are not checked, matching <see cref="MemoryMarshal.CreateReadOnlySpan"/>.</para>
    /// <para><see cref="MemoryMarshal.CreateReadOnlySpan"/> と同様に、呼び出し元のメモリ範囲の境界検査は行いません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="lengths"/> or <paramref name="strides"/> does not have length 1. / <paramref name="lengths"/> または <paramref name="strides"/> の長さが 1 でない場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">A length or stride is negative or exceeds <see cref="int.MaxValue"/>. / 長さまたはストライドが負、または <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="OverflowException">The spanned extent exceeds <see cref="int.MaxValue"/>. / 参照範囲の要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static ReadOnlyVectorSpan<T> Create(ref readonly T reference, ReadOnlySpan<nint> lengths, ReadOnlySpan<nint> strides)
    {
        var (length, _, stride, _) = StridedLayout.Validate(lengths, strides, 1, writable: false);
        return new(in reference, length, stride);
    }

    /// <summary>
    /// Gets the number of elements in the vector view.
    /// <para>ベクトルビューの要素数を取得します。</para>
    /// </summary>
    public int Length => (int)shape[0];

    /// <summary>
    /// Gets the number of dimensions, which is always 1.
    /// <para>次元数（常に 1）を取得します。</para>
    /// </summary>
    public int Rank => 1;

    /// <summary>
    /// Gets the total number of elements.
    /// <para>総要素数を取得します。</para>
    /// </summary>
    public nint FlattenedLength => Length;

    /// <summary>
    /// Gets a value indicating whether the view contains no elements.
    /// <para>ビューが空かどうかを示す値を取得します。</para>
    /// </summary>
    public bool IsEmpty => Length == 0;

    /// <summary>
    /// Gets the shape <c>[Length]</c> of the view.
    /// <para>ビューの形状 <c>[Length]</c> を取得します。</para>
    /// </summary>
    [UnscopedRef] public ReadOnlySpan<nint> Lengths => ((ReadOnlySpan<nint>)shape)[..1];

    /// <summary>
    /// Gets the element stride of the view (0 when <see cref="Length"/> is 0 or 1).
    /// <para>要素単位のストライドを取得します（<see cref="Length"/> が 0 または 1 の場合は 0）。</para>
    /// </summary>
    [UnscopedRef] public ReadOnlySpan<nint> Strides => ((ReadOnlySpan<nint>)shape)[1..];

    /// <summary>
    /// Gets a contiguous span from the first element through the last element, including any intervening elements outside the view.
    /// <para>先頭要素から最終要素までの連続領域を取得します（ストライド間のビュー外要素も含みます）。</para>
    /// </summary>
    public ReadOnlySpan<T> Data => MemoryMarshal.CreateReadOnlySpan(in reference, StridedLayout.Extent(Length, 1, Stride, 0));

    internal int Stride => (int)shape[1];
    internal ref readonly T Reference => ref reference;

    /// <summary>
    /// Gets a read-only reference to the element at the specified index.
    /// <para>指定した位置の要素への読み取り専用参照を取得します。</para>
    /// </summary>
    /// <param name="index">The zero-based element index. / 0 から始まる要素インデックス。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is outside the bounds of the view. / <paramref name="index"/> が範囲外の場合。</exception>
    public ref readonly T this[int index]
    {
        get
        {
            DenseStorage.CheckIndex(index, Length, nameof(index));
            return ref ElementAt(index);
        }
    }

    /// <summary>
    /// Gets a zero-copy read-only view of the specified subrange.
    /// <para>指定した範囲の部分ベクトルを参照する読み取り専用ビューを取得します。</para>
    /// </summary>
    /// <param name="range">The range of elements to slice. / 切り出す要素の範囲。</param>
    public ReadOnlyVectorSpan<T> this[Range range]
    {
        get
        {
            var (start, length) = range.GetOffsetAndLength(Length);
            return new(in length == 0 ? ref reference : ref ElementAt(start), length, Stride);
        }
    }

    /// <summary>
    /// Copies the elements of this view into a destination view of the same length.
    /// <para>このビューの要素を同じ長さの出力先ビューへコピーします。</para>
    /// </summary>
    /// <param name="destination">The destination vector view. / コピー先のベクトルビュー。</param>
    /// <remarks>
    /// <para>Throws an exception if <paramref name="destination"/> partially overlaps this view.</para>
    /// <para><paramref name="destination"/> がこのビューと部分的に重なっている場合は例外をスローします。</para>
    /// </remarks>
    public void CopyTo(VectorSpan<T> destination) => StridedKernels.Copy(AsMatrix(), destination.AsMatrix());

    /// <summary>
    /// Copies the elements of this view into a new <see cref="Vector{T}"/>.
    /// <para>このビューの要素を新しい <see cref="Vector{T}"/> へコピーします。</para>
    /// </summary>
    /// <returns>A new vector containing the copied elements. / 要素をコピーした新しいベクトル。</returns>
    public Vector<T> ToVector()
    {
        var result = Vector<T>.Uninitialized(Length);
        CopyTo(result);
        return result;
    }

    /// <summary>
    /// Copies the elements of this view into a new array.
    /// <para>このビューの要素を新しい配列へコピーします。</para>
    /// </summary>
    /// <returns>A new array containing the copied elements. / 要素をコピーした新しい配列。</returns>
    public T[] ToArray()
    {
        var result = new T[Length];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = ElementAt(i);
        }

        return result;
    }

    internal ref readonly T ElementAt(int index) => ref Unsafe.Add(ref Unsafe.AsRef(in reference), (nint)index * Stride);

    // Views the vector as a Length x 1 matrix to share elementwise kernel implementations.
    internal ReadOnlyMatrixSpan<T> AsMatrix() => new(in reference, Length, 1, Stride, 0);

    /// <summary>
    /// Returns a zero-copy read-only <c>Length × count</c> matrix view repeating this vector across <paramref name="count"/> columns.
    /// <para>このベクトルを <paramref name="count"/> 本の列として繰り返す、コピーなしの読み取り専用 <c>Length × count</c> 行列ビューを返します。</para>
    /// </summary>
    /// <param name="count">The number of columns. / 列数。</param>
    /// <returns>A read-only matrix view with zero column stride. / 列ストライドが 0 の読み取り専用行列ビュー。</returns>
    /// <remarks>
    /// <para>Broadcasts along the trailing (column) axis with stride 0, enabling column-wise operations such as <c>XMath.Subtract(X, mean.Broadcast(X.Columns), X)</c>. To broadcast along rows instead, use <see cref="BroadcastRows(int)"/>. Because distinct indices alias the same elements, the result is read-only.</para>
    /// <para>列軸のストライドを 0 にしたブロードキャストであり、<c>XMath.Subtract(X, mean.Broadcast(X.Columns), X)</c> のように行列の各列へベクトルを適用する際に使用します。各行にブロードキャストする場合は <see cref="BroadcastRows(int)"/> を使用してください。異なる添字が同じ要素を参照するため読み取り専用になります。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative. / <paramref name="count"/> が負の場合。</exception>
    public ReadOnlyMatrixSpan<T> Broadcast(int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        return new(in reference, Length, count, Stride, 0);
    }

    /// <summary>
    /// Returns a zero-copy read-only <c>rows × Length</c> matrix view repeating this vector across <paramref name="rows"/> rows.
    /// <para>このベクトルを <paramref name="rows"/> 本の行として繰り返す、コピーなしの読み取り専用 <c>rows × Length</c> 行列ビューを返します。</para>
    /// </summary>
    /// <param name="rows">The number of rows. / 行数。</param>
    /// <returns>A read-only matrix view with zero row stride. / 行ストライドが 0 の読み取り専用行列ビュー。</returns>
    /// <remarks>
    /// <para>Broadcasts along the row axis with stride 0, enabling row-wise operations such as <c>XMath.Hadamard(X, scale.BroadcastRows(X.Rows))</c>. Because distinct indices alias the same elements, the result is read-only.</para>
    /// <para>行軸のストライドを 0 にしたブロードキャストであり、<c>XMath.Hadamard(X, scale.BroadcastRows(X.Rows))</c> のように行列の各行へベクトルを適用する際に使用します。異なる添字が同じ要素を参照するため読み取り専用になります。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rows"/> is negative. / <paramref name="rows"/> が負の場合。</exception>
    public ReadOnlyMatrixSpan<T> BroadcastRows(int rows)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(rows);
        return new(in reference, rows, Length, 0, Stride);
    }

    /// <summary>
    /// Implicitly wraps a contiguous <see cref="ReadOnlySpan{T}"/> as a <see cref="ReadOnlyVectorSpan{T}"/>.
    /// <para>連続した <see cref="ReadOnlySpan{T}"/> を <see cref="ReadOnlyVectorSpan{T}"/> へ暗黙的に変換します。</para>
    /// </summary>
    /// <param name="data">The contiguous span to view. / 参照する連続領域。</param>
    /// <returns>A read-only vector view with stride 1. / ストライド 1 の読み取り専用ベクトルビュー。</returns>
    public static implicit operator ReadOnlyVectorSpan<T>(ReadOnlySpan<T> data) => new(data);

    /// <summary>
    /// Computes the elementwise sum of two vector views.
    /// <para>2 つのベクトルビューの要素ごとの和を計算します。</para>
    /// </summary>
    /// <param name="x">The left-hand vector. / 左辺のベクトル。</param>
    /// <param name="y">The right-hand vector. / 右辺のベクトル。</param>
    /// <returns>A new vector containing <c>x + y</c>. / <c>x + y</c> を格納した新しいベクトル。</returns>
    public static Vector<T> operator +(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y) => XMath.Add(x, y);

    /// <summary>
    /// Computes the elementwise difference of two vector views.
    /// <para>2 つのベクトルビューの要素ごとの差を計算します。</para>
    /// </summary>
    /// <param name="x">The left-hand vector. / 左辺のベクトル。</param>
    /// <param name="y">The right-hand vector. / 右辺のベクトル。</param>
    /// <returns>A new vector containing <c>x - y</c>. / <c>x - y</c> を格納した新しいベクトル。</returns>
    public static Vector<T> operator -(ReadOnlyVectorSpan<T> x, ReadOnlyVectorSpan<T> y) => XMath.Subtract(x, y);

    /// <summary>
    /// Negates each element of the vector view.
    /// <para>ベクトルビューの各要素の符号を反転します。</para>
    /// </summary>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>A new vector containing <c>-x</c>. / <c>-x</c> を格納した新しいベクトル。</returns>
    public static Vector<T> operator -(ReadOnlyVectorSpan<T> x) => XMath.Negate(x);

    /// <summary>
    /// Multiplies a vector view by a scalar.
    /// <para>ベクトルビューをスカラー倍します。</para>
    /// </summary>
    /// <param name="scalar">The scalar multiplier. / 掛けるスカラー値。</param>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>A new vector containing <c>scalar * x</c>. / <c>scalar * x</c> を格納した新しいベクトル。</returns>
    public static Vector<T> operator *(T scalar, ReadOnlyVectorSpan<T> x) => XMath.Multiply(x, scalar);

    /// <summary>
    /// Multiplies a vector view by a scalar.
    /// <para>ベクトルビューをスカラー倍します。</para>
    /// </summary>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="scalar">The scalar multiplier. / 掛けるスカラー値。</param>
    /// <returns>A new vector containing <c>x * scalar</c>. / <c>x * scalar</c> を格納した新しいベクトル。</returns>
    public static Vector<T> operator *(ReadOnlyVectorSpan<T> x, T scalar) => XMath.Multiply(x, scalar);

    /// <summary>
    /// Computes the matrix-vector product <c>a * x</c>.
    /// <para>行列ベクトル積 <c>a * x</c> を計算します。</para>
    /// </summary>
    /// <param name="a">The input matrix. / 入力行列。</param>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>A new vector containing <c>a * x</c>. / <c>a * x</c> を格納した新しいベクトル。</returns>
    public static Vector<T> operator *(ReadOnlyMatrixSpan<T> a, ReadOnlyVectorSpan<T> x) => XMath.Multiply(a, x);
}

/// <summary>
/// Represents a zero-copy, strided writable view over a one-dimensional vector.
/// <para>ストライド付きの 1 次元ベクトルを参照する、コピーなしの書き込み可能なビューを表します。</para>
/// </summary>
/// <typeparam name="T">The element type of the vector. / ベクトルの要素型。</typeparam>
/// <remarks>
/// <para><see cref="Data"/>, <see cref="Lengths"/>, and <see cref="Strides"/> can be passed directly to the <c>System.Numerics.Tensors.TensorSpan&lt;T&gt;</c> constructor.</para>
/// <para><see cref="Data"/>、<see cref="Lengths"/>、<see cref="Strides"/> は <c>System.Numerics.Tensors.TensorSpan&lt;T&gt;</c> のコンストラクターにそのまま渡せます。</para>
/// </remarks>
public readonly ref struct VectorSpan<T> where T : unmanaged, INumberBase<T>
{
    private readonly ref T reference;
    private readonly NintPair shape;

    internal VectorSpan(ref T reference, int length, int stride)
    {
        this.reference = ref reference;
        shape = NintPair.Create(length, StridedLayout.Normalize(length, stride));
    }

    /// <summary>
    /// Initializes a writable view over a contiguous span of elements.
    /// <para>連続した領域を参照する書き込み可能なビューを初期化します。</para>
    /// </summary>
    /// <param name="data">The contiguous span to view. / 参照する連続領域。</param>
    public VectorSpan(Span<T> data) : this(ref MemoryMarshal.GetReference(data), data.Length, 1) { }

    /// <summary>
    /// Initializes a writable view over the specified span with explicit length and stride.
    /// <para>長さとストライドを指定して領域を参照する書き込み可能なビューを初期化します。</para>
    /// </summary>
    /// <param name="data">The backing span starting at the first element and covering through the last element. / 先頭要素から最後の要素までを含む格納領域。</param>
    /// <param name="lengths">A length-1 span containing the vector length. / ベクトル長を含む長さ 1 の形状。</param>
    /// <param name="strides">A length-1 span containing the element stride (must be positive when length exceeds 1). / 要素ストライドを含む長さ 1 のスパン（長さが 2 以上の場合は正）。</param>
    /// <exception cref="ArgumentException"><paramref name="lengths"/> or <paramref name="strides"/> does not have length 1, <paramref name="data"/> is too short, or distinct indices alias the same element. / <paramref name="lengths"/> または <paramref name="strides"/> の長さが 1 でない、<paramref name="data"/> の長さが不足している、または異なる添字が同じ要素を参照する場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">A length or stride is negative or exceeds <see cref="int.MaxValue"/>. / 長さまたはストライドが負、または <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="OverflowException">The spanned extent exceeds <see cref="int.MaxValue"/>. / 参照範囲の要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public VectorSpan(Span<T> data, ReadOnlySpan<nint> lengths, ReadOnlySpan<nint> strides)
    {
        var (length, _, stride, _) = StridedLayout.Validate(lengths, strides, 1, writable: true);
        if (data.Length < StridedLayout.Extent(length, 1, stride, 0))
        {
            throw new ArgumentException("The data is too short for the lengths and strides.", nameof(data));
        }

        this = new(ref MemoryMarshal.GetReference(data), length, stride);
    }

    /// <summary>
    /// Creates a writable view from a reference to the first element, length, and stride.
    /// <para>先頭要素への参照・長さ・ストライドから書き込み可能なビューを作成します。</para>
    /// </summary>
    /// <param name="reference">A reference to the first element. / 先頭要素への参照。</param>
    /// <param name="lengths">A length-1 span containing the vector length. / ベクトル長を含む長さ 1 の形状。</param>
    /// <param name="strides">A length-1 span containing the element stride. / 要素ストライドを含む長さ 1 のスパン。</param>
    /// <returns>A writable view referencing the specified memory layout. / 指定したレイアウトを参照する書き込み可能なビュー。</returns>
    /// <remarks>
    /// <para>Bounds of the backing memory are not checked, matching <see cref="MemoryMarshal.CreateSpan"/>.</para>
    /// <para><see cref="MemoryMarshal.CreateSpan"/> と同様に、呼び出し元のメモリ範囲の境界検査は行いません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="lengths"/> or <paramref name="strides"/> does not have length 1, or distinct indices alias the same element. / <paramref name="lengths"/> または <paramref name="strides"/> の長さが 1 でない、または異なる添字が同じ要素を参照する場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException">A length or stride is negative or exceeds <see cref="int.MaxValue"/>. / 長さまたはストライドが負、または <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="OverflowException">The spanned extent exceeds <see cref="int.MaxValue"/>. / 参照範囲の要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static VectorSpan<T> Create(ref T reference, ReadOnlySpan<nint> lengths, ReadOnlySpan<nint> strides)
    {
        var (length, _, stride, _) = StridedLayout.Validate(lengths, strides, 1, writable: true);
        return new(ref reference, length, stride);
    }

    /// <summary>
    /// Gets the number of elements in the vector view.
    /// <para>ベクトルビューの要素数を取得します。</para>
    /// </summary>
    public int Length => (int)shape[0];

    /// <summary>
    /// Gets the number of dimensions, which is always 1.
    /// <para>次元数（常に 1）を取得します。</para>
    /// </summary>
    public int Rank => 1;

    /// <summary>
    /// Gets the total number of elements.
    /// <para>総要素数を取得します。</para>
    /// </summary>
    public nint FlattenedLength => Length;

    /// <summary>
    /// Gets a value indicating whether the view contains no elements.
    /// <para>ビューが空かどうかを示す値を取得します。</para>
    /// </summary>
    public bool IsEmpty => Length == 0;

    /// <summary>
    /// Gets the shape <c>[Length]</c> of the view.
    /// <para>ビューの形状 <c>[Length]</c> を取得します。</para>
    /// </summary>
    [UnscopedRef] public ReadOnlySpan<nint> Lengths => ((ReadOnlySpan<nint>)shape)[..1];

    /// <summary>
    /// Gets the element stride of the view (0 when <see cref="Length"/> is 0 or 1).
    /// <para>要素単位のストライドを取得します（<see cref="Length"/> が 0 または 1 の場合は 0）。</para>
    /// </summary>
    [UnscopedRef] public ReadOnlySpan<nint> Strides => ((ReadOnlySpan<nint>)shape)[1..];

    /// <summary>
    /// Gets a contiguous span from the first element through the last element, including any intervening elements outside the view.
    /// <para>先頭要素から最終要素までの連続領域を取得します（ストライド間のビュー外要素も含みます）。</para>
    /// </summary>
    public Span<T> Data => MemoryMarshal.CreateSpan(ref reference, StridedLayout.Extent(Length, 1, (int)shape[1], 0));

    /// <summary>
    /// Gets a writable reference to the element at the specified index.
    /// <para>指定した位置の要素への書き込み可能な参照を取得します。</para>
    /// </summary>
    /// <param name="index">The zero-based element index. / 0 から始まる要素インデックス。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is outside the bounds of the view. / <paramref name="index"/> が範囲外の場合。</exception>
    public ref T this[int index] => ref Unsafe.AsRef(in AsReadOnly()[index]);

    /// <summary>
    /// Gets a zero-copy writable view of the specified subrange.
    /// <para>指定した範囲の部分ベクトルを参照する書き込み可能なビューを取得します。</para>
    /// </summary>
    /// <param name="range">The range of elements to slice. / 切り出す要素の範囲。</param>
    public VectorSpan<T> this[Range range] => Writable(AsReadOnly()[range]);

    /// <summary>
    /// Returns a read-only view over the same elements.
    /// <para>同じ要素を参照する読み取り専用ビューを返します。</para>
    /// </summary>
    /// <returns>A read-only vector view. / 読み取り専用のベクトルビュー。</returns>
    public ReadOnlyVectorSpan<T> AsReadOnly() => new(in reference, Length, (int)shape[1]);

    /// <summary>
    /// Fills all elements of the vector view with the specified value.
    /// <para>ベクトルビューのすべての要素を指定した値で埋めます。</para>
    /// </summary>
    /// <param name="value">The value to assign to each element. / 各要素に設定する値。</param>
    public void Fill(T value)
    {
        for (int i = 0; i < Length; i++)
        {
            Unsafe.AsRef(in AsReadOnly().ElementAt(i)) = value;
        }
    }

    /// <summary>
    /// Copies the elements of this view into a destination view of the same length.
    /// <para>このビューの要素を同じ長さの出力先ビューへコピーします。</para>
    /// </summary>
    /// <param name="destination">The destination vector view. / コピー先のベクトルビュー。</param>
    /// <remarks>
    /// <para>Throws an exception if <paramref name="destination"/> partially overlaps this view.</para>
    /// <para><paramref name="destination"/> がこのビューと部分的に重なっている場合は例外をスローします。</para>
    /// </remarks>
    public void CopyTo(VectorSpan<T> destination) => AsReadOnly().CopyTo(destination);

    /// <summary>
    /// Copies the elements of this view into a new <see cref="Vector{T}"/>.
    /// <para>このビューの要素を新しい <see cref="Vector{T}"/> へコピーします。</para>
    /// </summary>
    /// <returns>A new vector containing the copied elements. / 要素をコピーした新しいベクトル。</returns>
    public Vector<T> ToVector() => AsReadOnly().ToVector();

    /// <summary>
    /// Copies the elements of this view into a new array.
    /// <para>このビューの要素を新しい配列へコピーします。</para>
    /// </summary>
    /// <returns>A new array containing the copied elements. / 要素をコピーした新しい配列。</returns>
    public T[] ToArray() => AsReadOnly().ToArray();

    internal MatrixSpan<T> AsMatrix() => new(ref reference, Length, 1, (int)shape[1], 0);

    /// <summary>
    /// Returns a zero-copy read-only <c>Length × count</c> matrix view repeating this vector across <paramref name="count"/> columns.
    /// <para>このベクトルを <paramref name="count"/> 本の列として繰り返す、コピーなしの読み取り専用 <c>Length × count</c> 行列ビューを返します。</para>
    /// </summary>
    /// <param name="count">The number of columns. / 列数。</param>
    /// <returns>A read-only matrix view with zero column stride. / 列ストライドが 0 の読み取り専用行列ビュー。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative. / <paramref name="count"/> が負の場合。</exception>
    public ReadOnlyMatrixSpan<T> Broadcast(int count) => AsReadOnly().Broadcast(count);

    /// <summary>
    /// Returns a zero-copy read-only <c>rows × Length</c> matrix view repeating this vector across <paramref name="rows"/> rows.
    /// <para>このベクトルを <paramref name="rows"/> 本の行として繰り返す、コピーなしの読み取り専用 <c>rows × Length</c> 行列ビューを返します。</para>
    /// </summary>
    /// <param name="rows">The number of rows. / 行数。</param>
    /// <returns>A read-only matrix view with zero row stride. / 行ストライドが 0 の読み取り専用行列ビュー。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rows"/> is negative. / <paramref name="rows"/> が負の場合。</exception>
    public ReadOnlyMatrixSpan<T> BroadcastRows(int rows) => AsReadOnly().BroadcastRows(rows);

    internal static VectorSpan<T> Writable(ReadOnlyVectorSpan<T> view) =>
        new(ref Unsafe.AsRef(in view.Reference), view.Length, view.Stride);

    /// <summary>
    /// Implicitly converts a writable <see cref="VectorSpan{T}"/> to a <see cref="ReadOnlyVectorSpan{T}"/>.
    /// <para>書き込み可能な <see cref="VectorSpan{T}"/> を <see cref="ReadOnlyVectorSpan{T}"/> へ暗黙的に変換します。</para>
    /// </summary>
    /// <param name="view">The writable view to convert. / 変換元の書き込み可能なビュー。</param>
    /// <returns>A read-only view referencing the same elements. / 同じ要素を参照する読み取り専用ビュー。</returns>
    public static implicit operator ReadOnlyVectorSpan<T>(VectorSpan<T> view) => view.AsReadOnly();

    /// <summary>
    /// Implicitly wraps a contiguous <see cref="Span{T}"/> as a <see cref="VectorSpan{T}"/>.
    /// <para>連続した <see cref="Span{T}"/> を <see cref="VectorSpan{T}"/> へ暗黙的に変換します。</para>
    /// </summary>
    /// <param name="data">The contiguous span to view. / 参照する連続領域。</param>
    /// <returns>A writable vector view with stride 1. / ストライド 1 の書き込み可能なベクトルビュー。</returns>
    public static implicit operator VectorSpan<T>(Span<T> data) => new(data);

    /// <summary>
    /// Computes the elementwise sum of two vector views.
    /// <para>2 つのベクトルビューの要素ごとの和を計算します。</para>
    /// </summary>
    /// <param name="x">The left-hand vector. / 左辺のベクトル。</param>
    /// <param name="y">The right-hand vector. / 右辺のベクトル。</param>
    /// <returns>A new vector containing <c>x + y</c>. / <c>x + y</c> を格納した新しいベクトル。</returns>
    public static Vector<T> operator +(VectorSpan<T> x, VectorSpan<T> y) => XMath.Add(x, y);

    /// <summary>
    /// Computes the elementwise difference of two vector views.
    /// <para>2 つのベクトルビューの要素ごとの差を計算します。</para>
    /// </summary>
    /// <param name="x">The left-hand vector. / 左辺のベクトル。</param>
    /// <param name="y">The right-hand vector. / 右辺のベクトル。</param>
    /// <returns>A new vector containing <c>x - y</c>. / <c>x - y</c> を格納した新しいベクトル。</returns>
    public static Vector<T> operator -(VectorSpan<T> x, VectorSpan<T> y) => XMath.Subtract(x, y);

    /// <summary>
    /// Negates each element of the vector view.
    /// <para>ベクトルビューの各要素の符号を反転します。</para>
    /// </summary>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>A new vector containing <c>-x</c>. / <c>-x</c> を格納した新しいベクトル。</returns>
    public static Vector<T> operator -(VectorSpan<T> x) => XMath.Negate(x);

    /// <summary>
    /// Multiplies a vector view by a scalar.
    /// <para>ベクトルビューをスカラー倍します。</para>
    /// </summary>
    /// <param name="scalar">The scalar multiplier. / 掛けるスカラー値。</param>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>A new vector containing <c>scalar * x</c>. / <c>scalar * x</c> を格納した新しいベクトル。</returns>
    public static Vector<T> operator *(T scalar, VectorSpan<T> x) => XMath.Multiply(x, scalar);

    /// <summary>
    /// Multiplies a vector view by a scalar.
    /// <para>ベクトルビューをスカラー倍します。</para>
    /// </summary>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="scalar">The scalar multiplier. / 掛けるスカラー値。</param>
    /// <returns>A new vector containing <c>x * scalar</c>. / <c>x * scalar</c> を格納した新しいベクトル。</returns>
    public static Vector<T> operator *(VectorSpan<T> x, T scalar) => XMath.Multiply(x, scalar);

    /// <summary>
    /// Computes the matrix-vector product <c>a * x</c>.
    /// <para>行列ベクトル積 <c>a * x</c> を計算します。</para>
    /// </summary>
    /// <param name="a">The input matrix. / 入力行列。</param>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>A new vector containing <c>a * x</c>. / <c>a * x</c> を格納した新しいベクトル。</returns>
    public static Vector<T> operator *(ReadOnlyMatrixSpan<T> a, VectorSpan<T> x) => XMath.Multiply(a, x);
}
