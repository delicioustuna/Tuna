using System.Numerics;

namespace Tuna.LinearAlgebra;

/// <summary>
/// Represents a dense one-dimensional numerical vector.
/// <para>1 次元の密ベクトルを表します。</para>
/// </summary>
/// <typeparam name="T">The element type of the vector. / ベクトルの要素型。</typeparam>
/// <remarks>
/// <para>Arithmetic operators delegate to <see cref="XMath"/> and allocate a new vector for the result. Compound assignment operators (<c>+=</c>, <c>-=</c>, <c>*=</c>) mutate this vector in place.</para>
/// <para>算術演算子は <see cref="XMath"/> を呼び出し、結果を格納した新しいベクトルを返します。複合代入演算子（<c>+=</c>、<c>-=</c>、<c>*=</c>）はこのベクトルをインプレースで更新します。</para>
/// </remarks>
public sealed class Vector<T> where T : unmanaged, INumberBase<T>
{
    private readonly T[] data;

    /// <summary>
    /// Initializes a new zero-filled vector of the specified length.
    /// <para>指定した長さの零ベクトルを初期化します。</para>
    /// </summary>
    /// <param name="length">The number of elements. / 要素数。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is negative. / <paramref name="length"/> が負の場合。</exception>
    public Vector(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        data = DenseStorage.Allocate<T>(length);
    }

    /// <summary>
    /// Initializes a new vector by copying the specified values.
    /// <para>指定した値をコピーして新しいベクトルを初期化します。</para>
    /// </summary>
    /// <param name="values">The elements to copy. / コピー元の要素。</param>
    public Vector(ReadOnlySpan<T> values) => data = values.ToArray();

    private Vector(T[] values, bool share) => data = share ? values : (T[])values.Clone();

    /// <summary>
    /// Creates a new vector whose elements are computed from their indices using a stateless operator: <c>result[i] = TOperator(i)</c>.
    /// <para>状態を持たない演算子を用いて添字から各要素を計算した新しいベクトル（<c>result[i] = TOperator(i)</c>）を作成します。</para>
    /// </summary>
    /// <typeparam name="TOperator">The stateless index operator type. / 状態を持たない添字演算子の型。</typeparam>
    /// <param name="length">The number of elements. / 要素数。</param>
    /// <returns>A new vector populated by <typeparamref name="TOperator"/>. / <typeparamref name="TOperator"/> で生成された新しいベクトル。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is negative. / <paramref name="length"/> が負の場合。</exception>
    public static Vector<T> Generate<TOperator>(int length)
        where TOperator : IVectorIndexOperator<T> =>
        Generate(length, default(StaticVectorIndexOperator<T, TOperator>));

    /// <summary>
    /// Creates a new vector whose elements are computed from their indices using a stateful operator: <c>result[i] = op(i)</c>.
    /// <para>状態を持つ演算子を用いて添字から各要素を計算した新しいベクトル（<c>result[i] = op(i)</c>）を作成します。</para>
    /// </summary>
    /// <typeparam name="TOperator">The stateful index operator type. / 状態を持つ添字演算子の型。</typeparam>
    /// <param name="length">The number of elements. / 要素数。</param>
    /// <param name="op">The operator evaluated for each index. / 各添字に対して評価する演算子。</param>
    /// <returns>A new vector populated by <paramref name="op"/>. / <paramref name="op"/> で生成された新しいベクトル。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is negative. / <paramref name="length"/> が負の場合。</exception>
    public static Vector<T> Generate<TOperator>(int length, TOperator op)
        where TOperator : IStatefulVectorIndexOperator<T>
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        var result = Uninitialized(length);
        XMath.Generate(op, result.AsVectorSpan());
        return result;
    }

    /// <summary>
    /// Creates a new vector whose elements are computed from their indices using a delegate: <c>result[i] = op(i)</c>.
    /// <para>デリゲートを用いて添字から各要素を計算した新しいベクトル（<c>result[i] = op(i)</c>）を作成します。</para>
    /// </summary>
    /// <param name="length">The number of elements. / 要素数。</param>
    /// <param name="op">The delegate evaluated for each index. / 各添字に対して評価するデリゲート。</param>
    /// <returns>A new vector populated by <paramref name="op"/>. / <paramref name="op"/> で生成された新しいベクトル。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="op"/> is <see langword="null"/>. / <paramref name="op"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is negative. / <paramref name="length"/> が負の場合。</exception>
    public static Vector<T> Generate(int length, Func<int, T> op) =>
        Generate(length, new DelegateVectorIndexOperator<T>(op));

    /// <summary>
    /// Gets the number of elements in the vector.
    /// <para>ベクトルの要素数を取得します。</para>
    /// </summary>
    public int Length => data.Length;

    /// <summary>
    /// Gets or sets the element at the specified index.
    /// <para>指定した位置の要素を取得または設定します。</para>
    /// </summary>
    /// <param name="index">The zero-based element index. / 0 から始まる要素インデックス。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is outside the bounds of the vector. / <paramref name="index"/> が範囲外の場合。</exception>
    public T this[int index]
    {
        get
        {
            DenseStorage.CheckIndex(index, Length, nameof(index));
            return data[index];
        }
        set
        {
            DenseStorage.CheckIndex(index, Length, nameof(index));
            data[index] = value;
        }
    }

    /// <summary>
    /// Gets a zero-copy writable view of the specified subrange of elements.
    /// <para>指定した範囲の要素を参照する、コピーなしの書き込み可能な部分ベクトルビューを取得します。</para>
    /// </summary>
    /// <param name="range">The range of elements to slice. / 切り出す要素の範囲。</param>
    public VectorSpan<T> this[Range range] => AsVectorSpan()[range];

    /// <summary>
    /// Returns a <see cref="Span{T}"/> sharing the underlying contiguous storage.
    /// <para>内部の連続した格納領域を共有する <see cref="Span{T}"/> を返します。</para>
    /// </summary>
    /// <returns>A span over the vector's storage. / ベクトルの格納領域を参照するスパン。</returns>
    public Span<T> AsSpan() => data;

    /// <summary>
    /// Returns a writable <see cref="VectorSpan{T}"/> view over the entire vector.
    /// <para>ベクトル全体を参照する書き込み可能な <see cref="VectorSpan{T}"/> ビューを返します。</para>
    /// </summary>
    /// <returns>A writable view sharing the vector's storage. / 格納領域を共有する書き込み可能なビュー。</returns>
    public VectorSpan<T> AsVectorSpan() => new(data);

    /// <summary>
    /// Returns a zero-copy read-only matrix view that broadcasts this vector as <paramref name="count"/> columns.
    /// <para>このベクトルを <paramref name="count"/> 本の列としてブロードキャストした、コピーなしの読み取り専用行列ビューを返します。</para>
    /// </summary>
    /// <param name="count">The number of columns in the resulting matrix view. / 結果の行列ビューの列数。</param>
    /// <returns>A read-only <c>Length × count</c> matrix view. / <c>Length × count</c> の読み取り専用行列ビュー。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative. / <paramref name="count"/> が負の場合。</exception>
    public ReadOnlyMatrixSpan<T> Broadcast(int count) => AsVectorSpan().Broadcast(count);

    /// <summary>
    /// Returns a zero-copy read-only matrix view that broadcasts this vector as <paramref name="rows"/> rows.
    /// <para>このベクトルを <paramref name="rows"/> 本の行としてブロードキャストした、コピーなしの読み取り専用行列ビューを返します。</para>
    /// </summary>
    /// <param name="rows">The number of rows in the resulting matrix view. / 結果の行列ビューの行数。</param>
    /// <returns>A read-only <c>rows × Length</c> matrix view. / <c>rows × Length</c> の読み取り専用行列ビュー。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rows"/> is negative. / <paramref name="rows"/> が負の場合。</exception>
    public ReadOnlyMatrixSpan<T> BroadcastRows(int rows) => AsVectorSpan().BroadcastRows(rows);

    /// <summary>
    /// Copies the elements of the vector to a new array.
    /// <para>ベクトルの要素を新しい配列へコピーします。</para>
    /// </summary>
    /// <returns>A new array containing a copy of the elements. / 要素をコピーした新しい配列。</returns>
    public T[] ToArray() => (T[])data.Clone();

    /// <summary>
    /// Creates a deep copy of this vector with independent storage.
    /// <para>独立した格納領域を持つこのベクトルの複製を作成します。</para>
    /// </summary>
    /// <returns>A new vector containing the same elements. / 同じ要素を持つ新しいベクトル。</returns>
    public Vector<T> Clone() => new(data, share: false);

    /// <summary>
    /// Reinterprets this vector as a column-major matrix sharing the same underlying storage.
    /// <para>同じ格納領域を共有する列優先の行列としてこのベクトルを再解釈します。</para>
    /// </summary>
    /// <param name="rows">The number of rows. / 行数。</param>
    /// <param name="columns">The number of columns. / 列数。</param>
    /// <returns>A column-major matrix sharing this vector's storage. / このベクトルと格納領域を共有する列優先の行列。</returns>
    /// <remarks>
    /// <para><c>rows * columns</c> must equal <see cref="Length"/>.</para>
    /// <para><c>rows * columns</c> は <see cref="Length"/> と一致している必要があります。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rows"/> or <paramref name="columns"/> is negative. / <paramref name="rows"/> または <paramref name="columns"/> が負の場合。</exception>
    /// <exception cref="OverflowException"><c>rows * columns</c> exceeds <see cref="int.MaxValue"/>. / <c>rows * columns</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="ArgumentException"><c>rows * columns</c> does not equal <see cref="Length"/>. / <c>rows * columns</c> が <see cref="Length"/> と一致しない場合。</exception>
    public Matrix<T> AsMatrix(int rows, int columns) => Matrix<T>.Wrap(rows, columns, data);

    /// <summary>
    /// Copies the elements of this vector into a new column-major matrix with the specified dimensions.
    /// <para>このベクトルの要素を指定した形状の新しい列優先行列へコピーします。</para>
    /// </summary>
    /// <param name="rows">The number of rows. / 行数。</param>
    /// <param name="columns">The number of columns. / 列数。</param>
    /// <returns>A new column-major matrix containing the copied elements. / 要素をコピーした新しい列優先行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rows"/> or <paramref name="columns"/> is negative. / <paramref name="rows"/> または <paramref name="columns"/> が負の場合。</exception>
    /// <exception cref="OverflowException"><c>rows * columns</c> exceeds <see cref="int.MaxValue"/>. / <c>rows * columns</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="ArgumentException"><c>rows * columns</c> does not equal <see cref="Length"/>. / <c>rows * columns</c> が <see cref="Length"/> と一致しない場合。</exception>
    public Matrix<T> ToMatrix(int rows, int columns) => AsMatrix(rows, columns).Clone();

    /// <summary>
    /// Converts each element to <typeparamref name="TTo"/> using checked numeric conversion (<see cref="INumberBase{TSelf}.CreateChecked{TOther}(TOther)"/>).
    /// <para>チェック付きの数値変換（<see cref="INumberBase{TSelf}.CreateChecked{TOther}(TOther)"/>）を用いて各要素を <typeparamref name="TTo"/> に変換します。</para>
    /// </summary>
    /// <typeparam name="TTo">The target element type. / 変換先の要素型。</typeparam>
    /// <returns>A new vector of the same length containing the converted elements. / 変換後の要素を格納した同じ長さの新しいベクトル。</returns>
    /// <exception cref="NotSupportedException">Conversion between <typeparamref name="T"/> and <typeparamref name="TTo"/> is not supported. / <typeparamref name="T"/> から <typeparamref name="TTo"/> への変換に対応していない場合。</exception>
    /// <exception cref="OverflowException">An element cannot be represented in <typeparamref name="TTo"/>. / いずれかの要素が <typeparamref name="TTo"/> で表現できない場合。</exception>
    public Vector<TTo> ConvertChecked<TTo>() where TTo : unmanaged, INumberBase<TTo> =>
        Vector<TTo>.Wrap(DenseStorage.ConvertChecked<T, TTo>(data));

    /// <summary>
    /// Implicitly converts a <see cref="Vector{T}"/> to a writable <see cref="VectorSpan{T}"/> view.
    /// <para><see cref="Vector{T}"/> を書き込み可能な <see cref="VectorSpan{T}"/> ビューへ暗黙的に変換します。</para>
    /// </summary>
    /// <param name="vector">The vector to view. / 対象のベクトル。</param>
    /// <returns>A writable view over <paramref name="vector"/>. / <paramref name="vector"/> を参照する書き込み可能なビュー。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="vector"/> is <see langword="null"/>. / <paramref name="vector"/> が <see langword="null"/> の場合。</exception>
    public static implicit operator VectorSpan<T>(Vector<T> vector) =>
        (vector ?? throw new ArgumentNullException(nameof(vector))).AsVectorSpan();

    /// <summary>
    /// Implicitly converts a <see cref="Vector{T}"/> to a read-only <see cref="ReadOnlyVectorSpan{T}"/> view.
    /// <para><see cref="Vector{T}"/> を読み取り専用の <see cref="ReadOnlyVectorSpan{T}"/> ビューへ暗黙的に変換します。</para>
    /// </summary>
    /// <param name="vector">The vector to view. / 対象のベクトル。</param>
    /// <returns>A read-only view over <paramref name="vector"/>. / <paramref name="vector"/> を参照する読み取り専用ビュー。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="vector"/> is <see langword="null"/>. / <paramref name="vector"/> が <see langword="null"/> の場合。</exception>
    public static implicit operator ReadOnlyVectorSpan<T>(Vector<T> vector) =>
        (vector ?? throw new ArgumentNullException(nameof(vector))).AsVectorSpan();

    /// <summary>
    /// Computes the elementwise sum of two vectors.
    /// <para>2 つのベクトルの要素ごとの和を計算します。</para>
    /// </summary>
    /// <param name="x">The left-hand vector. / 左辺のベクトル。</param>
    /// <param name="y">The right-hand vector. / 右辺のベクトル。</param>
    /// <returns>A new vector containing <c>x + y</c>. / <c>x + y</c> を格納した新しいベクトル。</returns>
    public static Vector<T> operator +(Vector<T> x, Vector<T> y) => XMath.Add(x, y);

    /// <summary>
    /// Computes the elementwise difference of two vectors.
    /// <para>2 つのベクトルの要素ごとの差を計算します。</para>
    /// </summary>
    /// <param name="x">The left-hand vector. / 左辺のベクトル。</param>
    /// <param name="y">The right-hand vector. / 右辺のベクトル。</param>
    /// <returns>A new vector containing <c>x - y</c>. / <c>x - y</c> を格納した新しいベクトル。</returns>
    public static Vector<T> operator -(Vector<T> x, Vector<T> y) => XMath.Subtract(x, y);

    /// <summary>
    /// Negates each element of the vector.
    /// <para>ベクトルの各要素の符号を反転します。</para>
    /// </summary>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>A new vector containing <c>-x</c>. / <c>-x</c> を格納した新しいベクトル。</returns>
    public static Vector<T> operator -(Vector<T> x) => XMath.Negate(x);

    /// <summary>
    /// Multiplies a vector by a scalar.
    /// <para>ベクトルをスカラー倍します。</para>
    /// </summary>
    /// <param name="scalar">The scalar multiplier. / 掛けるスカラー値。</param>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>A new vector containing <c>scalar * x</c>. / <c>scalar * x</c> を格納した新しいベクトル。</returns>
    public static Vector<T> operator *(T scalar, Vector<T> x) => XMath.Multiply(x, scalar);

    /// <summary>
    /// Multiplies a vector by a scalar.
    /// <para>ベクトルをスカラー倍します。</para>
    /// </summary>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="scalar">The scalar multiplier. / 掛けるスカラー値。</param>
    /// <returns>A new vector containing <c>x * scalar</c>. / <c>x * scalar</c> を格納した新しいベクトル。</returns>
    public static Vector<T> operator *(Vector<T> x, T scalar) => XMath.Multiply(x, scalar);

    /// <summary>
    /// Divides each element of a vector by a scalar.
    /// <para>ベクトルの各要素をスカラーで割ります。</para>
    /// </summary>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="scalar">The scalar divisor. / 除数。</param>
    /// <returns>A new vector containing the quotients. / 各要素の商を格納した新しいベクトル。</returns>
    public static Vector<T> operator /(Vector<T> x, T scalar) => XMath.Divide(x, scalar);

    /// <summary>
    /// Computes the matrix-vector product <c>a * x</c>.
    /// <para>行列ベクトル積 <c>a * x</c> を計算します。</para>
    /// </summary>
    /// <param name="a">The input matrix (container or view). / 入力行列（コンテナまたはビュー）。</param>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>A new vector containing <c>a * x</c>. / <c>a * x</c> を格納した新しいベクトル。</returns>
    public static Vector<T> operator *(ReadOnlyMatrixSpan<T> a, Vector<T> x) => XMath.Multiply(a, x);

    /// <summary>
    /// Adds the elements of <paramref name="other"/> to this vector in place.
    /// <para>このベクトルに <paramref name="other"/> をインプレースで加算します。</para>
    /// </summary>
    /// <param name="other">The vector to add. / 加算するベクトル。</param>
    public void operator +=(ReadOnlyVectorSpan<T> other) => XMath.Add(this, other, this);

    /// <summary>
    /// Subtracts the elements of <paramref name="other"/> from this vector in place.
    /// <para>このベクトルから <paramref name="other"/> をインプレースで減算します。</para>
    /// </summary>
    /// <param name="other">The vector to subtract. / 減算するベクトル。</param>
    public void operator -=(ReadOnlyVectorSpan<T> other) => XMath.Subtract(this, other, this);

    /// <summary>
    /// Scales this vector by <paramref name="scalar"/> in place.
    /// <para>このベクトルを <paramref name="scalar"/> でインプレースにスカラー倍します。</para>
    /// </summary>
    /// <param name="scalar">The scalar multiplier. / 掛けるスカラー値。</param>
    public void operator *=(T scalar) => XMath.InPlace.Multiply(this, scalar);

    /// <summary>
    /// Divides each element of this vector by a scalar in place.
    /// <para>このベクトルの各要素をインプレースでスカラー除算します。</para>
    /// </summary>
    /// <param name="scalar">The scalar divisor. / 除数。</param>
    public void operator /=(T scalar) => XMath.InPlace.Divide(this, scalar);

    /// <summary>
    /// Explicitly converts an array to a new <see cref="Vector{T}"/> by copying its elements.
    /// <para>配列の要素をコピーして新しい <see cref="Vector{T}"/> へ明示的に変換します。</para>
    /// </summary>
    /// <param name="values">The source array. / コピー元の配列。</param>
    /// <returns>A new vector containing the copied elements. / 要素をコピーした新しいベクトル。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="values"/> is <see langword="null"/>. / <paramref name="values"/> が <see langword="null"/> の場合。</exception>
    public static explicit operator Vector<T>(T[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        return new(values);
    }

    /// <summary>
    /// Explicitly converts a <see cref="Vector{T}"/> to a new array by copying its elements.
    /// <para><see cref="Vector{T}"/> の要素をコピーして新しい配列へ明示的に変換します。</para>
    /// </summary>
    /// <param name="vector">The source vector. / コピー元のベクトル。</param>
    /// <returns>A new array containing the copied elements. / 要素をコピーした新しい配列。</returns>
    public static explicit operator T[](Vector<T> vector) => vector.ToArray();

    internal static Vector<T> Wrap(T[] data) => new(data, share: true);

    // Allocates uninitialized storage for kernels that overwrite every element (not zero-filled).
    internal static Vector<T> Uninitialized(int length) => new(GC.AllocateUninitializedArray<T>(length), share: true);
}
