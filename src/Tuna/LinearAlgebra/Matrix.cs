using System.Globalization;
using System.Numerics;
using System.Text;

namespace Tuna.LinearAlgebra;

/// <summary>
/// Represents a dense column-major two-dimensional numerical matrix.
/// <para>列優先レイアウトの 2 次元密行列を表します。</para>
/// </summary>
/// <typeparam name="TElement">The element type of the matrix. / 行列の要素型。</typeparam>
/// <remarks>
/// <para>Elements are stored in column-major order at index <c>row + column * Rows</c>. Constructors, <c>To*</c> methods, and <see cref="Clone"/> copy elements, whereas <c>As*</c> methods and span views share the underlying storage.</para>
/// <para>Arithmetic operators delegate to <see cref="XMath"/> and allocate a new matrix. Compound assignment operators (<c>+=</c>, <c>-=</c>, <c>*=</c>) mutate this matrix in place.</para>
/// <para>要素は列優先順（<c>row + column * Rows</c>）で格納されます。コンストラクター・<c>To*</c> メソッド・<see cref="Clone"/> は要素をコピーし、<c>As*</c> メソッドおよびビューは格納領域を共有します。</para>
/// <para>算術演算子は <see cref="XMath"/> を呼び出して新しい行列を返し、複合代入演算子（<c>+=</c>、<c>-=</c>、<c>*=</c>）はこの行列をインプレースで更新します。</para>
/// </remarks>
public sealed class Matrix<TElement> where TElement : unmanaged, INumberBase<TElement>
{
    private readonly TElement[] data;

    /// <summary>
    /// Initializes a new zero-filled matrix with the specified dimensions.
    /// <para>指定した行数と列数の零行列を初期化します。</para>
    /// </summary>
    /// <param name="rows">The number of rows. / 行数。</param>
    /// <param name="columns">The number of columns. / 列数。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rows"/> or <paramref name="columns"/> is negative. / <paramref name="rows"/> または <paramref name="columns"/> が負の場合。</exception>
    /// <exception cref="OverflowException"><c>rows * columns</c> exceeds <see cref="int.MaxValue"/>. / 要素数 <c>rows * columns</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    public Matrix(int rows, int columns)
    {
        data = DenseStorage.Allocate<TElement>(DenseStorage.Length(rows, columns));
        Rows = rows;
        Columns = columns;
    }

    /// <summary>
    /// Initializes a new matrix by copying the elements of a two-dimensional array.
    /// <para>2 次元配列の要素をコピーして新しい行列を初期化します。</para>
    /// </summary>
    /// <param name="values">The source two-dimensional array. / コピー元の 2 次元配列。</param>
    /// <exception cref="ArgumentNullException"><paramref name="values"/> is <see langword="null"/>. / <paramref name="values"/> が <see langword="null"/> の場合。</exception>
    public Matrix(TElement[,] values) : this(
        (values ?? throw new ArgumentNullException(nameof(values))).GetLength(0), values.GetLength(1))
    {
        for (int j = 0; j < Columns; j++)
        {
            for (int i = 0; i < Rows; i++)
            {
                data[i + j * Rows] = values[i, j];
            }
        }
    }

    private Matrix(int rows, int columns, TElement[] storage)
    {
        DenseStorage.RequireLength(DenseStorage.Length(rows, columns), storage.Length);
        Rows = rows;
        Columns = columns;
        data = storage;
    }

    /// <summary>
    /// Gets the number of rows in the matrix.
    /// <para>行列の行数を取得します。</para>
    /// </summary>
    public int Rows { get; }

    /// <summary>
    /// Gets the number of columns in the matrix.
    /// <para>行列の列数を取得します。</para>
    /// </summary>
    public int Columns { get; }

    /// <summary>
    /// Gets the total number of elements (<c>Rows * Columns</c>).
    /// <para>総要素数（<c>Rows * Columns</c>）を取得します。</para>
    /// </summary>
    public int Length => data.Length;

    /// <summary>
    /// Gets a value indicating whether the matrix is square (<c>Rows == Columns</c>).
    /// <para>行列が正方行列（<c>Rows == Columns</c>）かどうかを示す値を取得します。</para>
    /// </summary>
    public bool IsSquare => Rows == Columns;

    /// <summary>
    /// Gets or sets the element at the specified row and column.
    /// <para>指定した行および列の要素を取得または設定します。</para>
    /// </summary>
    /// <param name="row">The zero-based row index. / 0 から始まる行インデックス。</param>
    /// <param name="column">The zero-based column index. / 0 から始まる列インデックス。</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="row"/> or <paramref name="column"/> is out of range. / <paramref name="row"/> または <paramref name="column"/> が範囲外の場合。</exception>
    public TElement this[int row, int column]
    {
        get
        {
            CheckIndex(row, column);
            return data[row + column * Rows];
        }
        set
        {
            CheckIndex(row, column);
            data[row + column * Rows] = value;
        }
    }

    /// <summary>
    /// Gets a zero-copy writable view of the specified submatrix.
    /// <para>指定した範囲の部分行列を参照する、コピーなしの書き込み可能なビューを取得します。</para>
    /// </summary>
    /// <param name="rows">The range of rows to slice. / 切り出す行の範囲。</param>
    /// <param name="columns">The range of columns to slice. / 切り出す列の範囲。</param>
    public MatrixSpan<TElement> this[Range rows, Range columns] => AsMatrixSpan()[rows, columns];

    /// <summary>
    /// Creates a new matrix by copying elements stored in column-major order.
    /// <para>列優先順で格納された要素をコピーして新しい行列を作成します。</para>
    /// </summary>
    /// <param name="rows">The number of rows. / 行数。</param>
    /// <param name="columns">The number of columns. / 列数。</param>
    /// <param name="values">The column-major source values of length <c>rows * columns</c>. / 長さ <c>rows * columns</c> の列優先入力値。</param>
    /// <returns>A new matrix containing the copied elements. / 要素をコピーした新しい行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rows"/> or <paramref name="columns"/> is negative. / <paramref name="rows"/> または <paramref name="columns"/> が負の場合。</exception>
    /// <exception cref="OverflowException"><c>rows * columns</c> exceeds <see cref="int.MaxValue"/>. / 要素数 <c>rows * columns</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="values"/> does not have length <c>rows * columns</c>. / <paramref name="values"/> の長さが <c>rows * columns</c> と一致しない場合。</exception>
    public static Matrix<TElement> FromColumnMajor(int rows, int columns, ReadOnlySpan<TElement> values)
    {
        DenseStorage.RequireLength(DenseStorage.Length(rows, columns), values.Length);
        return new(rows, columns, values.ToArray());
    }

    /// <summary>
    /// Creates a new column-major matrix by copying and reordering elements stored in row-major order.
    /// <para>行優先順で格納された要素を列優先順に並べ替えてコピーし、新しい行列を作成します。</para>
    /// </summary>
    /// <param name="rows">The number of rows. / 行数。</param>
    /// <param name="columns">The number of columns. / 列数。</param>
    /// <param name="values">The row-major source values of length <c>rows * columns</c>. / 長さ <c>rows * columns</c> の行優先入力値。</param>
    /// <returns>A new column-major matrix containing the copied elements. / 要素をコピーした新しい列優先行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rows"/> or <paramref name="columns"/> is negative. / <paramref name="rows"/> または <paramref name="columns"/> が負の場合。</exception>
    /// <exception cref="OverflowException"><c>rows * columns</c> exceeds <see cref="int.MaxValue"/>. / 要素数 <c>rows * columns</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="values"/> does not have length <c>rows * columns</c>. / <paramref name="values"/> の長さが <c>rows * columns</c> と一致しない場合。</exception>
    public static Matrix<TElement> FromRowMajor(int rows, int columns, ReadOnlySpan<TElement> values)
    {
        DenseStorage.RequireLength(DenseStorage.Length(rows, columns), values.Length);
        var result = new Matrix<TElement>(rows, columns);
        for (int j = 0; j < columns; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                result.data[i + j * rows] = values[i * columns + j];
            }
        }

        return result;
    }

    /// <summary>
    /// Creates a square diagonal matrix with the specified diagonal entries.
    /// <para>指定した対角成分を持つ正方対角行列を作成します。</para>
    /// </summary>
    /// <param name="values">The diagonal elements. / 対角成分の値。</param>
    /// <returns>A new square diagonal matrix. / 新しい対角行列。</returns>
    /// <exception cref="OverflowException"><c>values.Length * values.Length</c> exceeds <see cref="int.MaxValue"/>. / 要素数 <c>values.Length²</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static Matrix<TElement> FromDiagonal(ReadOnlySpan<TElement> values)
    {
        var result = new Matrix<TElement>(values.Length, values.Length);
        for (int i = 0; i < values.Length; i++)
        {
            result[i, i] = values[i];
        }

        return result;
    }

    /// <summary>
    /// Creates an identity matrix of the specified order.
    /// <para>指定した次数の単位行列を作成します。</para>
    /// </summary>
    /// <param name="size">The order of the identity matrix. / 単位行列の次数。</param>
    /// <returns>A new <paramref name="size"/> × <paramref name="size"/> identity matrix. / <paramref name="size"/> × <paramref name="size"/> の新しい単位行列。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="size"/> is negative. / <paramref name="size"/> が負の場合。</exception>
    /// <exception cref="OverflowException"><c>size * size</c> exceeds <see cref="int.MaxValue"/>. / 要素数 <c>size²</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static Matrix<TElement> Identity(int size)
    {
        var result = new Matrix<TElement>(size, size);
        result.Diagonal().Fill(TElement.One);
        return result;
    }

    /// <summary>
    /// Creates a new matrix whose elements are computed from their indices using a stateless operator: <c>result[i, j] = TOperator(i, j)</c>.
    /// <para>状態を持たない演算子を用いて添字から各要素を計算した新しい行列（<c>result[i, j] = TOperator(i, j)</c>）を作成します。</para>
    /// </summary>
    /// <typeparam name="TOperator">The stateless index operator type. / 状態を持たない添字演算子の型。</typeparam>
    /// <param name="rows">The number of rows. / 行数。</param>
    /// <param name="columns">The number of columns. / 列数。</param>
    /// <returns>A new matrix populated by <typeparamref name="TOperator"/>. / <typeparamref name="TOperator"/> で生成された新しい行列。</returns>
    /// <remarks>
    /// <para>Large matrices are evaluated in parallel across multiple threads; each element is evaluated independently once. To write into an existing buffer, use <see cref="XMath.Generate{T, TOperator}(MatrixSpan{T})"/>.</para>
    /// <para>大きな行列は複数スレッドで並列評価され、各要素は独立に 1 回ずつ計算されます。既存の出力先に書き込む場合は <see cref="XMath.Generate{T, TOperator}(MatrixSpan{T})"/> を使用してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rows"/> or <paramref name="columns"/> is negative. / <paramref name="rows"/> または <paramref name="columns"/> が負の場合。</exception>
    /// <exception cref="OverflowException"><c>rows * columns</c> exceeds <see cref="int.MaxValue"/>. / 要素数 <c>rows * columns</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static Matrix<TElement> Generate<TOperator>(int rows, int columns)
        where TOperator : IIndexOperator<TElement>
    {
        var result = Uninitialized(rows, columns);
        XMath.Generate<TElement, TOperator>(result);
        return result;
    }

    /// <summary>
    /// Creates a new matrix whose elements are computed from their indices using a stateful operator: <c>result[i, j] = op(i, j)</c>.
    /// <para>状態を持つ演算子を用いて添字から各要素を計算した新しい行列（<c>result[i, j] = op(i, j)</c>）を作成します。</para>
    /// </summary>
    /// <typeparam name="TOperator">The stateful index operator type. / 状態を持つ添字演算子の型。</typeparam>
    /// <param name="rows">The number of rows. / 行数。</param>
    /// <param name="columns">The number of columns. / 列数。</param>
    /// <param name="op">The operator evaluated for each index pair. / 各添字ペアに対して評価する演算子。</param>
    /// <returns>A new matrix populated by <paramref name="op"/>. / <paramref name="op"/> で生成された新しい行列。</returns>
    /// <remarks>
    /// <para>To write into an existing buffer, use <see cref="XMath.Generate{T, TOperator}(TOperator, MatrixSpan{T})"/>.</para>
    /// <para>既存の出力先に書き込む場合は <see cref="XMath.Generate{T, TOperator}(TOperator, MatrixSpan{T})"/> を使用してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rows"/> or <paramref name="columns"/> is negative. / <paramref name="rows"/> または <paramref name="columns"/> が負の場合。</exception>
    /// <exception cref="OverflowException"><c>rows * columns</c> exceeds <see cref="int.MaxValue"/>. / 要素数 <c>rows * columns</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static Matrix<TElement> Generate<TOperator>(int rows, int columns, TOperator op)
        where TOperator : IStatefulIndexOperator<TElement>
    {
        var result = Uninitialized(rows, columns);
        XMath.Generate(op, result.AsMatrixSpan());
        return result;
    }

    /// <summary>
    /// Creates a new matrix whose elements are computed from their indices using a delegate: <c>result[i, j] = op(i, j)</c>.
    /// <para>デリゲートを用いて添字から各要素を計算した新しい行列（<c>result[i, j] = op(i, j)</c>）を作成します。</para>
    /// </summary>
    /// <param name="rows">The number of rows. / 行数。</param>
    /// <param name="columns">The number of columns. / 列数。</param>
    /// <param name="op">The delegate evaluated for each index pair. / 各添字ペアに対して評価するデリゲート。</param>
    /// <returns>A new matrix populated by <paramref name="op"/>. / <paramref name="op"/> で生成された新しい行列。</returns>
    /// <remarks>
    /// <para>For large matrices, <paramref name="op"/> may be invoked concurrently from multiple threads in arbitrary order.</para>
    /// <para>大きな行列では、<paramref name="op"/> が複数スレッドから順不同で同時に呼び出される場合があります。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="op"/> is <see langword="null"/>. / <paramref name="op"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rows"/> or <paramref name="columns"/> is negative. / <paramref name="rows"/> または <paramref name="columns"/> が負の場合。</exception>
    /// <exception cref="OverflowException"><c>rows * columns</c> exceeds <see cref="int.MaxValue"/>. / 要素数 <c>rows * columns</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static Matrix<TElement> Generate(int rows, int columns, Func<int, int, TElement> op) =>
        Generate(rows, columns, new DelegateIndexOperator<TElement>(op));

    /// <summary>
    /// Creates a dense band matrix by evaluating <c>TOperator(i, j)</c> for <c>-lowerBandwidth &lt;= j - i &lt;= upperBandwidth</c> and setting elements outside the band to zero.
    /// <para><c>-lowerBandwidth &lt;= j - i &lt;= upperBandwidth</c> の範囲を状態のない演算子 <c>TOperator(i, j)</c> で計算し、帯外の要素を 0 とする密行列を作成します。</para>
    /// </summary>
    /// <typeparam name="TOperator">The stateless index operator type. / 状態を持たない添字演算子の型。</typeparam>
    /// <param name="rows">The number of rows. / 行数。</param>
    /// <param name="columns">The number of columns. / 列数。</param>
    /// <param name="lowerBandwidth">The number of subdiagonals below the main diagonal. / 主対角線より下の副対角線の本数。</param>
    /// <param name="upperBandwidth">The number of superdiagonals above the main diagonal. / 主対角線より上の副対角線の本数。</param>
    /// <returns>A new dense matrix containing the generated band values. / 帯行列の値を格納した新しい密行列。</returns>
    /// <remarks>
    /// <para>Only elements inside the band are evaluated. To write into an existing buffer, use <see cref="XMath.GenerateBand{T, TOperator}(int, int, MatrixSpan{T})"/>.</para>
    /// <para>帯の内部の要素のみを評価します。既存の出力先に書き込む場合は <see cref="XMath.GenerateBand{T, TOperator}(int, int, MatrixSpan{T})"/> を使用してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rows"/>, <paramref name="columns"/>, <paramref name="lowerBandwidth"/>, or <paramref name="upperBandwidth"/> is negative. / 行数、列数、または帯幅が負の場合。</exception>
    /// <exception cref="OverflowException"><c>rows * columns</c> exceeds <see cref="int.MaxValue"/>. / 要素数 <c>rows * columns</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static Matrix<TElement> GenerateBand<TOperator>(int rows, int columns, int lowerBandwidth, int upperBandwidth)
        where TOperator : IIndexOperator<TElement>
    {
        ArgumentOutOfRangeException.ThrowIfNegative(lowerBandwidth);
        ArgumentOutOfRangeException.ThrowIfNegative(upperBandwidth);
        var result = Uninitialized(rows, columns);
        XMath.GenerateBand<TElement, TOperator>(lowerBandwidth, upperBandwidth, result);
        return result;
    }

    /// <summary>
    /// Creates a dense band matrix by evaluating <c>op(i, j)</c> for <c>-lowerBandwidth &lt;= j - i &lt;= upperBandwidth</c> and setting elements outside the band to zero.
    /// <para><c>-lowerBandwidth &lt;= j - i &lt;= upperBandwidth</c> の範囲を状態を持つ演算子 <c>op(i, j)</c> で計算し、帯外の要素を 0 とする密行列を作成します。</para>
    /// </summary>
    /// <typeparam name="TOperator">The stateful index operator type. / 状態を持つ添字演算子の型。</typeparam>
    /// <param name="rows">The number of rows. / 行数。</param>
    /// <param name="columns">The number of columns. / 列数。</param>
    /// <param name="lowerBandwidth">The number of subdiagonals below the main diagonal. / 主対角線より下の副対角線の本数。</param>
    /// <param name="upperBandwidth">The number of superdiagonals above the main diagonal. / 主対角線より上の副対角線の本数。</param>
    /// <param name="op">The operator evaluated for each index pair inside the band. / 帯内の各添字ペアに対して評価する演算子。</param>
    /// <returns>A new dense matrix containing the generated band values. / 帯行列の値を格納した新しい密行列。</returns>
    /// <remarks>
    /// <para>To write into an existing buffer, use <see cref="XMath.GenerateBand{T, TOperator}(int, int, TOperator, MatrixSpan{T})"/>.</para>
    /// <para>既存の出力先に書き込む場合は <see cref="XMath.GenerateBand{T, TOperator}(int, int, TOperator, MatrixSpan{T})"/> を使用してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rows"/>, <paramref name="columns"/>, <paramref name="lowerBandwidth"/>, or <paramref name="upperBandwidth"/> is negative. / 行数、列数、または帯幅が負の場合。</exception>
    /// <exception cref="OverflowException"><c>rows * columns</c> exceeds <see cref="int.MaxValue"/>. / 要素数 <c>rows * columns</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static Matrix<TElement> GenerateBand<TOperator>(int rows, int columns, int lowerBandwidth, int upperBandwidth, TOperator op)
        where TOperator : IStatefulIndexOperator<TElement>
    {
        ArgumentOutOfRangeException.ThrowIfNegative(lowerBandwidth);
        ArgumentOutOfRangeException.ThrowIfNegative(upperBandwidth);
        var result = Uninitialized(rows, columns);
        XMath.GenerateBand(lowerBandwidth, upperBandwidth, op, result.AsMatrixSpan());
        return result;
    }

    /// <summary>
    /// Creates a dense band matrix by evaluating <paramref name="op"/> inside the band and setting elements outside the band to zero.
    /// <para>帯内の要素をデリゲート <paramref name="op"/> で計算し、帯外の要素を 0 とする密行列を作成します。</para>
    /// </summary>
    /// <param name="rows">The number of rows. / 行数。</param>
    /// <param name="columns">The number of columns. / 列数。</param>
    /// <param name="lowerBandwidth">The number of subdiagonals below the main diagonal. / 主対角線より下の副対角線の本数。</param>
    /// <param name="upperBandwidth">The number of superdiagonals above the main diagonal. / 主対角線より上の副対角線の本数。</param>
    /// <param name="op">The delegate evaluated for each index pair inside the band. / 帯内の各添字ペアに対して評価するデリゲート。</param>
    /// <returns>A new dense matrix containing the generated band values. / 帯行列の値を格納した新しい密行列。</returns>
    /// <remarks>
    /// <para>For large matrices, <paramref name="op"/> may be invoked concurrently from multiple threads in arbitrary order.</para>
    /// <para>大きな行列では、<paramref name="op"/> が複数スレッドから順不同で同時に呼び出される場合があります。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="op"/> is <see langword="null"/>. / <paramref name="op"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rows"/>, <paramref name="columns"/>, <paramref name="lowerBandwidth"/>, or <paramref name="upperBandwidth"/> is negative. / 行数、列数、または帯幅が負の場合。</exception>
    /// <exception cref="OverflowException"><c>rows * columns</c> exceeds <see cref="int.MaxValue"/>. / 要素数 <c>rows * columns</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static Matrix<TElement> GenerateBand(int rows, int columns, int lowerBandwidth, int upperBandwidth, Func<int, int, TElement> op) =>
        GenerateBand(rows, columns, lowerBandwidth, upperBandwidth, new DelegateIndexOperator<TElement>(op));

    /// <summary>
    /// Creates a symmetric matrix by evaluating a stateless operator <c>TOperator(i, j)</c> for the lower triangle (<c>i &gt;= j</c>) and mirroring it to the upper triangle.
    /// <para>下三角部分（<c>i &gt;= j</c>）を状態のない演算子 <c>TOperator(i, j)</c> で計算し、上三角部分へ対称にコピーした新しい対称行列を作成します。</para>
    /// </summary>
    /// <typeparam name="TOperator">The stateless index operator type. / 状態を持たない添字演算子の型。</typeparam>
    /// <param name="order">The order of the square matrix. / 正方行列の次数。</param>
    /// <returns>A new symmetric matrix. / 新しい対称行列。</returns>
    /// <remarks>
    /// <para>To write into an existing buffer, use <see cref="XMath.GenerateSymmetric{T, TOperator}(MatrixSpan{T})"/>.</para>
    /// <para>既存の出力先に書き込む場合は <see cref="XMath.GenerateSymmetric{T, TOperator}(MatrixSpan{T})"/> を使用してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="order"/> is negative. / <paramref name="order"/> が負の場合。</exception>
    /// <exception cref="OverflowException"><c>order * order</c> exceeds <see cref="int.MaxValue"/>. / 要素数 <c>order²</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static Matrix<TElement> GenerateSymmetric<TOperator>(int order)
        where TOperator : IIndexOperator<TElement>
    {
        var result = Uninitialized(order, order);
        XMath.GenerateSymmetric<TElement, TOperator>(result);
        return result;
    }

    /// <summary>
    /// Creates a symmetric matrix by evaluating a stateful operator <c>op(i, j)</c> for the lower triangle (<c>i &gt;= j</c>) and mirroring it to the upper triangle.
    /// <para>下三角部分（<c>i &gt;= j</c>）を状態を持つ演算子 <c>op(i, j)</c> で計算し、上三角部分へ対称にコピーした新しい対称行列を作成します。</para>
    /// </summary>
    /// <typeparam name="TOperator">The stateful index operator type. / 状態を持つ添字演算子の型。</typeparam>
    /// <param name="order">The order of the square matrix. / 正方行列の次数。</param>
    /// <param name="op">The operator evaluated for each lower-triangular index pair. / 下三角の各添字ペアに対して評価する演算子。</param>
    /// <returns>A new symmetric matrix. / 新しい対称行列。</returns>
    /// <remarks>
    /// <para>To write into an existing buffer, use <see cref="XMath.GenerateSymmetric{T, TOperator}(TOperator, MatrixSpan{T})"/>.</para>
    /// <para>既存の出力先に書き込む場合は <see cref="XMath.GenerateSymmetric{T, TOperator}(TOperator, MatrixSpan{T})"/> を使用してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="order"/> is negative. / <paramref name="order"/> が負の場合。</exception>
    /// <exception cref="OverflowException"><c>order * order</c> exceeds <see cref="int.MaxValue"/>. / 要素数 <c>order²</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static Matrix<TElement> GenerateSymmetric<TOperator>(int order, TOperator op)
        where TOperator : IStatefulIndexOperator<TElement>
    {
        var result = Uninitialized(order, order);
        XMath.GenerateSymmetric(op, result.AsMatrixSpan());
        return result;
    }

    /// <summary>
    /// Creates a symmetric matrix by evaluating <paramref name="op"/> for the lower triangle (<c>i &gt;= j</c>) and mirroring it to the upper triangle.
    /// <para>下三角部分（<c>i &gt;= j</c>）をデリゲート <paramref name="op"/> で計算し、上三角部分へ対称にコピーした新しい対称行列を作成します。</para>
    /// </summary>
    /// <param name="order">The order of the square matrix. / 正方行列の次数。</param>
    /// <param name="op">The delegate evaluated for each lower-triangular index pair. / 下三角の各添字ペアに対して評価するデリゲート。</param>
    /// <returns>A new symmetric matrix. / 新しい対称行列。</returns>
    /// <remarks>
    /// <para>For large matrices, <paramref name="op"/> may be invoked concurrently from multiple threads in arbitrary order.</para>
    /// <para>大きな行列では、<paramref name="op"/> が複数スレッドから順不同で同時に呼び出される場合があります。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="op"/> is <see langword="null"/>. / <paramref name="op"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="order"/> is negative. / <paramref name="order"/> が負の場合。</exception>
    /// <exception cref="OverflowException"><c>order * order</c> exceeds <see cref="int.MaxValue"/>. / 要素数 <c>order²</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static Matrix<TElement> GenerateSymmetric(int order, Func<int, int, TElement> op) =>
        GenerateSymmetric(order, new DelegateIndexOperator<TElement>(op));

    /// <summary>
    /// Creates a symmetric band matrix by evaluating <c>TOperator(i, j)</c> for <c>0 &lt;= i - j &lt;= bandwidth</c>, mirroring to the upper triangle, and zeroing elements outside the band.
    /// <para><c>0 &lt;= i - j &lt;= bandwidth</c> の範囲を状態のない演算子 <c>TOperator(i, j)</c> で計算して上三角へ対称にコピーし、帯外を 0 とする新しい対称帯行列を作成します。</para>
    /// </summary>
    /// <typeparam name="TOperator">The stateless index operator type. / 状態を持たない添字演算子の型。</typeparam>
    /// <param name="order">The order of the square matrix. / 正方行列の次数。</param>
    /// <param name="bandwidth">The half-bandwidth (number of subdiagonals). / 帯幅（主対角線より下の副対角線の本数）。</param>
    /// <returns>A new symmetric band matrix. / 新しい対称帯行列。</returns>
    /// <remarks>
    /// <para>To write into an existing buffer, use <see cref="XMath.GenerateSymmetric{T, TOperator}(int, MatrixSpan{T})"/>.</para>
    /// <para>既存の出力先に書き込む場合は <see cref="XMath.GenerateSymmetric{T, TOperator}(int, MatrixSpan{T})"/> を使用してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="order"/> or <paramref name="bandwidth"/> is negative. / <paramref name="order"/> または <paramref name="bandwidth"/> が負の場合。</exception>
    /// <exception cref="OverflowException"><c>order * order</c> exceeds <see cref="int.MaxValue"/>. / 要素数 <c>order²</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static Matrix<TElement> GenerateSymmetric<TOperator>(int order, int bandwidth)
        where TOperator : IIndexOperator<TElement>
    {
        ArgumentOutOfRangeException.ThrowIfNegative(bandwidth);
        var result = Uninitialized(order, order);
        XMath.GenerateSymmetric<TElement, TOperator>(bandwidth, result);
        return result;
    }

    /// <summary>
    /// Creates a symmetric band matrix by evaluating <c>op(i, j)</c> for <c>0 &lt;= i - j &lt;= bandwidth</c>, mirroring to the upper triangle, and zeroing elements outside the band.
    /// <para><c>0 &lt;= i - j &lt;= bandwidth</c> の範囲を状態を持つ演算子 <c>op(i, j)</c> で計算して上三角へ対称にコピーし、帯外を 0 とする新しい対称帯行列を作成します。</para>
    /// </summary>
    /// <typeparam name="TOperator">The stateful index operator type. / 状態を持つ添字演算子の型。</typeparam>
    /// <param name="order">The order of the square matrix. / 正方行列の次数。</param>
    /// <param name="bandwidth">The half-bandwidth (number of subdiagonals). / 帯幅（主対角線より下の副対角線の本数）。</param>
    /// <param name="op">The operator evaluated for each index pair in the lower band. / 下側帯内の各添字ペアに対して評価する演算子。</param>
    /// <returns>A new symmetric band matrix. / 新しい対称帯行列。</returns>
    /// <remarks>
    /// <para>To write into an existing buffer, use <see cref="XMath.GenerateSymmetric{T, TOperator}(int, TOperator, MatrixSpan{T})"/>.</para>
    /// <para>既存の出力先に書き込む場合は <see cref="XMath.GenerateSymmetric{T, TOperator}(int, TOperator, MatrixSpan{T})"/> を使用してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="order"/> or <paramref name="bandwidth"/> is negative. / <paramref name="order"/> または <paramref name="bandwidth"/> が負の場合。</exception>
    /// <exception cref="OverflowException"><c>order * order</c> exceeds <see cref="int.MaxValue"/>. / 要素数 <c>order²</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static Matrix<TElement> GenerateSymmetric<TOperator>(int order, int bandwidth, TOperator op)
        where TOperator : IStatefulIndexOperator<TElement>
    {
        ArgumentOutOfRangeException.ThrowIfNegative(bandwidth);
        var result = Uninitialized(order, order);
        XMath.GenerateSymmetric(bandwidth, op, result.AsMatrixSpan());
        return result;
    }

    /// <summary>
    /// Creates a symmetric band matrix by evaluating <paramref name="op"/> for the lower half of the band, mirroring to the upper triangle, and zeroing elements outside the band.
    /// <para>帯の下半分をデリゲート <paramref name="op"/> で計算して上三角へ対称にコピーし、帯外を 0 とする新しい対称帯行列を作成します。</para>
    /// </summary>
    /// <param name="order">The order of the square matrix. / 正方行列の次数。</param>
    /// <param name="bandwidth">The half-bandwidth (number of subdiagonals). / 帯幅（主対角線より下の副対角線の本数）。</param>
    /// <param name="op">The delegate evaluated for each index pair in the lower band. / 下側帯内の各添字ペアに対して評価するデリゲート。</param>
    /// <returns>A new symmetric band matrix. / 新しい対称帯行列。</returns>
    /// <remarks>
    /// <para>For large matrices, <paramref name="op"/> may be invoked concurrently from multiple threads in arbitrary order.</para>
    /// <para>大きな行列では、<paramref name="op"/> が複数スレッドから順不同で同時に呼び出される場合があります。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="op"/> is <see langword="null"/>. / <paramref name="op"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="order"/> or <paramref name="bandwidth"/> is negative. / <paramref name="order"/> または <paramref name="bandwidth"/> が負の場合。</exception>
    /// <exception cref="OverflowException"><c>order * order</c> exceeds <see cref="int.MaxValue"/>. / 要素数 <c>order²</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    public static Matrix<TElement> GenerateSymmetric(int order, int bandwidth, Func<int, int, TElement> op) =>
        GenerateSymmetric(order, bandwidth, new DelegateIndexOperator<TElement>(op));

    /// <summary>
    /// Returns a <see cref="Span{T}"/> sharing the underlying column-major contiguous storage.
    /// <para>内部の列優先連続格納領域を共有する <see cref="Span{T}"/> を返します。</para>
    /// </summary>
    /// <returns>A span over the matrix's column-major storage. / 行列の列優先格納領域を参照するスパン。</returns>
    public Span<TElement> AsSpan() => data;

    /// <summary>
    /// Returns a writable <see cref="MatrixSpan{TElement}"/> view over the entire matrix.
    /// <para>行列全体を参照する書き込み可能な <see cref="MatrixSpan{TElement}"/> ビューを返します。</para>
    /// </summary>
    /// <returns>A writable view sharing the matrix's storage. / 格納領域を共有する書き込み可能な行列ビュー。</returns>
    public MatrixSpan<TElement> AsMatrixSpan() => new(data, Rows, Columns);

    /// <summary>
    /// Gets a zero-copy writable view of the transpose <c>Aᵀ</c> (without complex conjugation).
    /// <para>転置 <c>Aᵀ</c> を参照するコピーなしの書き込み可能なビューを取得します（複素共役は取りません）。</para>
    /// </summary>
    /// <remarks>
    /// <para>To materialize a transposed copy, use <see cref="XMath.Transpose{T}(Matrix{T})"/>.</para>
    /// <para>転置した新しいコピーを作成する場合は <see cref="XMath.Transpose{T}(Matrix{T})"/> を使用してください。</para>
    /// </remarks>
    public MatrixSpan<TElement> T => AsMatrixSpan().T;

    /// <summary>
    /// Returns a zero-copy writable view of the main diagonal (length <c>min(Rows, Columns)</c>).
    /// <para>主対角成分を参照するコピーなしの書き込み可能なビュー（長さ <c>min(Rows, Columns)</c>）を返します。</para>
    /// </summary>
    /// <returns>A writable vector view over the main diagonal. / 主対角成分を参照する書き込み可能なベクトルビュー。</returns>
    public VectorSpan<TElement> Diagonal() => AsMatrixSpan().Diagonal();

    /// <summary>
    /// Returns a zero-copy writable view of the specified row (with stride equal to <see cref="Rows"/>).
    /// <para>指定した行を参照するコピーなしの書き込み可能なビュー（ストライドは <see cref="Rows"/>）を返します。</para>
    /// </summary>
    /// <param name="row">The zero-based row index. / 0 から始まる行インデックス。</param>
    /// <returns>A writable vector view over row <paramref name="row"/>. / 第 <paramref name="row"/> 行を参照する書き込み可能なベクトルビュー。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="row"/> is out of range. / <paramref name="row"/> が範囲外の場合。</exception>
    public VectorSpan<TElement> Row(int row) => AsMatrixSpan().Row(row);

    /// <summary>
    /// Returns a zero-copy contiguous writable view of the specified column.
    /// <para>指定した列を参照するコピーなしの連続した書き込み可能なビューを返します。</para>
    /// </summary>
    /// <param name="column">The zero-based column index. / 0 から始まる列インデックス。</param>
    /// <returns>A writable vector view over column <paramref name="column"/>. / 第 <paramref name="column"/> 列を参照する書き込み可能なベクトルビュー。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="column"/> is out of range. / <paramref name="column"/> が範囲外の場合。</exception>
    public VectorSpan<TElement> Column(int column) => AsMatrixSpan().Column(column);

    /// <summary>
    /// Returns a zero-copy read-only rank-3 tensor view that broadcasts this matrix <paramref name="count"/> times along the third axis.
    /// <para>この行列を第 3 軸方向に <paramref name="count"/> 個ブロードキャストした、コピーなしの読み取り専用 3 階テンソルビューを返します。</para>
    /// </summary>
    /// <param name="count">The number of repeated matrices. / 繰り返す行列の個数。</param>
    /// <returns>A read-only <c>[Rows, Columns, count]</c> tensor view. / <c>[Rows, Columns, count]</c> の読み取り専用テンソルビュー。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative. / <paramref name="count"/> が負の場合。</exception>
    public ReadOnlyTensorSpan<TElement> Broadcast(int count) => AsMatrixSpan().Broadcast(count);

    /// <summary>
    /// Returns a zero-copy writable rank-3 tensor view with a length-1 axis inserted at <paramref name="axis"/>.
    /// <para>位置 <paramref name="axis"/> に長さ 1 の軸を挿入した、コピーなしの書き込み可能な 3 階テンソルビューを返します。</para>
    /// </summary>
    /// <param name="axis">The zero-based axis position (0, 1, or 2). / 0 から始まる挿入位置（0、1、または 2）。</param>
    /// <returns>A writable rank-3 tensor view sharing this matrix's storage. / この行列と格納領域を共有する書き込み可能な 3 階テンソルビュー。</returns>
    /// <remarks>
    /// <para>See <see cref="ReadOnlyMatrixSpan{TElement}.Unsqueeze(int)"/> for the resulting shape of each <paramref name="axis"/> value.</para>
    /// <para>各 <paramref name="axis"/> に対する出力形状の詳細は <see cref="ReadOnlyMatrixSpan{TElement}.Unsqueeze(int)"/> を参照してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="axis"/> is not 0, 1, or 2. / <paramref name="axis"/> が 0、1、2 のいずれでもない場合。</exception>
    public TensorSpan<TElement> Unsqueeze(int axis) => AsMatrixSpan().Unsqueeze(axis);

    /// <summary>
    /// Copies the elements of this matrix to a new two-dimensional array preserving indices.
    /// <para>添字を維持してこの行列の要素を新しい 2 次元配列へコピーします。</para>
    /// </summary>
    /// <returns>A new two-dimensional array containing the copied elements. / 要素をコピーした新しい 2 次元配列。</returns>
    public TElement[,] ToArray() => AsMatrixSpan().ToArray();

    /// <summary>
    /// Creates a deep copy of this matrix with independent storage.
    /// <para>独立した格納領域を持つこの行列の複製を作成します。</para>
    /// </summary>
    /// <returns>A new matrix containing the same elements. / 同じ要素を持つ新しい行列。</returns>
    public Matrix<TElement> Clone() => new(Rows, Columns, (TElement[])data.Clone());

    /// <summary>
    /// Reinterprets this matrix as a vector in column-major order sharing the same underlying storage.
    /// <para>同じ格納領域を共有する列優先順のベクトルとしてこの行列を再解釈します。</para>
    /// </summary>
    /// <returns>A vector sharing this matrix's storage. / この行列と格納領域を共有するベクトル。</returns>
    public Vector<TElement> AsVector() => Vector<TElement>.Wrap(data);

    /// <summary>
    /// Copies the elements of this matrix in column-major order into a new vector.
    /// <para>この行列の要素を列優先順で新しいベクトルへコピーします。</para>
    /// </summary>
    /// <returns>A new vector containing the copied elements. / 要素をコピーした新しいベクトル。</returns>
    public Vector<TElement> ToVector() => new(data);

    /// <summary>
    /// Converts each element to <typeparamref name="TTo"/> using checked numeric conversion (<see cref="INumberBase{TSelf}.CreateChecked{TOther}(TOther)"/>).
    /// <para>チェック付きの数値変換（<see cref="INumberBase{TSelf}.CreateChecked{TOther}(TOther)"/>）を用いて各要素を <typeparamref name="TTo"/> に変換します。</para>
    /// </summary>
    /// <typeparam name="TTo">The target element type. / 変換先の要素型。</typeparam>
    /// <returns>A new matrix of the same shape containing the converted elements. / 変換後の要素を格納した同じ形状の新しい行列。</returns>
    /// <exception cref="NotSupportedException">Conversion between <typeparamref name="TElement"/> and <typeparamref name="TTo"/> is not supported. / <typeparamref name="TElement"/> から <typeparamref name="TTo"/> への変換に対応していない場合。</exception>
    /// <exception cref="OverflowException">An element cannot be represented in <typeparamref name="TTo"/>. / いずれかの要素が <typeparamref name="TTo"/> で表現できない場合。</exception>
    public Matrix<TTo> ConvertChecked<TTo>() where TTo : unmanaged, INumberBase<TTo> =>
        Matrix<TTo>.Wrap(Rows, Columns, DenseStorage.ConvertChecked<TElement, TTo>(data));

    /// <summary>
    /// Implicitly converts a <see cref="Matrix{TElement}"/> to a writable <see cref="MatrixSpan{TElement}"/> view.
    /// <para><see cref="Matrix{TElement}"/> を書き込み可能な <see cref="MatrixSpan{TElement}"/> ビューへ暗黙的に変換します。</para>
    /// </summary>
    /// <param name="matrix">The matrix to view. / 対象の行列。</param>
    /// <returns>A writable view over <paramref name="matrix"/>. / <paramref name="matrix"/> を参照する書き込み可能なビュー。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="matrix"/> is <see langword="null"/>. / <paramref name="matrix"/> が <see langword="null"/> の場合。</exception>
    public static implicit operator MatrixSpan<TElement>(Matrix<TElement> matrix) =>
        (matrix ?? throw new ArgumentNullException(nameof(matrix))).AsMatrixSpan();

    /// <summary>
    /// Implicitly converts a <see cref="Matrix{TElement}"/> to a read-only <see cref="ReadOnlyMatrixSpan{TElement}"/> view.
    /// <para><see cref="Matrix{TElement}"/> を読み取り専用の <see cref="ReadOnlyMatrixSpan{TElement}"/> ビューへ暗黙的に変換します。</para>
    /// </summary>
    /// <param name="matrix">The matrix to view. / 対象の行列。</param>
    /// <returns>A read-only view over <paramref name="matrix"/>. / <paramref name="matrix"/> を参照する読み取り専用ビュー。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="matrix"/> is <see langword="null"/>. / <paramref name="matrix"/> が <see langword="null"/> の場合。</exception>
    public static implicit operator ReadOnlyMatrixSpan<TElement>(Matrix<TElement> matrix) =>
        (matrix ?? throw new ArgumentNullException(nameof(matrix))).AsMatrixSpan();

    /// <summary>
    /// Computes the elementwise sum of two matrices.
    /// <para>2 つの行列の要素ごとの和を計算します。</para>
    /// </summary>
    /// <param name="a">The left-hand matrix. / 左辺の行列。</param>
    /// <param name="b">The right-hand matrix. / 右辺の行列。</param>
    /// <returns>A new matrix containing <c>a + b</c>. / <c>a + b</c> を格納した新しい行列。</returns>
    public static Matrix<TElement> operator +(Matrix<TElement> a, Matrix<TElement> b) => XMath.Add(a, b);

    /// <summary>
    /// Computes the elementwise difference of two matrices.
    /// <para>2 つの行列の要素ごとの差を計算します。</para>
    /// </summary>
    /// <param name="a">The left-hand matrix. / 左辺の行列。</param>
    /// <param name="b">The right-hand matrix. / 右辺の行列。</param>
    /// <returns>A new matrix containing <c>a - b</c>. / <c>a - b</c> を格納した新しい行列。</returns>
    public static Matrix<TElement> operator -(Matrix<TElement> a, Matrix<TElement> b) => XMath.Subtract(a, b);

    /// <summary>
    /// Negates each element of the matrix.
    /// <para>行列の各要素の符号を反転します。</para>
    /// </summary>
    /// <param name="a">The input matrix. / 入力行列。</param>
    /// <returns>A new matrix containing <c>-a</c>. / <c>-a</c> を格納した新しい行列。</returns>
    public static Matrix<TElement> operator -(Matrix<TElement> a) => XMath.Negate(a);

    /// <summary>
    /// Multiplies a matrix by a scalar.
    /// <para>行列をスカラー倍します。</para>
    /// </summary>
    /// <param name="scalar">The scalar multiplier. / 掛けるスカラー値。</param>
    /// <param name="a">The input matrix. / 入力行列。</param>
    /// <returns>A new matrix containing <c>scalar * a</c>. / <c>scalar * a</c> を格納した新しい行列。</returns>
    public static Matrix<TElement> operator *(TElement scalar, Matrix<TElement> a) => XMath.Multiply(a, scalar);

    /// <summary>
    /// Multiplies a matrix by a scalar.
    /// <para>行列をスカラー倍します。</para>
    /// </summary>
    /// <param name="a">The input matrix. / 入力行列。</param>
    /// <param name="scalar">The scalar multiplier. / 掛けるスカラー値。</param>
    /// <returns>A new matrix containing <c>a * scalar</c>. / <c>a * scalar</c> を格納した新しい行列。</returns>
    public static Matrix<TElement> operator *(Matrix<TElement> a, TElement scalar) => XMath.Multiply(a, scalar);

    /// <summary>
    /// Divides each element of a matrix by a scalar.
    /// <para>行列の各要素をスカラーで割ります。</para>
    /// </summary>
    /// <param name="a">The input matrix. / 入力行列。</param>
    /// <param name="scalar">The scalar divisor. / 除数。</param>
    /// <returns>A new matrix containing the quotients. / 各要素の商を格納した新しい行列。</returns>
    public static Matrix<TElement> operator /(Matrix<TElement> a, TElement scalar) => XMath.Divide(a, scalar);

    /// <summary>
    /// Computes the matrix product <c>a * b</c>.
    /// <para>行列積 <c>a * b</c> を計算します。</para>
    /// </summary>
    /// <param name="a">The left-hand matrix. / 左辺の行列。</param>
    /// <param name="b">The right-hand matrix. / 右辺の行列。</param>
    /// <returns>A new matrix containing <c>a * b</c>. / <c>a * b</c> を格納した新しい行列。</returns>
    /// <remarks>
    /// <para>Matrix-vector multiplication operators are defined on the vector types.</para>
    /// <para>行列ベクトル積の演算子はベクトル型側に定義されています。</para>
    /// </remarks>
    public static Matrix<TElement> operator *(Matrix<TElement> a, Matrix<TElement> b) => XMath.Multiply(a, b);

    /// <summary>
    /// Adds the elements of <paramref name="other"/> to this matrix in place.
    /// <para>この行列に <paramref name="other"/> をインプレースで加算します。</para>
    /// </summary>
    /// <param name="other">The matrix to add. / 加算する行列。</param>
    public void operator +=(ReadOnlyMatrixSpan<TElement> other) => XMath.Add(this, other, this);

    /// <summary>
    /// Subtracts the elements of <paramref name="other"/> from this matrix in place.
    /// <para>この行列から <paramref name="other"/> をインプレースで減算します。</para>
    /// </summary>
    /// <param name="other">The matrix to subtract. / 減算する行列。</param>
    public void operator -=(ReadOnlyMatrixSpan<TElement> other) => XMath.Subtract(this, other, this);

    /// <summary>
    /// Scales this matrix by <paramref name="scalar"/> in place.
    /// <para>この行列を <paramref name="scalar"/> でインプレースにスカラー倍します。</para>
    /// </summary>
    /// <param name="scalar">The scalar multiplier. / 掛けるスカラー値。</param>
    public void operator *=(TElement scalar) => XMath.InPlace.Multiply(this, scalar);

    /// <summary>
    /// Divides each element of this matrix by a scalar in place.
    /// <para>この行列の各要素をインプレースでスカラー除算します。</para>
    /// </summary>
    /// <param name="scalar">The scalar divisor. / 除数。</param>
    public void operator /=(TElement scalar) => XMath.InPlace.Divide(this, scalar);

    /// <summary>
    /// Explicitly converts a two-dimensional array to a new <see cref="Matrix{TElement}"/> by copying its elements.
    /// <para>2 次元配列の要素をコピーして新しい <see cref="Matrix{TElement}"/> へ明示的に変換します。</para>
    /// </summary>
    /// <param name="values">The source two-dimensional array. / コピー元の 2 次元配列。</param>
    /// <returns>A new matrix containing the copied elements. / 要素をコピーした新しい行列。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="values"/> is <see langword="null"/>. / <paramref name="values"/> が <see langword="null"/> の場合。</exception>
    public static explicit operator Matrix<TElement>(TElement[,] values) => new(values);

    /// <summary>
    /// Explicitly converts a <see cref="Matrix{TElement}"/> to a new two-dimensional array by copying its elements.
    /// <para><see cref="Matrix{TElement}"/> の要素をコピーして新しい 2 次元配列へ明示的に変換します。</para>
    /// </summary>
    /// <param name="matrix">The source matrix. / コピー元の行列。</param>
    /// <returns>A new two-dimensional array containing the copied elements. / 要素をコピーした新しい 2 次元配列。</returns>
    public static explicit operator TElement[,](Matrix<TElement> matrix) => matrix.ToArray();

    /// <summary>
    /// Returns a string representation containing the matrix dimensions and up to the leading 8 × 8 elements.
    /// <para>行列の形状と先頭 8 × 8 までの要素を含む文字列表現を返します。</para>
    /// </summary>
    /// <returns>A formatted summary string of the matrix. / 行列の形状と要素を表す文字列。</returns>
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.Append(CultureInfo.InvariantCulture, $"Matrix<{typeof(TElement).Name}> {Rows}x{Columns}");
        int shownRows = Math.Min(Rows, 8);
        int shownColumns = Math.Min(Columns, 8);
        for (int i = 0; i < shownRows; i++)
        {
            sb.AppendLine();
            for (int j = 0; j < shownColumns; j++)
            {
                sb.Append(CultureInfo.InvariantCulture, $"{data[i + j * Rows],14:G6}");
            }

            if (shownColumns < Columns)
            {
                sb.Append(" ...");
            }
        }

        if (shownRows < Rows)
        {
            sb.AppendLine().Append(" ...");
        }

        return sb.ToString();
    }

    internal static Matrix<TElement> Wrap(int rows, int columns, TElement[] data) => new(rows, columns, data);

    // Allocates uninitialized storage for kernels that overwrite every element (not zero-filled).
    internal static Matrix<TElement> Uninitialized(int rows, int columns) =>
        new(rows, columns, GC.AllocateUninitializedArray<TElement>(DenseStorage.Length(rows, columns)));
    internal TElement[] Data => data;

    private void CheckIndex(int row, int column)
    {
        DenseStorage.CheckIndex(row, Rows, nameof(row));
        DenseStorage.CheckIndex(column, Columns, nameof(column));
    }
}
