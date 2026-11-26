using System.Collections;
using System.Numerics;
using Tuna.LinearAlgebra;

namespace Tuna.Differentiation;

/// <summary>
/// Represents persistent learnable parameter storage (value and gradient buffers) that outlives a <see cref="Tape"/>.
/// <para><see cref="Tape"/> の寿命を超えて保持される学習パラメーター（値および勾配バッファ）を表します。</para>
/// </summary>
/// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
/// <remarks>
/// <para>
/// Attach a parameter to a recording via <see cref="Tape.UseVector{T}(Parameter{T})"/> or <see cref="Tape.UseMatrix{T}(Parameter{T})"/>.
/// Backpropagation accumulates into <see cref="Gradient"/> without overwriting it; clear gradients explicitly with <see cref="ZeroGradient"/>.
/// Values and gradients are stored flat in column-major order. Do not mutate <see cref="Value"/> between attaching the parameter and completing backpropagation, and do not backpropagate into the same parameter concurrently from multiple threads.
/// </para>
/// <para>
/// 記録への参加には <see cref="Tape.UseVector{T}(Parameter{T})"/> または <see cref="Tape.UseMatrix{T}(Parameter{T})"/> を使用します。
/// 逆伝播は <see cref="Gradient"/> に加算し上書きしないため、リセットが必要な場合は <see cref="ZeroGradient"/> などを明示的に呼び出してください。
/// 値と勾配は列優先でフラットに格納されます。記録に参加させてから逆伝播が完了するまで <see cref="Value"/> を変更しないでください。また、同一のパラメーターに対して複数スレッドから同時に逆伝播を実行しないでください。
/// </para>
/// </remarks>
public sealed class Parameter<T> where T : unmanaged, IFloatingPointIeee754<T>
{
    private Parameter(int rank, int rows, int columns, int matrixCount = 1)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(columns);
        ArgumentOutOfRangeException.ThrowIfNegative(matrixCount);
        TapeElementTypes.Require<T>();
        int length = matrixCount == 0 ? 0 : checked(rows * columns * matrixCount);
        Rank = rank;
        Rows = rows;
        Columns = columns;
        MatrixCount = matrixCount;
        ValueArray = new T[length];
        GradientArray = new T[length];
    }

    /// <summary>
    /// Creates a vector parameter initialized to zeros.
    /// <para>0 で初期化されたベクトル型の学習パラメーターを作成します。</para>
    /// </summary>
    /// <param name="length">The number of elements. / 要素数。</param>
    /// <returns>A vector parameter with zeroed values and gradients. / 値と勾配が 0 で初期化されたベクトル型の学習パラメーター。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is negative. / <paramref name="length"/> が負の場合。</exception>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see langword="float"/> nor <see langword="double"/>. / 要素型が <see langword="float"/> でも <see langword="double"/> でもない場合。</exception>
    public static Parameter<T> Vector(int length) => new(1, length, 1);

    /// <summary>
    /// Creates a matrix parameter initialized to zeros.
    /// <para>0 で初期化された行列型の学習パラメーターを作成します。</para>
    /// </summary>
    /// <param name="rows">The number of rows. / 行数。</param>
    /// <param name="columns">The number of columns. / 列数。</param>
    /// <returns>A matrix parameter with zeroed values and gradients. / 値と勾配が 0 で初期化された行列型の学習パラメーター。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="rows"/> or <paramref name="columns"/> is negative. / <paramref name="rows"/> または <paramref name="columns"/> が負の場合。</exception>
    /// <exception cref="OverflowException">The element count <c>rows * columns</c> exceeds <see cref="int.MaxValue"/>. / 要素数 <c>rows * columns</c> が <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see langword="float"/> nor <see langword="double"/>. / 要素型が <see langword="float"/> でも <see langword="double"/> でもない場合。</exception>
    public static Parameter<T> Matrix(int rows, int columns) => new(2, rows, columns);

    /// <summary>
    /// Creates a three-dimensional parameter initialized to zeros.
    /// <para>0 で初期化された3階テンソルの学習パラメーターを作成します。</para>
    /// </summary>
    /// <param name="rows">The row count in each matrix. / 各行列の行数。</param>
    /// <param name="columns">The column count in each matrix. / 各行列の列数。</param>
    /// <param name="matrixCount">The number of matrices. / 行列数。</param>
    /// <returns>A tensor parameter. / テンソルの学習パラメーター。</returns>
    /// <exception cref="ArgumentOutOfRangeException">A dimension is negative. / いずれかの次元が負の場合。</exception>
    /// <exception cref="OverflowException">The element count exceeds <see cref="int.MaxValue"/>. / 要素数が <see cref="int.MaxValue"/> を超える場合。</exception>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> is neither <see langword="float"/> nor <see langword="double"/>. / 要素型が <see langword="float"/> でも <see langword="double"/> でもない場合。</exception>
    public static Parameter<T> Tensor(int rows, int columns, int matrixCount) => new(3, rows, columns, matrixCount);

    /// <summary>
    /// Gets the rank of the parameter (1 for a vector, 2 for a matrix, 3 for a tensor).
    /// <para>パラメーターの次元数（ベクトルは1、行列は2、テンソルは3）を取得します。</para>
    /// </summary>
    public int Rank { get; }

    /// <summary>
    /// Gets the number of rows (equal to the element count for a vector).
    /// <para>行数（ベクトルの場合は要素数）を取得します。</para>
    /// </summary>
    public int Rows { get; }

    /// <summary>
    /// Gets the number of columns (1 for a vector).
    /// <para>列数（ベクトルの場合は 1）を取得します。</para>
    /// </summary>
    public int Columns { get; }

    /// <summary>
    /// Gets the number of matrices along the third axis, or 1 for vectors and matrices.
    /// <para>第3軸の行列数を取得します。ベクトルと行列の場合は1です。</para>
    /// </summary>
    public int MatrixCount { get; }

    /// <summary>
    /// Gets the total number of elements.
    /// <para>総要素数を取得します。</para>
    /// </summary>
    public int Length => ValueArray.Length;

    /// <summary>
    /// Gets a span over the parameter values in column-major order.
    /// <para>列優先で格納された値へのスパンを取得します。</para>
    /// </summary>
    public Span<T> Value => ValueArray;

    /// <summary>
    /// Gets a span over the accumulated gradients in column-major order.
    /// <para>逆伝播で加算された勾配へのスパン（列優先）を取得します。</para>
    /// </summary>
    public Span<T> Gradient => GradientArray;

    internal T[] ValueArray { get; }

    internal T[] GradientArray { get; }

    /// <summary>
    /// Returns a zero-copy vector view over <see cref="Value"/> for a rank-1 parameter.
    /// <para>1 階（ベクトル）パラメーターの <see cref="Value"/> に対するコピーなしのベクトルビューを返します。</para>
    /// </summary>
    /// <returns>A writable vector view over the parameter values. / パラメーターの値に対する書き込み可能なベクトルビュー。</returns>
    /// <exception cref="InvalidOperationException"><see cref="Rank"/> is not 1. / <see cref="Rank"/> が 1 でない場合。</exception>
    public VectorSpan<T> AsVectorSpan()
    {
        if (Rank != 1)
        {
            throw new InvalidOperationException("The parameter is not a vector.");
        }

        return new(ValueArray);
    }

    /// <summary>
    /// Returns a zero-copy matrix view over <see cref="Value"/> for a rank-2 parameter.
    /// <para>2 階（行列）パラメーターの <see cref="Value"/> に対するコピーなしの行列ビューを返します。</para>
    /// </summary>
    /// <returns>A writable matrix view over the parameter values. / パラメーターの値に対する書き込み可能な行列ビュー。</returns>
    /// <exception cref="InvalidOperationException"><see cref="Rank"/> is not 2. / <see cref="Rank"/> が 2 でない場合。</exception>
    public MatrixSpan<T> AsMatrixSpan()
    {
        if (Rank != 2)
        {
            throw new InvalidOperationException("The parameter is not a matrix.");
        }

        return new(ValueArray, Rows, Columns);
    }

    /// <summary>
    /// Returns a zero-copy tensor view over <see cref="Value"/> for a rank-3 parameter.
    /// <para>3 階（テンソル）パラメーターの <see cref="Value"/> に対するコピーなしのテンソルビューを返します。</para>
    /// </summary>
    /// <returns>A writable tensor view over the parameter values. / パラメーターの値に対する書き込み可能なテンソルビュー。</returns>
    /// <exception cref="InvalidOperationException"><see cref="Rank"/> is not 3. / <see cref="Rank"/> が 3 でない場合。</exception>
    public TensorSpan<T> AsTensorSpan()
    {
        if (Rank != 3)
        {
            throw new InvalidOperationException("The parameter is not a tensor.");
        }

        return new(ValueArray, Rows, Columns, MatrixCount);
    }

    /// <summary>
    /// Clears all gradient entries to zero.
    /// <para>勾配のすべての要素を 0 にリセットします。</para>
    /// </summary>
    public void ZeroGradient() => Array.Clear(GradientArray);
}

/// <summary>
/// Represents an ordered collection of distinct <see cref="Parameter{T}"/> instances.
/// <para>重複しない複数の <see cref="Parameter{T}"/> を順序付きで保持するコレクションを表します。</para>
/// </summary>
/// <typeparam name="T">The floating-point element type (<see langword="float"/> or <see langword="double"/>). / 要素の浮動小数点数型（<see langword="float"/> または <see langword="double"/>）。</typeparam>
/// <remarks>
/// <para>
/// Provides a unified view for optimizers updating parameter values and gradients together. Duplicate parameter references are not allowed.
/// </para>
/// <para>
/// オプティマイザーが複数のパラメーターの値と勾配をまとめて更新するための集合です。同じパラメーターを重複して含めることはできません。
/// </para>
/// </remarks>
public sealed class ParameterSet<T> : IReadOnlyList<Parameter<T>> where T : unmanaged, IFloatingPointIeee754<T>
{
    private readonly Parameter<T>[] parameters;

    /// <summary>
    /// Initializes a new <see cref="ParameterSet{T}"/> containing the specified parameters in order.
    /// <para>指定したパラメーターをその順序で保持する <see cref="ParameterSet{T}"/> を初期化します。</para>
    /// </summary>
    /// <param name="parameters">The distinct parameters to include. / 含める重複しないパラメーターの列。</param>
    /// <exception cref="ArgumentNullException">Any element of <paramref name="parameters"/> is <see langword="null"/>. / <paramref name="parameters"/> に <see langword="null"/> が含まれる場合。</exception>
    /// <exception cref="ArgumentException">The same parameter instance appears more than once. / 同一のパラメーターインスタンスが重複して含まれる場合。</exception>
    public ParameterSet(params ReadOnlySpan<Parameter<T>> parameters)
    {
        var copy = parameters.ToArray();
        long length = 0;
        for (int i = 0; i < copy.Length; i++)
        {
            ArgumentNullException.ThrowIfNull(copy[i], nameof(parameters));
            for (int j = 0; j < i; j++)
            {
                if (ReferenceEquals(copy[i], copy[j]))
                {
                    throw new ArgumentException("The same parameter appears more than once.", nameof(parameters));
                }
            }

            length += copy[i].Length;
        }

        this.parameters = copy;
        Length = length;
    }

    /// <summary>
    /// Gets the number of parameters in the set.
    /// <para>含まれるパラメーターの個数を取得します。</para>
    /// </summary>
    public int Count => parameters.Length;

    /// <summary>
    /// Gets the total number of scalar elements across all parameters.
    /// <para>すべてのパラメーターの総要素数を取得します。</para>
    /// </summary>
    public long Length { get; }

    /// <summary>
    /// Gets the parameter at the specified index.
    /// <para>指定したインデックスにあるパラメーターを取得します。</para>
    /// </summary>
    /// <param name="index">The zero-based index of the parameter. / 0 始まりのインデックス。</param>
    /// <value>The parameter at <paramref name="index"/>. / 指定位置のパラメーター。</value>
    public Parameter<T> this[int index] => parameters[index];

    /// <summary>
    /// Clears the gradients of all parameters in the set to zero.
    /// <para>すべてのパラメーターの勾配を 0 にリセットします。</para>
    /// </summary>
    public void ZeroGradients()
    {
        foreach (var parameter in parameters)
        {
            parameter.ZeroGradient();
        }
    }

    /// <summary>
    /// Returns an enumerator that iterates through the parameters in order.
    /// <para>パラメーターを順に反復処理する列挙子を返します。</para>
    /// </summary>
    /// <returns>An enumerator over the parameters in creation order. / パラメーターを順に列挙する列挙子。</returns>
    public IEnumerator<Parameter<T>> GetEnumerator() => ((IEnumerable<Parameter<T>>)parameters).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}

internal static class TapeElementTypes
{
    public static void Require<T>()
    {
        if (typeof(T) != typeof(double) && typeof(T) != typeof(float))
        {
            throw new NotSupportedException($"Array nodes support float and double, not {typeof(T).Name}.");
        }
    }
}
