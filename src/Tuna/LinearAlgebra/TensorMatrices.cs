using System.Numerics;

namespace Tuna.LinearAlgebra;

/// <summary>
/// Provides indexed read-only access and allocation-free enumeration over the matrix slices of a tensor.
/// <para>テンソル内の各行列スライスに対するインデックスアクセスと、アロケーションなしの列挙を提供する読み取り専用ビューです。</para>
/// </summary>
/// <typeparam name="TElement">The element type of the tensor. / テンソルの要素型。</typeparam>
/// <remarks>
/// <para>In batched matrix operations, the first two axes form each matrix and the remaining axis indexes the matrices. Use <c>foreach (var matrix in tensor.Matrices)</c> to enumerate slices without heap allocations.</para>
/// <para>バッチ行列演算では先頭 2 軸が行列、残りの軸が行列インデックスを表します。<c>foreach (var matrix in tensor.Matrices)</c> によりヒープ確保なしで各行列を列挙できます。</para>
/// </remarks>
public readonly ref struct ReadOnlyTensorMatrices<TElement> where TElement : unmanaged, INumberBase<TElement>
{
    private readonly ReadOnlyTensorSpan<TElement> tensor;

    internal ReadOnlyTensorMatrices(ReadOnlyTensorSpan<TElement> tensor) => this.tensor = tensor;

    /// <summary>
    /// Gets the number of matrix slices in the tensor.
    /// <para>テンソルに含まれる行列の個数を取得します。</para>
    /// </summary>
    /// <value>The number of matrices (equal to <c>Lengths[2]</c> for rank-3 tensors). / 行列の個数（3 階テンソルでは <c>Lengths[2]</c>）。</value>
    public int Count => tensor.MatrixCount;

    /// <summary>
    /// Gets a read-only view of the matrix at the specified index.
    /// <para>指定したインデックスにある行列の読み取り専用ビューを取得します。</para>
    /// </summary>
    /// <param name="index">The zero-based matrix index. / 0 から始まる行列インデックス。</param>
    /// <value>A read-only matrix view sharing the underlying storage. / 元の格納領域を共有する読み取り専用の行列ビュー。</value>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is less than 0 or greater than or equal to <see cref="Count"/>. / <paramref name="index"/> が 0 未満または <see cref="Count"/> 以上の場合。</exception>
    public ReadOnlyMatrixSpan<TElement> this[int index] => tensor.GetMatrix(index);

    /// <summary>
    /// Returns an enumerator that iterates through the matrix slices in index order.
    /// <para>行列スライスをインデックス順に反復処理する列挙子を返します。</para>
    /// </summary>
    /// <returns>An enumerator for the matrix slices. / 行列スライスの列挙子。</returns>
    public Enumerator GetEnumerator() => new(tensor);

    /// <summary>
    /// Enumerates the read-only matrix slices of a tensor without allocation.
    /// <para>テンソルの読み取り専用行列スライスをアロケーションなしで列挙する列挙子です。</para>
    /// </summary>
    public ref struct Enumerator
    {
        private readonly ReadOnlyTensorSpan<TElement> tensor;
        private int index;

        internal Enumerator(ReadOnlyTensorSpan<TElement> tensor)
        {
            this.tensor = tensor;
            index = -1;
        }

        /// <summary>
        /// Gets the matrix slice at the current position of the enumerator.
        /// <para>列挙子の現在位置にある行列スライスを取得します。</para>
        /// </summary>
        public readonly ReadOnlyMatrixSpan<TElement> Current => tensor.Matrix(index);

        /// <summary>
        /// Advances the enumerator to the next matrix slice.
        /// <para>列挙子を次の行列スライスへ進めます。</para>
        /// </summary>
        /// <returns><see langword="true"/> if the enumerator successfully advanced to the next matrix; <see langword="false"/> if the enumerator has passed the end. / 次の行列へ進めた場合は <see langword="true"/>、末尾に達した場合は <see langword="false"/>。</returns>
        public bool MoveNext() => ++index < tensor.MatrixCount;
    }
}

/// <summary>
/// Provides indexed writable access and allocation-free enumeration over the matrix slices of a tensor.
/// <para>テンソル内の各行列スライスに対する書き込み可能なインデックスアクセスと、アロケーションなしの列挙を提供するビューです。</para>
/// </summary>
/// <typeparam name="TElement">The element type of the tensor. / テンソルの要素型。</typeparam>
/// <remarks>
/// <para>In batched matrix operations, the first two axes form each matrix and the remaining axis indexes the matrices. Use <c>foreach (var matrix in tensor.Matrices)</c> to enumerate slices without heap allocations.</para>
/// <para>バッチ行列演算では先頭 2 軸が行列、残りの軸が行列インデックスを表します。<c>foreach (var matrix in tensor.Matrices)</c> によりヒープ確保なしで各行列を列挙できます。</para>
/// </remarks>
public readonly ref struct TensorMatrices<TElement> where TElement : unmanaged, INumberBase<TElement>
{
    private readonly TensorSpan<TElement> tensor;

    internal TensorMatrices(TensorSpan<TElement> tensor) => this.tensor = tensor;

    /// <summary>
    /// Gets the number of matrix slices in the tensor.
    /// <para>テンソルに含まれる行列の個数を取得します。</para>
    /// </summary>
    /// <value>The number of matrices (equal to <c>Lengths[2]</c> for rank-3 tensors). / 行列の個数（3 階テンソルでは <c>Lengths[2]</c>）。</value>
    public int Count => tensor.AsReadOnly().MatrixCount;

    /// <summary>
    /// Gets a writable view of the matrix at the specified index.
    /// <para>指定したインデックスにある行列の書き込み可能なビューを取得します。</para>
    /// </summary>
    /// <param name="index">The zero-based matrix index. / 0 から始まる行列インデックス。</param>
    /// <value>A writable matrix view sharing the underlying storage. / 元の格納領域を共有する書き込み可能な行列ビュー。</value>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is less than 0 or greater than or equal to <see cref="Count"/>. / <paramref name="index"/> が 0 未満または <see cref="Count"/> 以上の場合。</exception>
    public MatrixSpan<TElement> this[int index] => tensor.GetMatrix(index);

    /// <summary>
    /// Returns an enumerator that iterates through the matrix slices in index order.
    /// <para>行列スライスをインデックス順に反復処理する列挙子を返します。</para>
    /// </summary>
    /// <returns>An enumerator for the matrix slices. / 行列スライスの列挙子。</returns>
    public Enumerator GetEnumerator() => new(tensor.AsReadOnly());

    /// <summary>
    /// Returns a read-only view over the same matrix slices.
    /// <para>同じ行列スライスを参照する読み取り専用ビューを返します。</para>
    /// </summary>
    /// <returns>A read-only matrix collection view. / 読み取り専用の行列コレクションビュー。</returns>
    public ReadOnlyTensorMatrices<TElement> AsReadOnly() => new(tensor.AsReadOnly());

    /// <summary>
    /// Implicitly converts a writable matrix collection view to a read-only view.
    /// <para>書き込み可能な行列コレクションビューを読み取り専用ビューへ暗黙的に変換します。</para>
    /// </summary>
    /// <param name="matrices">The writable view to convert. / 変換元の書き込み可能なビュー。</param>
    /// <returns>A read-only view referencing the same tensor. / 同じテンソルを参照する読み取り専用ビュー。</returns>
    public static implicit operator ReadOnlyTensorMatrices<TElement>(TensorMatrices<TElement> matrices) => matrices.AsReadOnly();

    /// <summary>
    /// Enumerates the writable matrix slices of a tensor without allocation.
    /// <para>テンソルの書き込み可能な行列スライスをアロケーションなしで列挙する列挙子です。</para>
    /// </summary>
    public ref struct Enumerator
    {
        private readonly ReadOnlyTensorSpan<TElement> tensor;
        private int index;

        internal Enumerator(ReadOnlyTensorSpan<TElement> tensor)
        {
            this.tensor = tensor;
            index = -1;
        }

        /// <summary>
        /// Gets the matrix slice at the current position of the enumerator.
        /// <para>列挙子の現在位置にある行列スライスを取得します。</para>
        /// </summary>
        public readonly MatrixSpan<TElement> Current => MatrixSpan<TElement>.Writable(tensor.Matrix(index));

        /// <summary>
        /// Advances the enumerator to the next matrix slice.
        /// <para>列挙子を次の行列スライスへ進めます。</para>
        /// </summary>
        /// <returns><see langword="true"/> if the enumerator successfully advanced to the next matrix; <see langword="false"/> if the enumerator has passed the end. / 次の行列へ進めた場合は <see langword="true"/>、末尾に達した場合は <see langword="false"/>。</returns>
        public bool MoveNext() => ++index < tensor.MatrixCount;
    }
}
