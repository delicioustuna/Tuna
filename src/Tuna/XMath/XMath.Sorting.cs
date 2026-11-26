using System.Numerics;

namespace Tuna;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

public static partial class XMath
{
    /// <summary>
    /// Returns a new vector containing the elements of <paramref name="x"/> sorted in ascending order.
    /// <para><paramref name="x"/> の要素を昇順に並べ替えた新しいベクトルを返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>A new vector sorted in ascending order. / 昇順に並べ替えた新しいベクトル。</returns>
    /// <remarks>
    /// <para>Uses total order: <c>-0</c> precedes <c>+0</c>, and <c>NaN</c> values are placed at the end in their original relative order.</para>
    /// <para>全順序で並べ替えます（<c>-0</c> は <c>+0</c> より前、<c>NaN</c> は末尾に元の順序のまま配置されます）。</para>
    /// </remarks>
    [InferenceOverloads]
    public static Vector<T> Sort<T>(ReadOnlyVectorSpan<T> x)
        where T : unmanaged, INumber<T>
    {
        var result = Vector<T>.Uninitialized(x.Length);
        StridedKernels.Sort(x, result);
        return result;
    }

    /// <summary>
    /// Sorts the elements of <paramref name="x"/> in ascending order into <paramref name="destination"/>.
    /// <para><paramref name="x"/> の要素を昇順に並べ替え、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="destination">The destination span to receive the sorted elements. / 並べ替え結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>Uses total order (<c>-0</c> before <c>+0</c>, <c>NaN</c> at the end in original order). <paramref name="destination"/> may alias <paramref name="x"/> identically (in-place) or be disjoint; partial overlap is rejected.</para>
    /// <para>全順序で並べ替えます（<c>-0</c> は <c>+0</c> より前、<c>NaN</c> は末尾に元の順序のまま）。<paramref name="destination"/> は <paramref name="x"/> と完全に同一の領域（インプレース）または重なりのない領域を指定できます。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The lengths differ, or <paramref name="destination"/> partially overlaps <paramref name="x"/>. / 長さが一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と部分的に重なる場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> Sort<T>(ReadOnlyVectorSpan<T> x, VectorSpan<T> destination)
        where T : unmanaged, INumber<T>
    {
        StridedKernels.Sort(x, destination);
        return destination;
    }

    /// <summary>
    /// Returns the zero-based indices that would sort <paramref name="x"/> in ascending order.
    /// <para><paramref name="x"/> を昇順に並べ替えるための 0 始まりのインデックス配列を返します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <returns>A new array of zero-based indices that sort <paramref name="x"/>. / 昇順の並びに対応する 0 始まりのインデックス配列。</returns>
    /// <remarks>
    /// <para>Stable sort: indices of equal elements preserve their original order. Uses the same ordering as <see cref="Sort{T}(ReadOnlyVectorSpan{T})"/>.</para>
    /// <para>安定ソートであり、等しい値のインデックスは元の順序を保ちます。順序規則は <see cref="Sort{T}(ReadOnlyVectorSpan{T})"/> と同じです。</para>
    /// </remarks>
    [InferenceOverloads]
    public static int[] ArgSort<T>(ReadOnlyVectorSpan<T> x)
        where T : unmanaged, INumber<T>
    {
        var result = new int[x.Length];
        StridedKernels.ArgSort(x, result);
        return result;
    }

    /// <summary>
    /// Computes the zero-based indices that would sort <paramref name="x"/> in ascending order and writes them to <paramref name="destination"/>.
    /// <para><paramref name="x"/> を昇順に並べ替える 0 始まりのインデックスを計算し、<paramref name="destination"/> に書き込みます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="x">The input vector. / 入力ベクトル。</param>
    /// <param name="destination">The destination span to receive the indices. / インデックスを書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>Stable sort using the same ordering as <see cref="Sort{T}(ReadOnlyVectorSpan{T})"/>. <paramref name="destination"/> must not share memory with <paramref name="x"/>.</para>
    /// <para><see cref="Sort{T}(ReadOnlyVectorSpan{T})"/> と同じ順序規則による安定ソートです。<paramref name="destination"/> は <paramref name="x"/> と記憶域を共有できません。</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The lengths differ, or <paramref name="destination"/> shares memory with <paramref name="x"/>. / 長さが一致しない場合、または <paramref name="destination"/> が <paramref name="x"/> と記憶域を共有する場合。</exception>
    [InferenceOverloads]
    public static Span<int> ArgSort<T>(ReadOnlyVectorSpan<T> x, Span<int> destination)
        where T : unmanaged, INumber<T>
    {
        StridedKernels.ArgSort(x, destination);
        return destination;
    }

    public static partial class InPlace
    {
        /// <summary>
        /// Sorts the elements of <paramref name="x"/> in ascending order in place.
        /// <para><paramref name="x"/> の要素をインプレースで昇順に並べ替えます。</para>
        /// </summary>
        /// <typeparam name="T">The element type. / 要素型。</typeparam>
        /// <param name="x">The vector to sort in place. / インプレースで並べ替えるベクトル。</param>
        /// <remarks>
        /// <para>Uses the same ordering as <see cref="XMath.Sort{T}(ReadOnlyVectorSpan{T})"/>.</para>
        /// <para><see cref="XMath.Sort{T}(ReadOnlyVectorSpan{T})"/> と同じ順序規則を使用します。</para>
        /// </remarks>
        [InferenceOverloads]
        public static void Sort<T>(VectorSpan<T> x)
            where T : unmanaged, INumber<T> =>
            StridedKernels.Sort(x, x);
    }
}
