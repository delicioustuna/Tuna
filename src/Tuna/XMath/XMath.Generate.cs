using System.Numerics;

namespace Tuna;

// System.Numerics.Vector<T> より Tuna.LinearAlgebra.Vector<T> を優先させるため、名前空間の内側で取り込む。
using Tuna.LinearAlgebra;

public static partial class XMath
{
    /// <summary>
    /// Populates every element of <paramref name="destination"/> using a stateless index operator (<c>destination[i] = TOperator(i)</c>).
    /// <para>状態を持たないインデックス演算子でベクトルの全要素を設定します（<c>destination[i] = TOperator(i)</c>）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless vector index operator type. / 状態を持たないベクトルインデックス演算子の型。</typeparam>
    /// <param name="destination">The destination vector span. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static VectorSpan<T> Generate<T, TOperator>(VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IVectorIndexOperator<T> =>
        Generate(default(StaticVectorIndexOperator<T, TOperator>), destination);

    /// <summary>
    /// Populates every element of <paramref name="destination"/> using a stateful index operator (<c>destination[i] = op(i)</c>).
    /// <para>状態を持つインデックス演算子でベクトルの全要素を設定します（<c>destination[i] = op(i)</c>）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful vector index operator type. / 状態を持つベクトルインデックス演算子の型。</typeparam>
    /// <param name="op">The operator computing each element from its index. / インデックスから各要素を計算する演算子。</param>
    /// <param name="destination">The destination vector span. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static VectorSpan<T> Generate<T, TOperator>(TOperator op, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulVectorIndexOperator<T>
    {
        StridedKernels.Generate(destination, op);
        return destination;
    }

    /// <summary>
    /// Populates every element of <paramref name="destination"/> using a delegate (<c>destination[i] = op(i)</c>).
    /// <para>デリゲートを用いてベクトルの全要素を設定します（<c>destination[i] = op(i)</c>）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="op">The delegate computing each element from its zero-based index. / 0 始まりのインデックスから各要素を計算するデリゲート。</param>
    /// <param name="destination">The destination vector span. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="op"/> is <see langword="null"/>. / <paramref name="op"/> が <see langword="null"/> の場合。</exception>
    [InferenceOverloads]
    public static VectorSpan<T> Generate<T>(Func<int, T> op, VectorSpan<T> destination)
        where T : unmanaged, INumberBase<T> =>
        Generate(new DelegateVectorIndexOperator<T>(op), destination);

    /// <summary>
    /// Populates every element of <paramref name="destination"/> using a stateless index operator (<c>destination[i, j] = TOperator(i, j)</c>).
    /// <para>状態を持たないインデックス演算子で行列の全要素を設定します（<c>destination[i, j] = TOperator(i, j)</c>）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless matrix index operator type. / 状態を持たない行列インデックス演算子の型。</typeparam>
    /// <param name="destination">The destination matrix span. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>To allocate a new matrix, use <see cref="Matrix{TElement}.Generate{TOperator}(int, int)"/>.</para>
    /// <para>新しい行列を割り当てて生成するには <see cref="Matrix{TElement}.Generate{TOperator}(int, int)"/> を使用してください。</para>
    /// </remarks>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static MatrixSpan<T> Generate<T, TOperator>(MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IIndexOperator<T>
    {
        Generate(default(StaticIndexOperator<T, TOperator>), destination);
        return destination;
    }

    /// <summary>
    /// Populates every element of <paramref name="destination"/> using a stateful index operator (<c>destination[i, j] = op(i, j)</c>).
    /// <para>状態を持つインデックス演算子で行列の全要素を設定します（<c>destination[i, j] = op(i, j)</c>）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful matrix index operator type. / 状態を持つ行列インデックス演算子の型。</typeparam>
    /// <param name="op">The operator computing each element from its row and column indices. / 行・列インデックスから各要素を計算する演算子。</param>
    /// <param name="destination">The destination matrix span. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>To allocate a new matrix, use <see cref="Matrix{TElement}.Generate{TOperator}(int, int, TOperator)"/>.</para>
    /// <para>新しい行列を割り当てて生成するには <see cref="Matrix{TElement}.Generate{TOperator}(int, int, TOperator)"/> を使用してください。</para>
    /// </remarks>
    [InferenceOverloads]
    public static MatrixSpan<T> Generate<T, TOperator>(TOperator op, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulIndexOperator<T>
    {
        StridedKernels.Generate(destination, destination.Rows - 1, destination.Columns - 1, symmetric: false,
            StridedKernels.GeneratedElementWork, new IndexSource<T, TOperator>(op));
        return destination;
    }

    /// <summary>
    /// Populates every element of <paramref name="destination"/> using a delegate (<c>destination[i, j] = op(i, j)</c>).
    /// <para>デリゲートを用いて行列の全要素を設定します（<c>destination[i, j] = op(i, j)</c>）。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="op">The delegate computing each element from its zero-based row and column indices. / 0 始まりの行・列インデックスから各要素を計算するデリゲート。</param>
    /// <param name="destination">The destination matrix span. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>For large matrices, <paramref name="op"/> may be invoked concurrently from multiple threads in arbitrary order.</para>
    /// <para>大きな行列では、<paramref name="op"/> が複数スレッドから順不同に呼び出される場合があります。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="op"/> is <see langword="null"/>. / <paramref name="op"/> が <see langword="null"/> の場合。</exception>
    [InferenceOverloads]
    public static MatrixSpan<T> Generate<T>(Func<int, int, T> op, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T> =>
        Generate(new DelegateIndexOperator<T>(op), destination);

    /// <summary>
    /// Populates every element of <paramref name="destination"/> using a stateless tensor index operator.
    /// <para>状態を持たないテンソルインデックス演算子でテンソルの全要素を設定します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless tensor index operator type. / 状態を持たないテンソルインデックス演算子の型。</typeparam>
    /// <param name="destination">The destination tensor span. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static TensorSpan<T> Generate<T, TOperator>(TensorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : ITensorIndexOperator<T> =>
        Generate(default(StaticTensorIndexOperator<T, TOperator>), destination);

    /// <summary>
    /// Populates every element of <paramref name="destination"/> using a stateful tensor index operator.
    /// <para>状態を持つテンソルインデックス演算子でテンソルの全要素を設定します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful tensor index operator type. / 状態を持つテンソルインデックス演算子の型。</typeparam>
    /// <param name="op">The operator computing each element from its indices. / 全軸インデックスから各要素を計算する演算子。</param>
    /// <param name="destination">The destination tensor span. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    [InferenceOverloads]
    public static TensorSpan<T> Generate<T, TOperator>(TOperator op, TensorSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulTensorIndexOperator<T>
    {
        StridedKernels.Generate(destination, op);
        return destination;
    }

    /// <summary>
    /// Populates every element of <paramref name="destination"/> using a delegate over multi-dimensional indices.
    /// <para>全軸のインデックスを受け取るデリゲートを用いてテンソルの全要素を設定します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="op">The delegate computing each element from zero-based indices in shape order (the index span is valid only during the call). / 形状と同じ軸順の 0 始まりインデックスから各要素を計算するデリゲート（スパンは呼び出し中のみ有効）。</param>
    /// <param name="destination">The destination tensor span. / 結果を書き込む出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="op"/> is <see langword="null"/>. / <paramref name="op"/> が <see langword="null"/> の場合。</exception>
    [InferenceOverloads]
    public static TensorSpan<T> Generate<T>(Func<ReadOnlySpan<int>, T> op, TensorSpan<T> destination)
        where T : unmanaged, INumberBase<T> =>
        Generate(new DelegateTensorIndexOperator<T>(op), destination);

    /// <summary>
    /// Populates a banded matrix using a stateless index operator for entries with <c>−lowerBandwidth ≤ j − i ≤ upperBandwidth</c> and zeros elsewhere.
    /// <para>状態を持たない演算子で <c>−lowerBandwidth ≤ j − i ≤ upperBandwidth</c> の帯内要素を設定し、帯外を 0 で埋めます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless matrix index operator type. / 状態を持たない行列インデックス演算子の型。</typeparam>
    /// <param name="lowerBandwidth">The number of non-zero subdiagonals (must be non-negative). / 主対角より下の副対角の本数（0 以上）。</param>
    /// <param name="upperBandwidth">The number of non-zero superdiagonals (must be non-negative). / 主対角より上の副対角の本数（0 以上）。</param>
    /// <param name="destination">The dense destination matrix span (entries outside the band are overwritten with zero). / 結果を書き込む密行列の出力先（帯外も 0 で上書き）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>The operator is evaluated only within the band. To allocate a new matrix, use <see cref="Matrix{TElement}.GenerateBand{TOperator}(int, int, int, int)"/>.</para>
    /// <para>演算子は帯内の要素のみ評価されます。新しい行列を割り当てて生成するには <see cref="Matrix{TElement}.GenerateBand{TOperator}(int, int, int, int)"/> を使用してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="lowerBandwidth"/> or <paramref name="upperBandwidth"/> is negative. / 帯幅が負の場合。</exception>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static MatrixSpan<T> GenerateBand<T, TOperator>(int lowerBandwidth, int upperBandwidth, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IIndexOperator<T>
    {
        GenerateBand(lowerBandwidth, upperBandwidth, default(StaticIndexOperator<T, TOperator>), destination);
        return destination;
    }

    /// <summary>
    /// Populates a banded matrix using a stateful index operator for entries with <c>−lowerBandwidth ≤ j − i ≤ upperBandwidth</c> and zeros elsewhere.
    /// <para>状態を持つ演算子で <c>−lowerBandwidth ≤ j − i ≤ upperBandwidth</c> の帯内要素を設定し、帯外を 0 で埋めます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful matrix index operator type. / 状態を持つ行列インデックス演算子の型。</typeparam>
    /// <param name="lowerBandwidth">The number of non-zero subdiagonals (must be non-negative). / 主対角より下の副対角の本数（0 以上）。</param>
    /// <param name="upperBandwidth">The number of non-zero superdiagonals (must be non-negative). / 主対角より上の副対角の本数（0 以上）。</param>
    /// <param name="op">The operator computing each band element. / 帯内の各要素を計算する演算子。</param>
    /// <param name="destination">The dense destination matrix span (entries outside the band are overwritten with zero). / 結果を書き込む密行列の出力先（帯外も 0 で上書き）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>To allocate a new matrix, use <see cref="Matrix{TElement}.GenerateBand{TOperator}(int, int, int, int, TOperator)"/>.</para>
    /// <para>新しい行列を割り当てて生成するには <see cref="Matrix{TElement}.GenerateBand{TOperator}(int, int, int, int, TOperator)"/> を使用してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="lowerBandwidth"/> or <paramref name="upperBandwidth"/> is negative. / 帯幅が負の場合。</exception>
    [InferenceOverloads]
    public static MatrixSpan<T> GenerateBand<T, TOperator>(int lowerBandwidth, int upperBandwidth, TOperator op, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulIndexOperator<T>
    {
        ArgumentOutOfRangeException.ThrowIfNegative(lowerBandwidth);
        ArgumentOutOfRangeException.ThrowIfNegative(upperBandwidth);
        StridedKernels.Generate(destination, lowerBandwidth, upperBandwidth, symmetric: false,
            StridedKernels.GeneratedElementWork, new IndexSource<T, TOperator>(op));
        return destination;
    }

    /// <summary>
    /// Populates a banded matrix using a delegate for entries within the band and zeros elsewhere.
    /// <para>デリゲートを用いて帯内の要素を設定し、帯外を 0 で埋めます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="lowerBandwidth">The number of non-zero subdiagonals (must be non-negative). / 主対角より下の副対角の本数（0 以上）。</param>
    /// <param name="upperBandwidth">The number of non-zero superdiagonals (must be non-negative). / 主対角より上の副対角の本数（0 以上）。</param>
    /// <param name="op">The delegate computing each band element. / 帯内の各要素を計算するデリゲート。</param>
    /// <param name="destination">The dense destination matrix span (entries outside the band are overwritten with zero). / 結果を書き込む密行列の出力先（帯外も 0 で上書き）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>For large matrices, <paramref name="op"/> may be invoked concurrently from multiple threads in arbitrary order.</para>
    /// <para>大きな行列では、<paramref name="op"/> が複数スレッドから順不同に呼び出される場合があります。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="op"/> is <see langword="null"/>. / <paramref name="op"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="lowerBandwidth"/> or <paramref name="upperBandwidth"/> is negative. / 帯幅が負の場合。</exception>
    [InferenceOverloads]
    public static MatrixSpan<T> GenerateBand<T>(int lowerBandwidth, int upperBandwidth, Func<int, int, T> op, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T> =>
        GenerateBand(lowerBandwidth, upperBandwidth, new DelegateIndexOperator<T>(op), destination);

    /// <summary>
    /// Populates a symmetric matrix by evaluating a stateless index operator on the lower triangle (<c>i ≥ j</c>) and mirroring to the upper triangle.
    /// <para>状態を持たない演算子で下三角（<c>i ≥ j</c>）のみを評価し、上三角へ対称に転記して対称行列を生成します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless matrix index operator type. / 状態を持たない行列インデックス演算子の型。</typeparam>
    /// <param name="destination">The square destination matrix span. / 結果を書き込む正方行列の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>Evaluates only <c>n(n + 1) / 2</c> entries; the result is always symmetric regardless of whether <typeparamref name="TOperator"/> is symmetric. To allocate a new matrix, use <see cref="Matrix{TElement}.GenerateSymmetric{TOperator}(int)"/>.</para>
    /// <para>評価回数は <c>n(n + 1) / 2</c> 回のみで、演算の対称性にかかわらず結果は常に対称行列となります。新しい行列を割り当てて生成するには <see cref="Matrix{TElement}.GenerateSymmetric{TOperator}(int)"/> を使用してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="destination"/> is not square. / <paramref name="destination"/> が正方行列でない場合。</exception>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static MatrixSpan<T> GenerateSymmetric<T, TOperator>(MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IIndexOperator<T>
    {
        GenerateSymmetric(default(StaticIndexOperator<T, TOperator>), destination);
        return destination;
    }

    /// <summary>
    /// Populates a symmetric matrix by evaluating a stateful index operator on the lower triangle (<c>i ≥ j</c>) and mirroring to the upper triangle.
    /// <para>状態を持つ演算子で下三角（<c>i ≥ j</c>）のみを評価し、上三角へ対称に転記して対称行列を生成します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful matrix index operator type. / 状態を持つ行列インデックス演算子の型。</typeparam>
    /// <param name="op">The operator computing each lower-triangular element. / 下三角の各要素を計算する演算子。</param>
    /// <param name="destination">The square destination matrix span. / 結果を書き込む正方行列の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>To allocate a new matrix, use <see cref="Matrix{TElement}.GenerateSymmetric{TOperator}(int, TOperator)"/>.</para>
    /// <para>新しい行列を割り当てて生成するには <see cref="Matrix{TElement}.GenerateSymmetric{TOperator}(int, TOperator)"/> を使用してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="destination"/> is not square. / <paramref name="destination"/> が正方行列でない場合。</exception>
    [InferenceOverloads]
    public static MatrixSpan<T> GenerateSymmetric<T, TOperator>(TOperator op, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulIndexOperator<T>
    {
        GenerateSymmetric(destination.Rows, op, destination);
        return destination;
    }

    /// <summary>
    /// Populates a symmetric matrix by evaluating a delegate on the lower triangle (<c>i ≥ j</c>) and mirroring to the upper triangle.
    /// <para>デリゲートを用いて下三角（<c>i ≥ j</c>）のみを評価し、上三角へ対称に転記して対称行列を生成します。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="op">The delegate computing each lower-triangular element. / 下三角の各要素を計算するデリゲート。</param>
    /// <param name="destination">The square destination matrix span. / 結果を書き込む正方行列の出力先。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>For large matrices, <paramref name="op"/> may be invoked concurrently from multiple threads in arbitrary order.</para>
    /// <para>大きな行列では、<paramref name="op"/> が複数スレッドから順不同に呼び出される場合があります。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="op"/> is <see langword="null"/>. / <paramref name="op"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="destination"/> is not square. / <paramref name="destination"/> が正方行列でない場合。</exception>
    [InferenceOverloads]
    public static MatrixSpan<T> GenerateSymmetric<T>(Func<int, int, T> op, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T> =>
        GenerateSymmetric(new DelegateIndexOperator<T>(op), destination);

    /// <summary>
    /// Populates a symmetric banded matrix by evaluating a stateless index operator for <c>0 ≤ i − j ≤ bandwidth</c>, mirroring to the upper triangle, and zeroing entries outside the band.
    /// <para>状態を持たない演算子で <c>0 ≤ i − j ≤ bandwidth</c> の下三角帯内のみを評価し、上三角へ対称に転記して帯外を 0 で埋めます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateless matrix index operator type. / 状態を持たない行列インデックス演算子の型。</typeparam>
    /// <param name="bandwidth">The number of non-zero subdiagonals and superdiagonals (must be non-negative). / 主対角の上下の副対角の本数（0 以上）。</param>
    /// <param name="destination">The square destination matrix span (entries outside the band are overwritten with zero). / 結果を書き込む正方行列の出力先（帯外も 0 で上書き）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>To allocate a new matrix, use <see cref="Matrix{TElement}.GenerateSymmetric{TOperator}(int, int)"/>.</para>
    /// <para>新しい行列を割り当てて生成するには <see cref="Matrix{TElement}.GenerateSymmetric{TOperator}(int, int)"/> を使用してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="bandwidth"/> is negative. / <paramref name="bandwidth"/> が負の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="destination"/> is not square. / <paramref name="destination"/> が正方行列でない場合。</exception>
    [FixedElementTypeOverloads(typeof(Half), typeof(float), typeof(double), typeof(decimal), typeof(Complex))]
    public static MatrixSpan<T> GenerateSymmetric<T, TOperator>(int bandwidth, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IIndexOperator<T>
    {
        GenerateSymmetric(bandwidth, default(StaticIndexOperator<T, TOperator>), destination);
        return destination;
    }

    /// <summary>
    /// Populates a symmetric banded matrix by evaluating a stateful index operator for <c>0 ≤ i − j ≤ bandwidth</c>, mirroring to the upper triangle, and zeroing entries outside the band.
    /// <para>状態を持つ演算子で <c>0 ≤ i − j ≤ bandwidth</c> の下三角帯内のみを評価し、上三角へ対称に転記して帯外を 0 で埋めます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <typeparam name="TOperator">The stateful matrix index operator type. / 状態を持つ行列インデックス演算子の型。</typeparam>
    /// <param name="bandwidth">The number of non-zero subdiagonals and superdiagonals (must be non-negative). / 主対角の上下の副対角の本数（0 以上）。</param>
    /// <param name="op">The operator computing each lower-band element. / 下三角帯内の各要素を計算する演算子。</param>
    /// <param name="destination">The square destination matrix span (entries outside the band are overwritten with zero). / 結果を書き込む正方行列の出力先（帯外も 0 で上書き）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>To allocate a new matrix, use <see cref="Matrix{TElement}.GenerateSymmetric{TOperator}(int, int, TOperator)"/>.</para>
    /// <para>新しい行列を割り当てて生成するには <see cref="Matrix{TElement}.GenerateSymmetric{TOperator}(int, int, TOperator)"/> を使用してください。</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="bandwidth"/> is negative. / <paramref name="bandwidth"/> が負の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="destination"/> is not square. / <paramref name="destination"/> が正方行列でない場合。</exception>
    [InferenceOverloads]
    public static MatrixSpan<T> GenerateSymmetric<T, TOperator>(int bandwidth, TOperator op, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T>
        where TOperator : IStatefulIndexOperator<T>
    {
        ArgumentOutOfRangeException.ThrowIfNegative(bandwidth);
        if (destination.Rows != destination.Columns)
        {
            throw new ArgumentException("The destination must be square.", nameof(destination));
        }

        StridedKernels.Generate(destination, bandwidth, bandwidth, symmetric: true,
            StridedKernels.GeneratedElementWork, new IndexSource<T, TOperator>(op));
        return destination;
    }

    /// <summary>
    /// Populates a symmetric banded matrix using a delegate for <c>0 ≤ i − j ≤ bandwidth</c>, mirroring to the upper triangle, and zeroing entries outside the band.
    /// <para>デリゲートを用いて <c>0 ≤ i − j ≤ bandwidth</c> の下三角帯内のみを評価し、上三角へ対称に転記して帯外を 0 で埋めます。</para>
    /// </summary>
    /// <typeparam name="T">The element type. / 要素型。</typeparam>
    /// <param name="bandwidth">The number of non-zero subdiagonals and superdiagonals (must be non-negative). / 主対角の上下の副対角の本数（0 以上）。</param>
    /// <param name="op">The delegate computing each lower-band element. / 下三角帯内の各要素を計算するデリゲート。</param>
    /// <param name="destination">The square destination matrix span (entries outside the band are overwritten with zero). / 結果を書き込む正方行列の出力先（帯外も 0 で上書き）。</param>
    /// <returns><paramref name="destination"/>. / <paramref name="destination"/>。</returns>
    /// <remarks>
    /// <para>For large matrices, <paramref name="op"/> may be invoked concurrently from multiple threads in arbitrary order.</para>
    /// <para>大きな行列では、<paramref name="op"/> が複数スレッドから順不同に呼び出される場合があります。</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="op"/> is <see langword="null"/>. / <paramref name="op"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="bandwidth"/> is negative. / <paramref name="bandwidth"/> が負の場合。</exception>
    /// <exception cref="ArgumentException"><paramref name="destination"/> is not square. / <paramref name="destination"/> が正方行列でない場合。</exception>
    [InferenceOverloads]
    public static MatrixSpan<T> GenerateSymmetric<T>(int bandwidth, Func<int, int, T> op, MatrixSpan<T> destination)
        where T : unmanaged, INumberBase<T> =>
        GenerateSymmetric(bandwidth, new DelegateIndexOperator<T>(op), destination);
}
