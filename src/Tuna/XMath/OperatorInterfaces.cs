namespace Tuna;

/// <summary>
/// Represents a stateless unary elementwise operator.
/// <para>状態を持たない1引数の要素ごと演算を表します。</para>
/// </summary>
/// <typeparam name="T">The element type. / 要素型。</typeparam>
/// <remarks>
/// <para>Pass implementations as a type argument to <c>XMath.Map</c> (for example, <c>XMath.Map&lt;Softplus&gt;(x)</c> for standard element types, or <c>XMath.Map&lt;int, Square&gt;(x)</c> for others). Implement <see cref="IStatefulUnaryOperator{T}"/> when the operator requires state.</para>
/// <para><c>XMath.Map</c> に型引数として渡します（標準の要素型では <c>XMath.Map&lt;Softplus&gt;(x)</c>、それ以外の型では <c>XMath.Map&lt;int, Square&gt;(x)</c>）。係数などの状態が必要な場合は <see cref="IStatefulUnaryOperator{T}"/> を実装してください。</para>
/// </remarks>
public interface IUnaryOperator<T>
{
    /// <summary>
    /// Applies the operator to a single element.
    /// <para>1つの要素に演算を適用します。</para>
    /// </summary>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns>The transformed value. / 演算結果の値。</returns>
    static abstract T Invoke(T x);
}

/// <summary>
/// Represents a stateful unary elementwise operator.
/// <para>状態（係数など）を持つ1引数の要素ごと演算を表します。</para>
/// </summary>
/// <typeparam name="T">The element type. / 要素型。</typeparam>
/// <remarks>
/// <para>Implement as a value type (<c>struct</c>) and pass by value (for example, <c>XMath.Map(x, new Shift(0.5))</c>).</para>
/// <para>構造体として実装し、値として渡します（例: <c>XMath.Map(x, new Shift(0.5))</c>）。</para>
/// </remarks>
public interface IStatefulUnaryOperator<T>
{
    /// <summary>
    /// Applies the operator to a single element.
    /// <para>1つの要素に演算を適用します。</para>
    /// </summary>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns>The transformed value. / 演算結果の値。</returns>
    T Invoke(T x);
}

/// <summary>
/// Represents a stateless binary elementwise operator.
/// <para>状態を持たない2引数の要素ごと演算を表します。</para>
/// </summary>
/// <typeparam name="T">The element type. / 要素型。</typeparam>
/// <remarks>
/// <para>Pass implementations as a type argument to <c>XMath.Zip</c> (for example, <c>XMath.Zip&lt;Atan2&gt;(y, x)</c>). Implement <see cref="IStatefulBinaryOperator{T}"/> when the operator requires state.</para>
/// <para><c>XMath.Zip</c> に型引数として渡します（例: <c>XMath.Zip&lt;Atan2&gt;(y, x)</c>）。状態が必要な場合は <see cref="IStatefulBinaryOperator{T}"/> を実装してください。</para>
/// </remarks>
public interface IBinaryOperator<T>
{
    /// <summary>
    /// Applies the operator to a pair of elements.
    /// <para>2つの要素に演算を適用します。</para>
    /// </summary>
    /// <param name="x">The first input value. / 第1入力値。</param>
    /// <param name="y">The second input value. / 第2入力値。</param>
    /// <returns>The computed value. / 演算結果の値。</returns>
    static abstract T Invoke(T x, T y);
}

/// <summary>
/// Represents a stateful binary elementwise operator.
/// <para>状態（係数など）を持つ2引数の要素ごと演算を表します。</para>
/// </summary>
/// <typeparam name="T">The element type. / 要素型。</typeparam>
/// <remarks>
/// <para>Implement as a value type (<c>struct</c>) and pass by value (for example, <c>XMath.Zip(x, y, new Weighted(0.5))</c>).</para>
/// <para>構造体として実装し、値として渡します（例: <c>XMath.Zip(x, y, new Weighted(0.5))</c>）。</para>
/// </remarks>
public interface IStatefulBinaryOperator<T>
{
    /// <summary>
    /// Applies the operator to a pair of elements.
    /// <para>2つの要素に演算を適用します。</para>
    /// </summary>
    /// <param name="x">The first input value. / 第1入力値。</param>
    /// <param name="y">The second input value. / 第2入力値。</param>
    /// <returns>The computed value. / 演算結果の値。</returns>
    T Invoke(T x, T y);
}

/// <summary>
/// Represents a stateless ternary elementwise operator.
/// <para>状態を持たない3引数の要素ごと演算を表します。</para>
/// </summary>
/// <typeparam name="T">The element type. / 要素型。</typeparam>
/// <remarks>
/// <para>Pass implementations as a type argument to <c>XMath.Zip</c> (for example, <c>XMath.Zip&lt;Lerp&gt;(x, y, t)</c>). Implement <see cref="IStatefulTernaryOperator{T}"/> when the operator requires state.</para>
/// <para><c>XMath.Zip</c> に型引数として渡します（例: <c>XMath.Zip&lt;Lerp&gt;(x, y, t)</c>）。状態が必要な場合は <see cref="IStatefulTernaryOperator{T}"/> を実装してください。</para>
/// </remarks>
public interface ITernaryOperator<T>
{
    /// <summary>
    /// Applies the operator to three elements.
    /// <para>3つの要素に演算を適用します。</para>
    /// </summary>
    /// <param name="x">The first input value. / 第1入力値。</param>
    /// <param name="y">The second input value. / 第2入力値。</param>
    /// <param name="z">The third input value. / 第3入力値。</param>
    /// <returns>The computed value. / 演算結果の値。</returns>
    static abstract T Invoke(T x, T y, T z);
}

/// <summary>
/// Represents a stateful ternary elementwise operator.
/// <para>状態（係数など）を持つ3引数の要素ごと演算を表します。</para>
/// </summary>
/// <typeparam name="T">The element type. / 要素型。</typeparam>
/// <remarks>
/// <para>Implement as a value type (<c>struct</c>) and pass by value (for example, <c>XMath.Zip(x, y, z, new Clamp(0.5))</c>).</para>
/// <para>構造体として実装し、値として渡します（例: <c>XMath.Zip(x, y, z, new Clamp(0.5))</c>）。</para>
/// </remarks>
public interface IStatefulTernaryOperator<T>
{
    /// <summary>
    /// Applies the operator to three elements.
    /// <para>3つの要素に演算を適用します。</para>
    /// </summary>
    /// <param name="x">The first input value. / 第1入力値。</param>
    /// <param name="y">The second input value. / 第2入力値。</param>
    /// <param name="z">The third input value. / 第3入力値。</param>
    /// <returns>The computed value. / 演算結果の値。</returns>
    T Invoke(T x, T y, T z);
}

/// <summary>
/// Represents a stateless operator that maps two column vectors to a scalar (such as a distance or kernel function).
/// <para>2つの列ベクトルから1つのスカラー値を計算する、状態を持たない演算（距離関数やカーネル関数など）を表します。</para>
/// </summary>
/// <typeparam name="T">The element type. / 要素型。</typeparam>
/// <remarks>
/// <para>Pass implementations as a type argument to <c>XMath.Pairwise</c>. For large matrices, <see cref="Invoke"/> may be called concurrently in arbitrary order, so it must be pure and thread-safe.</para>
/// <para><c>XMath.Pairwise</c> に型引数として渡します。大きな行列では <see cref="Invoke"/> が複数スレッドから順不同に呼び出されるため、引数のみに依存するスレッドセーフな実装にしてください。</para>
/// </remarks>
public interface IPairwiseOperator<T> where T : unmanaged, System.Numerics.INumberBase<T>
{
    /// <summary>
    /// Applies the operator to a pair of vectors.
    /// <para>2つのベクトルに演算を適用します。</para>
    /// </summary>
    /// <param name="x">The first vector (a column of the first matrix). / 第1ベクトル（第1行列の1列）。</param>
    /// <param name="y">The second vector (a column of the second matrix). / 第2ベクトル（第2行列の1列）。</param>
    /// <returns>The computed scalar value. / 計算結果のスカラー値。</returns>
    static abstract T Invoke(LinearAlgebra.ReadOnlyVectorSpan<T> x, LinearAlgebra.ReadOnlyVectorSpan<T> y);
}

/// <summary>
/// Represents a stateful operator that maps two column vectors to a scalar.
/// <para>状態（長さ尺度など）を持ち、2つの列ベクトルから1つのスカラー値を計算する演算を表します。</para>
/// </summary>
/// <typeparam name="T">The element type. / 要素型。</typeparam>
/// <remarks>
/// <para>Implement as a value type (<c>struct</c>) and pass by value to <c>XMath.Pairwise</c>. For large matrices, copies of the operator may be invoked concurrently in arbitrary order, so state must be read-only.</para>
/// <para>構造体として実装し、<c>XMath.Pairwise</c> に値として渡します。大きな行列では演算子のコピーが複数スレッドから順不同に呼び出されるため、状態は読み取り専用にしてください。</para>
/// </remarks>
public interface IStatefulPairwiseOperator<T> where T : unmanaged, System.Numerics.INumberBase<T>
{
    /// <summary>
    /// Applies the operator to a pair of vectors.
    /// <para>2つのベクトルに演算を適用します。</para>
    /// </summary>
    /// <param name="x">The first vector (a column of the first matrix). / 第1ベクトル（第1行列の1列）。</param>
    /// <param name="y">The second vector (a column of the second matrix). / 第2ベクトル（第2行列の1列）。</param>
    /// <returns>The computed scalar value. / 計算結果のスカラー値。</returns>
    T Invoke(LinearAlgebra.ReadOnlyVectorSpan<T> x, LinearAlgebra.ReadOnlyVectorSpan<T> y);
}

/// <summary>
/// Represents a stateless operator that computes a matrix element from its row and column indices.
/// <para>行と列のインデックスから行列の要素を計算する、状態を持たない演算を表します。</para>
/// </summary>
/// <typeparam name="T">The element type. / 要素型。</typeparam>
/// <remarks>
/// <para>Pass implementations as a type argument to <c>XMath.Generate</c> or <c>Matrix&lt;T&gt;.Generate</c>. For large matrices, <see cref="Invoke"/> may be called concurrently in arbitrary order.</para>
/// <para><c>XMath.Generate</c> や <c>Matrix&lt;T&gt;.Generate</c> に型引数として渡します。大きな行列では <see cref="Invoke"/> が複数スレッドから順不同に呼び出されます。</para>
/// </remarks>
public interface IIndexOperator<T> where T : unmanaged, System.Numerics.INumberBase<T>
{
    /// <summary>
    /// Computes the element at the specified zero-based row and column indices.
    /// <para>指定した 0 始まりの行・列インデックスに対応する要素を計算します。</para>
    /// </summary>
    /// <param name="row">The zero-based row index. / 0 始まりの行インデックス。</param>
    /// <param name="column">The zero-based column index. / 0 始まりの列インデックス。</param>
    /// <returns>The computed element value. / 計算結果の要素値。</returns>
    static abstract T Invoke(int row, int column);
}

/// <summary>
/// Represents a stateful operator that computes a matrix element from its row and column indices.
/// <para>状態（係数や節点情報など）を持ち、行と列のインデックスから行列の要素を計算する演算を表します。</para>
/// </summary>
/// <typeparam name="T">The element type. / 要素型。</typeparam>
/// <remarks>
/// <para>Implement as a value type (<c>struct</c>) and pass by value to <c>XMath.Generate</c> or <c>Matrix&lt;T&gt;.Generate</c>. For large matrices, copies of the operator may be invoked concurrently in arbitrary order, so state must be read-only.</para>
/// <para>構造体として実装し、<c>XMath.Generate</c> や <c>Matrix&lt;T&gt;.Generate</c> に値として渡します。大きな行列では演算子のコピーが複数スレッドから順不同に呼び出されるため、状態は読み取り専用にしてください。</para>
/// </remarks>
public interface IStatefulIndexOperator<T> where T : unmanaged, System.Numerics.INumberBase<T>
{
    /// <summary>
    /// Computes the element at the specified zero-based row and column indices.
    /// <para>指定した 0 始まりの行・列インデックスに対応する要素を計算します。</para>
    /// </summary>
    /// <param name="row">The zero-based row index. / 0 始まりの行インデックス。</param>
    /// <param name="column">The zero-based column index. / 0 始まりの列インデックス。</param>
    /// <returns>The computed element value. / 計算結果の要素値。</returns>
    T Invoke(int row, int column);
}

/// <summary>
/// Represents a stateless operator that computes a vector element from its index.
/// <para>インデックスからベクトルの要素を計算する、状態を持たない演算を表します。</para>
/// </summary>
/// <typeparam name="T">The element type. / 要素型。</typeparam>
public interface IVectorIndexOperator<T> where T : unmanaged, System.Numerics.INumberBase<T>
{
    /// <summary>
    /// Computes the element at the specified zero-based index.
    /// <para>指定した 0 始まりのインデックスに対応する要素を計算します。</para>
    /// </summary>
    /// <param name="index">The zero-based element index. / 0 始まりの要素インデックス。</param>
    /// <returns>The computed element value. / 計算結果の要素値。</returns>
    static abstract T Invoke(int index);
}

/// <summary>
/// Represents a stateful operator that computes a vector element from its index.
/// <para>状態を持ち、インデックスからベクトルの要素を計算する演算を表します。</para>
/// </summary>
/// <typeparam name="T">The element type. / 要素型。</typeparam>
public interface IStatefulVectorIndexOperator<T> where T : unmanaged, System.Numerics.INumberBase<T>
{
    /// <summary>
    /// Computes the element at the specified zero-based index.
    /// <para>指定した 0 始まりのインデックスに対応する要素を計算します。</para>
    /// </summary>
    /// <param name="index">The zero-based element index. / 0 始まりの要素インデックス。</param>
    /// <returns>The computed element value. / 計算結果の要素値。</returns>
    T Invoke(int index);
}

/// <summary>
/// Represents a stateless operator that computes a tensor element from its multi-dimensional indices.
/// <para>全軸のインデックスからテンソルの要素を計算する、状態を持たない演算を表します。</para>
/// </summary>
/// <typeparam name="T">The element type. / 要素型。</typeparam>
public interface ITensorIndexOperator<T> where T : unmanaged, System.Numerics.INumberBase<T>
{
    /// <summary>
    /// Computes the element at the specified zero-based indices.
    /// <para>指定した 0 始まりの全軸インデックスに対応する要素を計算します。</para>
    /// </summary>
    /// <param name="indices">The zero-based indices in shape order, valid only for the duration of the call. / 形状と同じ軸順の 0 始まりインデックス（呼び出しの間のみ有効）。</param>
    /// <returns>The computed element value. / 計算結果の要素値。</returns>
    static abstract T Invoke(ReadOnlySpan<int> indices);
}

/// <summary>
/// Represents a stateful operator that computes a tensor element from its multi-dimensional indices.
/// <para>状態を持ち、全軸のインデックスからテンソルの要素を計算する演算を表します。</para>
/// </summary>
/// <typeparam name="T">The element type. / 要素型。</typeparam>
public interface IStatefulTensorIndexOperator<T> where T : unmanaged, System.Numerics.INumberBase<T>
{
    /// <summary>
    /// Computes the element at the specified zero-based indices.
    /// <para>指定した 0 始まりの全軸インデックスに対応する要素を計算します。</para>
    /// </summary>
    /// <param name="indices">The zero-based indices in shape order, valid only for the duration of the call. / 形状と同じ軸順の 0 始まりインデックス（呼び出しの間のみ有効）。</param>
    /// <returns>The computed element value. / 計算結果の要素値。</returns>
    T Invoke(ReadOnlySpan<int> indices);
}
