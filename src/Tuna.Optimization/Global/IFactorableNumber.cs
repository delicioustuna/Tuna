using System.Numerics;

namespace Tuna.Optimization.Global;

/// <summary>
/// Represents a number type supporting arithmetic and univariate elementary operations for factorable functions, implemented by <see cref="Interval"/>, <see cref="McCormick"/>, and <see cref="GradientDual"/>.
/// <para>分解可能な関数（四則演算と1変数初等関数の合成）を評価するための数値型インターフェースです。<see cref="Interval"/>（区間）、<see cref="McCormick"/>（McCormick 緩和）、<see cref="GradientDual"/>（値と勾配）が実装します。</para>
/// </summary>
/// <typeparam name="TSelf">The implementing number type. / 実装する数値型自身。</typeparam>
/// <remarks>
/// <para>Writing an objective once as <see cref="IFactorableFunction.Evaluate{T}"/> enables interval bounds, convex relaxations, and exact gradients from the same formula. <see cref="double"/> constants convert implicitly. Prefer <see cref="Square"/> over <c>x * x</c> for tighter intervals and relaxations.</para>
/// <para>目的関数を <see cref="IFactorableFunction.Evaluate{T}"/> として1度記述するだけで、同一の式から値の範囲・凸緩和・正確な勾配を計算できます。<see cref="double"/> 定数は暗黙的に変換されます。同一変数の相関を活かしてより狭い区間・緩和を得るには、<c>x * x</c> ではなく <see cref="Square"/> を使用してください。</para>
/// </remarks>
public interface IFactorableNumber<TSelf> :
    IAdditionOperators<TSelf, TSelf, TSelf>,
    ISubtractionOperators<TSelf, TSelf, TSelf>,
    IMultiplyOperators<TSelf, TSelf, TSelf>,
    IDivisionOperators<TSelf, TSelf, TSelf>,
    IUnaryNegationOperators<TSelf, TSelf>
    where TSelf : IFactorableNumber<TSelf>
{
    /// <summary>
    /// Implicitly converts a scalar constant to <typeparamref name="TSelf"/>.
    /// <para>スカラー定数を <typeparamref name="TSelf"/> に暗黙的に変換します。</para>
    /// </summary>
    /// <param name="value">The constant value. / 定数値。</param>
    static abstract implicit operator TSelf(double value);

    /// <summary>
    /// Adds a scalar constant to a value (<c>x + c</c>).
    /// <para>値にスカラー定数を加算します（<c>x + c</c>）。</para>
    /// </summary>
    /// <param name="x">The left operand. / 左辺。</param>
    /// <param name="c">The scalar constant. / スカラー定数。</param>
    /// <returns>The sum <c>x + c</c>. / 和 <c>x + c</c>。</returns>
    static abstract TSelf operator +(TSelf x, double c);

    /// <summary>
    /// Adds a value to a scalar constant (<c>c + x</c>).
    /// <para>スカラー定数に値を加算します（<c>c + x</c>）。</para>
    /// </summary>
    /// <param name="c">The scalar constant. / スカラー定数。</param>
    /// <param name="x">The right operand. / 右辺。</param>
    /// <returns>The sum <c>c + x</c>. / 和 <c>c + x</c>。</returns>
    static abstract TSelf operator +(double c, TSelf x);

    /// <summary>
    /// Subtracts a scalar constant from a value (<c>x − c</c>).
    /// <para>値からスカラー定数を減算します（<c>x − c</c>）。</para>
    /// </summary>
    /// <param name="x">The left operand. / 左辺。</param>
    /// <param name="c">The scalar constant. / スカラー定数。</param>
    /// <returns>The difference <c>x − c</c>. / 差 <c>x − c</c>。</returns>
    static abstract TSelf operator -(TSelf x, double c);

    /// <summary>
    /// Subtracts a value from a scalar constant (<c>c − x</c>).
    /// <para>スカラー定数から値を減算します（<c>c − x</c>）。</para>
    /// </summary>
    /// <param name="c">The scalar constant. / スカラー定数。</param>
    /// <param name="x">The right operand. / 右辺。</param>
    /// <returns>The difference <c>c − x</c>. / 差 <c>c − x</c>。</returns>
    static abstract TSelf operator -(double c, TSelf x);

    /// <summary>
    /// Multiplies a value by a scalar constant (<c>x · c</c>).
    /// <para>値にスカラー定数を掛けます（<c>x · c</c>）。</para>
    /// </summary>
    /// <param name="x">The left operand. / 左辺。</param>
    /// <param name="c">The scalar constant. / スカラー定数。</param>
    /// <returns>The product <c>x · c</c>. / 積 <c>x · c</c>。</returns>
    static abstract TSelf operator *(TSelf x, double c);

    /// <summary>
    /// Multiplies a scalar constant by a value (<c>c · x</c>).
    /// <para>スカラー定数に値を掛けます（<c>c · x</c>）。</para>
    /// </summary>
    /// <param name="c">The scalar constant. / スカラー定数。</param>
    /// <param name="x">The right operand. / 右辺。</param>
    /// <returns>The product <c>c · x</c>. / 積 <c>c · x</c>。</returns>
    static abstract TSelf operator *(double c, TSelf x);

    /// <summary>
    /// Divides a value by a scalar constant (<c>x / c</c>).
    /// <para>値をスカラー定数で割ります（<c>x / c</c>）。</para>
    /// </summary>
    /// <param name="x">The dividend. / 被除数。</param>
    /// <param name="c">The scalar divisor. / スカラー除数。</param>
    /// <returns>The quotient <c>x / c</c>. / 商 <c>x / c</c>。</returns>
    static abstract TSelf operator /(TSelf x, double c);

    /// <summary>
    /// Divides a scalar constant by a value (<c>c / x</c>).
    /// <para>スカラー定数を値で割ります（<c>c / x</c>）。</para>
    /// </summary>
    /// <param name="c">The scalar dividend. / スカラー被除数。</param>
    /// <param name="x">The divisor. / 除数。</param>
    /// <returns>The quotient <c>c / x</c>. / 商 <c>c / x</c>。</returns>
    static abstract TSelf operator /(double c, TSelf x);

    /// <summary>
    /// Computes the square <c>x²</c>.
    /// <para>2乗 <c>x²</c> を計算します。</para>
    /// </summary>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns>The square <c>x²</c>. / 2乗 <c>x²</c>。</returns>
    static abstract TSelf Square(TSelf x);

    /// <summary>
    /// Computes the integer power <c>xⁿ</c> (with <c>1 / x^|n|</c> for negative <paramref name="n"/>).
    /// <para>整数乗 <c>xⁿ</c>（負の <paramref name="n"/> では <c>1 / x^|n|</c>）を計算します。</para>
    /// </summary>
    /// <param name="x">The base value. / 底。</param>
    /// <param name="n">The integer exponent. / 整数指数。</param>
    /// <returns>The power <c>xⁿ</c>. / べき乗 <c>xⁿ</c>。</returns>
    static abstract TSelf Pow(TSelf x, int n);

    /// <summary>
    /// Computes the square root <c>√x</c> (for <c>x ≥ 0</c>).
    /// <para>平方根 <c>√x</c>（<c>x ≥ 0</c> の範囲）を計算します。</para>
    /// </summary>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns>The square root <c>√x</c>. / 平方根 <c>√x</c>。</returns>
    static abstract TSelf Sqrt(TSelf x);

    /// <summary>
    /// Computes the exponential <c>eˣ</c>.
    /// <para>指数関数 <c>eˣ</c> を計算します。</para>
    /// </summary>
    /// <param name="x">The exponent. / 指数。</param>
    /// <returns>The exponential <c>eˣ</c>. / 指数関数値 <c>eˣ</c>。</returns>
    static abstract TSelf Exp(TSelf x);

    /// <summary>
    /// Computes the natural logarithm <c>ln x</c> (for <c>x &gt; 0</c>).
    /// <para>自然対数 <c>ln x</c>（<c>x &gt; 0</c> の範囲）を計算します。</para>
    /// </summary>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns>The natural logarithm <c>ln x</c>. / 自然対数 <c>ln x</c>。</returns>
    static abstract TSelf Log(TSelf x);

    /// <summary>
    /// Computes the absolute value <c>|x|</c>.
    /// <para>絶対値 <c>|x|</c> を計算します。</para>
    /// </summary>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns>The absolute value <c>|x|</c>. / 絶対値 <c>|x|</c>。</returns>
    static abstract TSelf Abs(TSelf x);

    /// <summary>
    /// Computes the reciprocal <c>1 / x</c>.
    /// <para>逆数 <c>1 / x</c> を計算します。</para>
    /// </summary>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns>The reciprocal <c>1 / x</c>. / 逆数 <c>1 / x</c>。</returns>
    static abstract TSelf Reciprocal(TSelf x);

    /// <summary>
    /// Computes the maximum <c>max(x, y)</c>.
    /// <para>最大値 <c>max(x, y)</c> を計算します。</para>
    /// </summary>
    /// <param name="x">The first operand. / 第1引数。</param>
    /// <param name="y">The second operand. / 第2引数。</param>
    /// <returns>The maximum <c>max(x, y)</c>. / 最大値 <c>max(x, y)</c>。</returns>
    static abstract TSelf Max(TSelf x, TSelf y);

    /// <summary>
    /// Computes the minimum <c>min(x, y)</c>.
    /// <para>最小値 <c>min(x, y)</c> を計算します。</para>
    /// </summary>
    /// <param name="x">The first operand. / 第1引数。</param>
    /// <param name="y">The second operand. / 第2引数。</param>
    /// <returns>The minimum <c>min(x, y)</c>. / 最小値 <c>min(x, y)</c>。</returns>
    static abstract TSelf Min(TSelf x, TSelf y);
}

/// <summary>
/// Represents a factorable objective function evaluable over any <see cref="IFactorableNumber{TSelf}"/> (used by <see cref="BranchAndBound"/> and <see cref="FactorableObjective{TFunction}"/>).
/// <para>任意の <see cref="IFactorableNumber{TSelf}"/> で評価できる分解可能な目的関数を表します（<see cref="BranchAndBound"/> や <see cref="FactorableObjective{TFunction}"/> に渡します）。</para>
/// </summary>
/// <remarks>
/// <para>Evaluations must be deterministic: equal inputs must produce equal results.</para>
/// <para>評価は決定的である必要があります（同じ入力に対して常に同じ結果を返します）。</para>
/// </remarks>
public interface IFactorableFunction
{
    /// <summary>
    /// Evaluates the function at the specified point or box using arithmetic of type <typeparamref name="T"/>.
    /// <para>型 <typeparamref name="T"/> の演算を用いて、指定した点または区間上で関数を評価します。</para>
    /// </summary>
    /// <typeparam name="T">The factorable number type. / 分解可能数値の型。</typeparam>
    /// <param name="x">The input variables. / 入力変数のスパン。</param>
    /// <returns>The evaluated value, interval enclosure, McCormick relaxation, or gradient dual. / 評価結果の値、区間包含、McCormick 緩和、または勾配付きデュアル数。</returns>
    T Evaluate<T>(ReadOnlySpan<T> x) where T : IFactorableNumber<T>;
}
