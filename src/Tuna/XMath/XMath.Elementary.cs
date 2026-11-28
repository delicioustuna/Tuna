using System.Numerics;
using Tuna.LinearAlgebra;

namespace Tuna;

public static partial class XMath
{
    /// <summary>
    /// Computes the natural logarithm <c>ln(x)</c>.
    /// <para>自然対数 <c>ln(x)</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The numeric type. / 数値型。</typeparam>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns>The natural logarithm of <paramref name="x"/>. / <paramref name="x"/> の自然対数。</returns>
    [ElementwiseOperation(OperatorName = "Log", OperatorContainer = "Operators")]
    public static T Log<T>(T x)
        where T : unmanaged, ILogarithmicFunctions<T> =>
        T.Log(x);

    /// <summary>
    /// Computes the square root <c>√x</c>.
    /// <para>平方根 <c>√x</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The numeric type. / 数値型。</typeparam>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns>The square root of <paramref name="x"/>. / <paramref name="x"/> の平方根。</returns>
    [ElementwiseOperation(OperatorName = "Sqrt", OperatorContainer = "Operators")]
    public static T Sqrt<T>(T x)
        where T : unmanaged, IRootFunctions<T> =>
        T.Sqrt(x);

    /// <summary>
    /// Computes <c>√(x² + y²)</c> without undue intermediate overflow or underflow.
    /// <para>途中計算でのオーバーフローやアンダーフローを回避して <c>√(x² + y²)</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The numeric type. / 数値型。</typeparam>
    /// <param name="x">The first input value. / 第1入力値。</param>
    /// <param name="y">The second input value. / 第2入力値。</param>
    /// <returns><c>√(x² + y²)</c>. / <c>√(x² + y²)</c>。</returns>
    /// <remarks>
    /// <para>For IEEE 754 inputs, an infinity yields positive infinity even when the other input is NaN, and two zeros yield positive zero. Vector and matrix overloads accept <paramref name="y"/> as either a scalar or a container/view of the same shape as <paramref name="x"/>.</para>
    /// <para>IEEE 754 の入力では、一方が無限大ならもう一方が NaN でも正の無限大を返し、両方がゼロの場合は正のゼロを返します。ベクトル・行列オーバーロードでは、<paramref name="y"/> にスカラーまたは <paramref name="x"/> と同じ形状のベクトル・行列を指定できます。</para>
    /// </remarks>
    [ElementwiseOperation(OperatorName = "HypotScalar", OperatorContainer = "Operators")]
    [ElementwiseOperation(Over = [nameof(x), nameof(y)], OperatorName = "Hypot", OperatorContainer = "Operators")]
    public static T Hypot<T>(T x, T y)
        where T : unmanaged, IRootFunctions<T> =>
        T.Hypot(x, y);

    /// <summary>
    /// Computes the absolute value <c>|x|</c>.
    /// <para>絶対値 <c>|x|</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The numeric type. / 数値型。</typeparam>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns>The absolute value of <paramref name="x"/>. For <see cref="Complex"/>, returns a complex value whose real part is the magnitude. / <paramref name="x"/> の絶対値（<see cref="Complex"/> では大きさを実部に持つ複素数）。</returns>
    [ElementwiseOperation(OperatorName = "Abs", OperatorContainer = "Operators")]
    public static T Abs<T>(T x)
        where T : unmanaged, INumberBase<T> =>
        T.Abs(x);

    /// <summary>
    /// Computes the square <c>x²</c>.
    /// <para>2乗 <c>x²</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The numeric type. / 数値型。</typeparam>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns><c>x²</c>. / <c>x²</c>。</returns>
    [ElementwiseOperation(OperatorName = "Square", OperatorContainer = "Operators")]
    public static T Square<T>(T x)
        where T : unmanaged, INumberBase<T> =>
        unchecked(x * x);

    /// <summary>
    /// Computes the reciprocal <c>1 / x</c>.
    /// <para>逆数 <c>1 / x</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The numeric type. / 数値型。</typeparam>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns><c>1 / x</c>. / <c>1 / x</c>。</returns>
    [ElementwiseOperation(OperatorName = "Reciprocal", OperatorContainer = "Operators")]
    public static T Reciprocal<T>(T x)
        where T : unmanaged, INumberBase<T> =>
        T.One / x;

    /// <summary>
    /// Computes <paramref name="x"/> raised to the power <paramref name="y"/> (<c>x^y</c>).
    /// <para><paramref name="x"/> の <paramref name="y"/> 乗（<c>x^y</c>）を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The numeric type. / 数値型。</typeparam>
    /// <param name="x">The base value. / 底。</param>
    /// <param name="y">The exponent. / 指数。</param>
    /// <returns><c>x^y</c>. / <c>x^y</c>。</returns>
    /// <remarks>
    /// <para>Vector and matrix overloads accept <paramref name="y"/> as either a scalar or a container/view of the same shape as <paramref name="x"/>.</para>
    /// <para>ベクトル・行列オーバーロードでは、<paramref name="y"/> にスカラーまたは <paramref name="x"/> と同じ形状のベクトル・行列を指定できます。</para>
    /// </remarks>
    [ElementwiseOperation(OperatorName = "PowScalar", OperatorContainer = "Operators")]
    [ElementwiseOperation(Over = [nameof(x), nameof(y)], OperatorName = "Pow", OperatorContainer = "Operators")]
    public static T Pow<T>(T x, T y)
        where T : unmanaged, IPowerFunctions<T> =>
        T.Pow(x, y);

    /// <summary>
    /// Computes the quotient <c>x / y</c>.
    /// <para>商 <c>x / y</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The numeric type. / 数値型。</typeparam>
    /// <param name="x">The dividend. / 被除数。</param>
    /// <param name="y">The divisor. / 除数。</param>
    /// <returns><c>x / y</c>. / <c>x / y</c>。</returns>
    /// <remarks>
    /// <para>Vector, matrix, and tensor overloads accept <paramref name="y"/> as a scalar or a container or view with the same shape as <paramref name="x"/>.</para>
    /// <para>ベクトル・行列・テンソルのオーバーロードでは、<paramref name="y"/> にスカラーまたは <paramref name="x"/> と同じ形状のコンテナやビューを指定できます。</para>
    /// </remarks>
    [ElementwiseOperation(OperatorName = "DivideScalar", OperatorContainer = "Operators")]
    [ElementwiseOperation(Over = [nameof(x), nameof(y)], OperatorName = "Divide", OperatorContainer = "Operators")]
    public static T Divide<T>(T x, T y)
        where T : unmanaged, INumberBase<T> =>
        x / y;

    /// <summary>
    /// Returns the larger of two values (<c>max(x, y)</c>).
    /// <para>2つの値のうち大きい方（<c>max(x, y)</c>）を返します。</para>
    /// </summary>
    /// <typeparam name="T">The numeric type. / 数値型。</typeparam>
    /// <param name="x">The first value. / 第1入力値。</param>
    /// <param name="y">The second value. / 第2入力値。</param>
    /// <returns>The larger value; returns <c>NaN</c> if either input is <c>NaN</c>, and <c>+0</c> for <c>max(-0, +0)</c>. / 大きい方の値（いずれかが <c>NaN</c> なら <c>NaN</c>、<c>max(-0, +0) = +0</c>）。</returns>
    /// <remarks>
    /// <para>To find the maximum element of a vector, use the single-argument <see cref="Max{T}(LinearAlgebra.ReadOnlyVectorSpan{T})"/> overload.</para>
    /// <para>ベクトル全体の最大値を求めるには、1引数の <see cref="Max{T}(LinearAlgebra.ReadOnlyVectorSpan{T})"/> を使用してください。</para>
    /// </remarks>
    [ElementwiseOperation(OperatorName = "MaxScalar", OperatorContainer = "Operators")]
    [ElementwiseOperation(Over = [nameof(x), nameof(y)], OperatorName = "Max", OperatorContainer = "Operators")]
    public static T Max<T>(T x, T y)
        where T : unmanaged, INumber<T> =>
        T.Max(x, y);

    /// <summary>
    /// Returns the smaller of two values (<c>min(x, y)</c>).
    /// <para>2つの値のうち小さい方（<c>min(x, y)</c>）を返します。</para>
    /// </summary>
    /// <typeparam name="T">The numeric type. / 数値型。</typeparam>
    /// <param name="x">The first value. / 第1入力値。</param>
    /// <param name="y">The second value. / 第2入力値。</param>
    /// <returns>The smaller value; returns <c>NaN</c> if either input is <c>NaN</c>, and <c>-0</c> for <c>min(-0, +0)</c>. / 小さい方の値（いずれかが <c>NaN</c> なら <c>NaN</c>、<c>min(-0, +0) = -0</c>）。</returns>
    /// <remarks>
    /// <para>To find the minimum element of a vector, use the single-argument <see cref="Min{T}(LinearAlgebra.ReadOnlyVectorSpan{T})"/> overload.</para>
    /// <para>ベクトル全体の最小値を求めるには、1引数の <see cref="Min{T}(LinearAlgebra.ReadOnlyVectorSpan{T})"/> を使用してください。</para>
    /// </remarks>
    [ElementwiseOperation(OperatorName = "MinScalar", OperatorContainer = "Operators")]
    [ElementwiseOperation(Over = [nameof(x), nameof(y)], OperatorName = "Min", OperatorContainer = "Operators")]
    public static T Min<T>(T x, T y)
        where T : unmanaged, INumber<T> =>
        T.Min(x, y);

    /// <summary>
    /// Clamps <paramref name="x"/> to the inclusive range <c>[min, max]</c>.
    /// <para><paramref name="x"/> を閉区間 <c>[min, max]</c> に収めます。</para>
    /// </summary>
    /// <typeparam name="T">The numeric type. / 数値型。</typeparam>
    /// <param name="x">The input value. / 入力値。</param>
    /// <param name="min">The inclusive lower bound. / 下限。</param>
    /// <param name="max">The inclusive upper bound. / 上限。</param>
    /// <returns>The clamped value, or <c>NaN</c> if <paramref name="x"/> is <c>NaN</c>. / 範囲内に収めた値（<paramref name="x"/> が <c>NaN</c> の場合は <c>NaN</c>）。</returns>
    /// <exception cref="ArgumentException"><paramref name="min"/> is greater than <paramref name="max"/>. / <paramref name="min"/> が <paramref name="max"/> より大きい場合。</exception>
    [ElementwiseOperation(OperatorName = "Clamp", OperatorContainer = "Operators")]
    public static T Clamp<T>(T x, T min, T max)
        where T : unmanaged, INumber<T> =>
        T.Clamp(x, min, max);

    /// <summary>
    /// Computes <c>ln(1 + x)</c> accurately for values of <paramref name="x"/> near zero.
    /// <para>0 付近でも桁落ちしにくい方法で <c>ln(1 + x)</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns><c>ln(1 + x)</c>. / <c>ln(1 + x)</c>。</returns>
    [ElementwiseOperation(OperatorName = "Log1p", OperatorContainer = "Operators")]
    public static T Log1p<T>(T x)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        T.CreateTruncating(XMath.Special.Log1p(double.CreateTruncating(x)));

    /// <summary>
    /// Computes <c>eˣ - 1</c> accurately for values of <paramref name="x"/> near zero.
    /// <para>0 付近でも桁落ちしにくい方法で <c>eˣ - 1</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns><c>eˣ - 1</c>. / <c>eˣ - 1</c>。</returns>
    [ElementwiseOperation(OperatorName = "Expm1", OperatorContainer = "Operators")]
    public static T Expm1<T>(T x)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        T.CreateTruncating(XMath.Special.Expm1(double.CreateTruncating(x)));

    /// <summary>
    /// Computes the hyperbolic tangent <c>tanh(x)</c>.
    /// <para>双曲線正接 <c>tanh(x)</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The numeric type. / 数値型。</typeparam>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns>The hyperbolic tangent of <paramref name="x"/>. / <paramref name="x"/> の双曲線正接。</returns>
    /// <remarks>
    /// <para>For <see langword="float"/> and <see langword="double"/>, zeros retain their sign, NaN yields NaN, infinities map to the corresponding <c>±1</c>, and scalar and elementwise overloads use the same evaluation.</para>
    /// <para><see langword="float"/> と <see langword="double"/> では、ゼロの符号を保持し、NaN には NaN、無限大には符号に応じた <c>±1</c> を返します。スカラー版と要素別オーバーロードは同じ式で評価します。</para>
    /// </remarks>
    [ElementwiseOperation(OperatorName = "Tanh", OperatorContainer = "Operators")]
    public static T Tanh<T>(T x) where T : unmanaged, IHyperbolicFunctions<T> =>
        typeof(T) == typeof(double) || typeof(T) == typeof(float)
            ? T.CreateTruncating(DenseElementwise.Tanh(double.CreateTruncating(x)))
            : T.Tanh(x);

    /// <summary>
    /// Computes the logistic sigmoid function <c>σ(x) = 1 / (1 + e⁻ˣ)</c>.
    /// <para>ロジスティック（シグモイド）関数 <c>σ(x) = 1 / (1 + e⁻ˣ)</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns><c>σ(x)</c>. / <c>σ(x)</c>。</returns>
    [ElementwiseOperation(OperatorName = "Logistic", OperatorContainer = "Operators")]
    public static T Logistic<T>(T x)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        return T.CreateTruncating(LogisticOf(double.CreateTruncating(x)));
    }

    // x < 0 では e⁻ˣ がオーバーフローし得るため、eˣ / (1 + eˣ) で評価する。
    private static double LogisticOf(double v) => v >= 0 ? 1 / (1 + Math.Exp(-v)) : Math.Exp(v) / (1 + Math.Exp(v));

    /// <summary>
    /// Computes the logit function <c>ln(p / (1 - p))</c>, the inverse of <see cref="Logistic{T}(T)"/>.
    /// <para>ロジスティック関数の逆関数であるロジット関数 <c>ln(p / (1 - p))</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="p">The probability value in <c>[0, 1]</c>. / 確率値（<c>[0, 1]</c>）。</param>
    /// <returns><c>ln(p / (1 - p))</c>, or <c>NaN</c> if <paramref name="p"/> is outside <c>[0, 1]</c>. / <c>ln(p / (1 - p))</c>（<c>[0, 1]</c> の外は <c>NaN</c>）。</returns>
    [ElementwiseOperation(OperatorName = "Logit", OperatorContainer = "Operators")]
    public static T Logit<T>(T p)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        double v = double.CreateTruncating(p);
        // p ≈ 1/2 付近では p/(1 - p) ≈ 1 で ln が桁落ちするため、ln(1 + (2p - 1)/(1 - p)) で評価する（[1/4, 3/4] では 2p - 1 に丸め誤差がない）。
        return T.CreateTruncating(v is >= 0.25 and <= 0.75 ? XMath.Special.Log1p((2 * v - 1) / (1 - v)) : Math.Log(v / (1 - v)));
    }

    /// <summary>
    /// Computes the log-logistic function <c>ln σ(x) = -ln(1 + e⁻ˣ)</c> without intermediate overflow.
    /// <para>途中計算でのオーバーフローや桁落ちを回避して、ロジスティック関数の自然対数 <c>ln σ(x) = -ln(1 + e⁻ˣ)</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns><c>ln σ(x)</c>. / <c>ln σ(x)</c>。</returns>
    [ElementwiseOperation(OperatorName = "LogLogistic", OperatorContainer = "Operators")]
    public static T LogLogistic<T>(T x)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        T.CreateTruncating(-SoftplusOf(-double.CreateTruncating(x)));

    /// <summary>
    /// Computes <c>ln(1 - eˣ)</c> for <c>x ≤ 0</c> accurately without cancellation.
    /// <para><c>x ≤ 0</c> に対して、桁落ちを回避して <c>ln(1 - eˣ)</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns><c>ln(1 - eˣ)</c>, or <c>NaN</c> if <c>x &gt; 0</c>. / <c>ln(1 - eˣ)</c>（<c>x &gt; 0</c> の場合は <c>NaN</c>）。</returns>
    [ElementwiseOperation(OperatorName = "Log1mExp", OperatorContainer = "Operators")]
    public static T Log1mExp<T>(T x)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        T.CreateTruncating(XMath.Special.Log1mExp(double.CreateTruncating(x)));

    /// <summary>
    /// Computes <c>ln(1 + x) - x</c> for <c>x ≥ -1</c> accurately near zero.
    /// <para><c>x ≥ -1</c> に対して、0 付近での桁落ちを回避して <c>ln(1 + x) - x</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns><c>ln(1 + x) - x</c>; returns <c>-∞</c> for <c>x = -1</c> or <c>x = +∞</c>, and <c>NaN</c> for <c>x &lt; -1</c>. / <c>ln(1 + x) - x</c>（<c>x = -1</c> または <c>+∞</c> では <c>-∞</c>、<c>x &lt; -1</c> では <c>NaN</c>）。</returns>
    [ElementwiseOperation(OperatorName = "Log1pmx", OperatorContainer = "Operators")]
    public static T Log1pmx<T>(T x)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        double v = double.CreateTruncating(x);
        // x → +∞ の極限は -∞（ln(1 + ∞) - ∞ の NaN を回避）。NaN は級数反復に入る前に返す。
        if (double.IsNaN(v))
        {
            return x;
        }

        return T.CreateTruncating(double.IsPositiveInfinity(v) ? double.NegativeInfinity : XMath.Special.Log1pmx(v));
    }

    /// <summary>
    /// Computes <c>ln(eˣ + eʸ)</c> without intermediate overflow or underflow.
    /// <para>途中計算でのオーバーフローやアンダーフローを回避して <c>ln(eˣ + eʸ)</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The first log-space value. / 第1入力値。</param>
    /// <param name="y">The second log-space value. / 第2入力値。</param>
    /// <returns><c>ln(eˣ + eʸ)</c>. / <c>ln(eˣ + eʸ)</c>。</returns>
    /// <remarks>
    /// <para>Vector and matrix overloads accept <paramref name="y"/> as either a scalar or a container/view of the same shape as <paramref name="x"/>.</para>
    /// <para>ベクトル・行列オーバーロードでは、<paramref name="y"/> にスカラーまたは <paramref name="x"/> と同じ形状のベクトル・行列を指定できます。</para>
    /// </remarks>
    [ElementwiseOperation(OperatorName = "LogAddExpScalar", OperatorContainer = "Operators")]
    [ElementwiseOperation(Over = [nameof(x), nameof(y)], OperatorName = "LogAddExp", OperatorContainer = "Operators")]
    public static T LogAddExp<T>(T x, T y)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        T.CreateTruncating(LogAddExpOf(double.CreateTruncating(x), double.CreateTruncating(y)));

    /// <summary>
    /// Computes <c>ln(eˣ - eʸ)</c> for <c>x ≥ y</c> without intermediate overflow or cancellation.
    /// <para><c>x ≥ y</c> に対して、オーバーフローや桁落ちを回避して <c>ln(eˣ - eʸ)</c> を計算します。</para>
    /// </summary>
    /// <typeparam name="T">The floating-point type. / 浮動小数点型。</typeparam>
    /// <param name="x">The minuend in log space. / 被減数の対数。</param>
    /// <param name="y">The subtrahend in log space. / 減数の対数。</param>
    /// <returns><c>ln(eˣ - eʸ)</c>; returns <c>-∞</c> if <c>x = y</c>, and <c>NaN</c> if <c>x &lt; y</c>. / <c>ln(eˣ - eʸ)</c>（<c>x = y</c> では <c>-∞</c>、<c>x &lt; y</c> では <c>NaN</c>）。</returns>
    /// <remarks>
    /// <para>Vector and matrix overloads accept <paramref name="y"/> as either a scalar or a container/view of the same shape as <paramref name="x"/>.</para>
    /// <para>ベクトル・行列オーバーロードでは、<paramref name="y"/> にスカラーまたは <paramref name="x"/> と同じ形状のベクトル・行列を指定できます。</para>
    /// </remarks>
    [ElementwiseOperation(OperatorName = "LogSubExpScalar", OperatorContainer = "Operators")]
    [ElementwiseOperation(Over = [nameof(x), nameof(y)], OperatorName = "LogSubExp", OperatorContainer = "Operators")]
    public static T LogSubExp<T>(T x, T y)
        where T : unmanaged, IFloatingPointIeee754<T> =>
        T.CreateTruncating(LogSubExpOf(double.CreateTruncating(x), double.CreateTruncating(y)));

    private static double LogAddExpOf(double a, double b)
    {
        if (a < b)
        {
            (a, b) = (b, a);
        }

        // a = +∞ または b = -∞ で b - a が NaN にならないよう先に処理する。
        if (b == double.NegativeInfinity)
        {
            return a;
        }

        if (a == double.PositiveInfinity)
        {
            return double.IsNaN(b) ? b : a;
        }

        return a + XMath.Special.Log1p(Math.Exp(b - a));
    }

    private static double LogSubExpOf(double a, double b)
    {
        // a = +∞ または b = -∞ で b - a が NaN にならないよう先に処理する（a = b は -∞、a < b は NaN）。
        if (b == double.NegativeInfinity)
        {
            return a;
        }

        if (a == double.PositiveInfinity)
        {
            return b == double.PositiveInfinity || double.IsNaN(b) ? double.NaN : a;
        }

        return a + XMath.Special.Log1mExp(b - a);
    }
}
