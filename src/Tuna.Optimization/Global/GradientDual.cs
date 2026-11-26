using System.Globalization;
using Tuna.LinearAlgebra;
using static Tuna.Optimization.Global.TangentVectors;

namespace Tuna.Optimization.Global;

/// <summary>
/// Represents a forward-mode dual number carrying a scalar value and its gradient for factorable functions.
/// <para>分解可能な関数の値と勾配を前進型自動微分で計算するデュアル数型を表します。</para>
/// </summary>
/// <remarks>
/// <para>Create independent variables with <see cref="Variable(double, int, int)"/>. At non-differentiable points, <see cref="Abs"/> uses derivative 0 at 0, and <see cref="Max"/> / <see cref="Min"/> take the first operand's gradient on ties.</para>
/// <para>独立変数は <see cref="Variable(double, int, int)"/> で生成します。微分不可能点では、<see cref="Abs"/> は 0 での勾配を 0 とし、<see cref="Max"/> および <see cref="Min"/> は同値時に第1引数の勾配を採用します。</para>
/// </remarks>
public readonly struct GradientDual : IFactorableNumber<GradientDual>
{
    private readonly double[]? gradient;

    private GradientDual(double value, double[]? gradient)
    {
        Value = value;
        this.gradient = gradient;
    }

    /// <summary>
    /// Gets the primal scalar value.
    /// <para>主値（スカラー値）を取得します。</para>
    /// </summary>
    public double Value { get; }

    /// <summary>
    /// Gets the gradient vector (an empty span represents the zero vector).
    /// <para>勾配ベクトルを取得します（長さ 0 のスパンは零ベクトルを表します）。</para>
    /// </summary>
    public ReadOnlySpan<double> Gradient => gradient;

    /// <summary>
    /// Creates an independent variable at the specified coordinate index.
    /// <para>指定した座標インデックスの独立変数を作成します。</para>
    /// </summary>
    /// <param name="value">The variable value. / 変数の値。</param>
    /// <param name="index">The zero-based coordinate index. / 0 から始まる座標インデックス。</param>
    /// <param name="dimension">The total number of variables (at least 1). / 変数の総数（1 以上）。</param>
    /// <returns>A <see cref="GradientDual"/> with value <paramref name="value"/> and unit basis vector <c>e_index</c> as gradient. / 値 <paramref name="value"/> と単位ベクトル <c>e_index</c> を勾配に持つ <see cref="GradientDual"/>。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="dimension"/> is less than 1, or <paramref name="index"/> is outside <c>[0, dimension)</c>. / <paramref name="dimension"/> が 1 未満、または <paramref name="index"/> が範囲外の場合。</exception>
    public static GradientDual Variable(double value, int index, int dimension)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(dimension, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, dimension);
        var unit = new double[dimension];
        unit[index] = 1;
        return new(value, unit);
    }

    internal static GradientDual Variable(double value, double[] unit) => new(value, unit);

    /// <summary>
    /// Implicitly converts a scalar constant to a <see cref="GradientDual"/> with zero gradient.
    /// <para>スカラー定数を零勾配の <see cref="GradientDual"/> に暗黙的に変換します。</para>
    /// </summary>
    /// <param name="value">The constant value. / 定数値。</param>
    public static implicit operator GradientDual(double value) => new(value, null);

    /// <summary>
    /// Computes the sum <c>x + y</c> and propagates its gradient.
    /// <para>和 <c>x + y</c> とその勾配を計算します。</para>
    /// </summary>
    /// <param name="x">The left operand. / 左辺。</param>
    /// <param name="y">The right operand. / 右辺。</param>
    /// <returns>The sum <c>x + y</c>. / 和 <c>x + y</c>。</returns>
    public static GradientDual operator +(GradientDual x, GradientDual y) => new(x.Value + y.Value, Combine(x.gradient, 1, y.gradient, 1));

    /// <summary>
    /// Computes the difference <c>x − y</c> and propagates its gradient.
    /// <para>差 <c>x − y</c> とその勾配を計算します。</para>
    /// </summary>
    /// <param name="x">The left operand. / 左辺。</param>
    /// <param name="y">The right operand. / 右辺。</param>
    /// <returns>The difference <c>x − y</c>. / 差 <c>x − y</c>。</returns>
    public static GradientDual operator -(GradientDual x, GradientDual y) => new(x.Value - y.Value, Combine(x.gradient, 1, y.gradient, -1));

    /// <summary>
    /// Negates <paramref name="x"/> and its gradient (<c>−x</c>).
    /// <para><paramref name="x"/> とその勾配の符号を反転します（<c>−x</c>）。</para>
    /// </summary>
    /// <param name="x">The operand to negate. / 符号反転する値。</param>
    /// <returns>The negated value <c>−x</c>. / 符号反転した値 <c>−x</c>。</returns>
    public static GradientDual operator -(GradientDual x) => new(-x.Value, Scale(x.gradient, -1));

    /// <summary>
    /// Computes the product <c>x · y</c> and propagates its gradient.
    /// <para>積 <c>x · y</c> とその勾配を計算します。</para>
    /// </summary>
    /// <param name="x">The left operand. / 左辺。</param>
    /// <param name="y">The right operand. / 右辺。</param>
    /// <returns>The product <c>x · y</c>. / 積 <c>x · y</c>。</returns>
    public static GradientDual operator *(GradientDual x, GradientDual y) =>
        new(x.Value * y.Value, Combine(x.gradient, y.Value, y.gradient, x.Value));

    /// <summary>
    /// Computes the quotient <c>x / y</c> and propagates its gradient.
    /// <para>商 <c>x / y</c> とその勾配を計算します。</para>
    /// </summary>
    /// <param name="x">The dividend. / 被除数。</param>
    /// <param name="y">The divisor. / 除数。</param>
    /// <returns>The quotient <c>x / y</c>. / 商 <c>x / y</c>。</returns>
    public static GradientDual operator /(GradientDual x, GradientDual y)
    {
        double q = x.Value / y.Value;
        return new(q, Combine(x.gradient, 1 / y.Value, y.gradient, -q / y.Value));
    }

    /// <inheritdoc cref="IFactorableNumber{TSelf}.op_Addition(TSelf, double)"/>
    public static GradientDual operator +(GradientDual x, double c) => new(x.Value + c, x.gradient);

    /// <inheritdoc cref="IFactorableNumber{TSelf}.op_Addition(double, TSelf)"/>
    public static GradientDual operator +(double c, GradientDual x) => new(c + x.Value, x.gradient);

    /// <inheritdoc cref="IFactorableNumber{TSelf}.op_Subtraction(TSelf, double)"/>
    public static GradientDual operator -(GradientDual x, double c) => new(x.Value - c, x.gradient);

    /// <inheritdoc cref="IFactorableNumber{TSelf}.op_Subtraction(double, TSelf)"/>
    public static GradientDual operator -(double c, GradientDual x) => new(c - x.Value, Scale(x.gradient, -1));

    /// <inheritdoc cref="IFactorableNumber{TSelf}.op_Multiply(TSelf, double)"/>
    public static GradientDual operator *(GradientDual x, double c) => new(x.Value * c, Scale(x.gradient, c));

    /// <inheritdoc cref="IFactorableNumber{TSelf}.op_Multiply(double, TSelf)"/>
    public static GradientDual operator *(double c, GradientDual x) => new(c * x.Value, Scale(x.gradient, c));

    /// <inheritdoc cref="IFactorableNumber{TSelf}.op_Division(TSelf, double)"/>
    public static GradientDual operator /(GradientDual x, double c) => new(x.Value / c, Scale(x.gradient, 1 / c));

    /// <inheritdoc cref="IFactorableNumber{TSelf}.op_Division(double, TSelf)"/>
    public static GradientDual operator /(double c, GradientDual x)
    {
        double q = c / x.Value;
        return new(q, Scale(x.gradient, -q / x.Value));
    }

    /// <summary>
    /// Computes <c>x²</c> and propagates its gradient.
    /// <para><c>x²</c> とその勾配を計算します。</para>
    /// </summary>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns>The square <c>x²</c>. / 2乗 <c>x²</c>。</returns>
    public static GradientDual Square(GradientDual x) => new(x.Value * x.Value, Scale(x.gradient, 2 * x.Value));

    /// <summary>
    /// Computes the integer power <c>xⁿ</c> and propagates its gradient (constant 1 when <c>n == 0</c>).
    /// <para>整数乗 <c>xⁿ</c> とその勾配を計算します（<c>n == 0</c> の場合は定数 1）。</para>
    /// </summary>
    /// <param name="x">The base value. / 底。</param>
    /// <param name="n">The integer exponent. / 整数指数。</param>
    /// <returns>The power <c>xⁿ</c>. / べき乗 <c>xⁿ</c>。</returns>
    public static GradientDual Pow(GradientDual x, int n) =>
        n == 0 ? new(1, null) : new(Math.Pow(x.Value, n), Scale(x.gradient, n * Math.Pow(x.Value, n - 1.0)));

    /// <summary>
    /// Computes <c>√x</c> and propagates its gradient (local derivative is <see cref="double.PositiveInfinity"/> at 0).
    /// <para><c>√x</c> とその勾配を計算します（0 における局所微分は <see cref="double.PositiveInfinity"/>）。</para>
    /// </summary>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns>The square root <c>√x</c> with propagated gradient. / 平方根 <c>√x</c> と伝播された勾配。</returns>
    public static GradientDual Sqrt(GradientDual x)
    {
        double s = Math.Sqrt(x.Value);
        return new(s, Scale(x.gradient, 0.5 / s));
    }

    /// <summary>
    /// Computes <c>eˣ</c> and propagates its gradient.
    /// <para><c>eˣ</c> とその勾配を計算します。</para>
    /// </summary>
    /// <param name="x">The exponent. / 指数。</param>
    /// <returns>The exponential <c>eˣ</c>. / 指数関数値 <c>eˣ</c>。</returns>
    public static GradientDual Exp(GradientDual x)
    {
        double e = Math.Exp(x.Value);
        return new(e, Scale(x.gradient, e));
    }

    /// <summary>
    /// Computes <c>ln x</c> and propagates its gradient.
    /// <para><c>ln x</c> とその勾配を計算します。</para>
    /// </summary>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns>The natural logarithm <c>ln x</c>. / 自然対数 <c>ln x</c>。</returns>
    public static GradientDual Log(GradientDual x) => new(Math.Log(x.Value), Scale(x.gradient, 1 / x.Value));

    /// <summary>
    /// Computes <c>|x|</c> and propagates its gradient (derivative is 0 at 0).
    /// <para><c>|x|</c> とその勾配を計算します（0 での勾配は 0）。</para>
    /// </summary>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns>The absolute value <c>|x|</c>. / 絶対値 <c>|x|</c>。</returns>
    public static GradientDual Abs(GradientDual x) => new(Math.Abs(x.Value), Scale(x.gradient, Sign(x.Value)));

    /// <summary>
    /// Computes <c>1 / x</c> and propagates its gradient.
    /// <para><c>1 / x</c> とその勾配を計算します。</para>
    /// </summary>
    /// <param name="x">The input value. / 入力値。</param>
    /// <returns>The reciprocal <c>1 / x</c>. / 逆数 <c>1 / x</c>。</returns>
    public static GradientDual Reciprocal(GradientDual x)
    {
        double r = 1 / x.Value;
        return new(r, Scale(x.gradient, -r * r));
    }

    /// <summary>
    /// Computes <c>max(x, y)</c> and propagates the gradient of the larger operand (selecting <paramref name="x"/> on ties).
    /// <para><c>max(x, y)</c> を計算し、大きい側の勾配を伝播します（同値時は <paramref name="x"/> を選択）。</para>
    /// </summary>
    /// <param name="x">The first operand. / 第1引数。</param>
    /// <param name="y">The second operand. / 第2引数。</param>
    /// <returns>The maximum <c>max(x, y)</c>. / 最大値 <c>max(x, y)</c>。</returns>
    public static GradientDual Max(GradientDual x, GradientDual y) =>
        double.IsNaN(x.Value) || double.IsNaN(y.Value) ? double.NaN : x.Value >= y.Value ? x : y;

    /// <summary>
    /// Computes <c>min(x, y)</c> and propagates the gradient of the smaller operand (selecting <paramref name="x"/> on ties).
    /// <para><c>min(x, y)</c> を計算し、小さい側の勾配を伝播します（同値時は <paramref name="x"/> を選択）。</para>
    /// </summary>
    /// <param name="x">The first operand. / 第1引数。</param>
    /// <param name="y">The second operand. / 第2引数。</param>
    /// <returns>The minimum <c>min(x, y)</c>. / 最小値 <c>min(x, y)</c>。</returns>
    public static GradientDual Min(GradientDual x, GradientDual y) =>
        double.IsNaN(x.Value) || double.IsNaN(y.Value) ? double.NaN : x.Value <= y.Value ? x : y;

    /// <inheritdoc/>
    public override string ToString() => Value.ToString("R", CultureInfo.InvariantCulture);
}

/// <summary>
/// Wraps an <see cref="IFactorableFunction"/> as an <see cref="IDifferentiableObjectiveFunction"/> using <see cref="GradientDual"/> for exact automatic differentiation.
/// <para><see cref="IFactorableFunction"/> を <see cref="GradientDual"/> による自動微分付き <see cref="IDifferentiableObjectiveFunction"/> としてラップします（<see cref="Gradient.Lbfgs"/> などに渡せます）。</para>
/// </summary>
/// <typeparam name="TFunction">The factorable function type (struct implementations allow JIT specialization). / 分解可能関数の型（構造体として実装すると JIT が特殊化できます）。</typeparam>
/// <remarks>
/// <para>Value-only evaluations use zero-gradient constants to avoid gradient propagation overhead. Instances reuse internal workspace arrays and are not thread-safe.</para>
/// <para>値のみの評価では勾配を持たない定数として計算し、勾配伝播のコストを省きます。内部の作業用配列を再利用するためスレッドセーフではありません。</para>
/// </remarks>
public sealed class FactorableObjective<TFunction> : IDifferentiableObjectiveFunction
    where TFunction : IFactorableFunction
{
    private readonly TFunction function;
    private readonly double[][] units;
    private readonly GradientDual[] work;

    /// <summary>
    /// Initializes a new differentiable objective wrapper for <paramref name="function"/> in <paramref name="dimension"/> variables.
    /// <para><paramref name="dimension"/> 変数の分解可能関数 <paramref name="function"/> に対する微分可能目的関数ラッパーを初期化します。</para>
    /// </summary>
    /// <param name="function">The factorable function to wrap. / ラップする分解可能関数。</param>
    /// <param name="dimension">The number of variables (at least 1). / 変数の数（1 以上）。</param>
    /// <exception cref="ArgumentNullException"><paramref name="function"/> is <see langword="null"/>. / <paramref name="function"/> が <see langword="null"/> の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="dimension"/> is less than 1. / <paramref name="dimension"/> が 1 未満の場合。</exception>
    public FactorableObjective(TFunction function, int dimension)
    {
        ArgumentNullException.ThrowIfNull(function);
        ArgumentOutOfRangeException.ThrowIfLessThan(dimension, 1);
        this.function = function;
        units = Units(dimension);
        work = new GradientDual[dimension];
    }

    /// <summary>
    /// Gets the number of variables.
    /// <para>変数の数を取得します。</para>
    /// </summary>
    public int Dimension => work.Length;

    /// <summary>
    /// Evaluates the wrapped function at the specified point without computing gradients.
    /// <para>勾配を計算せずに、指定した点でラップした関数を評価します。</para>
    /// </summary>
    /// <param name="x">The point to evaluate (length <see cref="Dimension"/>). / 評価点（長さ <see cref="Dimension"/>）。</param>
    /// <returns>The objective value at <paramref name="x"/>. / <paramref name="x"/> での目的関数値。</returns>
    /// <exception cref="ArgumentException">The length of <paramref name="x"/> does not match <see cref="Dimension"/>. / <paramref name="x"/> の長さが <see cref="Dimension"/> と一致しない場合。</exception>
    public double Evaluate(ReadOnlyVectorSpan<double> x)
    {
        CheckLength(x.Length, nameof(x));
        for (int i = 0; i < work.Length; i++)
        {
            work[i] = x[i];
        }

        return function.Evaluate<GradientDual>(work).Value;
    }

    /// <summary>
    /// Evaluates the wrapped function at the specified point and writes its exact gradient to <paramref name="gradient"/>.
    /// <para>指定した点でラップした関数を評価し、正確な勾配を <paramref name="gradient"/> に書き込みます。</para>
    /// </summary>
    /// <param name="x">The point to evaluate (length <see cref="Dimension"/>). / 評価点（長さ <see cref="Dimension"/>）。</param>
    /// <param name="gradient">The destination span for the gradient (length <see cref="Dimension"/>). / 勾配の書き込み先（長さ <see cref="Dimension"/>）。</param>
    /// <returns>The objective value at <paramref name="x"/>. / <paramref name="x"/> での目的関数値。</returns>
    /// <exception cref="ArgumentException">The length of <paramref name="x"/> or <paramref name="gradient"/> does not match <see cref="Dimension"/>. / <paramref name="x"/> または <paramref name="gradient"/> の長さが <see cref="Dimension"/> と一致しない場合。</exception>
    public double Evaluate(ReadOnlyVectorSpan<double> x, VectorSpan<double> gradient)
    {
        CheckLength(x.Length, nameof(x));
        CheckLength(gradient.Length, nameof(gradient));
        for (int i = 0; i < work.Length; i++)
        {
            work[i] = GradientDual.Variable(x[i], units[i]);
        }

        GradientDual result = function.Evaluate<GradientDual>(work);
        ReadOnlySpan<double> g = result.Gradient;
        for (int i = 0; i < work.Length; i++)
        {
            gradient[i] = g.IsEmpty ? 0 : g[i];
        }

        return result.Value;
    }

    private void CheckLength(int length, string name)
    {
        if (length != work.Length)
        {
            throw new ArgumentException($"The length must be {work.Length}.", name);
        }
    }
}
