using System.Globalization;
using static Tuna.Optimization.Global.DirectedRounding;

namespace Tuna.Optimization.Global;

/// <summary>
/// Represents a closed real interval with outwardly rounded arithmetic.
/// <para>外向き丸めによる区間演算をサポートする実数閉区間を表します。</para>
/// </summary>
/// <remarks>
/// <para>Interval operations enclose the true range of the operation over their operands. Points outside an operation's domain are discarded, returning <see cref="Empty"/> if no point is defined. Repeated occurrences of a variable are treated independently; use <see cref="Square"/> instead of <c>x * x</c> for tighter bounds across zero.</para>
/// <para>各区間演算はオペランドの範囲に対する真の像を包含します。定義域外の部分は除外され、定義される点が残らない場合は <see cref="Empty"/> を返します。同一変数が式中に複数回現れても相関は考慮されないため、0 をまたぐ区間の2乗には <c>x * x</c> ではなく <see cref="Square"/> を使用してください。</para>
/// </remarks>
public readonly struct Interval : IFactorableNumber<Interval>, IEquatable<Interval>
{
    /// <summary>
    /// Gets the empty interval (both endpoints <see cref="double.NaN"/>).
    /// <para>空区間（両端が <see cref="double.NaN"/>）を取得します。</para>
    /// </summary>
    public static Interval Empty { get; } = Make(double.NaN, double.NaN);

    /// <summary>
    /// Gets the interval representing all real numbers <c>(−∞, +∞)</c>.
    /// <para>実数全体 <c>(−∞, +∞)</c> を表す区間を取得します。</para>
    /// </summary>
    public static Interval Entire { get; } = Make(double.NegativeInfinity, double.PositiveInfinity);

    /// <summary>
    /// Initializes a new interval <c>[lower, upper]</c>.
    /// <para>区間 <c>[lower, upper]</c> を初期化します。</para>
    /// </summary>
    /// <param name="lower">The lower endpoint (must not be <see cref="double.NaN"/> or <see cref="double.PositiveInfinity"/>). / 下端（<see cref="double.NaN"/> および <see cref="double.PositiveInfinity"/> は不可）。</param>
    /// <param name="upper">The upper endpoint (at least <paramref name="lower"/>; must not be <see cref="double.NaN"/> or <see cref="double.NegativeInfinity"/>). / 上端（<paramref name="lower"/> 以上。<see cref="double.NaN"/> および <see cref="double.NegativeInfinity"/> は不可）。</param>
    /// <exception cref="ArgumentException">An endpoint is <see cref="double.NaN"/>, <paramref name="lower"/> exceeds <paramref name="upper"/>, or the endpoints define no real numbers. / 端点が <see cref="double.NaN"/>、<paramref name="lower"/> &gt; <paramref name="upper"/>、または下端が +∞ もしくは上端が −∞ の場合。</exception>
    public Interval(double lower, double upper)
    {
        if (double.IsNaN(lower) || double.IsNaN(upper) || lower > upper || lower == double.PositiveInfinity || upper == double.NegativeInfinity)
        {
            throw new ArgumentException($"[{lower}, {upper}] is not an interval of real numbers.", nameof(lower));
        }

        Lower = lower;
        Upper = upper;
    }

    /// <summary>
    /// Initializes a degenerate point interval <c>[value, value]</c>.
    /// <para>1点区間 <c>[value, value]</c> を初期化します。</para>
    /// </summary>
    /// <param name="value">A finite real value. / 有限の実数値。</param>
    /// <exception cref="ArgumentException"><paramref name="value"/> is not finite. / <paramref name="value"/> が有限値でない場合。</exception>
    public Interval(double value)
    {
        if (!double.IsFinite(value))
        {
            throw new ArgumentException("The value must be finite.", nameof(value));
        }

        Lower = value;
        Upper = value;
    }

    private Interval(double lower, double upper, byte _)
    {
        Lower = lower;
        Upper = upper;
    }

    /// <summary>
    /// Gets the lower endpoint of the interval, or <see cref="double.NaN"/> if empty.
    /// <para>区間の下端を取得します（空区間の場合は <see cref="double.NaN"/>）。</para>
    /// </summary>
    public double Lower { get; }

    /// <summary>
    /// Gets the upper endpoint of the interval, or <see cref="double.NaN"/> if empty.
    /// <para>区間の上端を取得します（空区間の場合は <see cref="double.NaN"/>）。</para>
    /// </summary>
    public double Upper { get; }

    /// <summary>
    /// Gets a value indicating whether this interval is empty.
    /// <para>区間が空かどうかを取得します。</para>
    /// </summary>
    /// <value><see langword="true"/> if this interval is empty; otherwise, <see langword="false"/>. / 空区間の場合は <see langword="true"/>、それ以外は <see langword="false"/>。</value>
    public bool IsEmpty => double.IsNaN(Lower);

    /// <summary>
    /// Gets the outwardly rounded width <c>Upper − Lower</c>, or <see cref="double.NaN"/> if empty.
    /// <para>上向きに丸めた区間の幅 <c>Upper − Lower</c> を取得します（空区間の場合は <see cref="double.NaN"/>）。</para>
    /// </summary>
    public double Width => IsEmpty ? double.NaN : SubtractUp(Upper, Lower);

    /// <summary>
    /// Gets the midpoint of the interval.
    /// <para>区間の中点を取得します。</para>
    /// </summary>
    /// <value>A finite value inside the interval when both endpoints are finite; 0 for <see cref="Entire"/>; the infinite endpoint when unbounded on one side; or <see cref="double.NaN"/> if empty. / 両端が有限なら区間内の中点、<see cref="Entire"/> なら 0、片側無限ならその無限大、空区間なら <see cref="double.NaN"/>。</value>
    public double Midpoint => double.IsNegativeInfinity(Lower) && double.IsPositiveInfinity(Upper) ? 0 : MidpointOf(Lower, Upper);

    // 0.5*a + 0.5*b avoids overflow, but subnormal rounding can fall outside [min, max]; fall back to 0.5*(a + b) in that case.
    internal static double MidpointOf(double a, double b)
    {
        double middle = 0.5 * a + 0.5 * b;
        return middle >= Math.Min(a, b) && middle <= Math.Max(a, b) ? middle : 0.5 * (a + b);
    }

    /// <summary>
    /// Determines whether the interval contains the specified value.
    /// <para>指定した値がこの区間に含まれるかどうかを判定します。</para>
    /// </summary>
    /// <param name="value">The value to test. / 判定する値。</param>
    /// <returns><see langword="true"/> if <see cref="Lower"/> ≤ <paramref name="value"/> ≤ <see cref="Upper"/>; otherwise, <see langword="false"/>. / <paramref name="value"/> が区間に含まれる場合は <see langword="true"/>、それ以外は <see langword="false"/>。</returns>
    public bool Contains(double value) => Lower <= value && value <= Upper;

    /// <summary>
    /// Computes the intersection of this interval and <paramref name="other"/>.
    /// <para>この区間と <paramref name="other"/> の共通部分を計算します。</para>
    /// </summary>
    /// <param name="other">The other interval. / もう一方の区間。</param>
    /// <returns>The intersection, or <see cref="Empty"/> if disjoint. / 共通部分。交わらない場合は <see cref="Empty"/>。</returns>
    public Interval Intersect(Interval other)
    {
        double lo = Math.Max(Lower, other.Lower), hi = Math.Min(Upper, other.Upper);
        return lo <= hi ? Make(lo, hi) : Empty;
    }

    /// <summary>
    /// Computes the smallest interval enclosing both this interval and <paramref name="other"/>.
    /// <para>この区間と <paramref name="other"/> の両方を含む最小の区間（区間包）を計算します。</para>
    /// </summary>
    /// <param name="other">The other interval. / もう一方の区間。</param>
    /// <returns>The interval hull (returns the non-empty operand if one is empty). / 両方を含む最小の区間（片方が空の場合はもう一方）。</returns>
    public Interval Hull(Interval other) =>
        IsEmpty ? other : other.IsEmpty ? this : Make(Math.Min(Lower, other.Lower), Math.Max(Upper, other.Upper));

    /// <summary>
    /// Implicitly converts a scalar to a degenerate point interval (or <see cref="Empty"/> if non-finite).
    /// <para>スカラー値を1点区間に暗黙的に変換します（有限値でない場合は <see cref="Empty"/>）。</para>
    /// </summary>
    /// <param name="value">The scalar value. / スカラー値。</param>
    public static implicit operator Interval(double value) => double.IsFinite(value) ? Make(value, value) : Empty;

    /// <summary>
    /// Computes the outwardly rounded interval sum <c>x + y</c>.
    /// <para>外向き丸めによる区間和 <c>x + y</c> を計算します。</para>
    /// </summary>
    /// <param name="x">The left operand. / 左辺。</param>
    /// <param name="y">The right operand. / 右辺。</param>
    /// <returns>The interval sum <c>x + y</c>. / 区間和 <c>x + y</c>。</returns>
    public static Interval operator +(Interval x, Interval y) =>
        x.IsEmpty || y.IsEmpty ? Empty : Make(AddDown(x.Lower, y.Lower), AddUp(x.Upper, y.Upper));

    /// <summary>
    /// Computes the outwardly rounded interval difference <c>x − y</c>.
    /// <para>外向き丸めによる区間差 <c>x − y</c> を計算します。</para>
    /// </summary>
    /// <param name="x">The left operand. / 左辺。</param>
    /// <param name="y">The right operand. / 右辺。</param>
    /// <returns>The interval difference <c>x − y</c>. / 区間差 <c>x − y</c>。</returns>
    public static Interval operator -(Interval x, Interval y) =>
        x.IsEmpty || y.IsEmpty ? Empty : Make(SubtractDown(x.Lower, y.Upper), SubtractUp(x.Upper, y.Lower));

    /// <summary>
    /// Negates an interval (<c>−x</c>).
    /// <para>区間の符号を反転します（<c>−x</c>）。</para>
    /// </summary>
    /// <param name="x">The interval to negate. / 符号反転する区間。</param>
    /// <returns>The negated interval <c>[−Upper, −Lower]</c>. / 符号反転した区間 <c>[−Upper, −Lower]</c>。</returns>
    public static Interval operator -(Interval x) => Make(-x.Upper, -x.Lower);

    /// <summary>
    /// Computes the outwardly rounded interval product <c>x · y</c>.
    /// <para>外向き丸めによる区間積 <c>x · y</c> を計算します。</para>
    /// </summary>
    /// <param name="x">The left operand. / 左辺。</param>
    /// <param name="y">The right operand. / 右辺。</param>
    /// <returns>The interval product <c>x · y</c>. / 区間積 <c>x · y</c>。</returns>
    public static Interval operator *(Interval x, Interval y)
    {
        if (x.IsEmpty || y.IsEmpty)
        {
            return Empty;
        }

        double a = x.Lower, b = x.Upper, c = y.Lower, d = y.Upper;
        double lo = Math.Min(Math.Min(MultiplyDown(a, c), MultiplyDown(a, d)), Math.Min(MultiplyDown(b, c), MultiplyDown(b, d)));
        double hi = Math.Max(Math.Max(MultiplyUp(a, c), MultiplyUp(a, d)), Math.Max(MultiplyUp(b, c), MultiplyUp(b, d)));
        return Make(lo, hi);
    }

    /// <summary>
    /// Computes the outwardly rounded interval quotient <c>x / y</c>.
    /// <para>外向き丸めによる区間商 <c>x / y</c> を計算します（<paramref name="y"/> が <c>[0, 0]</c> なら <see cref="Empty"/>、0 を内部に含むなら <see cref="Entire"/>）。</para>
    /// </summary>
    /// <param name="x">The dividend interval. / 被除数区間。</param>
    /// <param name="y">The divisor interval. / 除数区間。</param>
    /// <returns>The interval quotient <c>x / y</c>. / 区間商 <c>x / y</c>。</returns>
    public static Interval operator /(Interval x, Interval y)
    {
        if (x.IsEmpty || y.IsEmpty)
        {
            return Empty;
        }

        double a = x.Lower, b = x.Upper, c = y.Lower, d = y.Upper;
        if ((c > 0 || d < 0) && double.IsFinite(a) && double.IsFinite(b) && double.IsFinite(c) && double.IsFinite(d))
        {
            double lo = Math.Min(Math.Min(DivideDown(a, c), DivideDown(a, d)), Math.Min(DivideDown(b, c), DivideDown(b, d)));
            double hi = Math.Max(Math.Max(DivideUp(a, c), DivideUp(a, d)), Math.Max(DivideUp(b, c), DivideUp(b, d)));
            return Make(lo, hi);
        }

        return x * Reciprocal(y);
    }

    /// <inheritdoc cref="IFactorableNumber{TSelf}.op_Addition(TSelf, double)"/>
    public static Interval operator +(Interval x, double c) => x + (Interval)c;

    /// <inheritdoc cref="IFactorableNumber{TSelf}.op_Addition(double, TSelf)"/>
    public static Interval operator +(double c, Interval x) => (Interval)c + x;

    /// <inheritdoc cref="IFactorableNumber{TSelf}.op_Subtraction(TSelf, double)"/>
    public static Interval operator -(Interval x, double c) => x - (Interval)c;

    /// <inheritdoc cref="IFactorableNumber{TSelf}.op_Subtraction(double, TSelf)"/>
    public static Interval operator -(double c, Interval x) => (Interval)c - x;

    /// <inheritdoc cref="IFactorableNumber{TSelf}.op_Multiply(TSelf, double)"/>
    public static Interval operator *(Interval x, double c) => x * (Interval)c;

    /// <inheritdoc cref="IFactorableNumber{TSelf}.op_Multiply(double, TSelf)"/>
    public static Interval operator *(double c, Interval x) => (Interval)c * x;

    /// <inheritdoc cref="IFactorableNumber{TSelf}.op_Division(TSelf, double)"/>
    public static Interval operator /(Interval x, double c) => x / (Interval)c;

    /// <inheritdoc cref="IFactorableNumber{TSelf}.op_Division(double, TSelf)"/>
    public static Interval operator /(double c, Interval x) => (Interval)c / x;

    /// <summary>
    /// Computes the outwardly rounded interval square <c>x²</c> (with non-negative lower endpoint).
    /// <para>外向き丸めによる区間2乗 <c>x²</c>（下端は 0 以上）を計算します。</para>
    /// </summary>
    /// <param name="x">The input interval. / 入力区間。</param>
    /// <returns>The enclosure of <c>x²</c>. / <c>x²</c> を包含する区間。</returns>
    public static Interval Square(Interval x)
    {
        if (x.IsEmpty)
        {
            return Empty;
        }

        double a = x.Lower, b = x.Upper;
        if (a >= 0)
        {
            return Make(Math.Max(0, MultiplyDown(a, a)), MultiplyUp(b, b));
        }

        if (b <= 0)
        {
            return Make(Math.Max(0, MultiplyDown(b, b)), MultiplyUp(a, a));
        }

        double m = Math.Max(-a, b);
        return Make(0, MultiplyUp(m, m));
    }

    /// <summary>
    /// Computes the outwardly rounded integer power <c>xⁿ</c> (<c>[1, 1]</c> for <c>n == 0</c>, <c>1 / x^|n|</c> for negative <paramref name="n"/>).
    /// <para>外向き丸めによる区間の整数乗 <c>xⁿ</c>（<c>n == 0</c> なら <c>[1, 1]</c>、負の <paramref name="n"/> なら <c>1 / x^|n|</c>）を計算します。</para>
    /// </summary>
    /// <param name="x">The base interval. / 底の区間。</param>
    /// <param name="n">The integer exponent (must not be <see cref="int.MinValue"/>). / 整数指数（<see cref="int.MinValue"/> は不可）。</param>
    /// <returns>The enclosure of <c>xⁿ</c>. / <c>xⁿ</c> を包含する区間。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="n"/> is <see cref="int.MinValue"/>. / <paramref name="n"/> が <see cref="int.MinValue"/> の場合。</exception>
    public static Interval Pow(Interval x, int n)
    {
        if (n == int.MinValue)
        {
            throw new ArgumentOutOfRangeException(nameof(n), "The exponent is too small.");
        }

        if (x.IsEmpty)
        {
            return Empty;
        }

        switch (n)
        {
            case 0:
                return Make(1, 1);
            case 1:
                return x;
            case 2:
                return Square(x);
            case < 0:
                return Reciprocal(Pow(x, -n));
        }

        double a = x.Lower, b = x.Upper;
        if ((n & 1) == 1)
        {
            return Make(PowSignedDown(a, n), -PowSignedDown(-b, n));
        }

        double lo = a >= 0 ? a : b <= 0 ? -b : 0;
        double hi = Math.Max(-a, b);
        return Make(PowDown(lo, n), PowUp(hi, n));
    }

    /// <summary>
    /// Computes the outwardly rounded square root <c>√x</c> over the non-negative part of <paramref name="x"/>.
    /// <para><paramref name="x"/> の非負部分に対する外向き丸めの平方根 <c>√x</c> を計算します。</para>
    /// </summary>
    /// <param name="x">The input interval. / 入力区間。</param>
    /// <returns>The enclosure of <c>√x</c>, or <see cref="Empty"/> if <paramref name="x"/> has no non-negative points. / <c>√x</c> を包含する区間（非負の点がない場合は <see cref="Empty"/>）。</returns>
    public static Interval Sqrt(Interval x) =>
        x.IsEmpty || x.Upper < 0 ? Empty : Make(SqrtDown(Math.Max(0, x.Lower)), SqrtUp(x.Upper));

    /// <summary>
    /// Computes the outwardly rounded exponential <c>eˣ</c>.
    /// <para>外向き丸めによる指数関数 <c>eˣ</c> を計算します。</para>
    /// </summary>
    /// <param name="x">The input interval. / 入力区間。</param>
    /// <returns>The enclosure of <c>eˣ</c>. / <c>eˣ</c> を包含する区間。</returns>
    public static Interval Exp(Interval x) =>
        x.IsEmpty ? Empty : Make(x.Lower == 0 ? 1 : Math.Max(0, Down2(Math.Exp(x.Lower))), x.Upper == 0 ? 1 : Up2(Math.Exp(x.Upper)));

    /// <summary>
    /// Computes the outwardly rounded natural logarithm <c>ln x</c> over the positive part of <paramref name="x"/>.
    /// <para><paramref name="x"/> の正の部分に対する外向き丸めの自然対数 <c>ln x</c> を計算します（下端が 0 以下なら下端は −∞）。</para>
    /// </summary>
    /// <param name="x">The input interval. / 入力区間。</param>
    /// <returns>The enclosure of <c>ln x</c>, or <see cref="Empty"/> if <paramref name="x"/> has no positive points. / <c>ln x</c> を包含する区間（正の点がない場合は <see cref="Empty"/>）。</returns>
    public static Interval Log(Interval x)
    {
        if (x.IsEmpty || x.Upper <= 0)
        {
            return Empty;
        }

        double lo = x.Lower <= 0 ? double.NegativeInfinity : x.Lower == 1 ? 0 : Down2(Math.Log(x.Lower));
        double hi = x.Upper == 1 ? 0 : Up2(Math.Log(x.Upper));
        return Make(lo, hi);
    }

    /// <summary>
    /// Computes the interval absolute value <c>|x|</c>.
    /// <para>区間の絶対値 <c>|x|</c> を計算します。</para>
    /// </summary>
    /// <param name="x">The input interval. / 入力区間。</param>
    /// <returns>The enclosure of <c>|x|</c>. / <c>|x|</c> を包含する区間。</returns>
    public static Interval Abs(Interval x) =>
        x.IsEmpty ? Empty
        : x.Lower >= 0 ? x
        : x.Upper <= 0 ? -x
        : Make(0, Math.Max(-x.Lower, x.Upper));

    /// <summary>
    /// Computes the outwardly rounded reciprocal <c>1 / x</c>.
    /// <para>外向き丸めによる逆数区間 <c>1 / x</c> を計算します。</para>
    /// </summary>
    /// <param name="x">The input interval. / 入力区間。</param>
    /// <returns>The enclosure of <c>1 / x</c>: <see cref="Empty"/> for <c>[0, 0]</c>, <see cref="Entire"/> if 0 is interior, or a half-infinite interval if an endpoint is 0. / <c>1 / x</c> を包含する区間。<c>[0, 0]</c> では <see cref="Empty"/>、0 が内部にあれば <see cref="Entire"/>、端点が 0 の場合は片側無限区間を返します。</returns>
    public static Interval Reciprocal(Interval x)
    {
        if (x.IsEmpty)
        {
            return Empty;
        }

        double a = x.Lower, b = x.Upper;
        if (a > 0)
        {
            return Make(Math.Max(0, DivideDown(1, b)), DivideUp(1, a));
        }

        if (b < 0)
        {
            return Make(DivideDown(1, b), Math.Min(0, DivideUp(1, a)));
        }

        if (a == 0 && b == 0)
        {
            return Empty;
        }

        if (a == 0)
        {
            return Make(Math.Max(0, DivideDown(1, b)), double.PositiveInfinity);
        }

        if (b == 0)
        {
            return Make(double.NegativeInfinity, Math.Min(0, DivideUp(1, a)));
        }

        return Entire;
    }

    /// <summary>
    /// Computes the interval maximum <c>max(x, y)</c>.
    /// <para>区間の最大値 <c>max(x, y)</c> を計算します。</para>
    /// </summary>
    /// <param name="x">The first interval. / 第1区間。</param>
    /// <param name="y">The second interval. / 第2区間。</param>
    /// <returns>The enclosure of <c>max(x, y)</c>. / <c>max(x, y)</c> を包含する区間。</returns>
    public static Interval Max(Interval x, Interval y) =>
        x.IsEmpty || y.IsEmpty ? Empty : Make(Math.Max(x.Lower, y.Lower), Math.Max(x.Upper, y.Upper));

    /// <summary>
    /// Computes the interval minimum <c>min(x, y)</c>.
    /// <para>区間の最小値 <c>min(x, y)</c> を計算します。</para>
    /// </summary>
    /// <param name="x">The first interval. / 第1区間。</param>
    /// <param name="y">The second interval. / 第2区間。</param>
    /// <returns>The enclosure of <c>min(x, y)</c>. / <c>min(x, y)</c> を包含する区間。</returns>
    public static Interval Min(Interval x, Interval y) =>
        x.IsEmpty || y.IsEmpty ? Empty : Make(Math.Min(x.Lower, y.Lower), Math.Min(x.Upper, y.Upper));

    /// <inheritdoc/>
    public bool Equals(Interval other) => Lower.Equals(other.Lower) && Upper.Equals(other.Upper);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is Interval other && Equals(other);

    /// <inheritdoc/>
    public override int GetHashCode() => HashCode.Combine(Lower, Upper);

    /// <summary>
    /// Determines whether two intervals have equal endpoints (two empty intervals are equal).
    /// <para>2つの区間の両端が等しいかどうかを判定します（空区間同士は等しいとみなされます）。</para>
    /// </summary>
    /// <param name="left">The left interval. / 左辺の区間。</param>
    /// <param name="right">The right interval. / 右辺の区間。</param>
    /// <returns><see langword="true"/> if equal; otherwise, <see langword="false"/>. / 等しい場合は <see langword="true"/>、それ以外は <see langword="false"/>。</returns>
    public static bool operator ==(Interval left, Interval right) => left.Equals(right);

    /// <summary>
    /// Determines whether two intervals differ in either endpoint.
    /// <para>2つの区間の少なくとも一方の端が異なるかどうかを判定します。</para>
    /// </summary>
    /// <param name="left">The left interval. / 左辺の区間。</param>
    /// <param name="right">The right interval. / 右辺の区間。</param>
    /// <returns><see langword="true"/> if not equal; otherwise, <see langword="false"/>. / 等しくない場合は <see langword="true"/>、それ以外は <see langword="false"/>。</returns>
    public static bool operator !=(Interval left, Interval right) => !left.Equals(right);

    /// <summary>
    /// Returns a string representation of the interval (<c>[Lower, Upper]</c> or <c>[empty]</c>).
    /// <para>区間の文字列表現（<c>[Lower, Upper]</c>、空区間では <c>[empty]</c>）を返します。</para>
    /// </summary>
    /// <returns>A formatted string representing this interval. / この区間を表す文字列。</returns>
    public override string ToString() =>
        IsEmpty ? "[empty]" : string.Create(CultureInfo.InvariantCulture, $"[{Lower:R}, {Upper:R}]");

    private static Interval Make(double lower, double upper) => new(lower, upper, 0);

    private static double PowDown(double v, int n) => v == 0 || v == 1 ? v : Math.Max(0, Down2(Math.Pow(v, n)));

    private static double PowUp(double v, int n) => v == 0 || v == 1 ? v : Up2(Math.Pow(v, n));

    private static double PowSignedDown(double v, int n) => v >= 0 ? PowDown(v, n) : -PowUp(-v, n);
}
