using System.Globalization;
using static Tuna.Optimization.Global.TangentVectors;

namespace Tuna.Optimization.Global;

/// <summary>
/// Represents an interval enclosure together with McCormick convex underestimator and concave overestimator relaxations (and their subgradients) for factorable functions.
/// <para>分解可能な関数に対する区間包含と、McCormick 凸下界・凹上界緩和（およびそれらの劣勾配・優勾配）を表します。</para>
/// </summary>
/// <remarks>
/// <para>Create independent variables with <see cref="Variable(Interval, double, int, int)"/>. At a reference point <c>p</c>, <see cref="Convex"/> and <see cref="ConvexSubgradient"/> define a valid affine underestimator <c>cv(p) + sᵀ(x − p)</c> over the box. <see cref="Range"/> uses outwardly rounded interval arithmetic, while relaxation values and subgradients use standard floating-point arithmetic (Mitsos, Chachuat &amp; Barton 2009).</para>
/// <para>独立変数は <see cref="Variable(Interval, double, int, int)"/> で生成します。基準点 <c>p</c> における <see cref="Convex"/> と <see cref="ConvexSubgradient"/> は、ボックス上の有効なアフィン下界 <c>cv(p) + sᵀ(x − p)</c> を与えます。<see cref="Range"/> は外向き丸めの区間演算を用い、緩和値と劣勾配は通常の浮動小数点丸めで計算します（Mitsos, Chachuat &amp; Barton 2009）。</para>
/// </remarks>
public readonly struct McCormick : IFactorableNumber<McCormick>
{
    // Relative outward shift on odd-power tangency points so rounding does not invert the relaxation.
    private const double TangentMargin = 1e-10;

    private static readonly McCormick Undefined = new(Interval.Empty, double.NaN, null, double.NaN, null);

    private readonly double[]? convexSubgradient;
    private readonly double[]? concaveSubgradient;

    private McCormick(Interval range, double convex, double[]? convexSubgradient, double concave, double[]? concaveSubgradient)
    {
        Range = range;
        Convex = convex;
        Concave = concave;
        this.convexSubgradient = convexSubgradient;
        this.concaveSubgradient = concaveSubgradient;
    }

    /// <summary>
    /// Gets the outwardly rounded interval enclosure over the box (<see cref="Interval.Empty"/> if undefined everywhere).
    /// <para>ボックス上の外向き丸め区間包含を取得します（ボックス全体で未定義の場合は <see cref="Interval.Empty"/>）。</para>
    /// </summary>
    public Interval Range { get; }

    /// <summary>
    /// Gets the value of the convex underestimator at the reference point (or <see cref="double.NaN"/> if empty).
    /// <para>基準点における凸下界関数の値を取得します（空の場合は <see cref="double.NaN"/>）。</para>
    /// </summary>
    public double Convex { get; }

    /// <summary>
    /// Gets the value of the concave overestimator at the reference point (or <see cref="double.NaN"/> if empty).
    /// <para>基準点における凹上界関数の値を取得します（空の場合は <see cref="double.NaN"/>）。</para>
    /// </summary>
    public double Concave { get; }

    /// <summary>
    /// Gets a subgradient of the convex underestimator at the reference point (an empty span represents the zero vector).
    /// <para>基準点における凸下界関数の劣勾配を取得します（長さ 0 のスパンは零ベクトルを表します）。</para>
    /// </summary>
    public ReadOnlySpan<double> ConvexSubgradient => convexSubgradient;

    /// <summary>
    /// Gets a supergradient of the concave overestimator at the reference point (an empty span represents the zero vector).
    /// <para>基準点における凹上界関数の優勾配を取得します（長さ 0 のスパンは零ベクトルを表します）。</para>
    /// </summary>
    public ReadOnlySpan<double> ConcaveSubgradient => concaveSubgradient;

    /// <summary>
    /// Creates an independent variable at the specified coordinate index.
    /// <para>指定した座標インデックスの独立変数を作成します。</para>
    /// </summary>
    /// <param name="range">The non-empty interval bound for this coordinate. / この座標の空でない区間範囲。</param>
    /// <param name="point">The finite reference coordinate value inside <paramref name="range"/>. / <paramref name="range"/> に含まれる有限の基準点座標。</param>
    /// <param name="index">The zero-based coordinate index. / 0 から始まる座標インデックス。</param>
    /// <param name="dimension">The total number of variables (at least 1). / 変数の総数（1 以上）。</param>
    /// <returns>A <see cref="McCormick"/> relaxation with <paramref name="range"/>, value <paramref name="point"/>, and unit basis vector <c>e_index</c>. / 範囲 <paramref name="range"/>、基準点値 <paramref name="point"/>、単位ベクトル <c>e_index</c> を持つ <see cref="McCormick"/> 緩和。</returns>
    /// <exception cref="ArgumentException"><paramref name="range"/> is empty, or <paramref name="point"/> is non-finite or outside <paramref name="range"/>. / <paramref name="range"/> が空、または <paramref name="point"/> が有限でないか範囲外の場合。</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="dimension"/> is less than 1, or <paramref name="index"/> is outside <c>[0, dimension)</c>. / <paramref name="dimension"/> が 1 未満、または <paramref name="index"/> が範囲外の場合。</exception>
    public static McCormick Variable(Interval range, double point, int index, int dimension)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(dimension, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, dimension);
        var unit = new double[dimension];
        unit[index] = 1;
        return Variable(range, point, unit);
    }

    internal static McCormick Variable(Interval range, double point, double[] unit)
    {
        if (range.IsEmpty)
        {
            throw new ArgumentException("The range must not be empty.", nameof(range));
        }

        if (!double.IsFinite(point) || !range.Contains(point))
        {
            throw new ArgumentException("The point must be finite and lie in the range.", nameof(point));
        }

        return new(range, point, unit, point, unit);
    }

    /// <summary>
    /// Implicitly converts a scalar constant to a <see cref="McCormick"/> relaxation (or empty if non-finite).
    /// <para>スカラー定数を <see cref="McCormick"/> 緩和に暗黙的に変換します（有限値でない場合は空）。</para>
    /// </summary>
    /// <param name="value">The constant value. / 定数値。</param>
    public static implicit operator McCormick(double value) => Create(value, value, null, value, null);

    /// <summary>
    /// Computes the McCormick relaxation of <c>x + y</c>.
    /// <para><c>x + y</c> の McCormick 緩和を計算します。</para>
    /// </summary>
    /// <param name="x">The left operand. / 左辺。</param>
    /// <param name="y">The right operand. / 右辺。</param>
    /// <returns>The relaxation of <c>x + y</c>. / <c>x + y</c> の緩和。</returns>
    public static McCormick operator +(McCormick x, McCormick y) =>
        Create(x.Range + y.Range,
            x.Convex + y.Convex, Combine(x.convexSubgradient, 1, y.convexSubgradient, 1),
            x.Concave + y.Concave, Combine(x.concaveSubgradient, 1, y.concaveSubgradient, 1));

    /// <summary>
    /// Computes the McCormick relaxation of <c>x − y</c>.
    /// <para><c>x − y</c> の McCormick 緩和を計算します。</para>
    /// </summary>
    /// <param name="x">The left operand. / 左辺。</param>
    /// <param name="y">The right operand. / 右辺。</param>
    /// <returns>The relaxation of <c>x − y</c>. / <c>x − y</c> の緩和。</returns>
    public static McCormick operator -(McCormick x, McCormick y) =>
        Create(x.Range - y.Range,
            x.Convex - y.Concave, Combine(x.convexSubgradient, 1, y.concaveSubgradient, -1),
            x.Concave - y.Convex, Combine(x.concaveSubgradient, 1, y.convexSubgradient, -1));

    /// <summary>
    /// Computes the McCormick relaxation of <c>−x</c>.
    /// <para><c>−x</c> の McCormick 緩和を計算します。</para>
    /// </summary>
    /// <param name="x">The operand to negate. / 符号反転する緩和。</param>
    /// <returns>The relaxation of <c>−x</c>. / <c>−x</c> の緩和。</returns>
    public static McCormick operator -(McCormick x) =>
        Create(-x.Range, -x.Concave, Scale(x.concaveSubgradient, -1), -x.Convex, Scale(x.convexSubgradient, -1));

    /// <summary>
    /// Computes the McCormick bilinear envelope relaxation of <c>x · y</c>.
    /// <para><c>x · y</c> の McCormick 双線形包絡緩和を計算します。</para>
    /// </summary>
    /// <param name="x">The left operand. / 左辺。</param>
    /// <param name="y">The right operand. / 右辺。</param>
    /// <returns>The relaxation of <c>x · y</c>. / <c>x · y</c> の緩和。</returns>
    public static McCormick operator *(McCormick x, McCormick y)
    {
        Interval range = x.Range * y.Range;
        if (range.IsEmpty)
        {
            return Undefined;
        }

        if (x.IsConstant)
        {
            return MultiplyBy(y, x.Range.Lower, range);
        }

        if (y.IsConstant)
        {
            return MultiplyBy(x, y.Range.Lower, range);
        }

        double xL = x.Range.Lower, xU = x.Range.Upper, yL = y.Range.Lower, yU = y.Range.Upper;
        if (!(double.IsFinite(xL) && double.IsFinite(xU) && double.IsFinite(yL) && double.IsFinite(yU)))
        {
            return Create(range, range.Lower, null, range.Upper, null);
        }

        // McCormick bilinear underestimators from (x − xL)(y − yL) ≥ 0 and (xU − x)(yU − y) ≥ 0.
        x.LowerTerm(yL, out double xa, out double[]? sxa);
        y.LowerTerm(xL, out double ya, out double[]? sya);
        double alpha1 = yL * xa + xL * ya - xL * yL;
        x.LowerTerm(yU, out double xb, out double[]? sxb);
        y.LowerTerm(xU, out double yb, out double[]? syb);
        double alpha2 = yU * xb + xU * yb - xU * yU;
        double convex;
        double[]? convexSub;
        if (alpha1 >= alpha2)
        {
            convex = alpha1;
            convexSub = Combine(sxa, yL, sya, xL);
        }
        else
        {
            convex = alpha2;
            convexSub = Combine(sxb, yU, syb, xU);
        }

        // McCormick bilinear overestimators from (x − xL)(yU − y) ≥ 0 and (xU − x)(y − yL) ≥ 0.
        x.UpperTerm(yU, out double xc, out double[]? sxc);
        y.UpperTerm(xL, out double yc, out double[]? syc);
        double beta1 = yU * xc + xL * yc - xL * yU;
        x.UpperTerm(yL, out double xd, out double[]? sxd);
        y.UpperTerm(xU, out double yd, out double[]? syd);
        double beta2 = yL * xd + xU * yd - xU * yL;
        double concave;
        double[]? concaveSub;
        if (beta1 <= beta2)
        {
            concave = beta1;
            concaveSub = Combine(sxc, yU, syc, xL);
        }
        else
        {
            concave = beta2;
            concaveSub = Combine(sxd, yL, syd, xU);
        }

        return Create(range, convex, convexSub, concave, concaveSub);
    }

    /// <summary>
    /// Computes the McCormick relaxation of <c>x / y</c> (tightened with the interval quotient <c>x.Range / y.Range</c>).
    /// <para><c>x / y</c> の McCormick 緩和を計算します（区間商 <c>x.Range / y.Range</c> で切り詰めます）。</para>
    /// </summary>
    /// <param name="x">The dividend. / 被除数。</param>
    /// <param name="y">The divisor. / 除数。</param>
    /// <returns>The relaxation of <c>x / y</c>. / <c>x / y</c> の緩和。</returns>
    public static McCormick operator /(McCormick x, McCormick y)
    {
        Interval range = x.Range / y.Range;
        if (range.IsEmpty)
        {
            return Undefined;
        }

        if (y.IsConstant)
        {
            return DivideBy(x, y.Range.Lower, range);
        }

        McCormick product = x * Reciprocal(y);
        return Retighten(product, range.Intersect(product.Range));
    }

    /// <inheritdoc cref="IFactorableNumber{TSelf}.op_Addition(TSelf, double)"/>
    public static McCormick operator +(McCormick x, double c) =>
        Create(x.Range + c, x.Convex + c, x.convexSubgradient, x.Concave + c, x.concaveSubgradient);

    /// <inheritdoc cref="IFactorableNumber{TSelf}.op_Addition(double, TSelf)"/>
    public static McCormick operator +(double c, McCormick x) => x + c;

    /// <inheritdoc cref="IFactorableNumber{TSelf}.op_Subtraction(TSelf, double)"/>
    public static McCormick operator -(McCormick x, double c) =>
        Create(x.Range - c, x.Convex - c, x.convexSubgradient, x.Concave - c, x.concaveSubgradient);

    /// <inheritdoc cref="IFactorableNumber{TSelf}.op_Subtraction(double, TSelf)"/>
    public static McCormick operator -(double c, McCormick x) =>
        Create(c - x.Range, c - x.Concave, Scale(x.concaveSubgradient, -1), c - x.Convex, Scale(x.convexSubgradient, -1));

    /// <inheritdoc cref="IFactorableNumber{TSelf}.op_Multiply(TSelf, double)"/>
    public static McCormick operator *(McCormick x, double c) => MultiplyBy(x, c, x.Range * c);

    /// <inheritdoc cref="IFactorableNumber{TSelf}.op_Multiply(double, TSelf)"/>
    public static McCormick operator *(double c, McCormick x) => MultiplyBy(x, c, x.Range * c);

    /// <inheritdoc cref="IFactorableNumber{TSelf}.op_Division(TSelf, double)"/>
    public static McCormick operator /(McCormick x, double c) => DivideBy(x, c, x.Range / c);

    /// <inheritdoc cref="IFactorableNumber{TSelf}.op_Division(double, TSelf)"/>
    public static McCormick operator /(double c, McCormick x)
    {
        Interval range = c / x.Range;
        McCormick scaled = MultiplyBy(Reciprocal(x), c, range);
        return scaled;
    }

    /// <summary>
    /// Computes the McCormick relaxation of <c>x²</c>.
    /// <para><c>x²</c> の McCormick 緩和を計算します。</para>
    /// </summary>
    /// <param name="x">The input relaxation. / 入力の緩和。</param>
    /// <returns>The relaxation of <c>x²</c>. / <c>x²</c> の緩和。</returns>
    public static McCormick Square(McCormick x)
    {
        Interval range = Interval.Square(x.Range);
        if (range.IsEmpty)
        {
            return Undefined;
        }

        double L = x.Range.Lower, U = x.Range.Upper;
        x.Mid(Math.Clamp(0, L, U), out double zu, out double[]? su);
        double convex = zu * zu;
        double[]? convexSub = Scale(su, 2 * zu);
        double concave = range.Upper;
        double[]? concaveSub = null;
        if (double.IsFinite(L) && double.IsFinite(U))
        {
            x.Mid(L + U >= 0 ? U : L, out double zo, out double[]? so);
            concave = (L + U) * zo - L * U;
            concaveSub = Scale(so, L + U);
        }

        return Create(range, convex, convexSub, concave, concaveSub);
    }

    /// <summary>
    /// Computes the McCormick relaxation of the integer power <c>xⁿ</c> (<c>1 / x^|n|</c> for negative <paramref name="n"/>).
    /// <para>整数乗 <c>xⁿ</c>（負の <paramref name="n"/> では <c>1 / x^|n|</c>）の McCormick 緩和を計算します。</para>
    /// </summary>
    /// <param name="x">The base relaxation. / 底の緩和。</param>
    /// <param name="n">The integer exponent (must not be <see cref="int.MinValue"/>). / 整数指数（<see cref="int.MinValue"/> は不可）。</param>
    /// <returns>The relaxation of <c>xⁿ</c>. / <c>xⁿ</c> の緩和。</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="n"/> is <see cref="int.MinValue"/>. / <paramref name="n"/> が <see cref="int.MinValue"/> の場合。</exception>
    public static McCormick Pow(McCormick x, int n)
    {
        if (n == int.MinValue)
        {
            throw new ArgumentOutOfRangeException(nameof(n), "The exponent is too small.");
        }

        if (x.Range.IsEmpty)
        {
            return Undefined;
        }

        switch (n)
        {
            case 0:
                return 1.0;
            case 1:
                return x;
            case 2:
                return Square(x);
            case < 0:
                return Reciprocal(Pow(x, -n));
        }

        Interval range = Interval.Pow(x.Range, n);
        double L = x.Range.Lower, U = x.Range.Upper;
        double fL = Math.Pow(L, n), fU = Math.Pow(U, n);
        double convex, concave;
        double[]? convexSub, concaveSub;
        if ((n & 1) == 0 || L >= 0)
        {
            x.Mid((n & 1) == 0 ? Math.Clamp(0, L, U) : L, out double zu, out double[]? su);
            convex = Math.Pow(zu, n);
            convexSub = Scale(su, n * Math.Pow(zu, n - 1));
            x.Mid(fU >= fL ? U : L, out double zo, out double[]? so);
            SecantOrBound(L, fL, U, fU, zo, so, range.Upper, out concave, out concaveSub);
        }
        else if (U <= 0)
        {
            x.Mid(U, out double zo, out double[]? so);
            concave = Math.Pow(zo, n);
            concaveSub = Scale(so, n * Math.Pow(zo, n - 1));
            x.Mid(L, out double zu, out double[]? su);
            SecantOrBound(L, fL, U, fU, zu, su, range.Lower, out convex, out convexSub);
        }
        else if (double.IsFinite(L) && double.IsFinite(U))
        {
            // Odd power straddling 0: convex envelope joins tangent from (L, Lⁿ) at t = −L·r with zⁿ,
            // where r solves (n − 1)rⁿ + n·rⁿ⁻¹ − 1 = 0; shifting t outward by TangentMargin preserves validity.
            double r = OddPowerRatio(n);
            x.Mid(L, out double zu, out double[]? su);
            if (U < -L * r * (1 - TangentMargin))
            {
                SecantOrBound(L, fL, U, fU, zu, su, range.Lower, out convex, out convexSub);
            }
            else
            {
                double t = -L * r * (1 + TangentMargin);
                double z = zu >= t ? zu : t;
                double slope = n * Math.Pow(z, n - 1);
                convex = zu >= t ? Math.Pow(zu, n) : Math.Pow(t, n) + slope * (zu - t);
                convexSub = Scale(su, slope);
            }

            // Symmetric concave envelope using tangent from (U, Uⁿ) at −U·r.
            x.Mid(U, out double zo, out double[]? so);
            if (L > -U * r * (1 - TangentMargin))
            {
                SecantOrBound(L, fL, U, fU, zo, so, range.Upper, out concave, out concaveSub);
            }
            else
            {
                double t = -U * r * (1 + TangentMargin);
                double z = zo <= t ? zo : t;
                double slope = n * Math.Pow(z, n - 1);
                concave = zo <= t ? Math.Pow(zo, n) : Math.Pow(t, n) + slope * (zo - t);
                concaveSub = Scale(so, slope);
            }
        }
        else
        {
            return Create(range, range.Lower, null, range.Upper, null);
        }

        return Create(range, convex, convexSub, concave, concaveSub);
    }

    /// <summary>
    /// Computes the McCormick relaxation of <c>√x</c> over the non-negative part of <paramref name="x"/>.
    /// <para><paramref name="x"/> の非負部分に対する <c>√x</c> の McCormick 緩和を計算します。</para>
    /// </summary>
    /// <param name="x">The input relaxation. / 入力の緩和。</param>
    /// <returns>The relaxation of <c>√x</c>. / <c>√x</c> の緩和。</returns>
    public static McCormick Sqrt(McCormick x)
    {
        Interval range = Interval.Sqrt(x.Range);
        if (range.IsEmpty)
        {
            return Undefined;
        }

        double L = Math.Max(0, x.Range.Lower), U = x.Range.Upper;
        x.Mid(U, out double zo, out double[]? so);
        double concave = range.Upper;
        double[]? concaveSub = null;
        if (zo > 0 && double.IsFinite(zo))
        {
            concave = Math.Sqrt(zo);
            concaveSub = Scale(so, 0.5 / concave);
        }

        x.Mid(L, out double zu, out double[]? su);
        SecantOrBound(L, Math.Sqrt(L), U, Math.Sqrt(U), zu, su, range.Lower, out double convex, out double[]? convexSub);
        return Create(range, convex, convexSub, concave, concaveSub);
    }

    /// <summary>
    /// Computes the McCormick relaxation of <c>eˣ</c>.
    /// <para><c>eˣ</c> の McCormick 緩和を計算します。</para>
    /// </summary>
    /// <param name="x">The input relaxation. / 入力の緩和。</param>
    /// <returns>The relaxation of <c>eˣ</c>. / <c>eˣ</c> の緩和。</returns>
    public static McCormick Exp(McCormick x)
    {
        Interval range = Interval.Exp(x.Range);
        if (range.IsEmpty)
        {
            return Undefined;
        }

        double L = x.Range.Lower, U = x.Range.Upper;
        x.Mid(L, out double zu, out double[]? su);
        double convex = Math.Exp(zu);
        double[]? convexSub = Scale(su, convex);
        x.Mid(U, out double zo, out double[]? so);
        SecantOrBound(L, Math.Exp(L), U, Math.Exp(U), zo, so, range.Upper, out double concave, out double[]? concaveSub);
        return Create(range, convex, convexSub, concave, concaveSub);
    }

    /// <summary>
    /// Computes the McCormick relaxation of <c>ln x</c> over the positive part of <paramref name="x"/> (convex bound is <see cref="double.NegativeInfinity"/> when lower endpoint ≤ 0).
    /// <para><paramref name="x"/> の正の部分に対する <c>ln x</c> の McCormick 緩和を計算します（下端が 0 以下の場合、凸下界は −∞）。</para>
    /// </summary>
    /// <param name="x">The input relaxation. / 入力の緩和。</param>
    /// <returns>The relaxation of <c>ln x</c>. / <c>ln x</c> の緩和。</returns>
    public static McCormick Log(McCormick x)
    {
        Interval range = Interval.Log(x.Range);
        if (range.IsEmpty)
        {
            return Undefined;
        }

        double L = x.Range.Lower, U = x.Range.Upper;
        x.Mid(U, out double zo, out double[]? so);
        double concave = range.Upper;
        double[]? concaveSub = null;
        if (zo > 0 && double.IsFinite(zo))
        {
            concave = Math.Log(zo);
            concaveSub = Scale(so, 1 / zo);
        }

        double convex = range.Lower;
        double[]? convexSub = null;
        if (L > 0)
        {
            x.Mid(L, out double zu, out double[]? su);
            SecantOrBound(L, Math.Log(L), U, Math.Log(U), zu, su, range.Lower, out convex, out convexSub);
        }

        return Create(range, convex, convexSub, concave, concaveSub);
    }

    /// <summary>
    /// Computes the McCormick relaxation of <c>|x|</c>.
    /// <para><c>|x|</c> の McCormick 緩和を計算します。</para>
    /// </summary>
    /// <param name="x">The input relaxation. / 入力の緩和。</param>
    /// <returns>The relaxation of <c>|x|</c>. / <c>|x|</c> の緩和。</returns>
    public static McCormick Abs(McCormick x)
    {
        Interval range = Interval.Abs(x.Range);
        if (range.IsEmpty)
        {
            return Undefined;
        }

        double L = x.Range.Lower, U = x.Range.Upper;
        x.Mid(Math.Clamp(0, L, U), out double zu, out double[]? su);
        double convex = Math.Abs(zu);
        double[]? convexSub = Scale(su, Sign(zu));
        double fL = Math.Abs(L), fU = Math.Abs(U);
        x.Mid(fU >= fL ? U : L, out double zo, out double[]? so);
        SecantOrBound(L, fL, U, fU, zo, so, range.Upper, out double concave, out double[]? concaveSub);
        return Create(range, convex, convexSub, concave, concaveSub);
    }

    /// <summary>
    /// Computes the McCormick relaxation of <c>1 / x</c> (falls back to constant interval bounds when 0 lies strictly inside <paramref name="x"/>).
    /// <para><c>1 / x</c> の McCormick 緩和を計算します（0 を内部に含む場合は区間端の定数緩和にフォールバックします）。</para>
    /// </summary>
    /// <param name="x">The input relaxation. / 入力の緩和。</param>
    /// <returns>The relaxation of <c>1 / x</c>. / <c>1 / x</c> の緩和。</returns>
    public static McCormick Reciprocal(McCormick x)
    {
        Interval range = Interval.Reciprocal(x.Range);
        if (range.IsEmpty)
        {
            return Undefined;
        }

        double L = x.Range.Lower, U = x.Range.Upper;
        if (U <= 0)
        {
            McCormick negated = Reciprocal(-x);
            return Retighten(-negated, range);
        }

        if (L < 0)
        {
            return Create(range, range.Lower, null, range.Upper, null);
        }

        x.Mid(U, out double zu, out double[]? su);
        double convex = range.Lower;
        double[]? convexSub = null;
        if (zu > 0 && double.IsFinite(zu))
        {
            convex = 1 / zu;
            convexSub = Scale(su, -convex * convex);
        }

        double concave = range.Upper;
        double[]? concaveSub = null;
        if (L > 0)
        {
            x.Mid(L, out double zo, out double[]? so);
            SecantOrBound(L, 1 / L, U, 1 / U, zo, so, range.Upper, out concave, out concaveSub);
        }

        return Create(range, convex, convexSub, concave, concaveSub);
    }

    /// <summary>
    /// Computes the McCormick relaxation of <c>max(x, y) = (x + y + |x − y|) / 2</c>, tightened with the interval maximum.
    /// <para><c>max(x, y) = (x + y + |x − y|) / 2</c> の McCormick 緩和を計算し、区間の最大値で切り詰めます。</para>
    /// </summary>
    /// <param name="x">The first operand. / 第1引数。</param>
    /// <param name="y">The second operand. / 第2引数。</param>
    /// <returns>The relaxation of <c>max(x, y)</c>. / <c>max(x, y)</c> の緩和。</returns>
    public static McCormick Max(McCormick x, McCormick y)
    {
        McCormick m = 0.5 * (x + y + Abs(x - y));
        return Retighten(m, Interval.Max(x.Range, y.Range).Intersect(m.Range));
    }

    /// <summary>
    /// Computes the McCormick relaxation of <c>min(x, y) = (x + y − |x − y|) / 2</c>, tightened with the interval minimum.
    /// <para><c>min(x, y) = (x + y − |x − y|) / 2</c> の McCormick 緩和を計算し、区間の最小値で切り詰めます。</para>
    /// </summary>
    /// <param name="x">The first operand. / 第1引数。</param>
    /// <param name="y">The second operand. / 第2引数。</param>
    /// <returns>The relaxation of <c>min(x, y)</c>. / <c>min(x, y)</c> の緩和。</returns>
    public static McCormick Min(McCormick x, McCormick y)
    {
        McCormick m = 0.5 * (x + y - Abs(x - y));
        return Retighten(m, Interval.Min(x.Range, y.Range).Intersect(m.Range));
    }

    /// <inheritdoc/>
    public override string ToString() =>
        string.Create(CultureInfo.InvariantCulture, $"{{ Range = {Range}, Convex = {Convex:R}, Concave = {Concave:R} }}");

    // Solves (n − 1)rⁿ + n·rⁿ⁻¹ − 1 = 0 in (0, 1) for odd n ≥ 3; monotone Newton convergence from r = 1 by convexity.
    internal static double OddPowerRatio(int n)
    {
        double r = 1;
        for (int k = 0; k < 200; k++)
        {
            double rn1 = Math.Pow(r, n - 2);
            double h = ((n - 1) * r + n) * rn1 * r - 1;
            double dh = n * (n - 1) * rn1 * (r + 1);
            double next = r - h / dh;
            if (!(next < r))
            {
                break;
            }

            r = next;
        }

        return r;
    }

    private bool IsConstant => convexSubgradient is null && concaveSubgradient is null && Range.Lower == Range.Upper;

    // Clamps relaxations to range: falling outside replaces with constant endpoint (zero subgradient);
    // slight inward rounding overshoot clamps value only while keeping the valid subgradient.
    private static McCormick Create(Interval range, double convex, double[]? convexSub, double concave, double[]? concaveSub)
    {
        if (range.IsEmpty)
        {
            return Undefined;
        }

        if (!(convex >= range.Lower))
        {
            convex = range.Lower;
            convexSub = null;
        }
        else if (convex > range.Upper)
        {
            convex = range.Upper;
        }

        if (!(concave <= range.Upper))
        {
            concave = range.Upper;
            concaveSub = null;
        }
        else if (concave < range.Lower)
        {
            concave = range.Lower;
        }

        return new(range, convex, convexSub, concave, concaveSub);
    }

    private static McCormick Retighten(McCormick m, Interval range) =>
        Create(range, m.Convex, m.convexSubgradient, m.Concave, m.concaveSubgradient);

    private static McCormick MultiplyBy(McCormick x, double c, Interval range)
    {
        if (range.IsEmpty || !double.IsFinite(c))
        {
            return Undefined;
        }

        return c >= 0
            ? Create(range, c * x.Convex, Scale(x.convexSubgradient, c), c * x.Concave, Scale(x.concaveSubgradient, c))
            : Create(range, c * x.Concave, Scale(x.concaveSubgradient, c), c * x.Convex, Scale(x.convexSubgradient, c));
    }

    private static McCormick DivideBy(McCormick x, double c, Interval range)
    {
        if (range.IsEmpty || c == 0 || !double.IsFinite(c))
        {
            return Undefined;
        }

        double inverse = 1 / c;
        return c > 0
            ? Create(range, x.Convex / c, Scale(x.convexSubgradient, inverse), x.Concave / c, Scale(x.concaveSubgradient, inverse))
            : Create(range, x.Concave / c, Scale(x.concaveSubgradient, inverse), x.Convex / c, Scale(x.convexSubgradient, inverse));
    }

    private void LowerTerm(double coefficient, out double value, out double[]? subgradient)
    {
        if (coefficient >= 0)
        {
            value = Convex;
            subgradient = convexSubgradient;
        }
        else
        {
            value = Concave;
            subgradient = concaveSubgradient;
        }
    }

    private void UpperTerm(double coefficient, out double value, out double[]? subgradient)
    {
        if (coefficient >= 0)
        {
            value = Concave;
            subgradient = concaveSubgradient;
        }
        else
        {
            value = Convex;
            subgradient = convexSubgradient;
        }
    }

    // Mid-point selection mid(Convex, Concave, target) per McCormick composition rule; selecting target yields zero subgradient.
    private void Mid(double target, out double z, out double[]? subgradient)
    {
        if (target < Convex)
        {
            z = Convex;
            subgradient = convexSubgradient;
        }
        else if (target > Concave)
        {
            z = Concave;
            subgradient = concaveSubgradient;
        }
        else
        {
            z = target;
            subgradient = null;
        }
    }

    private static void SecantOrBound(double l, double fl, double u, double fu, double z, double[]? inner, double bound,
        out double value, out double[]? subgradient)
    {
        if (double.IsFinite(l) && double.IsFinite(u) && double.IsFinite(fl) && double.IsFinite(fu))
        {
            double slope = u > l ? (fu - fl) / (u - l) : 0;
            double v = fl + slope * (z - l);
            if (double.IsFinite(slope) && double.IsFinite(v))
            {
                value = v;
                subgradient = Scale(inner, slope);
                return;
            }
        }

        value = bound;
        subgradient = null;
    }
}
